"""Named locks and caps (§8).

* Three lock names: ``main-checkout``, ``main-merge`` and the keyed family
  ``integration:<branch>``, always taken in ``LOCK_ORDER``
  (``integration:*`` → ``main-merge`` → ``main-checkout``; a gate slot is a leaf).
* A lock is held by a **process group** (the tool process), never by an LLM
  session across turns — except ``integration:*``, which a session may also
  hold across commands (§8.4); the holder's own tools then **co-hold** it.
* Liveness is probed **outside** the transaction; inside ``BEGIN IMMEDIATE``
  the lock is taken only if its row is still what was probed (compare-and-set).
* Caps: ``gates`` slots (``QS_CP_MAX_GATES``) and the node cap
  (``QS_CP_MAX_NODES``), counted across all runs.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from . import clock as clock_mod
from . import db, errors, liveness, nodes, tokens

MAIN_CHECKOUT = "main-checkout"
MAIN_MERGE = "main-merge"
INTEGRATION = "integration:"
GATES = "gates"
LOCK_WAIT_S = 600.0
GATE_WAIT_S = 1800.0
POLL_S = 1.0
DEFAULT_MAX_GATES = 2
DEFAULT_MAX_NODES = 4
_TERMINAL_TASKS = ("dropped", "merged", "validated")


def order_key(name: str) -> tuple[int, str]:
    """``LOCK_ORDER``: ``integration:*`` (lexical), then ``main-merge``, then ``main-checkout``."""
    if name.startswith(INTEGRATION) and len(name) > len(INTEGRATION):
        return (0, name)
    if name == MAIN_MERGE:
        return (1, name)
    if name == MAIN_CHECKOUT:
        return (2, name)
    raise errors.CpError("USAGE", f"unknown lock name {name!r}")


LOCK_ORDER = ("integration:*", MAIN_MERGE, MAIN_CHECKOUT)


def assert_sorted(names: Sequence[str]) -> None:
    keys = [order_key(n) for n in names]
    if any(a >= b for a, b in zip(keys, keys[1:])):
        raise errors.CpError("INTERNAL", f"locks not in LOCK_ORDER: {list(names)}")


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def max_gates() -> int:
    return _env_int("QS_CP_MAX_GATES", DEFAULT_MAX_GATES)


def max_nodes() -> int:
    return _env_int("QS_CP_MAX_NODES", DEFAULT_MAX_NODES)


# --------------------------------------------------------------------------- liveness of a lock row

_IDENT = (
    "holder_kind",
    "holder_pid",
    "holder_pid_start",
    "holder_pgid",
    "holder_session_id",
    "holder_epoch",
    "cohold_pid",
    "cohold_pgid",
    "token_subject",
    "acquired_at",
)


def _ident(row: sqlite3.Row | None) -> tuple[Any, ...] | None:
    return None if row is None else tuple(row[k] for k in _IDENT)


@dataclass(frozen=True)
class Facts:
    """Liveness facts about one lock row, gathered outside the transaction."""

    holder_alive: bool = False  # process-held rows
    listing: list[liveness.Agent] | None = None  # session-held rows
    pid_alive: bool = False  # session-held rows, used only when the listing failed
    cohold_alive: bool = False


def probe_row(
    row: sqlite3.Row | None,
    probe: liveness.ProcessProbe,
    claude: liveness.ClaudeCli,
    listing: list[liveness.Agent] | None = None,
) -> Facts:
    if row is None:
        return Facts()
    if row["holder_kind"] == "process":
        return Facts(holder_alive=probe.holder_alive(row["holder_pid"], row["holder_pid_start"], row["holder_pgid"]))
    listing = listing if listing is not None else claude.try_agents()
    cohold_alive = row["cohold_pid"] is not None and probe.holder_alive(
        row["cohold_pid"], row["cohold_pid_start"], row["cohold_pgid"]
    )
    pid_alive = listing is None and probe.alive(row["holder_pid"], row["holder_pid_start"])
    return Facts(listing=listing, pid_alive=pid_alive, cohold_alive=cohold_alive)


def session_holder_dead(conn: sqlite3.Connection, row: sqlite3.Row, facts: Facts) -> bool:
    """§8.4: token superseded, session gone, or owner finished (DB reads, inside the transaction)."""
    kind, subject = str(row["token_subject"]).split(":", 1)
    if kind == "node":
        node = conn.execute("SELECT state, run_id FROM nodes WHERE id = ?", (subject,)).fetchone()
        if node is None or node["state"] in nodes.TERMINAL:
            return True
        run_id = node["run_id"]
    else:
        lease = conn.execute("SELECT epoch, session_id FROM run_leases WHERE run_id = ?", (subject,)).fetchone()
        if lease is None or (lease["epoch"] > row["holder_epoch"] and lease["session_id"] != row["holder_session_id"]):
            return True
        run_id = subject
    run = conn.execute("SELECT state FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None or run["state"] != "open":
        return True
    branch = str(row["name"])[len(INTEGRATION) :]
    owner = conn.execute(
        "SELECT state FROM tasks WHERE branch = ? AND is_deliverable = 1 ORDER BY id DESC LIMIT 1", (branch,)
    ).fetchone()
    if owner is not None and owner["state"] in _TERMINAL_TASKS:
        return True
    if facts.listing is not None:
        return liveness.find(facts.listing, session_id=row["holder_session_id"]) is None
    return not facts.pid_alive


def row_free(conn: sqlite3.Connection, row: sqlite3.Row | None, facts: Facts) -> bool:
    if row is None:
        return True
    if row["holder_kind"] == "process":
        return not facts.holder_alive
    return session_holder_dead(conn, row, facts) and not facts.cohold_alive


def _busy(row: sqlite3.Row, name: str) -> errors.CpError:
    return errors.CpError(
        "BUSY",
        f"lock {name} is held",
        holder_actor=row["holder_actor"],
        purpose=row["purpose"],
        holder_kind=row["holder_kind"],
        holder_session_id=row["holder_session_id"],
    )


def _check_session_order(conn: sqlite3.Connection, who: tokens.Principal, name: str) -> None:
    """Keep ``LOCK_ORDER`` across commands: nothing that sorts before a session-held lock of this subject."""
    rows = conn.execute(
        "SELECT name FROM locks WHERE holder_kind = 'session' AND token_subject = ? AND name != ?", (who.subject, name)
    ).fetchall()
    for r in rows:
        if order_key(name) < order_key(r["name"]):
            raise errors.CpError("CONFLICT", f"release {r['name']} first (LOCK_ORDER)")


# --------------------------------------------------------------------------- process-held locks and gate slots


@dataclass
class Held:
    """What one ``hold()`` call owns."""

    holder: liveness.Holder
    token: str
    actor: str
    allow_stopped: bool = False
    locks: list[str] = field(default_factory=list)
    coheld: list[str] = field(default_factory=list)
    slots: list[tuple[str, int]] = field(default_factory=list)


def _deadline(clock: clock_mod.Clock, timeout: float) -> float:
    return clock.now().timestamp() + timeout


def _take_lock(
    conn: sqlite3.Connection,
    held: Held,
    name: str,
    *,
    purpose: str,
    clock: clock_mod.Clock,
    probe: liveness.ProcessProbe,
    claude: liveness.ClaudeCli,
    timeout: float,
    poll: float,
) -> None:
    deadline = _deadline(clock, timeout)
    me = held.holder
    while True:
        row = conn.execute("SELECT * FROM locks WHERE name = ?", (name,)).fetchone()
        facts = probe_row(row, probe, claude)
        restamp: tuple[int | None, str | None] | None = None
        if row is not None and row["holder_kind"] == "session" and facts.listing is not None:
            agent = liveness.find(facts.listing, session_id=row["holder_session_id"])
            if agent is not None and agent.pid is not None:
                restamp = (agent.pid, probe.start_of(agent.pid))
        blocker: sqlite3.Row | None = None
        with db.write(conn):
            who = tokens.require(conn, held.token, kinds={"run", "node"}, allow_stopped=held.allow_stopped)
            _check_session_order(conn, who, name)
            current = conn.execute("SELECT * FROM locks WHERE name = ?", (name,)).fetchone()
            if _ident(current) != _ident(row):
                continue  # changed since probed: probe again
            now = db.now(clock)
            if row_free(conn, current, facts):
                conn.execute("DELETE FROM locks WHERE name = ?", (name,))
                conn.execute(
                    "INSERT INTO locks (name, holder_kind, holder_pid, holder_pid_start, holder_pgid, holder_epoch,"
                    " holder_actor, token_subject, purpose, acquired_at) VALUES (?, 'process', ?, ?, ?, ?, ?, ?, ?, ?)",
                    (name, me.pid, me.pid_start, me.pgid, who.token.epoch, held.actor, who.subject, purpose, now),
                )
                held.locks.append(name)
                return
            assert current is not None
            if (
                current["holder_kind"] == "session"
                and current["token_subject"] == who.subject
                and current["holder_session_id"] == who.session_id
                and (current["cohold_pid"] is None or not facts.cohold_alive)
            ):
                pid, pid_start = (
                    restamp if restamp is not None else (current["holder_pid"], current["holder_pid_start"])
                )
                conn.execute(
                    "UPDATE locks SET cohold_pid = ?, cohold_pid_start = ?, cohold_pgid = ?, holder_epoch = ?,"
                    " holder_pid = ?, holder_pid_start = ? WHERE name = ?",
                    (me.pid, me.pid_start, me.pgid, who.token.epoch, pid, pid_start, name),
                )
                held.coheld.append(name)
                return
            blocker = current
        if clock.now().timestamp() >= deadline:
            assert blocker is not None
            raise _busy(blocker, name)
        clock.sleep(poll)


def _take_slot(
    conn: sqlite3.Connection,
    held: Held,
    cap: str,
    *,
    clock: clock_mod.Clock,
    probe: liveness.ProcessProbe,
    timeout: float,
    poll: float,
    limit: int,
) -> None:
    deadline = _deadline(clock, timeout)
    me = held.holder
    while True:
        rows = conn.execute("SELECT * FROM cap_slots WHERE cap = ?", (cap,)).fetchall()
        dead = {
            r["slot"]: r["holder_pid"]
            for r in rows
            if not probe.holder_alive(r["holder_pid"], r["holder_pid_start"], r["holder_pgid"])
        }
        with db.write(conn):
            tokens.require(conn, held.token, kinds={"run", "node"}, allow_stopped=held.allow_stopped)
            current = {r["slot"]: r for r in conn.execute("SELECT * FROM cap_slots WHERE cap = ?", (cap,))}
            for slot in range(limit):
                r = current.get(slot)
                if r is not None and dead.get(slot) != r["holder_pid"]:
                    continue
                conn.execute("DELETE FROM cap_slots WHERE cap = ? AND slot = ?", (cap, slot))
                conn.execute(
                    "INSERT INTO cap_slots (cap, slot, holder_pid, holder_pid_start, holder_pgid, holder_actor,"
                    " acquired_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (cap, slot, me.pid, me.pid_start, me.pgid, held.actor, db.now(clock)),
                )
                held.slots.append((cap, slot))
                return
        if clock.now().timestamp() >= deadline:
            raise errors.CpError("BUSY", f"all {limit} `{cap}` slots are taken")
        clock.sleep(poll)


def release(conn: sqlite3.Connection, held: Held) -> None:
    """Free exactly what this call still owns (single compare-and-set statements)."""
    me = held.holder
    for name in held.locks:
        conn.execute(
            "DELETE FROM locks WHERE name = ? AND holder_kind = 'process' AND holder_pid = ? AND holder_pgid IS ?",
            (name, me.pid, me.pgid),
        )
    for name in held.coheld:
        conn.execute(
            "UPDATE locks SET cohold_pid = NULL, cohold_pid_start = NULL, cohold_pgid = NULL"
            " WHERE name = ? AND cohold_pid = ? AND cohold_pgid IS ?",
            (name, me.pid, me.pgid),
        )
    for cap, slot in held.slots:
        conn.execute("DELETE FROM cap_slots WHERE cap = ? AND slot = ? AND holder_pid = ?", (cap, slot, me.pid))
    held.locks.clear()
    held.coheld.clear()
    held.slots.clear()


@contextmanager
def hold(
    names: Sequence[str],
    *,
    conn_factory: Callable[[], sqlite3.Connection],
    token: str,
    actor: str,
    purpose: str,
    clock: clock_mod.Clock,
    probe: liveness.ProcessProbe,
    claude: liveness.ClaudeCli,
    timeout: float = LOCK_WAIT_S,
    cap: str | None = None,
    cap_timeout: float = GATE_WAIT_S,
    holder: liveness.Holder | None = None,
    allow_stopped: bool = False,
    poll: float = POLL_S,
) -> Iterator[Held]:
    """Take ``names`` (sorted by ``LOCK_ORDER``, asserted), then a ``cap`` slot; free them on exit."""
    assert_sorted(names)
    held = Held(holder or probe.me(), token, actor, allow_stopped)
    conn = conn_factory()
    try:
        for name in names:
            _take_lock(
                conn, held, name, purpose=purpose, clock=clock, probe=probe, claude=claude, timeout=timeout, poll=poll
            )
        if cap is not None:
            _take_slot(conn, held, cap, clock=clock, probe=probe, timeout=cap_timeout, poll=poll, limit=max_gates())
        yield held
    finally:
        try:
            release(conn, held)
        finally:
            conn.close()


def recheck(conn: sqlite3.Connection, held: Held) -> None:
    """Before each step: the token is still valid and every lock / slot is still this call's."""
    me = held.holder
    with db.write(conn):
        tokens.require(conn, held.token, kinds={"run", "node"}, allow_stopped=held.allow_stopped)
        for name in held.locks:
            row = conn.execute(
                "SELECT holder_kind, holder_pid, holder_pgid FROM locks WHERE name = ?", (name,)
            ).fetchone()
            if row is None or (row["holder_kind"], row["holder_pid"], row["holder_pgid"]) != (
                "process",
                me.pid,
                me.pgid,
            ):
                raise errors.CpError("STALE_TOKEN", f"lock {name} is no longer held by this call")
        for name in held.coheld:
            row = conn.execute("SELECT cohold_pid, cohold_pgid FROM locks WHERE name = ?", (name,)).fetchone()
            if row is None or (row["cohold_pid"], row["cohold_pgid"]) != (me.pid, me.pgid):
                raise errors.CpError("STALE_TOKEN", f"lock {name} is no longer co-held by this call")
        for cap, slot in held.slots:
            row = conn.execute("SELECT holder_pid FROM cap_slots WHERE cap = ? AND slot = ?", (cap, slot)).fetchone()
            if row is None or row["holder_pid"] != me.pid:
                raise errors.CpError("STALE_TOKEN", f"`{cap}` slot {slot} is no longer held by this call")


# --------------------------------------------------------------------------- session-held integration locks (§8.4)


def _check_branch(conn: sqlite3.Connection, who: tokens.Principal, branch: str) -> None:
    if who.kind == "node":
        assert who.task_id is not None
        if nodes.deliverable_of(conn, who.task_id)["branch"] != branch:
            raise errors.CpError("CONFLICT", f"{branch} is not the branch of this node's deliverable")
        return
    found = conn.execute(
        "SELECT 1 FROM tasks WHERE run_id = ? AND is_deliverable = 1 AND branch = ?", (who.run_id, branch)
    ).fetchone()
    if found is None:
        raise errors.CpError("CONFLICT", f"{branch} is not the branch of a deliverable of run {who.run_id}")


def acquire_session(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    probe: liveness.ProcessProbe,
    claude: liveness.ClaudeCli,
    *,
    token: str,
    name: str,
    purpose: str,
    session_id: str,
    timeout: float = 0.0,
    poll: float = POLL_S,
) -> dict[str, Any]:
    if not name.startswith(INTEGRATION) or len(name) == len(INTEGRATION):
        raise errors.CpError("POLICY_REFUSED", "only `integration:<branch>` locks can be held by a session")
    deadline = _deadline(clock, timeout)
    while True:
        listing = claude.try_agents()
        if listing is None:
            raise errors.CpError("BUSY", "liveness unknown (the session listing failed), retry")
        mine = liveness.find(listing, session_id=session_id)
        pid = mine.pid if mine is not None else None
        pid_start = probe.start_of(pid) if pid is not None else None
        if pid is None or pid_start is None:
            raise errors.CpError("USAGE", f"session {session_id} is not running")
        row = conn.execute("SELECT * FROM locks WHERE name = ?", (name,)).fetchone()
        facts = probe_row(row, probe, claude, listing)
        blocker: sqlite3.Row | None = None
        with db.write(conn):
            who = tokens.require(conn, token, kinds={"run", "node"})
            if who.session_id != session_id:
                raise errors.CpError("CONFLICT", "the caller is not the token's own session")
            _check_branch(conn, who, name[len(INTEGRATION) :])
            _check_session_order(conn, who, name)
            current = conn.execute("SELECT * FROM locks WHERE name = ?", (name,)).fetchone()
            if _ident(current) != _ident(row):
                continue
            if (
                current is not None
                and current["holder_kind"] == "session"
                and current["token_subject"] == who.subject
                and current["holder_session_id"] == session_id
            ):
                conn.execute(
                    "UPDATE locks SET holder_epoch = ?, holder_pid = ?, holder_pid_start = ? WHERE name = ?",
                    (who.token.epoch, pid, pid_start, name),
                )
                return {"name": name, "status": "already_held"}
            if row_free(conn, current, facts):
                conn.execute("DELETE FROM locks WHERE name = ?", (name,))
                conn.execute(
                    "INSERT INTO locks (name, holder_kind, holder_pid, holder_pid_start, holder_session_id, holder_epoch,"
                    " holder_actor, token_subject, purpose, acquired_at) VALUES (?, 'session', ?, ?, ?, ?, ?, ?, ?, ?)",
                    (name, pid, pid_start, session_id, who.token.epoch, who.actor, who.subject, purpose, db.now(clock)),
                )
                return {"name": name, "status": "acquired"}
            blocker = current
        if clock.now().timestamp() >= deadline:
            assert blocker is not None
            raise _busy(blocker, name)
        clock.sleep(poll)


def release_session(conn: sqlite3.Connection, *, token: str, name: str, session_id: str) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run", "node"}, allow_stopped=True)
        row = conn.execute("SELECT * FROM locks WHERE name = ?", (name,)).fetchone()
        if row is None:
            return {"name": name, "released": False}
        if (
            row["holder_kind"] != "session"
            or row["token_subject"] != who.subject
            or row["holder_session_id"] != session_id
        ):
            raise errors.CpError("CONFLICT", f"lock {name} is not held by this session and token")
        conn.execute("DELETE FROM locks WHERE name = ?", (name,))
    return {"name": name, "released": True}


# --------------------------------------------------------------------------- the node cap (§8.3)


def spawn_holders_alive(conn: sqlite3.Connection, probe: liveness.ProcessProbe) -> dict[str, bool]:
    """Outside the transaction: is each ``spawning`` node's spawn tool-call holder alive?"""
    rows = conn.execute(
        "SELECT n.id, t.holder_pid, t.holder_pid_start, t.holder_pgid FROM nodes n"
        " LEFT JOIN tool_calls t ON (t.tool || '/' || t.key) = n.spawn_tool_key WHERE n.state = 'spawning'"
    ).fetchall()
    return {r["id"]: probe.holder_alive(r["holder_pid"], r["holder_pid_start"], r["holder_pgid"]) for r in rows}


def admit_node(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    listing: list[liveness.Agent] | None,
    holders_alive: dict[str, bool],
    limit: int,
    settle_s: float = nodes.LAUNCH_SETTLE_S,
) -> int:
    """Inside the reserving transaction: reap dead ``spawning`` rows, count the rest, ``BUSY`` at the cap."""
    count = 0
    rows = conn.execute(
        "SELECT n.*, t.state AS call_state FROM nodes n LEFT JOIN tool_calls t"
        " ON (t.tool || '/' || t.key) = n.spawn_tool_key WHERE n.state IN ('spawning', 'running', 'idle', 'taken_over')"
    ).fetchall()
    for row in rows:
        on_list = listing is not None and nodes.listed(listing, row) is not None
        if row["state"] == "spawning":
            elapsed = clock_mod.age(clock, row["launch_at"])
            settled = elapsed is None or elapsed >= settle_s
            alive = holders_alive.get(row["id"], True)
            if row["call_state"] != "succeeded" and not alive and listing is not None and not on_list and settled:
                conn.execute(
                    "UPDATE nodes SET state = 'reaped', updated_at = ? WHERE id = ? AND state = 'spawning'",
                    (db.now(clock), row["id"]),
                )
                continue
            count += 1
        elif listing is None or on_list:
            count += 1
    if count >= limit:
        raise errors.CpError("BUSY", f"node cap reached ({count}/{limit})")
    return count
