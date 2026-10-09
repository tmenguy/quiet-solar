"""Liveness and the watchdog (QS-406 §9).

The ``liveness_watchdog`` hook (every ``LIVENESS_EVERY_S``) takes one listing,
``claude.try_agents(timeout=HOOK_SUBPROCESS_S)`` (``None`` means unknown), and beats right after.

- **Nodes:** ``nodes.refresh``. ``node_dead``: a task in ``tasks.NODE_RANGE`` whose current generation
  is ``reaped``, seen on ``LIVENESS_CONFIRM`` listings in a row.
- **Orchestrators:** ``orchestrator_dead``: an open run whose lease session is absent from
  ``LIVENESS_CONFIRM`` successful listings in a row.
- A ``None`` listing leaves every liveness kind out of the sync and keeps the counters.
- **Cold state** (a new daemon): each sighting counter is seeded from the open ``alerts`` rows (an open
  subject counts as confirmed), and the kinds are left out until this daemon has taken
  ``LIVENESS_CONFIRM`` successful listings — so a restart is never a recurrence.

**The stalled-orchestrator ladder** (§9.2; QS-406 T1: no socket rung). A run is stalled when the listing
is known, it has no open ``orchestrator_dead``, its head orchestrator message has been visible for
``WATCHDOG_S``, no ``wait`` is live, and its lease session is listed idle. Then:

1. **messenger** — one ``claude --bg`` (``models``' fast class, ``--permission-mode auto``,
   ``--allowedTools=SendMessage``) that sends the run's orchestrator a wake text, recorded as a
   ``tool_calls`` row ``('watchdog-messenger', 'msg:<head id>')`` (idempotent; a cap of one launch per
   ``MESSENGER_TTL_S``). This is the one documented exception to "only a task's node is spawned".
2. **not listening** — ``orchestrator_not_listening`` while the run stays stalled ``WAKE_RETRY_S`` after
   the launch, at once when the launch failed or the run name is not bound to its lease session.

The ladder is derived each tick from the head's messenger row (no memory), so a new daemon adopts it.
Neither text ever contains a token. ``claude --bg`` runs with the CLI's own login: an expired login
exits 0 and the messenger then stops (T1) — the run is then flagged ``orchestrator_not_listening``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import models  # type: ignore[import-not-found]  # scripts/qs is on sys.path (cp.py, the conftest)

from . import activeloop, alerts, daemon, db, liveness, messages, nodes, paths, tasks, ticks, wait
from . import clock as clock_mod

if TYPE_CHECKING:
    from .activeloop import Seams

LIVENESS_WATCHDOG = "liveness_watchdog"
LIVENESS_EVERY_S = 30.0
LIVENESS_CONFIRM = 2
WATCHDOG_S = 120.0
WAKE_RETRY_S = 300.0
MESSENGER_TTL_S = 300.0  # = WAKE_RETRY_S: with N runs stalled at once, the last waits ~(N-1) × 5 min
MESSENGER_TOOL = "watchdog-messenger"
MESSENGER_ACTOR = "cp:daemon"

_THROTTLE = ticks.Throttle(LIVENESS_EVERY_S)


@dataclass
class _State:
    listings: int = 0
    seeded: bool = False
    node_seen: dict[str, int] = field(default_factory=dict)
    orch_seen: dict[str, int] = field(default_factory=dict)


_state = _State()


def _seed(conn: sqlite3.Connection) -> None:
    for row in conn.execute(
        "SELECT kind, subject FROM alerts WHERE cleared_at IS NULL AND kind IN (?, ?)",
        (alerts.NODE_DEAD, alerts.ORCHESTRATOR_DEAD),
    ).fetchall():
        seen = _state.node_seen if row["kind"] == alerts.NODE_DEAD else _state.orch_seen
        seen[row["subject"]] = LIVENESS_CONFIRM
    _state.seeded = True


def _count(seen: dict[str, int], present: dict[str, tuple[str, dict[str, object]]]) -> dict[str, int]:
    return {subject: seen.get(subject, 0) + 1 for subject in present}


def _dead_nodes(conn: sqlite3.Connection) -> dict[str, tuple[str, dict[str, object]]]:
    marks = ", ".join("?" for _ in tasks.NODE_RANGE)
    rows = conn.execute(
        f"SELECT n.id, n.run_id, n.task_id, n.generation FROM nodes n JOIN tasks t ON t.id = n.task_id"
        f" WHERE t.state IN ({marks}) AND n.state = 'reaped'"
        f" AND n.generation = (SELECT max(generation) FROM nodes WHERE task_id = n.task_id) ORDER BY n.id",
        tuple(sorted(tasks.NODE_RANGE)),
    ).fetchall()
    return {
        r["id"]: (r["run_id"], {"node_id": r["id"], "task_id": r["task_id"], "generation": r["generation"]})
        for r in rows
    }


def _dead_orchestrators(
    conn: sqlite3.Connection, listing: list[liveness.Agent]
) -> dict[str, tuple[str, dict[str, object]]]:
    rows = conn.execute(
        "SELECT r.id, l.session_id FROM runs r JOIN run_leases l ON l.run_id = r.id WHERE r.state = 'open' ORDER BY r.rowid"
    ).fetchall()
    return {
        f"{r['id']}:{r['session_id']}": (r["id"], {"run_id": r["id"], "session_id": r["session_id"]})
        for r in rows
        if liveness.find(listing, session_id=r["session_id"]) is None
    }


def liveness_watchdog_hook(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
    if not _THROTTLE.due(clock):
        return
    seams = activeloop.seams()
    listing = seams.claude.try_agents(timeout=ticks.HOOK_SUBPROCESS_S)
    daemon.beat(conn, clock)
    _fail_stale_messengers(conn, clock, seams)
    if listing is None:
        return  # unknown: nothing is synced, the counters and ladders are kept
    nodes.refresh(conn, clock, listing)
    if not _state.seeded:
        _seed(conn)
    _state.listings += 1
    _sync_liveness(conn, clock, listing)
    not_listening = [c for run in _open_leases(conn) if (c := _ladder(conn, clock, seams, listing, run)) is not None]
    alerts.sync(conn, clock, kinds={alerts.ORCHESTRATOR_NOT_LISTENING}, active=not_listening)


def _sync_liveness(conn: sqlite3.Connection, clock: clock_mod.Clock, listing: list[liveness.Agent]) -> None:
    dead_nodes = _dead_nodes(conn)
    dead_orchs = _dead_orchestrators(conn, listing)
    _state.node_seen = _count(_state.node_seen, dead_nodes)
    _state.orch_seen = _count(_state.orch_seen, dead_orchs)
    if _state.listings < LIVENESS_CONFIRM:
        return  # cold: not enough sightings by this daemon to clear or raise anything
    active = [
        alerts.Condition(alerts.NODE_DEAD, subject, (run,), payload)
        for subject, (run, payload) in dead_nodes.items()
        if _state.node_seen[subject] >= LIVENESS_CONFIRM
    ] + [
        alerts.Condition(alerts.ORCHESTRATOR_DEAD, subject, (run,), payload)
        for subject, (run, payload) in dead_orchs.items()
        if _state.orch_seen[subject] >= LIVENESS_CONFIRM
    ]
    alerts.sync(conn, clock, kinds={alerts.NODE_DEAD, alerts.ORCHESTRATOR_DEAD}, active=active)


# --------------------------------------------------------------------------- the ladder (§9.2)


def _fail_stale_messengers(conn: sqlite3.Connection, clock: clock_mod.Clock, seams: Seams) -> None:
    """A ``started`` messenger row whose holder is dead becomes ``failed`` (the launch never finished)."""
    rows = conn.execute(
        "SELECT key, holder_pid, holder_pid_start FROM tool_calls WHERE tool = ? AND state = 'started'",
        (MESSENGER_TOOL,),
    ).fetchall()
    dead = [r["key"] for r in rows if not seams.probe.holder_alive(r["holder_pid"], r["holder_pid_start"], None)]
    if dead:
        with db.write(conn):
            conn.executemany(
                "UPDATE tool_calls SET state = 'failed', finished_at = ? WHERE tool = ? AND key = ? AND state = 'started'",
                [(db.now(clock), MESSENGER_TOOL, key) for key in dead],
            )


def _open_leases(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT r.id, r.name, l.session_id, l.name_bound_session_id FROM runs r JOIN run_leases l ON l.run_id = r.id"
        " WHERE r.state = 'open' ORDER BY r.rowid"
    ).fetchall()


def _open_alert(conn: sqlite3.Connection, run_id: str, kind: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT payload FROM alerts WHERE run_id = ? AND kind = ? AND cleared_at IS NULL", (run_id, kind)
    ).fetchone()


def _kept(conn: sqlite3.Connection, run_id: str) -> alerts.Condition | None:
    """An open ``orchestrator_not_listening`` re-emitted as is: its run's state is unknown, not resolved."""
    row = _open_alert(conn, run_id, alerts.ORCHESTRATOR_NOT_LISTENING)
    return (
        None
        if row is None
        else alerts.Condition(alerts.ORCHESTRATOR_NOT_LISTENING, run_id, (run_id,), json.loads(row[0]))
    )


def cp_command(seams: Seams) -> str:
    return f"{seams.main}/venv/bin/python {seams.main}/scripts/qs/cp.py"


def wake_text(seams: Seams, run_id: str, waiting: int) -> str:
    return (
        f"Run {run_id} has {waiting} message(s) waiting in the Control Plane. Pop the next with"
        f" `{cp_command(seams)} msg pop --run {run_id} --as orchestrator --token <your run token>`, handle it,"
        " ack it; then restart `wait`."
    )


def messenger_prompt(run_name: str, text: str) -> str:
    return (
        f"You are a one-shot messenger. Call the SendMessage tool exactly once, to the session named `{run_name}`,"
        f" with this message: {text} Do nothing else, then stop."
    )


def messenger_args(run_id: str, head: int, prompt: str) -> list[str]:
    """``claude --bg`` arguments (D12, with T1's fixes: ``--allowedTools=`` is one word, or it swallows the prompt)."""
    return [
        "-n",
        f"qs-wake-{run_id}-m{head}",
        "--model",
        models.model_for("claude", "fast"),
        "--permission-mode",
        "auto",
        "--allowedTools=SendMessage",
        prompt,
    ]


def _cap_taken(conn: sqlite3.Connection, clock: clock_mod.Clock) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM tool_calls WHERE tool = ? AND state IN ('started', 'succeeded') AND started_at > ? LIMIT 1",
            (MESSENGER_TOOL, clock_mod.stamp(clock, plus=-MESSENGER_TTL_S)),
        ).fetchone()
        is not None
    )


def _launch(
    conn: sqlite3.Connection, clock: clock_mod.Clock, seams: Seams, run: sqlite3.Row, head: int, waiting: int
) -> bool | None:
    """Claim and launch the head's messenger → ``True`` launched, ``False`` failed, ``None`` lost the claim."""
    key = f"msg:{head}"
    args = messenger_args(run["id"], head, messenger_prompt(run["name"], wake_text(seams, run["id"], waiting)))
    me = seams.probe.me()
    with db.write(conn):
        if conn.execute("SELECT 1 FROM tool_calls WHERE tool = ? AND key = ?", (MESSENGER_TOOL, key)).fetchone():
            return None  # another tick (or daemon) claimed it first
        conn.execute(
            "INSERT INTO tool_calls (tool, key, args_hash, run_id, task_id, actor, args, state, holder_pid,"
            " holder_pid_start, started_at) VALUES (?, ?, '-', ?, NULL, ?, ?, 'started', ?, ?, ?)",
            (MESSENGER_TOOL, key, run["id"], MESSENGER_ACTOR, json.dumps(args), me.pid, me.pid_start, db.now(clock)),
        )
    directory = paths.ensure_private_dir(paths.messenger_dir())
    res = seams.claude.spawn_bg(args, cwd=directory, timeout=ticks.HOOK_SUBPROCESS_S)
    daemon.beat(conn, clock)
    with db.write(conn):
        conn.execute(
            "UPDATE tool_calls SET state = ?, finished_at = ?, exit_code = ?, result = ? WHERE tool = ? AND key = ?",
            (
                "succeeded" if res.ok else "failed",
                db.now(clock),
                res.returncode,
                json.dumps(
                    {"stdout_tail": res.stdout.strip().splitlines()[-5:], "stderr_tail": res.stderr.strip()[-300:]}
                ),
                MESSENGER_TOOL,
                key,
            ),
        )
    return res.ok


def _ladder(
    conn: sqlite3.Connection, clock: clock_mod.Clock, seams: Seams, listing: list[liveness.Agent], run: sqlite3.Row
) -> alerts.Condition | None:
    """The run's ``orchestrator_not_listening`` condition this tick, or ``None`` (launching the messenger on the way)."""
    run_id = run["id"]
    now = db.now(clock)
    head = messages.head_id(conn, run_id, messages.ORCHESTRATOR, now)
    if head is None:
        return None  # drained
    since = conn.execute("SELECT COALESCE(visible_at, created_at) FROM messages WHERE id = ?", (head,)).fetchone()[0]
    if since > clock_mod.stamp(clock, plus=-WATCHDOG_S):
        return None  # a new head: not stalled (yet)
    session = liveness.find(listing, session_id=run["session_id"])
    if (
        _open_alert(conn, run_id, alerts.ORCHESTRATOR_DEAD) is not None
        or wait.live_waiter(conn, seams.probe, run_id)
        or session is None
        or "idle" not in (session.status, session.state)
    ):
        return _kept(conn, run_id)  # dead, waiting, absent or busy: not this ladder's case
    waiting = messages.visible_count(conn, run_id, messages.ORCHESTRATOR, now)
    payload: dict[str, Any] = {"run_id": run_id, "head_message_id": head, "waiting": waiting}

    def not_listening(reason: str) -> alerts.Condition:
        return alerts.Condition(alerts.ORCHESTRATOR_NOT_LISTENING, run_id, (run_id,), {**payload, "reason": reason})

    row = conn.execute(
        "SELECT state, started_at FROM tool_calls WHERE tool = ? AND key = ?", (MESSENGER_TOOL, f"msg:{head}")
    ).fetchone()
    if row is None:
        if run["name_bound_session_id"] != run["session_id"]:
            return not_listening("name_unbound")  # `SendMessage` addresses the run name: it would reach another session
        if _cap_taken(conn, clock):
            return None  # another run's messenger holds the cap: retry on the next tick
        launched = _launch(conn, clock, seams, run, head, waiting)
        return not_listening("messenger_failed") if launched is False else None
    if row["state"] == "failed":
        return not_listening("messenger_failed")
    if row["state"] == "succeeded":
        try:
            due = clock_mod.parse(row["started_at"]) + timedelta(seconds=WAKE_RETRY_S) <= clock.now()
        except ValueError:
            due = True
        return not_listening("still_stalled") if due else None
    return None  # started, holder alive: the launch is in progress


def _reset_for_tests() -> None:
    global _state
    _state = _State()
