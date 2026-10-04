"""Runs, run leases and the run-name takeover rule (§6.2), plus ``session status`` (§6.4).

Liveness listings run **before** the transaction; inside ``BEGIN IMMEDIATE``
the claim proceeds only if the lease still matches what was probed
(compare-and-set). A failed listing is never read as absence.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from . import clock as clock_mod
from . import db, errors, liveness, tokens

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
CLAIM_ATTEMPTS = 2


def resolve(conn: sqlite3.Connection, ref: str) -> sqlite3.Row:
    """A run by id or by name."""
    row = conn.execute("SELECT * FROM runs WHERE id = ? OR name = ?", (ref, ref)).fetchone()
    if row is None:
        raise errors.CpError("NOT_FOUND", f"unknown run {ref}")
    return row


def open_run(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    name: str,
    title: str,
    session_id: str,
    session_name: str | None,
    permission_mode: str | None,
    full_grant: bool,
) -> dict[str, Any]:
    if not _NAME_RE.match(name):
        raise errors.CpError("USAGE", f"--name must be a slug ([a-z0-9-], ≤ 48 chars): {name!r}")
    nonce = tokens.new_nonce()
    now = db.now(clock)
    with db.write(conn):
        if conn.execute("SELECT 1 FROM runs WHERE name = ?", (name,)).fetchone() is not None:
            raise errors.CpError("CONFLICT", f"a run named {name!r} exists")
        run_id = db.next_id(conn, "run", "R")
        conn.execute(
            "INSERT INTO runs (id, name, title, state, created_at) VALUES (?, ?, ?, 'open', ?)",
            (run_id, name, title, now),
        )
        conn.execute(
            "INSERT INTO run_leases (run_id, epoch, nonce, session_id, session_name, permission_mode, full_grant,"
            " claimed_at, name_bound_session_id) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, nonce, session_id, session_name, permission_mode, int(full_grant), now, session_id),
        )
        conn.execute(
            "INSERT INTO run_sessions (run_id, epoch, session_id, claimed_at) VALUES (?, 1, ?, ?)",
            (run_id, session_id, now),
        )
    return {"run_id": run_id, "name": name, "epoch": 1, "token": tokens.mint("run", run_id, 1, nonce)}


def _holder_info(listing: list[liveness.Agent] | None, session_id: str, lease: sqlite3.Row) -> dict[str, Any]:
    agent = None if listing is None else liveness.find(listing, session_id=session_id)
    return {
        "holder_session_id": session_id,
        "holder_name": agent.name if agent else lease["session_name"],
        "holder_short_id": agent.id if agent else lease["short_id"],
    }


def claim(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    claude: liveness.ClaudeCli,
    *,
    run_ref: str,
    session_id: str,
    takeover: bool,
    session_name: str | None,
    permission_mode: str | None,
    full_grant: bool,
) -> dict[str, Any]:
    """Claim a run's lease: re-issue (same session), take (absent holder) or take over (``--takeover``)."""
    attempt = 0
    while True:
        attempt += 1
        run = resolve(conn, run_ref)
        probed = conn.execute("SELECT epoch, session_id FROM run_leases WHERE run_id = ?", (run["id"],)).fetchone()
        listing = claude.try_agents()
        with db.write(conn):
            lease = conn.execute("SELECT * FROM run_leases WHERE run_id = ?", (run["id"],)).fetchone()
            if conn.execute("SELECT state FROM runs WHERE id = ?", (run["id"],)).fetchone()["state"] != "open":
                raise errors.CpError("INVALID_STATE", f"run {run['id']} is closed")
            if (lease["epoch"], lease["session_id"]) != (probed["epoch"], probed["session_id"]):
                if takeover or attempt >= CLAIM_ATTEMPTS:
                    raise errors.CpError("CONFLICT", "lease changed, retry")
                continue
            return _claim_locked(
                conn, clock, run, lease, listing, session_id, takeover, session_name, permission_mode, full_grant
            )


def _claim_locked(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    run: sqlite3.Row,
    lease: sqlite3.Row,
    listing: list[liveness.Agent] | None,
    session_id: str,
    takeover: bool,
    session_name: str | None,
    permission_mode: str | None,
    full_grant: bool,
) -> dict[str, Any]:
    now = db.now(clock)
    holder = lease["session_id"]
    epoch = lease["epoch"] + 1
    nonce = tokens.new_nonce()
    me = None if listing is None else liveness.find(listing, session_id=session_id)
    short_id = me.id if me else None
    result: dict[str, Any] = {}
    name_bound = lease["name_bound_session_id"]
    pending = lease["pending_bind_session_id"]
    if holder == session_id:
        status = "reclaimed"
    else:
        holder_alive = listing is None or liveness.find(listing, session_id=holder) is not None
        if holder_alive and not takeover:
            raise errors.CpError(
                "CONFLICT",
                "the run's orchestrator is alive"
                + (" (liveness unknown)" if listing is None else "")
                + "; ask the maintainer, then retry with --takeover",
                **_holder_info(listing, holder, lease),
            )
        conn.execute(
            "UPDATE run_sessions SET superseded_at = ?, superseded_by = ? WHERE run_id = ? AND session_id = ?"
            " AND superseded_at IS NULL",
            (now, session_id, run["id"], holder),
        )
        if holder_alive:
            status = "taken_over"
            pending = session_id
            result["instructions"] = (
                "Ask the maintainer to archive the previous orchestrator session "
                f"({holder}), then call `cp.py run bind-name {run['id']} --token …`."
            )
        else:
            status = "claimed"
            name_bound = session_id
            pending = None
    conn.execute(
        "UPDATE run_leases SET epoch = ?, nonce = ?, session_id = ?, short_id = ?,"
        " session_name = COALESCE(?, session_name), permission_mode = COALESCE(?, permission_mode),"
        " full_grant = ?, claimed_at = ?, name_bound_session_id = ?, pending_bind_session_id = ? WHERE run_id = ?",
        (
            epoch,
            nonce,
            session_id,
            short_id,
            session_name,
            permission_mode,
            int(full_grant),
            now,
            name_bound,
            pending,
            run["id"],
        ),
    )
    conn.execute(
        "INSERT INTO run_sessions (run_id, epoch, session_id, claimed_at) VALUES (?, ?, ?, ?)",
        (run["id"], epoch, session_id, now),
    )
    return {
        "run_id": run["id"],
        "status": status,
        "epoch": epoch,
        "token": tokens.mint("run", run["id"], epoch, nonce),
        **result,
    }


def _own_run(conn: sqlite3.Connection, who: tokens.Principal, run_ref: str) -> None:
    if resolve(conn, run_ref)["id"] != who.run_id:
        raise errors.CpError("CONFLICT", f"the token is for run {who.run_id}, not {run_ref}")


def bind_name(conn: sqlite3.Connection, claude: liveness.ClaudeCli, *, token: str, run_ref: str) -> dict[str, Any]:
    listing = claude.try_agents()
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run"})
        _own_run(conn, who, run_ref)
        lease = conn.execute("SELECT * FROM run_leases WHERE run_id = ?", (who.run_id,)).fetchone()
        if lease["pending_bind_session_id"] is None:
            raise errors.CpError("INVALID_STATE", "no name binding is pending")
        if listing is None:
            raise errors.CpError("CONFLICT", "liveness unknown (the session listing failed); retry")
        if liveness.find(listing, session_id=lease["name_bound_session_id"]) is not None:
            raise errors.CpError(
                "CONFLICT",
                f"the session bound to the run name ({lease['name_bound_session_id']}) is still listed;"
                " ask the maintainer to archive it",
            )
        conn.execute(
            "UPDATE run_leases SET name_bound_session_id = pending_bind_session_id, pending_bind_session_id = NULL"
            " WHERE run_id = ?",
            (who.run_id,),
        )
    return {"run_id": who.run_id, "name_bound_session_id": lease["pending_bind_session_id"]}


def set_mode(conn: sqlite3.Connection, *, token: str, permission_mode: str, full_grant: bool | None) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run"})
        conn.execute(
            "UPDATE run_leases SET permission_mode = ?, full_grant = COALESCE(?, full_grant) WHERE run_id = ?",
            (permission_mode, None if full_grant is None else int(full_grant), who.run_id),
        )
        row = conn.execute(
            "SELECT permission_mode, full_grant FROM run_leases WHERE run_id = ?", (who.run_id,)
        ).fetchone()
    return {"run_id": who.run_id, "permission_mode": row["permission_mode"], "full_grant": bool(row["full_grant"])}


def require_open(conn: sqlite3.Connection, run_id: str) -> None:
    if conn.execute("SELECT state FROM runs WHERE id = ?", (run_id,)).fetchone()["state"] != "open":
        raise errors.CpError("INVALID_STATE", f"run {run_id} is closed")


def set_plan(conn: sqlite3.Connection, *, token: str, run_ref: str, plan: str) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run"})
        _own_run(conn, who, run_ref)
        require_open(conn, who.run_id)
        conn.execute("UPDATE runs SET global_plan = ? WHERE id = ?", (plan, who.run_id))
    return {"run_id": who.run_id, "bytes": len(plan.encode())}


def close(conn: sqlite3.Connection, clock: clock_mod.Clock, *, token: str) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run"})
        require_open(conn, who.run_id)
        conn.execute("UPDATE runs SET state = 'closed', closed_at = ? WHERE id = ?", (db.now(clock), who.run_id))
    return {"run_id": who.run_id, "state": "closed"}


def session_status(conn: sqlite3.Connection | None, session_id: str) -> dict[str, Any]:
    """``{role: orchestrator|node|none, state: current|superseded|stopped, run_id, task_id}``."""
    none = {"role": "none", "state": None, "run_id": None, "task_id": None}
    if conn is None:
        return none
    lease = conn.execute("SELECT run_id FROM run_leases WHERE session_id = ?", (session_id,)).fetchone()
    if lease is not None:
        return {"role": "orchestrator", "state": "current", "run_id": lease["run_id"], "task_id": None}
    old = conn.execute(
        "SELECT run_id FROM run_sessions WHERE session_id = ? AND superseded_at IS NOT NULL ORDER BY claimed_at DESC",
        (session_id,),
    ).fetchone()
    if old is not None:
        return {"role": "orchestrator", "state": "superseded", "run_id": old["run_id"], "task_id": None}
    node = conn.execute(
        "SELECT run_id, task_id, state FROM nodes WHERE session_id = ? ORDER BY generation DESC", (session_id,)
    ).fetchone()
    if node is not None:
        state = node["state"] if node["state"] in ("superseded", "stopped") else "current"
        return {"role": "node", "state": state, "run_id": node["run_id"], "task_id": node["task_id"]}
    return none
