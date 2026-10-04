"""Node sessions (§4 "Node states", §9.5): one ``nodes`` row per generation.

``TRANSITIONS`` is enforced like the task table. ``refresh`` reconciles
``running`` / ``idle`` rows with a liveness listing taken outside the
transaction; it never touches ``spawning`` rows (they belong to their tool
call and the cap's reap rule) nor terminal ones.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from . import clock as clock_mod
from . import db, errors, liveness, messages, tasks, tokens

LAUNCH_SETTLE_S = 60.0
TERMINAL = frozenset({"stopped", "superseded"})
TRANSITIONS: dict[str, frozenset[str]] = {
    "spawning": frozenset({"reaped", "running", "stopped", "superseded"}),
    "running": frozenset({"idle", "reaped", "taken_over", "stopped", "superseded"}),
    "idle": frozenset({"running", "reaped", "taken_over", "stopped", "superseded"}),
    "taken_over": frozenset({"running", "stopped", "superseded"}),
    "reaped": frozenset({"spawning", "stopped", "superseded"}),
    "stopped": frozenset(),
    "superseded": frozenset(),
}
LIVE_STATES = ("spawning", "running", "idle", "taken_over")


def check(current: str, to: str) -> None:
    if to not in TRANSITIONS.get(current, frozenset()):
        raise errors.CpError("INVALID_STATE", f"forbidden node transition {current} → {to}")


def move(
    conn: sqlite3.Connection, clock: clock_mod.Clock, node_id: str, to: str, *, expect: str | None = None, **cols: Any
) -> sqlite3.Row:
    """Transition a node inside the caller's transaction (a compare-and-set when ``expect`` is given)."""
    row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
    if row is None:
        raise errors.CpError("NOT_FOUND", f"unknown node {node_id}")
    if expect is not None and row["state"] != expect:
        raise errors.CpError("INVALID_STATE", f"node {node_id} is {row['state']}, expected {expect}")
    check(row["state"], to)
    sets = ", ".join(["state = ?", "updated_at = ?", *(f"{k} = ?" for k in cols)])
    conn.execute(f"UPDATE nodes SET {sets} WHERE id = ?", (to, db.now(clock), *cols.values(), node_id))
    return row


def current(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row:
    """The task's newest generation."""
    row = conn.execute("SELECT * FROM nodes WHERE task_id = ? ORDER BY generation DESC LIMIT 1", (task_id,)).fetchone()
    if row is None:
        raise errors.CpError("NOT_FOUND", f"task {task_id} has no node")
    return row


def listed(listing: list[liveness.Agent], row: sqlite3.Row) -> liveness.Agent | None:
    if row["session_id"] is not None:
        found = liveness.find(listing, session_id=row["session_id"])
        if found is not None:
            return found
    return liveness.find(listing, name=row["name"])


def refresh(
    conn: sqlite3.Connection, clock: clock_mod.Clock, listing: list[liveness.Agent] | None, run_id: str | None = None
) -> dict[str, Any]:
    """Compare-and-set ``running`` / ``idle`` / ``reaped`` from a listing taken before the transaction."""
    if listing is None:
        return {"refreshed": False, "changes": []}
    changes: list[tuple[str, str, str]] = []
    with db.write(conn):
        rows = conn.execute(
            "SELECT * FROM nodes WHERE state IN ('running', 'idle') AND (? IS NULL OR run_id = ?)", (run_id, run_id)
        ).fetchall()
        for row in rows:
            agent = listed(listing, row)
            if agent is None:
                new = "reaped"
            else:
                new = "idle" if "idle" in (agent.status, agent.state) else "running"
            if new != row["state"]:
                conn.execute(
                    "UPDATE nodes SET state = ?, updated_at = ? WHERE id = ? AND state = ?",
                    (new, db.now(clock), row["id"], row["state"]),
                )
                changes.append((row["id"], row["state"], new))
    return {"refreshed": True, "changes": changes}


def stop(conn: sqlite3.Connection, clock: clock_mod.Clock, *, token: str, task_id: str) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run"}, task_id=task_id)
        node = current(conn, task_id)
        move(conn, clock, node["id"], "stopped")
        msg = messages.post_locked(
            conn,
            clock,
            run_id=who.run_id,
            recipient=f"node:{task_id}",
            kind="stop",
            sender=who.actor,
            payload={"node_id": node["id"]},
        )
    return {"node_id": node["id"], "state": "stopped", "message_id": msg["id"]}


def take_over(conn: sqlite3.Connection, clock: clock_mod.Clock, *, token: str, task_id: str) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run"}, task_id=task_id)
        node = current(conn, task_id)
        if node["state"] not in ("running", "idle"):
            raise errors.CpError("INVALID_STATE", f"node {node['id']} is {node['state']}")
        move(conn, clock, node["id"], "taken_over", taken_over_at=db.now(clock))
        msg = messages.post_locked(
            conn,
            clock,
            run_id=who.run_id,
            recipient=messages.ORCHESTRATOR,
            kind="info",
            sender=who.actor,
            payload={"event": "taken_over", "task_id": task_id, "node_id": node["id"]},
        )
    return {"node_id": node["id"], "state": "taken_over", "message_id": msg["id"]}


def hand_back(
    conn: sqlite3.Connection, clock: clock_mod.Clock, *, token: str, task_id: str, summary: str
) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run", "node"}, task_id=task_id)
        node = current(conn, task_id)
        move(conn, clock, node["id"], "running", expect="taken_over", handed_back_at=db.now(clock))
        msg = messages.post_locked(
            conn,
            clock,
            run_id=who.run_id,
            recipient=messages.ORCHESTRATOR,
            kind="hand_back",
            sender=who.actor,
            payload={"task_id": task_id, "node_id": node["id"], "summary": summary},
        )
    return {"node_id": node["id"], "state": "running", "message_id": msg["id"]}


def deliverable_of(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row:
    """The deliverable a task belongs to: itself, or its ``deliverable_id``."""
    task = tasks.get(conn, task_id)
    return task if task["is_deliverable"] or task["deliverable_id"] is None else tasks.get(conn, task["deliverable_id"])
