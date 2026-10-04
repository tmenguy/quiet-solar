"""Run queues (§7.1): one per recipient, in order of arrival.

``queued → in_flight`` (visibility timeout + a fresh receipt) ``→ acked``, or
``dead`` once a message reached ``MAX_ATTEMPTS`` pops. Not acking in time is
the requeue path: an expired in-flight message is redelivered first (its id
is lower).
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from typing import Any

from . import clock as clock_mod
from . import db, errors, faults, runs, tokens

MAX_ATTEMPTS = 5
VISIBILITY_S = 900.0
ORCHESTRATOR = "orchestrator"


def canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_payload(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError as exc:
        raise errors.CpError("USAGE", f"the payload is not JSON: {exc}") from exc


def _check_recipient(conn: sqlite3.Connection, run_id: str, recipient: str) -> None:
    if recipient == ORCHESTRATOR:
        return
    if not recipient.startswith("node:"):
        raise errors.CpError("USAGE", f"recipient must be `orchestrator` or `node:<task>`: {recipient!r}")
    task = conn.execute("SELECT run_id FROM tasks WHERE id = ?", (recipient[5:],)).fetchone()
    if task is None or task["run_id"] != run_id:
        raise errors.CpError("NOT_FOUND", f"no task {recipient[5:]} in run {run_id}")


def post_locked(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    run_id: str,
    recipient: str,
    kind: str,
    sender: str,
    payload: Any,
    dedupe_key: str | None = None,
) -> dict[str, Any]:
    """Post inside the caller's transaction (also used by questions and reports)."""
    if not 1 <= len(kind) <= 32:
        raise errors.CpError("USAGE", "--kind must be 1 to 32 characters")
    _check_recipient(conn, run_id, recipient)
    if dedupe_key is not None:
        existing = conn.execute(
            "SELECT id FROM messages WHERE run_id = ? AND dedupe_key = ?", (run_id, dedupe_key)
        ).fetchone()
        if existing is not None:
            return {"id": existing["id"], "deduped": True}
    now = db.now(clock)
    cur = conn.execute(
        "INSERT INTO messages (run_id, recipient, kind, sender, payload, state, visible_at, dedupe_key, created_at)"
        " VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)",
        (run_id, recipient, kind, sender, canonical(payload), now, dedupe_key, now),
    )
    return {"id": cur.lastrowid, "deduped": False}


def _own_run(conn: sqlite3.Connection, who: tokens.Principal, run_ref: str) -> str:
    run = runs.resolve(conn, run_ref)
    if run["id"] != who.run_id:
        raise errors.CpError("CONFLICT", f"the token is for run {who.run_id}, not {run_ref}")
    return str(run["id"])


def post(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    token: str,
    run_ref: str,
    to: str,
    kind: str,
    payload: Any,
    dedupe_key: str | None,
) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run", "node"})
        run_id = _own_run(conn, who, run_ref)
        runs.require_open(conn, run_id)
        if who.kind == "node" and to != ORCHESTRATOR:
            raise errors.CpError("CONFLICT", "a node may only post to `orchestrator`")
        return post_locked(
            conn,
            clock,
            run_id=run_id,
            recipient=to,
            kind=kind,
            sender=who.actor,
            payload=payload,
            dedupe_key=dedupe_key,
        )


def _check_reader(who: tokens.Principal, recipient: str) -> None:
    if recipient == ORCHESTRATOR:
        if who.kind != "run":
            raise errors.CpError("CONFLICT", "only the run token reads the orchestrator's queue")
    elif who.kind != "node" or recipient != f"node:{who.task_id}":
        raise errors.CpError("CONFLICT", f"only {recipient}'s own node token reads its queue")


def pop(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    token: str,
    run_ref: str,
    recipient: str,
    visibility: float = VISIBILITY_S,
    max_attempts: int = MAX_ATTEMPTS,
) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run", "node"}, allow_stopped=True)
        run_id = _own_run(conn, who, run_ref)
        _check_reader(who, recipient)
        now = db.now(clock)
        while True:
            row = conn.execute(
                "SELECT * FROM messages WHERE run_id = ? AND recipient = ?"
                " AND (state = 'queued' OR (state = 'in_flight' AND visible_at <= ?)) ORDER BY id LIMIT 1",
                (run_id, recipient, now),
            ).fetchone()
            faults.hit("pop.after_select")
            if row is None:
                return {"empty": True}
            if row["attempts"] >= max_attempts:
                conn.execute("UPDATE messages SET state = 'dead' WHERE id = ?", (row["id"],))
                continue
            receipt = secrets.token_hex(8)
            conn.execute(
                "UPDATE messages SET state = 'in_flight', visible_at = ?, attempts = attempts + 1, receipt = ?,"
                " popped_by = ? WHERE id = ?",
                (clock_mod.stamp(clock, plus=visibility), receipt, who.actor, row["id"]),
            )
            return {
                "empty": False,
                "id": row["id"],
                "kind": row["kind"],
                "sender": row["sender"],
                "payload": json.loads(row["payload"]),
                "attempt": row["attempts"] + 1,
                "receipt": receipt,
            }


def ack(conn: sqlite3.Connection, clock: clock_mod.Clock, *, token: str, msg_id: int, receipt: str) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run", "node"}, allow_stopped=True)
        row = conn.execute("SELECT * FROM messages WHERE id = ?", (msg_id,)).fetchone()
        if row is None or row["run_id"] != who.run_id:
            raise errors.CpError("NOT_FOUND", f"no message {msg_id} in run {who.run_id}")
        _check_reader(who, row["recipient"])
        if row["state"] in ("queued", "dead"):
            raise errors.CpError("INVALID_STATE", f"message {msg_id} is {row['state']}")
        if row["receipt"] != receipt:
            raise errors.CpError("CONFLICT", f"stale receipt: message {msg_id} was redelivered")
        if row["state"] == "acked":
            return {"id": msg_id, "acked": True, "noop": True}
        conn.execute("UPDATE messages SET state = 'acked', acked_at = ? WHERE id = ?", (db.now(clock), msg_id))
    return {"id": msg_id, "acked": True, "noop": False}


def visible_count(conn: sqlite3.Connection, run_id: str, recipient: str, now: str) -> int:
    row = conn.execute(
        "SELECT count(*) FROM messages WHERE run_id = ? AND recipient = ?"
        " AND (state = 'queued' OR (state = 'in_flight' AND visible_at <= ?))",
        (run_id, recipient, now),
    ).fetchone()
    return int(row[0])


def head_id(conn: sqlite3.Connection, run_id: str, recipient: str, now: str) -> int | None:
    row = conn.execute(
        "SELECT id FROM messages WHERE run_id = ? AND recipient = ?"
        " AND (state = 'queued' OR (state = 'in_flight' AND visible_at <= ?)) ORDER BY id LIMIT 1",
        (run_id, recipient, now),
    ).fetchone()
    return None if row is None else int(row[0])
