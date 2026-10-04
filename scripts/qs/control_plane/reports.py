"""Node reports and per-task digests (§7.2).

A report is a ``reports`` row **and** a ``report`` message to the
orchestrator, in one transaction. A digest is replaced, never appended,
and capped at ``DIGEST_MAX_BYTES``.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from . import clock as clock_mod
from . import db, errors, messages, tasks, tokens

DIGEST_MAX_BYTES = 16 * 1024
STATUSES = ("converged", "continuing", "blocked")


def post_report(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    token: str,
    task_id: str,
    phase: str,
    round_: int,
    status: str,
    summary: str,
    fields: Any,
) -> dict[str, Any]:
    if status not in STATUSES:
        raise errors.CpError("USAGE", f"--status must be one of {', '.join(STATUSES)}")
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run", "node"}, task_id=task_id)
        tasks.get(conn, task_id)
        cur = conn.execute(
            "INSERT INTO reports (task_id, node_id, phase, round, status, summary, fields, at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, who.node_id, phase, round_, status, summary, messages.canonical(fields), db.now(clock)),
        )
        report_id = cur.lastrowid
        msg = messages.post_locked(
            conn,
            clock,
            run_id=who.run_id,
            recipient=messages.ORCHESTRATOR,
            kind="report",
            sender=who.actor,
            payload={
                "report_id": report_id,
                "task_id": task_id,
                "phase": phase,
                "round": round_,
                "status": status,
                "summary": summary,
            },
        )
    return {"report_id": report_id, "message_id": msg["id"]}


def put_digest(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    token: str,
    task_id: str,
    body: str,
    max_bytes: int = DIGEST_MAX_BYTES,
) -> dict[str, Any]:
    size = len(body.encode())
    if size > max_bytes:
        raise errors.CpError("CONFLICT", f"the digest is {size} bytes; the cap is {max_bytes}")
    with db.write(conn):
        tokens.require(conn, token, kinds={"run", "node"}, task_id=task_id)
        tasks.get(conn, task_id)
        conn.execute(
            "INSERT INTO digests (task_id, body, updated_at) VALUES (?, ?, ?) ON CONFLICT(task_id) DO UPDATE SET"
            " body = excluded.body, updated_at = excluded.updated_at",
            (task_id, body, db.now(clock)),
        )
    return {"task_id": task_id, "bytes": size}
