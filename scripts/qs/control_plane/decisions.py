"""Decisions (§7.2): what was decided, why, and on whose word."""

from __future__ import annotations

import sqlite3
from typing import Any

from . import clock as clock_mod
from . import db, tasks, tokens


def add(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    token: str,
    text: str,
    reason: str,
    source: str,
    task_id: str | None,
) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run"}, task_id=task_id)
        if task_id is not None:
            tasks.get(conn, task_id)
        cur = conn.execute(
            "INSERT INTO decisions (run_id, task_id, text, reason, source, at) VALUES (?, ?, ?, ?, ?, ?)",
            (who.run_id, task_id, text, reason, source, db.now(clock)),
        )
    return {"decision_id": cur.lastrowid}
