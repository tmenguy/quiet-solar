"""Questions (§7.2): ``open → asked → answered``.

Opening a question also posts a ``question`` message to the orchestrator;
answering posts an ``answer`` message back to the asking node, if a node
asked.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from . import clock as clock_mod
from . import db, errors, messages, tasks, tokens


def _get(conn: sqlite3.Connection, question_id: str, run_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM questions WHERE id = ? AND run_id = ?", (question_id, run_id)).fetchone()
    if row is None:
        raise errors.CpError("NOT_FOUND", f"no question {question_id} in run {run_id}")
    return row


def open_question(
    conn: sqlite3.Connection, clock: clock_mod.Clock, *, token: str, task_id: str, text: str, blocking: bool
) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run", "node"}, task_id=task_id)
        tasks.get(conn, task_id)
        question_id = db.next_id(conn, "question", "Q-")
        msg = messages.post_locked(
            conn,
            clock,
            run_id=who.run_id,
            recipient=messages.ORCHESTRATOR,
            kind="question",
            sender=who.actor,
            payload={"question_id": question_id, "task_id": task_id, "text": text, "blocking": blocking},
        )
        conn.execute(
            "INSERT INTO questions (id, run_id, task_id, message_id, text, blocking, state, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, 'open', ?)",
            (question_id, who.run_id, task_id, msg["id"], text, int(blocking), db.now(clock)),
        )
    return {"question_id": question_id, "message_id": msg["id"]}


def ask(conn: sqlite3.Connection, *, token: str, question_id: str) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run"})
        row = _get(conn, question_id, who.run_id)
        if row["state"] != "open":
            raise errors.CpError("INVALID_STATE", f"question {question_id} is {row['state']}")
        conn.execute("UPDATE questions SET state = 'asked' WHERE id = ?", (question_id,))
    return {"question_id": question_id, "state": "asked"}


def answer(
    conn: sqlite3.Connection, clock: clock_mod.Clock, *, token: str, question_id: str, answer_text: str, reason: str
) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run"})
        row = _get(conn, question_id, who.run_id)
        if row["state"] == "answered":
            raise errors.CpError("INVALID_STATE", f"question {question_id} is already answered")
        conn.execute(
            "UPDATE questions SET state = 'answered', answer = ?, reason = ?, answered_at = ? WHERE id = ?",
            (answer_text, reason, db.now(clock), question_id),
        )
        asker = conn.execute("SELECT sender FROM messages WHERE id = ?", (row["message_id"],)).fetchone()
        message_id = None
        if asker is not None and asker["sender"].startswith("node:"):
            message_id = messages.post_locked(
                conn,
                clock,
                run_id=who.run_id,
                recipient=asker["sender"],
                kind="answer",
                sender=who.actor,
                payload={"question_id": question_id, "answer": answer_text, "reason": reason},
            )["id"]
    return {"question_id": question_id, "state": "answered", "message_id": message_id}
