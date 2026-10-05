"""Fencing tokens (§6.1, §6.3): ``<subject>.<epoch>.<nonce>``.

A run token is ``run:R7.<epoch>.<nonce>``; a node token is
``node:N12.<generation>.<nonce>`` (one ``nodes`` row per generation). A token
is valid iff its subject row exists and both the epoch and the nonce match.
``require`` runs **inside** the write transaction it guards.
"""

from __future__ import annotations

import re
import secrets
import sqlite3
from collections.abc import Collection
from dataclasses import dataclass

from . import errors

INSTRUCTIONS = (
    "You were superseded. Stop your event loop now: do not start `wait` again. Tell the maintainer this session "
    "was superseded and ask them to archive it. Take no further action, including after an app restart."
)
_TOKEN_RE = re.compile(r"^(run|node):([A-Z]\d+)\.(\d+)\.([0-9a-f]{32})$")
TERMINAL_TASK_STATES = frozenset({"dropped", "merged", "validated"})


@dataclass(frozen=True)
class Token:
    kind: str
    subject: str
    epoch: int
    nonce: str

    @property
    def subject_ref(self) -> str:
        """The lock-row ``token_subject``: ``run:R7`` or ``node:N12`` (no epoch)."""
        return f"{self.kind}:{self.subject}"


def new_nonce() -> str:
    return secrets.token_hex(16)


def mint(kind: str, subject: str, epoch: int, nonce: str) -> str:
    return f"{kind}:{subject}.{epoch}.{nonce}"


def parse(text: str | None) -> Token:
    match = _TOKEN_RE.match(text or "")
    if match is None:
        raise errors.CpError("USAGE", "malformed --token")
    return Token(match[1], match[2], int(match[3]), match[4])


@dataclass(frozen=True)
class Principal:
    """Who a valid token speaks for."""

    token: Token
    run_id: str
    session_id: str | None
    node_id: str | None = None
    task_id: str | None = None
    node_state: str | None = None

    @property
    def kind(self) -> str:
        return self.token.kind

    @property
    def subject(self) -> str:
        return self.token.subject_ref

    @property
    def actor(self) -> str:
        return "orchestrator" if self.kind == "run" else f"node:{self.task_id}"


def _stale(detail: str, superseded_by: str | None, at: str | None) -> errors.CpError:
    return errors.CpError("STALE_TOKEN", detail, superseded_by=superseded_by, at=at, instructions=INSTRUCTIONS)


def _require_run(conn: sqlite3.Connection, tok: Token) -> Principal:
    lease = conn.execute("SELECT * FROM run_leases WHERE run_id = ?", (tok.subject,)).fetchone()
    if lease is None:
        raise errors.CpError("NOT_FOUND", f"unknown run {tok.subject}")
    if lease["epoch"] != tok.epoch or lease["nonce"] != tok.nonce:
        old = conn.execute(
            "SELECT superseded_by, superseded_at FROM run_sessions WHERE run_id = ? AND epoch = ?",
            (tok.subject, tok.epoch),
        ).fetchone()
        if old is not None and old["superseded_by"] is not None:
            raise _stale("this run token is superseded", old["superseded_by"], old["superseded_at"])
        raise _stale("this run token is stale", lease["session_id"], lease["claimed_at"])
    return Principal(tok, run_id=tok.subject, session_id=lease["session_id"])


def _require_node(conn: sqlite3.Connection, tok: Token, allow_stopped: bool) -> Principal:
    node = conn.execute("SELECT * FROM nodes WHERE id = ?", (tok.subject,)).fetchone()
    if node is None:
        raise errors.CpError("NOT_FOUND", f"unknown node {tok.subject}")
    if node["generation"] != tok.epoch or node["nonce"] != tok.nonce or node["state"] == "superseded":
        newer = conn.execute(
            "SELECT id, spawned_at FROM nodes WHERE task_id = ? AND generation > ? ORDER BY generation DESC LIMIT 1",
            (node["task_id"], node["generation"]),
        ).fetchone()
        if newer is not None:
            raise _stale("this node generation is superseded", newer["id"], newer["spawned_at"])
        raise _stale("this node token is stale", node["id"], node["updated_at"])
    if node["state"] == "stopped" and not allow_stopped:
        raise errors.CpError("STOPPED", f"node {node['id']} (task {node['task_id']}) is stopped")
    return Principal(
        tok,
        run_id=node["run_id"],
        session_id=node["session_id"],
        node_id=node["id"],
        task_id=node["task_id"],
        node_state=node["state"],
    )


def require(
    conn: sqlite3.Connection,
    token: str | None,
    *,
    kinds: Collection[str],
    task_id: str | None = None,
    allow_stopped: bool = False,
) -> Principal:
    """Validate ``token`` inside the caller's write transaction → its ``Principal``.

    ``STALE_TOKEN`` (with the self-end payload) when superseded; ``STOPPED``
    when the token's node is stopped (unless ``allow_stopped``); ``CONFLICT``
    for a token kind not accepted here, or a node token used on another task.
    """
    tok = parse(token)
    if tok.kind not in kinds:
        raise errors.CpError("CONFLICT", f"a {tok.kind} token is not accepted here (needs: {', '.join(sorted(kinds))})")
    principal = _require_run(conn, tok) if tok.kind == "run" else _require_node(conn, tok, allow_stopped)
    if task_id is not None:
        if principal.kind == "node" and principal.task_id != task_id:
            raise errors.CpError("CONFLICT", f"node token of task {principal.task_id} used on task {task_id}")
        task = conn.execute("SELECT run_id FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if task is None:
            raise errors.CpError("NOT_FOUND", f"unknown task {task_id}")
        if task["run_id"] not in (None, principal.run_id):
            raise errors.CpError("CONFLICT", f"task {task_id} belongs to run {task['run_id']}, not {principal.run_id}")
    return principal
