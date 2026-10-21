"""Node reports and per-task digests (§7.2), on an exact contract (#424 D4/D5).

A report is a ``reports`` row **and** a ``report`` message to the
orchestrator, in one transaction. Its phase is one of ``PHASES``, its status
one of ``STATUSES``, and its ``fields`` a closed JSON object (``FIELDS``).

A digest is the task's short memory: a JSON object with exactly the
``DIGEST_KEYS``, each a string (free Markdown; ``""`` when there is nothing to
say). It is replaced, never appended, stored as its canonical JSON and capped at
``DIGEST_MAX_BYTES``. Nothing here parses text: every check is of type and shape
only, and every refusal comes before the transaction (no row, no message).
Shape is ``USAGE``, size is ``CONFLICT``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from . import clock as clock_mod
from . import db, errors, messages, tasks, tokens
from .schema_ledger import PHASES

DIGEST_MAX_BYTES = 16 * 1024
STATUSES = ("converged", "continuing", "blocked")

FIELDS = ("blocking", "questions", "discovered")  # every key optional; `{}` is valid
DISCOVERED_KEYS = ("title", "why", "shrinks")  # title, why required; shrinks optional
FIELDS_MAX_BYTES = DIGEST_MAX_BYTES  # the canonical JSON's UTF-8 size
QUESTION_ID_RE = re.compile(r"Q-[1-9][0-9]*")  # `question open`'s ids, fullmatch
SQL_INT_MAX = 2**63 - 1  # a finding id is a SQLite INTEGER

DIGEST_KEYS = ("goal", "why", "plan", "done", "problems", "decisions", "questions", "discovered")


def _usage(detail: str) -> errors.CpError:
    return errors.CpError("USAGE", detail)


def _canonical_bytes(value: Any, what: str) -> bytes:
    """``value``'s canonical JSON in UTF-8; a lone surrogate (``json.loads`` accepts ``\\ud800``) is ``USAGE``."""
    try:
        return messages.canonical(value).encode()
    except UnicodeEncodeError as exc:
        raise _usage(f"{what} is not valid UTF-8 text (a lone surrogate)") from exc


def _list(fields: dict[str, Any], key: str) -> list[Any]:
    value = fields[key]
    if not isinstance(value, list):
        raise _usage(f"fields.{key} must be a list")
    return value


def _check_discovered(i: int, item: Any) -> None:
    where = f"fields.discovered[{i}]"
    if not isinstance(item, dict):
        raise _usage(f"{where} must be an object")
    unknown = sorted(set(item) - set(DISCOVERED_KEYS))
    if unknown:
        raise _usage(f"{where}: unknown key {unknown[0]!r}; the keys are {', '.join(DISCOVERED_KEYS)}")
    for key in ("title", "why"):
        if key not in item:
            raise _usage(f"{where}.{key} is required")
    for key in ("title", "why"):
        if not isinstance(item[key], str) or not item[key].strip():
            raise _usage(f"{where}.{key} must be a non-empty string")
    if "shrinks" in item and not isinstance(item["shrinks"], bool):
        raise _usage(f"{where}.shrinks must be true or false")


def validate_fields(fields: Any) -> None:
    """``CpError`` naming the first violation of the ``fields`` schema: ``USAGE`` for its shape, then ``CONFLICT``
    for its canonical size over ``FIELDS_MAX_BYTES``. Order: an object; no unknown key; each key in ``FIELDS``
    order (its type, then its items in order)."""
    if not isinstance(fields, dict):
        raise _usage("--fields-file must hold a JSON object")
    unknown = sorted(set(fields) - set(FIELDS))
    if unknown:
        raise _usage(f"fields: unknown key {unknown[0]!r}; the keys are {', '.join(FIELDS)}")
    if "blocking" in fields:
        for i, item in enumerate(_list(fields, "blocking")):
            if not isinstance(item, int) or isinstance(item, bool) or not 0 < item <= SQL_INT_MAX:
                raise _usage(f"fields.blocking[{i}] must be a positive integer (a finding id), not {item!r}")
    if "questions" in fields:
        for i, item in enumerate(_list(fields, "questions")):
            if not isinstance(item, str) or QUESTION_ID_RE.fullmatch(item) is None:
                raise _usage(f"fields.questions[{i}] must be a question id `Q-<n>`, not {item!r}")
    if "discovered" in fields:
        for i, item in enumerate(_list(fields, "discovered")):
            _check_discovered(i, item)
    size = len(_canonical_bytes(fields, "the fields"))
    if size > FIELDS_MAX_BYTES:
        raise errors.CpError("CONFLICT", f"the fields are {size} bytes; the cap is {FIELDS_MAX_BYTES}")


def validate_digest(digest: Any) -> None:
    """``USAGE`` unless ``digest`` is an object with exactly the ``DIGEST_KEYS``, each a string (any key order)."""
    if not isinstance(digest, dict):
        raise _usage(f"the digest must be a JSON object with the keys {', '.join(DIGEST_KEYS)}")
    unknown = sorted(set(digest) - set(DIGEST_KEYS))
    if unknown:
        raise _usage(f"digest: unknown key {unknown[0]!r}; the keys are {', '.join(DIGEST_KEYS)}")
    missing = [k for k in DIGEST_KEYS if k not in digest]
    if missing:
        raise _usage(f"digest: missing key {missing[0]!r} (an empty string says there is nothing to say)")
    for key in DIGEST_KEYS:
        if not isinstance(digest[key], str):
            raise _usage(f"digest.{key} must be a string")


def parse_digest(body: str) -> dict[str, str] | None:
    """A stored body as the digest object, or ``None`` when it is not one (a text digest put before #424).

    The one rule ``task show`` and the export share for reading a stored body.
    """
    try:
        digest = json.loads(body)
        validate_digest(digest)
    except ValueError, RecursionError, errors.CpError:
        return None
    parsed: dict[str, str] = digest
    return parsed


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
    if phase not in PHASES:  # exact: the convergence check compares `phase = 'build'`
        raise errors.CpError("USAGE", f"--phase must be one of {', '.join(PHASES)}")
    if status not in STATUSES:
        raise errors.CpError("USAGE", f"--status must be one of {', '.join(STATUSES)}")
    validate_fields(fields)
    _canonical_bytes(summary, "--summary")  # a lone surrogate (argv's `surrogateescape`) is USAGE, not INTERNAL
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
    digest: Any,
    max_bytes: int = DIGEST_MAX_BYTES,
) -> dict[str, Any]:
    validate_digest(digest)
    raw = _canonical_bytes(digest, "the digest")
    size, body = len(raw), raw.decode()
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
