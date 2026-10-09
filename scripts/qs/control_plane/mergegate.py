"""The merge gate (QS-406 §4.2, D18, D19): automatic merges open only after a self-check pass.

A leaf module (D17). The state lives in ``meta``:

- ``selfcheck`` = ``{code_version, ok, at, tries, failures}`` (one record: the latest version checked);
- ``selfcheck_override`` = ``{code_version, reason, at}`` (the maintainer's ``halt clear``);
- ``selfcheck_pending`` = ``{code_version, since}`` (first seen pending by ``tool merge``).

``merge_allowed`` runs inside the caller's ``db.write``, which must commit before any refusal is
raised, so ``selfcheck_pending.since`` survives a ``BUSY``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from . import clock as clock_mod
from . import db

SELFCHECK_RETRY_S = 600.0
SELFCHECK_ESCALATE_AFTER = 3
PENDING_ESCALATE_S = 600.0

RECORD = "selfcheck"
OVERRIDE = "selfcheck_override"
PENDING = "selfcheck_pending"

OPEN, PENDING_STATE, RETRYING, FAILED, STUCK = "open", "pending", "retrying", "failed", "stuck"


@dataclass(frozen=True)
class Verdict:
    state: str  # open | pending | retrying | failed | stuck
    reason: str
    next_retry_at: str | None = None


def get(conn: sqlite3.Connection, key: str) -> dict[str, Any] | None:
    """A ``meta`` JSON object; ``None`` when absent or unreadable."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    try:
        value = json.loads(row[0])
    except TypeError, ValueError:
        return None
    return value if isinstance(value, dict) else None


def _put(conn: sqlite3.Connection, key: str, value: dict[str, Any]) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, json.dumps(value, sort_keys=True)))


def _delete(conn: sqlite3.Connection, key: str) -> None:
    conn.execute("DELETE FROM meta WHERE key = ?", (key,))


def overridden(conn: sqlite3.Connection, version: str) -> bool:
    ov = get(conn, OVERRIDE)
    return ov is not None and ov.get("code_version") == version


def passed(conn: sqlite3.Connection, version: str) -> bool:
    rec = get(conn, RECORD)
    return rec is not None and rec.get("code_version") == version and rec.get("ok") is True


def failed_record(conn: sqlite3.Connection, version: str) -> dict[str, Any] | None:
    """The ``ok: false`` record for ``version``, if that is the latest record."""
    rec = get(conn, RECORD)
    if rec is None or rec.get("code_version") != version or rec.get("ok") is not False:
        return None
    return rec


def next_retry_at(rec: dict[str, Any]) -> str | None:
    try:
        return clock_mod.iso(clock_mod.parse(str(rec.get("at"))) + timedelta(seconds=SELFCHECK_RETRY_S))
    except ValueError:
        return None


def _pending_age(clock: clock_mod.Clock, pending: dict[str, Any]) -> float | None:
    try:
        return clock_mod.age(clock, str(pending.get("since")))
    except ValueError:
        return None


def _int_or(value: Any, default: int) -> int:
    """``int(value)``; ``default`` for a missing, zero or hand-edited value (``"x"``, a list…) (H6)."""
    try:
        return int(value or default)
    except TypeError, ValueError:
        return default


def merge_allowed(conn: sqlite3.Connection, version: str, clock: clock_mod.Clock) -> Verdict:
    """The gate's verdict for the code ``version`` on disk (may write ``selfcheck_pending``)."""
    v12 = version[:12]
    if overridden(conn, version):
        return Verdict(OPEN, f"merges reopened by the maintainer for {v12}")
    if passed(conn, version):
        return Verdict(OPEN, f"self-check passed for {v12}")
    rec = failed_record(conn, version)
    if rec is not None:
        tries = _int_or(rec.get("tries"), 1)
        if tries >= SELFCHECK_ESCALATE_AFTER:
            return Verdict(FAILED, f"automatic merges are halted: self-check failed for {v12}; ask the maintainer")
        nxt = next_retry_at(rec)
        when = "retried automatically" if nxt is None else f"retried at {nxt}"  # an unreadable `at`: no time
        return Verdict(
            RETRYING,
            f"self-check failed for {v12} ({tries} of {SELFCHECK_ESCALATE_AFTER} tries); it is {when};"
            " replay the same key later",
            next_retry_at=nxt,
        )
    pending = get(conn, PENDING)
    waited = _pending_age(clock, pending) if pending is not None and pending.get("code_version") == version else None
    if waited is None:  # first seen pending (or another version's mark, or an unreadable one): start counting
        _put(conn, PENDING, {"code_version": version, "since": db.now(clock)})
        waited = 0.0
    if waited > PENDING_ESCALATE_S:
        return Verdict(
            STUCK,
            f"the self-check has not run for {v12} in {int(waited // 60)} min; the daemon may not start",
        )
    return Verdict(PENDING_STATE, f"self-check pending for {v12}; replay the same key shortly")


def record(
    conn: sqlite3.Connection, clock: clock_mod.Clock, version: str, *, ok: bool, failures: list[dict[str, Any]]
) -> dict[str, Any]:
    """Record a self-check result for ``version`` (and clear its pending mark) → the record."""
    prev = failed_record(conn, version)
    tries = 0 if ok else (_int_or(prev.get("tries"), 0) + 1 if prev is not None else 1)
    rec = {"code_version": version, "ok": ok, "at": db.now(clock), "tries": tries, "failures": failures}
    _put(conn, RECORD, rec)
    _delete(conn, PENDING)
    return rec


def override(conn: sqlite3.Connection, clock: clock_mod.Clock, version: str, reason: str) -> dict[str, Any]:
    """The maintainer reopens merges for ``version`` only (``halt clear``) → the override."""
    ov = {"code_version": version, "reason": reason, "at": db.now(clock)}
    _put(conn, OVERRIDE, ov)
    _delete(conn, PENDING)
    return ov
