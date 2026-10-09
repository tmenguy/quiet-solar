"""Alerts (QS-406 §5): the kinds, fingerprints, and the occurrence engine.

An alert is raised **once per occurrence**: one ``alerts`` row per (run,
occurrence), and one orchestrator message posted with the fingerprint as its
``dedupe_key``. While the condition holds, later syncs only refresh the row;
when it stops holding the row is cleared (no message); a recurrence is a new
occurrence, with a new counter ``n`` in its fingerprint.

A hook syncs only the kinds whose input it knows this tick: a kind left out
of ``kinds`` is neither cleared nor raised. Cold in-memory state counts as
unknown, so a daemon restart is never a recurrence.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from . import clock as clock_mod
from . import db, messages

OVERLAP = "overlap"
OVERLAP_CROSS_RUN = "overlap_cross_run"
NODE_STALLED = "node_stalled"
TOO_MANY_ROUNDS = "too_many_rounds"
TASK_WITHOUT_NODE = "task_without_node"
DEPENDENCY_VIOLATED = "dependency_violated"
LOCK_HELD_LONG = "lock_held_long"
GATE_SLOT_HELD_LONG = "gate_slot_held_long"
LEFTOVER_ITEM = "leftover_item"
LEFTOVER_SCRATCH = "leftover_scratch"
DEPENDENCY_CYCLE = "dependency_cycle"
DEPENDENCY_CYCLE_CROSS_RUN = "dependency_cycle_cross_run"
DUPLICATE_TASK = "duplicate_task"
DUPLICATE_TASK_CROSS_RUN = "duplicate_task_cross_run"
CI_RED = "ci_red"
NODE_DEAD = "node_dead"
ORCHESTRATOR_DEAD = "orchestrator_dead"
ORCHESTRATOR_NOT_LISTENING = "orchestrator_not_listening"
SELFCHECK_FAILED = "selfcheck_failed"
BACKUP_FAILED = "backup_failed"
HOOK_ALERT = "hook_alert"  # a routed hook event whose own kind is unusable
ROUTED_HOOK_KINDS = frozenset({"queue_not_draining", "idle_without_wait", "merge_state_conflict"})

CROSS_RUN_KINDS = frozenset({OVERLAP_CROSS_RUN, DEPENDENCY_CYCLE_CROSS_RUN, DUPLICATE_TASK_CROSS_RUN})
KINDS = frozenset(
    {
        OVERLAP,
        NODE_STALLED,
        TOO_MANY_ROUNDS,
        TASK_WITHOUT_NODE,
        DEPENDENCY_VIOLATED,
        LOCK_HELD_LONG,
        GATE_SLOT_HELD_LONG,
        LEFTOVER_ITEM,
        LEFTOVER_SCRATCH,
        DEPENDENCY_CYCLE,
        DUPLICATE_TASK,
        CI_RED,
        NODE_DEAD,
        ORCHESTRATOR_DEAD,
        ORCHESTRATOR_NOT_LISTENING,
        SELFCHECK_FAILED,
        BACKUP_FAILED,
        HOOK_ALERT,
        *ROUTED_HOOK_KINDS,
        *CROSS_RUN_KINDS,
    }
)
MUST_FIX_KINDS = frozenset({CI_RED})

ROUNDS_ALERT = 5  # the epic; #375 may change it
ALERTS_LIMIT = 50  # the newest cleared alerts `snapshot` shows
SENDER = "cp:daemon"


def fingerprint(kind: str, subject: str, n: int) -> str:
    """``<kind>:<sha256(subject)[:16]>:<n>``; ``n`` is the occurrence counter (D6)."""
    return f"{kind}:{hashlib.sha256(subject.encode()).hexdigest()[:16]}:{n}"


def severity(kind: str) -> str:
    return "must-fix" if kind in MUST_FIX_KINDS else "alert"


@dataclass(frozen=True)
class Condition:
    """One active condition: raised to each of ``run_ids`` (``None`` and closed runs are dropped)."""

    kind: str
    subject: str
    run_ids: tuple[str | None, ...]
    payload: Mapping[str, Any]


def _open_runs(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT id FROM runs WHERE state = 'open'")}


def _raise(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    run_id: str,
    kind: str,
    subject: str,
    payload: Mapping[str, Any],
    one_shot: bool,
) -> dict[str, Any]:
    n = (
        1
        + conn.execute(
            "SELECT count(*) FROM alerts WHERE run_id = ? AND kind = ? AND subject = ?", (run_id, kind, subject)
        ).fetchone()[0]
    )
    fp = fingerprint(kind, subject, n)
    now = db.now(clock)
    cur = conn.execute(
        "INSERT INTO alerts (run_id, kind, subject, fingerprint, payload, first_seen, last_seen, cleared_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, kind, subject, fp, messages.canonical(dict(payload)), now, now, now if one_shot else None),
    )
    alert_id = cur.lastrowid
    posted = messages.post_locked(
        conn,
        clock,
        run_id=run_id,
        recipient=messages.ORCHESTRATOR,
        kind=kind,
        sender=SENDER,
        payload={**payload, "alert_id": alert_id, "fingerprint": fp, "severity": severity(kind)},
        dedupe_key=fp,
    )
    conn.execute("UPDATE alerts SET message_id = ? WHERE id = ?", (posted["id"], alert_id))
    return {
        "id": alert_id,
        "run_id": run_id,
        "kind": kind,
        "subject": subject,
        "fingerprint": fp,
        "message_id": posted["id"],
    }


def sync_locked(
    conn: sqlite3.Connection, clock: clock_mod.Clock, *, kinds: Collection[str], active: Iterable[Condition]
) -> dict[str, Any]:
    """Make the open alerts of ``kinds`` match ``active``, inside the caller's transaction.

    → ``{raised: [...], refreshed: n, cleared: [alert ids]}``. A condition whose kind is not in
    ``kinds`` is a bug (``ValueError``).
    """
    kinds = frozenset(kinds)
    open_runs = _open_runs(conn)
    now = db.now(clock)
    wanted: set[tuple[str, str, str]] = set()
    raised: list[dict[str, Any]] = []
    refreshed = 0
    for cond in active:
        if cond.kind not in kinds:
            raise ValueError(f"condition kind {cond.kind!r} is not among the synced kinds {sorted(kinds)}")
        for run_id in dict.fromkeys(cond.run_ids):
            if run_id is None or run_id not in open_runs or (run_id, cond.kind, cond.subject) in wanted:
                continue
            wanted.add((run_id, cond.kind, cond.subject))
            row = conn.execute(
                "SELECT id FROM alerts WHERE run_id = ? AND kind = ? AND subject = ? AND cleared_at IS NULL",
                (run_id, cond.kind, cond.subject),
            ).fetchone()
            if row is None:
                raised.append(
                    _raise(
                        conn,
                        clock,
                        run_id=run_id,
                        kind=cond.kind,
                        subject=cond.subject,
                        payload=cond.payload,
                        one_shot=False,
                    )
                )
            else:
                conn.execute(
                    "UPDATE alerts SET last_seen = ?, payload = ? WHERE id = ?",
                    (now, messages.canonical(dict(cond.payload)), row["id"]),
                )
                refreshed += 1
    marks = ", ".join("?" for _ in kinds)  # SQLite accepts an empty `IN ()`
    cleared = [
        row["id"]
        for row in conn.execute(
            f"SELECT id, run_id, kind, subject FROM alerts WHERE cleared_at IS NULL AND kind IN ({marks}) ORDER BY id",
            tuple(sorted(kinds)),
        ).fetchall()
        if (row["run_id"], row["kind"], row["subject"]) not in wanted
    ]
    conn.executemany("UPDATE alerts SET cleared_at = ? WHERE id = ?", [(now, i) for i in cleared])
    return {"raised": raised, "refreshed": refreshed, "cleared": cleared}


def sync(conn: sqlite3.Connection, clock: clock_mod.Clock, **kw: Any) -> dict[str, Any]:
    """``sync_locked`` in its own ``db.write``."""
    with db.write(conn):
        return sync_locked(conn, clock, **kw)


def event_locked(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    kind: str,
    subject: str,
    run_id: str | None,
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """A one-shot alert (``cleared_at = first_seen``), posted once; ``None`` for no run or a closed run."""
    if run_id is None or run_id not in _open_runs(conn):
        return None
    return _raise(conn, clock, run_id=run_id, kind=kind, subject=subject, payload=payload, one_shot=True)
