"""Hook-event routing (QS-406 §7): ``hook_events`` alerts reach their run's orchestrator queue.

A cursor in ``meta.hook_events_cursor`` (seeded by migration v3 past the existing history) marks the
last event read. Each tick, in one ``db.write``, the next events are read; every ``alert`` row whose
run is known and open becomes a one-shot alert (``alerts.event_locked``, subject ``hook:<id>``); the
cursor moves to the last id read, routable or not. Each event is routed inside its own SAVEPOINT: one
that raises is rolled back alone, logged, and counted unroutable, so it never blocks the cursor. An
event's own ``kind`` passes through unless it names a daemon-owned kind (``ci_red``,
``selfcheck_failed``…): that becomes ``hook_alert``. Unroutable rows stay visible in
``snapshot["hook_alerts"]``. ``hook_events.id`` is ``AUTOINCREMENT`` and SQLite serialises writers, so
ids commit in order and never go back.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from typing import Any

from . import alerts, db
from . import clock as clock_mod

HOOK_ROUTE = "hook_route"
CURSOR = "hook_events_cursor"
BATCH = 500
DAEMON_KINDS = alerts.KINDS - alerts.ROUTED_HOOK_KINDS  # a hook event may not pose as one of these


def _log(message: str) -> None:
    sys.stderr.write(f"[cp-hookroute] {message}\n")
    sys.stderr.flush()


def _detail(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except TypeError, ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _run_of(conn: sqlite3.Connection, detail: dict[str, Any]) -> str | None:
    if isinstance(detail.get("run_id"), str):
        return str(detail["run_id"])
    if isinstance(detail.get("task_id"), str):
        row = conn.execute("SELECT run_id FROM tasks WHERE id = ?", (detail["task_id"],)).fetchone()
        return None if row is None else row["run_id"]
    return None


def _kind(detail: dict[str, Any]) -> str:
    kind = detail.get("kind")
    if isinstance(kind, str) and 1 <= len(kind) <= 32 and kind not in DAEMON_KINDS:
        return kind
    return alerts.HOOK_ALERT


def route_locked(conn: sqlite3.Connection, clock: clock_mod.Clock) -> dict[str, Any]:
    """Route the next batch inside the caller's transaction → ``{read, routed, cursor}``."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (CURSOR,)).fetchone()
    cursor = int(row[0]) if row is not None else 0
    events = conn.execute(
        "SELECT id, hook, session_id, decision, detail FROM hook_events WHERE id > ? ORDER BY id LIMIT ?",
        (cursor, BATCH),
    ).fetchall()
    routed = 0
    for event in events:
        if event["decision"] != "alert":
            continue
        detail = _detail(event["detail"])
        conn.execute("SAVEPOINT hook_event")
        try:
            posted = alerts.event_locked(
                conn,
                clock,
                kind=_kind(detail),
                subject=f"hook:{event['id']}",
                run_id=_run_of(conn, detail),
                payload={
                    **detail,
                    "hook": event["hook"],
                    "hook_event_id": event["id"],
                    "session_id": event["session_id"],
                },
            )
        except Exception as exc:  # noqa: BLE001 — one bad event never stops the routing: skipped, logged
            conn.execute("ROLLBACK TO hook_event")
            posted = None
            _log(f"event {event['id']} not routed: {exc!r}")
        conn.execute("RELEASE hook_event")
        routed += posted is not None
    if events:
        cursor = events[-1]["id"]
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (CURSOR, str(cursor)))
    return {"read": len(events), "routed": routed, "cursor": cursor}


def hook_route_hook(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
    with db.write(conn):
        route_locked(conn, clock)
