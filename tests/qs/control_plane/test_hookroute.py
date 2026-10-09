"""QS-406 T8: hook-event routing (§7, AC 9)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from control_plane import alerts, db, hookroute

from .conftest import insert_task, open_run, run_cli, sql


def _event(path: Path, detail: Any, decision: str = "alert", hook: str = "stop") -> int:
    text = detail if isinstance(detail, str) else json.dumps(detail)
    sql(
        path,
        "INSERT INTO hook_events (hook, session_id, decision, detail, at) VALUES (?, 'S', ?, ?, 'x')",
        [hook, decision, text],
    )
    return int(sql(path, "SELECT max(id) FROM hook_events")[0][0])


def _cursor(path: Path) -> int:
    return int(sql(path, "SELECT value FROM meta WHERE key = 'hook_events_cursor'")[0][0])


def _queue(path: Path, run_id: str) -> list[tuple[str, dict[str, Any]]]:
    rows = sql(
        path, "SELECT kind, payload FROM messages WHERE run_id = ? AND sender = 'cp:daemon' ORDER BY id", [run_id]
    )
    return [(r[0], json.loads(r[1])) for r in rows]


def test_alerts_reach_the_right_queue_once(conn, migrated, fake_clock) -> None:
    r1, _ = open_run("r1", "S-1")
    r2, _ = open_run("r2", "S-2")
    insert_task(migrated, "T7", r2)
    a = _event(migrated, {"kind": "queue_not_draining", "run_id": r1})
    b = _event(migrated, {"kind": "merge_state_conflict", "task_id": "T7"}, hook="tool:merge")
    hookroute.hook_route_hook(conn, fake_clock)
    hookroute.hook_route_hook(conn, fake_clock)
    [(kind1, p1)] = _queue(migrated, r1)
    [(kind2, p2)] = _queue(migrated, r2)
    assert (kind1, kind2) == ("queue_not_draining", "merge_state_conflict")
    assert p1["hook_event_id"] == a and p2["hook_event_id"] == b and p2["hook"] == "tool:merge"
    rows = sql(migrated, "SELECT subject, cleared_at IS NOT NULL FROM alerts ORDER BY id")
    assert [tuple(r) for r in rows] == [(f"hook:{a}", 1), (f"hook:{b}", 1)]
    assert _cursor(migrated) == b


@pytest.mark.parametrize(
    ("kind", "expected"), [("k" * 20, "k" * 20), ("k" * 33, alerts.HOOK_ALERT), (7, alerts.HOOK_ALERT)]
)
def test_kinds_pass_through_or_fall_back(conn, migrated, fake_clock, kind: Any, expected: str) -> None:
    r1, _ = open_run()
    _event(migrated, {"kind": kind, "run_id": r1})
    hookroute.hook_route_hook(conn, fake_clock)
    assert [k for k, _ in _queue(migrated, r1)] == [expected]


def test_unroutable_rows_move_the_cursor(conn, migrated, fake_clock) -> None:
    r1, token = open_run()
    run_cli("run", "close", "--token", token)
    ids = [
        _event(migrated, {"kind": "idle_without_wait", "run_id": r1}),  # a closed run
        _event(migrated, {"kind": "k", "task_id": "T404"}),  # an unknown task
        _event(migrated, {"kind": "k"}),  # no run at all
        _event(migrated, "not json"),
        _event(migrated, "[1, 2]"),
        _event(migrated, {"kind": "k", "run_id": r1}, decision="deny"),  # not an alert
    ]
    with db.write(conn):
        out = hookroute.route_locked(conn, fake_clock)
    assert out == {"read": 6, "routed": 0, "cursor": ids[-1]}
    assert sql(migrated, "SELECT count(*) FROM alerts")[0][0] == 0
    assert run_cli("snapshot")[1]["hook_alerts"]  # still visible there


def test_the_cursor_persists_and_history_is_not_replayed(conn, migrated, fake_clock) -> None:
    r1, _ = open_run()
    old = _event(migrated, {"kind": "k", "run_id": r1})
    sql(migrated, "UPDATE meta SET value = ? WHERE key = 'hook_events_cursor'", [str(old)])  # as migration v3 seeds it
    hookroute.hook_route_hook(conn, fake_clock)  # a new daemon: reads from the stored cursor
    assert _queue(migrated, r1) == []
    new = _event(migrated, {"kind": "k", "run_id": r1})
    hookroute.hook_route_hook(conn, fake_clock)
    assert [p["hook_event_id"] for _, p in _queue(migrated, r1)] == [new]


def test_a_missing_cursor_starts_at_zero(conn, migrated, fake_clock) -> None:
    r1, _ = open_run()
    first = _event(migrated, {"kind": "k", "run_id": r1})
    sql(migrated, "DELETE FROM meta WHERE key = 'hook_events_cursor'")
    hookroute.hook_route_hook(conn, fake_clock)
    assert _cursor(migrated) == first and len(_queue(migrated, r1)) == 1


def test_batches_are_bounded(conn, migrated, fake_clock, monkeypatch) -> None:
    monkeypatch.setattr(hookroute, "BATCH", 2)
    for _ in range(3):
        _event(migrated, {"kind": "k"})
    hookroute.hook_route_hook(conn, fake_clock)
    assert _cursor(migrated) == 2
    hookroute.hook_route_hook(conn, fake_clock)
    assert _cursor(migrated) == 3
