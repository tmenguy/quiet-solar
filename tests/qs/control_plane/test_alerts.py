"""QS-406 T5: the alert kinds, fingerprints and the occurrence engine (§5, AC 2, AC 3)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from control_plane import activeloop, alerts, clock, db, ticks

from .conftest import open_run, run_cli, sql


def C(kind: str, subject: str, *runs: str | None, **payload: Any) -> alerts.Condition:
    return alerts.Condition(kind, subject, tuple(runs), payload)


def _msgs(path: Path) -> list[dict[str, Any]]:
    return [dict(r) for r in sql(path, "SELECT * FROM messages WHERE sender = 'cp:daemon' ORDER BY id")]


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(r) for r in sql(path, "SELECT * FROM alerts ORDER BY id")]


class TestKinds:
    def test_every_kind_fits_messages_kind(self) -> None:
        assert alerts.KINDS and all(1 <= len(k) <= 32 for k in alerts.KINDS)

    def test_cross_run_kinds_are_kinds(self) -> None:
        assert alerts.CROSS_RUN_KINDS < alerts.KINDS

    def test_rounds_alert(self) -> None:
        assert alerts.ROUNDS_ALERT == 5

    def test_fingerprint(self) -> None:
        a = alerts.fingerprint("overlap", "T3|T7", 1)
        assert a.startswith("overlap:") and a.endswith(":1") and len(a.split(":")[1]) == 16
        assert alerts.fingerprint("overlap", "T3|T7", 2) != a
        assert alerts.fingerprint("overlap", "T3|T8", 1) != a

    def test_severity(self) -> None:
        assert alerts.severity(alerts.CI_RED) == "must-fix" and alerts.severity(alerts.OVERLAP) == "alert"


class TestSync:
    def test_one_row_and_one_message_per_run(self, conn: sqlite3.Connection, migrated: Path, fake_clock) -> None:
        r1, _ = open_run("r1", "S-1")
        r2, _ = open_run("r2", "S-2")
        out = alerts.sync(conn, fake_clock, kinds={"overlap"}, active=[C("overlap", "T1|T2", r1, r2, r1, files=["a"])])
        assert len(out["raised"]) == 2
        rows = _rows(migrated)
        assert [(r["run_id"], r["kind"], r["subject"]) for r in rows] == [
            (r1, "overlap", "T1|T2"),
            (r2, "overlap", "T1|T2"),
        ]
        msgs = _msgs(migrated)
        assert [m["run_id"] for m in msgs] == [r1, r2] and all(m["recipient"] == "orchestrator" for m in msgs)
        fp = alerts.fingerprint("overlap", "T1|T2", 1)
        assert {m["dedupe_key"] for m in msgs} == {fp}
        payload = json.loads(msgs[0]["payload"])
        assert payload == {"files": ["a"], "alert_id": rows[0]["id"], "fingerprint": fp, "severity": "alert"}
        assert rows[0]["message_id"] == msgs[0]["id"] and json.loads(rows[0]["payload"]) == {"files": ["a"]}

    def test_the_next_tick_raises_nothing_and_refreshes(self, conn, migrated: Path, fake_clock) -> None:
        r1, _ = open_run()
        alerts.sync(conn, fake_clock, kinds={"overlap"}, active=[C("overlap", "s", r1, v=1)])
        fake_clock.advance(30)
        out = alerts.sync(conn, fake_clock, kinds={"overlap"}, active=[C("overlap", "s", r1, v=2)])
        assert out == {"raised": [], "refreshed": 1, "cleared": []}
        [row] = _rows(migrated)
        assert json.loads(row["payload"]) == {"v": 2} and row["last_seen"] == clock.stamp(fake_clock)
        assert len(_msgs(migrated)) == 1

    @pytest.mark.parametrize("advance", [0, 60])
    def test_clear_then_recur_is_a_new_occurrence(self, conn, migrated: Path, fake_clock, advance: float) -> None:
        r1, _ = open_run()
        alerts.sync(conn, fake_clock, kinds={"overlap"}, active=[C("overlap", "s", r1)])
        fake_clock.advance(advance)
        out = alerts.sync(conn, fake_clock, kinds={"overlap"}, active=[])
        assert len(out["cleared"]) == 1
        out = alerts.sync(conn, fake_clock, kinds={"overlap"}, active=[C("overlap", "s", r1)])
        assert out["raised"][0]["fingerprint"] == alerts.fingerprint("overlap", "s", 2)
        rows = _rows(migrated)
        assert rows[0]["cleared_at"] is not None and rows[1]["cleared_at"] is None
        assert len(_msgs(migrated)) == 2

    def test_omitted_kinds_neither_clear_nor_raise(self, conn, migrated: Path, fake_clock) -> None:
        r1, _ = open_run()
        alerts.sync(conn, fake_clock, kinds={"overlap"}, active=[C("overlap", "s", r1)])
        assert alerts.sync(conn, fake_clock, kinds={"node_stalled"}, active=[]) == {
            "raised": [],
            "refreshed": 0,
            "cleared": [],
        }
        assert _rows(migrated)[0]["cleared_at"] is None

    def test_a_condition_outside_kinds_is_a_bug(self, conn, fake_clock) -> None:
        r1, _ = open_run()
        with pytest.raises(ValueError, match="overlap"):
            alerts.sync(conn, fake_clock, kinds={"node_stalled"}, active=[C("overlap", "s", r1)])

    def test_a_daemon_restart_is_not_a_recurrence(self, conn, migrated: Path, fake_clock) -> None:
        r1, _ = open_run()
        alerts.sync(conn, fake_clock, kinds={"overlap"}, active=[C("overlap", "s", r1)])
        ticks._reset_for_tests()
        activeloop._reset_for_tests()
        out = alerts.sync(conn, fake_clock, kinds={"overlap"}, active=[C("overlap", "s", r1)])
        assert out["raised"] == [] and out["cleared"] == [] and len(_msgs(migrated)) == 1

    def test_closed_and_unknown_runs_get_nothing(self, conn, migrated: Path, fake_clock) -> None:
        r1, token = open_run()
        assert run_cli("run", "close", "--token", token)[0] == 0
        out = alerts.sync(conn, fake_clock, kinds={"overlap"}, active=[C("overlap", "s", r1, None)])
        assert out["raised"] == [] and _rows(migrated) == [] and _msgs(migrated) == []

    def test_closing_a_run_clears_its_open_rows(self, conn, migrated: Path, fake_clock) -> None:
        r1, token = open_run()
        alerts.sync(conn, fake_clock, kinds={"overlap"}, active=[C("overlap", "s", r1)])
        run_cli("run", "close", "--token", token)
        assert len(alerts.sync(conn, fake_clock, kinds={"overlap"}, active=[C("overlap", "s", r1)])["cleared"]) == 1

    def test_sync_locked_runs_in_the_callers_transaction(self, conn, migrated: Path, fake_clock) -> None:
        r1, _ = open_run()
        with pytest.raises(RuntimeError), db.write(conn):
            alerts.sync_locked(conn, fake_clock, kinds={"overlap"}, active=[C("overlap", "s", r1)])
            raise RuntimeError
        assert _rows(migrated) == [] and _msgs(migrated) == []


class TestEvent:
    def test_a_one_shot_alert(self, conn, migrated: Path, fake_clock) -> None:
        r1, _ = open_run()
        with db.write(conn):
            out = alerts.event_locked(
                conn, fake_clock, kind="queue_not_draining", subject="hook:7", run_id=r1, payload={"a": 1}
            )
        assert out is not None
        [row] = _rows(migrated)
        assert row["cleared_at"] == row["first_seen"] and row["message_id"] == out["message_id"]
        assert json.loads(_msgs(migrated)[0]["payload"])["fingerprint"] == alerts.fingerprint(
            "queue_not_draining", "hook:7", 1
        )

    def test_a_closed_or_missing_run_is_none(self, conn, fake_clock) -> None:
        r1, token = open_run()
        run_cli("run", "close", "--token", token)
        with db.write(conn):
            assert alerts.event_locked(conn, fake_clock, kind="k", subject="hook:1", run_id=r1, payload={}) is None
            assert alerts.event_locked(conn, fake_clock, kind="k", subject="hook:2", run_id=None, payload={}) is None


class TestSnapshotAlerts:
    def test_open_rows_plus_the_newest_cleared(self, conn, migrated: Path, fake_clock, monkeypatch) -> None:
        monkeypatch.setattr(alerts, "ALERTS_LIMIT", 2)
        r1, _ = open_run("r1", "S-1")
        r2, _ = open_run("r2", "S-2")
        for i in range(4):
            alerts.sync(conn, fake_clock, kinds={"overlap"}, active=[C("overlap", f"s{i}", r1)])
            alerts.sync(conn, fake_clock, kinds={"overlap"}, active=[])
        alerts.sync(conn, fake_clock, kinds={"overlap"}, active=[C("overlap", "open", r1, r2)])
        snap = run_cli("snapshot")[1]
        assert [(a["run_id"], a["subject"]) for a in snap["alerts"]] == [
            (r2, "open"),
            (r1, "open"),
            (r1, "s3"),
            (r1, "s2"),
        ]
        assert snap["alerts"][0]["payload"] == {}
        only = run_cli("snapshot", "--run", "r2")[1]
        assert [(a["run_id"], a["subject"]) for a in only["alerts"]] == [(r2, "open")]
