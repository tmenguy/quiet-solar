"""QS-406 T12: periodic backups, rotation, ``backup_failed``, ``daemon.stop`` and the restore (§10, AC 15, AC 16)."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from control_plane import activeloop, alerts, backups, ciwatch, clock, daemon, db, migrations, paths, restore, ticks

from .conftest import CUR, NEXT, ORCH, agent, insert_node, insert_task, open_run, run_cli, sql


def _hook(conn: sqlite3.Connection, fake_clock: clock.FakeClock) -> None:
    restore.backup_hook(conn, fake_clock)


def _periodic(path: Path) -> list[Path]:
    return sorted(backups.db_dir(path).glob("harness_state.periodic.*.db"))


def _meta(path: Path, key: str) -> Any:
    rows = sql(path, "SELECT value FROM meta WHERE key = ?", [key])
    return json.loads(rows[0][0]) if rows else None


def _dump(path: Path) -> list[str]:
    c = sqlite3.connect(path)
    try:
        return list(c.iterdump())
    finally:
        c.close()


def _staged(path: Path) -> list[Path]:
    return sorted(backups.db_dir(path).glob("harness_state.restoring.*"))


def _failed_alerts(path: Path) -> list[str]:
    return [r[0] for r in sql(path, "SELECT run_id FROM alerts WHERE kind = 'backup_failed' AND cleared_at IS NULL")]


# --------------------------------------------------------------------------- periodic backups (AC 15)


class TestPeriodic:
    def test_every_15_min_while_a_run_is_open(self, conn, migrated: Path, fake_clock) -> None:
        _hook(conn, fake_clock)
        assert not backups.db_dir(migrated).exists() or _periodic(migrated) == []  # no open run: no backup
        open_run()
        _hook(conn, fake_clock)
        [first] = _periodic(migrated)
        assert first.stat().st_mode & 0o777 == 0o600 and first.parent.stat().st_mode & 0o777 == 0o700
        assert first.name == f"harness_state.periodic.{fake_clock.now().strftime(backups.STAMP_FORMAT)}.db"
        fake_clock.advance(backups.BACKUP_EVERY_S - 1)
        _hook(conn, fake_clock)
        assert len(_periodic(migrated)) == 1
        fake_clock.advance(1)
        _hook(conn, fake_clock)
        assert len(_periodic(migrated)) == 2 and _meta(migrated, restore.LAST_AT) == db.now(fake_clock)

    def test_a_failure_alerts_every_run_waits_and_a_success_clears(
        self, conn, migrated, fake_clock, monkeypatch, capsys
    ) -> None:
        r1, _ = open_run("r1", "S-1")
        r2, _ = open_run("r2", "S-2")
        real = backups.take

        def full(*a: Any, **k: Any) -> Path:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(backups, "take", full)
        _hook(conn, fake_clock)
        assert sorted(_failed_alerts(migrated)) == sorted([r1, r2])
        assert _meta(migrated, restore.LAST_ERROR)["error"].startswith("OSError")
        fake_clock.advance(backups.BACKUP_EVERY_S - 1)
        _hook(conn, fake_clock)  # not retried before BACKUP_EVERY_S, logged once
        assert capsys.readouterr().err.count("backup failed") == 1
        ticks._reset_for_tests()
        activeloop._reset_for_tests()  # a restart: the alert comes from meta, not memory
        _hook(conn, fake_clock)
        assert len(_failed_alerts(migrated)) == 2
        monkeypatch.setattr(backups, "take", real)
        fake_clock.advance(1)
        _hook(conn, fake_clock)
        assert _failed_alerts(migrated) == [] and _meta(migrated, restore.LAST_ERROR) is None
        assert len(_periodic(migrated)) == 1

    def test_an_unreadable_stamp_is_old(self, conn, migrated, fake_clock) -> None:
        open_run()
        sql(migrated, "INSERT INTO meta (key, value) VALUES ('last_backup_at', 'garbage')")
        _hook(conn, fake_clock)
        assert len(_periodic(migrated)) == 1

    def test_the_rename_is_made_durable(self, conn, migrated, fake_clock, monkeypatch) -> None:
        synced: list[str] = []
        real_open, real_fsync = os.open, os.fsync
        fds: dict[int, str] = {}

        def spy_open(path: Any, flags: int, *a: Any) -> int:
            fd = real_open(path, flags, *a)
            fds[fd] = str(path)
            return fd

        def spy_fsync(fd: int) -> None:
            synced.append(fds.get(fd, "?"))
            real_fsync(fd)

        monkeypatch.setattr(backups.os, "open", spy_open)
        monkeypatch.setattr(backups.os, "fsync", spy_fsync)
        dest = backups.take(conn, fake_clock.now(), migrated)
        assert synced == [f"{dest}.partial", str(dest.parent)]

    def test_a_copy_failing_midway_leaves_nothing(self, conn, migrated, fake_clock, monkeypatch) -> None:
        open_run()
        monkeypatch.setattr(backups, "_journal_delete", lambda copy: "wal")
        _hook(conn, fake_clock)
        assert list(backups.db_dir(migrated).iterdir()) == [] and len(_failed_alerts(migrated)) == 1


class TestRotation:
    def _touch(self, d: Path, kind: str, at: datetime) -> Path:
        p = d / f"harness_state.{kind}.{at.strftime(backups.STAMP_FORMAT)}.db"
        p.write_text("")
        return p

    def test_24h_then_one_per_day_for_7_days(self, tmp_path: Path) -> None:
        now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
        recent = [self._touch(tmp_path, "periodic", now - timedelta(hours=h)) for h in (1, 5, 23)]
        day2 = [
            self._touch(tmp_path, "periodic", now - timedelta(hours=h)) for h in (30, 33)
        ]  # 2026-10-08 06:00 / 03:00
        day5 = self._touch(tmp_path, "periodic", now - timedelta(days=4, hours=1))
        old = self._touch(tmp_path, "periodic", now - timedelta(days=8))
        keep = [
            self._touch(tmp_path, "v1", now - timedelta(days=30)),
            tmp_path / "harness_state.replaced.20260101T000000000000Z.db",
            tmp_path / "harness_state.periodic.20261009T115500000000Z.db.partial",  # a copy being written
        ]
        keep[1].write_text("")
        keep[2].write_text("")
        deleted = backups.rotate(tmp_path, now)
        # 2026-10-08's newest copy is the 23 h one, kept with the last 24 h: day2's two older copies go (F15)
        assert sorted(deleted) == sorted([*day2, old])
        assert all(p.exists() for p in [*recent, day5, *keep])

    def test_the_24h_boundary_day_keeps_no_extra_copy(self, tmp_path: Path) -> None:
        now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
        inside = self._touch(tmp_path, "periodic", now - timedelta(hours=20))  # 2026-10-08 16:00, < 24 h
        outside = self._touch(tmp_path, "periodic", now - timedelta(hours=26))  # 2026-10-08 10:00, > 24 h
        assert backups.rotate(tmp_path, now) == [outside] and inside.exists()

    def test_leftover_partials_are_removed(self, tmp_path: Path) -> None:
        now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
        stale = tmp_path / f"harness_state.v3.{(now - timedelta(hours=1)).strftime(backups.STAMP_FORMAT)}.db.partial"
        fresh = (
            tmp_path
            / f"harness_state.periodic.{(now - timedelta(minutes=5)).strftime(backups.STAMP_FORMAT)}.db.partial"
        )
        odd = tmp_path / "harness_state.periodic.garbage.db.partial"
        for f in (stale, fresh, odd):
            f.write_text("")
        assert backups.rotate(tmp_path, now) == [stale]
        assert fresh.exists() and odd.exists() and not stale.exists()

    def test_a_crashed_restores_staged_copy_is_removed(self, tmp_path: Path) -> None:
        now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
        stamp = (now - timedelta(hours=1)).strftime(backups.STAMP_FORMAT)
        staged = tmp_path / f"harness_state.restoring.{stamp}.db.partial"
        staged.write_text("")
        assert backups.rotate(tmp_path, now) == [staged]

    def test_an_impossible_partial_stamp_never_stops_the_pruning(self, tmp_path: Path) -> None:
        now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
        odd = tmp_path / "harness_state.periodic.20261399T000000000000Z.db.partial"  # month 13
        odd.write_text("")
        old = self._touch(tmp_path, "periodic", now - timedelta(days=8))
        assert backups.rotate(tmp_path, now) == [old] and odd.exists()

    def test_an_impossible_backup_stamp_is_ignored(self, tmp_path: Path) -> None:
        now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
        odd = tmp_path / "harness_state.periodic.20261399T000000000000Z.db"
        odd.write_text("")
        old = self._touch(tmp_path, "periodic", now - timedelta(days=8))
        assert backups.stamp_of(odd.name) is None and backups.rotate(tmp_path, now) == [old]
        assert backups.choose_source(tmp_path) is None and odd.exists()

    def test_choose_source_newest_periodic_or_v_never_replaced_or_partial(self, tmp_path: Path) -> None:
        assert backups.choose_source(tmp_path) is None
        now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
        self._touch(tmp_path, "periodic", now - timedelta(hours=2))
        v = self._touch(tmp_path, "v1", now - timedelta(hours=1))
        (tmp_path / f"harness_state.replaced.{now.strftime(backups.STAMP_FORMAT)}.db").write_text("")
        (tmp_path / f"harness_state.periodic.{now.strftime(backups.STAMP_FORMAT)}.db.partial").write_text("")
        assert backups.choose_source(tmp_path) == v


# --------------------------------------------------------------------------- daemon.stop


class TestStop:
    def test_not_running(self, migrated, fake_clock, fake_probe, fake_kill) -> None:
        assert daemon.stop(migrated, clock=fake_clock, probe=fake_probe, kill=fake_kill) == {"status": "not_running"}
        sql(migrated, "INSERT INTO daemon_lease (id, pid, schema_version, started_at) VALUES (1, NULL, ?, 'x')", [CUR])
        assert daemon.stop(migrated, clock=fake_clock, probe=fake_probe, kill=fake_kill) == {"status": "not_running"}

    def _lease(self, path: Path, fake_clock: clock.FakeClock) -> None:
        sql(
            path,
            "INSERT OR REPLACE INTO daemon_lease (id, pid, pid_start, schema_version, started_at, heartbeat_at)"
            " VALUES (1, 7, 'start-7', ?, 'x', ?)",
            [CUR, clock.stamp(fake_clock)],
        )

    def test_stopped_when_the_singleton_is_free(self, migrated, fake_clock, fake_probe, fake_kill) -> None:
        self._lease(migrated, fake_clock)
        assert daemon.stop(migrated, clock=fake_clock, probe=fake_probe, kill=fake_kill) == {"status": "stopped"}

    def test_signalled_while_it_finishes_its_tick(self, migrated, fake_clock, fake_probe, fake_kill) -> None:
        self._lease(migrated, fake_clock)
        with db.file_lock(paths.sidecar(migrated, ".daemon.lock"), exclusive=True, timeout=0):
            out = daemon.stop(migrated, clock=fake_clock, probe=fake_probe, kill=fake_kill, restart_wait_s=1)
        assert out == {"status": "signalled", "pid": 7} and fake_kill.calls == [(7, 15)]


# --------------------------------------------------------------------------- restore (AC 16)


@pytest.fixture
def backed_up(conn: sqlite3.Connection, migrated: Path, fake_clock, fake_popen) -> dict[str, Any]:
    """A live DB with one run, a periodic backup of it, then later changes that the restore discards."""
    run_id, token = open_run()
    fake_popen.calls.clear()
    sql(
        migrated,
        "INSERT INTO daemon_lease (id, pid, pid_start, schema_version, started_at, heartbeat_at) VALUES (1, 77, 'start-77', ?, 'x', ?)",
        [CUR, clock.stamp(fake_clock, plus=-3600)],
    )
    src = backups.take(conn, fake_clock.now(), migrated)
    fake_clock.advance(600)
    sql(migrated, "INSERT INTO meta (key, value) VALUES ('later', '1')")
    sql(
        migrated,
        "INSERT INTO waiters (id, run_id, pid, pid_start, started_at, heartbeat_at) VALUES (42, ?, 9, 's', 'x', 'x')",
        [run_id],
    )
    sql(migrated, "INSERT OR REPLACE INTO meta (key, value) VALUES ('selfcheck_override', '{\"code_version\": \"v\"}')")
    return {"run": run_id, "token": token, "src": src}


class TestRestore:
    def test_restores_in_place_and_reports(self, migrated, backed_up, fake_popen, fake_clock) -> None:
        before = backed_up["src"].read_bytes()
        code, out = run_cli("restore", "--confirm")
        assert code == 0, out
        assert (
            out["restored_from"] == str(backed_up["src"])
            and "is lost" in out["message"]
            and "run claim" in out["message"]
        )
        assert out["backup_at"] == clock.iso(fake_clock.now() - timedelta(seconds=600))
        replaced = Path(out["replaced"])
        assert replaced.name.startswith("harness_state.replaced.") and replaced.parent == backups.db_dir(migrated)
        assert sql(replaced, "SELECT value FROM meta WHERE key = 'later'")  # the kept copy has the lost data
        assert sql(migrated, "SELECT value FROM meta WHERE key = 'later'") == []
        assert [tuple(r) for r in sql(migrated, "SELECT id, pid FROM waiters")] == [(42, 9)]
        assert _meta(migrated, "selfcheck_override") == {"code_version": "v"}
        assert [tuple(r) for r in sql(migrated, "SELECT pid, heartbeat_at FROM daemon_lease")] == [(None, None)]
        assert backed_up["src"].read_bytes() == before  # opened read-only
        assert len(fake_popen.calls) == 1 and not daemon.marker_path(migrated).exists()

    def test_a_waiter_of_a_run_newer_than_the_backup_is_carried(self, migrated, backed_up) -> None:
        # `waiters.run_id` has no foreign key (schema_v1): the carried row never fails the fix-up (G9)
        sql(
            migrated,
            "INSERT INTO waiters (id, run_id, pid, pid_start, started_at, heartbeat_at)"
            " VALUES (43, 'R-new', 8, 's', 'x', 'x')",
        )
        assert run_cli("restore", "--confirm")[0] == 0
        assert [tuple(r) for r in sql(migrated, "SELECT id, run_id FROM waiters ORDER BY id")] == [
            (42, backed_up["run"]),
            (43, "R-new"),
        ]
        assert sql(migrated, "SELECT count(*) FROM runs WHERE id = 'R-new'")[0][0] == 0

    def test_never_a_replaced_file(self, migrated, backed_up, fake_clock) -> None:
        assert run_cli("restore", "--confirm")[0] == 0
        assert run_cli("restore", "--confirm")[1]["restored_from"] == str(backed_up["src"])

    def test_the_next_ticks_resync(self, conn, migrated, backed_up, fake_github, fake_claude, fake_clock) -> None:
        insert_task(migrated, "T1", backed_up["run"], is_deliverable=1, pr_number=7)
        assert run_cli("restore", "--confirm")[0] == 0  # the backup predates T1
        insert_task(migrated, "T9", backed_up["run"], is_deliverable=1, pr_number=9, state="building")
        insert_node(migrated, "N1", backed_up["run"], "T9", state="running", session_id="S-n")
        fake_github.prs_by_number[9] = ciwatch.PrCi("OPEN", "h9", "SUCCESS", (), False)
        fake_claude.listing = [agent(ORCH)]
        c = db.connect(migrated)
        try:
            ciwatch.ci_watch_hook(c, fake_clock)
            from control_plane import watchdog

            watchdog.liveness_watchdog_hook(c, fake_clock)
        finally:
            c.close()
        assert tuple(sql(migrated, "SELECT ci_state, ci_sha FROM tasks WHERE id = 'T9'")[0]) == ("green", "h9")
        assert sql(migrated, "SELECT state FROM nodes WHERE id = 'N1'")[0][0] == "reaped"

    def test_a_v1_backup_is_restored_as_v1_then_migrated(self, migrated, fake_clock, monkeypatch) -> None:
        old = migrated.with_name("old.db")
        with monkeypatch.context() as m:
            m.setattr(migrations, "MIGRATIONS", migrations.MIGRATIONS[:1])
            migrations.migrate(old, role="test")
        sql(old, "INSERT INTO hook_events (hook, decision, detail, at) VALUES ('h', 'alert', '{}', 'x')")
        c = sqlite3.connect(old)
        dest = backups.db_dir(migrated) / "harness_state.v1.20261003T110000000000Z.db"
        backups.write_copy(c, dest)
        c.close()
        assert run_cli("restore", "--confirm")[0] == 0
        assert db.probe_version(migrated) == 1
        migrations.migrate(migrated, role="test")  # what the daemon `ensure` started does
        assert sql(migrated, "SELECT value FROM meta WHERE key = 'hook_events_cursor'")[0][0] == "1"
        assert sql(migrated, "SELECT count(*) FROM alerts")[0][0] == 0

    def test_a_missing_live_db_is_restored(self, conn, migrated, backed_up, fake_clock) -> None:
        conn.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(migrated) + suffix).unlink(missing_ok=True)
        code, out = run_cli("restore", "--confirm")
        assert code == 0 and out["replaced"] is None
        assert sql(migrated, "SELECT id FROM runs")[0][0] == backed_up["run"]

    def test_a_never_migrated_live_file_is_reused(self, migrated, backed_up, conn) -> None:
        conn.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(migrated) + suffix).unlink(missing_ok=True)
        sqlite3.connect(migrated).close()  # an empty file: user_version 0
        assert run_cli("restore", "--confirm")[1]["replaced"] is None

    def test_a_random_bytes_live_db_is_a_conflict(self, conn, migrated, backed_up, fake_popen) -> None:
        conn.close()
        for suffix in ("-wal", "-shm"):
            Path(str(migrated) + suffix).unlink(missing_ok=True)
        migrated.write_bytes(os.urandom(8192))
        code, out = run_cli("restore", "--confirm")
        assert out["error"] == "CONFLICT" and "move it aside" in out["hint"] and fake_popen.calls == []

    @pytest.mark.parametrize(
        "case", ["newer_live", "no_backup", "corrupt_backup", "newer_backup", "page_size", "live_call"]
    )
    def test_step_1_refusals_write_no_marker_and_call_no_ensure(
        self, migrated, backed_up, fake_popen, case: str
    ) -> None:
        expected = {
            "newer_live": "SCHEMA_TOO_NEW",
            "no_backup": "NOT_FOUND",
            "live_call": "BUSY",
        }.get(case, "CONFLICT")
        src = backed_up["src"]
        if case == "newer_live":
            sql(migrated, f"PRAGMA user_version = {NEXT}")
        elif case == "no_backup":
            src.unlink()
        elif case == "corrupt_backup":
            src.write_bytes(os.urandom(4096))
        elif case == "newer_backup":
            c = sqlite3.connect(src)
            c.execute(f"PRAGMA user_version = {NEXT}")
            c.close()
        elif case == "page_size":
            src.unlink()
            other = sqlite3.connect(":memory:")
            other.execute("PRAGMA page_size = 8192")
            for table in ("meta (key TEXT PRIMARY KEY, value TEXT)", "waiters (id)", "daemon_lease (id)"):
                other.execute(f"CREATE TABLE {table}")
            dest = backups.db_dir(migrated) / "harness_state.v1.20261003T110000000000Z.db"
            disk = sqlite3.connect(dest)
            disk.execute("PRAGMA page_size = 8192")
            other.backup(disk)
            disk.close()
        else:
            sql(
                migrated,
                "INSERT INTO tool_calls (tool, key, args_hash, actor, args, state, holder_pid, started_at)"
                " VALUES ('push', 'k', 'h', 'a', '{}', 'started', 5, 'x')",
            )
        code, out = run_cli("restore", "--confirm")
        assert out["error"] == expected, out
        assert fake_popen.calls == [] and not daemon.marker_path(migrated).exists()

    def test_no_confirm_is_usage(self, migrated, backed_up, fake_popen, fake_kill) -> None:
        assert run_cli("restore")[1]["error"] == "USAGE"
        assert fake_popen.calls == [] and fake_kill.calls == [] and not daemon.marker_path(migrated).exists()

    def test_a_held_lock_is_busy_and_cleans_up(
        self, migrated, backed_up, fake_clock, fake_probe, fake_kill, fake_popen
    ) -> None:
        with db.file_lock(paths.sidecar(migrated, ".migrate.lock"), exclusive=True, timeout=0):
            with pytest.raises(restore.errors.CpError) as exc:
                restore.restore(
                    migrated, clock=fake_clock, probe=fake_probe, kill=fake_kill, popen=fake_popen, lock_wait_s=0
                )
        assert exc.value.code == "BUSY" and "finishing a tick" in exc.value.detail
        assert not daemon.marker_path(migrated).exists() and len(fake_popen.calls) == 1

    def test_a_held_daemon_flock_is_busy(
        self, migrated, backed_up, fake_clock, fake_probe, fake_kill, fake_popen, monkeypatch
    ) -> None:
        monkeypatch.setattr(daemon, "stop", lambda *a, **k: {"status": "signalled"})
        with db.file_lock(paths.sidecar(migrated, ".daemon.lock"), exclusive=True, timeout=0):
            with pytest.raises(restore.errors.CpError) as exc:
                restore.restore(
                    migrated, clock=fake_clock, probe=fake_probe, kill=fake_kill, popen=fake_popen, lock_wait_s=0
                )
        assert exc.value.code == "BUSY" and not daemon.marker_path(migrated).exists()

    def test_a_live_call_found_after_the_stop_is_busy(
        self, migrated, backed_up, fake_clock, fake_probe, fake_kill, fake_popen, monkeypatch
    ) -> None:
        def stop_then_a_call_starts(*a: Any, **k: Any) -> dict[str, Any]:
            sql(
                migrated,
                "INSERT INTO tool_calls (tool, key, args_hash, actor, args, state, holder_pid, started_at)"
                " VALUES ('push', 'k', 'h', 'a', '{}', 'started', 5, 'x')",
            )
            return {"status": "stopped"}

        monkeypatch.setattr(daemon, "stop", stop_then_a_call_starts)
        with pytest.raises(restore.errors.CpError) as exc:
            restore.restore(
                migrated, clock=fake_clock, probe=fake_probe, kill=fake_kill, popen=fake_popen, lock_wait_s=0
            )
        assert exc.value.code == "BUSY" and not daemon.marker_path(migrated).exists() and len(fake_popen.calls) == 1

    def test_a_signalled_daemon_is_waited_for(
        self, migrated, backed_up, fake_clock, fake_probe, fake_kill, fake_popen, monkeypatch
    ) -> None:
        monkeypatch.setattr(daemon, "stop", lambda *a, **k: {"status": "signalled"})
        held = threading.Event()

        def finishing_tick() -> None:
            with db.file_lock(paths.sidecar(migrated, ".daemon.lock"), exclusive=True, timeout=0):
                held.set()
                time.sleep(0.3)

        t = threading.Thread(target=finishing_tick)
        t.start()
        held.wait(2)
        out = restore.restore(
            migrated, clock=fake_clock, probe=fake_probe, kill=fake_kill, popen=fake_popen, lock_wait_s=5
        )
        t.join()
        assert out["restored_from"] == str(backed_up["src"])

    def test_a_failing_ensure_never_hides_the_result(
        self, migrated, backed_up, fake_clock, fake_probe, fake_kill, monkeypatch, capsys
    ) -> None:
        def broken(*a: Any, **k: Any) -> Any:
            raise OSError("no fork")

        out = restore.restore(migrated, clock=fake_clock, probe=fake_probe, kill=fake_kill, popen=broken, lock_wait_s=0)
        assert out["replaced"] and "ensure after the restore failed" in capsys.readouterr().err

    def test_a_database_error_while_restoring_is_a_conflict(
        self, migrated, backed_up, fake_clock, fake_probe, fake_kill, fake_popen, monkeypatch
    ) -> None:
        def broken(*a: Any, **k: Any) -> Path:
            raise sqlite3.DatabaseError("disk image is malformed")

        monkeypatch.setattr(backups, "write_copy", broken)
        with pytest.raises(restore.errors.CpError) as exc:
            restore.restore(
                migrated, clock=fake_clock, probe=fake_probe, kill=fake_kill, popen=fake_popen, lock_wait_s=0
            )
        assert exc.value.code == "CONFLICT"

    @pytest.mark.parametrize("failure", ["malformed_row", "busy_begin"])
    def test_a_failed_fix_up_leaves_the_live_db_untouched(
        self, migrated, backed_up, fake_clock, fake_probe, fake_kill, fake_popen, monkeypatch, failure: str
    ) -> None:
        if failure == "malformed_row":
            monkeypatch.setattr(restore, "_carry", lambda conn: ([(1,)], {}))  # the insert fails
        else:
            real_begin = db.begin

            def begin(c: sqlite3.Connection, statement: str) -> None:
                if restore._db_path(c).name.endswith(".partial"):  # the staged copy only
                    raise restore.errors.CpError("BUSY", "database busy: database is locked")
                real_begin(c, statement)

            monkeypatch.setattr(restore.db, "begin", begin)
        before = _dump(migrated)
        with pytest.raises(restore.errors.CpError) as exc:
            restore.restore(
                migrated, clock=fake_clock, probe=fake_probe, kill=fake_kill, popen=fake_popen, lock_wait_s=0
            )
        err = exc.value
        assert err.code == "INTERNAL" and "not touched" in err.detail
        replaced = Path(err.extra["replaced"])
        assert replaced.name.startswith("harness_state.replaced.") and replaced.exists()
        assert str(replaced) in err.extra["hint"]
        assert _dump(migrated) == before and _staged(migrated) == []
        assert not daemon.marker_path(migrated).exists() and len(fake_popen.calls) == 1

    def test_a_failed_final_copy_names_the_replaced_copy(
        self, migrated, backed_up, fake_clock, fake_probe, fake_kill, fake_popen, monkeypatch
    ) -> None:
        def broken(stage: sqlite3.Connection, live: sqlite3.Connection) -> None:
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(restore, "_overwrite", broken)
        with pytest.raises(restore.errors.CpError) as exc:
            restore.restore(
                migrated, clock=fake_clock, probe=fake_probe, kill=fake_kill, popen=fake_popen, lock_wait_s=0
            )
        err = exc.value
        replaced = err.extra["replaced"]
        assert err.code == "INTERNAL" and Path(replaced).exists() and replaced in err.extra["hint"]
        assert "retry" not in err.extra["hint"] or "pre-restore" in err.extra["hint"]
        assert _staged(migrated) == [] and not daemon.marker_path(migrated).exists()

    def test_the_live_db_gets_one_write_from_a_private_staged_copy(
        self, migrated, backed_up, fake_clock, fake_probe, fake_kill, fake_popen, monkeypatch
    ) -> None:
        seen: list[tuple[str, int, list[int]]] = []
        real_fix_up, real_overwrite = restore._fix_up, restore._overwrite

        def fix_up(c: sqlite3.Connection, waiters: Any, meta: Any) -> None:
            staged = restore._db_path(c)
            live_waiters = [r[0] for r in sql(migrated, "SELECT id FROM waiters")]
            seen.append((staged.name, staged.stat().st_mode & 0o777, live_waiters))
            real_fix_up(c, waiters, meta)

        overwrites: list[int] = []

        def overwrite(stage: sqlite3.Connection, live: sqlite3.Connection) -> None:
            overwrites.append(1)
            real_overwrite(stage, live)

        monkeypatch.setattr(restore, "_fix_up", fix_up)
        monkeypatch.setattr(restore, "_overwrite", overwrite)
        restore.restore(migrated, clock=fake_clock, probe=fake_probe, kill=fake_kill, popen=fake_popen, lock_wait_s=0)
        [(name, mode, live_waiters)] = seen
        assert name.startswith("harness_state.restoring.") and name.endswith(".db.partial") and mode == 0o600
        assert live_waiters == [42]  # the fix-up ran before the live DB was touched
        assert overwrites == [1] and _staged(migrated) == []
        assert [tuple(r) for r in sql(migrated, "SELECT id, pid FROM waiters")] == [(42, 9)]

    def test_a_copy_failure_keeps_the_replaced_pointer(
        self, migrated, backed_up, fake_clock, fake_probe, fake_kill, fake_popen, monkeypatch
    ) -> None:
        real = db.connect

        def connect(path: Any, *a: Any, **k: Any) -> Any:
            if Path(path) == backed_up["src"]:
                raise sqlite3.OperationalError("unable to open database file")
            return real(path, *a, **k)

        monkeypatch.setattr(restore.db, "connect", connect)
        monkeypatch.setattr(restore, "_check_source", lambda src, live: (CUR, 4096))
        with pytest.raises(restore.errors.CpError) as exc:
            restore.restore(
                migrated, clock=fake_clock, probe=fake_probe, kill=fake_kill, popen=fake_popen, lock_wait_s=0
            )
        assert exc.value.code == "CONFLICT" and Path(exc.value.extra["replaced"]).exists()

    def test_a_source_missing_a_carried_table_is_refused_untouched(
        self, migrated, backed_up, fake_clock, fake_probe, fake_kill, fake_popen
    ) -> None:
        sql(backed_up["src"], "DROP TABLE waiters")
        before = migrated.read_bytes()
        with pytest.raises(restore.errors.CpError) as exc:
            restore.restore(
                migrated, clock=fake_clock, probe=fake_probe, kill=fake_kill, popen=fake_popen, lock_wait_s=0
            )
        assert exc.value.code == "CONFLICT" and "waiters" in exc.value.detail
        assert migrated.read_bytes() == before and fake_popen.calls == []
        assert not list(backups.db_dir(migrated).glob("harness_state.replaced.*"))


class TestMarker:
    def test_ensure_answers_restoring_and_spawns_nothing(self, migrated, fake_clock, fake_probe, fake_popen) -> None:
        daemon.marker_path(migrated).write_text(json.dumps({"pid": 123, "pid_start": "start-123", "at": "x"}))
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated)
        assert res == {"status": "restoring"} and fake_popen.calls == []
        assert daemon.run(fake_clock, probe=fake_probe, db_path=migrated) == {"ticks": 0, "exit": "restoring"}

    @pytest.mark.parametrize("content", ['{"pid": 123, "pid_start": "start-123"}', "not json"])
    def test_a_dead_or_unreadable_marker_is_unlinked(
        self, migrated, fake_clock, fake_probe, fake_popen, content: str
    ) -> None:
        fake_probe.kill(123)
        daemon.marker_path(migrated).write_text(content)
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated)
        assert res["status"] == "started" and not daemon.marker_path(migrated).exists()


def test_the_ask_guard_covers_restore() -> None:
    payload = {
        "session_id": None,
        "tool_name": "Bash",
        "tool_input": {"command": "python scripts/qs/cp.py restore --confirm"},
    }
    payload["permission_mode"] = "default"
    code, out = run_cli("hook", "pre-tool-use", stdin=json.dumps(payload))
    assert out["hookSpecificOutput"]["permissionDecision"] == "ask"
    assert "cp.py restore" in out["hookSpecificOutput"]["permissionDecisionReason"]


def test_the_backup_hook_is_built_in() -> None:
    assert restore.BACKUP in dict(ticks.registered()) and alerts.BACKUP_FAILED in alerts.KINDS


def test_a_backup_failing_quick_check_is_a_conflict(tmp_path: Path, monkeypatch) -> None:
    class Conn:
        def execute(self, statement: str) -> Any:
            value = "*** page 4 is never used" if "quick_check" in statement else 1
            return type("R", (), {"fetchone": lambda self: (value,), "fetchall": lambda self: [(value,)]})()

        def close(self) -> None:
            pass

    monkeypatch.setattr(db, "connect", lambda *a, **k: Conn())
    with pytest.raises(restore.errors.CpError, match="fails quick_check"):
        restore._check_source(tmp_path / "b.db", None)


# --------------------------------------------------------------------------- review fix #01 (F11, F16)


@pytest.mark.parametrize("stored", ['"a bare string"', "not json at all", "[1, 2]"])
def test_a_non_dict_last_error_never_stops_backups(conn, migrated, fake_clock, stored: str) -> None:
    open_run()
    sql(migrated, "INSERT INTO meta (key, value) VALUES (?, ?)", [restore.LAST_ERROR, stored])
    _hook(conn, fake_clock)
    assert len(_periodic(migrated)) == 1 and _meta(migrated, restore.LAST_ERROR) is None
    assert _failed_alerts(migrated) == []


def test_a_rotation_failure_is_not_a_backup_failure(conn, migrated, fake_clock, monkeypatch, capsys) -> None:
    open_run()

    def broken(directory: Path, now: datetime) -> list[Path]:
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(backups, "rotate", broken)
    _hook(conn, fake_clock)
    assert len(_periodic(migrated)) == 1 and _meta(migrated, restore.LAST_AT) == db.now(fake_clock)
    assert _failed_alerts(migrated) == [] and _meta(migrated, restore.LAST_ERROR) is None
    assert "rotation failed" in capsys.readouterr().err
