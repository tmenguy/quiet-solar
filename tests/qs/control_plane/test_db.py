"""Checkpoint 4: connections, write transactions and the entry schema check (§3, AC3)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from control_plane import db, errors, migrations, paths

from .conftest import CUR, V_NEXT


class TestConnect:
    def test_never_creates_the_file(self, db_path: Path) -> None:
        with pytest.raises(errors.CpError) as exc:
            db.connect(db_path)
        assert exc.value.code == "INTERNAL"
        assert not db_path.exists()

    def test_pragmas(self, migrated: Path) -> None:
        c = db.connect(migrated)
        assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert c.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert c.isolation_level is None
        c.close()
        ro = db.connect(migrated, mode="ro")
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("INSERT INTO meta (key, value) VALUES ('a', 'b')")
        ro.close()


class TestWrite:
    def test_commit_and_rollback(self, conn: sqlite3.Connection) -> None:
        with db.write(conn):
            conn.execute("INSERT INTO meta (key, value) VALUES ('a', '1')")
        with pytest.raises(RuntimeError), db.write(conn):
            conn.execute("INSERT INTO meta (key, value) VALUES ('b', '2')")
            raise RuntimeError
        assert [r[0] for r in conn.execute("SELECT key FROM meta WHERE key IN ('a', 'b')")] == ["a"]
        assert not conn.in_transaction

    def test_write_after_a_migration_is_refused(self, conn: sqlite3.Connection, migrated: Path) -> None:
        other = sqlite3.connect(migrated)
        other.execute(f"PRAGMA user_version = {V_NEXT}")
        other.close()
        with pytest.raises(errors.CpError) as exc, db.write(conn):
            conn.execute("INSERT INTO meta (key, value) VALUES ('a', '1')")
        assert exc.value.code == "SCHEMA_TOO_NEW" and exc.value.extra == {"db_version": V_NEXT, "code_version": CUR}
        assert conn.execute("SELECT count(*) FROM meta WHERE key = 'a'").fetchone()[0] == 0

    def test_busy(self, conn: sqlite3.Connection, migrated: Path) -> None:
        holder = db.connect(migrated)
        holder.execute("BEGIN IMMEDIATE")
        conn.execute("PRAGMA busy_timeout = 0")
        try:
            with pytest.raises(errors.CpError) as exc, db.write(conn):
                pass
            assert exc.value.code == "BUSY"
        finally:
            holder.execute("ROLLBACK")
            holder.close()

    def test_read_snapshot(self, conn: sqlite3.Connection) -> None:
        with db.read(conn):
            assert conn.in_transaction
            conn.execute("SELECT count(*) FROM runs").fetchone()
        assert not conn.in_transaction
        with db.read(conn):
            conn.execute("COMMIT")  # an early end is tolerated
        assert not conn.in_transaction


class TestFileLock:
    def test_exclusive_excludes(self, tmp_path: Path) -> None:
        lock = tmp_path / "x.lock"
        with db.file_lock(lock, exclusive=True, timeout=None) as got:
            assert got
            with db.file_lock(lock, exclusive=False, timeout=0.05) as other:
                assert not other
        with db.file_lock(lock, exclusive=False, timeout=0) as a, db.file_lock(lock, exclusive=False, timeout=0) as b:
            assert a and b


class TestProbeAndSidecar:
    def test_probe_version(self, db_path: Path) -> None:
        assert db.probe_version(db_path) is None
        migrations.migrate(db_path, role="test")
        assert db.probe_version(db_path) == CUR
        with db.file_lock(paths.sidecar(db_path, ".migrate.lock"), exclusive=True, timeout=0):
            assert db.probe_version(db_path) == -1

    def test_migrate_error(self, db_path: Path) -> None:
        side = paths.sidecar(db_path, ".migrate-error.json")
        assert db.migrate_error(db_path) is None
        side.write_text("{bad")
        assert db.migrate_error(db_path) is None
        side.write_text("[1]")
        assert db.migrate_error(db_path) is None
        side.write_text(json.dumps({"at": "x", "error": "e"}))
        assert db.migrate_error(db_path) == {"at": "x", "error": "e"}


def _set_version(path: Path, v: int) -> None:
    c = sqlite3.connect(path)
    c.execute(f"PRAGMA user_version = {v}")
    c.close()


class TestCheckSchema:
    def test_equal(self, migrated: Path, fake_clock) -> None:
        assert db.check_schema(migrated, wait=False, clock=fake_clock) == "ok"
        assert db.check_schema(migrated, wait=True, clock=fake_clock) == "ok"

    @pytest.mark.parametrize("wait", [False, True])
    def test_newer_refuses(self, migrated: Path, fake_clock, wait: bool) -> None:
        _set_version(migrated, V_NEXT)
        calls: list[int] = []
        with pytest.raises(errors.CpError) as exc:
            db.check_schema(migrated, wait=wait, clock=fake_clock, ensure=lambda: calls.append(1) or {})
        assert exc.value.code == "SCHEMA_TOO_NEW" and calls == []

    def test_no_wait(self, db_path: Path, fake_clock) -> None:
        assert db.check_schema(db_path, wait=False, clock=fake_clock) == "missing"
        migrations.migrate(db_path, role="test")
        _set_version(db_path, 0)
        with pytest.raises(errors.CpError) as exc:
            db.check_schema(db_path, wait=False, clock=fake_clock)
        assert exc.value.code == "SCHEMA_PENDING"
        assert fake_clock.sleeps == []

    def test_wait_until_the_daemon_migrated(self, db_path: Path, fake_clock) -> None:
        def ensure() -> dict:
            fake_clock.sleep = lambda s: migrations.migrate(db_path, role="test")  # migrates on the first poll
            return {"status": "started"}

        assert db.check_schema(db_path, wait=True, clock=fake_clock, ensure=ensure) == "ok"

    def test_wait_times_out_with_the_sidecar_error(self, db_path: Path, fake_clock) -> None:
        paths.sidecar(db_path, ".migrate-error.json").write_text(json.dumps({"at": "t", "error": "boom"}))
        with pytest.raises(errors.CpError) as exc:
            db.check_schema(db_path, wait=True, clock=fake_clock, wait_s=2, poll_s=0.5)
        assert exc.value.code == "SCHEMA_PENDING"
        assert exc.value.extra["migrate_error"] == {"at": "t", "error": "boom"}
        assert fake_clock.sleeps == [0.5] * 4

    def test_migrate_failed_returns_at_once(self, db_path: Path, fake_clock) -> None:
        with pytest.raises(errors.CpError) as exc:
            db.check_schema(
                db_path, wait=True, clock=fake_clock, ensure=lambda: {"status": "migrate_failed", "error": "e"}
            )
        assert exc.value.code == "SCHEMA_PENDING" and exc.value.extra == {"migrate_error": "e"}
        assert fake_clock.sleeps == []

    def test_becomes_newer_while_waiting(self, migrated: Path, fake_clock) -> None:
        _set_version(migrated, 0)
        fake_clock.sleep = lambda s: _set_version(migrated, 5)
        with pytest.raises(errors.CpError) as exc:
            db.check_schema(migrated, wait=True, clock=fake_clock)
        assert exc.value.code == "SCHEMA_TOO_NEW"


def test_next_id_and_helpers(conn: sqlite3.Connection, fake_clock) -> None:
    with db.write(conn):
        assert db.next_id(conn, "run", "R") == "R1"
        assert db.next_id(conn, "run", "R") == "R2"
        assert db.next_id(conn, "task", "T") == "T1"
    assert db.as_dict(None) is None
    assert db.as_dict(conn.execute("SELECT 1 AS a").fetchone()) == {"a": 1}
    assert db.now(fake_clock) == "2026-10-03T12:00:00.000000Z"


# --------------------------------------------------------------------------- review fix #01 (F20, F24)


class TestReviewFix01:
    def test_a_failed_commit_rolls_back(self, conn: sqlite3.Connection) -> None:
        """F20: a deferred foreign-key violation fails the COMMIT; the connection must not stay in a transaction."""
        conn.execute("CREATE TEMP TABLE p (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TEMP TABLE c (pid INTEGER REFERENCES p (id) DEFERRABLE INITIALLY DEFERRED)")
        with pytest.raises(sqlite3.IntegrityError), db.write(conn):
            conn.execute("INSERT INTO c (pid) VALUES (42)")
        assert not conn.in_transaction
        with db.write(conn):  # the connection is usable again
            conn.execute("INSERT INTO meta (key, value) VALUES ('ok', '1')")

    def test_only_busy_and_locked_map_to_busy(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN")
        try:
            with pytest.raises(errors.CpError) as exc, db.write(conn):
                pass
            assert exc.value.code == "INTERNAL" and "within a transaction" in exc.value.detail
        finally:
            conn.execute("ROLLBACK")

    def test_a_running_migration_has_its_own_message(self, migrated: Path, fake_clock) -> None:
        """F24: the ``_MIGRATING`` sentinel is not reported as "schema v-1"."""
        with db.file_lock(paths.sidecar(migrated, ".migrate.lock"), exclusive=True, timeout=0):
            with pytest.raises(errors.CpError) as exc:
                db.check_schema(migrated, wait=False, clock=fake_clock)
        assert exc.value.code == "SCHEMA_PENDING" and "a migration is in progress" in exc.value.detail
        assert "v-1" not in exc.value.detail
