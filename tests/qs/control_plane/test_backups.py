"""QS-406 T3: the per-DB backup directory and ``write_copy`` (§10, AC 15)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from control_plane import backups, daemon, errors, migrations, paths

from .conftest import sql


def _ver(p: Path) -> int:
    c = sqlite3.connect(p)
    try:
        return int(c.execute("PRAGMA user_version").fetchone()[0])
    finally:
        c.close()


def _mode(p: Path) -> int:
    return p.stat().st_mode & 0o777


def test_db_dir_is_per_db(tmp_path: Path, db_path: Path) -> None:
    d = backups.db_dir(db_path)
    assert d.parent == (tmp_path / "backups").resolve()
    assert d.name == hashlib.sha256(str(db_path.resolve()).encode()).hexdigest()[:8]
    assert backups.db_dir(tmp_path / "other.db") != d


class TestWriteCopy:
    def test_private_delete_journal_copy(self, conn: sqlite3.Connection, migrated: Path, tmp_path: Path) -> None:
        sql(migrated, "INSERT INTO meta (key, value) VALUES ('k', 'v')")
        dest = backups.db_dir(migrated) / "harness_state.periodic.20261009T120000000000Z.db"
        assert backups.write_copy(conn, dest) == dest
        assert _mode(dest) == 0o600
        assert _mode(dest.parent) == 0o700 and _mode(tmp_path / "backups") == 0o700
        assert dest.read_bytes()[18] == 1  # the header says rollback journal, not WAL
        copy = sqlite3.connect(dest)
        try:
            assert copy.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
            assert copy.execute("SELECT value FROM meta WHERE key = 'k'").fetchone()[0] == "v"
        finally:
            copy.close()
        assert sorted(p.name for p in dest.parent.iterdir()) == [dest.name]

    def test_a_failure_midway_leaves_nothing(self, conn: sqlite3.Connection, migrated: Path, monkeypatch) -> None:
        dest = backups.db_dir(migrated) / "harness_state.periodic.20261009T120000000000Z.db"
        monkeypatch.setattr(backups, "_journal_delete", lambda copy: "wal")
        with pytest.raises(sqlite3.OperationalError):
            backups.write_copy(conn, dest)
        assert list(dest.parent.iterdir()) == []

    def test_a_failing_source_leaves_nothing(self, migrated: Path) -> None:
        class Broken:
            def backup(self, target: sqlite3.Connection) -> None:
                raise sqlite3.OperationalError("disk I/O error")

        dest = backups.db_dir(migrated) / "x.db"
        with pytest.raises(sqlite3.OperationalError):
            backups.write_copy(Broken(), dest)  # type: ignore[arg-type]
        assert list(dest.parent.iterdir()) == []

    def test_a_foreign_backup_dir_is_refused(self, conn: sqlite3.Connection, migrated: Path, tmp_path, monkeypatch):
        (tmp_path / "backups").mkdir()
        monkeypatch.setattr(paths.os, "getuid", lambda: (tmp_path / "backups").stat().st_uid + 1)
        with pytest.raises(errors.CpError) as exc:
            backups.write_copy(conn, backups.db_dir(migrated) / "x.db")
        assert exc.value.code == "POLICY_REFUSED"


class TestMigrationBackups:
    def _v1(self, db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(migrations, "MIGRATIONS", migrations.MIGRATIONS[:1])
            migrations.migrate(db_path, role="test")

    def test_a_0755_backup_dir_left_by_399_is_tightened(self, db_path: Path, tmp_path: Path, monkeypatch) -> None:
        self._v1(db_path, monkeypatch)
        legacy = tmp_path / "backups"
        legacy.mkdir(mode=0o755)
        legacy.chmod(0o755)
        result = migrations.migrate(db_path, role="test")
        backup = Path(result["backup"])
        assert backup.parent == backups.db_dir(db_path) and backup.name.startswith("harness_state.v1.")
        assert _mode(legacy) == 0o700 and _mode(backup.parent) == 0o700 and _mode(backup) == 0o600
        assert _ver(backup) == 1

    def test_a_refused_backup_dir_fails_the_migration(
        self, db_path: Path, fake_main: Path, fake_clock, fake_probe, monkeypatch
    ) -> None:
        self._v1(db_path, monkeypatch)
        monkeypatch.setenv("QS_CP_BACKUP_DIR", str(fake_main / "bk"))
        with pytest.raises(errors.CpError) as exc:
            migrations.migrate(db_path, role="test")
        assert exc.value.code == "POLICY_REFUSED" and _ver(db_path) == 1
        with pytest.raises(errors.CpError) as exc:
            daemon.run(fake_clock, probe=fake_probe, db_path=db_path)
        assert exc.value.code == "INTERNAL" and _ver(db_path) == 1
        side = json.loads(paths.sidecar(db_path, ".migrate-error.json").read_text())
        assert "POLICY_REFUSED" in side["error"]
