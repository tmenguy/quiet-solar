"""Checkpoint 4: the migration registry and ``migrate()`` (AC2)."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from control_plane import SCHEMA_VERSION, db, errors, faults, migrations, paths, schema_v1

V2 = migrations.Migration(2, "test v2", ("CREATE TABLE extra (a INTEGER)", "CREATE TABLE extra2 (b INTEGER)"))


def _tables(path: Path) -> set[str]:
    c = sqlite3.connect(path)
    try:
        return {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        c.close()


def _version(path: Path) -> int:
    c = sqlite3.connect(path)
    try:
        return int(c.execute("PRAGMA user_version").fetchone()[0])
    finally:
        c.close()


def test_registry_has_no_gap() -> None:
    assert [m.version for m in migrations.MIGRATIONS] == list(range(1, len(migrations.MIGRATIONS) + 1))
    assert SCHEMA_VERSION == migrations.SCHEMA_VERSION == migrations.current_schema_version() == 1
    for m in migrations.MIGRATIONS:
        assert all(";" not in s.strip().rstrip(";") for s in m.statements)  # single statements


def test_fresh_db_is_created_with_every_table(db_path: Path) -> None:
    assert not db_path.exists()
    result = migrations.migrate(db_path, role="test")
    assert result == {"result": "migrated", "from": 0, "to": 1, "backup": None}
    assert set(schema_v1.TABLES) <= _tables(db_path)
    assert _version(db_path) == 1
    assert migrations.migrate(db_path, role="test") == {"result": "noop", "from": 1, "to": 1, "backup": None}


class TestLiveDbAuthorisation:
    def test_daemon_on_main_with_main_checked_out(self, fake_main: Path) -> None:
        live = fake_main / "harness_state.db"
        assert migrations.migrate(live, role="daemon")["to"] == 1

    @pytest.mark.parametrize("case", ["role", "feature-branch", "worktree-code", "other-checkout"])
    def test_refusals(self, case: str, fake_main: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        live = fake_main / "harness_state.db"
        role = "daemon"
        if case == "role":
            role = "test"
        elif case == "feature-branch":
            monkeypatch.setattr(paths, "main_head_branch", lambda m: None)
        elif case == "worktree-code":
            monkeypatch.setattr(paths, "code_root", lambda: tmp_path / "wt")
        else:
            other = tmp_path / "other"
            (other / ".git").mkdir(parents=True)
            live = other / "harness_state.db"
        with pytest.raises(errors.CpError) as exc:
            migrations.migrate(live, role=role)
        assert exc.value.code == "POLICY_REFUSED"
        assert not live.exists()


class TestUpgrade:
    def test_backup_before_migrating_an_existing_db(self, migrated: Path, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))
        result = migrations.migrate(migrated, role="test")
        assert result["from"] == 1 and result["to"] == 2
        backup = Path(result["backup"])
        assert backup.parent == (tmp_path / "backups").resolve()
        assert backup.name.startswith("harness_state.v1.") and backup.suffix == ".db"
        assert _version(backup) == 1 and "extra" not in _tables(backup)
        assert {"extra", "extra2"} <= _tables(migrated) and _version(migrated) == 2

    def test_fault_mid_step_leaves_no_partial_table(self, migrated: Path, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))
        with faults.arm("migrate.mid_step"), pytest.raises(faults.FaultInjected):
            migrations.migrate(migrated, role="test")
        assert _version(migrated) == 1
        assert "extra" not in _tables(migrated)
        assert len(list((tmp_path / "backups").glob("harness_state.v1.*.db"))) == 1

    def test_failing_statement_rolls_back(self, migrated: Path, monkeypatch) -> None:
        bad = migrations.Migration(2, "bad", ("CREATE TABLE extra (a INTEGER)", "NOT SQL"))
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, bad))
        with pytest.raises(sqlite3.OperationalError):
            migrations.migrate(migrated, role="test")
        assert _version(migrated) == 1 and "extra" not in _tables(migrated)

    def test_db_newer_than_the_code(self, migrated: Path) -> None:
        c = sqlite3.connect(migrated)
        c.execute("PRAGMA user_version = 7")
        c.close()
        with pytest.raises(errors.CpError) as exc:
            migrations.migrate(migrated, role="test")
        assert exc.value.code == "SCHEMA_TOO_NEW"


def test_two_concurrent_migrators(db_path: Path) -> None:
    barrier = threading.Barrier(2)
    results: list[str] = []

    def worker() -> None:
        barrier.wait(timeout=2)
        results.append(migrations.migrate(db_path, role="test")["result"])

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert sorted(results) == ["migrated", "noop"]


def test_lock_held_elsewhere_is_busy(db_path: Path) -> None:
    with db.file_lock(paths.sidecar(db_path, ".migrate.lock"), exclusive=True, timeout=0) as got:
        assert got
        with pytest.raises(errors.CpError) as exc:
            migrations.migrate(db_path, role="test", lock_timeout=0.05)
    assert exc.value.code == "BUSY"
