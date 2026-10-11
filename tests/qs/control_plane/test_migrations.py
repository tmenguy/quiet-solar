"""Checkpoint 4: the migration registry and ``migrate()`` (AC2)."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from control_plane import SCHEMA_VERSION, backups, db, errors, faults, migrations, paths, schema_v1

from .conftest import CUR, NEXT

V2 = migrations.Migration(NEXT, "test v2", ("CREATE TABLE extra (a INTEGER)", "CREATE TABLE extra2 (b INTEGER)"))


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
    assert SCHEMA_VERSION == migrations.SCHEMA_VERSION == migrations.current_schema_version() == CUR
    for m in migrations.MIGRATIONS:
        assert all(";" not in s.strip().rstrip(";") for s in m.statements)  # single statements


def test_the_ledger_migration_is_registered() -> None:
    """QS-375: the versions are exactly ``1..CUR`` and the ledger's step is one of them."""
    from control_plane import schema_ledger

    assert [m.version for m in migrations.MIGRATIONS] == list(range(1, CUR + 1))
    assert CUR >= 2
    assert [m.statements for m in migrations.MIGRATIONS].count(schema_ledger.STATEMENTS) == 1


def _ledger_index() -> int:
    from control_plane import schema_ledger

    return next(i for i, m in enumerate(migrations.MIGRATIONS) if m.statements == schema_ledger.STATEMENTS)


@pytest.fixture
def v1_db(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A pre-ledger DB with rows (QS-375 AC1): migrated by the registry *before* the ledger's step."""
    from .conftest import insert_task, sql

    with monkeypatch.context() as mp:  # never monkeypatch.undo(): it would revert _cp_isolation too
        mp.setattr(migrations, "MIGRATIONS", migrations.MIGRATIONS[: _ledger_index()])
        migrations.migrate(db_path, role="test")
        sql(db_path, "INSERT INTO runs (id, name, title, state, created_at) VALUES ('R1', 'r1', 't', 'open', 'x')")
        insert_task(db_path, "T1", "R1", ci_state="red", ci_sha="abc")
        sql(db_path, "INSERT INTO counters (kind, next) VALUES ('task', 2)")
    return db_path


def test_a_pre_ledger_db_is_upgraded(v1_db: Path, tmp_path: Path) -> None:
    from control_plane import schema_ledger

    from .conftest import sql

    i = _ledger_index()
    assert _version(v1_db) == i and not set(schema_ledger.TABLES) & _tables(v1_db)
    result = migrations.migrate(v1_db, role="test")
    assert result == {"result": "migrated", "from": i, "to": CUR, "backup": result["backup"]}
    assert isinstance(result["backup"], str) and Path(result["backup"]).exists()
    assert Path(result["backup"]).name.startswith(f"harness_state.v{i}.")
    assert set(schema_ledger.TABLES) <= _tables(v1_db) and _version(v1_db) == CUR
    assert [tuple(r) for r in sql(v1_db, "SELECT id, run_id, ci_state, ci_sha FROM tasks")] == [
        ("T1", "R1", "red", "abc")
    ]
    assert [tuple(r) for r in sql(v1_db, "SELECT kind, next FROM counters")] == [("task", 2)]


def test_fresh_db_is_created_with_every_table(db_path: Path) -> None:
    assert not db_path.exists()
    result = migrations.migrate(db_path, role="test")
    assert result == {"result": "migrated", "from": 0, "to": CUR, "backup": None}
    from control_plane import schema_ledger  # in-function: the red step of QS-375 kept collection alive

    assert set(schema_v1.TABLES) | set(schema_ledger.TABLES) <= _tables(db_path)
    assert _version(db_path) == CUR
    assert migrations.migrate(db_path, role="test") == {"result": "noop", "from": CUR, "to": CUR, "backup": None}


class TestLiveDbAuthorisation:
    def test_daemon_on_main_with_main_checked_out(self, fake_main: Path) -> None:
        live = fake_main / "harness_state.db"
        assert migrations.migrate(live, role="daemon")["to"] == CUR

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
        assert result["from"] == CUR and result["to"] == NEXT
        backup = Path(result["backup"])
        assert backup.parent == backups.db_dir(migrated)
        assert backup.name.startswith(f"harness_state.v{CUR}.") and backup.suffix == ".db"
        assert _version(backup) == CUR and "extra" not in _tables(backup)
        assert {"extra", "extra2"} <= _tables(migrated) and _version(migrated) == NEXT

    def test_fault_mid_step_leaves_no_partial_table(self, migrated: Path, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))
        with faults.arm("migrate.mid_step"), pytest.raises(faults.FaultInjected):
            migrations.migrate(migrated, role="test")
        assert _version(migrated) == CUR
        assert "extra" not in _tables(migrated)
        assert len(list(backups.db_dir(migrated).glob(f"harness_state.v{CUR}.*.db"))) == 1

    def test_failing_statement_rolls_back(self, migrated: Path, monkeypatch) -> None:
        bad = migrations.Migration(NEXT, "bad", ("CREATE TABLE extra (a INTEGER)", "NOT SQL"))
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, bad))
        with pytest.raises(sqlite3.OperationalError):
            migrations.migrate(migrated, role="test")
        assert _version(migrated) == CUR and "extra" not in _tables(migrated)

    def test_db_newer_than_the_code(self, migrated: Path) -> None:
        c = sqlite3.connect(migrated)
        c.execute(f"PRAGMA user_version = {NEXT}")
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


# --------------------------------------------------------------------------- review fix #01 (F6, F7)


class TestReviewFix01:
    def test_a_current_db_is_a_noop_on_a_feature_branch(self, fake_main: Path, monkeypatch) -> None:
        """F6: no migration needed → no authorisation needed, whatever main's checkout."""
        live = fake_main / "harness_state.db"
        migrations.migrate(live, role="daemon")
        monkeypatch.setattr(paths, "main_head_branch", lambda m: None)  # detached HEAD or a feature branch
        assert migrations.migrate(live, role="daemon")["result"] == "noop"
        assert migrations.migrate(live, role="test")["result"] == "noop"

    def test_an_older_live_db_is_still_refused_off_main(self, fake_main: Path, monkeypatch) -> None:
        live = fake_main / "harness_state.db"
        migrations.migrate(live, role="daemon")
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))
        monkeypatch.setattr(paths, "main_head_branch", lambda m: "QS_1")
        with pytest.raises(errors.CpError) as exc:
            migrations.migrate(live, role="daemon")
        assert exc.value.code == "POLICY_REFUSED" and _version(live) == CUR

    def test_a_busy_begin_is_busy(self, migrated: Path, monkeypatch) -> None:
        """F7: lock contention is ``BUSY``, never a raw ``OperationalError``."""
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))
        monkeypatch.setattr(db, "BUSY_TIMEOUT_MS", 20)
        holder = sqlite3.connect(migrated, isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(errors.CpError) as exc:
                migrations.migrate(migrated, role="test")
            assert exc.value.code == "BUSY"
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        assert _version(migrated) == CUR

    def test_an_auto_rolled_back_step_keeps_its_own_error(self, migrated: Path, monkeypatch) -> None:
        """F7: SQLite already rolled back; a bare ROLLBACK would hide the IntegrityError."""
        rb = migrations.Migration(
            NEXT,
            "rb",
            ("CREATE TABLE t (x UNIQUE)", "INSERT INTO t VALUES (1)", "INSERT OR ROLLBACK INTO t VALUES (1)"),
        )
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, rb))
        with pytest.raises(sqlite3.IntegrityError):
            migrations.migrate(migrated, role="test")
        assert _version(migrated) == CUR and "t" not in _tables(migrated)


# --------------------------------------------------------------------------- review fix #02 (G14)


class TestReviewFix02:
    @pytest.mark.parametrize("code", [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, 517])  # 517: SQLITE_BUSY_SNAPSHOT
    @pytest.mark.parametrize("where", ["user_version", "backup", "mid_step"])
    def test_any_busy_or_locked_error_is_busy(self, migrated: Path, monkeypatch, code: int, where: str) -> None:
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))

        def busy(*a: object, **k: object) -> object:
            exc = sqlite3.OperationalError("database is locked")
            exc.sqlite_errorcode = code  # type: ignore[attr-defined]
            raise exc

        if where == "user_version":
            monkeypatch.setattr(db, "user_version", busy)
        elif where == "backup":
            monkeypatch.setattr(migrations, "_backup", busy)
        else:
            monkeypatch.setattr(faults, "hit", busy)  # inside the step, after its first statement
        with pytest.raises(errors.CpError) as exc:
            migrations.migrate(migrated, role="test")
        assert exc.value.code == "BUSY" and _version(migrated) == CUR

    def test_another_operational_error_stays_itself(self, migrated: Path, monkeypatch) -> None:
        def broken(*a: object, **k: object) -> object:
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(db, "user_version", broken)
        with pytest.raises(sqlite3.OperationalError):
            migrations.migrate(migrated, role="test")
