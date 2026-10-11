"""QS-406 T2: migration v3 (v2 is #375's ledger) — the ``alerts`` table and the ``hook_events`` cursor (§11)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from control_plane import migrations, schema_v3

from .conftest import sql


def _v1_db(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(migrations, "MIGRATIONS", migrations.MIGRATIONS[:1])
        migrations.migrate(db_path, role="test")


def _cursor(path: Path) -> str:
    return str(sql(path, "SELECT value FROM meta WHERE key = 'hook_events_cursor'")[0][0])


def test_v3_is_registered() -> None:
    assert [m.version for m in migrations.MIGRATIONS][:3] == [1, 2, 3]
    assert migrations.MIGRATIONS[2].statements == schema_v3.STATEMENTS
    assert migrations.current_schema_version() >= 3


def test_fresh_db_has_alerts_and_a_zero_cursor(migrated: Path) -> None:
    cols = [r[1] for r in sql(migrated, "PRAGMA table_info(alerts)")]
    assert cols == [
        "id",
        "run_id",
        "kind",
        "subject",
        "fingerprint",
        "payload",
        "message_id",
        "first_seen",
        "last_seen",
        "cleared_at",
    ]
    assert sql(migrated, "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'alerts_open'")
    assert _cursor(migrated) == "0"


def test_upgrade_seeds_the_cursor_past_history(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _v1_db(db_path, monkeypatch)
    for i in range(3):
        sql(db_path, "INSERT INTO hook_events (hook, decision, detail, at) VALUES ('h', 'alert', '{}', ?)", (str(i),))
    result = migrations.migrate(db_path, role="test")
    assert result["from"] == 1 and result["to"] == migrations.current_schema_version()
    assert _cursor(db_path) == "3"


def test_alerts_constraints(migrated: Path) -> None:
    sql(migrated, "INSERT INTO runs (id, name, title, state, created_at) VALUES ('R1', 'r', 't', 'open', 'x')")
    row = "INSERT INTO alerts (run_id, kind, subject, fingerprint, payload, first_seen, last_seen) VALUES (?, ?, 's', ?, '{}', 'x', 'x')"
    sql(migrated, row, ("R1", "overlap", "f1"))
    with pytest.raises(sqlite3.IntegrityError):
        sql(migrated, row, ("R1", "overlap", "f1"))  # UNIQUE (run_id, fingerprint)
    with pytest.raises(sqlite3.IntegrityError):
        sql(migrated, row, ("R1", "", "f2"))  # kind length 1..32
    with pytest.raises(sqlite3.IntegrityError):
        sql(migrated, row, ("R1", "k" * 33, "f3"))
    with pytest.raises(sqlite3.IntegrityError):
        sql(migrated, row, ("R9", "overlap", "f4"))  # REFERENCES runs(id)
