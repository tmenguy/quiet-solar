"""#407 T1: migration v4 — ``reviews``, ``review_tasks``, ``review_triggers`` and the pilot cursor (D1, D13)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from control_plane import migrations, schema_reviews

from .conftest import CUR, insert_task, sql


def _v3_db(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(migrations, "MIGRATIONS", migrations.MIGRATIONS[:3])
        migrations.migrate(db_path, role="test")


def _cursor(path: Path) -> str:
    return str(sql(path, "SELECT value FROM meta WHERE key = ?", (schema_reviews.CURSOR_KEY,))[0][0])


def _cols(path: Path, table: str) -> list[str]:
    return [r[1] for r in sql(path, f"PRAGMA table_info({table})")]


def test_v4_is_registered() -> None:
    assert [m.version for m in migrations.MIGRATIONS][:4] == [1, 2, 3, 4]
    assert migrations.MIGRATIONS[3].statements == schema_reviews.STATEMENTS
    assert migrations.MIGRATIONS[3].name == "review sessions (#407)"
    assert CUR >= 4
    assert schema_reviews.STATEMENTS[-1].startswith("INSERT OR REPLACE INTO meta")  # the cursor seed is last


def test_fresh_db_has_the_tables_indexes_and_a_zero_cursor(migrated: Path) -> None:
    assert _cols(migrated, "reviews") == [
        "id",
        "kind",
        "coalesce_key",
        "home_run_id",
        "state",
        "agent",
        "model",
        "effort",
        "permission_mode",
        "nonce",
        "grant_nonce",
        "name",
        "session_id",
        "short_id",
        "dir",
        "tool_key",
        "requested_by",
        "summary",
        "error",
        "heads_at",
        "created_at",
        "launch_at",
        "finished_at",
        "updated_at",
    ]
    assert _cols(migrated, "review_tasks") == [
        "review_id",
        "task_id",
        "run_id",
        "base_sha",
        "head_sha",
        "head_state",
        "outcome",
        "reason",
        "blast_radius_id",
        "recorded_at",
    ]
    assert _cols(migrated, "review_triggers") == ["id", "review_id", "actor", "trigger", "tool_key", "outcome", "at"]
    indexes = {r[0] for r in sql(migrated, "SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {"reviews_in_flight", "reviews_state", "review_tasks_task", "review_triggers_pilot"} <= indexes
    assert _cursor(migrated) == "0"


def test_upgrade_from_v3_seeds_the_cursor_past_history(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _v3_db(db_path, monkeypatch)
    sql(db_path, "INSERT INTO runs (id, name, title, state, created_at) VALUES ('R1', 'r', 't', 'open', 'x')")
    insert_task(db_path, "T1", "R1")
    for i in range(3):
        sql(
            db_path,
            "INSERT INTO task_history (task_id, at, actor, to_state) VALUES ('T1', ?, 'a', 'building')",
            (str(i),),
        )
    result = migrations.migrate(db_path, role="test")
    assert result["from"] == 3 and result["to"] == CUR
    assert _cursor(db_path) == "3"


def _review(path: Path, rid: str, key: str, state: str = "requested") -> None:
    sql(
        path,
        "INSERT INTO reviews (id, kind, coalesce_key, state, agent, name, requested_by, created_at, updated_at)"
        " VALUES (?, 'global', ?, ?, 'a', ?, 'x', 'x', 'x')",
        (rid, key, state, f"n-{rid}"),
    )


def test_one_in_flight_review_per_coalesce_key(migrated: Path) -> None:
    _review(migrated, "V1", "global:R1")
    for state in ("requested", "launching", "running"):
        with pytest.raises(sqlite3.IntegrityError):
            _review(migrated, "V2", "global:R1", state)
    for i, state in enumerate(schema_reviews.CLOSED, 3):
        _review(migrated, f"V{i}", "global:R1", state)  # closed reviews of the key are kept
    sql(migrated, "UPDATE reviews SET state = 'failed' WHERE id = 'V1'")
    _review(migrated, "V9", "global:R1", "launching")  # the key is free again


def test_states_and_outcomes_are_checked(migrated: Path) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _review(migrated, "V1", "k", "done")
    _review(migrated, "V1", "k")
    sql(migrated, "INSERT INTO runs (id, name, title, state, created_at) VALUES ('R1', 'r', 't', 'open', 'x')")
    insert_task(migrated, "T1", "R1")
    row = "INSERT INTO review_tasks (review_id, task_id, run_id, head_state, outcome) VALUES ('V1', ?, 'R1', ?, ?)"
    sql(migrated, row, ("T1", None, None))  # pending: no head state, no outcome
    with pytest.raises(sqlite3.IntegrityError):
        sql(migrated, row, ("T1", "plan", None))  # PRIMARY KEY (review_id, task_id)
    insert_task(migrated, "T2", "R1")
    with pytest.raises(sqlite3.IntegrityError):
        sql(migrated, row, ("T2", "detached", None))
    with pytest.raises(sqlite3.IntegrityError):
        sql(migrated, row, ("T2", "plan", "maybe"))
    with pytest.raises(sqlite3.IntegrityError):
        sql(migrated, row, ("T9", "plan", "pass"))  # REFERENCES tasks(id)


def test_one_trigger_row_per_pilot_attempt(migrated: Path) -> None:
    row = (
        'INSERT INTO review_triggers (review_id, actor, "trigger", tool_key, outcome, at) VALUES (NULL, ?, ?, ?, ?, ?)'
    )
    sql(migrated, row, ("cp:review_pilot", "{}", "pilot:merge:T1:4:1", "queued", "x"))
    with pytest.raises(sqlite3.IntegrityError):
        sql(migrated, row, ("cp:review_pilot", "{}", "pilot:merge:T1:4:1", "queued", "x"))
    sql(migrated, row, ("cp:review_pilot", "{}", "pilot:merge:T1:4:2", "queued", "x"))  # the next attempt
    for _ in range(2):  # a run-token key may repeat: a skip, a join, a replay are separate rows
        sql(migrated, row, ("orchestrator", "{}", "task:R1:global:1", "joined", "x"))
    with pytest.raises(sqlite3.IntegrityError):
        sql(migrated, row, ("x", "{}", "k", "launched", "x"))
