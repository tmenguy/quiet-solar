"""Checkpoint 11: the per-merge export, ``snapshot`` and ``task show`` (AC1, AC16)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from control_plane import export, snapshot

from .conftest import ORCH, insert_node, open_run, run_cli, sql

GOLDEN = Path(__file__).parent / "golden"


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def linked(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q", "--allow-empty", "-m", "x", cwd=repo)
    _git("worktree", "add", "-q", "-b", "QS_11", str(tmp_path / "wt"), cwd=repo)
    return tmp_path / "wt"


def _file(tmp: Path, name: str, text: str) -> str:
    p = tmp / name
    p.write_text(text)
    return str(p)


@pytest.fixture
def world(migrated: Path, tmp_path: Path) -> dict[str, Any]:
    run_id, token = open_run()
    ok = lambda *a: run_cli(*a, "--token", token)  # noqa: E731
    assert (
        ok("task", "add", "--run", run_id, "--title", "The | Control Plane", "--kind", "feature", "--issue", "11")[0]
        == 0
    )
    ok("task", "add", "--run", run_id, "--title", "Other", "--kind", "bug")
    ok("task", "set", "--task", "T1", "--pr-number", "42", "--pr-url", "https://x/pull/42", "--branch", "QS_11")
    ok("criteria", "set", "--task", "T1", "--file", _file(tmp_path, "c.txt", "Schema v1\nQueues work\n"))
    ok("criteria", "validate", "--task", "T1")
    ok("criteria", "state", "--task", "T1", "--idx", "1", "--to", "met")
    ok("digest", "put", "--task", "T1", "--file", _file(tmp_path, "d.md", "Built the core.\n\nAll green.\n"))
    ok(
        "decision",
        "add",
        "--text",
        "Use SQLite WAL",
        "--reason",
        "concurrent readers",
        "--source",
        "maintainer",
        "--task",
        "T1",
    )
    ok("decision", "add", "--text", "Ship in one PR", "--reason", "asked", "--source", "maintainer", "--task", "T1")
    ok("task", "dep", "add", "--task", "T1", "--on", "T2")
    ok("task", "root", "add", "--run", run_id, "--task", "T1")
    ok("task", "work-list", "add", "--run", run_id, "--task", "T1")
    sql(migrated, "UPDATE tasks SET merge_sha = 'f00d' WHERE id = 'T1'")
    return {"run": run_id, "token": token, "db": migrated, "tmp": tmp_path}


class TestExport:
    def test_golden_summary(self, world, linked) -> None:
        code, out = run_cli("export-summary", "--task", "T1", "--out-worktree", str(linked))
        assert code == 0, out
        path = Path(out["path"])
        assert path == linked.resolve() / "docs" / "stories" / "QS-11.summary.md"
        assert path.read_text() == (GOLDEN / "summary.md").read_text()

    def test_refuses_the_main_checkout(self, world, fake_main) -> None:
        code, out = run_cli("export-summary", "--task", "T1", "--out-worktree", str(fake_main))
        assert code == 9 and out["error"] == "POLICY_REFUSED"
        assert not (fake_main / "docs").exists()

    def test_placeholders_and_a_ledger_section(self, world, linked) -> None:
        run_cli("task", "set", "--task", "T2", "--issue", "12", "--token", world["token"])
        export.LEDGER_SECTIONS.append(("Tokens", lambda conn, task: f"ledger of {task}\n"))
        run_cli("export-summary", "--task", "T2", "--out-worktree", str(linked))
        text = (linked / "docs" / "stories" / "QS-12.summary.md").read_text()
        for placeholder in (
            "_No digest recorded._",
            "_No criteria recorded._",
            "_No decisions recorded._",
            "No PR · merge pending",
        ):
            assert placeholder in text
        assert "### Tokens\n\nledger of T2\n" in text and "No ledger recorded." not in text

    def test_errors(self, world, linked, db_path) -> None:
        assert run_cli("export-summary", "--task", "T2", "--out-worktree", str(linked))[1]["error"] == "INVALID_STATE"
        assert run_cli("export-summary", "--task", "T9", "--out-worktree", str(linked))[1]["error"] == "NOT_FOUND"
        db_path.unlink()
        for side in ("-wal", "-shm"):
            Path(str(db_path) + side).unlink(missing_ok=True)
        assert run_cli("export-summary", "--task", "T1", "--out-worktree", str(linked))[1]["error"] == "NOT_FOUND"


def _shape(snap: dict[str, Any]) -> dict[str, Any]:
    """The key set: top-level keys, and the keys of each section's first item."""
    shape: dict[str, Any] = {}
    for key, value in snap.items():
        if isinstance(value, list):
            shape[key] = sorted(value[0]) if value else []
        elif isinstance(value, dict):
            shape[key] = sorted(value)
        else:
            shape[key] = type(value).__name__
    shape["runs.lease"] = sorted(snap["runs"][0]["lease"])
    return shape


@pytest.fixture
def busy_world(world, fake_clock) -> dict[str, Any]:
    w = world
    insert_node(w["db"], "N1", w["run"], "T1", session_id="S-n1", nonce="ab" * 16)
    msg = _file(w["tmp"], "m.json", '{"a": 1}')
    run_cli(
        "msg",
        "post",
        "--run",
        w["run"],
        "--to",
        "orchestrator",
        "--kind",
        "k",
        "--payload-file",
        msg,
        "--token",
        w["token"],
    )
    run_cli("question", "open", "--task", "T1", "--text-file", _file(w["tmp"], "q.md", "?"), "--token", w["token"])
    fields = _file(w["tmp"], "f.json", '{"tests": 1}')
    run_cli(
        "report",
        "post",
        "--task",
        "T1",
        "--phase",
        "build",
        "--round",
        "1",
        "--status",
        "continuing",
        "--summary",
        "s",
        "--fields-file",
        fields,
        "--token",
        w["token"],
    )
    sql(
        w["db"],
        "INSERT INTO integrations (item_task_id, deliverable_id, item_tip, result, at) VALUES ('T2', 'T1', 'abc', 'ok', 'x')",
    )
    sql(
        w["db"],
        "INSERT INTO tool_calls (tool, key, args_hash, run_id, task_id, actor, args, state, holder_pid, started_at) VALUES ('push', 'k', 'h', ?, 'T1', 'a', '{}', 'started', 5, ?)",
        [w["run"], "2026-10-03T11:59:00.000000Z"],
    )
    sql(
        w["db"],
        "INSERT INTO locks (name, holder_kind, holder_pid, holder_actor, token_subject, acquired_at) VALUES ('main-merge', 'process', 5, 'a', 'run:R1', 'x')",
    )
    sql(
        w["db"],
        "INSERT INTO cap_slots (cap, slot, holder_pid, holder_actor, acquired_at) VALUES ('gates', 0, 5, 'a', 'x')",
    )
    sql(
        w["db"],
        "INSERT INTO daemon_lease (id, pid, schema_version, started_at, heartbeat_at) VALUES (1, 9, 1, 'x', '2026-10-03T11:59:50.000000Z')",
    )
    sql(
        w["db"],
        "INSERT INTO hook_events (hook, session_id, decision, detail, at) VALUES ('stop', ?, 'alert', '{\"kind\": \"queue_not_draining\"}', 'x')",
        [ORCH],
    )
    return w


class TestSnapshot:
    def test_key_set_is_pinned(self, busy_world) -> None:
        code, snap = run_cli("snapshot")
        assert code == 0
        snap.pop("ok")
        assert tuple(snap) == tuple(sorted(snapshot.KEYS))
        assert _shape(snap) == json.loads((GOLDEN / "snapshot_keys.json").read_text())

    def test_no_nonce_ever(self, busy_world) -> None:
        blob = json.dumps(run_cli("snapshot")[1])
        for (nonce,) in sql(busy_world["db"], "SELECT nonce FROM run_leases UNION SELECT nonce FROM nodes"):
            assert nonce not in blob
        assert '"nonce"' not in blob

    def test_values(self, busy_world) -> None:
        snap = run_cli("snapshot", "--run", "r1")[1]
        assert snap["queues"] == [
            {"run_id": "R1", "recipient": "orchestrator", "depth": 3, "in_flight": 0, "dead": 0, "oldest_age_s": 0.0}
        ]
        assert snap["daemon"]["heartbeat_age_s"] == 10.0 and snap["daemon"]["code_schema_version"] == 1
        assert snap["tool_calls_in_flight"][0]["age_s"] == 60.0
        assert snap["alerts"][0]["detail"] == {"kind": "queue_not_draining"}
        assert snap["reports"][0]["fields"] == {"tests": 1}
        assert snap["digests"] == [{"task_id": "T1", "bytes": 28, "updated_at": "2026-10-03T12:00:00.000000Z"}]
        assert [t["id"] for t in snap["tasks"]] == ["T1", "T2"]
        open_run("r2", "S-2")
        assert [r["id"] for r in run_cli("snapshot", "--run", "R2")[1]["runs"]] == ["R2"]
        assert run_cli("snapshot", "--run", "nope")[1]["error"] == "NOT_FOUND"

    def test_missing_db_is_empty(self, db_path) -> None:
        code, snap = run_cli("snapshot", "--run", "R1")
        snap.pop("ok")
        assert code == 0 and snap == snapshot.empty() and tuple(snap) == tuple(sorted(snapshot.KEYS))

    def test_unparseable_json_columns_are_kept_as_text(self, busy_world) -> None:
        sql(busy_world["db"], "UPDATE hook_events SET detail = 'not json'")
        assert run_cli("snapshot")[1]["alerts"][0]["detail"] == "not json"


class TestTaskShow:
    def test_full_task_with_its_digest(self, busy_world) -> None:
        code, out = run_cli("task", "show", "--task", "T1")
        assert code == 0
        assert out["digest"]["body"] == "Built the core.\n\nAll green.\n"
        assert out["task"]["title"] == "The | Control Plane" and "nonce" not in out["nodes"][0]
        assert [c["state"] for c in out["criteria"]] == ["met", "open"] and out["depends_on"] == ["T2"]
        assert len(out["decisions"]) == 2 and len(out["reports"]) == 1 and len(out["questions"]) == 1
        assert [h["to_state"] for h in out["history"]] == ["proposed"]
        assert run_cli("task", "show", "--task", "T2")[1]["digest"] is None

    def test_unknown(self, world, db_path) -> None:
        assert run_cli("task", "show", "--task", "T9")[1]["error"] == "NOT_FOUND"
        db_path.unlink()
        for side in ("-wal", "-shm"):
            Path(str(db_path) + side).unlink(missing_ok=True)
        assert run_cli("task", "show", "--task", "T1")[1]["error"] == "NOT_FOUND"
