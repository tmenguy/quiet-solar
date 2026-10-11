"""Checkpoint 11: the per-merge export, ``snapshot`` and ``task show`` (AC1, AC16)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from control_plane import export, ledger, snapshot

from .conftest import CUR, ORCH, insert_node, open_run, run_cli, sql

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

    def test_empty_registry_renders_no_ledger(self, world, linked) -> None:
        export.LEDGER_SECTIONS.clear()
        run_cli("export-summary", "--task", "T1", "--out-worktree", str(linked))
        text = (linked / "docs" / "stories" / "QS-11.summary.md").read_text()
        assert text.endswith("## Ledger\n\nNo ledger recorded.\n")

    def test_register_export_is_idempotent(self) -> None:
        titles = [t for t, _ in export.LEDGER_SECTIONS]
        assert titles == ["Rounds", "Findings", "Blast radius"]
        ledger.register_export()
        assert [t for t, _ in export.LEDGER_SECTIONS] == titles
        export.LEDGER_SECTIONS.clear()
        ledger.register_export()
        assert [t for t, _ in export.LEDGER_SECTIONS] == titles

    def test_a_populated_ledger(self, world, linked) -> None:
        tok, tmp = world["token"], world["tmp"]
        ok = lambda *a: run_cli(*a, "--token", tok)  # noqa: E731
        ok("round", "start", "--task", "T1", "--phase", "plan")
        ok("round", "start", "--task", "T1", "--phase", "build", "--head", "h1", "--base", "b0")
        ok("round", "start", "--task", "T1", "--phase", "build", "--head", "h2")
        items = [
            {"severity": "must_fix", "category": "correctness", "title": "Off | by one", "body": "b", "symbol": "f"},
            {"severity": "should_fix", "category": "design", "title": "Rename", "body": "b", "symbol": "f"},
            {"severity": "nice_to_have", "category": "style", "title": "Nit", "body": "b", "symbol": "g"},
        ]
        opened = ok("finding", "open", "--task", "T1", "--phase", "build", "--source", "reviewer",
                    "--input", _file(tmp, "i.json", json.dumps(items)))  # fmt: skip
        assert opened[1]["ids"] == [1, 2, 3]
        ok("finding", "state", "1", "--to", "resolved", "--commit", "c1")
        ok("finding", "state", "2", "--to", "settled", "--reason", "keep the name")
        again = {"severity": "should_fix", "category": "test", "title": "Again", "body": "b", "symbol": "f"}
        ok("finding", "open", "--task", "T1", "--phase", "build", "--source", "reviewer",
           "--input", _file(tmp, "j.json", json.dumps(again)))  # fmt: skip
        ok("blast-radius", "set", "--task", "T1", "--value", "doubt", "--head-sha", "h1", "--review", "G1")
        ok("blast-radius", "set", "--task", "T1", "--value", "ok", "--head-sha", "h2", "--review", "G2",
           "--reason", "two files")  # fmt: skip
        assert run_cli("export-summary", "--task", "T1", "--out-worktree", str(linked))[0] == 0
        text = (linked / "docs" / "stories" / "QS-11.summary.md").read_text()
        ledger_md = text.split("## Ledger\n\n", 1)[1]
        assert ledger_md == (
            "### Rounds\n\n"
            "| phase | round | diff |\n|---|---|---|\n"
            "| plan | 1 | `?..?` |\n| build | 1 | `b0..h1` |\n| build | 2 | `h1..h2` |\n\n"
            "### Findings\n\n"
            "| # | phase / round | source | class | state | title | resolution | flags |\n"
            "|---|---|---|---|---|---|---|---|\n"
            "| 1 | build / 2 | reviewer | must_fix | resolved | Off \\| by one | c1 |  |\n"
            "| 2 | build / 2 | reviewer | should_fix | settled | Rename | keep the name |  |\n"
            "| 3 | build / 2 | reviewer | nice_to_have | open | Nit |  |  |\n"
            "| 4 | build / 2 | reviewer | should_fix | open | Again |  | overlaps_fix #1, overlaps_fix #2 |\n\n"
            "### Blast radius\n\n"
            "`ok` at `h2` (review G2) — two files\n"
        )
        ok("blast-radius", "set", "--task", "T1", "--value", "too_large", "--head-sha", "h3", "--review", "G3")
        run_cli("export-summary", "--task", "T1", "--out-worktree", str(linked))
        text = (linked / "docs" / "stories" / "QS-11.summary.md").read_text()
        assert text.endswith("### Blast radius\n\n`too_large` at `h3` (review G3)\n")

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
        "INSERT INTO daemon_lease (id, pid, schema_version, started_at, heartbeat_at) VALUES (1, 9, ?, 'x', '2026-10-03T11:59:50.000000Z')",
        [CUR],
    )
    sql(
        w["db"],
        "INSERT INTO hook_events (hook, session_id, decision, detail, at) VALUES ('stop', ?, 'alert', '{\"kind\": \"queue_not_draining\"}', 'x')",
        [ORCH],
    )
    sql(  # raw SQL: no message is posted, so the queue depth stays 3
        w["db"],
        "INSERT INTO alerts (run_id, kind, subject, fingerprint, payload, first_seen, last_seen)"
        " VALUES (?, 'overlap', 'T1|T2', 'overlap:x:1', '{\"files\": [\"a\"]}', 'x', 'x')",
        [w["run"]],
    )
    # #375: one row of each ledger table, so _shape pins their columns.
    ok = lambda *a: run_cli(*a, "--token", w["token"])  # noqa: E731
    assert ok("round", "start", "--task", "T1", "--phase", "build", "--head", "h1", "--base", "b0")[0] == 0
    item = _file(w["tmp"], "i.json", '{"severity": "should_fix", "category": "test", "title": "t", "body": "42"}')
    assert ok("finding", "open", "--task", "T1", "--phase", "build", "--source", "reviewer", "--input", item)[0] == 0
    assert ok("finding", "classify", "1", "--class", "must_fix")[0] == 0
    blast = ("blast-radius", "set", "--task", "T1", "--value", "ok", "--head-sha", "h1", "--review", "G1")
    assert ok(*blast)[0] == 0
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
        assert snap["daemon"]["heartbeat_age_s"] == 10.0 and snap["daemon"]["code_schema_version"] == CUR
        assert snap["tool_calls_in_flight"][0]["age_s"] == 60.0
        assert snap["hook_alerts"][0]["detail"] == {"kind": "queue_not_draining"}
        assert snap["alerts"][0]["payload"] == {"files": ["a"]} and snap["alerts"][0]["kind"] == "overlap"
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

    def test_the_ledger_tables(self, busy_world) -> None:
        snap = run_cli("snapshot")[1]
        assert [(r["phase"], r["round"], r["base_sha"], r["head_sha"]) for r in snap["rounds"]] == [
            ("build", 1, "b0", "h1")
        ]
        assert snap["findings"][0]["flags"] == [] and snap["findings"][0]["body"] == "42"  # never JSON-parsed
        assert [e["kind"] for e in snap["finding_events"]] == ["classify"]
        assert snap["blast_radius"][0]["value"] == "ok"

    def test_the_run_filter_on_the_ledger(self, busy_world) -> None:
        from .conftest import insert_task

        w = busy_world
        run2, tok2 = open_run("r2", "S-2")
        insert_task(w["db"], "T8", run2)
        insert_task(w["db"], "T9", None)  # no run: its findings show in the run that wrote them
        item = _file(w["tmp"], "k.json", '{"severity": "must_fix", "category": "test", "title": "u", "body": "b"}')
        for task, tok in (("T8", tok2), ("T9", tok2)):
            base = ("--task", task, "--token", tok)
            assert run_cli("round", "start", "--phase", "build", "--head", "h", *base)[0] == 0
            assert (
                run_cli("finding", "open", "--phase", "build", "--source", "reviewer", "--input", item, *base)[0] == 0
            )
            assert run_cli("blast-radius", "set", "--value", "ok", "--head-sha", "h", "--review", "G", *base)[0] == 0
        fid = run_cli("snapshot", "--run", "R2")[1]["findings"][0]["id"]
        assert run_cli("finding", "state", str(fid), "--to", "deferred", "--token", tok2)[0] == 0
        r1, r2, everything = (run_cli("snapshot", *a)[1] for a in (("--run", "R1"), ("--run", "R2"), ()))
        assert [r["task_id"] for r in r1["rounds"]] == ["T1"] and [r["task_id"] for r in r2["rounds"]] == ["T8"]
        assert [f["task_id"] for f in r1["findings"]] == ["T1"]
        assert [f["task_id"] for f in r2["findings"]] == ["T8", "T9"] and r2["findings"][1]["run_id"] == "R2"
        assert [e["finding_id"] for e in r1["finding_events"]] == [1]
        assert [e["finding_id"] for e in r2["finding_events"]] == [fid]
        assert [b["task_id"] for b in r1["blast_radius"]] == ["T1"] and [b["task_id"] for b in r2["blast_radius"]] == [
            "T8"
        ]
        assert (
            len(everything["findings"]) == 3 and len(everything["rounds"]) == 3 and len(everything["blast_radius"]) == 3
        )

    def test_unparseable_json_columns_are_kept_as_text(self, busy_world) -> None:
        sql(busy_world["db"], "UPDATE hook_events SET detail = 'not json'")
        assert run_cli("snapshot")[1]["hook_alerts"][0]["detail"] == "not json"


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

    def test_the_ledger_in_task_show(self, busy_world) -> None:
        out = run_cli("task", "show", "--task", "T1")[1]
        assert [r["round"] for r in out["rounds"]] == [1] and [f["id"] for f in out["findings"]] == [1]
        assert out["findings"][0]["body"] == "42" and out["findings"][0]["classification"] == "must_fix"
        assert [e["kind"] for e in out["finding_events"]] == ["classify"]
        assert [b["review"] for b in out["blast_radius"]] == ["G1"]
        assert out["convergence"]["build"] == {
            "converged": False,
            "consistent": False,
            "blocking": [1],
            "latest_round": 1,
            "counts": {"open": 1, "resolved": 0, "rejected": 0, "deferred": 0, "settled": 0},
        }
        assert out["convergence"]["plan"]["latest_round"] == 0
        empty = run_cli("task", "show", "--task", "T2")[1]
        assert empty["findings"] == [] and empty["rounds"] == [] and empty["blast_radius"] == []

    def test_unknown(self, world, db_path) -> None:
        assert run_cli("task", "show", "--task", "T9")[1]["error"] == "NOT_FOUND"
        db_path.unlink()
        for side in ("-wal", "-shm"):
            Path(str(db_path) + side).unlink(missing_ok=True)
        assert run_cli("task", "show", "--task", "T1")[1]["error"] == "NOT_FOUND"


def test_snapshot_orders_text_ids_by_insertion(world, fake_clock) -> None:
    """F1 sweep: ``ORDER BY id`` on ``T<n>`` text ids put T10 before T9."""
    from .conftest import insert_task

    for task_id in ("T9", "T10"):
        insert_task(world["db"], task_id, world["run"])
    ids = [t["id"] for t in run_cli("snapshot")[1]["tasks"]]
    assert ids.index("T9") < ids.index("T10")


# --------------------------------------------------------------------------- review fix #02 (G19)


def test_task_show_orders_questions_by_creation_then_insertion(migrated) -> None:
    from .conftest import insert_task, open_run, run_cli, sql

    run_id, token = open_run()
    insert_task(migrated, "T1", run_id)
    for qid in ("Q9", "Q10"):  # same created_at: insertion order, never the TEXT id ("Q10" < "Q9")
        sql(
            migrated,
            "INSERT INTO questions (id, run_id, task_id, text, state, created_at) VALUES (?, ?, 'T1', 'q', 'open', 'x')",
            [qid, run_id],
        )
    code, out = run_cli("task", "show", "--task", "T1")
    assert code == 0 and [q["id"] for q in out["questions"]] == ["Q9", "Q10"]
