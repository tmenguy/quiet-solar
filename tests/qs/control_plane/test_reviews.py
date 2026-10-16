"""#407: review sessions — ``request_locked`` (T4), the ``review`` tool (T5), ``finish`` and ``outcome`` (T8)."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from control_plane import db, errors, paths, reviews, schema_reviews

from .conftest import insert_task, open_run, sql


@pytest.fixture
def two_runs(migrated: Path) -> dict[str, Any]:
    """R1: deliverables T1, T2 (T2 split into item T3) and a merged T4; R2: deliverable T5 and an epic T6."""
    r1, tok1 = open_run()
    r2, tok2 = open_run("r2", "S-2")
    insert_task(migrated, "T1", r1, is_deliverable=1, branch="QS_1")
    insert_task(migrated, "T2", r1, is_deliverable=1, branch="QS_2")
    insert_task(migrated, "T3", r1, deliverable_id="T2", item_k=1, branch="QS_2_1")
    insert_task(migrated, "T4", r1, state="merged", is_deliverable=1, branch="QS_4", merge_sha="m" * 40)
    insert_task(migrated, "T5", r2, is_deliverable=1, branch="QS_5")
    insert_task(migrated, "T6", r2, kind="epic")
    return {"db": migrated, "r1": r1, "r2": r2, "tok1": tok1, "tok2": tok2}


def request(conn: sqlite3.Connection, clock: Any, **kw: Any) -> reviews.Requested:
    kw.setdefault("trigger", {"why": "test"})
    kw.setdefault("actor", "orchestrator")
    kw.setdefault("tool_key", "k1")
    kw.setdefault("grant", False)
    with db.write(conn):
        return reviews.request_locked(conn, clock, **kw)


def review_row(path: Path, rid: str) -> dict[str, Any]:
    return dict(sql(path, "SELECT * FROM reviews WHERE id = ?", [rid])[0])


def triggers(path: Path) -> list[tuple[Any, ...]]:
    return [tuple(r) for r in sql(path, "SELECT review_id, actor, tool_key, outcome FROM review_triggers ORDER BY id")]


def scope_of(path: Path, rid: str) -> list[tuple[str, str]]:
    return [
        tuple(r) for r in sql(path, "SELECT task_id, run_id FROM review_tasks WHERE review_id = ? ORDER BY 1", [rid])
    ]


# --------------------------------------------------------------------------- T4: request_locked


class TestGlobalScope:
    def test_created_on_the_run_token_path(self, two_runs, conn, fake_clock) -> None:
        out = request(conn, fake_clock, kind="global", run_id="R1")
        assert out == reviews.Created("V1", ["T1", "T2", "T3"])  # deliverables and items, not the merged T4
        row = review_row(two_runs["db"], "V1")
        assert (row["kind"], row["coalesce_key"], row["home_run_id"], row["state"]) == (
            "global",
            "global:R1",
            "R1",
            "launching",
        )
        assert len(row["nonce"]) == 32 and row["grant_nonce"] is None
        assert (row["agent"], row["permission_mode"], row["tool_key"], row["requested_by"]) == (
            "qs-global-review",
            "dontAsk",
            "review/k1",
            "orchestrator",
        )
        assert row["name"] == f"qs-review-{reviews.db_tag()}-V1" and row["heads_at"] is None
        assert scope_of(two_runs["db"], "V1") == [("T1", "R1"), ("T2", "R1"), ("T3", "R1")]
        pending = sql(two_runs["db"], "SELECT base_sha, head_sha, head_state, outcome FROM review_tasks")
        assert {tuple(r) for r in pending} == {(None, None, None, None)}
        assert triggers(two_runs["db"]) == [("V1", "orchestrator", "k1", "created")]

    def test_created_on_the_grant_path(self, two_runs, conn, fake_clock) -> None:
        out = request(conn, fake_clock, kind="global", run_id="R1", grant=True, actor="cp:review_pilot")
        assert isinstance(out, reviews.Created)
        row = review_row(two_runs["db"], "V1")
        assert row["state"] == "requested" and row["nonce"] is None and len(row["grant_nonce"]) == 32

    @pytest.mark.parametrize("n", [0, 1])
    def test_fewer_than_two_tasks_is_skipped(self, two_runs, conn, fake_clock, n: int) -> None:
        keep = ["T1", "T2", "T3"][:n]
        for task in {"T1", "T2", "T3"} - set(keep):
            sql(two_runs["db"], "UPDATE tasks SET state = 'dropped' WHERE id = ?", [task])
        out = request(conn, fake_clock, kind="global", run_id="R1")
        assert out == reviews.Skipped("single_task", keep)
        assert sql(two_runs["db"], "SELECT count(*) FROM reviews")[0][0] == 0 and triggers(two_runs["db"]) == []

    def test_two_tasks_including_an_item_is_enough(self, two_runs, conn, fake_clock) -> None:
        sql(two_runs["db"], "UPDATE tasks SET state = 'validated' WHERE id = 'T1'")
        assert request(conn, fake_clock, kind="global", run_id="R1") == reviews.Created("V1", ["T2", "T3"])

    def test_global_needs_a_run_and_a_known_kind(self, two_runs, conn, fake_clock) -> None:
        for kw in ({"kind": "global"}, {"kind": "final", "run_id": "R1"}):
            with pytest.raises(errors.CpError) as exc:
                request(conn, fake_clock, **kw)
            assert exc.value.code == "USAGE"


class TestCrossRunScope:
    def test_an_alert_event_across_two_runs(self, two_runs, conn, fake_clock) -> None:
        out = request(conn, fake_clock, kind="cross_run", tasks=["T5", "T1", "T1"], grant=True)
        assert out == reviews.Created("V1", ["T1", "T5"])
        row = review_row(two_runs["db"], "V1")
        assert (row["coalesce_key"], row["home_run_id"], row["agent"]) == (
            "cross_run:T1|T5",
            None,
            "qs-cross-run-review",
        )
        assert scope_of(two_runs["db"], "V1") == [("T1", "R1"), ("T5", "R2")]

    def test_closed_runs_and_unknown_tasks_are_dropped(self, two_runs, conn, fake_clock) -> None:
        r3, _ = open_run("r3", "S-3")
        insert_task(two_runs["db"], "T7", r3, is_deliverable=1)
        insert_task(two_runs["db"], "T8", None, is_deliverable=1)
        sql(two_runs["db"], "UPDATE runs SET state = 'closed' WHERE id = ?", [r3])
        out = request(conn, fake_clock, kind="cross_run", tasks=["T1", "T5", "T7", "T8", "T99"])
        assert out == reviews.Created("V1", ["T1", "T5"])
        assert review_row(two_runs["db"], "V1")["coalesce_key"] == "cross_run:T1|T5"  # the kept set's key

    def test_an_alert_event_left_in_one_open_run_is_skipped(self, two_runs, conn, fake_clock) -> None:
        sql(two_runs["db"], "UPDATE runs SET state = 'closed' WHERE id = 'R2'")
        assert request(conn, fake_clock, kind="cross_run", tasks=["T1", "T2", "T5"]) == reviews.Skipped(
            "single_run", ["T1", "T2"]
        )

    def test_nothing_non_terminal_left_is_skipped(self, two_runs, conn, fake_clock) -> None:
        sql(two_runs["db"], "UPDATE tasks SET state = 'merged' WHERE id IN ('T1', 'T5')")
        assert request(conn, fake_clock, kind="cross_run", tasks=["T1", "T5"]) == reviews.Skipped(
            "no_open_task", ["T1", "T5"]
        )

    def test_a_merge_event_keeps_its_merged_task_after_its_run_closed(self, two_runs, conn, fake_clock) -> None:
        sql(two_runs["db"], "UPDATE runs SET state = 'closed' WHERE id = 'R1'")
        merge = {"event": "merge:T4:9", "merged_task": "T4"}
        out = request(conn, fake_clock, kind="cross_run", tasks=["T4", "T5", "T1"], trigger=merge, grant=True)
        assert out == reviews.Created("V1", ["T4", "T5"])  # T1's run closed: dropped; T4 merged: kept

    def test_a_merge_event_needs_another_open_run(self, two_runs, conn, fake_clock) -> None:
        merge = {"event": "merge:T4:9", "merged_task": "T4"}
        out = request(conn, fake_clock, kind="cross_run", tasks=["T4", "T1"], trigger=merge)
        assert out == reviews.Skipped("no_other_run", ["T1", "T4"])
        sql(two_runs["db"], "UPDATE tasks SET state = 'dropped' WHERE id = 'T5'")
        out = request(conn, fake_clock, kind="cross_run", tasks=["T4", "T5"], trigger=merge)
        assert out == reviews.Skipped("no_open_task", ["T4", "T5"])

    def test_a_non_string_merged_task_is_an_alert_event(self, two_runs, conn, fake_clock) -> None:
        out = request(conn, fake_clock, kind="cross_run", tasks=["T1", "T5"], trigger={"merged_task": 7})
        assert isinstance(out, reviews.Created)


class TestJoinAndDefer:
    def test_join_while_the_heads_are_not_captured(self, two_runs, conn, fake_clock) -> None:
        assert isinstance(request(conn, fake_clock, kind="global", run_id="R1"), reviews.Created)
        out = request(conn, fake_clock, kind="global", run_id="R1", tool_key="k2", actor="orchestrator")
        assert out == reviews.Joined("V1")
        assert triggers(two_runs["db"]) == [
            ("V1", "orchestrator", "k1", "created"),
            ("V1", "orchestrator", "k2", "joined"),
        ]
        assert sql(two_runs["db"], "SELECT count(*) FROM reviews")[0][0] == 1

    def test_a_requested_review_is_joined_too(self, two_runs, conn, fake_clock) -> None:
        request(conn, fake_clock, kind="cross_run", tasks=["T1", "T5"], grant=True)
        assert request(conn, fake_clock, kind="cross_run", tasks=["T5", "T1"], tool_key="k2") == reviews.Joined("V1")

    def test_deferred_once_the_heads_are_captured(self, two_runs, conn, fake_clock) -> None:
        request(conn, fake_clock, kind="global", run_id="R1")
        sql(two_runs["db"], "UPDATE reviews SET heads_at = 'x'")
        assert request(conn, fake_clock, kind="global", run_id="R1", tool_key="k2") == reviews.Deferred("V1")
        assert triggers(two_runs["db"]) == [("V1", "orchestrator", "k1", "created")]  # nothing written

    def test_a_running_review_defers(self, two_runs, conn, fake_clock) -> None:
        request(conn, fake_clock, kind="global", run_id="R1")
        sql(two_runs["db"], "UPDATE reviews SET state = 'running', heads_at = 'x'")
        assert request(conn, fake_clock, kind="global", run_id="R1", tool_key="k2") == reviews.Deferred("V1")

    def test_a_global_scope_that_changed_defers(self, two_runs, conn, fake_clock) -> None:
        request(conn, fake_clock, kind="global", run_id="R1")
        insert_task(two_runs["db"], "T9", "R1", deliverable_id="T2", item_k=2)  # a split meanwhile
        assert request(conn, fake_clock, kind="global", run_id="R1", tool_key="k2") == reviews.Deferred("V1")

    @pytest.mark.parametrize("state", schema_reviews.CLOSED)
    def test_a_closed_review_frees_the_key(self, two_runs, conn, fake_clock, state: str) -> None:
        request(conn, fake_clock, kind="global", run_id="R1")
        sql(two_runs["db"], "UPDATE reviews SET state = ?", [state])
        assert request(conn, fake_clock, kind="global", run_id="R1", tool_key="k2") == reviews.Created(
            "V2", ["T1", "T2", "T3"]
        )

    def test_the_pilot_queue_row_is_updated_not_inserted(self, two_runs, conn, fake_clock) -> None:
        db_path = two_runs["db"]

        def queued(key: str) -> int:
            sql(
                db_path,
                'INSERT INTO review_triggers (actor, "trigger", tool_key, outcome, at)'
                " VALUES ('cp:review_pilot', '{}', ?, 'queued', 'x')",
                [key],
            )
            return int(sql(db_path, "SELECT max(id) FROM review_triggers")[0][0])

        pilot = {"actor": "cp:review_pilot", "grant": True}
        a = queued("pilot:e1:1")
        out = request(
            conn, fake_clock, kind="cross_run", tasks=["T1", "T5"], tool_key="pilot:e1:1", trigger_row_id=a, **pilot
        )
        assert out == reviews.Created("V1", ["T1", "T5"])
        assert review_row(db_path, "V1")["tool_key"] == "review/pilot:e1:1"
        b = queued("pilot:e2:1")
        out = request(
            conn, fake_clock, kind="cross_run", tasks=["T1", "T5"], tool_key="pilot:e2:1", trigger_row_id=b, **pilot
        )
        assert out == reviews.Joined("V1")
        sql(db_path, "UPDATE reviews SET heads_at = 'x'")
        c = queued("pilot:e3:1")
        out = request(
            conn, fake_clock, kind="cross_run", tasks=["T1", "T5"], tool_key="pilot:e3:1", trigger_row_id=c, **pilot
        )
        assert out == reviews.Deferred("V1")
        d = queued("pilot:e4:1")
        out = request(
            conn, fake_clock, kind="cross_run", tasks=["T1"], tool_key="pilot:e4:1", trigger_row_id=d, **pilot
        )
        assert out == reviews.Skipped("single_run", ["T1"])
        assert triggers(db_path) == [
            ("V1", "cp:review_pilot", "pilot:e1:1", "created"),
            ("V1", "cp:review_pilot", "pilot:e2:1", "joined"),
            (None, "cp:review_pilot", "pilot:e3:1", "queued"),  # deferred: left in the queue
            (None, "cp:review_pilot", "pilot:e4:1", "skipped"),
        ]


class TestReviewDir:
    def test_layout_and_modes(self, migrated, tmp_path: Path) -> None:
        d = reviews.review_dir("V3")
        root = (tmp_path / "reviews").resolve()
        assert d == root / reviews.db_tag() / "V3"
        for level in (root, root / reviews.db_tag(), d, d / "in"):
            assert level.is_dir() and os.stat(level).st_mode & 0o777 == 0o700
        assert reviews.review_dir("V3", create=False) == d

    def test_the_tag_is_the_selected_dbs(self, migrated, monkeypatch, tmp_path: Path) -> None:
        first = reviews.db_tag()
        monkeypatch.setenv("QS_CP_DB", str(tmp_path / "other.db"))
        assert reviews.db_tag() != first and len(first) == 8
        assert not reviews.review_dir("V1", create=False).exists()

    def test_inside_a_checkout_is_refused(self, monkeypatch, fake_main: Path) -> None:
        monkeypatch.setenv("QS_CP_REVIEW_DIR", str(fake_main / "reviews"))
        with pytest.raises(errors.CpError) as exc:
            reviews.review_dir("V1")
        assert exc.value.code == "POLICY_REFUSED"
        assert paths.review_root  # the same guard as every state directory


# --------------------------------------------------------------------------- T5: the `review` tool

M, M2 = "a0" * 20, "a2" * 20
H1, H2, H3, H5, H6 = ("b1" * 20, "b2" * 20, "b3" * 20, "b5" * 20, "b6" * 20)
B1, B2, B3, B5, B6, B12 = ("c1" * 20, "c2" * 20, "c3" * 20, "c5" * 20, "c6" * 20, "cc" * 20)
MS, P6 = "d6" * 20, "e6" * 20


class FakeGit:
    """``git -C <MAIN> …`` answered from refs, objects, parents and merge bases (only ``0`` means found)."""

    def __init__(self, runner: Any, main: Path) -> None:
        self.refs = {
            "refs/heads/main": M,
            "refs/remotes/origin/main": M,
            "refs/heads/QS_1": H1,
            "refs/heads/QS_2": H2,
            "refs/heads/QS_2_1": H3,
            "refs/heads/QS_5": H5,
            "refs/heads/QS_6": H6,
        }
        self.objects = {MS}
        self.parents = {MS: P6}
        self.bases = {(M, H1): B1, (M, H2): B2, (H2, H3): B3, (M, H5): B5, (M, H6): B6, (M2, H1): B12}
        self.ancestors: set[tuple[str, str]] = set()
        self.fail: dict[str, int] = {}
        self.calls: list[list[str]] = []
        self.on_call: Any = None
        runner.on(["git", "-C", str(main)], self)

    def __call__(self, call: Any) -> Any:
        from control_plane.runner import RunResult

        args = call.argv[3:]
        self.calls.append(args)
        if self.on_call is not None:
            self.on_call(args)
        op = args[0]
        if op in self.fail:
            return RunResult(self.fail[op], "", "boom")
        if op == "cat-file":
            return RunResult(0 if args[2].split("^")[0] in self.objects else 128, "", "")
        if op == "rev-parse":
            target = args[-1].removesuffix("^{commit}")
            sha = self.parents.get(target[:-2]) if target.endswith("^1") else self.refs.get(target)
            return RunResult(0, f"{sha}\n", "") if sha else RunResult(1, "", "")
        if args[1] == "--is-ancestor":  # merge-base
            return RunResult(0 if (args[2], args[3]) in self.ancestors else 1, "", "")
        sha = self.bases.get((args[1], args[2]))
        return RunResult(0, f"{sha}\n", "") if sha else RunResult(1, "", "")


@pytest.fixture
def rw(migrated: Path, fake_main: Path, fake_runner, fake_claude, fake_clock, fake_probe, deps, tmp_path: Path):
    """R1: deliverables T1, T2 (item T3) and T4 (no branch); R2: deliverable T5 and a merged T6."""
    r1, tok1 = open_run()
    r2, tok2 = open_run("r2", "S-2")
    insert_task(migrated, "T1", r1, is_deliverable=1, branch="QS_1", title="first")
    insert_task(migrated, "T2", r1, is_deliverable=1, branch="QS_2")
    insert_task(migrated, "T3", r1, deliverable_id="T2", item_k=1, branch="QS_2_1")
    insert_task(migrated, "T4", r1, is_deliverable=1)
    insert_task(migrated, "T5", r2, is_deliverable=1, branch="QS_5")
    insert_task(migrated, "T6", r2, state="merged", is_deliverable=1, branch="QS_6", merge_sha=MS)
    agents = fake_main / ".claude" / "agents"
    agents.mkdir(parents=True)
    for stem in schema_reviews.AGENTS.values():
        (agents / f"{stem}.md").write_text(f"---\nname: {stem}\n---\nbody\n")
    git = FakeGit(fake_runner, fake_main)
    launch = {"exit": 0, "list": True, "on_launch": None, "seq": 0}

    def claude_bg(call: Any) -> Any:
        from control_plane.runner import RunResult

        from .conftest import agent

        name = call.argv[call.argv.index("-n") + 1]
        if launch["on_launch"] is not None:
            launch["on_launch"](call)
        if launch["exit"] == 0 and launch["list"]:
            launch["seq"] += 1
            fake_claude.listing = [*(fake_claude.listing or []), agent(f"S-{name}", name, id=f"short-{launch['seq']}")]
        return RunResult(launch["exit"], "", "")

    fake_runner.on(["claude", "--bg"], claude_bg)
    fake_runner.calls.clear()
    return {
        "db": migrated,
        "main": fake_main,
        "tok1": tok1,
        "tok2": tok2,
        "git": git,
        "launch": launch,
        "runner": fake_runner,
        "claude": fake_claude,
        "clock": fake_clock,
        "probe": fake_probe,
        "deps": deps,
        "tmp": tmp_path,
    }


def tool_review(rw: dict[str, Any], key: str, token: str, **args: Any) -> tuple[int, Any]:
    from .conftest import run_cli

    f = rw["tmp"] / f"review-args-{key.replace(':', '_').replace('/', '_')}.json"
    f.write_text(__import__("json").dumps(args))
    return run_cli("tool", "review", "--key", key, "--args-file", str(f), "--token", token)


def global_r1(rw: dict[str, Any], key: str = "task:R1:global:1", **extra: Any) -> tuple[int, Any]:
    return tool_review(rw, key, rw["tok1"], kind="global", run="R1", **extra)


def launches(rw: dict[str, Any]) -> list[Any]:
    return rw["runner"].matching("claude", "--bg")


def grant_request(rw: dict[str, Any], *, kind: str = "cross_run", key: str = "pilot:alert:ev:1", **kw: Any) -> str:
    conn = db.connect(rw["db"])
    try:
        if kind == "cross_run":
            kw.setdefault("tasks", ["T1", "T5"])
        else:
            kw.setdefault("run_id", "R1")
        out = request(conn, rw["clock"], kind=kind, grant=True, actor="cp:review_pilot", tool_key=key, **kw)
    finally:
        conn.close()
    assert isinstance(out, reviews.Created)
    row = review_row(rw["db"], out.review_id)
    return f"review:{row['id']}.0.{row['grant_nonce']}"


def heads(path: Path, rid: str = "V1") -> dict[str, tuple[Any, ...]]:
    rows = sql(path, "SELECT task_id, base_sha, head_sha, head_state FROM review_tasks WHERE review_id = ?", [rid])
    return {r[0]: tuple(r[1:]) for r in rows}


def review_alerts(path: Path) -> list[tuple[Any, ...]]:
    return [tuple(r) for r in sql(path, "SELECT run_id, subject FROM alerts WHERE kind = 'review_failed' ORDER BY id")]


class TestLaunch:
    def test_a_global_review_through_the_run_token(self, rw, monkeypatch) -> None:
        from .conftest import models

        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "S-caller")
        code, out = global_r1(rw)
        assert code == 0, out
        name = f"qs-review-{reviews.db_tag()}-V1"
        assert out["result"] == {
            "review_id": "V1",
            "kind": "global",
            "name": name,
            "session_id": f"S-{name}",
            "state": "running",
            "tasks": [
                {"task_id": "T1", "run_id": "R1", "base_sha": B1, "head_sha": H1, "head_state": "branch"},
                {"task_id": "T2", "run_id": "R1", "base_sha": B2, "head_sha": H2, "head_state": "branch"},
                {"task_id": "T3", "run_id": "R1", "base_sha": B3, "head_sha": H3, "head_state": "branch"},
                {"task_id": "T4", "run_id": "R1", "base_sha": None, "head_sha": None, "head_state": "plan"},
            ],
        }
        assert "token" not in out and "nonce" not in __import__("json").dumps(out)
        row = review_row(rw["db"], "V1")
        model, effort = models.spawn_policy("qs-global-review", None)
        directory = reviews.review_dir("V1", create=False)
        assert (row["state"], row["model"], row["effort"], row["dir"], row["short_id"]) == (
            "running",
            model,
            effort,
            str(directory),
            "short-1",
        )
        assert row["heads_at"] and row["launch_at"] and row["session_id"] == f"S-{name}"
        assert (directory / ".claude" / "agents" / "qs-global-review.md").read_text().startswith("---\nname:")
        call = dict(sql(rw["db"], "SELECT * FROM tool_calls WHERE tool = 'review'")[0])
        assert (call["task_id"], call["run_id"], call["state"], call["actor"]) == (
            None,
            "R1",
            "succeeded",
            "orchestrator",
        )
        [launch] = launches(rw)
        assert launch.cwd == str(directory) and launch.detach
        assert "QS_CP_TOKEN" in launch.env_remove and "CLAUDE_CODE_SESSION_ID" in launch.env_remove

    def test_the_launch_argv_in_dont_ask(self, rw) -> None:
        import json as _json
        import sys

        from control_plane import hooks

        assert global_r1(rw)[0] == 0
        row = review_row(rw["db"], "V1")
        [launch] = launches(rw)
        argv = launch.argv
        main, directory, python = rw["main"], Path(row["dir"]), sys.executable  # no <MAIN>/venv here
        prefix = f"{python} {main}/scripts/qs/cp.py"
        allow = [
            f"Read(/{directory}/in/**)",
            f"Write(/{directory}/in/**)",
            *(
                f"Bash({prefix} {cmd}:*)"
                for cmd in ("snapshot", "task show", "ledger show", "review read", "finding open", "review finish")
            ),
        ]
        assert allow[0].startswith("Read(//")  # the absolute-path form
        assert argv[:10] == [
            "claude",
            "--bg",
            "--agent",
            "qs-global-review",
            "-n",
            row["name"],
            "--model",
            row["model"],
            "--permission-mode",
            "dontAsk",
        ]
        assert argv[10] == "--allowedTools=" + ",".join(allow)  # one word: never variadic
        assert argv[11] == "--settings"
        settings = _json.loads(argv[12])
        expected = {**hooks.hooks_settings("node", main), "env": {"QS_CP_DB": str(paths.select_db().resolve())}}
        if row["effort"]:
            expected["effortLevel"] = row["effort"]
        assert settings == expected
        assert len(argv) == 14
        text = argv[13]
        token = f"review:V1.1.{row['nonce']}"
        for fact in (
            "You are review V1 (global)",
            f"prefix (exactly this, one command per Bash call): {prefix}",
            f"{directory / 'in'}",
            token,
            "Your finding source: global_review",
            "Deadline: 2026-10-03T13:00:00.000000Z",
            f'- T1 | R1 | "first" | QS_1 | {B1} | {H1} | branch',
            '- T4 | R1 | "T4" | - | - | - | plan',
        ):
            assert fact in text, fact

    def test_the_venv_python_wins(self, rw) -> None:
        venv = rw["main"] / "venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").write_text("")
        assert global_r1(rw)[0] == 0
        assert f"Bash({venv / 'python'} {rw['main']}/scripts/qs/cp.py snapshot:*)" in launches(rw)[0].argv[10]

    def test_the_auto_fallback(self, rw, monkeypatch) -> None:
        monkeypatch.setattr(reviews, "REVIEW_PERMISSION_MODE", "auto")
        assert global_r1(rw)[0] == 0
        argv = launches(rw)[0].argv
        assert argv[8:11] == ["--permission-mode", "auto", "--disallowedTools=Edit,NotebookEdit"]
        assert review_row(rw["db"], "V1")["permission_mode"] == "auto"

    def test_launch_args_and_allowed_tools(self) -> None:
        d = Path("/x/V1")
        assert reviews.allowed_tools(Path("/m"), "/m/venv/bin/python", d)[:2] == [
            "Read(//x/V1/in/**)",
            "Write(//x/V1/in/**)",
        ]
        assert reviews.launch_args("auto", Path("/m"), "/p", d) == [
            "--permission-mode",
            "auto",
            "--disallowedTools=Edit,NotebookEdit",
        ]
        with pytest.raises(errors.CpError) as exc:
            reviews.launch_args("bypass", Path("/m"), "/p", d)
        assert exc.value.code == "USAGE"
        for main, python, directory in (
            (Path("/a b"), "/p", d),
            (Path("/m"), "/p;x", d),
            (Path("/m"), "/p", Path("/x/$(y)")),
            (Path("/m"), "/p`", d),
        ):
            with pytest.raises(errors.CpError) as exc:
                reviews.allowed_tools(main, python, directory)
            assert exc.value.code == "POLICY_REFUSED"

    def test_a_path_with_whitespace_fails_the_review(self, rw, monkeypatch) -> None:
        monkeypatch.setattr(reviews, "python_for", lambda main: "/a b/python")
        code, out = global_r1(rw)
        assert code == 1 and out["error"] == "TOOL_FAILED" and "whitespace" in out["detail"]
        assert review_row(rw["db"], "V1")["state"] == "failed" and launches(rw) == []
        assert review_alerts(rw["db"]) == [("R1", "review:V1:failed")]

    def test_the_effort_from_the_resolver(self, rw) -> None:
        import json as _json

        rw["deps"].resolve_model = lambda stem, lane: (f"m-{stem}", None)
        assert global_r1(rw)[0] == 0
        argv = launches(rw)[0].argv
        assert argv[argv.index("--model") + 1] == "m-qs-global-review"
        assert "effortLevel" not in _json.loads(argv[argv.index("--settings") + 1])
        rw["deps"].resolve_model = lambda stem, lane: ("m", "high")
        sql(rw["db"], "UPDATE reviews SET state = 'finished'")
        assert global_r1(rw, "task:R1:global:2")[0] == 0
        argv = launches(rw)[1].argv
        assert _json.loads(argv[argv.index("--settings") + 1])["effortLevel"] == "high"
        assert review_row(rw["db"], "V2")["effort"] == "high"

    def test_a_running_review_never_counts_against_the_node_cap(self, rw) -> None:
        from control_plane import locks

        assert global_r1(rw)[0] == 0
        conn = db.connect(rw["db"])
        try:
            with db.write(conn):
                assert locks.admit_node(conn, rw["clock"], listing=[], holders_alive={}, limit=1) == 0
        finally:
            conn.close()


class TestHeads:
    def test_merged_unresolved_and_a_cross_run_scope(self, rw) -> None:
        sql(rw["db"], "UPDATE tasks SET branch = 'QS_404' WHERE id = 'T1'")
        token = grant_request(rw, tasks=["T1", "T5", "T6"])
        assert tool_review(rw, "pilot:alert:ev:1", token, review="V1")[0] == 0
        assert heads(rw["db"]) == {
            "T1": (None, None, "unresolved"),
            "T5": (B5, H5, "branch"),
            "T6": (P6, MS, "merged"),
        }

    def test_a_merge_commit_not_present_locally_falls_through(self, rw) -> None:
        rw["git"].objects.clear()
        token = grant_request(rw, tasks=["T1", "T6"])
        assert tool_review(rw, "pilot:alert:ev:1", token, review="V1")[0] == 0
        assert heads(rw["db"])["T6"] == (B6, H6, "branch")

    def test_a_malformed_merge_sha_is_never_passed_to_git(self, rw) -> None:
        sql(rw["db"], "UPDATE tasks SET merge_sha = '--output=/tmp/x' WHERE id = 'T6'")
        token = grant_request(rw, tasks=["T1", "T6"])
        assert tool_review(rw, "pilot:alert:ev:1", token, review="V1")[0] == 0
        assert heads(rw["db"])["T6"] == (B6, H6, "branch")
        assert not [c for c in rw["git"].calls if c[0] == "cat-file"]

    def test_a_work_item_falls_back_to_main_when_its_deliverable_branch_is_gone(self, rw) -> None:
        del rw["git"].refs["refs/heads/QS_2"]
        rw["git"].bases[(M, H3)] = "f3" * 20
        assert global_r1(rw)[0] == 0
        assert heads(rw["db"])["T3"] == ("f3" * 20, H3, "branch")
        assert heads(rw["db"])["T2"] == (None, None, "unresolved")

    @pytest.mark.parametrize(
        ("local", "remote", "ancestors", "expected"),
        [
            (M, M2, {(M, M2)}, M2),  # local main behind origin/main: origin/main's fork point
            (M2, M, {(M, M2)}, M2),  # origin/main behind local main: local
            (M, M2, set(), M2),  # diverged: origin/main
            (None, M2, set(), M2),  # only origin/main
            (M2, None, set(), M2),  # only main
        ],
    )
    def test_the_main_base_is_child_14s_rule(self, rw, local, remote, ancestors, expected) -> None:
        refs = rw["git"].refs
        for ref, sha in (("refs/heads/main", local), ("refs/remotes/origin/main", remote)):
            if sha is None:
                del refs[ref]
            else:
                refs[ref] = sha
        rw["git"].ancestors = ancestors
        assert global_r1(rw)[0] == 0
        assert heads(rw["db"])["T1"] == ({M: B1, M2: B12}[expected], H1, "branch")

    def test_no_main_at_all_leaves_the_base_empty(self, rw) -> None:
        del rw["git"].refs["refs/heads/main"]
        del rw["git"].refs["refs/remotes/origin/main"]
        assert global_r1(rw)[0] == 0
        assert heads(rw["db"])["T1"] == (None, H1, "branch")

    @pytest.mark.parametrize(("op", "code"), [("rev-parse", 124), ("merge-base", 127), ("cat-file", 124)])
    def test_an_aborting_git_fails_the_step_and_records_no_head(self, rw, op: str, code: int) -> None:
        rw["git"].fail[op] = code
        token = grant_request(rw, tasks=["T1", "T5", "T6"])
        code_, out = tool_review(rw, "pilot:alert:ev:1", token, review="V1")
        assert code_ == 1 and out["error"] == "TOOL_FAILED" and f"exited {code}" in out["detail"]
        assert {h[2] for h in heads(rw["db"]).values()} == {None}
        row = review_row(rw["db"], "V1")
        assert row["state"] == "failed" and row["heads_at"] and f"exited {code}" in row["error"]
        assert review_alerts(rw["db"]) == [("R1", "review:V1:failed"), ("R2", "review:V1:failed")]
        assert launches(rw) == []

    def test_the_capture_marker_is_stamped_before_git_runs(self, rw) -> None:
        seen: list[Any] = []

        def interleave(args: list[str]) -> None:
            if not seen:
                conn = db.connect(rw["db"])
                try:
                    seen.append(request(conn, rw["clock"], kind="global", run_id="R1", tool_key="k2"))
                finally:
                    conn.close()

        rw["git"].on_call = interleave
        assert global_r1(rw)[0] == 0
        assert seen == [reviews.Deferred("V1")]

    def test_rows_already_captured_are_left_alone(self, rw) -> None:
        token = grant_request(rw, tasks=["T1", "T5"])
        sql(
            rw["db"],
            "UPDATE review_tasks SET base_sha = 'x', head_sha = 'y', head_state = 'branch' WHERE task_id = 'T1'",
        )
        assert tool_review(rw, "pilot:alert:ev:1", token, review="V1")[0] == 0
        assert heads(rw["db"]) == {"T1": ("x", "y", "branch"), "T5": (B5, H5, "branch")}


class TestPrepareFailures:
    def test_no_resolver(self, rw) -> None:
        rw["deps"].resolve_model = None
        code, out = global_r1(rw)
        assert code == 1 and "no model resolver" in out["detail"]
        assert review_row(rw["db"], "V1")["state"] == "failed" and launches(rw) == []

    def test_a_refusing_resolver(self, rw) -> None:
        def refuse(stem: str, lane: str | None) -> tuple[str, str | None]:
            raise ValueError(f"no policy row for {stem}")

        rw["deps"].resolve_model = refuse
        code, out = global_r1(rw)
        assert code == 1 and "model policy: no policy row for qs-global-review" in out["detail"]

    def test_a_missing_agent_file(self, rw) -> None:
        (rw["main"] / ".claude" / "agents" / "qs-global-review.md").unlink()
        code, out = global_r1(rw)
        assert code == 1 and "render the agents on main" in out["detail"]
        assert review_row(rw["db"], "V1")["state"] == "failed"

    def test_an_unwritable_review_dir(self, rw) -> None:
        directory = reviews.review_dir("V1")
        (directory / ".claude").write_text("a file, not a directory")
        code, out = global_r1(rw)
        assert code == 1 and "agent copy" in out["detail"]

    def test_a_review_dir_inside_a_checkout(self, rw, monkeypatch) -> None:
        monkeypatch.setenv("QS_CP_REVIEW_DIR", str(rw["main"] / "reviews"))
        code, out = global_r1(rw)
        assert code == 1 and "review directory" in out["detail"]


class TestIdentifyAndFailures:
    def test_never_listed_fails_the_review_with_one_alert_per_open_run(self, rw) -> None:
        rw["launch"]["list"] = False
        token = grant_request(rw)
        code, out = tool_review(rw, "pilot:alert:ev:1", token, review="V1")
        assert code == 1 and "was never listed" in out["detail"]
        row = review_row(rw["db"], "V1")
        assert row["state"] == "failed" and row["finished_at"]
        assert review_alerts(rw["db"]) == [("R1", "review:V1:failed"), ("R2", "review:V1:failed")]
        payload = __import__("json").loads(sql(rw["db"], "SELECT payload FROM alerts LIMIT 1")[0][0])
        assert payload == {
            "review_id": "V1",
            "kind": "cross_run",
            "state": "failed",
            "error": row["error"],
            "finding_ids": [],
        }
        assert sum(rw["deps"].clock.sleeps) >= 60  # it waited LAUNCH_SETTLE_S

    def test_a_failed_launch(self, rw) -> None:
        rw["launch"]["exit"] = 1
        code, out = global_r1(rw)
        assert code == 1 and "claude --bg exited 1" in out["detail"]
        assert review_row(rw["db"], "V1")["state"] == "failed"
        assert review_alerts(rw["db"]) == [("R1", "review:V1:failed")]

    def test_a_closed_run_gets_no_alert(self, rw) -> None:
        rw["launch"]["exit"] = 1
        token = grant_request(rw)
        sql(rw["db"], "UPDATE runs SET state = 'closed' WHERE id = 'R2'")
        assert tool_review(rw, "pilot:alert:ev:1", token, review="V1")[0] == 1
        assert review_alerts(rw["db"]) == [("R1", "review:V1:failed")]

    def test_identify_after_an_early_finish(self, rw) -> None:
        rw["launch"]["on_launch"] = lambda call: sql(rw["db"], "UPDATE reviews SET state = 'finished'")
        code, out = global_r1(rw)
        assert code == 0, out
        assert out["result"]["state"] == "finished"
        row = review_row(rw["db"], "V1")
        assert row["state"] == "finished" and row["session_id"] and row["short_id"] == "short-1"

    def test_identify_on_a_review_the_pilot_closed(self, rw) -> None:
        rw["launch"]["on_launch"] = lambda call: sql(rw["db"], "UPDATE reviews SET state = 'abandoned'")
        token = grant_request(rw)
        code, out = tool_review(rw, "pilot:alert:ev:1", token, review="V1")
        assert code == 3 and out["error"] == "STALE_TOKEN"  # the recheck before identify: the grant is void
        assert review_row(rw["db"], "V1")["state"] == "abandoned" and review_alerts(rw["db"]) == []

    def test_identify_refuses_a_failed_review_on_the_run_token_path(self, rw) -> None:
        rw["launch"]["on_launch"] = lambda call: sql(rw["db"], "UPDATE reviews SET state = 'failed'")
        code, out = global_r1(rw)
        assert code == 1 and "review V1 is failed" in out["detail"]
        assert review_alerts(rw["db"]) == []  # the guarded close changed nothing: no alert

    def test_on_success_refuses_a_review_closed_while_identifying(self, rw) -> None:
        def close() -> None:
            sql(rw["db"], "UPDATE reviews SET state = 'failed'")

        rw["launch"]["on_launch"] = lambda call: setattr(rw["claude"], "before_list", close)
        code, out = global_r1(rw)
        assert code == 1 and "review V1 is failed" in out["detail"]
        assert sql(rw["db"], "SELECT state FROM tool_calls WHERE tool = 'review'")[0][0] == "failed"


class TestPreconditionsAndTakeover:
    def _crash(self, rw, skip: int, run: Any) -> None:
        from control_plane import faults

        with faults.arm("review.before_effect", skip=skip), pytest.raises(faults.FaultInjected):
            run()
        rw["probe"].reap_thread()  # the crashed process is gone

    def test_a_takeover_finishes_identify_and_never_relaunches(self, rw) -> None:
        token = grant_request(rw)
        self._crash(rw, 3, lambda: tool_review(rw, "pilot:alert:ev:1", token, review="V1"))
        nonce = review_row(rw["db"], "V1")["nonce"]
        assert len(launches(rw)) == 1 and review_row(rw["db"], "V1")["state"] == "launching"
        code, out = tool_review(rw, "pilot:alert:ev:1", token, review="V1")
        assert code == 0, out
        assert out["result"]["state"] == "running" and len(launches(rw)) == 1
        assert review_row(rw["db"], "V1")["nonce"] == nonce

    def test_a_takeover_past_the_settle_deadline_fails(self, rw) -> None:
        rw["launch"]["list"] = False
        self._crash(rw, 3, lambda: global_r1(rw))
        rw["clock"].advance(120)
        code, out = global_r1(rw)
        assert code == 1 and "never listed" in out["detail"] and len(launches(rw)) == 1

    def test_a_takeover_of_a_capture_proceeds(self, rw) -> None:
        from control_plane import faults

        def crash(args: list[str]) -> None:
            rw["git"].on_call = None
            raise faults.FaultInjected("mid-capture")

        rw["git"].on_call = crash
        with pytest.raises(faults.FaultInjected):
            global_r1(rw)
        rw["probe"].reap_thread()
        assert review_row(rw["db"], "V1")["heads_at"] and heads(rw["db"])["T1"][2] is None
        code, out = global_r1(rw)
        assert code == 0, out
        assert heads(rw["db"])["T1"] == (B1, H1, "branch")

    def test_prepare_refuses_a_closed_review(self, rw) -> None:
        self._crash(rw, 1, lambda: global_r1(rw))
        sql(rw["db"], "UPDATE reviews SET state = 'failed'")
        code, out = global_r1(rw)
        assert code == 1 and "review V1 is failed" in out["detail"] and launches(rw) == []

    def test_prepare_refuses_a_review_closed_during_the_capture(self, rw) -> None:
        rw["git"].on_call = lambda args: sql(rw["db"], "UPDATE reviews SET state = 'failed'")
        code, out = global_r1(rw)
        assert code == 1 and "review V1 is failed" in out["detail"] and launches(rw) == []
        assert review_alerts(rw["db"]) == []

    def test_launch_refuses_a_closed_review(self, rw) -> None:
        self._crash(rw, 2, lambda: global_r1(rw))
        sql(rw["db"], "UPDATE reviews SET state = 'failed'")
        code, out = global_r1(rw)
        assert code == 1 and "review V1 is failed" in out["detail"] and launches(rw) == []

    def test_the_grant_path_on_a_row_already_launching_mints_nothing(self, rw) -> None:
        token = grant_request(rw)
        sql(rw["db"], "UPDATE reviews SET state = 'launching', nonce = ?", ["5" * 32])
        assert tool_review(rw, "pilot:alert:ev:1", token, review="V1")[0] == 0
        assert review_row(rw["db"], "V1")["nonce"] == "5" * 32

    def test_the_grant_path_on_a_running_row(self, rw) -> None:
        token = grant_request(rw)
        sql(rw["db"], "UPDATE reviews SET state = 'running'")
        code, out = tool_review(rw, "pilot:alert:ev:1", token, review="V1")
        assert code == 1 and out["result"]["error"] == "INVALID_STATE" and launches(rw) == []

    def test_the_grant_path_mints_the_session_nonce(self, rw) -> None:
        token = grant_request(rw)
        assert review_row(rw["db"], "V1")["nonce"] is None
        code, out = tool_review(rw, "pilot:alert:ev:1", token, review="V1")
        assert code == 0 and out["result"]["state"] == "running"
        row = review_row(rw["db"], "V1")
        assert len(row["nonce"]) == 32 and f"review:V1.1.{row['nonce']}" in launches(rw)[0].argv[-1]
        call = dict(sql(rw["db"], "SELECT * FROM tool_calls WHERE tool = 'review'")[0])
        assert (call["run_id"], call["task_id"], call["actor"], call["state"]) == (None, None, "review:V1", "succeeded")


class TestCoalescing:
    def test_a_skip_is_a_success_and_replays(self, rw) -> None:
        code, out = tool_review(rw, "task:R2:global:1", rw["tok2"], kind="global", run="R2")
        assert code == 0 and out["result"] == {"skipped": "single_task", "tasks": ["T5"]}
        code, again = tool_review(rw, "task:R2:global:1", rw["tok2"], kind="global", run="R2")
        assert code == 0 and again["replayed"] and again["result"] == out["result"]
        assert sql(rw["db"], "SELECT count(*) FROM reviews")[0][0] == 0
        assert sql(rw["db"], "SELECT count(*) FROM review_triggers")[0][0] == 0
        assert launches(rw) == []

    def test_a_deferral_is_busy_and_the_same_key_launches_after_the_close(self, rw) -> None:
        assert global_r1(rw)[0] == 0
        code, out = global_r1(rw, "task:R1:global:2")
        assert code == 6 and out["error"] == "BUSY" and out["review_id"] == "V1"
        assert "replay this key after it closes" in out["detail"]
        assert sql(rw["db"], "SELECT count(*) FROM tool_calls WHERE key = 'task:R1:global:2'")[0][0] == 0
        sql(rw["db"], "UPDATE reviews SET state = 'finished'")
        code, out = global_r1(rw, "task:R1:global:2")
        assert code == 0 and out["result"]["review_id"] == "V2" and len(launches(rw)) == 2

    def test_a_join_and_a_joined_review_that_fails(self, rw) -> None:
        grant_request(rw, kind="global")  # V1 requested: its heads are not captured yet
        code, out = global_r1(rw, "task:R1:global:1", trigger={"why": "merged T9"})
        assert code == 0 and out["result"] == {"coalesced_into": "V1"} and launches(rw) == []
        assert triggers(rw["db"])[-1] == ("V1", "orchestrator", "task:R1:global:1", "joined")
        stored = sql(rw["db"], 'SELECT "trigger" FROM review_triggers ORDER BY id DESC LIMIT 1')[0][0]
        assert __import__("json").loads(stored) == {"why": "merged T9"}
        sql(rw["db"], "UPDATE reviews SET state = 'failed'")
        code, again = global_r1(rw, "task:R1:global:1", trigger={"why": "merged T9"})
        assert code == 0 and again["replayed"] and again["result"] == {"coalesced_into": "V1"}
        code, fresh = global_r1(rw, "task:R1:global:2")
        assert code == 0 and fresh["result"]["review_id"] == "V2"


class TestMatrix:
    @pytest.mark.parametrize(
        ("args", "code", "error"),
        [
            ({"kind": "global", "run": "R1"}, 8, "CONFLICT"),  # R2's token on R1
            ({"kind": "cross_run", "run": "R2"}, 8, "CONFLICT"),
            ({"review": "V1"}, 8, "CONFLICT"),
            ({"kind": "global", "run": "R2", "review": "V1"}, 8, "CONFLICT"),
            ({"kind": "final", "run": "R2"}, 2, "USAGE"),
            ({"run": "R2"}, 2, "USAGE"),
            ({"kind": "global"}, 2, "USAGE"),
            ({"kind": "global", "run": "R2", "tasks": ["T5"]}, 2, "USAGE"),
            ({"kind": "global", "run": "R2", "trigger": "merged"}, 2, "USAGE"),
            ({"kind": "global", "run": "R9"}, 8, "NOT_FOUND"),
        ],
    )
    def test_a_run_token(self, rw, args, code, error) -> None:
        got, out = tool_review(rw, "k1", rw["tok2"], **args)
        assert (got, out["error"]) == (code, error), out
        assert sql(rw["db"], "SELECT count(*) FROM tool_calls")[0][0] == 0

    def test_a_run_token_may_name_its_run(self, rw) -> None:
        code, out = tool_review(rw, "k1", rw["tok2"], kind="global", run="r2")
        assert code == 0 and out["result"]["skipped"] == "single_task"

    def test_a_grant(self, rw) -> None:
        token = grant_request(rw, key="pilot:merge:T6:12:1")
        grant_request(rw, tasks=["T2", "T5"], key="pilot:alert:other:1")
        for key, args in (
            ("pilot:merge:T6:12:2", {"review": "V1"}),  # another key
            ("pilot:alert:other:1", {"review": "V1"}),  # V2's key
            ("pilot:merge:T6:12:1", {"review": "V2"}),
            ("pilot:merge:T6:12:1", {"kind": "global", "run": "R1"}),
            ("pilot:merge:T6:12:1", {"review": "V1", "kind": "cross_run"}),
        ):
            code, out = tool_review(rw, key, token, **args)
            assert code == 8 and out["error"] == "CONFLICT", (key, args, out)
        assert sql(rw["db"], "SELECT count(*) FROM tool_calls")[0][0] == 0
        code, out = tool_review(rw, "pilot:merge:T6:12:1", token, review="V1")  # its own key, colons included
        assert code == 0 and out["result"]["review_id"] == "V1"

    def test_a_session_token_and_a_node_token(self, rw) -> None:
        from .conftest import insert_node

        assert global_r1(rw)[0] == 0
        row = review_row(rw["db"], "V1")
        node = insert_node(rw["db"], "N1", "R1", "T1")
        for token in (f"review:V1.1.{row['nonce']}", node):
            code, out = tool_review(rw, "k9", token, kind="global", run="R1")
            assert code == 8 and out["error"] == "CONFLICT"
        assert sql(rw["db"], "SELECT count(*) FROM tool_calls WHERE key = 'k9'")[0][0] == 0


def test_register_review_tool_is_idempotent_and_selfchecked() -> None:
    from control_plane import selfcheck, tools

    reviews.register_review_tool()
    reviews.register_review_tool()
    assert tools.REGISTRY["review"] is reviews.SPEC
    assert "review" in selfcheck._expected_tools()
    tools.reset()
    assert "review" not in tools.REGISTRY
    reviews.register_review_tool()
    assert "review" in tools.REGISTRY


# --------------------------------------------------------------------------- T8: review finish, review_done, outcome


def session(rw: dict[str, Any], rid: str = "V1") -> str:
    return f"review:{rid}.1.{review_row(rw['db'], rid)['nonce']}"


def finish_cli(rw: dict[str, Any], token: str, data: Any, name: str = "finish.json") -> tuple[int, Any]:
    import json as _json

    from .conftest import run_cli

    f = rw["tmp"] / name
    f.write_text(data if isinstance(data, str) else _json.dumps(data))
    return run_cli("review", "finish", "--input", str(f), "--token", token)


GOOD = {
    "summary": "fits together",
    "tasks": {
        "T1": {"outcome": "pass", "blast_radius": "ok"},
        "T2": {"outcome": "fail", "reason": "breaks T1", "blast_radius": "doubt"},
        "T3": {"outcome": "pass"},
        "T4": {"outcome": "pass", "reason": "plan only"},
    },
}


def messages_of(path: Path) -> list[dict[str, Any]]:
    import json as _json

    rows = sql(
        path, "SELECT run_id, kind, sender, dedupe_key, payload FROM messages WHERE kind = 'review_done' ORDER BY id"
    )
    return [{**dict(r), "payload": _json.loads(r["payload"])} for r in rows]


class TestFinish:
    def test_a_global_review_hands_back(self, rw) -> None:
        from .conftest import run_cli

        assert global_r1(rw)[0] == 0
        token = session(rw)
        item = rw["tmp"] / "item.json"
        item.write_text('{"severity": "must_fix", "category": "design", "title": "T2 breaks T1", "body": "b"}')
        base = ("finding", "open", "--phase", "build", "--source", "global_review", "--input", str(item))
        assert run_cli(*base, "--task", "T2", "--token", token) == (0, {"ok": True, "ids": [1]})
        code, out = finish_cli(rw, token, GOOD)
        assert code == 0, out
        [msg] = messages_of(rw["db"])
        assert out == {
            "ok": True,
            "review_id": "V1",
            "state": "finished",
            "messages": [{"run_id": "R1", "message_id": msg_id(rw)}],
        }
        row = review_row(rw["db"], "V1")
        assert (row["state"], row["summary"]) == ("finished", "fits together") and row["finished_at"]
        rated = sql(rw["db"], "SELECT task_id, value, head_sha, review, reason, actor FROM blast_radius ORDER BY id")
        assert [tuple(r) for r in rated] == [
            ("T1", "ok", H1, "V1", None, "review:V1"),
            ("T2", "doubt", H2, "V1", "breaks T1", "review:V1"),
        ]
        recorded = sql(
            rw["db"],
            "SELECT task_id, outcome, reason, blast_radius_id FROM review_tasks WHERE review_id = 'V1' ORDER BY 1",
        )
        assert [tuple(r) for r in recorded] == [
            ("T1", "pass", None, 1),
            ("T2", "fail", "breaks T1", 2),
            ("T3", "pass", None, None),
            ("T4", "pass", "plan only", None),
        ]
        assert (msg["run_id"], msg["sender"], msg["dedupe_key"]) == ("R1", "review:V1", "review:V1")
        payload = msg["payload"]
        assert (payload["review_id"], payload["kind"], payload["summary"], payload["other_runs"]) == (
            "V1",
            "global",
            "fits together",
            [],
        )
        assert payload["tasks"][1] == {
            "task_id": "T2",
            "base_sha": B2,
            "head_sha": H2,
            "head_state": "branch",
            "outcome": "fail",
            "reason": "breaks T1",
            "blast_radius": "doubt",
            "finding_ids": [1],
        }
        code, again = finish_cli(rw, token, GOOD)  # no replay: the token is void
        assert (
            code == 3 and again["error"] == "STALE_TOKEN" and again["instructions"].startswith("This review is closed")
        )
        assert run_cli(*base, "--task", "T1", "--token", token)[1]["error"] == "STALE_TOKEN"

    def test_a_cross_run_review_posts_to_each_open_run(self, rw) -> None:
        token = grant_request(rw)
        assert tool_review(rw, "pilot:alert:ev:1", token, review="V1")[0] == 0
        r3, _ = open_run("r3", "S-3")
        done = {"summary": "s", "tasks": {"T1": {"outcome": "pass"}, "T5": {"outcome": "fail", "reason": "dup of T1"}}}
        bad = {"summary": "s", "tasks": {"T1": {"outcome": "pass", "blast_radius": "ok"}, "T5": {"outcome": "pass"}}}
        code, out = finish_cli(rw, session(rw), bad)
        assert code == 2 and "only a global review" in out["detail"]
        code, out = finish_cli(rw, session(rw), done)
        assert code == 0, out
        msgs = messages_of(rw["db"])
        assert [
            (m["run_id"], [t["task_id"] for t in m["payload"]["tasks"]], m["payload"]["other_runs"]) for m in msgs
        ] == [
            ("R1", ["T1"], ["R2"]),
            ("R2", ["T5"], ["R1"]),
        ]
        assert r3 not in {m["run_id"] for m in msgs}

    def test_a_closed_run_gets_no_review_done(self, rw) -> None:
        token = grant_request(rw)
        assert tool_review(rw, "pilot:alert:ev:1", token, review="V1")[0] == 0
        sql(rw["db"], "UPDATE runs SET state = 'closed' WHERE id = 'R2'")
        done = {"summary": "s", "tasks": {"T1": {"outcome": "pass"}, "T5": {"outcome": "pass"}}}
        assert finish_cli(rw, session(rw), done)[0] == 0
        assert [m["run_id"] for m in messages_of(rw["db"])] == ["R1"]

    def test_a_finish_while_still_launching(self, rw) -> None:
        assert global_r1(rw)[0] == 0
        sql(rw["db"], "UPDATE reviews SET state = 'launching'")
        assert finish_cli(rw, session(rw), GOOD)[0] == 0
        assert review_row(rw["db"], "V1")["state"] == "finished"

    def test_the_order_of_checks(self, rw) -> None:
        from .conftest import insert_node

        assert global_r1(rw)[0] == 0
        node = insert_node(rw["db"], "N1", "R1", "T1")
        assert finish_cli(rw, node, "not json")[1]["error"] == "USAGE"  # the file first
        assert finish_cli(rw, node, '{"tasks": {}, "tasks": {}}')[1]["error"] == "USAGE"  # duplicate keys
        assert finish_cli(rw, node, "{}")[1]["error"] == "CONFLICT"  # then the token
        assert finish_cli(rw, rw["tok1"], "{}")[1]["error"] == "CONFLICT"
        grant = f"review:V1.0.{'0' * 32}"
        assert finish_cli(rw, grant, "{}")[1]["error"] == "CONFLICT"
        assert finish_cli(rw, session(rw), "{}")[1]["error"] == "USAGE"  # then the shape
        from .conftest import run_cli

        assert run_cli("review", "finish", "--input", str(rw["tmp"] / "missing.json"), "--token", node)[1]["error"] == (
            "USAGE"
        )

    @pytest.mark.parametrize(
        ("change", "error", "detail"),
        [
            (lambda d: d["tasks"].pop("T3"), "USAGE", "missing: T3"),
            (lambda d: d["tasks"].update(T5={"outcome": "pass"}), "CONFLICT", "not in the scope"),
            (lambda d: d["tasks"]["T3"].update(outcome="maybe"), "USAGE", "pass or fail"),
            (lambda d: d["tasks"]["T2"].pop("reason"), "USAGE", "a fail needs a `reason`"),
            (lambda d: d["tasks"]["T2"].update(reason="  "), "USAGE", "a fail needs a `reason`"),
            (lambda d: d["tasks"]["T1"].pop("blast_radius"), "USAGE", "needs its `blast_radius`"),
            (lambda d: d["tasks"]["T4"].update(blast_radius="ok"), "USAGE", "no head was reviewed"),
            (lambda d: d["tasks"]["T1"].update(blast_radius="huge"), "USAGE", "ok, doubt or too_large"),
            (lambda d: d["tasks"]["T1"].update(severity="x"), "USAGE", "must be an object of"),
            (lambda d: d["tasks"].update(T3="pass"), "USAGE", "must be an object of"),
            (lambda d: d["tasks"]["T3"].update(reason=7), "USAGE", "must be a string"),
            (lambda d: d["tasks"]["T3"].update(reason="\ud800"), "USAGE", "not valid Unicode"),
            (lambda d: d.update(summary=["s"]), "USAGE", "must be a string"),
            (lambda d: d.update(extra=1), "USAGE", "--input must be"),
            (lambda d: d.update(tasks=[]), "USAGE", "--input must be"),
        ],
    )
    def test_validation(self, rw, change, error: str, detail: str) -> None:
        import copy

        assert global_r1(rw)[0] == 0
        data = copy.deepcopy(GOOD)
        change(data)
        code, out = finish_cli(rw, session(rw), data)
        assert out["error"] == error and detail in out["detail"], out
        assert review_row(rw["db"], "V1")["state"] == "running"
        assert sql(rw["db"], "SELECT count(*) FROM blast_radius")[0][0] == 0

    def test_an_item_blast_radius_is_optional(self, rw) -> None:
        import copy

        assert global_r1(rw)[0] == 0
        data = copy.deepcopy(GOOD)
        data["tasks"]["T3"]["blast_radius"] = "too_large"
        assert finish_cli(rw, session(rw), data)[0] == 0
        assert sql(rw["db"], "SELECT value FROM blast_radius WHERE task_id = 'T3'")[0][0] == "too_large"

    def test_atomic(self, rw, monkeypatch) -> None:
        from control_plane import messages

        assert global_r1(rw)[0] == 0

        def broken(*a: Any, **kw: Any) -> Any:
            raise errors.CpError("INTERNAL", "post failed")

        monkeypatch.setattr(messages, "post_locked", broken)
        assert finish_cli(rw, session(rw), GOOD)[1]["error"] == "INTERNAL"
        assert review_row(rw["db"], "V1")["state"] == "running"
        assert sql(rw["db"], "SELECT count(*) FROM blast_radius")[0][0] == 0
        assert {r[0] for r in sql(rw["db"], "SELECT outcome FROM review_tasks")} == {None}


def msg_id(rw: dict[str, Any]) -> int:
    return int(sql(rw["db"], "SELECT id FROM messages WHERE kind = 'review_done'")[0][0])


class TestOutcome:
    def _finished(self, rw, rid: str, finished_at: str, outcomes: dict[str, str], kind: str = "global") -> None:
        sql(
            rw["db"],
            "INSERT INTO reviews (id, kind, coalesce_key, state, agent, name, requested_by, created_at, updated_at,"
            " finished_at) VALUES (?, ?, ?, 'finished', 'a', ?, 'x', 'x', 'x', ?)",
            [rid, kind, f"k-{rid}", f"n-{rid}", finished_at],
        )
        for task, out in outcomes.items():
            sql(
                rw["db"],
                "INSERT INTO review_tasks (review_id, task_id, run_id, base_sha, head_sha, head_state, outcome, reason)"
                " VALUES (?, ?, 'R1', 'b', ?, 'branch', ?, ?)",
                [rid, task, H1, out, f"{rid}:{out}"],
            )

    def _conn_outcome(self, rw, *a: Any, **kw: Any) -> Any:
        conn = db.connect(rw["db"])
        try:
            return reviews.outcome(conn, *a, **kw)
        finally:
            conn.close()

    def test_the_latest_finished_for_that_task_kind_and_head(self, rw) -> None:
        self._finished(rw, "V8", "2026-10-03T10:00:00.000000Z", {"T1": "fail"})
        self._finished(rw, "V9", "2026-10-03T11:00:00.000000Z", {"T1": "pass"})
        self._finished(rw, "V10", "2026-10-03T11:00:00.000000Z", {"T1": "fail"})  # same time: the newer id
        self._finished(rw, "V11", "2026-10-03T12:00:00.000000Z", {"T1": "pass"}, kind="cross_run")
        assert self._conn_outcome(rw, "T1", kind="global", head_sha=H1) == {
            "review_id": "V10",
            "outcome": "fail",
            "reason": "V10:fail",
            "base_sha": "b",
            "head_sha": H1,
            "finished_at": "2026-10-03T11:00:00.000000Z",
        }
        assert self._conn_outcome(rw, "T1", kind="cross_run", head_sha=f" {H1} ")["review_id"] == "V11"
        assert self._conn_outcome(rw, "T1", kind="global", head_sha="other") is None
        assert self._conn_outcome(rw, "T2", kind="global", head_sha=H1) is None

    def test_no_head_is_advisory(self, rw) -> None:
        self._finished(rw, "V8", "2026-10-03T10:00:00.000000Z", {"T1": "pass"})
        for head in (None, "", "  "):
            assert self._conn_outcome(rw, "T1", kind="global", head_sha=head) is None

    def test_only_finished_reviews_with_an_outcome(self, rw) -> None:
        self._finished(rw, "V8", "2026-10-03T10:00:00.000000Z", {"T1": "pass"})
        sql(rw["db"], "UPDATE reviews SET state = 'abandoned'")
        assert self._conn_outcome(rw, "T1", kind="global", head_sha=H1) is None
        sql(rw["db"], "UPDATE reviews SET state = 'finished'")
        sql(rw["db"], "UPDATE review_tasks SET outcome = NULL")
        assert self._conn_outcome(rw, "T1", kind="global", head_sha=H1) is None
