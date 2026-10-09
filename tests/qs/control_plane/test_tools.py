"""Checkpoint 10: the tools layer — run_recorded's outcome table and the built-in tools (AC5 replay, AC8, AC14, AC17)."""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from control_plane import db, faults, locks, merge_policy, tools

from .conftest import (
    NEXT,
    ORCH,
    FakeClaude,
    FakeProbe,
    FakeRunner,
    agent,
    insert_node,
    insert_task,
    open_run,
    run_cli,
    sql,
)
from .toolsim import Sim


@dataclass
class W:
    db: Path
    run: str
    token: str
    sim: Sim
    tmp: Path
    main: Path
    claude: FakeClaude
    runner: FakeRunner
    probe: FakeProbe
    clock: Any
    files: dict[str, str]


@pytest.fixture
def w(migrated, tmp_path, fake_runner, fake_claude, fake_main, fake_probe, fake_clock) -> W:
    run_id, token = open_run()
    wt = tmp_path / "wt1"
    wt.mkdir()
    (wt / ".git").write_text("gitdir: /main/.git/worktrees/wt1\n")  # a linked worktree
    insert_task(
        migrated, "T1", run_id, issue_number=11, worktree=str(wt), branch="QS_11", is_deliverable=1, pr_number=5
    )
    sim = Sim(fake_runner, fake_claude, fake_main, wt).install()
    sim.registered.add(str(wt.resolve()))
    files = {}
    for name, text in (("prompt", "Build T1."), ("msg", "Continue."), ("body", "Issue body"), ("summary", "- did it")):
        f = tmp_path / f"{name}.md"
        f.write_text(text)
        files[name] = str(f)
    return W(migrated, run_id, token, sim, tmp_path, fake_main, fake_claude, fake_runner, fake_probe, fake_clock, files)


def tool(w: W, name: str, key: str, token: str | None = None, task: str = "T1", **args: Any) -> tuple[int, Any]:
    f = w.tmp / f"args-{name}-{key}.json".replace(":", "_").replace("/", "_").replace("\\", "_")
    f.write_text(json.dumps(args))
    return run_cli("tool", name, "--task", task, "--key", key, "--args-file", str(f), "--token", token or w.token)


def allow_all(task: Any) -> merge_policy.PolicyResult:
    return merge_policy.PolicyResult(True, "test policy: allow")


def call_row(w: W, name: str, key: str) -> dict | None:
    rows = sql(w.db, "SELECT * FROM tool_calls WHERE tool = ? AND key = ?", [name, key])
    return dict(rows[0]) if rows else None


def node_row(w: W, node_id: str) -> dict:
    return dict(sql(w.db, "SELECT * FROM nodes WHERE id = ?", [node_id])[0])


# --- the per-tool table: args, how to count its effect, what to prepare, the effecting step's index ---


def _prep_resume(w: W) -> None:
    insert_node(w.db, "N1", w.run, "T1", state="reaped", session_id="S-old", name="r1-T1-g1")
    w.sim.resumed_name = "r1-T1-g1"


def _prep_merge(w: W) -> None:
    sql(w.db, "UPDATE tasks SET state = 'ready_to_merge'")
    merge_policy.install(allow_all)


@dataclass(frozen=True)
class T:
    args: Callable[[W], dict[str, Any]]
    effect: Callable[[W], int]
    prep: Callable[[W], None] = lambda w: None
    effect_step: int = 0


TABLE: dict[str, T] = {
    "worktree-create": T(
        lambda w: {"phase": "/create-plan"}, lambda w: w.sim.created, lambda w: shutil.rmtree(w.sim.wt), 1
    ),
    "worktree-cleanup": T(lambda w: {}, lambda w: w.sim.effects("cleanup_worktree.py")),
    "spawn": T(
        lambda w: {"agent": "qs-node", "permission_mode": "auto", "prompt_file": w.files["prompt"]},
        lambda w: w.sim.effects("claude", "--bg"),
        effect_step=1,
    ),
    "resume": T(lambda w: {"message_file": w.files["msg"]}, lambda w: w.sim.effects("claude", "--bg"), _prep_resume, 1),
    "issue-create": T(
        lambda w: {"title": "New", "body_file": w.files["body"], "labels": ["kind:feature", "target:factory"]},
        lambda w: w.sim.effects("create_issue.py"),
    ),
    "pr-create": T(
        lambda w: {"title": "PR", "summary_file": w.files["summary"]}, lambda w: w.sim.effects("create_pr.py")
    ),
    "push": T(lambda w: {}, lambda w: w.sim.effects("git", "push")),
    "merge": T(lambda w: {}, lambda w: w.sim.effects("gh", "pr", "merge"), _prep_merge, 2),
}
SIDE_EFFECT_TOOLS = sorted(TABLE)


@pytest.mark.parametrize("name", SIDE_EFFECT_TOOLS)
class TestIdempotency:
    def test_replay_of_a_succeeded_key(self, w: W, name: str) -> None:
        t = TABLE[name]
        t.prep(w)
        code, first = tool(w, name, "k1", **t.args(w))
        assert code == 0, first
        assert first["replayed"] is False
        code, again = tool(w, name, "k1", **t.args(w))
        assert code == 0 and again["replayed"] is True and again["result"] == first["result"]
        assert t.effect(w) == 1

    def test_same_key_other_args_conflicts(self, w: W, name: str) -> None:
        t = TABLE[name]
        t.prep(w)
        assert tool(w, name, "k1", **t.args(w))[0] == 0
        code, out = tool(w, name, "k1", **{**t.args(w), "extra": 1})
        assert code == 8 and out["error"] == "CONFLICT"

    def test_fault_before_effect_then_replay(self, w: W, name: str) -> None:
        t = TABLE[name]
        t.prep(w)
        with faults.arm(f"{name}.before_effect", skip=t.effect_step), pytest.raises(faults.FaultInjected):
            tool(w, name, "k1", **t.args(w))
        assert call_row(w, name, "k1")["state"] == "started"
        assert tool(w, name, "k1", **t.args(w))[0] == 0
        assert t.effect(w) == 1

    def test_fault_after_effect_then_replay(self, w: W, name: str) -> None:
        t = TABLE[name]
        t.prep(w)
        with faults.arm(f"{name}.after_effect", skip=t.effect_step), pytest.raises(faults.FaultInjected):
            tool(w, name, "k1", **t.args(w))
        assert t.effect(w) == 1
        code, out = tool(w, name, "k1", **t.args(w))
        assert code == 0, out
        assert t.effect(w) == 1  # the probe saw the effect in place

    def test_two_concurrent_same_key_calls(self, w: W, name: str) -> None:
        t = TABLE[name]
        t.prep(w)
        w.sim.hold = threading.Event()
        results: list[tuple[int, Any]] = []
        first = threading.Thread(target=lambda: results.append(tool(w, name, "k1", **t.args(w))))
        first.start()
        assert w.sim.entered.wait(timeout=5)
        code, out = tool(w, name, "k1", **t.args(w))
        assert code == 6 and out["error"] == "BUSY"
        w.sim.hold.set()
        first.join(timeout=5)
        assert results[0][0] == 0
        assert t.effect(w) == 1

    def test_dead_pid_with_a_live_group_keeps_the_key_busy(self, w: W, name: str) -> None:
        t = TABLE[name]
        t.prep(w)
        with faults.arm(f"{name}.before_effect"), pytest.raises(faults.FaultInjected):
            tool(w, name, "k1", **t.args(w))
        sql(w.db, "UPDATE tool_calls SET holder_pid = 7, holder_pid_start = 'start-7', holder_pgid = 7")
        w.probe.kill(7, group=False)
        assert tool(w, name, "k1", **t.args(w))[1]["error"] == "BUSY"
        w.probe.dead_groups.add(7)
        assert tool(w, name, "k1", **t.args(w))[0] == 0

    def test_terminal_task(self, w: W, name: str) -> None:
        t = TABLE[name]
        t.prep(w)
        for state in ("dropped", "merged"):
            sql(w.db, "UPDATE tasks SET state = ?", [state])
            code, out = tool(w, name, f"k-{state}", **t.args(w))
            if name == "worktree-cleanup":
                assert code == 0
                continue
            assert code == 1 and out["error"] == "TOOL_FAILED" and out["result"]["error"] == "INVALID_STATE"
            assert call_row(w, name, f"k-{state}")["state"] == "failed"
        assert t.effect(w) == (1 if name == "worktree-cleanup" else 0)


def test_same_key_under_two_tools_is_two_calls(w: W) -> None:
    assert tool(w, "push", "k")[0] == 0
    assert tool(w, "issue-create", "k", **TABLE["issue-create"].args(w))[0] == 0
    assert (w.sim.effects("git", "push"), w.sim.effects("create_issue.py")) == (1, 1)


def test_same_key_on_another_task_conflicts(w: W) -> None:
    insert_task(w.db, "T2", w.run, worktree=str(w.sim.wt), branch="QS_11")
    assert tool(w, "push", "k")[0] == 0
    assert tool(w, "push", "k", task="T2")[1]["error"] == "CONFLICT"


def test_every_call_is_recorded_without_any_token(w: W) -> None:
    order = ["worktree-create", "issue-create", "pr-create", "push", "spawn", "resume", "merge", "worktree-cleanup"]
    assert sorted(order) == SIDE_EFFECT_TOOLS
    for name in order:
        t = TABLE[name]
        if name == "worktree-create":
            t.prep(w)
        elif name == "resume":  # the spawned node's session ended
            sql(w.db, "UPDATE nodes SET state = 'reaped'")
            w.sim.resumed_name = node_row(w, "N1")["name"]
            w.claude.listing = []
        elif name == "merge":
            t.prep(w)
        code, out = tool(w, name, f"audit-{name}", **t.args(w))
        assert code == 0, (name, out)
    rows = [dict(r) for r in sql(w.db, "SELECT * FROM tool_calls")]
    assert len(rows) == len(SIDE_EFFECT_TOOLS)
    secrets_ = [r[0] for r in sql(w.db, "SELECT nonce FROM run_leases")] + [
        r[0] for r in sql(w.db, "SELECT nonce FROM nodes")
    ]
    for row in rows:
        assert row["actor"] == "orchestrator" and row["started_at"] and row["finished_at"]
        assert row["exit_code"] == 0 and row["result"] is not None
        blob = row["args"] + row["outputs"] + row["result"]
        assert not any(s in blob for s in secrets_), row["tool"]


# --------------------------------------------------------------------------- the outcome table (§9.2)


class TestOutcomes:
    def test_step_failure_spends_the_key(self, w: W) -> None:
        from control_plane.runner import RunResult

        w.runner.on(["create_issue.py"], RunResult(1, json.dumps({"error": "gh down"}), "boom"))
        args = TABLE["issue-create"].args(w)
        code, out = tool(w, "issue-create", "k1", **args)
        assert code == 1 and out["error"] == "TOOL_FAILED" and out["result"]["output"]["exit_code"] == 1
        row = call_row(w, "issue-create", "k1")
        assert row["state"] == "failed" and row["exit_code"] == 1
        code, out = tool(w, "issue-create", "k1", **args)
        assert code == 1 and out["replayed"] is True
        assert w.sim.effects("create_issue.py") == 1
        w.runner.rules.pop()
        assert tool(w, "issue-create", "k2", **args)[0] == 0

    def test_busy_before_any_effect_releases_the_key(self, w: W, monkeypatch) -> None:
        monkeypatch.setenv("QS_CP_MAX_GATES", "1")
        monkeypatch.setattr(locks, "GATE_WAIT_S", 3.0)
        sql(
            w.db,
            "INSERT INTO cap_slots (cap, slot, holder_pid, holder_pgid, holder_actor, acquired_at) VALUES ('gates', 0, 9, 9, 'x', 'x')",
        )
        code, out = tool(w, "gate", "g1", mode="impacted")
        assert code == 6 and out["error"] == "BUSY"
        assert call_row(w, "gate", "g1") is None
        w.probe.kill(9)
        assert tool(w, "gate", "g1", mode="impacted")[0] == 0

    def test_policy_refusal_releases_and_the_same_key_works_later(self, w: W) -> None:
        sql(w.db, "UPDATE tasks SET state = 'ready_to_merge'")
        code, out = tool(w, "merge", "m1")
        assert code == 9 and out["error"] == "POLICY_REFUSED" and "child 7" in out["detail"]
        assert call_row(w, "merge", "m1") is None and w.sim.effects("gh", "pr", "merge") == 0
        merge_policy.install(allow_all)
        code, out = tool(w, "merge", "m1")
        assert code == 0 and out["result"] == {"merge_sha": "m" * 40}
        task = dict(sql(w.db, "SELECT state, merge_sha FROM tasks")[0])
        assert task == {"state": "merged", "merge_sha": "m" * 40}

    def test_stale_token_after_the_claim_releases(self, w: W) -> None:
        def going_stale(call: Any) -> Any:
            run_cli("run", "claim", w.run, "--session-id", ORCH)
            from control_plane.runner import RunResult

            return RunResult(0, "[]", "")

        w.runner.on(["gh", "issue", "list"], going_stale)
        code, out = tool(w, "issue-create", "k1", **TABLE["issue-create"].args(w))
        assert code == 3 and call_row(w, "issue-create", "k1") is None
        assert w.sim.effects("create_issue.py") == 0

    def test_stale_token_after_an_effect_leaves_the_call_started(self, w: W) -> None:
        sql(w.db, "UPDATE tasks SET state = 'ready_to_merge'")
        merge_policy.install(allow_all)

        def stale_policy(task: Any) -> merge_policy.PolicyResult:
            run_cli("run", "claim", w.run, "--session-id", ORCH)
            return merge_policy.PolicyResult(True, "ok")

        merge_policy.install(stale_policy)
        assert tool(w, "merge", "m1")[0] == 3  # recheck before `head` aborts at the next step boundary
        row = call_row(w, "merge", "m1")
        assert row["state"] == "started" and "policy" in json.loads(row["outputs"])
        assert w.sim.effects("gh", "pr", "merge") == 0

    def test_schema_change_mid_call_is_not_recorded(self, w: W) -> None:
        def migrated(call: Any) -> Any:
            sql(w.db, f"PRAGMA user_version = {NEXT}")
            from control_plane.runner import RunResult

            return RunResult(0, "[]", "")

        w.runner.on(["gh", "issue", "list"], migrated)
        code, out = tool(w, "issue-create", "k1", **TABLE["issue-create"].args(w))
        assert code == 5 and call_row(w, "issue-create", "k1")["state"] == "started"

    def test_claim_reprobes_a_row_changed_after_the_probe(self, w: W, monkeypatch) -> None:
        with faults.arm("push.before_effect"), pytest.raises(faults.FaultInjected):
            tool(w, "push", "k1")
        calls = {"n": 0}
        real = w.probe.holder_alive

        def racing(pid: int | None, start: str | None, pgid: int | None) -> bool:
            calls["n"] += 1
            if calls["n"] == 1:
                sql(w.db, "UPDATE tool_calls SET outputs = '{\"x\": 1}'")
            return real(pid, start, pgid)

        monkeypatch.setattr(w.probe, "holder_alive", racing)
        assert tool(w, "push", "k1")[0] == 0 and calls["n"] >= 2


# --------------------------------------------------------------------------- spawn / resume (§9.4)


SPAWN = {"agent": "qs-node", "permission_mode": "auto"}


def spawn(w: W, key: str, task: str = "T1", **extra: Any) -> tuple[int, Any]:
    return tool(w, "spawn", key, task=task, prompt_file=w.files["prompt"], **SPAWN, **extra)


def reap(w: W) -> None:
    """Another cap transaction: reaps dead, unlisted, settled `spawning` rows."""
    c = db.connect(w.db)
    alive = locks.spawn_holders_alive(c, w.probe)
    with db.write(c):
        locks.admit_node(c, w.clock, listing=w.claude.try_agents(), holders_alive=alive, limit=99)
    c.close()


class TestSpawn:
    def test_spawn_records_the_node_and_returns_its_token(self, w: W) -> None:
        code, out = spawn(w, "s1", model="claude-opus-5-5")
        assert code == 0, out
        res = out["result"]
        assert (res["node_id"], res["name"], res["generation"]) == ("N1", "r1-T1-g1", 1)
        assert res["session_id"] == "S-r1-T1-g1" and res["short_id"] == "short-1"
        node = node_row(w, "N1")
        assert node["state"] == "running" and node["permission_mode"] == "auto"
        assert out["token"] == f"node:N1.1.{node['nonce']}"
        [launch] = w.runner.matching("claude", "--bg")
        argv = launch.argv
        assert argv[:6] == ["claude", "--bg", "--agent", "qs-node", "-n", "r1-T1-g1"]
        assert argv[argv.index("--model") + 1] == "claude-opus-5-5"
        assert argv[argv.index("--permission-mode") + 1] == "auto"
        from control_plane import hooks

        assert json.loads(argv[argv.index("--settings") + 1]) == hooks.hooks_settings("node", w.main)
        assert argv[argv.index("--add-dir") + 1] == str(w.main)
        prompt = argv[-1]
        assert prompt.startswith("Build T1.\n\nControl Plane: ") and out["token"] in prompt
        assert launch.detach and "QS_CP_TOKEN" in launch.env_remove and launch.cwd == str(w.sim.wt)
        replay = spawn(w, "s1", model="claude-opus-5-5")[1]
        assert replay["replayed"] is True and replay["token"] == out["token"]

    def test_a_node_token_gets_no_token_back(self, w: W) -> None:
        node = insert_node(w.db, "N7", w.run, "T1", state="stopped", generation=9, name="x")
        code, out = run_cli("tool", "spawn", "--task", "T1", "--key", "z", "--token", node)
        assert code == 8 and out["error"] == "CONFLICT"  # spawn is run-only

    def test_live_generation_conflicts_unless_replace(self, w: W) -> None:
        spawn(w, "s1")
        code, out = spawn(w, "s2")
        assert code == 1 and out["result"]["error"] == "CONFLICT" and "--replace" in out["detail"]
        code, out = spawn(w, "s3", replace=True)
        assert code == 0 and out["result"]["generation"] == 2
        assert node_row(w, "N1")["state"] == "superseded"

    def test_replay_after_a_fault_at_reserve_reuses_the_row(self, w: W) -> None:
        with faults.arm("spawn.after_effect"), pytest.raises(faults.FaultInjected):
            spawn(w, "s1")
        assert node_row(w, "N1")["state"] == "spawning"
        assert spawn(w, "s1")[0] == 0
        assert sql(w.db, "SELECT count(*) FROM nodes")[0][0] == 1
        assert w.sim.effects("claude", "--bg") == 1

    def test_replay_after_the_reserved_row_was_reaped(self, w: W) -> None:
        with faults.arm("spawn.after_effect"), pytest.raises(faults.FaultInjected):
            spawn(w, "s1")
        reap(w)
        assert node_row(w, "N1")["state"] == "reaped"
        code, out = spawn(w, "s1")
        assert code == 0 and out["result"]["node_id"] == "N1"
        assert node_row(w, "N1")["state"] == "running" and w.sim.effects("claude", "--bg") == 1

    def test_replay_right_after_launch_does_not_relaunch(self, w: W) -> None:
        w.sim.list_on_launch = False
        code, out = spawn(w, "s1")
        assert code == 6 and "replay the same key" in out["detail"]  # identify timed out: retryable
        assert call_row(w, "spawn", "s1")["state"] == "started"
        sql(w.db, "UPDATE nodes SET launch_at = ?", [__import__("control_plane").clock.stamp(w.clock)])
        assert spawn(w, "s1")[0] == 6  # done-pending within LAUNCH_SETTLE_S: identify polls again, no relaunch
        assert w.sim.effects("claude", "--bg") == 1
        sql(w.db, "UPDATE nodes SET launch_at = ?", [__import__("control_plane").clock.stamp(w.clock)])
        reap(w)
        assert node_row(w, "N1")["state"] == "spawning"  # within LAUNCH_SETTLE_S: never reaped
        w.claude.listing = [agent("S-late", "r1-T1-g1", id="late")]
        code, out = spawn(w, "s1")
        assert code == 0 and out["result"]["session_id"] == "S-late"
        assert w.sim.effects("claude", "--bg") == 1

    def test_replay_after_the_session_died_and_the_row_was_reaped(self, w: W) -> None:
        w.sim.list_on_launch = False
        assert spawn(w, "s1")[0] == 6
        w.clock.advance(61)
        reap(w)
        assert node_row(w, "N1")["state"] == "reaped"
        w.sim.list_on_launch = True
        code, out = spawn(w, "s1")
        assert code == 0 and node_row(w, "N1")["state"] == "running"
        assert w.sim.effects("claude", "--bg") == 2  # relaunched once

    def test_state_drift_fails_cleanly(self, w: W) -> None:
        with faults.arm("spawn.after_effect"), pytest.raises(faults.FaultInjected):
            spawn(w, "s1")
        sql(w.db, "UPDATE nodes SET state = 'superseded'")
        code, out = spawn(w, "s1")
        assert code == 1 and out["result"]["error"] == "INVALID_STATE"
        assert call_row(w, "spawn", "s1")["state"] == "failed" and w.sim.effects("claude", "--bg") == 0

    def test_launch_failure(self, w: W) -> None:
        w.sim.launch_exit = 3
        code, out = spawn(w, "s1")
        assert code == 1 and out["result"]["output"] == {"exit_code": 3}

    def test_listing_failures_while_identifying(self, w: W) -> None:
        w.sim.list_on_launch = False
        w.claude.listing = None
        assert spawn(w, "s1")[0] == 6

    def test_argument_validation(self, w: W) -> None:
        assert tool(w, "spawn", "x", prompt_file=w.files["prompt"], agent="a")[1]["error"] == "USAGE"
        assert tool(w, "spawn", "x", agent="a", permission_mode="auto")[1]["error"] == "USAGE"
        assert (
            tool(w, "spawn", "x", agent="a", permission_mode="auto", prompt_file=str(w.tmp / "nope"))[1]["error"]
            == "USAGE"
        )
        insert_task(w.db, "T2", w.run)
        assert spawn(w, "x", task="T2")[1]["error"] == "INVALID_STATE"  # no worktree
        sql(w.db, "UPDATE tasks SET run_id = NULL WHERE id = 'T2'")
        sql(w.db, "UPDATE tasks SET worktree = 'x' WHERE id = 'T2'")
        assert spawn(w, "x", task="T2")[1]["error"] == "INVALID_STATE"  # no run


class TestResume:
    def _prep(self, w: W) -> None:
        _prep_resume(w)

    def resume(self, w: W, key: str) -> tuple[int, Any]:
        return tool(w, "resume", key, message_file=w.files["msg"])

    def test_bare_resume_keeps_the_generation(self, w: W) -> None:
        self._prep(w)
        w.sim.new_session_on_resume = True  # type: ignore[attr-defined]
        code, out = self.resume(w, "r1")
        assert code == 0, out
        [call] = w.runner.matching("claude", "--bg")
        assert call.argv == ["claude", "--bg", "--resume", "S-old", "Continue."]
        assert call.detach and "QS_CP_TOKEN" in call.env_remove
        node = node_row(w, "N1")
        assert (node["state"], node["generation"], node["session_id"]) == ("running", 1, "S-old-2")
        assert out["token"] == f"node:N1.1.{node['nonce']}"

    def test_refuses_a_live_node_and_one_that_never_launched(self, w: W) -> None:
        self._prep(w)
        w.claude.listing = [agent("S-old", "r1-T1-g1")]
        code, out = self.resume(w, "r1")
        assert code == 1 and out["result"]["error"] == "INVALID_STATE" and "SendMessage" in out["detail"]
        sql(w.db, "UPDATE nodes SET session_id = NULL")
        w.claude.listing = []
        code, out = self.resume(w, "r2")
        assert out["result"]["error"] == "INVALID_STATE" and "--replace" in out["detail"]
        sql(w.db, "UPDATE nodes SET session_id = 'S-old', state = 'running'")
        assert self.resume(w, "r3")[1]["result"]["error"] == "INVALID_STATE"
        assert w.sim.effects("claude", "--bg") == 0

    def test_failed_listing_is_busy_and_releases(self, w: W) -> None:
        self._prep(w)
        w.claude.listing = None
        assert self.resume(w, "r1")[0] == 6
        assert call_row(w, "resume", "r1") is None

    def test_replay_after_a_crash_past_launch_whose_row_was_reaped(self, w: W) -> None:
        self._prep(w)
        w.sim.list_on_launch = False
        with faults.arm("resume.after_effect", skip=1), pytest.raises(faults.FaultInjected):
            self.resume(w, "r1")
        w.clock.advance(61)
        reap(w)
        assert node_row(w, "N1")["state"] == "reaped"
        w.sim.list_on_launch = True
        code, out = self.resume(w, "r1")
        assert code == 0, out
        assert node_row(w, "N1")["state"] == "running" and w.sim.effects("claude", "--bg") == 2

    def test_a_stale_key_after_another_key_resumed_fails_cleanly(self, w: W) -> None:
        self._prep(w)
        with faults.arm("resume.after_effect"), pytest.raises(faults.FaultInjected):
            self.resume(w, "rA")
        sql(w.db, "UPDATE nodes SET state = 'running', spawn_tool_key = 'resume/rB'")
        code, out = self.resume(w, "rA")
        assert code == 1 and out["result"]["error"] == "INVALID_STATE"
        assert w.sim.effects("claude", "--bg") == 0


# --------------------------------------------------------------------------- per-tool specifics (AC14)


class TestWorktreeTools:
    def test_create_runs_in_main_under_the_main_checkout_lock(self, w: W) -> None:
        seen: dict[str, Any] = {}
        original = w.sim._setup

        def setup(call: Any) -> Any:
            seen["locks"] = [tuple(r) for r in sql(w.db, "SELECT name, holder_kind FROM locks")]
            seen["call"] = call
            return original(call)

        w.runner.on(["setup_task.py"], setup)
        shutil.rmtree(w.sim.wt)
        sql(w.db, "UPDATE tasks SET worktree = NULL, branch = NULL")
        code, out = tool(w, "worktree-create", "c1", phase="/diagnose-task")
        assert code == 0, out
        call = seen["call"]
        assert call.argv == [
            f"{w.main}/venv/bin/python",
            f"{w.main}/scripts/qs/setup_task.py",
            "11",
            "--harness",
            "claude-code",
            "--next-cmd",
            "/diagnose-task",
        ]
        assert call.cwd == str(w.main) and call.env_extra == {"QS_CP_TOKEN": w.token}
        assert seen["locks"] == [("main-checkout", "process")]
        task = dict(sql(w.db, "SELECT worktree, branch FROM tasks")[0])
        assert task == {"worktree": str(w.sim.wt.resolve()), "branch": "QS_11"}
        assert (w.main / ".git" / "hooks" / "pre-push").exists()
        assert sql(w.db, "SELECT count(*) FROM locks")[0][0] == 0

    def test_create_fails_before_any_effect_when_the_hook_is_refused(self, w: W) -> None:
        hook = w.main / ".git" / "hooks" / "pre-push"
        hook.parent.mkdir(parents=True)
        hook.write_text("#!/bin/sh\nexit 0\n")
        code, out = tool(w, "worktree-create", "c1", phase="/create-plan")
        assert code == 9 and "foreign" in out["detail"]
        assert w.sim.effects("setup_task.py") == 0 and call_row(w, "worktree-create", "c1") is None

    def test_create_validation_and_script_errors(self, w: W) -> None:
        from control_plane.runner import RunResult

        assert tool(w, "worktree-create", "c0", phase="/nope")[1]["error"] == "USAGE"
        w.runner.on(["setup_task.py"], RunResult(0, "not json", ""))
        assert tool(w, "worktree-create", "c1", phase="/create-plan")[1]["result"]["error"] == "TOOL_FAILED"
        w.runner.on(["setup_task.py"], RunResult(0, json.dumps({"branch": "QS_11"}), ""))
        assert tool(w, "worktree-create", "c2", phase="/create-plan")[1]["result"] == {
            "worktree": None,
            "branch": "QS_11",
        }
        sql(w.db, "UPDATE tasks SET issue_number = NULL")
        assert tool(w, "worktree-create", "c3", phase="/create-plan")[1]["error"] == "INVALID_STATE"

    def test_cleanup_classifies_on_removed(self, w: W) -> None:
        from control_plane.runner import RunResult

        call = None
        for status in ("removed-branch-kept", "action_required"):
            w.sim.cleanup_status = status
            code, out = tool(w, "worktree-cleanup", f"x-{status}")
            assert code == 1 and status in out["detail"]
        w.runner.on(["cleanup_worktree.py"], RunResult(0, "garbage", ""))
        assert tool(w, "worktree-cleanup", "x-garbage")[0] == 1
        w.runner.on(["cleanup_worktree.py"], RunResult(2, json.dumps({"status": "removed"}), ""))
        assert tool(w, "worktree-cleanup", "x-exit")[0] == 1
        w.runner.rules = [r for r in w.runner.rules if r[0] != ("cleanup_worktree.py",)]
        w.sim.install()
        w.sim.cleanup_status = "removed"
        code, out = tool(w, "worktree-cleanup", "x-ok")
        assert code == 0 and out["result"] == {"worktree": None}
        call = w.runner.matching("cleanup_worktree.py")[-1]
        assert call.argv[2:] == ["--work-dir", str(w.sim.wt), "--issue", "11", "--force"] and call.cwd == str(w.main)
        assert sql(w.db, "SELECT worktree FROM tasks")[0][0] is None

    def test_cleanup_prunes_a_stale_registration(self, w: W) -> None:
        shutil.rmtree(w.sim.wt)  # gone, but still registered
        code, out = tool(w, "worktree-cleanup", "p1")
        assert code == 0 and w.sim.effects("cleanup_worktree.py") == 0
        assert w.sim.effects("git", "-C", str(w.main), "worktree", "prune") == 1
        assert not w.sim.registered - {str(w.main)}

    def test_cleanup_when_already_clean_or_unknown(self, w: W) -> None:
        from control_plane.runner import RunResult

        shutil.rmtree(w.sim.wt)
        w.sim.registered.clear()
        assert tool(w, "worktree-cleanup", "p1")[0] == 0  # a fresh key sees the effect in place
        assert w.sim.effects("cleanup_worktree.py") == 0 and w.sim.effects("worktree", "prune") == 0
        assert tool(w, "worktree-cleanup", "p2")[0] == 0  # no worktree recorded any more
        sql(w.db, "UPDATE tasks SET worktree = ?", [str(w.sim.wt)])
        w.runner.on(["worktree", "list"], RunResult(1, "", "boom"))
        assert tool(w, "worktree-cleanup", "p3")[0] == 0  # listing failed: prune anyway
        w.runner.on(["worktree", "prune"], RunResult(1, "", "boom"))
        sql(w.db, "UPDATE tasks SET worktree = ?", [str(w.sim.wt)])
        assert tool(w, "worktree-cleanup", "p4")[0] == 1


class TestGate:
    def test_gate_runs_in_the_worktree_under_a_slot(self, w: W) -> None:
        seen = {}
        original = w.sim._gate

        def gate(call: Any) -> Any:
            seen["slots"] = sql(w.db, "SELECT count(*) FROM cap_slots WHERE cap = 'gates'")[0][0]
            return original(call)

        w.runner.on(["quality_gate.py"], gate)
        code, out = tool(w, "gate", "g1", mode="quick", paths=["tests/qs"])
        assert code == 0 and seen["slots"] == 1 and len(out["result"]["tail"]) == 60
        call = w.runner.matching("quality_gate.py")[-1]
        assert call.argv == [f"{w.sim.wt}/venv/bin/python", "scripts/qs/quality_gate.py", "--quick", "tests/qs"]
        assert call.cwd == str(w.sim.wt) and "QS_CP_TOKEN" in call.env_remove and not call.env_extra
        assert tool(w, "gate", "g2", mode="impacted")[0] == 0
        assert w.runner.matching("quality_gate.py")[-1].argv[-1] == "--impacted"

    def test_red_gate_and_bad_mode(self, w: W) -> None:
        w.sim.gate_exit = 1
        code, out = tool(w, "gate", "g1", mode="impacted")
        assert code == 1 and out["result"]["output"]["exit_code"] == 1 and len(out["result"]["output"]["tail"]) == 60
        assert tool(w, "gate", "g2", mode="quick")[1]["error"] == "USAGE"

    def test_node_token_on_its_own_task_and_stopped(self, w: W) -> None:
        node = insert_node(w.db, "N5", w.run, "T1")
        assert tool(w, "gate", "g1", token=node, mode="impacted")[0] == 0
        insert_task(w.db, "T2", w.run, worktree=str(w.sim.wt))
        assert tool(w, "gate", "g2", token=node, task="T2", mode="impacted")[1]["error"] == "CONFLICT"
        assert tool(w, "issue-create", "i1", token=node, **TABLE["issue-create"].args(w))[1]["error"] == "CONFLICT"
        sql(w.db, "UPDATE nodes SET state = 'stopped'")
        for name in ("gate", "push", "pr-create"):
            code, out = tool(
                w, name, "z", token=node, **({"mode": "impacted"} if name == "gate" else TABLE[name].args(w))
            )
            assert code == 4 and out["error"] == "STOPPED"


class TestIssueAndPr:
    def test_issue_create(self, w: W) -> None:
        code, out = tool(w, "issue-create", "msg:7", **TABLE["issue-create"].args(w))
        assert code == 0 and out["result"] == {"issue_number": 100, "url": "https://x/issues/100"}
        call = w.runner.matching("create_issue.py")[0]
        assert call.argv[2:4] == ["--title", "New"] and call.argv[-2:] == ["--labels", "kind:feature,target:factory"]
        assert call.argv[5] == "Issue body\n\n<!-- qs-cp-key: issue-create/msg:7 -->"
        assert sql(w.db, "SELECT issue_number FROM tasks")[0][0] == 100
        assert tool(w, "issue-create", "k2", title="x", body_file=w.files["body"], labels="a")[0] == 0
        assert w.runner.matching("create_issue.py")[1].argv[-1] == "a"
        assert tool(w, "issue-create", "k3", body_file=w.files["body"])[1]["error"] == "USAGE"

    @pytest.mark.parametrize("answer", [(0, "not json"), (0, json.dumps({"a": 1})), (1, "")])
    def test_a_failing_gh_listing_is_busy_and_creates_nothing(self, w: W, answer: tuple[int, str]) -> None:
        """F11: a failed probe is "unknown, replay later", never "not created"."""
        from control_plane.runner import RunResult

        w.runner.on(["gh", "issue", "list"], RunResult(answer[0], answer[1], "boom"))
        code, out = tool(w, "issue-create", "k1", **TABLE["issue-create"].args(w))
        assert code == 6 and out["error"] == "BUSY"
        assert w.sim.effects("create_issue.py") == 0 and call_row(w, "issue-create", "k1") is None
        w.runner.on(["gh", "pr", "list"], RunResult(answer[0], answer[1], "boom"))
        code, out = tool(w, "pr-create", "p1", **TABLE["pr-create"].args(w))
        assert code == 6 and w.sim.effects("create_pr.py") == 0

    def test_pr_create_with_a_new_key_after_a_closed_pr(self, w: W) -> None:
        assert tool(w, "pr-create", "p1", **TABLE["pr-create"].args(w))[1]["result"]["pr_number"] == 200
        w.sim.prs[0]["state"] = "CLOSED"
        assert tool(w, "pr-create", "p2", **TABLE["pr-create"].args(w))[1]["result"]["pr_number"] == 201
        assert sql(w.db, "SELECT pr_number, pr_url FROM tasks")[0][:] == (201, "https://x/pull/201")
        assert tool(w, "pr-create", "p3", summary_file=w.files["summary"])[1]["error"] == "USAGE"

    def test_pr_create_injects_a_token_that_passes_pre_push(self, w: W) -> None:
        from control_plane import hooks

        assert tool(w, "pr-create", "p1", **TABLE["pr-create"].args(w))[0] == 0
        call = w.runner.matching("create_pr.py")[0]
        assert call.cwd == str(w.sim.wt) and call.argv[-2:] == ["--issue", "11"]
        token = call.env_extra["QS_CP_TOKEN"]
        check = FakeRunner()
        check.on(["git", "rev-parse", "--show-toplevel"], str(w.sim.wt))
        assert hooks.pre_push_decision(
            "refs/heads/QS_11 a refs/heads/QS_11 b", clock=w.clock, run=check, token=token
        ) == (0, "")


class TestPushAndMerge:
    def test_push(self, w: W) -> None:
        code, out = tool(w, "push", "k1")
        assert code == 0 and out["result"] == {"sha": "a" * 40}
        call = w.runner.matching("git", "push")[0]
        assert call.argv == ["git", "push", "-u", "origin", "QS_11"] and call.env_extra == {"QS_CP_TOKEN": w.token}
        w.sim.head = "b" * 40
        assert tool(w, "push", "k2")[0] == 0 and w.sim.effects("git", "push") == 2
        assert tool(w, "push", "k3")[0] == 0 and w.sim.effects("git", "push") == 2  # already there

    def test_push_failures(self, w: W) -> None:
        from control_plane.runner import RunResult

        w.runner.on(["git", "push"], RunResult(1, "", "rejected"))
        assert tool(w, "push", "k1")[0] == 1
        w.runner.on(["git", "rev-parse", "HEAD"], RunResult(128, "", "x"))
        assert tool(w, "push", "k2")[0] == 1

    def test_merge_holds_its_locks_and_matches_the_head(self, w: W) -> None:
        _prep_merge(w)
        seen = {}
        original = w.sim._merge

        def merge(call: Any) -> Any:
            seen["locks"] = sorted(r[0] for r in sql(w.db, "SELECT name FROM locks"))
            return original(call)

        w.runner.on(["gh", "pr", "merge"], merge)
        assert tool(w, "merge", "m1")[0] == 0
        assert seen["locks"] == ["integration:QS_11", "main-merge"]
        call = w.runner.matching("gh", "pr", "merge")[0]
        assert call.argv == ["gh", "pr", "merge", "5", "--merge", "--match-head-commit", "a" * 40]
        assert call.cwd == str(w.main)

    def test_merge_replay_after_the_merge_skips_the_policy(self, w: W) -> None:
        _prep_merge(w)
        with faults.arm("merge.after_effect", skip=2), pytest.raises(faults.FaultInjected):
            tool(w, "merge", "m1")
        merge_policy.reset()  # would refuse
        code, out = tool(w, "merge", "m1")
        assert code == 0 and out["result"]["merge_sha"] == "m" * 40

    def test_merge_replay_before_the_merge_rejudges(self, w: W) -> None:
        _prep_merge(w)
        verdicts: list[int] = []

        def counting(task: Any) -> merge_policy.PolicyResult:
            verdicts.append(1)
            return merge_policy.PolicyResult(True, "ok")

        merge_policy.install(counting)
        with faults.arm("merge.after_effect", skip=1), pytest.raises(faults.FaultInjected):
            tool(w, "merge", "m1")
        assert tool(w, "merge", "m1")[0] == 0
        assert len(verdicts) == 2 and len(w.runner.matching("--json", "headRefOid")) == 2

    def test_merge_drift_and_failures(self, w: W) -> None:
        from control_plane.runner import RunResult

        _prep_merge(w)
        with faults.arm("merge.after_effect"), pytest.raises(faults.FaultInjected):
            tool(w, "merge", "m1")
        sql(w.db, "UPDATE tasks SET state = 'blocked', blocked_from = 'ready_to_merge'")
        assert tool(w, "merge", "m1")[1]["result"]["error"] == "INVALID_STATE"
        sql(w.db, "UPDATE tasks SET state = 'ready_to_merge'")
        w.runner.on(["--json", "headRefOid"], RunResult(1, "", "x"))
        assert tool(w, "merge", "m2")[0] == 1
        w.runner.on(["--json", "headRefOid"], RunResult(0, "nope", ""))
        assert tool(w, "merge", "m3")[0] == 1
        w.runner.rules = [r for r in w.runner.rules if r[0] != ("--json", "headRefOid")]
        w.runner.on(["gh", "pr", "merge"], RunResult(1, "", "not mergeable"))
        assert tool(w, "merge", "m4")[0] == 1

    def test_merged_pr_on_a_blocked_task_reports_a_state_conflict(self, w: W) -> None:
        _prep_merge(w)
        w.sim.pr_state = "MERGED"
        sql(w.db, "UPDATE tasks SET state = 'blocked', blocked_from = 'ready_to_merge'")
        code, out = tool(w, "merge", "m1")
        assert code == 0 and out["result"] == {
            "merge_sha": "m" * 40,
            "state_conflict": {"expected": "ready_to_merge", "actual": "blocked"},
        }
        assert tuple(sql(w.db, "SELECT state, merge_sha FROM tasks")[0]) == ("blocked", "m" * 40)


# --------------------------------------------------------------------------- the frozen API, the CLI, queue replay, races


class TestApi:
    def test_frozen_api_and_a_dummy_tool_through_the_cli(self, w: W) -> None:
        import dataclasses
        import inspect

        assert [f.name for f in dataclasses.fields(tools.Step)] == ["name", "run", "inject_token", "detach"]
        assert [f.name for f in dataclasses.fields(tools.ToolSpec)] == [
            "name",
            "steps",
            "locks",
            "cap",
            "probe",
            "guard",
            "on_success",
            "refuse_when_stopped",
            "token_kinds",
        ]
        assert {"task", "args", "outputs", "runner", "claude", "main"} <= {
            f.name for f in dataclasses.fields(tools.StepCtx)
        }
        assert list(inspect.signature(tools.invoke).parameters)[:6] == [
            "name",
            "key",
            "task_id",
            "args",
            "token",
            "actor",
        ]
        seen: dict[str, Any] = {}

        def record(ctx: tools.StepCtx) -> dict[str, Any]:
            with ctx.write() as conn:
                conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('dummy', ?)", (ctx.key,))
                ctx.record_output({"wrote": True})
            seen["locks"] = sorted(r[0] for r in sql(w.db, "SELECT name FROM locks"))
            seen["slots"] = sql(w.db, "SELECT count(*) FROM cap_slots")[0][0]
            return {"ignored": True}

        spec = tools.ToolSpec(
            "dummy",
            lambda task, args: (
                tools.Step("record", record),
                tools.argv_step("echo", lambda ctx: ["echo", str(ctx.args["word"])], lambda ctx: ctx.main),
            ),
            locks=lambda task, args: (f"integration:{task['branch']}",),
            cap="gates",
            probe=lambda ctx: {"echo": None},
            on_success=lambda conn, ctx: {"outputs": dict(ctx.outputs)},
        )
        tools.register(spec)
        with pytest.raises(tools.errors.CpError) as exc:
            tools.register(spec)
        assert exc.value.code == "CONFLICT"
        code, out = tool(w, "dummy", "d1", word="hi")
        assert code == 0, out
        assert out["result"]["outputs"] == {"record": {"wrote": True}, "echo": {"exit_code": 0, "tail": []}}
        assert seen == {"locks": ["integration:QS_11"], "slots": 1}
        assert sql(w.db, "SELECT value FROM meta WHERE key = 'dummy'")[0][0] == "d1"
        from control_plane.runner import RunResult

        w.runner.on(["echo"], RunResult(2, "x", "y"))
        code, out = tool(w, "dummy", "d2", word="hi")

        assert code == 1 and out["result"]["output"] == {"exit_code": 2, "tail": ["x", "y"]}
        tools.reset()
        assert "dummy" not in tools.REGISTRY and "spawn" in tools.REGISTRY

    def test_every_registered_tool_takes_its_locks_in_order(self) -> None:
        task = {"id": "T1", "branch": "QS_1_1", "pr_number": 1, "issue_number": 1, "worktree": "/w", "run_id": "R1"}
        for spec in tools.REGISTRY.values():
            locks.assert_sorted(list(spec.locks(task, {})))  # type: ignore[arg-type]

    def test_invoke_and_validation(self, w: W, monkeypatch) -> None:
        with pytest.raises(tools.errors.CpError) as exc:
            tools.invoke("nope", key="k", task_id="T1", args={}, token=w.token, actor="x")
        assert exc.value.code == "NOT_FOUND"
        assert tool(w, "push", "")[1]["error"] == "USAGE"
        assert tool(w, "push", "k" * 201)[1]["error"] == "USAGE"
        bad = w.tmp / "list.json"
        bad.write_text("[1]")
        assert (
            run_cli("tool", "push", "--task", "T1", "--key", "k", "--args-file", str(bad), "--token", w.token)[1][
                "error"
            ]
            == "USAGE"
        )
        assert run_cli("tool", "push", "--task", "T1", "--key", "k1", "--token", w.token)[0] == 0
        ctx = tools.default_ctx()
        assert ctx.main == w.main and isinstance(ctx.probe, tools.liveness.ProcessProbe)
        ctx.conn_factory().close()
        monkeypatch.setattr(
            tools,
            "default_ctx",
            lambda: tools.Ctx(w.runner, lambda: db.connect(w.db), w.clock, w.probe, w.claude, w.main),
        )
        assert tools.invoke("push", key="k2", task_id="T1", args={}, token=w.token, actor="me")["replayed"] is False
        assert call_row(w, "push", "k2")["actor"] == "me"


def _pop(w: W) -> dict:
    return run_cli("msg", "pop", "--run", w.run, "--as", "orchestrator", "--token", w.token)[1]


@pytest.mark.parametrize("name", ["issue-create", "spawn", "merge"])
def test_queue_replay(w: W, name: str) -> None:
    from control_plane import messages

    t = TABLE[name]
    t.prep(w)
    payload = w.tmp / "p.json"
    payload.write_text("{}")
    run_cli(
        "msg",
        "post",
        "--run",
        w.run,
        "--to",
        "orchestrator",
        "--kind",
        "go",
        "--payload-file",
        str(payload),
        "--token",
        w.token,
    )
    first = _pop(w)
    assert tool(w, name, f"msg:{first['id']}", **t.args(w))[0] == 0
    w.clock.advance(messages.VISIBILITY_S + 1)  # never acked: redelivered
    again = _pop(w)
    assert again["id"] == first["id"] and again["attempt"] == 2
    code, out = tool(w, name, f"msg:{again['id']}", **t.args(w))
    assert code == 0 and out["replayed"] is True
    assert t.effect(w) == 1


class TestNodeCapRaces:
    def _barrier(self, w: W) -> None:
        barrier = threading.Barrier(2)
        count = {"n": 0}
        lock = threading.Lock()

        def before_list() -> None:
            with lock:
                count["n"] += 1
                mine = count["n"] <= 2
            if mine:
                barrier.wait(timeout=2)

        w.claude.before_list = before_list

    def _race(self, *calls: Callable[[], tuple[int, Any]]) -> list[int]:
        results: list[int] = []
        threads = [threading.Thread(target=lambda c=c: results.append(c()[0])) for c in calls]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        return sorted(results)

    def test_two_spawns_at_cap_minus_one(self, w: W, monkeypatch) -> None:
        monkeypatch.setenv("QS_CP_MAX_NODES", "2")
        for task in ("T2", "T3", "T4"):
            insert_task(w.db, task, w.run, worktree=str(w.sim.wt))
        insert_node(w.db, "N9", w.run, "T4", session_id="S-9", name="n9")
        w.claude.listing = [agent("S-9", "n9")]
        self._barrier(w)
        assert self._race(lambda: spawn(w, "a", task="T2"), lambda: spawn(w, "b", task="T3")) == [0, 6]

    def test_a_resume_racing_a_spawn(self, w: W, monkeypatch) -> None:
        monkeypatch.setenv("QS_CP_MAX_NODES", "1")
        _prep_resume(w)
        insert_task(w.db, "T2", w.run, worktree=str(w.sim.wt))
        self._barrier(w)
        sql(w.db, "INSERT INTO counters (kind, next) VALUES ('node', 100)")  # N1 was inserted directly
        results = self._race(
            lambda: spawn(w, "a", task="T2"), lambda: tool(w, "resume", "r", message_file=w.files["msg"])
        )
        assert results == [0, 6]


def test_resume_reserve_sees_a_node_that_came_alive(w: W) -> None:
    _prep_resume(w)

    def came_alive() -> None:  # between the guard and the reserve transaction, the node is woken and listed
        if w.claude.before_list is not None:
            w.claude.before_list = None
            sql(w.db, "UPDATE nodes SET state = 'running'")
            w.claude.listing = [agent("S-old", "r1-T1-g1")]

    w.claude.before_list = came_alive
    code, out = tool(w, "resume", "r1", message_file=w.files["msg"])
    assert code == 1 and out["result"]["error"] == "INVALID_STATE" and "SendMessage" in out["detail"]


def test_resume_probe_ignores_its_row_once_running(w: W) -> None:
    _prep_resume(w)
    with faults.arm("resume.after_effect"), pytest.raises(faults.FaultInjected):
        tool(w, "resume", "rA", message_file=w.files["msg"])
    sql(w.db, "UPDATE nodes SET state = 'running'")  # still spawn_tool_key = resume/rA
    code, out = tool(w, "resume", "rA", message_file=w.files["msg"])
    assert code == 1 and out["result"]["error"] == "INVALID_STATE"


# --------------------------------------------------------------------------- review fix #01


class TestReviewFix01:
    # F3
    def test_replace_at_the_node_cap(self, w: W, monkeypatch) -> None:
        monkeypatch.setenv("QS_CP_MAX_NODES", "1")
        assert spawn(w, "s1")[0] == 0
        code, out = spawn(w, "s2", replace=True)
        assert code == 0, out
        assert node_row(w, "N1")["state"] == "superseded" and out["result"]["generation"] == 2

    def test_a_busy_replace_keeps_the_live_node(self, w: W, monkeypatch) -> None:
        monkeypatch.setenv("QS_CP_MAX_NODES", "1")
        insert_task(w.db, "T2", w.run, worktree=str(w.sim.wt))
        insert_node(w.db, "N7", w.run, "T1", session_id="S-7", name="n7")
        insert_node(w.db, "N9", w.run, "T2", session_id="S-9", name="n9")
        w.claude.listing = [agent("S-7", "n7"), agent("S-9", "n9")]
        code, out = spawn(w, "s1", replace=True)
        assert code == 6 and out["error"] == "BUSY"
        assert node_row(w, "N7")["state"] == "idle"  # refreshed from the listing; the supersede was rolled back
        assert w.sim.effects("claude", "--bg") == 0

    # F10
    def test_replay_after_the_prompt_file_was_deleted(self, w: W) -> None:
        code, first = spawn(w, "s1")
        assert code == 0
        Path(w.files["prompt"]).unlink()
        code, again = spawn(w, "s1")
        assert code == 0 and again["replayed"] is True and again["result"] == first["result"]

    def test_replay_of_push_after_the_worktree_was_cleared(self, w: W) -> None:
        code, first = tool(w, "push", "k1")
        assert code == 0
        sql(w.db, "UPDATE tasks SET worktree = NULL")  # worktree-cleanup ran meanwhile
        code, again = tool(w, "push", "k1")
        assert code == 0 and again["replayed"] is True and again["result"] == first["result"]

    def test_a_replayed_failure_needs_no_steps_either(self, w: W) -> None:
        from control_plane.runner import RunResult

        w.runner.on(["git", "push"], RunResult(1, "", "rejected"))
        assert tool(w, "push", "k1")[0] == 1
        sql(w.db, "UPDATE tasks SET worktree = NULL")
        code, out = tool(w, "push", "k1")
        assert code == 1 and out["error"] == "TOOL_FAILED" and out["replayed"] is True

    def test_an_edited_file_under_the_same_key_conflicts(self, w: W) -> None:
        args = TABLE["issue-create"].args(w)
        assert tool(w, "issue-create", "k1", **args)[0] == 0
        Path(w.files["body"]).write_text("Another body")
        code, out = tool(w, "issue-create", "k1", **args)
        assert code == 8 and out["error"] == "CONFLICT"
        Path(w.files["body"]).write_text("Issue body")
        assert tool(w, "issue-create", "k1", **args)[1]["replayed"] is True

    def test_an_edited_file_conflicts_on_a_started_call_too(self, w: W) -> None:
        args = TABLE["issue-create"].args(w)
        with faults.arm("issue-create.before_effect"), pytest.raises(faults.FaultInjected):
            tool(w, "issue-create", "k1", **args)
        Path(w.files["body"]).write_text("Another body")
        assert tool(w, "issue-create", "k1", **args)[1]["error"] == "CONFLICT"

    # F11: see TestReviewFix02 (G1) — a consistent listing first, then the marker search

    # F12
    def test_cleanup_refuses_the_main_checkout(self, w: W) -> None:
        (w.main / ".git" / "config").write_text("")
        sql(w.db, "UPDATE tasks SET worktree = ?", [str(w.main)])
        code, out = tool(w, "worktree-cleanup", "c1")
        assert code == 9 and out["error"] == "POLICY_REFUSED"
        assert w.sim.effects("cleanup_worktree.py") == 0 and call_row(w, "worktree-cleanup", "c1") is None

    def test_cleanup_refuses_a_directory_that_is_not_a_linked_worktree(self, w: W) -> None:
        (w.sim.wt / ".git").unlink()
        (w.sim.wt / ".git").mkdir()  # a full clone, not a linked worktree
        code, out = tool(w, "worktree-cleanup", "c1")
        assert code == 9 and out["error"] == "POLICY_REFUSED" and ".git" in out["detail"]
        assert w.sim.effects("cleanup_worktree.py") == 0 and call_row(w, "worktree-cleanup", "c1") is None  # G20

    def test_cleanup_of_a_shared_worktree_only_clears_this_task(self, w: W) -> None:
        insert_task(w.db, "T2", w.run, worktree=str(w.sim.wt), branch="QS_11")
        insert_task(w.db, "T3", w.run, state="dropped", worktree=str(w.sim.wt))  # terminal: does not count
        code, out = tool(w, "worktree-cleanup", "c1")
        assert code == 0 and out["result"] == {"worktree": None, "shared_with": "T2"}
        assert w.sim.effects("cleanup_worktree.py") == 0 and w.sim.effects("worktree", "prune") == 0
        assert w.sim.wt.exists()
        assert {r[0]: r[1] for r in sql(w.db, "SELECT id, worktree FROM tasks")} == {
            "T1": None,
            "T2": str(w.sim.wt),
            "T3": str(w.sim.wt),
        }

    # F14
    def test_a_task_blocked_during_the_merge_still_records_the_merge(self, w: W) -> None:
        _prep_merge(w)
        original = w.sim._merge

        def merge(call: Any) -> Any:
            sql(w.db, "UPDATE tasks SET state = 'blocked', blocked_from = 'ready_to_merge'")
            return original(call)

        w.runner.on(["gh", "pr", "merge"], merge)
        code, out = tool(w, "merge", "m1")
        assert code == 0 and out["result"]["state_conflict"] == {"expected": "ready_to_merge", "actual": "blocked"}
        assert tuple(sql(w.db, "SELECT state, merge_sha FROM tasks")[0]) == ("blocked", "m" * 40)
        assert call_row(w, "merge", "m1")["state"] == "succeeded"

    # F16
    def test_cli_tool_locks_and_slots_carry_the_actor(self, w: W) -> None:
        seen: dict[str, Any] = {}
        original_setup, original_gate = w.sim._setup, w.sim._gate

        def setup(call: Any) -> Any:
            seen["lock"] = sql(w.db, "SELECT holder_actor FROM locks")[0][0]
            return original_setup(call)

        def gate(call: Any) -> Any:
            seen["slot"] = sql(w.db, "SELECT holder_actor FROM cap_slots")[0][0]
            return original_gate(call)

        w.runner.on(["setup_task.py"], setup)
        w.runner.on(["quality_gate.py"], gate)
        assert tool(w, "worktree-create", "c1", phase="/create-plan")[0] == 0
        assert tool(w, "gate", "g1", mode="impacted")[0] == 0
        assert seen == {"lock": "orchestrator", "slot": "orchestrator"}

    # F19
    def test_a_pending_launch_that_never_appeared_is_relaunched(self, w: W) -> None:
        w.sim.list_on_launch = False
        assert spawn(w, "s1")[0] == 6  # identify timed out at the settle deadline; this process exited
        w.sim.list_on_launch = True
        code, out = spawn(w, "s1")
        assert code == 0, out
        assert node_row(w, "N1")["state"] == "running" and w.sim.effects("claude", "--bg") == 2

    def test_a_pending_launch_is_kept_while_the_listing_fails(self, w: W) -> None:
        w.sim.list_on_launch = False
        assert spawn(w, "s1")[0] == 6
        w.claude.listing = None
        assert spawn(w, "s1")[0] == 6
        assert node_row(w, "N1")["state"] == "spawning" and w.sim.effects("claude", "--bg") == 1

    # F22
    def test_a_claim_taken_over_mid_step_is_stale(self, w: W) -> None:
        original = w.sim._push

        def taken(call: Any) -> Any:
            sql(w.db, "UPDATE tool_calls SET holder_pid = 999, holder_pgid = 999")
            return original(call)

        w.runner.on(["git", "push"], taken)
        code, out = tool(w, "push", "k1")
        assert code == 8 and out["error"] == "CONFLICT" and out["claim_taken_over"] is True  # G12
        assert call_row(w, "push", "k1")["holder_pid"] == 999

    def test_a_claim_taken_over_before_record_output_is_stale(self, w: W) -> None:
        def taken() -> None:
            w.claude.before_list = None
            sql(w.db, "UPDATE tool_calls SET holder_pid = 999, holder_pgid = 999")

        w.claude.before_list = taken
        code, out = spawn(w, "s1")
        assert code == 8 and out["claim_taken_over"] is True and sql(w.db, "SELECT count(*) FROM nodes")[0][0] == 0

    def test_a_claim_taken_over_before_succeed_or_fail_is_stale(self, w: W) -> None:
        def take(ctx: tools.StepCtx) -> None:
            sql(w.db, "UPDATE tool_calls SET holder_pid = 999, holder_pgid = 999")

        def boom(ctx: tools.StepCtx) -> Any:
            take(ctx)
            raise tools.StepFailed("boom")

        tools.register(tools.ToolSpec("t-succeed", lambda task, args: (), guard=take))
        tools.register(tools.ToolSpec("t-fail", lambda task, args: (tools.Step("boom", boom),)))
        assert tool(w, "t-succeed", "k1")[1]["claim_taken_over"] is True
        assert tool(w, "t-fail", "k1")[1]["claim_taken_over"] is True
        assert call_row(w, "t-fail", "k1")["state"] == "started"

    # F24 (acceptance-auditor depth)
    def test_worktree_create_with_a_fresh_key_on_an_existing_worktree(self, w: W) -> None:
        assert tool(w, "worktree-create", "c1", phase="/create-plan")[0] == 0  # the fixture's worktree exists
        assert tool(w, "worktree-create", "c2", phase="/create-plan")[0] == 0
        assert w.sim.created == 0 and w.sim.effects("setup_task.py") == 2  # idempotent: nothing created twice
        shutil.rmtree(w.sim.wt)
        assert tool(w, "worktree-create", "c3", phase="/create-plan")[0] == 0
        assert tool(w, "worktree-create", "c4", phase="/create-plan")[0] == 0
        assert w.sim.created == 1

    def test_an_over_cap_resume_is_refused_before_any_launch(self, w: W, monkeypatch) -> None:
        monkeypatch.setenv("QS_CP_MAX_NODES", "1")
        _prep_resume(w)
        insert_task(w.db, "T2", w.run, worktree=str(w.sim.wt))
        insert_node(w.db, "N9", w.run, "T2", session_id="S-9", name="n9")
        w.claude.listing = [agent("S-9", "n9")]
        code, out = tool(w, "resume", "r1", message_file=w.files["msg"])
        assert code == 6 and out["error"] == "BUSY" and "node cap" in out["detail"]
        assert w.sim.effects("claude", "--bg") == 0 and call_row(w, "resume", "r1") is None
        assert node_row(w, "N1")["state"] == "reaped"

    def test_a_call_finished_between_the_replay_check_and_the_claim(self, w: W) -> None:
        ctx = tools.Ctx(w.runner, lambda: db.connect(w.db), w.clock, w.probe, w.claude, w.main)
        nested = {"done": False}

        def steps(task: Any, args: Any) -> tuple[tools.Step, ...]:
            if not nested["done"]:  # another process runs the same key to completion right now
                nested["done"] = True
                tools.invoke("t-race", key="k1", task_id="T1", args={}, token=w.token, actor="other", ctx=ctx)
            return ()

        tools.register(tools.ToolSpec("t-race", steps))
        out = tools.invoke("t-race", key="k1", task_id="T1", args={}, token=w.token, actor="me", ctx=ctx)
        assert out["replayed"] is True and call_row(w, "t-race", "k1")["actor"] == "other"


# --------------------------------------------------------------------------- review fix #02


def _issue_lists(w: W) -> tuple[list[Any], list[Any]]:
    calls = w.runner.matching("gh", "issue", "list")
    return [c for c in calls if "--search" not in c.argv], [c for c in calls if "--search" in c.argv]


class TestReviewFix02:
    # G1
    def test_the_issue_probe_lists_first_then_searches(self, w: W) -> None:
        assert tool(w, "issue-create", "msg:7", **TABLE["issue-create"].args(w))[0] == 0
        [listing], [search] = _issue_lists(w)
        assert "--search" not in listing.argv
        assert listing.argv[listing.argv.index("--limit") + 1] == "100"
        assert search.argv[search.argv.index("--search") + 1] == '"qs-cp-key: issue-create/msg:7" in:body'
        assert search.argv[search.argv.index("--limit") + 1] == "100"

    def test_a_marker_in_the_listing_makes_no_search(self, w: W) -> None:
        args = TABLE["issue-create"].args(w)
        with faults.arm("issue-create.after_effect"), pytest.raises(faults.FaultInjected):
            tool(w, "issue-create", "k1", **args)
        w.runner.calls.clear()
        code, out = tool(w, "issue-create", "k1", **args)
        assert code == 0 and out["result"]["issue_number"] == 100
        listings, searches = _issue_lists(w)
        assert len(listings) == 1 and searches == []
        assert w.sim.effects("create_issue.py") == 0 and len(w.sim.issues) == 1  # no second issue

    def test_a_listing_miss_then_a_search_hit_is_done(self, w: W) -> None:
        from control_plane.runner import RunResult

        args = TABLE["issue-create"].args(w)
        with faults.arm("issue-create.after_effect"), pytest.raises(faults.FaultInjected):
            tool(w, "issue-create", "k1", **args)
        w.runner.on(["issue", "list", "--state", "all", "--limit"], RunResult(0, "[]", ""))  # lagging listing
        code, out = tool(w, "issue-create", "k1", **args)
        assert code == 0 and out["result"]["issue_number"] == 100
        assert w.sim.effects("create_issue.py") == 1

    def test_a_search_hit_must_hold_the_exact_marker(self, w: W) -> None:
        from control_plane.runner import RunResult

        other = [{"number": 9, "url": "u", "body": "<!-- qs-cp-key: issue-create/k10 -->"}]
        w.runner.on(["--search"], RunResult(0, json.dumps(other), ""))
        assert tool(w, "issue-create", "k1", **TABLE["issue-create"].args(w))[0] == 0
        assert w.sim.effects("create_issue.py") == 1

    @pytest.mark.parametrize("which", ["listing", "search"])
    def test_a_failed_listing_or_search_is_busy(self, w: W, which: str) -> None:
        from control_plane.runner import RunResult

        pattern = ["--search"] if which == "search" else ["issue", "list", "--state", "all", "--limit"]
        w.runner.on(pattern, RunResult(1, "", "boom"))
        code, out = tool(w, "issue-create", "k1", **TABLE["issue-create"].args(w))
        assert code == 6 and out["error"] == "BUSY" and w.sim.effects("create_issue.py") == 0
        w.runner.on(pattern, RunResult(0, "{}", ""))
        assert tool(w, "issue-create", "k1", **TABLE["issue-create"].args(w))[0] == 6

    @pytest.mark.parametrize("key", ['a"b', "a b", "a\\b", "é"])
    def test_keys_are_restricted(self, w: W, key: str) -> None:
        code, out = tool(w, "push", key)
        assert code == 2 and out["error"] == "USAGE" and w.sim.effects("git", "push") == 0

    def test_documented_key_forms_are_accepted(self, w: W) -> None:
        for key in ("msg:12", "task:T3:gate", "a.b_c-d/e"):
            assert tool(w, "push", key)[0] == 0

    # G2
    def test_a_relaunch_rotates_the_token_and_the_name(self, w: W) -> None:
        import re

        w.sim.list_on_launch = False
        assert spawn(w, "s1")[0] == 6
        [first] = w.runner.matching("claude", "--bg")
        old_name = first.argv[first.argv.index("-n") + 1]
        old_token = re.search(r"token (node:\S+)", first.argv[-1])[1]  # type: ignore[index]
        w.clock.advance(61)
        reap(w)
        w.sim.list_on_launch = True
        code, out = spawn(w, "s1")
        assert code == 0, out
        second = w.runner.matching("claude", "--bg")[-1]
        new_name = second.argv[second.argv.index("-n") + 1]
        assert (old_name, new_name) == ("r1-T1-g1", "r1-T1-g1-r1")
        assert out["token"] != old_token and out["token"] in second.argv[-1]
        assert node_row(w, "N1")["name"] == new_name and out["result"]["name"] == new_name
        code, res = tool(w, "gate", "g1", token=old_token, mode="impacted")
        assert code == 3 and res["error"] == "STALE_TOKEN"
        assert tool(w, "gate", "g2", token=out["token"], mode="impacted")[0] == 0

    def test_a_second_relaunch_counts_up(self, w: W) -> None:
        w.sim.list_on_launch = False
        assert spawn(w, "s1")[0] == 6
        for _ in range(2):
            w.clock.advance(61)
            reap(w)
            assert spawn(w, "s1")[0] == 6
        names = [c.argv[c.argv.index("-n") + 1] for c in w.runner.matching("claude", "--bg")]
        assert names == ["r1-T1-g1", "r1-T1-g1-r1", "r1-T1-g1-r2"]

    def test_a_late_listing_of_the_old_name_is_not_adopted(self, w: W) -> None:
        from control_plane import clock as clock_mod

        w.sim.list_on_launch = False
        assert spawn(w, "s1")[0] == 6
        w.clock.advance(61)
        reap(w)
        assert spawn(w, "s1")[0] == 6  # relaunched as r1-T1-g1-r1, never listed yet
        sql(w.db, "UPDATE nodes SET launch_at = ?", [clock_mod.stamp(w.clock)])
        w.claude.listing = [agent("S-first", "r1-T1-g1", id="first")]  # the first launch shows up late
        assert spawn(w, "s1")[0] == 6
        assert node_row(w, "N1")["session_id"] is None and w.sim.effects("claude", "--bg") == 2
        sql(w.db, "UPDATE nodes SET launch_at = ?", [clock_mod.stamp(w.clock)])
        w.claude.listing = [agent("S-first", "r1-T1-g1"), agent("S-second", "r1-T1-g1-r1", id="second")]
        code, out = spawn(w, "s1")
        assert code == 0 and out["result"]["session_id"] == "S-second"

    def test_the_reap_of_an_own_row_that_moved_meanwhile_is_a_noop(self, w: W) -> None:
        w.sim.list_on_launch = False
        assert spawn(w, "s1")[0] == 6
        w.clock.advance(61)
        real = w.claude.try_agents

        def moved() -> Any:
            sql(
                w.db, "UPDATE nodes SET state = 'superseded'"
            )  # another process changed the row after the probe read it
            w.claude.try_agents = real  # type: ignore[method-assign]
            return real()

        w.claude.try_agents = moved  # type: ignore[method-assign]
        code, out = spawn(w, "s1")
        assert code == 1 and out["result"]["error"] == "INVALID_STATE"
        assert node_row(w, "N1")["state"] == "superseded"

    # G5
    def test_a_takeover_after_the_effect_needs_no_file(self, w: W) -> None:
        args = TABLE["issue-create"].args(w)
        with faults.arm("issue-create.after_effect"), pytest.raises(faults.FaultInjected):
            tool(w, "issue-create", "k1", **args)
        Path(w.files["body"]).unlink()
        code, out = tool(w, "issue-create", "k1", **args)
        assert code == 0, out
        assert w.sim.effects("create_issue.py") == 1

    def test_a_spawn_takeover_after_the_launch_needs_no_prompt(self, w: W) -> None:
        with faults.arm("spawn.after_effect", skip=1), pytest.raises(faults.FaultInjected):
            spawn(w, "s1")
        Path(w.files["prompt"]).unlink()
        code, out = spawn(w, "s1")
        assert code == 0, out
        assert w.sim.effects("claude", "--bg") == 1

    def test_a_takeover_before_the_effect_with_the_file_gone_is_usage_and_retryable(self, w: W) -> None:
        args = TABLE["issue-create"].args(w)
        with faults.arm("issue-create.before_effect"), pytest.raises(faults.FaultInjected):
            tool(w, "issue-create", "k1", **args)
        Path(w.files["body"]).unlink()
        code, out = tool(w, "issue-create", "k1", **args)
        assert code == 2 and out["error"] == "USAGE" and w.sim.effects("create_issue.py") == 0
        assert call_row(w, "issue-create", "k1") is None  # released: no effect happened
        Path(w.files["body"]).write_text("Issue body")
        assert tool(w, "issue-create", "k1", **args)[0] == 0

    def test_a_fresh_call_with_a_missing_file_has_no_effect(self, w: W) -> None:
        for name in ("spawn", "resume", "issue-create", "pr-create"):
            t = TABLE[name]
            t.prep(w)
            args = {k: (str(w.tmp / "missing.md") if k.endswith("_file") else v) for k, v in t.args(w).items()}
            code, out = tool(w, name, f"m-{name}", **args)
            assert code == 2 and out["error"] == "USAGE", (name, out)
            assert call_row(w, name, f"m-{name}") is None
        assert sql(w.db, "SELECT count(*) FROM nodes WHERE state = 'spawning'")[0][0] == 0
        assert w.sim.effects("claude", "--bg") == 0 and w.sim.effects("create_issue.py") == 0

    def test_a_non_regular_file_is_usage(self, w: W) -> None:
        import os

        fifo = w.tmp / "fifo"
        os.mkfifo(fifo)
        code, out = tool(w, "spawn", "f1", prompt_file=str(fifo), **SPAWN)
        assert code == 2 and out["error"] == "USAGE" and "regular" in out["detail"]
        code, out = tool(w, "spawn", "f2", prompt_file=str(w.tmp), **SPAWN)
        assert code == 2 and out["error"] == "USAGE"

    def test_a_file_is_read_once(self, w: W, monkeypatch) -> None:
        reads: list[str] = []
        real = Path.read_bytes

        def counting(self: Path) -> bytes:
            reads.append(str(self))
            return real(self)

        monkeypatch.setattr(Path, "read_bytes", counting)
        assert tool(w, "issue-create", "k1", **TABLE["issue-create"].args(w))[0] == 0
        assert reads.count(w.files["body"]) == 1

    def test_a_file_that_is_not_utf8_is_usage(self, w: W) -> None:
        Path(w.files["body"]).write_bytes(b"\xff\xfe\x00bad")
        code, out = tool(w, "issue-create", "k1", **TABLE["issue-create"].args(w))
        assert code == 2 and out["error"] == "USAGE" and call_row(w, "issue-create", "k1") is None

    def test_an_unreadable_file_counts_as_missing(self, w: W, monkeypatch) -> None:
        real = Path.read_bytes

        def refused(self: Path) -> bytes:
            if str(self) == w.files["body"]:
                raise PermissionError("denied")
            return real(self)

        monkeypatch.setattr(Path, "read_bytes", refused)
        code, out = tool(w, "issue-create", "k1", **TABLE["issue-create"].args(w))
        assert code == 2 and out["error"] == "USAGE"

    # G10
    @pytest.mark.parametrize("answer", [(1, ""), (0, "nope"), (0, "[]")])
    def test_an_unknown_merge_probe_is_busy(self, w: W, answer: tuple[int, str]) -> None:
        from control_plane.runner import RunResult

        _prep_merge(w)
        w.runner.on(["--json", "state,mergeCommit"], RunResult(answer[0], answer[1], "x"))
        code, out = tool(w, "merge", "m1")
        assert code == 6 and out["error"] == "BUSY"
        assert w.sim.effects("gh", "pr", "merge") == 0 and call_row(w, "merge", "m1") is None

    # G11
    def test_a_missing_merge_commit_is_read_again_once(self, w: W) -> None:
        from control_plane.runner import RunResult

        _prep_merge(w)
        answers = iter([json.dumps({"mergeCommit": None}), json.dumps({"mergeCommit": {"oid": "n" * 40}})])
        w.runner.on(["--json", "mergeCommit"], lambda c: RunResult(0, next(answers), ""))
        code, out = tool(w, "merge", "m1")
        assert code == 0 and out["result"]["merge_sha"] == "n" * 40
        assert len(w.runner.matching("--json", "mergeCommit")) == 2

    def test_a_merge_sha_never_overwrites_with_null(self, w: W) -> None:
        from control_plane.runner import RunResult

        _prep_merge(w)
        sql(w.db, "UPDATE tasks SET merge_sha = ?", ["o" * 40])
        w.runner.on(["--json", "mergeCommit"], RunResult(0, json.dumps({"mergeCommit": None}), ""))
        code, out = tool(w, "merge", "m1")
        assert code == 0 and out["result"]["merge_sha"] is None
        assert tuple(sql(w.db, "SELECT state, merge_sha FROM tasks")[0]) == ("merged", "o" * 40)

    def test_an_already_merged_pr_with_no_commit_is_read_again(self, w: W) -> None:
        from control_plane.runner import RunResult

        _prep_merge(w)
        answers = iter(
            [json.dumps({"state": "MERGED", "mergeCommit": None}), json.dumps({"mergeCommit": {"oid": "p" * 40}})]
        )
        w.runner.on(["gh", "pr", "view"], lambda c: RunResult(0, next(answers), ""))
        code, out = tool(w, "merge", "m1")
        assert code == 0 and out["result"]["merge_sha"] == "p" * 40 and w.sim.effects("gh", "pr", "merge") == 0

    def test_a_task_already_merged_is_a_noop_success(self, w: W) -> None:
        _prep_merge(w)
        w.sim.pr_state = "MERGED"
        sql(w.db, "UPDATE tasks SET state = 'merged'")
        code, out = tool(w, "merge", "m1")
        assert code == 0 and out["result"] == {"merge_sha": "m" * 40, "noop": True}
        assert sql(w.db, "SELECT count(*) FROM hook_events")[0][0] == 0

    # G12: the F22 tests above now expect CONFLICT with claim_taken_over

    # G21
    def test_a_state_conflict_is_also_an_alert(self, w: W) -> None:
        _prep_merge(w)
        w.sim.pr_state = "MERGED"
        sql(w.db, "UPDATE tasks SET state = 'blocked', blocked_from = 'ready_to_merge'")
        assert tool(w, "merge", "m1")[0] == 0
        [row] = [dict(r) for r in sql(w.db, "SELECT * FROM hook_events")]
        assert row["decision"] == "alert" and row["hook"] == "tool:merge" and row["session_id"] == ORCH
        assert json.loads(row["detail"]) == {
            "kind": "merge_state_conflict",
            "task_id": "T1",
            "key": "m1",
            "merge_sha": "m" * 40,
            "expected": "ready_to_merge",
            "actual": "blocked",
        }
        code, snap = run_cli("snapshot")
        assert code == 0 and snap["alerts"][0]["hook"] == "tool:merge"


# --------------------------------------------------------------------------- review fix #03 (H1, H6, H8)


class TestReviewFix03:
    # H1
    @pytest.mark.parametrize("name", ["spawn", "resume"])
    def test_a_fresh_launch_with_a_non_utf8_file_has_no_effect(self, w: W, name: str) -> None:
        t = TABLE[name]
        t.prep(w)
        nodes_before = [dict(r) for r in sql(w.db, "SELECT * FROM nodes")]
        args = t.args(w)
        [file_arg] = [k for k in args if k.endswith("_file")]
        Path(args[file_arg]).write_bytes(b"\xff\xfe\x00bad")
        code, out = tool(w, name, "u1", **args)
        assert code == 2 and out["error"] == "USAGE" and "UTF-8" in out["detail"], out
        assert call_row(w, name, "u1") is None and w.sim.effects("claude", "--bg") == 0
        assert [dict(r) for r in sql(w.db, "SELECT * FROM nodes")] == nodes_before  # no row, no launch_at

    @pytest.mark.parametrize("name", ["spawn", "resume"])
    def test_a_takeover_after_reserve_with_the_file_gone_writes_no_launch_at(self, w: W, name: str) -> None:
        t = TABLE[name]
        t.prep(w)
        args = t.args(w)
        [file_arg] = [k for k in args if k.endswith("_file")]
        text = Path(args[file_arg]).read_text()
        with faults.arm(f"{name}.after_effect"), pytest.raises(faults.FaultInjected):
            tool(w, name, "k1", **args)  # reserve done, the launch not yet
        Path(args[file_arg]).unlink()
        code, out = tool(w, name, "k1", **args)
        assert code == 2 and out["error"] == "USAGE", out
        assert node_row(w, "N1")["launch_at"] is None and w.sim.effects("claude", "--bg") == 0
        assert call_row(w, name, "k1")["state"] == "started"
        Path(args[file_arg]).write_text(text)
        code, out = tool(w, name, "k1", **args)
        assert code == 0, out
        assert w.sim.effects("claude", "--bg") == 1

    def test_a_fresh_insert_after_the_holder_released_needs_every_file(self, w: W) -> None:
        ctx = tools.Ctx(w.runner, lambda: db.connect(w.db), w.clock, w.probe, w.claude, w.main)
        sql(
            w.db,
            "INSERT INTO tool_calls (tool, key, args_hash, task_id, actor, args, state, holder_pid, holder_pid_start,"
            " outputs, started_at) VALUES ('t-gone', 'k1', 'x:-', 'T1', 'a', '{}', 'started', 7, 'start-7', '{}', 'x')",
        )

        def steps(task: Any, args: Any) -> tuple[tools.Step, ...]:
            sql(w.db, "DELETE FROM tool_calls WHERE tool = 't-gone'")  # the in-flight holder released its claim
            return ()

        tools.register(tools.ToolSpec("t-gone", steps))
        args = {"body_file": str(w.tmp / "missing.md")}
        with pytest.raises(tools.errors.CpError) as exc:
            tools.invoke("t-gone", key="k1", task_id="T1", args=args, token=w.token, actor="me", ctx=ctx)
        assert exc.value.code == "USAGE" and call_row(w, "t-gone", "k1") is None

    # H6
    def test_a_late_first_launch_listed_at_the_retake_is_adopted(self, w: W) -> None:
        import re

        w.sim.list_on_launch = False
        assert spawn(w, "s1")[0] == 6
        [first] = w.runner.matching("claude", "--bg")
        old_token = re.search(r"token (node:\S+)", first.argv[-1])[1]  # type: ignore[index]
        w.clock.advance(61)
        reap(w)
        assert node_row(w, "N1")["state"] == "reaped"
        w.claude.listing = [agent("S-first", "r1-T1-g1", id="first")]  # the first launch came up late
        code, out = spawn(w, "s1")
        assert code == 0, out
        assert w.sim.effects("claude", "--bg") == 1  # adopted: no relaunch
        assert out["token"] == old_token and out["result"]["name"] == "r1-T1-g1"
        node = node_row(w, "N1")
        assert (node["state"], node["session_id"], node["short_id"]) == ("running", "S-first", "first")
        assert w.runner.matching("claude", "stop") == []

    def test_a_late_first_launch_seen_after_the_relaunch_is_stopped(self, w: W) -> None:
        w.sim.list_on_launch = False
        assert spawn(w, "s1")[0] == 6
        for _ in range(2):  # two relaunches: r1-T1-g1-r1, then r1-T1-g1-r2
            w.clock.advance(61)
            reap(w)
            assert spawn(w, "s1")[0] == 6
        from control_plane import clock as clock_mod

        sql(w.db, "UPDATE nodes SET launch_at = ?", [clock_mod.stamp(w.clock)])
        wt = str(w.sim.wt)
        w.claude.listing = [
            agent("S-elsewhere", "r1-T1-g1", id="elsewhere", cwd="/another/checkout"),  # J2: not ours
            agent("S-nocwd", "r1-T1-g1", id="nocwd"),  # J2: no cwd, not proven ours
            agent("S-first", "r1-T1-g1", id="first", cwd=wt),
            agent("S-mid", "r1-T1-g1-r1", cwd=wt),  # no short id: stopped by its session id
            agent("S-new", "r1-T1-g1-r2", id="new", cwd=wt),
            agent("S-other", "r1-T2-g1", id="other", cwd=wt),
        ]
        code, out = spawn(w, "s1")
        assert code == 0 and out["result"]["session_id"] == "S-new"
        stops = [c.argv[-1] for c in w.runner.matching("claude", "stop")]
        assert stops == ["first", "S-mid"]

    # J3
    def test_an_adopted_late_relaunch_stops_the_earlier_launch(self, w: W) -> None:
        w.sim.list_on_launch = False
        assert spawn(w, "s1")[0] == 6
        for _ in range(2):  # relaunched as r1-T1-g1-r1, then reaped again
            w.clock.advance(61)
            reap(w)
            if node_row(w, "N1")["name"] == "r1-T1-g1":
                assert spawn(w, "s1")[0] == 6
        assert (node_row(w, "N1")["state"], node_row(w, "N1")["name"]) == ("reaped", "r1-T1-g1-r1")
        wt = str(w.sim.wt)
        w.claude.listing = [
            agent("S-first", "r1-T1-g1", id="first", cwd=wt),
            agent("S-alien", "r1-T1-g1", id="alien", cwd="/another/checkout"),
            agent("S-mid", "r1-T1-g1-r1", id="mid", cwd=wt),  # the relaunch came up late
        ]
        code, out = spawn(w, "s1")
        assert code == 0 and out["result"]["session_id"] == "S-mid"
        assert w.sim.effects("claude", "--bg") == 2  # adopted: no third launch
        assert [c.argv[-1] for c in w.runner.matching("claude", "stop")] == ["first"]

    def test_a_failed_stop_of_a_late_first_launch_is_ignored(self, w: W) -> None:
        from control_plane.runner import RunResult

        w.sim.list_on_launch = False
        assert spawn(w, "s1")[0] == 6
        w.clock.advance(61)
        reap(w)
        w.runner.on(["claude", "stop"], RunResult(1, "", "no such session"))
        w.sim.list_on_launch = True
        w.claude.listing = []
        real = w.sim._claude_bg

        def late_first(call: Any) -> Any:
            res = real(call)
            w.claude.listing = [*(w.claude.listing or []), agent("S-first", "r1-T1-g1", id="first", cwd=str(w.sim.wt))]
            return res

        w.runner.on(["claude", "--bg"], late_first)
        code, out = spawn(w, "s1")
        assert code == 0 and out["result"]["name"] == "r1-T1-g1-r1"
        assert len(w.runner.matching("claude", "stop")) == 1

    # H8
    def test_the_merge_sha_is_read_again_after_a_pause(self, w: W) -> None:
        from control_plane.runner import RunResult

        _prep_merge(w)
        answers = iter([json.dumps({"mergeCommit": None}), json.dumps({"mergeCommit": {"oid": "n" * 40}})])
        w.runner.on(["--json", "mergeCommit"], lambda c: RunResult(0, next(answers), ""))
        sleeps_before = list(w.clock.sleeps)
        assert tool(w, "merge", "m1")[0] == 0
        assert w.clock.sleeps[len(sleeps_before) :] == [tools.MERGE_SHA_RETRY_S]


# --- QS-400: the deliverable-only tools refuse a work item --------------------------------------------------------

DELIVERABLE_ONLY = ("worktree-create", "issue-create", "pr-create", "push", "merge")


@pytest.mark.parametrize("name", DELIVERABLE_ONLY)
def test_a_deliverable_only_tool_refuses_a_work_item(w: W, name: str) -> None:
    """A work item lands in its deliverable's PR: no issue, worktree QS_<M>, PR, push or merge of its own.

    Refused when the steps are built — a bare ``INVALID_STATE`` before the claim: no ``tool_calls`` row, no probe
    (``issue-create``'s and ``pr-create``'s probes call ``gh``), no effect."""
    t = TABLE[name]
    t.prep(w)
    insert_task(w.db, "T0", w.run, issue_number=10, branch="QS_10", is_deliverable=1)
    sql(w.db, "UPDATE tasks SET deliverable_id = 'T0', item_k = 1, is_deliverable = 0 WHERE id = 'T1'")
    w.runner.calls.clear()
    code, out = tool(w, name, "k1", **t.args(w))
    assert code == 8 and out["error"] == "INVALID_STATE", out
    assert "own issue" in out["detail"]
    assert call_row(w, name, "k1") is None
    assert t.effect(w) == 0
    assert w.runner.calls == []  # no probe ran (`issue-create` / `pr-create` probes call gh)
