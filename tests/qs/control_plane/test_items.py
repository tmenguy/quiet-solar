"""QS-400 §9: the item tools — ``item-create``, ``item-cleanup`` and ``integrate-start|finish|drop`` (AC 12, 13).

Every script is simulated by ``ItemSim`` through #399's ``FakeRunner``: a
``QS_N`` history (the commits that are ancestors of ``QS_N``), the item tips,
and one integration scratch per item. The rules answer exactly what the real
scripts print, so the tools are exercised end to end through ``cp.py tool``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from control_plane import cli, faults, items, locks, tools
from control_plane.runner import RunResult

from .conftest import ORCH, Call, FakeClaude, FakeRunner, agent, insert_node, insert_task, open_run, run_cli, sql

INTEGRATE_TOOLS = ("integrate-start", "integrate-finish", "integrate-drop")
ALL_TOOLS = ("item-create", "item-cleanup", *INTEGRATE_TOOLS)


def _opt(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _ok(payload: dict[str, Any]) -> RunResult:
    return RunResult(0, json.dumps(payload), "")


def _err(code: str, **extra: Any) -> RunResult:
    return RunResult(1, json.dumps({"error": code, "detail": code, **extra}), "")


@dataclass
class Scratch:
    base: str
    item_tip: str
    head: str
    phase: str


@dataclass
class ItemSim:
    """``setup_task.py --item``, ``cleanup_worktree.py --item``, ``integrate_item.py`` and ``git rev-parse``."""

    runner: FakeRunner
    main: Path
    tip: str | None = "d0"  # refs/heads/QS_7; None: the branch is deleted
    history: list[str] = field(default_factory=lambda: ["d0"])
    item_tips: dict[int, str] = field(default_factory=lambda: {1: "i1", 2: "i2"})
    scratches: dict[int, Scratch] = field(default_factory=dict)
    conflicts: bool = False
    gate_results: list[str] = field(default_factory=list)
    gates: int = 0
    merges: int = 0
    created: int = 0
    busy: set[str] = field(default_factory=set)
    cleanup_status: str = "removed"
    hooks: dict[str, Callable[[Call], None]] = field(default_factory=dict)

    def install(self) -> ItemSim:
        self.runner.on(["setup_task.py"], self._setup)
        self.runner.on(["cleanup_worktree.py"], self._cleanup)
        self.runner.on(["integrate_item.py"], self._integrate)
        self.runner.on(["git", "rev-parse", "--verify"], self._rev_parse)
        return self

    def wt(self, k: int = 1) -> Path:
        return self.main.parent / f"{self.main.name}-worktrees" / f"QS_7_{k}"

    def _hook(self, name: str, call: Call) -> None:
        if name in self.hooks:
            self.hooks[name](call)

    def _setup(self, call: Call) -> RunResult:
        self._hook("setup", call)
        n, k = int(call.argv[2]), int(_opt(call.argv, "--item"))
        wt = self.main.parent / f"{self.main.name}-worktrees" / f"QS_{n}_{k}"
        if not wt.exists():
            self.created += 1
        wt.mkdir(parents=True, exist_ok=True)
        payload = {"issue_number": n, "item": k, "branch": f"QS_{n}_{k}", "worktree_path": str(wt)}
        return _ok({**payload, "no_worktree": False, "harness": "claude-code"})

    def _cleanup(self, call: Call) -> RunResult:
        self._hook("cleanup", call)
        if self.cleanup_status == "garbage":
            return RunResult(0, "not json", "")
        return _ok({"status": self.cleanup_status, "branch": f"QS_7_{_opt(call.argv, '--item')}"})

    def _rev_parse(self, call: Call) -> RunResult:
        ref = call.argv[-1]
        k = int(ref.rsplit("_", 1)[1])
        if k not in self.item_tips:
            return RunResult(128, "", f"fatal: Needed a single revision {ref}")
        return RunResult(0, self.item_tips[k] + "\n", "")

    def _integrate(self, call: Call) -> RunResult:
        i = next(j for j, a in enumerate(call.argv) if a.endswith("integrate_item.py"))
        sub = call.argv[i + 1]
        self._hook(sub, call)
        if sub in self.busy:
            return _err("scratch-busy", max_wait_s=3300)
        k = int(_opt(call.argv, "--item"))
        handler: Callable[[list[str], int], RunResult] = getattr(self, f"_{sub}")
        return handler(call.argv, k)

    def _prepare(self, argv: list[str], k: int) -> RunResult:
        tip = _opt(argv, "--item-tip")
        if self.tip is None or k not in self.item_tips:
            return _err("missing-ref")
        s = self.scratches.get(k)
        base = {} if s is None else {"base": s.base}
        if tip in self.history:
            return _ok({"status": "already-integrated", "item_tip": tip, **base})
        if s is None:
            self.merges += 1
            head = self.tip if self.conflicts else f"m{self.merges}"
            s = Scratch(self.tip, tip, head, "conflicts" if self.conflicts else "merged")
            self.scratches[k] = s
        elif s.base != self.tip:
            return _err("stale-scratch")
        out = {"status": s.phase, "item_tip": tip, "base": s.base, "scratch": f"{self.wt(k)}_integration"}
        if s.phase == "conflicts":
            out["files"] = ["a.py"]
        return _ok(out)

    def _check(self, argv: list[str], k: int) -> RunResult:
        s = self.scratches.get(k)
        if s is None:
            return _err("no-scratch")
        ready = {"status": "ready", "head": s.head, "base": s.base, "item_tip": s.item_tip}
        if s.head != s.base and s.head in self.history:
            return _ok({**ready, "moved": True})
        if s.head == s.base:
            return _err("not-a-merge")
        if self.tip != s.base:
            return _err("deliverable-moved")
        return _ok({**ready, "moved": False})

    def _gate(self, argv: list[str], k: int) -> RunResult:
        s = self.scratches[k]
        if s.head != _opt(argv, "--expect-head"):
            return _err("head-moved")
        self.gates += 1
        status = self.gate_results.pop(0) if self.gate_results else "green"
        return _ok({"status": status, **({"tail": ["E   assert 1 == 2"]} if status == "red" else {})})

    def _move(self, argv: list[str], k: int) -> RunResult:
        new, old = _opt(argv, "--new"), _opt(argv, "--old")
        if self.scratches[k].head != new:
            return _err("head-moved")
        if new in self.history:
            return _ok({"status": "already-moved"})
        if self.tip != old:
            return _err("deliverable-moved")
        self.history.append(new)
        self.tip = new
        return _ok({"status": "moved"})

    def _drop(self, argv: list[str], k: int) -> RunResult:
        s = self.scratches.pop(k, None)
        if s is None:
            return _ok({"status": "nothing-to-drop"})
        out: dict[str, Any] = {"status": "dropped", "dropped_head": s.head, "discarded_files": []}
        out["discarded_snapshot"] = None
        if self.tip is None:
            out.update(integrated=None, deliverable_missing=True)
        else:
            out["integrated"] = s.item_tip in self.history
        return _ok(out)

    def commit_on_deliverable(self, sha: str) -> None:
        self.history.append(sha)
        self.tip = sha

    def count(self, *pattern: str) -> int:
        return len(self.runner.matching(*pattern))

    def calls(self, sub: str) -> list[Call]:
        return self.runner.matching("integrate_item.py", sub)


@dataclass
class W:
    db: Path
    run: str
    token: str
    node: str
    sim: ItemSim
    tmp: Path
    main: Path
    runner: FakeRunner
    claude: FakeClaude


@pytest.fixture
def w(migrated, tmp_path, fake_runner, fake_claude, fake_main, monkeypatch) -> W:
    monkeypatch.setattr(locks, "LOCK_WAIT_S", 5.0)
    monkeypatch.setattr(locks, "GATE_WAIT_S", 5.0)
    run_id, token = open_run()
    insert_task(migrated, "T1", run_id, issue_number=7, branch="QS_7", is_deliverable=1)
    insert_task(migrated, "T2", run_id, deliverable_id="T1", item_k=1)
    insert_task(migrated, "T3", run_id, deliverable_id="T1", item_k=2)
    node = insert_node(migrated, "N2", run_id, "T2", session_id="S-n2", name="n2")
    fake_claude.listing = [agent(ORCH, pid=101, kind="interactive"), agent("S-n2", "n2", pid=202)]
    sim = ItemSim(fake_runner, fake_main).install()
    return W(migrated, run_id, token, node, sim, tmp_path, fake_main, fake_runner, fake_claude)


@pytest.fixture
def wi(w: W) -> W:
    """Both items created (``branch`` set) and the item's node holding ``integration:QS_7``."""
    sql(w.db, "UPDATE tasks SET branch = 'QS_7_' || item_k WHERE deliverable_id = 'T1'")
    acquire(w)
    w.runner.calls.clear()
    return w


def tool(w: W, name: str, key: str, token: str | None = None, task: str = "T2", **args: Any) -> tuple[int, Any]:
    f = w.tmp / f"args-{name}-{key}.json".replace(":", "_").replace("/", "_")
    f.write_text(json.dumps(args))
    return run_cli("tool", name, "--task", task, "--key", key, "--args-file", str(f), "--token", token or w.token)


def acquire(w: W, token: str | None = None, session: str = "S-n2", name: str = "integration:QS_7") -> None:
    argv = ["lock", "acquire", "--name", name, "--purpose", "integrate item 1", "--session-id", session]
    code, out = run_cli(*argv, "--token", token or w.node)
    assert code == 0, out


def call_row(w: W, name: str, key: str) -> dict | None:
    rows = sql(w.db, "SELECT * FROM tool_calls WHERE tool = ? AND key = ?", [name, key])
    return dict(rows[0]) if rows else None


def outputs(w: W, name: str, key: str) -> dict[str, Any]:
    row = call_row(w, name, key)
    assert row is not None
    return dict(json.loads(row["outputs"]))


def rows(w: W) -> list[tuple[Any, ...]]:
    return [
        tuple(r)
        for r in sql(
            w.db, "SELECT item_task_id, deliverable_id, item_tip, merge_commit, result, tool_call_key FROM integrations"
        )
    ]


def lock_rows(w: W) -> list[tuple[Any, ...]]:
    return [tuple(r) for r in sql(w.db, "SELECT name, holder_kind FROM locks ORDER BY name")]


def task_col(w: W, task: str, col: str) -> Any:
    return sql(w.db, f"SELECT {col} FROM tasks WHERE id = ?", [task])[0][0]


def snapshot_locks(w: W, hook: str, seen: dict[str, Any]) -> None:
    def take(call: Call) -> None:
        seen.setdefault(hook, []).append(lock_rows(w))
        seen.setdefault(f"{hook}:slots", []).append(sql(w.db, "SELECT count(*) FROM cap_slots")[0][0])

    w.sim.hooks[hook] = take


# --------------------------------------------------------------------------- shape, before the claim


@pytest.mark.parametrize("name", ALL_TOOLS)
def test_a_non_item_task_is_a_bare_invalid_state(w: W, name: str) -> None:
    code, out = tool(w, name, "k1", task="T1")
    assert code == 8 and out["error"] == "INVALID_STATE", out
    assert call_row(w, name, "k1") is None and w.runner.calls == []


@pytest.mark.parametrize("branch", [None, "QS_7", "QS_7_01", "QS_7_0", "QS_7_1\n", "feature"])
@pytest.mark.parametrize("name", INTEGRATE_TOOLS)
def test_integrate_without_an_item_branch_is_a_bare_invalid_state(w: W, name: str, branch: str | None) -> None:
    acquire(w)
    sql(w.db, "UPDATE tasks SET branch = ? WHERE id = 'T2'", [branch])
    w.runner.calls.clear()
    code, out = tool(w, name, "k1", token=w.node)
    assert code == 8 and out["error"] == "INVALID_STATE", out
    assert call_row(w, name, "k1") is None and w.runner.calls == []


def test_a_non_boolean_cleanup_flag_is_usage(w: W) -> None:
    for flag in ("delete_branch", "discard_unintegrated", "force"):
        code, out = tool(w, "item-cleanup", f"k-{flag}", **{flag: "yes"})
        assert code == 2 and out["error"] == "USAGE" and flag in out["detail"]
        assert call_row(w, "item-cleanup", f"k-{flag}") is None
    assert w.runner.calls == []


# --------------------------------------------------------------------------- item-create


class TestItemCreate:
    def test_runs_setup_from_main_under_main_checkout(self, w: W) -> None:
        seen: dict[str, Any] = {}
        snapshot_locks(w, "setup", seen)
        code, out = tool(w, "item-create", "item:T2:create:1")
        assert code == 0, out
        wt = str(w.sim.wt(1).resolve())
        assert out["result"] == {"worktree": wt, "branch": "QS_7_1"}
        (call,) = w.runner.matching("setup_task.py")
        assert call.argv == [
            str(w.main / "venv" / "bin" / "python"),
            str(w.main / "scripts" / "qs" / "setup_task.py"),
            "7",
            "--item",
            "1",
            "--harness",
            "claude-code",
        ]
        assert call.cwd == str(w.main) and call.timeout == 600
        assert "QS_CP_TOKEN" in call.env_remove and "QS_CP_TOKEN" not in call.env_extra  # no inject_token (D7)
        assert seen["setup"] == [[("main-checkout", "process")]]
        assert (task_col(w, "T2", "worktree"), task_col(w, "T2", "branch")) == (wt, "QS_7_1")
        assert lock_rows(w) == []

    def test_same_key_replay_has_no_second_effect(self, w: W) -> None:
        code, first = tool(w, "item-create", "c1")
        assert code == 0
        code, again = tool(w, "item-create", "c1")
        assert code == 0 and again["replayed"] is True and again["result"] == first["result"]
        assert w.sim.count("setup_task.py") == 1 and w.sim.created == 1

    def test_the_second_item_takes_its_own_k(self, w: W) -> None:
        code, out = tool(w, "item-create", "c1", task="T3")
        assert code == 0 and out["result"]["branch"] == "QS_7_2"
        assert w.runner.matching("setup_task.py")[0].argv[2:5] == ["7", "--item", "2"]

    @pytest.mark.parametrize(("task", "state"), [("T2", "dropped"), ("T2", "merged"), ("T1", "merged")])
    def test_a_terminal_item_or_deliverable_is_refused_by_the_guard(self, w: W, task: str, state: str) -> None:
        sql(w.db, "UPDATE tasks SET state = ? WHERE id = ?", [state, task])
        code, out = tool(w, "item-create", "c1")
        assert code == 1 and out["error"] == "TOOL_FAILED" and out["result"]["error"] == "INVALID_STATE", out
        assert call_row(w, "item-create", "c1")["state"] == "failed"
        assert w.runner.calls == []

    @pytest.mark.parametrize(("issue", "branch"), [(None, "QS_7"), (7, "QS_8"), (7, None)])
    def test_the_deliverable_must_be_qs_issue(self, w: W, issue: int | None, branch: str | None) -> None:
        sql(w.db, "UPDATE tasks SET issue_number = ?, branch = ? WHERE id = 'T1'", [issue, branch])
        code, out = tool(w, "item-create", "c1")
        assert code == 1 and out["result"]["error"] == "INVALID_STATE", out
        assert w.runner.calls == []

    def test_a_failed_setup_spends_the_key(self, w: W) -> None:
        w.runner.on(["setup_task.py"], RunResult(1, json.dumps({"error": "deliverable branch QS_7 not found"}), ""))
        code, out = tool(w, "item-create", "c1")
        assert code == 1 and out["result"]["output"]["json"]["error"] == "deliverable branch QS_7 not found"
        assert task_col(w, "T2", "branch") is None

    def test_a_node_token_is_refused(self, w: W) -> None:
        code, out = tool(w, "item-create", "c1", token=w.node)
        assert code == 8 and out["error"] == "CONFLICT"


# --------------------------------------------------------------------------- item-cleanup


class TestItemCleanup:
    @pytest.fixture
    def wc(self, w: W) -> W:
        assert tool(w, "item-create", "c1")[0] == 0
        w.sim.scratches[1] = Scratch("d0", "i1", "m9", "merged")
        w.runner.calls.clear()
        return w

    def test_drops_the_scratch_then_cleans_under_both_locks(self, wc: W) -> None:
        w = wc
        seen: dict[str, Any] = {}
        snapshot_locks(w, "drop", seen)
        snapshot_locks(w, "cleanup", seen)
        code, out = tool(w, "item-cleanup", "item:T2:cleanup:1", delete_branch=True)
        assert code == 0, out
        drop, cleanup = w.runner.calls  # drop_scratch runs first
        assert drop.argv == [
            str(w.main / "venv" / "bin" / "python"),
            str(w.main / "scripts" / "qs" / "integrate_item.py"),
            "drop",
            "--issue",
            "7",
            "--item",
            "1",
        ]
        assert cleanup.argv == [
            str(w.main / "venv" / "bin" / "python"),
            str(w.main / "scripts" / "qs" / "cleanup_worktree.py"),
            "--issue",
            "7",
            "--item",
            "1",
            "--work-dir",
            str(w.sim.wt(1).resolve()),
            "--delete-branch",
        ]
        assert {c.cwd for c in w.runner.calls} == {str(w.main)} and {c.timeout for c in w.runner.calls} == {600}
        both = [("integration:QS_7", "process"), ("main-checkout", "process")]
        assert seen["drop"] == [both] and seen["cleanup"] == [both]
        assert out["result"] == {
            "status": "removed",
            "branch": "QS_7_1",
            "scratch": {
                "outcome": "dropped",
                "dropped_head": "m9",
                "integrated": False,
                "discarded_snapshot": None,
                "discarded_files": [],
            },
        }
        assert task_col(w, "T2", "worktree") is None and task_col(w, "T2", "branch") == "QS_7_1"
        assert lock_rows(w) == []

    def test_every_flag_is_passed(self, wc: W) -> None:
        code, out = tool(wc, "item-cleanup", "k1", delete_branch=True, discard_unintegrated=True, force=True)
        assert code == 0, out
        assert out["result"]["scratch"]["outcome"] == "dropped"
        argv = wc.runner.matching("cleanup_worktree.py")[0].argv
        assert argv[-3:] == ["--delete-branch", "--discard-unintegrated", "--force"]

    def test_no_flags_and_nothing_to_drop(self, wc: W) -> None:
        wc.sim.scratches.clear()
        wc.sim.cleanup_status = "removed-branch-kept"
        code, out = tool(wc, "item-cleanup", "k1")
        assert code == 0, out
        assert out["result"]["status"] == "removed-branch-kept" and out["result"]["scratch"] == {
            "outcome": "nothing-to-drop"
        }
        assert wc.runner.matching("cleanup_worktree.py")[0].argv[-2:] == ["--work-dir", str(wc.sim.wt(1).resolve())]

    def test_the_work_dir_fallback_when_worktree_is_null(self, w: W) -> None:
        sql(w.db, "UPDATE tasks SET branch = 'QS_7_1' WHERE id = 'T2'")
        assert tool(w, "item-cleanup", "k1")[0] == 0
        argv = w.runner.matching("cleanup_worktree.py")[0].argv
        assert argv[argv.index("--work-dir") + 1] == str(w.main.parent / "main-worktrees" / "QS_7_1")

    @pytest.mark.parametrize(("task", "state"), [("T1", "merged"), ("T2", "dropped")])
    def test_runs_on_a_merged_deliverable_or_a_dropped_item(self, wc: W, task: str, state: str) -> None:
        sql(wc.db, "UPDATE tasks SET state = ? WHERE id = ?", [state, task])
        code, out = tool(wc, "item-cleanup", "k1", delete_branch=True)
        assert code == 0, out
        assert wc.sim.scratches == {} and task_col(wc, "T2", "worktree") is None

    def test_a_live_session_lock_of_another_session_is_busy_before_any_effect(self, wc: W) -> None:
        acquire(wc)  # the item's node integrates
        wc.runner.calls.clear()
        code, out = tool(wc, "item-cleanup", "k1")
        assert code == 6 and out["error"] == "BUSY", out
        assert call_row(wc, "item-cleanup", "k1") is None and wc.runner.calls == []
        assert lock_rows(wc) == [("integration:QS_7", "session")]

    def test_the_callers_own_session_lock_is_co_held(self, wc: W) -> None:
        acquire(wc, token=wc.token, session=ORCH)
        seen: dict[str, Any] = {}
        snapshot_locks(wc, "drop", seen)
        code, out = tool(wc, "item-cleanup", "k1")
        assert code == 0, out
        assert seen["drop"] == [[("integration:QS_7", "session"), ("main-checkout", "process")]]
        assert lock_rows(wc) == [("integration:QS_7", "session")]  # a tool never releases a session lock

    def test_a_null_branch_takes_main_checkout_only(self, w: W) -> None:
        seen: dict[str, Any] = {}
        snapshot_locks(w, "cleanup", seen)
        acquire(w)  # held by another session: not taken, not waited on
        code, out = tool(w, "item-cleanup", "k1")
        assert code == 0, out
        assert seen["cleanup"] == [[("integration:QS_7", "session"), ("main-checkout", "process")]]

    @pytest.mark.parametrize("branch", ["QS_8_1", "QS_7_2", "feature"])
    def test_a_mismatched_branch_is_the_guards_invalid_state(self, w: W, branch: str) -> None:
        sql(w.db, "UPDATE tasks SET branch = ? WHERE id = 'T2'", [branch])
        code, out = tool(w, "item-cleanup", "k1")
        assert code == 1 and out["result"]["error"] == "INVALID_STATE", out
        assert w.runner.calls == []

    def test_a_session_lock_sorting_after_is_a_conflict_with_no_effect(self, w: W) -> None:
        insert_task(w.db, "T5", w.run, issue_number=10, branch="QS_10", is_deliverable=1)
        insert_task(w.db, "T6", w.run, deliverable_id="T5", item_k=1, branch="QS_10_1")
        sql(
            w.db,
            "INSERT INTO locks (name, holder_kind, holder_session_id, holder_actor, token_subject, acquired_at)"
            " VALUES ('integration:QS_9', 'session', ?, 'orchestrator', 'run:R1', 'x')",
            [ORCH],
        )
        code, out = tool(w, "item-cleanup", "k1", task="T6")
        assert code == 1 and out["error"] == "TOOL_FAILED" and out["result"]["error"] == "CONFLICT", out
        assert w.runner.calls == []

    def test_a_failed_drop_stops_before_cleanup(self, wc: W) -> None:
        wc.sim.busy.add("drop")
        code, out = tool(wc, "item-cleanup", "k1")
        assert code == 1 and out["error"] == "TOOL_FAILED"
        assert out["result"]["output"]["json"]["error"] == "scratch-busy"
        assert wc.sim.count("cleanup_worktree.py") == 0 and task_col(wc, "T2", "worktree") is not None

    @pytest.mark.parametrize("status", ["action_required", "error", "garbage"])
    def test_a_cleanup_that_did_not_remove_fails_with_its_json(self, wc: W, status: str) -> None:
        wc.sim.cleanup_status = status
        code, out = tool(wc, "item-cleanup", "k1")
        assert code == 1 and out["error"] == "TOOL_FAILED", out
        expected = None if status == "garbage" else {"status": status, "branch": "QS_7_1"}
        assert out["result"]["output"]["json"] == expected
        assert task_col(wc, "T2", "worktree") is not None

    def test_same_key_replay_has_no_second_effect(self, wc: W) -> None:
        assert tool(wc, "item-cleanup", "k1")[0] == 0
        code, out = tool(wc, "item-cleanup", "k1")
        assert code == 0 and out["replayed"] is True
        assert wc.sim.count("integrate_item.py") == 1 and wc.sim.count("cleanup_worktree.py") == 1

    def test_an_unexpected_drop_status_fails(self, wc: W) -> None:
        wc.runner.on(["integrate_item.py", "drop"], _ok({"status": "weird"}))
        code, out = tool(wc, "item-cleanup", "k1")
        assert code == 1 and "'weird'" in out["detail"], out


# --------------------------------------------------------------------------- integrate-*: the session check


def _insert_lock(w: W, kind: str, session: str | None, subject: str = "node:N2") -> None:
    sql(
        w.db,
        "INSERT INTO locks (name, holder_kind, holder_pid, holder_pid_start, holder_pgid, holder_session_id,"
        " holder_actor, token_subject, acquired_at) VALUES ('integration:QS_7', ?, 9, 'start-9', 9, ?, 'x', ?, 'x')",
        [kind, session, subject],
    )


LOCK_STATES: dict[str, Callable[[W], None]] = {
    "free": lambda w: None,
    "another-session": lambda w: acquire(w, token=w.token, session=ORCH),
    "process-held": lambda w: _insert_lock(w, "process", None),
    "same-subject-other-session": lambda w: _insert_lock(w, "session", "S-other"),
}


@pytest.mark.parametrize("state", sorted(LOCK_STATES))
@pytest.mark.parametrize("name", INTEGRATE_TOOLS)
def test_integrate_without_the_callers_session_lock_is_policy_refused(w: W, name: str, state: str) -> None:
    sql(w.db, "UPDATE tasks SET branch = 'QS_7_1' WHERE id = 'T2'")
    LOCK_STATES[state](w)
    w.runner.calls.clear()
    code, out = tool(w, name, "k1", token=w.node)
    assert code == 9 and out["error"] == "POLICY_REFUSED", out
    assert "cp.py lock acquire --name integration:QS_7" in out["detail"]
    assert call_row(w, name, "k1") is None and w.runner.calls == []


class TestIntegrateGuards:
    @pytest.mark.parametrize("name", INTEGRATE_TOOLS)
    def test_a_mismatched_item_branch(self, wi: W, name: str) -> None:
        sql(wi.db, "UPDATE tasks SET branch = 'QS_7_2' WHERE id = 'T2'")
        code, out = tool(wi, name, "k1", token=wi.node)
        assert code == 1 and out["result"]["error"] == "INVALID_STATE", out
        assert wi.runner.calls == []

    @pytest.mark.parametrize("name", ["integrate-start", "integrate-finish"])
    def test_live_guard_refuses_a_terminal_deliverable_or_item(self, wi: W, name: str) -> None:
        for i, (task, state) in enumerate((("T1", "merged"), ("T2", "dropped"))):
            sql(wi.db, "UPDATE tasks SET state = 'building'")
            sql(wi.db, "UPDATE tasks SET state = ? WHERE id = ?", [state, task])
            code, out = tool(wi, name, f"k{i}", token=wi.node)
            assert code == 1 and out["result"]["error"] == "INVALID_STATE", out
        assert wi.runner.calls == []

    def test_drop_runs_on_a_terminal_deliverable(self, wi: W) -> None:
        wi.sim.scratches[1] = Scratch("d0", "i1", "m9", "merged")
        sql(wi.db, "UPDATE tasks SET state = 'merged' WHERE id = 'T1'")
        code, out = tool(wi, "integrate-drop", "k1", token=wi.node)
        assert code == 0 and out["result"]["outcome"] == "dropped", out

    def test_a_stale_token_between_steps_runs_no_further_step(self, wi: W) -> None:
        wi.sim.scratches[1] = Scratch("d0", "i1", "m1", "merged")
        wi.sim.hooks["check"] = lambda call: sql(wi.db, "UPDATE nodes SET nonce = ? WHERE id = 'N2'", ["f" * 32])
        code, out = tool(wi, "integrate-finish", "k1", token=wi.node)
        assert code == 3 and out["error"] == "STALE_TOKEN", out
        assert wi.sim.gates == 0 and wi.sim.calls("move") == [] and rows(wi) == []

    def test_a_run_token_that_holds_the_lock_is_allowed(self, w: W) -> None:
        sql(w.db, "UPDATE tasks SET branch = 'QS_7_1' WHERE id = 'T2'")
        acquire(w, token=w.token, session=ORCH)
        code, out = tool(w, "integrate-start", "k1")
        assert code == 0 and out["result"]["outcome"] == "merged", out
        assert lock_rows(w) == [("integration:QS_7", "session")]


# --------------------------------------------------------------------------- integrate-start


class TestIntegrateStart:
    def test_prepare_with_the_item_tip_merged_writes_no_row(self, wi: W) -> None:
        seen: dict[str, Any] = {}
        snapshot_locks(wi, "prepare", seen)
        code, out = tool(wi, "integrate-start", "item:T2:start:i1.1", token=wi.node)
        assert code == 0, out
        (rev,) = wi.runner.matching("git", "rev-parse", "--verify")
        assert rev.argv == ["git", "rev-parse", "--verify", "refs/heads/QS_7_1"] and rev.cwd == str(wi.main)
        (prep,) = wi.sim.calls("prepare")
        assert prep.argv[1:] == [
            str(wi.main / "scripts" / "qs" / "integrate_item.py"),
            "prepare",
            "--issue",
            "7",
            "--item",
            "1",
            "--item-tip",
            "i1",
        ]
        assert prep.argv[0] == str(wi.main / "venv" / "bin" / "python") and prep.timeout == 600
        assert out["result"] == {
            "outcome": "merged",
            "item_tip": "i1",
            "base": "d0",
            "scratch": f"{wi.sim.wt(1)}_integration",
        }
        assert seen["prepare"] == [[("integration:QS_7", "session"), ("main-checkout", "process")]]
        assert rows(wi) == []
        cohold = sql(wi.db, "SELECT cohold_pid FROM locks WHERE name = 'integration:QS_7'")[0][0]
        assert cohold is None and lock_rows(wi) == [("integration:QS_7", "session")]

    def test_one_conflict_row_per_answered_start(self, wi: W) -> None:
        wi.sim.conflicts = True
        for n in (1, 2):
            code, out = tool(wi, "integrate-start", f"s.{n}", token=wi.node)
            assert code == 0 and out["result"]["outcome"] == "conflicts" and out["result"]["files"] == ["a.py"]
        assert rows(wi) == [
            ("T2", "T1", "i1", None, "conflict", "integrate-start/s.1"),
            ("T2", "T1", "i1", None, "conflict", "integrate-start/s.2"),
        ]

    def test_already_integrated_is_a_noop_row(self, wi: W) -> None:
        wi.sim.history.append("i1")
        code, out = tool(wi, "integrate-start", "s.1", token=wi.node)
        assert code == 0 and out["result"] == {"outcome": "already-integrated", "item_tip": "i1"}, out
        assert rows(wi) == [("T2", "T1", "i1", None, "noop", "integrate-start/s.1")]

    def test_a_missing_item_branch_fails_the_step(self, wi: W) -> None:
        del wi.sim.item_tips[1]
        code, out = tool(wi, "integrate-start", "s.1", token=wi.node)
        assert code == 1 and out["error"] == "TOOL_FAILED" and out["result"]["output"]["exit_code"] == 128
        assert wi.sim.calls("prepare") == []

    def test_a_script_error_fails_the_step(self, wi: W) -> None:
        wi.sim.scratches[1] = Scratch("old", "i1", "m9", "merged")
        code, out = tool(wi, "integrate-start", "s.1", token=wi.node)
        assert code == 1 and out["result"]["output"]["json"]["error"] == "stale-scratch"

    def test_start_drop_start_with_new_keys_makes_a_new_scratch(self, wi: W) -> None:
        assert tool(wi, "integrate-start", "s.1", token=wi.node)[1]["result"]["outcome"] == "merged"
        code, out = tool(wi, "integrate-drop", "d.1", token=wi.node)
        assert code == 0 and out["result"] == {
            "outcome": "dropped",
            "dropped_head": "m1",
            "integrated": False,
            "discarded_snapshot": None,
            "discarded_files": [],
        }
        code, out = tool(wi, "integrate-start", "s.2", token=wi.node)
        assert code == 0 and out["result"]["outcome"] == "merged"
        assert wi.sim.scratches[1].head == "m2" and wi.sim.merges == 2

    def test_drop_with_nothing_to_drop(self, wi: W) -> None:
        code, out = tool(wi, "integrate-drop", "d.1", token=wi.node)
        assert code == 0 and out["result"] == {"outcome": "nothing-to-drop"}


# --------------------------------------------------------------------------- integrate-finish


class TestIntegrateFinish:
    @pytest.fixture
    def wf(self, wi: W) -> W:
        assert tool(wi, "integrate-start", "s.1", token=wi.node)[0] == 0
        wi.runner.calls.clear()
        return wi

    def test_green_takes_a_gate_slot_moves_and_writes_one_ok_row(self, wf: W) -> None:
        seen: dict[str, Any] = {}
        snapshot_locks(wf, "gate", seen)
        code, out = tool(wf, "integrate-finish", "item:T2:finish:m1.1", token=wf.node)
        assert code == 0, out
        assert out["result"] == {"outcome": "ok", "merge_commit": "m1", "item_tip": "i1"}
        assert [c.argv[2] for c in wf.runner.calls] == ["check", "gate", "move"]
        (gate,) = wf.sim.calls("gate")
        assert gate.argv[-2:] == ["--expect-head", "m1"] and gate.timeout == 3600
        assert wf.sim.calls("move")[0].argv[-4:] == ["--new", "m1", "--old", "d0"]
        assert seen["gate"] == [[("integration:QS_7", "session")]] and seen["gate:slots"] == [1]
        assert rows(wf) == [("T2", "T1", "i1", "m1", "ok", "integrate-finish/item:T2:finish:m1.1")]
        assert wf.sim.tip == "m1"
        assert lock_rows(wf) == [("integration:QS_7", "session")]  # the session still holds it
        assert sql(wf.db, "SELECT count(*) FROM cap_slots")[0][0] == 0

    def test_red_skips_the_move_and_keeps_the_scratch(self, wf: W) -> None:
        wf.sim.gate_results = ["red"]
        code, out = tool(wf, "integrate-finish", "f.1", token=wf.node)
        assert code == 0, out
        assert out["result"] == {"outcome": "gate_red", "item_tip": "i1", "tail": ["E   assert 1 == 2"]}
        assert outputs(wf, "integrate-finish", "f.1")["move"] == {"status": "skipped", "reason": "red"}
        assert wf.sim.calls("move") == [] and 1 in wf.sim.scratches and wf.sim.tip == "d0"
        assert rows(wf) == [("T2", "T1", "i1", None, "gate_red", "integrate-finish/f.1")]
        code, out = tool(wf, "integrate-finish", "f.2", token=wf.node)  # a flaky gate retried with .2
        assert code == 0 and out["result"]["outcome"] == "ok" and wf.sim.gates == 2
        assert [r[4] for r in rows(wf)] == ["gate_red", "ok"]

    @pytest.mark.parametrize("skip", [1, 2, 3])
    def test_a_crash_replayed_with_the_same_key(self, wf: W, skip: int) -> None:
        with faults.arm("integrate-finish.after_effect", skip=skip), pytest.raises(faults.FaultInjected):
            tool(wf, "integrate-finish", "f.1", token=wf.node)
        code, out = tool(wf, "integrate-finish", "f.1", token=wf.node)
        assert code == 0 and out["result"]["outcome"] == "ok", out
        assert wf.sim.gates == (2 if skip == 1 else 1)  # after gate: its output was never persisted
        expected_move = "already-moved" if skip == 2 else "moved"
        assert outputs(wf, "integrate-finish", "f.1")["move"]["status"] == expected_move
        assert len(wf.sim.calls("move")) == (2 if skip == 2 else 1)
        assert [r[4] for r in rows(wf)] == ["ok"]

    def test_a_new_key_after_a_crash_after_move_and_a_commit_on_the_deliverable(self, wf: W) -> None:
        with faults.arm("integrate-finish.after_effect", skip=2), pytest.raises(faults.FaultInjected):
            tool(wf, "integrate-finish", "f.1", token=wf.node)
        wf.sim.commit_on_deliverable("x1")
        code, out = tool(wf, "integrate-finish", "f.2", token=wf.node)
        assert code == 0 and out["result"] == {"outcome": "ok", "merge_commit": "m1", "item_tip": "i1"}, out
        out2 = outputs(wf, "integrate-finish", "f.2")
        assert out2["check"]["moved"] is True and out2["gate"] == {"status": "skipped", "reason": "moved"}
        assert out2["move"]["status"] == "already-moved" and wf.sim.gates == 1
        assert [r[4] for r in rows(wf)] == ["ok"]

    def test_a_completed_finish_then_a_new_key_keeps_one_ok_row(self, wf: W) -> None:
        assert tool(wf, "integrate-finish", "f.1", token=wf.node)[0] == 0
        code, out = tool(wf, "integrate-finish", "f.2", token=wf.node)
        assert code == 0 and out["result"]["outcome"] == "ok"
        record = outputs(wf, "integrate-finish", "f.2")
        assert record["check"]["moved"] is True and record["record"]["deduplicated"] is True
        assert [r[3:5] for r in rows(wf)] == [("m1", "ok")]

    def test_a_check_error_fails_before_the_gate(self, wf: W) -> None:
        wf.sim.commit_on_deliverable("x1")  # deliverable-moved
        code, out = tool(wf, "integrate-finish", "f.1", token=wf.node)
        assert code == 1 and out["result"]["output"]["json"]["error"] == "deliverable-moved"
        assert wf.sim.gates == 0 and rows(wf) == []

    def test_a_full_gate_cap_is_busy_before_any_effect(self, wf: W, monkeypatch) -> None:
        monkeypatch.setenv("QS_CP_MAX_GATES", "1")
        sql(
            wf.db,
            "INSERT INTO cap_slots (cap, slot, holder_pid, holder_pid_start, holder_pgid, holder_actor, acquired_at)"
            " VALUES ('gates', 0, 77, 'start-77', 77, 'other', 'x')",
        )
        code, out = tool(wf, "integrate-finish", "f.1", token=wf.node)
        assert code == 6 and out["error"] == "BUSY", out
        assert call_row(wf, "integrate-finish", "f.1") is None and wf.runner.calls == []


# --------------------------------------------------------------------------- cleanup after the deliverable merged


@pytest.mark.parametrize("deleted", [False, True])
def test_item_cleanup_after_the_deliverable_merged_drops_each_scratch(wi: W, deleted: bool) -> None:
    w = wi
    w.sim.scratches[1] = Scratch("d0", "i1", "m1", "merged")
    w.sim.scratches[2] = Scratch("d0", "i2", "m2", "merged")
    w.sim.history += ["i1", "m1"]  # item 1 was integrated, item 2 never was
    if deleted:
        w.sim.tip = None
    sql(w.db, "UPDATE tasks SET state = 'merged' WHERE id = 'T1'")
    seen: dict[str, Any] = {}
    snapshot_locks(w, "drop", seen)
    results = []
    for task in ("T2", "T3"):
        code, out = tool(w, "item-cleanup", f"item:{task}:cleanup:1", task=task, delete_branch=True)
        assert code == 0, out
        results.append(out["result"]["scratch"])
    assert [c.argv[c.argv.index("--item") + 1] for c in w.sim.calls("drop")] == ["1", "2"]
    assert w.sim.scratches == {}
    process = [("integration:QS_7", "process"), ("main-checkout", "process")]
    assert seen["drop"] == [process, process]  # the dead session lock was taken as a process lock
    if deleted:
        assert [(r["integrated"], r["deliverable_missing"]) for r in results] == [(None, True), (None, True)]
    else:
        assert [r["integrated"] for r in results] == [True, False]
    assert lock_rows(w) == []


# --------------------------------------------------------------------------- registration


def test_register_item_tools_is_idempotent_and_the_tools_are_in_the_cli(capsys) -> None:
    items.register_item_tools()
    items.register_item_tools()
    assert set(items.NAMES) <= set(tools.REGISTRY)
    tools.reset()
    assert not set(items.NAMES) & set(tools.REGISTRY)
    with pytest.raises(SystemExit) as exc:
        cli.main(["tool", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert all(name in text for name in ALL_TOOLS)
    assert set(items.NAMES) <= set(tools.REGISTRY)  # main() registered them again
