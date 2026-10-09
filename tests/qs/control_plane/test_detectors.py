"""QS-406 T7: the detectors (§6, AC 8, and AC 3's overlap restart)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from control_plane import activeloop, clock, detectors, errors, tasks, ticks
from control_plane.runner import RunResult

from .conftest import Call, FakeRunner, insert_node, insert_task, open_run, sql


@dataclass
class FakeGit:
    """Scripted git: refs, diffs, ancestry, merge-tree, worktree list; counts calls."""

    refs: dict[str, str] = field(default_factory=dict)
    diffs: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    ancestors: set[tuple[str, str]] = field(default_factory=set)
    conflicts: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    worktrees: list[str] = field(default_factory=list)
    fail: set[str] = field(default_factory=set)  # subcommands that exit 128
    calls: list[list[str]] = field(default_factory=list)

    def install(self, runner: FakeRunner) -> FakeGit:
        runner.on(("git",), self)
        return self

    def __call__(self, call: Call) -> RunResult:
        args = call.argv[3:]  # after `git -C <main>`
        self.calls.append(args)
        assert call.timeout == 20
        if args[0] in self.fail:
            return RunResult(128, "", "fatal: boom")
        if args[0] == "rev-parse":
            sha = self.refs.get(args[-1])
            return RunResult(0, sha + "\n", "") if sha else RunResult(1, "", "")
        if args[0] == "merge-base":
            return RunResult(0 if (args[2], args[3]) in self.ancestors else 1, "", "")
        if args[0] == "diff":
            base, tip = args[2].split("...")
            return RunResult(0, "\n".join(self.diffs.get((base, tip), [])) + "\n", "")
        if args[0] == "merge-tree":
            a, b = args[-2], args[-1]
            files = self.conflicts.get((a, b))
            return RunResult(1, "tree\n" + "\n".join(files) + "\n", "") if files else RunResult(0, "tree\n", "")
        if args[0] == "worktree":
            return RunResult(0, "".join(f"worktree {p}\nHEAD x\n\n" for p in self.worktrees), "")
        raise AssertionError(args)

    def count(self) -> int:
        return len(self.calls)


@pytest.fixture
def git(fake_runner: FakeRunner) -> FakeGit:
    return FakeGit(refs={"refs/heads/main": "m"}).install(fake_runner)


def _tick(conn: sqlite3.Connection, fake_clock: clock.FakeClock, advance: float = detectors.DETECT_EVERY_S) -> None:
    detectors.detectors_hook(conn, fake_clock)
    fake_clock.advance(advance)


def _open(path: Path) -> list[tuple[str, str, str]]:
    return [
        (r[0], r[1], r[2])
        for r in sql(
            path, "SELECT run_id, kind, subject FROM alerts WHERE cleared_at IS NULL ORDER BY run_id, kind, subject"
        )
    ]


def _payload(path: Path, kind: str) -> dict[str, Any]:
    return json.loads(sql(path, "SELECT payload FROM alerts WHERE kind = ? ORDER BY id DESC", [kind])[0][0])


def _ago(fake_clock: clock.FakeClock, seconds: float) -> str:
    return clock.stamp(fake_clock, plus=-seconds)


# --------------------------------------------------------------------------- overlap


class TestOverlap:
    def _two_runs(self, migrated: Path, git: FakeGit) -> tuple[str, str]:
        r1, _ = open_run("r1", "S-1")
        r2, _ = open_run("r2", "S-2")
        insert_task(migrated, "T1", r1, branch="QS_1", is_deliverable=1)
        insert_task(migrated, "T2", r2, branch="QS_2", is_deliverable=1)
        git.refs.update({"refs/heads/QS_1": "t1", "refs/heads/QS_2": "t2"})
        git.diffs.update({("m", "t1"): ["a.py", "b.py"], ("m", "t2"): ["b.py", "c.py"]})
        git.conflicts[("t1", "t2")] = ["b.py"]
        return r1, r2

    def test_two_runs_deliverables_cross_run_with_conflicts(self, conn, migrated, git, fake_clock) -> None:
        r1, r2 = self._two_runs(migrated, git)
        _tick(conn, fake_clock)
        assert _open(migrated) == [(r1, "overlap_cross_run", "T1|T2"), (r2, "overlap_cross_run", "T1|T2")]
        assert _payload(migrated, "overlap_cross_run") == {
            "tasks": ["T1", "T2"],
            "files": ["b.py"],
            "truncated": False,
            "conflicts": ["b.py"],
        }

    def test_clears_when_fixed(self, conn, migrated, git, fake_clock) -> None:
        self._two_runs(migrated, git)
        _tick(conn, fake_clock)
        git.refs["refs/heads/QS_2"] = "t2b"
        git.diffs[("m", "t2b")] = ["c.py"]
        _tick(conn, fake_clock, detectors.OVERLAP_EVERY_S)
        _tick(conn, fake_clock)
        assert _open(migrated) == []

    def test_items(self, conn, migrated, git, fake_clock) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1, branch="QS_1", is_deliverable=1)
        insert_task(migrated, "T2", r1, branch="QS_2", is_deliverable=1)
        insert_task(migrated, "T3", r1, branch="QS_1_1", deliverable_id="T1", item_k=1)
        insert_task(migrated, "T4", r1, branch="QS_1_2", deliverable_id="T1", item_k=2)
        git.refs.update(
            {"refs/heads/QS_1": "t1", "refs/heads/QS_2": "t2", "refs/heads/QS_1_1": "i1", "refs/heads/QS_1_2": "i2"}
        )
        git.diffs.update(
            {("m", "t1"): ["a.py"], ("m", "t2"): ["x.py"], ("t1", "i1"): ["a.py", "x.py"], ("t1", "i2"): ["a.py"]}
        )
        _tick(conn, fake_clock)
        subjects = {s for _, _, s in _open(migrated)}
        assert subjects == {"T2|T3", "T3|T4"}  # never an item with its own deliverable (T1|T3, T1|T4)

    def test_unresolvable_branches_and_bases_are_skipped(self, conn, migrated, git, fake_clock) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1, branch="QS_1", is_deliverable=1)  # no ref
        insert_task(migrated, "T2", r1, branch="QS_2", is_deliverable=1)
        insert_task(migrated, "T3", r1, branch="QS_2_1", deliverable_id="T2", item_k=1)
        insert_task(migrated, "T5", r1, is_deliverable=1)
        insert_task(migrated, "T6", r1, branch="QS_5_1", deliverable_id="T5", item_k=1)  # the deliverable has no branch
        git.refs.update({"refs/heads/QS_2_1": "i1", "refs/heads/QS_5_1": "i5"})  # QS_2 (the item's base) is missing
        _tick(conn, fake_clock)
        assert _open(migrated) == [] and detectors._overlap.last == []

    @pytest.mark.parametrize(
        ("local", "remote", "ancestors", "base"),
        [
            ("m", None, set(), "m"),
            (None, "o", set(), "o"),
            ("m", "m", set(), "m"),
            ("m", "o", {("m", "o")}, "o"),
            ("m", "o", {("o", "m")}, "m"),
            ("m", "o", set(), "o"),
        ],
    )
    def test_the_main_base(self, conn, git, fake_clock, local, remote, ancestors, base) -> None:
        git.refs = {k: v for k, v in (("refs/heads/main", local), ("refs/remotes/origin/main", remote)) if v}
        git.ancestors = ancestors
        assert detectors._main_base(detectors._Git(conn, fake_clock, activeloop.seams())) == base

    def test_a_git_failure_clears_nothing(self, conn, migrated, git, fake_clock) -> None:
        self._two_runs(migrated, git)
        _tick(conn, fake_clock)
        git.refs["refs/heads/QS_2"] = "t2b"  # a moved tip: the diff must be recomputed
        git.fail.add("diff")
        _tick(conn, fake_clock, detectors.OVERLAP_EVERY_S)
        _tick(conn, fake_clock)
        assert len(_open(migrated)) == 2

    def test_a_failed_rev_parse_leaves_both_kinds_out(self, conn, migrated, git, fake_clock, capsys) -> None:
        self._two_runs(migrated, git)
        _tick(conn, fake_clock)
        _failing_rev_parse(git, "refs/heads/main")  # main's tip is unknown: the whole walk is
        _tick(conn, fake_clock, detectors.OVERLAP_EVERY_S)
        assert detectors.detect_overlap(conn, fake_clock, activeloop.seams()) == detectors.NONE
        assert len(_open(migrated)) == 2 and "overlap: git rev-parse" in capsys.readouterr().err

    def test_cached_conditions_are_re_emitted_and_git_busy_skips(
        self, conn, migrated, git, fake_clock, fake_main
    ) -> None:
        self._two_runs(migrated, git)
        _tick(conn, fake_clock)
        n = git.count()
        _tick(conn, fake_clock)  # not due yet: no git, the cached conditions are re-emitted
        assert git.count() == n and len(_open(migrated)) == 2
        (fake_main / ".git" / "MERGE_HEAD").write_text("x")
        fake_clock.advance(detectors.OVERLAP_EVERY_S)
        _tick(conn, fake_clock)
        assert git.count() == n and len(_open(migrated)) == 2

    def test_a_cold_daemon_omits_overlap_until_a_full_walk(self, conn, migrated, git, fake_clock, fake_main) -> None:
        self._two_runs(migrated, git)
        _tick(conn, fake_clock)
        messages = sql(migrated, "SELECT count(*) FROM messages")[0][0]
        ticks._reset_for_tests()
        activeloop._reset_for_tests()  # a restart
        (fake_main / ".git" / "MERGE_HEAD").write_text("x")  # no walk possible: overlap is unknown
        assert detectors.detect_overlap(conn, fake_clock, activeloop.seams()) == detectors.NONE
        _tick(conn, fake_clock)
        assert len(_open(migrated)) == 2
        (fake_main / ".git" / "MERGE_HEAD").unlink()
        _tick(conn, fake_clock)
        assert len(_open(migrated)) == 2 and sql(migrated, "SELECT count(*) FROM messages")[0][0] == messages

    def test_a_tick_makes_at_most_the_budget_and_the_walk_completes(self, conn, migrated, git, fake_clock) -> None:
        r1, _ = open_run()
        for i in range(1, 11):
            insert_task(migrated, f"T{i}", r1, branch=f"QS_{i}", is_deliverable=1)
            git.refs[f"refs/heads/QS_{i}"] = f"t{i}"
            git.diffs[("m", f"t{i}")] = ["shared.py"]
        per_tick = []
        for _ in range(10):
            before = git.count()
            kinds, conds = detectors.detect_overlap(conn, fake_clock, activeloop.seams())
            per_tick.append(git.count() - before)
            if kinds:
                break
            assert conds == []  # cold: nothing to re-emit yet
        assert max(per_tick) <= detectors.OVERLAP_MAX_CALLS and len(per_tick) > 1
        assert len(conds) == 45 and not detectors._overlap.walking

    def test_files_are_truncated(self, conn, migrated, git, fake_clock) -> None:
        self._two_runs(migrated, git)
        many = [f"f{i:03}.py" for i in range(60)]
        git.diffs.update({("m", "t1"): many, ("m", "t2"): many})
        _, conds = detectors.detect_overlap(conn, fake_clock, activeloop.seams())
        assert len(conds[0].payload["files"]) == 50 and conds[0].payload["truncated"] is True


# --------------------------------------------------------------------------- stalled node, rounds


class TestStalled:
    def test_a_quiet_running_node_is_stalled(self, conn, migrated, git, fake_clock) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1)
        insert_node(migrated, "N1", r1, "T1", launch_at=_ago(fake_clock, 4000), spawned_at=_ago(fake_clock, 4000))
        _tick(conn, fake_clock)
        assert _open(migrated) == [(r1, "node_stalled", "N1")]
        sql(
            migrated,
            "INSERT INTO messages (run_id, recipient, kind, sender, payload, state, created_at) VALUES (?, 'orchestrator', 'k', 'node:T1', '{}', 'queued', ?)",
            [r1, clock.stamp(fake_clock)],
        )
        _tick(conn, fake_clock)
        assert _open(migrated) == []

    def test_not_while_a_tool_call_is_in_flight(self, conn, migrated, git, fake_clock) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1)
        insert_node(migrated, "N1", r1, "T1", launch_at=_ago(fake_clock, 4000), spawned_at=_ago(fake_clock, 4000))
        sql(
            migrated,
            "INSERT INTO tool_calls (tool, key, args_hash, run_id, task_id, actor, args, state, holder_pid,"
            " holder_pid_start, started_at) VALUES ('push', 'k', 'h', ?, 'T1', 'node:T1', '{}', 'started', 4242,"
            " 'start-4242', ?)",
            [r1, _ago(fake_clock, 5000)],
        )
        _tick(conn, fake_clock)
        assert _open(migrated) == []

    def test_a_started_call_whose_holder_is_dead_does_not_hide_the_stall(
        self, conn, migrated, git, fake_clock, fake_probe
    ) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1)
        insert_node(migrated, "N1", r1, "T1", launch_at=_ago(fake_clock, 4000), spawned_at=_ago(fake_clock, 4000))
        sql(
            migrated,
            "INSERT INTO tool_calls (tool, key, args_hash, run_id, task_id, actor, args, state, holder_pid,"
            " holder_pid_start, started_at) VALUES ('push', 'k', 'h', ?, 'T1', 'node:T1', '{}', 'started', 4242,"
            " 'start-4242', ?)",
            [r1, _ago(fake_clock, 5000)],
        )
        fake_probe.kill(4242)  # a SIGKILLed `cp.py`: the row stays `started` forever
        _tick(conn, fake_clock)
        assert _open(migrated) == [(r1, "node_stalled", "N1")]


class TestRounds:
    def test_round_6_then_7(self, conn, migrated, git, fake_clock) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1)
        for rnd in range(1, 7):
            sql(
                migrated,
                "INSERT INTO reports (task_id, phase, round, status, summary, fields, at) VALUES ('T1', 'build', ?, 'continuing', 's', '{}', 'x')",
                [rnd],
            )
        _tick(conn, fake_clock)
        assert _open(migrated) == [(r1, "too_many_rounds", "T1:build:r6")]
        sql(
            migrated,
            "INSERT INTO reports (task_id, phase, round, status, summary, fields, at) VALUES ('T1', 'build', 7, 'continuing', 's', '{}', 'x')",
        )
        _tick(conn, fake_clock)
        assert _open(migrated) == [(r1, "too_many_rounds", "T1:build:r7")]


# --------------------------------------------------------------------------- state anomalies


class TestAnomalies:
    def test_task_without_node(self, conn, migrated, git, fake_clock) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1, state="building")
        sql(
            migrated,
            "INSERT INTO task_history (task_id, at, actor, to_state) VALUES ('T1', ?, 'a', 'building')",
            [_ago(fake_clock, 700)],
        )
        _tick(conn, fake_clock)
        assert _open(migrated) == [(r1, "task_without_node", "T1")]
        c = conn
        from control_plane import db

        with db.write(c):
            tasks.update_fields(c, fake_clock, "T1", {"ci_state": "green"})  # writes no history: the grace holds
        _tick(conn, fake_clock)
        assert _open(migrated) == [(r1, "task_without_node", "T1")]
        insert_node(migrated, "N1", r1, "T1", state="running", launch_at=clock.stamp(fake_clock))
        _tick(conn, fake_clock)
        assert _open(migrated) == []

    @pytest.mark.parametrize(
        ("dependent", "blocked_from", "dependency", "flagged"),
        [
            ("building", None, "ready", True),
            ("ready_to_merge", None, "dropped", True),
            ("planning", None, "merged", False),
            ("contracted", None, "validated", False),
            ("blocked", "building", "building", True),
            ("blocked", "ready", "building", False),
            ("ready", None, "building", False),
            ("merged", None, "dropped", False),
        ],
    )
    def test_dependency_violated(
        self, conn, migrated, git, fake_clock, dependent, blocked_from, dependency, flagged
    ) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1, state=dependent, blocked_from=blocked_from)
        insert_task(migrated, "T2", r1, state=dependency)
        sql(migrated, "INSERT INTO task_deps (task_id, depends_on) VALUES ('T1', 'T2')")
        kinds, conds = detectors.detect_anomalies(conn, fake_clock, activeloop.seams())
        violated = [c for c in conds if c.kind == "dependency_violated"]
        assert [c.subject for c in violated] == (["T1->T2"] if flagged else [])

    @pytest.mark.parametrize(
        ("name", "holder_kind", "age", "flagged"),
        [
            ("main-merge", "process", 1300, True),
            ("main-checkout", "process", 1100, False),
            ("integration:QS_9", "process", 1300, True),
            ("integration:QS_9", "session", 3000, False),  # co-held by integrate-finish for 50 min
            ("integration:QS_9", "session", 11000, True),
            ("other", "session", 3700, True),
            ("other", "process", 3500, False),
        ],
    )
    def test_lock_held_long(self, conn, migrated, git, fake_clock, name, holder_kind, age, flagged) -> None:
        r1, _ = open_run()
        sql(
            migrated,
            "INSERT INTO locks (name, holder_kind, holder_actor, token_subject, acquired_at) VALUES (?, ?, 'a', 'run:R1', ?)",
            [name, holder_kind, _ago(fake_clock, age)],
        )
        _, conds = detectors.detect_anomalies(conn, fake_clock, activeloop.seams())
        held = [c for c in conds if c.kind == "lock_held_long"]
        assert len(held) == int(flagged) and all(c.run_ids == (r1,) for c in held)

    def test_lock_routing(self, conn, migrated, git, fake_clock) -> None:
        r1, _ = open_run("r1", "S-1")
        r2, _ = open_run("r2", "S-2")
        insert_task(migrated, "T1", r2)
        insert_node(migrated, "N1", r2, "T1")
        old = _ago(fake_clock, 4000)
        for name, subject in (("a", "node:N1"), ("b", "node:N9"), ("c", "x:y")):
            sql(
                migrated,
                "INSERT INTO locks (name, holder_kind, holder_actor, token_subject, acquired_at) VALUES (?, 'session', 'a', ?, ?)",
                [name, subject, old],
            )
        _, conds = detectors.detect_anomalies(conn, fake_clock, activeloop.seams())
        routes = {c.subject.split("@")[0]: c.run_ids for c in conds if c.kind == "lock_held_long"}
        assert routes == {"a": (r2,), "b": (r1, r2), "c": (r1, r2)}

    def test_gate_slot_held_long(self, conn, migrated, git, fake_clock) -> None:
        r1, _ = open_run("r1", "S-1")
        r2, _ = open_run("r2", "S-2")
        for slot, pid in ((0, 5), (1, 6)):
            sql(
                migrated,
                "INSERT INTO cap_slots (cap, slot, holder_pid, holder_actor, acquired_at) VALUES ('gates', ?, ?, 'a', ?)",
                [slot, pid, _ago(fake_clock, 5000)],
            )
        sql(
            migrated,
            "INSERT INTO tool_calls (tool, key, args_hash, run_id, actor, args, state, holder_pid, started_at) VALUES ('gate', 'k', 'h', ?, 'a', '{}', 'started', 5, 'x')",
            [r2],
        )
        _, conds = detectors.detect_anomalies(conn, fake_clock, activeloop.seams())
        routes = {c.subject.split("@")[0]: c.run_ids for c in conds if c.kind == "gate_slot_held_long"}
        assert routes == {"gates#0": (r2,), "gates#1": (r1, r2)}

    def test_leftovers(self, conn, migrated, git, fake_clock, tmp_path) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1, state="validated", branch="QS_1", is_deliverable=1)
        insert_task(migrated, "T2", r1, branch="QS_1_1", deliverable_id="T1", item_k=1, worktree="/w/QS_1_1")
        insert_task(migrated, "T3", r1, branch="QS_1_2", deliverable_id="T1", item_k=2)
        git.worktrees = ["/m", f"{tmp_path}/QS_1_2_integration", "/x/QS_7_1_integration", "/y/other"]
        _tick(conn, fake_clock)
        assert _open(migrated) == [(r1, "leftover_item", "T2"), (r1, "leftover_scratch", "T3")]
        git.fail.add("worktree")  # a failed listing: leftover_scratch is unknown, so it stays
        _tick(conn, fake_clock)
        assert (r1, "leftover_scratch", "T3") in _open(migrated)

    def test_no_finished_item_needs_no_listing(self, conn, migrated, git, fake_clock) -> None:
        _tick(conn, fake_clock)
        assert git.calls == []


# --------------------------------------------------------------------------- cycles, duplicates


class TestCyclesAndDuplicates:
    def test_a_two_run_cycle_alerts_both_runs(self, conn, migrated, git, fake_clock) -> None:
        r1, _ = open_run("r1", "S-1")
        r2, _ = open_run("r2", "S-2")
        for tid, run in (("T1", r1), ("X", r1), ("T2", r2), ("Y", r2), ("A", r1), ("B", r1)):
            insert_task(migrated, tid, run, state="proposed", title=f"title {tid}")
        for a, b in (("T1", "X"), ("X", "T2"), ("T2", "Y"), ("Y", "T1"), ("A", "B"), ("B", "A")):
            sql(migrated, "INSERT INTO task_deps (task_id, depends_on) VALUES (?, ?)", [a, b])
        _tick(conn, fake_clock)
        assert _open(migrated) == [
            (r1, "dependency_cycle", "A|B"),
            (r1, "dependency_cycle_cross_run", "T1|T2|X|Y"),
            (r2, "dependency_cycle_cross_run", "T1|T2|X|Y"),
        ]

    def test_strongly_connected(self) -> None:
        comps = detectors.strongly_connected(
            ["a", "b", "c", "d"], {"a": ["b"], "b": ["c", "a"], "c": ["d"], "d": ["c"]}
        )
        assert sorted(sorted(c) for c in comps) == [["a", "b"], ["c", "d"]]

    def test_a_cross_run_duplicate(self, conn, migrated, git, fake_clock) -> None:
        r1, _ = open_run("r1", "S-1")
        r2, _ = open_run("r2", "S-2")
        insert_task(migrated, "T1", r1, title="QS-12: Fix the Thing!")
        insert_task(migrated, "T2", r2, title="fix  the_thing")
        insert_task(migrated, "T3", r1, title="!!!")  # normalises to nothing
        insert_task(migrated, "T4", r1, title="!!!")
        insert_task(migrated, "T5", r1, title="fix the thing", state="merged")  # terminal: ignored
        _tick(conn, fake_clock)
        assert _open(migrated) == [(r1, "duplicate_task_cross_run", "T1|T2"), (r2, "duplicate_task_cross_run", "T1|T2")]

    @pytest.mark.parametrize(
        ("title", "norm"),
        [("QS-12: Fix it", "fix it"), ("qs_7 — Ｆｕｌｌ width", "full width"), ("Straße", "strasse"), ("  ", "")],
    )
    def test_normalise_title(self, title: str, norm: str) -> None:
        assert detectors.normalise_title(title) == norm


# --------------------------------------------------------------------------- robustness, throttle


def test_x_stamped_rows_never_crash_or_fire(conn, migrated, git, fake_clock) -> None:
    r1, _ = open_run()
    insert_task(migrated, "T1", r1, state="building")
    insert_node(migrated, "N1", r1, "T1", launch_at="x")
    sql(
        migrated,
        "INSERT INTO locks (name, holder_kind, holder_actor, token_subject, acquired_at) VALUES ('l', 'process', 'a', 'run:R1', 'x')",
    )
    sql(
        migrated,
        "INSERT INTO cap_slots (cap, slot, holder_pid, holder_actor, acquired_at) VALUES ('gates', 0, 5, 'a', 'x')",
    )
    _tick(conn, fake_clock)
    assert _open(migrated) == []


def test_the_hook_is_throttled(conn, migrated, git, fake_clock, monkeypatch) -> None:
    runs: list[int] = []
    monkeypatch.setattr(detectors, "ALL", ((lambda c, k, s: runs.append(1) or detectors.NONE),))
    detectors.detectors_hook(conn, fake_clock)
    fake_clock.advance(detectors.DETECT_EVERY_S - 1)
    detectors.detectors_hook(conn, fake_clock)
    fake_clock.advance(1)
    detectors.detectors_hook(conn, fake_clock)
    assert len(runs) == 2


# --------------------------------------------------------------------------- review fix #01 (F6, F8, F9, F19, F20)


class TestReviewFix01:
    def _three(self, migrated: Path, git: FakeGit) -> str:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1, branch="QS_1", is_deliverable=1)
        insert_task(migrated, "T2", r1, branch="QS_2", is_deliverable=1)
        insert_task(migrated, "T9", r1, branch="QS_9", is_deliverable=1)  # an orphan branch: no merge base
        git.refs.update({"refs/heads/QS_1": "t1", "refs/heads/QS_2": "t2", "refs/heads/QS_9": "t9"})
        git.diffs.update({("m", "t1"): ["a.py"], ("m", "t2"): ["a.py"]})
        return r1

    def test_one_bad_branch_does_not_disable_overlap(self, conn, migrated, git, fake_clock, capsys) -> None:
        r1 = self._three(migrated, git)
        real = git.__call__

        def no_base(call: Call) -> RunResult:
            if call.argv[3] == "diff" and call.argv[-1].endswith("...t9"):
                git.calls.append(call.argv[3:])
                return RunResult(128, "", "fatal: no merge base")
            return real(call)

        git_runner = activeloop.seams().runner
        assert isinstance(git_runner, FakeRunner)
        git_runner.on(("git",), no_base)
        _tick(conn, fake_clock)
        assert _open(migrated) == [(r1, "overlap", "T1|T2")]
        _tick(conn, fake_clock, detectors.OVERLAP_EVERY_S)
        _tick(conn, fake_clock)
        assert capsys.readouterr().err.count("no merge base") == 1  # logged once

    def test_a_failed_merge_tree_reports_unknown_conflicts(self, conn, migrated, git, fake_clock) -> None:
        self._three(migrated, git)
        git.fail.add("merge-tree")
        _tick(conn, fake_clock)
        assert _payload(migrated, "overlap")["conflicts"] is None

    def test_an_unknown_branch_keeps_its_last_alert(self, conn, migrated, git, fake_clock) -> None:
        r1 = self._three(migrated, git)
        insert_task(migrated, "T3", r1, branch="QS_3", is_deliverable=1)
        git.refs["refs/heads/QS_3"] = "t3"
        git.diffs.update({("m", "t3"): ["b.py"], ("m", "t1"): ["a.py", "b.py"]})
        _tick(conn, fake_clock)
        assert {s for _, _, s in _open(migrated)} == {"T1|T2", "T1|T3"}
        git.refs["refs/heads/QS_3"] = "t3b"  # moved, and its diff now fails
        real = git.__call__

        def failing(call: Call) -> RunResult:
            if call.argv[3] == "diff" and call.argv[-1].endswith("...t3b"):
                return RunResult(128, "", "fatal: boom")
            return real(call)

        activeloop.seams().runner.on(("git",), failing)  # type: ignore[attr-defined]
        git.refs["refs/heads/QS_2"] = "t2b"  # and T2 stopped touching a.py
        git.diffs[("m", "t2b")] = ["z.py"]
        _tick(conn, fake_clock, detectors.OVERLAP_EVERY_S)
        _tick(conn, fake_clock)
        assert {s for _, _, s in _open(migrated)} == {"T1|T3"}  # T3 is unknown: kept; T1|T2 is known gone

    def test_caches_are_pruned_after_a_complete_walk(self, conn, migrated, git, fake_clock) -> None:
        self._three(migrated, git)
        git.diffs[("m", "t9")] = ["q.py"]
        _tick(conn, fake_clock)
        assert ("m", "t9") in detectors._overlap.diffs
        sql(migrated, "UPDATE tasks SET state = 'merged' WHERE id = 'T9'")
        git.refs["refs/heads/QS_2"] = "t2b"
        git.diffs[("m", "t2b")] = ["a.py"]
        _tick(conn, fake_clock, detectors.OVERLAP_EVERY_S)
        _tick(conn, fake_clock)
        assert set(detectors._overlap.diffs) == {("m", "t1"), ("m", "t2b")}
        assert set(detectors._overlap.conflicts) == {("t1", "t2b")}

    def test_one_raising_detector_does_not_skip_the_rest(
        self, conn, migrated, git, fake_clock, monkeypatch, capsys
    ) -> None:
        synced: list[int] = []

        def boom(c: Any, k: Any, s: Any) -> detectors.Result:
            raise RuntimeError("detector exploded")

        def fine(c: Any, k: Any, s: Any) -> detectors.Result:
            synced.append(1)
            return detectors.NONE

        monkeypatch.setattr(detectors, "ALL", (boom, fine))
        detectors.detectors_hook(conn, fake_clock)
        assert synced == [1] and "detector exploded" in capsys.readouterr().err

    def test_in_lists_are_bound_parameters(self) -> None:
        assert detectors._in(("a", "b'c")) == ("(?, ?)", ("a", "b'c"))


# --------------------------------------------------------------------------- review fix #02 (G2, G13, G14, G17)


def _failing_diff(git: FakeGit, tip_suffix: str, result: RunResult) -> None:
    real = git.__call__

    def respond(call: Call) -> RunResult:
        if call.argv[3] == "diff" and call.argv[-1].endswith(tip_suffix):
            git.calls.append(call.argv[3:])
            return result
        return real(call)

    activeloop.seams().runner.on(("git",), respond)  # type: ignore[attr-defined]


def _failing_rev_parse(git: FakeGit, ref: str) -> None:
    real = git.__call__

    def respond(call: Call) -> RunResult:
        if call.argv[3] == "rev-parse" and call.argv[-1] == ref:
            git.calls.append(call.argv[3:])
            return RunResult(128, "", f"fatal: bad ref {ref}")
        return real(call)

    activeloop.seams().runner.on(("git",), respond)  # type: ignore[attr-defined]


def _diff_calls(git: FakeGit, tip_suffix: str = "") -> int:
    return sum(1 for c in git.calls if c[0] == "diff" and c[-1].endswith(tip_suffix))


class TestReviewFix02:
    @pytest.mark.parametrize("code", [124, 127])
    def test_a_timeout_or_a_missing_git_aborts_the_walk(self, conn, migrated, git, fake_clock, capsys, code) -> None:
        r1, _ = open_run()
        for i in range(1, 6):
            insert_task(migrated, f"T{i}", r1, branch=f"QS_{i}", is_deliverable=1)
            git.refs[f"refs/heads/QS_{i}"] = f"t{i}"
        _failing_diff(git, "", RunResult(code, "", "timed out"))
        assert detectors.detect_overlap(conn, fake_clock, activeloop.seams()) == detectors.NONE
        assert _diff_calls(git) == 1 and not detectors._overlap.walking
        assert f"exited {code}" in capsys.readouterr().err

    def test_a_failed_diff_is_not_retried_by_a_resumed_walk(self, conn, migrated, git, fake_clock) -> None:
        r1, _ = open_run()
        for i in range(1, 11):
            insert_task(migrated, f"T{i}", r1, branch=f"QS_{i}", is_deliverable=1)
            git.refs[f"refs/heads/QS_{i}"] = f"t{i}"
            git.diffs[("m", f"t{i}")] = ["shared.py"]
        _failing_diff(git, "...t1", RunResult(128, "", "fatal: no merge base"))
        for _ in range(10):
            kinds, _conds = detectors.detect_overlap(conn, fake_clock, activeloop.seams())
            if kinds:
                break
        assert kinds and _diff_calls(git, "...t1") == 1
        fake_clock.advance(detectors.OVERLAP_EVERY_S)
        detectors.detect_overlap(conn, fake_clock, activeloop.seams())  # a new walk retries it
        assert _diff_calls(git, "...t1") == 2

    def test_a_failed_merge_tree_is_not_retried_by_a_resumed_walk(self, conn, migrated, git, fake_clock) -> None:
        r1, _ = open_run()
        for i in range(1, 11):
            insert_task(migrated, f"T{i}", r1, branch=f"QS_{i}", is_deliverable=1)
            git.refs[f"refs/heads/QS_{i}"] = f"t{i}"
            git.diffs[("m", f"t{i}")] = ["shared.py"]
        real = git.__call__

        def respond(call: Call) -> RunResult:
            if call.argv[3] == "merge-tree" and call.argv[-2:] == ["t1", "t2"]:
                git.calls.append(call.argv[3:])
                return RunResult(128, "", "fatal: boom")
            return real(call)

        activeloop.seams().runner.on(("git",), respond)  # type: ignore[attr-defined]
        runs = 0
        for runs in range(1, 11):
            kinds, conds = detectors.detect_overlap(conn, fake_clock, activeloop.seams())
            if kinds:
                break
        assert runs > 1 and sum(1 for c in git.calls if c[0] == "merge-tree" and c[-2:] == ["t1", "t2"]) == 1
        assert next(c for c in conds if c.subject == "T1|T2").payload["conflicts"] is None

    def test_a_merge_tree_timeout_aborts_the_walk(self, conn, migrated, git, fake_clock) -> None:
        TestOverlap()._two_runs(migrated, git)
        real = git.__call__

        def respond(call: Call) -> RunResult:
            if call.argv[3] == "merge-tree":
                return RunResult(124, "", "")
            return real(call)

        activeloop.seams().runner.on(("git",), respond)  # type: ignore[attr-defined]
        assert detectors.detect_overlap(conn, fake_clock, activeloop.seams()) == detectors.NONE

    def test_a_branch_failing_with_changing_messages_is_logged_once_per_episode(
        self, conn, migrated, git, fake_clock, capsys
    ) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1, branch="QS_1", is_deliverable=1)
        for tip, stderr in (("t1", "fatal: no merge base"), ("t1b", "fatal: bad object t1b")):
            git.refs["refs/heads/QS_1"] = tip
            _failing_diff(git, f"...{tip}", RunResult(128, "", stderr))
            detectors.detect_overlap(conn, fake_clock, activeloop.seams())
            fake_clock.advance(detectors.OVERLAP_EVERY_S)
        assert capsys.readouterr().err.count("overlap: T1") == 1
        git.refs["refs/heads/QS_1"] = "t1c"  # its diff works again
        activeloop.seams().runner.on(("git",), git)  # type: ignore[attr-defined]
        detectors.detect_overlap(conn, fake_clock, activeloop.seams())
        fake_clock.advance(detectors.OVERLAP_EVERY_S)
        git.refs["refs/heads/QS_1"] = "t1d"
        _failing_diff(git, "...t1d", RunResult(128, "", "fatal: again"))
        detectors.detect_overlap(conn, fake_clock, activeloop.seams())
        assert capsys.readouterr().err.count("overlap: T1") == 1  # a new episode: logged again

    def test_a_raising_detector_is_logged_once(self, conn, migrated, git, fake_clock, monkeypatch, capsys) -> None:
        fail = [True]

        def flaky(c: Any, k: Any, s: Any) -> detectors.Result:
            if fail[0]:
                raise RuntimeError(f"exploded at {fake_clock.now()}")
            return detectors.NONE

        monkeypatch.setattr(detectors, "ALL", (flaky,))
        _tick(conn, fake_clock)
        _tick(conn, fake_clock)
        assert capsys.readouterr().err.count("exploded") == 1
        fail[0] = False
        _tick(conn, fake_clock)
        fail[0] = True
        _tick(conn, fake_clock)
        assert capsys.readouterr().err.count("exploded") == 1  # recovered, then failed again

    def _alerted_pair(self, conn, migrated: Path, git: FakeGit, fake_clock) -> str:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1, branch="QS_1", is_deliverable=1)
        insert_task(migrated, "T3", r1, branch="QS_3", is_deliverable=1)
        git.refs.update({"refs/heads/QS_1": "t1", "refs/heads/QS_3": "t3"})
        git.diffs.update({("m", "t1"): ["a.py"], ("m", "t3"): ["a.py"]})
        _tick(conn, fake_clock)
        assert _open(migrated) == [(r1, "overlap", "T1|T3")]
        git.refs["refs/heads/QS_3"] = "t3b"
        _failing_diff(git, "...t3b", RunResult(128, "", "fatal: boom"))  # T3 is unknown from now on
        return r1

    def test_a_kept_alert_whose_partner_is_gone_is_cleared(self, conn, migrated, git, fake_clock) -> None:
        self._alerted_pair(conn, migrated, git, fake_clock)
        sql(migrated, "UPDATE tasks SET state = 'merged' WHERE id = 'T1'")
        _tick(conn, fake_clock, detectors.OVERLAP_EVERY_S)
        _tick(conn, fake_clock)
        assert _open(migrated) == []

    def test_a_restarted_daemon_keeps_an_unknown_branchs_alert(self, conn, migrated, git, fake_clock) -> None:
        r1 = self._alerted_pair(conn, migrated, git, fake_clock)
        messages = sql(migrated, "SELECT count(*) FROM messages")[0][0]
        detectors._overlap = detectors._OverlapState()  # a restart: no last result in memory
        ticks._reset_for_tests()
        _tick(conn, fake_clock, detectors.OVERLAP_EVERY_S)
        assert _open(migrated) == [(r1, "overlap", "T1|T3")]
        assert sql(migrated, "SELECT count(*) FROM messages")[0][0] == messages  # not a recurrence

    def test_a_restarted_daemon_seeds_only_alerts_of_an_unknown_task(self, conn, migrated, git, fake_clock) -> None:
        self._alerted_pair(conn, migrated, git, fake_clock)
        activeloop.seams().runner.on(("git",), git)  # type: ignore[attr-defined]
        git.diffs[("m", "t3b")] = ["z.py"]  # T3 is known again and no longer overlaps
        detectors._overlap = detectors._OverlapState()
        ticks._reset_for_tests()
        _tick(conn, fake_clock, detectors.OVERLAP_EVERY_S)
        assert _open(migrated) == []

    def test_the_gate_slot_holder_matches_its_pid_start(self, conn, migrated, git, fake_clock) -> None:
        r1, _ = open_run("r1", "S-1")
        r2, _ = open_run("r2", "S-2")
        sql(
            migrated,
            "INSERT INTO cap_slots (cap, slot, holder_pid, holder_pid_start, holder_actor, acquired_at)"
            " VALUES ('gates', 0, 5, 'start-new', 'a', ?)",
            [_ago(fake_clock, 5000)],
        )
        sql(  # an older process that had pid 5
            migrated,
            "INSERT INTO tool_calls (tool, key, args_hash, run_id, actor, args, state, holder_pid, holder_pid_start,"
            " started_at) VALUES ('gate', 'k', 'h', ?, 'a', '{}', 'started', 5, 'start-old', 'x')",
            [r2],
        )
        _, conds = detectors.detect_anomalies(conn, fake_clock, activeloop.seams())
        [cond] = [c for c in conds if c.kind == "gate_slot_held_long"]
        assert cond.run_ids == (r1, r2)  # not routed to r2 by a reused pid


def test_a_self_dependency_cannot_be_stored(conn, migrated) -> None:
    # G15: `task_deps` has CHECK (task_id != depends_on), so a one-node cycle never reaches `detect_cycles`
    r1, _ = open_run()
    insert_task(migrated, "T1", r1)
    with pytest.raises(sqlite3.IntegrityError):
        sql(migrated, "INSERT INTO task_deps (task_id, depends_on) VALUES ('T1', 'T1')")


# --------------------------------------------------------------------------- review fix #03 (H3, H4, H6)


class TestReviewFix03:
    def _three(self, migrated: Path, git: FakeGit) -> None:
        r1, _ = open_run()
        for i in range(1, 4):
            insert_task(migrated, f"T{i}", r1, branch=f"QS_{i}", is_deliverable=1)
            git.refs[f"refs/heads/QS_{i}"] = f"t{i}"
            git.diffs[("m", f"t{i}")] = ["shared.py"]

    def test_an_unexpected_error_mid_walk_resets_the_walk(self, conn, migrated, git, fake_clock, monkeypatch) -> None:
        self._three(migrated, git)
        _failing_diff(git, "...t1", RunResult(128, "", "fatal: no merge base"))
        raised: list[int] = []
        real_beat = detectors.daemon.beat

        def beat(c: sqlite3.Connection, k: clock.Clock) -> None:
            if _diff_calls(git, "...t2") == 1 and not raised:
                raised.append(1)
                raise errors.CpError("INTERNAL", "database disk image is malformed")
            real_beat(c, k)

        monkeypatch.setattr(detectors.daemon, "beat", beat)
        with pytest.raises(errors.CpError):
            detectors.detect_overlap(conn, fake_clock, activeloop.seams())
        assert _diff_calls(git, "...t1") == 1 and not detectors._overlap.failed_diffs
        assert not detectors._overlap.refs and not detectors._overlap.walking
        detectors.detect_overlap(conn, fake_clock, activeloop.seams())  # it backs off: no new walk (I4)
        assert _diff_calls(git, "...t1") == 1
        fake_clock.advance(detectors.OVERLAP_EVERY_S)
        detectors.detect_overlap(conn, fake_clock, activeloop.seams())  # the next walk retries T1's diff
        assert _diff_calls(git, "...t1") == 2

    def test_an_aborted_walk_waits_the_overlap_interval(self, conn, migrated, git, fake_clock) -> None:
        self._three(migrated, git)
        _failing_diff(git, "", RunResult(124, "", "timed out"))
        assert detectors.detect_overlap(conn, fake_clock, activeloop.seams()) == detectors.NONE
        assert _diff_calls(git) == 1
        fake_clock.advance(detectors.DETECT_EVERY_S)
        detectors.detect_overlap(conn, fake_clock, activeloop.seams())  # not due yet: no new walk
        assert _diff_calls(git) == 1
        fake_clock.advance(detectors.OVERLAP_EVERY_S - detectors.DETECT_EVERY_S)
        detectors.detect_overlap(conn, fake_clock, activeloop.seams())
        assert _diff_calls(git) == 2

    def test_a_failing_worktree_listing_is_logged_once_per_episode(
        self, conn, migrated, git, fake_clock, capsys
    ) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1, state="validated", branch="QS_1", is_deliverable=1)
        insert_task(migrated, "T3", r1, branch="QS_1_2", deliverable_id="T1", item_k=2)
        git.fail.add("worktree")
        for _ in range(2):
            assert detectors._leftover_scratch(conn, fake_clock, activeloop.seams()) is None
        assert capsys.readouterr().err.count("leftover scratch") == 1
        git.fail.discard("worktree")
        assert detectors._leftover_scratch(conn, fake_clock, activeloop.seams()) == []
        git.fail.add("worktree")
        assert detectors._leftover_scratch(conn, fake_clock, activeloop.seams()) is None
        assert capsys.readouterr().err.count("leftover scratch") == 1  # a new episode


# --------------------------------------------------------------------------- review fix #04 (I4)


class TestReviewFix04:
    def test_an_unexpected_error_backs_off_the_overlap_interval(
        self, conn, migrated, git, fake_clock, monkeypatch
    ) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1, branch="QS_1", is_deliverable=1)
        git.refs["refs/heads/QS_1"] = "t1"

        def beat(c: sqlite3.Connection, k: clock.Clock) -> None:
            raise errors.CpError("INTERNAL", "database disk image is malformed")  # a persistent failure

        monkeypatch.setattr(detectors.daemon, "beat", beat)
        with pytest.raises(errors.CpError):
            detectors.detect_overlap(conn, fake_clock, activeloop.seams())
        n = git.count()
        fake_clock.advance(detectors.DETECT_EVERY_S)
        detectors.detect_overlap(conn, fake_clock, activeloop.seams())  # not due: no re-walk every tick
        assert git.count() == n
        fake_clock.advance(detectors.OVERLAP_EVERY_S)
        with pytest.raises(errors.CpError):
            detectors.detect_overlap(conn, fake_clock, activeloop.seams())
        assert git.count() == n + 1

    @pytest.mark.parametrize("bad", ["refs/heads/QS_2", "refs/heads/QS_1"])
    def test_a_failed_rev_parse_of_one_branch_marks_only_its_tasks_unknown(
        self, conn, migrated, git, fake_clock, capsys, bad: str
    ) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1, branch="QS_1", is_deliverable=1)
        insert_task(migrated, "T2", r1, branch="QS_2", is_deliverable=1)
        insert_task(migrated, "T3", r1, branch="QS_3", is_deliverable=1)
        insert_task(migrated, "T4", r1, branch="QS_4", deliverable_id="T1", item_k=4)  # based on T1's branch
        git.refs.update({f"refs/heads/QS_{i}": f"t{i}" for i in range(1, 5)})
        git.diffs.update({("m", "t1"): ["a.py"], ("m", "t2"): ["a.py"], ("m", "t3"): ["a.py"], ("t1", "t4"): ["a.py"]})
        _tick(conn, fake_clock, detectors.OVERLAP_EVERY_S)
        before = _open(migrated)
        assert (r1, "overlap", "T2|T3") in before
        _failing_rev_parse(git, bad)
        for _ in range(2):  # two complete walks
            _tick(conn, fake_clock, detectors.OVERLAP_EVERY_S)
            assert detectors._overlap.last is not None and not detectors._overlap.walking
            assert _open(migrated) == before  # the unknown tasks keep their alerts; the others are recomputed
        err = capsys.readouterr().err
        affected = {"refs/heads/QS_2": 1, "refs/heads/QS_1": 2}[bad]  # QS_1 is also T4's base
        assert err.count(f"bad ref {bad}") == affected  # once per task, not once per walk
        assert "overlap: git rev-parse" not in err  # the walk itself did not fail


# --------------------------------------------------------------------------- review fix #05 (J1)


class TestReviewFix05:
    def test_a_repo_wide_rev_parse_failure_fails_the_walk_with_one_call(
        self, conn, migrated, git, fake_clock, capsys
    ) -> None:
        r1, _ = open_run()
        for i in range(1, detectors.OVERLAP_MAX_CALLS + 6):  # more tasks than one tick's budget
            insert_task(migrated, f"T{i:02}", r1, branch=f"QS_{i}", is_deliverable=1)
            git.refs[f"refs/heads/QS_{i}"] = f"t{i}"
        git.fail.add("rev-parse")  # dubious ownership, a corrupt ref store: every rev-parse exits 128
        assert detectors.detect_overlap(conn, fake_clock, activeloop.seams()) == detectors.NONE
        assert git.count() == 1 and not detectors._overlap.walking  # main's rev-parse fails first: the H4 path
        assert "overlap: git rev-parse" in capsys.readouterr().err
        fake_clock.advance(detectors.OVERLAP_EVERY_S - 1)
        detectors.detect_overlap(conn, fake_clock, activeloop.seams())  # not due: no new walk
        assert git.count() == 1
        fake_clock.advance(1)
        detectors.detect_overlap(conn, fake_clock, activeloop.seams())
        assert git.count() == 2

    def test_no_task_no_git(self, conn, migrated, git, fake_clock) -> None:
        assert detectors.detect_overlap(conn, fake_clock, activeloop.seams()) == (detectors.OVERLAP_KINDS, [])
        assert git.count() == 0

    def test_a_broken_deliverable_ref_fails_once_across_a_resumed_walk(
        self, conn, migrated, git, fake_clock, capsys
    ) -> None:
        r1, _ = open_run()
        insert_task(migrated, "D", r1, branch="QS_9", is_deliverable=1)  # its ref is broken
        insert_task(migrated, "A1", r1, branch="QS_9_1", deliverable_id="D", item_k=1)  # walked first
        insert_task(migrated, "Z2", r1, branch="QS_9_2", deliverable_id="D", item_k=2)  # walked last
        git.refs.update({"refs/heads/QS_9_1": "i1", "refs/heads/QS_9_2": "i2"})
        for i in range(1, 16):  # 2 calls each: the walk needs more than one tick
            insert_task(migrated, f"F{i:02}", r1, branch=f"QS_{100 + i}", is_deliverable=1)
            git.refs[f"refs/heads/QS_{100 + i}"] = f"t{i}"
        _failing_rev_parse(git, "refs/heads/QS_9")
        ticks_taken = 0
        while True:
            ticks_taken += 1
            kinds, _conds = detectors.detect_overlap(conn, fake_clock, activeloop.seams())
            if kinds:
                break
            assert ticks_taken < 10
        assert ticks_taken > 1 and not detectors._overlap.walking and not detectors._overlap.failed_refs
        assert sum(1 for c in git.calls if c[0] == "rev-parse" and c[-1] == "refs/heads/QS_9") == 1
        assert capsys.readouterr().err.count("bad ref refs/heads/QS_9") == 3  # once per task (D, A1, Z2)


# --------------------------------------------------------------------------- review fix #06 (K1)


class TestReviewFix06:
    @pytest.mark.parametrize("code", [124, 127])
    def test_a_task_rev_parse_timeout_or_missing_git_aborts_the_walk(
        self, conn, migrated, git, fake_clock, capsys, code
    ) -> None:
        r1, _ = open_run()
        for i in range(1, 4):
            insert_task(migrated, f"T{i}", r1, branch=f"QS_{i}", is_deliverable=1)
            git.refs[f"refs/heads/QS_{i}"] = f"t{i}"
            git.diffs[("m", f"t{i}")] = ["shared.py"]
        kinds, _conds = detectors.detect_overlap(conn, fake_clock, activeloop.seams())
        last = detectors._overlap.last
        assert kinds and last
        fake_clock.advance(detectors.OVERLAP_EVERY_S)
        real = git.__call__

        def respond(call: Call) -> RunResult:  # main resolves; T2's own branch tip does not
            if call.argv[3] == "rev-parse" and call.argv[-1] == "refs/heads/QS_2":
                git.calls.append(call.argv[3:])
                return RunResult(code, "", "timed out")
            return real(call)

        activeloop.seams().runner.on(("git",), respond)  # type: ignore[attr-defined]
        git.calls.clear()
        assert detectors.detect_overlap(conn, fake_clock, activeloop.seams()) == detectors.NONE  # the H4 path
        assert detectors._overlap.last == last and detectors._overlap.last_at == fake_clock.now()
        assert not detectors._overlap.walking and not detectors._overlap.failed_refs  # an abort is not a bad ref
        resolved = [c[-1] for c in git.calls if c[0] == "rev-parse"]
        assert "refs/heads/main" in resolved and "refs/heads/QS_3" not in resolved  # the walk stopped at T2
        assert f"exited {code}" in capsys.readouterr().err
