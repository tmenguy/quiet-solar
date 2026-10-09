"""QS-406 T9: the CI watcher (§8, AC 10)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from control_plane import activeloop, alerts, ciwatch, clock, tasks
from control_plane.runner import RunResult

from .conftest import Call, FakeRunner, insert_task, open_run, run_cli, sql


def _pr(
    state: str = "OPEN", head: str = "h1", rollup: str | None = "SUCCESS", failing: tuple[str, ...] = ()
) -> ciwatch.PrCi:
    return ciwatch.PrCi(state, head, rollup, failing, False)


def _tick(conn: Any, fake_clock: clock.FakeClock) -> None:
    ciwatch.ci_watch_hook(conn, fake_clock)


def _ci(path: Path, task: str) -> tuple[Any, Any]:
    row = sql(path, "SELECT ci_state, ci_sha FROM tasks WHERE id = ?", [task])[0]
    return row[0], row[1]


def _red(path: Path) -> list[tuple[str, str]]:
    return [(r[0], r[1]) for r in sql(path, "SELECT subject, cleared_at FROM alerts WHERE kind = 'ci_red' ORDER BY id")]


@pytest.fixture
def watched(migrated: Path) -> str:
    r1, _ = open_run()
    insert_task(migrated, "T1", r1, is_deliverable=1, pr_number=7)
    return r1


# --------------------------------------------------------------------------- the real GitHub on a FakeRunner


def _graphql(
    prs: dict[int, dict[str, Any] | None], *, remaining: int = 4000, errors: list[Any] | None = None, code: int = 0
):
    """A FakeRunner response that answers every alias in the query from ``prs`` (absent → a NOT_FOUND null)."""

    def respond(call: Call) -> RunResult:
        q = call.argv[-1]
        numbers = [int(n) for n in re.findall(r"p(\d+): pullRequest", q)]
        repo: dict[str, Any] = {}
        errs = list(errors or [])
        for n in numbers:
            node = prs.get(n)
            repo[f"p{n}"] = node
            if node is None:
                errs.append({"type": "NOT_FOUND", "path": ["repository", f"p{n}"]})
        doc: dict[str, Any] = {
            "data": {"rateLimit": {"remaining": remaining, "resetAt": "2026-10-03T13:00:00Z"}, "repository": repo}
        }
        if errs:
            doc["errors"] = errs
        return RunResult(1 if errs else code, json.dumps(doc), "")

    return respond


def _node(
    rollup: str | None, head: str = "h1", contexts: list[dict[str, Any]] | None = None, more: bool = False
) -> dict[str, Any]:
    commit = (
        None
        if rollup is None
        else {"state": rollup, "contexts": {"pageInfo": {"hasNextPage": more}, "nodes": contexts or []}}
    )
    return {"state": "OPEN", "headRefOid": head, "commits": {"nodes": [{"commit": {"statusCheckRollup": commit}}]}}


@pytest.fixture
def real_github(fake_runner: FakeRunner, fake_main: Path, monkeypatch) -> ciwatch.GitHub:
    fake_runner.on(("remote", "get-url", "origin"), "https://github.com/o/r.git\n")
    gh = ciwatch.GitHub(fake_runner, fake_main)
    _use_github(monkeypatch, gh)
    return gh


def _use_github(monkeypatch: pytest.MonkeyPatch, gh: ciwatch.GitHub) -> None:
    seams = activeloop.seams()
    monkeypatch.setattr(
        activeloop, "make_seams", lambda: activeloop.Seams(seams.runner, seams.probe, seams.claude, seams.main, gh)
    )
    activeloop._reset_for_tests()


class TestRealGitHub:
    def test_thirty_prs_take_two_calls(self, conn, migrated, real_github, fake_runner, fake_clock) -> None:
        r1, _ = open_run()
        for n in range(1, 31):
            insert_task(migrated, f"T{n}", r1, is_deliverable=1, pr_number=n)
        fake_runner.on(("gh", "api", "graphql"), _graphql({n: _node("SUCCESS") for n in range(1, 31)}))
        _tick(conn, fake_clock)
        calls = fake_runner.matching("gh", "api", "graphql")
        assert len(calls) == 2 and all(c.timeout == 20 for c in calls)
        assert 'repository(owner: "o", name: "r")' in calls[0].argv[-1]
        assert _ci(migrated, "T30") == ("green", "h1")

    def test_a_not_found_alias_never_blocks_the_others(
        self, conn, migrated, real_github, fake_runner, fake_clock
    ) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1, is_deliverable=1, pr_number=6)
        insert_task(migrated, "T2", r1, is_deliverable=1, pr_number=7)
        fake_runner.on(("gh", "api", "graphql"), _graphql({6: _node("FAILURE")}))  # p7 is null + NOT_FOUND, exit 1
        _tick(conn, fake_clock)
        assert _ci(migrated, "T1") == ("red", "h1") and _ci(migrated, "T2") == (None, None)

    @pytest.mark.parametrize(
        "result", [RunResult(127, "", "gh: not found"), RunResult(1, "not json", ""), RunResult(0, "{}", "")]
    )
    def test_failures_back_off_and_write_nothing(
        self, conn, migrated, watched, real_github, fake_runner, fake_clock, result
    ) -> None:
        fake_runner.on(("gh", "api", "graphql"), result)
        waits = []
        for _ in range(6):
            _tick(conn, fake_clock)
            w = ciwatch._watcher
            waits.append((w.next_at - fake_clock.now()).total_seconds())
            fake_clock.advance(waits[-1])
        assert waits == [60, 120, 240, 480, 900, 900]
        assert _ci(migrated, "T1") == (None, None)

    def test_a_failure_in_batch_2_writes_nothing_and_clears_no_red(
        self, conn, migrated, real_github, fake_runner, fake_clock
    ) -> None:
        r1, _ = open_run()
        for n in range(1, 31):
            insert_task(migrated, f"T{n}", r1, is_deliverable=1, pr_number=n)
        sql(migrated, "UPDATE tasks SET ci_state = 'red', ci_sha = 'h0' WHERE id = 'T1'")
        _tick(conn, fake_clock)  # the first poll fails (unscripted gh): ci_red is derived from the stored row
        assert _red(migrated) == [("T1@h0", None)]
        good = _graphql({n: _node("SUCCESS") for n in range(1, 31)})
        state = {"n": 0}

        def second_fails(call: Call) -> RunResult:
            state["n"] += 1
            return good(call) if state["n"] == 1 else RunResult(1, "", "HTTP 502")

        fake_runner.on(("gh", "api", "graphql"), second_fails)
        fake_clock.advance(ciwatch.CI_BACKOFF_MAX_S)
        _tick(conn, fake_clock)
        assert state["n"] == 2 and _ci(migrated, "T2") == (None, None) and _ci(migrated, "T1") == ("red", "h0")
        assert _red(migrated) == [("T1@h0", None)]

    def test_failing_checks_and_truncation(self, conn, migrated, watched, real_github, fake_runner, fake_clock) -> None:
        contexts = [
            {"__typename": "CheckRun", "name": "tests", "conclusion": "FAILURE"},
            {"__typename": "CheckRun", "name": "lint", "conclusion": "SUCCESS"},
            {"__typename": "CheckRun", "name": "slow", "conclusion": "TIMED_OUT"},
            {"__typename": "StatusContext", "context": "ci/legacy", "state": "ERROR"},
            {"__typename": "StatusContext", "context": "ok", "state": "SUCCESS"},
        ]
        fake_runner.on(("gh", "api", "graphql"), _graphql({7: _node("FAILURE", contexts=contexts, more=True)}))
        _tick(conn, fake_clock)
        payload = json.loads(sql(migrated, "SELECT payload FROM alerts WHERE kind = 'ci_red'")[0][0])
        assert payload == {
            "task_id": "T1",
            "pr_number": 7,
            "sha": "h1",
            "failing": ["tests", "slow", "ci/legacy"],
            "truncated": True,
            "severity": "must-fix",
        }
        msg = json.loads(sql(migrated, "SELECT payload FROM messages WHERE kind = 'ci_red'")[0][0])
        assert msg["severity"] == "must-fix"

    def test_a_pr_with_no_commit(self, real_github, fake_runner) -> None:
        node = {"state": "OPEN", "headRefOid": "h", "commits": {"nodes": []}}
        fake_runner.on(("gh", "api", "graphql"), _graphql({7: node}))
        prs, rate = real_github.prs([7])
        assert prs[7] == ciwatch.PrCi("OPEN", "h", None, (), False) and rate.remaining == 4000

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/o/r.git",
            "https://github.com/o/r",
            "git@github.com:o/r.git",
            "ssh://git@github.com/o/r.git",
        ],
    )
    def test_remote_forms(self, url: str) -> None:
        assert ciwatch.parse_remote(url + "\n") == ("o", "r")

    def test_an_unparseable_remote_is_logged_once(
        self, conn, migrated, watched, fake_runner, fake_main, monkeypatch, fake_clock, capsys
    ) -> None:
        fake_runner.on(("remote", "get-url", "origin"), "https://gitlab.com/o/r.git\n")
        gh = ciwatch.GitHub(fake_runner, fake_main)
        for _ in range(3):
            with pytest.raises(ciwatch.CiFailure):
                gh.prs([7])
        assert len(fake_runner.matching("remote", "get-url")) == 1
        _use_github(monkeypatch, gh)
        _tick(conn, fake_clock)
        assert capsys.readouterr().err.count("not a GitHub remote") == 1

    def test_a_failing_remote_command_is_retried(self, fake_runner, fake_main) -> None:
        fake_runner.on(("remote", "get-url", "origin"), RunResult(128, "", "no remote"))
        gh = ciwatch.GitHub(fake_runner, fake_main)
        for _ in range(2):
            with pytest.raises(ciwatch.CiFailure) as exc:
                gh.repo()
            assert not exc.value.quiet
        assert len(fake_runner.matching("remote", "get-url")) == 2


# --------------------------------------------------------------------------- the schedule and the mapping (FakeGitHub)


class TestSchedule:
    def _wait(self, fake_clock: clock.FakeClock) -> float:
        return (ciwatch._watcher.next_at - fake_clock.now()).total_seconds()

    @pytest.mark.parametrize(("rollup", "expected"), [("PENDING", 30), ("EXPECTED", 30), ("SUCCESS", 300), (None, 300)])
    def test_fast_while_pending_slow_when_settled(
        self, conn, migrated, watched, fake_github, fake_clock, rollup, expected
    ) -> None:
        fake_github.prs_by_number[7] = _pr(rollup=rollup)
        _tick(conn, fake_clock)  # the first poll sees a moved head (no ci_sha yet): fast
        assert self._wait(fake_clock) == 30
        fake_clock.advance(30)
        _tick(conn, fake_clock)
        assert self._wait(fake_clock) == expected

    def test_not_due_means_no_call(self, conn, migrated, watched, fake_github, fake_clock) -> None:
        fake_github.prs_by_number[7] = _pr()
        _tick(conn, fake_clock)
        fake_clock.advance(10)
        _tick(conn, fake_clock)
        assert len(fake_github.calls) == 1

    def test_nothing_watched_polls_nothing(self, conn, migrated, fake_github, fake_clock) -> None:
        _tick(conn, fake_clock)
        assert fake_github.calls == [] and self._wait(fake_clock) == 300

    def test_a_low_rate_waits_for_the_reset(self, conn, migrated, watched, fake_github, fake_clock) -> None:
        fake_github.prs_by_number[7] = _pr(rollup="PENDING")
        fake_github.rate = ciwatch.Rate(
            150, clock.iso(fake_clock.now() + __import__("datetime").timedelta(seconds=1200))
        )
        _tick(conn, fake_clock)
        assert self._wait(fake_clock) == 1200
        fake_github.rate = ciwatch.Rate(150, "garbage")
        fake_clock.advance(1200)
        _tick(conn, fake_clock)
        assert self._wait(fake_clock) == 300

    def test_writes_only_on_a_change(self, conn, migrated, watched, fake_github, fake_clock, monkeypatch) -> None:
        writes: list[dict[str, Any]] = []
        real = tasks.update_fields
        monkeypatch.setattr(tasks, "update_fields", lambda c, k, t, f: writes.append(f) or real(c, k, t, f))
        fake_github.prs_by_number[7] = _pr()
        for _ in range(3):
            _tick(conn, fake_clock)
            fake_clock.advance(300)
        assert writes == [{"ci_state": "green", "ci_sha": "h1"}]

    def test_red_once_per_sha_then_again(self, conn, migrated, watched, fake_github, fake_clock) -> None:
        fake_github.prs_by_number[7] = _pr(rollup="FAILURE", head="s1", failing=("tests",))
        _tick(conn, fake_clock)
        fake_clock.advance(300)
        _tick(conn, fake_clock)
        assert _red(migrated) == [("T1@s1", None)]
        assert sql(migrated, "SELECT count(*) FROM messages WHERE kind = 'ci_red'")[0][0] == 1
        fake_github.prs_by_number[7] = _pr(rollup="FAILURE", head="s2")
        fake_clock.advance(300)
        _tick(conn, fake_clock)
        assert [s for s, cleared in _red(migrated) if cleared is None] == ["T1@s2"]
        assert sql(migrated, "SELECT count(*) FROM messages WHERE kind = 'ci_red'")[0][0] == 2

    def test_items_terminal_tasks_and_closed_runs_are_skipped(self, conn, migrated, fake_github, fake_clock) -> None:
        r1, _ = open_run("r1", "S-1")
        r2, token2 = open_run("r2", "S-2")
        insert_task(migrated, "T1", r1, is_deliverable=1, pr_number=1)
        insert_task(migrated, "T2", r1, is_deliverable=0, deliverable_id="T1", item_k=1, pr_number=2)
        insert_task(migrated, "T3", r1, is_deliverable=1, pr_number=3, state="merged")
        insert_task(migrated, "T4", r2, is_deliverable=1, pr_number=4)
        run_cli("run", "close", "--token", token2)
        _tick(conn, fake_clock)
        assert fake_github.calls == [[1]]

    def test_a_closed_pr_clears_then_a_reopened_red_one_is_picked_up(
        self, conn, migrated, watched, fake_github, fake_clock
    ) -> None:
        fake_github.prs_by_number[7] = _pr(rollup="FAILURE", head="s1")
        _tick(conn, fake_clock)
        assert _red(migrated)[0][1] is None
        fake_github.prs_by_number[7] = _pr(state="CLOSED", head="s1", rollup="FAILURE")
        fake_clock.advance(300)
        _tick(conn, fake_clock)
        assert _ci(migrated, "T1") == (None, None) and _red(migrated)[0][1] is not None
        assert self._wait(fake_clock) == 300  # settled: the PR stays in the batch
        fake_github.prs_by_number[7] = _pr(rollup="FAILURE", head="s1")
        fake_clock.advance(300)
        _tick(conn, fake_clock)
        assert _ci(migrated, "T1") == ("red", "s1") and [c for _, c in _red(migrated)].count(None) == 1

    def test_a_new_daemon_polls_at_once_and_keeps_the_stored_red(
        self, conn, migrated, watched, fake_github, fake_clock
    ) -> None:
        fake_github.fail = ciwatch.CiFailure("down")
        sql(migrated, "UPDATE tasks SET ci_state = 'red', ci_sha = 's9' WHERE id = 'T1'")
        _tick(conn, fake_clock)
        payload = json.loads(sql(migrated, "SELECT payload FROM alerts WHERE kind = 'ci_red'")[0][0])
        assert payload["sha"] == "s9" and payload["failing"] == [] and len(fake_github.calls) == 1
        activeloop._reset_for_tests()
        _tick(conn, fake_clock)
        assert len(fake_github.calls) == 2


def test_the_ci_state_mapping() -> None:
    assert ciwatch.ci_state(_pr(rollup="ERROR")) == ("red", "h1")
    assert ciwatch.ci_state(_pr(state="MERGED")) == (None, None)
    assert ciwatch.ci_state(_pr(rollup="PENDING")) == ("pending", "h1")


def test_kinds() -> None:
    assert alerts.severity(alerts.CI_RED) == "must-fix"
