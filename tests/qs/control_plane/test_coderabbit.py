"""#425 D7: the CodeRabbit tools — ``tool coderabbit-trigger`` asks for a head's review, ``tool coderabbit`` ingests it.

Fixtures (``fixtures/coderabbit/``): trimmed captures from PR #430 (reviews ``5598133439`` / ``5601237998``, their
inline comments, the head's ``CodeRabbit`` statuses), a PR #432 comment (the ``·`` header form), the "Review rate
limited" status seen on PR #420, the ``CodeRabbit`` check runs of PR #435 (``check-runs-completed.json``: head
``01fc582e``, "Finished"; ``check-runs-skipped.json``: head ``bed28561``, "Review skipped" — CodeRabbit reports
through a check run since 2026-10-27), and hand-made cases (``check-runs-in-progress.json`` among them).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from control_plane import clock, coderabbit, db, node_policy, selfcheck, tools
from control_plane.runner import RunResult

from .conftest import FakeRunner, insert_node, insert_task, open_run, run_cli, sql

FIX = Path(__file__).parent / "fixtures" / "coderabbit"
H = "0123456789abcdef0123456789abcdef01234567"  # the hand-made head
H430 = "fcb132c1d5d039b4d8d482e6e969f02be4cd2159"  # PR #430's head reviewed by 5598133439


def fixture(name: str) -> Any:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def reviews_argv(pr: int = 7) -> tuple[str, ...]:
    return ("gh", "api", "--paginate", "--slurp", f"repos/{{owner}}/{{repo}}/pulls/{pr}/reviews")


def comments_argv(pr: int = 7) -> tuple[str, ...]:
    return ("gh", "api", "--paginate", "--slurp", f"repos/{{owner}}/{{repo}}/pulls/{pr}/comments")


def statuses_argv(head: str = H) -> tuple[str, ...]:
    return ("gh", "api", "--paginate", "--slurp", f"repos/{{owner}}/{{repo}}/commits/{head}/statuses")


def checks_argv(head: str = H) -> tuple[str, ...]:
    path = f"repos/{{owner}}/{{repo}}/commits/{head}/check-runs?check_name=CodeRabbit"
    return ("gh", "api", "--paginate", "--slurp", path)


NO_CHECKS = [{"total_count": 0, "check_runs": []}]


def check_run(
    status: str = "completed",
    conclusion: str | None = "success",
    at: str = "2026-10-27T15:37:32Z",
    *,
    run_id: int = 1,
    title: str = "Finished",
    summary: str = "Result: Success - Review completed",
    slug: str = "coderabbitai",
) -> dict[str, Any]:
    """A hand-made ``CodeRabbit`` check run; ``at`` is its ``completed_at`` (``started_at`` when not completed)."""
    done = status == "completed"
    return {
        "id": run_id,
        "name": "CodeRabbit",
        "status": status,
        "conclusion": conclusion,
        "started_at": None if done else at,
        "completed_at": at if done else None,
        "output": {"title": title, "summary": summary},
        "app": {"slug": slug},
    }


COMMENT = ("gh", "pr", "comment", "7", "--body", "@coderabbitai review")


@dataclass
class C:
    db: Path
    run: str
    token: str
    node: str
    runner: FakeRunner
    clock: clock.FakeClock
    tmp: Path
    n: int = 0

    def tool(self, name: str, head: str = H, *, key: str | None = None, token: str | None = None) -> tuple[int, Any]:
        self.n += 1
        f = self.tmp / f"args-{self.n}.json"
        f.write_text(json.dumps({"head": head}))
        key = key or f"task:T1:{name}:{head}"
        return run_cli("tool", name, "--task", "T1", "--key", key, "--args-file", str(f), "--token", token or self.node)

    def round(self, head: str = H) -> int:
        code, out = run_cli("round", "start", "--task", "T1", "--phase", "build", "--head", head, "--token", self.node)
        assert code == 0, out
        return int(out["round"])

    def gh(
        self, reviews: Any = None, comments: Any = None, statuses: Any = None, head: str = H, checks: Any = None
    ) -> None:
        """Register the answers; with ``statuses`` and no ``checks``, the head has no ``CodeRabbit`` check run."""
        if statuses is not None and checks is None:
            checks = NO_CHECKS
        pairs = ((reviews_argv(), reviews), (comments_argv(), comments), (statuses_argv(head), statuses))
        for argv, data in (*pairs, (checks_argv(head), checks)):
            if data is not None:
                self.runner.on(argv, json.dumps(data))

    def at(self, stamp: str, plus_s: float) -> None:
        """Set the clock to ``stamp`` + ``plus_s`` (the fixtures' stamps are later than FakeClock's start)."""
        target = datetime.fromisoformat(stamp) + timedelta(seconds=plus_s)
        self.clock.advance((target - self.clock.now()).total_seconds())

    def calls(self, name: str) -> list[dict[str, Any]]:
        return [dict(r) for r in sql(self.db, "SELECT * FROM tool_calls WHERE tool = ? ORDER BY rowid", [name])]

    def findings(self) -> list[dict[str, Any]]:
        return [dict(r) for r in sql(self.db, "SELECT * FROM findings ORDER BY id")]


@pytest.fixture
def c(migrated: Path, tmp_path: Path, fake_runner: FakeRunner, fake_clock: clock.FakeClock) -> C:
    run_id, token = open_run()
    insert_task(migrated, "T1", run_id, is_deliverable=1, pr_number=7, issue_number=11, branch="QS_11")
    node = insert_node(migrated, "N1", run_id, "T1")
    fake_runner.calls.clear()
    return C(migrated, run_id, token, node, fake_runner, fake_clock, tmp_path)


def _completed_at() -> str:
    return str(fixture("statuses-completed.json")[0][0]["created_at"])


def _check_completed_at() -> str:
    return str(fixture("check-runs-completed.json")[0]["check_runs"][0]["completed_at"])


# --------------------------------------------------------------------------- the registry


def test_register_coderabbit_tools_is_idempotent_and_selfchecked() -> None:
    coderabbit.register_coderabbit_tools()
    coderabbit.register_coderabbit_tools()
    assert {"coderabbit", "coderabbit-trigger"} <= set(tools.REGISTRY)
    assert {"coderabbit", "coderabbit-trigger"} <= selfcheck._expected_tools()
    for name in ("coderabbit", "coderabbit-trigger"):
        spec = tools.REGISTRY[name]
        assert spec.token_kinds == frozenset({"run", "node"}) and spec.task_bound


def test_the_node_policy_values() -> None:
    """The ceilings the maintainer confirmed at #425's FINALIZE (2026-10-27)."""
    assert (node_policy.CR_MAX_REVIEWS, node_policy.CR_WAIT_S, node_policy.CR_MAX_FINDINGS) == (5, 900, 30)
    assert (node_policy.CR_SETTLE_S, node_policy.ROTATE_ROUNDS, node_policy.CI_WAIT_MAX_S) == (120, 4, 2700)


# --------------------------------------------------------------------------- refusals before the claim


@pytest.mark.parametrize("name", ["coderabbit", "coderabbit-trigger"])
@pytest.mark.parametrize(
    ("prep", "head", "error"),
    [
        ("UPDATE tasks SET state = 'merged'", H, "INVALID_STATE"),
        ("UPDATE tasks SET deliverable_id = 'T0', is_deliverable = 0", H, "INVALID_STATE"),
        ("UPDATE tasks SET pr_number = NULL", H, "INVALID_STATE"),
        (None, "abc", "USAGE"),
        (None, "g" * 40, "USAGE"),
    ],
)
def test_refused_before_the_claim(c: C, name: str, prep: str | None, head: str, error: str) -> None:
    if prep:
        if "deliverable_id" in prep:
            insert_task(c.db, "T0", c.run, is_deliverable=1)
        sql(c.db, prep)
    code, out = c.tool(name, head, key=f"task:T1:{name}:x")
    assert out["error"] == error, out
    assert c.calls(name) == [] and c.runner.calls == []


def test_a_non_string_head_is_usage(c: C) -> None:
    f = c.tmp / "args-int.json"
    f.write_text(json.dumps({"head": 7}))
    code, out = run_cli("tool", "coderabbit", "--task", "T1", "--key", "k", "--args-file", str(f), "--token", c.node)
    assert out["error"] == "USAGE" and c.calls("coderabbit") == []


# --------------------------------------------------------------------------- tool coderabbit-trigger


def test_trigger_posts_the_request_outside_the_transaction(c: C) -> None:
    c.gh(statuses=fixture("statuses-skipped.json"))

    def comment(call: Any) -> RunResult:
        conn = db.connect(c.db)
        try:  # outside any transaction: another writer can take the lock while the comment is posted
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ROLLBACK")
        finally:
            conn.close()
        return RunResult(0, "https://github.com/o/r/pull/7#issuecomment-1\n", "")

    c.runner.on(COMMENT, comment)
    code, out = c.tool("coderabbit-trigger")
    assert code == 0 and out["result"] == {"status": "triggered"}, out
    [call] = c.runner.matching(*COMMENT)
    assert call.timeout == coderabbit.GH_TIMEOUT_S and call.cwd is not None
    code, again = c.tool("coderabbit-trigger")
    assert again["replayed"] is True and len(c.runner.matching(*COMMENT)) == 1


@pytest.mark.parametrize("statuses", ["statuses-completed.json", "statuses-in-progress.json"])
def test_trigger_answers_already_when_reviewed_or_in_progress(c: C, statuses: str) -> None:
    c.gh(statuses=fixture(statuses))
    code, out = c.tool("coderabbit-trigger")
    assert code == 0 and out["result"] == {"status": "already"}
    assert c.runner.matching(*COMMENT) == []


@pytest.mark.parametrize("statuses", ["statuses-skipped.json", "statuses-rate-limited.json", [[]]])
def test_trigger_requests_a_review_that_was_not_requested(c: C, statuses: Any) -> None:
    c.gh(statuses=fixture(statuses) if isinstance(statuses, str) else statuses)
    code, out = c.tool("coderabbit-trigger")
    assert out["result"] == {"status": "triggered"} and len(c.runner.matching(*COMMENT)) == 1


def test_trigger_ceiling_is_checked_first_and_counts_triggered_only(c: C) -> None:
    heads = [f"{i:040x}" for i in range(1, node_policy.CR_MAX_REVIEWS + 3)]
    for head in heads[:2]:  # two `already` answers: recorded, not counted
        c.gh(statuses=fixture("statuses-completed.json"), head=head)
        assert c.tool("coderabbit-trigger", head)[1]["result"]["status"] == "already"
    for head in heads[2 : 2 + node_policy.CR_MAX_REVIEWS]:
        c.gh(statuses=[[]], head=head)
        assert c.tool("coderabbit-trigger", head)[1]["result"]["status"] == "triggered"
    c.runner.calls.clear()
    code, out = c.tool("coderabbit-trigger", "e" * 40)
    assert code == 0 and out["result"] == {"status": "ceiling"}
    assert c.runner.calls == []  # a local check: no `gh` call at all


@pytest.mark.parametrize(
    "result", [RunResult(1, "", "HTTP 502"), RunResult(0, "not json", ""), RunResult(0, '{"a": 1}', "")]
)
def test_trigger_unknown_state_is_busy_and_releases(c: C, result: RunResult) -> None:
    c.runner.on(statuses_argv(), result)
    code, out = c.tool("coderabbit-trigger")
    assert code == 6 and out["error"] == "BUSY"
    assert c.calls("coderabbit-trigger") == [] and c.runner.matching(*COMMENT) == []


def test_trigger_failed_comment_is_busy(c: C) -> None:
    c.gh(statuses=[[]])
    c.runner.on(COMMENT, RunResult(1, "", "HTTP 403"))
    code, out = c.tool("coderabbit-trigger")
    assert code == 6 and out["error"] == "BUSY" and c.calls("coderabbit-trigger") == []
    c.runner.on(COMMENT, RunResult(0, "", ""))
    assert c.tool("coderabbit-trigger")[1]["result"] == {"status": "triggered"}  # the same key, later


def test_trigger_with_a_run_token(c: C) -> None:
    c.gh(statuses=[[]])
    assert c.tool("coderabbit-trigger", token=c.token)[1]["result"] == {"status": "triggered"}


# --------------------------------------------------------------------------- the head's state from its check run (M1)


@pytest.mark.parametrize(
    "checks",
    [
        "check-runs-completed.json",
        "check-runs-in-progress.json",
        [{"total_count": 1, "check_runs": [check_run("queued", None)]}],
    ],
)
def test_trigger_answers_already_on_a_check_run_reviewed_or_in_progress(c: C, checks: Any) -> None:
    c.gh(statuses=[[]], checks=fixture(checks) if isinstance(checks, str) else checks)
    code, out = c.tool("coderabbit-trigger")
    assert code == 0 and out["result"] == {"status": "already"}, out
    assert c.runner.matching(*COMMENT) == []
    [call] = c.runner.matching(*checks_argv())
    assert call.timeout == coderabbit.GH_TIMEOUT_S


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(None, id="live-review-skipped"),
        pytest.param(check_run(title="Review rate limited"), id="rate-limited-title"),
        pytest.param(check_run(title="Done", summary="Result: Skipped - Review skipped\n| a | Completed |"), id="skip"),
        pytest.param(check_run(conclusion="failure"), id="a-failed-run"),
        pytest.param(check_run(conclusion="cancelled", title="Cancelled"), id="a-cancelled-run"),
        pytest.param(check_run(slug="someone-else"), id="another-apps-run"),
        pytest.param({**check_run(), "name": "CodeRabbit lint"}, id="another-name"),
        pytest.param({**check_run(), "app": None}, id="a-null-app"),
        pytest.param({**check_run(), "status": None}, id="an-odd-status"),
        pytest.param({**check_run(), "output": None}, id="a-null-output-is-not-a-review"),
        pytest.param("not a run", id="not-a-dict"),
    ],
)
def test_trigger_requests_a_review_on_a_check_run_that_is_not_a_review(c: C, run: Any) -> None:
    checks = fixture("check-runs-skipped.json") if run is None else [{"total_count": 1, "check_runs": [run]}]
    c.gh(statuses=[[]], checks=checks)
    code, out = c.tool("coderabbit-trigger")
    assert out["result"] == {"status": "triggered"}, out


def test_a_success_whose_steps_say_skipped_is_still_a_review(c: C) -> None:
    """Only the title and the summary's first line count: a skipped step in the table is not a skipped review."""
    run = check_run(summary="Result: Success - Review completed\n| Run analysis checks | Skipped | 0s |")
    c.gh(statuses=[[]], checks=[{"total_count": 1, "check_runs": [run]}])
    assert c.tool("coderabbit-trigger")[1]["result"] == {"status": "already"}


@pytest.mark.parametrize(
    ("statuses", "checks", "expected"),
    [
        pytest.param("statuses-skipped.json", "check-runs-completed.json", "already", id="newer-check-run-completed"),
        pytest.param("statuses-completed.json", "check-runs-skipped.json", "triggered", id="newer-check-run-skipped"),
        pytest.param(
            [
                [
                    {
                        "context": "CodeRabbit",
                        "state": "success",
                        "description": "Review skipped: x",
                        "created_at": "2026-10-28T00:00:00Z",
                    }
                ]
            ],
            "check-runs-completed.json",
            "triggered",
            id="newer-status-skipped",
        ),
        pytest.param(
            [
                [
                    {
                        "context": "CodeRabbit",
                        "state": "pending",
                        "description": "Review in progress",
                        "created_at": "2026-10-28T00:00:00Z",
                    }
                ]
            ],
            "check-runs-skipped.json",
            "already",
            id="newer-status-in-progress",
        ),
    ],
)
def test_the_newer_of_status_and_check_run_wins(c: C, statuses: Any, checks: str, expected: str) -> None:
    c.gh(statuses=fixture(statuses) if isinstance(statuses, str) else statuses, checks=fixture(checks))
    assert c.tool("coderabbit-trigger")[1]["result"] == {"status": expected}


@pytest.mark.parametrize(
    ("runs", "expected"),
    [
        pytest.param([check_run(run_id=2), check_run("in_progress", None, run_id=1)], "already", id="same-stamp-id-2"),
        pytest.param(
            [check_run(conclusion="skipped", title="Review skipped", run_id=2), check_run(run_id=1)],
            "triggered",
            id="same-stamp-the-higher-id-wins",
        ),
        pytest.param(
            [check_run(conclusion="skipped", title="Review skipped", at="soon", run_id=9), check_run(run_id=1)],
            "already",
            id="an-unparseable-stamp-loses",
        ),
        pytest.param(
            [{**check_run(conclusion="skipped", title="Review skipped"), "id": None}, check_run(run_id=0)],
            "already",
            id="a-missing-id-loses-a-tie",
        ),
    ],
)
def test_the_latest_check_run_is_chosen_deterministically(c: C, runs: list[Any], expected: str) -> None:
    for order in (runs, list(reversed(runs))):
        c.runner.calls.clear()
        c.gh(statuses=[[]], checks=[{"total_count": len(order), "check_runs": order}])
        key = f"task:T1:coderabbit-trigger:{len(c.runner.rules)}"
        assert c.tool("coderabbit-trigger", key=key)[1]["result"] == {"status": expected}


def test_a_check_run_ties_with_a_status_and_wins(c: C) -> None:
    st = [
        [
            {
                "context": "CodeRabbit",
                "state": "success",
                "description": "Review skipped: x",
                "created_at": "2026-10-27T15:37:32Z",
            }
        ]
    ]
    c.gh(statuses=st, checks=[{"total_count": 1, "check_runs": [check_run()]}])
    assert c.tool("coderabbit-trigger")[1]["result"] == {"status": "already"}


def test_two_check_run_pages_are_flattened(c: C) -> None:
    pages = [*fixture("check-runs-skipped.json"), *fixture("check-runs-completed.json")]
    c.gh(statuses=[[]], checks=pages)
    assert c.tool("coderabbit-trigger")[1]["result"] == {"status": "triggered"}  # the skipped run is the newer


@pytest.mark.parametrize(
    "result",
    [
        RunResult(1, "", "HTTP 502"),
        RunResult(0, "not json", ""),
        RunResult(0, '{"total_count": 0, "check_runs": []}', ""),  # not slurped: not a list of pages
        RunResult(0, "[[]]", ""),
        RunResult(0, '[{"total_count": 1}]', ""),
        RunResult(0, '[{"check_runs": {}}]', ""),
    ],
)
def test_a_failed_or_unparseable_check_runs_answer_is_busy(c: C, result: RunResult) -> None:
    c.gh(statuses=fixture("statuses-skipped.json"))
    c.runner.on(checks_argv(), result)
    code, out = c.tool("coderabbit-trigger")
    assert code == 6 and out["error"] == "BUSY", out
    assert c.calls("coderabbit-trigger") == [] and c.runner.matching(*COMMENT) == []
    c.round()
    c.gh(reviews=[[]])
    code, out = c.tool("coderabbit")
    assert code == 6 and out["error"] == "BUSY" and c.calls("coderabbit") == []


def test_a_settled_completed_check_run_with_no_review_object_is_reviewed(c: C) -> None:
    c.round()
    c.gh(reviews=[[]], statuses=[[]], checks=fixture("check-runs-completed.json"))
    c.at(_check_completed_at(), node_policy.CR_SETTLE_S - 1)
    code, out = c.tool("coderabbit")
    assert code == 6 and out["error"] == "BUSY" and "not reviewed yet (completed)" in out["detail"]
    c.at(_check_completed_at(), node_policy.CR_SETTLE_S)
    code, out = c.tool("coderabbit")  # the same key replays and settles
    assert code == 0 and out["result"] == {
        "status": "reviewed",
        "review_ids": [],
        "ids": [],
        "dropped": 0,
        "body_comments": False,
    }
    assert c.findings() == []


@pytest.mark.parametrize("checks", ["check-runs-in-progress.json", "check-runs-skipped.json"])
def test_an_unfinished_or_skipped_check_run_with_no_review_object_is_busy(c: C, checks: str) -> None:
    c.round()
    c.gh(reviews=[[]], statuses=fixture("statuses-completed.json"), checks=fixture(checks))
    c.at(_check_completed_at(), 3600)
    code, out = c.tool("coderabbit")
    assert code == 6 and out["error"] == "BUSY" and "not reviewed yet" in out["detail"]


# --------------------------------------------------------------------------- tool coderabbit: not reviewed yet


def test_no_build_round_for_the_head_is_busy(c: C) -> None:
    code, out = c.tool("coderabbit")
    assert code == 6 and out["error"] == "BUSY" and "no build round" in out["detail"]
    assert c.calls("coderabbit") == [] and c.runner.calls == []


@pytest.mark.parametrize(
    "statuses", ["statuses-in-progress.json", "statuses-skipped.json", "statuses-rate-limited.json", [[]]]
)
def test_no_review_object_and_no_completed_status_is_busy(c: C, statuses: Any) -> None:
    c.round()
    c.gh(reviews=[[]], statuses=fixture(statuses) if isinstance(statuses, str) else statuses)
    c.at(_completed_at(), 3600)
    code, out = c.tool("coderabbit")
    assert code == 6 and out["error"] == "BUSY" and "not reviewed yet" in out["detail"]
    assert c.calls("coderabbit") == []
    c.gh(reviews=fixture("handmade-reviews.json"), comments=fixture("handmade-comments.json"))
    code, out = c.tool("coderabbit")  # the same key replays once the review is there
    assert code == 0 and out["result"]["status"] == "reviewed"


def test_a_fresh_completed_status_with_no_review_object_is_busy(c: C) -> None:
    c.round()
    c.gh(reviews=[[]], statuses=fixture("statuses-completed.json"))
    c.at(_completed_at(), node_policy.CR_SETTLE_S - 1)
    assert c.tool("coderabbit")[1]["error"] == "BUSY"


def test_a_settled_completed_status_with_no_review_object_is_reviewed(c: C) -> None:
    c.round()
    c.gh(reviews=fixture("handmade-reviews.json")[1:], statuses=fixture("statuses-completed.json"))  # another head's
    c.at(_completed_at(), node_policy.CR_SETTLE_S)
    code, out = c.tool("coderabbit")
    assert code == 0 and out["result"] == {
        "status": "reviewed",
        "review_ids": [],
        "ids": [],
        "dropped": 0,
        "body_comments": False,
    }
    assert c.runner.matching(*comments_argv()) == [] and c.findings() == []


def test_an_unparseable_created_at_counts_as_fresh(c: C) -> None:
    c.round()
    st = [[{"context": "CodeRabbit", "state": "success", "description": "Review completed", "created_at": "soon"}]]
    c.gh(reviews=[[]], statuses=st)
    c.at(_completed_at(), 3600)
    assert c.tool("coderabbit")[1]["error"] == "BUSY"


def test_a_naive_created_at_is_utc(c: C) -> None:
    c.round()
    st = [
        [
            {
                "context": "CodeRabbit",
                "state": "success",
                "description": "Review completed",
                "created_at": "2026-10-21T14:12:04",
            }
        ]
    ]
    c.gh(reviews=[[]], statuses=st)
    c.at(_completed_at(), node_policy.CR_SETTLE_S)
    assert c.tool("coderabbit")[1]["result"]["status"] == "reviewed"


def test_the_latest_coderabbit_status_wins_and_other_contexts_are_ignored(c: C) -> None:
    c.round()
    st = fixture("statuses-completed.json")
    st[0].append({"context": "ci/other", "state": "pending", "description": "x", "created_at": "2026-10-22T00:00:00Z"})
    st[0].append("not a status")
    c.gh(reviews=[[]], statuses=[list(reversed(st[0]))])
    c.at(_completed_at(), node_policy.CR_SETTLE_S)
    assert c.tool("coderabbit")[1]["result"]["status"] == "reviewed"


@pytest.mark.parametrize(
    ("which", "result"),
    [
        ("reviews", RunResult(1, "", "HTTP 502")),
        ("reviews", RunResult(0, "{", "")),
        ("reviews", RunResult(0, '{"a": 1}', "")),
        ("reviews", RunResult(0, '[{"a": 1}]', "")),
        ("comments", RunResult(1, "", "HTTP 502")),
        ("statuses", RunResult(0, "[[", "")),
    ],
)
def test_a_failed_or_unparseable_gh_answer_is_busy(c: C, which: str, result: RunResult) -> None:
    c.round()
    c.gh(reviews=[[]] if which == "statuses" else fixture("handmade-reviews.json"), statuses=[[]])
    argv = {"reviews": reviews_argv(), "comments": comments_argv(), "statuses": statuses_argv()}[which]
    c.runner.on(argv, result)
    code, out = c.tool("coderabbit")
    assert code == 6 and out["error"] == "BUSY" and c.calls("coderabbit") == [] and c.findings() == []


# --------------------------------------------------------------------------- tool coderabbit: ingesting


def test_the_pr430_review_is_ingested_before_review_completed(c: C) -> None:
    """The review object precedes the "Review completed" status: it is ingested whatever the status says."""
    c.round(H430)
    c.gh(
        reviews=fixture("pr430-reviews.json"),
        comments=fixture("pr430-comments.json"),
        statuses=fixture("statuses-in-progress.json"),
        head=H430,
    )
    code, out = c.tool("coderabbit", H430)
    assert code == 0, out
    res = out["result"]
    assert res["status"] == "reviewed" and res["review_ids"] == [5598133439]
    assert len(res["ids"]) == 3 and res["dropped"] == 0 and res["body_comments"] is False
    rows = c.findings()
    assert [r["file"] for r in rows] == [
        "docs/workflow/harness.md",
        "scripts/qs/control_plane/export.py",
        "scripts/qs/control_plane/reports.py",
    ]
    first = rows[0]
    assert (first["source"], first["reviewer"], first["phase"], first["round"]) == (
        "coderabbit",
        "coderabbitai",
        "build",
        1,
    )
    assert (first["severity"], first["category"]) == ("nice_to_have", "correctness")
    assert first["title"] == "Mark the real-node check as pending."
    assert (first["line_start"], first["line_end"]) == (420, 421)  # outdated: the `original_*` pair
    assert (rows[1]["line_start"], rows[1]["line_end"]) == (27, 27)
    assert "**🎯 Functional Correctness**" in first["body"]
    assert c.runner.matching(*statuses_argv(H430)) == []  # a review object needs no status read
    assert c.runner.matching(*checks_argv(H430)) == []


def test_the_handmade_mapping(c: C) -> None:
    c.round()
    c.gh(reviews=fixture("handmade-reviews.json"), comments=fixture("handmade-comments.json"))
    code, out = c.tool("coderabbit")
    assert code == 0, out
    res = out["result"]
    assert res["review_ids"] == [9001] and res["dropped"] == 1  # `../evil.py`
    by_file = {r["file"]: r for r in c.findings()}
    assert set(by_file) == {"a.py", "b.py", "s.py", "p.py", "c.py", "r.py", "d.py", "m.py", "q.py", "f.py"}
    want = {
        "a.py": ("must_fix", "correctness", "Guard the empty list.", 8, 10),
        "b.py": ("should_fix", "correctness", "Close the handle on error.", 5, 5),
        "s.py": ("nice_to_have", "security", "Quote the shell argument.", 2, 3),
        "p.py": ("nice_to_have", "performance", "Hoist the regex.", 30, 30),
        "c.py": ("should_fix", "other", "CodeRabbit: c.py:7", 7, 7),
        "r.py": ("should_fix", "edge-case", "Retry on a timeout.", 12, 12),
        "d.py": ("nice_to_have", "design", "CodeRabbit: d.py:3", 3, 3),
        "q.py": ("should_fix", "edge-case", "Pass the remaining budget.", 44, 50),
        "f.py": ("nice_to_have", "correctness", "CodeRabbit: f.py", None, None),  # a file-level comment
    }
    for path, (sev, cat, title, start, end) in want.items():
        row = by_file[path]
        assert (row["severity"], row["category"], row["title"], row["line_start"], row["line_end"]) == (
            sev,
            cat,
            title,
            start,
            end,
        ), path
    m = by_file["m.py"]["title"]
    assert len(m) == coderabbit.TITLE_MAX and "\x1f" not in m and m.startswith("Keep the very")
    assert [r["file"] for r in c.findings()][0] == "a.py"  # most severe first


def test_the_new_header_form_of_pr432(c: C) -> None:
    comment = fixture("pr432-comment.json")
    head = comment["commit_id"]
    c.round(head)
    review = {
        "id": comment["pull_request_review_id"],
        "user": {"login": "coderabbitai[bot]"},
        "commit_id": head,
        "body": "",
    }
    c.gh(reviews=[[review]], comments=[[comment]], head=head)
    code, out = c.tool("coderabbit", head)
    [row] = c.findings()
    assert (row["severity"], row["category"]) == ("nice_to_have", "edge-case")
    assert row["title"] == "Pass the remaining stop budget to `Claude.stop`."  # the first bold line outside <details>
    assert (row["line_start"], row["line_end"]) == (624, 650)


def test_a_review_with_no_integer_id_is_skipped(c: C) -> None:
    """S5: a bot review of the head with no (or an odd) ``id`` is skipped, never a crash; a ``bool`` is not an
    integer id (R2 polish: ``True == 1``)."""
    c.round()
    reviews = fixture("handmade-reviews.json")
    for bad in ({}, {"id": "9002"}, {"id": True}):
        reviews[0].append({**bad, "user": {"login": "coderabbitai[bot]"}, "commit_id": H, "body": ""})
    c.gh(reviews=reviews, comments=fixture("handmade-comments.json"))
    code, out = c.tool("coderabbit")
    assert code == 0 and out["result"]["review_ids"] == [9001], out


def test_only_integer_comment_ids_are_de_duplicated(c: C) -> None:
    """R2 polish: an id-less comment never swallows the next one; the same integer id twice is one finding."""
    c.round()
    base = fixture("handmade-comments.json")[0][1]  # b.py, a Major
    comments = [
        [
            {**base, "id": None, "path": "n1.py"},
            {**base, "id": None, "path": "n2.py"},
            {k: v for k, v in base.items() if k != "id"} | {"path": "n3.py"},
            {**base, "id": 77, "path": "dup.py"},
            {**base, "id": 77, "path": "dup.py"},
        ]
    ]
    c.gh(reviews=fixture("handmade-reviews.json"), comments=comments)
    code, out = c.tool("coderabbit")
    assert code == 0, out
    assert sorted(r["file"] for r in c.findings()) == ["dup.py", "n1.py", "n2.py", "n3.py"]


def test_odd_comment_ids_are_type_checked_before_membership(c: C) -> None:
    """R3 polish: an unhashable ``id`` or ``pull_request_review_id`` is never a ``TypeError``, and a ``bool``
    ``pull_request_review_id`` never matches review id 1 (``True == 1``)."""
    c.round()
    reviews = fixture("handmade-reviews.json")
    reviews[0].append({"id": 1, "user": {"login": "coderabbitai[bot]"}, "commit_id": H, "body": ""})
    base = fixture("handmade-comments.json")[0][1]  # b.py, a Major of review 9001
    comments = [
        [
            {**base, "id": [1], "path": "list-id.py"},  # kept: an unhashable id just never de-duplicates
            {**base, "id": 81, "pull_request_review_id": {}, "path": "dict-review.py"},
            {**base, "id": 82, "pull_request_review_id": True, "path": "bool-review.py"},
            {**base, "id": 83, "pull_request_review_id": 1, "path": "review-1.py"},
        ]
    ]
    c.gh(reviews=reviews, comments=comments)
    code, out = c.tool("coderabbit")
    assert code == 0, out
    assert out["result"]["review_ids"] == [9001, 1]
    assert sorted(r["file"] for r in c.findings()) == ["list-id.py", "review-1.py"]


def test_two_pages_are_flattened_and_several_reviews_merged(c: C) -> None:
    c.round()
    reviews = fixture("handmade-reviews.json")
    second = {
        "id": 9003,
        "user": {"login": "coderabbitai[bot]"},
        "commit_id": H.upper(),
        "body": "**Actionable comments posted: 1**",
    }
    reviews[1].append(second)
    comments = fixture("handmade-comments.json")
    comments.append([{**comments[0][1], "id": 40, "pull_request_review_id": 9003, "path": "z.py"}])
    c.gh(reviews=reviews, comments=comments)
    res = c.tool("coderabbit")[1]["result"]
    assert res["review_ids"] == [9001, 9003]
    assert "z.py" in {r["file"] for r in c.findings()}


def test_the_ceiling_on_findings_keeps_the_most_severe(c: C, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coderabbit, "CR_MAX_FINDINGS", 2)
    c.round()
    c.gh(reviews=fixture("handmade-reviews.json"), comments=fixture("handmade-comments.json"))
    res = c.tool("coderabbit")[1]["result"]
    assert len(res["ids"]) == 2 and res["dropped"] == 1 + 8
    assert [r["severity"] for r in c.findings()] == ["must_fix", "should_fix"]


def test_an_empty_batch_is_recorded(c: C) -> None:
    c.round()
    c.gh(reviews=fixture("handmade-reviews.json"), comments=[[], []])
    res = c.tool("coderabbit")[1]["result"]
    assert res == {"status": "reviewed", "review_ids": [9001], "ids": [], "dropped": 0, "body_comments": False}
    assert len(c.calls("coderabbit")) == 1 and c.calls("coderabbit")[0]["state"] == "succeeded"


@pytest.mark.parametrize(("body", "expected"), [("review-body-sections.txt", True), (None, False)])
def test_body_comments(c: C, body: str | None, expected: bool) -> None:
    c.round()
    reviews = fixture("handmade-reviews.json")
    reviews[0][0]["body"] = (
        (FIX / body).read_text(encoding="utf-8") if body else "<details><summary>ℹ️ Review info</summary></details>"
    )
    c.gh(reviews=reviews, comments=[[]])
    assert c.tool("coderabbit")[1]["result"]["body_comments"] is expected


def test_the_ingest_replays_and_is_idempotent(c: C) -> None:
    c.round()
    c.gh(reviews=fixture("handmade-reviews.json"), comments=fixture("handmade-comments.json"))
    first = c.tool("coderabbit")[1]
    n = len(c.findings())
    again = c.tool("coderabbit")[1]
    assert again["replayed"] is True and again["result"] == first["result"] and len(c.findings()) == n
    other = c.tool("coderabbit", key="task:T1:coderabbit:other-key")[1]  # a new key: the ledger's replay key holds
    assert other["result"]["ids"] == first["result"]["ids"] and len(c.findings()) == n


def test_findings_land_on_the_heads_latest_round(c: C) -> None:
    assert c.round() == 1
    assert c.round("f" * 40) == 2
    assert c.round() == 3
    c.gh(reviews=fixture("handmade-reviews.json"), comments=fixture("handmade-comments.json"))
    c.tool("coderabbit")
    assert {r["round"] for r in c.findings()} == {3}


def test_the_ingest_with_a_run_token_records_the_run_actor(c: C) -> None:
    c.round()
    c.gh(reviews=fixture("handmade-reviews.json"), comments=fixture("handmade-comments.json"))
    assert c.tool("coderabbit", token=c.token)[0] == 0
    assert {r["actor"] for r in c.findings()} == {"orchestrator"}


# --------------------------------------------------------------------------- the parsing helpers


@pytest.mark.parametrize(
    ("body", "severity", "category"),
    [
        ("**🎯 Functional Correctness** | **🟡 Minor**", "nice_to_have", "correctness"),
        ("_⚠️ Potential issue_ | _🔴 Critical_", "must_fix", "correctness"),
        ("**🔗 Data Integrity & Integration** | **🟠 Major**", "should_fix", "correctness"),
        ("**🔒 Security & Privacy** | **🛡️ Detected with Advanced Tier** | **🟡 Minor**", "nice_to_have", "security"),
        ("**🛡️ Detected with Advanced Tier** | **🔒 Security & Privacy** | **🔵 Trivial**", "nice_to_have", "security"),
        ("**🧪 Something New** | **🟠 Major**", "should_fix", "other"),
        ("no header at all", "should_fix", "other"),
        ("**Only a bold title.**", "should_fix", "other"),
    ],
)
def test_the_header_mapping(body: str, severity: str, category: str) -> None:
    assert coderabbit.header(body) == (severity, category)
