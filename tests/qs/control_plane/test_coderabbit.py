"""#425 D7: the CodeRabbit tools — ``tool coderabbit-trigger`` asks for a head's review, ``tool coderabbit`` ingests it.

Fixtures (``fixtures/coderabbit/``): trimmed captures from PR #430 (reviews ``5598133439`` / ``5601237998``, their
inline comments, the head's ``CodeRabbit`` statuses), a PR #432 comment (the ``·`` header form), the "Review rate
limited" status seen on PR #420, and hand-made cases.
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

    def gh(self, reviews: Any = None, comments: Any = None, statuses: Any = None, head: str = H) -> None:
        for argv, data in ((reviews_argv(), reviews), (comments_argv(), comments), (statuses_argv(head), statuses)):
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
