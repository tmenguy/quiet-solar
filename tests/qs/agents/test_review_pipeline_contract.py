"""QS-453 — review pipeline contract: finding verifier, ledger, triage buckets.

Pins the prompt text of the two code-review orchestrators
(``qs-review-task``, ``qs-verify-task``), the new
``qs-review-finding-verifier`` sub-agent and the touched reviewers, on the
unbound render of both harnesses. Behaviour beyond the prompt text is the
ledger script's job (``tests/qs/test_review_ledger.py``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from tests.qs.agents._rendered import REPO_ROOT, agents_dir

import models  # type: ignore[import-not-found]  # isort: skip
import render_agents  # type: ignore[import-not-found]  # isort: skip

_HARNESSES = ("claude", "opencode")
_ORCHESTRATORS = ("qs-review-task.md", "qs-verify-task.md")
_VERIFIER = "qs-review-finding-verifier"
_BASH_FENCE_RE = re.compile(r"^```bash\n(.*?)\n^```$", re.DOTALL | re.MULTILINE)
_HEADINGS = (
    "### 3. Consolidate findings",
    "### 4. Verify findings",
    "### 5. Record findings",
    "### 6. Zero-blocking fast path",
    "### 7. Interactive triage",
    "### 8. Record decisions, fix plan, commit",
)
_BUCKETS = ("FLIP-FLOP", "Reopened", "Unverified", "Dropped by verifier", "Blocking", "Nice-to-have")
_LEDGER_EXCLUDE = "':(exclude)docs/stories/*.review_ledger.json'"


def _body(harness: str, filename: str) -> str:
    return (agents_dir(harness) / filename).read_text(encoding="utf-8")


def _frontmatter(text: str) -> dict:
    end = text.find("\n---\n", 4)
    return yaml.safe_load(text[4:end])


def _section(body: str, heading: str) -> str:
    """Text from the column-0 ``heading`` up to the next column-0 ``##``/``###`` heading."""
    match = re.search(rf"^{re.escape(heading)}$", body, re.MULTILINE)
    assert match, f"missing column-0 heading {heading!r}"
    rest = body[match.end() :]
    nxt = re.search(r"^#{2,3} ", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def _norm(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------- #
# AC-1 — the verifier sub-agent
# --------------------------------------------------------------------------- #


def test_verifier_registered_as_deep_subagent() -> None:
    assert _VERIFIER in render_agents.SUBAGENTS
    assert len(render_agents.SUBAGENTS) == 13
    assert len(render_agents.SUBAGENTS | render_agents.ORCHESTRATORS) == 23
    assert models.resolve(None, _VERIFIER) == "deep"


def test_verifier_claude_tools_are_read_only() -> None:
    fm = _frontmatter(_body("claude", f"{_VERIFIER}.md"))
    assert {t.strip() for t in fm["tools"].split(",")} == {"Read", "Grep", "Glob", "LSP"}


def test_verifier_opencode_permissions_deny_edit_and_bash() -> None:
    fm = _frontmatter(_body("opencode", f"{_VERIFIER}.md"))
    assert fm["mode"] == "subagent"
    assert fm["hidden"] is True
    assert fm["permission"]["edit"] == "deny"
    assert fm["permission"]["bash"] == {"*": "deny"}


@pytest.mark.parametrize("harness", _HARNESSES)
def test_verifier_body_has_key_table_and_literal_example(harness: str) -> None:
    body = _body(harness, f"{_VERIFIER}.md")
    for row in (
        "| `mode` |",
        "| `checks[].ref` / `.status` |",
        "| `findings[].verdict` |",
        "| `findings[].evidence` |",
        "| `findings[].reproduces_when` |",
        "| `findings[].severity_action` |",
        "| `findings[].relation` |",
        "| `findings[].note` |",
    ):
        assert row in body, row
    example = re.search(r"^```json\n(.*?)\n^```$", body, re.DOTALL | re.MULTILINE)
    assert example, "missing literal example object"
    obj = json.loads(example.group(1))
    assert set(obj) == {"mode", "checks", "findings"}
    assert set(obj["findings"][0]) == {
        "key",
        "source",
        "severity",
        "where",
        "title",
        "verdict",
        "evidence",
        "reproduces_when",
        "severity_action",
        "relation",
        "note",
    }
    assert "[[" not in body.split("\n---\n", 1)[1]


# --------------------------------------------------------------------------- #
# AC-2 / AC-4 — orchestrator protocol shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("harness", _HARNESSES)
@pytest.mark.parametrize("filename", _ORCHESTRATORS)
def test_orchestrator_headings_in_order(harness: str, filename: str) -> None:
    body = _body(harness, filename)
    positions = []
    for heading in _HEADINGS:
        match = re.search(rf"^{re.escape(heading)}$", body, re.MULTILINE)
        assert match, f"{filename}: missing column-0 {heading!r}"
        positions.append(match.start())
    assert positions == sorted(positions)
    assert re.search(r"^### 1\. Fetch the PR diff and open the round$", body, re.MULTILINE)


@pytest.mark.parametrize("harness", _HARNESSES)
@pytest.mark.parametrize("filename", _ORCHESTRATORS)
def test_orchestrator_step4_names_verifier_and_fallback(harness: str, filename: str) -> None:
    step4 = _norm(_section(_body(harness, filename), "### 4. Verify findings"))
    assert f"`{_VERIFIER}`" in step4
    assert '"mode": "fallback"' in step4
    for clause in (
        '"verdict": null',
        '"evidence": null',
        '"reproduces_when": null',
        '"severity_action": "keep"',
        '"note": "unverified"',
        '"checks": []',
    ):
        assert clause in step4, clause


@pytest.mark.parametrize("harness", _HARNESSES)
@pytest.mark.parametrize("filename", _ORCHESTRATORS)
def test_orchestrator_uses_every_ledger_subcommand(harness: str, filename: str) -> None:
    body = _body(harness, filename)
    for sub in ("start-round", "add", "set", "check-round", "show"):
        assert f"review_ledger.py {sub}" in body, sub
    assert "<<'QS_LEDGER'" in body


@pytest.mark.parametrize("harness", _HARNESSES)
@pytest.mark.parametrize("filename", _ORCHESTRATORS)
def test_orchestrator_triage_buckets_in_order_with_defaults(harness: str, filename: str) -> None:
    step7 = _section(_body(harness, filename), "### 7. Interactive triage")
    rows = re.findall(r"^\| (\d) \| `([^`]+)` \|(.*)\|$", step7, re.MULTILINE)
    assert [label for _n, label, _rest in rows] == list(_BUCKETS)
    assert [int(n) for n, _l, _r in rows] == list(range(1, 7))
    defaults = {label: rest for _n, label, rest in rows}
    assert "none — explicit decision" in defaults["FLIP-FLOP"]
    assert "none — explicit decision" in defaults["Unverified"]
    assert "reject" in defaults["Dropped by verifier"]
    assert "fix" in defaults["Reopened"]
    assert "fix" in defaults["Blocking"]
    assert "reject (opt-in)" in defaults["Nice-to-have"]
    assert '"fix all" covers rows 2 + 5' in _norm(step7)


@pytest.mark.parametrize("harness", _HARNESSES)
@pytest.mark.parametrize("filename", _ORCHESTRATORS)
def test_orchestrator_fast_path_only_in_verified_mode(harness: str, filename: str) -> None:
    step6 = _norm(_section(_body(harness, filename), "### 6. Zero-blocking fast path"))
    assert "only in `verified` mode" in step6
    assert "accepted_unchecked" in step6


@pytest.mark.parametrize("harness", _HARNESSES)
@pytest.mark.parametrize("filename", _ORCHESTRATORS)
def test_orchestrator_check_round_before_fix_plan(harness: str, filename: str) -> None:
    body = _body(harness, filename)
    step8 = _section(body, "### 8. Record decisions, fix plan, commit")
    assert "review_ledger.py check-round" in step8
    assert step8.index("review_ledger.py check-round") < step8.index("next_review_fix_path")
    # The fix-plan template (a fenced block with its own headings) names findings by ledger ID.
    assert body.index("next_review_fix_path") < body.index("### [must-fix] F-003 — <title>")


# --------------------------------------------------------------------------- #
# AC-5 — ledger commit fences
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("harness", _HARNESSES)
@pytest.mark.parametrize("filename", _ORCHESTRATORS)
def test_orchestrator_commit_fences_carry_the_ledger(harness: str, filename: str) -> None:
    commits = [f for f in _BASH_FENCE_RE.findall(_body(harness, filename)) if "git commit" in f]
    assert len(commits) >= 2
    for fence in commits:
        assert "review_ledger.json" in fence


# --------------------------------------------------------------------------- #
# AC-6 — delta mode per orchestrator
# --------------------------------------------------------------------------- #

_DELTA = {
    "qs-review-task.md": {
        "qs-review-blind-hunter": "delta",
        "qs-review-edge-case-hunter": "delta",
        "qs-review-acceptance-auditor": "always full",
        "qs-review-coderabbit": "always full",
    },
    "qs-verify-task.md": {
        "qs-review-edge-case-hunter": "delta",
        "qs-review-coderabbit": "always full",
        "qs-review-regression-proof": "always full",
    },
}


@pytest.mark.parametrize("harness", _HARNESSES)
@pytest.mark.parametrize("filename", _ORCHESTRATORS)
def test_orchestrator_delta_roster(harness: str, filename: str) -> None:
    body = _body(harness, filename)
    rows = dict(re.findall(r"^\| `(qs-review-[a-z-]+)` \| (delta|always full)\b", body, re.MULTILINE))
    assert rows == _DELTA[filename]
    flat = _norm(body)
    assert "docs/stories/QS-{{issue}}.review_ledger.json" in flat
    assert "docs/stories/QS-{{issue}}.story_review_fix_*" in flat


# --------------------------------------------------------------------------- #
# AC-7 / AC-8 / AC-9 — reviewers and verify-task specifics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("harness", _HARNESSES)
def test_blind_hunter_is_muzzled(harness: str) -> None:
    body = _body(harness, "qs-review-blind-hunter.md")
    for gone in ("Suspicious comments", "Style/lint violations", "#### nice-to-have", "≤2 lines"):
        assert gone not in body, gone
    assert "Reproduces when:" in body


@pytest.mark.parametrize("harness", _HARNESSES)
def test_verify_task_keeps_pr_comments(harness: str) -> None:
    flat = _norm(_body(harness, "qs-verify-task.md"))
    assert "post them as a PR comment" in flat
    assert "post the triage summary as a PR comment" in flat
    assert "only durable record" not in flat
    assert "handoff as step 4" not in flat


@pytest.mark.parametrize("harness", _HARNESSES)
def test_regression_proof_counts_ledger_as_explained(harness: str) -> None:
    assert "docs/stories/QS-<N>.review_ledger.json" in _body(harness, "qs-review-regression-proof.md")


@pytest.mark.parametrize("harness", _HARNESSES)
def test_coderabbit_parses_coderabbit_authored_only(harness: str) -> None:
    assert "authored by CodeRabbit" in _norm(_body(harness, "qs-review-coderabbit.md"))


# --------------------------------------------------------------------------- #
# AC-11 / AC-12 — permissions and implement phases
# --------------------------------------------------------------------------- #


def test_claude_settings_allow_review_ledger() -> None:
    allow = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))["permissions"]["allow"]
    assert "Bash(python scripts/qs/review_ledger.py:*)" in allow
    assert "Bash(python **/scripts/qs/review_ledger.py:*)" in allow


@pytest.mark.parametrize("harness", _HARNESSES)
@pytest.mark.parametrize("filename", ["qs-implement-task.md", "qs-implement-setup-task.md"])
def test_implement_git_add_excludes_ledger(harness: str, filename: str) -> None:
    adds = [
        line
        for fence in _BASH_FENCE_RE.findall(_body(harness, filename))
        for line in fence.splitlines()
        if line.startswith("git add ")
    ]
    assert adds
    for line in adds:
        assert _LEDGER_EXCLUDE in line, line


def test_paths_exist() -> None:
    assert Path(REPO_ROOT / "scripts" / "qs" / "review_ledger.py").is_file()
