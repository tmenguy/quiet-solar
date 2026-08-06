"""Tests for ``scripts/qs/create_pr.py`` — epic linkage + Lane note (QS-332 B5).

Machine-owned, no agent involvement: the parent epic is resolved from
the issue body (``targets.parse_parent_epic``) and appended as ``Refs``
inside the fixes-line slot; a cross-target diff injects a ``## Lane
note`` section recomputed from ``targets.classify`` — never relayed
from gate output.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

PR_URL = "https://github.com/tmenguy/quiet-solar/pull/99\n"

GIT_BRANCH = ["git", "branch", "--show-current"]
GIT_PUSH = ["git", "push", "-u"]
GIT_DIFF = ["git", "diff", "--name-only"]
GH_PR_LIST = ["gh", "pr", "list"]
GH_PR_CREATE = ["gh", "pr", "create"]
GH_ISSUE_VIEW = ["gh", "issue", "view"]


def _make_fake_run(
    *,
    issue_body: str = "",
    issue_labels: list[str] | None = None,
    changed_files: list[str] | None = None,
    issue_view_rc: int = 0,
    null_labels: bool = False,
):
    """Return ``(fake_run, seen)``; ``seen`` records every command.

    ``null_labels`` emits a literal JSON ``"labels": null`` (what the API
    can actually return), which is NOT the same as an empty list — see
    the review-fix #03 test below.
    """
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(list(cmd))
        head = cmd[:3]
        if head == GIT_BRANCH:
            return subprocess.CompletedProcess(cmd, 0, stdout="QS_42\n", stderr="")
        if head == GH_PR_LIST:
            return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")
        if head == GIT_PUSH:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if head == GIT_DIFF:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="\n".join(changed_files or []) + "\n", stderr=""
            )
        if head == GH_ISSUE_VIEW:
            payload = json.dumps(
                {
                    "body": issue_body,
                    "labels": None
                    if null_labels
                    else [{"name": name} for name in (issue_labels or [])],
                }
            )
            return subprocess.CompletedProcess(
                cmd, issue_view_rc, stdout=payload, stderr=""
            )
        if head == GH_PR_CREATE:
            return subprocess.CompletedProcess(cmd, 0, stdout=PR_URL, stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    return fake_run, seen


def _created_body(seen: list[list[str]]) -> str:
    create = next(cmd for cmd in seen if cmd[:3] == GH_PR_CREATE)
    return create[create.index("--body") + 1]


def _run_main(monkeypatch: pytest.MonkeyPatch, fake_run, argv: list[str] | None = None) -> None:
    import create_pr
    import utils

    monkeypatch.setattr(utils, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        ["create_pr.py", "--title", "t", "--summary", "- s", "--risk", "LOW", *(argv or [])],
    )
    create_pr.main()


def _expected_plain_body() -> str:
    """Today's body, byte-for-byte, for issue 42 / summary '- s' / risk LOW."""
    risk_lines = [
        "- [ ] CRITICAL (solver, constraints, charger budgeting)",
        "- [ ] HIGH (load base, constants, orchestration)",
        "- [ ] MEDIUM (device-specific: car, person, battery, solar)",
        "- [x] LOW (platforms, UI, docs)",
    ]
    return f"""## Summary
- s

Fixes #42

## Testing
- [x] Tests added/updated for new behavior
- [x] 100% coverage verified
- [x] No flaky tests introduced

## Code quality
- [x] Ruff passes (lint + format)
- [x] MyPy passes
- [x] No new `# type: ignore` or `noqa` without justification

## Risk assessment
{chr(10).join(risk_lines)}

---
Generated with [Claude Code](https://claude.com/claude-code)"""


def test_no_epic_no_crossing_body_is_byte_identical_to_today(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_run, seen = _make_fake_run(
        issue_body="no epic here",
        issue_labels=["kind:feature", "target:factory", "scale:task"],
        changed_files=["scripts/qs/targets.py", "docs/workflow/lanes/bug-product.md"],
    )
    _run_main(monkeypatch, fake_run)
    assert _created_body(seen) == _expected_plain_body()
    assert json.loads(capsys.readouterr().out)["pr_number"] == 99


def test_declared_parent_epic_appends_refs_in_fixes_slot(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_run, seen = _make_fake_run(
        issue_body="intro\n\nRefs #321\n",
        issue_labels=["kind:feature", "target:factory", "scale:task"],
        changed_files=["scripts/qs/targets.py"],
    )
    _run_main(monkeypatch, fake_run)
    body = _created_body(seen)
    # Pinned placement (review planner): inside the existing fixes-line slot.
    assert "\nFixes #42\nRefs #321\n" in body


def test_crossing_diff_injects_lane_note(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_run, seen = _make_fake_run(
        issue_body="",
        issue_labels=["kind:feature", "target:factory", "scale:task"],
        changed_files=[
            "scripts/qs/targets.py",
            "custom_components/quiet_solar/solver.py",
            "tests/test_solver.py",
            "docs/stories/QS-42.story.md",  # neutral — never listed
        ],
    )
    _run_main(monkeypatch, fake_run)
    body = _created_body(seen)
    assert "## Lane note" in body
    assert "custom_components/quiet_solar/solver.py" in body
    assert "tests/test_solver.py" in body
    assert "docs/stories/QS-42.story.md" not in body


def test_no_crossing_means_no_lane_note(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_run, seen = _make_fake_run(
        issue_body="",
        issue_labels=["kind:bug", "target:product", "scale:task"],
        changed_files=["custom_components/quiet_solar/solver.py", "tests/test_solver.py"],
    )
    _run_main(monkeypatch, fake_run)
    assert "## Lane note" not in _created_body(seen)


def test_null_labels_still_emits_the_parent_epic_refs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review-fix #03: with a bare ``.get("labels", [])`` a
    ``"labels": null`` response was caught by the broad ``except
    TypeError`` and silently dropped a perfectly readable parent-epic
    ``Refs`` (and any Lane note). The epic comes from the BODY — a null
    labels field must not cost it."""
    fake_run, seen = _make_fake_run(
        issue_body="intro\n\nRefs #321\n",
        null_labels=True,
        changed_files=["scripts/qs/targets.py"],
    )
    _run_main(monkeypatch, fake_run)
    body = _created_body(seen)
    assert "\nFixes #42\nRefs #321\n" in body
    # No declared target is resolvable, so no Lane note — but the PR is
    # otherwise intact.
    assert "## Lane note" not in body


def test_issue_view_failure_degrades_to_todays_body(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failing ``gh issue view`` must not block the PR — no Refs, no note."""
    fake_run, seen = _make_fake_run(
        issue_view_rc=1,
        changed_files=["custom_components/quiet_solar/solver.py"],
    )
    _run_main(monkeypatch, fake_run)
    assert _created_body(seen) == _expected_plain_body()
