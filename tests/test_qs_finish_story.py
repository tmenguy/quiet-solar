"""Tests for scripts/qs/finish_story.py orchestrator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add scripts/qs to path so we can import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "qs"))

from finish_story import (
    phase_merge,
    phase_prepare,
    phase_report,
    phase_validate,
    run_finish_story,
)


def _make_run(responses: dict | None = None):
    """Create a fake subprocess.run that returns canned responses."""
    responses = responses or {}

    def fake_run(cmd, **kwargs):
        key = " ".join(cmd[:4])
        result = MagicMock(returncode=0, stdout="", stderr="")
        for pattern, config in responses.items():
            if pattern in key:
                result.returncode = config.get("returncode", 0)
                result.stdout = config.get("stdout", "")
                result.stderr = config.get("stderr", "")
                return result
        return result

    return fake_run


# --- phase_prepare tests ---


def test_phase_prepare_with_changes(monkeypatch, tmp_path):
    """Commits pending changes and finds existing PR."""
    story_file = tmp_path / "story.md"
    story_file.write_text("# Story\nStatus: in-progress\n")

    responses = {
        "git add": {"returncode": 0},
        "git reset": {"returncode": 0},
        "git diff --cached": {"stdout": "scripts/qs/utils.py\n"},
        "git commit": {"returncode": 0},
        "git push": {"returncode": 0},
        "gh pr list": {"stdout": json.dumps([{"number": 10, "url": "https://github.com/org/repo/pull/10", "state": "OPEN"}])},
    }
    monkeypatch.setattr(subprocess, "run", _make_run(responses))

    result = phase_prepare(branch="QS_51", issue_number=51, story_file=str(story_file))
    assert result["commit"]["committed"] is True
    assert result["pr"]["pr_number"] == 10


def test_phase_prepare_no_pr_creates_one(monkeypatch, tmp_path):
    """Creates PR when none exists."""
    story_file = tmp_path / "story.md"
    story_file.write_text("# Story 1.12\nStatus: in-progress\n")

    call_count = {"pr_list": 0}

    def fake_run(cmd, **kwargs):
        key = " ".join(cmd[:4])
        result = MagicMock(returncode=0, stdout="", stderr="")
        if "gh pr list" in key:
            result.stdout = "[]"
        if "gh pr create" in key:
            result.stdout = "https://github.com/org/repo/pull/20\n"
        if "git diff --cached" in key:
            result.stdout = ""
        if "git diff --name-only" in key:
            result.stdout = "scripts/qs/utils.py\n"
        if "git log" in key:
            result.stdout = "abc1234 feat: add utils\n"
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = phase_prepare(branch="QS_51", issue_number=51, story_file=str(story_file))
    assert result["pr"]["pr_number"] == 20


# --- phase_validate tests ---


def test_phase_validate_all_pass(monkeypatch):
    """All validations pass."""
    responses = {
        "gh pr checks": {"stdout": json.dumps([{"name": "build", "state": "COMPLETED", "conclusion": "SUCCESS"}])},
        "gh pr view": {"stdout": json.dumps({"body": "Fixes #51"})},
    }
    monkeypatch.setattr(subprocess, "run", _make_run(responses))

    # Mock quality gate
    monkeypatch.setattr(
        "finish_story.run_quality_gate",
        lambda: True,
    )

    result = phase_validate(pr_number=10, issue_number=51)
    assert result["quality_gate"]["passed"] is True
    assert result["ci"]["all_passed"] is True
    assert result["issue_link"]["linked"] is True


def test_phase_validate_quality_gate_fails(monkeypatch):
    """Reports quality gate failure."""
    responses = {
        "gh pr checks": {"stdout": "[]"},
        "gh pr view": {"stdout": json.dumps({"body": "Fixes #51"})},
    }
    monkeypatch.setattr(subprocess, "run", _make_run(responses))
    monkeypatch.setattr("finish_story.run_quality_gate", lambda: False)

    result = phase_validate(pr_number=10, issue_number=51)
    assert result["quality_gate"]["passed"] is False


# --- phase_merge tests ---


def test_phase_merge_happy_path(monkeypatch, tmp_path):
    """Full merge + post-merge succeeds."""
    story_file = tmp_path / "story.md"
    story_file.write_text("# Story\nStatus: in-progress\n")

    responses = {
        "gh pr merge": {"returncode": 0},
        "gh issue view": {"stdout": json.dumps({"state": "OPEN"})},
        "gh issue close": {"returncode": 0},
        "git worktree list": {"stdout": f"worktree {tmp_path}\n"},
        "git checkout main": {"returncode": 0},
        "git pull": {"returncode": 0},
        "git branch -d": {"returncode": 0},
    }
    monkeypatch.setattr(subprocess, "run", _make_run(responses))

    # Mock cleanup_worktree and update_main to avoid real file operations
    monkeypatch.setattr("finish_story.cleanup_worktree", lambda n: {"cleaned": True})
    monkeypatch.setattr("finish_story.update_main", lambda: {"updated": True})
    monkeypatch.setattr("finish_story.update_epics", lambda k, **kw: {"updated": True})

    result = phase_merge(
        pr_number=10,
        issue_number=51,
        story_file=str(story_file),
        story_key="1.12",
    )
    assert result["merge"]["merged"] is True
    assert result["story_status"]["updated"] is True


def test_phase_merge_merge_fails(monkeypatch, tmp_path):
    """Reports merge failure with recovery instructions."""
    story_file = tmp_path / "story.md"
    story_file.write_text("# Story\nStatus: in-progress\n")

    responses = {
        "gh pr merge": {"returncode": 1, "stderr": "merge conflict"},
    }
    monkeypatch.setattr(subprocess, "run", _make_run(responses))

    result = phase_merge(
        pr_number=10,
        issue_number=51,
        story_file=str(story_file),
        story_key="1.12",
    )
    assert result["merge"]["merged"] is False


# --- phase_report tests ---


def test_phase_report_success():
    """Generates success report with release suggestion."""
    steps = {
        "prepare": {"commit": {"committed": True}, "pr": {"pr_number": 10}},
        "validate": {"quality_gate": {"passed": True}, "ci": {"all_passed": True}},
        "merge": {"merge": {"merged": True}},
    }
    report = phase_report(steps, changed_files=["custom_components/quiet_solar/foo.py"])
    assert report["success"] is True
    assert report["release_suggestion"] == "release"


def test_phase_report_failure_with_recovery():
    """Generates failure report with recovery instructions."""
    steps = {
        "prepare": {"commit": {"committed": True}, "pr": {"pr_number": 10}},
        "validate": {"quality_gate": {"passed": False}},
    }
    report = phase_report(steps, changed_files=[], failed_phase="validate")
    assert report["success"] is False
    assert "recovery" in report


def test_phase_report_no_release_for_process_only():
    """No release suggestion for process-only changes."""
    steps = {
        "prepare": {"commit": {"committed": True}, "pr": {"pr_number": 10}},
        "validate": {"quality_gate": {"passed": True}, "ci": {"all_passed": True}},
        "merge": {"merge": {"merged": True}},
    }
    report = phase_report(steps, changed_files=["scripts/qs/utils.py"])
    assert report["release_suggestion"] == "no-release"


# --- run_finish_story (full orchestrator) tests ---


def test_run_finish_story_auto_detect(monkeypatch, tmp_path):
    """Auto-detects branch, issue, story, PR with no args."""
    story_file = tmp_path / "_bmad-output" / "implementation-artifacts" / "1-12-story.md"
    story_file.parent.mkdir(parents=True)
    story_file.write_text("# Story 1.12\nStatus: in-progress\n## Tasks\n")

    def fake_run(cmd, **kwargs):
        key = " ".join(cmd[:4])
        result = MagicMock(returncode=0, stdout="", stderr="")
        if "git branch --show-current" in key:
            result.stdout = "QS_51\n"
        elif "git rev-parse --show-toplevel" in key:
            result.stdout = str(tmp_path) + "\n"
        elif "gh pr list" in key:
            result.stdout = json.dumps([{"number": 10, "url": "https://github.com/org/repo/pull/10", "state": "OPEN"}])
        elif "git diff --cached" in key:
            result.stdout = ""
        elif "git diff --name-only" in key:
            result.stdout = "scripts/qs/utils.py\n"
        elif "gh pr checks" in key:
            result.stdout = "[]"
        elif "gh pr view" in key:
            result.stdout = json.dumps({"body": "Fixes #51"})
        elif "gh pr merge" in key:
            result.returncode = 0
        elif "gh issue view" in key:
            result.stdout = json.dumps({"state": "OPEN"})
        elif "gh issue close" in key:
            result.returncode = 0
        elif "git worktree list" in key:
            result.stdout = f"worktree {tmp_path}\n"
        elif "git checkout main" in key:
            result.returncode = 0
        elif "git pull" in key:
            result.returncode = 0
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("finish_story.run_quality_gate", lambda: True)
    monkeypatch.setattr("finish_story.cleanup_worktree", lambda n: {"cleaned": True})
    monkeypatch.setattr("finish_story.update_main", lambda: {"updated": True})
    monkeypatch.setattr("finish_story.update_epics", lambda k, **kw: {"updated": True})

    report = run_finish_story()
    assert report["success"] is True


def test_run_finish_story_with_overrides(monkeypatch, tmp_path):
    """Override flags still work."""
    story_file = tmp_path / "story.md"
    story_file.write_text("# Story\nStatus: in-progress\n")

    def fake_run(cmd, **kwargs):
        key = " ".join(cmd[:4])
        result = MagicMock(returncode=0, stdout="", stderr="")
        if "gh pr list" in key:
            result.stdout = json.dumps([{"number": 99, "url": "https://github.com/org/repo/pull/99", "state": "OPEN"}])
        elif "git diff --cached" in key:
            result.stdout = ""
        elif "git diff --name-only" in key:
            result.stdout = ""
        elif "gh pr checks" in key:
            result.stdout = "[]"
        elif "gh pr view" in key:
            result.stdout = json.dumps({"body": "Fixes #51"})
        elif "gh pr merge" in key:
            result.returncode = 0
        elif "gh issue view" in key:
            result.stdout = json.dumps({"state": "CLOSED"})
        elif "git worktree list" in key:
            result.stdout = f"worktree {tmp_path}\n"
        elif "git checkout main" in key:
            result.returncode = 0
        elif "git pull" in key:
            result.returncode = 0
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("finish_story.run_quality_gate", lambda: True)
    monkeypatch.setattr("finish_story.cleanup_worktree", lambda n: {"cleaned": True})
    monkeypatch.setattr("finish_story.update_main", lambda: {"updated": True})
    monkeypatch.setattr("finish_story.update_epics", lambda k, **kw: {"updated": True})

    report = run_finish_story(
        pr_number=99,
        issue_number=51,
        story_file=str(story_file),
        story_key="1.12",
    )
    assert report["success"] is True


def test_run_finish_story_quality_gate_blocks(monkeypatch, tmp_path):
    """Blocks when quality gate fails."""
    story_file = tmp_path / "story.md"
    story_file.write_text("# Story\nStatus: in-progress\n")

    def fake_run(cmd, **kwargs):
        key = " ".join(cmd[:4])
        result = MagicMock(returncode=0, stdout="", stderr="")
        if "gh pr list" in key:
            result.stdout = json.dumps([{"number": 10, "url": "https://github.com/org/repo/pull/10", "state": "OPEN"}])
        elif "git diff --cached" in key:
            result.stdout = ""
        elif "git diff --name-only" in key:
            result.stdout = ""
        elif "gh pr checks" in key:
            result.stdout = "[]"
        elif "gh pr view" in key:
            result.stdout = json.dumps({"body": "Fixes #51"})
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("finish_story.run_quality_gate", lambda: False)

    report = run_finish_story(
        pr_number=10,
        issue_number=51,
        story_file=str(story_file),
    )
    assert report["success"] is False
    assert "recovery" in report


def test_run_finish_story_skip_quality_gate(monkeypatch, tmp_path):
    """Skips quality gate when flag set."""
    story_file = tmp_path / "story.md"
    story_file.write_text("# Story\nStatus: in-progress\n")

    def fake_run(cmd, **kwargs):
        key = " ".join(cmd[:4])
        result = MagicMock(returncode=0, stdout="", stderr="")
        if "gh pr list" in key:
            result.stdout = json.dumps([{"number": 10, "url": "https://github.com/org/repo/pull/10", "state": "OPEN"}])
        elif "git diff --cached" in key:
            result.stdout = ""
        elif "git diff --name-only" in key:
            result.stdout = ""
        elif "gh pr checks" in key:
            result.stdout = "[]"
        elif "gh pr view" in key:
            result.stdout = json.dumps({"body": "Fixes #51"})
        elif "gh pr merge" in key:
            result.returncode = 0
        elif "gh issue view" in key:
            result.stdout = json.dumps({"state": "CLOSED"})
        elif "git worktree list" in key:
            result.stdout = f"worktree {tmp_path}\n"
        elif "git checkout main" in key:
            result.returncode = 0
        elif "git pull" in key:
            result.returncode = 0
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("finish_story.cleanup_worktree", lambda n: {"cleaned": True})
    monkeypatch.setattr("finish_story.update_main", lambda: {"updated": True})
    monkeypatch.setattr("finish_story.update_epics", lambda k, **kw: {"updated": True})

    report = run_finish_story(
        pr_number=10,
        issue_number=51,
        story_file=str(story_file),
        skip_quality_gate=True,
    )
    assert report["success"] is True
