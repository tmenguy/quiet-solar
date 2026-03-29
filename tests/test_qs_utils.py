"""Tests for scripts/qs/utils.py workflow helper functions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add scripts/qs to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "qs"))

from utils import (
    auto_commit_and_push,
    check_ci_status,
    close_issue_if_open,
    ensure_issue_link,
    find_pr_for_branch,
    get_changed_files,
    suggest_release,
    update_story_status,
)

# --- auto_commit_and_push tests ---


def test_auto_commit_and_push_with_changes(monkeypatch):
    """Commits and pushes when there are staged changes."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        result = MagicMock(returncode=0, stdout="", stderr="")
        # git diff --cached --name-only returns files
        if cmd[:3] == ["git", "diff", "--cached"]:
            result.stdout = "custom_components/quiet_solar/foo.py\ntests/test_foo.py\n"
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = auto_commit_and_push("test commit")
    assert result["committed"] is True
    assert result["pushed"] is True
    assert len(result["files"]) == 2


def test_auto_commit_and_push_no_changes(monkeypatch):
    """Does nothing when there are no changes to commit."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stdout="", stderr="")
        # git add returns ok but no changes staged
        if cmd[:3] == ["git", "diff", "--cached"]:
            result.stdout = ""
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = auto_commit_and_push("test commit")
    assert result["committed"] is False
    assert result["pushed"] is False
    assert result["files"] == []


def test_auto_commit_and_push_excludes_junk(monkeypatch):
    """Excludes .DS_Store, __pycache__, venv, config from staging."""
    add_calls = []

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stdout="", stderr="")
        if cmd[0] == "git" and cmd[1] == "add":
            add_calls.append(cmd)
        if cmd[:3] == ["git", "diff", "--cached"]:
            result.stdout = "scripts/qs/utils.py\n"
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    auto_commit_and_push("test commit")
    # Verify no junk paths in add commands
    for call in add_calls:
        for arg in call:
            assert ".DS_Store" not in arg
            assert "__pycache__" not in arg
            assert "venv" not in arg


def test_auto_commit_and_push_custom_paths(monkeypatch):
    """Accepts custom paths to stage."""
    add_calls = []

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stdout="", stderr="")
        if cmd[0] == "git" and cmd[1] == "add":
            add_calls.append(cmd)
        if cmd[:3] == ["git", "diff", "--cached"]:
            result.stdout = "docs/readme.md\n"
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    auto_commit_and_push("test", paths=["docs/"])
    # Should have called git add with docs/
    assert any("docs/" in str(call) for call in add_calls)


def test_auto_commit_and_push_push_fails(monkeypatch):
    """Reports push failure but commit still succeeded."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["git", "diff", "--cached"]:
            result.stdout = "scripts/qs/utils.py\n"
        if cmd[0] == "git" and cmd[1] == "push":
            result.returncode = 1
            result.stderr = "push failed"
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = auto_commit_and_push("test commit")
    assert result["committed"] is True
    assert result["pushed"] is False


def test_auto_commit_and_push_commit_fails(monkeypatch):
    """Reports commit failure when git commit returns non-zero."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["git", "diff", "--cached"]:
            result.stdout = "scripts/qs/utils.py\n"
        if cmd[0] == "git" and cmd[1] == "commit":
            result.returncode = 1
            result.stderr = "pre-commit hook failed"
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = auto_commit_and_push("test commit")
    assert result["committed"] is False
    assert result["pushed"] is False
    assert "pre-commit hook failed" in result.get("detail", "")


# --- find_pr_for_branch tests ---


def test_find_pr_for_branch_found(monkeypatch):
    """Returns PR info when a PR exists for the branch."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stderr="")
        result.stdout = json.dumps([{"number": 42, "url": "https://github.com/org/repo/pull/42", "state": "OPEN"}])
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = find_pr_for_branch("QS_51")
    assert result is not None
    assert result["pr_number"] == 42
    assert result["url"] == "https://github.com/org/repo/pull/42"


def test_find_pr_for_branch_not_found(monkeypatch):
    """Returns None when no PR exists."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stderr="")
        result.stdout = "[]"
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = find_pr_for_branch("QS_99")
    assert result is None


def test_find_pr_for_branch_gh_error(monkeypatch):
    """Returns None on gh CLI error."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=1, stdout="", stderr="not found")
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = find_pr_for_branch("QS_51")
    assert result is None


# --- check_ci_status tests ---


def test_check_ci_status_all_pass(monkeypatch):
    """Reports all checks passed."""
    checks = [
        {"name": "build", "state": "SUCCESS", "bucket": "pass"},
        {"name": "lint", "state": "SUCCESS", "bucket": "pass"},
    ]

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stderr="")
        result.stdout = json.dumps(checks)
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = check_ci_status(42)
    assert result["all_passed"] is True
    assert result["failed"] == []
    assert result["pending"] == []


def test_check_ci_status_some_fail(monkeypatch):
    """Reports failed checks."""
    checks = [
        {"name": "build", "state": "SUCCESS", "bucket": "pass"},
        {"name": "lint", "state": "FAILURE", "bucket": "fail"},
    ]

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stderr="")
        result.stdout = json.dumps(checks)
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = check_ci_status(42)
    assert result["all_passed"] is False
    assert "lint" in result["failed"]


def test_check_ci_status_pending(monkeypatch):
    """Reports pending checks."""
    checks = [
        {"name": "build", "state": "IN_PROGRESS", "bucket": "pending"},
    ]

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stderr="")
        result.stdout = json.dumps(checks)
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = check_ci_status(42)
    assert result["all_passed"] is False
    assert "build" in result["pending"]


def test_check_ci_status_no_checks(monkeypatch):
    """Handles no CI checks gracefully."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stderr="")
        result.stdout = "[]"
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = check_ci_status(42)
    assert result["all_passed"] is True
    assert result["checks"] == []


def test_check_ci_status_gh_error(monkeypatch):
    """Handles gh CLI error."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=1, stdout="", stderr="error")
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = check_ci_status(42)
    assert result["all_passed"] is False
    assert "error" in result.get("detail", "")


# --- ensure_issue_link tests ---


def test_ensure_issue_link_already_present(monkeypatch):
    """Does nothing when issue link already in PR body."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stderr="")
        if "view" in cmd:
            result.stdout = json.dumps({"body": "Some text\nFixes #51\nMore text"})
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = ensure_issue_link(42, 51)
    assert result["linked"] is True
    assert result["added"] is False


def test_ensure_issue_link_missing_and_added(monkeypatch):
    """Adds issue link when missing from PR body."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        result = MagicMock(returncode=0, stderr="")
        if "view" in cmd:
            result.stdout = json.dumps({"body": "Some PR description"})
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = ensure_issue_link(42, 51)
    assert result["linked"] is True
    assert result["added"] is True
    # Verify gh pr edit was called
    assert any("edit" in str(c) for c in calls)


def test_ensure_issue_link_closes_variant(monkeypatch):
    """Detects Closes #N variant."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stderr="")
        if "view" in cmd:
            result.stdout = json.dumps({"body": "Closes #51"})
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = ensure_issue_link(42, 51)
    assert result["linked"] is True
    assert result["added"] is False


def test_ensure_issue_link_edit_failure(monkeypatch):
    """Reports failure when gh pr edit fails."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stderr="")
        if "view" in cmd:
            result.stdout = json.dumps({"body": "No link here"})
        if "edit" in cmd:
            result.returncode = 1
            result.stderr = "edit failed"
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = ensure_issue_link(42, 51)
    assert result["linked"] is False
    assert result["added"] is False


# --- close_issue_if_open tests ---


def test_close_issue_if_open_closes(monkeypatch):
    """Closes an open issue."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stderr="")
        if "view" in cmd:
            result.stdout = json.dumps({"state": "OPEN"})
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = close_issue_if_open(51)
    assert result["closed"] is True
    assert result["was_open"] is True


def test_close_issue_if_open_already_closed(monkeypatch):
    """Does nothing when issue is already closed."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stderr="")
        if "view" in cmd:
            result.stdout = json.dumps({"state": "CLOSED"})
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = close_issue_if_open(51)
    assert result["closed"] is False
    assert result["was_open"] is False


def test_close_issue_if_open_with_comment(monkeypatch):
    """Passes comment to gh issue close."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        result = MagicMock(returncode=0, stderr="")
        if "view" in cmd:
            result.stdout = json.dumps({"state": "OPEN"})
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    close_issue_if_open(51, comment="Done via script")
    close_cmd = [c for c in calls if "close" in c]
    assert len(close_cmd) == 1
    assert "--comment" in close_cmd[0]


def test_close_issue_if_open_close_fails(monkeypatch):
    """Reports failure when gh issue close fails."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stderr="")
        if "view" in cmd:
            result.stdout = json.dumps({"state": "OPEN"})
        if "close" in cmd:
            result.returncode = 1
            result.stderr = "close failed"
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = close_issue_if_open(51)
    assert result["closed"] is False
    assert result["was_open"] is True


# --- update_story_status tests ---


def test_update_story_status_updates(tmp_path):
    """Updates Status line in story file."""
    story = tmp_path / "story.md"
    story.write_text("# Story\n\nStatus: ready-for-dev\n\n## Tasks\n")
    result = update_story_status(str(story), "done")
    assert result["updated"] is True
    assert "Status: done" in story.read_text()


def test_update_story_status_missing_status_line(tmp_path):
    """Handles story file without Status line."""
    story = tmp_path / "story.md"
    story.write_text("# Story\n\n## Tasks\n")
    result = update_story_status(str(story), "done")
    assert result["updated"] is False


def test_update_story_status_various_formats(tmp_path):
    """Handles various Status line formats."""
    story = tmp_path / "story.md"
    story.write_text("# Story\n\nStatus: in-progress\n\n## Tasks\n")
    result = update_story_status(str(story), "done")
    assert result["updated"] is True
    content = story.read_text()
    assert "Status: done" in content
    assert "Status: in-progress" not in content


# --- suggest_release tests ---


def test_suggest_release_production_changes():
    """Suggests release when custom_components/ changed."""
    result = suggest_release(["custom_components/quiet_solar/solver.py", "tests/test_solver.py"])
    assert result == "release"


def test_suggest_release_process_only():
    """No release needed for process-only changes."""
    result = suggest_release(["scripts/qs/utils.py", "_qsprocess/skills/finish-story.md"])
    assert result == "no-release"


def test_suggest_release_mixed():
    """Suggests release for mixed changes."""
    result = suggest_release(["scripts/qs/utils.py", "custom_components/quiet_solar/const.py"])
    assert result == "release"


def test_suggest_release_empty():
    """No release for empty file list."""
    result = suggest_release([])
    assert result == "no-release"


# --- get_changed_files tests ---


def test_get_changed_files_success(monkeypatch):
    """Returns list of changed files."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stderr="")
        result.stdout = "custom_components/quiet_solar/foo.py\ntests/test_foo.py\n"
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = get_changed_files()
    assert result == ["custom_components/quiet_solar/foo.py", "tests/test_foo.py"]


def test_get_changed_files_empty(monkeypatch):
    """Returns empty list when no changes."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stderr="")
        result.stdout = ""
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = get_changed_files()
    assert result == []


def test_get_changed_files_git_error(monkeypatch):
    """Returns empty list on git error."""

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=1, stdout="", stderr="error")
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = get_changed_files()
    assert result == []
