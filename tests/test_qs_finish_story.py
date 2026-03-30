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
    commit_housekeeping,
    phase_merge,
    phase_prepare,
    phase_report,
    phase_validate,
    resolve_story_file_to_main,
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
        "gh pr list": {
            "stdout": json.dumps([{"number": 10, "url": "https://github.com/org/repo/pull/10", "state": "OPEN"}])
        },
    }
    monkeypatch.setattr(subprocess, "run", _make_run(responses))

    result = phase_prepare(branch="QS_51", issue_number=51, story_file=str(story_file))
    assert result["commit"]["committed"] is True
    assert result["pr"]["pr_number"] == 10


def test_phase_prepare_no_pr_creates_one(monkeypatch, tmp_path):
    """Creates PR when none exists."""
    story_file = tmp_path / "story.md"
    story_file.write_text("# Story 1.12\nStatus: in-progress\n")

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
        "gh pr checks": {"stdout": json.dumps([{"name": "build", "state": "SUCCESS", "bucket": "pass"}])},
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


# --- resolve_story_file_to_main tests ---


def test_resolve_story_file_to_main_already_in_main(monkeypatch, tmp_path):
    """Returns path as-is when already under main worktree."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    story_file = str(main_dir / "_bmad-output" / "story.md")

    monkeypatch.setattr("finish_story.get_main_worktree", lambda: main_dir)

    result = resolve_story_file_to_main(story_file)
    assert result == story_file


def test_resolve_story_file_to_main_from_worktree(monkeypatch, tmp_path):
    """Converts worktree path to main worktree equivalent."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    worktree_dir = tmp_path / "worktrees" / "QS_55"
    worktree_dir.mkdir(parents=True)

    story_file = str(worktree_dir / "_bmad-output" / "story.md")

    monkeypatch.setattr("finish_story.get_main_worktree", lambda: main_dir)
    monkeypatch.setattr("finish_story.get_repo_root", lambda: worktree_dir)

    result = resolve_story_file_to_main(story_file)
    assert result == str(main_dir / "_bmad-output" / "story.md")


def test_resolve_story_file_to_main_empty_string(monkeypatch):
    """Returns empty string for empty input."""
    result = resolve_story_file_to_main("")
    assert result == ""


def test_resolve_story_file_to_main_unrelated_path(monkeypatch, tmp_path):
    """Returns path as-is when not under any worktree."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    worktree_dir = tmp_path / "worktrees" / "QS_55"
    worktree_dir.mkdir(parents=True)

    # A path not under main_dir or worktree_dir
    story_file = "/some/random/path/story.md"

    monkeypatch.setattr("finish_story.get_main_worktree", lambda: main_dir)
    monkeypatch.setattr("finish_story.get_repo_root", lambda: worktree_dir)

    result = resolve_story_file_to_main(story_file)
    assert result == story_file


# --- commit_housekeeping tests ---


def test_commit_housekeeping_with_changes(monkeypatch, tmp_path):
    """Commits and pushes when there are staged changes."""
    monkeypatch.setattr("finish_story.get_main_worktree", lambda: tmp_path)

    responses = {
        "git add": {"returncode": 0},
        "git diff --cached": {"stdout": "_bmad-output/story.md\n_bmad-output/epics.md\n"},
        "git commit": {"returncode": 0},
        "git push": {"returncode": 0},
    }
    monkeypatch.setattr(subprocess, "run", _make_run(responses))

    result = commit_housekeeping("1.13")
    assert result["committed"] is True
    assert result["pushed"] is True
    assert len(result["files"]) == 2


def test_commit_housekeeping_no_changes(monkeypatch, tmp_path):
    """Reports no commit when nothing was staged."""
    monkeypatch.setattr("finish_story.get_main_worktree", lambda: tmp_path)

    responses = {
        "git add": {"returncode": 0},
        "git diff --cached": {"stdout": ""},
    }
    monkeypatch.setattr(subprocess, "run", _make_run(responses))

    result = commit_housekeeping("1.13")
    assert result["committed"] is False


def test_commit_housekeeping_commit_fails(monkeypatch, tmp_path):
    """Reports failure when commit fails."""
    monkeypatch.setattr("finish_story.get_main_worktree", lambda: tmp_path)

    responses = {
        "git add": {"returncode": 0},
        "git diff --cached": {"stdout": "_bmad-output/story.md\n"},
        "git commit": {"returncode": 1, "stderr": "nothing to commit"},
    }
    monkeypatch.setattr(subprocess, "run", _make_run(responses))

    result = commit_housekeeping("1.13")
    assert result["committed"] is False


def test_commit_housekeeping_no_story_key(monkeypatch, tmp_path):
    """Uses generic message when no story key provided."""
    monkeypatch.setattr("finish_story.get_main_worktree", lambda: tmp_path)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        result = MagicMock(returncode=0, stdout="", stderr="")
        key = " ".join(cmd[:4])
        if "git diff --cached" in key:
            result.stdout = "_bmad-output/story.md\n"
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = commit_housekeeping(None)
    assert result["committed"] is True
    # Verify the generic message was used
    commit_cmd = [c for c in calls if "commit" in c]
    assert any("update story status and epics" in " ".join(c) for c in commit_cmd)


# --- phase_merge tests ---


def test_phase_merge_happy_path(monkeypatch, tmp_path):
    """Full merge + post-merge succeeds with correct ordering."""
    story_file = tmp_path / "story.md"
    story_file.write_text("# Story\nStatus: in-progress\n")

    responses = {
        "gh pr merge": {"returncode": 0},
        "gh issue view": {"stdout": json.dumps({"state": "OPEN"})},
        "gh issue close": {"returncode": 0},
        "git worktree list": {"stdout": f"worktree {tmp_path}\n"},
        "git checkout main": {"returncode": 0},
        "git pull": {"returncode": 0},
    }
    monkeypatch.setattr(subprocess, "run", _make_run(responses))

    monkeypatch.setattr("finish_story.resolve_story_file_to_main", lambda f: str(story_file))
    monkeypatch.setattr(
        "finish_story.commit_housekeeping", lambda k: {"committed": True, "pushed": True, "files": ["story.md"]}
    )
    monkeypatch.setattr("finish_story.cleanup_worktree", lambda n: {"cleaned": True})
    monkeypatch.setattr("finish_story.update_epics", lambda k, **kw: {"updated": True})

    result = phase_merge(
        pr_number=10,
        issue_number=51,
        story_file=str(story_file),
        story_key="1.12",
    )
    assert result["merge"]["merged"] is True
    assert result["main"]["updated"] is True
    assert result["story_status"]["updated"] is True
    assert result["housekeeping_commit"]["committed"] is True
    assert result["cleanup"]["cleaned"] is True


def test_phase_merge_merge_fails(monkeypatch, tmp_path):
    """Reports merge failure — no post-merge steps executed."""
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
    assert result["cleanup"]["cleaned"] is False
    assert result["housekeeping_commit"]["committed"] is False


def test_phase_merge_correct_execution_order(monkeypatch, tmp_path):
    """Verifies the exact execution order: merge -> close -> main -> story -> epics -> commit -> cleanup."""
    story_file = tmp_path / "story.md"
    story_file.write_text("# Story\nStatus: in-progress\n")

    execution_order = []

    def track_merge_pr(n):
        execution_order.append("merge_pr")
        return {"merged": True}

    def track_close_issue(n, *, comment=None):
        execution_order.append("close_issue")
        return {"closed": True, "was_open": True}

    def track_update_main():
        execution_order.append("update_main")
        return {"updated": True}

    def track_resolve(f):
        execution_order.append("resolve_story_file")
        return str(story_file)

    def track_update_story(f, s):
        execution_order.append("update_story_status")
        return {"updated": True}

    def track_update_epics(k, **kw):
        execution_order.append("update_epics")
        return {"updated": True}

    def track_commit(k):
        execution_order.append("commit_housekeeping")
        return {"committed": True, "pushed": True, "files": []}

    def track_cleanup(n):
        execution_order.append("cleanup_worktree")
        return {"cleaned": True}

    monkeypatch.setattr("finish_story.merge_pr", track_merge_pr)
    monkeypatch.setattr("finish_story.close_issue_if_open", track_close_issue)
    monkeypatch.setattr("finish_story.update_main", track_update_main)
    monkeypatch.setattr("finish_story.resolve_story_file_to_main", track_resolve)
    monkeypatch.setattr("finish_story.update_story_status", track_update_story)
    monkeypatch.setattr("finish_story.update_epics", track_update_epics)
    monkeypatch.setattr("finish_story.commit_housekeeping", track_commit)
    monkeypatch.setattr("finish_story.cleanup_worktree", track_cleanup)

    phase_merge(
        pr_number=10,
        issue_number=51,
        story_file=str(story_file),
        story_key="1.12",
    )

    assert execution_order == [
        "merge_pr",
        "close_issue",
        "update_main",
        "resolve_story_file",
        "update_story_status",
        "update_epics",
        "commit_housekeeping",
        "cleanup_worktree",
    ]


def test_phase_merge_cleanup_runs_last_even_on_partial_failure(monkeypatch, tmp_path):
    """Cleanup runs last even when story/epics updates fail."""
    story_file = tmp_path / "story.md"
    story_file.write_text("# Story\nStatus: in-progress\n")

    execution_order = []

    monkeypatch.setattr("finish_story.merge_pr", lambda n: (execution_order.append("merge"), {"merged": True})[1])
    monkeypatch.setattr("finish_story.close_issue_if_open", lambda n, **kw: {"closed": True, "was_open": True})
    monkeypatch.setattr("finish_story.update_main", lambda: (execution_order.append("main"), {"updated": True})[1])
    monkeypatch.setattr("finish_story.resolve_story_file_to_main", lambda f: "/nonexistent/story.md")
    monkeypatch.setattr("finish_story.update_story_status", lambda f, s: {"updated": False, "detail": "file not found"})
    monkeypatch.setattr("finish_story.update_epics", lambda k, **kw: {"updated": False, "detail": "not found"})
    monkeypatch.setattr("finish_story.commit_housekeeping", lambda k: {"committed": False, "detail": "no changes"})
    monkeypatch.setattr(
        "finish_story.cleanup_worktree", lambda n: (execution_order.append("cleanup"), {"cleaned": True})[1]
    )

    result = phase_merge(
        pr_number=10,
        issue_number=51,
        story_file=str(story_file),
        story_key="1.12",
    )
    # Merge succeeded
    assert result["merge"]["merged"] is True
    # Story/epics failed but that's OK
    assert result["story_status"]["updated"] is False
    assert result["epics"]["updated"] is False
    # Cleanup still ran (and it ran after main)
    assert result["cleanup"]["cleaned"] is True
    assert execution_order == ["merge", "main", "cleanup"]


# --- phase_report tests ---


def test_phase_report_success():
    """Generates success report with release suggestion."""
    steps = {
        "prepare": {"commit": {"committed": True}, "pr": {"pr_number": 10}},
        "validate": {"quality_gate": {"passed": True}, "ci": {"all_passed": True}},
        "merge": {"merge": {"merged": True}},
    }
    report = phase_report(steps, main_dir="/tmp/main", changed_files=["custom_components/quiet_solar/foo.py"])
    assert report["success"] is True
    assert report["release_suggestion"] == "release"


def test_phase_report_failure_with_recovery():
    """Generates failure report with recovery instructions."""
    steps = {
        "prepare": {"commit": {"committed": True}, "pr": {"pr_number": 10}},
        "validate": {"quality_gate": {"passed": False}},
    }
    report = phase_report(steps, main_dir="/tmp/main", changed_files=[], failed_phase="validate")
    assert report["success"] is False
    assert "recovery" in report


def test_phase_report_no_release_for_process_only():
    """No release suggestion for process-only changes."""
    steps = {
        "prepare": {"commit": {"committed": True}, "pr": {"pr_number": 10}},
        "validate": {"quality_gate": {"passed": True}, "ci": {"all_passed": True}},
        "merge": {"merge": {"merged": True}},
    }
    report = phase_report(steps, main_dir="/tmp/main", changed_files=["scripts/qs/utils.py"])
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
    monkeypatch.setattr("finish_story.commit_housekeeping", lambda k: {"committed": True, "pushed": True, "files": []})
    monkeypatch.setattr("finish_story.cleanup_worktree", lambda n: {"cleaned": True})
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
    monkeypatch.setattr("finish_story.commit_housekeeping", lambda k: {"committed": True, "pushed": True, "files": []})
    monkeypatch.setattr("finish_story.cleanup_worktree", lambda n: {"cleaned": True})
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
    monkeypatch.setattr("finish_story.commit_housekeeping", lambda k: {"committed": True, "pushed": True, "files": []})
    monkeypatch.setattr("finish_story.cleanup_worktree", lambda n: {"cleaned": True})
    monkeypatch.setattr("finish_story.update_epics", lambda k, **kw: {"updated": True})

    report = run_finish_story(
        pr_number=10,
        issue_number=51,
        story_file=str(story_file),
        skip_quality_gate=True,
    )
    assert report["success"] is True
