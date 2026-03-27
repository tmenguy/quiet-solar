#!/usr/bin/env python3
"""Finish a story: zero-arg orchestrator that auto-detects everything.

Usage:
    python scripts/qs/finish_story.py [--pr N] [--issue N] [--story-key X.Y]
                                       [--story-file PATH] [--skip-quality-gate]

All flags are optional overrides. By default, auto-detects from the current branch.

Output: JSON with every step's status and recovery instructions on failure.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from utils import (
    auto_commit_and_push,
    check_ci_status,
    close_issue_if_open,
    ensure_issue_link,
    find_pr_for_branch,
    find_story_file,
    get_changed_files,
    get_current_branch,
    get_issue_from_branch,
    get_main_worktree,
    output_json,
    run_gh,
    run_git,
    suggest_release,
    update_story_status,
)


def run_quality_gate() -> bool:
    """Run quality gates, return True if all pass."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "quality_gate.py"), "--json"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def merge_pr(pr_number: int) -> dict:
    """Merge PR with merge commit, delete remote branch."""
    result = run_gh([
        "pr", "merge", str(pr_number), "--merge", "--delete-branch",
    ], check=False)
    if result.returncode != 0:
        return {"merged": False, "detail": result.stderr.strip()}
    return {"merged": True}


def cleanup_worktree(issue_number: int) -> dict:
    """Clean up worktree and local branch."""
    main_dir = get_main_worktree()
    cleanup_script = main_dir / "scripts" / "worktree-cleanup.sh"

    if not cleanup_script.exists():
        return {"cleaned": False, "detail": "cleanup script not found"}

    result = subprocess.run(
        ["bash", str(cleanup_script), str(issue_number)],
        capture_output=True, text=True, cwd=str(main_dir),
        input="\n",  # Auto-confirm any prompts
    )
    if result.returncode != 0:
        # Worktree might not exist (no-worktree mode)
        # Try deleting local branch directly
        run_git(["branch", "-d", f"QS_{issue_number}"], check=False, cwd=str(main_dir))
        return {"cleaned": True, "detail": "no worktree, deleted local branch"}

    return {"cleaned": True}


def update_main() -> dict:
    """Switch to main and pull."""
    main_dir = get_main_worktree()
    run_git(["checkout", "main"], cwd=str(main_dir))
    result = run_git(["pull"], cwd=str(main_dir), check=False)
    return {"updated": result.returncode == 0}


def update_epics(story_key: str, *, status: str = "DONE") -> dict:
    """Update epics.md to mark a story as done."""
    main_dir = get_main_worktree()
    epics_file = main_dir / "_bmad-output" / "planning-artifacts" / "epics.md"

    if not epics_file.exists():
        return {"updated": False, "detail": "epics.md not found"}

    content = epics_file.read_text()
    # Normalize key: "3.2" -> match "Story 3.2"
    pattern = re.compile(
        rf"(###\s+Story\s+{re.escape(story_key)}:\s+[^\n]*?)(\s*\[(?:DONE|DISMISSED)\])?\s*$",
        re.MULTILINE,
    )
    match = pattern.search(content)
    if not match:
        return {"updated": False, "detail": f"Story {story_key} not found in epics.md"}

    if match.group(2):
        return {"updated": False, "detail": f"Story {story_key} already marked as {match.group(2).strip()}"}

    new_content = content[:match.start()] + match.group(1) + f" [{status}]" + content[match.end():]
    epics_file.write_text(new_content)
    return {"updated": True, "story_key": story_key, "status": status}


# --- Orchestrator phases ---


def phase_prepare(
    *,
    branch: str,
    issue_number: int,
    story_file: str,
) -> dict:
    """Phase 1: Commit pending changes, find or create PR.

    Returns {"commit": dict, "pr": dict}.
    """
    # Auto-commit pending changes
    commit_result = auto_commit_and_push(f"fix: final changes for #{issue_number}")

    # Find existing PR
    pr_info = find_pr_for_branch(branch)
    if pr_info:
        return {"commit": commit_result, "pr": pr_info}

    # Create PR — build title and summary from story + git log
    story_path = Path(story_file)
    title = f"feat: story {issue_number}"
    if story_path.exists():
        first_heading = ""
        for line in story_path.read_text().split("\n"):
            if line.startswith("# "):
                first_heading = line[2:].strip()
                break
        if first_heading:
            title = first_heading[:70]

    # Get summary from git log
    log_result = run_git(["log", "--oneline", "main..HEAD"], check=False)
    summary = log_result.stdout.strip()[:300] if log_result.returncode == 0 else "Implementation complete"

    changed = get_changed_files()
    fixes_line = f"\nFixes #{issue_number}\n" if issue_number else ""
    body = f"## Summary\n{summary}\n{fixes_line}"

    # Push first
    run_git(["push", "-u", "origin", branch], check=False)

    create_result = run_gh(
        ["pr", "create", "--title", title, "--body", body],
        check=False,
    )
    if create_result.returncode != 0:
        return {
            "commit": commit_result,
            "pr": {"error": "PR creation failed", "detail": create_result.stderr.strip()},
        }

    url = create_result.stdout.strip()
    pr_number = int(url.rstrip("/").split("/")[-1])
    return {
        "commit": commit_result,
        "pr": {"pr_number": pr_number, "url": url, "created": True},
    }


def phase_validate(*, pr_number: int, issue_number: int, skip_quality_gate: bool = False) -> dict:
    """Phase 2: Run quality gate, check CI, verify issue link.

    Returns {"quality_gate": dict, "ci": dict, "issue_link": dict}.
    """
    # Quality gate
    if skip_quality_gate:
        qg_result = {"passed": True, "skipped": True}
    else:
        qg_passed = run_quality_gate()
        qg_result = {"passed": qg_passed}

    # CI status
    ci_result = check_ci_status(pr_number)

    # Issue link
    link_result = ensure_issue_link(pr_number, issue_number)

    return {
        "quality_gate": qg_result,
        "ci": ci_result,
        "issue_link": link_result,
    }


def phase_merge(
    *,
    pr_number: int,
    issue_number: int,
    story_file: str,
    story_key: str | None = None,
) -> dict:
    """Phase 3: Merge PR, close issue, update story status, update epics, cleanup.

    Returns {"merge": dict, "issue": dict, "story_status": dict, "epics": dict, "cleanup": dict, "main": dict}.
    """
    # Merge
    merge_result = merge_pr(pr_number)
    if not merge_result.get("merged"):
        return {
            "merge": merge_result,
            "issue": {"closed": False},
            "story_status": {"updated": False},
            "epics": {"updated": False},
            "cleanup": {"cleaned": False},
            "main": {"updated": False},
        }

    # Close issue
    issue_result = close_issue_if_open(
        issue_number,
        comment=f"Merged via PR #{pr_number}",
    )

    # Update story status
    story_result = update_story_status(story_file, "done")

    # Update epics
    epics_result = {"updated": False, "detail": "no story key"}
    if story_key:
        epics_result = update_epics(story_key)

    # Cleanup worktree
    cleanup_result = cleanup_worktree(issue_number)

    # Update main
    main_result = update_main()

    return {
        "merge": merge_result,
        "issue": issue_result,
        "story_status": story_result,
        "epics": epics_result,
        "cleanup": cleanup_result,
        "main": main_result,
    }


def phase_report(
    steps: dict,
    *,
    changed_files: list[str] | None = None,
    failed_phase: str | None = None,
) -> dict:
    """Phase 4: Build structured JSON report.

    Returns the full report with success status, all step results,
    release suggestion, and recovery instructions on failure.
    """
    release = suggest_release(changed_files or [])

    if failed_phase:
        recovery_map = {
            "prepare": "Fix commit/push issues, then re-run finish-story",
            "validate": "Fix quality gate or CI failures, push fixes, then re-run finish-story",
            "merge": "Resolve merge conflicts or PR issues, then re-run finish-story",
        }
        return {
            "success": False,
            "failed_phase": failed_phase,
            "steps": steps,
            "recovery": recovery_map.get(failed_phase, "Check the error details and retry"),
            "release_suggestion": release,
        }

    return {
        "success": True,
        "steps": steps,
        "release_suggestion": release,
    }


def run_finish_story(
    *,
    pr_number: int | None = None,
    issue_number: int | None = None,
    story_file: str | None = None,
    story_key: str | None = None,
    skip_quality_gate: bool = False,
) -> dict:
    """Run the full finish-story orchestration.

    All parameters are optional — auto-detects from current branch by default.
    """
    steps: dict = {}

    # Auto-detect from branch
    branch = get_current_branch()
    if issue_number is None:
        issue_number = get_issue_from_branch(branch)
    if issue_number is None:
        return phase_report(steps, failed_phase="prepare")

    # Auto-discover story file
    if story_file is None:
        found = find_story_file(story_key)
        if found:
            story_file = str(found)
        else:
            story_file = ""

    # Extract story key from filename if not provided
    if story_key is None and story_file:
        # "1-12-systematic-finish-story-enhancement.md" -> "1.12"
        name = Path(story_file).stem
        key_match = re.match(r"^(\d+)-(\d+)", name)
        if key_match:
            story_key = f"{key_match.group(1)}.{key_match.group(2)}"

    # Phase 1: Prepare
    prepare_result = phase_prepare(
        branch=branch,
        issue_number=issue_number,
        story_file=story_file,
    )
    steps["prepare"] = prepare_result

    # Resolve PR number
    pr = prepare_result.get("pr", {})
    if pr_number is None:
        pr_number = pr.get("pr_number")
    if pr_number is None:
        return phase_report(steps, failed_phase="prepare")

    # Phase 2: Validate
    validate_result = phase_validate(
        pr_number=pr_number,
        issue_number=issue_number,
        skip_quality_gate=skip_quality_gate,
    )
    steps["validate"] = validate_result

    # Block on quality gate failure
    if not validate_result["quality_gate"]["passed"]:
        return phase_report(steps, failed_phase="validate")

    # Phase 3: Merge + post-merge
    merge_result = phase_merge(
        pr_number=pr_number,
        issue_number=issue_number,
        story_file=story_file,
        story_key=story_key,
    )
    steps["merge"] = merge_result

    if not merge_result["merge"].get("merged"):
        return phase_report(steps, failed_phase="merge")

    # Phase 4: Report
    changed = get_changed_files()
    return phase_report(steps, changed_files=changed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finish story — zero-arg orchestrator")
    parser.add_argument("--pr", type=int, default=None, help="PR number (auto-detected if omitted)")
    parser.add_argument("--issue", type=int, default=None, help="Issue number (auto-detected from branch)")
    parser.add_argument("--story-key", default=None, help="Story key (e.g., '1.12') for epics update")
    parser.add_argument("--story-file", default=None, help="Path to story artifact")
    parser.add_argument("--skip-quality-gate", action="store_true", help="Skip quality gate check")
    args = parser.parse_args()

    report = run_finish_story(
        pr_number=args.pr,
        issue_number=args.issue,
        story_file=args.story_file,
        story_key=args.story_key,
        skip_quality_gate=args.skip_quality_gate,
    )

    output_json(report)
    if not report.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
