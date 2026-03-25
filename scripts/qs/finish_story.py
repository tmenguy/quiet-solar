#!/usr/bin/env python3
"""Finish a story: merge PR, clean up worktree, update main.

Usage:
    python scripts/qs/finish_story.py <pr_number> [--story-key 3.2] [--skip-quality-gate]

Output: JSON with merge status and cleanup results.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from utils import get_issue_from_branch, get_main_worktree, output_json, run_gh, run_git


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


def update_epics(story_key: str, status: str = "DONE") -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Finish story")
    parser.add_argument("pr_number", type=int, help="PR number to merge")
    parser.add_argument("--story-key", default=None, help="Story key to mark done in epics.md")
    parser.add_argument("--skip-quality-gate", action="store_true", help="Skip final quality gate")
    args = parser.parse_args()

    steps: dict = {}

    # Step 0: Fetch branch name BEFORE merge (branch gets deleted during merge)
    pr_info = run_gh(["pr", "view", str(args.pr_number), "--json", "headRefName"], check=False)
    issue_number = None
    if pr_info.returncode == 0:
        import json
        try:
            data = json.loads(pr_info.stdout)
            branch = data.get("headRefName", "")
            issue_number = get_issue_from_branch(branch)
        except Exception:
            pass

    # Step 1: Quality gate (optional skip)
    if not args.skip_quality_gate:
        qg_passed = run_quality_gate()
        steps["quality_gate"] = {"passed": qg_passed}
        if not qg_passed:
            output_json({"error": "Quality gate failed. Fix issues before merging.", "steps": steps})
            sys.exit(1)

    # Step 2: Merge PR
    merge_result = merge_pr(args.pr_number)
    steps["merge"] = merge_result
    if not merge_result.get("merged"):
        output_json({"error": "Merge failed", "steps": steps})
        sys.exit(1)

    # Step 3: Cleanup worktree
    if issue_number:
        steps["cleanup"] = cleanup_worktree(issue_number)
    else:
        steps["cleanup"] = {"cleaned": False, "detail": "could not determine issue number"}

    # Step 4: Update main
    steps["update_main"] = update_main()

    # Step 5: Update epics
    if args.story_key:
        steps["epics"] = update_epics(args.story_key)

    output_json({"success": True, "steps": steps})


if __name__ == "__main__":
    main()
