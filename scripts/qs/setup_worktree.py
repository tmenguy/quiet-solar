#!/usr/bin/env python3
"""Set up a worktree for story implementation.

Usage:
    python scripts/qs/setup_worktree.py <issue_number> [--no-worktree]

Output: JSON with worktree path, branch, and launch commands.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from utils import build_next_step, find_story_file, get_main_worktree, get_worktree_dir, output_json, run_git


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up worktree for development")
    parser.add_argument("issue_number", type=int, help="GitHub issue number")
    parser.add_argument("--title", default=None, help="Issue/story title for display")
    parser.add_argument("--no-worktree", action="store_true", help="Use branch in main dir instead")
    args = parser.parse_args()

    issue = args.issue_number
    branch = f"QS_{issue}"
    main_dir = get_main_worktree()

    # Check for uncommitted changes before switching branches
    dirty = run_git(["status", "--porcelain"], cwd=str(main_dir), check=False)
    tracked_changes = [l for l in dirty.stdout.strip().split("\n") if l.strip() and not l.startswith("??")]
    if tracked_changes:
        output_json({
            "error": "Main worktree has uncommitted tracked changes. Commit or stash them first.",
            "dirty_files": tracked_changes[:10],
        })
        sys.exit(1)

    # Ensure we're on main and up to date
    run_git(["checkout", "main"], cwd=str(main_dir))
    run_git(["pull"], cwd=str(main_dir))

    if args.no_worktree:
        # Simple branch mode
        run_git(["checkout", "-b", branch], cwd=str(main_dir))
        work_dir = str(main_dir)
    else:
        # Worktree mode — use existing setup script
        setup_script = main_dir / "scripts" / "worktree-setup.sh"
        result = subprocess.run(
            ["bash", str(setup_script), str(issue)],
            capture_output=True, text=True, cwd=str(main_dir),
        )
        if result.returncode != 0:
            output_json({"error": "Worktree setup failed", "detail": result.stderr.strip() or result.stdout.strip()})
            sys.exit(1)
        work_dir = str(get_worktree_dir(issue))

    # Discover story file by issue number
    sf = find_story_file(issue)
    story_file_rel = str(sf.relative_to(get_main_worktree())) if sf else ""

    title = args.title or f"Issue #{issue}"

    implement_prompt = f"/implement-story --issue {issue}"
    next_step = build_next_step(
        work_dir, issue, title, skill_prompt=implement_prompt,
    )

    output_json({
        "issue_number": issue,
        "branch": branch,
        "worktree_path": work_dir,
        "story_file": story_file_rel,
        **next_step,
    })


if __name__ == "__main__":
    main()
