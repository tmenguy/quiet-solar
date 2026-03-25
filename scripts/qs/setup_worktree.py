#!/usr/bin/env python3
"""Set up a worktree for story implementation.

Usage:
    python scripts/qs/setup_worktree.py <issue_number> [--story-file PATH] [--no-worktree]

Output: JSON with worktree path, branch, and launch commands.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from utils import get_main_worktree, get_worktree_dir, output_json, run_git


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up worktree for development")
    parser.add_argument("issue_number", type=int, help="GitHub issue number")
    parser.add_argument("--story-file", default=None, help="Path to story file")
    parser.add_argument("--story-key", default=None, help="Story key like 3.2")
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

    # Build context for the implementation agent
    story_file_rel = ""
    if args.story_file:
        story_file_rel = args.story_file
    elif args.story_key:
        # Find story file in the worktree
        from utils import find_story_file
        sf = find_story_file(args.story_key)
        if sf:
            story_file_rel = str(sf.relative_to(get_main_worktree()))

    title = args.title or f"Issue #{issue}"

    # Build claude launch command
    claude_cmd = f'cd "{work_dir}" && claude --name "QS_{issue}: {title}"'
    implement_prompt = f"/implement-story --issue {issue}"
    if story_file_rel:
        implement_prompt += f" --story-file {story_file_rel}"

    output_json({
        "issue_number": issue,
        "branch": branch,
        "worktree_path": work_dir,
        "story_file": story_file_rel,
        "launch_command": claude_cmd,
        "first_prompt": implement_prompt,
        "instructions": f"Run this to start implementation:\n  {claude_cmd}\nThen type:\n  {implement_prompt}",
    })


if __name__ == "__main__":
    main()
