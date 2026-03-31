#!/usr/bin/env python3
"""Set up branch and worktree for a new task (without touching main's checkout).

Usage:
    python scripts/qs/setup_task.py <issue_number> --title "..." [--no-worktree] [--story-key X] [--plan /path]

Output: JSON with worktree path, branch, and next-step commands for /create-plan.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from utils import build_next_step, get_main_worktree, get_worktree_dir, output_json, run_git


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up branch and worktree for a task")
    parser.add_argument("issue_number", type=int, help="GitHub issue number")
    parser.add_argument("--title", default=None, help="Issue/story title for display")
    parser.add_argument("--no-worktree", action="store_true", help="Create branch only, no worktree")
    parser.add_argument("--story-key", default=None, help="Story key (e.g., 3.2) to pass to /create-plan")
    parser.add_argument("--plan", default=None, help="Path to external plan .md file to pass to /create-plan")
    args = parser.parse_args()

    issue = args.issue_number
    branch = f"QS_{issue}"
    main_dir = get_main_worktree()

    # Fetch latest without touching main's checkout state
    run_git(["fetch", "origin"], cwd=str(main_dir))

    if args.no_worktree:
        # Create branch from origin/main without switching
        result = run_git(["branch", branch, "origin/main"], cwd=str(main_dir), check=False)
        if result.returncode != 0:
            # Branch might already exist — that's OK
            if "already exists" not in result.stderr:
                output_json({"error": "Failed to create branch", "detail": result.stderr.strip()})
                sys.exit(1)
        work_dir = str(main_dir)
    else:
        # Create worktree (which also creates/reuses the branch)
        setup_script = main_dir / "scripts" / "worktree-setup.sh"
        result = subprocess.run(
            ["bash", str(setup_script), str(issue)],
            capture_output=True, text=True, cwd=str(main_dir),
        )
        if result.returncode != 0:
            output_json({"error": "Worktree setup failed", "detail": result.stderr.strip() or result.stdout.strip()})
            sys.exit(1)
        work_dir = str(get_worktree_dir(issue))

    title = args.title or f"Issue #{issue}"

    # Build next-step command for /create-plan
    skill_parts = [f"/create-plan --issue {issue}"]
    if args.story_key:
        skill_parts.append(f"--story-key {args.story_key}")
    if args.plan:
        skill_parts.append(f"--plan {args.plan}")
    skill_prompt = " ".join(skill_parts)

    next_step = build_next_step(
        work_dir, issue, title, skill_prompt=skill_prompt,
    )

    output_json({
        "issue_number": issue,
        "branch": branch,
        "worktree_path": work_dir,
        "no_worktree": args.no_worktree,
        **next_step,
    })


if __name__ == "__main__":
    main()
