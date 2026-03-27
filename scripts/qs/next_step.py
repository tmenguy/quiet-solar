#!/usr/bin/env python3
"""Build unified next-step commands for skill transitions.

Usage:
    python scripts/qs/next_step.py --skill review-story --issue 42 --pr 5 --work-dir DIR --title TITLE

Output: JSON with same_context and new_context commands.
"""

from __future__ import annotations

import argparse

from utils import claude_launch_command, output_json


def build_skill_prompt(skill: str, *, issue: int | None = None, pr: int | None = None,
                       story_file: str | None = None, story_key: str | None = None) -> str:
    """Build the /skill --args string for same-context use."""
    parts = [f"/{skill}"]
    if issue is not None:
        parts.append(f"--issue {issue}")
    if pr is not None:
        parts.append(f"--pr {pr}")
    if story_file:
        parts.append(f"--story-file {story_file}")
    if story_key:
        parts.append(f"--story-key {story_key}")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build next-step commands for skill transitions")
    parser.add_argument("--skill", required=True, help="Target skill name (e.g., review-story)")
    parser.add_argument("--issue", type=int, default=None, help="GitHub issue number")
    parser.add_argument("--pr", type=int, default=None, help="PR number")
    parser.add_argument("--story-file", default=None, help="Path to story file")
    parser.add_argument("--story-key", default=None, help="Story key like 3.2")
    parser.add_argument("--work-dir", required=True, help="Worktree directory path")
    parser.add_argument("--title", required=True, help="Issue/story title for display")
    args = parser.parse_args()

    # Use issue or PR for the tab title; prefer issue, fall back to PR
    issue = args.issue or args.pr or 0

    same_context = build_skill_prompt(
        args.skill,
        issue=args.issue,
        pr=args.pr,
        story_file=args.story_file,
        story_key=args.story_key,
    )

    new_context = claude_launch_command(
        args.work_dir, issue, args.title, prompt=same_context,
    )

    output_json({
        "same_context": same_context,
        "new_context": new_context,
    })


if __name__ == "__main__":
    main()
