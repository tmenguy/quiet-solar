#!/usr/bin/env python3
"""Discover task context for the current worktree.

Static agents call this once on startup to learn everything they need
about the current task: issue, title, branch, story file, PR number,
worktree path, and harness.

Usage::

    python scripts/qs/context.py            # JSON to stdout
    python scripts/qs/context.py --issue 42 # force a specific issue

Source of truth:

- ``issue``: parsed from ``git branch --show-current`` (``QS_<N>``)
- ``title``: from ``gh issue view <N>``
- ``story_file``: ``docs/stories/QS-<N>.story.md`` (if it exists)
- ``pr_number``: from ``gh pr list --head <branch>`` (if open)
- ``worktree``: current working directory
- ``harness``: from :mod:`scripts.qs.harness`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from harness import detect as detect_harness  # type: ignore[import-not-found]

from utils import (  # type: ignore[import-not-found]
    find_latest_review_fix,
    find_pr_for_branch,
    find_story_file,
    get_current_branch,
    get_issue_from_branch,
    get_repo_root,
    output_json,
    run_gh,
)


def _issue_title(issue: int) -> str:
    result = run_gh(["issue", "view", str(issue), "--json", "title", "-q", ".title"], check=False)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def build_context(issue_override: int | None = None) -> dict:
    """Assemble the context dictionary for the current task."""
    branch = get_current_branch()
    issue = issue_override or get_issue_from_branch(branch)

    title = _issue_title(issue) if issue else ""

    story_path: Path | None = find_story_file(issue) if issue else None
    review_fix_path: Path | None = find_latest_review_fix(issue) if issue else None

    pr_info = find_pr_for_branch(branch) if branch else None

    return {
        "harness": detect_harness(),
        "branch": branch,
        "issue": issue,
        "title": title,
        "story_file": str(story_path) if story_path else "",
        "story_exists": bool(story_path),
        "latest_review_fix": str(review_fix_path) if review_fix_path else "",
        "pr_number": pr_info["pr_number"] if pr_info else None,
        "pr_url": pr_info["url"] if pr_info else "",
        "worktree": str(get_repo_root()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover the current task's context.")
    parser.add_argument("--issue", type=int, default=None, help="Force a specific issue number.")
    args = parser.parse_args()

    ctx = build_context(issue_override=args.issue)
    output_json(ctx)
    sys.exit(0)


if __name__ == "__main__":
    main()
