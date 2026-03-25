#!/usr/bin/env python3
"""Create a GitHub issue for a story or bug.

Usage:
    python scripts/qs/create_issue.py --title "..." [--body "..."] [--story-key 3.2] [--labels bug]

Output: JSON with issue number and URL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from utils import find_story_file, output_json, run_gh


def main() -> None:
    parser = argparse.ArgumentParser(description="Create GitHub issue")
    parser.add_argument("--title", required=True, help="Issue title")
    parser.add_argument("--body", default="", help="Issue body text")
    parser.add_argument("--story-key", default=None, help="Story key like 3.2 to pull context from story file")
    parser.add_argument("--story-file", default=None, help="Path to story file")
    parser.add_argument("--labels", default="", help="Comma-separated labels")
    args = parser.parse_args()

    body = args.body

    # If story-key or story-file provided, build body from story
    story_path = None
    if args.story_file:
        story_path = Path(args.story_file)
    elif args.story_key:
        story_path = find_story_file(args.story_key)

    if story_path and story_path.exists():
        story_content = story_path.read_text()
        # Extract description (first non-frontmatter paragraph)
        in_frontmatter = False
        desc_lines = []
        for line in story_content.split("\n"):
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                continue
            if not in_frontmatter and line.strip() and not line.startswith("#"):
                desc_lines.append(line)
                if len(desc_lines) >= 5:
                    break

        body = f"## Story {args.story_key or ''}\n\n"
        body += "\n".join(desc_lines) if desc_lines else "See story file for details."
        body += f"\n\n---\nCreated from story: `{story_path.name}`"

    # Build gh command
    cmd = ["issue", "create", "--title", args.title, "--body", body]
    if args.labels:
        cmd.extend(["--label", args.labels])

    result = run_gh(cmd, check=False)
    if result.returncode != 0:
        output_json({"error": "Failed to create issue", "detail": result.stderr.strip()})
        sys.exit(1)

    # Parse issue URL from output
    url = result.stdout.strip()
    # Extract issue number from URL like https://github.com/user/repo/issues/42
    issue_number = int(url.rstrip("/").split("/")[-1])

    output_json({
        "issue_number": issue_number,
        "url": url,
        "branch": f"QS_{issue_number}",
        "title": args.title,
    })


if __name__ == "__main__":
    main()
