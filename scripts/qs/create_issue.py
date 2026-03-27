#!/usr/bin/env python3
"""Create a GitHub issue for a story or bug.

Usage:
    python scripts/qs/create_issue.py --title "..." [--body "..."] [--labels bug]

Output: JSON with issue number and URL.
"""

from __future__ import annotations

import argparse
import sys

from utils import output_json, run_gh


def main() -> None:
    parser = argparse.ArgumentParser(description="Create GitHub issue")
    parser.add_argument("--title", required=True, help="Issue title")
    parser.add_argument("--body", default="", help="Issue body text")
    parser.add_argument("--labels", default="", help="Comma-separated labels")
    args = parser.parse_args()

    # Build gh command
    cmd = ["issue", "create", "--title", args.title, "--body", args.body]
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
