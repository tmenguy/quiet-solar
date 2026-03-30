#!/usr/bin/env python3
"""Fetch an existing GitHub issue's details.

Usage:
    python scripts/qs/fetch_issue.py --issue 42

Output: JSON with issue number, title, body, labels, and derived type.
"""

from __future__ import annotations

import argparse
import json
import sys

from utils import output_json, run_gh


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GitHub issue details")
    parser.add_argument("--issue", required=True, type=int, help="GitHub issue number")
    args = parser.parse_args()

    result = run_gh(
        ["issue", "view", str(args.issue), "--json", "number,title,body,labels,state"],
        check=False,
    )
    if result.returncode != 0:
        output_json({"error": f"Failed to fetch issue #{args.issue}", "detail": result.stderr.strip()})
        sys.exit(1)

    try:
        data = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        output_json({"error": "Invalid JSON from gh CLI", "detail": result.stdout.strip()})
        sys.exit(1)

    label_names = [lb["name"] for lb in data.get("labels", [])]

    # Derive story type from labels: bug > enhancement/feature > default (feature)
    if "bug" in label_names:
        story_type = "bug"
    elif any(lb in label_names for lb in ("enhancement", "feature")):
        story_type = "feature"
    else:
        story_type = "feature"

    output_json({
        "issue_number": data["number"],
        "title": data.get("title", ""),
        "body": data.get("body", ""),
        "labels": label_names,
        "state": data.get("state", ""),
        "story_type": story_type,
        "branch": f"QS_{data['number']}",
    })


if __name__ == "__main__":
    main()
