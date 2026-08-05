#!/usr/bin/env python3
"""Fetch an existing GitHub issue's details.

Usage:
    python scripts/qs/fetch_issue.py --issue 42

Output: JSON with issue number, title, body, labels, the parsed lane
axes (``kind``/``target``/``scale``/``lane``), ``parent_epic`` and
``declaration_complete`` (QS-332). The axis parsing and the
declaration truth table live in :mod:`targets` — one table, three
consumers.
"""

from __future__ import annotations

import argparse
import json
import sys

import targets

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

    # `or []` / `or ""` (QS-332 review-fix #02): the API can return a
    # present-but-null field (`"labels": null`, `"body": null`) — a bare
    # `.get` default doesn't catch that, and iterating `None` /
    # `parse_parent_epic(None)` raised a raw traceback instead of this
    # script's structured error JSON. Same guard as `context.py` and
    # `create_pr.py` (the fix-plan-#01 pattern, applied to the third
    # consumer it missed).
    label_names = [lb["name"] for lb in data.get("labels") or []]
    body = data.get("body") or ""
    axes = targets.parse_axes(label_names)
    declaration_ok, _missing, _message = targets.validate_declaration(label_names)

    output_json({
        "issue_number": data["number"],
        "title": data.get("title", ""),
        "body": body,
        "labels": label_names,
        "state": data.get("state", ""),
        "kind": axes["kind"],
        "target": axes["target"],
        "scale": axes["scale"],
        "lane": axes["lane"],
        "parent_epic": targets.parse_parent_epic(body),
        "declaration_complete": declaration_ok,
        "branch": f"QS_{data['number']}",
    })


if __name__ == "__main__":
    main()
