#!/usr/bin/env python3
"""Create a pull request with the standard template.

Usage:
    python scripts/qs/create_pr.py --title "..." --summary "..." [--issue N] [--risk CRITICAL]

Output: JSON with PR number and URL.

QS-332 (B5) — two machine-owned additions, both fed by ONE
``gh issue view N --json body,labels`` call:

- **auto-``Refs``**: a parent epic declared in the issue body
  (``targets.parse_parent_epic`` — explicit, never guessed) is appended
  inside the fixes-line slot (``Fixes #N`` then ``Refs #E``) so a child
  PR never auto-closes its epic. No CLI override flags by design
  (review SG2-01): the escape hatch for a wrong ``Refs`` is editing the
  issue body, the declared source of truth.
- **Lane note**: the changed files are classified against the issue's
  declared ``target:*`` label; when the diff crosses targets, a
  ``## Lane note`` section listing the crossing files verbatim is
  injected into the PR body. Recomputed from ``targets.classify`` — no
  sentinel-string parsing of gate output (review N-4/DP2-02).

A failing/unparseable issue lookup degrades to today's body (no Refs,
no note) rather than blocking the PR.
"""

from __future__ import annotations

import argparse
import json
import sys

import targets

from utils import (
    detect_risk_level,
    find_pr_for_branch,
    get_changed_files,
    get_current_branch,
    get_issue_from_branch,
    output_json,
    run_gh,
    run_git,
)


def _epic_and_lane_note(issue: int, changed: list[str]) -> tuple[int | None, str]:
    """Return ``(parent_epic, lane_note_section)`` for the PR body.

    One ``gh issue view --json body,labels`` call feeds both. Any failure
    (non-zero exit, bad JSON) degrades to ``(None, "")`` — the PR body
    stays byte-identical to today's.
    """
    result = run_gh(["issue", "view", str(issue), "--json", "body,labels"], check=False)
    if result.returncode != 0:
        return None, ""
    try:
        data = json.loads(result.stdout)
        issue_body = data.get("body") or ""
        # `or []` (QS-332 review-fix #03): a `"labels": null` response is
        # valid JSON and its BODY is perfectly readable — the bare
        # `.get(..., [])` raised `TypeError` into the broad `except`
        # below and silently dropped the parent-epic `Refs` (and any
        # Lane note) for no good reason.
        labels = [lb["name"] for lb in data.get("labels") or []]
    except (json.JSONDecodeError, TypeError, KeyError):
        return None, ""

    epic = targets.parse_parent_epic(issue_body)

    declared = targets.parse_axes(labels)["target"]
    lane_section = ""
    if declared:
        opposite = "product" if declared == "factory" else "factory"
        crossing = [f for f in changed if targets.classify(f) == opposite]
        if crossing:
            file_list = "".join(f"- `{f}`\n" for f in crossing)
            lane_section = (
                "\n## Lane note\n"
                f"This task declares `target:{declared}` but the diff touches "
                f"{opposite}-classified files:\n\n"
                f"{file_list}\n"
                f"Verify these serve the declared {declared} purpose "
                "(purpose, not path, is the classifier). If the "
                f"{opposite}-side portion is substantial, consider splitting "
                "the issue.\n"
            )
    return epic, lane_section


def main() -> None:
    parser = argparse.ArgumentParser(description="Create pull request")
    parser.add_argument("--title", required=True, help="PR title (under 70 chars)")
    parser.add_argument("--summary", required=True, help="1-3 bullet summary")
    parser.add_argument("--issue", type=int, default=None, help="GitHub issue number")
    parser.add_argument("--risk", default=None, help="Risk level override: CRITICAL, HIGH, MEDIUM, LOW")
    args = parser.parse_args()

    branch = get_current_branch()
    issue = args.issue or get_issue_from_branch(branch)

    # Check for existing PR to prevent duplicates
    existing = find_pr_for_branch(branch)
    if existing:
        output_json({
            "pr_number": existing["pr_number"],
            "url": existing["url"],
            "branch": branch,
            "issue": issue,
            "already_existed": True,
        })
        return

    # Push branch
    push_result = run_git(["push", "-u", "origin", branch], check=False)
    if push_result.returncode != 0:
        output_json({"error": "Push failed", "detail": push_result.stderr.strip()})
        sys.exit(1)

    # Detect risk from changed files
    changed = get_changed_files()
    risks = [args.risk] if args.risk else detect_risk_level(changed)

    # Build risk checkboxes
    risk_lines = []
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        checked = "x" if level in risks else " "
        labels = {
            "CRITICAL": "solver, constraints, charger budgeting",
            "HIGH": "load base, constants, orchestration",
            "MEDIUM": "device-specific: car, person, battery, solar",
            "LOW": "platforms, UI, docs",
        }
        risk_lines.append(f"- [{checked}] {level} ({labels[level]})")

    # Build PR body
    fixes_line = f"\nFixes #{issue}\n" if issue else ""
    lane_section = ""
    if issue:
        epic, lane_section = _epic_and_lane_note(issue, changed)
        if epic is not None:
            fixes_line = f"\nFixes #{issue}\nRefs #{epic}\n"
    body = f"""## Summary
{args.summary}
{fixes_line}
## Testing
- [x] Tests added/updated for new behavior
- [x] 100% coverage verified
- [x] No flaky tests introduced

## Code quality
- [x] Ruff passes (lint + format)
- [x] MyPy passes
- [x] No new `# type: ignore` or `noqa` without justification

## Risk assessment
{chr(10).join(risk_lines)}
{lane_section}
---
Generated with [Claude Code](https://claude.com/claude-code)"""

    cmd = ["pr", "create", "--title", args.title, "--body", body]
    result = run_gh(cmd, check=False)

    if result.returncode != 0:
        output_json({"error": "PR creation failed", "detail": result.stderr.strip()})
        sys.exit(1)

    url = result.stdout.strip()
    pr_number = int(url.rstrip("/").split("/")[-1])

    output_json({
        "pr_number": pr_number,
        "url": url,
        "branch": branch,
        "issue": issue,
        "risks": risks,
    })


if __name__ == "__main__":
    main()
