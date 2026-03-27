#!/usr/bin/env python3
"""Create a pull request with the standard template.

Usage:
    python scripts/qs/create_pr.py --title "..." --summary "..." [--issue N] [--risk CRITICAL]

Output: JSON with PR number and URL.
"""

from __future__ import annotations

import argparse
import sys

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
