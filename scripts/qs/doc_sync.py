#!/usr/bin/env python3
"""Compare story artifact against actual implementation for doc-sync gate.

Usage:
    python scripts/qs/doc_sync.py <story_file> [--base-branch main] [--json]

Reads the story artifact, extracts tasks and acceptance criteria,
compares against the git diff, and reports discrepancies.

Output: JSON report of tasks, ACs, changed files, and discrepancies.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from utils import output_json, run_git


def parse_story(story_path: Path) -> dict:
    """Parse a story markdown file to extract tasks and acceptance criteria."""
    content = story_path.read_text()

    # Extract acceptance criteria
    acs: list[dict] = []
    ac_section = re.search(
        r"## Acceptance Criteria\s*\n(.*?)(?=\n## |\Z)", content, re.DOTALL
    )
    if ac_section:
        # Match numbered ACs (1. **Given** ... or just numbered items)
        for match in re.finditer(
            r"(\d+)\.\s+\*\*Given\*\*\s+(.*?)(?=\n\d+\.\s+\*\*Given\*\*|\n## |\Z)",
            ac_section.group(1),
            re.DOTALL,
        ):
            acs.append({"number": int(match.group(1)), "text": match.group(2).strip()[:200]})

    # Extract tasks with completion status
    tasks: list[dict] = []
    task_section = re.search(
        r"## Tasks\s*/?\s*Subtasks\s*\n(.*?)(?=\n## |\Z)", content, re.DOTALL
    )
    if task_section:
        for match in re.finditer(
            r"- \[([ xX])\] (Task \d+.*?)(?=\n- \[|\Z)",
            task_section.group(1),
            re.DOTALL,
        ):
            done = match.group(1).lower() == "x"
            # Extract subtasks
            subtasks = []
            for sub in re.finditer(
                r"  - \[([ xX])\] (\d+\.\d+.*?)$",
                match.group(2),
                re.MULTILINE,
            ):
                subtasks.append({
                    "id": sub.group(2).split()[0],
                    "text": sub.group(2).strip()[:150],
                    "done": sub.group(1).lower() == "x",
                })

            task_line = match.group(2).split("\n")[0].strip()
            tasks.append({
                "text": task_line[:150],
                "done": done,
                "subtasks": subtasks,
            })

    # Extract dev notes section (for "no production code" hints)
    dev_notes = ""
    notes_section = re.search(
        r"## Dev Notes\s*\n(.*?)(?=\n## |\Z)", content, re.DOTALL
    )
    if notes_section:
        dev_notes = notes_section.group(1).strip()[:500]

    return {
        "file": str(story_path),
        "acceptance_criteria": acs,
        "tasks": tasks,
        "dev_notes_excerpt": dev_notes,
    }


def get_changed_files(base_branch: str) -> list[str]:
    """Get files changed between base branch and HEAD."""
    result = run_git(
        ["diff", "--name-only", f"{base_branch}...HEAD"],
        check=False,
    )
    if result.returncode != 0:
        # Try without three-dot (branch might not exist locally)
        result = run_git(
            ["diff", "--name-only", base_branch, "HEAD"],
            check=False,
        )
    if result.returncode != 0:
        return []
    return [f for f in result.stdout.strip().split("\n") if f]


def get_test_files(changed_files: list[str]) -> list[str]:
    """Filter changed files to just test files."""
    return [f for f in changed_files if f.startswith("tests/") and f.endswith(".py")]


def find_discrepancies(story: dict, changed_files: list[str]) -> list[dict]:
    """Compare story artifact against implementation and find discrepancies."""
    discrepancies: list[dict] = []
    test_files = get_test_files(changed_files)
    src_files = [f for f in changed_files if f.startswith("custom_components/")]
    skill_files = [f for f in changed_files if f.startswith("_qsprocess/")]
    doc_files = [f for f in changed_files if f.startswith("_bmad-output/")]

    # Check: tasks marked done but no corresponding files changed
    all_files = changed_files
    tasks_done = [t for t in story["tasks"] if t["done"]]
    tasks_not_done = [t for t in story["tasks"] if not t["done"]]

    if tasks_not_done:
        discrepancies.append({
            "type": "tasks_incomplete",
            "severity": "warning",
            "message": f"{len(tasks_not_done)} task(s) not marked as done",
            "details": [t["text"] for t in tasks_not_done],
        })

    # Check: if story says "no production code" but src files changed
    is_process_story = "no production" in story["dev_notes_excerpt"].lower() or \
                       "process" in story["dev_notes_excerpt"].lower()
    if is_process_story and src_files:
        discrepancies.append({
            "type": "unexpected_src_changes",
            "severity": "info",
            "message": "Story marked as process/tooling but production code was changed",
            "details": src_files,
        })

    # Check: if src files changed but no test files changed
    if src_files and not test_files:
        discrepancies.append({
            "type": "missing_tests",
            "severity": "warning",
            "message": "Source files changed but no test files were modified",
            "details": src_files,
        })

    # Check: ACs exist but none reference test coverage
    if story["acceptance_criteria"] and not test_files and src_files:
        discrepancies.append({
            "type": "acs_without_test_evidence",
            "severity": "warning",
            "message": "Acceptance criteria defined but no tests added/modified",
            "details": [f"AC#{ac['number']}" for ac in story["acceptance_criteria"]],
        })

    # Summary of what was changed
    if not all_files:
        discrepancies.append({
            "type": "no_changes",
            "severity": "error",
            "message": "No files changed compared to base branch",
            "details": [],
        })

    return discrepancies


def main() -> None:
    parser = argparse.ArgumentParser(description="Doc-sync gate: compare story vs implementation")
    parser.add_argument("story_file", help="Path to story artifact markdown file")
    parser.add_argument("--base-branch", default="main", help="Base branch to diff against")
    parser.add_argument("--json", action="store_true", help="JSON output (default is human-readable)")
    args = parser.parse_args()

    story_path = Path(args.story_file)
    if not story_path.exists():
        output_json({"error": f"Story file not found: {args.story_file}"})
        sys.exit(1)

    story = parse_story(story_path)
    changed_files = get_changed_files(args.base_branch)
    discrepancies = find_discrepancies(story, changed_files)

    report = {
        "story_file": str(story_path),
        "tasks_total": len(story["tasks"]),
        "tasks_done": sum(1 for t in story["tasks"] if t["done"]),
        "acceptance_criteria_count": len(story["acceptance_criteria"]),
        "changed_files": changed_files,
        "changed_files_count": len(changed_files),
        "discrepancies": discrepancies,
        "discrepancy_count": len(discrepancies),
        "gate_status": "needs_review" if discrepancies else "clean",
    }

    if args.json:
        output_json(report)
    else:
        print(f"Story: {story_path.name}")
        print(f"Tasks: {report['tasks_done']}/{report['tasks_total']} done")
        print(f"ACs: {report['acceptance_criteria_count']}")
        print(f"Changed files: {report['changed_files_count']}")
        print()
        if discrepancies:
            print("Discrepancies found:")
            for d in discrepancies:
                severity = d["severity"].upper()
                print(f"  [{severity}] {d['message']}")
                for detail in d["details"][:5]:
                    print(f"           {detail}")
        else:
            print("No discrepancies found. Doc-sync gate is clean.")

    # Exit 0 always — discrepancies are advisory, the agent decides what to block on
    sys.exit(0)


if __name__ == "__main__":
    main()
