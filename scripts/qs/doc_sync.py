#!/usr/bin/env python3
"""Compare story artifact against actual implementation for doc-sync gate.

Usage:
    python scripts/qs/doc_sync.py <story_file> [--base-branch main] [--repo-path PATH] [--json]

Reads the story artifact, extracts tasks and acceptance criteria,
compares against the git diff, and reports discrepancies.

Output: JSON report of tasks, ACs, changed files, and discrepancies.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from utils import find_story_file, output_json, run_git


def parse_story(story_path: Path) -> dict:
    """Parse a story markdown file to extract tasks and acceptance criteria."""
    content = story_path.read_text()

    # Extract acceptance criteria
    acs: list[dict] = []
    ac_section = re.search(
        r"## Acceptance Criteria\s*\n(.*?)(?=\n## |\Z)", content, re.DOTALL
    )
    if ac_section:
        # Match numbered ACs with **Given** format
        for match in re.finditer(
            r"(\d+)\.\s+\*\*Given\*\*\s+(.*?)(?=\n\d+\.\s+\*\*Given\*\*|\n## |\Z)",
            ac_section.group(1),
            re.DOTALL,
        ):
            acs.append({"number": int(match.group(1)), "text": match.group(2).strip()})

        # F16: Fallback — if no **Given** ACs found, count plain numbered items
        if not acs:
            for match in re.finditer(r"^(\d+)\.\s+(.+)$", ac_section.group(1), re.MULTILINE):
                acs.append({"number": int(match.group(1)), "text": match.group(2).strip()})

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
                    "text": sub.group(2).strip(),
                    "done": sub.group(1).lower() == "x",
                })

            task_line = match.group(2).split("\n")[0].strip()
            tasks.append({
                "text": task_line,
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


def get_changed_files(base_branch: str, *, cwd: str | None = None) -> tuple[list[str], str]:
    """Get files changed between base branch and HEAD.

    Returns (file_list, error_message). error_message is empty on success.
    """
    result = run_git(
        ["diff", "--name-only", f"{base_branch}...HEAD"],
        check=False,
        cwd=cwd,
    )
    if result.returncode != 0:
        first_error = result.stderr.strip()
        # Try without three-dot (branch might not exist locally)
        result = run_git(
            ["diff", "--name-only", base_branch, "HEAD"],
            check=False,
            cwd=cwd,
        )
    if result.returncode != 0:
        git_error = result.stderr.strip() or first_error
        return [], f"git diff failed: {git_error}"
    return [f for f in result.stdout.strip().split("\n") if f], ""


def get_test_files(changed_files: list[str]) -> list[str]:
    """Filter changed files to just test files."""
    return [f for f in changed_files if f.startswith("tests/") and f.endswith(".py")]


def find_discrepancies(story: dict, changed_files: list[str], git_error: str) -> list[dict]:
    """Compare story artifact against implementation and find discrepancies."""
    discrepancies: list[dict] = []
    test_files = get_test_files(changed_files)
    src_files = [f for f in changed_files if f.startswith("custom_components/")]
    skill_files = [f for f in changed_files if f.startswith("_qsprocess/")]
    doc_files = [f for f in changed_files if f.startswith("_bmad-output/")]

    tasks_done = [t for t in story["tasks"] if t["done"]]
    tasks_not_done = [t for t in story["tasks"] if not t["done"]]

    if tasks_not_done:
        discrepancies.append({
            "type": "tasks_incomplete",
            "severity": "warning",
            "message": f"{len(tasks_not_done)} task(s) not marked as done",
            "details": [t["text"] for t in tasks_not_done],
        })

    # F1: Check tasks marked done but no corresponding file changes
    if tasks_done and not changed_files and not git_error:
        discrepancies.append({
            "type": "tasks_done_no_changes",
            "severity": "warning",
            "message": f"{len(tasks_done)} task(s) marked done but no files changed",
            "details": [t["text"] for t in tasks_done],
        })

    # F2: Check for scope changes not reflected in docs
    if skill_files and not doc_files:
        discrepancies.append({
            "type": "scope_change_undocumented",
            "severity": "warning",
            "message": "Process/skill files changed but no documentation files updated",
            "details": skill_files,
        })

    # F8: Use specific phrase for process story detection
    is_process_story = "no production" in story["dev_notes_excerpt"].lower() or \
                       "process/tooling story" in story["dev_notes_excerpt"].lower()
    if is_process_story and src_files:
        discrepancies.append({
            "type": "unexpected_src_changes",
            "severity": "info",
            "message": "Story marked as process/tooling but production code was changed",
            "details": src_files,
        })

    # F10: Merged missing_tests and acs_without_test_evidence into single check
    if src_files and not test_files:
        ac_detail = [f"AC#{ac['number']}" for ac in story["acceptance_criteria"]] if story["acceptance_criteria"] else []
        discrepancies.append({
            "type": "missing_tests",
            "severity": "warning",
            "message": "Source files changed but no test files were modified"
                       + (f" ({len(ac_detail)} ACs lack test evidence)" if ac_detail else ""),
            "details": src_files + (["ACs without test evidence: " + ", ".join(ac_detail)] if ac_detail else []),
        })

    # F9: Report git errors instead of silently returning empty
    if git_error:
        discrepancies.append({
            "type": "git_error",
            "severity": "error",
            "message": f"Could not determine changed files: {git_error}",
            "details": [],
        })
    elif not changed_files:
        discrepancies.append({
            "type": "no_changes",
            "severity": "error",
            "message": "No files changed compared to base branch",
            "details": [],
        })

    return discrepancies


def main() -> None:
    parser = argparse.ArgumentParser(description="Doc-sync gate: compare story vs implementation")
    parser.add_argument("story_file", nargs="?", default=None, help="Path to story artifact markdown file")
    parser.add_argument("--issue", type=int, default=None, help="GitHub issue number (alternative to story_file path)")
    parser.add_argument("--base-branch", default="main", help="Base branch to diff against")
    parser.add_argument("--repo-path", default=None, help="Path to repo for git operations (defaults to cwd)")
    parser.add_argument("--json", action="store_true", help="JSON output (default is human-readable)")
    args = parser.parse_args()

    # Resolve story file: --issue takes precedence, then positional path
    story_path: Path | None = None
    if args.issue:
        story_path = find_story_file(args.issue)
    elif args.story_file:
        story_path = Path(args.story_file)

    if story_path is None or not story_path.exists():
        msg = f"Story file not found for issue #{args.issue}" if args.issue else f"Story file not found: {args.story_file}"
        if args.json:
            output_json({"error": msg})
        else:
            print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)

    story = parse_story(story_path)
    # F5: Support --repo-path so the script works from any directory
    changed_files, git_error = get_changed_files(args.base_branch, cwd=args.repo_path)
    discrepancies = find_discrepancies(story, changed_files, git_error)

    # F13: Differentiate gate status by severity
    max_severity = "clean"
    for d in discrepancies:
        if d["severity"] == "error":
            max_severity = "needs_review"
            break
        if d["severity"] == "warning":
            max_severity = "needs_review"
        elif d["severity"] == "info" and max_severity == "clean":
            max_severity = "advisory"

    report = {
        "story_file": str(story_path),
        "tasks_total": len(story["tasks"]),
        "tasks_done": sum(1 for t in story["tasks"] if t["done"]),
        "acceptance_criteria_count": len(story["acceptance_criteria"]),
        "changed_files": changed_files,
        "changed_files_count": len(changed_files),
        "discrepancies": discrepancies,
        "discrepancy_count": len(discrepancies),
        "gate_status": max_severity,
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
