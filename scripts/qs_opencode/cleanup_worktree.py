#!/usr/bin/env python3
"""Clean up a git worktree: remove per-task agents, un-register, delete directory.

Replaces the multi-step cleanup dance (cleanup_agents -> git worktree remove ->
hope the dir is gone) with a single script that handles safety checks.

Usage::

    # Check + clean (safe mode -- aborts if dirty)
    python scripts/qs_opencode/cleanup_worktree.py \
        --work-dir /path/to/worktree --issue 42

    # Push first, then clean
    python scripts/qs_opencode/cleanup_worktree.py \
        --work-dir /path/to/worktree --issue 42 --push-first

    # Force clean (lose uncommitted/unpushed changes)
    python scripts/qs_opencode/cleanup_worktree.py \
        --work-dir /path/to/worktree --issue 42 --force

    # Dry run
    python scripts/qs_opencode/cleanup_worktree.py \
        --work-dir /path/to/worktree --issue 42 --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from utils import get_main_worktree, output_json  # type: ignore[import-not-found]


def check_worktree_status(work_dir: Path) -> dict:
    """Check for uncommitted changes and unpushed commits.

    Returns a dict with:
        safe_to_remove: bool
        uncommitted_files: list[str]
        unpushed_commits: int
        branch: str
    """
    result = subprocess.run(
        ["git", "-C", str(work_dir), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
    )
    branch = result.stdout.strip() if result.returncode == 0 else "unknown"

    result = subprocess.run(
        ["git", "-C", str(work_dir), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    uncommitted_files = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    result = subprocess.run(
        ["git", "-C", str(work_dir), "log", "@{u}..HEAD", "--oneline"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        unpushed_commits = -1  # no upstream tracking
    else:
        unpushed_lines = [line for line in result.stdout.splitlines() if line.strip()]
        unpushed_commits = len(unpushed_lines)

    safe = len(uncommitted_files) == 0 and unpushed_commits == 0

    return {
        "safe_to_remove": safe,
        "uncommitted_files": uncommitted_files,
        "unpushed_commits": unpushed_commits,
        "branch": branch,
    }


def remove_agent_files(work_dir: Path, issue: int) -> list[str]:
    """Remove per-task agent files (inline logic from cleanup_agents.py)."""
    agent_dir = work_dir / ".opencode" / "agents"
    if not agent_dir.is_dir():
        return []

    pattern = f"qs-*-QS-{issue}.md"
    matches = sorted(agent_dir.glob(pattern))
    removed: list[str] = []
    for path in matches:
        rel = str(path.relative_to(work_dir))
        try:
            path.unlink()
            removed.append(rel)
        except OSError as exc:
            removed.append(f"FAILED {rel}: {exc}")
    return removed


def list_agent_files(work_dir: Path, issue: int) -> list[str]:
    """List per-task agent files without removing them (for dry-run)."""
    agent_dir = work_dir / ".opencode" / "agents"
    if not agent_dir.is_dir():
        return []
    return [str(p.relative_to(work_dir)) for p in sorted(agent_dir.glob(f"qs-*-QS-{issue}.md"))]


def remove_worktree(work_dir: Path) -> str | None:
    """Un-register and delete the worktree. Returns error message or None."""
    main_wt = get_main_worktree()
    result = subprocess.run(
        ["git", "-C", str(main_wt), "worktree", "remove", str(work_dir), "--force"],
        capture_output=True,
        text=True,
    )
    error = None
    if result.returncode != 0:
        error = result.stderr.strip()

    # Fallback: physically remove directory if still present
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

    return error


def push_branch(work_dir: Path) -> tuple[bool, str]:
    """Push the current branch. Returns (success, output)."""
    result = subprocess.run(
        ["git", "-C", str(work_dir), "push"],
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip() or result.stderr.strip()
    return result.returncode == 0, output


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Clean up a git worktree (agents + unregister + delete)",
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip safety checks, delete everything",
    )
    parser.add_argument(
        "--push-first",
        action="store_true",
        help="Push current branch before cleanup",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without doing it",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()

    if not work_dir.exists():
        output_json(
            {
                "status": "error",
                "message": f"Worktree directory does not exist: {work_dir}",
            }
        )
        return

    # Step 1: Check status (unless --force)
    if not args.force:
        status = check_worktree_status(work_dir)

        if not status["safe_to_remove"] and not args.push_first:
            output_json(
                {
                    "status": "action_required",
                    "safe_to_remove": False,
                    "uncommitted_files": status["uncommitted_files"],
                    "unpushed_commits": status["unpushed_commits"],
                    "branch": status["branch"],
                    "message": "Worktree has uncommitted changes and/or unpushed commits.",
                    "options": {
                        "--force": "Delete worktree and lose all uncommitted/unpushed changes",
                        "--push-first": "Push current branch to origin, then delete worktree",
                    },
                }
            )
            return

        # --push-first only handles unpushed commits; uncommitted files
        # would be silently lost when the worktree is deleted.
        if args.push_first and status["uncommitted_files"]:
            output_json(
                {
                    "status": "action_required",
                    "safe_to_remove": False,
                    "uncommitted_files": status["uncommitted_files"],
                    "unpushed_commits": status["unpushed_commits"],
                    "branch": status["branch"],
                    "message": (
                        "Worktree has uncommitted files that --push-first cannot save. "
                        "Commit them first, or use --force to discard."
                    ),
                    "options": {
                        "--force": "Delete worktree and lose all uncommitted changes",
                    },
                }
            )
            return

    # Step 2: Push if requested
    if args.push_first:
        if args.dry_run:
            output_json(
                {
                    "status": "dry_run",
                    "would_push": True,
                    "would_remove_agents": list_agent_files(work_dir, args.issue),
                    "would_remove_worktree": str(work_dir),
                }
            )
            return
        success, push_output = push_branch(work_dir)
        if not success:
            output_json(
                {
                    "status": "error",
                    "message": f"Push failed: {push_output}",
                }
            )
            return

    if args.dry_run:
        output_json(
            {
                "status": "dry_run",
                "would_remove_agents": list_agent_files(work_dir, args.issue),
                "would_remove_worktree": str(work_dir),
            }
        )
        return

    # Step 3: Remove agent files
    agents_removed = remove_agent_files(work_dir, args.issue)

    # Step 4: Remove worktree
    wt_error = remove_worktree(work_dir)

    output_json(
        {
            "status": "removed",
            "agents_removed": agents_removed,
            "worktree_path": str(work_dir),
            "worktree_remove_error": wt_error,
            "message": f"Worktree QS_{args.issue} fully cleaned up.",
        }
    )


if __name__ == "__main__":  # pragma: no cover
    main()
