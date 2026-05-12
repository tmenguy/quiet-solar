#!/usr/bin/env python3
"""Clean up a git worktree after a task is done.

Replaces the old multi-step dance with one script that handles safety
checks. Static-agent pipeline has no per-task agent files to remove —
this script only un-registers the worktree and deletes its directory.

Usage::

    # Check + clean (safe mode: aborts if dirty)
    python scripts/qs/cleanup_worktree.py --work-dir /path --issue 42

    # Push first, then clean
    python scripts/qs/cleanup_worktree.py --work-dir /path --issue 42 --push-first

    # Force (lose uncommitted/unpushed changes)
    python scripts/qs/cleanup_worktree.py --work-dir /path --issue 42 --force

    # Dry run
    python scripts/qs/cleanup_worktree.py --work-dir /path --issue 42 --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from utils import get_main_worktree, output_json  # type: ignore[import-not-found]


def check_worktree_status(work_dir: Path) -> dict:
    """Inspect the worktree for uncommitted changes and unpushed commits."""
    branch_result = subprocess.run(
        ["git", "-C", str(work_dir), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
    )
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"

    status_result = subprocess.run(
        ["git", "-C", str(work_dir), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    uncommitted_files = [line.strip() for line in status_result.stdout.splitlines() if line.strip()]

    log_result = subprocess.run(
        ["git", "-C", str(work_dir), "log", "@{u}..HEAD", "--oneline"],
        capture_output=True,
        text=True,
    )
    if log_result.returncode != 0:
        unpushed_commits = -1
    else:
        unpushed_commits = len([line for line in log_result.stdout.splitlines() if line.strip()])

    return {
        "safe_to_remove": not uncommitted_files and unpushed_commits == 0,
        "uncommitted_files": uncommitted_files,
        "unpushed_commits": unpushed_commits,
        "branch": branch,
    }


def push_branch(work_dir: Path) -> tuple[bool, str]:
    """Push the current branch from the worktree."""
    result = subprocess.run(
        ["git", "-C", str(work_dir), "push"],
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip() or result.stderr.strip()
    return result.returncode == 0, output


def remove_worktree(work_dir: Path) -> str | None:
    """Un-register and delete the worktree; return an error string or None."""
    error: str | None = None
    try:
        main_wt = get_main_worktree()
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        error = f"Could not determine main worktree: {exc}"
        main_wt = None

    if main_wt is not None:
        result = subprocess.run(
            ["git", "-C", str(main_wt), "worktree", "remove", str(work_dir), "--force"],
            capture_output=True,
            text=True,
            cwd=str(main_wt),
        )
        if result.returncode != 0:
            error = result.stderr.strip() or f"git worktree remove exited {result.returncode}"

    if work_dir.exists():
        try:
            shutil.rmtree(work_dir)
        except OSError as exc:
            rmtree_err = f"shutil.rmtree failed: {exc}"
            error = f"{error}; {rmtree_err}" if error else rmtree_err

    return error


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description="Clean up a git worktree.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="Skip safety checks.")
    parser.add_argument("--push-first", action="store_true", help="Push before cleanup.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan; don't act.")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()

    if not work_dir.exists():
        output_json({
            "status": "error",
            "message": f"Worktree directory does not exist: {work_dir}",
        })
        return

    if args.dry_run:
        output_json({
            "status": "dry_run",
            "would_remove_worktree": str(work_dir),
            "would_push": bool(args.push_first),
            "issue": args.issue,
        })
        return

    if not args.force:
        status = check_worktree_status(work_dir)

        if not status["safe_to_remove"] and not args.push_first:
            output_json({
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
            })
            return

        if args.push_first and status["uncommitted_files"]:
            output_json({
                "status": "action_required",
                "safe_to_remove": False,
                "uncommitted_files": status["uncommitted_files"],
                "unpushed_commits": status["unpushed_commits"],
                "branch": status["branch"],
                "message": (
                    "Worktree has uncommitted files that --push-first cannot save. "
                    "Commit them first, or use --force to discard."
                ),
                "options": {"--force": "Delete worktree and lose all uncommitted changes"},
            })
            return

    if args.push_first:
        ok, push_output = push_branch(work_dir)
        if not ok:
            output_json({
                "status": "error",
                "message": f"Push failed: {push_output}",
            })
            return

    wt_error = remove_worktree(work_dir)
    status = "error" if wt_error else "removed"
    message = (
        f"Worktree removal failed: {wt_error}"
        if wt_error
        else f"Worktree QS_{args.issue} fully cleaned up."
    )
    output_json({
        "status": status,
        "worktree_path": str(work_dir),
        "worktree_remove_error": wt_error,
        "message": message,
    })


if __name__ == "__main__":  # pragma: no cover
    main()
