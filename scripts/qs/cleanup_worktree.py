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

    # Also delete the local QS_<N> branch (QS-340 — the epic × factory
    # lane, so a re-entry starts fresh from origin/main)
    python scripts/qs/cleanup_worktree.py --work-dir /path --issue 42 --force --delete-branch
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


_PROTECTED_BRANCHES = frozenset({"main", "master"})


def delete_local_branch(main_wt: Path, branch: str) -> tuple[bool, str | None]:
    """``git branch -D <branch>`` in the main worktree; ``(deleted, error)``.

    Runs with ``cwd=main_wt`` so it works even when the process cwd was
    the (now removed) worktree. Refuses ``main`` / ``master``.
    """
    if branch in _PROTECTED_BRANCHES:
        return False, f"refusing to delete protected branch: {branch}"
    result = subprocess.run(
        ["git", "-C", str(main_wt), "branch", "-D", branch],
        capture_output=True,
        text=True,
        cwd=str(main_wt),
        check=False,
    )
    if result.returncode != 0:
        return False, result.stderr.strip() or f"git branch -D {branch} exited {result.returncode}"
    return True, None


def _current_branch(work_dir: Path) -> str | None:
    """The worktree's checked-out branch, or ``None`` if it can't be read (S3d)."""
    result = subprocess.run(
        ["git", "-C", str(work_dir), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _retry_branch_delete(work_dir: Path, branch_name: str, issue: int) -> None:
    """S3(c): the worktree dir is already gone — prune the stale registration and
    delete the local branch so a failed first run can be retried."""
    branch_error: str | None = None
    branch_deleted = False
    try:
        main_wt: Path | None = get_main_worktree()
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        main_wt = None
        branch_error = f"Could not determine main worktree: {exc}"
    if main_wt is not None:
        subprocess.run(
            ["git", "-C", str(main_wt), "worktree", "prune"],
            capture_output=True,
            text=True,
            cwd=str(main_wt),
            check=False,
        )
        branch_deleted, branch_error = delete_local_branch(main_wt, branch_name)
    status = "removed" if branch_deleted else "removed-branch-kept"
    message = (
        f"Worktree directory was already gone; branch {branch_name} deleted."
        if branch_deleted
        else f"Worktree directory was already gone; branch {branch_name} kept: {branch_error}"
    )
    output_json({
        "status": status,
        "worktree_path": str(work_dir),
        "worktree_remove_error": None,
        "branch_deleted": branch_deleted,
        "branch_delete_error": branch_error,
        "message": message,
    })


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description="Clean up a git worktree.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="Skip safety checks.")
    parser.add_argument("--push-first", action="store_true", help="Push before cleanup.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan; don't act.")
    parser.add_argument(
        "--delete-branch",
        action="store_true",
        help="Also delete the local QS_<N> branch after removing the worktree.",
    )
    args = parser.parse_args()
    branch_name = f"QS_{args.issue}"

    work_dir = Path(args.work_dir).resolve()

    if args.dry_run:
        output_json({
            "status": "dry_run",
            "would_remove_worktree": str(work_dir),
            "would_push": bool(args.push_first),
            "would_delete_branch": branch_name if args.delete_branch else None,
            "issue": args.issue,
        })
        return

    if not work_dir.exists():
        if args.delete_branch:
            # S3(c): a failed first run left the dir gone but the branch behind
            # — prune and delete so cleanup can be retried instead of dead-ending.
            _retry_branch_delete(work_dir, branch_name, args.issue)
            return
        output_json({
            "status": "error",
            "message": f"Worktree directory does not exist: {work_dir}",
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

    # S3(d): read the worktree's checked-out branch BEFORE removal so a
    # mismatched --issue can't force-delete an unrelated QS_<N> branch.
    main_wt: Path | None = None
    branch_error: str | None = None
    mismatch = False
    if args.delete_branch:
        checked_out = _current_branch(work_dir)
        if checked_out is not None and checked_out != branch_name:
            mismatch = True
            branch_error = (
                f"refusing to delete {branch_name}: worktree is checked out on "
                f"{checked_out!r} (mismatched --issue)"
            )
        else:
            # Resolve the main worktree BEFORE the removal: afterwards the
            # process cwd may be gone and `git worktree list` dies (QS-340).
            try:
                main_wt = get_main_worktree()
            except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
                branch_error = f"Could not determine main worktree: {exc}"

    wt_error = remove_worktree(work_dir)

    branch_deleted = False
    # S3(a): a failed worktree removal masks its own error if we press on to the
    # branch — only delete when removal succeeded and the branch is safe.
    if args.delete_branch and not wt_error and not mismatch and main_wt is not None:
        branch_deleted, branch_error = delete_local_branch(main_wt, branch_name)

    if wt_error:
        status = "error"
        message = f"Worktree removal failed: {wt_error}"
    elif args.delete_branch and not branch_deleted:
        # S3(b): the branch was requested but kept — a distinct, honest status.
        status = "removed-branch-kept"
        message = f"Worktree QS_{args.issue} removed; branch {branch_name} kept: {branch_error}"
    else:
        status = "removed"
        message = f"Worktree QS_{args.issue} fully cleaned up."
    output_json({
        "status": status,
        "worktree_path": str(work_dir),
        "worktree_remove_error": wt_error,
        "branch_deleted": branch_deleted,
        "branch_delete_error": branch_error,
        "message": message,
    })


if __name__ == "__main__":  # pragma: no cover
    main()
