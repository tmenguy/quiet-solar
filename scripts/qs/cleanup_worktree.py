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
    """The worktree's checked-out branch, or ``None`` when it is unknown (S3).

    Returns ``None`` for a detached HEAD (``rev-parse --abbrev-ref`` prints the
    literal ``HEAD``) or when the read fails at all — both are "unknown", to be
    resolved against the worktree registration, not mistaken for a branch named
    ``HEAD`` and treated as a mismatch.
    """
    result = subprocess.run(
        ["git", "-C", str(work_dir), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return None if branch in ("", "HEAD") else branch


def _branch_exists(main_wt: Path, branch: str) -> bool:
    """Whether ``refs/heads/<branch>`` exists in the main worktree (S4)."""
    result = subprocess.run(
        ["git", "-C", str(main_wt), "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
        capture_output=True,
        text=True,
        cwd=str(main_wt),
        check=False,
    )
    return result.returncode == 0


def _registered_branch(main_wt: Path, work_dir: Path) -> tuple[str, str | None]:
    """Look ``work_dir`` up in ``git worktree list --porcelain`` (S3/S4).

    Returns ``(kind, branch)`` where ``kind`` is:
    ``"branch"`` (checks out ``branch``), ``"detached"``, ``"absent"`` (no
    entry for ``work_dir``) or ``"unreadable"`` (the listing itself failed).
    """
    result = subprocess.run(
        ["git", "-C", str(main_wt), "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=str(main_wt),
        check=False,
    )
    if result.returncode != 0:
        return "unreadable", None
    target = str(work_dir)
    current: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            current = line[len("worktree ") :]
        elif current == target and line.startswith("branch "):
            ref = line[len("branch ") :]
            short = ref[len("refs/heads/") :] if ref.startswith("refs/heads/") else ref
            return "branch", short
        elif current == target and line == "detached":
            return "detached", None
    return "absent", None


def _delete_branch_after_removal(main_wt: Path, branch: str) -> dict:
    """Guarded ``QS_<N>`` branch delete (S4). Returns ``branch_deleted``,
    ``branch_delete_error``, ``branch_absent`` and ``checked_out_elsewhere``.

    An already-absent branch is reported as deleted (a re-run after a fully
    successful first run is not a failure). A ``branch -D`` that fails because
    the branch is checked out in another worktree is flagged distinctly.
    """
    if not _branch_exists(main_wt, branch):
        return {
            "branch_deleted": True,
            "branch_delete_error": None,
            "branch_absent": True,
            "checked_out_elsewhere": False,
        }
    deleted, error = delete_local_branch(main_wt, branch)
    checked_out = (
        not deleted
        and error is not None
        and ("used by worktree" in error or "checked out at" in error)
    )
    return {
        "branch_deleted": deleted,
        "branch_delete_error": error,
        "branch_absent": False,
        "checked_out_elsewhere": checked_out,
    }


def _prune_worktrees(main_wt: Path) -> None:
    subprocess.run(
        ["git", "-C", str(main_wt), "worktree", "prune"],
        capture_output=True,
        text=True,
        cwd=str(main_wt),
        check=False,
    )


def _emit(work_dir: Path, *, status: str, wt_error: str | None, fields: dict, message: str) -> None:
    """One place that writes the cleanup result JSON."""
    output_json({
        "status": status,
        "worktree_path": str(work_dir),
        "worktree_remove_error": wt_error,
        "branch_deleted": fields.get("branch_deleted", False),
        "branch_delete_error": fields.get("branch_delete_error"),
        "branch_absent": fields.get("branch_absent", False),
        "message": message,
    })


def _retry_branch_delete(work_dir: Path, branch_name: str) -> None:
    """S4: the worktree dir is already gone (and ``--force`` was given). Prune the
    stale registration and delete the local branch so a failed first run can be
    retried — but only after proving the registration belonged to this branch.
    Nothing proves a gone dir with no registration ever belonged to ``QS_<N>``,
    so that (and a registration on another branch) is refused."""
    try:
        main_wt: Path | None = get_main_worktree()
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        _emit(
            work_dir,
            status="removed-branch-kept",
            wt_error=None,
            fields={"branch_delete_error": f"Could not determine main worktree: {exc}"},
            message=f"Worktree directory was already gone; branch {branch_name} kept.",
        )
        return

    kind, registered = _registered_branch(main_wt, work_dir)
    if kind == "branch" and registered != branch_name:
        _emit(
            work_dir,
            status="error",
            wt_error=None,
            fields={
                "branch_delete_error": (
                    f"refusing to delete {branch_name}: the stale worktree "
                    f"registration for {work_dir} is on {registered!r} (mismatched --issue)"
                )
            },
            message=f"Refusing to touch {branch_name}: worktree registration is on {registered!r}.",
        )
        return
    if kind in ("absent", "unreadable"):
        _emit(
            work_dir,
            status="removed-branch-kept",
            wt_error=None,
            fields={
                "branch_delete_error": (
                    f"no worktree registration for {work_dir}; nothing proves it "
                    f"belonged to {branch_name}, so the branch is kept"
                )
            },
            message=f"Worktree directory was already gone; branch {branch_name} kept (no registration).",
        )
        return

    # kind == "detached", or "branch" with registered == branch_name: safe to
    # prune and delete.
    _prune_worktrees(main_wt)
    result = _delete_branch_after_removal(main_wt, branch_name)
    if result["checked_out_elsewhere"]:
        status = "branch-checked-out-elsewhere"
    elif result["branch_deleted"]:
        status = "removed"
    else:
        status = "removed-branch-kept"
    if result["branch_deleted"]:
        message = f"Worktree directory was already gone; branch {branch_name} deleted."
    else:
        message = (
            f"Worktree directory was already gone; branch {branch_name} kept: "
            f"{result['branch_delete_error']}"
        )
    _emit(work_dir, status=status, wt_error=None, fields=result, message=message)


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
        if args.delete_branch and args.force:
            # S4: a failed first run left the dir gone but the branch behind.
            # ``--force`` is required (as for discarding a live worktree) — prune
            # and delete so cleanup can be retried instead of dead-ending.
            _retry_branch_delete(work_dir, branch_name)
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

    if not args.delete_branch:
        wt_error = remove_worktree(work_dir)
        if wt_error:
            _emit(
                work_dir,
                status="error",
                wt_error=wt_error,
                fields={},
                message=f"Worktree removal failed: {wt_error}",
            )
            return
        _emit(
            work_dir,
            status="removed",
            wt_error=None,
            fields={},
            message=f"Worktree QS_{args.issue} fully cleaned up.",
        )
        return

    _cleanup_with_branch(work_dir, args.issue, branch_name)


def _cleanup_with_branch(work_dir: Path, issue: int, branch_name: str) -> None:  # noqa: C901
    """Remove a live worktree and (safely) its local branch (S3 / N4)."""
    # S3: read the worktree's checked-out branch BEFORE removal so a mismatched
    # --issue can't force-delete or force-remove an unrelated QS_<N> worktree.
    checked_out = _current_branch(work_dir)
    try:
        main_wt: Path | None = get_main_worktree()
        main_wt_error: str | None = None
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        main_wt = None
        main_wt_error = f"Could not determine main worktree: {exc}"

    if checked_out is not None and checked_out != branch_name:
        # S3: a real mismatched branch — refuse entirely and touch nothing.
        _emit(
            work_dir,
            status="error",
            wt_error=None,
            fields={
                "branch_delete_error": (
                    f"refusing to remove worktree: it is checked out on {checked_out!r}, "
                    f"not {branch_name} (mismatched --issue) — nothing was touched"
                )
            },
            message=f"Refusing to remove {work_dir}: checked out on {checked_out!r}, not {branch_name}.",
        )
        return

    # ``delete_branch_ok`` stays True only when the HEAD read confirmed
    # ``branch_name``. An unknown HEAD (detached/unreadable) never deletes the
    # branch, but may still allow removing the worktree once ownership is
    # confirmed via the porcelain registration (S3).
    delete_branch_ok = checked_out == branch_name
    if checked_out is None:
        if main_wt is None:
            _emit(
                work_dir,
                status="error",
                wt_error=None,
                fields={"branch_delete_error": main_wt_error},
                message=f"Refusing to remove {work_dir}: HEAD is unknown and {main_wt_error}.",
            )
            return
        kind, registered = _registered_branch(main_wt, work_dir)
        if kind == "branch" and registered != branch_name:
            _emit(
                work_dir,
                status="error",
                wt_error=None,
                fields={
                    "branch_delete_error": (
                        f"refusing to remove worktree: it is registered on {registered!r}, "
                        f"not {branch_name} (mismatched --issue) — nothing was touched"
                    )
                },
                message=f"Refusing to remove {work_dir}: registered on {registered!r}, not {branch_name}.",
            )
            return
        if kind in ("absent", "unreadable"):
            _emit(
                work_dir,
                status="error",
                wt_error=None,
                fields={
                    "branch_delete_error": (
                        f"refusing to remove worktree: HEAD is unknown and its ownership by "
                        f"{branch_name} cannot be confirmed ({kind} registration)"
                    )
                },
                message=f"Refusing to remove {work_dir}: unknown HEAD, {kind} registration.",
            )
            return
        # kind == "detached", or "branch" with registered == branch_name: safe to
        # remove the worktree, but keep the branch (HEAD was unknown).

    wt_error = remove_worktree(work_dir)
    if wt_error and work_dir.exists():
        # S3: a genuine removal failure with the dir still present — do not press
        # on to the branch (that would mask the removal error).
        _emit(
            work_dir,
            status="error",
            wt_error=wt_error,
            fields={},
            message=f"Worktree removal failed: {wt_error}",
        )
        return

    if wt_error and main_wt is not None:
        # N4: remove_worktree reported an error but the dir is actually gone —
        # prune the stale registration and continue to the guarded branch delete.
        _prune_worktrees(main_wt)

    if not delete_branch_ok:
        # S3: the branch is kept on purpose (unknown HEAD).
        _emit(
            work_dir,
            status="removed-branch-kept",
            wt_error=wt_error,
            fields={
                "branch_delete_error": (
                    f"branch {branch_name} kept: the worktree HEAD was detached or unreadable"
                )
            },
            message=f"Worktree {work_dir} removed; branch {branch_name} kept (unknown HEAD).",
        )
        return

    if main_wt is None:
        _emit(
            work_dir,
            status="removed-branch-kept",
            wt_error=wt_error,
            fields={"branch_delete_error": main_wt_error},
            message=f"Worktree QS_{issue} removed; branch {branch_name} kept: {main_wt_error}",
        )
        return

    result = _delete_branch_after_removal(main_wt, branch_name)
    if result["checked_out_elsewhere"]:
        status = "branch-checked-out-elsewhere"
        message = (
            f"Worktree {work_dir} removed; branch {branch_name} is checked out in "
            f"another worktree: {result['branch_delete_error']}"
        )
    elif result["branch_deleted"]:
        status = "removed"
        message = f"Worktree QS_{issue} fully cleaned up."
    else:
        status = "removed-branch-kept"
        message = f"Worktree QS_{issue} removed; branch {branch_name} kept: {result['branch_delete_error']}"
    _emit(work_dir, status=status, wt_error=wt_error, fields=result, message=message)


if __name__ == "__main__":  # pragma: no cover
    main()
