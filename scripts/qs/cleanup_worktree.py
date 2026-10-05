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

    # Clean a work item's worktree QS_<N>_<k> (QS-400 D5); with --delete-branch
    # the item branch goes only when fully integrated into QS_<N>, unless
    # --discard-unintegrated
    python scripts/qs/cleanup_worktree.py --work-dir /path --issue 42 --item 3 --delete-branch
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from utils import (  # type: ignore[import-not-found]
    get_main_worktree,
    output_json,
    positive_int,
    task_branch_name,
)


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
    """Un-register and delete the worktree; return an error string or None.

    M1: the unconditional ``shutil.rmtree`` fallback only runs once ``work_dir``
    is proven to be a **registered linked worktree** of this repo. Ownership is
    proven against the *main worktree's* porcelain registration — never by a
    ``rev-parse`` inside ``work_dir``, which answers for whatever repo contains
    the path. So:

    - if the main worktree can't be determined, nothing is touched (a rmtree
      with no ownership proof is exactly the hazard) (c);
    - the main checkout itself is refused — ``git worktree remove`` fails on it
      and the fallback would otherwise delete it, ``.git`` and all (a);
    - a path with no linked-worktree registration (a subdirectory, an unrelated
      clone, a stray directory) is refused (a/b).

    S4: when git already dropped the registration (``absent``) but left the
    directory behind — a partly-failed ``git worktree remove`` (a
    permission-locked subdirectory, say) — a *second* ownership proof is
    accepted: a ``<work_dir>/.git`` **file** whose ``gitdir:`` line resolves
    (its target may already be pruned) into this repo's ``worktrees/`` admin
    directory. Only then is the leftover ``rmtree``'d, with no
    ``git worktree remove``. A missing ``.git``, a ``.git`` directory (a clone
    or the main checkout) or a gitdir pointing at another repo all keep refusing.

    N1: the main worktree is resolved via ``_main_worktree_or_fallback`` so a cwd
    inside the just-removed worktree does not break resolution.
    """
    try:
        main_wt = _main_worktree_or_fallback().resolve()
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        # M1 (c): without the main worktree we cannot prove ownership.
        return f"Could not determine main worktree: {exc}"

    if work_dir.resolve() == main_wt:
        # M1 (a): the main checkout is never a linked worktree — refuse.
        return f"refusing to remove {work_dir}: it is the main checkout, not a linked worktree"

    kind, _registered = _registered_branch(main_wt, work_dir)
    if kind == "absent":
        # S4: git may have dropped the registration but failed to delete the dir
        # (M1 regression — before M1 the rmtree fallback cleaned this up). Remove
        # the leftover only when its .git proves it was this repo's worktree.
        if _abandoned_worktree_gitdir(main_wt, work_dir):
            leftover_error: str | None = None
            if work_dir.exists():
                try:
                    shutil.rmtree(work_dir)
                except OSError as exc:
                    leftover_error = f"shutil.rmtree failed: {exc}"
            _prune_worktrees(main_wt)
            return leftover_error
        return (
            "refusing to remove "
            f"{work_dir}: it is not a registered linked worktree (absent) — "
            "nothing was touched"
        )
    if kind == "unreadable":
        # M1 (a/b): the listing itself failed — ownership can't be proven.
        return (
            "refusing to remove "
            f"{work_dir}: it is not a registered linked worktree (unreadable) — "
            "nothing was touched"
        )

    error: str | None = None
    result = subprocess.run(
        ["git", "-C", str(main_wt), "worktree", "remove", str(work_dir), "--force"],
        capture_output=True,
        text=True,
        cwd=str(main_wt),
    )
    if result.returncode != 0:
        error = result.stderr.strip() or f"git worktree remove exited {result.returncode}"

    if work_dir.exists():
        # M1 (b): ownership is proven, so the rmtree fallback is safe here.
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

    Uses ``git branch --show-current`` (N1): a detached HEAD prints nothing, and
    unlike ``rev-parse --abbrev-ref HEAD`` it is never confused into
    ``heads/QS_N`` by a same-named tag. An empty result (detached HEAD) or a
    failed read both map to ``None`` — "unknown", to be resolved against the
    worktree registration, not mistaken for a branch and treated as a mismatch.
    """
    result = subprocess.run(
        ["git", "-C", str(work_dir), "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


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
    # N5: compare resolved paths, not raw strings — the porcelain path and the
    # caller's ``work_dir`` can differ byte-for-byte yet name the same directory
    # (a symlinked component, ``/var`` vs ``/private/var`` on macOS, …).
    target = work_dir.resolve()
    current: Path | None = None
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            current = Path(line[len("worktree ") :]).resolve()
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


def _main_worktree_or_fallback() -> Path:
    """The main worktree — resolved from the cwd, else from this script (N2).

    ``get_main_worktree()`` runs ``git worktree list`` in the *process cwd*. In
    the retry path the cwd is often the just-removed worktree, so it dies. Fall
    back to running the same listing anchored at this script's own repo
    (``scripts/qs/`` → repo root two levels up). If both fail, re-raise the
    original error so the caller can report it.

    The fallback only helps when the script runs from **outside** the removed
    worktree (its own repo checkout is intact); a cwd inside a vanished worktree
    is exactly the case it rescues.
    """
    try:
        return get_main_worktree()
    except (RuntimeError, subprocess.CalledProcessError, OSError):
        anchor = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "-C", str(anchor), "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(anchor),
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("worktree "):
                    return Path(line[len("worktree ") :])
        raise


def _git_common_dir(main_wt: Path) -> Path | None:
    """The main worktree's shared git common dir (``<main>/.git``), or ``None``."""
    result = subprocess.run(
        ["git", "-C", str(main_wt), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        cwd=str(main_wt),
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    common = Path(result.stdout.strip())
    if not common.is_absolute():
        common = main_wt / common
    return Path(os.path.realpath(common))


def _abandoned_worktree_gitdir(main_wt: Path, work_dir: Path) -> bool:
    """S4: whether ``work_dir`` is the leftover directory of a *this-repo* linked
    worktree whose porcelain registration git already dropped.

    Proof: a ``<work_dir>/.git`` **file** whose ``gitdir:`` line resolves (its
    target may already be pruned) to ``<main common dir>/worktrees/<id>``. A
    missing ``.git``, a ``.git`` directory (a clone or the main checkout) or a
    gitdir pointing at another repo all return ``False`` — nothing is removed.
    """
    dotgit = work_dir / ".git"
    if not dotgit.is_file():
        return False
    try:
        text = dotgit.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    prefix = "gitdir:"
    line = next((ln.strip() for ln in text.splitlines() if ln.strip().startswith(prefix)), None)
    if line is None:
        return False
    raw = line[len(prefix) :].strip()
    if not raw:
        return False
    gitdir = Path(raw)
    if not gitdir.is_absolute():
        gitdir = work_dir / gitdir
    common = _git_common_dir(main_wt)
    if common is None:
        return False
    expected = common / "worktrees"
    # realpath resolves the existing prefix even when the leaf is already pruned.
    resolved = Path(os.path.realpath(gitdir))
    if resolved.parent != Path(os.path.realpath(expected)):
        return False
    # S1 (#05): a genuine leftover from a partially-failed ``git worktree remove``
    # has had its admin dir deleted by git already. A worktree relocated with plain
    # ``mv`` (not ``git worktree move``) or copied keeps a *live* admin dir
    # registered under its old path; its new path looks 'absent' yet its gitdir
    # still resolves here. Require the admin dir to be gone so such a live,
    # uncommitted-work-bearing directory is never rmtree'd.
    return not resolved.exists()


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
    """S4/S1: the worktree dir is already gone (and ``--force`` was given). Prune
    the stale registration and delete the local branch so a failed first run can
    be retried — but only after **the porcelain registration proves** the branch
    belonged to this worktree. Only ``kind == "branch"`` with a matching
    ``registered`` deletes: a ``detached`` registration proves nothing (S1), a
    registration on another branch is a mismatch, and an absent/unreadable one
    proves nothing either — all keep the branch."""
    try:
        main_wt = _main_worktree_or_fallback()
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
    if kind == "branch" and registered == branch_name:
        # Proven ours — prune the stale registration and delete the branch.
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
        return

    if kind == "branch":  # registered != branch_name — mismatched --issue.
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

    if kind == "detached":
        # S1: prune the stale registration, but a detached registration proves
        # nothing about ownership — keep the branch (as the live path does).
        _prune_worktrees(main_wt)
        _emit(
            work_dir,
            status="removed-branch-kept",
            wt_error=None,
            fields={
                "branch_delete_error": (
                    f"registration was detached; nothing proves it belonged to {branch_name}"
                )
            },
            message=f"Worktree directory was already gone; branch {branch_name} kept (detached registration).",
        )
        return

    # kind in ("absent", "unreadable")
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
    parser.add_argument(
        "--item",
        type=positive_int,
        default=None,
        help="Clean the work item QS_<N>_<K>'s worktree instead of the task's (QS-400).",
    )
    parser.add_argument(
        "--discard-unintegrated",
        action="store_true",
        help="With --item --delete-branch: delete the item branch even if not integrated into QS_<N>.",
    )
    args = parser.parse_args()
    branch_name = f"QS_{args.issue}"

    work_dir = Path(args.work_dir).resolve()

    if args.item is not None and args.push_first:
        output_json({"status": "error", "message": "--push-first is not supported with --item"})
        return
    if args.discard_unintegrated and (args.item is None or not args.delete_branch):
        output_json({"status": "error", "message": "--discard-unintegrated requires --item and --delete-branch"})
        return

    if args.dry_run:
        dry: dict = {
            "status": "dry_run",
            "would_remove_worktree": str(work_dir),
            "would_push": bool(args.push_first),
            "would_delete_branch": task_branch_name(args.issue, args.item) if args.delete_branch else None,
            "issue": args.issue,
        }
        if args.item is not None:
            dry["item"] = args.item
        output_json(dry)
        return

    if args.item is not None:
        _cleanup_item(
            work_dir,
            args.issue,
            args.item,
            force=args.force,
            delete_branch=args.delete_branch,
            discard=args.discard_unintegrated,
        )
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
        if wt_error and work_dir.exists():
            _emit(
                work_dir,
                status="error",
                wt_error=wt_error,
                fields={},
                message=f"Worktree removal failed: {wt_error}",
            )
            return
        if wt_error:
            # N2: remove_worktree flagged an error but the directory is actually
            # gone — prune the stale registration and report success with the
            # error noted, rather than dead-ending on a spurious failure.
            try:
                _prune_worktrees(_main_worktree_or_fallback())
            except (RuntimeError, subprocess.CalledProcessError, OSError):
                pass
        _emit(
            work_dir,
            status="removed",
            wt_error=wt_error,
            fields={},
            message=f"Worktree QS_{args.issue} fully cleaned up.",
        )
        return

    _cleanup_with_branch(work_dir, args.issue, branch_name)


def _cleanup_with_branch(work_dir: Path, issue: int, branch_name: str) -> None:  # noqa: C901
    """Remove a live worktree and (safely) its local branch (M1 / S3 / N4).

    Ownership is proven **primarily** by the porcelain worktree registration in
    the main worktree (M1 (d)): the registration must place ``work_dir`` on
    ``branch_name`` (or be ``detached``). ``rev-parse`` HEAD is only a secondary
    cross-check that gates the *branch deletion*. This keeps a mistyped
    ``--issue``, the main checkout, a subdirectory, or an unrelated clone from
    ever being removed.
    """
    try:
        main_wt = _main_worktree_or_fallback().resolve()
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        # M1 (c): without the main worktree we cannot prove ownership — refuse.
        # N4: a worktree-level refusal belongs in ``worktree_remove_error``.
        main_wt_error = f"Could not determine main worktree: {exc}"
        _emit(
            work_dir,
            status="error",
            wt_error=main_wt_error,
            fields={},
            message=f"Refusing to remove {work_dir}: {main_wt_error}.",
        )
        return

    if work_dir.resolve() == main_wt:
        # M1 (a): never remove the main checkout. N4: worktree-level refusal.
        _emit(
            work_dir,
            status="error",
            wt_error=(
                f"refusing to remove {work_dir}: it is the main checkout, not a linked worktree"
            ),
            fields={},
            message=f"Refusing to remove {work_dir}: it is the main checkout.",
        )
        return

    kind, registered = _registered_branch(main_wt, work_dir)
    if kind in ("absent", "unreadable"):
        # M1 (a/b): not a registered linked worktree (a subdirectory, an
        # unrelated clone, a stray path) — refuse and touch nothing. N4: this
        # is a worktree-level refusal, and the message tells the user what to do.
        _emit(
            work_dir,
            status="error",
            wt_error=(
                f"refusing to remove {work_dir}: it is not a registered linked "
                f"worktree ({kind}) — nothing was touched"
            ),
            fields={},
            message=(
                f"Refusing to remove {work_dir}: not a registered linked worktree ({kind}). "
                "Inspect and delete it manually if it is a stale leftover."
            ),
        )
        return
    if kind == "branch" and registered != branch_name:
        # S3/M1 (d): the registration is on another branch — mismatched --issue.
        # N4: worktree-level refusal.
        _emit(
            work_dir,
            status="error",
            wt_error=(
                f"refusing to remove worktree: it is registered on {registered!r}, "
                f"not {branch_name} (mismatched --issue) — nothing was touched"
            ),
            fields={},
            message=f"Refusing to remove {work_dir}: registered on {registered!r}, not {branch_name}.",
        )
        return
    if kind == "detached" and work_dir.name != branch_name:
        # S3 (#04): a detached registration is not proof of --issue ownership.
        # Only remove it when the directory follows the QS_<N> naming convention
        # of worktree-setup.sh — else a mistyped --issue could force-remove
        # another task's detached worktree. N4: worktree-level refusal.
        _emit(
            work_dir,
            status="error",
            wt_error=(
                f"refusing to remove worktree: it has a detached registration and its "
                f"directory name {work_dir.name!r} is not {branch_name} (mismatched "
                "--issue) — nothing was touched"
            ),
            fields={},
            message=(
                f"Refusing to remove {work_dir}: detached registration and dir name "
                f"is not {branch_name}."
            ),
        )
        return

    # Ownership is proven by the registration (kind == "branch" and matches, or
    # "detached"). The branch is deleted only when the secondary HEAD cross-check
    # also confirms it — a detached/unreadable HEAD, or a detached registration,
    # keeps the branch (S3).
    checked_out = _current_branch(work_dir)
    delete_branch_ok = kind == "branch" and checked_out == branch_name

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

    if wt_error:
        # N4: remove_worktree reported an error but the dir is actually gone —
        # prune the stale registration and continue to the guarded branch delete.
        _prune_worktrees(main_wt)

    if not delete_branch_ok:
        # S3/N3: the branch is kept on purpose — report the *actual* reason rather
        # than a fixed "(unknown HEAD)". A detached registration, an unreadable
        # HEAD and a HEAD that disagrees with the registration are distinct.
        if kind == "detached":
            reason = "detached registration"
        elif checked_out is None:
            reason = "HEAD unreadable"
        else:
            reason = f"HEAD `{checked_out}` does not match the registration"
        _emit(
            work_dir,
            status="removed-branch-kept",
            wt_error=wt_error,
            fields={"branch_delete_error": f"branch {branch_name} kept: {reason}"},
            message=f"Worktree {work_dir} removed; branch {branch_name} kept ({reason}).",
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


# --- QS-400 D5: work-item cleanup -------------------------------------------


def _unintegrated_count(git_dir: Path, deliverable: str, item: str) -> int:
    """Commits on ``item`` not reachable from ``deliverable`` (merges included).

    ``git rev-list --count refs/heads/<deliverable>..refs/heads/<item>``; ``-1``
    on a missing ref or any git failure. Module-level so tests can patch it.
    """
    result = subprocess.run(
        ["git", "-C", str(git_dir), "rev-list", "--count", f"refs/heads/{deliverable}..refs/heads/{item}"],
        capture_output=True,
        text=True,
        cwd=str(git_dir),
        check=False,
    )
    if result.returncode != 0:
        return -1
    try:
        return int(result.stdout.strip())
    except ValueError:
        return -1


def _item_registration(main_wt: Path, item_branch: str) -> Path | None:
    """The (resolved) path of the worktree checking out ``refs/heads/<item_branch>``.

    ``None`` when no worktree has it checked out, or when the listing fails.
    """
    result = subprocess.run(
        ["git", "-C", str(main_wt), "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=str(main_wt),
        check=False,
    )
    if result.returncode != 0:
        return None
    current: Path | None = None
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            current = Path(line[len("worktree ") :])
        elif line == f"branch refs/heads/{item_branch}" and current is not None:
            return current.resolve()
    return None


_ITEM_FORCE_OPTION = {
    "--force": "discard uncommitted files / the detached HEAD's commits; the branch is never deleted by --force"
}


def _cleanup_item(  # noqa: C901
    work_dir: Path,
    issue: int,
    item: int,
    *,
    force: bool,
    delete_branch: bool,
    discard: bool,
) -> None:
    """Remove a work item's worktree, then (``--delete-branch``) its branch (D5).

    Two independent steps, each with its own proof:

    1. the worktree step removes ``work_dir`` only when its registration proves
       it belongs to the item (on ``QS_<N>_<k>``, or detached in a directory
       named ``QS_<N>_<k>``) — never bypassed, whatever the flags — and only
       when it is clean and not detached, unless ``--force``;
    2. the branch step deletes ``QS_<N>_<k>`` only when fully integrated into
       ``QS_<N>`` (``rev-list --count`` is 0), unless ``discard``.

    ``error`` / ``action_required`` stop the call before the branch step. Emits
    one JSON object with every key always present.
    """
    item_branch = task_branch_name(issue, item)
    deliverable = task_branch_name(issue)
    out: dict = {
        "status": "removed",
        "message": "",
        "worktree_path": str(work_dir),
        "worktree_removed": False,
        "worktree_absent": False,
        "worktree_remove_error": None,
        "stale_directory": None,
        "branch": item_branch,
        "branch_deleted": False,
        "branch_absent": False,
        "branch_kept_reason": None,
        "branch_delete_error": None,
        "unintegrated_commits": None,
        "deleted_tip": None,
        "uncommitted_files": [],
        "detached": False,
        "options": {},
    }

    def finish(status: str, message: str) -> None:
        out["status"] = status
        out["message"] = message
        output_json(out)

    # 1. Main worktree and the item's registration.
    try:
        main_wt = _main_worktree_or_fallback().resolve()
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        finish("error", f"Refusing to touch {work_dir}: could not determine main worktree: {exc}")
        return
    reg_path = _item_registration(main_wt, item_branch)

    if work_dir.exists():
        # 2. Worktree step — ownership first, never bypassed.
        if work_dir == main_wt:
            finish("error", f"Refusing to touch {work_dir}: it is the main checkout — nothing was touched")
            return
        kind, registered = _registered_branch(main_wt, work_dir)
        owned = (kind == "branch" and registered == item_branch) or (
            kind == "detached" and work_dir.name == item_branch
        )
        if owned:
            out["detached"] = kind == "detached"
            if not force:
                status_result = subprocess.run(
                    ["git", "-C", str(work_dir), "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if status_result.returncode != 0:
                    detail = status_result.stderr.strip() or f"git status exited {status_result.returncode}"
                    finish("error", f"Could not read the status of {work_dir}: {detail} — nothing was touched")
                    return
                uncommitted = [ln.strip() for ln in status_result.stdout.splitlines() if ln.strip()]
                if uncommitted or kind == "detached":
                    out["uncommitted_files"] = uncommitted
                    out["options"] = dict(_ITEM_FORCE_OPTION)
                    reasons = []
                    if uncommitted:
                        reasons.append(f"{len(uncommitted)} uncommitted file(s)")
                    if kind == "detached":
                        reasons.append(f"a detached HEAD (not on {item_branch})")
                    finish(
                        "action_required",
                        f"Item worktree {work_dir} has {' and '.join(reasons)}; re-run with --force to discard.",
                    )
                    return
            wt_error = remove_worktree(work_dir)
            if wt_error and work_dir.exists():
                out["worktree_remove_error"] = wt_error
                finish("error", f"Item worktree removal failed: {wt_error}")
                return
            out["worktree_removed"] = True
            out["worktree_remove_error"] = wt_error
            if wt_error:
                _prune_worktrees(main_wt)  # N2: the dir is gone, clear the stale registration
        elif kind == "absent" and work_dir.name == item_branch and reg_path is None:
            out["stale_directory"] = str(work_dir)
        else:
            if kind == "branch":
                desc = f"registered on {registered!r}, not {item_branch}"
            elif kind == "detached":
                desc = f"a detached worktree whose directory name {work_dir.name!r} is not {item_branch}"
            elif kind == "unreadable":
                desc = "its registration is unreadable"
            elif reg_path is not None:
                desc = f"not a registered worktree ({item_branch} is checked out at {reg_path})"
            else:
                desc = f"not a registered worktree, and its directory name is not {item_branch}"
            finish("error", f"Refusing to touch {work_dir}: {desc} — nothing was touched")
            return
    else:
        # 3. The directory is gone.
        if reg_path is not None and reg_path != work_dir:
            finish(
                "error",
                f"{item_branch} is checked out at {reg_path}, not at {work_dir} — nothing was touched",
            )
            return
        out["worktree_absent"] = True
        _prune_worktrees(main_wt)

    if out["worktree_removed"]:
        wt_part = f"Item worktree {work_dir} removed"
    elif out["stale_directory"]:
        wt_part = (
            f"{work_dir} is a stale unregistered leftover, left on disk (remove it by hand, or "
            f"`worktree-setup.sh {issue} {item}` recovers it)"
        )
    else:
        wt_part = f"Item worktree {work_dir} already gone"

    if not delete_branch:
        finish("removed", f"{wt_part}; branch {item_branch} not requested for deletion.")
        return

    # 4. Branch step.
    if not _branch_exists(main_wt, item_branch):
        out["branch_absent"] = True
        finish("removed", f"{wt_part}; branch {item_branch} already absent.")
        return
    deliverable_missing = not _branch_exists(main_wt, deliverable)
    n = _unintegrated_count(main_wt, deliverable, item_branch)
    if n >= 0:
        out["unintegrated_commits"] = n

    if not discard:
        if deliverable_missing:
            out["branch_kept_reason"] = "deliverable-missing"
            finish("removed-branch-kept", f"{wt_part}; {deliverable} is gone, cannot prove integration; branch kept")
            return
        if n > 0:
            out["branch_kept_reason"] = "unintegrated"
            finish(
                "removed-branch-kept",
                f"{wt_part}; {item_branch} has {n} commit(s) not in {deliverable}; branch kept",
            )
            return
        if n == -1:
            out["branch_kept_reason"] = "count-failed"
            finish(
                "removed-branch-kept",
                f"{wt_part}; could not count {item_branch}'s commits not in {deliverable}; branch kept",
            )
            return

    tip_result = subprocess.run(
        ["git", "-C", str(main_wt), "rev-parse", "--verify", "--quiet", f"refs/heads/{item_branch}"],
        capture_output=True,
        text=True,
        cwd=str(main_wt),
        check=False,
    )
    out["deleted_tip"] = tip_result.stdout.strip() or None
    deleted, error = delete_local_branch(main_wt, item_branch)
    if not deleted:
        out["branch_kept_reason"] = "delete-failed"
        out["branch_delete_error"] = error
        finish("removed-branch-kept", f"{wt_part}; git branch -D {item_branch} failed: {error}; branch kept")
        return
    out["branch_deleted"] = True
    finish(
        "removed",
        f"{wt_part}; branch {item_branch} deleted (undo: git branch {item_branch} {out['deleted_tip']}).",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
