#!/usr/bin/env python3
"""Integrate a work item into its deliverable's branch — the git mechanics (QS-400 §8).

Usage::

    integrate_item.py prepare --issue N --item K --item-tip SHA
    integrate_item.py check   --issue N --item K
    integrate_item.py gate    --issue N --item K --expect-head SHA
    integrate_item.py move    --issue N --item K --new SHA --old SHA
    integrate_item.py drop    --issue N --item K

A plain script, like ``setup_task.py``: git only, one JSON object on stdout.
Exit 0 for every outcome (``status`` says which), exit 1 with
``{"error": "<code>", "detail": ...}`` for a refusal or a failure.

**It takes no Control Plane lock.** It is called only by the Control Plane
tools of ``control_plane/items.py`` (``integrate-start`` / ``-finish`` /
``-drop`` and ``item-cleanup``), which hold the session-held
``integration:QS_<N>`` lock, the ``gates`` slot and the fencing. What this
script adds is a **per-item exclusive file lock** (``flock`` on
``<git common dir>/qs-integration-QS_<N>_<K>.lock``, non-blocking →
``scratch-busy``), held for the whole subcommand — and, for ``gate``, by the
detached gate watchdog that inherits it, so nothing removes a scratch under a
running gate.

Each item has its own scratch worktree ``<repo>-worktrees/QS_<N>_<K>_integration``
(detached, created by ``worktree-setup.sh N K --integration``). Its state file
``qs-integration.json`` lives in the scratch's own git dir. The state is a
hint; git decides. Every subcommand is content-idempotent: run again, it
answers from the git state and does nothing more.

Nothing here pushes (D7): ``move`` only moves the local ``QS_<N>``.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

from utils import get_integration_dir, get_main_worktree, positive_int  # type: ignore[import-not-found]

DEFAULT_GATE_TIMEOUT_S = 3300
TAIL_LINES = 40
STATE_FILE = "qs-integration.json"
GATE_RESULT_FILE = "qs-integration-gate.json"
GATE_LOG_FILE = "qs-integration-gate.log"
DROP_INDEX_FILE = "qs-integration-drop.index"
MARKER_RE = r"^(<<<<<<<|>>>>>>>|\|\|\|\|\|\|\|)( |$)"
OPERATION_PATHS = (
    "MERGE_HEAD",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "rebase-merge",
    "rebase-apply",
    "sequencer",
    "BISECT_LOG",
)
PHASES = ("created", "merged", "conflicts")


def gate_timeout_s() -> int:
    """The gate deadline; ``QS_INTEGRATE_GATE_TIMEOUT_S`` overrides it (tests)."""
    raw = os.environ.get("QS_INTEGRATE_GATE_TIMEOUT_S", "")
    return int(raw) if raw.isascii() and raw.isdigit() and int(raw) > 0 else DEFAULT_GATE_TIMEOUT_S


class Refusal(Exception):
    """A refusal or failure: ``{"error": code, "detail": ..., **extra}``, exit 1."""

    def __init__(self, code: str, detail: str, **extra: Any) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.extra = extra


# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------


def git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",  # raw `-z` paths may not be UTF-8
        check=False,
        env={**os.environ, **env} if env else None,
    )


def rev(cwd: Path, ref: str) -> str | None:
    res = git(cwd, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    return res.stdout.strip() if res.returncode == 0 and res.stdout.strip() else None


def is_ancestor(cwd: Path, ancestor: str, descendant: str) -> bool:
    return git(cwd, "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0


def git_dir(worktree: Path) -> Path:
    res = git(worktree, "rev-parse", "--absolute-git-dir")
    if res.returncode != 0:
        raise Refusal("stale-scratch", f"cannot read the git dir of {worktree}: {res.stderr.strip()}")
    return Path(res.stdout.strip())


def common_dir(main: Path) -> Path:
    res = git(main, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if res.returncode != 0:
        raise Refusal("git-failed", f"cannot read the git common dir: {res.stderr.strip()}")
    return Path(res.stdout.strip())


def unmerged_files(worktree: Path) -> list[str]:
    """Unmerged paths, NUL-separated so a non-ASCII name is never quoted."""
    res = git(worktree, "diff", "--name-only", "-z", "--diff-filter=U")
    return sorted({name for name in res.stdout.split("\0") if name})


def dirty_files(worktree: Path, *, untracked: bool = True) -> list[str]:
    """Changed paths from ``status --porcelain -z`` (unquoted; a rename's source entry skipped)."""
    args = ["status", "--porcelain", "-z", "--untracked-files=all" if untracked else "--untracked-files=no"]
    res = git(worktree, *args)
    if res.returncode != 0:
        raise Refusal("git-failed", f"git status failed in {worktree}: {res.stderr.strip()}")
    files: list[str] = []
    entries = iter(res.stdout.split("\0"))
    for entry in entries:
        if not entry:
            continue
        files.append(entry[3:])
        if "R" in entry[:2] or "C" in entry[:2]:
            next(entries, None)  # the rename / copy source
    return files


def operation_in_progress(worktree: Path) -> str | None:
    gdir = git_dir(worktree)
    for name in OPERATION_PATHS:
        if (gdir / name).exists():
            return name
    return None


def worktrees(main: Path) -> list[tuple[Path, str | None]]:
    """``(path, branch ref or None)`` for every registered worktree."""
    res = git(main, "worktree", "list", "--porcelain")
    if res.returncode != 0:
        raise Refusal("git-failed", f"git worktree list failed: {res.stderr.strip()}")
    out: list[tuple[Path, str | None]] = []
    path: Path | None = None
    branch: str | None = None
    for line in [*res.stdout.splitlines(), ""]:
        if line.startswith("worktree "):
            path, branch = Path(line[len("worktree ") :]), None
        elif line.startswith("branch "):
            branch = line[len("branch ") :]
        elif not line and path is not None:
            out.append((path, branch))
            path = None
    return out


def _same_path(a: Path, b: Path) -> bool:
    return a == b or a.resolve() == b.resolve()


def is_registered(main: Path, path: Path) -> bool:
    return any(_same_path(p, path) for p, _ in worktrees(main))


def checkout_of(main: Path, branch: str) -> Path | None:
    """The worktree holding ``branch``; a registration whose directory is gone does not count."""
    for path, ref in worktrees(main):
        if ref == f"refs/heads/{branch}" and path.is_dir():
            return path
    return None


def _tail(text: str) -> list[str]:
    return text.splitlines()[-TAIL_LINES:]


# ---------------------------------------------------------------------------
# context, lock, state
# ---------------------------------------------------------------------------


class Ctx:
    """Everything a subcommand needs about deliverable ``N`` and item ``K``."""

    def __init__(self, issue: int, item: int) -> None:
        self.issue = issue
        self.item = item
        self.main = get_main_worktree()
        self.deliverable = f"QS_{issue}"
        self.item_branch = f"QS_{issue}_{item}"
        self.scratch = get_integration_dir(issue, item)
        self.lock_path = common_dir(self.main) / f"qs-integration-{self.item_branch}.lock"
        self.lock_fd = -1

    def take_lock(self) -> None:
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise Refusal(
                "scratch-busy",
                f"another integrate_item.py call (or a running gate) holds {self.item_branch}'s scratch; "
                "wait up to max_wait_s, then retry with a new key",
                max_wait_s=gate_timeout_s(),
            ) from None
        self.lock_fd = fd

    def deliverable_tip(self) -> str | None:
        return rev(self.main, f"refs/heads/{self.deliverable}")

    def state_path(self) -> Path:
        return git_dir(self.scratch) / STATE_FILE

    def read_state(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.state_path().read_text(encoding="utf-8"))
        except OSError, ValueError, Refusal:
            return None
        if not isinstance(data, dict) or data.get("phase") not in PHASES:
            return None
        if data.get("issue") != self.issue or data.get("item") != self.item:
            return None
        if not isinstance(data.get("base"), str) or not isinstance(data.get("item_tip"), str):
            return None
        return data

    def write_state(self, state: dict[str, Any]) -> None:
        path = self.state_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def require_state(self) -> dict[str, Any]:
        if not self.scratch.is_dir():
            raise Refusal("no-scratch", f"no integration scratch at {self.scratch}; run prepare first")
        state = self.read_state()
        if state is None:
            raise Refusal(
                "stale-scratch",
                f"{self.scratch} has no readable state for item {self.item}; drop it and start again",
            )
        return state

    def remove_scratch(self) -> None:
        """``git worktree remove -f -f`` (a locked scratch too) + prune; never follows the venv symlink.

        A registration that survives the prune → ``drop-failed``.
        """
        if self.scratch.exists():
            res = git(self.main, "worktree", "remove", "--force", "--force", str(self.scratch))
            if res.returncode != 0 and self.scratch.exists():
                shutil.rmtree(self.scratch)
        git(self.main, "worktree", "prune")
        if is_registered(self.main, self.scratch):
            git(self.main, "worktree", "unlock", str(self.scratch))
            git(self.main, "worktree", "prune")
            if is_registered(self.main, self.scratch):
                raise Refusal("drop-failed", f"{self.scratch} is still registered after git worktree prune")


def _integrated(ctx: Ctx, state: dict[str, Any]) -> tuple[bool, str | None]:
    """The scratch's merge happened and is an ancestor of ``QS_<N>`` (§8 "Integrated")."""
    head = rev(ctx.scratch, "HEAD")
    tip = ctx.deliverable_tip()
    ok = (
        state["phase"] in ("merged", "conflicts")
        and head is not None
        and tip is not None
        and head != state["base"]
        and is_ancestor(ctx.scratch, state["item_tip"], head)
        and is_ancestor(ctx.scratch, head, tip)
        and operation_in_progress(ctx.scratch) is None
    )
    return ok, head


def _deliverable_worktree_ready(ctx: Ctx) -> Path | None:
    """The worktree holding ``QS_<N>``, refused if it has tracked changes or an operation in progress."""
    wt = checkout_of(ctx.main, ctx.deliverable)
    if wt is None:
        return None
    changed = dirty_files(wt, untracked=False)
    op = operation_in_progress(wt)
    if changed or op:
        raise Refusal(
            "deliverable-worktree-busy",
            f"{ctx.deliverable} is checked out at {wt} with "
            + (f"tracked changes {changed}" if changed else f"{op} in progress"),
            worktree=str(wt),
            files=changed,
        )
    return wt


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def _merge_parents(ctx: Ctx, commit: str) -> list[str]:
    res = git(ctx.scratch, "rev-list", "--parents", "-n", "1", commit)
    return res.stdout.split()[1:]


def cmd_prepare(ctx: Ctx, item_tip_arg: str) -> dict[str, Any]:
    tip = ctx.deliverable_tip()
    item_ref = rev(ctx.main, f"refs/heads/{ctx.item_branch}")
    if tip is None or item_ref is None:
        missing = ctx.deliverable if tip is None else ctx.item_branch
        raise Refusal("missing-ref", f"refs/heads/{missing} not found")
    item_tip = rev(ctx.main, item_tip_arg)
    if item_tip != item_ref:
        raise Refusal("item-moved", f"{ctx.item_branch} is at {item_ref}, not {item_tip_arg}", item_tip=item_ref)

    exists = ctx.scratch.is_dir()
    state = ctx.read_state() if exists else None
    result: dict[str, Any] = {"item_tip": item_tip, "scratch": str(ctx.scratch)}
    if state is not None:
        result["base"] = state["base"]

    if is_ancestor(ctx.main, item_tip, tip):
        return {"status": "already-integrated", **result}

    if exists:
        return _resume(ctx, state, tip, item_tip, result)

    setup = subprocess.run(
        ["bash", str(ctx.main / "scripts" / "worktree-setup.sh"), str(ctx.issue), str(ctx.item), "--integration"],
        capture_output=True,
        text=True,
        cwd=str(ctx.main),
        check=False,
    )
    if setup.returncode == 3:
        raise Refusal("stale-scratch", f"an integration scratch is registered at {ctx.scratch}; drop it first")
    if setup.returncode != 0:
        raise Refusal("setup-failed", (setup.stdout + setup.stderr).strip())
    base = rev(ctx.scratch, "HEAD")
    assert base is not None  # just created, detached at QS_<N>
    state = {
        "issue": ctx.issue,
        "item": ctx.item,
        "base": base,
        "item_tip": item_tip,
        "phase": "created",
        "conflicted_files": [],
    }
    ctx.write_state(state)
    result["base"] = base
    return _merge(ctx, state, result)


def _resume(ctx: Ctx, state: dict[str, Any] | None, tip: str, item_tip: str, result: dict[str, Any]) -> dict[str, Any]:
    """Prepare on an existing scratch: resume, repair a provable crash window, or ``stale-scratch``."""
    if state is None:
        raise Refusal("stale-scratch", f"{ctx.scratch} has no readable state for item {ctx.item}; drop it")
    if state["base"] != tip:
        raise Refusal(
            "stale-scratch", f"{ctx.deliverable} moved since the scratch was made; drop it and start again", **result
        )
    if state["item_tip"] != item_tip:
        raise Refusal("stale-scratch", f"the scratch integrates {state['item_tip']}, not {item_tip}; drop it", **result)
    head = rev(ctx.scratch, "HEAD")
    merge_head = rev(ctx.scratch, "MERGE_HEAD")
    base = state["base"]
    phase = state["phase"]
    if phase == "created":
        if merge_head is None and head == base:
            return _merge(ctx, state, result)
        if merge_head is None and head is not None and _merge_parents(ctx, head) == [base, item_tip]:
            state["phase"] = "merged"  # crash window (a): merged, state not rewritten
            ctx.write_state(state)
            return {"status": "merged", **result}
        files = unmerged_files(ctx.scratch)
        if head == base and merge_head == item_tip and files:
            state["phase"], state["conflicted_files"] = "conflicts", files  # crash window (b)
            ctx.write_state(state)
            return {"status": "conflicts", "files": files, **result}
    elif phase == "conflicts" and (merge_head is not None or head != base):
        return {"status": "conflicts", "files": unmerged_files(ctx.scratch), **result}
    elif phase == "merged" and head != base:
        return {"status": "merged", **result}
    raise Refusal("stale-scratch", f"the scratch's state ({phase}) does not match git; drop it", **result)


def _merge(ctx: Ctx, state: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    res = git(
        ctx.scratch,
        "merge",
        "--no-ff",
        "--no-edit",
        "-m",
        f"QS-{ctx.issue}: integrate item {ctx.item}",
        state["item_tip"],
    )
    if res.returncode == 0:
        state["phase"] = "merged"
        ctx.write_state(state)
        return {"status": "merged", **result}
    if rev(ctx.scratch, "MERGE_HEAD") is None:
        detail = (res.stdout + res.stderr).strip()
        try:
            ctx.remove_scratch()
        except Refusal as exc:  # keep the merge's own failure visible
            raise Refusal("merge-failed", detail, cleanup_error=exc.detail, **result) from None
        raise Refusal("merge-failed", detail, **result)
    files = unmerged_files(ctx.scratch)
    state["phase"], state["conflicted_files"] = "conflicts", files
    ctx.write_state(state)
    return {"status": "conflicts", "files": files, **result}


def cmd_check(ctx: Ctx) -> dict[str, Any]:
    state = ctx.require_state()
    if state["phase"] == "created":
        raise Refusal("stale-scratch", "the scratch was never merged (phase created); run prepare again or drop it")
    base, item_tip = state["base"], state["item_tip"]
    integrated, head = _integrated(ctx, state)
    out = {"status": "ready", "head": head, "base": base, "item_tip": item_tip}
    if integrated:
        return {**out, "moved": True}
    if rev(ctx.scratch, "MERGE_HEAD") is not None:
        raise Refusal("merge-in-progress", "conclude the merge (git commit) or drop the scratch")
    unmerged = unmerged_files(ctx.scratch)
    if unmerged:
        raise Refusal("unmerged-paths", "resolve and git add every unmerged path", files=unmerged)
    dirty = dirty_files(ctx.scratch)
    if dirty:
        raise Refusal("dirty-scratch", "commit or discard every change in the scratch", files=dirty)
    if (
        head is None
        or head == base
        or not is_ancestor(ctx.scratch, base, head)
        or not is_ancestor(ctx.scratch, item_tip, head)
    ):
        raise Refusal("not-a-merge", f"HEAD {head} does not contain both {base} and the item tip {item_tip}")
    markers = _conflict_markers(ctx, state.get("conflicted_files") or [])
    if markers:
        raise Refusal("conflict-markers", "conflict markers remain in previously conflicted files", files=markers)
    tip = ctx.deliverable_tip()
    if tip != base:
        raise Refusal("deliverable-moved", f"{ctx.deliverable} is at {tip}, not {base}; drop and start again")
    _deliverable_worktree_ready(ctx)
    return {**out, "moved": False}


def _conflict_markers(ctx: Ctx, files: list[str]) -> list[str]:
    if not files:
        return []
    pathspecs = [f":(literal){name}" for name in files]
    res = git(ctx.scratch, "-c", "core.quotePath=false", "grep", "-l", "-z", "-E", MARKER_RE, "HEAD", "--", *pathspecs)
    return sorted(hit.split(":", 1)[1] for hit in res.stdout.split("\0") if ":" in hit)


def cmd_gate(ctx: Ctx, expect_head: str) -> dict[str, Any]:
    ctx.require_state()
    python = ctx.scratch / "venv" / "bin" / "python"
    if not python.exists():
        raise Refusal("no-venv", f"{python} not found")
    _require_head(ctx, expect_head, "dirty-scratch")

    gdir = git_dir(ctx.scratch)
    result_path, log_path = gdir / GATE_RESULT_FILE, gdir / GATE_LOG_FILE
    result_path.unlink(missing_ok=True)
    deadline = time.time() + gate_timeout_s()
    argv = [str(python), "scripts/qs/quality_gate.py", "--impacted"]
    with log_path.open("w", encoding="utf-8") as log:
        watchdog = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "_gate-run",
                "--lock-fd",
                str(ctx.lock_fd),
                "--scratch",
                str(ctx.scratch),
                "--result",
                str(result_path),
                "--log",
                str(log_path),
                "--deadline",
                repr(deadline),
                "--",
                *argv,
            ],
            start_new_session=True,
            pass_fds=(ctx.lock_fd,),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )
    watchdog.wait()
    try:
        gate = json.loads(result_path.read_text(encoding="utf-8"))
        status = gate["status"]
    except OSError, ValueError, KeyError, TypeError:
        raise Refusal("gate-run-failed", "the gate watchdog wrote no result", tail=_read_tail(log_path)) from None
    if status == "gate-timeout":
        raise Refusal("gate-timeout", f"the gate ran past its {gate_timeout_s()} s deadline and was killed")
    _require_head(ctx, expect_head, "gate-modified-tree")
    if status == "green":
        return {"status": "green"}
    return {"status": "red", "tail": gate.get("tail", [])}


def _require_head(ctx: Ctx, expect_head: str, dirty_code: str) -> None:
    head = rev(ctx.scratch, "HEAD")
    if head is None or head != rev(ctx.scratch, expect_head):
        raise Refusal("head-moved", f"the scratch HEAD is {head}, not {expect_head}", head=head)
    dirty = dirty_files(ctx.scratch)
    if dirty:
        raise Refusal(dirty_code, "the scratch tree is not clean", files=dirty)


def _read_tail(path: Path) -> list[str]:
    try:
        return _tail(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return []


def cmd_gate_run(lock_fd: int, scratch: Path, result: Path, log: Path, deadline: float, argv: list[str]) -> int:
    """The gate watchdog: a detached session leader holding the inherited item lock.

    Runs the gate in its own process group, kills the whole group at the
    deadline, writes the result atomically, then exits — releasing its copy of
    the lock only once the gate's group is gone. ``lock_fd`` is never reopened.
    """
    del log  # stdout/stderr already point at the log (set by the parent)
    proc = subprocess.Popen(
        argv,
        cwd=str(scratch),
        start_new_session=True,
        pass_fds=(lock_fd,),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    try:
        output, _ = proc.communicate(timeout=max(deadline - time.time(), 0))
    except subprocess.TimeoutExpired:
        _killpg(proc.pid)
        proc.wait()
        payload: dict[str, Any] = {"status": "gate-timeout"}
    else:
        _killpg(proc.pid)  # stragglers of the gate's group, if any
        print(output, end="")
        payload = {"status": "green"} if proc.returncode == 0 else {"status": "red", "tail": _tail(output)}
    tmp = result.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, result)
    return 0


def _killpg(pgid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


def cmd_move(ctx: Ctx, new: str, old: str) -> dict[str, Any]:
    ctx.require_state()
    head = rev(ctx.scratch, "HEAD")
    new_sha, old_sha = rev(ctx.main, new), rev(ctx.main, old)
    if head is None or head != new_sha:
        raise Refusal("head-moved", f"the scratch HEAD is {head}, not {new}", head=head)
    tip = ctx.deliverable_tip()
    if tip is not None and is_ancestor(ctx.main, new_sha, tip):
        return {"status": "already-moved", "head": new_sha}
    if tip is None or tip != old_sha:
        raise Refusal("deliverable-moved", f"{ctx.deliverable} is at {tip}, not {old}")
    wt = _deliverable_worktree_ready(ctx)
    if wt is not None:
        res = git(wt, "merge", "--ff-only", new_sha)
        now = ctx.deliverable_tip()
        if now != new_sha:
            if now is None:
                raise Refusal("move-failed", f"{ctx.deliverable} is gone after the fast-forward")
            if is_ancestor(ctx.main, new_sha, now):
                return {"status": "already-moved", "head": new_sha, "now": now}
            if now != old_sha:
                raise Refusal("deliverable-moved", f"{ctx.deliverable} moved during the fast-forward (now {now})")
            raise Refusal("move-failed", res.stderr.strip() or "QS_<N> did not reach the new head")
    else:
        res = git(ctx.main, "update-ref", f"refs/heads/{ctx.deliverable}", new_sha, old_sha)
        if res.returncode != 0:
            now = ctx.deliverable_tip()
            if now is None:
                raise Refusal("move-failed", f"{ctx.deliverable} is gone after the failed update")
            if now != old_sha and is_ancestor(ctx.main, new_sha, now):
                return {"status": "already-moved", "head": new_sha, "now": now}
            if now != old_sha:
                raise Refusal("deliverable-moved", f"{ctx.deliverable} moved during the update")
            raise Refusal("move-failed", res.stderr.strip())
    return {"status": "moved", "head": new_sha}


def _scratch_git_broken(ctx: Ctx) -> bool:
    """The scratch's ``.git`` provably is not this item's worktree (review fix #02 B).

    True when the ``.git`` file is missing (git would then climb to an
    enclosing repository), its ``gitdir:`` target is missing, or it points
    outside ``<common dir>/worktrees/`` or not back at this scratch. Any other
    state is a real worktree: a git failure there re-raises.
    """
    dot_git = ctx.scratch / ".git"
    if not dot_git.is_file():
        return True
    try:
        line = dot_git.read_text(encoding="utf-8", errors="surrogateescape").strip()
    except OSError:
        return False  # unreadable is not proof: let git decide (→ drop-failed if nothing reads)
    if not line.startswith("gitdir:"):
        return True
    admin = Path(line[len("gitdir:") :].strip())
    if not admin.is_absolute():  # `worktree.useRelativePaths`: relative to the worktree
        admin = ctx.scratch / admin
    if not admin.is_dir() or not _same_file(admin.parent, common_dir(ctx.main) / "worktrees"):
        return True
    try:
        back = Path((admin / "gitdir").read_text(encoding="utf-8", errors="surrogateescape").strip())
    except OSError:
        return False
    if not back.is_absolute():  # relative to the admin dir
        back = admin / back
    return not _same_file(back, dot_git)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def cmd_drop(ctx: Ctx) -> dict[str, Any]:
    if not ctx.scratch.is_dir():
        if is_registered(ctx.main, ctx.scratch):
            ctx.remove_scratch()  # prune, unlock fallback, drop-failed if it survives
            return {"status": "dropped", "pruned": True, "dropped_head": None, "integrated": None}
        return {"status": "nothing-to-drop"}
    if not is_registered(ctx.main, ctx.scratch):
        if os.path.lexists(ctx.scratch / ".git"):
            # Looks like a worktree moved here by hand (its registration names
            # another path): it may hold work — never remove it blind.
            raise Refusal(
                "drop-failed",
                f"{ctx.scratch} holds a .git but is not registered at this path; "
                f"run `git worktree repair {ctx.scratch}` (or move it away), then drop again",
            )
        # A leftover directory at this item's scratch path that git does not
        # know (e.g. a half-removed scratch): nothing to snapshot, remove it
        # so `worktree-setup.sh --integration` stops answering exit 3.
        shutil.rmtree(ctx.scratch)
        return {"status": "dropped", "not_a_worktree": True, "dropped_head": None, "integrated": None}
    if _scratch_git_broken(ctx):
        # Registered, but its `.git` is gone or foreign: nothing readable to
        # snapshot (and never the enclosing repo's) — remove it, so `prepare`
        # is not blocked forever.
        ctx.remove_scratch()
        return {"status": "dropped", "git_unreadable": True, "dropped_head": None, "integrated": None}
    state = ctx.read_state()
    head = rev(ctx.scratch, "HEAD")
    tip = ctx.deliverable_tip()
    integrated: bool | None = None
    if state is not None and tip is not None:
        integrated = is_ancestor(ctx.main, state["item_tip"], tip)
    out: dict[str, Any] = {
        "status": "dropped",
        "dropped_head": head,
        "integrated": integrated,
        "deliverable_missing": tip is None,
        "discarded_files": [],
        "discarded_snapshot": None,
    }
    try:
        files: list[str] | None = dirty_files(ctx.scratch)
    except Refusal as exc:
        if head is None:  # nothing readable, yet `.git` looks valid: never remove blind
            raise Refusal("drop-failed", f"cannot read {ctx.scratch}: {exc.detail}") from None
        files = None  # e.g. a corrupt index: the snapshot uses its own index
        out["status_unreadable"] = True
    if files is None or files:  # unknown or dirty: always keep an undo point
        out["discarded_files"] = files
        out["discarded_snapshot"] = _snapshot(ctx, head)
    ctx.remove_scratch()
    return out


def _snapshot(ctx: Ctx, head: str | None) -> str:
    """A commit of the whole working tree (unmerged, untracked, marker files) on a temporary index."""
    index = git_dir(ctx.scratch) / DROP_INDEX_FILE
    env = {"GIT_INDEX_FILE": str(index)}
    try:
        base = ("read-tree", "HEAD") if head is not None else ("read-tree", "--empty")  # an orphan HEAD
        for args in (base, ("add", "-A")):
            res = git(ctx.scratch, *args, env=env)
            if res.returncode != 0:
                raise Refusal("drop-failed", f"git {args[0]} failed: {res.stderr.strip()}")
        tree = git(ctx.scratch, "write-tree", env=env)
        if tree.returncode != 0:
            raise Refusal("drop-failed", f"git write-tree failed: {tree.stderr.strip()}")
        parents = ["-p", head] if head is not None else []
        merge_head = rev(ctx.scratch, "MERGE_HEAD")
        if merge_head is not None:
            parents += ["-p", merge_head]
        commit = git(
            ctx.scratch,
            "commit-tree",
            tree.stdout.strip(),
            "-m",
            f"QS-{ctx.issue}: drop snapshot of item {ctx.item}",
            *parents,
        )
        if commit.returncode != 0:
            raise Refusal("drop-failed", f"git commit-tree failed: {commit.stderr.strip()}")
        return commit.stdout.strip()
    finally:
        index.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Integrate a work item into its deliverable (QS-400)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add(name: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        p.add_argument("--issue", type=positive_int, required=True)
        p.add_argument("--item", type=positive_int, required=True)
        return p

    add("prepare").add_argument("--item-tip", required=True)
    add("check")
    add("gate").add_argument("--expect-head", required=True)
    move = add("move")
    move.add_argument("--new", required=True)
    move.add_argument("--old", required=True)
    add("drop")

    run = sub.add_parser("_gate-run")  # internal: the gate watchdog
    run.add_argument("--lock-fd", type=int, required=True)
    run.add_argument("--scratch", type=Path, required=True)
    run.add_argument("--result", type=Path, required=True)
    run.add_argument("--log", type=Path, required=True)
    run.add_argument("--deadline", type=float, required=True)
    run.add_argument("gate_argv", nargs=argparse.REMAINDER)
    return parser


def _json_safe(value: Any) -> Any:
    """Lone surrogates (non-UTF-8 path bytes, decoded with surrogateescape) → ``\\xNN`` text.

    The Control Plane stores the outputs in sqlite, which refuses lone
    surrogates; internally the raw names still round-trip as pathspecs.
    """
    if isinstance(value, str):
        try:
            return value.encode("utf-8", "surrogateescape").decode("utf-8", "backslashreplace")
        except UnicodeError:  # a surrogate outside U+DC80..U+DCFF (e.g. a hand-edited state)
            return value.encode("utf-8", "backslashreplace").decode("utf-8")
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _emit(data: dict[str, Any], code: int = 0) -> NoReturn:
    print(json.dumps(_json_safe(data), sort_keys=True))
    sys.exit(code)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "_gate-run":
        gate_argv = args.gate_argv[1:] if args.gate_argv[:1] == ["--"] else args.gate_argv
        sys.exit(cmd_gate_run(args.lock_fd, args.scratch, args.result, args.log, args.deadline, gate_argv))
    try:
        ctx = Ctx(args.issue, args.item)
        ctx.take_lock()
        if args.cmd == "prepare":
            out = cmd_prepare(ctx, args.item_tip)
        elif args.cmd == "check":
            out = cmd_check(ctx)
        elif args.cmd == "gate":
            out = cmd_gate(ctx, args.expect_head)
        elif args.cmd == "move":
            out = cmd_move(ctx, args.new, args.old)
        else:
            out = cmd_drop(ctx)
    except Refusal as exc:
        _emit({"error": exc.code, "detail": exc.detail, **exc.extra}, 1)
    except OSError as exc:
        _emit({"error": "io-failed", "detail": str(exc)}, 1)
    _emit(out)


if __name__ == "__main__":
    main()
