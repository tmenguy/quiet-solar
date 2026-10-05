"""Real-git tests for ``scripts/qs/integrate_item.py`` (QS-400 §8, AC 10–11).

``integrate_item.py`` is driven as a subprocess, as the Control Plane tools
drive it, from the fixture clone (its main checkout). The integration scratch
reaches the fixture's fake ``venv/bin/python`` through its ``venv`` symlink;
``QS_FAKE_GATE`` picks the fake gate's behaviour.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tests.qs._gitrepo import REPO_ROOT, Repo, make_repo

SCRIPT = REPO_ROOT / "scripts" / "qs" / "integrate_item.py"


class Env:
    """The fixture repo plus deliverable ``QS_42`` and item ``QS_42_1`` with one commit each."""

    def __init__(self, repo: Repo) -> None:
        self.repo = repo
        self.scratch = repo.worktrees / "QS_42_1_integration"

    def commit(self, branch: str, files: dict[str, str | None], msg: str = "c") -> str:
        """Commit ``files`` (``None`` deletes) on ``branch`` through a throwaway worktree."""
        wt = self.repo.root / f"tmp-{branch}-{time.monotonic_ns()}"
        self.repo.git("worktree", "add", "-q", str(wt), branch)
        try:
            for name, content in files.items():
                if content is None:
                    self.repo.git("rm", "-q", name, cwd=wt)
                else:
                    (wt / name).write_text(content)
                    self.repo.git("add", name, cwd=wt)
            self.repo.git("commit", "-q", "-m", msg, cwd=wt)
            return self.repo.rev("HEAD", cwd=wt)
        finally:
            self.repo.git("worktree", "remove", "--force", str(wt))

    def tip(self, ref: str) -> str:
        return self.repo.rev(ref)

    def run(self, *args: str, env: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
        res = self.popen(*args, env=env)
        out, err = res.communicate(timeout=120)
        try:
            data = json.loads(out)
        except ValueError:
            raise AssertionError(f"not JSON: {out!r} {err!r}") from None
        return res.returncode, data

    def popen(self, *args: str, env: dict[str, str] | None = None) -> subprocess.Popen[str]:
        sub, rest = args[0], args[1:]
        return subprocess.Popen(
            [sys.executable, str(SCRIPT), sub, "--issue", "42", "--item", "1", *rest],
            cwd=str(self.repo.clone),
            env={**self.repo.env, **(env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def prepare(self, tip: str | None = None) -> tuple[int, dict[str, Any]]:
        return self.run("prepare", "--item-tip", tip or self.tip("refs/heads/QS_42_1"))

    def scratch_git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self.repo.git(*args, cwd=self.scratch, check=check)

    def state_path(self) -> Path:
        return Path(self.scratch_git("rev-parse", "--absolute-git-dir").stdout.strip()) / "qs-integration.json"

    def state(self) -> dict[str, Any]:
        return json.loads(self.state_path().read_text())

    def set_state(self, **changes: Any) -> None:
        state = self.state()
        state.update(changes)
        self.state_path().write_text(json.dumps(state))

    def lock_path(self) -> Path:
        common = self.repo.git("rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip()
        return Path(common) / "qs-integration-QS_42_1.lock"


@pytest.fixture
def env(tmp_path: Path) -> Env:
    repo = make_repo(tmp_path)
    repo.git("branch", "QS_42", "main")
    e = Env(repo)
    e.commit("QS_42", {"deliv.txt": "deliverable\n", "shared.txt": "base\n"}, "deliverable work")
    repo.git("branch", "QS_42_1", "QS_42")
    e.commit("QS_42_1", {"item.txt": "item\n"}, "item work")
    return e


def conflicting(env: Env) -> None:
    """Make the item and the deliverable edit ``shared.txt`` differently."""
    env.commit("QS_42_1", {"shared.txt": "item side\n"})
    env.commit("QS_42", {"shared.txt": "deliverable side\n"})


def resolve(env: Env, content: str = "resolved\n") -> None:
    (env.scratch / "shared.txt").write_text(content)
    env.scratch_git("add", "shared.txt")
    env.scratch_git("commit", "-q", "--no-edit")


# ---------------------------------------------------------------------------
# AC 10 — the clean path
# ---------------------------------------------------------------------------


def test_clean_path(env: Env) -> None:
    old_tip = env.tip("refs/heads/QS_42")
    item_tip = env.tip("refs/heads/QS_42_1")
    rc, out = env.prepare()
    assert (rc, out["status"]) == (0, "merged"), out
    assert out["item_tip"] == item_tip and out["base"] == old_tip
    assert env.state()["phase"] == "merged"

    rc, check = env.run("check")
    assert (rc, check["status"], check["moved"]) == (0, "ready", False), check
    head = check["head"]

    rc, gate = env.run("gate", "--expect-head", head)
    assert (rc, gate["status"]) == (0, "green"), gate

    rc, move = env.run("move", "--new", head, "--old", old_tip)
    assert (rc, move["status"]) == (0, "moved"), move
    assert env.tip("refs/heads/QS_42") == head
    parents = env.repo.git("rev-list", "--parents", "-n", "1", head).stdout.split()[1:]
    assert parents == [old_tip, item_tip]

    rc, again = env.run("move", "--new", head, "--old", old_tip)
    assert (rc, again["status"]) == (0, "already-moved")

    rc, check = env.run("check")
    assert (rc, check["moved"]) == (0, True)
    env.commit("QS_42", {"later.txt": "later\n"})
    rc, check = env.run("check")
    assert (rc, check["moved"]) == (0, True), check

    # prepare while the integrated scratch still exists
    rc, out = env.prepare()
    assert (rc, out["status"], out["item_tip"]) == (0, "already-integrated", item_tip)
    assert "base" in out

    rc, drop = env.run("drop")
    assert (rc, drop["status"], drop["integrated"]) == (0, "dropped", True), drop
    assert not env.scratch.exists()
    assert (env.repo.clone / "venv" / "bin" / "python").exists()
    rc, drop = env.run("drop")
    assert (rc, drop["status"]) == (0, "nothing-to-drop")

    rc, out = env.prepare()
    assert (rc, out["status"], out["item_tip"]) == (0, "already-integrated", item_tip)
    assert "base" not in out
    count = env.repo.git("rev-list", "--count", "refs/heads/QS_42..refs/heads/QS_42_1").stdout.strip()
    assert count == "0"


# ---------------------------------------------------------------------------
# AC 11 — concurrency
# ---------------------------------------------------------------------------


def test_scratch_busy_while_the_lock_is_held(env: Env) -> None:
    assert env.prepare()[0] == 0
    fd = os.open(env.lock_path(), os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rc, out = env.run("check")
        assert (rc, out["error"]) == (1, "scratch-busy")
        assert out["max_wait_s"] == 3300
    finally:
        os.close(fd)
    assert env.run("check")[0] == 0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_killed_caller_leaves_the_watchdog_holding_the_lock_until_the_deadline(env: Env, tmp_path: Path) -> None:
    assert env.prepare()[0] == 0
    head = env.run("check")[1]["head"]
    fake = {"QS_FAKE_GATE": "sleep", "QS_FAKE_DIR": str(tmp_path), "QS_INTEGRATE_GATE_TIMEOUT_S": "20"}
    proc = env.popen("gate", "--expect-head", head, env=fake)
    log = Path(env.scratch_git("rev-parse", "--absolute-git-dir").stdout.strip()) / "qs-integration-gate.log"
    pids_file = tmp_path / "gate.pids"
    deadline = time.time() + 15
    while not (log.exists() and pids_file.exists() and len(pids_file.read_text().split()) == 2):
        assert time.time() < deadline, "the gate never started"
        time.sleep(0.1)
    proc.kill()
    proc.wait()

    rc, out = env.run("check", env={"QS_INTEGRATE_GATE_TIMEOUT_S": "20"})
    assert (rc, out["error"], out["max_wait_s"]) == (1, "scratch-busy", 20)

    pids = [int(p) for p in pids_file.read_text().split()]
    deadline = time.time() + 40
    while True:
        rc, out = env.run("check")
        if rc == 0:
            break
        assert out["error"] == "scratch-busy"
        assert time.time() < deadline, "the watchdog never released the lock"
        time.sleep(0.5)
    assert not any(_alive(pid) for pid in pids)


def test_watchdog_without_result_is_gate_run_failed(env: Env) -> None:
    assert env.prepare()[0] == 0
    head = env.run("check")[1]["head"]
    rc, out = env.run("gate", "--expect-head", head, env={"QS_FAKE_GATE": "killparent"})
    assert (rc, out["error"]) == (1, "gate-run-failed"), out
    assert isinstance(out["tail"], list)


# ---------------------------------------------------------------------------
# AC 11 — prepare
# ---------------------------------------------------------------------------


def test_prepare_missing_refs(env: Env) -> None:
    item_tip = env.tip("refs/heads/QS_42_1")
    env.repo.git("branch", "-D", "QS_42_1")
    rc, out = env.prepare(item_tip)
    assert (rc, out["error"]) == (1, "missing-ref")


def test_prepare_missing_deliverable(env: Env) -> None:
    env.repo.git("branch", "-D", "QS_42")
    rc, out = env.prepare()
    assert (rc, out["error"]) == (1, "missing-ref")


def test_prepare_item_moved(env: Env) -> None:
    rc, out = env.prepare(env.tip("refs/heads/QS_42"))
    assert (rc, out["error"]) == (1, "item-moved")


def test_prepare_conflicts(env: Env) -> None:
    conflicting(env)
    rc, out = env.prepare()
    assert (rc, out["status"], out["files"]) == (0, "conflicts", ["shared.txt"]), out
    assert out["scratch"] == str(env.scratch)
    assert env.state()["phase"] == "conflicts"
    assert env.state()["conflicted_files"] == ["shared.txt"]
    # resumed, mid-resolution and after the resolution commit
    rc, out = env.prepare()
    assert (rc, out["status"]) == (0, "conflicts")
    resolve(env)
    rc, out = env.prepare()
    assert (rc, out["status"]) == (0, "conflicts")


def test_prepare_resumes_a_created_scratch(env: Env) -> None:
    assert env.prepare()[0] == 0
    env.scratch_git("reset", "-q", "--hard", env.state()["base"])
    env.set_state(phase="created")
    rc, out = env.prepare()
    assert (rc, out["status"]) == (0, "merged"), out
    assert env.state()["phase"] == "merged"


def test_prepare_stale_for_another_tip(env: Env) -> None:
    assert env.prepare()[0] == 0
    env.commit("QS_42_1", {"more.txt": "more\n"})
    rc, out = env.prepare()
    assert (rc, out["error"]) == (1, "stale-scratch")


def test_prepare_stale_without_state(env: Env) -> None:
    assert env.prepare()[0] == 0
    env.state_path().unlink()
    rc, out = env.prepare()
    assert (rc, out["error"]) == (1, "stale-scratch")


def test_prepare_stale_after_merge_abort(env: Env) -> None:
    conflicting(env)
    assert env.prepare()[1]["status"] == "conflicts"
    env.scratch_git("merge", "--abort")
    rc, out = env.prepare()
    assert (rc, out["error"]) == (1, "stale-scratch")


def test_prepare_stale_created_after_deliverable_moved(env: Env) -> None:
    assert env.prepare()[0] == 0
    env.scratch_git("reset", "-q", "--hard", env.state()["base"])
    env.set_state(phase="created")
    env.commit("QS_42", {"moved.txt": "x\n"})
    rc, out = env.prepare()
    assert (rc, out["error"]) == (1, "stale-scratch")


def test_prepare_stale_merged_after_deliverable_moved(env: Env) -> None:
    assert env.prepare()[0] == 0
    env.commit("QS_42", {"moved.txt": "x\n"})
    rc, out = env.prepare()
    assert (rc, out["error"]) == (1, "stale-scratch")


def test_prepare_stale_for_another_item(env: Env) -> None:
    assert env.prepare()[0] == 0
    env.set_state(item=2)
    rc, out = env.prepare()
    assert (rc, out["error"]) == (1, "stale-scratch")


def test_prepare_crash_window_a(env: Env) -> None:
    assert env.prepare()[0] == 0
    env.set_state(phase="created")
    rc, out = env.prepare()
    assert (rc, out["status"]) == (0, "merged")
    assert env.state()["phase"] == "merged"


def test_prepare_crash_window_b(env: Env) -> None:
    conflicting(env)
    assert env.prepare()[1]["status"] == "conflicts"
    env.set_state(phase="created", conflicted_files=[])
    rc, out = env.prepare()
    assert (rc, out["status"], out["files"]) == (0, "conflicts", ["shared.txt"])
    assert env.state()["phase"] == "conflicts"


def test_prepare_merge_failed_removes_the_scratch(env: Env) -> None:
    # An untracked file in the fresh scratch would be overwritten by the merge.
    env.commit("QS_42_1", {"seeded.txt": "tracked by the item\n"})
    real = env.repo.clone / "scripts" / "worktree-setup.sh"
    original = real.read_text()
    real.write_text(
        original.replace(
            'echo ""\nif [ "$MODE" = integration ]; then',
            'echo ""\nif [ "$MODE" = integration ]; then echo untracked > "$WORKTREE_DIR/seeded.txt"; fi\n'
            'if [ "$MODE" = integration ]; then',
        )
    )
    rc, out = env.prepare()
    assert (rc, out["error"]) == (1, "merge-failed"), out
    assert not env.scratch.exists()
    assert "QS_42_1_integration" not in env.repo.git("worktree", "list").stdout


def test_prepare_setup_failed(env: Env) -> None:
    (env.repo.clone / "scripts" / "worktree-setup.sh").write_text("#!/bin/bash\necho broken\nexit 1\n")
    rc, out = env.prepare()
    assert (rc, out["error"]) == (1, "setup-failed")
    assert "broken" in out["detail"]


def test_prepare_registered_scratch_without_directory_is_stale(env: Env) -> None:
    assert env.prepare()[0] == 0
    shutil.rmtree(env.scratch)
    env.commit("QS_42_1", {"again.txt": "x\n"})
    rc, out = env.prepare()
    assert (rc, out["error"]) == (1, "stale-scratch")


def test_item_two_is_not_blocked_by_item_one(env: Env) -> None:
    assert env.prepare()[0] == 0
    env.set_state(item=9)  # item 1's scratch is now stale
    env.repo.git("branch", "QS_42_2", "QS_42")
    env.commit("QS_42_2", {"two.txt": "2\n"})
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "prepare", "--issue", "42", "--item", "2", "--item-tip", env.tip("QS_42_2")],
        cwd=str(env.repo.clone),
        env=env.repo.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stdout
    assert json.loads(res.stdout)["status"] == "merged"


# ---------------------------------------------------------------------------
# AC 11 — check, gate, move: no scratch / no state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args", [("check",), ("gate", "--expect-head", "HEAD"), ("move", "--new", "HEAD", "--old", "HEAD")]
)
def test_no_scratch_and_missing_state(env: Env, args: tuple[str, ...]) -> None:
    rc, out = env.run(*args)
    assert (rc, out["error"]) == (1, "no-scratch")
    assert env.prepare()[0] == 0
    env.state_path().unlink()
    rc, out = env.run(*args)
    assert (rc, out["error"]) == (1, "stale-scratch")


# ---------------------------------------------------------------------------
# AC 11 — check
# ---------------------------------------------------------------------------


def test_check_merge_in_progress_and_unmerged(env: Env) -> None:
    conflicting(env)
    assert env.prepare()[1]["status"] == "conflicts"
    rc, out = env.run("check")
    assert (rc, out["error"]) == (1, "merge-in-progress")
    gdir = Path(env.scratch_git("rev-parse", "--absolute-git-dir").stdout.strip())
    (gdir / "MERGE_HEAD").unlink()  # unmerged index entries remain
    rc, out = env.run("check")
    assert (rc, out["error"], out["files"]) == (1, "unmerged-paths", ["shared.txt"])


@pytest.mark.parametrize("kind", ["tracked", "untracked"])
def test_check_dirty_scratch(env: Env, kind: str) -> None:
    assert env.prepare()[0] == 0
    target = env.scratch / ("item.txt" if kind == "tracked" else "new.txt")
    target.write_text("dirty\n")
    rc, out = env.run("check")
    assert (rc, out["error"]) == (1, "dirty-scratch")
    assert target.name in out["files"]


def test_check_not_a_merge(env: Env) -> None:
    assert env.prepare()[0] == 0
    env.scratch_git("reset", "-q", "--hard", env.state()["base"])
    rc, out = env.run("check")
    assert (rc, out["error"]) == (1, "not-a-merge")


def test_check_conflict_markers(env: Env) -> None:
    conflicting(env)
    assert env.prepare()[1]["status"] == "conflicts"
    (env.scratch / "item.txt").write_text("<<<<<<< not a conflicted file\n")
    env.scratch_git("add", "item.txt")
    resolve(env, "<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> item\n")
    rc, out = env.run("check")
    assert (rc, out["error"], out["files"]) == (1, "conflict-markers", ["shared.txt"])


def test_check_marker_in_another_file_passes(env: Env) -> None:
    conflicting(env)
    assert env.prepare()[1]["status"] == "conflicts"
    (env.scratch / "item.txt").write_text("<<<<<<< not a conflicted file\n")
    env.scratch_git("add", "item.txt")
    resolve(env)
    rc, out = env.run("check")
    assert (rc, out["status"]) == (0, "ready"), out


def test_check_deleted_conflicted_file_passes(env: Env) -> None:
    conflicting(env)
    assert env.prepare()[1]["status"] == "conflicts"
    env.scratch_git("rm", "-q", "shared.txt")
    env.scratch_git("commit", "-q", "--no-edit")
    rc, out = env.run("check")
    assert (rc, out["status"]) == (0, "ready"), out


def test_check_deliverable_moved(env: Env) -> None:
    assert env.prepare()[0] == 0
    env.commit("QS_42", {"moved.txt": "x\n"})
    rc, out = env.run("check")
    assert (rc, out["error"]) == (1, "deliverable-moved")


def _deliverable_worktree(env: Env) -> Path:
    wt = env.repo.root / "deliverable-wt"
    env.repo.git("worktree", "add", "-q", str(wt), "QS_42")
    return wt


def test_check_deliverable_worktree_dirty(env: Env) -> None:
    wt = _deliverable_worktree(env)
    assert env.prepare()[0] == 0
    (wt / "deliv.txt").write_text("edited\n")
    rc, out = env.run("check")
    assert (rc, out["error"]) == (1, "deliverable-worktree-busy")
    (wt / "deliv.txt").write_text("deliverable\n")
    (wt / "untracked.txt").write_text("fine\n")
    assert env.run("check")[0] == 0


def test_check_deliverable_worktree_cherry_pick_in_progress(env: Env) -> None:
    wt = _deliverable_worktree(env)
    assert env.prepare()[0] == 0
    gdir = Path(env.repo.git("rev-parse", "--absolute-git-dir", cwd=wt).stdout.strip())
    (gdir / "CHERRY_PICK_HEAD").write_text(env.tip("QS_42_1") + "\n")
    rc, out = env.run("check")
    assert (rc, out["error"]) == (1, "deliverable-worktree-busy")


def test_check_never_moved_after_merge_abort(env: Env) -> None:
    conflicting(env)
    assert env.prepare()[1]["status"] == "conflicts"
    env.scratch_git("merge", "--abort")
    rc, out = env.run("check")
    assert rc == 1 and out["error"] == "not-a-merge"


def test_check_never_moved_at_phase_created(env: Env) -> None:
    assert env.prepare()[0] == 0
    env.scratch_git("reset", "-q", "--hard", env.state()["base"])
    env.set_state(phase="created")
    rc, out = env.run("check")
    assert rc == 1 and out["error"] == "stale-scratch"


def test_check_never_moved_when_head_is_base(env: Env) -> None:
    assert env.prepare()[0] == 0
    base = env.state()["base"]
    env.scratch_git("reset", "-q", "--hard", base)
    # HEAD == base is an ancestor of QS_42, but nothing was merged.
    rc, out = env.run("check")
    assert rc == 1 and out["error"] == "not-a-merge"


# ---------------------------------------------------------------------------
# AC 11 — gate
# ---------------------------------------------------------------------------


def _ready(env: Env) -> str:
    assert env.prepare()[0] == 0
    rc, out = env.run("check")
    assert rc == 0, out
    return str(out["head"])


def test_gate_red_with_tail(env: Env) -> None:
    head = _ready(env)
    rc, out = env.run("gate", "--expect-head", head, env={"QS_FAKE_GATE": "red"})
    assert (rc, out["status"]) == (0, "red")
    assert len(out["tail"]) == 40
    assert out["tail"][-1] == "red line 49"


def test_gate_no_venv(env: Env) -> None:
    head = _ready(env)
    (env.scratch / "venv").unlink()
    rc, out = env.run("gate", "--expect-head", head)
    assert (rc, out["error"]) == (1, "no-venv")


def test_gate_refuses_a_moved_head_or_dirty_tree_before_running(env: Env) -> None:
    head = _ready(env)
    rc, out = env.run("gate", "--expect-head", env.state()["base"])
    assert (rc, out["error"]) == (1, "head-moved")
    (env.scratch / "new.txt").write_text("x\n")
    rc, out = env.run("gate", "--expect-head", head)
    assert (rc, out["error"]) == (1, "dirty-scratch")


def test_gate_commit_during_gate_is_head_moved(env: Env) -> None:
    head = _ready(env)
    old = env.state()["base"]
    rc, out = env.run("gate", "--expect-head", head, env={"QS_FAKE_GATE": "commit"})
    assert (rc, out["error"]) == (1, "head-moved"), out
    rc, out = env.run("move", "--new", head, "--old", old)
    assert (rc, out["error"]) == (1, "head-moved")
    assert env.tip("QS_42") == old


def test_gate_modified_tree(env: Env) -> None:
    head = _ready(env)
    rc, out = env.run("gate", "--expect-head", head, env={"QS_FAKE_GATE": "touch"})
    assert (rc, out["error"], out["files"]) == (1, "gate-modified-tree", ["README.md"])


def test_gate_timeout_kills_the_group(env: Env, tmp_path: Path) -> None:
    head = _ready(env)
    fake = {"QS_FAKE_GATE": "sleep", "QS_FAKE_DIR": str(tmp_path), "QS_INTEGRATE_GATE_TIMEOUT_S": "2"}
    rc, out = env.run("gate", "--expect-head", head, env=fake)
    assert (rc, out["error"]) == (1, "gate-timeout"), out
    pids = [int(p) for p in (tmp_path / "gate.pids").read_text().split()]
    time.sleep(0.2)
    assert not any(_alive(pid) for pid in pids)
    assert env.run("check")[0] == 0  # the lock is free


# ---------------------------------------------------------------------------
# AC 11 — move
# ---------------------------------------------------------------------------


def test_move_head_moved_and_deliverable_moved(env: Env) -> None:
    head = _ready(env)
    old = env.state()["base"]
    rc, out = env.run("move", "--new", old, "--old", old)
    assert (rc, out["error"]) == (1, "head-moved")
    env.commit("QS_42", {"moved.txt": "x\n"})
    rc, out = env.run("move", "--new", head, "--old", old)
    assert (rc, out["error"]) == (1, "deliverable-moved")


def test_move_fast_forwards_a_clean_deliverable_worktree(env: Env) -> None:
    wt = _deliverable_worktree(env)
    head = _ready(env)
    rc, out = env.run("move", "--new", head, "--old", env.state()["base"])
    assert (rc, out["status"]) == (0, "moved"), out
    assert env.tip("QS_42") == head
    assert env.repo.rev("HEAD", cwd=wt) == head
    assert env.repo.git("status", "--porcelain", "--untracked-files=no", cwd=wt).stdout == ""
    assert (wt / "item.txt").exists()


def test_move_refuses_a_dirty_deliverable_worktree(env: Env) -> None:
    wt = _deliverable_worktree(env)
    head = _ready(env)
    old = env.state()["base"]
    (wt / "deliv.txt").write_text("edited\n")
    rc, out = env.run("move", "--new", head, "--old", old)
    assert (rc, out["error"]) == (1, "deliverable-worktree-busy")
    assert env.tip("QS_42") == old


# ---------------------------------------------------------------------------
# AC 11 — drop
# ---------------------------------------------------------------------------


def test_drop_at_base_is_not_integrated(env: Env) -> None:
    assert env.prepare()[0] == 0
    base = env.state()["base"]
    env.scratch_git("reset", "-q", "--hard", base)
    rc, out = env.run("drop")
    assert (rc, out["status"], out["integrated"], out["dropped_head"]) == (0, "dropped", False, base)
    assert out["discarded_snapshot"] is None and out["discarded_files"] == []


def test_drop_mid_resolution_snapshots_everything(env: Env) -> None:
    conflicting(env)
    env.commit("QS_42_1", {"other.txt": "item other\n"})
    env.commit("QS_42", {"other.txt": "deliv other\n"})
    assert env.prepare()[1]["status"] == "conflicts"
    # shared.txt resolved (staged), other.txt still unmerged with markers, plus an untracked file.
    (env.scratch / "shared.txt").write_text("resolved\n")
    env.scratch_git("add", "shared.txt")
    (env.scratch / "untracked.txt").write_text("new\n")
    head = env.repo.rev("HEAD", cwd=env.scratch)
    merge_head = env.repo.rev("MERGE_HEAD", cwd=env.scratch)

    rc, out = env.run("drop")
    assert (rc, out["status"]) == (0, "dropped"), out
    assert not env.scratch.exists()
    snap = out["discarded_snapshot"]
    assert set(out["discarded_files"]) >= {"shared.txt", "other.txt", "untracked.txt"}
    assert env.repo.git("show", f"{snap}:shared.txt").stdout == "resolved\n"
    assert env.repo.git("show", f"{snap}:untracked.txt").stdout == "new\n"
    assert "<<<<<<<" in env.repo.git("show", f"{snap}:other.txt").stdout
    parents = env.repo.git("rev-list", "--parents", "-n", "1", snap).stdout.split()[1:]
    assert parents == [head, merge_head]


def test_drop_without_state(env: Env) -> None:
    assert env.prepare()[0] == 0
    env.state_path().unlink()
    rc, out = env.run("drop")
    assert (rc, out["status"], out["integrated"]) == (0, "dropped", None)


def test_drop_with_deliverable_deleted(env: Env) -> None:
    assert env.prepare()[0] == 0
    env.repo.git("branch", "-D", "QS_42")
    rc, out = env.run("drop")
    assert (rc, out["status"], out["integrated"], out["deliverable_missing"]) == (0, "dropped", None, True)
    assert not env.scratch.exists()


def test_drop_registered_scratch_whose_directory_is_gone(env: Env) -> None:
    assert env.prepare()[0] == 0
    shutil.rmtree(env.scratch)
    rc, out = env.run("drop")
    assert (rc, out["status"], out["pruned"]) == (0, "dropped", True)
    assert "QS_42_1_integration" not in env.repo.git("worktree", "list").stdout


def test_drop_busy(env: Env) -> None:
    assert env.prepare()[0] == 0
    fd = os.open(env.lock_path(), os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rc, out = env.run("drop")
        assert (rc, out["error"]) == (1, "scratch-busy")
    finally:
        os.close(fd)
    assert env.scratch.exists()


def test_start_drop_start(env: Env) -> None:
    assert env.prepare()[1]["status"] == "merged"
    assert env.run("drop")[1]["status"] == "dropped"
    rc, out = env.prepare()
    assert (rc, out["status"]) == (0, "merged")


def test_bad_arguments_exit_2(env: Env) -> None:
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "check", "--issue", "42", "--item", "01"],
        cwd=str(env.repo.clone),
        env=env.repo.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 2


def test_drop_removes_a_leftover_that_is_not_a_worktree(env: Env) -> None:
    env.scratch.mkdir(parents=True)
    (env.scratch / "junk.txt").write_text("x\n")
    (env.scratch / "venv").symlink_to(env.repo.clone / "venv")
    rc, out = env.run("drop")
    assert (rc, out["status"], out["dropped_head"]) == (0, "dropped", None), out
    assert out["not_a_worktree"] is True
    assert not env.scratch.exists()
    assert (env.repo.clone / "venv" / "bin" / "python").exists()


# ---------------------------------------------------------------------------
# Review fix #01
# ---------------------------------------------------------------------------


def test_drop_a_scratch_whose_git_file_is_gone(env: Env) -> None:
    """Fix 1: a registered scratch with no ``.git`` is still dropped (no ``stale-scratch`` loop)."""
    assert env.prepare()[0] == 0
    (env.scratch / ".git").unlink()
    rc, out = env.run("drop")
    assert (rc, out["status"], out["git_unreadable"]) == (0, "dropped", True), out
    assert not env.scratch.exists()
    assert "QS_42_1_integration" not in env.repo.git("worktree", "list").stdout
    assert env.prepare()[1]["status"] == "merged"


def test_stale_deliverable_registration_is_ignored(env: Env) -> None:
    """Fix 2: a registration of QS_42 whose directory is gone neither blocks check nor move."""
    wt = _deliverable_worktree(env)
    shutil.rmtree(wt)
    head = _ready(env)
    rc, out = env.run("move", "--new", head, "--old", env.state()["base"])
    assert (rc, out["status"]) == (0, "moved"), out
    assert env.tip("QS_42") == head


def test_non_ascii_conflicted_file_is_scanned_for_markers(env: Env) -> None:
    """Fix 3: paths are read unquoted; the marker grep sees ``café.txt``."""
    env.commit("QS_42_1", {"café.txt": "item side\n"})
    env.commit("QS_42", {"café.txt": "deliverable side\n"})
    rc, out = env.prepare()
    assert (rc, out["status"], out["files"]) == (0, "conflicts", ["café.txt"]), out
    (env.scratch / "café.txt").write_text("<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> item\n")
    env.scratch_git("add", "café.txt")
    env.scratch_git("commit", "-q", "--no-edit")
    rc, out = env.run("check")
    assert (rc, out["error"], out["files"]) == (1, "conflict-markers", ["café.txt"])


def test_dirty_files_are_reported_unquoted(env: Env) -> None:
    assert env.prepare()[0] == 0
    (env.scratch / "naïve.txt").write_text("x\n")
    rc, out = env.run("check")
    assert (rc, out["error"], out["files"]) == (1, "dirty-scratch", ["naïve.txt"])


def test_drop_a_locked_scratch(env: Env) -> None:
    """Fix 7: a locked scratch is removed and unregistered too."""
    assert env.prepare()[0] == 0
    env.repo.git("worktree", "lock", str(env.scratch))
    rc, out = env.run("drop")
    assert (rc, out["status"]) == (0, "dropped"), out
    assert "QS_42_1_integration" not in env.repo.git("worktree", "list").stdout


def test_io_error_is_json(env: Env) -> None:
    """Fix 8: an OSError (here: the lock file cannot be created) is a JSON ``io-failed``."""
    lock = env.lock_path()
    lock.mkdir()  # os.open(O_RDWR) on a directory → IsADirectoryError
    rc, out = env.run("check")
    assert (rc, out["error"]) == (1, "io-failed")


def test_non_ascii_digit_timeout_falls_back_to_the_default(env: Env) -> None:
    """Fix 10."""
    fd = os.open(env.lock_path(), os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rc, out = env.run("check", env={"QS_INTEGRATE_GATE_TIMEOUT_S": "²"})
        assert (rc, out["error"], out["max_wait_s"]) == (1, "scratch-busy", 3300)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Review fix #02
# ---------------------------------------------------------------------------


def test_drop_a_locked_registration_whose_directory_is_gone(env: Env) -> None:
    """A: no prepare/drop loop on a locked registration with its directory removed by hand."""
    assert env.prepare()[0] == 0
    env.repo.git("worktree", "lock", str(env.scratch))
    shutil.rmtree(env.scratch)
    rc, out = env.run("drop")
    assert (rc, out["status"], out["pruned"]) == (0, "dropped", True), out
    assert "QS_42_1_integration" not in env.repo.git("worktree", "list").stdout
    assert env.prepare()[1]["status"] == "merged"


def test_drop_with_a_git_file_pointing_nowhere(env: Env) -> None:
    """B: a ``.git`` file whose target is gone is provably unreadable."""
    assert env.prepare()[0] == 0
    (env.scratch / ".git").write_text("gitdir: /nonexistent/qs-400\n")
    rc, out = env.run("drop")
    assert (rc, out["status"], out["git_unreadable"]) == (0, "dropped", True), out
    assert not env.scratch.exists()


def test_drop_inside_an_enclosing_repository_never_touches_it(env: Env) -> None:
    """B: with the scratch's ``.git`` gone, git would find the enclosing repo — drop must not use it."""
    outer = env.repo.root
    env.repo.git("init", "-q", str(outer))
    (outer / "outer.txt").write_text("outer\n")
    env.repo.git("add", "outer.txt", cwd=outer)
    env.repo.git("commit", "-q", "-m", "outer", cwd=outer)
    outer_head = env.repo.rev("HEAD", cwd=outer)
    objects_before = env.repo.git("count-objects", "-v", cwd=outer).stdout
    assert env.prepare()[0] == 0
    (env.scratch / ".git").unlink()
    rc, out = env.run("drop")
    assert (rc, out["status"], out["git_unreadable"]) == (0, "dropped", True), out
    assert out["dropped_head"] is None
    assert env.repo.rev("HEAD", cwd=outer) == outer_head
    assert env.repo.git("count-objects", "-v", cwd=outer).stdout == objects_before


def test_drop_with_a_corrupt_index_still_snapshots(env: Env) -> None:
    """C: ``git status`` fails on a corrupt index; the snapshot uses its own index."""
    assert env.prepare()[0] == 0
    (env.scratch / "work.txt").write_text("unsaved work\n")
    gdir = Path(env.scratch_git("rev-parse", "--absolute-git-dir").stdout.strip())
    (gdir / "index").write_bytes(b"junk")
    rc, out = env.run("drop")
    assert (rc, out["status"], out["status_unreadable"]) == (0, "dropped", True), out
    assert out["discarded_files"] is None
    snap = out["discarded_snapshot"]
    assert env.repo.git("show", f"{snap}:work.txt").stdout == "unsaved work\n"
    assert not env.scratch.exists()


def test_non_utf8_paths_do_not_raise(env: Env) -> None:
    """D: raw ``-z`` bytes are decoded with surrogateescape."""
    import integrate_item

    res = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        input=b"x\n",
        cwd=str(env.repo.clone),
        env=env.repo.env,
        capture_output=True,
        check=True,
    )
    sha = res.stdout.decode().strip()
    subprocess.run(
        [b"git", b"update-index", b"--add", b"--cacheinfo", b"100644," + sha.encode() + b",caf\xe9.txt"],
        cwd=str(env.repo.clone),
        env=env.repo.env,
        check=True,
    )
    old = os.environ.copy()
    os.environ.update(env.repo.env)
    try:
        files = integrate_item.dirty_files(env.repo.clone, untracked=False)
    finally:
        os.environ.clear()
        os.environ.update(old)
    assert any(name.startswith("caf") and name.endswith(".txt") for name in files)
    json.dumps(files)  # stays JSON-serialisable


def _hook(env: Env, body: str) -> None:
    hooks = Path(env.repo.git("rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip()) / "hooks"
    hooks.mkdir(exist_ok=True)
    hook = hooks / "post-merge"
    hook.write_text("#!/bin/sh\n" + body + "\n")
    hook.chmod(0o755)


def test_move_ff_race_moved_elsewhere_is_deliverable_moved(env: Env) -> None:
    """G: QS_42 jumps elsewhere right after the fast-forward."""
    _deliverable_worktree(env)
    head = _ready(env)
    _hook(env, f"git update-ref refs/heads/QS_42 {env.tip('main')}")
    rc, out = env.run("move", "--new", head, "--old", env.state()["base"])
    assert (rc, out["error"]) == (1, "deliverable-moved"), out


def test_move_ff_race_descendant_is_already_moved(env: Env) -> None:
    """G: QS_42 gains a commit on top of the new head right after the fast-forward."""
    _deliverable_worktree(env)
    head = _ready(env)
    _hook(env, 'c=$(git commit-tree "HEAD^{tree}" -p HEAD -m later) && git update-ref refs/heads/QS_42 "$c"')
    rc, out = env.run("move", "--new", head, "--old", env.state()["base"])
    assert (rc, out["status"]) == (0, "already-moved"), out


def test_move_ff_reset_back_is_move_failed(env: Env) -> None:
    """G: QS_42 is back at the old tip after the fast-forward → move-failed."""
    _deliverable_worktree(env)
    head = _ready(env)
    old = env.state()["base"]
    _hook(env, f"git update-ref refs/heads/QS_42 {old}")
    rc, out = env.run("move", "--new", head, "--old", old)
    assert (rc, out["error"]) == (1, "move-failed"), out


def test_dirty_scratch_lists_a_staged_rename_once(env: Env) -> None:
    """H: ``status -z`` prints a rename as ``new\\0old``; only the new name is reported."""
    assert env.prepare()[0] == 0
    env.scratch_git("mv", "item.txt", "renamed.txt")
    rc, out = env.run("check")
    assert (rc, out["error"], out["files"]) == (1, "dirty-scratch", ["renamed.txt"])


def test_drop_refuses_a_valid_looking_but_unreadable_scratch(env: Env) -> None:
    """B: ``.git`` points at this scratch's admin dir, yet git reads nothing → ``drop-failed``, nothing removed."""
    assert env.prepare()[0] == 0
    gdir = Path(env.scratch_git("rev-parse", "--absolute-git-dir").stdout.strip())
    (gdir / "HEAD").write_text("garbage\n")
    rc, out = env.run("drop")
    assert (rc, out["error"]) == (1, "drop-failed"), out
    assert env.scratch.exists()


# ---------------------------------------------------------------------------
# Review fix #03
# ---------------------------------------------------------------------------


def test_drop_with_relative_worktree_paths_still_snapshots(env: Env) -> None:
    """A healthy scratch made with ``worktree.useRelativePaths`` is not mistaken for a broken one."""
    env.repo.git("config", "worktree.useRelativePaths", "true")
    assert env.prepare()[0] == 0
    gitfile = (env.scratch / ".git").read_text()
    assert not gitfile.split(":", 1)[1].strip().startswith("/"), gitfile  # really relative
    (env.scratch / "work.txt").write_text("keep me\n")
    rc, out = env.run("drop")
    assert (rc, out["status"]) == (0, "dropped"), out
    assert "git_unreadable" not in out
    assert env.repo.git("show", f"{out['discarded_snapshot']}:work.txt").stdout == "keep me\n"


def test_already_moved_after_ff_reports_the_current_tip(env: Env) -> None:
    _deliverable_worktree(env)
    head = _ready(env)
    _hook(env, 'c=$(git commit-tree "HEAD^{tree}" -p HEAD -m later) && git update-ref refs/heads/QS_42 "$c"')
    rc, out = env.run("move", "--new", head, "--old", env.state()["base"])
    assert (rc, out["status"], out["head"]) == (0, "already-moved", head)
    assert out["now"] == env.tip("QS_42") != head


def test_unreadable_git_file_is_not_proof_of_breakage(env: Env) -> None:
    """SF-1: an unreadable ``.git`` lets git decide — never a snapshot-less removal."""
    assert env.prepare()[0] == 0
    dot_git = env.scratch / ".git"
    dot_git.chmod(0o000)
    try:
        rc, out = env.run("drop")
    finally:
        dot_git.chmod(0o644)
    assert (rc, out["error"]) == (1, "drop-failed"), out
    assert env.scratch.exists()


def test_emitted_json_carries_no_lone_surrogates(env: Env) -> None:
    """SF-2: a non-UTF-8 name is emitted as valid Unicode (the Control Plane stores it in sqlite)."""
    import integrate_item

    text = integrate_item._json_safe({"files": ["caf\udce9.txt", "ok"], "n": 1, "nested": [{"x": "\udcff"}]})
    dumped = json.dumps(text, ensure_ascii=False)
    dumped.encode("utf-8")  # no lone surrogate survives
    assert text["files"] == ["caf\\xe9.txt", "ok"]


def test_drop_an_orphan_head_snapshots_without_parent(env: Env) -> None:
    """SF-3: no HEAD commit, yet dirty files → a parentless snapshot, never a silent loss."""
    assert env.prepare()[0] == 0
    env.scratch_git("checkout", "-q", "--orphan", "x")
    (env.scratch / "p.txt").write_text("precious\n")
    rc, out = env.run("drop")
    assert (rc, out["status"], out["dropped_head"]) == (0, "dropped", None), out
    snap = out["discarded_snapshot"]
    assert env.repo.git("show", f"{snap}:p.txt").stdout == "precious\n"
    assert env.repo.git("rev-list", "--parents", "-n", "1", snap).stdout.split()[1:] == []


def test_move_ff_race_deliverable_deleted_is_move_failed(env: Env) -> None:
    """NF-3: QS_42 deleted right after the fast-forward → move-failed."""
    _deliverable_worktree(env)
    head = _ready(env)
    _hook(env, "git update-ref -d refs/heads/QS_42")
    rc, out = env.run("move", "--new", head, "--old", env.state()["base"])
    assert (rc, out["error"]) == (1, "move-failed"), out


def test_drop_with_a_git_file_pointing_at_another_worktree(env: Env) -> None:
    """B: a ``.git`` pointing at another worktree's admin dir is not this scratch — removed, the other intact."""
    other = _deliverable_worktree(env)
    assert env.prepare()[0] == 0
    other_admin = env.repo.git("rev-parse", "--absolute-git-dir", cwd=other).stdout.strip()
    (env.scratch / ".git").write_text(f"gitdir: {other_admin}\n")
    rc, out = env.run("drop")
    assert (rc, out["status"], out["git_unreadable"]) == (0, "dropped", True), out
    assert not env.scratch.exists()
    assert env.repo.git("status", "--porcelain", cwd=other).returncode == 0


def test_merge_failed_keeps_its_detail_when_cleanup_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """I: a failing scratch removal is attached as ``cleanup_error``; the code stays ``merge-failed``."""
    import integrate_item

    class FakeCtx:
        issue, item, scratch = 42, 1, tmp_path

        def remove_scratch(self) -> None:
            raise integrate_item.Refusal("drop-failed", "boom")

    monkeypatch.setattr(
        integrate_item, "git", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "merge boom")
    )
    monkeypatch.setattr(integrate_item, "rev", lambda *a, **k: None)
    state = {"item_tip": "abc"}
    with pytest.raises(integrate_item.Refusal) as exc:
        integrate_item._merge(FakeCtx(), state, {"item_tip": "abc"})  # type: ignore[arg-type]
    assert exc.value.code == "merge-failed"
    assert exc.value.detail == "merge boom"
    assert exc.value.extra["cleanup_error"] == "boom"
