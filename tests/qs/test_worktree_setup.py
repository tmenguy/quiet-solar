"""Real-git tests for ``scripts/worktree-setup.sh`` (QS-400).

AC 1 — the frozen ``QS_<N>`` task path is unchanged (regression);
AC 3 — item mode ``worktree-setup.sh <N> <k>``;
AC 9 — integration mode ``worktree-setup.sh <N> <k> --integration``.

The fixture is a bare ``origin`` plus a clone holding a copy of the script,
with an isolated git config. The script runs under ``/bin/bash`` when it
exists (macOS bash 3.2), else ``bash``.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.qs._gitrepo import Repo, make_repo


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[Repo]:
    yield make_repo(tmp_path)


def assert_seeded(wt: Path, *, full: bool = True) -> None:
    assert (wt / "venv").is_symlink()
    if full:
        assert (wt / "config" / "secrets.yaml").is_symlink()
        assert (wt / "custom_components" / "other").is_symlink()
    else:
        assert not (wt / "config").exists()
        assert not (wt / "custom_components" / "other").exists()
    for cache in (".mypy_cache", ".testmondata"):
        assert (wt / cache).exists()
        assert not (wt / cache).is_symlink()


def upstream(repo: Repo, cwd: Path) -> str:
    res = repo.git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", cwd=cwd, check=False)
    return res.stdout.strip() if res.returncode == 0 else ""


def origin_heads(repo: Repo) -> set[str]:
    out = repo.git("ls-remote", "--heads", str(repo.origin)).stdout
    return {line.split()[1].removeprefix("refs/heads/") for line in out.splitlines()}


# ---------------------------------------------------------------------------
# AC 1 — the task path is unchanged
# ---------------------------------------------------------------------------


def test_task_mode_is_unchanged(repo: Repo) -> None:
    res = repo.setup("42")
    assert res.returncode == 0, res.stdout + res.stderr
    wt = repo.worktrees / "QS_42"
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD", cwd=wt).stdout.strip() == "QS_42"
    assert upstream(repo, wt) == "origin/QS_42"
    assert "QS_42" in origin_heads(repo)
    assert_seeded(wt)

    again = repo.setup("42")
    assert again.returncode == 0
    assert "already set up" in again.stdout


@pytest.mark.parametrize(
    "args",
    [[], ["x"], ["42", "0"], ["42", "01"], ["42", "x"], ["42", "1", "--bogus"], ["42", "1", "--integration", "y"]],
)
def test_usage_errors(repo: Repo, args: list[str]) -> None:
    res = repo.setup(*args)
    assert res.returncode == 1
    assert not repo.worktrees.exists() or not any(repo.worktrees.iterdir())


# ---------------------------------------------------------------------------
# AC 3 — item mode
# ---------------------------------------------------------------------------


@pytest.fixture
def deliverable(repo: Repo) -> Repo:
    """``QS_42`` as a plain local branch with one commit past main (not a task worktree)."""
    repo.git("branch", "QS_42", "main")
    wt = repo.root / "tmp-qs42"
    repo.git("worktree", "add", "-q", str(wt), "QS_42")
    (wt / "deliv.txt").write_text("deliverable\n")
    repo.git("add", "deliv.txt", cwd=wt)
    repo.git("commit", "-q", "-m", "deliverable work", cwd=wt)
    repo.git("worktree", "remove", str(wt))
    return repo


def test_item_mode_creates_a_local_item_branch(deliverable: Repo) -> None:
    repo = deliverable
    # origin moves on main; a fetch would update refs/remotes/origin/main.
    other = repo.root / "other"
    subprocess.run(["git", "clone", "-q", str(repo.origin), str(other)], env=repo.env, check=True)
    (other / "x.txt").write_text("x\n")
    repo.git("add", "x.txt", cwd=other)
    repo.git("commit", "-q", "-m", "remote move", cwd=other)
    repo.git("push", "-q", "origin", "main", cwd=other)
    before = repo.rev("refs/remotes/origin/main")

    res = repo.setup("42", "1")
    assert res.returncode == 0, res.stdout + res.stderr
    wt = repo.worktrees / "QS_42_1"
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD", cwd=wt).stdout.strip() == "QS_42_1"
    assert repo.rev("HEAD", cwd=wt) == repo.rev("refs/heads/QS_42")
    assert upstream(repo, wt) == ""
    assert "QS_42_1" not in origin_heads(repo)
    assert repo.rev("refs/remotes/origin/main") == before
    assert_seeded(wt)
    assert "local branch, not published" in res.stdout
    assert "Fetching" not in res.stdout

    again = repo.setup("42", "1")
    assert again.returncode == 0
    assert "already set up on QS_42_1 (local branch)" in again.stdout


@pytest.mark.parametrize("content", [None, ".DS_Store"])
def test_item_mode_recovers_a_stale_directory(deliverable: Repo, content: str | None) -> None:
    repo = deliverable
    stale = repo.worktrees / "QS_42_1"
    stale.mkdir(parents=True)
    if content:
        (stale / content).write_text("")
    res = repo.setup("42", "1")
    assert res.returncode == 0, res.stdout + res.stderr
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD", cwd=stale).stdout.strip() == "QS_42_1"


def test_item_mode_without_deliverable_fails_fresh(repo: Repo) -> None:
    res = repo.setup("42", "1")
    assert res.returncode == 1
    assert "deliverable branch QS_42 not found" in res.stdout
    assert repo.git("show-ref", "--verify", "--quiet", "refs/heads/QS_42_1", check=False).returncode != 0


def test_item_mode_reuse_arm_requires_the_deliverable(deliverable: Repo) -> None:
    repo = deliverable
    repo.git("branch", "QS_42_1", "QS_42")
    repo.git("branch", "-D", "QS_42")
    res = repo.setup("42", "1")
    assert res.returncode == 1
    assert "deliverable branch QS_42 not found" in res.stdout
    assert not (repo.worktrees / "QS_42_1").exists()


def test_item_mode_healthy_item_still_reports_set_up_without_deliverable(deliverable: Repo) -> None:
    repo = deliverable
    assert repo.setup("42", "1").returncode == 0
    repo.git("branch", "-D", "QS_42")
    res = repo.setup("42", "1")
    assert res.returncode == 0
    assert "already set up" in res.stdout


def test_item_mode_recovery_without_deliverable_touches_nothing(deliverable: Repo) -> None:
    repo = deliverable
    stale = repo.worktrees / "QS_42_1"
    stale.mkdir(parents=True)
    repo.git("branch", "-D", "QS_42")
    res = repo.setup("42", "1")
    assert res.returncode == 1
    assert "deliverable branch QS_42 not found" in res.stdout
    assert stale.is_dir()


def test_item_mode_reuses_an_existing_item_branch(deliverable: Repo) -> None:
    repo = deliverable
    assert repo.setup("42", "1").returncode == 0
    wt = repo.worktrees / "QS_42_1"
    (wt / "item.txt").write_text("item\n")
    repo.git("add", "item.txt", cwd=wt)
    repo.git("commit", "-q", "-m", "item work", cwd=wt)
    tip = repo.rev("HEAD", cwd=wt)
    repo.git("worktree", "remove", "--force", str(wt))
    res = repo.setup("42", "1")
    assert res.returncode == 0, res.stdout + res.stderr
    assert repo.rev("HEAD", cwd=wt) == tip
    assert "diverged from main" not in res.stdout


def test_item_branch_checked_out_elsewhere_fails(deliverable: Repo) -> None:
    repo = deliverable
    repo.git("branch", "QS_42_1", "QS_42")
    elsewhere = repo.root / "elsewhere"
    repo.git("worktree", "add", "-q", str(elsewhere), "QS_42_1")
    res = repo.setup("42", "1")
    assert res.returncode != 0


# ---------------------------------------------------------------------------
# AC 9 — integration mode
# ---------------------------------------------------------------------------


def test_integration_mode_creates_a_detached_scratch(deliverable: Repo) -> None:
    repo = deliverable
    branches_before = repo.git("for-each-ref", "--format=%(refname)", "refs/heads").stdout
    res = repo.setup("42", "1", "--integration")
    assert res.returncode == 0, res.stdout + res.stderr
    scratch = repo.worktrees / "QS_42_1_integration"
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD", cwd=scratch).stdout.strip() == "HEAD"
    assert repo.rev("HEAD", cwd=scratch) == repo.rev("refs/heads/QS_42")
    assert_seeded(scratch, full=False)
    assert repo.git("for-each-ref", "--format=%(refname)", "refs/heads").stdout == branches_before
    assert origin_heads(repo) == {"main"}
    assert "integration scratch, detached at QS_42" in res.stdout
    assert repo.git("status", "--porcelain", cwd=scratch).stdout == ""
    # The fake interpreter is reached through the venv symlink.
    assert subprocess.run([str(scratch / "venv" / "bin" / "python")], check=False).returncode == 0

    (scratch / "marker").write_text("keep me\n")
    second = repo.setup("42", "1", "--integration")
    assert second.returncode == 3
    assert "integration scratch already exists" in second.stdout
    assert (scratch / "marker").read_text() == "keep me\n"


def test_integration_mode_registration_without_directory_exits_3(deliverable: Repo) -> None:
    repo = deliverable
    assert repo.setup("42", "1", "--integration").returncode == 0
    scratch = repo.worktrees / "QS_42_1_integration"
    shutil.rmtree(scratch)
    res = repo.setup("42", "1", "--integration")
    assert res.returncode == 3


def test_integration_mode_without_deliverable_exits_1(repo: Repo) -> None:
    res = repo.setup("42", "1", "--integration")
    assert res.returncode == 1
    assert "deliverable branch QS_42 not found" in res.stdout
    assert not (repo.worktrees / "QS_42_1_integration").exists()


def test_integration_mode_alongside_the_item_worktree(deliverable: Repo) -> None:
    """The item worktree holds QS_42_1; the scratch is detached, so both coexist."""
    repo = deliverable
    assert repo.setup("42", "1").returncode == 0
    res = repo.setup("42", "1", "--integration")
    assert res.returncode == 0, res.stdout + res.stderr
