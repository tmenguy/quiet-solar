"""Real-git tests for ``scripts/worktree-setup.sh``.

The ``QS_<N>`` task path: a branch cut from ``origin/main``, published with an
upstream, the worktree seeded from the main checkout; a re-run is a no-op.

The fixture is a bare ``origin`` plus a clone holding a copy of the script,
with an isolated git config. The script runs under ``/bin/bash`` when it
exists (macOS bash 3.2), else ``bash``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.qs._gitrepo import Repo, make_repo


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[Repo]:
    yield make_repo(tmp_path)


def assert_seeded(wt: Path) -> None:
    assert (wt / "venv").is_symlink()
    assert (wt / "config" / "secrets.yaml").is_symlink()
    assert (wt / "custom_components" / "other").is_symlink()
    for cache in (".mypy_cache", ".testmondata"):
        assert (wt / cache).exists()
        assert not (wt / cache).is_symlink()


def upstream(repo: Repo, cwd: Path) -> str:
    res = repo.git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", cwd=cwd, check=False)
    return res.stdout.strip() if res.returncode == 0 else ""


def origin_heads(repo: Repo) -> set[str]:
    out = repo.git("ls-remote", "--heads", str(repo.origin)).stdout
    return {line.split()[1].removeprefix("refs/heads/") for line in out.splitlines()}


def test_task_mode(repo: Repo) -> None:
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


@pytest.mark.parametrize("args", [[], ["x"], ["42", "1"]])
def test_usage_errors(repo: Repo, args: list[str]) -> None:
    res = repo.setup(*args)
    assert res.returncode == 1
    assert not repo.worktrees.exists() or not any(repo.worktrees.iterdir())
