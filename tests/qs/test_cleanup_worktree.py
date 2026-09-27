"""Tests for ``scripts/qs/cleanup_worktree.py --delete-branch`` (QS-340).

The epic × factory lane's ``finish-task`` discards the short-lived epic
worktree AND its local branch, so a re-entry starts fresh from
``origin/main``. The main worktree must be resolved **before** the
worktree is removed — afterwards the process cwd (often the worktree
itself) is gone and ``git worktree list`` dies.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "qs" / "cleanup_worktree.py"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def main_and_worktree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    cfg = tmp_path / "gitconfig"
    cfg.write_text("[user]\n\tname = T\n\temail = t@example.invalid\n[init]\n\tdefaultBranch = main\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    main = tmp_path / "repo"
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    (main / "README.md").write_text("x\n")
    _git(main, "add", ".")
    _git(main, "commit", "-q", "-m", "init")
    work = tmp_path / "repo-worktrees" / "QS_77"
    _git(main, "worktree", "add", "-q", "-b", "QS_77", str(work))
    return main, work


def _branches(main: Path) -> list[str]:
    return _git(main, "branch", "--format=%(refname:short)").split()


def _run_main(monkeypatch: pytest.MonkeyPatch, capsys, argv: list[str]) -> dict:
    import cleanup_worktree

    monkeypatch.setattr("sys.argv", ["cleanup_worktree.py", *argv])
    cleanup_worktree.main()
    return json.loads(capsys.readouterr().out)


def test_delete_branch_removes_worktree_and_local_branch(
    main_and_worktree, monkeypatch, capsys
) -> None:
    main, work = main_and_worktree
    monkeypatch.chdir(main)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "removed"
    assert out["branch_deleted"] is True
    assert not work.exists()
    assert "QS_77" not in _branches(main)


def test_without_the_flag_the_branch_is_kept(main_and_worktree, monkeypatch, capsys) -> None:
    main, work = main_and_worktree
    monkeypatch.chdir(main)
    out = _run_main(monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force"])
    assert out["status"] == "removed"
    assert out["branch_deleted"] is False
    assert "QS_77" in _branches(main)


def test_launched_from_inside_the_worktree_still_deletes_the_branch(main_and_worktree) -> None:
    """Process cwd == work_dir: the git call must run with ``cwd=main_wt``."""
    main, work = main_and_worktree
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--work-dir", str(work), "--issue", "77",
         "--force", "--delete-branch"],
        cwd=work, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["branch_deleted"] is True, out
    assert "QS_77" not in _branches(main)


def test_mismatched_issue_does_not_delete_an_unrelated_branch(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S3(d): --force with a wrong --issue must not force-delete another QS_<N> branch."""
    main, work = main_and_worktree  # worktree is on QS_77
    monkeypatch.chdir(main)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "78", "--force", "--delete-branch"]
    )
    assert out["status"] == "removed-branch-kept"
    assert out["branch_deleted"] is False
    assert "QS_78" in out["branch_delete_error"]
    assert "QS_77" in out["branch_delete_error"]
    assert "QS_77" in _branches(main)  # the real branch is preserved
    assert not work.exists()


def test_worktree_removal_failure_keeps_the_branch(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S3(a): a failed worktree removal must not proceed to delete the branch."""
    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    monkeypatch.setattr(cleanup_worktree, "remove_worktree", lambda wd: "boom removing")
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "error"
    assert out["branch_deleted"] is False
    assert "QS_77" in _branches(main)  # branch preserved because removal failed


def test_delete_branch_when_worktree_dir_already_gone(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S3(c): a failed first run left the dir gone but the branch behind — retry works."""
    import shutil

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    shutil.rmtree(work)  # the dir is gone, but the worktree registration is stale
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "removed"
    assert out["branch_deleted"] is True
    assert "QS_77" not in _branches(main)


def test_dir_gone_without_delete_branch_still_errors(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S3(c): without --delete-branch a missing dir still reports the old error."""
    import shutil

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    shutil.rmtree(work)
    out = _run_main(monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force"])
    assert out["status"] == "error"
    assert "does not exist" in out["message"]


def test_main_worktree_lookup_failure_is_reported(main_and_worktree, monkeypatch, capsys) -> None:
    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)

    def boom():
        raise RuntimeError("No git worktrees found")

    monkeypatch.setattr(cleanup_worktree, "get_main_worktree", boom)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["branch_deleted"] is False
    assert "No git worktrees found" in out["branch_delete_error"]


@pytest.mark.parametrize("branch", ["main", "master"])
def test_protected_branches_are_refused(tmp_path: Path, branch: str) -> None:
    import cleanup_worktree

    deleted, error = cleanup_worktree.delete_local_branch(tmp_path, branch)
    assert deleted is False
    assert "refusing" in (error or "")


def test_dry_run_announces_the_branch_deletion(main_and_worktree, monkeypatch, capsys) -> None:
    main, work = main_and_worktree
    monkeypatch.chdir(main)
    out = _run_main(
        monkeypatch, capsys,
        ["--work-dir", str(work), "--issue", "77", "--dry-run", "--delete-branch"],
    )
    assert out["status"] == "dry_run"
    assert out["would_delete_branch"] == "QS_77"
    assert work.exists() and "QS_77" in _branches(main)
