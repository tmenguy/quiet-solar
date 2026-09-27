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


def test_mismatched_issue_touches_nothing(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S3: --force with a wrong --issue must not remove the worktree OR delete a
    branch — it refuses and leaves everything intact."""
    main, work = main_and_worktree  # worktree is on QS_77
    monkeypatch.chdir(main)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "78", "--force", "--delete-branch"]
    )
    assert out["status"] == "error"
    assert out["branch_deleted"] is False
    assert "QS_78" in out["branch_delete_error"]
    assert "QS_77" in out["branch_delete_error"]
    assert "QS_77" in _branches(main)  # the real branch is preserved
    assert work.exists()  # S3: the wrong worktree survives


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


# --- S3: unknown HEAD (detached / unreadable) -------------------------------


def test_detached_head_worktree_is_removed_but_branch_kept(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S3: a detached-HEAD worktree owned by QS_77 (per the porcelain listing) is
    removed, but the branch is kept because HEAD couldn't confirm it."""
    main, work = main_and_worktree
    monkeypatch.chdir(main)
    subprocess.run(["git", "-C", str(work), "checkout", "--detach"], check=True, capture_output=True)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "removed-branch-kept"
    assert out["branch_deleted"] is False
    assert not work.exists()
    assert "QS_77" in _branches(main)  # kept: HEAD was unknown


def test_unreadable_head_removes_worktree_but_keeps_branch(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S3: an unreadable HEAD (None) with the porcelain registration confirming
    QS_77 removes the worktree but keeps the branch."""
    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    monkeypatch.setattr(cleanup_worktree, "_current_branch", lambda wd: None)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "removed-branch-kept"
    assert not work.exists()
    assert "QS_77" in _branches(main)


def test_unknown_head_registered_on_another_branch_refuses(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S3: unknown HEAD + a porcelain registration on another branch refuses
    entirely and touches nothing."""
    import cleanup_worktree

    main, work = main_and_worktree  # registered on QS_77
    monkeypatch.chdir(main)
    monkeypatch.setattr(cleanup_worktree, "_current_branch", lambda wd: None)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "78", "--force", "--delete-branch"]
    )
    assert out["status"] == "error"
    assert work.exists()
    assert "QS_77" in _branches(main)


def test_unknown_head_without_a_main_worktree_refuses(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S3: unknown HEAD and no resolvable main worktree — can't confirm ownership,
    so refuse and touch nothing."""
    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    monkeypatch.setattr(cleanup_worktree, "_current_branch", lambda wd: None)

    def boom():
        raise RuntimeError("No git worktrees found")

    monkeypatch.setattr(cleanup_worktree, "get_main_worktree", boom)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "error"
    assert work.exists()


# --- N4: removal error but the dir is actually gone -------------------------


def test_removal_error_with_dir_gone_still_deletes_branch(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """N4: remove_worktree reports an error but the dir is gone — prune and
    proceed to the guarded branch delete."""
    import shutil

    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)

    def fake_remove(wd):
        shutil.rmtree(wd)  # the dir really is gone, but registration is left stale
        return "git worktree remove hiccup"

    monkeypatch.setattr(cleanup_worktree, "remove_worktree", fake_remove)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "removed"
    assert out["branch_deleted"] is True
    assert out["worktree_remove_error"] == "git worktree remove hiccup"
    assert "QS_77" not in _branches(main)


# --- S4: dir-gone retry safety ---------------------------------------------


def test_dir_gone_retry_requires_force(main_and_worktree, monkeypatch, capsys) -> None:
    """S4: without --force, a missing dir keeps the old 'does not exist' error."""
    import shutil

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    shutil.rmtree(work)
    out = _run_main(monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--delete-branch"])
    assert out["status"] == "error"
    assert "does not exist" in out["message"]
    assert "QS_77" in _branches(main)  # the branch is untouched


def test_dir_gone_mismatch_refuses(main_and_worktree, monkeypatch, capsys) -> None:
    """S4: a gone dir whose stale registration is on another branch refuses."""
    import shutil

    main, work = main_and_worktree  # registered on QS_77
    monkeypatch.chdir(main)
    shutil.rmtree(work)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "78", "--force", "--delete-branch"]
    )
    assert out["status"] == "error"
    assert "QS_77" in _branches(main)


def test_dir_gone_with_no_registration_keeps_the_branch(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S4: a gone dir with NO registration proves nothing — keep the branch."""
    import shutil

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    shutil.rmtree(work)
    subprocess.run(["git", "-C", str(main), "worktree", "prune"], check=True, capture_output=True)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "removed-branch-kept"
    assert out["branch_deleted"] is False
    assert "QS_77" in _branches(main)  # kept: nothing proved it was ours


def test_dir_gone_retry_main_worktree_lookup_failure(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S7: get_main_worktree failing in the retry path is reported, branch kept."""
    import shutil

    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    shutil.rmtree(work)

    def boom():
        raise RuntimeError("no worktrees at all")

    # N2: the retry path now resolves the main worktree via
    # ``_main_worktree_or_fallback`` — patch that so BOTH the cwd lookup and the
    # script-location fallback are simulated as failing.
    monkeypatch.setattr(cleanup_worktree, "_main_worktree_or_fallback", boom)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "removed-branch-kept"
    assert "no worktrees at all" in out["branch_delete_error"]


# --- S4/S7: branch-delete outcomes -----------------------------------------


def test_delete_branch_after_removal_reports_an_absent_branch(main_and_worktree) -> None:
    """S4: a branch that is already gone counts as deleted (a re-run isn't a failure)."""
    import cleanup_worktree

    main, _work = main_and_worktree
    result = cleanup_worktree._delete_branch_after_removal(main, "QS_999")
    assert result["branch_deleted"] is True
    assert result["branch_absent"] is True


def test_delete_branch_after_removal_detects_checked_out_elsewhere(main_and_worktree) -> None:
    """S4: deleting a branch still checked out in a live worktree is flagged."""
    import cleanup_worktree

    main, _work = main_and_worktree  # QS_77 is checked out in the live worktree
    result = cleanup_worktree._delete_branch_after_removal(main, "QS_77")
    assert result["branch_deleted"] is False
    assert result["checked_out_elsewhere"] is True


def test_branch_checked_out_elsewhere_surfaces_a_distinct_status(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S4: when branch -D fails because the branch is checked out elsewhere, the
    cleanup reports the distinct branch-checked-out-elsewhere status."""
    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    real_delete = cleanup_worktree.delete_local_branch
    monkeypatch.setattr(
        cleanup_worktree,
        "delete_local_branch",
        lambda mw, br: (False, "error: cannot delete branch used by worktree at '/x'"),
    )
    del real_delete
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "branch-checked-out-elsewhere"
    assert out["branch_deleted"] is False


def test_branch_delete_failure_is_reported(main_and_worktree, monkeypatch, capsys) -> None:
    """S7/AC7: a genuine `git branch -D` failure (after the S4 exists-check) is
    reported as removed-branch-kept with git's stderr."""
    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    real_run = subprocess.run

    def fake_run(cmd, *a, **k):
        if cmd[:1] == ["git"] and "branch" in cmd and "-D" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="fatal: could not remove ref")
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(cleanup_worktree.subprocess, "run", fake_run)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "removed-branch-kept"
    assert "could not remove ref" in out["branch_delete_error"]


# --- S1: detached stale registration on the dir-gone retry path -------------


def test_dir_gone_detached_registration_keeps_the_branch(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S1: dir gone + a stale DETACHED registration proves nothing — the branch
    is kept (consistent with the live detached path), never force-deleted by a
    mistyped --issue."""
    import shutil

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    subprocess.run(["git", "-C", str(work), "checkout", "--detach"], check=True, capture_output=True)
    shutil.rmtree(work)  # dir gone; the registration is now detached and stale
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "removed-branch-kept"
    assert out["branch_deleted"] is False
    assert "QS_77" in _branches(main)  # kept: a detached registration proves nothing
    assert "detached" in out["branch_delete_error"]


# --- M1: never rmtree the main checkout or an unregistered path --------------


def test_m1_refuses_to_remove_the_main_checkout(main_and_worktree, monkeypatch, capsys) -> None:
    """M1: --work-dir <main> must never delete the main checkout (.git and all)."""
    main, work = main_and_worktree
    _git(main, "checkout", "-q", "-b", "QS_9")  # put main on a QS_ branch
    monkeypatch.chdir(main)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(main), "--issue", "9", "--force", "--delete-branch"]
    )
    assert out["status"] == "error"
    assert main.exists() and (main / ".git").exists()


def test_m1_refuses_a_subdirectory_of_a_linked_worktree(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """M1: a subdirectory of a worktree is not itself a registered worktree."""
    main, work = main_and_worktree
    sub = work / "sub"
    sub.mkdir()
    (sub / "keep.txt").write_text("keep\n")
    monkeypatch.chdir(main)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(sub), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "error"
    assert sub.exists()


def test_m1_refuses_an_unrelated_clone(main_and_worktree, monkeypatch, capsys) -> None:
    """M1: an unrelated clone that happens to be on QS_N is not registered in this
    repo's worktree list — refuse."""
    main, work = main_and_worktree
    other = main.parent / "unrelated"
    other.mkdir()
    _git(other, "init", "-q", "-b", "main")
    (other / "a.txt").write_text("x\n")
    _git(other, "add", ".")
    _git(other, "commit", "-q", "-m", "init")
    _git(other, "checkout", "-q", "-b", "QS_77")
    monkeypatch.chdir(main)  # the process runs from THIS repo's main checkout
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(other), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "error"
    assert other.exists()


def test_m1_refuses_when_main_worktree_undeterminable(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """M1 (c): with no resolvable main worktree, ownership can't be proven — refuse
    and touch nothing (never rmtree)."""
    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)

    def boom():
        raise RuntimeError("No git worktrees found")

    monkeypatch.setattr(cleanup_worktree, "get_main_worktree", boom)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "error"
    assert work.exists()


# --- M1: remove_worktree is itself a gatekeeper (non-delete-branch path) -----


def test_remove_worktree_refuses_the_main_checkout(main_and_worktree, monkeypatch) -> None:
    import cleanup_worktree

    main, _work = main_and_worktree
    monkeypatch.chdir(main)
    error = cleanup_worktree.remove_worktree(main)
    assert error is not None and "main checkout" in error
    assert main.exists() and (main / ".git").exists()


def test_remove_worktree_refuses_an_unregistered_path(main_and_worktree, monkeypatch) -> None:
    import cleanup_worktree

    main, work = main_and_worktree
    sub = work / "sub"
    sub.mkdir()
    monkeypatch.chdir(main)
    error = cleanup_worktree.remove_worktree(sub)
    assert error is not None and "not a registered linked worktree" in error
    assert sub.exists()


def test_remove_worktree_refuses_an_unreadable_registration(
    main_and_worktree, monkeypatch
) -> None:
    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    monkeypatch.setattr(cleanup_worktree, "_registered_branch", lambda mw, wd: ("unreadable", None))
    error = cleanup_worktree.remove_worktree(work)
    assert error is not None and "unreadable" in error
    assert work.exists()


def test_remove_worktree_refuses_when_main_undeterminable(main_and_worktree, monkeypatch) -> None:
    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)

    def boom():
        raise RuntimeError("No git worktrees found")

    monkeypatch.setattr(cleanup_worktree, "get_main_worktree", boom)
    error = cleanup_worktree.remove_worktree(work)
    assert error is not None and "Could not determine main worktree" in error
    assert work.exists()


def test_non_delete_branch_removal_refuses_the_main_checkout(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """M1: the plain removal path (no --delete-branch) is guarded by
    remove_worktree too."""
    main, work = main_and_worktree
    monkeypatch.chdir(main)
    out = _run_main(monkeypatch, capsys, ["--work-dir", str(main), "--issue", "77", "--force"])
    assert out["status"] == "error"
    assert main.exists() and (main / ".git").exists()


# --- N1: _current_branch is not confused by a same-named tag ----------------


def test_current_branch_ignores_a_same_named_tag(main_and_worktree) -> None:
    """N1: with a tag QS_77 alongside branch QS_77, `git branch --show-current`
    still returns the bare branch name (rev-parse --abbrev-ref would say
    heads/QS_77)."""
    import cleanup_worktree

    main, work = main_and_worktree
    subprocess.run(["git", "-C", str(work), "tag", "QS_77"], check=True, capture_output=True)
    assert cleanup_worktree._current_branch(work) == "QS_77"


def test_current_branch_is_none_when_detached(main_and_worktree) -> None:
    import cleanup_worktree

    main, work = main_and_worktree
    subprocess.run(["git", "-C", str(work), "checkout", "--detach"], check=True, capture_output=True)
    assert cleanup_worktree._current_branch(work) is None


def test_current_branch_is_none_when_read_fails(tmp_path) -> None:
    """A non-git directory makes `git branch --show-current` exit non-zero → None."""
    import cleanup_worktree

    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    assert cleanup_worktree._current_branch(not_a_repo) is None


# --- N2: main-worktree fallback resolves from the script's own location ------


def test_main_worktree_or_fallback_uses_script_location(monkeypatch) -> None:
    """N2: when the cwd lookup fails, the fallback anchors at this script's repo."""
    import cleanup_worktree

    def boom():
        raise RuntimeError("cwd gone")

    monkeypatch.setattr(cleanup_worktree, "get_main_worktree", boom)
    result = cleanup_worktree._main_worktree_or_fallback()
    assert result.exists()


def test_main_worktree_or_fallback_reraises_when_both_fail(monkeypatch) -> None:
    """N2: if the fallback listing also fails, the original error propagates."""
    import cleanup_worktree

    def boom():
        raise RuntimeError("boom-original")

    monkeypatch.setattr(cleanup_worktree, "get_main_worktree", boom)
    real_run = subprocess.run

    def fake_run(cmd, *a, **k):
        if "worktree" in cmd and "list" in cmd:
            return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal")
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(cleanup_worktree.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="boom-original"):
        cleanup_worktree._main_worktree_or_fallback()


# --- N5: _registered_branch compares resolved paths -------------------------


def test_registered_branch_resolves_symlinked_paths(main_and_worktree, tmp_path) -> None:
    """N5: a path reaching the worktree through a symlink still matches its
    registration (byte-different string, same resolved directory)."""
    import cleanup_worktree

    main, work = main_and_worktree
    alias = tmp_path / "alias"
    alias.symlink_to(work.parent)  # alias -> <repo>-worktrees
    via_link = alias / work.name
    assert str(via_link) != str(work)
    kind, branch = cleanup_worktree._registered_branch(main, via_link)
    assert kind == "branch" and branch == "QS_77"
