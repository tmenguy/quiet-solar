"""Tests for ``scripts/qs/cleanup_worktree.py --delete-branch`` (QS-340).

The epic × factory lane's ``finish-task`` discards the short-lived epic
worktree AND its local branch, so a re-entry starts fresh from
``origin/main``. The main worktree must be resolved **before** the
worktree is removed — afterwards the process cwd (often the worktree
itself) is gone and ``git worktree list`` dies.
"""

from __future__ import annotations

import json
import os
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
    # N4: a worktree-level refusal is reported in worktree_remove_error.
    assert "QS_78" in out["worktree_remove_error"]
    assert "QS_77" in out["worktree_remove_error"]
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

    # N1: _cleanup_with_branch resolves via _main_worktree_or_fallback now, so
    # patch that to simulate an undeterminable main worktree.
    monkeypatch.setattr(cleanup_worktree, "_main_worktree_or_fallback", boom)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["branch_deleted"] is False
    # N4: an undeterminable-main refusal is a worktree-level refusal.
    assert "No git worktrees found" in out["worktree_remove_error"]


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
    # N3: the kept-branch reason names the detached registration, not "unknown HEAD".
    assert "detached registration" in out["branch_delete_error"]


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
    # N3: an unreadable HEAD is reported as such, not a generic "unknown HEAD".
    assert "HEAD unreadable" in out["branch_delete_error"]


def test_head_mismatches_registration_keeps_branch_with_accurate_reason(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """N3: a live HEAD that disagrees with the registration keeps the branch, and
    the reason names the mismatching HEAD rather than a fixed '(unknown HEAD)'."""
    import cleanup_worktree

    main, work = main_and_worktree  # registered on QS_77
    monkeypatch.chdir(main)
    monkeypatch.setattr(cleanup_worktree, "_current_branch", lambda wd: "QS_other")
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "removed-branch-kept"
    assert not work.exists()
    assert "QS_77" in _branches(main)
    assert "HEAD `QS_other` does not match the registration" in out["branch_delete_error"]


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

    # N1: the live path resolves via _main_worktree_or_fallback.
    monkeypatch.setattr(cleanup_worktree, "_main_worktree_or_fallback", boom)
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

    # N1: ownership resolution goes through _main_worktree_or_fallback.
    monkeypatch.setattr(cleanup_worktree, "_main_worktree_or_fallback", boom)
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

    # N1: remove_worktree resolves the main worktree via the fallback.
    monkeypatch.setattr(cleanup_worktree, "_main_worktree_or_fallback", boom)
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


# --- S3 (#04): a detached registration needs an --issue dir-name match ---------


def test_detached_registration_mismatched_dir_name_refuses(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S3 (#04): a detached registration is not proof of --issue ownership. With a
    dir name that is not QS_<issue>, removal is refused and nothing is touched.
    (The matching-name case is covered by
    ``test_detached_head_worktree_is_removed_but_branch_kept``.)"""
    main, work = main_and_worktree  # dir name is QS_77
    monkeypatch.chdir(main)
    subprocess.run(["git", "-C", str(work), "checkout", "--detach"], check=True, capture_output=True)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "78", "--force", "--delete-branch"]
    )
    assert out["status"] == "error"
    assert work.exists()
    assert "detached" in out["worktree_remove_error"]
    assert "QS_78" in out["worktree_remove_error"]
    assert "QS_77" in _branches(main)


# --- S4 (#04): leftover dir from a partially failed remove is retryable --------


def test_leftover_dir_after_failed_remove_is_cleaned_on_retry(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S4 (#04): a partially failed ``git worktree remove`` can drop the
    registration (and its admin dir) yet leave the worktree directory — with its
    ``.git`` file — behind. A re-run recognises that leftover via its ``.git``
    file's gitdir and removes it (no ``git worktree remove``).

    The leftover state is built deterministically (M1 #06): relying on git's
    partial-failure order — which subdir it deletes before hitting a locked one —
    is platform- and version-dependent (macOS kept ``.git``, the Linux CI runner
    deleted it first), so instead we save the ``.git`` file, run a real successful
    ``git worktree remove --force``, then recreate exactly the surviving state."""
    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    # Save the worktree's .git file, then remove the worktree for real. This
    # succeeds and prunes both the admin dir and the directory itself.
    saved_dotgit = (work / ".git").read_text()
    subprocess.run(
        ["git", "-C", str(main), "worktree", "remove", str(work), "--force"],
        check=True,
        capture_output=True,
    )
    common = cleanup_worktree._git_common_dir(main)
    assert common is not None
    admin = Path(common) / "worktrees" / "QS_77"
    # Recreate exactly the leftover a partial failure would leave: the directory
    # with its original .git file plus a content file, registration dropped and
    # admin dir gone.
    work.mkdir(parents=True, exist_ok=True)
    (work / ".git").write_text(saved_dotgit)
    (work / "leftover.txt").write_text("x\n")
    assert cleanup_worktree._registered_branch(main, work)[0] == "absent"
    assert not admin.exists()
    # The re-run proves the leftover is ours and removes it (no git worktree remove).
    out = _run_main(monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force"])
    assert out["status"] == "removed", out
    assert not work.exists()


def test_absent_leftover_with_foreign_gitdir_is_refused(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S4 (#04): an 'absent' dir whose ``.git`` gitdir points into another repo is
    not ours — refuse and leave it (and its contents) intact."""
    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    subprocess.run(
        ["git", "-C", str(main), "worktree", "remove", str(work), "--force"],
        check=True,
        capture_output=True,
    )
    work.mkdir(parents=True, exist_ok=True)
    other = main.parent / "elsewhere"
    (work / ".git").write_text(f"gitdir: {other}/.git/worktrees/QS_77\n")
    (work / "keep.txt").write_text("keep\n")
    error = cleanup_worktree.remove_worktree(work)
    assert error is not None and "not a registered linked worktree" in error
    assert work.exists() and (work / "keep.txt").exists()


def test_abandoned_gitdir_rejects_a_missing_and_a_directory_dotgit(
    main_and_worktree, tmp_path
) -> None:
    """S4 (#04): the ownership proof rejects a missing ``.git`` and a ``.git``
    *directory* (a clone / main checkout), and accepts only a matching gitdir file."""
    import cleanup_worktree

    main, _work = main_and_worktree
    # missing .git
    bare = tmp_path / "bare"
    bare.mkdir()
    assert cleanup_worktree._abandoned_worktree_gitdir(main, bare) is False
    # a .git directory (a clone)
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / ".git").mkdir()
    assert cleanup_worktree._abandoned_worktree_gitdir(main, clone) is False
    # a matching gitdir file resolves into this repo's worktrees dir AND its
    # admin dir is gone (a genuine leftover, S1 #05) → accepted
    common = cleanup_worktree._git_common_dir(main)
    assert common is not None
    ours = tmp_path / "ours"
    ours.mkdir()
    (ours / ".git").write_text(f"gitdir: {common}/worktrees/QS_gone\n")  # admin dir pruned
    assert cleanup_worktree._abandoned_worktree_gitdir(main, ours) is True


# --- S1 (#05): a plain-``mv``'d worktree keeps a live admin dir → refuse -------


def test_mvd_worktree_with_live_admin_dir_is_refused(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S1 (#05): a worktree relocated with plain ``mv`` (not ``git worktree move``)
    keeps a *live* admin dir registered under its old path. Its new path looks
    unregistered ('absent') and its ``.git`` gitdir still resolves under this
    repo's ``worktrees/``, so the #04 proof passed and would rmtree it. The admin
    dir still existing must now refuse it, preserving its uncommitted work."""
    main, work = main_and_worktree
    monkeypatch.chdir(main)
    (work / "wip.txt").write_text("precious\n")  # uncommitted work we must not lose
    moved = work.parent / "QS_77_moved"
    work.rename(moved)  # plain mv: <common>/worktrees/QS_77 admin dir stays live
    # sanity: the moved path is unregistered, its .git still points into our repo
    assert cleanup_worktree_module()._registered_branch(main, moved)[0] == "absent"
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(moved), "--issue", "77", "--force"]
    )
    assert out["status"] == "error"
    assert moved.exists() and (moved / "wip.txt").read_text() == "precious\n"


def cleanup_worktree_module():
    import cleanup_worktree

    return cleanup_worktree


def test_abandoned_gitdir_rejects_a_live_admin_dir(main_and_worktree, tmp_path) -> None:
    """S1 (#05): a matching gitdir whose admin dir still EXISTS (a moved/copied
    worktree, not a leftover) is refused — only a leftover whose admin dir git
    already deleted is accepted."""
    import cleanup_worktree

    main, _work = main_and_worktree
    common = cleanup_worktree._git_common_dir(main)
    assert common is not None
    live = tmp_path / "live"
    live.mkdir()
    (live / ".git").write_text(f"gitdir: {common}/worktrees/QS_77\n")  # QS_77 admin dir is live
    assert cleanup_worktree._abandoned_worktree_gitdir(main, live) is False


# --- S2 (#05): unit tests for the #04 defensive branches ----------------------


def test_leftover_rmtree_failure_is_reported(main_and_worktree, monkeypatch) -> None:
    """S2 (#05): the S4 leftover path reports an rmtree failure instead of
    swallowing it (``remove_worktree`` lines ~128-132)."""
    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    monkeypatch.setattr(cleanup_worktree, "_registered_branch", lambda mw, wd: ("absent", None))
    monkeypatch.setattr(cleanup_worktree, "_abandoned_worktree_gitdir", lambda mw, wd: True)
    monkeypatch.setattr(cleanup_worktree, "_prune_worktrees", lambda mw: None)

    def boom(*_a, **_k):
        raise OSError("locked subtree")

    monkeypatch.setattr(cleanup_worktree.shutil, "rmtree", boom)
    error = cleanup_worktree.remove_worktree(work)
    assert error is not None and "shutil.rmtree failed" in error and "locked subtree" in error
    assert work.exists()


def test_git_common_dir_is_none_on_failure(main_and_worktree, monkeypatch) -> None:
    """S2 (#05): ``_git_common_dir`` returns None when rev-parse exits non-zero
    (line ~327)."""
    import cleanup_worktree

    main, _work = main_and_worktree
    real_run = subprocess.run

    def fake_run(cmd, *a, **k):
        if cmd[:1] == ["git"] and "rev-parse" in cmd and "--git-common-dir" in cmd:
            return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal")
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(cleanup_worktree.subprocess, "run", fake_run)
    assert cleanup_worktree._git_common_dir(main) is None


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root can read a 000 file, so the unreadable-.git branch can't be provoked",
)
def test_abandoned_gitdir_rejects_an_unreadable_dotgit(main_and_worktree, tmp_path) -> None:
    """S2 (#05): an unreadable ``.git`` file returns False (lines ~347-349)."""
    import cleanup_worktree

    main, _work = main_and_worktree
    d = tmp_path / "unreadable"
    d.mkdir()
    dotgit = d / ".git"
    dotgit.write_text("gitdir: /somewhere\n")
    os.chmod(dotgit, 0o000)
    try:
        assert cleanup_worktree._abandoned_worktree_gitdir(main, d) is False
    finally:
        os.chmod(dotgit, 0o644)


def test_abandoned_gitdir_rejects_missing_empty_and_relative_gitdir(
    main_and_worktree, tmp_path
) -> None:
    """S2 (#05): a ``.git`` file with no ``gitdir:`` line (line ~353), an empty
    value (line ~356) and a relative gitdir resolved against work_dir (line ~359)
    all return False."""
    import cleanup_worktree

    main, _work = main_and_worktree
    no_line = tmp_path / "no_line"
    no_line.mkdir()
    (no_line / ".git").write_text("something: else\n")
    assert cleanup_worktree._abandoned_worktree_gitdir(main, no_line) is False

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / ".git").write_text("gitdir:   \n")
    assert cleanup_worktree._abandoned_worktree_gitdir(main, empty) is False

    relative = tmp_path / "relative"
    relative.mkdir()
    (relative / ".git").write_text("gitdir: ../elsewhere/.git/worktrees/QS_1\n")
    assert cleanup_worktree._abandoned_worktree_gitdir(main, relative) is False


def test_abandoned_gitdir_false_when_common_dir_is_none(
    main_and_worktree, tmp_path, monkeypatch
) -> None:
    """S2 (#05): with a valid absolute gitdir but no resolvable common dir, the
    proof returns False (line ~362)."""
    import cleanup_worktree

    main, _work = main_and_worktree
    d = tmp_path / "ok"
    d.mkdir()
    (d / ".git").write_text("gitdir: /abs/repo/.git/worktrees/QS_1\n")
    monkeypatch.setattr(cleanup_worktree, "_git_common_dir", lambda mw: None)
    assert cleanup_worktree._abandoned_worktree_gitdir(main, d) is False


def test_task_path_prune_exception_is_swallowed(main_and_worktree, monkeypatch, capsys) -> None:
    """S2 (#05): the N2 task-path prune swallows a main-worktree resolution error
    and still reports ``removed`` (lines ~587-588)."""
    import shutil

    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)

    def fake_remove(wd):
        shutil.rmtree(wd)  # dir really gone; registration left stale
        return "git worktree remove hiccup"

    monkeypatch.setattr(cleanup_worktree, "remove_worktree", fake_remove)

    def boom():
        raise RuntimeError("no main worktree")

    monkeypatch.setattr(cleanup_worktree, "_main_worktree_or_fallback", boom)
    out = _run_main(monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force"])
    assert out["status"] == "removed"
    assert out["worktree_remove_error"] == "git worktree remove hiccup"


# --- S5 (#04): coverage for the newly reachable branches ----------------------


def test_registered_branch_reports_unreadable_when_list_fails(
    main_and_worktree, monkeypatch
) -> None:
    """S5: a failing ``git worktree list --porcelain`` maps to 'unreadable'."""
    import cleanup_worktree

    main, work = main_and_worktree
    real_run = subprocess.run

    def fake_run(cmd, *a, **k):
        if cmd[:1] == ["git"] and "worktree" in cmd and "list" in cmd:
            return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal")
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(cleanup_worktree.subprocess, "run", fake_run)
    kind, branch = cleanup_worktree._registered_branch(main, work)
    assert kind == "unreadable" and branch is None


def test_remove_worktree_reports_a_git_remove_failure_with_dir_present(
    main_and_worktree, monkeypatch
) -> None:
    """S5: git worktree remove fails and the dir is still present — the error is
    returned (the rmtree fallback also fails), nothing silently dropped."""
    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    real_run = subprocess.run

    def fake_run(cmd, *a, **k):
        if cmd[:1] == ["git"] and "worktree" in cmd and "remove" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="fatal: cannot remove")
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(cleanup_worktree.subprocess, "run", fake_run)

    def failing_rmtree(*_a, **_k):
        raise OSError("nope")

    monkeypatch.setattr(cleanup_worktree.shutil, "rmtree", failing_rmtree)
    error = cleanup_worktree.remove_worktree(work)
    assert error is not None
    assert "cannot remove" in error and "rmtree failed" in error
    assert work.exists()


def test_dir_gone_retry_branch_checked_out_elsewhere(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S5: the retry path — the dir is gone, the registration proves QS_77, but
    branch -D reports it's checked out elsewhere → branch-checked-out-elsewhere."""
    import shutil

    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    shutil.rmtree(work)
    monkeypatch.setattr(
        cleanup_worktree,
        "delete_local_branch",
        lambda mw, br: (False, "error: cannot delete branch used by worktree at '/x'"),
    )
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "branch-checked-out-elsewhere", out
    assert "QS_77" in _branches(main)


def test_dir_gone_retry_branch_delete_fails_generically(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """S5: the retry path — the dir is gone, the registration proves QS_77, but
    branch -D fails for a non-checkout reason → removed-branch-kept."""
    import shutil

    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)
    shutil.rmtree(work)
    monkeypatch.setattr(
        cleanup_worktree,
        "delete_local_branch",
        lambda mw, br: (False, "fatal: could not remove ref"),
    )
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "removed-branch-kept", out
    assert "could not remove ref" in out["branch_delete_error"]
    assert "QS_77" in _branches(main)


# --- N1 (#04): the live paths use the main-worktree fallback -------------------


def test_remove_worktree_resolves_main_via_fallback(main_and_worktree, monkeypatch) -> None:
    """N1: remove_worktree resolves the main worktree via
    ``_main_worktree_or_fallback`` (not ``get_main_worktree`` directly), so a cwd
    inside the removed worktree doesn't break resolution."""
    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)

    def boom():
        raise RuntimeError("cwd gone")

    monkeypatch.setattr(cleanup_worktree, "get_main_worktree", boom)
    monkeypatch.setattr(cleanup_worktree, "_main_worktree_or_fallback", lambda: main)
    error = cleanup_worktree.remove_worktree(work)
    assert error is None
    assert not work.exists()


def test_cleanup_with_branch_resolves_main_via_fallback(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """N1: the --delete-branch live path resolves via ``_main_worktree_or_fallback``."""
    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)

    def boom():
        raise RuntimeError("cwd gone")

    monkeypatch.setattr(cleanup_worktree, "get_main_worktree", boom)
    monkeypatch.setattr(cleanup_worktree, "_main_worktree_or_fallback", lambda: main)
    out = _run_main(
        monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force", "--delete-branch"]
    )
    assert out["status"] == "removed"
    assert not work.exists()
    assert "QS_77" not in _branches(main)


# --- N2 (#04): task path — rmtree succeeded but remove_worktree flagged error --


def test_removal_error_with_dir_gone_reports_removed_task_path(
    main_and_worktree, monkeypatch, capsys
) -> None:
    """N2 (#04): the non-delete-branch path mirrors the --delete-branch N4 — when
    remove_worktree reports an error but the dir is actually gone, prune the stale
    registration and report ``removed`` with the error noted."""
    import shutil

    import cleanup_worktree

    main, work = main_and_worktree
    monkeypatch.chdir(main)

    def fake_remove(wd):
        shutil.rmtree(wd)  # the dir really is gone, registration left stale
        return "git worktree remove hiccup"

    monkeypatch.setattr(cleanup_worktree, "remove_worktree", fake_remove)
    out = _run_main(monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--force"])
    assert out["status"] == "removed"
    assert out["worktree_remove_error"] == "git worktree remove hiccup"
    # the stale registration was pruned
    listing = _git(main, "worktree", "list", "--porcelain")
    assert str(work) not in listing


# --- QS-400 D5: item cleanup (``--item K``) ---------------------------------


@pytest.fixture
def main_and_item(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A repo with a ``QS_77`` branch (no worktree) and the item worktree
    ``repo-worktrees/QS_77_1`` on ``QS_77_1``, forked from ``refs/heads/QS_77``."""
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
    _git(main, "branch", "QS_77")
    item = tmp_path / "repo-worktrees" / "QS_77_1"
    _git(main, "worktree", "add", "-q", "-b", "QS_77_1", str(item), "refs/heads/QS_77")
    monkeypatch.chdir(main)
    return main, item


def _commit_on_item(item: Path, name: str = "work.txt") -> None:
    (item / name).write_text(f"{name}\n")
    _git(item, "add", name)
    _git(item, "commit", "-q", "-m", f"item: {name}")


def _merge_item_into_deliverable(main: Path) -> None:
    """A real merge commit ``QS_77 <- QS_77_1`` built without a worktree on QS_77."""
    tree = _git(main, "rev-parse", "refs/heads/QS_77_1^{tree}").strip()
    merge = _git(
        main, "commit-tree", tree, "-p", "refs/heads/QS_77", "-p", "refs/heads/QS_77_1", "-m", "merge QS_77_1"
    ).strip()
    _git(main, "update-ref", "refs/heads/QS_77", merge)


def _tip(main: Path, branch: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=main,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _item_argv(work: Path, *flags: str) -> list[str]:
    return ["--work-dir", str(work), "--issue", "77", "--item", "1", *flags]


_ITEM_KEYS = {
    "status",
    "message",
    "worktree_path",
    "worktree_removed",
    "worktree_absent",
    "worktree_remove_error",
    "stale_directory",
    "branch",
    "branch_deleted",
    "branch_absent",
    "branch_kept_reason",
    "branch_delete_error",
    "unintegrated_commits",
    "deleted_tip",
    "uncommitted_files",
    "detached",
    "detached_head",
    "options",
}


def _run_item(monkeypatch: pytest.MonkeyPatch, capsys, work: Path, *flags: str) -> dict:
    out = _run_main(monkeypatch, capsys, _item_argv(work, *flags))
    assert set(out) == _ITEM_KEYS, out
    assert out["branch"] == "QS_77_1"
    return out


_OWNERSHIP_FLAGS = [
    pytest.param((), id="no-flags"),
    pytest.param(("--force",), id="force"),
    pytest.param(("--force", "--delete-branch", "--discard-unintegrated"), id="force-delete-discard"),
]


def _ownership_target(kind: str, main: Path, tmp_path: Path) -> Path:
    if kind == "deliverable-worktree":
        target = tmp_path / "repo-worktrees" / "QS_77"
        _git(main, "worktree", "add", "-q", str(target), "QS_77")
    elif kind == "other-branch":
        target = tmp_path / "repo-worktrees" / "QS_88_1"
        _git(main, "worktree", "add", "-q", "-b", "QS_88_1", str(target))
    elif kind == "detached-other-name":
        target = tmp_path / "repo-worktrees" / "QS_77_2"
        _git(main, "worktree", "add", "-q", "--detach", str(target))
    elif kind == "main-checkout":
        target = main
    else:  # unregistered directory with another name
        target = tmp_path / "somewhere" / "QS_77_9"
        target.mkdir(parents=True)
    return target.resolve()


@pytest.mark.parametrize(
    "kind",
    ["deliverable-worktree", "other-branch", "detached-other-name", "main-checkout", "unregistered-other-name"],
)
@pytest.mark.parametrize("flags", _OWNERSHIP_FLAGS)
def test_item_ownership_refusals_touch_nothing(
    main_and_item, tmp_path, monkeypatch, capsys, kind: str, flags: tuple[str, ...]
) -> None:
    main, item = main_and_item
    _commit_on_item(item)
    target = _ownership_target(kind, main, tmp_path)
    before = (_tip(main, "QS_77"), _tip(main, "QS_77_1"))
    out = _run_item(monkeypatch, capsys, target, *flags)
    assert out["status"] == "error", out
    assert out["worktree_removed"] is False
    assert out["branch_deleted"] is False
    assert target.exists()
    assert item.exists()
    assert (_tip(main, "QS_77"), _tip(main, "QS_77_1")) == before


def test_item_ownership_error_names_the_registration(main_and_item, tmp_path, monkeypatch, capsys) -> None:
    main, _item = main_and_item
    target = _ownership_target("deliverable-worktree", main, tmp_path)
    out = _run_item(monkeypatch, capsys, target, "--force")
    assert "QS_77" in out["message"]
    assert "nothing was touched" in out["message"]


def test_item_unreadable_registration_is_refused(main_and_item, monkeypatch, capsys) -> None:
    import cleanup_worktree

    main, item = main_and_item
    monkeypatch.setattr(cleanup_worktree, "_registered_branch", lambda m, w: ("unreadable", None))
    out = _run_item(monkeypatch, capsys, item, "--force")
    assert out["status"] == "error"
    assert "unreadable" in out["message"]
    assert item.exists() and _tip(main, "QS_77_1") is not None


@pytest.mark.parametrize("flags", [(), ("--delete-branch",)])
def test_item_work_dir_absent_but_checked_out_elsewhere_is_an_error(
    main_and_item, tmp_path, monkeypatch, capsys, flags: tuple[str, ...]
) -> None:
    main, item = main_and_item
    typo = tmp_path / "elsewhere" / "QS_77_1"
    out = _run_item(monkeypatch, capsys, typo, *flags)
    assert out["status"] == "error"
    assert str(item.resolve()) in out["message"]
    assert out["worktree_absent"] is False
    assert out["branch_deleted"] is False
    assert item.exists() and _tip(main, "QS_77_1") is not None


def test_item_main_worktree_lookup_failure_is_an_error(main_and_item, monkeypatch, capsys) -> None:
    import cleanup_worktree

    main, item = main_and_item

    def boom() -> Path:
        raise RuntimeError("No git worktrees found")

    monkeypatch.setattr(cleanup_worktree, "_main_worktree_or_fallback", boom)
    out = _run_item(monkeypatch, capsys, item, "--force", "--delete-branch")
    assert out["status"] == "error"
    assert "No git worktrees found" in out["message"]
    assert item.exists() and _tip(main, "QS_77_1") is not None


def test_item_clean_worktree_is_removed_without_force(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    out = _run_item(monkeypatch, capsys, item)
    assert out["status"] == "removed"
    assert out["worktree_removed"] is True
    assert out["worktree_absent"] is False
    assert out["detached"] is False
    assert out["detached_head"] is None
    assert out["options"] == {}
    assert not item.exists()
    assert _tip(main, "QS_77_1") is not None


def test_item_unintegrated_without_delete_branch_keeps_the_branch_silently(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    _commit_on_item(item)
    tip = _tip(main, "QS_77_1")
    out = _run_item(monkeypatch, capsys, item)
    assert out["status"] == "removed"
    assert out["worktree_removed"] is True
    assert out["branch_kept_reason"] is None
    assert out["unintegrated_commits"] is None
    assert not item.exists()
    assert _tip(main, "QS_77_1") == tip


def test_item_uncommitted_files_require_force(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    (item / "scratch.txt").write_text("wip\n")
    out = _run_item(monkeypatch, capsys, item)
    assert out["status"] == "action_required"
    assert "--force" in out["options"]
    assert any("scratch.txt" in f for f in out["uncommitted_files"])
    assert out["worktree_removed"] is False
    assert item.exists()


def test_item_uncommitted_files_with_force_are_removed(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    (item / "scratch.txt").write_text("wip\n")
    out = _run_item(monkeypatch, capsys, item, "--force")
    assert out["status"] == "removed"
    assert out["worktree_removed"] is True
    assert not item.exists()
    assert _tip(main, "QS_77_1") is not None  # --force never deletes the branch


def test_item_uncommitted_files_with_delete_branch_stop_before_the_branch(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    (item / "scratch.txt").write_text("wip\n")
    tip = _tip(main, "QS_77_1")
    out = _run_item(monkeypatch, capsys, item, "--delete-branch")
    assert out["status"] == "action_required"
    assert out["branch_deleted"] is False
    assert out["deleted_tip"] is None
    assert item.exists()
    assert _tip(main, "QS_77_1") == tip


def test_item_detached_requires_force(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    _git(item, "checkout", "-q", "--detach")
    out = _run_item(monkeypatch, capsys, item)
    assert out["status"] == "action_required"
    assert out["detached"] is True
    assert out["detached_head"] is None  # nothing removed
    assert "--force" in out["options"]
    assert item.exists()


def test_item_detached_with_force_is_removed(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    _git(item, "checkout", "-q", "--detach")
    _commit_on_item(item, "detached.txt")  # a commit only the detached HEAD reaches
    head = _git(item, "rev-parse", "HEAD").strip()
    out = _run_item(monkeypatch, capsys, item, "--force")
    assert out["status"] == "removed"
    assert out["worktree_removed"] is True
    assert out["detached_head"] == head  # the undo point, read before the removal
    assert f"git branch <name> {head}" in out["message"]
    assert not item.exists()
    assert _tip(main, "QS_77_1") is not None and _tip(main, "QS_77_1") != head


def test_item_detached_removal_failure_records_no_detached_head(main_and_item, monkeypatch, capsys) -> None:
    import cleanup_worktree

    main, item = main_and_item
    _git(item, "checkout", "-q", "--detach")
    monkeypatch.setattr(cleanup_worktree, "remove_worktree", lambda wd: "boom removing")
    out = _run_item(monkeypatch, capsys, item, "--force")
    assert out["status"] == "error"
    assert out["detached_head"] is None
    assert item.exists()


def test_item_status_failure_is_an_error(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    (item / ".git").write_text("gitdir: /nonexistent/for/sure\n")
    out = _run_item(monkeypatch, capsys, item)
    assert out["status"] == "error"
    assert item.exists() and _tip(main, "QS_77_1") is not None


def test_item_removal_failure_with_dir_present_is_an_error(main_and_item, monkeypatch, capsys) -> None:
    import cleanup_worktree

    main, item = main_and_item
    monkeypatch.setattr(cleanup_worktree, "remove_worktree", lambda wd: "boom removing")
    out = _run_item(monkeypatch, capsys, item, "--delete-branch")
    assert out["status"] == "error"
    assert out["worktree_remove_error"] == "boom removing"
    assert out["branch_deleted"] is False
    assert _tip(main, "QS_77_1") is not None


def test_item_removal_error_with_dir_gone_prunes_and_continues(main_and_item, monkeypatch, capsys) -> None:
    import shutil

    import cleanup_worktree

    main, item = main_and_item

    def fake_remove(wd: Path) -> str:
        shutil.rmtree(wd)
        return "git worktree remove hiccup"

    monkeypatch.setattr(cleanup_worktree, "remove_worktree", fake_remove)
    out = _run_item(monkeypatch, capsys, item, "--delete-branch")
    assert out["status"] == "removed"
    assert out["worktree_removed"] is True
    assert out["worktree_remove_error"] == "git worktree remove hiccup"
    assert out["branch_deleted"] is True  # the prune cleared the registration
    assert _tip(main, "QS_77_1") is None


# --- --delete-branch outcomes -------------------------------------------------


def test_item_integrated_branch_is_deleted(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    _commit_on_item(item)
    _merge_item_into_deliverable(main)
    tip = _tip(main, "QS_77_1")
    out = _run_item(monkeypatch, capsys, item, "--delete-branch")
    assert out["status"] == "removed"
    assert out["worktree_removed"] is True
    assert out["branch_deleted"] is True
    assert out["deleted_tip"] == tip
    assert out["unintegrated_commits"] == 0
    assert out["branch_kept_reason"] is None
    assert _tip(main, "QS_77_1") is None
    assert _tip(main, "QS_77") is not None


def test_item_unintegrated_branch_is_kept(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    _commit_on_item(item, "a.txt")
    _commit_on_item(item, "b.txt")
    tip = _tip(main, "QS_77_1")
    out = _run_item(monkeypatch, capsys, item, "--delete-branch")
    assert out["status"] == "removed-branch-kept"
    assert out["worktree_removed"] is True
    assert out["branch_kept_reason"] == "unintegrated"
    assert out["unintegrated_commits"] == 2
    assert out["branch_deleted"] is False
    assert out["deleted_tip"] is None
    assert out["message"].endswith("; branch kept")
    assert not item.exists()
    assert _tip(main, "QS_77_1") == tip


def test_item_unintegrated_branch_with_discard_is_deleted(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    _commit_on_item(item, "a.txt")
    _commit_on_item(item, "b.txt")
    tip = _tip(main, "QS_77_1")
    out = _run_item(monkeypatch, capsys, item, "--delete-branch", "--discard-unintegrated")
    assert out["status"] == "removed"
    assert out["branch_deleted"] is True
    assert out["deleted_tip"] == tip
    assert out["unintegrated_commits"] == 2
    assert out["branch_kept_reason"] is None
    assert _tip(main, "QS_77_1") is None


def test_item_deliverable_missing_keeps_the_branch(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    _git(main, "branch", "-D", "QS_77")
    tip = _tip(main, "QS_77_1")
    out = _run_item(monkeypatch, capsys, item, "--delete-branch")
    assert out["status"] == "removed-branch-kept"
    assert out["branch_kept_reason"] == "deliverable-missing"
    assert out["unintegrated_commits"] is None
    assert out["message"].endswith("; branch kept")
    assert _tip(main, "QS_77_1") == tip


def test_item_deliverable_missing_with_discard_deletes(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    _git(main, "branch", "-D", "QS_77")
    tip = _tip(main, "QS_77_1")
    out = _run_item(monkeypatch, capsys, item, "--delete-branch", "--discard-unintegrated")
    assert out["status"] == "removed"
    assert out["branch_deleted"] is True
    assert out["deleted_tip"] == tip
    assert _tip(main, "QS_77_1") is None


def test_item_count_failure_keeps_the_branch(main_and_item, monkeypatch, capsys) -> None:
    import cleanup_worktree

    main, item = main_and_item
    monkeypatch.setattr(cleanup_worktree, "_unintegrated_count", lambda g, d, tip: -1)
    out = _run_item(monkeypatch, capsys, item, "--delete-branch")
    assert out["status"] == "removed-branch-kept"
    assert out["branch_kept_reason"] == "count-failed"
    assert out["unintegrated_commits"] is None
    assert out["message"].endswith("; branch kept")
    assert _tip(main, "QS_77_1") is not None


def test_item_counts_the_tip_read_before_the_count_and_reports_it(main_and_item, monkeypatch, capsys) -> None:
    """The tip is read first; the count runs on that sha, so ``deleted_tip`` is the commit proven integrated."""
    import cleanup_worktree

    main, item = main_and_item
    _commit_on_item(item)
    _merge_item_into_deliverable(main)
    tip = _tip(main, "QS_77_1")
    real = cleanup_worktree._unintegrated_count
    seen: list[tuple[str, str | None]] = []

    def spy(git_dir: Path, deliverable: str, item_tip: str | None) -> int:
        seen.append((deliverable, item_tip))
        n = real(git_dir, deliverable, item_tip)
        _git(main, "update-ref", "refs/heads/QS_77_1", "refs/heads/main")  # the branch moves after the count
        return n

    monkeypatch.setattr(cleanup_worktree, "_unintegrated_count", spy)
    out = _run_item(monkeypatch, capsys, item, "--delete-branch")
    assert seen == [("QS_77", tip)]
    assert out["status"] == "removed" and out["unintegrated_commits"] == 0
    assert out["deleted_tip"] == tip
    assert f"git branch QS_77_1 {tip}" in out["message"]


def test_item_unreadable_tip_keeps_the_branch(main_and_item, monkeypatch, capsys) -> None:
    import cleanup_worktree

    main, item = main_and_item
    monkeypatch.setattr(cleanup_worktree, "_branch_tip", lambda g, b: None)
    out = _run_item(monkeypatch, capsys, item, "--delete-branch")
    assert out["status"] == "removed-branch-kept"
    assert out["branch_kept_reason"] == "count-failed"
    assert out["deleted_tip"] is None
    assert _tip(main, "QS_77_1") is not None


def test_item_delete_failure_keeps_the_branch(main_and_item, monkeypatch, capsys) -> None:
    """The item stays checked out (a locked registration survives the prune) at its
    default path, and ``--work-dir`` is that path, missing: ``git branch -D`` refuses."""
    import shutil

    main, item = main_and_item
    _git(main, "worktree", "lock", str(item))
    shutil.rmtree(item)
    tip = _tip(main, "QS_77_1")
    out = _run_item(monkeypatch, capsys, item, "--delete-branch")
    assert out["status"] == "removed-branch-kept"
    assert out["worktree_absent"] is True
    assert out["branch_kept_reason"] == "delete-failed"
    assert out["branch_deleted"] is False
    assert "QS_77_1" in out["branch_delete_error"]
    assert out["deleted_tip"] == tip  # recorded before the attempt
    assert out["message"].endswith("; branch kept")
    assert _tip(main, "QS_77_1") == tip


# --- worktree already gone ----------------------------------------------------


def _remove_item_registration(main: Path, item: Path) -> None:
    _git(main, "worktree", "remove", "--force", str(item))


def test_item_pruned_worktree_integrated_branch_is_deleted(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    _commit_on_item(item)
    _merge_item_into_deliverable(main)
    _remove_item_registration(main, item)
    out = _run_item(monkeypatch, capsys, item, "--delete-branch")
    assert out["status"] == "removed"
    assert out["worktree_absent"] is True
    assert out["worktree_removed"] is False
    assert out["branch_deleted"] is True
    assert _tip(main, "QS_77_1") is None


def test_item_pruned_worktree_unintegrated_branch_is_kept(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    _commit_on_item(item)
    _remove_item_registration(main, item)
    out = _run_item(monkeypatch, capsys, item, "--delete-branch")
    assert out["status"] == "removed-branch-kept"
    assert out["worktree_absent"] is True
    assert out["branch_kept_reason"] == "unintegrated"
    assert out["unintegrated_commits"] == 1
    assert _tip(main, "QS_77_1") is not None


def test_item_pruned_worktree_unintegrated_with_discard_is_deleted(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    _commit_on_item(item)
    _remove_item_registration(main, item)
    out = _run_item(monkeypatch, capsys, item, "--delete-branch", "--discard-unintegrated")
    assert out["status"] == "removed"
    assert out["branch_deleted"] is True
    assert out["unintegrated_commits"] == 1
    assert _tip(main, "QS_77_1") is None


def test_item_rm_rf_worktree_with_kept_registration_integrated_is_deleted(main_and_item, monkeypatch, capsys) -> None:
    import shutil

    main, item = main_and_item
    _commit_on_item(item)
    _merge_item_into_deliverable(main)
    shutil.rmtree(item)
    out = _run_item(monkeypatch, capsys, item, "--delete-branch")
    assert out["status"] == "removed"
    assert out["worktree_absent"] is True
    assert out["branch_deleted"] is True
    assert out["branch_kept_reason"] is None
    assert _tip(main, "QS_77_1") is None
    assert str(item) not in _git(main, "worktree", "list", "--porcelain")


def test_item_dir_gone_without_delete_branch_is_removed(main_and_item, monkeypatch, capsys) -> None:
    import shutil

    main, item = main_and_item
    shutil.rmtree(item)
    out = _run_item(monkeypatch, capsys, item)
    assert out["status"] == "removed"
    assert out["worktree_absent"] is True
    assert out["worktree_removed"] is False
    assert out["branch_deleted"] is False
    assert _tip(main, "QS_77_1") is not None
    assert str(item) not in _git(main, "worktree", "list", "--porcelain")


def test_item_stale_leftover_is_left_and_the_branch_deleted(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    _remove_item_registration(main, item)
    item.mkdir(parents=True)
    (item / "leftover.txt").write_text("x\n")
    out = _run_item(monkeypatch, capsys, item, "--delete-branch")
    assert out["status"] == "removed"
    assert out["stale_directory"] == str(item.resolve())
    assert out["worktree_removed"] is False
    assert out["branch_deleted"] is True
    assert (item / "leftover.txt").exists()
    assert _tip(main, "QS_77_1") is None


def test_item_unregistered_named_dir_while_item_checked_out_elsewhere_is_refused(
    main_and_item, tmp_path, monkeypatch, capsys
) -> None:
    """A directory named ``QS_77_1`` is not a stale leftover while the item is
    registered at another path: it is an unregistered directory → error."""
    main, item = main_and_item
    other = tmp_path / "elsewhere" / "QS_77_1"
    other.mkdir(parents=True)
    out = _run_item(monkeypatch, capsys, other, "--delete-branch", "--discard-unintegrated", "--force")
    assert out["status"] == "error"
    assert other.exists() and item.exists()
    assert _tip(main, "QS_77_1") is not None


def test_item_unregistered_dir_with_another_name_is_refused_when_item_unregistered(
    main_and_item, tmp_path, monkeypatch, capsys
) -> None:
    """With no worktree on the item at all, an unregistered directory with another
    name is still refused: only a directory named ``QS_77_1`` is a stale leftover."""
    main, item = main_and_item
    _remove_item_registration(main, item)
    other = tmp_path / "somewhere" / "QS_77_9"
    other.mkdir(parents=True)
    out = _run_item(monkeypatch, capsys, other, "--delete-branch", "--discard-unintegrated", "--force")
    assert out["status"] == "error"
    assert "not a registered worktree" in out["message"]
    assert other.exists()
    assert _tip(main, "QS_77_1") is not None


def test_item_ref_already_gone_is_removed(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    _remove_item_registration(main, item)
    _git(main, "branch", "-D", "QS_77_1")
    out = _run_item(monkeypatch, capsys, item, "--delete-branch")
    assert out["status"] == "removed"
    assert out["branch_absent"] is True
    assert out["branch_deleted"] is False
    assert out["worktree_absent"] is True


# --- misuse and dry run -------------------------------------------------------


@pytest.mark.parametrize("dry", [(), ("--dry-run",)])
@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--item", "1", "--push-first"], id="push-first-with-item"),
        pytest.param(["--item", "1", "--discard-unintegrated"], id="discard-without-delete-branch"),
        pytest.param(["--delete-branch", "--discard-unintegrated"], id="discard-without-item"),
        pytest.param(["--discard-unintegrated"], id="discard-alone"),
    ],
)
def test_item_misuse_is_refused(main_and_item, monkeypatch, capsys, argv: list[str], dry: tuple[str, ...]) -> None:
    main, item = main_and_item
    before = (_tip(main, "QS_77"), _tip(main, "QS_77_1"))
    out = _run_main(monkeypatch, capsys, ["--work-dir", str(item), "--issue", "77", *argv, *dry])
    assert out["status"] == "error"
    assert item.exists()
    assert (_tip(main, "QS_77"), _tip(main, "QS_77_1")) == before


def test_item_dry_run_announces_the_item_branch(main_and_item, monkeypatch, capsys) -> None:
    main, item = main_and_item
    out = _run_main(monkeypatch, capsys, _item_argv(item, "--dry-run", "--delete-branch"))
    assert out["status"] == "dry_run"
    assert out["would_delete_branch"] == "QS_77_1"
    assert out["item"] == 1
    assert item.exists() and _tip(main, "QS_77_1") is not None


def test_task_dry_run_has_no_item_key(main_and_worktree, monkeypatch, capsys) -> None:
    main, work = main_and_worktree
    monkeypatch.chdir(main)
    out = _run_main(monkeypatch, capsys, ["--work-dir", str(work), "--issue", "77", "--dry-run"])
    assert "item" not in out


@pytest.mark.parametrize("raw", ["0", "01", "x"])
def test_item_rejects_a_non_positive_integer(main_and_item, monkeypatch, raw: str) -> None:
    import cleanup_worktree

    _main, item = main_and_item
    monkeypatch.setattr("sys.argv", ["cleanup_worktree.py", "--work-dir", str(item), "--issue", "77", "--item", raw])
    with pytest.raises(SystemExit):
        cleanup_worktree.main()


# --- helpers ------------------------------------------------------------------


def test_unintegrated_count_counts_and_fails_on_a_missing_ref(main_and_item) -> None:
    import cleanup_worktree

    main, item = main_and_item
    assert cleanup_worktree._unintegrated_count(main, "QS_77", _tip(main, "QS_77_1")) == 0
    _commit_on_item(item)
    tip = _tip(main, "QS_77_1")
    assert cleanup_worktree._unintegrated_count(main, "QS_77", tip) == 1
    assert cleanup_worktree._unintegrated_count(main, "QS_77", "f" * 40) == -1
    assert cleanup_worktree._unintegrated_count(main, "QS_77", None) == -1
    assert cleanup_worktree._unintegrated_count(main, "QS_404", tip) == -1


def test_branch_tip_reads_the_sha_or_none(main_and_item) -> None:
    import cleanup_worktree

    main, _item = main_and_item
    assert cleanup_worktree._branch_tip(main, "QS_77_1") == _tip(main, "QS_77_1")
    assert cleanup_worktree._branch_tip(main, "QS_77_404") is None


def test_unintegrated_count_is_minus_one_on_unparsable_output(main_and_item, monkeypatch) -> None:
    import cleanup_worktree

    main, _item = main_and_item
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="not a number\n", stderr="")
    monkeypatch.setattr(cleanup_worktree.subprocess, "run", lambda *a, **k: fake)
    assert cleanup_worktree._unintegrated_count(main, "QS_77", "f" * 40) == -1


def test_item_registration_finds_the_item_worktree(main_and_item, tmp_path) -> None:
    import cleanup_worktree

    main, item = main_and_item
    assert cleanup_worktree._item_registration(main, "QS_77_1") == item.resolve()
    assert cleanup_worktree._item_registration(main, "QS_77_2") is None
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    assert cleanup_worktree._item_registration(not_a_repo, "QS_77_1") is None
