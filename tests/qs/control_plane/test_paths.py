"""Checkpoint 2: main-checkout detection, the path guard, the backup dir (AC4)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from control_plane import errors, paths

from .conftest import REAL_CODE_ROOT, REAL_MAIN_CHECKOUT, REAL_MAIN_HEAD_BRANCH, SCRIPTS_QS


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


class TestMainCheckout:
    def test_code_root_is_the_repo(self) -> None:
        assert REAL_CODE_ROOT() == SCRIPTS_QS.parents[1]

    def test_git_directory_layout(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        assert REAL_MAIN_CHECKOUT(tmp_path) == tmp_path

    def test_linked_worktree_layout(self, tmp_path: Path) -> None:
        main = tmp_path / "repo"
        main.mkdir()
        _git("init", "-q", "-b", "main", cwd=main)
        _git("-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q", "--allow-empty", "-m", "x", cwd=main)
        _git("worktree", "add", "-q", "-b", "QS_1", str(tmp_path / "wt"), cwd=main)
        assert REAL_MAIN_CHECKOUT(tmp_path / "wt") == main.resolve()
        assert REAL_MAIN_CHECKOUT(main) == main

    def test_relative_gitdir_without_commondir(self, tmp_path: Path) -> None:
        (tmp_path / "gd").mkdir()
        wt = tmp_path / "wt"
        wt.mkdir()
        (wt / ".git").write_text("gitdir: ../gd\n")
        assert REAL_MAIN_CHECKOUT(wt) == tmp_path.resolve()

    @pytest.mark.parametrize("content", [None, "nonsense\n"])
    def test_not_a_checkout(self, tmp_path: Path, content: str | None) -> None:
        if content is not None:
            (tmp_path / ".git").write_text(content)
        with pytest.raises(errors.CpError) as exc:
            REAL_MAIN_CHECKOUT(tmp_path)
        assert exc.value.code == "PATH_GUARD"

    def test_is_main_checkout_and_main(self, fake_main: Path, tmp_path: Path, monkeypatch) -> None:
        assert paths.main() == fake_main
        assert paths.is_main_checkout(fake_main)
        monkeypatch.setattr(paths, "main_checkout", REAL_MAIN_CHECKOUT)
        (tmp_path / "x").mkdir()
        (tmp_path / "x" / ".git").write_text("gitdir: ../main/.git\n")
        assert not paths.is_main_checkout(tmp_path / "x")


class TestMainHeadBranch:
    @pytest.mark.parametrize(
        ("head", "expected"),
        [
            ("ref: refs/heads/main\n", "main"),
            ("ref: refs/heads/QS_399\n", None),
            ("0123456789abcdef0123456789abcdef01234567\n", None),
            (None, None),
        ],
        ids=["main", "feature", "detached", "missing"],
    )
    def test_head(self, tmp_path: Path, head: str | None, expected: str | None) -> None:
        (tmp_path / ".git").mkdir()
        if head is not None:
            (tmp_path / ".git" / "HEAD").write_text(head)
        assert REAL_MAIN_HEAD_BRANCH(tmp_path) == expected


class TestLiveDb:
    def test_live_db_and_detection(self, fake_main: Path, tmp_path: Path) -> None:
        assert paths.live_db(fake_main) == fake_main / "harness_state.db"
        assert paths.is_live_db_path(fake_main / "harness_state.db")
        assert not paths.is_live_db_path(tmp_path / "harness_state.db")
        assert not paths.is_live_db_path(fake_main / "other.db")

    def test_sidecar(self, tmp_path: Path) -> None:
        assert paths.sidecar(tmp_path / "a.db", ".migrate.lock") == tmp_path / "a.db.migrate.lock"


class TestSelectDb:
    def test_temporary_db_works(self, tmp_path: Path) -> None:
        assert paths.select_db() == (tmp_path / "state" / "test_state.db").resolve()

    def test_worktree_code_naming_main_live_db(self, monkeypatch, fake_main: Path, tmp_path: Path) -> None:
        monkeypatch.delenv("PYTEST_CURRENT_TEST")
        monkeypatch.setattr(paths, "code_root", lambda: tmp_path / "wt")
        monkeypatch.setenv("QS_CP_DB", str(fake_main / "harness_state.db"))
        with pytest.raises(errors.CpError) as exc:
            paths.select_db()
        assert exc.value.code == "PATH_GUARD" and exc.value.exit_code == 7
        assert "QS_CP_DB" in exc.value.extra["hint"]

    def test_worktree_code_without_env(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.delenv("PYTEST_CURRENT_TEST")
        monkeypatch.delenv("QS_CP_DB")
        monkeypatch.setattr(paths, "code_root", lambda: tmp_path / "wt")
        with pytest.raises(errors.CpError) as exc:
            paths.select_db()
        assert exc.value.code == "PATH_GUARD"

    def test_main_code_naming_another_checkouts_db(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.delenv("PYTEST_CURRENT_TEST")
        other = tmp_path / "other"
        (other / ".git").mkdir(parents=True)
        monkeypatch.setenv("QS_CP_DB", str(other / "harness_state.db"))
        with pytest.raises(errors.CpError) as exc:
            paths.select_db()
        assert exc.value.code == "PATH_GUARD"

    def test_any_live_db_under_pytest(self, monkeypatch, fake_main: Path) -> None:
        monkeypatch.setenv("QS_CP_DB", str(fake_main / "harness_state.db"))
        with pytest.raises(errors.CpError) as exc:
            paths.select_db()
        assert exc.value.code == "PATH_GUARD" and "pytest" in exc.value.detail
        monkeypatch.delenv("QS_CP_DB")
        with pytest.raises(errors.CpError) as exc:
            paths.select_db()
        assert exc.value.code == "PATH_GUARD" and "pytest" in exc.value.detail

    def test_main_code_opens_its_own_live_db(self, monkeypatch, fake_main: Path) -> None:
        monkeypatch.delenv("PYTEST_CURRENT_TEST")
        monkeypatch.setenv("QS_CP_DB", str(fake_main / "harness_state.db"))
        assert paths.select_db() == fake_main / "harness_state.db"
        monkeypatch.delenv("QS_CP_DB")
        assert paths.select_db() == fake_main / "harness_state.db"


class TestBackupDir:
    def test_env(self, tmp_path: Path) -> None:
        assert paths.backup_dir() == (tmp_path / "backups").resolve()

    def test_default_under_home(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.delenv("QS_CP_BACKUP_DIR")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        assert paths.backup_dir() == (tmp_path / "home" / ".local/state/quiet-solar/cp-backups").resolve()

    @pytest.mark.parametrize("where", ["main", "code"])
    def test_inside_a_checkout_is_refused(self, monkeypatch, tmp_path: Path, fake_main: Path, where: str) -> None:
        code = tmp_path / "wt"
        monkeypatch.setattr(paths, "code_root", lambda: code)
        target = fake_main / "bk" if where == "main" else code / "bk"
        monkeypatch.setenv("QS_CP_BACKUP_DIR", str(target))
        with pytest.raises(errors.CpError) as exc:
            paths.backup_dir()
        assert exc.value.code == "POLICY_REFUSED"
