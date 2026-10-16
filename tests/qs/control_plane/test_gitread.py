"""#407 T6: ``cp.py review read`` — git, read-only, fixed argv, hex shas only (D6a)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from control_plane import errors, gitread
from control_plane.runner import RunResult

from .conftest import run_cli

A, B = "a" * 40, "b" * 7
PREFIX = ["-c", "core.pager=cat", "-c", "diff.external=", "-c", "core.quotePath=false"]


def _git(main: Path) -> list[str]:
    return ["git", "-C", str(main), *PREFIX]


class TestArgv:
    @pytest.mark.parametrize(
        ("op", "given", "tail"),
        [
            ("diff", {"base": A, "head": B}, ["diff", "--no-ext-diff", "--no-textconv", "--no-color", A, B]),
            (
                "diff",
                {"base": A, "head": B, "path": "a//b/./c.py"},
                ["diff", "--no-ext-diff", "--no-textconv", "--no-color", A, B, "--", "a/b/c.py"],
            ),
            ("show", {"rev": A, "path": "x/y.py"}, ["cat-file", "blob", f"{A}:x/y.py"]),
            ("grep", {"rev": A, "pattern": "-O"}, ["grep", "-n", "--no-color", "-e", "-O", A]),
            ("grep", {"rev": A, "pattern": "x", "path": "d"}, ["grep", "-n", "--no-color", "-e", "x", A, "--", "d"]),
            ("ls-tree", {"rev": A}, ["ls-tree", "-r", "--name-only", A]),
            ("ls-tree", {"rev": A, "path": "-p"}, ["ls-tree", "-r", "--name-only", A, "--", "-p"]),
            ("ls-tree", {"rev": "f" * 64}, ["ls-tree", "-r", "--name-only", "f" * 64]),  # a SHA-256 object name
            (
                "files",
                {"base": A, "head": B},
                ["diff", "--no-ext-diff", "--no-textconv", "--no-color", "--name-status", A, B],
            ),
            (  # a truncated `files` answer is narrowed by a pathspec
                "files",
                {"base": A, "head": B, "path": "d"},
                ["diff", "--no-ext-diff", "--no-textconv", "--no-color", "--name-status", A, B, "--", "d"],
            ),
        ],
    )
    def test_each_op_is_one_fixed_argv(self, fake_main: Path, op: str, given: dict[str, str], tail: list[str]) -> None:
        assert gitread.argv(fake_main, op, given) == [*_git(fake_main), *tail]

    @pytest.mark.parametrize(
        ("op", "given"),
        [
            ("log", {"rev": A}),
            ("diff", {"base": A}),  # missing --head
            ("diff", {"base": A, "head": B, "rev": A}),  # not taken
            ("show", {"rev": A}),  # --path required
            ("show", {"rev": A, "path": " ./ "}),  # a path naming nothing
            ("grep", {"rev": A}),
            ("ls-tree", {"rev": A, "pattern": "x"}),
            ("diff", {"base": "main", "head": B}),  # a ref, not a sha
            ("diff", {"base": A, "head": "HEAD~1"}),
            ("ls-tree", {"rev": "--output=/tmp/x"}),
            ("ls-tree", {"rev": "-O"}),
            ("ls-tree", {"rev": "abc"}),  # too short
            ("ls-tree", {"rev": "A" * 40}),  # upper case
            ("ls-tree", {"rev": "a" * 65}),
            ("files", {"base": A}),  # missing --head
            ("files", {"base": A, "head": B, "rev": A}),  # not taken
            ("files", {"base": "HEAD", "head": B}),
            ("ls-tree", {"rev": f"{A}\n"}),
            ("show", {"rev": A, "path": "../secret"}),
            ("show", {"rev": A, "path": "/etc/passwd"}),
            ("show", {"rev": A, "path": "a.py:12"}),
        ],
    )
    def test_refusals_are_usage(self, fake_main: Path, op: str, given: dict[str, str]) -> None:
        with pytest.raises(errors.CpError) as exc:
            gitread.argv(fake_main, op, given)
        assert exc.value.code == "USAGE"


class TestRead:
    def test_runs_with_the_fixed_environment_and_surrogateescape(self, fake_runner, fake_main: Path) -> None:
        fake_runner.on(["ls-tree"], RunResult(0, "a.py\nb.py\n", ""))
        assert gitread.read(fake_runner, fake_main, "ls-tree", {"rev": A}) == {
            "output": "a.py\nb.py\n",
            "truncated": False,
            "lossy": False,
        }
        [call] = fake_runner.calls
        assert call.cwd == str(fake_main) and call.timeout == gitread.READ_TIMEOUT_S
        assert call.env_extra == {"GIT_CONFIG_NOSYSTEM": "1", "GIT_PAGER": "cat"}
        assert set(gitread.ENV_REMOVED) <= set(call.env_remove) and call.decode_errors == "surrogateescape"
        assert set(gitread.ENV_REMOVED) == {
            "GIT_EXTERNAL_DIFF",
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_NAMESPACE",
            "GIT_COMMON_DIR",
        }

    def test_every_inherited_git_variable_is_removed(self, fake_runner, fake_main: Path, monkeypatch) -> None:
        monkeypatch.setenv("GIT_ICASE_PATHSPECS", "1")
        monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'core.quotepath=true'")
        monkeypatch.setenv("GIT_PAGER", "less")  # set by ENV: never removed
        fake_runner.on(["ls-tree"], RunResult(0, "", ""))
        gitread.read(fake_runner, fake_main, "ls-tree", {"rev": A})
        [call] = fake_runner.calls
        assert {"GIT_ICASE_PATHSPECS", "GIT_CONFIG_PARAMETERS"} <= set(call.env_remove)
        assert not set(gitread.ENV) & set(call.env_remove)
        assert gitread.env_removed({"GIT_X": "1", "GIT_PAGER": "x", "PATH": "/"}) == (*gitread.ENV_REMOVED, "GIT_X")

    def test_truncation_cuts_on_bytes(self, fake_runner, fake_main: Path, monkeypatch) -> None:
        monkeypatch.setattr(gitread, "READ_MAX_BYTES", 4)
        fake_runner.on(["cat-file"], RunResult(0, "ééé", ""))  # 6 bytes
        out = gitread.read(fake_runner, fake_main, "show", {"rev": A, "path": "f"})
        assert out == {"output": "éé", "truncated": True, "lossy": False}
        fake_runner.on(["cat-file"], RunResult(0, "aéé", ""))  # the cut splits a character: it backs off to it
        out = gitread.read(fake_runner, fake_main, "show", {"rev": A, "path": "f"})
        assert out == {"output": "aé", "truncated": True, "lossy": False}
        fake_runner.on(["cat-file"], RunResult(0, "ab€", ""))  # a 3-byte character split after 2 of its bytes
        assert gitread.read(fake_runner, fake_main, "show", {"rev": A, "path": "f"})["output"] == "ab"

    def test_lossy_only_over_the_bytes_kept(self, fake_runner, fake_main: Path, monkeypatch) -> None:
        monkeypatch.setattr(gitread, "READ_MAX_BYTES", 4)
        fake_runner.on(["cat-file"], RunResult(0, b"abcd\xff".decode("utf-8", "surrogateescape"), ""))
        out = gitread.read(fake_runner, fake_main, "show", {"rev": A, "path": "f"})
        assert out == {"output": "abcd", "truncated": True, "lossy": False}  # the bad byte is past the cut
        fake_runner.on(["cat-file"], RunResult(0, b"a\xffcde".decode("utf-8", "surrogateescape"), ""))
        out = gitread.read(fake_runner, fake_main, "show", {"rev": A, "path": "f"})
        assert out == {"output": "a�cd", "truncated": True, "lossy": True}

    def test_lossy_on_a_byte_that_is_not_utf8(self, fake_runner, fake_main: Path) -> None:
        fake_runner.on(["cat-file"], RunResult(0, b"a\xffb".decode("utf-8", "surrogateescape"), ""))
        assert gitread.read(fake_runner, fake_main, "show", {"rev": A, "path": "f"}) == {
            "output": "a�b",
            "truncated": False,
            "lossy": True,
        }
        fake_runner.on(["cat-file"], RunResult(0, "a�b", ""))  # a real U+FFFD in the file
        assert gitread.read(fake_runner, fake_main, "show", {"rev": A, "path": "f"})["lossy"] is False

    def test_grep_without_a_match_is_empty(self, fake_runner, fake_main: Path) -> None:
        fake_runner.on(["grep"], RunResult(1, "", ""))
        assert gitread.read(fake_runner, fake_main, "grep", {"rev": A, "pattern": "x"})["output"] == ""

    @pytest.mark.parametrize(("code", "error"), [(128, "NOT_FOUND"), (1, "INTERNAL"), (124, "INTERNAL")])
    def test_a_failed_git(self, fake_runner, fake_main: Path, code: int, error: str) -> None:
        fake_runner.on(["diff"], RunResult(code, "", "fatal: bad object \udcff"))
        with pytest.raises(errors.CpError) as exc:
            gitread.read(fake_runner, fake_main, "diff", {"base": A, "head": B})
        assert exc.value.code == error and f"git diff exited {code}" in exc.value.detail and "�" in exc.value.detail


class TestCommand:
    def test_through_the_cli(self, fake_runner, fake_main: Path) -> None:
        fake_runner.on(["cat-file"], RunResult(0, "print(1)\n", ""))
        code, out = run_cli("review", "read", "show", "--rev", A, "--path", "src/a.py")
        assert code == 0 and out == {"ok": True, "output": "print(1)\n", "truncated": False, "lossy": False}
        assert fake_runner.calls[-1].argv == [*_git(fake_main), "cat-file", "blob", f"{A}:src/a.py"]

    def test_files_through_the_cli(self, fake_runner, fake_main: Path) -> None:
        fake_runner.on(["diff"], RunResult(0, "M\ta.py\n", ""))
        code, out = run_cli("review", "read", "files", "--base", A, "--head", B)
        assert code == 0 and out["output"] == "M\ta.py\n"
        assert fake_runner.calls[-1].argv[-3:] == ["--name-status", A, B]
        assert run_cli("review", "read", "files", "--base", A)[1]["error"] == "USAGE"

    def test_refusals_through_the_cli(self, fake_runner) -> None:
        assert run_cli("review", "read", "log", "--rev", A)[1]["error"] == "USAGE"  # not an op
        assert run_cli("review", "read", "diff", "--base", "HEAD", "--head", A)[1]["error"] == "USAGE"
        assert run_cli("review", "read", "ls-tree", "--rev", A, "--output", "x")[1]["error"] == "USAGE"
        assert fake_runner.calls == []

    def test_against_a_real_repository(self, tmp_path: Path, monkeypatch) -> None:
        """A real ``Runner`` and a real git: every op, and a binary file is ``lossy``."""
        from control_plane import paths, runner

        repo = tmp_path / "repo"
        repo.mkdir()

        def g(*args: str) -> str:
            return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()

        g("init", "-q", "-b", "main")
        (repo / "a.py").write_text("x = 1\n")
        g("add", ".")
        g("-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q", "-m", "one")
        base = g("rev-parse", "HEAD")
        (repo / "a.py").write_text("x = 2\n")
        (repo / "bin.dat").write_bytes(b"\xff\xfe\x00z")
        (repo / "d").mkdir()
        (repo / "d" / "é.py").write_text("y = 1\n")
        g("add", ".")
        g("-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q", "-m", "two")
        head = g("rev-parse", "HEAD")
        real = runner.Runner()
        assert "+x = 2" in gitread.read(real, repo, "diff", {"base": base, "head": head, "path": "a.py"})["output"]
        assert gitread.read(real, repo, "show", {"rev": head[:7], "path": "a.py"})["output"] == "x = 2\n"
        assert gitread.read(real, repo, "grep", {"rev": head, "pattern": "x ="})["output"] == f"{head}:a.py:1:x = 2\n"
        assert gitread.read(real, repo, "ls-tree", {"rev": head})["output"] == "a.py\nbin.dat\nd/é.py\n"
        files = gitread.read(real, repo, "files", {"base": base, "head": head})["output"]
        assert files == "M\ta.py\nA\tbin.dat\nA\td/é.py\n"  # unquoted: the name round-trips to --path
        assert "+y = 1" in gitread.read(real, repo, "diff", {"base": base, "head": head, "path": "d/é.py"})["output"]
        assert gitread.read(real, repo, "files", {"base": base, "head": head, "path": "d"})["output"] == "A\td/é.py\n"
        binary = gitread.read(real, repo, "show", {"rev": head, "path": "bin.dat"})
        assert binary["lossy"] is True and binary["output"].endswith("z")
        with pytest.raises(errors.CpError) as exc:
            gitread.read(real, repo, "show", {"rev": head, "path": "missing.py"})
        assert exc.value.code == "NOT_FOUND"
        assert paths  # the CLI form runs in paths.main(); here the repo is passed directly
