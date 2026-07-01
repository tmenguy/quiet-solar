"""Tests for quality_gate.py caching functionality."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts/qs to path so we can import quality_gate
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "qs"
sys.path.insert(0, str(SCRIPTS_DIR))

import quality_gate

QG_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "quality_gate"


# --- Helpers ---


def _make_all_pass_results() -> list[dict]:
    """Return gate results where everything passes."""
    return [
        {"name": "ruff_format", "passed": True, "detail": ""},
        {"name": "ruff_lint", "passed": True, "detail": ""},
        {"name": "mypy", "passed": True, "detail": ""},
        {"name": "translations", "passed": True, "detail": ""},
        {"name": "pytest", "passed": True, "coverage": "100%", "missing": [], "detail": "", "stderr": ""},
    ]


def _patch_git_state(branch: str = "QS_76", commit: str = "abc123", is_clean: bool = True):
    """Patch _get_git_state to return controlled values."""
    return patch.object(quality_gate, "_get_git_state", return_value=(branch, commit, is_clean))


def _patch_full_scope():
    """Force full scope so dev-only detection doesn't skip gates."""
    return patch.object(
        quality_gate,
        "_detect_scope",
        return_value={"scope": "full", "changed_test_files": [], "reason": "patched for test"},
    )


def _patch_all_gates(results: list[dict] | None = None):
    """Context manager that patches all five gate check functions."""
    r = results or _make_all_pass_results()

    class _Ctx:
        def __init__(self) -> None:
            self.patches = [
                patch.object(quality_gate, "check_ruff_format", return_value=r[0]),
                patch.object(quality_gate, "check_ruff_lint", return_value=r[1]),
                patch.object(quality_gate, "check_mypy", return_value=r[2]),
                patch.object(quality_gate, "check_translations", return_value=r[3]),
                patch.object(quality_gate, "check_pytest", return_value=r[4]),
            ]
            self.mocks: list = []

        def __enter__(self):
            self.mocks = [p.__enter__() for p in self.patches]
            return self.mocks

        def __exit__(self, *args):
            for p in self.patches:
                p.__exit__(*args)

    return _Ctx()


# --- Task 1: _get_git_state ---


class TestGetGitState:
    """Tests for _get_git_state."""

    def test_returns_branch_commit_clean(self) -> None:
        """_get_git_state returns a 3-tuple from git commands."""
        branch, commit, is_clean = quality_gate._get_git_state()
        # We're in a real git repo, so these should be non-empty
        assert isinstance(branch, str)
        assert len(branch) > 0
        assert isinstance(commit, str)
        assert len(commit) == 40  # full SHA
        assert isinstance(is_clean, bool)


# --- Task 1: cache read/write ---


class TestCacheReadWrite:
    """Tests for _read_cache and _write_cache."""

    def test_write_then_read_round_trip(self, tmp_path: Path) -> None:
        cache_path = tmp_path / ".quality_gate_cache"
        results = [{"name": "pytest", "passed": True}]

        with patch.object(quality_gate, "CACHE_FILE", cache_path):
            quality_gate._write_cache("QS_76", "abc123", results)
            data = quality_gate._read_cache()

        assert data is not None
        assert data["branch"] == "QS_76"
        assert data["commit"] == "abc123"
        assert data["all_passed"] is True
        assert data["results"] == results
        assert "timestamp" in data

    def test_read_returns_none_when_no_cache(self, tmp_path: Path) -> None:
        with patch.object(quality_gate, "CACHE_FILE", tmp_path / ".quality_gate_cache"):
            assert quality_gate._read_cache() is None

    def test_read_returns_none_on_corrupt_json(self, tmp_path: Path) -> None:
        cache_path = tmp_path / ".quality_gate_cache"
        cache_path.write_text("not json{{{")
        with patch.object(quality_gate, "CACHE_FILE", cache_path):
            assert quality_gate._read_cache() is None

    def test_read_returns_none_on_missing_keys(self, tmp_path: Path) -> None:
        cache_path = tmp_path / ".quality_gate_cache"
        cache_path.write_text('{"unexpected": true}')
        with patch.object(quality_gate, "CACHE_FILE", cache_path):
            assert quality_gate._read_cache() is None


# --- Task 1: cache validity ---


class TestIsCacheValid:
    """Tests for _is_cache_valid."""

    def test_valid_when_branch_commit_match_and_clean(self) -> None:
        cache = {"branch": "QS_76", "commit": "abc123", "results": []}
        assert quality_gate._is_cache_valid(cache, "QS_76", "abc123", is_clean=True) is True

    def test_invalid_when_branch_differs(self) -> None:
        cache = {"branch": "QS_76", "commit": "abc123", "results": []}
        assert quality_gate._is_cache_valid(cache, "QS_99", "abc123", is_clean=True) is False

    def test_invalid_when_commit_differs(self) -> None:
        cache = {"branch": "QS_76", "commit": "abc123", "results": []}
        assert quality_gate._is_cache_valid(cache, "QS_76", "def456", is_clean=True) is False

    def test_invalid_when_dirty_tree(self) -> None:
        cache = {"branch": "QS_76", "commit": "abc123", "results": []}
        assert quality_gate._is_cache_valid(cache, "QS_76", "abc123", is_clean=False) is False

    def test_invalid_when_cache_is_none(self) -> None:
        assert quality_gate._is_cache_valid(None, "QS_76", "abc123", is_clean=True) is False


# --- Task 2: CLI flags and main() integration ---


class TestCacheCliIntegration:
    """Tests for --cache/--no-cache flags and main() caching behavior."""

    def test_cache_hit_skips_gate_execution(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """AC 2: cache hit returns cached results without running gates."""
        cached_results = _make_all_pass_results()
        cache_path = tmp_path / ".quality_gate_cache"
        cache_path.write_text(
            json.dumps(
                {
                    "branch": "QS_76",
                    "commit": "abc123",
                    "all_passed": True,
                    "results": cached_results,
                    "timestamp": "",
                }
            )
        )

        with (
            patch("sys.argv", ["quality_gate.py", "--cache", "--json"]),
            _patch_git_state("QS_76", "abc123", True),
            patch.object(quality_gate, "CACHE_FILE", cache_path),
            _patch_all_gates() as mocks,
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()

        assert exc_info.value.code == 0
        for m in mocks:
            m.assert_not_called()

        output = json.loads(capsys.readouterr().out)
        assert output["cached"] is True
        assert output["all_passed"] is True

    def test_cache_miss_runs_gates_and_writes_cache(self, tmp_path: Path) -> None:
        """AC 1: on pass with --cache, writes cache file."""
        cache_path = tmp_path / ".quality_gate_cache"

        with (
            patch("sys.argv", ["quality_gate.py", "--cache", "--json"]),
            _patch_git_state("QS_76", "abc123", True),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", cache_path),
            _patch_all_gates(),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        assert cache_path.exists()
        data = json.loads(cache_path.read_text())
        assert data["branch"] == "QS_76"
        assert data["commit"] == "abc123"

    def test_cache_miss_when_commit_changed(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """AC 3: different commit invalidates cache."""
        cache_path = tmp_path / ".quality_gate_cache"
        cache_path.write_text(
            json.dumps(
                {
                    "branch": "QS_76",
                    "commit": "old_commit",
                    "all_passed": True,
                    "results": _make_all_pass_results(),
                    "timestamp": "",
                }
            )
        )

        with (
            patch("sys.argv", ["quality_gate.py", "--cache", "--json"]),
            _patch_git_state("QS_76", "new_commit", True),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", cache_path),
            _patch_all_gates() as mocks,
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        mocks[0].assert_called_once()  # gates ran
        output = json.loads(capsys.readouterr().out)
        assert output["cached"] is False

    def test_cache_miss_when_dirty_tree(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """AC 3: dirty working tree invalidates cache."""
        cache_path = tmp_path / ".quality_gate_cache"
        cache_path.write_text(
            json.dumps(
                {
                    "branch": "QS_76",
                    "commit": "abc123",
                    "all_passed": True,
                    "results": _make_all_pass_results(),
                    "timestamp": "",
                }
            )
        )

        with (
            patch("sys.argv", ["quality_gate.py", "--cache", "--json"]),
            _patch_git_state("QS_76", "abc123", False),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", cache_path),
            _patch_all_gates() as mocks,
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        mocks[0].assert_called_once()
        output = json.loads(capsys.readouterr().out)
        assert output["cached"] is False

    def test_fix_bypasses_cache(self, tmp_path: Path) -> None:
        """AC 4: --fix always runs fresh."""
        cache_path = tmp_path / ".quality_gate_cache"
        cache_path.write_text(
            json.dumps(
                {
                    "branch": "QS_76",
                    "commit": "abc123",
                    "all_passed": True,
                    "results": _make_all_pass_results(),
                    "timestamp": "",
                }
            )
        )

        with (
            patch("sys.argv", ["quality_gate.py", "--cache", "--fix", "--json"]),
            _patch_git_state("QS_76", "abc123", True),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", cache_path),
            _patch_all_gates() as mocks,
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        mocks[0].assert_called_once()

    def test_no_cache_forces_fresh_run(self, tmp_path: Path) -> None:
        """AC 5: --no-cache forces fresh run even with valid cache."""
        cache_path = tmp_path / ".quality_gate_cache"
        cache_path.write_text(
            json.dumps(
                {
                    "branch": "QS_76",
                    "commit": "abc123",
                    "all_passed": True,
                    "results": _make_all_pass_results(),
                    "timestamp": "",
                }
            )
        )

        with (
            patch("sys.argv", ["quality_gate.py", "--cache", "--no-cache", "--json"]),
            _patch_git_state("QS_76", "abc123", True),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", cache_path),
            _patch_all_gates() as mocks,
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        mocks[0].assert_called_once()

    def test_default_no_cache_flag_never_uses_cache(self, tmp_path: Path) -> None:
        """AC 7: without --cache, behavior identical to current (no caching)."""
        cache_path = tmp_path / ".quality_gate_cache"
        cache_path.write_text(
            json.dumps(
                {
                    "branch": "QS_76",
                    "commit": "abc123",
                    "all_passed": True,
                    "results": _make_all_pass_results(),
                    "timestamp": "",
                }
            )
        )

        with (
            patch("sys.argv", ["quality_gate.py", "--json"]),
            _patch_git_state("QS_76", "abc123", True),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", cache_path),
            _patch_all_gates() as mocks,
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        # Gates still run — --cache not passed
        mocks[0].assert_called_once()

    def test_cache_not_written_when_gates_fail(self, tmp_path: Path) -> None:
        """Cache should only be written when all gates pass."""
        cache_path = tmp_path / ".quality_gate_cache"
        failing = _make_all_pass_results()
        failing[0] = {**failing[0], "passed": False}

        with (
            patch("sys.argv", ["quality_gate.py", "--cache", "--json"]),
            _patch_git_state("QS_76", "abc123", True),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", cache_path),
            _patch_all_gates(failing),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        assert not cache_path.exists()

    def test_cache_not_written_when_dirty_after_gates(self, tmp_path: Path) -> None:
        """If tree becomes dirty during gate run (e.g. --fix), skip cache write."""
        cache_path = tmp_path / ".quality_gate_cache"
        call_count = 0

        def git_state_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("QS_76", "abc123", True)  # First call: clean (no cache file)
            return ("QS_76", "abc123", False)  # Second call: dirty after gates

        with (
            patch("sys.argv", ["quality_gate.py", "--cache", "--json"]),
            patch.object(quality_gate, "_get_git_state", side_effect=git_state_side_effect),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", cache_path),
            _patch_all_gates(),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        assert not cache_path.exists()

    def test_cache_hit_human_readable_output(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Cache hit in human-readable mode shows cached indicator."""
        cache_path = tmp_path / ".quality_gate_cache"
        cache_path.write_text(
            json.dumps(
                {
                    "branch": "QS_76",
                    "commit": "abc123",
                    "all_passed": True,
                    "results": _make_all_pass_results(),
                    "timestamp": "",
                }
            )
        )

        with (
            patch("sys.argv", ["quality_gate.py", "--cache"]),
            _patch_git_state("QS_76", "abc123", True),
            patch.object(quality_gate, "CACHE_FILE", cache_path),
            _patch_all_gates() as mocks,
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()

        assert exc_info.value.code == 0
        for m in mocks:
            m.assert_not_called()
        output = capsys.readouterr().out
        assert "cached" in output.lower()

    def test_dev_only_scope_skips_lint_gates_and_runs_pytest_only(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Dev-only scope skips ruff/mypy/translations, runs only pytest on changed files."""
        cache_path = tmp_path / ".quality_gate_cache"
        dev_only_scope = {
            "scope": "dev-only",
            "changed_test_files": ["tests/test_example.py"],
            "reason": "only dev/test files changed (1 files)",
        }
        pytest_result = {"name": "pytest", "passed": True, "detail": ""}

        with (
            patch("sys.argv", ["quality_gate.py", "--json"]),
            _patch_git_state("QS_76", "abc123", True),
            patch.object(quality_gate, "_detect_scope", return_value=dev_only_scope),
            patch.object(quality_gate, "CACHE_FILE", cache_path),
            patch.object(quality_gate, "check_pytest_files", return_value=pytest_result) as mock_pytest_files,
            _patch_all_gates() as mocks,
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()

        assert exc_info.value.code == 0
        # mocks order: [ruff_format, ruff_lint, mypy, translations, pytest]
        # Lint gates should NOT have been called
        for m in mocks[:4]:  # ruff_format, ruff_lint, mypy, translations
            m.assert_not_called()
        # Full pytest should NOT have been called either
        mocks[4].assert_not_called()
        # Only check_pytest_files should have been called
        mock_pytest_files.assert_called_once_with(["tests/test_example.py"])
        output = json.loads(capsys.readouterr().out)
        assert output["scope"] == "dev-only"
        assert output["all_passed"] is True


# --- T1: pytest invocation with xdist + sysmon + suppress html ---


class TestCheckPytestInvocation:
    """Tests for the pytest gate command construction (AC1, AC2, AC4)."""

    def test_default_uses_n_auto_and_sysmon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """check_pytest builds cmd with `-n auto` and runs with COVERAGE_CORE=sysmon."""
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        captured: dict = {}

        def fake_stream(cmd: list[str]) -> dict:
            captured["cmd"] = cmd
            captured["env"] = quality_gate._pytest_env()
            return {"name": "pytest", "passed": True, "coverage": "100%", "missing": [], "detail": "", "stderr": ""}

        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate, "_stream_pytest", side_effect=fake_stream),
        ):
            quality_gate.check_pytest()

        cmd = captured["cmd"]
        assert "-n" in cmd
        n_index = cmd.index("-n")
        assert cmd[n_index + 1] == "auto"
        assert captured["env"]["COVERAGE_CORE"] == "sysmon"

    def test_env_override_changes_workers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """QS_QG_PYTEST_WORKERS=4 → cmd contains `-n 4`."""
        monkeypatch.setenv("QS_QG_PYTEST_WORKERS", "4")
        captured: dict = {}

        def fake_stream(cmd: list[str]) -> dict:
            captured["cmd"] = cmd
            return {"name": "pytest", "passed": True, "coverage": "100%", "missing": [], "detail": "", "stderr": ""}

        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate, "_stream_pytest", side_effect=fake_stream),
        ):
            quality_gate.check_pytest()

        cmd = captured["cmd"]
        assert "-n" in cmd
        assert cmd[cmd.index("-n") + 1] == "4"

    def test_zero_workers_means_serial(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """QS_QG_PYTEST_WORKERS=0 → no `-n` in cmd."""
        monkeypatch.setenv("QS_QG_PYTEST_WORKERS", "0")
        captured: dict = {}

        def fake_stream(cmd: list[str]) -> dict:
            captured["cmd"] = cmd
            return {"name": "pytest", "passed": True, "coverage": "100%", "missing": [], "detail": "", "stderr": ""}

        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate, "_stream_pytest", side_effect=fake_stream),
        ):
            quality_gate.check_pytest()

        assert "-n" not in captured["cmd"]

    def test_missing_xdist_falls_back_serial(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When xdist is not importable, gate runs serially and warns to stderr."""
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        captured: dict = {}

        def fake_stream(cmd: list[str]) -> dict:
            captured["cmd"] = cmd
            return {"name": "pytest", "passed": True, "coverage": "100%", "missing": [], "detail": "", "stderr": ""}

        with (
            patch.object(quality_gate, "_has_xdist", return_value=False),
            patch.object(quality_gate, "_stream_pytest", side_effect=fake_stream),
        ):
            quality_gate.check_pytest()

        assert "-n" not in captured["cmd"]
        err = capsys.readouterr().err
        assert "xdist not available" in err

    def test_cov_report_empty_appended(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cmd contains `--cov-report=` (empty value) to override pytest.ini's html default."""
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        captured: dict = {}

        def fake_stream(cmd: list[str]) -> dict:
            captured["cmd"] = cmd
            return {"name": "pytest", "passed": True, "coverage": "100%", "missing": [], "detail": "", "stderr": ""}

        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate, "_stream_pytest", side_effect=fake_stream),
        ):
            quality_gate.check_pytest()

        assert "--cov-report=" in captured["cmd"]

    def test_collect_only_subprocess_has_no_n_and_no_sysmon(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The --collect-only count subprocess must not get `-n` or COVERAGE_CORE=sysmon.

        Both the count subprocess and the main pytest run go through subprocess.Popen
        so we can capture each call's env kwarg and verify the two-subprocess invariant
        (count subprocess inherits parent env, main subprocess adds COVERAGE_CORE).
        """
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        popen_calls: list = []

        class FakePopen:
            def __init__(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
                popen_calls.append({"cmd": list(cmd), "env": kwargs.get("env")})
                self.stdout = io.StringIO("0 tests collected\n")
                self.stderr = io.StringIO("")
                self.returncode = 0

            def communicate(self, *a, **kw):  # type: ignore[no-untyped-def]
                return ("0 tests collected\n", "")

            def wait(self):  # type: ignore[no-untyped-def]
                return 0

        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate.subprocess, "Popen", FakePopen),
        ):
            quality_gate.check_pytest()

        assert len(popen_calls) == 2, f"expected 2 Popen calls, got {len(popen_calls)}"

        # First Popen is the --collect-only count subprocess.
        collect = popen_calls[0]
        assert "--collect-only" in collect["cmd"]
        assert "-n" not in collect["cmd"]
        # S8: collect subprocess must NOT inherit COVERAGE_CORE=sysmon.
        # Either env is None (uses parent env) or, if set, must not select sysmon.
        collect_env = collect["env"]
        if collect_env is not None:
            assert collect_env.get("COVERAGE_CORE") != "sysmon", (
                "collect-only subprocess unexpectedly got COVERAGE_CORE=sysmon"
            )

        # Second Popen is the main pytest run — must have sysmon and -n.
        main = popen_calls[1]
        assert "-n" in main["cmd"]
        main_env = main["env"]
        assert main_env is not None
        assert main_env.get("COVERAGE_CORE") == "sysmon"


# --- T2: concurrent cheap gates, pytest serialized last (AC3) ---


class TestConcurrentGates:
    """Tests for parallel execution of cheap gates and serial pytest after."""

    def test_cheap_gates_run_concurrently(self, tmp_path: Path) -> None:
        """Cheap gates run in parallel — verified deterministically via a Barrier.

        Each mocked cheap gate calls `barrier.wait(timeout=2.0)`. If any gate
        ran serially, the barrier times out and the test fails deterministically;
        if all four ran in parallel, the barrier releases all of them at once.
        """
        barrier = threading.Barrier(4)

        def make_gate(name: str):  # type: ignore[no-untyped-def]
            def _fn(**kwargs):  # type: ignore[no-untyped-def]
                try:
                    barrier.wait(timeout=2.0)
                except threading.BrokenBarrierError:
                    pytest.fail(f"gate {name} did not run concurrently with the others")
                return {"name": name, "passed": True, "detail": ""}

            return _fn

        pytest_result = {
            "name": "pytest",
            "passed": True,
            "coverage": "100%",
            "missing": [],
            "detail": "",
            "stderr": "",
        }

        start = time.monotonic()
        with (
            patch("sys.argv", ["quality_gate.py", "--json"]),
            _patch_git_state(),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", tmp_path / ".quality_gate_cache"),
            patch.object(quality_gate, "check_ruff_format", side_effect=make_gate("ruff_format")),
            patch.object(quality_gate, "check_ruff_lint", side_effect=make_gate("ruff_lint")),
            patch.object(quality_gate, "check_mypy", side_effect=make_gate("mypy")),
            patch.object(quality_gate, "check_translations", side_effect=make_gate("translations")),
            patch.object(quality_gate, "check_pytest", return_value=pytest_result),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()
        elapsed = time.monotonic() - start

        # Sanity ceiling — the Barrier is the primary signal, this just guards
        # against truly absurd wall-clock blow-ups (e.g. silent hang on a slow CI).
        assert elapsed < 5.0, f"expected concurrent execution, took {elapsed:.2f}s"

    def test_pytest_runs_after_cheap_gates(self, tmp_path: Path) -> None:
        """check_pytest is called only after all 4 cheap gates have completed."""
        order: list[str] = []
        cheap_started: list[str] = []

        def make_cheap(name: str):  # type: ignore[no-untyped-def]
            def _fn(**kwargs):  # type: ignore[no-untyped-def]
                cheap_started.append(name)
                # Sleep briefly so all four are in-flight if parallel
                time.sleep(0.05)
                order.append(name)
                return {"name": name, "passed": True, "detail": ""}

            return _fn

        def fake_pytest():  # type: ignore[no-untyped-def]
            order.append("pytest")
            return {"name": "pytest", "passed": True, "coverage": "100%", "missing": [], "detail": "", "stderr": ""}

        with (
            patch("sys.argv", ["quality_gate.py", "--json"]),
            _patch_git_state(),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", tmp_path / ".quality_gate_cache"),
            patch.object(quality_gate, "check_ruff_format", side_effect=make_cheap("ruff_format")),
            patch.object(quality_gate, "check_ruff_lint", side_effect=make_cheap("ruff_lint")),
            patch.object(quality_gate, "check_mypy", side_effect=make_cheap("mypy")),
            patch.object(quality_gate, "check_translations", side_effect=make_cheap("translations")),
            patch.object(quality_gate, "check_pytest", side_effect=fake_pytest),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        # All 4 cheap gates should be in order before pytest
        cheap_names = {"ruff_format", "ruff_lint", "mypy", "translations"}
        pytest_index = order.index("pytest")
        cheap_completion_indexes = [order.index(name) for name in cheap_names]
        assert all(i < pytest_index for i in cheap_completion_indexes), (
            f"pytest must run after all cheap gates; order={order}"
        )

    def test_results_preserve_canonical_order(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """JSON output gates appear in canonical order regardless of completion order."""

        def make(name: str, delay: float):  # type: ignore[no-untyped-def]
            def _fn(**kwargs):  # type: ignore[no-untyped-def]
                time.sleep(delay)
                return {"name": name, "passed": True, "detail": ""}

            return _fn

        # Mypy finishes first, ruff_format last — should still be reported in canonical order
        with (
            patch("sys.argv", ["quality_gate.py", "--json"]),
            _patch_git_state(),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", tmp_path / ".quality_gate_cache"),
            patch.object(quality_gate, "check_ruff_format", side_effect=make("ruff_format", 0.15)),
            patch.object(quality_gate, "check_ruff_lint", side_effect=make("ruff_lint", 0.10)),
            patch.object(quality_gate, "check_mypy", side_effect=make("mypy", 0.01)),
            patch.object(quality_gate, "check_translations", side_effect=make("translations", 0.05)),
            patch.object(
                quality_gate,
                "check_pytest",
                return_value={
                    "name": "pytest",
                    "passed": True,
                    "coverage": "100%",
                    "missing": [],
                    "detail": "",
                    "stderr": "",
                },
            ),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        out = json.loads(capsys.readouterr().out)
        names = [g["name"] for g in out["gates"]]
        assert names == ["ruff_format", "ruff_lint", "mypy", "translations", "pytest"]

    def test_emit_writes_to_stderr_with_prefix(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """_emit(name, line) writes `[<name>] <line>\\n` to stderr."""
        quality_gate._emit("mypy", "running")
        err = capsys.readouterr().err
        assert err == "[mypy] running\n"

    def test_existing_gates_have_no_self_prefix(self) -> None:
        """Cheap gates no longer write the legacy `  <gate>:` self-prefix to stderr."""
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = ""
        fake_result.stderr = ""

        with (
            patch.object(quality_gate, "_run", return_value=fake_result),
            patch("sys.stderr", new_callable=io.StringIO) as fake_err,
        ):
            quality_gate.check_ruff_lint(fix=False)

        captured = fake_err.getvalue()
        # Old self-prefix was "  ruff lint: running..." with two-space indent
        assert "  ruff lint:" not in captured
        # New format uses [ruff_lint] prefix
        assert "[ruff_lint]" in captured


# --- T3: output-mode-agnostic progress parser (AC5) ---


class TestStreamPytestParser:
    """Tests for _parse_pytest_output against captured fixtures."""

    def test_parse_seq_q(self) -> None:
        text = (QG_FIXTURES_DIR / "seq_q_pass.txt").read_text()
        counts = quality_gate._parse_pytest_output(text)
        assert counts["passed"] == 21
        assert counts["failed"] == 0
        assert counts["errors"] == 0

    def test_parse_seq_q_cov(self) -> None:
        text = (QG_FIXTURES_DIR / "seq_q_cov_pass.txt").read_text()
        counts = quality_gate._parse_pytest_output(text)
        assert counts["passed"] == 21
        assert counts["failed"] == 0
        assert counts["errors"] == 0

    def test_parse_xdist_q(self) -> None:
        text = (QG_FIXTURES_DIR / "xdist_q_pass.txt").read_text()
        counts = quality_gate._parse_pytest_output(text)
        assert counts["passed"] == 21
        assert counts["failed"] == 0
        assert counts["errors"] == 0

    def test_parse_with_failures(self) -> None:
        text = (QG_FIXTURES_DIR / "with_failures.txt").read_text()
        counts = quality_gate._parse_pytest_output(text)
        assert counts["failed"] == 3
        # Summary line is authoritative: "3 failed, 18 passed"
        assert counts["passed"] == 18

    def test_parse_with_xdist_worker_prefix(self) -> None:
        """Lines like `[gw0] ...` are stripped before progress parsing."""
        text = "[gw0] .....\n[gw1] .....\n10 passed in 0.50s\n"
        counts = quality_gate._parse_pytest_output(text)
        assert counts["passed"] == 10


# --- T4: pytest.ini regression guard ---


class TestPytestIniRegression:
    """Regression guard for pytest.ini's --cov-report config."""

    def test_pytest_ini_has_no_html_default(self) -> None:
        """The pytest.ini default addopts must not include --cov-report=html."""
        repo_root = Path(__file__).resolve().parent.parent
        pytest_ini = (repo_root / "pytest.ini").read_text()
        assert "--cov-report=html" not in pytest_ini


# --- T9 regression: CI workflow yaml is valid and uses xdist + sysmon ---


class TestCiWorkflowConfig:
    """Regression guard for .github/workflows/pr-quality.yml."""

    def test_workflow_yaml_is_valid_and_uses_xdist_sysmon(self) -> None:
        """Workflow parses as YAML, test job uses -n auto and COVERAGE_CORE=sysmon."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        repo_root = Path(__file__).resolve().parent.parent
        wf_path = repo_root / ".github" / "workflows" / "pr-quality.yml"
        if not wf_path.exists():
            pytest.skip("workflow file missing")

        data = yaml.safe_load(wf_path.read_text())
        # Find the `test` job
        test_job = data["jobs"]["test"]
        steps = test_job["steps"]
        run_steps = [s for s in steps if "run" in s and "pytest" in s.get("run", "")]
        assert run_steps, "no pytest step found in test job"
        pytest_step = run_steps[0]
        assert "-n auto" in pytest_step["run"] or "--numprocesses auto" in pytest_step["run"]
        env = pytest_step.get("env", {})
        assert env.get("COVERAGE_CORE") == "sysmon"


# --- B1: requirements_test.txt includes pytest-xdist ---


class TestRequirementsTestDeps:
    """Regression guard for requirements_test.txt."""

    def test_pytest_xdist_declared(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        reqs = (repo_root / "requirements_test.txt").read_text()
        assert "pytest-xdist" in reqs


# --- Review-fix #01 M1: --fix serializes ruff_format and ruff_lint ---


class TestFixModeSerializesRuffGates:
    """Tests for M1: under --fix, ruff_format and ruff_lint cannot run concurrently
    because both write the same files; serialize them to avoid the race.
    """

    def test_fix_mode_serializes_ruff_gates(self, tmp_path: Path) -> None:
        """With --fix, ruff_format and ruff_lint windows do NOT overlap."""
        timestamps: dict[str, dict[str, float]] = {}
        lock = threading.Lock()

        def make_recorded(name: str):  # type: ignore[no-untyped-def]
            def _fn(**kwargs):  # type: ignore[no-untyped-def]
                with lock:
                    timestamps.setdefault(name, {})["start"] = time.monotonic()
                # Long enough to ensure overlap would be observable in parallel.
                time.sleep(0.1)
                with lock:
                    timestamps[name]["finish"] = time.monotonic()
                return {"name": name, "passed": True, "detail": ""}

            return _fn

        with (
            patch("sys.argv", ["quality_gate.py", "--fix", "--json"]),
            _patch_git_state(),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", tmp_path / ".quality_gate_cache"),
            patch.object(quality_gate, "check_ruff_format", side_effect=make_recorded("ruff_format")),
            patch.object(quality_gate, "check_ruff_lint", side_effect=make_recorded("ruff_lint")),
            patch.object(quality_gate, "check_mypy", side_effect=make_recorded("mypy")),
            patch.object(quality_gate, "check_translations", side_effect=make_recorded("translations")),
            patch.object(
                quality_gate,
                "check_pytest",
                return_value={
                    "name": "pytest",
                    "passed": True,
                    "coverage": "100%",
                    "missing": [],
                    "detail": "",
                    "stderr": "",
                },
            ),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        rf = timestamps["ruff_format"]
        rl = timestamps["ruff_lint"]
        # One must complete entirely before the other starts.
        assert rf["finish"] <= rl["start"] or rl["finish"] <= rf["start"], (
            f"ruff gates overlapped under --fix: format={rf}, lint={rl}"
        )

    def test_no_fix_mode_keeps_ruff_gates_parallel(self, tmp_path: Path) -> None:
        """Without --fix, ruff_format and ruff_lint windows DO overlap (concurrency preserved)."""
        timestamps: dict[str, dict[str, float]] = {}
        lock = threading.Lock()
        # Hold both ruff gates simultaneously to force observable overlap.
        # If they ran serially we'd time out here.
        rendezvous = threading.Barrier(2, timeout=2.0)

        def make_ruff(name: str):  # type: ignore[no-untyped-def]
            def _fn(**kwargs):  # type: ignore[no-untyped-def]
                with lock:
                    timestamps.setdefault(name, {})["start"] = time.monotonic()
                try:
                    rendezvous.wait()
                except threading.BrokenBarrierError:
                    pytest.fail(f"ruff gate {name} did not run concurrently with the other")
                with lock:
                    timestamps[name]["finish"] = time.monotonic()
                return {"name": name, "passed": True, "detail": ""}

            return _fn

        def make_quick(name: str):  # type: ignore[no-untyped-def]
            def _fn(**kwargs):  # type: ignore[no-untyped-def]
                return {"name": name, "passed": True, "detail": ""}

            return _fn

        with (
            patch("sys.argv", ["quality_gate.py", "--json"]),
            _patch_git_state(),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", tmp_path / ".quality_gate_cache"),
            patch.object(quality_gate, "check_ruff_format", side_effect=make_ruff("ruff_format")),
            patch.object(quality_gate, "check_ruff_lint", side_effect=make_ruff("ruff_lint")),
            patch.object(quality_gate, "check_mypy", side_effect=make_quick("mypy")),
            patch.object(quality_gate, "check_translations", side_effect=make_quick("translations")),
            patch.object(
                quality_gate,
                "check_pytest",
                return_value={
                    "name": "pytest",
                    "passed": True,
                    "coverage": "100%",
                    "missing": [],
                    "detail": "",
                    "stderr": "",
                },
            ),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        rf = timestamps["ruff_format"]
        rl = timestamps["ruff_lint"]
        # Windows overlap: each starts before the other finishes.
        assert rf["start"] < rl["finish"] and rl["start"] < rf["finish"], (
            f"ruff gates did not run concurrently without --fix: format={rf}, lint={rl}"
        )


# --- Review-fix #01 M2: _has_xdist probes VENV_PYTHON, not orchestrator ---


class TestHasXdistProbe:
    """Tests for M2: _has_xdist must probe the venv interpreter, not the
    orchestrator process (the two can be different Pythons).
    """

    def test_has_xdist_probes_venv_python(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_has_xdist invokes VENV_PYTHON via subprocess to check for xdist."""
        monkeypatch.setattr(quality_gate, "_HAS_XDIST_CACHE", None)
        captured_cmds: list[list[str]] = []

        def fake_run(cmd, cwd=None):  # type: ignore[no-untyped-def]
            captured_cmds.append(list(cmd))
            r = MagicMock()
            r.returncode = 0
            return r

        with patch.object(quality_gate, "_run", side_effect=fake_run):
            quality_gate._has_xdist()

        assert captured_cmds, "expected at least one _run call from _has_xdist"
        cmd = captured_cmds[0]
        assert cmd[0] == quality_gate.VENV_PYTHON
        assert cmd[1] == "-c"
        # The probe body uses find_spec to check for xdist
        assert "find_spec" in cmd[2]
        assert "xdist" in cmd[2]

    def test_has_xdist_true_when_venv_returns_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(quality_gate, "_HAS_XDIST_CACHE", None)
        r = MagicMock()
        r.returncode = 0
        with patch.object(quality_gate, "_run", return_value=r):
            assert quality_gate._has_xdist() is True

    def test_has_xdist_false_when_venv_returns_nonzero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(quality_gate, "_HAS_XDIST_CACHE", None)
        r = MagicMock()
        r.returncode = 1
        with patch.object(quality_gate, "_run", return_value=r):
            assert quality_gate._has_xdist() is False

    def test_has_xdist_caches_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repeated calls reuse the cached probe result — no second subprocess."""
        monkeypatch.setattr(quality_gate, "_HAS_XDIST_CACHE", None)
        call_count = 0

        def fake_run(cmd, cwd=None):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            r.returncode = 0
            return r

        with patch.object(quality_gate, "_run", side_effect=fake_run):
            quality_gate._has_xdist()
            quality_gate._has_xdist()
            quality_gate._has_xdist()

        assert call_count == 1, f"expected 1 probe call, got {call_count}"


# --- Review-fix #01 S1: --cov-report= empty must come BEFORE term-missing ---


class TestCovReportOrdering:
    """Tests for S1: empty --cov-report= must precede positive entries
    so it clears inherited reports without wiping the explicit ones we add.
    """

    def test_cov_report_empty_precedes_term_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        captured: dict = {}

        def fake_stream(cmd: list[str]) -> dict:
            captured["cmd"] = cmd
            return {"name": "pytest", "passed": True, "coverage": "100%", "missing": [], "detail": "", "stderr": ""}

        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate, "_stream_pytest", side_effect=fake_stream),
        ):
            quality_gate.check_pytest()

        cmd = captured["cmd"]
        empty_idx = cmd.index("--cov-report=")
        term_idx = cmd.index("--cov-report=term-missing")
        assert empty_idx < term_idx, (
            "empty --cov-report= must come BEFORE --cov-report=term-missing (otherwise it wipes term-missing)"
        )


# --- Review-fix #01 S2 + S3: parser handles xfailed/xpassed/skipped + anchoring ---


class TestParserExtendedCounts:
    """Tests for S2: _parse_pytest_output tracks skipped/xfailed/xpassed.
    Tests for S3: parser only treats lines containing "in <duration>s" as summaries.
    """

    def test_parse_with_skips_and_xfailed(self) -> None:
        """Fixture has 3 passed, 2 skipped, 1 xfailed in summary."""
        text = (QG_FIXTURES_DIR / "seq_q_with_skips.txt").read_text()
        counts = quality_gate._parse_pytest_output(text)
        assert counts["passed"] == 3
        assert counts["skipped"] == 2
        assert counts["xfailed"] == 1
        final_total = (
            counts["passed"]
            + counts["failed"]
            + counts["errors"]
            + counts["skipped"]
            + counts["xfailed"]
            + counts["xpassed"]
        )
        assert final_total == 6

    def test_parse_with_xpassed(self) -> None:
        """Synthetic summary with `xpassed` — must be tracked separately."""
        text = "....                                                                     [100%]\n4 passed, 2 xpassed in 0.10s\n"
        counts = quality_gate._parse_pytest_output(text)
        assert counts["passed"] == 4
        assert counts["xpassed"] == 2

    def test_parser_ignores_non_summary_lines_with_passed_word(self) -> None:
        """S3: only lines containing "in <duration>s" qualify as summaries.

        A noise line like "5 passed checks remaining" looks like a pytest summary
        on a superficial regex match but lacks the timing token. The authoritative
        summary line ("10 passed, 0 failed in 1.23s") wins.
        """
        text = "5 passed checks remaining\n10 passed, 0 failed in 1.23s\n"
        counts = quality_gate._parse_pytest_output(text)
        assert counts["passed"] == 10, f"parser misread non-summary line as summary; got passed={counts['passed']}"


# --- Review-fix #01 S4: _pytest_workers normalizes/validates env value ---


class TestPytestWorkersValidation:
    """Tests for S4: env value is normalized (strip), validated (positive int
    or "auto"), and falls back to "auto" with a warning on invalid input.
    """

    def test_workers_strips_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QS_QG_PYTEST_WORKERS", " 4 ")
        with patch.object(quality_gate, "_has_xdist", return_value=True):
            assert quality_gate._pytest_workers() == "4"

    def test_workers_invalid_value_warns_and_uses_auto(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("QS_QG_PYTEST_WORKERS", "four")
        with patch.object(quality_gate, "_has_xdist", return_value=True):
            result = quality_gate._pytest_workers()
        assert result == "auto"
        err = capsys.readouterr().err
        assert "invalid" in err.lower()
        assert "four" in err

    def test_workers_negative_value_falls_back_to_auto(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("QS_QG_PYTEST_WORKERS", "-1")
        with patch.object(quality_gate, "_has_xdist", return_value=True):
            result = quality_gate._pytest_workers()
        assert result == "auto"
        err = capsys.readouterr().err
        assert "invalid" in err.lower()
        assert "-1" in err

    def test_workers_auto_case_insensitive(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """'AUTO' (any case) → 'auto'."""
        monkeypatch.setenv("QS_QG_PYTEST_WORKERS", "AUTO")
        with patch.object(quality_gate, "_has_xdist", return_value=True):
            assert quality_gate._pytest_workers() == "auto"

    def test_workers_empty_string_means_serial(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty value (or all-whitespace) → None (serial)."""
        monkeypatch.setenv("QS_QG_PYTEST_WORKERS", "   ")
        with patch.object(quality_gate, "_has_xdist", return_value=True):
            assert quality_gate._pytest_workers() is None


# --- Review-fix #01 S5: Popen uses explicit UTF-8 with replace errors ---


class TestPopenUtf8Encoding:
    """Tests for S5: both Popen calls in _stream_pytest must explicitly use
    encoding='utf-8' and errors='replace' so decoding never crashes under
    LANG=C / LC_ALL=POSIX environments.
    """

    def test_stream_pytest_popen_uses_utf8_replace(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        popen_calls: list[dict] = []

        class FakePopen:
            def __init__(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
                popen_calls.append({"cmd": list(cmd), "kwargs": kwargs})
                self.stdout = io.StringIO("0 tests collected\n")
                self.stderr = io.StringIO("")
                self.returncode = 0

            def communicate(self, *a, **kw):  # type: ignore[no-untyped-def]
                return ("0 tests collected\n", "")

            def wait(self):  # type: ignore[no-untyped-def]
                return 0

        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate.subprocess, "Popen", FakePopen),
        ):
            quality_gate.check_pytest()

        assert len(popen_calls) == 2, f"expected 2 Popen calls, got {len(popen_calls)}"
        for i, call in enumerate(popen_calls):
            kwargs = call["kwargs"]
            assert kwargs.get("encoding") == "utf-8", f"Popen call {i} missing encoding='utf-8': kwargs={kwargs!r}"
            assert kwargs.get("errors") == "replace", f"Popen call {i} missing errors='replace': kwargs={kwargs!r}"


# --- Review-fix #01 S6: cheap-gate exception synthesizes failure, pipeline continues ---


class TestCheapGateExceptionHandling:
    """Tests for S6: if a cheap gate raises, main() synthesizes a failure
    result so the standard FAILED-gates path runs (no traceback escape).
    """

    def test_cheap_gate_exception_does_not_crash_main(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """RuntimeError from check_mypy → mypy reported as failed, pytest still runs."""
        pytest_result = {
            "name": "pytest",
            "passed": True,
            "coverage": "100%",
            "missing": [],
            "detail": "",
            "stderr": "",
        }

        with (
            patch("sys.argv", ["quality_gate.py", "--json"]),
            _patch_git_state(),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", tmp_path / ".quality_gate_cache"),
            patch.object(
                quality_gate,
                "check_ruff_format",
                return_value={
                    "name": "ruff_format",
                    "passed": True,
                    "detail": "",
                },
            ),
            patch.object(
                quality_gate,
                "check_ruff_lint",
                return_value={
                    "name": "ruff_lint",
                    "passed": True,
                    "detail": "",
                },
            ),
            patch.object(quality_gate, "check_mypy", side_effect=RuntimeError("boom")),
            patch.object(
                quality_gate,
                "check_translations",
                return_value={
                    "name": "translations",
                    "passed": True,
                    "detail": "",
                },
            ),
            patch.object(quality_gate, "check_pytest", return_value=pytest_result),
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()

        # mypy synthesized failure → exit 1
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        names_to_results = {g["name"]: g for g in output["gates"]}
        mypy_result = names_to_results["mypy"]
        assert mypy_result["passed"] is False
        combined = str(mypy_result.get("stderr", "")) + str(mypy_result.get("detail", ""))
        assert "boom" in combined, f"expected exception message in mypy result, got {mypy_result!r}"
        # pytest still ran (came after cheap gates)
        assert names_to_results["pytest"]["passed"] is True

    def test_cheap_gate_exception_under_fix_does_not_crash_main(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """S6 + M1: exception under --fix (where ruff gates run in a composite
        future) still synthesizes failure results without escaping."""
        pytest_result = {
            "name": "pytest",
            "passed": True,
            "coverage": "100%",
            "missing": [],
            "detail": "",
            "stderr": "",
        }

        with (
            patch("sys.argv", ["quality_gate.py", "--fix", "--json"]),
            _patch_git_state(),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", tmp_path / ".quality_gate_cache"),
            patch.object(quality_gate, "check_ruff_format", side_effect=RuntimeError("ruff_format boom")),
            patch.object(
                quality_gate,
                "check_ruff_lint",
                return_value={
                    "name": "ruff_lint",
                    "passed": True,
                    "detail": "",
                },
            ),
            patch.object(
                quality_gate,
                "check_mypy",
                return_value={
                    "name": "mypy",
                    "passed": True,
                    "detail": "",
                },
            ),
            patch.object(
                quality_gate,
                "check_translations",
                return_value={
                    "name": "translations",
                    "passed": True,
                    "detail": "",
                },
            ),
            patch.object(quality_gate, "check_pytest", return_value=pytest_result),
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()

        assert exc_info.value.code == 1
        output = json.loads(capsys.readouterr().out)
        names_to_results = {g["name"]: g for g in output["gates"]}
        # The composite ruff pair must report ruff_format as failed.
        assert names_to_results["ruff_format"]["passed"] is False
        assert "boom" in (
            str(names_to_results["ruff_format"].get("stderr", ""))
            + str(names_to_results["ruff_format"].get("detail", ""))
        )


# --- Review-fix #01 S7: explicit serial mode emits a distinct warning ---


class TestExplicitSerialWarning:
    """Tests for S7: when QS_QG_PYTEST_WORKERS=0 (or "") AND xdist is available,
    emit a "by request" warning so the user knows the override took effect.
    """

    def test_explicit_serial_emits_distinct_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("QS_QG_PYTEST_WORKERS", "0")
        captured_cmd: dict = {}

        def fake_stream(cmd: list[str]) -> dict:
            captured_cmd["cmd"] = cmd
            return {"name": "pytest", "passed": True, "coverage": "100%", "missing": [], "detail": "", "stderr": ""}

        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate, "_stream_pytest", side_effect=fake_stream),
        ):
            quality_gate.check_pytest()

        err = capsys.readouterr().err
        assert "by request" in err, f"expected 'by request' warning, got: {err!r}"
        # Must NOT be the xdist-missing warning, which has different wording.
        assert "not available" not in err, f"got xdist-missing warning instead of explicit-serial warning: {err!r}"
        # -n must not be in the cmd (serial mode confirmed)
        assert "-n" not in captured_cmd["cmd"]


# --- Review-fix #01 S10: real check_pytest run does not produce htmlcov ---


class TestHtmlcovNotWritten:
    """Tests for S10: AC4's "no htmlcov/ directory written" is verified end-to-end."""

    def test_qg_run_does_not_write_htmlcov(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A real check_pytest run against a tmp tree does not produce htmlcov/.

        Mimics the project's real `pytest.ini` (post-T4: no `--cov-report=html`
        in addopts) and a trivial 100%-covered package. End-to-end regression
        guard against a future re-introduction of `--cov-report=html` in
        addopts or a regression in the cmd-construction that re-enables it.

        Note: pytest-cov treats `--cov-report=` (empty) as "no-op" when other
        `--cov-report=*` entries exist (only clears when it's the sole entry).
        So the actual mechanism preventing htmlcov in the real project is T4
        (no `--cov-report=html` in pytest.ini addopts), not the cmd-level
        empty override — that's why this test mimics the real pytest.ini
        without html rather than testing the empty override's effect.
        """
        venv_python = Path(quality_gate.VENV_PYTHON)
        if not venv_python.exists():
            pytest.skip("venv python not available")

        # Force serial — xdist is not needed for a one-test sanity run.
        monkeypatch.setenv("QS_QG_PYTEST_WORKERS", "0")

        src = tmp_path / "src_pkg"
        src.mkdir()
        (src / "__init__.py").write_text("def hello():\n    return 1\n")

        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_trivial.py").write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(tmp_path)!r})\n"
            "from src_pkg import hello\n"
            "def test_one():\n    assert hello() == 1\n"
        )

        # Mimic the real (post-T4) pytest.ini's relevant addopts — no html.
        # `asyncio_mode = auto` matches the real pytest.ini so pytest-asyncio's
        # autouse async fixtures don't error out under this sub-pytest run.
        (tmp_path / "pytest.ini").write_text(
            "[pytest]\naddopts =\n    --strict-markers\n    -ra\n    --cov-report=term-missing\nasyncio_mode = auto\n"
        )

        monkeypatch.setattr(quality_gate, "TESTS_DIR", tests)
        monkeypatch.setattr(quality_gate, "SRC_DIR", src)
        monkeypatch.setattr(quality_gate, "REPO_ROOT", tmp_path)

        result = quality_gate.check_pytest()

        assert result["passed"] is True, (
            f"trivial test must pass; detail={result.get('detail')!r}, stderr={result.get('stderr')!r}"
        )
        assert not (tmp_path / "htmlcov").exists(), (
            "htmlcov/ was unexpectedly created — pytest.ini addopts or the QG"
            " cmd construction re-enabled the html report"
        )


# --- QS-183 T6: --quick fast-iteration mode ---


class TestQuickMode:
    """Tests for `--quick PATH [PATH ...]` (QS-183 Category B).

    `--quick` runs `pytest` on the cited paths with xdist + sysmon, and
    skips every other gate (ruff / mypy / translations / coverage / cache
    / scope detection). Mutually exclusive with `--cache`, `--no-cache`,
    `--full`, and `--fix`.
    """

    @pytest.mark.parametrize(
        "argv_paths",
        [
            ["tests/test_foo.py"],
            ["tests/test_foo.py", "tests/test_bar.py"],
            ["tests/ha_tests"],
            ["tests/test_foo.py", "tests/ha_tests"],
        ],
        ids=["single-file", "multi-file", "directory", "mixed-file-and-dir"],
    )
    def test_quick_invokes_check_pytest_files_with_paths(
        self,
        argv_paths: list[str],
    ) -> None:
        """`--quick` forwards positional paths to `check_pytest_files` unchanged."""
        pytest_result = {"name": "pytest", "passed": True, "detail": ""}

        with (
            patch(
                "sys.argv",
                ["quality_gate.py", "--quick", *argv_paths],
            ),
            patch.object(
                quality_gate,
                "check_pytest_files",
                return_value=pytest_result,
            ) as mock_pytest_files,
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()

        assert exc_info.value.code == 0
        mock_pytest_files.assert_called_once_with(argv_paths)

    def test_quick_skips_everything_else(self, tmp_path: Path) -> None:
        """`--quick` does not call any other gate, cache, or scope helper."""
        pytest_result = {"name": "pytest", "passed": True, "detail": ""}

        with (
            patch("sys.argv", ["quality_gate.py", "--quick", "tests/test_foo.py"]),
            patch.object(quality_gate, "CACHE_FILE", tmp_path / ".quality_gate_cache"),
            patch.object(quality_gate, "check_ruff_lint") as mock_ruff_lint,
            patch.object(quality_gate, "check_ruff_format") as mock_ruff_format,
            patch.object(quality_gate, "check_mypy") as mock_mypy,
            patch.object(quality_gate, "check_translations") as mock_trans,
            patch.object(quality_gate, "check_pytest") as mock_full_pytest,
            patch.object(quality_gate, "_get_git_state") as mock_git_state,
            patch.object(quality_gate, "_detect_scope") as mock_detect_scope,
            patch.object(quality_gate, "_read_cache") as mock_read_cache,
            patch.object(quality_gate, "_write_cache") as mock_write_cache,
            patch.object(
                quality_gate,
                "check_pytest_files",
                return_value=pytest_result,
            ),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        # None of the skipped helpers may have been called.
        for mock in (
            mock_ruff_lint,
            mock_ruff_format,
            mock_mypy,
            mock_trans,
            mock_full_pytest,
            mock_git_state,
            mock_detect_scope,
            mock_read_cache,
            mock_write_cache,
        ):
            mock.assert_not_called()

    @pytest.mark.parametrize(
        ("pytest_passed", "expected_exit"),
        [(True, 0), (False, 1)],
        ids=["pass→0", "fail→1"],
    )
    def test_quick_exit_code_propagates(
        self,
        pytest_passed: bool,
        expected_exit: int,
    ) -> None:
        """`--quick` exits 0 iff the underlying pytest passes, 1 otherwise."""
        result = {"name": "pytest", "passed": pytest_passed, "detail": ""}
        with (
            patch("sys.argv", ["quality_gate.py", "--quick", "tests/test_foo.py"]),
            patch.object(quality_gate, "check_pytest_files", return_value=result),
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()
        assert exc_info.value.code == expected_exit

    def test_quick_emits_banner(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`--quick` prints a `[quick] running ...` banner to stderr."""
        result = {"name": "pytest", "passed": True, "detail": ""}
        with (
            patch(
                "sys.argv",
                ["quality_gate.py", "--quick", "tests/test_foo.py", "tests/ha_tests"],
            ),
            patch.object(quality_gate, "check_pytest_files", return_value=result),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()
        err = capsys.readouterr().err
        assert err.startswith("[quick] running "), f"banner missing/wrong: {err!r}"
        assert "xdist + sysmon" in err, f"banner missing 'xdist + sysmon': {err!r}"

    def test_quick_rejects_empty_args(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`--quick` with no paths fails at argparse layer (exit 2)."""
        with (
            patch("sys.argv", ["quality_gate.py", "--quick"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "--quick" in err, f"argparse error must name --quick: {err!r}"

    @pytest.mark.parametrize(
        "conflict_flag",
        ["--cache", "--no-cache", "--full", "--fix"],
    )
    def test_quick_mutex_matrix(
        self,
        conflict_flag: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`--quick` combined with any of --cache/--no-cache/--full/--fix → exit 2."""
        with (
            patch(
                "sys.argv",
                [
                    "quality_gate.py",
                    "--quick",
                    "tests/test_x.py",
                    conflict_flag,
                ],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "you cannot combine --quick with --cache, --no-cache, --full, or --fix" in err, (
            f"mutex message missing/changed: {err!r}"
        )

    @pytest.mark.parametrize(
        ("workers_value", "expected_in_cmd"),
        [("auto", True), ("4", True), (None, False)],
        ids=["auto", "fixed-count", "serial"],
    )
    def test_check_pytest_files_uses_workers_when_resolver_returns_value(
        self,
        workers_value: str | None,
        expected_in_cmd: bool,
    ) -> None:
        """`check_pytest_files` adds `-n <workers>` iff `_pytest_workers()` returns one."""
        captured: dict = {}

        def fake_stream(
            cmd: list[str],
            collect_targets: list[str] | None = None,
        ) -> dict:
            captured["cmd"] = cmd
            captured["collect_targets"] = collect_targets
            return {"name": "pytest", "passed": True, "detail": ""}

        with (
            patch.object(quality_gate, "_pytest_workers", return_value=workers_value),
            patch.object(quality_gate, "_stream_pytest", side_effect=fake_stream),
        ):
            quality_gate.check_pytest_files(["tests/test_x.py"])

        cmd = captured["cmd"]
        assert ("-n" in cmd) is expected_in_cmd, (
            f"-n presence ({'-n' in cmd}) != expected ({expected_in_cmd}); cmd={cmd!r}"
        )
        if expected_in_cmd:
            n_idx = cmd.index("-n")
            assert cmd[n_idx + 1] == workers_value, f"-n value mismatch; want {workers_value!r}, got {cmd[n_idx + 1]!r}"

    def test_quick_collect_only_uses_cited_paths_not_tests_dir(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Review-fix #01 finding 2: `--quick` count subprocess must collect only
        the cited paths, not walk the entire `tests/` tree.

        `_stream_pytest`'s upfront `pytest --collect-only` call has historically
        been hardcoded to `TESTS_DIR`, costing 1–3s cold even when the caller
        only wants a single file. Wire `collect_targets` end-to-end from
        `check_pytest_files` so the count subprocess receives the same paths
        as the main run.
        """
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        popen_calls: list[dict] = []

        class FakePopen:
            def __init__(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
                popen_calls.append({"cmd": list(cmd), "kwargs": kwargs})
                self.stdout = io.StringIO("0 tests collected\n")
                self.stderr = io.StringIO("")
                self.returncode = 0

            def communicate(self, *a, **kw):  # type: ignore[no-untyped-def]
                return ("0 tests collected\n", "")

            def wait(self):  # type: ignore[no-untyped-def]
                return 0

        with (
            patch.object(quality_gate, "_has_xdist", return_value=False),
            patch.object(quality_gate.subprocess, "Popen", FakePopen),
        ):
            quality_gate.check_pytest_files(["tests/test_factories_pytest_opt_out.py"])

        assert len(popen_calls) == 2, f"expected 2 Popen calls, got {len(popen_calls)}"
        collect_cmd = popen_calls[0]["cmd"]
        assert "--collect-only" in collect_cmd

        # The cited file (resolved against REPO_ROOT) must appear in the count
        # cmd; the full tests/ tree path must NOT.
        cited = str(quality_gate.REPO_ROOT / "tests/test_factories_pytest_opt_out.py")
        assert cited in collect_cmd, f"collect-only cmd must include the cited path {cited!r}; got {collect_cmd!r}"
        assert str(quality_gate.TESTS_DIR) not in collect_cmd, (
            f"collect-only cmd must NOT walk full TESTS_DIR; got {collect_cmd!r}"
        )

    def test_check_pytest_full_path_still_uses_tests_dir_for_collect(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Negative half of finding 2 — the full-gate `check_pytest()` caller
        must KEEP collecting against `TESTS_DIR` (its semantics are
        "whole-suite coverage", so the count subprocess walking everything
        is correct there).
        """
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        popen_calls: list[dict] = []

        class FakePopen:
            def __init__(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
                popen_calls.append({"cmd": list(cmd), "kwargs": kwargs})
                self.stdout = io.StringIO("0 tests collected\n")
                self.stderr = io.StringIO("")
                self.returncode = 0

            def communicate(self, *a, **kw):  # type: ignore[no-untyped-def]
                return ("0 tests collected\n", "")

            def wait(self):  # type: ignore[no-untyped-def]
                return 0

        with (
            patch.object(quality_gate, "_has_xdist", return_value=False),
            patch.object(quality_gate.subprocess, "Popen", FakePopen),
        ):
            quality_gate.check_pytest()

        assert len(popen_calls) == 2, f"expected 2 Popen calls, got {len(popen_calls)}"
        collect_cmd = popen_calls[0]["cmd"]
        assert "--collect-only" in collect_cmd
        assert str(quality_gate.TESTS_DIR) in collect_cmd, (
            f"full-gate collect-only must include TESTS_DIR; got {collect_cmd!r}"
        )

    @pytest.mark.parametrize(
        "bad_path",
        ["/etc/passwd", "../outside.py", "/tmp/foo.py"],
        ids=["absolute-system", "parent-escape", "absolute-tmp"],
    )
    def test_quick_rejects_paths_outside_repo(
        self,
        bad_path: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Review-fix #01 finding 7: `--quick` rejects paths that escape REPO_ROOT.

        `REPO_ROOT / "/etc/passwd"` silently discards REPO_ROOT (pathlib
        semantics) and `../foo` walks out of the tree. Both must error
        with exit 2 + a clear message.
        """
        with (
            patch("sys.argv", ["quality_gate.py", "--quick", bad_path]),
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "must be inside the repo" in err, f"path-escape message missing/changed: {err!r}"
        assert bad_path in err, f"offending path must appear in error: {err!r}"

    @pytest.mark.parametrize(
        "argv_tail",
        [[""], ["tests/test_foo.py", ""], ["", "tests/test_foo.py"]],
        ids=["only-empty", "trailing-empty", "leading-empty"],
    )
    def test_quick_rejects_empty_string_paths(
        self,
        argv_tail: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Review-fix #01 finding 8: `--quick ""` must NOT silently subvert
        the contract.

        argparse `nargs="+"` accepts an empty string as a positional, and
        `REPO_ROOT / ""` resolves back to REPO_ROOT, so pytest would walk
        the entire suite. Reject empty-string paths explicitly.
        """
        with (
            patch("sys.argv", ["quality_gate.py", "--quick", *argv_tail]),
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "must be non-empty" in err, f"empty-path message missing/changed: {err!r}"


# --- QS-208 T1.1: _is_ui_asset detector ---


class TestIsUIAsset:
    """Tests for the `_is_ui_asset` UI-asset classifier (AC-6)."""

    def test_recognizes_j2_template_anywhere_under_ui(self) -> None:
        """Both top-level and nested `.j2` files under `ui/` count as UI assets."""
        assert (
            quality_gate._is_ui_asset("custom_components/quiet_solar/ui/quiet_solar_dashboard_template.yaml.j2") is True
        )
        assert quality_gate._is_ui_asset("custom_components/quiet_solar/ui/subdir/partial.j2") is True

    def test_recognizes_any_file_under_resources(self) -> None:
        """Any file under `ui/resources/` is a UI asset, regardless of extension.

        Convention: nothing under `ui/resources/` is Python. Even a hypothetical
        `.py` file there is treated as a UI asset (and would be a category error
        — Python code belongs outside `resources/`).
        """
        assert quality_gate._is_ui_asset("custom_components/quiet_solar/ui/resources/qs-car-card.js") is True
        assert quality_gate._is_ui_asset("custom_components/quiet_solar/ui/resources/sub/nested.css") is True
        # Convention-documenting test: nothing should be .py here, but if it
        # is, it still routes through the UI fast path. Users should move it.
        assert quality_gate._is_ui_asset("custom_components/quiet_solar/ui/resources/hypothetical.py") is True

    def test_rejects_python_at_ui_root(self) -> None:
        """`.py` files directly under `ui/` are Python production code, not UI assets."""
        assert quality_gate._is_ui_asset("custom_components/quiet_solar/ui/dashboard.py") is False
        assert quality_gate._is_ui_asset("custom_components/quiet_solar/ui/__init__.py") is False

    def test_rejects_paths_outside_ui(self) -> None:
        """Files outside `custom_components/quiet_solar/ui/` are never UI assets."""
        assert quality_gate._is_ui_asset("custom_components/quiet_solar/home_model/foo.py") is False
        assert quality_gate._is_ui_asset("custom_components/quiet_solar/ha_model/bar.py") is False
        assert quality_gate._is_ui_asset("tests/test_baz.py") is False
        assert quality_gate._is_ui_asset("scripts/qs/quality_gate.py") is False


# --- QS-208 T1.2: _detect_scope returns "ui-only" ---


class TestDetectScopeUIOnly:
    """Tests for the new `"ui-only"` branch in `_detect_scope` (AC-1, AC-3, AC-5)."""

    def test_returns_ui_only_when_only_j2_changed(self) -> None:
        """Diff of one `.j2` template → scope is `"ui-only"`."""
        info = quality_gate._detect_scope(["custom_components/quiet_solar/ui/quiet_solar_dashboard_template.yaml.j2"])
        assert info["scope"] == "ui-only"
        assert info["changed_test_files"] == []

    def test_returns_ui_only_when_only_resources_changed(self) -> None:
        """Diff of multiple files under `ui/resources/` → scope is `"ui-only"`."""
        info = quality_gate._detect_scope(
            [
                "custom_components/quiet_solar/ui/resources/qs-car-card.js",
                "custom_components/quiet_solar/ui/resources/qs-water-boiler-card.js",
            ]
        )
        assert info["scope"] == "ui-only"

    def test_returns_full_when_dashboard_py_also_changed(self) -> None:
        """Diff containing `ui/dashboard.py` plus a `.j2` → scope is `"full"`.

        The Python module is production code under `quiet_solar/`; it doesn't
        match `_is_dev_only` or `_is_ui_asset` and must force the full gate.
        """
        info = quality_gate._detect_scope(
            [
                "custom_components/quiet_solar/ui/quiet_solar_dashboard_template.yaml.j2",
                "custom_components/quiet_solar/ui/dashboard.py",
            ]
        )
        assert info["scope"] == "full"
        assert "dashboard.py" in info["reason"]

    def test_returns_full_when_init_py_also_changed(self) -> None:
        """Diff containing `ui/__init__.py` plus a `.j2` → scope is `"full"`."""
        info = quality_gate._detect_scope(
            [
                "custom_components/quiet_solar/ui/quiet_solar_dashboard_template.yaml.j2",
                "custom_components/quiet_solar/ui/__init__.py",
            ]
        )
        assert info["scope"] == "full"
        assert "__init__.py" in info["reason"]

    def test_returns_ui_only_when_mixed_with_dev_only_and_dedupes(self) -> None:
        """UI assets mixed with dev-only paths still resolve to `"ui-only"`,
        and changed test files surface in `changed_test_files`.
        """
        info = quality_gate._detect_scope(
            [
                "custom_components/quiet_solar/ui/quiet_solar_dashboard_template.yaml.j2",
                "docs/stories/QS-208.story.md",
                "tests/test_dashboard_rendering.py",
            ]
        )
        assert info["scope"] == "ui-only"
        assert info["changed_test_files"] == ["tests/test_dashboard_rendering.py"]


# --- QS-208 T1.3 + T1.4: main() dispatches ui-only branch correctly ---


class TestUIOnlyMainBranch:
    """End-to-end tests for the ui-only branch in `main()` (AC-1, AC-2, AC-4, AC-5)."""

    def test_ui_only_scope_skips_lint_gates_and_runs_only_dashboard_rendering(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """ui-only scope: skips all lint gates + full pytest; runs only the
        canonical dashboard-rendering test; emits the UI-ONLY banner; JSON
        scope field is `"ui-only"`."""
        cache_path = tmp_path / ".quality_gate_cache"
        ui_only_scope = {
            "scope": "ui-only",
            "changed_test_files": [],
            "reason": "only UI assets and dev files changed (1 UI asset(s), 1 total)",
        }
        pytest_result = {"name": "pytest", "passed": True, "detail": ""}

        with (
            patch("sys.argv", ["quality_gate.py", "--json"]),
            _patch_git_state("QS_208", "abc123", True),
            patch.object(quality_gate, "_detect_scope", return_value=ui_only_scope),
            patch.object(quality_gate, "CACHE_FILE", cache_path),
            patch.object(quality_gate, "check_pytest_files", return_value=pytest_result) as mock_pytest_files,
            _patch_all_gates() as mocks,
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()

        assert exc_info.value.code == 0
        # mocks order: [ruff_format, ruff_lint, mypy, translations, pytest]
        for m in mocks[:4]:
            m.assert_not_called()
        # Full pytest must NOT run either.
        mocks[4].assert_not_called()
        # Only the UI fast-path pytest invocation runs, on the canonical file.
        mock_pytest_files.assert_called_once_with(["tests/test_dashboard_rendering.py"])

        captured = capsys.readouterr()
        # JSON output: scope is "ui-only".
        output = json.loads(captured.out)
        assert output["scope"] == "ui-only"
        assert output["all_passed"] is True
        # Stderr banner: exact text per AC-1.
        assert "scope: UI-ONLY" in captured.err
        assert "skipping ruff, mypy, translations, full coverage" in captured.err

    def test_ui_only_scope_dedupes_when_canonical_test_in_changed_set(
        self,
        tmp_path: Path,
    ) -> None:
        """When the canonical test file is itself in the diff, the merged
        list still contains it exactly once (set semantics)."""
        cache_path = tmp_path / ".quality_gate_cache"
        ui_only_scope = {
            "scope": "ui-only",
            "changed_test_files": [
                "tests/test_dashboard_rendering.py",
                "tests/test_other.py",
            ],
            "reason": "only UI assets and dev files changed",
        }
        pytest_result = {"name": "pytest", "passed": True, "detail": ""}

        with (
            patch("sys.argv", ["quality_gate.py", "--json"]),
            _patch_git_state("QS_208", "abc123", True),
            patch.object(quality_gate, "_detect_scope", return_value=ui_only_scope),
            patch.object(quality_gate, "CACHE_FILE", cache_path),
            patch.object(quality_gate, "check_pytest_files", return_value=pytest_result) as mock_pytest_files,
            _patch_all_gates(),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        mock_pytest_files.assert_called_once_with(["tests/test_dashboard_rendering.py", "tests/test_other.py"])

    def test_full_flag_overrides_ui_only_scope(
        self,
        tmp_path: Path,
    ) -> None:
        """`--full` forces the full gate even when ui-only would be detected."""
        cache_path = tmp_path / ".quality_gate_cache"
        ui_only_scope = {
            "scope": "ui-only",
            "changed_test_files": [],
            "reason": "only UI assets and dev files changed",
        }

        with (
            patch("sys.argv", ["quality_gate.py", "--json", "--full"]),
            _patch_git_state("QS_208", "abc123", True),
            patch.object(quality_gate, "_detect_scope", return_value=ui_only_scope),
            patch.object(quality_gate, "CACHE_FILE", cache_path),
            patch.object(quality_gate, "check_pytest_files") as mock_pytest_files,
            _patch_all_gates() as mocks,
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        # All four lint gates + full pytest must have been called.
        for m in mocks:
            m.assert_called()
        # The UI fast-path pytest must NOT have been called.
        mock_pytest_files.assert_not_called()


# ===========================================================================
# QS-276 — `--impacted` inner loop + `--seed-testmon`
# ===========================================================================


def _cp(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    """Build a CompletedProcess stand-in for mocking `quality_gate._run`."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestImpactedToolingAvailable:
    """`_impacted_tooling_available` probes the venv for testmon + diff_cover."""

    @pytest.mark.parametrize(
        ("probe_rc", "expected"),
        [(0, True), (1, False)],
        ids=["both-importable", "missing"],
    )
    def test_probe_result_maps_to_bool(self, probe_rc: int, expected: bool) -> None:
        with patch.object(quality_gate, "_run", return_value=_cp(probe_rc)) as mock_run:
            assert quality_gate._impacted_tooling_available() is expected
        # Probes the venv interpreter, not the orchestrator.
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == quality_gate.VENV_PYTHON
        assert "diff_cover" in cmd[-1] and "testmon" in cmd[-1]


class TestIsCi:
    """`_is_ci` recognizes the `CI` env var and the GitHub Actions provider var."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("true", True),
            ("1", True),
            ("TRUE", True),
            ("  true ", True),
            # review-fix N3: broaden the truthy set beyond {1, true}.
            ("yes", True),
            ("on", True),
            # review-fix NH4: single-letter spellings some providers emit.
            ("y", True),
            ("t", True),
            ("T", True),
            ("false", False),
            ("0", False),
            ("n", False),
            ("", False),
        ],
    )
    def test_ci_env(self, value: str, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.setenv("CI", value)
        assert quality_gate._is_ci() is expected

    def test_ci_unset_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        assert quality_gate._is_ci() is False

    def test_github_actions_provider_var_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """review-fix N3: GitHub Actions sets GITHUB_ACTIONS=true even if CI is unset."""
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        assert quality_gate._is_ci() is True


class TestResolveDiffBase:
    """`_resolve_diff_base` walks origin/main → main → upstream merge-base."""

    @staticmethod
    def _router(
        rev_parse: dict[str, int],
        *,
        upstream: tuple[int, str] = (1, ""),
        merge_base: tuple[int, str] = (1, ""),
        reachable: dict[str, int] | None = None,
    ):
        """Build a `_run` side_effect keyed on the git subcommand.

        `reachable` (NH2) maps a candidate ref → returncode for the
        `git merge-base <ref> HEAD` reachability probe; default 0
        (reachable) so the pre-NH2 tests keep passing unchanged.
        """
        reachable = reachable or {}

        def _side_effect(cmd: list[str], *_a, **_k) -> subprocess.CompletedProcess[str]:
            if cmd[:2] == ["git", "fetch"]:
                return _cp(0)
            if cmd[:3] == ["git", "rev-parse", "--verify"]:
                return _cp(rev_parse.get(cmd[3], 1))
            if cmd[:2] == ["git", "rev-parse"]:  # @{u} upstream lookup
                return _cp(upstream[0], stdout=upstream[1])
            if cmd[:2] == ["git", "merge-base"]:
                # NH2 reachability probe is `merge-base <ref> HEAD`; the
                # upstream-path call is `merge-base HEAD <tracked>`.
                if cmd[3] == "HEAD":
                    return _cp(reachable.get(cmd[2], 0))
                return _cp(merge_base[0], stdout=merge_base[1])
            raise AssertionError(f"unexpected cmd {cmd!r}")

        return _side_effect

    def test_origin_main_wins(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(quality_gate, "_run", side_effect=self._router({"origin/main": 0, "main": 0})):
            assert quality_gate._resolve_diff_base() == "origin/main"
        # review-fix NH2 (#05): the chosen base is announced for debuggability.
        assert "diff base: origin/main" in capsys.readouterr().err

    def test_fetches_origin_main_first(self) -> None:
        with patch.object(quality_gate, "_run", side_effect=self._router({"origin/main": 0})) as mock_run:
            quality_gate._resolve_diff_base()
        first = mock_run.call_args_list[0].args[0]
        assert first == ["git", "fetch", "origin", "main"]

    def test_falls_back_to_local_main(self) -> None:
        with patch.object(quality_gate, "_run", side_effect=self._router({"origin/main": 1, "main": 0})):
            assert quality_gate._resolve_diff_base() == "main"

    def test_falls_back_to_upstream_merge_base(self, capsys: pytest.CaptureFixture[str]) -> None:
        router = self._router(
            {"origin/main": 1, "main": 1},
            upstream=(0, "origin/feature"),
            merge_base=(0, "deadbeefcafe"),
        )
        with patch.object(quality_gate, "_run", side_effect=router):
            assert quality_gate._resolve_diff_base() == "deadbeefcafe"
        # review-fix NH2 (#05): the merge-base sha + tracked ref are announced.
        assert "diff base: deadbeefcafe (merge-base with origin/feature)" in capsys.readouterr().err

    def test_none_when_no_upstream(self) -> None:
        router = self._router({"origin/main": 1, "main": 1}, upstream=(1, ""))
        with patch.object(quality_gate, "_run", side_effect=router):
            assert quality_gate._resolve_diff_base() is None

    def test_none_when_merge_base_fails(self) -> None:
        router = self._router(
            {"origin/main": 1, "main": 1},
            upstream=(0, "origin/feature"),
            merge_base=(1, ""),
        )
        with patch.object(quality_gate, "_run", side_effect=router):
            assert quality_gate._resolve_diff_base() is None

    def test_fetch_is_bounded_by_timeout(self) -> None:
        """review-fix S1: the hot-path fetch must pass a subprocess timeout."""
        with patch.object(quality_gate, "_run", side_effect=self._router({"origin/main": 0})) as mock_run:
            quality_gate._resolve_diff_base()
        fetch_call = mock_run.call_args_list[0]
        assert fetch_call.args[0] == ["git", "fetch", "origin", "main"]
        assert fetch_call.kwargs.get("timeout") == quality_gate._FETCH_TIMEOUT_SECONDS

    def test_fetch_failure_emits_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        """review-fix S1: a non-zero/timed-out fetch warns so a stale base is observable."""

        def _side_effect(cmd: list[str], *_a, **_k) -> subprocess.CompletedProcess[str]:
            if cmd[:2] == ["git", "fetch"]:
                return _cp(124, stderr="timed out after 15.0s")  # simulate a hung remote
            if cmd[:3] == ["git", "rev-parse", "--verify"] and cmd[3] == "origin/main":
                return _cp(0)
            if cmd[:2] == ["git", "merge-base"]:  # NH2 reachability probe — reachable
                return _cp(0)
            return _cp(1)

        with patch.object(quality_gate, "_run", side_effect=_side_effect):
            base = quality_gate._resolve_diff_base()
        assert base == "origin/main"  # stale local origin/main still resolves
        assert "git fetch origin main` failed/timed out" in capsys.readouterr().err

    def test_skips_origin_main_without_merge_base(self, capsys: pytest.CaptureFixture[str]) -> None:
        """review-fix NH2: a resolvable but unreachable ref (shallow clone) is skipped."""
        router = self._router(
            {"origin/main": 0, "main": 0},
            reachable={"origin/main": 1},  # origin/main has no common ancestor
        )
        with patch.object(quality_gate, "_run", side_effect=router):
            assert quality_gate._resolve_diff_base() == "main"  # falls through to reachable main
        assert "no merge-base with HEAD" in capsys.readouterr().err

    def test_none_when_no_ref_is_reachable(self) -> None:
        """review-fix NH2: both refs resolve but neither is reachable, and no upstream → None."""
        router = self._router(
            {"origin/main": 0, "main": 0},
            reachable={"origin/main": 1, "main": 1},
            upstream=(1, ""),
        )
        with patch.object(quality_gate, "_run", side_effect=router):
            assert quality_gate._resolve_diff_base() is None


class TestRunTimeout:
    """`_run` (review-fix S1) surfaces a subprocess timeout as a non-zero result."""

    def test_timeout_returns_nonzero_completed_process(self) -> None:
        with patch.object(
            quality_gate.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd=["git", "fetch"], timeout=5.0),
        ):
            result = quality_gate._run(["git", "fetch"], timeout=5.0)
        assert result.returncode == 124
        assert "timed out" in result.stderr

    def test_timeout_is_forwarded_to_subprocess_run(self) -> None:
        with patch.object(quality_gate.subprocess, "run", return_value=_cp(0)) as mock_run:
            quality_gate._run(["git", "status"], timeout=3.0)
        assert mock_run.call_args.kwargs.get("timeout") == 3.0

    def test_run_pins_utf8_replace_decoding(self) -> None:
        """review-fix SF1: _run must decode as utf-8/replace, not the locale codec."""
        with patch.object(quality_gate.subprocess, "run", return_value=_cp(0)) as mock_run:
            quality_gate._run(["git", "status"])
        assert mock_run.call_args.kwargs.get("encoding") == "utf-8"
        assert mock_run.call_args.kwargs.get("errors") == "replace"

    def test_run_decodes_non_ascii_output_without_crashing(self) -> None:
        """review-fix SF1: real non-ASCII subprocess output round-trips, never raises.

        review-fix MF1 (#04): use `sys.executable` (always present) rather
        than `VENV_PYTHON` (absent on CI runners) so this real-subprocess
        test runs everywhere — `_run` is interpreter-agnostic.
        """
        result = quality_gate._run([sys.executable, "-c", "import sys; sys.stdout.write('café—✓ déjà')"])
        assert result.returncode == 0
        assert "café" in result.stdout and "déjà" in result.stdout

    def test_missing_executable_returns_127(self) -> None:
        """review-fix MF1 (#04): a missing interpreter degrades to rc 127, not a raised FileNotFoundError."""
        with patch.object(quality_gate.subprocess, "run", side_effect=FileNotFoundError("no such file")):
            result = quality_gate._run(["/nonexistent/venv/bin/python", "-c", "pass"])
        assert result.returncode == 127
        assert "no such file" in result.stderr

    def test_timeout_whitespace_only_stderr_has_no_leading_blank(self) -> None:
        """review-fix SF-C (#04): whitespace-only stderr must not inject a leading blank line."""
        with patch.object(
            quality_gate.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd=["x"], timeout=5.0, output="", stderr="   \n  "),
        ):
            result = quality_gate._run(["x"], timeout=5.0)
        assert result.stderr == "timed out after 5.0s"  # not "...\ntimed out..."

    def test_timeout_preserves_partial_output(self) -> None:
        """review-fix NH1: partial stdout/stderr captured before the timeout is retained."""
        with patch.object(
            quality_gate.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(
                cmd=["x"], timeout=5.0, output="partial out", stderr="partial err"
            ),
        ):
            result = quality_gate._run(["x"], timeout=5.0)
        assert result.returncode == 124
        assert result.stdout == "partial out"
        assert "partial err" in result.stderr
        assert "timed out" in result.stderr  # the timeout marker is still appended

    def test_timeout_decodes_bytes_partial_output(self) -> None:
        """review-fix NH1: bytes partial output is decoded utf-8/replace."""
        with patch.object(
            quality_gate.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd=["x"], timeout=5.0, output=b"caf\xc3\xa9"),
        ):
            result = quality_gate._run(["x"], timeout=5.0)
        assert result.stdout == "café"


class TestEnsureTestmonDbSafe:
    """`_ensure_testmon_db_safe` deletes only a non-SQLite `.testmondata`."""

    def test_absent_is_noop(self, tmp_path: Path) -> None:
        db = tmp_path / ".testmondata"
        with patch.object(quality_gate, "TESTMON_DATA", db):
            quality_gate._ensure_testmon_db_safe()  # must not raise
        assert not db.exists()

    def test_ensure_safe_removes_orphan_sidecars_when_primary_absent(self, tmp_path: Path) -> None:
        """QS-283 A2 (AC#2): primary `.testmondata` gone but `-wal`/`-shm`
        linger (a run killed mid-`_purge_testmon_db`, which unlinks the primary
        first). Without cleanup, testmon reopens an empty DB against the stale
        WAL and selects `0 tests`. `_ensure_testmon_db_safe` must unlink the
        orphan sidecars so the next run rebuilds cleanly (select-all)."""
        db = tmp_path / ".testmondata"  # absent (never created)
        wal = tmp_path / ".testmondata-wal"
        wal.write_bytes(b"orphan-wal")
        shm = tmp_path / ".testmondata-shm"
        shm.write_bytes(b"orphan-shm")
        with patch.object(quality_gate, "TESTMON_DATA", db):
            quality_gate._ensure_testmon_db_safe()
        assert not db.exists()
        assert not wal.exists() and not shm.exists(), "orphan WAL/SHM sidecars must be removed"

    def test_ensure_safe_absent_primary_no_sidecars_is_noop(self, tmp_path: Path) -> None:
        """QS-283 A2: absent primary with NO sidecars must not raise (the
        ordinary first-ever-run case); `missing_ok=True` tolerates it."""
        db = tmp_path / ".testmondata"
        with patch.object(quality_gate, "TESTMON_DATA", db):
            quality_gate._ensure_testmon_db_safe()  # must not raise
        assert not db.exists()

    def test_valid_sqlite_matching_schema_is_kept(self, tmp_path: Path) -> None:
        db = tmp_path / ".testmondata"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (x)")
        conn.execute("PRAGMA user_version = 14")
        conn.commit()
        conn.close()
        with (
            patch.object(quality_gate, "TESTMON_DATA", db),
            # QS-278 #01-1: matching schema → valid incremental baseline → kept.
            patch.object(quality_gate, "_testmon_schema_version", return_value=14),
        ):
            quality_gate._ensure_testmon_db_safe()
        assert db.exists(), "a valid, schema-matching SQLite DB must be preserved"

    def test_schema_version_mismatch_is_removed_with_sidecars(self, tmp_path: Path) -> None:
        """QS-278 #01-1: testmon rebuilds in place on a `user_version` mismatch,
        leaving the file present. We must purge it (and its WAL/SHM sidecars)
        so the select-all run resets the accumulated `--cov-append` coverage."""
        db = tmp_path / ".testmondata"
        conn = sqlite3.connect(str(db))
        conn.execute("PRAGMA user_version = 13")  # stale (testmon expects 14)
        conn.commit()
        conn.close()
        wal = tmp_path / ".testmondata-wal"
        wal.write_bytes(b"stale-wal")
        shm = tmp_path / ".testmondata-shm"
        shm.write_bytes(b"stale-shm")
        with (
            patch.object(quality_gate, "TESTMON_DATA", db),
            patch.object(quality_gate, "_testmon_schema_version", return_value=14),
        ):
            quality_gate._ensure_testmon_db_safe()
        assert not db.exists(), "a schema-mismatched .testmondata must be purged"
        assert not wal.exists() and not shm.exists(), "WAL/SHM sidecars must be purged too"

    def test_unknown_schema_version_keeps_db(self, tmp_path: Path) -> None:
        """QS-278 #01-1: when the expected schema can't be probed (testmon not
        importable), fall back to leaving a readable DB intact, not purging."""
        db = tmp_path / ".testmondata"
        conn = sqlite3.connect(str(db))
        conn.execute("PRAGMA user_version = 7")
        conn.commit()
        conn.close()
        with (
            patch.object(quality_gate, "TESTMON_DATA", db),
            patch.object(quality_gate, "_testmon_schema_version", return_value=None),
        ):
            quality_gate._ensure_testmon_db_safe()
        assert db.exists(), "an unknown expected schema must not trigger a purge"

    def test_corrupt_db_is_removed_with_sidecars(self, tmp_path: Path) -> None:
        """QS-278 #01-2: a corrupt DB AND its orphaned WAL/SHM sidecars are purged."""
        db = tmp_path / ".testmondata"
        db.write_bytes(b"this is not a sqlite database")
        wal = tmp_path / ".testmondata-wal"
        wal.write_bytes(b"orphan-wal")
        shm = tmp_path / ".testmondata-shm"
        shm.write_bytes(b"orphan-shm")
        with patch.object(quality_gate, "TESTMON_DATA", db):
            quality_gate._ensure_testmon_db_safe()
        assert not db.exists(), "corrupt .testmondata must be removed to force select-all"
        assert not wal.exists() and not shm.exists(), "orphaned WAL/SHM sidecars must be removed"

    def test_unlink_missing_safe_when_db_vanishes(self, tmp_path: Path) -> None:
        """review-fix N6: the file disappearing between probe and unlink must not raise."""
        db = tmp_path / ".testmondata"
        db.write_bytes(b"corrupt")

        def _boom(*_a: object, **_k: object) -> None:
            db.unlink()  # simulate a concurrent run removing it first
            raise sqlite3.DatabaseError("file is not a database")

        with (
            patch.object(quality_gate, "TESTMON_DATA", db),
            patch.object(quality_gate.sqlite3, "connect", side_effect=_boom),
        ):
            quality_gate._ensure_testmon_db_safe()  # must NOT raise FileNotFoundError
        assert not db.exists()

    def test_locked_but_valid_db_is_preserved(self, tmp_path: Path) -> None:
        """review-fix SF1: 'database is locked' (OperationalError) must NOT delete a valid baseline.

        OperationalError is a subclass of DatabaseError, so the corruption
        probe would otherwise wipe a recoverable DB that is merely busy
        with a concurrent --seed-testmon/--impacted run.
        """
        db = tmp_path / ".testmondata"
        db.write_bytes(b"placeholder-valid-baseline")

        def _locked(*_a: object, **_k: object) -> None:
            raise sqlite3.OperationalError("database is locked")

        with (
            patch.object(quality_gate, "TESTMON_DATA", db),
            patch.object(quality_gate.sqlite3, "connect", side_effect=_locked),
        ):
            quality_gate._ensure_testmon_db_safe()
        assert db.exists(), "a locked-but-valid .testmondata must be left intact"


class TestBuildImpactedCmds:
    """Pure argv builders for the `--impacted` seam."""

    def test_testmon_supports_xdist_enabled_by_default(self) -> None:
        """review-fix: testmon 2.2.0 attributes coverage across xdist workers, so we parallelize."""
        assert quality_gate._TESTMON_SUPPORTS_XDIST is True

    def test_testmon_cmd_shape_when_xdist_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(quality_gate, "_TESTMON_SUPPORTS_XDIST", False)
        cmd = quality_gate._build_testmon_cmd()
        assert cmd[:4] == [quality_gate.VENV_PYTHON, "-m", "pytest", "--testmon"]
        assert f"--cov={quality_gate.SRC_DIR}" in cmd
        # QS-278: coverage accumulates across inner-loop runs so a 0/partial
        # testmon reselection still covers every line changed vs origin/main.
        assert "--cov-append" in cmd
        # Empty --cov-report= must precede the xml report (clears pytest.ini).
        assert cmd.index("--cov-report=") < cmd.index(f"--cov-report=xml:{quality_gate.COVERAGE_XML}")
        assert "--cov-fail-under=100" not in cmd  # verdict is diff-cover's job
        assert "-n" not in cmd  # serial only when testmon⊕xdist is disabled
        # review-fix MF1: the self-test file is excluded BY PATH, not by the
        # shared `integration` marker (domain integration tests cover
        # production code and must stay selected).
        assert f"--ignore={quality_gate.TESTS_DIR / 'test_quality_gate.py'}" in cmd

    def test_testmon_cmd_does_not_deselect_integration_marker(self) -> None:
        """review-fix MF1: must NOT carry `-m "not integration"` — that dropped domain coverage.

        review-fix MF1 (#04): patch `_pytest_workers` so this argv unit test
        never reaches the real `VENV_PYTHON` xdist probe (absent on CI).
        """
        with patch.object(quality_gate, "_pytest_workers", return_value=None):
            cmd = quality_gate._build_testmon_cmd()
        assert "not integration" not in cmd
        assert "-m" not in cmd[3:]  # no marker filter beyond `python -m pytest`

    def test_testmon_cmd_ignores_only_the_selftest_file(self) -> None:
        """review-fix MF1: exactly one --ignore, targeting the testmon self-tests."""
        with patch.object(quality_gate, "_pytest_workers", return_value=None):
            cmd = quality_gate._build_testmon_cmd()
        ignores = [a for a in cmd if a.startswith("--ignore=")]
        assert ignores == [f"--ignore={quality_gate.TESTS_DIR / 'test_quality_gate.py'}"]

    def test_testmon_cmd_adds_workers_when_xdist_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(quality_gate, "_TESTMON_SUPPORTS_XDIST", True)
        with patch.object(quality_gate, "_pytest_workers", return_value="auto"):
            cmd = quality_gate._build_testmon_cmd()
        assert cmd[cmd.index("-n") + 1] == "auto"

    def test_testmon_cmd_serial_when_xdist_enabled_but_workers_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(quality_gate, "_TESTMON_SUPPORTS_XDIST", True)
        with patch.object(quality_gate, "_pytest_workers", return_value=None):
            cmd = quality_gate._build_testmon_cmd()
        assert "-n" not in cmd

    def test_diff_cover_cmd(self) -> None:
        cmd = quality_gate._build_diff_cover_cmd("origin/main")
        assert cmd == [
            quality_gate._venv_tool("diff-cover"),
            str(quality_gate.COVERAGE_XML),
            "--compare-branch=origin/main",
            # review-fix SF-A (#04): untracked new files must count as changes.
            "--include-untracked",
            "--fail-under=100",
        ]


class TestCheckImpacted:
    """`check_impacted` orchestrator + exit-code mapping (mocked seam)."""

    @pytest.fixture(autouse=True)
    def _testmon_db_present(self, tmp_path_factory: pytest.TempPathFactory):
        """QS-278: by default point `TESTMON_DATA` at an existing file so the
        fresh-baseline branch (which resets the real `.coverage` via
        `--cov-append`) never fires against the real FS during these
        mocked-seam tests. Dedicated tests below re-patch it to exercise the
        branch explicitly.

        QS-283 A1: also point `COVERAGE_DATA` at a tmp file so the orphan-shard
        glob `check_impacted` now runs at the top can never reap the real
        repo's `.coverage.*` shards mid-suite."""
        root = tmp_path_factory.mktemp("tmdb")
        db = root / ".testmondata"
        db.write_bytes(b"x")
        with (
            patch.object(quality_gate, "TESTMON_DATA", db),
            patch.object(quality_gate, "COVERAGE_DATA", root / ".coverage"),
        ):
            yield

    def test_tooling_missing_returns_3(self) -> None:
        with patch.object(quality_gate, "_impacted_tooling_available", return_value=False):
            assert quality_gate.check_impacted() == 3

    def test_no_base_in_ci_returns_4(self) -> None:
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_resolve_diff_base", return_value=None),
            patch.object(quality_gate, "_is_ci", return_value=True),
        ):
            assert quality_gate.check_impacted() == 4

    def test_no_base_locally_warns_and_passes(self) -> None:
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_resolve_diff_base", return_value=None),
            patch.object(quality_gate, "_is_ci", return_value=False),
        ):
            assert quality_gate.check_impacted() == 0

    def test_selected_tests_fail_returns_1(self) -> None:
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_resolve_diff_base", return_value="origin/main"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            # Isolate _run to the diff-cover call: the cmd builder probes
            # xdist via _run, so stub it (it is unit-tested separately).
            patch.object(quality_gate, "_build_testmon_cmd", return_value=["pytest"]),
            patch.object(quality_gate, "_stream_pytest", return_value={"name": "pytest", "passed": False}),
            patch.object(quality_gate, "_run") as mock_run,
        ):
            assert quality_gate.check_impacted() == 1
        mock_run.assert_not_called()  # diff-cover never runs if tests failed

    def test_diff_coverage_below_100_returns_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        xml = tmp_path / "coverage.xml"
        absent_db = tmp_path / ".testmondata"  # never created → select-all, no self-heal

        def _emit_xml(*_a: object, **_k: object) -> dict:
            xml.write_text("<coverage/>")  # simulate pytest-cov writing a fresh report
            return {"name": "pytest", "passed": True}

        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_resolve_diff_base", return_value="origin/main"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            # QS-283 A4: an absent DB → select-all (was_incremental False), so a
            # changed-line FAIL is ground truth and the self-heal retry is skipped.
            patch.object(quality_gate, "TESTMON_DATA", absent_db),
            patch.object(quality_gate, "_reset_coverage_data"),
            patch.object(quality_gate, "COVERAGE_XML", xml),
            patch.object(quality_gate, "_build_testmon_cmd", return_value=["pytest"]),
            patch.object(quality_gate, "_stream_pytest", side_effect=_emit_xml),
            patch.object(quality_gate, "_run", return_value=_cp(1, stdout="Coverage: 50%", stderr="fail")),
        ):
            assert quality_gate.check_impacted() == 1
        err = capsys.readouterr().err
        assert "changed lines <100% covered" in err
        # review-fix S5: failure points at a reseed when the baseline may be stale.
        assert "--seed-testmon" in err

    def test_diff_cover_timeout_returns_1_with_distinct_verdict(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """review-fix SF2 (#03): a timed-out diff-cover (124) reports a timeout, not a coverage verdict."""
        xml = tmp_path / "coverage.xml"

        def _emit_xml(*_a: object, **_k: object) -> dict:
            xml.write_text("<coverage/>")
            return {"name": "pytest", "passed": True}

        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_resolve_diff_base", return_value="origin/main"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            patch.object(quality_gate, "COVERAGE_XML", xml),
            patch.object(quality_gate, "_build_testmon_cmd", return_value=["pytest"]),
            patch.object(quality_gate, "_stream_pytest", side_effect=_emit_xml),
            patch.object(quality_gate, "_run", return_value=_cp(124, stderr="timed out after 60.0s")),
        ):
            assert quality_gate.check_impacted() == 1
        err = capsys.readouterr().err
        assert "diff-cover timed out" in err
        assert "changed lines <100%" not in err

    def test_missing_coverage_xml_returns_1(self, tmp_path: Path) -> None:
        """review-fix N7: if pytest-cov fails to emit coverage.xml, fail loudly (don't diff-cover a stale file)."""
        missing = tmp_path / "coverage.xml"  # never created
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_resolve_diff_base", return_value="origin/main"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            patch.object(quality_gate, "COVERAGE_XML", missing),
            patch.object(quality_gate, "_build_testmon_cmd", return_value=["pytest"]),
            patch.object(quality_gate, "_stream_pytest", return_value={"name": "pytest", "passed": True}),
            patch.object(quality_gate, "_run") as mock_run,
        ):
            assert quality_gate.check_impacted() == 1
        mock_run.assert_not_called()  # diff-cover never runs without a coverage report

    def test_pass_returns_0(self, tmp_path: Path) -> None:
        xml = tmp_path / "coverage.xml"

        def _emit_xml(*_a: object, **_k: object) -> dict:
            xml.write_text("<coverage/>")  # fresh report, written AFTER the SF1 pre-delete
            return {"name": "pytest", "passed": True}

        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_resolve_diff_base", return_value="origin/main"),
            patch.object(quality_gate, "_ensure_testmon_db_safe") as mock_safe,
            patch.object(quality_gate, "COVERAGE_XML", xml),
            patch.object(quality_gate, "_build_testmon_cmd", return_value=["pytest"]),
            patch.object(quality_gate, "_stream_pytest", side_effect=_emit_xml),
            patch.object(quality_gate, "_run", return_value=_cp(0, stdout="Coverage: 100%")) as mock_run,
        ):
            assert quality_gate.check_impacted() == 0
        mock_safe.assert_called_once()
        dc_cmd = mock_run.call_args.args[0]
        assert dc_cmd[2] == "--compare-branch=origin/main"
        # review-fix NH1: the diff-cover subprocess is bounded by a timeout.
        assert mock_run.call_args.kwargs.get("timeout") == quality_gate._DIFF_COVER_TIMEOUT_SECONDS

    def test_stale_coverage_xml_cleared_so_emission_failure_fails(self, tmp_path: Path) -> None:
        """review-fix SF1 (#05): a stale coverage.xml must not mask a pytest-cov emission failure.

        A previous run's report is present, but this run's (mocked) pytest
        does NOT emit a fresh one. The pre-run unlink + exists-guard must
        still FAIL (return 1) and never run diff-cover against stale data.
        """
        xml = tmp_path / "coverage.xml"
        xml.write_text("<coverage>STALE from a previous run</coverage>")
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_resolve_diff_base", return_value="origin/main"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            patch.object(quality_gate, "COVERAGE_XML", xml),
            patch.object(quality_gate, "_build_testmon_cmd", return_value=["pytest"]),
            # pytest runs but does NOT (re)write coverage.xml this time.
            patch.object(quality_gate, "_stream_pytest", return_value={"name": "pytest", "passed": True}),
            patch.object(quality_gate, "_run") as mock_run,
        ):
            assert quality_gate.check_impacted() == 1
        assert not xml.exists(), "the stale coverage.xml must have been cleared before the run"
        mock_run.assert_not_called()  # diff-cover never scores against a stale report

    def test_fresh_baseline_resets_accumulated_coverage(self, tmp_path: Path) -> None:
        """QS-278: an absent `.testmondata` (testmon about to select ALL tests)
        resets the accumulated `--cov-append` coverage data before the run."""
        xml = tmp_path / "coverage.xml"
        absent_db = tmp_path / ".testmondata"  # never created

        def _emit_xml(*_a: object, **_k: object) -> dict:
            xml.write_text("<coverage/>")
            return {"name": "pytest", "passed": True}

        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_resolve_diff_base", return_value="origin/main"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            patch.object(quality_gate, "TESTMON_DATA", absent_db),
            patch.object(quality_gate, "COVERAGE_XML", xml),
            patch.object(quality_gate, "_reset_coverage_data") as mock_reset,
            patch.object(quality_gate, "_build_testmon_cmd", return_value=["pytest"]),
            patch.object(quality_gate, "_stream_pytest", side_effect=_emit_xml),
            patch.object(quality_gate, "_run", return_value=_cp(0, stdout="Coverage: 100%")),
        ):
            assert quality_gate.check_impacted() == 0
        mock_reset.assert_called_once()

    def test_existing_baseline_keeps_accumulated_coverage(self, tmp_path: Path) -> None:
        """QS-278: when `.testmondata` exists, coverage accumulation is
        intentional — the reset must NOT fire (so a 0/partial reselection
        keeps prior runs' coverage of changed-vs-origin lines)."""
        xml = tmp_path / "coverage.xml"
        present_db = tmp_path / ".testmondata"
        present_db.write_bytes(b"x")

        def _emit_xml(*_a: object, **_k: object) -> dict:
            xml.write_text("<coverage/>")
            return {"name": "pytest", "passed": True}

        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_resolve_diff_base", return_value="origin/main"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            patch.object(quality_gate, "TESTMON_DATA", present_db),
            patch.object(quality_gate, "COVERAGE_XML", xml),
            patch.object(quality_gate, "_reset_coverage_data") as mock_reset,
            patch.object(quality_gate, "_build_testmon_cmd", return_value=["pytest"]),
            patch.object(quality_gate, "_stream_pytest", side_effect=_emit_xml),
            patch.object(quality_gate, "_run", return_value=_cp(0, stdout="Coverage: 100%")),
        ):
            assert quality_gate.check_impacted() == 0
        mock_reset.assert_not_called()


class TestTestmonSchemaVersion:
    """QS-278 #01-1: `_testmon_schema_version` probes VENV_PYTHON for testmon's DATA_VERSION."""

    def test_parses_int_from_probe_stdout(self) -> None:
        with patch.object(quality_gate, "_run", return_value=_cp(0, stdout="14\n")) as mock_run:
            assert quality_gate._testmon_schema_version() == 14
        assert mock_run.call_args.args[0][0] == quality_gate.VENV_PYTHON

    def test_returns_none_when_probe_fails(self) -> None:
        with patch.object(quality_gate, "_run", return_value=_cp(1, stderr="ModuleNotFoundError")):
            assert quality_gate._testmon_schema_version() is None

    def test_returns_none_on_unparseable_stdout(self) -> None:
        with patch.object(quality_gate, "_run", return_value=_cp(0, stdout="not-an-int")):
            assert quality_gate._testmon_schema_version() is None


class TestResetCoverageData:
    """QS-278: `_reset_coverage_data` clears the persistent coverage data."""

    def test_removes_primary_data_and_xdist_shards(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # #01-5: shards are globbed from COVERAGE_DATA's own dir — patching it
        # alone must clear both the primary file and the shards.
        data = tmp_path / ".coverage"
        data.write_text("primary")
        shard = tmp_path / ".coverage.host.12345"
        shard.write_text("shard")
        monkeypatch.setattr(quality_gate, "COVERAGE_DATA", data)

        quality_gate._reset_coverage_data()

        assert not data.exists()
        assert not shard.exists()

    def test_is_noop_when_no_data_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absent data must not raise (first-ever run)."""
        monkeypatch.setattr(quality_gate, "COVERAGE_DATA", tmp_path / ".coverage")
        quality_gate._reset_coverage_data()  # must not raise


class TestCleanOrphanCovShards:
    """QS-283 A1 (AC#1): `_clean_orphan_cov_shards` reaps only `.coverage.*`
    shards; the combined `.coverage` survives."""

    def test_removes_shard_but_keeps_combined_coverage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        combined = tmp_path / ".coverage"
        combined.write_text("combined")
        shard = tmp_path / ".coverage.host.4242.XYZ"
        shard.write_text("orphan shard")
        monkeypatch.setattr(quality_gate, "COVERAGE_DATA", combined)

        quality_gate._clean_orphan_cov_shards()

        assert combined.exists(), "the combined .coverage must survive (warm baseline)"
        assert not shard.exists(), "a pre-existing orphan shard must be removed"

    def test_is_noop_when_no_shards_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        combined = tmp_path / ".coverage"
        combined.write_text("combined")
        monkeypatch.setattr(quality_gate, "COVERAGE_DATA", combined)
        quality_gate._clean_orphan_cov_shards()  # must not raise
        assert combined.exists()

    def test_uses_same_glob_as_reset_coverage_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two helpers must share the shard-matching rule (they differ
        ONLY in whether the primary `.coverage` is also unlinked)."""
        combined = tmp_path / ".coverage"
        combined.write_text("combined")
        for name in (".coverage.a.1", ".coverage.b.2"):
            (tmp_path / name).write_text("shard")
        monkeypatch.setattr(quality_gate, "COVERAGE_DATA", combined)

        quality_gate._clean_orphan_cov_shards()
        remaining = sorted(p.name for p in tmp_path.glob(".coverage*"))
        assert remaining == [".coverage"], remaining


class TestRebuildTestmonBaseline:
    """QS-283: `_rebuild_testmon_baseline` purges the DB AND clears coverage."""

    def test_purges_db_and_resets_coverage(self) -> None:
        with (
            patch.object(quality_gate, "_purge_testmon_db") as mock_purge,
            patch.object(quality_gate, "_reset_coverage_data") as mock_reset,
        ):
            quality_gate._rebuild_testmon_baseline()
        mock_purge.assert_called_once()
        mock_reset.assert_called_once()


class TestCheckImpactedSelfHeal:
    """QS-283 A4 (AC#4, AC#7): the self-heal retry on an incremental
    changed-line FAIL.

    The seam: patch `_run_impacted_pass` with a list of verdicts (its two
    return values drive the retry branch), patch `_rebuild_testmon_baseline`
    (so the retry touches no disk), patch `_clean_orphan_cov_shards` (so the
    real `.coverage` dir is never globbed), and control `was_incremental` by
    seeding / clearing `.testmondata` under `tmp_path`.
    """

    CHANGED = quality_gate._IMPACTED_CHANGED_LINES_UNCOVERED
    PASS = quality_gate._IMPACTED_PASS
    TESTS_FAILED = quality_gate._IMPACTED_TESTS_FAILED

    def _run(
        self,
        *,
        db: Path,
        verdicts: list[str],
        select_all: list[bool] | None = None,
    ):
        """Drive `check_impacted` with a mocked `_run_impacted_pass`.

        `_run_impacted_pass` returns `(verdict, ran_select_all)`; `select_all`
        supplies the per-call `ran_select_all` flag (defaults to all False —
        an incremental selection)."""
        flags = select_all if select_all is not None else [False] * len(verdicts)
        mock_pass = MagicMock(side_effect=list(zip(verdicts, flags)))
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_resolve_diff_base", return_value="origin/main"),
            patch.object(quality_gate, "TESTMON_DATA", db),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "_rebuild_testmon_baseline") as mock_rebuild,
            patch.object(quality_gate, "_run_impacted_pass", mock_pass),
        ):
            rc = quality_gate.check_impacted()
        return rc, mock_rebuild, mock_pass

    @staticmethod
    def _incremental_db(tmp_path: Path) -> Path:
        db = tmp_path / ".testmondata"
        db.write_bytes(b"warm-baseline")  # present + non-empty → was_incremental True
        return db

    def test_incremental_false_fail_recovers_to_pass(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An incremental changed-line FAIL that is a desync recovers to PASS
        after exactly one rebuild + retry; the self-heal notice is emitted."""
        rc, mock_rebuild, mock_pass = self._run(
            db=self._incremental_db(tmp_path), verdicts=[self.CHANGED, self.PASS]
        )
        assert rc == 0
        mock_rebuild.assert_called_once()
        assert mock_pass.call_count == 2, "exactly one rebuild + one retry"
        assert "rebuilding testmon baseline" in capsys.readouterr().err

    def test_incremental_genuine_gap_still_exits_1(self, tmp_path: Path) -> None:
        """A genuine gap fails the retry too → exit 1 (one rebuild, one retry)."""
        rc, mock_rebuild, mock_pass = self._run(
            db=self._incremental_db(tmp_path), verdicts=[self.CHANGED, self.CHANGED]
        )
        assert rc == 1
        mock_rebuild.assert_called_once()
        assert mock_pass.call_count == 2

    def test_select_all_fail_does_not_retry(self, tmp_path: Path) -> None:
        """A select-all run (absent DB → was_incremental False) FAILs as ground
        truth — no wasted rebuild/retry on the normal TDD-red case."""
        absent = tmp_path / ".testmondata"  # never created
        rc, mock_rebuild, mock_pass = self._run(db=absent, verdicts=[self.CHANGED])
        assert rc == 1
        mock_rebuild.assert_not_called()
        assert mock_pass.call_count == 1

    def test_empty_db_is_not_incremental_so_no_retry(self, tmp_path: Path) -> None:
        """A present-but-empty `.testmondata` (size 0) is select-all, not
        incremental — `was_incremental` keys on size>0, so no retry fires."""
        empty = tmp_path / ".testmondata"
        empty.write_bytes(b"")  # present but zero-length
        rc, mock_rebuild, mock_pass = self._run(db=empty, verdicts=[self.CHANGED])
        assert rc == 1
        mock_rebuild.assert_not_called()
        assert mock_pass.call_count == 1

    def test_testmondata_vanishing_before_stat_is_non_incremental(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Review fix #02: if `.testmondata` is unlinked (concurrent run / other
        worktree / mid-purge) so `TESTMON_DATA.stat()` raises `FileNotFoundError`,
        `check_impacted` must NOT crash — it treats the run as non-incremental
        (was_incremental False), so a changed-line FAIL exits 1 with no retry."""
        fake_db = MagicMock()
        fake_db.stat.side_effect = FileNotFoundError  # vanished between probe and read
        mock_pass = MagicMock(side_effect=[(self.CHANGED, False)])
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_resolve_diff_base", return_value="origin/main"),
            patch.object(quality_gate, "TESTMON_DATA", fake_db),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "_rebuild_testmon_baseline") as mock_rebuild,
            patch.object(quality_gate, "_run_impacted_pass", mock_pass),
        ):
            assert quality_gate.check_impacted() == 1  # must not raise
        fake_db.stat.assert_called_once()
        mock_rebuild.assert_not_called()  # non-incremental → no self-heal retry
        assert mock_pass.call_count == 1
        assert "rebuilding testmon baseline" not in capsys.readouterr().err

    def test_first_pass_success_never_rebuilds_or_emits_notice(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A never-failed PASS is distinguishable from a recovered PASS: no
        rebuild, no self-heal notice."""
        rc, mock_rebuild, mock_pass = self._run(
            db=self._incremental_db(tmp_path), verdicts=[self.PASS]
        )
        assert rc == 0
        mock_rebuild.assert_not_called()
        assert mock_pass.call_count == 1
        assert "rebuilding testmon baseline" not in capsys.readouterr().err

    def test_non_retriable_verdict_exits_1_without_retry(self, tmp_path: Path) -> None:
        """A non-changed-line failure (e.g. selected tests failed) is genuine
        even on an incremental run — it must not trigger the self-heal."""
        rc, mock_rebuild, mock_pass = self._run(
            db=self._incremental_db(tmp_path), verdicts=[self.TESTS_FAILED]
        )
        assert rc == 1
        mock_rebuild.assert_not_called()
        assert mock_pass.call_count == 1

    def test_first_pass_select_alled_suppresses_retry(self, tmp_path: Path) -> None:
        """Review fix #01: even with a warm pre-hygiene baseline
        (`was_incremental` True), if the FIRST pass itself select-all'd
        (`ran_select_all` True — hygiene purged a corrupt/schema-mismatched DB
        mid-pass) the changed-line FAIL is ground truth, so no retry fires."""
        rc, mock_rebuild, mock_pass = self._run(
            db=self._incremental_db(tmp_path), verdicts=[self.CHANGED], select_all=[True]
        )
        assert rc == 1
        mock_rebuild.assert_not_called()
        assert mock_pass.call_count == 1

    def test_corrupt_baseline_purged_midpass_does_not_self_heal(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Review fix #01 (real path, no mock of `_run_impacted_pass`): a
        present+non-empty but CORRUPT `.testmondata` makes `was_incremental`
        True, yet `_ensure_testmon_db_safe` purges it inside the first pass so
        that pass select-alls. A genuine changed-line gap must run EXACTLY one
        pass, emit NO self-heal notice, and exit 1 — not waste a second
        full-suite select-all."""
        db = tmp_path / ".testmondata"
        db.write_bytes(b"not a sqlite database")  # warm pre-hygiene, purged as corrupt
        xml = tmp_path / "coverage.xml"

        def _emit_xml(*_a: object, **_k: object) -> dict:
            xml.write_text("<coverage/>")
            return {"name": "pytest", "passed": True}

        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_resolve_diff_base", return_value="origin/main"),
            patch.object(quality_gate, "TESTMON_DATA", db),
            patch.object(quality_gate, "COVERAGE_DATA", tmp_path / ".coverage"),
            patch.object(quality_gate, "COVERAGE_XML", xml),
            # _ensure_testmon_db_safe runs for REAL → detects corruption → purges
            # db → the pass select-alls (ran_select_all True).
            patch.object(quality_gate, "_build_testmon_cmd", return_value=["pytest"]),
            patch.object(quality_gate, "_stream_pytest", side_effect=_emit_xml) as mock_stream,
            patch.object(quality_gate, "_run", return_value=_cp(1, stdout="Coverage: 50%", stderr="gap")),
            patch.object(quality_gate, "_rebuild_testmon_baseline") as mock_rebuild,
        ):
            assert quality_gate.check_impacted() == 1
        assert not db.exists(), "the corrupt baseline must have been purged by hygiene"
        mock_rebuild.assert_not_called()
        assert mock_stream.call_count == 1, "exactly one pass — no wasted self-heal retry"
        assert "rebuilding testmon baseline" not in capsys.readouterr().err

    def test_retry_non_coverage_failure_surfaces_its_diagnostic(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Review fix #03 (real `_run_impacted_pass`): a self-heal retry whose
        SECOND pass fails for a non-coverage reason (selected tests failed)
        exits 1 AND still surfaces that pass's per-verdict diagnostic on stderr
        — the collapse to exit 1 does not swallow the message (the fix-#01
        claim, now under test).

        First pass: warm incremental baseline → changed-line miss (triggers
        self-heal). Second pass (post-rebuild): selected tests fail."""
        db = tmp_path / ".testmondata"
        db.write_bytes(b"warm-baseline")  # present + non-empty → was_incremental True
        xml = tmp_path / "coverage.xml"

        def _stream(*_a: object, **_k: object) -> dict:
            # Call 1 (first pass): write coverage.xml and pass so diff-cover runs.
            # Call 2 (retry pass): selected tests fail → _IMPACTED_TESTS_FAILED.
            if _stream.calls == 0:
                xml.write_text("<coverage/>")
                _stream.calls += 1
                return {"name": "pytest", "passed": True}
            _stream.calls += 1
            return {"name": "pytest", "passed": False}

        _stream.calls = 0  # type: ignore[attr-defined]

        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_resolve_diff_base", return_value="origin/main"),
            patch.object(quality_gate, "TESTMON_DATA", db),
            patch.object(quality_gate, "COVERAGE_DATA", tmp_path / ".coverage"),
            patch.object(quality_gate, "COVERAGE_XML", xml),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            # No-op hygiene so the warm DB stays present → both passes incremental.
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            patch.object(quality_gate, "_build_testmon_cmd", return_value=["pytest"]),
            patch.object(quality_gate, "_stream_pytest", side_effect=_stream),
            # First pass diff-cover → non-zero (changed lines uncovered).
            patch.object(quality_gate, "_run", return_value=_cp(1, stdout="Coverage: 50%", stderr="gap")),
            # Rebuild is a no-op (keeps db present) so the retry runs the real pass.
            patch.object(quality_gate, "_rebuild_testmon_baseline") as mock_rebuild,
        ):
            assert quality_gate.check_impacted() == 1
        mock_rebuild.assert_called_once()  # self-heal fired
        err = capsys.readouterr().err
        assert "rebuilding testmon baseline" in err
        # The retry's own per-verdict diagnostic must still surface.
        assert "FAIL (selected tests failed)" in err


class TestTestmonAvailable:
    """review-fix S2: `_testmon_available` probes ONLY testmon, never diff-cover."""

    @pytest.mark.parametrize(
        ("probe_rc", "expected"),
        [(0, True), (1, False)],
        ids=["importable", "missing"],
    )
    def test_probe_result_maps_to_bool(self, probe_rc: int, expected: bool) -> None:
        with patch.object(quality_gate, "_run", return_value=_cp(probe_rc)) as mock_run:
            assert quality_gate._testmon_available() is expected
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == quality_gate.VENV_PYTHON
        assert "testmon" in cmd[-1]
        assert "diff_cover" not in cmd[-1]  # narrower than _impacted_tooling_available


class TestSeedTestmon:
    """`seed_testmon` refreshes the DB with no pass/fail verdict."""

    @pytest.fixture(autouse=True)
    def _stub_rebuild(self, tmp_path: Path):
        """QS-283 A3: `seed_testmon` now calls `_rebuild_testmon_baseline`,
        which purges the real `.testmondata` and `.coverage`. Stub it by
        default so these mocked-seam tests never touch the real FS; the
        dedicated ordering test re-patches it with its own spy.

        QS-286: also redirect `SEED_STATUS` to a tmp sibling so the marker
        writes never pollute the repo root, and tests can read it back."""
        with (
            patch.object(quality_gate, "_rebuild_testmon_baseline"),
            patch.object(quality_gate, "SEED_STATUS", tmp_path / ".testmondata.seed-status"),
        ):
            yield

    def _marker(self) -> dict:
        return json.loads(quality_gate.SEED_STATUS.read_text())

    def test_running_marker_written_after_probe_before_pytest(self) -> None:
        """AC#1: a `running` marker (with pid + started) exists by the time
        the pytest pass runs — captured here from inside `_stream_pytest`."""
        seen: dict = {}
        with (
            patch.object(quality_gate, "_testmon_available", return_value=True),
            patch.object(quality_gate, "_build_seed_testmon_cmd", return_value=["SEED_CMD"]),
            patch.object(
                quality_gate,
                "_stream_pytest",
                side_effect=lambda *_a, **_k: (seen.update(self._marker()), {"returncode": 0})[1],
            ),
        ):
            assert quality_gate.seed_testmon() == 0
        assert seen["state"] == "running"
        assert seen["pid"] == os.getpid()
        assert "started" in seen

    @pytest.mark.parametrize("rc", [0, 1], ids=["clean", "test-failures"])
    def test_ok_marker_on_rc_lt_2(self, rc: int) -> None:
        """AC#1: rc < 2 (DB written) → final marker state=ok with returncode/finished."""
        with (
            patch.object(quality_gate, "_testmon_available", return_value=True),
            patch.object(quality_gate, "_build_seed_testmon_cmd", return_value=["SEED_CMD"]),
            patch.object(quality_gate, "_stream_pytest", return_value={"returncode": rc}),
        ):
            assert quality_gate.seed_testmon() == 0
        marker = self._marker()
        assert marker["state"] == "ok"
        assert marker["returncode"] == rc
        assert "finished" in marker and "started" in marker

    def test_incomplete_marker_on_rc_ge_2(self) -> None:
        """AC#1: rc >= 2 (suspect DB) → final marker state=incomplete."""
        with (
            patch.object(quality_gate, "_testmon_available", return_value=True),
            patch.object(quality_gate, "_build_seed_testmon_cmd", return_value=["SEED_CMD"]),
            patch.object(quality_gate, "_stream_pytest", return_value={"returncode": 2}),
        ):
            assert quality_gate.seed_testmon() == 0
        marker = self._marker()
        assert marker["state"] == "incomplete"
        assert marker["returncode"] == 2

    def test_skipped_marker_and_no_running_when_tooling_missing(self) -> None:
        """AC#2: not-importable writes a `skipped` marker (with reason) and
        returns 3; NO `running` marker is written on that path."""
        with patch.object(quality_gate, "_testmon_available", return_value=False):
            assert quality_gate.seed_testmon() == 3
        marker = self._marker()
        assert marker["state"] == "skipped"
        assert "not importable" in marker["reason"]
        assert "pid" not in marker  # never reached the `running` write

    def test_tooling_missing_returns_3(self) -> None:
        # review-fix S2: gated on the testmon-only probe, NOT the full impacted set.
        with patch.object(quality_gate, "_testmon_available", return_value=False):
            assert quality_gate.seed_testmon() == 3

    def test_seed_not_blocked_when_only_diff_cover_missing(self) -> None:
        """review-fix S2: seeding never calls diff-cover, so a missing diff-cover must not block it.

        review-fix MF1 (#04): stub `_build_seed_testmon_cmd` so this unit test
        never reaches the real `VENV_PYTHON` xdist probe (absent on CI).
        """
        with (
            patch.object(quality_gate, "_testmon_available", return_value=True),
            patch.object(quality_gate, "_impacted_tooling_available", return_value=False) as mock_full,
            patch.object(quality_gate, "_build_seed_testmon_cmd", return_value=["SEED_CMD"]),
            patch.object(quality_gate, "_stream_pytest", return_value={"passed": True, "returncode": 0}),
        ):
            assert quality_gate.seed_testmon() == 0
        mock_full.assert_not_called()  # the full (diff-cover-inclusive) probe is never consulted

    def test_seed_rebuilds_baseline_before_select_all(self) -> None:
        """QS-283 A3 (AC#3): `seed_testmon` calls `_rebuild_testmon_baseline`
        (purge + coverage reset + shard clear) BEFORE the select-all pytest
        pass, so a reseed against an advanced baseline still fully
        re-fingerprints (no "0 changed" dead end)."""
        order: list[str] = []
        with (
            patch.object(quality_gate, "_testmon_available", return_value=True),
            patch.object(
                quality_gate, "_rebuild_testmon_baseline", side_effect=lambda: order.append("rebuild")
            ) as mock_rebuild,
            patch.object(quality_gate, "_build_seed_testmon_cmd", return_value=["SEED_CMD"]),
            patch.object(
                quality_gate,
                "_stream_pytest",
                side_effect=lambda *_a, **_k: (order.append("pytest"), {"passed": True, "returncode": 0})[1],
            ),
        ):
            assert quality_gate.seed_testmon() == 0
        mock_rebuild.assert_called_once()
        assert order == ["rebuild", "pytest"], "rebuild must run before the select-all pass"

    def test_success_returns_0_regardless_of_test_outcome(self) -> None:
        # A failing test (rc=1) still updates the DB → seed returns 0, no warning.
        with (
            patch.object(quality_gate, "_testmon_available", return_value=True),
            patch.object(quality_gate, "_build_seed_testmon_cmd", return_value=["SEED_CMD"]),
            patch.object(
                quality_gate, "_stream_pytest", return_value={"passed": False, "returncode": 1}
            ) as mock_stream,
        ):
            assert quality_gate.seed_testmon() == 0
        assert mock_stream.call_args.args[0] == ["SEED_CMD"]

    def test_collection_crash_warns_but_stays_best_effort(self, capsys: pytest.CaptureFixture[str]) -> None:
        """review-fix NH4 (#03): a pytest exit >=2 (collection error/crash) warns but still returns 0."""
        with (
            patch.object(quality_gate, "_testmon_available", return_value=True),
            patch.object(quality_gate, "_build_seed_testmon_cmd", return_value=["SEED_CMD"]),
            patch.object(quality_gate, "_stream_pytest", return_value={"passed": False, "returncode": 2}),
        ):
            assert quality_gate.seed_testmon() == 0  # best-effort: never fatal
        err = capsys.readouterr().err
        assert "exited 2" in err
        assert ".testmondata may be incomplete" in err

    def test_seed_cmd_parallelizes_when_xdist_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """review-fix: seeding is the heaviest testmon pass, so it runs under -n auto."""
        monkeypatch.setattr(quality_gate, "_TESTMON_SUPPORTS_XDIST", True)
        with patch.object(quality_gate, "_pytest_workers", return_value="auto"):
            cmd = quality_gate._build_seed_testmon_cmd()
        assert cmd[:5] == [quality_gate.VENV_PYTHON, "-m", "pytest", "--testmon", "-q"]
        assert cmd[cmd.index("-n") + 1] == "auto"

    def test_seed_cmd_serial_when_xdist_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(quality_gate, "_TESTMON_SUPPORTS_XDIST", False)
        cmd = quality_gate._build_seed_testmon_cmd()
        assert cmd == [quality_gate.VENV_PYTHON, "-m", "pytest", "--testmon", "-q"]

    def test_seed_cmd_serial_when_workers_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(quality_gate, "_TESTMON_SUPPORTS_XDIST", True)
        with patch.object(quality_gate, "_pytest_workers", return_value=None):
            cmd = quality_gate._build_seed_testmon_cmd()
        assert "-n" not in cmd


class TestWriteSeedStatus:
    """QS-286 AC#3: `_write_seed_status` is atomic + best-effort."""

    @pytest.fixture(autouse=True)
    def _redirect(self, tmp_path: Path):
        with patch.object(quality_gate, "SEED_STATUS", tmp_path / ".testmondata.seed-status"):
            yield

    def test_writes_state_and_fields_no_temp_leftover(self) -> None:
        quality_gate._write_seed_status("running", pid=7, started=1.5)
        marker = json.loads(quality_gate.SEED_STATUS.read_text())
        assert marker == {"state": "running", "pid": 7, "started": 1.5}
        tmp = quality_gate.SEED_STATUS.with_suffix(quality_gate.SEED_STATUS.suffix + ".tmp")
        assert not tmp.exists(), "temp sibling must be cleaned up after a successful write"

    def test_uses_atomic_os_replace(self) -> None:
        with patch.object(quality_gate.os, "replace") as mock_replace:
            quality_gate._write_seed_status("ok", returncode=0)
        mock_replace.assert_called_once()

    def test_best_effort_swallows_and_cleans_up_on_failure(self) -> None:
        """A write failure must NOT raise (never abort the detached rebuild),
        and the temp file must still be unlinked by the `finally`."""

        def raiser(*_a, **_k):
            raise OSError("boom")

        with patch.object(quality_gate.os, "replace", side_effect=raiser):
            quality_gate._write_seed_status("ok", returncode=0)  # must not raise
        tmp = quality_gate.SEED_STATUS.with_suffix(quality_gate.SEED_STATUS.suffix + ".tmp")
        assert not tmp.exists()
        assert not quality_gate.SEED_STATUS.exists()  # replace never happened


class TestPidAlive:
    """QS-286: `_pid_alive` maps os.kill(pid, 0) outcomes to liveness."""

    def test_dead_when_process_lookup_error(self) -> None:
        with patch.object(quality_gate.os, "kill", side_effect=ProcessLookupError):
            assert quality_gate._pid_alive(999999) is False

    def test_alive_when_permission_error(self) -> None:
        with patch.object(quality_gate.os, "kill", side_effect=PermissionError):
            assert quality_gate._pid_alive(1) is True

    def test_alive_on_success(self) -> None:
        with patch.object(quality_gate.os, "kill", return_value=None) as mock_kill:
            assert quality_gate._pid_alive(1234) is True
        mock_kill.assert_called_once_with(1234, 0)


class TestSeedTestmonStatus:
    """QS-286 AC#4: `seed_testmon_status` — distinct message + 4-code exit
    for every originating marker condition. Read-only."""

    @pytest.fixture(autouse=True)
    def _redirect(self, tmp_path: Path):
        with patch.object(quality_gate, "SEED_STATUS", tmp_path / ".testmondata.seed-status"):
            yield

    def _write(self, marker: object) -> None:
        quality_gate.SEED_STATUS.write_text(json.dumps(marker))

    def test_ok_exits_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        self._write({"state": "ok", "finished": 100.0})
        assert quality_gate.seed_testmon_status() == 0
        assert "safe to close" in capsys.readouterr().out.lower()

    def test_running_alive_exits_4(self, capsys: pytest.CaptureFixture[str]) -> None:
        self._write({"state": "running", "pid": 42, "started": 1.0})
        with patch.object(quality_gate, "_pid_alive", return_value=True):
            assert quality_gate.seed_testmon_status() == 4
        assert "still running" in capsys.readouterr().out.lower()

    def test_running_dead_exits_1_interrupted(self, capsys: pytest.CaptureFixture[str]) -> None:
        self._write({"state": "running", "pid": 42, "started": 1.0})
        with patch.object(quality_gate, "_pid_alive", return_value=False):
            assert quality_gate.seed_testmon_status() == 1
        assert "interrupted" in capsys.readouterr().out.lower()

    def test_incomplete_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        self._write({"state": "incomplete", "returncode": 2})
        assert quality_gate.seed_testmon_status() == 1
        assert "finished with errors" in capsys.readouterr().out.lower()

    def test_skipped_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        self._write({"state": "skipped", "reason": "pytest-testmon not importable"})
        assert quality_gate.seed_testmon_status() == 1
        out = capsys.readouterr().out.lower()
        assert "skipped" in out and "no baseline was written" in out

    def test_missing_marker_exits_3(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert not quality_gate.SEED_STATUS.exists()
        assert quality_gate.seed_testmon_status() == 3
        assert "no baseline refresh" in capsys.readouterr().out.lower()

    def test_unparseable_marker_exits_3(self, capsys: pytest.CaptureFixture[str]) -> None:
        quality_gate.SEED_STATUS.write_text("{not json")
        assert quality_gate.seed_testmon_status() == 3
        assert "unreadable" in capsys.readouterr().out.lower()

    def test_unknown_state_exits_3(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A parseable marker with an unexpected state is treated as unreadable."""
        self._write({"state": "bogus"})
        assert quality_gate.seed_testmon_status() == 3
        assert "unreadable" in capsys.readouterr().out.lower()

    # review-fix #01 must-fix: malformed-but-parseable markers must route to
    # the unreadable→3 path, never crash the read-only status command.
    @pytest.mark.parametrize(
        "payload",
        [5, "x", None, [1], 3.14],
        ids=["int", "str", "null", "array", "float"],
    )
    def test_non_dict_payload_exits_3(
        self, payload: object, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._write(payload)
        assert quality_gate.seed_testmon_status() == 3  # no AttributeError
        assert "unreadable" in capsys.readouterr().out.lower()

    @pytest.mark.parametrize(
        "marker",
        [
            {"state": "running"},
            {"state": "running", "pid": None},
            {"state": "running", "pid": "x"},
            {"state": "running", "pid": 1.5},
        ],
        ids=["no-pid", "pid-null", "pid-str", "pid-float"],
    )
    def test_running_with_bad_pid_exits_3(
        self, marker: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A `running` marker without an int pid is unreadable — never reaches
        `_pid_alive` (which would `os.kill(None/str, 0)` → TypeError)."""
        self._write(marker)
        with patch.object(quality_gate, "_pid_alive") as mock_alive:
            assert quality_gate.seed_testmon_status() == 3  # no TypeError
        mock_alive.assert_not_called()  # bad pid never hits the syscall seam
        assert "unreadable" in capsys.readouterr().out.lower()

    # review-fix #01 nice-to-have: a marker missing a display-only field prints
    # a readable placeholder, not literal "None" — and keeps its exit code.
    def test_ok_missing_finished_prints_placeholder(self, capsys: pytest.CaptureFixture[str]) -> None:
        self._write({"state": "ok"})
        assert quality_gate.seed_testmon_status() == 0
        out = capsys.readouterr().out
        assert "None" not in out
        assert "an unknown time" in out

    def test_incomplete_missing_returncode_prints_placeholder(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._write({"state": "incomplete"})
        assert quality_gate.seed_testmon_status() == 1
        out = capsys.readouterr().out
        assert "None" not in out
        assert "exit unknown" in out

    def test_reader_is_read_only(self) -> None:
        """AC#4: no pytest / coverage / testmon import — the reader touches
        none of the heavy seams."""
        self._write({"state": "ok", "finished": 1.0})
        with (
            patch.object(quality_gate, "_stream_pytest") as mock_stream,
            patch.object(quality_gate, "_testmon_available") as mock_probe,
            patch.object(quality_gate, "_rebuild_testmon_baseline") as mock_rebuild,
        ):
            quality_gate.seed_testmon_status()
        mock_stream.assert_not_called()
        mock_probe.assert_not_called()
        mock_rebuild.assert_not_called()


class TestImpactedCli:
    """`main()` wiring: short-circuit, exit-code passthrough, mutex."""

    # review-fix NH3: this table covers the codes `check_impacted` itself
    # returns (0/1/3/4). Exit code 2 (usage/mutex error) is raised by
    # `parser.error` BEFORE `check_impacted` runs, so its dedicated rows
    # live in `test_impacted_mutex_exits_2` and `test_seed_testmon_mutex_exits_2`
    # — together they give the "dedicated test per exit-code row" guarantee.
    @pytest.mark.parametrize("exit_code", [0, 1, 3, 4], ids=["pass", "fail", "tooling", "no-base"])
    def test_impacted_exits_with_check_impacted_code(self, exit_code: int) -> None:
        with (
            patch("sys.argv", ["quality_gate.py", "--impacted"]),
            patch.object(quality_gate, "check_impacted", return_value=exit_code) as mock_check,
            pytest.raises(SystemExit) as exc,
        ):
            quality_gate.main()
        assert exc.value.code == exit_code
        mock_check.assert_called_once_with()

    def test_impacted_short_circuits_before_scope_and_cache(self) -> None:
        with (
            patch("sys.argv", ["quality_gate.py", "--impacted"]),
            patch.object(quality_gate, "check_impacted", return_value=0),
            patch.object(quality_gate, "_detect_scope") as mock_scope,
            patch.object(quality_gate, "_read_cache") as mock_cache,
            patch.object(quality_gate, "_get_changed_files") as mock_changed,
            pytest.raises(SystemExit),
        ):
            quality_gate.main()
        for m in (mock_scope, mock_cache, mock_changed):
            m.assert_not_called()

    @pytest.mark.parametrize(
        "conflict",
        [["--cache"], ["--no-cache"], ["--full"], ["--fix"], ["--quick", "tests/test_x.py"]],
        ids=["cache", "no-cache", "full", "fix", "quick"],
    )
    def test_impacted_mutex_exits_2(self, conflict: list[str], capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("sys.argv", ["quality_gate.py", "--impacted", *conflict]),
            pytest.raises(SystemExit) as exc,
        ):
            quality_gate.main()
        assert exc.value.code == 2
        assert "you cannot combine --impacted with" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "conflict",
        [
            ["--impacted"],
            ["--cache"],
            ["--no-cache"],
            ["--full"],
            ["--fix"],
            ["--quick", "tests/test_x.py"],
        ],
        ids=["impacted", "cache", "no-cache", "full", "fix", "quick"],
    )
    def test_seed_testmon_mutex_exits_2(
        self, conflict: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """review-fix M1: --seed-testmon combined with any execution mode is a usage error."""
        with (
            patch("sys.argv", ["quality_gate.py", "--seed-testmon", *conflict]),
            patch.object(quality_gate, "seed_testmon") as mock_seed,
            patch.object(quality_gate, "check_impacted") as mock_impacted,
            pytest.raises(SystemExit) as exc,
        ):
            quality_gate.main()
        assert exc.value.code == 2
        assert "you cannot combine --seed-testmon with" in capsys.readouterr().err
        # The conflicting request must NOT silently execute either mode.
        mock_seed.assert_not_called()
        mock_impacted.assert_not_called()

    @pytest.mark.parametrize("seed_code", [0, 3], ids=["ok", "tooling-missing"])
    def test_seed_testmon_cli_passthrough(self, seed_code: int) -> None:
        with (
            patch("sys.argv", ["quality_gate.py", "--seed-testmon"]),
            patch.object(quality_gate, "seed_testmon", return_value=seed_code) as mock_seed,
            patch.object(quality_gate, "_detect_scope") as mock_scope,
            pytest.raises(SystemExit) as exc,
        ):
            quality_gate.main()
        assert exc.value.code == seed_code
        mock_seed.assert_called_once_with()
        mock_scope.assert_not_called()

    @pytest.mark.parametrize(
        "conflict",
        [
            ["--seed-testmon"],
            ["--impacted"],
            ["--cache"],
            ["--no-cache"],
            ["--full"],
            ["--fix"],
            ["--quick", "tests/test_x.py"],
        ],
        ids=["seed", "impacted", "cache", "no-cache", "full", "fix", "quick"],
    )
    def test_seed_testmon_status_mutex_exits_2(
        self, conflict: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """AC#5: --seed-testmon-status combined with any mode is a usage error."""
        with (
            patch("sys.argv", ["quality_gate.py", "--seed-testmon-status", *conflict]),
            patch.object(quality_gate, "seed_testmon_status") as mock_status,
            patch.object(quality_gate, "seed_testmon") as mock_seed,
            pytest.raises(SystemExit) as exc,
        ):
            quality_gate.main()
        assert exc.value.code == 2
        assert "you cannot combine --seed-testmon-status with" in capsys.readouterr().err
        mock_status.assert_not_called()
        mock_seed.assert_not_called()

    @pytest.mark.parametrize("status_code", [0, 1, 3, 4], ids=["ok", "rerun", "no-status", "running"])
    def test_seed_testmon_status_cli_passthrough(self, status_code: int) -> None:
        """AC#4: --seed-testmon-status short-circuits before scope, returning
        the reader's exit code verbatim."""
        with (
            patch("sys.argv", ["quality_gate.py", "--seed-testmon-status"]),
            patch.object(quality_gate, "seed_testmon_status", return_value=status_code) as mock_status,
            patch.object(quality_gate, "_detect_scope") as mock_scope,
            pytest.raises(SystemExit) as exc,
        ):
            quality_gate.main()
        assert exc.value.code == status_code
        mock_status.assert_called_once_with()
        mock_scope.assert_not_called()


class TestImpactedDeps:
    """Regression guards for requirements_test.txt + .gitignore (AC#1)."""

    def test_testmon_and_diff_cover_pinned(self) -> None:
        reqs = (Path(__file__).resolve().parent.parent / "requirements_test.txt").read_text()
        assert "pytest-testmon==" in reqs
        assert "diff-cover==" in reqs

    def test_testmondata_gitignored(self) -> None:
        gi = (Path(__file__).resolve().parent.parent / ".gitignore").read_text()
        assert any(line.strip() == ".testmondata" for line in gi.splitlines())

    def test_seed_status_gitignored(self) -> None:
        """QS-286 AC#8: exact `.testmondata.seed-status` line (matches the
        exact-match `.testmondata` convention, not a glob)."""
        gi = (Path(__file__).resolve().parent.parent / ".gitignore").read_text()
        assert any(line.strip() == ".testmondata.seed-status" for line in gi.splitlines())


class TestProjectRulesDocGuards:
    """review-fix N2: content guard for the AC#12 doc edits (not just drift-checker)."""

    def _rules(self) -> str:
        return (
            Path(__file__).resolve().parent.parent / "docs" / "workflow" / "project-rules.md"
        ).read_text()

    def test_seed_testmon_carveout_heading_present(self) -> None:
        rules = self._rules()
        assert "Carve-out — `--seed-testmon`" in rules
        assert "single pytest owner" in rules  # the carve-out's rationale

    def test_cache_quick_impacted_reconciliation_present(self) -> None:
        rules = self._rules()
        assert "Local-vs-CI coverage invariant" in rules
        assert "`--impacted` is mutually exclusive" in rules

    def test_seed_testmon_status_command_reference_present(self) -> None:
        """review-fix #01 (AC#9): pin the --seed-testmon-status command-reference
        addition, mirroring test_seed_status_gitignored's exact-line guard."""
        rules = self._rules()
        assert "quality_gate.py --seed-testmon-status" in rules
        assert "companion" in rules  # documented as the --seed-testmon companion


class TestWorktreeSetupSeedsCaches:
    """AC#8: worktree-setup.sh copies (never symlinks) .testmondata + .mypy_cache."""

    def _script(self) -> str:
        return (Path(__file__).resolve().parent.parent / "scripts" / "worktree-setup.sh").read_text()

    def _seed_block(self) -> str:
        """Return only the QS-276 cache-seeding block (review-fix N1).

        Scoping the symlink-rejection assertions to this block is essential:
        the rest of the script legitimately uses `ln -s` for config /
        custom_components links.
        """
        body = self._script()
        start = body.index("# QS-276: seed cold-start caches")
        end = body.index("# QS-276 end: cache seeding")
        return body[start:end]

    def test_copies_both_caches(self) -> None:
        block = self._seed_block()
        assert ".mypy_cache" in block and ".testmondata" in block
        assert "cp -R" in block, "caches must be copied, not symlinked"

    def test_loop_enumerates_each_cache_explicitly(self) -> None:
        """review-fix NH5: both caches are handled by the same copy loop, not one-off.

        Asserts the loop header names BOTH caches, so the regression can't
        pass with only one cache genuinely copied and the other appearing
        solely in a warning line.
        """
        block = self._seed_block()
        assert "for cache in .mypy_cache .testmondata; do" in block
        # The copy is keyed on the loop variable (applies to every cache),
        # not hard-coded for a single cache name.
        assert 'cp -R "$src" "$dst"' in block

    def test_documents_file_vs_dir_cp(self) -> None:
        """review-fix NH3: the block notes that cp -R handles both a file and a directory."""
        block = self._seed_block()
        assert "directory (.mypy_cache)" in block and "single file (.testmondata)" in block

    def test_seeding_never_symlinks(self) -> None:
        """review-fix N1: reject any symlink in the seeding block (copy, not link)."""
        block = self._seed_block()
        assert "ln -s" not in block and "ln -sf" not in block

    def test_copy_is_error_guarded(self) -> None:
        """review-fix S4: a failed copy must clean up the partial result, not wedge the cache."""
        block = self._seed_block()
        assert "rm -rf" in block
        assert "failed to copy" in block

    def test_existing_dst_is_refreshed_not_silently_skipped(self) -> None:
        """review-fix S4: a pre-existing (possibly truncated) cache is refreshed, not skipped silently."""
        block = self._seed_block()
        assert "already present" in block

    def test_absent_remediation_is_cache_specific_and_runnable(self) -> None:
        """review-fix S4: the cache-miss hint cites a real command, --seed-testmon for the DB."""
        block = self._seed_block()
        assert "--seed-testmon" in block  # .testmondata remediation is executable

    def test_warns_when_absent(self) -> None:
        assert "Warning:" in self._script() and "absent in main worktree" in self._script()


class TestFinishTaskRefreshesBaseline:
    """AC#9: all three finish-task harness copies refresh via --seed-testmon."""

    @pytest.mark.parametrize("harness", [".claude", ".cursor", ".opencode"])
    def test_seed_testmon_refresh_present(self, harness: str) -> None:
        body = (Path(__file__).resolve().parent.parent / harness / "agents" / "qs-finish-task.md").read_text()
        assert "--seed-testmon" in body
        assert "git worktree list --porcelain" in body  # MAIN_DIR captured before cleanup
        assert "nohup" in body  # detached / best-effort

    @pytest.mark.parametrize("harness", [".claude", ".cursor", ".opencode"])
    def test_completion_signal_present(self, harness: str) -> None:
        """QS-286 AC#7: the detached refresh deletes the stale marker, logs to
        `.testmondata.seed.log`, and points the user at --seed-testmon-status."""
        body = (Path(__file__).resolve().parent.parent / harness / "agents" / "qs-finish-task.md").read_text()
        assert 'rm -f "$MAIN_DIR/.testmondata.seed-status"' in body  # stale-marker guard
        assert '>"$MAIN_DIR/.testmondata.seed.log" 2>&1' in body  # truncate redirect
        assert "--seed-testmon-status" in body  # how to check
        assert "safe to close this terminal" in body
        # the old silent seed redirect is gone (other >/dev/null uses remain)
        assert "--seed-testmon >/dev/null 2>&1" not in body

    def test_seed_launch_block_byte_identical_across_harnesses(self) -> None:
        """Harness-sync: the QS-286 completion-signal block is identical in
        all three finish-task copies."""
        blocks = []
        for harness in (".claude", ".cursor", ".opencode"):
            body = (
                Path(__file__).resolve().parent.parent / harness / "agents" / "qs-finish-task.md"
            ).read_text()
            # review-fix #01 nice-to-have: anchor the slice on code-adjacent
            # markers (the stale-marker rm and the exact redirect line), not on
            # prose ending in a literal period, so added narrative can't
            # truncate the slice inconsistently.
            start = body.index('rm -f "$MAIN_DIR/.testmondata.seed-status"')
            redirect = '>"$MAIN_DIR/.testmondata.seed.log" 2>&1'
            end = body.index(redirect, start) + len(redirect)
            blocks.append(body[start:end])
        assert blocks[0] == blocks[1] == blocks[2]

    @pytest.mark.parametrize("harness", [".claude", ".cursor", ".opencode"])
    def test_interpreter_is_probed_not_hardcoded(self, harness: str) -> None:
        """review-fix S3: probe for a usable interpreter; warn instead of a false success if none."""
        body = (Path(__file__).resolve().parent.parent / harness / "agents" / "qs-finish-task.md").read_text()
        assert "command -v python3" in body or "command -v python" in body
        assert "no usable Python interpreter" in body


class TestImplementAgentsDefaultImpacted:
    """AC#10: implement agents default to --impacted; review-task untouched."""

    @pytest.mark.parametrize("harness", [".claude", ".cursor", ".opencode"])
    @pytest.mark.parametrize("agent", ["qs-implement-task", "qs-implement-setup-task"])
    def test_implement_agents_use_impacted(self, harness: str, agent: str) -> None:
        body = (Path(__file__).resolve().parent.parent / harness / "agents" / f"{agent}.md").read_text()
        assert "quality_gate.py --impacted" in body

    @pytest.mark.parametrize("harness", [".claude", ".cursor", ".opencode"])
    @pytest.mark.parametrize("agent", ["qs-implement-task", "qs-implement-setup-task"])
    def test_b1_all_six_agents_mandate_impacted(self, harness: str, agent: str) -> None:
        """QS-283 B1 (AC#6): all six implement agents mandate `--impacted`
        before commit/PR and forbid substituting the full gate locally."""
        body = (Path(__file__).resolve().parent.parent / harness / "agents" / f"{agent}.md").read_text()
        flat = " ".join(body.split())  # normalize markdown line-wrapping
        assert "**ALWAYS** run the impacted" in flat
        assert "Do **not** run, or substitute, the full gate locally" in flat

    @pytest.mark.parametrize("harness", [".claude", ".cursor", ".opencode"])
    def test_b2_b3_implement_task_closes_loophole(self, harness: str) -> None:
        """QS-283 B2/B3 (AC#6): the three `qs-implement-task.md` copies delete
        the unchanged-code escape clause (B2) and forbid the full-gate
        diagnostic escape (B3)."""
        body = (Path(__file__).resolve().parent.parent / harness / "agents" / "qs-implement-task.md").read_text()
        flat = " ".join(body.split())
        # B2: the "coverage lost in unchanged code" license must be gone.
        assert "suspect coverage lost" not in flat
        assert "CI's exclusive job" in flat
        # B3: the "fix autonomously and re-run" nudge to the full gate is gone.
        assert "fix autonomously and re-run" not in flat
        assert "never switch to the full gate to diagnose" in flat

    @pytest.mark.parametrize("harness", [".claude", ".cursor", ".opencode"])
    def test_implement_task_intro_names_impacted_not_full_gate(self, harness: str) -> None:
        """Review fix #03: the intro summary line and frontmatter description
        must NOT instruct running the full gate locally (the QS-283 regression
        class) — they name the impacted gate as the inner-loop command."""
        body = (Path(__file__).resolve().parent.parent / harness / "agents" / "qs-implement-task.md").read_text()
        flat = " ".join(body.split())
        # The self-contradictory stale phrasing must never reappear.
        assert "run the full quality gate, and open a PR" not in flat
        assert "must pass the full quality gate" not in flat
        # The intro/description names the impacted gate instead.
        assert "impacted quality gate" in flat

    @pytest.mark.parametrize("harness", [".claude", ".cursor", ".opencode"])
    def test_review_task_untouched_by_impacted(self, harness: str) -> None:
        body = (Path(__file__).resolve().parent.parent / harness / "agents" / "qs-review-task.md").read_text()
        assert "--impacted" not in body


def _run_testmon(repo: Path, *, cov: bool = False, xml: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run `pytest --testmon` in an isolated subprocess inside `repo`.

    Plugin autoload is disabled so the host's pytest-homeassistant /
    asyncio plugin stack can't crash collection in the throwaway repo;
    testmon (and pytest-cov when measuring) are loaded explicitly.
    cacheprovider is a pytest builtin and stays loaded.
    """
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    cmd = [quality_gate.VENV_PYTHON, "-m", "pytest", "--testmon", "-q", "-p", "testmon.pytest_testmon"]
    if cov:
        assert xml is not None
        cmd += ["-p", "pytest_cov", "--cov=pkg", "--cov-report=", f"--cov-report=xml:{xml}"]
    return subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, env=env)


def _selected_count(result: subprocess.CompletedProcess[str]) -> int:
    """Parse the number of tests testmon actually ran from pytest output."""
    out = result.stdout
    if "no tests ran" in out:
        return 0
    m = re.search(r"(\d+) passed", out)
    return int(m.group(1)) if m else -1


@pytest.mark.integration
class TestImpactedIntegrationRealTestmon:
    """AC#5/#6/#7: genuine testmon block-fingerprinting in a throwaway repo.

    These exercise REAL pytest-testmon + diff-cover (no mocks) to prove
    the correctness basis of `--impacted`: a changed line is "covered"
    iff a selected test ran it, and testmon selects a superset of the
    tests that cover the diff.
    """

    @pytest.fixture
    def repo(self, tmp_path: Path) -> Path:
        if not quality_gate._impacted_tooling_available():
            pytest.skip("pytest-testmon / diff-cover not importable")
        repo = tmp_path / "repo"
        (repo / "pkg").mkdir(parents=True)
        (repo / "tests").mkdir()
        (repo / "pkg" / "__init__.py").write_text("")
        (repo / "pkg" / "calc.py").write_text("X = 1\n\n\ndef add(a, b):\n    return a + b\n")
        (repo / "pkg" / "isolated_const.py").write_text("UNUSED = 1\n")
        (repo / "tests" / "test_calc.py").write_text(
            "from pkg.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        )
        for args in (
            ["init", "-q"],
            ["config", "user.email", "t@t.co"],
            ["config", "user.name", "t"],
            ["add", "-A"],
            ["commit", "-qm", "base"],
        ):
            subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True)
        # Seed the testmon baseline (selects all → records coverage).
        seed = _run_testmon(repo)
        assert _selected_count(seed) == 1, seed.stdout + seed.stderr
        return repo

    def test_noop_selects_zero(self, repo: Path) -> None:
        """Nothing changed since the seed → testmon selects zero tests."""
        assert _selected_count(_run_testmon(repo)) == 0

    def test_edit_to_uncovered_code_selects_zero(self, repo: Path) -> None:
        """AC#5: a new constant in a module no test exercises selects zero."""
        (repo / "pkg" / "isolated_const.py").write_text("UNUSED = 1\nNEW_CONST = 2\n")
        assert _selected_count(_run_testmon(repo)) == 0

    def test_edit_to_covered_code_reselects_its_test(self, repo: Path) -> None:
        """Superset property: editing a covered function reselects its test."""
        (repo / "pkg" / "calc.py").write_text("X = 1\n\n\ndef add(a, b):\n    return a + b + 0\n")
        assert _selected_count(_run_testmon(repo)) == 1

    def test_new_untested_function_fails_diff_cover(self, repo: Path) -> None:
        """AC#6: a new untested function → 0% on the new lines → diff-cover fails."""
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), check=True, capture_output=True, text=True
        ).stdout.strip()
        (repo / "pkg" / "calc.py").write_text(
            "X = 1\n\n\ndef add(a, b):\n    return a + b\n\n\ndef untested(z):\n    return z * 99\n"
        )
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-qm", "untested"], cwd=str(repo), check=True, capture_output=True, text=True)
        xml = repo / "coverage.xml"
        _run_testmon(repo, cov=True, xml=xml)
        assert xml.exists()
        dc = subprocess.run(
            [quality_gate._venv_tool("diff-cover"), str(xml), f"--compare-branch={base}", "--fail-under=100"],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )
        assert dc.returncode != 0, f"diff-cover should fail on the untested function:\n{dc.stdout}"

    def test_untracked_new_file_fails_only_with_include_untracked(self, repo: Path) -> None:
        """review-fix SF-A (#04): a brand-new UNTRACKED file with an uncovered function.

        Proves the dominant inner-loop case (new code starts untracked):
        without `--include-untracked` diff-cover scores a vacuous 100% PASS;
        with it (the argv `_build_diff_cover_cmd` now emits) it FAILs.
        """
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), check=True, capture_output=True, text=True
        ).stdout.strip()
        # New file, NEVER `git add`-ed, with an untested function.
        (repo / "pkg" / "untracked_mod.py").write_text("def untracked_fn(z):\n    return z * 99\n")
        xml = repo / "coverage.xml"
        _run_testmon(repo, cov=True, xml=xml)
        assert xml.exists()

        def _dc(*extra: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [quality_gate._venv_tool("diff-cover"), str(xml), f"--compare-branch={base}", *extra, "--fail-under=100"],
                cwd=str(repo),
                capture_output=True,
                text=True,
            )

        # Without the flag: untracked file ignored → vacuous PASS (the bug).
        assert _dc().returncode == 0
        # With the flag (what we now emit): the uncovered new lines FAIL.
        assert _dc("--include-untracked").returncode != 0

    def test_corrupt_db_selects_all(self, repo: Path) -> None:
        """AC#7: a corrupt .testmondata → fail-safe deletes it → select-all."""
        db = repo / ".testmondata"
        db.write_bytes(b"not a sqlite database")
        # Mirror check_impacted's fail-safe against the throwaway DB.
        with patch.object(quality_gate, "TESTMON_DATA", db):
            quality_gate._ensure_testmon_db_safe()
        assert not db.exists()
        # With the corrupt DB gone, testmon rebuilds and selects all tests.
        assert _selected_count(_run_testmon(repo)) == 1


@pytest.mark.integration
class TestImpactedSelfHealIntegration:
    """QS-283 AC#5: reproduce the killed-run testmon/coverage desync at the
    real `_run_testmon` + `.testmondata` level, then prove A4's recovery by
    calling `_rebuild_testmon_baseline` directly with the path constants
    patched to a throwaway repo — NOT by driving the full `check_impacted()`
    orchestrator (which would need the import-time `SRC_DIR`/`TESTS_DIR`
    repointed and plugin autoload disabled, neither expressible here).
    """

    def _feature_repo(self, tmp_path: Path) -> tuple[Path, str]:
        """A repo whose HEAD adds a COVERED `feature()` over a base commit.

        Returns `(repo, base_rev)` so diff-cover can score the feature's
        changed lines against the pre-feature base.
        """
        if not quality_gate._impacted_tooling_available():
            pytest.skip("pytest-testmon / diff-cover not importable")
        repo = tmp_path / "repo"
        (repo / "pkg").mkdir(parents=True)
        (repo / "tests").mkdir()
        (repo / "pkg" / "__init__.py").write_text("")
        (repo / "pkg" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
        (repo / "tests" / "test_calc.py").write_text(
            "from pkg.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        )
        for args in (
            ["init", "-q"],
            ["config", "user.email", "t@t.co"],
            ["config", "user.name", "t"],
            ["add", "-A"],
            ["commit", "-qm", "base"],
        ):
            subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True)
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), check=True, capture_output=True, text=True
        ).stdout.strip()
        # Add a COVERED feature + its test, then commit it.
        (repo / "pkg" / "calc.py").write_text(
            "def add(a, b):\n    return a + b\n\n\ndef feature(z):\n    return z * 2\n"
        )
        (repo / "tests" / "test_calc.py").write_text(
            "from pkg.calc import add, feature\n\n\n"
            "def test_add():\n    assert add(1, 2) == 3\n\n\n"
            "def test_feature():\n    assert feature(3) == 6\n"
        )
        for args in (["add", "-A"], ["commit", "-qm", "feature"]):
            subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True)
        return repo, base

    @staticmethod
    def _diff_cover(repo: Path, xml: Path, base: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                quality_gate._venv_tool("diff-cover"),
                str(xml),
                f"--compare-branch={base}",
                "--include-untracked",
                "--fail-under=100",
            ],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )

    def test_desync_deadlocks_then_self_heal_recovers(self, tmp_path: Path) -> None:
        repo, base = self._feature_repo(tmp_path)
        xml = repo / "coverage.xml"
        # Seed the warm baseline: testmon records both tests + their coverage.
        assert _selected_count(_run_testmon(repo)) == 2

        # DESYNC: testmon is warm (thinks nothing changed) so a re-run selects
        # ZERO tests — but the coverage measured by that 0-test run does NOT
        # cover the committed `feature()` lines. diff-cover vs base therefore
        # FAILs, and testmon refuses to reselect: the deadlock a killed run
        # leaves behind (advanced .testmondata, lost coverage).
        deadlock = _run_testmon(repo, cov=True, xml=xml)
        assert _selected_count(deadlock) == 0, "warm baseline must select zero"
        assert xml.exists()
        assert self._diff_cover(repo, xml, base).returncode != 0, (
            "the desync must FAIL diff-cover (changed feature lines uncovered)"
        )

        # RECOVERY: A4's shared helper — purge the testmon DB + clear coverage
        # — with the path constants patched to this repo. Then a re-run
        # select-alls, re-covers the feature, and diff-cover PASSes.
        with (
            patch.object(quality_gate, "TESTMON_DATA", repo / ".testmondata"),
            patch.object(quality_gate, "COVERAGE_DATA", repo / ".coverage"),
            patch.object(quality_gate, "COVERAGE_XML", xml),
        ):
            quality_gate._rebuild_testmon_baseline()
        recovered = _run_testmon(repo, cov=True, xml=xml)
        assert _selected_count(recovered) == 2, "purged baseline must select-all"
        assert self._diff_cover(repo, xml, base).returncode == 0, (
            "after the rebuild the feature lines are covered → diff-cover PASS"
        )

    def test_genuinely_untested_function_still_fails_after_rebuild(self, tmp_path: Path) -> None:
        """A real coverage gap is NOT masked by the recovery: a committed
        function with no covering test still FAILs diff-cover after a rebuild +
        select-all (no false PASS)."""
        repo, base = self._feature_repo(tmp_path)
        xml = repo / "coverage.xml"
        # Add a genuinely UNTESTED function on top of the feature and commit it.
        (repo / "pkg" / "calc.py").write_text(
            "def add(a, b):\n    return a + b\n\n\n"
            "def feature(z):\n    return z * 2\n\n\n"
            "def untested(q):\n    return q - 1\n"
        )
        for args in (["add", "-A"], ["commit", "-qm", "untested"]):
            subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True)
        with (
            patch.object(quality_gate, "TESTMON_DATA", repo / ".testmondata"),
            patch.object(quality_gate, "COVERAGE_DATA", repo / ".coverage"),
            patch.object(quality_gate, "COVERAGE_XML", xml),
        ):
            quality_gate._rebuild_testmon_baseline()
        assert _selected_count(_run_testmon(repo, cov=True, xml=xml)) == 2
        assert self._diff_cover(repo, xml, base).returncode != 0, (
            "a genuinely untested function must still FAIL after the rebuild"
        )


@pytest.mark.integration
class TestTestmonRelocationInvariant:
    """review-fix SF3: enforce the cross-worktree `.testmondata` relocation invariant.

    `worktree-setup.sh` COPIES `.testmondata` from the main worktree into
    a freshly-created one. The safety claim — "selects more, never fewer"
    — must be *enforced*, not just asserted in a comment: copying the DB
    across worktrees may never cause testmon to UNDER-select impacted
    tests. testmon keys on rootdir-relative paths + file-content
    checksums, so a relocated DB still reselects any file whose content
    differs from the baseline. This proves it with real testmon.
    """

    def _make_repo(self, root: Path, *, calc_body: str) -> Path:
        if not quality_gate._impacted_tooling_available():
            pytest.skip("pytest-testmon / diff-cover not importable")
        (root / "pkg").mkdir(parents=True)
        (root / "tests").mkdir()
        (root / "pkg" / "__init__.py").write_text("")
        (root / "pkg" / "calc.py").write_text(calc_body)
        (root / "tests" / "test_calc.py").write_text(
            "from pkg.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        )
        for args in (
            ["init", "-q"],
            ["config", "user.email", "t@t.co"],
            ["config", "user.name", "t"],
            ["add", "-A"],
            ["commit", "-qm", "base"],
        ):
            subprocess.run(["git", *args], cwd=str(root), check=True, capture_output=True, text=True)
        return root

    def test_relocated_db_with_changed_content_reselects_never_underselects(self, tmp_path: Path) -> None:
        base_body = "def add(a, b):\n    return a + b\n"
        # "main" worktree: seed the baseline against the original content.
        main = self._make_repo(tmp_path / "main", calc_body=base_body)
        seed = _run_testmon(main)
        assert _selected_count(seed) == 1, seed.stdout + seed.stderr
        # A noop re-run in the SAME repo selects zero (warm baseline).
        assert _selected_count(_run_testmon(main)) == 0

        # "worktree": identical repo, but the COVERED file has different
        # content than the seeded baseline. Relocate (copy) the DB in.
        work = self._make_repo(tmp_path / "work", calc_body="def add(a, b):\n    return a + b + 0\n")
        shutil.copy2(main / ".testmondata", work / ".testmondata")

        # Invariant: the relocated DB must NOT skip the test covering the
        # changed file — it reselects it (selects more, never fewer).
        selected = _selected_count(_run_testmon(work))
        assert selected == 1, (
            "relocated .testmondata under-selected a changed-content test "
            f"(got {selected}); the 'never fewer' invariant is violated"
        )
