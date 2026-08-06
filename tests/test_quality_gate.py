"""Tests for quality_gate.py caching functionality."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
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


_DEFAULT_EARLY_EXIT_PATHS = ["custom_components/quiet_solar/home_model/load.py"]

# QS-332: the REAL issue resolver, captured at import time — the autouse
# `_lane_check_isolated` fixture below patches the module attribute for
# every test, so `TestResolveLaneIssue` calls this reference instead.
_REAL_RESOLVE_LANE_ISSUE = quality_gate._resolve_lane_issue


@pytest.fixture(autouse=True)
def _lane_check_isolated(tmp_path_factory: pytest.TempPathFactory):
    """QS-332 existing-test audit: keep every test in this module
    network-free. The lane check `check_impacted()` and `main()` now run
    would otherwise resolve THIS repo's real `QS_<N>` branch and issue a
    real `gh issue view` (or write the real label cache). Stub the issue
    resolution to "not a task branch" (silent skip) by default; lane
    tests re-patch `_resolve_lane_issue` inside their own bodies (a
    nested `patch.object` cleanly overrides this one). The cache file is
    pointed at a tmp dir so no test touches the repo's real
    `.lane_check_cache`.
    """
    root = tmp_path_factory.mktemp("lane")
    with (
        patch.object(quality_gate, "_resolve_lane_issue", return_value=None),
        patch.object(quality_gate, "LANE_CACHE_FILE", root / ".lane_check_cache"),
    ):
        yield

# A real sentinel object, so the default is not a
# `str` masquerading as a `list[str] | None` behind a blanket `type: ignore`.
# `None` cannot be the default here — it is a MEANINGFUL value (git failed →
# unknown → no early exit) that several tests pass deliberately.
_KEEP_PY = object()


def _patch_early_exit(paths: list[str] | None | object = _KEEP_PY):
    """Patch QS-290's non-`.py` early-exit seam in `check_impacted`.

    The seam (`_impacted_early_exit_paths`) is deliberately zero-arg and total:
    ONE patch here silences EVERY git call at that seat. Every
    `check_impacted` test must patch it, because the default working-tree state
    is not controllable from a unit test — a genuinely non-`.py` tree would make
    the exit fire and turn assertions about the downstream pipeline into
    vacuous greens (or env-dependent failures, e.g. `test_no_base_in_ci_returns_4`
    returning 0).

    Default (the `_KEEP_PY` sentinel): a single `.py` path, so the exit provably
    does NOT fire.
    """
    resolved = _DEFAULT_EARLY_EXIT_PATHS if paths is _KEEP_PY else paths
    return patch.object(quality_gate, "_impacted_early_exit_paths", return_value=resolved)


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
    """Regression guard for .github/workflows/pr-quality.yml (QS-292 shape)."""

    @staticmethod
    def _load_workflow(filename: str = "pr-quality.yml") -> dict[str, Any]:
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        repo_root = Path(__file__).resolve().parent.parent
        wf_path = repo_root / ".github" / "workflows" / filename
        if not wf_path.exists():
            pytest.skip("workflow file missing")
        loaded: dict[str, Any] = yaml.safe_load(wf_path.read_text())
        return loaded

    def test_shard_job_uses_xdist_sysmon(self) -> None:
        """The shard job's pytest step uses -n auto and COVERAGE_CORE=sysmon.

        The step is selected BY NAME (`Run shard …`), not by "first step
        whose run contains pytest" — test-shard has two pytest steps and
        the collect-only one comes first and carries neither flag.
        """
        data = self._load_workflow()
        shard_job = data["jobs"]["test-shard"]
        run_steps = [s for s in shard_job["steps"] if s.get("name", "").startswith("Run shard")]
        assert run_steps, "no 'Run shard …' step found in test-shard job"
        pytest_step = run_steps[0]
        assert "-n auto" in pytest_step["run"] or "--numprocesses auto" in pytest_step["run"]
        env = pytest_step.get("env", {})
        assert env.get("COVERAGE_CORE") == "sysmon"

    def test_required_check_name_is_pinned(self) -> None:
        """The aggregation job's name is the LITERAL required status check.

        Branch protection on `main` requires exactly the string
        `Tests (100% Coverage)`; renaming it blocks every PR forever.
        """
        data = self._load_workflow()
        assert data["jobs"]["test"]["name"] == "Tests (100% Coverage)"

    def test_aggregation_runs_even_when_a_shard_fails(self) -> None:
        """`if: !cancelled()` + a needs.test-shard.result guard step.

        Without them a failing shard SKIPS the aggregation job, and
        branch protection treats a skipped required check as passing.
        `!cancelled()` gives identical anti-skip protection to
        `always()` WITHOUT turning a cancelled run into a hard red
        (review fix #01 S3). (`if` is a safe YAML key — unlike `on`, it
        is not a YAML 1.1 boolean.)
        """
        data = self._load_workflow()
        test_job = data["jobs"]["test"]
        condition = str(test_job["if"])
        assert "cancelled()" in condition and "!" in condition, (
            f"jobs.test `if` must be a !cancelled() guard, got {condition!r}"
        )
        assert "always()" not in condition, (
            "always() also fires on cancellation, turning a cancelled run into a hard red"
        )
        assert any("needs.test-shard.result" in str(s.get("if", "")) for s in test_job["steps"]), (
            "no step in jobs.test guards on needs.test-shard.result"
        )

    def test_translations_value_check_lives_in_required_job(self) -> None:
        """D-14: the translations check must sit INSIDE the required context.

        `Tests (100% Coverage)` is the only required status check on
        `main`, so a value-stale en.json is blocked only if the check
        runs in that job. A future edit could delete the step and every
        other gate would stay green (review fix #01 S7).
        """
        data = self._load_workflow()
        steps = data["jobs"]["test"]["steps"]
        gen_steps = [s for s in steps if "generate-translations.sh" in s.get("run", "")]
        assert gen_steps, "no generate-translations.sh step in the required `test` job"
        # The check is only meaningful if the regenerated output is then
        # compared — a bare generator call always exits 0.
        assert any("git diff" in s.get("run", "") for s in gen_steps), (
            "generate-translations.sh runs but its output is never diffed"
        )

    def test_split_counts_agree(self) -> None:
        """All four hard-coded shard counts agree.

        len(matrix.shard), the `--splits N` in the shard pytest command,
        the `--splits N` passed to ci_reconcile_shards.py, and the `/N`
        suffix in the shard job's display name (review fix #01 S10a —
        moving to 6 shards would otherwise leave job names reading `/4`).
        """
        data = self._load_workflow()
        shard_job = data["jobs"]["test-shard"]
        matrix_len = len(shard_job["strategy"]["matrix"]["shard"])

        run_steps = [s for s in shard_job["steps"] if s.get("name", "").startswith("Run shard")]
        assert run_steps
        match = re.search(r"--splits (\d+)", run_steps[0]["run"])
        assert match, "no --splits flag in the shard pytest command"
        pytest_splits = int(match.group(1))

        reconcile_steps = [s for s in data["jobs"]["test"]["steps"] if "ci_reconcile_shards.py" in s.get("run", "")]
        assert reconcile_steps, "no ci_reconcile_shards.py step in jobs.test"
        match = re.search(r"--splits (\d+)", reconcile_steps[0]["run"])
        assert match, "no --splits flag in the reconcile command"
        reconcile_splits = int(match.group(1))

        match = re.search(r"/(\d+)\s*$", shard_job["name"])
        assert match, f"shard job name has no /N suffix: {shard_job['name']!r}"
        name_splits = int(match.group(1))

        assert matrix_len == pytest_splits == reconcile_splits == name_splits

    def test_coverage_pin_matches_requirements(self) -> None:
        """The aggregation job's `coverage==` pin twins requirements_test.txt.

        The shards' data must be written and read by the same coverage
        version; bumping one side only silently breaks that invariant
        (review fix #01 S10b).
        """
        data = self._load_workflow()
        wf_pins = {
            m.group(1)
            for s in data["jobs"]["test"]["steps"]
            if (m := re.search(r"coverage==([\d.]+)", s.get("run", "")))
        }
        assert len(wf_pins) == 1, f"expected exactly one coverage== pin, got {wf_pins}"

        repo_root = Path(__file__).resolve().parent.parent
        reqs = (repo_root / "requirements_test.txt").read_text()
        match = re.search(r"^coverage==([\d.]+)", reqs, re.MULTILINE)
        assert match, "no top-level coverage== pin in requirements_test.txt"
        assert wf_pins == {match.group(1)}, (
            f"workflow pins coverage {wf_pins}, requirements_test.txt pins {match.group(1)}"
        )

    def test_shard_and_aggregation_jobs_are_hardened(self) -> None:
        """Both new jobs drop credentials and declare least-privilege perms.

        Each runs `pip install` from third-party packages; leaving the
        git credential store populated hands a compromised dependency a
        usable token (review fix #01 S4, zizmor `artipacked` /
        `excessive-permissions`).
        """
        data = self._load_workflow()
        for job_name in ("test-shard", "test"):
            job = data["jobs"][job_name]
            assert job.get("permissions") == {"contents": "read"}, (
                f"jobs.{job_name} lacks `permissions: contents: read`"
            )
            checkouts = [s for s in job["steps"] if "actions/checkout" in str(s.get("uses", ""))]
            assert checkouts, f"jobs.{job_name} has no checkout step"
            for step in checkouts:
                assert step.get("with", {}).get("persist-credentials") is False, (
                    f"jobs.{job_name} checkout must set persist-credentials: false"
                )

    def test_release_workflow_is_aligned(self) -> None:
        """AC-7 / D-15: release.yml matches pr-quality's pytest invocation.

        Nothing else in `tests/` references release.yml, so the exact
        drift D-15 documents could silently recur (review fix #01 S8).
        """
        data = self._load_workflow("release.yml")
        steps = data["jobs"]["quality-gate"]["steps"]

        pytest_steps = [s for s in steps if "pytest" in s.get("run", "")]
        assert pytest_steps, "no pytest step in release.yml quality-gate job"
        step = pytest_steps[0]
        assert "-n auto" in step["run"] or "--numprocesses auto" in step["run"]
        assert step.get("env", {}).get("COVERAGE_CORE") == "sysmon"

        setup_steps = [s for s in steps if "actions/setup-python" in str(s.get("uses", ""))]
        assert setup_steps, "no setup-python step in release.yml"
        assert setup_steps[0].get("with", {}).get("cache") == "pip"


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
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`--quick` prints a `[quick] running ...` banner to stderr.

        The xdist half of the banner is environment-dependent (see the two
        tests below), so this pins the environment explicitly rather than
        inheriting it. Without that, the test passes locally — where the venv
        has xdist — and fails on CI, which has no venv at all and so correctly
        prints `single-process`.
        """
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        result = {"name": "pytest", "passed": True, "detail": ""}
        with (
            patch(
                "sys.argv",
                ["quality_gate.py", "--quick", "tests/test_foo.py", "tests/ha_tests"],
            ),
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate, "check_pytest_files", return_value=result),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()
        err = capsys.readouterr().err
        assert err.startswith("[quick] running "), f"banner missing/wrong: {err!r}"
        # QS-290 (S-1): xdist is now conditional on the small-run threshold, so
        # the banner must not promise it unconditionally.
        assert "xdist + sysmon" not in err, f"banner still claims unconditional xdist: {err!r}"
        assert f"xdist above {quality_gate._SERIAL_MAX_TESTS} tests" in err, f"banner missing threshold: {err!r}"

    def test_quick_banner_does_not_promise_xdist_when_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The banner printed "xdist above N tests" unconditionally, so it still
        promised xdist when xdist is absent or `QS_QG_PYTEST_WORKERS=0` — the same
        over-claim this change removes from the "xdist + sysmon" wording."""
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        result = {"name": "pytest", "passed": True, "detail": ""}
        with (
            patch("sys.argv", ["quality_gate.py", "--quick", "tests/test_foo.py"]),
            patch.object(quality_gate, "_has_xdist", return_value=False),
            patch.object(quality_gate, "check_pytest_files", return_value=result),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()
        err = capsys.readouterr().err
        assert err.startswith("[quick] running "), err
        assert "xdist" not in err, err
        assert "single-process" in err, err

    def test_quick_banner_does_not_promise_xdist_when_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("QS_QG_PYTEST_WORKERS", "0")
        result = {"name": "pytest", "passed": True, "detail": ""}
        with (
            patch("sys.argv", ["quality_gate.py", "--quick", "tests/test_foo.py"]),
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate, "check_pytest_files", return_value=result),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()
        assert "xdist" not in capsys.readouterr().err

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
        """`check_pytest_files` adds `-n <workers>` iff `_pytest_workers()` returns one.

        QS-290: the collected count is pinned ABOVE `_SERIAL_MAX_TESTS` so the
        serial fast path cannot demote the `auto`/`4` params — this test is
        about the resolver's value reaching the argv, not the threshold.
        """
        captured: dict = {}

        def fake_stream(cmd: list[str], total_tests: int | None = None) -> dict:
            captured["cmd"] = cmd
            captured["total_tests"] = total_tests
            return {"name": "pytest", "passed": True, "detail": ""}

        with (
            patch.object(quality_gate, "_pytest_workers", return_value=workers_value),
            patch.object(
                quality_gate,
                "_collect_test_count",
                return_value=quality_gate._SERIAL_MAX_TESTS + 1,
            ),
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


# --- QS-290 S-1: serial fast path for small test-file runs ---


class _CountingFakePopen:
    """A `subprocess.Popen` stand-in that records every spawned argv.

    `collect_stdout` is what the `--collect-only` probe reads via
    `communicate()`; `run_stdout` is streamed line-by-line as the main
    pytest run's stdout. `returncode` is what the main run exits with.
    """

    calls: list[list[str]] = []
    collect_stdout = "0 tests collected\n"
    collect_returncode = 0
    run_stdout = ""
    run_returncode = 0

    def __init__(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
        type(self).calls.append(list(cmd))
        self._is_collect = "--collect-only" in cmd
        text = type(self).collect_stdout if self._is_collect else type(self).run_stdout
        self.stdout = io.StringIO(text)
        self.stderr = io.StringIO("")
        self.returncode = (
            type(self).collect_returncode if self._is_collect else type(self).run_returncode
        )

    def communicate(self, *a, **kw):  # type: ignore[no-untyped-def]
        return (self.stdout.getvalue(), "")

    def wait(self):  # type: ignore[no-untyped-def]
        return self.returncode


def _fake_popen(
    *,
    collect_stdout: str = "0 tests collected\n",
    collect_returncode: int = 0,
    run_stdout: str = "",
    run_returncode: int = 0,
) -> type[_CountingFakePopen]:
    """Build a fresh `_CountingFakePopen` subclass with its own `calls` list."""
    return type(
        "FakePopen",
        (_CountingFakePopen,),
        {
            "calls": [],
            "collect_stdout": collect_stdout,
            "collect_returncode": collect_returncode,
            "run_stdout": run_stdout,
            "run_returncode": run_returncode,
        },
    )


class TestCollectTestCount:
    """QS-290 (S-1): `_collect_test_count` is the extracted collect-only probe.

    It returns `int` on a parseable count and `None` when the count is
    unknown — `None` is NOT 0: callers must fall back to `-n auto` rather
    than mistake an unparseable probe for a tiny run.
    """

    def test_returns_parsed_count(self) -> None:
        fake = _fake_popen(collect_stdout="123 tests collected in 1.2s\n")
        with patch.object(quality_gate.subprocess, "Popen", fake):
            assert quality_gate._collect_test_count(["/x/tests"]) == 123
        assert len(fake.calls) == 1
        assert "--collect-only" in fake.calls[0]
        assert "/x/tests" in fake.calls[0]

    @pytest.mark.parametrize(
        ("stdout", "expected"),
        [
            ("3 tests collected in 0.00s\n", 3),
            ("1/3 tests collected (2 deselected) in 0.00s\n", 1),
            ("12/340 tests collected (328 deselected) in 1.20s\n", 12),
        ],
        ids=["plain", "deselected", "deselected-large"],
    )
    def test_parses_the_deselection_form(self, stdout: str, expected: int) -> None:
        """pytest prints `N/M tests collected (K deselected)` whenever anything
        deselects — a `-k`/`-m` in `pytest.ini`'s `addopts` (a `slow` marker is
        already declared here) or a `pytest_collection_modifyitems` hook.

        Verified against real pytest:
            plain:       `3 tests collected in 0.01s`
            deselected:  `1/3 tests collected (2 deselected) in 0.00s`

        `.match()` anchors at position 0, so the `1/` prefix made this return
        `None` → `_resolve_files_workers` always answered `-n auto` → the serial
        fast path this story exists to deliver silently stopped existing, with
        no warning and no symptom but the lost seconds. The SELECTED count is
        the numerator — that is what actually runs.
        """
        fake = _fake_popen(collect_stdout=stdout)
        with patch.object(quality_gate.subprocess, "Popen", fake):
            assert quality_gate._collect_test_count(["/x"]) == expected

    def test_a_parametrize_id_cannot_masquerade_as_the_summary(self) -> None:
        """`.search()` is unanchored, so a parametrized test ID printed by
        `--collect-only` can satisfy the pattern BEFORE the real summary line.

        Verified: a param value of `"12/340 tests collected (328 deselected)"`
        yields an id the probe parses, returning 12 for a 3-test file — silently
        corrupting both the worker decision and the progress denominator, with
        no symptom. Latent today only because `TestCollectTestCount` happens to
        pin `ids=[...]`; dropping that would arm it. Real ids always begin with
        the file path, so anchoring at column 0 is immune.
        """
        fake = _fake_popen(
            collect_stdout=(
                "tests/test_x.py::test_p[12/340 tests collected (328 deselected)]\n"
                "3 tests collected in 0.01s\n"
            )
        )
        with patch.object(quality_gate.subprocess, "Popen", fake):
            assert quality_gate._collect_test_count(["/x"]) == 3

    def test_returns_one_for_singular_wording(self) -> None:
        fake = _fake_popen(collect_stdout="1 test collected in 0.1s\n")
        with patch.object(quality_gate.subprocess, "Popen", fake):
            assert quality_gate._collect_test_count(["/x"]) == 1

    def test_returns_none_when_unparseable(self) -> None:
        fake = _fake_popen(collect_stdout="ERROR: file or directory not found\n")
        with patch.object(quality_gate.subprocess, "Popen", fake):
            assert quality_gate._collect_test_count(["/x"]) is None

    def test_nonzero_returncode_is_unknown(self) -> None:
        """A partially-failed collection prints a
        parseable count (`"5 tests collected, 2 errors"`) alongside a non-zero
        exit. Trusting that number silently routed a broken collection onto the
        serial fast path. A non-zero rc is exactly the "unknown" the docstring
        already argues must degrade to `-n auto`."""
        fake = _fake_popen(
            collect_stdout="5 tests collected, 2 errors in 0.4s\n",
            collect_returncode=2,
        )
        with patch.object(quality_gate.subprocess, "Popen", fake):
            assert quality_gate._collect_test_count(["/x"]) is None

    def test_zero_returncode_with_a_count_is_trusted(self) -> None:
        """The negative half: a clean collection is still used."""
        fake = _fake_popen(collect_stdout="5 tests collected in 0.4s\n")
        with patch.object(quality_gate.subprocess, "Popen", fake):
            assert quality_gate._collect_test_count(["/x"]) == 5

    def test_exit_5_is_a_known_zero_not_unknown(self) -> None:
        """pytest exits **5** for "no tests collected" — a KNOWN count of zero.

        Verified: `pytest --collect-only -q test_empty.py` prints
        `no tests collected` and exits 5. Folding that into "unknown" made
        `_resolve_files_workers` return `-n auto` and spin up xdist for a target
        with zero tests — precisely the fixed overhead S-1 exists to remove — and
        left the "a genuine 0 is authoritative" branch unreachable from this
        caller. A partial collection (`5 tests collected, 2 errors`) exits **2**,
        so the protection against trusting that count is untouched.
        """
        fake = _fake_popen(collect_stdout="no tests collected in 0.01s\n", collect_returncode=5)
        with patch.object(quality_gate.subprocess, "Popen", fake):
            assert quality_gate._collect_test_count(["/x"]) == 0

    @pytest.mark.parametrize("rc", [1, 2, 3, 4], ids=["failures", "interrupted", "internal", "usage"])
    def test_other_nonzero_codes_stay_unknown(self, rc: int) -> None:
        fake = _fake_popen(collect_stdout="5 tests collected, 2 errors\n", collect_returncode=rc)
        with patch.object(quality_gate.subprocess, "Popen", fake):
            assert quality_gate._collect_test_count(["/x"]) is None

    def test_probe_is_bounded_by_a_timeout(self) -> None:
        """Every other inner-loop subprocess in this module is bounded (`_run`'s
        `timeout`, used for `git fetch` and diff-cover). An unbounded collection
        probe could block the sub-15 s promise indefinitely."""
        seen: dict = {}
        killed: list[bool] = []

        class HangingPopen(_CountingFakePopen):
            calls: list[list[str]] = []
            _n = 0

            def communicate(self, *a, **kw):  # type: ignore[no-untyped-def]
                type(self)._n += 1
                if type(self)._n == 1:
                    seen.update(kw)
                    raise subprocess.TimeoutExpired(cmd=["pytest"], timeout=kw.get("timeout"))
                # A real `communicate()` after `kill()` returns normally — it is
                # how the child gets reaped.
                return ("", "")

            def kill(self):  # type: ignore[no-untyped-def]
                killed.append(True)

        with patch.object(quality_gate.subprocess, "Popen", HangingPopen):
            assert quality_gate._collect_test_count(["/x"]) is None
        assert seen.get("timeout") == quality_gate._COLLECT_TIMEOUT_SECONDS
        assert killed, "the hung probe must be killed so it cannot outlive the gate"

    def test_reaping_communicate_is_bounded(self) -> None:
        """The reap must not become an unbounded hang.

        If collection leaves a grandchild holding the inherited stdout/stderr
        pipes (a conftest or plugin that spawns a helper at import), `SIGKILL`
        reaches only the direct child, so an unbounded reap turns the 60 s
        collect budget into an indefinite stall of the whole gate — strictly
        worse than the `ResourceWarning` the reap exists to remove.
        """
        seen: list[dict] = []

        class WedgedPopen(_CountingFakePopen):
            calls: list[list[str]] = []

            def communicate(self, *a, **kw):  # type: ignore[no-untyped-def]
                seen.append(kw)
                raise subprocess.TimeoutExpired(cmd=["pytest"], timeout=kw.get("timeout"))

            def kill(self):  # type: ignore[no-untyped-def]
                pass

        with patch.object(quality_gate.subprocess, "Popen", WedgedPopen):
            # A second TimeoutExpired from the reap must be swallowed, not raised.
            assert quality_gate._collect_test_count(["/x"]) is None
        assert len(seen) == 2, seen
        assert seen[1].get("timeout") == quality_gate._COLLECT_REAP_TIMEOUT_SECONDS

    def test_killed_probe_is_reaped(self) -> None:
        """`kill()` alone leaves the child unreaped with both pipes open, so
        CPython prints `ResourceWarning: subprocess N is still running` into the
        middle of gate output. The kill must be followed by a `communicate()`.
        """
        events: list[str] = []

        class HangingPopen(_CountingFakePopen):
            calls: list[list[str]] = []

            def communicate(self, *a, **kw):  # type: ignore[no-untyped-def]
                events.append("communicate")
                if len(events) == 1:  # the first (timed-out) wait
                    raise subprocess.TimeoutExpired(cmd=["pytest"], timeout=1)
                return ("", "")

            def kill(self):  # type: ignore[no-untyped-def]
                events.append("kill")

        with patch.object(quality_gate.subprocess, "Popen", HangingPopen):
            assert quality_gate._collect_test_count(["/x"]) is None
        assert events == ["communicate", "kill", "communicate"], events

    def test_pins_utf8_replace_and_no_coverage_core(self) -> None:
        """Same encoding contract as the run it precedes (S5), and the probe
        must NOT inherit `COVERAGE_CORE` (coverage is inactive at collection)."""
        seen: dict = {}

        class FakePopen(_CountingFakePopen):
            calls: list[list[str]] = []

            def __init__(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
                seen.update(kwargs)
                super().__init__(cmd, **kwargs)

        with patch.object(quality_gate.subprocess, "Popen", FakePopen):
            quality_gate._collect_test_count(["/x"])
        assert seen.get("encoding") == "utf-8"
        assert seen.get("errors") == "replace"
        assert "env" not in seen, f"the probe must inherit the parent env verbatim: {seen!r}"


class TestStreamPytestTotalTests:
    """QS-290 (S-1/S-5): `_stream_pytest` takes ONE `total_tests` parameter.

    `None` → collect the whole `TESTS_DIR` denominator itself (the full-gate
    and seed paths). An int → trust the caller and spawn NOTHING extra.
    The old `collect_targets` parameter is gone.
    """

    def test_int_total_spawns_no_collect_subprocess(self) -> None:
        fake = _fake_popen(run_stdout="..\n2 passed in 0.1s\n")
        with patch.object(quality_gate.subprocess, "Popen", fake):
            quality_gate._stream_pytest(["pytest"], total_tests=7)
        assert len(fake.calls) == 1, f"expected only the run, got {fake.calls!r}"
        assert not any("--collect-only" in c for c in fake.calls)

    def test_none_total_collects_tests_dir(self) -> None:
        fake = _fake_popen(collect_stdout="9 tests collected\n", run_stdout="")
        with patch.object(quality_gate.subprocess, "Popen", fake):
            quality_gate._stream_pytest(["pytest"], total_tests=None)
        assert len(fake.calls) == 2
        assert str(quality_gate.TESTS_DIR) in fake.calls[0]

    def test_collect_targets_parameter_is_gone(self) -> None:
        """The two-parameter overload the delta-auditor flagged must not exist."""
        import inspect

        params = inspect.signature(quality_gate._stream_pytest).parameters
        assert "collect_targets" not in params
        assert list(params) == ["cmd", "total_tests"], list(params)

    # --- 0 must not double as "unknown" ---

    def test_learn_sentinel_is_distinct_from_a_genuine_zero(self) -> None:
        """`0` meant both "genuinely no tests" and "unknown, learn from the
        stream", which contradicted the `None`-vs-`0` care taken in
        `_collect_test_count`. A named negative sentinel is impossible as a real
        count, so it cannot collide, and keeps ONE parameter (four reviewers
        rejected re-splitting it)."""
        assert quality_gate._LEARN_FROM_STREAM < 0

    def test_genuine_zero_does_not_learn_from_the_stream(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A caller that KNOWS the target collects 0 tests is authoritative; a
        stray `[N/M]` must not silently redefine its denominator."""
        fake = _fake_popen(run_stdout="..            [ 2/99]\n2 passed in 0.1s\n")
        with patch.object(quality_gate.subprocess, "Popen", fake):
            quality_gate._stream_pytest(["pytest"], total_tests=0)
        err = capsys.readouterr().err
        assert "/99" not in err, err

    def test_learn_sentinel_does_learn_from_the_stream(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake = _fake_popen(run_stdout="..            [ 2/20]\n2 passed in 0.1s\n")
        with patch.object(quality_gate.subprocess, "Popen", fake):
            quality_gate._stream_pytest(["pytest"], total_tests=quality_gate._LEARN_FROM_STREAM)
        assert "(2/20)" in capsys.readouterr().err

    # --- Keep a terminal line on a 0-test run ---

    def test_zero_selected_run_still_reports_done(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The single most common inner-loop invocation is a `.py` change that
        testmon deselects to 0 tests. The `total_tests > 0 or final_total > 0`
        guard made the `pytest: done (…)` line vanish there, leaving NOTHING
        between `selecting impacted tests (testmon)…` and the verdict. The old
        `done (0/6912)` had the wrong denominator but did confirm the pass ran.
        """
        fake = _fake_popen(run_stdout="no tests ran in 0.4s\n")
        with patch.object(quality_gate.subprocess, "Popen", fake):
            quality_gate._stream_pytest(["pytest"], total_tests=quality_gate._LEARN_FROM_STREAM)
        err = capsys.readouterr().err
        assert "pytest: done (0/0)" in err, err
        assert "passed=0 failed=0 errors=0" in err, err


class TestSerialFastPath:
    """QS-290 (S-1, AC7/AC8): `check_pytest_files` skips the xdist spin-up for
    small targets, using ONE collect-only probe for both the denominator and
    the worker decision."""

    @staticmethod
    def _run(
        count: int | None,
        *,
        env_workers: str | None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> dict:
        if env_workers is None:
            monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        else:
            monkeypatch.setenv("QS_QG_PYTEST_WORKERS", env_workers)
        captured: dict = {}

        def fake_stream(cmd: list[str], total_tests: int | None = None) -> dict:
            captured["cmd"] = cmd
            captured["total_tests"] = total_tests
            return {"name": "pytest", "passed": True, "detail": ""}

        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate, "_collect_test_count", return_value=count),
            patch.object(quality_gate, "_stream_pytest", side_effect=fake_stream),
        ):
            quality_gate.check_pytest_files(["tests/test_x.py"])
        return captured

    def test_at_threshold_runs_serial(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cap = self._run(quality_gate._SERIAL_MAX_TESTS, env_workers=None, monkeypatch=monkeypatch)
        assert "-n" not in cap["cmd"], cap["cmd"]
        assert cap["total_tests"] == quality_gate._SERIAL_MAX_TESTS
        err = capsys.readouterr().err
        assert err.startswith("[pytest] "), err
        assert f"{quality_gate._SERIAL_MAX_TESTS} tests <= {quality_gate._SERIAL_MAX_TESTS}" in err, err
        assert "single-process" in err, err

    def test_just_above_threshold_uses_xdist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cap = self._run(quality_gate._SERIAL_MAX_TESTS + 1, env_workers=None, monkeypatch=monkeypatch)
        assert cap["cmd"][cap["cmd"].index("-n") + 1] == "auto"
        assert cap["total_tests"] == quality_gate._SERIAL_MAX_TESTS + 1

    def test_unknown_count_uses_xdist_and_learn_sentinel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC7: `None` is unknown → `-n auto`, and NOT forwarded as `None` (which
        `_stream_pytest` would read as "collect the whole tree yourself" and
        spawn a second probe).

        The forwarded value is now the explicit
        `_LEARN_FROM_STREAM` sentinel rather than `0`, so a genuine zero-test
        target stays distinguishable from an unknown one.
        """
        cap = self._run(None, env_workers=None, monkeypatch=monkeypatch)
        assert cap["cmd"][cap["cmd"].index("-n") + 1] == "auto"
        assert cap["total_tests"] == quality_gate._LEARN_FROM_STREAM

    def test_explicit_zero_is_routed_through_the_provenance_check(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`QS_QG_PYTEST_WORKERS=0` must be honoured *as an explicit request*, not
        as a side effect of an earlier `workers is None` return.

        Observable difference: an explicit serial request is the user's decision,
        so the threshold banner (which explains OUR decision) must stay silent —
        and it must stay silent even on a target that would trip the threshold.
        """
        cap = self._run(3, env_workers="0", monkeypatch=monkeypatch)
        assert "-n" not in cap["cmd"], cap["cmd"]
        assert "single-process (skipping xdist spin-up)" not in capsys.readouterr().err

    @pytest.mark.parametrize("value", ["", "0", "auto", "4"], ids=["empty", "zero", "auto", "count"])
    def test_valid_values_are_explicit(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QS_QG_PYTEST_WORKERS", value)
        assert quality_gate._workers_env_is_explicit() is True

    @pytest.mark.parametrize("value", ["autoo", "-1", "4x", "abc"], ids=["typo", "neg", "suffix", "word"])
    def test_malformed_values_are_not_explicit(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QS_QG_PYTEST_WORKERS", value)
        assert quality_gate._workers_env_is_explicit() is False

    def test_unset_is_not_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        assert quality_gate._workers_env_is_explicit() is False

    # --- a malformed value is not an explicit request ---

    @pytest.mark.parametrize("bad", ["autoo", "-1", "4x"], ids=["typo", "negative", "suffix"])
    def test_malformed_env_value_does_not_count_as_explicit(
        self, bad: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`_pytest_workers()` warns and falls back to `"auto"` for a malformed
        value. The `os.environ.get(...) is not None` provenance check then read
        the typo as deliberate and forced xdist onto a 3-test target, silently
        losing the fast path. "Invalid" must behave as "unset"."""
        cap = self._run(3, env_workers=bad, monkeypatch=monkeypatch)
        assert "-n" not in cap["cmd"], cap["cmd"]
        err = capsys.readouterr().err
        assert "invalid QS_QG_PYTEST_WORKERS" in err, err  # the warning still fires
        assert "single-process" in err, err

    def test_env_zero_forces_serial_on_a_big_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC8: an explicit `0` is honored verbatim even for 1000 tests."""
        cap = self._run(1000, env_workers="0", monkeypatch=monkeypatch)
        assert "-n" not in cap["cmd"], cap["cmd"]

    def test_env_eight_forces_xdist_on_a_tiny_target(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """AC8: an explicit `8` beats the threshold for a 3-test target — and the
        serial banner must NOT be printed."""
        cap = self._run(3, env_workers="8", monkeypatch=monkeypatch)
        assert cap["cmd"][cap["cmd"].index("-n") + 1] == "8"
        assert "single-process" not in capsys.readouterr().err

    def test_env_auto_forces_xdist_on_a_tiny_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`_pytest_workers()` returns `"auto"` both when unset and when set to
        `auto`, so provenance must come from `os.environ.get(...) is not None`."""
        cap = self._run(3, env_workers="auto", monkeypatch=monkeypatch)
        assert cap["cmd"][cap["cmd"].index("-n") + 1] == "auto"

    def test_no_xdist_stays_serial_without_banner(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)

        def fake_stream(cmd: list[str], total_tests: int | None = None) -> dict:
            return {"name": "pytest", "passed": True, "detail": ""}

        with (
            patch.object(quality_gate, "_has_xdist", return_value=False),
            patch.object(quality_gate, "_collect_test_count", return_value=3),
            patch.object(quality_gate, "_stream_pytest", side_effect=fake_stream),
        ):
            quality_gate.check_pytest_files(["tests/test_x.py"])
        assert "single-process" not in capsys.readouterr().err

    def test_argv_requests_the_count_progress_style(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On an unknown count `check_pytest_files` forwards `_LEARN_FROM_STREAM`,
        but with a plain `-q` argv pytest prints `[NN%]` — the `[N/M]` shape never
        appears and the denominator is never learned, so mid-run progress silently
        vanished on exactly that path (the retired whole-tree `--collect-only`
        used to supply a denominator here)."""
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        cap = self._run(None, env_workers=None, monkeypatch=monkeypatch)
        cmd = cap["cmd"]
        assert "-o" in cmd, cmd
        assert cmd[cmd.index("-o") + 1] == "console_output_style=count", cmd

    def test_progress_is_emitted_on_the_unknown_count_path(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """End-to-end: unknown count → the run's own `[N/M]` yields progress."""
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        fake = _fake_popen(
            collect_stdout="ERROR: unparseable\n",
            collect_returncode=3,
            run_stdout="..  [ 2/20]\n..  [ 4/20]\n4 passed in 0.2s\n",
        )
        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate.subprocess, "Popen", fake),
        ):
            quality_gate.check_pytest_files(["tests/test_x.py"])
        assert "(2/20)" in capsys.readouterr().err

    def test_zero_test_target_runs_serial(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`--quick` on a path with no tests (pytest rc 5 → a KNOWN 0) takes the
        serial fast path instead of booting xdist for nothing."""
        cap = self._run(0, env_workers=None, monkeypatch=monkeypatch)
        assert "-n" not in cap["cmd"], cap["cmd"]
        assert cap["total_tests"] == 0

    def test_exactly_one_collect_only_subprocess(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC7: one `--collect-only` spawn per `check_pytest_files` call — the
        count feeds BOTH the worker decision and the progress denominator."""
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        fake = _fake_popen(collect_stdout="3 tests collected\n")
        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate.subprocess, "Popen", fake),
        ):
            quality_gate.check_pytest_files(["tests/test_x.py"])
        collect_spawns = [c for c in fake.calls if "--collect-only" in c]
        assert len(collect_spawns) == 1, f"expected 1 collect-only spawn, got {fake.calls!r}"
        assert len(fake.calls) == 2, fake.calls

    def test_dev_only_scope_path_gets_the_fast_path(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """AC7: the fast path lands in `check_pytest_files`, which also serves
        `main()`'s dev-only scope — intentional (1–3 changed test files)."""
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        captured: dict = {}

        def fake_stream(cmd: list[str], total_tests: int | None = None) -> dict:
            captured["cmd"] = cmd
            return {"name": "pytest", "passed": True, "detail": ""}

        with (
            patch("sys.argv", ["quality_gate.py", "--json"]),
            patch.object(quality_gate, "_get_changed_files", return_value=["tests/test_x.py"]),
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate, "_collect_test_count", return_value=4),
            patch.object(quality_gate, "_stream_pytest", side_effect=fake_stream),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()
        assert "-n" not in captured["cmd"], captured["cmd"]
        assert "single-process" in capsys.readouterr().err


class TestFullGateUntouchedByFastPath:
    """QS-290 AC11: `check_pytest` keeps whole-tree collection AND `-n auto`."""

    def test_full_gate_still_collects_tests_dir_and_requests_auto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        fake = _fake_popen(collect_stdout="3 tests collected\n")
        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate.subprocess, "Popen", fake),
        ):
            quality_gate.check_pytest()
        assert len(fake.calls) == 2, fake.calls
        assert str(quality_gate.TESTS_DIR) in fake.calls[0]
        run_cmd = fake.calls[1]
        # A 3-test whole-tree count must NOT demote the full gate to serial.
        assert run_cmd[run_cmd.index("-n") + 1] == "auto", run_cmd


class TestPytestExitCodeFidelity:
    """QS-290 AC13 (permanent): a non-zero pytest exit propagates through
    `_stream_pytest` / `check_pytest_files` on BOTH argv shapes, and the
    reported count matches what pytest reported.

    This story rewrites the very code path used to verify itself, so a
    vacuous green is the specific risk worth a permanent guard.
    """

    _STDOUT = "..F\n2 passed, 1 failed in 0.3s\n"

    @pytest.mark.parametrize(
        ("count", "expect_xdist"),
        [(3, False), (500, True)],
        ids=["serial", "xdist"],
    )
    def test_failure_propagates_and_count_is_reported(
        self,
        count: int,
        expect_xdist: bool,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        fake = _fake_popen(
            collect_stdout=f"{count} tests collected\n",
            run_stdout=self._STDOUT,
            run_returncode=1,
        )
        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate.subprocess, "Popen", fake),
        ):
            result = quality_gate.check_pytest_files(["tests/test_x.py"])

        assert result["passed"] is False
        assert result["returncode"] == 1
        run_cmd = fake.calls[1]
        assert ("-n" in run_cmd) is expect_xdist, run_cmd
        err = capsys.readouterr().err
        # pytest reported 2 passed + 1 failed → the terminal line reports 3.
        assert "done (3/" in err, err
        assert "passed=2 failed=1" in err, err

    @pytest.mark.parametrize("rc", [1, 2, 5], ids=["failures", "collect-error", "internal"])
    def test_stream_pytest_surfaces_raw_returncode(self, rc: int) -> None:
        fake = _fake_popen(run_stdout=self._STDOUT, run_returncode=rc)
        with patch.object(quality_gate.subprocess, "Popen", fake):
            result = quality_gate._stream_pytest(["pytest"], total_tests=3)
        assert result["returncode"] == rc
        assert result["passed"] is False


# --- QS-290 S-5: denominator from the run's own output, no collect-only ---


class TestCountProgressSuffix:
    """QS-290 (S-5, AC6): `-o console_output_style=count` makes pytest print
    `[N/M]` instead of `[NN%]`, under `-q` in BOTH serial and xdist mode, and
    it reports the SELECTED count under deselection. `_clean_pytest_line` must
    strip that shape too, or a progress line stops being "all progress
    characters" and the dot tally silently breaks.
    """

    @pytest.mark.parametrize(
        "raw",
        [
            "... [3/3]",
            "...                     [  3/682]",
            "[gw0] ...                [ 69/682]",
            "... [43%]",
        ],
        ids=["count-tight", "count-padded", "xdist-count", "legacy-percent"],
    )
    def test_progress_suffixes_are_stripped(self, raw: str) -> None:
        assert quality_gate._clean_pytest_line(raw) == "..."

    def test_stripped_line_still_tallies(self) -> None:
        text = "..F                    [  3/100]\n"
        counts = quality_gate._parse_pytest_output(text)
        assert counts["passed"] == 2
        assert counts["failed"] == 1

    def test_count_suffix_re_captures_only_the_total(self) -> None:
        """The `current` group was captured but
        never used, so a future edit could silently shift `group(2)`."""
        m = quality_gate._COUNT_SUFFIX_RE.search("...   [ 69/682]")
        assert m is not None
        assert m.groups() == ("682",), m.groups()

    def test_build_testmon_cmd_requests_count_style(self) -> None:
        """AC6: the impacted pass asks pytest for the count style so the
        denominator can be read off the run itself."""
        with patch.object(quality_gate, "_pytest_workers", return_value=None):
            cmd = quality_gate._build_testmon_cmd()
        assert "-o" in cmd
        assert cmd[cmd.index("-o") + 1] == "console_output_style=count"


class TestStreamPytestLearnsDenominator:
    """QS-290 (S-5, AC6): with `total_tests=0`, `_stream_pytest` learns the
    denominator from the first `[N/M]` suffix in the stream instead of paying
    for an upfront `--collect-only` subprocess.
    """

    def test_learns_total_from_stream_and_emits_existing_format(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stream = "..                 [ 2/20]\n..                 [ 4/20]\n4 passed in 0.2s\n"
        fake = _fake_popen(run_stdout=stream)
        with patch.object(quality_gate.subprocess, "Popen", fake):
            quality_gate._stream_pytest(["pytest"], total_tests=quality_gate._LEARN_FROM_STREAM)
        assert len(fake.calls) == 1, f"no collect-only may be spawned: {fake.calls!r}"
        err = capsys.readouterr().err
        # The emitted line keeps TODAY's format — QS-299's --seed-testmon-follow
        # parses this exact shape.
        assert "  pytest: 10% (2/20) | passed=2 failed=0 errors=0" in err, err
        assert "  pytest: done (4/20) | passed=4 failed=0 errors=0" in err, err

    def test_no_count_suffix_emits_no_percentage_and_falls_back(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """AC6: with no `[N/M]` ever seen, no percentage line is emitted and the
        terminal line's denominator falls back to the observed count."""
        fake = _fake_popen(run_stdout="...\n3 passed in 0.1s\n")
        with patch.object(quality_gate.subprocess, "Popen", fake):
            quality_gate._stream_pytest(["pytest"], total_tests=quality_gate._LEARN_FROM_STREAM)
        err = capsys.readouterr().err
        assert "%" not in err, err
        assert "  pytest: done (3/3) | passed=3 failed=0 errors=0" in err, err

    def test_non_progress_line_cannot_seed_the_denominator(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The learn block latched the FIRST line anywhere in the stream matching
        `[N/M]`, not the first *progress* line. Collection errors print before any
        progress, so a traceback or source line ending in that shape — e.g.
        `RATIO = SCALE[1/2]` — set a bogus denominator for the whole run and the
        progress line then read `100% (2/2)` for a 40-test run. Cosmetic (the
        verdict comes from the exit code), but actively misleading.
        """
        stream = (
            "ERROR collecting tests/test_x.py\n"
            "    RATIO = SCALE[1/2]\n"
            "....  [ 4/40]\n"
            "4 passed in 0.2s\n"
        )
        fake = _fake_popen(run_stdout=stream)
        with patch.object(quality_gate.subprocess, "Popen", fake):
            quality_gate._stream_pytest(["pytest"], total_tests=quality_gate._LEARN_FROM_STREAM)
        err = capsys.readouterr().err
        assert "/2)" not in err, f"a non-progress line seeded the denominator: {err!r}"
        assert "(4/40)" in err, err

    def test_caller_count_wins_over_the_stream(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A caller-supplied count is authoritative — the stream must not
        overwrite it (the `--quick` path already paid for a real collection)."""
        fake = _fake_popen(run_stdout="..                 [ 2/20]\n2 passed in 0.1s\n")
        with patch.object(quality_gate.subprocess, "Popen", fake):
            quality_gate._stream_pytest(["pytest"], total_tests=2)
        assert "  pytest: done (2/2) |" in capsys.readouterr().err

    def test_emitted_progress_line_round_trips_through_parse_seed_progress(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """AC5/AC6 (QS-299 guard): a progress line captured from a REAL
        `_stream_pytest` emission — fake pytest stdout through the real
        formatter — still parses via `_parse_seed_progress`."""
        stream = "..                 [ 2/20]\n..                 [ 4/20]\n4 passed in 0.2s\n"
        fake = _fake_popen(run_stdout=stream)
        with patch.object(quality_gate.subprocess, "Popen", fake):
            quality_gate._stream_pytest(["pytest"], total_tests=quality_gate._LEARN_FROM_STREAM)
        assert quality_gate._parse_seed_progress(capsys.readouterr().err) == (20, 4, 20)


class TestImpactedPassNoCollectOnly:
    """QS-290 (S-5, AC4/AC5): the impacted pass spawns no collect-only probe;
    the seed path keeps its whole-tree denominator."""

    def test_run_impacted_pass_learns_its_denominator(self, tmp_path: Path) -> None:
        xml = tmp_path / "coverage.xml"
        captured: dict = {}

        def fake_stream(cmd: list[str], total_tests: int | None = None) -> dict:
            captured["total_tests"] = total_tests
            xml.write_text("<coverage/>")
            return {"name": "pytest", "passed": True}

        with (
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            patch.object(quality_gate, "TESTMON_DATA", tmp_path / ".testmondata"),
            patch.object(quality_gate, "_reset_coverage_data"),
            patch.object(quality_gate, "COVERAGE_XML", xml),
            patch.object(quality_gate, "_build_testmon_cmd", return_value=["pytest"]),
            patch.object(quality_gate, "_stream_pytest", side_effect=fake_stream),
            patch.object(quality_gate, "_run", return_value=_cp(0, stdout="100%")),
        ):
            quality_gate._run_impacted_pass("origin/main")
        assert captured["total_tests"] == quality_gate._LEARN_FROM_STREAM, (
            "the impacted pass must not ask _stream_pytest to collect a denominator"
        )

    def test_seed_testmon_keeps_its_collect_only_denominator(self) -> None:
        """AC5: unchanged path — its 3–5 s is noise inside a ~20-minute select-all."""
        captured: dict = {}

        def fake_stream(cmd: list[str], total_tests: int | None = None) -> dict:
            captured["total_tests"] = total_tests
            return {"name": "pytest", "passed": True, "returncode": 0}

        with (
            patch.object(quality_gate, "_testmon_available", return_value=True),
            patch.object(quality_gate, "_claim_and_preempt"),
            patch.object(quality_gate, "_rebuild_testmon_baseline"),
            patch.object(quality_gate, "_write_completion_if_owner"),
            patch.object(quality_gate, "_stream_pytest", side_effect=fake_stream),
        ):
            assert quality_gate.seed_testmon(token="T") == 0
        assert captured["total_tests"] is None, (
            "seed_testmon must keep the whole-tree collect-only denominator"
        )


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


# --- QS-290 D-18: truthful --full scope reason + reason pluralization ---


class TestDetectScopeReasonPluralization:
    """QS-290 (D-18, AC10): the dev-only / ui-only reason strings must agree
    with themselves grammatically — `(1 file)`, not `(1 files)`.

    There was previously NO assertion anywhere on the dev-only reason
    string (the `_detect_scope` call sites in the main() tests supply it as
    a patched *input*), so the mis-pluralization survived untested.
    """

    def test_dev_only_singular(self) -> None:
        info = quality_gate._detect_scope(["docs/workflow/project-rules.md"])
        assert info["scope"] == "dev-only"
        assert info["reason"] == "only dev/test files changed (1 file)"

    def test_dev_only_plural(self) -> None:
        info = quality_gate._detect_scope(["docs/a.md", "scripts/qs/quality_gate.py"])
        assert info["scope"] == "dev-only"
        assert info["reason"] == "only dev/test files changed (2 files)"

    def test_ui_only_singular(self) -> None:
        info = quality_gate._detect_scope(["custom_components/quiet_solar/ui/a.j2"])
        assert info["scope"] == "ui-only"
        assert info["reason"] == "only UI assets and dev files changed (1 UI asset, 1 file)"

    def test_ui_only_plural(self) -> None:
        info = quality_gate._detect_scope(
            [
                "custom_components/quiet_solar/ui/a.j2",
                "custom_components/quiet_solar/ui/resources/b.js",
                "docs/c.md",
            ]
        )
        assert info["scope"] == "ui-only"
        assert info["reason"] == "only UI assets and dev files changed (2 UI assets, 3 files)"


class TestFullFlagScopeReason:
    """QS-290 (D-18, AC10): `--full` forces `scope = "full"` but used to keep
    `_detect_scope`'s reason, printing the self-contradicting
    `scope: FULL (only dev/test files changed (1 files))`.
    """

    @staticmethod
    def _run_main_full(scope_info: dict) -> None:
        """Drive `main() --full` with a patched `_detect_scope`.

        Callers read stderr via `capsys` (this
        used to `return ""`, which every caller ignored).
        """
        with (
            patch("sys.argv", ["quality_gate.py", "--full", "--json"]),
            patch.object(quality_gate, "_get_changed_files", return_value=["x"]),
            patch.object(quality_gate, "_detect_scope", return_value=scope_info),
            _patch_all_gates(),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

    def test_full_flag_on_dev_only_tree_prints_flag_reason(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run_main_full(
            {
                "scope": "dev-only",
                "changed_test_files": [],
                "reason": "only dev/test files changed (1 file)",
            }
        )
        err = capsys.readouterr().err
        assert "scope: FULL (--full flag)" in err, err
        assert "only dev/test files changed" not in err, err

    def test_full_flag_keeps_production_reason_when_already_full(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The substitution must NOT clobber a genuinely useful full-scope
        reason (`production files changed: …`) when the detected scope was
        already `full`."""
        self._run_main_full(
            {
                "scope": "full",
                "changed_test_files": [],
                "reason": "production files changed: custom_components/quiet_solar/home_model/load.py",
            }
        )
        err = capsys.readouterr().err
        assert "scope: FULL (production files changed: " in err, err
        assert "--full flag" not in err, err


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
        git_path: tuple[int, str] = (1, ""),
        remote_url: tuple[int, str] = (0, "github"),
    ):
        """Build a `_run` side_effect keyed on the git subcommand.

        `reachable` (NH2) maps a candidate ref → returncode for the
        `git merge-base <ref> HEAD` reachability probe; default 0
        (reachable) so the pre-NH2 tests keep passing unchanged.

        `remote_url` answers `git remote get-url origin`; the default matches the
        `_marker` helper's FETCH_HEAD content so the TTL can arm.

        `git_path` (QS-290 S-7) answers `git rev-parse --git-path FETCH_HEAD`.
        The default `(1, "")` means "no marker resolvable" → fetch exactly as
        before the TTL existed. Without this EXPLICIT arm an unmatched
        `--git-path` fell into the `@{u}` arm and passed only by luck.
        """
        reachable = reachable or {}

        def _side_effect(cmd: list[str], *_a, **_k) -> subprocess.CompletedProcess[str]:
            if cmd[:2] == ["git", "fetch"]:
                return _cp(0)
            if cmd[:3] == ["git", "rev-parse", "--git-path"]:
                return _cp(git_path[0], stdout=git_path[1])
            if cmd[:2] == ["git", "remote"]:
                return _cp(remote_url[0], stdout=remote_url[1])
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

    @staticmethod
    def _fetch_calls(mock_run) -> list:  # type: ignore[no-untyped-def]
        """Select the `git fetch` invocations by PREDICATE, not by index.

        QS-290 (S-7) put the `rev-parse --git-path FETCH_HEAD` TTL probe ahead
        of the fetch, so index 0 is no longer the fetch.
        """
        return [c for c in mock_run.call_args_list if c.args[0][:2] == ["git", "fetch"]]

    def test_fetches_origin_main_before_resolving_refs(self) -> None:
        with patch.object(quality_gate, "_run", side_effect=self._router({"origin/main": 0})) as mock_run:
            quality_gate._resolve_diff_base()
        argvs = [c.args[0] for c in mock_run.call_args_list]
        fetch_idx = argvs.index(["git", "fetch", "origin", "main"])
        verify_idx = next(i for i, a in enumerate(argvs) if a[:3] == ["git", "rev-parse", "--verify"])
        assert fetch_idx < verify_idx, argvs

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
        fetch_calls = self._fetch_calls(mock_run)
        assert len(fetch_calls) == 1, [c.args[0] for c in mock_run.call_args_list]
        assert fetch_calls[0].kwargs.get("timeout") == quality_gate._FETCH_TIMEOUT_SECONDS

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


class TestFetchTtl:
    """QS-290 (S-7, AC9): `--impacted` TTL-caches its `git fetch origin main`.

    Repeat `.py` runs inside a 10-minute window skip the ~0.5 s (15 s-capped)
    fetch. Staleness direction is safe by construction: an older base yields a
    SUPERSET of changed lines, so a TTL hit can turn a PASS into a FAIL, never
    a FAIL into a PASS.
    """

    def _router(self, marker: Path | None, **kw):  # type: ignore[no-untyped-def]
        return TestResolveDiffBase._router(
            kw.pop("rev_parse", {"origin/main": 0}),
            git_path=(0, str(marker)) if marker is not None else (1, ""),
            **kw,
        )

    # --- the TTL must be armed only by a fetch of ORIGIN's main ---

    def test_marker_recording_another_remotes_main_still_fetches(self, tmp_path: Path) -> None:
        """`branch 'main' of <url>` carries the URL, and matching only the prefix
        accepted ANY remote's main. Verified in a two-remote clone: after
        `git fetch upstream main` the marker reads `branch 'main' of …/upstream`
        while `origin/main` — the ref the base ladder actually resolves — was
        never fetched and may be arbitrarily stale. That is the standard fork
        workflow (`origin` = fork, `upstream` = canonical), and the skip line
        claims `origin/main`, so the marker must support that claim.
        """
        marker = self._marker(
            tmp_path, age_seconds=1, content="beef\t\tbranch 'main' of https://host/upstream\n"
        )
        router = self._router(marker, remote_url=(0, "https://host/fork"))
        with patch.object(quality_gate, "_run", side_effect=router) as mock_run:
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert len(TestResolveDiffBase._fetch_calls(mock_run)) == 1

    def test_marker_recording_origins_main_skips(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        marker = self._marker(
            tmp_path, age_seconds=30, content="beef\t\tbranch 'main' of https://host/fork\n"
        )
        router = self._router(marker, remote_url=(0, "https://host/fork"))
        with patch.object(quality_gate, "_run", side_effect=router) as mock_run:
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert TestResolveDiffBase._fetch_calls(mock_run) == []
        assert "fetch skipped" in capsys.readouterr().err

    def test_dot_git_suffix_does_not_break_the_match(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """git STRIPS a trailing `.git` when writing FETCH_HEAD but
        `git remote get-url origin` keeps it — verified on this very repo
        (`…/quiet-solar.git` vs `…/quiet-solar`) and in a local clone with an
        explicit `.git` path. A naive equality check would therefore never match
        and would silently disable the TTL everywhere — the same
        feature-deleting trap that ruled out keying on `refs/remotes/origin/main`.
        """
        marker = self._marker(
            tmp_path, age_seconds=30, content="beef\t\tbranch 'main' of https://host/repo\n"
        )
        router = self._router(marker, remote_url=(0, "https://host/repo.git"))
        with patch.object(quality_gate, "_run", side_effect=router) as mock_run:
            quality_gate._resolve_diff_base()
        assert TestResolveDiffBase._fetch_calls(mock_run) == []
        assert "fetch skipped" in capsys.readouterr().err

    def test_unresolvable_origin_url_still_fetches(self, tmp_path: Path) -> None:
        """No `origin` remote at all (or any probe failure) → unknown → fetch."""
        marker = self._marker(tmp_path, age_seconds=30)
        router = self._router(marker, remote_url=(128, ""))
        with patch.object(quality_gate, "_run", side_effect=router) as mock_run:
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert len(TestResolveDiffBase._fetch_calls(mock_run)) == 1

    def test_empty_origin_url_still_fetches(self, tmp_path: Path) -> None:
        marker = self._marker(tmp_path, age_seconds=30)
        router = self._router(marker, remote_url=(0, "  \n"))
        with patch.object(quality_gate, "_run", side_effect=router) as mock_run:
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert len(TestResolveDiffBase._fetch_calls(mock_run)) == 1

    @pytest.fixture(autouse=True)
    def _not_ci(self):
        """The TTL is now bypassed under CI, so
        every test here must state that it is exercising the LOCAL path."""
        with patch.object(quality_gate, "_is_ci", return_value=False):
            yield

    @staticmethod
    def _marker(
        tmp_path: Path,
        age_seconds: float,
        content: str = "deadbeef\t\tbranch 'main' of github\n",
    ) -> Path:
        marker = tmp_path / "FETCH_HEAD"
        marker.write_text(content)
        stamp = time.time() - age_seconds
        os.utime(marker, (stamp, stamp))
        return marker

    def test_marker_path_comes_from_git_rev_parse_git_path(self, tmp_path: Path) -> None:
        """AC9: the marker is resolved via git, never hardcoded — in a LINKED
        worktree `.git` is a FILE, so a literal `.git/FETCH_HEAD` would be a
        path under a non-directory and the TTL would silently never hit."""
        marker = self._marker(tmp_path, age_seconds=60)
        with patch.object(quality_gate, "_run", side_effect=self._router(marker)) as mock_run:
            quality_gate._resolve_diff_base()
        probes = [c.args[0] for c in mock_run.call_args_list if c.args[0][:2] == ["git", "rev-parse"]]
        assert ["git", "rev-parse", "--git-path", "FETCH_HEAD"] in probes, probes

    def test_inside_ttl_skips_fetch_and_emits_skip_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        marker = self._marker(tmp_path, age_seconds=180)
        with patch.object(quality_gate, "_run", side_effect=self._router(marker)) as mock_run:
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert TestResolveDiffBase._fetch_calls(mock_run) == []
        err = capsys.readouterr().err
        assert "fetch skipped (origin/main fetched 3m ago)" in err, err

    def test_skip_line_renders_seconds_under_a_minute(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        marker = self._marker(tmp_path, age_seconds=12)
        with patch.object(quality_gate, "_run", side_effect=self._router(marker)):
            quality_gate._resolve_diff_base()
        assert "fetch skipped (origin/main fetched 12s ago)" in capsys.readouterr().err

    def test_outside_ttl_fetches(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        marker = self._marker(tmp_path, age_seconds=quality_gate._FETCH_TTL_SECONDS + 5)
        with patch.object(quality_gate, "_run", side_effect=self._router(marker)) as mock_run:
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert len(TestResolveDiffBase._fetch_calls(mock_run)) == 1
        assert "fetch skipped" not in capsys.readouterr().err

    def test_missing_marker_fetches(self, tmp_path: Path) -> None:
        absent = tmp_path / "FETCH_HEAD"  # never created
        with patch.object(quality_gate, "_run", side_effect=self._router(absent)) as mock_run:
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert len(TestResolveDiffBase._fetch_calls(mock_run)) == 1

    def test_rev_parse_failure_fetches(self) -> None:
        """A failing `--git-path` probe → fetch exactly as before the TTL."""
        with patch.object(quality_gate, "_run", side_effect=self._router(None)) as mock_run:
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert len(TestResolveDiffBase._fetch_calls(mock_run)) == 1

    def test_empty_git_path_stdout_fetches(self) -> None:
        """A zero-rc probe with empty stdout is unusable — fetch."""
        router = TestResolveDiffBase._router({"origin/main": 0}, git_path=(0, "  \n"))
        with patch.object(quality_gate, "_run", side_effect=router) as mock_run:
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert len(TestResolveDiffBase._fetch_calls(mock_run)) == 1

    def test_fetch_failure_warning_survives_the_ttl_path(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """AC9: on the fetch-taken path the S1 stale-base warning is unchanged."""

        def _side_effect(cmd: list[str], *_a, **_k) -> subprocess.CompletedProcess[str]:
            if cmd[:3] == ["git", "rev-parse", "--git-path"]:
                return _cp(1)
            if cmd[:2] == ["git", "fetch"]:
                return _cp(124, stderr="timed out after 15.0s")
            if cmd[:3] == ["git", "rev-parse", "--verify"] and cmd[3] == "origin/main":
                return _cp(0)
            if cmd[:2] == ["git", "merge-base"]:
                return _cp(0)
            return _cp(1)

        with patch.object(quality_gate, "_run", side_effect=_side_effect):
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert "git fetch origin main` failed/timed out" in capsys.readouterr().err

    def test_unresolvable_base_after_a_skip_retries_with_the_fetch(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """AC9: if the base fails to resolve after a TTL-skipped fetch, retry
        ONCE with the fetch — the skip must never be the reason we give up."""
        marker = self._marker(tmp_path, age_seconds=60)
        state = {"fetched": False}

        def _side_effect(cmd: list[str], *_a, **_k) -> subprocess.CompletedProcess[str]:
            if cmd[:3] == ["git", "rev-parse", "--git-path"]:
                return _cp(0, stdout=str(marker))
            if cmd[:2] == ["git", "remote"]:
                return _cp(0, stdout="github")
            if cmd[:2] == ["git", "fetch"]:
                state["fetched"] = True
                return _cp(0)
            if cmd[:3] == ["git", "rev-parse", "--verify"]:
                # origin/main only becomes resolvable once the fetch has run.
                return _cp(0 if state["fetched"] and cmd[3] == "origin/main" else 1)
            if cmd[:2] == ["git", "rev-parse"]:
                return _cp(1)
            if cmd[:2] == ["git", "merge-base"]:
                return _cp(0)
            raise AssertionError(f"unexpected cmd {cmd!r}")

        with patch.object(quality_gate, "_run", side_effect=_side_effect) as mock_run:
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert len(TestResolveDiffBase._fetch_calls(mock_run)) == 1, "exactly one retry fetch"
        err = capsys.readouterr().err
        assert "fetch skipped" in err, err

    def test_no_infinite_retry_when_the_fetch_cannot_help(self, tmp_path: Path) -> None:
        """The retry is bounded: an unresolvable base still returns None after
        exactly one extra fetch."""
        marker = self._marker(tmp_path, age_seconds=60)
        router = self._router(marker, rev_parse={"origin/main": 1, "main": 1}, upstream=(1, ""))
        with patch.object(quality_gate, "_run", side_effect=router) as mock_run:
            assert quality_gate._resolve_diff_base() is None
        assert len(TestResolveDiffBase._fetch_calls(mock_run)) == 1

    # --- A FAILED fetch must not read as freshness ---

    def test_zero_byte_marker_still_fetches(self, tmp_path: Path) -> None:
        """A failed fetch (offline / dropped VPN / hung remote) still bumps
        `FETCH_HEAD`'s mtime but TRUNCATES it to 0 bytes — verified against real
        git: rc=128, mtime advanced, size 0. Treating that as freshness
        suppressed the QS-276 S1 stale-base warning for the whole 10-minute
        window and never retried, even seconds after connectivity returned. The
        TTL would have turned an existing safety net OFF.
        """
        marker = self._marker(tmp_path, age_seconds=5, content="")
        with patch.object(quality_gate, "_run", side_effect=self._router(marker)) as mock_run:
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert len(TestResolveDiffBase._fetch_calls(mock_run)) == 1

    def test_failed_fetch_warning_is_not_suppressed_on_the_next_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The consequence that actually bites: the S1 warning must keep firing
        run after run while the remote is unreachable."""
        marker = self._marker(tmp_path, age_seconds=5, content="")

        def _side_effect(cmd: list[str], *_a, **_k) -> subprocess.CompletedProcess[str]:
            if cmd[:3] == ["git", "rev-parse", "--git-path"]:
                return _cp(0, stdout=str(marker))
            if cmd[:2] == ["git", "remote"]:
                return _cp(0, stdout="github")
            if cmd[:2] == ["git", "fetch"]:
                return _cp(128, stderr="Could not read from remote repository.")
            if cmd[:3] == ["git", "rev-parse", "--verify"] and cmd[3] == "origin/main":
                return _cp(0)
            if cmd[:2] == ["git", "merge-base"]:
                return _cp(0)
            return _cp(1)

        with patch.object(quality_gate, "_run", side_effect=_side_effect):
            quality_gate._resolve_diff_base()
        assert "git fetch origin main` failed/timed out" in capsys.readouterr().err

    # --- Never skip in CI ---

    def test_ci_bypasses_the_ttl(self, tmp_path: Path) -> None:
        """`actions/checkout` leaves a seconds-old `FETCH_HEAD`, so without an
        explicit bypass a CI `--impacted` would skip its own fetch and resolve
        against whatever base checkout happened to leave. The constant's comment
        already claimed "CI always fetches fresh"; now something enforces it."""
        marker = self._marker(tmp_path, age_seconds=1)
        with (
            patch.object(quality_gate, "_is_ci", return_value=True),
            patch.object(quality_gate, "_run", side_effect=self._router(marker)) as mock_run,
        ):
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert len(TestResolveDiffBase._fetch_calls(mock_run)) == 1

    def test_ci_bypass_does_not_even_probe_the_marker(self, tmp_path: Path) -> None:
        """Cheaper and clearer: under CI the marker is irrelevant, so don't ask."""
        marker = self._marker(tmp_path, age_seconds=1)
        with (
            patch.object(quality_gate, "_is_ci", return_value=True),
            patch.object(quality_gate, "_run", side_effect=self._router(marker)) as mock_run,
        ):
            quality_gate._resolve_diff_base()
        probes = [c.args[0] for c in mock_run.call_args_list if c.args[0][:3] == ["git", "rev-parse", "--git-path"]]
        assert probes == [], probes

    # --- The marker must record a MAIN fetch ---

    def test_marker_recording_another_branch_still_fetches(self, tmp_path: Path) -> None:
        """`FETCH_HEAD` records the last fetch of *anything*. `gh pr checkout
        <n>`, `git fetch origin <other-branch>`, a tag fetch from an editor
        plugin — each rewrites the marker without advancing `origin/main`. So a
        fresh mtime alone cannot support the claim that origin/main is fresh.
        Verified against real git: `git fetch origin other` writes
        `branch 'other' of …`.
        """
        marker = self._marker(
            tmp_path, age_seconds=1, content="cafe1234\t\tbranch 'other' of /tmp/up\n"
        )
        with patch.object(quality_gate, "_run", side_effect=self._router(marker)) as mock_run:
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert len(TestResolveDiffBase._fetch_calls(mock_run)) == 1

    def test_marker_recording_main_among_several_refs_skips(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A bare `git fetch origin` brings main down too and marks the other
        refs `not-for-merge`; that IS a legitimate skip. Verified format:
        `<sha>\\tnot-for-merge\\tbranch 'other' of …`."""
        marker = self._marker(
            tmp_path,
            age_seconds=30,
            content=(
                "aaaa1111\t\tbranch 'main' of github\n"
                "bbbb2222\tnot-for-merge\tbranch 'other' of github\n"
            ),
        )
        with patch.object(quality_gate, "_run", side_effect=self._router(marker)) as mock_run:
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert TestResolveDiffBase._fetch_calls(mock_run) == []
        assert "fetch skipped" in capsys.readouterr().err

    def test_main_only_as_not_for_merge_still_counts(self, tmp_path: Path) -> None:
        """main can appear flagged `not-for-merge` (e.g. fetched while on another
        branch). The ref still came down, so the skip is legitimate."""
        marker = self._marker(
            tmp_path, age_seconds=30, content="aaaa1111\tnot-for-merge\tbranch 'main' of github\n"
        )
        with patch.object(quality_gate, "_run", side_effect=self._router(marker)) as mock_run:
            quality_gate._resolve_diff_base()
        assert TestResolveDiffBase._fetch_calls(mock_run) == []

    def test_unreadable_marker_still_fetches(self, tmp_path: Path) -> None:
        """A directory where the marker should be (or any read error) is unknown."""
        weird = tmp_path / "FETCH_HEAD"
        weird.mkdir()
        with patch.object(quality_gate, "_run", side_effect=self._router(weird)) as mock_run:
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert len(TestResolveDiffBase._fetch_calls(mock_run)) == 1

    # --- A future mtime is unknown, not fresh ---

    def test_future_mtime_still_fetches(self, tmp_path: Path) -> None:
        """`max(0.0, now - mtime)` turned an impossible timestamp into age 0.0,
        so the fetch was skipped for the ENTIRE duration of the skew, not for 10
        minutes. Reproduces on an NTP step, VM suspend/resume, dual boot, or a
        network share whose server clock runs ahead. `None` — the module's
        established "unknown, do not skip" convention — is the right answer."""
        marker = self._marker(tmp_path, age_seconds=-3600)  # one hour in the FUTURE
        with patch.object(quality_gate, "_run", side_effect=self._router(marker)) as mock_run:
            assert quality_gate._resolve_diff_base() == "origin/main"
        assert len(TestResolveDiffBase._fetch_calls(mock_run)) == 1

    # --- No duplicate NH2 warning on the retry ---

    def test_shallow_clone_warning_is_not_duplicated_by_the_retry(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Skip → ladder warns and returns None → fetch → ladder walks again.

        The NH2 "no merge-base with HEAD (shallow clone?)" line must be emitted
        ONCE PER CANDIDATE REF (two here: `origin/main` and `main`) — i.e. only
        on the first walk. Before the fix the retry walked the ladder again and
        printed all of them a second time, for four lines total.
        """
        marker = self._marker(tmp_path, age_seconds=60)
        router = self._router(
            marker,
            rev_parse={"origin/main": 0, "main": 0},
            reachable={"origin/main": 1, "main": 1},
            upstream=(1, ""),
        )
        with patch.object(quality_gate, "_run", side_effect=router):
            assert quality_gate._resolve_diff_base() is None
        err = capsys.readouterr().err
        assert err.count("no merge-base with HEAD") == 2, (
            f"expected one warning per candidate ref on the FIRST walk only; got:\n{err}"
        )


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
            side_effect=subprocess.TimeoutExpired(cmd=["x"], timeout=5.0, output="partial out", stderr="partial err"),
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
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            _patch_early_exit(),
            patch.object(quality_gate, "_resolve_diff_base", return_value=None),
            patch.object(quality_gate, "_is_ci", return_value=True),
        ):
            assert quality_gate.check_impacted() == 4

    def test_no_base_locally_warns_and_passes(self) -> None:
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            _patch_early_exit(),
            patch.object(quality_gate, "_resolve_diff_base", return_value=None),
            patch.object(quality_gate, "_is_ci", return_value=False),
        ):
            assert quality_gate.check_impacted() == 0

    def test_selected_tests_fail_returns_1(self) -> None:
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            _patch_early_exit(),
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

    def test_diff_coverage_below_100_returns_1(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        xml = tmp_path / "coverage.xml"
        absent_db = tmp_path / ".testmondata"  # never created → select-all, no self-heal

        def _emit_xml(*_a: object, **_k: object) -> dict:
            xml.write_text("<coverage/>")  # simulate pytest-cov writing a fresh report
            return {"name": "pytest", "passed": True}

        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            _patch_early_exit(),
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
            _patch_early_exit(),
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
            _patch_early_exit(),
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
            _patch_early_exit(),
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
            _patch_early_exit(),
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
            _patch_early_exit(),
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
            _patch_early_exit(),
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


class TestImpactedEarlyExitPaths:
    """QS-290 (S-4, AC2): `_impacted_early_exit_paths` against REAL git.

    The helper must return a genuine SUPERSET of the paths diff-cover would
    consider, or the early exit becomes a false PASS. It is exercised against
    throwaway repos rather than mocks precisely because the claim under test is
    about git's diff semantics, not about our control flow.

    Zero-arg and total by design: it resolves the base itself. With the base
    resolve left in `check_impacted`, that `_run` call would sit OUTSIDE the
    seam and still fire in every test that patches `_run` wholesale — defeating
    the "patch one seam" remedy the 16 existing call sites rely on.
    """

    @staticmethod
    def _git(repo: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=str(repo), check=True, capture_output=True, text=True
        ).stdout

    @pytest.fixture
    def repo(self, tmp_path: Path) -> Path:
        """A repo on branch `feature`, forked from `main`, with one doc edit."""
        repo = tmp_path / "repo"
        (repo / "docs").mkdir(parents=True)
        (repo / "docs" / "a.md").write_text("base\n")
        (repo / "mod.py").write_text("A = 1\n")
        self._git(repo, "init", "-q", "-b", "main")
        self._git(repo, "config", "user.email", "t@t.co")
        self._git(repo, "config", "user.name", "t")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-qm", "base")
        self._git(repo, "checkout", "-qb", "feature")
        (repo / "docs" / "a.md").write_text("edited\n")
        return repo

    def _paths(self, repo: Path) -> list[str] | None:
        with patch.object(quality_gate, "REPO_ROOT", repo):
            return quality_gate._impacted_early_exit_paths()

    @staticmethod
    def _has_py(paths: list[str] | None) -> bool:
        assert paths is not None
        return any(p.endswith(".py") for p in paths)

    def test_doc_only_change_has_no_python(self, repo: Path) -> None:
        paths = self._paths(repo)
        assert paths == ["docs/a.md"], paths

    def test_untracked_py_defeats_the_exit(self, repo: Path) -> None:
        (repo / "brand_new.py").write_text("def f():\n    return 1\n")
        assert self._has_py(self._paths(repo))

    def test_staged_only_py_defeats_the_exit(self, repo: Path) -> None:
        (repo / "mod.py").write_text("A = 2\n")
        self._git(repo, "add", "mod.py")
        assert self._has_py(self._paths(repo))

    def test_unstaged_only_py_defeats_the_exit(self, repo: Path) -> None:
        (repo / "mod.py").write_text("A = 2\n")
        assert self._has_py(self._paths(repo))

    def test_committed_only_py_defeats_the_exit(self, repo: Path) -> None:
        (repo / "mod.py").write_text("A = 2\n")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-qm", "code")
        assert self._has_py(self._paths(repo))

    def test_deleted_py_defeats_the_exit(self, repo: Path) -> None:
        self._git(repo, "rm", "-q", "mod.py")
        assert self._has_py(self._paths(repo))

    def test_py_matching_mains_content_defeats_the_exit(self, repo: Path) -> None:
        """The merge-base case — the false-PASS hole a two-dot diff would open.

        The branch changes `mod.py`, and main independently lands the SAME
        content (cherry-pick / squash-merge already merged). `git diff
        <main-tip>` sees identical content and lists nothing, while diff-cover's
        three-dot `main...HEAD` range DOES score those lines.
        """
        (repo / "mod.py").write_text("A = 2\n")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-qm", "feature change")
        self._git(repo, "checkout", "-q", "main")
        (repo / "mod.py").write_text("A = 2\n")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-qm", "same content on main")
        self._git(repo, "checkout", "-q", "feature")

        # Contrast: the rejected two-dot form is blind to the .py here.
        assert "mod.py" not in self._git(repo, "diff", "--name-only", "main")
        # The three-dot range diff-cover uses is NOT blind.
        assert "mod.py" in self._git(repo, "diff", "--name-only", "main...HEAD")
        # So neither is our merge-base-based helper.
        assert self._has_py(self._paths(repo))

    def test_branch_behind_main_still_early_exits(self, repo: Path) -> None:
        """The other hole a two-dot diff would open: main's OWN `.py` commits
        would appear in `git diff main`, so a branch even slightly behind main
        could never early-exit — the feature would be dead on arrival."""
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-qm", "doc edit")
        self._git(repo, "checkout", "-q", "main")
        (repo / "other.py").write_text("B = 1\n")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-qm", "main advances")
        self._git(repo, "checkout", "-q", "feature")

        # Contrast: two-dot drags in main's own new .py.
        assert "other.py" in self._git(repo, "diff", "--name-only", "main")
        # Merge-base scoping keeps the change set ours alone.
        assert self._paths(repo) == ["docs/a.md"]

    # --- Core.quotePath must not fail the exit OPEN ---

    NON_ASCII_PY = "tests/test_données.py"

    def test_non_ascii_tracked_py_defeats_the_exit(self, repo: Path) -> None:
        """`core.quotePath` defaults to TRUE, so `git diff --name-only` emits
        `"tests/test_donn\\303\\251es.py"` — C-quoted, WITH surrounding double
        quotes. That string does not end in `.py`, so the only Python file in
        the change set became invisible and the gate returned 0 having run
        NOTHING: a silent fail-OPEN in the one function whose contract is
        fail-closed. NUL-delimited output removes the whole quoting class.
        """
        (repo / "tests").mkdir()
        (repo / self.NON_ASCII_PY).write_text("def f():\n    return 1\n")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-qm", "non-ascii py")
        paths = self._paths(repo)
        assert paths is not None
        assert self.NON_ASCII_PY in paths, paths
        assert not any(p.startswith('"') for p in paths), f"C-quoted path leaked: {paths!r}"
        assert self._has_py(paths)

    def test_non_ascii_untracked_py_defeats_the_exit(self, repo: Path) -> None:
        """`git ls-files --others` quotes too, so the untracked half needs `-z`
        just as much as the diff half."""
        (repo / "tests").mkdir()
        (repo / self.NON_ASCII_PY).write_text("def f():\n    return 1\n")
        paths = self._paths(repo)
        assert paths is not None
        assert self.NON_ASCII_PY in paths, paths
        assert self._has_py(paths)

    def test_quote_path_is_actually_on_in_this_repo(self, repo: Path) -> None:
        """Pin the premise: if git ever changed its default, or the repo set
        `core.quotePath=false`, the two tests above would pass vacuously and
        stop guarding anything."""
        (repo / "tests").mkdir()
        (repo / self.NON_ASCII_PY).write_text("x = 1\n")
        quoted = self._git(repo, "ls-files", "--others", "--exclude-standard")
        assert '\\303\\251' in quoted, (
            f"expected C-quoted output from plain git (proving -z is load-bearing); got {quoted!r}"
        )

    def test_path_with_a_literal_newline_defeats_the_exit(self, repo: Path) -> None:
        """Why `-z` rather than `-c core.quotePath=false`: the latter still
        breaks on a path containing a literal newline, which splits into two
        bogus fields."""
        weird = "tests/we\nird.py"
        (repo / "tests").mkdir()
        (repo / weird).write_text("y = 1\n")
        paths = self._paths(repo)
        assert paths is not None
        assert weird in paths, paths
        assert self._has_py(paths)

    # --- Union the committed range ---

    def test_py_reverted_in_the_worktree_defeats_the_exit(self, repo: Path) -> None:
        """`git diff <mb>` compares the merge-base tree to the WORKING tree, so
        a `.py` changed in a branch COMMIT but restored to merge-base content in
        the worktree vanished from it — while diff-cover's three-dot
        `base...HEAD` still scores that commit's added lines. The exit fired and
        returned 0 on a change set diff-cover would very likely have FAILed.
        """
        (repo / "mod.py").write_text("A = 1\n\n\ndef added():\n    return 2\n")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-qm", "add a function")
        # Restore merge-base content in the worktree only (commit still has it).
        (repo / "mod.py").write_text("A = 1\n")

        # Premise: the working-tree diff alone is blind here...
        assert "mod.py" not in self._git(repo, "diff", "--name-only", "main")
        # ...while the range diff-cover actually uses is not.
        assert "mod.py" in self._git(repo, "diff", "--name-only", "main...HEAD")
        # So the union must see it.
        assert self._has_py(self._paths(repo))

    def test_staged_py_reverted_in_the_worktree_defeats_the_exit(self, repo: Path) -> None:
        """The INDEX is invisible to the other three rungs.

        `git diff <mb>` bypasses the index, `git diff <mb> HEAD` sees only
        commits, and `ls-files --others` only untracked files. So a `.py` whose
        STAGED content differs from the merge-base while its WORKTREE content
        equals it appears in none of them — yet diff-cover scores it
        (`GitDiffReporter` is built with `ignore_staged=False`, so
        `git diff --cached -U0` is in its range) and, worse, it is exactly what
        the `git commit` immediately after this pre-commit gate will land.

        Same trigger class via `git apply --cached`, or `git add -p` followed by
        `git restore --source=HEAD --worktree <file>`.
        """
        mb = self._git(repo, "merge-base", "main", "HEAD").strip()
        (repo / "mod.py").write_text("A = 2\n")
        self._git(repo, "add", "mod.py")
        # Revert the WORKTREE to merge-base content, leaving the index changed —
        # one of the real trigger sequences (`git add -p` then restore).
        self._git(repo, "restore", "--source", mb, "--worktree", "mod.py")
        assert (repo / "mod.py").read_text() == "A = 1\n"

        # Premise: none of the other three rungs can see it.
        assert "mod.py" not in self._git(repo, "diff", "--name-only", mb)
        assert "mod.py" not in self._git(repo, "diff", "--name-only", mb, "HEAD")
        assert "mod.py" not in self._git(repo, "ls-files", "--others", "--exclude-standard")
        # ...while what `git commit` would land plainly contains it.
        assert "mod.py" in self._git(repo, "diff", "--cached", "--name-only", mb)

        paths = self._paths(repo)
        assert paths is not None
        assert "mod.py" in paths, paths
        assert self._has_py(paths)

    def test_committed_range_is_unioned_not_substituted(self, repo: Path) -> None:
        """The committed range is an ADDITION: a worktree-only edit that was
        never committed must still be reported."""
        (repo / "unstaged.py").write_text("Z = 1\n")
        self._git(repo, "add", "unstaged.py")
        paths = self._paths(repo)
        assert paths is not None
        assert "unstaged.py" in paths, paths

    def test_prefers_origin_main_over_local_main(self, repo: Path) -> None:
        """Base ladder rung 1. `origin/main` is created as a real remote-tracking
        ref pointing at the fork point, so a resolvable `origin/main` is used."""
        head = self._git(repo, "rev-parse", "main").strip()
        self._git(repo, "update-ref", "refs/remotes/origin/main", head)
        with patch.object(quality_gate, "_run", wraps=quality_gate._run) as spy:
            self._paths(repo)
        verified = [c.args[0][3] for c in spy.call_args_list if c.args[0][:3] == ["git", "rev-parse", "--verify"]]
        assert verified == ["origin/main"], verified

    @pytest.mark.parametrize(
        "failing",
        [
            ("git", "rev-parse", "--verify"),
            ("git", "merge-base"),
            ("git", "diff", "--name-only", "-z"),
            ("git", "ls-files"),
        ],
        ids=["base", "merge-base", "diff", "ls-files"],
    )
    def test_fail_closed_on_any_git_failure(self, failing: tuple[str, ...]) -> None:
        """AC2: a non-zero return from ANY git call → None → no exit.

        `_get_changed_files` silently drops failed calls; copying that here
        would convert a git failure into a false PASS.

        The two `git diff` rungs are both covered by the `diff` id, which
        matches either invocation; the index rung has its own case below.
        """
        prefix = list(failing)

        def _side_effect(cmd: list[str], *_a, **_k) -> subprocess.CompletedProcess[str]:
            if cmd[: len(prefix)] == prefix:
                return _cp(128, stderr="fatal: git said no")
            if cmd[:2] == ["git", "merge-base"]:
                return _cp(0, stdout="deadbeef\n")
            return _cp(0, stdout="")

        with patch.object(quality_gate, "_run", side_effect=_side_effect):
            assert quality_gate._impacted_early_exit_paths() is None

    def test_fail_closed_when_only_the_committed_range_fails(self) -> None:
        """The NEW call must fail closed on its own,
        not merely be covered by a prefix that also matches the older one."""

        def _side_effect(cmd: list[str], *_a, **_k) -> subprocess.CompletedProcess[str]:
            if cmd[:2] == ["git", "merge-base"]:
                return _cp(0, stdout="deadbeef\n")
            # The committed range is the only diff with a trailing "HEAD".
            if cmd[:2] == ["git", "diff"] and cmd[-1] == "HEAD":
                return _cp(128, stderr="fatal")
            return _cp(0, stdout="")

        with patch.object(quality_gate, "_run", side_effect=_side_effect):
            assert quality_gate._impacted_early_exit_paths() is None

    def test_fail_closed_when_only_the_index_rung_fails(self) -> None:
        """The index rung must fail closed on its own."""

        def _side_effect(cmd: list[str], *_a, **_k) -> subprocess.CompletedProcess[str]:
            if cmd[:2] == ["git", "merge-base"]:
                return _cp(0, stdout="deadbeef\n")
            if cmd[:2] == ["git", "diff"] and "--cached" in cmd:
                return _cp(128, stderr="fatal")
            return _cp(0, stdout="")

        with patch.object(quality_gate, "_run", side_effect=_side_effect):
            assert quality_gate._impacted_early_exit_paths() is None

    def test_all_git_output_is_nul_delimited(self) -> None:
        """Every path-listing call must pass `-z`, or
        the C-quoting fail-open returns through whichever one was missed."""
        seen: list[list[str]] = []

        def _side_effect(cmd: list[str], *_a, **_k) -> subprocess.CompletedProcess[str]:
            seen.append(cmd)
            if cmd[:2] == ["git", "merge-base"]:
                return _cp(0, stdout="deadbeef\n")
            return _cp(0, stdout="")

        with patch.object(quality_gate, "_run", side_effect=_side_effect):
            quality_gate._impacted_early_exit_paths()
        listing = [c for c in seen if c[:2] == ["git", "diff"] or c[:2] == ["git", "ls-files"]]
        assert len(listing) == 4, listing
        for cmd in listing:
            assert "-z" in cmd, f"path-listing call without -z: {cmd!r}"

    def test_empty_merge_base_stdout_fails_closed(self) -> None:
        """A zero-rc `merge-base` with no sha is unusable — do NOT early-exit."""

        def _side_effect(cmd: list[str], *_a, **_k) -> subprocess.CompletedProcess[str]:
            return _cp(0, stdout="")

        with patch.object(quality_gate, "_run", side_effect=_side_effect):
            assert quality_gate._impacted_early_exit_paths() is None


class TestImpactedNonPyEarlyExit:
    """QS-290 (S-4, AC1/AC3/AC4): a non-`.py` `--impacted` run does no work.

    testmon fingerprints AST blocks in `.py` files only — verified with a
    positive control: editing `docs/workflow/overview.md`, the very file a
    test module reads and asserts on at runtime, selects ZERO tests. So
    `--impacted` was ALREADY blind here — provided the baseline is WARM.
    Against a cold `.testmondata` testmon select-alls instead, so the exit is
    gated on warmth; see the cold-baseline tests below.
    """

    @pytest.fixture(autouse=True)
    def _warm_baseline(self, tmp_path_factory: pytest.TempPathFactory):
        """Pin a warm baseline: the exit now requires one.

        Without this the "exit fires" tests inherit the ambient `.testmondata` —
        green locally, red on CI, which has no baseline at all. (Same class of
        environment assumption that broke `test_quick_emits_banner` on CI.)
        Tests exercising the COLD path re-patch `TESTMON_DATA` themselves.
        """
        db = tmp_path_factory.mktemp("warm") / ".testmondata"
        db.write_bytes(b"warm-baseline")
        with patch.object(quality_gate, "TESTMON_DATA", db):
            yield

    def test_returns_0_without_spawning_anything(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            _patch_early_exit(["docs/workflow/project-rules.md", "CLAUDE.md"]),
            patch.object(quality_gate, "_resolve_diff_base") as mock_base,
            patch.object(quality_gate, "_stream_pytest") as mock_stream,
            patch.object(quality_gate, "_run") as mock_run,
        ):
            assert quality_gate.check_impacted() == 0
        mock_base.assert_not_called()  # no base resolve → no git fetch
        mock_stream.assert_not_called()  # no pytest
        mock_run.assert_not_called()  # no diff-cover, no git at all
        err = capsys.readouterr().err
        assert "no Python files changed" in err, err
        assert "--quick tests/qs" in err, err

    def test_no_git_fetch_argv_on_the_exit_path(self) -> None:
        """AC3: assert on the argv, not just on `_resolve_diff_base` — the
        fetch must be unreachable however it is spelled."""
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            _patch_early_exit(["docs/a.md"]),
            patch.object(quality_gate, "_run", return_value=_cp(0)) as mock_run,
            patch.object(quality_gate, "_stream_pytest"),
        ):
            assert quality_gate.check_impacted() == 0
        argvs = [c.args[0] for c in mock_run.call_args_list]
        assert not any(a[:2] == ["git", "fetch"] for a in argvs), argvs

    def test_empty_change_set_also_exits(self) -> None:
        """Nothing changed at all → diff-cover has nothing to score → vacuous."""
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            _patch_early_exit([]),
            patch.object(quality_gate, "_resolve_diff_base") as mock_base,
        ):
            assert quality_gate.check_impacted() == 0
        mock_base.assert_not_called()

    def test_unknown_paths_do_not_exit(self) -> None:
        """`None` (a git failure) must fall through to the full pipeline."""
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            _patch_early_exit(None),
            patch.object(quality_gate, "_resolve_diff_base", return_value=None),
            patch.object(quality_gate, "_is_ci", return_value=True),
        ):
            assert quality_gate.check_impacted() == 4

    @pytest.mark.parametrize("sidecar", ["-wal", "-shm"], ids=["wal", "shm"])
    def test_sidecars_alongside_the_primary_are_not_warm(
        self, tmp_path: Path, sidecar: str
    ) -> None:
        """`st_size > 0` is only a PROXY for "testmon has a usable baseline",
        and it is unsound while sidecars sit next to the primary.

        Reproduced against real testmon 2.2.0 — a `pytest --testmon` killed
        mid-run leaves:

            .testmondata 4096 B   .testmondata-wal 140112   .testmondata-shm 32768

        `PRAGMA user_version` still reads 14, so `_ensure_testmon_db_safe` does
        NOT purge, and the orphan-sidecar branch only fires when the primary is
        ABSENT — which this is not. So the baseline reads "warm" while testmon
        actually select-alls: a non-`.py` change against that DB ran **40
        passed**, versus `no tests ran` against a cleanly seeded one.

        The same sidecar signature covers an in-flight `--seed-testmon` /
        rebuild, whose window is minutes long on a 7 000-test suite.

        Note this test does NOT use the class's `_warm_baseline` fixture shape:
        that writes 13 bytes of non-SQLite, which pins *size*, not usability.
        """
        db = tmp_path / ".testmondata"
        db.write_bytes(b"\x00" * 4096)
        assert quality_gate.__dict__  # sanity: module imported
        with patch.object(quality_gate, "TESTMON_DATA", db):
            assert quality_gate._testmon_baseline_warm() is True  # control
            (tmp_path / f".testmondata{sidecar}").write_bytes(b"x")
            assert quality_gate._testmon_baseline_warm() is False

    def test_sidecar_state_does_not_early_exit(self, tmp_path: Path) -> None:
        """End-to-end: an interrupted/in-flight baseline must fall through to
        the full pass rather than returning a vacuous 0."""
        db = tmp_path / ".testmondata"
        db.write_bytes(b"\x00" * 4096)
        (tmp_path / ".testmondata-wal").write_bytes(b"stale frames")
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            patch.object(quality_gate, "TESTMON_DATA", db),
            _patch_early_exit(["docs/a.md"]),
            patch.object(quality_gate, "_resolve_diff_base", return_value=None),
            patch.object(quality_gate, "_is_ci", return_value=True),
        ):
            assert quality_gate.check_impacted() == 4

    def test_primary_alone_still_early_exits(self, tmp_path: Path) -> None:
        """Negative half: no sidecars → warm → the fast path survives."""
        db = tmp_path / ".testmondata"
        db.write_bytes(b"\x00" * 4096)
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            patch.object(quality_gate, "TESTMON_DATA", db),
            _patch_early_exit(["docs/a.md"]),
            patch.object(quality_gate, "_resolve_diff_base") as mock_base,
        ):
            assert quality_gate.check_impacted() == 0
        mock_base.assert_not_called()

    def test_cold_baseline_does_not_early_exit(self, tmp_path: Path) -> None:
        """A COLD `.testmondata` makes the exit a VERDICT change, not a cost shift.

        The exit assumes testmon "could never select a test" for a non-`.py`
        change set. That holds only for a WARM baseline. Proven against real
        testmon:

            COLD (no .testmondata), note.txt changed -> 1 passed     (select-all)
            WARM,                   note.txt changed -> no tests ran (0 selected)

        So a doc-only change that breaks a doc-pinned guard test exited 1 before
        this PR and would return 0 after — a false PASS, the one defect class a
        quality gate must never ship. This PR's OWN guard tests
        (`test_impacted_non_py_honesty.py`, `TestProjectRulesDocGuards`) are
        precisely the tests a cold-baseline doc-only change would stop running.
        """
        absent = tmp_path / ".testmondata"  # never created → cold
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            patch.object(quality_gate, "TESTMON_DATA", absent),
            _patch_early_exit(["docs/a.md"]),
            patch.object(quality_gate, "_resolve_diff_base", return_value=None),
            patch.object(quality_gate, "_is_ci", return_value=True),
        ):
            # Falls through to the full pipeline; the no-base CI branch proves
            # we got past the seam rather than returning its 0.
            assert quality_gate.check_impacted() == 4

    def test_empty_baseline_does_not_early_exit(self, tmp_path: Path) -> None:
        """A present-but-zero-length `.testmondata` is equally cold."""
        empty = tmp_path / ".testmondata"
        empty.write_bytes(b"")
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            patch.object(quality_gate, "TESTMON_DATA", empty),
            _patch_early_exit(["docs/a.md"]),
            patch.object(quality_gate, "_resolve_diff_base", return_value=None),
            patch.object(quality_gate, "_is_ci", return_value=True),
        ):
            assert quality_gate.check_impacted() == 4

    def test_corrupt_baseline_purged_by_hygiene_does_not_early_exit(
        self, tmp_path: Path
    ) -> None:
        """The warm probe must be read AFTER hygiene.

        A present, non-empty but CORRUPT `.testmondata` looks warm, yet
        `_ensure_testmon_db_safe` purges it — after which testmon select-alls.
        Gating on the pre-hygiene signal alone would leave exactly the hole
        this fix closes.
        """
        db = tmp_path / ".testmondata"
        db.write_bytes(b"not a sqlite database")
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "COVERAGE_DATA", tmp_path / ".coverage"),
            patch.object(quality_gate, "TESTMON_DATA", db),
            _patch_early_exit(["docs/a.md"]),
            patch.object(quality_gate, "_resolve_diff_base", return_value=None),
            patch.object(quality_gate, "_is_ci", return_value=True),
        ):
            # Real hygiene runs, detects corruption, purges → cold → no exit.
            assert quality_gate.check_impacted() == 4
        assert not db.exists(), "the corrupt baseline must have been purged"

    def test_warm_baseline_still_early_exits(self, tmp_path: Path) -> None:
        """The negative half: the fast path must NOT be silently disabled."""
        warm = tmp_path / ".testmondata"
        warm.write_bytes(b"warm-baseline")
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            patch.object(quality_gate, "TESTMON_DATA", warm),
            _patch_early_exit(["docs/a.md"]),
            patch.object(quality_gate, "_resolve_diff_base") as mock_base,
            patch.object(quality_gate, "_stream_pytest") as mock_stream,
        ):
            assert quality_gate.check_impacted() == 0
        mock_base.assert_not_called()
        mock_stream.assert_not_called()

    def test_vanished_baseline_does_not_early_exit(self) -> None:
        """An unreadable / vanished baseline is treated as cold, not warm."""
        fake_db = MagicMock()
        fake_db.stat.side_effect = FileNotFoundError
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            patch.object(quality_gate, "TESTMON_DATA", fake_db),
            _patch_early_exit(["docs/a.md"]),
            patch.object(quality_gate, "_resolve_diff_base", return_value=None),
            patch.object(quality_gate, "_is_ci", return_value=True),
        ):
            assert quality_gate.check_impacted() == 4

    def test_hygiene_runs_before_the_seam(self, tmp_path: Path) -> None:
        """AC3: `_clean_orphan_cov_shards` must be hoisted ABOVE the seam — the
        exit returns before the old seat would ever have been reached, so
        without the hoist an orphan-shard-leaving crash would never be reaped
        on a non-`.py` run."""
        manager = MagicMock()
        # A real path, not a MagicMock: the warm probe now also checks for
        # `-wal`/`-shm` siblings, and every attribute of a MagicMock is truthy.
        warm = tmp_path / ".testmondata"
        warm.write_bytes(b"\x00" * 4096)
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "TESTMON_DATA", warm),
            patch.object(quality_gate, "_clean_orphan_cov_shards") as mock_clean,
            patch.object(quality_gate, "_ensure_testmon_db_safe") as mock_safe,
            patch.object(
                quality_gate, "_impacted_early_exit_paths", return_value=["docs/a.md"]
            ) as mock_seam,
        ):
            manager.attach_mock(mock_clean, "clean")
            manager.attach_mock(mock_safe, "safe")
            manager.attach_mock(mock_seam, "seam")
            assert quality_gate.check_impacted() == 0
        # DB hygiene is hoisted above the seam too: the exit returns before
        # `_run_impacted_pass`, so otherwise a corrupt `.testmondata` is NEVER
        # purged and a developer on a doc-only stretch keeps returning 0 in
        # milliseconds while the DB stays broken.
        assert [c[0] for c in manager.mock_calls] == ["clean", "safe", "seam"], manager.mock_calls

    def test_py_in_change_set_runs_the_pass_with_no_collect_only(self, tmp_path: Path) -> None:
        """AC4: on a change set CONTAINING a `.py` file the exit provably does
        not fire — a testmon pass IS spawned, and no spawned argv contains
        `--collect-only` (the denominator now comes from the run itself)."""
        xml = tmp_path / "coverage.xml"
        base_fake = _fake_popen(run_stdout="..            [ 2/2]\n2 passed in 0.1s\n")

        class fake(base_fake):  # noqa: N801 — a throwaway Popen stand-in
            """Writes coverage.xml on exit, like the real pytest-cov run."""

            calls: list[list[str]] = []

            def wait(self):  # type: ignore[no-untyped-def]
                xml.write_text("<coverage/>")
                return super().wait()

        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            _patch_early_exit(["docs/a.md", "custom_components/quiet_solar/home_model/load.py"]),
            patch.object(quality_gate, "_resolve_diff_base", return_value="origin/main"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            patch.object(quality_gate, "TESTMON_DATA", tmp_path / ".testmondata"),
            patch.object(quality_gate, "_reset_coverage_data"),
            patch.object(quality_gate, "COVERAGE_XML", xml),
            patch.object(quality_gate, "_build_testmon_cmd", return_value=["pytest"]),
            patch.object(quality_gate.subprocess, "Popen", fake),
            patch.object(quality_gate, "_run", return_value=_cp(0, stdout="100%")),
        ):
            assert quality_gate.check_impacted() == 0
        assert len(fake.calls) == 1, f"expected exactly the testmon pass, got {fake.calls!r}"
        assert not any("--collect-only" in c for c in fake.calls), fake.calls


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

    def test_removes_primary_data_and_xdist_shards(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_is_noop_when_no_data_present(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Absent data must not raise (first-ever run)."""
        monkeypatch.setattr(quality_gate, "COVERAGE_DATA", tmp_path / ".coverage")
        quality_gate._reset_coverage_data()  # must not raise


class TestCleanOrphanCovShards:
    """QS-283 A1 (AC#1): `_clean_orphan_cov_shards` reaps only `.coverage.*`
    shards; the combined `.coverage` survives."""

    def test_removes_shard_but_keeps_combined_coverage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        combined = tmp_path / ".coverage"
        combined.write_text("combined")
        shard = tmp_path / ".coverage.host.4242.XYZ"
        shard.write_text("orphan shard")
        monkeypatch.setattr(quality_gate, "COVERAGE_DATA", combined)

        quality_gate._clean_orphan_cov_shards()

        assert combined.exists(), "the combined .coverage must survive (warm baseline)"
        assert not shard.exists(), "a pre-existing orphan shard must be removed"

    def test_is_noop_when_no_shards_present(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        combined = tmp_path / ".coverage"
        combined.write_text("combined")
        monkeypatch.setattr(quality_gate, "COVERAGE_DATA", combined)
        quality_gate._clean_orphan_cov_shards()  # must not raise
        assert combined.exists()

    def test_uses_same_glob_as_reset_coverage_data(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
            _patch_early_exit(),
            patch.object(quality_gate, "_resolve_diff_base", return_value="origin/main"),
            patch.object(quality_gate, "TESTMON_DATA", db),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
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

    def test_incremental_false_fail_recovers_to_pass(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """An incremental changed-line FAIL that is a desync recovers to PASS
        after exactly one rebuild + retry; the self-heal notice is emitted."""
        rc, mock_rebuild, mock_pass = self._run(db=self._incremental_db(tmp_path), verdicts=[self.CHANGED, self.PASS])
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

    def test_testmondata_vanishing_before_stat_is_non_incremental(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Review fix #02: if `.testmondata` is unlinked (concurrent run / other
        worktree / mid-purge) so `TESTMON_DATA.stat()` raises `FileNotFoundError`,
        `check_impacted` must NOT crash — it treats the run as non-incremental
        (was_incremental False), so a changed-line FAIL exits 1 with no retry."""
        fake_db = MagicMock()
        fake_db.stat.side_effect = FileNotFoundError  # vanished between probe and read
        mock_pass = MagicMock(side_effect=[(self.CHANGED, False)])
        with (
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True),
            _patch_early_exit(),
            patch.object(quality_gate, "_resolve_diff_base", return_value="origin/main"),
            patch.object(quality_gate, "TESTMON_DATA", fake_db),
            patch.object(quality_gate, "_clean_orphan_cov_shards"),
            # Hygiene is hoisted into `check_impacted`, so it is no longer
            # covered by the `_run_impacted_pass` mock. Left real, it would
            # `sqlite3.connect(str(MagicMock))` and create a file literally
            # named `<MagicMock id=...>` in the repo root.
            patch.object(quality_gate, "_ensure_testmon_db_safe"),
            patch.object(quality_gate, "_rebuild_testmon_baseline") as mock_rebuild,
            patch.object(quality_gate, "_run_impacted_pass", mock_pass),
        ):
            assert quality_gate.check_impacted() == 1  # must not raise
        # The warm probe is read TWICE now: once pre-hygiene (the QS-283
        # self-heal signal) and once post-hygiene (the early-exit gate). What
        # matters is that a vanishing baseline raises through neither.
        assert fake_db.stat.call_count >= 1
        mock_rebuild.assert_not_called()  # non-incremental → no self-heal retry
        assert mock_pass.call_count == 1
        assert "rebuilding testmon baseline" not in capsys.readouterr().err

    def test_first_pass_success_never_rebuilds_or_emits_notice(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A never-failed PASS is distinguishable from a recovered PASS: no
        rebuild, no self-heal notice."""
        rc, mock_rebuild, mock_pass = self._run(db=self._incremental_db(tmp_path), verdicts=[self.PASS])
        assert rc == 0
        mock_rebuild.assert_not_called()
        assert mock_pass.call_count == 1
        assert "rebuilding testmon baseline" not in capsys.readouterr().err

    def test_non_retriable_verdict_exits_1_without_retry(self, tmp_path: Path) -> None:
        """A non-changed-line failure (e.g. selected tests failed) is genuine
        even on an incremental run — it must not trigger the self-heal."""
        rc, mock_rebuild, mock_pass = self._run(db=self._incremental_db(tmp_path), verdicts=[self.TESTS_FAILED])
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
            _patch_early_exit(),
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
            _patch_early_exit(),
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
        assert "pid" not in marker  # review-fix #04: dropped from completion marker

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
        assert "pid" not in marker  # review-fix #04: dropped from completion marker

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

    def test_best_effort_swallows_and_cleans_up_on_failure(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A write failure must NOT raise (never abort the detached rebuild),
        the temp file must still be unlinked by the `finally`, AND a single
        diagnostic line is emitted (review-fix #02)."""

        def raiser(*_a, **_k):
            raise OSError("boom")

        with patch.object(quality_gate.os, "replace", side_effect=raiser):
            quality_gate._write_seed_status("ok", returncode=0)  # must not raise
        tmp = quality_gate.SEED_STATUS.with_suffix(quality_gate.SEED_STATUS.suffix + ".tmp")
        assert not tmp.exists()
        assert not quality_gate.SEED_STATUS.exists()  # replace never happened
        err = capsys.readouterr().err
        assert "could not write status marker" in err
        assert "boom" in err  # the swallowed exception is surfaced for tracing


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

    # review-fix #04: `_pid_alive` is total — an untrusted pid that makes
    # os.kill raise anything other than PermissionError is treated as dead,
    # never propagating out of the read-only status query.
    @pytest.mark.parametrize("exc", [OverflowError, ValueError, TypeError], ids=["overflow", "value", "type"])
    def test_dead_on_other_os_kill_errors(self, exc: type[Exception]) -> None:
        with patch.object(quality_gate.os, "kill", side_effect=exc):
            assert quality_gate._pid_alive(10**19) is False

    def test_out_of_range_pid_is_dead_no_raise(self) -> None:
        """Real os.kill: a pid beyond pid_t (10**19) raises OverflowError, which
        `_pid_alive` swallows → dead (no patching of the syscall)."""
        assert quality_gate._pid_alive(10**19) is False


class TestFmtSeedTime:
    """QS-286 review-fix #02: epoch marker fields render as readable UTC ISO."""

    def test_numeric_renders_utc_iso(self) -> None:
        # 1_700_000_000 == 2023-11-14T22:13:20+00:00 (UTC, seconds precision).
        assert quality_gate._fmt_seed_time(1_700_000_000) == "2023-11-14T22:13:20+00:00"

    def test_float_renders_seconds_precision(self) -> None:
        out = quality_gate._fmt_seed_time(1_700_000_000.987654)
        assert out == "2023-11-14T22:13:20+00:00"  # sub-second truncated

    @pytest.mark.parametrize("value", [None, "x", True], ids=["none", "str", "bool"])
    def test_non_numeric_is_placeholder(self, value: object) -> None:
        assert quality_gate._fmt_seed_time(value) == "an unknown time"

    # review-fix #03: non-finite / out-of-range epochs must not raise
    # (json.loads accepts Infinity/-Infinity/NaN); they placeholder instead.
    @pytest.mark.parametrize(
        "value",
        [float("inf"), float("-inf"), float("nan"), 1e400, 10**400, -(10**400)],
        ids=["inf", "-inf", "nan", "1e400", "huge-int", "huge-neg-int"],
    )
    def test_non_finite_or_out_of_range_is_placeholder(self, value: object) -> None:
        assert quality_gate._fmt_seed_time(value) == "an unknown time"


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
        out = capsys.readouterr().out
        assert "safe to close" in out.lower()
        # review-fix #02: epoch is rendered as a readable UTC ISO string, not
        # the raw float.
        assert "100.0" not in out
        assert quality_gate._fmt_seed_time(100.0) in out

    def test_running_alive_exits_4(self, capsys: pytest.CaptureFixture[str]) -> None:
        self._write({"state": "running", "pid": 42, "started": 1.0})
        with patch.object(quality_gate, "_pid_alive", return_value=True):
            assert quality_gate.seed_testmon_status() == 4
        out = capsys.readouterr().out
        assert "still running" in out.lower()
        assert quality_gate._fmt_seed_time(1.0) in out  # started rendered readably

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
    def test_non_dict_payload_exits_3(self, payload: object, capsys: pytest.CaptureFixture[str]) -> None:
        self._write(payload)
        assert quality_gate.seed_testmon_status() == 3  # no AttributeError
        assert "unreadable" in capsys.readouterr().out.lower()

    # review-fix #01 must-fix + #02 should-fix: a `running` marker whose pid is
    # not a positive, non-bool int is unreadable → 3, and must never reach the
    # `os.kill` seam. Covers missing/null/str/float pids (would TypeError) AND
    # pid 0 / -1 / bool (would target the process group / all processes and
    # spuriously report "still running").
    @pytest.mark.parametrize(
        "marker",
        [
            {"state": "running"},
            {"state": "running", "pid": None},
            {"state": "running", "pid": "x"},
            {"state": "running", "pid": 1.5},
            {"state": "running", "pid": 0},
            {"state": "running", "pid": -1},
            {"state": "running", "pid": True},
        ],
        ids=["no-pid", "pid-null", "pid-str", "pid-float", "pid-zero", "pid-neg", "pid-bool"],
    )
    def test_running_with_bad_pid_exits_3(self, marker: dict, capsys: pytest.CaptureFixture[str]) -> None:
        """A `running` marker without a positive, non-bool int pid is
        unreadable — never reaches `_pid_alive`/`os.kill`."""
        self._write(marker)
        with (
            patch.object(quality_gate, "_pid_alive") as mock_alive,
            patch.object(quality_gate.os, "kill") as mock_kill,
        ):
            assert quality_gate.seed_testmon_status() == 3  # no TypeError, no false "running"
        mock_alive.assert_not_called()  # bad pid never hits the syscall seam
        mock_kill.assert_not_called()
        assert "unreadable" in capsys.readouterr().out.lower()

    def test_running_out_of_range_pid_interrupted_no_crash(self, capsys: pytest.CaptureFixture[str]) -> None:
        """review-fix #04 must-fix: a positive pid beyond pid_t (10**19) passes
        the positivity guard, reaches the real os.kill (→ OverflowError), and is
        treated as dead → interrupted (exit 1), never crashing the query."""
        self._write({"state": "running", "pid": 10**19, "started": 1.0})
        assert quality_gate.seed_testmon_status() == 1  # no OverflowError
        assert "interrupted" in capsys.readouterr().out.lower()

    # review-fix #01 nice-to-have: a marker missing a display-only field prints
    # a readable placeholder, not literal "None" — and keeps its exit code.
    def test_ok_missing_finished_prints_placeholder(self, capsys: pytest.CaptureFixture[str]) -> None:
        self._write({"state": "ok"})
        assert quality_gate.seed_testmon_status() == 0
        out = capsys.readouterr().out
        assert "None" not in out
        assert "an unknown time" in out

    # review-fix #03 must-fix: a non-finite / out-of-range epoch in a marker
    # must not crash the read-only status query — it placeholders and keeps
    # its exit code. json.dumps emits Infinity/-Infinity/NaN literals that
    # json.loads accepts, mirroring a torn / hand-edited marker.
    @pytest.mark.parametrize(
        "finished",
        [float("inf"), float("-inf"), float("nan"), 10**400],
        ids=["inf", "-inf", "nan", "huge-int"],
    )
    def test_ok_with_bad_epoch_placeholders_no_crash(
        self, finished: object, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._write({"state": "ok", "finished": finished})
        assert quality_gate.seed_testmon_status() == 0  # no OverflowError/ValueError
        out = capsys.readouterr().out
        assert "safe to close" in out.lower()
        assert "an unknown time" in out

    @pytest.mark.parametrize(
        "started",
        [float("inf"), float("-inf"), float("nan"), 10**400],
        ids=["inf", "-inf", "nan", "huge-int"],
    )
    def test_running_with_bad_epoch_placeholders_no_crash(
        self, started: object, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._write({"state": "running", "pid": 42, "started": started})
        with patch.object(quality_gate, "_pid_alive", return_value=True):
            assert quality_gate.seed_testmon_status() == 4  # no crash before exit 4
        out = capsys.readouterr().out
        assert "still running" in out.lower()
        assert "an unknown time" in out

    def test_incomplete_missing_returncode_prints_placeholder(self, capsys: pytest.CaptureFixture[str]) -> None:
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


# ---------------------------------------------------------------------------
# QS-299 — inline follower + last-wins preemption
# ---------------------------------------------------------------------------


import contextlib as _contextlib  # noqa: E402 — local alias for the fake-lock helper


@_contextlib.contextmanager
def _fake_lock(acquired: bool):
    """A stand-in for `quality_gate._seed_lock()` yielding a fixed acquired flag."""
    yield acquired


def _fake_lock_factory(acquired: bool):
    """Return a no-arg callable producing a `_fake_lock(acquired)` context manager."""
    return lambda: _fake_lock(acquired)


class TestReadSeedMarker:
    """QS-299: `_read_seed_marker` — tolerant JSON-object reader (None otherwise)."""

    @pytest.fixture(autouse=True)
    def _redirect(self, tmp_path: Path):
        with patch.object(quality_gate, "SEED_STATUS", tmp_path / ".testmondata.seed-status"):
            yield

    def test_missing_returns_none(self) -> None:
        assert quality_gate._read_seed_marker() is None

    def test_unparseable_returns_none(self) -> None:
        quality_gate.SEED_STATUS.write_text("{not json")
        assert quality_gate._read_seed_marker() is None

    @pytest.mark.parametrize("payload", [5, "x", None, [1], 3.14], ids=["int", "str", "null", "arr", "float"])
    def test_non_dict_returns_none(self, payload: object) -> None:
        quality_gate.SEED_STATUS.write_text(json.dumps(payload))
        assert quality_gate._read_seed_marker() is None

    def test_valid_dict_returned(self) -> None:
        quality_gate.SEED_STATUS.write_text(json.dumps({"state": "ok", "token": "t"}))
        assert quality_gate._read_seed_marker() == {"state": "ok", "token": "t"}

    def test_status_keeps_distinct_missing_vs_unreadable(self, capsys: pytest.CaptureFixture[str]) -> None:
        """AC#7: the extraction must NOT collapse the two distinct messages."""
        assert quality_gate.seed_testmon_status() == 3
        assert "no baseline refresh" in capsys.readouterr().out.lower()
        quality_gate.SEED_STATUS.write_text("{not json")
        assert quality_gate.seed_testmon_status() == 3
        assert "unreadable" in capsys.readouterr().out.lower()


class TestSeedLock:
    """QS-299: `_seed_lock` — best-effort advisory lock (yields acquired flag)."""

    @pytest.fixture(autouse=True)
    def _redirect(self, tmp_path: Path):
        with patch.object(quality_gate, "SEED_STATUS", tmp_path / ".testmondata.seed-status"):
            yield

    def test_lock_path_tracks_seed_status(self) -> None:
        assert quality_gate._seed_lock_path() == quality_gate.SEED_STATUS.with_suffix(
            quality_gate.SEED_STATUS.suffix + ".lock"
        )

    def test_acquires_when_free(self) -> None:
        with quality_gate._seed_lock() as locked:
            assert locked is True

    def test_yields_false_when_flock_busy(self) -> None:
        with (
            patch.object(quality_gate.fcntl, "flock", side_effect=OSError("busy")),
            quality_gate._seed_lock() as locked,
        ):
            assert locked is False

    def test_yields_false_when_lockfile_unopenable(self, tmp_path: Path) -> None:
        bad = tmp_path / "nonexistent-dir" / "x.seed-status"
        with (
            patch.object(quality_gate, "SEED_STATUS", bad),
            quality_gate._seed_lock() as locked,
        ):
            assert locked is False

    def test_releases_and_propagates_body_exception(self) -> None:
        """A body exception must propagate (lock still released via finally)."""
        with pytest.raises(ValueError):  # noqa: PT011
            with quality_gate._seed_lock() as locked:
                assert locked is True
                raise ValueError("boom")
        # Lock is free again — can re-acquire.
        with quality_gate._seed_lock() as locked:
            assert locked is True


class TestDetachSession:
    """QS-299 AC#4: `_detach_session` — setsid + pgid-when-group-leader."""

    def test_records_pgid_on_setsid_success(self) -> None:
        with (
            patch.object(quality_gate.os, "setsid") as mock_setsid,
            patch.object(quality_gate.os, "getpgrp", return_value=4242),
            patch.object(quality_gate.os, "getpid", return_value=4242),
        ):
            assert quality_gate._detach_session() == 4242
        mock_setsid.assert_called_once_with()

    def test_records_pgid_on_setsid_eperm_already_leader(self) -> None:
        """setsid raises EPERM when we already lead a group — still record it."""
        with (
            patch.object(quality_gate.os, "setsid", side_effect=OSError("EPERM")),
            patch.object(quality_gate.os, "getpgrp", return_value=99),
            patch.object(quality_gate.os, "getpid", return_value=99),
        ):
            assert quality_gate._detach_session() == 99

    def test_no_pgid_when_not_group_leader(self) -> None:
        with (
            patch.object(quality_gate.os, "setsid"),
            patch.object(quality_gate.os, "getpgrp", return_value=1),
            patch.object(quality_gate.os, "getpid", return_value=2),
        ):
            assert quality_gate._detach_session() is None


class TestTerminateProcessGroup:
    """QS-299 AC#3: `_terminate_process_group` — killpg-when-pgid else kill; best-effort."""

    def test_killpg_when_pgid(self) -> None:
        with patch.object(quality_gate.os, "killpg") as mock_killpg:
            quality_gate._terminate_process_group(500, 501)
        mock_killpg.assert_called_once_with(500, signal.SIGTERM)

    def test_kill_when_no_pgid(self) -> None:
        with patch.object(quality_gate.os, "kill") as mock_kill:
            quality_gate._terminate_process_group(None, 777)
        mock_kill.assert_called_once_with(777, signal.SIGTERM)

    @pytest.mark.parametrize(
        "exc", [ProcessLookupError, PermissionError, OverflowError, ValueError, TypeError, OSError]
    )
    def test_best_effort_never_raises(self, exc: type[Exception]) -> None:
        with patch.object(quality_gate.os, "killpg", side_effect=exc):
            quality_gate._terminate_process_group(1, 2)  # must not raise
        with patch.object(quality_gate.os, "kill", side_effect=exc):
            quality_gate._terminate_process_group(None, 2)  # must not raise

    def test_emits_one_diagnostic(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(quality_gate.os, "killpg"):
            quality_gate._terminate_process_group(3, 4)
        assert "preempting previous baseline refresh" in capsys.readouterr().err


class TestFindRunningPredecessor:
    """QS-299: `_find_running_predecessor` selection rules."""

    @pytest.fixture(autouse=True)
    def _redirect(self, tmp_path: Path):
        with patch.object(quality_gate, "SEED_STATUS", tmp_path / ".testmondata.seed-status"):
            yield

    def _write(self, marker: object) -> None:
        quality_gate.SEED_STATUS.write_text(json.dumps(marker))

    def test_none_when_no_marker(self) -> None:
        assert quality_gate._find_running_predecessor("me") is None

    def test_live_foreign_running_returns_pid_pgid(self) -> None:
        self._write({"state": "running", "token": "other", "pid": 42, "pgid": 40})
        with patch.object(quality_gate, "_pid_alive", return_value=True):
            assert quality_gate._find_running_predecessor("me") == (42, 40)

    def test_no_pgid_field_returns_pid_none(self) -> None:
        self._write({"state": "running", "token": "other", "pid": 42})
        with patch.object(quality_gate, "_pid_alive", return_value=True):
            assert quality_gate._find_running_predecessor("me") == (42, None)

    def test_same_token_returns_none(self) -> None:
        self._write({"state": "running", "token": "me", "pid": 42})
        with patch.object(quality_gate, "_pid_alive", return_value=True):
            assert quality_gate._find_running_predecessor("me") is None

    def test_dead_pid_returns_none(self) -> None:
        self._write({"state": "running", "token": "other", "pid": 42})
        with patch.object(quality_gate, "_pid_alive", return_value=False):
            assert quality_gate._find_running_predecessor("me") is None

    def test_non_running_state_returns_none(self) -> None:
        self._write({"state": "ok", "token": "other", "pid": 42})
        assert quality_gate._find_running_predecessor("me") is None

    @pytest.mark.parametrize("pid", [None, "x", 0, -1, True, 1.5], ids=["none", "str", "zero", "neg", "bool", "float"])
    def test_bad_pid_returns_none(self, pid: object) -> None:
        self._write({"state": "running", "token": "other", "pid": pid})
        assert quality_gate._find_running_predecessor("me") is None

    @pytest.mark.parametrize("pgid", [0, -1, True, "x", 1.5], ids=["zero", "neg", "bool", "str", "float"])
    def test_bad_pgid_falls_back_to_none(self, pgid: object) -> None:
        self._write({"state": "running", "token": "other", "pid": 42, "pgid": pgid})
        with patch.object(quality_gate, "_pid_alive", return_value=True):
            assert quality_gate._find_running_predecessor("me") == (42, None)


class TestClaimAndPreempt:
    """QS-299 AC#3: read→claim→kill under the lock; lock-fail skips preemption."""

    @pytest.fixture(autouse=True)
    def _redirect(self, tmp_path: Path):
        with patch.object(quality_gate, "SEED_STATUS", tmp_path / ".testmondata.seed-status"):
            yield

    def _marker(self) -> dict:
        return json.loads(quality_gate.SEED_STATUS.read_text())

    def test_claims_and_preempts_in_order(self) -> None:
        order: list[str] = []
        with (
            patch.object(quality_gate, "_seed_lock", _fake_lock_factory(True)),
            patch.object(quality_gate, "_find_running_predecessor", return_value=(42, 40)),
            patch.object(
                quality_gate,
                "_write_seed_status",
                side_effect=lambda *a, **k: order.append("claim"),
            ),
            patch.object(
                quality_gate,
                "_terminate_process_group",
                side_effect=lambda *a, **k: order.append("kill"),
            ) as mock_kill,
        ):
            quality_gate._claim_and_preempt("me", 100, 90, 1.0)
        assert order == ["claim", "kill"], "claim must be written BEFORE the kill"
        mock_kill.assert_called_once_with(40, 42)

    def test_no_predecessor_no_kill(self) -> None:
        with (
            patch.object(quality_gate, "_seed_lock", _fake_lock_factory(True)),
            patch.object(quality_gate, "_find_running_predecessor", return_value=None),
            patch.object(quality_gate, "_terminate_process_group") as mock_kill,
        ):
            quality_gate._claim_and_preempt("me", 100, 90, 1.0)
        mock_kill.assert_not_called()
        marker = self._marker()
        assert marker["state"] == "running" and marker["token"] == "me"
        assert marker["pid"] == 100 and marker["pgid"] == 90

    def test_no_pgid_field_when_pgid_none(self) -> None:
        with (
            patch.object(quality_gate, "_seed_lock", _fake_lock_factory(True)),
            patch.object(quality_gate, "_find_running_predecessor", return_value=None),
        ):
            quality_gate._claim_and_preempt("me", 100, None, 1.0)
        assert "pgid" not in self._marker()

    def test_lock_busy_skips_preemption_but_claims(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch.object(quality_gate, "_seed_lock", _fake_lock_factory(False)),
            patch.object(quality_gate, "_find_running_predecessor") as mock_find,
            patch.object(quality_gate, "_terminate_process_group") as mock_kill,
        ):
            quality_gate._claim_and_preempt("me", 100, 90, 1.0)
        mock_find.assert_not_called()
        mock_kill.assert_not_called()
        assert self._marker()["state"] == "running"
        assert "skipping preemption" in capsys.readouterr().err


class TestWriteCompletionIfOwner:
    """QS-299 AC#5: token-guarded completion write."""

    @pytest.fixture(autouse=True)
    def _redirect(self, tmp_path: Path):
        with patch.object(quality_gate, "SEED_STATUS", tmp_path / ".testmondata.seed-status"):
            yield

    def test_writes_when_token_matches(self) -> None:
        quality_gate.SEED_STATUS.write_text(json.dumps({"state": "running", "token": "me"}))
        with patch.object(quality_gate, "_seed_lock", _fake_lock_factory(True)):
            quality_gate._write_completion_if_owner("me", "ok", returncode=0)
        marker = json.loads(quality_gate.SEED_STATUS.read_text())
        assert marker == {"state": "ok", "token": "me", "returncode": 0}

    def test_skips_when_token_differs(self) -> None:
        quality_gate.SEED_STATUS.write_text(json.dumps({"state": "running", "token": "winner"}))
        with patch.object(quality_gate, "_seed_lock", _fake_lock_factory(True)):
            quality_gate._write_completion_if_owner("me", "ok", returncode=0)
        # Winner's marker untouched.
        assert json.loads(quality_gate.SEED_STATUS.read_text())["token"] == "winner"

    def test_skips_when_marker_missing(self) -> None:
        with patch.object(quality_gate, "_seed_lock", _fake_lock_factory(True)):
            quality_gate._write_completion_if_owner("me", "ok", returncode=0)
        assert not quality_gate.SEED_STATUS.exists()


class TestSeedTestmonTokenPreemption:
    """QS-299: `seed_testmon(token, detached)` wiring — token invariant,
    detach gating, preemption, token-guarded completion."""

    @pytest.fixture(autouse=True)
    def _stub(self, tmp_path: Path):
        with (
            patch.object(quality_gate, "_rebuild_testmon_baseline"),
            patch.object(quality_gate, "SEED_STATUS", tmp_path / ".testmondata.seed-status"),
            patch.object(quality_gate, "_seed_lock", _fake_lock_factory(True)),
        ):
            yield

    def _marker(self) -> dict:
        return json.loads(quality_gate.SEED_STATUS.read_text())

    def test_uses_given_token_end_to_end(self) -> None:
        with (
            patch.object(quality_gate, "_testmon_available", return_value=True),
            patch.object(quality_gate, "_build_seed_testmon_cmd", return_value=["S"]),
            patch.object(quality_gate, "_find_running_predecessor", return_value=None),
            patch.object(quality_gate, "_stream_pytest", return_value={"returncode": 0}),
        ):
            assert quality_gate.seed_testmon(token="tok123") == 0
        assert self._marker()["token"] == "tok123"

    def test_generates_token_when_absent(self) -> None:
        with (
            patch.object(quality_gate, "_testmon_available", return_value=True),
            patch.object(quality_gate, "_build_seed_testmon_cmd", return_value=["S"]),
            patch.object(quality_gate, "_find_running_predecessor", return_value=None),
            patch.object(quality_gate, "_stream_pytest", return_value={"returncode": 0}),
        ):
            assert quality_gate.seed_testmon() == 0
        token = self._marker()["token"]
        assert isinstance(token, str) and len(token) == 32  # uuid4().hex

    def test_skipped_marker_carries_token(self) -> None:
        with patch.object(quality_gate, "_testmon_available", return_value=False):
            assert quality_gate.seed_testmon(token="tokX") == 3
        marker = self._marker()
        assert marker["state"] == "skipped" and marker["token"] == "tokX"

    def test_detached_true_calls_setsid_and_records_pgid(self) -> None:
        with (
            patch.object(quality_gate, "_testmon_available", return_value=True),
            patch.object(quality_gate, "_build_seed_testmon_cmd", return_value=["S"]),
            patch.object(quality_gate, "_find_running_predecessor", return_value=None),
            patch.object(quality_gate, "_detach_session", return_value=555) as mock_detach,
            patch.object(quality_gate, "_stream_pytest", return_value={"returncode": 0}),
        ):
            # capture the running marker mid-pass (completion marker drops pgid).
            seen: dict = {}
            with patch.object(
                quality_gate,
                "_stream_pytest",
                side_effect=lambda *a, **k: (seen.update(self._marker()), {"returncode": 0})[1],
            ):
                assert quality_gate.seed_testmon(token="t", detached=True) == 0
        mock_detach.assert_called_once_with()
        assert seen["pgid"] == 555

    def test_detached_false_skips_setsid(self) -> None:
        with (
            patch.object(quality_gate, "_testmon_available", return_value=True),
            patch.object(quality_gate, "_build_seed_testmon_cmd", return_value=["S"]),
            patch.object(quality_gate, "_find_running_predecessor", return_value=None),
            patch.object(quality_gate, "_detach_session") as mock_detach,
            patch.object(quality_gate, "_stream_pytest", return_value={"returncode": 0}),
        ):
            assert quality_gate.seed_testmon(token="t", detached=False) == 0
        mock_detach.assert_not_called()

    def test_preempts_live_predecessor(self) -> None:
        with (
            patch.object(quality_gate, "_testmon_available", return_value=True),
            patch.object(quality_gate, "_build_seed_testmon_cmd", return_value=["S"]),
            patch.object(quality_gate, "_find_running_predecessor", return_value=(42, 40)),
            patch.object(quality_gate, "_terminate_process_group") as mock_kill,
            patch.object(quality_gate, "_stream_pytest", return_value={"returncode": 0}),
        ):
            assert quality_gate.seed_testmon(token="t2") == 0
        mock_kill.assert_called_once_with(40, 42)

    def test_completion_write_is_token_guarded(self) -> None:
        """AC#5: a preempted run (marker flipped to a foreign token during the
        pass) must NOT clobber the winner's marker on completion."""
        with (
            patch.object(quality_gate, "_testmon_available", return_value=True),
            patch.object(quality_gate, "_build_seed_testmon_cmd", return_value=["S"]),
            patch.object(quality_gate, "_find_running_predecessor", return_value=None),
            patch.object(
                quality_gate,
                "_stream_pytest",
                # a fresher seed claims the marker while we run.
                side_effect=lambda *a, **k: (
                    quality_gate.SEED_STATUS.write_text(json.dumps({"state": "running", "token": "winner"})),
                    {"returncode": 0},
                )[1],
            ),
        ):
            assert quality_gate.seed_testmon(token="loser") == 0
        assert json.loads(quality_gate.SEED_STATUS.read_text())["token"] == "winner"


class TestParseSeedProgress:
    """QS-299 AC#2: last-match parse of the live `_stream_pytest` progress format."""

    def test_parses_live_captured_format(self) -> None:
        # The exact line _stream_pytest synthesizes under -n auto (captured live).
        line = "  pytest: 21% (72/330) | passed=72 failed=0 errors=0\n"
        assert quality_gate._parse_seed_progress(line) == (21, 72, 330)

    def test_takes_last_match(self) -> None:
        log = (
            "  pytest: 21% (72/330) | passed=72 failed=0 errors=0\n"
            "  pytest: 65% (216/330) | passed=216 failed=0 errors=0\n"
        )
        assert quality_gate._parse_seed_progress(log) == (65, 216, 330)

    def test_does_not_match_done_line(self) -> None:
        assert quality_gate._parse_seed_progress("  pytest: done (330/330) | passed=329\n") is None

    def test_none_when_empty(self) -> None:
        assert quality_gate._parse_seed_progress("") is None

    def test_production_format_still_parses(self) -> None:
        """Regression tripwire: if `_stream_pytest`'s format drifts, this breaks."""
        # Reconstruct the exact production f-string.
        pct, current, total, lp, lf, le = 43, 144, 330, 144, 0, 0
        line = f"  pytest: {pct}% ({current}/{total}) | passed={lp} failed={lf} errors={le}\n"
        assert quality_gate._parse_seed_progress(line) == (43, 144, 330)


class TestFmtElapsed:
    """QS-299: `_fmt_elapsed` renders mm:ss."""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0, "00:00"), (5, "00:05"), (65, "01:05"), (3661, "61:01"), (-3, "00:00")],
    )
    def test_formats(self, seconds: float, expected: str) -> None:
        assert quality_gate._fmt_elapsed(seconds) == expected


class TestMarkerElapsed:
    """QS-299 AC#2: wall-clock elapsed from `started`."""

    def test_now_minus_started(self) -> None:
        with patch.object(quality_gate.time, "time", return_value=100.0):
            assert quality_gate._marker_elapsed({"started": 40.0}) == 60.0

    @pytest.mark.parametrize(
        "started",
        [None, "x", True, float("inf"), float("nan"), 10**400],
        ids=["none", "str", "bool", "inf", "nan", "huge"],
    )
    def test_bad_started_is_zero(self, started: object) -> None:
        with patch.object(quality_gate.time, "time", return_value=100.0):
            assert quality_gate._marker_elapsed({"started": started}) == 0.0

    def test_never_negative(self) -> None:
        with patch.object(quality_gate.time, "time", return_value=10.0):
            assert quality_gate._marker_elapsed({"started": 40.0}) == 0.0


class TestPrintSeedProgress:
    """QS-299 AC#2: progress line format + degraded fallback."""

    def test_full_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch.object(quality_gate, "_read_seed_log_tail", return_value="  pytest: 21% (72/330) | x\n"),
            patch.object(quality_gate, "_marker_elapsed", return_value=65.0),
        ):
            quality_gate._print_seed_progress({"started": 1.0})
        out = capsys.readouterr().out
        # QS-299 review-fix #01 (finding #2): plain-ASCII separator (no `·`).
        assert "refreshing baseline: 72/330 tests (21%) - elapsed 01:05" in out
        assert "·" not in out

    def test_degraded_when_unparseable(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch.object(quality_gate, "_read_seed_log_tail", return_value="no progress here"),
            patch.object(quality_gate, "_marker_elapsed", return_value=5.0),
        ):
            quality_gate._print_seed_progress({"started": 1.0})
        out = capsys.readouterr().out
        assert "refreshing baseline: elapsed 00:05" in out
        assert "tests" not in out


class TestReadSeedLogTail:
    """QS-299: `_read_seed_log_tail` — best-effort TAIL read (review-fix #01 #5)."""

    def test_reads_content(self, tmp_path: Path) -> None:
        log = tmp_path / "seed.log"
        log.write_text("hello")
        with patch.object(quality_gate, "SEED_LOG", log):
            assert quality_gate._read_seed_log_tail() == "hello"

    def test_missing_returns_empty(self, tmp_path: Path) -> None:
        with patch.object(quality_gate, "SEED_LOG", tmp_path / "nope.log"):
            assert quality_gate._read_seed_log_tail() == ""

    def test_reads_only_the_tail(self, tmp_path: Path) -> None:
        """A large log is not read whole: only ~`_SEED_LOG_TAIL_BYTES` come back,
        and the freshest progress line (at the end) survives + parses."""
        log = tmp_path / "seed.log"
        head = "OLD 0% (0/330)\n" * 5000  # far bigger than the tail window
        tail_line = "  pytest: 99% (327/330) | passed=327 failed=0 errors=0\n"
        log.write_text(head + tail_line)
        with patch.object(quality_gate, "SEED_LOG", log):
            out = quality_gate._read_seed_log_tail()
        assert len(out) <= quality_gate._SEED_LOG_TAIL_BYTES
        assert len(out) < len(head)  # NOT the whole file
        assert quality_gate._parse_seed_progress(out) == (99, 327, 330)

    def test_tail_smaller_than_window_returns_all(self, tmp_path: Path) -> None:
        log = tmp_path / "seed.log"
        log.write_text("short content")
        with patch.object(quality_gate, "SEED_LOG", log):
            assert quality_gate._read_seed_log_tail() == "short content"


class TestSeedTestmonFollow:
    """QS-299 AC#1/#2: the inline follower's exit table + progress line."""

    @pytest.fixture(autouse=True)
    def _seams(self, tmp_path: Path):
        """Redirect SEED_STATUS + neutralize sleep; each test drives the clock
        and marker reads via its own patches."""
        with (
            patch.object(quality_gate, "SEED_STATUS", tmp_path / ".testmondata.seed-status"),
            patch.object(quality_gate.time, "sleep"),
            patch.object(quality_gate.time, "monotonic", return_value=0.0),
        ):
            yield

    def test_ok_exits_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(quality_gate, "_read_seed_marker", return_value={"token": "t", "state": "ok"}):
            assert quality_gate.seed_testmon_follow("t") == 0
        out = capsys.readouterr().out.lower()
        assert "safe to close" in out
        assert out.isascii()  # review-fix #01 (#2): no non-ASCII glyphs

    def test_output_is_ascii_only(self) -> None:
        """review-fix #01 (finding #2): every verdict/progress line is plain
        ASCII, so piping under a non-UTF-8 locale can never raise."""
        markers = [
            {"token": "t", "state": "ok"},
            {"token": "t", "state": "incomplete"},
            {"token": "t", "state": "skipped", "reason": "x"},
            {"token": "other", "state": "running"},
        ]
        for m in markers:
            with patch.object(quality_gate, "_read_seed_marker", return_value=m):
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    quality_gate.seed_testmon_follow("t")
                assert buf.getvalue().isascii(), m

    def test_foreign_marker_past_grace_exits_5_unconfirmed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """review-fix #01/#02 (findings #1/#3): a foreign token that persists the
        whole startup grace (our seed never claims) → exit 5 with the SOFTENED
        "could not confirm" message (it may just be a slow cold-start), not a
        definite supersession."""
        with patch.object(quality_gate, "_read_seed_marker", return_value={"token": "other", "state": "running"}):
            assert quality_gate.seed_testmon_follow("t") == 5
        out = capsys.readouterr().out.lower()
        assert "waiting for baseline refresh" in out  # grace polled first
        assert "could not confirm" in out and "--seed-testmon-status" in out
        assert "safe to close" in out
        assert out.isascii()

    def test_stale_foreign_marker_then_own_claim_streams_progress(self, capsys: pytest.CaptureFixture[str]) -> None:
        """review-fix #01 (finding #1) MUST-FIX: a stale foreign marker present at
        startup must NOT trigger a spurious exit 5 — once our own seed claims the
        marker within the grace window we stream our own progress to completion."""
        stale = {"token": "prev-task", "state": "running", "pid": 7}
        mine_running = {"token": "t", "state": "running", "pid": 42, "started": 1.0}
        mine_ok = {"token": "t", "state": "ok"}
        with (
            patch.object(
                quality_gate,
                "_read_seed_marker",
                side_effect=[stale, stale, mine_running, mine_ok],
            ),
            patch.object(quality_gate, "_pid_alive", return_value=True),
            patch.object(quality_gate, "_print_seed_progress") as mock_prog,
        ):
            assert quality_gate.seed_testmon_follow("t") == 0
        out = capsys.readouterr().out.lower()
        assert "waiting for baseline refresh" in out  # tolerated the stale marker
        assert "fresher baseline" not in out  # no spurious supersession
        mock_prog.assert_called_once()  # streamed our own progress
        assert "safe to close" in out

    def test_own_token_then_foreign_reports_taken_over(self, capsys: pytest.CaptureFixture[str]) -> None:
        """review-fix #03 (finding #1): our token owned the marker (`running(T)`),
        then a foreign token owns it (`running(U)`) — the reachable production
        ordering `running(T) → ok(T) → running(U)` OR `running(T) → running(U)`.
        The token-guarded single marker means `ok(T)` is unrecoverable, so the
        follower reports HONEST uncertainty (exit 5), never a false "was
        stopped"/"fresher" claim, and never a fabricated completion."""
        mine = {"token": "t", "state": "running", "pid": 42, "started": 1.0}
        foreign = {"token": "U", "state": "running", "pid": 99}
        with (
            # only 2 reads now — the dead recovery re-read was removed.
            patch.object(quality_gate, "_read_seed_marker", side_effect=[mine, foreign]),
            patch.object(quality_gate, "_pid_alive", return_value=True),
            patch.object(quality_gate, "_print_seed_progress"),
        ):
            assert quality_gate.seed_testmon_follow("t") == 5
        out = capsys.readouterr().out.lower()
        assert "another baseline refresh now holds the marker" in out
        assert "either completed or was superseded" in out
        assert "--seed-testmon-status" in out and "safe to close" in out
        # honest — neither a false "stopped" nor a fabricated completion:
        assert "was stopped" not in out
        assert "[ok]" not in out
        assert out.isascii()

    def test_incomplete_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(quality_gate, "_read_seed_marker", return_value={"token": "t", "state": "incomplete"}):
            assert quality_gate.seed_testmon_follow("t") == 1
        assert "may be partial" in capsys.readouterr().out.lower()

    def test_skipped_exits_1_benign(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(
            quality_gate,
            "_read_seed_marker",
            return_value={"token": "t", "state": "skipped", "reason": "no testmon"},
        ):
            assert quality_gate.seed_testmon_follow("t") == 1
        out = capsys.readouterr().out.lower()
        assert "skipped" in out and "no baseline was written" in out and "safe to close" in out

    def test_unknown_state_exits_3(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(quality_gate, "_read_seed_marker", return_value={"token": "t", "state": "bogus"}):
            assert quality_gate.seed_testmon_follow("t") == 3
        assert "unreadable" in capsys.readouterr().out.lower()

    def test_running_alive_prints_progress_then_completes(self, capsys: pytest.CaptureFixture[str]) -> None:
        markers = [
            {"token": "t", "state": "running", "pid": 42, "started": 1.0},
            {"token": "t", "state": "ok"},
        ]
        with (
            patch.object(quality_gate, "_read_seed_marker", side_effect=markers),
            patch.object(quality_gate, "_pid_alive", return_value=True),
            patch.object(quality_gate, "_print_seed_progress") as mock_prog,
        ):
            assert quality_gate.seed_testmon_follow("t") == 0
        mock_prog.assert_called_once()
        assert "safe to close" in capsys.readouterr().out.lower()

    def test_running_past_max_wait_exits_4(self, capsys: pytest.CaptureFixture[str]) -> None:
        # deadline = monotonic()[0] + MAX_WAIT; make the in-loop check exceed it.
        with (
            patch.object(
                quality_gate.time,
                "monotonic",
                side_effect=[0.0, quality_gate._SEED_FOLLOW_MAX_WAIT_S + 1],
            ),
            patch.object(
                quality_gate,
                "_read_seed_marker",
                return_value={"token": "t", "state": "running", "pid": 42, "started": 1.0},
            ),
            patch.object(quality_gate, "_pid_alive", return_value=True),
            patch.object(quality_gate, "_print_seed_progress"),
        ):
            assert quality_gate.seed_testmon_follow("t") == 4
        out = capsys.readouterr().out.lower()
        # review-fix #05 (finding #4): the minutes are derived from the constant.
        mins = quality_gate._SEED_FOLLOW_MAX_WAIT_S // 60
        assert f"still running after {mins}m" in out and "--seed-token t" in out

    def test_running_dead_confirmed_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        """R1 re-poll: dead pid confirmed on re-read → interrupted."""
        dead = {"token": "t", "state": "running", "pid": 42, "started": 1.0}
        with (
            patch.object(quality_gate, "_read_seed_marker", side_effect=[dead, dead]),
            patch.object(quality_gate, "_pid_alive", return_value=False),
        ):
            assert quality_gate.seed_testmon_follow("t") == 1
        assert "may be partial" in capsys.readouterr().out.lower()

    def test_running_dead_then_ok_recovers(self, capsys: pytest.CaptureFixture[str]) -> None:
        """R1 re-poll: marker flips to ok before we conclude interrupted (the
        re-poll re-read + the top-of-loop re-classify both read the marker)."""
        dead = {"token": "t", "state": "running", "pid": 42, "started": 1.0}
        ok = {"token": "t", "state": "ok"}
        with (
            patch.object(quality_gate, "_read_seed_marker", side_effect=[dead, ok, ok]),
            patch.object(quality_gate, "_pid_alive", return_value=False),
        ):
            assert quality_gate.seed_testmon_follow("t") == 0
        assert "safe to close" in capsys.readouterr().out.lower()

    def test_running_dead_then_superseded(self) -> None:
        """R1 re-poll: token flips (superseded) → exit 5. Reads: initial dead,
        R1 re-read (new), loop-top (new) → own-token→foreign taken-over verdict."""
        dead = {"token": "t", "state": "running", "pid": 42, "started": 1.0}
        new = {"token": "new", "state": "running"}
        with (
            patch.object(quality_gate, "_read_seed_marker", side_effect=[dead, new, new]),
            patch.object(quality_gate, "_pid_alive", return_value=False),
        ):
            assert quality_gate.seed_testmon_follow("t") == 5

    def test_running_missing_pid_treated_dead(self, capsys: pytest.CaptureFixture[str]) -> None:
        no_pid = {"token": "t", "state": "running", "started": 1.0}
        with (
            patch.object(quality_gate, "_read_seed_marker", side_effect=[no_pid, no_pid]),
            patch.object(quality_gate, "_pid_alive") as mock_alive,
        ):
            assert quality_gate.seed_testmon_follow("t") == 1
        # a missing pid never reaches the syscall seam in the first observation.
        mock_alive.assert_not_called()

    def test_repoll_running_alive_loops_then_completes(self, capsys: pytest.CaptureFixture[str]) -> None:
        """review-fix #02 (finding #9): the R1 re-poll branch where the re-read is
        running + my token + pid ALIVE falls through to `continue`; the follower
        loops once more and then exits 0 on a terminal `ok`."""
        running = {"token": "t", "state": "running", "pid": 42, "started": 1.0}
        ok = {"token": "t", "state": "ok"}
        with (
            patch.object(quality_gate, "_read_seed_marker", side_effect=[running, running, ok]),
            # initial observation dead → R1 re-poll; re-read alive → continue.
            patch.object(quality_gate, "_pid_alive", side_effect=[False, True]),
            patch.object(quality_gate, "_print_seed_progress"),
        ):
            assert quality_gate.seed_testmon_follow("t") == 0
        assert "safe to close" in capsys.readouterr().out.lower()

    def test_deadline_enforced_on_all_paths_exits_4(self, capsys: pytest.CaptureFixture[str]) -> None:
        """review-fix #02 (finding #4): the max-wait deadline is checked at the TOP
        of the loop, so even a flapping marker (None <-> running) that never
        reaches a terminal state eventually returns exit 4."""
        running = {"token": "t", "state": "running", "pid": 42, "started": 1.0}
        big = quality_gate._SEED_FOLLOW_MAX_WAIT_S + 1
        with (
            # deadline calc(0), iter1 top(0), iter2 top(0), iter3 top(big)->exit 4
            patch.object(quality_gate.time, "monotonic", side_effect=[0.0, 0.0, 0.0, big]),
            # iter1: running(alive) → progress; iter2: None(owned) → re-read running → continue
            patch.object(quality_gate, "_read_seed_marker", side_effect=[running, None, running]),
            patch.object(quality_gate, "_pid_alive", return_value=True),
            patch.object(quality_gate, "_print_seed_progress"),
        ):
            assert quality_gate.seed_testmon_follow("t") == 4
        mins = quality_gate._SEED_FOLLOW_MAX_WAIT_S // 60
        assert f"still running after {mins}m" in capsys.readouterr().out.lower()

    def test_startup_grace_then_exit_3(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(quality_gate, "_read_seed_marker", return_value=None):
            # SEED_STATUS never created → missing → grace exhausts → exit 3.
            assert quality_gate.seed_testmon_follow("t") == 3
        out = capsys.readouterr().out.lower()
        assert "waiting for baseline refresh" in out and "unreadable" in out

    def test_startup_grace_then_marker_appears(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(quality_gate, "_read_seed_marker", side_effect=[None, {"token": "t", "state": "ok"}]):
            assert quality_gate.seed_testmon_follow("t") == 0
        out = capsys.readouterr().out.lower()
        assert "waiting for baseline refresh" in out and "safe to close" in out

    def test_present_but_unreadable_reread_exits_3(self, capsys: pytest.CaptureFixture[str]) -> None:
        # marker file exists (so not startup grace) but reads as None twice.
        quality_gate.SEED_STATUS.write_text("{not json")
        with patch.object(quality_gate, "_read_seed_marker", return_value=None):
            assert quality_gate.seed_testmon_follow("t") == 3
        assert "unreadable" in capsys.readouterr().out.lower()

    def test_present_but_unreadable_then_recovers(self) -> None:
        quality_gate.SEED_STATUS.write_text("{not json")
        ok = {"token": "t", "state": "ok"}
        # None (present→re-read), non-None on re-read, then re-classify at loop top.
        with patch.object(quality_gate, "_read_seed_marker", side_effect=[None, ok, ok]):
            assert quality_gate.seed_testmon_follow("t") == 0

    def test_is_read_only(self) -> None:
        """AC#1: no pytest/coverage/testmon seam is touched."""
        with (
            patch.object(quality_gate, "_read_seed_marker", return_value={"token": "t", "state": "ok"}),
            patch.object(quality_gate, "_stream_pytest") as mock_stream,
            patch.object(quality_gate, "_testmon_available") as mock_probe,
            patch.object(quality_gate, "_rebuild_testmon_baseline") as mock_rebuild,
        ):
            quality_gate.seed_testmon_follow("t")
        mock_stream.assert_not_called()
        mock_probe.assert_not_called()
        mock_rebuild.assert_not_called()


class TestSeedFollowCli:
    """QS-299 AC#6: CLI wiring for --seed-testmon-follow/--seed-token/--detached."""

    def test_follow_passthrough(self) -> None:
        with (
            patch("sys.argv", ["quality_gate.py", "--seed-testmon-follow", "--seed-token", "T"]),
            patch.object(quality_gate, "seed_testmon_follow", return_value=5) as mock_follow,
            patch.object(quality_gate, "_detect_scope") as mock_scope,
            pytest.raises(SystemExit) as exc,
        ):
            quality_gate.main()
        assert exc.value.code == 5
        mock_follow.assert_called_once_with("T")
        mock_scope.assert_not_called()

    def test_seed_testmon_threads_token_and_detached(self) -> None:
        with (
            patch("sys.argv", ["quality_gate.py", "--seed-testmon", "--detached", "--seed-token", "T"]),
            patch.object(quality_gate, "seed_testmon", return_value=0) as mock_seed,
            pytest.raises(SystemExit),
        ):
            quality_gate.main()
        mock_seed.assert_called_once_with(token="T", detached=True)

    def test_padded_token_is_trimmed_seed(self) -> None:
        """review-fix #04 (finding #4): a token with surrounding whitespace is
        accepted and normalized (trimmed) before dispatch."""
        with (
            patch("sys.argv", ["quality_gate.py", "--seed-testmon", "--seed-token", "  abc  "]),
            patch.object(quality_gate, "seed_testmon", return_value=0) as mock_seed,
            pytest.raises(SystemExit),
        ):
            quality_gate.main()
        mock_seed.assert_called_once_with(token="abc", detached=False)

    def test_padded_token_is_trimmed_follow(self) -> None:
        """review-fix #04 (finding #4): the follower receives the trimmed token so
        it compares strictly identical to the seed's."""
        with (
            patch("sys.argv", ["quality_gate.py", "--seed-testmon-follow", "--seed-token", "\tabc\n"]),
            patch.object(quality_gate, "seed_testmon_follow", return_value=0) as mock_follow,
            pytest.raises(SystemExit),
        ):
            quality_gate.main()
        mock_follow.assert_called_once_with("abc")

    def test_follow_requires_token(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("sys.argv", ["quality_gate.py", "--seed-testmon-follow"]),
            pytest.raises(SystemExit) as exc,
        ):
            quality_gate.main()
        assert exc.value.code == 2
        assert "--seed-testmon-follow requires --seed-token" in capsys.readouterr().err

    @pytest.mark.parametrize("blank", ["", "   ", "\t"], ids=["empty", "spaces", "tab"])
    @pytest.mark.parametrize(
        "mode",
        [["--seed-testmon", "--detached"], ["--seed-testmon-follow"]],
        ids=["seed", "follow"],
    )
    def test_empty_or_whitespace_token_exits_2(
        self, blank: str, mode: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """review-fix #01 (finding #3): a blank --seed-token is rejected for BOTH
        subcommands (never silently treated as absent vs literal)."""
        with (
            patch("sys.argv", ["quality_gate.py", *mode, "--seed-token", blank]),
            patch.object(quality_gate, "seed_testmon") as mock_seed,
            patch.object(quality_gate, "seed_testmon_follow") as mock_follow,
            pytest.raises(SystemExit) as exc,
        ):
            quality_gate.main()
        assert exc.value.code == 2
        assert "--seed-token must not be empty or whitespace" in capsys.readouterr().err
        mock_seed.assert_not_called()
        mock_follow.assert_not_called()

    @pytest.mark.parametrize(
        "argv",
        [
            ["--seed-token", "T"],
            ["--seed-token", "T", "--impacted"],
            ["--detached"],
            ["--detached", "--seed-testmon-follow", "--seed-token", "T"],
        ],
        ids=["stray-token", "token-with-impacted", "stray-detached", "detached-with-follow"],
    )
    def test_stray_token_or_detached_exits_2(self, argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("sys.argv", ["quality_gate.py", *argv]),
            pytest.raises(SystemExit) as exc,
        ):
            quality_gate.main()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "only valid with" in err

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
    def test_follow_mutex_exits_2(self, conflict: list[str], capsys: pytest.CaptureFixture[str]) -> None:
        """--seed-testmon-follow vs an execution mode (seed-mode pairs are covered
        by test_seed_modes_pairwise_mutex_exits_2)."""
        with (
            patch("sys.argv", ["quality_gate.py", "--seed-testmon-follow", "--seed-token", "T", *conflict]),
            patch.object(quality_gate, "seed_testmon_follow") as mock_follow,
            pytest.raises(SystemExit) as exc,
        ):
            quality_gate.main()
        assert exc.value.code == 2
        assert "you cannot combine --seed-testmon-follow with" in capsys.readouterr().err
        mock_follow.assert_not_called()

    @pytest.mark.parametrize(
        "pair",
        [
            ["--seed-testmon", "--seed-testmon-status"],
            ["--seed-testmon", "--seed-testmon-follow", "--seed-token", "T"],
            ["--seed-testmon-status", "--seed-testmon-follow", "--seed-token", "T"],
            # reversed order — the centralized check is order-independent.
            ["--seed-testmon-status", "--seed-testmon"],
            ["--seed-testmon-follow", "--seed-token", "T", "--seed-testmon"],
            ["--seed-testmon-follow", "--seed-token", "T", "--seed-testmon-status"],
        ],
        ids=["seed+status", "seed+follow", "status+follow", "status+seed", "follow+seed", "follow+status"],
    )
    def test_seed_modes_pairwise_mutex_exits_2(self, pair: list[str], capsys: pytest.CaptureFixture[str]) -> None:
        """review-fix #02 (finding #2): the three seed subcommands are pairwise
        mutually exclusive via ONE centralized, order-independent check — neither
        the seed side nor the follow side silently wins."""
        with (
            patch("sys.argv", ["quality_gate.py", *pair]),
            patch.object(quality_gate, "seed_testmon") as mock_seed,
            patch.object(quality_gate, "seed_testmon_status") as mock_status,
            patch.object(quality_gate, "seed_testmon_follow") as mock_follow,
            pytest.raises(SystemExit) as exc,
        ):
            quality_gate.main()
        assert exc.value.code == 2
        assert "mutually exclusive" in capsys.readouterr().err
        mock_seed.assert_not_called()
        mock_status.assert_not_called()
        mock_follow.assert_not_called()


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
    def test_seed_testmon_mutex_exits_2(self, conflict: list[str], capsys: pytest.CaptureFixture[str]) -> None:
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
        # QS-299: token + detached are threaded from the CLI (defaults here).
        mock_seed.assert_called_once_with(token=None, detached=False)
        mock_scope.assert_not_called()

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
    def test_seed_testmon_status_mutex_exits_2(self, conflict: list[str], capsys: pytest.CaptureFixture[str]) -> None:
        """AC#5: --seed-testmon-status combined with an execution mode is a usage
        error (seed-mode pairs → test_seed_modes_pairwise_mutex_exits_2)."""
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

    def test_seed_status_path_is_repo_root_relative(self) -> None:
        """QS-286 AC#6 (review-fix #04): the marker constant is anchored under
        REPO_ROOT (`__file__`-relative, cwd-independent), mirroring TESTMON_DATA
        — so the detached run and a later status query resolve the same file."""
        assert quality_gate.SEED_STATUS == quality_gate.REPO_ROOT / ".testmondata.seed-status"
        assert quality_gate.SEED_STATUS.parent == quality_gate.REPO_ROOT
        assert quality_gate.SEED_STATUS.is_absolute()
        # Shares the main worktree's .testmondata directory (same relocation
        # invariant as the testmon DB).
        assert quality_gate.SEED_STATUS.parent == quality_gate.TESTMON_DATA.parent


class TestProjectRulesDocGuards:
    """review-fix N2: content guard for the AC#12 doc edits (not just drift-checker)."""

    def _rules(self) -> str:
        return (Path(__file__).resolve().parent.parent / "docs" / "workflow" / "project-rules.md").read_text()

    def _rules_flat(self) -> str:
        """`project-rules.md` with runs of whitespace collapsed to single spaces.

        Review-fix #01 lesson (same root cause as nice-to-have 21): the doc is
        hard-wrapped, so where a phrase happens to break across lines is an
        accident of formatting and must not be part of a guard's contract.
        Phrase guards run against this; guards for runnable single-line commands
        keep using the raw text.
        """
        return " ".join(self._rules().split())

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

    def test_phase_protocols_step5_completion_signal_present(self) -> None:
        """review-fix #04 (AC#9): pin the phase-protocols.md step-5 completion
        -signal note, symmetric to the project-rules guard above so a drift /
        revert of the step-5 line is caught."""
        proto = (Path(__file__).resolve().parent.parent / "docs" / "workflow" / "phase-protocols.md").read_text()
        assert "quality_gate.py --seed-testmon-status" in proto
        assert ".testmondata.seed.log" in proto

    def test_non_python_early_exit_documented(self) -> None:
        """QS-290 (S-4, task 7): the canonical statement that a non-`.py`
        `--impacted` run checks NOTHING, and that `--quick tests/qs` is the
        verification rather than a supplement, must live in project-rules.md —
        it is the one behavioural consequence a reader could otherwise
        mistake for a real green."""
        flat = self._rules_flat()
        assert "Non-Python change sets" in flat
        assert "exits early and checks **nothing**" in flat
        assert "is not a supplement there, it is **the** verification" in flat
        # nice-to-have 18: the doc's UI hint must agree with `_IMPACTED_NON_PY_LINES`.
        for hint in quality_gate._IMPACTED_NON_PY_LINES:
            for target in re.findall(r"--quick [\w./]+", hint):
                assert f"quality_gate.py {target}" in self._rules(), (
                    f"the gate prints {target!r} but project-rules.md does not: the two "
                    "hints must not send the reader to different commands"
                )
        # should-fix 8's fail-closed qualification belongs in the doc too.
        assert "fails closed" in flat
        # The exit is only sound on a WARM baseline; the doc must not promise a
        # blanket "cost shift, never a verdict" — against a cold baseline
        # testmon select-alls and those tests can fail.
        assert "Cold baselines do not take the exit" in flat
        assert "changes cost, never a verdict" not in flat, (
            "this claim is false for a cold baseline — the exit is gated on warmth"
        )
        # The warmth gate is not airtight: testmon's environment fingerprint
        # (#341) still slips through, so the doc must not claim it is.
        assert "#341" in flat, "the known-gap caveat must stay documented"

    def test_serial_fast_path_documented(self) -> None:
        """QS-290 (S-1, task 7): `--quick`'s comment block claimed
        "Uses xdist + sysmon" unconditionally. That is now false below the
        threshold."""
        flat = self._rules_flat()
        assert "Uses xdist + sysmon" not in flat, "the unconditional-xdist claim must be gone"
        assert "single-process" in flat
        assert f"{quality_gate._SERIAL_MAX_TESTS} collected" in flat

    def test_testing_layers_cross_references_rather_than_restating(self) -> None:
        """The agent-facing concept doc points at the canonical statement
        instead of duplicating it (a second copy is a second thing to drift)."""
        layers = (
            Path(__file__).resolve().parent.parent / "docs" / "agents" / "concepts" / "testing-layers.md"
        ).read_text()
        assert "Non-Python change sets" in layers
        assert "project-rules.md" in layers

    def test_seed_testmon_follow_documented_in_both_places(self) -> None:
        """QS-299 AC#10: `--seed-testmon-follow` appears in BOTH the command-help
        block AND the carve-out prose paragraph."""
        rules = self._rules()
        # command-help block (the runnable command line).
        assert "quality_gate.py --seed-testmon-follow --seed-token" in rules
        # carve-out prose: named as a read-only companion + exit-5 supersession.
        assert "streaming follower" in rules
        assert "superseded" in rules


class TestFinishTaskFollowerAgents:
    """QS-299 AC#9: all three qs-finish-task.md agents launch the tokened
    detached seed + stream the follower inline, in lockstep."""

    _AGENTS = (
        ".claude/agents/qs-finish-task.md",
        ".cursor/agents/qs-finish-task.md",
        ".opencode/agents/qs-finish-task.md",
    )

    def _text(self, rel: str) -> str:
        return (Path(__file__).resolve().parent.parent / rel).read_text()

    @pytest.mark.parametrize("rel", _AGENTS)
    def test_launches_tokened_detached_seed(self, rel: str) -> None:
        body = self._text(rel)
        assert "--seed-testmon --detached --seed-token" in body
        assert "</dev/null" in body  # fully backgrounded

    @pytest.mark.parametrize("rel", _AGENTS)
    def test_empty_seed_token_guard_present(self, rel: str) -> None:
        """review-fix #03 (finding #5): the launch is guarded by a non-empty
        SEED_TOKEN check so a uuid failure can't invoke the CLI with an empty
        --seed-token (which it rejects with exit 2 — a silent no-op)."""
        assert '[ -n "$SEED_TOKEN" ]' in self._text(rel)

    @pytest.mark.parametrize("rel", _AGENTS)
    def test_no_rm_f_marker(self, rel: str) -> None:
        """The new seed must READ the predecessor marker to preempt it — never rm."""
        assert 'rm -f "$MAIN_DIR/.testmondata.seed-status"' not in self._text(rel)

    @pytest.mark.parametrize("rel", _AGENTS)
    def test_streams_follower_inline(self, rel: str) -> None:
        body = self._text(rel)
        assert "--seed-testmon-follow --seed-token" in body
        # foreground form is labeled conceptual only.
        assert "conceptual only" in body

    @pytest.mark.parametrize("rel", _AGENTS)
    def test_follower_exit_is_informational(self, rel: str) -> None:
        body = self._text(rel)
        assert "completion signal, not a gate" in body

    @pytest.mark.parametrize("rel", _AGENTS)
    def test_hard_rules_carveout_names_follower(self, rel: str) -> None:
        body = self._text(rel)
        assert "`--seed-testmon-follow` is a read-only" in body

    @pytest.mark.parametrize("rel", _AGENTS)
    def test_no_residual_primary_status_instruction(self, rel: str) -> None:
        """The old 'run --seed-testmon-status yourself from the main checkout'
        primary path must be gone (kept only as an optional scripting fallback)."""
        body = self._text(rel)
        assert "run from the main checkout, $MAIN_DIR)" not in body


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
        """QS-299 (supersedes QS-286): the detached refresh logs to
        `.testmondata.seed.log`, streams the follower inline, and culminates in
        a "safe to close this terminal" verdict — WITHOUT the old rm -f marker
        or the manual --seed-testmon-status primary path."""
        body = (Path(__file__).resolve().parent.parent / harness / "agents" / "qs-finish-task.md").read_text()
        assert '>"$MAIN_DIR/.testmondata.seed.log" 2>&1' in body  # log redirect
        assert "--seed-testmon-follow --seed-token" in body  # inline streaming
        assert "safe to close this terminal" in body
        # QS-299: the stale-marker rm is gone (it would blind preemption).
        assert 'rm -f "$MAIN_DIR/.testmondata.seed-status"' not in body
        # the old silent seed redirect is gone (other >/dev/null uses remain)
        assert "--seed-testmon >/dev/null 2>&1" not in body

    def test_seed_launch_block_byte_identical_across_harnesses(self) -> None:
        """Harness-sync: the QS-299 tokened-detached seed LAUNCH block (bash) is
        byte-identical in all three finish-task copies (the per-harness
        background+monitor prose is allowed to differ)."""
        blocks = []
        for harness in (".claude", ".cursor", ".opencode"):
            body = (Path(__file__).resolve().parent.parent / harness / "agents" / "qs-finish-task.md").read_text()
            # Anchor on the token generation and the exact detached-launch
            # redirect line — both code-adjacent, so per-harness follower prose
            # after the fence can't truncate the slice inconsistently.
            start = body.index('SEED_TOKEN="$("$QG_PY"')
            redirect = '</dev/null >"$MAIN_DIR/.testmondata.seed.log" 2>&1 & )'
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
                [
                    quality_gate._venv_tool("diff-cover"),
                    str(xml),
                    f"--compare-branch={base}",
                    *extra,
                    "--fail-under=100",
                ],
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


# --- QS-332 (B1): the lane check ---


_LANE_FACTORY_TASK = ["kind:feature", "target:factory", "scale:task"]
_LANE_PRODUCT_TASK = ["kind:bug", "target:product", "scale:task"]


def _patch_lane_issue(issue: int = 332, branch: str = "QS_332"):
    return patch.object(quality_gate, "_resolve_lane_issue", return_value=(issue, branch))


def _patch_lane_labels(labels: list[str] | None):
    return patch.object(quality_gate, "_fetch_lane_labels", return_value=labels)


def _patch_lane_files(paths: list[str] | None):
    """Patch the fail-closed lane change-set seam (review-fix #01: the
    lane check computes its own tracked-only, NUL-delimited change set —
    `None` means a git failure and must reach the fail-closed arm)."""
    return patch.object(quality_gate, "_lane_changed_files", return_value=paths)


class TestLaneCheckHelper:
    """`_check_lane_targets` — never exits, never prints (review R2-12);
    the result object carries `declaration_missing` / `warning` / `fyi`
    and each call site maps it."""

    def test_skips_silently_when_no_issue_resolvable(self) -> None:
        with (
            patch.object(quality_gate, "_resolve_lane_issue", return_value=None),
            patch.object(quality_gate, "_lane_changed_files") as files_mock,
        ):
            res = quality_gate._check_lane_targets()
        assert res == quality_gate.LaneCheckResult(None, None, None)
        # Review-fix #01: the change-set git calls only run once an issue
        # resolves — a non-task branch pays nothing.
        files_mock.assert_not_called()

    def test_missing_declaration_fails_with_shape_aware_backfill(self) -> None:
        with (
            _patch_lane_issue(),
            _patch_lane_labels(["bug"]),
            _patch_lane_files(["docs/workflow/x.md"]),
        ):
            res = quality_gate._check_lane_targets()
        assert res.declaration_missing is not None
        assert "gh issue edit 332 --add-label" in res.declaration_missing
        assert "<N>" not in res.declaration_missing
        assert res.warning is None

    def test_cross_target_diff_warns_and_lists_all_crossing_files(self) -> None:
        files = [
            "scripts/qs/x.py",
            "custom_components/quiet_solar/a.py",
            "tests/test_a.py",
            "docs/stories/QS-332.story.md",  # neutral — never listed
        ]
        with (
            _patch_lane_issue(),
            _patch_lane_labels(_LANE_FACTORY_TASK),
            _patch_lane_files(files),
        ):
            res = quality_gate._check_lane_targets()
        assert res.declaration_missing is None
        assert res.warning is not None
        assert "custom_components/quiet_solar/a.py" in res.warning
        assert "tests/test_a.py" in res.warning
        assert "docs/stories/QS-332.story.md" not in res.warning
        # The split recommendation ALWAYS prints — "substantial" is human
        # judgment, not a computed threshold (review R2-09).
        assert "split" in res.warning.lower()
        assert "verify these serve the declared factory purpose" in res.warning.lower()

    def test_no_crossing_means_no_warning(self) -> None:
        files = ["scripts/qs/x.py", "docs/workflow/lanes/bug-product.md", "README.md"]
        with (
            _patch_lane_issue(),
            _patch_lane_labels(_LANE_FACTORY_TASK),
            _patch_lane_files(files),
        ):
            res = quality_gate._check_lane_targets()
        assert res == quality_gate.LaneCheckResult(None, None, None)

    def test_unknown_paths_are_fyi_and_neutral_stays_silent(self) -> None:
        files = ["mystery.bin", "docs/stories/QS-332.story.md", "scripts/x.sh"]
        with (
            _patch_lane_issue(),
            _patch_lane_labels(_LANE_FACTORY_TASK),
            _patch_lane_files(files),
        ):
            res = quality_gate._check_lane_targets()
        assert res.fyi is not None
        assert "mystery.bin" in res.fyi
        assert "docs/stories/QS-332.story.md" not in res.fyi
        assert res.warning is None
        assert res.declaration_missing is None

    def test_epic_declaration_is_valid_and_classified_against_its_target(self) -> None:
        """A `scale:epic` declaration has no kind and is NOT asked to grow
        one (review PC-11); the crossing check runs against its target."""
        with (
            _patch_lane_issue(),
            _patch_lane_labels(["scale:epic", "target:factory"]),
            _patch_lane_files(["custom_components/quiet_solar/a.py"]),
        ):
            res = quality_gate._check_lane_targets()
        assert res.declaration_missing is None
        assert res.warning is not None

    def test_gh_failure_local_warns_and_skips(self) -> None:
        with (
            _patch_lane_issue(),
            _patch_lane_labels(None),
            _patch_lane_files(["scripts/qs/x.py"]),
            patch.object(quality_gate, "_is_ci", return_value=False),
        ):
            res = quality_gate._check_lane_targets()
        assert res.declaration_missing is None
        assert res.warning is not None  # the skip notice rides the warning channel

    def test_gh_failure_in_ci_fails_closed(self) -> None:
        with (
            _patch_lane_issue(),
            _patch_lane_labels(None),
            _patch_lane_files(["scripts/qs/x.py"]),
            patch.object(quality_gate, "_is_ci", return_value=True),
        ):
            res = quality_gate._check_lane_targets()
        assert res.declaration_missing is not None

    @pytest.mark.parametrize("is_ci", [False, True], ids=["local", "ci"])
    def test_unknown_change_set_has_gh_failure_semantics(self, is_ci: bool) -> None:
        """`_impacted_early_exit_paths() is None` (git failure) is treated
        exactly like a `gh` failure: local warn+skip, CI fail closed."""
        with (
            _patch_lane_issue(),
            _patch_lane_labels(_LANE_FACTORY_TASK),
            _patch_lane_files(None),
            patch.object(quality_gate, "_is_ci", return_value=is_ci),
        ):
            res = quality_gate._check_lane_targets()
        if is_ci:
            assert res.declaration_missing is not None
        else:
            assert res.declaration_missing is None
            assert res.warning is not None


class TestEmitLaneCheck:
    """The call-site mapping: fyi/warning → stderr (NEVER stdout — `--json`
    must stay parseable, review DP2-03); declaration_missing → rc 1."""

    def test_warning_and_fyi_go_to_stderr_only(self, capsys: pytest.CaptureFixture[str]) -> None:
        res = quality_gate.LaneCheckResult(None, "WARN-TEXT", "FYI-TEXT")
        assert quality_gate._emit_lane_check(res) == 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "WARN-TEXT" in captured.err
        assert "FYI-TEXT" in captured.err

    def test_declaration_missing_maps_to_failure(self, capsys: pytest.CaptureFixture[str]) -> None:
        res = quality_gate.LaneCheckResult("MISSING-TEXT", None, None)
        assert quality_gate._emit_lane_check(res) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "MISSING-TEXT" in captured.err

    def test_skip_result_is_silent(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert quality_gate._emit_lane_check(quality_gate.LaneCheckResult(None, None, None)) == 0
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""


class TestResolveLaneIssue:
    """Branch → issue resolution, incl. the CI detached-HEAD fallback
    (review N-1: `git branch --show-current` is empty on a pull_request
    merge-ref checkout; `--branch "${{ github.head_ref }}"` fills in)."""

    def _patch_branch(self, branch: str):
        def fake_run(cmd, **kwargs):
            assert cmd == ["git", "branch", "--show-current"]
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{branch}\n", stderr="")

        return patch.object(quality_gate, "_run", side_effect=fake_run)

    def test_qs_branch_resolves(self) -> None:
        with self._patch_branch("QS_45"):
            assert _REAL_RESOLVE_LANE_ISSUE(None) == (45, "QS_45")

    def test_non_task_branch_resolves_to_none(self) -> None:
        with self._patch_branch("main"):
            assert _REAL_RESOLVE_LANE_ISSUE(None) is None

    def test_detached_head_in_ci_falls_back_to_branch_override(self) -> None:
        with self._patch_branch(""), patch.object(quality_gate, "_is_ci", return_value=True):
            assert _REAL_RESOLVE_LANE_ISSUE("QS_45") == (45, "QS_45")

    def test_detached_head_locally_ignores_override(self) -> None:
        with self._patch_branch(""), patch.object(quality_gate, "_is_ci", return_value=False):
            assert _REAL_RESOLVE_LANE_ISSUE("QS_45") is None

    def test_detached_head_in_ci_without_override_is_none(self) -> None:
        with self._patch_branch(""), patch.object(quality_gate, "_is_ci", return_value=True):
            assert _REAL_RESOLVE_LANE_ISSUE(None) is None


class TestLaneLabelCache:
    """The label marker-file cache: keyed on (issue, branch), 10-minute
    TTL, `_is_ci()` bypass, and ONLY complete declarations are cached (a
    backfill-then-re-run can never be re-failed by a stale cache)."""

    def _gh_run(self, labels: list[str], calls: list[list[str]]):
        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            payload = json.dumps({"labels": [{"name": n} for n in labels]})
            return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr="")

        return patch.object(quality_gate, "_run", side_effect=fake_run)

    def test_complete_declaration_is_cached_and_reused(self, tmp_path: Path) -> None:
        calls: list[list[str]] = []
        cache = tmp_path / ".lane_check_cache"
        with (
            patch.object(quality_gate, "LANE_CACHE_FILE", cache),
            patch.object(quality_gate, "_is_ci", return_value=False),
            self._gh_run(_LANE_FACTORY_TASK, calls),
        ):
            assert quality_gate._fetch_lane_labels(332, "QS_332") == _LANE_FACTORY_TASK
            assert quality_gate._fetch_lane_labels(332, "QS_332") == _LANE_FACTORY_TASK
        assert len(calls) == 1  # second hit served from the cache
        assert cache.exists()

    def test_incomplete_declaration_is_never_cached(self, tmp_path: Path) -> None:
        calls: list[list[str]] = []
        cache = tmp_path / ".lane_check_cache"
        with (
            patch.object(quality_gate, "LANE_CACHE_FILE", cache),
            patch.object(quality_gate, "_is_ci", return_value=False),
            self._gh_run(["bug"], calls),
        ):
            assert quality_gate._fetch_lane_labels(332, "QS_332") == ["bug"]
            assert quality_gate._fetch_lane_labels(332, "QS_332") == ["bug"]
        assert len(calls) == 2  # no cache write, no cache hit
        assert not cache.exists()

    def test_cache_is_keyed_on_issue_and_branch(self, tmp_path: Path) -> None:
        calls: list[list[str]] = []
        cache = tmp_path / ".lane_check_cache"
        with (
            patch.object(quality_gate, "LANE_CACHE_FILE", cache),
            patch.object(quality_gate, "_is_ci", return_value=False),
            self._gh_run(_LANE_FACTORY_TASK, calls),
        ):
            quality_gate._fetch_lane_labels(332, "QS_332")
            quality_gate._fetch_lane_labels(45, "QS_45")
        assert len(calls) == 2

    def test_expired_cache_refetches(self, tmp_path: Path) -> None:
        calls: list[list[str]] = []
        cache = tmp_path / ".lane_check_cache"
        cache.write_text(
            json.dumps(
                {
                    "issue": 332,
                    "branch": "QS_332",
                    "labels": _LANE_FACTORY_TASK,
                    "time": time.time() - quality_gate._LANE_CACHE_TTL_S - 1,
                }
            )
        )
        with (
            patch.object(quality_gate, "LANE_CACHE_FILE", cache),
            patch.object(quality_gate, "_is_ci", return_value=False),
            self._gh_run(_LANE_FACTORY_TASK, calls),
        ):
            quality_gate._fetch_lane_labels(332, "QS_332")
        assert len(calls) == 1

    def test_ci_bypasses_the_cache(self, tmp_path: Path) -> None:
        calls: list[list[str]] = []
        cache = tmp_path / ".lane_check_cache"
        cache.write_text(
            json.dumps(
                {
                    "issue": 332,
                    "branch": "QS_332",
                    "labels": _LANE_PRODUCT_TASK,
                    "time": time.time(),
                }
            )
        )
        with (
            patch.object(quality_gate, "LANE_CACHE_FILE", cache),
            patch.object(quality_gate, "_is_ci", return_value=True),
            self._gh_run(_LANE_FACTORY_TASK, calls),
        ):
            assert quality_gate._fetch_lane_labels(332, "QS_332") == _LANE_FACTORY_TASK
        assert len(calls) == 1

    def test_gh_failure_returns_none(self, tmp_path: Path) -> None:
        def failing_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

        with (
            patch.object(quality_gate, "LANE_CACHE_FILE", tmp_path / "c"),
            patch.object(quality_gate, "_is_ci", return_value=False),
            patch.object(quality_gate, "_run", side_effect=failing_run),
        ):
            assert quality_gate._fetch_lane_labels(332, "QS_332") is None


class TestLaneCheckImpactedCallSite:
    """Call site 1 (review CR-2 / planner N-3): inside `check_impacted()`,
    after `_ensure_testmon_db_safe()` and BEFORE the warm-baseline
    early-exit block — the hook must run on a pure-docs change set."""

    @pytest.fixture(autouse=True)
    def _testmon_db_present(self, tmp_path_factory: pytest.TempPathFactory):
        root = tmp_path_factory.mktemp("tmdb-lane")
        db = root / ".testmondata"
        db.write_bytes(b"x")
        with (
            patch.object(quality_gate, "TESTMON_DATA", db),
            patch.object(quality_gate, "COVERAGE_DATA", root / ".coverage"),
        ):
            yield

    def _base_patches(self) -> contextlib.ExitStack:
        stack = contextlib.ExitStack()
        stack.enter_context(
            patch.object(quality_gate, "_impacted_tooling_available", return_value=True)
        )
        stack.enter_context(patch.object(quality_gate, "_clean_orphan_cov_shards"))
        stack.enter_context(patch.object(quality_gate, "_ensure_testmon_db_safe"))
        return stack

    def test_pure_docs_crossing_warns_but_early_exit_still_passes(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A cross-target pure-docs diff WARNS (exit code unaffected) even
        though the non-`.py` early exit then fires — the lane check runs
        before the warm-baseline block."""
        with (
            self._base_patches(),
            _patch_early_exit(["docs/agents/concepts/solver.md"]),
            _patch_lane_files(["docs/agents/concepts/solver.md"]),  # product-classified
            _patch_lane_issue(),
            _patch_lane_labels(_LANE_FACTORY_TASK),
        ):
            assert quality_gate.check_impacted() == 0
        err = capsys.readouterr().err
        assert "docs/agents/concepts/solver.md" in err
        assert "no Python files changed" in err  # the early exit still fired

    def test_untracked_scratch_file_never_warns(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Review-fix #01: the lane classification runs on the TRACKED
        change set only — an untracked local scratch file under a product
        prefix (present in `_impacted_early_exit_paths`'s union, which
        keeps it for the early-exit decision) must not earn the loud
        cross-target banner that CI would not print for the same tree."""
        with (
            self._base_patches(),
            _patch_early_exit(
                ["docs/workflow/x.md", "custom_components/quiet_solar/scratch.py"]
            ),
            _patch_lane_files(["docs/workflow/x.md"]),  # tracked only
            _patch_lane_issue(),
            _patch_lane_labels(_LANE_FACTORY_TASK),
            patch.object(quality_gate, "_resolve_diff_base", return_value=None),
            patch.object(quality_gate, "_is_ci", return_value=False),
        ):
            assert quality_gate.check_impacted() == 0
        err = capsys.readouterr().err
        assert "LANE WARNING" not in err

    def test_pure_docs_missing_declaration_fails(self) -> None:
        """A missing declaration FAILS even on a pure-docs change set."""
        with (
            self._base_patches(),
            _patch_early_exit(["docs/workflow/x.md"]),
            _patch_lane_files(["docs/workflow/x.md"]),
            _patch_lane_issue(),
            _patch_lane_labels([]),
        ):
            assert quality_gate.check_impacted() == 1

    def test_git_failure_locally_warns_and_pipeline_continues(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            self._base_patches(),
            _patch_early_exit(None),
            _patch_lane_files(None),
            _patch_lane_issue(),
            _patch_lane_labels(_LANE_FACTORY_TASK),
            patch.object(quality_gate, "_resolve_diff_base", return_value=None),
            patch.object(quality_gate, "_is_ci", return_value=False),
        ):
            assert quality_gate.check_impacted() == 0
        assert "lane check" in capsys.readouterr().err

    def test_git_failure_in_ci_fails_closed(self) -> None:
        with (
            self._base_patches(),
            _patch_early_exit(None),
            _patch_lane_files(None),
            _patch_lane_issue(),
            _patch_lane_labels(_LANE_FACTORY_TASK),
            patch.object(quality_gate, "_is_ci", return_value=True),
        ):
            assert quality_gate.check_impacted() == 1

    def test_no_issue_resolvable_skips_and_pipeline_runs(self) -> None:
        """Not a task branch → the check silently skips (most fixtures)."""
        with (
            self._base_patches(),
            _patch_early_exit(["docs/x.md"]),
        ):
            # autouse fixture already stubs _resolve_lane_issue → None
            assert quality_gate.check_impacted() == 0


class TestLaneCheckMainCallSite:
    """Call site 2: `main()`, between `_get_changed_files()` and
    `_detect_scope(...)`, for the full gate."""

    def test_missing_declaration_fails_before_any_gate_runs(self) -> None:
        with (
            patch("sys.argv", ["quality_gate.py", "--json"]),
            patch.object(quality_gate, "_get_changed_files", return_value=["scripts/qs/x.py"]),
            _patch_lane_files(["scripts/qs/x.py"]),
            _patch_lane_issue(),
            _patch_lane_labels([]),
            _patch_all_gates() as mocks,
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()
        assert exc_info.value.code == 1
        for m in mocks:
            m.assert_not_called()

    def test_warning_keeps_json_stdout_parseable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Warning/FYI go to stderr; `--json` stdout stays machine-readable
        (review DP2-03)."""
        with (
            patch("sys.argv", ["quality_gate.py", "--json"]),
            patch.object(
                quality_gate,
                "_get_changed_files",
                return_value=["custom_components/quiet_solar/a.py", "mystery.bin"],
            ),
            _patch_lane_files(["custom_components/quiet_solar/a.py", "mystery.bin"]),
            _patch_lane_issue(),
            _patch_lane_labels(_LANE_FACTORY_TASK),
            _patch_full_scope(),
            _patch_all_gates(),
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)  # must not raise
        assert output["all_passed"] is True
        assert "custom_components/quiet_solar/a.py" in captured.err
        assert "mystery.bin" in captured.err  # the unknown-path FYI

    def test_quick_mode_never_runs_the_lane_check(self) -> None:
        with (
            patch("sys.argv", ["quality_gate.py", "--quick", "tests/qs"]),
            patch.object(quality_gate, "_check_lane_targets") as lane_mock,
            patch.object(
                quality_gate,
                "check_pytest_files",
                return_value={"name": "pytest", "passed": True, "detail": ""},
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()
        assert exc_info.value.code == 0
        lane_mock.assert_not_called()

    def test_cache_hit_skips_the_lane_check_by_design(self, tmp_path: Path) -> None:
        """KNOWN skip, recorded (review planner): the `--cache`
        short-circuit sits before the lane seam. Acceptable because
        `--cache` requires a previously green run on a clean tree and the
        mandated pre-commit form is `--impacted`."""
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
            _patch_git_state("QS_76", "abc123", True),
            patch.object(quality_gate, "CACHE_FILE", cache_path),
            patch.object(quality_gate, "_check_lane_targets") as lane_mock,
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()
        assert exc_info.value.code == 0
        lane_mock.assert_not_called()


class TestLaneCheckSubcommand:
    """`--lane-check` (reviews CR-3 + N-1): a cheap subcommand running the
    lane steps only — no pytest, no coverage — in the execution-mode
    mutex ladder, short-circuiting before scope detection."""

    @pytest.mark.parametrize(
        "conflict",
        ["--impacted", "--quick", "--cache", "--no-cache", "--full", "--fix",
         "--seed-testmon", "--seed-testmon-status", "--seed-testmon-follow"],
    )
    def test_mutex_ladder_membership(self, conflict: str) -> None:
        argv = ["quality_gate.py", "--lane-check", conflict]
        if conflict == "--quick":
            argv.append("tests/qs")
        with patch("sys.argv", argv), pytest.raises(SystemExit) as exc_info:
            quality_gate.main()
        assert exc_info.value.code == 2

    def test_branch_flag_requires_lane_check(self) -> None:
        with (
            patch("sys.argv", ["quality_gate.py", "--branch", "QS_45"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()
        assert exc_info.value.code == 2

    def test_forwards_branch_override_and_runs_no_gates(self) -> None:
        with (
            patch("sys.argv", ["quality_gate.py", "--lane-check", "--branch", "QS_45"]),
            patch.object(
                quality_gate,
                "_check_lane_targets",
                return_value=quality_gate.LaneCheckResult(None, None, None),
            ) as lane_mock,
            _patch_all_gates() as mocks,
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()
        assert exc_info.value.code == 0
        lane_mock.assert_called_once_with("QS_45")
        for m in mocks:
            m.assert_not_called()

    def test_git_failure_in_ci_fails_closed_end_to_end(self) -> None:
        """Review-fix #01: `_get_changed_files` maps git failures to `[]`,
        which silently degraded the CI job to declaration-only. The lane
        check now computes its own FAIL-CLOSED change set: a git failure
        (`None`) in CI exits non-zero from the subcommand."""
        with (
            patch("sys.argv", ["quality_gate.py", "--lane-check"]),
            _patch_lane_files(None),
            _patch_lane_issue(),
            _patch_lane_labels(_LANE_FACTORY_TASK),
            patch.object(quality_gate, "_is_ci", return_value=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()
        assert exc_info.value.code == 1

    def test_warning_lands_in_github_step_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        summary = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        with (
            patch("sys.argv", ["quality_gate.py", "--lane-check"]),
            _patch_lane_files(["custom_components/quiet_solar/a.py"]),
            _patch_lane_issue(),
            _patch_lane_labels(_LANE_FACTORY_TASK),
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()
        assert exc_info.value.code == 0  # a crossing never fails
        assert "custom_components/quiet_solar/a.py" in summary.read_text(encoding="utf-8")

    def test_missing_declaration_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        with (
            patch("sys.argv", ["quality_gate.py", "--lane-check"]),
            _patch_lane_files(["scripts/x.sh"]),
            _patch_lane_issue(),
            _patch_lane_labels([]),
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()
        assert exc_info.value.code == 1


class TestResolveLaneIssueUnderscore:
    """Review-fix #01: `int()` accepts underscore digit-grouping, so a
    branch `QS_332_2` parsed as issue #3322 and the gate enforced the
    wrong issue's declaration."""

    def _patch_branch(self, branch: str):
        def fake_run(cmd, **kwargs):
            assert cmd == ["git", "branch", "--show-current"]
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{branch}\n", stderr="")

        return patch.object(quality_gate, "_run", side_effect=fake_run)

    @pytest.mark.parametrize("branch", ["QS_332_2", "QS_1_000", "QS_", "QS_x2"])
    def test_non_pure_digit_suffix_resolves_to_none(self, branch: str) -> None:
        with self._patch_branch(branch):
            assert _REAL_RESOLVE_LANE_ISSUE(None) is None

    def test_ci_override_with_underscore_grouping_is_rejected_too(self) -> None:
        with self._patch_branch(""), patch.object(quality_gate, "_is_ci", return_value=True):
            assert _REAL_RESOLVE_LANE_ISSUE("QS_332_2") is None


class TestFetchLaneLabelsTimeout:
    """Review-fix #01: the `gh issue view` call sits on the `--impacted`
    pre-commit hot path — it must carry a bounded timeout (a half-dead
    network with a cold cache must not hang the sub-15s gate). rc 124
    maps to the existing `!= 0 → None → local warn+skip` path."""

    def test_gh_call_carries_a_bounded_timeout(self, tmp_path: Path) -> None:
        seen_timeouts: list[float | None] = []

        def fake_run(cmd, timeout=None, **kwargs):
            seen_timeouts.append(timeout)
            payload = json.dumps({"labels": [{"name": n} for n in _LANE_FACTORY_TASK]})
            return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr="")

        with (
            patch.object(quality_gate, "LANE_CACHE_FILE", tmp_path / "c"),
            patch.object(quality_gate, "_is_ci", return_value=False),
            patch.object(quality_gate, "_run", side_effect=fake_run),
        ):
            quality_gate._fetch_lane_labels(332, "QS_332")
        assert len(seen_timeouts) == 1
        assert seen_timeouts[0] is not None and seen_timeouts[0] > 0

    def test_timeout_rc_124_degrades_to_none(self, tmp_path: Path) -> None:
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 124, stdout="", stderr="timed out")

        with (
            patch.object(quality_gate, "LANE_CACHE_FILE", tmp_path / "c"),
            patch.object(quality_gate, "_is_ci", return_value=False),
            patch.object(quality_gate, "_run", side_effect=fake_run),
        ):
            assert quality_gate._fetch_lane_labels(332, "QS_332") is None


class TestLaneChangedFiles:
    """Review-fix #01: the lane check's own change-set helper — tracked
    paths only, NUL-delimited (`-z`, the QS-290 non-ASCII treatment),
    returncode-checked and FAIL-CLOSED (`None` on any git failure, so the
    CI fail-closed arm of `_check_lane_targets` is reachable where CI
    actually runs)."""

    def _fake_run(self, outputs: dict[str, tuple[int, str]]):
        def fake_run(cmd, **kwargs):
            key = " ".join(cmd)
            for fragment, (rc, out) in outputs.items():
                if fragment in key:
                    return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr="")
            raise AssertionError(f"unexpected command: {cmd}")

        return patch.object(quality_gate, "_run", side_effect=fake_run)

    def test_union_of_the_three_tracked_diffs_sorted(self) -> None:
        with self._fake_run(
            {
                "origin/main...HEAD": (0, "b.md\0a.py\0"),
                "--cached": (0, "c.md\0"),
                "diff --name-only -z HEAD": (0, "a.py\0d.md\0"),
            }
        ):
            assert quality_gate._lane_changed_files() == ["a.py", "b.md", "c.md", "d.md"]

    @pytest.mark.parametrize(
        "failing", ["origin/main...HEAD", "--cached", "diff --name-only -z HEAD"]
    )
    def test_any_git_failure_returns_none(self, failing: str) -> None:
        outputs = {
            "origin/main...HEAD": (0, "a.py\0"),
            "--cached": (0, ""),
            "diff --name-only -z HEAD": (0, ""),
        }
        outputs[failing] = (128, "")
        with self._fake_run(outputs):
            assert quality_gate._lane_changed_files() is None

    def test_non_ascii_paths_arrive_unquoted(self) -> None:
        """`-z` emits raw bytes — no `core.quotePath` C-quoting, so
        `docs/workflow/données.md` classifies factory, not unknown."""
        with self._fake_run(
            {
                "origin/main...HEAD": (0, "docs/workflow/données.md\0"),
                "--cached": (0, ""),
                "diff --name-only -z HEAD": (0, ""),
            }
        ):
            files = quality_gate._lane_changed_files()
        assert files == ["docs/workflow/données.md"]
        import targets as targets_mod

        assert targets_mod.classify(files[0]) == "factory"

    def test_untracked_files_are_not_listed(self) -> None:
        """No `git ls-files --others` rung — tracked diffs only (the
        untracked union stays in `_impacted_early_exit_paths`, for the
        early-exit decision alone)."""
        commands: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            commands.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch.object(quality_gate, "_run", side_effect=fake_run):
            assert quality_gate._lane_changed_files() == []
        assert all("ls-files" not in " ".join(cmd) for cmd in commands)


class TestSeedModeSkipsLaneCheck:
    """Review-fix #01: AC-5's "never runs in --seed-testmon" clause was
    unpinned (only the `--quick` half had a dedicated skip test)."""

    def test_seed_testmon_never_runs_the_lane_check(self) -> None:
        with (
            patch("sys.argv", ["quality_gate.py", "--seed-testmon"]),
            patch.object(quality_gate, "_check_lane_targets") as lane_mock,
            patch.object(quality_gate, "seed_testmon", return_value=0),
            pytest.raises(SystemExit) as exc_info,
        ):
            quality_gate.main()
        assert exc_info.value.code == 0
        lane_mock.assert_not_called()


class TestFetchLaneLabelsNullLabels:
    """Review-fix #03: `"labels": null` is a valid-JSON response the API
    can return. With a bare `.get("labels", [])` the comprehension raised
    `TypeError`, was swallowed by the broad `except`, and degraded to
    `None` — i.e. the gh-FAILURE path (local warn+skip / CI fail closed)
    for a response that was perfectly readable. The honest answer is an
    empty label list, which then fails the declaration check on its own
    merits."""

    def test_null_labels_resolve_to_an_empty_list(self, tmp_path: Path) -> None:
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps({"labels": None}), stderr=""
            )

        with (
            patch.object(quality_gate, "LANE_CACHE_FILE", tmp_path / "c"),
            patch.object(quality_gate, "_is_ci", return_value=False),
            patch.object(quality_gate, "_run", side_effect=fake_run),
        ):
            assert quality_gate._fetch_lane_labels(332, "QS_332") == []

    def test_null_labels_fail_the_declaration_rather_than_the_fetch(
        self, tmp_path: Path
    ) -> None:
        """End-to-end: the result is a declaration FAILURE (actionable),
        not a `gh`-unavailable warn+skip (silent locally)."""
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps({"labels": None}), stderr=""
            )

        with (
            patch.object(quality_gate, "LANE_CACHE_FILE", tmp_path / "c"),
            patch.object(quality_gate, "_is_ci", return_value=False),
            patch.object(quality_gate, "_run", side_effect=fake_run),
            _patch_lane_issue(),
            _patch_lane_files(["scripts/qs/x.py"]),
        ):
            res = quality_gate._check_lane_targets()
        assert res.declaration_missing is not None
        assert "gh issue edit 332" in res.declaration_missing

    def test_malformed_label_entries_degrade_to_none(self, tmp_path: Path) -> None:
        """A malformed ENTRY (null / no `name`) is genuinely unreadable —
        it keeps the existing unavailable semantics, unlike a null field."""
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps({"labels": [{"id": 1}]}), stderr=""
            )

        with (
            patch.object(quality_gate, "LANE_CACHE_FILE", tmp_path / "c"),
            patch.object(quality_gate, "_is_ci", return_value=False),
            patch.object(quality_gate, "_run", side_effect=fake_run),
        ):
            assert quality_gate._fetch_lane_labels(332, "QS_332") is None


class TestFetchLaneLabelsNonDictJson:
    """Review-fix #04: a non-dict top-level JSON value (`null`, `[]`, `42`)
    made `data.get(...)` raise `AttributeError`, which was not in the
    except tuple — it escaped as a raw traceback instead of the `None`
    (gh-unavailable) degrade path."""

    @pytest.mark.parametrize("raw", ["null", "[]", "42"])
    def test_non_dict_json_degrades_to_none(self, tmp_path: Path, raw: str) -> None:
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout=raw, stderr="")

        with (
            patch.object(quality_gate, "LANE_CACHE_FILE", tmp_path / "c"),
            patch.object(quality_gate, "_is_ci", return_value=False),
            patch.object(quality_gate, "_run", side_effect=fake_run),
        ):
            assert quality_gate._fetch_lane_labels(332, "QS_332") is None
