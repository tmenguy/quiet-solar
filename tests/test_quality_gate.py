"""Tests for quality_gate.py caching functionality."""

from __future__ import annotations

import io
import json
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
        assert (
            "you cannot combine --quick with --cache, --no-cache, --full, or --fix"
            in err
        ), f"mutex message missing/changed: {err!r}"

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
            assert cmd[n_idx + 1] == workers_value, (
                f"-n value mismatch; want {workers_value!r}, got {cmd[n_idx + 1]!r}"
            )

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
        assert cited in collect_cmd, (
            f"collect-only cmd must include the cited path {cited!r}; got {collect_cmd!r}"
        )
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
        assert "must be inside the repo" in err, (
            f"path-escape message missing/changed: {err!r}"
        )
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
        assert "must be non-empty" in err, (
            f"empty-path message missing/changed: {err!r}"
        )
