"""Tests for quality_gate.py caching functionality."""

from __future__ import annotations

import io
import json
import sys
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
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """AC 2: cache hit returns cached results without running gates."""
        cached_results = _make_all_pass_results()
        cache_path = tmp_path / ".quality_gate_cache"
        cache_path.write_text(json.dumps({
            "branch": "QS_76", "commit": "abc123",
            "all_passed": True, "results": cached_results, "timestamp": "",
        }))

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
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """AC 3: different commit invalidates cache."""
        cache_path = tmp_path / ".quality_gate_cache"
        cache_path.write_text(json.dumps({
            "branch": "QS_76", "commit": "old_commit",
            "all_passed": True, "results": _make_all_pass_results(), "timestamp": "",
        }))

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
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """AC 3: dirty working tree invalidates cache."""
        cache_path = tmp_path / ".quality_gate_cache"
        cache_path.write_text(json.dumps({
            "branch": "QS_76", "commit": "abc123",
            "all_passed": True, "results": _make_all_pass_results(), "timestamp": "",
        }))

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
        cache_path.write_text(json.dumps({
            "branch": "QS_76", "commit": "abc123",
            "all_passed": True, "results": _make_all_pass_results(), "timestamp": "",
        }))

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
        cache_path.write_text(json.dumps({
            "branch": "QS_76", "commit": "abc123",
            "all_passed": True, "results": _make_all_pass_results(), "timestamp": "",
        }))

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
        cache_path.write_text(json.dumps({
            "branch": "QS_76", "commit": "abc123",
            "all_passed": True, "results": _make_all_pass_results(), "timestamp": "",
        }))

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
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Cache hit in human-readable mode shows cached indicator."""
        cache_path = tmp_path / ".quality_gate_cache"
        cache_path.write_text(json.dumps({
            "branch": "QS_76", "commit": "abc123",
            "all_passed": True, "results": _make_all_pass_results(), "timestamp": "",
        }))

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
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
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
            return {"name": "pytest", "passed": True, "coverage": "100%",
                    "missing": [], "detail": "", "stderr": ""}

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
            return {"name": "pytest", "passed": True, "coverage": "100%",
                    "missing": [], "detail": "", "stderr": ""}

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
            return {"name": "pytest", "passed": True, "coverage": "100%",
                    "missing": [], "detail": "", "stderr": ""}

        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate, "_stream_pytest", side_effect=fake_stream),
        ):
            quality_gate.check_pytest()

        assert "-n" not in captured["cmd"]

    def test_missing_xdist_falls_back_serial(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When xdist is not importable, gate runs serially and warns to stderr."""
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        captured: dict = {}

        def fake_stream(cmd: list[str]) -> dict:
            captured["cmd"] = cmd
            return {"name": "pytest", "passed": True, "coverage": "100%",
                    "missing": [], "detail": "", "stderr": ""}

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
            return {"name": "pytest", "passed": True, "coverage": "100%",
                    "missing": [], "detail": "", "stderr": ""}

        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate, "_stream_pytest", side_effect=fake_stream),
        ):
            quality_gate.check_pytest()

        assert "--cov-report=" in captured["cmd"]

    def test_collect_only_subprocess_has_no_n_and_no_sysmon(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The --collect-only count subprocess must not get `-n` or COVERAGE_CORE=sysmon."""
        monkeypatch.delenv("QS_QG_PYTEST_WORKERS", raising=False)
        run_calls: list = []
        popen_calls: list = []

        def fake_run(cmd, cwd=None):  # type: ignore[no-untyped-def]
            run_calls.append({"cmd": list(cmd)})
            r = MagicMock()
            r.stdout = "0 tests collected\n"
            r.stderr = ""
            r.returncode = 0
            return r

        class FakePopen:
            def __init__(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
                popen_calls.append({"cmd": list(cmd), "env": kwargs.get("env")})
                self.stdout = io.StringIO("")
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self):  # type: ignore[no-untyped-def]
                return 0

        with (
            patch.object(quality_gate, "_has_xdist", return_value=True),
            patch.object(quality_gate, "_run", side_effect=fake_run),
            patch.object(quality_gate.subprocess, "Popen", FakePopen),
        ):
            quality_gate.check_pytest()

        # First call to _run is the --collect-only subprocess
        collect_cmd = run_calls[0]["cmd"]
        assert "--collect-only" in collect_cmd
        assert "-n" not in collect_cmd

        # The main Popen call (pytest run) should have COVERAGE_CORE=sysmon in env
        main_env = popen_calls[0]["env"]
        assert main_env is not None
        assert main_env.get("COVERAGE_CORE") == "sysmon"


# --- T2: concurrent cheap gates, pytest serialized last (AC3) ---


class TestConcurrentGates:
    """Tests for parallel execution of cheap gates and serial pytest after."""

    def test_cheap_gates_run_concurrently(self, tmp_path: Path) -> None:
        """Cheap gates run in parallel — wall-clock well under 4x serial sleep."""
        sleep_s = 0.3

        def slow(name: str):  # type: ignore[no-untyped-def]
            def _fn(**kwargs):  # type: ignore[no-untyped-def]
                time.sleep(sleep_s)
                return {"name": name, "passed": True, "detail": ""}
            return _fn

        pytest_result = {
            "name": "pytest", "passed": True, "coverage": "100%",
            "missing": [], "detail": "", "stderr": "",
        }

        start = time.monotonic()
        with (
            patch("sys.argv", ["quality_gate.py", "--json"]),
            _patch_git_state(),
            _patch_full_scope(),
            patch.object(quality_gate, "CACHE_FILE", tmp_path / ".quality_gate_cache"),
            patch.object(quality_gate, "check_ruff_format", side_effect=slow("ruff_format")),
            patch.object(quality_gate, "check_ruff_lint", side_effect=slow("ruff_lint")),
            patch.object(quality_gate, "check_mypy", side_effect=slow("mypy")),
            patch.object(quality_gate, "check_translations", side_effect=slow("translations")),
            patch.object(quality_gate, "check_pytest", return_value=pytest_result),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()
        elapsed = time.monotonic() - start

        # 4 cheap gates serial would take 4*sleep_s = 1.2s.
        # With ThreadPoolExecutor(max_workers=4), should be ~sleep_s = 0.3s.
        # Generous threshold (3x sleep) avoids flakiness on slow CI machines.
        assert elapsed < sleep_s * 3, f"expected concurrent execution (<{sleep_s*3}s), took {elapsed:.2f}s"

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
            return {"name": "pytest", "passed": True, "coverage": "100%",
                    "missing": [], "detail": "", "stderr": ""}

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
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
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
            patch.object(quality_gate, "check_pytest", return_value={
                "name": "pytest", "passed": True, "coverage": "100%",
                "missing": [], "detail": "", "stderr": "",
            }),
            pytest.raises(SystemExit),
        ):
            quality_gate.main()

        out = json.loads(capsys.readouterr().out)
        names = [g["name"] for g in out["gates"]]
        assert names == ["ruff_format", "ruff_lint", "mypy", "translations", "pytest"]

    def test_emit_writes_to_stderr_with_prefix(
        self, capsys: pytest.CaptureFixture[str],
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
