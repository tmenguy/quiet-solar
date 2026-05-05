"""Tests for quality_gate.py caching functionality."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts/qs to path so we can import quality_gate
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "qs"
sys.path.insert(0, str(SCRIPTS_DIR))

import quality_gate


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
