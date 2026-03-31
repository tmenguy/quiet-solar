"""Tests for scripts/qs/review_pr.py — CodeRabbit integration."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "qs"


def _import_review_pr():
    """Import review_pr from scripts/qs/ by manipulating sys.path."""
    old_path = sys.path[:]
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        for mod_name in ("review_pr", "utils"):
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        return importlib.import_module("review_pr")
    finally:
        sys.path[:] = old_path


# ---------------------------------------------------------------------------
# wait_for_coderabbit tests
# ---------------------------------------------------------------------------


class TestWaitForCoderabbit:
    """Tests for wait_for_coderabbit() function."""

    def test_matches_rest_author_with_bot_suffix(self):
        """REST API author 'coderabbitai[bot]' is matched."""
        mod = _import_review_pr()
        comments = [
            {"author": "coderabbitai[bot]", "body": "Suggestion", "resolved": False},
            {"author": "human", "body": "LGTM", "resolved": False},
        ]
        with patch.object(mod, "fetch_pr_comments", return_value=comments):
            result = mod.wait_for_coderabbit(1, timeout=5)
        assert len(result) == 1
        assert result[0]["author"] == "coderabbitai[bot]"

    def test_matches_graphql_author_without_bot_suffix(self):
        """GraphQL API author 'coderabbitai' (no [bot]) is also matched."""
        mod = _import_review_pr()
        comments = [
            {"author": "coderabbitai", "body": "Review", "resolved": False},
        ]
        with patch.object(mod, "fetch_pr_comments", return_value=comments):
            result = mod.wait_for_coderabbit(1, timeout=5)
        assert len(result) == 1
        assert result[0]["author"] == "coderabbitai"

    def test_returns_empty_on_timeout(self):
        """When no CodeRabbit comments appear, return empty after timeout."""
        mod = _import_review_pr()
        with (
            patch.object(mod, "fetch_pr_comments", return_value=[]),
            patch.object(mod, "time") as mock_time,
        ):
            # Simulate: first call returns 0, second returns timeout+1
            mock_time.time.side_effect = [0, 0, 10]
            mock_time.sleep = MagicMock()
            result = mod.wait_for_coderabbit(1, timeout=5)
        assert result == []

    def test_does_not_match_copilot_author(self):
        """Old copilot/github-actions authors are not matched."""
        mod = _import_review_pr()
        comments = [
            {"author": "copilot", "body": "Old comment", "resolved": False},
            {"author": "github-actions", "body": "CI", "resolved": False},
        ]
        with (
            patch.object(mod, "fetch_pr_comments", return_value=comments),
            patch.object(mod, "time") as mock_time,
        ):
            mock_time.time.side_effect = [0, 0, 10]
            mock_time.sleep = MagicMock()
            result = mod.wait_for_coderabbit(1, timeout=5)
        assert result == []


# ---------------------------------------------------------------------------
# CLI / main() tests
# ---------------------------------------------------------------------------


class TestMainCli:
    """Tests for the main() CLI interface."""

    def test_no_trigger_copilot_flag(self):
        """--trigger-copilot flag should not exist."""
        mod = _import_review_pr()
        with pytest.raises(SystemExit):
            mod.main(["1", "--trigger-copilot"])

    def test_wait_coderabbit_flag_exists(self):
        """--wait-coderabbit flag should be accepted."""
        mod = _import_review_pr()
        with (
            patch.object(mod, "wait_for_coderabbit", return_value=[]) as mock_wait,
            patch.object(mod, "output_json"),
        ):
            mod.main(["1", "--wait-coderabbit", "30"])
        mock_wait.assert_called_once_with(1, 30)

    def test_wait_coderabbit_json_key(self):
        """--wait-coderabbit should produce 'coderabbit_comments' key."""
        mod = _import_review_pr()
        coderabbit_comments = [{"author": "coderabbitai[bot]", "body": "Fix this"}]
        with (
            patch.object(mod, "wait_for_coderabbit", return_value=coderabbit_comments),
            patch.object(mod, "output_json") as mock_output,
        ):
            mod.main(["1", "--wait-coderabbit", "30"])
        output_data = mock_output.call_args[0][0]
        assert "coderabbit_comments" in output_data
        assert "copilot_comments" not in output_data
        assert "copilot" not in output_data

    def test_fetch_comments_still_works(self):
        """--fetch-comments flag still works."""
        mod = _import_review_pr()
        comments = [{"author": "user", "body": "Comment", "resolved": False}]
        with (
            patch.object(mod, "fetch_pr_comments", return_value=comments),
            patch.object(mod, "output_json") as mock_output,
        ):
            mod.main(["1", "--fetch-comments"])
        output_data = mock_output.call_args[0][0]
        assert "comments" in output_data
        assert output_data["unresolved_count"] == 1


# ---------------------------------------------------------------------------
# Module-level checks
# ---------------------------------------------------------------------------


class TestModuleCleanup:
    """Verify Copilot references are fully removed."""

    def test_no_trigger_copilot_function(self):
        """trigger_copilot_review function should not exist."""
        mod = _import_review_pr()
        assert not hasattr(mod, "trigger_copilot_review")

    def test_no_wait_for_copilot_function(self):
        """wait_for_copilot function should not exist."""
        mod = _import_review_pr()
        assert not hasattr(mod, "wait_for_copilot")

    def test_has_wait_for_coderabbit_function(self):
        """wait_for_coderabbit function should exist."""
        mod = _import_review_pr()
        assert hasattr(mod, "wait_for_coderabbit")
