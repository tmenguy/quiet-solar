"""Tests for scripts/qs/ utilities and next_step command builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "qs"


# ---------------------------------------------------------------------------
# claude_launch_command tests
# ---------------------------------------------------------------------------

def _import_utils():
    """Import utils from scripts/qs/ by manipulating sys.path."""
    import importlib
    old_path = sys.path[:]
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        # Force reimport to get fresh module
        if "utils" in sys.modules:
            del sys.modules["utils"]
        return importlib.import_module("utils")
    finally:
        sys.path[:] = old_path


class TestClaudeLaunchCommand:
    """Tests for claude_launch_command() in utils.py."""

    def test_without_prompt_unchanged(self):
        """Existing behavior: no prompt arg produces command without positional arg."""
        utils = _import_utils()
        cmd = utils.claude_launch_command("/tmp/work", 42, "Fix bug")
        assert "claude" in cmd
        assert "/tmp/work" in cmd
        assert "QS_42" in cmd
        # Should NOT contain a trailing quoted prompt
        assert cmd.endswith(f"--name {__import__('shlex').quote('QS_42: Fix bug')}")

    def test_with_prompt_appended(self):
        """When prompt is provided, it is appended as quoted positional arg."""
        utils = _import_utils()
        cmd = utils.claude_launch_command("/tmp/work", 42, "Fix bug", prompt="/review-story --pr 5")
        assert "claude" in cmd
        # The prompt should appear at the end, properly quoted
        assert "'/review-story --pr 5'" in cmd or '"/review-story --pr 5"' in cmd

    def test_with_prompt_none_same_as_without(self):
        """Passing prompt=None is identical to not passing prompt."""
        utils = _import_utils()
        cmd_without = utils.claude_launch_command("/tmp/work", 42, "Fix bug")
        cmd_none = utils.claude_launch_command("/tmp/work", 42, "Fix bug", prompt=None)
        assert cmd_without == cmd_none

    def test_prompt_with_special_characters(self):
        """Prompt with special shell characters is properly escaped."""
        utils = _import_utils()
        cmd = utils.claude_launch_command("/tmp/work", 42, "Fix bug", prompt='/implement-story --issue 42 --story-file "path with spaces"')
        # Should not raise and should contain escaped content
        assert "claude" in cmd
        assert "implement-story" in cmd


# ---------------------------------------------------------------------------
# next_step.py tests
# ---------------------------------------------------------------------------

class TestNextStep:
    """Tests for next_step.py script."""

    def _run_next_step(self, args: list[str]) -> dict:
        """Run next_step.py and return parsed JSON output."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "next_step.py"), *args],
            capture_output=True,
            text=True,
            cwd=str(SCRIPTS_DIR),
        )
        assert result.returncode == 0, f"next_step.py failed: {result.stderr}"
        return json.loads(result.stdout)

    def test_review_story_transition(self):
        """Generate next-step for implement -> review transition."""
        data = self._run_next_step([
            "--skill", "review-story",
            "--issue", "42",
            "--pr", "5",
            "--work-dir", "/tmp/work",
            "--title", "Fix bug",
        ])
        assert "same_context" in data
        assert "new_context" in data
        assert "/review-story" in data["same_context"]
        assert "--pr 5" in data["same_context"]
        assert "--issue 42" in data["same_context"]
        assert "claude" in data["new_context"]

    def test_implement_story_transition(self):
        """Generate next-step for setup -> implement transition."""
        data = self._run_next_step([
            "--skill", "implement-story",
            "--issue", "42",
            "--story-file", "path/to/story.md",
            "--work-dir", "/tmp/work",
            "--title", "Fix bug",
        ])
        assert "/implement-story" in data["same_context"]
        assert "--issue 42" in data["same_context"]
        assert "--story-file path/to/story.md" in data["same_context"]
        assert "claude" in data["new_context"]

    def test_finish_story_transition(self):
        """Generate next-step for review -> finish transition."""
        data = self._run_next_step([
            "--skill", "finish-story",
            "--pr", "5",
            "--story-key", "3.2",
            "--work-dir", "/tmp/work",
            "--title", "Fix bug",
        ])
        assert "/finish-story" in data["same_context"]
        assert "--pr 5" in data["same_context"]
        assert "--story-key 3.2" in data["same_context"]
        assert "claude" in data["new_context"]

    def test_missing_required_args(self):
        """Missing --skill should fail."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "next_step.py"),
             "--work-dir", "/tmp", "--title", "X"],
            capture_output=True,
            text=True,
            cwd=str(SCRIPTS_DIR),
        )
        assert result.returncode != 0
