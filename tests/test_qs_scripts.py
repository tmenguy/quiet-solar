"""Tests for scripts/qs/ utilities and next_step command builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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

    def _read_script(self, script_path: str) -> str:
        """Read the generated launch script contents."""
        from pathlib import Path

        # claude_launch_command returns "sh /path/to/script.sh"
        path = script_path.removeprefix("sh ")
        return Path(path).read_text()

    def test_without_prompt_unchanged(self):
        """No prompt arg produces script without positional arg."""
        utils = _import_utils()
        script_path = utils.claude_launch_command("/tmp/work", 42, "Fix bug")
        contents = self._read_script(script_path)
        assert "claude" in contents
        assert "/tmp/work" in contents
        assert "QS_42" in contents
        # Should NOT contain a trailing quoted prompt
        assert contents.strip().endswith(f"--name {__import__('shlex').quote('QS_42: Fix bug')}")

    def test_with_prompt_appended(self):
        """When prompt is provided, it is appended in the script."""
        utils = _import_utils()
        script_path = utils.claude_launch_command("/tmp/work", 42, "Fix bug", prompt="/review-story --pr 5")
        contents = self._read_script(script_path)
        assert "claude" in contents
        # The prompt should appear at the end, properly quoted
        assert "'/review-story --pr 5'" in contents or '"/review-story --pr 5"' in contents

    def test_with_prompt_none_same_as_without(self):
        """Passing prompt=None is identical to not passing prompt."""
        utils = _import_utils()
        path_without = utils.claude_launch_command("/tmp/work", 42, "Fix bug")
        path_none = utils.claude_launch_command("/tmp/work", 42, "Fix bug", prompt=None)
        assert self._read_script(path_without) == self._read_script(path_none)

    def test_prompt_with_special_characters(self):
        """Prompt with special shell characters is properly escaped."""
        utils = _import_utils()
        script_path = utils.claude_launch_command(
            "/tmp/work", 42, "Fix bug", prompt='/implement-story --issue 42 --story-file "path with spaces"'
        )
        contents = self._read_script(script_path)
        # Should not raise and should contain escaped content
        assert "claude" in contents
        assert "implement-story" in contents


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

    def _read_new_context_script(self, data: dict) -> str:
        """Read the launch script pointed to by new_context."""
        from pathlib import Path

        # new_context is "sh /path/to/script.sh"
        path = data["new_context"].removeprefix("sh ")
        return Path(path).read_text()

    def test_review_story_transition(self):
        """Generate next-step for implement -> review transition."""
        data = self._run_next_step(
            [
                "--skill",
                "review-story",
                "--issue",
                "42",
                "--pr",
                "5",
                "--work-dir",
                "/tmp/work",
                "--title",
                "Fix bug",
            ]
        )
        assert "same_context" in data
        assert "new_context" in data
        assert "/review-story" in data["same_context"]
        assert "--pr 5" in data["same_context"]
        assert "--issue 42" in data["same_context"]
        script = self._read_new_context_script(data)
        assert "claude" in script

    def test_implement_story_transition(self):
        """Generate next-step for setup -> implement transition."""
        data = self._run_next_step(
            [
                "--skill",
                "implement-story",
                "--issue",
                "42",
                "--story-file",
                "path/to/story.md",
                "--work-dir",
                "/tmp/work",
                "--title",
                "Fix bug",
            ]
        )
        assert "/implement-story" in data["same_context"]
        assert "--issue 42" in data["same_context"]
        assert "--story-file path/to/story.md" in data["same_context"]
        script = self._read_new_context_script(data)
        assert "claude" in script

    def test_finish_story_transition(self):
        """Generate next-step for review -> finish transition."""
        data = self._run_next_step(
            [
                "--skill",
                "finish-story",
                "--pr",
                "5",
                "--story-key",
                "3.2",
                "--work-dir",
                "/tmp/work",
                "--title",
                "Fix bug",
            ]
        )
        assert "/finish-story" in data["same_context"]
        assert "--pr 5" in data["same_context"]
        assert "--story-key 3.2" in data["same_context"]
        script = self._read_new_context_script(data)
        assert "claude" in script
        # No --issue provided; tab title should use PR number, not 0
        assert "QS_5" in script
        assert "QS_0" not in script

    def test_missing_required_args(self):
        """Missing --skill should fail."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "next_step.py"), "--work-dir", "/tmp", "--title", "X"],
            capture_output=True,
            text=True,
            cwd=str(SCRIPTS_DIR),
        )
        assert result.returncode != 0
