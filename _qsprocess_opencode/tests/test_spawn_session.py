"""Tests for spawn_session.py — reload integration + curl-based API.

Run with: source venv/bin/activate && pytest _qsprocess_opencode/tests/ -v
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import spawn_session


class TestReloadInstance:
    """Tests for the reload_instance() call in spawn_session."""

    def test_reload_called_before_session_creation(self) -> None:
        """reload_instance() must be called BEFORE session creation."""
        call_order: list[str] = []

        def tracking_reload(*, directory: str | None = None) -> bool:
            call_order.append("reload")
            return True

        def tracking_wait(
            agent: str, *, directory: str | None = None, timeout: float = 15, poll_interval: float = 0.5
        ) -> bool:
            call_order.append("wait_for_agent")
            return True

        def tracking_curl(
            method: str,
            path: str,
            body: dict | None = None,
            *,
            directory: str | None = None,
            timeout: int = 10,
        ) -> tuple[bool, str]:
            call_order.append(f"{method} {path}")
            if path == "/session":
                return True, json.dumps({"id": "sess-123"})
            return True, ""

        with (
            patch.object(spawn_session, "_reload_instance", side_effect=tracking_reload),
            patch.object(spawn_session, "_wait_for_agent", side_effect=tracking_wait),
            patch.object(spawn_session, "_curl", side_effect=tracking_curl),
        ):
            spawn_session.spawn_session(
                agent="qs-test-QS-1",
                prompt="test",
                directory="/tmp/test",
            )

        assert call_order[0] == "reload"
        assert call_order[1] == "wait_for_agent"
        assert call_order[2] == "POST /session"

    def test_reload_failure_does_not_abort(self) -> None:
        """If reload fails, spawn_session should still create the session."""
        call_order: list[str] = []

        def failing_reload(*, directory: str | None = None) -> bool:
            call_order.append("reload_failed")
            return False

        def tracking_curl(
            method: str,
            path: str,
            body: dict | None = None,
            *,
            directory: str | None = None,
            timeout: int = 10,
        ) -> tuple[bool, str]:
            call_order.append(f"{method} {path}")
            if path == "/session":
                return True, json.dumps({"id": "sess-123"})
            return True, ""

        with (
            patch.object(spawn_session, "_reload_instance", side_effect=failing_reload),
            patch.object(spawn_session, "_wait_for_agent", return_value=False),
            patch.object(spawn_session, "_curl", side_effect=tracking_curl),
        ):
            result = spawn_session.spawn_session(
                agent="qs-test-QS-1",
                prompt="test",
                directory="/tmp/test",
            )

        assert result["status"] == "spawned"
        assert "POST /session" in call_order

    def test_reload_passes_directory_in_header(self) -> None:
        """_reload_instance should pass directory via -H flag to curl."""
        with patch("spawn_session.subprocess.Popen") as mock_popen:
            spawn_session._reload_instance(directory="/my/worktree")

        cmd = mock_popen.call_args[0][0]
        h_idx = cmd.index("-H")
        assert "x-opencode-directory: /my/worktree" in cmd[h_idx + 1]

    def test_reload_uses_start_new_session(self) -> None:
        """Reload must be fully detached (start_new_session=True)."""
        with patch("spawn_session.subprocess.Popen") as mock_popen:
            spawn_session._reload_instance(directory="/tmp/test")

        _, kwargs = mock_popen.call_args
        assert kwargs.get("start_new_session") is True


class TestWaitForAgent:
    """Tests for _wait_for_agent polling."""

    def test_agent_found_immediately(self) -> None:
        """Returns True when agent is already loaded."""
        agents = [{"name": "qs-review-task-QS-42"}, {"name": "build"}]
        with patch.object(spawn_session, "_curl", return_value=(True, json.dumps(agents))):
            result = spawn_session._wait_for_agent("qs-review-task-QS-42", timeout=1, poll_interval=0.1)
        assert result is True

    def test_agent_found_after_polling(self) -> None:
        """Returns True when agent appears after a few polls."""
        call_count = 0

        def evolving_curl(
            method: str,
            path: str,
            body: dict | None = None,
            *,
            directory: str | None = None,
            timeout: int = 10,
        ) -> tuple[bool, str]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return True, json.dumps([{"name": "build"}])
            return True, json.dumps([{"name": "build"}, {"name": "qs-test-QS-1"}])

        with patch.object(spawn_session, "_curl", side_effect=evolving_curl):
            result = spawn_session._wait_for_agent("qs-test-QS-1", timeout=5, poll_interval=0.1)

        assert result is True
        assert call_count >= 3

    def test_agent_not_found_timeout(self) -> None:
        """Returns False when agent never appears."""
        with patch.object(spawn_session, "_curl", return_value=(True, json.dumps([{"name": "build"}]))):
            result = spawn_session._wait_for_agent("qs-missing-QS-99", timeout=0.3, poll_interval=0.1)
        assert result is False

    def test_curl_failure_keeps_polling(self) -> None:
        """Curl failures don't abort — keeps polling until timeout."""
        call_count = 0

        def flaky_curl(
            method: str,
            path: str,
            body: dict | None = None,
            *,
            directory: str | None = None,
            timeout: int = 10,
        ) -> tuple[bool, str]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return False, ""
            return True, json.dumps([{"name": "qs-test-QS-1"}])

        with patch.object(spawn_session, "_curl", side_effect=flaky_curl):
            result = spawn_session._wait_for_agent("qs-test-QS-1", timeout=5, poll_interval=0.1)

        assert result is True


class TestCurl:
    """Tests for the _curl helper."""

    def test_curl_builds_correct_command(self) -> None:
        """_curl should build the right curl command."""
        with patch("spawn_session.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0, stdout='{"id":"x"}', stderr="")
            ok, body = spawn_session._curl(
                "POST",
                "/session",
                {"title": "test"},
                directory="/my/dir",
                timeout=10,
            )

        assert ok is True
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "curl"
        assert "-X" in cmd
        assert "POST" in cmd
        assert any("/session" in a for a in cmd)
        # Check directory header
        h_idx = [i for i, a in enumerate(cmd) if a == "-H" and i + 1 < len(cmd)]
        headers = [cmd[i + 1] for i in h_idx]
        assert any("x-opencode-directory: /my/dir" in h for h in headers)

    def test_curl_failure_returns_false(self) -> None:
        """_curl should return (False, ...) when curl exits non-zero."""
        with patch("spawn_session.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 1, stdout="", stderr="timeout")
            ok, body = spawn_session._curl("POST", "/instance/reload")

        assert ok is False
