"""Tests for spawn_session.py — session creation + async reload.

Run with: source venv/bin/activate && pytest _qsprocess_opencode/tests/ -v
"""

from __future__ import annotations

import http.client
import json
import urllib.error
from unittest.mock import MagicMock, patch

import spawn_session


def _mock_response(data: bytes = b"", status: int = 200) -> MagicMock:
    """Create a mock ``http.client.HTTPResponse`` for ``urlopen``."""
    resp = MagicMock(spec=http.client.HTTPResponse)
    resp.read.return_value = data
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestApi:
    """Tests for the _api helper."""

    def test_api_returns_parsed_json(self) -> None:
        """_api should return parsed JSON dict on success."""
        resp_data = json.dumps({"id": "sess-123"}).encode()
        with patch(
            "spawn_session.urllib.request.urlopen",
            return_value=_mock_response(resp_data),
        ):
            result = spawn_session._api("POST", "/session", {"title": "test"})

        assert result == {"id": "sess-123"}

    def test_api_sends_directory_header(self) -> None:
        """_api should send x-opencode-directory header when given."""
        with patch(
            "spawn_session.urllib.request.urlopen",
            return_value=_mock_response(b"{}"),
        ) as mock_urlopen:
            spawn_session._api("GET", "/agent", directory="/my/dir")

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("X-opencode-directory") == "/my/dir"

    def test_api_raises_system_exit_on_error(self) -> None:
        """_api should raise SystemExit on connection failure."""
        import pytest

        with (
            patch(
                "spawn_session.urllib.request.urlopen",
                side_effect=urllib.error.URLError("refused"),
            ),
            pytest.raises(SystemExit),
        ):
            spawn_session._api("POST", "/session", {"title": "test"})

    def test_api_empty_response_returns_none(self) -> None:
        """_api should return None for empty response body."""
        with patch(
            "spawn_session.urllib.request.urlopen",
            return_value=_mock_response(b""),
        ):
            result = spawn_session._api("POST", "/session/x/prompt_async", {"agent": "a"})

        assert result is None


class TestRequestReloadAsync:
    """Tests for request_reload_async (detached fire-and-forget)."""

    def test_spawns_detached_shell_with_sleep_and_curl(self) -> None:
        """Should spawn 'sh -c sleep N; curl ...' with start_new_session."""
        with patch("spawn_session.subprocess.Popen") as mock_popen:
            spawn_session.request_reload_async(directory="/my/worktree", delay_seconds=3.0)

        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        cmd = args[0]
        assert cmd[0] == "sh"
        assert cmd[1] == "-c"
        shell_script = cmd[2]
        assert "sleep 3.0" in shell_script
        assert "/instance/reload" in shell_script
        assert "x-opencode-directory: /my/worktree" in shell_script
        assert kwargs["start_new_session"] is True
        assert kwargs["close_fds"] is True

    def test_defaults_directory_to_cwd(self) -> None:
        """When directory=None, should use os.getcwd()."""
        with (
            patch("spawn_session.subprocess.Popen") as mock_popen,
            patch("spawn_session.os.getcwd", return_value="/fallback/dir"),
        ):
            spawn_session.request_reload_async(directory=None)

        shell_script = mock_popen.call_args[0][0][2]
        assert "/fallback/dir" in shell_script

    def test_os_error_does_not_raise(self) -> None:
        """Should handle OSError gracefully (just warn)."""
        with patch("spawn_session.subprocess.Popen", side_effect=OSError("no sh")):
            # Should not raise
            spawn_session.request_reload_async(directory="/tmp/test")


class TestSpawnSession:
    """Tests for spawn_session() — blank session + async reload."""

    def test_creates_session_then_fires_reload(self) -> None:
        """Session creation must happen BEFORE reload."""
        call_order: list[str] = []

        def tracking_api(
            method: str,
            path: str,
            body: dict | None = None,
            *,
            directory: str | None = None,
            timeout: int = 10,
        ) -> dict | list | None:
            call_order.append(f"{method} {path}")
            if path == "/session":
                return {"id": "sess-123"}
            return None

        def tracking_reload(**kwargs: object) -> None:
            call_order.append("reload_async")

        with (
            patch.object(spawn_session, "_api", side_effect=tracking_api),
            patch.object(
                spawn_session,
                "request_reload_async",
                side_effect=tracking_reload,
            ),
        ):
            result = spawn_session.spawn_session(
                agent="qs-test-QS-1",
                directory="/tmp/test",
            )

        assert call_order == ["POST /session", "reload_async"]
        assert result["status"] == "session_created"
        assert result["session_id"] == "sess-123"
        assert result["agent"] == "qs-test-QS-1"

    def test_no_prompt_sent(self) -> None:
        """spawn_session should NOT send any prompt."""
        api_calls: list[str] = []

        def tracking_api(
            method: str,
            path: str,
            body: dict | None = None,
            *,
            directory: str | None = None,
            timeout: int = 10,
        ) -> dict | list | None:
            api_calls.append(path)
            if path == "/session":
                return {"id": "sess-123"}
            return None

        with (
            patch.object(spawn_session, "_api", side_effect=tracking_api),
            patch.object(spawn_session, "request_reload_async"),
        ):
            spawn_session.spawn_session(
                agent="qs-test-QS-1",
                directory="/tmp/test",
            )

        assert not any("prompt" in call for call in api_calls)

    def test_passes_delay_to_reload(self) -> None:
        """delay_seconds should be forwarded to request_reload_async."""
        with (
            patch.object(spawn_session, "_api", return_value={"id": "sess-123"}),
            patch.object(spawn_session, "request_reload_async") as mock_reload,
        ):
            spawn_session.spawn_session(
                agent="qs-test-QS-1",
                directory="/tmp/test",
                delay_seconds=5.0,
            )

        mock_reload.assert_called_once_with(directory="/tmp/test", delay_seconds=5.0)

    def test_instructions_include_agent_and_title(self) -> None:
        """Output should include human-readable instructions."""
        with (
            patch.object(spawn_session, "_api", return_value={"id": "sess-123"}),
            patch.object(spawn_session, "request_reload_async"),
        ):
            result = spawn_session.spawn_session(
                agent="qs-review-task-QS-42",
                title="QS-42: review-task",
                directory="/tmp/test",
            )

        assert "qs-review-task-QS-42" in result["instructions"]
        assert "QS-42: review-task" in result["instructions"]
        assert "Refresh browser" in result["instructions"]
        assert "picker" in result["instructions"]

    def test_title_defaults_to_agent(self) -> None:
        """When title is None, agent name should be used."""
        with (
            patch.object(spawn_session, "_api", return_value={"id": "sess-123"}),
            patch.object(spawn_session, "request_reload_async"),
        ):
            result = spawn_session.spawn_session(
                agent="qs-test-QS-1",
                directory="/tmp/test",
            )

        assert result["title"] == "qs-test-QS-1"

    def test_session_creation_failure_exits(self) -> None:
        """If POST /session fails, should sys.exit."""
        import pytest

        with (
            patch.object(
                spawn_session,
                "_api",
                side_effect=SystemExit(1),
            ),
            pytest.raises(SystemExit),
        ):
            spawn_session.spawn_session(
                agent="qs-test-QS-1",
                directory="/tmp/test",
            )


class TestMainCli:
    """Tests for the CLI entry point."""

    def test_default_delay_is_2(self) -> None:
        """Default --delay should be 2.0."""
        with (
            patch.object(
                spawn_session,
                "spawn_session",
                return_value={
                    "session_id": "x",
                    "agent": "a",
                    "title": "t",
                    "directory": "/d",
                    "status": "session_created",
                    "instructions": "ok",
                    "delay_seconds": 2.0,
                },
            ) as mock_spawn,
            patch("spawn_session.output_json"),
            patch(
                "sys.argv",
                [
                    "spawn_session.py",
                    "--agent",
                    "qs-test-QS-1",
                    "--directory",
                    "/d",
                ],
            ),
        ):
            spawn_session.main()

        assert mock_spawn.call_args[1]["delay_seconds"] == 2.0

    def test_custom_delay(self) -> None:
        """--delay should be passed through."""
        with (
            patch.object(
                spawn_session,
                "spawn_session",
                return_value={
                    "session_id": "x",
                    "agent": "a",
                    "title": "t",
                    "directory": "/d",
                    "status": "session_created",
                    "instructions": "ok",
                    "delay_seconds": 5.0,
                },
            ) as mock_spawn,
            patch("spawn_session.output_json"),
            patch(
                "sys.argv",
                [
                    "spawn_session.py",
                    "--agent",
                    "qs-test-QS-1",
                    "--directory",
                    "/d",
                    "--delay",
                    "5",
                ],
            ),
        ):
            spawn_session.main()

        assert mock_spawn.call_args[1]["delay_seconds"] == 5.0
