"""Tests for spawn_session.py — reload integration + urllib-based API.

Run with: source venv/bin/activate && pytest _qsprocess_opencode/tests/ -v
"""

from __future__ import annotations

import http.client
import json
import urllib.error
from io import BytesIO
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


class TestReloadInstance:
    """Tests for the _reload_instance() urllib call."""

    def test_reload_success(self) -> None:
        """_reload_instance returns True on 200."""
        with patch("spawn_session.urllib.request.urlopen", return_value=_mock_response()):
            result = spawn_session._reload_instance(directory="/tmp/test")

        assert result is True

    def test_reload_passes_directory_header(self) -> None:
        """_reload_instance sets x-opencode-directory header."""
        with patch("spawn_session.urllib.request.urlopen", return_value=_mock_response()) as mock_urlopen:
            spawn_session._reload_instance(directory="/my/worktree")

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("X-opencode-directory") == "/my/worktree"

    def test_reload_sends_empty_body(self) -> None:
        """_reload_instance sends data=b'' for clean Content-Length: 0."""
        with patch("spawn_session.urllib.request.urlopen", return_value=_mock_response()) as mock_urlopen:
            spawn_session._reload_instance(directory="/tmp/test")

        req = mock_urlopen.call_args[0][0]
        assert req.data == b""

    def test_reload_uses_post_method(self) -> None:
        """_reload_instance uses POST."""
        with patch("spawn_session.urllib.request.urlopen", return_value=_mock_response()) as mock_urlopen:
            spawn_session._reload_instance(directory="/tmp/test")

        req = mock_urlopen.call_args[0][0]
        assert req.get_method() == "POST"

    def test_reload_uses_10s_timeout(self) -> None:
        """_reload_instance uses timeout=10."""
        with patch("spawn_session.urllib.request.urlopen", return_value=_mock_response()) as mock_urlopen:
            spawn_session._reload_instance(directory="/tmp/test")

        _, kwargs = mock_urlopen.call_args
        assert kwargs.get("timeout") == 10

    def test_reload_failure_returns_false(self) -> None:
        """_reload_instance returns False on network error."""
        with patch(
            "spawn_session.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = spawn_session._reload_instance(directory="/tmp/test")

        assert result is False

    def test_reload_http_error_returns_false(self) -> None:
        """_reload_instance returns False on HTTP 400/500."""
        err = urllib.error.HTTPError(
            url="http://127.0.0.1:4096/instance/reload",
            code=400,
            msg="Bad Request",
            hdrs={},  # type: ignore[arg-type]
            fp=BytesIO(b""),
        )
        with patch("spawn_session.urllib.request.urlopen", side_effect=err):
            result = spawn_session._reload_instance(directory="/tmp/test")

        assert result is False

    def test_reload_warns_when_directory_none(self) -> None:
        """_reload_instance warns when directory is None."""
        import io

        captured_stderr = io.StringIO()
        with (
            patch("spawn_session.urllib.request.urlopen", return_value=_mock_response()),
            patch("sys.stderr", captured_stderr),
        ):
            result = spawn_session._reload_instance(directory=None)

        assert result is True
        assert "reload called without directory" in captured_stderr.getvalue()

    def test_reload_called_before_session_creation(self) -> None:
        """_reload_instance must be called BEFORE session creation."""
        call_order: list[str] = []

        def tracking_reload(*, directory: str | None = None) -> bool:
            call_order.append("reload")
            return True

        def tracking_wait(
            agent: str,
            *,
            directory: str | None = None,
            timeout: float = 15,
            poll_interval: float = 0.5,
        ) -> bool:
            call_order.append("wait_for_agent")
            return True

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

        with (
            patch.object(spawn_session, "_reload_instance", side_effect=tracking_reload),
            patch.object(spawn_session, "_wait_for_agent", side_effect=tracking_wait),
            patch.object(spawn_session, "_api", side_effect=tracking_api),
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

        with (
            patch.object(spawn_session, "_reload_instance", side_effect=failing_reload),
            patch.object(spawn_session, "_wait_for_agent", return_value=False),
            patch.object(spawn_session, "_api", side_effect=tracking_api),
        ):
            result = spawn_session.spawn_session(
                agent="qs-test-QS-1",
                prompt="test",
                directory="/tmp/test",
            )

        assert result["status"] == "spawned"
        assert "POST /session" in call_order


class TestWaitForAgent:
    """Tests for _wait_for_agent polling."""

    def test_agent_found_immediately(self) -> None:
        """Returns True when agent is already loaded."""
        agents = [{"name": "qs-review-task-QS-42"}, {"name": "build"}]
        with patch.object(spawn_session, "_api_safe", return_value=(True, agents)):
            result = spawn_session._wait_for_agent("qs-review-task-QS-42", timeout=1, poll_interval=0.1)
        assert result is True

    def test_agent_found_after_polling(self) -> None:
        """Returns True when agent appears after a few polls."""
        call_count = 0

        def evolving_api_safe(
            method: str,
            path: str,
            body: dict | None = None,
            *,
            directory: str | None = None,
            timeout: int = 10,
        ) -> tuple[bool, list | None]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return True, [{"name": "build"}]
            return True, [{"name": "build"}, {"name": "qs-test-QS-1"}]

        with patch.object(spawn_session, "_api_safe", side_effect=evolving_api_safe):
            result = spawn_session._wait_for_agent("qs-test-QS-1", timeout=5, poll_interval=0.1)

        assert result is True
        assert call_count >= 3

    def test_agent_not_found_timeout(self) -> None:
        """Returns False when agent never appears."""
        with patch.object(spawn_session, "_api_safe", return_value=(True, [{"name": "build"}])):
            result = spawn_session._wait_for_agent("qs-missing-QS-99", timeout=0.3, poll_interval=0.1)
        assert result is False

    def test_api_failure_keeps_polling(self) -> None:
        """API failures don't abort — keeps polling until timeout."""
        call_count = 0

        def flaky_api_safe(
            method: str,
            path: str,
            body: dict | None = None,
            *,
            directory: str | None = None,
            timeout: int = 10,
        ) -> tuple[bool, list | None]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return False, None
            return True, [{"name": "qs-test-QS-1"}]

        with patch.object(spawn_session, "_api_safe", side_effect=flaky_api_safe):
            result = spawn_session._wait_for_agent("qs-test-QS-1", timeout=5, poll_interval=0.1)

        assert result is True


class TestApi:
    """Tests for the _api and _api_safe helpers."""

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

    def test_api_safe_returns_tuple_on_success(self) -> None:
        """_api_safe should return (True, parsed_json)."""
        resp_data = json.dumps([{"name": "build"}]).encode()
        with patch(
            "spawn_session.urllib.request.urlopen",
            return_value=_mock_response(resp_data),
        ):
            ok, result = spawn_session._api_safe("GET", "/agent")

        assert ok is True
        assert result == [{"name": "build"}]

    def test_api_safe_returns_false_on_error(self) -> None:
        """_api_safe should return (False, None) on network error."""
        with patch(
            "spawn_session.urllib.request.urlopen",
            side_effect=urllib.error.URLError("refused"),
        ):
            ok, result = spawn_session._api_safe("GET", "/agent")

        assert ok is False
        assert result is None

    def test_api_empty_response_returns_none(self) -> None:
        """_api should return None for empty response body."""
        with patch(
            "spawn_session.urllib.request.urlopen",
            return_value=_mock_response(b""),
        ):
            result = spawn_session._api("POST", "/session/x/prompt_async", {"agent": "a"})

        assert result is None
