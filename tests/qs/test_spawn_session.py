"""Tests for ``scripts/qs/spawn_session.py``.

Covers:

- Happy path: POST /session + POST /session/<id>/prompt_async, JSON
  payload on stdout, exit 0.
- Hard negative: NO ``/instance/reload`` HTTP call, NO
  ``subprocess.Popen``, NO ``--delay`` flag (argparse exits 2 on it).
- CLI-fallback semantics (AC #5 / #7): every failure class — URLError,
  HTTPError, OSError, TimeoutError, JSONDecodeError, missing ``id`` —
  routes to ``fallback_cli`` (opencode on PATH) or
  ``fallback_unavailable`` (no opencode on PATH).
- Orphan-session handling (AC #6): if prompt_async fails after a
  successful /session, attempt DELETE /session/<id>; on DELETE
  success emit ``fallback_cli``; on DELETE failure emit
  ``session_orphaned`` with the session id surfaced.
- Default + custom prompt + default title behaviour.

Format-agnostic assertions (Task 5.2 / F12 guard): stdout is parsed
via ``json.loads`` — never raw substring matching — because the new
``scripts/qs/utils.py:output_json`` pretty-prints (``indent=2``).
"""

from __future__ import annotations

import http.client
import json
import shlex
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #


def _mock_response(data: bytes = b"", status: int = 200) -> MagicMock:
    """Build a mock ``http.client.HTTPResponse`` suitable for ``urlopen``."""
    resp = MagicMock(spec=http.client.HTTPResponse)
    resp.read.return_value = data
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _http_error(status: int = 500, msg: str = "Server Error") -> urllib.error.HTTPError:
    """Build an ``HTTPError`` instance suitable as a urlopen side_effect."""
    return urllib.error.HTTPError(
        "http://127.0.0.1:4096/session", status, msg, {}, None,  # type: ignore[arg-type]
    )


def _sequence_urlopen(responses: Iterable[object]) -> MagicMock:
    """Return a urlopen mock whose side_effect cycles through ``responses``.

    Each entry is either a ``MagicMock`` (returned) or an exception
    instance (raised). Exception classes are NOT supported — pass an
    instance so the test reads naturally.
    """
    items: list[object] = list(responses)

    def _side(*_args: object, **_kwargs: object) -> object:
        if not items:
            raise AssertionError("urlopen called more times than expected")
        value = items.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    mock = MagicMock(side_effect=_side)
    return mock


def _run_main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int | None:
    """Invoke ``spawn_session.main()`` with the given argv; return SystemExit code (or None)."""
    import spawn_session  # type: ignore[import-not-found]

    monkeypatch.setattr("sys.argv", ["spawn_session.py", *argv])
    try:
        spawn_session.main()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else None
    return 0


# --------------------------------------------------------------------------- #
# Task 5.1 — _api directory header
# --------------------------------------------------------------------------- #


def test_api_helper_sends_directory_header() -> None:
    """``_api`` sends the ``x-opencode-directory`` header when ``directory`` is set."""
    import spawn_session  # type: ignore[import-not-found]

    with patch.object(
        spawn_session.urllib.request, "urlopen",
        return_value=_mock_response(b"{}"),
    ) as mock_urlopen:
        spawn_session._api(
            "POST", "/session", {"title": "t"}, directory="/my/dir",
        )

    req = mock_urlopen.call_args[0][0]
    assert req.get_header("X-opencode-directory") == "/my/dir"


# --------------------------------------------------------------------------- #
# Task 5.2 — happy path
# --------------------------------------------------------------------------- #


def test_spawn_session_happy_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """``main()`` posts /session, then prompt_async, emits AC #1 payload, exit 0."""
    import spawn_session  # type: ignore[import-not-found]

    calls: list[tuple[str, str, bytes | None]] = []

    def _capture(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        calls.append((req.get_method(), req.full_url, req.data))
        if req.full_url.endswith("/session"):
            return _mock_response(json.dumps({"id": "sess-123"}).encode())
        return _mock_response(b"")  # 204 from prompt_async

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _capture)

    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/wt",
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "session_created"
    assert payload["session_id"] == "sess-123"
    assert payload["agent"] == "qs-create-plan"
    assert payload["title"] == "qs-create-plan"  # default = agent name
    assert payload["directory"] == "/tmp/wt"
    assert payload["prompt"] == "Begin your phase protocol."

    # Call ordering — session create, then prompt_async
    assert len(calls) == 2
    method0, url0, body0 = calls[0]
    method1, url1, body1 = calls[1]
    assert method0 == "POST" and url0.endswith("/session")
    assert method1 == "POST" and url1.endswith("/session/sess-123/prompt_async")

    # Body shape pinned (Task 1.3)
    session_body = json.loads(body0 or b"{}")
    assert session_body == {"title": "qs-create-plan"}
    prompt_body = json.loads(body1 or b"{}")
    assert prompt_body == {
        "agent": "qs-create-plan",
        "parts": [{"type": "text", "text": "Begin your phase protocol."}],
    }


# --------------------------------------------------------------------------- #
# Task 5.3 — NO /instance/reload, NO subprocess.Popen
# --------------------------------------------------------------------------- #


def test_no_instance_reload_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path must not touch /instance/reload nor spawn a subprocess."""
    import spawn_session  # type: ignore[import-not-found]

    seen_urls: list[str] = []

    def _capture(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        seen_urls.append(req.full_url)
        if req.full_url.endswith("/session"):
            return _mock_response(json.dumps({"id": "sess-X"}).encode())
        return _mock_response(b"")

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _capture)
    with patch.object(subprocess, "Popen") as mock_popen:
        exit_code = _run_main(
            monkeypatch,
            "--agent", "qs-create-plan",
            "--directory", "/tmp/wt",
        )

    assert exit_code == 0
    assert not any("/instance/reload" in url for url in seen_urls), seen_urls
    mock_popen.assert_not_called()


# --------------------------------------------------------------------------- #
# Task 5.4 — --delay flag rejected
# --------------------------------------------------------------------------- #


def test_no_delay_flag_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """argparse rejects ``--delay`` with exit 2 ('unrecognized arguments')."""
    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/wt",
        "--delay", "5",
    )
    assert exit_code == 2


# --------------------------------------------------------------------------- #
# Task 5.5 / 5.6 — prompt default + custom
# --------------------------------------------------------------------------- #


def test_default_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``--prompt``, the prompt_async body carries the default kickoff."""
    import spawn_session  # type: ignore[import-not-found]

    bodies: list[bytes | None] = []

    def _capture(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        bodies.append(req.data)
        if req.full_url.endswith("/session"):
            return _mock_response(json.dumps({"id": "sess-X"}).encode())
        return _mock_response(b"")

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _capture)
    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/wt",
    )
    assert exit_code == 0
    prompt_body = json.loads(bodies[1] or b"{}")
    assert prompt_body["parts"][0]["text"] == "Begin your phase protocol."


def test_custom_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--prompt`` is forwarded verbatim into prompt_async."""
    import spawn_session  # type: ignore[import-not-found]

    bodies: list[bytes | None] = []

    def _capture(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        bodies.append(req.data)
        if req.full_url.endswith("/session"):
            return _mock_response(json.dumps({"id": "sess-X"}).encode())
        return _mock_response(b"")

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _capture)
    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/wt",
        "--prompt", "Custom kickoff",
    )
    assert exit_code == 0
    prompt_body = json.loads(bodies[1] or b"{}")
    assert prompt_body["parts"][0]["text"] == "Custom kickoff"


# --------------------------------------------------------------------------- #
# Task 5.7 — server unreachable → fallback_cli
# --------------------------------------------------------------------------- #


def test_server_unreachable_fallback_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """URLError + opencode on PATH → fallback_cli, exit 0."""
    import spawn_session  # type: ignore[import-not-found]

    monkeypatch.setattr(
        spawn_session.urllib.request, "urlopen",
        MagicMock(side_effect=urllib.error.URLError("connection refused")),
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/opencode" if name == "opencode" else None)

    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
        "--prompt", "P",
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "fallback_cli"
    assert payload["agent"] == "qs-create-plan"
    assert payload["directory"] == "/tmp/x"
    expected_cmd = (
        f"opencode {shlex.quote('/tmp/x')} "
        f"--agent {shlex.quote('qs-create-plan')} "
        f"--prompt {shlex.quote('P')}"
    )
    assert payload["new_context_cli"] == expected_cmd
    assert "connection refused" in payload["detail"]


# --------------------------------------------------------------------------- #
# Task 5.8 — server unreachable, no CLI → fallback_unavailable
# --------------------------------------------------------------------------- #


def test_server_unreachable_no_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """URLError + no opencode on PATH → fallback_unavailable, exit 2."""
    import spawn_session  # type: ignore[import-not-found]

    monkeypatch.setattr(
        spawn_session.urllib.request, "urlopen",
        MagicMock(side_effect=urllib.error.URLError("connection refused")),
    )
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
    )
    assert exit_code == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fallback_unavailable"
    assert payload["agent"] == "qs-create-plan"
    assert payload["directory"] == "/tmp/x"
    assert "opencode CLI not on PATH" in payload["detail"]
    assert "connection refused" in payload["detail"]
    # The CLI command is NOT promised in this shape (per AC #5).
    assert "new_context_cli" not in payload


# --------------------------------------------------------------------------- #
# Task 5.9 — timeout → fallback path
# --------------------------------------------------------------------------- #


def test_timeout_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A socket.timeout raised by urlopen routes through the fallback path."""
    import spawn_session  # type: ignore[import-not-found]

    monkeypatch.setattr(
        spawn_session.urllib.request, "urlopen",
        MagicMock(side_effect=socket.timeout("timed out")),
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/opencode" if name == "opencode" else None)

    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fallback_cli"
    assert "timed out" in payload["detail"]


# --------------------------------------------------------------------------- #
# Task 5.10 — HTTPError → fallback path
# --------------------------------------------------------------------------- #


def test_http_error_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """An HTTPError (e.g. 500) from /session routes through the fallback path."""
    import spawn_session  # type: ignore[import-not-found]

    monkeypatch.setattr(
        spawn_session.urllib.request, "urlopen",
        MagicMock(side_effect=_http_error(500, "boom")),
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/opencode" if name == "opencode" else None)

    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fallback_cli"


# --------------------------------------------------------------------------- #
# Task 5.11 — malformed JSON → fallback path
# --------------------------------------------------------------------------- #


def test_malformed_json_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-JSON response body from /session triggers the fallback path."""
    import spawn_session  # type: ignore[import-not-found]

    monkeypatch.setattr(
        spawn_session.urllib.request, "urlopen",
        MagicMock(return_value=_mock_response(b"not json")),
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/opencode" if name == "opencode" else None)

    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fallback_cli"


# --------------------------------------------------------------------------- #
# Task 5.12 — missing session id → fallback
# --------------------------------------------------------------------------- #


def test_missing_session_id_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """``/session`` returning ``{}`` (no ``id``) triggers the fallback path."""
    import spawn_session  # type: ignore[import-not-found]

    monkeypatch.setattr(
        spawn_session.urllib.request, "urlopen",
        MagicMock(return_value=_mock_response(b"{}")),
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/opencode" if name == "opencode" else None)

    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fallback_cli"


def test_non_dict_session_response_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """``/session`` returning a JSON array (not a dict) triggers fallback.

    Defense-in-depth (sibling of test_missing_session_id_falls_back):
    the server contract returns ``Session.Info`` (an object), but if a
    future server version returns a different shape — or a proxy
    rewrites it — we still want a clean fallback rather than a
    ``TypeError`` on ``.get('id')``.
    """
    import spawn_session  # type: ignore[import-not-found]

    monkeypatch.setattr(
        spawn_session.urllib.request, "urlopen",
        MagicMock(return_value=_mock_response(b"[]")),
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/opencode" if name == "opencode" else None)

    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fallback_cli"
    assert "not a dict" in payload["detail"]


# --------------------------------------------------------------------------- #
# Task 5.13 — prompt_async failure → DELETE attempt → fallback_cli
# --------------------------------------------------------------------------- #


def test_prompt_async_failure_attempts_delete(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """prompt_async fails → DELETE attempted → on success, fallback_cli."""
    import spawn_session  # type: ignore[import-not-found]

    seen_requests: list[urllib.request.Request] = []

    def _side(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        seen_requests.append(req)
        url = req.full_url
        method = req.get_method()
        if method == "POST" and url.endswith("/session"):
            return _mock_response(json.dumps({"id": "sess-X"}).encode())
        if method == "POST" and url.endswith("/session/sess-X/prompt_async"):
            raise _http_error(500, "boom")
        if method == "DELETE" and url.endswith("/session/sess-X"):
            return _mock_response(b"true")
        raise AssertionError(f"unexpected call: {method} {url}")

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _side)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/opencode" if name == "opencode" else None)

    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fallback_cli"

    # DELETE was actually issued with the right URL + directory header
    delete_calls = [r for r in seen_requests if r.get_method() == "DELETE"]
    assert len(delete_calls) == 1
    assert delete_calls[0].full_url.endswith("/session/sess-X")
    assert delete_calls[0].get_header("X-opencode-directory") == "/tmp/x"


# --------------------------------------------------------------------------- #
# Task 5.14 — prompt_async failure + DELETE failure → session_orphaned
# --------------------------------------------------------------------------- #


def test_prompt_async_failure_delete_fails_emits_orphan(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """If DELETE also fails, surface ``session_orphaned`` with the id, exit 2."""
    import spawn_session  # type: ignore[import-not-found]

    def _side(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        url = req.full_url
        method = req.get_method()
        if method == "POST" and url.endswith("/session"):
            return _mock_response(json.dumps({"id": "sess-X"}).encode())
        if method == "POST" and url.endswith("/session/sess-X/prompt_async"):
            raise _http_error(500, "prompt boom")
        if method == "DELETE" and url.endswith("/session/sess-X"):
            raise _http_error(404, "delete boom")
        raise AssertionError(f"unexpected call: {method} {url}")

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _side)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/opencode" if name == "opencode" else None)

    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
    )
    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "session_orphaned"
    assert payload["session_id"] == "sess-X"
    assert payload["agent"] == "qs-create-plan"
    assert payload["directory"] == "/tmp/x"
    assert "new_context_cli" in payload
    assert "prompt boom" in payload["detail"] or "delete boom" in payload["detail"]


# --------------------------------------------------------------------------- #
# Task 5.15 — stderr message on failure
# --------------------------------------------------------------------------- #


def test_stderr_message_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A human-readable line lands on stderr describing the failure."""
    import spawn_session  # type: ignore[import-not-found]

    monkeypatch.setattr(
        spawn_session.urllib.request, "urlopen",
        MagicMock(side_effect=urllib.error.URLError("connection refused")),
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/opencode" if name == "opencode" else None)

    _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
    )
    err = capsys.readouterr().err
    assert err.strip(), "expected a human-readable line on stderr"
    # Single readable line (no JSON-style {} structure on stderr).
    assert "{" not in err.splitlines()[0]


# --------------------------------------------------------------------------- #
# Task 5.16 — title default = agent name
# --------------------------------------------------------------------------- #


def test_title_defaults_to_agent_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``--title``, ``POST /session`` body uses the agent name."""
    import spawn_session  # type: ignore[import-not-found]

    bodies: list[bytes | None] = []

    def _capture(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        bodies.append(req.data)
        if req.full_url.endswith("/session"):
            return _mock_response(json.dumps({"id": "sess-X"}).encode())
        return _mock_response(b"")

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _capture)
    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
    )
    assert exit_code == 0
    session_body = json.loads(bodies[0] or b"{}")
    assert session_body == {"title": "qs-create-plan"}


# --------------------------------------------------------------------------- #
# Extra coverage — explicit title argument
# --------------------------------------------------------------------------- #


def test_custom_title_forwarded_to_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--title`` is forwarded verbatim into ``POST /session`` body."""
    import spawn_session  # type: ignore[import-not-found]

    bodies: list[bytes | None] = []

    def _capture(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        bodies.append(req.data)
        if req.full_url.endswith("/session"):
            return _mock_response(json.dumps({"id": "sess-X"}).encode())
        return _mock_response(b"")

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _capture)
    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
        "--title", "QS_42: Foo",
    )
    assert exit_code == 0
    session_body = json.loads(bodies[0] or b"{}")
    assert session_body == {"title": "QS_42: Foo"}


# --------------------------------------------------------------------------- #
# Extra coverage — port override + _api list/None branches
# --------------------------------------------------------------------------- #


def test_api_returns_none_on_empty_body() -> None:
    """``_api`` returns ``None`` when the server replies with empty body."""
    import spawn_session  # type: ignore[import-not-found]

    with patch.object(
        spawn_session.urllib.request, "urlopen",
        return_value=_mock_response(b""),
    ):
        result = spawn_session._api("POST", "/session/x/prompt_async", {"a": 1})
    assert result is None


def test_base_url_honours_env_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OPENCODE_SERVER_PORT`` overrides the default port."""
    import spawn_session  # type: ignore[import-not-found]

    monkeypatch.setenv("OPENCODE_SERVER_PORT", "9001")
    assert spawn_session._base_url() == "http://127.0.0.1:9001"


def test_base_url_defaults_to_4096(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``OPENCODE_SERVER_PORT``, default port is 4096."""
    import spawn_session  # type: ignore[import-not-found]

    monkeypatch.delenv("OPENCODE_SERVER_PORT", raising=False)
    assert spawn_session._base_url() == "http://127.0.0.1:4096"
