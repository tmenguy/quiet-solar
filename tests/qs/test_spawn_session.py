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
from unittest.mock import MagicMock, patch

import pytest


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
    # Both error halves must be present in the combined detail — using
    # ``and`` (not ``or``) so a future regression that drops one half
    # cannot pass silently. The combined-detail structure is pinned by
    # the literal ``"; DELETE"`` separator.
    assert "prompt boom" in payload["detail"]
    assert "delete boom" in payload["detail"]
    assert "; DELETE" in payload["detail"]


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


# --------------------------------------------------------------------------- #
# Review fix plan #01 — must-fix #1: http.client.InvalidURL crashes the script
# --------------------------------------------------------------------------- #


def test_invalid_url_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """``http.client.InvalidURL`` (malformed port, control chars) routes through fallback.

    A malformed ``OPENCODE_SERVER_PORT`` (e.g. ``"4096 "``), or a
    control character in ``session_id``, causes ``urllib.request`` to
    raise ``InvalidURL`` — which does NOT subclass ``URLError`` /
    ``HTTPError`` / ``OSError``. The fix extends the except tuple to
    catch ``http.client.HTTPException`` (the broader superclass).
    """
    import spawn_session  # type: ignore[import-not-found]

    monkeypatch.setattr(
        spawn_session.urllib.request, "urlopen",
        MagicMock(side_effect=http.client.InvalidURL("bad url")),
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
    assert "bad url" in payload["detail"]


# --------------------------------------------------------------------------- #
# Review fix plan #01 — should-fix #7: session_id must be URL-escaped
# --------------------------------------------------------------------------- #


def test_spawn_session_url_escapes_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server-provided ``id`` with embedded ``/`` is URL-escaped before interpolation.

    Note: ids containing ``..`` are now rejected upfront by review fix
    #03 should-fix #4 (see ``test_spawn_session_rejects_traversal_id``).
    This test uses ``abc/foo/delete-me`` (slashes, no traversal
    pattern) to pin the URL-quoting behaviour on its own.
    """
    import spawn_session  # type: ignore[import-not-found]

    seen_urls: list[str] = []

    def _capture(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        seen_urls.append(req.full_url)
        if req.full_url.endswith("/session"):
            return _mock_response(json.dumps({"id": "abc/foo/delete-me"}).encode())
        return _mock_response(b"")

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _capture)
    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
    )
    assert exit_code == 0
    # The second urlopen call should hit a URL-escaped session_id.
    prompt_calls = [u for u in seen_urls if "prompt_async" in u]
    assert len(prompt_calls) == 1
    assert "abc%2Ffoo%2Fdelete-me" in prompt_calls[0], prompt_calls
    # And the raw "/abc/foo/delete-me/" pattern must NOT appear in
    # the path — slashes are escaped to %2F.
    assert "/abc/foo/delete-me/prompt_async" not in prompt_calls[0]


# --------------------------------------------------------------------------- #
# Review fix plan #01 — should-fix #8: empty --directory rejected upfront
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_dir", ["", "   ", "\t"])
def test_empty_directory_rejected(monkeypatch: pytest.MonkeyPatch, bad_dir: str) -> None:
    """An empty / whitespace-only ``--directory`` exits 2 via argparse error.

    Without this guard, ``if directory:`` in ``_api`` evaluates ``""``
    as falsy and silently omits the ``x-opencode-directory`` header
    → session lands in the global workspace, not the worktree.
    """
    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", bad_dir,
    )
    assert exit_code == 2


# --------------------------------------------------------------------------- #
# Review fix plan #01 — should-fix #11: dedicated OSError regression test
# --------------------------------------------------------------------------- #


def test_oserror_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A bare ``OSError`` (disk-level / socket-level) routes through the fallback path.

    ``socket.timeout`` is an ``OSError`` subclass in 3.10+, so the
    timeout test exercises the except clause transitively — but AC #5
    enumerates ``OSError`` as a distinct failure class. This pins the
    enumeration.
    """
    import spawn_session  # type: ignore[import-not-found]

    monkeypatch.setattr(
        spawn_session.urllib.request, "urlopen",
        MagicMock(side_effect=OSError("disk-level")),
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
    assert "disk-level" in payload["detail"]


# --------------------------------------------------------------------------- #
# Review fix plan #01 — should-fix #12: timeout-on-prompt_async regression test
# --------------------------------------------------------------------------- #


def test_prompt_async_timeout_attempts_delete(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A ``socket.timeout`` raised by prompt_async still triggers DELETE cleanup.

    AC #7 wording is "on either ``POST /session`` or
    ``POST /session/<id>/prompt_async``". The earlier
    ``test_timeout_falls_back`` only exercises the first call. This
    pins the second-call timeout path.
    """
    import spawn_session  # type: ignore[import-not-found]

    seen_methods: list[tuple[str, str]] = []

    def _side(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        seen_methods.append((req.get_method(), req.full_url))
        url = req.full_url
        method = req.get_method()
        if method == "POST" and url.endswith("/session"):
            return _mock_response(json.dumps({"id": "sess-X"}).encode())
        if method == "POST" and "prompt_async" in url:
            raise socket.timeout("timed out")
        if method == "DELETE":
            return _mock_response(b"true")
        raise AssertionError(f"unexpected: {method} {url}")

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
    # DELETE was actually issued — orphan cleaned up.
    assert any(m == "DELETE" for m, _ in seen_methods)


# --------------------------------------------------------------------------- #
# Review fix plan #01 — nice-to-have #22: orphan branch without opencode CLI
# --------------------------------------------------------------------------- #


def test_orphan_without_opencode_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Orphan + no opencode CLI on PATH → drop ``new_context_cli``, surface the issue.

    When the server creates a session but then becomes unreachable for
    both prompt_async AND DELETE, AND the user has no opencode CLI to
    fall back on, the orphan-session payload would otherwise emit a
    ``new_context_cli`` that the user cannot execute. The fix probes
    ``shutil.which`` in the orphan branch too and drops the key when
    no CLI is available, appending a clearer recovery note to ``detail``.
    """
    import spawn_session  # type: ignore[import-not-found]

    def _side(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        url = req.full_url
        method = req.get_method()
        if method == "POST" and url.endswith("/session"):
            return _mock_response(json.dumps({"id": "sess-X"}).encode())
        if method == "POST" and "prompt_async" in url:
            raise _http_error(500, "prompt boom")
        if method == "DELETE":
            raise _http_error(404, "delete boom")
        raise AssertionError(f"unexpected: {method} {url}")

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _side)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
    )
    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "session_orphaned"
    assert payload["session_id"] == "sess-X"
    # ``new_context_cli`` is dropped because the CLI binary isn't on PATH.
    assert "new_context_cli" not in payload
    assert "opencode CLI" in payload["detail"]


# --------------------------------------------------------------------------- #
# Review fix plan #01 — nice-to-have #23: empty --prompt / --title rejected
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_prompt", ["", "   ", "\t"])
def test_empty_prompt_rejected(monkeypatch: pytest.MonkeyPatch, bad_prompt: str) -> None:
    """An empty / whitespace-only ``--prompt`` exits 2 via argparse error."""
    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
        "--prompt", bad_prompt,
    )
    assert exit_code == 2


@pytest.mark.parametrize("bad_title", ["", "   ", "\t"])
def test_empty_title_rejected(monkeypatch: pytest.MonkeyPatch, bad_title: str) -> None:
    """An explicitly empty ``--title`` exits 2; omitting the flag still works."""
    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
        "--title", bad_title,
    )
    assert exit_code == 2


# --------------------------------------------------------------------------- #
# Review fix plan #02 — must-fix #1: falsy / non-string `id` falls back too
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("session_body", [
    b'{"id": ""}',      # empty string — currently survives the None check
    b'{"id": 0}',       # zero — currently produces /session/0/prompt_async
    b'{"id": false}',   # bool — same
    b'{"id": ["x"]}',   # list — survives, str(list) is unhelpful in URL
    b'{"id": null}',    # the case the None-guard already caught
])
def test_falsy_or_non_string_id_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    session_body: bytes,
) -> None:
    """A non-string / falsy / whitespace ``id`` triggers the CLI fallback upfront.

    The tightened guard rejects the response BEFORE attempting
    ``POST /session/<id>/prompt_async`` so we don't issue nonsense
    URLs like ``/session/0/prompt_async`` or ``/session//prompt_async``.
    Only the initial ``POST /session`` urlopen call should be made;
    prompt_async must not be reached.
    """
    import spawn_session  # type: ignore[import-not-found]

    seen_urls: list[str] = []

    def _capture(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        seen_urls.append(req.full_url)
        if req.full_url.endswith("/session"):
            return _mock_response(session_body)
        raise AssertionError(f"unexpected call: {req.full_url}")

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _capture)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/opencode" if name == "opencode" else None)

    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fallback_cli"
    # Only one urlopen call — the bad id was rejected before
    # prompt_async, no nonsense URL was issued.
    assert len(seen_urls) == 1, f"expected 1 urlopen call, got: {seen_urls}"
    assert seen_urls[0].endswith("/session")


# --------------------------------------------------------------------------- #
# Review fix plan #02 — must-fix #2: empty --agent rejected
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_agent", ["", "   ", "\t"])
def test_empty_agent_rejected(monkeypatch: pytest.MonkeyPatch, bad_agent: str) -> None:
    """An empty / whitespace-only ``--agent`` exits 2 via argparse error.

    Parallel guard to ``--directory`` / ``--prompt`` / ``--title``
    (fix plan #01 should-fix #8 + nice-to-have #23). Without it, an
    empty agent reaches ``prompt_async`` as ``{"agent": ""}`` and the
    server returns an opaque error.
    """
    exit_code = _run_main(
        monkeypatch,
        "--agent", bad_agent,
        "--directory", "/tmp/x",
    )
    assert exit_code == 2


# --------------------------------------------------------------------------- #
# Review fix plan #02 — should-fix #8: surrounding whitespace stripped
# --------------------------------------------------------------------------- #


def test_directory_whitespace_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--directory '  /tmp/wt  '`` lands ``"/tmp/wt"`` (no surrounding ws) in the header.

    Shell-history copy-paste artifacts shouldn't break the
    worktree-to-server binding silently.
    """
    import spawn_session  # type: ignore[import-not-found]

    seen_headers: list[str | None] = []

    def _capture(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        seen_headers.append(req.get_header("X-opencode-directory"))
        if req.full_url.endswith("/session"):
            return _mock_response(json.dumps({"id": "sess-X"}).encode())
        return _mock_response(b"")

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _capture)
    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "  /tmp/wt  ",
    )
    assert exit_code == 0
    assert seen_headers
    assert seen_headers[0] == "/tmp/wt", f"header was not stripped: {seen_headers[0]!r}"


def test_agent_whitespace_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--agent '  qs-create-plan  '`` lands stripped in the prompt_async body."""
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
        "--agent", "  qs-create-plan  ",
        "--directory", "/tmp/x",
    )
    assert exit_code == 0
    prompt_body = json.loads(bodies[1] or b"{}")
    assert prompt_body["agent"] == "qs-create-plan"


def test_prompt_whitespace_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--prompt '  hi  '`` lands stripped in the prompt_async body."""
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
        "--prompt", "  hi  ",
    )
    assert exit_code == 0
    prompt_body = json.loads(bodies[1] or b"{}")
    assert prompt_body["parts"][0]["text"] == "hi"


def test_title_whitespace_stripped_when_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--title '  T  '`` (explicit) lands stripped in the /session body."""
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
        "--title", "  My Title  ",
    )
    assert exit_code == 0
    session_body = json.loads(bodies[0] or b"{}")
    assert session_body == {"title": "My Title"}


# --------------------------------------------------------------------------- #
# Review fix plan #03 — must-fix #1: embedded CR/LF in header raises ValueError
# --------------------------------------------------------------------------- #


def test_invalid_header_value_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A ``ValueError`` from ``urlopen`` (invalid header value) routes through fallback.

    ``http.client`` raises ``ValueError: Invalid header value`` when
    CR/LF or other illegal chars sneak into the header value (e.g.
    via a header-injection attempt). ``ValueError`` is not a subclass
    of ``HTTPError`` / ``URLError`` / ``OSError`` /
    ``HTTPException`` / ``JSONDecodeError``, so before this fix the
    exception would propagate out and crash the script with a
    traceback (review fix #03 must-fix #1).
    """
    import spawn_session  # type: ignore[import-not-found]

    monkeypatch.setattr(
        spawn_session.urllib.request, "urlopen",
        MagicMock(side_effect=ValueError("Invalid header value b'/tmp/wt\\nfoo'")),
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
    assert "Invalid header value" in payload["detail"]


@pytest.mark.parametrize("bad_value", [
    "/tmp/wt\n",
    "/tmp/wt\r",
    "/tmp/wt\x00",
    "/tmp\nwt",         # interior newline
    "/tmp\x01wt",       # interior control char
])
def test_control_chars_in_directory_rejected(
    monkeypatch: pytest.MonkeyPatch, bad_value: str,
) -> None:
    """Control characters in ``--directory`` exit 2 via parser.error.

    Layered defense alongside the widened ``_api`` except (review fix
    #03 must-fix #1). The upstream guard gives a clearer message;
    the downstream except is the safety net.
    """
    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", bad_value,
    )
    assert exit_code == 2


@pytest.mark.parametrize("bad_value", ["qs\n", "qs\r", "qs\x00", "qs\x01plan"])
def test_control_chars_in_agent_rejected(
    monkeypatch: pytest.MonkeyPatch, bad_value: str,
) -> None:
    """Control characters in ``--agent`` exit 2 via parser.error."""
    exit_code = _run_main(
        monkeypatch,
        "--agent", bad_value,
        "--directory", "/tmp/x",
    )
    assert exit_code == 2


@pytest.mark.parametrize("bad_value", ["hi\n", "hi\r", "hi\x00", "hi\x01there"])
def test_control_chars_in_prompt_rejected(
    monkeypatch: pytest.MonkeyPatch, bad_value: str,
) -> None:
    """Control characters in ``--prompt`` exit 2 via parser.error."""
    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
        "--prompt", bad_value,
    )
    assert exit_code == 2


@pytest.mark.parametrize("bad_value", ["T\n", "T\r", "T\x00", "T\x01end"])
def test_control_chars_in_title_rejected(
    monkeypatch: pytest.MonkeyPatch, bad_value: str,
) -> None:
    """Control characters in ``--title`` exit 2 via parser.error."""
    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
        "--title", bad_value,
    )
    assert exit_code == 2


# --------------------------------------------------------------------------- #
# Review fix plan #03 — should-fix #4: reject `..` traversal segments in id
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_id", [
    "abc/../delete-me",   # traversal pattern
    "..",                 # bare parent
    "../etc/passwd",      # rooted traversal
    "abc..def",           # interior `..` (rejected for uniformity)
])
def test_spawn_session_rejects_traversal_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bad_id: str,
) -> None:
    """A server-returned ``id`` containing ``..`` is rejected upfront.

    Even after ``urllib.parse.quote`` escaping, ``..`` literally
    passes through (it's in the RFC 3986 unreserved set). A
    URL-normalizing middleware (corporate proxy / reverse proxy /
    container ingress) could rewrite the path and forward a
    traversal-resolved URL to the OpenCode server. Two-layer
    defense: pre-call validation + URL escaping (review fix #03
    should-fix #4).
    """
    import spawn_session  # type: ignore[import-not-found]

    seen_urls: list[str] = []

    def _capture(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        seen_urls.append(req.full_url)
        if req.full_url.endswith("/session"):
            return _mock_response(json.dumps({"id": bad_id}).encode())
        raise AssertionError(f"unexpected call: {req.full_url}")

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _capture)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/opencode" if name == "opencode" else None)

    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fallback_cli"
    # Only the initial /session POST happened — the bad id was
    # rejected before any prompt_async URL with `..` could be issued.
    assert len(seen_urls) == 1, f"prompt_async should NOT have been called: {seen_urls}"


# --------------------------------------------------------------------------- #
# Review fix plan #03 — should-fix #5: orphan stderr mentions missing CLI
# --------------------------------------------------------------------------- #


def test_orphan_stderr_mentions_missing_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """When orphan + no opencode CLI, stderr line includes the CLI-missing hint."""
    import spawn_session  # type: ignore[import-not-found]

    def _side(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        url = req.full_url
        method = req.get_method()
        if method == "POST" and url.endswith("/session"):
            return _mock_response(json.dumps({"id": "sess-X"}).encode())
        if method == "POST" and "prompt_async" in url:
            raise _http_error(500, "prompt boom")
        if method == "DELETE":
            raise _http_error(404, "delete boom")
        raise AssertionError(f"unexpected: {method} {url}")

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _side)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    exit_code = _run_main(
        monkeypatch,
        "--agent", "qs-create-plan",
        "--directory", "/tmp/x",
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    # User-visible stderr line must surface the missing-CLI hint
    # (not just buried in the JSON detail).
    assert "opencode CLI" in err
    assert "sess-X" in err


# --------------------------------------------------------------------------- #
# Review fix plan #03 — nice-to-have #21: spawn_session() normalizes empty title
# --------------------------------------------------------------------------- #


def test_spawn_session_normalizes_empty_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``spawn_session(title="")`` (programmer-direct) → falls back to agent name."""
    import spawn_session  # type: ignore[import-not-found]

    bodies: list[bytes | None] = []

    def _capture(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        bodies.append(req.data)
        if req.full_url.endswith("/session"):
            return _mock_response(json.dumps({"id": "sess-X"}).encode())
        return _mock_response(b"")

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _capture)
    result = spawn_session.spawn_session(
        agent="qs-create-plan", directory="/tmp/x", title="",
    )
    # title falls back to the agent name when empty / whitespace.
    assert result["title"] == "qs-create-plan"
    session_body = json.loads(bodies[0] or b"{}")
    assert session_body == {"title": "qs-create-plan"}


@pytest.mark.parametrize("bad_title", ["   ", "\t", "\n"])
def test_spawn_session_normalizes_whitespace_title(
    monkeypatch: pytest.MonkeyPatch, bad_title: str,
) -> None:
    """Whitespace-only ``title`` from programmer-direct call → also falls back."""
    import spawn_session  # type: ignore[import-not-found]

    bodies: list[bytes | None] = []

    def _capture(req: urllib.request.Request, timeout: int = 10) -> MagicMock:
        del timeout
        bodies.append(req.data)
        if req.full_url.endswith("/session"):
            return _mock_response(json.dumps({"id": "sess-X"}).encode())
        return _mock_response(b"")

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", _capture)
    result = spawn_session.spawn_session(
        agent="qs-create-plan", directory="/tmp/x", title=bad_title,
    )
    assert result["title"] == "qs-create-plan"
