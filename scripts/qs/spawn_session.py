#!/usr/bin/env python3
"""Spawn a new interactive OpenCode session via the HTTP API.

The new pipeline activates the next phase's agent and sends a kickoff
prompt in a single synchronous flow by calling::

    POST /session                         # creates a new session
    POST /session/<id>/prompt_async       # binds agent + sends prompt

No ``/instance/reload`` is needed: static ``.opencode/agents/`` files
are discovered at server startup, so we never have to make the server
re-scan agent files mid-flight. Skipping reload eliminates the
``SessionRunState`` cancellation race that blocked the legacy pipeline
from activating the agent in the same request.

CLI fallback semantics (AC #5 / #6 / #7 of QS-177)
--------------------------------------------------

When the OpenCode HTTP API is unreachable — connection refused, DNS
failure, timeout, ``HTTPError``, ``OSError``, ``JSONDecodeError``, or
missing ``id`` in the response — the script probes
``shutil.which('opencode')`` and emits a CLI-form fallback payload so
the user still lands in the next phase. If ``opencode`` is also
missing from PATH, the script exits 2 with ``fallback_unavailable``.

Orphan-session handling: if ``POST /session`` succeeds but the
subsequent ``POST /session/<id>/prompt_async`` fails, the script
attempts a best-effort ``DELETE /session/<id>``. On DELETE success
the user gets the standard ``fallback_cli`` payload (orphan cleaned
up). On DELETE failure the orphan session id is surfaced via
``status: "session_orphaned"`` (exit 2) so the user can clean it up
manually in the OpenCode UI.

Known limitation (per AC #12 of QS-177): the kickoff API succeeds
even when ``.opencode/agents/qs-<phase>.md`` does not yet exist, but
the session lands on the default OpenCode agent instead of the
intended phase orchestrator. Mirror ``.claude/agents/*.md`` into
``.opencode/agents/`` (with frontmatter conversion: ``tools:`` →
``permission:``, ``mode: primary`` / ``mode: subagent``) to enable
agent activation. This is a documented follow-up, not a silent bug.

Usage::

    python scripts/qs/spawn_session.py \\
        --agent qs-create-plan \\
        --directory /path/to/worktree \\
        [--title "QS_42: Foo"] \\
        [--prompt "Begin your phase protocol."]

The ``--directory`` flag is **required** — it lands in the
``x-opencode-directory`` header so the session is created under the
worktree's OpenCode workspace (otherwise the session is invisible in
the sidebar). The ``--title`` defaults to the agent name; the
``--prompt`` defaults to ``"Begin your phase protocol."``.

There is **no** ``--delay`` flag and **no** ``subprocess`` use —
agent activation happens in-band on ``POST /session/<id>/prompt_async``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import urllib.error
import urllib.request

from utils import output_json  # type: ignore[import-not-found]

# Default OpenCode server port. The CLI listens on localhost; the port
# can be overridden via the ``OPENCODE_SERVER_PORT`` env var.
DEFAULT_PORT = 4096

# Default kickoff prompt — matches the convention in the Claude /
# Cursor launchers (``next_prompt or "Begin your phase protocol."``).
DEFAULT_KICKOFF = "Begin your phase protocol."

# Default HTTP timeout for every OpenCode API call (seconds). Matches
# the legacy ``scripts/qs_opencode/spawn_session.py:_api`` timeout.
DEFAULT_TIMEOUT = 10


class SpawnSessionError(RuntimeError):
    """Raised by ``_api`` / ``spawn_session`` on any HTTP/JSON failure.

    Carries the attempted URL and the underlying cause so callers
    (notably ``main()``) can produce a structured fallback payload
    without re-parsing the exception message.

    Attributes:
        url: The URL that failed (or the synthetic URL of the
            response-shape check that failed).
        cause: The underlying exception, if any.
        orphan_session_id: When set, signals that ``POST /session``
            succeeded but a subsequent call failed AND best-effort
            ``DELETE /session/<id>`` also failed — the user needs to
            clean the orphan up manually.
    """

    def __init__(
        self,
        message: str,
        *,
        url: str = "",
        cause: BaseException | None = None,
        orphan_session_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.cause = cause
        self.orphan_session_id = orphan_session_id


def _base_url() -> str:
    """Return ``http://127.0.0.1:<port>`` for the OpenCode server."""
    port = os.environ.get("OPENCODE_SERVER_PORT", str(DEFAULT_PORT))
    return f"http://127.0.0.1:{port}"


def _api(
    method: str,
    path: str,
    body: dict | None = None,
    *,
    directory: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict | list | None:
    """Call the OpenCode HTTP API and return parsed JSON (or ``None``).

    Raises ``SpawnSessionError`` on any failure class (``URLError``,
    ``HTTPError``, ``OSError``, ``TimeoutError``,
    ``JSONDecodeError``). The original exception is attached via
    ``__cause__`` so callers can inspect it if needed.

    Args:
        method: HTTP method (``"POST"``, ``"DELETE"``).
        path: URL path under ``_base_url()`` (e.g. ``"/session"``).
        body: Optional JSON body. ``None`` sends no body.
        directory: Worktree directory; sent as the
            ``x-opencode-directory`` header so the session is bound to
            the correct OpenCode workspace.
        timeout: HTTP timeout in seconds. Defaults to
            ``DEFAULT_TIMEOUT`` (matching the legacy pipeline).

    Returns:
        Parsed JSON (``dict`` or ``list``), or ``None`` when the
        response body is empty (e.g. a 204 from ``prompt_async``).
    """
    url = f"{_base_url()}{path}"
    headers: dict[str, str] = {}
    if directory:
        headers["x-opencode-directory"] = directory

    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return None
            return json.loads(raw)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        # ``OSError`` covers ``socket.timeout`` / ``TimeoutError`` in
        # modern Python; ``URLError`` covers DNS failure, connection
        # refused, and other transport errors; ``HTTPError`` covers
        # non-2xx HTTP status codes.
        raise SpawnSessionError(
            f"{method} {path} failed: {exc}", url=url, cause=exc,
        ) from exc
    except json.JSONDecodeError as exc:
        raise SpawnSessionError(
            f"{method} {path} returned malformed JSON: {exc}",
            url=url,
            cause=exc,
        ) from exc


def spawn_session(
    *,
    agent: str,
    directory: str,
    title: str | None = None,
    prompt: str = DEFAULT_KICKOFF,
) -> dict:
    """Create a new OpenCode session and activate ``agent`` with ``prompt``.

    Args:
        agent: Agent name to bind (e.g. ``"qs-create-plan"``).
        directory: Worktree directory; required by the OpenCode server
            via the ``x-opencode-directory`` header.
        title: Optional session title; defaults to ``agent`` when
            ``None``.
        prompt: Kickoff prompt sent inside ``parts[0].text``; defaults
            to ``DEFAULT_KICKOFF``.

    Returns:
        Dict with keys ``session_id``, ``agent``, ``title``,
        ``directory``, ``prompt``, ``status: "session_created"`` on
        success.

    Raises:
        SpawnSessionError: on any HTTP/JSON failure. When
            ``POST /session`` succeeded but a subsequent call failed
            AND ``DELETE /session/<id>`` also failed, the exception's
            ``.orphan_session_id`` carries the leaked session id so
            ``main()`` can surface it to the user.
    """
    actual_title = title if title is not None else agent

    # 1. Create session
    session = _api(
        "POST", "/session", {"title": actual_title}, directory=directory,
    )
    if not isinstance(session, dict):
        raise SpawnSessionError(
            f"Unexpected /session response (not a dict): {session!r}",
            url=f"{_base_url()}/session",
        )
    session_id = session.get("id") or session.get("ID")
    if not session_id:
        raise SpawnSessionError(
            f"No id in /session response: {session!r}",
            url=f"{_base_url()}/session",
        )

    # 2. Send prompt + activate agent (in-band, single call)
    try:
        _api(
            "POST",
            f"/session/{session_id}/prompt_async",
            {"agent": agent, "parts": [{"type": "text", "text": prompt}]},
            directory=directory,
        )
    except SpawnSessionError as prompt_exc:
        # Best-effort cleanup so we don't leak an orphan session in
        # the OpenCode UI. If DELETE also fails, surface the orphan
        # id via ``orphan_session_id`` so main() can emit the
        # ``session_orphaned`` payload (AC #6).
        delete_path = f"/session/{session_id}"
        try:
            _api("DELETE", delete_path, directory=directory)
        except SpawnSessionError as delete_exc:
            combined = (
                f"prompt_async failed: {prompt_exc}; "
                f"DELETE {delete_path} also failed: {delete_exc}"
            )
            raise SpawnSessionError(
                combined,
                url=f"{_base_url()}{delete_path}",
                cause=prompt_exc,
                orphan_session_id=str(session_id),
            ) from prompt_exc
        # DELETE succeeded — re-raise the original prompt_async error
        # so main() falls back to the CLI form (the user still needs
        # to land in the next phase).
        raise

    return {
        "session_id": str(session_id),
        "agent": agent,
        "title": actual_title,
        "directory": directory,
        "prompt": prompt,
        "status": "session_created",
    }


def _build_cli_command(agent: str, directory: str, prompt: str) -> str:
    """Return the bare ``opencode <dir> --agent <a> --prompt <p>`` fallback command.

    Every interpolated value is shell-escaped via ``shlex.quote``
    (parallel to ``scripts/qs_opencode/utils.py:opencode_launch_command``).
    """
    return (
        f"opencode {shlex.quote(directory)} "
        f"--agent {shlex.quote(agent)} "
        f"--prompt {shlex.quote(prompt)}"
    )


def _emit_fallback(
    *,
    agent: str,
    directory: str,
    prompt: str,
    exc: SpawnSessionError,
) -> int:
    """Print the appropriate fallback payload + return the desired exit code.

    Decision tree:

    - Orphan (``exc.orphan_session_id`` set) → ``session_orphaned``,
      exit 2.
    - ``opencode`` on PATH → ``fallback_cli``, exit 0.
    - ``opencode`` missing → ``fallback_unavailable``, exit 2.
    """
    detail = str(exc)
    new_context_cli = _build_cli_command(agent, directory, prompt)

    if exc.orphan_session_id is not None:
        print(
            f"ERROR: OpenCode session {exc.orphan_session_id} was created "
            f"but agent activation failed and cleanup also failed; "
            f"clean up the orphan session manually in the OpenCode UI.",
            file=sys.stderr,
        )
        output_json({
            "status": "session_orphaned",
            "session_id": exc.orphan_session_id,
            "agent": agent,
            "directory": directory,
            "new_context_cli": new_context_cli,
            "detail": detail,
        })
        return 2

    if shutil.which("opencode"):
        print(
            f"WARNING: OpenCode HTTP API unreachable ({detail}); "
            f"falling back to opencode CLI.",
            file=sys.stderr,
        )
        output_json({
            "status": "fallback_cli",
            "agent": agent,
            "directory": directory,
            "new_context_cli": new_context_cli,
            "detail": detail,
        })
        return 0

    print(
        f"ERROR: OpenCode HTTP API unreachable AND opencode CLI not on "
        f"PATH ({detail}); cannot launch the next phase.",
        file=sys.stderr,
    )
    output_json({
        "status": "fallback_unavailable",
        "agent": agent,
        "directory": directory,
        "detail": (
            f"{detail}; opencode CLI not on PATH — install or activate "
            f"the server manually"
        ),
    })
    return 2


def main() -> None:
    """CLI entry point — see module docstring for usage."""
    parser = argparse.ArgumentParser(
        description=(
            "Spawn a new interactive OpenCode session via the HTTP API "
            "and activate `--agent` with `--prompt` in a single call."
        ),
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Agent name to bind to the session (e.g. `qs-create-plan`).",
    )
    parser.add_argument(
        "--directory",
        required=True,
        help=(
            "Worktree directory; lands in the `x-opencode-directory` "
            "header so the session is created under the correct "
            "OpenCode workspace."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Session title (defaults to the `--agent` value).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_KICKOFF,
        help=(
            "Kickoff prompt sent inside `parts[0].text` "
            f"(defaults to {DEFAULT_KICKOFF!r})."
        ),
    )
    args = parser.parse_args()

    try:
        result = spawn_session(
            agent=args.agent,
            directory=args.directory,
            title=args.title,
            prompt=args.prompt,
        )
    except SpawnSessionError as exc:
        sys.exit(_emit_fallback(
            agent=args.agent,
            directory=args.directory,
            prompt=args.prompt,
            exc=exc,
        ))

    output_json(result)
    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
