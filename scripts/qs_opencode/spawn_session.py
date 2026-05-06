#!/usr/bin/env python3
"""Spawn a new interactive OpenCode session via the HTTP API.

Instead of Task-spawning a non-interactive subagent, this script creates a
new session that appears in the OpenCode sidebar — fully interactive.

It uses the local OpenCode server API (default port 4096) to:
1. Reload the instance so newly-rendered agents are discovered.
2. Create a new session.
3. Send an async prompt with the agent and kickoff message.

All HTTP calls use ``urllib`` from the standard library.  The reload
endpoint (``POST /instance/reload``) is fire-and-forget on the server —
it schedules the reload and responds immediately with 200, avoiding the
deadlock that previously occurred when the reload waited for session
disposal while the session was blocked on the HTTP response.

Usage::

    python scripts/qs_opencode/spawn_session.py \\
        --agent qs-implement-task-QS-42 \\
        --prompt "Begin your phase protocol." \\
        --title "QS-42: implement-task" \\
        --directory /path/to/worktree

The ``--directory`` flag is **required** — it tells the OpenCode server
which project/sandbox the session belongs to.  Without it the middleware
falls back to ``process.cwd()`` which typically resolves to the global
project, making the session invisible in the sandbox sidebar.

The script outputs JSON with the created session ID so the calling agent
can report success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

from utils import output_json  # type: ignore[import-not-found]

# Default OpenCode server URL.  The port can be overridden via
# OPENCODE_SERVER_PORT (the server always listens on localhost).
DEFAULT_PORT = 4096


def _base_url() -> str:
    port = os.environ.get("OPENCODE_SERVER_PORT", str(DEFAULT_PORT))
    return f"http://127.0.0.1:{port}"


def _api(
    method: str,
    path: str,
    body: dict | None = None,
    *,
    directory: str | None = None,
    timeout: int = 10,
) -> dict | list | None:
    """Call the OpenCode HTTP API and return parsed JSON (or ``None``).

    Raises ``SystemExit`` on connection errors so callers get a clear
    failure instead of a silent ``None``.
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
        print(f"ERROR: {method} {path} failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _api_safe(
    method: str,
    path: str,
    body: dict | None = None,
    *,
    directory: str | None = None,
    timeout: int = 10,
) -> tuple[bool, dict | list | None]:
    """Like ``_api`` but returns ``(ok, result)`` instead of raising."""
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
                return True, None
            return True, json.loads(raw)
    except urllib.error.HTTPError, urllib.error.URLError, OSError:
        return False, None
    except json.JSONDecodeError, TypeError:
        return True, None


def _reload_instance(*, directory: str | None = None) -> bool:
    """Call ``POST /instance/reload`` so the server re-scans agent files.

    The server now handles reload as fire-and-forget — it schedules the
    reload in the background and responds with 200 immediately.  This
    breaks the deadlock that previously occurred when the reload tried to
    dispose instances (including cancelling active session runners) while
    the session was blocked waiting for this very HTTP response.

    Best-effort: logs on failure but never aborts.
    """
    if not directory:
        print(
            "WARNING: reload called without directory — server may reload wrong context",
            file=sys.stderr,
        )
    url = f"{_base_url()}/instance/reload"
    headers: dict[str, str] = {}
    if directory:
        headers["x-opencode-directory"] = directory
    req = urllib.request.Request(url, data=b"", method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        print(f"WARNING: /instance/reload failed ({exc})", file=sys.stderr)
        return False


def _wait_for_agent(
    agent: str,
    *,
    directory: str | None = None,
    timeout: float = 15,
    poll_interval: float = 0.5,
) -> bool:
    """Poll ``GET /agent`` until *agent* appears in the loaded agents list.

    Returns ``True`` if the agent was found within *timeout* seconds,
    ``False`` otherwise.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ok, agents = _api_safe("GET", "/agent", directory=directory, timeout=3)
        if ok and isinstance(agents, list):
            names = {a.get("name", "") for a in agents if isinstance(a, dict)}
            if agent in names:
                return True
        time.sleep(poll_interval)
    return False


def spawn_session(
    *,
    agent: str,
    prompt: str,
    title: str | None = None,
    directory: str | None = None,
) -> dict:
    """Create a new interactive session and send the kickoff prompt.

    *directory* must point to the worktree / sandbox so that the session
    is created under the correct project and appears in the sidebar.
    """
    # 0. Reload so the server discovers newly-rendered agent files
    _reload_instance(directory=directory)

    # 1. Wait for the agent to become available (poll GET /agent)
    if not _wait_for_agent(agent, directory=directory):
        print(
            f"WARNING: agent {agent!r} not found after reload; session may fail to activate it.",
            file=sys.stderr,
        )

    # 2. Create a new session
    session = _api(
        "POST",
        "/session",
        {"title": title or agent},
        directory=directory,
    )
    if not isinstance(session, dict):
        print(f"ERROR: Unexpected session response: {session}", file=sys.stderr)
        sys.exit(1)

    session_id = session.get("id") or session.get("ID")
    if not session_id:
        print(f"ERROR: No session ID in response: {session}", file=sys.stderr)
        sys.exit(1)

    # 3. Send the prompt asynchronously (returns empty, session starts working)
    try:
        _api(
            "POST",
            f"/session/{session_id}/prompt_async",
            {
                "agent": agent,
                "parts": [{"type": "text", "text": prompt}],
            },
            directory=directory,
        )
    except SystemExit:
        print(
            f"WARNING: prompt_async failed for session {session_id}",
            file=sys.stderr,
        )

    return {
        "session_id": session_id,
        "agent": agent,
        "title": title or agent,
        "directory": directory,
        "status": "spawned",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spawn a new interactive OpenCode session via HTTP API",
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Agent name to activate (e.g. qs-implement-task-QS-42)",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Kickoff prompt to send to the new session",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Session title (defaults to agent name)",
    )
    parser.add_argument(
        "--directory",
        default=None,
        help=(
            "Worktree / sandbox directory for the session.  Sent via "
            "x-opencode-directory header so the session is created under "
            "the correct project.  Required for sessions to appear in "
            "the sandbox sidebar."
        ),
    )
    args = parser.parse_args()

    result = spawn_session(
        agent=args.agent,
        prompt=args.prompt,
        title=args.title,
        directory=args.directory,
    )
    output_json(result)


if __name__ == "__main__":
    main()
