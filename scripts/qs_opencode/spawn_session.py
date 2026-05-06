#!/usr/bin/env python3
"""Spawn a new interactive OpenCode session via the HTTP API.

Instead of Task-spawning a non-interactive subagent, this script creates a
new session that appears in the OpenCode sidebar — fully interactive.

It uses the local OpenCode server API (default port 4096) to:
1. Reload the instance so newly-rendered agents are discovered.
2. Create a new session.
3. Send an async prompt with the agent and kickoff message.

All HTTP calls use ``curl`` via ``subprocess.run`` because Python's
``urllib`` deadlocks when called from within an OpenCode agent's Bash
tool execution (the server thread servicing the tool call can't
concurrently process the urllib request from the same process).

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
import subprocess
import sys
import time

from utils import output_json  # type: ignore[import-not-found]

# Default OpenCode server URL.  The port can be overridden via
# OPENCODE_SERVER_PORT (the server always listens on localhost).
DEFAULT_PORT = 4096


def _base_url() -> str:
    port = os.environ.get("OPENCODE_SERVER_PORT", str(DEFAULT_PORT))
    return f"http://127.0.0.1:{port}"


def _curl(
    method: str,
    path: str,
    body: dict | None = None,
    *,
    directory: str | None = None,
    timeout: int = 10,
) -> tuple[bool, str]:
    """Make an HTTP request via curl. Returns (success, response_body)."""
    url = f"{_base_url()}{path}"
    cmd = ["curl", "-s", "--max-time", str(timeout), "-X", method]
    if body is not None:
        cmd.extend(["-H", "Content-Type: application/json", "-d", json.dumps(body)])
    if directory:
        cmd.extend(["-H", f"x-opencode-directory: {directory}"])
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    return result.returncode == 0, result.stdout


def _reload_instance(*, directory: str | None = None) -> bool:
    """Call ``POST /instance/reload`` so the server re-scans agent files.

    The reload endpoint deadlocks when called synchronously from within
    an agent's Bash tool (the server thread servicing the tool can't
    concurrently process reload).  We launch curl fully detached
    (``start_new_session=True``) so it runs independently — the parent
    process and the Bash tool do NOT wait for it.

    Best-effort: logs on failure but never aborts.
    """
    url = f"{_base_url()}/instance/reload"
    curl_cmd = ["curl", "-s", "--max-time", "60", "-X", "POST", url]
    if directory:
        curl_cmd.extend(["-H", f"x-opencode-directory: {directory}"])
    try:
        subprocess.Popen(  # noqa: S603
            curl_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except OSError as exc:
        print(
            f"WARNING: could not launch reload ({exc})",
            file=sys.stderr,
        )
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
    ``False`` otherwise. Falls back to a 2s sleep on any error.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ok, body = _curl("GET", "/agent", directory=directory, timeout=3)
        if ok and body:
            try:
                agents = json.loads(body)
                names = {a.get("name", "") for a in agents}
                if agent in names:
                    return True
            except json.JSONDecodeError, TypeError:
                pass
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
    ok, body = _curl(
        "POST",
        "/session",
        {"title": title or agent},
        directory=directory,
    )
    if not ok or not body:
        print("ERROR: Failed to create session (curl failed)", file=sys.stderr)
        sys.exit(1)

    try:
        session = json.loads(body)
    except json.JSONDecodeError:
        print(f"ERROR: Unexpected session response: {body}", file=sys.stderr)
        sys.exit(1)

    session_id = session.get("id") or session.get("ID")
    if not session_id:
        print(f"ERROR: No session ID in response: {session}", file=sys.stderr)
        sys.exit(1)

    # 3. Send the prompt asynchronously (returns empty, session starts working)
    ok, _ = _curl(
        "POST",
        f"/session/{session_id}/prompt_async",
        {
            "agent": agent,
            "parts": [{"type": "text", "text": prompt}],
        },
        directory=directory,
    )
    if not ok:
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
