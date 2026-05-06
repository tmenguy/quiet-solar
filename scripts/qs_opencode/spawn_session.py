#!/usr/bin/env python3
"""Spawn a new interactive OpenCode session via the HTTP API.

Instead of Task-spawning a non-interactive subagent, this script creates a
new **blank** session (no agent, no prompt) and then fires a delayed
``/instance/reload`` so the server discovers newly-rendered agent files.

The calling agent should tell the user to:
1. Refresh the browser (to see the reload take effect).
2. Switch to the newly created session (by title).
3. Select the target agent from the agent picker.
4. Type the kickoff prompt.

**Why no prompt/agent activation?**  ``POST /instance/reload`` disposes
all ``SessionRunState`` caches, which cancels every busy session runner.
If reload fires while our bash tool is still running, the calling session
gets killed.  The only safe pattern is: fire reload as the *very last*
side-effect, then yield the final assistant message.  But that means we
can't wait for the reload to complete, can't poll for agents, and can't
activate an agent on the new session — all of those require the reload
to have already finished.

Usage::

    python scripts/qs_opencode/spawn_session.py \\
        --agent qs-review-task-QS-42 \\
        --title "QS-42: review-task" \\
        --directory /path/to/worktree

The ``--directory`` flag is **required** — it tells the OpenCode server
which project/sandbox the session belongs to.  Without it the server
returns 400 on reload and the session may be invisible in the sidebar.

The script outputs JSON with the created session details and instructions
for the user.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
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


def request_reload_async(
    *,
    directory: str | None = None,
    delay_seconds: float = 2.0,
) -> None:
    """Fire ``POST /instance/reload`` AFTER our own session has gone idle.

    Spawns a detached ``sleep N; curl …`` process and returns immediately.
    The HTTP request reaches the server only after *delay_seconds*, giving
    the calling session time to finish and go idle — avoiding the deadlock
    caused by ``Instance.reload → cancel runner → await done``.

    Why a delay: opencode's ``Instance.reload`` disposes ``SessionRunState``,
    whose finalizer cancels every BUSY session runner on this instance.  If
    the HTTP request reaches the server while our calling session is still
    busy (e.g. still rendering its final assistant message), our own runner
    gets cancelled mid-task.  By spawning a detached ``sleep N; curl …`` and
    returning immediately, we let our session finish, transition to idle,
    and be removed from the runners map.  Then the reload arrives, finds no
    busy runners to cancel, and proceeds cleanly.
    """
    if not directory:
        directory = os.getcwd()

    base = _base_url()
    cmd = (
        f"sleep {delay_seconds}; "
        f"curl -s -X POST {shlex.quote(base)}/instance/reload "
        f"-H {shlex.quote(f'x-opencode-directory: {directory}')} "
        f"> /dev/null 2>&1"
    )
    try:
        subprocess.Popen(  # noqa: S603
            ["sh", "-c", cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        print(
            f"WARNING: failed to spawn /instance/reload trigger ({exc}); "
            f"agents will not refresh until next manual /reload.",
            file=sys.stderr,
        )


def spawn_session(
    *,
    agent: str,
    title: str | None = None,
    directory: str | None = None,
    delay_seconds: float = 2.0,
) -> dict:
    """Create a blank session and schedule a delayed reload.

    The session is created immediately (no agent activation, no prompt).
    Then ``request_reload_async`` fires a detached ``sleep + curl`` so the
    server re-scans agent files after the calling session goes idle.

    The caller should instruct the user to:
    1. Refresh the browser after ~delay_seconds.
    2. Switch to the new session (by *title*).
    3. Select *agent* from the agent picker.
    4. Type the kickoff prompt.

    *directory* must point to the worktree / sandbox so that the session
    is created under the correct project and appears in the sidebar.
    """
    # 1. Create a blank session (no agent, no prompt)
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

    # 2. Fire reload LAST — detached, delayed, never blocks.
    request_reload_async(directory=directory, delay_seconds=delay_seconds)

    return {
        "session_id": session_id,
        "agent": agent,
        "title": title or agent,
        "directory": directory,
        "delay_seconds": delay_seconds,
        "status": "session_created",
        "instructions": (
            f"Session '{title or agent}' created. "
            f"Instance reload scheduled (fires in {delay_seconds}s). "
            f"Refresh browser, switch to the session, "
            f"select agent '{agent}' from the picker, then type the kickoff prompt."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spawn a new interactive OpenCode session via HTTP API",
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Agent name to activate (e.g. qs-review-task-QS-42)",
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
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help=(
            "Seconds to wait before the reload fires.  Must be long enough "
            "for the calling session to finish and go idle.  Default: 2.0"
        ),
    )
    args = parser.parse_args()

    result = spawn_session(
        agent=args.agent,
        title=args.title,
        directory=args.directory,
        delay_seconds=args.delay,
    )
    output_json(result)


if __name__ == "__main__":
    main()
