#!/usr/bin/env python3
"""Spawn a new interactive OpenCode session via the HTTP API.

Instead of Task-spawning a non-interactive subagent, this script creates a
new session that appears in the OpenCode sidebar — fully interactive.

It uses the local OpenCode server API (default port 4096) to:
1. Create a new session.
2. Send an async prompt with the agent and kickoff message.

Usage::

    python scripts/qs_opencode/spawn_session.py \
        --agent qs-implement-task-QS-42 \
        --prompt "Begin your phase protocol." \
        --title "QS-42: implement-task"

The script outputs JSON with the created session ID so the calling agent
can report success.
"""

from __future__ import annotations

import argparse
import json
import os
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


def _api(method: str, path: str, body: dict | None = None) -> dict:
    """Make a JSON request to the OpenCode server API."""
    url = f"{_base_url()}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw)  # type: ignore[no-any-return]
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(errors="replace")
        print(
            f"ERROR: OpenCode API {method} {path} returned {exc.code}: "
            f"{error_body}",
            file=sys.stderr,
        )
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(
            f"ERROR: Cannot reach OpenCode server at {url}: {exc.reason}\n"
            f"Is OpenCode running?  Check port {_base_url()}",
            file=sys.stderr,
        )
        sys.exit(1)


def spawn_session(
    *,
    agent: str,
    prompt: str,
    title: str | None = None,
) -> dict:
    """Create a new interactive session and send the kickoff prompt."""
    # 1. Create a new session
    session = _api("POST", "/session", {"title": title or agent})
    session_id = session.get("id") or session.get("ID")
    if not session_id:
        print(f"ERROR: Unexpected session response: {session}", file=sys.stderr)
        sys.exit(1)

    # 2. Send the prompt asynchronously (returns 204, session starts working)
    _api("POST", f"/session/{session_id}/prompt_async", {
        "agent": agent,
        "parts": [{"type": "text", "text": prompt}],
    })

    return {
        "session_id": session_id,
        "agent": agent,
        "title": title or agent,
        "status": "spawned",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spawn a new interactive OpenCode session via HTTP API",
    )
    parser.add_argument(
        "--agent", required=True,
        help="Agent name to activate (e.g. qs-implement-task-QS-42)",
    )
    parser.add_argument(
        "--prompt", required=True,
        help="Kickoff prompt to send to the new session",
    )
    parser.add_argument(
        "--title", default=None,
        help="Session title (defaults to agent name)",
    )
    args = parser.parse_args()

    result = spawn_session(
        agent=args.agent,
        prompt=args.prompt,
        title=args.title,
    )
    output_json(result)


if __name__ == "__main__":
    main()
