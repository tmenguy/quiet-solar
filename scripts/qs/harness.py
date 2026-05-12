#!/usr/bin/env python3
"""Detect which harness is driving the current session.

Harnesses supported:

- ``claude-code`` — Claude Code CLI or Desktop
- ``cursor`` — Cursor IDE (2.4+ subagent support)
- ``opencode`` — OpenCode CLI / sandbox
- ``codex`` — OpenAI Codex (stub)

Resolution order:

1. ``QS_HARNESS`` env var (explicit override; the most reliable signal).
2. ``CLAUDECODE=1`` set by the Claude Code launcher.
3. ``OPENCODE_SERVER_PORT`` set by OpenCode sandbox.
4. ``CURSOR_TRACE_ID`` set by Cursor terminals.
5. Any ``CODEX_AGENT_*`` env var set by Codex.
6. Default: ``claude-code``.

Run directly to print the detected harness:

    python scripts/qs/harness.py            # prints just the harness
    python scripts/qs/harness.py --json     # prints {"harness": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Literal

Harness = Literal["claude-code", "cursor", "opencode", "codex"]

VALID_HARNESSES: tuple[Harness, ...] = ("claude-code", "cursor", "opencode", "codex")


def detect() -> Harness:
    """Return the detected harness, defaulting to ``claude-code``."""
    explicit = os.environ.get("QS_HARNESS", "").lower().strip()
    if explicit in VALID_HARNESSES:
        return explicit  # type: ignore[return-value]
    # Legacy alias kept for ergonomics.
    if explicit in ("claude", "claude_code", "claudecode"):
        return "claude-code"

    if os.environ.get("CLAUDECODE"):
        return "claude-code"
    if os.environ.get("OPENCODE_SERVER_PORT"):
        return "opencode"
    if os.environ.get("CURSOR_TRACE_ID"):
        return "cursor"
    if any(k.startswith("CODEX_AGENT_") for k in os.environ):
        return "codex"

    return "claude-code"


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect the current AI harness.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    args = parser.parse_args()

    harness = detect()
    if args.json:
        print(json.dumps({"harness": harness}))
    else:
        print(harness)
    sys.exit(0)


if __name__ == "__main__":
    main()
