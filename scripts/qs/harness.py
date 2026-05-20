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

# Legacy-alias map. ``detect`` (QS_HARNESS env var) and ``canonicalize``
# (--harness argparse flag, review fix #01 N8) share this table so the
# two entry points agree on what the user-typed name resolves to.
_LEGACY_ALIASES: dict[str, str] = {
    "claude": "claude-code",
    "claude_code": "claude-code",
    "claudecode": "claude-code",
}


def canonicalize(name: str) -> str:
    """Return the canonical harness name, mapping legacy aliases through.

    Idempotent: a canonical name passes through unchanged. An unknown
    name passes through unchanged too — argparse's ``choices=`` is
    the upstream guard that rejects garbage; ``canonicalize`` only
    collapses the documented aliases (review fix #01 N8).

    Whitespace and case are normalised before mapping so a user-typed
    ``"  Claude  "`` resolves the same as ``"claude-code"``.
    """
    normalised = name.lower().strip()
    if normalised in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[normalised]
    return normalised if normalised in VALID_HARNESSES else name


def harness_choices() -> list[str]:
    """Return the full list of acceptable ``--harness`` argparse choices.

    Includes the canonical names AND every legacy alias from
    ``_LEGACY_ALIASES`` so a user typing ``--harness claude`` passes
    argparse's ``choices=`` guard (review fix #01 N8). ``canonicalize``
    is then applied post-argparse to collapse aliases to canonical
    names before dispatch.
    """
    return [*VALID_HARNESSES, *_LEGACY_ALIASES.keys()]


def detect() -> Harness:
    """Return the detected harness, defaulting to ``claude-code``."""
    explicit = os.environ.get("QS_HARNESS", "").lower().strip()
    if explicit in VALID_HARNESSES:
        return explicit  # type: ignore[return-value]
    # Legacy alias kept for ergonomics — shares the same table as
    # ``canonicalize`` (review fix #01 N8).
    if explicit in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[explicit]  # type: ignore[return-value]

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
