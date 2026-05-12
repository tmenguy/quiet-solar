#!/usr/bin/env python3
"""Emit a harness-aware handoff payload for the next phase.

Phases that cross workspaces (e.g. ``setup-task`` on main → ``create-plan``
in the worktree) end by calling this script to build a launcher payload
matching the detected harness.

Usage::

    python scripts/qs/next_step.py \\
        --next-cmd "/create-plan" \\
        --work-dir /path/to/worktree \\
        --issue 42 \\
        --title "Story 3.2: foo bar" \\
        [--next-prompt "Begin your phase protocol."] \\
        [--harness HARNESS_OVERRIDE]
"""

from __future__ import annotations

import argparse
import sys

from harness import VALID_HARNESSES  # type: ignore[import-not-found]
from harness import detect as detect_harness
from launchers import claude as claude_launcher  # type: ignore[import-not-found]
from launchers import codex as codex_launcher  # type: ignore[import-not-found]
from launchers import cursor as cursor_launcher  # type: ignore[import-not-found]
from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

from utils import output_json  # type: ignore[import-not-found]

_LAUNCHERS = {
    "claude-code": claude_launcher,
    "cursor": cursor_launcher,
    "opencode": opencode_launcher,
    "codex": codex_launcher,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a harness-aware handoff payload.")
    parser.add_argument("--next-cmd", required=True, help="Slash command for the next phase (e.g. /create-plan)")
    parser.add_argument("--work-dir", required=True, help="Worktree the new session should open in")
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--next-prompt",
        default=None,
        help="Optional preload prompt for the new session.",
    )
    parser.add_argument(
        "--harness",
        default=None,
        choices=list(VALID_HARNESSES),
        help="Override the detected harness.",
    )
    args = parser.parse_args()

    harness = args.harness or detect_harness()
    launcher = _LAUNCHERS[harness]
    payload = launcher.build_payload(
        args.work_dir,
        args.issue,
        args.title,
        next_cmd=args.next_cmd,
        next_prompt=args.next_prompt,
    )
    payload["harness"] = harness
    output_json(payload)
    sys.exit(0)


if __name__ == "__main__":
    main()
