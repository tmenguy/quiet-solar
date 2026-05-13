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

``--next-cmd`` accepts both ``/create-plan`` (back-compat) and
``create-plan`` (bare phase name) for the claude/cursor launchers.
Validation is delegated to the launcher's ``build_payload`` so the
codex and opencode launchers — which have no agent mapping today —
accept any non-empty ``--next-cmd`` string unchanged. On a known
harness with an unknown phase, ``build_payload`` raises
``UnknownPhaseError`` and this script writes a JSON error to stdout +
exits non-zero. Other ``ValueError`` subclasses (from future launcher
code) are deliberately NOT caught — review-fix #02 SF1.

Empty / whitespace-only ``--next-cmd`` is rejected by ``main()`` after
``parser.parse_args()`` returns, for ALL harnesses (review-fix #02
SF2). The check sits after parse rather than inside an argparse
``type=`` callable because argparse type errors print a usage banner
to stderr, but the rest of this script speaks a JSON error contract
(``{"error": ..., "value": ..., ...}``) — putting the check in main()
keeps the contract uniform. Trailing/leading whitespace inside an
otherwise-non-empty ``--next-cmd`` (e.g. ``"create-plan "``) IS
preserved verbatim under codex/opencode (review-fix #03 NTH7); those
launchers treat ``--next-cmd`` as free-form, so an intentional space
is the user's call.

**Bimodal error contract** (review-fix #03 NTH9): expected failures
(unknown phase, empty next-cmd) emit a JSON ``{"error": ...}`` payload
and exit 1. Programmer errors (any other exception from a launcher,
including non-phase ``ValueError`` from future launcher code) propagate
uncaught — the user gets a Python traceback. This is intentional:
programmer errors should fail loudly so the bug is visible in CI logs;
user errors should emit a parseable JSON contract.

The phase-name → agent-name resolution is a static dict
(``launchers/phases.py``) — no filesystem scan, no ``Path.cwd()`` call —
so this script works from any CWD (worktree, main checkout, ``/tmp``,
…). That's AC-9 of QS-175.
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
from launchers.phases import UnknownPhaseError  # type: ignore[import-not-found]

from utils import output_json  # type: ignore[import-not-found]

# Public mapping (review-fix #03 SF1) — tests and future extension code
# treat this as the harness-dispatch configuration table. Keeping it
# under a public name avoids SLF001-style import smells in test code
# that monkeypatches the dispatcher.
LAUNCHERS = {
    "claude-code": claude_launcher,
    "cursor": cursor_launcher,
    "opencode": opencode_launcher,
    "codex": codex_launcher,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a harness-aware handoff payload.")
    parser.add_argument(
        "--next-cmd",
        required=True,
        help=(
            "Phase name for the next step. Accepts either the bare form "
            "('create-plan') or the slash form ('/create-plan') for "
            "back-compat under the claude/cursor launchers. Free-form "
            "strings are passed through unchanged under codex/opencode "
            "(no agent mapping there). See --next-prompt for an initial "
            "prompt that loads into the new session."
        ),
    )
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

    # Reject empty / whitespace-only --next-cmd for every harness
    # (review-fix #02 SF2). Without this, codex/opencode would silently
    # pass the empty string through to ``build_payload`` and emit a
    # garbled payload with ``same_context: ""``. The first ``not`` form
    # is dropped (review-fix #03 NTH2) — ``"".strip()`` is also falsy,
    # so a single check covers both empty and whitespace cases.
    if not args.next_cmd.strip():
        output_json({
            "error": "empty next-cmd",
            "value": args.next_cmd,
            "detail": "--next-cmd must be a non-empty, non-whitespace string",
        })
        sys.exit(1)

    harness = args.harness or detect_harness()
    launcher = LAUNCHERS[harness]
    # Delegate validation to the launcher: claude/cursor enforce the
    # phase mapping inside ``build_payload``; codex/opencode accept any
    # ``next_cmd`` string. We catch only ``UnknownPhaseError`` here —
    # other ``ValueError`` subclasses must propagate so a future failure
    # mode isn't misreported as "unknown phase" (review-fix #02 SF1).
    try:
        payload = launcher.build_payload(
            args.work_dir,
            args.issue,
            args.title,
            next_cmd=args.next_cmd,
            next_prompt=args.next_prompt,
        )
    except UnknownPhaseError as exc:
        output_json({
            "error": "unknown phase",
            "value": exc.value,
            "known": exc.known,
        })
        sys.exit(1)
    payload["harness"] = harness
    output_json(payload)
    sys.exit(0)


if __name__ == "__main__":
    main()
