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
keeps the contract uniform.

Trailing/leading whitespace inside an otherwise-non-empty
``--next-cmd`` IS preserved verbatim under codex (the only remaining
free-form harness). Claude, cursor, and opencode resolve
``--next-cmd`` strictly via ``PHASE_TO_AGENT`` and reject unknown
values (including those with stray whitespace) with exit code 1
(review fix #02 should-fix #13 — the pre-QS-177 docstring claimed
opencode was free-form too, but the new pipeline made opencode
strict).

**Error contract** (review-fix #03 NTH9, extended in review-fix #04
NTH5/NTH6/NTH7). Four exit shapes, ordered by where they're caught:

- **Argparse user errors** (exit 2, stderr banner) — missing required
  flag, unknown ``--harness`` value (constrained by ``choices=`` so
  the harness name is rejected before dispatch instead of
  ``KeyError``-ing inside ``LAUNCHERS[harness]``). Standard argparse
  contract; we do not catch or reformat.
- **Expected user errors** (exit 1, JSON payload) — unknown phase
  (caught as ``UnknownPhaseError``) and empty / whitespace-only
  ``--next-cmd`` (rejected in main()). Both emit
  ``{"error": ..., "value": ..., ...}`` so downstream scripts can
  parse the failure mode programmatically.
- **Programmer errors** (Python traceback, non-zero exit) — any other
  exception from a launcher, including non-phase ``ValueError`` from
  future launcher code. NOT caught: programmer errors should fail
  loudly so the bug is visible in CI logs.
- **KeyboardInterrupt** (Python traceback) — Ctrl-C during a
  ``build_payload`` call propagates uncaught. Acceptable because a
  half-written launcher script in ``/tmp`` doesn't break anything; the
  user re-runs the phase and the script is overwritten.

The phase-name → agent-name resolution is a static dict
(``launchers/phases.py``) — no filesystem scan, no ``Path.cwd()`` call —
so this script works from any CWD (worktree, main checkout, ``/tmp``,
…). That's AC-9 of QS-175.
"""

from __future__ import annotations

import argparse
import sys

from harness import canonicalize as canonicalize_harness  # type: ignore[import-not-found]
from harness import detect as detect_harness
from harness import harness_choices
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
        # ``harness_choices()`` returns the canonical names PLUS the
        # legacy aliases (review fix #01 N7 + N8): the canonical names
        # match ``LAUNCHERS`` (this module's dispatch table), and the
        # aliases let a user typing ``--harness claude`` pass argparse
        # before ``canonicalize`` collapses it to ``claude-code`` for
        # dispatch.
        choices=harness_choices(),
        help="Override the detected harness.",
    )
    # Optional flags for the review-task → implement-task common loop.
    # When BOTH are provided, the launcher payload gains an
    # ``existing_session_prompt`` field — the paste-into-existing-
    # session prompt the user can drop into an already-running
    # ``qs-implement-task`` session instead of spinning up a fresh
    # terminal. Either flag absent → field omitted from the payload
    # entirely (preserves back-compat with every existing caller).
    # Review fix #01 should-fix #17.
    parser.add_argument(
        "--fix-plan-path",
        default=None,
        help=(
            "Path to the review-fix plan markdown file (absolute or "
            "worktree-relative). When provided alongside `--pr-number`, "
            "the payload includes an `existing_session_prompt` for "
            "pasting into an already-running implement-task session."
        ),
    )
    parser.add_argument(
        "--pr-number",
        type=int,
        default=None,
        help=(
            "PR number for the existing-session prompt. Paired with "
            "`--fix-plan-path`; either flag alone is a no-op."
        ),
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

    # Reject empty / whitespace ``--work-dir`` upstream — without
    # this guard, the opencode launcher would build a
    # ``python scripts/qs/spawn_session.py … --directory ''``
    # invocation, the user pastes it, and the failure fires far from
    # the original mistake (review fix #03 should-fix #9). Strip
    # back after validation for parity with spawn_session's main().
    if not args.work_dir.strip():
        parser.error("--work-dir must be a non-empty path")
    args.work_dir = args.work_dir.strip()

    # GitHub PR numbers are always positive integers. ``type=int``
    # accepts the full int range, so ``--pr-number 0`` or
    # ``--pr-number -1`` would otherwise build a confusing
    # ``existing_session_prompt`` with literal ``#0`` / ``#-1``
    # (review fix #02 should-fix #11).
    if args.pr_number is not None and args.pr_number <= 0:
        parser.error("--pr-number must be a positive integer")

    # Canonicalize legacy aliases (review fix #01 N8) before dispatch
    # so the ``LAUNCHERS[harness]`` lookup is keyed on the canonical
    # name — even if the user typed an alias accepted by
    # ``harness_choices()``.
    harness = canonicalize_harness(args.harness) if args.harness else detect_harness()
    launcher = LAUNCHERS[harness]
    # Delegate validation to the launcher: claude/cursor enforce the
    # phase mapping inside ``build_payload``; codex/opencode accept any
    # ``next_cmd`` string. We catch only ``UnknownPhaseError`` here —
    # other ``ValueError`` subclasses must propagate so a future failure
    # mode isn't misreported as "unknown phase" (review-fix #02 SF1).
    try:
        # Pass ``caller="next_step"`` explicitly rather than relying
        # on the launcher signature default — if a launcher ever
        # changes its default, this call site would silently drift
        # (review fix #03 nice-to-have #23). The matching
        # ``caller="setup_task"`` is already explicit in
        # ``setup_task.py``.
        payload = launcher.build_payload(
            args.work_dir,
            args.issue,
            args.title,
            next_cmd=args.next_cmd,
            next_prompt=args.next_prompt,
            caller="next_step",
            fix_plan_path=args.fix_plan_path,
            pr_number=args.pr_number,
        )
    except UnknownPhaseError as exc:
        output_json({
            "error": "unknown phase",
            "value": exc.value,
            "known": exc.known,
        })
        sys.exit(1)
    payload["harness"] = harness
    # Review fix #01 S9: always emit ``existing_session_prompt`` so the
    # consuming agent prose has a single shape to check
    # (``null`` vs. non-empty string) instead of distinguishing
    # ``key missing`` from ``key present but null``. The launchers
    # only add the key when both ``--fix-plan-path`` and ``--pr-number``
    # are provided; ``setdefault`` fills the gap without overriding a
    # legitimate launcher-emitted value.
    payload.setdefault("existing_session_prompt", None)
    output_json(payload)
    sys.exit(0)


if __name__ == "__main__":
    main()
