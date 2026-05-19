#!/usr/bin/env python3
"""Build the launcher payload used by ``qs-setup-task`` (Phase 1).

Phase 1 does **not** use Task-handoff because Phase 2 runs in a *new*
OpenCode session rooted at the worktree. This script emits JSON
containing:

- ``same_context`` — instruction to continue in the current session
- ``new_context`` — ``sh /tmp/qs_oc_launch_<N>.sh`` one-liner that opens
  a new OpenCode on the worktree with the rendered create-plan agent
  pre-activated via ``--agent`` and a kickoff ``--prompt``
- ``pycharm_context`` / ``pycharm_applescript_context`` — macOS + PyCharm
  convenience variants (only when both are detected)

Usage::

    python scripts/qs_opencode/launch_opencode.py \\
        --issue 42 \\
        --title "Story 3.2: foo" \\
        --work-dir /path/to/worktree \\
        --agent qs-create-plan-QS-42 \\
        --preload-command "Begin your phase protocol."
"""

from __future__ import annotations

import argparse

from utils import build_launcher_payload, output_json  # type: ignore[import-not-found]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build OpenCode launcher payload for Phase 1 handoff",
    )
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--agent",
        default=None,
        help='Rendered agent to activate, e.g. "qs-create-plan-QS-42". Passed to opencode as --agent.',
    )
    parser.add_argument(
        "--preload-command",
        default=None,
        help='Initial prompt for the new session, e.g. "Begin your phase protocol." Passed to opencode as --prompt.',
    )
    parser.add_argument(
        "--same-context",
        default=None,
        help="Override the same-context text. Defaults to the preload command.",
    )
    args = parser.parse_args()

    same_context = args.same_context or (
        args.preload_command or (f"Activate agent {args.agent} and run its phase protocol." if args.agent else "")
    )
    payload = build_launcher_payload(
        args.work_dir,
        args.issue,
        args.title,
        agent=args.agent,
        preload_command=args.preload_command,
        same_context_text=same_context,
    )
    output_json(payload)


if __name__ == "__main__":
    main()
