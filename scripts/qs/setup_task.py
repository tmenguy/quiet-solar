#!/usr/bin/env python3
"""Create a branch + worktree for a task and emit the launcher.

Usage::

    python scripts/qs/setup_task.py <issue_number> --title "..."
        [--no-worktree] [--harness HARNESS] [--next-cmd "/create-plan"]

Output: JSON containing worktree path, branch, and a harness-specific
launcher payload (``new_context`` is the shell command or instructions
the agent should surface to the user).
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from harness import VALID_HARNESSES  # type: ignore[import-not-found]
from harness import detect as detect_harness
from launchers import claude as claude_launcher  # type: ignore[import-not-found]
from launchers import codex as codex_launcher  # type: ignore[import-not-found]
from launchers import cursor as cursor_launcher  # type: ignore[import-not-found]
from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

from utils import (  # type: ignore[import-not-found]
    get_main_worktree,
    get_worktree_dir,
    output_json,
    run_git,
)

# Public mapping (review-fix #04 SF1) — promoted to match the
# round-3 SF1 rename of next_step.LAUNCHERS. The two dispatch tables
# are conceptually the same configuration; keeping the naming
# convention aligned avoids drift and lets test code monkeypatch
# either via the public attribute. Kept as a local copy (not imported
# from ``next_step``) so ``setup_task`` stays independent of the
# next-phase dispatcher.
LAUNCHERS = {
    "claude-code": claude_launcher,
    "cursor": cursor_launcher,
    "opencode": opencode_launcher,
    "codex": codex_launcher,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create branch + worktree for a task")
    parser.add_argument("issue_number", type=int, help="GitHub issue number")
    parser.add_argument("--title", default=None, help="Issue/story title for display")
    parser.add_argument("--no-worktree", action="store_true", help="Branch only — no worktree")
    parser.add_argument(
        "--harness",
        default=None,
        choices=list(VALID_HARNESSES),
        help="Override the detected harness.",
    )
    parser.add_argument(
        "--next-cmd",
        default="/create-plan",
        help="Slash command to surface for the next phase.",
    )
    parser.add_argument(
        "--next-prompt",
        default=None,
        help="Optional preload prompt for the new session.",
    )
    args = parser.parse_args()

    issue = args.issue_number
    branch = f"QS_{issue}"
    main_dir = get_main_worktree()

    run_git(["fetch", "origin"], cwd=str(main_dir))

    if args.no_worktree:
        result = run_git(["branch", branch, "origin/main"], cwd=str(main_dir), check=False)
        if result.returncode != 0 and "already exists" not in result.stderr:
            output_json({"error": "Failed to create branch", "detail": result.stderr.strip()})
            sys.exit(1)
        work_dir = str(main_dir)
    else:
        setup_script = main_dir / "scripts" / "worktree-setup.sh"
        result = subprocess.run(
            ["bash", str(setup_script), str(issue)],
            capture_output=True,
            text=True,
            cwd=str(main_dir),
        )
        if result.returncode != 0:
            output_json({
                "error": "Worktree setup failed",
                "detail": result.stderr.strip() or result.stdout.strip(),
            })
            sys.exit(1)
        work_dir = str(get_worktree_dir(issue))

    title = args.title or f"Issue #{issue}"

    harness = args.harness or detect_harness()
    launcher = LAUNCHERS[harness]
    # ``caller="setup_task"`` tells the OpenCode launcher that this is
    # the Phase 1 → create-plan cross-workspace handoff (the new worktree
    # is a different OpenCode workspace than the main checkout), so it
    # should emit the CLI-form launcher instead of the HTTP-API
    # ``spawn_session.py`` invocation. Other launchers accept and ignore
    # the kwarg (QS-177 AC #8 / #9).
    launcher_payload = launcher.build_payload(
        work_dir,
        issue,
        title,
        next_cmd=args.next_cmd,
        next_prompt=args.next_prompt,
        caller="setup_task",
    )

    output_json({
        "issue_number": issue,
        "branch": branch,
        "worktree_path": work_dir,
        "no_worktree": args.no_worktree,
        "harness": harness,
        **launcher_payload,
    })


if __name__ == "__main__":
    main()
