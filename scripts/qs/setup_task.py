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
import json
import subprocess
import sys

import targets  # type: ignore[import-not-found]

from harness import canonicalize as canonicalize_harness  # type: ignore[import-not-found]
from harness import detect as detect_harness
from harness import harness_choices
from launchers import claude as claude_launcher  # type: ignore[import-not-found]
from launchers import codex as codex_launcher  # type: ignore[import-not-found]
from launchers import cursor as cursor_launcher  # type: ignore[import-not-found]
from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

from utils import (  # type: ignore[import-not-found]
    get_main_worktree,
    get_worktree_dir,
    output_json,
    run_gh,
    run_git,
)


def check_declaration(issue: int) -> list[str]:
    """Refuse to proceed unless ``issue`` carries a complete lane declaration.

    Returns the issue's label names so the caller can enforce
    scale-specific invariants without a second ``gh`` call
    (:func:`refuse_if_epic`).

    QS-332 B2 — enforcement by construction: this runs BEFORE any
    branch/worktree work. One ``gh issue view --json labels`` call
    (``check=False`` with an explicit JSON-error path, like its peers);
    the validity rule itself lives in :func:`targets.validate_declaration`
    (one truth table, three consumers). The refusal message carries the
    exact shape-aware ``gh issue edit --add-label`` backfill command.
    """
    result = run_gh(["issue", "view", str(issue), "--json", "labels"], check=False)
    if result.returncode != 0:
        output_json({
            "error": f"Failed to fetch labels for issue #{issue}",
            "detail": result.stderr.strip(),
        })
        sys.exit(1)
    try:
        # `or []` (QS-332 review-fix #03): `"labels": null` IS valid JSON,
        # so reporting "Invalid JSON from gh CLI" for it was misleading —
        # the honest verdict is the ordinary missing-declaration refusal
        # below, which prints an actionable backfill command.
        labels = [lb["name"] for lb in json.loads(result.stdout).get("labels") or []]
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        # `AttributeError` (QS-332 review-fix #04): a non-dict top-level
        # value (`null`, `[]`, `42`) makes `.get` raise, which used to
        # escape this guard as a raw traceback.
        output_json({"error": "Invalid JSON from gh CLI", "detail": result.stdout.strip()})
        sys.exit(1)

    ok, missing, message = targets.validate_declaration(labels)
    if not ok:
        output_json({
            "error": f"issue #{issue} has no complete lane declaration — refusing to proceed",
            "missing": missing,
            "detail": message.replace("<N>", str(issue)),
        })
        sys.exit(1)
    return labels


def refuse_if_epic(issue: int, labels: list[str]) -> None:
    """Refuse to cut a branch/worktree for a ``scale:epic`` issue.

    QS-332 review-fix #04 (must-fix). The epic model
    (:doc:`docs/epics/QS-321`) is explicit: an epic has **no implement
    phase; no branch, worktree, or PR** — its output is a rationale
    document on ``main`` plus child issues. Step 2 of the setup agent used
    to run this script unconditionally, so picking an epic lane (or
    passing an existing epic via ``--issue N``) cut a worktree anyway.

    Machine-enforced here rather than prompt-obeyed, matching the story's
    "machine-checked rather than prompt-obeyed" philosophy, and enforced
    for ``--no-worktree`` too: that flag still creates a **branch**, which
    the model forbids as well. Consumes ``check_declaration``'s labels —
    no extra ``gh`` call.
    """
    if targets.parse_axes(labels)["scale"] != "epic":
        return
    output_json({
        "error": (
            f"issue #{issue} is scale:epic — refusing to create a branch or worktree"
        ),
        "scale": "epic",
        "detail": (
            "An epic has no implement phase and no branch, worktree, or PR. "
            "Its output is a rationale document on `main` plus child issues: "
            "decompose it into child tasks (each child is its own task lane, "
            "carrying `Refs #<epic>`) and run setup-task on those instead."
        ),
    })
    sys.exit(1)

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
        # ``harness_choices()`` returns the canonical names PLUS the
        # legacy aliases (review fix #01 N7 + N8) so a user typing
        # ``--harness claude`` passes argparse and is canonicalized to
        # ``claude-code`` before dispatch.
        choices=harness_choices(),
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

    # QS-332 B2: an issue must be born in exactly one lane; refuse an
    # undeclared/inconsistent one before touching git. Review-fix #04:
    # and refuse an EPIC outright — no branch, no worktree (the labels
    # come from the same fetch, so this costs no extra `gh` call).
    labels = check_declaration(issue)
    refuse_if_epic(issue, labels)

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

    # Apply the legacy-alias mapping (review fix #01 N8): argparse
    # accepted aliases via ``choices=harness_choices()``; canonicalize
    # collapses them to canonical names before dispatch so the
    # ``LAUNCHERS[harness]`` lookup never KeyErrors on a legacy alias.
    harness = canonicalize_harness(args.harness) if args.harness else detect_harness()
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
