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
from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

from utils import (  # type: ignore[import-not-found]
    get_main_worktree,
    get_worktree_dir,
    output_json,
    run_gh,
    run_git,
)


def _phase(next_cmd: str) -> str:
    """Normalise a next-command to its phase name (N7).

    Exactly one leading ``/`` is stripped — so ``/create-plan`` and
    ``create-plan`` both give ``create-plan`` while ``//decompose-epic`` stays
    ``/decompose-epic`` (still refused). One shared helper for the three call
    sites (:func:`refuse_if_epic`, :func:`refuse_decompose_epic_for_wrong_lane`,
    :func:`_fail_render`) that previously each inlined the same slice, matching
    ``launchers.phases.resolve_agent_for_next_cmd``.
    """
    return next_cmd[1:] if next_cmd.startswith("/") else next_cmd


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


def refuse_if_epic(
    issue: int,
    labels: list[str],
    next_cmd: str = "/create-plan",
    *,
    no_worktree: bool = False,
) -> None:
    """Refuse to cut a branch/worktree for a ``scale:epic`` issue — except
    the epic × factory lane's short-lived docs-only worktree (QS-340).

    QS-332 review-fix #04 (must-fix): an epic has **no implement phase and
    no PR** — its output is a rationale document on ``main`` plus child
    issues. QS-340 codifies the epic × factory lane
    (``docs/workflow/lanes/epic-factory.md``): its ``decompose-epic``
    session runs in a short-lived worktree that is discarded once the
    document is on ``main``. So ``scale:epic`` is allowed **only** when the
    next phase is ``decompose-epic``, the target is ``factory`` and a
    worktree is cut (``--no-worktree`` would run the session on the main
    checkout). Everything else is refused as before — epic × product waits
    for #339.

    ``next_cmd`` is normalised exactly like
    ``launchers.phases.resolve_agent_for_next_cmd`` (one leading ``/``
    stripped, so ``//decompose-epic`` stays refused). Consumes
    ``check_declaration``'s labels — no extra ``gh`` call.
    """
    axes = targets.parse_axes(labels)
    if axes["scale"] != "epic":
        return
    phase = _phase(next_cmd)
    if phase == "decompose-epic" and axes["target"] == "factory" and not no_worktree:
        return
    output_json({
        "error": (
            f"issue #{issue} is scale:epic — refusing to create a branch or worktree"
        ),
        "scale": "epic",
        "detail": (
            "An epic has no implement phase and no PR. Its output is a "
            "rationale document on `main` plus child issues. For an epic × "
            "factory issue, run setup-task with `--next-cmd decompose-epic` "
            "and a worktree (no `--no-worktree`): the decompose-epic session "
            "files each child as its own task lane, declaring the parent with "
            "a `### Parent epic` section. The epic × product lane is not "
            "codified yet (#339)."
        ),
    })
    sys.exit(1)


def refuse_decompose_epic_for_wrong_lane(
    issue: int, labels: list[str], next_cmd: str = "/create-plan"
) -> None:
    """S6: ``--next-cmd decompose-epic`` is valid only for the epic × factory lane.

    ``refuse_if_epic`` returns early for a non-epic issue, so without this a
    feature or bug task routed to ``decompose-epic`` would still get a branch
    and worktree cut for an agent that refuses at once. Refuse up front. The
    phase is normalised exactly like :func:`refuse_if_epic` (one leading
    ``/`` stripped). Consumes ``check_declaration``'s labels — no extra
    ``gh`` call.
    """
    phase = _phase(next_cmd)
    if phase != "decompose-epic":
        return
    axes = targets.parse_axes(labels)
    if axes["lane"] == "epic-factory":
        return
    output_json({
        "error": (
            f"issue #{issue} is not an epic × factory issue — refusing "
            "`--next-cmd decompose-epic`"
        ),
        "lane": axes["lane"],
        "detail": (
            "`decompose-epic` runs only in the epic × factory lane "
            "(scale:epic × target:factory). This issue's lane is "
            f"{axes['lane'] or 'undeclared'!r}; route it to its own phase "
            "(e.g. `--next-cmd create-plan` for a task)."
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
    "opencode": opencode_launcher,
    "codex": codex_launcher,
}


def _fail_render(exc: Exception, work_dir: str, issue: int, title: str, next_cmd: str) -> None:
    """Emit the JSON render-failure error and exit 1 (QS-357).

    The branch/worktree already exist, so the remedy is to render by hand
    and then rebuild the launcher payload from the existing worktree — the
    ``detail`` names both commands verbatim, with the real next phase
    (QS-340: one leading ``/`` stripped, as ``next_step.py`` expects).
    """
    phase = _phase(next_cmd)
    output_json({
        "error": "agent render failed",
        "detail": (
            f"{exc}. Remedy: python scripts/qs/render_agents.py --work-dir "
            f"{work_dir}, then python scripts/qs/next_step.py --next-cmd "
            f"{phase} --work-dir {work_dir} --issue {issue} --title "
            f"{title!r} for the launcher"
        ),
    })
    sys.exit(1)


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
    # and refuse an EPIC — except the epic × factory decompose-epic
    # worktree (QS-340). The labels come from the same fetch, so this
    # costs no extra `gh` call.
    labels = check_declaration(issue)
    refuse_if_epic(issue, labels, args.next_cmd, no_worktree=args.no_worktree)
    refuse_decompose_epic_for_wrong_lane(issue, labels, args.next_cmd)

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

    # QS-357: render the per-worktree harness agent files before building the
    # launcher payload, for BOTH the worktree and the --no-worktree branch.
    # The branch/worktree already exist here, so a render failure is fatal —
    # a session would otherwise open with no (or stale) agent definitions.
    # ``fetch=False``: title + labels are already in hand (no second `gh`
    # call). The import is function-local so a missing ``jinja2`` surfaces as
    # ``ImportError`` inside this guard rather than at module import.
    try:
        import render_agents  # noqa: PLC0415 — local so a missing jinja2 is caught here

        render_context = render_agents.build_render_context(
            work_dir, issue=issue, title=title, labels=labels, fetch=False,
        )
        render_agents.render_all(work_dir, context=render_context)
    except ImportError as exc:
        _fail_render(exc, work_dir, issue, title, args.next_cmd)
    except render_agents.RenderError as exc:
        _fail_render(exc, work_dir, issue, title, args.next_cmd)

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
        # QS-358: the labels' lane feeds the Claude pin's effortLevel.
        lane=targets.parse_axes(labels)["lane"] or None,
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
