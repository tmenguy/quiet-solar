#!/usr/bin/env python3
"""Emit a Task-handoff payload for the next OpenCode phase.

Unlike the Claude/Cursor flow, each OpenCode phase ends by:

1. **Rendering** the next phase's per-task agent file via
   ``scripts/qs_opencode/render_agent.py``.
2. **Spawning** that freshly-rendered agent via the Task tool (or printing
   exact instructions if the environment can't auto-spawn).

This script produces a single JSON payload describing both steps, so the
current subagent can execute them deterministically at end-of-phase.

Usage::

    python scripts/qs_opencode/next_step.py \\
        --phase implement-task \\
        --issue 42 --pr 5 \\
        --work-dir /path/to/worktree \\
        --title "Story 3.2: foo bar" \\
        --story-file _qsprocess_opencode/stories/QS-42.story.md

The ``--phase`` argument is the phase that is **finishing** (not the next
one). The script maps it to the next agent via ``PHASE_TRANSITIONS``.
"""

from __future__ import annotations

import argparse
import shlex

from utils import (  # type: ignore[import-not-found]
    build_launcher_payload,
    output_json,
)


# Each finished phase maps to the set of agents that must be rendered before
# the next Task spawn. Most phases render exactly one agent; implement-task
# renders the review orchestrator plus four reviewer sub-roles so review-task
# can spawn them immediately without further rendering.
PHASE_TRANSITIONS: dict[str, dict[str, object] | None] = {
    "setup-task": {
        # Phase 1's handoff is a new OpenCode session, NOT a Task spawn.
        # setup-task renders qs-create-plan-QS-<N>.md into the worktree and
        # delegates the session launch to launch_opencode.py. This entry is
        # therefore unused by next_step.py; kept for completeness.
        "next_agent_phase": "create-plan",
        "render_phases": ["create-plan"],
        "handoff": "launcher",
    },
    "create-plan": {
        "next_agent_phase": "implement-task",
        "render_phases": ["implement-task"],
        "handoff": "task",
    },
    "implement-task": {
        "next_agent_phase": "review-task",
        # Render the orchestrator AND all four reviewer sub-roles so
        # review-task can Task-spawn them in parallel without further I/O.
        "render_phases": [
            "review-task",
            "review-blind-hunter",
            "review-edge-case-hunter",
            "review-acceptance-auditor",
            "review-coderabbit",
        ],
        "handoff": "task",
    },
    "review-task": {
        "next_agent_phase": "finish-task",
        "render_phases": ["finish-task"],
        "handoff": "task",
    },
    "finish-task": None,  # terminal — release is a separate static agent
    "release": None,  # terminal (kept for completeness)
}


def _render_cmd(
    *,
    phase: str,
    work_dir: str,
    issue: int,
    title: str,
    story_file: str | None,
    pr: int | None,
) -> str:
    """Build the exact ``render_agent.py`` invocation for one phase."""
    parts = [
        "python",
        "scripts/qs_opencode/render_agent.py",
        "--phase",
        phase,
        "--work-dir",
        shlex.quote(work_dir),
        "--issue",
        str(issue),
        "--title",
        shlex.quote(title),
    ]
    if story_file:
        parts.extend(["--story-file", shlex.quote(story_file)])
    if pr is not None:
        parts.extend(["--pr", str(pr)])
    return " ".join(parts)


def _spawn_prompt(
    *,
    next_agent: str,
    issue: int,
    pr: int | None,
    work_dir: str,
    title: str,
    story_file: str | None,
) -> str:
    """Build the prompt to pass to ``Task(next_agent, prompt=...)``."""
    lines = [
        f"You are continuing the Quiet Solar pipeline. Activate agent "
        f"`{next_agent}` which was rendered per-task for this issue.",
        "",
        "Context (also baked into your system prompt):",
        f"- issue: #{issue}",
        f"- title: {title}",
        f"- worktree: {work_dir}",
    ]
    if pr is not None:
        lines.append(f"- pr: #{pr}")
    if story_file:
        lines.append(f"- story_file: {story_file}")
    lines.append("")
    lines.append(
        "All task-specific context and tool permissions are baked into "
        "your agent file. Begin your phase protocol now."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit a Task-handoff payload for the next OpenCode phase",
    )
    parser.add_argument(
        "--phase", required=True, choices=sorted(PHASE_TRANSITIONS.keys()),
        help="Current phase (the one finishing)",
    )
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--pr", type=int, default=None)
    parser.add_argument("--story-file", default=None)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    transition = PHASE_TRANSITIONS[args.phase]
    if transition is None:
        output_json({
            "phase_done": args.phase,
            "next_agent": None,
            "summary": f"Phase {args.phase} is terminal; no handoff emitted.",
        })
        return

    assert isinstance(transition, dict)  # for type-checker
    next_phase = str(transition["next_agent_phase"])
    next_agent = f"qs-{next_phase}-QS-{args.issue}"

    render_cmds = [
        _render_cmd(
            phase=str(phase),
            work_dir=args.work_dir,
            issue=args.issue,
            title=args.title,
            story_file=args.story_file,
            pr=args.pr,
        )
        for phase in transition["render_phases"]  # type: ignore[arg-type]
    ]

    prompt = _spawn_prompt(
        next_agent=next_agent,
        issue=args.issue,
        pr=args.pr,
        work_dir=args.work_dir,
        title=args.title,
        story_file=args.story_file,
    )

    # Build a launcher-command fallback. If the running OpenCode does not
    # hot-reload newly-rendered agent files (so Task-spawn fails with
    # "unknown agent"), the finishing agent prints this command; the user
    # runs it to start a fresh OpenCode session where the new agent file
    # is guaranteed to be visible.
    launcher_payload = build_launcher_payload(
        args.work_dir,
        args.issue,
        args.title,
        agent=next_agent,
        preload_command="Begin your phase protocol.",
        same_context_text=f"Activate {next_agent} and begin.",
    )

    instructions = [
        "To complete the handoff, execute these steps in order:",
        "",
        "1. Render the next agent file(s):",
        *[f"   $ {cmd}" for cmd in render_cmds],
        "",
        "2. Try to spawn the next agent in the current session via the Task tool:",
        f"   Task(subagent_type={next_agent!r}, prompt=<spawn_prompt below>)",
        "",
        "3. If Task-spawn fails because OpenCode does not hot-reload agent",
        "   files mid-session (symptom: 'unknown agent' / 'agent not found'),",
        "   fall back to the launcher: present `launcher_command` to the user",
        "   and stop. The user will run it to start a fresh OpenCode session",
        "   where the newly-rendered agent file is visible.",
    ]

    output_json({
        "phase_done": args.phase,
        "next_phase": next_phase,
        "next_agent": next_agent,
        "render_commands": render_cmds,
        "spawn_prompt": prompt,
        "launcher_command": launcher_payload["new_context"],
        "launcher_payload": launcher_payload,
        "instructions_for_current_agent": "\n".join(instructions),
        "optional": bool(transition.get("optional", False)),
        "context": {
            "issue_number": args.issue,
            "pr_number": args.pr,
            "story_file": args.story_file,
            "worktree": args.work_dir,
            "title": args.title,
        },
        "summary": (
            f"Handoff: render {len(render_cmds)} agent(s), try Task-spawn "
            f"{next_agent}, fallback to launcher if hot-reload unavailable"
        ),
    })


if __name__ == "__main__":
    main()
