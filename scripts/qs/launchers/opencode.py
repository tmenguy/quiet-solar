"""Launcher payload for the new OpenCode pipeline.

This launcher has a bifurcated contract driven by the ``caller=`` kwarg:

- ``caller="next_step"`` (the default) — intermediate-phase handoff
  inside an already-open worktree. We POST to the OpenCode HTTP API
  via ``scripts/qs/spawn_session.py`` to create a new session in the
  *same* OpenCode instance with the next phase's agent already bound
  and the kickoff prompt already sent. No ``/instance/reload`` is
  needed (static ``.opencode/agents/`` files are discovered at server
  startup), so agent activation and prompt arrival happen in-band on
  ``POST /session/<id>/prompt_async``.

- ``caller="setup_task"`` — Phase 1 → ``create-plan`` handoff that
  crosses workspaces (the new worktree is a different OpenCode
  workspace, so HTTP-API session creation against the main-checkout
  server cannot open it as a project). We emit a CLI-form
  ``sh /tmp/qs_oc_launch_<N>.sh`` one-liner whose generated script
  runs ``opencode <worktree> --agent <name> --prompt <kickoff>``
  (parallel shape to ``scripts/qs_opencode/utils.py:opencode_launch_command``).

The HTTP API path falls back to the CLI form when the OpenCode server
is unreachable; ``spawn_session.py`` decides the fallback at runtime
and surfaces the result via stdout JSON. See the script's docstring
for the full fallback decision tree.

Known limitation (per QS-177 AC #12): the activation API call succeeds
even when ``.opencode/agents/qs-<phase>.md`` does not yet exist, but
the session lands on the default OpenCode agent instead of the
intended phase orchestrator. Mirror ``.claude/agents/*.md`` into
``.opencode/agents/`` (with frontmatter conversion: ``tools:`` →
``permission:``, ``mode: primary`` / ``mode: subagent``) to enable
agent activation. This is a documented follow-up, not a silent bug.
"""

from __future__ import annotations

import shlex
import tempfile
from pathlib import Path
from typing import Literal

from launchers.phases import (  # type: ignore[import-not-found]
    resolve_agent_for_next_cmd,
)

# Default kickoff prompt — mirrors ``scripts/qs/spawn_session.py``'s
# ``DEFAULT_KICKOFF``. Kept here as a local string so the launcher
# stays self-contained (no import from the script we shell out to).
DEFAULT_KICKOFF = "Begin your phase protocol."

# ``caller`` literal — both call sites (``setup_task.py`` and
# ``next_step.py``) dispatch to the same launcher; this kwarg lets the
# OpenCode launcher tell "Phase 1 cross-workspace" apart from
# "intermediate in-worktree handoff". Other launchers accept the kwarg
# but ignore it (documented as "reserved for harness-specific
# bifurcation").
Caller = Literal["setup_task", "next_step"]


def _spawn_session_command(
    work_dir: str,
    issue: int | str,
    title: str,
    *,
    agent: str,
    kickoff: str,
) -> str:
    """Return the ``python scripts/qs/spawn_session.py …`` invocation.

    Single-line shell command (no newlines, no ``sh /tmp/…`` wrapper);
    every interpolated value is shell-escaped via ``shlex.quote``
    matching the convention in ``scripts/qs/launchers/claude.py``.
    """
    tab_title = f"QS_{issue}: {title}"
    return (
        "python scripts/qs/spawn_session.py "
        f"--agent {shlex.quote(agent)} "
        f"--directory {shlex.quote(work_dir)} "
        f"--title {shlex.quote(tab_title)} "
        f"--prompt {shlex.quote(kickoff)}"
    )


def _opencode_cli_command(
    work_dir: str,
    issue: int | str,
    title: str,
    *,
    agent: str,
    kickoff: str,
) -> str:
    """Return a ``sh /tmp/qs_oc_launch_<N>.sh`` one-liner (CLI form).

    Parallel to ``scripts/qs_opencode/utils.py:opencode_launch_command``
    — but re-implemented here to keep this module self-contained
    (Task 2.4 in QS-177's story).

    The generated script invokes ``opencode <worktree> --agent <name>
    --prompt <kickoff>`` with the worktree, agent, and kickoff all
    shell-escaped via ``shlex.quote``. A short banner echoes the
    intended phase + kickoff so the user can paste the prompt
    manually if the ``--prompt`` flag ever changes.
    """
    tab_title = f"QS_{issue}: {title}"
    safe_title = shlex.quote(tab_title)
    safe_dir = shlex.quote(work_dir)
    safe_agent = shlex.quote(agent)
    safe_kickoff = shlex.quote(kickoff)

    opencode_cmd = (
        f"opencode {safe_dir} --agent {safe_agent} --prompt {safe_kickoff}"
    )
    full_cmd = (
        f"printf '\\033]0;%s\\007' {safe_title} && "
        f"echo '── Activating agent: {agent} ──' && "
        f"echo '── Initial prompt (paste manually if preload fails) ──' && "
        f"echo {safe_kickoff} && "
        f"{opencode_cmd}"
    )

    script_path = Path(tempfile.gettempdir()) / f"qs_oc_launch_{issue}.sh"
    script_path.write_text(f"#!/bin/sh\n{full_cmd}\n")
    script_path.chmod(0o755)
    return f"sh {script_path}"


def build_payload(
    work_dir: str,
    issue: int | str,
    title: str,
    *,
    next_cmd: str,
    next_prompt: str | None = None,
    caller: Caller = "next_step",
) -> dict:
    """Return the OpenCode launcher payload for ``next_cmd``.

    Args:
        work_dir: Worktree directory the new session should open in.
        issue: Issue number (used for tab title + temp-script path).
        title: Issue/story title (used for the tab title).
        next_cmd: Slash command or bare phase name for the next phase
            (e.g. ``"create-plan"`` or ``"/create-plan"``). Validation
            is delegated to ``resolve_agent_for_next_cmd``.
        next_prompt: Optional kickoff prompt; defaults to
            ``DEFAULT_KICKOFF`` when ``None``.
        caller: ``"next_step"`` (default) → HTTP API form; or
            ``"setup_task"`` → CLI form (cross-workspace launch).

    Returns:
        Dict with ``tool``, ``agent``, ``same_context``, ``new_context``.

    Raises:
        UnknownPhaseError: when ``next_cmd`` is not a known phase. The
            caller (typically ``next_step.py``) catches this and emits
            a JSON error payload.
    """
    agent = resolve_agent_for_next_cmd(next_cmd)
    kickoff = next_prompt if next_prompt is not None else DEFAULT_KICKOFF

    if caller == "setup_task":
        new_context = _opencode_cli_command(
            work_dir, issue, title, agent=agent, kickoff=kickoff,
        )
    else:
        new_context = _spawn_session_command(
            work_dir, issue, title, agent=agent, kickoff=kickoff,
        )

    return {
        "tool": "opencode",
        "agent": agent,
        "same_context": next_cmd,
        "new_context": new_context,
    }
