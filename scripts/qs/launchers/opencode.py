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
    build_existing_session_prompt,
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
    """Return a ``sh <tempfile>`` one-liner (CLI form).

    Parallel to ``scripts/qs_opencode/utils.py:opencode_launch_command``
    — but re-implemented here to keep this module self-contained
    (Task 2.4 in QS-177's story).

    The generated script invokes ``opencode <worktree> --agent <name>
    --prompt <kickoff>`` with the worktree, agent, and kickoff all
    shell-escaped via ``shlex.quote``. A short banner echoes the
    intended phase + kickoff so the user can paste the prompt
    manually if the ``--prompt`` flag ever changes.

    Security / robustness (review fix #01):

    - ``issue`` is coerced via ``int()`` so a malicious string value
      (e.g. ``"42; rm -rf $HOME"``) raises ``ValueError`` upfront
      rather than producing a script path with shell metacharacters
      (must-fix #2).
    - The script is written via ``tempfile.NamedTemporaryFile`` to a
      uniquely-named path under ``tempfile.gettempdir()`` — eliminates
      the symlink-attack vector against a deterministic
      ``/tmp/qs_oc_launch_<N>.sh`` path (should-fix #4).
    - Every interpolated value in the generated shell script — banner
      echo arguments included — is ``shlex.quote``'d to handle quotes
      or backslashes in synthetic agent names (should-fix #5).
    - The returned ``sh <path>`` invocation has its path
      ``shlex.quote``'d so a ``TMPDIR`` with embedded whitespace
      doesn't split the command (covered by must-fix #2 too).
    - On ``OSError`` during file write / chmod (read-only filesystem,
      sandboxed CI, etc.), we fall back to returning the bare
      ``opencode <worktree> --agent <name> --prompt <kickoff>``
      command inline — the user still gets a working command instead
      of an uncaught traceback (should-fix #6).
    """
    # Coerce ``issue`` to int up-front (review fix #01 must-fix #2 —
    # closes the shell-injection vector at the type boundary).
    issue_int = int(issue)

    tab_title = f"QS_{issue_int}: {title}"
    safe_title = shlex.quote(tab_title)
    safe_dir = shlex.quote(work_dir)
    safe_agent = shlex.quote(agent)
    safe_kickoff = shlex.quote(kickoff)
    # Shell-quote the entire banner echo argument so a synthetic
    # agent name containing single quotes or backslashes still
    # produces a syntactically valid script (review fix #01
    # should-fix #5).
    safe_activate_banner = shlex.quote(f"── Activating agent: {agent} ──")
    safe_preload_banner = shlex.quote(
        "── Initial prompt (paste manually if preload fails) ──",
    )

    opencode_cmd = (
        f"opencode {safe_dir} --agent {safe_agent} --prompt {safe_kickoff}"
    )
    full_cmd = (
        f"printf '\\033]0;%s\\007' {safe_title} && "
        f"echo {safe_activate_banner} && "
        f"echo {safe_preload_banner} && "
        f"echo {safe_kickoff} && "
        f"{opencode_cmd}"
    )

    try:
        # ``NamedTemporaryFile`` generates a unique path under
        # ``tempfile.gettempdir()`` — concurrent invocations can't
        # race on a shared filename and a hostile symlink at a
        # deterministic path can't redirect our write.
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"qs_oc_launch_{issue_int}_",
            suffix=".sh",
            dir=tempfile.gettempdir(),
            delete=False,
        ) as f:
            f.write(f"#!/bin/sh\n{full_cmd}\n")
            script_path = Path(f.name)
        # Owner-only rwx: ``sh <path>`` only needs read perm to
        # execute the contents, so 0o700 is the strictest safe
        # mode.
        script_path.chmod(0o700)
    except OSError:
        # Read-only filesystem, sandboxed CI, non-POSIX share —
        # any I/O failure here lands us in the inline-command
        # fallback so the launcher still emits a working
        # ``new_context`` instead of an uncaught traceback
        # propagating to ``setup_task.main()``.
        return opencode_cmd

    return f"sh {shlex.quote(str(script_path))}"


def build_payload(
    work_dir: str,
    issue: int | str,
    title: str,
    *,
    next_cmd: str,
    next_prompt: str | None = None,
    caller: Caller = "next_step",
    fix_plan_path: str | None = None,
    pr_number: int | None = None,
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
        fix_plan_path: Optional path to a review-fix plan markdown
            file. When both ``fix_plan_path`` and ``pr_number`` are
            provided, the payload gains an ``existing_session_prompt``
            field — the paste-into-existing-session prompt for the
            review-task → implement-task common loop.
        pr_number: Optional PR number for the existing-session prompt.

    Returns:
        Dict with ``tool``, ``agent``, ``same_context``,
        ``new_context``, and optionally ``existing_session_prompt``.

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

    payload: dict = {
        "tool": "opencode",
        "agent": agent,
        "same_context": next_cmd,
        "new_context": new_context,
    }
    existing_prompt = build_existing_session_prompt(
        work_dir, fix_plan_path, pr_number,
    )
    if existing_prompt is not None:
        payload["existing_session_prompt"] = existing_prompt
    return payload
