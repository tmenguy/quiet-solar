"""Launcher payload for Cursor (2.4+).

Cursor exposes two launchers:

- ``cursor <path>`` — opens the Cursor IDE on a folder. Best for an
  interactive workflow where the user wants the GUI.
- ``cursor-agent --workspace <path> [prompt]`` — the headless Cursor CLI
  agent. Best for scripted / terminal-only flows. When the binary is on
  PATH we add ``--agent qs-<phase>`` so the new session boots directly
  into the phase orchestrator (parity with Claude's ``claude --agent``,
  see QS-175).

We emit a short ``sh /tmp/qs_cursor_<N>.sh`` one-liner that prefers
``cursor`` (the IDE launcher — most common dev experience) and falls
back to a clear text instruction when ``cursor`` is not on PATH. The
payload also includes a ``cli_context`` key with the equivalent
``cursor-agent`` invocation for users who prefer terminal-only flows.

Cursor 2.4+ supports subagents in `.cursor/agents/` and slash-command
invocation (`/<name>`). The same set of phase agents we ship for Claude
Code is mirrored under `.cursor/agents/` with `readonly:` instead of
`tools:` in the frontmatter.
"""

from __future__ import annotations

import shlex
import shutil
import tempfile
from pathlib import Path

from launchers.phases import (  # type: ignore[import-not-found]
    resolve_agent_for_next_cmd,
)


def _cursor_ide_command(
    work_dir: str,
    issue: int | str,
    title: str,
    *,
    next_cmd: str,
    next_prompt: str | None,
) -> str:
    """Build a ``sh /tmp/qs_cursor_<N>.sh`` one-liner that opens Cursor on a worktree."""
    tab_title = f"QS_{issue}: {title}"
    safe_title = shlex.quote(tab_title)
    safe_dir = shlex.quote(work_dir)

    banner_lines = [
        f"echo '── Cursor opening on {tab_title} ──'",
        f"echo '── In the chat, type: {next_cmd} ──'",
    ]
    if next_prompt:
        banner_lines.append(
            f"echo {shlex.quote(f'── Initial prompt: {next_prompt} ──')}",
        )

    banner = " && ".join(banner_lines)
    body = (
        f"#!/bin/sh\n"
        f"{banner} && "
        f"printf '\\033]0;%s\\007' {safe_title} && "
        f"cursor {safe_dir}\n"
    )

    script_path = Path(tempfile.gettempdir()) / f"qs_cursor_{issue}.sh"
    script_path.write_text(body)
    script_path.chmod(0o755)
    return f"sh {script_path}"


def _slash_form(next_cmd: str) -> str:
    """Normalize a phase reference to its slash form (``"/create-plan"``).

    Used in the Cursor fallback prompt so the user lands on a slash
    command they can paste into Cursor's chat — review-fix #2 catches
    the regression where a bare phase (``"create-plan"``) was embedded
    in the fallback prompt with no leading slash.
    """
    return next_cmd if next_cmd.startswith("/") else f"/{next_cmd}"


def _cursor_cli_command(
    work_dir: str,
    *,
    next_cmd: str,
    next_prompt: str | None,
    agent: str | None,
) -> str:
    """Build the equivalent headless ``cursor-agent`` invocation.

    When ``cursor-agent`` is on PATH we add ``--agent qs-<phase>`` so the
    new session boots straight into the phase orchestrator (QS-175,
    parity with Claude's ``claude --agent``).

    When the binary is missing we fall back to the legacy prompt-positional
    form — the user is expected to type ``/<phase>`` themselves in chat
    once they open Cursor manually. The manual equivalent is documented
    in ``build_payload``'s fallback ``instructions`` block.
    """
    safe_dir = shlex.quote(work_dir)
    if agent is not None:
        # Preferred path — Cursor 2.4+ supports ``--agent <name>``. The
        # next_prompt is appended as the initial chat message; the agent
        # body is loaded as the system prompt.
        invocation = f"cursor-agent --workspace {safe_dir} --agent {shlex.quote(agent)}"
        if next_prompt:
            invocation += f" {shlex.quote(next_prompt)}"
        return invocation
    # Fallback — older cursor-agent binaries / no binary at all. Pre-fill
    # the prompt with the SLASH form of the command (review-fix #2) so the
    # user lands on the right phase once the session opens. Manual
    # equivalent: open Cursor on the worktree → in chat, type ``/<phase>``.
    prompt_parts = [_slash_form(next_cmd)]
    if next_prompt:
        prompt_parts.append(next_prompt)
    prompt = "\n\n".join(prompt_parts)
    return f"cursor-agent --workspace {safe_dir} {shlex.quote(prompt)}"


def build_payload(
    work_dir: str,
    issue: int | str,
    title: str,
    *,
    next_cmd: str,
    next_prompt: str | None = None,
) -> dict:
    """Return launcher payload for Cursor.

    Always returns ``new_context`` (the IDE launcher script), ``agent``
    (the resolved ``qs-<phase>`` name) and ``cli_context`` (the
    ``cursor-agent`` invocation). When ``cursor-agent`` is on PATH the
    ``cli_context`` includes ``--agent qs-<phase>`` so the new session
    boots straight into the phase orchestrator (parity with Claude's
    ``claude --agent``). When ``cursor-agent`` is missing, ``cli_context``
    falls back to the legacy prompt-positional form and the user types
    ``/<phase>`` manually in Cursor's chat after the IDE opens.

    Raises:
        ValueError: if ``next_cmd`` is not a known phase. No silent
            fallback — same contract as the Claude launcher.
    """
    agent = resolve_agent_for_next_cmd(next_cmd)
    # Use ``--agent`` only when the cursor-agent binary is on PATH; older
    # versions don't understand it. We probe presence rather than
    # ``--help``-sniffing to keep this cheap and version-agnostic.
    agent_for_cli = agent if shutil.which("cursor-agent") else None

    payload: dict = {
        "tool": "cursor",
        "agent": agent,
        "same_context": next_cmd,
        "cli_context": _cursor_cli_command(
            work_dir,
            next_cmd=next_cmd,
            next_prompt=next_prompt,
            agent=agent_for_cli,
        ),
        "issue": issue,
        "work_dir": work_dir,
    }

    if shutil.which("cursor"):
        payload["new_context"] = _cursor_ide_command(
            work_dir, issue, title, next_cmd=next_cmd, next_prompt=next_prompt,
        )
    else:
        instructions = [
            "Cursor IDE CLI not on PATH. Open the worktree manually:",
            f"    {work_dir}",
            "",
            "Or use the headless Cursor agent CLI:",
            f"    {payload['cli_context']}",
            "",
            # Manual equivalent of the ``--agent`` path: the user opens
            # Cursor and types the slash command in chat. Same degraded
            # one-shot UX as Claude Desktop, documented honestly. The
            # slash form is what Cursor's chat expects, so we normalize
            # here too (review-fix #2) — a bare phase name would not be
            # recognized as a slash command.
            f"Then in chat, type: {_slash_form(next_cmd)}",
        ]
        if next_prompt:
            instructions.extend(["", "Initial prompt:", next_prompt])
        payload["new_context"] = "\n".join(instructions)

    return payload
