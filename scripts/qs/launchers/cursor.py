"""Launcher payload for Cursor (2.4+).

Cursor exposes two launchers:

- ``cursor <path>`` — opens the Cursor IDE on a folder. Best for an
  interactive workflow where the user wants the GUI.
- ``cursor-agent --workspace <path> [prompt]`` — the headless Cursor CLI
  agent. Best for scripted / terminal-only flows.

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


def _cursor_cli_command(
    work_dir: str,
    *,
    next_cmd: str,
    next_prompt: str | None,
) -> str:
    """Build the equivalent headless ``cursor-agent`` invocation."""
    safe_dir = shlex.quote(work_dir)
    # Cursor CLI passes the prompt as a positional arg.  We pre-fill it with
    # the slash command (and optional preload prompt) so the agent kicks
    # off on its own when the session opens.
    prompt_parts = [next_cmd]
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

    Always returns ``new_context`` (the IDE launcher script) and
    ``cli_context`` (the ``cursor-agent`` invocation).  When the
    ``cursor`` CLI is not on PATH, ``new_context`` falls back to plain
    instructions so the user can act manually.
    """
    payload: dict = {
        "tool": "cursor",
        "same_context": next_cmd,
        "cli_context": _cursor_cli_command(
            work_dir, next_cmd=next_cmd, next_prompt=next_prompt,
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
            f"Then in chat, type: {next_cmd}",
        ]
        if next_prompt:
            instructions.extend(["", "Initial prompt:", next_prompt])
        payload["new_context"] = "\n".join(instructions)

    return payload
