"""Launcher payload for Claude Code (CLI + Desktop on macOS).

The Claude launcher emits a ``sh /tmp/qs_launch_<N>.sh`` one-liner whose
generated script invokes::

    claude {CLAUDE_LAUNCH_OPTS} --agent qs-<phase> --name 'QS_<N>: <title>'

(Single quotes — ``shlex.quote`` wraps the ``--name`` argument and the
``--agent`` agent name in single-quote form; the docstring example
mirrors the rendered shell line.)

The ``--agent`` flag is what makes the new session interactive: Claude
Code loads the matching ``.claude/agents/qs-<phase>.md`` body as the
system prompt and the user can converse with the persona mid-flight.
This is the QS-175 fix for the "non-interactive Agent-tool sub-process"
UX of the older slash-command path.

Concurrency note: the script path is deterministic per issue number
(``/tmp/qs_launch_<N>.sh``), so two simultaneous setup-task runs on the
SAME issue would race on the file. This is fine for the single-user
dev pipeline this script is built for; switching to
``NamedTemporaryFile`` would lose the predictable path that the
``new_context`` consumers rely on.
"""

from __future__ import annotations

import platform
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from launchers.phases import (  # type: ignore[import-not-found]
    build_existing_session_prompt,
    resolve_agent_for_next_cmd,
)

# ``caller`` literal — reserved for harness-specific bifurcation
# (the OpenCode launcher uses it to switch between HTTP-API and
# CLI-form payloads; Claude's path is identical for both). Kept as a
# no-op kwarg so all launchers can be dispatched uniformly from
# ``setup_task.py`` and ``next_step.py``.
Caller = Literal["setup_task", "next_step"]

# Extra flags appended to ``claude`` invocations. Kept narrow on purpose
# — users can override via env or by editing this constant.
CLAUDE_LAUNCH_OPTS = "--dangerously-skip-permissions --model opus --effort max"


def _pycharm_bin() -> str | None:
    """Return the PyCharm command or ``.app`` path on macOS, else ``None``."""
    if platform.system() != "Darwin":
        return None
    bin_path = shutil.which("pycharm")
    if bin_path:
        return bin_path
    for app in (
        "/Applications/PyCharm.app",
        "/Applications/PyCharm Professional.app",
        "/Applications/PyCharm CE.app",
    ):
        if Path(app).exists():
            return app
    return None


def _pycharm_open_cmd(pycharm_bin: str, work_dir: str) -> str:
    safe_dir = shlex.quote(work_dir)
    if pycharm_bin.endswith(".app"):
        return f"open -na {shlex.quote(pycharm_bin)} --args {safe_dir}"
    return f"{shlex.quote(pycharm_bin)} {safe_dir}"


def _claude_command(
    work_dir: str,
    issue: int | str,
    title: str,
    *,
    agent: str,
    next_prompt: str | None,
) -> str:
    """Build a short ``sh /tmp/qs_launch_<N>.sh`` one-liner to open Claude.

    The generated script invokes ``claude --agent <agent>`` so the new
    session boots straight into the phase orchestrator persona (QS-175).
    """
    tab_title = f"QS_{issue}: {title}"
    safe_title = shlex.quote(tab_title)
    safe_dir = shlex.quote(work_dir)
    safe_agent = shlex.quote(agent)

    full_cmd = (
        f"printf '\\033]0;%s\\007' {safe_title} && "
        f"cd {safe_dir} && "
        f"claude {CLAUDE_LAUNCH_OPTS} --agent {safe_agent} --name {safe_title}"
    )
    if next_prompt is not None:
        full_cmd += f" {shlex.quote(next_prompt)}"

    script_path = Path(tempfile.gettempdir()) / f"qs_launch_{issue}.sh"
    script_path.write_text(f"#!/bin/sh\n{full_cmd}\n")
    script_path.chmod(0o755)
    return f"sh {script_path}"


def _pycharm_clipboard_command(
    work_dir: str,
    issue: int | str,
    *,
    claude_cmd: str,
    pycharm_bin: str,
) -> str:
    """Open PyCharm on the worktree and copy ``claude_cmd`` to the clipboard."""
    safe_cmd = shlex.quote(claude_cmd)
    open_cmd = _pycharm_open_cmd(pycharm_bin, work_dir)
    script_body = (
        "#!/bin/sh\n"
        f"echo {safe_cmd} | pbcopy\n"
        f"{open_cmd}\n"
        'echo "PyCharm opening on worktree. Command copied to clipboard."\n'
        'echo "In PyCharm: Option+F12 (terminal) -> Cmd+V (paste) -> Enter"\n'
    )
    script_path = Path(tempfile.gettempdir()) / f"qs_pycharm_{issue}.sh"
    script_path.write_text(script_body)
    script_path.chmod(0o755)
    return f"sh {script_path}"


def _pycharm_applescript_command(
    work_dir: str,
    issue: int | str,
    *,
    claude_cmd: str,
    pycharm_bin: str,
) -> str:
    """Open PyCharm and AppleScript-type the claude command into its terminal."""
    safe_cmd = shlex.quote(claude_cmd)
    open_cmd = _pycharm_open_cmd(pycharm_bin, work_dir)
    applescript = (
        'tell application "PyCharm" to activate\n'
        "delay 3\n"
        'tell application "System Events"\n'
        "    key code 111 using {option down}\n"
        "    delay 1\n"
        f'    keystroke "{claude_cmd}"\n'
        "    keystroke return\n"
        "end tell\n"
    )
    safe_applescript = shlex.quote(applescript)
    script_body = (
        "#!/bin/sh\n"
        f"echo {safe_cmd} | pbcopy\n"
        f"{open_cmd}\n"
        'echo "PyCharm opening. Attempting to auto-type command in terminal..."\n'
        'echo "(Requires Accessibility permissions for this terminal app)"\n'
        'echo "Fallback: Option+F12 -> Cmd+V -> Enter"\n'
        "sleep 4\n"
        f"osascript -e {safe_applescript}\n"
    )
    script_path = Path(tempfile.gettempdir()) / f"qs_pycharm_as_{issue}.sh"
    script_path.write_text(script_body)
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
    fix_plan_path: str | None = None,
    pr_number: int | None = None,
) -> dict:
    """Build the launcher payload for Claude Code.

    Args:
        work_dir: Worktree directory the new session should open in.
        issue: Issue number (used for tab title + script path).
        title: Issue title (used for tab title).
        next_cmd: Slash command the user types after the session opens
            (e.g. ``"/create-plan"``).  Surfaced as ``same_context`` so
            the agent can suggest the user run it in the current session
            if they prefer.
        next_prompt: Optional preload prompt for the new session.
        caller: Reserved for harness-specific bifurcation (used by the
            OpenCode launcher). Claude's behaviour is identical for
            both call sites, so the value is accepted and ignored.
        fix_plan_path: Optional path to a review-fix plan markdown
            file. When both ``fix_plan_path`` and ``pr_number`` are
            provided, the payload gains an ``existing_session_prompt``
            field — the prompt the user can paste into an already-
            running ``qs-implement-task`` session (review-task →
            implement-task common loop). See
            ``launchers/phases.py::build_existing_session_prompt``.
        pr_number: Optional PR number for the existing-session prompt.

    Returns:
        A dict with ``tool``, ``agent``, ``same_context``, ``new_context``,
        optionally ``existing_session_prompt``, and (on macOS with
        PyCharm installed) ``pycharm_context`` /
        ``pycharm_applescript_context`` keys.

    Raises:
        ValueError: if ``next_cmd`` is not a known phase. No silent
            fallback — free-form prompts go through ``--next-prompt``.
    """
    del caller  # reserved for harness-specific bifurcation; not used here
    agent = resolve_agent_for_next_cmd(next_cmd)
    new_context = _claude_command(
        work_dir, issue, title, agent=agent, next_prompt=next_prompt,
    )

    payload: dict = {
        "tool": "claude-code",
        "agent": agent,
        "same_context": next_cmd,
        "new_context": new_context,
    }

    existing_prompt = build_existing_session_prompt(
        work_dir, fix_plan_path, pr_number,
    )
    if existing_prompt is not None:
        payload["existing_session_prompt"] = existing_prompt

    pycharm_bin = _pycharm_bin()
    if pycharm_bin:
        payload["pycharm_context"] = _pycharm_clipboard_command(
            work_dir, issue, claude_cmd=new_context, pycharm_bin=pycharm_bin,
        )
        payload["pycharm_applescript_context"] = _pycharm_applescript_command(
            work_dir, issue, claude_cmd=new_context, pycharm_bin=pycharm_bin,
        )

    return payload
