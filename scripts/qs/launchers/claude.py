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

import json
import os
import platform
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Literal

from launchers.phases import (  # type: ignore[import-not-found]
    build_existing_session_prompt,
    resolve_agent_for_next_cmd,
)

from utils import is_worktree  # type: ignore[import-not-found]

# ``caller`` literal — reserved for harness-specific bifurcation
# (the OpenCode launcher uses it to switch between HTTP-API and
# CLI-form payloads; Claude's path is identical for both). Kept as a
# no-op kwarg so all launchers can be dispatched uniformly from
# ``setup_task.py`` and ``next_step.py``.
Caller = Literal["setup_task", "next_step"]

# Extra flags appended to ``claude`` invocations. Kept narrow on purpose
# — users can override via env or by editing this constant.
CLAUDE_LAUNCH_OPTS = "--dangerously-skip-permissions --model opus"


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


def _write_phase_agent(work_dir: str, agent: str) -> bool:
    """Pin ``agent`` into ``<work_dir>/.claude/settings.local.json`` (QS-311).

    The Claude Code **GUI** has no ``--agent`` flag, so the only way to
    boot a GUI session as a phase orchestrator is the ``agent`` settings
    key, which Claude Code reads at *session* start. Writing it at every
    handoff keeps the worktree pinned to the phase the pipeline just
    handed off to. CLI sessions are unaffected: ``--agent`` overrides the
    setting. See ``docs/workflow/harness.md`` → "GUI launch surface
    (Claude Code Desktop)".

    Guards, in this order (the order is load-bearing):

    1. the phase agent file must exist at
       ``<work_dir>/.claude/agents/<agent>.md`` — a pure filesystem check,
       first so it short-circuits before any subprocess. An unknown agent
       name falls back to the default agent *silently*, and the GUI
       displays no agent name, so a bad pin would be invisible.
       (Project-scoped agents only; user-scope ``~/.claude/agents/`` is
       out of scope.)
    2. ``work_dir`` must be a worktree — never the main checkout (covers
       ``--no-worktree`` and the main-checkout phases).

    ``agent`` is always replaced; every other top-level key is preserved
    (shallow merge). A corrupt or non-object existing file is rebuilt from
    scratch with a warning: the file is machine-written and gitignored, and
    skipping would leave the phase silently unbound.

    Best-effort by contract — a handoff must never break because of this
    write. Warnings go to ``sys.stderr``; ``stdout`` carries the JSON
    payload that ``next_step.py`` callers parse.

    Returns:
        ``True`` if the file was written, ``False`` on any skip or failure.
    """
    claude_dir = Path(work_dir) / ".claude"
    if not (claude_dir / "agents" / f"{agent}.md").is_file():
        return False
    if not is_worktree(work_dir):
        return False

    target = claude_dir / "settings.local.json"
    settings: dict = {}
    if target.exists():
        loaded: object = None
        try:
            loaded = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # ``ValueError`` covers json.JSONDecodeError AND
            # UnicodeDecodeError (which is NOT an OSError).
            print(
                f"warning: could not read {target} ({exc}); rewriting it",
                file=sys.stderr,
            )
        if isinstance(loaded, dict):
            settings = loaded
        elif loaded is not None:
            # Valid JSON, wrong shape (e.g. ``[1, 2]``) — the shallow
            # merge below would raise outside any guard.
            print(
                f"warning: {target} is not a JSON object; rewriting it",
                file=sys.stderr,
            )

    settings["agent"] = agent
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        print(f"warning: could not write {target} ({exc})", file=sys.stderr)
        return False
    finally:
        tmp.unlink(missing_ok=True)
    return True


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

    Side effect (QS-311, deliberate — flagged here because the name says
    "build"): also pins the resolved agent into
    ``<work_dir>/.claude/settings.local.json`` via ``_write_phase_agent``,
    so a Claude Code **GUI** session opened on the worktree boots as the
    phase orchestrator (the GUI has no ``--agent`` flag). It lives here
    rather than in the two callers (``setup_task.py`` / ``next_step.py``)
    to avoid duplicating the call at every handoff site. The write is
    guarded to real worktrees that already contain the agent file, and is
    inert for CLI sessions because ``--agent`` takes precedence.

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
    # Side effect (QS-311): pin the phase agent into the worktree's local
    # settings so a GUI session there boots as this orchestrator. Guarded
    # and best-effort — see ``_write_phase_agent``.
    _write_phase_agent(work_dir, agent)
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
