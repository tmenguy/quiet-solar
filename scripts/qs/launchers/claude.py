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

import contextlib
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

# Separate import block on purpose: ``utils`` is a sibling top-level module
# under ``scripts/qs/``, not part of the ``launchers`` package. Not an isort
# ``I001`` violation — verify with ``ruff check --select I001`` before
# "fixing" it.
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

# Mode for a freshly created pin file. Owner-only because this file can
# carry an ``env`` block with a token, and because a fresh worktree has no
# existing mode to honour. An existing file's mode is copied, not replaced.
_PRIVATE_MODE = 0o600


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


def _is_linked_worktree(work_dir: str) -> bool:
    """Return ``True`` if ``work_dir`` is a **linked git worktree**.

    A linked worktree's ``.git`` is a **file** holding a ``gitdir:``
    pointer; the main checkout's — and any second clone's — is a directory,
    and a throwaway path has none. That one ``stat`` is the actual
    containment check, so it runs first.

    ``utils.is_worktree`` alone is **not** sufficient and must not be used
    as if it were: it is ``resolve() != get_main_worktree().resolve()``,
    i.e. "is not the main checkout". It answers ``True`` for any throwaway
    path, and since ``get_main_worktree()`` takes no ``cwd``, even for this
    repo's main checkout when called from inside a different repo. It is
    kept here as the explicit statement of intent, after the real check.
    """
    if not (Path(work_dir) / ".git").is_file():
        return False
    return is_worktree(work_dir)


def _read_settings(target: Path) -> dict | None:
    """Return ``target``'s settings dict, or ``None`` to leave it alone.

    ``None`` means "do not write anything, skip the pin". The user's bytes
    are never modified by this function or by anything downstream of a
    ``None``: this file holds the user's own ``permissions`` decisions, so
    anything we do not fully understand is left alone.

    Three outcomes:

    * **absent** → ``{}``; there is nothing to preserve, so the caller
      writes a fresh file.
    * **present and a JSON object** → the parsed dict, for a shallow merge.
      Decoded as ``utf-8-sig`` so a leading BOM — which several editors
      write by default, and which is *valid* — parses instead of being
      treated as corruption.
    * **anything else** → ``None`` with a warning naming the file, the
      reason, and the remedy. That covers unreadable (``OSError``:
      ``EACCES``, ``EINTR``, a lock, ``EIO`` on a network mount),
      unparseable (``ValueError``, which subsumes ``json.JSONDecodeError``
      *and* ``UnicodeDecodeError``), and parsed-but-not-an-object
      (``null``, ``[1, 2]``, ``"x"`` — a shallow merge would raise outside
      any guard).

    The skip is **terminal, not transient**: nothing repairs the file, so
    every later handoff re-reads it and refuses again. That is why the
    warnings name the remedy explicitly.
    """
    if not target.exists():
        return {}
    try:
        raw = target.read_bytes()
    except OSError as exc:
        print(
            f"warning: could not read {target} ({exc}); leaving it untouched "
            f"and skipping the phase pin — pass --agent, or fix the file's "
            f"permissions to pin it again",
            file=sys.stderr,
        )
        return None
    try:
        parsed = json.loads(raw.decode("utf-8-sig"))
    except ValueError as exc:
        print(
            f"warning: {target} does not parse as JSON ({exc}); leaving it "
            f"untouched and skipping the phase pin — every later handoff "
            f"will skip too until it is repaired; "
            f"`rm {target}` recreates it at 0600",
            file=sys.stderr,
        )
        return None
    if not isinstance(parsed, dict):
        print(
            f"warning: {target} is not a JSON object "
            f"(got {type(parsed).__name__}); leaving it untouched and "
            f"skipping the phase pin — every later handoff will skip too "
            f"until it is repaired; `rm {target}` recreates it at 0600",
            file=sys.stderr,
        )
        return None
    return parsed


def _render(settings: dict, agent: str) -> str:
    """Return the on-disk form of ``settings`` with ``agent`` pinned."""
    return json.dumps({**settings, "agent": agent}, indent=2) + "\n"


def _late_render(target: Path, agent: str) -> str | None:
    """Re-render from ``target``'s current bytes, or ``None`` to keep the first.

    Shrinks — it does not close — the read-modify-write race against the
    live Claude Code session that owns this file: the handoff normally runs
    from inside a session on this very worktree, so a permission the user
    approves between our first read and the publish would otherwise be
    dropped. Silent by design: the first read already warned about anything
    wrong, and anything unreadable, unparseable or non-object here simply
    leaves the first render standing. ``utf-8-sig`` for the same reason as
    ``_read_settings``.
    """
    try:
        parsed = json.loads(target.read_bytes().decode("utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(parsed, dict):
        # A live session replaced the object with a non-object between our
        # two reads. Keep the first render rather than merging onto
        # something a shallow merge would raise on.
        return None
    return _render(parsed, agent)


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
       displays no agent name, so a bad pin would be invisible. This skip
       warns on stderr: it is a real anomaly, not a designed no-op.
       (Project-scoped agents only; user-scope ``~/.claude/agents/`` is
       out of scope.)
    2. ``work_dir`` must be a **linked worktree** — see
       ``_is_linked_worktree``. Silent: it is the designed no-op for
       ``--no-worktree`` and the main-checkout phases, which the caller
       reports via ``phase_agent_pinned``.

    ``agent`` is always replaced; every other top-level key is preserved
    (shallow merge). This file is **not** machine-written — Claude Code
    persists the user's per-project ``permissions`` decisions (and
    ``model``, ``env``, …) in it. Two consequences, both deliberate:

    * anything we cannot read, or cannot parse as a JSON object, is **left
      exactly as it is** and the pin is skipped;
    * a **symlink** at ``.claude`` or at the settings file is **refused**,
      not followed, so every write destination is a literal path inside
      ``work_dir``.

    Best-effort by contract — a handoff must never break because of this
    write, hence the suppressed temp cleanup. Warnings go to
    ``sys.stderr``; ``stdout`` carries the JSON payload that
    ``next_step.py`` callers parse.

    **No ``fsync``** before the replace, matching
    ``quality_gate.py::_write_seed_status`` — the in-repo atomic-write
    precedent. A crash between the write and the replace can therefore
    publish a short file; re-running the handoff repairs it, and diverging
    from the precedent for that is not worth it.

    Returns:
        ``True`` if the file was written, ``False`` on any skip or failure —
        surfaced to callers as the ``phase_agent_pinned`` payload key. The
        handoff prose must not assert the pin without consulting it.
    """
    claude_dir = Path(work_dir) / ".claude"
    if not (claude_dir / "agents" / f"{agent}.md").is_file():
        print(
            f"warning: no {agent}.md under {claude_dir / 'agents'}; "
            f"not pinning the phase agent",
            file=sys.stderr,
        )
        return False
    if not _is_linked_worktree(work_dir):
        return False

    target = claude_dir / "settings.local.json"
    # Refuse a symlink anywhere on the path we are about to write; never
    # follow one. Checking only the pin file was not enough: with
    # ``.claude`` itself a link — a plausible way to share one agents
    # directory — the file is not a link, so the write followed the
    # directory and landed in the main checkout (or ``~/.claude``) while
    # reporting success. Refusing both makes the containment property true
    # by construction: every path below is literally inside ``work_dir``.
    for suspect in (claude_dir, target):
        if suspect.is_symlink():
            print(
                f"warning: {suspect} is a symlink; refusing to pin through "
                f"it, because the write would leave the worktree — pass "
                f"--agent instead",
                file=sys.stderr,
            )
            return False

    settings = _read_settings(target)
    if settings is None:
        return False

    content = _render(settings, agent)
    tmp = target.with_suffix(f"{target.suffix}.{os.getpid()}.tmp")
    try:
        # Ordinary high-level file operations only: ``write_text`` writes
        # fully or raises, and ``copymode`` is one call for "keep whatever
        # mode the user chose". Hand-rolled descriptor-level writing was
        # tried here and produced two distinct defects (a discarded
        # partial-write result, and a temp that followed a symlink), so it
        # is deliberately not used.
        tmp.write_text(content, encoding="utf-8")
        late = _late_render(target, agent)
        if late is not None and late != content:
            tmp.write_text(late, encoding="utf-8")
        if target.exists():
            shutil.copymode(target, tmp)  # keep a deliberate chmod
        else:
            tmp.chmod(_PRIVATE_MODE)  # fresh file: owner-only
        os.replace(tmp, target)
    except OSError as exc:
        print(f"warning: could not write {target} ({exc})", file=sys.stderr)
        return False
    finally:
        # ``missing_ok=True`` only covers FileNotFoundError; EACCES on the
        # directory or EIO on a network mount would otherwise propagate out
        # of this ``finally`` and break the handoff.
        with contextlib.suppress(OSError):
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
        A dict with ``tool``, ``agent``, ``phase_agent_pinned``,
        ``same_context``, ``new_context``, optionally
        ``existing_session_prompt``, and (on macOS with PyCharm installed)
        ``pycharm_context`` / ``pycharm_applescript_context`` keys.
        ``phase_agent_pinned`` is ``False`` whenever the GUI pin was
        skipped or failed — the orchestrator must not claim the pin as fact
        without consulting it. There is deliberately no key reporting a
        settings rebuild: the writer never discards the user's settings, so
        there is nothing of that kind to report.

    Raises:
        ValueError: if ``next_cmd`` is not a known phase. No silent
            fallback — free-form prompts go through ``--next-prompt``.
    """
    del caller  # reserved for harness-specific bifurcation; not used here
    agent = resolve_agent_for_next_cmd(next_cmd)
    # Side effect (QS-311): pin the phase agent into the worktree's local
    # settings so a GUI session there boots as this orchestrator. Guarded
    # and best-effort — see ``_write_phase_agent``. The result is surfaced
    # as ``phase_agent_pinned``: the GUI handoff blocks must not assert a
    # pin that is deterministically absent on ``--no-worktree`` and
    # silently absent on any write failure.
    pinned = _write_phase_agent(work_dir, agent)
    new_context = _claude_command(
        work_dir, issue, title, agent=agent, next_prompt=next_prompt,
    )

    payload: dict = {
        "tool": "claude-code",
        "agent": agent,
        "phase_agent_pinned": pinned,
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
