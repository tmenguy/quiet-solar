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
from typing import Literal, NamedTuple

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

# Fail-closed mode for the pin family (settings file, ``.bak``, temp) when
# the target's own mode cannot be read — typically because there is no
# target yet, which is every fresh worktree. Owner-only: the file can carry
# an ``env`` block with a token (review-fix #02 A / N-a).
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

    A containment check, and deliberately not ``utils.is_worktree`` alone:
    that function is ``resolve() != get_main_worktree().resolve()``, i.e. it
    means "**is not the main checkout**". It returns ``True`` for any
    throwaway path, and because ``get_main_worktree()`` takes no ``cwd`` it
    even returns ``True`` for *this* repo's main checkout when the launcher
    runs from a cwd inside a different git repo — the exact outcome the
    guard exists to prevent (review-fix #01 S4).

    A linked worktree's ``.git`` is a **file** holding a ``gitdir:``
    pointer; the main checkout's — and any second clone's — is a
    directory, and a throwaway path has none. That one ``stat`` is the
    containment property, so it runs first and ``is_worktree`` stays as the
    explicit statement of intent.
    """
    if not (Path(work_dir) / ".git").is_file():
        return False
    return is_worktree(work_dir)


def _target_mode(target: Path) -> int:
    """Return the permission bits every file in the pin family should carry.

    ``target``'s own bits when it exists, so a deliberate ``chmod`` is
    honoured; otherwise ``_PRIVATE_MODE`` — **fail closed** (review-fix #02
    N-a). The absent-target branch is not an edge case: ``worktree-setup.sh``
    seeds only ``.mypy_cache`` / ``.testmondata``, never a settings file, so
    *every* fresh worktree takes it. Defaulting to the umask there created
    the pin file ``0o644`` and froze that in for every later writer that
    preserves the existing mode.

    Residual, by design: an **existing** file's looser mode is preserved,
    not tightened — the user's ``chmod`` wins either way. A worktree already
    pinned by an earlier version therefore keeps ``0o644`` until the file is
    removed and recreated; ``rm .claude/settings.local.json`` is the fix,
    and the next handoff recreates it at ``_PRIVATE_MODE``.
    """
    try:
        return target.stat().st_mode & 0o7777
    except OSError:
        return _PRIVATE_MODE


def _apply_mode(path: Path, mode: int) -> None:
    """``chmod`` ``path`` to ``mode``, best-effort (never breaks a handoff)."""
    with contextlib.suppress(OSError):
        os.chmod(path, mode)


def _backup(target: Path, raw: bytes) -> None:
    """Copy ``raw`` (``target``'s current bytes) to a ``.bak`` sibling.

    Called only on the rebuild path, so a ``permissions.allow`` list lost
    to an unparseable body stays recoverable. Best-effort like everything
    else here; the ``.bak`` name is covered by the same ``.gitignore``
    pattern as the settings file itself.

    Two properties this got wrong when it shipped:

    * **It must not leak what ``_preserve_mode`` protects** (review-fix #02
      A). This is the one path that *copies* the bytes rather than
      replacing them, so a ``0o600`` file carrying an ``env`` token was
      republished at ``0o666 & ~umask`` under a ``.bak`` name. The mode is
      applied to an **empty** file before the content goes in, so there is
      no window in which the token exists at the looser mode.
    * **It must not clobber a good backup with nothing** (review-fix #02
      C). An empty body is exactly what a ``SIGKILL`` / full disk leaves
      behind while Claude Code rewrites the file non-atomically; copying
      that over a ``.bak`` holding the user's real approvals destroys the
      only recoverable copy. The skip gets its own warning, because the
      generic "could not parse … rebuilding it" line does not say that
      recovery is now impossible.
    """
    backup = target.with_suffix(target.suffix + ".bak")
    if not raw.strip():
        print(
            f"warning: {target} is empty; NOT overwriting {backup} with it — "
            f"this rebuild is unrecoverable"
            + (" (the existing backup is older)" if backup.exists() else ""),
            file=sys.stderr,
        )
        return
    mode = _target_mode(target)
    try:
        # Create/truncate first, tighten, then fill: the bytes never exist
        # at a looser mode. ``touch`` applies ``mode & ~umask`` on creation
        # and nothing at all if the file already exists, so the explicit
        # ``_apply_mode`` covers both cases exactly.
        backup.touch(mode=mode, exist_ok=True)
        _apply_mode(backup, mode)
        backup.write_bytes(raw)
    except OSError as exc:
        print(
            f"warning: could not back up {target} to {backup} ({exc})",
            file=sys.stderr,
        )


def _read_settings(target: Path) -> tuple[dict | None, bool]:
    """Return ``(settings, rebuilt)`` for ``target``.

    ``settings`` is ``None`` when the file must not be touched at all.
    ``rebuilt`` is ``True`` when the user's existing content was discarded
    — surfaced all the way out as the ``settings_rebuilt`` payload key
    (review-fix #02 N-c), because the alternative signal is a stderr line
    that no orchestrator is instructed to relay.

    Three outcomes, deliberately distinct (review-fix #01 M1 / S5):

    * **absent** → ``({}, False)``; there is nothing to preserve.
    * **unreadable** (``OSError``: ``EACCES``, ``EINTR``, a lock, ``EIO``
      on a network mount) → ``(None, False)``. A file we cannot read is a
      file we must not replace: the condition is typically transient and
      this file carries the user's own ``permissions`` decisions.
    * **unparseable or not an object** (``ValueError`` — which covers both
      ``json.JSONDecodeError`` and ``UnicodeDecodeError`` from the decode,
      plus ``null`` / ``[1, 2]`` bodies) → ``({}, True)`` after a warning
      and a ``.bak`` copy. Skipping instead would leave the phase silently
      unbound, which the invisible-agent trap makes worse than a rebuild.
    """
    if not target.exists():
        return {}, False
    try:
        raw = target.read_bytes()
    except OSError as exc:
        print(
            f"warning: could not read {target} ({exc}); leaving it untouched",
            file=sys.stderr,
        )
        return None, False
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        print(
            f"warning: could not parse {target} ({exc}); rebuilding it",
            file=sys.stderr,
        )
        _backup(target, raw)
        return {}, True
    if not isinstance(parsed, dict):
        # Valid JSON, wrong shape (``null``, ``[1, 2]``) — the shallow
        # merge would raise outside any guard. Tracked as its own branch
        # rather than via a ``None`` sentinel, which used to let ``null``
        # through both checks and rebuild the file with no warning at all.
        print(
            f"warning: {target} is not a JSON object; rebuilding it",
            file=sys.stderr,
        )
        _backup(target, raw)
        return {}, True
    return parsed, False


def _render(settings: dict, agent: str) -> str:
    """Return the on-disk form of ``settings`` with ``agent`` pinned."""
    return json.dumps({**settings, "agent": agent}, indent=2) + "\n"


def _late_render(target: Path, agent: str) -> str | None:
    """Re-render from ``target``'s current bytes, or ``None`` to keep the first.

    Shrinks — it does not close — the read-modify-write race against the
    live Claude Code session that owns this file (review-fix #01 S2): the
    handoff normally runs from inside a session on this very worktree, so a
    permission the user approves between our first read and the publish
    would otherwise be dropped. Silent by design: the noisy paths already
    warned on the first read, and anything unreadable or malformed here
    just leaves that first render standing.
    """
    try:
        parsed = json.loads(target.read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return _render(parsed, agent)


def _preserve_mode(target: Path, tmp: Path) -> None:
    """Carry ``target``'s permission bits onto ``tmp`` before the replace.

    ``write_text`` creates the temp at ``0o666 & ~umask`` and
    ``os.replace`` keeps the *temp's* mode, so a ``chmod 600`` (this file
    can carry an ``env`` block with a token) would be silently undone.
    When there is no target yet, ``_target_mode`` falls back to
    ``_PRIVATE_MODE`` rather than leaving the umask default in place
    (review-fix #02 N-a).

    **Caveat — mode only** (review-fix #02 N-b). ``st_mode`` is all that is
    carried: macOS ACLs (``chmod +a``), SELinux labels and other extended
    attributes are dropped by ``os.replace``, so a file whose real
    restriction lives in an ACL comes back looking correctly-moded while
    the restriction is gone. Preserving those portably is not free, and no
    other writer in this repo does it; recorded here rather than fixed.
    """
    _apply_mode(tmp, _target_mode(target))


class _PinResult(NamedTuple):
    """Outcome of one pin attempt, as surfaced in the launcher payload."""

    pinned: bool
    rebuilt: bool


def _write_phase_agent(work_dir: str, agent: str) -> _PinResult:
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
    (shallow merge). This file is **not** purely machine-written — Claude
    Code persists the user's per-project ``permissions`` decisions (and
    ``model``, ``env``, …) in it — so a *read* failure never rewrites it,
    and the rebuild path that an unparseable body does trigger keeps the
    old bytes in a ``.bak`` sibling.

    Best-effort by contract — a handoff must never break because of this
    write, hence the suppressed temp cleanup. Warnings go to
    ``sys.stderr``; ``stdout`` carries the JSON payload that
    ``next_step.py`` callers parse.

    **No ``fsync``** before the replace, matching
    ``quality_gate.py::_write_seed_status`` — the in-repo atomic-write
    precedent this writer was modelled on, whose docstring says the same in
    the same words. A crash between the write and the replace can therefore
    publish a short file. Verified deliberate rather than assumed: there is
    no ``fsync`` anywhere under ``scripts/``, and a stale or truncated pin
    is recoverable by re-running the handoff, so consistency with the
    precedent wins over diverging here (review-fix #02 N-h).

    Returns:
        ``_PinResult(pinned, rebuilt)`` — surfaced to callers as the
        ``phase_agent_pinned`` and ``settings_rebuilt`` payload keys. The
        handoff prose must not assert the pin without consulting the first,
        nor report a clean pin without consulting the second.
    """
    claude_dir = Path(work_dir) / ".claude"
    if not (claude_dir / "agents" / f"{agent}.md").is_file():
        print(
            f"warning: no {agent}.md under {claude_dir / 'agents'}; "
            f"not pinning the phase agent",
            file=sys.stderr,
        )
        return _PinResult(False, False)
    if not _is_linked_worktree(work_dir):
        return _PinResult(False, False)

    target = claude_dir / "settings.local.json"
    # Follow a symlink to its destination before choosing the temp and the
    # replace target (review-fix #02 D). ``os.replace`` onto the link would
    # swap the link itself for a regular file: the content looks right at
    # handoff time, but the shared file the user deliberately linked to
    # never receives another update — surfacing much later as "my approvals
    # stopped syncing". Resolving preserves the intent instead of refusing
    # it. Symlinking a shared settings file is the natural workaround for
    # re-approving permissions in every fresh per-task worktree.
    if target.is_symlink():
        target = target.resolve()

    settings, rebuilt = _read_settings(target)
    if settings is None:
        return _PinResult(False, False)

    content = _render(settings, agent)
    tmp = target.with_suffix(f"{target.suffix}.{os.getpid()}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        late = _late_render(target, agent)
        if late is not None and late != content:
            tmp.write_text(late, encoding="utf-8")
        _preserve_mode(target, tmp)
        os.replace(tmp, target)
    except OSError as exc:
        print(f"warning: could not write {target} ({exc})", file=sys.stderr)
        return _PinResult(False, rebuilt)
    finally:
        # ``missing_ok=True`` only covers FileNotFoundError; EACCES on the
        # directory or EIO on a network mount would propagate out of a
        # bare ``finally`` and break the handoff (review-fix #01 S1).
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    return _PinResult(True, rebuilt)


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
        ``settings_rebuilt``, ``same_context``, ``new_context``, optionally
        ``existing_session_prompt``, and (on macOS with PyCharm installed)
        ``pycharm_context`` / ``pycharm_applescript_context`` keys.
        ``phase_agent_pinned`` is ``False`` whenever the GUI pin was
        skipped or failed — the orchestrator must not claim the pin as
        fact without consulting it. ``settings_rebuilt`` is ``True`` when
        an unparseable local settings file was discarded (old bytes kept in
        a ``.bak`` sibling); the orchestrator must relay that, since a
        rebuild is invisible in ``phase_agent_pinned`` alone.

    Raises:
        ValueError: if ``next_cmd`` is not a known phase. No silent
            fallback — free-form prompts go through ``--next-prompt``.
    """
    del caller  # reserved for harness-specific bifurcation; not used here
    agent = resolve_agent_for_next_cmd(next_cmd)
    # Side effect (QS-311): pin the phase agent into the worktree's local
    # settings so a GUI session there boots as this orchestrator. Guarded
    # and best-effort — see ``_write_phase_agent``. The result is surfaced
    # as ``phase_agent_pinned`` (review-fix #01 M3): discarding it left the
    # GUI handoff blocks asserting a pin that is deterministically absent
    # on ``--no-worktree`` and silently absent on any write failure.
    pin = _write_phase_agent(work_dir, agent)
    new_context = _claude_command(
        work_dir, issue, title, agent=agent, next_prompt=next_prompt,
    )

    payload: dict = {
        "tool": "claude-code",
        "agent": agent,
        "phase_agent_pinned": pin.pinned,
        "settings_rebuilt": pin.rebuilt,
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
