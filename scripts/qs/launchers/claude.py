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
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

import models  # type: ignore[import-not-found]
from launchers.phases import (  # type: ignore[import-not-found]
    build_existing_session_prompt,
    resolve_agent_for_next_cmd,
)

# Separate import block on purpose: ``utils`` is a sibling top-level module
# under ``scripts/qs/``, not part of the ``launchers`` package. Not an isort
# ``I001`` violation — verify with ``ruff check --select I001`` before
# "fixing" it. (``models`` is a sibling too, but isort sorts it into the
# block above.)
from utils import is_worktree  # type: ignore[import-not-found]

# ``caller`` literal — reserved for harness-specific bifurcation
# (the OpenCode launcher uses it to switch between HTTP-API and
# CLI-form payloads; Claude's path is identical for both). Kept as a
# no-op kwarg so all launchers can be dispatched uniformly from
# ``setup_task.py`` and ``next_step.py``.
Caller = Literal["setup_task", "next_step"]

# Extra flags appended to ``claude`` invocations. Kept narrow on purpose.
# No ``--model`` (QS-358 D9): the model comes from the policy in
# ``scripts/qs/models.py``, rendered as a full model ID into each agent's
# frontmatter (D20). Frontmatter decides the model on the CLI and for
# sub-agents; the Claude GUI's model picker decides the main session's, so
# the payload names it as ``phase_model`` (QS-367 E7). The GUI pin carries
# the phase's ``effortLevel`` (D19). A one-off ``claude --model <id>`` still
# overrides the frontmatter.
CLAUDE_LAUNCH_OPTS = "--dangerously-skip-permissions"

# Mode for a freshly created pin file. Owner-only because this file can
# carry an ``env`` block with a token, and because a fresh worktree has no
# existing mode to honour. An existing file's mode is copied, not replaced.
_PRIVATE_MODE = 0o600


def check_cli_floor(
    version_text: str, floor: tuple[int, int, int] = models.CLAUDE_CLI_FLOOR,
) -> str | None:
    """Warn if the Claude Code CLI ``version_text`` is below ``floor``.

    Parses a ``MAJOR.MINOR.PATCH`` triple out of ``claude --version`` output
    (e.g. ``"2.1.278 (Claude Code)"`` — deliberately below the floor) into a
    tuple and compares it to ``floor`` — default
    :data:`models.CLAUDE_CLI_FLOOR`, the ``deep`` ``claude-opus-5-5``
    requires (QS-367 S4/E8), referenced from there so the floor and the
    model needing it cannot drift. A triple *tagged* as the build version —
    either suffixed ``(Claude Code)`` or prefixed ``Claude Code`` — wins
    over any earlier bare triple, so an ``"Update available: 2.1.290"``
    banner cannot mask the real build version printed as
    ``"2.1.278 (Claude Code)"`` or ``"Claude Code 2.1.278"``
    (QS-367 S2/N8); absent any tag the first bare triple is used. The scan
    tolerates a leading ``v`` and a banner line before the version. Each
    digit group is bounded to nine digits so a pathological run cannot
    overflow ``int`` (QS-367 N1). Returns a one-line warning string when
    strictly below the floor, else ``None``.

    Pure and total: unparseable input (no dotted triple) returns ``None``
    rather than raising — a best-effort guard must never itself break a
    handoff. The warning names the ``deep`` model from
    :data:`models.HARNESS_MODELS` (QS-367 S4) so a future bump cannot leave
    the message naming the wrong build.
    """
    match = re.search(
        r"(?<![\d.])(\d{1,9})\.(\d{1,9})\.(\d{1,9})\s*\(Claude Code\)"
        r"|Claude Code\s+v?(\d{1,9})\.(\d{1,9})\.(\d{1,9})",
        version_text,
    ) or re.search(r"(?<![\d.])(\d{1,9})\.(\d{1,9})\.(\d{1,9})", version_text)
    if match is None:
        return None
    # The tagged alternation yields six groups (the unmatched branch is all
    # ``None``); the bare fallback yields three. Filter to the three that
    # matched, whichever branch won.
    digits = [g for g in match.groups() if g is not None]
    version = (int(digits[0]), int(digits[1]), int(digits[2]))
    if version >= floor:
        return None
    floor_s = ".".join(str(part) for part in floor)
    version_s = ".".join(str(part) for part in version)
    deep_model = models.model_for("claude", "deep")
    return (
        f"warning: the `claude` on PATH is Claude Code {version_s}, below "
        f"{floor_s}; the `deep` agents pin `{deep_model}`, which needs ≥ "
        f"{floor_s} (older builds 400 on it). Upgrade the CLI (`claude` will "
        f"fail mid-fan-out until you do)."
    )


def _warn_if_cli_below_floor() -> None:
    """Best-effort: warn on stderr if the local ``claude`` CLI predates the floor.

    Runs ``claude --version`` with a short timeout and hands the output to
    :func:`check_cli_floor`. Any failure — binary missing
    (``FileNotFoundError``), a timeout, a non-UTF-8 shim whose output raises
    ``UnicodeDecodeError`` (a ``ValueError``; guarded belt-and-braces despite
    ``errors="replace"``), or any other ``OSError`` / ``SubprocessError`` —
    is swallowed silently; the guard must never block or alter the payload.
    ``stdin`` is closed (``DEVNULL``) so the child cannot inherit the
    caller's stdin, and both streams are concatenated and scanned (some
    builds print the version to ``stderr``, or a banner to ``stdout`` ahead
    of it, so scanning only the first non-empty stream would miss it —
    QS-367 S2). Only a successfully parsed, below-floor version prints, and
    only to ``sys.stderr`` (``stdout`` carries the JSON payload), matching
    how the render warnings behave.
    """
    try:
        proc = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        # Inside the ``try`` on purpose: ``check_cli_floor`` is pure and total
        # for sane input, but a pathological version string could still make
        # ``int()`` raise — the widened ``except`` keeps even that from
        # breaking the handoff (QS-367 N1).
        warning = check_cli_floor(f"{proc.stdout or ''}\n{proc.stderr or ''}")
    except (OSError, ValueError, subprocess.SubprocessError):
        return
    if warning is not None:
        print(warning, file=sys.stderr)


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


def _render(settings: dict, agent: str, effort: str | None) -> str:
    """Return the on-disk form of ``settings`` with ``agent`` (and effort) pinned.

    The writer owns exactly two keys (QS-358): ``agent``, and
    ``effortLevel`` — set to ``effort``, or removed when ``effort`` is
    ``None`` so a ``fast`` phase never inherits the previous phase's level.
    It never writes ``model``; a ``model`` already in the file is the
    user's and is kept.
    """
    merged = {**settings, "agent": agent}
    if effort is None:
        merged.pop("effortLevel", None)
    else:
        merged["effortLevel"] = effort
    return json.dumps(merged, indent=2) + "\n"


def _late_render(target: Path, agent: str, effort: str | None) -> str | None:
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
    return _render(parsed, agent, effort)


def _write_phase_agent(work_dir: str, agent: str, effort: str | None) -> bool:
    """Pin ``agent`` into ``<work_dir>/.claude/settings.local.json`` (QS-311).

    QS-358: the pin also carries the phase's ``effortLevel`` (``effort``;
    removed when ``None``). Frontmatter ``effort:`` reaches sub-agents but
    not the main session, so this key is the main session's only path. The
    settings pin fixes the **agent**, not the model: frontmatter decides the
    model on the CLI surface (and for sub-agents), while the Claude GUI's
    model picker decides the main session's and must be set by hand — the
    handoff names it as ``phase_model`` (QS-367 E7). A settings ``model``
    key loses to the GUI picker, so this writer never sets one.

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

    ``agent`` is always replaced (``effortLevel`` set or removed); every
    other top-level key is preserved
    (shallow merge). This file is **not** machine-written — Claude Code
    persists the user's per-project ``permissions`` decisions (and
    ``model``, ``env``, …) in it. Two consequences, both deliberate:

    * anything we cannot read, or cannot parse as a JSON object, is **left
      exactly as it is** and the pin is skipped;
    * a **symlink** is **refused**, not followed, at any of the three paths
      this function touches: ``.claude``, the settings file, and the temp
      sibling. Those three are the writes, so refusing them keeps the writes
      inside ``work_dir``. (Stated as the enumeration it is, rather than as
      an invariant: an earlier phrasing claimed containment "by
      construction" while the temp was still missing from the list.)

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
    tmp = target.with_suffix(f"{target.suffix}.{os.getpid()}.tmp")
    # Refuse a symlink at any of the three paths this function writes to or
    # through; never follow one. Each was found the hard way: a link at
    # ``.claude`` let the write land in the main checkout while reporting
    # success, and a link at the *temp* name — reachable via a leftover temp
    # plus a reused PID — sent the merged settings outside the worktree and
    # then renamed the link onto the pin file, leaving the worktree
    # permanently unpinnable. ``write_text`` follows links, so this check is
    # what keeps the writes inside ``work_dir``.
    for suspect in (claude_dir, target, tmp):
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

    content = _render(settings, agent, effort)
    try:
        # Ordinary high-level file operations only: ``write_text`` writes
        # fully or raises, and ``copymode`` is one call for "keep whatever
        # mode the user chose". Hand-rolled descriptor-level writing was
        # tried here and produced two distinct defects (a discarded
        # partial-write result, and a temp that followed a symlink), so it
        # is deliberately not used.
        tmp.write_text(content, encoding="utf-8")
        late = _late_render(target, agent, effort)
        if late is not None and late != content:
            tmp.write_text(late, encoding="utf-8")
        # Non-fatal on purpose: the content is already written and
        # ``os.replace`` needs only directory permission, so a mode failure
        # (``EPERM`` on a chmod-hostile mount, a settings file owned by
        # another uid, or a ``FileNotFoundError`` if the live session
        # replaced ``target`` between the check and the stat) degrades to
        # "published at the default mode" rather than "not pinned at all",
        # which would be permanent for that worktree.
        #
        # It warns, though: silence here is sticky. One failure publishes at
        # ``0o666 & ~umask``, Claude Code may then persist an ``env`` token
        # into that same file, and every later handoff faithfully copies the
        # widened mode forward — so a transient failure would otherwise
        # leave a secrets-bearing file world-readable for good, with an
        # empty stderr and ``phase_agent_pinned: True``.
        try:
            if target.exists():
                shutil.copymode(target, tmp)  # keep a deliberate chmod
            else:
                tmp.chmod(_PRIVATE_MODE)  # fresh file: owner-only
        except OSError as exc:
            print(
                f"warning: could not set the mode of {target} ({exc}); "
                f"publishing it at the default mode rather than skipping "
                f"the pin — check the mode if the file holds secrets",
                file=sys.stderr,
            )
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


# QS-372: the Claude user-facing handoff block, produced here once instead of
# being re-typed (and drifting) at every phase template's handoff site. The
# wording is byte-for-byte the pre-QS-372 rendered review-task block, except
# the new unpinned notice (``_GUI_UNPINNED_BLOCK``, QS-372 AC6 (3)); the
# templates now print ``handoff_text`` verbatim. ``_FALLBACK_PREAMBLE`` and the
# GUI blocks are fixed text; only the values in braces vary.
_FALLBACK_PREAMBLE = (
    "Fallback (stay in this session, degraded one-shot UX via the Agent tool \u2014\n"
    "kept for any chat without a CLI launcher; the GUI can instead run the phase\n"
    "agent directly, see `docs/workflow/harness.md`):"
)

_GUI_PINNED_BLOCK = (
    "[Claude Code GUI] the worktree should now be pinned to `{agent}` in\n"
    "`.claude/settings.local.json` (the payload's `phase_agent_pinned` reports\n"
    "whether that write happened \u2014 it is always skipped on a main checkout).\n"
    "The GUI displays the active agent nowhere, so if the phase looks wrong,\n"
    "use the Preferred line above, where `--agent` always wins.\n"
    "  \u2022 **New session** (not a restored one \u2014 the GUI reopens the last session)\n"
    "  \u2022 Select directory `{work_dir}`\n"
    "  \u2022 Name it `QS_{issue} {phase}`\n"
    "  \u2022 **Pick model `{phase_model}`** in the model picker (the GUI ignores\n"
    "    the agent's model \u2014 see harness.md); if the picker does not offer it,\n"
    "    use the Preferred `--agent` line above (its frontmatter pins the model)\n"
    "  \u2022 See `docs/workflow/harness.md` \u2192\n"
    '    "GUI launch surface (Claude Code Desktop)".'
)

# ``phase_agent_pinned: false`` cannot tell "no pin" from a **stale** pin left
# by the previous phase, so the GUI bullets are dropped entirely and the user
# is routed to the ``--agent`` line (the pre-QS-372 templates asked the LLM to
# make this cut; it is now deterministic).
_GUI_UNPINNED_BLOCK = (
    "[Claude Code GUI] the phase pin was not written (`phase_agent_pinned`\n"
    "is false), and the worktree may still carry the previous phase's pin \u2014\n"
    "use the Preferred `--agent` line above, which is correct either way."
)


def _handoff_text(
    *,
    agent: str,
    new_context: str,
    work_dir: str,
    issue: int | str,
    phase_model: str,
    pinned: bool,
    existing_session_prompt: str | None,
) -> str:
    """Return the ready-to-print Claude handoff block (QS-372).

    Blocks are separated by one blank line; there is no trailing newline.
    The existing-session block appears only for a non-empty
    ``existing_session_prompt`` (every non-empty prompt line indented two
    spaces; blank lines stay empty); the GUI block depends on ``pinned``
    (see ``_GUI_UNPINNED_BLOCK``).
    """
    phase = agent.removeprefix("qs-")
    blocks = [
        f"Next phase: {phase}.",
        f"Preferred (opens a fresh interactive `claude --agent {agent}` session):\n"
        f"  {new_context}",
    ]
    if existing_session_prompt:
        prompt = "\n".join(
            f"  {line}" if line else "" for line in existing_session_prompt.split("\n")
        )
        blocks.append(
            "Already running an implementation session?\n"
            f"Paste this prompt into it:\n{prompt}"
        )
    blocks.append(f"{_FALLBACK_PREAMBLE}\n  /{phase}")
    if pinned:
        blocks.append(
            _GUI_PINNED_BLOCK.format(
                agent=agent, work_dir=work_dir, issue=issue, phase=phase,
                phase_model=phase_model,
            )
        )
    else:
        blocks.append(_GUI_UNPINNED_BLOCK)
    return "\n\n".join(blocks)


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
    lane: str | None = None,
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
    inert for CLI sessions because ``--agent`` takes precedence. The pin
    also carries the phase's effort level (QS-358), resolved from
    ``scripts/qs/models.py`` for ``lane``. The model is not *pinned*:
    frontmatter decides it on the CLI and for sub-agents, while the Claude
    GUI's model picker decides the main session's (a settings ``model`` key
    loses to it), so the payload instead *names* it as ``phase_model`` for
    the GUI handoff to surface (QS-367 E7).

    Args:
        work_dir: Worktree directory the new session should open in.
        issue: Issue number (used for tab title + script path).
        title: Issue title (used for tab title).
        next_cmd: Slash command the user types after the session opens
            (e.g. ``"/create-plan"``).  Surfaced as ``same_context`` so
            the agent can suggest the user run it in the current session
            if they prefer.
        next_prompt: Optional preload prompt for the new session.
        caller: Which script is handing off. ``"next_step"`` (a
            mid-pipeline phase handoff) additionally gets ``handoff_text``;
            ``"setup_task"`` keeps its own inline block in the setup-task
            agent, so its payload gains no unread key (QS-372 D2).
        fix_plan_path: Optional path to a review-fix plan markdown
            file. When both ``fix_plan_path`` and ``pr_number`` are
            provided, the payload gains an ``existing_session_prompt``
            field — the prompt the user can paste into an already-
            running ``qs-implement-task`` session (review-task →
            implement-task common loop). See
            ``launchers/phases.py::build_existing_session_prompt``.
        pr_number: Optional PR number for the existing-session prompt.
        lane: The task's lane (e.g. ``"feature-factory"``) or ``None``;
            selects the planning orchestrators' class, hence the pinned
            ``effortLevel`` (QS-358).

    Returns:
        A dict with ``tool``, ``agent``, ``phase_agent_pinned``,
        ``phase_model`` (the Claude model the GUI user must pick — QS-367
        E7), ``same_context``, ``new_context``, optionally
        ``existing_session_prompt``, for ``caller == "next_step"`` a
        ``handoff_text`` (the ready-to-print user-facing handoff block —
        see ``_handoff_text``, QS-372), and (on macOS with PyCharm installed)
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
    agent = resolve_agent_for_next_cmd(next_cmd)
    # Best-effort CLI floor check (QS-367 S4): warn to stderr if the local
    # ``claude`` predates the build ``claude-opus-5-5`` needs. Never blocks
    # or alters the payload. Runs AFTER ``resolve_agent_for_next_cmd`` so an
    # unknown phase raises first and never spawns ``claude`` (QS-367 N2).
    _warn_if_cli_below_floor()
    # ``agent`` is a PHASE_TO_AGENT value here (an unknown phase raised
    # above), and every one has a policy row (tests/qs/test_models.py).
    model_class = models.resolve(lane, agent)
    effort = models.effort_for(model_class)
    # QS-367 N7: resolve the Claude model BEFORE the settings write. A class
    # lacking a Claude row would otherwise leave the pin written and then
    # raise a bare ``KeyError`` from the payload dict below. Unreachable while
    # ``test_harness_rows_complete`` holds — defence in depth, and it reads
    # in the same place as ``effort``.
    phase_model = models.model_for("claude", model_class)
    # Side effect (QS-311): pin the phase agent into the worktree's local
    # settings so a GUI session there boots as this orchestrator. Guarded
    # and best-effort — see ``_write_phase_agent``. The result is surfaced
    # as ``phase_agent_pinned``: the GUI handoff blocks must not assert a
    # pin that is deterministically absent on ``--no-worktree`` and
    # silently absent on any write failure.
    pinned = _write_phase_agent(work_dir, agent, effort)
    new_context = _claude_command(
        work_dir, issue, title, agent=agent, next_prompt=next_prompt,
    )

    payload: dict = {
        "tool": "claude-code",
        "agent": agent,
        "phase_agent_pinned": pinned,
        # QS-367 E7: the Claude GUI's model picker decides the main
        # session's model — frontmatter and a settings ``model`` key both
        # lose to it — so the handoff names the model the user must pick.
        "phase_model": phase_model,
        "same_context": next_cmd,
        "new_context": new_context,
    }

    existing_prompt = build_existing_session_prompt(
        work_dir, fix_plan_path, pr_number,
    )
    if existing_prompt is not None:
        payload["existing_session_prompt"] = existing_prompt

    if caller == "next_step":
        payload["handoff_text"] = _handoff_text(
            agent=agent,
            new_context=new_context,
            work_dir=work_dir,
            issue=issue,
            phase_model=phase_model,
            pinned=pinned,
            existing_session_prompt=existing_prompt,
        )

    pycharm_bin = _pycharm_bin()
    if pycharm_bin:
        payload["pycharm_context"] = _pycharm_clipboard_command(
            work_dir, issue, claude_cmd=new_context, pycharm_bin=pycharm_bin,
        )
        payload["pycharm_applescript_context"] = _pycharm_applescript_command(
            work_dir, issue, claude_cmd=new_context, pycharm_bin=pycharm_bin,
        )

    return payload
