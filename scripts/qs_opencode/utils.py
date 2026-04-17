"""Self-contained utilities for ``scripts/qs_opencode/``.

Mirrors patterns from ``scripts/qs/utils.py`` without importing from it so
the OpenCode integration can evolve independently of the Claude/Cursor one.
"""

from __future__ import annotations

import json
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Launch options
# ---------------------------------------------------------------------------

#: Extra ``opencode`` CLI flags (beyond ``--agent`` / ``--prompt``) used when
#: opening a new session on a worktree. Confirmed via ``opencode --help``:
#:   opencode [project] --agent <name> --prompt <text> [other flags...]
#: Leave empty by default; users can set e.g. ``"--model opus"``.
OPENCODE_LAUNCH_OPTS = ""


def output_json(data: dict) -> None:
    """Print ``data`` as JSON to stdout (compact, single-line safe for logs)."""
    json.dump(data, sys.stdout)
    sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# PyCharm detection (parity with scripts/qs/utils.py)
# ---------------------------------------------------------------------------


def detect_pycharm() -> str | None:
    """Return the PyCharm command or ``.app`` path on macOS, else ``None``."""
    if platform.system() != "Darwin":
        return None

    pycharm_bin = shutil.which("pycharm")
    if pycharm_bin:
        return pycharm_bin

    for app in (
        "/Applications/PyCharm.app",
        "/Applications/PyCharm Professional.app",
        "/Applications/PyCharm CE.app",
    ):
        if Path(app).exists():
            return app

    return None


def _pycharm_open_cmd(pycharm_bin: str, work_dir: str) -> str:
    """Build the shell command to open PyCharm on a directory."""
    safe_dir = shlex.quote(work_dir)
    if pycharm_bin.endswith(".app"):
        return f"open -na {shlex.quote(pycharm_bin)} --args {safe_dir}"
    return f"{shlex.quote(pycharm_bin)} {safe_dir}"


# ---------------------------------------------------------------------------
# Git worktree helpers
# ---------------------------------------------------------------------------


def get_main_worktree() -> Path:
    """Return the primary (non-worktree) git directory for this repo."""
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    # First ``worktree`` line is the main checkout.
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            return Path(line.split(" ", 1)[1])
    raise RuntimeError("No git worktrees found")


def is_worktree(work_dir: str) -> bool:
    """Return ``True`` when ``work_dir`` is a worktree (not the main repo)."""
    try:
        return Path(work_dir).resolve() != get_main_worktree().resolve()
    except (subprocess.CalledProcessError, OSError, RuntimeError):
        return False


# ---------------------------------------------------------------------------
# Launcher builders — OpenCode-specific
# ---------------------------------------------------------------------------


def opencode_launch_command(
    work_dir: str,
    issue: int | str,
    title: str,
    *,
    agent: str | None = None,
    preload_command: str | None = None,
    tab_title: str | None = None,
) -> str:
    """Build a short launch command that opens a new OpenCode on ``work_dir``.

    The full command is written to a temp ``.sh`` file so the returned
    string is a short ``sh /tmp/...`` one-liner safe for copy-paste.

    OpenCode CLI supports ``--agent <name>`` and ``--prompt <text>`` as
    top-level options (confirmed via ``opencode --help``). We use:

    - ``--agent`` to pre-activate the rendered per-task agent
    - ``--prompt`` to send the initial kickoff message

    A printed banner also shows the intended prompt so the user can paste
    it manually if the flags ever change.
    """
    tab_title = tab_title or f"QS_{issue}: {title}"
    safe_title = shlex.quote(tab_title)
    safe_dir = shlex.quote(work_dir)

    opencode_parts = ["opencode", safe_dir]
    if OPENCODE_LAUNCH_OPTS:
        opencode_parts.append(OPENCODE_LAUNCH_OPTS)
    if agent:
        opencode_parts.extend(["--agent", shlex.quote(agent)])
    if preload_command:
        opencode_parts.extend(["--prompt", shlex.quote(preload_command)])

    parts = [
        f"printf '\\033]0;%s\\007' {safe_title}",
        " ".join(opencode_parts),
    ]
    full_cmd = " && ".join(parts)

    script_path = Path(tempfile.gettempdir()) / f"qs_oc_launch_{issue}.sh"
    banner_lines = []
    if agent:
        banner_lines.append(
            f"echo '── Activating agent: {agent} ──'",
        )
    if preload_command:
        banner_lines.append(
            "echo '── Initial prompt (paste manually if preload fails) ──'",
        )
        banner_lines.append(f"echo {shlex.quote(preload_command)}")
    banner = (" && ".join(banner_lines) + " && ") if banner_lines else ""
    script_path.write_text(f"#!/bin/sh\n{banner}{full_cmd}\n")
    script_path.chmod(0o755)
    return f"sh {script_path}"


def pycharm_launch_command(
    work_dir: str,
    issue: int | str,
    *,
    opencode_cmd: str,
    pycharm_bin: str,
) -> str:
    """Option C: open PyCharm + copy OpenCode launch command to clipboard."""
    safe_cmd = shlex.quote(opencode_cmd)
    open_cmd = _pycharm_open_cmd(pycharm_bin, work_dir)

    script_body = (
        "#!/bin/sh\n"
        f"echo {safe_cmd} | pbcopy\n"
        f"{open_cmd}\n"
        'echo "PyCharm opening on worktree. OpenCode command copied to clipboard."\n'
        'echo "In PyCharm: Option+F12 (terminal) -> Cmd+V (paste) -> Enter"\n'
    )
    script_path = Path(tempfile.gettempdir()) / f"qs_oc_pycharm_{issue}.sh"
    script_path.write_text(script_body)
    script_path.chmod(0o755)
    return f"sh {script_path}"


def pycharm_applescript_launch_command(
    work_dir: str,
    issue: int | str,
    *,
    opencode_cmd: str,
    pycharm_bin: str,
) -> str:
    """Option D: open PyCharm + AppleScript auto-type in terminal (macOS)."""
    safe_cmd = shlex.quote(opencode_cmd)
    open_cmd = _pycharm_open_cmd(pycharm_bin, work_dir)

    applescript = (
        'tell application "PyCharm" to activate\n'
        "delay 3\n"
        'tell application "System Events"\n'
        "    key code 111 using {option down}\n"  # Opt+F12 opens terminal
        "    delay 1\n"
        f'    keystroke "{opencode_cmd}"\n'
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
    script_path = Path(tempfile.gettempdir()) / f"qs_oc_pycharm_as_{issue}.sh"
    script_path.write_text(script_body)
    script_path.chmod(0o755)
    return f"sh {script_path}"


def build_launcher_payload(
    work_dir: str,
    issue: int | str,
    title: str,
    *,
    agent: str | None = None,
    preload_command: str | None,
    same_context_text: str,
) -> dict:
    """Build the JSON payload consumed by ``qs-setup-task`` for its final message."""
    new_context = opencode_launch_command(
        work_dir, issue, title,
        agent=agent,
        preload_command=preload_command,
    )

    payload: dict = {
        "tool": "opencode",
        "same_context": same_context_text,
        "new_context": new_context,
    }

    pycharm_bin = detect_pycharm()
    if pycharm_bin and is_worktree(work_dir):
        payload["pycharm_context"] = pycharm_launch_command(
            work_dir, issue, opencode_cmd=new_context, pycharm_bin=pycharm_bin,
        )
        payload["pycharm_applescript_context"] = pycharm_applescript_launch_command(
            work_dir, issue, opencode_cmd=new_context, pycharm_bin=pycharm_bin,
        )

    return payload
