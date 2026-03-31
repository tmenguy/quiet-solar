"""Shared utilities for QS development scripts."""

from __future__ import annotations

import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], *, check: bool = True, capture: bool = True, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command, return CompletedProcess."""
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=check,
        cwd=cwd,
    )


def run_gh(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    """Run a gh CLI command."""
    return run(["gh", *args], **kwargs)


def run_git(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    """Run a git command."""
    return run(["git", *args], **kwargs)


def get_main_worktree() -> Path:
    """Get the main worktree path (works from any worktree)."""
    result = run_git(["worktree", "list", "--porcelain"])
    first_line = result.stdout.strip().split("\n")[0]
    return Path(first_line.replace("worktree ", ""))


def get_repo_root() -> Path:
    """Get the git repo root of the current working directory."""
    result = run_git(["rev-parse", "--show-toplevel"])
    return Path(result.stdout.strip())


def get_current_branch() -> str:
    """Get the current branch name."""
    result = run_git(["branch", "--show-current"])
    return result.stdout.strip()


def get_issue_from_branch(branch: str | None = None) -> int | None:
    """Extract issue number from branch name QS_XX."""
    branch = branch or get_current_branch()
    if branch.startswith("QS_"):
        try:
            return int(branch[3:])
        except ValueError:
            return None
    return None


def ensure_gh_auth() -> bool:
    """Check gh CLI is authenticated."""
    result = run(["gh", "auth", "status"], check=False)
    return result.returncode == 0


def ensure_venv() -> str:
    """Return path to venv python. Fails if venv not found."""
    main = get_main_worktree()
    venv_python = main / "venv" / "bin" / "python"
    if not venv_python.exists():
        # Check local venv (might be in worktree via symlink)
        local_venv = get_repo_root() / "venv" / "bin" / "python"
        if local_venv.exists():
            return str(local_venv)
        print(json.dumps({"error": "venv not found", "detail": f"Expected at {venv_python}"}))
        sys.exit(1)
    return str(venv_python)


def output_json(data: dict) -> None:
    """Print JSON to stdout for LLM consumption."""
    print(json.dumps(data, indent=2))


def get_worktree_dir(issue_number: int) -> Path:
    """Get the worktree directory path for an issue."""
    main = get_main_worktree()
    basename = main.name
    return main.parent / f"{basename}-worktrees" / f"QS_{issue_number}"


def find_story_file(issue_number: int) -> Path | None:
    """Find a story file by GitHub issue number.

    Lookup strategy:
    1. Glob for ``*Github-#N*`` in the filename (new convention).
    2. Fallback: scan frontmatter of all ``.md`` files for ``issue: N`` (legacy files).

    Returns None if no match — never falls back to "most recent file".
    """
    root = get_repo_root()
    artifacts_dir = root / "_bmad-output" / "implementation-artifacts"
    if not artifacts_dir.exists():
        return None

    # Primary: filename contains Github-#N
    for f in artifacts_dir.glob(f"*Github-#{issue_number}*"):
        if f.suffix == ".md":
            return f

    # Fallback: scan frontmatter for issue: N
    issue_re = re.compile(rf"^\s*issue:\s*\"?{issue_number}\"?\s*$", re.MULTILINE)
    for f in sorted(artifacts_dir.glob("*.md")):
        # Read only first 15 lines (frontmatter area)
        try:
            with f.open() as fh:
                head = "".join(fh.readline() for _ in range(15))
        except OSError:
            continue
        if issue_re.search(head):
            return f

    return None


CLAUDE_LAUNCH_OPTS = "--dangerously-skip-permissions --model opus --effort max"


def detect_tool() -> str:
    """Detect which AI tool is driving the session.

    Returns "cursor" or "claude" (default).
    Checks, in order: --tool CLI flag (handled by caller), QS_TOOL env var,
    CURSOR_TRACE_ID env var (set by Cursor IDE terminals).
    """
    import os

    tool = os.environ.get("QS_TOOL", "").lower()
    if tool in ("cursor", "claude"):
        return tool
    if os.environ.get("CURSOR_TRACE_ID"):
        return "cursor"
    return "claude"


def claude_launch_command(
    work_dir: str,
    issue: int | str,
    title: str,
    *,
    prompt: str | None = None,
    tab_title: str | None = None,
) -> str:
    """Build a short launch command that delegates to a temp script.

    The full command is written to a temp .sh file so that the returned
    string is always a short ``sh /tmp/...`` one-liner safe for
    copy-paste across terminals (no line-wrap issues).

    Returns ``sh /path/to/qs_launch_<issue>.sh`` — ready to copy-paste.
    """
    tab_title = tab_title or f"QS_{issue}: {title}"
    safe_title = shlex.quote(tab_title)
    safe_dir = shlex.quote(work_dir)
    full_cmd = (
        f"printf '\\033]0;%s\\007' {safe_title} && "
        f"cd {safe_dir} && "
        f"claude {CLAUDE_LAUNCH_OPTS} --name {safe_title}"
    )
    if prompt is not None:
        full_cmd += f' {shlex.quote(prompt)}'

    script_path = Path(tempfile.gettempdir()) / f"qs_launch_{issue}.sh"
    script_path.write_text(f"#!/bin/sh\n{full_cmd}\n")
    script_path.chmod(0o755)

    return f"sh {script_path}"


def detect_pycharm() -> str | None:
    """Detect PyCharm installation on macOS.

    Returns the pycharm command/path or None if not found.
    Checks: pycharm on PATH (Toolbox), then known .app locations.
    """
    if platform.system() != "Darwin":
        return None

    # Toolbox or manual symlink on PATH
    pycharm_bin = shutil.which("pycharm")
    if pycharm_bin:
        return pycharm_bin

    # Direct .app installations
    app_candidates = [
        "/Applications/PyCharm.app",
        "/Applications/PyCharm Professional.app",
        "/Applications/PyCharm CE.app",
    ]
    for app in app_candidates:
        if Path(app).exists():
            return app

    return None


def _is_worktree(work_dir: str) -> bool:
    """Check if work_dir is a git worktree (not the main repo)."""
    try:
        main = get_main_worktree()
        return Path(work_dir).resolve() != main.resolve()
    except (subprocess.CalledProcessError, OSError):
        return False


def _pycharm_open_cmd(pycharm_bin: str, work_dir: str) -> str:
    """Build the shell command to open PyCharm on a directory."""
    safe_dir = shlex.quote(work_dir)
    if pycharm_bin.endswith(".app"):
        return f"open -na {shlex.quote(pycharm_bin)} --args {safe_dir}"
    return f"{shlex.quote(pycharm_bin)} {safe_dir}"


def pycharm_launch_command(
    work_dir: str,
    issue: int | str,
    *,
    claude_cmd: str,
    pycharm_bin: str,
) -> str:
    """Build Option C: open PyCharm + copy Claude command to clipboard.

    Writes a temp script that copies the Claude launch command to the
    clipboard via pbcopy, opens PyCharm on the worktree, and prints
    instructions for the user.

    Returns ``sh /path/to/qs_pycharm_<issue>.sh``.
    """
    safe_claude_cmd = shlex.quote(claude_cmd)
    open_cmd = _pycharm_open_cmd(pycharm_bin, work_dir)

    script_body = (
        "#!/bin/sh\n"
        f"echo {safe_claude_cmd} | pbcopy\n"
        f"{open_cmd}\n"
        'echo "PyCharm opening on worktree. Command copied to clipboard."\n'
        'echo "In PyCharm: Option+F12 (terminal) -> Cmd+V (paste) -> Enter"\n'
    )

    script_path = Path(tempfile.gettempdir()) / f"qs_pycharm_{issue}.sh"
    script_path.write_text(script_body)
    script_path.chmod(0o755)

    return f"sh {script_path}"


def pycharm_applescript_launch_command(
    work_dir: str,
    issue: int | str,
    *,
    claude_cmd: str,
    pycharm_bin: str,
) -> str:
    """Build Option D: open PyCharm + AppleScript to auto-type in terminal.

    Like Option C but attempts to automate the paste via AppleScript
    keystroke injection. Requires macOS Accessibility permissions.

    Returns ``sh /path/to/qs_pycharm_as_<issue>.sh``.
    """
    safe_claude_cmd = shlex.quote(claude_cmd)
    open_cmd = _pycharm_open_cmd(pycharm_bin, work_dir)

    # AppleScript: wait for PyCharm, open terminal, type the command
    applescript = (
        'tell application "PyCharm" to activate\n'
        "delay 3\n"
        'tell application "System Events"\n'
        "    -- Option+F12 opens the integrated terminal\n"
        "    key code 111 using {option down}\n"
        "    delay 1\n"
        f'    keystroke "{claude_cmd}"\n'
        "    keystroke return\n"
        "end tell\n"
    )
    safe_applescript = shlex.quote(applescript)

    script_body = (
        "#!/bin/sh\n"
        f"echo {safe_claude_cmd} | pbcopy\n"
        f"{open_cmd}\n"
        'echo "PyCharm opening. Attempting to auto-type command in terminal..."\n'
        'echo "(Requires Accessibility permissions for this terminal app)"\n'
        'echo "Fallback: Option+F12 -> Cmd+V -> Enter"\n'
        f"sleep 4\n"
        f"osascript -e {safe_applescript}\n"
    )

    script_path = Path(tempfile.gettempdir()) / f"qs_pycharm_as_{issue}.sh"
    script_path.write_text(script_body)
    script_path.chmod(0o755)

    return f"sh {script_path}"


def build_next_step(
    work_dir: str,
    issue: int | str,
    title: str,
    *,
    skill_prompt: str,
    tool: str | None = None,
    tab_title: str | None = None,
) -> dict:
    """Build tool-appropriate next-step instructions.

    Returns a dict with keys: same_context, new_context, tool, and
    optionally pycharm_context / pycharm_applescript_context when
    PyCharm is detected and work_dir is a worktree.
    """
    detected = tool or detect_tool()

    if detected == "cursor":
        new_context = (
            f"Open folder `{work_dir}` as a new Cursor workspace, then type:\n"
            f"  {skill_prompt}"
        )
    else:
        new_context = claude_launch_command(
            work_dir, issue, title, prompt=skill_prompt,
            tab_title=tab_title,
        )

    result: dict = {
        "same_context": skill_prompt,
        "new_context": new_context,
        "tool": detected,
    }

    # PyCharm options — only when worktree exists
    pycharm_bin = detect_pycharm()
    if pycharm_bin and _is_worktree(work_dir):
        # The Claude command that will be pasted/typed in PyCharm's terminal
        claude_cmd = claude_launch_command(
            work_dir, issue, title, prompt=skill_prompt,
            tab_title=tab_title,
        )
        result["pycharm_context"] = pycharm_launch_command(
            work_dir, issue, claude_cmd=claude_cmd, pycharm_bin=pycharm_bin,
        )
        result["pycharm_applescript_context"] = pycharm_applescript_launch_command(
            work_dir, issue, claude_cmd=claude_cmd, pycharm_bin=pycharm_bin,
        )

    return result


def detect_risk_level(changed_files: list[str]) -> list[str]:
    """Determine risk levels from changed file paths."""
    risk_map = {
        "CRITICAL": ["solver.py", "constraints.py", "charger.py", "dynamic_group.py"],
        "HIGH": ["load.py", "const.py", "ha_model/home.py", "ha_model/device.py"],
        "MEDIUM": ["ha_model/person.py", "ha_model/car.py", "ha_model/battery.py", "ha_model/solar.py", "config_flow.py"],
        "LOW": ["sensor.py", "switch.py", "number.py", "select.py", "button.py", "ui/"],
    }
    levels = set()
    for f in changed_files:
        for level, patterns in risk_map.items():
            if any(p in f for p in patterns):
                levels.add(level)
    return sorted(levels, key=["CRITICAL", "HIGH", "MEDIUM", "LOW"].index) if levels else ["LOW"]


# --- Workflow helpers (reusable across scripts) ---

_DEFAULT_STAGE_PATHS = [
    "custom_components/quiet_solar/",
    "tests/",
    "_bmad-output/",
    "_qsprocess/",
    "scripts/",
]

_EXCLUDE_PATTERNS = [
    ".DS_Store",
    "__pycache__",
    "venv/",
    "config/",
    ".idea/",
    ".vscode/",
]


def get_changed_files() -> list[str]:
    """Get files changed in this branch vs main."""
    result = run_git(["diff", "--name-only", "main...HEAD"], check=False)
    if result.returncode != 0:
        return []
    return [f for f in result.stdout.strip().split("\n") if f]


def auto_commit_and_push(
    message: str,
    *,
    paths: list[str] | None = None,
) -> dict:
    """Stage safe paths, commit if changes exist, push.

    Only stages files in known-safe paths. Never stages junk files.
    Returns {"committed": bool, "pushed": bool, "files": list}.
    """
    stage_paths = paths or _DEFAULT_STAGE_PATHS

    # Stage each path (git add is a no-op for non-existent paths)
    for p in stage_paths:
        run_git(["add", p], check=False)

    # Remove any junk that might have been staged
    for pattern in _EXCLUDE_PATTERNS:
        run_git(["reset", "HEAD", "--", f"*{pattern}*"], check=False)

    # Check what's actually staged
    result = run_git(["diff", "--cached", "--name-only"], check=False)
    files = [f for f in result.stdout.strip().split("\n") if f]

    if not files:
        return {"committed": False, "pushed": False, "files": []}

    # Commit
    commit_result = run_git(["commit", "-m", message], check=False)
    if commit_result.returncode != 0:
        return {"committed": False, "pushed": False, "files": files, "detail": commit_result.stderr.strip()}

    # Push
    push_result = run_git(["push"], check=False)
    pushed = push_result.returncode == 0

    return {"committed": True, "pushed": pushed, "files": files}


def find_pr_for_branch(branch: str) -> dict | None:
    """Find an open PR for the given branch.

    Returns {"pr_number": int, "url": str} or None.
    """
    result = run_gh(
        ["pr", "list", "--head", branch, "--json", "number,url,state"],
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        prs = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    if not prs:
        return None
    pr = prs[0]
    return {"pr_number": pr["number"], "url": pr["url"]}


def check_ci_status(pr_number: int) -> dict:
    """Check CI check status for a PR.

    Returns {"checks": list, "all_passed": bool, "pending": list, "failed": list}.
    Uses gh pr checks fields: name, state (SUCCESS|FAILURE|IN_PROGRESS|...), bucket.
    """
    result = run_gh(
        ["pr", "checks", str(pr_number), "--json", "name,state,bucket"],
        check=False,
    )
    if result.returncode != 0:
        return {
            "checks": [],
            "all_passed": False,
            "pending": [],
            "failed": [],
            "detail": result.stderr.strip(),
        }
    try:
        checks = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return {"checks": [], "all_passed": False, "pending": [], "failed": [], "detail": "invalid JSON"}

    failed = []
    pending = []
    for c in checks:
        state = c.get("state", "")
        if state == "SUCCESS":
            continue
        elif state in ("IN_PROGRESS", "QUEUED", "PENDING", "REQUESTED", "WAITING"):
            pending.append(c["name"])
        else:
            failed.append(c["name"])

    return {
        "checks": checks,
        "all_passed": len(failed) == 0 and len(pending) == 0,
        "pending": pending,
        "failed": failed,
    }


def ensure_issue_link(pr_number: int, issue_number: int) -> dict:
    """Ensure PR body contains a Fixes/Closes #N link.

    Returns {"linked": bool, "added": bool}.
    """
    result = run_gh(
        ["pr", "view", str(pr_number), "--json", "body"],
        check=False,
    )
    if result.returncode != 0:
        return {"linked": False, "added": False, "detail": result.stderr.strip()}

    try:
        body = json.loads(result.stdout).get("body", "")
    except (json.JSONDecodeError, TypeError):
        return {"linked": False, "added": False, "detail": "invalid JSON"}

    # Check for existing link
    pattern = rf"(?:Fixes|Closes|Resolves)\s+#{issue_number}\b"
    if re.search(pattern, body, re.IGNORECASE):
        return {"linked": True, "added": False}

    # Add link
    new_body = body.rstrip() + f"\n\nFixes #{issue_number}\n"
    edit_result = run_gh(
        ["pr", "edit", str(pr_number), "--body", new_body],
        check=False,
    )
    if edit_result.returncode != 0:
        return {"linked": False, "added": False, "detail": edit_result.stderr.strip()}

    return {"linked": True, "added": True}


def close_issue_if_open(issue_number: int, *, comment: str | None = None) -> dict:
    """Close a GitHub issue if it is still open.

    Returns {"closed": bool, "was_open": bool}.
    """
    result = run_gh(
        ["issue", "view", str(issue_number), "--json", "state"],
        check=False,
    )
    if result.returncode != 0:
        return {"closed": False, "was_open": False, "detail": result.stderr.strip()}

    try:
        state = json.loads(result.stdout).get("state", "")
    except (json.JSONDecodeError, TypeError):
        return {"closed": False, "was_open": False, "detail": "invalid JSON"}

    if state != "OPEN":
        return {"closed": False, "was_open": False}

    cmd = ["issue", "close", str(issue_number)]
    if comment:
        cmd.extend(["--comment", comment])
    close_result = run_gh(cmd, check=False)

    if close_result.returncode != 0:
        return {"closed": False, "was_open": True, "detail": close_result.stderr.strip()}

    return {"closed": True, "was_open": True}


def update_story_status(story_file: str, status: str) -> dict:
    """Update the Status: line in a story markdown file.

    Returns {"updated": bool}.
    """
    if not story_file:
        return {"updated": False, "detail": "no story file"}
    path = Path(story_file)
    if not path.is_file():
        return {"updated": False, "detail": "file not found"}

    content = path.read_text()
    new_content, count = re.subn(
        r"^(Status:\s*).*$",
        rf"\g<1>{status}",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if count == 0:
        return {"updated": False, "detail": "no Status: line found"}

    path.write_text(new_content)
    return {"updated": True}


def suggest_release(changed_files: list[str]) -> str:
    """Suggest whether a release is needed based on changed files.

    Returns "release" if production code changed, "no-release" otherwise.
    """
    for f in changed_files:
        if f.startswith("custom_components/"):
            return "release"
    return "no-release"
