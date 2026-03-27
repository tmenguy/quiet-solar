"""Shared utilities for QS development scripts."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
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


def find_story_file(story_key: str | None = None) -> Path | None:
    """Find a story file by key (e.g., '3.2') or return the most recent one."""
    root = get_repo_root()
    artifacts_dir = root / "_bmad-output" / "implementation-artifacts"
    if not artifacts_dir.exists():
        return None

    if story_key:
        # Normalize: "3.2" -> "3-2"
        normalized = story_key.replace(".", "-")
        for f in artifacts_dir.glob(f"{normalized}-*.md"):
            return f
        # Also try direct match
        for f in artifacts_dir.glob(f"*{normalized}*.md"):
            return f
        return None

    # Return most recently modified story file
    story_files = sorted(artifacts_dir.glob("*.md"), key=os.path.getmtime, reverse=True)
    return story_files[0] if story_files else None


CLAUDE_LAUNCH_OPTS = "--dangerously-skip-permissions --model opus --effort max"


def claude_launch_command(work_dir: str, issue: int, title: str) -> str:
    """Build the full claude launch command with terminal title and standard options."""
    tab_title = f"QS_{issue}: {title}"
    safe_title = shlex.quote(tab_title)
    safe_dir = shlex.quote(work_dir)
    return (
        f'printf "\\e]0;%s\\a" {safe_title} && '
        f'cd {safe_dir} && '
        f'claude {CLAUDE_LAUNCH_OPTS} --name {safe_title}'
    )


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
    "custom_components/",
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
    """
    result = run_gh(
        ["pr", "checks", str(pr_number), "--json", "name,state,conclusion"],
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
        conclusion = c.get("conclusion", "")
        if state != "COMPLETED":
            pending.append(c["name"])
        elif conclusion != "SUCCESS":
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
    path = Path(story_file)
    if not path.exists():
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
