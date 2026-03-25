"""Shared utilities for QS development scripts."""

from __future__ import annotations

import json
import os
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
