"""Shared utilities for QS development scripts.

Harness-specific launcher code lives in :mod:`scripts.qs.launchers`; this
module contains only the genuinely shared helpers (git, gh, story-file
discovery, commit/PR plumbing).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command, returning the completed process."""
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=check,
        cwd=cwd,
    )


def run_gh(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    """Run a ``gh`` CLI command."""
    return run(["gh", *args], **kwargs)


def run_git(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    """Run a ``git`` command."""
    return run(["git", *args], **kwargs)


# ---------------------------------------------------------------------------
# Repo / worktree discovery
# ---------------------------------------------------------------------------


def get_main_worktree() -> Path:
    """Return the primary (non-worktree) git directory for this repo."""
    result = run_git(["worktree", "list", "--porcelain"])
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            return Path(line.split(" ", 1)[1])
    raise RuntimeError("No git worktrees found")


def get_repo_root() -> Path:
    """Return the git repo root of the current working directory."""
    result = run_git(["rev-parse", "--show-toplevel"])
    return Path(result.stdout.strip())


def get_current_branch() -> str:
    """Return the current branch name."""
    result = run_git(["branch", "--show-current"])
    return result.stdout.strip()


def get_issue_from_branch(branch: str | None = None) -> int | None:
    """Parse the issue number out of ``QS_<N>`` branch names."""
    branch = branch or get_current_branch()
    if branch.startswith("QS_"):
        try:
            return int(branch[3:])
        except ValueError:
            return None
    return None


def get_worktree_dir(issue_number: int) -> Path:
    """Return the conventional worktree path for an issue."""
    main = get_main_worktree()
    basename = main.name
    return main.parent / f"{basename}-worktrees" / f"QS_{issue_number}"


def is_worktree(work_dir: str | Path) -> bool:
    """Return ``True`` if ``work_dir`` is a worktree (not the main repo)."""
    try:
        return Path(work_dir).resolve() != get_main_worktree().resolve()
    except (subprocess.CalledProcessError, OSError, RuntimeError):
        return False


# ---------------------------------------------------------------------------
# Auth / venv preflight
# ---------------------------------------------------------------------------


def ensure_gh_auth() -> bool:
    """Return ``True`` if the ``gh`` CLI is authenticated."""
    result = run(["gh", "auth", "status"], check=False)
    return result.returncode == 0


def ensure_venv() -> str:
    """Return the path to the project venv's python; fail if missing."""
    main = get_main_worktree()
    venv_python = main / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    local_venv = get_repo_root() / "venv" / "bin" / "python"
    if local_venv.exists():
        return str(local_venv)
    print(json.dumps({"error": "venv not found", "detail": f"Expected at {venv_python}"}))
    sys.exit(1)


# ---------------------------------------------------------------------------
# JSON output (LLM consumption)
# ---------------------------------------------------------------------------


def output_json(data: dict) -> None:
    """Print ``data`` as pretty JSON to stdout."""
    print(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Story file discovery
# ---------------------------------------------------------------------------


# Stories live under docs/stories/. Both the new static-agent pipeline
# and the legacy OpenCode pipeline (whose templates were updated to
# point here) read from this single canonical location.
STORY_DIR = "docs/stories"


def find_story_file(issue: int) -> Path | None:
    """Return the story file for an issue, or ``None`` if not found."""
    root = get_repo_root()
    direct = root / STORY_DIR / f"QS-{issue}.story.md"
    if direct.is_file():
        return direct
    return None


def find_latest_review_fix(issue: int) -> Path | None:
    """Return the most recent ``QS-<N>.story_review_fix_#NN.md`` (if any)."""
    root = get_repo_root()
    pattern = f"QS-{issue}.story_review_fix_*.md"
    matches = sorted((root / STORY_DIR).glob(pattern))
    return matches[-1] if matches else None


def next_review_fix_path(issue: int) -> Path:
    """Compute the next auto-incremented review-fix file path."""
    root = get_repo_root()
    counter = 1
    existing = sorted((root / STORY_DIR).glob(f"QS-{issue}.story_review_fix_#*.md"))
    if existing:
        last = existing[-1]
        match = re.search(r"_review_fix_#(\d+)\.md$", last.name)
        if match:
            counter = int(match.group(1)) + 1
    return root / STORY_DIR / f"QS-{issue}.story_review_fix_#{counter:02d}.md"


# ---------------------------------------------------------------------------
# Risk detection
# ---------------------------------------------------------------------------


_RISK_MAP: dict[str, list[str]] = {
    "CRITICAL": ["solver.py", "constraints.py", "charger.py", "dynamic_group.py"],
    "HIGH": ["load.py", "const.py", "ha_model/home.py", "ha_model/device.py"],
    "MEDIUM": [
        "ha_model/person.py",
        "ha_model/car.py",
        "ha_model/battery.py",
        "ha_model/solar.py",
        "config_flow.py",
    ],
    "LOW": [
        "sensor.py",
        "switch.py",
        "number.py",
        "select.py",
        "button.py",
        "ui/",
    ],
}


def detect_risk_level(changed_files: list[str]) -> list[str]:
    """Determine risk levels from changed file paths."""
    levels: set[str] = set()
    for f in changed_files:
        for level, patterns in _RISK_MAP.items():
            if any(p in f for p in patterns):
                levels.add(level)
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    return sorted(levels, key=order.index) if levels else ["LOW"]


# ---------------------------------------------------------------------------
# Commit + PR plumbing
# ---------------------------------------------------------------------------


_DEFAULT_STAGE_PATHS = [
    "custom_components/quiet_solar/",
    "tests/",
    "scripts/",
    "docs/stories/",
    "docs/",
    ".claude/",
    ".cursor/",
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
    """Return files changed in this branch vs ``main``."""
    result = run_git(["diff", "--name-only", "main...HEAD"], check=False)
    if result.returncode != 0:
        return []
    return [f for f in result.stdout.strip().split("\n") if f]


def auto_commit_and_push(
    message: str,
    *,
    paths: list[str] | None = None,
) -> dict:
    """Stage known-safe paths, commit if changes exist, push."""
    stage_paths = paths or _DEFAULT_STAGE_PATHS

    for p in stage_paths:
        run_git(["add", p], check=False)

    for pattern in _EXCLUDE_PATTERNS:
        run_git(["reset", "HEAD", "--", f"*{pattern}*"], check=False)

    result = run_git(["diff", "--cached", "--name-only"], check=False)
    files = [f for f in result.stdout.strip().split("\n") if f]

    if not files:
        return {"committed": False, "pushed": False, "files": []}

    commit_result = run_git(["commit", "-m", message], check=False)
    if commit_result.returncode != 0:
        return {
            "committed": False,
            "pushed": False,
            "files": files,
            "detail": commit_result.stderr.strip(),
        }

    push_result = run_git(["push"], check=False)
    pushed = push_result.returncode == 0
    return {"committed": True, "pushed": pushed, "files": files}


def find_pr_for_branch(branch: str) -> dict | None:
    """Return ``{"pr_number", "url"}`` for the open PR on ``branch``, else None."""
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
    return {"pr_number": prs[0]["number"], "url": prs[0]["url"]}


def check_ci_status(pr_number: int) -> dict:
    """Return CI check status for a PR."""
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
        return {
            "checks": [],
            "all_passed": False,
            "pending": [],
            "failed": [],
            "detail": "invalid JSON",
        }

    failed: list[str] = []
    pending: list[str] = []
    for c in checks:
        state = c.get("state", "")
        if state == "SUCCESS":
            continue
        if state in ("IN_PROGRESS", "QUEUED", "PENDING", "REQUESTED", "WAITING"):
            pending.append(c["name"])
        else:
            failed.append(c["name"])

    return {
        "checks": checks,
        "all_passed": not failed and not pending,
        "pending": pending,
        "failed": failed,
    }


def ensure_issue_link(pr_number: int, issue_number: int) -> dict:
    """Ensure PR body has a ``Fixes #<N>`` link; add it if missing."""
    result = run_gh(["pr", "view", str(pr_number), "--json", "body"], check=False)
    if result.returncode != 0:
        return {"linked": False, "added": False, "detail": result.stderr.strip()}
    try:
        body = json.loads(result.stdout).get("body", "")
    except (json.JSONDecodeError, TypeError):
        return {"linked": False, "added": False, "detail": "invalid JSON"}

    pattern = rf"(?:Fixes|Closes|Resolves)\s+#{issue_number}\b"
    if re.search(pattern, body, re.IGNORECASE):
        return {"linked": True, "added": False}

    new_body = body.rstrip() + f"\n\nFixes #{issue_number}\n"
    edit_result = run_gh(["pr", "edit", str(pr_number), "--body", new_body], check=False)
    if edit_result.returncode != 0:
        return {"linked": False, "added": False, "detail": edit_result.stderr.strip()}
    return {"linked": True, "added": True}


def close_issue_if_open(issue_number: int, *, comment: str | None = None) -> dict:
    """Close ``issue_number`` if it is still open."""
    result = run_gh(["issue", "view", str(issue_number), "--json", "state"], check=False)
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


def update_story_status(story_file: str | Path, status: str) -> dict:
    """Update the ``Status:`` line in a story markdown file."""
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
    """Return ``"release"`` if production code changed, else ``"no-release"``."""
    for f in changed_files:
        if f.startswith("custom_components/"):
            return "release"
    return "no-release"
