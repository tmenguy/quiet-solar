"""Tests for ``scripts/qs/next_step.py`` invoked as a subprocess.

``next_step.py`` is stdlib-only and must work from any CWD (AC-9). The
phase-name → agent-name resolution is a static dict — no filesystem
scan, no ``Path.cwd()`` — so running from ``/tmp`` resolves the same as
running from a worktree.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_QS = Path(__file__).resolve().parents[3] / "scripts" / "qs"
NEXT_STEP_PY = SCRIPTS_QS / "next_step.py"


def _run(args: list[str], *, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``next_step.py`` with the given args."""
    return subprocess.run(
        [sys.executable, str(NEXT_STEP_PY), *args],
        capture_output=True,
        text=True,
        cwd=cwd or str(SCRIPTS_QS),
    )


def test_valid_phase_emits_payload_and_exits_zero(tmp_path: Path) -> None:
    """Valid phase → JSON payload on stdout, exit 0. CWD-independent."""
    result = _run(
        [
            "--next-cmd", "create-plan",
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Fix bug",
            "--harness", "claude-code",
        ],
        cwd=str(tmp_path),  # outside the repo — AC-9
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["harness"] == "claude-code"
    assert payload["agent"] == "qs-create-plan"
    assert payload["same_context"] == "create-plan"
    assert payload["new_context"].startswith("sh ")


def test_slash_form_accepted_for_back_compat(tmp_path: Path) -> None:
    """Slash form continues to work — old callers pass ``/create-plan``."""
    result = _run(
        [
            "--next-cmd", "/create-plan",
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Fix bug",
            "--harness", "claude-code",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["agent"] == "qs-create-plan"


def test_unknown_phase_emits_error_json_and_exits_nonzero(tmp_path: Path) -> None:
    """Unknown phase → JSON error on stdout, non-zero exit."""
    result = _run(
        [
            "--next-cmd", "bogus-phase",
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Fix bug",
            "--harness", "claude-code",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["error"] == "unknown phase"
    assert payload["value"] == "bogus-phase"
    assert "create-plan" in payload["known"]
    assert "release" in payload["known"]


def test_cursor_harness_branch(tmp_path: Path) -> None:
    """``--harness cursor --next-cmd create-plan`` routes through cursor.py."""
    result = _run(
        [
            "--next-cmd", "create-plan",
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Fix bug",
            "--harness", "cursor",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["tool"] == "cursor"
    assert payload["harness"] == "cursor"
    assert payload["agent"] == "qs-create-plan"


@pytest.mark.parametrize("phase", [
    "setup-task",
    "create-plan",
    "implement-task",
    "implement-setup-task",
    "review-task",
    "finish-task",
    "release",
])
def test_every_known_phase_resolves(phase: str, tmp_path: Path) -> None:
    """Every known phase resolves end-to-end via the CLI."""
    result = _run(
        [
            "--next-cmd", phase,
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Title",
            "--harness", "claude-code",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["agent"] == f"qs-{phase}"
