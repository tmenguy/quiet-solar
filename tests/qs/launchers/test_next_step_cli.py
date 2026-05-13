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


# --------------------------------------------------------------------------- #
# Free-form harness (codex) must NOT be regressed by the strict
# claude/cursor validation. The codex launcher carries no agent mapping
# today, so next_step.py must let it pass any --next-cmd value through
# unchanged. Regression catch for review-fix #1 + #5.
#
# OpenCode used to be in this list, but with the new static-agent
# pipeline (QS-177) opencode now resolves agents like claude/cursor —
# unknown phases raise UnknownPhaseError and emit the
# `{"error": "unknown phase", ...}` JSON contract. See
# `test_opencode_rejects_unknown_phase` and `test_opencode_happy_path`
# below for the new pins.
# --------------------------------------------------------------------------- #


def test_codex_accepts_free_form_next_cmd(tmp_path: Path) -> None:
    """Codex launcher must accept any --next-cmd string (no agent mapping)."""
    result = _run(
        [
            "--next-cmd", "anything-goes-here",
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Title",
            "--harness", "codex",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["tool"] == "codex"
    assert payload["same_context"] == "anything-goes-here"


def test_opencode_rejects_unknown_phase(tmp_path: Path) -> None:
    """OpenCode now resolves agents like claude/cursor — unknown phase → JSON error, exit 1.

    Contract change from the legacy pipeline (QS-177 Task 7.3). The
    OpenCode launcher is no longer a free-form passthrough; it enforces
    the same phase mapping as claude/cursor.
    """
    result = _run(
        [
            "--next-cmd", "bogus",
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Title",
            "--harness", "opencode",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["error"] == "unknown phase"
    assert payload["value"] == "bogus"


def test_opencode_happy_path(tmp_path: Path) -> None:
    """OpenCode happy path → JSON payload with spawn_session invocation, exit 0.

    QS-177 Task 7.4 — the new opencode launcher emits a
    ``python scripts/qs/spawn_session.py …`` ``new_context`` for
    ``caller='next_step'`` (the default when dispatched via
    ``next_step.py``).
    """
    result = _run(
        [
            "--next-cmd", "create-plan",
            "--work-dir", "/tmp/wt",
            "--issue", "42",
            "--title", "Foo",
            "--harness", "opencode",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["tool"] == "opencode"
    assert payload["agent"] == "qs-create-plan"
    assert payload["same_context"] == "create-plan"
    assert payload["new_context"].startswith("python scripts/qs/spawn_session.py")


def test_codex_passes_known_phase_through_unchanged(tmp_path: Path) -> None:
    """Even a known phase under codex stays free-form — no agent key added."""
    result = _run(
        [
            "--next-cmd", "create-plan",
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Title",
            "--harness", "codex",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["tool"] == "codex"
    # codex launcher is a stub; it doesn't resolve agents.
    assert "agent" not in payload


# --------------------------------------------------------------------------- #
# Empty / whitespace --next-cmd must be rejected for ALL harnesses, including
# codex/opencode (which previously accepted it silently and produced a
# garbled payload with same_context: ""). Review-fix #02 SF2.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "harness", ["claude-code", "cursor", "codex", "opencode"],
)
@pytest.mark.parametrize("bad_next_cmd", ["", "   ", "\t"])
def test_empty_or_whitespace_next_cmd_rejected_for_all_harnesses(
    harness: str, bad_next_cmd: str, tmp_path: Path,
) -> None:
    """Empty / whitespace-only --next-cmd → JSON error + exit 1, every harness."""
    result = _run(
        [
            "--next-cmd", bad_next_cmd,
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Title",
            "--harness", harness,
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode != 0, (
        f"--harness {harness} accepted empty --next-cmd silently (regression)"
    )
    # JSON error shape is identical to the unknown-phase case so downstream
    # parsing stays simple.
    payload = json.loads(result.stdout)
    assert payload["error"] == "empty next-cmd"
    assert payload["value"] == bad_next_cmd
