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

    AC #4 mandates exit code **1 specifically** (parity with
    claude/cursor) AND a ``known: [...]`` key in the JSON error
    payload — both pinned here (review fix #01 must-fix #3).
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
    # AC #4 — exit 1 (not just non-zero) to match the claude/cursor
    # contract; ``2`` is reserved for argparse user errors.
    assert result.returncode == 1, result.stderr
    payload = json.loads(result.stdout)
    assert payload["error"] == "unknown phase"
    assert payload["value"] == "bogus"
    # AC #4 — ``known`` list must surface so the user can see what
    # they meant to type (and downstream tooling can render hints).
    assert "known" in payload
    assert isinstance(payload["known"], list)
    assert len(payload["known"]) > 0
    assert "create-plan" in payload["known"]


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


# --------------------------------------------------------------------------- #
# Review fix plan #01 — should-fix #17: existing-session prompt for the
# review-task → implement-task common loop. When both `--fix-plan-path`
# and `--pr-number` are provided, the payload carries an
# `existing_session_prompt` field; either flag absent omits the key.
# --------------------------------------------------------------------------- #


def test_existing_session_prompt_emitted_when_fix_plan_and_pr_provided(tmp_path: Path) -> None:
    """Both flags present → payload has ``existing_session_prompt`` with #PR and rel path."""
    result = _run(
        [
            "--next-cmd", "implement-task",
            "--work-dir", "/tmp/wt",
            "--issue", "177",
            "--title", "Test",
            "--harness", "claude-code",
            "--fix-plan-path", "/tmp/wt/docs/stories/QS-177.story_review_fix_#01.md",
            "--pr-number", "179",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "existing_session_prompt" in payload
    prompt = payload["existing_session_prompt"]
    # Path is worktree-relative — no absolute prefix leakage.
    assert "docs/stories/QS-177.story_review_fix_#01.md" in prompt
    assert "/tmp/wt/" not in prompt
    # PR number is surfaced for the user.
    assert "#179" in prompt


def test_existing_session_prompt_omitted_when_flags_missing(tmp_path: Path) -> None:
    """Neither flag provided → payload has NO ``existing_session_prompt`` key."""
    result = _run(
        [
            "--next-cmd", "implement-task",
            "--work-dir", "/tmp/wt",
            "--issue", "177",
            "--title", "Test",
            "--harness", "claude-code",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "existing_session_prompt" not in payload


def test_existing_session_prompt_omitted_when_only_fix_plan_provided(tmp_path: Path) -> None:
    """``--fix-plan-path`` alone (no PR) → key omitted (back-compat preservation)."""
    result = _run(
        [
            "--next-cmd", "implement-task",
            "--work-dir", "/tmp/wt",
            "--issue", "177",
            "--title", "Test",
            "--harness", "claude-code",
            "--fix-plan-path", "/tmp/wt/docs/stories/QS-177.story_review_fix_#01.md",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "existing_session_prompt" not in payload


def test_existing_session_prompt_omitted_when_only_pr_provided(tmp_path: Path) -> None:
    """``--pr-number`` alone (no fix-plan path) → key omitted (back-compat preservation)."""
    result = _run(
        [
            "--next-cmd", "implement-task",
            "--work-dir", "/tmp/wt",
            "--issue", "177",
            "--title", "Test",
            "--harness", "claude-code",
            "--pr-number", "179",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "existing_session_prompt" not in payload


@pytest.mark.parametrize("harness", ["claude-code", "cursor", "codex", "opencode"])
def test_existing_session_prompt_emitted_for_all_harnesses(
    harness: str, tmp_path: Path,
) -> None:
    """Every launcher emits ``existing_session_prompt`` with the same prompt body."""
    result = _run(
        [
            "--next-cmd", "implement-task",
            "--work-dir", "/tmp/wt",
            "--issue", "177",
            "--title", "Test",
            "--harness", harness,
            "--fix-plan-path", "/tmp/wt/docs/stories/QS-177.story_review_fix_#01.md",
            "--pr-number", "179",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "existing_session_prompt" in payload
    prompt = payload["existing_session_prompt"]
    assert "docs/stories/QS-177.story_review_fix_#01.md" in prompt
    assert "#179" in prompt


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
