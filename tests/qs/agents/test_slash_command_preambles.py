"""Lock the AC-5 contract: every slash-command file in ``.claude/commands/``
ships the "Preferred entry / degraded fallback" preamble.

This is the regression catch for QS-175 review-fix #8 — AC-5 is enforced
only by manual review without this test. Each of the 6 slash-command
files (release.md is intentionally excluded — see QS-175 OUT OF SCOPE)
must contain the canonical preamble markers explaining the fallback
nature of the slash-command path.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def _normalize_whitespace(body: str) -> str:
    """Collapse newlines + markdown blockquote markers (``> ``) to single spaces.

    Markdown wraps long lines, so a phrase like ``Claude Desktop`` may appear
    as ``Claude\\n> Desktop`` in the source. We normalize before substring
    checks so the tests are robust to line wrap.
    """
    # Strip ``> `` blockquote prefixes (at line start) and collapse all
    # whitespace runs to a single space.
    stripped = re.sub(r"(?m)^\s*>\s?", "", body)
    return re.sub(r"\s+", " ", stripped)

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMANDS_DIR = REPO_ROOT / ".claude" / "commands"

# Exactly the 6 slash-command files listed in QS-175 Task 7. release.md is
# excluded — that's a manual command on the main checkout, not part of the
# orchestrator chain.
_SLASH_COMMAND_FILES = [
    "setup-task.md",
    "create-plan.md",
    "implement-task.md",
    "implement-setup-task.md",
    "review-task.md",
    "finish-task.md",
]


@pytest.mark.parametrize("filename", _SLASH_COMMAND_FILES)
def test_slash_command_has_preferred_entry_marker(filename: str) -> None:
    """The preamble names ``claude --agent qs-<phase>`` as the preferred entry."""
    body = _normalize_whitespace((COMMANDS_DIR / filename).read_text())
    assert "Preferred entry" in body, (
        f"{filename}: missing 'Preferred entry' label — AC-5 requires every "
        f"slash command to label the launcher path as preferred."
    )
    assert "claude --agent qs-" in body, (
        f"{filename}: missing 'claude --agent qs-...' reference — the preamble "
        f"must name the launcher invocation explicitly."
    )


@pytest.mark.parametrize("filename", _SLASH_COMMAND_FILES)
def test_slash_command_has_degraded_fallback_marker(filename: str) -> None:
    """The preamble explicitly labels itself as the degraded fallback."""
    body = _normalize_whitespace((COMMANDS_DIR / filename).read_text())
    assert "degraded fallback" in body, (
        f"{filename}: missing 'degraded fallback' label — AC-5 requires every "
        f"slash command to be flagged as the broken-by-design fallback UX."
    )


@pytest.mark.parametrize("filename", _SLASH_COMMAND_FILES)
def test_slash_command_mentions_claude_desktop(filename: str) -> None:
    """Each preamble names Claude Desktop as the reason to keep the fallback."""
    body = _normalize_whitespace((COMMANDS_DIR / filename).read_text())
    assert "Claude Desktop" in body, (
        f"{filename}: missing 'Claude Desktop' rationale — AC-5 / AC-8 "
        f"require honest naming of the environment that needs this fallback."
    )


@pytest.mark.parametrize("filename", _SLASH_COMMAND_FILES)
def test_slash_command_mentions_qs_175(filename: str) -> None:
    """Each preamble references the QS-175 mitigation framing."""
    body = _normalize_whitespace((COMMANDS_DIR / filename).read_text())
    assert "QS-175" in body, (
        f"{filename}: missing 'QS-175' reference — the framing 'broken-by-"
        f"design UX QS-175 mitigates' is what makes the preamble actionable; "
        f"don't drop it on edit."
    )
