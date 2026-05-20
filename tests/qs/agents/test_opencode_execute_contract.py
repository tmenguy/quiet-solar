"""Pin the OpenCode auto-execute contract (AC-4 / AC-5 / AC-6).

OpenCode mid-pipeline agents (create-plan, implement-task,
implement-setup-task, review-task) must instruct the agent to RUN
``new_context`` via the Bash tool and verify the binary
``status == "session_created"`` AND ``agent == "qs-<expected>"``
contract. The OpenCode setup-task agent + every Claude / Cursor agent
preserve the existing print-for-user pattern (no auto-execute prose).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
OPENCODE_DIR = REPO_ROOT / ".opencode" / "agents"
CLAUDE_DIR = REPO_ROOT / ".claude" / "agents"
CURSOR_DIR = REPO_ROOT / ".cursor" / "agents"

# Distinctive substring from the canonical auto-execute block (Task
# 3.2-3.5). Choosing a phrase that is unique to the auto-execute
# pattern AND unlikely to drift; the print-for-user block uses
# "Preferred (..." / "Fallback (..." instead.
EXECUTE_MARKER = "Run `new_context` via the Bash tool"
SUCCESS_CONTRACT_MARKER = 'status == "session_created"'
AGENT_VERIFY_MARKER = 'agent == "qs-'  # partial — phase varies

OPENCODE_EXECUTE_AGENTS = (
    "qs-create-plan",
    "qs-implement-task",
    "qs-implement-setup-task",
    "qs-review-task",
)


@pytest.mark.parametrize("name", OPENCODE_EXECUTE_AGENTS)
def test_opencode_agent_auto_executes_new_context(name: str) -> None:
    body = (OPENCODE_DIR / f"{name}.md").read_text(encoding="utf-8")
    assert EXECUTE_MARKER in body, (
        f"{name}: missing auto-execute imperative. The body must "
        f"instruct the agent to run new_context via the Bash tool "
        f"(see QS-190 AC-4)."
    )
    assert SUCCESS_CONTRACT_MARKER in body, (
        f"{name}: missing the binary success-contract check on "
        f"`status == \"session_created\"`."
    )
    assert AGENT_VERIFY_MARKER in body, (
        f"{name}: missing the `agent == \"qs-...\"` verification."
    )


def test_opencode_review_task_has_two_execute_blocks() -> None:
    """review-task has 2 next_step.py callsites — both must auto-execute.

    Counts occurrences of the EXECUTE_MARKER. Asserts ``>= 2`` so the
    test is robust to future prose elaboration; the lower bound is
    what matters.
    """
    body = (OPENCODE_DIR / "qs-review-task.md").read_text(encoding="utf-8")
    count = body.count(EXECUTE_MARKER)
    assert count >= 2, (
        f"qs-review-task.md should auto-execute both new_context "
        f"callsites (zero-findings + fix-plan); found {count} "
        f"occurrences of {EXECUTE_MARKER!r}."
    )


def test_opencode_setup_task_preserves_print_for_user() -> None:
    """Cross-workspace setup-task keeps the print-for-user pattern (AC-5)."""
    body = (OPENCODE_DIR / "qs-setup-task.md").read_text(encoding="utf-8")
    assert EXECUTE_MARKER not in body, (
        "qs-setup-task.md should NOT auto-execute new_context — "
        "setup-task crosses OpenCode workspaces and emits a CLI-form "
        "launcher (see QS-190 AC-5)."
    )


@pytest.mark.parametrize("agent_file", sorted(CLAUDE_DIR.glob("qs-*.md")))
def test_claude_agents_preserve_print_for_user(agent_file: Path) -> None:
    """Every Claude phase orchestrator stays print-for-user (AC-6)."""
    body = agent_file.read_text(encoding="utf-8")
    assert EXECUTE_MARKER not in body, (
        f"{agent_file.relative_to(REPO_ROOT)}: Claude agents must "
        f"preserve the existing print-for-user pattern. Found the "
        f"auto-execute marker — that's OpenCode-only (QS-190 AC-6)."
    )


@pytest.mark.parametrize("agent_file", sorted(CURSOR_DIR.glob("qs-*.md")))
def test_cursor_agents_preserve_print_for_user(agent_file: Path) -> None:
    """Every Cursor phase orchestrator stays print-for-user (AC-6)."""
    body = agent_file.read_text(encoding="utf-8")
    assert EXECUTE_MARKER not in body, (
        f"{agent_file.relative_to(REPO_ROOT)}: Cursor agents must "
        f"preserve the existing print-for-user pattern (QS-190 AC-6)."
    )
