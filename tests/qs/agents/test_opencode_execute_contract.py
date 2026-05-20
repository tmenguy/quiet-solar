"""Pin the OpenCode auto-execute contract (AC-4 / AC-5 / AC-6).

OpenCode mid-pipeline agents (create-plan, implement-task,
implement-setup-task, review-task) must instruct the agent to RUN
``new_context`` via the Bash tool and verify the binary
``status == "session_created"`` AND the resolved agent name contract.
The OpenCode setup-task agent + every Claude / Cursor agent preserve
the existing print-for-user pattern (no auto-execute prose).

Review fix #01 hardens the prose:

- **S7** — the agent-name comparison is in **prose** ("equals
  ``qs-`` followed by the phase name passed to ``--next-cmd``"); the
  ``agent == "qs-{{NEXT_PHASE}}"`` literal previously relied on the
  LLM substituting the template, and a literal comparison would
  misfire.
- **S8** — each block tells the agent to STOP on a ``next_step.py``
  error payload (key ``error`` in the JSON) BEFORE attempting to run
  ``new_context``.
- **S11** — each block tells the agent to handle a Bash-tool failure
  that produces no JSON output (e.g. permission denied).
- **N10** — the bash block above is prefaced by a substitution
  reminder so the agent doesn't run it with unresolved Jinja
  placeholders.
- **N3** — emoji-free success-report blocks (project rule: "only use
  emojis if the user explicitly requests it").
- **S13** — the AC-5 rationale paragraph in the OpenCode setup-task
  agent file is pinned by a distinctive substring.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
OPENCODE_DIR = REPO_ROOT / ".opencode" / "agents"
CLAUDE_DIR = REPO_ROOT / ".claude" / "agents"
CURSOR_DIR = REPO_ROOT / ".cursor" / "agents"

# Distinctive substring from the canonical auto-execute block (QS-190
# Task 3.2-3.5). Choosing a phrase that is unique to the auto-execute
# pattern AND unlikely to drift; the print-for-user block uses
# "Preferred (..." / "Fallback (..." instead.
EXECUTE_MARKER = "Run `new_context` via the Bash tool"
SUCCESS_CONTRACT_MARKER = 'status == "session_created"'

# Review fix #01 prose pins. ``AGENT_VERIFY_PROSE_RE`` accepts any
# whitespace (including a wrapped line) between tokens because the
# canonical prose naturally wraps inside markdown bullets.
AGENT_VERIFY_PROSE_RE = re.compile(
    r"equals\s+`qs-`\s+followed\s+by\s+the\s+phase\s+name",
)
NEXT_STEP_ERROR_GUARD = "If the `next_step.py` JSON contains an `error` key"
BASH_FAILURE_GUARD = "If the Bash tool returns an error before producing any JSON output"
SUBSTITUTION_REMINDER = "**Before running** — substitute"

OPENCODE_EXECUTE_AGENTS = (
    "qs-create-plan",
    "qs-implement-task",
    "qs-implement-setup-task",
    "qs-review-task",
)


@pytest.mark.parametrize("name", OPENCODE_EXECUTE_AGENTS)
def test_opencode_agent_auto_executes_new_context(name: str) -> None:
    """Core auto-execute imperative + success contract markers (AC-4)."""
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


@pytest.mark.parametrize("name", OPENCODE_EXECUTE_AGENTS)
def test_opencode_agent_uses_prose_for_agent_name_check(name: str) -> None:
    """S7: the agent-name half of the contract is in prose, not template.

    The QS-190 v1 canonical block compared ``agent == "qs-{{phase}}"``
    with a literal Jinja-style placeholder. An LLM that compared
    literally would treat every successful spawn as a failure. The
    prose form ("equals ``qs-`` followed by the phase name passed to
    ``--next-cmd``") leaves no ambiguity.
    """
    body = (OPENCODE_DIR / f"{name}.md").read_text(encoding="utf-8")
    assert AGENT_VERIFY_PROSE_RE.search(body), (
        f"{name}: agent-name verification must be in prose, not a "
        f"literal `agent == \"qs-{{...}}\"` template (review-fix #01 S7). "
        f"Look for `equals \\`qs-\\` followed by the phase name` "
        f"(line wraps allowed)."
    )
    # Inverse: a literal ``"qs-{{NEXT_PHASE}}"`` or
    # ``"qs-{{next_implement}}"`` template (i.e. the dangerous form
    # used in the QS-190 v1 comparison) must NOT appear. The bash
    # argument ``--next-cmd "{{NEXT_PHASE}}"`` is fine — that's the
    # legitimate placeholder for the user to substitute.
    assert '"qs-{{NEXT_PHASE}}"' not in body, (
        f"{name}: still contains the literal "
        f"`\"qs-{{NEXT_PHASE}}\"` comparison string; review-fix #01 "
        f"S7 wants this replaced with prose."
    )
    assert '"qs-{{next_implement}}"' not in body, (
        f"{name}: still contains the literal "
        f"`\"qs-{{next_implement}}\"` comparison string; review-fix "
        f"#01 S7 wants this replaced with prose."
    )


@pytest.mark.parametrize("name", OPENCODE_EXECUTE_AGENTS)
def test_opencode_agent_handles_next_step_error_payload(name: str) -> None:
    """S8: each auto-execute block tells the agent how to STOP on
    a ``next_step.py`` error JSON before running ``new_context``.
    """
    body = (OPENCODE_DIR / f"{name}.md").read_text(encoding="utf-8")
    assert NEXT_STEP_ERROR_GUARD in body, (
        f"{name}: missing the next_step.py error-key guard. The "
        f"prose must say: '{NEXT_STEP_ERROR_GUARD}' (review-fix #01 S8)."
    )


@pytest.mark.parametrize("name", OPENCODE_EXECUTE_AGENTS)
def test_opencode_agent_handles_bash_tool_failure(name: str) -> None:
    """S11: each auto-execute block tells the agent how to handle a
    Bash-tool failure that produces no JSON output (e.g. permission
    denied, missing interpreter).
    """
    body = (OPENCODE_DIR / f"{name}.md").read_text(encoding="utf-8")
    assert BASH_FAILURE_GUARD in body, (
        f"{name}: missing the Bash-tool failure guard. The prose must "
        f"say: '{BASH_FAILURE_GUARD}' (review-fix #01 S11)."
    )


@pytest.mark.parametrize("name", OPENCODE_EXECUTE_AGENTS)
def test_opencode_agent_substitution_reminder(name: str) -> None:
    """N10: each auto-execute bash block is prefaced by a substitution
    reminder so the agent doesn't run the block with literal
    ``{{NEXT_PHASE}}`` / ``{{next_implement}}`` tokens.
    """
    body = (OPENCODE_DIR / f"{name}.md").read_text(encoding="utf-8")
    assert SUBSTITUTION_REMINDER in body, (
        f"{name}: missing the substitution reminder above the bash "
        f"block. The prose must say (verbatim): "
        f"'{SUBSTITUTION_REMINDER}' (review-fix #01 N10)."
    )


@pytest.mark.parametrize("name", OPENCODE_EXECUTE_AGENTS)
def test_opencode_agent_no_emoji_in_success_block(name: str) -> None:
    """N3: project rule forbids emoji in agent files unless the user
    asked for them. The QS-190 v1 success-report blocks used ``✅``;
    review-fix #01 N3 replaces them with ``[OK]``.
    """
    body = (OPENCODE_DIR / f"{name}.md").read_text(encoding="utf-8")
    assert "✅" not in body, (
        f"{name}: contains ``✅`` emoji — project rule forbids emojis "
        f"in agent files. Replace with ``[OK]`` (review-fix #01 N3)."
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


def test_opencode_setup_task_carries_rationale_block() -> None:
    """S13: AC-5 rationale paragraph is present (not just the EXECUTE_MARKER
    absence).

    The rationale block explains WHY setup-task is the print-for-user
    exception (cross-workspace launch). The previous test only
    asserted EXECUTE_MARKER absence, which a future refactor could
    silently satisfy by deleting the rationale. Pinning a distinctive
    phrase guards against that.
    """
    body = (OPENCODE_DIR / "qs-setup-task.md").read_text(encoding="utf-8")
    assert "Why setup-task stays print-for-user" in body, (
        "qs-setup-task.md: missing the AC-5 rationale block headed "
        "'Why setup-task stays print-for-user'. Without this, a "
        "future reader has no explanation for why setup-task is the "
        "asymmetric exception (review-fix #01 S13)."
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
