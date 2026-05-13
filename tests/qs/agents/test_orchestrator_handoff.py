"""Lock the AC-3 contract: every phase orchestrator's handoff section emits
BOTH a launcher block (``claude --agent qs-<phase>``) AND a slash-command
fallback block (``/<phase>``).

This is the regression catch for QS-175 review-fix #7 — without it the
two-block pattern is enforced only by manual review. The test also catches
review-fix #3 (qs-setup-task fallback drift): it asserts each fallback
mentions ``/<phase>`` for a concrete known phase, not a verbatim template
variable like ``{{same_context}}``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENTS_DIR = REPO_ROOT / ".claude" / "agents"

# Each entry: (orchestrator file, the next-phase slash that MUST appear in
# the handoff wording somewhere). The five "normal" orchestrators ship the
# strict two-block ``Preferred`` / ``Fallback`` pattern; qs-finish-task is
# the deliberate exception — see ``_FINISH_TASK_EXCEPTIONS`` below.
_TWO_BLOCK_ORCHESTRATORS = [
    ("qs-setup-task.md", "/create-plan"),
    ("qs-create-plan.md", "/{{NEXT_PHASE}}"),
    ("qs-implement-task.md", "/review-task"),
    ("qs-implement-setup-task.md", "/review-task"),
    ("qs-review-task.md", "/finish-task"),
]


@pytest.mark.parametrize(("filename", "expected_slash"), _TWO_BLOCK_ORCHESTRATORS)
def test_orchestrator_has_two_block_handoff(filename: str, expected_slash: str) -> None:
    """Every orchestrator's handoff contains 'Preferred' AND 'Fallback' markers."""
    path = AGENTS_DIR / filename
    body = path.read_text()
    assert "Preferred" in body, (
        f"{filename}: missing 'Preferred' marker — every orchestrator must "
        f"label the interactive launcher path."
    )
    assert "Fallback" in body, (
        f"{filename}: missing 'Fallback' marker — every orchestrator must "
        f"label the degraded slash-command path."
    )


@pytest.mark.parametrize(("filename", "expected_slash"), _TWO_BLOCK_ORCHESTRATORS)
def test_orchestrator_handoff_mentions_claude_agent_invocation(
    filename: str, expected_slash: str,
) -> None:
    """Handoff text references ``claude --agent qs-<phase>`` — the preferred path."""
    path = AGENTS_DIR / filename
    body = path.read_text()
    # The orchestrator may emit either a literal ``claude --agent qs-...``
    # string (qs-setup-task) or use the new_context variable that resolves
    # to one at runtime (qs-create-plan, qs-implement-task, etc.). Either
    # way the launcher concept must be named in the file body.
    assert "claude --agent" in body, (
        f"{filename}: the orchestrator handoff must reference "
        f"`claude --agent qs-<phase>` so the user knows the preferred path."
    )


@pytest.mark.parametrize(("filename", "expected_slash"), _TWO_BLOCK_ORCHESTRATORS)
def test_orchestrator_fallback_uses_concrete_slash_form(
    filename: str, expected_slash: str,
) -> None:
    """Each fallback block names a real ``/<phase>`` slash, not ``{{same_context}}``."""
    path = AGENTS_DIR / filename
    body = path.read_text()
    assert expected_slash in body, (
        f"{filename}: expected fallback to mention {expected_slash!r}; "
        f"this catches the qs-setup-task regression where {{{{same_context}}}} "
        f"was emitted instead of a hardcoded slash form."
    )


# qs-finish-task is the deliberate exception — release runs on a different
# workspace (main checkout), so the agent body shows both forms as
# alternatives in plain prose rather than the strict two-block pattern.
# Per QS-175 OUT OF SCOPE: "DO NOT call the launcher with --next-cmd release".


def test_finish_task_mentions_both_release_forms() -> None:
    """qs-finish-task's release suggestion mentions both forms as alternatives."""
    body = (AGENTS_DIR / "qs-finish-task.md").read_text()
    assert "/release" in body, "qs-finish-task: missing '/release' fallback mention"
    assert "claude --agent qs-release" in body, (
        "qs-finish-task: missing 'claude --agent qs-release' — the interactive "
        "form must also be named so users on CLI know the preferred path."
    )


def test_finish_task_does_not_call_next_step_for_release() -> None:
    """qs-finish-task must NOT call ``next_step.py --next-cmd release`` (OUT OF SCOPE).

    The agent body may *mention* the forbidden pattern in prose (e.g. "we
    don't build a launcher with `--next-cmd release`" — that's the OUT OF
    SCOPE explanation). What's forbidden is an ACTIVE invocation: a
    ``scripts/qs/next_step.py`` call followed by ``--next-cmd release``
    in the same code block.
    """
    body = (AGENTS_DIR / "qs-finish-task.md").read_text()
    # Look for an active invocation: next_step.py + --next-cmd "release"
    # within a window of ~200 chars (a single code block). This pattern
    # would be the real OUT OF SCOPE violation.
    forbidden = re.compile(
        r"scripts/qs/next_step\.py[^\n`]{0,200}?--next-cmd\s+[\"']?release",
        re.DOTALL,
    )
    match = forbidden.search(body)
    assert not match, (
        f"qs-finish-task: active 'next_step.py --next-cmd release' invocation "
        f"found ({match.group()!r}). Release runs on the main checkout "
        f"(different workspace); per QS-175 OUT OF SCOPE the agent surfaces "
        f"the text suggestion but does not emit a launcher payload."
    )


def test_setup_task_fallback_does_not_use_same_context_template() -> None:
    """qs-setup-task's fallback block must hardcode ``/create-plan``, not template."""
    body = (AGENTS_DIR / "qs-setup-task.md").read_text()
    # Find the fallback block content — between the literal ``Fallback``
    # marker (capital F, distinguishing it from prose mentions of
    # "fallback" earlier in the file) and its captured line.
    fallback_match = re.search(
        r"Fallback[^:]*:[^\n]*\n([^\n]+)\n", body,
    )
    assert fallback_match, "qs-setup-task: 'Fallback' block not found"
    fallback_line = fallback_match.group(1)
    assert "{{same_context}}" not in fallback_line, (
        "qs-setup-task fallback uses {{same_context}} template — must hardcode "
        "/create-plan to match the five peer orchestrators (review-fix #3)."
    )
    assert "/create-plan" in fallback_line, (
        f"qs-setup-task fallback should hardcode '/create-plan', got: "
        f"{fallback_line!r}"
    )
