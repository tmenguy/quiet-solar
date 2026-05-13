"""Lock the AC-3 contract: every phase orchestrator's handoff section emits
BOTH a launcher block (``claude --agent qs-<phase>``) AND a slash-command
fallback block (``/<phase>``).

This is the regression catch for QS-175 review-fix #07 — without it the
two-block pattern is enforced only by manual review.

Round-2 review-fix #03 / #04 / #05 / #06 cleanups:
- ``qs-create-plan`` is split out into a dedicated test because its
  ``NEXT_PHASE`` is dynamic (the orchestrator picks
  ``implement-task`` vs ``implement-setup-task`` at runtime) and a
  hardcoded slash form isn't possible there.
- The parametrise lists are split so unused ``expected_slash`` params
  don't trigger ruff ``ARG001``.
- The ``Fallback`` line scan is now a line-by-line walk rather than a
  brittle ``re.search`` with a greedy ``[^:]*`` pattern that could match
  past a future "Note:" parenthetical.
- The ``does-not-call-next-step-for-release`` regex strips inline +
  fenced backticks before scanning so a backtick-wrapped ``python ...
  --next-cmd release`` invocation can't slip past.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENTS_DIR = REPO_ROOT / ".claude" / "agents"

# Pattern for an ACTIVE ``next_step.py --next-cmd release`` invocation.
# Uses ``.`` with ``re.DOTALL`` so backslash-continued multi-line shell
# forms (``python ... \\\n    --next-cmd release``) are caught — the
# previous ``[^\n]`` form silently let them through (review-fix #04
# SF2). Compiled once at module load and shared between the
# qs-finish-task assertion and the regex-shape regression tests.
_FORBIDDEN_RELEASE_INVOCATION = re.compile(
    r"scripts/qs/next_step\.py.{0,200}?--next-cmd\s+[\"']?release",
    re.DOTALL,
)


# All 6 orchestrators that ship the strict two-block ``Preferred`` /
# ``Fallback`` pattern. ``qs-finish-task`` is the deliberate exception
# (release follow-up is text-only — see QS-175 OUT OF SCOPE) and gets
# its own dedicated tests below.
_TWO_BLOCK_ORCHESTRATORS = [
    "qs-setup-task.md",
    "qs-create-plan.md",
    "qs-implement-task.md",
    "qs-implement-setup-task.md",
    "qs-review-task.md",
]

# Five of those orchestrators emit a hardcoded ``/<phase>`` in the
# fallback block. ``qs-create-plan`` is dynamic (the NEXT_PHASE depends
# on the diff) and gets its own dedicated test asserting it uses a
# placeholder of the right SHAPE, but explicitly NOT
# ``{{same_context}}``.
_HARDCODED_FALLBACK = [
    ("qs-setup-task.md", "/create-plan"),
    ("qs-implement-task.md", "/review-task"),
    ("qs-implement-setup-task.md", "/review-task"),
    ("qs-review-task.md", "/finish-task"),
]


@pytest.mark.parametrize("filename", _TWO_BLOCK_ORCHESTRATORS)
def test_orchestrator_has_two_block_handoff(filename: str) -> None:
    """Every orchestrator's handoff contains 'Preferred' AND 'Fallback' markers."""
    body = (AGENTS_DIR / filename).read_text()
    assert "Preferred" in body, (
        f"{filename}: missing 'Preferred' marker — every orchestrator must "
        f"label the interactive launcher path."
    )
    assert "Fallback" in body, (
        f"{filename}: missing 'Fallback' marker — every orchestrator must "
        f"label the degraded slash-command path."
    )


@pytest.mark.parametrize("filename", _TWO_BLOCK_ORCHESTRATORS)
def test_orchestrator_handoff_mentions_claude_agent_invocation(filename: str) -> None:
    """Handoff text references ``claude --agent qs-<phase>`` — the preferred path."""
    body = (AGENTS_DIR / filename).read_text()
    # The orchestrator may emit either a literal ``claude --agent qs-...``
    # string (qs-setup-task) or use the new_context variable that resolves
    # to one at runtime (qs-create-plan, qs-implement-task, etc.). Either
    # way the launcher concept must be named in the file body.
    assert "claude --agent" in body, (
        f"{filename}: the orchestrator handoff must reference "
        f"`claude --agent qs-<phase>` so the user knows the preferred path."
    )


@pytest.mark.parametrize(("filename", "expected_slash"), _HARDCODED_FALLBACK)
def test_orchestrator_fallback_uses_concrete_slash_form(
    filename: str, expected_slash: str,
) -> None:
    """Each fixed-next-phase fallback block names a real ``/<phase>``."""
    body = (AGENTS_DIR / filename).read_text()
    assert expected_slash in body, (
        f"{filename}: expected fallback to mention {expected_slash!r}; "
        f"this catches the qs-setup-task-style regression where "
        f"{{{{same_context}}}} or another verbatim template variable "
        f"would be emitted instead of a hardcoded slash form."
    )


# --------------------------------------------------------------------------- #
# qs-create-plan is the dynamic-next-phase exception. The fallback line
# can't be a single hardcoded ``/<phase>`` because the orchestrator picks
# ``implement-task`` vs ``implement-setup-task`` based on its task
# breakdown. The fallback uses a ``/{{NEXT_PHASE}}`` placeholder that
# the persona substitutes at runtime — what we forbid is the
# ``{{same_context}}`` shape that caused the qs-setup-task regression.
# --------------------------------------------------------------------------- #


def test_create_plan_fallback_uses_next_phase_placeholder_not_same_context() -> None:
    """qs-create-plan fallback uses ``/<NEXT_PHASE>`` placeholder, not ``{{same_context}}``."""
    body = (AGENTS_DIR / "qs-create-plan.md").read_text()
    fallback_line = _find_fallback_line(body)
    assert fallback_line is not None, "qs-create-plan: 'Fallback' block not found"
    assert "{{same_context}}" not in fallback_line, (
        f"qs-create-plan fallback uses {{{{same_context}}}} — that's the "
        f"verbatim template variable that caused the qs-setup-task "
        f"regression; use a slash-form placeholder like /{{{{NEXT_PHASE}}}} "
        f"instead. Got: {fallback_line!r}"
    )
    assert fallback_line.lstrip().startswith("/"), (
        f"qs-create-plan fallback should start with '/' (slash form). "
        f"Got: {fallback_line!r}"
    )


# --------------------------------------------------------------------------- #
# qs-setup-task: same dedicated assertion as before — hardcode /create-plan,
# don't fall back to a {{same_context}} template. This is the case the
# round-1 fix targeted.
# --------------------------------------------------------------------------- #


def test_setup_task_fallback_does_not_use_same_context_template() -> None:
    """qs-setup-task's fallback block must hardcode ``/create-plan``, not template."""
    body = (AGENTS_DIR / "qs-setup-task.md").read_text()
    fallback_line = _find_fallback_line(body)
    assert fallback_line is not None, "qs-setup-task: 'Fallback' block not found"
    assert "{{same_context}}" not in fallback_line, (
        "qs-setup-task fallback uses {{same_context}} template — must hardcode "
        "/create-plan to match the four peer orchestrators (review-fix #03)."
    )
    assert "/create-plan" in fallback_line, (
        f"qs-setup-task fallback should hardcode '/create-plan', got: "
        f"{fallback_line!r}"
    )


# --------------------------------------------------------------------------- #
# qs-finish-task is the deliberate exception — release runs on a different
# workspace (main checkout), so the agent body shows both forms as
# alternatives in plain prose rather than the strict two-block pattern.
# Per QS-175 OUT OF SCOPE: "DO NOT call the launcher with --next-cmd release".
# --------------------------------------------------------------------------- #


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
    in real source, not inside backticks.

    We strip both inline and fenced backticks before scanning so a stray
    backtick-wrapped ``python scripts/qs/next_step.py --next-cmd release``
    on a single line can't bypass the check. The regex uses ``.`` with
    ``re.DOTALL`` (instead of ``[^\\n]``) so a backslash-continued
    multi-line invocation also gets caught (review-fix #04 SF2).
    """
    body = (AGENTS_DIR / "qs-finish-task.md").read_text()
    stripped = _strip_backticks(body)
    match = _FORBIDDEN_RELEASE_INVOCATION.search(stripped)
    assert not match, (
        f"qs-finish-task: active 'next_step.py --next-cmd release' invocation "
        f"found ({match.group()!r}). Release runs on the main checkout "
        f"(different workspace); per QS-175 OUT OF SCOPE the agent surfaces "
        f"the text suggestion but does not emit a launcher payload."
    )


# Regression patterns the forbidden-invocation regex MUST catch. Round-4
# SF2: the round-2 implementation used ``[^\n]`` which silently let
# backslash-continued multi-line invocations through.
_MULTILINE_INVOCATION = (
    "python scripts/qs/next_step.py \\\n"
    "    --next-cmd release\n"
)
_SINGLE_LINE_INVOCATION = (
    "python scripts/qs/next_step.py --next-cmd release\n"
)
_INVOCATION_QUOTED = (
    'python scripts/qs/next_step.py --next-cmd "release"\n'
)


@pytest.mark.parametrize("snippet", [
    _SINGLE_LINE_INVOCATION,
    _MULTILINE_INVOCATION,
    _INVOCATION_QUOTED,
])
def test_forbidden_release_regex_catches_invocation_forms(snippet: str) -> None:
    """The regex catches every active invocation shape — single, multi, quoted."""
    assert _FORBIDDEN_RELEASE_INVOCATION.search(snippet), (
        f"Forbidden-invocation regex missed {snippet!r}. Without DOTALL + "
        f"``.`` across newlines, a backslash-continued shell line slips "
        f"through (review-fix #04 SF2)."
    )


def test_forbidden_release_regex_ignores_prose_mention() -> None:
    """Bare prose mentioning ``--next-cmd release`` (without the script path) is fine."""
    # The agent body uses this exact wording in its OUT OF SCOPE note —
    # only an ACTIVE invocation paired with the script path is forbidden.
    prose = "We don't build a launcher with --next-cmd release."
    assert _FORBIDDEN_RELEASE_INVOCATION.search(prose) is None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _find_fallback_line(body: str) -> str | None:
    """Return the first non-empty line following the literal ``Fallback`` marker.

    Replaces the brittle ``re.search(r"Fallback[^:]*:...")`` pattern (review-
    fix #02 NTH2). Iterates lines: once a line starting with ``Fallback``
    (capital F — the marker, not a prose mention) is found, walk forward
    until a content line that starts with ``/`` or ``{{`` is reached.
    That's the slash-fallback (or placeholder) line.
    """
    lines = body.splitlines()
    in_fallback_block = False
    for line in lines:
        stripped = line.strip()
        if not in_fallback_block:
            # Match a ``Fallback`` marker — must start with that literal
            # word (after any leading whitespace). Excludes prose
            # mentions like "the fallback path" that appear earlier in
            # the file body.
            if stripped.startswith("Fallback"):
                in_fallback_block = True
            continue
        # In the fallback block — find the first line whose stripped
        # content begins with ``/`` (the slash-form fallback) or with
        # ``{{`` (a placeholder, e.g. ``{{same_context}}``). Either is a
        # candidate "fallback line" for downstream tests to inspect.
        # Blank lines, the preamble's closing ``):`` line, and any
        # other intermediate content are naturally skipped by the
        # implicit loop continuation (review-fix #04 NTH2: the trailing
        # explicit ``if/continue`` was dead code).
        if stripped.startswith(("/", "{{")):
            return line
    return None


def _strip_backticks(text: str) -> str:
    """Remove single and triple backtick fences, keeping the inner content.

    Review-fix #02 NTH3: without this, a fenced one-line invocation
    like a backtick-wrapped ``python scripts/qs/next_step.py
    --next-cmd release`` would slip past the
    ``does-not-call-next-step-for-release`` regex.
    """
    # Strip triple-fence blocks first (they may span multiple lines), then
    # single-tick inline code. Both forms are replaced with their inner
    # text so the search can find an active invocation regardless of
    # markdown formatting.
    text = re.sub(r"```[a-zA-Z]*\n?([\s\S]*?)```", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text
