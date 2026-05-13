"""Phase name → static-agent name mapping (single source of truth).

Both ``launchers/claude.py`` (used by ``setup_task.py`` and friends) and
``next_step.py`` import this module to validate ``--next-cmd`` values and
resolve them to the ``qs-<phase>`` agent that lives under
``.claude/agents/``.

Keeping the mapping in one module makes the canonical name set
grep-able. The ``qs-<phase>`` convention (every value is
``f"qs-{key}"``) is enforced by
``tests/qs/launchers/test_phase_mapping.py::test_phase_to_agent_values_match_qs_prefix``;
the mapping stays as an explicit dict so that adding a new phase
remains a single-line, reviewable change.

Public names (``PHASE_TO_AGENT``, ``resolve_agent_for_next_cmd``,
``UnknownPhaseError``, ``build_existing_session_prompt``) are
intentional — they are imported across modules (``claude.py``,
``cursor.py``, ``next_step.py``, tests). Ruff's ``PLC2701`` would
flag underscore-prefixed names as private-import violations.
"""

from __future__ import annotations

import os
from collections.abc import Sequence


class UnknownPhaseError(ValueError):
    """Raised by ``resolve_agent_for_next_cmd`` for an unknown phase.

    Subclass of ``ValueError`` so legacy ``except ValueError`` blocks
    keep working. Carrying the ``value``, ``phase``, and ``known``
    attributes lets callers (notably ``next_step.py``) format their JSON
    error payload without re-parsing the exception message — review-fix
    #02 SF1.

    Attributes:
        value: The original ``next_cmd`` input that failed resolution
            (e.g. ``"/bogus"`` if the caller passed the slash form).
        phase: The post-normalization lookup key actually consulted in
            ``PHASE_TO_AGENT`` (e.g. ``"bogus"`` for ``value="/bogus"``).
            Surfaced separately so a debugger doesn't have to re-derive
            it from ``value`` (review-fix #04 NTH4).
        known: The list of known phase names — what the caller should
            display as recovery hints. Accepts any ``Sequence[str]`` so
            tuple-based callers (test fixtures) don't trigger mypy
            noise (review-fix #04 NTH3).
    """

    def __init__(
        self,
        value: str,
        known: Sequence[str],
        *,
        phase: str | None = None,
    ) -> None:
        # ``known`` is normalised to ``list`` so callers can rely on
        # standard list operations (sorted, indexing) without surprise.
        known_list = list(known)
        # ``phase`` defaults to the post-normalization lookup key —
        # which is what ``resolve_agent_for_next_cmd`` actually consulted.
        # Callers that build the exception synthetically (e.g. tests)
        # can pass it explicitly.
        if phase is None:
            phase = value[1:] if value.startswith("/") else value
        # Comma-separated rendering reads more cleanly than ``repr(list)``
        # — the latter mixes square brackets and single quotes that look
        # like noise when ``value`` is also a short string (review-fix
        # #03 NTH5). Include the post-normalization lookup key when it
        # differs from ``value`` so a debugger doesn't have to re-derive
        # it (review-fix #04 NTH4).
        if phase != value:
            head = f"Unknown phase {value!r} (lookup key {phase!r})"
        else:
            head = f"Unknown phase {value!r}"
        super().__init__(f"{head}; known phases: {', '.join(known_list)}")
        self.value = value
        self.phase = phase
        self.known = known_list


# Static — phase name → agent file stem (without ``.md``). Every entry is
# ``"<phase>": f"qs-<phase>"`` by convention (pinned by a regression test
# in tests/qs/launchers/test_phase_mapping.py).
PHASE_TO_AGENT: dict[str, str] = {
    "setup-task": "qs-setup-task",
    "create-plan": "qs-create-plan",
    "implement-task": "qs-implement-task",
    "implement-setup-task": "qs-implement-setup-task",
    "review-task": "qs-review-task",
    "finish-task": "qs-finish-task",
    "release": "qs-release",
}


def resolve_agent_for_next_cmd(next_cmd: str) -> str:
    """Return the ``qs-<phase>`` agent name for a ``--next-cmd`` value.

    Accepts either the slash form (``"/create-plan"`` — what older
    callers pass) or the bare phase name (``"create-plan"``). Raises
    ``UnknownPhaseError`` (a ``ValueError`` subclass) on any unknown
    phase. There is **no** silent fallback: free-form prompts get the
    separate ``--next-prompt`` arg.

    Args:
        next_cmd: ``"/create-plan"`` or ``"create-plan"`` etc.

    Returns:
        The matching agent file stem (e.g. ``"qs-create-plan"``).

    Raises:
        UnknownPhaseError: if ``next_cmd`` is not in ``PHASE_TO_AGENT``.
            The exception's ``.value`` and ``.known`` attributes carry
            the raw input and the list of accepted phases so callers
            don't have to re-parse the message.
    """
    # Strip exactly one leading slash. ``//create-plan`` is intentionally
    # rejected — it indicates a caller bug, not a phase name we should heal.
    phase = next_cmd[1:] if next_cmd.startswith("/") else next_cmd
    if phase in PHASE_TO_AGENT:
        return PHASE_TO_AGENT[phase]
    # Pass ``phase`` explicitly so the exception's ``.phase`` attribute
    # records the post-normalization lookup key (review-fix #04 NTH4).
    raise UnknownPhaseError(next_cmd, sorted(PHASE_TO_AGENT), phase=phase)


def build_existing_session_prompt(
    work_dir: str,
    fix_plan_path: str | None,
    pr_number: int | None,
) -> str | None:
    """Return the existing-session kickoff prompt, or ``None`` if inputs are incomplete.

    Review-task → implement-task is the most common loop in the
    pipeline. After ``qs-review-task`` writes a fix plan and lands a
    new commit, the user often has an already-running
    ``qs-implement-task`` session they can paste a short prompt into
    instead of spinning up a fresh terminal. This helper builds that
    paste-into-existing-session prompt.

    Args:
        work_dir: Worktree directory. Used as the base for normalising
            ``fix_plan_path`` to a worktree-relative path so the
            prompt does not leak an absolute ``/Users/...`` prefix.
        fix_plan_path: Absolute or relative path to the fix-plan
            markdown file. ``None`` (or empty) → no prompt is built.
        pr_number: The open PR number that the next implement-task
            session should push back to. ``None`` → no prompt is built.

    Returns:
        A multi-line prompt string when both inputs are present, else
        ``None`` (callers omit the ``existing_session_prompt`` key
        from their payload entirely).

    Review fix plan #01 — should-fix #17: shared helper so the prompt
    content stays harness-agnostic. Each launcher decides whether to
    surface the field via ``build_payload``; the wrapper text in the
    agent presentation differs per harness but the prompt body does
    not.
    """
    if not fix_plan_path or pr_number is None:
        return None

    # Normalise the path to worktree-relative. If ``fix_plan_path`` is
    # absolute and lives under ``work_dir``, strip the prefix; if
    # it's already relative, pass it through verbatim. Anything
    # outside the worktree (rare) goes through ``os.path.relpath``
    # which returns a ``../...`` form.
    if os.path.isabs(fix_plan_path):
        try:
            fix_plan_rel = os.path.relpath(fix_plan_path, work_dir)
        except ValueError:
            # Cross-drive on Windows or similar — fall back to the
            # raw absolute path. Acceptable degraded UX.
            fix_plan_rel = fix_plan_path
    else:
        fix_plan_rel = fix_plan_path

    return (
        f"A new review fix plan landed: {fix_plan_rel}\n"
        f"Re-run `python scripts/qs/context.py` to pick up "
        f"`latest_review_fix`, then apply the plan per your phase "
        f"protocol (TDD, quality gate, push to the existing PR "
        f"#{pr_number})."
    )
