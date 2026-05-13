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
``UnknownPhaseError``) are intentional — they are imported across
modules (``claude.py``, ``cursor.py``, ``next_step.py``, tests). Ruff's
``PLC2701`` would flag underscore-prefixed names as private-import
violations.
"""

from __future__ import annotations


class UnknownPhaseError(ValueError):
    """Raised by ``resolve_agent_for_next_cmd`` for an unknown phase.

    Subclass of ``ValueError`` so legacy ``except ValueError`` blocks
    keep working. Carrying the ``value`` and ``known`` attributes lets
    callers (notably ``next_step.py``) format their JSON error payload
    without re-parsing the exception message — review-fix #02 SF1.
    """

    def __init__(self, value: str, known: list[str]) -> None:
        # Comma-separated rendering reads more cleanly than ``repr(list)``
        # — the latter mixes square brackets and single quotes that look
        # like noise when ``value`` is also a short string (review-fix
        # #03 NTH5).
        super().__init__(
            f"Unknown phase {value!r}; known phases: {', '.join(known)}",
        )
        self.value = value
        self.known = known


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
    raise UnknownPhaseError(next_cmd, sorted(PHASE_TO_AGENT))
