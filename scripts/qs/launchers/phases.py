"""Phase name → static-agent name mapping (single source of truth).

Both ``launchers/claude.py`` (used by ``setup_task.py`` and friends) and
``next_step.py`` import this module to validate ``--next-cmd`` values and
resolve them to the ``qs-<phase>`` agent that lives under
``.claude/agents/``.

Keeping the mapping in one module avoids the drift bug where the launcher
and the dispatcher disagree on what counts as a valid phase. Adding a new
phase = add one entry here.

Public names (``PHASE_TO_AGENT``, ``resolve_agent_for_next_cmd``) are
intentional — they are imported across modules (``claude.py``,
``cursor.py``, ``next_step.py``, tests). Ruff's ``PLC2701`` would flag
underscore-prefixed names as private-import violations.
"""

from __future__ import annotations

# Static — phase name → agent file stem (without ``.md``). Every entry is
# ``"<phase>": f"qs-<phase>"`` by convention, but we keep the mapping
# explicit so a typo in either side is immediately visible.
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
    ``ValueError`` on any unknown phase. There is **no** silent fallback:
    free-form prompts get the separate ``--next-prompt`` arg.

    Args:
        next_cmd: ``"/create-plan"`` or ``"create-plan"`` etc.

    Returns:
        The matching agent file stem (e.g. ``"qs-create-plan"``).

    Raises:
        ValueError: if ``next_cmd`` is not in ``PHASE_TO_AGENT``. The
            error message names the invalid value and lists the known
            phases so the user immediately sees what they meant to type.
    """
    # Strip exactly one leading slash. ``//create-plan`` is intentionally
    # rejected — it indicates a caller bug, not a phase name we should heal.
    phase = next_cmd[1:] if next_cmd.startswith("/") else next_cmd
    if phase in PHASE_TO_AGENT:
        return PHASE_TO_AGENT[phase]
    raise ValueError(
        f"Unknown phase {next_cmd!r}; known phases: {sorted(PHASE_TO_AGENT)}",
    )
