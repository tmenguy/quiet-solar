"""Phase-name → agent-name mapping (single source of truth).

Lives in ``scripts/qs/launchers/phases.py`` and is imported by both
``claude.py`` and ``next_step.py``. The mapping covers every phase the
QS pipeline knows about; ``_resolve_agent_for_next_cmd`` accepts either
the slash form (``"/create-plan"``) or the bare phase name and raises
``ValueError`` for anything it does not know.
"""

from __future__ import annotations

import pytest


# The exhaustive list of phases the pipeline ships. Reviewers will catch any
# drift between this and ``_PHASE_TO_AGENT``.
_KNOWN_PHASES = (
    "setup-task",
    "create-plan",
    "implement-task",
    "implement-setup-task",
    "review-task",
    "finish-task",
    "release",
)


def test_phase_to_agent_covers_every_known_phase() -> None:
    """Every QS phase has an entry in the mapping."""
    from launchers.phases import _PHASE_TO_AGENT  # type: ignore[import-not-found]

    assert set(_PHASE_TO_AGENT) == set(_KNOWN_PHASES)


def test_phase_to_agent_values_match_qs_prefix() -> None:
    """Every value is ``qs-<key>`` — the static-agent file convention."""
    from launchers.phases import _PHASE_TO_AGENT  # type: ignore[import-not-found]

    for phase, agent in _PHASE_TO_AGENT.items():
        assert agent == f"qs-{phase}", (
            f"Expected mapping {phase!r} → 'qs-{phase}', got {agent!r}"
        )


@pytest.mark.parametrize("phase", _KNOWN_PHASES)
def test_resolve_bare_phase_name(phase: str) -> None:
    """Bare phase name resolves to the qs-prefixed agent."""
    from launchers.phases import _resolve_agent_for_next_cmd  # type: ignore[import-not-found]

    assert _resolve_agent_for_next_cmd(phase) == f"qs-{phase}"


@pytest.mark.parametrize("phase", _KNOWN_PHASES)
def test_resolve_slash_phase_name(phase: str) -> None:
    """Slash form (back-compat for callers that pass /create-plan)."""
    from launchers.phases import _resolve_agent_for_next_cmd  # type: ignore[import-not-found]

    assert _resolve_agent_for_next_cmd(f"/{phase}") == f"qs-{phase}"


@pytest.mark.parametrize("bogus", ["foo", "", "  ", "/nope", "/", "create plan"])
def test_resolve_unknown_phase_raises(bogus: str) -> None:
    """Unknown phase raises ValueError naming the value and listing known phases."""
    from launchers.phases import (  # type: ignore[import-not-found]
        _PHASE_TO_AGENT,
        _resolve_agent_for_next_cmd,
    )

    with pytest.raises(ValueError) as excinfo:
        _resolve_agent_for_next_cmd(bogus)

    msg = str(excinfo.value)
    # The message must name the invalid value AND list known phases (so the
    # user can see what they meant to type).
    assert repr(bogus) in msg or bogus in msg
    for phase in _PHASE_TO_AGENT:
        assert phase in msg


def test_resolve_strips_only_leading_slash() -> None:
    """``//create-plan`` is not valid — we strip exactly one leading slash."""
    from launchers.phases import _resolve_agent_for_next_cmd  # type: ignore[import-not-found]

    with pytest.raises(ValueError):
        _resolve_agent_for_next_cmd("//create-plan")
