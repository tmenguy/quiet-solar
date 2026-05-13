"""Phase-name → agent-name mapping (single source of truth).

Lives in ``scripts/qs/launchers/phases.py`` and is imported by both
``claude.py`` and ``next_step.py``. The mapping covers every phase the
QS pipeline knows about; ``resolve_agent_for_next_cmd`` accepts either
the slash form (``"/create-plan"``) or the bare phase name and raises
``ValueError`` for anything it does not know.
"""

from __future__ import annotations

import pytest

# The exhaustive list of phases the pipeline ships. Reviewers will catch any
# drift between this and ``PHASE_TO_AGENT``.
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
    from launchers.phases import PHASE_TO_AGENT  # type: ignore[import-not-found]

    assert set(PHASE_TO_AGENT) == set(_KNOWN_PHASES)


def test_phase_to_agent_values_match_qs_prefix() -> None:
    """Every value is ``qs-<key>`` — the static-agent file convention."""
    from launchers.phases import PHASE_TO_AGENT  # type: ignore[import-not-found]

    for phase, agent in PHASE_TO_AGENT.items():
        assert agent == f"qs-{phase}", (
            f"Expected mapping {phase!r} → 'qs-{phase}', got {agent!r}"
        )


@pytest.mark.parametrize("phase", _KNOWN_PHASES)
def test_resolve_bare_phase_name(phase: str) -> None:
    """Bare phase name resolves to the qs-prefixed agent."""
    from launchers.phases import resolve_agent_for_next_cmd  # type: ignore[import-not-found]

    assert resolve_agent_for_next_cmd(phase) == f"qs-{phase}"


@pytest.mark.parametrize("phase", _KNOWN_PHASES)
def test_resolve_slash_phase_name(phase: str) -> None:
    """Slash form (back-compat for callers that pass /create-plan)."""
    from launchers.phases import resolve_agent_for_next_cmd  # type: ignore[import-not-found]

    assert resolve_agent_for_next_cmd(f"/{phase}") == f"qs-{phase}"


@pytest.mark.parametrize("bogus", ["foo", "  ", "/nope", "/", "create plan"])
def test_resolve_unknown_phase_raises(bogus: str) -> None:
    """Unknown phase raises ValueError naming the value and listing known phases.

    Note: the empty string ``""`` is intentionally excluded from this
    parametrise list (review-fix #03 MF1). It would make
    ``bogus in msg`` vacuously true (every string contains ``""``),
    silently masking a regression in error-message formatting. The
    empty-input invariant is pinned at the CLI layer instead by
    ``test_empty_or_whitespace_next_cmd_rejected_for_all_harnesses``
    in ``test_next_step_cli.py``.
    """
    from launchers.phases import (  # type: ignore[import-not-found]
        PHASE_TO_AGENT,
        resolve_agent_for_next_cmd,
    )

    with pytest.raises(ValueError) as excinfo:
        resolve_agent_for_next_cmd(bogus)

    msg = str(excinfo.value)
    # The message must name the invalid value AND list known phases (so the
    # user can see what they meant to type).
    assert repr(bogus) in msg or bogus in msg
    for phase in PHASE_TO_AGENT:
        assert phase in msg


def test_resolve_strips_only_leading_slash() -> None:
    """``//create-plan`` is not valid — we strip exactly one leading slash."""
    from launchers.phases import resolve_agent_for_next_cmd  # type: ignore[import-not-found]

    with pytest.raises(ValueError):
        resolve_agent_for_next_cmd("//create-plan")


# --------------------------------------------------------------------------- #
# UnknownPhaseError contract — review-fix #02 SF1.
#
# The narrowing requires a dedicated exception so next_step.py can catch
# JUST the "unknown phase" case (not every ValueError raised by build_payload
# or its callees). The exception subclasses ValueError to preserve back-compat
# with any legacy ``except ValueError`` block.
# --------------------------------------------------------------------------- #


def test_unknown_phase_error_is_a_value_error() -> None:
    """``UnknownPhaseError`` is a subclass of ``ValueError`` (back-compat)."""
    from launchers.phases import UnknownPhaseError  # type: ignore[import-not-found]

    assert issubclass(UnknownPhaseError, ValueError)


def test_unknown_phase_error_carries_value_and_known_attrs() -> None:
    """The exception exposes ``.value`` and ``.known`` so callers needn't re-parse."""
    from launchers.phases import UnknownPhaseError  # type: ignore[import-not-found]

    exc = UnknownPhaseError("bogus", ["a", "b"])
    assert exc.value == "bogus"
    assert exc.known == ["a", "b"]
    # Message names the invalid value so a bare ``str(exc)`` is useful too.
    assert "bogus" in str(exc)


def test_resolve_raises_unknown_phase_error_specifically() -> None:
    """``resolve_agent_for_next_cmd`` raises ``UnknownPhaseError``, not plain ``ValueError``."""
    from launchers.phases import (  # type: ignore[import-not-found]
        UnknownPhaseError,
        resolve_agent_for_next_cmd,
    )

    with pytest.raises(UnknownPhaseError) as excinfo:
        resolve_agent_for_next_cmd("bogus-phase")
    assert excinfo.value.value == "bogus-phase"
    assert "create-plan" in excinfo.value.known
