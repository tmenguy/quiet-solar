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
    """The exception exposes ``.value``, ``.phase``, and ``.known`` (review-fix #04 NTH4)."""
    from launchers.phases import UnknownPhaseError  # type: ignore[import-not-found]

    exc = UnknownPhaseError("bogus", ["a", "b"])
    assert exc.value == "bogus"
    assert exc.known == ["a", "b"]
    # When ``value`` has no leading slash, ``.phase`` defaults to the
    # same string — no normalization needed.
    assert exc.phase == "bogus"
    # Message names the invalid value so a bare ``str(exc)`` is useful too.
    assert "bogus" in str(exc)


def test_unknown_phase_error_phase_strips_leading_slash() -> None:
    """``.phase`` carries the post-normalization lookup key (review-fix #04 NTH4)."""
    from launchers.phases import UnknownPhaseError  # type: ignore[import-not-found]

    exc = UnknownPhaseError("/bogus", ["a", "b"])
    assert exc.value == "/bogus"
    # ``.phase`` is what ``resolve_agent_for_next_cmd`` actually looked
    # up — strips the single leading slash. Saves the debugger from
    # re-deriving the lookup key.
    assert exc.phase == "bogus"
    # Message surfaces BOTH so the caller sees the user's input AND the
    # internal lookup key.
    msg = str(exc)
    assert "/bogus" in msg
    assert "bogus" in msg


def test_unknown_phase_error_accepts_tuple_for_known() -> None:
    """``known`` accepts any Sequence (review-fix #04 NTH3)."""
    from launchers.phases import UnknownPhaseError  # type: ignore[import-not-found]

    exc = UnknownPhaseError("bogus", ("a", "b"))
    # Internally normalized to list so callers don't get surprised by
    # mismatched return types.
    assert exc.known == ["a", "b"]


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


# --------------------------------------------------------------------------- #
# Review fix plan #02 — should-fix #6 / #12: build_existing_session_prompt edge cases
# --------------------------------------------------------------------------- #


def test_build_existing_session_prompt_happy_path() -> None:
    """Happy path: absolute fix-plan-path under work-dir → worktree-relative + #PR."""
    from launchers.phases import build_existing_session_prompt  # type: ignore[import-not-found]

    result = build_existing_session_prompt(
        "/tmp/wt",
        "/tmp/wt/docs/stories/QS-177.story_review_fix_#01.md",
        179,
    )
    assert result is not None
    assert "docs/stories/QS-177.story_review_fix_#01.md" in result
    assert "#179" in result
    assert "/tmp/wt/" not in result


def test_build_existing_session_prompt_omits_when_inputs_missing() -> None:
    """Either input missing → ``None`` (caller omits the key)."""
    from launchers.phases import build_existing_session_prompt  # type: ignore[import-not-found]

    assert build_existing_session_prompt("/work", None, 42) is None
    assert build_existing_session_prompt("/work", "/work/x.md", None) is None
    assert build_existing_session_prompt("/work", "", 42) is None


@pytest.mark.parametrize("bad_path", ["   ", "\t", "\n", " \t \n "])
def test_build_existing_session_prompt_rejects_whitespace_fix_plan_path(bad_path: str) -> None:
    """Whitespace-only ``fix_plan_path`` is also rejected.

    Without this guard, a value like ``"   "`` would produce a
    confusing ``"A new review fix plan landed:    \n…"`` prompt.
    Review fix #02 should-fix #12.
    """
    from launchers.phases import build_existing_session_prompt  # type: ignore[import-not-found]

    assert build_existing_session_prompt("/work", bad_path, 42) is None


def test_build_existing_session_prompt_handles_relpath_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``os.path.relpath`` raises ``ValueError`` (cross-drive on Windows etc.), fall back to absolute.

    On POSIX we simulate the failure by monkeypatching ``os.path.relpath``
    to raise — the function should still return a valid prompt with
    the absolute path embedded (the user can copy it).
    """
    import os.path

    from launchers.phases import build_existing_session_prompt  # type: ignore[import-not-found]

    def _raise(*_args: object, **_kwargs: object) -> str:
        raise ValueError("path is on mount '/' which is not the start of work_dir")

    monkeypatch.setattr(os.path, "relpath", _raise)

    result = build_existing_session_prompt(
        "/work", "/somewhere/else/fix-plan.md", 42,
    )
    assert result is not None
    # Falls back to the absolute path verbatim — still functional.
    assert "/somewhere/else/fix-plan.md" in result
    assert "#42" in result


# --------------------------------------------------------------------------- #
# Review fix plan #03 — should-fix #2: pr_number > 0 at helper boundary
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_pr", [0, -1, -100])
def test_build_existing_session_prompt_rejects_non_positive_pr(bad_pr: int) -> None:
    """Non-positive ``pr_number`` returns ``None`` at the helper boundary.

    Parity with the CLI-layer ``parser.error("--pr-number must be a
    positive integer")`` in ``next_step.py``. Programmer-direct
    callers (and unit tests) get the same contract.
    """
    from launchers.phases import build_existing_session_prompt  # type: ignore[import-not-found]

    assert build_existing_session_prompt("/work", "/work/fix.md", bad_pr) is None


# --------------------------------------------------------------------------- #
# Review fix plan #03 — should-fix #8: empty work_dir falls back to absolute
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_work_dir", ["", "   ", "\t"])
def test_build_existing_session_prompt_handles_empty_work_dir(bad_work_dir: str) -> None:
    """Empty / whitespace ``work_dir`` falls back to the verbatim absolute path.

    Without this guard, ``os.path.relpath("/abs/path/to/fix.md", "")``
    returns ``"../../../../../abs/path/to/fix.md"`` (relative from
    the CWD up to root, then back down) — a confusing artifact in
    the user-visible prompt.
    """
    from launchers.phases import build_existing_session_prompt  # type: ignore[import-not-found]

    result = build_existing_session_prompt(
        bad_work_dir, "/abs/path/to/fix.md", 42,
    )
    assert result is not None
    assert "/abs/path/to/fix.md" in result
    # The traversal artifact must NOT appear.
    assert "../" not in result
    assert "#42" in result


# --------------------------------------------------------------------------- #
# Review fix plan #03 — should-fix #10: strip fix_plan_path after guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", [
    "  docs/foo.md",      # leading
    "docs/foo.md  ",      # trailing
    "  docs/foo.md  ",    # both
])
def test_build_existing_session_prompt_strips_fix_plan_path(path: str) -> None:
    """Surrounding whitespace is stripped from ``fix_plan_path`` after the empty guard.

    The empty/whitespace guard validates a stripped view, but the
    untrimmed value was then passed to ``os.path.isabs`` /
    ``relpath`` — producing a prompt with leading or trailing
    whitespace artifacts (review fix #03 should-fix #10).
    """
    from launchers.phases import build_existing_session_prompt  # type: ignore[import-not-found]

    result = build_existing_session_prompt("/work", path, 42)
    assert result is not None
    # The path appears stripped in the prompt — no whitespace
    # adjacent to ``landed:`` or just before ``\nRe-run``.
    assert "landed: docs/foo.md\n" in result, result
    assert "docs/foo.md  " not in result  # no trailing space
