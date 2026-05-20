"""Tests for ``scripts/qs/harness.py`` and harness-flag symmetry between
``setup_task.py`` and ``next_step.py``.

Review fix #01 — N7 + N8:

- **N7**: ``setup_task.py`` and ``next_step.py`` both surface
  ``--harness`` with a ``choices=`` enumeration. Today both lists are
  identical, but ``setup_task.py`` sources its choices from
  ``harness.VALID_HARNESSES`` and ``next_step.py`` sources from its
  local ``LAUNCHERS`` dispatch table. Aligning both on
  ``list(LAUNCHERS)`` (the dispatch table is the authoritative source
  — a name absent from it would ``KeyError`` at runtime) prevents
  silent drift.
- **N8**: ``harness.detect()`` accepts legacy aliases (``claude`` →
  ``claude-code``) when sourced from ``QS_HARNESS``; the new
  ``canonicalize()`` helper exposes the same alias-mapping for the
  ``--harness`` argparse flag so a user typing ``--harness claude``
  doesn't trip ``argparse.error("invalid choice")``.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# N8 — canonicalize() helper maps legacy aliases to canonical names
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("alias", "expected"), [
    ("claude", "claude-code"),
    ("claude_code", "claude-code"),
    ("claudecode", "claude-code"),
    ("claude-code", "claude-code"),  # canonical → canonical (idempotent)
    ("opencode", "opencode"),
    ("cursor", "cursor"),
    ("codex", "codex"),
])
def test_canonicalize_legacy_alias(alias: str, expected: str) -> None:
    """``canonicalize(alias)`` maps every legacy alias to a canonical name."""
    import harness  # type: ignore[import-not-found]

    assert harness.canonicalize(alias) == expected


def test_canonicalize_passes_through_unknown_name() -> None:
    """An unknown name passes through unchanged — argparse ``choices=``
    is the upstream guard. ``canonicalize`` only collapses known
    aliases; it doesn't validate the result.
    """
    import harness  # type: ignore[import-not-found]

    # Unknown alias not in the map → unchanged.
    assert harness.canonicalize("bogus") == "bogus"


def test_canonicalize_is_case_insensitive() -> None:
    """Case normalisation: ``Claude`` / ``CLAUDE`` map to ``claude-code``."""
    import harness  # type: ignore[import-not-found]

    assert harness.canonicalize("Claude") == "claude-code"
    assert harness.canonicalize("CLAUDE") == "claude-code"


def test_canonicalize_strips_whitespace() -> None:
    """Leading / trailing whitespace is stripped before mapping."""
    import harness  # type: ignore[import-not-found]

    assert harness.canonicalize("  claude  ") == "claude-code"


# --------------------------------------------------------------------------- #
# N7 — setup_task.py and next_step.py both use list(LAUNCHERS) as choices
# --------------------------------------------------------------------------- #


def test_setup_task_choices_match_next_step_choices() -> None:
    """N7: ``--harness`` choices in both scripts come from the same
    source. The dispatch table (``LAUNCHERS``) is authoritative —
    a harness name without a launcher would ``KeyError`` at runtime.
    """
    import next_step  # type: ignore[import-not-found]
    import setup_task  # type: ignore[import-not-found]

    # The two LAUNCHERS dispatch tables are independently defined but
    # logically the same set of harnesses (both map ``claude-code`` /
    # ``cursor`` / ``opencode`` / ``codex`` to the corresponding
    # launcher module).
    assert set(next_step.LAUNCHERS) == set(setup_task.LAUNCHERS), (
        "next_step.LAUNCHERS and setup_task.LAUNCHERS must enumerate "
        "the same set of harness names (review fix #01 N7)."
    )


# --------------------------------------------------------------------------- #
# detect() preserved-behaviour pins
# --------------------------------------------------------------------------- #


def test_detect_explicit_qs_harness_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    """``QS_HARNESS=claude-code`` (canonical) → ``claude-code``."""
    import harness  # type: ignore[import-not-found]

    monkeypatch.setenv("QS_HARNESS", "claude-code")
    assert harness.detect() == "claude-code"


def test_detect_explicit_qs_harness_legacy_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """``QS_HARNESS=claude`` (legacy alias) → ``claude-code``."""
    import harness  # type: ignore[import-not-found]

    monkeypatch.setenv("QS_HARNESS", "claude")
    assert harness.detect() == "claude-code"


def test_detect_defaults_to_claude_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """No harness signals → ``claude-code`` (default)."""
    import harness  # type: ignore[import-not-found]

    for var in (
        "QS_HARNESS",
        "CLAUDECODE",
        "OPENCODE_SERVER_PORT",
        "CURSOR_TRACE_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    # Remove any CODEX_AGENT_* env var
    import os  # noqa: PLC0415
    for k in list(os.environ):
        if k.startswith("CODEX_AGENT_"):
            monkeypatch.delenv(k, raising=False)
    assert harness.detect() == "claude-code"


# --------------------------------------------------------------------------- #
# N8 — harness_choices() includes both canonical names and legacy aliases
# --------------------------------------------------------------------------- #


def test_harness_choices_contains_canonical_and_aliases() -> None:
    """``harness_choices()`` covers every canonical name AND every alias."""
    import harness  # type: ignore[import-not-found]

    choices = harness.harness_choices()
    # Canonical names from VALID_HARNESSES.
    for canonical in harness.VALID_HARNESSES:
        assert canonical in choices, (
            f"harness_choices missing canonical name: {canonical}"
        )
    # Legacy aliases — at minimum the three documented in QS-190 review-fix #01.
    for alias in ("claude", "claude_code", "claudecode"):
        assert alias in choices, (
            f"harness_choices missing legacy alias: {alias} "
            f"(review fix #01 N8 — `--harness claude` must pass argparse)"
        )
