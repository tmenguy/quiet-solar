"""QS-367 N9: pin AC8's model-policy doc tokens so they are not grep-only.

The story's AC8 tokens were verified by hand during review. This test
locks them so a future edit cannot silently drop the GUI model-picker
prose from ``harness.md`` or revert the "five classes" wording — the same
pinned-docs style as the sibling tests in ``tests/qs/docs/``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

HARNESS_MD = REPO_ROOT / "docs" / "workflow" / "harness.md"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
PROJECT_RULES_MD = REPO_ROOT / "docs" / "workflow" / "project-rules.md"


def test_harness_md_carries_gui_model_prose() -> None:
    """``harness.md`` names ``phase_model`` and the ``model picker`` gesture."""
    body = HARNESS_MD.read_text()
    assert "phase_model" in body, "harness.md must name `phase_model` (QS-367 E7)."
    assert "model picker" in body, "harness.md must name the `model picker` (QS-367 E7)."


def test_harness_md_omits_settings_pinned_model_claim() -> None:
    """The GUI model is picker-decided, never settings-pinned (QS-367 E7)."""
    body = HARNESS_MD.read_text()
    assert "settings-pinned" not in body, (
        "harness.md must not describe the GUI model as `settings-pinned` — "
        "the picker decides it, not the settings file (QS-367 E7)."
    )


def test_class_count_docs_say_five_not_four() -> None:
    """``CLAUDE.md`` and ``project-rules.md`` say five classes, never four."""
    for doc in (CLAUDE_MD, PROJECT_RULES_MD):
        lowered = doc.read_text().lower()
        assert "five classes" in lowered, (
            f"{doc.name} must say 'five classes' (QS-367 added the `build` class)."
        )
        assert "four classes" not in lowered, (
            f"{doc.name} still says 'four classes' — there are five (QS-367)."
        )
