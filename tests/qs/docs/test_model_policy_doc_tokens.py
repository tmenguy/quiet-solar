"""QS-367 N9 / S3: pin AC8's model-policy doc tokens so they are not grep-only.

The story's AC8 tokens were verified by hand during review. This test locks
the concrete ones so a future edit cannot silently drop them. What it pins:

* ``harness.md`` — ``phase_model`` and ``model picker`` (the GUI gesture),
  ``Five classes``, the ``build`` model IDs ``claude-opus-5-5`` /
  ``claude-opus-5.5``, the ```build``` class token, and the CLI floor
  ``2.1.280``;
* ``docs/agents/glossary.md`` — the ```build``` class token;
* ``CLAUDE.md`` / ``project-rules.md`` — "five classes" (never "four");
* and, negatively, that ``harness.md`` never describes the GUI model as
  ``settings-pinned`` nor claims frontmatter decides it ``on every surface``.

Every check runs over whitespace-normalised text (``" ".join(body.split())``)
so a re-wrapped phrase can neither hide a required token nor slip a banned
claim through. The same pinned-docs style as the sibling tests in
``tests/qs/docs/``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

HARNESS_MD = REPO_ROOT / "docs" / "workflow" / "harness.md"
GLOSSARY_MD = REPO_ROOT / "docs" / "agents" / "glossary.md"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
PROJECT_RULES_MD = REPO_ROOT / "docs" / "workflow" / "project-rules.md"


def _normalised(path: Path) -> str:
    """Return ``path``'s text with every whitespace run collapsed to a space."""
    return " ".join(path.read_text().split())


def test_harness_md_carries_gui_model_prose() -> None:
    """``harness.md`` names ``phase_model`` and the ``model picker`` gesture."""
    body = _normalised(HARNESS_MD)
    assert "phase_model" in body, "harness.md must name `phase_model` (QS-367 E7)."
    assert "model picker" in body, "harness.md must name the `model picker` (QS-367 E7)."


def test_harness_md_carries_ac8_build_tokens() -> None:
    """``harness.md`` names the ``build`` class, its model IDs, and the CLI floor."""
    body = _normalised(HARNESS_MD)
    for token in (
        "Five classes",
        "claude-opus-5-5",
        "claude-opus-5.5",
        "`build`",
        "2.1.280",
    ):
        assert token in body, (
            f"harness.md must name AC8 token {token!r} (QS-367 S3)."
        )


def test_glossary_carries_build_class_token() -> None:
    """``glossary.md`` names the ``build`` model class (AC8)."""
    body = _normalised(GLOSSARY_MD)
    assert "`build`" in body, (
        "glossary.md must name the `build` model class (QS-367 S3)."
    )


def test_harness_md_omits_settings_pinned_model_claim() -> None:
    """The GUI model is picker-decided, never settings-pinned (QS-367 E7)."""
    body = _normalised(HARNESS_MD)
    assert "settings-pinned" not in body, (
        "harness.md must not describe the GUI model as `settings-pinned` — "
        "the picker decides it, not the settings file (QS-367 E7)."
    )
    assert "on every surface" not in body, (
        "harness.md must not claim frontmatter decides the model `on every "
        "surface` — the GUI main session's model comes from the picker, not "
        "frontmatter (QS-367 S3)."
    )


def test_class_count_docs_say_five_not_four() -> None:
    """``CLAUDE.md`` and ``project-rules.md`` say five classes, never four."""
    for doc in (CLAUDE_MD, PROJECT_RULES_MD):
        lowered = _normalised(doc).lower()
        assert "five classes" in lowered, (
            f"{doc.name} must say 'five classes' (QS-367 added the `build` class)."
        )
        assert "four classes" not in lowered, (
            f"{doc.name} still says 'four classes' — there are five (QS-367)."
        )
