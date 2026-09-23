"""QS-367 N9 / S3: pin AC8's model-policy doc tokens so they are not grep-only.

The story's AC8 tokens were verified by hand during review. This test locks
the concrete ones so a future edit cannot silently drop them. What it pins:

* ``harness.md`` — ``phase_model`` and ``model picker`` (the GUI gesture),
  ``Five classes``, the ``deep`` model IDs ``claude-opus-5-5`` /
  ``claude-opus-5.5``, the ```build``` class token with its ``build`` model
  IDs ``claude-opus-4-8`` / ``claude-opus-4.8``, and the CLI floor
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


def test_harness_md_carries_ac8_model_tokens() -> None:
    """``harness.md`` names the ``deep`` model IDs, the ``build`` row, and the CLI floor.

    The ``deep`` IDs (``claude-opus-5-5`` / ``claude-opus-5.5``) are what
    ``HARNESS_MODELS`` maps to the ``deep`` class; the ``build`` row is the
    ```build``` class token together with its ``claude-opus-4-8`` /
    ``claude-opus-4.8`` model IDs. Both are pinned so an edit cannot drop
    either class's IDs (QS-367 S1/S3).
    """
    body = _normalised(HARNESS_MD)
    for token in (
        "Five classes",
        "claude-opus-5-5",  # deep model IDs
        "claude-opus-5.5",
        "`build`",  # build class token + its model IDs
        "claude-opus-4-8",
        "claude-opus-4.8",
        "2.1.280",
    ):
        assert token in body, (
            f"harness.md must name AC8 token {token!r} (QS-367 S3)."
        )


def test_glossary_carries_build_class_token() -> None:
    """``glossary.md`` names the ``build`` model class and never says "four classes"."""
    body = _normalised(GLOSSARY_MD)
    assert "`build`" in body, (
        "glossary.md must name the `build` model class (QS-367 S3)."
    )
    # N3: ban the stale count here too — the negative was previously only
    # guarded in CLAUDE.md / project-rules.md.
    assert "four classes" not in body.lower(), (
        "glossary.md still says 'four classes' — there are five (QS-367)."
    )


def test_harness_md_omits_settings_pinned_model_claim() -> None:
    """The GUI model is picker-decided, never settings-pinned; never "four classes"."""
    # N3: lowercase before every negative check so a re-cased phrase (the
    # `main` text literally read "Four classes") cannot slip a banned claim
    # through.
    body = _normalised(HARNESS_MD).lower()
    assert "settings-pinned" not in body, (
        "harness.md must not describe the GUI model as `settings-pinned` — "
        "the picker decides it, not the settings file (QS-367 E7)."
    )
    assert "on every surface" not in body, (
        "harness.md must not claim frontmatter decides the model `on every "
        "surface` — the GUI main session's model comes from the picker, not "
        "frontmatter (QS-367 S3)."
    )
    assert "four classes" not in body, (
        "harness.md still says 'four classes' — there are five (QS-367)."
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
