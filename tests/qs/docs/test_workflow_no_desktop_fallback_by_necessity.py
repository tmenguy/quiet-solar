"""QS-311 AC9: the "by necessity" inference stays retracted.

`overview.md` used to enumerate three mechanisms the Claude Code GUI
lacks (URL scheme, CLI argument pass-through, a persona-preloading UI
gesture) and conclude that GUI users therefore "land on the
slash-command fallback path **by necessity**".

The enumeration is **true** and is deliberately NOT banned here — a
future author must stay free to state it, and `harness.md`'s GUI section
does exactly that. What is false is the conclusion: a fourth mechanism
exists, the `agent` key in `.claude/settings.local.json`, which binds a
phase orchestrator in the GUI. This test pins only the retracted
inference so it cannot creep back into the routing prose.

Scope notes:

- Whitespace-normalised bodies are **required**, not cosmetic: the
  original sentence wrapped between "fallback path" and "by necessity",
  so a naive substring scan passed before the edit and proved nothing.
- `docs/stories/**` is excluded. Stories are immutable records of what
  was believed at the time; `harness.md` is the single current source of
  truth (risk R4).
- `AGENTS.md` and `.claude/commands/**` are verified free of the phrase
  today and are left out of the glob to keep the net minimal.
- Review-fix #01 N1: the banned string is the fuller
  `"fallback path by necessity"`, not a bare `"by necessity"`. The bare
  form is ordinary English ("the solver is iterative by necessity") and
  banning it across every workflow doc would false-positive on a future
  sentence that has nothing to do with launch surfaces. The anchor is
  still specific enough that the retracted sentence cannot return —
  it *is* the sentence, verbatim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# The retracted inference. Lower-cased comparison so "By necessity" at the
# start of a sentence cannot slip through; anchored on the full clause so
# an unrelated "by necessity" elsewhere is not a false positive (N1).
RETRACTED_INFERENCE = "fallback path by necessity"


def _scanned_files() -> list[Path]:
    """Return the workflow docs plus `CLAUDE.md` — the surface-routing prose.

    Recursive glob (QS-332): `docs/workflow/lanes/*.md` are full copies
    of `phase-protocols.md` and future divergent lane protocols — a
    non-recursive glob would silently skip them.
    """
    files = sorted((REPO_ROOT / "docs" / "workflow").rglob("*.md"))
    files.append(REPO_ROOT / "CLAUDE.md")
    return files


def test_scan_set_is_non_empty() -> None:
    """Guard the guard: a glob that matches nothing would pass vacuously."""
    files = _scanned_files()
    assert len(files) >= 2, f"unexpected scan set: {files}"
    assert all(path.is_file() for path in files), f"missing file in {files}"


@pytest.mark.parametrize(
    "doc_path", _scanned_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_doc_does_not_claim_gui_users_are_forced_onto_the_fallback(
    doc_path: Path,
) -> None:
    """No workflow doc concludes the slash fallback is used "by necessity"."""
    normalized = " ".join(doc_path.read_text(encoding="utf-8").split()).lower()
    assert RETRACTED_INFERENCE not in normalized, (
        f"{doc_path.relative_to(REPO_ROOT)} says {RETRACTED_INFERENCE!r}. "
        f"The Claude Code GUI can run a phase orchestrator directly via the "
        f"`agent` key in `.claude/settings.local.json` — see harness.md → "
        f"'GUI launch surface (Claude Code Desktop)'. Enumerating what the "
        f"GUI genuinely lacks is fine; concluding the fallback is forced is "
        f"not (QS-311 AC9)."
    )
