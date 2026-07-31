"""QS-311 AC1: `.claude/settings.local.json` is gitignored — narrowly.

The launcher writes the per-worktree GUI phase pin into
`<worktree>/.claude/settings.local.json` (see
`scripts/qs/launchers/claude.py::_write_phase_agent`). That file is
machine-written, per-developer, and must never be committed — while the
rest of `.claude/` (agents, commands, `settings.json`) is *tracked* and
must stay tracked.

These assertions read `.gitignore` **as text** rather than calling
`git check-ignore`: `check-ignore` also consults the user's global
excludes (`~/.config/git/ignore`), where the pattern happens to be
present on the author's machine (finding F11), so a `check-ignore`
based test would pass spuriously on a repo that never gained the entry.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GITIGNORE = REPO_ROOT / ".gitignore"

# The exact pattern AC1 requires.
REQUIRED_PATTERN = ".claude/settings.local.json"

# Patterns that would ignore *more* of `.claude/` than intended. The
# tracked agent/command files must never become invisible to git. This
# blacklist is the checkable form of "narrow" — no gitignore-semantics
# matcher is implemented here.
FORBIDDEN_PATTERNS = (
    ".claude",
    ".claude/",
    ".claude/*",
    ".claude/*.json",
)


def _patterns() -> list[str]:
    """Return stripped, non-comment, non-empty lines of `.gitignore`."""
    lines = GITIGNORE.read_text(encoding="utf-8").splitlines()
    return [
        stripped
        for stripped in (line.strip() for line in lines)
        if stripped and not stripped.startswith("#")
    ]


def test_gitignore_ignores_claude_local_settings() -> None:
    """`.claude/settings.local.json` appears verbatim as a pattern."""
    patterns = _patterns()
    assert REQUIRED_PATTERN in patterns, (
        f"{REQUIRED_PATTERN!r} missing from .gitignore — the launcher's "
        f"per-worktree GUI phase pin would be offered for commit "
        f"(QS-311 AC1). Patterns: {patterns}"
    )


def test_gitignore_does_not_ignore_the_whole_claude_directory() -> None:
    """No broad `.claude` pattern hides the tracked agent/command files."""
    patterns = _patterns()
    for forbidden in FORBIDDEN_PATTERNS:
        assert forbidden not in patterns, (
            f".gitignore contains the over-broad pattern {forbidden!r}. "
            f"`.claude/agents/`, `.claude/commands/` and "
            f"`.claude/settings.json` are tracked; only "
            f"{REQUIRED_PATTERN!r} may be ignored (QS-311 AC1)."
        )
