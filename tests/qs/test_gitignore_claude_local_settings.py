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

Review-fix #01 M2: the pattern carries a trailing `*` so it also covers
the writer's **siblings** — the atomic-write temp
(`settings.local.json.<pid>.tmp`). `finally` cleanup covers Ctrl-C and
`SystemExit` but not `SIGKILL`, an OOM kill, or power loss, so a temp can
survive; un-ignored it would be swept into the next commit by
`utils.auto_commit_and_push` (which stages `.claude/` wholesale) and would
make `qs-finish-task` prompt "Force-delete and lose this work?" over a
stray temp file.

`settings.local.json.bak` stays in the covered list even though review-fix
#03 removed the code that created it: worktrees pinned by an earlier
revision can still hold one, and it must not become a phantom untracked
file there.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GITIGNORE = REPO_ROOT / ".gitignore"

# The exact pattern AC1 requires (review-fix #01 M2: plus the trailing
# ``*`` so the writer's temp / backup siblings are covered too).
REQUIRED_PATTERN = ".claude/settings.local.json*"

# Filenames the pattern must cover. The bare settings file is AC1's
# original requirement; the other two are the writer's siblings.
COVERED_FILENAMES = (
    ".claude/settings.local.json",
    ".claude/settings.local.json.12345.tmp",
    ".claude/settings.local.json.bak",
)

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
    """`.claude/settings.local.json*` appears verbatim as a pattern."""
    patterns = _patterns()
    assert REQUIRED_PATTERN in patterns, (
        f"{REQUIRED_PATTERN!r} missing from .gitignore — the launcher's "
        f"per-worktree GUI phase pin would be offered for commit "
        f"(QS-311 AC1). Patterns: {patterns}"
    )


@pytest.mark.parametrize("filename", COVERED_FILENAMES)
def test_gitignore_pattern_covers_the_writer_siblings(filename: str) -> None:
    """Each file the pin writer can leave behind is matched by some pattern.

    Matched with `fnmatch` against the patterns actually read from
    `.gitignore` — not against `REQUIRED_PATTERN` — so the assertion tests
    the file rather than restating this module's own constant. `fnmatch`
    is not a gitignore engine (its `*` crosses `/`), which is harmless for
    a literal path carrying a single trailing `*`.
    """
    matching = [p for p in _patterns() if fnmatch.fnmatch(filename, p)]
    assert matching, (
        f"{filename!r} is matched by no .gitignore pattern — the pin "
        f"writer's leftovers (atomic-write temp, rebuild backup) would be "
        f"offered for commit, then swept into the next commit by "
        f"`auto_commit_and_push` (review-fix #01 M2)."
    )


def test_gitignore_has_no_double_trailing_blank_line() -> None:
    """The file ends with exactly one newline — no trailing blank line.

    Review-fix #01 N8. The blank line *before* the `# QS-311` comment
    separates it from the previous stanza and stays.
    """
    text = GITIGNORE.read_text(encoding="utf-8")
    assert text.endswith("\n"), ".gitignore must end with a newline"
    assert not text.endswith("\n\n"), (
        ".gitignore ends with a blank line (double trailing newline)."
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
