"""QS-357 AC12: no doc links to a rendered agent file.

The harness agent files under ``.claude/agents/`` and ``.opencode/agents/``
are gitignored rendered outputs (QS-357) — a Markdown link to one would
dangle on a fresh clone. The tracked source of truth is a template under
``scripts/qs/agent_templates/``; docs must link there (or not at all).

This scans ``docs/workflow/**``, ``docs/agents/**`` and ``CLAUDE.md`` for
Markdown link targets pointing at ``.claude/agents/*.md`` /
``.opencode/agents/*.md`` and fails on any hit — guarding the 40+ link
repoints of this task against regression.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Markdown link target that lands on a rendered agent file, at any depth
# of ``../`` prefix.
_AGENT_LINK_RE = re.compile(r"\]\([^)]*\.(?:claude|opencode)/agents/[^)]+\.md\)")


def _scanned_files() -> list[Path]:
    files: list[Path] = [REPO_ROOT / "CLAUDE.md"]
    for root in ("docs/workflow", "docs/agents"):
        files.extend(sorted((REPO_ROOT / root).rglob("*.md")))
    return [f for f in files if f.is_file()]


@pytest.mark.parametrize("path", _scanned_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_links_to_rendered_agent_files(path: Path) -> None:
    hits = _AGENT_LINK_RE.findall(path.read_text(encoding="utf-8"))
    assert not hits, (
        f"{path.relative_to(REPO_ROOT)} links to a rendered agent file "
        f"{hits}; link to the template under scripts/qs/agent_templates/ "
        f"instead (QS-357 AC12)."
    )
