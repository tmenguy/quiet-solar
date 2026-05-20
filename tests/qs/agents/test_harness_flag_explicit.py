"""Pin the AC-1 contract: every next_step.py / setup_task.py callsite
inside .opencode/agents/, .claude/agents/, .cursor/agents/ passes
the explicit ``--harness <name>`` flag matching its directory.

Without this guard, env-var-based harness auto-detection
(scripts/qs/harness.py) silently degrades to the ``claude-code``
default whenever the tool execution environment doesn't carry
``OPENCODE_SERVER_PORT`` / ``CURSOR_TRACE_ID`` — which is precisely
the failure mode QS-190 closes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

HARNESS_BY_DIR = {
    ".opencode/agents": "opencode",
    ".claude/agents": "claude-code",
    ".cursor/agents": "cursor",
}

# Regex matches either next_step.py or setup_task.py followed by any
# characters (DOTALL — so backslash-continued multi-line invocations
# still match), then ``--harness <name>``. The greedy gap is bounded
# by ``.{0,400}`` so the pattern can't drift across an unrelated
# downstream invocation in the same file body.
_CALLSITE_RE_FMT = (
    r"(next_step\.py|setup_task\.py).{{0,400}}?--harness\s+{harness}\b"
)


def _all_agent_files() -> list[tuple[Path, str]]:
    """Yield (file, expected_harness) for every .md file in each harness dir.

    Files that DO NOT contain a next_step.py / setup_task.py
    invocation are skipped — only callsite-bearing files are
    asserted. Today that's 11 files yielding 13 callsites; the helper
    re-scans the live filesystem so new agent files are picked up
    automatically.
    """
    out: list[tuple[Path, str]] = []
    for rel, harness in HARNESS_BY_DIR.items():
        for f in sorted((REPO_ROOT / rel).glob("*.md")):
            body = f.read_text(encoding="utf-8")
            if "next_step.py" in body or "setup_task.py" in body:
                out.append((f, harness))
    return out


@pytest.mark.parametrize(("agent_file", "expected_harness"), _all_agent_files())
def test_callsite_passes_explicit_harness_flag(
    agent_file: Path, expected_harness: str,
) -> None:
    body = agent_file.read_text(encoding="utf-8")
    pattern = re.compile(
        _CALLSITE_RE_FMT.format(harness=re.escape(expected_harness)),
        re.DOTALL,
    )
    assert pattern.search(body), (
        f"{agent_file.relative_to(REPO_ROOT)} contains a next_step.py / "
        f"setup_task.py invocation but no matching "
        f"`--harness {expected_harness}` flag within 400 chars of the "
        f"script name. Env-var harness auto-detection is unreliable "
        f"inside tool execution environments — the flag must be "
        f"explicit (see QS-190 AC-1)."
    )


def test_thirteen_callsites_total() -> None:
    """Sanity-check pin: today there are exactly 13 callsites across the
    three harness dirs (6 OpenCode + 6 Claude + 1 Cursor). If this drifts,
    update the count + AC-1 enumeration in the story.

    Counts the literal invocation prefix ``scripts/qs/next_step.py`` /
    ``scripts/qs/setup_task.py`` so the canonical auto-execute block's
    prose mention of ``next_step.py`` (without the ``scripts/qs/``
    prefix) is excluded — only actual bash invocations are counted.
    """
    total = 0
    for rel in HARNESS_BY_DIR:
        for f in (REPO_ROOT / rel).glob("*.md"):
            body = f.read_text(encoding="utf-8")
            total += body.count("scripts/qs/next_step.py")
            total += body.count("scripts/qs/setup_task.py")
    assert total == 13, (
        f"Expected 13 scripts/qs/next_step.py / scripts/qs/setup_task.py "
        f"callsites across the three harness dirs; found {total}. If "
        f"this changed intentionally, update the count + the AC-1 "
        f"enumeration in docs/stories/QS-190.story.md."
    )
