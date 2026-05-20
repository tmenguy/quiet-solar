"""Pin the AC-1 contract: every next_step.py / setup_task.py callsite
inside .opencode/agents/, .claude/agents/, .cursor/agents/ passes
the explicit ``--harness <name>`` flag matching its directory.

Without this guard, env-var-based harness auto-detection
(scripts/qs/harness.py) silently degrades to the ``claude-code``
default whenever the tool execution environment doesn't carry
``OPENCODE_SERVER_PORT`` / ``CURSOR_TRACE_ID`` — which is precisely
the failure mode QS-190 closes.

Review fix #01 — S2 + S3 + S10 + N11:

- The check is now **code-fence-scoped**: we extract ``` ```bash ``` ```
  fenced blocks from each agent .md, and for every block that contains
  ``scripts/qs/next_step.py`` or ``scripts/qs/setup_task.py`` we
  require the *same* block to also contain ``--harness <expected>``.
  This eliminates the regex's 400-char cross-contamination risk where
  callsite #1 in a multi-callsite file could pair with callsite #2's
  flag (S10).
- The ``scripts/qs/`` prefix is required, so prose mentions of
  ``next_step.py`` in surrounding text don't count as callsites (S2).
- The 13-callsites-total sanity-pin is replaced by a per-file "no
  callsite missing its flag" assertion (S3); the sanity-pin now only
  requires ``>= 13`` so a future legitimate addition doesn't break
  the test.
- The AC-1 enumeration table in
  ``docs/stories/QS-190.story.md`` is the authoritative reference
  (N11) — see the table headed
  "**Given** any of the **13** ``next_step.py`` / ``setup_task.py``
  callsites" for the source of truth.
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

# Matches a ``` ```bash ... ``` ``` fenced block whose opening and
# closing fences are both at start-of-line (no leading indent). The
# ``bash`` language tag is required so other fenced blocks (e.g.
# ``` ```text ``` ```) aren't picked up as bash callsites. The
# anchoring is needed because indented fences (e.g. a ``` ```bash ``` ```
# nested inside a numbered list — 3-space indent) would otherwise
# confuse the closing-fence match: an indented opening followed by an
# indented closing has whitespace between the newline and the
# backticks, which breaks a naive ``\n```\`` close pattern; line-anchoring
# both ends makes the match unambiguous (review-fix #01 S10).
_BASH_FENCE_RE = re.compile(
    r"^```bash\n(.*?)\n^```$",
    re.DOTALL | re.MULTILINE,
)

# A callsite is a ``scripts/qs/(next_step|setup_task).py`` literal inside
# a ``bash`` fence. The ``scripts/qs/`` prefix is required so prose
# mentions of the bare script name don't count.
_CALLSITE_RE = re.compile(r"scripts/qs/(?:next_step|setup_task)\.py")


def _bash_fences(body: str) -> list[str]:
    """Return the body of every ``` ```bash ``` ``` fenced block in ``body``."""
    return _BASH_FENCE_RE.findall(body)


def _callsite_fences(body: str) -> list[str]:
    """Return ``bash``-fenced blocks that actually invoke a callsite."""
    return [fence for fence in _bash_fences(body) if _CALLSITE_RE.search(fence)]


def _all_agent_files() -> list[tuple[Path, str]]:
    """Yield (file, expected_harness) for every callsite-bearing .md file."""
    out: list[tuple[Path, str]] = []
    for rel, harness in HARNESS_BY_DIR.items():
        for f in sorted((REPO_ROOT / rel).glob("*.md")):
            body = f.read_text(encoding="utf-8")
            if _callsite_fences(body):
                out.append((f, harness))
    return out


@pytest.mark.parametrize(("agent_file", "expected_harness"), _all_agent_files())
def test_every_callsite_fence_carries_explicit_harness_flag(
    agent_file: Path, expected_harness: str,
) -> None:
    """Per-file pin (S2 + S3 + S10): every callsite fence in this file
    contains the matching ``--harness <name>`` flag inside the same
    fence. Counts callsites missing the flag and asserts zero.
    """
    body = agent_file.read_text(encoding="utf-8")
    flag_re = re.compile(
        rf"--harness\s+{re.escape(expected_harness)}\b",
    )
    callsite_count = 0
    missing_count = 0
    for fence in _callsite_fences(body):
        # A single fence can carry MULTIPLE callsites (today none do, but
        # a future refactor might consolidate). Count each callsite
        # occurrence and require the flag to appear at least once per
        # callsite in the same fence.
        callsites_in_fence = len(_CALLSITE_RE.findall(fence))
        flags_in_fence = len(flag_re.findall(fence))
        callsite_count += callsites_in_fence
        if flags_in_fence < callsites_in_fence:
            missing_count += callsites_in_fence - flags_in_fence

    assert callsite_count > 0, (
        f"{agent_file.relative_to(REPO_ROOT)}: contained no "
        f"scripts/qs/next_step.py / setup_task.py callsite — was this "
        f"file picked up correctly?"
    )
    assert missing_count == 0, (
        f"{agent_file.relative_to(REPO_ROOT)}: {missing_count} of "
        f"{callsite_count} callsite(s) in this file are missing a "
        f"matching `--harness {expected_harness}` flag inside the "
        f"SAME ```bash``` fence. Env-var harness auto-detection is "
        f"unreliable inside tool execution environments — the flag "
        f"must be explicit and co-located with the script invocation "
        f"(QS-190 AC-1; review-fix #01 S10)."
    )


def test_aggregate_callsites_at_least_thirteen() -> None:
    """Sanity-pin (S3): the AC-1 enumeration table lists 13 callsites.

    Today there are exactly 13 (6 OpenCode + 6 Claude + 1 Cursor); a
    future legitimate addition should not break this test as long as
    the new callsite ALSO carries its ``--harness`` flag — that
    invariant is enforced per-file by
    ``test_every_callsite_fence_carries_explicit_harness_flag``.

    Authoritative reference: the AC-1 enumeration table in
    ``docs/stories/QS-190.story.md`` (look for the row table headed
    "Given any of the 13 next_step.py / setup_task.py callsites").
    """
    total = 0
    for rel in HARNESS_BY_DIR:
        for f in (REPO_ROOT / rel).glob("*.md"):
            body = f.read_text(encoding="utf-8")
            for fence in _callsite_fences(body):
                total += len(_CALLSITE_RE.findall(fence))
    assert total >= 13, (
        f"Expected at least 13 scripts/qs/next_step.py / setup_task.py "
        f"callsites across the three harness dirs; found {total}. The "
        f"AC-1 enumeration table in docs/stories/QS-190.story.md lists "
        f"the canonical 13."
    )
