"""Lock the AC-6 doc structure contract: the new ``Orchestrators are
interactive sessions; sub-agents are parallel fan-out`` section is
positioned immediately after the existing ``Adversarial review`` section
in ``docs/workflow/overview.md``.

This is the regression catch for QS-175 review-fix #9 — without it, the
"immediately after" placement is enforced only by manual review.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OVERVIEW = REPO_ROOT / "docs" / "workflow" / "overview.md"
HARNESS_DOC = REPO_ROOT / "docs" / "workflow" / "harness.md"

# Canonical title of the QS-175 paragraph section in overview.md. Centralised
# here so a doc edit doesn't drift between the multiple assertions that
# reference it (review-fix #02 NTH11).
CANONICAL_QS175_HEADING = (
    "Orchestrators are interactive sessions; sub-agents are parallel fan-out"
)


def _section_headings() -> list[tuple[int, str]]:
    """Return list of ``(line_index, heading_text)`` for every H2 heading."""
    headings: list[tuple[int, str]] = []
    for i, line in enumerate(OVERVIEW.read_text().splitlines()):
        m = re.match(r"^##\s+(.*)$", line)
        if m:
            headings.append((i, m.group(1).strip()))
    return headings


def test_overview_has_canonical_qs175_section() -> None:
    """The canonical paragraph section exists."""
    headings = [h for _, h in _section_headings()]
    target = CANONICAL_QS175_HEADING
    assert target in headings, (
        f"overview.md is missing the canonical QS-175 section. Headings: {headings}"
    )


def test_canonical_section_immediately_follows_adversarial_review() -> None:
    """The new section is placed right after the 'Adversarial review' section."""
    headings = [h for _, h in _section_headings()]
    adversarial_idx = None
    for i, h in enumerate(headings):
        if h.lower().startswith("adversarial review"):
            adversarial_idx = i
            break
    assert adversarial_idx is not None, (
        f"overview.md: 'Adversarial review' section heading not found. "
        f"Headings: {headings}"
    )
    target = CANONICAL_QS175_HEADING
    assert adversarial_idx + 1 < len(headings), (
        "overview.md: no heading after 'Adversarial review'"
    )
    next_heading = headings[adversarial_idx + 1]
    assert next_heading == target, (
        f"AC-6: '{target}' must immediately follow 'Adversarial review', "
        f"got {next_heading!r} instead."
    )


def test_harness_doc_does_not_claim_agent_is_always_emitted() -> None:
    """``harness.md`` must NOT claim every launcher emits ``agent`` (review-fix #03 SF3).

    Codex and OpenCode launchers accept free-form ``--next-cmd`` values
    that don't map to a static phase, so they don't emit an ``agent``
    key in the payload. The contract paragraph must reflect this — a
    blanket "all launchers return agent" claim contradicts
    ``test_codex_passes_known_phase_through_unchanged``.
    """
    body = HARNESS_DOC.read_text()
    # Forbid the over-broad claim. We accept any rewording that doesn't
    # assert "agent" is part of the minimum surface for every launcher.
    forbidden_phrasings = [
        "at minimum `tool`, `agent`, `same_context`, `new_context`",
        "minimum: tool, agent, same_context",
    ]
    for phrase in forbidden_phrasings:
        assert phrase not in body, (
            f"harness.md contains the wrong agent-is-universal claim "
            f"({phrase!r}). Codex/opencode payloads don't include "
            f"``agent`` — see test_codex_passes_known_phase_through_unchanged."
        )


def test_harness_doc_documents_codex_opencode_agent_exception() -> None:
    """``harness.md`` must explicitly note that codex/opencode skip the ``agent`` key."""
    body = HARNESS_DOC.read_text()
    # The doc must say codex and opencode don't emit agent. Tolerant of
    # markdown line wrap via simple whitespace collapse.
    normalized = " ".join(body.split())
    assert "codex" in normalized.lower() and "opencode" in normalized.lower(), (
        "harness.md must name both codex and opencode launchers in the "
        "agent-contract paragraph."
    )
    # Either explicit "do not emit agent" wording or "without agent" form
    # is acceptable; we look for the conceptual marker.
    has_exception_clause = (
        "do NOT emit" in body
        or "do not emit `agent`" in body
        or "without `agent`" in body
        or "skip `agent`" in body
        or "no `agent`" in body
    )
    assert has_exception_clause, (
        "harness.md must explicitly state that Codex / OpenCode launchers "
        "don't emit the ``agent`` key (review-fix #03 SF3)."
    )


def test_overview_documents_claude_desktop_limitation() -> None:
    """AC-8: overview.md (or phase-protocols.md) calls out Desktop's limit."""
    body = OVERVIEW.read_text()
    assert "Claude Desktop" in body and "limitation" in body.lower(), (
        "overview.md is missing the Claude Desktop limitation subsection "
        "(AC-8). The doc must honestly state Desktop has no `--agent` "
        "equivalent and direct users to the slash-command fallback."
    )
    # Must specifically mention pycharm_context as the bridge.
    assert "pycharm_context" in body, (
        "overview.md should mention pycharm_context as the suggested bridge "
        "for users who can't use the CLI launcher directly."
    )
