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

import pytest

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
    # is acceptable; we look for the conceptual marker. Normalize to
    # lowercase so case variations like "Do not emit `agent`" don't make
    # the test brittle (review-fix #04 NTH10).
    body_lower = body.lower()
    has_exception_clause = any(
        candidate in body_lower
        for candidate in (
            "do not emit",
            "without `agent`",
            "skip `agent`",
            "no `agent`",
        )
    )
    assert has_exception_clause, (
        "harness.md must explicitly state that Codex / OpenCode launchers "
        "don't emit the ``agent`` key (review-fix #03 SF3)."
    )


# --------------------------------------------------------------------------- #
# QS-311 AC7 — the GUI launch-surface section.
#
# The section is deliberately NOT a harness identifier: the GUI shares
# `.claude/` wholesale, so `claude-gui` would be a category error (one
# harness == one agent directory). Its content is prose and therefore
# reviewer-verified; only the four tokens below are gate-checked, because
# a section that omits any of them fails to answer the question a GUI
# user actually arrives with.
# --------------------------------------------------------------------------- #

GUI_SECTION_HEADING = "GUI launch surface (Claude Code Desktop)"

GUI_SECTION_REQUIRED_TOKENS = (
    # the mechanism
    "settings.local.json",
    # the CLI precedence rule that makes the pin inert for `claude --agent`
    "--agent",
    # the hybrid CLI→GUI bridge
    "/desktop",
    # The main-checkout gap: those phases are never pinned. Review-fix #01
    # N3: this used to be the token `"release"`, which any mention of a
    # release anywhere in the section would satisfy — it pinned nothing.
    "main checkout",
    # Review-fix #01 S7: the pin is gitignored by design, so it does NOT
    # follow into a sub-worktree the GUI creates for itself. The pre-PR
    # setup-task prose acknowledged that mode and the PR deleted it; the
    # Traps list has to carry it instead.
    "isolation",
    # Review-fix #02 E: by the section's own stated mechanism, a HEADLESS
    # invocation (`claude -p …`, an Agent-SDK run) with cwd inside a pinned
    # worktree inherits the pin too — with no CLI header to reveal it. The
    # worst case is a `qs-finish-task` pin, which merges PRs and removes
    # worktrees.
    "headless",
    # Review-fix #03 B2/C4: a symlinked pin file is REFUSED (following it
    # would move the write outside the worktree, breaking guard 2's
    # containment invariant). One of the two states that now silently
    # produce no pin, so the section has to name it.
    "symlink",
)


def _gui_section_body() -> str:
    """Return the text of the GUI launch-surface section of harness.md."""
    body = HARNESS_DOC.read_text()
    heading = f"## {GUI_SECTION_HEADING}"
    assert heading in body, (
        f"harness.md is missing the {heading!r} section (QS-311 AC7). "
        f"The GUI is a launch surface of the Claude harness, so it is "
        f"documented here rather than as a new harness identifier."
    )
    start = body.index(heading) + len(heading)
    rest = body[start:]
    next_h2 = re.search(r"^## ", rest, re.MULTILINE)
    return rest[: next_h2.start()] if next_h2 else rest


def test_harness_doc_has_gui_launch_surface_section() -> None:
    """The section exists and is an H2 (asserted inside the helper)."""
    assert _gui_section_body().strip(), (
        f"harness.md's {GUI_SECTION_HEADING!r} section is empty."
    )


@pytest.mark.parametrize("token", GUI_SECTION_REQUIRED_TOKENS)
def test_gui_launch_surface_section_names_required_tokens(token: str) -> None:
    """Each load-bearing token appears in the GUI section.

    Whitespace-normalised: these are prose tokens and the section
    line-wraps at 72 columns, so a multi-word token like ``main checkout``
    would otherwise fail purely on where the wrap landed. Case-normalised
    for the same reason — a token that happens to open a sentence or a
    bolded Traps lead-in is capitalised, and which of those a sentence
    lands on is not the property being pinned.
    """
    section = " ".join(_gui_section_body().split()).lower()
    token = token.lower()
    assert token in section, (
        f"harness.md's {GUI_SECTION_HEADING!r} section does not mention "
        f"{token!r} (QS-311 AC7)."
    )


def test_gui_section_is_not_advertised_as_a_harness_identifier() -> None:
    """The bare token ``claude-gui`` must never appear as a harness value.

    Decision 1: the GUI is a launch surface, not a harness. Naming
    ``claude-gui`` anywhere invites a reader (or a future agent) to pass it
    to ``--harness``, where it resolves to nothing.
    """
    body = HARNESS_DOC.read_text()
    assert "claude-gui" not in body, (
        "harness.md names `claude-gui` — there is no such harness. GUI "
        "sessions use `--harness claude-code` (QS-311 Decision 1)."
    )


def test_overview_documents_claude_desktop_limitation() -> None:
    """AC-8: overview.md (or phase-protocols.md) calls out Desktop's limit.

    The three tokens below must all stay present. Review-fix #01 S10: the
    failure message used to add "**and direct users to the slash-command
    fallback**" — the very inference AC8 retracts and AC9 bans. What AC8
    requires is that the *limitation* stays documented (there is still no
    way to launch a GUI session programmatically) alongside the fourth
    mechanism that makes the fallback unnecessary.

    Review-fix #01 N2: the ``"by necessity"`` ban that used to live here
    too now has a dedicated owner —
    ``test_workflow_no_desktop_fallback_by_necessity.py`` — which scans
    every workflow doc rather than this one file.
    """
    body = OVERVIEW.read_text()
    assert "Claude Desktop" in body and "limitation" in body.lower(), (
        "overview.md is missing the Claude Desktop limitation subsection "
        "(AC-8). The doc must honestly state that Desktop offers no way to "
        "*launch* a session on a directory programmatically — while naming "
        "the `agent` settings key that binds the orchestrator once a GUI "
        "session is open."
    )
    # Must specifically mention pycharm_context as the bridge.
    assert "pycharm_context" in body, (
        "overview.md should mention pycharm_context as the suggested bridge "
        "for users who can't use the CLI launcher directly."
    )
    # Whitespace-normalised because this paragraph line-wraps mid-phrase.
    normalized = " ".join(body.split())
    assert "settings.local.json" in normalized, (
        "overview.md must name the fourth mechanism — the `agent` key in "
        "`.claude/settings.local.json` — alongside the three that really "
        "are missing (URL scheme, argv pass-through, UI gesture)."
    )
