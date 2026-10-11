"""Lock the AC-6 doc structure contract: the new ``Orchestrators are
interactive sessions; sub-agents are parallel fan-out`` section is
positioned immediately after the existing ``Adversarial review`` section
in ``docs/workflow/overview.md``.

This is the regression catch for QS-175 review-fix #9 — without it, the
"immediately after" placement is enforced only by manual review.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

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


# --------------------------------------------------------------------------- #
# QS-417 -- the Mermaid diagram of the static-agent pipeline.
#
# "The six phases" keeps its one-line text sketch (the default path) and
# adds, right after it, one Mermaid flowchart with the lane variants, the
# review/implement fix loop and a hand (maintainer touchpoint) line in each
# phase node. The test pins the placement, the lead-in and the diagram's
# full structure: every line must parse (_parse_pipeline_diagram), and the
# node set, the phase names in each node, the hand in each node, the edges
# with their arrow style and label, and the subgraph membership are all
# compared with the constants below (review fix #01). A link line inside a
# subgraph is rejected, the hand must open its own line and every subgraph
# title is checked (review fix #02). A `~~~` endpoint or a `style` target
# must be a declared node or subgraph id (review fix #03). It does NOT
# detect a phase added to the pipeline later, nor check label wording
# beyond the phase names and the hand: both stay a review responsibility.
# --------------------------------------------------------------------------- #

# Node id -> the phase names its label must contain.
PIPELINE_DIAGRAM_NODE_PHASES: dict[str, tuple[str, ...]] = {
    "setup": ("setup-task",),
    "plan": ("create-plan",),
    "diag": ("diagnose-task",),
    "dec": ("decompose-epic",),
    "impl": ("implement-task", "implement-setup-task"),
    "rev": ("review-task",),
    "ver": ("verify-task",),
    "fin": ("finish-task",),
    "rel": ("release",),
}

PIPELINE_DIAGRAM_NODES = set(PIPELINE_DIAGRAM_NODE_PHASES)

# (src, dst) -> (arrow, label), exactly as in overview.md. "\u00d7" is the
# multiplication sign of the lane names. It is escaped because ruff's
# confusable-character rules (RUF001/RUF003) flag a literal one; the only
# non-ASCII literal this QS-417 block adds is the hand.
PIPELINE_DIAGRAM_EDGES: dict[tuple[str, str], tuple[str, str | None]] = {
    ("setup", "plan"): ("-->", None),
    ("setup", "diag"): ("-->", None),
    ("setup", "dec"): ("-->", None),
    ("plan", "impl"): ("-->", "story committed"),
    ("diag", "impl"): ("-->", "diagnosis committed"),
    ("impl", "rev"): ("-->", "PR open"),
    ("impl", "ver"): ("-->", "PR open (bug \u00d7 product)"),
    ("rev", "impl"): ("-->", "review-fix plan #NN"),
    ("ver", "impl"): ("-->", "review-fix plan #NN"),
    ("rev", "fin"): ("-->", "nothing to fix / all skipped"),
    ("ver", "fin"): ("-->", "nothing to fix / all skipped"),
    ("diag", "fin"): ("-->", "superseded / no defect"),
    ("dec", "fin"): ("-->", "epic doc landed"),
    ("fin", "rel"): ("-.->", "separate decision"),
}

# Subgraph id -> its member node ids.
PIPELINE_DIAGRAM_SUBGRAPHS = {
    "STD": {"plan", "rev"},
    "BUG": {"diag", "ver"},
    "EPIC": {"dec"},
}

# Subgraph id -> ASCII substrings its title must contain.
PIPELINE_DIAGRAM_SUBGRAPH_TITLE_WORDS = {
    "STD": ("feature", "bug", "factory"),
    "BUG": ("bug", "product"),
    "EPIC": ("epic", "factory"),
}

PIPELINE_DIAGRAM_LEAD_IN = "The sketch above is the default path."

_DIAGRAM_HEADER_RE = re.compile(r"^flowchart (TB|LR|TD|BT|RL)$")
_DIAGRAM_NODE_RE = re.compile(r'^(\w+)\["([^"]*)"\]$')
_DIAGRAM_EDGE_RE = re.compile(r'^(\w+)\s*(-->|-\.->)(?:\|"([^"]*)"\|)?\s*(\w+)$')
_DIAGRAM_SUBGRAPH_RE = re.compile(r'^subgraph (\w+)\["([^"]*)"\]$')
# `~~~` is an invisible layout link: it changes placement only, so it is
# allowed at top level and not pinned as an edge. Like any link line, it is
# rejected inside a subgraph (it would pull its endpoints into that box).
_DIAGRAM_INVISIBLE_LINK_RE = re.compile(r"^(\w+)\s*~~~\s*(\w+)$")
# `style` creates a vertex when its target does not exist, so its target is
# captured and checked like a `~~~` endpoint (review fix #03).
_DIAGRAM_STYLE_RE = re.compile(r"^style (\w+) .+$")
# Lines that are recognised but carry nothing the test pins: classDef,
# class (it creates no node) and a comment.
_DIAGRAM_IGNORED_RES = (
    re.compile(r"^classDef \w+ .+$"),
    re.compile(r"^class [\w,]+ \w+$"),
    re.compile(r"^%%"),
)


class PipelineDiagram(NamedTuple):
    """The parsed structure of the pipeline diagram."""

    nodes: dict[str, str]
    edges: dict[tuple[str, str], tuple[str, str | None]]
    subgraphs: dict[str, tuple[str, set[str]]]


def _parse_pipeline_diagram(block: str) -> PipelineDiagram:
    """Classify every non-blank line of the Mermaid block.

    Raises ``AssertionError`` on a line of no known kind, a missing
    header, a duplicate node or edge, a nested, unopened or unclosed
    subgraph, or a link line (edge or ``~~~``) inside a subgraph: Mermaid
    makes both endpoints of such a line members of that subgraph. Also
    raises when a ``~~~`` endpoint or a ``style`` target is not a declared
    node or subgraph id: Mermaid would draw it as a ghost node.
    """
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    assert lines and _DIAGRAM_HEADER_RE.match(lines[0]), (
        f"The pipeline diagram must open with a Mermaid `flowchart` header "
        f"(QS-417), got {lines[:1]!r}."
    )
    nodes: dict[str, str] = {}
    edges: dict[tuple[str, str], tuple[str, str | None]] = {}
    subgraphs: dict[str, tuple[str, set[str]]] = {}
    current: str | None = None
    # ``~~~`` endpoints and ``style`` targets, checked once all ids are known.
    references: list[tuple[str, str]] = []
    for line in lines[1:]:
        if m := _DIAGRAM_SUBGRAPH_RE.match(line):
            assert current is None, f"nested subgraph: {line!r}"
            assert m[1] not in subgraphs, f"duplicate subgraph: {line!r}"
            current = m[1]
            subgraphs[current] = (m[2], set())
        elif line == "end":
            assert current is not None, "`end` with no open subgraph"
            current = None
        elif m := _DIAGRAM_NODE_RE.match(line):
            assert m[1] not in nodes, f"duplicate node declaration: {line!r}"
            nodes[m[1]] = m[2]
            if current is not None:
                subgraphs[current][1].add(m[1])
        elif m := _DIAGRAM_EDGE_RE.match(line):
            assert current is None, (
                f"links must be declared outside subgraphs: {line!r}"
            )
            key = (m[1], m[4])
            assert key not in edges, f"duplicate edge: {line!r}"
            edges[key] = (m[2], m[3])
        elif m := _DIAGRAM_INVISIBLE_LINK_RE.match(line):
            assert current is None, (
                f"links must be declared outside subgraphs: {line!r}"
            )
            references += [(m[1], line), (m[2], line)]
        elif m := _DIAGRAM_STYLE_RE.match(line):
            references.append((m[1], line))
        elif not any(rx.match(line) for rx in _DIAGRAM_IGNORED_RES):
            raise AssertionError(f"unrecognised diagram line: {line!r}")
    assert current is None, f"subgraph {current!r} is never closed"
    for ref, line in references:
        assert ref in nodes or ref in subgraphs, (
            f"link to an undeclared id: {ref!r} in {line!r}"
        )
    return PipelineDiagram(nodes, edges, subgraphs)


def _assert_pipeline_diagram(block: str) -> None:
    """Parse the block and compare it with the pinned structure."""
    parsed = _parse_pipeline_diagram(block)

    assert set(parsed.nodes) == PIPELINE_DIAGRAM_NODES, (
        f"Pipeline diagram node ids drifted (QS-417): got "
        f"{sorted(parsed.nodes)}, expected {sorted(PIPELINE_DIAGRAM_NODES)}."
    )
    for node_id, label in parsed.nodes.items():
        assert "<br/>✋" in label, (
            f"Pipeline diagram node {node_id!r} has no maintainer "
            f"touchpoint on its own line (QS-417)."
        )
        for phase in PIPELINE_DIAGRAM_NODE_PHASES[node_id]:
            assert re.search(rf"(?<![\w-]){re.escape(phase)}(?![\w-])", label), (
                f"Pipeline diagram node {node_id!r} does not name the "
                f"{phase!r} phase (QS-417)."
            )

    assert parsed.edges == PIPELINE_DIAGRAM_EDGES, (
        f"Pipeline diagram edges drifted (QS-417): got {parsed.edges}, "
        f"expected {PIPELINE_DIAGRAM_EDGES}."
    )

    membership = {sg: members for sg, (_, members) in parsed.subgraphs.items()}
    assert membership == PIPELINE_DIAGRAM_SUBGRAPHS, (
        f"Pipeline diagram subgraph membership drifted (QS-417): got "
        f"{membership}, expected {PIPELINE_DIAGRAM_SUBGRAPHS}."
    )
    for sg, words in PIPELINE_DIAGRAM_SUBGRAPH_TITLE_WORDS.items():
        title = parsed.subgraphs[sg][0].lower()
        for word in words:
            assert word in title, (
                f"Pipeline diagram subgraph {sg!r} title {title!r} does not "
                f"name {word!r} (QS-417)."
            )


def _six_phases_section() -> str:
    """Return the text of the ``## The six phases`` section of overview.md."""
    body = OVERVIEW.read_text(encoding="utf-8")
    heading = "## The six phases"
    assert heading in body, f"overview.md is missing the {heading!r} section."
    start = body.index(heading) + len(heading)
    m = re.search(r"^## ", body[start:], re.M)
    return body[start : start + m.start()] if m else body[start:]


def _mermaid_blocks(section: str) -> list[str]:
    """Return the bodies of the ```mermaid blocks in *section*."""
    return re.findall(r"^```mermaid\n(.*?)^```", section, re.S | re.M)


def test_six_phases_section_has_pipeline_diagram() -> None:
    """QS-417: one Mermaid flowchart after the text sketch, fully pinned."""
    section = _six_phases_section()
    blocks = _mermaid_blocks(section)
    assert len(blocks) == 1, (
        f"overview.md 'The six phases' must hold exactly one ```mermaid "
        f"block (QS-417), found {len(blocks)}."
    )

    assert "```text" in section, (
        "The one-line ```text sketch must stay in 'The six phases' (QS-417)."
    )
    text_idx = section.index("```text")
    mermaid_idx = section.index("```mermaid")
    assert text_idx < mermaid_idx, (
        "The Mermaid diagram must come after the ```text sketch (QS-417)."
    )
    assert PIPELINE_DIAGRAM_LEAD_IN in section[text_idx:mermaid_idx], (
        f"The lead-in paragraph ({PIPELINE_DIAGRAM_LEAD_IN!r}) must sit "
        f"between the text sketch and the Mermaid diagram (QS-417)."
    )

    _assert_pipeline_diagram(blocks[0])


def _move_ver_out_of_bug(block: str) -> str:
    """Remove the ``ver`` node line and re-add it after the last ``end``."""
    line = next(ln for ln in block.splitlines(keepends=True) if "ver[" in ln)
    head, sep, tail = block.replace(line, "").rpartition("    end\n")
    return head + sep + line + tail


def _move_setup_edge_into_std(block: str) -> str:
    """Move the ``setup --> plan`` line to just before the first ``end``."""
    line = "    setup --> plan\n"
    return block.replace(line, "").replace("    end\n", line + "    end\n", 1)


# Each case mutates the real block once; the strengthened checks must
# reject every one of them (review fix #01). The first nine are the fix
# plan's required cases, the rest cover the parser's other rejections.
PIPELINE_DIAGRAM_MUTATIONS: dict[str, Callable[[str], str]] = {
    "solid-separate-decision": lambda b: b.replace("fin -.->", "fin -->"),
    "dashed-decompose-edge": lambda b: b.replace("dec -->", "dec -.->"),
    "dropped-fix-plan-label": lambda b: b.replace(
        'rev -->|"review-fix plan #NN"| impl', "rev --> impl"
    ),
    "thick-arrow-edge": lambda b: b + "    rel ==> setup\n",
    "chained-edge": lambda b: b + "    rel --> setup --> plan\n",
    "round-node-shape": lambda b: b + '    x("stray")\n',
    "ver-moved-out-of-bug-lane": _move_ver_out_of_bug,
    "phase-name-only-in-comment": lambda b: (
        b.replace('rel["release ', 'rel["') + "    %% release\n"
    ),
    # Declared first, so a last-wins dict would keep the real label.
    "duplicate-node-declaration": lambda b: b.replace(
        "flowchart TB\n", 'flowchart TB\n    setup["setup-task"]\n'
    ),
    "ampersand-edge": lambda b: b + "    rev & ver --> rel\n",
    "dash-text-label": lambda b: b + "    rel -- again --> setup\n",
    "semicolon-edge": lambda b: b.replace(
        "setup --> plan\n", "setup --> plan;\n"
    ),
    "diamond-node-shape": lambda b: b + "    x{stray}\n",
    "duplicate-edge": lambda b: b + "    setup --> plan\n",
    "missing-header": lambda b: b.replace("flowchart TB\n", ""),
    "nested-subgraph": lambda b: b.replace(
        '    subgraph BUG["', '    subgraph X["x"]\n    subgraph BUG["'
    ),
    "duplicate-subgraph": lambda b: b + '    subgraph BUG["bug"]\n    end\n',
    "stray-end": lambda b: b + "    end\n",
    "unclosed-subgraph": lambda b: b + '    subgraph X["x"]\n',
    # Review fix #02: a link line inside a subgraph makes both endpoints
    # members of that subgraph, so the parser rejects it.
    "edge-inside-subgraph": _move_setup_edge_into_std,
    "invisible-link-inside-subgraph": lambda b: b.replace(
        "    end\n", "    impl ~~~ rev\n    end\n", 1
    ),
    "hand-not-on-own-line": lambda b: b.replace(
        "<br/>✋ merge authorization", " ✋ merge authorization"
    ),
    "std-title-drift": lambda b: re.sub(
        r'subgraph STD\["[^"]*"\]', 'subgraph STD["default lanes"]', b
    ),
    # Review fix #03: Mermaid draws a ghost node for an undeclared id
    # named by a `~~~` link or a `style` statement.
    "invisible-link-unknown-endpoint": lambda b: b + "    plan ~~~ ghost\n",
    "style-unknown-id": lambda b: b.replace("style EPIC ", "style EPCI "),
}


@pytest.mark.parametrize("case", sorted(PIPELINE_DIAGRAM_MUTATIONS))
def test_pipeline_diagram_rejects_mutation(case: str) -> None:
    """QS-417 review fix #01: each structural drift fails the pin."""
    (block,) = _mermaid_blocks(_six_phases_section())
    mutated = PIPELINE_DIAGRAM_MUTATIONS[case](block)
    assert mutated != block, f"Mutation {case!r} no longer applies (stale case)."
    with pytest.raises(AssertionError):
        _assert_pipeline_diagram(mutated)


def test_pipeline_diagram_accepts_top_level_invisible_link() -> None:
    """QS-417 review fix #02: a top-level ``~~~`` stays allowed, not an edge."""
    (block,) = _mermaid_blocks(_six_phases_section())
    with_link = block + "    plan ~~~ diag\n"
    _assert_pipeline_diagram(with_link)
    parsed = _parse_pipeline_diagram(with_link)
    assert parsed.edges == _parse_pipeline_diagram(block).edges
