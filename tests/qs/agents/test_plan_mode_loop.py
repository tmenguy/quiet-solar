"""Pin the QS-266 interactive plan-mode loop across all three harnesses.

QS-266 reshapes ``qs-create-plan`` from a linear pipeline into a
user-driven **mode loop** (DISCUSS / REVIEW / TRIAGE / FINALIZE) and
adds a fifth, diff-aware plan reviewer (``qs-plan-delta-auditor``).

The repo tests agent bodies by **static substring pinning** (see the
sibling ``test_doc_maintenance_parity.py``), not by executing the
personas. Each QS-266 acceptance criterion (AC1–AC6) is therefore a
literal marker that must appear in every harness copy of
``qs-create-plan`` — and AC7 is the existence of the new sub-agent in
all three harness directories.

Pattern follows ``test_doc_maintenance_parity.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Three harnesses to mirror across.
HARNESS_DIRS: tuple[Path, ...] = (
    REPO_ROOT / ".claude" / "agents",
    REPO_ROOT / ".cursor" / "agents",
    REPO_ROOT / ".opencode" / "agents",
)

# Markers that pin AC1–AC6 in the body of every ``qs-create-plan`` copy.
# Each entry is a literal substring that the mode-loop rewrite must
# contain. They are byte-aligned across harnesses (the core protocol
# body differs only in frontmatter + the harness-specific spawn/handoff
# and LSP sections — see project-rules.md "Harness sync").
CREATE_PLAN_REQUIRED_MARKERS: tuple[str, ...] = (
    # AC1 — mode loop present.
    "## Modes",
    "DISCUSS",
    "REVIEW",
    "TRIAGE",
    "FINALIZE",
    # AC2 — early persistence rule.
    "committed only at FINALIZE",
    # AC3 — status banner template.
    "[DISCUSS] story vN · changed-since-last-review:",
    "changed-since-last-review",
    # AC4 — three intents.
    "three intents",
    # AC5 — delta-auditor wired (round asymmetry + in-context diff +
    # finding-state model).
    "qs-plan-delta-auditor",
    "in-context diff",
    "open/resolved/rejected",
    # AC6 — advisory finalize.
    "never hard-block",
    "confirm before FINALIZE",
    "ship anyway",
)

# AC2 / AC4 — strings that the rewrite must REMOVE. The old linear
# protocol told the agent to defer writing the story (contradicts early
# persistence, F10) and there are no scoped/quick reviews (F3).
CREATE_PLAN_FORBIDDEN_MARKERS: tuple[str, ...] = (
    "Do NOT write yet",
    "do NOT write the file yet",
    "review scope",
    "quick check",
)

# AC7 — the new diff-aware reviewer exists in all three harnesses.
DELTA_AUDITOR_NAME = "qs-plan-delta-auditor"


# AC8 — the three workflow docs must describe the 5th reviewer and the
# round asymmetry (review-fix #01 should-fix). ``check_doc_drift.py``
# only enforces co-modification (that the docs were *edited* alongside
# the agents), not content, so without these pins a future edit could
# silently drop the round-asymmetry paragraph. Per-doc marker sets
# (each doc phrases it slightly differently — see the doc bodies).
DOC_AC8_MARKERS: dict[str, tuple[str, ...]] = {
    "docs/workflow/overview.md": (
        "qs-plan-delta-auditor",
        "Round 2+",
        "diff-aware",
        "in-context diff",
    ),
    "docs/workflow/adversarial-review.md": (
        "qs-plan-delta-auditor",
        "Round asymmetry",
        "round 2+",
        "diff-aware",
    ),
    "docs/workflow/phase-protocols.md": (
        "qs-plan-delta-auditor",
        "round 2+",
        "in-context diff",
    ),
}


def _harness_id(p: Path) -> str:
    # Mirror test_doc_maintenance_parity: strip the leading dot so the
    # parametrize id is ``claude`` rather than ``.claude`` (a leading dot
    # is awkward in test-selection filters). Run these via
    # ``python scripts/qs/quality_gate.py --quick tests/qs/agents`` — the
    # single test entry point.
    return p.parent.name.lstrip(".")


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
@pytest.mark.parametrize("marker", CREATE_PLAN_REQUIRED_MARKERS)
def test_create_plan_body_contains_mode_loop_marker(
    harness_dir: Path, marker: str,
) -> None:
    """Every ``qs-create-plan`` copy pins the AC1–AC6 mode-loop markers."""
    path = harness_dir / "qs-create-plan.md"
    assert path.is_file(), f"Missing agent file: {path}"
    body = path.read_text(encoding="utf-8")
    assert marker in body, (
        f"{path}: missing QS-266 mode-loop marker {marker!r}. AC1–AC6 are "
        f"pinned as literal substrings in every harness copy of "
        f"qs-create-plan; the mode-loop rewrite must keep this marker aligned "
        f"across .claude/, .cursor/, and .opencode/."
    )


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
@pytest.mark.parametrize("marker", CREATE_PLAN_FORBIDDEN_MARKERS)
def test_create_plan_body_drops_linear_pipeline_marker(
    harness_dir: Path, marker: str,
) -> None:
    """The linear-pipeline / scoped-review strings are gone (AC2/AC4, F10/F3)."""
    body = (harness_dir / "qs-create-plan.md").read_text(encoding="utf-8")
    assert marker not in body, (
        f"{harness_dir / 'qs-create-plan.md'}: stale string {marker!r} must be "
        f"removed. QS-266 replaces the 'draft in memory, do not write yet' "
        f"linear protocol with early persistence (F10) and cuts scoped / "
        f"quick-check reviews (F3)."
    )


@pytest.mark.parametrize(
    ("doc_rel", "markers"),
    list(DOC_AC8_MARKERS.items()),
    # Explicit id list (one per doc) — a callable ``ids`` is invoked
    # per-arg (doc_rel AND markers), so the markers tuple rendered an
    # empty trailing-dash segment (review-fix #02). The doc path alone
    # is the clean, filterable id.
    ids=list(DOC_AC8_MARKERS.keys()),
)
def test_workflow_docs_describe_fifth_reviewer(
    doc_rel: str, markers: tuple[str, ...],
) -> None:
    """AC8 — the workflow docs pin the 5th reviewer + round asymmetry.

    Content pin (review-fix #01): ``check_doc_drift.py`` enforces only
    that these docs are co-edited with the agents, not what they say.
    These markers fail loudly if the round-asymmetry / delta-auditor
    prose is dropped on a future edit.
    """
    path = REPO_ROOT / doc_rel
    assert path.is_file(), f"Missing workflow doc: {path}"
    body = path.read_text(encoding="utf-8")
    for marker in markers:
        assert marker in body, (
            f"{doc_rel}: missing AC8 marker {marker!r} — the doc must keep "
            f"describing the 5th reviewer (qs-plan-delta-auditor) and the "
            f"round-1 vs round-2+ asymmetry."
        )


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_delta_auditor_exists_in_every_harness(harness_dir: Path) -> None:
    """AC7 — ``qs-plan-delta-auditor`` ships in all three harness dirs."""
    path = harness_dir / f"{DELTA_AUDITOR_NAME}.md"
    assert path.is_file(), (
        f"Missing new sub-agent file: {path}. AC7 requires "
        f"qs-plan-delta-auditor in .claude/, .cursor/, and .opencode/ "
        f"(parity enforced by this pinning test, not the drift checker — a "
        f"brand-new single-harness agent is exempt from co-modification)."
    )
