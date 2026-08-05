"""QS-332: the 6 workflow lanes exist and are identical by construction.

Identity is ENFORCED, divergence is an allowlist (review PC-01/PC-03):
each lane file must be byte-identical to ``phase-protocols.md`` unless
its basename is in ``DIVERGED``. The set is empty at QS-332 merge; each
lane PR's (#335-#340) first act is adding its own file here. This makes
AC-1 machine-verified and closes the 7-copy silent-drift window between
this PR and the last lane PR.

Known, pre-authorized cosmetic cost (review DP2-06): relative links
inside the copies resolve wrong one directory deeper. Do NOT "fix" them
while a lane is in this identity set — that breaks identity; a lane PR
may fix its own file's links when it diverges.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LANES_DIR = REPO_ROOT / "docs" / "workflow" / "lanes"
PHASE_PROTOCOLS = REPO_ROOT / "docs" / "workflow" / "phase-protocols.md"

# Lane name = {kind}-{target} for tasks, epic-{target} for epics — the
# 6 lanes of epic #321.
LANES = (
    "bug-product",
    "bug-factory",
    "feature-product",
    "feature-factory",
    "epic-product",
    "epic-factory",
)

# Lane files allowed to diverge from phase-protocols.md. EMPTY at QS-332
# merge; a lane PR adds its own basename (e.g. "bug-product.md") as its
# first act, then writes its divergence.
DIVERGED: frozenset[str] = frozenset()


def test_exactly_the_six_lane_files_exist() -> None:
    assert sorted(p.name for p in LANES_DIR.glob("*.md")) == sorted(
        f"{lane}.md" for lane in LANES
    )


def test_phase_protocols_still_exists() -> None:
    """The source stays in place and stays authoritative until lanes
    diverge (its removal would break the step-5 string pin in
    ``tests/test_quality_gate.py`` and the qs-review-task reference)."""
    assert PHASE_PROTOCOLS.is_file()


@pytest.mark.parametrize("lane", LANES)
def test_lane_file_is_byte_identical_unless_diverged(lane: str) -> None:
    lane_file = LANES_DIR / f"{lane}.md"
    if lane_file.name in DIVERGED:
        pytest.skip(f"{lane_file.name} has diverged (allowlisted)")
    assert lane_file.read_bytes() == PHASE_PROTOCOLS.read_bytes(), (
        f"docs/workflow/lanes/{lane}.md is not byte-identical to "
        f"phase-protocols.md. Until a lane PR adds it to DIVERGED, every "
        f"lane file is an exact copy — divergence happens one PR per lane "
        f"(#335-#340), whose first act is the allowlist entry."
    )


def test_diverged_entries_are_real_lane_files() -> None:
    """A typo'd allowlist entry would silently skip nothing."""
    assert DIVERGED <= {f"{lane}.md" for lane in LANES}
