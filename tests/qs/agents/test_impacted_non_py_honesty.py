"""Pin the honesty of `qs-implement-setup-task`'s `--impacted` prose.

QS-290 (S-4) gave `--impacted` an early exit: a change set with no `.py`
file at all spawns no pytest, no diff-cover and no `git fetch`, and
returns 0. testmon fingerprints AST blocks in `.py` files only, so such
a run could never have selected a test anyway — the exit changed the
cost (10-24 s → two git calls), not the verdict.

That makes one sentence in `qs-implement-setup-task.md` actively
misleading, and it matters *most* in exactly that agent: it is the
dev-environment implement variant, so its change sets are the ones most
likely to be pure non-`.py` (a docs-only edit, an agent-file-only edit).
The agent must not read "`--impacted` guards the tooling" and conclude a
green means something when nothing ran.

The contract pinned here:

1. the agent body must NOT claim, unconditionally, that the tooling's
   own correctness is guarded by `--impacted`'s testmon-selected tests;
2. it must state that a non-`.py` change set checks nothing;
3. it must name `--quick tests/qs` as the verification in that case,
   not merely as a supplement.

Mirrored across all three harnesses (Claude / Cursor / OpenCode) so an
edit to one harness alone fails, matching the harness-sync rule in
`docs/workflow/project-rules.md`.

Pattern follows `test_doc_maintenance_parity.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

HARNESS_DIRS: tuple[Path, ...] = (
    REPO_ROOT / ".claude" / "agents",
    REPO_ROOT / ".cursor" / "agents",
    REPO_ROOT / ".opencode" / "agents",
)

AGENT_NAME = "qs-implement-setup-task"

# The exact wording that QS-290 falsified. Kept verbatim as a tripwire: a
# revert or a copy-paste from an older harness copy re-introduces it.
FALSIFIED_CLAIM = (
    "but the gate/tooling's own correctness is\nstill guarded by its "
    "testmon-selected tests under `--impacted`"
)


def _harness_id(p: Path) -> str:
    return p.parent.name.lstrip(".")


def _body(harness_dir: Path) -> str:
    path = harness_dir / f"{AGENT_NAME}.md"
    assert path.is_file(), f"Missing agent file: {path}"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_does_not_claim_impacted_guards_the_tooling(harness_dir: Path) -> None:
    """The unconditional "`--impacted` guards the tooling" claim must be gone."""
    assert FALSIFIED_CLAIM not in _body(harness_dir), (
        f"{harness_dir / f'{AGENT_NAME}.md'}: this claim is false for a "
        "change set with no `.py` file — QS-290's early exit runs nothing "
        "at all there. Say it is guarded only when the change set DOES "
        "contain a `.py` file."
    )


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_states_non_py_change_sets_check_nothing(harness_dir: Path) -> None:
    """The agent must know that a non-`.py` green means nothing ran."""
    body = _body(harness_dir)
    assert "no `.py` file at all" in body, (
        f"{harness_dir / f'{AGENT_NAME}.md'}: must state that a change set "
        "with no `.py` file makes `--impacted` exit early."
    )
    assert "check\n**nothing**" in body or "check **nothing**" in body, (
        f"{harness_dir / f'{AGENT_NAME}.md'}: must say the non-`.py` run "
        "checks NOTHING — 'fast no-op' alone reads as 'cheap but still valid'."
    )


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_names_quick_tests_qs_as_the_verification(harness_dir: Path) -> None:
    """`--quick tests/qs` is THE check for a non-`.py` change set."""
    body = _body(harness_dir)
    assert "not a supplement" in body, (
        f"{harness_dir / f'{AGENT_NAME}.md'}: must promote `--quick tests/qs` "
        "from supplement to sole verification for non-`.py` change sets."
    )
    assert "quality_gate.py --quick tests/qs" in body, (
        f"{harness_dir / f'{AGENT_NAME}.md'}: the runnable command must stay present."
    )
