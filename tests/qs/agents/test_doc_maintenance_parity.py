"""Pin the doc-maintenance step across the four orchestrator agents.

QS-185 introduced an addressable documentation hierarchy under
``docs/agents/`` plus a drift-checker script
(``scripts/qs/check_doc_drift.py``). For the drift checker to be
useful, four agent bodies must invoke it at the right phase:

- ``qs-create-plan``      — during plan drafting, run
  ``check_doc_drift.py --paths <planned_files>``.
- ``qs-implement-task``   — at the quality gate, run
  ``check_doc_drift.py`` on the staged diff.
- ``qs-implement-setup-task`` — same.
- ``qs-review-task``      — during consolidation, flag a must-fix
  finding if the PR shows no ``## Doc maintenance`` justification
  and ``check_doc_drift.py`` would have failed.

Body content is mirrored across three harnesses (Claude / Cursor /
OpenCode) — the frontmatter format is harness-specific and stays
untouched. This test pins the invocation token in every harness so
a future edit that only touches one harness fails CI.

Pattern follows ``test_opencode_agents.py`` (the existing static-
agent contract test).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# The four orchestrator agents that AC-9 requires us to update.
DOC_MAINTENANCE_AGENT_NAMES: tuple[str, ...] = (
    "qs-create-plan",
    "qs-implement-task",
    "qs-implement-setup-task",
    "qs-review-task",
)

# Three harnesses to mirror across.
HARNESS_DIRS: tuple[Path, ...] = (
    REPO_ROOT / ".claude" / "agents",
    REPO_ROOT / ".cursor" / "agents",
    REPO_ROOT / ".opencode" / "agents",
)

# The exact token that every agent body must contain. The trailing
# space-and-no-arg form lets each agent customize its own invocation
# (``--paths``, ``--json``, on-the-staged-diff, etc.) without
# fragmenting the contract.
DRIFT_CHECKER_TOKEN: str = "scripts/qs/check_doc_drift.py"


# Implement-variant agents (run between TDD red-green-refactor and
# the commit): the doc-maintenance step must instruct the agent to
# stage first, then run the checker. Pinned post-review-fix #01 S1.
IMPLEMENT_AGENT_NAMES: tuple[str, ...] = (
    "qs-implement-task",
    "qs-implement-setup-task",
)

# Review-variant agents (post-PR): the doc-maintenance step inspects
# the PR diff, not a "staged" diff (no canonical staged state exists
# at review time). Pinned post-review-fix #01 S1.
REVIEW_AGENT_NAMES: tuple[str, ...] = ("qs-review-task",)


# Review-fix #01 N6: strip the leading dot from harness ids so
# ``pytest -k claude`` works without quoting concerns. Without
# ``lstrip('.')``, the id is ``.claude`` and the dot is treated as a
# regex meta-char in ``-k`` expressions.
def _harness_id(p: Path) -> str:
    return p.parent.name.lstrip(".")


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
@pytest.mark.parametrize("agent_name", DOC_MAINTENANCE_AGENT_NAMES)
def test_agent_body_invokes_drift_checker(harness_dir: Path, agent_name: str) -> None:
    """Every orchestrator agent in every harness mentions the drift checker."""
    path = harness_dir / f"{agent_name}.md"
    assert path.is_file(), f"Missing agent file: {path}"
    body = path.read_text(encoding="utf-8")
    assert DRIFT_CHECKER_TOKEN in body, (
        f"{path}: agent body does not invoke '{DRIFT_CHECKER_TOKEN}'. "
        f"QS-185 AC-9 / AC-10 require every orchestrator agent to wire "
        f"the doc-maintenance step into its phase protocol; this test "
        f"pins the contract across .claude/, .cursor/, and .opencode/."
    )


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
@pytest.mark.parametrize("agent_name", IMPLEMENT_AGENT_NAMES)
def test_implement_agents_say_after_staging(harness_dir: Path, agent_name: str) -> None:
    """Implement-variant agents must say "After staging", not "Before staging".

    Review-fix #01 S1: a pre-stage drift check sees an empty cached
    diff and silently no-ops. The contract is now: stage first
    (``git add``), then run the checker.
    """
    body = (harness_dir / f"{agent_name}.md").read_text(encoding="utf-8")
    assert "After staging" in body, (
        f"{harness_dir / f'{agent_name}.md'}: implement-variant agents "
        "must instruct the agent to stage *first* and then run the "
        "drift checker (review-fix #01 S1). Look for 'After staging'."
    )
    assert "Before staging," not in body, (
        f"{harness_dir / f'{agent_name}.md'}: the legacy 'Before "
        "staging,' wording silently no-ops because the cached diff "
        "is empty (review-fix #01 S1)."
    )


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
@pytest.mark.parametrize("agent_name", REVIEW_AGENT_NAMES)
def test_review_agents_avoid_pr_staged_diff(harness_dir: Path, agent_name: str) -> None:
    """Review-variant agents must reference "PR diff", not "PR's staged diff".

    Review-fix #01 S1: there is no canonical staged state at review
    time; the reviewer evaluates the PR diff directly.
    """
    body = (harness_dir / f"{agent_name}.md").read_text(encoding="utf-8")
    assert "PR's staged diff" not in body, (
        f"{harness_dir / f'{agent_name}.md'}: 'PR's staged diff' is "
        "meaningless at review time (review-fix #01 S1). Reference "
        "the PR diff directly."
    )
    assert "PR diff" in body, (
        f"{harness_dir / f'{agent_name}.md'}: review-variant agents "
        "must explain the doc-maintenance audit against the PR diff "
        "(review-fix #01 S1)."
    )
