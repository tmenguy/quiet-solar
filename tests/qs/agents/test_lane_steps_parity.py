"""QS-332: pin the three lane step kinds across the three harnesses.

Canonical list (review I-2/R2-08 — kept consistent with story task 11
and AC-5):

(a) the **declaration step** — including the amended speed-rule wording —
    in ``qs-setup-task`` ×3;
(b) the **lane-read step** in the 4 orchestrators
    (qs-create-plan, qs-implement-task, qs-implement-setup-task,
    qs-review-task) ×3;
(c) the **ask-and-backfill-on-declaration-FAIL step** in the implement
    variants ×2 ×3.

There is deliberately NO Lane-note relay step — surfacing the crossing
in the PR body is machine-owned by ``create_pr.py`` (review N-4), pinned
in ``tests/qs/test_create_pr.py``. ``qs-finish-task`` is deliberately
excluded from (b): it has no lane-sensitive behaviour while lanes are
identical (story D2, review SG2-03).

Pattern follows ``test_doc_maintenance_parity.py``.
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

LANE_READ_AGENT_NAMES: tuple[str, ...] = (
    "qs-create-plan",
    "qs-implement-task",
    "qs-implement-setup-task",
    "qs-review-task",
    "qs-diagnose-task",
    "qs-verify-task",
)

IMPLEMENT_AGENT_NAMES: tuple[str, ...] = (
    "qs-implement-task",
    "qs-implement-setup-task",
)


def _harness_id(p: Path) -> str:
    return p.parent.name.lstrip(".")


def _body(harness_dir: Path, agent_name: str) -> str:
    path = harness_dir / f"{agent_name}.md"
    assert path.is_file(), f"Missing agent file: {path}"
    return path.read_text(encoding="utf-8")


# --- (a) the declaration step in qs-setup-task ------------------------------


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_setup_task_carries_the_declaration_step(harness_dir: Path) -> None:
    body = _body(harness_dir, "qs-setup-task")
    assert "Lane declaration" in body
    # Existing-issue path: use-if-complete / ask-only-missing.
    assert "declaration_complete" in body
    assert "exactly the missing axes" in body
    # New-issue path: the labels passthrough and the six options.
    assert "--labels" in body
    assert "harness feature" in body
    # The optional piggybacked epic question (review SG-C3).
    assert "part of an epic?" in body
    # The bright-line trigger (review PC-08): explicit words only.
    assert "explicit lane name" in body


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_setup_task_epic_lane_creates_no_worktree(harness_dir: Path) -> None:
    """Review-fix #04 (must-fix): step 2 used to run `setup_task.py`
    unconditionally, cutting a branch + worktree even for an epic lane —
    contradicting the epic model this very PR establishes ("No implement
    phase; no branch, worktree, or PR"; output = a rationale doc on
    `main` + child issues). The prompt must stop before step 2 for an
    epic; `setup_task.py` enforces the same invariant machine-side."""
    # Whitespace-normalised: the prose wraps mid-clause, so a naive
    # substring scan would pass vacuously (the trap
    # `test_workflow_no_desktop_fallback_by_necessity.py` documents).
    body = " ".join(_body(harness_dir, "qs-setup-task").split())
    assert "no branch, worktree, or PR" in body, (
        f"{harness_dir / 'qs-setup-task.md'}: the epic lane must not reach "
        "the branch/worktree step"
    )
    # The terminal state is named, so the agent doesn't improvise one.
    assert "child issues" in body
    # And the step-2 header states the precondition it now carries.
    assert "Set up branch and worktree + emit launcher (tasks only)" in body


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_setup_task_speed_rule_is_amended(harness_dir: Path) -> None:
    """The old absolute speed rule would contradict the one permitted
    lane/epic question (review planner R2, story task 11)."""
    body = _body(harness_dir, "qs-setup-task")
    assert "except" in body and "the single lane/epic question" in body
    assert "The launcher must come within a few seconds" not in body


# --- (b) the lane-read step in the 4 orchestrators --------------------------


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
@pytest.mark.parametrize("agent_name", LANE_READ_AGENT_NAMES)
def test_orchestrators_read_their_lane_file(harness_dir: Path, agent_name: str) -> None:
    body = _body(harness_dir, agent_name)
    assert "docs/workflow/lanes/<lane>.md" in body, (
        f"{harness_dir / f'{agent_name}.md'}: missing the lane-read step "
        "(read docs/workflow/lanes/<lane>.md, <lane> from context.py)"
    )
    # Empty-lane fallback for pre-existing worktrees / legacy tasks.
    assert "phase-protocols.md" in body


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
@pytest.mark.parametrize("agent_name", LANE_READ_AGENT_NAMES)
def test_lane_block_is_a_clean_mirrored_paragraph(
    harness_dir: Path, agent_name: str
) -> None:
    """Review-fix #01: the lane-read block must sit between exactly one
    blank line on each side in every copy — some copies had a doubled
    blank above and NO blank below (two rendered paragraphs merged).
    These files are pinned mirrors; whitespace drift is exactly the
    class the parity tests exist to prevent, and the substring pins
    above don't see it."""
    body = _body(harness_dir, agent_name)
    start = body.index("**Lane (QS-332).**")
    assert body[start - 2 : start] == "\n\n", "one blank line before the block"
    assert body[start - 3 : start] != "\n\n\n", "no doubled blank line before"
    tail = body[start:]
    # Review-fix #05: qs-diagnose-task's block gained the empty-lane
    # story-overwrite guard, so its terminal sentence is now the guard's
    # ("...convergence overwrites it."), not "on the fallback.".
    sentinel = next(
        marker
        for marker in (
            "a parallel path).\n",
            "convergence overwrites it.\n",
            "on the fallback.\n",
        )
        if marker in tail
    )
    end = tail.index(sentinel) + len(sentinel)
    assert tail[end] == "\n", "one blank line after the block"
    assert tail[end : end + 2] != "\n\n", "no doubled blank line after"


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_finish_task_is_deliberately_not_wired(harness_dir: Path) -> None:
    """qs-finish-task has no lane-sensitive behaviour while lanes are
    identical — wiring it now would be a step with no reader (SG2-03).
    A lane PR that diverges finish behaviour adds the wiring then."""
    body = _body(harness_dir, "qs-finish-task")
    assert "docs/workflow/lanes/<lane>.md" not in body


# --- (c) ask-and-backfill in the implement variants -------------------------


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
@pytest.mark.parametrize("agent_name", IMPLEMENT_AGENT_NAMES)
def test_implement_variants_carry_ask_and_backfill(
    harness_dir: Path, agent_name: str
) -> None:
    body = _body(harness_dir, agent_name)
    assert "lane check FAILED" in body, (
        f"{harness_dir / f'{agent_name}.md'}: missing the "
        "ask-and-backfill-on-declaration-FAIL step"
    )
    assert "gh issue edit" in body
    assert "re-run the gate" in body
    # Review-fix #01: the gate's remediation may include `--remove-label`
    # lines and user-chosen substitutions — the step must say "apply the
    # remediation", not "run the exact --add-label command" (which is no
    # longer always the printed shape).
    assert "apply the remediation the gate printed" in body
    assert "run the exact `gh issue edit <N> --add-label ...` command" not in body
