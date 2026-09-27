"""QS-332: pin the three lane step kinds across both harnesses.

Canonical list (review I-2/R2-08 — kept consistent with story task 11
and AC-5):

(a) the **declaration step** — including the amended speed-rule wording —
    in ``qs-setup-task`` ×2;
(b) the **lane-read step** in the 4 orchestrators
    (qs-create-plan, qs-implement-task, qs-implement-setup-task,
    qs-review-task) ×2;
(c) the **ask-and-backfill-on-declaration-FAIL step** in the implement
    variants ×2 ×2.

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

from tests.qs.agents._rendered import agents_dir

REPO_ROOT = Path(__file__).resolve().parents[3]

HARNESS_DIRS: tuple[Path, ...] = (
    agents_dir("claude"),
    agents_dir("opencode"),
)

LANE_READ_AGENT_NAMES: tuple[str, ...] = (
    "qs-create-plan",
    "qs-implement-task",
    "qs-implement-setup-task",
    "qs-review-task",
    "qs-diagnose-task",
    "qs-verify-task",
    "qs-decompose-epic",
)

IMPLEMENT_AGENT_NAMES: tuple[str, ...] = (
    "qs-implement-task",
    "qs-implement-setup-task",
)

# The lane-read block's terminal sentence, per agent (review-fix #06):
# the previous first-match tuple scanned the whole file remainder, so a
# marker phrase appearing in a later paragraph of another agent's body
# could hijack the block boundary and make the whitespace assertions
# inspect arbitrary bytes (vacuous pass or spurious fail). The
# implement variants end on the before-first-commit rule; diagnose-task
# ends on the lane-independent story-overwrite guard's cold-start
# resume sentence (review-fix #05/#06/#07); the rest end on the
# read-only fallback sentence.
LANE_BLOCK_TERMINALS: dict[str, str] = {
    "qs-create-plan": "on the fallback.\n",
    "qs-implement-task": "a parallel path).\n",
    "qs-implement-setup-task": "a parallel path).\n",
    "qs-review-task": "on the fallback.\n",
    "qs-diagnose-task": "adopt it\nas the current diagnosis state (resume, don't restart).\n",
    "qs-verify-task": "on the fallback.\n",
    "qs-decompose-epic": "on the fallback.\n",
}


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
def test_setup_task_epic_product_stops_epic_factory_routes_to_decompose(
    harness_dir: Path,
) -> None:
    """QS-340 (was review-fix #04's "epic creates no worktree"): the epic ×
    product lane still stops before step 2 — no branch, worktree, or PR —
    while the epic × factory lane now runs step 2 like a task, with
    ``NEXT_PHASE = decompose-epic`` (a short-lived docs-only worktree).
    ``setup_task.py`` enforces the same split machine-side."""
    # Whitespace-normalised: the prose wraps mid-clause, so a naive
    # substring scan would pass vacuously (the trap
    # `test_workflow_no_desktop_fallback_by_necessity.py` documents).
    body = " ".join(_body(harness_dir, "qs-setup-task").split())
    assert "no branch, worktree, or PR" in body, (
        f"{harness_dir / 'qs-setup-task.md'}: the epic × product lane must "
        "not reach the branch/worktree step"
    )
    assert "#339" in body
    assert "child issues" in body
    assert "NEXT_PHASE = decompose-epic" in body
    assert "short-lived docs-only worktree" in body
    # The step-2 header no longer claims to be task-only.
    assert "### 2. Set up branch and worktree + emit launcher" in body
    assert "(tasks only)" not in body
    assert "`--no-worktree` is refused for an epic" in body


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_setup_task_declares_the_parent_epic_as_a_section(harness_dir: Path) -> None:
    """QS-340 gap 6: a stray ``Refs #N`` is read as the parent epic by the
    parser fallback, so setup-task declares the parent with the structured
    ``### Parent epic`` section instead."""
    body = _body(harness_dir, "qs-setup-task")
    assert "### Parent epic" in body
    assert "Refs #{{epic}}" not in body
    # S6: the parent-epic backfill must append via --body-file, never replace the
    # whole body with --body (which would wipe the issue text).
    assert "--body-file" in body
    assert "gh issue view {{N}} --json body --jq .body" in body


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
    sentinel = LANE_BLOCK_TERMINALS[agent_name]
    assert sentinel in tail, (
        f"{harness_dir / f'{agent_name}.md'}: lane block missing its "
        f"terminal sentence {sentinel!r}"
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


# --- (d) QS-340: qs-decompose-epic safety sentinels -------------------------

_DECOMPOSE_TEMPLATE = REPO_ROOT / "scripts" / "qs" / "agent_templates" / "qs-decompose-epic.md.j2"


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_decompose_epic_refuses_outside_the_epic_factory_lane(harness_dir: Path) -> None:
    body = " ".join(_body(harness_dir, "qs-decompose-epic").split())
    assert 'scale == "epic"' in body and 'target == "factory"' in body, (
        f"{harness_dir / 'qs-decompose-epic.md'}: the refusal guard must name "
        "both axes it checks"
    )
    assert "Refuse unless" in body


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_decompose_epic_hands_off_only_after_a_successful_land(harness_dir: Path) -> None:
    body = " ".join(_body(harness_dir, "qs-decompose-epic").split())
    assert "Hand off to `finish-task` only after `epic_doc.py land` succeeded" in body


def test_decompose_epic_body_never_mentions_story_files() -> None:
    """An epic writes no story file. Pinned on the template SOURCE's body
    block: the rendered file's shared Reference map lists ``docs/stories/``
    for every orchestrator, which is not this agent's own prose."""
    source = _DECOMPOSE_TEMPLATE.read_text(encoding="utf-8")
    body = source.split("[% block body %]", 1)[1].split("[% endblock %]", 1)[0]
    assert "docs/stories/" not in body


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
@pytest.mark.parametrize("agent_name", ["qs-create-plan", "qs-diagnose-task"])
def test_child_planners_read_the_parent_epic_doc(harness_dir: Path, agent_name: str) -> None:
    """QS-340 gap 8: a child's planning phase reads its parent epic's doc."""
    body = _body(harness_dir, agent_name)
    assert "`parent_epic_doc`" in body
    assert "git show origin/main:docs/epics/QS-<parent_epic>.md" in body


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_finish_task_case_a_has_the_epic_variant(harness_dir: Path) -> None:
    """QS-340: an epic never has a PR, so Case A is its path — probed by
    ``epic_doc.py status`` (the landed doc is not unpushed work), cleaned
    with ``--delete-branch``, and the epic issue is never closed."""
    body = " ".join(_body(harness_dir, "qs-finish-task").split())
    assert "python scripts/qs/epic_doc.py status --issue {{issue}}" in body
    assert "`safe_to_discard: true`" in body
    assert "--force --delete-branch" in body or "--force \\ --delete-branch" in body
    assert "epic session closed — doc on `main`, issue #{{issue}} stays open" in body
    assert "Never close the epic issue" in body
    assert "`scale`" in body
    # S5: a non-ok probe must STOP, not silently offer a force-delete, and the
    # kept-branch cleanup statuses must be handled honestly.
    assert "is not `ok`" in body
    assert "without a successful probe" in body
    assert '`status: "branch-checked-out-elsewhere"`' in body
    assert '`status: "removed-branch-kept"`' in body
    # S3 (fix plan #03): the force-delete prompt lists what would be lost using
    # the probe's `changed` and `unpushed_commits` fields.
    assert "`changed`" in body and "`unpushed_commits`" in body
    # S5 (fix plan #03): the error bullet names the actual set fields, not a
    # single hardcoded worktree_remove_error, and states a refusal touched
    # nothing.
    assert "STOP and surface `message`" in body
    assert "On a refusal" in body
