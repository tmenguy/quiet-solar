"""Pin the honesty of `qs-implement-setup-task`'s `--impacted` prose.

QS-290 (S-4) gave `--impacted` an early exit: a change set with no `.py`
file at all spawns no pytest, no diff-cover and no `git fetch`, just a
handful of local git calls, and returns 0. testmon fingerprints AST
blocks in `.py` files only, so such a run could never have selected a
test anyway — the exit changed the cost, not the verdict.

That makes the agent's `--impacted` prose actively misleading, and it
matters *most* in exactly this agent: it is the dev-environment
implement variant, so its change sets are the ones most likely to be
pure non-`.py` (a docs-only edit, an agent-file-only edit). The agent
must not read "`--impacted` guards the tooling" and conclude a green
means something when nothing ran.

The contract pinned here:

1. no claim that testmon-selected tests guard anything may stand
   UNSCOPED — each must carry the "change set contains a `.py` file"
   condition. This is asserted **semantically**, not by rejecting one
   exact phrasing: an earlier version matched a single literal string
   (line break included), which a copy-edit could sidestep while
   restoring an equally false claim in different words;
2. the body must state that a non-`.py` change set checks nothing;
3. it must name `--quick tests/qs` as the verification in that case,
   not merely as a supplement;
4. it must say the early exit **fails closed** on a git-probe failure,
   so "no exit message" never reads as "the exit fired silently";
5. it must name `--quick tests/test_quality_gate.py` for gate changes,
   since the impacted pass `--ignore`s that file BY PATH — a `.py`
   change is not automatically a verified one;
6. its automatic `git add` must stage the dev-tooling tests, or a guard
   like this one can be written and then left uncommitted.

Mirrored across all three harnesses (Claude / Cursor / OpenCode) so an
edit to one harness alone fails, matching the harness-sync rule in
`docs/workflow/project-rules.md`.

Pattern follows `test_doc_maintenance_parity.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

HARNESS_DIRS: tuple[Path, ...] = (
    REPO_ROOT / ".claude" / "agents",
    REPO_ROOT / ".cursor" / "agents",
    REPO_ROOT / ".opencode" / "agents",
)

AGENT_NAME = "qs-implement-setup-task"

# Assert the SEMANTIC, not one phrasing.
# The original guard rejected a single exact string (including its line break),
# so a copy-edit could restore an unconditional claim in different words while
# every test still passed.
#
# The semantic: wherever the body claims testmon-selected tests guard the
# tooling, that claim must be scoped to a change set that CONTAINS a `.py`
# file. So every mention of testmon selection must sit inside a paragraph that
# also states the `.py` condition.
_TESTMON_CLAIM_RE = re.compile(r"testmon[- ]selected tests", re.IGNORECASE)
_PY_CONDITION_RE = re.compile(
    r"(contains? a `\.py` file|does\*{0,2} contain a `\.py` file|`\.py` presence)",
    re.IGNORECASE,
)


def _harness_id(p: Path) -> str:
    return p.parent.name.lstrip(".")


def _body(harness_dir: Path) -> str:
    path = harness_dir / f"{AGENT_NAME}.md"
    assert path.is_file(), f"Missing agent file: {path}"
    return path.read_text(encoding="utf-8")


def _flat(harness_dir: Path) -> str:
    """The body with all runs of whitespace collapsed to single spaces."""
    return " ".join(_body(harness_dir).split())


def _paragraphs(body: str) -> list[str]:
    """Split on blank lines, then flatten each block's internal wrapping.

    These files are hard-wrapped at ~70 columns, so a sentence and its
    qualifying clause routinely sit on different lines; matching per-line would
    make the guard depend on where the wrap happens to fall.
    """
    return [" ".join(block.split()) for block in body.split("\n\n")]


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_testmon_guarantee_is_scoped_to_py_change_sets(harness_dir: Path) -> None:
    """Any "testmon-selected tests guard this" claim must carry the `.py` condition.

    Unconditionally, that claim is FALSE: QS-290's early exit runs nothing at
    all for a change set with no `.py` file.
    """
    body = _body(harness_dir)
    claiming = [p for p in _paragraphs(body) if _TESTMON_CLAIM_RE.search(p)]
    assert claiming, (
        f"{harness_dir / f'{AGENT_NAME}.md'}: expected the body to discuss what "
        "testmon-selected tests do and do not guarantee; found no mention at all."
    )
    for para in claiming:
        assert _PY_CONDITION_RE.search(para), (
            f"{harness_dir / f'{AGENT_NAME}.md'}: this claim is unscoped, so it is "
            "false for a change set with no `.py` file (the early exit runs "
            f"nothing there). Qualify it with the `.py` condition:\n\n{para}"
        )


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_states_non_py_change_sets_check_nothing(harness_dir: Path) -> None:
    """The agent must know that a non-`.py` green means nothing ran.

    Assertions run against whitespace-normalized text: these files are
    hard-wrapped, so a phrase's line break is an accident of formatting and must
    not be part of the contract.
    """
    body = _flat(harness_dir)
    assert "no `.py` file at all" in body, (
        f"{harness_dir / f'{AGENT_NAME}.md'}: must state that a change set "
        "with no `.py` file makes `--impacted` exit early."
    )
    assert "checks **nothing**" in body, (
        f"{harness_dir / f'{AGENT_NAME}.md'}: must say the non-`.py` run "
        "checks NOTHING — 'fast no-op' alone reads as 'cheap but still valid'."
    )


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_names_quick_tests_qs_as_the_verification(harness_dir: Path) -> None:
    """`--quick tests/qs` is THE check for a non-`.py` change set."""
    body = _flat(harness_dir)
    assert "not a supplement" in body, (
        f"{harness_dir / f'{AGENT_NAME}.md'}: must promote `--quick tests/qs` "
        "from supplement to sole verification for non-`.py` change sets."
    )
    assert "quality_gate.py --quick tests/qs" in body, (
        f"{harness_dir / f'{AGENT_NAME}.md'}: the runnable command must stay present."
    )


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_states_the_early_exit_fails_closed(harness_dir: Path) -> None:
    """`.py` presence is not the WHOLE decision.

    A failed git probe does not take the exit — silence about the exit never
    means the exit fired.
    """
    body = _flat(harness_dir)
    assert "fails closed" in body, (
        f"{harness_dir / f'{AGENT_NAME}.md'}: must say the early exit fails "
        "CLOSED on a git-probe failure."
    )


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_names_the_quality_gate_self_test_target(harness_dir: Path) -> None:
    """A `.py` change is not automatically covered.

    The impacted pass excludes `tests/test_quality_gate.py` BY PATH, so a change
    to the gate itself is never verified by `--impacted` — the agent must be told
    the explicit target.
    """
    body = _flat(harness_dir)
    assert "quality_gate.py --quick tests/test_quality_gate.py" in body, (
        f"{harness_dir / f'{AGENT_NAME}.md'}: must name the explicit quick "
        "target for quality-gate changes (the impacted pass --ignores that file)."
    )
    assert "select" in body and "zero" in body, (
        f"{harness_dir / f'{AGENT_NAME}.md'}: must warn that a `.py` edit can "
        "still select zero tests."
    )


@pytest.mark.parametrize("harness_dir", HARNESS_DIRS, ids=_harness_id)
def test_staging_command_includes_dev_tooling_tests(harness_dir: Path) -> None:
    """The automatic `git add` must stage the tests.

    This agent is permitted to write dev-tooling tests, but its staging command
    omitted `tests/` entirely — so an agent could commit prose changes while
    silently leaving the regression guard uncommitted. (This very PR only landed
    its guard because the story carried a hand-staging warning.)
    """
    body = _body(harness_dir)
    add_lines = [ln for ln in body.splitlines() if ln.strip().startswith("git add ")]
    assert add_lines, f"{harness_dir / f'{AGENT_NAME}.md'}: no `git add` command found"
    for line in add_lines:
        assert "tests/qs/" in line, (
            f"{harness_dir / f'{AGENT_NAME}.md'}: staging command omits dev-tooling "
            f"tests, so a new guard can be written and then left uncommitted:\n{line}"
        )
        assert "tests/test_quality_gate.py" in line, (
            f"{harness_dir / f'{AGENT_NAME}.md'}: staging command omits the "
            f"quality-gate self-tests:\n{line}"
        )
