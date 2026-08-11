"""Lock the AC-3 contract: every phase orchestrator's handoff section emits
BOTH a launcher block (``claude --agent qs-<phase>``) AND a slash-command
fallback block (``/<phase>``).

This is the regression catch for QS-175 review-fix #07 — without it the
two-block pattern is enforced only by manual review.

Round-2 review-fix #03 / #04 / #05 / #06 cleanups:
- ``qs-create-plan`` is split out into a dedicated test because its
  ``NEXT_PHASE`` is dynamic (the orchestrator picks
  ``implement-task`` vs ``implement-setup-task`` at runtime) and a
  hardcoded slash form isn't possible there.
- The parametrise lists are split so unused ``expected_slash`` params
  don't trigger ruff ``ARG001``.
- The ``Fallback`` line scan is now a line-by-line walk rather than a
  brittle ``re.search`` with a greedy ``[^:]*`` pattern that could match
  past a future "Note:" parenthetical.
- The ``does-not-call-next-step-for-release`` regex strips inline +
  fenced backticks before scanning so a backtick-wrapped ``python ...
  --next-cmd release`` invocation can't slip past.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENTS_DIR = REPO_ROOT / ".claude" / "agents"

# Pattern for an ACTIVE ``next_step.py --next-cmd release`` invocation.
# Uses ``.`` with ``re.DOTALL`` so backslash-continued multi-line shell
# forms (``python ... \\\n    --next-cmd release``) are caught — the
# previous ``[^\n]`` form silently let them through (review-fix #04
# SF2). Compiled once at module load and shared between the
# qs-finish-task assertion and the regex-shape regression tests.
_FORBIDDEN_RELEASE_INVOCATION = re.compile(
    r"scripts/qs/next_step\.py.{0,200}?--next-cmd\s+[\"']?release",
    re.DOTALL,
)


# The 7 two-block orchestrators — every phase orchestrator that ships
# the strict ``Preferred`` / ``Fallback`` pattern. ``qs-finish-task``
# and ``qs-release`` deliberately do NOT ship it (release follow-up is
# text-only — see QS-175 OUT OF SCOPE); ``qs-finish-task`` gets its
# own dedicated tests below. QS-335 added the bug ×
# product lane's ``qs-diagnose-task`` (lifted from create-plan) and
# ``qs-verify-task`` (lifted from review-task).
_TWO_BLOCK_ORCHESTRATORS = [
    "qs-setup-task.md",
    "qs-create-plan.md",
    "qs-diagnose-task.md",
    "qs-implement-task.md",
    "qs-implement-setup-task.md",
    "qs-review-task.md",
    "qs-verify-task.md",
]

# Those orchestrators whose fallback block names a fixed ``/<phase>``
# token. ``qs-create-plan`` is dynamic (the NEXT_PHASE depends on the
# diff) and gets its own dedicated test asserting it uses a placeholder
# of the right SHAPE, but explicitly NOT ``{{same_context}}``.
# QS-335: ``qs-diagnose-task`` is likewise dynamic (fix / finish exits)
# but its fallback line carries the literal ``/{{NEXT_PHASE}}`` token, so
# the concrete-slash assertion still holds against that literal;
# ``qs-verify-task``'s clean-path fallback is the fixed ``/finish-task``.
_HARDCODED_FALLBACK = [
    ("qs-setup-task.md", "/create-plan"),
    ("qs-implement-task.md", "/review-task"),
    ("qs-implement-setup-task.md", "/review-task"),
    ("qs-review-task.md", "/finish-task"),
    ("qs-diagnose-task.md", "/{{NEXT_PHASE}}"),
    ("qs-verify-task.md", "/finish-task"),
]


@pytest.mark.parametrize("filename", _TWO_BLOCK_ORCHESTRATORS)
def test_orchestrator_has_two_block_handoff(filename: str) -> None:
    """Every orchestrator's handoff contains 'Preferred' AND 'Fallback' markers."""
    body = (AGENTS_DIR / filename).read_text()
    assert "Preferred" in body, (
        f"{filename}: missing 'Preferred' marker — every orchestrator must "
        f"label the interactive launcher path."
    )
    assert "Fallback" in body, (
        f"{filename}: missing 'Fallback' marker — every orchestrator must "
        f"label the degraded slash-command path."
    )


@pytest.mark.parametrize("filename", _TWO_BLOCK_ORCHESTRATORS)
def test_orchestrator_handoff_mentions_claude_agent_invocation(filename: str) -> None:
    """Handoff text references ``claude --agent qs-<phase>`` — the preferred path."""
    body = (AGENTS_DIR / filename).read_text()
    # The orchestrator may emit either a literal ``claude --agent qs-...``
    # string (qs-setup-task) or use the new_context variable that resolves
    # to one at runtime (qs-create-plan, qs-implement-task, etc.). Either
    # way the launcher concept must be named in the file body.
    assert "claude --agent" in body, (
        f"{filename}: the orchestrator handoff must reference "
        f"`claude --agent qs-<phase>` so the user knows the preferred path."
    )


@pytest.mark.parametrize(("filename", "expected_slash"), _HARDCODED_FALLBACK)
def test_orchestrator_fallback_uses_concrete_slash_form(
    filename: str, expected_slash: str,
) -> None:
    """Each fixed-next-phase fallback block names a real ``/<phase>``."""
    body = (AGENTS_DIR / filename).read_text()
    assert expected_slash in body, (
        f"{filename}: expected fallback to mention {expected_slash!r}; "
        f"this catches the qs-setup-task-style regression where "
        f"{{{{same_context}}}} or another verbatim template variable "
        f"would be emitted instead of a hardcoded slash form."
    )


# --------------------------------------------------------------------------- #
# qs-create-plan is the dynamic-next-phase exception. The fallback line
# can't be a single hardcoded ``/<phase>`` because the orchestrator picks
# ``implement-task`` vs ``implement-setup-task`` based on its task
# breakdown. The fallback uses a ``/{{NEXT_PHASE}}`` placeholder that
# the persona substitutes at runtime — what we forbid is the
# ``{{same_context}}`` shape that caused the qs-setup-task regression.
# --------------------------------------------------------------------------- #


def test_create_plan_fallback_uses_next_phase_placeholder_not_same_context() -> None:
    """qs-create-plan fallback uses ``/<NEXT_PHASE>`` placeholder, not ``{{same_context}}``."""
    body = (AGENTS_DIR / "qs-create-plan.md").read_text()
    fallback_line = _find_fallback_line(body)
    assert fallback_line is not None, "qs-create-plan: 'Fallback' block not found"
    assert "{{same_context}}" not in fallback_line, (
        f"qs-create-plan fallback uses {{{{same_context}}}} — that's the "
        f"verbatim template variable that caused the qs-setup-task "
        f"regression; use a slash-form placeholder like /{{{{NEXT_PHASE}}}} "
        f"instead. Got: {fallback_line!r}"
    )
    assert fallback_line.lstrip().startswith("/"), (
        f"qs-create-plan fallback should start with '/' (slash form). "
        f"Got: {fallback_line!r}"
    )


# --------------------------------------------------------------------------- #
# qs-setup-task: same dedicated assertion as before — hardcode /create-plan,
# don't fall back to a {{same_context}} template. This is the case the
# round-1 fix targeted.
# --------------------------------------------------------------------------- #


def test_setup_task_fallback_does_not_use_same_context_template() -> None:
    """qs-setup-task's fallback block must hardcode ``/create-plan``, not template."""
    body = (AGENTS_DIR / "qs-setup-task.md").read_text()
    fallback_line = _find_fallback_line(body)
    assert fallback_line is not None, "qs-setup-task: 'Fallback' block not found"
    assert "{{same_context}}" not in fallback_line, (
        "qs-setup-task fallback uses {{same_context}} template — must hardcode "
        "/create-plan to match the four peer orchestrators (review-fix #03)."
    )
    assert "/create-plan" in fallback_line, (
        f"qs-setup-task fallback should hardcode '/create-plan', got: "
        f"{fallback_line!r}"
    )


# --------------------------------------------------------------------------- #
# qs-finish-task is the deliberate exception — release runs on a different
# workspace (main checkout), so the agent body shows both forms as
# alternatives in plain prose rather than the strict two-block pattern.
# Per QS-175 OUT OF SCOPE: "DO NOT call the launcher with --next-cmd release".
# --------------------------------------------------------------------------- #


def test_finish_task_mentions_both_release_forms() -> None:
    """qs-finish-task's release suggestion mentions both forms as alternatives."""
    body = (AGENTS_DIR / "qs-finish-task.md").read_text()
    assert "/release" in body, "qs-finish-task: missing '/release' fallback mention"
    assert "claude --agent qs-release" in body, (
        "qs-finish-task: missing 'claude --agent qs-release' — the interactive "
        "form must also be named so users on CLI know the preferred path."
    )


def test_finish_task_does_not_call_next_step_for_release() -> None:
    """qs-finish-task must NOT call ``next_step.py --next-cmd release`` (OUT OF SCOPE).

    The agent body may *mention* the forbidden pattern in prose (e.g. "we
    don't build a launcher with `--next-cmd release`" — that's the OUT OF
    SCOPE explanation). What's forbidden is an ACTIVE invocation: a
    ``scripts/qs/next_step.py`` call followed by ``--next-cmd release``
    in real source, not inside backticks.

    We strip both inline and fenced backticks before scanning so a stray
    backtick-wrapped ``python scripts/qs/next_step.py --next-cmd release``
    on a single line can't bypass the check. The regex uses ``.`` with
    ``re.DOTALL`` (instead of ``[^\\n]``) so a backslash-continued
    multi-line invocation also gets caught (review-fix #04 SF2).
    """
    body = (AGENTS_DIR / "qs-finish-task.md").read_text()
    stripped = _strip_backticks(body)
    match = _FORBIDDEN_RELEASE_INVOCATION.search(stripped)
    assert not match, (
        f"qs-finish-task: active 'next_step.py --next-cmd release' invocation "
        f"found ({match.group()!r}). Release runs on the main checkout "
        f"(different workspace); per QS-175 OUT OF SCOPE the agent surfaces "
        f"the text suggestion but does not emit a launcher payload."
    )


# Regression patterns the forbidden-invocation regex MUST catch. Round-4
# SF2: the round-2 implementation used ``[^\n]`` which silently let
# backslash-continued multi-line invocations through.
_MULTILINE_INVOCATION = (
    "python scripts/qs/next_step.py \\\n"
    "    --next-cmd release\n"
)
_SINGLE_LINE_INVOCATION = (
    "python scripts/qs/next_step.py --next-cmd release\n"
)
_INVOCATION_QUOTED = (
    'python scripts/qs/next_step.py --next-cmd "release"\n'
)


@pytest.mark.parametrize("snippet", [
    _SINGLE_LINE_INVOCATION,
    _MULTILINE_INVOCATION,
    _INVOCATION_QUOTED,
])
def test_forbidden_release_regex_catches_invocation_forms(snippet: str) -> None:
    """The regex catches every active invocation shape — single, multi, quoted."""
    assert _FORBIDDEN_RELEASE_INVOCATION.search(snippet), (
        f"Forbidden-invocation regex missed {snippet!r}. Without DOTALL + "
        f"``.`` across newlines, a backslash-continued shell line slips "
        f"through (review-fix #04 SF2)."
    )


def test_forbidden_release_regex_ignores_prose_mention() -> None:
    """Bare prose mentioning ``--next-cmd release`` (without the script path) is fine."""
    # The agent body uses this exact wording in its OUT OF SCOPE note —
    # only an ACTIVE invocation paired with the script path is forbidden.
    prose = "We don't build a launcher with --next-cmd release."
    assert _FORBIDDEN_RELEASE_INVOCATION.search(prose) is None


# --------------------------------------------------------------------------- #
# QS-311 AC5 — the GUI launch-surface block.
#
# The Claude Code GUI has no ``--agent`` flag, so the launcher pins the
# next phase into ``<worktree>/.claude/settings.local.json`` and each
# handoff must print the GUI gesture (New session → select directory →
# name it). The set of orchestrators is exactly the two-block set — the
# 7 worktree handoffs — and the equality is asserted below.
#
# Review-fix #01 S3: this list used to be a bare ALIAS of
# ``_TWO_BLOCK_ORCHESTRATORS``, which made that equality assertion an
# object compared with itself — it could never fail, so the "pinned so it
# can't drift" docstring was false. It is now an independent literal, so
# the assertion is a real cross-check.
# --------------------------------------------------------------------------- #

_GUI_BLOCK_ORCHESTRATORS = [
    "qs-setup-task.md",
    "qs-create-plan.md",
    "qs-diagnose-task.md",
    "qs-implement-task.md",
    "qs-implement-setup-task.md",
    "qs-review-task.md",
    "qs-verify-task.md",
]

_GUI_BLOCK_MARKER = "[Claude Code GUI]"

# Tokens every GUI block must carry. Review-fix #01 N4: these used to be
# folded into one ``[\s\S]{0,400}?`` regex window, so a single added
# sentence dropped the match count and failed with a misleading "block
# missing" message — precisely the trap M3's rewording would have sprung.
# Asserted one by one instead, and the block is delimited by the enclosing
# fence rather than a character budget.
_GUI_BLOCK_REQUIRED_TOKENS = (
    "`.claude/settings.local.json`",
    "New session",
    "GUI launch surface (Claude Code Desktop)",
    # The consumer of the payload key. Decision 8 declined `gui_context`
    # precisely because no consumer read it; `phase_agent_pinned` earns its
    # place only if the handoff prose actually consults it, so that is
    # pinned here rather than left to review (review-fix #01 M3).
    "`phase_agent_pinned`",
)

# ``qs-review-task`` hands off twice — the zero-findings → finish-task
# path and the fix-plan loop back to the implement phase. A single block
# would leave the review-found-problems hop with no GUI instructions.
_GUI_BLOCK_COUNTS = {name: 1 for name in _GUI_BLOCK_ORCHESTRATORS}
_GUI_BLOCK_COUNTS["qs-review-task.md"] = 2
# QS-335: qs-verify-task lifts review-task's two handoffs (clean →
# finish-task, fixes → implement-task then re-run verify-task).
_GUI_BLOCK_COUNTS["qs-verify-task.md"] = 2

# The fallback line each orchestrator's handoff must still expose after
# the GUI block was inserted. ``qs-create-plan`` is the dynamic-next-phase
# exception, so its expectation is the placeholder rather than a concrete
# slash form — review-fix #01 S9: AC5 says "for each" of the five, and
# leaving create-plan out of the shadow check covered only 4 of 5.
_FALLBACK_LINE_EXPECTATION = dict(_HARDCODED_FALLBACK)
_FALLBACK_LINE_EXPECTATION["qs-create-plan.md"] = "/{{NEXT_PHASE}}"


def _gui_blocks(body: str) -> list[str]:
    """Return each ``[Claude Code GUI]`` block, delimited by its closing fence.

    Splitting on the marker guarantees a block can never absorb a later
    one; truncating at the next ``\\n``-anchored triple fence keeps it from
    absorbing the rest of the file. Both bounds are structural, unlike the
    fixed character window this replaces (review-fix #01 N4).
    """
    blocks: list[str] = []
    for chunk in body.split(_GUI_BLOCK_MARKER)[1:]:
        end = chunk.find("\n```")
        blocks.append(chunk if end == -1 else chunk[:end])
    return blocks


def test_gui_block_orchestrator_set_tracks_two_block_set() -> None:
    """The GUI-block list equals the two-block list — a real cross-check now.

    Review-fix #02 N-k: compared as **sets**, because the property is
    "the same orchestrators", not "in the same order" — a harmless
    reordering of the pre-existing list should not fail this.
    """
    assert set(_GUI_BLOCK_ORCHESTRATORS) == set(_TWO_BLOCK_ORCHESTRATORS), (
        "QS-311 AC5: the GUI launch-surface block belongs to exactly the "
        "orchestrators that emit a worktree handoff — the two-block set. "
        "If a new orchestrator joins one list it must join the other."
    )
    assert len(_GUI_BLOCK_ORCHESTRATORS) == len(set(_GUI_BLOCK_ORCHESTRATORS)), (
        "duplicate entry in _GUI_BLOCK_ORCHESTRATORS — the set comparison "
        "above would hide it, and the per-file block counts would double."
    )
    assert _GUI_BLOCK_ORCHESTRATORS is not _TWO_BLOCK_ORCHESTRATORS, (
        "review-fix #01 S3: the two lists must be independent literals. An "
        "alias makes the equality above compare an object with itself, so "
        "it can never fail and pins nothing."
    )


@pytest.mark.parametrize("filename", _GUI_BLOCK_ORCHESTRATORS)
def test_orchestrator_has_gui_launch_block(filename: str) -> None:
    """Each worktree handoff prints the ``[Claude Code GUI]`` block."""
    body = (AGENTS_DIR / filename).read_text()
    found = _gui_blocks(body)
    expected = _GUI_BLOCK_COUNTS[filename]
    assert len(found) == expected, (
        f"{filename}: expected {expected} '[Claude Code GUI]' block(s), "
        f"found {len(found)}. GUI users have no `--agent` flag — the "
        f"handoff must name the gesture (QS-311 AC5)."
    )


@pytest.mark.parametrize("token", _GUI_BLOCK_REQUIRED_TOKENS)
@pytest.mark.parametrize("filename", _GUI_BLOCK_ORCHESTRATORS)
def test_gui_block_names_required_tokens(filename: str, token: str) -> None:
    """Every GUI block names the mechanism, the gesture, and the doc section."""
    body = (AGENTS_DIR / filename).read_text()
    blocks = _gui_blocks(body)
    assert blocks, f"{filename}: no '[Claude Code GUI]' block at all"
    missing = [i for i, block in enumerate(blocks) if token not in block]
    assert not missing, (
        f"{filename}: GUI block(s) {missing} do not mention {token!r} "
        f"(QS-311 AC5). Each block must stand alone — a reader of one "
        f"handoff never sees the others."
    )


@pytest.mark.parametrize("filename", _GUI_BLOCK_ORCHESTRATORS)
def test_gui_block_states_the_pin_conditionally(filename: str) -> None:
    """No GUI block may assert the pin as accomplished fact.

    Review-fix #01 M3: the blocks read "the worktree **is now pinned**",
    but the writer's return value was discarded, the GUI displays the
    active agent *nowhere*, and the claim is deterministically FALSE for
    ``setup_task.py --no-worktree`` (which hands the main checkout to the
    launcher, where guard 2 always skips). The wording must be conditional
    and must name the ``--agent`` escape hatch.
    """
    body = (AGENTS_DIR / filename).read_text()
    for i, block in enumerate(_gui_blocks(body)):
        # Review-fix #02 N-d: require the hedge POSITIVELY. Banning one
        # phrasing of its negation is not the same property — "the worktree
        # **is pinned** to `qs-X`" re-asserts the pin as fact and passed
        # both of the assertions below.
        assert "should now be pinned" in block, (
            f"{filename}: GUI block {i} must hedge the pin with the literal "
            f"'should now be pinned'. The write can silently skip or fail, "
            f"so the prose may not assert it (review-fix #01 M3 / #02 N-d)."
        )
        assert "is now pinned" not in block, (
            f"{filename}: GUI block {i} asserts the pin as fact ('is now "
            f"pinned'). The write can silently skip or fail — say 'should "
            f"now be pinned' and name the fallback (review-fix #01 M3)."
        )
        assert "--agent" in block, (
            f"{filename}: GUI block {i} does not name the `--agent` escape "
            f"hatch. When the pin is missing the GUI gives no signal, so "
            f"the block must say how to recover (review-fix #01 M3)."
        )


# The sentence every orchestrator must carry in its `phase_agent_pinned:
# false` branch. Byte-identical across all six sites (qs-review-task hands
# off twice), so it is pinned here as one constant rather than described.
_STALE_PIN_SENTENCE = (
    "On `false` the worktree may still carry the **previous** phase's pin, "
    "which `false` cannot distinguish from no pin at all — so drop the GUI "
    "block entirely (pin sentence and bullets) and route the user to the "
    "Preferred `--agent` line, which is correct either way."
)


@pytest.mark.parametrize("filename", _GUI_BLOCK_ORCHESTRATORS)
def test_false_branch_carries_the_stale_pin_hazard(filename: str) -> None:
    """Review-fix #06 F5: `false` must not route the user into the GUI at all.

    The instruction used to be "drop the GUI block's pin sentence and point
    at the Preferred line instead", which leaves the *New session / Select
    directory / Name it* bullets standing. `phase_agent_pinned: false` cannot
    distinguish "no pin" from "**stale** pin", so following those bullets can
    open a GUI session still bound to the previous phase's orchestrator —
    with the agent name displayed nowhere, and orchestrators commit and push.

    Concrete trigger: an implement→review handoff where guard 1 fires because
    the branch renamed or deleted `.claude/agents/qs-review-task.md` — which
    is the kind of thing a branch editing agent files does. The GUI session
    then boots `qs-implement-task` under the review phase's name.

    Whitespace-normalised because the sentence line-wraps differently in the
    indented (`qs-create-plan`) and unindented sites; what must be identical
    is the wording, not the wrap.
    """
    body = " ".join((AGENTS_DIR / filename).read_text().split())
    expected = " ".join(_STALE_PIN_SENTENCE.split())
    assert expected in body, (
        f"{filename}: the `phase_agent_pinned: false` branch does not carry "
        f"the stale-pin hazard. Expected this sentence verbatim:\n{expected}"
    )


@pytest.mark.parametrize("filename", ["qs-review-task.md", "qs-verify-task.md"])
def test_review_task_carries_the_stale_pin_hazard_at_both_handoffs(
    filename: str,
) -> None:
    """``qs-review-task`` / ``qs-verify-task`` hand off twice, so each needs
    the sentence twice.

    The per-file test above is satisfied by one occurrence; this is the same
    two-handoff asymmetry ``_GUI_BLOCK_COUNTS`` exists for. Without it the
    fix-plan loop back to the implement phase keeps the old wording. QS-335:
    qs-verify-task is the bug × product lane's two-handoff review-variant.
    """
    body = " ".join((AGENTS_DIR / filename).read_text().split())
    expected = " ".join(_STALE_PIN_SENTENCE.split())
    assert body.count(expected) == 2, (
        f"{filename}: expected the stale-pin sentence at both handoff "
        f"sites, found {body.count(expected)}"
    )


@pytest.mark.parametrize("filename", _GUI_BLOCK_ORCHESTRATORS)
def test_gui_block_does_not_shadow_fallback_line(filename: str) -> None:
    """The GUI block must not become the line ``_find_fallback_line`` returns.

    That helper walks forward from the ``Fallback`` marker to the first
    line whose ``strip()`` starts with ``/`` or ``{{``. Every path inside
    the GUI block is therefore backticked — an unbackticked worktree path
    or ``{{worktree}}`` on its own line would hijack the scan and break
    the fallback assertions above.

    Review-fix #01 S9: parametrized over all five orchestrators, not just
    the four with a hardcoded fallback. ``qs-create-plan``'s expectation is
    its ``/{{NEXT_PHASE}}`` placeholder — the same shape
    ``test_create_plan_fallback_uses_next_phase_placeholder_not_same_context``
    checks from the other direction.
    """
    body = (AGENTS_DIR / filename).read_text()
    expected = _FALLBACK_LINE_EXPECTATION[filename]
    fallback_line = _find_fallback_line(body)
    assert fallback_line is not None, f"{filename}: 'Fallback' block not found"
    assert expected in fallback_line, (
        f"{filename}: the GUI block shadowed the fallback line — expected "
        f"{expected!r}, got {fallback_line!r}. Backtick every path "
        f"inside the GUI block (QS-311 AC5)."
    )


# --------------------------------------------------------------------------- #
# QS-311 AC6 — the Cursor / OpenCode counterparts carry a byte-identical
# pointer line to ``harness.md``. Harness sync is a path-level
# co-modification check (no content parity), so a cross-reference is the
# minimal honest edit: the GUI is a Claude-only launch surface, and those
# trees have no Desktop prose to extend.
# --------------------------------------------------------------------------- #

_COUNTERPART_DIRS = (".cursor", ".opencode")

# The block all 10 counterparts carry verbatim. Review-fix #01 M3 added the
# second sentence (the pin is conditional, and nothing in *their* harness
# reads the flag); review-fix #03 C7 wrapped the whole thing to the ~72
# columns the surrounding docs use — it was ~150 chars/line, and since the
# tests pin it verbatim, every future wrap would have cost 10 files plus this
# constant. Wrapped once, here, while those 10 files were being touched
# anyway. Review-fix #03 B1 dropped `settings_rebuilt` from it again, since
# Option B removed that key.
_POINTER_BLOCK = "\n".join([
    "> Launch surfaces for the Claude harness (including the GUI) are",
    "> documented in",
    "> [docs/workflow/harness.md](../../docs/workflow/harness.md).",
    "> That doc's GUI phase pin is best-effort: the Claude payload",
    "> reports the outcome as `phase_agent_pinned`, and no other harness",
    "> reads it.",
])


@pytest.mark.parametrize("harness_dir", _COUNTERPART_DIRS)
@pytest.mark.parametrize("filename", _GUI_BLOCK_ORCHESTRATORS)
def test_counterpart_agents_point_at_harness_doc(
    harness_dir: str, filename: str,
) -> None:
    """All 10 counterparts carry the identical pointer block."""
    path = REPO_ROOT / harness_dir / "agents" / filename
    assert path.is_file(), f"missing counterpart agent file: {path}"
    body = path.read_text()
    assert _POINTER_BLOCK in body, (
        f"{harness_dir}/agents/{filename}: missing the verbatim harness.md "
        f"pointer block (QS-311 AC6). Expected:\n{_POINTER_BLOCK}"
    )


def test_pointer_block_stays_within_the_doc_line_width() -> None:
    """Review-fix #03 C7: the pinned block must not drift back to ~150 chars.

    It is duplicated across 10 files and pinned verbatim by the test above,
    so a re-widening is 11 files to undo. Guarding the width here makes that
    a test failure instead of a future finding.
    """
    for line in _POINTER_BLOCK.split("\n"):
        assert len(line) <= 78, (
            f"pointer-block line is {len(line)} chars, over the ~72-column "
            f"convention of the surrounding docs: {line!r}"
        )


# --------------------------------------------------------------------------- #
# QS-335 D3 — the two shared orchestrators carry the lane-resolved dynamic
# handoff. ``qs-setup-task`` routes ``create-plan`` by default and
# ``diagnose-task`` when the lane is ``bug-product``; ``qs-implement-task``
# routes ``review-task`` by default and ``verify-task`` for ``bug-product``.
# Both branches must appear in every one of the 3 harness copies. Pattern
# of ``test_lane_steps_parity.py`` (HARNESS_DIRS-parametrized). The
# fallback line keeps its literal default as the first slash token with
# the bug-product branch appended mid-sentence (D3 pin compatibility).
# ``qs-create-plan``, ``qs-review-task``, ``qs-implement-setup-task`` stay
# byte-unchanged — verified at review time by ``git diff`` against main
# (AC-4), no parity test built.
# --------------------------------------------------------------------------- #

_ROUTING_HARNESS_DIRS = (
    REPO_ROOT / ".claude" / "agents",
    REPO_ROOT / ".cursor" / "agents",
    REPO_ROOT / ".opencode" / "agents",
)

# agent file -> (default-lane phase token, bug-product-lane phase token)
_LANE_ROUTING = {
    "qs-setup-task.md": ("/create-plan", "/diagnose-task"),
    "qs-implement-task.md": ("/review-task", "/verify-task"),
}


@pytest.mark.parametrize(
    "harness_dir", _ROUTING_HARNESS_DIRS, ids=lambda p: p.parent.name.lstrip(".")
)
@pytest.mark.parametrize("filename", sorted(_LANE_ROUTING))
def test_shared_orchestrator_carries_both_lane_branches(
    harness_dir: Path, filename: str,
) -> None:
    """Both the default and the bug-product routing branch appear in each copy."""
    default_tok, bug_tok = _LANE_ROUTING[filename]
    path = harness_dir / filename
    assert path.is_file(), f"missing agent file: {path}"
    body = path.read_text(encoding="utf-8")
    assert default_tok in body, (
        f"{path}: missing the default-lane routing token {default_tok!r} "
        f"(QS-335 D3)."
    )
    assert bug_tok in body, (
        f"{path}: missing the bug-product routing branch {bug_tok!r} — the "
        f"handoff must resolve the next phase from the lane (QS-335 D3)."
    )
    assert "bug-product" in body, (
        f"{path}: the lane-resolved handoff must name the `bug-product` lane "
        f"that selects the alternate next phase (QS-335 D3)."
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _find_fallback_line(body: str) -> str | None:
    """Return the first non-empty line following the literal ``Fallback`` marker.

    Replaces the brittle ``re.search(r"Fallback[^:]*:...")`` pattern (review-
    fix #02 NTH2). Iterates lines: once a line starting with ``Fallback``
    (capital F — the marker, not a prose mention) is found, walk forward
    until a content line that starts with ``/`` or ``{{`` is reached.
    That's the slash-fallback (or placeholder) line.
    """
    lines = body.splitlines()
    in_fallback_block = False
    for line in lines:
        stripped = line.strip()
        if not in_fallback_block:
            # Match a ``Fallback`` marker — must start with that literal
            # word (after any leading whitespace). Excludes prose
            # mentions like "the fallback path" that appear earlier in
            # the file body.
            if stripped.startswith("Fallback"):
                in_fallback_block = True
            continue
        # In the fallback block — find the first line whose stripped
        # content begins with ``/`` (the slash-form fallback) or with
        # ``{{`` (a placeholder, e.g. ``{{same_context}}``). Either is a
        # candidate "fallback line" for downstream tests to inspect.
        # Blank lines, the preamble's closing ``):`` line, and any
        # other intermediate content are naturally skipped by the
        # implicit loop continuation (review-fix #04 NTH2: the trailing
        # explicit ``if/continue`` was dead code).
        if stripped.startswith(("/", "{{")):
            return line
    return None


def _strip_backticks(text: str) -> str:
    """Remove single and triple backtick fences, keeping the inner content.

    Review-fix #02 NTH3: without this, a fenced one-line invocation
    like a backtick-wrapped ``python scripts/qs/next_step.py
    --next-cmd release`` would slip past the
    ``does-not-call-next-step-for-release`` regex.
    """
    # Strip triple-fence blocks first (they may span multiple lines), then
    # single-tick inline code. Both forms are replaced with their inner
    # text so the search can find an active invocation regardless of
    # markdown formatting.
    text = re.sub(r"```[a-zA-Z]*\n?([\s\S]*?)```", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text
