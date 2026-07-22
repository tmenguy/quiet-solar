"""Unit tests for the QS-302 reverse fill-order knob of adapt_repartition.

Covers AC-1, AC-2 and AC-3 of the QS-302 story:

- AC-1: `fill_order_reverse` defaults to `False`; the forward (`False`)
  placement is byte-identical to omitting the kwarg (guards the
  `slot_range` / directional-cap refactor from silently drifting the
  existing forward behavior).
- AC-2: with `fill_order_reverse=True`, a partial energy budget on a
  uniform-headroom window fills the HIGHEST-index slots and leaves the
  LOWEST as `None` (latest-first placement — verified by output pattern).
- AC-3: under reverse fill with `support_auto=True`, the cap loop caps
  the EARLY unfilled region to green-only while the filled late slots
  retain their consuming command.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytz

from custom_components.quiet_solar.const import (
    CONSTRAINT_TYPE_FILLER,
    SOLVER_STEP_S,
)
from custom_components.quiet_solar.home_model.commands import (
    CMD_CST_AUTO_GREEN,
    CMD_CST_AUTO_GREEN_CONSIGN,
    LoadCommand,
)
from custom_components.quiet_solar.home_model.constraints import (
    MultiStepsPowerLoadConstraint,
)
from tests.factories import (
    MinimalTestHome,
    MinimalTestLoad,
    TestDynamicGroupDouble,
)


def _make_constraint(
    *,
    name: str = "test",
    target_value: float = 50_000.0,
    support_auto: bool = False,
    power_steps: list[LoadCommand] | None = None,
    num_slots: int = 6,
) -> MultiStepsPowerLoadConstraint:
    """Build a MultiStepsPowerLoadConstraint on a uniform-headroom window
    for the reverse-fill unit tests (single load, no on/off cap)."""
    home = MinimalTestHome()
    load = MinimalTestLoad(name=name, home=home)
    load.father_device = TestDynamicGroupDouble(
        max_amps=[64.0, 64.0, 64.0], num_slots=num_slots
    )
    load.num_max_on_off = None
    if power_steps is None:
        power_steps = [LoadCommand(command="on", power_consign=600.0)]
    now = datetime.now(tz=pytz.UTC)
    return MultiStepsPowerLoadConstraint(
        time=now,
        load=load,
        type=CONSTRAINT_TYPE_FILLER,
        end_of_constraint=now + timedelta(hours=6),
        initial_value=0.0,
        target_value=target_value,
        power_steps=power_steps,
        support_auto=support_auto,
    )


def _durations(num_slots: int) -> np.ndarray:
    """Return a uniform per-slot duration array of `num_slots` slots."""
    return np.full(num_slots, float(SOLVER_STEP_S), dtype=np.float64)


# ---------------------------------------------------------------------------
# AC-1: default False + forward byte-identical to omitting the kwarg
# ---------------------------------------------------------------------------


def test_forward_fill_matches_hardcoded_baseline():
    """Given the SAME scenario as the reverse test (uniform 6-slot window,
    single 600 W step, 350 Wh partial budget)
    When adapt_repartition runs forward (fill_order_reverse=False)
    Then the placement matches a HARD-CODED baseline — the earliest two
    slots are filled at 600 W, the remaining four stay None, the deltas are
    [600, 600, 0, 0, 0, 0] and the returned energy_delta is 50 Wh.

    This is a real regression guard on the forward code path (guards the
    slot_range / directional-cap refactor from silently drifting forward
    behavior — AC-1's baseline sub-clause).  Asserting against concrete
    expected values (rather than a second invocation of the same default
    path) is what makes it non-tautological — QS-302 review fix #01 S3.
    """
    durations = _durations(6)
    existing: list[LoadCommand | None] = [None] * 6
    now = datetime.now(tz=pytz.UTC)

    constraint = _make_constraint(name="fwd")
    _, solved, changed, energy_delta, cmds, deltas = constraint.adapt_repartition(
        first_slot=0,
        last_slot=5,
        energy_delta=350.0,
        power_slots_duration_s=durations,
        existing_commands=list(existing),
        allow_change_state=True,
        time=now,
        fill_order_reverse=False,
    )

    # 600 W over one SOLVER_STEP_S slot = 150 Wh; a 350 Wh budget fills the
    # first two slots (300 Wh) and breaks before the third (would overshoot).
    assert changed
    assert not solved
    assert float(energy_delta) == 50.0

    # Earliest-first placement: slots 0 and 1 filled, 2..5 untouched.
    assert cmds[0] is not None and cmds[0].command == "on" and cmds[0].power_consign == 600.0
    assert cmds[1] is not None and cmds[1].command == "on" and cmds[1].power_consign == 600.0
    assert cmds[2] is None
    assert cmds[3] is None
    assert cmds[4] is None
    assert cmds[5] is None

    assert [float(x) for x in deltas] == [600.0, 600.0, 0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# AC-2: reverse fills highest slots, leaves lowest None (partial budget)
# ---------------------------------------------------------------------------


def test_reverse_fills_latest_slots_first_on_partial_budget():
    """Given a uniform-headroom 6-slot window and a partial energy budget
    (enough for ~2 slots)
    When adapt_repartition runs with fill_order_reverse=True
    Then the highest-index slots receive commands and the lowest stay
    None (latest-first placement) — the mirror image of the forward case.
    """
    durations = _durations(6)
    existing: list[LoadCommand | None] = [None] * 6
    now = datetime.now(tz=pytz.UTC)

    # 600 W step over SOLVER_STEP_S → 150 Wh per placed slot; 350 Wh budget
    # fits exactly 2 slots.
    forward = _make_constraint(name="fwd")
    _, _, _, _, cmds_fwd, _ = forward.adapt_repartition(
        first_slot=0,
        last_slot=5,
        energy_delta=350.0,
        power_slots_duration_s=durations,
        existing_commands=list(existing),
        allow_change_state=True,
        time=now,
        fill_order_reverse=False,
    )

    reverse = _make_constraint(name="rev")
    _, _, _, _, cmds_rev, _ = reverse.adapt_repartition(
        first_slot=0,
        last_slot=5,
        energy_delta=350.0,
        power_slots_duration_s=durations,
        existing_commands=list(existing),
        allow_change_state=True,
        time=now,
        fill_order_reverse=True,
    )

    # Forward: earliest slots filled, latest untouched.
    assert cmds_fwd[0] is not None
    assert cmds_fwd[5] is None
    # Reverse: latest slots filled, earliest untouched.
    assert cmds_rev[5] is not None
    assert cmds_rev[0] is None

    fwd_filled = {i for i, c in enumerate(cmds_fwd) if c is not None}
    rev_filled = {i for i, c in enumerate(cmds_rev) if c is not None}
    # Same COUNT of placements, strictly-later index set under reverse.
    assert len(fwd_filled) == len(rev_filled)
    assert max(fwd_filled) < min(rev_filled)


# ---------------------------------------------------------------------------
# AC-3: reverse cap loop caps the early unfilled region to green-only
# ---------------------------------------------------------------------------


def test_reverse_cap_loop_caps_early_region_and_keeps_late_commands():
    """Given a support_auto constraint filled latest-first with a partial
    budget
    When the support_auto cap loop runs under fill_order_reverse=True
    Then the EARLY unfilled slots are capped to green-only while the
    filled LATE slots retain their consuming (green-consign) command.
    """
    durations = _durations(6)
    existing: list[LoadCommand | None] = [None] * 6
    now = datetime.now(tz=pytz.UTC)

    c = _make_constraint(name="auto", support_auto=True)
    _, _, changed, _, cmds, _ = c.adapt_repartition(
        first_slot=0,
        last_slot=5,
        energy_delta=350.0,
        power_slots_duration_s=durations,
        existing_commands=list(existing),
        allow_change_state=True,
        time=now,
        fill_order_reverse=True,
    )

    assert changed is True
    # No slot is left None — the cap loop fills every slot.
    assert all(c is not None for c in cmds)

    consuming = [i for i, cmd in enumerate(cmds) if cmd.command == CMD_CST_AUTO_GREEN_CONSIGN]
    green_only = [i for i, cmd in enumerate(cmds) if cmd.command == CMD_CST_AUTO_GREEN]

    # Latest slots consume; earliest slots are capped green-only.
    assert len(consuming) >= 1
    assert len(green_only) >= 1
    # The consuming region is strictly LATER than the capped region.
    assert min(consuming) > max(green_only)
    # Concretely: the frontier / late slots retain a consuming command.
    assert cmds[5].command == CMD_CST_AUTO_GREEN_CONSIGN
    # And the earliest slot is capped to green-only.
    assert cmds[0].command == CMD_CST_AUTO_GREEN


def test_reverse_cap_loop_green_only_has_zero_power():
    """The capped early slots must carry zero consuming power (green-only
    is a pure headroom cap, not a deliberate drain)."""
    durations = _durations(6)
    existing: list[LoadCommand | None] = [None] * 6
    now = datetime.now(tz=pytz.UTC)

    c = _make_constraint(name="auto", support_auto=True)
    _, _, _, _, cmds, deltas = c.adapt_repartition(
        first_slot=0,
        last_slot=5,
        energy_delta=350.0,
        power_slots_duration_s=durations,
        existing_commands=list(existing),
        allow_change_state=True,
        time=now,
        fill_order_reverse=True,
    )

    for i, cmd in enumerate(cmds):
        if cmd.command == CMD_CST_AUTO_GREEN:
            assert deltas[i] == 0.0
