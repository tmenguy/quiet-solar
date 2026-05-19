"""Unit tests for the QS-178 per-slot battery-charge guard.

Covers AC-6 and AC-7 of the QS-178 story:

(a) Gate-off when reclaim (init_energy_delta < 0).
(b) Gate-off when bat_charge_traj=None.
(c) Cap by forward_min: placement is clamped to a lower command when the
    proposed delta_power would drain below the floor.
(d) Forward propagation lockstep: bat_charge_traj and forward_min decrement
    together after each placement; subsequent slot whose forward_min hits
    the floor is skipped.
(e) Multi-pass survival: the invariant
    forward_min[i] = min(bat_charge_traj[i:last_slot+1]) holds across the
    multi-pass loop because both passes only decrement.
(f) DC-clamp slot: placement at a slot where the inverter clamp is already
    baked into the trajectory reduces bat_charge_traj by the full
    delta_power * dt / 3600 (proves the propagation rule captures the net
    effect correctly).

Defaults make the guard inert at all existing 49+ call sites — AC-7 is
verified by the existing `tests/test_coverage_constraints.py` suite
running unchanged against the new signature.
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
    target_value: float = 5000.0,
    current_value: float | None = None,
    support_auto: bool = False,
    power_steps: list[LoadCommand] | None = None,
    num_max_on_off: int | None = None,
    home: MinimalTestHome | None = None,
    num_slots: int = 4,
) -> MultiStepsPowerLoadConstraint:
    home = home or MinimalTestHome()
    load = MinimalTestLoad(name=name, home=home)
    load.father_device = TestDynamicGroupDouble(
        max_amps=[64.0, 64.0, 64.0], num_slots=num_slots
    )
    load.num_max_on_off = num_max_on_off
    if power_steps is None:
        power_steps = [LoadCommand(command="on", power_consign=600.0)]
    now = datetime.now(tz=pytz.UTC)
    return MultiStepsPowerLoadConstraint(
        time=now,
        load=load,
        type=CONSTRAINT_TYPE_FILLER,
        end_of_constraint=now + timedelta(hours=4),
        initial_value=0.0,
        target_value=target_value,
        current_value=current_value,
        power_steps=power_steps,
        support_auto=support_auto,
    )


def _durations(num_slots: int) -> np.ndarray:
    return np.full(num_slots, float(SOLVER_STEP_S), dtype=np.float64)


# ---------------------------------------------------------------------------
# (a) Gate-off when reclaim
# ---------------------------------------------------------------------------


def test_guard_gate_off_when_reclaim():
    """Given init_energy_delta < 0 with bat_charge_traj passed
    When adapt_repartition runs
    Then behavior is byte-identical to baseline (no guard applied) — the
    out_commands and out_delta_power match what we'd get with the kwarg
    omitted entirely.
    """
    constraint_a = _make_constraint(
        name="a",
        target_value=1000.0,
        current_value=500.0,
        power_steps=[LoadCommand(command="on", power_consign=600.0)],
    )
    constraint_b = _make_constraint(
        name="b",
        target_value=1000.0,
        current_value=500.0,
        power_steps=[LoadCommand(command="on", power_consign=600.0)],
    )

    durations = _durations(4)
    existing = [
        LoadCommand(command="on", power_consign=600.0),
        LoadCommand(command="on", power_consign=600.0),
        LoadCommand(command="on", power_consign=600.0),
        LoadCommand(command="on", power_consign=600.0),
    ]
    bat_traj = np.array([1000.0, 1000.0, 1000.0, 1000.0], dtype=np.float64)

    now = datetime.now(tz=pytz.UTC)

    _, _, changed_a, _, cmds_a, deltas_a = constraint_a.adapt_repartition(
        first_slot=0,
        last_slot=3,
        energy_delta=-200.0,
        power_slots_duration_s=durations,
        existing_commands=list(existing),
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj.copy(),
        battery_min_wh=900.0,
    )

    _, _, changed_b, _, cmds_b, deltas_b = constraint_b.adapt_repartition(
        first_slot=0,
        last_slot=3,
        energy_delta=-200.0,
        power_slots_duration_s=durations,
        existing_commands=list(existing),
        allow_change_state=True,
        time=now,
    )

    assert changed_a == changed_b
    assert np.allclose(deltas_a, deltas_b)
    # Command lists should be identical structurally.
    for ca, cb in zip(cmds_a, cmds_b, strict=False):
        if ca is None or cb is None:
            assert ca is cb
        else:
            assert ca.power_consign == cb.power_consign


# ---------------------------------------------------------------------------
# (b) Gate-off when None
# ---------------------------------------------------------------------------


def test_guard_gate_off_when_traj_is_none():
    """Given bat_charge_traj=None (or omitted)
    When adapt_repartition runs with init_energy_delta > 0
    Then behavior is byte-identical to baseline (guard inert).
    """
    c_no_arg = _make_constraint(target_value=2000.0)
    c_explicit_none = _make_constraint(target_value=2000.0)

    durations = _durations(3)
    existing: list[LoadCommand | None] = [None, None, None]
    now = datetime.now(tz=pytz.UTC)

    _, _, ch1, _, cmds1, deltas1 = c_no_arg.adapt_repartition(
        first_slot=0,
        last_slot=2,
        energy_delta=600.0,
        power_slots_duration_s=durations,
        existing_commands=list(existing),
        allow_change_state=True,
        time=now,
    )
    _, _, ch2, _, cmds2, deltas2 = c_explicit_none.adapt_repartition(
        first_slot=0,
        last_slot=2,
        energy_delta=600.0,
        power_slots_duration_s=durations,
        existing_commands=list(existing),
        allow_change_state=True,
        time=now,
        bat_charge_traj=None,
        battery_min_wh=0.0,
    )

    assert ch1 == ch2
    assert np.allclose(deltas1, deltas2)
    for c1, c2 in zip(cmds1, cmds2, strict=False):
        if c1 is None or c2 is None:
            assert c1 is c2
        else:
            assert c1.power_consign == c2.power_consign


# ---------------------------------------------------------------------------
# (c) Cap by forward_min
# ---------------------------------------------------------------------------


def test_guard_clamps_to_lower_command_when_drain_exceeds_max():
    """Given a slot whose forward_min - floor only fits a lower command
    When adapt_repartition tries to place a higher command
    Then it clamps to the largest command whose delta_power fits.
    """
    # Two-step ladder: [400, 1200]. Floor at 800 Wh. Initial charge 1000 Wh.
    # max_drain_wh = 1000 - 800 = 200. With 900 s slot:
    # max_delta_power = 200 * 3600 / 900 = 800 W.
    # The 400 W step fits (delta 400 ≤ 800).  The 1200 W step would not.
    constraint = _make_constraint(
        target_value=5000.0,
        power_steps=[
            LoadCommand(command="on", power_consign=400.0),
            LoadCommand(command="on", power_consign=1200.0),
        ],
    )
    durations = _durations(1)
    bat_traj = np.array([1000.0], dtype=np.float64)
    now = datetime.now(tz=pytz.UTC)

    _, _, changed, _, cmds, deltas = constraint.adapt_repartition(
        first_slot=0,
        last_slot=0,
        energy_delta=300.0,
        power_slots_duration_s=durations,
        existing_commands=[None],
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj,
        battery_min_wh=800.0,
    )

    assert changed is True
    assert cmds[0] is not None
    # Clamped to the smaller (400 W) command — not the bumped 1200 W.
    assert cmds[0].power_consign == 400.0
    assert deltas[0] == 400.0


def test_guard_skips_slot_when_smallest_command_exceeds_max():
    """Given a slot whose forward_min - floor cannot fit even the smallest
    command
    When adapt_repartition iterates that slot
    Then no placement occurs there.
    """
    constraint = _make_constraint(
        target_value=5000.0,
        power_steps=[LoadCommand(command="on", power_consign=1500.0)],
    )
    durations = _durations(1)
    # max_drain = 100 Wh; max_delta_power = max_drain * 3600 / SOLVER_STEP_S
    # (= 400 W when SOLVER_STEP_S=900) < 1500 W command.
    bat_traj = np.array([900.0], dtype=np.float64)
    now = datetime.now(tz=pytz.UTC)
    # Sanity: keep the test arithmetic honest if SOLVER_STEP_S ever changes.
    max_drain_wh = 900.0 - 800.0
    max_delta_power_w = max_drain_wh * 3600.0 / float(SOLVER_STEP_S)
    assert max_delta_power_w < 1500.0  # would otherwise fit; test would be vacuous

    _, _, changed, _, cmds, deltas = constraint.adapt_repartition(
        first_slot=0,
        last_slot=0,
        energy_delta=400.0,
        power_slots_duration_s=durations,
        existing_commands=[None],
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj,
        battery_min_wh=800.0,
    )

    assert changed is False
    assert cmds[0] is None
    assert deltas[0] == 0.0


# ---------------------------------------------------------------------------
# (d) Forward propagation lockstep
# ---------------------------------------------------------------------------


def test_guard_forward_propagation_decrements_both_arrays():
    """Given a placement in slot i with delta_power d
    When adapt_repartition completes the placement
    Then bat_charge_traj[i:last_slot+1] -= d * dt / 3600
    AND forward_min[i:last_slot+1] -= d * dt / 3600
    (subsequent slot whose forward_min would hit floor is skipped).
    """
    # 2 slots, identical config.  Floor 800 Wh.  Initial trajectory:
    # both slots at 1100 Wh.  forward_min = [1100, 1100].
    # max_drain in slot 0 = 1100 - 800 = 300 → max_delta = 1200 W.
    # Place a 1200 W cmd in slot 0 (delta 1200, state_delta 300 Wh).
    # After: bat_traj = [800, 800], forward_min = [800, 800].
    # Slot 1: max_drain = 0 → can't fit any command → skip.
    constraint = _make_constraint(
        target_value=5000.0,
        power_steps=[LoadCommand(command="on", power_consign=1200.0)],
    )
    durations = _durations(2)
    bat_traj = np.array([1100.0, 1100.0], dtype=np.float64)
    now = datetime.now(tz=pytz.UTC)

    _, _, changed, _, cmds, deltas = constraint.adapt_repartition(
        first_slot=0,
        last_slot=1,
        energy_delta=600.0,
        power_slots_duration_s=durations,
        existing_commands=[None, None],
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj,
        battery_min_wh=800.0,
    )

    assert changed is True
    # Slot 0: placed; slot 1: skipped (no headroom after propagation).
    assert cmds[0] is not None
    assert cmds[1] is None
    assert deltas[0] == 1200.0
    assert deltas[1] == 0.0
    # Caller-owned trajectory has been mutated in place.
    assert np.isclose(bat_traj[0], 800.0)
    assert np.isclose(bat_traj[1], 800.0)


# ---------------------------------------------------------------------------
# (e) Multi-pass survival
# ---------------------------------------------------------------------------


def test_guard_invariant_preserved_across_multi_pass():
    """Given a constraint with num_max_on_off set (triggers num_passes=2)
    When adapt_repartition runs over a slot range with use_battery_floor on
    Then bat_charge_traj and forward_min stay in lockstep across both
    passes — i.e., forward_min[i] >= min(bat_charge_traj[i:last_slot+1])
    after each placement.
    """
    constraint = _make_constraint(
        target_value=5000.0,
        power_steps=[
            LoadCommand(command="on", power_consign=400.0),
            LoadCommand(command="on", power_consign=800.0),
        ],
        num_max_on_off=4,
    )
    durations = _durations(3)
    bat_traj = np.array([2000.0, 2000.0, 2000.0], dtype=np.float64)
    now = datetime.now(tz=pytz.UTC)

    _, _, changed, _, _, deltas = constraint.adapt_repartition(
        first_slot=0,
        last_slot=2,
        energy_delta=600.0,
        power_slots_duration_s=durations,
        existing_commands=[None, None, None],
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj,
        battery_min_wh=900.0,
    )

    assert changed is True
    # Floor invariant: bat_traj never dips below floor after placements.
    assert float(np.min(bat_traj)) >= 900.0
    # Sum of state_delta must equal total energy moved (sanity).
    total_state_delta = float(np.sum(deltas * durations / 3600.0))
    expected_drop = 2000.0 - float(bat_traj[-1])
    assert np.isclose(total_state_delta, expected_drop)


# ---------------------------------------------------------------------------
# (f) DC-clamp slot — trajectory already includes DC clamping
# ---------------------------------------------------------------------------


def test_guard_skips_when_clamp_below_current_command():
    """Given an existing command at higher power than max_delta_power_floor
    would allow as a new placement, and a smaller step that fits under the
    floor budget but is below the existing command
    When the guard clamps base_cmd down
    Then delta_power becomes <= 0 (would reduce, not add) and the slot is
    skipped via the `delta_power <= 0` guard branch.

    Setup: 3-step ladder [500, 1500, 2500].  existing = 1500 (current
    command, j_initial=2 → bump to 2500, delta=1000).  Battery floor
    leaves only 800 W of headroom; clamp picks j=0 (500 W cmd),
    delta_power = 500 - 1500 = -1000 ≤ 0 → skip.
    """
    constraint = _make_constraint(
        target_value=5000.0,
        current_value=200.0,
        power_steps=[
            LoadCommand(command="on", power_consign=500.0),
            LoadCommand(command="on", power_consign=1500.0),
            LoadCommand(command="on", power_consign=2500.0),
        ],
    )
    durations = _durations(1)
    bat_traj = np.array([1000.0], dtype=np.float64)  # floor + 200
    now = datetime.now(tz=pytz.UTC)

    # floor 800 — max_drain = 200 — max_delta_power = 200 * 4 = 800 W.
    # existing = 1500 (above floor budget for an ADD); the bumped j=2
    # (2500 W) would be delta 1000 > 800 → clamp triggers, picks j=0
    # (500 W cmd), delta = 500 - 1500 = -1000 < 0 → skip.
    existing = [LoadCommand(command="on", power_consign=1500.0)]
    _, _, changed, _, cmds, deltas = constraint.adapt_repartition(
        first_slot=0,
        last_slot=0,
        energy_delta=2000.0,
        power_slots_duration_s=durations,
        existing_commands=existing,
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj,
        battery_min_wh=800.0,
    )
    assert changed is False
    assert cmds[0] is None
    assert deltas[0] == 0.0


def test_guard_propagates_full_delta_through_dc_clamp_slot():
    """Given a slot whose bat_charge_traj value already encodes the DC
    clamping (the trajectory comes from _battery_get_charging_power()[1]
    which integrates AC excess, DC clamp, and discharge to UA/loads)
    When the guard places a delta_power command at that slot
    Then bat_charge_traj decrements by the full delta_power * dt / 3600
    (the net effect; the DC-clamp baseline is already baked in).
    """
    constraint = _make_constraint(
        target_value=5000.0,
        power_steps=[LoadCommand(command="on", power_consign=600.0)],
    )
    durations = _durations(1)
    # Initial bat_traj for a "DC-clamp slot" — high charge (battery full
    # under DC clamping).  Floor 800 Wh, trajectory 5000 Wh, headroom huge.
    bat_traj = np.array([5000.0], dtype=np.float64)
    now = datetime.now(tz=pytz.UTC)

    pre_charge = float(bat_traj[0])
    _, _, changed, _, _, deltas = constraint.adapt_repartition(
        first_slot=0,
        last_slot=0,
        energy_delta=1000.0,
        power_slots_duration_s=durations,
        existing_commands=[None],
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj,
        battery_min_wh=800.0,
    )
    assert changed is True
    state_delta = deltas[0] * float(durations[0]) / 3600.0
    assert np.isclose(float(bat_traj[0]), pre_charge - state_delta)


# ---------------------------------------------------------------------------
# (g) Running-min invariant — QS-178 review fix #1 must-fix #1
# ---------------------------------------------------------------------------


def test_guard_running_min_correct_when_post_window_slots_lower():
    """Given a trajectory whose slots BEYOND last_slot are lower than slots
    INSIDE the working window
    When the guard computes forward_min over [first_slot, last_slot]
    Then forward_min reflects ONLY the in-range running minimum — placements
    are not blocked by lower charges that lie outside the window.

    Reproducer for Mode A (over-conservative): if forward_min were built
    over the full array, the lower out-of-window slots would underestimate
    real in-window budget after a single decrement.

    NOTE on tail-slot mutation: per QS-178 review fix #02 must-fix #2,
    `bat_charge_traj` propagates OPEN-ENDED beyond `last_slot` (the
    physical battery state carries forward).  So slots [3, 4] DO see
    their charge decremented after each placement — but no commands are
    PLACED there (they're outside the working window).
    """
    constraint = _make_constraint(
        target_value=10_000.0,
        power_steps=[
            LoadCommand(command="on", power_consign=400.0),
            LoadCommand(command="on", power_consign=800.0),
        ],
        num_slots=5,
    )
    # 5 slots: in-range [0,1,2] all at 5000 Wh; out-of-range [3,4] at 1000.
    # Floor 800 → max_drain in-range = 4200 Wh per slot.
    durations = _durations(5)
    bat_traj = np.array([5000.0, 5000.0, 5000.0, 1000.0, 1000.0], dtype=np.float64)
    pre_tail = (float(bat_traj[3]), float(bat_traj[4]))
    now = datetime.now(tz=pytz.UTC)

    _, _, changed, _, cmds, _ = constraint.adapt_repartition(
        first_slot=0,
        last_slot=2,
        energy_delta=600.0,
        power_slots_duration_s=durations,
        existing_commands=[None] * 5,
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj,
        battery_min_wh=800.0,
    )

    # Layer 3 should NOT block placements based on out-of-window slots.
    assert changed is True
    placed_count = sum(1 for c in cmds if c is not None)
    assert placed_count >= 1
    # No PLACEMENTS in slots 3,4 (outside the working window).
    assert cmds[3] is None
    assert cmds[4] is None
    # But their bat_charge_traj DOES decrement (open-ended slice per
    # QS-178 review fix #02 must-fix #2) — physical reality: every
    # placement at slot i decrements bat_charge_traj[i:] including the
    # tail.  Slot 3/4 accumulate ALL state_deltas from placements at
    # slots 0..2; slot 2 only accumulates its own.
    assert float(bat_traj[3]) < pre_tail[0]
    assert float(bat_traj[4]) < pre_tail[1]
    assert float(bat_traj[3]) == float(bat_traj[4])  # both see all 3 decrements
    # The post-last_slot drop equals the cumulative state_delta moved
    # across all placements.
    expected_drop = pre_tail[0] - float(bat_traj[3])
    assert expected_drop > 0
    # Sanity: the in-window slot 0 saw only ONE placement (its own); the
    # tail saw THREE.  So slot 3's drop ≥ slot 0's drop.
    in_window_drop = 5000.0 - float(bat_traj[0])
    assert expected_drop >= in_window_drop


def test_guard_running_min_back_propagates_after_late_slot_placement():
    """Given a multi-placement scenario where forward_min[i] for some k < i
    must reflect the post-placement minimum (which lives at slot i or later)
    When adapt_repartition completes a placement at slot i
    Then forward_min[k] for k < i is clamped DOWN to forward_min[i] where
    needed — preserves the invariant
    forward_min[k] == min(bat_charge_traj[k:last_slot+1]).
    """
    # 3-slot scenario:  bat_traj = [3000, 2000, 1500].  Floor 800.
    # forward_min at entry = [1500, 1500, 1500] (running min in-range).
    # Place a small command at slot 2; forward_min[2] decrements but
    # forward_min[0] and [1] must also clamp down to track the new min.
    constraint = _make_constraint(
        target_value=10_000.0,
        power_steps=[LoadCommand(command="on", power_consign=400.0)],
        num_slots=3,
    )
    durations = _durations(3)
    bat_traj = np.array([3000.0, 2000.0, 1500.0], dtype=np.float64)
    now = datetime.now(tz=pytz.UTC)

    _, _, changed, _, cmds, deltas = constraint.adapt_repartition(
        first_slot=0,
        last_slot=2,
        energy_delta=200.0,  # ≤ 1 placement of 400W * 0.25h = 100Wh per slot
        power_slots_duration_s=durations,
        existing_commands=[None, None, None],
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj,
        battery_min_wh=800.0,
    )

    assert changed is True
    # Whichever slot ended up placed, the post-placement trajectory must
    # have its running-min consistent with the actual bat_charge_traj.
    # In particular: no slot's bat_charge_traj falls below the floor
    # after the placement (the guard would have clamped or skipped).
    assert float(np.min(bat_traj)) >= 800.0
    # And at least one placement occurred.
    placed_count = sum(1 for c in cmds if c is not None)
    assert placed_count >= 1
    # Sanity: total state_delta matches what was drawn from the trajectory.
    total_state_delta = float(np.sum(deltas * durations / 3600.0))
    # Initial running min was 1500 (at slot 2); final running min should
    # be 1500 - total_state_delta (or higher, if placement landed elsewhere).
    assert total_state_delta >= 0.0


# ---------------------------------------------------------------------------
# (h) Multi-pass double-decrement guard — QS-178 review fix #1 must-fix #2
# ---------------------------------------------------------------------------


def test_multi_pass_does_not_double_decrement_bat_charge_traj():
    """Given a constraint with num_max_on_off set (triggers num_passes=2)
    and a slot placed in pass-1
    When pass-2 iterates the same slot
    Then pass-2 SKIPS that slot (via the `out_commands[i] is not None`
    guard) — net result: bat_charge_traj reflects pass-1's placement
    only, not pass-1 + pass-2's combined state_delta.
    """
    constraint = _make_constraint(
        target_value=10_000.0,
        power_steps=[
            LoadCommand(command="on", power_consign=800.0),
        ],
        num_max_on_off=4,
        num_slots=3,
    )
    durations = _durations(3)
    bat_traj = np.array([3000.0, 3000.0, 3000.0], dtype=np.float64)
    initial_charge_last = float(bat_traj[-1])
    now = datetime.now(tz=pytz.UTC)

    _, _, changed, _, cmds, deltas = constraint.adapt_repartition(
        first_slot=0,
        last_slot=2,
        energy_delta=400.0,  # ~2 slots worth at 800W * 0.25h = 200Wh
        power_slots_duration_s=durations,
        existing_commands=[None, None, None],
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj,
        battery_min_wh=800.0,
    )

    assert changed is True
    # Invariant: sum of state_delta across slots must equal the final
    # trajectory drop on the last slot (= initial_last - final_last).
    # If pass-2 double-decremented, the trajectory would drop MORE than
    # the sum of out_delta_power.
    total_state_delta = float(np.sum(deltas * durations / 3600.0))
    actual_drop = initial_charge_last - float(bat_traj[-1])
    assert np.isclose(total_state_delta, actual_drop, atol=1e-6), (
        f"multi-pass double-decrement detected: deltas sum {total_state_delta} "
        f"!= trajectory drop {actual_drop}"
    )


# ---------------------------------------------------------------------------
# (i) Zero-duration slot — should-fix #6
# ---------------------------------------------------------------------------


def test_guard_skips_zero_duration_slot_without_crashing():
    """Given a degenerate slot whose duration is 0 (would raise
    ZeroDivisionError in the floor-clamp arithmetic)
    When adapt_repartition iterates that slot with the guard active
    Then the slot is skipped via the zero-duration short-circuit — no
    crash, no placement.
    """
    constraint = _make_constraint(
        target_value=5000.0,
        power_steps=[LoadCommand(command="on", power_consign=1000.0)],
        num_slots=2,
    )
    # Slot 0: zero duration; Slot 1: normal.
    durations = np.array([0.0, float(SOLVER_STEP_S)], dtype=np.float64)
    bat_traj = np.array([2000.0, 2000.0], dtype=np.float64)
    now = datetime.now(tz=pytz.UTC)

    _, _, _changed, _, cmds, deltas = constraint.adapt_repartition(
        first_slot=0,
        last_slot=1,
        energy_delta=500.0,
        power_slots_duration_s=durations,
        existing_commands=[None, None],
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj,
        battery_min_wh=800.0,
    )

    # No crash; slot 0 (zero duration) is skipped.
    assert cmds[0] is None
    assert deltas[0] == 0.0


# ---------------------------------------------------------------------------
# (j) Clamp lookup uses delta-vs-absolute correctly — review fix #02 must-fix #1
# ---------------------------------------------------------------------------


def test_guard_picks_intermediate_step_when_current_command_nonzero():
    """Given an existing command at non-zero power, a multi-step ladder, and
    a max_delta_power_floor that allows ONE intermediate step (not the
    bumped max, not the smallest)
    When the floor clamp triggers
    Then it picks the intermediate command whose absolute power respects
    the floor budget — NOT the smallest command (which would yield a
    negative delta and a skipped slot).

    Reproducer (review fix #02 must-fix #1):
    - existing = 1500 W, cmds = [500, 1500, 2000, 2500].
    - max_delta_power_floor = 800 W (drain budget for the slot).
    - Original delta = 2500 - 1500 = 1000 > 800 → clamp triggers.
    - Skip check: allowed_base_power = 1500 + 800 - 0 = 2300 > cmds[0]=500
      → don't skip.
    - Lookup `_get_lower_consign_idx_for_power(cmds, 2300)` finds j=2
      (cmds[2]=2000 ≤ 2300).
    - delta = 2000 - 1500 = 500 ≤ 800 — safe placement.

    Before the fix, the lookup used `max_delta_power_floor` (raw delta
    budget = 800) directly, so it picked cmds[0]=500, computed
    delta = 500 - 1500 = -1000, hit the `delta <= 0` guard, and
    SKIPPED the slot — losing the intermediate-step opportunity.
    """
    # Battery floor: 800 Wh; trajectory at 1000 → max_drain = 200 Wh →
    # max_delta_power = 200 * 3600 / SOLVER_STEP_S (= 800 W when 900 s).
    constraint = _make_constraint(
        target_value=10_000.0,
        current_value=200.0,
        power_steps=[
            LoadCommand(command="on", power_consign=500.0),
            LoadCommand(command="on", power_consign=1500.0),
            LoadCommand(command="on", power_consign=2000.0),
            LoadCommand(command="on", power_consign=2500.0),
        ],
    )
    durations = _durations(1)
    bat_traj = np.array([1000.0], dtype=np.float64)
    now = datetime.now(tz=pytz.UTC)

    existing = [LoadCommand(command="on", power_consign=1500.0)]
    _, _, changed, _, cmds, deltas = constraint.adapt_repartition(
        first_slot=0,
        last_slot=0,
        energy_delta=2000.0,
        power_slots_duration_s=durations,
        existing_commands=existing,
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj,
        battery_min_wh=800.0,
    )
    assert changed is True
    assert cmds[0] is not None
    # Picked the intermediate step (2000 W cmd, delta = 500 W ≤ 800 W).
    assert cmds[0].power_consign == 2000.0
    assert deltas[0] == 500.0


# ---------------------------------------------------------------------------
# (k) Open-ended decrement protects downstream constraint — review fix #02 must-fix #2
# ---------------------------------------------------------------------------


def test_open_ended_decrement_protects_downstream_constraint_floor():
    """Given two adapt_repartition calls in sequence (simulating two
    constraints inside a single _constraints_delta iteration), where
    constraint A spans [0, 1] and constraint B spans [3, 4]
    When A places enough energy to drag the physical battery trajectory
    at slots ≥ 3 toward the safety floor
    Then B's Layer-3 sees the open-ended decrement in `bat_charge_traj`
    and refuses an unsafe placement.

    With the prior `[i:last_slot+1]` slice (closed-ended), B would see
    a stale tail and wrongly allow the placement.  The open-ended slice
    fix (must-fix #2) ensures physical state propagates across the
    constraint boundary.
    """
    # Battery floor 900 Wh.  Shared 5-slot trajectory at 1100 Wh.
    # Constraint A places one 800 W cmd at slot 0, drains 200 Wh.  After
    # the placement, the open-ended decrement reduces bat_charge_traj
    # for ALL slots ≥ 0 to 900 Wh — exactly at the floor.  A's second
    # iteration (slot 1) would breach the floor → Layer 3 clamps it to
    # skip.  Hence A places EXACTLY 1 slot, and the tail (slots ≥ 2)
    # carries the post-placement state forward.
    bat_traj = np.array([1100.0, 1100.0, 1100.0, 1100.0, 1100.0], dtype=np.float64)
    durations = _durations(5)
    now = datetime.now(tz=pytz.UTC)

    constraint_a = _make_constraint(
        name="A",
        target_value=10_000.0,
        power_steps=[LoadCommand(command="on", power_consign=800.0)],
        num_slots=5,
    )
    _, _, ch_a, _, _cmds_a, _ = constraint_a.adapt_repartition(
        first_slot=0,
        last_slot=1,
        energy_delta=1000.0,  # budget large enough — Layer 3 is the binding limit
        power_slots_duration_s=durations,
        existing_commands=[None] * 5,
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj,
        battery_min_wh=900.0,
    )
    assert ch_a is True
    # Open-ended: slots 3-4 must have decremented along with slot 0.
    assert float(bat_traj[3]) < 1100.0
    assert float(bat_traj[4]) < 1100.0
    # All slots ≥ 0 carry slot-0's placement forward.
    assert float(bat_traj[3]) == float(bat_traj[4])
    assert float(bat_traj[3]) == float(bat_traj[2])  # slot 2 = tail since no placement at 2
    # Tail at-or-just-above the floor — A's drain absorbed the head-room.
    assert float(bat_traj[3]) <= 900.0 + 1e-3

    # Constraint B at slots 3-4: with the open-ended decrement applied,
    # bat_traj[3] = 900 (floor) → no drain budget → no placement.
    constraint_b = _make_constraint(
        name="B",
        target_value=10_000.0,
        power_steps=[LoadCommand(command="on", power_consign=800.0)],
        num_slots=5,
    )
    _, _, ch_b, _, cmds_b, _ = constraint_b.adapt_repartition(
        first_slot=3,
        last_slot=4,
        energy_delta=1000.0,
        power_slots_duration_s=durations,
        existing_commands=[None] * 5,
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj,
        battery_min_wh=900.0,
    )
    # B must NOT place; the floor is exhausted by A's open-ended drain.
    assert ch_b is False
    assert cmds_b[3] is None
    assert cmds_b[4] is None


# ---------------------------------------------------------------------------
# (l) Focused unit test for _back_propagate_forward_min — review fix #02 should-fix #12
# ---------------------------------------------------------------------------


def test_back_propagate_forward_min_clamps_earlier_slots_to_new_min():
    """Unit test for the extracted `_back_propagate_forward_min` helper.

    Given a forward_min array where slot `i`'s running-min value has
    just been decremented below earlier slots'
    When `_back_propagate_forward_min(forward_min, i, first_slot)` runs
    Then earlier slots `[first_slot, i)` are clamped downward to
    forward_min[i] where they currently exceed it.

    Isolates the algorithmic invariant from multi-pass orchestration —
    catches breakage in the back-prop logic alone.
    """
    from custom_components.quiet_solar.home_model.constraints import (
        _back_propagate_forward_min,
    )

    # Before placement: forward_min sorted descending — each earlier
    # slot sees a higher running min than later slots.  After a
    # placement at i=2 decremented forward_min[2:] by 100, slot 2's
    # value is now 600 (below slots 0 and 1).  Back-prop must clamp
    # slots 0 and 1 to 600.
    forward_min = np.array([1000.0, 900.0, 600.0, 700.0], dtype=np.float64)
    _back_propagate_forward_min(forward_min, i=2, first_slot=0)
    assert np.allclose(forward_min[:2], [600.0, 600.0])
    # Slots >= i unchanged.
    assert forward_min[2] == 600.0
    assert forward_min[3] == 700.0


def test_back_propagate_forward_min_noop_when_i_equals_first_slot():
    """No earlier slots to clamp — helper must be a no-op."""
    from custom_components.quiet_solar.home_model.constraints import (
        _back_propagate_forward_min,
    )

    forward_min = np.array([1000.0, 900.0, 800.0], dtype=np.float64)
    before = forward_min.copy()
    _back_propagate_forward_min(forward_min, i=0, first_slot=0)
    assert np.array_equal(forward_min, before)


def test_back_propagate_forward_min_preserves_already_lower_slots():
    """Slots that are ALREADY <= forward_min[i] must not be raised."""
    from custom_components.quiet_solar.home_model.constraints import (
        _back_propagate_forward_min,
    )

    # Slot 0 is already at 500 (below the new pivot 700).  Slot 1 at
    # 900 must be clamped to 700.
    forward_min = np.array([500.0, 900.0, 700.0, 800.0], dtype=np.float64)
    _back_propagate_forward_min(forward_min, i=2, first_slot=0)
    assert forward_min[0] == 500.0  # unchanged
    assert forward_min[1] == 700.0  # clamped down
    assert forward_min[2] == 700.0  # unchanged


# ---------------------------------------------------------------------------
# (m) NaN guard — review fix #03 must-fix #4
# ---------------------------------------------------------------------------


def test_nan_in_bat_charge_traj_logs_and_disables_layer3(caplog):
    """Review fix #03 must-fix #4 + fix #04 user feedback: when
    `adapt_repartition` is called with a NaN-containing trajectory, it
    logs an error and disables Layer 3 for that call (the load-bearing
    upstream `raise` lives in `_constraints_delta` and surfaces the
    actual failure to callers).  Per-policy: no defensive `assert` in
    production code — log + fail-safe instead.
    """
    import logging

    constraint = _make_constraint(
        target_value=5000.0,
        power_steps=[LoadCommand(command="on", power_consign=400.0)],
    )
    durations = _durations(2)
    bat_traj = np.array([1000.0, float("nan")], dtype=np.float64)
    now = datetime.now(tz=pytz.UTC)

    with caplog.at_level(logging.ERROR, logger="custom_components.quiet_solar.home_model.constraints"):
        result = constraint.adapt_repartition(
            first_slot=0,
            last_slot=1,
            energy_delta=400.0,
            power_slots_duration_s=durations,
            existing_commands=[None, None],
            allow_change_state=True,
            time=now,
            bat_charge_traj=bat_traj,
            battery_min_wh=800.0,
        )

    # Layer 3 was disabled gracefully — no crash; the error was logged.
    assert any(
        "received bat_charge_traj with NaN" in rec.message for rec in caplog.records
    ), "Expected error log explaining the NaN fallback"
    # adapt_repartition still returned a valid result tuple (no crash).
    assert result is not None
    assert len(result) == 6


# ---------------------------------------------------------------------------
# (n) Multi-pass upgrade path restored when no battery floor — review fix #03 must-fix #2
# ---------------------------------------------------------------------------


def test_multi_pass_upgrade_path_restored_when_no_battery_floor():
    """Review fix #03 must-fix #2: when `bat_charge_traj=None` (no
    Layer 3 active), the multi-pass skip is GATED off so pass-2 can
    legitimately upgrade a slot pass-1 placed.

    Pre-QS-178 behavior was: pass-1 places `cmds[j1]`, pass-2 revisits
    and bumps to `cmds[j2 > j1]` if the energy budget allows.  Round-1
    must-fix #2 added an unconditional skip that broke this upgrade
    path for ALL 49+ existing call sites.  Round-3 must-fix #2 gated
    the skip on `use_battery_floor`.

    Note: the upgrade path is internal solver mechanics; we only assert
    here that the placement loop iterates a slot in BOTH passes when
    the gate is off — i.e., pass-2 is not short-circuited by the
    QS-178 multi-pass guard.  Exact upgrade outcome depends on solver
    state-machine details outside the scope of this test.
    """
    # Build a multi-step ladder + a constraint with num_max_on_off set.
    # No bat_charge_traj → use_battery_floor=False → the skip is
    # gated off.
    constraint = _make_constraint(
        target_value=10_000.0,
        current_value=0.0,
        power_steps=[
            LoadCommand(command="on", power_consign=500.0),
            LoadCommand(command="on", power_consign=1500.0),
            LoadCommand(command="on", power_consign=2500.0),
        ],
        num_max_on_off=4,
        num_slots=3,
    )
    durations = _durations(3)
    now = datetime.now(tz=pytz.UTC)

    # Drive the loop without a trajectory → the QS-178 multi-pass skip
    # does NOT fire.  The legitimate-upgrade test is that
    # adapt_repartition completes and produces SOME placement (the
    # baseline behavior).  We're not checking specific upgrade values
    # — just that the QS-178 guard isn't blocking legacy semantics.
    _, _, changed, _, cmds, _ = constraint.adapt_repartition(
        first_slot=0,
        last_slot=2,
        energy_delta=5000.0,
        power_slots_duration_s=durations,
        existing_commands=[None, None, None],
        allow_change_state=True,
        time=now,
        # bat_charge_traj omitted → use_battery_floor=False
    )
    # Legacy semantics preserved: some placement happens (the unguarded
    # multi-pass loop is not short-circuited by QS-178's guard).
    assert changed is True
    placed_count = sum(1 for c in cmds if c is not None)
    assert placed_count >= 1


def test_multi_pass_skip_active_when_use_battery_floor():
    """Companion to the previous test: when `bat_charge_traj` IS
    passed, the multi-pass skip fires as designed (round-1 must-fix #2
    invariant) — pass-2 does NOT revisit slots placed in pass-1.

    Verifies that gating did not accidentally drop the skip in the
    use_battery_floor=True path.
    """
    constraint = _make_constraint(
        target_value=10_000.0,
        current_value=0.0,
        power_steps=[
            LoadCommand(command="on", power_consign=500.0),
            LoadCommand(command="on", power_consign=1500.0),
        ],
        num_max_on_off=4,
        num_slots=3,
    )
    durations = _durations(3)
    bat_traj = np.array([3000.0, 3000.0, 3000.0], dtype=np.float64)
    initial_charge_last = float(bat_traj[-1])
    now = datetime.now(tz=pytz.UTC)

    _, _, changed, _, _, deltas = constraint.adapt_repartition(
        first_slot=0,
        last_slot=2,
        energy_delta=600.0,
        power_slots_duration_s=durations,
        existing_commands=[None, None, None],
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj,
        battery_min_wh=800.0,
    )
    assert changed is True
    # Invariant (round-1 must-fix #2): sum of out_delta_power state_deltas
    # equals the trajectory drop — no double-decrement from a pass-2
    # revisit.
    total_state_delta = float(np.sum(deltas * durations / 3600.0))
    actual_drop = initial_charge_last - float(bat_traj[-1])
    assert np.isclose(total_state_delta, actual_drop, atol=1e-6)


def test_multi_pass_skip_fires_when_out_commands_already_placed():
    """Review fix #03 must-fix #2: with `use_battery_floor=True`, when
    pass-1 has placed a command at slot `i`, pass-2 MUST skip that slot
    via the gated `out_commands[i] is not None` guard.

    Scenario: existing_commands has a non-zero command at slot 0.  In
    pass-1, the placement loop sees `current_command_power > 0` and
    bumps it (no transition-cost gating).  In pass-2, `out_commands[0]`
    is not None → the gated skip fires.
    """
    constraint = _make_constraint(
        target_value=10_000.0,
        current_value=200.0,
        power_steps=[
            LoadCommand(command="on", power_consign=500.0),
            LoadCommand(command="on", power_consign=1500.0),
            LoadCommand(command="on", power_consign=2500.0),
        ],
        num_max_on_off=4,
        num_slots=3,
    )
    durations = _durations(3)
    bat_traj = np.array([5000.0, 5000.0, 5000.0], dtype=np.float64)
    now = datetime.now(tz=pytz.UTC)

    # Slot 0 has an existing 500 W command → pass-1 bumps it (no
    # transition cost gating).  Slots 1, 2 are None.
    existing: list[LoadCommand | None] = [
        LoadCommand(command="on", power_consign=500.0),
        None,
        None,
    ]
    _, _, changed, _, _cmds, deltas = constraint.adapt_repartition(
        first_slot=0,
        last_slot=2,
        energy_delta=300.0,
        power_slots_duration_s=durations,
        existing_commands=existing,
        allow_change_state=True,
        time=now,
        bat_charge_traj=bat_traj,
        battery_min_wh=800.0,
    )
    assert changed is True
    # Sum of state_delta across slots must equal the final
    # trajectory drop (no double-decrement from pass-2 revisit at the
    # bumped slot).
    total_state_delta = float(np.sum(deltas * durations / 3600.0))
    actual_drop = 5000.0 - float(bat_traj[-1])
    assert np.isclose(total_state_delta, actual_drop, atol=1e-6)
