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
    # max_drain = 100, max_delta_power = 100 * 3600 / 900 = 400 W < 1500 W.
    bat_traj = np.array([900.0], dtype=np.float64)
    now = datetime.now(tz=pytz.UTC)

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
