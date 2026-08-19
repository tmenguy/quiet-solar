"""Solver-side tests for the QS-349 battery discharge safety floor.

The floor F is `Battery.min_discharging_power`. The solver must model the
real hardware behaviour: during a `CMD_GREEN_CHARGE_ONLY` slot the battery
still discharges up to F when house demand exceeds solar. The price-bucket
optimizer may only move the *compressible* part of a slot's discharge
(everything above the incompressible leak `min(discharge, F * duration)`).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest
import pytz

from custom_components.quiet_solar.const import (
    CONF_BATTERY_CAPACITY,
    CONF_BATTERY_MAX_CHARGE_POWER_VALUE,
    CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE,
    CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE,
)
from custom_components.quiet_solar.home_model.battery import Battery
from custom_components.quiet_solar.home_model.commands import (
    CMD_GREEN_CHARGE_AND_DISCHARGE,
    CMD_GREEN_CHARGE_ONLY,
    copy_command,
)
from custom_components.quiet_solar.home_model.solver import PeriodSolver


def _build_solver(
    *,
    floor: float,
    pv_w: float,
    ua_w: float,
    current_charge: float,
    tariffs=0.2,
    hours: int = 1,
    minutes: int | None = None,
    capacity: float = 10000.0,
    max_dis: float = 5000.0,
):
    """Build a PeriodSolver + battery with a controlled uniform scenario."""
    start = datetime(2024, 6, 1, 0, 0, tzinfo=pytz.UTC)
    span = timedelta(minutes=minutes) if minutes is not None else timedelta(hours=hours)
    end = start + span
    pv = [(start, pv_w), (end, pv_w)]
    ua = [(start, ua_w), (end, ua_w)]
    battery = Battery(
        name="bat",
        **{
            CONF_BATTERY_CAPACITY: capacity,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: max_dis,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: max_dis,
            CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: floor,
        },
    )
    battery._current_charge_value = current_charge
    solver = PeriodSolver(
        start_time=start,
        end_time=end,
        tariffs=tariffs,
        actionable_loads=[],
        battery=battery,
        pv_forecast=pv,
        unavoidable_consumption_forecast=ua,
    )
    return solver, battery


def _green_only_cmds(solver):
    num = len(solver._available_power_no_battery)
    return [copy_command(CMD_GREEN_CHARGE_ONLY) for _ in range(num)]


# ---------------------------------------------------------------------------
# AC 8 — solver models the floor during CMD_GREEN_CHARGE_ONLY slots
# ---------------------------------------------------------------------------


def test_green_only_slot_discharges_up_to_floor_when_demand_exceeds_floor():
    """Demand > F: modelled discharge is exactly F (the binding bound)."""
    solver, battery = _build_solver(floor=300, pv_w=1000, ua_w=3000, current_charge=8000)
    result = solver._battery_get_charging_power(existing_battery_commands=_green_only_cmds(solver))
    battery_charge, battery_commands = result[1], result[2]
    leak_buckets = result[8]
    dur_h = float(solver._durations_s[0]) / 3600.0

    # capped at the floor, not 0 and not the full 2000 W demand
    assert battery_commands[0].power_consign == pytest.approx(-300.0)
    assert battery_charge[0] == pytest.approx(8000.0 - 300.0 * dur_h)
    # the whole (floor-capped) discharge is incompressible leak
    price = solver._prices[0]
    assert leak_buckets[price] == pytest.approx(300.0 * dur_h * len(battery_commands))


def test_green_only_slot_discharges_only_demand_when_demand_below_floor():
    """Demand < F: modelled discharge is the demand; leak is the demand's energy."""
    solver, _ = _build_solver(floor=300, pv_w=1000, ua_w=1100, current_charge=8000)
    result = solver._battery_get_charging_power(existing_battery_commands=_green_only_cmds(solver))
    battery_commands = result[2]
    leak_buckets = result[8]
    dur_h = float(solver._durations_s[0]) / 3600.0

    assert battery_commands[0].power_consign == pytest.approx(-100.0)
    price = solver._prices[0]
    assert leak_buckets[price] == pytest.approx(100.0 * dur_h * len(battery_commands))


def test_green_only_surplus_slot_still_charges():
    """A surplus slot keeps charging (the floor cap never blocks charging)."""
    solver, _ = _build_solver(floor=300, pv_w=3000, ua_w=1000, current_charge=8000)
    result = solver._battery_get_charging_power(existing_battery_commands=_green_only_cmds(solver))
    battery_commands = result[2]
    leak_buckets = result[8]

    assert battery_commands[0].power_consign > 0.0
    # a charging slot contributes no leak
    assert leak_buckets == {}


def test_green_only_discharge_soc_limited_below_floor():
    """AC 8 SOC-bound companion: near-empty SOC binds discharge below F."""
    # current_charge 50 Wh over a 900 s slot => possible discharge 200 W < F
    solver, _ = _build_solver(floor=300, pv_w=1000, ua_w=3000, current_charge=50, capacity=10000)
    result = solver._battery_get_charging_power(existing_battery_commands=_green_only_cmds(solver))
    battery_commands = result[2]

    assert battery_commands[0].power_consign == pytest.approx(-200.0)


def test_floor_zero_green_only_forbids_discharge():
    """F = 0 reduces to today's behaviour: a demand slot never discharges."""
    solver, _ = _build_solver(floor=0, pv_w=1000, ua_w=3000, current_charge=8000)
    result = solver._battery_get_charging_power(existing_battery_commands=_green_only_cmds(solver))
    battery_commands = result[2]
    leak_buckets = result[8]

    assert battery_commands[0].power_consign == pytest.approx(0.0)
    assert leak_buckets == {}


# ---------------------------------------------------------------------------
# AC 9 — leak-aware flip site (compressible vs incompressible discharge)
# ---------------------------------------------------------------------------


def test_flip_keeps_leak_and_preserves_residual_budget():
    """A flipped slot discharges only its leak (-F) and consumes NO budget.

    Post-clamp debit: the leak is incompressible, so a flipped slot's surviving
    discharge is exactly the leak and it debits nothing — the residual budget
    stays available for later same-price slots (matches pre-QS-349 semantics).
    """
    solver, _ = _build_solver(floor=300, pv_w=0, ua_w=2000, current_charge=8000, minutes=15)
    price = solver._prices[0]
    dur_h = float(solver._durations_s[0]) / 3600.0
    # slot discharge = 2000 W => 500 Wh; leak = 300 W * 0.25 h = 75 Wh;
    # compressible = 425 Wh. Budget 300 Wh < compressible => flip.
    budget = {price: 300.0}
    cmds = [copy_command(CMD_GREEN_CHARGE_AND_DISCHARGE) for _ in solver._available_power_no_battery]

    result = solver._battery_get_charging_power(existing_battery_commands=cmds, limited_discharge_per_price=budget)
    battery_commands = result[2]

    # first slot flipped to green-charge-only, discharging only the leak (-F)
    assert battery_commands[0].is_like(CMD_GREEN_CHARGE_ONLY)
    assert battery_commands[0].power_consign == pytest.approx(-300.0)
    # residual budget preserved (flipped slot consumes none of it)
    assert budget[price] == pytest.approx(300.0)
    # leak energy per slot is F * duration
    assert result[8][price] == pytest.approx(300.0 * dur_h * len(battery_commands))


def test_no_flip_when_compressible_budget_survives():
    """Ample budget: no flip, full discharge, budget debited by compressible part."""
    solver, _ = _build_solver(floor=300, pv_w=0, ua_w=2000, current_charge=8000, minutes=15)
    price = solver._prices[0]
    budget = {price: 1000.0}
    cmds = [copy_command(CMD_GREEN_CHARGE_AND_DISCHARGE) for _ in solver._available_power_no_battery]

    result = solver._battery_get_charging_power(existing_battery_commands=cmds, limited_discharge_per_price=budget)
    battery_commands = result[2]

    assert not battery_commands[0].is_like(CMD_GREEN_CHARGE_ONLY)
    assert battery_commands[0].power_consign == pytest.approx(-2000.0)
    # 1000 - (500 - 75) = 575 Wh remaining
    assert budget[price] == pytest.approx(575.0)


def test_flip_debit_f_zero_preserves_residual_and_lets_smaller_slots_discharge():
    """AC 3 regression (M1): at F = 0 the flip debit reduces to pre-QS-349 behaviour.

    A budget smaller than a big slot's discharge flips that slot (holding charge)
    WITHOUT consuming the budget, so a later smaller same-price slot can still
    discharge. This is the exact no-op invariant the pre-clamp formula broke.
    """
    solver, _ = _build_solver(floor=0, pv_w=0, ua_w=2000, current_charge=8000, minutes=30)
    # two same-price slots: a big one (2400 W -> 600 Wh) then a small one
    # (1200 W -> 300 Wh); budget 500 Wh sits between them
    solver._available_power_no_battery = np.array([2400.0, 1200.0])
    price = solver._prices[0]
    budget = {price: 500.0}
    cmds = [copy_command(CMD_GREEN_CHARGE_AND_DISCHARGE) for _ in solver._available_power_no_battery]

    result = solver._battery_get_charging_power(existing_battery_commands=cmds, limited_discharge_per_price=budget)
    battery_commands = result[2]

    # big slot flipped (holds charge), discharges nothing at F = 0
    assert battery_commands[0].is_like(CMD_GREEN_CHARGE_ONLY)
    assert battery_commands[0].power_consign == pytest.approx(0.0)
    # residual budget preserved across the flip
    # small slot NOT flipped: still discharges, debiting the budget by its energy
    assert not battery_commands[1].is_like(CMD_GREEN_CHARGE_ONLY)
    assert battery_commands[1].power_consign == pytest.approx(-1200.0)
    assert budget[price] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# AC 9 bullet 3 — leak normalization applied exactly once (R4)
# ---------------------------------------------------------------------------


def test_leak_normalize_discharged_buckets_subtracts_once():
    """The helper subtracts the leak exactly once (single subtraction)."""
    discharged = {0.1: 500.0, 0.3: 200.0}
    leak = {0.1: 75.0, 0.3: 50.0}
    result = PeriodSolver._leak_normalize_discharged_buckets(discharged, leak)
    assert result[0.1] == pytest.approx(425.0)
    assert result[0.3] == pytest.approx(150.0)
    # a SECOND call would double-subtract — proving why it must run once
    PeriodSolver._leak_normalize_discharged_buckets(result, leak)
    assert result[0.1] == pytest.approx(350.0)
    assert result[0.3] == pytest.approx(100.0)


def test_leak_normalize_discharged_buckets_clamps_at_zero():
    """A leak >= the discharged energy clamps the bucket at 0 (never negative)."""
    result = PeriodSolver._leak_normalize_discharged_buckets({0.1: 50.0}, {0.1: 75.0})
    assert result[0.1] == 0.0


# ---------------------------------------------------------------------------
# AC 9 — leak-aware price-bucket allocation over a full solve
# ---------------------------------------------------------------------------


def _two_price_tariffs(start, n_slots):
    """Alternating cheap/expensive prices, one entry per 15-min slot."""
    tariffs = []
    for i in range(n_slots):
        t = start + timedelta(minutes=15 * i)
        price = (0.10 if i % 2 == 0 else 0.30) / 1000.0
        tariffs.append((t, price))
    return tariffs


def test_full_solve_two_price_flipped_slots_keep_their_leak():
    """Leak-aware allocation: flipped cheap slots still discharge the floor.

    With a limited SOC and demand exceeding what the battery can serve, the
    allocation moves compressible discharge from cheap to expensive buckets.
    At F > 0 a flipped slot keeps its incompressible leak (never a hard 0),
    exercising the one-time bucket normalization before the allocation loop.
    """
    start = datetime(2024, 6, 1, 0, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=2)
    n_slots = 8
    tariffs = _two_price_tariffs(start, n_slots)
    pv = [(start + timedelta(hours=h), 0.0) for h in range(3)]
    ua = [(start + timedelta(hours=h), 2000.0) for h in range(3)]

    battery = Battery(
        name="bat",
        **{
            CONF_BATTERY_CAPACITY: 10000.0,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000.0,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000.0,
            CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: 300.0,
        },
    )
    battery._current_charge_value = 1000.0

    solver = PeriodSolver(
        start_time=start,
        end_time=end,
        tariffs=tariffs,
        actionable_loads=[],
        battery=battery,
        pv_forecast=pv,
        unavoidable_consumption_forecast=ua,
    )
    solver.solve(with_self_test=True)

    per_slot = solver._final_battery_commands
    assert per_slot is not None
    # every flipped (green-charge-only) slot keeps the leak: discharge >= -F,
    # never fully forbidden below the floor
    flipped = 0
    for cmd in per_slot:
        if cmd is not None and cmd.is_like(CMD_GREEN_CHARGE_ONLY):
            flipped += 1
            assert cmd.power_consign >= -300.0 - 1e-6
            assert cmd.power_consign <= 0.0
    # S3: the assertions above must not pass vacuously — AC 9 requires >= 1 flip
    assert flipped > 0


def test_two_price_no_phantom_savings_expensive_bucket_matches_recompute():
    """AC 9 / M2: the expensive-bucket grid residual has no F x flipped-duration shortfall.

    Constrain the cheap-price discharge budget to force flips on cheap slots, then
    re-simulate with the resulting commands. The expensive bucket's predicted grid
    residual must equal the recomputed one (the flip's accounting is leak-honest —
    it does not claim savings the hardware won't deliver).
    """
    start = datetime(2024, 6, 1, 0, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=2)
    tariffs = _two_price_tariffs(start, 8)
    pv = [(start + timedelta(hours=h), 0.0) for h in range(3)]
    ua = [(start + timedelta(hours=h), 2000.0) for h in range(3)]
    battery = Battery(
        name="bat",
        **{
            CONF_BATTERY_CAPACITY: 10000.0,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000.0,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000.0,
            CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: 300.0,
        },
    )
    battery._current_charge_value = 2000.0
    solver = PeriodSolver(
        start_time=start,
        end_time=end,
        tariffs=tariffs,
        actionable_loads=[],
        battery=battery,
        pv_forecast=pv,
        unavoidable_consumption_forecast=ua,
    )

    cheap = min(solver._prices_ordered_values)
    expensive = max(solver._prices_ordered_values)

    # plan: cap cheap-price discharge to force flips on cheap slots
    budget = {cheap: 200.0}
    r_plan = solver._battery_get_charging_power(limited_discharge_per_price=dict(budget))
    plan_cmds = r_plan[2]
    plan_expensive_grid = r_plan[4].get(expensive, 0.0)

    # at least one cheap slot flipped to the leak (non-vacuous)
    assert any(c.is_like(CMD_GREEN_CHARGE_ONLY) and c.power_consign == pytest.approx(-300.0) for c in plan_cmds)

    # recompute with the resulting commands, no budget
    recompute_cmds = [copy_command(c) for c in plan_cmds]
    r_re = solver._battery_get_charging_power(existing_battery_commands=recompute_cmds)
    re_expensive_grid = r_re[4].get(expensive, 0.0)

    assert re_expensive_grid == pytest.approx(plan_expensive_grid)
    assert plan_expensive_grid > 0.0  # non-vacuous: the expensive bucket does import


def test_full_solve_three_price_normalization_applied_once():
    """3-price companion: solve completes and never over-subtracts the leak.

    The inner allocation loop revisits cheap buckets on every outer
    iteration; the leak is normalized once before the loop so no discharged
    bucket is double-subtracted. The battery must still discharge at least
    its total leak across the horizon.
    """
    start = datetime(2024, 6, 1, 0, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=3)
    n_slots = 12
    tariffs = []
    for i in range(n_slots):
        t = start + timedelta(minutes=15 * i)
        price = (0.10 + 0.10 * (i % 3)) / 1000.0
        tariffs.append((t, price))
    pv = [(start + timedelta(hours=h), 0.0) for h in range(4)]
    ua = [(start + timedelta(hours=h), 2000.0) for h in range(4)]

    battery = Battery(
        name="bat",
        **{
            CONF_BATTERY_CAPACITY: 10000.0,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000.0,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000.0,
            CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: 300.0,
        },
    )
    battery._current_charge_value = 1500.0

    solver = PeriodSolver(
        start_time=start,
        end_time=end,
        tariffs=tariffs,
        actionable_loads=[],
        battery=battery,
        pv_forecast=pv,
        unavoidable_consumption_forecast=ua,
    )
    # R4: the leak normalization must run exactly ONCE (never per revisit)
    with patch.object(
        PeriodSolver,
        "_leak_normalize_discharged_buckets",
        wraps=PeriodSolver._leak_normalize_discharged_buckets,
    ) as normalize_spy:
        solver.solve(with_self_test=True)
    assert normalize_spy.call_count == 1

    per_slot = solver._final_battery_commands
    assert per_slot is not None
    # no flipped slot ever discharges more than the floor (leak only)
    flipped = 0
    for cmd in per_slot:
        if cmd is not None and cmd.is_like(CMD_GREEN_CHARGE_ONLY):
            flipped += 1
            assert cmd.power_consign >= -300.0 - 1e-6
            assert cmd.power_consign <= 0.0
    # S3: non-vacuous — the 3-price scenario must actually produce flips
    assert flipped > 0


def test_full_solve_two_price_floor_zero_reduces_to_baseline():
    """AC 3 / S4: an F = 0 full solve still runs the allocation (normalization no-op).

    Exercises the leak-normalization block with empty leak buckets (the loop
    iterates zero times) and confirms F = 0 never leaves a flipped slot
    discharging the floor (it discharges 0, as pre-QS-349).
    """
    start = datetime(2024, 6, 1, 0, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=2)
    tariffs = _two_price_tariffs(start, 8)
    pv = [(start + timedelta(hours=h), 0.0) for h in range(3)]
    ua = [(start + timedelta(hours=h), 2000.0) for h in range(3)]
    battery = Battery(
        name="bat",
        **{
            CONF_BATTERY_CAPACITY: 10000.0,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000.0,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000.0,
            CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: 0.0,
        },
    )
    battery._current_charge_value = 1000.0
    solver = PeriodSolver(
        start_time=start,
        end_time=end,
        tariffs=tariffs,
        actionable_loads=[],
        battery=battery,
        pv_forecast=pv,
        unavoidable_consumption_forecast=ua,
    )
    solver.solve(with_self_test=True)

    per_slot = solver._final_battery_commands
    assert per_slot is not None
    for cmd in per_slot:
        if cmd is not None and cmd.is_like(CMD_GREEN_CHARGE_ONLY):
            # F = 0: a flipped slot holds all charge (no leak)
            assert cmd.power_consign == pytest.approx(0.0)
