"""Solver-side tests for the QS-349 battery discharge safety floor.

The floor F is `Battery.min_discharging_power`. The solver must model the
real hardware behaviour: during a `CMD_GREEN_CHARGE_ONLY` slot the battery
still discharges up to F when house demand exceeds solar. The price-bucket
optimizer may only move the *compressible* part of a slot's discharge
(everything above the incompressible leak `min(discharge, F * duration)`).
"""

from __future__ import annotations

from datetime import datetime, timedelta

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


def test_flip_keeps_leak_and_debits_only_compressible_budget():
    """A flipped slot discharges its leak (not 0); budget debited compressible-only."""
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
    # compressible budget fully consumed on this flip
    assert budget[price] == pytest.approx(0.0)
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
    for cmd in per_slot:
        if cmd is not None and cmd.is_like(CMD_GREEN_CHARGE_ONLY):
            assert cmd.power_consign >= -300.0 - 1e-6
            assert cmd.power_consign <= 0.0


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
    solver.solve(with_self_test=True)

    per_slot = solver._final_battery_commands
    assert per_slot is not None
    # no flipped slot ever discharges more than the floor (leak only)
    for cmd in per_slot:
        if cmd is not None and cmd.is_like(CMD_GREEN_CHARGE_ONLY):
            assert cmd.power_consign >= -300.0 - 1e-6
            assert cmd.power_consign <= 0.0
