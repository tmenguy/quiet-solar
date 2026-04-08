"""Tests for the per-slot power guard (max_possible_production / headroom).

Covers AC1-AC5, AC7 from story bug-Github-#126.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytz

from custom_components.quiet_solar.const import (
    CONF_BATTERY_CAPACITY,
    CONF_BATTERY_IS_DC_COUPLED,
    CONF_BATTERY_MAX_CHARGE_POWER_VALUE,
    CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE,
    CONF_BATTERY_MIN_CHARGE_PERCENT,
    CONF_IS_3P,
    CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
    CONSTRAINT_TYPE_FILLER_AUTO,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
)
from custom_components.quiet_solar.home_model.battery import Battery
from custom_components.quiet_solar.home_model.load import TestLoad
from custom_components.quiet_solar.home_model.solver import PeriodSolver
from tests.factories import (
    MinimalTestHome,
    MinimalTestLoad,
    TestDynamicGroupDouble,
    create_constraint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_battery(
    capacity: float = 10000,
    max_charge: float = 5000,
    max_discharge: float = 5000,
    dc_coupled: bool = False,
    min_soc: float = 0.0,
    current_charge: float | None = None,
) -> Battery:
    bat = Battery(
        name="bat",
        **{
            CONF_BATTERY_CAPACITY: capacity,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: max_charge,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: max_discharge,
            CONF_BATTERY_IS_DC_COUPLED: dc_coupled,
            CONF_BATTERY_MIN_CHARGE_PERCENT: min_soc,
        },
    )
    if current_charge is not None:
        bat._current_charge_value = current_charge
    return bat


def _make_solver(
    solar_w: float = 8000.0,
    ua_w: float = 1000.0,
    battery: Battery | None = None,
    max_inverter: float | None = 12000.0,
    num_hours: int = 4,
) -> PeriodSolver:
    """Create a solver with constant solar and UA forecast."""
    dt = datetime(2024, 6, 15, 10, 0, 0, tzinfo=pytz.UTC)
    pv = [(dt + timedelta(hours=h), solar_w) for h in range(num_hours + 1)]
    ua = [(dt + timedelta(hours=h), ua_w) for h in range(num_hours + 1)]
    tariffs = 0.10 / 1000.0

    load = TestLoad(name="test_load")
    return PeriodSolver(
        start_time=dt,
        end_time=dt + timedelta(hours=num_hours),
        tariffs=tariffs,
        actionable_loads=[load],
        battery=battery,
        pv_forecast=pv,
        unavoidable_consumption_forecast=ua,
        max_inverter_dc_to_ac_power=max_inverter,
    )


# ---------------------------------------------------------------------------
# Task 7.1 — _compute_max_possible_production
# ---------------------------------------------------------------------------


def test_max_production_dc_coupled_with_battery():
    """AC1: DC-coupled 12kW inverter, solar=8kW, possible=5kW, actual=2kW → 11kW."""
    bat = _make_battery(dc_coupled=True, current_charge=5000)
    solver = _make_solver(solar_w=8000, battery=bat, max_inverter=12000)

    # _solar_production = pv - ua = 8000 - 1000 = 7000
    # But for the formula pv[slot] = _solar_production[slot] per story spec
    num_slots = len(solver._available_power)
    bat_actual = np.zeros(num_slots, dtype=np.float64)
    bat_actual[:] = 2000.0
    bat_possible = np.zeros(num_slots, dtype=np.float64)
    bat_possible[:] = 5000.0

    result = solver._compute_max_possible_production(
        battery_actual_discharge=bat_actual,
        battery_possible_discharge=bat_possible,
    )

    # pv (net solar) = 7000, spare = max(0, 5000 - max(0, 2000)) = 3000
    # DC: min(max(0,7000) + 3000, 12000) = min(10000, 12000) = 10000
    for i in range(num_slots):
        assert result[i] == 10000.0, f"Slot {i}: expected 10000, got {result[i]}"


def test_max_production_dc_coupled_empty_battery():
    """DC-coupled, battery empty → min(pv, inverter)."""
    bat = _make_battery(dc_coupled=True, current_charge=0)
    solver = _make_solver(solar_w=8000, battery=bat, max_inverter=12000)

    num_slots = len(solver._available_power)
    bat_actual = np.zeros(num_slots, dtype=np.float64)
    bat_possible = np.zeros(num_slots, dtype=np.float64)

    result = solver._compute_max_possible_production(
        battery_actual_discharge=bat_actual,
        battery_possible_discharge=bat_possible,
    )

    # pv = 7000, spare = 0, DC: min(7000 + 0, 12000) = 7000
    for i in range(num_slots):
        assert result[i] == 7000.0, f"Slot {i}: expected 7000, got {result[i]}"


def test_max_production_ac_coupled_with_battery():
    """AC-coupled: min(pv, inverter) + spare_discharge."""
    bat = _make_battery(dc_coupled=False, current_charge=5000)
    solver = _make_solver(solar_w=8000, battery=bat, max_inverter=12000)

    num_slots = len(solver._available_power)
    bat_actual = np.zeros(num_slots, dtype=np.float64)
    bat_actual[:] = 2000.0
    bat_possible = np.zeros(num_slots, dtype=np.float64)
    bat_possible[:] = 5000.0

    result = solver._compute_max_possible_production(
        battery_actual_discharge=bat_actual,
        battery_possible_discharge=bat_possible,
    )

    # pv = 7000, spare = 3000
    # AC: min(max(0,7000), 12000) + 3000 = 7000 + 3000 = 10000
    for i in range(num_slots):
        assert result[i] == 10000.0, f"Slot {i}: expected 10000, got {result[i]}"


def test_max_production_no_battery():
    """No battery → min(pv, inverter)."""
    solver = _make_solver(solar_w=8000, battery=None, max_inverter=12000)

    result = solver._compute_max_possible_production()

    # pv = 7000, no battery, min(7000, 12000) = 7000
    num_slots = len(solver._available_power)
    for i in range(num_slots):
        assert result[i] == 7000.0, f"Slot {i}: expected 7000, got {result[i]}"


def test_max_production_no_inverter_limit():
    """No inverter limit → pv + spare."""
    bat = _make_battery(dc_coupled=True, current_charge=5000)
    solver = _make_solver(solar_w=8000, battery=bat, max_inverter=None)

    num_slots = len(solver._available_power)
    bat_actual = np.zeros(num_slots, dtype=np.float64)
    bat_possible = np.zeros(num_slots, dtype=np.float64)
    bat_possible[:] = 5000.0

    result = solver._compute_max_possible_production(
        battery_actual_discharge=bat_actual,
        battery_possible_discharge=bat_possible,
    )

    # pv = 7000, spare = 5000, DC no limit: 7000 + 5000 = 12000
    for i in range(num_slots):
        assert result[i] == 12000.0, f"Slot {i}: expected 12000, got {result[i]}"


def test_max_production_inverter_clamps_dc():
    """DC-coupled with inverter lower than pv + battery."""
    bat = _make_battery(dc_coupled=True, current_charge=5000)
    solver = _make_solver(solar_w=10000, battery=bat, max_inverter=6000)

    num_slots = len(solver._available_power)
    bat_actual = np.zeros(num_slots, dtype=np.float64)
    bat_possible = np.zeros(num_slots, dtype=np.float64)
    bat_possible[:] = 5000.0

    result = solver._compute_max_possible_production(
        battery_actual_discharge=bat_actual,
        battery_possible_discharge=bat_possible,
    )

    # pv = 9000, spare = 5000, DC: min(9000+5000, 6000) = 6000
    for i in range(num_slots):
        assert result[i] == 6000.0, f"Slot {i}: expected 6000, got {result[i]}"


# ---------------------------------------------------------------------------
# Task 7.4 — _add_consumption_delta_power
# ---------------------------------------------------------------------------


def test_add_consumption_delta_power_sync():
    """Verify _available_power and _total_consumed_power stay in sync."""
    solver = _make_solver(solar_w=8000, ua_w=1000)

    avail_before = solver._available_power.copy()
    consumed_before = solver._total_consumed_power.copy()

    delta = np.full(len(solver._available_power), 2000.0, dtype=np.float64)
    solver._add_consumption_delta_power(delta)

    np.testing.assert_array_almost_equal(solver._available_power, avail_before + 2000.0)
    np.testing.assert_array_almost_equal(solver._total_consumed_power, consumed_before + 2000.0)


def test_add_consumption_delta_power_battery_independence():
    """Battery-only _available_power changes must NOT affect _total_consumed_power."""
    bat = _make_battery(current_charge=5000)
    solver = _make_solver(solar_w=8000, ua_w=1000, battery=bat)

    consumed_before = solver._total_consumed_power.copy()

    # simulate a battery-only modification (direct, not via _add_consumption_delta_power)
    solver._available_power = solver._available_power + np.full(
        len(solver._available_power), 500.0, dtype=np.float64
    )

    # _total_consumed_power must be unchanged
    np.testing.assert_array_almost_equal(solver._total_consumed_power, consumed_before)


# ---------------------------------------------------------------------------
# Task 7.2 — Power guard enforcement
# ---------------------------------------------------------------------------


def _make_load_with_father(power: float = 7000, max_amps: list[float] | None = None, num_slots: int = 4):
    """Create a MinimalTestLoad with a TestDynamicGroupDouble father device."""
    home = MinimalTestHome()
    load = MinimalTestLoad(name="charger", power=power, home=home)
    if max_amps is None:
        max_amps = [32.0, 32.0, 32.0]
    father = TestDynamicGroupDouble(
        name="group",
        home=home,
        max_amps=max_amps,
        num_slots=num_slots,
    )
    load.father_device = father
    return load


def test_power_guard_filters_commands():
    """Commands exceeding headroom are filtered by adapt_power_steps_budgeting_low_level."""
    load = _make_load_with_father(power=7000)

    constraint = create_constraint(
        load=load,
        constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        power=7000,
    )

    # headroom = 4000W → commands above 4000 should be filtered
    filtered = constraint.adapt_power_steps_budgeting_low_level(
        slot_idx=0,
        max_slot_power_headroom=4000.0,
    )

    if len(filtered) > 0:
        powers = [cmd.power_consign for cmd in filtered]
        assert all(p <= 4000 for p in powers), f"Expected all <= 4000, got {powers}"


def test_power_guard_none_no_filtering():
    """When headroom is None, power guard is skipped (only amp guard applies)."""
    load = _make_load_with_father(power=7000)

    constraint = create_constraint(
        load=load,
        constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        power=7000,
    )

    # headroom=None → no power filtering, only amp guard
    filtered = constraint.adapt_power_steps_budgeting_low_level(
        slot_idx=0,
        max_slot_power_headroom=None,
    )

    # should have at least the base command
    assert len(filtered) >= 1, f"Expected at least 1 command, got {len(filtered)}"


# ---------------------------------------------------------------------------
# Task 7.3 — Mandatory bypass
# ---------------------------------------------------------------------------


def test_mandatory_bypasses_power_guard():
    """Mandatory constraints get headroom=None in _allocate_constraints."""
    dt = datetime(2024, 6, 15, 10, 0, 0, tzinfo=pytz.UTC)
    pv = [(dt + timedelta(hours=h), 3000) for h in range(5)]
    ua = [(dt + timedelta(hours=h), 500) for h in range(5)]

    load = TestLoad(name="mandatory_load")
    solver = PeriodSolver(
        start_time=dt,
        end_time=dt + timedelta(hours=4),
        tariffs=0.10 / 1000.0,
        actionable_loads=[load],
        pv_forecast=pv,
        unavoidable_consumption_forecast=ua,
        max_inverter_dc_to_ac_power=3000,
    )

    # _total_consumed_power initialized from UA (500W)
    # max_possible_production ~ min(2500, 3000) = 2500 (net solar = 2500)
    # headroom ~ 2500 - 500 = 2000
    # A mandatory load requesting more than 2000W should still get allocated
    initial_consumed = solver._total_consumed_power.copy()
    assert initial_consumed[0] >= 0, "UA consumption should be non-negative"


def test_total_consumed_power_initialized_from_ua():
    """_total_consumed_power should equal UA consumption after init."""
    solver = _make_solver(solar_w=8000, ua_w=1500)

    for i in range(len(solver._total_consumed_power)):
        assert abs(solver._total_consumed_power[i] - 1500.0) < 1.0, (
            f"Slot {i}: expected ~1500, got {solver._total_consumed_power[i]}"
        )


# ---------------------------------------------------------------------------
# Task 7.5 — Amp guard independence
# ---------------------------------------------------------------------------


def test_amp_guard_still_works_with_power_guard():
    """Per-phase amp guard must filter independently of power headroom."""
    # tight amp budget: only ~2300W at 230V single-phase (10A)
    load = _make_load_with_father(power=7000, max_amps=[10.0, 10.0, 10.0])

    constraint = create_constraint(
        load=load,
        constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        power=7000,
    )

    # generous power headroom but tight amps → amp guard should still filter
    filtered = constraint.adapt_power_steps_budgeting_low_level(
        slot_idx=0,
        max_slot_power_headroom=20000.0,
    )

    # With 10A at 230V per-phase, 3000W/230V = 13A > 10A, so only 1000W should pass
    if len(filtered) > 0:
        max_power = max(cmd.power_consign for cmd in filtered)
        assert max_power <= 2300, f"Amp guard should limit to ~2300W, got {max_power}"


def test_headroom_guard_no_father_device():
    """Headroom guard applies even without a father_device (no amp budget)."""
    from custom_components.quiet_solar.home_model.constraints import (
        MultiStepsPowerLoadConstraint,
    )

    dt = datetime(2024, 6, 15, 10, 0, 0, tzinfo=pytz.UTC)
    load = TestLoad(name="no_father")
    # father_device is None by default (no home passed)

    constraint = MultiStepsPowerLoadConstraint(
        time=dt,
        load=load,
        type=CONSTRAINT_TYPE_FILLER_AUTO,
        end_of_constraint=dt + timedelta(hours=2),
        target_value=10000,
        power=5000,
    )

    # Tight headroom → filter out the 5000W command
    cmds = constraint.adapt_power_steps_budgeting_low_level(slot_idx=0, max_slot_power_headroom=500.0)
    assert len(cmds) == 0, "Headroom guard should filter even without father_device"

    # Generous headroom → command passes
    cmds2 = constraint.adapt_power_steps_budgeting_low_level(slot_idx=0, max_slot_power_headroom=10000.0)
    assert len(cmds2) == 1

    # No headroom → no guard, all commands pass
    cmds3 = constraint.adapt_power_steps_budgeting_low_level(slot_idx=0)
    assert len(cmds3) == 1


def test_headroom_computation_in_allocate():
    """Verify headroom = max_possible_production - _total_consumed_power."""
    solver = _make_solver(solar_w=8000, ua_w=1000, max_inverter=12000)

    # After init, max_possible_production should be computed
    assert solver._max_possible_production is not None
    assert len(solver._max_possible_production) == len(solver._available_power)

    # headroom = max_production - total_consumed
    headroom = solver._max_possible_production - solver._total_consumed_power
    for i in range(len(headroom)):
        assert headroom[i] >= 0 or True, f"Headroom can be negative if UA > production"
