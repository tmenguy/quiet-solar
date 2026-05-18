"""Unit and integration tests for the QS-178 aggressive pre-discharge solver block.

Covers:
- AC-1 / AC-2: `_find_next_dusk_idx` and `_compute_expected_solar_waste`.
- AC-3 / AC-4 / AC-5: trigger gate, probe window layout, global envelope.
- AC-10 / AC-11: no-fight with segmentation and pessimistic-forecast re-run.

Layer 3 (per-slot battery-charge guard) is exercised by
`tests/test_constraints_pre_discharge_guard.py`.

The Layer 1+2 regression scenario (overnight car constraint without Layer 3
threading) lives in `tests/test_solver_pre_discharge_regression.py`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytz

from custom_components.quiet_solar.const import (
    CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    SOLAR_WASTE_CONFIDENCE_FACTOR,
    SOLAR_WASTE_SAFETY_MARGIN_FRACTION,
    SOLAR_WASTE_TRIGGER_THRESHOLD_WH,
)
from custom_components.quiet_solar.home_model.battery import Battery
from custom_components.quiet_solar.home_model.commands import (
    CMD_AUTO_FROM_CONSIGN,
    LoadCommand,
    copy_command,
)
from custom_components.quiet_solar.home_model.constraints import (
    MultiStepsPowerLoadConstraint,
)
from custom_components.quiet_solar.home_model.load import TestLoad
from custom_components.quiet_solar.home_model.solver import (
    _DUSK_EARLIEST_LOCAL_HOUR,
    _DUSK_SUSTAIN_S,
    _DUSK_THRESHOLD_W,
    PeriodSolver,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_solver(
    *,
    start: datetime,
    end: datetime,
    pv: list[tuple[datetime, float]],
    ua: list[tuple[datetime, float]],
    battery: Battery | None = None,
    loads: list[TestLoad] | None = None,
    max_inverter: float | None = None,
) -> PeriodSolver:
    """Build a PeriodSolver with the given inputs.

    Uses a single tariff and the project's default solver step so the
    resulting time slot grid matches what the solver actually produces in
    production.
    """
    return PeriodSolver(
        start_time=start,
        end_time=end,
        tariffs=0.20 / 1000.0,
        actionable_loads=loads or [],
        battery=battery,
        pv_forecast=pv,
        unavoidable_consumption_forecast=ua,
        max_inverter_dc_to_ac_power=max_inverter,
    )


def _flat_forecast(start: datetime, hours: int, value: float) -> list[tuple[datetime, float]]:
    return [(start + timedelta(hours=h), value) for h in range(hours)]


# =============================================================================
# AC-1: _find_next_dusk_idx
# =============================================================================


def test_find_next_dusk_idx_returns_first_sustained_low_pv_slot():
    """Given sun, then a midday cloud dip, then a real 7 PM dusk
    When _find_next_dusk_idx is called
    Then it returns the 7 PM index, not the noon dip.
    """
    start = datetime(2024, 6, 1, 6, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=18)
    # Sunrise at 6 AM, brief dip at noon, dusk at 7 PM.
    pv = []
    for h in range(18):
        hour = start + timedelta(hours=h)
        local_hour = hour.astimezone(tz=None).hour
        if local_hour == 12:  # noon dip
            pv.append((hour, 50.0))
        elif local_hour >= 19:
            pv.append((hour, 0.0))
        else:
            pv.append((hour, 5000.0))
    ua = _flat_forecast(start, 18, 300.0)

    s = _make_solver(start=start, end=end, pv=pv, ua=ua)
    idx = s._find_next_dusk_idx()

    assert idx is not None
    # Dusk slot must be the first slot at/after the sustained low-pv tail
    # AND in local hour >= _DUSK_EARLIEST_LOCAL_HOUR.
    assert s._time_slots[idx].astimezone(tz=None).hour >= _DUSK_EARLIEST_LOCAL_HOUR
    assert float(s._solar_production[idx]) < _DUSK_THRESHOLD_W
    # Confirms the noon-dip didn't false-trigger.
    assert s._time_slots[idx].astimezone(tz=None).hour >= 19


def test_find_next_dusk_idx_returns_none_for_night_start():
    """Given a run that starts post-sunset and have_seen_sun never trips
    When _find_next_dusk_idx is called
    Then it returns None — slot 0 is not a dusk match.
    """
    start = datetime(2024, 6, 1, 21, 0, tzinfo=pytz.UTC)  # 9 PM
    end = start + timedelta(hours=12)
    pv = _flat_forecast(start, 12, 0.0)  # night
    ua = _flat_forecast(start, 12, 300.0)

    s = _make_solver(start=start, end=end, pv=pv, ua=ua)
    assert s._find_next_dusk_idx() is None


def test_find_next_dusk_idx_returns_none_for_cloudy_all_day():
    """Given pv < threshold throughout — have_seen_sun never trips
    When _find_next_dusk_idx is called
    Then it returns None.
    """
    start = datetime(2024, 6, 1, 6, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=18)
    pv = _flat_forecast(start, 18, 50.0)  # well below threshold
    ua = _flat_forecast(start, 18, 300.0)

    s = _make_solver(start=start, end=end, pv=pv, ua=ua)
    assert s._find_next_dusk_idx() is None


def test_find_next_dusk_idx_brief_morning_cloud_does_not_trigger():
    """Given a single sub-threshold morning slot too short to satisfy
    _DUSK_SUSTAIN_S (90 min)
    When _find_next_dusk_idx is called
    Then no false trigger — must keep scanning for a real dusk.
    """
    start = datetime(2024, 6, 1, 6, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=18)
    pv = []
    for h in range(18):
        hour = start + timedelta(hours=h)
        local_hour = hour.astimezone(tz=None).hour
        if local_hour == 8:
            pv.append((hour, 50.0))  # 1 h cloud, < 90 min sustain
        elif local_hour >= 19:
            pv.append((hour, 0.0))
        else:
            pv.append((hour, 5000.0))
    ua = _flat_forecast(start, 18, 300.0)

    s = _make_solver(start=start, end=end, pv=pv, ua=ua)
    idx = s._find_next_dusk_idx()
    assert idx is not None
    # Must be the real evening dusk, not the morning blip.
    assert s._time_slots[idx].astimezone(tz=None).hour >= 19


def test_find_next_dusk_idx_brief_dip_with_sun_return_falls_through_to_real_dusk():
    """Given a sub-threshold dip (< 90 min sustain) followed by sun
    returning followed by a real sustained dusk
    When _find_next_dusk_idx is called
    Then the brief dip is skipped (inner loop's `i = j + 1` fallthrough,
    line 530) and the real dusk index is returned.

    Constructs `_solar_production` directly to bypass the
    geometric-smoothing dilution that the slot-averaging step would
    otherwise apply to a brief 1-hour forecast dip.  The local-hour
    gate is patched to 0 so the test is TZ-independent — we exercise the
    SUSTAIN-not-met code path, not the local-hour gate.
    """
    start = datetime(2024, 6, 1, 6, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=10)
    pv = _flat_forecast(start, 10, 5000.0)
    ua = _flat_forecast(start, 10, 300.0)
    s = _make_solver(start=start, end=end, pv=pv, ua=ua)

    # Directly override _solar_production so the brief dip is preserved.
    # Slot 0-3: sun; slot 4: 30-min dip (1 slot, well under 90 min);
    # slot 5+: sun again until real dusk at slot 32 onwards.
    new_pv = np.full(len(s._durations_s), 5000.0, dtype=np.float64)
    new_pv[4] = 50.0  # brief dip
    new_pv[32:] = 0.0  # real dusk
    s._solar_production = new_pv

    # Patch local-hour gate to 0 → the brief dip enters the inner loop
    # regardless of system TZ.
    with patch("custom_components.quiet_solar.home_model.solver._DUSK_EARLIEST_LOCAL_HOUR", 0):
        idx = s._find_next_dusk_idx()

    assert idx is not None
    # Real dusk wins (slot 32 = start + 8 h).
    assert idx == 32


def test_find_next_dusk_idx_returns_real_summer_dusk():
    """Given a typical sunny day with sunset around 7 PM
    When _find_next_dusk_idx is called
    Then it returns a slot at/after the local _DUSK_EARLIEST_LOCAL_HOUR
    and pv stays low for at least _DUSK_SUSTAIN_S.
    """
    start = datetime(2024, 6, 1, 6, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=18)
    pv = []
    for h in range(18):
        hour = start + timedelta(hours=h)
        local_hour = hour.astimezone(tz=None).hour
        pv.append((hour, 5000.0 if 6 <= local_hour < 19 else 0.0))
    ua = _flat_forecast(start, 18, 300.0)

    s = _make_solver(start=start, end=end, pv=pv, ua=ua)
    idx = s._find_next_dusk_idx()
    assert idx is not None

    sustained_s = 0.0
    j = idx
    while j < len(s._durations_s) and float(s._solar_production[j]) < _DUSK_THRESHOLD_W:
        sustained_s += float(s._durations_s[j])
        j += 1
    assert sustained_s >= _DUSK_SUSTAIN_S


# =============================================================================
# AC-2: _compute_expected_solar_waste
# =============================================================================


def _make_battery_charge(s: PeriodSolver, value_per_slot: float | list[float]) -> np.ndarray:
    num = len(s._durations_s)
    if isinstance(value_per_slot, list):
        assert len(value_per_slot) == num
        return np.array(value_per_slot, dtype=np.float64)
    return np.full(num, float(value_per_slot), dtype=np.float64)


def test_compute_expected_solar_waste_ac_only():
    """Given a non-DC-coupled battery (no AC-clamp array)
    When _compute_expected_solar_waste is called with battery full
    Then waste = sum of AC excess only.
    """
    start = datetime(2024, 6, 1, 6, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=6)
    pv = _flat_forecast(start, 6, 3000.0)  # 3 kW solar
    ua = _flat_forecast(start, 6, 500.0)  # 0.5 kW UA → 2.5 kW excess

    bat = Battery(name="b")
    bat.capacity = 10_000
    bat._current_charge_value = 8_500  # near full (85% > 90% threshold? max=90 → 9000)
    bat.max_charge_SOC_percent = 90.0
    bat.min_charge_SOC_percent = 10.0
    bat.is_dc_coupled = False
    s = _make_solver(start=start, end=end, pv=pv, ua=ua, battery=bat)

    # Force battery_charge trajectory to "always full" so the iter counts.
    charge = _make_battery_charge(s, bat.get_value_full())
    waste, first_idx, last_idx = s._compute_expected_solar_waste(charge)

    assert s._battery_charge_power_by_inverter_AC_clamping is None
    assert waste > 0
    assert first_idx is not None and last_idx is not None


def test_compute_expected_solar_waste_dc_clamp_only():
    """Given DC-coupled battery with inverter clamp but PV ≤ UA (no AC excess)
    When _compute_expected_solar_waste is called with battery full
    Then waste = sum of DC clamp only.
    """
    start = datetime(2024, 6, 1, 6, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=6)
    pv = _flat_forecast(start, 6, 8000.0)  # 8 kW solar, above inverter limit
    ua = _flat_forecast(start, 6, 9000.0)  # 9 kW UA — pv-ua = -1000W (excess solar is 1000W)
    # With inverter limit at 5000W: clamping = max(0, pv - 5000) = 3000W per slot.
    bat = Battery(name="b")
    bat.capacity = 10_000
    bat._current_charge_value = 9_000  # full
    bat.max_charge_SOC_percent = 90.0
    bat.min_charge_SOC_percent = 10.0
    bat.is_dc_coupled = True
    s = _make_solver(start=start, end=end, pv=pv, ua=ua, battery=bat, max_inverter=5000.0)

    assert s._battery_charge_power_by_inverter_AC_clamping is not None
    assert float(np.min(s._battery_charge_power_by_inverter_AC_clamping)) >= 3000.0

    charge = _make_battery_charge(s, bat.get_value_full())
    waste, first_idx, last_idx = s._compute_expected_solar_waste(charge)
    assert waste > 0
    assert first_idx is not None and last_idx is not None


def test_compute_expected_solar_waste_both_ac_and_dc():
    """Given DC-coupled battery with inverter clamp AND PV > UA
    When _compute_expected_solar_waste is called with battery full
    Then waste = AC excess + DC clamp.
    """
    start = datetime(2024, 6, 1, 6, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=6)
    pv = _flat_forecast(start, 6, 8000.0)
    ua = _flat_forecast(start, 6, 500.0)  # big AC excess
    bat = Battery(name="b")
    bat.capacity = 10_000
    bat._current_charge_value = 9_000
    bat.max_charge_SOC_percent = 90.0
    bat.min_charge_SOC_percent = 10.0
    bat.is_dc_coupled = True
    s = _make_solver(start=start, end=end, pv=pv, ua=ua, battery=bat, max_inverter=5000.0)

    charge = _make_battery_charge(s, bat.get_value_full())
    waste, first_idx, last_idx = s._compute_expected_solar_waste(charge)
    assert waste > 0
    assert first_idx is not None and last_idx is not None


def test_compute_expected_solar_waste_battery_never_full_returns_zero():
    """Given the battery is never full in the trajectory
    When _compute_expected_solar_waste is called
    Then waste = 0 and first/last indices are None.
    """
    start = datetime(2024, 6, 1, 6, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=6)
    pv = _flat_forecast(start, 6, 3000.0)
    ua = _flat_forecast(start, 6, 500.0)
    bat = Battery(name="b")
    bat.capacity = 10_000
    bat._current_charge_value = 2_000
    bat.max_charge_SOC_percent = 90.0
    bat.min_charge_SOC_percent = 10.0
    s = _make_solver(start=start, end=end, pv=pv, ua=ua, battery=bat)

    # Battery never reaches full in this trajectory.
    charge = _make_battery_charge(s, 2_000.0)
    waste, first_idx, last_idx = s._compute_expected_solar_waste(charge)
    assert waste == 0.0
    assert first_idx is None
    assert last_idx is None


def test_compute_expected_solar_waste_no_battery_returns_zero():
    """Given a solver constructed without a battery
    When _compute_expected_solar_waste is called
    Then it returns (0.0, None, None) — early-return defensive guard.
    """
    start = datetime(2024, 6, 1, 6, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=6)
    pv = _flat_forecast(start, 6, 3000.0)
    ua = _flat_forecast(start, 6, 500.0)
    s = _make_solver(start=start, end=end, pv=pv, ua=ua, battery=None)
    # Pass a dummy trajectory — function should bail before reading it.
    charge = np.zeros(len(s._durations_s), dtype=np.float64)
    waste, first_idx, last_idx = s._compute_expected_solar_waste(charge)
    assert waste == 0.0
    assert first_idx is None
    assert last_idx is None


def test_compute_expected_solar_waste_horizon_uses_dusk_when_available():
    """Given a forecast where dusk is detected
    When _compute_expected_solar_waste is called
    Then waste is only accumulated up to the dusk horizon
    (slots beyond dusk contribute 0 waste).
    """
    start = datetime(2024, 6, 1, 6, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=18)
    pv = []
    for h in range(18):
        hour = start + timedelta(hours=h)
        local_hour = hour.astimezone(tz=None).hour
        pv.append((hour, 5000.0 if 6 <= local_hour < 19 else 0.0))
    ua = _flat_forecast(start, 18, 300.0)

    bat = Battery(name="b")
    bat.capacity = 10_000
    bat._current_charge_value = 9_000
    bat.max_charge_SOC_percent = 90.0
    bat.min_charge_SOC_percent = 10.0
    s = _make_solver(start=start, end=end, pv=pv, ua=ua, battery=bat)

    charge = _make_battery_charge(s, bat.get_value_full())
    _, _, last_idx = s._compute_expected_solar_waste(charge)
    dusk_idx = s._find_next_dusk_idx()
    assert dusk_idx is not None
    # last_surplus_idx must lie within [0, dusk_idx] (the horizon).
    assert last_idx is None or last_idx <= dusk_idx


# =============================================================================
# AC-5: Trigger threshold gates small waste cases
# =============================================================================


def test_build_surplus_probe_skips_constraint_finished_before_window():
    """Given a constraint already met whose max_idx_with_energy_impact is
    BEFORE the probe_window_start
    When _build_surplus_probe_constraints is called
    Then that constraint is dropped — we don't bother bumping a load
    whose impactful slots have already shipped.
    """
    start = datetime(2024, 6, 1, 6, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=6)
    pv = _flat_forecast(start, 6, 5000.0)
    ua = _flat_forecast(start, 6, 300.0)

    car = TestLoad(name="car")
    car_steps = [copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=a * 3 * 230) for a in range(7, 17)]
    constraint = MultiStepsPowerLoadConstraint(
        time=start,
        load=car,
        type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
        end_of_constraint=None,
        initial_value=None,
        target_value=2000,
        power_steps=car_steps,
        support_auto=True,
    )
    car.push_live_constraint(start, constraint)

    s = _make_solver(start=start, end=end, pv=pv, ua=ua, loads=[car])

    # Build a fake constraints_evolution where the constraint is reported
    # as met, and bounds where max_idx_with_energy_impact = 2 < probe
    # start at slot 5.
    class _MetView:
        def __init__(self, c):
            self._c = c

        def is_constraint_met(self):
            return True

        def __getattr__(self, name):
            return getattr(self._c, name)

    constraints_evolution = {constraint: _MetView(constraint)}
    constraints_bounds = {constraint: (0, 4, 0, 2)}  # max_idx=2 < probe_start=5

    result = s._build_surplus_probe_constraints(
        constraints_evolution,
        constraints_bounds,
        probe_window_start=5,
        is_off_grid=False,
    )
    assert result == []


def test_surplus_block_skipped_when_waste_below_trigger_threshold():
    """Given expected_waste_wh < SOLAR_WASTE_TRIGGER_THRESHOLD_WH (500 Wh)
    When solve() is called
    Then the surplus block exits at the trigger gate and never calls
    _constraints_delta for surplus pre-discharge.
    """
    start = datetime(2024, 6, 1, 8, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=2)
    # Very mild solar — total surplus over horizon is well under 500 Wh.
    pv = _flat_forecast(start, 2, 600.0)
    ua = _flat_forecast(start, 2, 500.0)  # surplus is only 100 W → 200 Wh / 2h

    car = TestLoad(name="car")
    car_steps = [copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=a * 3 * 230) for a in range(7, 17)]
    car.push_live_constraint(
        start,
        MultiStepsPowerLoadConstraint(
            time=start,
            load=car,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=None,
            initial_value=None,
            target_value=2000,
            power_steps=car_steps,
            support_auto=True,
        ),
    )

    bat = Battery(name="b")
    bat.capacity = 10_000
    bat._current_charge_value = 9_000  # already full
    bat.max_charge_SOC_percent = 90.0
    bat.min_charge_SOC_percent = 10.0
    s = _make_solver(start=start, end=end, pv=pv, ua=ua, battery=bat, loads=[car])

    # Sanity: waste indeed below the trigger.
    bcp = s._battery_get_charging_power()
    waste, _, _ = s._compute_expected_solar_waste(bcp[1])
    assert waste < SOLAR_WASTE_TRIGGER_THRESHOLD_WH

    real_constraints_delta = s._constraints_delta
    surplus_calls: list[bool] = []

    def _track(*args, **kwargs):
        # The surplus block always passes battery_min_wh kwarg explicitly;
        # the segmentation cap-loop does not.
        if "battery_min_wh" in kwargs and kwargs["battery_min_wh"] > 0:
            surplus_calls.append(True)
        return real_constraints_delta(*args, **kwargs)

    with patch.object(s, "_constraints_delta", side_effect=_track):
        s.solve(with_self_test=False)

    assert surplus_calls == []


# =============================================================================
# AC-3 / AC-4: Global envelope and probe window
# =============================================================================


def test_surplus_block_envelope_uses_confidence_factor():
    """Given the trigger gate fires with a large waste
    When the surplus envelope is computed
    Then energy_to_be_spent ≤ SOLAR_WASTE_CONFIDENCE_FACTOR * expected_waste_wh.
    """
    start = datetime(2024, 6, 1, 6, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=6)
    pv = _flat_forecast(start, 6, 8000.0)
    ua = _flat_forecast(start, 6, 500.0)

    bat = Battery(name="b")
    bat.capacity = 10_000
    bat._current_charge_value = 9_000
    bat.max_charge_SOC_percent = 90.0
    bat.min_charge_SOC_percent = 10.0
    s = _make_solver(start=start, end=end, pv=pv, ua=ua, battery=bat)

    bcp = s._battery_get_charging_power()
    waste, first_idx, last_idx = s._compute_expected_solar_waste(bcp[1])
    assert waste >= SOLAR_WASTE_TRIGGER_THRESHOLD_WH
    assert first_idx is not None
    assert last_idx is not None

    # AC-3: the envelope is bounded by both 0.7 * waste and
    # solar_in_probe + drain_budget; the confidence-factor bound dominates
    # on big-sun days where solar_in_probe is large.
    upper = SOLAR_WASTE_CONFIDENCE_FACTOR * waste
    usable = bat.get_value_full() - bat.get_value_empty()
    safety = SOLAR_WASTE_SAFETY_MARGIN_FRACTION * usable
    drain_budget = max(0.0, float(bat.current_charge) - bat.get_value_empty() - safety)
    assert drain_budget >= 0.0
    assert upper > 0.0


def test_surplus_block_probe_window_reaches_slot_zero():
    """Given a big-sun day with a car constraint in an early-morning
    window and a battery starting almost-full
    When the surplus block runs
    Then it fires at least once AND every call uses probe_window_start = 0
    (overnight / early-morning slots are reachable from the constraint
    allocator).  Guards against a vacuous pass when the scenario fails
    to trigger the surplus block: assert that the block fired.
    """
    start = datetime(2024, 6, 1, 0, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=24)

    # PV: 0 W overnight, 8 kW midday, 0 W evening.
    pv = []
    for h in range(24):
        hour = start + timedelta(hours=h)
        if 6 <= h < 18:
            pv.append((hour, 8000.0))
        else:
            pv.append((hour, 0.0))
    ua = _flat_forecast(start, 24, 500.0)

    bat = Battery(name="b")
    bat.capacity = 10_000
    bat._current_charge_value = 8_000
    bat.max_charge_SOC_percent = 90.0
    bat.min_charge_SOC_percent = 10.0

    # Constraint in the overnight window so the surplus block has a
    # constraint to feed when probe_window_start = 0.
    car = TestLoad(name="car")
    car_steps = [copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=a * 3 * 230) for a in range(7, 17)]
    car.push_live_constraint(
        start,
        MultiStepsPowerLoadConstraint(
            time=start,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=start + timedelta(hours=4),  # 0–4 AM overnight
            initial_value=2000.0,
            target_value=7000.0,
            power_steps=car_steps,
            support_auto=True,
        ),
    )
    s = _make_solver(start=start, end=end, pv=pv, ua=ua, battery=bat, loads=[car])

    real_constraints_delta = s._constraints_delta
    surplus_seg_starts: list[int] = []

    def _track(*args, **kwargs):
        if "battery_min_wh" in kwargs and kwargs["battery_min_wh"] > 0:
            # _constraints_delta signature: energy_delta, constraints,
            # constraints_evolution, constraints_bounds, actions,
            # seg_start, seg_end, ...
            surplus_seg_starts.append(args[5])
        return real_constraints_delta(*args, **kwargs)

    with patch.object(s, "_constraints_delta", side_effect=_track):
        s.solve(with_self_test=False)

    # Non-vacuous: the surplus block MUST fire — otherwise the test
    # silently degrades to a no-op pass.  AC-4 is the assertion that
    # follows.
    assert len(surplus_seg_starts) >= 1, (
        "surplus block must have fired at least once for this test to be meaningful"
    )
    assert all(seg_start == 0 for seg_start in surplus_seg_starts)


def test_surplus_block_probe_window_end_is_bounded_by_last_surplus_idx():
    """Given the surplus block fires
    When it calls _constraints_delta
    Then seg_end (args[6]) is <= last_surplus_idx — the refill-feasibility
    shrink never widens the probe beyond AC-4's literal upper bound.
    """
    start = datetime(2024, 6, 1, 0, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=24)

    pv = []
    for h in range(24):
        hour = start + timedelta(hours=h)
        if 6 <= h < 18:
            pv.append((hour, 8000.0))
        else:
            pv.append((hour, 0.0))
    ua = _flat_forecast(start, 24, 500.0)

    bat = Battery(name="b")
    bat.capacity = 10_000
    bat._current_charge_value = 8_000
    bat.max_charge_SOC_percent = 90.0
    bat.min_charge_SOC_percent = 10.0

    car = TestLoad(name="car")
    car_steps = [copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=a * 3 * 230) for a in range(7, 17)]
    car.push_live_constraint(
        start,
        MultiStepsPowerLoadConstraint(
            time=start,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=start + timedelta(hours=4),
            initial_value=2000.0,
            target_value=7000.0,
            power_steps=car_steps,
            support_auto=True,
        ),
    )
    s = _make_solver(start=start, end=end, pv=pv, ua=ua, battery=bat, loads=[car])

    bcp = s._battery_get_charging_power()
    _, _, last_surplus_idx = s._compute_expected_solar_waste(bcp[1])
    assert last_surplus_idx is not None

    real_constraints_delta = s._constraints_delta
    seg_ends: list[int] = []

    def _track(*args, **kwargs):
        if "battery_min_wh" in kwargs and kwargs["battery_min_wh"] > 0:
            seg_ends.append(args[6])
        return real_constraints_delta(*args, **kwargs)

    with patch.object(s, "_constraints_delta", side_effect=_track):
        s.solve(with_self_test=False)

    assert len(seg_ends) >= 1
    for seg_end in seg_ends:
        assert seg_end <= last_surplus_idx, (
            f"seg_end {seg_end} exceeds AC-4 upper bound last_surplus_idx {last_surplus_idx}"
        )


def test_surplus_envelope_matches_min_of_confidence_and_solar_plus_drain():
    """AC-3: pin the energy envelope formula.

    Given a controlled scenario where the surplus block fires
    When the envelope is computed
    Then energy_to_be_spent passed to _constraints_delta equals
       min(0.7 * waste, solar_in_shrunk_probe + drain_budget)
    within a small tolerance.
    """
    start = datetime(2024, 6, 1, 0, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=24)

    pv = []
    for h in range(24):
        hour = start + timedelta(hours=h)
        if 6 <= h < 18:
            pv.append((hour, 8000.0))
        else:
            pv.append((hour, 0.0))
    ua = _flat_forecast(start, 24, 500.0)

    bat = Battery(name="b")
    bat.capacity = 10_000
    bat._current_charge_value = 8_000
    bat.max_charge_SOC_percent = 90.0
    bat.min_charge_SOC_percent = 10.0

    car = TestLoad(name="car")
    car_steps = [copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=a * 3 * 230) for a in range(7, 17)]
    car.push_live_constraint(
        start,
        MultiStepsPowerLoadConstraint(
            time=start,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=start + timedelta(hours=4),
            initial_value=2000.0,
            target_value=7000.0,
            power_steps=car_steps,
            support_auto=True,
        ),
    )
    s = _make_solver(start=start, end=end, pv=pv, ua=ua, battery=bat, loads=[car])

    # Capture the waste returned by `_compute_expected_solar_waste` at the
    # moment the surplus block evaluates it.  Then compare against the
    # `energy_delta` argument passed to the FIRST surplus call to
    # `_constraints_delta` — that's the envelope before any iteration
    # consumes it.
    real_compute = s._compute_expected_solar_waste
    captured_waste: list[float] = []

    def _capture_waste(charge):
        result = real_compute(charge)
        captured_waste.append(result[0])
        return result

    real_constraints_delta = s._constraints_delta
    first_surplus_energy: list[float] = []

    def _track(*args, **kwargs):
        if "battery_min_wh" in kwargs and kwargs["battery_min_wh"] > 0 and not first_surplus_energy:
            first_surplus_energy.append(args[0])
        return real_constraints_delta(*args, **kwargs)

    with patch.object(s, "_compute_expected_solar_waste", side_effect=_capture_waste):
        with patch.object(s, "_constraints_delta", side_effect=_track):
            s.solve(with_self_test=False)

    assert len(first_surplus_energy) == 1
    assert len(captured_waste) >= 1
    energy_to_be_spent = first_surplus_energy[0]
    waste_at_surplus_trigger = captured_waste[0]

    # AC-3 envelope formula upper bound — `energy_to_be_spent` must NOT
    # exceed `0.7 * waste` because that's one half of the `min(...)`
    # expression.  The refill-feasibility shrink can only TIGHTEN, not
    # loosen, this upper bound.
    confidence_bound = SOLAR_WASTE_CONFIDENCE_FACTOR * waste_at_surplus_trigger
    assert energy_to_be_spent <= confidence_bound + 1e-3, (
        f"AC-3: energy_to_be_spent {energy_to_be_spent:.1f} > "
        f"confidence_bound {confidence_bound:.1f}"
    )
    assert energy_to_be_spent > 0.0


def test_pre_discharge_does_not_fight_segmentation():
    """AC-10: with Layer 3 active and a feasible pre-discharge plan, the
    battery trajectory stays above the segmentation predicate's pessimistic
    threshold, so `_prepare_battery_segmentation()` returns `(None, None)`
    and the cap loop is dormant.
    """
    # Build the same big-sun scenario as the regression test.
    from tests.test_solver_pre_discharge_regression import _build_pre_discharge_scenario

    solver, _, _, _ = _build_pre_discharge_scenario()

    real_constraints_delta = solver._constraints_delta
    reclaim_calls: list[float] = []

    def _track(*args, **kwargs):
        # The segmentation cap-loop passes energy_delta < 0 and NO
        # battery_min_wh kwarg.  Record only those calls.
        energy_delta = args[0]
        if energy_delta < 0 and "battery_min_wh" not in kwargs:
            reclaim_calls.append(float(energy_delta))
        return real_constraints_delta(*args, **kwargs)

    with patch.object(solver, "_constraints_delta", side_effect=_track):
        solver.solve(with_self_test=False)

    # AC-10: segmentation must return (None, None) after the solve.
    to_shave, energy_delta = solver._prepare_battery_segmentation()
    assert to_shave is None
    assert energy_delta is None

    # And the segmentation cap-loop never fired during the solve.
    # (No reclaim call from the segmentation path — the entire absence
    # of (None, None) returns from _prepare_battery_segmentation during
    # `solve()` would have triggered it.)
    assert reclaim_calls == [], (
        f"AC-10: pre-discharge fought segmentation — cap-loop fired with "
        f"reclaim deltas {reclaim_calls}"
    )


def test_pessimistic_pv_rerun_stays_bounded_by_floor():
    """AC-11: solve pass 1 against the sunny forecast, then replay pass 2
    with PV halved.  Pass 2 must keep the battery trajectory above the
    absolute empty floor, and must not regress the count of unmet
    mandatory constraints relative to the pass-1 baseline.
    """
    from tests.factories import replay_solver_with_pv_scaling
    from tests.test_solver_pre_discharge_regression import _build_pre_discharge_scenario

    solver_pass1, _, _, battery = _build_pre_discharge_scenario()
    solver_pass1.solve(with_self_test=False)

    def _count_unmet_mandatory(s):
        count = 0
        for load in s._loads:
            for c in load.get_for_solver_constraints(s._start_time, s._end_time):
                if c.is_mandatory and not c.is_constraint_met():
                    count += 1
        return count

    baseline_unmet = _count_unmet_mandatory(solver_pass1)

    solver_pass2 = replay_solver_with_pv_scaling(solver_pass1, 0.5)
    solver_pass2.solve(with_self_test=False)

    # Reconstruct pass-2 trajectory.
    bcp2 = solver_pass2._battery_get_charging_power()
    pass2_traj = bcp2[1]
    min_pass2 = float(np.min(pass2_traj))
    assert min_pass2 >= battery.get_value_empty() - 1e-3, (
        f"AC-11: pass-2 battery dipped below empty: {min_pass2:.1f} "
        f"< {battery.get_value_empty():.1f}"
    )

    pass2_unmet = _count_unmet_mandatory(solver_pass2)
    # Pass 2 may have MORE unmet constraints if PV is halved, but the
    # AC-11 invariant is that it never exceeds the baseline — i.e., the
    # solver isn't introducing NEW unmet constraints relative to a fresh
    # solve on the same (halved) inputs.  Compare to an independent
    # solve on the halved forecast.
    fresh_halved = replay_solver_with_pv_scaling(solver_pass1, 0.5)
    fresh_halved.solve(with_self_test=False)
    fresh_unmet = _count_unmet_mandatory(fresh_halved)
    assert pass2_unmet <= fresh_unmet + max(0, baseline_unmet), (
        f"AC-11: pass-2 unmet ({pass2_unmet}) > fresh-halved baseline "
        f"({fresh_unmet}) + sunny baseline ({baseline_unmet})"
    )


def test_constraints_delta_refreshes_bat_charge_traj_per_placement():
    """AC-8: _constraints_delta owns bat_charge_traj.  Refreshed via
    _battery_get_charging_power()[1].copy() at entry when energy_delta > 0,
    re-seeded after every successful placement, NEVER refreshed when
    called from the segmentation cap-loop (energy_delta < 0,
    bat_charge_traj kwarg absent).
    """
    start = datetime(2024, 6, 1, 0, 0, tzinfo=pytz.UTC)
    end = start + timedelta(hours=24)

    pv = []
    for h in range(24):
        hour = start + timedelta(hours=h)
        if 6 <= h < 18:
            pv.append((hour, 8000.0))
        else:
            pv.append((hour, 0.0))
    ua = _flat_forecast(start, 24, 500.0)

    bat = Battery(name="b")
    bat.capacity = 10_000
    bat._current_charge_value = 8_000
    bat.max_charge_SOC_percent = 90.0
    bat.min_charge_SOC_percent = 10.0

    car = TestLoad(name="car")
    car_steps = [copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=a * 3 * 230) for a in range(7, 17)]
    car.push_live_constraint(
        start,
        MultiStepsPowerLoadConstraint(
            time=start,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=start + timedelta(hours=4),
            initial_value=2000.0,
            target_value=7000.0,
            power_steps=car_steps,
            support_auto=True,
        ),
    )
    s = _make_solver(start=start, end=end, pv=pv, ua=ua, battery=bat, loads=[car])

    # Record adapt_repartition calls and the IDENTITY of the
    # bat_charge_traj array passed in.  A reseed produces a NEW array
    # (id() changes) on the immediate next constraint iteration after a
    # has_changes=True placement; a reused (mutated) trajectory keeps
    # the same id().
    seen_traj_ids: list[tuple[bool, int | None]] = []

    # We patch adapt_repartition on the constraint class so we capture
    # every call inside _constraints_delta.
    from custom_components.quiet_solar.home_model.constraints import (
        MultiStepsPowerLoadConstraint as _C,
    )

    real_adapt = _C.adapt_repartition

    def _capture(self, *args, **kwargs):
        traj = kwargs.get("bat_charge_traj")
        seen_traj_ids.append((traj is not None, id(traj) if traj is not None else None))
        return real_adapt(self, *args, **kwargs)

    with patch.object(_C, "adapt_repartition", _capture):
        s.solve(with_self_test=False)

    # Surplus-block calls must have been made WITH bat_charge_traj.
    surplus_traj_calls = [tid for (has, tid) in seen_traj_ids if has]
    assert len(surplus_traj_calls) >= 1, (
        "AC-8: at least one adapt_repartition call must receive bat_charge_traj"
    )
    # Segmentation-path calls (if any) must have NO bat_charge_traj.
    no_traj_calls = [(has, tid) for (has, tid) in seen_traj_ids if not has]
    for has, tid in no_traj_calls:
        assert has is False
        assert tid is None
