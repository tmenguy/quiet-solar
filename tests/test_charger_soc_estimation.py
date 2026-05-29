"""Charger-side tests for the estimated-SOC accumulator (Story QS-243).

Covers the float accumulator with its dedicated integration cursor in
`constraint_update_value_callback_soc`: no loss / no double-count across
cycles, the pure-delta case, the absolute-estimate case, and the real-power
check being skipped while estimating.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz
from homeassistant.const import CONF_NAME

from custom_components.quiet_solar.const import (
    CONF_CAR_BATTERY_CAPACITY,
    CONF_CAR_CHARGE_PERCENT_SENSOR,
    CONF_CHARGER_DEVICE_WALLBOX,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CHARGER_MIN_CHARGE,
    CONF_DYN_GROUP_MAX_PHASE_AMPS,
    CONF_IS_3P,
    CONF_MONO_PHASE,
    DATA_HANDLER,
    DOMAIN,
)
from custom_components.quiet_solar.ha_model.car import QSCar
from custom_components.quiet_solar.ha_model.charger import QSChargerWallbox
from custom_components.quiet_solar.ha_model.dynamic_group import QSDynamicGroup
from custom_components.quiet_solar.ha_model.home import QSHome
from custom_components.quiet_solar.home_model.commands import CMD_ON, copy_command


class _Ct:
    def __init__(self, now: datetime, current_value: float = 0.0, target_value: float = 80.0) -> None:
        self.current_value = current_value
        self.target_value = target_value
        self.first_value_update = now - timedelta(hours=1)
        self.last_value_update = now
        self.last_value_change_update = now - timedelta(minutes=10)

    def is_constraint_met(self, time: datetime, current_value: float) -> bool:
        return False


def _build_charger_and_car(no_sensor: bool):
    hass = MagicMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    data_handler = MagicMock()
    hass.data = {DOMAIN: {DATA_HANDLER: data_handler}}
    config_entry = MagicMock()
    config_entry.entry_id = "test_entry_id"
    config_entry.data = {}

    home = QSHome(
        **{
            CONF_NAME: "TestHome",
            CONF_DYN_GROUP_MAX_PHASE_AMPS: 33,
            CONF_IS_3P: True,
            "hass": hass,
            "config_entry": config_entry,
        }
    )
    data_handler.home = home

    group = QSDynamicGroup(
        **{
            CONF_NAME: "Wallboxes",
            CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            CONF_IS_3P: True,
            "home": home,
            "hass": hass,
            "config_entry": config_entry,
        }
    )
    home.add_device(group)

    now = datetime.now(pytz.UTC)
    with patch("custom_components.quiet_solar.ha_model.charger.entity_registry") as mock_reg:
        mock_reg.async_get = MagicMock()
        mock_reg.async_entries_for_device = MagicMock(return_value=[])
        charger = QSChargerWallbox(
            **{
                CONF_NAME: "Wallbox_1",
                CONF_MONO_PHASE: 1,
                CONF_CHARGER_DEVICE_WALLBOX: "device_wallbox_1",
                CONF_CHARGER_MIN_CHARGE: 6,
                CONF_CHARGER_MAX_CHARGE: 16,
                CONF_IS_3P: False,
                "dynamic_group_name": "Wallboxes",
                "home": home,
                "hass": hass,
                "config_entry": config_entry,
            }
        )
    home.add_device(charger)

    car_conf = {
        CONF_NAME: "EstCar",
        CONF_CAR_BATTERY_CAPACITY: 60000,
        "home": home,
        "hass": hass,
        "config_entry": config_entry,
    }
    if no_sensor:
        car_conf[CONF_CAR_CHARGE_PERCENT_SENSOR] = None
    else:
        car_conf[CONF_CAR_CHARGE_PERCENT_SENSOR] = "sensor.est_car_soc"
    car = QSCar(**car_conf)

    charger.car = car
    car.charger = charger

    # Neutralize the heavy charger plumbing for a focused callback test.
    charger._do_update_charger_state = AsyncMock()
    charger.is_not_plugged = MagicMock(return_value=False)
    charger.current_command = copy_command(CMD_ON)
    charger.is_car_stopped_asking_current = MagicMock(return_value=False)
    charger.is_charging_power_zero = MagicMock(return_value=True)
    charger.on_device_state_change = AsyncMock()
    charger.possible_charge_error_start_time = None
    charger.charger_group.dyn_handle = AsyncMock()

    return charger, car, now


async def test_accumulator_no_loss_no_double_count():
    """N slow cycles each add exactly one increment (pure-delta, no base)."""
    charger, car, now = _build_charger_and_car(no_sensor=True)
    charger._compute_added_charge_update = MagicMock(return_value=0.4)

    times = [now + timedelta(minutes=15 * i) for i in range(5)]
    for t in times:
        ct = _Ct(t)
        await charger.constraint_update_value_callback_soc(ct, t, is_target_percent=True)

    # First cycle only anchors the cursor (no integration); the next four
    # each add 0.4 → 4 * 0.4 = 1.6, with no loss and no double-count.
    assert car._computed_added_delta_soc_percent == pytest.approx(1.6)
    assert car.has_soc_estimate() is False  # pure-delta: no base


async def test_accumulator_absolute_estimate_with_base():
    """With a system base, result tracks clamp(base + accumulated delta)."""
    charger, car, now = _build_charger_and_car(no_sensor=True)
    car._last_valid_base_soc_value = 50.0
    car._computed_added_delta_soc_percent = 0.0
    charger._compute_added_charge_update = MagicMock(return_value=2.0)

    t0 = now
    await charger.constraint_update_value_callback_soc(_Ct(t0), t0, is_target_percent=True)
    # anchor only
    assert car._computed_added_delta_soc_percent == 0.0

    t1 = now + timedelta(minutes=15)
    result, _cont = await charger.constraint_update_value_callback_soc(_Ct(t1), t1, is_target_percent=True)
    assert car._computed_added_delta_soc_percent == pytest.approx(2.0)
    # result derives from the estimate (base + delta) = 52
    assert car.get_car_charge_percent(t1) == pytest.approx(52.0)


async def test_real_power_check_skipped_when_estimating():
    """While estimating, the 'no power detected' check is skipped."""
    charger, car, now = _build_charger_and_car(no_sensor=True)
    car._last_valid_base_soc_value = 10.0  # far below target → would trigger check
    car._computed_added_delta_soc_percent = 0.0
    charger._compute_added_charge_update = MagicMock(return_value=1.0)
    charger._expected_charge_state.value = True
    charger._expected_charge_state.last_ping_time_success = now - timedelta(seconds=3600)

    t1 = now + timedelta(minutes=15)
    await charger.constraint_update_value_callback_soc(_Ct(t1, target_value=80.0), t1, is_target_percent=True)
    charger.on_device_state_change.assert_not_awaited()


async def test_accumulator_inc_none_does_not_advance():
    """When the energy probe returns None, the accumulator and cursor hold."""
    charger, car, now = _build_charger_and_car(no_sensor=True)
    car._last_valid_base_soc_value = 40.0
    car._computed_added_delta_soc_percent = 1.0
    charger._compute_added_charge_update = MagicMock(return_value=None)

    t0 = now
    await charger.constraint_update_value_callback_soc(_Ct(t0), t0, is_target_percent=True)
    cursor_after_anchor = car._delta_soc_last_integration_time

    t1 = now + timedelta(minutes=15)
    await charger.constraint_update_value_callback_soc(_Ct(t1), t1, is_target_percent=True)
    # inc is None → accumulator unchanged, cursor not advanced past the anchor
    assert car._computed_added_delta_soc_percent == 1.0
    assert car._delta_soc_last_integration_time == cursor_after_anchor


async def test_real_sensor_path_not_estimating():
    """A healthy real sensor is read raw and drives the result (not estimated)."""
    charger, car, now = _build_charger_and_car(no_sensor=False)
    car._entity_probed_last_valid_state[car.car_charge_percent_sensor] = (now, 47.0, {})
    charger._compute_added_charge_update = MagicMock(return_value=1.0)

    assert car.is_in_soc_estimation_mode(now) is False
    result, _cont = await charger.constraint_update_value_callback_soc(_Ct(now), now, is_target_percent=True)
    assert result == 47.0


async def test_pure_delta_result_calculus_clamped_to_100():
    """N1 — the pure-delta accumulator value is clamped to <=100 before use."""
    charger, car, now = _build_charger_and_car(no_sensor=True)
    car._computed_added_delta_soc_percent = 130.0  # already over 100 (long session)
    car._delta_soc_last_integration_time = now - timedelta(minutes=15)
    charger._compute_added_charge_update = MagicMock(return_value=10.0)
    result, _cont = await charger.constraint_update_value_callback_soc(
        _Ct(now, target_value=100.0), now, is_target_percent=True
    )
    assert result == 100.0


async def test_manual_override_on_healthy_car_does_not_suppress_fault_check():
    """S1 — a manual override on a healthy, sensor-equipped car must NOT disable
    the zero-power hardware-fault detection."""
    charger, car, now = _build_charger_and_car(no_sensor=False)
    # Healthy fresh sensor, but a manual override is active → estimating, yet the
    # SOC sensor is NOT distrusted.
    car._entity_probed_last_valid_state[car.car_charge_percent_sensor] = (now, 50.0, {})
    car._user_base_soc_value = 10.0  # far below target → fault window applies
    car._computed_added_delta_soc_percent = 0.0
    assert car.is_in_soc_estimation_mode(now) is True
    assert car.is_soc_sensor_distrusted() is False
    charger._compute_added_charge_update = MagicMock(return_value=0.0)
    charger._expected_charge_state.value = True
    charger._expected_charge_state.last_ping_time_success = now - timedelta(seconds=3600)

    t1 = now + timedelta(minutes=15)
    await charger.constraint_update_value_callback_soc(_Ct(t1, target_value=80.0), t1, is_target_percent=True)
    # power is zero (is_charging_power_zero mocked True) → fault reported
    charger.on_device_state_change.assert_awaited()


async def test_real_efficiency_integral_within_1pct():
    """S2 / AC5 — the real efficiency-aware integral matches an independent
    computation within ±1% SOC, with no double-count across N cycles."""
    charger, car, now = _build_charger_and_car(no_sensor=True)
    car.car_battery_capacity = 60000  # Wh
    car._last_valid_base_soc_value = 50.0  # base B
    car._computed_added_delta_soc_percent = 0.0
    eff = charger.efficiency_factor  # the charger's real efficiency factor

    # Drive the REAL _compute_added_charge_update via the energy source: a fixed
    # 11 kW over each 15-minute slice → 2750 Wh/slice.
    power_w = 11000.0
    dt_s = 15 * 60
    energy_per_slice_wh = power_w * dt_s / 3600.0
    charger._can_use_group_power_sensor = MagicMock(return_value=False)
    charger.get_device_real_energy = MagicMock(return_value=energy_per_slice_wh)

    n_integrating_cycles = 4
    t = now
    await charger.constraint_update_value_callback_soc(_Ct(t), t, is_target_percent=True)  # anchor
    for _ in range(n_integrating_cycles):
        t = t + timedelta(seconds=dt_s)
        result, _cont = await charger.constraint_update_value_callback_soc(_Ct(t), t, is_target_percent=True)

    # Independent integral: B + sum(100 * (P*dt/e) / C)
    expected_delta = n_integrating_cycles * 100.0 * (energy_per_slice_wh / eff) / 60000.0
    expected = max(0.0, min(100.0, 50.0 + expected_delta))
    assert abs(result - expected) <= 1.0
    # no double-count: accumulator equals exactly the integrated delta
    assert abs(car._computed_added_delta_soc_percent - expected_delta) <= 0.01


