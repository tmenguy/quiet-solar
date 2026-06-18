"""Tests for the estimated-SOC model (Story QS-243).

Covers the effective-SOC accessors, estimation-mode gating, the
`can_use_charge_percent_constraints` flip, manual entry / reset, persistence,
the per-cycle recovery state machine, the fresh→stale capture, and the
estimation-vs-staleness orthogonality.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz

from custom_components.quiet_solar.const import (
    BINARY_SENSOR_CAR_IS_SOC_ESTIMATED,
    BUTTON_CAR_RESET_SOC_ESTIMATE,
    CAR_SOC_STALE_THRESHOLD_S,
    CAR_STALE_MODE_AUTO,
    CAR_STALE_MODE_FORCE_NOT_STALE,
    CAR_STALE_MODE_FORCE_STALE,
    NUMBER_CAR_MANUAL_SOC_PERCENT,
)
from custom_components.quiet_solar.ha_model.car import QSCar


# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_car(fake_hass, mock_data_handler, **overrides) -> QSCar:
    from tests.ha_tests.const import MOCK_CAR_CONFIG

    config = {
        **MOCK_CAR_CONFIG,
        "hass": fake_hass,
        "config_entry": MagicMock(),
        "data_handler": mock_data_handler,
        "home": mock_data_handler.home,
    }
    config.update(overrides)
    return QSCar(**config)


@pytest.fixture
def est_car(fake_hass, mock_data_handler, current_time):
    """A normal car: SOC sensor + battery capacity, not invited."""
    return _make_car(fake_hass, mock_data_handler)


@pytest.fixture
def est_car_no_sensor(fake_hass, mock_data_handler, current_time):
    """A real car with a true capacity but no SOC sensor."""
    from custom_components.quiet_solar.const import CONF_CAR_CHARGE_PERCENT_SENSOR

    return _make_car(fake_hass, mock_data_handler, **{CONF_CAR_CHARGE_PERCENT_SENSOR: None})


def _set_soc(car: QSCar, value, time):
    car._entity_probed_last_valid_state[car.car_charge_percent_sensor] = (time, value, {})


# ── Constants ────────────────────────────────────────────────────────────


def test_constants():
    assert BINARY_SENSOR_CAR_IS_SOC_ESTIMATED == "qs_car_is_soc_estimated"
    assert BUTTON_CAR_RESET_SOC_ESTIMATE == "qs_car_reset_soc_estimate"
    assert NUMBER_CAR_MANUAL_SOC_PERCENT == "qs_car_manual_soc_percent"


# ── AC1: healthy sensor passthrough ──────────────────────────────────────


def test_healthy_sensor_passthrough(est_car, current_time):
    _set_soc(est_car, 42.0, current_time)
    assert est_car.is_in_soc_estimation_mode(current_time) is False
    assert est_car.get_car_charge_percent(current_time) == 42.0


# ── Estimation-mode gating arms ──────────────────────────────────────────


def test_estimation_mode_invited(est_car):
    est_car.car_is_invited = True
    assert est_car.is_in_soc_estimation_mode() is False


def test_estimation_mode_no_capacity(est_car):
    est_car.car_battery_capacity = None
    assert est_car.is_in_soc_estimation_mode() is False
    est_car.car_battery_capacity = 0
    assert est_car.is_in_soc_estimation_mode() is False


def test_estimation_mode_no_sensor(est_car_no_sensor):
    assert est_car_no_sensor.is_in_soc_estimation_mode() is True


def test_estimation_mode_stale_percent(est_car):
    est_car.car_api_stale_percent_mode = True
    assert est_car.is_in_soc_estimation_mode() is True


def test_estimation_mode_manual_base_on_healthy(est_car):
    est_car._user_base_soc_value = 55.0
    assert est_car.is_in_soc_estimation_mode() is True


# ── _estimated_soc_percent: three arms + clamp ───────────────────────────


def test_estimated_soc_user_base(est_car):
    est_car._user_base_soc_value = 40.0
    est_car._computed_added_delta_soc_percent = 5.0
    assert est_car._estimated_soc_percent == 45.0


def test_estimated_soc_system_base(est_car):
    est_car._last_valid_base_soc_value = 30.0
    est_car._computed_added_delta_soc_percent = 2.0
    assert est_car._estimated_soc_percent == 32.0


def test_estimated_soc_no_base_is_none(est_car):
    est_car._computed_added_delta_soc_percent = 7.0
    assert est_car._estimated_soc_percent is None


def test_estimated_soc_clamped(est_car):
    est_car._user_base_soc_value = 98.0
    est_car._computed_added_delta_soc_percent = 10.0
    assert est_car._estimated_soc_percent == 100.0
    est_car._user_base_soc_value = 2.0
    est_car._computed_added_delta_soc_percent = -10.0
    assert est_car._estimated_soc_percent == 0.0


def test_estimated_soc_delta_none(est_car):
    est_car._user_base_soc_value = 50.0
    est_car._computed_added_delta_soc_percent = None
    assert est_car._estimated_soc_percent == 50.0


def test_get_car_charge_percent_returns_estimate(est_car, current_time):
    _set_soc(est_car, 10.0, current_time)
    est_car._user_base_soc_value = 60.0
    est_car._computed_added_delta_soc_percent = 5.0
    # estimate wins over raw sensor
    assert est_car.get_car_charge_percent(current_time) == 65.0
    assert est_car.get_car_charge_percent_raw_sensor(current_time) == 10.0


def test_raw_sensor_none_when_no_sensor(est_car_no_sensor, current_time):
    assert est_car_no_sensor.get_car_charge_percent_raw_sensor(current_time) is None


def test_pure_delta_get_charge_percent_none(est_car_no_sensor):
    # estimating, no base -> get_car_charge_percent returns None
    assert est_car_no_sensor.get_car_charge_percent() is None
    assert est_car_no_sensor._estimated_soc_percent is None


# ── AC4 + invariants: can_use_charge_percent_constraints ─────────────────


def test_can_use_percent_real_sensor_and_capacity(est_car):
    assert est_car.can_use_charge_percent_constraints() is True


def test_can_use_percent_no_sensor_with_capacity(est_car_no_sensor):
    assert est_car_no_sensor.can_use_charge_percent_constraints() is True


def test_can_use_percent_invited_false(est_car):
    est_car.car_is_invited = True
    assert est_car.can_use_charge_percent_constraints() is False


def test_can_use_percent_no_capacity_false(est_car):
    est_car.car_battery_capacity = None
    assert est_car.can_use_charge_percent_constraints() is False
    est_car.car_battery_capacity = 0
    assert est_car.can_use_charge_percent_constraints() is False


# ── Manual entry / reset ─────────────────────────────────────────────────


async def test_user_set_manual_soc(est_car, current_time):
    _set_soc(est_car, 30.0, current_time - timedelta(seconds=10))
    est_car.charger = None
    await est_car.user_set_manual_soc_percent(72.4)
    assert est_car._user_base_soc_value == 72.0  # int() then float
    assert est_car._user_base_soc_entry_sensor_value == 30.0
    assert est_car._computed_added_delta_soc_percent == 0.0
    assert est_car._delta_soc_last_integration_time is None


async def test_user_set_manual_soc_clamped(est_car):
    est_car.charger = None
    await est_car.user_set_manual_soc_percent(150)
    assert est_car._user_base_soc_value == 100.0
    await est_car.user_set_manual_soc_percent(-5)
    assert est_car._user_base_soc_value == 0.0


async def test_user_set_manual_soc_for_init_noop(est_car):
    await est_car.user_set_manual_soc_percent(50, for_init=True)
    assert est_car._user_base_soc_value is None


async def test_user_set_manual_soc_updates_charger(est_car):
    charger = MagicMock()
    charger.update_charger_for_user_change = AsyncMock()
    est_car.charger = charger
    await est_car.user_set_manual_soc_percent(40)
    charger.update_charger_for_user_change.assert_awaited_once()


def test_user_reset_soc_estimate(est_car):
    est_car._user_base_soc_value = 10.0
    est_car._last_valid_base_soc_value = 20.0
    est_car._computed_added_delta_soc_percent = 3.0
    est_car._user_base_soc_entry_sensor_value = 5.0
    est_car._user_base_soc_entry_api_stale = True
    est_car._delta_soc_last_integration_time = "x"
    est_car.reset_soc_estimate()
    assert est_car._user_base_soc_value is None
    assert est_car._last_valid_base_soc_value is None
    assert est_car._computed_added_delta_soc_percent is None
    assert est_car._user_base_soc_entry_sensor_value is None
    assert est_car._user_base_soc_entry_api_stale is None
    assert est_car._delta_soc_last_integration_time is None


async def test_user_button_reset_no_charger(est_car):
    est_car._user_base_soc_value = 10.0
    est_car.charger = None
    await est_car.user_button_reset_soc_estimate()
    assert est_car._user_base_soc_value is None


async def test_user_button_reset_with_charger(est_car):
    charger = MagicMock()
    charger.update_charger_for_user_change = AsyncMock()
    est_car.charger = charger
    est_car._user_base_soc_value = 10.0
    await est_car.user_button_reset_soc_estimate()
    assert est_car._user_base_soc_value is None
    charger.update_charger_for_user_change.assert_awaited_once()


def test_manual_soc_number_value_property(est_car):
    assert est_car.qs_car_manual_soc_percent == 0.0
    est_car._user_base_soc_value = 63.0
    assert est_car.qs_car_manual_soc_percent == 63.0


async def test_user_clean_and_reset_clears_estimate(est_car):
    est_car._user_base_soc_value = 44.0
    est_car.charger = None
    est_car.home = None
    await est_car.user_clean_and_reset()
    assert est_car._user_base_soc_value is None


# ── Persistence ──────────────────────────────────────────────────────────


def test_persistence_round_trip(est_car, fake_hass, mock_data_handler, current_time):
    est_car._user_base_soc_value = 40.0
    est_car._last_valid_base_soc_value = 35.0
    est_car._computed_added_delta_soc_percent = 4.5
    est_car._user_base_soc_entry_sensor_value = 33.0

    data: dict = {}
    est_car.update_to_be_saved_extra_device_info(data)
    assert data["user_base_soc_value"] == 40.0
    assert data["last_valid_base_soc_value"] == 35.0
    assert data["computed_added_delta_soc_percent"] == 4.5
    assert data["user_base_soc_entry_sensor_value"] == 33.0

    other = _make_car(fake_hass, mock_data_handler)
    other.use_saved_extra_device_info(data)
    assert other._user_base_soc_value == 40.0
    assert other._last_valid_base_soc_value == 35.0
    assert other._computed_added_delta_soc_percent == 4.5
    assert other._user_base_soc_entry_sensor_value == 33.0
    assert other._delta_soc_last_integration_time is None


def test_persistence_entry_api_stale_round_trip(est_car, fake_hass, mock_data_handler):
    est_car._user_base_soc_value = 40.0
    est_car._user_base_soc_entry_api_stale = True
    data: dict = {}
    est_car.update_to_be_saved_extra_device_info(data)
    assert data["user_base_soc_entry_api_stale"] is True
    other = _make_car(fake_hass, mock_data_handler)
    other.use_saved_extra_device_info(data)
    assert other._user_base_soc_entry_api_stale is True


def test_persistence_pre_qs243_blob(est_car):
    # A saved blob from before this story lacks the SOC keys.
    est_car._user_base_soc_value = 12.0
    est_car._user_base_soc_entry_api_stale = True
    est_car.use_saved_extra_device_info({"current_forecasted_person_name_from_boot": None})
    assert est_car._user_base_soc_value is None
    assert est_car._last_valid_base_soc_value is None
    assert est_car._computed_added_delta_soc_percent is None
    assert est_car._user_base_soc_entry_sensor_value is None
    assert est_car._user_base_soc_entry_api_stale is None


# ── Recovery state machine: _update_soc_estimation ───────────────────────


def test_recovery_any_change_clears(est_car, current_time):
    est_car._user_base_soc_value = 70.0
    est_car._user_base_soc_entry_sensor_value = 50.0
    _set_soc(est_car, 51.0, current_time)  # different from entry ref
    est_car._update_soc_estimation(current_time)
    assert est_car._user_base_soc_value is None


def test_recovery_same_value_persists(est_car, current_time):
    est_car._user_base_soc_value = 70.0
    est_car._user_base_soc_entry_sensor_value = 50.0
    _set_soc(est_car, 50.0, current_time)  # same as entry ref
    est_car._update_soc_estimation(current_time)
    assert est_car._user_base_soc_value == 70.0


def test_recovery_ref_none_clears_on_fresh(est_car, current_time):
    est_car._user_base_soc_value = 70.0
    est_car._user_base_soc_entry_sensor_value = None
    _set_soc(est_car, 50.0, current_time)
    est_car._update_soc_estimation(current_time)
    assert est_car._user_base_soc_value is None


def test_recovery_no_user_base_noop(est_car, current_time):
    _set_soc(est_car, 50.0, current_time)
    est_car._update_soc_estimation(current_time)  # nothing to do
    assert est_car._user_base_soc_value is None


def test_recovery_no_sensor_noop(est_car_no_sensor, current_time):
    est_car_no_sensor._user_base_soc_value = 70.0
    est_car_no_sensor._update_soc_estimation(current_time)
    assert est_car_no_sensor._user_base_soc_value == 70.0


def test_recovery_stale_sensor_skips(est_car, current_time):
    est_car._user_base_soc_value = 70.0
    est_car._user_base_soc_entry_sensor_value = 50.0
    stale_time = current_time - timedelta(seconds=CAR_SOC_STALE_THRESHOLD_S + 1)
    _set_soc(est_car, 51.0, stale_time)
    est_car._update_soc_estimation(current_time)
    assert est_car._user_base_soc_value == 70.0


async def test_user_set_manual_records_entry_api_stale_true(est_car, current_time):
    _set_soc(est_car, 30.0, current_time)
    est_car.car_api_stale_percent_mode = True
    est_car.charger = None
    await est_car.user_set_manual_soc_percent(60)
    assert est_car._user_base_soc_entry_api_stale is True


async def test_manual_soc_finite_guard(est_car):
    est_car.charger = None
    await est_car.user_set_manual_soc_percent(float("nan"))
    assert est_car._user_base_soc_value is None
    await est_car.user_set_manual_soc_percent(float("inf"))
    assert est_car._user_base_soc_value is None
    await est_car.user_set_manual_soc_percent("not-a-number")
    assert est_car._user_base_soc_value is None


async def test_manual_soc_rounds_not_truncates(est_car):
    est_car.charger = None
    await est_car.user_set_manual_soc_percent(50.9)
    assert est_car._user_base_soc_value == 51.0


async def test_manual_soc_half_up_rounding(est_car):
    # N3 — half-up, not Python banker's rounding (round(2.5) == 2).
    est_car.charger = None
    await est_car.user_set_manual_soc_percent(2.5)
    assert est_car._user_base_soc_value == 3.0


def test_is_soc_sensor_distrusted(est_car, est_car_no_sensor):
    assert est_car.is_soc_sensor_distrusted() is False
    est_car.car_api_stale_percent_mode = True
    assert est_car.is_soc_sensor_distrusted() is True
    # no-sensor car is always distrusted
    assert est_car_no_sensor.is_soc_sensor_distrusted() is True


# ── M1: 4-case override recovery keyed on entry API state ────────────────


def test_recovery_case1_entered_during_stale_clears_on_exit(est_car, current_time):
    # Case 1: override entered during stale mode → clear once stale exits.
    est_car._user_base_soc_value = 70.0
    est_car._user_base_soc_entry_api_stale = True
    est_car._user_base_soc_entry_sensor_value = None
    _set_soc(est_car, 55.0, current_time)

    # Still stale → keep the override.
    est_car.car_api_stale_percent_mode = True
    est_car._update_soc_estimation(current_time)
    assert est_car._user_base_soc_value == 70.0

    # Exited stale + fresh valid sensor → live sensor wins.
    est_car.car_api_stale_percent_mode = False
    est_car._update_soc_estimation(current_time)
    assert est_car._user_base_soc_value is None


def test_recovery_case1_force_not_stale_clears_while_time_stale(est_car, current_time):
    # S2 — override entered during stale; user sets Force-Not-Stale while the SOC
    # sensor is still time-stale → the (force-trusted) live sensor takes over.
    est_car._user_base_soc_value = 70.0
    est_car._user_base_soc_entry_api_stale = True
    est_car._user_base_soc_entry_sensor_value = None
    stale_time = current_time - timedelta(seconds=CAR_SOC_STALE_THRESHOLD_S + 1)
    _set_soc(est_car, 55.0, stale_time)  # genuinely time-stale value
    est_car.car_stale_mode_override = CAR_STALE_MODE_FORCE_NOT_STALE
    est_car.car_api_stale_percent_mode = False  # forced not-stale → exited stale mode
    est_car._update_soc_estimation(current_time)
    assert est_car._user_base_soc_value is None


def test_recovery_case1_time_stale_without_force_keeps(est_car, current_time):
    # Without Force-Not-Stale, a time-stale sensor does not trigger recovery.
    est_car._user_base_soc_value = 70.0
    est_car._user_base_soc_entry_api_stale = True
    est_car._user_base_soc_entry_sensor_value = None
    stale_time = current_time - timedelta(seconds=CAR_SOC_STALE_THRESHOLD_S + 1)
    _set_soc(est_car, 55.0, stale_time)
    est_car.car_api_stale_percent_mode = False
    est_car._update_soc_estimation(current_time)
    assert est_car._user_base_soc_value == 70.0


def test_number_entity_resyncs_on_runtime_reset(est_car):
    # S1 — the manual-SOC number entity tracks the device value, so a runtime
    # reset (e.g. plug-in / reset button / recovery) is reflected in the card.
    from custom_components.quiet_solar.number import create_ha_number_for_QSCar

    entity = create_ha_number_for_QSCar(est_car)[0]
    entity.hass = est_car.hass
    entity.async_write_ha_state = MagicMock()

    est_car._user_base_soc_value = 80.0
    entity.async_update_callback(datetime.now(tz=pytz.UTC))
    assert entity._attr_native_value == 80.0

    est_car.reset_soc_estimate()
    entity.async_update_callback(datetime.now(tz=pytz.UTC))
    assert entity._attr_native_value == 0.0


def test_recovery_case2_stale_blip_preserves_delta(est_car, current_time):
    # Case 2 regression: a transient stale blip must not lose the delta.
    est_car._user_base_soc_value = 60.0
    est_car._user_base_soc_entry_api_stale = False
    est_car._user_base_soc_entry_sensor_value = 50.0
    est_car._computed_added_delta_soc_percent = 5.0
    est_car._delta_soc_last_integration_time = current_time

    # Stale blip recovers: accumulator preserved (user override owns it).
    est_car.car_api_stale_percent_mode = True
    est_car._exit_stale_mode()
    assert est_car._computed_added_delta_soc_percent == 5.0

    # Sensor still reads the entry value → keep base + delta.
    _set_soc(est_car, 50.0, current_time)
    est_car._update_soc_estimation(current_time)
    assert est_car._user_base_soc_value == 60.0
    assert est_car._computed_added_delta_soc_percent == 5.0
    assert est_car._estimated_soc_percent == 65.0


def test_recovery_raw_none_skips(est_car, current_time):
    # Defensive guard: sensor not stale but the raw read still yields None.
    est_car._user_base_soc_value = 70.0
    est_car._user_base_soc_entry_sensor_value = 50.0
    est_car._is_soc_sensor_stale = lambda time: False
    est_car.get_car_charge_percent_raw_sensor = lambda time=None, tolerance_seconds=None: None
    est_car._update_soc_estimation(current_time)
    assert est_car._user_base_soc_value == 70.0


# ── Fresh→stale capture ──────────────────────────────────────────────────


def test_capture_last_valid_base(est_car, current_time):
    _set_soc(est_car, 64.0, current_time)
    est_car._enter_stale_percent_mode(current_time)
    assert est_car.car_api_stale_percent_mode is True
    assert est_car._last_valid_base_soc_value == 64.0
    assert est_car._computed_added_delta_soc_percent == 0.0
    assert est_car._delta_soc_last_integration_time is None


def test_capture_skipped_when_user_base(est_car, current_time):
    _set_soc(est_car, 64.0, current_time)
    est_car._user_base_soc_value = 80.0
    est_car._enter_stale_percent_mode(current_time)
    assert est_car._last_valid_base_soc_value is None


def test_capture_skipped_when_system_base_exists(est_car, current_time):
    _set_soc(est_car, 64.0, current_time)
    est_car._last_valid_base_soc_value = 30.0
    est_car._enter_stale_percent_mode(current_time)
    assert est_car._last_valid_base_soc_value == 30.0


def test_capture_skipped_when_raw_none(est_car, current_time):
    est_car._entity_probed_last_valid_state[est_car.car_charge_percent_sensor] = None
    est_car._enter_stale_percent_mode(current_time)
    assert est_car._last_valid_base_soc_value is None


def test_enter_stale_for_init_no_capture(est_car, current_time):
    _set_soc(est_car, 64.0, current_time)
    est_car._enter_stale_percent_mode(None, for_init=True)
    assert est_car.car_api_stale_percent_mode is True
    assert est_car._last_valid_base_soc_value is None


# ── S4 / AC3: end-to-end fresh→stale→recover flow ────────────────────────


def test_end_to_end_stale_capture_and_recovery(est_car, current_time):
    """Drive `_update_car_api_staleness` across a full fresh→stale→recover cycle:
    capture fires exactly once on the genuine edge, then recovery hands SOC back
    to the live sensor."""
    tracker = est_car.car_tracker
    plugged = est_car.car_plugged
    soc = est_car.car_charge_percent_sensor

    def _fresh_non_soc(t):
        est_car._entity_probed_last_valid_state[tracker] = (t, "home", {})
        est_car._entity_probed_last_valid_state[plugged] = (t, "on", {})

    # 1) Everything fresh, SOC = 60 → not stale.
    _fresh_non_soc(current_time)
    est_car._entity_probed_last_valid_state[soc] = (current_time, 60.0, {})
    est_car._update_car_api_staleness(current_time)
    assert est_car.car_api_stale_percent_mode is False
    assert est_car.get_car_charge_percent(current_time) == 60.0

    # 2) SOC sensor goes stale (others fresh) → SOC-only stale edge captures L=60.
    stale_time = current_time - timedelta(seconds=CAR_SOC_STALE_THRESHOLD_S + 1)
    est_car._entity_probed_last_valid_state[soc] = (stale_time, 60.0, {})
    _fresh_non_soc(current_time)
    est_car._update_car_api_staleness(current_time)
    assert est_car.car_api_stale_percent_mode is True
    assert est_car._last_valid_base_soc_value == 60.0
    assert est_car._computed_added_delta_soc_percent == 0.0
    assert est_car.get_car_charge_percent(current_time) == 60.0  # base + 0 delta

    # 3) A second stale cycle must NOT re-capture (genuine-edge guard).
    est_car._last_valid_base_soc_value = 999.0  # sentinel — must be left untouched
    est_car._update_car_api_staleness(current_time)
    assert est_car._last_valid_base_soc_value == 999.0
    est_car._last_valid_base_soc_value = 60.0

    # 4) API recovers: SOC fresh again with a new value → live sensor takes over.
    recover_time = current_time + timedelta(seconds=10)
    _fresh_non_soc(recover_time)
    est_car._entity_probed_last_valid_state[soc] = (recover_time, 65.0, {})
    est_car._update_car_api_staleness(recover_time)
    assert est_car.car_api_stale_percent_mode is False
    assert est_car._last_valid_base_soc_value is None
    assert est_car.get_car_charge_percent(recover_time) == 65.0


# ── _exit_stale_mode clears system base ──────────────────────────────────


def test_exit_stale_clears_system_base_no_user_override(est_car):
    # No user override: the system-base recovery clears base + accumulator.
    est_car.car_api_stale_percent_mode = True
    est_car._last_valid_base_soc_value = 40.0
    est_car._computed_added_delta_soc_percent = 6.0
    est_car._delta_soc_last_integration_time = "x"
    est_car._exit_stale_mode()
    assert est_car._last_valid_base_soc_value is None
    assert est_car._computed_added_delta_soc_percent is None
    assert est_car._delta_soc_last_integration_time is None


def test_exit_stale_preserves_accumulator_with_user_override(est_car):
    # M1 defect 2: a transient stale blip must NOT wipe an override's delta.
    est_car.car_api_stale_percent_mode = True
    est_car._last_valid_base_soc_value = 40.0
    est_car._computed_added_delta_soc_percent = 6.0
    est_car._delta_soc_last_integration_time = "x"
    est_car._user_base_soc_value = 88.0
    est_car._exit_stale_mode()
    # the override owns its accumulator lifecycle → delta + cursor preserved
    assert est_car._computed_added_delta_soc_percent == 6.0
    assert est_car._delta_soc_last_integration_time == "x"
    # user base untouched; system base cleared
    assert est_car._user_base_soc_value == 88.0
    assert est_car._last_valid_base_soc_value is None


# ── AC12: orthogonality ──────────────────────────────────────────────────


def test_orthogonality_manual_override_not_stale(est_car, current_time):
    fresh = current_time - timedelta(seconds=30)
    for sensor_id in est_car._car_api_all_sensors:
        est_car._entity_probed_last_valid_state[sensor_id] = (fresh, "value", {})
    est_car._user_base_soc_value = 55.0
    est_car._car_api_stale = False
    assert est_car.is_in_soc_estimation_mode(current_time) is True
    assert est_car.is_car_effectively_stale(current_time) is False


# ── is_car_charge_growing gating ─────────────────────────────────────────


def test_is_car_charge_growing_none_when_estimating(est_car_no_sensor, current_time):
    assert est_car_no_sensor.is_car_charge_growing(60, current_time) is None


def test_is_car_charge_growing_reads_sensor_when_not_estimating(est_car, current_time):
    _set_soc(est_car, 50.0, current_time)
    # not estimating -> delegates to is_sensor_growing (returns None/bool, no exception)
    result = est_car.is_car_charge_growing(60, current_time)
    assert result is None or isinstance(result, bool)


# ── Platform entity creation ─────────────────────────────────────────────


def test_create_number_for_car(est_car):
    from custom_components.quiet_solar.number import create_ha_number_for_QSCar

    entities = create_ha_number_for_QSCar(est_car)
    keys = [e.entity_description.translation_key for e in entities]
    assert NUMBER_CAR_MANUAL_SOC_PERCENT in keys


def test_create_number_for_invited_car_empty(est_car):
    from custom_components.quiet_solar.number import create_ha_number_for_QSCar

    est_car.car_is_invited = True
    assert create_ha_number_for_QSCar(est_car) == []


async def test_number_async_set_fn_writes_manual_soc(est_car):
    from custom_components.quiet_solar.number import create_ha_number_for_QSCar

    est_car.charger = None
    entities = create_ha_number_for_QSCar(est_car)
    desc = entities[0].entity_description
    await desc.async_set_fn(est_car, 66, False)
    assert est_car._user_base_soc_value == 66.0


def test_create_button_for_car_includes_reset(est_car):
    from custom_components.quiet_solar.button import create_ha_button_for_QSCar

    entities = create_ha_button_for_QSCar(est_car)
    keys = [e.entity_description.key for e in entities]
    assert BUTTON_CAR_RESET_SOC_ESTIMATE in keys


def test_create_button_for_invited_car_no_reset(est_car):
    from custom_components.quiet_solar.button import create_ha_button_for_QSCar

    est_car.car_is_invited = True
    entities = create_ha_button_for_QSCar(est_car)
    keys = [e.entity_description.key for e in entities]
    assert BUTTON_CAR_RESET_SOC_ESTIMATE not in keys


def test_create_binary_sensor_for_car_includes_estimated(est_car):
    from custom_components.quiet_solar.binary_sensor import create_ha_binary_sensor_for_QSCar

    entities = create_ha_binary_sensor_for_QSCar(est_car)
    keys = [e.entity_description.key for e in entities]
    assert BINARY_SENSOR_CAR_IS_SOC_ESTIMATED in keys


def test_binary_sensor_estimated_value_fn(est_car, current_time):
    """The asterisk is driven by is_in_soc_estimation_mode, not has_soc_estimate (AC5)."""
    from custom_components.quiet_solar.binary_sensor import create_ha_binary_sensor_for_QSCar

    entities = create_ha_binary_sensor_for_QSCar(est_car)
    est = next(e for e in entities if e.entity_description.key == BINARY_SENSOR_CAR_IS_SOC_ESTIMATED)

    # Fresh SOC sensor, no override → not estimating → no asterisk
    _set_soc(est_car, 42.0, current_time)
    assert est.entity_description.value_fn(est_car, "k") is False

    # A manual SOC value active → estimating → asterisk
    est_car._user_base_soc_value = 50.0
    assert est.entity_description.value_fn(est_car, "k") is True
    est_car._user_base_soc_value = None

    # SOC stale-percent mode (force-stale / SOC stale / API failure) → asterisk
    est_car.car_api_stale_percent_mode = True
    assert est.entity_description.value_fn(est_car, "k") is True
    est_car.car_api_stale_percent_mode = False


def test_binary_sensor_estimated_value_fn_no_sensor(est_car_no_sensor):
    """A car without a SOC sensor always estimates → asterisk always on (AC5)."""
    from custom_components.quiet_solar.binary_sensor import create_ha_binary_sensor_for_QSCar

    entities = create_ha_binary_sensor_for_QSCar(est_car_no_sensor)
    est = next(e for e in entities if e.entity_description.key == BINARY_SENSOR_CAR_IS_SOC_ESTIMATED)
    assert est.entity_description.value_fn(est_car_no_sensor, "k") is True


def test_create_binary_sensor_invited_no_estimated(est_car):
    from custom_components.quiet_solar.binary_sensor import create_ha_binary_sensor_for_QSCar

    est_car.car_is_invited = True
    entities = create_ha_binary_sensor_for_QSCar(est_car)
    keys = [e.entity_description.key for e in entities]
    assert BINARY_SENSOR_CAR_IS_SOC_ESTIMATED not in keys


# ── SF4: manual-SOC reset clears value + asterisk (AC7) ───────────────────


def test_manual_soc_reset_case1_clears_value_and_asterisk(est_car, current_time):
    """SF4/AC7 (Case 1): a manual SOC value entered while stale clears once the car
    exits stale mode and a fresh raw read lands — and the asterisk clears with it."""
    est_car.car_api_stale_percent_mode = True
    est_car._user_base_soc_value = 60.0
    est_car._user_base_soc_entry_api_stale = True
    est_car._user_base_soc_entry_sensor_value = None
    assert est_car.is_in_soc_estimation_mode(current_time) is True  # asterisk on (manual value)

    # Car has exited stale-percent mode and a fresh valid read arrives
    est_car.car_api_stale_percent_mode = False
    _set_soc(est_car, 72.0, current_time)
    est_car._update_soc_estimation(current_time)

    assert est_car._user_base_soc_value is None  # manual value cleared
    assert est_car.is_in_soc_estimation_mode(current_time) is False  # asterisk cleared


def test_manual_soc_reset_case2_clears_on_differing_read(est_car, current_time):
    """SF4/AC7 (Case 2): entered not-stale with a value — clears on a differing fresh read."""
    est_car._user_base_soc_value = 60.0
    est_car._user_base_soc_entry_api_stale = False
    est_car._user_base_soc_entry_sensor_value = 50.0

    _set_soc(est_car, 72.0, current_time)  # differs from the 50.0 entry reference
    est_car._update_soc_estimation(current_time)

    assert est_car._user_base_soc_value is None
    assert est_car.is_in_soc_estimation_mode(current_time) is False


def test_manual_soc_reset_case3_clears_on_any_valid_read(est_car, current_time):
    """SF4/R2-SF1/AC7 (Case 3): entered not-stale with no valid entry value — any valid
    fresh raw read clears the manual SOC value (and the asterisk), with no force override."""
    est_car._user_base_soc_value = 60.0
    est_car._user_base_soc_entry_api_stale = False
    est_car._user_base_soc_entry_sensor_value = None  # Case 3 — no valid entry reference
    assert est_car.car_stale_mode_override == CAR_STALE_MODE_AUTO  # no force path in play
    assert est_car.is_in_soc_estimation_mode(current_time) is True  # asterisk on

    _set_soc(est_car, 72.0, current_time)  # any valid fresh read
    est_car._update_soc_estimation(current_time)

    assert est_car._user_base_soc_value is None
    assert est_car.is_in_soc_estimation_mode(current_time) is False  # asterisk cleared


def test_manual_soc_reset_force_not_stale_proceeds_when_time_stale(est_car, current_time):
    """SF4/AC7: Force-Not-Stale treats the sensor as trusted, so the reset proceeds
    even when the SOC sensor is time-stale."""
    est_car.car_stale_mode_override = CAR_STALE_MODE_FORCE_NOT_STALE
    est_car._user_base_soc_value = 60.0
    est_car._user_base_soc_entry_api_stale = False
    est_car._user_base_soc_entry_sensor_value = None  # Case 3 — any valid value clears

    # Sensor is time-stale (>1h) but Force-Not-Stale asserts it is trusted
    stale = current_time - timedelta(seconds=CAR_SOC_STALE_THRESHOLD_S + 1)
    _set_soc(est_car, 72.0, stale)
    est_car._update_soc_estimation(current_time)

    assert est_car._user_base_soc_value is None
    assert est_car.is_in_soc_estimation_mode(current_time) is False


# ── SF5: force overrides drive the asterisk via the rewired value_fn (AC6) ─


async def test_force_stale_drives_asterisk_via_value_fn(est_car, current_time):
    """SF5/AC6: force-stale enters estimation → asterisk on through the rewired value_fn."""
    from custom_components.quiet_solar.binary_sensor import create_ha_binary_sensor_for_QSCar

    entities = create_ha_binary_sensor_for_QSCar(est_car)
    est = next(e for e in entities if e.entity_description.key == BINARY_SENSOR_CAR_IS_SOC_ESTIMATED)

    _set_soc(est_car, 50.0, current_time)
    assert est.entity_description.value_fn(est_car, "k") is False

    await est_car.user_set_stale_mode(CAR_STALE_MODE_FORCE_STALE, for_init=True)
    assert est_car.car_api_stale_percent_mode is True
    assert est.entity_description.value_fn(est_car, "k") is True


def test_force_not_stale_recovers_and_clears_asterisk(est_car, current_time):
    """SF5/AC6: force-not-stale lets recovery proceed; the asterisk clears after exit."""
    est_car.car_api_stale_percent_mode = True
    est_car.car_stale_mode_override = CAR_STALE_MODE_FORCE_NOT_STALE

    assert est_car.can_exit_stale_percent_mode(current_time) is True
    est_car._exit_stale_mode()
    assert est_car.is_in_soc_estimation_mode(current_time) is False
