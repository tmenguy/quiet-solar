"""Tests for car API staleness detection (Story 3.9).

Tests cover:
- Constants existence and values
- Sensor classification (critical vs supplementary)
- Staleness detection (Feature A)
- Contradiction detection (Feature B)
- Stale-percent mode behavior
- Recovery logic
- Effective stale state with select override
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytz
import pytest

from custom_components.quiet_solar.const import (
    BINARY_SENSOR_CAR_API_OK,
    BINARY_SENSOR_CAR_IS_STALE,
    CAR_API_STALE_THRESHOLD_S,
    CAR_STALE_MODE_AUTO,
    CAR_STALE_MODE_FORCE_NOT_STALE,
    CAR_STALE_MODE_FORCE_STALE,
    SELECT_CAR_STALE_MODE,
)


# ── Task 1: Constants and Sensor Classification ──────────────────────────


class TestConstants:
    """Verify all staleness constants exist with correct values."""

    def test_car_api_stale_threshold(self):
        assert CAR_API_STALE_THRESHOLD_S == 4 * 3600

    def test_binary_sensor_car_api_ok(self):
        assert BINARY_SENSOR_CAR_API_OK == "qs_car_api_ok"

    def test_binary_sensor_car_is_stale(self):
        assert BINARY_SENSOR_CAR_IS_STALE == "qs_car_is_stale"

    def test_select_car_stale_mode(self):
        assert SELECT_CAR_STALE_MODE == "qs_car_stale_mode"

    def test_stale_mode_options(self):
        assert CAR_STALE_MODE_AUTO == "auto"
        assert CAR_STALE_MODE_FORCE_STALE == "force_stale"
        assert CAR_STALE_MODE_FORCE_NOT_STALE == "force_not_stale"


class TestSensorClassification:
    """Verify sensor tier classification on QSCar instances."""

    def test_critical_sensors_include_tracker_and_plugged(self, real_car):
        """Critical sensors are car_tracker and car_plugged."""
        critical = real_car._car_api_critical_sensors
        assert real_car.car_tracker in critical
        assert real_car.car_plugged in critical

    def test_supplementary_sensors_include_soc_odo_range(self, real_car):
        """Supplementary sensors are SOC, odometer, and estimated range."""
        supplementary = real_car._car_api_supplementary_sensors
        assert real_car.car_charge_percent_sensor in supplementary
        assert real_car.car_odometer_sensor in supplementary
        assert real_car.car_estimated_range_sensor in supplementary

    def test_all_sensors_list_has_no_none(self, real_car):
        """_car_api_all_sensors filters out None entries."""
        for sensor in real_car._car_api_all_sensors:
            assert sensor is not None

    def test_all_sensors_combines_both_tiers(self, real_car):
        """All sensors list combines critical and supplementary (minus Nones)."""
        expected_count = sum(
            1 for s in real_car._car_api_critical_sensors + real_car._car_api_supplementary_sensors if s is not None
        )
        assert len(real_car._car_api_all_sensors) == expected_count

    def test_initial_staleness_flags_are_false(self, real_car):
        """All staleness flags start as False."""
        assert real_car._car_api_stale is False
        assert real_car._was_car_api_stale is False
        assert real_car._car_api_stale_since is None
        assert real_car.car_stale_mode_override == CAR_STALE_MODE_AUTO
        assert real_car.car_api_stale_percent_mode is False
        assert real_car._car_api_inferred_home is False
        assert real_car._car_api_inferred_plugged is False

    def test_invited_car_has_sensor_lists(self, real_invited_car):
        """Invited cars still have sensor lists (may be empty if no sensors configured)."""
        assert hasattr(real_invited_car, "_car_api_all_sensors")
        assert hasattr(real_invited_car, "_car_api_critical_sensors")


# ── Task 2: Staleness Detection — Feature A ─────────────────────────────


class TestStalenessDetection:
    """Test is_car_api_stale() with various sensor freshness combinations."""

    def test_fresh_data_not_stale(self, real_car, current_time):
        """When all sensors have recent data, car is not stale."""
        # Set all sensors to have recent last_valid timestamps
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (current_time, "some_value", {})
        assert real_car.is_car_api_stale(current_time) is False

    def test_all_sensors_stale_triggers_stale(self, real_car, current_time):
        """When ALL sensors are older than threshold, car IS stale."""
        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (stale_time, "some_value", {})
        assert real_car.is_car_api_stale(current_time) is True

    def test_partial_stale_not_triggered(self, real_car, current_time):
        """When SOME sensors are stale but not all, car is NOT stale."""
        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)
        fresh_time = current_time - timedelta(seconds=60)
        sensors = real_car._car_api_all_sensors
        # Make first sensor fresh, rest stale
        real_car._entity_probed_last_valid_state[sensors[0]] = (fresh_time, "value", {})
        for sensor_id in sensors[1:]:
            real_car._entity_probed_last_valid_state[sensor_id] = (stale_time, "value", {})
        assert real_car.is_car_api_stale(current_time) is False

    def test_threshold_boundary_not_stale(self, real_car, current_time):
        """At exactly the threshold, car is NOT stale (uses > not >=)."""
        boundary_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (boundary_time, "value", {})
        assert real_car.is_car_api_stale(current_time) is False

    def test_threshold_boundary_plus_one_stale(self, real_car, current_time):
        """Just past the threshold, car IS stale."""
        boundary_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (boundary_time, "value", {})
        assert real_car.is_car_api_stale(current_time) is True

    def test_invited_car_never_stale(self, real_invited_car, current_time):
        """Invited cars always return False for staleness."""
        assert real_invited_car.is_car_api_stale(current_time) is False

    def test_no_sensors_not_stale(self, fake_hass, mock_data_handler):
        """Car with no API sensors configured is not stale."""
        from custom_components.quiet_solar.ha_model.car import QSCar

        config = {
            "name": "No Sensor Car",
            "device_type": "QSCar",
            "hass": fake_hass,
            "config_entry": MagicMock(),
            "data_handler": mock_data_handler,
            "home": mock_data_handler.home,
            "car_battery_capacity": 60000,
            "car_charger_min_charge": 6,
            "car_charger_max_charge": 32,
        }
        car = QSCar(**config)
        now = datetime.now(tz=pytz.UTC)
        assert car.is_car_api_stale(now) is False

    def test_sensor_never_updated_is_stale(self, real_car, current_time):
        """Sensors that have never reported (None last_valid) are treated as stale."""
        # Default state: _entity_probed_last_valid_state entries are None
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = None
        assert real_car.is_car_api_stale(current_time) is True

    def test_is_car_api_ok_is_inverse(self, real_car, current_time):
        """is_car_api_ok is the inverse of is_car_api_stale."""
        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})
        assert real_car.is_car_api_ok(current_time) is True

        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (stale_time, "value", {})
        assert real_car.is_car_api_ok(current_time) is False


class TestEffectiveStaleState:
    """Test is_car_effectively_stale() with select override combinations."""

    def test_auto_mode_fresh_api(self, real_car, current_time):
        """Auto mode + fresh API = not effectively stale."""
        real_car.car_stale_mode_override = CAR_STALE_MODE_AUTO
        real_car._car_api_stale = False
        assert real_car.is_car_effectively_stale(current_time) is False

    def test_auto_mode_stale_api(self, real_car, current_time):
        """Auto mode + stale API = effectively stale."""
        real_car.car_stale_mode_override = CAR_STALE_MODE_AUTO
        real_car._car_api_stale = True
        assert real_car.is_car_effectively_stale(current_time) is True

    def test_force_stale_overrides_fresh_api(self, real_car, current_time):
        """Force Stale overrides fresh API data."""
        real_car.car_stale_mode_override = CAR_STALE_MODE_FORCE_STALE
        real_car._car_api_stale = False
        assert real_car.is_car_effectively_stale(current_time) is True

    def test_force_not_stale_overrides_stale_api(self, real_car, current_time):
        """Force Not Stale overrides stale API data."""
        real_car.car_stale_mode_override = CAR_STALE_MODE_FORCE_NOT_STALE
        real_car._car_api_stale = True
        assert real_car.is_car_effectively_stale(current_time) is False


class TestStalenessTransitions:
    """Test _update_car_api_staleness() transition detection."""

    def test_fresh_to_stale_transition(self, real_car, current_time):
        """Transition from fresh to stale sets stale flags and stale_since."""
        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (stale_time, "value", {})

        real_car._update_car_api_staleness(current_time)

        assert real_car._car_api_stale is True
        assert real_car._was_car_api_stale is True
        assert real_car._car_api_stale_since == current_time
        assert real_car.car_api_stale_percent_mode is True

    def test_stale_to_fresh_transition(self, real_car, current_time):
        """Transition from stale to fresh clears all stale flags."""
        # Setup: car is stale
        real_car._car_api_stale = True
        real_car._was_car_api_stale = True
        real_car._car_api_stale_since = current_time - timedelta(hours=2)
        real_car.car_api_stale_percent_mode = True

        # Make sensors fresh
        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})

        real_car._update_car_api_staleness(current_time)

        assert real_car._car_api_stale is False
        assert real_car._was_car_api_stale is False
        assert real_car.car_api_stale_percent_mode is False
        assert real_car._car_api_stale_since is None

    def test_already_stale_no_repeated_transition(self, real_car, current_time):
        """If already stale, repeated checks don't reset stale_since."""
        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (stale_time, "value", {})

        original_since = current_time - timedelta(hours=1)
        real_car._car_api_stale = True
        real_car._was_car_api_stale = True
        real_car._car_api_stale_since = original_since

        real_car._update_car_api_staleness(current_time)

        assert real_car._car_api_stale_since == original_since

    def test_car_api_stale_percent_mode_is_public(self, real_car):
        """car_api_stale_percent_mode is a public attribute."""
        assert real_car.car_api_stale_percent_mode is False
        real_car.car_api_stale_percent_mode = True
        assert real_car.car_api_stale_percent_mode is True

    async def test_update_states_calls_staleness_check(self, real_car, current_time):
        """update_states drives _update_car_api_staleness each cycle."""
        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (stale_time, "value", {})

        # Before update_states, car is not stale
        assert real_car._car_api_stale is False

        # Mock super().update_states to avoid full device cycle in test
        with patch(
            "custom_components.quiet_solar.ha_model.device.HADeviceMixin.update_states",
            return_value=None,
        ):
            await real_car.update_states(current_time)

        # After update_states, staleness was detected
        assert real_car._car_api_stale is True
        assert real_car._was_car_api_stale is True

    def test_stale_to_fresh_clears_inferred_flags_without_percent_mode(self, real_car, current_time):
        """Stale->fresh transition clears inferred flags even when not in stale-percent mode (F2)."""
        # Setup: stale but NOT in stale-percent mode
        real_car._car_api_stale = True
        real_car._was_car_api_stale = True
        real_car._car_api_stale_since = current_time - timedelta(hours=2)
        real_car.car_api_stale_percent_mode = False
        real_car._car_api_inferred_home = True
        real_car._car_api_inferred_plugged = True

        # Make sensors fresh
        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})

        real_car._update_car_api_staleness(current_time)

        # All flags should be cleared
        assert real_car._car_api_inferred_home is False
        assert real_car._car_api_inferred_plugged is False
        assert real_car._car_api_stale is False
        assert real_car._car_api_stale_since is None

    def test_no_double_exit_on_percent_recovery(self, real_car, current_time):
        """Recovery via can_exit_stale_percent_mode doesn't trigger duplicate transition (F3)."""
        # Setup: stale with percent mode, then sensors recover
        real_car._car_api_stale = True
        real_car._was_car_api_stale = True
        real_car.car_api_stale_percent_mode = True
        real_car._car_api_stale_since = current_time - timedelta(hours=2)

        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "on", {})

        # Track notification count
        real_car.hass = MagicMock()
        real_car.home = MagicMock()

        real_car._update_car_api_staleness(current_time)

        # Only one notification (recovery), not two
        assert real_car.hass.async_create_task.call_count == 1
        assert real_car._car_api_stale is False
        assert real_car._was_car_api_stale is False


# ── Task 3: Contradiction Detection — Feature B ─────────────────────────


class TestContradictionDetection:
    """Test manual assignment contradiction detection."""

    def test_contradiction_when_api_says_not_home(self, real_car, current_time):
        """Manual assign while API says not_home triggers immediate stale."""
        # API reports car is away
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (current_time, "not_home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (current_time, "on", {})

        real_car.check_manual_assignment_contradiction("Test Charger", current_time)

        assert real_car._car_api_stale is True
        assert real_car._car_api_inferred_home is True
        assert real_car._car_api_inferred_plugged is True
        assert real_car.car_api_stale_percent_mode is True
        assert real_car._was_car_api_stale is True

    def test_contradiction_when_api_says_not_plugged(self, real_car, current_time):
        """Manual assign while API says not_plugged triggers immediate stale."""
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (current_time, "home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (current_time, "off", {})

        real_car.check_manual_assignment_contradiction("Test Charger", current_time)

        assert real_car._car_api_stale is True
        assert real_car._car_api_inferred_home is True
        assert real_car._car_api_inferred_plugged is True

    def test_no_contradiction_when_api_agrees(self, real_car, current_time):
        """Manual assign with API agreeing does NOT trigger stale."""
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (current_time, "home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (current_time, "on", {})

        real_car.check_manual_assignment_contradiction("Test Charger", current_time)

        assert real_car._car_api_stale is False
        assert real_car._car_api_inferred_home is False
        assert real_car._car_api_inferred_plugged is False

    def test_contradiction_skipped_when_force_not_stale(self, real_car, current_time):
        """Force Not Stale mode skips contradiction detection."""
        real_car.car_stale_mode_override = CAR_STALE_MODE_FORCE_NOT_STALE
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (current_time, "not_home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (current_time, "off", {})

        real_car.check_manual_assignment_contradiction("Test Charger", current_time)

        assert real_car._car_api_stale is False
        assert real_car._car_api_inferred_home is False

    def test_stale_since_set_on_contradiction(self, real_car, current_time):
        """Contradiction sets stale_since timestamp."""
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (current_time, "not_home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (current_time, "on", {})

        real_car.check_manual_assignment_contradiction("Test Charger", current_time)

        assert real_car._car_api_stale_since == current_time


class TestInferredFlagOverrides:
    """Test that inferred flags override is_car_home/is_car_plugged."""

    def test_is_car_home_returns_true_when_inferred(self, real_car, current_time):
        """is_car_home returns True when _car_api_inferred_home is set."""
        real_car._car_api_inferred_home = True
        # API says not_home but inferred overrides
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (current_time, "not_home", {})
        assert real_car.is_car_home(current_time) is True

    def test_is_car_plugged_returns_true_when_inferred(self, real_car, current_time):
        """is_car_plugged returns True when _car_api_inferred_plugged is set."""
        real_car._car_api_inferred_plugged = True
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (current_time, "off", {})
        assert real_car.is_car_plugged(current_time) is True

    def test_raw_home_ignores_inferred(self, real_car, current_time):
        """_get_raw_is_car_home reads the actual API, not inferred."""
        real_car._car_api_inferred_home = True
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (current_time, "not_home", {})
        assert real_car._get_raw_is_car_home(current_time) is False

    def test_raw_plugged_ignores_inferred(self, real_car, current_time):
        """_get_raw_is_car_plugged reads the actual API, not inferred."""
        real_car._car_api_inferred_plugged = True
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (current_time, "off", {})
        assert real_car._get_raw_is_car_plugged(current_time) is False

    def test_clear_inferred_flags_on_detach(self, real_car):
        """clear_inferred_flags clears both flags."""
        real_car._car_api_inferred_home = True
        real_car._car_api_inferred_plugged = True
        real_car.clear_inferred_flags()
        assert real_car._car_api_inferred_home is False
        assert real_car._car_api_inferred_plugged is False

    def test_is_car_home_no_inferred_reads_api(self, real_car, current_time):
        """Without inferred flag, is_car_home reads the actual API."""
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (current_time, "home", {})
        assert real_car.is_car_home(current_time) is True

        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (current_time, "not_home", {})
        assert real_car.is_car_home(current_time) is False


# ── Task 4: Stale-Percent Mode ──────────────────────────────────────────


class TestStalePercentMode:
    """Test SOC bypass and percent mode override in stale mode."""

    def test_get_car_charge_percent_returns_none_when_stale(self, real_car, current_time):
        """SOC sensor is bypassed (returns None) when in stale-percent mode."""
        # Setup: sensor has a value
        real_car._entity_probed_last_valid_state[real_car.car_charge_percent_sensor] = (current_time, 75.0, {})
        assert real_car.get_car_charge_percent(current_time) == 75.0

        # Enable stale-percent mode
        real_car.car_api_stale_percent_mode = True
        assert real_car.get_car_charge_percent(current_time) is None

    def test_get_car_charge_percent_works_when_not_stale(self, real_car, current_time):
        """SOC sensor works normally when not in stale mode."""
        real_car._entity_probed_last_valid_state[real_car.car_charge_percent_sensor] = (current_time, 50.0, {})
        real_car.car_api_stale_percent_mode = False
        assert real_car.get_car_charge_percent(current_time) == 50.0

    def test_can_use_charge_percent_constraints_true_in_stale_mode(self, real_car):
        """can_use_charge_percent_constraints returns True in stale-percent mode."""
        real_car.car_api_stale_percent_mode = True
        assert real_car.can_use_charge_percent_constraints() is True

    def test_can_use_charge_percent_constraints_respects_normal_logic(self, real_car):
        """Without stale mode, normal logic applies."""
        real_car.car_api_stale_percent_mode = False
        real_car._use_percent_mode = False
        assert real_car.can_use_charge_percent_constraints() is False

    def test_percent_mode_sensor_returns_on_when_stale(self, real_car, current_time):
        """car_use_percent_mode_sensor_state_getter returns 'on' in stale-percent mode."""
        real_car.car_api_stale_percent_mode = True
        result = real_car.car_use_percent_mode_sensor_state_getter("test_entity", current_time)
        assert result[1] == "on"
        assert real_car._use_percent_mode is True

    def test_percent_mode_sensor_normal_when_not_stale(self, real_car, current_time):
        """car_use_percent_mode_sensor_state_getter follows normal logic when not stale."""
        real_car.car_api_stale_percent_mode = False
        # No sensor data — should return "off"
        real_car._entity_probed_last_valid_state[real_car.car_charge_percent_sensor] = None
        result = real_car.car_use_percent_mode_sensor_state_getter("test_entity", current_time)
        assert result[1] == "off"

    def test_stale_percent_mode_activated_on_stale_transition(self, real_car, current_time):
        """When car goes stale and can use percent, stale-percent mode activates."""
        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (stale_time, "value", {})

        real_car._update_car_api_staleness(current_time)

        assert real_car.car_api_stale_percent_mode is True

    def test_get_car_charge_energy_returns_none_in_stale_mode(self, real_car, current_time):
        """get_car_charge_energy also returns None since it calls get_car_charge_percent."""
        real_car.car_api_stale_percent_mode = True
        assert real_car.get_car_charge_energy(current_time) is None

    def test_exit_stale_mode_clears_stale_percent_mode(self, real_car):
        """_exit_stale_mode clears the stale-percent mode flag."""
        real_car.car_api_stale_percent_mode = True
        real_car._exit_stale_mode()
        assert real_car.car_api_stale_percent_mode is False


# ── Task 5: Recovery Logic ──────────────────────────────────────────────


class TestRecoveryLogic:
    """Test can_exit_stale_percent_mode() and recovery behavior."""

    def _setup_stale_car(self, car, current_time):
        """Helper to put car into stale-percent mode."""
        car._car_api_stale = True
        car._was_car_api_stale = True
        car.car_api_stale_percent_mode = True
        car._car_api_stale_since = current_time - timedelta(hours=2)

    def test_cannot_exit_if_not_in_stale_mode(self, real_car, current_time):
        """No exit if car isn't in stale-percent mode."""
        assert real_car.can_exit_stale_percent_mode(current_time) is False

    def test_force_stale_blocks_recovery(self, real_car, current_time):
        """Force Stale select blocks recovery even with fresh data."""
        self._setup_stale_car(real_car, current_time)
        real_car.car_stale_mode_override = CAR_STALE_MODE_FORCE_STALE
        # Make sensors fresh
        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is False

    def test_force_not_stale_allows_immediate_exit(self, real_car, current_time):
        """Force Not Stale allows immediate recovery."""
        self._setup_stale_car(real_car, current_time)
        real_car.car_stale_mode_override = CAR_STALE_MODE_FORCE_NOT_STALE
        assert real_car.can_exit_stale_percent_mode(current_time) is True

    def test_auto_mode_sensors_still_stale_blocks_exit(self, real_car, current_time):
        """Cannot exit if sensors are still stale in auto mode."""
        self._setup_stale_car(real_car, current_time)
        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (stale_time, "value", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is False

    def test_odometer_only_update_does_not_clear_stale(self, real_car, current_time):
        """Odometer-only recovery doesn't clear stale (not a critical sensor)."""
        self._setup_stale_car(real_car, current_time)
        fresh_time = current_time - timedelta(seconds=60)
        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)

        # Make all sensors fresh (to pass is_car_api_stale check)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})

        # But critical sensors don't confirm physical state (not "home", not "on")
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "not_home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})

        assert real_car.can_exit_stale_percent_mode(current_time) is False

    def test_home_tracker_recovery_clears_stale(self, real_car, current_time):
        """Home tracker reporting 'home' clears stale (when no inferred plug)."""
        self._setup_stale_car(real_car, current_time)
        fresh_time = current_time - timedelta(seconds=60)

        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})

        assert real_car.can_exit_stale_percent_mode(current_time) is True

    def test_plug_required_for_inferred_plug_recovery(self, real_car, current_time):
        """When car was manually assigned (inferred_plugged), plug sensor must recover."""
        self._setup_stale_car(real_car, current_time)
        real_car._car_api_inferred_plugged = True
        fresh_time = current_time - timedelta(seconds=60)

        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})

        # Home is home but plugged is off — not enough for inferred-plug recovery
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is False

        # Plug sensor now reports "on" — recovery allowed
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "on", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is True

    def test_recovery_via_update_clears_all_flags(self, real_car, current_time):
        """Full recovery cycle through _update_car_api_staleness clears everything."""
        self._setup_stale_car(real_car, current_time)
        fresh_time = current_time - timedelta(seconds=60)

        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "on", {})

        real_car._update_car_api_staleness(current_time)

        assert real_car._car_api_stale is False
        assert real_car.car_api_stale_percent_mode is False
        assert real_car._car_api_inferred_home is False
        assert real_car._car_api_inferred_plugged is False
        assert real_car._was_car_api_stale is False

    def test_recovery_uses_raw_sensors_not_inferred(self, real_car, current_time):
        """Recovery check reads raw API values, not the inferred overrides."""
        self._setup_stale_car(real_car, current_time)
        real_car._car_api_inferred_home = True
        real_car._car_api_inferred_plugged = True
        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)

        # All sensors stale — raw API is still stale
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (stale_time, "value", {})

        # Even though inferred flags are True, can't exit because raw sensors are stale
        assert real_car.can_exit_stale_percent_mode(current_time) is False


# ── Task 6: Select + Binary Sensor Entities ─────────────────────────────


class TestStaleSelectEntity:
    """Test stale mode select entity creation and behavior."""

    def test_select_entity_created(self, real_car_with_home, mock_data_handler):
        """select.py creates a stale mode select for QSCar."""
        from custom_components.quiet_solar.select import create_ha_select_for_QSCar

        entities = create_ha_select_for_QSCar(real_car_with_home)
        keys = [e.entity_description.key for e in entities]
        assert SELECT_CAR_STALE_MODE in keys

    def test_select_options(self, real_car_with_home, mock_data_handler):
        """Stale mode select has correct options."""
        from custom_components.quiet_solar.select import create_ha_select_for_QSCar

        entities = create_ha_select_for_QSCar(real_car_with_home)
        stale_select = next(e for e in entities if e.entity_description.key == SELECT_CAR_STALE_MODE)
        assert stale_select.entity_description.options == [
            CAR_STALE_MODE_AUTO,
            CAR_STALE_MODE_FORCE_STALE,
            CAR_STALE_MODE_FORCE_NOT_STALE,
        ]

    def test_select_default_is_auto(self, real_car_with_home, mock_data_handler):
        """Stale mode select defaults to 'auto'."""
        from custom_components.quiet_solar.select import create_ha_select_for_QSCar

        entities = create_ha_select_for_QSCar(real_car_with_home)
        stale_select = next(e for e in entities if e.entity_description.key == SELECT_CAR_STALE_MODE)
        assert stale_select.entity_description.qs_default_option == CAR_STALE_MODE_AUTO

    async def test_user_set_stale_mode_updates_override(self, real_car, current_time):
        """user_set_stale_mode updates the car_stale_mode_override."""
        await real_car.user_set_stale_mode(CAR_STALE_MODE_FORCE_STALE)
        assert real_car.car_stale_mode_override == CAR_STALE_MODE_FORCE_STALE

    async def test_user_set_stale_mode_for_init_skips_reevaluate(self, real_car, current_time):
        """user_set_stale_mode with for_init=True skips staleness re-evaluation."""
        real_car._car_api_stale = False
        await real_car.user_set_stale_mode(CAR_STALE_MODE_FORCE_STALE, for_init=True)
        assert real_car.car_stale_mode_override == CAR_STALE_MODE_FORCE_STALE
        # for_init doesn't trigger _update_car_api_staleness, so _was_car_api_stale stays False
        assert real_car._was_car_api_stale is False


class TestNotifications:
    """Test notification scheduling for stale transitions."""

    def test_stale_transition_schedules_notification(self, real_car, current_time):
        """Going stale schedules a notification."""
        # Setup hass with async_create_task
        real_car.hass = MagicMock()
        real_car.home = MagicMock()

        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (stale_time, "value", {})

        real_car._update_car_api_staleness(current_time)

        real_car.hass.async_create_task.assert_called_once()

    def test_contradiction_schedules_notification(self, real_car, current_time):
        """Contradiction detection schedules a notification."""
        real_car.hass = MagicMock()
        real_car.home = MagicMock()
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (current_time, "not_home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (current_time, "on", {})

        real_car.check_manual_assignment_contradiction("Test Charger", current_time)

        real_car.hass.async_create_task.assert_called_once()

    def test_no_notification_when_no_hass(self, real_car, current_time):
        """No crash when hass is None during notification."""
        real_car.hass = None
        real_car.home = MagicMock()
        # Should not raise
        real_car._schedule_notification("title", "message")


class TestBinarySensorEntities:
    """Test binary sensor entity creation."""

    def test_api_ok_binary_sensor_created(self, real_car, mock_data_handler):
        """binary_sensor.py creates api_ok sensor for QSCar."""
        from custom_components.quiet_solar.binary_sensor import create_ha_binary_sensor_for_QSCar

        entities = create_ha_binary_sensor_for_QSCar(real_car)
        keys = [e.entity_description.key for e in entities]
        assert "qs_car_api_ok" in keys

    def test_is_stale_binary_sensor_created(self, real_car, mock_data_handler):
        """binary_sensor.py creates is_stale sensor for QSCar."""
        from custom_components.quiet_solar.binary_sensor import create_ha_binary_sensor_for_QSCar

        entities = create_ha_binary_sensor_for_QSCar(real_car)
        keys = [e.entity_description.key for e in entities]
        assert "qs_car_is_stale" in keys

    def test_api_ok_sensor_device_class(self, real_car, mock_data_handler):
        """api_ok binary sensor has connectivity device class."""
        from homeassistant.components.binary_sensor import BinarySensorDeviceClass

        from custom_components.quiet_solar.binary_sensor import create_ha_binary_sensor_for_QSCar

        entities = create_ha_binary_sensor_for_QSCar(real_car)
        api_ok = next(e for e in entities if e.entity_description.key == "qs_car_api_ok")
        assert api_ok.entity_description.device_class == BinarySensorDeviceClass.CONNECTIVITY


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def real_car(fake_hass, mock_data_handler, current_time):
    """Create a real QSCar instance with standard config."""
    from tests.ha_tests.const import MOCK_CAR_CONFIG

    from custom_components.quiet_solar.ha_model.car import QSCar

    config = {
        **MOCK_CAR_CONFIG,
        "hass": fake_hass,
        "config_entry": MagicMock(),
        "data_handler": mock_data_handler,
        "home": mock_data_handler.home,
        "car_odometer_sensor": "sensor.test_car_odometer",
        "car_estimated_range_sensor": "sensor.test_car_range",
    }
    car = QSCar(**config)
    return car


@pytest.fixture
def real_invited_car(fake_hass, mock_data_handler, current_time):
    """Create a real QSCar instance configured as invited."""
    from tests.ha_tests.const import MOCK_CAR_CONFIG

    from custom_components.quiet_solar.ha_model.car import QSCar

    config = {
        **MOCK_CAR_CONFIG,
        "hass": fake_hass,
        "config_entry": MagicMock(),
        "data_handler": mock_data_handler,
        "home": mock_data_handler.home,
        "car_is_invited": True,
    }
    car = QSCar(**config)
    return car


@pytest.fixture
def real_car_with_home(fake_hass, mock_data_handler, current_time):
    """Create a real QSCar with a home that has _chargers and _cars."""
    from tests.ha_tests.const import MOCK_CAR_CONFIG

    from custom_components.quiet_solar.ha_model.car import QSCar

    home_mock = MagicMock()
    home_mock._chargers = []
    home_mock._cars = []
    home_mock.get_person_by_name = MagicMock(return_value=None)

    config = {
        **MOCK_CAR_CONFIG,
        "hass": fake_hass,
        "config_entry": MagicMock(),
        "data_handler": mock_data_handler,
        "home": home_mock,
        "car_odometer_sensor": "sensor.test_car_odometer",
        "car_estimated_range_sensor": "sensor.test_car_range",
    }
    car = QSCar(**config)
    home_mock._cars.append(car)
    return car
