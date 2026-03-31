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
from unittest.mock import AsyncMock, MagicMock, patch

import pytz
import pytest

from custom_components.quiet_solar.const import (
    BINARY_SENSOR_CAR_API_OK,
    BINARY_SENSOR_CAR_IS_STALE,
    CAR_API_STALE_THRESHOLD_S,
    CAR_NOT_HOME_AUTO_RESET_S,
    CAR_SOC_STALE_THRESHOLD_S,
    CAR_STALE_MODE_AUTO,
    CAR_STALE_MODE_FORCE_NOT_STALE,
    CAR_STALE_MODE_FORCE_STALE,
    FORCE_CAR_NO_CHARGER_CONNECTED,
    SELECT_CAR_STALE_MODE,
)


# ── Task 1: Constants and Sensor Classification ──────────────────────────


class TestConstants:
    """Verify all staleness constants exist with correct values."""

    def test_car_api_stale_threshold(self):
        assert CAR_API_STALE_THRESHOLD_S == 6 * 3600

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
        # Setup: stale with percent mode, connected to charger
        real_car._car_api_stale = True
        real_car._was_car_api_stale = True
        real_car.car_api_stale_percent_mode = True
        real_car._car_api_stale_since = current_time - timedelta(hours=2)
        real_car.charger = MagicMock(name="Test Charger")

        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "on", {})

        # Track notification count
        real_car.hass = MagicMock()
        real_car.home = MagicMock()
        real_car.current_forecasted_person = MagicMock()

        real_car._update_car_api_staleness(current_time)

        # Only one notification (recovery), not two
        assert real_car.hass.async_create_task.call_count == 1
        assert real_car._car_api_stale is False
        assert real_car._was_car_api_stale is False

    def test_stale_sensor_details_never_updated(self, real_car, current_time):
        """_get_stale_sensor_details reports 'never updated' for sensors with no data."""
        # Clear all sensor data
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = None
        result = real_car._get_stale_sensor_details(current_time)
        assert "never updated" in result


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

    def test_contradiction_skipped_when_already_stale(self, real_car, current_time):
        """Already-stale car skips contradiction check to avoid duplicate notifications."""
        real_car._car_api_stale = True
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (current_time, "not_home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (current_time, "off", {})

        real_car.hass = MagicMock()
        real_car.home = MagicMock()
        real_car.check_manual_assignment_contradiction("Test Charger", current_time)

        # No notification sent — early return
        real_car.hass.async_create_task.assert_not_called()

    def test_contradiction_skipped_when_stale_percent_mode(self, real_car, current_time):
        """Stale-percent mode skips contradiction check."""
        real_car.car_api_stale_percent_mode = True
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (current_time, "not_home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (current_time, "off", {})

        real_car.hass = MagicMock()
        real_car.home = MagicMock()
        real_car.check_manual_assignment_contradiction("Test Charger", current_time)

        real_car.hass.async_create_task.assert_not_called()


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

    def test_charger_detach_car_clears_inferred_flags(self, real_car):
        """charger.detach_car() clears inferred flags on the car."""
        from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric

        real_car._car_api_inferred_home = True
        real_car._car_api_inferred_plugged = True

        charger = MagicMock(spec=QSChargerGeneric)
        charger.car = real_car
        # Call the real detach_car logic
        QSChargerGeneric.detach_car(charger)

        assert real_car._car_api_inferred_home is False
        assert real_car._car_api_inferred_plugged is False
        assert real_car.charger is None

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

    def test_can_use_charge_percent_constraints_always_static(self, real_car):
        """can_use_charge_percent_constraints always delegates to static check."""
        # Non-invited car with valid config → True regardless of stale mode
        real_car.car_api_stale_percent_mode = True
        assert real_car.can_use_charge_percent_constraints() is True
        real_car.car_api_stale_percent_mode = False
        assert real_car.can_use_charge_percent_constraints() is True
        # Invited car: static check fails even when stale
        real_car.car_is_invited = True
        real_car.car_api_stale_percent_mode = True
        assert real_car.can_use_charge_percent_constraints() is False

    def test_is_soc_sensor_stale_fresh(self, real_car, current_time):
        """SOC sensor with recent data is not stale."""
        real_car._entity_probed_last_valid_state[real_car.car_charge_percent_sensor] = (
            current_time - timedelta(seconds=60),
            50.0,
            {},
        )
        assert real_car._is_soc_sensor_stale(current_time) is False

    def test_is_soc_sensor_stale_old(self, real_car, current_time):
        """SOC sensor older than threshold is stale."""
        real_car._entity_probed_last_valid_state[real_car.car_charge_percent_sensor] = (
            current_time - timedelta(seconds=CAR_SOC_STALE_THRESHOLD_S + 1),
            50.0,
            {},
        )
        assert real_car._is_soc_sensor_stale(current_time) is True

    def test_is_soc_sensor_stale_no_data(self, real_car, current_time):
        """SOC sensor with no data is stale."""
        real_car._entity_probed_last_valid_state[real_car.car_charge_percent_sensor] = None
        assert real_car._is_soc_sensor_stale(current_time) is True

    def test_is_soc_sensor_stale_invited_car(self, real_invited_car, current_time):
        """Invited car is never SOC-stale."""
        assert real_invited_car._is_soc_sensor_stale(current_time) is False

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

    def test_connected_car_needs_home_and_plugged(self, real_car, current_time):
        """Connected car can't exit if home or plug sensors don't confirm."""
        self._setup_stale_car(real_car, current_time)
        real_car.charger = MagicMock(name="Test Charger")
        fresh_time = current_time - timedelta(seconds=60)

        # Make all sensors fresh
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})

        # Plugged but not home — blocked
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "not_home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "on", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is False

        # Home but not plugged — blocked
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is False

    def test_not_connected_plug_off_allows_exit(self, real_car, current_time):
        """Not-connected car with plug=off and SOC fresh can exit stale."""
        self._setup_stale_car(real_car, current_time)
        fresh_time = current_time - timedelta(seconds=60)

        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "not_home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})

        assert real_car.can_exit_stale_percent_mode(current_time) is True

    def test_connected_car_plug_on_home_home_allows_exit(self, real_car, current_time):
        """Connected car exits stale when plug=on and home=home."""
        self._setup_stale_car(real_car, current_time)
        real_car.charger = MagicMock(name="Test Charger")
        fresh_time = current_time - timedelta(seconds=60)

        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})

        # Home is home but plugged is off — blocked (connected path needs both)
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is False

        # Plug sensor now reports "on" — recovery allowed
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "on", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is True

    def test_recovery_via_update_clears_all_flags(self, real_car, current_time):
        """Full recovery cycle through _update_car_api_staleness clears everything."""
        self._setup_stale_car(real_car, current_time)
        real_car.charger = MagicMock(name="Test Charger")
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


# ── SOC-only Staleness ──────────────────────────────────────────────────


class TestSocOnlyStaleness:
    """Test SOC-only stale entry, recovery, and edge cases."""

    def _make_soc_only_stale(self, car, current_time):
        """Make only the SOC sensor stale, keep other sensors fresh."""
        fresh_time = current_time - timedelta(seconds=60)
        soc_stale_time = current_time - timedelta(seconds=CAR_SOC_STALE_THRESHOLD_S + 1)

        for sensor_id in car._car_api_all_sensors:
            car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})
        car._entity_probed_last_valid_state[car.car_charge_percent_sensor] = (soc_stale_time, 50.0, {})

    def test_soc_only_stale_enters_stale_percent_mode(self, real_car, current_time):
        """When only SOC sensor is stale, car enters stale-percent mode."""
        self._make_soc_only_stale(real_car, current_time)
        real_car._update_car_api_staleness(current_time)

        assert real_car.car_api_stale_percent_mode is True
        assert real_car._car_api_stale is False  # not full stale
        assert real_car.is_car_effectively_stale(current_time) is True

    def test_is_car_charge_growing_returns_none_when_stale(self, real_car, current_time):
        """is_car_charge_growing returns None when in stale-percent mode."""
        real_car.car_api_stale_percent_mode = True
        result = real_car.is_car_charge_growing(300.0, current_time)
        assert result is None

    def test_soc_only_stale_sends_notification(self, real_car, current_time):
        """SOC-only stale triggers a notification mentioning SOC sensor."""
        real_car.hass = MagicMock()
        real_car.home = MagicMock()
        real_car.current_forecasted_person = MagicMock()
        self._make_soc_only_stale(real_car, current_time)
        real_car._update_car_api_staleness(current_time)

        real_car.hass.async_create_task.assert_called_once()

    def test_soc_only_recovery_when_soc_refreshes(self, real_car, current_time):
        """SOC-only stale recovers when SOC sensor becomes fresh and plug confirms unplugged."""
        self._make_soc_only_stale(real_car, current_time)
        real_car._update_car_api_staleness(current_time)
        assert real_car.car_api_stale_percent_mode is True

        # SOC sensor becomes fresh, plug confirms unplugged (not-connected path)
        fresh_time = current_time - timedelta(seconds=60)
        real_car._entity_probed_last_valid_state[real_car.car_charge_percent_sensor] = (fresh_time, 55.0, {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})
        real_car._update_car_api_staleness(current_time)

        assert real_car.car_api_stale_percent_mode is False
        assert real_car._was_car_api_stale is False

    def test_soc_only_recovery_not_connected_plug_off(self, real_car, current_time):
        """SOC-only stale + not-connected + plug=off + SOC fresh → exit allowed."""
        self._make_soc_only_stale(real_car, current_time)
        real_car._update_car_api_staleness(current_time)

        # Make SOC fresh, car is not home and not plugged (not-connected path)
        fresh_time = current_time - timedelta(seconds=60)
        real_car._entity_probed_last_valid_state[real_car.car_charge_percent_sensor] = (fresh_time, 55.0, {})
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "not_home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})

        assert real_car.can_exit_stale_percent_mode(current_time) is True

    def test_soc_only_stale_not_for_invited_cars(self, real_invited_car, current_time):
        """Invited cars never enter SOC-only stale mode."""
        real_invited_car._update_car_api_staleness(current_time)
        assert real_invited_car.car_api_stale_percent_mode is False

    def test_soc_only_stale_escalates_to_full_stale(self, real_car, current_time):
        """SOC-only stale escalates to full stale when all sensors go stale."""
        self._make_soc_only_stale(real_car, current_time)
        real_car._update_car_api_staleness(current_time)
        assert real_car.car_api_stale_percent_mode is True
        assert real_car._car_api_stale is False

        # Now all sensors go stale
        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (stale_time, "value", {})
        real_car._update_car_api_staleness(current_time)

        assert real_car._car_api_stale is True
        assert real_car.car_api_stale_percent_mode is True

    def test_full_stale_degraded_blocks_recovery_while_soc_stale(self, real_car, current_time):
        """Full stale with non-SOC recovery blocks exit while SOC is still stale."""
        # Enter full stale
        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (stale_time, "value", {})
        real_car._update_car_api_staleness(current_time)
        assert real_car._car_api_stale is True
        assert real_car.car_api_stale_percent_mode is True

        # Connect to charger for connected exit path
        real_car.charger = MagicMock(name="Test Charger")

        # Non-SOC sensors recover, but SOC stays stale
        fresh_time = current_time - timedelta(seconds=60)
        soc_stale_time = current_time - timedelta(seconds=CAR_SOC_STALE_THRESHOLD_S + 1)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})
        real_car._entity_probed_last_valid_state[real_car.car_charge_percent_sensor] = (soc_stale_time, 50.0, {})
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "on", {})

        real_car._update_car_api_staleness(current_time)

        # Still in stale-percent mode because SOC is stale
        assert real_car.car_api_stale_percent_mode is True

    def test_effectively_stale_includes_soc_only(self, real_car, current_time):
        """is_car_effectively_stale returns True for SOC-only stale."""
        real_car._car_api_stale = False
        real_car.car_api_stale_percent_mode = True
        assert real_car.is_car_effectively_stale(current_time) is True


# ── Context-Aware Exit Logic ───────────────────────────────────────────


class TestContextAwareExit:
    """Test can_exit_stale_percent_mode with charger-based branching."""

    def _setup_stale_car(self, car, current_time):
        """Helper to put car into stale-percent mode with fresh sensors."""
        car._car_api_stale = True
        car._was_car_api_stale = True
        car.car_api_stale_percent_mode = True
        car._car_api_stale_since = current_time - timedelta(hours=2)
        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in car._car_api_all_sensors:
            car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})

    def test_exit_connected_plugged_and_home(self, real_car, current_time):
        """Connected car + plug=on + home=home → exit."""
        self._setup_stale_car(real_car, current_time)
        real_car.charger = MagicMock(name="Test Charger")
        fresh_time = current_time - timedelta(seconds=60)
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "on", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is True

    def test_exit_connected_home_false_blocks(self, real_car, current_time):
        """Connected car + home≠home → blocked."""
        self._setup_stale_car(real_car, current_time)
        real_car.charger = MagicMock(name="Test Charger")
        fresh_time = current_time - timedelta(seconds=60)
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "not_home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "on", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is False

    def test_exit_connected_plug_off_blocks(self, real_car, current_time):
        """Connected car + plug=off → blocked."""
        self._setup_stale_car(real_car, current_time)
        real_car.charger = MagicMock(name="Test Charger")
        fresh_time = current_time - timedelta(seconds=60)
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is False

    def test_exit_not_connected_plug_false(self, real_car, current_time):
        """Not-connected car + plug=off + SOC fresh → exit."""
        self._setup_stale_car(real_car, current_time)
        fresh_time = current_time - timedelta(seconds=60)
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is True

    def test_exit_not_connected_plug_true_home_true_allows_exit(self, real_car, current_time):
        """Not-connected car + plug=on + home=home → exit (car at charger, not yet attached)."""
        self._setup_stale_car(real_car, current_time)
        fresh_time = current_time - timedelta(seconds=60)
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "on", {})
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is True

    def test_exit_not_connected_plug_true_not_home_blocks(self, real_car, current_time):
        """Not-connected car + plug=on + home≠home → blocked (might be plugged elsewhere)."""
        self._setup_stale_car(real_car, current_time)
        fresh_time = current_time - timedelta(seconds=60)
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "on", {})
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "not_home", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is False

    def test_exit_not_connected_plug_true_no_tracker_allows_exit(self, real_car, current_time):
        """Not-connected car + plug=on + no tracker → exit (can't verify home, trust plug)."""
        self._setup_stale_car(real_car, current_time)
        real_car.car_tracker = None
        fresh_time = current_time - timedelta(seconds=60)
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "on", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is True

    def test_exit_soc_stale_blocks_both_paths(self, real_car, current_time):
        """SOC >1h blocks exit regardless of path."""
        self._setup_stale_car(real_car, current_time)
        real_car.charger = MagicMock(name="Test Charger")
        fresh_time = current_time - timedelta(seconds=60)
        soc_stale_time = current_time - timedelta(seconds=CAR_SOC_STALE_THRESHOLD_S + 1)
        real_car._entity_probed_last_valid_state[real_car.car_charge_percent_sensor] = (soc_stale_time, 50.0, {})
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "on", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is False

        # Also blocks not-connected path
        real_car.charger = None
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is False

    def test_exit_all_sensors_must_be_available(self, real_car, current_time):
        """One sensor never updated → blocked."""
        self._setup_stale_car(real_car, current_time)
        # Clear one sensor's data
        real_car._entity_probed_last_valid_state[real_car.car_odometer_sensor] = None
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (
            current_time - timedelta(seconds=60), "off", {}
        )
        assert real_car.can_exit_stale_percent_mode(current_time) is False

    def test_exit_all_sensors_stale_blocks(self, real_car, current_time):
        """No sensor moved in CAR_API_STALE_THRESHOLD_S → blocked."""
        self._setup_stale_car(real_car, current_time)
        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (stale_time, "value", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is False

    def test_exit_no_plug_sensor_connected(self, real_car, current_time):
        """No plug sensor + connected → exit (plug check skipped)."""
        self._setup_stale_car(real_car, current_time)
        real_car.charger = MagicMock(name="Test Charger")
        real_car.car_plugged = None
        fresh_time = current_time - timedelta(seconds=60)
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is True

    def test_exit_no_plug_sensor_not_connected(self, real_car, current_time):
        """No plug sensor + not connected → exit (plug check skipped)."""
        self._setup_stale_car(real_car, current_time)
        real_car.car_plugged = None
        assert real_car.can_exit_stale_percent_mode(current_time) is True

    def test_exit_not_connected_plug_none_blocks(self, real_car, current_time):
        """Not-connected car + plug sensor exists but returns None → blocked."""
        self._setup_stale_car(real_car, current_time)
        fresh_time = current_time - timedelta(seconds=60)
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})
        # Sensor reported (passes common checks) but raw read returns None
        with patch.object(real_car, "_get_raw_is_car_plugged", return_value=None):
            assert real_car.can_exit_stale_percent_mode(current_time) is False

    def test_exit_no_home_sensor_connected(self, real_car, current_time):
        """No home sensor + connected + plug=on → exit (home check skipped)."""
        self._setup_stale_car(real_car, current_time)
        real_car.charger = MagicMock(name="Test Charger")
        real_car.car_tracker = None
        fresh_time = current_time - timedelta(seconds=60)
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "on", {})
        assert real_car.can_exit_stale_percent_mode(current_time) is True

    def test_exit_both_sensors_none_connected_blocks(self, real_car, current_time):
        """Connected car with plug=None + home=None from sensor → blocked."""
        self._setup_stale_car(real_car, current_time)
        real_car.charger = MagicMock(name="Test Charger")
        # Sensors exist but return None (unavailable)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (
                current_time - timedelta(seconds=60), "value", {}
            )
        # Make home and plug return None by having no valid value
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = None
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = None
        # _have_all_api_sensors_reported returns False → blocked
        assert real_car.can_exit_stale_percent_mode(current_time) is False


# ── Periodic Contradiction Check ───────────────────────────────────────


class TestPeriodicContradiction:
    """Test periodic contradiction check for attached cars."""

    def test_periodic_contradiction_attached_car(self, real_car, current_time):
        """Attached car with plug=off triggers stale via update cycle."""
        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})

        real_car.charger = MagicMock(name="Test Charger")
        real_car.charger.name = "Test Charger"
        real_car._update_car_api_staleness(current_time)

        assert real_car._car_api_stale is True
        assert real_car._car_api_inferred_plugged is True

    def test_periodic_contradiction_not_attached(self, real_car, current_time):
        """Not-attached car doesn't trigger contradiction check."""
        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})

        real_car._update_car_api_staleness(current_time)

        assert real_car._car_api_stale is False
        assert real_car._car_api_inferred_plugged is False

    def test_periodic_contradiction_already_stale_no_repeat(self, real_car, current_time):
        """Already-stale car doesn't re-trigger contradiction."""
        real_car._car_api_stale = True
        real_car._was_car_api_stale = True
        real_car.car_api_stale_percent_mode = True
        real_car.charger = MagicMock(name="Test Charger")

        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})

        real_car.hass = MagicMock()
        real_car.home = MagicMock()

        # Guard prevents check_manual_assignment_contradiction from being called
        with patch.object(real_car, "check_manual_assignment_contradiction") as mock_check:
            real_car._update_car_api_staleness(current_time)
            mock_check.assert_not_called()

    def test_periodic_contradiction_skipped_force_not_stale(self, real_car, current_time):
        """Force Not Stale skips periodic contradiction check."""
        real_car.charger = MagicMock(name="Test Charger")
        real_car.charger.name = "Test Charger"
        real_car.car_stale_mode_override = CAR_STALE_MODE_FORCE_NOT_STALE

        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (fresh_time, "off", {})

        real_car._update_car_api_staleness(current_time)
        assert real_car._car_api_stale is False


# ── _have_all_api_sensors_reported ─────────────────────────────────────


class TestAllApiSensorsAvailable:
    """Test _have_all_api_sensors_reported() helper."""

    def test_all_valid(self, real_car, current_time):
        """All sensors have valid readings → True."""
        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})
        assert real_car._have_all_api_sensors_reported(current_time) is True

    def test_one_missing(self, real_car, current_time):
        """One sensor has no valid reading → False."""
        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})
        real_car._entity_probed_last_valid_state[real_car.car_odometer_sensor] = None
        assert real_car._have_all_api_sensors_reported(current_time) is False

    def test_empty_sensor_list(self, real_invited_car, current_time):
        """No sensors tracked → True (vacuously)."""
        real_invited_car._car_api_all_sensors = []
        assert real_invited_car._have_all_api_sensors_reported(current_time) is True


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
        real_car.current_forecasted_person = MagicMock()

        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (stale_time, "value", {})

        real_car._update_car_api_staleness(current_time)

        real_car.hass.async_create_task.assert_called_once()

    def test_contradiction_schedules_notification(self, real_car, current_time):
        """Contradiction detection schedules a notification."""
        real_car.hass = MagicMock()
        real_car.home = MagicMock()
        real_car.current_forecasted_person = MagicMock()
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (current_time, "not_home", {})
        real_car._entity_probed_last_valid_state[real_car.car_plugged] = (current_time, "on", {})

        real_car.check_manual_assignment_contradiction("Test Charger", current_time)

        real_car.hass.async_create_task.assert_called_once()

    def test_no_notification_when_no_hass(self, real_car, current_time):
        """No crash when hass is None during notification."""
        real_car.hass = None
        real_car.current_forecasted_person = MagicMock()
        # Should not raise
        real_car._schedule_person_notification("title", "message")

    def test_no_notification_when_no_person(self, real_car, current_time):
        """No crash when person is None during notification."""
        real_car.hass = MagicMock()
        real_car.current_forecasted_person = None
        # Should not raise
        real_car._schedule_person_notification("title", "message")
        real_car.hass.async_create_task.assert_not_called()

    def test_force_stale_skips_notification(self, real_car, current_time):
        """User-initiated Force Stale does not send a notification."""
        real_car.hass = MagicMock()
        real_car.home = MagicMock()
        real_car.current_forecasted_person = MagicMock()
        real_car.car_stale_mode_override = CAR_STALE_MODE_FORCE_STALE

        stale_time = current_time - timedelta(seconds=CAR_API_STALE_THRESHOLD_S + 1)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (stale_time, "value", {})

        real_car._update_car_api_staleness(current_time)

        # Force Stale is user-initiated, no notification expected
        real_car.hass.async_create_task.assert_not_called()


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


# ── Scoring Instant-Check Fallback (Bug #92, Root Cause 2) ───────────


class TestScoringInstantCheckFallback:
    """Test get_car_score falls back to instant check when for_duration returns False."""

    def _make_charger(self):
        """Create a charger for scoring tests."""
        from tests.test_charger_additional_coverage import (
            create_charger_generic,
            create_mock_hass,
            create_mock_home,
        )

        hass = create_mock_hass()
        home = create_mock_home(hass)
        charger = create_charger_generic(hass, home)
        charger.car = None
        charger.car_attach_time = None
        return charger

    def test_plug_duration_false_instant_true_gives_reduced_score(self):
        """Car API just started reporting plugged (duration < 15s) → reduced score_plug_bump=2."""
        car = MagicMock()
        car.name = "TestCar"
        car.car_is_invited = False
        car.get_user_originated = MagicMock(return_value=None)
        # for_duration=15 returns False (just started), instant returns True
        car.is_car_plugged = MagicMock(side_effect=lambda time, for_duration=None: (
            False if for_duration == 15 else True
        ))
        car.is_car_home = MagicMock(return_value=True)
        car.get_continuous_plug_duration = MagicMock(return_value=5.0)
        car.get_car_coordinates = MagicMock(return_value=(None, None))

        charger = self._make_charger()
        now = datetime.now(pytz.UTC)
        score_instant = charger.get_car_score(car, now, {})
        assert score_instant > 0, f"Expected non-zero score from instant fallback, got {score_instant}"

        # Compare with full-duration score to verify reduced weight
        car2 = MagicMock()
        car2.name = "TestCar2"
        car2.car_is_invited = False
        car2.get_user_originated = MagicMock(return_value=None)
        car2.is_car_plugged = MagicMock(return_value=True)
        car2.is_car_home = MagicMock(return_value=True)
        car2.get_continuous_plug_duration = MagicMock(return_value=5.0)
        car2.get_car_coordinates = MagicMock(return_value=(None, None))
        score_full = charger.get_car_score(car2, now, {})
        assert score_instant < score_full, (
            f"Instant fallback score ({score_instant}) should be less than full score ({score_full})"
        )

    def test_plug_duration_true_gives_full_score(self):
        """Car API reports plugged for > 15s → full score_plug_bump=5."""
        car = MagicMock()
        car.name = "TestCar"
        car.car_is_invited = False
        car.get_user_originated = MagicMock(return_value=None)
        car.is_car_plugged = MagicMock(return_value=True)
        car.is_car_home = MagicMock(return_value=True)
        car.get_continuous_plug_duration = MagicMock(return_value=120.0)
        car.get_car_coordinates = MagicMock(return_value=(None, None))

        charger = self._make_charger()
        now = datetime.now(pytz.UTC)
        score = charger.get_car_score(car, now, {})
        assert score > 0

    def test_home_duration_false_instant_true_gives_score(self):
        """Car home sensor just started (duration < 15s) → instant home fallback contributes."""
        car = MagicMock()
        car.name = "TestCar"
        car.car_is_invited = False
        car.get_user_originated = MagicMock(return_value=None)
        # Plug is confirmed (for_duration returns True)
        car.is_car_plugged = MagicMock(return_value=True)
        # Home: for_duration=15 returns False, instant returns True
        car.is_car_home = MagicMock(side_effect=lambda time, for_duration=None: (
            False if for_duration == 15 else True
        ))
        car.get_continuous_plug_duration = MagicMock(return_value=120.0)
        car.get_car_coordinates = MagicMock(return_value=(None, None))

        charger = self._make_charger()
        now = datetime.now(pytz.UTC)
        with patch.object(charger, "get_continuous_plug_duration", return_value=100.0):
            score = charger.get_car_score(car, now, {})
        assert score > 0, f"Expected non-zero score from home instant fallback, got {score}"

    def test_plug_instant_false_gives_zero_score(self):
        """Car not plugged at all → score stays 0."""
        car = MagicMock()
        car.name = "TestCar"
        car.car_is_invited = False
        car.get_user_originated = MagicMock(return_value=None)
        car.is_car_plugged = MagicMock(return_value=False)
        car.is_car_home = MagicMock(return_value=True)
        car.get_car_coordinates = MagicMock(return_value=(None, None))

        charger = self._make_charger()
        now = datetime.now(pytz.UTC)
        score = charger.get_car_score(car, now, {})
        assert score == 0.0


# ── Departure Auto-Reset (Bug #92, Root Cause 3) ─────────────────────


class TestDepartureAutoReset:
    """Test auto-reset of car state after confirmed departure."""

    async def test_departure_15min_clears_user_originated(self, real_car, current_time):
        """Car home → car leaves → 15 min passes → user-originated state cleared."""
        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})

        # Set some user-originated state
        real_car.set_user_originated("charger_name", FORCE_CAR_NO_CHARGER_CONNECTED)
        assert real_car.has_user_originated("charger_name")

        # Car is home
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "home", {})
        await real_car._check_departure_auto_reset(current_time)
        assert real_car._car_not_home_since is None
        assert real_car.has_user_originated("charger_name")

        # Car leaves home
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "not_home", {})
        await real_car._check_departure_auto_reset(current_time)
        assert real_car._car_not_home_since == current_time
        assert real_car.has_user_originated("charger_name")  # Not yet cleared

        # 15 minutes later — should trigger reset (once)
        later_time = current_time + timedelta(seconds=CAR_NOT_HOME_AUTO_RESET_S)
        with patch.object(real_car, "user_clean_and_reset", new_callable=AsyncMock) as mock_reset:
            await real_car._check_departure_auto_reset(later_time)
            mock_reset.assert_called_once()
        assert real_car._departure_auto_reset_done is True

        # Subsequent cycles should NOT re-trigger the reset
        even_later = later_time + timedelta(seconds=CAR_NOT_HOME_AUTO_RESET_S)
        with patch.object(real_car, "user_clean_and_reset", new_callable=AsyncMock) as mock_reset:
            await real_car._check_departure_auto_reset(even_later)
            mock_reset.assert_not_called()

    async def test_departure_10min_preserves_state(self, real_car, current_time):
        """Car home → car leaves → only 10 min → user-originated state preserved."""
        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})

        real_car.set_user_originated("charger_name", FORCE_CAR_NO_CHARGER_CONNECTED)

        # Car leaves
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "not_home", {})
        await real_car._check_departure_auto_reset(current_time)
        assert real_car._car_not_home_since == current_time

        # Only 10 minutes later — should NOT trigger reset
        later_time = current_time + timedelta(minutes=10)
        with patch.object(real_car, "user_clean_and_reset", new_callable=AsyncMock) as mock_reset:
            await real_car._check_departure_auto_reset(later_time)
            mock_reset.assert_not_called()
        assert real_car.has_user_originated("charger_name")

    async def test_gps_glitch_preserves_state(self, real_car, current_time):
        """Car home → brief GPS glitch (not-home for 5 min then back) → state preserved."""
        fresh_time = current_time - timedelta(seconds=60)
        for sensor_id in real_car._car_api_all_sensors:
            real_car._entity_probed_last_valid_state[sensor_id] = (fresh_time, "value", {})

        real_car.set_user_originated("charger_name", FORCE_CAR_NO_CHARGER_CONNECTED)

        # Car appears to leave (GPS glitch)
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (fresh_time, "not_home", {})
        await real_car._check_departure_auto_reset(current_time)
        assert real_car._car_not_home_since == current_time

        # 5 minutes later, car comes back home
        later_time = current_time + timedelta(minutes=5)
        real_car._entity_probed_last_valid_state[real_car.car_tracker] = (later_time, "home", {})
        await real_car._check_departure_auto_reset(later_time)
        assert real_car._car_not_home_since is None
        assert real_car._departure_auto_reset_done is False  # Re-armed for next departure
        assert real_car.has_user_originated("charger_name")  # State preserved

    async def test_no_tracker_skips_reset(self, real_car, current_time):
        """Car with no tracker → no auto-reset attempted."""
        real_car.car_tracker = None
        real_car.set_user_originated("charger_name", FORCE_CAR_NO_CHARGER_CONNECTED)

        await real_car._check_departure_auto_reset(current_time)
        assert real_car._car_not_home_since is None
        assert real_car.has_user_originated("charger_name")


# ── Integration: Guest-to-Known-Car Transition (Bug #92) ─────────────


class TestGuestToKnownCarTransition:
    """End-to-end scenario tests for the guest-to-real-car swap."""

    def _make_charger_and_car(self):
        """Create charger with a real car and guest car for integration testing."""
        from tests.test_charger_additional_coverage import (
            create_charger_generic,
            create_mock_hass,
            create_mock_home,
        )

        hass = create_mock_hass()
        home = create_mock_home(hass)
        charger = create_charger_generic(hass, home)

        # Real car — not invited, plugged+home via instant check
        real_car = MagicMock()
        real_car.name = "Twingo"
        real_car.car_is_invited = False
        real_car.get_user_originated = MagicMock(return_value=None)
        real_car.get_car_coordinates = MagicMock(return_value=(None, None))
        real_car.charger = None

        # Guest car — invited
        guest_car = MagicMock()
        guest_car.name = "TestCharger generic car"
        guest_car.car_is_invited = True
        guest_car.get_user_originated = MagicMock(return_value=None)

        home._cars = [real_car]
        home.get_car_by_name = MagicMock(return_value=None)
        charger.car = None
        charger.car_attach_time = None

        return charger, real_car, guest_car, home

    def test_known_car_replaces_guest_via_scoring(self):
        """Charger plugged → guest car attached → API reports plugged+home → known car selected."""
        charger, real_car, guest_car, home = self._make_charger_and_car()
        now = datetime.now(pytz.UTC)

        # Car API just started reporting plugged (instant only, duration < 15s)
        real_car.is_car_plugged = MagicMock(side_effect=lambda time, for_duration=None: (
            False if for_duration == 15 else True
        ))
        real_car.is_car_home = MagicMock(return_value=True)
        real_car.get_continuous_plug_duration = MagicMock(return_value=5.0)

        # Charger has been plugged for hours (with guest car)
        with patch.object(charger, "is_plugged", return_value=True):
            best = charger.get_best_car(now)

        # Known car should be selected (not the guest)
        assert best is not None
        assert best.name == "Twingo", f"Expected Twingo, got {best.name}"

    def test_force_no_charger_cleared_after_departure(self):
        """Car had FORCE_NO_CHARGER → left home 15 min → came back → scores normally."""
        charger, real_car, guest_car, home = self._make_charger_and_car()
        now = datetime.now(pytz.UTC)

        # Car has FORCE_NO_CHARGER set — should be skipped
        real_car.get_user_originated = MagicMock(
            side_effect=lambda key, default=None: (
                FORCE_CAR_NO_CHARGER_CONNECTED if key == "charger_name" else default
            )
        )
        real_car.is_car_plugged = MagicMock(return_value=True)
        real_car.is_car_home = MagicMock(return_value=True)
        real_car.get_continuous_plug_duration = MagicMock(return_value=120.0)

        with patch.object(charger, "is_plugged", return_value=True):
            best = charger.get_best_car(now)
        # Car should get generic fallback because FORCE flag blocks the real car
        assert best.name == charger._default_generic_car.name

        # After departure reset, FORCE flag is cleared
        real_car.get_user_originated = MagicMock(return_value=None)
        with patch.object(charger, "is_plugged", return_value=True):
            best = charger.get_best_car(now)
        assert best.name == "Twingo"

    def test_manual_selection_overrides_scoring(self):
        """User manual selection takes priority over scoring while car is home."""
        charger, real_car, guest_car, home = self._make_charger_and_car()
        now = datetime.now(pytz.UTC)

        # Another car is also available
        other_car = MagicMock()
        other_car.name = "OtherCar"
        other_car.car_is_invited = False
        other_car.get_user_originated = MagicMock(return_value=None)
        other_car.is_car_plugged = MagicMock(return_value=True)
        other_car.is_car_home = MagicMock(return_value=True)
        other_car.get_continuous_plug_duration = MagicMock(return_value=120.0)
        other_car.get_car_coordinates = MagicMock(return_value=(None, None))
        other_car.charger = None
        home._cars.append(other_car)

        real_car.is_car_plugged = MagicMock(return_value=True)
        real_car.is_car_home = MagicMock(return_value=True)
        real_car.get_continuous_plug_duration = MagicMock(return_value=120.0)

        # User manually selects Twingo
        charger.set_user_originated("car_name", "Twingo")
        home.get_car_by_name = MagicMock(return_value=real_car)

        best = charger.get_best_car(now)
        assert best.name == "Twingo"
