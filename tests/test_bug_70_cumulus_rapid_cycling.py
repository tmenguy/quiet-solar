"""Tests for bug #70: cumulus rapid cycling caused by filler-constraint infinite loop.

Verifies fixes for:
- Task 1: Skip filler constraint when car is already fully charged (prevents push-remove-push loop)
- Task 2: Bistate metrics fallback to _last_completed_constraint after removal
"""

from __future__ import annotations

import unittest
from datetime import datetime, time as dt_time, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from custom_components.quiet_solar.const import (
    CONF_CHARGER_CONSUMPTION,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_PLUGGED,
    CONF_CHARGER_STATUS_SENSOR,
    CONF_IS_3P,
    CONF_MONO_PHASE,
    CONF_SWITCH,
    CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
    CONSTRAINT_TYPE_FILLER,
    DATA_HANDLER,
    DOMAIN,
)
from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric
from custom_components.quiet_solar.home_model.commands import LoadCommand
from custom_components.quiet_solar.home_model.constraints import (
    DATETIME_MIN_UTC,
    TimeBasedSimplePowerLoadConstraint,
)
from tests.factories import create_minimal_home_model

# =============================================================================
# Test helpers (following test_bug_48 pattern)
# =============================================================================


def create_mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.services = MagicMock()
    hass.data = {DOMAIN: {DATA_HANDLER: MagicMock()}}
    return hass


def create_mock_home(hass):
    """Create a mock QSHome instance."""
    home = create_minimal_home_model()
    home.hass = hass
    home.battery = None
    home.is_off_grid = MagicMock(return_value=False)
    home.get_available_power_values = MagicMock(return_value=None)
    home.get_grid_consumption_power_values = MagicMock(return_value=None)
    home.battery_can_discharge = MagicMock(return_value=True)
    home.get_tariff = MagicMock(return_value=0.15)
    home.get_best_tariff = MagicMock(return_value=0.10)
    home.force_next_solve = MagicMock()
    home.get_car_by_name = MagicMock(return_value=None)
    return home


def create_charger_generic(hass, home, name="TestCharger", **extra_config):
    """Create a QSChargerGeneric instance for testing."""
    config_entry = MagicMock()
    config_entry.entry_id = f"test_entry_{name}"
    config_entry.data = {}

    config = {
        "name": name,
        "hass": hass,
        "home": home,
        "config_entry": config_entry,
        CONF_CHARGER_MIN_CHARGE: 6,
        CONF_CHARGER_MAX_CHARGE: 32,
        CONF_CHARGER_CONSUMPTION: 70,
        CONF_IS_3P: True,
        CONF_MONO_PHASE: 1,
        CONF_CHARGER_STATUS_SENSOR: f"sensor.{name}_status",
        CONF_CHARGER_PLUGGED: f"sensor.{name}_plugged",
        CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER: f"number.{name}_max_current",
    }
    config.update(extra_config)

    with patch("custom_components.quiet_solar.ha_model.charger.entity_registry"):
        charger = QSChargerGeneric(**config)

    return charger


def create_mock_car(
    name="TestCar",
    target_charge=95.0,
    car_default_charge=100.0,
    car_battery_capacity=50000.0,
    current_charge=100.0,
    min_ok_soc=20.0,
):
    """Create a mock car with configurable charge parameters."""
    mock_car = MagicMock()
    mock_car.name = name
    mock_car.car_battery_capacity = car_battery_capacity
    mock_car.car_default_charge = car_default_charge
    mock_car.efficiency_factor = 1.0
    mock_car.do_force_next_charge = False
    mock_car.do_next_charge_time = None
    mock_car.can_use_charge_percent_constraints.return_value = True
    mock_car.setup_car_charge_target_if_needed = AsyncMock(return_value=target_charge)
    mock_car.get_car_charge_percent.return_value = current_charge
    mock_car.get_car_target_SOC.return_value = target_charge
    mock_car.get_car_minimum_ok_SOC.return_value = min_ok_soc
    mock_car.get_next_scheduled_event = AsyncMock(return_value=(None, None))
    mock_car.get_best_person_next_need = AsyncMock(return_value=(None, None, None, None))
    mock_car.set_next_charge_target_percent = AsyncMock()

    user_values: dict = {}

    def has_user_originated(key):
        return key in user_values

    def get_user_originated(key):
        return user_values.get(key)

    def set_user_originated(key, value):
        user_values[key] = value

    def clear_user_originated(key):
        user_values.pop(key, None)

    def clear_all_user_originated():
        user_values.clear()

    mock_car.has_user_originated = MagicMock(side_effect=has_user_originated)
    mock_car.get_user_originated = MagicMock(side_effect=get_user_originated)
    mock_car.set_user_originated = MagicMock(side_effect=set_user_originated)
    mock_car.clear_user_originated = MagicMock(side_effect=clear_user_originated)
    mock_car.clear_all_user_originated = MagicMock(side_effect=clear_all_user_originated)

    return mock_car


def setup_charger_with_plugged_car(charger, mock_car, time):
    """Set up charger to be plugged in with a car, past boot time."""
    charger.car = mock_car
    charger._boot_time = None
    charger._boot_time_adjusted = None
    charger._power_steps = [LoadCommand(command="on", power_consign=7000.0)]


# =============================================================================
# Task 1: Filler constraint skipped when car fully charged
# =============================================================================


class TestFillerConstraintSkippedWhenFullyCharged(unittest.IsolatedAsyncioTestCase):
    """Test that no filler constraint is pushed when the car is already at max charge."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(self.hass, self.home)
        self.time = datetime(2026, 3, 30, 10, 0, 0, tzinfo=pytz.UTC)

    async def test_fully_charged_car_does_not_push_filler(self):
        """Car at 100% with target 95% and max 100% should NOT push filler constraint.

        This is the exact scenario from bug #70: realized >= max_target_charge,
        so the filler would be immediately met, removed, force_solving=True,
        and the cycle repeats every 7 seconds.
        """
        mock_car = create_mock_car(
            target_charge=95.0,
            car_default_charge=100.0,
            current_charge=100.0,  # fully charged
        )
        setup_charger_with_plugged_car(self.charger, mock_car, self.time)

        push_calls = []
        original_push = self.charger.push_live_constraint

        def tracking_push(time, constraint):
            push_calls.append(constraint)
            return original_push(time, constraint)

        with (
            patch.object(self.charger, "is_not_plugged", return_value=False),
            patch.object(self.charger, "is_plugged", return_value=True),
            patch.object(self.charger, "is_charger_unavailable", return_value=False),
            patch.object(self.charger, "probe_for_possible_needed_reboot", return_value=False),
            patch.object(self.charger, "get_best_car", return_value=mock_car),
            patch.object(
                self.charger, "clean_constraints_for_load_param_and_if_same_key_same_value_info", return_value=False
            ),
            patch.object(self.charger, "push_live_constraint", side_effect=tracking_push),
        ):
            await self.charger.check_load_activity_and_constraints(self.time)

        filler_pushes = [
            c for c in push_calls if c._type in (CONSTRAINT_TYPE_FILLER, CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN)
        ]
        assert filler_pushes == [], (
            f"Filler constraint should NOT be pushed for fully-charged car, "
            f"but got {len(filler_pushes)} filler push(es)"
        )

    async def test_partially_charged_car_still_pushes_filler(self):
        """Car at 96% with target 95% and max 100% SHOULD push filler (96 < 100).

        The car passed its target but has room to charge to max_target_charge.
        """
        mock_car = create_mock_car(
            target_charge=95.0,
            car_default_charge=100.0,
            current_charge=96.0,  # past target but not at max
        )
        setup_charger_with_plugged_car(self.charger, mock_car, self.time)

        push_calls = []
        original_push = self.charger.push_live_constraint

        def tracking_push(time, constraint):
            push_calls.append(constraint)
            return original_push(time, constraint)

        with (
            patch.object(self.charger, "is_not_plugged", return_value=False),
            patch.object(self.charger, "is_plugged", return_value=True),
            patch.object(self.charger, "is_charger_unavailable", return_value=False),
            patch.object(self.charger, "probe_for_possible_needed_reboot", return_value=False),
            patch.object(self.charger, "get_best_car", return_value=mock_car),
            patch.object(
                self.charger, "clean_constraints_for_load_param_and_if_same_key_same_value_info", return_value=False
            ),
            patch.object(self.charger, "push_live_constraint", side_effect=tracking_push),
        ):
            await self.charger.check_load_activity_and_constraints(self.time)

        filler_pushes = [
            c for c in push_calls if c._type in (CONSTRAINT_TYPE_FILLER, CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN)
        ]
        assert len(filler_pushes) > 0, (
            "Filler constraint SHOULD be pushed for partially-charged car (96% < 100% max)"
        )
        # The filler should target max_target_charge (100%)
        filler = filler_pushes[0]
        assert filler.target_value == pytest.approx(100.0), (
            f"Filler target should be max_target_charge (100%), got {filler.target_value}"
        )


# =============================================================================
# Task 2: Bistate metrics fallback to _last_completed_constraint
# =============================================================================


class TestBistateMetricsFallback(unittest.TestCase):
    """Test that bistate update_current_metrics falls back to _last_completed_constraint."""

    def _create_bistate_device(self):
        """Create a concrete bistate device for testing metrics."""
        from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration

        hass = create_mock_hass()
        home = create_mock_home(hass)
        config_entry = MagicMock()
        config_entry.entry_id = "test_bistate_metrics"
        config_entry.data = {}

        class ConcreteBiState(QSBiStateDuration):
            def __init__(self, **kwargs):
                if "switch_entity" in kwargs:
                    kwargs[CONF_SWITCH] = kwargs.pop("switch_entity")
                elif CONF_SWITCH not in kwargs:
                    kwargs[CONF_SWITCH] = "switch.test_device"
                super().__init__(**kwargs)

            async def execute_command_system(self, time, command, state):
                return True

            def get_virtual_current_constraint_translation_key(self):
                return "test_key"

            def get_select_translation_key(self):
                return "test_select_key"

        device = ConcreteBiState(
            hass=hass,
            config_entry=config_entry,
            home=home,
            name="Test Cumulus",
        )
        device.power_use = 2000.0
        device.default_on_finish_time = dt_time(hour=0, minute=0, second=0)
        return device

    def _create_time_constraint(self, device, time, target_seconds, current_seconds, start_offset_h=0):
        """Create a TimeBasedSimplePowerLoadConstraint for testing."""
        start = time - timedelta(hours=start_offset_h) if start_offset_h else DATETIME_MIN_UTC
        end = time + timedelta(hours=12)
        return TimeBasedSimplePowerLoadConstraint(
            type=1,
            time=time,
            load=device,
            power=device.power_use,
            initial_value=0,
            target_value=target_seconds,
            current_value=current_seconds,
            start_of_constraint=start,
            end_of_constraint=end,
        )

    def test_metrics_show_completed_hours_after_constraint_removal(self):
        """After constraint removal, metrics should fall back to _last_completed_constraint.

        This is bug #70 task 2: when the active constraint is removed,
        qs_bistate_current_on_h was reset to 0 instead of showing the completed 3h.
        """
        device = self._create_bistate_device()
        time = datetime(2026, 3, 30, 18, 0, 0, tzinfo=pytz.UTC)

        completed_ct = self._create_time_constraint(
            device, time, target_seconds=3 * 3600.0, current_seconds=3 * 3600.0, start_offset_h=6
        )

        # Constraint was removed (empty list) but we have a completed one
        device._constraints = []
        device._last_completed_constraint = completed_ct

        device.update_current_metrics(time)

        assert device.qs_bistate_current_on_h == pytest.approx(3.0), (
            f"Should show 3h from completed constraint, got {device.qs_bistate_current_on_h}"
        )
        assert device.qs_bistate_current_duration_h == pytest.approx(3.0), (
            f"Should show 3h duration from completed constraint, got {device.qs_bistate_current_duration_h}"
        )

    def test_metrics_show_active_constraint_hours_when_live(self):
        """When an active constraint exists, metrics should show its current progress."""
        device = self._create_bistate_device()
        time = datetime(2026, 3, 30, 18, 0, 0, tzinfo=pytz.UTC)

        active_ct = self._create_time_constraint(
            device, time, target_seconds=3 * 3600.0, current_seconds=1.5 * 3600.0, start_offset_h=3
        )

        device._constraints = [active_ct]
        device._last_completed_constraint = None

        device.update_current_metrics(time)

        assert device.qs_bistate_current_on_h == pytest.approx(1.5), (
            f"Should show 1.5h from active constraint, got {device.qs_bistate_current_on_h}"
        )
        assert device.qs_bistate_current_duration_h == pytest.approx(3.0), (
            f"Should show 3h target duration, got {device.qs_bistate_current_duration_h}"
        )

    def test_metrics_zero_when_no_constraints_at_all(self):
        """When no constraints exist and no completed constraint, metrics should be zero."""
        device = self._create_bistate_device()
        time = datetime(2026, 3, 30, 18, 0, 0, tzinfo=pytz.UTC)

        device._constraints = []
        device._last_completed_constraint = None

        device.update_current_metrics(time)

        assert device.qs_bistate_current_on_h == 0.0
        assert device.qs_bistate_current_duration_h == 0.0


if __name__ == "__main__":
    unittest.main()
