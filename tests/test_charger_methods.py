"""
Additional tests for charger.py to improve coverage of specific methods.
Focuses on plugged state, charge state, car management, and constraint handling.
"""
import unittest
from unittest.mock import MagicMock, Mock, patch, AsyncMock, PropertyMock
from datetime import datetime, timedelta, time as dt_time
import pytz
import pytest
import asyncio

from homeassistant.const import CONF_NAME, STATE_UNKNOWN, STATE_UNAVAILABLE

from custom_components.quiet_solar.ha_model.home import QSHome
from custom_components.quiet_solar.ha_model.dynamic_group import QSDynamicGroup
from custom_components.quiet_solar.ha_model.charger import (
    QSChargerWallbox,
    QSChargerOCPP,
    QSChargerGroup,
    QSChargerStatus,
    QSStateCmd,
    QSChargerStates,
    QSChargerGeneric,
    WallboxChargerStatus,
    QSOCPPv16v201ChargePointStatus,
    CHARGER_CHECK_STATE_WINDOW_S,
    CHARGER_ADAPTATION_WINDOW_S,
    CHARGER_STATE_REFRESH_INTERVAL_S,
    TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_OFF_TO_ON_S,
    TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_ON_TO_OFF_S,
)
from custom_components.quiet_solar.ha_model.car import QSCar
from custom_components.quiet_solar.home_model.commands import (
    CMD_AUTO_FROM_CONSIGN,
    CMD_AUTO_GREEN_ONLY,
    CMD_AUTO_GREEN_CAP,
    CMD_AUTO_GREEN_CONSIGN,
    CMD_AUTO_PRICE,
    CMD_ON,
    CMD_OFF,
    LoadCommand,
    copy_command
)
from custom_components.quiet_solar.const import (
    CONF_DYN_GROUP_MAX_PHASE_AMPS,
    CONF_MONO_PHASE,
    CONF_CHARGER_DEVICE_WALLBOX,
    CONF_CHARGER_DEVICE_OCPP,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CHARGER_CONSUMPTION,
    CONF_CHARGER_STATUS_SENSOR,
    CONF_CHARGER_PLUGGED,
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER,
    CONF_CHARGER_PAUSE_RESUME_SWITCH,
    CONF_CHARGER_THREE_TO_ONE_PHASE_SWITCH,
    CONF_CHARGER_REBOOT_BUTTON,
    CONF_CHARGER_LONGITUDE,
    CONF_CHARGER_LATITUDE,
    CONF_IS_3P,
    DOMAIN,
    DATA_HANDLER,
    CHARGER_NO_CAR_CONNECTED,
    CAR_CHARGE_TYPE_NOT_CHARGING,
    CAR_CHARGE_TYPE_NOT_PLUGGED,
    CAR_CHARGE_TYPE_FAULTED,
    CAR_CHARGE_TYPE_AS_FAST_AS_POSSIBLE,
    CAR_CHARGE_TYPE_SCHEDULE,
    CAR_CHARGE_TYPE_SOLAR_PRIORITY_BEFORE_BATTERY,
    CAR_CHARGE_TYPE_SOLAR_AFTER_BATTERY,
    CAR_CHARGE_TYPE_TARGET_MET,
)
from custom_components.quiet_solar.home_model.constraints import DATETIME_MAX_UTC


def create_mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.services = MagicMock()
    hass.data = {
        DOMAIN: {
            DATA_HANDLER: MagicMock()
        }
    }
    return hass


def create_mock_home(hass):
    """Create a mock QSHome instance."""
    home = MagicMock()
    home.hass = hass
    home._chargers = []
    home._cars = []
    home.battery = None
    home.get_available_power_values = MagicMock(return_value=None)
    home.get_grid_consumption_power_values = MagicMock(return_value=None)
    home.battery_can_discharge = MagicMock(return_value=True)
    home.get_tariff = MagicMock(return_value=0.15)
    home.get_best_tariff = MagicMock(return_value=0.10)
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

    with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
        charger = QSChargerGeneric(**config)

    return charger


class TestQSChargerGenericPluggedState(unittest.TestCase):
    """Test plugged state methods in QSChargerGeneric."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(self.hass, self.home)

    def test_is_optimistic_plugged_returns_direct_result(self):
        """Test is_optimistic_plugged returns direct result when available."""
        with patch.object(self.charger, 'is_plugged', return_value=True):
            time = datetime.now(pytz.UTC)
            result = self.charger.is_optimistic_plugged(time)
        self.assertTrue(result)

    def test_is_optimistic_plugged_checks_duration_if_none(self):
        """Test is_optimistic_plugged checks with duration if direct is None."""
        time = datetime.now(pytz.UTC)
        with patch.object(self.charger, 'is_plugged') as mock_is_plugged:
            # First call returns None, second call returns True
            mock_is_plugged.side_effect = [None, True]
            result = self.charger.is_optimistic_plugged(time)

        self.assertTrue(result)
        self.assertEqual(mock_is_plugged.call_count, 2)

    def test_get_continuous_plug_duration(self):
        """Test get_continuous_plug_duration method."""
        time = datetime.now(pytz.UTC)
        with patch.object(self.charger, 'get_last_state_value_duration', return_value=(120.0, time)):
            result = self.charger.get_continuous_plug_duration(time)

        self.assertEqual(result, 120.0)

    def test_is_charger_plugged_now_with_status(self):
        """Test is_charger_plugged_now with status sensor."""
        time = datetime.now(pytz.UTC)

        mock_state = MagicMock()
        mock_state.state = "Charging"
        mock_state.last_updated = time
        self.hass.states.get.return_value = mock_state

        with patch.object(self.charger, 'get_car_plugged_in_status_vals', return_value=["Charging", "Ready"]):
            is_plugged, state_time = self.charger.is_charger_plugged_now(time)

        self.assertTrue(is_plugged)
        self.assertEqual(state_time, time)

    def test_is_charger_plugged_now_unplugged(self):
        """Test is_charger_plugged_now when unplugged."""
        time = datetime.now(pytz.UTC)

        mock_state = MagicMock()
        mock_state.state = "Disconnected"
        mock_state.last_updated = time
        self.hass.states.get.return_value = mock_state

        with patch.object(self.charger, 'get_car_plugged_in_status_vals', return_value=["Charging", "Ready"]):
            is_plugged, state_time = self.charger.is_charger_plugged_now(time)

        self.assertFalse(is_plugged)

    def test_is_charger_plugged_now_unknown_state(self):
        """Test is_charger_plugged_now with unknown state."""
        time = datetime.now(pytz.UTC)

        mock_state = MagicMock()
        mock_state.state = STATE_UNKNOWN
        mock_state.last_updated = time
        self.hass.states.get.return_value = mock_state

        is_plugged, state_time = self.charger.is_charger_plugged_now(time)

        self.assertIsNone(is_plugged)


class TestQSChargerGenericChargeState(unittest.TestCase):
    """Test charge state methods in QSChargerGeneric."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(self.hass, self.home)

    def test_is_charge_enabled_when_plugged(self):
        """Test is_charge_enabled returns True when charging."""
        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'is_optimistic_plugged', return_value=True), \
             patch.object(self.charger, '_check_charger_status', return_value=True):
            result = self.charger.is_charge_enabled(time)

        self.assertTrue(result)

    def test_is_charge_enabled_when_not_plugged(self):
        """Test is_charge_enabled returns False when not plugged."""
        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'is_optimistic_plugged', return_value=False):
            result = self.charger.is_charge_enabled(time)

        self.assertFalse(result)

    def test_is_charge_disabled(self):
        """Test is_charge_disabled method."""
        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'is_optimistic_plugged', return_value=True), \
             patch.object(self.charger, '_check_charger_status', return_value=True):
            result = self.charger.is_charge_disabled(time)

        self.assertTrue(result)

    def test_is_charger_unavailable_no_sensor(self):
        """Test is_charger_unavailable without status sensor."""
        time = datetime.now(pytz.UTC)
        self.charger.charger_status_sensor = None

        result = self.charger.is_charger_unavailable(time)
        self.assertFalse(result)

    def test_is_charger_unavailable_unavailable_state(self):
        """Test is_charger_unavailable with unavailable state."""
        time = datetime.now(pytz.UTC)

        self.hass.states.get.return_value = None

        result = self.charger.is_charger_unavailable(time)
        self.assertTrue(result)

    def test_is_charging_power_zero_true(self):
        """Test is_charging_power_zero when power is zero."""
        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'get_average_power', return_value=30.0), \
             patch.object(self.charger, 'dampening_power_value_for_car_consumption', return_value=0.0):
            result = self.charger.is_charging_power_zero(time, 60.0)

        self.assertTrue(result)

    def test_is_charging_power_zero_false(self):
        """Test is_charging_power_zero when power is not zero."""
        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'get_average_power', return_value=5000.0), \
             patch.object(self.charger, 'dampening_power_value_for_car_consumption', return_value=5000.0):
            result = self.charger.is_charging_power_zero(time, 60.0)

        self.assertFalse(result)

    def test_is_charging_power_zero_none(self):
        """Test is_charging_power_zero when power is None."""
        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'get_average_power', return_value=None):
            result = self.charger.is_charging_power_zero(time, 60.0)

        self.assertIsNone(result)


class TestQSChargerGenericCarManagement(unittest.TestCase):
    """Test car management methods in QSChargerGeneric."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(self.hass, self.home)

    def test_attach_car(self):
        """Test attach_car method."""
        mock_car = MagicMock()
        mock_car.name = "TestCar"
        mock_car.calendar = None
        mock_car.reset = MagicMock()
        mock_car.get_charge_power_per_phase_A = MagicMock(return_value=([1000, 2000, 3000], 6, 16))

        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'can_do_3_to_1_phase_switch', return_value=False), \
             patch.object(type(self.charger), 'physical_3p', new_callable=PropertyMock) as mock_3p, \
             patch.object(type(self.charger), 'min_charge', new_callable=PropertyMock) as mock_min, \
             patch.object(type(self.charger), 'max_charge', new_callable=PropertyMock) as mock_max:
            mock_3p.return_value = True
            mock_min.return_value = 6
            mock_max.return_value = 16

            self.charger.attach_car(mock_car, time)

        self.assertEqual(self.charger.car, mock_car)
        self.assertEqual(self.charger.car_attach_time, time)
        self.assertEqual(mock_car.charger, self.charger)
        mock_car.reset.assert_called_once()

    def test_attach_same_car(self):
        """Test attach_car with same car (should return early)."""
        mock_car = MagicMock()
        mock_car.name = "TestCar"

        self.charger.car = mock_car
        time = datetime.now(pytz.UTC)

        self.charger.attach_car(mock_car, time)

        # Should return without doing anything
        self.assertEqual(self.charger.car, mock_car)

    def test_detach_car(self):
        """Test detach_car method."""
        mock_car = MagicMock()
        mock_car.name = "TestCar"
        self.charger.car = mock_car
        mock_car.charger = self.charger
        self.charger.car_attach_time = datetime.now(pytz.UTC)
        self.charger._power_steps = [1, 2, 3]

        self.charger.detach_car()

        self.assertIsNone(self.charger.car)
        self.assertIsNone(self.charger.car_attach_time)
        self.assertEqual(self.charger._power_steps, [])
        self.assertIsNone(mock_car.charger)

    def test_min_charge_with_car(self):
        """Test min_charge property with car attached."""
        mock_car = MagicMock()
        mock_car.car_charger_min_charge = 8
        self.charger.charger_min_charge = 6
        self.charger.car = mock_car

        result = self.charger.min_charge
        self.assertEqual(result, 8)  # Should return car's min (higher)

    def test_min_charge_without_car(self):
        """Test min_charge property without car."""
        self.charger.charger_min_charge = 6
        self.charger.car = None

        result = self.charger.min_charge
        self.assertEqual(result, 6)

    def test_max_charge_with_car(self):
        """Test max_charge property with car attached."""
        mock_car = MagicMock()
        mock_car.car_charger_max_charge = 24
        self.charger.charger_max_charge = 32
        self.charger.car = mock_car

        result = self.charger.max_charge
        self.assertEqual(result, 24)  # Should return car's max (lower)

    def test_max_charge_without_car(self):
        """Test max_charge property without car."""
        self.charger.charger_max_charge = 32
        self.charger.car = None

        result = self.charger.max_charge
        self.assertEqual(result, 32)

    def test_get_min_max_power_no_power_steps(self):
        """Test get_min_max_power with no power steps."""
        self.charger._power_steps = []
        self.charger.car = MagicMock()

        min_power, max_power = self.charger.get_min_max_power()

        self.assertEqual(min_power, 0.0)
        self.assertEqual(max_power, 0.0)

    def test_get_min_max_power_no_car(self):
        """Test get_min_max_power without car."""
        self.charger._power_steps = [MagicMock(power_consign=1000), MagicMock(power_consign=5000)]
        self.charger.car = None

        min_power, max_power = self.charger.get_min_max_power()

        self.assertEqual(min_power, 0.0)
        self.assertEqual(max_power, 0.0)


class TestQSChargerGenericChargeType(unittest.TestCase):
    """Test charge type methods in QSChargerGeneric."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(self.hass, self.home)

    def test_get_charge_type_no_car(self):
        """Test get_charge_type when no car is attached."""
        self.charger.car = None

        with patch.object(self.charger, 'is_charger_faulted', return_value=False):
            charge_type, constraint = self.charger.get_charge_type()

        self.assertEqual(charge_type, CAR_CHARGE_TYPE_NOT_PLUGGED)
        self.assertIsNone(constraint)

    def test_get_charge_type_faulted(self):
        """Test get_charge_type when charger is faulted."""
        self.charger.car = MagicMock()

        with patch.object(self.charger, 'is_charger_faulted', return_value=True):
            charge_type, constraint = self.charger.get_charge_type()

        self.assertEqual(charge_type, CAR_CHARGE_TYPE_FAULTED)

    def test_get_charge_type_as_fast_as_possible(self):
        """Test get_charge_type with as_fast_as_possible constraint."""
        self.charger.car = MagicMock()

        mock_constraint = MagicMock()
        mock_constraint.is_constraint_active_for_time_period.return_value = True
        mock_constraint.as_fast_as_possible = True

        self.charger._constraints = [mock_constraint]

        with patch.object(self.charger, 'is_charger_faulted', return_value=False):
            charge_type, constraint = self.charger.get_charge_type()

        self.assertEqual(charge_type, CAR_CHARGE_TYPE_AS_FAST_AS_POSSIBLE)
        self.assertEqual(constraint, mock_constraint)

    def test_get_charge_type_scheduled(self):
        """Test get_charge_type with scheduled constraint."""
        self.charger.car = MagicMock()

        mock_constraint = MagicMock()
        mock_constraint.is_constraint_active_for_time_period.return_value = True
        mock_constraint.as_fast_as_possible = False
        mock_constraint.end_of_constraint = datetime.now(pytz.UTC) + timedelta(hours=1)
        mock_constraint.load_info = None

        self.charger._constraints = [mock_constraint]

        with patch.object(self.charger, 'is_charger_faulted', return_value=False):
            charge_type, constraint = self.charger.get_charge_type()

        self.assertEqual(charge_type, CAR_CHARGE_TYPE_SCHEDULE)

    def test_get_charge_type_target_met(self):
        """Test get_charge_type when target is met."""
        self.charger.car = MagicMock()

        mock_constraint = MagicMock()
        mock_constraint.is_constraint_active_for_time_period.return_value = False
        mock_constraint.is_constraint_met.return_value = True

        self.charger._constraints = [mock_constraint]

        with patch.object(self.charger, 'is_charger_faulted', return_value=False):
            charge_type, constraint = self.charger.get_charge_type()

        self.assertEqual(charge_type, CAR_CHARGE_TYPE_TARGET_MET)

    def test_get_charge_type_before_battery(self):
        """Test get_charge_type with before battery priority."""
        self.charger.car = MagicMock()
        time = datetime.now(pytz.UTC)

        mock_constraint = MagicMock()
        mock_constraint.is_constraint_active_for_time_period.return_value = True
        mock_constraint.as_fast_as_possible = False
        mock_constraint.end_of_constraint = DATETIME_MAX_UTC
        mock_constraint.load_info = None

        self.charger._constraints = [mock_constraint]

        with patch.object(self.charger, 'is_charger_faulted', return_value=False), \
             patch.object(self.charger, 'compute_is_before_battery', return_value=True):
            charge_type, constraint = self.charger.get_charge_type()

        self.assertEqual(charge_type, CAR_CHARGE_TYPE_SOLAR_PRIORITY_BEFORE_BATTERY)

    def test_get_charge_type_after_battery(self):
        """Test get_charge_type with after battery priority."""
        self.charger.car = MagicMock()
        time = datetime.now(pytz.UTC)

        mock_constraint = MagicMock()
        mock_constraint.is_constraint_active_for_time_period.return_value = True
        mock_constraint.as_fast_as_possible = False
        mock_constraint.end_of_constraint = DATETIME_MAX_UTC
        mock_constraint.load_info = None

        self.charger._constraints = [mock_constraint]

        with patch.object(self.charger, 'is_charger_faulted', return_value=False), \
             patch.object(self.charger, 'compute_is_before_battery', return_value=False):
            charge_type, constraint = self.charger.get_charge_type()

        self.assertEqual(charge_type, CAR_CHARGE_TYPE_SOLAR_AFTER_BATTERY)


class TestQSChargerGenericAsync(unittest.IsolatedAsyncioTestCase):
    """Test async methods in QSChargerGeneric."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(self.hass, self.home)

    async def test_set_max_charging_current(self):
        """Test set_max_charging_current method."""
        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'get_max_charging_amp_per_phase', return_value=10), \
             patch.object(self.charger, 'low_level_set_max_charging_current', new_callable=AsyncMock) as mock_low:
            mock_low.return_value = True
            result = await self.charger.set_max_charging_current(16, time)

        self.assertTrue(result)
        mock_low.assert_called_once()

    async def test_set_max_charging_current_no_change(self):
        """Test set_max_charging_current when current is same."""
        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'get_max_charging_amp_per_phase', return_value=16), \
             patch.object(self.charger, 'low_level_set_max_charging_current', new_callable=AsyncMock) as mock_low:
            result = await self.charger.set_max_charging_current(16, time)

        self.assertFalse(result)
        mock_low.assert_not_called()

    async def test_stop_charge(self):
        """Test stop_charge method."""
        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'is_charge_enabled', return_value=True), \
             patch.object(self.charger, 'low_level_stop_charge', new_callable=AsyncMock) as mock_stop:
            await self.charger.stop_charge(time)

        mock_stop.assert_called_once_with(time)

    async def test_stop_charge_already_stopped(self):
        """Test stop_charge when already stopped."""
        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'is_charge_enabled', return_value=False), \
             patch.object(self.charger, 'low_level_stop_charge', new_callable=AsyncMock) as mock_stop:
            await self.charger.stop_charge(time)

        mock_stop.assert_not_called()

    async def test_start_charge(self):
        """Test start_charge method."""
        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'is_charge_disabled', return_value=True), \
             patch.object(self.charger, 'low_level_start_charge', new_callable=AsyncMock) as mock_start:
            await self.charger.start_charge(time)

        mock_start.assert_called_once_with(time)

    async def test_reboot(self):
        """Test reboot method."""
        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'can_reboot', return_value=True), \
             patch.object(self.charger, 'low_level_reboot', new_callable=AsyncMock) as mock_reboot:
            await self.charger.reboot(time)

        self.assertEqual(self.charger._asked_for_reboot_at_time, time)
        mock_reboot.assert_called_once_with(time)

    async def test_reboot_cannot_reboot(self):
        """Test reboot method when cannot reboot."""
        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'can_reboot', return_value=False), \
             patch.object(self.charger, 'low_level_reboot', new_callable=AsyncMock) as mock_reboot:
            await self.charger.reboot(time)

        self.assertIsNone(self.charger._asked_for_reboot_at_time)
        mock_reboot.assert_not_called()


class TestQSChargerGenericPhaseSwitch(unittest.IsolatedAsyncioTestCase):
    """Test phase switch methods in QSChargerGeneric."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(
            self.hass, self.home,
            **{CONF_CHARGER_THREE_TO_ONE_PHASE_SWITCH: "switch.phase_switch"}
        )

    def test_can_do_3_to_1_phase_switch_with_switch(self):
        """Test can_do_3_to_1_phase_switch when switch is configured."""
        with patch.object(type(self.charger), 'physical_3p', new_callable=PropertyMock) as mock_3p:
            mock_3p.return_value = True
            result = self.charger.can_do_3_to_1_phase_switch()

        self.assertTrue(result)

    def test_can_do_3_to_1_phase_switch_without_switch(self):
        """Test can_do_3_to_1_phase_switch when no switch is configured."""
        self.charger.charger_three_to_one_phase_switch = None

        result = self.charger.can_do_3_to_1_phase_switch()
        self.assertFalse(result)

    def test_can_do_3_to_1_phase_switch_not_3p(self):
        """Test can_do_3_to_1_phase_switch when charger is not 3P."""
        with patch.object(type(self.charger), 'physical_3p', new_callable=PropertyMock) as mock_3p:
            mock_3p.return_value = False
            result = self.charger.can_do_3_to_1_phase_switch()

        self.assertFalse(result)

    async def test_set_charging_num_phases_with_switch(self):
        """Test set_charging_num_phases with phase switch."""
        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'can_do_3_to_1_phase_switch', return_value=True), \
             patch.object(type(self.charger), 'physical_3p', new_callable=PropertyMock) as mock_3p, \
             patch.object(type(self.charger), 'current_num_phases', new_callable=PropertyMock) as mock_num, \
             patch.object(self.charger, 'low_level_set_charging_num_phases', new_callable=AsyncMock) as mock_low:
            mock_3p.return_value = True
            mock_num.return_value = 3
            mock_low.return_value = True

            result = await self.charger.set_charging_num_phases(1, time)

        self.assertTrue(result)
        mock_low.assert_called_once_with(1, time)

    async def test_set_charging_num_phases_no_switch(self):
        """Test set_charging_num_phases without phase switch."""
        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'can_do_3_to_1_phase_switch', return_value=False), \
             patch.object(type(self.charger), 'physical_num_phases', new_callable=PropertyMock) as mock_num:
            mock_num.return_value = 1

            result = await self.charger.set_charging_num_phases(1, time)

        self.assertTrue(result)


class TestQSChargerReboot(unittest.IsolatedAsyncioTestCase):
    """Test reboot-related methods in QSChargerGeneric."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(
            self.hass, self.home,
            **{CONF_CHARGER_REBOOT_BUTTON: "button.charger_reboot"}
        )

    def test_can_reboot_with_button(self):
        """Test can_reboot when button is configured."""
        result = self.charger.can_reboot()
        self.assertTrue(result)

    def test_can_reboot_without_button(self):
        """Test can_reboot when no button is configured."""
        self.charger.charger_reboot_button = None

        result = self.charger.can_reboot()
        self.assertFalse(result)

    def test_probe_for_possible_needed_reboot_no_button(self):
        """Test probe_for_possible_needed_reboot when cannot reboot."""
        self.charger.charger_reboot_button = None
        time = datetime.now(pytz.UTC)

        result = self.charger.probe_for_possible_needed_reboot(time)
        self.assertFalse(result)

    async def test_check_if_reboot_happened_cannot_reboot(self):
        """Test check_if_reboot_happened when cannot reboot."""
        from_time = datetime.now(pytz.UTC)
        to_time = from_time + timedelta(seconds=300)

        with patch.object(self.charger, 'can_reboot', return_value=False):
            result = await self.charger.check_if_reboot_happened(from_time, to_time)

        self.assertTrue(result)

    async def test_check_if_reboot_happened_too_short(self):
        """Test check_if_reboot_happened with too short duration."""
        from_time = datetime.now(pytz.UTC)
        to_time = from_time + timedelta(seconds=30)

        with patch.object(self.charger, 'can_reboot', return_value=True):
            result = await self.charger.check_if_reboot_happened(from_time, to_time)

        self.assertFalse(result)


class TestQSChargerGenericSavedInfo(unittest.TestCase):
    """Test saved info methods in QSChargerGeneric."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(self.hass, self.home)

    def test_update_to_be_saved_extra_device_info(self):
        """Test update_to_be_saved_extra_device_info method."""
        self.charger.user_attached_car_name = "TestCar"
        self.charger._auto_constraints_cleaned_at_user_reset = []

        data = {}
        self.charger.update_to_be_saved_extra_device_info(data)

        self.assertEqual(data["user_attached_car_name"], "TestCar")
        self.assertEqual(data["auto_constraints_cleaned_at_user_reset"], [])

    def test_use_saved_extra_device_info(self):
        """Test use_saved_extra_device_info method."""
        stored_info = {
            "user_attached_car_name": "SavedCar",
            "auto_constraints_cleaned_at_user_reset": []
        }

        self.charger.use_saved_extra_device_info(stored_info)

        self.assertEqual(self.charger.user_attached_car_name, "SavedCar")


if __name__ == '__main__':
    unittest.main()
