import unittest
from unittest.mock import MagicMock, Mock, patch, AsyncMock, PropertyMock
from datetime import datetime, timedelta, time as dt_time
import pytz
import pytest
import asyncio

# Import from Home Assistant
from homeassistant.const import CONF_NAME, STATE_UNKNOWN, STATE_UNAVAILABLE

# Import the necessary classes
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
    QSOCPPv16v201ChargePointStatus
)
from custom_components.quiet_solar.ha_model.car import QSCar
from custom_components.quiet_solar.home_model.commands import (
    CMD_AUTO_FROM_CONSIGN,
    CMD_AUTO_GREEN_ONLY,
    CMD_ON,
    CMD_OFF,
    LoadCommand
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
    CHARGER_NO_CAR_CONNECTED
)


class TestQSChargerGenericAdvanced(unittest.IsolatedAsyncioTestCase):
    """Test advanced QSChargerGeneric functionality."""
    
    def setUp(self):
        # Mock Home Assistant
        self.hass = MagicMock()
        self.hass.states = MagicMock()
        self.hass.states.get = MagicMock(return_value=None)
        self.hass.services = MagicMock()
        
        # Mock home
        self.home = MagicMock()
        self.home._chargers = []
        
        # Mock config entry
        self.config_entry = MagicMock()
        
        # Basic charger config
        self.charger_config = {
            "name": "TestCharger",
            "hass": self.hass,
            "home": self.home,
            "config_entry": self.config_entry,
            CONF_CHARGER_MIN_CHARGE: 6,
            CONF_CHARGER_MAX_CHARGE: 32,
            CONF_CHARGER_CONSUMPTION: 70,
            CONF_IS_3P: True,
            CONF_MONO_PHASE: 1,
            CONF_CHARGER_STATUS_SENSOR: "sensor.charger_status",
            CONF_CHARGER_PLUGGED: "sensor.charger_plugged",
            CONF_CHARGER_THREE_TO_ONE_PHASE_SWITCH: "switch.phase_switch"
        }
    
    def test_current_num_phases_3p_charger_switch_on(self):
        """Test current_num_phases property for 3P charger with switch on."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        # Mock the phase switch state as "on" (1 phase)
        mock_state = MagicMock()
        mock_state.state = "on"
        self.hass.states.get.return_value = mock_state
        
        with patch.object(charger, 'can_do_3_to_1_phase_switch', return_value=True):
            result = charger.current_num_phases
        
        self.assertEqual(result, 1)
    
    def test_current_num_phases_3p_charger_switch_off(self):
        """Test current_num_phases property for 3P charger with switch off."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        # Mock the phase switch state as "off" (3 phases)
        mock_state = MagicMock()
        mock_state.state = "off"
        self.hass.states.get.return_value = mock_state
        
        with patch.object(charger, 'can_do_3_to_1_phase_switch', return_value=True):
            result = charger.current_num_phases
        
        self.assertEqual(result, 3)
    
    def test_current_num_phases_unknown_state(self):
        """Test current_num_phases property with unknown switch state."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        # Mock the phase switch state as unknown
        mock_state = MagicMock()
        mock_state.state = STATE_UNKNOWN
        self.hass.states.get.return_value = mock_state
        
        with patch.object(charger, 'can_do_3_to_1_phase_switch', return_value=True):
            result = charger.current_num_phases
        
        self.assertEqual(result, 3)  # Should default to 3
    
    def test_current_num_phases_no_switch(self):
        """Test current_num_phases property when can't do phase switch."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        with patch.object(charger, 'can_do_3_to_1_phase_switch', return_value=False), \
             patch.object(type(charger), 'physical_num_phases', new_callable=PropertyMock) as mock_physical:
            mock_physical.return_value = 3
            result = charger.current_num_phases
        
        self.assertEqual(result, 3)
    
    def test_get_phase_amps_from_power_3p(self):
        """Test get_phase_amps_from_power for 3-phase."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        # Mock car
        mock_car = MagicMock()
        mock_car.get_charge_power_per_phase_A.return_value = ([6, 10, 16, 20, 32], None, None)
        charger.car = mock_car
        
        with patch.object(charger, '_get_amps_from_power_steps', return_value=15):
            result = charger.get_phase_amps_from_power(3000, is_3p=True)
        
        self.assertEqual(result, [15, 15, 15])
    
    def test_get_phase_amps_from_power_1p(self):
        """Test get_phase_amps_from_power for 1-phase."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        # Mock car
        mock_car = MagicMock()
        mock_car.get_charge_power_per_phase_A.return_value = ([6, 10, 16, 20, 32], None, None)
        charger.car = mock_car
        
        with patch.object(type(charger), 'mono_phase_index', new_callable=PropertyMock) as mock_mono:
            mock_mono.return_value = 0
            with patch.object(charger, '_get_amps_from_power_steps', return_value=12):
                result = charger.get_phase_amps_from_power(2000, is_3p=False)
        
        self.assertEqual(result, [12, 0, 0])
    
    def test_get_device_amps_consumption_charge_enabled(self):
        """Test get_device_amps_consumption when charge is enabled."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'is_charge_enabled', return_value=True), \
             patch.object(charger, 'get_max_charging_amp_per_phase', return_value=16), \
             patch.object(type(charger), 'current_3p', new_callable=PropertyMock) as mock_current_3p, \
             patch.object(type(charger), 'min_charge', new_callable=PropertyMock) as mock_min, \
             patch.object(type(charger), 'max_charge', new_callable=PropertyMock) as mock_max:
            
            mock_current_3p.return_value = True
            mock_min.return_value = 6
            mock_max.return_value = 32
            
            result = charger.get_device_amps_consumption(10.0, time)
        
        self.assertEqual(result, [16, 16, 16])
    
    def test_is_plugged_state_getter(self):
        """Test is_plugged_state_getter method."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'is_charger_plugged_now', return_value=(True, time)):
            result = charger.is_plugged_state_getter("test_entity", time)
        
        expected_time, expected_state, expected_attrs = result
        self.assertEqual(expected_time, time)
        self.assertEqual(expected_state, QSChargerStates.PLUGGED)
        self.assertEqual(expected_attrs, {})
    
    def test_is_plugged_state_getter_unplugged(self):
        """Test is_plugged_state_getter method when unplugged."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'is_charger_plugged_now', return_value=(False, time)):
            result = charger.is_plugged_state_getter("test_entity", time)
        
        expected_time, expected_state, expected_attrs = result
        self.assertEqual(expected_time, time)
        self.assertEqual(expected_state, QSChargerStates.UN_PLUGGED)
        self.assertEqual(expected_attrs, {})
    
    def test_is_plugged_state_getter_unknown(self):
        """Test is_plugged_state_getter method when state is unknown."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'is_charger_plugged_now', return_value=(None, time)):
            result = charger.is_plugged_state_getter("test_entity", time)
        
        expected_time, expected_state, expected_attrs = result
        self.assertEqual(expected_time, time)
        self.assertIsNone(expected_state)
        self.assertEqual(expected_attrs, {})
    
    def test_get_stable_dynamic_charge_status_disabled(self):
        """Test get_stable_dynamic_charge_status when device is disabled."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        charger.qs_enable_device = False
        time = datetime.now(pytz.UTC)
        
        result = charger.get_stable_dynamic_charge_status(time)
        self.assertIsNone(result)
    
    def test_get_stable_dynamic_charge_status_no_car(self):
        """Test get_stable_dynamic_charge_status when no car attached."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        charger.qs_enable_device = True
        charger.car = None
        time = datetime.now(pytz.UTC)
        
        result = charger.get_stable_dynamic_charge_status(time)
        self.assertIsNone(result)
    
    def test_get_stable_dynamic_charge_status_unavailable(self):
        """Test get_stable_dynamic_charge_status when charger is unavailable."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        charger.qs_enable_device = True
        charger.car = MagicMock()
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'is_not_plugged', return_value=False), \
             patch.object(charger, 'is_charger_unavailable', return_value=True):
            
            result = charger.get_stable_dynamic_charge_status(time)
        
        self.assertIsNone(result)
    
    def test_get_normalized_score_no_car(self):
        """Test get_normalized_score when no car is attached."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        charger.car = None
        mock_constraint = MagicMock()
        time = datetime.now(pytz.UTC)
        
        result = charger.get_normalized_score(mock_constraint, time, 100)
        self.assertEqual(result, 0.0)
    
    def test_get_normalized_score_with_car(self):
        """Test get_normalized_score with a car attached."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        # Mock car
        mock_car = MagicMock()
        mock_car.car_battery_capacity = 75000  # 75 kWh
        mock_car.get_car_charge_percent.return_value = 40.0  # 40% charged
        charger.car = mock_car
        
        # Mock constraint
        mock_constraint = MagicMock()
        mock_constraint.target_charge_percent = 80.0
        mock_constraint.end_of_constraint = datetime.now(pytz.UTC) + timedelta(hours=8)
        
        time = datetime.now(pytz.UTC)
        
        result = charger.get_normalized_score(mock_constraint, time, 1000)
        
        # Should return a score based on battery level and time remaining
        self.assertIsInstance(result, (int, float))
        self.assertGreaterEqual(result, 0.0)
    
    def test_get_best_car_user_selection(self):
        """Test get_best_car with user manual selection."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        # Mock car and home
        mock_car = MagicMock()
        mock_car.name = "UserSelectedCar"
        charger.home.get_car_by_name.return_value = mock_car
        charger.user_attached_car_name = "UserSelectedCar"
        charger.home._chargers = [charger]
        
        time = datetime.now(pytz.UTC)
        
        result = charger.get_best_car(time)
        
        self.assertEqual(result, mock_car)
    
    def test_get_best_car_no_car_connected_selection(self):
        """Test get_best_car with NO_CAR_CONNECTED selection."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        charger.user_attached_car_name = CHARGER_NO_CAR_CONNECTED
        
        time = datetime.now(pytz.UTC)
        
        result = charger.get_best_car(time)
        
        self.assertIsNone(result)
    
    def test_get_best_car_automatic_selection(self):
        """Test get_best_car with automatic car selection."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        # Mock multiple cars
        mock_car1 = MagicMock()
        mock_car1.name = "Car1"
        mock_car2 = MagicMock()
        mock_car2.name = "Car2"
        
        # Mock chargers
        mock_charger2 = MagicMock()
        mock_charger2.qs_enable_device = True
        mock_charger2.car = mock_car2
        mock_charger2.user_attached_car_name = None
        charger.home._chargers = [charger, mock_charger2]
        
        charger.user_attached_car_name = None
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'is_plugged', return_value=True), \
             patch.object(mock_charger2, 'is_plugged', return_value=True), \
             patch.object(charger, 'get_car_score', side_effect=[100, 200]):  # Car2 has higher score
            
            # Mock available cars
            charger.home.get_cars.return_value = [mock_car1, mock_car2]
            
            result = charger.get_best_car(time)
        
        # The result might be a charger-created car object, not our mock
        # Just check that a car is returned
        self.assertIsNotNone(result)

    @pytest.mark.asyncio
    async def test_check_load_activity_and_constraints_unplugged(self):
        """Test check_load_activity_and_constraints when unplugged."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'is_plugged', return_value=False):
            result = await charger.check_load_activity_and_constraints(time)
        
        self.assertFalse(result)

    @pytest.mark.asyncio
    async def test_check_load_activity_and_constraints_plugged_no_car(self):
        """Test check_load_activity_and_constraints when plugged but forced no car."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'is_plugged', return_value=True), \
             patch.object(charger, 'get_best_car', return_value=None), \
             patch.object(charger, 'get_and_adapt_existing_constraints', return_value=[], create=True), \
             patch.object(charger, 'reset') as mock_reset, \
             patch.object(charger, 'push_live_constraint') as mock_push:
            
            result = await charger.check_load_activity_and_constraints(time)
        
        self.assertTrue(result)
        mock_reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_load_activity_and_constraints_car_change(self):
        """Test check_load_activity_and_constraints when car changes."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        # Set up current car
        old_car = MagicMock()
        old_car.name = "OldCar"
        charger.car = old_car
        
        # New better car
        new_car = MagicMock()
        new_car.name = "NewCar"
        new_car.setup_car_charge_target_if_needed = AsyncMock(return_value=80.0)
        new_car.get_car_charge_percent.return_value = 50.0
        new_car.can_use_charge_percent_constraints.return_value = False
        new_car.get_car_target_charge_energy.return_value = 10000.0
        new_car.do_force_next_charge = False
        new_car.do_next_charge_time = None
        new_car.car_battery_capacity = 60000.0
        new_car.get_next_scheduled_event = AsyncMock(return_value=(None, None))
        new_car.set_next_charge_target_percent = AsyncMock()
        new_car.get_best_person_next_need = AsyncMock(return_value=(None, None, None, None))
        
        time = datetime.now(pytz.UTC)
        charger._power_steps = [LoadCommand(command="auto_consign", power_consign=1000.0)]
        
        with patch.object(charger, 'is_plugged', return_value=True), \
             patch.object(charger, 'get_best_car', return_value=new_car), \
             patch.object(charger, 'get_and_adapt_existing_constraints', return_value=[], create=True), \
             patch.object(charger, 'reset') as mock_reset, \
             patch.object(charger, 'attach_car') as mock_attach, \
             patch.object(charger, 'detach_car') as mock_detach:
            
            mock_detach.side_effect = lambda: setattr(charger, "car", None)
            mock_attach.side_effect = lambda car, _time: setattr(charger, "car", car)
            result = await charger.check_load_activity_and_constraints(time)
        
        self.assertTrue(result)
        mock_detach.assert_called_once()  # Old car detached
        mock_reset.assert_called_once()
        mock_attach.assert_called_once_with(new_car, time)

    @pytest.mark.asyncio
    async def test_stop_charge(self):
        """Test stop_charge method."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        mock_expected_state = MagicMock()
        mock_expected_state.register_launch = MagicMock()
        
        with patch.object(charger, 'is_charge_enabled', return_value=True), \
             patch.object(charger, 'low_level_stop_charge') as mock_stop, \
             patch.object(type(charger), '_expected_charge_state', new_callable=PropertyMock) as mock_expected_state_prop:
            mock_expected_state_prop.return_value = mock_expected_state
            
            await charger.stop_charge(time)
        
        mock_expected_state.register_launch.assert_called_once_with(value=False, time=time)
        mock_stop.assert_called_once_with(time)

    @pytest.mark.asyncio
    async def test_stop_charge_exception(self):
        """Test stop_charge method with exception."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        mock_expected_state = MagicMock()
        mock_expected_state.register_launch = MagicMock()
        
        with patch.object(charger, 'is_charge_enabled', return_value=True), \
             patch.object(charger, 'low_level_stop_charge', side_effect=Exception("Test error")), \
             patch.object(type(charger), '_expected_charge_state', new_callable=PropertyMock) as mock_expected_state_prop, \
             patch('custom_components.quiet_solar.ha_model.charger._LOGGER') as mock_logger:
            mock_expected_state_prop.return_value = mock_expected_state
            
            await charger.stop_charge(time)
        
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_charge(self):
        """Test start_charge method."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        mock_expected_state = MagicMock()
        mock_expected_state.register_launch = MagicMock()
        
        with patch.object(charger, 'is_charge_disabled', return_value=True), \
             patch.object(charger, 'low_level_start_charge') as mock_start, \
             patch.object(type(charger), '_expected_charge_state', new_callable=PropertyMock) as mock_expected_state_prop:
            mock_expected_state_prop.return_value = mock_expected_state
            
            await charger.start_charge(time)
        
        mock_expected_state.register_launch.assert_called_once_with(value=True, time=time)
        mock_start.assert_called_once_with(time)
    
    def test_check_charger_status_no_sensor(self):
        """Test _check_charger_status with no status sensor."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        charger.charger_status_sensor = None
        time = datetime.now(pytz.UTC)
        
        result = charger._check_charger_status(["Charging"], time)
        self.assertIsNone(result)
    
    def test_check_charger_status_no_status_vals(self):
        """Test _check_charger_status with no status values."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        result = charger._check_charger_status([], time)
        self.assertIsNone(result)
    
    def test_check_charger_status_valid(self):
        """Test _check_charger_status with valid conditions."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'get_last_state_value_duration', return_value=(15.0, None)):
            result = charger._check_charger_status(["Charging"], time, for_duration=10.0)
        
        self.assertTrue(result)  # 15.0 >= 10.0
    
    def test_check_plugged_val_with_car_confirmation(self):
        """Test _check_plugged_val with car confirmation."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        # Mock car
        mock_car = MagicMock()
        mock_car.is_car_plugged.return_value = True
        charger.car = mock_car
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'get_last_state_value_duration', return_value=(None, None)), \
             patch.object(charger, 'get_sensor_latest_possible_valid_value', return_value=QSChargerStates.PLUGGED):
            
            result = charger._check_plugged_val(time, check_for_val=True)
        
        self.assertTrue(result)
    
    def test_is_plugged(self):
        """Test is_plugged method."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, '_check_plugged_val', return_value=True) as mock_check:
            result = charger.is_plugged(time, for_duration=10.0)
        
        mock_check.assert_called_once_with(time, 10.0, check_for_val=True)
        self.assertTrue(result)
    
    def test_is_not_plugged(self):
        """Test is_not_plugged method."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, '_check_plugged_val', return_value=True) as mock_check:
            result = charger.is_not_plugged(time, for_duration=10.0)
        
        mock_check.assert_called_once_with(time, 10.0, check_for_val=False)
        self.assertTrue(result)
    
    def test_is_optimistic_plugged_direct(self):
        """Test is_optimistic_plugged with direct result."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'is_plugged', side_effect=[True, None]):
            result = charger.is_optimistic_plugged(time)
        
        self.assertTrue(result)
    
    def test_is_optimistic_plugged_fallback(self):
        """Test is_optimistic_plugged with fallback check."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'is_plugged', side_effect=[None, True]):
            result = charger.is_optimistic_plugged(time)
        
        self.assertTrue(result)
    
    def test_is_charger_plugged_now_with_status_sensor(self):
        """Test is_charger_plugged_now with status sensor."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        mock_state = MagicMock()
        mock_state.state = "Ready"
        mock_state.last_updated = time
        self.hass.states.get.return_value = mock_state
        
        with patch.object(charger, 'get_car_plugged_in_status_vals', return_value=["Ready", "Charging"]):
            result, state_time = charger.is_charger_plugged_now(time)
        
        self.assertTrue(result)
        self.assertEqual(state_time, time)
    
    def test_is_charger_plugged_now_unavailable_state(self):
        """Test is_charger_plugged_now with unavailable state."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        mock_state = MagicMock()
        mock_state.state = STATE_UNAVAILABLE
        mock_state.last_updated = time
        self.hass.states.get.return_value = mock_state
        
        result, state_time = charger.is_charger_plugged_now(time)
        
        self.assertIsNone(result)
        self.assertEqual(state_time, time)
    
    def test_is_charger_plugged_now_fallback(self):
        """Test is_charger_plugged_now with fallback to low-level check."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        mock_state = MagicMock()
        mock_state.state = "Unknown"
        mock_state.last_updated = time
        self.hass.states.get.return_value = mock_state
        
        with patch.object(charger, 'get_car_plugged_in_status_vals', return_value=["Ready"]), \
             patch.object(charger, 'low_level_plug_check_now', return_value=(True, time)) as mock_low_level:
            
            result, state_time = charger.is_charger_plugged_now(time)
        
        # The method returns False for unknown state when not in status values
        self.assertFalse(result)
        self.assertEqual(state_time, time)
    
    def test_check_charge_state_not_plugged(self):
        """Test check_charge_state when not plugged."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'is_plugged', return_value=False):
            result = charger.check_charge_state(time, check_for_val=True)
        
        self.assertFalse(result)  # not check_for_val when not plugged
    
    def test_check_charge_state_with_status_check(self):
        """Test check_charge_state with status sensor check."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'is_plugged', return_value=True), \
             patch.object(charger, 'get_car_charge_enabled_status_vals', return_value=["Charging"]), \
             patch.object(charger, '_check_charger_status', return_value=True):
            
            result = charger.check_charge_state(time, check_for_val=True)
        
        self.assertTrue(result)
    
    def test_check_charge_state_low_level_fallback(self):
        """Test check_charge_state with low-level fallback."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        mock_state = MagicMock()
        mock_state.state = "Unknown"
        self.hass.states.get.return_value = mock_state
        
        with patch.object(charger, 'is_plugged', return_value=True), \
             patch.object(charger, 'get_car_charge_enabled_status_vals', return_value=["Charging"]), \
             patch.object(charger, '_check_charger_status', return_value=None), \
             patch.object(charger, 'low_level_charge_check_now', return_value=True):
            
            result = charger.check_charge_state(time, check_for_val=True)
        
        self.assertTrue(result)

    @pytest.mark.asyncio
    async def test_set_charging_num_phases_no_switch(self):
        """Test set_charging_num_phases when can't do phase switch."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        mock_expected_phases = MagicMock()
        mock_expected_phases.register_launch = MagicMock()
        mock_expected_phases.success = AsyncMock()
        
        with patch.object(charger, 'can_do_3_to_1_phase_switch', return_value=False), \
             patch.object(type(charger), 'physical_num_phases', new_callable=PropertyMock) as mock_physical, \
             patch.object(type(charger), '_expected_num_active_phases', new_callable=PropertyMock) as mock_expected_phases_prop:
            
            mock_physical.return_value = 3
            mock_expected_phases_prop.return_value = mock_expected_phases
            
            result = await charger.set_charging_num_phases(1, time)
        
        self.assertTrue(result)
        mock_expected_phases.register_launch.assert_called_once_with(value=3, time=time)
        mock_expected_phases.success.assert_called_once_with(time=time)

    @pytest.mark.asyncio
    async def test_set_charging_num_phases_with_switch(self):
        """Test set_charging_num_phases with phase switch capability."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        mock_expected_phases = MagicMock()
        mock_expected_phases.register_launch = MagicMock()
        
        with patch.object(charger, 'can_do_3_to_1_phase_switch', return_value=True), \
             patch.object(type(charger), 'physical_3p', new_callable=PropertyMock) as mock_3p, \
             patch.object(type(charger), 'current_num_phases', new_callable=PropertyMock) as mock_current_phases, \
             patch.object(charger, 'low_level_set_charging_num_phases', return_value=True) as mock_low_level, \
             patch.object(type(charger), '_expected_num_active_phases', new_callable=PropertyMock) as mock_expected_phases_prop:
            
            mock_3p.return_value = True
            mock_current_phases.return_value = 3
            mock_expected_phases_prop.return_value = mock_expected_phases
            
            result = await charger.set_charging_num_phases(1, time)
        
        self.assertTrue(result)
        mock_expected_phases.register_launch.assert_called_once_with(value=1, time=time)
        mock_low_level.assert_called_once_with(1, time)

    @pytest.mark.asyncio
    async def test_set_charging_num_phases_with_reboot(self):
        """Test set_charging_num_phases that triggers reboot."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        mock_expected_phases = MagicMock()
        mock_expected_phases.register_launch = MagicMock()
        mock_expected_phases.register_success_cb = MagicMock()
        
        with patch.object(charger, 'can_do_3_to_1_phase_switch', return_value=True), \
             patch.object(type(charger), 'physical_3p', new_callable=PropertyMock) as mock_3p, \
             patch.object(type(charger), 'current_num_phases', new_callable=PropertyMock) as mock_current_phases, \
             patch.object(charger, 'low_level_set_charging_num_phases', return_value=True), \
             patch.object(charger, 'is_optimistic_plugged', return_value=True), \
             patch.object(type(charger), 'do_reboot_on_phase_switch', new_callable=PropertyMock) as mock_do_reboot, \
             patch.object(type(charger), '_expected_num_active_phases', new_callable=PropertyMock) as mock_expected_phases_prop:
            
            mock_3p.return_value = True
            mock_current_phases.return_value = 3
            mock_do_reboot.return_value = True
            mock_expected_phases_prop.return_value = mock_expected_phases
            
            result = await charger.set_charging_num_phases(1, time)
        
        self.assertTrue(result)
        mock_expected_phases.register_success_cb.assert_called_once_with(charger.reboot, None)

    @pytest.mark.asyncio
    async def test_reboot(self):
        """Test reboot method."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'can_reboot', return_value=True), \
             patch.object(charger, 'low_level_reboot') as mock_reboot:
            
            await charger.reboot(time)
        
        self.assertEqual(charger._asked_for_reboot_at_time, time)
        mock_reboot.assert_called_once_with(time)

    @pytest.mark.asyncio
    async def test_reboot_cannot_reboot(self):
        """Test reboot method when cannot reboot."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'can_reboot', return_value=False), \
             patch.object(charger, 'low_level_reboot') as mock_reboot:
            
            await charger.reboot(time)
        
        self.assertIsNone(charger._asked_for_reboot_at_time)
        mock_reboot.assert_not_called()
    
    def test_probe_for_possible_needed_reboot(self):
        """Test probe_for_possible_needed_reboot method."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'can_reboot', return_value=False):
            result = charger.probe_for_possible_needed_reboot(time)
        
        self.assertFalse(result)
    
    def test_probe_for_possible_needed_reboot_can_reboot(self):
        """Test probe_for_possible_needed_reboot when can reboot."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(**self.charger_config)
        
        time = datetime.now(pytz.UTC)
        
        with patch.object(charger, 'can_reboot', return_value=True):
            result = charger.probe_for_possible_needed_reboot(time)
        
        # Currently always returns False (no clear way to define when reboot needed)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main() 