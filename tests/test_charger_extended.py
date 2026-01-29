"""
Extended tests for charger.py to improve coverage.
Focuses on QSChargerStatus, QSChargerGroup, and QSChargerGeneric methods.
"""
import unittest
from unittest.mock import MagicMock, Mock, patch, AsyncMock, PropertyMock
from datetime import datetime, timedelta, time as dt_time
import pytz
import pytest
import asyncio

# Import from Home Assistant
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
    STATE_CMD_RETRY_NUMBER,
    CHARGER_ADAPTATION_WINDOW_S,
    TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_OFF_TO_ON_S,
    TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_ON_TO_OFF_S,
    TIME_OK_BETWEEN_CHANGING_CHARGER_PHASES,
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
    CONF_CAR_CHARGER_MIN_CHARGE,
    CONF_CAR_CHARGER_MAX_CHARGE,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
)
from tests.factories import create_minimal_home_model


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
    home = create_minimal_home_model()
    home.hass = hass
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


class TestQSChargerStatus(unittest.TestCase):
    """Test QSChargerStatus class methods."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(self.hass, self.home)
        self.status = QSChargerStatus(self.charger)

    def test_init(self):
        """Test QSChargerStatus initialization."""
        self.assertEqual(self.status.charger, self.charger)
        self.assertIsNone(self.status.accurate_current_power)
        self.assertIsNone(self.status.secondary_current_power)
        self.assertIsNone(self.status.command)
        self.assertIsNone(self.status.current_real_max_charging_amp)
        self.assertIsNone(self.status.current_active_phase_number)
        self.assertIsNone(self.status.possible_amps)
        self.assertIsNone(self.status.possible_num_phases)
        self.assertIsNone(self.status.budgeted_amp)
        self.assertIsNone(self.status.budgeted_num_phases)
        self.assertEqual(self.status.charge_score, 0)
        self.assertFalse(self.status.can_be_started_and_stopped)
        self.assertFalse(self.status.is_before_battery)
        self.assertFalse(self.status.bump_solar)

    def test_duplicate(self):
        """Test duplicate method creates proper copy."""
        # Set up the status
        self.status.accurate_current_power = 5000.0
        self.status.secondary_current_power = 4500.0
        self.status.command = CMD_AUTO_FROM_CONSIGN
        self.status.current_real_max_charging_amp = 16
        self.status.current_active_phase_number = 3
        self.status.possible_amps = [6, 10, 16, 20, 32]
        self.status.possible_num_phases = [1, 3]
        self.status.budgeted_amp = 20
        self.status.budgeted_num_phases = 3
        self.status.charge_score = 100
        self.status.can_be_started_and_stopped = True
        self.status.is_before_battery = True
        self.status.bump_solar = True

        # Duplicate
        copy = self.status.duplicate()

        # Verify all values are copied
        self.assertEqual(copy.accurate_current_power, 5000.0)
        self.assertEqual(copy.secondary_current_power, 4500.0)
        self.assertEqual(copy.command, CMD_AUTO_FROM_CONSIGN)
        self.assertEqual(copy.current_real_max_charging_amp, 16)
        self.assertEqual(copy.current_active_phase_number, 3)
        self.assertEqual(copy.possible_amps, [6, 10, 16, 20, 32])
        self.assertEqual(copy.possible_num_phases, [1, 3])
        self.assertEqual(copy.budgeted_amp, 20)
        self.assertEqual(copy.budgeted_num_phases, 3)
        self.assertEqual(copy.charge_score, 100)
        self.assertTrue(copy.can_be_started_and_stopped)
        self.assertTrue(copy.is_before_battery)
        self.assertTrue(copy.bump_solar)

        # Verify it's a true copy (modifying original doesn't affect copy)
        self.status.possible_amps.append(40)
        self.assertNotIn(40, copy.possible_amps)

    def test_name_property_with_car(self):
        """Test name property when car is attached."""
        mock_car = MagicMock()
        mock_car.name = "TestCar"
        self.charger.car = mock_car

        name = self.status.name
        self.assertEqual(name, "TestCharger/TestCar")

    def test_name_property_without_car(self):
        """Test name property when no car is attached."""
        self.charger.car = None

        name = self.status.name
        self.assertEqual(name, "TestCharger/NO CAR")

    def test_get_amps_from_values_1_phase(self):
        """Test get_amps_from_values for 1-phase charger."""
        with patch.object(type(self.charger), 'mono_phase_index', new_callable=PropertyMock) as mock_idx:
            mock_idx.return_value = 0
            result = self.status.get_amps_from_values(16, 1)

        self.assertEqual(result, [16, 0.0, 0.0])

    def test_get_amps_from_values_3_phase(self):
        """Test get_amps_from_values for 3-phase charger."""
        result = self.status.get_amps_from_values(16, 3)
        self.assertEqual(result, [16, 16, 16])

    def test_get_current_charging_amps(self):
        """Test get_current_charging_amps method."""
        self.status.current_real_max_charging_amp = 20
        self.status.current_active_phase_number = 3

        result = self.status.get_current_charging_amps()
        self.assertEqual(result, [20, 20, 20])

    def test_get_budget_amps(self):
        """Test get_budget_amps method."""
        self.status.budgeted_amp = 15
        self.status.budgeted_num_phases = 3

        result = self.status.get_budget_amps()
        self.assertEqual(result, [15, 15, 15])

    def test_get_diff_power(self):
        """Test get_diff_power method."""
        with patch.object(self.charger, 'get_delta_dampened_power', return_value=2000.0):
            result = self.status.get_diff_power(10, 3, 15, 3)

        self.assertEqual(result, 2000.0)

    def test_get_diff_power_none_result(self):
        """Test get_diff_power when it returns None."""
        with patch.object(self.charger, 'get_delta_dampened_power', return_value=None):
            result = self.status.get_diff_power(10, 3, 15, 3)

        self.assertIsNone(result)

    def test_get_amps_phase_switch_1_to_3(self):
        """Test get_amps_phase_switch from 1 to 3 phases."""
        self.status.possible_amps = [6, 10, 16, 32]

        with patch.object(type(self.charger), 'mono_phase_index', new_callable=PropertyMock) as mock_idx:
            mock_idx.return_value = 0
            try_amp, to_phase, try_amps = self.status.get_amps_phase_switch(15, 1)

        self.assertEqual(to_phase, 3)
        # 15 // 3 = 5, but min is 6
        self.assertEqual(try_amp, 6)

    def test_get_amps_phase_switch_3_to_1(self):
        """Test get_amps_phase_switch from 3 to 1 phase."""
        self.status.possible_amps = [6, 10, 16, 32]

        with patch.object(type(self.charger), 'mono_phase_index', new_callable=PropertyMock) as mock_idx:
            mock_idx.return_value = 0
            try_amp, to_phase, try_amps = self.status.get_amps_phase_switch(10, 3)

        self.assertEqual(to_phase, 1)
        # 10 * 3 = 30, max is 32
        self.assertEqual(try_amp, 30)

    def test_can_change_budget_increase_from_min(self):
        """Test can_change_budget for increasing from minimum."""
        self.status.budgeted_amp = 6
        self.status.budgeted_num_phases = 3
        self.status.possible_amps = [6, 10, 16, 32]
        self.status.possible_num_phases = [3]

        next_amp, next_phases = self.status.can_change_budget(
            allow_state_change=True,
            allow_phase_change=False,
            increase=True
        )

        self.assertEqual(next_amp, 7)
        self.assertEqual(next_phases, 3)

    def test_can_change_budget_decrease_to_min(self):
        """Test can_change_budget for decreasing to minimum."""
        self.status.budgeted_amp = 7
        self.status.budgeted_num_phases = 3
        self.status.possible_amps = [6, 10, 16, 32]
        self.status.possible_num_phases = [3]

        next_amp, next_phases = self.status.can_change_budget(
            allow_state_change=True,
            allow_phase_change=False,
            increase=False
        )

        self.assertEqual(next_amp, 6)
        self.assertEqual(next_phases, 3)

    def test_can_change_budget_decrease_to_zero(self):
        """Test can_change_budget for decreasing to zero (stop)."""
        self.status.budgeted_amp = 6
        self.status.budgeted_num_phases = 3
        self.status.possible_amps = [0, 6, 10, 16, 32]  # 0 means can stop
        self.status.possible_num_phases = [3]

        next_amp, next_phases = self.status.can_change_budget(
            allow_state_change=True,
            allow_phase_change=False,
            increase=False
        )

        self.assertEqual(next_amp, 0)

    def test_can_change_budget_no_state_change_at_min(self):
        """Test can_change_budget when at minimum and state change not allowed."""
        self.status.budgeted_amp = 6
        self.status.budgeted_num_phases = 3
        self.status.possible_amps = [0, 6, 10, 16, 32]
        self.status.possible_num_phases = [3]

        next_amp, next_phases = self.status.can_change_budget(
            allow_state_change=False,  # Can't change state
            allow_phase_change=False,
            increase=False
        )

        self.assertIsNone(next_amp)

    def test_can_change_budget_at_max(self):
        """Test can_change_budget when at maximum."""
        self.status.budgeted_amp = 32
        self.status.budgeted_num_phases = 3
        self.status.possible_amps = [6, 10, 16, 32]
        self.status.possible_num_phases = [3]

        next_amp, next_phases = self.status.can_change_budget(
            allow_state_change=True,
            allow_phase_change=False,
            increase=True
        )

        self.assertIsNone(next_amp)  # Can't increase beyond max

    def test_can_change_budget_none_budgeted(self):
        """Test can_change_budget when budgeted_amp is None."""
        self.status.budgeted_amp = None
        self.status.budgeted_num_phases = 3
        self.status.possible_amps = [6, 10, 16, 32]
        self.status.possible_num_phases = [3]

        next_amp, next_phases = self.status.can_change_budget(
            allow_state_change=True,
            allow_phase_change=False,
            increase=True
        )

        self.assertIsNone(next_amp)

    def test_get_consign_amps_values_with_power_consign(self):
        """Test get_consign_amps_values with power consign."""
        self.status.command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=5000)
        self.status.current_active_phase_number = 3

        # Mock car
        mock_car = MagicMock()
        mock_car.get_charge_power_per_phase_A.return_value = ([1000, 2000, 3000, 4000, 5000, 6000] * 6, None, None)
        self.charger.car = mock_car

        with patch.object(self.charger, 'can_do_3_to_1_phase_switch', return_value=False), \
             patch.object(type(self.charger), 'physical_3p', new_callable=PropertyMock) as mock_3p, \
             patch.object(self.charger, '_get_amps_from_power_steps', return_value=16), \
             patch.object(type(self.charger), 'min_charge', new_callable=PropertyMock) as mock_min, \
             patch.object(type(self.charger), 'max_charge', new_callable=PropertyMock) as mock_max:

            mock_3p.return_value = True
            mock_min.return_value = 6
            mock_max.return_value = 32

            num_phases, consign_amp = self.status.get_consign_amps_values(consign_is_minimum=True)

        self.assertEqual(consign_amp, 16)

    def test_get_consign_amps_values_no_power_consign_minimum(self):
        """Test get_consign_amps_values without power consign (minimum)."""
        self.status.command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=None)

        with patch.object(type(self.charger), 'min_charge', new_callable=PropertyMock) as mock_min, \
             patch.object(type(self.charger), 'max_charge', new_callable=PropertyMock) as mock_max:
            mock_min.return_value = 6
            mock_max.return_value = 32

            num_phases, consign_amp = self.status.get_consign_amps_values(consign_is_minimum=True)

        self.assertEqual(consign_amp, 6)

    def test_get_consign_amps_values_no_power_consign_maximum(self):
        """Test get_consign_amps_values without power consign (maximum)."""
        self.status.command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=None)

        with patch.object(type(self.charger), 'min_charge', new_callable=PropertyMock) as mock_min, \
             patch.object(type(self.charger), 'max_charge', new_callable=PropertyMock) as mock_max:
            mock_min.return_value = 6
            mock_max.return_value = 32

            num_phases, consign_amp = self.status.get_consign_amps_values(consign_is_minimum=False)

        self.assertEqual(consign_amp, 32)


class TestQSChargerGroupBasics(unittest.TestCase):
    """Test QSChargerGroup class basic methods."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)

        # Create dynamic group
        self.dynamic_group = MagicMock(spec=QSDynamicGroup)
        self.dynamic_group.name = "TestGroup"
        self.dynamic_group.home = self.home
        self.dynamic_group._childrens = []
        self.dynamic_group.dyn_group_max_phase_current = [32, 32, 32]
        self.dynamic_group.dyn_group_max_phase_current_conf = 32

        # Create charger group
        self.charger_group = QSChargerGroup(self.dynamic_group)

    def test_init(self):
        """Test QSChargerGroup initialization."""
        self.assertEqual(self.charger_group.dynamic_group, self.dynamic_group)
        self.assertEqual(self.charger_group.name, "TestGroup")
        self.assertEqual(self.charger_group.home, self.home)
        self.assertEqual(self.charger_group._chargers, [])
        self.assertIsNone(self.charger_group.know_reduced_state)
        self.assertIsNone(self.charger_group.know_reduced_state_real_power)

    def test_dampening_power_value_for_car_consumption_none(self):
        """Test dampening_power_value_for_car_consumption with None."""
        result = self.charger_group.dampening_power_value_for_car_consumption(None)
        self.assertIsNone(result)

    def test_dampening_power_value_for_car_consumption_below_threshold(self):
        """Test dampening_power_value_for_car_consumption below threshold."""
        self.charger_group.charger_consumption_W = 100.0
        result = self.charger_group.dampening_power_value_for_car_consumption(50.0)
        self.assertEqual(result, 0.0)

    def test_dampening_power_value_for_car_consumption_above_threshold(self):
        """Test dampening_power_value_for_car_consumption above threshold."""
        self.charger_group.charger_consumption_W = 100.0
        result = self.charger_group.dampening_power_value_for_car_consumption(500.0)
        self.assertEqual(result, 500.0)

    def test_get_budget_diffs(self):
        """Test get_budget_diffs method."""
        # Create mock charger statuses
        cs1 = MagicMock()
        cs1.get_current_charging_amps.return_value = [10, 0, 0]
        cs1.get_budget_amps.return_value = [15, 0, 0]
        cs1.get_diff_power.return_value = 1000.0

        cs2 = MagicMock()
        cs2.get_current_charging_amps.return_value = [0, 10, 0]
        cs2.get_budget_amps.return_value = [0, 12, 0]
        cs2.get_diff_power.return_value = 500.0

        actionable_chargers = [cs1, cs2]

        diff_power, new_sum_amps, current_amps = self.charger_group.get_budget_diffs(actionable_chargers)

        self.assertEqual(diff_power, 1500.0)
        self.assertEqual(new_sum_amps, [15, 12, 0])
        self.assertEqual(current_amps, [10, 10, 0])


class TestQSChargerGroupAsync(unittest.IsolatedAsyncioTestCase):
    """Test QSChargerGroup async methods."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)

        # Create dynamic group
        self.dynamic_group = MagicMock(spec=QSDynamicGroup)
        self.dynamic_group.name = "TestGroup"
        self.dynamic_group.home = self.home
        self.dynamic_group._childrens = []
        self.dynamic_group.dyn_group_max_phase_current = [32, 32, 32]
        self.dynamic_group.dyn_group_max_phase_current_conf = 32
        self.dynamic_group.is_current_acceptable = MagicMock(return_value=True)
        self.dynamic_group.is_current_acceptable_and_diff = MagicMock(return_value=(True, [0, 0, 0]))

        # Create charger group
        self.charger_group = QSChargerGroup(self.dynamic_group)

    async def test_ensure_correct_state_no_chargers(self):
        """Test ensure_correct_state with no chargers."""
        self.charger_group._chargers = []
        time = datetime.now(pytz.UTC)

        actionable, verified_time = await self.charger_group.ensure_correct_state(time)

        self.assertEqual(actionable, [])
        self.assertIsNone(verified_time)

    async def test_ensure_correct_state_disabled_charger(self):
        """Test ensure_correct_state with disabled charger."""
        mock_charger = MagicMock()
        mock_charger.qs_enable_device = False
        self.charger_group._chargers = [mock_charger]

        time = datetime.now(pytz.UTC)
        actionable, verified_time = await self.charger_group.ensure_correct_state(time)

        self.assertEqual(actionable, [])

    async def test_budgeting_algorithm_minimize_diffs_empty(self):
        """Test budgeting_algorithm_minimize_diffs with empty list."""
        time = datetime.now(pytz.UTC)

        result, should_reset, done_reset = await self.charger_group.budgeting_algorithm_minimize_diffs(
            [], 5000.0, 5000.0, False, time
        )

        self.assertTrue(result)

    async def test_do_prepare_budgets_for_algo(self):
        """Test _do_prepare_budgets_for_algo method."""
        # Create mock charger status
        mock_charger = MagicMock()
        mock_charger.mono_phase_index = 0

        cs = MagicMock()
        cs.charger = mock_charger
        cs.current_real_max_charging_amp = 10
        cs.current_active_phase_number = 3
        cs.possible_amps = [6, 10, 16, 32]
        cs.possible_num_phases = [1, 3]
        cs.get_amps_from_values = MagicMock(return_value=[6, 6, 6])

        actionable_chargers = [cs]

        current_amps, has_phase_changes, mandatory_amps = await self.charger_group._do_prepare_budgets_for_algo(
            actionable_chargers, do_reset_allocation=True
        )

        # With reset allocation, should use minimum values
        self.assertEqual(cs.budgeted_amp, 6)  # possible_amps[0]
        self.assertEqual(cs.budgeted_num_phases, 1)  # min of possible_num_phases
        self.assertTrue(has_phase_changes)

    async def test_do_prepare_budgets_for_algo_no_reset(self):
        """Test _do_prepare_budgets_for_algo without reset allocation."""
        mock_charger = MagicMock()
        mock_charger.mono_phase_index = 0

        cs = MagicMock()
        cs.charger = mock_charger
        cs.current_real_max_charging_amp = 10
        cs.current_active_phase_number = 3
        cs.possible_amps = [6, 10, 16, 32]
        cs.possible_num_phases = [1, 3]
        cs.get_amps_from_values = MagicMock(return_value=[10, 10, 10])

        actionable_chargers = [cs]

        current_amps, has_phase_changes, mandatory_amps = await self.charger_group._do_prepare_budgets_for_algo(
            actionable_chargers, do_reset_allocation=False
        )

        # Without reset, should keep current values
        self.assertEqual(cs.budgeted_amp, 10)
        self.assertEqual(cs.budgeted_num_phases, 3)


class TestQSChargerGenericExtended(unittest.TestCase):
    """Extended tests for QSChargerGeneric."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(self.hass, self.home)

    def test_reset(self):
        """Test reset method."""
        self.charger.car = MagicMock()
        self.charger.reset()

        self.assertIsNone(self.charger.car)

    def test_is_in_state_reset_true(self):
        """Test is_in_state_reset when state is reset."""
        self.charger._inner_expected_charge_state = None

        result = self.charger.is_in_state_reset()
        self.assertTrue(result)

    def test_is_in_state_reset_false(self):
        """Test is_in_state_reset when state is not reset."""
        self.charger._expected_charge_state.value = True
        self.charger._expected_amperage.value = 16
        self.charger._expected_num_active_phases.value = 3

        result = self.charger.is_in_state_reset()
        self.assertFalse(result)

    def test_dampening_power_value_for_car_consumption_none(self):
        """Test dampening_power_value_for_car_consumption with None."""
        result = self.charger.dampening_power_value_for_car_consumption(None)
        self.assertIsNone(result)

    def test_dampening_power_value_for_car_consumption_below_threshold(self):
        """Test dampening_power_value_for_car_consumption below charger consumption."""
        self.charger.charger_consumption_W = 100
        result = self.charger.dampening_power_value_for_car_consumption(50.0)
        self.assertEqual(result, 0.0)

    def test_dampening_power_value_for_car_consumption_above_threshold(self):
        """Test dampening_power_value_for_car_consumption above charger consumption."""
        self.charger.charger_consumption_W = 100
        result = self.charger.dampening_power_value_for_car_consumption(500.0)
        self.assertEqual(result, 500.0)

    def test_get_car_options_plugged(self):
        """Test get_car_options when charger is plugged."""
        # Mock cars
        mock_car1 = MagicMock()
        mock_car1.name = "Car1"
        mock_car2 = MagicMock()
        mock_car2.name = "Car2"
        self.home._cars = [mock_car1, mock_car2]

        with patch.object(self.charger, 'is_optimistic_plugged', return_value=True):
            options = self.charger.get_car_options()

        self.assertIn("Car1", options)
        self.assertIn("Car2", options)
        self.assertIn(CHARGER_NO_CAR_CONNECTED, options)

    def test_get_car_options_unplugged(self):
        """Test get_car_options when charger is unplugged."""
        with patch.object(self.charger, 'is_optimistic_plugged', return_value=False):
            options = self.charger.get_car_options()

        self.assertEqual(options, [CHARGER_NO_CAR_CONNECTED])

    def test_get_current_selected_car_option_user_attached(self):
        """Test get_current_selected_car_option with user attached car."""
        self.charger.user_attached_car_name = "MyCar"

        result = self.charger.get_current_selected_car_option()
        self.assertEqual(result, "MyCar")

    def test_get_current_selected_car_option_no_car(self):
        """Test get_current_selected_car_option with no car."""
        self.charger.user_attached_car_name = None
        self.charger.car = None

        result = self.charger.get_current_selected_car_option()
        self.assertIsNone(result)

    def test_get_current_selected_car_option_attached_car(self):
        """Test get_current_selected_car_option with attached car."""
        self.charger.user_attached_car_name = None
        mock_car = MagicMock()
        mock_car.name = "AttachedCar"
        self.charger.car = mock_car

        result = self.charger.get_current_selected_car_option()
        self.assertEqual(result, "AttachedCar")

    def test_default_charge_time_getter_with_car(self):
        """Test default_charge_time getter with car."""
        mock_car = MagicMock()
        mock_car.default_charge_time = dt_time(8, 0)
        self.charger.car = mock_car

        result = self.charger.default_charge_time
        self.assertEqual(result, dt_time(8, 0))

    def test_default_charge_time_getter_no_car(self):
        """Test default_charge_time getter without car."""
        self.charger.car = None

        result = self.charger.default_charge_time
        self.assertIsNone(result)

    def test_default_charge_time_setter_with_car(self):
        """Test default_charge_time setter with car."""
        mock_car = MagicMock()
        self.charger.car = mock_car

        self.charger.default_charge_time = dt_time(9, 30)
        self.assertEqual(mock_car.default_charge_time, dt_time(9, 30))

    def test_can_add_default_charge_true(self):
        """Test can_add_default_charge when car can add charge."""
        mock_car = MagicMock()
        mock_car.can_add_default_charge.return_value = True
        self.charger.car = mock_car

        result = self.charger.can_add_default_charge()
        self.assertTrue(result)

    def test_can_add_default_charge_no_car(self):
        """Test can_add_default_charge without car."""
        self.charger.car = None

        result = self.charger.can_add_default_charge()
        self.assertFalse(result)

    def test_can_force_a_charge_now_true(self):
        """Test can_force_a_charge_now when car can force charge."""
        mock_car = MagicMock()
        mock_car.can_force_a_charge_now.return_value = True
        self.charger.car = mock_car

        result = self.charger.can_force_a_charge_now()
        self.assertTrue(result)

    def test_can_force_a_charge_now_no_car(self):
        """Test can_force_a_charge_now without car."""
        self.charger.car = None

        result = self.charger.can_force_a_charge_now()
        self.assertFalse(result)

    def test_qs_bump_solar_charge_priority_getter_with_car(self):
        """Test qs_bump_solar_charge_priority getter with car."""
        mock_car = MagicMock()
        mock_car.qs_bump_solar_charge_priority = True
        self.charger.car = mock_car

        result = self.charger.qs_bump_solar_charge_priority
        self.assertTrue(result)

    def test_qs_bump_solar_charge_priority_getter_no_car(self):
        """Test qs_bump_solar_charge_priority getter without car."""
        self.charger.car = None

        result = self.charger.qs_bump_solar_charge_priority
        self.assertFalse(result)

    def test_get_normalized_score_no_car(self):
        """Test get_normalized_score without car."""
        self.charger.car = None
        time = datetime.now(pytz.UTC)

        result = self.charger.get_normalized_score(None, time)
        self.assertEqual(result, 0.0)

    def test_get_normalized_score_with_car_no_constraint(self):
        """Test get_normalized_score with car but no constraint."""
        mock_car = MagicMock()
        mock_car.get_car_charge_percent.return_value = 50.0
        mock_car.car_battery_capacity = 60000  # 60kWh
        self.charger.car = mock_car

        time = datetime.now(pytz.UTC)
        result = self.charger.get_normalized_score(None, time)

        # Should have a non-zero score based on battery level
        self.assertGreater(result, 0.0)

    def test_get_normalized_score_with_fast_charge(self):
        """Test get_normalized_score with as_fast_as_possible constraint."""
        mock_car = MagicMock()
        mock_car.get_car_charge_percent.return_value = 20.0
        mock_car.car_battery_capacity = 80000
        self.charger.car = mock_car

        mock_constraint = MagicMock()
        mock_constraint.as_fast_as_possible = True
        mock_constraint.is_before_battery = False
        mock_constraint.end_of_constraint = datetime.now(pytz.UTC) + timedelta(hours=2)
        mock_constraint.is_mandatory = True

        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'compute_is_before_battery', return_value=False), \
             patch.object(type(self.charger), 'qs_bump_solar_charge_priority', new_callable=PropertyMock) as mock_bump:
            mock_bump.return_value = False
            result = self.charger.get_normalized_score(mock_constraint, time)

        # Fast charge should have a high score
        self.assertGreater(result, 0.0)


class TestQSStateCmdExtended(unittest.IsolatedAsyncioTestCase):
    """Extended tests for QSStateCmd."""

    def test_is_ok_to_launch_first_time(self):
        """Test is_ok_to_launch first time."""
        state_cmd = QSStateCmd()
        time = datetime.now(pytz.UTC)

        result = state_cmd.is_ok_to_launch(True, time)
        self.assertTrue(result)
        self.assertEqual(state_cmd._num_launched, 0)

    def test_register_launch(self):
        """Test register_launch method."""
        state_cmd = QSStateCmd()
        time = datetime.now(pytz.UTC)

        state_cmd.register_launch(True, time)

        self.assertEqual(state_cmd._num_launched, 1)
        self.assertEqual(state_cmd.last_time_set, time)
        self.assertTrue(state_cmd.value)

    def test_is_ok_to_launch_after_max_retries(self):
        """Test is_ok_to_launch after max retries."""
        state_cmd = QSStateCmd()
        time = datetime.now(pytz.UTC)

        # Register max retries + 1
        for _ in range(STATE_CMD_RETRY_NUMBER + 2):
            state_cmd.register_launch(True, time)

        result = state_cmd.is_ok_to_launch(True, time)
        self.assertFalse(result)

    def test_is_ok_to_launch_after_retry_timeout(self):
        """Test is_ok_to_launch after retry timeout."""
        state_cmd = QSStateCmd(command_retries_s=10.0)
        time1 = datetime.now(pytz.UTC)
        time2 = time1 + timedelta(seconds=15)

        state_cmd.register_launch(True, time1)

        # After timeout, should be ok to launch again
        result = state_cmd.is_ok_to_launch(True, time2)
        self.assertTrue(result)

    async def test_success_resets_counters(self):
        """Test success method resets counters."""
        state_cmd = QSStateCmd()
        time = datetime.now(pytz.UTC)

        # Set up some state
        state_cmd._num_launched = 3
        state_cmd.last_time_set = time

        await state_cmd.success(time)

        self.assertEqual(state_cmd._num_launched, 0)
        self.assertIsNone(state_cmd.last_time_set)
        self.assertEqual(state_cmd.first_time_success, time)
        self.assertEqual(state_cmd.last_ping_time_success, time)

    async def test_success_with_callback(self):
        """Test success method calls callback."""
        state_cmd = QSStateCmd()
        time = datetime.now(pytz.UTC)

        callback_called = False
        callback_time = None

        async def test_callback(time, **kwargs):
            nonlocal callback_called, callback_time
            callback_called = True
            callback_time = time

        state_cmd.register_success_cb(test_callback, {"key": "value"})
        await state_cmd.success(time)

        self.assertTrue(callback_called)
        self.assertEqual(callback_time, time)

        # Callback should be cleared after success
        self.assertIsNone(state_cmd.on_success_action_cb)
        self.assertIsNone(state_cmd.on_success_action_cb_kwargs)


class TestWallboxChargerStatus(unittest.TestCase):
    """Test WallboxChargerStatus enum values."""

    def test_charging_status(self):
        """Test that CHARGING status is defined."""
        self.assertEqual(WallboxChargerStatus.CHARGING, "Charging")

    def test_disconnected_status(self):
        """Test that DISCONNECTED status is defined."""
        self.assertEqual(WallboxChargerStatus.DISCONNECTED, "Disconnected")

    def test_ready_status(self):
        """Test that READY status is defined."""
        self.assertEqual(WallboxChargerStatus.READY, "Ready")

    def test_paused_status(self):
        """Test that PAUSED status is defined."""
        self.assertEqual(WallboxChargerStatus.PAUSED, "Paused")


class TestOCPPChargePointStatus(unittest.TestCase):
    """Test QSOCPPv16v201ChargePointStatus enum values."""

    def test_available_status(self):
        """Test that available status is defined."""
        self.assertEqual(QSOCPPv16v201ChargePointStatus.available, "Available")

    def test_charging_status(self):
        """Test that charging status is defined."""
        self.assertEqual(QSOCPPv16v201ChargePointStatus.charging, "Charging")

    def test_suspended_evse_status(self):
        """Test that suspended_evse status is defined."""
        self.assertEqual(QSOCPPv16v201ChargePointStatus.suspended_evse, "SuspendedEVSE")

    def test_suspended_ev_status(self):
        """Test that suspended_ev status is defined."""
        self.assertEqual(QSOCPPv16v201ChargePointStatus.suspended_ev, "SuspendedEV")

    def test_faulted_status(self):
        """Test that faulted status is defined."""
        self.assertEqual(QSOCPPv16v201ChargePointStatus.faulted, "Faulted")


class TestQSChargerGenericCarSelection(unittest.TestCase):
    """Test car selection methods in QSChargerGeneric."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(self.hass, self.home)

    def test_get_best_car_user_attached(self):
        """Test get_best_car with user attached car."""
        mock_car = MagicMock()
        mock_car.name = "UserCar"
        mock_car.user_attached_charger_name = None

        self.home.get_car_by_name = MagicMock(return_value=mock_car)
        self.charger.user_attached_car_name = "UserCar"

        time = datetime.now(pytz.UTC)
        result = self.charger.get_best_car(time)

        self.assertEqual(result, mock_car)

    def test_get_best_car_no_car_connected(self):
        """Test get_best_car when user set CHARGER_NO_CAR_CONNECTED."""
        self.charger.user_attached_car_name = CHARGER_NO_CAR_CONNECTED

        time = datetime.now(pytz.UTC)
        result = self.charger.get_best_car(time)

        self.assertIsNone(result)

    def test_get_car_score_zero_for_wrong_car(self):
        """Test get_car_score returns 0 for wrong car when another is attached."""
        mock_car = MagicMock()
        mock_car.name = "WrongCar"
        mock_car.car_is_invited = False

        attached_car = MagicMock()
        attached_car.name = "AttachedCar"

        self.home.get_car_by_name = MagicMock(return_value=attached_car)
        self.charger.user_attached_car_name = "AttachedCar"

        time = datetime.now(pytz.UTC)
        cache = {}

        result = self.charger.get_car_score(mock_car, time, cache)
        self.assertEqual(result, 0.0)


class TestQSChargerGenericConstraints(unittest.TestCase):
    """Test constraint-related methods in QSChargerGeneric."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(self.hass, self.home)

    def test_compute_is_before_battery_no_constraint(self):
        """Test compute_is_before_battery with no constraint."""
        result = self.charger.compute_is_before_battery(None)
        self.assertFalse(result)

    def test_compute_is_before_battery_with_constraint(self):
        """Test compute_is_before_battery with constraint."""
        mock_constraint = MagicMock()
        mock_constraint.is_before_battery = True

        time = datetime.now(pytz.UTC)

        with patch.object(self.charger, 'get_current_active_constraint', return_value=None):
            result = self.charger.compute_is_before_battery(mock_constraint, time)

        self.assertTrue(result)


class TestQSChargerGroupApplyBudgets(unittest.IsolatedAsyncioTestCase):
    """Test apply_budgets and apply_budget_strategy methods."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)

        self.dynamic_group = MagicMock(spec=QSDynamicGroup)
        self.dynamic_group.name = "TestGroup"
        self.dynamic_group.home = self.home
        self.dynamic_group._childrens = []
        self.dynamic_group.dyn_group_max_phase_current = [32, 32, 32]
        self.dynamic_group.is_current_acceptable = MagicMock(return_value=True)

        self.charger_group = QSChargerGroup(self.dynamic_group)

    async def test_apply_budgets_empty_list(self):
        """Test apply_budgets with empty list."""
        time = datetime.now(pytz.UTC)

        self.charger_group.remaining_budget_to_apply = ["sentinel"]
        self.charger_group.know_reduced_state = {"charger": "state"}

        await self.charger_group.apply_budgets([], [], time, check_charger_state=False)
        self.assertEqual(self.charger_group.remaining_budget_to_apply, ["sentinel"])
        self.assertEqual(self.charger_group.know_reduced_state, {"charger": "state"})

    async def test_apply_budget_strategy_empty_list(self):
        """Test apply_budget_strategy with empty list."""
        time = datetime.now(pytz.UTC)

        self.charger_group.remaining_budget_to_apply = ["sentinel"]
        self.charger_group.know_reduced_state = {"charger": "state"}

        await self.charger_group.apply_budget_strategy([], None, time)
        self.dynamic_group.is_current_acceptable.assert_not_called()
        self.assertEqual(self.charger_group.remaining_budget_to_apply, ["sentinel"])
        self.assertEqual(self.charger_group.know_reduced_state, {"charger": "state"})

    async def test_apply_budget_strategy_sets_know_reduced_state(self):
        """Test apply_budget_strategy sets know_reduced_state."""
        mock_charger = MagicMock()
        mock_charger._expected_charge_state = MagicMock()
        mock_charger._expected_amperage = MagicMock()
        mock_charger._expected_num_active_phases = MagicMock()
        mock_charger._ensure_correct_state = AsyncMock()
        mock_charger.min_charge = 6
        mock_charger.max_charge = 32
        mock_charger.charger_default_idle_charge = 7
        mock_charger.num_on_off = 0

        cs = MagicMock()
        cs.charger = mock_charger
        cs.current_real_max_charging_amp = 10
        cs.current_active_phase_number = 3
        cs.budgeted_amp = 15
        cs.budgeted_num_phases = 3
        cs.get_current_charging_amps.return_value = [10, 10, 10]
        cs.get_budget_amps.return_value = [15, 15, 15]

        time = datetime.now(pytz.UTC)

        await self.charger_group.apply_budget_strategy([cs], 5000.0, time)

        self.assertIsNotNone(self.charger_group.know_reduced_state)
        self.assertEqual(self.charger_group.know_reduced_state_real_power, 5000.0)


if __name__ == '__main__':
    unittest.main()
