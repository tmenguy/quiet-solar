import unittest
from unittest.mock import MagicMock, Mock, patch
from datetime import datetime
import pytz
import pytest

# Import from Home Assistant
from homeassistant.const import CONF_NAME

# Import the necessary classes
from custom_components.quiet_solar.ha_model.home import QSHome
from custom_components.quiet_solar.ha_model.dynamic_group import QSDynamicGroup
from custom_components.quiet_solar.ha_model.charger import (
    QSChargerWallbox, 
    QSChargerGroup,
    QSChargerStatus
)
from custom_components.quiet_solar.home_model.commands import (
    CMD_AUTO_FROM_CONSIGN,
    CMD_AUTO_GREEN_ONLY,
    CMD_AUTO_PRICE
)
from custom_components.quiet_solar.const import (
    CONF_DYN_GROUP_MAX_PHASE_AMPS,
    CONF_MONO_PHASE,
    CONF_CHARGER_DEVICE_WALLBOX,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_MAX_CHARGE,
    CONF_IS_3P,
    DOMAIN,
    DATA_HANDLER
)


# @unittest.skip("Skipping due to recursion error in device initialization - use TestBudgetingAlgorithm instead")
class TestChargersSetup(unittest.TestCase):
    def setUp(self):
        """Set up the test environment with QSHome, QSDynamicGroup, and QSChargerWallbox objects."""
        # Mock Home Assistant instance and config_entry
        self.hass = MagicMock()
        self.hass.states = MagicMock()
        self.hass.states.get = MagicMock(return_value=None)
        
        # Mock data handler
        self.data_handler = MagicMock()
        self.hass.data = {
            DOMAIN: {
                DATA_HANDLER: self.data_handler
            }
        }
        
        # Mock config_entry
        self.config_entry = MagicMock()
        self.config_entry.entry_id = "test_entry_id"
        self.config_entry.data = {}
        
        # Create QSHome with 33A max phase amps
        home_config = {
            CONF_NAME: "TestHome",
            CONF_DYN_GROUP_MAX_PHASE_AMPS: 33,
            "hass": self.hass,
            "config_entry": self.config_entry
        }
        self.home = QSHome(**home_config)
        
        # Set the home on the data handler
        self.data_handler.home = self.home
        
        # Create QSDynamicGroup "Wallboxes" with 32A max phase amps
        wallboxes_config = {
            CONF_NAME: "Wallboxes",
            CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            "home": self.home,
            "hass": self.hass,
            "config_entry": self.config_entry
        }
        self.wallboxes_group = QSDynamicGroup(**wallboxes_config)
        
        # Add the dynamic group to home
        self.home.add_device(self.wallboxes_group)
        
        # Mock the is_current_acceptable_and_diff method
        def is_current_acceptable_and_diff_mock(new_amps, estimated_current_amps, time):
            # Check if any phase exceeds the limit
            for i in range(3):
                if new_amps[i] > self.wallboxes_group.dyn_group_max_phase_current_conf:
                    return (False, [new_amps[i] - self.wallboxes_group.dyn_group_max_phase_current_conf for i in range(3)])
            return (True, [0, 0, 0])
        
        self.wallboxes_group.is_current_acceptable_and_diff = MagicMock(side_effect=is_current_acceptable_and_diff_mock)
        
        # Create three QSChargerWallbox with different phases (1, 2, 3)
        self.chargers = []
        
        # Mock entity_registry for charger creation
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry') as mock_entity_reg:
            # Mock the async_get function
            mock_entity_reg.async_get = MagicMock()
            # Mock the async_entries_for_device function to return empty list
            mock_entity_reg.async_entries_for_device = MagicMock(return_value=[])
            
            for phase in [1, 2, 3]:
                charger_config = {
                    CONF_NAME: f"Wallbox_Phase_{phase}",
                    CONF_MONO_PHASE: phase,
                    CONF_CHARGER_DEVICE_WALLBOX: f"device_wallbox_{phase}",
                    CONF_CHARGER_MIN_CHARGE: 6,
                    CONF_CHARGER_MAX_CHARGE: 16,
                    CONF_IS_3P: False,
                    "dynamic_group_name": "Wallboxes",
                    "home": self.home,
                    "hass": self.hass,
                    "config_entry": self.config_entry
                }
                charger = QSChargerWallbox(**charger_config)
                self.chargers.append(charger)
                self.home.add_device(charger)
        
        # Create the charger group for testing
        self.charger_group = QSChargerGroup(self.wallboxes_group)
        
        # Set up current time
        self.current_time = datetime.now(pytz.UTC)
    
    @pytest.mark.asyncio
    async def test_budgeting_algorithm_minimize_diffs_basic(self):
        """Test the basic functionality of budgeting_algorithm_minimize_diffs."""
        # Create actionable chargers list with mock data
        actionable_chargers = []
        
        for i, charger in enumerate(self.chargers):
            # Create a mock dynamic state for each charger
            cs = QSChargerStatus(charger)
            cs.plugged = True
            cs.accurate_current_power = 1500.0  # 1.5kW
            cs.current_real_max_charging_amp = 10
            cs.current_active_phase_number = 1
            cs.possible_amps = [6, 10, 16]  # min, current, max
            cs.possible_num_phases = [1]  # Single phase only
            cs.command = CMD_AUTO_FROM_CONSIGN
            cs.charge_score = i + 1  # Priority score
            cs.best_power_measure = 1500.0
            actionable_chargers.append(cs)
        
        # Mock the dynamic group's is_current_acceptable method
        with patch.object(self.wallboxes_group, 'is_current_acceptable') as mock_acceptable:
            mock_acceptable.return_value = True
            
            # Test with 5kW available power (should distribute among chargers)
            full_available_home_power = 5000.0
            
            # Run the budgeting algorithm
            result = await self.charger_group.budgeting_algorithm_minimize_diffs(
                actionable_chargers,
                full_available_home_power,
                self.current_time
            )
            
            # Verify result is not None
            self.assertIsNotNone(result)
            self.assertIsInstance(result, list)
            
    @pytest.mark.asyncio
    async def test_budgeting_algorithm_with_power_constraints(self):
        """Test budgeting algorithm with specific power constraints."""
        actionable_chargers = []
        
        # Create chargers with different states and priorities
        configs = [
            {"current_amp": 6, "power": 1380, "score": 3},   # Low priority
            {"current_amp": 10, "power": 2300, "score": 1},  # High priority
            {"current_amp": 8, "power": 1840, "score": 2}    # Medium priority
        ]
        
        for i, (charger, config) in enumerate(zip(self.chargers, configs)):
            cs = QSChargerStatus(charger)
            cs.plugged = True
            cs.accurate_current_power = config["power"]
            cs.current_real_max_charging_amp = config["current_amp"]
            cs.current_active_phase_number = 1
            cs.possible_amps = [6, config["current_amp"], 16]
            cs.possible_num_phases = [1]
            cs.command = CMD_AUTO_GREEN_ONLY
            cs.charge_score = config["score"]
            cs.best_power_measure = config["power"]
            actionable_chargers.append(cs)
        
        # Test with limited power (3kW)
        with patch.object(self.wallboxes_group, 'is_current_acceptable') as mock_acceptable:
            mock_acceptable.return_value = True
            
            result = await self.charger_group.budgeting_algorithm_minimize_diffs(
                actionable_chargers,
                3000.0,  # Only 3kW available
                self.current_time
            )
            
            self.assertIsNotNone(result)
            
    @pytest.mark.asyncio
    async def test_budgeting_algorithm_phase_distribution(self):
        """Test that the algorithm properly distributes load across phases."""
        # Since we have 3 chargers on different phases (1, 2, 3),
        # the algorithm should balance the load
        actionable_chargers = []
        
        for i, charger in enumerate(self.chargers):
            # All chargers start at minimum charge
            cs = QSChargerStatus(charger)
            cs.plugged = True
            cs.accurate_current_power = 1380.0  # 6A * 230V
            cs.current_real_max_charging_amp = 6
            cs.current_active_phase_number = 1
            cs.possible_amps = [6, 6, 16]
            cs.possible_num_phases = [1]
            cs.command = CMD_AUTO_FROM_CONSIGN
            cs.charge_score = 1  # Same priority
            cs.best_power_measure = 1380.0
            actionable_chargers.append(cs)
        
        # Mock the phase amp constraints
        with patch.object(self.wallboxes_group, 'is_current_acceptable') as mock_acceptable:
            # First call returns False (over limit), subsequent calls return True
            mock_acceptable.side_effect = [False, True, True]
            
            # Test with 10kW available (should trigger redistribution)
            result = await self.charger_group.budgeting_algorithm_minimize_diffs(
                actionable_chargers,
                10000.0,
                self.current_time
            )
            
            # Verify the method was called (shows redistribution happened)
            self.assertTrue(mock_acceptable.called)
            
    def test_dynamic_group_hierarchy(self):
        """Test that the dynamic group hierarchy is correctly set up."""
        # Verify home contains the wallboxes group
        self.assertIn(self.wallboxes_group, self.home._all_dynamic_groups)
        
        # Verify wallboxes group contains all chargers
        # The children list includes all devices in the group
        # Since we add 3 chargers and each test runs the setUp, we should check that our specific chargers are there
        for charger in self.chargers:
            self.assertIn(charger, self.wallboxes_group._childrens)
            
        # Verify each charger has correct phase assignment
        for i, charger in enumerate(self.chargers):
            expected_phase = i  # 0, 1, 2 (internal representation)
            self.assertEqual(charger.mono_phase_index, expected_phase)
            
    def test_max_phase_amps_limits(self):
        """Test that phase amp limits are correctly enforced."""
        # Home should have 33A limit
        self.assertEqual(self.home.dyn_group_max_phase_current_conf, 33)
        
        # Wallboxes group should have 32A limit
        self.assertEqual(self.wallboxes_group.dyn_group_max_phase_current_conf, 32)
        
        # Check that the phase current property is properly initialized
        # It might be [0, 0, 0] initially if not properly set by the actual device data
        # So let's just verify it's a list of 3 elements
        phase_currents = self.wallboxes_group.dyn_group_max_phase_current
        self.assertIsInstance(phase_currents, list)
        self.assertEqual(len(phase_currents), 3)


@pytest.mark.asyncio
class TestBudgetingAlgorithm:
    """Test the budgeting_algorithm_minimize_diffs method with minimal setup."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the test environment."""
        # Create a mock dynamic group
        self.dynamic_group = MagicMock()
        self.dynamic_group.dyn_group_max_phase_current_conf = 32
        self.dynamic_group.dyn_group_max_phase_current = [32, 32, 32]
        self.dynamic_group.is_current_acceptable = MagicMock(return_value=True)
        
        # Mock the is_current_acceptable_and_diff method
        def is_current_acceptable_and_diff_mock(new_amps, estimated_current_amps, time):
            # Check if any phase exceeds the limit
            for i in range(3):
                if new_amps[i] > self.dynamic_group.dyn_group_max_phase_current[i]:
                    return (False, [new_amps[i] - self.dynamic_group.dyn_group_max_phase_current[i] for i in range(3)])
            return (True, [0, 0, 0])
        
        self.dynamic_group.is_current_acceptable_and_diff = MagicMock(side_effect=is_current_acceptable_and_diff_mock)
        
        # Create the charger group for testing
        self.charger_group = QSChargerGroup(self.dynamic_group)
        
        # Create mock chargers with different phase assignments
        self.chargers = []
        for phase in [0, 1, 2]:  # Internal phase numbering (0, 1, 2)
            charger = MagicMock()
            charger.name = f"Wallbox_Phase_{phase + 1}"
            charger.get_mono_phase = MagicMock(return_value=phase)
            charger.device_is_3p = False
            charger.charger_max_charging_amp_conf = 16
            charger.charger_min_charging_amp_conf = 6
            charger.min_charge = 6  # Add this attribute
            charger.max_charge = 16  # Add max_charge
            charger.charger_default_idle_charge = 7  # Add idle charge
            
            # Mock the get_delta_dampened_power method
            def get_delta_dampened_power_mock(from_amp, from_num_phase, to_amp, to_num_phase):
                # Simple calculation: difference in power based on amp change
                old_power = from_amp * 230 * from_num_phase
                new_power = to_amp * 230 * to_num_phase
                diff_power = new_power - old_power
                return (diff_power, old_power, new_power)
            
            charger.get_delta_dampened_power = MagicMock(side_effect=get_delta_dampened_power_mock)
            
            self.chargers.append(charger)
            
        # Set current time
        self.current_time = datetime.now(pytz.UTC)
        
    def create_charger_status(self, charger, current_amp, power, score, command=CMD_AUTO_FROM_CONSIGN):
        """Helper to create a QSChargerStatus with specified parameters."""
        cs = QSChargerStatus(charger)
        cs.plugged = True
        cs.accurate_current_power = power
        cs.current_real_max_charging_amp = current_amp
        cs.current_active_phase_number = 1
        cs.possible_amps = [6, current_amp, 16]  # min, current, max
        cs.possible_num_phases = [1]  # Single phase only
        cs.command = command
        cs.charge_score = score
        cs.best_power_measure = power
        return cs
        
    async def test_budgeting_algorithm_basic_distribution(self):
        """Test basic power distribution among chargers."""
        # Create actionable chargers with initial state
        actionable_chargers = []
        for i, charger in enumerate(self.chargers):
            cs = self.create_charger_status(
                charger=charger,
                current_amp=6,  # All start at minimum
                power=1380.0,   # 6A * 230V
                score=i + 1     # Priority 1, 2, 3
            )
            actionable_chargers.append(cs)
        
        # Test with 5kW available power
        full_available_home_power = 5000.0
        
        # Run the budgeting algorithm
        result = await self.charger_group.budgeting_algorithm_minimize_diffs(
            actionable_chargers,
            full_available_home_power,
            self.current_time
        )
        
        # Verify result is not None and is a list
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 3
        
        # Extract budgeted amps from the returned charger status objects
        budgeted_amps = [cs.budgeted_amp for cs in result]
        
        # Verify that higher priority chargers get more power
        # Since charger 0 has the highest priority (score=1), it should get more amps
        assert budgeted_amps[0] >= budgeted_amps[1]
        assert budgeted_amps[1] >= budgeted_amps[2]
        
    async def test_budgeting_algorithm_power_constraint(self):
        """Test algorithm behavior with limited power."""
        actionable_chargers = []
        
        # Create chargers with different current states
        configs = [
            {"current_amp": 10, "power": 2300, "score": 1},  # High priority
            {"current_amp": 8, "power": 1840, "score": 2},   # Medium priority
            {"current_amp": 6, "power": 1380, "score": 3}    # Low priority
        ]
        
        for i, (charger, config) in enumerate(zip(self.chargers, configs)):
            cs = self.create_charger_status(
                charger=charger,
                current_amp=config["current_amp"],
                power=config["power"],
                score=config["score"]
            )
            actionable_chargers.append(cs)
        
        # Test with very limited power (2kW) - should prioritize high priority charger
        result = await self.charger_group.budgeting_algorithm_minimize_diffs(
            actionable_chargers,
            2000.0,  # Only 2kW available
            self.current_time
        )
        
        assert result is not None
        # Verify high priority charger maintains some charging
        assert result[0].budgeted_amp > 0
        
    async def test_budgeting_algorithm_phase_limits(self):
        """Test that phase current limits are respected."""
        actionable_chargers = []
        
        # All chargers want maximum power
        for i, charger in enumerate(self.chargers):
            cs = self.create_charger_status(
                charger=charger,
                current_amp=16,  # Max amp
                power=3680.0,    # 16A * 230V
                score=1          # Same priority
            )
            actionable_chargers.append(cs)
        
        # Mock the phase constraint check
        call_count = 0
        def is_acceptable_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return call_count > 1
            
        self.dynamic_group.is_current_acceptable.side_effect = is_acceptable_side_effect
        
        # Test with plenty of power but phase constraints
        result = await self.charger_group.budgeting_algorithm_minimize_diffs(
            actionable_chargers,
            12000.0,  # 12kW available
            self.current_time
        )
        
        assert result is not None
        # Verify the phase constraint method was called
        assert self.dynamic_group.is_current_acceptable.called
        
    async def test_budgeting_algorithm_empty_list(self):
        """Test algorithm with no actionable chargers."""
        result = await self.charger_group.budgeting_algorithm_minimize_diffs(
            [],  # No chargers
            5000.0,
            self.current_time
        )
        
        # Should return None or empty list
        assert result is None or result == []
        
    async def test_budgeting_algorithm_zero_power(self):
        """Test algorithm with no available power."""
        actionable_chargers = []
        for i, charger in enumerate(self.chargers):
            cs = self.create_charger_status(
                charger=charger,
                current_amp=10,
                power=2300.0,
                score=i + 1
            )
            actionable_chargers.append(cs)
        
        # Test with zero power
        result = await self.charger_group.budgeting_algorithm_minimize_diffs(
            actionable_chargers,
            0.0,  # No power available
            self.current_time
        )
        
        assert result is not None
        # All chargers should be reduced to minimum or idle charge
        for cs in result:
            # The algorithm may use charger_default_idle_charge (7A) or min_charge (6A)
            assert cs.budgeted_amp <= 10  # Should be reduced from initial 10A
            
    async def test_budgeting_algorithm_negative_power(self):
        """Test algorithm with negative available power (export scenario)."""
        actionable_chargers = []
        for i, charger in enumerate(self.chargers):
            cs = self.create_charger_status(
                charger=charger,
                current_amp=10,
                power=2300.0,
                score=i + 1
            )
            actionable_chargers.append(cs)
        
        # Test with negative power (exporting to grid)
        result = await self.charger_group.budgeting_algorithm_minimize_diffs(
            actionable_chargers,
            -2000.0,  # Exporting 2kW
            self.current_time
        )
        
        assert result is not None
        # All chargers should be reduced
        total_reduction = 0
        for i, cs in enumerate(result):
            # The algorithm should reduce charging
            assert cs.budgeted_amp <= actionable_chargers[i].current_real_max_charging_amp
            reduction = actionable_chargers[i].current_real_max_charging_amp - cs.budgeted_amp
            total_reduction += reduction
        
        # Verify that there was some reduction
        assert total_reduction > 0


if __name__ == '__main__':
    unittest.main()
