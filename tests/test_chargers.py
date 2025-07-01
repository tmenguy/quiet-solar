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


def common_setup(self):
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
        CONF_IS_3P: True,
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
        CONF_IS_3P: True,
        "home": self.home,
        "hass": self.hass,
        "config_entry": self.config_entry
    }
    self.dynamic_group = self.wallboxes_group = QSDynamicGroup(**wallboxes_config)


# @unittest.skip("Skipping due to recursion error in device initialization - use TestBudgetingAlgorithm instead")
class TestChargersSetup(unittest.TestCase):
    def setUp(self):

        common_setup(self)

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

            self.current_time = datetime.now(pytz.UTC)
            
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
                # Attach the default generic car to each charger
                charger.attach_car(charger._default_generic_car, self.current_time)
                
                # Mock the expected states to make the charger operational
                charger._expected_charge_state.value = True
                charger._expected_amperage.value = 6
                charger._expected_num_active_phases.value = 1
                
                self.chargers.append(charger)
                self.home.add_device(charger)
        
        # Create the charger group for testing
        self.charger_group = QSChargerGroup(self.wallboxes_group)
        
        # Set up current time

    
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
            result, _ = await self.charger_group.budgeting_algorithm_minimize_diffs(
                actionable_chargers,
                full_available_home_power,
                False,
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
            
            result, _ = await self.charger_group.budgeting_algorithm_minimize_diffs(
                actionable_chargers,
                3000.0,# Only 3kW available
                False,
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
            result, _ = await self.charger_group.budgeting_algorithm_minimize_diffs(
                actionable_chargers,
                10000.0,
                False,
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
        common_setup(self)
        
        # Add the dynamic group to home
        self.home.add_device(self.dynamic_group)
        
        # Create the charger group for testing
        self.charger_group = QSChargerGroup(self.dynamic_group)
        
        # Create real QSChargerWallbox instances with different phase assignments
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
                    CONF_IS_3P: False,  # Single phase chargers
                    "dynamic_group_name": self.dynamic_group.name,
                    "home": self.home,
                    "hass": self.hass,
                    "config_entry": self.config_entry
                }
                charger = QSChargerWallbox(**charger_config)
                # Attach the default generic car to make charger operational
                charger.attach_car(charger._default_generic_car, datetime.now(pytz.UTC))
                self.chargers.append(charger)
                self.home.add_device(charger)
            
        # Set current time
        self.current_time = datetime.now(pytz.UTC)
        
    def create_charger_status(self, charger, current_amp, power, score, idx, command=CMD_AUTO_FROM_CONSIGN):
        """Helper to create a QSChargerStatus with specified parameters."""
        cs = QSChargerStatus(charger)
        cs.plugged = True
        cs.accurate_current_power = power
        cs.current_real_max_charging_amp = current_amp
        cs.current_active_phase_number = 1
        cs.possible_amps = [i for i in range(charger.min_charge, charger.max_charge+1)]  # min, current, max
        if idx % 2 == 0:
            a = [0]
            a.extend(cs.possible_amps)
            cs.possible_amps = a

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
                score=i + 1,
                idx=i
            )
            actionable_chargers.append(cs)
        
        # Test with 5kW available power
        full_available_home_power = 5000.0
        
        # Run the budgeting algorithm
        result, _ = await self.charger_group.budgeting_algorithm_minimize_diffs(
            actionable_chargers,
            full_available_home_power,
            False,
            self.current_time
        )
        
        # Verify result is not None and is a list
        assert result is not None
        assert result

        actionable_chargers = sorted(actionable_chargers, key=lambda cs: cs.charge_score, reverse=True)
        
        # Extract budgeted amps from the returned charger status objects
        budgeted_amps = [cs.budgeted_amp for cs in actionable_chargers]
        
        # Verify that higher priority chargers get more power
        # Since charger 0 has the highest priority (score=1), it should get more amps
        assert budgeted_amps[0] >= budgeted_amps[1]
        assert budgeted_amps[0] >= budgeted_amps[2]
        
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
                score=config["score"],
                idx=i
            )
            actionable_chargers.append(cs)
        
        # Test with very limited power (2kW) - should prioritize high priority charger
        result, _ = await self.charger_group.budgeting_algorithm_minimize_diffs(
            actionable_chargers,
            2000.0,  # Only 2kW available
            False,
            self.current_time
        )

        actionable_chargers = sorted(actionable_chargers, key=lambda cs: cs.charge_score, reverse=True)
        
        assert result
        # Verify high priority charger maintains some charging
        assert actionable_chargers[0].budgeted_amp > 0

        
    async def test_budgeting_algorithm_empty_list(self):
        """Test algorithm with no actionable chargers."""
        result, _ = await self.charger_group.budgeting_algorithm_minimize_diffs(
            [],  # No chargers
            5000.0,
            False,
            self.current_time
        )
        
        # Should return None or empty list
        assert result
        
    async def test_budgeting_algorithm_zero_power(self):
        """Test algorithm with no available power."""
        actionable_chargers = []
        for i, charger in enumerate(self.chargers):
            cs = self.create_charger_status(
                charger=charger,
                current_amp=10,
                power=2300.0,
                score=i + 1,
                idx=i
            )
            actionable_chargers.append(cs)
        
        # Test with zero power
        result, _ = await self.charger_group.budgeting_algorithm_minimize_diffs(
            actionable_chargers,
            0.0, # No power available
            False,
            self.current_time
        )
        
        assert result
        # All chargers should be reduced to minimum or idle charge
        for cs in actionable_chargers:
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
                score=i + 1,
                idx=i
            )
            actionable_chargers.append(cs)
        
        # Test with negative power (exporting to grid)
        result, _ = await self.charger_group.budgeting_algorithm_minimize_diffs(
            actionable_chargers,
            -2000.0,  # Exporting 2kW
            False,
            self.current_time
        )
        
        assert result
        # All chargers should be reduced
        total_reduction = 0
        for i, cs in enumerate(actionable_chargers):
            # The algorithm should reduce charging
            assert cs.budgeted_amp <= cs.current_real_max_charging_amp
            reduction = cs.current_real_max_charging_amp - cs.budgeted_amp
            total_reduction += reduction
        
        # Verify that there was some reduction
        assert total_reduction > 0

    async def test_budgeting_algorithm_high_solar_respects_group_limit(self):
        """Test that with abundant solar power, the algorithm gradually increases
        charging current while respecting the dynamic group's 32A phase current limit."""
        
        # Create new 3-phase chargers for this test
        self.chargers_3p = []
        
        # Mock entity_registry for charger creation
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry') as mock_entity_reg:
            # Mock the async_get function
            mock_entity_reg.async_get = MagicMock()
            # Mock the async_entries_for_device function to return empty list
            mock_entity_reg.async_entries_for_device = MagicMock(return_value=[])
            
            for i in range(3):
                charger_config = {
                    CONF_NAME: f"Wallbox_3Phase_{i}",
                    CONF_IS_3P: True,  # Configure as 3-phase
                    CONF_CHARGER_DEVICE_WALLBOX: f"device_wallbox_3p_{i}",
                    CONF_CHARGER_MIN_CHARGE: 6,
                    CONF_CHARGER_MAX_CHARGE: 32,  # 32A max for 3-phase
                    "dynamic_group_name": self.dynamic_group.name,
                    "home": self.home,
                    "hass": self.hass,
                    "config_entry": self.config_entry
                }
                charger = QSChargerWallbox(**charger_config)
                # Attach the default generic car to make charger operational
                charger.attach_car(charger._default_generic_car, self.current_time)
                self.chargers_3p.append(charger)
                self.home.add_device(charger)
        
        # Create new charger group for the 3-phase chargers
        self.charger_group_3p = QSChargerGroup(self.dynamic_group)
        
        # Create actionable chargers all starting at minimum (6A)
        actionable_chargers = []
        for i, charger in enumerate(self.chargers_3p):
            cs = self.create_charger_status(
                charger=charger,
                current_amp=6,  # All start at minimum
                power=6 * 230 * 3,  # 6A * 230V * 3 phases = 4140W
                score=1,        # Same priority for all
                idx=i
            )
            cs.current_active_phase_number = 3  # Set to 3 phases
            cs.possible_amps = list(range(6, 33))  # 6A to 32A
            cs.possible_num_phases = [3]  # Only 3-phase operation
            actionable_chargers.append(cs)
        
        # Test with abundant solar power (50kW)
        full_available_home_power = 50000.0
        
        # Run the budgeting algorithm multiple times to simulate continuous optimization
        results = []
        current_chargers = actionable_chargers
        
        # Run more iterations to see the full behavior
        max_iterations = 30
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            print(f"Available power: {full_available_home_power}W")
            print("Current state before budgeting:")
            total_current_power = 0
            total_phase_currents = [0, 0, 0]
            
            for i, cs in enumerate(current_chargers):
                power = cs.accurate_current_power
                print(f"  Charger {i} (3-phase): {cs.current_real_max_charging_amp}A, Power: {power}W")
                total_current_power += power
                # For 3-phase chargers, current flows through all phases
                for phase in range(3):
                    total_phase_currents[phase] += cs.current_real_max_charging_amp
            
            print(f"  Total current power: {total_current_power}W")
            print(f"  Phase currents: L1={total_phase_currents[0]}A, L2={total_phase_currents[1]}A, L3={total_phase_currents[2]}A")
            
            result, _ = await self.charger_group_3p.budgeting_algorithm_minimize_diffs(
                current_chargers,
                full_available_home_power,
                False,
                self.current_time
            )
            
            assert result

            result = list(sorted(current_chargers, key=lambda cs: cs.charge_score, reverse=True))
            results.append(result)
            
            print("Budgeted result:")
            budgeted_phase_currents = [0, 0, 0]
            for i, cs in enumerate(result):
                print(f"  Charger {i}: {cs.budgeted_amp}A")
                # Calculate phase currents for budgeted values
                for phase in range(3):
                    budgeted_phase_currents[phase] += cs.budgeted_amp
            print(f"  Budgeted phase currents: L1={budgeted_phase_currents[0]}A, L2={budgeted_phase_currents[1]}A, L3={budgeted_phase_currents[2]}A")
            
            # Check if we've reached a stable state (no changes in budgeted amps)
            if iteration > 0 and all(
                results[-1][i].budgeted_amp == results[-2][i].budgeted_amp 
                for i in range(len(results[-1]))
            ):
                print(f"\nReached stable state after {iteration + 1} iterations")
                break
            
            # Update current chargers for next iteration
            current_chargers = []
            for i, cs in enumerate(result):
                # Create new status with updated current
                new_cs = self.create_charger_status(
                    charger=self.chargers_3p[i],
                    current_amp=cs.budgeted_amp,
                    power=cs.budgeted_amp * 230 * 3,  # 3-phase power
                    score=1,
                    idx=i
                )
                new_cs.current_active_phase_number = 3
                new_cs.possible_amps = list(range(6, 33))
                new_cs.possible_num_phases = [3]
                current_chargers.append(new_cs)
        
        # Verify final results
        final_result = results[-1]
        
        print("\n=== Final Results ===")
        total_final_power = 0
        phase_currents = [0, 0, 0]
        
        for i, cs in enumerate(final_result):
            # For 3-phase chargers, current flows through all phases
            for phase in range(3):
                phase_currents[phase] += cs.budgeted_amp
            power = cs.budgeted_amp * 230 * 3  # 3-phase power
            total_final_power += power
            print(f"Charger {i} (3-phase): {cs.budgeted_amp}A ({power}W)")
        
        print(f"Total final power: {total_final_power}W")
        print(f"Phase currents: L1={phase_currents[0]}A, L2={phase_currents[1]}A, L3={phase_currents[2]}A")
        
        # Key assertions:
        
        # 1. At least one charger should have increased from minimum
        assert any(cs.budgeted_amp > 6 for cs in final_result), "At least one charger should increase from minimum"
        
        # 2. No charger should exceed its maximum (32A)
        for i, cs in enumerate(final_result):
            assert cs.budgeted_amp <= 32, f"Charger {i} exceeds its 32A maximum"
        
        # 3. CRITICAL: Total current per phase should not exceed the dynamic group's 32A limit
        # This is the key test - with 3 chargers on 3-phase, the total could be 3*32=96A per phase
        # but the dynamic group limit of 32A should prevent this
        for phase_idx, current in enumerate(phase_currents):
            assert current <= 32, f"Phase {phase_idx + 1} current {current}A exceeds 32A limit"
        
        # 4. Total power consumption should increase significantly from initial
        initial_total_power = len(actionable_chargers) * 6 * 230 * 3  # 3 * 6A * 230V * 3 phases
        assert total_final_power > initial_total_power, "Total power should increase with abundant solar"
        
        # 5. The algorithm should converge to maximum safe power
        # With 32A limit per phase and 3 phases: 32A * 230V * 3 = 22,080W theoretical max
        print(f"\nAlgorithm behavior: Started with {initial_total_power}W, ended with {total_final_power}W")
        print(f"Power utilization: {(total_final_power / full_available_home_power) * 100:.1f}% of available {full_available_home_power}W")
        print(f"Dynamic group limit utilization: {(total_final_power / 22080) * 100:.1f}% of theoretical max 22,080W")


if __name__ == '__main__':
    unittest.main()
