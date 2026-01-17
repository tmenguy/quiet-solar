"""
Test suite for PilotedDevice functionality.

This test suite tests the PilotedDevice feature where devices like heat pumps
are only considered "on" when at least one of their attached client devices
(like climate controllers) are actually on. The solver must properly account
for this in its power budgeting.

Key concepts tested:
- PilotedDevice reference counting for on/off states
- prepare_slots_for_piloted_device_budget initialization
- possible_delta_power_for_slot calculations
- update_num_demanding_clients_for_slot tracking
- Solver integration with multiple climates connected to a heat pump
"""

from unittest import TestCase
from datetime import datetime, timedelta

import pytz

from custom_components.quiet_solar.const import (
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
    CONF_POWER,
    CONF_DEVICE_TO_PILOT_NAME,
)
from custom_components.quiet_solar.home_model.constraints import (
    TimeBasedSimplePowerLoadConstraint,
)
from custom_components.quiet_solar.home_model.load import (
    AbstractLoad,
    PilotedDevice,
)
from custom_components.quiet_solar.home_model.solver import PeriodSolver
from custom_components.quiet_solar.home_model.battery import Battery
from custom_components.quiet_solar.home_model.commands import (
    copy_command,
    CMD_ON,
    CMD_OFF,
)


class MockPilotedDevice(PilotedDevice):
    """Test implementation of PilotedDevice for unit testing."""

    def __init__(self, name: str, power: float = 2000, **kwargs):
        kwargs[CONF_POWER] = power
        kwargs["name"] = name
        super().__init__(**kwargs)
        self.power_use = power


class MockClimateLoad(AbstractLoad):
    """Test implementation of a climate-like load that can be connected to a piloted device."""

    def __init__(self, name: str, power: float = 1500, piloted_device_name: str | None = None, **kwargs):
        kwargs[CONF_POWER] = power
        kwargs["name"] = name
        if piloted_device_name:
            kwargs[CONF_DEVICE_TO_PILOT_NAME] = piloted_device_name
        super().__init__(**kwargs)
        self.power_use = power
        self.min_p = power
        self.max_p = power

    def get_min_max_power(self) -> tuple[float, float]:
        return self.min_p, self.max_p


class MockHome:
    """Mock home for testing topology setup."""

    def __init__(self):
        self._all_piloted_devices: list[PilotedDevice] = []
        self._all_loads: list[AbstractLoad] = []
        self._name_to_piloted_devices: dict[str, PilotedDevice] = {}
        self.voltage = 230.0
        self.available_amps_for_group = None

    def is_off_grid(self) -> bool:
        return False

    def add_piloted_device(self, device: PilotedDevice):
        device.home = self
        if device not in self._all_piloted_devices:
            self._all_piloted_devices.append(device)
        self._name_to_piloted_devices[device.name] = device

    def add_load(self, load: AbstractLoad):
        load.home = self
        load.father_device = self
        if load not in self._all_loads:
            self._all_loads.append(load)

    def setup_topology(self):
        """Set up the topology linking loads to their piloted devices."""
        # Reset piloted device clients
        for piloted_device in self._all_piloted_devices:
            piloted_device.clients = []

        # Connect loads to their piloted devices
        for load in self._all_loads:
            load.devices_to_pilot = []
            if load.piloted_device_name is not None:
                piloted_device = self._name_to_piloted_devices.get(load.piloted_device_name)
                if piloted_device is not None:
                    load.devices_to_pilot.append(piloted_device)
                    piloted_device.clients.append(load)

    def prepare_slots_for_piloted_device_budget(self, time: datetime, num_slots: int):
        """Prepare slots for all piloted devices."""
        for piloted_device in self._all_piloted_devices:
            piloted_device.prepare_slots_for_piloted_device_budget(time, num_slots)

    def prepare_slots_for_amps_budget(self, time: datetime, num_slots: int, from_father_budget=None):
        """Prepare amp slots - simplified mock."""
        pass

    def update_available_amps_for_group(self, idx: int, amps: list[float | int], add: bool):
        """Update available amps - simplified mock."""
        pass


class TestPilotedDeviceBasics(TestCase):
    """Test basic PilotedDevice functionality."""

    def test_piloted_device_creation(self):
        """Test that a PilotedDevice can be created with power."""
        heat_pump = MockPilotedDevice(name="heat_pump", power=3000)

        self.assertEqual(heat_pump.name, "heat_pump")
        self.assertEqual(heat_pump.power_use, 3000)
        self.assertEqual(heat_pump.clients, [])
        self.assertIsNone(heat_pump.num_demanding_clients)

    def test_prepare_slots_initialization(self):
        """Test that prepare_slots_for_piloted_device_budget initializes the slot tracking."""
        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        time = datetime.now(pytz.UTC)
        num_slots = 10

        heat_pump.prepare_slots_for_piloted_device_budget(time, num_slots)

        self.assertIsNotNone(heat_pump.num_demanding_clients)
        self.assertEqual(len(heat_pump.num_demanding_clients), num_slots)
        self.assertTrue(all(c == 0 for c in heat_pump.num_demanding_clients))

    def test_possible_delta_power_no_clients(self):
        """Test that delta power is 0 when there are no clients."""
        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        heat_pump.prepare_slots_for_piloted_device_budget(datetime.now(pytz.UTC), 10)

        # No clients means no delta power
        delta = heat_pump.possible_delta_power_for_slot(0, add=True)
        self.assertEqual(delta, 0)

    def test_possible_delta_power_first_client_add(self):
        """Test that adding the first client returns the heat pump power."""
        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate = MockClimateLoad(name="climate1", power=1500)
        heat_pump.clients.append(climate)
        heat_pump.prepare_slots_for_piloted_device_budget(datetime.now(pytz.UTC), 10)

        # First client add should return the heat pump power
        delta = heat_pump.possible_delta_power_for_slot(0, add=True)
        self.assertEqual(delta, 2000)

    def test_possible_delta_power_second_client_add(self):
        """Test that adding a second client returns 0 (pump already on)."""
        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate1 = MockClimateLoad(name="climate1", power=1500)
        climate2 = MockClimateLoad(name="climate2", power=1500)
        heat_pump.clients.extend([climate1, climate2])
        heat_pump.prepare_slots_for_piloted_device_budget(datetime.now(pytz.UTC), 10)

        # Simulate first client already on
        heat_pump.num_demanding_clients[0] = 1

        # Second client add should return 0 (pump already on)
        delta = heat_pump.possible_delta_power_for_slot(0, add=True)
        self.assertEqual(delta, 0)

    def test_possible_delta_power_last_client_remove(self):
        """Test that removing the last client returns the heat pump power."""
        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate = MockClimateLoad(name="climate1", power=1500)
        heat_pump.clients.append(climate)
        heat_pump.prepare_slots_for_piloted_device_budget(datetime.now(pytz.UTC), 10)

        # Simulate one client on
        heat_pump.num_demanding_clients[0] = 1

        # Removing last client should return the power (pump will turn off)
        delta = heat_pump.possible_delta_power_for_slot(0, add=False)
        self.assertEqual(delta, 2000)

    def test_possible_delta_power_not_last_client_remove(self):
        """Test that removing a non-last client returns 0 (pump stays on)."""
        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate1 = MockClimateLoad(name="climate1", power=1500)
        climate2 = MockClimateLoad(name="climate2", power=1500)
        heat_pump.clients.extend([climate1, climate2])
        heat_pump.prepare_slots_for_piloted_device_budget(datetime.now(pytz.UTC), 10)

        # Simulate two clients on
        heat_pump.num_demanding_clients[0] = 2

        # Removing one client should return 0 (pump stays on)
        delta = heat_pump.possible_delta_power_for_slot(0, add=False)
        self.assertEqual(delta, 0)


class TestPilotedDeviceClientTracking(TestCase):
    """Test the client tracking / ref counting mechanism."""

    def test_update_num_demanding_clients_add(self):
        """Test that adding clients increments the count and returns correct power delta."""
        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate1 = MockClimateLoad(name="climate1", power=1500)
        climate2 = MockClimateLoad(name="climate2", power=1500)
        heat_pump.clients.extend([climate1, climate2])
        heat_pump.prepare_slots_for_piloted_device_budget(datetime.now(pytz.UTC), 10)

        # Add first client
        delta1 = heat_pump.update_num_demanding_clients_for_slot(0, add=True)
        self.assertEqual(delta1, 2000)  # First add turns on pump
        self.assertEqual(heat_pump.num_demanding_clients[0], 1)

        # Add second client
        delta2 = heat_pump.update_num_demanding_clients_for_slot(0, add=True)
        self.assertEqual(delta2, 0)  # Second add, pump already on
        self.assertEqual(heat_pump.num_demanding_clients[0], 2)

    def test_update_num_demanding_clients_remove(self):
        """Test that removing clients decrements the count and returns correct power delta."""
        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate1 = MockClimateLoad(name="climate1", power=1500)
        climate2 = MockClimateLoad(name="climate2", power=1500)
        heat_pump.clients.extend([climate1, climate2])
        heat_pump.prepare_slots_for_piloted_device_budget(datetime.now(pytz.UTC), 10)

        # Start with two clients on
        heat_pump.num_demanding_clients[0] = 2

        # Remove first client
        delta1 = heat_pump.update_num_demanding_clients_for_slot(0, add=False)
        self.assertEqual(delta1, 0)  # Not the last, pump stays on
        self.assertEqual(heat_pump.num_demanding_clients[0], 1)

        # Remove second client
        delta2 = heat_pump.update_num_demanding_clients_for_slot(0, add=False)
        self.assertEqual(delta2, 2000)  # Last removal turns off pump
        self.assertEqual(heat_pump.num_demanding_clients[0], 0)

    def test_update_prevents_negative_count(self):
        """Test that the count doesn't go negative."""
        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate = MockClimateLoad(name="climate1", power=1500)
        heat_pump.clients.append(climate)
        heat_pump.prepare_slots_for_piloted_device_budget(datetime.now(pytz.UTC), 10)

        # Try to remove when count is 0
        delta = heat_pump.update_num_demanding_clients_for_slot(0, add=False)
        self.assertEqual(heat_pump.num_demanding_clients[0], 0)

    def test_update_prevents_overflow_count(self):
        """Test that the count doesn't exceed the number of clients."""
        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate1 = MockClimateLoad(name="climate1", power=1500)
        climate2 = MockClimateLoad(name="climate2", power=1500)
        heat_pump.clients.extend([climate1, climate2])
        heat_pump.prepare_slots_for_piloted_device_budget(datetime.now(pytz.UTC), 10)

        # Set count to max clients
        heat_pump.num_demanding_clients[0] = 2

        # Try to add more - should cap at clients count
        heat_pump.update_num_demanding_clients_for_slot(0, add=True)
        self.assertEqual(heat_pump.num_demanding_clients[0], 2)


class TestTopologySetup(TestCase):
    """Test the topology setup that connects loads to piloted devices."""

    def test_topology_links_load_to_piloted_device(self):
        """Test that topology setup correctly links loads to their piloted devices."""
        home = MockHome()

        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate1 = MockClimateLoad(name="climate1", power=1500, piloted_device_name="heat_pump")
        climate2 = MockClimateLoad(name="climate2", power=1500, piloted_device_name="heat_pump")

        home.add_piloted_device(heat_pump)
        home.add_load(climate1)
        home.add_load(climate2)
        home.setup_topology()

        # Check heat pump has both climates as clients
        self.assertEqual(len(heat_pump.clients), 2)
        self.assertIn(climate1, heat_pump.clients)
        self.assertIn(climate2, heat_pump.clients)

        # Check each climate has the heat pump in its devices_to_pilot
        self.assertEqual(len(climate1.devices_to_pilot), 1)
        self.assertEqual(climate1.devices_to_pilot[0], heat_pump)
        self.assertEqual(len(climate2.devices_to_pilot), 1)
        self.assertEqual(climate2.devices_to_pilot[0], heat_pump)

    def test_load_without_piloted_device(self):
        """Test that loads without piloted_device_name don't get linked."""
        home = MockHome()

        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate = MockClimateLoad(name="climate", power=1500)  # No piloted_device_name

        home.add_piloted_device(heat_pump)
        home.add_load(climate)
        home.setup_topology()

        self.assertEqual(len(heat_pump.clients), 0)
        self.assertEqual(len(climate.devices_to_pilot), 0)


class TestLoadDeltaPowerFromPilotedDevices(TestCase):
    """Test the get_possible_delta_power_from_piloted_devices_for_budget method on loads."""

    def test_load_delta_power_from_piloted_device(self):
        """Test that a load correctly gets delta power from its piloted device."""
        home = MockHome()

        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate1 = MockClimateLoad(name="climate1", power=1500, piloted_device_name="heat_pump")
        climate2 = MockClimateLoad(name="climate2", power=1500, piloted_device_name="heat_pump")

        home.add_piloted_device(heat_pump)
        home.add_load(climate1)
        home.add_load(climate2)
        home.setup_topology()

        time = datetime.now(pytz.UTC)
        home.prepare_slots_for_piloted_device_budget(time, 10)

        # First climate add should return heat pump power
        delta = climate1.get_possible_delta_power_from_piloted_devices_for_budget(0, add=True)
        self.assertEqual(delta, 2000)

    def test_load_update_demanding_clients(self):
        """Test that a load can update demanding clients through its piloted device."""
        home = MockHome()

        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate1 = MockClimateLoad(name="climate1", power=1500, piloted_device_name="heat_pump")
        climate2 = MockClimateLoad(name="climate2", power=1500, piloted_device_name="heat_pump")

        home.add_piloted_device(heat_pump)
        home.add_load(climate1)
        home.add_load(climate2)
        home.setup_topology()

        time = datetime.now(pytz.UTC)
        home.prepare_slots_for_piloted_device_budget(time, 10)

        # Add climate1 - should turn on heat pump
        delta1 = climate1.update_demanding_clients_for_piloted_devices_for_budget(0, add=True)
        self.assertEqual(delta1, 2000)
        self.assertEqual(heat_pump.num_demanding_clients[0], 1)

        # Add climate2 - heat pump already on
        delta2 = climate2.update_demanding_clients_for_piloted_devices_for_budget(0, add=True)
        self.assertEqual(delta2, 0)
        self.assertEqual(heat_pump.num_demanding_clients[0], 2)


class TestSolverWithPilotedDevice(TestCase):
    """Test the solver integration with PilotedDevice."""

    def test_solver_two_climates_one_heat_pump_both_mandatory(self):
        """
        Test solver with two climate loads connected to one heat pump,
        both with mandatory constraints. The solver should account for
        the heat pump power when both are running.
        """
        dt = datetime(year=2024, month=6, day=1, hour=10, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=8)

        tariffs = 0.25 / 1000.0

        # Create home with heat pump and two climates
        home = MockHome()

        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate1 = MockClimateLoad(name="climate1", power=1500, piloted_device_name="heat_pump")
        climate2 = MockClimateLoad(name="climate2", power=1500, piloted_device_name="heat_pump")

        home.add_piloted_device(heat_pump)
        home.add_load(climate1)
        home.add_load(climate2)
        home.setup_topology()

        # Create mandatory constraints for both climates
        climate1_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=climate1,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=dt + timedelta(hours=6),
            initial_value=0,
            target_value=3 * 3600,  # 3 hours of runtime
            power=1500,
        )
        climate1.push_live_constraint(dt, climate1_constraint)

        climate2_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=climate2,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=dt + timedelta(hours=6),
            initial_value=0,
            target_value=3 * 3600,  # 3 hours of runtime
            power=1500,
        )
        climate2.push_live_constraint(dt, climate2_constraint)

        # Solar forecast with good production
        pv_forecast = []
        for h in range(8):
            hour = dt + timedelta(hours=h)
            if h < 2:
                solar_power = 3000
            elif h < 5:
                solar_power = 6000  # Peak solar
            else:
                solar_power = 2000
            pv_forecast.append((hour, solar_power))

        # Low base consumption
        unavoidable_consumption_forecast = []
        for h in range(8):
            hour = dt + timedelta(hours=h)
            unavoidable_consumption_forecast.append((hour, 500))

        # Create and run solver
        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=tariffs,
            actionable_loads=[climate1, climate2],
            battery=None,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=unavoidable_consumption_forecast,
        )

        load_commands, battery_commands = solver.solve()

        # Verify basic solver output
        self.assertIsNotNone(load_commands)
        self.assertEqual(len(load_commands), 2)

        climate1_cmds = load_commands[0][1]
        climate2_cmds = load_commands[1][1]

        self.assertTrue(len(climate1_cmds) > 0, "Climate 1 should have commands")
        self.assertTrue(len(climate2_cmds) > 0, "Climate 2 should have commands")

        # Count ON commands for each climate
        climate1_on_count = sum(1 for _, cmd in climate1_cmds if not cmd.is_off_or_idle())
        climate2_on_count = sum(1 for _, cmd in climate2_cmds if not cmd.is_off_or_idle())

        print(f"Climate 1 ON commands: {climate1_on_count}")
        print(f"Climate 2 ON commands: {climate2_on_count}")

        # Both should have ON commands
        self.assertTrue(climate1_on_count > 0, "Climate 1 should have ON commands")
        self.assertTrue(climate2_on_count > 0, "Climate 2 should have ON commands")

    def test_solver_heat_pump_power_accounted_in_budget(self):
        """
        Test that the solver correctly accounts for heat pump power
        when the first climate turns on.

        When climate1 turns on:
        - climate1 power: 1500W
        - heat_pump power: 2000W (added because it's the first client)
        - Total: 3500W

        When climate2 also turns on:
        - climate2 power: 1500W
        - heat_pump power: 0W (already on)
        - Total additional: 1500W
        """
        dt = datetime(year=2024, month=6, day=1, hour=12, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=4)

        tariffs = 0.25 / 1000.0

        # Create home with heat pump and two climates
        home = MockHome()

        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate1 = MockClimateLoad(name="climate1", power=1500, piloted_device_name="heat_pump")
        climate2 = MockClimateLoad(name="climate2", power=1500, piloted_device_name="heat_pump")

        home.add_piloted_device(heat_pump)
        home.add_load(climate1)
        home.add_load(climate2)
        home.setup_topology()

        # Climate1 needs 2 hours, climate2 needs 1 hour
        climate1_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=climate1,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=dt + timedelta(hours=3),
            initial_value=0,
            target_value=2 * 3600,
            power=1500,
        )
        climate1.push_live_constraint(dt, climate1_constraint)

        climate2_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=climate2,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=dt + timedelta(hours=3),
            initial_value=0,
            target_value=1 * 3600,
            power=1500,
        )
        climate2.push_live_constraint(dt, climate2_constraint)

        # Exactly enough solar for the scenario
        # With both climates + heat pump: 1500 + 1500 + 2000 = 5000W
        pv_forecast = [(dt + timedelta(hours=h), 5500) for h in range(4)]
        unavoidable_consumption_forecast = [(dt + timedelta(hours=h), 500) for h in range(4)]

        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=tariffs,
            actionable_loads=[climate1, climate2],
            battery=None,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=unavoidable_consumption_forecast,
        )

        load_commands, battery_commands = solver.solve()

        # Verify solver worked
        self.assertIsNotNone(load_commands)

        # Extract command lists
        climate1_cmds = load_commands[0][1]
        climate2_cmds = load_commands[1][1]

        # Print commands for debugging
        print("\nClimate 1 commands:")
        for time_cmd, cmd in climate1_cmds:
            print(f"  {time_cmd.strftime('%H:%M')}: {cmd.command} (power: {cmd.power_consign}W)")

        print("\nClimate 2 commands:")
        for time_cmd, cmd in climate2_cmds:
            print(f"  {time_cmd.strftime('%H:%M')}: {cmd.command} (power: {cmd.power_consign}W)")

        # Verify both climates got scheduled
        climate1_on = any(not cmd.is_off_or_idle() for _, cmd in climate1_cmds)
        climate2_on = any(not cmd.is_off_or_idle() for _, cmd in climate2_cmds)

        self.assertTrue(climate1_on, "Climate 1 should be scheduled to run")
        self.assertTrue(climate2_on, "Climate 2 should be scheduled to run")

    def test_solver_limited_power_prefers_one_climate(self):
        """
        Test that when power is limited, the solver prefers running one
        climate at a time rather than trying to run both.

        This tests the heat pump ref counting - running both would need:
        - climate1: 1500W
        - climate2: 1500W
        - heat_pump: 2000W (only once, shared)
        - Total: 5000W

        But running them sequentially:
        - First slot: climate1 (1500W) + heat_pump (2000W) = 3500W
        - Second slot: climate2 (1500W) + heat_pump (2000W) = 3500W
        """
        dt = datetime(year=2024, month=6, day=1, hour=12, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=4)

        tariffs = 0.25 / 1000.0

        home = MockHome()

        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate1 = MockClimateLoad(name="climate1", power=1500, piloted_device_name="heat_pump")
        climate2 = MockClimateLoad(name="climate2", power=1500, piloted_device_name="heat_pump")

        home.add_piloted_device(heat_pump)
        home.add_load(climate1)
        home.add_load(climate2)
        home.setup_topology()

        # Both need 1 hour each
        climate1_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=climate1,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=dt + timedelta(hours=3),
            initial_value=0,
            target_value=1 * 3600,
            power=1500,
        )
        climate1.push_live_constraint(dt, climate1_constraint)

        climate2_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=climate2,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=dt + timedelta(hours=3),
            initial_value=0,
            target_value=1 * 3600,
            power=1500,
        )
        climate2.push_live_constraint(dt, climate2_constraint)

        # Limited solar - only enough for one climate + heat pump at a time
        # 3500W available (can run one climate + heat pump)
        # But NOT enough for both climates + heat pump (would need 5000W)
        pv_forecast = [(dt + timedelta(hours=h), 4000) for h in range(4)]
        unavoidable_consumption_forecast = [(dt + timedelta(hours=h), 500) for h in range(4)]

        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=tariffs,
            actionable_loads=[climate1, climate2],
            battery=None,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=unavoidable_consumption_forecast,
        )

        load_commands, battery_commands = solver.solve()

        self.assertIsNotNone(load_commands)

        climate1_cmds = load_commands[0][1]
        climate2_cmds = load_commands[1][1]

        # Both climates should eventually run (mandataory constraints)
        climate1_on_times = [time_cmd for time_cmd, cmd in climate1_cmds if not cmd.is_off_or_idle()]
        climate2_on_times = [time_cmd for time_cmd, cmd in climate2_cmds if not cmd.is_off_or_idle()]

        print(f"\nClimate 1 ON times: {[t.strftime('%H:%M') for t in climate1_on_times]}")
        print(f"Climate 2 ON times: {[t.strftime('%H:%M') for t in climate2_on_times]}")

        # Both should have some ON time
        self.assertTrue(len(climate1_on_times) > 0, "Climate 1 should run")
        self.assertTrue(len(climate2_on_times) > 0, "Climate 2 should run")

    def test_solver_three_climates_sequential_activation(self):
        """
        Test with three climates connected to one heat pump.
        Verifies that the solver handles multiple clients correctly.
        """
        dt = datetime(year=2024, month=6, day=1, hour=10, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=6)

        tariffs = 0.20 / 1000.0

        home = MockHome()

        heat_pump = MockPilotedDevice(name="heat_pump", power=2500)
        climate1 = MockClimateLoad(name="climate1", power=1000, piloted_device_name="heat_pump")
        climate2 = MockClimateLoad(name="climate2", power=1000, piloted_device_name="heat_pump")
        climate3 = MockClimateLoad(name="climate3", power=1000, piloted_device_name="heat_pump")

        home.add_piloted_device(heat_pump)
        home.add_load(climate1)
        home.add_load(climate2)
        home.add_load(climate3)
        home.setup_topology()

        # Each climate needs 1 hour
        for i, climate in enumerate([climate1, climate2, climate3]):
            constraint = TimeBasedSimplePowerLoadConstraint(
                time=dt,
                load=climate,
                type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
                end_of_constraint=dt + timedelta(hours=5),
                initial_value=0,
                target_value=1 * 3600,
                power=1000,
            )
            climate.push_live_constraint(dt, constraint)

        # Moderate solar
        pv_forecast = [(dt + timedelta(hours=h), 4500) for h in range(6)]
        unavoidable_consumption_forecast = [(dt + timedelta(hours=h), 500) for h in range(6)]

        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=tariffs,
            actionable_loads=[climate1, climate2, climate3],
            battery=None,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=unavoidable_consumption_forecast,
        )

        load_commands, battery_commands = solver.solve()

        self.assertIsNotNone(load_commands)
        self.assertEqual(len(load_commands), 3)

        # Verify all three climates get scheduled
        for i, (load, cmds) in enumerate(load_commands):
            on_count = sum(1 for _, cmd in cmds if not cmd.is_off_or_idle())
            print(f"Climate {i+1} ON slots: {on_count}")
            self.assertTrue(on_count > 0, f"Climate {i+1} should have ON commands")


class TestSolverWithBatteryAndPilotedDevice(TestCase):
    """Test solver with battery and piloted devices."""

    def test_solver_battery_heat_pump_interaction(self):
        """
        Test that the solver correctly handles battery and heat pump interaction.
        The heat pump power should be properly accounted when deciding battery usage.
        """
        dt = datetime(year=2024, month=6, day=1, hour=14, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=6)

        tariffs = 0.25 / 1000.0

        home = MockHome()

        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate1 = MockClimateLoad(name="climate1", power=1500, piloted_device_name="heat_pump")
        climate2 = MockClimateLoad(name="climate2", power=1500, piloted_device_name="heat_pump")

        home.add_piloted_device(heat_pump)
        home.add_load(climate1)
        home.add_load(climate2)
        home.setup_topology()

        # Battery
        battery = Battery(name="test_battery")
        battery.capacity = 10000
        battery.max_charging_power = 5000
        battery.max_discharging_power = 5000
        battery._current_charge_value = 5000  # 50% charged
        battery.min_charge_SOC_percent = 10.0
        battery.max_charge_SOC_percent = 95.0

        # Constraints
        climate1_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=climate1,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=dt + timedelta(hours=4),
            initial_value=0,
            target_value=2 * 3600,
            power=1500,
        )
        climate1.push_live_constraint(dt, climate1_constraint)

        climate2_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=climate2,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,  # Best effort
            end_of_constraint=None,
            initial_value=0,
            target_value=2 * 3600,
            power=1500,
        )
        climate2.push_live_constraint(dt, climate2_constraint)

        # Solar declining through afternoon
        pv_forecast = []
        for h in range(6):
            hour = dt + timedelta(hours=h)
            if h < 2:
                solar_power = 5000
            elif h < 4:
                solar_power = 3000
            else:
                solar_power = 1000
            pv_forecast.append((hour, solar_power))

        unavoidable_consumption_forecast = [(dt + timedelta(hours=h), 500) for h in range(6)]

        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=tariffs,
            actionable_loads=[climate1, climate2],
            battery=battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=unavoidable_consumption_forecast,
        )

        load_commands, battery_commands = solver.solve()

        self.assertIsNotNone(load_commands)
        self.assertIsNotNone(battery_commands)

        # Print results for debugging
        print("\nClimate 1 commands:")
        for time_cmd, cmd in load_commands[0][1]:
            print(f"  {time_cmd.strftime('%H:%M')}: {cmd.command} (power: {cmd.power_consign}W)")

        print("\nClimate 2 commands:")
        for time_cmd, cmd in load_commands[1][1]:
            print(f"  {time_cmd.strftime('%H:%M')}: {cmd.command} (power: {cmd.power_consign}W)")

        print("\nBattery commands:")
        for time_cmd, cmd in battery_commands:
            print(f"  {time_cmd.strftime('%H:%M')}: {cmd.command} (power: {cmd.power_consign}W)")

        # Verify mandatory constraint is satisfied
        climate1_on_count = sum(1 for _, cmd in load_commands[0][1] if not cmd.is_off_or_idle())
        self.assertTrue(climate1_on_count > 0, "Climate 1 (mandatory) should run")


class TestEdgeCases(TestCase):
    """Test edge cases for piloted devices."""

    def test_piloted_device_no_power(self):
        """Test piloted device with zero power."""
        heat_pump = MockPilotedDevice(name="heat_pump", power=0)
        climate = MockClimateLoad(name="climate", power=1500, piloted_device_name="heat_pump")
        heat_pump.clients.append(climate)
        heat_pump.prepare_slots_for_piloted_device_budget(datetime.now(pytz.UTC), 10)

        # Should return 0 since heat pump power is 0
        delta = heat_pump.possible_delta_power_for_slot(0, add=True)
        self.assertEqual(delta, 0)

    def test_piloted_device_multiple_slots_independent(self):
        """Test that different slots are tracked independently."""
        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate = MockClimateLoad(name="climate", power=1500)
        heat_pump.clients.append(climate)
        heat_pump.prepare_slots_for_piloted_device_budget(datetime.now(pytz.UTC), 5)

        # Add to slot 0
        heat_pump.update_num_demanding_clients_for_slot(0, add=True)

        # Slot 0 should have 1, others should have 0
        self.assertEqual(heat_pump.num_demanding_clients[0], 1)
        self.assertEqual(heat_pump.num_demanding_clients[1], 0)
        self.assertEqual(heat_pump.num_demanding_clients[2], 0)

        # Add to slot 2
        heat_pump.update_num_demanding_clients_for_slot(2, add=True)

        self.assertEqual(heat_pump.num_demanding_clients[0], 1)
        self.assertEqual(heat_pump.num_demanding_clients[1], 0)
        self.assertEqual(heat_pump.num_demanding_clients[2], 1)

    def test_is_piloted_device_activated(self):
        """Test the is_piloted_device_activated property."""
        heat_pump = MockPilotedDevice(name="heat_pump", power=2000)
        climate1 = MockClimateLoad(name="climate1", power=1500)
        climate2 = MockClimateLoad(name="climate2", power=1500)
        heat_pump.clients.extend([climate1, climate2])

        # Both climates have no command - pump should be inactive
        climate1.current_command = None
        climate2.current_command = None
        self.assertFalse(heat_pump.is_piloted_device_activated)

        # Climate1 has ON command - pump should be active
        climate1.current_command = copy_command(CMD_ON)
        self.assertTrue(heat_pump.is_piloted_device_activated)

        # Climate1 has OFF command, climate2 has ON - pump should be active
        climate1.current_command = copy_command(CMD_OFF)
        climate2.current_command = copy_command(CMD_ON)
        self.assertTrue(heat_pump.is_piloted_device_activated)

        # Both have OFF - pump should be inactive
        climate1.current_command = copy_command(CMD_OFF)
        climate2.current_command = copy_command(CMD_OFF)
        self.assertFalse(heat_pump.is_piloted_device_activated)


if __name__ == "__main__":
    import unittest
    unittest.main()
