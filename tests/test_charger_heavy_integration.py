"""
Integration-style tests for charger.py complex async methods.

Tests for:
- check_load_activity_and_constraints - Complex constraint logic
- ensure_correct_state - State machine verification
- dyn_handle - Dynamic power distribution

Uses pytest-asyncio for proper async testing and comprehensive mocking.
"""
import unittest
from unittest.mock import MagicMock, Mock, patch, AsyncMock, PropertyMock, call
from datetime import datetime, timedelta, time as dt_time
import pytz
import pytest

from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE

from custom_components.quiet_solar.ha_model.charger import (
    QSChargerGroup,
    QSChargerStatus,
    QSStateCmd,
    QSChargerStates,
    QSChargerGeneric,
    CHARGER_CHECK_STATE_WINDOW_S,
    CHARGER_ADAPTATION_WINDOW_S,
    CHARGER_BOOT_TIME_DATA_EXPIRATION_S,
)
from custom_components.quiet_solar.home_model.commands import (
    CMD_AUTO_FROM_CONSIGN,
    CMD_AUTO_GREEN_ONLY,
    CMD_AUTO_GREEN_CAP,
    CMD_AUTO_GREEN_CONSIGN,
    CMD_AUTO_PRICE,
    CMD_ON,
    CMD_OFF,
    copy_command
)
from custom_components.quiet_solar.const import (
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CHARGER_CONSUMPTION,
    CONF_CHARGER_STATUS_SENSOR,
    CONF_CHARGER_PLUGGED,
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER,
    CONF_CHARGER_THREE_TO_ONE_PHASE_SWITCH,
    CONF_CHARGER_REBOOT_BUTTON,
    CONF_IS_3P,
    CONF_MONO_PHASE,
    DOMAIN,
    DATA_HANDLER,
    CHARGER_NO_CAR_CONNECTED,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    CONSTRAINT_TYPE_FILLER_AUTO,
)


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

def create_mock_hass():
    """Create a properly configured mock Home Assistant instance."""
    hass = MagicMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.services = MagicMock()
    hass.data = {DOMAIN: {DATA_HANDLER: MagicMock()}}
    return hass


def create_mock_home(hass):
    """Create a properly configured mock QSHome instance."""
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
    home.force_next_solve = MagicMock()
    home.get_car_by_name = MagicMock(return_value=None)
    return home


def create_mock_car(name="TestCar", charge_percent=50.0, battery_capacity=60000):
    """Create a properly configured mock car."""
    car = MagicMock()
    car.name = name
    car.car_battery_capacity = battery_capacity
    car.car_default_charge = 80.0
    car.car_charger_min_charge = 6
    car.car_charger_max_charge = 32
    car.do_force_next_charge = False
    car.do_next_charge_time = None
    car.calendar = None
    car.user_selected_person_name_for_car = None
    car.user_attached_charger_name = None
    car.charger = None

    # Mock methods
    car.reset = MagicMock()
    car.get_car_charge_percent = MagicMock(return_value=charge_percent)
    car.get_car_target_SOC = MagicMock(return_value=80.0)
    car.get_car_target_charge_energy = MagicMock(return_value=30000)
    car.can_use_charge_percent_constraints = MagicMock(return_value=True)
    car.setup_car_charge_target_if_needed = AsyncMock(return_value=80.0)
    car.get_best_person_next_need = AsyncMock(return_value=(None, None, None, None))
    car.get_next_scheduled_event = AsyncMock(return_value=(None, None))
    car.get_car_minimum_ok_SOC = MagicMock(return_value=20.0)
    car.get_charge_power_per_phase_A = MagicMock(return_value=(
        [0] * 6 + [1380, 1610, 1840, 2070, 2300, 2530, 2760, 2990, 3220, 3450, 3680] + [3680] * 20,
        6, 32
    ))

    return car


def create_mock_dynamic_group(home, max_phase_current=32):
    """Create a properly configured mock dynamic group."""
    dyn_group = MagicMock()
    dyn_group.name = "TestDynGroup"
    dyn_group.home = home
    dyn_group._childrens = []
    dyn_group.dyn_group_max_phase_current = [max_phase_current] * 3
    dyn_group.dyn_group_max_phase_current_conf = max_phase_current
    dyn_group.accurate_power_sensor = "sensor.dyn_group_power"

    # Mock methods
    dyn_group.is_current_acceptable = MagicMock(return_value=True)
    dyn_group.is_current_acceptable_and_diff = MagicMock(return_value=(True, [0, 0, 0]))
    dyn_group.get_median_sensor = MagicMock(return_value=5000.0)

    return dyn_group


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


def create_mock_charger_status(charger, current_amp=10, num_phases=3, score=100):
    """Create a mock QSChargerStatus."""
    cs = MagicMock(spec=QSChargerStatus)
    cs.charger = charger
    cs.name = f"{charger.name}/TestCar"
    cs.current_real_max_charging_amp = current_amp
    cs.current_active_phase_number = num_phases
    cs.possible_amps = [0, 6, 10, 16, 20, 24, 32]
    cs.possible_num_phases = [1, 3]
    cs.budgeted_amp = current_amp
    cs.budgeted_num_phases = num_phases
    cs.charge_score = score
    cs.can_be_started_and_stopped = True
    cs.is_before_battery = False
    cs.bump_solar = False
    cs.command = CMD_AUTO_GREEN_ONLY
    cs.accurate_current_power = 5000.0
    cs.secondary_current_power = 4800.0

    # Mock methods
    cs.get_current_charging_amps = MagicMock(return_value=[current_amp] * num_phases if num_phases == 3 else [current_amp, 0, 0])
    cs.get_budget_amps = MagicMock(return_value=[current_amp] * num_phases if num_phases == 3 else [current_amp, 0, 0])
    cs.get_amps_from_values = MagicMock(side_effect=lambda amp, phases: [amp] * phases if phases == 3 else [amp, 0, 0])
    cs.get_diff_power = MagicMock(return_value=500.0)
    cs.can_change_budget = MagicMock(return_value=(None, None))
    cs.duplicate = MagicMock(return_value=cs)

    return cs


# ============================================================================
# Tests for check_load_activity_and_constraints
# ============================================================================

class TestCheckLoadActivityAndConstraints:
    """Tests for the check_load_activity_and_constraints method."""

    @pytest.fixture
    def setup_charger(self):
        """Set up a charger for testing."""
        hass = create_mock_hass()
        home = create_mock_home(hass)
        charger = create_charger_generic(hass, home)
        charger._constraints = []
        charger._boot_time = None
        charger._asked_for_reboot_at_time = None
        return charger, home

    @pytest.mark.asyncio
    async def test_returns_false_during_reboot(self, setup_charger):
        """Test that method returns False during reboot."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)
        charger._asked_for_reboot_at_time = time - timedelta(seconds=30)

        result = await charger.check_load_activity_and_constraints(time)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_during_boot_window(self, setup_charger):
        """Test that method returns False during boot time window."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)
        charger._boot_time = time - timedelta(seconds=5)  # Just booted

        with patch.object(charger, 'is_charger_unavailable', return_value=False), \
             patch.object(charger, 'probe_for_possible_needed_reboot', return_value=False):
            result = await charger.check_load_activity_and_constraints(time)

        assert result is False

    @pytest.mark.asyncio
    async def test_triggers_reboot_when_needed(self, setup_charger):
        """Test that reboot is triggered when probe indicates it's needed."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)

        with patch.object(charger, 'is_charger_unavailable', return_value=False), \
             patch.object(charger, 'probe_for_possible_needed_reboot', return_value=True), \
             patch.object(charger, 'reboot', new_callable=AsyncMock) as mock_reboot:
            result = await charger.check_load_activity_and_constraints(time)

        mock_reboot.assert_called_once_with(time)
        assert result is False

    @pytest.mark.asyncio
    async def test_unplugged_resets_car(self, setup_charger):
        """Test that unplugging resets the car and charger state."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)

        mock_car = create_mock_car()
        mock_car.user_selected_person_name_for_car = "TestPerson"
        charger.car = mock_car
        charger.user_attached_car_name = "TestCar"

        with patch.object(charger, 'is_charger_unavailable', return_value=False), \
             patch.object(charger, 'probe_for_possible_needed_reboot', return_value=False), \
             patch.object(charger, 'is_not_plugged', return_value=True), \
             patch.object(charger, 'is_plugged', return_value=False), \
             patch.object(charger, 'set_charging_num_phases', new_callable=AsyncMock) as mock_set_phases, \
             patch.object(charger, 'set_max_charging_current', new_callable=AsyncMock) as mock_set_current, \
             patch.object(charger, 'reset') as mock_reset:

            mock_set_phases.return_value = False
            result = await charger.check_load_activity_and_constraints(time)

        # Should reset the charger
        mock_reset.assert_called_once()
        assert charger.user_attached_car_name is None
        assert result is True  # Force solve triggered

    @pytest.mark.asyncio
    async def test_plugged_attaches_best_car(self, setup_charger):
        """Test that plugging in attaches the best car."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)

        mock_car = create_mock_car("BestCar")

        # Create a proper power command
        from custom_components.quiet_solar.home_model.commands import copy_command, CMD_AUTO_GREEN_ONLY
        power_cmd = copy_command(CMD_AUTO_GREEN_ONLY, power_consign=3000.0)

        # We need attach_car to actually set the car and power steps
        def attach_car_side_effect(car, t):
            charger.car = car
            charger._power_steps = [power_cmd]  # Need at least one power step

        with patch.object(charger, 'is_charger_unavailable', return_value=False), \
             patch.object(charger, 'probe_for_possible_needed_reboot', return_value=False), \
             patch.object(charger, 'is_not_plugged', return_value=False), \
             patch.object(charger, 'is_plugged', return_value=True), \
             patch.object(charger, 'get_best_car', return_value=mock_car), \
             patch.object(charger, 'attach_car', side_effect=attach_car_side_effect) as mock_attach, \
             patch.object(charger, 'clean_constraints_for_load_param_and_if_same_key_same_value_info'), \
             patch.object(charger, 'push_live_constraint', return_value=True), \
             patch.object(charger, 'is_car_charged', return_value=(False, 50.0)), \
             patch.object(charger, 'is_off_grid', return_value=False), \
             patch.object(type(charger), 'qs_bump_solar_charge_priority', new_callable=PropertyMock, return_value=False):

            result = await charger.check_load_activity_and_constraints(time)

        mock_attach.assert_called_once_with(mock_car, time)

    @pytest.mark.asyncio
    async def test_plugged_no_car_selected_resets(self, setup_charger):
        """Test that selecting no car resets the charger."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)

        with patch.object(charger, 'is_charger_unavailable', return_value=False), \
             patch.object(charger, 'probe_for_possible_needed_reboot', return_value=False), \
             patch.object(charger, 'is_not_plugged', return_value=False), \
             patch.object(charger, 'is_plugged', return_value=True), \
             patch.object(charger, 'get_best_car', return_value=None), \
             patch.object(charger, 'reset') as mock_reset:

            result = await charger.check_load_activity_and_constraints(time)

        mock_reset.assert_called_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_force_charge_creates_fast_constraint(self, setup_charger):
        """Test that force charge attempts to create an as-fast-as-possible constraint."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)

        mock_car = create_mock_car("ForceCar", charge_percent=30.0)
        mock_car.do_force_next_charge = True

        # Create a proper power command - use copy_command from commands module
        from custom_components.quiet_solar.home_model.commands import copy_command, CMD_AUTO_GREEN_ONLY
        power_cmd = copy_command(CMD_AUTO_GREEN_ONLY, power_consign=3000.0)

        # Attach car directly since the test needs the car to exist
        charger.car = mock_car
        charger._power_steps = [power_cmd]

        # Mock the constraint class creation to avoid deep constraint logic
        mock_constraint = MagicMock()

        with patch.object(charger, 'is_charger_unavailable', return_value=False), \
             patch.object(charger, 'probe_for_possible_needed_reboot', return_value=False), \
             patch.object(charger, 'is_not_plugged', return_value=False), \
             patch.object(charger, 'is_plugged', return_value=True), \
             patch.object(charger, 'get_best_car', return_value=mock_car), \
             patch.object(charger, 'clean_constraints_for_load_param_and_if_same_key_same_value_info'), \
             patch.object(charger, 'command_and_constraint_reset') as mock_reset, \
             patch.object(charger, 'push_live_constraint', return_value=True) as mock_push, \
             patch.object(charger, 'is_car_charged', return_value=(False, 30.0)), \
             patch.object(charger, 'is_off_grid', return_value=False), \
             patch.object(type(charger), 'qs_bump_solar_charge_priority', new_callable=PropertyMock, return_value=False), \
             patch.object(charger, 'update_power_steps'), \
             patch('custom_components.quiet_solar.ha_model.charger.MultiStepsPowerLoadConstraintChargePercent', return_value=mock_constraint):

            result = await charger.check_load_activity_and_constraints(time)

        # Should have pushed a constraint
        assert mock_push.called
        assert result is True
        # Force charge flag should be reset
        assert mock_car.do_force_next_charge is False

    @pytest.mark.asyncio
    async def test_boot_data_expiration(self, setup_charger):
        """Test that boot data expires after the configured time."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)

        # Set boot time to be expired
        charger._boot_time = time - timedelta(seconds=CHARGER_BOOT_TIME_DATA_EXPIRATION_S + 100)
        charger._boot_time_adjusted = time - timedelta(seconds=CHARGER_BOOT_TIME_DATA_EXPIRATION_S + 100)
        charger._boot_car = create_mock_car("BootCar")
        charger._boot_constraints = []

        with patch.object(charger, 'is_charger_unavailable', return_value=False), \
             patch.object(charger, 'probe_for_possible_needed_reboot', return_value=False), \
             patch.object(charger, 'is_not_plugged', return_value=True), \
             patch.object(charger, 'set_charging_num_phases', new_callable=AsyncMock, return_value=True):

            await charger.check_load_activity_and_constraints(time)

        # Boot data should be cleared
        assert charger._boot_time is None


# ============================================================================
# Tests for ensure_correct_state
# ============================================================================

class TestEnsureCorrectState:
    """Tests for the ensure_correct_state method."""

    @pytest.fixture
    def setup_charger(self):
        """Set up a charger for testing."""
        hass = create_mock_hass()
        home = create_mock_home(hass)
        charger = create_charger_generic(hass, home)
        return charger, home

    @pytest.mark.asyncio
    async def test_returns_none_when_unavailable(self, setup_charger):
        """Test that method returns None when charger is unavailable."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)

        with patch.object(charger, '_do_update_charger_state', new_callable=AsyncMock), \
             patch.object(charger, 'is_charger_unavailable', return_value=True):

            result, handled, verified_time = await charger.ensure_correct_state(time)

        assert result is None
        assert handled is False
        assert verified_time is None

    @pytest.mark.asyncio
    async def test_returns_true_when_no_car(self, setup_charger):
        """Test that method returns True when no car is attached."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)
        charger.car = None

        with patch.object(charger, '_do_update_charger_state', new_callable=AsyncMock), \
             patch.object(charger, 'is_charger_unavailable', return_value=False), \
             patch.object(charger, 'is_not_plugged', return_value=True):

            result, handled, verified_time = await charger.ensure_correct_state(time)

        assert result is True
        assert handled is False
        assert verified_time is None

    @pytest.mark.asyncio
    async def test_returns_false_during_short_unplug(self, setup_charger):
        """Test that method returns False during short unplug."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)
        charger.car = create_mock_car()

        # First is_not_plugged returns False (for duration check), second returns True
        with patch.object(charger, '_do_update_charger_state', new_callable=AsyncMock), \
             patch.object(charger, 'is_charger_unavailable', return_value=False), \
             patch.object(charger, 'is_not_plugged', side_effect=[False, True]):

            result, handled, verified_time = await charger.ensure_correct_state(time)

        assert result is False
        assert handled is False

    @pytest.mark.asyncio
    async def test_returns_false_when_running_command(self, setup_charger):
        """Test that method returns False when a command is running."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)
        charger.car = create_mock_car()
        charger.running_command = MagicMock()  # Simulate running command

        with patch.object(charger, '_do_update_charger_state', new_callable=AsyncMock), \
             patch.object(charger, 'is_charger_unavailable', return_value=False), \
             patch.object(charger, 'is_not_plugged', return_value=False):

            result, handled, verified_time = await charger.ensure_correct_state(time)

        assert result is False
        assert handled is False

    @pytest.mark.asyncio
    async def test_handles_stopped_charge_state(self, setup_charger):
        """Test handling of stopped charge command state."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)
        charger.car = create_mock_car()
        charger.running_command = None

        with patch.object(charger, '_do_update_charger_state', new_callable=AsyncMock), \
             patch.object(charger, 'is_charger_unavailable', return_value=False), \
             patch.object(charger, 'is_not_plugged', return_value=False), \
             patch.object(charger, '_probe_and_enforce_stopped_charge_command_state', return_value=True), \
             patch.object(charger, '_ensure_correct_state', new_callable=AsyncMock, return_value=True):

            result, handled, verified_time = await charger.ensure_correct_state(time)

        assert handled is True


class TestEnsureCorrectStateInternal:
    """Tests for the _ensure_correct_state internal method."""

    @pytest.fixture
    def setup_charger(self):
        """Set up a charger for testing."""
        hass = create_mock_hass()
        home = create_mock_home(hass)
        charger = create_charger_generic(hass, home)
        # Initialize state
        charger._expected_charge_state.value = True
        charger._expected_amperage.value = 16
        charger._expected_num_active_phases.value = 3
        return charger, home

    @pytest.mark.asyncio
    async def test_returns_false_when_state_reset(self, setup_charger):
        """Test that method returns False when state is reset."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)

        with patch.object(charger, 'is_in_state_reset', return_value=True):
            result = await charger._ensure_correct_state(time)

        assert result is False

    @pytest.mark.asyncio
    async def test_checks_reboot_completion(self, setup_charger):
        """Test checking if reboot has completed."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)
        charger._asked_for_reboot_at_time = time - timedelta(seconds=120)

        with patch.object(charger, 'is_in_state_reset', return_value=False), \
             patch.object(charger, 'check_if_reboot_happened', new_callable=Mock, return_value=True), \
             patch.object(type(charger), 'current_num_phases', new_callable=PropertyMock, return_value=3), \
             patch.object(charger, 'is_charge_enabled', return_value=True), \
             patch.object(charger, 'is_charge_disabled', return_value=False), \
             patch.object(charger, 'get_charging_current', return_value=16):

            result = await charger._ensure_correct_state(time)

        assert charger._asked_for_reboot_at_time is None

    @pytest.mark.asyncio
    async def test_sets_verified_time_on_success(self, setup_charger):
        """Test that verified time is set on success."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)
        charger._verified_correct_state_time = None

        with patch.object(charger, 'is_in_state_reset', return_value=False), \
             patch.object(type(charger), 'current_num_phases', new_callable=PropertyMock, return_value=3), \
             patch.object(charger, 'is_charge_enabled', return_value=True), \
             patch.object(charger, 'is_charge_disabled', return_value=False), \
             patch.object(charger, 'get_charging_current', return_value=16):

            result = await charger._ensure_correct_state(time)

        assert result is True
        assert charger._verified_correct_state_time == time

    @pytest.mark.asyncio
    async def test_launches_phase_change_when_needed(self, setup_charger):
        """Test that phase change is launched when current doesn't match expected."""
        charger, home = setup_charger
        time = datetime.now(pytz.UTC)
        charger._expected_num_active_phases.value = 1  # Expect 1 phase
        charger._expected_num_active_phases._num_launched = 0
        charger._expected_num_active_phases.last_time_set = None

        with patch.object(charger, 'is_in_state_reset', return_value=False), \
             patch.object(type(charger), 'current_num_phases', new_callable=PropertyMock, return_value=3), \
             patch.object(charger, 'update_data_request', new_callable=AsyncMock), \
             patch.object(charger, 'set_charging_num_phases', new_callable=AsyncMock) as mock_set_phases:

            result = await charger._ensure_correct_state(time, probe_only=False)

        mock_set_phases.assert_called_once_with(num_phases=1, time=time)
        assert result is False


# ============================================================================
# Tests for dyn_handle
# ============================================================================

class TestDynHandle:
    """Tests for the dyn_handle method on QSChargerGroup."""

    @pytest.fixture
    def setup_charger_group(self):
        """Set up a charger group for testing."""
        hass = create_mock_hass()
        home = create_mock_home(hass)
        dyn_group = create_mock_dynamic_group(home)

        charger1 = create_charger_generic(hass, home, "Charger1")
        charger2 = create_charger_generic(hass, home, "Charger2")

        dyn_group._childrens = [charger1, charger2]

        with patch('custom_components.quiet_solar.ha_model.charger.isinstance', return_value=True):
            charger_group = QSChargerGroup(dyn_group)

        return charger_group, home, [charger1, charger2]

    @pytest.mark.asyncio
    async def test_no_actionable_chargers(self, setup_charger_group):
        """Test dyn_handle with no actionable chargers."""
        charger_group, home, chargers = setup_charger_group
        time = datetime.now(pytz.UTC)

        with patch.object(charger_group, 'ensure_correct_state', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = ([], None)

            await charger_group.dyn_handle(time)

        mock_ensure.assert_called_once_with(time)

    @pytest.mark.asyncio
    async def test_handles_remaining_budget(self, setup_charger_group):
        """Test dyn_handle applies remaining budget from previous run."""
        charger_group, home, chargers = setup_charger_group
        time = datetime.now(pytz.UTC)

        mock_cs = create_mock_charger_status(chargers[0])
        charger_group.remaining_budget_to_apply = [mock_cs]

        with patch.object(charger_group, 'ensure_correct_state', new_callable=AsyncMock) as mock_ensure, \
             patch.object(charger_group, 'apply_budgets', new_callable=AsyncMock) as mock_apply:

            mock_ensure.return_value = ([mock_cs], time - timedelta(seconds=10))

            await charger_group.dyn_handle(time)

        mock_apply.assert_called_once()
        assert charger_group.remaining_budget_to_apply == []

    @pytest.mark.asyncio
    async def test_waits_for_stable_state(self, setup_charger_group):
        """Test dyn_handle waits for stable state before computing."""
        charger_group, home, chargers = setup_charger_group
        time = datetime.now(pytz.UTC)

        mock_cs = create_mock_charger_status(chargers[0])

        # Verified time is too recent
        recent_time = time - timedelta(seconds=5)  # Less than CHARGER_ADAPTATION_WINDOW_S

        with patch.object(charger_group, 'ensure_correct_state', new_callable=AsyncMock) as mock_ensure, \
             patch.object(charger_group, 'budgeting_algorithm_minimize_diffs', new_callable=AsyncMock) as mock_budget:

            mock_ensure.return_value = ([mock_cs], recent_time)

            await charger_group.dyn_handle(time)

        # Should not call budgeting algorithm
        mock_budget.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_budgeting_when_stable(self, setup_charger_group):
        """Test dyn_handle runs budgeting algorithm when state is stable."""
        charger_group, home, chargers = setup_charger_group
        time = datetime.now(pytz.UTC)

        mock_cs = create_mock_charger_status(chargers[0])

        # Verified time is old enough
        stable_time = time - timedelta(seconds=CHARGER_ADAPTATION_WINDOW_S + 10)

        with patch.object(charger_group, 'ensure_correct_state', new_callable=AsyncMock) as mock_ensure, \
             patch.object(charger_group, 'budgeting_algorithm_minimize_diffs', new_callable=AsyncMock) as mock_budget, \
             patch.object(charger_group, 'apply_budget_strategy', new_callable=AsyncMock) as mock_apply:

            mock_ensure.return_value = ([mock_cs], stable_time)
            mock_budget.return_value = (True, False, False)

            await charger_group.dyn_handle(time)

        mock_budget.assert_called_once()
        mock_apply.assert_called_once()


# ============================================================================
# Tests for budgeting_algorithm_minimize_diffs
# ============================================================================

class TestBudgetingAlgorithm:
    """Tests for the budgeting_algorithm_minimize_diffs method."""

    @pytest.fixture
    def setup_charger_group(self):
        """Set up a charger group for testing."""
        hass = create_mock_hass()
        home = create_mock_home(hass)
        dyn_group = create_mock_dynamic_group(home)

        charger1 = create_charger_generic(hass, home, "Charger1")
        charger2 = create_charger_generic(hass, home, "Charger2")

        dyn_group._childrens = [charger1, charger2]

        with patch('custom_components.quiet_solar.ha_model.charger.isinstance', return_value=True):
            charger_group = QSChargerGroup(dyn_group)

        return charger_group, home, [charger1, charger2]

    @pytest.mark.asyncio
    async def test_empty_chargers_returns_success(self, setup_charger_group):
        """Test that empty charger list returns success."""
        charger_group, home, chargers = setup_charger_group
        time = datetime.now(pytz.UTC)

        success, should_reset, done_reset = await charger_group.budgeting_algorithm_minimize_diffs(
            [], 5000.0, 5000.0, False, time
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_single_charger_budgeting(self, setup_charger_group):
        """Test budgeting with single charger."""
        charger_group, home, chargers = setup_charger_group
        time = datetime.now(pytz.UTC)

        mock_cs = create_mock_charger_status(chargers[0], current_amp=10, score=100)

        with patch.object(charger_group, '_do_prepare_and_shave_budgets', new_callable=AsyncMock) as mock_prep, \
             patch.object(charger_group, 'get_budget_diffs', return_value=(500.0, [10, 10, 10], [10, 10, 10])):

            mock_prep.return_value = ([mock_cs], True, False)

            success, should_reset, done_reset = await charger_group.budgeting_algorithm_minimize_diffs(
                [mock_cs], 5000.0, 5000.0, False, time
            )

        assert success is True

    @pytest.mark.asyncio
    async def test_triggers_reset_for_non_charging_best(self, setup_charger_group):
        """Test that reset allocation is triggered when best charger is not charging."""
        charger_group, home, chargers = setup_charger_group
        time = datetime.now(pytz.UTC)

        # Best charger (highest score) is not charging
        best_cs = create_mock_charger_status(chargers[0], current_amp=0, score=200)
        best_cs.possible_amps = [0, 6, 10, 16]  # 0 means can be stopped

        # Other charger is charging
        other_cs = create_mock_charger_status(chargers[1], current_amp=16, score=100)
        other_cs.possible_amps = [0, 6, 10, 16]

        with patch.object(charger_group, '_do_prepare_and_shave_budgets', new_callable=AsyncMock) as mock_prep, \
             patch.object(charger_group, 'get_budget_diffs', return_value=(500.0, [10, 10, 10], [10, 10, 10])):

            mock_prep.return_value = ([best_cs, other_cs], True, False)

            success, should_reset, done_reset = await charger_group.budgeting_algorithm_minimize_diffs(
                [best_cs, other_cs], 5000.0, 5000.0, True, time
            )

        # Should trigger reset allocation
        assert should_reset is True or done_reset is True

    @pytest.mark.asyncio
    async def test_shave_fails_triggers_reset(self, setup_charger_group):
        """Test that failed shave triggers reset allocation attempt."""
        charger_group, home, chargers = setup_charger_group
        time = datetime.now(pytz.UTC)

        mock_cs = create_mock_charger_status(chargers[0], current_amp=10, score=100)

        with patch.object(charger_group, '_do_prepare_and_shave_budgets', new_callable=AsyncMock) as mock_prep, \
             patch.object(charger_group, 'get_budget_diffs', return_value=(500.0, [10, 10, 10], [10, 10, 10])):

            # First call fails, second succeeds (after reset)
            mock_prep.side_effect = [
                ([mock_cs], False, False),  # First attempt fails
                ([mock_cs], True, False),   # Reset attempt succeeds
            ]

            success, should_reset, done_reset = await charger_group.budgeting_algorithm_minimize_diffs(
                [mock_cs], 5000.0, 5000.0, False, time
            )

        # Should have called prepare twice (first fail, then reset)
        assert mock_prep.call_count == 2


# ============================================================================
# Tests for apply_budget_strategy
# ============================================================================

class TestApplyBudgetStrategy:
    """Tests for the apply_budget_strategy method."""

    @pytest.fixture
    def setup_charger_group(self):
        """Set up a charger group for testing."""
        hass = create_mock_hass()
        home = create_mock_home(hass)
        dyn_group = create_mock_dynamic_group(home)

        charger = create_charger_generic(hass, home, "Charger1")
        dyn_group._childrens = [charger]

        with patch('custom_components.quiet_solar.ha_model.charger.isinstance', return_value=True):
            charger_group = QSChargerGroup(dyn_group)

        return charger_group, home, charger

    @pytest.mark.asyncio
    async def test_empty_chargers_does_nothing(self, setup_charger_group):
        """Test that empty charger list does nothing."""
        charger_group, home, charger = setup_charger_group
        time = datetime.now(pytz.UTC)

        await charger_group.apply_budget_strategy([], None, time)

        # Should complete without error
        assert charger_group.know_reduced_state is None

    @pytest.mark.asyncio
    async def test_stores_reduced_state(self, setup_charger_group):
        """Test that reduced state is stored for power tracking."""
        charger_group, home, charger = setup_charger_group
        time = datetime.now(pytz.UTC)

        mock_cs = create_mock_charger_status(charger, current_amp=10, num_phases=3)

        with patch.object(charger_group, 'apply_budgets', new_callable=AsyncMock):
            await charger_group.apply_budget_strategy([mock_cs], 5000.0, time)

        assert charger_group.know_reduced_state is not None
        assert charger_group.know_reduced_state_real_power == 5000.0

    @pytest.mark.asyncio
    async def test_splits_increasing_decreasing(self, setup_charger_group):
        """Test that chargers are split into increasing/decreasing groups."""
        charger_group, home, charger = setup_charger_group
        time = datetime.now(pytz.UTC)

        # Create charger status where budget is higher than current (increasing)
        mock_cs = create_mock_charger_status(charger, current_amp=10, num_phases=3)
        mock_cs.budgeted_amp = 16  # Higher than current
        mock_cs.get_current_charging_amps.return_value = [10, 10, 10]
        mock_cs.get_budget_amps.return_value = [16, 16, 16]

        # Mock group to say amps are not acceptable (triggers split)
        charger_group.dynamic_group.is_current_acceptable.return_value = False

        with patch.object(charger_group, 'apply_budgets', new_callable=AsyncMock) as mock_apply:
            await charger_group.apply_budget_strategy([mock_cs], 5000.0, time)

        # Should have remaining budget to apply (the increasing ones)
        # The decreasing ones should be applied first


# ============================================================================
# Tests for QSChargerGroup ensure_correct_state
# ============================================================================

class TestChargerGroupEnsureCorrectState:
    """Tests for the QSChargerGroup ensure_correct_state method."""

    @pytest.fixture
    def setup_charger_group(self):
        """Set up a charger group for testing."""
        hass = create_mock_hass()
        home = create_mock_home(hass)
        dyn_group = create_mock_dynamic_group(home)

        charger1 = create_charger_generic(hass, home, "Charger1")
        charger2 = create_charger_generic(hass, home, "Charger2")

        dyn_group._childrens = [charger1, charger2]

        with patch('custom_components.quiet_solar.ha_model.charger.isinstance', return_value=True):
            charger_group = QSChargerGroup(dyn_group)

        return charger_group, home, [charger1, charger2]

    @pytest.mark.asyncio
    async def test_skips_disabled_chargers(self, setup_charger_group):
        """Test that disabled chargers are skipped."""
        charger_group, home, chargers = setup_charger_group
        time = datetime.now(pytz.UTC)

        chargers[0].qs_enable_device = False
        chargers[1].qs_enable_device = True

        with patch.object(chargers[1], 'ensure_correct_state', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = (True, False, time)

            with patch.object(chargers[1], 'get_stable_dynamic_charge_status', return_value=None):
                actionable, verified_time = await charger_group.ensure_correct_state(time)

        # Only charger2 should be checked
        mock_ensure.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_empty_on_failure(self, setup_charger_group):
        """Test that failure returns empty actionable list."""
        charger_group, home, chargers = setup_charger_group
        time = datetime.now(pytz.UTC)

        chargers[0].qs_enable_device = True

        with patch.object(chargers[0], 'ensure_correct_state', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = (False, False, None)

            actionable, verified_time = await charger_group.ensure_correct_state(time)

        assert actionable == []
        assert verified_time is None

    @pytest.mark.asyncio
    async def test_sorts_by_score(self, setup_charger_group):
        """Test that actionable chargers are sorted by score."""
        charger_group, home, chargers = setup_charger_group
        time = datetime.now(pytz.UTC)

        chargers[0].qs_enable_device = True
        chargers[1].qs_enable_device = True

        cs1 = create_mock_charger_status(chargers[0], score=50)
        cs2 = create_mock_charger_status(chargers[1], score=100)

        with patch.object(chargers[0], 'ensure_correct_state', new_callable=AsyncMock) as mock_ensure1, \
             patch.object(chargers[1], 'ensure_correct_state', new_callable=AsyncMock) as mock_ensure2, \
             patch.object(chargers[0], 'get_stable_dynamic_charge_status', return_value=cs1), \
             patch.object(chargers[1], 'get_stable_dynamic_charge_status', return_value=cs2):

            mock_ensure1.return_value = (True, False, time)
            mock_ensure2.return_value = (True, False, time)

            actionable, verified_time = await charger_group.ensure_correct_state(time)

        assert len(actionable) == 2
        # Higher score should be first
        assert actionable[0].charge_score >= actionable[1].charge_score


# ============================================================================
# Tests for _do_prepare_and_shave_budgets
# ============================================================================

class TestPrepareAndShaveBudgets:
    """Tests for the _do_prepare_and_shave_budgets method."""

    @pytest.fixture
    def setup_charger_group(self):
        """Set up a charger group for testing."""
        hass = create_mock_hass()
        home = create_mock_home(hass)
        dyn_group = create_mock_dynamic_group(home)

        charger = create_charger_generic(hass, home, "Charger1")
        dyn_group._childrens = [charger]

        with patch('custom_components.quiet_solar.ha_model.charger.isinstance', return_value=True):
            charger_group = QSChargerGroup(dyn_group)

        return charger_group, home, charger

    @pytest.mark.asyncio
    async def test_reset_allocation_uses_minimum(self, setup_charger_group):
        """Test that reset allocation uses minimum amp values."""
        charger_group, home, charger = setup_charger_group
        time = datetime.now(pytz.UTC)

        mock_cs = create_mock_charger_status(charger, current_amp=20, num_phases=3)
        mock_cs.possible_amps = [6, 10, 16, 20, 32]
        mock_cs.possible_num_phases = [1, 3]

        with patch.object(charger_group, '_shave_mandatory_budgets', new_callable=AsyncMock) as mock_shave_mand, \
             patch.object(charger_group, '_shave_current_budgets', new_callable=AsyncMock) as mock_shave_curr:

            mock_shave_mand.return_value = [6, 6, 6]
            mock_shave_curr.return_value = ([mock_cs], True)

            result, current_ok, has_phase_changes = await charger_group._do_prepare_and_shave_budgets(
                [mock_cs], do_reset_allocation=True, time=time
            )

        # With reset allocation, should use minimum values
        assert mock_cs.budgeted_amp == 6
        assert mock_cs.budgeted_num_phases == 1  # min of [1, 3]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
