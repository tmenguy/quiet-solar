"""Comprehensive tests for QSChargerGeneric and related classes in ha_model/charger.py.

This test file targets the 40% -> 80%+ coverage gap by testing:
- QSStateCmd state machine
- QSChargerStatus budget management
- QSChargerGroup coordination
- Power budgeting algorithms
- Charger state transitions
- Car-charger integration
"""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from datetime import timedelta, time as dt_time

from homeassistant.const import (
    Platform,
    STATE_UNKNOWN,
    STATE_UNAVAILABLE,
    CONF_NAME,
)
import pytz

from custom_components.quiet_solar.ha_model.charger import (
    QSChargerGeneric,
    QSChargerWallbox,
    QSChargerOCPP,
    QSChargerGroup,
    QSChargerStatus,
    QSStateCmd,
    QSChargerStates,
    WallboxChargerStatus,
    QSOCPPv16v201ChargePointStatus,
    STATE_CMD_RETRY_NUMBER,
    STATE_CMD_TIME_BETWEEN_RETRY_S,
    CHARGER_BOOT_TIME_DATA_EXPIRATION_S,
    CHARGER_ADAPTATION_WINDOW_S,
    TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_OFF_TO_ON_S,
    TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_ON_TO_OFF_S,
)
from custom_components.quiet_solar.ha_model.dynamic_group import QSDynamicGroup
from custom_components.quiet_solar.home_model.commands import (
    LoadCommand,
    CMD_ON,
    CMD_OFF,
    CMD_AUTO_FROM_CONSIGN,
    CMD_AUTO_GREEN_ONLY,
    CMD_AUTO_GREEN_CAP,
    CMD_AUTO_GREEN_CONSIGN,
    CMD_AUTO_PRICE,
)
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CHARGER_CONSUMPTION,
    CONF_IS_3P,
    CONF_MONO_PHASE,
    CONF_CHARGER_STATUS_SENSOR,
    CONF_CHARGER_PLUGGED,
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER,
    CONF_CHARGER_PAUSE_RESUME_SWITCH,
    CHARGER_NO_CAR_CONNECTED,
)

from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from tests.factories import create_minimal_home_model


def _charger_config_entry() -> MockConfigEntry:
    """Create MockConfigEntry for charger tests."""
    return MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_charger_entry",
        data={CONF_NAME: "Test Charger"},
        title="Test Charger",
    )


def _charger_home():
    """Create mock home for charger tests."""
    home = create_minimal_home_model()
    home.is_off_grid = MagicMock(return_value=False)
    home.force_next_solve = MagicMock()
    return home


@pytest.fixture
def charger_config_entry() -> MockConfigEntry:
    """Config entry for charger tests."""
    return _charger_config_entry()


@pytest.fixture
def charger_home():
    """Mock home for charger tests."""
    return _charger_home()


@pytest.fixture
def charger_data_handler(charger_home):
    """Data handler for charger tests."""
    handler = MagicMock()
    handler.home = charger_home
    return handler


@pytest.fixture
def charger_hass_data(hass: HomeAssistant, charger_data_handler):
    """Set hass.data[DOMAIN][DATA_HANDLER] for charger tests."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][DATA_HANDLER] = charger_data_handler


# ============================================================================
# Tests for QSStateCmd class
# ============================================================================

class TestQSStateCmd:
    """Test QSStateCmd state machine."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        cmd = QSStateCmd()

        assert cmd.value is None
        assert cmd._num_launched == 0
        assert cmd._num_set == 0
        assert cmd.last_time_set is None
        assert cmd.last_change_asked is None

    def test_init_custom_retries(self):
        """Test initialization with custom retry settings."""
        cmd = QSStateCmd(
            initial_num_in_out_immediate=2,
            command_retries_s=30.0
        )

        assert cmd.initial_num_in_out_immediate == 2
        assert cmd.command_retries_s == 30.0

    def test_reset_clears_state(self):
        """Test reset clears all state."""
        cmd = QSStateCmd()
        cmd.value = True
        cmd._num_launched = 3
        cmd._num_set = 2
        cmd.last_time_set = datetime.datetime.now(pytz.UTC)

        cmd.reset()

        assert cmd.value is None
        assert cmd._num_launched == 0
        assert cmd._num_set == 0
        assert cmd.last_time_set is None

    def test_set_new_value_returns_true(self):
        """Test set returns True when value changes."""
        cmd = QSStateCmd()
        time = datetime.datetime.now(pytz.UTC)

        result = cmd.set(True, time)

        assert result is True
        assert cmd.value is True
        assert cmd.last_change_asked == time

    def test_set_same_value_returns_false(self):
        """Test set returns False when value unchanged."""
        cmd = QSStateCmd()
        cmd.value = True

        result = cmd.set(True, datetime.datetime.now(pytz.UTC))

        assert result is False

    def test_set_increments_num_set(self):
        """Test set increments _num_set counter."""
        cmd = QSStateCmd()
        time = datetime.datetime.now(pytz.UTC)

        cmd.set(True, time)
        assert cmd._num_set == 1

        cmd.set(False, time + timedelta(seconds=1))
        assert cmd._num_set == 2

    def test_is_ok_to_set_first_time(self):
        """Test is_ok_to_set returns True on first call."""
        cmd = QSStateCmd()
        time = datetime.datetime.now(pytz.UTC)

        result = cmd.is_ok_to_set(time, min_change_time=60.0)

        assert result is True

    def test_is_ok_to_set_within_min_time(self):
        """Test is_ok_to_set returns False within min_change_time."""
        cmd = QSStateCmd()
        time = datetime.datetime.now(pytz.UTC)

        cmd.set(True, time)
        cmd._num_set = 5  # Past initial immediate period

        result = cmd.is_ok_to_set(time + timedelta(seconds=30), min_change_time=60.0)

        assert result is False

    def test_is_ok_to_set_after_min_time(self):
        """Test is_ok_to_set returns True after min_change_time."""
        cmd = QSStateCmd()
        time = datetime.datetime.now(pytz.UTC)

        cmd.set(True, time)
        cmd._num_set = 5

        result = cmd.is_ok_to_set(time + timedelta(seconds=120), min_change_time=60.0)

        assert result is True

    def test_is_ok_to_set_initial_immediate(self):
        """Test is_ok_to_set returns True during initial immediate period."""
        cmd = QSStateCmd(initial_num_in_out_immediate=3)
        time = datetime.datetime.now(pytz.UTC)

        cmd.set(True, time)
        # _num_set = 1, which is <= initial_num_in_out_immediate (3)

        result = cmd.is_ok_to_set(time + timedelta(seconds=1), min_change_time=60.0)

        assert result is True

    def test_is_ok_to_launch_first_time(self):
        """Test is_ok_to_launch returns True on first launch."""
        cmd = QSStateCmd()
        time = datetime.datetime.now(pytz.UTC)

        result = cmd.is_ok_to_launch(True, time)

        assert result is True

    def test_is_ok_to_launch_exceeds_retry_limit(self):
        """Test is_ok_to_launch returns False after retry limit."""
        cmd = QSStateCmd()
        cmd._num_launched = STATE_CMD_RETRY_NUMBER + 1
        cmd.value = True
        time = datetime.datetime.now(pytz.UTC)

        result = cmd.is_ok_to_launch(True, time)

        assert result is False

    def test_is_ok_to_launch_within_retry_window(self):
        """Test is_ok_to_launch returns False within retry window."""
        cmd = QSStateCmd()
        time = datetime.datetime.now(pytz.UTC)

        cmd.value = True
        cmd._num_launched = 2
        cmd.last_time_set = time

        result = cmd.is_ok_to_launch(True, time + timedelta(seconds=1))

        assert result is False

    def test_is_ok_to_launch_after_retry_window(self):
        """Test is_ok_to_launch returns True after retry window."""
        cmd = QSStateCmd()
        time = datetime.datetime.now(pytz.UTC)

        cmd.value = True
        cmd._num_launched = 2
        cmd.last_time_set = time

        result = cmd.is_ok_to_launch(
            True,
            time + timedelta(seconds=STATE_CMD_TIME_BETWEEN_RETRY_S + 10)
        )

        assert result is True

    def test_register_launch_increments_counter(self):
        """Test register_launch increments _num_launched."""
        cmd = QSStateCmd()
        time = datetime.datetime.now(pytz.UTC)

        cmd.register_launch(True, time)

        assert cmd._num_launched == 1
        assert cmd.last_time_set == time

    @pytest.mark.asyncio
    async def test_success_resets_launch_state(self):
        """Test success resets launch-related state."""
        cmd = QSStateCmd()
        cmd._num_launched = 3
        cmd.last_time_set = datetime.datetime.now(pytz.UTC)
        time = datetime.datetime.now(pytz.UTC)

        await cmd.success(time)

        assert cmd._num_launched == 0
        assert cmd.last_time_set is None
        assert cmd.first_time_success == time

    @pytest.mark.asyncio
    async def test_success_calls_callback(self):
        """Test success calls registered callback."""
        cmd = QSStateCmd()
        callback = AsyncMock()
        cmd.on_success_action_cb = callback
        cmd.on_success_action_cb_kwargs = {"arg1": "value1"}

        time = datetime.datetime.now(pytz.UTC)
        await cmd.success(time)

        callback.assert_called_once_with(time=time, arg1="value1")

    @pytest.mark.asyncio
    async def test_success_clears_callback(self):
        """Test success clears callback after calling."""
        cmd = QSStateCmd()
        cmd.on_success_action_cb = AsyncMock()
        cmd.on_success_action_cb_kwargs = {}

        await cmd.success(datetime.datetime.now(pytz.UTC))

        assert cmd.on_success_action_cb is None
        assert cmd.on_success_action_cb_kwargs is None

    def test_register_success_cb(self):
        """Test register_success_cb stores callback."""
        cmd = QSStateCmd()
        callback = AsyncMock()
        kwargs = {"key": "value"}

        cmd.register_success_cb(callback, kwargs)

        assert cmd.on_success_action_cb == callback
        assert cmd.on_success_action_cb_kwargs == kwargs

    def test_register_success_cb_none_kwargs(self):
        """Test register_success_cb with None kwargs."""
        cmd = QSStateCmd()
        callback = AsyncMock()

        cmd.register_success_cb(callback, None)

        assert cmd.on_success_action_cb_kwargs == {}


# ============================================================================
# Tests for QSChargerStatus class
# ============================================================================

class TestQSChargerStatus:
    """Test QSChargerStatus budget management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_charger = MagicMock()
        self.mock_charger.name = "Test Charger"
        self.mock_charger.min_charge = 6
        self.mock_charger.max_charge = 32
        self.mock_charger.physical_3p = True
        self.mock_charger.mono_phase_index = 0

    def test_init_defaults(self):
        """Test initialization with defaults."""
        status = QSChargerStatus(self.mock_charger)

        assert status.charger == self.mock_charger
        assert status.accurate_current_power is None
        assert status.command is None
        assert status.current_real_max_charging_amp is None
        assert status.budgeted_amp is None
        assert status.charge_score == 0

    def test_name_property_no_car(self):
        """Test name property without car."""
        self.mock_charger.car = None
        status = QSChargerStatus(self.mock_charger)

        assert status.name == "Test Charger/NO CAR"

    def test_name_property_with_car(self):
        """Test name property with car."""
        mock_car = MagicMock()
        mock_car.name = "Tesla Model 3"
        self.mock_charger.car = mock_car
        status = QSChargerStatus(self.mock_charger)

        assert status.name == "Test Charger/Tesla Model 3"

    def test_duplicate(self):
        """Test duplicate creates a copy."""
        status = QSChargerStatus(self.mock_charger)
        status.accurate_current_power = 5000.0
        status.budgeted_amp = 16
        status.budgeted_num_phases = 3
        status.charge_score = 100

        copy = status.duplicate()

        assert copy.charger == status.charger
        assert copy.accurate_current_power == 5000.0
        assert copy.budgeted_amp == 16
        assert copy.charge_score == 100

    def test_get_amps_from_values_single_phase(self):
        """Test get_amps_from_values for single phase."""
        status = QSChargerStatus(self.mock_charger)
        self.mock_charger.mono_phase_index = 1

        result = status.get_amps_from_values(16, 1)

        assert result == [0.0, 16, 0.0]

    def test_get_amps_from_values_three_phase(self):
        """Test get_amps_from_values for three phase."""
        status = QSChargerStatus(self.mock_charger)

        result = status.get_amps_from_values(16, 3)

        assert result == [16, 16, 16]

    def test_get_current_charging_amps(self):
        """Test get_current_charging_amps."""
        status = QSChargerStatus(self.mock_charger)
        status.current_real_max_charging_amp = 20
        status.current_active_phase_number = 3

        result = status.get_current_charging_amps()

        assert result == [20, 20, 20]

    def test_get_budget_amps(self):
        """Test get_budget_amps."""
        status = QSChargerStatus(self.mock_charger)
        status.budgeted_amp = 12
        status.budgeted_num_phases = 1
        self.mock_charger.mono_phase_index = 2

        result = status.get_budget_amps()

        assert result == [0.0, 0.0, 12]

    def test_can_change_budget_increase_from_zero(self):
        """Test can_change_budget increase from zero."""
        status = QSChargerStatus(self.mock_charger)
        status.budgeted_amp = 0
        status.budgeted_num_phases = 3
        status.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        status.possible_num_phases = [3]

        next_amp, next_phases = status.can_change_budget(
            allow_state_change=True,
            allow_phase_change=False,
            increase=True
        )

        # Should go to minimum charge (6)
        assert next_amp == 6
        assert next_phases == 3

    def test_can_change_budget_increase_normal(self):
        """Test can_change_budget normal increase."""
        status = QSChargerStatus(self.mock_charger)
        status.budgeted_amp = 16
        status.budgeted_num_phases = 3
        status.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        status.possible_num_phases = [3]

        next_amp, next_phases = status.can_change_budget(
            allow_state_change=False,
            allow_phase_change=False,
            increase=True
        )

        assert next_amp == 17
        assert next_phases == 3

    def test_can_change_budget_decrease_normal(self):
        """Test can_change_budget normal decrease."""
        status = QSChargerStatus(self.mock_charger)
        status.budgeted_amp = 16
        status.budgeted_num_phases = 3
        status.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        status.possible_num_phases = [3]

        next_amp, next_phases = status.can_change_budget(
            allow_state_change=False,
            allow_phase_change=False,
            increase=False
        )

        assert next_amp == 15
        assert next_phases == 3

    def test_can_change_budget_decrease_to_zero(self):
        """Test can_change_budget decrease to zero."""
        status = QSChargerStatus(self.mock_charger)
        status.budgeted_amp = 6
        status.budgeted_num_phases = 1
        status.possible_amps = [0, 6, 7, 8]
        status.possible_num_phases = [1]

        next_amp, next_phases = status.can_change_budget(
            allow_state_change=True,
            allow_phase_change=False,
            increase=False
        )

        assert next_amp == 0
        assert next_phases == 1

    def test_can_change_budget_at_max_returns_none(self):
        """Test can_change_budget at max returns None."""
        status = QSChargerStatus(self.mock_charger)
        status.budgeted_amp = 32
        status.budgeted_num_phases = 3
        status.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        status.possible_num_phases = [3]

        next_amp, next_phases = status.can_change_budget(
            allow_state_change=False,
            allow_phase_change=False,
            increase=True
        )

        assert next_amp is None

    def test_can_change_budget_none_budgeted_amp(self):
        """Test can_change_budget with None budgeted_amp."""
        status = QSChargerStatus(self.mock_charger)
        status.budgeted_amp = None
        status.possible_num_phases = [3]

        next_amp, next_phases = status.can_change_budget()

        assert next_amp is None

    def test_get_amps_phase_switch_1p_to_3p(self):
        """Test get_amps_phase_switch from 1 phase to 3 phase."""
        status = QSChargerStatus(self.mock_charger)
        status.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        try_amps, to_phase, amps_list = status.get_amps_phase_switch(
            from_amp=18, from_num_phase=1, delta_for_borders=0
        )

        # 18A 1P -> 6A 3P (18/3 = 6)
        assert to_phase == 3
        assert try_amps == 6

    def test_get_amps_phase_switch_3p_to_1p(self):
        """Test get_amps_phase_switch from 3 phase to 1 phase."""
        status = QSChargerStatus(self.mock_charger)
        status.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        try_amps, to_phase, amps_list = status.get_amps_phase_switch(
            from_amp=6, from_num_phase=3, delta_for_borders=0
        )

        # 6A 3P -> 18A 1P (6*3 = 18)
        assert to_phase == 1
        assert try_amps == 18


# ============================================================================
# Tests for QSChargerGroup class
# ============================================================================

class TestQSChargerGroup:
    """Test QSChargerGroup coordination."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dynamic_group = MagicMock(spec=QSDynamicGroup)
        self.mock_dynamic_group.name = "Test Group"
        self.mock_dynamic_group._childrens = []
        self.mock_dynamic_group.home = create_minimal_home_model()

    def test_init_with_no_chargers(self):
        """Test initialization with no chargers."""
        group = QSChargerGroup(self.mock_dynamic_group)

        assert group._chargers == []
        assert group.charger_consumption_W == 0.0
        assert group.dync_group_chargers_only is True

    def test_init_with_chargers(self):
        """Test initialization with chargers."""
        mock_charger1 = MagicMock(spec=QSChargerGeneric)
        mock_charger1.charger_consumption_W = 50.0
        mock_charger2 = MagicMock(spec=QSChargerGeneric)
        mock_charger2.charger_consumption_W = 70.0

        self.mock_dynamic_group._childrens = [mock_charger1, mock_charger2]

        group = QSChargerGroup(self.mock_dynamic_group)

        assert len(group._chargers) == 2
        assert group.charger_consumption_W == 120.0

    def test_init_mixed_devices(self):
        """Test initialization with mixed device types."""
        mock_charger = MagicMock(spec=QSChargerGeneric)
        mock_charger.charger_consumption_W = 50.0
        mock_other = MagicMock()  # Not a charger

        self.mock_dynamic_group._childrens = [mock_charger, mock_other]

        group = QSChargerGroup(self.mock_dynamic_group)

        assert len(group._chargers) == 1
        assert group.dync_group_chargers_only is False

    def test_dampening_power_value_below_consumption(self):
        """Test dampening filters out values below charger consumption."""
        mock_charger = MagicMock(spec=QSChargerGeneric)
        mock_charger.charger_consumption_W = 100.0
        self.mock_dynamic_group._childrens = [mock_charger]

        group = QSChargerGroup(self.mock_dynamic_group)

        # Value below consumption should return 0
        result = group.dampening_power_value_for_car_consumption(50.0)
        assert result == 0.0

        # Value above consumption should pass through
        result = group.dampening_power_value_for_car_consumption(200.0)
        assert result == 200.0

    def test_dampening_power_value_none(self):
        """Test dampening handles None value."""
        group = QSChargerGroup(self.mock_dynamic_group)

        result = group.dampening_power_value_for_car_consumption(None)
        assert result is None

    def test_get_budget_diffs(self):
        """Test get_budget_diffs calculation."""
        mock_charger = MagicMock(spec=QSChargerGeneric)
        mock_charger.charger_consumption_W = 50.0
        self.mock_dynamic_group._childrens = [mock_charger]

        group = QSChargerGroup(self.mock_dynamic_group)

        # Create mock charger status
        mock_status = MagicMock(spec=QSChargerStatus)
        mock_status.get_current_charging_amps.return_value = [10.0, 10.0, 10.0]
        mock_status.get_budget_amps.return_value = [16.0, 16.0, 16.0]
        mock_status.current_real_max_charging_amp = 10
        mock_status.current_active_phase_number = 3
        mock_status.budgeted_amp = 16
        mock_status.budgeted_num_phases = 3
        mock_status.get_diff_power.return_value = 1380.0  # 6A * 230V

        diff_power, new_sum_amps, current_amps = group.get_budget_diffs([mock_status])

        assert diff_power == 1380.0
        assert new_sum_amps == [16.0, 16.0, 16.0]
        assert current_amps == [10.0, 10.0, 10.0]


# ============================================================================
# Tests for QSChargerGeneric class
# ============================================================================

class TestQSChargerGenericInit:
    """Test QSChargerGeneric initialization."""

    def test_init_with_minimal_params(
        self,
        hass: HomeAssistant,
        charger_config_entry: MockConfigEntry,
        charger_home,
        charger_data_handler,
        charger_hass_data,
    ):
        """Test initialization with minimal parameters."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(
                hass=hass,
                config_entry=charger_config_entry,
                home=charger_home,
                **{
                    CONF_NAME: "Test Charger",
                    CONF_CHARGER_MIN_CHARGE: 6,
                    CONF_CHARGER_MAX_CHARGE: 32,
                    CONF_IS_3P: True,
                }
            )

            assert charger.name == "Test Charger"
            assert charger.min_charge == 6
            assert charger.max_charge == 32
            assert charger.physical_3p is True

    def test_init_with_consumption(
        self,
        hass: HomeAssistant,
        charger_config_entry: MockConfigEntry,
        charger_home,
        charger_data_handler,
        charger_hass_data,
    ):
        """Test initialization with charger consumption."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(
                hass=hass,
                config_entry=charger_config_entry,
                home=charger_home,
                **{
                    CONF_NAME: "Test Charger",
                    CONF_CHARGER_MIN_CHARGE: 6,
                    CONF_CHARGER_MAX_CHARGE: 32,
                    CONF_IS_3P: True,
                    CONF_CHARGER_CONSUMPTION: 100,
                }
            )

            assert charger.charger_consumption_W == 100

    def test_init_mono_phase_index(
        self,
        hass: HomeAssistant,
        charger_config_entry: MockConfigEntry,
        charger_home,
        charger_data_handler,
        charger_hass_data,
    ):
        """Test mono phase index initialization."""
        with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
            charger = QSChargerGeneric(
                hass=hass,
                config_entry=charger_config_entry,
                home=charger_home,
                **{
                    CONF_NAME: "Test Charger",
                    CONF_CHARGER_MIN_CHARGE: 6,
                    CONF_CHARGER_MAX_CHARGE: 32,
                    CONF_IS_3P: False,
                    CONF_MONO_PHASE: 2,
                }
            )

            assert charger.mono_phase_index == 1  # 0-indexed


# ============================================================================
# Tests for charger state management
# ============================================================================

@pytest.fixture
def charger_generic(hass, charger_config_entry, charger_home, charger_data_handler, charger_hass_data):
    """Create QSChargerGeneric instance for tests."""
    with patch('custom_components.quiet_solar.ha_model.charger.entity_registry'):
        return QSChargerGeneric(
            hass=hass,
            config_entry=charger_config_entry,
            home=charger_home,
            **{
                CONF_NAME: "Test Charger",
                CONF_CHARGER_MIN_CHARGE: 6,
                CONF_CHARGER_MAX_CHARGE: 32,
                CONF_IS_3P: True,
            }
        )


class TestQSChargerStateManagement:
    """Test charger state management methods."""

    def test_is_plugged_returns_none_when_no_sensor(self, charger_generic):
        """Test is_plugged returns None when no sensor configured."""
        charger_generic.charger_plugged = None
        time = datetime.datetime.now(pytz.UTC)

        result = charger_generic.is_plugged(time)

        # Should return None when no sensor configured
        assert result is None

    def test_can_do_3_to_1_phase_switch_without_switch(self, charger_generic):
        """Test can_do_3_to_1_phase_switch returns False without switch."""
        charger_generic.charger_3_to_1_phase_switch = None

        result = charger_generic.can_do_3_to_1_phase_switch()

        assert result is False

    def test_car_property_no_car(self, charger_generic):
        """Test car property when no car connected."""
        # By default, no car is set
        assert charger_generic.car is None

    def test_user_selected_car_name(self, charger_generic):
        """Test _user_selected_car_name property."""
        charger_generic._user_selected_car_name = "My Tesla"

        assert charger_generic._user_selected_car_name == "My Tesla"


# ============================================================================
# Tests for Wallbox and OCPP charger status parsing
# ============================================================================

class TestWallboxChargerStatus:
    """Test Wallbox charger status enum."""

    def test_charging_status(self):
        """Test charging status value."""
        assert WallboxChargerStatus.CHARGING == "Charging"

    def test_disconnected_status(self):
        """Test disconnected status value."""
        assert WallboxChargerStatus.DISCONNECTED == "Disconnected"

    def test_paused_status(self):
        """Test paused status value."""
        assert WallboxChargerStatus.PAUSED == "Paused"


class TestOCPPChargerStatus:
    """Test OCPP charger status enum."""

    def test_available_status(self):
        """Test available status value."""
        assert QSOCPPv16v201ChargePointStatus.available == "Available"

    def test_charging_status(self):
        """Test charging status value."""
        assert QSOCPPv16v201ChargePointStatus.charging == "Charging"

    def test_suspended_evse_status(self):
        """Test suspended EVSE status value."""
        assert QSOCPPv16v201ChargePointStatus.suspended_evse == "SuspendedEVSE"

    def test_suspended_ev_status(self):
        """Test suspended EV status value."""
        assert QSOCPPv16v201ChargePointStatus.suspended_ev == "SuspendedEV"


# ============================================================================
# Tests for power budget algorithms
# ============================================================================

class TestChargerPowerBudget:
    """Test power budget calculation methods."""

    def test_theoretical_power_calculation(self, charger_generic):
        """Test theoretical power calculation for given amps."""
        # For 16A at 230V three phase, theoretical: 16A * 230V * 3 = 11,040W
        # The charger should have voltage from home
        assert charger_generic.voltage == 230.0
        assert charger_generic.min_charge == 6
        assert charger_generic.max_charge == 32

    def test_update_amps_with_delta_increase(self, charger_generic):
        """Test update_amps_with_delta with positive delta."""
        from_amps = [10.0, 10.0, 10.0]

        result = charger_generic.update_amps_with_delta(from_amps, delta=2, is_3p=True)

        assert result == [12.0, 12.0, 12.0]

    def test_update_amps_with_delta_decrease(self, charger_generic):
        """Test update_amps_with_delta with negative delta."""
        from_amps = [10.0, 10.0, 10.0]

        result = charger_generic.update_amps_with_delta(from_amps, delta=-2, is_3p=True)

        assert result == [8.0, 8.0, 8.0]

    def test_update_amps_with_delta_single_phase(self, charger_generic):
        """Test update_amps_with_delta for single phase."""
        from_amps = [0.0, 10.0, 0.0]

        result = charger_generic.update_amps_with_delta(from_amps, delta=3, is_3p=False)

        # For single phase, only one element should change
        total = sum(result)
        assert total == 13.0


# ============================================================================
# Tests for charger check_load_activity_and_constraints
# ============================================================================

class TestChargerCheckLoadActivity:
    """Test check_load_activity_and_constraints method."""

    @pytest.mark.asyncio
    async def test_check_load_returns_false_during_reboot(self, charger_generic):
        """Test returns False during reboot window."""
        time = datetime.datetime.now(pytz.UTC)
        charger_generic._asked_for_reboot_at_time = time - timedelta(seconds=30)

        result = await charger_generic.check_load_activity_and_constraints(time)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_load_boot_time_window(self, charger_generic):
        """Test behavior during boot time window."""
        time = datetime.datetime.now(pytz.UTC)
        charger_generic._boot_time = time - timedelta(seconds=5)

        with patch.object(charger_generic, 'is_charger_unavailable', return_value=False), \
             patch.object(charger_generic, 'probe_for_possible_needed_reboot', return_value=False):
            result = await charger_generic.check_load_activity_and_constraints(time)

        # Should return False during boot window
        assert result is False
