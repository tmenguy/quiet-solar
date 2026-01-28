"""Tests for QSBiStateDuration class in ha_model/bistate_duration.py."""
from __future__ import annotations

import datetime
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import time as dt_time
import pytz

from homeassistant.const import (
    Platform,
    STATE_UNKNOWN,
    STATE_UNAVAILABLE,
    CONF_NAME,
)
from homeassistant.components.recorder.models import LazyState

from custom_components.quiet_solar.ha_model.bistate_duration import (
    QSBiStateDuration,
    bistate_modes,
    MAX_USER_OVERRIDE_DURATION_S,
)
from custom_components.quiet_solar.home_model.commands import (
    CMD_ON,
    CMD_OFF,
    CMD_IDLE,
)
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    CONSTRAINT_TYPE_FILLER_AUTO,
    CONF_SWITCH,
    SOLVER_STEP_S,
    CONF_ACCURATE_POWER_SENSOR,
)

from tests.test_helpers import FakeHass, FakeConfigEntry


class ConcreteBiStateDevice(QSBiStateDuration):
    """Concrete implementation of QSBiStateDuration for testing."""

    def __init__(self, **kwargs):
        # Handle switch_entity by converting to CONF_SWITCH key that AbstractLoad expects
        if "switch_entity" in kwargs:
            kwargs[CONF_SWITCH] = kwargs.pop("switch_entity")
        elif CONF_SWITCH not in kwargs:
            kwargs[CONF_SWITCH] = "switch.test_device"
        super().__init__(**kwargs)
        self._execute_command_system_calls = []

    async def execute_command_system(self, time, command, state):
        """Track calls to execute_command_system."""
        self._execute_command_system_calls.append((time, command, state))
        return True

    def get_virtual_current_constraint_translation_key(self):
        return "test_constraint_key"

    def get_select_translation_key(self):
        return "test_select_key"


class TestQSBiStateDurationInit:
    """Test QSBiStateDuration initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_init_default_values(self):
        """Test initialization with default values."""
        device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device"}
        )

        assert device.bistate_mode == "bistate_mode_auto"
        assert device.default_on_duration == 1.0
        assert device.default_on_finish_time == dt_time(hour=0, minute=0, second=0)
        assert device.override_duration == MAX_USER_OVERRIDE_DURATION_S // 3600
        assert device._state_on == "on"
        assert device._state_off == "off"
        assert device._bistate_mode_on == "bistate_mode_on"
        assert device._bistate_mode_off == "bistate_mode_off"
        assert device.is_load_time_sensitive is True

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            switch_entity="switch.custom_device",
            **{CONF_NAME: "Custom Device"}
        )

        assert device.bistate_entity == "switch.custom_device"

    def test_bistate_entity_equals_switch_entity(self):
        """Test that bistate_entity is set to switch_entity."""
        device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            switch_entity="switch.my_switch",
            **{CONF_NAME: "Test Device"}
        )

        assert device.bistate_entity == "switch.my_switch"


class TestQSBiStateDurationPowerFromSwitchState:
    """Test get_power_from_switch_state method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device"}
        )
        self.device.power_use = 1000.0

    def test_power_from_state_on(self):
        """Test get_power_from_switch_state returns power_use for ON state."""
        result = self.device.get_power_from_switch_state("on")
        assert result == 1000.0

    def test_power_from_state_off(self):
        """Test get_power_from_switch_state returns 0 for OFF state."""
        result = self.device.get_power_from_switch_state("off")
        assert result == 0.0

    def test_power_from_state_none(self):
        """Test get_power_from_switch_state returns None for None state."""
        result = self.device.get_power_from_switch_state(None)
        assert result is None

    def test_power_from_state_other(self):
        """Test get_power_from_switch_state returns 0 for other states."""
        result = self.device.get_power_from_switch_state("unknown")
        assert result == 0.0


class TestQSBiStateDurationModes:
    """Test bistate mode related methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_get_bistate_modes_with_user_override_support(self):
        """Test get_bistate_modes when user override is supported."""
        device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device"}
        )
        device.load_is_auto_to_be_boosted = False

        modes = device.get_bistate_modes()

        # Should include base modes plus on/off modes
        assert "bistate_mode_auto" in modes
        assert "bistate_mode_exact_calendar" in modes
        assert "bistate_mode_default" in modes
        assert "bistate_mode_on" in modes
        assert "bistate_mode_off" in modes

    def test_get_bistate_modes_without_user_override_support(self):
        """Test get_bistate_modes when user override is not supported."""
        device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device"}
        )
        device.load_is_auto_to_be_boosted = True

        modes = device.get_bistate_modes()

        # Should only include base modes
        assert modes == bistate_modes
        assert "bistate_mode_on" not in modes
        assert "bistate_mode_off" not in modes

    def test_support_green_only_switch_non_boosted(self):
        """Test support_green_only_switch for non-boosted load."""
        device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device"}
        )
        device.load_is_auto_to_be_boosted = False

        assert device.support_green_only_switch() is True

    def test_support_green_only_switch_boosted(self):
        """Test support_green_only_switch for boosted load."""
        device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device"}
        )
        device.load_is_auto_to_be_boosted = True

        assert device.support_green_only_switch() is False

    def test_support_user_override_non_boosted(self):
        """Test support_user_override for non-boosted load."""
        device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device"}
        )
        device.load_is_auto_to_be_boosted = False

        assert device.support_user_override() is True

    def test_support_user_override_boosted(self):
        """Test support_user_override for boosted load."""
        device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device"}
        )
        device.load_is_auto_to_be_boosted = True

        assert device.support_user_override() is False


class TestQSBiStateDurationPowerUse:
    """Test power_use property computations."""

    async def test_power_use_uses_average_sensor_from_history(self):
        """Test power_use uses get_average_sensor over LazyState history."""
        hass = FakeHass()
        config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        home = MagicMock()
        data_handler = MagicMock()
        data_handler.home = home
        hass.data[DOMAIN][DATA_HANDLER] = data_handler

        entity_id = "sensor.test_power"
        device = ConcreteBiStateDevice(
            hass=hass,
            config_entry=config_entry,
            home=home,
            **{
                CONF_NAME: "Test Device",
                CONF_ACCURATE_POWER_SENSOR: entity_id,
            },
        )
        device.power_use = 900.0

        time_now = datetime.datetime.now(tz=pytz.UTC)
        attr_cache: dict[str, dict[str, object]] = {}

        def _lazy_state(power: float, when: datetime.datetime) -> LazyState:
            row = SimpleNamespace(
                attributes='{"unit_of_measurement": "W"}',
                last_changed_ts=when.timestamp(),
                last_updated_ts=when.timestamp(),
            )
            return LazyState(
                row=row,
                attr_cache=attr_cache,
                start_time_ts=when.timestamp(),
                entity_id=entity_id,
                state=str(power),
                last_updated_ts=when.timestamp(),
                no_attributes=False,
            )

        states = [
            _lazy_state(1000.0, time_now - datetime.timedelta(hours=2)),
            _lazy_state(2000.0, time_now - datetime.timedelta(hours=1)),
        ]

        with patch(
            "custom_components.quiet_solar.ha_model.device.load_from_history",
            new_callable=AsyncMock,
            return_value=states,
        ):
            await device._async_bootstrap_from_history(entity_id, time_now)

        assert device.power_use == pytest.approx(1500.0, rel=0.01)


class TestQSBiStateDurationExpectedState:
    """Test expected_state_from_command methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device"}
        )

    def test_expected_state_from_command_on(self):
        """Test expected_state_from_command with CMD_ON."""
        result = self.device.expected_state_from_command(CMD_ON)
        assert result == "on"

    def test_expected_state_from_command_off(self):
        """Test expected_state_from_command with CMD_OFF."""
        result = self.device.expected_state_from_command(CMD_OFF)
        assert result == "off"

    def test_expected_state_from_command_idle(self):
        """Test expected_state_from_command with CMD_IDLE."""
        result = self.device.expected_state_from_command(CMD_IDLE)
        assert result == "off"

    def test_expected_state_from_command_none(self):
        """Test expected_state_from_command with None."""
        result = self.device.expected_state_from_command(None)
        assert result is None

    def test_expected_state_from_command_or_user_no_override(self):
        """Test expected_state_from_command_or_user without override."""
        self.device.external_user_initiated_state = None

        result = self.device.expected_state_from_command_or_user(CMD_ON)

        assert result == "on"

    def test_expected_state_from_command_or_user_with_override(self):
        """Test expected_state_from_command_or_user with user override."""
        self.device.external_user_initiated_state = "off"

        result = self.device.expected_state_from_command_or_user(CMD_ON)

        # Should return user override state, not command state
        assert result == "off"


class TestQSBiStateDurationProbeIfCommandSet:
    """Test probe_if_command_set method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            switch_entity="switch.test_device",
            **{CONF_NAME: "Test Device"}
        )
        self.device.external_user_initiated_state = None

    @pytest.mark.asyncio
    async def test_probe_command_matches(self):
        """Test probe_if_command_set when state matches expected."""
        self.hass.states.set("switch.test_device", "on")
        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.probe_if_command_set(time, CMD_ON)

        assert result is True

    @pytest.mark.asyncio
    async def test_probe_command_does_not_match(self):
        """Test probe_if_command_set when state doesn't match."""
        self.hass.states.set("switch.test_device", "off")
        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.probe_if_command_set(time, CMD_ON)

        assert result is False

    @pytest.mark.asyncio
    async def test_probe_state_unavailable(self):
        """Test probe_if_command_set when state is unavailable."""
        self.hass.states.set("switch.test_device", STATE_UNAVAILABLE)
        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.probe_if_command_set(time, CMD_ON)

        assert result is None

    @pytest.mark.asyncio
    async def test_probe_state_unknown(self):
        """Test probe_if_command_set when state is unknown."""
        self.hass.states.set("switch.test_device", STATE_UNKNOWN)
        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.probe_if_command_set(time, CMD_ON)

        assert result is None

    @pytest.mark.asyncio
    async def test_probe_with_user_override(self):
        """Test probe_if_command_set respects user override."""
        self.hass.states.set("switch.test_device", "off")
        self.device.external_user_initiated_state = "off"
        time = datetime.datetime.now(pytz.UTC)

        # Even though command is ON, user override is OFF
        result = await self.device.probe_if_command_set(time, CMD_ON)

        # Should return True because state matches user override
        assert result is True


class TestQSBiStateDurationExecuteCommand:
    """Test execute_command method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            switch_entity="switch.test_device",
            **{CONF_NAME: "Test Device"}
        )

    @pytest.mark.asyncio
    async def test_execute_command_no_override(self):
        """Test execute_command without user override."""
        self.device.external_user_initiated_state = None
        self.device.get_current_active_constraint = MagicMock(return_value=None)
        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.execute_command(time, CMD_ON)

        assert result is True
        assert len(self.device._execute_command_system_calls) == 1
        assert self.device._execute_command_system_calls[0][1] == CMD_ON

    @pytest.mark.asyncio
    async def test_execute_command_with_override_state_matches(self):
        """Test execute_command when override state matches current state."""
        self.hass.states.set("switch.test_device", "off")
        self.device.external_user_initiated_state = "off"
        self.device.get_current_active_constraint = MagicMock(return_value=None)
        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.execute_command(time, CMD_ON)

        # Should return True early because override state matches
        assert result is True
        # execute_command_system should not be called
        assert len(self.device._execute_command_system_calls) == 0

    @pytest.mark.asyncio
    async def test_execute_command_with_override_constraint_not_mandatory(self):
        """Test execute_command with non-mandatory override constraint and idle command."""
        self.hass.states.set("switch.test_device", "on")
        self.device.external_user_initiated_state = "on"

        # Create a non-mandatory constraint with load_param
        mock_constraint = MagicMock()
        mock_constraint.is_mandatory = False
        mock_constraint.load_param = "on"
        self.device.get_current_active_constraint = MagicMock(return_value=mock_constraint)

        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.execute_command(time, CMD_IDLE)

        # Should call execute_command_system with override_state=None
        assert len(self.device._execute_command_system_calls) == 1
        assert self.device._execute_command_system_calls[0][2] is None


class TestQSBiStateDurationPlatforms:
    """Test get_platforms method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_get_platforms(self):
        """Test get_platforms returns expected platforms."""
        device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device"}
        )

        platforms = device.get_platforms()

        assert Platform.SENSOR in platforms
        assert Platform.SWITCH in platforms
        assert Platform.SELECT in platforms
        assert Platform.TIME in platforms
        assert Platform.NUMBER in platforms


class TestQSBiStateDurationCheckLoadActivityModeOff:
    """Test check_load_activity_and_constraints with mode_off."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            switch_entity="switch.test_device",
            **{CONF_NAME: "Test Device"}
        )
        self.device.load_is_auto_to_be_boosted = False
        self.device._constraints = []
        self.device.command_and_constraint_reset = MagicMock()

    @pytest.mark.asyncio
    async def test_mode_off_removes_constraints(self):
        """Test that bistate_mode_off removes all constraints."""
        self.device.bistate_mode = "bistate_mode_off"
        self.device._constraints = [MagicMock()]  # Has constraints
        self.device.is_load_command_set = MagicMock(return_value=False)

        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.check_load_activity_and_constraints(time)

        assert result is True  # Should force next solve
        self.device.command_and_constraint_reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_mode_off_no_constraints(self):
        """Test bistate_mode_off with no existing constraints."""
        self.device.bistate_mode = "bistate_mode_off"
        self.device._constraints = []
        self.device.is_load_command_set = MagicMock(return_value=False)

        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.check_load_activity_and_constraints(time)

        # No force solve needed if no constraints
        self.device.command_and_constraint_reset.assert_called_once()


class TestQSBiStateDurationCheckLoadActivityModeOn:
    """Test check_load_activity_and_constraints with mode_on."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            switch_entity="switch.test_device",
            **{CONF_NAME: "Test Device"}
        )
        self.device.load_is_auto_to_be_boosted = False
        self.device._constraints = []
        self.device.power_use = 1000.0

    @pytest.mark.asyncio
    async def test_mode_on_creates_25h_constraint(self):
        """Test that bistate_mode_on creates a 25-hour constraint."""
        self.device.bistate_mode = "bistate_mode_on"
        self.device.is_load_command_set = MagicMock(return_value=False)
        self.device.get_proper_local_adapted_tomorrow = MagicMock(
            return_value=datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=24)
        )
        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.check_load_activity_and_constraints(time)

        assert result is True

        constraint = self.device.get_current_active_constraint(time)

        assert constraint is not None

        # Check the constraint that was created

        assert constraint.target_value == 25 * 3600.0  # 25 hours
        assert constraint.from_user is True


class TestQSBiStateDurationCheckLoadActivityModeDefault:
    """Test check_load_activity_and_constraints with mode_default."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            switch_entity="switch.test_device",
            **{CONF_NAME: "Test Device"}
        )
        self.device.load_is_auto_to_be_boosted = False
        self.device._constraints = []
        self.device.power_use = 1000.0
        self.device.default_on_duration = 2.0
        self.device.default_on_finish_time = dt_time(hour=7, minute=0, second=0)

    @pytest.mark.asyncio
    async def test_mode_default_with_duration_and_finish_time(self):
        """Test bistate_mode_default creates constraint with duration/finish time."""
        self.device.bistate_mode = "bistate_mode_default"
        self.device.is_load_command_set = MagicMock(return_value=False)
        self.device.get_next_time_from_hours = MagicMock(
            return_value=datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=8)
        )

        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.check_load_activity_and_constraints(time)

        constraint = self.device.get_current_active_constraint(time)
        assert constraint is not None
        assert result

        # Check constraint was created
        assert constraint.target_value == 2.0 * 3600.0  # 2 hours in seconds

    @pytest.mark.asyncio
    async def test_mode_default_no_duration(self):
        """Test bistate_mode_default without duration does nothing."""
        self.device.bistate_mode = "bistate_mode_default"
        self.device.default_on_duration = None
        self.device.is_load_command_set = MagicMock(return_value=False)
        self.device.push_agenda_constraints = MagicMock(return_value=False)

        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.check_load_activity_and_constraints(time)

        # Should not create constraint
        self.device.push_agenda_constraints.assert_not_called()


class TestQSBiStateDurationCheckLoadActivityModeAuto:
    """Test check_load_activity_and_constraints with mode_auto."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            switch_entity="switch.test_device",
            **{CONF_NAME: "Test Device"}
        )
        self.device.load_is_auto_to_be_boosted = False
        self.device._constraints = []
        self.device.power_use = 1000.0

    @pytest.mark.asyncio
    async def test_mode_auto_with_scheduled_event(self):
        """Test bistate_mode_auto creates constraint from scheduled event."""
        self.device.bistate_mode = "bistate_mode_auto"
        self.device.is_load_command_set = MagicMock(return_value=False)

        start = datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=1)
        end = datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=3)
        self.device.get_next_scheduled_events = AsyncMock(return_value=[(start, end)])

        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.check_load_activity_and_constraints(time)

        constraint = self.device._constraints[0] if self.device._constraints else None
        assert constraint is not None
        assert result

        # Target value should be duration in seconds
        expected_duration = (end - start).total_seconds()
        assert constraint.target_value == expected_duration
        # start_schedule should be DATETIME_MIN_UTC for auto mode (None gets converted to this)
        # This is because the constraint class converts None to DATETIME_MIN_UTC
        DATETIME_MIN_UTC = datetime.datetime.min.replace(tzinfo=pytz.UTC)
        assert constraint.start_of_constraint == DATETIME_MIN_UTC

    @pytest.mark.asyncio
    async def test_mode_auto_no_scheduled_event(self):
        """Test bistate_mode_auto with no scheduled event."""
        self.device.bistate_mode = "bistate_mode_auto"
        self.device.is_load_command_set = MagicMock(return_value=False)
        self.device.get_next_scheduled_event = AsyncMock(return_value=(None, None))
        self.device.push_agenda_constraints = MagicMock(return_value=False)

        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.check_load_activity_and_constraints(time)

        # Should not create constraint
        self.device.push_agenda_constraints.assert_not_called()


class TestQSBiStateDurationCheckLoadActivityModeExactCalendar:
    """Test check_load_activity_and_constraints with mode_exact_calendar."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            switch_entity="switch.test_device",
            **{CONF_NAME: "Test Device"}
        )
        self.device.load_is_auto_to_be_boosted = False
        self.device._constraints = []
        self.device.power_use = 1000.0

    @pytest.mark.asyncio
    async def test_mode_exact_calendar_with_scheduled_event(self):
        """Test bistate_mode_exact_calendar creates exact constraint."""
        self.device.bistate_mode = "bistate_mode_exact_calendar"
        self.device.is_load_command_set = MagicMock(return_value=False)

        start = datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=1)
        end = datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=3)
        self.device.get_next_scheduled_events = AsyncMock(return_value=[(start, end)])


        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.check_load_activity_and_constraints(time)

        constraint = self.device._constraints[0] if self.device._constraints else None
        assert constraint is not None
        assert result

        # Target value should include 10 min leg room
        expected_duration = (end - start).total_seconds()
        assert constraint.target_value >= expected_duration
        assert constraint.target_value <= expected_duration + 2*SOLVER_STEP_S + 2

        # always_end_at_end_of_constraint should be True
        assert constraint.always_end_at_end_of_constraint is True


class TestQSBiStateDurationCheckLoadActivityUserOverride:
    """Test check_load_activity_and_constraints with user override scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            switch_entity="switch.test_device",
            **{CONF_NAME: "Test Device"}
        )
        self.device.load_is_auto_to_be_boosted = False
        self.device._constraints = []
        self.device.power_use = 1000.0
        self.device.current_command = None
        self.device.running_command = None

    @pytest.mark.asyncio
    async def test_user_override_detected(self):
        """Test that user override is detected when state differs from expected."""
        self.device.bistate_mode = "bistate_mode_auto"
        self.device.is_load_command_set = MagicMock(return_value=True)
        self.device.external_user_initiated_state = None
        self.device.external_user_initiated_state_time = None
        self.device.asked_for_reset_user_initiated_state_time = None
        self.device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        # Set current command to OFF but actual state is ON
        self.device.current_command = CMD_OFF
        self.device.running_command = CMD_OFF
        self.hass.states.set("switch.test_device", "on")

        self.device.set_live_constraints = MagicMock()
        self.device.push_live_constraint = MagicMock(return_value=True)
        self.device.command_and_constraint_reset = MagicMock()
        self.device.get_next_scheduled_event = AsyncMock(return_value=(None, None))


        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.check_load_activity_and_constraints(time)

        # User override should be detected
        assert self.device.external_user_initiated_state == "on"
        assert self.device.external_user_initiated_state_time == time

    @pytest.mark.asyncio
    async def test_user_override_idle_resets_constraints(self):
        """Test that user override to idle state resets constraints."""
        self.device.bistate_mode = "bistate_mode_auto"
        self.device.is_load_command_set = MagicMock(return_value=True)
        self.device.external_user_initiated_state = None
        self.device.external_user_initiated_state_time = None
        self.device.asked_for_reset_user_initiated_state_time = None
        self.device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        # Set current command to ON but actual state is OFF (idle)
        self.device.current_command = CMD_ON
        self.device.running_command = CMD_ON
        self.hass.states.set("switch.test_device", "off")  # OFF = idle for bistate

        self.device.set_live_constraints = MagicMock()
        self.device.push_live_constraint = MagicMock(return_value=True)
        self.device.command_and_constraint_reset = MagicMock()
        self.device.get_next_scheduled_event = AsyncMock(return_value=(None, None))


        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.check_load_activity_and_constraints(time)

        # Should reset because user set to idle state
        self.device.command_and_constraint_reset.assert_called_once()


class TestQSBiStateDurationAbstractMethods:
    """Test abstract methods are properly defined."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_get_virtual_current_constraint_translation_key(self):
        """Test get_virtual_current_constraint_translation_key returns value."""
        device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device"}
        )

        assert device.get_virtual_current_constraint_translation_key() == "test_constraint_key"

    def test_get_select_translation_key(self):
        """Test get_select_translation_key returns value."""
        device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device"}
        )

        assert device.get_select_translation_key() == "test_select_key"


class TestQSBiStateDurationBestEffortLoad:
    """Test best effort load behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_bistate_entry",
            data={CONF_NAME: "Test BiState"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = ConcreteBiStateDevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            switch_entity="switch.test_device",
            **{CONF_NAME: "Test Device"}
        )
        self.device.load_is_auto_to_be_boosted = False
        self.device.qs_best_effort_green_only = False
        self.device._constraints = []
        self.device.power_use = 1000.0

    @pytest.mark.asyncio
    async def test_best_effort_load_uses_filler_constraint_type(self):
        """Test that best effort load uses FILLER_AUTO constraint type."""
        self.device.bistate_mode = "bistate_mode_auto"
        self.device.is_load_command_set = MagicMock(return_value=False)
        self.device.is_best_effort_only_load = MagicMock(return_value=True)

        start = datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=1)
        end = datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=3)
        self.device.get_next_scheduled_events = AsyncMock(return_value=[(start, end)])

        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.check_load_activity_and_constraints(time)

        constraint = self.device._constraints[0] if self.device._constraints else None
        assert constraint is not None
        assert result

        # Best effort loads should use FILLER_AUTO type
        assert constraint.type == CONSTRAINT_TYPE_FILLER_AUTO

    @pytest.mark.asyncio
    async def test_non_best_effort_load_uses_mandatory_constraint_type(self):
        """Test that non-best effort load uses MANDATORY constraint type."""
        self.device.bistate_mode = "bistate_mode_auto"
        self.device.is_load_command_set = MagicMock(return_value=False)
        # Set the underlying properties that control is_best_effort_only_load() result
        # is_best_effort_only_load returns: self.load_is_auto_to_be_boosted or self.qs_best_effort_green_only
        self.device.load_is_auto_to_be_boosted = False
        self.device.qs_best_effort_green_only = False
        # The constraint.type property returns _degraded_type if is_off_grid() is True
        # is_off_grid() calls self.home.is_off_grid(), so we need to mock it
        self.home.is_off_grid = MagicMock(return_value=False)

        start = datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=1)
        end = datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=3)
        self.device.get_next_scheduled_events = AsyncMock(return_value=[(start, end)])

        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.check_load_activity_and_constraints(time)

        constraint = self.device._constraints[0] if self.device._constraints else None
        assert constraint is not None
        assert result

        # Non-best effort loads should use MANDATORY type
        assert constraint.type == CONSTRAINT_TYPE_MANDATORY_END_TIME
