"""Tests for QSOnOffDuration class in ha_model/on_off_duration.py."""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import time as dt_time

from homeassistant.const import (
    Platform,
    STATE_UNKNOWN,
    STATE_UNAVAILABLE,
    CONF_NAME,
    SERVICE_TURN_ON,
    SERVICE_TURN_OFF,
)

from custom_components.quiet_solar.ha_model.on_off_duration import QSOnOffDuration
from custom_components.quiet_solar.home_model.commands import CMD_ON, CMD_OFF, CMD_IDLE
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    SENSOR_CONSTRAINT_SENSOR_ON_OFF,
    CONF_SWITCH,
)

from tests.test_helpers import FakeHass, FakeConfigEntry
import pytz


class TestQSOnOffDurationInit:
    """Test QSOnOffDuration initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_on_off_entry",
            data={CONF_NAME: "Test OnOff"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_init_default_values(self):
        """Test initialization with default values."""
        device = QSOnOffDuration(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device", CONF_SWITCH: "switch.test_device"}
        )

        assert device._state_on == "on"
        assert device._state_off == "off"
        assert device._bistate_mode_on == "on_off_mode_on"
        assert device._bistate_mode_off == "on_off_mode_off"

    def test_bistate_entity_equals_switch_entity(self):
        """Test that bistate_entity is set to switch_entity."""
        device = QSOnOffDuration(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device", CONF_SWITCH: "switch.my_switch"}
        )

        assert device.bistate_entity == "switch.my_switch"
        assert device.switch_entity == "switch.my_switch"


class TestQSOnOffDurationTranslationKeys:
    """Test translation key methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_on_off_entry",
            data={CONF_NAME: "Test OnOff"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSOnOffDuration(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device", CONF_SWITCH: "switch.test_device"}
        )

    def test_get_virtual_current_constraint_translation_key(self):
        """Test get_virtual_current_constraint_translation_key returns correct key."""
        result = self.device.get_virtual_current_constraint_translation_key()
        assert result == SENSOR_CONSTRAINT_SENSOR_ON_OFF

    def test_get_select_translation_key(self):
        """Test get_select_translation_key returns correct key."""
        result = self.device.get_select_translation_key()
        assert result == "on_off_mode"


class TestQSOnOffDurationExecuteCommandSystem:
    """Test execute_command_system method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_on_off_entry",
            data={CONF_NAME: "Test OnOff"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSOnOffDuration(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device", CONF_SWITCH: "switch.test_device"}
        )

    @pytest.mark.asyncio
    async def test_execute_command_turn_on(self):
        """Test execute_command_system with CMD_ON turns on switch."""
        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.execute_command_system(time, CMD_ON, state=None)

        assert result is False  # Method returns False
        # Verify service was called
        calls = self.hass.services.calls
        assert len(calls) >= 1
        # Find the switch call
        switch_calls = [c for c in calls if c[0] == Platform.SWITCH and c[1] == SERVICE_TURN_ON]
        assert len(switch_calls) == 1

    @pytest.mark.asyncio
    async def test_execute_command_turn_off(self):
        """Test execute_command_system with CMD_OFF turns off switch."""
        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.execute_command_system(time, CMD_OFF, state=None)

        assert result is False
        calls = self.hass.services.calls
        switch_calls = [c for c in calls if c[0] == Platform.SWITCH and c[1] == SERVICE_TURN_OFF]
        assert len(switch_calls) == 1

    @pytest.mark.asyncio
    async def test_execute_command_idle(self):
        """Test execute_command_system with CMD_IDLE turns off switch."""
        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.execute_command_system(time, CMD_IDLE, state=None)

        assert result is False
        calls = self.hass.services.calls
        switch_calls = [c for c in calls if c[0] == Platform.SWITCH and c[1] == SERVICE_TURN_OFF]
        assert len(switch_calls) == 1

    @pytest.mark.asyncio
    async def test_execute_command_with_override_state_on(self):
        """Test execute_command_system with override state ON."""
        time = datetime.datetime.now(pytz.UTC)

        # state="on" means turn on regardless of command
        result = await self.device.execute_command_system(time, CMD_OFF, state="on")

        assert result is False
        calls = self.hass.services.calls
        switch_calls = [c for c in calls if c[0] == Platform.SWITCH and c[1] == SERVICE_TURN_ON]
        assert len(switch_calls) == 1

    @pytest.mark.asyncio
    async def test_execute_command_with_override_state_off(self):
        """Test execute_command_system with override state OFF (idle)."""
        time = datetime.datetime.now(pytz.UTC)

        # state="off" (which equals expected_state_from_command(CMD_IDLE)) means turn off
        result = await self.device.execute_command_system(time, CMD_ON, state="off")

        assert result is False
        calls = self.hass.services.calls
        switch_calls = [c for c in calls if c[0] == Platform.SWITCH and c[1] == SERVICE_TURN_OFF]
        assert len(switch_calls) == 1

    @pytest.mark.asyncio
    async def test_execute_command_invalid_raises(self):
        """Test execute_command_system with invalid command raises ValueError."""
        time = datetime.datetime.now(pytz.UTC)

        # Create an invalid command that is not ON, OFF, or IDLE
        from custom_components.quiet_solar.home_model.commands import LoadCommand
        invalid_cmd = LoadCommand(command="invalid", power_consign=0)

        with pytest.raises(ValueError, match="Invalid command"):
            await self.device.execute_command_system(time, invalid_cmd, state=None)


class TestQSOnOffDurationStateValues:
    """Test state value configurations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_on_off_entry",
            data={CONF_NAME: "Test OnOff"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSOnOffDuration(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Device", CONF_SWITCH: "switch.test_device"}
        )

    def test_expected_state_from_command_on(self):
        """Test expected_state_from_command returns 'on' for CMD_ON."""
        result = self.device.expected_state_from_command(CMD_ON)
        assert result == "on"

    def test_expected_state_from_command_off(self):
        """Test expected_state_from_command returns 'off' for CMD_OFF."""
        result = self.device.expected_state_from_command(CMD_OFF)
        assert result == "off"

    def test_expected_state_from_command_idle(self):
        """Test expected_state_from_command returns 'off' for CMD_IDLE."""
        result = self.device.expected_state_from_command(CMD_IDLE)
        assert result == "off"

    def test_expected_state_from_command_none(self):
        """Test expected_state_from_command returns None for None command."""
        result = self.device.expected_state_from_command(None)
        assert result is None
