"""Tests for QSOnOffDuration class in ha_model/on_off_duration.py."""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock, patch
from datetime import time as dt_time

from homeassistant.const import (
    Platform,
    STATE_UNKNOWN,
    STATE_UNAVAILABLE,
    CONF_NAME,
    SERVICE_TURN_ON,
    SERVICE_TURN_OFF,
)
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.ha_model.on_off_duration import QSOnOffDuration
from custom_components.quiet_solar.home_model.commands import CMD_ON, CMD_OFF, CMD_IDLE
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    SENSOR_CONSTRAINT_SENSOR_ON_OFF,
    CONF_SWITCH,
)

import pytz


@pytest.fixture
def on_off_config_entry() -> MockConfigEntry:
    """Config entry for on/off duration tests."""
    return MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_on_off_entry",
        data={CONF_NAME: "Test OnOff"},
        title="Test OnOff",
    )


@pytest.fixture
def on_off_home():
    """Mock home for on/off duration tests."""
    return MagicMock()


@pytest.fixture
def on_off_data_handler(on_off_home):
    """Data handler for on/off duration tests."""
    handler = MagicMock()
    handler.home = on_off_home
    return handler


@pytest.fixture
def on_off_hass_data(hass: HomeAssistant, on_off_data_handler):
    """Set hass.data[DOMAIN][DATA_HANDLER] for on/off tests."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][DATA_HANDLER] = on_off_data_handler


@pytest.fixture
def on_off_device(hass, on_off_config_entry, on_off_home, on_off_data_handler, on_off_hass_data):
    """QSOnOffDuration instance for tests."""
    return QSOnOffDuration(
        hass=hass,
        config_entry=on_off_config_entry,
        home=on_off_home,
        **{CONF_NAME: "Test Device", CONF_SWITCH: "switch.test_device"}
    )


@pytest.fixture
def recorded_service_calls(hass: HomeAssistant):
    """Record service calls (domain, service, service_data) for assertions."""
    from homeassistant.core import ServiceRegistry

    recorded = []

    async def record_only(self, domain, service, service_data=None, **kwargs):
        recorded.append((domain, service, service_data or {}))

    with patch.object(ServiceRegistry, "async_call", record_only):
        yield recorded


class TestQSOnOffDurationInit:
    """Test QSOnOffDuration initialization."""

    def test_init_default_values(
        self, hass, on_off_config_entry, on_off_home, on_off_data_handler, on_off_hass_data
    ):
        """Test initialization with default values."""
        device = QSOnOffDuration(
            hass=hass,
            config_entry=on_off_config_entry,
            home=on_off_home,
            **{CONF_NAME: "Test Device", CONF_SWITCH: "switch.test_device"}
        )

        assert device._state_on == "on"
        assert device._state_off == "off"
        assert device._bistate_mode_on == "on_off_mode_on"
        assert device._bistate_mode_off == "on_off_mode_off"

    def test_bistate_entity_equals_switch_entity(
        self, hass, on_off_config_entry, on_off_home, on_off_data_handler, on_off_hass_data
    ):
        """Test that bistate_entity is set to switch_entity."""
        device = QSOnOffDuration(
            hass=hass,
            config_entry=on_off_config_entry,
            home=on_off_home,
            **{CONF_NAME: "Test Device", CONF_SWITCH: "switch.my_switch"}
        )

        assert device.bistate_entity == "switch.my_switch"
        assert device.switch_entity == "switch.my_switch"


class TestQSOnOffDurationTranslationKeys:
    """Test translation key methods."""

    def test_get_virtual_current_constraint_translation_key(self, on_off_device):
        """Test get_virtual_current_constraint_translation_key returns correct key."""
        result = on_off_device.get_virtual_current_constraint_translation_key()
        assert result == SENSOR_CONSTRAINT_SENSOR_ON_OFF

    def test_get_select_translation_key(self, on_off_device):
        """Test get_select_translation_key returns correct key."""
        result = on_off_device.get_select_translation_key()
        assert result == "on_off_mode"


class TestQSOnOffDurationExecuteCommandSystem:
    """Test execute_command_system method."""

    @pytest.mark.asyncio
    async def test_execute_command_turn_on(self, on_off_device, recorded_service_calls):
        """Test execute_command_system with CMD_ON turns on switch."""
        time = datetime.datetime.now(pytz.UTC)

        result = await on_off_device.execute_command_system(time, CMD_ON, state=None)

        assert result is False  # Method returns False
        switch_calls = [
            c for c in recorded_service_calls
            if c[0] == Platform.SWITCH and c[1] == SERVICE_TURN_ON
        ]
        assert len(switch_calls) == 1

    @pytest.mark.asyncio
    async def test_execute_command_turn_off(self, on_off_device, recorded_service_calls):
        """Test execute_command_system with CMD_OFF turns off switch."""
        time = datetime.datetime.now(pytz.UTC)

        result = await on_off_device.execute_command_system(time, CMD_OFF, state=None)

        assert result is False
        switch_calls = [
            c for c in recorded_service_calls
            if c[0] == Platform.SWITCH and c[1] == SERVICE_TURN_OFF
        ]
        assert len(switch_calls) == 1

    @pytest.mark.asyncio
    async def test_execute_command_idle(self, on_off_device, recorded_service_calls):
        """Test execute_command_system with CMD_IDLE turns off switch."""
        time = datetime.datetime.now(pytz.UTC)

        result = await on_off_device.execute_command_system(time, CMD_IDLE, state=None)

        assert result is False
        switch_calls = [
            c for c in recorded_service_calls
            if c[0] == Platform.SWITCH and c[1] == SERVICE_TURN_OFF
        ]
        assert len(switch_calls) == 1

    @pytest.mark.asyncio
    async def test_execute_command_with_override_state_on(
        self, on_off_device, recorded_service_calls
    ):
        """Test execute_command_system with override state ON."""
        time = datetime.datetime.now(pytz.UTC)

        result = await on_off_device.execute_command_system(time, CMD_OFF, state="on")

        assert result is False
        switch_calls = [
            c for c in recorded_service_calls
            if c[0] == Platform.SWITCH and c[1] == SERVICE_TURN_ON
        ]
        assert len(switch_calls) == 1

    @pytest.mark.asyncio
    async def test_execute_command_with_override_state_off(
        self, on_off_device, recorded_service_calls
    ):
        """Test execute_command_system with override state OFF (idle)."""
        time = datetime.datetime.now(pytz.UTC)

        result = await on_off_device.execute_command_system(time, CMD_ON, state="off")

        assert result is False
        switch_calls = [
            c for c in recorded_service_calls
            if c[0] == Platform.SWITCH and c[1] == SERVICE_TURN_OFF
        ]
        assert len(switch_calls) == 1

    @pytest.mark.asyncio
    async def test_execute_command_invalid_raises(self, on_off_device):
        """Test execute_command_system with invalid command raises ValueError."""
        time = datetime.datetime.now(pytz.UTC)

        from custom_components.quiet_solar.home_model.commands import LoadCommand
        invalid_cmd = LoadCommand(command="invalid", power_consign=0)

        with pytest.raises(ValueError, match="Invalid command"):
            await on_off_device.execute_command_system(time, invalid_cmd, state=None)


class TestQSOnOffDurationStateValues:
    """Test state value configurations."""

    def test_expected_state_from_command_on(self, on_off_device):
        """Test expected_state_from_command returns 'on' for CMD_ON."""
        result = on_off_device.expected_state_from_command(CMD_ON)
        assert result == "on"

    def test_expected_state_from_command_off(self, on_off_device):
        """Test expected_state_from_command returns 'off' for CMD_OFF."""
        result = on_off_device.expected_state_from_command(CMD_OFF)
        assert result == "off"

    def test_expected_state_from_command_idle(self, on_off_device):
        """Test expected_state_from_command returns 'off' for CMD_IDLE."""
        result = on_off_device.expected_state_from_command(CMD_IDLE)
        assert result == "off"

    def test_expected_state_from_command_none(self, on_off_device):
        """Test expected_state_from_command returns None for None command."""
        result = on_off_device.expected_state_from_command(None)
        assert result is None
