"""Extended tests for QSClimateDuration in ha_model/climate_controller.py."""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from homeassistant.const import CONF_NAME
from homeassistant.components import climate
from homeassistant.components.climate import HVACMode
import pytz

from custom_components.quiet_solar.ha_model.climate_controller import QSClimateDuration, get_hvac_modes
from custom_components.quiet_solar.home_model.commands import CMD_ON, CMD_OFF, CMD_IDLE
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_CLIMATE,
    CONF_CLIMATE_HVAC_MODE_ON,
    CONF_CLIMATE_HVAC_MODE_OFF,
    SENSOR_CONSTRAINT_SENSOR_CLIMATE,
    CONF_SWITCH,
)

from tests.conftest import FakeHass, FakeConfigEntry


class TestQSClimateDurationInit:
    """Test QSClimateDuration initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_climate_entry",
            data={CONF_NAME: "Test Climate"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_init_with_default_hvac_modes(self):
        """Test initialization with default HVAC modes."""
        device = QSClimateDuration(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Living Room Climate",
                CONF_SWITCH: "switch.climate_helper",
                CONF_CLIMATE: "climate.living_room",
            }
        )

        assert device.name == "Living Room Climate"
        assert device.climate_entity == "climate.living_room"
        assert device._state_off == str(HVACMode.OFF.value)
        assert device._state_on == str(HVACMode.AUTO.value)
        assert device.is_load_time_sensitive is True

    def test_init_with_custom_hvac_modes(self):
        """Test initialization with custom HVAC modes."""
        device = QSClimateDuration(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Bedroom Climate",
                CONF_SWITCH: "switch.climate_helper",
                CONF_CLIMATE: "climate.bedroom",
                CONF_CLIMATE_HVAC_MODE_ON: "heat",
                CONF_CLIMATE_HVAC_MODE_OFF: "fan_only",
            }
        )

        assert device._state_on == "heat"
        assert device._state_off == "fan_only"
        assert device._bistate_mode_on == "heat"
        assert device._bistate_mode_off == "fan_only"

    def test_init_bistate_entity_equals_climate_entity(self):
        """Test that bistate_entity is set to climate_entity."""
        device = QSClimateDuration(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Climate",
                CONF_SWITCH: "switch.climate_helper",
                CONF_CLIMATE: "climate.test",
            }
        )

        assert device.bistate_entity == "climate.test"


class TestQSClimateDurationProperties:
    """Test climate state properties."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_climate_entry",
            data={CONF_NAME: "Test Climate"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSClimateDuration(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Climate",
                CONF_SWITCH: "switch.climate_helper",
                CONF_CLIMATE: "climate.test",
            }
        )

    def test_climate_state_on_getter(self):
        """Test climate_state_on getter."""
        assert self.device.climate_state_on == str(HVACMode.AUTO.value)

    def test_climate_state_on_setter(self):
        """Test climate_state_on setter."""
        self.device.climate_state_on = "heat"
        assert self.device.climate_state_on == "heat"
        assert self.device._state_on == "heat"

    def test_climate_state_off_getter(self):
        """Test climate_state_off getter."""
        assert self.device.climate_state_off == str(HVACMode.OFF.value)

    def test_climate_state_off_setter(self):
        """Test climate_state_off setter."""
        self.device.climate_state_off = "fan_only"
        assert self.device.climate_state_off == "fan_only"
        assert self.device._state_off == "fan_only"


class TestQSClimateDurationTranslationKeys:
    """Test translation key methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_climate_entry",
            data={CONF_NAME: "Test Climate"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSClimateDuration(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Climate",
                CONF_SWITCH: "switch.climate_helper",
                CONF_CLIMATE: "climate.test",
            }
        )

    def test_get_virtual_current_constraint_translation_key(self):
        """Test get_virtual_current_constraint_translation_key returns correct key."""
        result = self.device.get_virtual_current_constraint_translation_key()
        assert result == SENSOR_CONSTRAINT_SENSOR_CLIMATE

    def test_get_select_translation_key(self):
        """Test get_select_translation_key returns correct key."""
        result = self.device.get_select_translation_key()
        assert result == "climate_mode"


class TestQSClimateDurationExecuteCommand:
    """Test execute_command_system method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_climate_entry",
            data={CONF_NAME: "Test Climate"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSClimateDuration(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Climate",
                CONF_SWITCH: "switch.climate_helper",
                CONF_CLIMATE: "climate.test",
                CONF_CLIMATE_HVAC_MODE_ON: "heat",
                CONF_CLIMATE_HVAC_MODE_OFF: "off",
            }
        )

    @pytest.mark.asyncio
    async def test_execute_command_turn_on(self):
        """Test execute_command_system with CMD_ON sets HVAC to heat."""
        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.execute_command_system(time, CMD_ON, state=None)

        assert result is False
        # Verify climate service was called
        calls = self.hass.services.calls
        climate_calls = [c for c in calls if c[0] == climate.DOMAIN]
        assert len(climate_calls) >= 1
        assert climate_calls[0][1] == climate.SERVICE_SET_HVAC_MODE

    @pytest.mark.asyncio
    async def test_execute_command_turn_off(self):
        """Test execute_command_system with CMD_OFF sets HVAC to off."""
        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.execute_command_system(time, CMD_OFF, state=None)

        assert result is False
        calls = self.hass.services.calls
        climate_calls = [c for c in calls if c[0] == climate.DOMAIN]
        assert len(climate_calls) >= 1

    @pytest.mark.asyncio
    async def test_execute_command_idle(self):
        """Test execute_command_system with CMD_IDLE sets HVAC to off."""
        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.execute_command_system(time, CMD_IDLE, state=None)

        assert result is False
        calls = self.hass.services.calls
        climate_calls = [c for c in calls if c[0] == climate.DOMAIN]
        assert len(climate_calls) >= 1

    @pytest.mark.asyncio
    async def test_execute_command_with_override_state(self):
        """Test execute_command_system with explicit state override."""
        time = datetime.datetime.now(pytz.UTC)

        result = await self.device.execute_command_system(time, CMD_ON, state="cool")

        assert result is False
        calls = self.hass.services.calls
        # Should have set HVAC mode to "cool" regardless of command
        climate_calls = [c for c in calls if c[0] == climate.DOMAIN]
        assert len(climate_calls) >= 1
        # Check that HVAC mode was set to "cool"
        last_call = climate_calls[-1]
        assert last_call[2].get(climate.ATTR_HVAC_MODE) == "cool"

    @pytest.mark.asyncio
    async def test_execute_command_invalid_raises(self):
        """Test execute_command_system with invalid command raises ValueError."""
        time = datetime.datetime.now(pytz.UTC)

        from custom_components.quiet_solar.home_model.commands import LoadCommand
        invalid_cmd = LoadCommand(command="invalid", power_consign=0)

        with pytest.raises(ValueError, match="Invalid command"):
            await self.device.execute_command_system(time, invalid_cmd, state=None)


class TestQSClimateDurationModes:
    """Test get_possibles_modes method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_climate_entry",
            data={CONF_NAME: "Test Climate"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSClimateDuration(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Climate",
                CONF_SWITCH: "switch.climate_helper",
                CONF_CLIMATE: "climate.test",
            }
        )

    def test_get_possibles_modes(self):
        """Test get_possibles_modes returns modes from registry."""
        mock_entry = MagicMock()
        mock_entry.capabilities = {"hvac_modes": ["off", "heat", "cool", "auto"]}

        mock_registry = MagicMock()
        mock_registry.async_get.return_value = mock_entry

        with patch("custom_components.quiet_solar.ha_model.climate_controller.er.async_get", return_value=mock_registry):
            modes = self.device.get_possibles_modes()

        assert modes == ["off", "heat", "cool", "auto"]
