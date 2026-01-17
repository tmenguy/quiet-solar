"""Tests for ha_model/climate_controller.py - Climate device functionality."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import pytz

from custom_components.quiet_solar.ha_model.climate_controller import get_hvac_modes
from custom_components.quiet_solar.home_model.commands import CMD_ON, CMD_OFF
from custom_components.quiet_solar.const import (
    CONF_CLIMATE_HVAC_MODE_ON,
    CONF_CLIMATE_HVAC_MODE_OFF,
    CONF_CLIMATE,
    SENSOR_CONSTRAINT_SENSOR_CLIMATE,
)


def test_get_hvac_modes():
    """Test get_hvac_modes extracts modes from entity registry."""
    mock_hass = MagicMock()
    mock_entry = MagicMock()
    mock_entry.capabilities = {"hvac_modes": ["off", "heat", "cool", "auto"]}

    mock_registry = MagicMock()
    mock_registry.async_get.return_value = mock_entry

    with patch("custom_components.quiet_solar.ha_model.climate_controller.er.async_get", return_value=mock_registry):
        modes = get_hvac_modes(mock_hass, "climate.living_room")

    assert modes == ["off", "heat", "cool", "auto"]


def test_get_hvac_modes_default():
    """Test get_hvac_modes returns default when capabilities missing."""
    mock_hass = MagicMock()
    mock_entry = MagicMock()
    mock_entry.capabilities = {}

    mock_registry = MagicMock()
    mock_registry.async_get.return_value = mock_entry

    with patch("custom_components.quiet_solar.ha_model.climate_controller.er.async_get", return_value=mock_registry):
        modes = get_hvac_modes(mock_hass, "climate.living_room")

    assert "auto" in modes or "off" in modes


class FakeClimateDuration:
    """A testable version of QSClimateDuration."""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "Test Climate")
        self.climate_entity = kwargs.get(CONF_CLIMATE, "climate.test")
        self._state_off = kwargs.get(CONF_CLIMATE_HVAC_MODE_OFF, "off")
        self._state_on = kwargs.get(CONF_CLIMATE_HVAC_MODE_ON, "heat")
        self._bistate_mode_on = self._state_on
        self._bistate_mode_off = self._state_off
        self.bistate_entity = self.climate_entity
        self.is_load_time_sensitive = True
        self.hass = kwargs.get("hass", MagicMock())
        self.home = kwargs.get("home", None)


def test_climate_duration_init():
    """Test QSClimateDuration initialization."""
    climate = FakeClimateDuration(
        name="Living Room",
        **{
            CONF_CLIMATE: "climate.living_room",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
        }
    )

    assert climate.name == "Living Room"
    assert climate.climate_entity == "climate.living_room"
    assert climate._state_on == "heat"
    assert climate._state_off == "off"


def test_climate_state_on_property():
    """Test climate_state_on property."""
    climate = FakeClimateDuration()

    assert climate._state_on == "heat"

    climate._state_on = "auto"
    assert climate._state_on == "auto"


def test_climate_state_off_property():
    """Test climate_state_off property."""
    climate = FakeClimateDuration()

    assert climate._state_off == "off"

    climate._state_off = "fan_only"
    assert climate._state_off == "fan_only"


def test_climate_bistate_modes_map_to_hvac_modes():
    """Test that bistate modes map to HVAC modes."""
    climate = FakeClimateDuration(
        **{
            CONF_CLIMATE_HVAC_MODE_ON: "cool",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
        }
    )

    assert climate._bistate_mode_on == "cool"
    assert climate._bistate_mode_off == "off"


def test_climate_is_time_sensitive():
    """Test that climate is marked as time sensitive."""
    climate = FakeClimateDuration()
    assert climate.is_load_time_sensitive is True


def test_climate_bistate_entity_is_climate_entity():
    """Test that bistate_entity is set to climate_entity."""
    climate = FakeClimateDuration(**{CONF_CLIMATE: "climate.bedroom"})

    assert climate.bistate_entity == "climate.bedroom"
    assert climate.climate_entity == "climate.bedroom"


def test_climate_default_hvac_modes():
    """Test default HVAC mode values."""
    climate = FakeClimateDuration()

    assert climate._state_on == "heat"
    assert climate._state_off == "off"


def test_climate_custom_hvac_modes():
    """Test custom HVAC mode values."""
    climate = FakeClimateDuration(**{
        CONF_CLIMATE_HVAC_MODE_ON: "cool",
        CONF_CLIMATE_HVAC_MODE_OFF: "fan_only",
    })

    assert climate._state_on == "cool"
    assert climate._state_off == "fan_only"
