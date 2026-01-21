"""Tests for quiet_solar ha_model/heat_pump.py."""

import pytest
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import DOMAIN
from custom_components.quiet_solar.ha_model.heat_pump import QSHeatPump


async def test_heat_pump_get_platforms(hass: HomeAssistant) -> None:
    """Test heat pump platform list includes binary sensor."""
    hass.data.setdefault(DOMAIN, {})
    config_entry = MockConfigEntry(domain=DOMAIN, data={}, entry_id="heat_pump_entry")

    heat_pump = QSHeatPump(hass, config_entry, name="Test Heat Pump")
    platforms = set(heat_pump.get_platforms())

    assert Platform.BINARY_SENSOR in platforms
    assert Platform.SENSOR in platforms
    assert Platform.BUTTON in platforms
