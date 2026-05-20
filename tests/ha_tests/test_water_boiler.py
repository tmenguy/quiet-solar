"""Tests for quiet_solar ha_model/water_boiler.py.

QS-194: new `water_boiler` (cumulus) load type — a thin subclass of
`QSOnOffDuration` with a dedicated config step, dashboard section,
select-mode translation key, and an optional water-tank temperature
sensor (plumbing only — no constraint/solver logic acts on it).
"""

from __future__ import annotations

import pytest
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import DOMAIN
from custom_components.quiet_solar.ha_model.on_off_duration import QSOnOffDuration
from custom_components.quiet_solar.ha_model.water_boiler import QSWaterBoiler

from .const import (
    MOCK_WATER_BOILER_CONFIG,
    MOCK_WATER_BOILER_CONFIG_NO_TEMP,
)

pytestmark = pytest.mark.usefixtures("mock_sensor_states")


async def test_water_boiler_device_type_registered(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """A water_boiler config entry creates a QSWaterBoiler reachable via the home."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_WATER_BOILER_CONFIG,
        entry_id="water_boiler_registered_test",
        title=f"water_boiler: {MOCK_WATER_BOILER_CONFIG['name']}",
        unique_id="quiet_solar_water_boiler_registered_test",
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED

    device = hass.data[DOMAIN].get(entry.entry_id)
    assert device is not None
    assert isinstance(device, QSWaterBoiler)
    # QSWaterBoiler is a subclass of QSOnOffDuration — isinstance check
    # preserved for existing call-sites.
    assert isinstance(device, QSOnOffDuration)
    assert device.device_type == "water_boiler"
    assert QSWaterBoiler.conf_type_name == "water_boiler"

    # Reachable via the home's dashboard-section lookup
    home = hass.data[DOMAIN].get(home_config_entry.entry_id)
    assert home is not None
    devices_in_section = home.get_devices_for_dashboard_section("water_boilers")
    assert device in devices_in_section


async def test_water_boiler_with_temperature_sensor_attaches_probe(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When configured with a temp sensor, the entity id is stored and probed."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_WATER_BOILER_CONFIG,
        entry_id="water_boiler_probe_test",
        title=f"water_boiler: {MOCK_WATER_BOILER_CONFIG['name']}",
        unique_id="quiet_solar_water_boiler_probe_test",
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    device = hass.data[DOMAIN].get(entry.entry_id)
    assert device is not None

    assert device.water_boiler_temperature_sensor == "sensor.test_water_boiler_temperature"
    # attach_ha_state_to_probe populated the probe registry for the sensor
    assert "sensor.test_water_boiler_temperature" in device._entity_probed_state
    assert device._entity_probed_state_is_numerical["sensor.test_water_boiler_temperature"] is True


async def test_water_boiler_without_temperature_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When no temp sensor is configured, the field is None and no probe is attached."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_WATER_BOILER_CONFIG_NO_TEMP,
        entry_id="water_boiler_no_temp_test",
        title=f"water_boiler: {MOCK_WATER_BOILER_CONFIG_NO_TEMP['name']}",
        unique_id="quiet_solar_water_boiler_no_temp_test",
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    device = hass.data[DOMAIN].get(entry.entry_id)
    assert device is not None
    assert device.water_boiler_temperature_sensor is None
    # No probe entry should exist for the (missing) sensor entity id
    assert "sensor.test_water_boiler_temperature" not in device._entity_probed_state


async def test_water_boiler_select_translation_key(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """QSWaterBoiler returns the dedicated `water_boiler_mode` translation key."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_WATER_BOILER_CONFIG,
        entry_id="water_boiler_select_key_test",
        title=f"water_boiler: {MOCK_WATER_BOILER_CONFIG['name']}",
        unique_id="quiet_solar_water_boiler_select_key_test",
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    device = hass.data[DOMAIN].get(entry.entry_id)
    assert device is not None
    assert device.get_select_translation_key() == "water_boiler_mode"
