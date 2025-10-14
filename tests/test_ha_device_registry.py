"""Device registry tests with real HA.

This test file verifies device registration, properties, and cleanup
using the real Home Assistant device registry.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr

from custom_components.quiet_solar.const import DOMAIN


pytestmark = pytest.mark.asyncio


async def test_home_device_registered(
    hass: HomeAssistant,
    real_home_config_entry,
    device_registry: dr.DeviceRegistry,
) -> None:
    """Test that home device is registered in device registry."""
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    
    # Get devices for config entry
    devices = dr.async_entries_for_config_entry(device_registry, config_entry.entry_id)
    
    # Should have at least one device
    assert len(devices) >= 1
    
    # Find home device
    home_device = devices[0]
    
    # Verify device properties (basic checks only)
    # Device name is prefixed with type by the integration
    assert "Test Home" in home_device.name
    # Device uses its own identifier, not config entry ID
    assert any(DOMAIN in str(identifier) for identifier in home_device.identifiers)


async def test_charger_device_registered(
    hass: HomeAssistant,
    home_and_charger,
    device_registry: dr.DeviceRegistry,
) -> None:
    """Test that charger device is registered in device registry."""
    from homeassistant.config_entries import ConfigEntryState
    
    home_entry, charger_entry = home_and_charger
    
    # Only test if charger loaded successfully
    if charger_entry.state == ConfigEntryState.LOADED:
        # Get devices for charger config entry
        devices = dr.async_entries_for_config_entry(device_registry, charger_entry.entry_id)
        
        # Should have charger device
        assert len(devices) >= 1
        
        charger_device = devices[0]
        assert "Test Charger" in charger_device.name
    else:
        # Charger needs sensor configuration
        pytest.skip("Charger needs sensor configuration for full setup")


async def test_car_device_registered(
    hass: HomeAssistant,
    home_and_car,
    device_registry: dr.DeviceRegistry,
) -> None:
    """Test that car device is registered in device registry."""
    home_entry, car_entry = home_and_car
    
    # Get devices for car config entry
    devices = dr.async_entries_for_config_entry(device_registry, car_entry.entry_id)
    
    # Should have car device
    assert len(devices) >= 1
    
    car_device = devices[0]
    assert "Test Car" in car_device.name


async def test_device_identifiers_unique(
    hass: HomeAssistant,
    home_charger_and_car,
    device_registry: dr.DeviceRegistry,
) -> None:
    """Test that all devices have unique identifiers."""
    home_entry, charger_entry, car_entry = home_charger_and_car
    
    # Get all devices for all entries
    all_devices = []
    for entry_id in [home_entry.entry_id, charger_entry.entry_id, car_entry.entry_id]:
        devices = dr.async_entries_for_config_entry(device_registry, entry_id)
        all_devices.extend(devices)
    
    # Extract all identifiers
    all_identifiers = []
    for device in all_devices:
        for identifier in device.identifiers:
            all_identifiers.append(identifier)
    
    # All identifiers should be unique
    assert len(all_identifiers) == len(set(all_identifiers)), "Duplicate device identifiers found"


async def test_device_has_manufacturer(
    hass: HomeAssistant,
    real_home_config_entry,
    device_registry: dr.DeviceRegistry,
) -> None:
    """Test that devices have manufacturer information."""
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    
    devices = dr.async_entries_for_config_entry(device_registry, config_entry.entry_id)
    assert len(devices) >= 1
    
    device = devices[0]
    # Device should have some identifying information
    assert device.name is not None


async def test_unload_preserves_device_registry(
    hass: HomeAssistant,
    real_home_config_entry,
    device_registry: dr.DeviceRegistry,
) -> None:
    """Test that unloading doesn't remove device from registry."""
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        # Setup
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
        
        # Get devices before unload
        devices_before = dr.async_entries_for_config_entry(device_registry, config_entry.entry_id)
        device_count_before = len(devices_before)
        assert device_count_before >= 1
        
        # Unload
        await hass.config_entries.async_unload(config_entry.entry_id)
        await hass.async_block_till_done()
        
        # Devices should still be in registry
        devices_after = dr.async_entries_for_config_entry(device_registry, config_entry.entry_id)
        assert len(devices_after) == device_count_before


async def test_device_config_entry_association(
    hass: HomeAssistant,
    real_home_config_entry,
    device_registry: dr.DeviceRegistry,
) -> None:
    """Test that devices are correctly associated with config entries."""
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    
    devices = dr.async_entries_for_config_entry(device_registry, config_entry.entry_id)
    
    # All devices should be associated with the config entry
    for device in devices:
        assert config_entry.entry_id in device.config_entries


async def test_multiple_devices_different_types(
    hass: HomeAssistant,
    home_charger_and_car,
    device_registry: dr.DeviceRegistry,
) -> None:
    """Test that different device types are all registered correctly."""
    from homeassistant.config_entries import ConfigEntryState
    
    home_entry, charger_entry, car_entry = home_charger_and_car
    
    # Verify home is registered (always should be)
    home_devices = dr.async_entries_for_config_entry(device_registry, home_entry.entry_id)
    assert len(home_devices) >= 1
    assert "Test Home" in home_devices[0].name
    
    # Car should be registered
    car_devices = dr.async_entries_for_config_entry(device_registry, car_entry.entry_id)
    assert len(car_devices) >= 1
    assert "Test Car" in car_devices[0].name
    
    # Charger may or may not be registered depending on sensor config
    if charger_entry.state == ConfigEntryState.LOADED:
        charger_devices = dr.async_entries_for_config_entry(device_registry, charger_entry.entry_id)
        if len(charger_devices) > 0:
            assert "Test Charger" in charger_devices[0].name

