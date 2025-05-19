from __future__ import annotations

import asyncio
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.helpers.typing import ConfigType
from homeassistant.helpers import service


from .data_handler import QSDataHandler

from homeassistant.helpers import config_validation as cv


from .const import (
    DATA_DEVICE_IDS,
    DOMAIN,
    DATA_HANDLER, DEVICE_TYPE
)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Quiet Solar component"""
    hass.data[DOMAIN] = {}
    
    # Register reload service
    register_reload_service(hass)
    
    return True


async def async_reload_quiet_solar(hass: HomeAssistant, except_for_entry_id=None):
    # Then reload the entire integration by getting all entries and reloading them
    entries = hass.config_entries.async_entries(DOMAIN)
    for entry in entries:
        if except_for_entry_id is not None and except_for_entry_id == entry.entry_id:
            continue
        await hass.config_entries.async_unload(entry.entry_id)

    hass.data[DOMAIN] = {}

    for entry in entries:
        await hass.config_entries.async_reload(entry.entry_id)

@callback
def register_reload_service(hass: HomeAssistant) -> None:
    """Register reload service for Quiet Solar."""
    async def _reload_integration(call: ServiceCall) -> None:
        """Reload all Quiet Solar config entries."""
        await async_reload_quiet_solar(hass)
    
    service.async_register_admin_service(
        hass,
        DOMAIN,
        "reload",
        _reload_integration,
    )

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the Quiet Solar integration."""
    data_handler = hass.data[DOMAIN].get(DATA_HANDLER)
    if data_handler is None:
        data_handler = QSDataHandler(hass)
        hass.data[DOMAIN][DATA_HANDLER] = data_handler

    await data_handler.async_add_entry(entry)
    
    # Register update listener to reload the entry when config entry is updated
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    data_handler = hass.data[DOMAIN].get(DATA_HANDLER)
    if data_handler and entry.entry_id in hass.data[DOMAIN]:
        device = hass.data[DOMAIN][entry.entry_id]
        platforms = device.get_platforms()

        if device.home:
            device.home.remove_device(device)

        if platforms:
            unload_ok = await hass.config_entries.async_unload_platforms(entry, platforms)
            if unload_ok:
                hass.data[DOMAIN].pop(entry.entry_id)
                return True
    
    return False

async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the config entry."""
    await hass.config_entries.async_reload(entry.entry_id)

