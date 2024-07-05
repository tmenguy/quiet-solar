from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType

from .data_handler import QSDataHandler

from homeassistant.helpers import config_validation as cv


from .const import (
    DATA_DEVICE_IDS,
    DOMAIN,
    PLATFORMS,
    DATA_HANDLER, DEVICE_TYPE
)
from .entity import LOAD_TYPES

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Quiet Solar component"""
    hass.data[DOMAIN] = {}

    return True

async def entry_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update listener.   Reload the data handler when the entry is updated.
     https://community.home-assistant.io/t/config-flow-how-to-update-an-existing-entity/522442/8 """
    await hass.config_entries.async_reload(entry.entry_id)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the Quiet Solar integration."""
    data_handler = hass.data[DOMAIN].get(DATA_HANDLER)
    if data_handler is None:
        data_handler = QSDataHandler(hass)
        hass.data[DOMAIN][DATA_HANDLER] = data_handler

    type = entry.data.get(DEVICE_TYPE)

    d = None
    if type is not None:
        if type in LOAD_TYPES:
            d = LOAD_TYPES[type](hass=hass, **entry.data)
        else:
            for t in LOAD_TYPES.values():
                # if t is a dict, then we can iterate on it ... only one level :)
                if isinstance(t, dict) and type in t:
                    d = t[type](hass=hass, **entry.data)
                    break

        if d:
            data_handler.add_device(d)

            platforms = d.get_platforms()

            if platforms:
                await hass.config_entries.async_forward_entry_setups(
                    entry, platforms
                )

            entry.async_on_unload(entry.add_update_listener(entry_update_listener))

    return True

