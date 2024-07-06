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

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Quiet Solar component"""
    hass.data[DOMAIN] = {}

    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the Quiet Solar integration."""
    data_handler = hass.data[DOMAIN].get(DATA_HANDLER)
    if data_handler is None:
        data_handler = QSDataHandler(hass)
        hass.data[DOMAIN][DATA_HANDLER] = data_handler

    await data_handler.async_add_entry(entry)

    return True

