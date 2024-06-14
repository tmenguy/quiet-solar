from dataclasses import dataclass

from homeassistant.config_entries import ConfigType
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from home_model.load import AbstractLoad

from homeassistant.helpers.event import async_track_time_interval
from datetime import datetime, timedelta

from .const import (
    DATA_DEVICE_IDS,
    DOMAIN,
    PLATFORMS,
    DATA_HANDLER, DEVICE_TYPE
)



class QSDataHandler:

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize self."""
        self.hass : HomeAssistant = hass
        self._home = None

    async def async_add_entry(self, config_entry: ConfigEntry) -> None:

        device_type = config_entry.data[DEVICE_TYPE]

        await self.async_dispatch()

        if device_type == "home":

            if self._home is None:
                config_entry.async_on_unload(
                    async_track_time_interval(
                        self.hass, self.async_update, timedelta(seconds=self._scan_interval)
                    )
                )
            self._home = device


    async def async_dispatch(self) -> None:
        """Dispatch the creation of entities from the configuration."""






    async def async_update(self, event_time: datetime) -> None:
        pass




