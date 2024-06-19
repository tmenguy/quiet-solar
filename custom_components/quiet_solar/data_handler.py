
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.event import async_track_time_interval
from datetime import datetime, timedelta

from .const import (
    DATA_DEVICE_IDS,
    DOMAIN,
    PLATFORMS,
    DATA_HANDLER, DEVICE_TYPE
)
from .ha_model.home import QSHome
from .home_model.load import AbstractDevice


class QSDataHandler:

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize self."""
        self.hass : HomeAssistant = hass
        self._home: QSHome | None = None
        self._cached_devices :list[AbstractDevice] = []
        self._scan_interval = 1


    def add_device(self, device:AbstractDevice) -> None:
        """Add devices to the data handler."""
        if self._home is None:
            if isinstance(device, QSHome):
                self._home = device
                for d in self._cached_devices:
                    self._home.add_device(d)
                self._cached_devices = []
            else:
                self._cached_devices.append(device)
        else:
            self._home.add_device(device)


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
            #self._home = device


    async def async_dispatch(self) -> None:
        """Dispatch the creation of entities from the configuration."""


    async def async_update(self, event_time: datetime) -> None:
        pass




