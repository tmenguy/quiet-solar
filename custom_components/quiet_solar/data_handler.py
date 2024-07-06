
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.event import async_track_time_interval
from datetime import datetime, timedelta

from .entity import create_device_from_type
from .const import (
    DATA_DEVICE_IDS,
    DOMAIN,
    PLATFORMS,
    DATA_HANDLER, DEVICE_TYPE
)
from quiet_solar.ha_model.home import QSHome
from quiet_solar.home_model.load import AbstractDevice

async def entry_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update listener.   Reload the data handler when the entry is updated.
     https://community.home-assistant.io/t/config-flow-how-to-update-an-existing-entity/522442/8 """
    await hass.config_entries.async_reload(entry.entry_id)


class QSDataHandler:

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize self."""
        self.hass : HomeAssistant = hass
        self.home: QSHome | None = None
        self._cached_devices :list[AbstractDevice] = []
        self._scan_interval = 1


    def add_device(self, device:AbstractDevice) -> None:
        """Add devices to the data handler."""
        if self.home is None:
            if isinstance(device, QSHome):
                self.home = device
                for d in self._cached_devices:
                    self.home.add_device(d)
                self._cached_devices = []

            else:
                self._cached_devices.append(device)
        else:
            self.home.add_device(device)


    async def async_add_entry(self, config_entry: ConfigEntry) -> None:

        type = config_entry.data.get(DEVICE_TYPE)

        d = create_device_from_type(self.hass, type, config_entry.data)

        self.hass.data[DOMAIN][config_entry.entry_id] = d
        do_home_register = False


        if self.home is None:
            if isinstance(d, QSHome):
                self.home = d
                do_home_register = True
                for d in self._cached_devices:
                    self.home.add_device(d)
                self._cached_devices = []

            else:
                self._cached_devices.append(d)
        else:
            self.home.add_device(d)


        platforms = d.get_platforms()

        if platforms:
            await self.hass.config_entries.async_forward_entry_setups(
                config_entry, platforms
            )

        # config_entry.async_on_unload(config_entry.add_update_listener(entry_update_listener))

        if do_home_register:

            config_entry.async_on_unload(
                async_track_time_interval(
                    self.hass, self.async_update, timedelta(seconds=self._scan_interval)
                )
            )


    async def async_update(self, event_time: datetime) -> None:
        await self.home.update(event_time)




