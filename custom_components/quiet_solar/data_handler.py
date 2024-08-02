
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.event import async_track_time_interval
from datetime import datetime, timedelta

from .entity import create_device_from_type
from .const import (
    DOMAIN,
    DEVICE_TYPE
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
        self._cached_config_entries :list[ConfigEntry] = []
        self._scan_interval = 1
        self._refresh_constraints_interval = 10
        self.update_subscribers = {}

    def _add_device(self, config_entry: ConfigEntry ):
        type = config_entry.data.get(DEVICE_TYPE)
        d = create_device_from_type(hass=self.hass, home=self.home, type=type, config_entry=config_entry)
        self.hass.data[DOMAIN][config_entry.entry_id] = d
        self.home.add_device(d)
        return d

    def subscribe_to_update(self, subscriber, update_callback):
        self.update_subscribers[update_callback] = subscriber

    def unsubscribe_from_update(self, update_callback):
        return self.update_subscribers.pop(update_callback, None)


    async def async_add_entry(self, config_entry: ConfigEntry) -> None:

        type = config_entry.data.get(DEVICE_TYPE)
        do_home_register = False
        config_entry_to_forward = []

        if self.home is None:
            if type == "home":
                self.home = create_device_from_type(hass=self.hass, home=None, type=type, config_entry=config_entry)
                self.hass.data[DOMAIN][config_entry.entry_id] = self.home
                do_home_register = True
                config_entry_to_forward = [self.home]
                for c_c_entry in self._cached_config_entries:
                    c_d = self._add_device(c_c_entry)
                    config_entry_to_forward.append(c_d)
                self._cached_config_entries = []
            else:
                self._cached_config_entries.append(config_entry)
        else:
            config_entry_to_forward = [self._add_device(config_entry)]


        for d in config_entry_to_forward:

            platforms = d.get_platforms()

            if platforms:
                await self.hass.config_entries.async_forward_entry_setups(
                    d.config_entry, platforms
                )

            # config_entry.async_on_unload(d.config_entry.add_update_listener(entry_update_listener))

        if do_home_register:

            config_entry.async_on_unload(
                async_track_time_interval(
                    self.hass, self.async_update, timedelta(seconds=self._scan_interval)
                )
            )

            config_entry.async_on_unload(
                async_track_time_interval(
                    self.hass, self.async_update_loads_contraints, timedelta(seconds=self._refresh_constraints_interval)
                )
            )


    async def async_update(self, event_time: datetime) -> None:
        await self.home.update(event_time)

    async def async_update_loads_contraints(self, event_time: datetime) -> None:
        await self.home.update_loads_constraints(event_time)




