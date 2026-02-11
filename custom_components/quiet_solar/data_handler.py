import logging
import asyncio

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.event import async_track_time_interval
from datetime import datetime, timedelta

from .entity import create_device_from_type
from .const import (
    DOMAIN,
    DEVICE_TYPE
)
from .ui.dashboard import async_auto_generate_if_first_install

_LOGGER = logging.getLogger(__name__)

from .ha_model.home import QSHome

#async def entry_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
#    """Update listener.   Reload the data handler when the entry is updated.
#     https://community.home-assistant.io/t/config-flow-how-to-update-an-existing-entity/522442/8 """
#    await hass.config_entries.async_reload(entry.entry_id)

class QSDataHandler:

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize self."""
        self.hass : HomeAssistant = hass
        self.home: QSHome | None = None
        self._cached_config_entries :list[ConfigEntry] = []

        self._load_update_scan_interval = 7
        self._refresh_states_interval = 4
        self._refresh_forecast_probers_interval = 30

        self._update_loads_lock = asyncio.Lock()
        self._update_all_states_lock = asyncio.Lock()
        self._update_forecast_probers_lock = asyncio.Lock()


    def _add_device(self, config_entry: ConfigEntry ):
        type = config_entry.data.get(DEVICE_TYPE)
        d = create_device_from_type(hass=self.hass, home=self.home, type=type, config_entry=config_entry)
        if d is None:
            _LOGGER.error("Could not create device: %s", config_entry.entry_id)
            return None
        self.hass.data[DOMAIN][config_entry.entry_id] = d
        self.home.add_device(d)

        return d

    async def async_add_entry(self, config_entry: ConfigEntry) -> None:

        type = config_entry.data.get(DEVICE_TYPE)
        do_home_register = False
        config_entry_to_forward = []

        if self.home is None:
            if type == QSHome.conf_type_name:
                self.home = create_device_from_type(hass=self.hass, home=None, type=type, config_entry=config_entry)
                self.hass.data[DOMAIN][config_entry.entry_id] = self.home
                do_home_register = True
                config_entry_to_forward = [self.home]
                for c_c_entry in self._cached_config_entries:
                    c_d = self._add_device(c_c_entry)
                    if c_d is not None:
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

            d.config_entry_initialized = True

            # config_entry.async_on_unload(d.config_entry.add_update_listener(entry_update_listener))

        if do_home_register:

            config_entry.async_on_unload(
                async_track_time_interval(
                    self.hass, self.async_update_loads, timedelta(seconds=self._load_update_scan_interval)
                )
            )

            config_entry.async_on_unload(
                async_track_time_interval(
                    self.hass, self.async_update_all_states, timedelta(seconds=self._refresh_states_interval)
                )
            )

            config_entry.async_on_unload(
                async_track_time_interval(
                    self.hass, self.async_update_forecast_probers, timedelta(seconds=self._refresh_forecast_probers_interval)
                )
            )

            # On first install, auto-generate dashboards now that the home
            # and all its devices are ready.  On subsequent startups this is
            # a no-op (tracking data already exists).
            try:
                await async_auto_generate_if_first_install(self.home)
            except Exception:
                _LOGGER.warning(
                    "Auto-generation of dashboards failed", exc_info=True
                )


    async def async_update_loads(self, event_time: datetime) -> None:
        if self._update_loads_lock.locked():
            _LOGGER.info("Re-entry detected in async_update_loads, skipping this run.")
            return
        async with self._update_loads_lock:
            try:
                await self.home.update_loads(event_time)
            except Exception as e:
                _LOGGER.error("Error updating loads: %s", e, exc_info=True, stack_info=True)

    async def async_update_all_states(self, event_time: datetime) -> None:
        if self._update_all_states_lock.locked():
            _LOGGER.info("Re-entry detected in async_update_all_states, skipping this run.")
            return
        async with self._update_all_states_lock:
            try:
                await self.home.update_all_states(event_time)
            except Exception as e:
                _LOGGER.error("Error updating all states: %s", e, exc_info=True)

    async def async_update_forecast_probers(self, event_time: datetime) -> None:
        if self._update_forecast_probers_lock.locked():
            _LOGGER.info("Re-entry detected in async_update_forecast_probers, skipping this run.")
            return
        async with self._update_forecast_probers_lock:
            try:
                await self.home.update_forecast_probers(event_time)
            except Exception as e:
                _LOGGER.error("Error updating forecast probers: %s", e, exc_info=True)