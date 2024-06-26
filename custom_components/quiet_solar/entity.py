from dataclasses import dataclass
from typing import Any

from .data_handler import QSDataHandler
from quiet_solar.ha_model.home import QSHome
from .ha_model.battery import QSBattery
from .ha_model.car import QSCar
from .ha_model.charger import QSChargerOCPP, QSChargerWallbox, QSChargerGeneric
from .ha_model.fp_heater import QSFPHeater
from .ha_model.on_off_duration import QSOnOffDuration
from .ha_model.pool import QSPool
from .ha_model.solar import QSSolar
from .home_model.load import AbstractLoad, AbstractDevice
from homeassistant.core import callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity
from .const import (
    DEFAULT_ATTRIBUTION,
    DOMAIN,
    MANUFACTURER,
)




LOAD_TYPES = {
    "home": QSHome,
    "battery": QSBattery,
    "solar": QSSolar,
    "charger": {"charger_ocpp": QSChargerOCPP, "charger_wallbox": QSChargerWallbox, "charger_generic": QSChargerGeneric},
    "car": QSCar,
    "pool": QSPool,
    "on_off_duration": QSOnOffDuration,
    "fp_heater": QSFPHeater
}


class QSBaseEntity(Entity):
    """QS entity base class."""

    _attr_attribution = DEFAULT_ATTRIBUTION
    _attr_has_entity_name = True

    def __init__(self, data_handler: QSDataHandler) -> None:
        """Set up Netatmo entity base."""
        self.data_handler = data_handler
        self._publishers: list[dict[str, Any]] = []
        self._attr_extra_state_attributes = {}

    async def async_added_to_hass(self) -> None:
        """Entity created."""
        await super().async_added_to_hass()

    async def async_will_remove_from_hass(self) -> None:
        """Run when entity will be removed from hass."""
        await super().async_will_remove_from_hass()

    @callback
    def async_update_callback(self) -> None:
        """Update the entity's state."""
        raise NotImplementedError

# this one is to be used for 'exported" HA entities that are describing a load, and so passthrough control of it
class QSDeviceEntity(QSBaseEntity):
    """QS entity base class."""
    device : AbstractDevice
    def __init__(self, data_handler: QSDataHandler, device: AbstractDevice) -> None:
        """Set up Netatmo entity base."""
        super().__init__(data_handler)
        self.device = device

        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, device.device_id)},
            name=f"{device.device_type} {device.name}",
            manufacturer=MANUFACTURER,
            model=device.device_type
        )

    @property
    def device_type(self) -> str:
        return self.device.device_type
   # @property
   # def home(self) -> Home:
   #     """Return the home this room belongs to."""
   #     return self.device.device.home




