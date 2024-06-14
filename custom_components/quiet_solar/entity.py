from dataclasses import dataclass
from typing import Any

from homeassistant.core import HomeAssistant
from abc import abstractmethod

from data_handler import QSDataHandler
from home_model.home import Home
from home_model.load import AbstractLoad
from abc import ABC
from homeassistant.core import callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity
from .const import (
    DEFAULT_ATTRIBUTION,
    DOMAIN,
    MANUFACTURER,
)


LOAD_TYPES = {
    "home": "QSDeviceEntityHome",
    "battery": "QSDeviceEntityBattery",
    "solar": "QSDeviceEntitySolar",
    "charger": "QSDeviceEntityCharger",
    "car": "QSDeviceEntityCar",
    "pool": "QSDeviceEntityPool",
    "switch": "SQSDeviceEntitySwitch"
}


@dataclass
class QSLoad:
    """QS device class proxy"""
    data_handler: QSDataHandler
    load: AbstractLoad

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


class QSDeviceEntity(QSBaseEntity):
    """QS entity base class."""
    load : QSLoad
    def __init__(self, data_handler: QSDataHandler, load: QSLoad) -> None:
        """Set up Netatmo entity base."""
        super().__init__(data_handler)
        self.load = load

        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, load.load.load_id)},
            name=load.load.name,
            manufacturer=MANUFACTURER,
            model=load.load.load_type()
        )

    @property
    def device_type(self) -> str:
        return self.load.load.load_type()
    @property
    def home(self) -> Home:
        """Return the home this room belongs to."""
        return self.load.load.home




