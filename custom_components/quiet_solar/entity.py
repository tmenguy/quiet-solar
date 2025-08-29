from datetime import datetime

from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.typing import UNDEFINED
from homeassistant.core import callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity, async_generate_entity_id

from .ha_model.climate_controller import QSClimateDuration
from .ha_model.dynamic_group import QSDynamicGroup
from .ha_model.home import QSHome
from .ha_model.battery import QSBattery
from .ha_model.car import QSCar
from .ha_model.charger import QSChargerOCPP, QSChargerWallbox, QSChargerGeneric
from .ha_model.on_off_duration import QSOnOffDuration
from .ha_model.pool import QSPool
from .ha_model.solar import QSSolar
from .home_model.load import AbstractDevice
from .const import (
    DEFAULT_ATTRIBUTION,
    DOMAIN,
    MANUFACTURER, ENTITY_ID_FORMAT, )
from .ha_model.device import HADeviceMixin


LOAD_TYPE_LIST = [QSHome, QSBattery, QSSolar, QSChargerOCPP, QSChargerWallbox, QSChargerGeneric, QSCar, QSPool, QSOnOffDuration, QSClimateDuration, QSDynamicGroup]
LOAD_TYPE__DICT = {t.conf_type_name:t for t in LOAD_TYPE_LIST}

LOAD_NAMES = {
    QSHome.conf_type_name : "home",
    QSBattery.conf_type_name: "battery",
    QSSolar.conf_type_name: "solar",
    "charger": "charger",
    QSChargerOCPP.conf_type_name: "charger",
    QSChargerWallbox.conf_type_name: "charger",
    QSChargerGeneric.conf_type_name: "charger",
    QSCar.conf_type_name : "car",
    QSPool.conf_type_name:"pool",
    QSOnOffDuration.conf_type_name: "on/off",
    QSClimateDuration.conf_type_name:"climate",
    QSDynamicGroup.conf_type_name:"group"
}


def create_device_from_type(hass, home, type, config_entry: ConfigEntry):

    if config_entry is None:
        data = {}
    else:
        data = config_entry.data
    d = None
    if type is not None:
        if type in LOAD_TYPE__DICT:
            d = LOAD_TYPE__DICT[type](hass=hass, home=home, config_entry=config_entry, **data)
    return d


class QSBaseEntity(Entity):
    """QS entity base class."""

    _attr_attribution = DEFAULT_ATTRIBUTION
    _attr_has_entity_name = True

    def __init__(self, data_handler, description) -> None:
        """Set up QS entity base."""
        self.data_handler = data_handler
        self._attr_extra_state_attributes = {}
        self.entity_description = description

        if not (self.entity_description.name is UNDEFINED or self.entity_description.name is None):
            self._attr_has_entity_name = False
        if not (self.entity_description.translation_key is UNDEFINED or self.entity_description.translation_key is None):
            self._attr_has_entity_name = True

    def _set_availabiltiy(self):
        self._attr_available = True

    @callback
    def async_update_callback(self, time:datetime) -> None:
        """Update the entity's state."""
        self._set_availabiltiy()


# this one is to be used for 'exported" HA entities that are describing a load, and so passthrough control of it
class QSDeviceEntity(QSBaseEntity):
    """QS entity base class."""
    device : AbstractDevice

    def __init__(self, data_handler, device: AbstractDevice, description) -> None:
        """Set up Quiet Solar entity base."""
        super().__init__(data_handler=data_handler, description=description)
        self.device = device
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, device.device_id)},
            name=f"{LOAD_NAMES.get(device.device_type, device.device_type)} {device.name}",
            manufacturer=MANUFACTURER,
            model=device.device_type
        )
        self._attr_unique_id = f"{self.device.device_id}-{description.key}"
        self.entity_id = async_generate_entity_id(
            ENTITY_ID_FORMAT, name=self._attr_unique_id, hass=data_handler.hass
        )

    @property
    def device_type(self) -> str:
        return self.device.device_type

    async def async_added_to_hass(self) -> None:
        """Entity created."""
        await super().async_added_to_hass()

        if isinstance(self.device, HADeviceMixin):
            self.device.attach_exposed_has_entity(self)

        self._set_availabiltiy()

    def _set_availabiltiy(self):
        if self.device.qs_enable_device is False:
            self._attr_available = False
        else:
            self._attr_available = True


   # @property
   # def home(self) -> Home:
   #     """Return the home this room belongs to."""
   #     return self.device.device.home



