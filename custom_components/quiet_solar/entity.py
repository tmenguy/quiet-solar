from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.typing import UNDEFINED
from homeassistant.core import callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity

from .ha_model.home import QSHome
from .ha_model.battery import QSBattery
from .ha_model.car import QSCar
from .ha_model.charger import QSChargerOCPP, QSChargerWallbox, QSChargerGeneric
from .ha_model.fp_heater import QSFPHeater
from .ha_model.on_off_duration import QSOnOffDuration
from .ha_model.pool import QSPool
from .ha_model.solar import QSSolar
from .home_model.load import AbstractDevice
from .const import (
    DEFAULT_ATTRIBUTION,
    DOMAIN,
    MANUFACTURER,
)
from .ha_model.device import HADeviceMixin

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

def create_device_from_type(hass, home, type, config_entry: ConfigEntry):

    if config_entry is None:
        data = {}
    else:
        data = config_entry.data
    d = None
    if type is not None:
        if type in LOAD_TYPES:
            d = LOAD_TYPES[type](hass=hass, home=home, config_entry=config_entry, **data)
        else:
            for t in LOAD_TYPES.values():
                # if t is a dict, then we can iterate on it ... only one level :)
                if isinstance(t, dict) and type in t:
                    d = t[type](hass=hass, home=home, config_entry=config_entry, **data)
                    break
    return d


class QSBaseEntity(Entity):
    """QS entity base class."""

    _attr_attribution = DEFAULT_ATTRIBUTION
    _attr_has_entity_name = False

    def __init__(self, data_handler, description) -> None:
        """Set up QS entity base."""
        self.data_handler = data_handler
        self._attr_extra_state_attributes = {}
        self.entity_description = description

        if self.entity_description.name is UNDEFINED:
            _attr_has_entity_name = True

    @callback
    def async_update_callback(self) -> None:
        """Update the entity's state."""
        return





# this one is to be used for 'exported" HA entities that are describing a load, and so passthrough control of it
class QSDeviceEntity(QSBaseEntity):
    """QS entity base class."""
    device : AbstractDevice
    def __init__(self, data_handler, device: AbstractDevice, description) -> None:
        """Set up Netatmo entity base."""
        super().__init__(data_handler=data_handler, description=description)
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

    async def async_added_to_hass(self) -> None:
        """Entity created."""
        await super().async_added_to_hass()

        if isinstance(self.device, HADeviceMixin):
            self.device.attach_exposed_has_entity(self)

   # @property
   # def home(self) -> Home:
   #     """Return the home this room belongs to."""
   #     return self.device.device.home



