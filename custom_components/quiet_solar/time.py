import logging
from dataclasses import dataclass
from datetime import time as dt_time
from datetime import datetime

from homeassistant.components.time import TimeEntity, TimeEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity, ExtraStoredData

from .ha_model.bistate_duration import QSBiStateDuration
from .ha_model.car import QSCar
from .ha_model.charger import QSChargerGeneric


from .home_model.load import AbstractDevice
from .const import (
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

from .entity import QSDeviceEntity

@dataclass(frozen=True, kw_only=True)
class QSTimeEntityDescription(TimeEntityDescription):
    qs_default_option:str | None  = None


def create_ha_time_for_QSCharger(device: QSChargerGeneric):
    entities = []

    selected_car_description = QSTimeEntityDescription(
        key="default_charge_time",
        translation_key="default_charge_time",
    )
    entities.append(QSBaseTime(data_handler=device.data_handler, device=device, description=selected_car_description))
    return entities

def create_ha_time_for_QSCar(device: QSCar):
    entities = []

    selected_car_description = QSTimeEntityDescription(
        key="default_charge_time",
        translation_key="default_charge_time",
    )
    entities.append(QSBaseTime(data_handler=device.data_handler, device=device, description=selected_car_description))
    return entities


def create_ha_time_for_QSBiStateDuration(device: QSBiStateDuration):
    entities = []

    selected_car_description = QSTimeEntityDescription(
        key="default_on_finish_time",
        translation_key="default_on_finish_time",
    )
    entities.append(QSBaseTime(data_handler=device.data_handler, device=device, description=selected_car_description))

    return entities

def create_ha_time(device: AbstractDevice):
    ret = []

    if isinstance(device, QSChargerGeneric):
        ret.extend(create_ha_time_for_QSCharger(device))

    if isinstance(device, QSCar):
        ret.extend(create_ha_time_for_QSCar(device))

    if isinstance(device, QSBiStateDuration):
        ret.extend(create_ha_time_for_QSBiStateDuration(device))

    return ret

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Quiet Solar time platform."""
    device = hass.data[DOMAIN].get(entry.entry_id)

    if device:

        entities = create_ha_time(device)

        if entities:
            async_add_entities(entities)
    return

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry):
    device = hass.data[DOMAIN].get(entry.entry_id)
    if device:
        try:
            if device.home:
                device.home.remove_device(device)
        except Exception as e:
            pass


    return True

class QSBaseTime(QSDeviceEntity, TimeEntity, RestoreEntity):
    """Implementation of a Qs DateTime sensor."""

    def __init__(
        self,
        data_handler,
        device: AbstractDevice,
        description: QSTimeEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(data_handler=data_handler, device=device, description=description)
        self._set_availabiltiy()

    async def async_added_to_hass(self) -> None:
        """Restore last state."""
        await super().async_added_to_hass()
        last_state = await self.async_get_last_state()

        val = None
        if ( last_state is not None
            and last_state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE)
        ):
            val = dt_time.fromisoformat(last_state.state)

        if val is None:
            val = getattr(self.device, self.entity_description.key, None)

        if val is None:
            val = dt_time(hour=7, minute=0, second=0)

        self._set_availabiltiy()

        await self.async_set_value(val)

    async def async_set_value(self, value: dt_time) -> None:
        """Change the value."""
        self._attr_native_value = value
        try:
            setattr(self.device, self.entity_description.key, value)
        except:
            _LOGGER.info(f"can't set time {value} on {self.device.name} for {self.entity_description.key}")
        self._set_availabiltiy()
        self.async_write_ha_state()

    @callback
    def async_update_callback(self, time:datetime) -> None:
        """Update the entity's state."""
        val = getattr(self.device, self.entity_description.key)

        if val is None:
            val = dt_time(hour=7, minute=0, second=0)

        self._set_availabiltiy()

        self._attr_native_value = val
        self.async_write_ha_state()

