import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Any, Coroutine

from homeassistant.components.switch import SwitchEntity, SwitchEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity, ExtraStoredData

from . import DOMAIN
from .const import SWITCH_CAR_NEXT_CHARGE_FULL, SWITCH_BEST_EFFORT_GREEN_ONLY, ENTITY_ID_FORMAT, \
    SWITCH_ENABLE_DEVICE, SWITCH_CAR_BUMP_SOLAR_CHARGE_PRIORITY
from .ha_model.car import QSCar
from .ha_model.charger import QSChargerGeneric
from .ha_model.device import HADeviceMixin
from .entity import QSDeviceEntity

_LOGGER = logging.getLogger(__name__)

from .ha_model.pool import QSPool
from .home_model.load import AbstractDevice, AbstractLoad


def create_ha_switch_for_QSCharger(device: QSChargerGeneric):
    entities = []

    qs_bump_solar_priority = QSSwitchEntityDescription(
        key=SWITCH_CAR_BUMP_SOLAR_CHARGE_PRIORITY,
        translation_key=SWITCH_CAR_BUMP_SOLAR_CHARGE_PRIORITY,
    )

    entities.append(QSSwitchEntityChargerOrCar(data_handler=device.data_handler, device=device, description=qs_bump_solar_priority))

    return entities


def create_ha_switch_for_QSCar(device: QSCar):
    entities = []

    qs_bump_solar_priority = QSSwitchEntityDescription(
        key=SWITCH_CAR_BUMP_SOLAR_CHARGE_PRIORITY,
        translation_key=SWITCH_CAR_BUMP_SOLAR_CHARGE_PRIORITY,
    )

    entities.append(QSSwitchEntityChargerOrCar(data_handler=device.data_handler, device=device, description=qs_bump_solar_priority))

    return entities

def create_ha_switch_for_QSPool(device: QSPool):
    entities = []
    return entities



def create_ha_switch_for_AbstractLoad(device: AbstractLoad):

    entities = []

    data_handler = None
    if isinstance(device, HADeviceMixin):
        data_handler = device.data_handler

    if device.support_green_only_switch():
        qs_green_only_description = QSSwitchEntityDescription(
            key=SWITCH_BEST_EFFORT_GREEN_ONLY,
            translation_key=SWITCH_BEST_EFFORT_GREEN_ONLY,
        )
        entities.append(QSSwitchEntityWithRestore(data_handler=data_handler, device=device, description=qs_green_only_description))


    qs_load_enabled_description = QSSwitchEntityDescription(
        key=SWITCH_ENABLE_DEVICE,
        translation_key=SWITCH_ENABLE_DEVICE,
    )
    entities.append(QSSwitchEntityWithRestore(data_handler=data_handler, device=device, description=qs_load_enabled_description))


    return entities

def create_ha_switch(device: AbstractDevice):

    ret = []
    if isinstance(device, QSChargerGeneric):
        ret.extend(create_ha_switch_for_QSCharger(device))

    if isinstance(device, QSCar):
        ret.extend(create_ha_switch_for_QSCar(device))

    if isinstance(device, QSPool):
        ret.extend(create_ha_switch_for_QSPool(device))

    if isinstance(device, AbstractLoad):
        ret.extend(create_ha_switch_for_AbstractLoad(device))

    return ret

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Quiet Solar switch platform."""
    device = hass.data[DOMAIN].get(entry.entry_id)

    if device:

        entities = create_ha_switch(device)
        for attached_device in device.get_attached_virtual_devices():
            entities.extend(create_ha_switch(attached_device))

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
            _LOGGER.error("async_unload_entry switch: exception for device %s %s", device.name, e, exc_info=True, stack_info=True)


    return True

@dataclass(frozen=True, kw_only=True)
class QSSwitchEntityDescription(SwitchEntityDescription):
    """Class describing qs switch button entities."""
    async_switch: Callable[[AbstractDevice, bool, bool], Coroutine] | None = None



@dataclass
class QSExtraStoredData(ExtraStoredData):
    """Object to hold extra stored data."""
    native_is_on: bool | None

    def as_dict(self) -> dict[str, Any]:
        """Return a dict representation of the text data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, restored: dict[str, Any]):
        """Initialize a stored text state from a dict."""
        try:
            return cls(
                restored["native_is_on"],
            )
        except Exception as e:
            _LOGGER.error("QSExtraStoredData.from_dict switch exception %s %s", restored, e, exc_info=True,
                          stack_info=True)
            return None

class QSSwitchEntity(QSDeviceEntity, SwitchEntity):

    entity_description: QSSwitchEntityDescription
    def __init__(
        self,
        data_handler,
        device: AbstractDevice,
        description: QSSwitchEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(data_handler=data_handler, device=device, description=description)
        self._set_availabiltiy()

    def _set_availabiltiy(self):
        if self.device.qs_enable_device is False and self.entity_description.key != SWITCH_ENABLE_DEVICE:
            self._attr_available = False
        else:
            self._attr_available = True

    @callback
    def async_update_callback(self, time:datetime) -> None:
        """Update the entity's state."""
        self._set_availabiltiy()

        if hasattr(self.device, self.entity_description.key):

            attr_val = getattr(self.device, self.entity_description.key, False)

            if attr_val != self._attr_is_on:
                self._attr_is_on = attr_val
                self.async_write_ha_state()

    async def async_turn_on(self, **kwargs: Any) -> None:

        _LOGGER.info(f"QSSwitchEntity:async_turn_on : {self.entity_description.key} on {self.device.name}")

        for_init = kwargs.pop('for_init', False)

        if self.entity_description.async_switch is not None:
            await self.entity_description.async_switch(self.device, True, for_init)
        else:
            setattr(self.device, self.entity_description.key, True)

        self._attr_is_on = True
        self._set_availabiltiy()
        self.async_write_ha_state()
        if self.device.home and not for_init:
            await self.device.home.force_update_all()

    async def async_turn_off(self, **kwargs: Any) -> None:

        _LOGGER.info(f"QSSwitchEntity:async_turn_off : {self.entity_description.key} on {self.device.name}")

        for_init = kwargs.pop('for_init', False)

        if self.entity_description.async_switch is not None:
            await self.entity_description.async_switch(self.device, False, for_init)
        else:
            setattr(self.device, self.entity_description.key, False)

        self._attr_is_on = False
        self._set_availabiltiy()
        self.async_write_ha_state()
        if self.device.home  and not for_init:
            await self.device.home.force_update_all()


class QSSwitchEntityWithRestore(QSSwitchEntity, RestoreEntity):
    """Mixin for button specific attributes."""

    @property
    def extra_restore_state_data(self) -> QSExtraStoredData:
        """Return sensor specific state data to be restored."""
        return QSExtraStoredData(self._attr_is_on)

    async def async_get_last_switch_data(self) -> QSExtraStoredData | None:
        """Restore native_value and native_unit_of_measurement."""
        if (restored_last_extra_data := await self.async_get_last_extra_data()) is None:
            return None
        return QSExtraStoredData.from_dict(restored_last_extra_data.as_dict())

    async def async_added_to_hass(self) -> None:
        """Restore ATTR_CHANGED_BY on startup since it is likely no longer in the activity log."""
        await super().async_added_to_hass()

        self._attr_is_on = None

        last_sensor_state = await self.async_get_last_switch_data()
        if (
            not last_sensor_state
        ):
            pass
        else:
            self._attr_is_on = last_sensor_state.native_is_on

        if self._attr_is_on is None:
            if hasattr(self.device, self.entity_description.key):
                self._attr_is_on = getattr(self.device, self.entity_description.key, False)
            else:
                self._attr_is_on = False


        if self._attr_is_on:
            await self.async_turn_on(for_init=True)
        else:
            await self.async_turn_off(for_init=True)


class QSSwitchEntityChargerOrCar(QSSwitchEntityWithRestore):

    def car(self) -> QSCar | None:
        """Return the car associated with this charger."""
        if isinstance(self.device, QSChargerGeneric):
            return self.device.car
        elif isinstance(self.device, QSCar):
            return self.device
        return None

    def charger(self) -> QSChargerGeneric | None:
        """Return the charger associated with this car."""
        if isinstance(self.device, QSChargerGeneric):
            return self.device
        elif isinstance(self.device, QSCar):
            return self.device.charger
        return None

    def _set_availabiltiy(self):

        if isinstance(self.device, QSChargerGeneric):
            self._attr_available = self.device.car is not None
            if self.device.qs_enable_device is False:
                self._attr_available = False
        elif isinstance(self.device, QSCar):
            self._attr_available = self.device.charger is not None
            if self.device.charger is not None and self.device.charger.qs_enable_device is False:
                self._attr_available = False
        else:
            self._attr_available = True




