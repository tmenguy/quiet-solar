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
    SWITCH_POOL_FORCE_WINTER_MODE, SWITCH_ENABLE_DEVICE, SWITCH_CAR_BUMP_SOLAR_CHARGE_PRIORITY
from .ha_model.car import QSCar
from .ha_model.charger import QSChargerGeneric
from .ha_model.device import HADeviceMixin
from .entity import QSDeviceEntity

from homeassistant.const import (
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    Platform
)

from .ha_model.pool import QSPool
from .home_model.load import AbstractDevice, AbstractLoad


def create_ha_switch_for_QSCharger(device: QSChargerGeneric):
    entities = []


    qs_next_charge_full = QSSwitchEntityDescription(
        key=SWITCH_CAR_NEXT_CHARGE_FULL,
        translation_key=SWITCH_CAR_NEXT_CHARGE_FULL,
    )

    entities.append(QSSwitchEntityChargerOrCarFullCharge(data_handler=device.data_handler, device=device, description=qs_next_charge_full))

    qs_bump_solar_priority = QSSwitchEntityDescription(
        key=SWITCH_CAR_BUMP_SOLAR_CHARGE_PRIORITY,
        translation_key=SWITCH_CAR_BUMP_SOLAR_CHARGE_PRIORITY,
    )

    entities.append(QSSwitchEntityChargerOrCar(data_handler=device.data_handler, device=device, description=qs_bump_solar_priority))



    return entities


def create_ha_switch_for_QSCar(device: QSCar):
    entities = []


    qs_next_charge_full = QSSwitchEntityDescription(
        key=SWITCH_CAR_NEXT_CHARGE_FULL,
        translation_key=SWITCH_CAR_NEXT_CHARGE_FULL,
    )

    entities.append(QSSwitchEntityChargerOrCarFullCharge(data_handler=device.data_handler, device=device, description=qs_next_charge_full))

    qs_bump_solar_priority = QSSwitchEntityDescription(
        key=SWITCH_CAR_BUMP_SOLAR_CHARGE_PRIORITY,
        translation_key=SWITCH_CAR_BUMP_SOLAR_CHARGE_PRIORITY,
    )

    entities.append(QSSwitchEntityChargerOrCar(data_handler=device.data_handler, device=device, description=qs_bump_solar_priority))



    return entities

def create_ha_switch_for_QSPool(device: QSPool):
    entities = []


    qs_force_winter = QSSwitchEntityDescription(
        key=SWITCH_POOL_FORCE_WINTER_MODE,
        translation_key=SWITCH_POOL_FORCE_WINTER_MODE,
    )

    entities.append(QSSwitchEntityWithRestore(data_handler=device.data_handler, device=device, description=qs_force_winter))


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
    """Set up the Netatmo switch platform."""
    device = hass.data[DOMAIN].get(entry.entry_id)

    if device:

        entities = create_ha_switch(device)

        if entities:
            async_add_entities(entities)

    return


@dataclass(frozen=True, kw_only=True)
class QSSwitchEntityDescription(SwitchEntityDescription):
    """Class describing Renault button entities."""
    set_val: Callable[[AbstractDevice, bool], Coroutine] | None = None



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
        except KeyError:
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

        setattr(self.device, self.entity_description.key, True)

        self._attr_is_on = True
        self._set_availabiltiy()
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:

        setattr(self.device, self.entity_description.key, False)

        self._attr_is_on = False
        self._set_availabiltiy()
        self.async_write_ha_state()


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

        if self._attr_is_on is  None:
            self._attr_is_on = getattr(self.device, self.entity_description.key, False)


        if self._attr_is_on:
            await self.async_turn_on()
        else:
            await self.async_turn_off()


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




class QSSwitchEntityChargerOrCarFullCharge(QSSwitchEntityChargerOrCar):

    @callback
    def async_update_callback(self, time:datetime) -> None:
        """Update the entity's state."""
        self._set_availabiltiy()
        has_set = False
        car = self.car()

        if car is not None:
            if car.car_default_charge == 100:
                # force it at on in case the car wants a hundred anyway
                self._attr_is_on = True
                has_set = True

            if not has_set:
                self._attr_is_on = car.is_next_charge_full()

        self.async_write_ha_state()

    async def async_turn_on(self, **kwargs: Any) -> None:

        self._set_availabiltiy()
        car = self.car()

        if car is not None:
            if car.car_default_charge == 100:
                # no need to force the state of the charge here
                await car.set_next_charge_full_or_not(False)
            else:
                await car.set_next_charge_full_or_not(True)

        self._attr_is_on = True
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:

        self._set_availabiltiy()
        car = self.car()

        if car is not None:
            await car.set_next_charge_full_or_not(False)

        has_set = False
        if car is not None:
            if car.car_default_charge == 100:
                self._attr_is_on = True
                has_set = True


        if not has_set:
            self._attr_is_on = False

        self.async_write_ha_state()


