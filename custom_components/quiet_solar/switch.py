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
    SWITCH_POOL_FORCE_WINTER_MODE
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

    entities.append(QSSwitchEntityChargerFullCharge(data_handler=device.data_handler, device=device, description=qs_next_charge_full))


    return entities

def create_ha_switch_for_QSPool(device: QSPool):
    entities = []


    qs_force_winter = QSSwitchEntityDescription(
        key=SWITCH_POOL_FORCE_WINTER_MODE,
        translation_key=SWITCH_POOL_FORCE_WINTER_MODE,
    )

    entities.append(QSSwitchEntity(data_handler=device.data_handler, device=device, description=qs_force_winter))


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
        entities.append(QSSwitchEntity(data_handler=data_handler, device=device, description=qs_green_only_description))




    return entities

def create_ha_switch(device: AbstractDevice):

    ret = []
    if isinstance(device, QSChargerGeneric):
        ret.extend(create_ha_switch_for_QSCharger(device))

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

class QSSwitchEntity(QSDeviceEntity, SwitchEntity, RestoreEntity):
    """Mixin for button specific attributes."""

    entity_description: QSSwitchEntityDescription
    def __init__(
        self,
        data_handler,
        device: AbstractDevice,
        description: QSSwitchEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(data_handler=data_handler, device=device, description=description)

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

        self._attr_is_on = False

        last_sensor_state = await self.async_get_last_switch_data()
        if (
            not last_sensor_state
        ):
            pass
        else:
            self._attr_is_on = last_sensor_state.native_is_on

        if self._attr_is_on is  None:
            self._attr_is_on = False


        if self._attr_is_on:
            await self.async_turn_on()
        else:
            await self.async_turn_off()

    @callback
    def async_update_callback(self, time:datetime) -> None:
        """Update the entity's state."""
        pass


    async def async_turn_on(self, **kwargs: Any) -> None:

        setattr(self.device, self.entity_description.key, True)

        self._attr_is_on = True
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:

        setattr(self.device, self.entity_description.key, False)

        self._attr_is_on = False
        self.async_write_ha_state()


class QSSwitchEntityChargerFullCharge(QSSwitchEntity):

    device: QSChargerGeneric

    @callback
    def async_update_callback(self, time:datetime) -> None:
        """Update the entity's state."""
        # if self.device.car is not None:
        #    if self.device.car.car_default_charge == 100:
        #        # force it at on in case the car wants a hundred anyway
        #        self._attr_is_on = True
        pass

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the zone on."""
        #await self.device.async_on()
        if isinstance(self.device, QSChargerGeneric):
            await self.device.set_next_charge_full_or_not(True)

        self._attr_is_on = True
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the zone off."""
        #await self.device.async_off()
        new_value = False
        if isinstance(self.device, QSChargerGeneric):
            await self.device.set_next_charge_full_or_not(False)

        self._attr_is_on = new_value
        self.async_write_ha_state()
