from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Any, Coroutine

from homeassistant.components.switch import SwitchEntity, SwitchEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity, ExtraStoredData

from . import DOMAIN
from .const import SWITCH_CAR_NEXT_CHARGE_FULL
from .ha_model.charger import QSChargerGeneric
from .home_model.commands import LoadCommand, CMD_ON, CMD_IDLE, CMD_OFF
from .entity import QSDeviceEntity

from homeassistant.const import (
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    Platform
)

from .home_model.load import AbstractDevice


def create_ha_switch_for_QSCharger(device: QSChargerGeneric):
    entities = []


    qs_reset_history = QSSwitchEntityDescription(
        key=SWITCH_CAR_NEXT_CHARGE_FULL,
        translation_key=SWITCH_CAR_NEXT_CHARGE_FULL,
        async_press=lambda x: x.device.reset_forecasts(),
    )

    entities.append(QSSwitchEntity(data_handler=device.data_handler, device=device, description=qs_reset_history))


    return entities

def create_ha_button(device: AbstractDevice):

    ret = []
    if isinstance(device, QSChargerGeneric):
        ret.extend(create_ha_switch_for_QSCharger(device))

    return ret

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Netatmo switch platform."""
    device = hass.data[DOMAIN].get(entry.entry_id)

    if device:

        entities = create_ha_button(device)

        if entities:
            async_add_entities(entities)

    return


@dataclass(frozen=True, kw_only=True)
class QSSwitchEntityDescription(SwitchEntityDescription):
    """Class describing Renault button entities."""
    async_press: Callable[[Any], Coroutine]


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
    _attr_has_entity_name = True
    def __init__(
        self,
        data_handler,
        device: AbstractDevice,
        description: QSSwitchEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        self._attr_has_entity_name = True
        super().__init__(data_handler=data_handler, device=device, description=description)
        self.entity_description = description

        self._attr_unique_id = (
            f"switch-{self.device.device_id}-{description.key}"
        )

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
        if isinstance(self.device, QSChargerGeneric):
            if self.device.car is not None:
                if self.device.car.car_default_charge == 100:
                    # force it at on in case the car wants a hundred anyway
                    self._attr_is_on = True


    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the zone on."""
        #await self.device.async_on()
        if isinstance(self.device, QSChargerGeneric):
            self.device.set_next_charge_full_or_not(True)
        self._attr_is_on = True
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the zone off."""
        #await self.device.async_off()
        new_value = False
        if isinstance(self.device, QSChargerGeneric):
            self.device.set_next_charge_full_or_not(False)

            if self.device.is_next_charge_full():
                new_value = True


        if  self._attr_is_on != new_value:
            self._attr_is_on = new_value
            self.async_write_ha_state()




# a HaSwitchLoad is a load that is controlled by a switch entity, and it have to expose a switch for direct control to enforce a constraint
class QSSwitchLoad(QSDeviceEntity, SwitchEntity):


    _attr_name = None

    def __init__(self, switch_entity:str, **kwargs):
        super().__init__(**kwargs)
        self._switch_entity = switch_entity

    async def execute_command(self, time: datetime, command:LoadCommand) -> bool:
        if command.is_like(CMD_ON):
            action = SERVICE_TURN_ON
        elif command.is_like(CMD_OFF) or command.is_like(CMD_IDLE):
            action = SERVICE_TURN_OFF
        else:
            raise ValueError("Invalid command")

        await self.hass.services.async_call(
            domain=Platform.SWITCH,
            service=action,
            target={"entity_id": self._switch_entity},
        )

        return False
