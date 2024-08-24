from dataclasses import dataclass
from typing import Callable, Coroutine, Any

from homeassistant.components.button import ButtonEntityDescription, ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, BUTTON_HOME_RESET_HISTORY, BUTTON_HOME_SERIALIZE_FOR_DEBUG, BUTTON_CAR_NEXT_CHARGE_FORCE_NOW
from .entity import QSDeviceEntity
from .ha_model.charger import QSChargerGeneric
from .ha_model.home import QSHome
from .home_model.load import AbstractDevice


def create_ha_button_for_QSHome(device: QSHome):
    entities = []


    qs_reset_history = QSButtonEntityDescription(
        key=BUTTON_HOME_RESET_HISTORY,
        translation_key=BUTTON_HOME_RESET_HISTORY,
        async_press=lambda x: x.device.reset_forecasts(),
    )

    entities.append(QSButtonEntity(data_handler=device.data_handler, device=device, description=qs_reset_history))

    home_serialize_debug = QSButtonEntityDescription(
        key=BUTTON_HOME_SERIALIZE_FOR_DEBUG,
        translation_key=BUTTON_HOME_SERIALIZE_FOR_DEBUG,
        async_press=lambda x: x.device.dump_for_debug(),
    )

    entities.append(QSButtonEntity(data_handler=device.data_handler, device=device, description=home_serialize_debug))

    return entities


def create_ha_button_for_QSChargerGeneric(device: QSChargerGeneric):
    entities = []


    qs_reset_history = QSButtonEntityDescription(
        key=BUTTON_CAR_NEXT_CHARGE_FORCE_NOW,
        translation_key=BUTTON_CAR_NEXT_CHARGE_FORCE_NOW,
        async_press=lambda x: x.device.force_charge_now(),
    )

    entities.append(QSButtonEntity(data_handler=device.data_handler, device=device, description=qs_reset_history))

    return entities

def create_ha_button(device: AbstractDevice):

    ret = []
    if isinstance(device, QSHome):
        ret.extend(create_ha_button_for_QSHome(device))

    if isinstance(device, QSChargerGeneric):
        ret.extend(create_ha_button_for_QSChargerGeneric(device))
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
class QSButtonEntityDescription(ButtonEntityDescription):
    """Class describing Renault button entities."""
    async_press: Callable[[Any], Coroutine]



class QSButtonEntity(QSDeviceEntity, ButtonEntity):
    """Mixin for button specific attributes."""

    entity_description: QSButtonEntityDescription
    _attr_has_entity_name = True
    def __init__(
        self,
        data_handler,
        device: AbstractDevice,
        description: QSButtonEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        self._attr_has_entity_name = True
        super().__init__(data_handler=data_handler, device=device, description=description)
        self.entity_description = description

        self._attr_unique_id = (
            f"button-{self.device.device_id}-{description.key}"
        )
    async def async_press(self) -> None:
        """Process the button press."""
        await self.entity_description.async_press(self)

