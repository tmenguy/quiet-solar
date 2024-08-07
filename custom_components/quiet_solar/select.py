from dataclasses import dataclass

from homeassistant.components.select import SelectEntityDescription, SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .entity import QSDeviceEntity
from .ha_model.car import QSCar
from .ha_model.charger import QSChargerGeneric
from .ha_model.home import QSHome, QSHomeMode
from .home_model.load import AbstractDevice
from .const import (
    DOMAIN,
)
@dataclass(frozen=True, kw_only=True)
class QSSelectEntityDescription(SelectEntityDescription):
    qs_default_option:str | None  = None


def create_ha_select_for_QSCar(device: QSCar):
    entities = []
    return entities

def create_ha_select_for_QSCharger(device: QSChargerGeneric):
    entities = []
    return entities

def create_ha_select_for_QSHome(device: QSHome):
    entities = []

    home_mode_description = QSSelectEntityDescription(
        key="home_mode",
        translation_key="home_mode",
        options= list(map(str, QSHomeMode)),
        qs_default_option=QSHomeMode.HOME_MODE_SENSORS_ONLY.value
    )
    entities.append(QSBaseSelectRestore(data_handler=device.data_handler, device=device, description=home_mode_description))

    return entities


def create_ha_select(device: AbstractDevice):

    if isinstance(device, QSCar):
        return create_ha_select_for_QSCar(device)
    elif isinstance(device, QSChargerGeneric):
        return create_ha_select_for_QSCharger(device)
    elif isinstance(device, QSHome):
        return create_ha_select_for_QSHome(device)

    return []

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Netatmo switch platform."""
    device = hass.data[DOMAIN].get(entry.entry_id)

    if device:

        entities = create_ha_select(device)

        if entities:
            async_add_entities(entities)

    return


class QSBaseSelect(QSDeviceEntity, SelectEntity):
    """Implementation of a Netatmo sensor."""

    def __init__(
        self,
        data_handler,
        device: AbstractDevice,
        description: QSSelectEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(data_handler=data_handler, device=device, description=description)
        self.entity_description = description

        self._attr_unique_id = (
            f"{self.device.device_id}-{description.key}"
        )
        self._attr_available = True

        if description.qs_default_option:
            self._attr_current_option = description.qs_default_option

    async def async_select_option(self, option: str) -> None:
        """Select an option."""
        self._attr_current_option = option
        setattr(self.device, self.entity_description.key, option)
        self.async_write_ha_state()

    @callback
    def async_update_callback(self) -> None:
        """Update the entity's state."""
        if self.hass is None:
            return

        #if self.entity_description.value_fn is None:
        if (option := getattr(self.device, self.entity_description.key)) is None:
            return

        self._attr_current_option = option
        self.async_write_ha_state()

class QSBaseSelectRestore(QSBaseSelect, RestoreEntity):
    """Entity to represent VAD sensitivity."""

    def __init__(
        self,
        data_handler,
        device: AbstractDevice,
        description: QSSelectEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(data_handler=data_handler, device=device, description=description)
        self._attr_available = True

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()

        self._attr_available = True

        state = await self.async_get_last_state()
        if state is not None and state.state in self.options:
            await self.async_select_option(state.state)


