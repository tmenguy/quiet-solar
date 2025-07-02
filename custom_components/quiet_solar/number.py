import logging
from dataclasses import dataclass


from homeassistant.components.number import NumberEntity, NumberEntityDescription, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE, UnitOfTime
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity, ExtraStoredData

from .ha_model.bistate_duration import QSBiStateDuration
from .home_model.load import AbstractDevice
from .const import (
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

from .entity import QSDeviceEntity

@dataclass(frozen=True, kw_only=True)
class QSNumberEntityDescription(NumberEntityDescription):
    qs_default_option:str | None  = None


def create_ha_number_for_QSBiStateDuration(device: QSBiStateDuration):
    entities = []

    selected_car_description = QSNumberEntityDescription(
        key="default_on_duration",
        translation_key="default_on_duration",
        native_max_value=24,
        native_min_value=0,
        native_step=0.5,
        native_unit_of_measurement=UnitOfTime.HOURS,
        mode=NumberMode.BOX,
    )
    entities.append(QSBaseNumber(data_handler=device.data_handler, device=device, description=selected_car_description))

    return entities

def create_ha_number(device: AbstractDevice):
    ret = []

    if isinstance(device, QSBiStateDuration):
        ret.extend(create_ha_number_for_QSBiStateDuration(device))

    return ret

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:

    device = hass.data[DOMAIN].get(entry.entry_id)

    if device:

        entities = create_ha_number(device)

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

class QSBaseNumber(QSDeviceEntity, NumberEntity, RestoreEntity):
    """Implementation of a Qs DateTime sensor."""

    def __init__(
        self,
        data_handler,
        device: AbstractDevice,
        description: QSNumberEntityDescription,
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
            try:
                val = float(last_state.state)
            except:
                val = None

        if val is None:
            val = float(getattr(self.device, self.entity_description.key, 1.0))

        if val is None:
            val = 1.0

        await self.async_set_native_value(val)

    async def async_set_native_value(self, value: float) -> None:
        """Change the value."""
        self._attr_native_value = value
        try:
            setattr(self.device, self.entity_description.key, value)
        except:
            _LOGGER.info(f"can't set number {value} on {self.device.name} for {self.entity_description.key}")
        self._set_availabiltiy()
        self.async_write_ha_state()