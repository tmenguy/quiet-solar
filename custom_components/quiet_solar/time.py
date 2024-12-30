import logging
from dataclasses import dataclass
from datetime import time as dt_time

import pytz
from homeassistant.components.time import TimeEntity, TimeEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity, ExtraStoredData


from .ha_model.charger import QSChargerGeneric

from .home_model.load import AbstractDevice
from .const import (
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

from quiet_solar.entity import QSDeviceEntity

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


def create_ha_time(device: AbstractDevice):
    ret = []

    if isinstance(device, QSChargerGeneric):
        ret.extend(create_ha_time_for_QSCharger(device))

    return ret

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Netatmo switch platform."""
    device = hass.data[DOMAIN].get(entry.entry_id)

    if device:

        entities = create_ha_time(device)

        if entities:
            async_add_entities(entities)
    return


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
        self._attr_available = True

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

        await self.async_set_value(val)

    async def async_set_value(self, value: dt_time) -> None:
        """Change the value."""
        self._attr_native_value = value
        try:
            setattr(self.device, self.entity_description.key, value)
        except:
            _LOGGER.info(f"can't set time {value} on {self.device.name} for {self.entity_description.key}")
        self.async_write_ha_state()

