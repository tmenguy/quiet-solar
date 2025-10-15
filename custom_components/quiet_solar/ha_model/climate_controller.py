import logging
from datetime import datetime
from typing import Any

from homeassistant.components import climate
from homeassistant.components.climate import HVACMode

from .bistate_duration import QSBiStateDuration
from ..const import SENSOR_CONSTRAINT_SENSOR_CLIMATE, CONF_CLIMATE_HVAC_MODE_ON, CONF_CLIMATE_HVAC_MODE_OFF, \
    CONF_CLIMATE, CONF_TYPE_NAME_QSClimateDuration

from ..home_model.commands import LoadCommand, CMD_ON
from homeassistant.const import ATTR_ENTITY_ID


from homeassistant.helpers import entity_registry as er


def get_hvac_modes(hass, entity_id):
    registry = er.async_get(hass)
    entry = registry.async_get(entity_id)
    return entry.capabilities.get("hvac_modes", [HVACMode.AUTO.value, HVACMode.OFF.value])

_LOGGER = logging.getLogger(__name__)
class QSClimateDuration(QSBiStateDuration):

    conf_type_name = CONF_TYPE_NAME_QSClimateDuration

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.climate_entity = kwargs.pop(CONF_CLIMATE, None)
        self._state_off = kwargs.pop(CONF_CLIMATE_HVAC_MODE_OFF, str(HVACMode.OFF.value))
        self._state_on = kwargs.pop(CONF_CLIMATE_HVAC_MODE_ON, str(HVACMode.AUTO.value))

        # get the HVAC mode for on / off for the climate entity
        self._bistate_mode_on = self._state_on
        self._bistate_mode_off = self._state_off
        self.bistate_entity = self.climate_entity
        self.is_load_time_sensitive = True

    def get_virtual_current_constraint_translation_key(self) -> str | None:
        return SENSOR_CONSTRAINT_SENSOR_CLIMATE

    def get_select_translation_key(self) -> str | None:
        """ return the translation key for the select """
        return "climate_mode"

    async def execute_command_system(self, time: datetime, command:LoadCommand) -> bool | None:

        if command.is_like(CMD_ON):
            hvac_mode = self._bistate_mode_on
        elif command.is_off_or_idle():
            hvac_mode = self._bistate_mode_off
        else:
            raise ValueError("Invalid command")

        data: dict[str, Any] = {ATTR_ENTITY_ID: self.bistate_entity}
        service = climate.SERVICE_SET_HVAC_MODE

        data[climate.ATTR_HVAC_MODE] =hvac_mode
        domain = climate.DOMAIN

        await self.hass.services.async_call(
            domain, service, data
        )

        return False