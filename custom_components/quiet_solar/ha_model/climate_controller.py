import logging
from datetime import datetime

from homeassistant.components.climate import HVACMode

from ..const import (
    CONF_CLIMATE,
    CONF_CLIMATE_HVAC_MODE_OFF,
    CONF_CLIMATE_HVAC_MODE_ON,
    SENSOR_CONSTRAINT_SENSOR_CLIMATE,
    CONF_TYPE_NAME_QSClimateDuration,
)
from ..home_model.commands import LoadCommand
from .bistate_duration import QSBiStateDuration
from .bistate_transport import ClimateTransport, get_hvac_modes

# Re-exported for backwards compatibility — callers that previously
# imported `get_hvac_modes` from this module continue to work. The
# canonical home for the helper is now `bistate_transport.py`.
__all__ = ["QSClimateDuration", "get_hvac_modes"]


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

        self._transport: ClimateTransport = ClimateTransport(self.climate_entity, self._state_on, self._state_off)

    @property
    def climate_state_on(self):
        return self._state_on

    @climate_state_on.setter
    def climate_state_on(self, value):
        self._state_on = value
        self._bistate_mode_on = value
        self._transport.state_on = value

    @property
    def climate_state_off(self):
        return self._state_off

    @climate_state_off.setter
    def climate_state_off(self, value):
        self._state_off = value
        self._bistate_mode_off = value
        self._transport.state_off = value

    def get_possibles_modes(self):
        """return the possible modes for the climate entity"""
        return self._transport.mode_options(self.hass)

    def get_virtual_current_constraint_translation_key(self) -> str | None:
        return SENSOR_CONSTRAINT_SENSOR_CLIMATE

    def get_select_translation_key(self) -> str | None:
        """return the translation key for the select"""
        return "climate_mode"

    # exception catched above execute_command
    async def execute_command_system(self, time: datetime, command: LoadCommand, state: str | None) -> bool | None:
        return await self._transport.execute(
            self.hass, command, state, self._transport.state_on, self._transport.state_off
        )
