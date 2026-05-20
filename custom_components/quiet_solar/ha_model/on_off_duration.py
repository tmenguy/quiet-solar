import logging
from datetime import datetime

from ..const import SENSOR_CONSTRAINT_SENSOR_ON_OFF, CONF_TYPE_NAME_QSOnOffDuration
from ..home_model.commands import LoadCommand
from .bistate_duration import QSBiStateDuration
from .bistate_transport import SwitchTransport

_LOGGER = logging.getLogger(__name__)


class QSOnOffDuration(QSBiStateDuration):
    conf_type_name = CONF_TYPE_NAME_QSOnOffDuration

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._state_on = "on"
        self._state_off = "off"
        self._bistate_mode_on = "on_off_mode_on"
        self._bistate_mode_off = "on_off_mode_off"
        self.bistate_entity = self.switch_entity
        self._transport: SwitchTransport = SwitchTransport(self.switch_entity)

    def get_virtual_current_constraint_translation_key(self) -> str | None:
        return SENSOR_CONSTRAINT_SENSOR_ON_OFF

    def get_select_translation_key(self) -> str | None:
        """return the translation key for the select"""
        return "on_off_mode"

    # exception catched above execute_command
    async def execute_command_system(self, time: datetime, command: LoadCommand, state: str | None) -> bool | None:
        return await self._transport.execute(self.hass, command, state, self._state_on, self._state_off)
