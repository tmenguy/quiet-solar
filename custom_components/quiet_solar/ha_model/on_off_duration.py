import logging
from datetime import datetime, timedelta


from .bistate_duration import QSBiStateDuration
from ..const import SENSOR_CONSTRAINT_SENSOR_ON_OFF, CONF_TYPE_NAME_QSOnOffDuration
from ..home_model.commands import LoadCommand, CMD_ON
from homeassistant.const import Platform, SERVICE_TURN_ON, SERVICE_TURN_OFF



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

    def get_virtual_current_constraint_translation_key(self) -> str | None:
        return SENSOR_CONSTRAINT_SENSOR_ON_OFF

    def get_select_translation_key(self) -> str | None:
        """ return the translation key for the select """
        return "on_off_mode"

    # exception catched above execute_command
    async def execute_command_system(self, time: datetime, command:LoadCommand) -> bool | None:
        if command.is_like(CMD_ON):
            action = SERVICE_TURN_ON
        elif command.is_off_or_idle():
            action = SERVICE_TURN_OFF
        else:
            raise ValueError("Invalid command")

        _LOGGER.info(f"Executing on/off command {action} on {self.bistate_entity}")

        # exception catched above execute_command
        await self.hass.services.async_call(
            domain=Platform.SWITCH,
            service=action,
            target={"entity_id": self.bistate_entity}
        )
        return False