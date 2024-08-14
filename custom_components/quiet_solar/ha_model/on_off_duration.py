from datetime import datetime

from ..ha_model.device import HADeviceMixin
from ..home_model.commands import LoadCommand, CMD_ON, CMD_OFF, CMD_IDLE
from ..home_model.load import AbstractLoad
from homeassistant.const import Platform, SERVICE_TURN_ON, SERVICE_TURN_OFF, STATE_UNKNOWN, STATE_UNAVAILABLE


class QSOnOffDuration(HADeviceMixin, AbstractLoad):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_platforms(self):
        return [ Platform.SENSOR, Platform.SWITCH ]

    async def execute_command(self, time: datetime, command:LoadCommand):
        if command == CMD_ON:
            action = SERVICE_TURN_ON
        elif command == CMD_OFF or command == CMD_IDLE:
            action = SERVICE_TURN_OFF
        else:
            raise ValueError("Invalid command")

        await self.hass.services.async_call(
            domain=Platform.SWITCH,
            service=action,
            target={"entity_id": self.switch_entity},
        )

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool:
        """ check the states of the switch to see if the command is set """
        state = self.hass.states.get(self.switch_entity) # may be faster to get the python entity object no?

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return False

        if command == CMD_ON:
            return state.state == "on"
        else:
            return state.state == "off"

