from datetime import datetime

from quiet_solar.ha_model.device import HADeviceMixin
from quiet_solar.home_model.commands import LoadCommand, CMD_ON
from quiet_solar.home_model.load import AbstractLoad
from homeassistant.const import Platform, SERVICE_TURN_ON, SERVICE_TURN_OFF, STATE_UNKNOWN, STATE_UNAVAILABLE


class QSOnOffDuration(HADeviceMixin, AbstractLoad):

    def __init__(self, switch:str, switch_power:float, **kwargs):
        super().__init__(**kwargs)
        self._switch_entity = switch
        self._switch_power = switch_power

    def get_platforms(self):
        return [ Platform.SENSOR, Platform.SWITCH ]

    async def execute_command(self, time: datetime, command:LoadCommand):
        if command.command == "on":
            action = SERVICE_TURN_ON
        elif command.command == "off":
            action = SERVICE_TURN_OFF
        else:
            raise ValueError("Invalid command")

        await self.hass.services.async_call(
            domain=Platform.SWITCH,
            service=action,
            target={"entity_id": self._switch_entity},
        )

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool:
        """ check the states of the switch to see if the command is set """
        state = self.hass.states.get(self._switch_entity) # may be faster to get the python entity object no?

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return False

        return state.state == "on" if command == CMD_ON else state.state == "off"

