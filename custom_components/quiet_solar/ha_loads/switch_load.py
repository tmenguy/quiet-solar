from home_model.commands import LoadCommand
from ha_loads.ha_load import HALoad

from homeassistant.const import (
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    Platform
)

class OnOffLoad(HALoad):

    def __init__(self, hass, name:str, power:float, switch_entity:str):
        super().__init__(hass, name)
        self._power = power
        self._switch_entity = switch_entity

    async def execute_command(self, command:LoadCommand):
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
