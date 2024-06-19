from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .home_model.commands import LoadCommand
from .entity import QSDeviceEntity

from homeassistant.const import (
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    Platform
)



async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Netatmo switch platform."""

    return
    @callback
    def _create_entity(netatmo_device: NetatmoDevice) -> None:
        #entity = NetatmoSwitch(netatmo_device)
        #_LOGGER.debug("Adding switch %s", entity)
        #async_add_entities([entity])
        pass

    entry.async_on_unload(
        async_dispatcher_connect(hass, NETATMO_CREATE_SWITCH, _create_entity)
    )



# a HaSwitchLoad is a load that is controlled by a switch entity, and it have to expose a switch for direct control to enforce a constraint
class QSSwitchLoad(QSDeviceEntity, SwitchEntity):


    _attr_name = None

    def __init__(self, switch_entity:str, **kwargs):
        super().__init__(**kwargs)
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
