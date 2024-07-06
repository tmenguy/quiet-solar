from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from .const import (
    DATA_DEVICE_IDS,
    DOMAIN,
    DATA_HANDLER, DEVICE_TYPE
)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Netatmo switch platform."""
    device = hass.data[DOMAIN][entry.entry_id]
    data_handler = hass.data[DOMAIN][DATA_HANDLER]

    #home is not necessarily set already ... but depending on the device we will creat e various sensors....


    return