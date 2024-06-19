
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback



from homeassistant.components.calendar import (
    EVENT_END,
    EVENT_RRULE,
    EVENT_START,
    CalendarEntity,
    CalendarEntityFeature,
    CalendarEvent,
)
async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:

    return

    """Set up the local calendar platform."""
    from homeassistant.components.local_calendar.calendar import LocalCalendarEntity, PRODID
    class QSCalendar(LocalCalendarEntity):

        def __init__(self, store, calendar, name, unique_id):
            super().__init__(store, calendar, name, unique_id)
            self._attr_supported_features = (
                    CalendarEntityFeature.CREATE_EVENT
                    | CalendarEntityFeature.DELETE_EVENT
                    | CalendarEntityFeature.UPDATE_EVENT
            )
            self._attr_has_entity_name = True


    store = hass.data[DOMAIN][config_entry.entry_id]
    ics = await store.async_load()
    #from ical.calendar_stream import IcsCalendarStream
    calendar = IcsCalendarStream.calendar_from_ics(ics)
    calendar.prodid = PRODID

    name = config_entry.data[CONF_CALENDAR_NAME]
    entity = LocalCalendarEntity(store, calendar, name, unique_id=config_entry.entry_id)
    async_add_entities([entity], True)



