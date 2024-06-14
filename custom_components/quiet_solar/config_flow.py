from abc import abstractmethod

from homeassistant import config_entries
from homeassistant.config_entries import (
    SOURCE_REAUTH,
    ConfigEntry,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_NAME, ATTR_UNIT_OF_MEASUREMENT, UnitOfPower, UnitOfElectricCurrent, UnitOfTemperature, UnitOfEnergy, UnitOfElectricPotential,  PERCENTAGE
from homeassistant.core import callback

from entity import LOAD_TYPES
from .const import DOMAIN, DEVICE_TYPE, CONF_GRID_POWER_SENSOR, CONF_GRID_POWER_SENSOR_INVERTED, \
    CONF_INVERTER_ACTIVE_POWER_SENSOR, CONF_INVERTER_INPUT_POWER_SENSOR, \
    CONF_BATTERY_CHARGE_DISCHARGE_SENSOR, CONF_BATTERY_CAPACITY, CONF_CHARGER_MAX_CHARGE, CONF_CHARGER_MIN_CHARGE, \
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, CONF_CHARGER_PAUSE_RESUME_SWITCH, CONF_CAR_CHARGE_PERCENT_SENSOR, \
    CONF_CAR_BATTERY_CAPACITY, CONF_CAR_CHARGER_MIN_CHARGE, CONF_CAR_CHARGER_MAX_CHARGE, CONF_HOME_VOLTAGE, \
    CONF_POOL_TEMPERATURE_SENSOR, CONF_POOL_PUMP_POWER, CONF_POOL_PUMP_SWITCH, CONF_SWITCH_POWER, CONF_SWITCH, CONF_TYPE
from homeassistant.helpers import config_validation as cv, selector
from typing import TYPE_CHECKING
import voluptuous as vol
from homeassistant.core import HomeAssistant



from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.components.number import DOMAIN as NUMBER_DOMAIN
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.components.input_number import DOMAIN as INPUT_NUMBER_DOMAIN




def power_selector_compatible(
    hass: HomeAssistant,
) -> selector.EntitySelector:
    """Return an entity selector which compatible entities."""

    ALLOWED_DOMAINS = [SENSOR_DOMAIN]
    entities = [
        ent.entity_id
        for ent in hass.states.async_all(ALLOWED_DOMAINS)
        if ent.attributes.get(ATTR_UNIT_OF_MEASUREMENT) in UnitOfPower
        and ent.domain in ALLOWED_DOMAINS
    ]

    return selector.EntitySelector(
        selector.EntitySelectorConfig(include_entities=entities)
    )


def temperature_selector_compatible(
    hass: HomeAssistant,
) -> selector.EntitySelector:
    """Return an entity selector which compatible entities."""

    ALLOWED_DOMAINS = [SENSOR_DOMAIN]
    entities = [
        ent.entity_id
        for ent in hass.states.async_all(ALLOWED_DOMAINS)
        if ent.attributes.get(ATTR_UNIT_OF_MEASUREMENT) in UnitOfTemperature
        and ent.domain in ALLOWED_DOMAINS
    ]

    return selector.EntitySelector(
        selector.EntitySelectorConfig(include_entities=entities)
    )

def percent_selector_compatible(
    hass: HomeAssistant,
) -> selector.EntitySelector:
    """Return an entity selector which compatible entities."""

    ALLOWED_DOMAINS = [SENSOR_DOMAIN]
    entities = [
        ent.entity_id
        for ent in hass.states.async_all(ALLOWED_DOMAINS)
        if ent.attributes.get(ATTR_UNIT_OF_MEASUREMENT) == PERCENTAGE
        and ent.domain in ALLOWED_DOMAINS
    ]

    return selector.EntitySelector(
        selector.EntitySelectorConfig(include_entities=entities)
    )




class QSFlowHandlerMixin(config_entries.ConfigEntryBaseFlow if TYPE_CHECKING else object):

    @abstractmethod
    async def async_entry_next(self, data):
        """Handle the next step based on user input."""

    async def async_step_home(self, user_input=None):

        TYPE = "home"
        if user_input is not None:
            #do sotme stuff to update
            user_input[DEVICE_TYPE] = TYPE
            return self.async_entry_next(user_input)

        schema = vol.Schema(
            {
                vol.Required(CONF_NAME): cv.string,

                vol.Optional(CONF_HOME_VOLTAGE, default=self.config_entry.data.get(CONF_HOME_VOLTAGE, 230)):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfElectricPotential.VOLT,
                        )
                    ),

                vol.Optional(CONF_GRID_POWER_SENSOR,
                             default=self.config_entry.data.get(CONF_GRID_POWER_SENSOR)):
                    power_selector_compatible(self.hass),
                vol.Optional(CONF_GRID_POWER_SENSOR_INVERTED, default=self.config_entry.data.get(CONF_GRID_POWER_SENSOR_INVERTED)):
                    cv.boolean,
            }
        )


        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_solar(self, user_input=None):

        TYPE = "solar"
        if user_input is not None:
            #do sotme stuff to update
            user_input[DEVICE_TYPE] = TYPE
            return self.async_entry_next(user_input)

        schema = vol.Schema(
            {
                vol.Required(CONF_NAME): cv.string,

                vol.Optional(CONF_INVERTER_ACTIVE_POWER_SENSOR,
                             default=self.config_entry.data.get(CONF_INVERTER_ACTIVE_POWER_SENSOR)):
                    power_selector_compatible(self.hass),

                vol.Optional(CONF_INVERTER_INPUT_POWER_SENSOR,
                             default=self.config_entry.data.get(CONF_INVERTER_INPUT_POWER_SENSOR)):
                    power_selector_compatible(self.hass),
            }
        )

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_charger(self, user_input=None):

        TYPE = "charger"
        if user_input is not None:
            #do sotme stuff to update
            user_input[DEVICE_TYPE] = TYPE
            return self.async_entry_next(user_input)

        schema = vol.Schema(
            {
                vol.Required(CONF_NAME): cv.string,

                vol.Required(CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER,
                             default=self.config_entry.data.get(CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER)):
                    selector.EntitySelector(
                        selector.EntitySelectorConfig(domain=[NUMBER_DOMAIN])
                    ),

                vol.Optional(CONF_CHARGER_PAUSE_RESUME_SWITCH,
                             default=self.config_entry.data.get(CONF_CHARGER_PAUSE_RESUME_SWITCH)):
                    selector.EntitySelector(
                        selector.EntitySelectorConfig(domain=[SWITCH_DOMAIN])
                    ),

                vol.Optional(CONF_CHARGER_MAX_CHARGE, default=self.config_entry.data.get(CONF_CHARGER_MAX_CHARGE, 32)):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfElectricCurrent.AMPERE,
                        )
                    ),
                vol.Optional(CONF_CHARGER_MIN_CHARGE, default=self.config_entry.data.get(CONF_CHARGER_MIN_CHARGE, 6)):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfElectricCurrent.AMPERE,
                        )
                    ),
            }
        )

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_battery(self, user_input=None):

        TYPE = "battery"
        if user_input is not None:
            #do sotme stuff to update
            user_input[DEVICE_TYPE] = TYPE
            return self.async_entry_next(user_input)

        schema = vol.Schema(
            {
                vol.Required(CONF_NAME): cv.string,

                vol.Optional(CONF_BATTERY_CHARGE_DISCHARGE_SENSOR,
                             default=self.config_entry.data.get(CONF_BATTERY_CHARGE_DISCHARGE_SENSOR)):
                    power_selector_compatible(self.hass),

                vol.Optional(CONF_BATTERY_CAPACITY, default=self.config_entry.data.get(CONF_BATTERY_CAPACITY, 0)):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfEnergy.WATT_HOUR,
                        )
                    ),
            }
        )

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_car(self, user_input=None):

        TYPE = "car"
        if user_input is not None:
            #do sotme stuff to update
            user_input[DEVICE_TYPE] = TYPE
            return self.async_entry_next(user_input)

        schema = vol.Schema(

            {
                vol.Required(CONF_NAME): cv.string,

                vol.Optional(CONF_CAR_CHARGE_PERCENT_SENSOR,
                             default=self.config_entry.data.get(CONF_CAR_CHARGE_PERCENT_SENSOR)):
                    percent_selector_compatible(self.hass),

                vol.Optional(CONF_CAR_BATTERY_CAPACITY, default=self.config_entry.data.get(CONF_CAR_BATTERY_CAPACITY, 0)):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfEnergy.WATT_HOUR,
                        )
                    ),
                vol.Optional(CONF_CAR_CHARGER_MIN_CHARGE, default=self.config_entry.data.get(CONF_CAR_CHARGER_MIN_CHARGE, 6)):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfElectricCurrent.AMPERE,
                        )
                    ),

                vol.Optional(CONF_CAR_CHARGER_MAX_CHARGE, default=self.config_entry.data.get(CONF_CAR_CHARGER_MAX_CHARGE, 32)):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfElectricCurrent.AMPERE,
                        )
                    ),
            }
        )

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_pool(self, user_input=None):

        TYPE = "pool"
        if user_input is not None:
            #do sotme stuff to update
            user_input[DEVICE_TYPE] = TYPE
            return self.async_entry_next(user_input)

        schema = vol.Schema(
            {
                vol.Required(CONF_NAME): cv.string,


                vol.Required(CONF_POOL_TEMPERATURE_SENSOR,
                             default=self.config_entry.data.get(CONF_POOL_TEMPERATURE_SENSOR)):
                    temperature_selector_compatible(self.hass),

                vol.Required(CONF_POOL_PUMP_POWER, default=self.config_entry.data.get(CONF_POOL_PUMP_POWER, 1500)):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfPower.WATT,
                        )
                    ),

                vol.Required(CONF_POOL_PUMP_SWITCH,
                             default=self.config_entry.data.get(CONF_POOL_PUMP_SWITCH)):
                    selector.EntitySelector(
                        selector.EntitySelectorConfig(domain=[SWITCH_DOMAIN])
                    ),

            }
        )

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_switch(self, user_input=None):

        TYPE = "switch"
        if user_input is not None:
            #do sotme stuff to update
            user_input[DEVICE_TYPE] = TYPE
            return self.async_entry_next(user_input)

        schema = vol.Schema(
            {
                vol.Required(CONF_NAME): cv.string,

                vol.Required(CONF_SWITCH_POWER, default=self.config_entry.data.get(CONF_SWITCH_POWER, 1500)):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfPower.WATT,
                        )
                    ),

                vol.Required(CONF_SWITCH,
                             default=self.config_entry.data.get(CONF_SWITCH)):
                    selector.EntitySelector(
                        selector.EntitySelectorConfig(domain=[SWITCH_DOMAIN])
                    ),

            }
        )

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )





class QSFlowHandler(QSFlowHandlerMixin, config_entries.ConfigFlow, domain=DOMAIN):

    VERSION = 1
    MINOR_VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Get the options flow for this handler."""
        return QSOptionsFlowHandler(config_entry)

    async def async_step_user(self, user_input=None):
        """initial step (menu) for user initiated flows"""

        await self.async_set_unique_id(DOMAIN)

        return self.async_show_menu(
            step_id="user",
            menu_options=[t for t in LOAD_TYPES],
        )
    async def async_entry_next(self, data):
        """Handle the next step based on user input."""
        return self.async_create_entry(title=f"{data.get(CONF_NAME, "load")} QS Load", data=data)



class QSOptionsFlowHandler(QSFlowHandlerMixin, OptionsFlow):

    VERSION = 1
    MINOR_VERSION = 1

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize Quiet Solar options flow."""
        self.config_entry = config_entry
        self.options = dict(config_entry.options)
        self._errors = {}


    async def async_entry_next(self, data):
        """Handle the next step based on user input."""
        self.hass.config_entries.async_update_entry(
            self.config_entry, data=data, options=self.config_entry.options
        )
        return self.async_create_entry(title=None, data={})


    async def async_step_init(self, user_input=None) -> ConfigFlowResult:
        """Manage the options."""
        type = self.config_entry.data.get(DEVICE_TYPE)
        if type is None:
            return self.async_create_entry(title=None, data={})

        step_name = f"async_step_{type}"

        return await getattr(self, step_name)( user_input=user_input)

