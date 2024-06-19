
from abc import abstractmethod
from dataclasses import dataclass

from homeassistant import config_entries
from homeassistant.config_entries import (
    SOURCE_REAUTH,
    ConfigEntry,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_NAME, ATTR_UNIT_OF_MEASUREMENT, UnitOfPower, UnitOfElectricCurrent, UnitOfTemperature, UnitOfEnergy, UnitOfElectricPotential,  PERCENTAGE
from homeassistant.core import callback

from .entity import LOAD_TYPES
from .const import DOMAIN, DEVICE_TYPE, CONF_GRID_POWER_SENSOR, CONF_GRID_POWER_SENSOR_INVERTED, \
    CONF_INVERTER_ACTIVE_POWER_SENSOR, CONF_INVERTER_INPUT_POWER_SENSOR, \
    CONF_BATTERY_CHARGE_DISCHARGE_SENSOR, CONF_BATTERY_CAPACITY, CONF_CHARGER_MAX_CHARGE, CONF_CHARGER_MIN_CHARGE, \
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, CONF_CHARGER_PAUSE_RESUME_SWITCH, CONF_CAR_CHARGE_PERCENT_SENSOR, \
    CONF_CAR_BATTERY_CAPACITY, CONF_CAR_CHARGER_MIN_CHARGE, CONF_CAR_CHARGER_MAX_CHARGE, CONF_HOME_VOLTAGE, \
    CONF_POOL_TEMPERATURE_SENSOR, CONF_POOL_PUMP_POWER, CONF_POOL_PUMP_SWITCH, CONF_SWITCH_POWER, CONF_SWITCH, \
    CONF_CAR_PLUGGED, CONF_CHARGER_PLUGGED, CONF_CAR_TRACKER, CONF_CHARGER_DEVICE_OCPP, CONF_CHARGER_DEVICE_WALLBOX, \
    CONF_CHARGER_IS_3P
from homeassistant.helpers import config_validation as cv, selector
from typing import TYPE_CHECKING
import voluptuous as vol
from homeassistant.core import HomeAssistant



from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN, SensorDeviceClass
from homeassistant.components.number import DOMAIN as NUMBER_DOMAIN
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.components.input_number import DOMAIN as INPUT_NUMBER_DOMAIN
from homeassistant.components.binary_sensor import DOMAIN as BINARY_SENSOR_DOMAIN
from homeassistant.components.device_tracker import DOMAIN as DEVICE_TRACKER_DOMAIN



def selectable_power_entities(hass: HomeAssistant) -> list:
    """Return an entity selector which compatible entities."""

    ALLOWED_DOMAINS = [SENSOR_DOMAIN]
    entities = [
        ent.entity_id
        for ent in hass.states.async_all(ALLOWED_DOMAINS)
        if ent.attributes.get(ATTR_UNIT_OF_MEASUREMENT) in UnitOfPower
        and ent.domain in ALLOWED_DOMAINS
    ]
    return entities

def selectable_temperature_entities(
    hass: HomeAssistant,
) -> list:
    """Return an entity selector which compatible entities."""

    ALLOWED_DOMAINS = [SENSOR_DOMAIN]
    entities = [
        ent.entity_id
        for ent in hass.states.async_all(ALLOWED_DOMAINS)
        if ent.attributes.get(ATTR_UNIT_OF_MEASUREMENT) in UnitOfTemperature
        and ent.domain in ALLOWED_DOMAINS
    ]
    return entities

def selectable_percent_entities(
    hass: HomeAssistant,
) -> list:
    """Return an entity selector which compatible entities."""

    ALLOWED_DOMAINS = [SENSOR_DOMAIN]
    entities = [
        ent.entity_id
        for ent in hass.states.async_all(ALLOWED_DOMAINS)
        if ent.attributes.get(ATTR_UNIT_OF_MEASUREMENT) == PERCENTAGE
        and ent.domain in ALLOWED_DOMAINS
    ]
    return entities




class QSFlowHandlerMixin(config_entries.ConfigEntryBaseFlow if TYPE_CHECKING else object):


    def add_entity_selector(self, sc_dict, key, is_required, entity_list=None, domain=None):

        default = self.config_entry.data.get(key)

        if default is None:
            if is_required:
                key_sc = vol.Required(key)
            else:
                key_sc = vol.Optional(key)
        else:
            if is_required:
                key_sc = vol.Required(key, default=default)
            else:
                key_sc = vol.Optional(key, default=default)

        if entity_list:
            vals_sc = selector.EntitySelector(
                selector.EntitySelectorConfig(include_entities=entity_list)
            )
        else:
            if domain:
                vals_sc = selector.EntitySelector(domain=domain)
            else:
                vals_sc = selector.EntitySelector()

        sc_dict[key_sc] = vals_sc

    def get_common_schema(self) -> dict:

        default = self.config_entry.data.get(CONF_NAME)
        if default is None:
            return {
            vol.Required(CONF_NAME): cv.string,
        }
        return {
            vol.Required(CONF_NAME, default=default): cv.string,
        }

    def get_all_charger_schema_base(self):

        sc = self.get_common_schema()

        sc.update( {
                    vol.Required(CONF_NAME): cv.string,

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
                    vol.Optional(CONF_CHARGER_IS_3P,
                                 default=self.config_entry.data.get(CONF_CHARGER_IS_3P, False)):
                        cv.boolean,

                })
        return sc


    @abstractmethod
    async def async_entry_next(self, data):
        """Handle the next step based on user input."""

    async def async_step_home(self, user_input=None):

        TYPE = "home"
        if user_input is not None:
            #do sotme stuff to update
            user_input[DEVICE_TYPE] = TYPE
            r = await self.async_entry_next(user_input)
            return r

        sc_dict = self.get_common_schema()

        sc_dict.update(  {

                vol.Optional(CONF_HOME_VOLTAGE, default=self.config_entry.data.get(CONF_HOME_VOLTAGE, 230)):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfElectricPotential.VOLT,
                        )
                    )
                }
        )

        power_entities = selectable_power_entities(self.hass)
        if len(power_entities) > 0 :
            self.add_entity_selector(sc_dict, CONF_GRID_POWER_SENSOR, False, entity_list=power_entities)
            sc_dict.update({

                vol.Optional(CONF_GRID_POWER_SENSOR_INVERTED, default=self.config_entry.data.get(CONF_GRID_POWER_SENSOR_INVERTED, False)):
                    cv.boolean,
            }
            )

        schema = vol.Schema(sc_dict)

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

        sc_dict = self.get_common_schema()

        power_entities = selectable_power_entities(self.hass)
        if len(power_entities) > 0 :
            self.add_entity_selector(sc_dict, CONF_INVERTER_ACTIVE_POWER_SENSOR, False, entity_list=power_entities)
            self.add_entity_selector(sc_dict, CONF_INVERTER_INPUT_POWER_SENSOR, False, entity_list=power_entities)

        schema = vol.Schema(sc_dict)

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_charger_generic(self, user_input=None):

        TYPE = "charger_generic"
        if user_input is not None:
            #do sotme stuff to update
            user_input[DEVICE_TYPE] = TYPE
            return self.async_entry_next(user_input)

        sc_dict = self.get_all_charger_schema_base()
        self.add_entity_selector(sc_dict, CONF_CHARGER_PLUGGED, True, domain=[BINARY_SENSOR_DOMAIN])
        self.add_entity_selector(sc_dict, CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, True, domain=[NUMBER_DOMAIN])
        self.add_entity_selector(sc_dict, CONF_CHARGER_PAUSE_RESUME_SWITCH, True, domain=[SWITCH_DOMAIN])

        schema = vol.Schema(sc_dict)

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_charger_ocpp(self, user_input=None):

        TYPE = "charger_ocpp"
        if user_input is not None:
            #do sotme stuff to update
            user_input[DEVICE_TYPE] = TYPE
            return self.async_entry_next(user_input)

        sc_dict = self.get_all_charger_schema_base()

        default = self.config_entry.data.get(CONF_CHARGER_DEVICE_OCPP)
        if default is None:
            key_sc = vol.Required(CONF_CHARGER_DEVICE_OCPP)
        else:
            key_sc = vol.Required(CONF_CHARGER_DEVICE_OCPP, default=default)

        sc_dict[key_sc] = selector.DeviceSelector(
                            selector.DeviceSelectorConfig(
                                entity=selector.EntityFilterSelectorConfig(
                                    domain="ocpp",
                                    device_class=SensorDeviceClass.ENERGY)
                                )
                            )
        schema = vol.Schema(sc_dict)

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_charger_wallbox(self, user_input=None):

        TYPE = "charger_wallbox"
        if user_input is not None:
            #do sotme stuff to update
            user_input[DEVICE_TYPE] = TYPE
            return self.async_entry_next(user_input)

        sc_dict = self.get_all_charger_schema_base()

        default = self.config_entry.data.get(CONF_CHARGER_DEVICE_WALLBOX)
        if default is None:
            key_sc = vol.Required(CONF_CHARGER_DEVICE_WALLBOX)
        else:
            key_sc = vol.Required(CONF_CHARGER_DEVICE_WALLBOX, default=default)

        sc_dict[key_sc] = selector.DeviceSelector(
                            selector.DeviceSelectorConfig(
                                entity=selector.EntityFilterSelectorConfig(
                                    domain="wallbox",
                                    device_class=SensorDeviceClass.ENERGY)
                                )
                            )

        schema = vol.Schema(sc_dict)

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )
    async def async_step_battery(self, user_input=None):

        TYPE = "battery"
        if user_input is not None:
            #do sotme stuff to update
            user_input[DEVICE_TYPE] = TYPE
            r = await self.async_entry_next(user_input)
            return r

        sc_dict = self.get_common_schema()

        power_entities = selectable_power_entities(self.hass)
        if len(power_entities) > 0 :
            self.add_entity_selector(sc_dict, CONF_BATTERY_CHARGE_DISCHARGE_SENSOR, False, entity_list=power_entities)


        sc_dict.update(
            {

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

        schema = vol.Schema(sc_dict)

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

        sc_dict = self.get_common_schema()

        self.add_entity_selector(sc_dict, CONF_CAR_PLUGGED, True, domain=[BINARY_SENSOR_DOMAIN])
        self.add_entity_selector(sc_dict, CONF_CAR_TRACKER, False, domain=[DEVICE_TRACKER_DOMAIN])

        percent_entities = selectable_percent_entities(self.hass)
        if len(percent_entities) > 0 :
            self.add_entity_selector(sc_dict, CONF_CAR_CHARGE_PERCENT_SENSOR, False, entity_list=percent_entities)

        sc_dict.update(

            {
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

        schema = vol.Schema(sc_dict)

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

        sc_dict = self.get_common_schema()

        temperature_entities = selectable_temperature_entities(self.hass)
        if len(temperature_entities) > 0 :
            self.add_entity_selector(sc_dict, CONF_POOL_TEMPERATURE_SENSOR, True, entity_list=temperature_entities)

        self.add_entity_selector(sc_dict, CONF_POOL_PUMP_SWITCH, True, domain=[SWITCH_DOMAIN])

        sc_dict.update(
            {
                vol.Required(CONF_POOL_PUMP_POWER, default=self.config_entry.data.get(CONF_POOL_PUMP_POWER, 1500)):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfPower.WATT,
                        )
                    )
            }
        )

        schema = vol.Schema(sc_dict)

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_on_off_duration(self, user_input=None):

        TYPE = "switch"
        if user_input is not None:
            #do sotme stuff to update
            user_input[DEVICE_TYPE] = TYPE
            return self.async_entry_next(user_input)

        sc_dict = self.get_common_schema()

        self.add_entity_selector(sc_dict, CONF_SWITCH, True, domain=[SWITCH_DOMAIN])

        sc_dict.update(
            {
                vol.Required(CONF_SWITCH_POWER, default=self.config_entry.data.get(CONF_SWITCH_POWER, 1500)):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfPower.WATT,
                        )
                    )
            }
        )

        schema = vol.Schema(sc_dict)

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

@dataclass
class FakeConfigEntry:
    data = {}
    options = {}

class QSFlowHandler(QSFlowHandlerMixin, config_entries.ConfigFlow, domain=DOMAIN):

    VERSION = 1
    MINOR_VERSION = 1
    config_entry: FakeConfigEntry = FakeConfigEntry()

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

    async def async_step_charger(self, user_input=None):

        return self.async_show_menu(
            step_id="user",
            menu_options=[t for t in LOAD_TYPES["charger"]],
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

