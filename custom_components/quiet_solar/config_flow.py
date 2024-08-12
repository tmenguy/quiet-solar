import logging

from homeassistant.helpers.selector import SelectSelector, SelectSelectorConfig, SelectOptionDict, SelectSelectorMode
from homeassistant.core import callback

from abc import abstractmethod
from dataclasses import dataclass

from homeassistant import config_entries
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_NAME, ATTR_UNIT_OF_MEASUREMENT, UnitOfPower, UnitOfElectricCurrent, UnitOfTemperature, UnitOfEnergy, UnitOfElectricPotential,  PERCENTAGE

from homeassistant.helpers import config_validation as cv, selector
from typing import TYPE_CHECKING
import voluptuous as vol
from homeassistant.core import HomeAssistant

from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.components.number import DOMAIN as NUMBER_DOMAIN
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.components.select import DOMAIN as SELECT_DOMAIN
from homeassistant.components.binary_sensor import DOMAIN as BINARY_SENSOR_DOMAIN
from homeassistant.components.device_tracker import DOMAIN as DEVICE_TRACKER_DOMAIN



from .entity import LOAD_TYPES
from .const import DOMAIN, DEVICE_TYPE, CONF_GRID_POWER_SENSOR, CONF_GRID_POWER_SENSOR_INVERTED, \
    CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR, CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR, \
    CONF_BATTERY_CHARGE_DISCHARGE_SENSOR, CONF_BATTERY_CAPACITY, CONF_CHARGER_MAX_CHARGE, CONF_CHARGER_MIN_CHARGE, \
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, CONF_CHARGER_PAUSE_RESUME_SWITCH, CONF_CAR_CHARGE_PERCENT_SENSOR, \
    CONF_CAR_BATTERY_CAPACITY, CONF_CAR_CHARGER_MIN_CHARGE, CONF_CAR_CHARGER_MAX_CHARGE, CONF_HOME_VOLTAGE, \
    CONF_POOL_TEMPERATURE_SENSOR, CONF_SWITCH, \
    CONF_CAR_PLUGGED, CONF_CHARGER_PLUGGED, CONF_CAR_TRACKER, CONF_CHARGER_DEVICE_OCPP, CONF_CHARGER_DEVICE_WALLBOX, \
    CONF_CHARGER_IS_3P, DATA_HANDLER, CONF_POOL_IS_PUMP_VARIABLE_SPEED, CONF_POWER, CONF_SELECT, \
    CONF_ACCURATE_POWER_SENSOR, \
    CONF_CAR_CUSTOM_POWER_CHARGE_VALUES, CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P, \
    CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER, CONF_BATTERY_MAX_CHARGE_POWER_NUMBER, \
    CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE, CONF_BATTERY_MAX_CHARGE_POWER_VALUE, SOLCAST_SOLAR_DOMAIN, \
    OPEN_METEO_SOLAR_DOMAIN, CONF_SOLAR_FORECAST_PROVIDER

_LOGGER = logging.getLogger(__name__)

def selectable_power_entities(hass: HomeAssistant, domains=None) -> list:
    """Return an entity selector which compatible entities."""

    if domains is None:
        ALLOWED_DOMAINS = [SENSOR_DOMAIN]
    else:
        ALLOWED_DOMAINS = domains
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
                vals_sc = selector.EntitySelector(selector.EntitySelectorConfig(domain=domain))
            else:
                vals_sc = selector.EntitySelector()

        sc_dict[key_sc] = vals_sc

    def get_common_schema(self, add_power_value_selector=None, add_load_power_sensor=False, add_load_power_sensor_mandatory=False) -> dict:

        default = self.config_entry.data.get(CONF_NAME)


        if default is None:
            sc = {
            vol.Required(CONF_NAME): cv.string,
        }
        else:
            sc = {
                vol.Required(CONF_NAME, default=default): cv.string,
            }

        if add_power_value_selector:

            sc.update({vol.Required(CONF_POWER, default=self.config_entry.data.get(CONF_POWER, add_power_value_selector)):
            selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                    unit_of_measurement=UnitOfPower.WATT,
                )
            )})

        if add_load_power_sensor:

            power_entities = selectable_power_entities(self.hass)
            if len(power_entities) > 0:
                self.add_entity_selector(sc, CONF_ACCURATE_POWER_SENSOR, add_load_power_sensor_mandatory, entity_list=power_entities)

        return sc





    def get_all_charger_schema_base(self, add_load_power_sensor_mandatory ):

        sc = self.get_common_schema(add_load_power_sensor=True, add_load_power_sensor_mandatory=add_load_power_sensor_mandatory)

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


    def get_entry_title(self, data):
        return f"{data.get(DEVICE_TYPE, "unknown")}: {data.get(CONF_NAME, "device")}"
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
            r = await self.async_entry_next(user_input)
            return r

        sc_dict = self.get_common_schema()

        power_entities = selectable_power_entities(self.hass)
        if len(power_entities) > 0 :
            self.add_entity_selector(sc_dict, CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR, False, entity_list=power_entities)
            self.add_entity_selector(sc_dict, CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR, False, entity_list=power_entities)


        solcast_entries = self.hass.data.get(SOLCAST_SOLAR_DOMAIN, {})
        options = []
        if solcast_entries:
            options.append(SelectOptionDict(value=SOLCAST_SOLAR_DOMAIN, label="Solcast"))

        open_solar_entries = self.hass.data.get(OPEN_METEO_SOLAR_DOMAIN, {})
        if open_solar_entries:
            options.append(SelectOptionDict(value=OPEN_METEO_SOLAR_DOMAIN, label="Open Meteo Forecast"))

        if options:
            default = self.config_entry.data.get(CONF_SOLAR_FORECAST_PROVIDER)

            if default:
                keysc = vol.Required(CONF_SOLAR_FORECAST_PROVIDER, default=default)
            else:
                keysc = vol.Required(CONF_SOLAR_FORECAST_PROVIDER)

            sc_dict.update({
                keysc: SelectSelector(
                    SelectSelectorConfig(
                        options=options,
                        mode=SelectSelectorMode.LIST,

                    )
                )})

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
            r = await self.async_entry_next(user_input)
            return r

        sc_dict = self.get_all_charger_schema_base(add_load_power_sensor_mandatory=True)
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
            r = await self.async_entry_next(user_input)
            return r

        sc_dict = self.get_all_charger_schema_base(add_load_power_sensor_mandatory=False)

        default = self.config_entry.data.get(CONF_CHARGER_DEVICE_OCPP)
        if default is None:
            key_sc = vol.Required(CONF_CHARGER_DEVICE_OCPP)
        else:
            key_sc = vol.Required(CONF_CHARGER_DEVICE_OCPP, default=default)

        sc_dict[key_sc] = selector.DeviceSelector(
                            selector.DeviceSelectorConfig(
                                integration="OCPP",
                                entity=selector.EntityFilterSelectorConfig(
                                    domain="sensor")
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
            r = await self.async_entry_next(user_input)
            return r
        sc_dict = self.get_all_charger_schema_base(add_load_power_sensor_mandatory=False)

        default = self.config_entry.data.get(CONF_CHARGER_DEVICE_WALLBOX)
        if default is None:
            key_sc = vol.Required(CONF_CHARGER_DEVICE_WALLBOX)
        else:
            key_sc = vol.Required(CONF_CHARGER_DEVICE_WALLBOX, default=default)

        sc_dict[key_sc] = selector.DeviceSelector(
                            selector.DeviceSelectorConfig(
                                integration="wallbox"
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

        number_entites = selectable_power_entities(self.hass, domains=[NUMBER_DOMAIN])
        if len(number_entites) > 0:
            self.add_entity_selector(sc_dict, CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER, False, entity_list=number_entites)
            self.add_entity_selector(sc_dict, CONF_BATTERY_MAX_CHARGE_POWER_NUMBER, False,
                                     entity_list=number_entites)

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
                vol.Optional(CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE, default=self.config_entry.data.get(CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE, 0)):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfPower.WATT,
                        )
                    ),
                vol.Optional(CONF_BATTERY_MAX_CHARGE_POWER_VALUE,
                             default=self.config_entry.data.get(CONF_BATTERY_MAX_CHARGE_POWER_VALUE, 0)):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfPower.WATT,
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

        orig_dampening = self.config_entry.data.get(CONF_CAR_CUSTOM_POWER_CHARGE_VALUES, False)

        TYPE = "car"
        do_force_dampening = False

        min_charge = self.config_entry.data.get(CONF_CAR_CHARGER_MIN_CHARGE, 6)
        max_charge = self.config_entry.data.get(CONF_CAR_CHARGER_MAX_CHARGE, 32)

        if user_input is not None:

            if "force_dampening" in user_input:
                do_force_dampening = True
            else:
                #do some stuff to update
                new_dampeneing = user_input.get(CONF_CAR_CUSTOM_POWER_CHARGE_VALUES, False)
                user_input[DEVICE_TYPE] = TYPE

                if new_dampeneing is True and (user_input.get(CONF_CAR_CHARGER_MIN_CHARGE, 0) != min_charge or user_input.get(CONF_CAR_CHARGER_MAX_CHARGE, 0) != max_charge):
                    #force dampening
                    orig_dampening = False

                if orig_dampening is False and new_dampeneing is True:

                    #self.hass.config_entries.async_update_entry(
                    #    self.config_entry, data=user_input, options=self.config_entry.options
                    #)
                    # or more simply:
                    self.config_entry.data = user_input

                    return await self.async_step_car({"force_dampening": True})


                r =  await self.async_entry_next(user_input)
                return r




        sc_dict = self.get_common_schema()

        self.add_entity_selector(sc_dict, CONF_CAR_PLUGGED, False, domain=[BINARY_SENSOR_DOMAIN])
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

                vol.Optional(CONF_CAR_CUSTOM_POWER_CHARGE_VALUES,
                             default=self.config_entry.data.get(CONF_CAR_CUSTOM_POWER_CHARGE_VALUES, False)):
                    cv.boolean,
            }
        )

        if do_force_dampening is True or orig_dampening is True:

            sc_dict[vol.Optional(CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P,
                                 default=self.config_entry.data.get(CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P, False))] = cv.boolean

            for a in range(int(min_charge), int(max_charge) + 1):

                sc_dict[vol.Optional(f"charge_{a}",
                     default=self.config_entry.data.get(f"charge_{a}", -1))] = selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=-1,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfPower.WATT,
                        )
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
            r = await self.async_entry_next(user_input)
            return r

        sc_dict = self.get_common_schema(add_power_value_selector=1500, add_load_power_sensor=True)

        temperature_entities = selectable_temperature_entities(self.hass)
        if len(temperature_entities) > 0 :
            self.add_entity_selector(sc_dict, CONF_POOL_TEMPERATURE_SENSOR, True, entity_list=temperature_entities)

        self.add_entity_selector(sc_dict, CONF_SWITCH, True, domain=[SWITCH_DOMAIN])

        sc_dict.update(
            {
                vol.Optional(CONF_POOL_IS_PUMP_VARIABLE_SPEED,
                             default=self.config_entry.data.get(CONF_POOL_IS_PUMP_VARIABLE_SPEED, False)):
                    cv.boolean,
            }
        )

        schema = vol.Schema(sc_dict)

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_on_off_duration(self, user_input=None):

        TYPE = "on_off_duration"
        if user_input is not None:
            #do sotme stuff to update
            user_input[DEVICE_TYPE] = TYPE
            r = await self.async_entry_next(user_input)
            return r

        sc_dict = self.get_common_schema(add_power_value_selector=1000, add_load_power_sensor=True)

        self.add_entity_selector(sc_dict, CONF_SWITCH, True, domain=[SWITCH_DOMAIN])

        schema = vol.Schema(sc_dict)

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_fp_heater(self, user_input=None):

        TYPE = "fp_heater"
        if user_input is not None:
            #do sotme stuff to update
            user_input[DEVICE_TYPE] = TYPE
            r = await self.async_entry_next(user_input)
            return r

        sc_dict = self.get_common_schema(add_power_value_selector=1000, add_load_power_sensor=True)

        self.add_entity_selector(sc_dict, CONF_SWITCH, True, domain=[SWITCH_DOMAIN])

        self.add_entity_selector(sc_dict, CONF_SELECT, True, domain=[SELECT_DOMAIN])

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
    config_entry: FakeConfigEntry | None | ConfigEntry = FakeConfigEntry()

    @staticmethod
    @callback
    def async_get_options_flow(
         config_entry: ConfigEntry,
     ) -> OptionsFlow:
         """Get the options flow for this handler."""
         return QSOptionsFlowHandler(config_entry)

    async def async_step_user(self, user_input=None):
        """initial step (menu) for user initiated flows"""

        possible_menus = ["home"]
        data_handler = self.hass.data.get(DOMAIN, {}). get(DATA_HANDLER)
        if data_handler is not None:
            if data_handler.home is not None:
                possible_menus = [t for t in LOAD_TYPES if t != "home"]
                if data_handler.home._battery is not None:
                    possible_menus.remove("battery")
                if data_handler.home._solar_plant is not None:
                    possible_menus.remove("solar")



        return self.async_show_menu(
            step_id="user",
            menu_options=possible_menus,
        )

    # async def async_step_reconfigure(self, user_input: dict | None = None):
    #    """Step when user reconfigures the integration."""
    #
    #    self.config_entry = self.hass.config_entries.async_get_entry(
    #        self.context["entry_id"]
    #    )
    #    if self.config_entry is None:
    #        return self.async_create_entry(title="", data={})
    #    type = self.config_entry.data.get(DEVICE_TYPE)
    #    if type is None:
    #        return self.async_create_entry(title="", data={})
    #
    #
    #    step_name = f"async_step_{type}"
    #
    #    return await getattr(self, step_name)( user_input=user_input)

    async def async_step_charger(self, user_input=None):

        return self.async_show_menu(
            step_id="charger",
            menu_options=[t for t in LOAD_TYPES["charger"]],
        )

    async def async_entry_next(self, data):
        """Handle the next step based on user input."""
        u_id = f"Quiet Solar: {data.get(CONF_NAME, "device")} {data.get(DEVICE_TYPE, "unknown")}"
        await self.async_set_unique_id(u_id)
        return self.async_create_entry(title=self.get_entry_title(data), data=data)



class QSOptionsFlowHandler(QSFlowHandlerMixin, OptionsFlow):

    VERSION = 1
    MINOR_VERSION = 1

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize Quiet Solar options flow."""
        super().__init__()
        self.config_entry = config_entry
        self.options = dict(config_entry.options)
        self._errors = {}


    async def async_entry_next(self, data):
        """Handle the next step based on user input."""
        self.hass.config_entries.async_update_entry(
            self.config_entry, data=data, options=self.config_entry.options, title=self.get_entry_title(data)
        )
        return self.async_create_entry(title=None, data={})

    async def async_step_init(self, user_input=None) -> ConfigFlowResult:
        """Manage the options."""

        type = self.config_entry.data.get(DEVICE_TYPE)
        if type is None:
            return self.async_create_entry(title=None, data={})


        step_name = f"async_step_{type}"

        return await getattr(self, step_name)( user_input=user_input)

