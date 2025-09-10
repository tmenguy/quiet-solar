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

from awesomeversion import AwesomeVersion
from homeassistant.const import __version__ as HAVERSION, STATE_UNKNOWN, STATE_UNAVAILABLE

from homeassistant.const import CONF_NAME, ATTR_UNIT_OF_MEASUREMENT, UnitOfPower, UnitOfElectricCurrent, \
    UnitOfTemperature, UnitOfEnergy, UnitOfElectricPotential, PERCENTAGE, UnitOfTime

from homeassistant.helpers import config_validation as cv, selector
from typing import TYPE_CHECKING
import voluptuous as vol
from homeassistant.core import HomeAssistant

from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.components.number import DOMAIN as NUMBER_DOMAIN
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.components.climate import DOMAIN as CLIMATE_DOMAIN, HVACMode

from homeassistant.components.binary_sensor import DOMAIN as BINARY_SENSOR_DOMAIN
from homeassistant.components.device_tracker import DOMAIN as DEVICE_TRACKER_DOMAIN
from homeassistant.components.calendar import DOMAIN as CALENDAR_DOMAIN
from homeassistant.components.notify import DOMAIN as NOTIFY_DOMAIN

from . import async_reload_quiet_solar
from .entity import LOAD_NAMES
from .const import DOMAIN, DEVICE_TYPE, CONF_GRID_POWER_SENSOR, CONF_GRID_POWER_SENSOR_INVERTED, \
    CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR, CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR, \
    CONF_BATTERY_CHARGE_DISCHARGE_SENSOR, CONF_BATTERY_CAPACITY, CONF_CHARGER_MAX_CHARGE, CONF_CHARGER_MIN_CHARGE, \
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, CONF_CHARGER_PAUSE_RESUME_SWITCH, CONF_CAR_CHARGE_PERCENT_SENSOR, \
    CONF_CAR_CHARGE_PERCENT_MAX_NUMBER, \
    CONF_CAR_BATTERY_CAPACITY, CONF_CAR_CHARGER_MIN_CHARGE, CONF_CAR_CHARGER_MAX_CHARGE, CONF_HOME_VOLTAGE, \
    CONF_POOL_TEMPERATURE_SENSOR, CONF_SWITCH, \
    CONF_CAR_PLUGGED, CONF_CHARGER_PLUGGED, CONF_CAR_TRACKER, CONF_CHARGER_DEVICE_OCPP, CONF_CHARGER_DEVICE_WALLBOX, \
    CONF_IS_3P, DATA_HANDLER, CONF_POOL_IS_PUMP_VARIABLE_SPEED, CONF_POWER, CONF_SELECT, \
    CONF_ACCURATE_POWER_SENSOR, CONF_NUM_MAX_ON_OFF, \
    CONF_CAR_CUSTOM_POWER_CHARGE_VALUES, CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P, \
    CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER, CONF_BATTERY_MAX_CHARGE_POWER_NUMBER, \
    CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE, CONF_BATTERY_MAX_CHARGE_POWER_VALUE, SOLCAST_SOLAR_DOMAIN, \
    OPEN_METEO_SOLAR_DOMAIN, CONF_SOLAR_FORECAST_PROVIDER, CONF_BATTERY_CHARGE_PERCENT_SENSOR, CONF_CALENDAR, \
    CONF_DEFAULT_CAR_CHARGE, CONF_HOME_START_OFF_PEAK_RANGE_1, CONF_HOME_END_OFF_PEAK_RANGE_1, \
    CONF_HOME_START_OFF_PEAK_RANGE_2, CONF_HOME_END_OFF_PEAK_RANGE_2, CONF_HOME_PEAK_PRICE, CONF_HOME_OFF_PEAK_PRICE, \
    CONF_LOAD_IS_BOOST_ONLY, CONF_CAR_IS_INVITED, POOL_TEMP_STEPS, CONF_MOBILE_APP, CONF_MOBILE_APP_NOTHING, \
    CONF_MOBILE_APP_URL, CONF_DEVICE_EFFICIENCY, CONF_CHARGER_LATITUDE, CONF_CHARGER_LONGITUDE, \
    CONF_BATTERY_MIN_CHARGE_PERCENT, CONF_BATTERY_MAX_CHARGE_PERCENT, CONF_BATTERY_CHARGE_FROM_GRID_SWITCH, \
    CONF_DYN_GROUP_MAX_PHASE_AMPS, CONF_DEVICE_DYNAMIC_GROUP_NAME, CONF_CLIMATE, CONF_CLIMATE_HVAC_MODE_OFF, \
    CONF_CLIMATE_HVAC_MODE_ON, CONF_PHASE_1_AMPS_SENSOR, CONF_PHASE_2_AMPS_SENSOR, CONF_PHASE_3_AMPS_SENSOR, \
    CONF_CHARGER_THREE_TO_ONE_PHASE_SWITCH, CONF_MONO_PHASE, CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS, \
    CONF_MINIMUM_OK_CAR_CHARGE, DASHBOARD_NUM_SECTION_MAX, CONF_DASHBOARD_SECTION_NAME, CONF_DASHBOARD_SECTION_ICON, \
    DASHBOARD_DEFAULT_SECTIONS, CONF_DEVICE_DASHBOARD_SECTION, DASHBOARD_DEVICE_SECTION_TRANSLATION_KEY, \
    DASHBOARD_NO_SECTION, LOAD_TYPE_DASHBOARD_DEFAULT_SECTION
from .ha_model.climate_controller import get_hvac_modes, QSClimateDuration
from .home_model.load import map_section_selected_name_in_section_list
from .ha_model.dynamic_group import QSDynamicGroup

from .ha_model.home import QSHome
from .ha_model.battery import QSBattery
from .ha_model.car import QSCar
from .ha_model.charger import QSChargerOCPP, QSChargerWallbox, QSChargerGeneric
from .ha_model.on_off_duration import QSOnOffDuration
from .ha_model.pool import QSPool
from .ha_model.solar import QSSolar

_LOGGER = logging.getLogger(__name__)

LOAD_TYPES_MENU = {
    QSHome.conf_type_name:None,
    QSBattery.conf_type_name:None,
    QSSolar.conf_type_name:None,
    "charger": {QSChargerOCPP.conf_type_name: None, QSChargerGeneric.conf_type_name:None}, #QSChargerWallbox.conf_type_name:None
    QSCar.conf_type_name:None,
    QSPool.conf_type_name:None,
    QSOnOffDuration.conf_type_name:None,
    QSClimateDuration.conf_type_name:None,
    QSDynamicGroup.conf_type_name:None
}




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


def selectable_amps_entities(hass: HomeAssistant, domains=None) -> list:
    """Return an entity selector which compatible entities."""

    if domains is None:
        ALLOWED_DOMAINS = [SENSOR_DOMAIN]
    else:
        ALLOWED_DOMAINS = domains
    entities = [
        ent.entity_id
        for ent in hass.states.async_all(ALLOWED_DOMAINS)
        if ent.attributes.get(ATTR_UNIT_OF_MEASUREMENT) in UnitOfElectricCurrent
        and ent.domain in ALLOWED_DOMAINS
    ]
    return entities

def selectable_calendar_entities(hass: HomeAssistant, domains=None) -> list:
    """Return an entity selector which compatible entities."""

    if domains is None:
        ALLOWED_DOMAINS = [CALENDAR_DOMAIN]
    else:
        ALLOWED_DOMAINS = domains
    entities = [
        ent.entity_id
        for ent in hass.states.async_all(ALLOWED_DOMAINS)
        if ent.domain in ALLOWED_DOMAINS
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

def selectable_percent_sensor_entities(
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

def selectable_percent_number_entities(
    hass: HomeAssistant,
) -> list:
    """Return an entity selector which compatible entities."""

    ALLOWED_DOMAINS = [NUMBER_DOMAIN]
    entities = [
        ent.entity_id
        for ent in hass.states.async_all(ALLOWED_DOMAINS)
        if ent.attributes.get(ATTR_UNIT_OF_MEASUREMENT) == PERCENTAGE
        and ent.domain in ALLOWED_DOMAINS
    ]
    return entities

def _get_reset_selector_entity_name(key:str):
    return f"{key}_qs_reset_selector"

def _get_entity_key_from_selector_key(key:str):
    if key.endswith("_qs_reset_selector"):
        return key.replace("_qs_reset_selector", "")
    return None

class QSFlowHandlerMixin(config_entries.ConfigEntryBaseFlow if TYPE_CHECKING else object):

    """Common methods for Quiet Solar config flows."""
    @abstractmethod
    def is_creation_flow(self)-> bool:
        """Return True if this is a creation flow, False if it is an options flow."""


    def add_entity_selector(self, sc_dict, key, is_required, entity_list=None, domain=None):

        default = self.config_entry.data.get(key)

        if default is None:
            if is_required:
                key_sc = vol.Required(key)
            else:
                key_sc = vol.Optional(key)
        else:
            if is_required:
                key_sc = vol.Required(key, description={"suggested_value": default})
            else:
                key_sc = vol.Optional(key, description={"suggested_value": default})

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

        # no need anymore .... solved by not doing default ... but description={"suggested_value": default}
        # if is_required is False:
        #    key_sc = vol.Optional(_get_reset_selector_entity_name(key), default=False, msg="Reset the entity selector", description="Reset the entity selector")
        #    sc_dict[key_sc] = cv.boolean


    def get_common_schema(self,
                          type,
                          add_power_value_selector=None,
                          add_load_power_sensor=False,
                          add_load_power_sensor_mandatory=False,
                          add_calendar=False,
                          add_boost_only=False,
                          add_mobile_app=False,
                          add_efficiency_selector=False,
                          add_is_3p=False,
                          add_max_phase_amps_selector=None,
                          add_power_group_selector=False,
                          add_max_on_off=False,
                          add_amps_sensors=False,
                          add_phase_number=False
                          ) -> dict:

        default_name = self.config_entry.data.get(CONF_NAME)


        if default_name is None:
            sc = {
            vol.Required(CONF_NAME): cv.string,
        }
        else:
            sc = {
                vol.Required(CONF_NAME, default=default_name): cv.string,
            }

        default_section = LOAD_TYPE_DASHBOARD_DEFAULT_SECTION.get(type)

        if default_section is not None:

            data_handler = self.hass.data.get(DOMAIN, {}).get(DATA_HANDLER)
            home: QSHome | None = None
            if data_handler and data_handler.home:
                home = data_handler.home

            options_raw_list = None
            if home is not None and home.dashboard_sections is not None and len(home.dashboard_sections) > 0:
                # we can have the list of sections
                options_raw_list = home.dashboard_sections

            if options_raw_list is None:
                options_raw_list = DASHBOARD_DEFAULT_SECTIONS

            # ok we want a section for those ones
            good_value = self.config_entry.data.get(CONF_DEVICE_DASHBOARD_SECTION, default_section)
            found_idx, options = map_section_selected_name_in_section_list(section_stored_name=good_value, section_list=options_raw_list, compute_options=True)

            if found_idx is None:
                good_value = DASHBOARD_NO_SECTION
            else:
                good_value = options[found_idx+1]

            dashboard_section = {
                vol.Required(CONF_DEVICE_DASHBOARD_SECTION, default=good_value): selector.SelectSelector(
                    SelectSelectorConfig(
                        options=options,
                        mode=SelectSelectorMode.DROPDOWN,
                        translation_key=DASHBOARD_DEVICE_SECTION_TRANSLATION_KEY
                    )
                )
            }
            sc.update(dashboard_section)


        if add_power_group_selector:
            data_handler = self.hass.data.get(DOMAIN, {}).get(DATA_HANDLER)
            if data_handler.home is not None and len(data_handler.home._all_dynamic_groups) > 1:
                # ok we can add it there is more than the home in the list
                # we will allow to select in the list of dynamic groups names
                default_group : str = "HOME"
                group_name_list : list[str] = []
                for group in data_handler.home._all_dynamic_groups:
                    if isinstance(group, QSHome):
                        default_group = f"default home root - {group.name}"
                    else:
                        if default_name == group.name and type==QSDynamicGroup.conf_type_name:
                            # do not add yourself...
                            pass
                        else:
                            group_name_list.append(group.name)
                if len(group_name_list) > 0:
                    options = [default_group]
                    options.extend(group_name_list)

                    sc.update({
                        vol.Required(CONF_DEVICE_DYNAMIC_GROUP_NAME, default=self.config_entry.data.get(CONF_DEVICE_DYNAMIC_GROUP_NAME, default_group)):
                            SelectSelector(
                                SelectSelectorConfig(
                                    options=options,
                                    mode=SelectSelectorMode.DROPDOWN,
                                )
                            )
                    })


        if add_is_3p:
            sc.update({
                vol.Optional(CONF_IS_3P,
                             default=self.config_entry.data.get(CONF_IS_3P, False)):
                cv.boolean,
            })

        if add_phase_number:
            sc.update({vol.Optional(CONF_MONO_PHASE,
                                 description={"suggested_value": self.config_entry.data.get(CONF_MONO_PHASE)}) :
                selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1,
                        max=3,
                        step=1,
                        mode=selector.NumberSelectorMode.BOX
                    )
                )
            })

        if add_boost_only:
            sc.update({
                vol.Optional(CONF_LOAD_IS_BOOST_ONLY,
                             default=self.config_entry.data.get(CONF_LOAD_IS_BOOST_ONLY, False)):
                cv.boolean,
            })

        if add_max_on_off:
            sc.update({
                vol.Optional(CONF_NUM_MAX_ON_OFF,
                             description={"suggested_value":self.config_entry.data.get(CONF_NUM_MAX_ON_OFF)}):
                selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1,
                        step=1,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
            })

        if add_power_value_selector:

            sc.update({vol.Required(CONF_POWER, description={"suggested_value":self.config_entry.data.get(CONF_POWER, add_power_value_selector)}):
            selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                    unit_of_measurement=UnitOfPower.WATT,
                )
            )})



        if add_max_phase_amps_selector:

            sc.update({vol.Optional(CONF_DYN_GROUP_MAX_PHASE_AMPS, description={"suggested_value":self.config_entry.data.get(CONF_DYN_GROUP_MAX_PHASE_AMPS, add_max_phase_amps_selector)}):
            selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                    unit_of_measurement=UnitOfElectricCurrent.AMPERE,
                )
            )})

        if add_load_power_sensor:

            power_entities = selectable_power_entities(self.hass)
            if len(power_entities) > 0:
                self.add_entity_selector(sc, CONF_ACCURATE_POWER_SENSOR, add_load_power_sensor_mandatory, entity_list=power_entities)

        if add_amps_sensors:

            amps_entities = selectable_amps_entities(self.hass)

            if len(amps_entities) > 0:
                self.add_entity_selector(sc, CONF_PHASE_1_AMPS_SENSOR, False,
                                         entity_list=amps_entities)
                self.add_entity_selector(sc, CONF_PHASE_2_AMPS_SENSOR, False,
                                             entity_list=amps_entities)
                self.add_entity_selector(sc, CONF_PHASE_3_AMPS_SENSOR, False,
                                             entity_list=amps_entities)


        if add_efficiency_selector:
            sc.update({
                vol.Optional(CONF_DEVICE_EFFICIENCY, description={"suggested_value":self.config_entry.data.get(CONF_DEVICE_EFFICIENCY, 100)}):
            selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1,
                    max=100,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                    unit_of_measurement=PERCENTAGE,
                )
            )})

        if add_calendar:
            entities = selectable_calendar_entities(self.hass)
            if len(entities) > 0:
                self.add_entity_selector(sc, CONF_CALENDAR, False, entity_list=entities)

        if add_mobile_app:
            options = []
            # hass.services.has_service("notify", "mobile_app_loaded_late")
            for service_name in self.hass.services.async_services_for_domain(NOTIFY_DOMAIN):
                if "mobile" in service_name:
                    options.append(service_name)



            if options:
                options.append(CONF_MOBILE_APP_NOTHING)
                default = self.config_entry.data.get(CONF_MOBILE_APP)

                if default:
                    keysc = vol.Optional(CONF_MOBILE_APP, description={"suggested_value": default})
                else:
                    keysc = vol.Optional(CONF_MOBILE_APP, default=CONF_MOBILE_APP_NOTHING)

                sc.update({
                    keysc: SelectSelector(
                        SelectSelectorConfig(
                            options=options,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    )})

                default = self.config_entry.data.get(CONF_MOBILE_APP_URL)
                if default is None:
                    sc.update({
                        vol.Optional(CONF_MOBILE_APP_URL): cv.string,
                    })
                else:
                    sc.update({
                        vol.Optional(CONF_MOBILE_APP_URL, description={"suggested_value": default}): cv.string,
                    })


        return sc


    def get_all_charger_schema_base(self, type, add_load_power_sensor_mandatory ):

        sc = self.get_common_schema(type=type,
                                    add_load_power_sensor=True,
                                    add_load_power_sensor_mandatory=add_load_power_sensor_mandatory,
                                    add_calendar=False,
                                    add_efficiency_selector=False,
                                    add_is_3p=True,
                                    add_power_group_selector=True,
                                    add_phase_number=True)

        default = self.config_entry.data.get(CONF_CHARGER_LATITUDE)
        if default is None:
            sc.update( {
                vol.Optional(CONF_CHARGER_LATITUDE):
                    vol.Maybe(vol.Coerce(float)),
            })
        else:

            sc.update( {
                vol.Optional(CONF_CHARGER_LATITUDE, description={"suggested_value": default}):
                    vol.Maybe(vol.Coerce(float)),
            })

        default = self.config_entry.data.get(CONF_CHARGER_LONGITUDE)
        if default is None:

            sc.update( {
                vol.Optional(CONF_CHARGER_LONGITUDE):
                    vol.Maybe(vol.Coerce(float)),
            })
        else:
            sc.update( {
                vol.Optional(CONF_CHARGER_LONGITUDE, description={"suggested_value": default}):
                    vol.Maybe(vol.Coerce(float)),
            })

        sc.update( {
                    vol.Optional(CONF_CHARGER_MAX_CHARGE, description={"suggested_value":self.config_entry.data.get(CONF_CHARGER_MAX_CHARGE, 32)}):
                        selector.NumberSelector(
                            selector.NumberSelectorConfig(
                                min=1,
                                step=1,
                                mode=selector.NumberSelectorMode.BOX,
                                unit_of_measurement=UnitOfElectricCurrent.AMPERE,
                            )
                        ),
                    vol.Optional(CONF_CHARGER_MIN_CHARGE, description={"suggested_value":self.config_entry.data.get(CONF_CHARGER_MIN_CHARGE, 6)}):
                        selector.NumberSelector(
                            selector.NumberSelectorConfig(
                                min=1,
                                step=1,
                                mode=selector.NumberSelectorMode.BOX,
                                unit_of_measurement=UnitOfElectricCurrent.AMPERE,
                            )
                        ),
                })
        return sc


    @abstractmethod
    async def async_entry_next(self, data, type):
        """Handle the next step based on user input."""
        data[DEVICE_TYPE] = type
        self.clean_data(data)
        return await self._async_entry_next(data)


    @abstractmethod
    async def _async_entry_next(self, data):
        """Handle the next step based on user input."""

    def clean_data(self, data):
        to_be_removed = []
        for k, v in data.items():
            if v is None:
                to_be_removed.append(k)
            else:
                key_to_be_reset = _get_entity_key_from_selector_key(k)
                if key_to_be_reset:
                    if v:
                        to_be_removed.append(key_to_be_reset)
                    data[k] = False
        for k in to_be_removed:
            data.pop(k)


    def get_entry_title(self, data):

        type = data.get(DEVICE_TYPE, None)
        if type is None:
            name = "unknown"
        else:
            name = LOAD_NAMES.get(type, "unknown")

        return f"{name}: {data.get(CONF_NAME, "device")}"
    async def async_step_home(self, user_input=None):

        TYPE = QSHome.conf_type_name
        if user_input is not None:
            #do some stuff to update
            r = await self.async_entry_next(user_input, TYPE)
            return r

        sc_dict = self.get_common_schema(type=TYPE, add_is_3p=True, add_max_phase_amps_selector=54, add_amps_sensors=True)

        sc_dict.update(  {

                vol.Optional(CONF_HOME_VOLTAGE, description={"suggested_value":self.config_entry.data.get(CONF_HOME_VOLTAGE, 230)}):
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

        sc_dict.update({
            vol.Required(CONF_HOME_PEAK_PRICE,
                         description={"suggested_value":self.config_entry.data.get(CONF_HOME_PEAK_PRICE,
                                                            0.27)}): float,
            vol.Required(CONF_HOME_OFF_PEAK_PRICE,
                         description={"suggested_value":self.config_entry.data.get(CONF_HOME_OFF_PEAK_PRICE,
                                                            0.2068)}): float,

            vol.Required(CONF_HOME_START_OFF_PEAK_RANGE_1,
                         description={"suggested_value":self.config_entry.data.get(CONF_HOME_START_OFF_PEAK_RANGE_1,
                                                            "00:00:00")}): selector.TimeSelector(),
            vol.Required(CONF_HOME_END_OFF_PEAK_RANGE_1,
                         description={"suggested_value":self.config_entry.data.get(CONF_HOME_END_OFF_PEAK_RANGE_1,
                                                            "00:00:00")}): selector.TimeSelector(),
            vol.Required(CONF_HOME_START_OFF_PEAK_RANGE_2,
                         description={"suggested_value":self.config_entry.data.get(CONF_HOME_START_OFF_PEAK_RANGE_2,
                                                            "00:00:00")}): selector.TimeSelector(),
            vol.Required(CONF_HOME_END_OFF_PEAK_RANGE_2,
                         description={"suggested_value":self.config_entry.data.get(CONF_HOME_END_OFF_PEAK_RANGE_2,
                                                            "00:00:00")}): selector.TimeSelector(),

        })

        for i in range(0, DASHBOARD_NUM_SECTION_MAX):
            key_section_name = f"{CONF_DASHBOARD_SECTION_NAME}_{i}"
            key_section_icon = f"{CONF_DASHBOARD_SECTION_ICON}_{i}"

            default_section_name = None
            default_section_icon = None
            if i < len(DASHBOARD_DEFAULT_SECTIONS):
                default_section_name = DASHBOARD_DEFAULT_SECTIONS[i][0]
                default_section_icon = DASHBOARD_DEFAULT_SECTIONS[i][1]

            sc_dict.update({
                vol.Optional(key_section_name,
                             description={"suggested_value":self.config_entry.data.get(key_section_name,
                                                                default_section_name)}): cv.string,
                vol.Optional(key_section_icon,
                             description={
                                 "suggested_value": self.config_entry.data.get(key_section_icon,
                                                                               default_section_icon)}): cv.string,
            })

        schema = vol.Schema(sc_dict)

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )


    async def async_step_solar(self, user_input=None):

        TYPE = QSSolar.conf_type_name
        if user_input is not None:
            #do sotme stuff to update
            r = await self.async_entry_next(user_input, TYPE)
            return r

        sc_dict = self.get_common_schema(type=TYPE)

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
                keysc = vol.Required(CONF_SOLAR_FORECAST_PROVIDER, description={"suggested_value":default})
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

        TYPE = QSChargerGeneric.conf_type_name
        if user_input is not None:
            #do sotme stuff to update
            r = await self.async_entry_next(user_input, TYPE)
            return r

        sc_dict = self.get_all_charger_schema_base(type=TYPE, add_load_power_sensor_mandatory=True)
        self.add_entity_selector(sc_dict, CONF_CHARGER_PLUGGED, True, domain=[BINARY_SENSOR_DOMAIN])
        self.add_entity_selector(sc_dict, CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, True, domain=[NUMBER_DOMAIN])
        self.add_entity_selector(sc_dict, CONF_CHARGER_PAUSE_RESUME_SWITCH, True, domain=[SWITCH_DOMAIN])
        self.add_entity_selector(sc_dict, CONF_CHARGER_THREE_TO_ONE_PHASE_SWITCH, False, domain=[SWITCH_DOMAIN])

        schema = vol.Schema(sc_dict)

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_charger_ocpp(self, user_input=None):

        TYPE = QSChargerOCPP.conf_type_name
        if user_input is not None:
            #do sotme stuff to update
            r = await self.async_entry_next(user_input, TYPE)
            return r

        sc_dict = self.get_all_charger_schema_base(type=TYPE, add_load_power_sensor_mandatory=False)

        default = self.config_entry.data.get(CONF_CHARGER_DEVICE_OCPP)
        if default is None:
            key_sc = vol.Required(CONF_CHARGER_DEVICE_OCPP)
        else:
            key_sc = vol.Required(CONF_CHARGER_DEVICE_OCPP, description={"suggested_value":default})

        sc_dict[key_sc] = selector.DeviceSelector(
                            selector.DeviceSelectorConfig(
                                integration="ocpp"
                                )
                            )
        schema = vol.Schema(sc_dict)

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_charger_wallbox(self, user_input=None):

        TYPE = QSChargerWallbox.conf_type_name
        if user_input is not None:
            #do sotme stuff to update
            r = await self.async_entry_next(user_input, TYPE)
            return r
        sc_dict = self.get_all_charger_schema_base(type=TYPE, add_load_power_sensor_mandatory=False)

        default = self.config_entry.data.get(CONF_CHARGER_DEVICE_WALLBOX)
        if default is None:
            key_sc = vol.Required(CONF_CHARGER_DEVICE_WALLBOX)
        else:
            key_sc = vol.Required(CONF_CHARGER_DEVICE_WALLBOX, description={"suggested_value":default})

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

        TYPE = QSBattery.conf_type_name

        if user_input is not None:
            #do sotme stuff to update
            r = await self.async_entry_next(user_input, TYPE)
            return r

        sc_dict = self.get_common_schema(type=TYPE)

        power_entities = selectable_power_entities(self.hass)
        if len(power_entities) > 0 :
            self.add_entity_selector(sc_dict, CONF_BATTERY_CHARGE_DISCHARGE_SENSOR, False, entity_list=power_entities)

        number_entites = selectable_power_entities(self.hass, domains=[NUMBER_DOMAIN])
        if len(number_entites) > 0:
            self.add_entity_selector(sc_dict, CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER, False, entity_list=number_entites)
            self.add_entity_selector(sc_dict, CONF_BATTERY_MAX_CHARGE_POWER_NUMBER, False,
                                     entity_list=number_entites)

        percent_entities = selectable_percent_sensor_entities(self.hass)
        if len(percent_entities) > 0 :
            self.add_entity_selector(sc_dict, CONF_BATTERY_CHARGE_PERCENT_SENSOR, False, entity_list=percent_entities)

        self.add_entity_selector(sc_dict, CONF_BATTERY_CHARGE_FROM_GRID_SWITCH, False, domain=[SWITCH_DOMAIN])

        sc_dict.update(
            {

                vol.Optional(CONF_BATTERY_CAPACITY, description={"suggested_value":self.config_entry.data.get(CONF_BATTERY_CAPACITY, 0)}):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfEnergy.WATT_HOUR,
                        )
                    ),
                vol.Optional(CONF_BATTERY_MIN_CHARGE_PERCENT, description={"suggested_value":self.config_entry.data.get(CONF_BATTERY_MIN_CHARGE_PERCENT, 0)}):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0,
                            max=100,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=PERCENTAGE,
                        )
                    ),
                vol.Optional(CONF_BATTERY_MAX_CHARGE_PERCENT,
                             description={"suggested_value":self.config_entry.data.get(CONF_BATTERY_MAX_CHARGE_PERCENT, 100)}):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0,
                            max=100,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=PERCENTAGE,
                        )
                    ),
                vol.Optional(CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE,
                             description={"suggested_value":self.config_entry.data.get(CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE, 0)}):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfPower.WATT,
                        )
                    ),
                vol.Optional(CONF_BATTERY_MAX_CHARGE_POWER_VALUE,
                             description={"suggested_value":self.config_entry.data.get(CONF_BATTERY_MAX_CHARGE_POWER_VALUE, 0)}):
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

        TYPE = QSCar.conf_type_name
        do_force_dampening = False

        min_charge = self.config_entry.data.get(CONF_CAR_CHARGER_MIN_CHARGE, 6)
        max_charge = self.config_entry.data.get(CONF_CAR_CHARGER_MAX_CHARGE, 32)

        if user_input is not None:

            if "force_dampening" in user_input:
                do_force_dampening = True
            else:
                #do some stuff to update
                new_dampening = user_input.get(CONF_CAR_CUSTOM_POWER_CHARGE_VALUES, False)
                user_input[DEVICE_TYPE] = TYPE

                # Check if we need to regenerate dampening fields
                charge_values_changed = (user_input.get(CONF_CAR_CHARGER_MIN_CHARGE, 0) != min_charge or 
                                       user_input.get(CONF_CAR_CHARGER_MAX_CHARGE, 0) != max_charge)
                
                # Only force dampening redisplay if:
                # 1. Dampening is being newly enabled, OR
                # 2. Dampening was already enabled AND charge values changed (need new range)
                need_dampening_redisplay = (
                    (orig_dampening is False and new_dampening is True) or
                    (orig_dampening is True and new_dampening is True and charge_values_changed)
                )

                if need_dampening_redisplay:
                    #self.hass.config_entries.async_update_entry(
                    #    self.config_entry, data=user_input, options=self.config_entry.options
                    #)
                    # or more simply:
                    self.hass.config_entries.async_update_entry(self.config_entry, data=user_input)
                    return await self.async_step_car({"force_dampening": True})

                r =  await self.async_entry_next(user_input, TYPE)
                return r


        sc_dict = self.get_common_schema(type=TYPE, add_calendar=True, add_mobile_app=True, add_efficiency_selector=True)

        sc_dict.update(
            {
                vol.Optional(CONF_CAR_IS_INVITED, default=self.config_entry.data.get(CONF_CAR_IS_INVITED, False)):
                    cv.boolean,
            }
        )

        self.add_entity_selector(sc_dict, CONF_CAR_PLUGGED, False, domain=[BINARY_SENSOR_DOMAIN])
        self.add_entity_selector(sc_dict, CONF_CAR_TRACKER, False, domain=[DEVICE_TRACKER_DOMAIN])


        percent_entities = selectable_percent_sensor_entities(self.hass)
        if len(percent_entities) > 0 :
            self.add_entity_selector(sc_dict, CONF_CAR_CHARGE_PERCENT_SENSOR, False, entity_list=percent_entities)

        num_percent_entity = selectable_percent_number_entities(self.hass)
        if len(num_percent_entity) > 0 :
            self.add_entity_selector(sc_dict, CONF_CAR_CHARGE_PERCENT_MAX_NUMBER, False, entity_list=num_percent_entity)
            if self.config_entry.data.get(CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS) is None or self.config_entry.data.get(CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS) == "":
                sc_dict.update(
                    {
                        vol.Optional(CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS):
                            cv.string,
                    }
                )

            else:
                sc_dict.update(
                    {
                        vol.Optional(CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS, description={"suggested_value":self.config_entry.data.get(CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS, "")}):
                            cv.string,
                    }
                )



        sc_dict.update(
            {
                vol.Optional(CONF_CAR_BATTERY_CAPACITY, description={"suggested_value":self.config_entry.data.get(CONF_CAR_BATTERY_CAPACITY, 0)}):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfEnergy.WATT_HOUR,
                        )
                    ),
                vol.Optional(CONF_CAR_CHARGER_MIN_CHARGE, description={"suggested_value": self.config_entry.data.get(CONF_CAR_CHARGER_MIN_CHARGE, 6)}):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfElectricCurrent.AMPERE,
                        )
                    ),

                vol.Optional(CONF_CAR_CHARGER_MAX_CHARGE, description={"suggested_value":self.config_entry.data.get(CONF_CAR_CHARGER_MAX_CHARGE, 32)}):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=UnitOfElectricCurrent.AMPERE,
                        )
                    ),
                vol.Optional(CONF_DEFAULT_CAR_CHARGE, description={"suggested_value":int(self.config_entry.data.get(CONF_DEFAULT_CAR_CHARGE, 100))}):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0,
                            max=100,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=PERCENTAGE,
                        )
                    ),
                vol.Optional(CONF_MINIMUM_OK_CAR_CHARGE,
                             description={"suggested_value": int(self.config_entry.data.get(CONF_MINIMUM_OK_CAR_CHARGE, 50))}):
                    selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0,
                            max=100,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement=PERCENTAGE,
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
                     description={"suggested_value":self.config_entry.data.get(f"charge_{a}")})] = selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0,
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

        TYPE = QSPool.conf_type_name
        if user_input is not None:
            #do some stuff to update
            for min_temp, max_temp, default in POOL_TEMP_STEPS:
                if user_input.get(f"water_temp_{max_temp}", -1) == -1:
                    user_input[f"water_temp_{max_temp}"] = default

            r = await self.async_entry_next(user_input, TYPE)
            return r

        sc_dict = self.get_common_schema(type=TYPE,
                                         add_power_value_selector=1500,
                                         add_load_power_sensor=True,
                                         add_calendar=False,
                                         add_power_group_selector=True,
                                         add_max_on_off=True)

        self.add_entity_selector(sc_dict, CONF_SWITCH, True, domain=[SWITCH_DOMAIN])

        temperature_entities = selectable_temperature_entities(self.hass)
        if len(temperature_entities) > 0 :
            self.add_entity_selector(sc_dict, CONF_POOL_TEMPERATURE_SENSOR, True, entity_list=temperature_entities)



        for min_temp, max_temp, default in POOL_TEMP_STEPS:
            sc_dict[vol.Optional(f"water_temp_{max_temp}",
                                 description={"suggested_value":self.config_entry.data.get(f"water_temp_{max_temp}", default)})] = selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=-1,
                    step=1,
                    max=24,
                    mode=selector.NumberSelectorMode.BOX,
                    unit_of_measurement=UnitOfTime.HOURS,
                )
            )

        schema = vol.Schema(sc_dict)

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_on_off_duration(self, user_input=None):

        TYPE = QSOnOffDuration.conf_type_name
        if user_input is not None:
            #do sotme stuff to update
            r = await self.async_entry_next(user_input, TYPE)
            return r

        sc_dict = self.get_common_schema(type=TYPE,
                                         add_power_value_selector=1000,
                                         add_load_power_sensor=True,
                                         add_calendar=True,
                                         add_boost_only=True,
                                         add_power_group_selector=True,
                                         add_max_on_off=True)

        self.add_entity_selector(sc_dict, CONF_SWITCH, True, domain=[SWITCH_DOMAIN])



        schema = vol.Schema(sc_dict)

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_climate(self, user_input=None):

        TYPE = QSClimateDuration.conf_type_name

        orig_climate_entity = self.config_entry.data.get(CONF_CLIMATE)

        if user_input is not None:
            #do some stuff to update
            if "force_climate" in user_input:
                pass
            else:
                if (user_input.get(CONF_CLIMATE) != orig_climate_entity or
                        user_input.get(CONF_CLIMATE_HVAC_MODE_OFF) is None or
                        user_input.get(CONF_CLIMATE_HVAC_MODE_ON) is None):
                    # we need to force to come back to reselect the hvac modes
                    self.config_entry.data = user_input
                    return await self.async_step_climate({"force_climate": True})


                r = await self.async_entry_next(user_input, TYPE)
                return r

        sc_dict = self.get_common_schema(type=TYPE,
                                         add_power_value_selector=1000,
                                         add_load_power_sensor=True,
                                         add_calendar=True,
                                         add_boost_only=True,
                                         add_power_group_selector=True,
                                         add_max_on_off=True)

        self.add_entity_selector(sc_dict, CONF_CLIMATE, True, domain=[CLIMATE_DOMAIN])

        climate_entity = self.config_entry.data.get(CONF_CLIMATE)

        if climate_entity is not None:

            hvac_modes = get_hvac_modes(self.hass, climate_entity)

            sc_dict.update({
                vol.Required(CONF_CLIMATE_HVAC_MODE_OFF,
                             default=self.config_entry.data.get(CONF_CLIMATE_HVAC_MODE_OFF)):
                    SelectSelector(
                        SelectSelectorConfig(
                            options=hvac_modes,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    )
            })
            sc_dict.update({
                vol.Required(CONF_CLIMATE_HVAC_MODE_ON,
                             default=self.config_entry.data.get(CONF_CLIMATE_HVAC_MODE_ON)):
                    SelectSelector(
                        SelectSelectorConfig(
                            options=hvac_modes,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    )
            })



        schema = vol.Schema(sc_dict)

        return self.async_show_form(
            step_id=TYPE,
            data_schema=schema
        )

    async def async_step_dynamic_group(self, user_input=None):

        TYPE = QSDynamicGroup.conf_type_name
        if user_input is not None:
            #do somestuff to update
            r = await self.async_entry_next(user_input, TYPE)
            return r

        sc_dict = self.get_common_schema(type=TYPE,
                                         add_load_power_sensor=True,
                                         add_is_3p=True,
                                         add_max_phase_amps_selector=32,
                                         add_power_group_selector=True,
                                         add_amps_sensors=True
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
    config_entry: FakeConfigEntry | None | ConfigEntry = FakeConfigEntry()

    @staticmethod
    @callback
    def async_get_options_flow(
         config_entry: ConfigEntry,
     ) -> OptionsFlow:
         """Get the options flow for this handler."""
         return QSOptionsFlowHandler(config_entry)

    def is_creation_flow(self)-> bool:
        return True

    async def async_step_user(self, user_input=None):
        """initial step (menu) for user initiated flows"""

        possible_menus = [QSHome.conf_type_name]
        data_handler = self.hass.data.get(DOMAIN, {}). get(DATA_HANDLER)
        if data_handler is not None:
            if data_handler.home is not None:
                possible_menus = [t for t in LOAD_TYPES_MENU if t != QSHome.conf_type_name]
                if data_handler.home._battery is not None:
                    possible_menus.remove(QSBattery.conf_type_name)
                if data_handler.home._solar_plant is not None:
                    possible_menus.remove(QSSolar.conf_type_name)



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
            menu_options=[t for t in LOAD_TYPES_MENU["charger"]],
        )

    async def _async_entry_next(self, data):
        """Handle the next step based on user input."""
        u_id = f"Quiet Solar: {data.get(CONF_NAME, "device")} {data.get(DEVICE_TYPE, "unknown")}"
        await self.async_set_unique_id(u_id)
        return self.async_create_entry(title=self.get_entry_title(data), data=data)



# Version threshold for config_entry setting in options flow
# See: https://github.com/home-assistant/core/pull/129562

HA_OPTIONS_FLOW_VERSION_THRESHOLD = "2024.11.99"

class QSOptionsFlowHandler(QSFlowHandlerMixin, OptionsFlow):

    VERSION = 1
    MINOR_VERSION = 1

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize Quiet Solar options flow."""
        super().__init__()

        if AwesomeVersion(HAVERSION) < HA_OPTIONS_FLOW_VERSION_THRESHOLD:
            self.config_entry = config_entry

        self._errors = {}

    def is_creation_flow(self)-> bool:
        return False

    async def _async_entry_next(self, data):
        """Handle the next step based on user input."""
        self.hass.config_entries.async_update_entry(
            self.config_entry, data=data, options=self.config_entry.options, title=self.get_entry_title(data)
        )

        # always reset everything to be sure all is well set
        await async_reload_quiet_solar(self.hass, except_for_entry_id=self.config_entry.entry_id)

        return self.async_create_entry(title=None, data={})

    async def async_step_init(self, user_input=None) -> ConfigFlowResult:
        """Manage the options."""

        type = self.config_entry.data.get(DEVICE_TYPE)
        if type is None:
            return self.async_create_entry(title=None, data={})


        step_name = f"async_step_{type}"

        return await getattr(self, step_name)( user_input=user_input)

