import logging
from datetime import datetime

import pytz

DATETIME_MAX_UTC = datetime.max.replace(tzinfo=pytz.utc)
DATETIME_MIN_UTC = datetime.min.replace(tzinfo=pytz.utc)

MAX_POSSIBLE_APERAGE = 64

DOMAIN = "quiet_solar"
MANUFACTURER = "QuietSolarAbstraction"
DEFAULT_ATTRIBUTION = f"Data provided by {MANUFACTURER}"
DATA_HANDLER = "quiet_solar_data_handler"
DATA_DEVICE_IDS = "quiet_solar_device_ids"

LOGGER = logging.getLogger(__package__)

DEVICE_TYPE = "device_type"
COMMAND_BASED_POWER_SENSOR = "command_based_power_sensor"
HOME_CONSUMPTION_SENSOR = "home_consumption_sensor"
HOME_NON_CONTROLLED_CONSUMPTION_SENSOR = "home_non_controlled_consumption_sensor"

CONF_POWER = "power"
CONF_ACCURATE_POWER_SENSOR = "accurate_power_sensor"
CONF_SWITCH = "switch"
CONF_SELECT = "select"



CONF_HOME_VOLTAGE = "home_voltage"

CONF_GRID_POWER_SENSOR = "grid_active_power_sensor"
CONF_GRID_POWER_SENSOR_INVERTED = "grid_active_power_sensor_inverted"

CONF_INVERTER_ACTIVE_POWER_SENSOR = "inverter_active_power_sensor"
CONF_INVERTER_INPUT_POWER_SENSOR = "inverter_input_power_sensor"

CONF_BATTERY_CHARGE_DISCHARGE_SENSOR = "battery_charge_discharge_sensor"
CONF_BATTERY_CAPACITY = "battery_capacity"
CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE = "battery_max_discharge_power_value"
CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER = "battery_max_discharge_power_number"
CONF_BATTERY_MAX_CHARGE_POWER_VALUE = "battery_max_charge_power_value"
CONF_BATTERY_MAX_CHARGE_POWER_NUMBER = "battery_max_charge_power_number"


CONF_CHARGER_MAX_CHARGE = "charger_max_charge"
CONF_CHARGER_MIN_CHARGE = "charger_min_charge"
CONF_CHARGER_IS_3P = "charger_is_3p"
CONF_CHARGER_CONSUMPTION = "charger_consumption"


CONF_CHARGER_DEVICE_OCPP = "charger_device_ocpp"
CONF_CHARGER_DEVICE_WALLBOX = "charger_device_wallbox"

CONF_CHARGER_PLUGGED = "charger_plugged"
CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER = "charger_max_charging_current_number"
CONF_CHARGER_PAUSE_RESUME_SWITCH = "charger_pause_resume_switch"

CONF_CHARGER_STATUS_SENSOR = "charger_status_sensor"

CONF_CAR_TRACKER = "car_tracker"
CONF_CAR_PLUGGED = "car_plugged"
CONF_CAR_CHARGE_PERCENT_SENSOR = "car_charge_percent_sensor"
CONF_CAR_BATTERY_CAPACITY = "car_battery_capacity"
CONF_CAR_CHARGER_MIN_CHARGE = "car_charger_min_charge"
CONF_CAR_CHARGER_MAX_CHARGE = "car_charger_max_charge"
CONF_CAR_CUSTOM_POWER_CHARGE_VALUES = "car_use_custom_power_charge_values"
CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P = "car_is_custom_power_charge_values_3p"

CONF_POOL_TEMPERATURE_SENSOR = "pool_temperature_sensor"
CONF_POOL_IS_PUMP_VARIABLE_SPEED = "pool_is_pump_variable_speed"



