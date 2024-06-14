from homeassistant.const import Platform
import logging

DOMAIN = "quiet_solar"
MANUFACTURER = "QuietSolarAbstraction"
DEFAULT_ATTRIBUTION = f"Data provided by {MANUFACTURER}"
DATA_HANDLER = "quiet_solar_data_handler"
DATA_DEVICE_IDS = "quiet_solar_device_ids"

LOGGER = logging.getLogger(__package__)

PLATFORMS = [
    Platform.BINARY_SENSOR,
    #Platform.CLIMATE, may be usefull for defining constraint on climate? especially homekit?
    #Platform.COVER, may be usefull for defining constraint on climate? especially homekit?
    Platform.SELECT,
    Platform.SENSOR,
    Platform.SWITCH,
]

DEVICE_TYPE = "device_type"

CONF_HOME_VOLTAGE = "home_voltage"

CONF_GRID_POWER_SENSOR = "grid_active_power_sensor"
CONF_GRID_POWER_SENSOR_INVERTED = "grid_active_power_sensor_inverted"

CONF_INVERTER_ACTIVE_POWER_SENSOR = "inverter_active_power_sensor"
CONF_INVERTER_INPUT_POWER_SENSOR = "inverter_input_power_sensor"

CONF_BATTERY_CHARGE_DISCHARGE_SENSOR = "battery_charge_discharge_sensor"
CONF_BATTERY_CAPACITY = "battery_capacity"


CONF_CHARGER_MAX_CHARGE = "charger_max_charge"
CONF_CHARGER_MIN_CHARGE = "charger_min_charge"
CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER = "charger_max_charging_current_number"
CONF_CHARGER_PAUSE_RESUME_SWITCH = "charger_pause_resume_switch"

CONF_CAR_CHARGE_PERCENT_SENSOR = "car_charge_percent_sensor"
CONF_CAR_BATTERY_CAPACITY = "car_battery_capacity"
CONF_CAR_CHARGER_MIN_CHARGE = "car_charger_min_charge"
CONF_CAR_CHARGER_MAX_CHARGE = "car_charger_max_charge"

CONF_POOL_TEMPERATURE_SENSOR = "pool_temperature_sensor"
CONF_POOL_PUMP_POWER = "pool_pump_power"
CONF_POOL_PUMP_SWITCH = "pool_pump_switch"

CONF_SWITCH_POWER  = "switch_power"
CONF_SWITCH = "switch"