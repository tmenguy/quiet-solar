MAX_POSSIBLE_APERAGE = 64

DOMAIN = "quiet_solar"
MANUFACTURER = "QuietSolarAbstraction"
DEFAULT_ATTRIBUTION = f"Data provided by {MANUFACTURER}"
DATA_HANDLER = "quiet_solar_data_handler"
DATA_DEVICE_IDS = "quiet_solar_device_ids"

DEVICE_TYPE = "device_type"
COMMAND_BASED_POWER_SENSOR = "command_based_power_sensor"
HOME_CONSUMPTION_SENSOR = "home_consumption_sensor"
HOME_NON_CONTROLLED_CONSUMPTION_SENSOR = "home_non_controlled_consumption_sensor"
HOME_AVAILABLE_POWER_SENSOR = "home_available_power_sensor"

CONF_POWER = "power"
CONF_ACCURATE_POWER_SENSOR = "accurate_power_sensor"
CONF_SWITCH = "switch"
CONF_SELECT = "select"
CONF_CALENDAR = "calendar"



CONF_HOME_VOLTAGE = "home_voltage"

CONF_GRID_POWER_SENSOR = "grid_active_power_sensor"
CONF_GRID_POWER_SENSOR_INVERTED = "grid_active_power_sensor_inverted"

CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR = "inverter_active_power_sensor"
CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR = "inverter_input_power_sensor"
CONF_SOLAR_FORECAST_PROVIDER = "solar_forecast_provider"

CONF_BATTERY_CHARGE_DISCHARGE_SENSOR = "battery_charge_discharge_sensor"
CONF_BATTERY_CHARGE_PERCENT_SENSOR = "battery_charge_percent_sensor"
CONF_BATTERY_CAPACITY = "battery_capacity"
CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE = "battery_max_discharge_power_value"
CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER = "battery_max_discharge_power_number"
CONF_BATTERY_MAX_CHARGE_POWER_VALUE = "battery_max_charge_power_value"
CONF_BATTERY_MAX_CHARGE_POWER_NUMBER = "battery_max_charge_power_number"


CONF_CHARGER_MAX_CHARGE = "charger_max_charge"
CONF_CHARGER_MIN_CHARGE = "charger_min_charge"
CONF_CHARGER_IS_3P = "charger_is_3p"
CONF_CHARGER_CONSUMPTION = "charger_consumption"
CONF_DEFAULT_CAR_CHARGE = "default_car_charge"

CONF_CHARGER_DEVICE_OCPP = "charger_device_ocpp"
CONF_CHARGER_DEVICE_WALLBOX = "charger_device_wallbox"

CONF_CHARGER_PLUGGED = "charger_plugged"
CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER = "charger_max_charging_current_number"
CONF_CHARGER_PAUSE_RESUME_SWITCH = "charger_pause_resume_switch"

CONF_CHARGER_STATUS_SENSOR = "charger_status_sensor"


CHARGER_NO_CAR_CONNECTED = "No car connected"


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


SOLCAST_SOLAR_DOMAIN = "solcast_solar"
OPEN_METEO_SOLAR_DOMAIN = "open_meteo_solar_forecast"

SENSOR_HOME_AVAILABLE_EXTRA_POWER = "qs_home_extra_available_power"
SENSOR_HOME_CONSUMPTION_POWER = "qs_home_consumption_power"
SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER = "qs_home_non_controlled_consumption_power"

CONF_HOME_START_OFF_PEAK_RANGE_1 = "home_start_off_peak_range_1"
CONF_HOME_END_OFF_PEAK_RANGE_1 = "home_end_off_peak_range_1"
CONF_HOME_START_OFF_PEAK_RANGE_2 = "home_start_off_peak_range_2"
CONF_HOME_END_OFF_PEAK_RANGE_2 = "home_end_off_peak_range_2"
CONF_HOME_PEAK_PRICE = "home_peak_price"
CONF_HOME_OFF_PEAK_PRICE = "home_off_peak_price"


BUTTON_HOME_RESET_HISTORY = "qs_home_reset_history"
BUTTON_HOME_SERIALIZE_FOR_DEBUG = "qs_home_serialize_for_debug"

SWITCH_CAR_NEXT_CHARGE_FULL = "qs_next_car_charge_full"

BUTTON_CAR_NEXT_CHARGE_FORCE_NOW = "qs_next_car_charge_force_now"



HA_CONSTRAINT_SENSOR_HISTORY = "qs_stored_constraints"
FLOATING_PERIOD_S = 30 * 3600


# better for serialization that using an IntEnum
CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE = 10
CONSTRAINT_TYPE_MANDATORY_END_TIME  = 9
CONSTRAINT_TYPE_BEFORE_BATTERY_AUTO_GREEN = 8
CONSTRAINT_TYPE_FILLER = 7
CONSTRAINT_TYPE_FILLER_AUTO = 6

