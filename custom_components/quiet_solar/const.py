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
CONF_PHASE_1_AMPS_SENSOR = "phase_1_amps_sensor"
CONF_PHASE_2_AMPS_SENSOR = "phase_2_amps_sensor"
CONF_PHASE_3_AMPS_SENSOR = "phase_3_amps_sensor"

CONF_SWITCH = "switch"
CONF_SELECT = "select"
CONF_CALENDAR = "calendar"
CONF_IS_3P = "device_is_3p"
CONF_LOAD_IS_BOOST_ONLY = "load_is_boost_only"
CONF_DEVICE_EFFICIENCY = "device_efficiency"
CONF_DEVICE_DYNAMIC_GROUP_NAME = "dynamic_group_name"
CONF_NUM_MAX_ON_OFF = "num_max_on_off"

CONF_HOME_VOLTAGE = "home_voltage"

CONF_GRID_POWER_SENSOR = "grid_active_power_sensor"
CONF_GRID_POWER_SENSOR_INVERTED = "grid_active_power_sensor_inverted"

CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR = "inverter_active_power_sensor"
CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR = "inverter_input_power_sensor"
CONF_SOLAR_FORECAST_PROVIDER = "solar_forecast_provider"

CONF_BATTERY_CHARGE_DISCHARGE_SENSOR = "battery_charge_discharge_sensor"
CONF_BATTERY_CHARGE_PERCENT_SENSOR = "battery_charge_percent_sensor"
CONF_BATTERY_CHARGE_FROM_GRID_SWITCH = "battery_charge_from_grid_switch"
CONF_BATTERY_CAPACITY = "battery_capacity"
CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE = "battery_max_discharge_power_value"
CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER = "battery_max_discharge_power_number"
CONF_BATTERY_MAX_CHARGE_POWER_VALUE = "battery_max_charge_power_value"
CONF_BATTERY_MAX_CHARGE_POWER_NUMBER = "battery_max_charge_power_number"
CONF_BATTERY_MIN_CHARGE_PERCENT = "battery_min_charge_percent_value"
CONF_BATTERY_MAX_CHARGE_PERCENT = "battery_max_charge_percent_value"

CONF_CLIMATE = "climate"
CONF_CLIMATE_HVAC_MODE_ON = "climate_hvac_on"
CONF_CLIMATE_HVAC_MODE_OFF = "climate_hvac_off"

CONF_DYN_GROUP_MAX_PHASE_AMPS = "dyn_group_max_phase_amps"

CONF_CHARGER_MAX_CHARGE = "charger_max_charge"
CONF_CHARGER_MIN_CHARGE = "charger_min_charge"

CONF_CHARGER_CONSUMPTION = "charger_consumption"
CONF_DEFAULT_CAR_CHARGE = "default_car_charge"
CONF_CHARGER_LATITUDE = "charger_latitude"
CONF_CHARGER_LONGITUDE = "charger_longitude"

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
CONF_CAR_CHARGE_PERCENT_MAX_NUMBER = "car_charge_percent_max_number"
CONF_CAR_IS_DEFAULT = "car_is_default"
CONF_CAR_BATTERY_CAPACITY = "car_battery_capacity"
CONF_CAR_CHARGER_MIN_CHARGE = "car_charger_min_charge"
CONF_CAR_CHARGER_MAX_CHARGE = "car_charger_max_charge"
CONF_CAR_CUSTOM_POWER_CHARGE_VALUES = "car_use_custom_power_charge_values"
CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P = "car_is_custom_power_charge_values_3p"
CONF_MOBILE_APP = "mobile_app"
CONF_MOBILE_APP_URL = "mobile_app_url"
CONF_MOBILE_APP_NOTHING = "No Notifications"

CONF_POOL_TEMPERATURE_SENSOR = "pool_temperature_sensor"
CONF_POOL_IS_PUMP_VARIABLE_SPEED = "pool_is_pump_variable_speed"

CONF_POOL_WINTER_IDX = 0
CONF_POOL_DEFAULT_IDX = 4
POOL_TEMP_STEPS = [
    [-100, 10, 2],
    [10, 12, 4],
    [12, 16, 7],
    [16, 24, 12],
    [24, 27, 14],
    [27, 30, 20],
    [30, 99, 24]
]


SOLCAST_SOLAR_DOMAIN = "solcast_solar"
OPEN_METEO_SOLAR_DOMAIN = "open_meteo_solar_forecast"

SENSOR_HOME_AVAILABLE_EXTRA_POWER = "qs_home_extra_available_power"
SENSOR_HOME_CONSUMPTION_POWER = "qs_home_consumption_power"
SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER = "qs_home_non_controlled_consumption_power"
FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER = "sensor."+SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER
SENSOR_LOAD_CURRENT_COMMAND = "qs_load_current_command_sensor"
SENSOR_LOAD_OVERRIDE_STATE = "qs_load_override_state_sensor"

SENSOR_LOAD_BEST_POWER_VALUE = "qs_load_best_power_value"
SENSOR_CONSTRAINT_SENSOR = "qs_constraint_sensor"
SENSOR_CONSTRAINT_SENSOR_VALUE = "qs_constraint_sensor_value"
SENSOR_CONSTRAINT_SENSOR_ENERGY = "qs_constraint_sensor_energy"
SENSOR_CONSTRAINT_SENSOR_COMPLETION = "qs_constraint_sensor_completion"
SENSOR_CONSTRAINT_SENSOR_CHARGE = "qs_constraint_sensor_charge"
SENSOR_CONSTRAINT_SENSOR_ON_OFF = "qs_constraint_sensor_on_off"
SENSOR_CONSTRAINT_SENSOR_CLIMATE = "qs_constraint_sensor_climate"
SENSOR_CONSTRAINT_SENSOR_POOL = "qs_constraint_sensor_pool"

QSForecastHomeNonControlledSensors = {
    "qs_no_control_forecast_15mn": 15*60,
    "qs_no_control_forecast_30mn": 30*60,
    "qs_no_control_forecast_1h": 60*60,
    "qs_no_control_forecast_3h": 3*60*60,
    "qs_no_control_forecast_6h": 6*60*60
}

QSForecastSolarSensors = {
    "qs_solar_forecast_15mn": 15*60,
    "qs_solar_forecast_30mn": 30*60,
    "qs_solar_forecast_1h": 60*60,
    "qs_solar_forecast_3h": 3*60*60,
    "qs_solar_forecast_6h": 6*60*60
}

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
BUTTON_CAR_NEXT_CHARGE_ADD_DEFAULT = "qs_next_car_charge_add_default"


BUTTON_LOAD_MARK_CURRENT_CONSTRAINT_DONE = "qs_load_mark_current_constraint_done"
BUTTON_LOAD_RESET_OVERRIDE_STATE = "qs_load_reset_override_state"

SWITCH_BEST_EFFORT_GREEN_ONLY = "qs_best_effort_green_only"
SWITCH_ENABLE_DEVICE = "qs_enable_device"
SWITCH_POOL_FORCE_WINTER_MODE = "qs_pool_force_winter_mode"


HA_CONSTRAINT_SENSOR_HISTORY = "qs_stored_constraints"
HA_CONSTRAINT_SENSOR_LAST_EXECUTED_CONSTRAINT = "qs_last_executed_constraint"
HA_CONSTRAINT_SENSOR_LOAD_INFO = "qs_stored_load_info"
FLOATING_PERIOD_S = 30 * 3600


# better for serialization that using an IntEnum
CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE = 10
CONSTRAINT_TYPE_MANDATORY_END_TIME  = 9
CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN = 8
CONSTRAINT_TYPE_FILLER = 7
CONSTRAINT_TYPE_FILLER_AUTO = 6

ENTITY_ID_FORMAT = DOMAIN + ".{}"

DEVICE_CHANGE_CONSTRAINT = "change_constraint"
DEVICE_CHANGE_CONSTRAINT_COMPLETED = "change_constraint_completed"

