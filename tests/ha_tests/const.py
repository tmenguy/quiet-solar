"""Constants for quiet_solar HA tests."""

from homeassistant.const import CONF_NAME

from custom_components.quiet_solar.const import (
    CONF_HOME_VOLTAGE,
    CONF_IS_3P,
    CONF_GRID_POWER_SENSOR,
    CONF_GRID_POWER_SENSOR_INVERTED,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CHARGER_DEVICE_WALLBOX,
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER,
    CONF_CHARGER_PAUSE_RESUME_SWITCH,
    CONF_CHARGER_STATUS_SENSOR,
    CONF_CHARGER_PLUGGED,
    CONF_MONO_PHASE,
    CONF_CAR_TRACKER,
    CONF_CAR_PLUGGED,
    CONF_CAR_CHARGE_PERCENT_SENSOR,
    CONF_CAR_BATTERY_CAPACITY,
    CONF_CAR_CHARGER_MIN_CHARGE,
    CONF_CAR_CHARGER_MAX_CHARGE,
    CONF_DEFAULT_CAR_CHARGE,
    CONF_MINIMUM_OK_CAR_CHARGE,
    CONF_PERSON_PERSON_ENTITY,
    CONF_PERSON_AUTHORIZED_CARS,
    CONF_PERSON_PREFERRED_CAR,
    CONF_BATTERY_CHARGE_DISCHARGE_SENSOR,
    CONF_BATTERY_CHARGE_PERCENT_SENSOR,
    CONF_BATTERY_CAPACITY,
    CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE,
    CONF_BATTERY_MAX_CHARGE_POWER_VALUE,
    CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR,
    CONF_DYN_GROUP_MAX_PHASE_AMPS,
    DEVICE_TYPE,
    CONF_TYPE_NAME_QSHome,
    CONF_TYPE_NAME_QSCar,
    CONF_TYPE_NAME_QSChargerWallbox,
    CONF_TYPE_NAME_QSChargerGeneric,
    CONF_TYPE_NAME_QSPerson,
    CONF_TYPE_NAME_QSBattery,
    CONF_TYPE_NAME_QSSolar,
    CONF_TYPE_NAME_QSDynamicGroup,
    CONF_TYPE_NAME_QSHeatPump,
)

# Mock Home configuration
MOCK_HOME_CONFIG = {
    CONF_NAME: "Test Home",
    DEVICE_TYPE: CONF_TYPE_NAME_QSHome,
    CONF_HOME_VOLTAGE: 230,
    CONF_IS_3P: True,
    CONF_GRID_POWER_SENSOR: "sensor.grid_power",
    CONF_GRID_POWER_SENSOR_INVERTED: False,
    CONF_DYN_GROUP_MAX_PHASE_AMPS: 63,
}

# Mock Car configuration
MOCK_CAR_CONFIG = {
    CONF_NAME: "Test Car",
    DEVICE_TYPE: CONF_TYPE_NAME_QSCar,
    CONF_CAR_TRACKER: "device_tracker.test_car",
    CONF_CAR_PLUGGED: "binary_sensor.test_car_plugged",
    CONF_CAR_CHARGE_PERCENT_SENSOR: "sensor.test_car_soc",
    CONF_CAR_BATTERY_CAPACITY: 60000,  # 60 kWh in Wh
    CONF_CAR_CHARGER_MIN_CHARGE: 6,
    CONF_CAR_CHARGER_MAX_CHARGE: 32,
    CONF_DEFAULT_CAR_CHARGE: 80.0,
    CONF_MINIMUM_OK_CAR_CHARGE: 20.0,
}

# Mock Charger configuration (Generic type - doesn't require device lookup)
MOCK_CHARGER_CONFIG = {
    CONF_NAME: "Test Charger",
    DEVICE_TYPE: CONF_TYPE_NAME_QSChargerGeneric,
    CONF_CHARGER_MIN_CHARGE: 6,
    CONF_CHARGER_MAX_CHARGE: 32,
    CONF_IS_3P: False,
    CONF_MONO_PHASE: 1,
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER: "number.test_charger_max_current",
    CONF_CHARGER_PAUSE_RESUME_SWITCH: "switch.test_charger_pause_resume",
    CONF_CHARGER_STATUS_SENSOR: "sensor.test_charger_status",
    CONF_CHARGER_PLUGGED: "binary_sensor.test_charger_plugged",
}

# Mock Person configuration
MOCK_PERSON_CONFIG = {
    CONF_NAME: "Test Person",
    DEVICE_TYPE: CONF_TYPE_NAME_QSPerson,
    CONF_PERSON_PERSON_ENTITY: "person.test_person",
    CONF_PERSON_AUTHORIZED_CARS: ["Test Car"],
    CONF_PERSON_PREFERRED_CAR: "Test Car",
}

# Mock Battery configuration
MOCK_BATTERY_CONFIG = {
    CONF_NAME: "Test Battery",
    DEVICE_TYPE: CONF_TYPE_NAME_QSBattery,
    CONF_BATTERY_CHARGE_DISCHARGE_SENSOR: "sensor.battery_power",
    CONF_BATTERY_CHARGE_PERCENT_SENSOR: "sensor.battery_soc",
    CONF_BATTERY_CAPACITY: 10000,  # 10 kWh in Wh
    CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
    CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
}

# Mock Solar configuration
MOCK_SOLAR_CONFIG = {
    CONF_NAME: "Test Solar",
    DEVICE_TYPE: CONF_TYPE_NAME_QSSolar,
    CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
}

# Mock Dynamic Group configuration
MOCK_DYNAMIC_GROUP_CONFIG = {
    CONF_NAME: "Test Charger Group",
    DEVICE_TYPE: CONF_TYPE_NAME_QSDynamicGroup,
    CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
    CONF_IS_3P: True,
}

# Mock Heat Pump configuration (PilotedDevice)
MOCK_HEAT_PUMP_CONFIG = {
    CONF_NAME: "Test Heat Pump",
    DEVICE_TYPE: CONF_TYPE_NAME_QSHeatPump,
}

# Entry IDs for testing
MOCK_HOME_ENTRY_ID = "home_entry_123"
MOCK_CAR_ENTRY_ID = "car_entry_123"
MOCK_CHARGER_ENTRY_ID = "charger_entry_123"
MOCK_PERSON_ENTRY_ID = "person_entry_123"
MOCK_BATTERY_ENTRY_ID = "battery_entry_123"
MOCK_SOLAR_ENTRY_ID = "solar_entry_123"
MOCK_DYNAMIC_GROUP_ENTRY_ID = "dynamic_group_entry_123"
MOCK_HEAT_PUMP_ENTRY_ID = "heat_pump_entry_123"

# Mock sensor states for testing
MOCK_SENSOR_STATES = {
    "sensor.grid_power": {"state": "500", "attributes": {"unit_of_measurement": "W"}},
    "sensor.solar_power": {"state": "3000", "attributes": {"unit_of_measurement": "W"}},
    "sensor.battery_power": {"state": "-1000", "attributes": {"unit_of_measurement": "W"}},
    "sensor.battery_soc": {"state": "75", "attributes": {"unit_of_measurement": "%"}},
    "sensor.test_car_soc": {"state": "50", "attributes": {"unit_of_measurement": "%"}},
    "device_tracker.test_car": {"state": "home", "attributes": {"latitude": 48.8566, "longitude": 2.3522}},
    "binary_sensor.test_car_plugged": {"state": "on", "attributes": {}},
    "person.test_person": {"state": "home", "attributes": {}},
    "zone.home": {"state": "zoning", "attributes": {"latitude": 48.8566, "longitude": 2.3522, "radius": 100}},
    # Charger entities
    "number.test_charger_max_current": {"state": "32", "attributes": {"min": 6, "max": 32, "step": 1, "unit_of_measurement": "A"}},
    "switch.test_charger_pause_resume": {"state": "on", "attributes": {}},
    "sensor.test_charger_status": {"state": "Charging", "attributes": {}},
    "binary_sensor.test_charger_plugged": {"state": "on", "attributes": {}},
}
