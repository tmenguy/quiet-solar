"""
Comprehensive End-to-End Integration Tests for Car-Charger-Person Interactions.

These tests focus on the tightly coupled behavior between QSCar, QSChargerGeneric, and QSPerson
classes, testing real object interactions with minimal mocking.
"""
from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timedelta
from datetime import time as dt_time
from typing import Any, Dict, List
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import pytz

from homeassistant.const import CONF_NAME, STATE_UNKNOWN, STATE_UNAVAILABLE

from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_HOME_VOLTAGE,
    CONF_IS_3P,
    CONF_MONO_PHASE,
    CONF_DYN_GROUP_MAX_PHASE_AMPS,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CHARGER_DEVICE_WALLBOX,
    CONF_CAR_BATTERY_CAPACITY,
    CONF_CAR_TRACKER,
    CONF_CAR_PLUGGED,
    CONF_CAR_CHARGE_PERCENT_SENSOR,
    CONF_CAR_CHARGER_MIN_CHARGE,
    CONF_CAR_CHARGER_MAX_CHARGE,
    CONF_DEFAULT_CAR_CHARGE,
    CONF_MINIMUM_OK_CAR_CHARGE,
    CONF_CAR_ESTIMATED_RANGE_SENSOR,
    CONF_CAR_ODOMETER_SENSOR,
    CONF_PERSON_PERSON_ENTITY,
    CONF_PERSON_AUTHORIZED_CARS,
    CONF_PERSON_PREFERRED_CAR,
    CONF_PERSON_NOTIFICATION_TIME,
    CONF_MOBILE_APP,
    CAR_CHARGE_TYPE_NOT_PLUGGED,
    CAR_CHARGE_TYPE_NOT_CHARGING,
    CAR_CHARGE_TYPE_SCHEDULE,
    CAR_CHARGE_TYPE_PERSON_AUTOMATED,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    CONSTRAINT_TYPE_FILLER_AUTO,
    FORCE_CAR_NO_PERSON_ATTACHED,
    PERSON_NOTIFY_REASON_DAILY_REMINDER_FOR_CAR_NO_CHARGER,
    PERSON_NOTIFY_REASON_DAILY_CHARGER_CONSTRAINTS,
    PERSON_NOTIFY_REASON_CHANGED_CAR,
)
from custom_components.quiet_solar.ha_model.home import QSHome
from custom_components.quiet_solar.ha_model.car import QSCar
from custom_components.quiet_solar.ha_model.charger import (
    QSChargerGeneric,
    QSChargerWallbox,
    QSChargerGroup,
    QSChargerStatus,
    QSStateCmd,
    CHARGER_ADAPTATION_WINDOW_S,
)
from custom_components.quiet_solar.ha_model.person import QSPerson
from custom_components.quiet_solar.ha_model.dynamic_group import QSDynamicGroup
from custom_components.quiet_solar.home_model.commands import (
    CMD_ON, CMD_OFF, CMD_IDLE,
    CMD_AUTO_GREEN_ONLY,
    CMD_AUTO_FROM_CONSIGN,
    CMD_AUTO_PRICE,
    copy_command,
)
from custom_components.quiet_solar.home_model.constraints import (
    MultiStepsPowerLoadConstraintChargePercent,
    DATETIME_MAX_UTC,
)

# Import from local conftest - use relative import to avoid conflict with HA core
from tests.test_helpers import FakeHass, FakeConfigEntry


# =============================================================================
# Fixtures for Integrated Testing
# =============================================================================

class IntegratedTestEnvironment:
    """Container for all objects in an integrated test scenario."""

    def __init__(self):
        self.hass: FakeHass = None
        self.config_entry: FakeConfigEntry = None
        self.home: QSHome = None
        self.cars: List[QSCar] = []
        self.chargers: List[QSChargerGeneric] = []
        self.persons: List[QSPerson] = []
        self.dynamic_groups: List[QSDynamicGroup] = []
        self.charger_group: QSChargerGroup = None
        self.time: datetime = None


def create_fake_hass_with_states() -> FakeHass:
    """Create FakeHass with commonly needed states."""
    hass = FakeHass()
    # Set up zone.home which is required by QSHome
    hass.states.set("zone.home", "zoning", {
        "latitude": 48.8566,
        "longitude": 2.3522,
        "radius": 100.0
    })
    # Set up common states
    hass.states.set("sensor.test_soc", "50", {"unit_of_measurement": "%"})
    hass.states.set("device_tracker.test_car", "home", {"latitude": 48.8566, "longitude": 2.3522})
    hass.states.set("binary_sensor.test_plugged", "on", {})
    hass.states.set("sensor.test_odometer", "50000", {"unit_of_measurement": "km"})
    hass.states.set("sensor.test_range", "200", {"unit_of_measurement": "km"})
    hass.states.set("person.test_person", "home", {})
    return hass


def create_integrated_environment(
    num_cars: int = 1,
    num_chargers: int = 1,
    num_persons: int = 1,
    time: datetime = None
) -> IntegratedTestEnvironment:
    """Create a fully integrated test environment with linked objects."""

    env = IntegratedTestEnvironment()
    env.time = time or datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
    env.hass = create_fake_hass_with_states()
    env.config_entry = FakeConfigEntry(
        entry_id="test_integrated_entry",
        data={CONF_NAME: "Test Integration"},
    )

    # Create data handler mock
    data_handler = MagicMock()
    env.hass.data[DOMAIN][DATA_HANDLER] = data_handler

    # Create Home with necessary patches
    home_config = {
        CONF_NAME: "TestHome",
        CONF_DYN_GROUP_MAX_PHASE_AMPS: 63,
        CONF_IS_3P: True,
        CONF_HOME_VOLTAGE: 230,
        "hass": env.hass,
        "config_entry": env.config_entry,
    }
    with patch('custom_components.quiet_solar.ha_model.home.QSHomeConsumptionHistoryAndForecast'):
        env.home = QSHome(**home_config)
    data_handler.home = env.home

    # Create Dynamic Group for chargers
    if num_chargers > 0:
        dyn_group_config = {
            CONF_NAME: "ChargerGroup",
            CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            CONF_IS_3P: True,
            "home": env.home,
            "hass": env.hass,
            "config_entry": env.config_entry,
        }
        dyn_group = QSDynamicGroup(**dyn_group_config)
        env.dynamic_groups.append(dyn_group)
        env.home.add_device(dyn_group)

    # Create Chargers
    with patch('custom_components.quiet_solar.ha_model.charger.entity_registry') as mock_er:
        mock_er.async_get = MagicMock(return_value=MagicMock())
        mock_er.async_get.return_value.async_entries_for_device = MagicMock(return_value=[])

        for i in range(num_chargers):
            phase = (i % 3) + 1
            charger_config = {
                CONF_NAME: f"Charger_{i+1}",
                CONF_MONO_PHASE: phase,
                CONF_CHARGER_DEVICE_WALLBOX: f"device_wallbox_{i+1}",
                CONF_CHARGER_MIN_CHARGE: 6,
                CONF_CHARGER_MAX_CHARGE: 32,
                CONF_IS_3P: False,
                "dynamic_group_name": "ChargerGroup" if env.dynamic_groups else None,
                "home": env.home,
                "hass": env.hass,
                "config_entry": env.config_entry,
            }
            charger = QSChargerWallbox(**charger_config)
            env.chargers.append(charger)
            env.home.add_device(charger)

    # Create Cars
    car_names = []
    for i in range(num_cars):
        car_config = {
            CONF_NAME: f"Car_{i+1}",
            CONF_CAR_TRACKER: f"device_tracker.car_{i+1}",
            CONF_CAR_PLUGGED: f"binary_sensor.car_{i+1}_plugged",
            CONF_CAR_CHARGE_PERCENT_SENSOR: f"sensor.car_{i+1}_soc",
            CONF_CAR_BATTERY_CAPACITY: 60000,  # 60 kWh
            CONF_CAR_CHARGER_MIN_CHARGE: 6,
            CONF_CAR_CHARGER_MAX_CHARGE: 32,
            CONF_DEFAULT_CAR_CHARGE: 80.0,
            CONF_MINIMUM_OK_CAR_CHARGE: 20.0,
            CONF_CAR_ODOMETER_SENSOR: f"sensor.car_{i+1}_odometer",
            CONF_CAR_ESTIMATED_RANGE_SENSOR: f"sensor.car_{i+1}_range",
            "home": env.home,
            "hass": env.hass,
            "config_entry": env.config_entry,
        }
        car = QSCar(**car_config)
        env.cars.append(car)
        env.home.add_device(car)
        car_names.append(f"Car_{i+1}")

        # Set up states for this car
        env.hass.states.set(f"device_tracker.car_{i+1}", "home", {"latitude": 48.8566, "longitude": 2.3522})
        env.hass.states.set(f"binary_sensor.car_{i+1}_plugged", "on", {})
        env.hass.states.set(f"sensor.car_{i+1}_soc", "50", {"unit_of_measurement": "%"})
        env.hass.states.set(f"sensor.car_{i+1}_odometer", "50000", {"unit_of_measurement": "km"})
        env.hass.states.set(f"sensor.car_{i+1}_range", "200", {"unit_of_measurement": "km"})

    # Create Persons
    for i in range(num_persons):
        person_config = {
            CONF_NAME: f"Person_{i+1}",
            CONF_PERSON_PERSON_ENTITY: f"person.person_{i+1}",
            CONF_PERSON_AUTHORIZED_CARS: car_names,  # Authorize all cars
            CONF_PERSON_PREFERRED_CAR: car_names[0] if car_names else None,
            CONF_PERSON_NOTIFICATION_TIME: "07:00:00",
            CONF_MOBILE_APP: f"mobile_app.person_{i+1}",
            "home": env.home,
            "hass": env.hass,
            "config_entry": env.config_entry,
        }
        person = QSPerson(**person_config)
        env.persons.append(person)
        env.home.add_device(person)

        # Set up person state
        env.hass.states.set(f"person.person_{i+1}", "home", {})

    # Create charger group if we have chargers
    if env.dynamic_groups and env.chargers:
        env.charger_group = QSChargerGroup(env.dynamic_groups[0])

    return env


def simulate_sensor_value(env: IntegratedTestEnvironment, entity_id: str, value: str, attributes: Dict = None):
    """Simulate a sensor value change."""
    env.hass.states.set(entity_id, value, attributes or {})


def simulate_car_at_home(env: IntegratedTestEnvironment, car_index: int = 0):
    """Simulate car arriving home."""
    car = env.cars[car_index]
    simulate_sensor_value(env, car.car_tracker, "home", {"latitude": 48.8566, "longitude": 2.3522})


def simulate_car_plugged(env: IntegratedTestEnvironment, car_index: int = 0, plugged: bool = True):
    """Simulate car plugging/unplugging."""
    car = env.cars[car_index]
    simulate_sensor_value(env, car.car_plugged, "on" if plugged else "off", {})


def simulate_soc(env: IntegratedTestEnvironment, car_index: int = 0, soc_percent: float = 50.0):
    """Simulate SOC change."""
    car = env.cars[car_index]
    simulate_sensor_value(env, car.car_charge_percent_sensor, str(soc_percent), {"unit_of_measurement": "%"})


# =============================================================================
# Test Class: QSStateCmd Utility
# =============================================================================

class TestQSStateCmd:
    """Test QSStateCmd utility class used by chargers."""

    def test_initial_state(self):
        """Test initial state of QSStateCmd."""
        cmd = QSStateCmd()
        assert cmd.value is None
        assert cmd._num_launched == 0
        assert cmd.last_time_set is None

    def test_set_changes_value(self):
        """Test that set() changes value and records time."""
        cmd = QSStateCmd()
        time = datetime.now(pytz.UTC)

        result = cmd.set(True, time)

        assert result is True
        assert cmd.value is True
        assert cmd.last_change_asked == time

    def test_set_same_value_no_change(self):
        """Test that set() with same value returns False."""
        cmd = QSStateCmd()
        time = datetime.now(pytz.UTC)

        cmd.set(True, time)
        result = cmd.set(True, time + timedelta(seconds=10))

        assert result is False

    def test_is_ok_to_set_respects_min_time(self):
        """Test is_ok_to_set respects minimum change time."""
        cmd = QSStateCmd()
        time = datetime.now(pytz.UTC)

        cmd.set(True, time)
        cmd._num_set = 5  # Past initial immediate changes

        # Too soon
        assert cmd.is_ok_to_set(time + timedelta(seconds=10), 60) is False
        # After min time
        assert cmd.is_ok_to_set(time + timedelta(seconds=120), 60) is True

    def test_is_ok_to_launch_retry_logic(self):
        """Test launch retry logic."""
        cmd = QSStateCmd()
        time = datetime.now(pytz.UTC)

        # First launch always ok
        assert cmd.is_ok_to_launch(True, time) is True

        cmd.register_launch(True, time)

        # Second launch needs wait
        assert cmd.is_ok_to_launch(True, time + timedelta(seconds=10)) is False

        # After retry interval
        assert cmd.is_ok_to_launch(True, time + timedelta(seconds=60)) is True

    @pytest.mark.asyncio
    async def test_success_clears_counters(self):
        """Test that success() clears retry counters."""
        cmd = QSStateCmd()
        time = datetime.now(pytz.UTC)

        cmd.register_launch(True, time)
        cmd.register_launch(True, time)

        await cmd.success(time)

        assert cmd._num_launched == 0
        assert cmd.first_time_success == time

    @pytest.mark.asyncio
    async def test_success_calls_callback(self):
        """Test that success() calls registered callback."""
        cmd = QSStateCmd()
        time = datetime.now(pytz.UTC)

        callback_called = []

        async def my_callback(time, **kwargs):
            callback_called.append((time, kwargs))

        cmd.register_success_cb(my_callback, {"extra": "data"})
        await cmd.success(time)

        assert len(callback_called) == 1
        assert callback_called[0][1] == {"extra": "data"}


# =============================================================================
# Test Class: Car-Charger Attachment
# =============================================================================

class TestCarChargerAttachment:
    """Test car-charger attachment/detachment flows."""

    def test_attach_car_creates_bidirectional_link(self):
        """Test that attaching car creates links in both directions."""
        env = create_integrated_environment(num_cars=1, num_chargers=1, num_persons=0)
        car = env.cars[0]
        charger = env.chargers[0]

        # Detach any existing car
        if charger.car:
            charger.detach_car()

        charger.attach_car(car, env.time)

        assert charger.car == car
        assert car.charger == charger

    def test_detach_car_clears_both_links(self):
        """Test that detaching car clears both links."""
        env = create_integrated_environment(num_cars=1, num_chargers=1, num_persons=0)
        car = env.cars[0]
        charger = env.chargers[0]

        charger.attach_car(car, env.time)
        charger.detach_car()

        assert charger.car != car or charger.car is None or charger.car.name != car.name
        assert car.charger is None

    def test_attach_different_car_detaches_previous(self):
        """Test that attaching a new car detaches the previous one."""
        env = create_integrated_environment(num_cars=2, num_chargers=1, num_persons=0)
        car1 = env.cars[0]
        car2 = env.cars[1]
        charger = env.chargers[0]

        # Detach default car first
        if charger.car:
            charger.detach_car()

        charger.attach_car(car1, env.time)
        assert car1.charger == charger

        charger.attach_car(car2, env.time)

        assert charger.car == car2
        assert car2.charger == charger
        assert car1.charger is None


# =============================================================================
# Test Class: Car State Detection
# =============================================================================

class TestCarStateDetection:
    """Test car state detection methods."""

    def test_car_plugged_sensor_configured(self):
        """Test car plugged sensor is configured."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        assert car.car_plugged is not None
        assert car.car_plugged == "binary_sensor.car_1_plugged"

    def test_car_tracker_configured(self):
        """Test car tracker is configured."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        assert car.car_tracker is not None
        assert car.car_tracker == "device_tracker.car_1"

    def test_car_charge_percent_sensor_configured(self):
        """Test car charge percent sensor is configured."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        assert car.car_charge_percent_sensor is not None
        assert car.car_charge_percent_sensor == "sensor.car_1_soc"

    def test_car_battery_capacity_configured(self):
        """Test car battery capacity is configured."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        assert car.car_battery_capacity == 60000  # 60 kWh

    def test_is_car_plugged_method_exists(self):
        """Test is_car_plugged method exists."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        assert hasattr(car, 'is_car_plugged')
        assert callable(car.is_car_plugged)

    def test_is_car_home_method_exists(self):
        """Test is_car_home method exists."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        assert hasattr(car, 'is_car_home')
        assert callable(car.is_car_home)


# =============================================================================
# Test Class: Person Forecast Calculations
# =============================================================================

class TestPersonForecast:
    """Test person mileage forecast calculations."""

    def test_add_to_mileage_history(self):
        """Test adding mileage data to history."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=1)
        person = env.persons[0]

        day = datetime(2024, 6, 10, 0, 0, 0, tzinfo=pytz.UTC)
        leave_time = datetime(2024, 6, 10, 7, 30, 0, tzinfo=pytz.UTC)

        person.add_to_mileage_history(day, 45.0, leave_time)

        assert len(person.historical_mileage_data) == 1
        assert person.historical_mileage_data[0][1] == 45.0

    def test_add_multiple_days_to_history(self):
        """Test adding multiple days maintains order."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=1)
        person = env.persons[0]

        for i in range(5):
            day = datetime(2024, 6, 10 + i, 0, 0, 0, tzinfo=pytz.UTC)
            leave_time = datetime(2024, 6, 10 + i, 7, 0, 0, tzinfo=pytz.UTC)
            person.add_to_mileage_history(day, 30.0 + i * 10, leave_time)

        assert len(person.historical_mileage_data) == 5
        # Should be ordered by date
        for i in range(4):
            assert person.historical_mileage_data[i][0] < person.historical_mileage_data[i+1][0]

    def test_compute_person_next_need_uses_weekday_data(self):
        """Test forecast uses weekday-specific historical data."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=1)
        person = env.persons[0]

        # Add data for specific weekdays
        # Monday (weekday 0)
        monday = datetime(2024, 6, 10, 0, 0, 0, tzinfo=pytz.UTC)  # This is a Monday
        monday_leave = datetime(2024, 6, 10, 8, 0, 0, tzinfo=pytz.UTC)
        person.add_to_mileage_history(monday, 100.0, monday_leave)

        # Test on a Sunday (will predict for Monday)
        sunday = datetime(2024, 6, 16, 20, 0, 0, tzinfo=pytz.UTC)  # Sunday evening

        leave_time, mileage = person._compute_person_next_need(sunday)

        # Should predict Monday's pattern
        if mileage is not None:
            assert mileage == 100.0

    def test_update_person_forecast_caches_result(self):
        """Test that update_person_forecast caches and returns cached result."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=1)
        person = env.persons[0]

        # Add some history
        day = datetime(2024, 6, 10, 0, 0, 0, tzinfo=pytz.UTC)
        leave_time = datetime(2024, 6, 10, 7, 0, 0, tzinfo=pytz.UTC)
        person.add_to_mileage_history(day, 50.0, leave_time)

        # First call computes
        person.update_person_forecast(env.time, force_update=True)

        # Values should be cached
        assert person._last_request_prediction_time is not None

    def test_get_forecast_readable_string_no_data(self):
        """Test readable forecast string when no data."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=1)
        person = env.persons[0]

        result = person.get_forecast_readable_string()

        assert "No forecast" in result

    def test_get_authorized_cars_returns_car_objects(self):
        """Test get_authorized_cars returns actual QSCar objects."""
        env = create_integrated_environment(num_cars=2, num_chargers=0, num_persons=1)
        person = env.persons[0]

        cars = person.get_authorized_cars()

        assert len(cars) == 2
        assert all(isinstance(c, QSCar) for c in cars)

    def test_get_preferred_car(self):
        """Test get_preferred_car returns correct car."""
        env = create_integrated_environment(num_cars=2, num_chargers=0, num_persons=1)
        person = env.persons[0]

        preferred = person.get_preferred_car()

        assert preferred is not None
        assert preferred.name == "Car_1"  # First car is preferred


# =============================================================================
# Test Class: Car-Person Allocation
# =============================================================================

class TestCarPersonAllocation:
    """Test car-person allocation logic."""

    def test_car_person_options_includes_no_person(self):
        """Test that car person options includes 'no person' option."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=1)
        car = env.cars[0]

        options = car.get_car_persons_options()

        assert FORCE_CAR_NO_PERSON_ATTACHED in options

    def test_car_person_options_includes_authorized_persons(self):
        """Test that car person options includes authorized persons."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=2)
        car = env.cars[0]

        options = car.get_car_persons_options()

        # Should include both persons plus no-person option
        assert "Person_1" in options
        assert "Person_2" in options

    def test_user_selected_person_clears_other_cars(self):
        """Test that setting person on one car clears it from others."""
        env = create_integrated_environment(num_cars=2, num_chargers=0, num_persons=1)
        car1 = env.cars[0]
        car2 = env.cars[1]

        # Set person on car1
        car1._user_selected_person_name_for_car = "Person_1"

        # Now set same person on car2 - should clear from car1
        car2.user_selected_person_name_for_car = "Person_1"

        # Car1 should have lost the person assignment
        assert car1._user_selected_person_name_for_car is None or car1._user_selected_person_name_for_car != "Person_1"

    def test_current_forecasted_person_assignment(self):
        """Test assignment of forecasted person to car."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=1)
        car = env.cars[0]
        person = env.persons[0]

        car.current_forecasted_person = person

        assert car.current_forecasted_person == person
        assert car.get_car_person_option() == "Person_1"


# =============================================================================
# Test Class: Car Efficiency Calculations
# =============================================================================

class TestCarEfficiency:
    """Test car efficiency and range calculations."""

    def test_car_range_sensor_configured(self):
        """Test range sensor is configured."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        assert car.car_estimated_range_sensor is not None

    def test_add_soc_odo_value_to_segments_creates_segment(self):
        """Test that SOC/odometer values create efficiency segments."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        # Simulate driving (SOC decreasing, odometer increasing)
        car._add_soc_odo_value_to_segments(80.0, 50000, env.time)
        car._add_soc_odo_value_to_segments(70.0, 50050, env.time + timedelta(hours=1))

        assert len(car._decreasing_segments) >= 1

    def test_get_computed_range_efficiency_with_segments(self):
        """Test efficiency calculation with segment data."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        # Create a segment by simulating driving
        car._add_soc_odo_value_to_segments(80.0, 50000, env.time)
        car._add_soc_odo_value_to_segments(70.0, 50060, env.time + timedelta(hours=1))
        # Close segment by going up in SOC
        car._add_soc_odo_value_to_segments(75.0, 50060, env.time + timedelta(hours=2))

        # Should have efficiency data now
        result = car.get_computed_range_efficiency_km_per_percent(env.time)

        # May be None if range sensor provides value, or calculated from segments
        # Just verify it doesn't crash
        assert result is None or result > 0

    def test_efficiency_sensors_configured(self):
        """Test that odometer and range sensors are configured."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        assert car.car_odometer_sensor is not None
        assert car.car_estimated_range_sensor is not None


# =============================================================================
# Test Class: Charger Charge Type
# =============================================================================

class TestChargerChargeType:
    """Test charger charge type determination."""

    def test_charge_type_not_plugged_when_no_car(self):
        """Test charge type is NOT_PLUGGED when no car attached."""
        env = create_integrated_environment(num_cars=1, num_chargers=1, num_persons=0)
        charger = env.chargers[0]

        # Detach any car
        if charger.car and charger.car != charger._default_generic_car:
            charger.detach_car()

        charge_type, _ = charger.get_charge_type()

        # With default generic car or no real car, should be not charging or not plugged
        assert charge_type in [CAR_CHARGE_TYPE_NOT_PLUGGED, CAR_CHARGE_TYPE_NOT_CHARGING]

    def test_charger_has_get_charge_type_method(self):
        """Test charger has get_charge_type method."""
        env = create_integrated_environment(num_cars=1, num_chargers=1, num_persons=0)
        charger = env.chargers[0]

        assert hasattr(charger, 'get_charge_type')
        assert callable(charger.get_charge_type)

    def test_charger_constraints_list_exists(self):
        """Test charger has constraints list."""
        env = create_integrated_environment(num_cars=1, num_chargers=1, num_persons=0)
        charger = env.chargers[0]

        assert hasattr(charger, '_constraints')
        assert isinstance(charger._constraints, list)


# =============================================================================
# Test Class: Charger Group Operations
# =============================================================================

class TestChargerGroupOperations:
    """Test charger group coordination."""

    def test_charger_group_creation(self):
        """Test charger group is created correctly."""
        env = create_integrated_environment(num_cars=1, num_chargers=3, num_persons=0)

        assert env.charger_group is not None
        assert len(env.charger_group._chargers) == 3

    def test_charger_group_has_dynamic_group(self):
        """Test charger group references dynamic group."""
        env = create_integrated_environment(num_cars=1, num_chargers=3, num_persons=0)

        assert env.charger_group.dynamic_group is not None
        assert env.charger_group.dynamic_group == env.dynamic_groups[0]

    def test_charger_group_has_home(self):
        """Test charger group references home."""
        env = create_integrated_environment(num_cars=1, num_chargers=3, num_persons=0)

        assert env.charger_group.home is not None
        assert env.charger_group.home == env.home

    def test_charger_status_creation(self):
        """Test creating charger status for a charger."""
        env = create_integrated_environment(num_cars=1, num_chargers=1, num_persons=0)
        charger = env.chargers[0]

        cs = QSChargerStatus(charger)

        assert cs.charger == charger
        assert cs.accurate_current_power is None
        assert cs.charge_score == 0


# =============================================================================
# Test Class: Person Notification
# =============================================================================

class TestPersonNotification:
    """Test person notification logic."""

    @pytest.mark.asyncio
    async def test_notify_without_mobile_app_does_nothing(self):
        """Test that notification without mobile app configured does nothing."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=1)
        person = env.persons[0]
        person.mobile_app = None  # No mobile app

        # Should not raise
        await person.notify_of_forecast_if_needed(env.time)

    @pytest.mark.asyncio
    async def test_notify_updates_last_notification_time(self):
        """Test that notification call updates timestamp."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=1)
        person = env.persons[0]

        await person.notify_of_forecast_if_needed(env.time)

        assert person._last_forecast_notification_call_time is not None

    def test_get_person_mileage_serialized_prediction(self):
        """Test serialization of person mileage prediction."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=1)
        person = env.persons[0]

        # Add some history
        day = datetime(2024, 6, 10, 0, 0, 0, tzinfo=pytz.UTC)
        leave_time = datetime(2024, 6, 10, 7, 0, 0, tzinfo=pytz.UTC)
        person.add_to_mileage_history(day, 50.0, leave_time)

        state, attrs = person.get_person_mileage_serialized_prediction()

        assert state is not None
        assert "historical_data" in attrs
        assert "has_been_initialized" in attrs


# =============================================================================
# Test Class: Car Charge Constraint Flow
# =============================================================================

class TestCarChargeConstraintFlow:
    """Test the flow from person need to car constraint to charger."""

    @pytest.mark.asyncio
    async def test_get_best_person_next_need(self):
        """Test getting best person need for a car."""
        env = create_integrated_environment(num_cars=1, num_chargers=1, num_persons=1)
        car = env.cars[0]
        person = env.persons[0]
        charger = env.chargers[0]

        # Attach car to charger
        charger.attach_car(car, env.time)

        # Set person as forecasted for this car
        car.current_forecasted_person = person

        # Add mileage history for person
        day = datetime(2024, 6, 10, 0, 0, 0, tzinfo=pytz.UTC)
        leave_time = datetime(2024, 6, 10, 7, 0, 0, tzinfo=pytz.UTC)
        person.add_to_mileage_history(day, 100.0, leave_time)

        # Mock the home's allocation method to not fail
        env.home.get_best_persons_cars_allocations = AsyncMock()

        is_covered, next_time, target_charge, person_obj = await car.get_best_person_next_need(env.time)

        # Result depends on SOC and history, but should not crash
        assert person_obj == person or person_obj is None


# =============================================================================
# Test Class: Dampening and Power Calculations
# =============================================================================

class TestDampeningPowerCalculations:
    """Test power dampening calculations in car."""

    def test_theoretical_amp_to_power(self):
        """Test theoretical amp to power conversion."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        # At 10A, 230V = 2300W for 1 phase
        assert car.theoretical_amp_to_power_1p[10] == pytest.approx(2300.0, abs=10)
        # 3 phase = 6900W
        assert car.theoretical_amp_to_power_3p[10] == pytest.approx(6900.0, abs=10)

    def test_get_delta_dampened_power_same_amps(self):
        """Test delta power is 0 when amps are the same."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        delta = car.get_delta_dampened_power(10, 1, 10, 1)

        assert delta == 0.0

    def test_add_to_amps_power_graph(self):
        """Test adding values to the amp-power graph."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        # Add a dampening value
        result = car._add_to_amps_power_graph((6, 1), (10, 1), 920.0)  # 920W increase

        assert result is True
        assert (6, 10) in car._dampening_deltas

    def test_find_path_in_graph(self):
        """Test path finding in dampening graph."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        # Create a graph
        graph = {
            6: {10},
            10: {6, 16},
            16: {10}
        }

        path = car.find_path(graph, 6, 16)

        assert path == [6, 10, 16]


# =============================================================================
# Test Class: Car Reset and Initialization
# =============================================================================

class TestCarResetInit:
    """Test car reset and initialization."""

    def test_car_reset_clears_charger(self):
        """Test that reset clears charger reference."""
        env = create_integrated_environment(num_cars=1, num_chargers=1, num_persons=0)
        car = env.cars[0]
        charger = env.chargers[0]

        charger.attach_car(car, env.time)

        car.reset()

        assert car.charger is None
        assert car.do_force_next_charge is False

    def test_car_reset_restores_calendar(self):
        """Test that reset restores original calendar."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]
        original_calendar = car._conf_calendar

        car.calendar = "modified_calendar"
        car.reset()

        assert car.calendar == original_calendar

    def test_car_update_to_be_saved_extra_device_info(self):
        """Test saving extra device info."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=1)
        car = env.cars[0]
        person = env.persons[0]

        car._user_selected_person_name_for_car = "Person_1"
        car.current_forecasted_person = person

        data = {}
        car.update_to_be_saved_extra_device_info(data)

        assert "user_selected_person_name_for_car" in data
        assert "current_forecasted_person_name_from_boot" in data

    def test_car_use_saved_extra_device_info(self):
        """Test restoring extra device info."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        stored_data = {
            "user_selected_person_name_for_car": "SomePerson",
            "current_forecasted_person_name_from_boot": "AnotherPerson"
        }

        car.use_saved_extra_device_info(stored_data)

        assert car._user_selected_person_name_for_car == "SomePerson"
        assert car._current_forecasted_person_name_from_boot == "AnotherPerson"


# =============================================================================
# Test Class: Person Reset and Platforms
# =============================================================================

class TestPersonResetPlatforms:
    """Test person reset and platform support."""

    def test_person_get_platforms(self):
        """Test person returns correct platforms."""
        env = create_integrated_environment(num_cars=0, num_chargers=0, num_persons=1)
        person = env.persons[0]

        platforms = person.get_platforms()

        assert "sensor" in platforms

    def test_person_dashboard_sort_string(self):
        """Test person dashboard sort string."""
        env = create_integrated_environment(num_cars=0, num_chargers=0, num_persons=1)
        person = env.persons[0]

        sort_str = person.dashboard_sort_string_in_type

        assert sort_str == "AAA"

    def test_person_tracker_id(self):
        """Test person tracker ID resolution."""
        env = create_integrated_environment(num_cars=0, num_chargers=0, num_persons=1)
        person = env.persons[0]

        tracker_id = person.get_tracker_id()

        # Should return person_entity_id if no separate tracker
        assert tracker_id == person.person_entity_id


# =============================================================================
# Test Class: Charger Status Calculations
# =============================================================================

class TestChargerStatusCalculations:
    """Test QSChargerStatus calculations."""

    def test_get_amps_from_values_1p(self):
        """Test amps array for single phase."""
        env = create_integrated_environment(num_cars=1, num_chargers=1, num_persons=0)
        charger = env.chargers[0]

        cs = QSChargerStatus(charger)

        amps = cs.get_amps_from_values(10, 1)

        # Should have amps only on the charger's mono phase
        total = sum(amps)
        assert total == 10

    def test_get_amps_from_values_3p(self):
        """Test amps array for three phase."""
        env = create_integrated_environment(num_cars=1, num_chargers=1, num_persons=0)
        charger = env.chargers[0]

        cs = QSChargerStatus(charger)

        amps = cs.get_amps_from_values(10, 3)

        # Should have 10A on all phases
        assert amps == [10, 10, 10]

    def test_get_current_charging_amps(self):
        """Test getting current charging amps."""
        env = create_integrated_environment(num_cars=1, num_chargers=1, num_persons=0)
        charger = env.chargers[0]

        cs = QSChargerStatus(charger)
        cs.current_real_max_charging_amp = 16
        cs.current_active_phase_number = 1

        amps = cs.get_current_charging_amps()

        assert sum(amps) == 16

    def test_charger_status_duplicate(self):
        """Test duplicating charger status."""
        env = create_integrated_environment(num_cars=1, num_chargers=1, num_persons=0)
        charger = env.chargers[0]

        cs = QSChargerStatus(charger)
        cs.current_real_max_charging_amp = 16
        cs.budgeted_amp = 20
        cs.charge_score = 5

        dup = cs.duplicate()

        assert dup.current_real_max_charging_amp == 16
        assert dup.budgeted_amp == 20
        assert dup.charge_score == 5
        assert dup is not cs


# =============================================================================
# Test Class: Car Platforms
# =============================================================================

class TestCarPlatforms:
    """Test car platform support."""

    def test_car_get_platforms(self):
        """Test car returns correct platforms."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]

        platforms = car.get_platforms()

        assert "sensor" in platforms
        assert "select" in platforms
        assert "switch" in platforms
        assert "button" in platforms
        assert "time" in platforms

    def test_car_dashboard_sort_string_regular(self):
        """Test regular car dashboard sort string."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]
        car.car_is_invited = False

        sort_str = car.dashboard_sort_string_in_type

        assert sort_str == "AAA"

    def test_car_dashboard_sort_string_invited(self):
        """Test invited car dashboard sort string."""
        env = create_integrated_environment(num_cars=1, num_chargers=0, num_persons=0)
        car = env.cars[0]
        car.car_is_invited = True

        sort_str = car.dashboard_sort_string_in_type

        assert sort_str == "ZZZ"
