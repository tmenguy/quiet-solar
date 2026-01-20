"""Tests for QSCar in ha_model/car.py."""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import timedelta

from homeassistant.const import CONF_NAME, STATE_UNKNOWN, STATE_UNAVAILABLE
import pytz

from custom_components.quiet_solar.ha_model.car import QSCar, MIN_CHARGE_POWER_W
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_CAR_PLUGGED,
    CONF_CAR_TRACKER,
    CONF_CAR_CHARGE_PERCENT_SENSOR,
    CONF_CAR_CHARGE_PERCENT_MAX_NUMBER,
    CONF_CAR_BATTERY_CAPACITY,
    CONF_CAR_CHARGER_MIN_CHARGE,
    CONF_CAR_CHARGER_MAX_CHARGE,
    CONF_CAR_CUSTOM_POWER_CHARGE_VALUES,
    CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P,
    CONF_DEFAULT_CAR_CHARGE,
    CONF_MINIMUM_OK_CAR_CHARGE,
    CONF_CAR_IS_INVITED,
    MAX_POSSIBLE_AMPERAGE,
    CONF_CAR_ODOMETER_SENSOR,
    CONF_CAR_ESTIMATED_RANGE_SENSOR,
)

from tests.conftest import FakeHass, FakeConfigEntry


class TestQSCarInit:
    """Test QSCar initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_car_entry",
            data={CONF_NAME: "Test Car"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.home.latitude = 48.8566
        self.home.longitude = 2.3522
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
            }
        )

        assert car.name == "Test Car"
        assert car.car_tracker == "device_tracker.car"
        assert car.charger is None

    def test_init_with_all_sensors(self):
        """Test initialization with all sensors."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                CONF_CAR_PLUGGED: "binary_sensor.car_plugged",
                CONF_CAR_CHARGE_PERCENT_SENSOR: "sensor.car_soc",
                CONF_CAR_CHARGE_PERCENT_MAX_NUMBER: "number.car_max_soc",
                CONF_CAR_ODOMETER_SENSOR: "sensor.car_odometer",
                CONF_CAR_ESTIMATED_RANGE_SENSOR: "sensor.car_range",
            }
        )

        assert car.car_plugged == "binary_sensor.car_plugged"
        assert car.car_charge_percent_sensor == "sensor.car_soc"
        assert car.car_charge_percent_max_number == "number.car_max_soc"
        assert car.car_odometer_sensor == "sensor.car_odometer"
        assert car.car_estimated_range_sensor == "sensor.car_range"

    def test_init_with_battery_capacity(self):
        """Test initialization with battery capacity."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                CONF_CAR_BATTERY_CAPACITY: 75000,  # 75 kWh
            }
        )

        assert car.car_battery_capacity == 75000

    def test_init_with_charger_limits(self):
        """Test initialization with charger current limits."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                CONF_CAR_CHARGER_MIN_CHARGE: 6,
                CONF_CAR_CHARGER_MAX_CHARGE: 16,
            }
        )

        assert car.car_charger_min_charge == 6
        assert car.car_charger_max_charge == 16

    def test_init_with_default_charge(self):
        """Test initialization with default charge settings."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                CONF_DEFAULT_CAR_CHARGE: 80.0,
                CONF_MINIMUM_OK_CAR_CHARGE: 20.0,
            }
        )

        assert car.car_default_charge == 80.0
        assert car.car_minimum_ok_charge == 20.0

    def test_init_invited_car(self):
        """Test initialization with invited car flag."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Guest Car",
                CONF_CAR_TRACKER: "device_tracker.guest_car",
                CONF_CAR_IS_INVITED: True,
            }
        )

        assert car.car_is_invited is True

    def test_init_creates_amp_to_power_tables(self):
        """Test that amp-to-power tables are initialized."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
            }
        )

        # Should have tables for 1p and 3p
        assert len(car.amp_to_power_1p) == MAX_POSSIBLE_AMPERAGE
        assert len(car.amp_to_power_3p) == MAX_POSSIBLE_AMPERAGE

    def test_init_theoretical_power_calculation(self):
        """Test theoretical power calculation in init."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
            }
        )

        # For 10A at 230V
        assert car.theoretical_amp_to_power_1p[10] == pytest.approx(2300.0, abs=10)
        assert car.theoretical_amp_to_power_3p[10] == pytest.approx(6900.0, abs=10)


class TestQSCarCustomPowerValues:
    """Test QSCar with custom power charge values."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_car_entry",
            data={CONF_NAME: "Test Car"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_init_with_custom_power_values_1p(self):
        """Test initialization with custom 1-phase power values."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                CONF_CAR_CHARGER_MIN_CHARGE: 6,
                CONF_CAR_CHARGER_MAX_CHARGE: 16,
                CONF_CAR_CUSTOM_POWER_CHARGE_VALUES: True,
                CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P: False,
                "charge_6": 1200,
                "charge_10": 2100,
                "charge_16": 3500,
            }
        )

        assert car.car_use_custom_power_charge_values is True
        assert car.car_is_custom_power_charge_values_3p is False
        assert car.customized_amp_to_power_1p[6] == 1200
        assert car.customized_amp_to_power_1p[10] == 2100
        assert car.customized_amp_to_power_1p[16] == 3500

    def test_init_with_custom_power_values_3p(self):
        """Test initialization with custom 3-phase power values."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                CONF_CAR_CHARGER_MIN_CHARGE: 6,
                CONF_CAR_CHARGER_MAX_CHARGE: 16,
                CONF_CAR_CUSTOM_POWER_CHARGE_VALUES: True,
                CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P: True,
                "charge_6": 4000,
                "charge_10": 7000,
                "charge_16": 11000,
            }
        )

        assert car.car_is_custom_power_charge_values_3p is True
        assert car.customized_amp_to_power_3p[6] == 4000
        assert car.customized_amp_to_power_3p[10] == 7000
        assert car.customized_amp_to_power_3p[16] == 11000


class TestQSCarChargePercent:
    """Test QSCar charge percent methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_car_entry",
            data={CONF_NAME: "Test Car"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                CONF_CAR_CHARGE_PERCENT_SENSOR: "sensor.car_soc",
                CONF_CAR_BATTERY_CAPACITY: 75000,
            }
        )

    def test_get_car_charge_percent_valid(self):
        """Test get_car_charge_percent with valid value."""
        self.car.get_sensor_latest_possible_valid_value = MagicMock(return_value=80.0)

        result = self.car.get_car_charge_percent()

        assert result == 80.0

    def test_get_car_charge_percent_none(self):
        """Test get_car_charge_percent with None value."""
        self.car.get_sensor_latest_possible_valid_value = MagicMock(return_value=None)

        result = self.car.get_car_charge_percent()

        assert result is None

    def test_get_car_battery_capacity(self):
        """Test car_battery_capacity is set correctly."""
        assert self.car.car_battery_capacity == 75000


class TestQSCarPluggedState:
    """Test QSCar plugged state methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_car_entry",
            data={CONF_NAME: "Test Car"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                CONF_CAR_PLUGGED: "binary_sensor.car_plugged",
            }
        )

    def test_car_plugged_sensor_configured(self):
        """Test car_plugged sensor is configured correctly."""
        assert self.car.car_plugged == "binary_sensor.car_plugged"

    def test_car_has_plugged_check(self):
        """Test car has is_car_plugged method."""
        assert hasattr(self.car, 'is_car_plugged')


class TestQSCarLocation:
    """Test QSCar location methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_car_entry",
            data={CONF_NAME: "Test Car"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.home.latitude = 48.8566
        self.home.longitude = 2.3522
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
            }
        )

    def test_car_tracker_configured(self):
        """Test car_tracker is configured correctly."""
        assert self.car.car_tracker == "device_tracker.car"

    def test_car_has_location_methods(self):
        """Test car has location-related methods."""
        assert hasattr(self.car, 'get_car_coordinates')
        assert hasattr(self.car, 'is_car_home')


class TestQSCarAmpToPower:
    """Test QSCar amp to power conversion."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_car_entry",
            data={CONF_NAME: "Test Car"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                CONF_CAR_CHARGER_MIN_CHARGE: 6,
                CONF_CAR_CHARGER_MAX_CHARGE: 32,
            }
        )

    def test_get_power_from_amps_1p(self):
        """Test getting power from amps for 1-phase."""
        # 10A at 230V = 2300W
        power = self.car.amp_to_power_1p[10]
        assert power == pytest.approx(2300.0, abs=10)

    def test_get_power_from_amps_3p(self):
        """Test getting power from amps for 3-phase."""
        # 10A at 230V * 3 = 6900W
        power = self.car.amp_to_power_3p[10]
        assert power == pytest.approx(6900.0, abs=10)

    def test_amp_to_power_zero(self):
        """Test amp to power at 0 amps."""
        assert self.car.amp_to_power_1p[0] == 0.0
        assert self.car.amp_to_power_3p[0] == 0.0


class TestQSCarChargePercentSteps:
    """Test QSCar charge percent max number steps."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_car_entry",
            data={CONF_NAME: "Test Car"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_charge_percent_steps_parsed(self):
        """Test charge percent steps are parsed correctly."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                "car_charge_percent_max_number_steps": "50, 80, 90",
            }
        )

        assert 50 in car.car_charge_percent_max_number_steps
        assert 80 in car.car_charge_percent_max_number_steps
        assert 90 in car.car_charge_percent_max_number_steps
        assert 100 in car.car_charge_percent_max_number_steps  # 100 is always added

    def test_charge_percent_steps_sorted(self):
        """Test charge percent steps are sorted."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                "car_charge_percent_max_number_steps": "90, 50, 80",
            }
        )

        # Should be sorted
        for i in range(len(car.car_charge_percent_max_number_steps) - 1):
            assert car.car_charge_percent_max_number_steps[i] <= car.car_charge_percent_max_number_steps[i + 1]

    def test_charge_percent_steps_invalid(self):
        """Test charge percent steps with invalid values."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                "car_charge_percent_max_number_steps": "abc, 50, xyz",
            }
        )

        # Should be empty due to parsing error
        assert car.car_charge_percent_max_number_steps == []

    def test_charge_percent_steps_empty_string(self):
        """Test charge percent steps with empty string."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                "car_charge_percent_max_number_steps": "",
            }
        )

        assert car.car_charge_percent_max_number_steps == []


class TestQSCarChargerLimits:
    """Test QSCar charger limits."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_car_entry",
            data={CONF_NAME: "Test Car"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_charger_min_charge_clamped(self):
        """Test charger min charge is clamped to 0."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                CONF_CAR_CHARGER_MIN_CHARGE: -5,  # Negative
            }
        )

        assert car.car_charger_min_charge == 0

    def test_charger_max_charge_clamped(self):
        """Test charger max charge is within bounds."""
        car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                CONF_CAR_CHARGER_MAX_CHARGE: 32,  # A reasonable value
            }
        )

        # Should be a positive integer within bounds
        assert car.car_charger_max_charge >= 0
        assert isinstance(car.car_charger_max_charge, int)


class TestQSCarDoForceNextCharge:
    """Test QSCar force next charge flag."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_car_entry",
            data={CONF_NAME: "Test Car"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.car = QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
            }
        )

    def test_do_force_next_charge_default(self):
        """Test do_force_next_charge defaults to False."""
        assert self.car.do_force_next_charge is False

    def test_do_force_next_charge_can_be_set(self):
        """Test do_force_next_charge can be set."""
        self.car.do_force_next_charge = True
        assert self.car.do_force_next_charge is True
