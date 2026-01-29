"""Tests for QSCar in ha_model/car.py."""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import timedelta

from homeassistant.const import CONF_NAME, STATE_UNKNOWN, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant
import pytz

from pytest_homeassistant_custom_component.common import MockConfigEntry

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

from tests.factories import create_minimal_home_model


@pytest.fixture
def car_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Mock config entry for car tests."""
    return MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_car_entry",
        data={CONF_NAME: "Test Car"},
        title="Test Car",
    )


@pytest.fixture
def car_home(hass: HomeAssistant):
    """Home and data handler for car tests."""
    home = create_minimal_home_model()
    home.is_off_grid = MagicMock(return_value=False)
    home.voltage = 230.0
    home.latitude = 48.8566
    home.longitude = 2.3522
    data_handler = MagicMock()
    data_handler.home = home
    hass.data.setdefault(DOMAIN, {})[DATA_HANDLER] = data_handler
    return home


def test_init_minimal(hass: HomeAssistant, car_config_entry, car_home):
    """Test initialization with minimal parameters."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
            }
        )

    assert car.name == "Test Car"
    assert car.car_tracker == "device_tracker.car"
    assert car.charger is None


def test_init_with_all_sensors(hass: HomeAssistant, car_config_entry, car_home):
    """Test initialization with all sensors."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
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


def test_init_with_battery_capacity(hass: HomeAssistant, car_config_entry, car_home):
    """Test initialization with battery capacity."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                CONF_CAR_BATTERY_CAPACITY: 75000,  # 75 kWh
            }
        )

    assert car.car_battery_capacity == 75000


def test_init_with_charger_limits(hass: HomeAssistant, car_config_entry, car_home):
    """Test initialization with charger current limits."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                CONF_CAR_CHARGER_MIN_CHARGE: 6,
                CONF_CAR_CHARGER_MAX_CHARGE: 16,
            }
        )

    assert car.car_charger_min_charge == 6
    assert car.car_charger_max_charge == 16


def test_init_with_default_charge(hass: HomeAssistant, car_config_entry, car_home):
    """Test initialization with default charge settings."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
                CONF_DEFAULT_CAR_CHARGE: 80.0,
                CONF_MINIMUM_OK_CAR_CHARGE: 20.0,
            }
        )

    assert car.car_default_charge == 80.0
    assert car.car_minimum_ok_charge == 20.0


def test_init_invited_car(hass: HomeAssistant, car_config_entry, car_home):
    """Test initialization with invited car flag."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
            **{
                CONF_NAME: "Guest Car",
                CONF_CAR_TRACKER: "device_tracker.guest_car",
                CONF_CAR_IS_INVITED: True,
            }
        )

    assert car.car_is_invited is True


def test_init_creates_amp_to_power_tables(hass: HomeAssistant, car_config_entry, car_home):
    """Test that amp-to-power tables are initialized."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
            }
        )

    # Should have tables for 1p and 3p
    assert len(car.amp_to_power_1p) == MAX_POSSIBLE_AMPERAGE
    assert len(car.amp_to_power_3p) == MAX_POSSIBLE_AMPERAGE


def test_init_theoretical_power_calculation(hass: HomeAssistant, car_config_entry, car_home):
    """Test theoretical power calculation in init."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
            **{
                CONF_NAME: "Test Car",
                CONF_CAR_TRACKER: "device_tracker.car",
            }
        )

    # For 10A at 230V
    assert car.theoretical_amp_to_power_1p[10] == pytest.approx(2300.0, abs=10)
    assert car.theoretical_amp_to_power_3p[10] == pytest.approx(6900.0, abs=10)


def test_init_with_custom_power_values_1p(hass: HomeAssistant, car_config_entry, car_home):
    """Test initialization with custom 1-phase power values."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
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


def test_init_with_custom_power_values_3p(hass: HomeAssistant, car_config_entry, car_home):
    """Test initialization with custom 3-phase power values."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
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


@pytest.fixture
def car_with_charge_percent(hass: HomeAssistant, car_config_entry, car_home):
    """Car instance with charge percent sensor for charge percent tests."""
    return QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
        **{
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
            CONF_CAR_CHARGE_PERCENT_SENSOR: "sensor.car_soc",
            CONF_CAR_BATTERY_CAPACITY: 75000,
        }
    )


def test_get_car_charge_percent_valid(car_with_charge_percent):
    """Test get_car_charge_percent with valid value."""
    car_with_charge_percent.get_sensor_latest_possible_valid_value = MagicMock(
        return_value=80.0
    )

    result = car_with_charge_percent.get_car_charge_percent()

    assert result == 80.0


def test_get_car_charge_percent_none(car_with_charge_percent):
    """Test get_car_charge_percent with None value."""
    car_with_charge_percent.get_sensor_latest_possible_valid_value = MagicMock(
        return_value=None
    )

    result = car_with_charge_percent.get_car_charge_percent()

    assert result is None


def test_get_car_battery_capacity(car_with_charge_percent):
    """Test car_battery_capacity is set correctly."""
    assert car_with_charge_percent.car_battery_capacity == 75000


def test_car_plugged_sensor_configured(hass: HomeAssistant, car_config_entry, car_home):
    """Test car_plugged sensor is configured correctly."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
        **{
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
            CONF_CAR_PLUGGED: "binary_sensor.car_plugged",
        }
    )
    assert car.car_plugged == "binary_sensor.car_plugged"


def test_car_has_plugged_check(hass: HomeAssistant, car_config_entry, car_home):
    """Test car has is_car_plugged method."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
        **{
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
            CONF_CAR_PLUGGED: "binary_sensor.car_plugged",
        }
    )
    assert hasattr(car, "is_car_plugged")


def test_car_tracker_configured(hass: HomeAssistant, car_config_entry, car_home):
    """Test car_tracker is configured correctly."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
        **{CONF_NAME: "Test Car", CONF_CAR_TRACKER: "device_tracker.car"}
    )
    assert car.car_tracker == "device_tracker.car"


def test_car_has_location_methods(hass: HomeAssistant, car_config_entry, car_home):
    """Test car has location-related methods."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
        **{CONF_NAME: "Test Car", CONF_CAR_TRACKER: "device_tracker.car"}
    )
    assert hasattr(car, "get_car_coordinates")
    assert hasattr(car, "is_car_home")


@pytest.fixture
def car_amp_to_power(hass: HomeAssistant, car_config_entry, car_home):
    """Car instance for amp-to-power tests."""
    return QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
        **{
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
            CONF_CAR_CHARGER_MIN_CHARGE: 6,
            CONF_CAR_CHARGER_MAX_CHARGE: 32,
        }
    )


def test_get_power_from_amps_1p(car_amp_to_power):
    """Test getting power from amps for 1-phase."""
    # 10A at 230V = 2300W
    power = car_amp_to_power.amp_to_power_1p[10]
    assert power == pytest.approx(2300.0, abs=10)


def test_get_power_from_amps_3p(car_amp_to_power):
    """Test getting power from amps for 3-phase."""
    # 10A at 230V * 3 = 6900W
    power = car_amp_to_power.amp_to_power_3p[10]
    assert power == pytest.approx(6900.0, abs=10)


def test_amp_to_power_zero(car_amp_to_power):
    """Test amp to power at 0 amps."""
    assert car_amp_to_power.amp_to_power_1p[0] == 0.0
    assert car_amp_to_power.amp_to_power_3p[0] == 0.0


def test_charge_percent_steps_parsed(hass: HomeAssistant, car_config_entry, car_home):
    """Test charge percent steps are parsed correctly."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
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


def test_charge_percent_steps_sorted(hass: HomeAssistant, car_config_entry, car_home):
    """Test charge percent steps are sorted."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
        **{
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
            "car_charge_percent_max_number_steps": "90, 50, 80",
        }
    )
    for i in range(len(car.car_charge_percent_max_number_steps) - 1):
        assert (
            car.car_charge_percent_max_number_steps[i]
            <= car.car_charge_percent_max_number_steps[i + 1]
        )


def test_charge_percent_steps_invalid(hass: HomeAssistant, car_config_entry, car_home):
    """Test charge percent steps with invalid values."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
        **{
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
            "car_charge_percent_max_number_steps": "abc, 50, xyz",
        }
    )
    assert car.car_charge_percent_max_number_steps == []


def test_charge_percent_steps_empty_string(
    hass: HomeAssistant, car_config_entry, car_home
):
    """Test charge percent steps with empty string."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
        **{
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
            "car_charge_percent_max_number_steps": "",
        }
    )
    assert car.car_charge_percent_max_number_steps == []


def test_charger_min_charge_clamped(hass: HomeAssistant, car_config_entry, car_home):
    """Test charger min charge is clamped to 0."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
        **{
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
            CONF_CAR_CHARGER_MIN_CHARGE: -5,  # Negative
        }
    )
    assert car.car_charger_min_charge == 0


def test_charger_max_charge_clamped(hass: HomeAssistant, car_config_entry, car_home):
    """Test charger max charge is within bounds."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
        **{
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
            CONF_CAR_CHARGER_MAX_CHARGE: 32,  # A reasonable value
        }
    )
    assert car.car_charger_max_charge >= 0
    assert isinstance(car.car_charger_max_charge, int)


def test_do_force_next_charge_default(hass: HomeAssistant, car_config_entry, car_home):
    """Test do_force_next_charge defaults to False."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
        **{CONF_NAME: "Test Car", CONF_CAR_TRACKER: "device_tracker.car"}
    )
    assert car.do_force_next_charge is False


def test_do_force_next_charge_can_be_set(
    hass: HomeAssistant, car_config_entry, car_home
):
    """Test do_force_next_charge can be set."""
    car = QSCar(
        hass=hass,
        config_entry=car_config_entry,
        home=car_home,
        **{CONF_NAME: "Test Car", CONF_CAR_TRACKER: "device_tracker.car"}
    )
    car.do_force_next_charge = True
    assert car.do_force_next_charge is True
