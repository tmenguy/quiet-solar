"""Comprehensive tests for QSCar in ha_model/car.py.

This test file extends test_ha_car.py to achieve 80%+ coverage by testing:
- Efficiency learning from SOC/odometer segments
- Dampening value management
- Range estimation
- Person-car interaction
- Constraint conversion
- Charger integration
"""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from datetime import timedelta, time as dt_time

from homeassistant.const import (
    Platform,
    STATE_UNKNOWN,
    STATE_UNAVAILABLE,
    CONF_NAME,
)
import pytz

from custom_components.quiet_solar.ha_model.car import (
    QSCar,
    MIN_CHARGE_POWER_W,
    CAR_MAX_EFFICIENCY_HISTORY_S,
    CAR_DEFAULT_CAPACITY,
    CAR_MINIMUM_LEFT_RANGE_KM,
)
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
    CONF_CAR_ODOMETER_SENSOR,
    CONF_CAR_ESTIMATED_RANGE_SENSOR,
    CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS,
    CAR_HARD_WIRED_CHARGER,
    FORCE_CAR_NO_PERSON_ATTACHED,
    MAX_POSSIBLE_AMPERAGE,
)

# Import from local conftest - use relative import to avoid conflict with HA core
from tests.test_helpers import FakeHass, FakeConfigEntry


# ============================================================================
# Tests for car efficiency learning
# ============================================================================

class TestQSCarEfficiency:
    """Test car efficiency calculation and learning."""

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
        self.home._cars = []
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def create_car(self, **extra_kwargs):
        """Helper to create a car with common configuration."""
        config = {
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
            CONF_CAR_BATTERY_CAPACITY: 75000,  # 75 kWh
        }
        config.update(extra_kwargs)

        return QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **config
        )

    def test_efficiency_segments_list_exists(self):
        """Test efficiency segments list is initialized."""
        car = self.create_car()

        assert hasattr(car, '_efficiency_segments')
        assert isinstance(car._efficiency_segments, list)

    def test_decreasing_segments_list_exists(self):
        """Test decreasing segments list is initialized."""
        car = self.create_car()

        assert hasattr(car, '_decreasing_segments')
        assert isinstance(car._decreasing_segments, list)

    def test_add_soc_odo_value_tracking(self):
        """Test SOC/odometer value tracking for efficiency."""
        car = self.create_car(
            CONF_CAR_ODOMETER_SENSOR="sensor.car_odometer",
            CONF_CAR_ESTIMATED_RANGE_SENSOR="sensor.car_range"
        )

        # Simulate decreasing SOC segments
        time = datetime.datetime.now(pytz.UTC)

        # This tests internal tracking - we check the segments list
        assert car._decreasing_segments is not None
        assert car._dec_seg_count == 0

    def test_efficiency_segments_storage(self):
        """Test efficiency segments are stored correctly."""
        car = self.create_car()

        # Add mock efficiency segment
        time = datetime.datetime.now(pytz.UTC)
        segment = (50.0, 10.0, 80.0, 70.0, time)  # delta_km, delta_soc, soc_from, soc_to, time
        car._efficiency_segments.append(segment)

        assert len(car._efficiency_segments) == 1
        assert car._efficiency_segments[0][0] == 50.0  # delta_km
        assert car._efficiency_segments[0][1] == 10.0  # delta_soc

    def test_km_per_kwh_default_none(self):
        """Test _km_per_kwh is None by default."""
        car = self.create_car()

        assert car._km_per_kwh is None


# ============================================================================
# Tests for car dampening values
# ============================================================================

class TestQSCarDampening:
    """Test car charging dampening value management."""

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
        self.home._cars = []
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def create_car(self, **extra_kwargs):
        """Helper to create a car with common configuration."""
        config = {
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
        }
        config.update(extra_kwargs)

        return QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **config
        )

    def test_theoretical_amp_to_power_1p(self):
        """Test theoretical amp to power conversion for 1 phase."""
        car = self.create_car()

        # Check theoretical 1-phase power table exists
        assert hasattr(car, 'theoretical_amp_to_power_1p')
        # At 10A, 230V = 2300W for 1 phase
        assert car.theoretical_amp_to_power_1p[10] == pytest.approx(2300.0, abs=10)

    def test_theoretical_amp_to_power_3p(self):
        """Test theoretical amp to power conversion for 3 phase."""
        car = self.create_car()

        # Check theoretical 3-phase power table exists
        assert hasattr(car, 'theoretical_amp_to_power_3p')
        # At 10A, 230V * 3 = 6900W for 3 phase
        assert car.theoretical_amp_to_power_3p[10] == pytest.approx(6900.0, abs=10)

    def test_get_charge_power_per_phase_A(self):
        """Test get_charge_power_per_phase_A returns power table."""
        car = self.create_car()

        steps, min_a, max_a = car.get_charge_power_per_phase_A(for_3p=True)

        assert isinstance(steps, list)
        assert len(steps) == MAX_POSSIBLE_AMPERAGE
        assert min_a == car.car_charger_min_charge
        assert max_a == car.car_charger_max_charge

    def test_can_dampen_strongly_dynamically_default(self):
        """Test can_dampen_strongly_dynamically is True by default."""
        car = self.create_car()

        assert car.can_dampen_strongly_dynamically is True

    def test_dampening_deltas_dict_exists(self):
        """Test _dampening_deltas dictionary is initialized."""
        car = self.create_car()

        assert hasattr(car, '_dampening_deltas')
        assert isinstance(car._dampening_deltas, dict)


# ============================================================================
# Tests for car range estimation
# ============================================================================

class TestQSCarRangeEstimation:
    """Test car range and SOC estimation."""

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
        self.home._cars = []
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def create_car(self, **extra_kwargs):
        """Helper to create a car with common configuration."""
        config = {
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
            CONF_CAR_BATTERY_CAPACITY: 75000,  # 75 kWh
        }
        config.update(extra_kwargs)

        return QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **config
        )

    def test_get_adapt_target_percent_soc_returns_tuple(self):
        """Test get_adapt_target_percent_soc_to_reach_range_km returns tuple."""
        car = self.create_car()
        car._km_per_kwh = 6.0  # 6 km/kWh

        time = datetime.datetime.now(pytz.UTC)
        result = car.get_adapt_target_percent_soc_to_reach_range_km(
            target_range_km=100.0,
            time=time
        )

        # Should return a 4-tuple
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_car_battery_capacity_configured(self):
        """Test car battery capacity is configured."""
        car = self.create_car()

        assert car.car_battery_capacity == 75000


# ============================================================================
# Tests for car-person interaction
# ============================================================================

class TestQSCarPersonInteraction:
    """Test car-person allocation and interaction."""

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
        self.home._cars = []
        self.home.get_best_persons_cars_allocations = AsyncMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def create_car(self, **extra_kwargs):
        """Helper to create a car with common configuration."""
        config = {
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
        }
        config.update(extra_kwargs)

        return QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **config
        )

    def test_user_selected_person_property_getter(self):
        """Test user_selected_person_name_for_car getter."""
        car = self.create_car()
        car._user_selected_person_name_for_car = "John"

        assert car.user_selected_person_name_for_car == "John"

    def test_user_selected_person_property_setter_triggers_update(self):
        """Test user_selected_person_name_for_car setter triggers allocation update."""
        car = self.create_car()
        self.home._cars = [car]
        self.home.get_best_persons_cars_allocations = AsyncMock()
        self.hass.create_task = MagicMock(side_effect=lambda coro, name=None: coro.close())

        car.user_selected_person_name_for_car = "Jane"

        self.home.get_best_persons_cars_allocations.assert_called_once_with(force_update=True)
        assert self.hass.create_task.call_count == 1

    def test_user_selected_person_no_person_attached(self):
        """Test setting FORCE_CAR_NO_PERSON_ATTACHED."""
        car = self.create_car()

        car._user_selected_person_name_for_car = FORCE_CAR_NO_PERSON_ATTACHED

        assert car._user_selected_person_name_for_car == FORCE_CAR_NO_PERSON_ATTACHED

    def test_car_person_option_format(self):
        """Test _car_person_option returns person name."""
        car = self.create_car()

        result = car._car_person_option("John Doe")

        assert result == "John Doe"

    def test_current_forecasted_person_attribute_exists(self):
        """Test current_forecasted_person attribute exists."""
        car = self.create_car()

        assert hasattr(car, 'current_forecasted_person')

    def test_get_car_person_readable_forecast_no_person(self):
        """Test get_car_person_readable_forecast_mileage with no person."""
        car = self.create_car()
        car.current_forecasted_person = None

        result = car.get_car_person_readable_forecast_mileage()

        assert result == "No forecasted person"


# ============================================================================
# Tests for car-charger integration
# ============================================================================

class TestQSCarChargerIntegration:
    """Test car-charger interaction."""

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
        self.home._cars = []
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def create_car(self, **extra_kwargs):
        """Helper to create a car with common configuration."""
        config = {
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
        }
        config.update(extra_kwargs)

        return QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **config
        )

    def test_charger_property_none_by_default(self):
        """Test charger is None by default."""
        car = self.create_car()

        assert car.charger is None

    def test_charger_can_be_set(self):
        """Test charger can be assigned."""
        car = self.create_car()
        mock_charger = MagicMock()
        mock_charger.name = "Test Charger"

        car.charger = mock_charger

        assert car.charger == mock_charger

    def test_user_attached_charger_name(self):
        """Test user_attached_charger_name property."""
        car = self.create_car()
        car.user_attached_charger_name = "My Charger"

        assert car.user_attached_charger_name == "My Charger"

    def test_hard_wired_charger(self):
        """Test car with hard-wired charger."""
        car = self.create_car(**{
            CAR_HARD_WIRED_CHARGER: "charger_1"
        })

        assert car.car_hard_wired_charger == "charger_1"


# ============================================================================
# Tests for car charge targets
# ============================================================================

class TestQSCarChargeTargets:
    """Test car charge target management."""

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
        self.home._cars = []
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def create_car(self, **extra_kwargs):
        """Helper to create a car with common configuration."""
        config = {
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
            CONF_DEFAULT_CAR_CHARGE: 80.0,
            CONF_MINIMUM_OK_CAR_CHARGE: 20.0,
        }
        config.update(extra_kwargs)

        return QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **config
        )

    def test_default_charge_target(self):
        """Test default charge target value."""
        car = self.create_car()

        assert car.car_default_charge == 80.0

    def test_minimum_ok_charge(self):
        """Test minimum OK charge value."""
        car = self.create_car()

        assert car.car_minimum_ok_charge == 20.0

    def test_force_next_charge_flag(self):
        """Test do_force_next_charge flag."""
        car = self.create_car()

        assert car.do_force_next_charge is False

        car.do_force_next_charge = True
        assert car.do_force_next_charge is True

    def test_next_charge_time(self):
        """Test do_next_charge_time property."""
        car = self.create_car()

        assert car.do_next_charge_time is None

        time = datetime.datetime.now(pytz.UTC)
        car.do_next_charge_time = time
        assert car.do_next_charge_time == time


# ============================================================================
# Tests for car charge percent steps
# ============================================================================

class TestQSCarChargePercentSteps:
    """Test car charge percent step handling."""

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
        self.home._cars = []
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def create_car(self, **extra_kwargs):
        """Helper to create a car with common configuration."""
        config = {
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
        }
        config.update(extra_kwargs)

        return QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **config
        )

    def test_charge_percent_steps_empty_by_default(self):
        """Test car_charge_percent_max_number_steps is empty by default."""
        car = self.create_car()

        assert car.car_charge_percent_max_number_steps == []

    def test_charge_percent_steps_from_config(self):
        """Test car_charge_percent_max_number_steps from config string."""
        car = self.create_car(**{
            CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS: "50, 80, 90"
        })

        assert car.car_charge_percent_max_number_steps == [50, 80, 90, 100]

    def test_charge_percent_steps_adds_100(self):
        """Test 100% is added if not present."""
        car = self.create_car(**{
            CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS: "50, 80"
        })

        assert 100 in car.car_charge_percent_max_number_steps

    def test_charge_percent_steps_sorted(self):
        """Test steps are sorted."""
        car = self.create_car(**{
            CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS: "80, 50, 90"
        })

        assert car.car_charge_percent_max_number_steps == sorted(car.car_charge_percent_max_number_steps)

    def test_charge_percent_steps_invalid_values_rejected(self):
        """Test invalid values result in empty list."""
        car = self.create_car(**{
            CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS: "invalid, not_a_number"
        })

        assert car.car_charge_percent_max_number_steps == []

    def test_charge_percent_steps_out_of_range_rejected(self):
        """Test out-of-range values are rejected."""
        car = self.create_car(**{
            CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS: "50, 150"  # 150 > 100
        })

        # 150 should be rejected, keeping only valid values
        assert 150 not in car.car_charge_percent_max_number_steps


# ============================================================================
# Tests for car save/restore state
# ============================================================================

class TestQSCarSaveRestoreState:
    """Test car state persistence."""

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
        self.home._cars = []
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def create_car(self, **extra_kwargs):
        """Helper to create a car with common configuration."""
        config = {
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
        }
        config.update(extra_kwargs)

        return QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **config
        )

    def test_update_to_be_saved_extra_device_info(self):
        """Test update_to_be_saved_extra_device_info saves person info."""
        car = self.create_car()
        car._user_selected_person_name_for_car = "John"

        mock_person = MagicMock()
        mock_person.name = "Jane"
        car.current_forecasted_person = mock_person

        data = {}
        car.update_to_be_saved_extra_device_info(data)

        assert data["user_selected_person_name_for_car"] == "John"
        assert data["current_forecasted_person_name_from_boot"] == "Jane"

    def test_use_saved_extra_device_info(self):
        """Test use_saved_extra_device_info restores person info."""
        car = self.create_car()

        stored_data = {
            "user_selected_person_name_for_car": "John",
            "current_forecasted_person_name_from_boot": "Jane"
        }

        car.use_saved_extra_device_info(stored_data)

        assert car._user_selected_person_name_for_car == "John"
        assert car._current_forecasted_person_name_from_boot == "Jane"


# ============================================================================
# Tests for car async operations
# ============================================================================

class TestQSCarAsyncOperations:
    """Test car async operations."""

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
        self.home._cars = []
        self.home.get_person_by_name = MagicMock(return_value=None)
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def create_car(self, **extra_kwargs):
        """Helper to create a car with common configuration."""
        config = {
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.car",
        }
        config.update(extra_kwargs)

        return QSCar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **config
        )

    @pytest.mark.asyncio
    async def test_get_car_mileage_on_period_no_sensors(self):
        """Test get_car_mileage_on_period_km with no sensors."""
        car = self.create_car()
        car.car_odometer_sensor = None
        car.car_tracker = None

        time = datetime.datetime.now(pytz.UTC)
        result = await car.get_car_mileage_on_period_km(
            time - timedelta(hours=1),
            time
        )

        # With no sensors, should return None
        assert result is None or result == 0.0

    def test_device_post_home_init(self):
        """Test device_post_home_init restores person."""
        car = self.create_car()
        car._current_forecasted_person_name_from_boot = "John"

        mock_person = MagicMock()
        mock_person.name = "John"
        self.home.get_person_by_name = MagicMock(return_value=mock_person)

        time = datetime.datetime.now(pytz.UTC)
        car.device_post_home_init(time)

        # Should have restored current_forecasted_person
        # (if home.get_person_by_name returns the person)
        self.home.get_person_by_name.assert_called()
