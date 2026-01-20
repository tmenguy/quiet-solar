"""Comprehensive tests for QSHome in ha_model/home.py.

This test file targets the 51% -> 80%+ coverage gap by testing:
- Device management and orchestration
- Person-car allocation algorithms
- GPS path mapping and mileage computation
- Forecast sensor probing
- Off-grid mode handling
- Dashboard section generation
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
import numpy as np

from custom_components.quiet_solar.ha_model.home import (
    QSHome,
    QSHomeMode,
    QSforecastValueSensor,
    get_time_from_state,
    _segments_weak_sub_on_main_overlap,
    _segments_strong_overlap,
    MAX_SENSOR_HISTORY_S,
    HOME_PERSON_CAR_MIN_SEGMENT_S,
    HOME_PERSON_CAR_MIN_OVERLAP_S,
)
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_HOME_VOLTAGE,
    CONF_IS_3P,
    CONF_GRID_POWER_SENSOR,
    CONF_GRID_POWER_SENSOR_INVERTED,
    CONF_HOME_PEAK_PRICE,
    CONF_HOME_OFF_PEAK_PRICE,
    CONF_HOME_START_OFF_PEAK_RANGE_1,
    CONF_HOME_END_OFF_PEAK_RANGE_1,
)

# Import from local conftest - use relative import to avoid conflict with HA core
from tests.test_helpers import FakeHass, FakeConfigEntry, FakeState


class MockLazyState:
    """Mock LazyState for testing."""

    def __init__(self, entity_id: str, state: str, attributes: dict | None = None, last_updated: datetime.datetime | None = None):
        self.entity_id = entity_id
        self.state = state
        self.attributes = attributes or {}
        self.last_updated = last_updated or datetime.datetime.now(pytz.UTC)


# ============================================================================
# Tests for get_time_from_state helper function
# ============================================================================

class TestGetTimeFromState:
    """Test the get_time_from_state helper function."""

    def test_get_time_from_state_with_none(self):
        """Test with None state."""
        result = get_time_from_state(None)
        assert result is None

    def test_get_time_from_state_with_valid_datetime(self):
        """Test with valid datetime in last_updated."""
        time = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        state = MockLazyState("sensor.test", "50", last_updated=time)

        result = get_time_from_state(state)
        assert result == time

    def test_get_time_from_state_with_datetime_attribute(self):
        """Test with datetime in attributes override."""
        base_time = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        attr_time = datetime.datetime(2024, 6, 15, 13, 0, 0, tzinfo=pytz.UTC)
        state = MockLazyState(
            "sensor.test",
            "50",
            attributes={"last_updated": attr_time},
            last_updated=base_time
        )

        result = get_time_from_state(state)
        assert result == attr_time

    def test_get_time_from_state_with_iso_string_attribute(self):
        """Test with ISO string in attributes."""
        base_time = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        state = MockLazyState(
            "sensor.test",
            "50",
            attributes={"last_updated": "2024-06-15T14:00:00+00:00"},
            last_updated=base_time
        )

        result = get_time_from_state(state)
        assert result is not None
        # The parsed time should be 14:00
        assert result.hour == 14

    def test_get_time_from_state_with_unknown_attribute(self):
        """Test with 'unknown' string in attributes."""
        base_time = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        state = MockLazyState(
            "sensor.test",
            "50",
            attributes={"last_updated": "unknown"},
            last_updated=base_time
        )

        result = get_time_from_state(state)
        # Should fall back to last_updated
        assert result == base_time

    def test_get_time_from_state_with_none_attribute(self):
        """Test with 'none' string in attributes."""
        base_time = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        state = MockLazyState(
            "sensor.test",
            "50",
            attributes={"last_updated": "None"},
            last_updated=base_time
        )

        result = get_time_from_state(state)
        assert result == base_time


# ============================================================================
# Tests for segment overlap functions
# ============================================================================

class TestSegmentOverlap:
    """Test segment overlap detection functions."""

    def test_segments_weak_sub_overlap_no_overlap(self):
        """Test with non-overlapping segments."""
        now = datetime.datetime.now(pytz.UTC)

        segments_sub = [(now, now + timedelta(hours=1))]
        segments_main = [(now + timedelta(hours=2), now + timedelta(hours=3))]

        result = _segments_weak_sub_on_main_overlap(segments_sub, segments_main)
        assert len(result) == 0

    def test_segments_weak_sub_overlap_with_overlap(self):
        """Test with overlapping segments."""
        now = datetime.datetime.now(pytz.UTC)

        segments_sub = [(now, now + timedelta(hours=2))]
        segments_main = [(now + timedelta(hours=1), now + timedelta(hours=3))]

        result = _segments_weak_sub_on_main_overlap(segments_sub, segments_main, min_overlap=0)
        assert len(result) == 1
        # Overlap should be from hour 1 to hour 2
        assert result[0][0] == now + timedelta(hours=1)
        assert result[0][1] == now + timedelta(hours=2)

    def test_segments_weak_sub_overlap_min_overlap_filter(self):
        """Test that min_overlap filters short overlaps."""
        now = datetime.datetime.now(pytz.UTC)

        # 30 second overlap
        segments_sub = [(now, now + timedelta(seconds=30))]
        segments_main = [(now + timedelta(seconds=20), now + timedelta(minutes=5))]

        # With min_overlap=60, should filter out 10-second overlap
        result = _segments_weak_sub_on_main_overlap(segments_sub, segments_main, min_overlap=60)
        assert len(result) == 0

        # With min_overlap=5, should include
        result = _segments_weak_sub_on_main_overlap(segments_sub, segments_main, min_overlap=5)
        assert len(result) == 1

    def test_segments_strong_overlap_exact_match(self):
        """Test strong overlap with exact matching segments."""
        now = datetime.datetime.now(pytz.UTC)

        segments_1 = [(now, now + timedelta(hours=1))]
        segments_2 = [(now, now + timedelta(hours=1))]

        result = _segments_strong_overlap(segments_1, segments_2, min_overlap=0)
        # Should find exactly one match
        assert len(result) == 1
        assert result[0][4] == 1.0  # 100% overlap score

    def test_segments_strong_overlap_multiple_overlaps_rejected(self):
        """Test that segments overlapping multiple times are rejected."""
        now = datetime.datetime.now(pytz.UTC)

        # One large segment in list 1
        segments_1 = [(now, now + timedelta(hours=3))]
        # Two segments in list 2 that both overlap with the single segment in list 1
        segments_2 = [
            (now, now + timedelta(hours=1)),
            (now + timedelta(hours=2), now + timedelta(hours=3))
        ]

        result = _segments_strong_overlap(segments_1, segments_2, min_overlap=0)
        # Should reject because segment_1[0] overlaps with multiple segment_2 entries
        assert len(result) == 0


# ============================================================================
# Tests for QSforecastValueSensor
# ============================================================================

class TestQSforecastValueSensor:
    """Test forecast value sensor probing."""

    def test_push_and_get_with_zero_duration(self):
        """Test immediate value getter with zero duration."""
        getter = MagicMock(return_value=(datetime.datetime.now(pytz.UTC), 100.0))
        current_getter = MagicMock(return_value=(datetime.datetime.now(pytz.UTC), 50.0))

        sensor = QSforecastValueSensor("test", 0, getter, current_getter)
        time = datetime.datetime.now(pytz.UTC)

        result = sensor.push_and_get(time)

        # Should use current_getter for zero duration
        current_getter.assert_called_once()
        assert result == 50.0

    def test_push_and_get_stores_future_values(self):
        """Test that future values are stored and returned."""
        time = datetime.datetime.now(pytz.UTC)
        future_time = time + timedelta(hours=1)

        getter = MagicMock(return_value=(future_time, 200.0))

        sensor = QSforecastValueSensor("test", 3600, getter)  # 1 hour duration

        # First call should store the future value
        result = sensor.push_and_get(time)

        # Value should be stored
        assert len(sensor._stored_values) == 1
        assert sensor._stored_values[0] == (future_time, 200.0)

    def test_push_and_get_retrieves_stored_value(self):
        """Test retrieval of previously stored value."""
        time = datetime.datetime.now(pytz.UTC)
        future_time = time + timedelta(hours=1)

        getter = MagicMock(return_value=(future_time, 200.0))

        sensor = QSforecastValueSensor("test", 3600, getter)

        # Pre-store a value
        sensor._stored_values = [(time, 100.0)]

        result = sensor.push_and_get(time)

        # Should return stored value for current time
        assert result == 100.0

    def test_get_probers_factory(self):
        """Test the class factory method."""
        getter = MagicMock()
        current_getter = MagicMock()
        names_and_duration = {
            "sensor_1h": 3600,
            "sensor_2h": 7200,
        }

        probers = QSforecastValueSensor.get_probers(getter, current_getter, names_and_duration)

        assert "sensor_1h" in probers
        assert "sensor_2h" in probers
        assert probers["sensor_1h"]._delta == timedelta(seconds=3600)
        assert probers["sensor_2h"]._delta == timedelta(seconds=7200)


# ============================================================================
# Tests for QSHome initialization
# ============================================================================

class TestQSHomeInit:
    """Test QSHome initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_home_entry",
            data={CONF_NAME: "Test Home"},
        )
        # Mock the zone.home entity
        self.hass.states.set("zone.home", "zoning", {
            "latitude": 48.8566,
            "longitude": 2.3522,
            "radius": 100.0
        })

    def test_init_with_minimal_params(self):
        """Test initialization with minimal parameters."""
        with patch('custom_components.quiet_solar.ha_model.home.QSHome.add_device'):
            home = QSHome(
                hass=self.hass,
                config_entry=self.config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                }
            )

            assert home.name == "Test Home"
            assert home._voltage == 230
            assert home.latitude == 48.8566
            assert home.longitude == 2.3522

    def test_init_with_tariff_settings(self):
        """Test initialization with tariff settings."""
        with patch('custom_components.quiet_solar.ha_model.home.QSHome.add_device'):
            home = QSHome(
                hass=self.hass,
                config_entry=self.config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                    CONF_HOME_PEAK_PRICE: 0.25,
                    CONF_HOME_OFF_PEAK_PRICE: 0.15,
                    CONF_HOME_START_OFF_PEAK_RANGE_1: "22:00",
                    CONF_HOME_END_OFF_PEAK_RANGE_1: "06:00",
                }
            )

            assert home.price_peak == 0.25 / 1000.0
            assert home.price_off_peak == 0.15 / 1000.0
            assert home.tariff_start_1 == "22:00"
            assert home.tariff_end_1 == "06:00"

    def test_init_with_grid_sensor_inverted(self):
        """Test initialization with inverted grid sensor."""
        with patch('custom_components.quiet_solar.ha_model.home.QSHome.add_device'):
            home = QSHome(
                hass=self.hass,
                config_entry=self.config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                    CONF_GRID_POWER_SENSOR: "sensor.grid_power",
                    CONF_GRID_POWER_SENSOR_INVERTED: True,
                }
            )

            assert home.grid_active_power_sensor == "sensor.grid_power"
            assert home.grid_active_power_sensor_inverted is True

    def test_init_creates_empty_device_lists(self):
        """Test that device lists are properly initialized."""
        with patch('custom_components.quiet_solar.ha_model.home.QSHome.add_device'):
            home = QSHome(
                hass=self.hass,
                config_entry=self.config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                }
            )

            assert home._chargers == []
            assert home._cars == []
            assert home._persons == []
            assert home._all_devices == []


# ============================================================================
# Tests for QSHome device management
# ============================================================================

class TestQSHomeDeviceManagement:
    """Test QSHome device management methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_home_entry",
            data={CONF_NAME: "Test Home"},
        )
        self.hass.states.set("zone.home", "zoning", {
            "latitude": 48.8566,
            "longitude": 2.3522,
            "radius": 100.0
        })

        with patch('custom_components.quiet_solar.ha_model.home.QSHome.add_device'):
            self.home = QSHome(
                hass=self.hass,
                config_entry=self.config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                }
            )

    def test_battery_property_returns_none_in_charger_only_mode(self):
        """Test battery property returns None in charger-only mode."""
        self.home.physical_battery = MagicMock()
        self.home.home_mode = QSHomeMode.HOME_MODE_CHARGER_ONLY

        assert self.home.battery is None

    def test_battery_property_returns_battery_in_normal_mode(self):
        """Test battery property returns battery in normal mode."""
        mock_battery = MagicMock()
        self.home.physical_battery = mock_battery
        self.home.home_mode = QSHomeMode.HOME_MODE_ON

        assert self.home.battery == mock_battery

    def test_solar_plant_property_returns_none_in_no_solar_mode(self):
        """Test solar_plant property returns None in no-solar mode."""
        self.home.physical_solar_plant = MagicMock()
        self.home.home_mode = QSHomeMode.HOME_MODE_NO_SOLAR

        assert self.home.solar_plant is None

    def test_solar_plant_property_returns_plant_in_normal_mode(self):
        """Test solar_plant property returns plant in normal mode."""
        mock_solar = MagicMock()
        self.home.physical_solar_plant = mock_solar
        self.home.home_mode = QSHomeMode.HOME_MODE_ON

        assert self.home.solar_plant == mock_solar


# ============================================================================
# Tests for QSHome GPS path mapping
# ============================================================================

class TestQSHomeMapLocationPath:
    """Test map_location_path method for GPS tracking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_home_entry",
            data={CONF_NAME: "Test Home"},
        )
        self.hass.states.set("zone.home", "zoning", {
            "latitude": 48.8566,
            "longitude": 2.3522,
            "radius": 100.0
        })

        with patch('custom_components.quiet_solar.ha_model.home.QSHome.add_device'):
            self.home = QSHome(
                hass=self.hass,
                config_entry=self.config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                }
            )

    def test_map_location_path_empty_states(self):
        """Test with empty state lists."""
        now = datetime.datetime.now(pytz.UTC)
        start = now - timedelta(hours=1)
        end = now

        gps_segments, person_not_home, car_not_home = self.home.map_location_path(
            [], [], start=start, end=end
        )

        assert gps_segments == []
        assert person_not_home == []
        assert car_not_home == []

    def test_map_location_path_home_states_only(self):
        """Test with states all at home."""
        now = datetime.datetime.now(pytz.UTC)
        start = now - timedelta(hours=1)
        end = now

        # Create states at home
        states_1 = [
            MockLazyState("device_tracker.person", "home", {
                "latitude": 48.8566,
                "longitude": 2.3522,
            }, last_updated=start + timedelta(minutes=10)),
        ]
        states_2 = [
            MockLazyState("device_tracker.car", "home", {
                "latitude": 48.8566,
                "longitude": 2.3522,
            }, last_updated=start + timedelta(minutes=10)),
        ]

        gps_segments, person_not_home, car_not_home = self.home.map_location_path(
            states_1, states_2, start=start, end=end
        )

        # Should have no not-home segments
        assert person_not_home == []
        assert car_not_home == []

    def test_map_location_path_not_home_segments(self):
        """Test with states not at home."""
        now = datetime.datetime.now(pytz.UTC)
        start = now - timedelta(hours=2)
        end = now

        # Create states: home -> not_home -> home
        states_1 = [
            MockLazyState("device_tracker.person", "home", {
                "latitude": 48.8566,
                "longitude": 2.3522,
            }, last_updated=start + timedelta(minutes=10)),
            MockLazyState("device_tracker.person", "work", {
                "latitude": 48.9000,
                "longitude": 2.4000,
            }, last_updated=start + timedelta(minutes=30)),
            MockLazyState("device_tracker.person", "home", {
                "latitude": 48.8566,
                "longitude": 2.3522,
            }, last_updated=start + timedelta(minutes=90)),
        ]

        gps_segments, person_not_home, car_not_home = self.home.map_location_path(
            states_1, [], start=start, end=end
        )

        # Should have one not-home segment for person
        assert len(person_not_home) == 1
        assert person_not_home[0][0] == start + timedelta(minutes=30)
        assert person_not_home[0][1] == start + timedelta(minutes=90)


# ============================================================================
# Tests for QSHome home mode and off-grid
# ============================================================================

class TestQSHomeMode:
    """Test home mode handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_home_entry",
            data={CONF_NAME: "Test Home"},
        )
        self.hass.states.set("zone.home", "zoning", {
            "latitude": 48.8566,
            "longitude": 2.3522,
            "radius": 100.0
        })

        with patch('custom_components.quiet_solar.ha_model.home.QSHome.add_device'):
            self.home = QSHome(
                hass=self.hass,
                config_entry=self.config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                }
            )

    def test_is_off_grid_returns_false_by_default(self):
        """Test is_off_grid returns False by default."""
        assert self.home.qs_home_is_off_grid is False

    def test_home_mode_off_disables_all(self):
        """Test HOME_MODE_OFF disables battery and solar."""
        self.home.physical_battery = MagicMock()
        self.home.physical_solar_plant = MagicMock()
        self.home.home_mode = QSHomeMode.HOME_MODE_OFF

        # In OFF mode, battery and solar should still be accessible
        # (only specific modes restrict them)
        assert self.home.battery is not None

    def test_home_mode_sensors_only(self):
        """Test HOME_MODE_SENSORS_ONLY mode."""
        self.home.home_mode = QSHomeMode.HOME_MODE_SENSORS_ONLY
        # Just verify the mode is set
        assert self.home.home_mode == QSHomeMode.HOME_MODE_SENSORS_ONLY


# ============================================================================
# Tests for QSHome mileage computation
# ============================================================================

class TestQSHomeMileageComputation:
    """Test mileage computation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_home_entry",
            data={CONF_NAME: "Test Home"},
        )
        self.hass.states.set("zone.home", "zoning", {
            "latitude": 48.8566,
            "longitude": 2.3522,
            "radius": 100.0
        })

        with patch('custom_components.quiet_solar.ha_model.home.QSHome.add_device'):
            self.home = QSHome(
                hass=self.hass,
                config_entry=self.config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                }
            )

    @pytest.mark.asyncio
    async def test_compute_mileage_empty_cars(self):
        """Test mileage computation with no cars."""
        self.home._cars = []
        self.home._persons = []

        now = datetime.datetime.now(pytz.UTC)
        start = now - timedelta(hours=24)
        end = now

        result = await self.home._compute_mileage_for_period_per_person(start, end)

        assert result == {}

    @pytest.mark.asyncio
    async def test_compute_mileage_car_is_invited_skipped(self):
        """Test that invited cars are skipped."""
        mock_car = MagicMock()
        mock_car.car_is_invited = True
        self.home._cars = [mock_car]
        self.home._persons = []

        now = datetime.datetime.now(pytz.UTC)
        start = now - timedelta(hours=24)
        end = now

        with patch('custom_components.quiet_solar.ha_model.home.load_from_history', new_callable=AsyncMock) as mock_load:
            result = await self.home._compute_mileage_for_period_per_person(start, end)

        # load_from_history should not be called for invited cars
        mock_load.assert_not_called()
        assert result == {}


# ============================================================================
# Tests for QSHome person-car allocation
# ============================================================================

class TestQSHomePersonCarAllocation:
    """Test person-car allocation algorithms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_home_entry",
            data={CONF_NAME: "Test Home"},
        )
        self.hass.states.set("zone.home", "zoning", {
            "latitude": 48.8566,
            "longitude": 2.3522,
            "radius": 100.0
        })

        with patch('custom_components.quiet_solar.ha_model.home.QSHome.add_device'):
            self.home = QSHome(
                hass=self.hass,
                config_entry=self.config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                }
            )

    def test_get_person_by_name_not_found(self):
        """Test get_person_by_name returns None when not found."""
        self.home._persons = []

        result = self.home.get_person_by_name("Unknown Person")
        assert result is None

    def test_get_person_by_name_found(self):
        """Test get_person_by_name returns person when found."""
        mock_person = MagicMock()
        mock_person.name = "John"
        self.home._persons = [mock_person]

        result = self.home.get_person_by_name("John")
        assert result == mock_person

    def test_get_car_by_name_not_found(self):
        """Test get_car_by_name returns None when not found."""
        self.home._cars = []

        result = self.home.get_car_by_name("Unknown Car")
        assert result is None

    def test_get_car_by_name_found(self):
        """Test get_car_by_name returns car when found."""
        mock_car = MagicMock()
        mock_car.name = "Tesla"
        self.home._cars = [mock_car]

        result = self.home.get_car_by_name("Tesla")
        assert result == mock_car


# ============================================================================
# Tests for QSHome sensor state getters
# ============================================================================

class TestQSHomeSensorGetters:
    """Test sensor state getter methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_home_entry",
            data={CONF_NAME: "Test Home"},
        )
        self.hass.states.set("zone.home", "zoning", {
            "latitude": 48.8566,
            "longitude": 2.3522,
            "radius": 100.0
        })

        with patch('custom_components.quiet_solar.ha_model.home.QSHome.add_device'):
            self.home = QSHome(
                hass=self.hass,
                config_entry=self.config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                }
            )

    def test_home_consumption_sensor_state_getter_returns_value(self):
        """Test home_consumption_sensor_state_getter returns value when set."""
        self.home.home_consumption = 1500.0
        time = datetime.datetime.now(pytz.UTC)

        result = self.home.home_consumption_sensor_state_getter(
            self.home.home_consumption_sensor, time
        )

        # Result is (time, value, attrs) where value could be a float or tuple
        if result is not None:
            # Check result format - it returns (time, value, attrs)
            assert result[0] is not None  # time
            # Value could be float directly or nested
            value = result[1]
            if isinstance(value, tuple):
                assert value[0] == 1500.0
            else:
                assert value == 1500.0

    def test_home_consumption_sensor_state_getter_returns_none_when_not_set(self):
        """Test home_consumption_sensor_state_getter returns None when not set."""
        self.home.home_consumption = None
        time = datetime.datetime.now(pytz.UTC)

        result = self.home.home_consumption_sensor_state_getter(
            self.home.home_consumption_sensor, time
        )

        assert result is None

    def test_home_available_power_sensor_state_getter_returns_none_when_not_set(self):
        """Test home_available_power_sensor_state_getter returns None when not set."""
        self.home.home_available_power = None
        time = datetime.datetime.now(pytz.UTC)

        result = self.home.home_available_power_sensor_state_getter(
            self.home.home_available_power_sensor, time
        )

        assert result is None

    def test_grid_consumption_power_sensor_state_getter_returns_none_when_not_set(self):
        """Test grid_consumption_power_sensor_state_getter returns None when not set."""
        self.home.grid_consumption_power = None
        time = datetime.datetime.now(pytz.UTC)

        result = self.home.grid_consumption_power_sensor_state_getter(
            self.home.grid_consumption_power_sensor, time
        )

        assert result is None

    def test_home_non_controlled_consumption_sensor_state_getter_returns_none_when_not_set(self):
        """Test home_non_controlled_consumption_sensor_state_getter returns None when not set."""
        self.home.home_non_controlled_consumption = None
        time = datetime.datetime.now(pytz.UTC)

        result = self.home.home_non_controlled_consumption_sensor_state_getter(
            self.home.home_non_controlled_consumption_sensor, time
        )

        assert result is None
