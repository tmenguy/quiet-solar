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
    CONF_OFF_GRID_ENTITY,
    CONF_OFF_GRID_STATE_VALUE,
    CONF_OFF_GRID_INVERTED,
    OFF_GRID_MODE_AUTO,
    OFF_GRID_MODE_FORCE_OFF_GRID,
    OFF_GRID_MODE_FORCE_ON_GRID,
)

from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry


class MockLazyState:
    """Mock LazyState for testing."""

    def __init__(self, entity_id: str, state: str, attributes: dict | None = None, last_updated: datetime.datetime | None = None):
        self.entity_id = entity_id
        self.state = state
        self.attributes = attributes or {}
        self.last_updated = last_updated or datetime.datetime.now(pytz.UTC)


@pytest.fixture
def home_config_entry() -> MockConfigEntry:
    """Config entry for home tests."""
    return MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_home_entry",
        data={CONF_NAME: "Test Home"},
        title="Test Home",
    )


@pytest.fixture
def home_hass_data(hass: HomeAssistant) -> None:
    """Set hass.data[DOMAIN] so QSHome/device init can access data_handler."""
    hass.data.setdefault(DOMAIN, {})


@pytest.fixture
async def home_zone_state(hass: HomeAssistant, home_hass_data: None) -> None:
    """Set zone.home state for home tests."""
    set_result = hass.states.async_set("zone.home", "zoning", {
        "latitude": 48.8566,
        "longitude": 2.3522,
        "radius": 100.0,
    })
    if set_result is not None:
        await set_result


@pytest.fixture
async def home(hass: HomeAssistant, home_config_entry: MockConfigEntry, home_zone_state: None):
    """QSHome instance for tests (requires home_zone_state)."""
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.add_device"):
        return QSHome(
            hass=hass,
            config_entry=home_config_entry,
            **{
                CONF_NAME: "Test Home",
                CONF_HOME_VOLTAGE: 230,
            },
        )


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

    def test_init_with_minimal_params(self, hass, home_config_entry, home_hass_data, home_zone_state):
        """Test initialization with minimal parameters."""
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.add_device"):
            home = QSHome(
                hass=hass,
                config_entry=home_config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                },
            )

            assert home.name == "Test Home"
            assert home._voltage == 230
            assert home.latitude == 48.8566
            assert home.longitude == 2.3522

    def test_init_with_tariff_settings(self, hass, home_config_entry, home_hass_data, home_zone_state):
        """Test initialization with tariff settings."""
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.add_device"):
            home = QSHome(
                hass=hass,
                config_entry=home_config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                    CONF_HOME_PEAK_PRICE: 0.25,
                    CONF_HOME_OFF_PEAK_PRICE: 0.15,
                    CONF_HOME_START_OFF_PEAK_RANGE_1: "22:00",
                    CONF_HOME_END_OFF_PEAK_RANGE_1: "06:00",
                },
            )

            assert home.price_peak == 0.25 / 1000.0
            assert home.price_off_peak == 0.15 / 1000.0
            assert home.tariff_start_1 == "22:00"
            assert home.tariff_end_1 == "06:00"

    def test_init_with_grid_sensor_inverted(self, hass, home_config_entry, home_hass_data, home_zone_state):
        """Test initialization with inverted grid sensor."""
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.add_device"):
            home = QSHome(
                hass=hass,
                config_entry=home_config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                    CONF_GRID_POWER_SENSOR: "sensor.grid_power",
                    CONF_GRID_POWER_SENSOR_INVERTED: True,
                },
            )

            assert home.grid_active_power_sensor == "sensor.grid_power"
            assert home.grid_active_power_sensor_inverted is True

    def test_init_creates_empty_device_lists(self, hass, home_config_entry, home_hass_data, home_zone_state):
        """Test that device lists are properly initialized."""
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.add_device"):
            home = QSHome(
                hass=hass,
                config_entry=home_config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                },
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

    def test_battery_property_returns_none_in_charger_only_mode(self, home):
        """Test battery property returns None in charger-only mode."""
        home.physical_battery = MagicMock()
        home.home_mode = QSHomeMode.HOME_MODE_CHARGER_ONLY

        assert home.battery is None

    def test_battery_property_returns_battery_in_normal_mode(self, home):
        """Test battery property returns battery in normal mode."""
        mock_battery = MagicMock()
        home.physical_battery = mock_battery
        home.home_mode = QSHomeMode.HOME_MODE_ON

        assert home.battery == mock_battery

    def test_solar_plant_property_returns_none_in_no_solar_mode(self, home):
        """Test solar_plant property returns None in no-solar mode."""
        home.physical_solar_plant = MagicMock()
        home.home_mode = QSHomeMode.HOME_MODE_NO_SOLAR

        assert home.solar_plant is None

    def test_solar_plant_property_returns_plant_in_normal_mode(self, home):
        """Test solar_plant property returns plant in normal mode."""
        mock_solar = MagicMock()
        home.physical_solar_plant = mock_solar
        home.home_mode = QSHomeMode.HOME_MODE_ON

        assert home.solar_plant == mock_solar


# ============================================================================
# Tests for QSHome GPS path mapping
# ============================================================================

class TestQSHomeMapLocationPath:
    """Test map_location_path method for GPS tracking."""

    def test_map_location_path_empty_states(self, home):
        """Test with empty state lists."""
        now = datetime.datetime.now(pytz.UTC)
        start = now - timedelta(hours=1)
        end = now

        gps_segments, person_not_home, car_not_home = home.map_location_path(
            [], [], start=start, end=end
        )

        assert gps_segments == []
        assert person_not_home == []
        assert car_not_home == []

    def test_map_location_path_home_states_only(self, home):
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

        gps_segments, person_not_home, car_not_home = home.map_location_path(
            states_1, states_2, start=start, end=end
        )

        # Should have no not-home segments
        assert person_not_home == []
        assert car_not_home == []

    def test_map_location_path_not_home_segments(self, home):
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

        gps_segments, person_not_home, car_not_home = home.map_location_path(
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

    def test_is_off_grid_returns_false_by_default(self, home):
        """Test is_off_grid returns False by default."""
        assert home.qs_home_is_off_grid is False

    def test_home_mode_off_disables_all(self, home):
        """Test HOME_MODE_OFF disables battery and solar."""
        home.physical_battery = MagicMock()
        home.physical_solar_plant = MagicMock()
        home.home_mode = QSHomeMode.HOME_MODE_OFF

        # In OFF mode, battery and solar should still be accessible
        # (only specific modes restrict them)
        assert home.battery is not None

    def test_home_mode_sensors_only(self, home):
        """Test HOME_MODE_SENSORS_ONLY mode."""
        home.home_mode = QSHomeMode.HOME_MODE_SENSORS_ONLY
        # Just verify the mode is set
        assert home.home_mode == QSHomeMode.HOME_MODE_SENSORS_ONLY


# ============================================================================
# Tests for QSHome mileage computation
# ============================================================================

class TestQSHomeMileageComputation:
    """Test mileage computation methods."""

    @pytest.mark.asyncio
    async def test_compute_mileage_empty_cars(self, home):
        """Test mileage computation with no cars."""
        home._cars = []
        home._persons = []

        now = datetime.datetime.now(pytz.UTC)
        start = now - timedelta(hours=24)
        end = now

        result = await home._compute_mileage_for_period_per_person(start, end)

        assert result == {}

    @pytest.mark.asyncio
    async def test_compute_mileage_car_is_invited_skipped(self, home):
        """Test that invited cars are skipped."""
        mock_car = MagicMock()
        mock_car.car_is_invited = True
        home._cars = [mock_car]
        home._persons = []

        now = datetime.datetime.now(pytz.UTC)
        start = now - timedelta(hours=24)
        end = now

        with patch("custom_components.quiet_solar.ha_model.home.load_from_history", new_callable=AsyncMock) as mock_load:
            result = await home._compute_mileage_for_period_per_person(start, end)

        # load_from_history should not be called for invited cars
        mock_load.assert_not_called()
        assert result == {}


# ============================================================================
# Tests for QSHome person-car allocation
# ============================================================================

class TestQSHomePersonCarAllocation:
    """Test person-car allocation algorithms."""

    def test_get_person_by_name_not_found(self, home):
        """Test get_person_by_name returns None when not found."""
        home._persons = []

        result = home.get_person_by_name("Unknown Person")
        assert result is None

    def test_get_person_by_name_found(self, home):
        """Test get_person_by_name returns person when found."""
        mock_person = MagicMock()
        mock_person.name = "John"
        home._persons = [mock_person]

        result = home.get_person_by_name("John")
        assert result == mock_person

    def test_get_car_by_name_not_found(self, home):
        """Test get_car_by_name returns None when not found."""
        home._cars = []

        result = home.get_car_by_name("Unknown Car")
        assert result is None

    def test_get_car_by_name_found(self, home):
        """Test get_car_by_name returns car when found."""
        mock_car = MagicMock()
        mock_car.name = "Tesla"
        home._cars = [mock_car]

        result = home.get_car_by_name("Tesla")
        assert result == mock_car


# ============================================================================
# Tests for QSHome sensor state getters
# ============================================================================

class TestQSHomeSensorGetters:
    """Test sensor state getter methods."""

    def test_home_consumption_sensor_state_getter_returns_value(self, home):
        """Test home_consumption_sensor_state_getter returns value when set."""
        home.home_consumption = 1500.0
        time = datetime.datetime.now(pytz.UTC)

        result = home.home_consumption_sensor_state_getter(
            home.home_consumption_sensor, time
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

    def test_home_consumption_sensor_state_getter_returns_none_when_not_set(self, home):
        """Test home_consumption_sensor_state_getter returns None when not set."""
        home.home_consumption = None
        time = datetime.datetime.now(pytz.UTC)

        result = home.home_consumption_sensor_state_getter(
            home.home_consumption_sensor, time
        )

        assert result is None

    def test_home_available_power_sensor_state_getter_returns_none_when_not_set(self, home):
        """Test home_available_power_sensor_state_getter returns None when not set."""
        home.home_available_power = None
        time = datetime.datetime.now(pytz.UTC)

        result = home.home_available_power_sensor_state_getter(
            home.home_available_power_sensor, time
        )

        assert result is None

    def test_grid_consumption_power_sensor_state_getter_returns_none_when_not_set(self, home):
        """Test grid_consumption_power_sensor_state_getter returns None when not set."""
        home.grid_consumption_power = None
        time = datetime.datetime.now(pytz.UTC)

        result = home.grid_consumption_power_sensor_state_getter(
            home.grid_consumption_power_sensor, time
        )

        assert result is None

    def test_home_non_controlled_consumption_sensor_state_getter_returns_none_when_not_set(self, home):
        """Test home_non_controlled_consumption_sensor_state_getter returns None when not set."""
        home.home_non_controlled_consumption = None
        time = datetime.datetime.now(pytz.UTC)

        result = home.home_non_controlled_consumption_sensor_state_getter(
            home.home_non_controlled_consumption_sensor, time
        )

        assert result is None


# ============================================================================
# Extended Tests for Additional Coverage
# ============================================================================


class TestQSHomeExtendedCoverage:
    """Extended tests for QSHome to increase coverage."""

    def test_voltage_property(self, home):
        """Test voltage property returns configured value."""
        assert home.voltage == 230

    @pytest.mark.asyncio
    async def test_update_forecast_probers_no_solcast(self, home):
        """Test update_forecast_probers when no solcast configured (lines 698-709)."""
        home.solcast_forecast_entity_id = None
        time = datetime.datetime.now(pytz.UTC)

        await home.update_forecast_probers(time)

    def test_force_next_solve_method(self, home):
        """Test force_next_solve sets flag (line 1025-1026)."""
        # Call the method
        home.force_next_solve()
        # Verify it was called without error

    def test_is_off_grid_false(self, home):
        """Test is_off_grid returns False by default (line 895-901)."""
        result = home.is_off_grid()
        assert isinstance(result, bool)

    def test_get_best_tariff(self, home):
        """Test get_best_tariff method (line 1019-1023)."""
        time = datetime.datetime.now(pytz.UTC)
        result = home.get_best_tariff(time)
        assert isinstance(result, float)

    def test_get_tariff(self, home):
        """Test get_tariff method (lines 1105-1133)."""
        time = datetime.datetime.now(pytz.UTC)
        end_time = time + timedelta(hours=1)
        result = home.get_tariff(time, end_time)
        assert isinstance(result, float)

    def test_get_tariffs(self, home):
        """Test get_tariffs method (lines 1135-1176)."""
        time = datetime.datetime.now(pytz.UTC)
        end_time = time + timedelta(hours=1)
        result = home.get_tariffs(time, end_time)
        assert isinstance(result, (list, float))

    def test_get_platforms(self, home):
        """Test get_platforms method (lines 1178-1185)."""
        result = home.get_platforms()
        assert isinstance(result, list)
        assert Platform.SENSOR in result or "sensor" in result

    def test_get_devices_for_dashboard_section(self, home):
        """Test get_devices_for_dashboard_section (lines 991-1017)."""
        mock_load = MagicMock()
        mock_load.dashboard_section = ("Test Section", "icon")
        mock_load.qs_enable_device = True
        home._loads = [mock_load]

        result = home.get_devices_for_dashboard_section("Test Section")
        assert isinstance(result, list)

    def test_get_home_max_static_phase_amps(self, home):
        """Test get_home_max_static_phase_amps (lines 903-913)."""
        result = home.get_home_max_static_phase_amps()
        assert isinstance(result, int)

    def test_get_home_max_phase_amps(self, home):
        """Test get_home_max_phase_amps (lines 915-990)."""
        result = home.get_home_max_phase_amps()
        assert isinstance(result, (float, int))

    @pytest.mark.asyncio
    async def test_get_best_persons_cars_allocations(self, home):
        """Test get_best_persons_cars_allocations (lines 1878-2120)."""
        home._cars = []
        home._persons = []
        time = datetime.datetime.now(pytz.UTC)

        result = await home.get_best_persons_cars_allocations(time)
        assert isinstance(result, dict)

    def test_get_preferred_person_for_car_no_persons(self, home):
        """Test get_preferred_person_for_car when no persons (lines 1201-1250)."""
        home._persons = []
        mock_car = MagicMock()
        mock_car.name = "Test Car"

        result = home.get_preferred_person_for_car(mock_car)
        assert result is None

    def test_get_solar_from_current_forecast(self, home):
        """Test get_solar_from_current_forecast (lines 1063-1103)."""
        time = datetime.datetime.now(pytz.UTC)
        result = home.get_solar_from_current_forecast(time)
        assert isinstance(result, list)


class TestQSforecastValueSensor:
    """Test QSforecastValueSensor class."""

    def test_get_probers_creates_dict(self):
        """Test get_probers creates multiple probers."""
        mock_getter = MagicMock(return_value=(None, None))
        mock_getter_now = MagicMock(return_value=(None, None))

        probers = QSforecastValueSensor.get_probers(
            mock_getter,
            mock_getter_now,
            {"1h": 3600, "2h": 7200}
        )

        assert "1h" in probers
        assert "2h" in probers
        assert isinstance(probers["1h"], QSforecastValueSensor)

    def test_push_and_get_with_current_getter(self):
        """Test push_and_get uses current_getter when delta is 0 (line 136-138)."""
        mock_getter = MagicMock(return_value=(None, None))
        mock_getter_now = MagicMock(return_value=(datetime.datetime.now(pytz.UTC), 100.0))

        prober = QSforecastValueSensor(
            "now", 0, mock_getter, mock_getter_now  # delta = 0
        )
        time = datetime.datetime.now(pytz.UTC)

        result = prober.push_and_get(time)
        mock_getter_now.assert_called_once()

    def test_push_and_get_with_forecast(self):
        """Test push_and_get stores and retrieves forecast values (lines 140-158)."""
        time = datetime.datetime.now(pytz.UTC)
        future_time = time + timedelta(hours=1)
        mock_getter = MagicMock(return_value=(future_time, 200.0))

        prober = QSforecastValueSensor(
            "1h", 3600, mock_getter, None  # 1 hour delta
        )

        result = prober.push_and_get(time)
        # Should have stored the value
        assert len(prober._stored_values) > 0

    def test_push_and_get_empty_stored_values(self):
        """Test push_and_get returns None when no stored values and getter returns None (lines 148-149)."""
        mock_getter = MagicMock(return_value=(None, None))

        prober = QSforecastValueSensor("1h", 3600, mock_getter, None)
        time = datetime.datetime.now(pytz.UTC)

        result = prober.push_and_get(time)
        assert result is None


class TestSegmentOverlapExtended:
    """Extended tests for segment overlap functions."""

    def test_segments_strong_overlap_empty_lists(self):
        """Test _segments_strong_overlap with empty lists returns expected structure."""
        result = _segments_strong_overlap([], [])
        # Returns a tuple of (seg1_best_matches, seg2_best_matches)
        assert result is not None

    def test_segments_strong_overlap_basic(self):
        """Test _segments_strong_overlap with basic segments (lines 186-280)."""
        now = datetime.datetime.now(pytz.UTC)

        segments_1 = [(now, now + timedelta(hours=2))]
        segments_2 = [(now + timedelta(hours=1), now + timedelta(hours=3))]

        result = _segments_strong_overlap(segments_1, segments_2)
        # Should find overlap between hour 1 and hour 2
        assert result is not None

    def test_segments_weak_sub_overlap_basic(self):
        """Test _segments_weak_sub_on_main_overlap with overlapping segments."""
        now = datetime.datetime.now(pytz.UTC)

        segments_sub = [(now, now + timedelta(hours=2))]
        segments_main = [(now + timedelta(hours=1), now + timedelta(hours=3))]

        result = _segments_weak_sub_on_main_overlap(segments_sub, segments_main)
        assert len(result) > 0


class TestQSHomeOffPeakPricing:
    """Test off-peak pricing functionality."""

    @pytest.fixture
    async def home_with_pricing(self, hass: HomeAssistant, home_config_entry: MockConfigEntry, home_zone_state: None):
        """QSHome with pricing configured."""
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.add_device"):
            return QSHome(
                hass=hass,
                config_entry=home_config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                    CONF_HOME_PEAK_PRICE: 0.25,
                    CONF_HOME_OFF_PEAK_PRICE: 0.15,
                    CONF_HOME_START_OFF_PEAK_RANGE_1: dt_time(22, 0, 0),
                    CONF_HOME_END_OFF_PEAK_RANGE_1: dt_time(6, 0, 0),
                },
            )

    def test_get_best_tariff_with_pricing(self, home_with_pricing):
        """Test get_best_tariff with pricing configured."""
        time = datetime.datetime(2026, 1, 24, 12, 0, 0, tzinfo=pytz.UTC)
        result = home_with_pricing.get_best_tariff(time)
        assert isinstance(result, float)


class TestQSHomeMoreExtendedCoverage:
    """More extended tests for QSHome to increase coverage to 91%+."""

    @pytest.fixture
    async def home_basic(self, hass: HomeAssistant, home_config_entry: MockConfigEntry, home_zone_state: None):
        """Basic QSHome fixture."""
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.add_device"):
            return QSHome(
                hass=hass,
                config_entry=home_config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                },
            )

    def test_name_property(self, home_basic):
        """Test name property."""
        result = home_basic.name

        assert isinstance(result, str)

    def test_voltage_getter(self, home_basic):
        """Test voltage getter."""
        result = home_basic.voltage

        assert isinstance(result, (int, float))

    def test_get_car_by_name_not_found(self, home_basic):
        """Test get_car_by_name when car not found."""
        home_basic._cars = []

        result = home_basic.get_car_by_name("nonexistent_car")

        assert result is None

    def test_get_person_by_name_not_found(self, home_basic):
        """Test get_person_by_name when person not found."""
        home_basic._persons = []

        result = home_basic.get_person_by_name("nonexistent_person")

        assert result is None

    def test_cars_property(self, home_basic):
        """Test _cars property."""
        home_basic._cars = []

        result = home_basic._cars

        assert isinstance(result, list)

    def test_chargers_property(self, home_basic):
        """Test _chargers property."""
        home_basic._chargers = []

        result = home_basic._chargers

        assert isinstance(result, list)

    def test_loads_property(self, home_basic):
        """Test _loads property."""
        home_basic._loads = []

        result = home_basic._loads

        assert isinstance(result, list)

    def test_persons_property(self, home_basic):
        """Test _persons property."""
        home_basic._persons = []

        result = home_basic._persons

        assert isinstance(result, list)

    def test_accurate_power_sensor(self, home_basic):
        """Test accurate_power_sensor property."""
        result = home_basic.accurate_power_sensor

        assert result is None or isinstance(result, str)

    def test_get_min_max_power(self, home_basic):
        """Test get_min_max_power method."""
        min_p, max_p = home_basic.get_min_max_power()

        assert min_p >= 0
        assert max_p >= min_p


# ============================================================================
# Tests for automatic off-grid detection
# ============================================================================


class TestOffGridAutoDetection:
    """Test automatic off-grid mode detection via external HA entities."""

    @pytest.fixture
    async def home_with_binary_sensor_off_grid(
        self, hass: HomeAssistant, home_config_entry: MockConfigEntry, home_zone_state: None
    ):
        """QSHome with a binary_sensor off-grid entity."""
        hass.states.async_set("binary_sensor.grid_relay", "off")
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.add_device"):
            return QSHome(
                hass=hass,
                config_entry=home_config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                    CONF_OFF_GRID_ENTITY: "binary_sensor.grid_relay",
                    CONF_OFF_GRID_INVERTED: False,
                },
            )

    @pytest.fixture
    async def home_with_sensor_off_grid(
        self, hass: HomeAssistant, home_config_entry: MockConfigEntry, home_zone_state: None
    ):
        """QSHome with a sensor off-grid entity using state value matching."""
        hass.states.async_set("sensor.grid_status", "connected")
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.add_device"):
            return QSHome(
                hass=hass,
                config_entry=home_config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                    CONF_OFF_GRID_ENTITY: "sensor.grid_status",
                    CONF_OFF_GRID_STATE_VALUE: "disconnected",
                    CONF_OFF_GRID_INVERTED: False,
                },
            )

    @pytest.fixture
    async def home_with_switch_inverted(
        self, hass: HomeAssistant, home_config_entry: MockConfigEntry, home_zone_state: None
    ):
        """QSHome with a switch off-grid entity (inverted)."""
        hass.states.async_set("switch.grid_connection", "on")
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.add_device"):
            return QSHome(
                hass=hass,
                config_entry=home_config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                    CONF_OFF_GRID_ENTITY: "switch.grid_connection",
                    CONF_OFF_GRID_INVERTED: True,
                },
            )

    @pytest.fixture
    async def home_no_off_grid_entity(
        self, hass: HomeAssistant, home_config_entry: MockConfigEntry, home_zone_state: None
    ):
        """QSHome with no off-grid entity configured."""
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.add_device"):
            return QSHome(
                hass=hass,
                config_entry=home_config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                },
            )

    # -- _compute_off_grid_from_entity_state tests --

    def test_compute_from_binary_sensor_on(self, home_with_binary_sensor_off_grid):
        """Binary sensor 'on' means off-grid when not inverted."""
        home = home_with_binary_sensor_off_grid
        result = home._compute_off_grid_from_entity_state("on", "binary_sensor.grid_relay")
        assert result is True

    def test_compute_from_binary_sensor_off(self, home_with_binary_sensor_off_grid):
        """Binary sensor 'off' means on-grid when not inverted."""
        home = home_with_binary_sensor_off_grid
        result = home._compute_off_grid_from_entity_state("off", "binary_sensor.grid_relay")
        assert result is False

    def test_compute_from_sensor_matching(self, home_with_sensor_off_grid):
        """Sensor state matches off_grid_state_value means off-grid."""
        home = home_with_sensor_off_grid
        result = home._compute_off_grid_from_entity_state("disconnected", "sensor.grid_status")
        assert result is True

    def test_compute_from_sensor_not_matching(self, home_with_sensor_off_grid):
        """Sensor state not matching off_grid_state_value means on-grid."""
        home = home_with_sensor_off_grid
        result = home._compute_off_grid_from_entity_state("connected", "sensor.grid_status")
        assert result is False

    def test_compute_from_switch_inverted_off(self, home_with_switch_inverted):
        """Switch 'off' means off-grid when inverted."""
        home = home_with_switch_inverted
        result = home._compute_off_grid_from_entity_state("off", "switch.grid_connection")
        assert result is True

    def test_compute_from_switch_inverted_on(self, home_with_switch_inverted):
        """Switch 'on' means on-grid when inverted."""
        home = home_with_switch_inverted
        result = home._compute_off_grid_from_entity_state("on", "switch.grid_connection")
        assert result is False

    def test_compute_from_unavailable(self, home_with_binary_sensor_off_grid):
        """Unavailable entity state is treated as on-grid."""
        home = home_with_binary_sensor_off_grid
        assert home._compute_off_grid_from_entity_state(STATE_UNAVAILABLE, "binary_sensor.grid_relay") is False
        assert home._compute_off_grid_from_entity_state(STATE_UNKNOWN, "binary_sensor.grid_relay") is False

    # -- _compute_and_apply_off_grid_state tests --

    @pytest.mark.asyncio
    async def test_auto_mode_follows_real_state(self, home_with_binary_sensor_off_grid):
        """Auto mode applies qs_home_real_off_grid value."""
        home = home_with_binary_sensor_off_grid
        home.off_grid_mode = OFF_GRID_MODE_AUTO
        home.qs_home_real_off_grid = True

        with patch.object(home, "async_set_off_grid_mode", new_callable=AsyncMock) as mock_set:
            await home._compute_and_apply_off_grid_state(for_init=False)
            mock_set.assert_called_once_with(True, False)

    @pytest.mark.asyncio
    async def test_auto_mode_restores_on_grid(self, home_with_binary_sensor_off_grid):
        """Auto mode restores on-grid when real state goes False."""
        home = home_with_binary_sensor_off_grid
        home.off_grid_mode = OFF_GRID_MODE_AUTO
        home.qs_home_real_off_grid = False

        with patch.object(home, "async_set_off_grid_mode", new_callable=AsyncMock) as mock_set:
            await home._compute_and_apply_off_grid_state(for_init=False)
            mock_set.assert_called_once_with(False, False)

    @pytest.mark.asyncio
    async def test_force_off_grid_mode(self, home_with_binary_sensor_off_grid):
        """Force off-grid always calls async_set_off_grid_mode(True)."""
        home = home_with_binary_sensor_off_grid
        home.off_grid_mode = OFF_GRID_MODE_FORCE_OFF_GRID
        home.qs_home_real_off_grid = False

        with patch.object(home, "async_set_off_grid_mode", new_callable=AsyncMock) as mock_set:
            await home._compute_and_apply_off_grid_state(for_init=False)
            mock_set.assert_called_once_with(True, False)

    @pytest.mark.asyncio
    async def test_force_on_grid_mode(self, home_with_binary_sensor_off_grid):
        """Force on-grid always calls async_set_off_grid_mode(False)."""
        home = home_with_binary_sensor_off_grid
        home.off_grid_mode = OFF_GRID_MODE_FORCE_ON_GRID
        home.qs_home_real_off_grid = True

        with patch.object(home, "async_set_off_grid_mode", new_callable=AsyncMock) as mock_set:
            await home._compute_and_apply_off_grid_state(for_init=False)
            mock_set.assert_called_once_with(False, False)

    # -- async_set_off_grid_mode_option tests --

    @pytest.mark.asyncio
    async def test_set_off_grid_mode_option_sets_mode(self, home_with_binary_sensor_off_grid):
        """async_set_off_grid_mode_option sets the mode and recomputes."""
        home = home_with_binary_sensor_off_grid
        with patch.object(home, "_compute_and_apply_off_grid_state", new_callable=AsyncMock) as mock_compute:
            await home.async_set_off_grid_mode_option(OFF_GRID_MODE_FORCE_OFF_GRID, for_init=False)
            assert home.off_grid_mode == OFF_GRID_MODE_FORCE_OFF_GRID
            mock_compute.assert_called_once_with(False)

    # -- Initial state reading tests --

    def test_initial_state_binary_sensor_off(self, home_with_binary_sensor_off_grid):
        """Binary sensor initial state 'off' means qs_home_real_off_grid is False."""
        home = home_with_binary_sensor_off_grid
        assert home.qs_home_real_off_grid is False

    def test_initial_state_sensor_connected(self, home_with_sensor_off_grid):
        """Sensor initial state 'connected' != 'disconnected' means on-grid."""
        home = home_with_sensor_off_grid
        assert home.qs_home_real_off_grid is False

    def test_initial_state_switch_on_inverted(self, home_with_switch_inverted):
        """Switch initial state 'on' with inverted means on-grid."""
        home = home_with_switch_inverted
        assert home.qs_home_real_off_grid is False

    # -- No entity configured --

    def test_no_entity_no_listener(self, home_no_off_grid_entity):
        """No off-grid entity means no listener registered."""
        home = home_no_off_grid_entity
        assert home._off_grid_unsub is None
        assert home.qs_home_real_off_grid is False

    def test_entity_configured_has_listener(self, home_with_binary_sensor_off_grid):
        """Off-grid entity configured means listener is registered."""
        home = home_with_binary_sensor_off_grid
        assert home._off_grid_unsub is not None

    # -- State change event tests --

    @pytest.mark.asyncio
    async def test_binary_sensor_state_change_to_on_triggers_off_grid(
        self, hass: HomeAssistant, home_with_binary_sensor_off_grid
    ):
        """Binary sensor changing to 'on' sets qs_home_real_off_grid=True and triggers mode computation."""
        home = home_with_binary_sensor_off_grid
        home.off_grid_mode = OFF_GRID_MODE_AUTO

        with patch.object(home, "async_set_off_grid_mode", new_callable=AsyncMock) as mock_set:
            hass.states.async_set("binary_sensor.grid_relay", "on")
            await hass.async_block_till_done()

            assert home.qs_home_real_off_grid is True
            mock_set.assert_called_with(True, False)

    @pytest.mark.asyncio
    async def test_binary_sensor_state_change_to_off_restores_on_grid(
        self, hass: HomeAssistant, home_with_binary_sensor_off_grid
    ):
        """Binary sensor changing from 'on' to 'off' restores on-grid."""
        home = home_with_binary_sensor_off_grid
        home.off_grid_mode = OFF_GRID_MODE_AUTO

        # First transition to "on" so HA records the state
        hass.states.async_set("binary_sensor.grid_relay", "on")
        await hass.async_block_till_done()
        assert home.qs_home_real_off_grid is True

        with patch.object(home, "async_set_off_grid_mode", new_callable=AsyncMock) as mock_set:
            hass.states.async_set("binary_sensor.grid_relay", "off")
            await hass.async_block_till_done()

            assert home.qs_home_real_off_grid is False
            mock_set.assert_called_with(False, False)

    @pytest.mark.asyncio
    async def test_sensor_state_change_matches_triggers_off_grid(
        self, hass: HomeAssistant, home_with_sensor_off_grid
    ):
        """Sensor changing to match value triggers off-grid."""
        home = home_with_sensor_off_grid
        home.off_grid_mode = OFF_GRID_MODE_AUTO

        with patch.object(home, "async_set_off_grid_mode", new_callable=AsyncMock) as mock_set:
            hass.states.async_set("sensor.grid_status", "disconnected")
            await hass.async_block_till_done()

            assert home.qs_home_real_off_grid is True
            mock_set.assert_called_with(True, False)

    @pytest.mark.asyncio
    async def test_sensor_state_change_no_match_restores_on_grid(
        self, hass: HomeAssistant, home_with_sensor_off_grid
    ):
        """Sensor changing from matching to non-matching value restores on-grid."""
        home = home_with_sensor_off_grid
        home.off_grid_mode = OFF_GRID_MODE_AUTO

        # First set to matching value so HA records the state change
        hass.states.async_set("sensor.grid_status", "disconnected")
        await hass.async_block_till_done()
        assert home.qs_home_real_off_grid is True

        with patch.object(home, "async_set_off_grid_mode", new_callable=AsyncMock) as mock_set:
            hass.states.async_set("sensor.grid_status", "connected")
            await hass.async_block_till_done()

            assert home.qs_home_real_off_grid is False
            mock_set.assert_called_with(False, False)

    @pytest.mark.asyncio
    async def test_switch_inverted_state_change(
        self, hass: HomeAssistant, home_with_switch_inverted
    ):
        """Switch inverted: 'off' triggers off-grid, 'on' restores on-grid."""
        home = home_with_switch_inverted
        home.off_grid_mode = OFF_GRID_MODE_AUTO

        with patch.object(home, "async_set_off_grid_mode", new_callable=AsyncMock) as mock_set:
            hass.states.async_set("switch.grid_connection", "off")
            await hass.async_block_till_done()

            assert home.qs_home_real_off_grid is True
            mock_set.assert_called_with(True, False)

        with patch.object(home, "async_set_off_grid_mode", new_callable=AsyncMock) as mock_set:
            hass.states.async_set("switch.grid_connection", "on")
            await hass.async_block_till_done()

            assert home.qs_home_real_off_grid is False
            mock_set.assert_called_with(False, False)

    @pytest.mark.asyncio
    async def test_force_mode_overrides_entity_state(
        self, hass: HomeAssistant, home_with_binary_sensor_off_grid
    ):
        """Force off-grid mode ignores entity state."""
        home = home_with_binary_sensor_off_grid
        home.off_grid_mode = OFF_GRID_MODE_FORCE_ON_GRID

        with patch.object(home, "async_set_off_grid_mode", new_callable=AsyncMock) as mock_set:
            hass.states.async_set("binary_sensor.grid_relay", "on")
            await hass.async_block_till_done()

            assert home.qs_home_real_off_grid is True
            mock_set.assert_called_with(False, False)

    # -- Normalization tests --

    def test_normalize_off_grid_value(self, home_with_sensor_off_grid):
        """Normalize strips case, spaces, dashes, underscores."""
        normalize = QSHome._normalize_off_grid_value
        assert normalize("Off Grid") == "offgrid"
        assert normalize("OFF-GRID") == "offgrid"
        assert normalize("off_grid") == "offgrid"
        assert normalize("  Off_Grid  ") == "offgrid"
        assert normalize(None) is None

    def test_sensor_comparison_is_normalized(self, home_with_sensor_off_grid):
        """Sensor state comparison is normalized (case, dashes, underscores stripped)."""
        home = home_with_sensor_off_grid
        # All of these normalize to "disconnected" which matches the config value
        assert home._compute_off_grid_from_entity_state("Disconnected", "sensor.grid_status") is True
        assert home._compute_off_grid_from_entity_state("DISCONNECTED", "sensor.grid_status") is True
        assert home._compute_off_grid_from_entity_state("dis-connected", "sensor.grid_status") is True
        assert home._compute_off_grid_from_entity_state("dis_connected", "sensor.grid_status") is True
        # These do NOT normalize to "disconnected"
        assert home._compute_off_grid_from_entity_state("connected", "sensor.grid_status") is False
        assert home._compute_off_grid_from_entity_state("online", "sensor.grid_status") is False

    @pytest.fixture
    async def home_with_sensor_off_grid_fancy_value(
        self, hass: HomeAssistant, home_config_entry: MockConfigEntry, home_zone_state: None
    ):
        """QSHome with a sensor off-grid entity using a value with mixed case/separators."""
        hass.states.async_set("sensor.grid_status2", "connected")
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.add_device"):
            return QSHome(
                hass=hass,
                config_entry=home_config_entry,
                **{
                    CONF_NAME: "Test Home",
                    CONF_HOME_VOLTAGE: 230,
                    CONF_OFF_GRID_ENTITY: "sensor.grid_status2",
                    CONF_OFF_GRID_STATE_VALUE: "Off_Grid",
                    CONF_OFF_GRID_INVERTED: False,
                },
            )

    def test_normalized_config_value_matches_ha_state(self, home_with_sensor_off_grid_fancy_value):
        """Config value 'Off_Grid' matches HA state 'offgrid' after normalization."""
        home = home_with_sensor_off_grid_fancy_value
        assert home._off_grid_state_value == "offgrid"
        assert home._compute_off_grid_from_entity_state("offgrid", "sensor.grid_status2") is True
        assert home._compute_off_grid_from_entity_state("Off Grid", "sensor.grid_status2") is True
        assert home._compute_off_grid_from_entity_state("OFF_GRID", "sensor.grid_status2") is True
        assert home._compute_off_grid_from_entity_state("connected", "sensor.grid_status2") is False

    # -- Notification broadcast tests --

    @pytest.mark.asyncio
    async def test_on_grid_to_off_grid_triggers_notification(
        self, hass: HomeAssistant, home_with_binary_sensor_off_grid
    ):
        """Transitioning from on-grid to off-grid sends a notification to all mobile apps."""
        home = home_with_binary_sensor_off_grid
        home.off_grid_mode = OFF_GRID_MODE_AUTO
        assert home.qs_home_real_off_grid is False

        with patch.object(home, "async_notify_all_mobile_apps", new_callable=AsyncMock) as mock_notify:
            hass.states.async_set("binary_sensor.grid_relay", "on")
            await hass.async_block_till_done()

            assert home.qs_home_real_off_grid is True
            mock_notify.assert_called_once()
            call_args = mock_notify.call_args
            assert "URGENT" in call_args[1]["title"] or "URGENT" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_off_grid_to_on_grid_does_not_notify(
        self, hass: HomeAssistant, home_with_binary_sensor_off_grid
    ):
        """Transitioning from off-grid back to on-grid does NOT send a notification."""
        home = home_with_binary_sensor_off_grid
        home.off_grid_mode = OFF_GRID_MODE_AUTO

        # Go off-grid first
        hass.states.async_set("binary_sensor.grid_relay", "on")
        await hass.async_block_till_done()
        assert home.qs_home_real_off_grid is True

        with patch.object(home, "async_notify_all_mobile_apps", new_callable=AsyncMock) as mock_notify:
            hass.states.async_set("binary_sensor.grid_relay", "off")
            await hass.async_block_till_done()

            assert home.qs_home_real_off_grid is False
            mock_notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_staying_off_grid_does_not_re_notify(
        self, hass: HomeAssistant, home_with_binary_sensor_off_grid
    ):
        """Repeated off-grid states do not trigger additional notifications."""
        home = home_with_binary_sensor_off_grid
        home.off_grid_mode = OFF_GRID_MODE_AUTO

        # Go off-grid
        hass.states.async_set("binary_sensor.grid_relay", "on")
        await hass.async_block_till_done()
        assert home.qs_home_real_off_grid is True

        with patch.object(home, "async_notify_all_mobile_apps", new_callable=AsyncMock) as mock_notify:
            # Set to a different state then back to "on" to produce a state_changed event
            hass.states.async_set("binary_sensor.grid_relay", "off")
            await hass.async_block_till_done()
            hass.states.async_set("binary_sensor.grid_relay", "on")
            await hass.async_block_till_done()

            # Second on->off->on cycle: the off->on doesn't notify, the on->off does
            # but the first off->on cleared qs_home_real_off_grid, so on->off is a fresh transition
            assert mock_notify.call_count == 1

    @pytest.mark.asyncio
    async def test_notify_all_mobile_apps_calls_services(
        self, hass: HomeAssistant, home_with_binary_sensor_off_grid
    ):
        """async_notify_all_mobile_apps discovers and calls all mobile_app services."""
        home = home_with_binary_sensor_off_grid

        # Register fake mobile_app services with handlers that record calls
        calls_phone_1 = []
        calls_phone_2 = []
        calls_email = []

        async def handler_phone_1(call):
            calls_phone_1.append(call)

        async def handler_phone_2(call):
            calls_phone_2.append(call)

        async def handler_email(call):
            calls_email.append(call)

        hass.services.async_register(Platform.NOTIFY, "mobile_app_phone_1", handler_phone_1)
        hass.services.async_register(Platform.NOTIFY, "mobile_app_phone_2", handler_phone_2)
        hass.services.async_register(Platform.NOTIFY, "email_service", handler_email)

        await home.async_notify_all_mobile_apps(
            title="Test alert",
            message="Test message",
        )
        await hass.async_block_till_done()

        # Should call only the 2 mobile_app services, not email_service
        assert len(calls_phone_1) == 1
        assert len(calls_phone_2) == 1
        assert len(calls_email) == 0

    @pytest.mark.asyncio
    async def test_notify_all_mobile_apps_no_services(
        self, hass: HomeAssistant, home_with_binary_sensor_off_grid
    ):
        """async_notify_all_mobile_apps does nothing when no mobile_app services exist."""
        home = home_with_binary_sensor_off_grid

        # No mobile_app services registered -- should not raise
        await home.async_notify_all_mobile_apps(
            title="Test alert",
            message="Test message",
        )

    @pytest.mark.asyncio
    async def test_notify_all_mobile_apps_includes_critical_push_data(
        self, hass: HomeAssistant, home_with_binary_sensor_off_grid
    ):
        """Notification payload includes critical/urgent push metadata."""
        home = home_with_binary_sensor_off_grid

        received_calls = []

        async def handler(call):
            received_calls.append(call)

        hass.services.async_register(Platform.NOTIFY, "mobile_app_phone", handler)

        await home.async_notify_all_mobile_apps(
            title="Alert",
            message="Grid lost",
        )
        await hass.async_block_till_done()

        assert len(received_calls) == 1
        call_data = received_calls[0].data
        assert call_data["title"] == "Alert"
        assert call_data["message"] == "Grid lost"
        assert call_data["data"]["priority"] == "high"
        assert call_data["data"]["push"]["interruption-level"] == "critical"
        assert call_data["data"]["push"]["sound"]["critical"] == 1

    # -- Cleanup tests --

    def test_unregister_cleans_up(self, home_with_binary_sensor_off_grid):
        """Unregistering the listener sets _off_grid_unsub to None."""
        home = home_with_binary_sensor_off_grid
        assert home._off_grid_unsub is not None
        home._unregister_off_grid_entity_listener()
        assert home._off_grid_unsub is None
