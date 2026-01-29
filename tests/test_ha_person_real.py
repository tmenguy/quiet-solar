"""Extended tests for QSPerson class in ha_model/person.py."""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import time as dt_time, timedelta

from homeassistant.const import (
    Platform,
    STATE_UNKNOWN,
    STATE_UNAVAILABLE,
    CONF_NAME,
)
from homeassistant.core import HomeAssistant
import pytz

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.ha_model.person import QSPerson
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_PERSON_PERSON_ENTITY,
    CONF_PERSON_TRACKER,
    CONF_PERSON_AUTHORIZED_CARS,
    CONF_PERSON_PREFERRED_CAR,
    CONF_PERSON_NOTIFICATION_TIME,
    CONF_MOBILE_APP,
    CONF_MOBILE_APP_URL,
    PERSON_NOTIFY_REASON_DAILY_REMINDER_FOR_CAR_NO_CHARGER,
    PERSON_NOTIFY_REASON_DAILY_CHARGER_CONSTRAINTS,
    PERSON_NOTIFY_REASON_CHANGED_CAR,
)


@pytest.fixture
def person_config_entry() -> MockConfigEntry:
    """Config entry for person tests."""
    return MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_person_entry",
        data={CONF_NAME: "Test Person"},
        title="Test Person",
    )


@pytest.fixture
def person_home():
    """Mock home for person tests."""
    return MagicMock()


@pytest.fixture
def person_data_handler(person_home):
    """Data handler for person tests."""
    handler = MagicMock()
    handler.home = person_home
    return handler


@pytest.fixture
def person_hass_data(hass: HomeAssistant, person_data_handler):
    """Set hass.data[DOMAIN][DATA_HANDLER] for person tests."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][DATA_HANDLER] = person_data_handler


@pytest.fixture
def person(hass, person_config_entry, person_home, person_hass_data):
    """Default QSPerson instance for tests that need one (John Doe, authorized_cars ['car1'])."""
    return QSPerson(
        hass=hass,
        config_entry=person_config_entry,
        home=person_home,
        **{
            CONF_NAME: "John Doe",
            CONF_PERSON_PERSON_ENTITY: "person.john_doe",
            CONF_PERSON_AUTHORIZED_CARS: ["car1"],
        },
    )


class TestQSPersonInit:
    """Test QSPerson initialization."""

    def test_init_with_minimal_params(self, hass, person_config_entry, person_home, person_hass_data):
        """Test initialization with minimal parameters."""
        person = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
            },
        )

        assert person.name == "John Doe"
        assert person.person_entity_id == "person.john_doe"
        assert person.authorized_cars == []
        assert person.preferred_car is None
        assert person.notification_time is None
        assert person.notification_dt_time is None

    def test_init_with_all_params(self, hass, person_config_entry, person_home, person_hass_data):
        """Test initialization with all parameters."""
        person = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
                CONF_PERSON_TRACKER: "device_tracker.john_phone",
                CONF_PERSON_AUTHORIZED_CARS: ["car1", "car2"],
                CONF_PERSON_PREFERRED_CAR: "car1",
                CONF_PERSON_NOTIFICATION_TIME: "08:00:00",
                CONF_MOBILE_APP: "mobile_app_john",
                CONF_MOBILE_APP_URL: "/dashboard",
            },
        )

        assert person.person_tracker_id == "device_tracker.john_phone"
        assert person.authorized_cars == ["car1", "car2"]
        assert person.preferred_car == "car1"
        assert person.notification_time == "08:00:00"
        assert person.notification_dt_time == dt_time(hour=8, minute=0, second=0)
        assert person.mobile_app == "mobile_app_john"
        assert person.mobile_app_url == "/dashboard"

    def test_init_adds_preferred_car_to_authorized(self, hass, person_config_entry, person_home, person_hass_data):
        """Test that preferred car is added to authorized cars if missing."""
        person = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
                CONF_PERSON_AUTHORIZED_CARS: ["car1"],
                CONF_PERSON_PREFERRED_CAR: "car2",  # Not in authorized_cars
            },
        )

        assert "car2" in person.authorized_cars

    def test_init_mobile_app_url_normalization(self, hass, person_config_entry, person_home, person_hass_data):
        """Test mobile app URL normalization."""
        # Test with URL without leading slash
        person1 = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
                CONF_MOBILE_APP_URL: "dashboard",
            },
        )
        assert person1.mobile_app_url == "/dashboard"

        # Test with empty URL
        person2 = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
                CONF_MOBILE_APP_URL: "",
            },
        )
        assert person2.mobile_app_url is None

        # Test with just slash
        person3 = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
                CONF_MOBILE_APP_URL: "/",
            },
        )
        assert person3.mobile_app_url is None


class TestQSPersonTracker:
    """Test tracker-related methods."""

    def test_get_tracker_id_with_tracker(self, hass, person_config_entry, person_home, person_hass_data):
        """Test get_tracker_id returns tracker when set."""
        person = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
                CONF_PERSON_TRACKER: "device_tracker.john_phone",
            },
        )

        result = person.get_tracker_id()
        assert result == "device_tracker.john_phone"

    def test_get_tracker_id_without_tracker(self, hass, person_config_entry, person_home, person_hass_data):
        """Test get_tracker_id returns person entity when no tracker set."""
        person = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
            },
        )

        result = person.get_tracker_id()
        assert result == "person.john_doe"


class TestQSPersonMileageHistory:
    """Test mileage history methods."""

    def test_add_to_mileage_history_first_entry(self, person):
        """Test adding first mileage history entry."""
        day = datetime.datetime(2024, 6, 15, 0, 0, 0, tzinfo=pytz.UTC)
        leave_time = datetime.datetime(2024, 6, 15, 8, 0, 0, tzinfo=pytz.UTC)

        person.add_to_mileage_history(day, 50.0, leave_time)

        assert len(person.historical_mileage_data) == 1
        assert person.historical_mileage_data[0][1] == 50.0

    def test_add_to_mileage_history_multiple_entries(self, person):
        """Test adding multiple mileage history entries."""
        day1 = datetime.datetime(2024, 6, 15, 0, 0, 0, tzinfo=pytz.UTC)
        day2 = datetime.datetime(2024, 6, 16, 0, 0, 0, tzinfo=pytz.UTC)
        leave_time = datetime.datetime(2024, 6, 15, 8, 0, 0, tzinfo=pytz.UTC)

        person.add_to_mileage_history(day1, 50.0, leave_time)
        person.add_to_mileage_history(day2, 75.0, leave_time + timedelta(days=1))

        assert len(person.historical_mileage_data) == 2

    def test_add_to_mileage_history_updates_existing(self, person):
        """Test that adding to same day updates existing entry."""
        day = datetime.datetime(2024, 6, 15, 0, 0, 0, tzinfo=pytz.UTC)
        leave_time = datetime.datetime(2024, 6, 15, 8, 0, 0, tzinfo=pytz.UTC)

        person.add_to_mileage_history(day, 50.0, leave_time)
        person.add_to_mileage_history(day, 100.0, leave_time)  # Same day, different mileage

        assert len(person.historical_mileage_data) == 1
        assert person.historical_mileage_data[0][1] == 100.0

    def test_add_to_mileage_history_none_leave_time(self, person):
        """Test that None leave_time logs warning and returns early."""
        day = datetime.datetime(2024, 6, 15, 0, 0, 0, tzinfo=pytz.UTC)

        person.add_to_mileage_history(day, 50.0, None)

        # Should not add entry without leave time
        assert len(person.historical_mileage_data) == 0

    def test_add_to_mileage_history_resets_prediction_time(self, person):
        """Test that adding history resets prediction time."""
        person._last_request_prediction_time = datetime.datetime.now(pytz.UTC)

        day = datetime.datetime(2024, 6, 15, 0, 0, 0, tzinfo=pytz.UTC)
        leave_time = datetime.datetime(2024, 6, 15, 8, 0, 0, tzinfo=pytz.UTC)

        person.add_to_mileage_history(day, 50.0, leave_time)

        assert person._last_request_prediction_time is None


class TestQSPersonPrediction:
    """Test prediction methods."""

    def test_get_best_week_day_guess_no_data(self, person):
        """Test _get_best_week_day_guess with no data."""
        mileage, leave_time = person._get_best_week_day_guess(0)  # Monday

        assert mileage is None
        assert leave_time is None

    def test_get_best_week_day_guess_with_data(self, person):
        """Test _get_best_week_day_guess with historical data."""
        # Add data for Monday (weekday 0)
        monday1 = datetime.datetime(2024, 6, 10, 0, 0, 0, tzinfo=pytz.UTC)  # A Monday
        leave_time1 = datetime.datetime(2024, 6, 10, 8, 0, 0, tzinfo=pytz.UTC)
        monday2 = datetime.datetime(2024, 6, 17, 0, 0, 0, tzinfo=pytz.UTC)  # Next Monday
        leave_time2 = datetime.datetime(2024, 6, 17, 7, 30, 0, tzinfo=pytz.UTC)

        person.add_to_mileage_history(monday1, 50.0, leave_time1)
        person.add_to_mileage_history(monday2, 75.0, leave_time2)

        mileage, leave_time = person._get_best_week_day_guess(0)  # Monday

        # Should return max mileage and earliest leave time
        assert mileage == 75.0  # Max of 50 and 75
        # Leave time should be 7:30 (earliest)

    def test_compute_person_next_need_no_history(self, person):
        """Test _compute_person_next_need with no history."""
        time = datetime.datetime.now(pytz.UTC)

        leave_time, mileage = person._compute_person_next_need(time)

        assert leave_time is None
        assert mileage is None

    def test_should_recompute_history_no_cars(self, hass, person_config_entry, person_home, person_hass_data):
        """Test should_recompute_history returns False with no authorized cars."""
        person = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
                CONF_PERSON_AUTHORIZED_CARS: [],
            },
        )

        result = person.should_recompute_history(datetime.datetime.now(pytz.UTC))

        assert result is False

    def test_should_recompute_history_not_initialized(self, person):
        """Test should_recompute_history returns True when not initialized."""
        person.has_been_initialized = False

        result = person.should_recompute_history(datetime.datetime.now(pytz.UTC))

        assert result is True

    def test_update_person_forecast_caches_result(self, person):
        """Test update_person_forecast caches and reuses result."""
        time = datetime.datetime.now(pytz.UTC)

        # First call
        person.update_person_forecast(time)
        first_time = person._last_request_prediction_time

        # Second call immediately after (should use cache)
        person.update_person_forecast(time)

        assert person._last_request_prediction_time == first_time


class TestQSPersonAuthorizedCars:
    """Test authorized cars methods."""

    def test_get_authorized_cars_empty(self, hass, person_config_entry, person_home, person_hass_data):
        """Test get_authorized_cars with no cars."""
        person = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
            },
        )
        person_home.get_car_by_name = MagicMock(return_value=None)

        result = person.get_authorized_cars()

        assert result == []

    def test_get_authorized_cars_with_valid_cars(self, hass, person_config_entry, person_home, person_hass_data):
        """Test get_authorized_cars returns valid QSCar objects."""
        person = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
                CONF_PERSON_AUTHORIZED_CARS: ["car1", "car2"],
            },
        )

        # Mock car objects
        from custom_components.quiet_solar.ha_model.car import QSCar
        mock_car1 = MagicMock(spec=QSCar)
        mock_car2 = MagicMock(spec=QSCar)

        def get_car(name):
            if name == "car1":
                return mock_car1
            elif name == "car2":
                return mock_car2
            return None

        person_home.get_car_by_name = MagicMock(side_effect=get_car)

        result = person.get_authorized_cars()

        assert len(result) == 2
        assert mock_car1 in result
        assert mock_car2 in result

    def test_get_preferred_car_none(self, hass, person_config_entry, person_home, person_hass_data):
        """Test get_preferred_car when no preferred car set."""
        person = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
            },
        )

        result = person.get_preferred_car()

        assert result is None

    def test_get_preferred_car_valid(self, hass, person_config_entry, person_home, person_hass_data):
        """Test get_preferred_car returns valid QSCar."""
        person = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
                CONF_PERSON_PREFERRED_CAR: "my_car",
            },
        )

        from custom_components.quiet_solar.ha_model.car import QSCar
        mock_car = MagicMock(spec=QSCar)
        person_home.get_car_by_name = MagicMock(return_value=mock_car)

        result = person.get_preferred_car()

        assert result == mock_car


class TestQSPersonForecastString:
    """Test forecast string methods."""

    def test_get_forecast_readable_string_no_forecast(self, person):
        """Test get_forecast_readable_string with no forecast."""
        person.predicted_mileage = None
        person.predicted_leave_time = None

        result = person.get_forecast_readable_string()

        assert result == "No forecast"

    def test_get_forecast_readable_string_with_forecast(self, person):
        """Test get_forecast_readable_string with forecast."""
        person.predicted_mileage = 50.0
        person.predicted_leave_time = datetime.datetime(2024, 6, 15, 8, 0, 0, tzinfo=pytz.UTC)
        # Mock update_person_forecast to prevent it from resetting our values
        person.update_person_forecast = MagicMock(return_value=(person.predicted_leave_time, person.predicted_mileage))

        result = person.get_forecast_readable_string()

        assert "50km" in result


class TestQSPersonPlatforms:
    """Test platform support."""

    def test_get_platforms(self, hass, person_config_entry, person_home, person_hass_data):
        """Test get_platforms returns SENSOR."""
        person = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
            },
        )

        platforms = person.get_platforms()

        assert Platform.SENSOR in platforms

    def test_dashboard_sort_string(self, hass, person_config_entry, person_home, person_hass_data):
        """Test dashboard_sort_string_in_type returns 'AAA'."""
        person = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
            },
        )

        result = person.dashboard_sort_string_in_type

        assert result == "AAA"


class TestQSPersonSerialization:
    """Test serialization methods."""

    def test_get_person_mileage_serialized_prediction_no_forecast(self, person):
        """Test serialization with no forecast."""
        person.predicted_mileage = None
        person.predicted_leave_time = None
        person.has_been_initialized = False

        state_value, attributes = person.get_person_mileage_serialized_prediction()

        assert state_value == "No forecast"
        assert attributes["predicted_mileage"] is None
        assert attributes["predicted_leave_time"] is None
        assert attributes["has_been_initialized"] is False

    def test_get_person_mileage_serialized_prediction_with_forecast(self, person):
        """Test serialization with forecast."""
        person.predicted_mileage = 50.0
        person.predicted_leave_time = datetime.datetime(2024, 6, 15, 8, 0, 0, tzinfo=pytz.UTC)
        person.has_been_initialized = True
        # Mock update_person_forecast to prevent it from resetting our values
        person.update_person_forecast = MagicMock(return_value=(person.predicted_leave_time, person.predicted_mileage))

        state_value, attributes = person.get_person_mileage_serialized_prediction()

        assert "50km" in state_value
        assert attributes["predicted_mileage"] == 50.0
        assert attributes["predicted_leave_time"] is not None
        assert attributes["has_been_initialized"] is True


class TestQSPersonReset:
    """Test reset method."""

    def test_reset(self, hass, person_config_entry, person_home, person_hass_data):
        """Test reset method calls parent."""
        person = QSPerson(
            hass=hass,
            config_entry=person_config_entry,
            home=person_home,
            **{
                CONF_NAME: "John Doe",
                CONF_PERSON_PERSON_ENTITY: "person.john_doe",
            },
        )

        person._constraints = [MagicMock()]
        person.current_command = MagicMock()

        person.reset()
        assert person._constraints == []
        assert person.current_command is None
