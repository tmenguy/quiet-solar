"""Tests for QSPerson class."""
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytz

from custom_components.quiet_solar.ha_model.person import QSPerson
from custom_components.quiet_solar.ha_model.home import QSHome
from custom_components.quiet_solar.ha_model.car import QSCar
from tests.conftest import FakeHass, FakeState


class TestQSPerson(unittest.TestCase):
    """Test QSPerson class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.fake_hass = FakeHass()
        self.home = QSHome(hass=self.fake_hass, config_entry=None, name="test home")

    def test_persons_creation(self):
        """Test creating a QSPerson instance."""
        person = QSPerson(
            hass=self.fake_hass,
            home=self.home,
            config_entry=None,
            name="John Doe",
            person_person_entity="person.john_doe"
        )

        assert person.name == "John Doe"
        assert person.person_entity_id == "person.john_doe"


    def test_is_person_home(self):
        """Test checking if person is home."""
        person = QSPerson(
            hass=self.fake_hass,
            home=self.home,
            config_entry=None,
            name="John Doe",
            person_person_entity="person.john_doe"
        )

        # Mock the get_sensor_latest_possible_valid_value method
        with patch.object(person, 'get_sensor_latest_possible_valid_value', return_value="home"):
            time = datetime.now(pytz.UTC)
            is_home = person.is_person_home(time)
            assert is_home is True

        with patch.object(person, 'get_sensor_latest_possible_valid_value', return_value="away"):
            time = datetime.now(pytz.UTC)
            is_home = person.is_person_home(time)
            assert is_home is False


class TestQSPersonCarCorrelation(unittest.TestCase):
    """Test QSPerson car correlation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.fake_hass = FakeHass()
        self.home = QSHome(hass=self.fake_hass, config_entry=None, name="test home")

        # Create a car
        self.car = QSCar(
            hass=self.fake_hass,
            home=self.home,
            config_entry=None,
            name="Test Car",
            car_tracker="device_tracker.test_car"
        )
        self.home.add_device(self.car)




class TestQSPersonHomeIntegration(unittest.TestCase):
    """Test QSPerson integration with QSHome."""

    def setUp(self):
        """Set up test fixtures."""
        self.fake_hass = FakeHass()
        self.home = QSHome(hass=self.fake_hass, config_entry=None, name="test home")

    def test_home_adds_person_to_persons_list(self):
        """Test that home adds person to its people list."""
        person = QSPerson(
            hass=self.fake_hass,
            home=self.home,
            config_entry=None,
            name="John Doe",
            person_entity_id="person.john_doe"
        )

        self.home.add_device(person)

        assert person in self.home._persons
        assert len(self.home._persons) == 1

    def test_home_removes_person_from_persons_list(self):
        """Test that home removes person from its people list."""
        person = QSPerson(
            hass=self.fake_hass,
            home=self.home,
            config_entry=None,
            name="John Doe",
            person_entity_id="person.john_doe"
        )

        self.home.add_device(person)
        assert person in self.home._persons

        self.home.remove_device(person)
        assert person not in self.home._persons

    # REMOVED: These tests are for methods that were removed
    # QSPerson are now created through config flow, not auto-discovery
    # def test_home_get_attached_virtual_devices(self):
    # def test_discover_and_create_persons(self):
    # def test_discover_and_create_persons_no_duplicates(self):


class TestQSPersonPlatforms(unittest.TestCase):
    """Test QSPerson platform support."""

    def setUp(self):
        """Set up test fixtures."""
        self.fake_hass = FakeHass()
        self.home = QSHome(hass=self.fake_hass, config_entry=None, name="test home")

    def test_get_platforms(self):
        """Test that QSPerson returns correct platforms."""
        person = QSPerson(
            hass=self.fake_hass,
            home=self.home,
            config_entry=None,
            name="John Doe",
            person_entity_id="person.john_doe"
        )

        platforms = person.get_platforms()

        # Should support sensor platform
        from homeassistant.const import Platform
        assert Platform.SENSOR in platforms

    def test_dashboard_sort_string(self):
        """Test dashboard sort string."""
        person = QSPerson(
            hass=self.fake_hass,
            home=self.home,
            config_entry=None,
            name="John Doe",
            person_entity_id="person.john_doe"
        )

        sort_string = person.dashboard_sort_string_in_type
        assert sort_string == "AAA"


if __name__ == '__main__':
    unittest.main()

