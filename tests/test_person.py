"""Tests for QSPerson class."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import pytz

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from custom_components.quiet_solar.ha_model.person import QSPerson
from custom_components.quiet_solar.ha_model.home import QSHome
from custom_components.quiet_solar.ha_model.car import QSCar
from custom_components.quiet_solar.const import DOMAIN


@pytest.fixture
def person_home(hass: HomeAssistant):
    """QSHome instance for person tests."""
    hass.data.setdefault(DOMAIN, {})
    return QSHome(hass=hass, config_entry=None, name="test home")


def test_persons_creation(hass: HomeAssistant, person_home):
    """Test creating a QSPerson instance."""
    person = QSPerson(
        hass=hass,
        home=person_home,
        config_entry=None,
        name="John Doe",
        person_person_entity="person.john_doe",
    )

    assert person.name == "John Doe"
    assert person.person_entity_id == "person.john_doe"


@pytest.fixture
def person_home_with_car(hass: HomeAssistant, person_home):
    """QSHome with a car for car correlation tests."""
    car = QSCar(
        hass=hass,
        home=person_home,
        config_entry=None,
        name="Test Car",
        car_tracker="device_tracker.test_car",
    )
    person_home.add_device(car)
    return person_home


def test_home_adds_person_to_persons_list(hass: HomeAssistant, person_home):
    """Test that home adds person to its people list."""
    person = QSPerson(
        hass=hass,
        home=person_home,
        config_entry=None,
        name="John Doe",
        person_entity_id="person.john_doe",
    )

    person_home.add_device(person)

    assert person in person_home._persons
    assert len(person_home._persons) == 1


def test_home_removes_person_from_persons_list(hass: HomeAssistant, person_home):
    """Test that home removes person from its people list."""
    person = QSPerson(
        hass=hass,
        home=person_home,
        config_entry=None,
        name="John Doe",
        person_entity_id="person.john_doe",
    )

    person_home.add_device(person)
    assert person in person_home._persons

    person_home.remove_device(person)
    assert person not in person_home._persons


def test_get_platforms(hass: HomeAssistant, person_home):
    """Test that QSPerson returns correct platforms."""
    person = QSPerson(
        hass=hass,
        home=person_home,
        config_entry=None,
        name="John Doe",
        person_entity_id="person.john_doe",
    )

    platforms = person.get_platforms()

    assert Platform.SENSOR in platforms


def test_dashboard_sort_string(hass: HomeAssistant, person_home):
    """Test dashboard sort string."""
    person = QSPerson(
        hass=hass,
        home=person_home,
        config_entry=None,
        name="John Doe",
        person_entity_id="person.john_doe",
    )

    sort_string = person.dashboard_sort_string_in_type
    assert sort_string == "AAA"
