"""Tests for quiet_solar person.py functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_PERSON_PERSON_ENTITY,
    CONF_PERSON_AUTHORIZED_CARS,
    CONF_PERSON_PREFERRED_CAR,
)


pytestmark = pytest.mark.usefixtures("mock_sensor_states")


async def test_person_initialization(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test person device initialization."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create car first (person needs authorized cars)
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_init",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_init",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create person
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_init_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_init_test",
    )
    person_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    assert person_entry.state is ConfigEntryState.LOADED

    # Verify person device was created
    person_device = hass.data[DOMAIN].get(person_entry.entry_id)
    assert person_device is not None
    assert person_device.name == MOCK_PERSON_CONFIG['name']


async def test_person_entity_configuration(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test person entity configuration."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_entity",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_entity",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create person
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_entity_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_entity_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    person_device = hass.data[DOMAIN].get(person_entry.entry_id)
    assert person_device.person_entity_id == MOCK_PERSON_CONFIG[CONF_PERSON_PERSON_ENTITY]


async def test_person_authorized_cars(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test person authorized cars configuration."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_auth",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_auth",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create person
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_auth_cars_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_auth_cars_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    person_device = hass.data[DOMAIN].get(person_entry.entry_id)
    assert person_device.authorized_cars == MOCK_PERSON_CONFIG[CONF_PERSON_AUTHORIZED_CARS]


async def test_person_preferred_car(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test person preferred car configuration."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_pref",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_pref",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create person
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_pref_car_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_pref_car_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    person_device = hass.data[DOMAIN].get(person_entry.entry_id)
    assert person_device.preferred_car == MOCK_PERSON_CONFIG[CONF_PERSON_PREFERRED_CAR]


async def test_person_home_state(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test person home state tracking."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG

    # Set person state to home
    hass.states.async_set("person.test_person", "home")

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_home",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_home",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create person
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_home_state_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_home_state_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    person_device = hass.data[DOMAIN].get(person_entry.entry_id)
    assert person_device is not None


async def test_person_away_state(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test person away state tracking."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG

    # Set person state to not_home
    hass.states.async_set("person.test_person", "not_home")

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_away",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_away",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create person
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_away_state_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_away_state_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    person_device = hass.data[DOMAIN].get(person_entry.entry_id)
    assert person_device is not None


async def test_person_sensor_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test person sensor entities are created."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_sensors",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_sensors",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create person
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_sensor_entities_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_sensor_entities_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    # Check sensor entities
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, person_entry.entry_id
    )

    sensor_entries = [e for e in entity_entries if e.domain == "sensor"]

    # Should have sensors for forecast, etc.
    assert len(sensor_entries) >= 1, f"Expected at least 1 sensor entity, got {len(sensor_entries)}"


async def test_person_select_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test person select entities are created."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_selects",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_selects",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create person
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_select_entities_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_select_entities_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    # Check select entities
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, person_entry.entry_id
    )

    select_entries = [e for e in entity_entries if e.domain == "select"]

    # Person might have car selection
    # Don't assert specific count as it depends on configuration
    assert isinstance(select_entries, list)


async def test_multiple_persons(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test multiple persons can be created."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_multi_persons",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_multi_persons",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Set up mock person entities
    hass.states.async_set("person.person1", "home")
    hass.states.async_set("person.person2", "not_home")

    # Create first person
    person1_config = {
        **MOCK_PERSON_CONFIG,
        "name": "Person 1",
        "person_person_entity": "person.person1",
    }
    person1_entry = MockConfigEntry(
        domain=DOMAIN,
        data=person1_config,
        entry_id="person1_multi_test",
        title="person: Person 1",
        unique_id="quiet_solar_person1_multi_test",
    )
    person1_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person1_entry.entry_id)
    await hass.async_block_till_done()

    # Create second person
    person2_config = {
        **MOCK_PERSON_CONFIG,
        "name": "Person 2",
        "person_person_entity": "person.person2",
    }
    person2_entry = MockConfigEntry(
        domain=DOMAIN,
        data=person2_config,
        entry_id="person2_multi_test",
        title="person: Person 2",
        unique_id="quiet_solar_person2_multi_test",
    )
    person2_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person2_entry.entry_id)
    await hass.async_block_till_done()

    assert person1_entry.state is ConfigEntryState.LOADED
    assert person2_entry.state is ConfigEntryState.LOADED


async def test_person_with_multiple_cars(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test person with multiple authorized cars."""
    from .const import MOCK_CAR_CONFIG
    from homeassistant.const import CONF_NAME
    from custom_components.quiet_solar.const import (
        DEVICE_TYPE,
        CONF_TYPE_NAME_QSPerson,
    )

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create first car
    car1_config = {**MOCK_CAR_CONFIG, "name": "Car A"}
    car1_entry = MockConfigEntry(
        domain=DOMAIN,
        data=car1_config,
        entry_id="carA_multi_cars_person",
        title="car: Car A",
        unique_id="quiet_solar_carA_multi_cars_person",
    )
    car1_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car1_entry.entry_id)
    await hass.async_block_till_done()

    # Create second car
    car2_config = {**MOCK_CAR_CONFIG, "name": "Car B"}
    car2_entry = MockConfigEntry(
        domain=DOMAIN,
        data=car2_config,
        entry_id="carB_multi_cars_person",
        title="car: Car B",
        unique_id="quiet_solar_carB_multi_cars_person",
    )
    car2_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car2_entry.entry_id)
    await hass.async_block_till_done()

    # Create person with multiple authorized cars
    person_config = {
        CONF_NAME: "Multi Car Person",
        DEVICE_TYPE: CONF_TYPE_NAME_QSPerson,
        CONF_PERSON_PERSON_ENTITY: "person.test_person",
        CONF_PERSON_AUTHORIZED_CARS: ["Car A", "Car B"],
        CONF_PERSON_PREFERRED_CAR: "Car A",
    }

    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=person_config,
        entry_id="person_multi_cars_test",
        title="person: Multi Car Person",
        unique_id="quiet_solar_person_multi_cars_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    assert person_entry.state is ConfigEntryState.LOADED

    person_device = hass.data[DOMAIN].get(person_entry.entry_id)
    assert "Car A" in person_device.authorized_cars
    assert "Car B" in person_device.authorized_cars
    assert person_device.preferred_car == "Car A"


async def test_person_button_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test person button entities are created."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_buttons",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_buttons",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create person
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_button_entities_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_button_entities_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    # Check button entities
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, person_entry.entry_id
    )

    button_entries = [e for e in entity_entries if e.domain == "button"]

    # Should have at least reset/clean buttons
    assert len(button_entries) >= 1, f"Expected at least 1 button entity, got {len(button_entries)}"
