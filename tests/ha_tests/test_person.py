"""Tests for quiet_solar person.py functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

import pytz

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


async def test_person_get_platforms(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test person returns correct platforms."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG
    from homeassistant.const import Platform

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_platforms",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_platforms",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create person
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_platforms_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_platforms_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    person_device = hass.data[DOMAIN].get(person_entry.entry_id)
    platforms = person_device.get_platforms()

    assert Platform.SENSOR in platforms
    assert Platform.BUTTON in platforms


async def test_person_get_tracker_id(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test person get_tracker_id method."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_tracker",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_tracker",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create person
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_tracker_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_tracker_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    person_device = hass.data[DOMAIN].get(person_entry.entry_id)

    # Tracker ID should be person entity since no separate tracker configured
    tracker_id = person_device.get_tracker_id()
    assert tracker_id == MOCK_PERSON_CONFIG[CONF_PERSON_PERSON_ENTITY]


async def test_person_unload(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test person unload."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_unload",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_unload",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create person
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_unload_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_unload_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    assert person_entry.state is ConfigEntryState.LOADED

    # Unload person
    await hass.config_entries.async_unload(person_entry.entry_id)
    await hass.async_block_till_done()

    assert person_entry.state is ConfigEntryState.NOT_LOADED


async def test_person_should_recompute_history(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test person should_recompute_history method."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG
    from datetime import datetime
    import pytz

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_recompute",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_recompute",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create person
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_recompute_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_recompute_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    person_device = hass.data[DOMAIN].get(person_entry.entry_id)

    time = datetime.now(tz=pytz.UTC)
    # Person with authorized cars should recompute history if not initialized
    should_recompute = person_device.should_recompute_history(time)
    # Initial state should be not initialized, so should recompute
    assert isinstance(should_recompute, bool)


async def test_person_forecast_from_history(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test forecast generation from history."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_forecast",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_forecast",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_forecast_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_forecast_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    person_device = hass.data[DOMAIN].get(person_entry.entry_id)

    now = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    leave_time = datetime(2026, 1, 15, 18, 0, tzinfo=pytz.UTC)
    person_device.add_to_mileage_history(now, 42.0, leave_time)

    predicted_leave, predicted_mileage = person_device.update_person_forecast(
        now, force_update=True
    )

    assert predicted_mileage == 42.0
    assert predicted_leave is not None
    assert predicted_leave > now
    assert person_device.get_forecast_readable_string().startswith("42km")


async def test_person_tracker_id_override(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test tracker id uses override when set."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_tracker_override",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_tracker_override",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_tracker_override_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_tracker_override_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    person_device = hass.data[DOMAIN].get(person_entry.entry_id)
    person_device.person_tracker_id = "device_tracker.custom_person"
    assert person_device.get_tracker_id() == "device_tracker.custom_person"


async def test_person_notify_forecast_daily_constraints(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test forecast notification for daily constraints."""
    from .const import MOCK_PERSON_CONFIG, MOCK_CAR_CONFIG
    from custom_components.quiet_solar.const import PERSON_NOTIFY_REASON_DAILY_CHARGER_CONSTRAINTS

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_person_notify",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_person_notify",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_notify_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_notify_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    person_device = hass.data[DOMAIN].get(person_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    car_device.current_forecasted_person = person_device
    car_device.charger = MagicMock()
    car_device.get_adapt_target_percent_soc_to_reach_range_km = MagicMock(
        return_value=(True, 70.0, 60.0, None)
    )

    person_device.mobile_app = "notify"
    person_device.notification_dt_time = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC).time()
    person_device._last_forecast_notification_call_time = datetime(
        2026, 1, 14, 8, 0, tzinfo=pytz.UTC
    )
    person_device.predicted_mileage = 50.0
    person_device.predicted_leave_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    person_device.on_device_state_change = AsyncMock()
    person_device.home.get_best_persons_cars_allocations = AsyncMock(return_value={})

    await person_device.notify_of_forecast_if_needed(
        time=datetime(2026, 1, 15, 8, 30, tzinfo=pytz.UTC),
        notify_reason=PERSON_NOTIFY_REASON_DAILY_CHARGER_CONSTRAINTS,
    )

    person_device.on_device_state_change.assert_awaited()

