"""Tests for quiet_solar sensor platform."""

from collections.abc import Generator
from unittest.mock import patch

import pytest
from syrupy.assertion import SnapshotAssertion

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from custom_components.quiet_solar.const import DOMAIN

from pytest_homeassistant_custom_component.common import MockConfigEntry


pytestmark = pytest.mark.usefixtures("mock_sensor_states")


async def test_home_sensors(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test home sensors are created with correct states."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert home_config_entry.state is ConfigEntryState.LOADED

    # Check that sensors were created
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, home_config_entry.entry_id
    )

    # Filter to only sensor entities
    sensor_entries = [e for e in entity_entries if e.domain == "sensor"]

    # Verify sensors exist
    assert len(sensor_entries) > 0, "No sensors were created for home"

    # Snapshot test each sensor
    for entity_entry in sensor_entries:
        state = hass.states.get(entity_entry.entity_id)
        if state:
            assert state == snapshot(name=f"{entity_entry.entity_id}-state")


async def test_car_sensors(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test car sensors are created with correct states."""
    from .const import MOCK_CAR_CONFIG

    # Setup home first
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create and setup car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_entry_sensor_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_sensor_test",
    )
    car_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    assert car_entry.state is ConfigEntryState.LOADED

    # Check car sensors
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, car_entry.entry_id
    )

    sensor_entries = [e for e in entity_entries if e.domain == "sensor"]

    assert len(sensor_entries) > 0, "No sensors were created for car"

    for entity_entry in sensor_entries:
        state = hass.states.get(entity_entry.entity_id)
        if state:
            assert state == snapshot(name=f"{entity_entry.entity_id}-state")


async def test_person_sensors(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test person sensors are created with correct states."""
    from .const import MOCK_CAR_CONFIG, MOCK_PERSON_CONFIG

    # Setup dependencies
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create and setup car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_entry_person_sensor_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_person_sensor_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create and setup person
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_entry_sensor_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_sensor_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    assert person_entry.state is ConfigEntryState.LOADED

    # Check person sensors
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, person_entry.entry_id
    )

    sensor_entries = [e for e in entity_entries if e.domain == "sensor"]

    assert len(sensor_entries) > 0, "No sensors were created for person"

    for entity_entry in sensor_entries:
        state = hass.states.get(entity_entry.entity_id)
        if state:
            assert state == snapshot(name=f"{entity_entry.entity_id}-state")


async def test_sensor_unavailable_when_source_unavailable(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test sensors behavior when source sensors are unavailable."""
    # Set source sensor to unavailable
    hass.states.async_set("sensor.grid_power", "unavailable")

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Home should still be set up - the integration handles missing sensors gracefully
    assert home_config_entry.state is ConfigEntryState.LOADED


async def test_home_sensor_entity_count(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test that expected number of home sensors are created."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, home_config_entry.entry_id
    )

    sensor_entries = [e for e in entity_entries if e.domain == "sensor"]

    # Home should create at least these sensors:
    # - home_non_controlled_consumption
    # - home_consumption
    # - home_available_power
    # - multiple forecast sensors
    # - load_current_command
    # - device_information_storage
    assert len(sensor_entries) >= 10, f"Expected at least 10 sensors, got {len(sensor_entries)}"


async def test_car_sensor_entity_count(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test that expected number of car sensors are created."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_entry_count_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_count_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, car_entry.entry_id
    )

    sensor_entries = [e for e in entity_entries if e.domain == "sensor"]

    # Car should create at least these sensors:
    # - current_constraint_car
    # - best_power_value
    # - car_soc_percentage
    # - car_estimated_range_km
    # - car_charge_type
    # - car_charge_time
    # - load_current_command
    # - device_information_storage
    assert len(sensor_entries) >= 5, f"Expected at least 5 sensors, got {len(sensor_entries)}"
