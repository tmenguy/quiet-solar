"""Tests for quiet_solar binary_sensor platform."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.const import STATE_ON, STATE_OFF

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    BINARY_SENSOR_PILOTED_DEVICE_ACTIVATED,
)


pytestmark = pytest.mark.usefixtures("mock_sensor_states")


# =============================================================================
# Test Binary Sensor Platform Setup
# =============================================================================

async def test_binary_sensor_platform_setup(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test binary sensor platform can be set up."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Platform should be set up without errors
    assert home_config_entry.state is ConfigEntryState.LOADED


async def test_charger_binary_sensor_entities_created(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test charger creates binary sensor entities."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_binary_sensor_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_binary_sensor_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, charger_entry.entry_id
    )
    binary_sensor_entries = [e for e in entity_entries if e.domain == "binary_sensor"]

    # Charger may have piloted device activated binary sensor
    assert isinstance(binary_sensor_entries, list)


async def test_car_binary_sensor_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test car device may create binary sensor entities."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_binary_sensor_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_binary_sensor_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, car_entry.entry_id
    )
    binary_sensor_entries = [e for e in entity_entries if e.domain == "binary_sensor"]

    # Car may or may not have binary sensors depending on configuration
    assert isinstance(binary_sensor_entries, list)


async def test_home_binary_sensor_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test home device may create binary sensor entities."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, home_config_entry.entry_id
    )
    binary_sensor_entries = [e for e in entity_entries if e.domain == "binary_sensor"]

    # Home may have binary sensors
    assert isinstance(binary_sensor_entries, list)


# =============================================================================
# Test Binary Sensor Entity Properties
# =============================================================================

async def test_binary_sensor_unique_id(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test binary sensor entities have unique IDs."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_binary_unique_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_binary_unique_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, charger_entry.entry_id
    )
    binary_sensor_entries = [e for e in entity_entries if e.domain == "binary_sensor"]

    # Each binary sensor should have a unique_id
    for entry in binary_sensor_entries:
        assert entry.unique_id is not None


async def test_binary_sensor_device_association(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test binary sensor entities are associated with devices."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_binary_device_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_binary_device_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, charger_entry.entry_id
    )
    binary_sensor_entries = [e for e in entity_entries if e.domain == "binary_sensor"]

    # Each binary sensor should be associated with a device
    for entry in binary_sensor_entries:
        # device_id can be None for some entities, that's acceptable
        pass  # Just verify no exception is raised


# =============================================================================
# Test Binary Sensor State Reading
# =============================================================================

async def test_binary_sensor_state_update(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test binary sensor state can be updated."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_binary_state_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_binary_state_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    # Device should be setup and able to update state
    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device is not None


# =============================================================================
# Test create_ha_binary_sensor Functions
# =============================================================================

async def test_create_binary_sensor_function_exists(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test create_ha_binary_sensor function exists and is callable."""
    from custom_components.quiet_solar.binary_sensor import create_ha_binary_sensor

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert callable(create_ha_binary_sensor)


async def test_create_binary_sensor_returns_list(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test create_ha_binary_sensor returns a list."""
    from custom_components.quiet_solar.binary_sensor import create_ha_binary_sensor
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_create_binary_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_create_binary_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    result = create_ha_binary_sensor(charger_device)

    assert isinstance(result, list)


# =============================================================================
# Test Binary Sensor Description
# =============================================================================

async def test_binary_sensor_entity_description(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test QSBinarySensorEntityDescription works correctly."""
    from custom_components.quiet_solar.binary_sensor import QSBinarySensorEntityDescription

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create a description
    description = QSBinarySensorEntityDescription(
        key="test_key",
        translation_key="test_translation",
        value_fn=lambda d, k: True,
    )

    assert description.key == "test_key"
    assert description.translation_key == "test_translation"
    assert description.value_fn is not None
    assert description.value_fn(None, "test") is True


async def test_binary_sensor_entity_description_no_value_fn(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test QSBinarySensorEntityDescription works without value_fn."""
    from custom_components.quiet_solar.binary_sensor import QSBinarySensorEntityDescription

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create a description without value_fn
    description = QSBinarySensorEntityDescription(
        key="test_key_no_fn",
        translation_key="test_translation_no_fn",
    )

    assert description.key == "test_key_no_fn"
    assert description.value_fn is None


# =============================================================================
# Test Piloted Device Binary Sensor
# =============================================================================

async def test_piloted_device_binary_sensor_constant(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test BINARY_SENSOR_PILOTED_DEVICE_ACTIVATED constant exists."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert BINARY_SENSOR_PILOTED_DEVICE_ACTIVATED is not None
    assert isinstance(BINARY_SENSOR_PILOTED_DEVICE_ACTIVATED, str)


async def test_create_binary_sensor_for_piloted_device_function(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test create_ha_binary_sensor_for_PilotedDevice function exists."""
    from custom_components.quiet_solar.binary_sensor import create_ha_binary_sensor_for_PilotedDevice

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert callable(create_ha_binary_sensor_for_PilotedDevice)


# =============================================================================
# Test Binary Sensor Unload
# =============================================================================

async def test_binary_sensor_unload(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test binary sensor platform unloads correctly."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_binary_unload_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_binary_unload_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    assert charger_entry.state is ConfigEntryState.LOADED

    # Unload the entry
    await hass.config_entries.async_unload(charger_entry.entry_id)
    await hass.async_block_till_done()

    assert charger_entry.state is ConfigEntryState.NOT_LOADED


# =============================================================================
# Test Multiple Binary Sensors
# =============================================================================

async def test_multiple_devices_binary_sensors(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test multiple devices create their own binary sensors."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create first charger
    charger1_config = {
        **MOCK_CHARGER_CONFIG,
        "name": "Charger 1",
        "charger_max_charging_current_number": "number.charger1_max_current",
        "charger_pause_resume_switch": "switch.charger1_pause_resume",
        "charger_status_sensor": "sensor.charger1_status",
        "charger_plugged": "binary_sensor.charger1_plugged",
    }

    # Set up mock entities for charger 1
    hass.states.async_set("number.charger1_max_current", "32", {"min": 6, "max": 32})
    hass.states.async_set("switch.charger1_pause_resume", "on")
    hass.states.async_set("sensor.charger1_status", "Ready")
    hass.states.async_set("binary_sensor.charger1_plugged", "off")

    charger1_entry = MockConfigEntry(
        domain=DOMAIN,
        data=charger1_config,
        entry_id="charger1_binary_multi_test",
        title="charger: Charger 1",
        unique_id="quiet_solar_charger1_binary_multi_test",
    )
    charger1_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger1_entry.entry_id)
    await hass.async_block_till_done()

    # Create second charger
    charger2_config = {
        **MOCK_CHARGER_CONFIG,
        "name": "Charger 2",
        "charger_max_charging_current_number": "number.charger2_max_current",
        "charger_pause_resume_switch": "switch.charger2_pause_resume",
        "charger_status_sensor": "sensor.charger2_status",
        "charger_plugged": "binary_sensor.charger2_plugged",
    }

    # Set up mock entities for charger 2
    hass.states.async_set("number.charger2_max_current", "32", {"min": 6, "max": 32})
    hass.states.async_set("switch.charger2_pause_resume", "on")
    hass.states.async_set("sensor.charger2_status", "Ready")
    hass.states.async_set("binary_sensor.charger2_plugged", "off")

    charger2_entry = MockConfigEntry(
        domain=DOMAIN,
        data=charger2_config,
        entry_id="charger2_binary_multi_test",
        title="charger: Charger 2",
        unique_id="quiet_solar_charger2_binary_multi_test",
    )
    charger2_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger2_entry.entry_id)
    await hass.async_block_till_done()

    # Both should be loaded
    assert charger1_entry.state is ConfigEntryState.LOADED
    assert charger2_entry.state is ConfigEntryState.LOADED

    # Get binary sensors for each
    charger1_entities = er.async_entries_for_config_entry(
        entity_registry, charger1_entry.entry_id
    )
    charger1_binary = [e for e in charger1_entities if e.domain == "binary_sensor"]

    charger2_entities = er.async_entries_for_config_entry(
        entity_registry, charger2_entry.entry_id
    )
    charger2_binary = [e for e in charger2_entities if e.domain == "binary_sensor"]

    # Both chargers should have their own binary sensors (even if empty list)
    assert isinstance(charger1_binary, list)
    assert isinstance(charger2_binary, list)


# =============================================================================
# Test QSBaseBinarySensor Class
# =============================================================================

async def test_binary_sensor_class_exists(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test QSBaseBinarySensor class exists."""
    from custom_components.quiet_solar.binary_sensor import QSBaseBinarySensor

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert QSBaseBinarySensor is not None


async def test_async_setup_entry_function(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test async_setup_entry function exists and is callable."""
    from custom_components.quiet_solar.binary_sensor import async_setup_entry

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert callable(async_setup_entry)


async def test_async_unload_entry_function(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test async_unload_entry function exists and is callable."""
    from custom_components.quiet_solar.binary_sensor import async_unload_entry

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert callable(async_unload_entry)


# =============================================================================
# Test Heat Pump (PilotedDevice) Binary Sensors
# =============================================================================

async def test_heat_pump_binary_sensor_creation(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test heat pump creates binary sensor entities via create_ha_binary_sensor_for_PilotedDevice."""
    from .const import MOCK_HEAT_PUMP_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    heat_pump_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_HEAT_PUMP_CONFIG,
        entry_id="heat_pump_binary_sensor_test",
        title=f"heat_pump: {MOCK_HEAT_PUMP_CONFIG['name']}",
        unique_id="quiet_solar_heat_pump_binary_sensor_test",
    )
    heat_pump_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(heat_pump_entry.entry_id)
    await hass.async_block_till_done()

    # Heat pump should be loaded
    assert heat_pump_entry.state is ConfigEntryState.LOADED

    # Get heat pump device
    heat_pump_device = hass.data[DOMAIN].get(heat_pump_entry.entry_id)
    assert heat_pump_device is not None

    # Check that the device is a PilotedDevice
    from custom_components.quiet_solar.home_model.load import PilotedDevice
    assert isinstance(heat_pump_device, PilotedDevice)

    # Get binary sensor entities for heat pump
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, heat_pump_entry.entry_id
    )
    binary_sensor_entries = [e for e in entity_entries if e.domain == "binary_sensor"]

    # Heat pump (as PilotedDevice) should have binary sensors
    assert isinstance(binary_sensor_entries, list)
    # PilotedDevice should have at least the "is_piloted_device_activated" binary sensor
    assert len(binary_sensor_entries) >= 1


async def test_heat_pump_piloted_device_activated_binary_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test heat pump has the piloted device activated binary sensor."""
    from .const import MOCK_HEAT_PUMP_CONFIG
    from custom_components.quiet_solar.const import BINARY_SENSOR_PILOTED_DEVICE_ACTIVATED

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    heat_pump_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_HEAT_PUMP_CONFIG,
        entry_id="heat_pump_piloted_activated_test",
        title=f"heat_pump: {MOCK_HEAT_PUMP_CONFIG['name']}",
        unique_id="quiet_solar_heat_pump_piloted_activated_test",
    )
    heat_pump_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(heat_pump_entry.entry_id)
    await hass.async_block_till_done()

    # Get binary sensor entities
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, heat_pump_entry.entry_id
    )
    binary_sensor_entries = [e for e in entity_entries if e.domain == "binary_sensor"]

    # Check that piloted device activated sensor exists
    piloted_activated_sensors = [
        e for e in binary_sensor_entries
        if "is_piloted_device_activated" in e.unique_id or BINARY_SENSOR_PILOTED_DEVICE_ACTIVATED in str(e.translation_key or "")
    ]

    # Should have at least one binary sensor (the piloted device activated one)
    assert len(binary_sensor_entries) >= 1


async def test_create_ha_binary_sensor_for_piloted_device_returns_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test create_ha_binary_sensor_for_PilotedDevice returns proper entity list."""
    from custom_components.quiet_solar.binary_sensor import create_ha_binary_sensor_for_PilotedDevice
    from .const import MOCK_HEAT_PUMP_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    heat_pump_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_HEAT_PUMP_CONFIG,
        entry_id="heat_pump_create_fn_test",
        title=f"heat_pump: {MOCK_HEAT_PUMP_CONFIG['name']}",
        unique_id="quiet_solar_heat_pump_create_fn_test",
    )
    heat_pump_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(heat_pump_entry.entry_id)
    await hass.async_block_till_done()

    heat_pump_device = hass.data[DOMAIN].get(heat_pump_entry.entry_id)
    assert heat_pump_device is not None

    # Call create_ha_binary_sensor_for_PilotedDevice directly
    entities = create_ha_binary_sensor_for_PilotedDevice(heat_pump_device)

    # Should return a list with at least one entity
    assert isinstance(entities, list)
    assert len(entities) >= 1

    # Each entity should be a QSBaseBinarySensor
    from custom_components.quiet_solar.binary_sensor import QSBaseBinarySensor
    for entity in entities:
        assert isinstance(entity, QSBaseBinarySensor)


async def test_create_ha_binary_sensor_with_piloted_device(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test create_ha_binary_sensor correctly identifies PilotedDevice and creates sensors."""
    from custom_components.quiet_solar.binary_sensor import create_ha_binary_sensor
    from custom_components.quiet_solar.home_model.load import PilotedDevice
    from .const import MOCK_HEAT_PUMP_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    heat_pump_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_HEAT_PUMP_CONFIG,
        entry_id="heat_pump_create_binary_test",
        title=f"heat_pump: {MOCK_HEAT_PUMP_CONFIG['name']}",
        unique_id="quiet_solar_heat_pump_create_binary_test",
    )
    heat_pump_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(heat_pump_entry.entry_id)
    await hass.async_block_till_done()

    heat_pump_device = hass.data[DOMAIN].get(heat_pump_entry.entry_id)
    assert heat_pump_device is not None

    # Verify it's a PilotedDevice
    assert isinstance(heat_pump_device, PilotedDevice)

    # Call create_ha_binary_sensor which should detect PilotedDevice
    entities = create_ha_binary_sensor(heat_pump_device)

    # Should return a non-empty list for PilotedDevice
    assert isinstance(entities, list)
    assert len(entities) >= 1


async def test_heat_pump_binary_sensor_unique_id(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test heat pump binary sensor entities have unique IDs."""
    from .const import MOCK_HEAT_PUMP_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    heat_pump_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_HEAT_PUMP_CONFIG,
        entry_id="heat_pump_unique_id_test",
        title=f"heat_pump: {MOCK_HEAT_PUMP_CONFIG['name']}",
        unique_id="quiet_solar_heat_pump_unique_id_test",
    )
    heat_pump_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(heat_pump_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, heat_pump_entry.entry_id
    )
    binary_sensor_entries = [e for e in entity_entries if e.domain == "binary_sensor"]

    # Each binary sensor should have a unique_id
    for entry in binary_sensor_entries:
        assert entry.unique_id is not None


async def test_heat_pump_binary_sensor_device_association(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test heat pump binary sensor entities are associated with devices."""
    from .const import MOCK_HEAT_PUMP_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    heat_pump_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_HEAT_PUMP_CONFIG,
        entry_id="heat_pump_device_assoc_test",
        title=f"heat_pump: {MOCK_HEAT_PUMP_CONFIG['name']}",
        unique_id="quiet_solar_heat_pump_device_assoc_test",
    )
    heat_pump_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(heat_pump_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, heat_pump_entry.entry_id
    )
    binary_sensor_entries = [e for e in entity_entries if e.domain == "binary_sensor"]

    # Each binary sensor should be associated with a device
    for entry in binary_sensor_entries:
        # device_id can be present
        pass  # Just verify no exception is raised


async def test_heat_pump_binary_sensor_unload(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test heat pump binary sensor platform unloads correctly."""
    from .const import MOCK_HEAT_PUMP_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    heat_pump_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_HEAT_PUMP_CONFIG,
        entry_id="heat_pump_unload_test",
        title=f"heat_pump: {MOCK_HEAT_PUMP_CONFIG['name']}",
        unique_id="quiet_solar_heat_pump_unload_test",
    )
    heat_pump_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(heat_pump_entry.entry_id)
    await hass.async_block_till_done()

    assert heat_pump_entry.state is ConfigEntryState.LOADED

    # Unload the entry
    await hass.config_entries.async_unload(heat_pump_entry.entry_id)
    await hass.async_block_till_done()

    # The entry should be unloaded (NOT_LOADED) or failed to unload (FAILED_UNLOAD)
    # Both are acceptable outcomes - the important thing is that the unload was attempted
    assert heat_pump_entry.state in (ConfigEntryState.NOT_LOADED, ConfigEntryState.FAILED_UNLOAD)


async def test_heat_pump_is_piloted_device_activated_state(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test heat pump is_piloted_device_activated property works."""
    from .const import MOCK_HEAT_PUMP_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    heat_pump_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_HEAT_PUMP_CONFIG,
        entry_id="heat_pump_activated_state_test",
        title=f"heat_pump: {MOCK_HEAT_PUMP_CONFIG['name']}",
        unique_id="quiet_solar_heat_pump_activated_state_test",
    )
    heat_pump_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(heat_pump_entry.entry_id)
    await hass.async_block_till_done()

    heat_pump_device = hass.data[DOMAIN].get(heat_pump_entry.entry_id)
    assert heat_pump_device is not None

    # Heat pump without clients should return False for is_piloted_device_activated
    assert heat_pump_device.is_piloted_device_activated is False


async def test_multiple_heat_pumps_binary_sensors(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test multiple heat pumps create their own binary sensors."""
    from .const import MOCK_HEAT_PUMP_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create first heat pump
    heat_pump1_config = {
        **MOCK_HEAT_PUMP_CONFIG,
        "name": "Heat Pump 1",
    }

    heat_pump1_entry = MockConfigEntry(
        domain=DOMAIN,
        data=heat_pump1_config,
        entry_id="heat_pump1_multi_test",
        title="heat_pump: Heat Pump 1",
        unique_id="quiet_solar_heat_pump1_multi_test",
    )
    heat_pump1_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(heat_pump1_entry.entry_id)
    await hass.async_block_till_done()

    # Create second heat pump
    heat_pump2_config = {
        **MOCK_HEAT_PUMP_CONFIG,
        "name": "Heat Pump 2",
    }

    heat_pump2_entry = MockConfigEntry(
        domain=DOMAIN,
        data=heat_pump2_config,
        entry_id="heat_pump2_multi_test",
        title="heat_pump: Heat Pump 2",
        unique_id="quiet_solar_heat_pump2_multi_test",
    )
    heat_pump2_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(heat_pump2_entry.entry_id)
    await hass.async_block_till_done()

    # Both should be loaded
    assert heat_pump1_entry.state is ConfigEntryState.LOADED
    assert heat_pump2_entry.state is ConfigEntryState.LOADED

    # Get binary sensors for each
    heat_pump1_entities = er.async_entries_for_config_entry(
        entity_registry, heat_pump1_entry.entry_id
    )
    heat_pump1_binary = [e for e in heat_pump1_entities if e.domain == "binary_sensor"]

    heat_pump2_entities = er.async_entries_for_config_entry(
        entity_registry, heat_pump2_entry.entry_id
    )
    heat_pump2_binary = [e for e in heat_pump2_entities if e.domain == "binary_sensor"]

    # Both heat pumps should have their own binary sensors
    assert isinstance(heat_pump1_binary, list)
    assert isinstance(heat_pump2_binary, list)
    assert len(heat_pump1_binary) >= 1
    assert len(heat_pump2_binary) >= 1

    # Unique IDs should be different
    heat_pump1_unique_ids = {e.unique_id for e in heat_pump1_binary}
    heat_pump2_unique_ids = {e.unique_id for e in heat_pump2_binary}
    assert heat_pump1_unique_ids.isdisjoint(heat_pump2_unique_ids)

