"""Tests for quiet_solar charger.py functionality."""

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
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_MAX_CHARGE,
)


pytestmark = pytest.mark.usefixtures("mock_sensor_states")


async def test_charger_initialization(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger device initialization."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_init_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_init_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    assert charger_entry.state is ConfigEntryState.LOADED

    # Verify charger device was created
    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device is not None
    assert charger_device.name == MOCK_CHARGER_CONFIG['name']


async def test_charger_amp_limits(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger amperage limits configuration."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_amp_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_amp_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device.charger_min_charge == MOCK_CHARGER_CONFIG[CONF_CHARGER_MIN_CHARGE]
    assert charger_device.charger_max_charge == MOCK_CHARGER_CONFIG[CONF_CHARGER_MAX_CHARGE]


async def test_charger_status_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger status sensor reading."""
    from .const import MOCK_CHARGER_CONFIG

    # Set up mock status sensor
    hass.states.async_set("sensor.test_charger_status", "Charging")

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_status_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_status_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device.charger_status_sensor == "sensor.test_charger_status"


async def test_charger_pause_resume_switch(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger pause/resume switch configuration."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_switch_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_switch_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device.charger_pause_resume_switch == "switch.test_charger_pause_resume"


async def test_charger_max_current_number(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger max current number entity configuration."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_current_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_current_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device.charger_max_charging_current_number == "number.test_charger_max_current"


async def test_charger_plugged_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger plugged sensor configuration."""
    from .const import MOCK_CHARGER_CONFIG

    # Set up mock plugged sensor
    hass.states.async_set("binary_sensor.test_charger_plugged", "on")

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_plugged_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_plugged_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device.charger_plugged == "binary_sensor.test_charger_plugged"


async def test_charger_select_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test charger select entities are created."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_select_entity_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_select_entity_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    # Check select entities
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, charger_entry.entry_id
    )

    select_entries = [e for e in entity_entries if e.domain == "select"]

    # Should have at least connected car selection
    assert len(select_entries) >= 1, f"Expected at least 1 select entity, got {len(select_entries)}"


async def test_charger_button_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test charger button entities are created."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_button_entity_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_button_entity_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    # Check button entities
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, charger_entry.entry_id
    )

    button_entries = [e for e in entity_entries if e.domain == "button"]

    # Should have buttons: reset, clean constraints
    assert len(button_entries) >= 2, f"Expected at least 2 button entities, got {len(button_entries)}"


async def test_charger_generic_car_creation(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test that charger creates a generic car for unknown vehicles."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_generic_car_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_generic_car_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    # Charger should have a default generic car
    assert hasattr(charger_device, '_default_generic_car')
    assert charger_device._default_generic_car is not None


async def test_charger_with_car_connected(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger with a car connected."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create charger
    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_connected_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_connected_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_connected_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_connected_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    assert charger_entry.state is ConfigEntryState.LOADED
    assert car_entry.state is ConfigEntryState.LOADED


async def test_multiple_chargers(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test multiple chargers can be created."""
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
        entry_id="charger1_multi_test",
        title="charger: Charger 1",
        unique_id="quiet_solar_charger1_multi_test",
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
        entry_id="charger2_multi_test",
        title="charger: Charger 2",
        unique_id="quiet_solar_charger2_multi_test",
    )
    charger2_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger2_entry.entry_id)
    await hass.async_block_till_done()

    assert charger1_entry.state is ConfigEntryState.LOADED
    assert charger2_entry.state is ConfigEntryState.LOADED


async def test_charger_single_phase(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger configured for single phase."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,  # Already configured for single phase
        entry_id="charger_1p_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_1p_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device is not None
    # Single phase config
    assert MOCK_CHARGER_CONFIG.get("device_is_3p") is False


async def test_charger_three_phase(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger configured for three phase."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create 3-phase charger config
    charger_3p_config = {
        **MOCK_CHARGER_CONFIG,
        "name": "Charger 3P",
        "device_is_3p": True,
        "charger_max_charging_current_number": "number.charger_3p_max_current",
        "charger_pause_resume_switch": "switch.charger_3p_pause_resume",
        "charger_status_sensor": "sensor.charger_3p_status",
        "charger_plugged": "binary_sensor.charger_3p_plugged",
    }

    # Set up mock entities
    hass.states.async_set("number.charger_3p_max_current", "32", {"min": 6, "max": 32})
    hass.states.async_set("switch.charger_3p_pause_resume", "on")
    hass.states.async_set("sensor.charger_3p_status", "Ready")
    hass.states.async_set("binary_sensor.charger_3p_plugged", "off")

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=charger_3p_config,
        entry_id="charger_3p_test",
        title="charger: Charger 3P",
        unique_id="quiet_solar_charger_3p_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device is not None
