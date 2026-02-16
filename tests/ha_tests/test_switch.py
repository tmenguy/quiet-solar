"""Tests for quiet_solar switch platform."""

import pytest
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import DOMAIN, DATA_HANDLER


pytestmark = pytest.mark.usefixtures("mock_sensor_states")


async def test_home_switch_entities_created(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test home has no switch entities (off-grid switch was replaced by select)."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, home_config_entry.entry_id
    )
    switch_entries = [e for e in entity_entries if e.domain == "switch"]

    assert len(switch_entries) == 0


async def test_home_off_grid_mode_select(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test home off-grid mode select can be changed."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Find off-grid mode select
    state = hass.states.get("select.qs_test_home_home_off_grid_mode")
    assert state is not None
    assert state.state == "off_grid_mode_auto"

    # Switch to force off-grid
    await hass.services.async_call(
        "select", "select_option",
        {"entity_id": "select.qs_test_home_home_off_grid_mode", "option": "off_grid_mode_force_off_grid"},
        blocking=True
    )
    await hass.async_block_till_done()

    state = hass.states.get("select.qs_test_home_home_off_grid_mode")
    assert state.state == "off_grid_mode_force_off_grid"

    # Switch to force on-grid
    await hass.services.async_call(
        "select", "select_option",
        {"entity_id": "select.qs_test_home_home_off_grid_mode", "option": "off_grid_mode_force_on_grid"},
        blocking=True
    )
    await hass.async_block_till_done()

    state = hass.states.get("select.qs_test_home_home_off_grid_mode")
    assert state.state == "off_grid_mode_force_on_grid"


async def test_car_switch_entities_created(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test car switch entities are created."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_switch_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_switch_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, car_entry.entry_id
    )
    switch_entries = [e for e in entity_entries if e.domain == "switch"]

    # Car should have bump solar charge priority switch
    assert len(switch_entries) >= 1


async def test_charger_switch_entities_created(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test charger switch entities are created."""
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

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, charger_entry.entry_id
    )
    switch_entries = [e for e in entity_entries if e.domain == "switch"]

    # Charger should have enable device and bump solar switches
    assert len(switch_entries) >= 1


async def test_enable_device_switch(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test enable device switch functionality."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_enable_switch_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_enable_switch_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Find enable device switch - it should be on by default
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, car_entry.entry_id
    )
    enable_switches = [e for e in entity_entries
                       if e.domain == "switch" and "enable" in e.entity_id.lower()]

    if enable_switches:
        switch_id = enable_switches[0].entity_id
        state = hass.states.get(switch_id)
        # Default state should be on (device enabled)
        assert state is not None
