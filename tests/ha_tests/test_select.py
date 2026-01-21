"""Tests for quiet_solar select platform."""

import pytest
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import DOMAIN, DATA_HANDLER


pytestmark = pytest.mark.usefixtures("mock_sensor_states")


async def test_home_select_entities_created(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test home select entities are created."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, home_config_entry.entry_id
    )
    select_entries = [e for e in entity_entries if e.domain == "select"]
    assert len(select_entries) >= 1


async def test_home_mode_select_options(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test home mode select has correct options."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    state = hass.states.get("select.qs_test_home_home_home_mode")
    assert state is not None

    options = state.attributes.get("options", [])
    assert len(options) > 0


async def test_home_mode_select_change(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test home mode select can be changed."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    state = hass.states.get("select.qs_test_home_home_home_mode")
    assert state is not None

    options = state.attributes.get("options", [])
    if len(options) > 1:
        new_option = options[1] if state.state == options[0] else options[0]

        await hass.services.async_call(
            "select", "select_option",
            {"entity_id": "select.qs_test_home_home_home_mode", "option": new_option},
            blocking=True
        )
        await hass.async_block_till_done()

        new_state = hass.states.get("select.qs_test_home_home_home_mode")
        assert new_state.state == new_option


async def test_car_select_entities_created(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test car select entities are created."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_select_entity_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_select_entity_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, car_entry.entry_id
    )
    select_entries = [e for e in entity_entries if e.domain == "select"]
    assert len(select_entries) >= 2


async def test_charger_select_entities_created(
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
        entry_id="charger_select_entity_test2",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_select_entity_test2",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, charger_entry.entry_id
    )
    select_entries = [e for e in entity_entries if e.domain == "select"]
    assert len(select_entries) >= 1
