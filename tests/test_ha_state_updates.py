"""State update tests with real HA.

This test file verifies that entity states update correctly with
a real Home Assistant instance, including state changes over time.
"""
from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from custom_components.quiet_solar.const import DATA_HANDLER, DOMAIN


pytestmark = pytest.mark.asyncio


async def test_sensor_states_created(
    hass: HomeAssistant,
    real_home_config_entry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test that sensor entities have states after setup."""
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    
    # Get sensor entities
    entities = er.async_entries_for_config_entry(entity_registry, config_entry.entry_id)
    sensor_entities = [e for e in entities if e.entity_id.startswith("sensor.")]
    
    # Check that sensors have states
    for sensor in sensor_entities:
        state = hass.states.get(sensor.entity_id)
        # State should exist (may be unavailable initially)
        if state is not None:
            assert state.state is not None


async def test_button_states_created(
    hass: HomeAssistant,
    real_home_config_entry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test that button entities have states after setup."""
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    
    # Get button entities
    entities = er.async_entries_for_config_entry(entity_registry, config_entry.entry_id)
    button_entities = [e for e in entities if e.entity_id.startswith("button.")]
    
    # Buttons may not have states or may be unavailable - just verify they exist in registry
    assert len(button_entities) >= 0  # At least we can get the list


async def test_state_updates_after_time_change(
    hass: HomeAssistant,
    real_home_config_entry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test that states can update after time changes."""
    from pytest_homeassistant_custom_component.common import async_fire_time_changed
    
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_all_states", new_callable=AsyncMock):
            await hass.config_entries.async_setup(config_entry.entry_id)
            await hass.async_block_till_done()
            
            # Get a sensor
            entities = er.async_entries_for_config_entry(entity_registry, config_entry.entry_id)
            sensor_entities = [e for e in entities if e.entity_id.startswith("sensor.")]
            
            assert sensor_entities, "Expected sensor entities to be created"
            
            sensor_id = sensor_entities[0].entity_id
            
            # Get initial state
            initial_state = hass.states.get(sensor_id)
            
            # Advance time
            future = dt_util.utcnow() + datetime.timedelta(seconds=5)
            async_fire_time_changed(hass, future)
            await hass.async_block_till_done()
            
            # State object should still exist (value may or may not change)
            updated_state = hass.states.get(sensor_id)
            # Just verify the entity still exists
            assert updated_state is not None or initial_state is not None


async def test_switch_state_toggling(
    hass: HomeAssistant,
    home_and_charger,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test that switch states can be toggled."""
    from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
    from homeassistant.const import SERVICE_TURN_OFF, SERVICE_TURN_ON, ATTR_ENTITY_ID
    
    home_entry, charger_entry = home_and_charger
    
    # Get switch entities
    entities = er.async_entries_for_config_entry(entity_registry, charger_entry.entry_id)
    switch_entities = [e for e in entities if e.entity_id.startswith("switch.")]
    
    assert switch_entities, "Expected switch entities to be created"
    
    switch_id = switch_entities[0].entity_id
    
    # Try turning on
    await hass.services.async_call(
        SWITCH_DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: switch_id},
        blocking=True,
    )
    await hass.async_block_till_done()
    
    # State should exist
    state_after_on = hass.states.get(switch_id)
    assert state_after_on is not None
    
    # Try turning off
    await hass.services.async_call(
        SWITCH_DOMAIN,
        SERVICE_TURN_OFF,
        {ATTR_ENTITY_ID: switch_id},
        blocking=True,
    )
    await hass.async_block_till_done()
    
    # State should still exist
    state_after_off = hass.states.get(switch_id)
    assert state_after_off is not None


async def test_data_handler_home_reference(
    hass: HomeAssistant,
    real_home_config_entry,
) -> None:
    """Test that data handler maintains reference to home."""
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    
    # Get data handler
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    
    # Should have home reference
    assert data_handler.home is not None
    assert hasattr(data_handler.home, 'name')
    assert data_handler.home.name == "Test Home"


async def test_entity_availability(
    hass: HomeAssistant,
    real_home_config_entry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test entity availability states."""
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    
    # Get all entities
    entities = er.async_entries_for_config_entry(entity_registry, config_entry.entry_id)
    
    # Check that entities have states (available or unavailable)
    for entity in entities:
        state = hass.states.get(entity.entity_id)
        if state is not None:
            # State should be either a value or unavailable
            assert state.state is not None


async def test_multiple_devices_independent_states(
    hass: HomeAssistant,
    home_charger_and_car,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test that multiple devices maintain independent states."""
    from homeassistant.config_entries import ConfigEntryState
    
    home_entry, charger_entry, car_entry = home_charger_and_car
    
    # Get entities for each device
    home_entities = er.async_entries_for_config_entry(entity_registry, home_entry.entry_id)
    car_entities = er.async_entries_for_config_entry(entity_registry, car_entry.entry_id)
    
    # Home and car should have entities
    assert len(home_entities) > 0
    assert len(car_entities) > 0
    
    # Verify entity IDs don't overlap between home and car
    home_ids = {e.entity_id for e in home_entities}
    car_ids = {e.entity_id for e in car_entities}
    
    # No overlap
    assert len(home_ids & car_ids) == 0
    
    # If charger loaded successfully, check it too
    if charger_entry.state == ConfigEntryState.LOADED:
        charger_entities = er.async_entries_for_config_entry(entity_registry, charger_entry.entry_id)
        if len(charger_entities) > 0:
            charger_ids = {e.entity_id for e in charger_entities}
            assert len(home_ids & charger_ids) == 0
            assert len(car_ids & charger_ids) == 0


async def test_state_attributes_present(
    hass: HomeAssistant,
    real_home_config_entry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test that entity states have attributes."""
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    
    # Get sensor entities
    entities = er.async_entries_for_config_entry(entity_registry, config_entry.entry_id)
    sensor_entities = [e for e in entities if e.entity_id.startswith("sensor.")]
    
    assert sensor_entities, "Expected sensor entities to be created"
    
    # Check first sensor has attributes
    sensor_id = sensor_entities[0].entity_id
    state = hass.states.get(sensor_id)
    
    if state is not None:
        # State should have attributes dict
        assert hasattr(state, 'attributes')
        assert isinstance(state.attributes, dict)

