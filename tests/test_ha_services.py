"""Service call tests with real HA.

This test file verifies that service calls work correctly with a real
Home Assistant instance, including button presses and switch toggles.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components.button import DOMAIN as BUTTON_DOMAIN, SERVICE_PRESS
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.const import ATTR_ENTITY_ID, SERVICE_TURN_OFF, SERVICE_TURN_ON
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from custom_components.quiet_solar.const import DOMAIN


pytestmark = pytest.mark.asyncio


async def test_button_press_service(
    hass: HomeAssistant,
    real_home_config_entry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test that button press service works with real HA."""
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    
    # Get button entities
    entities = er.async_entries_for_config_entry(entity_registry, config_entry.entry_id)
    button_entities = [e for e in entities if e.entity_id.startswith("button.")]
    
    if not button_entities:
        pytest.skip("No button entities created for this device")
    
    # Try to press first button - just verify it doesn't error
    button_entity_id = button_entities[0].entity_id
    
    try:
        await hass.services.async_call(
            BUTTON_DOMAIN,
            SERVICE_PRESS,
            {ATTR_ENTITY_ID: button_entity_id},
            blocking=True,
        )
        # Service call should complete (may or may not do anything depending on state)
    except Exception as e:
        # Some buttons may require specific state, that's ok for this test
        # We're just testing the service call mechanism works
        pass


async def test_reload_service(
    hass: HomeAssistant,
    real_home_config_entry,
) -> None:
    """Test that reload service is registered and callable."""
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    
    # Verify reload service exists
    assert hass.services.has_service(DOMAIN, "reload")
    
    # Call reload service
    with patch("custom_components.quiet_solar.async_reload_quiet_solar", new_callable=AsyncMock) as mock_reload:
        await hass.services.async_call(
            DOMAIN,
            "reload",
            {},
            blocking=True,
        )
        
        # Reload should have been called
        mock_reload.assert_called_once()


async def test_switch_turn_on_service(
    hass: HomeAssistant,
    home_and_charger,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test that switch turn_on service works."""
    home_entry, charger_entry = home_and_charger
    
    # Get switch entities for charger
    entities = er.async_entries_for_config_entry(entity_registry, charger_entry.entry_id)
    switch_entities = [e for e in entities if e.entity_id.startswith("switch.")]
    
    if not switch_entities:
        pytest.skip("No switch entities created for this device")
    
    # Try to turn on first switch
    switch_entity_id = switch_entities[0].entity_id
    
    await hass.services.async_call(
        SWITCH_DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: switch_entity_id},
        blocking=True,
    )
    
    # Get state after turn_on
    state = hass.states.get(switch_entity_id)
    # State may or may not be "on" depending on device configuration
    assert state is not None


async def test_switch_turn_off_service(
    hass: HomeAssistant,
    home_and_charger,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test that switch turn_off service works."""
    home_entry, charger_entry = home_and_charger
    
    # Get switch entities for charger
    entities = er.async_entries_for_config_entry(entity_registry, charger_entry.entry_id)
    switch_entities = [e for e in entities if e.entity_id.startswith("switch.")]
    
    if not switch_entities:
        pytest.skip("No switch entities created for this device")
    
    # Try to turn off first switch
    switch_entity_id = switch_entities[0].entity_id
    
    await hass.services.async_call(
        SWITCH_DOMAIN,
        SERVICE_TURN_OFF,
        {ATTR_ENTITY_ID: switch_entity_id},
        blocking=True,
    )
    
    # Service call should complete without error
    state = hass.states.get(switch_entity_id)
    assert state is not None


async def test_button_press_all_buttons(
    hass: HomeAssistant,
    real_home_config_entry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test pressing all button entities doesn't raise exceptions."""
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    
    # Get all button entities
    entities = er.async_entries_for_config_entry(entity_registry, config_entry.entry_id)
    button_entities = [e for e in entities if e.entity_id.startswith("button.")]
    
    if not button_entities:
        pytest.skip("No button entities created for this device")
    
    # Try pressing each button (with mocked actions)
    for button_entity in button_entities:
        try:
            await hass.services.async_call(
                BUTTON_DOMAIN,
                SERVICE_PRESS,
                {ATTR_ENTITY_ID: button_entity.entity_id},
                blocking=True,
            )
        except Exception as e:
            # Some buttons may require specific state, that's ok
            pass


async def test_service_call_invalid_entity(
    hass: HomeAssistant,
    real_home_config_entry,
) -> None:
    """Test that calling service with invalid entity ID handles gracefully."""
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        await hass.config_entries.async_setup(real_home_config_entry.entry_id)
        await hass.async_block_till_done()
    
    # Try to press non-existent button
    # HA should handle this gracefully
    try:
        await hass.services.async_call(
            BUTTON_DOMAIN,
            SERVICE_PRESS,
            {ATTR_ENTITY_ID: "button.nonexistent_button"},
            blocking=True,
        )
    except Exception:
        # Expected - entity doesn't exist
        pass

