"""Tests for quiet_solar __init__.py integration setup."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.const import Platform

from custom_components.quiet_solar import (
    async_setup,
    async_setup_entry,
    async_unload_entry,
    async_reload_entry,
    async_reload_quiet_solar,
    register_reload_service,
    register_ocpp_notification_listener,
)
from custom_components.quiet_solar.const import DOMAIN, DATA_HANDLER, DEVICE_TYPE
from custom_components.quiet_solar.data_handler import QSDataHandler
from custom_components.quiet_solar.ha_model.home import QSHome
from tests.test_helpers import create_mock_device


@pytest.mark.asyncio
async def test_async_setup_registers_services(fake_hass):
    """Test that async_setup registers reload service and OCPP listener."""
    result = await async_setup(fake_hass, {})
    
    assert result is True
    assert DOMAIN in fake_hass.data
    # Bus should have registered the call_service listener for OCPP
    assert "call_service" in fake_hass.bus.listeners


@pytest.mark.asyncio
async def test_register_reload_service(fake_hass):
    """Test reload service registration."""
    with patch('custom_components.quiet_solar.service.async_register_admin_service') as mock_register:
        register_reload_service(fake_hass)
        
        mock_register.assert_called_once()
        assert mock_register.call_args[0][0] == fake_hass
        assert mock_register.call_args[0][1] == DOMAIN
        assert mock_register.call_args[0][2] == "reload"


@pytest.mark.asyncio
async def test_register_ocpp_notification_listener(fake_hass):
    """Test OCPP notification listener registration."""
    register_ocpp_notification_listener(fake_hass)
    
    assert "call_service" in fake_hass.bus.listeners


@pytest.mark.asyncio
async def test_async_setup_entry_creates_data_handler(fake_hass, mock_home_config_entry):
    """Test setup entry creates data handler if not exists."""
    mock_home_config_entry.data[DEVICE_TYPE] = QSHome.conf_type_name
    
    with patch.object(QSDataHandler, 'async_add_entry', new_callable=AsyncMock) as mock_add:
        result = await async_setup_entry(fake_hass, mock_home_config_entry)
        
        assert result is True
        assert DATA_HANDLER in fake_hass.data[DOMAIN]
        assert isinstance(fake_hass.data[DOMAIN][DATA_HANDLER], QSDataHandler)
        mock_add.assert_called_once_with(mock_home_config_entry)


@pytest.mark.asyncio
async def test_async_setup_entry_reuses_existing_data_handler(fake_hass, mock_home_config_entry):
    """Test setup entry reuses existing data handler."""
    existing_handler = QSDataHandler(fake_hass)
    fake_hass.data[DOMAIN][DATA_HANDLER] = existing_handler
    
    with patch.object(existing_handler, 'async_add_entry', new_callable=AsyncMock) as mock_add:
        result = await async_setup_entry(fake_hass, mock_home_config_entry)
        
        assert result is True
        assert fake_hass.data[DOMAIN][DATA_HANDLER] is existing_handler
        mock_add.assert_called_once()


@pytest.mark.asyncio
async def test_async_setup_entry_registers_update_listener(fake_hass, mock_home_config_entry):
    """Test that update listener is registered."""
    with patch.object(QSDataHandler, 'async_add_entry', new_callable=AsyncMock):
        await async_setup_entry(fake_hass, mock_home_config_entry)
        
        # Check that async_on_unload was called (update listener should be registered)
        assert len(mock_home_config_entry._on_unload_callbacks) > 0


@pytest.mark.asyncio
async def test_async_unload_entry_removes_device(fake_hass, mock_config_entry):
    """Test unload entry removes device from home."""
    mock_device = create_mock_device("test_device", platforms=[Platform.SENSOR])
    mock_home = MagicMock()
    mock_device.home = mock_home
    mock_device.get_platforms.return_value = [Platform.SENSOR]
    
    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device
    fake_hass.data[DOMAIN][DATA_HANDLER] = MagicMock()
    
    with patch.object(fake_hass.config_entries, 'async_unload_platforms', return_value=True):
        result = await async_unload_entry(fake_hass, mock_config_entry)
        
        assert result is True
        mock_home.remove_device.assert_called_once_with(mock_device)
        assert mock_config_entry.entry_id not in fake_hass.data[DOMAIN]


@pytest.mark.asyncio
async def test_async_unload_entry_no_device(fake_hass, mock_config_entry):
    """Test unload entry when device doesn't exist."""
    result = await async_unload_entry(fake_hass, mock_config_entry)
    
    assert result is False


@pytest.mark.asyncio
async def test_async_reload_entry(fake_hass, mock_config_entry):
    """Test reload entry."""
    with patch.object(fake_hass.config_entries, 'async_reload', new_callable=AsyncMock) as mock_reload:
        await async_reload_entry(fake_hass, mock_config_entry)
        
        mock_reload.assert_called_once_with(mock_config_entry.entry_id)


@pytest.mark.asyncio
async def test_async_reload_quiet_solar_all_entries(fake_hass):
    """Test reloading all quiet solar entries."""
    entry1 = MagicMock()
    entry1.entry_id = "entry1"
    entry2 = MagicMock()
    entry2.entry_id = "entry2"
    
    fake_hass.data[DOMAIN] = {
        "entry1": entry1,
        "entry2": entry2,
    }
    
    with patch.object(fake_hass.config_entries, 'async_entries', return_value=[entry1, entry2]):
        with patch.object(fake_hass.config_entries, 'async_unload', new_callable=AsyncMock) as mock_unload:
            with patch.object(fake_hass.config_entries, 'async_reload', new_callable=AsyncMock) as mock_reload:
                await async_reload_quiet_solar(fake_hass)
                
                assert mock_unload.call_count == 2
                assert mock_reload.call_count == 2


@pytest.mark.asyncio
async def test_async_reload_quiet_solar_except_one(fake_hass):
    """Test reloading entries except specified one."""
    entry1 = MagicMock()
    entry1.entry_id = "entry1"
    entry2 = MagicMock()
    entry2.entry_id = "entry2"
    
    with patch.object(fake_hass.config_entries, 'async_entries', return_value=[entry1, entry2]):
        with patch.object(fake_hass.config_entries, 'async_unload', new_callable=AsyncMock) as mock_unload:
            with patch.object(fake_hass.config_entries, 'async_reload', new_callable=AsyncMock) as mock_reload:
                await async_reload_quiet_solar(fake_hass, except_for_entry_id="entry1")
                
                # Only entry2 should be reloaded
                assert mock_unload.call_count == 1
                mock_unload.assert_called_with("entry2")


@pytest.mark.asyncio
async def test_async_reload_quiet_solar_handles_errors(fake_hass):
    """Test that reload continues even if individual entries fail."""
    entry1 = MagicMock()
    entry1.entry_id = "entry1"
    entry2 = MagicMock()
    entry2.entry_id = "entry2"
    
    with patch.object(fake_hass.config_entries, 'async_entries', return_value=[entry1, entry2]):
        with patch.object(fake_hass.config_entries, 'async_unload', side_effect=[Exception("Test error"), None]) as mock_unload:
            with patch.object(fake_hass.config_entries, 'async_reload', new_callable=AsyncMock) as mock_reload:
                # Should not raise exception
                await async_reload_quiet_solar(fake_hass)
                
                assert mock_unload.call_count == 2
                assert mock_reload.call_count == 2


@pytest.mark.asyncio
async def test_ocpp_notification_listener_filters_non_ocpp(fake_hass):
    """Test OCPP listener ignores non-OCPP notifications."""
    mock_charger = MagicMock()
    mock_charger.handle_ocpp_notification = AsyncMock()
    mock_home = MagicMock()
    mock_home._chargers = [mock_charger]
    fake_hass.data[DOMAIN][DATA_HANDLER] = MagicMock(home=mock_home)
    
    register_ocpp_notification_listener(fake_hass)
    
    # Fire a non-OCPP notification
    await fake_hass.bus.async_fire("call_service", {
        "domain": "other_domain",
        "service": "create",
    })
    
    mock_charger.handle_ocpp_notification.assert_not_awaited()


@pytest.mark.asyncio
async def test_ocpp_notification_forwards_to_chargers(fake_hass):
    """Test OCPP notification is forwarded to OCPP chargers."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerOCPP
    
    mock_home = MagicMock()
    mock_charger = MagicMock(spec=QSChargerOCPP)
    mock_charger.handle_ocpp_notification = AsyncMock()
    mock_home._chargers = [mock_charger]
    
    mock_handler = MagicMock()
    mock_handler.home = mock_home
    fake_hass.data[DOMAIN][DATA_HANDLER] = mock_handler
    
    with patch('custom_components.quiet_solar._is_notification_for_charger', return_value=True):
        register_ocpp_notification_listener(fake_hass)
        
        # Fire OCPP notification
        await fake_hass.bus.async_fire("call_service", {
            "domain": "persistent_notification",
            "service": "create",
            "service_data": {
                "title": "OCPP Alert",
                "message": "Charger status changed",
            }
        })
        
        # Charger should receive notification
        await asyncio.sleep(0.1)  # Give async tasks time to complete
        mock_charger.handle_ocpp_notification.assert_awaited_once_with(
            "Charger status changed",
            "OCPP Alert",
        )
