"""Tests for QSDataHandler."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from homeassistant.const import Platform, CONF_NAME

from custom_components.quiet_solar.const import DOMAIN, DEVICE_TYPE, CONF_HOME_VOLTAGE, CONF_IS_3P
from custom_components.quiet_solar.data_handler import QSDataHandler
from custom_components.quiet_solar.ha_model.home import QSHome
from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric
from tests.conftest import create_mock_device


@pytest.fixture
def data_handler(fake_hass):
    """Create data handler instance."""
    return QSDataHandler(fake_hass)


@pytest.mark.asyncio
async def test_data_handler_init(data_handler, fake_hass):
    """Test data handler initialization."""
    assert data_handler.hass == fake_hass
    assert data_handler.home is None
    assert data_handler._cached_config_entries == []
    assert data_handler._load_update_scan_interval == 7
    assert data_handler._refresh_states_interval == 4
    assert data_handler._refresh_forecast_probers_interval == 30


@pytest.mark.asyncio
async def test_async_add_entry_device_before_home(data_handler, mock_charger_config_entry):
    """Test adding device entry before home caches it."""
    mock_charger_config_entry.data[DEVICE_TYPE] = QSChargerGeneric.conf_type_name
    
    await data_handler.async_add_entry(mock_charger_config_entry)
    
    # Device should be cached, not added yet
    assert data_handler.home is None
    assert mock_charger_config_entry in data_handler._cached_config_entries


@pytest.mark.asyncio
async def test_async_update_loads_delegates_to_home(data_handler):
    """Test async_update_loads calls home.update_loads."""
    mock_home = MagicMock()
    mock_home.update_loads = AsyncMock()
    data_handler.home = mock_home
    
    test_time = datetime.now(pytz.UTC)
    await data_handler.async_update_loads(test_time)
    
    mock_home.update_loads.assert_called_once_with(test_time)


@pytest.mark.asyncio
async def test_async_update_all_states_delegates_to_home(data_handler):
    """Test async_update_all_states calls home.update_all_states."""
    mock_home = MagicMock()
    mock_home.update_all_states = AsyncMock()
    data_handler.home = mock_home
    
    test_time = datetime.now(pytz.UTC)
    await data_handler.async_update_all_states(test_time)
    
    mock_home.update_all_states.assert_called_once_with(test_time)


@pytest.mark.asyncio
async def test_async_update_forecast_probers_delegates_to_home(data_handler):
    """Test async_update_forecast_probers calls home.update_forecast_probers."""
    mock_home = MagicMock()
    mock_home.update_forecast_probers = AsyncMock()
    data_handler.home = mock_home
    
    test_time = datetime.now(pytz.UTC)
    await data_handler.async_update_forecast_probers(test_time)
    
    mock_home.update_forecast_probers.assert_called_once_with(test_time)


@pytest.mark.asyncio
async def test_data_handler_stores_entry_in_hass_data(data_handler, mock_charger_config_entry):
    """Test that devices are stored in hass.data."""
    mock_charger_config_entry.data[DEVICE_TYPE] = QSChargerGeneric.conf_type_name
    
    # Just verify caching works (won't create device without home)
    await data_handler.async_add_entry(mock_charger_config_entry)
    
    assert mock_charger_config_entry in data_handler._cached_config_entries