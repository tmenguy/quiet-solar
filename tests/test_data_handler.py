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
from tests.test_helpers import create_mock_device


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


def test_add_device_handles_creation_failure(data_handler, mock_charger_config_entry):
    """Test _add_device handles creation failures."""
    with patch("custom_components.quiet_solar.data_handler.create_device_from_type", return_value=None):
        result = data_handler._add_device(mock_charger_config_entry)

    assert result is None


@pytest.mark.asyncio
async def test_async_add_entry_rehydrates_cached_entries(fake_hass):
    """Test cached entries are attached when home is added."""
    data_handler = QSDataHandler(fake_hass)
    cached_entry = MagicMock()
    cached_entry.entry_id = "cached_entry"
    cached_entry.data = {DEVICE_TYPE: QSChargerGeneric.conf_type_name}
    data_handler._cached_config_entries.append(cached_entry)

    home_entry = MagicMock()
    home_entry.entry_id = "home_entry"
    home_entry.data = {DEVICE_TYPE: QSHome.conf_type_name}
    home_entry.async_on_unload = MagicMock()

    home_device = MagicMock()
    home_device.add_device = MagicMock()
    home_device.get_platforms = MagicMock(return_value=[Platform.SENSOR])
    home_device.config_entry = home_entry
    home_device.config_entry_initialized = False

    cached_device = MagicMock()
    cached_device.get_platforms = MagicMock(return_value=[Platform.SENSOR])
    cached_device.config_entry = cached_entry
    cached_device.config_entry_initialized = False

    with patch(
        "custom_components.quiet_solar.data_handler.create_device_from_type",
        side_effect=[home_device, cached_device],
    ):
        await data_handler.async_add_entry(home_entry)

    assert data_handler.home is home_device
    assert cached_entry not in data_handler._cached_config_entries


@pytest.mark.asyncio
async def test_async_update_loads_skips_when_locked(data_handler):
    """Test async_update_loads skips when lock is held."""
    data_handler.home = MagicMock()
    data_handler.home.update_loads = AsyncMock()

    await data_handler._update_loads_lock.acquire()
    try:
        await data_handler.async_update_loads(datetime.now(pytz.UTC))
    finally:
        data_handler._update_loads_lock.release()

    data_handler.home.update_loads.assert_not_called()


@pytest.mark.asyncio
async def test_async_update_loads_logs_error(data_handler):
    """Test async_update_loads handles errors."""
    data_handler.home = MagicMock()
    data_handler.home.update_loads = AsyncMock(side_effect=RuntimeError("boom"))

    await data_handler.async_update_loads(datetime.now(pytz.UTC))

    data_handler.home.update_loads.assert_called_once()


@pytest.mark.asyncio
async def test_async_update_all_states_skips_when_locked(data_handler):
    """Test async_update_all_states skips when lock is held."""
    data_handler.home = MagicMock()
    data_handler.home.update_all_states = AsyncMock()

    await data_handler._update_all_states_lock.acquire()
    try:
        await data_handler.async_update_all_states(datetime.now(pytz.UTC))
    finally:
        data_handler._update_all_states_lock.release()

    data_handler.home.update_all_states.assert_not_called()


@pytest.mark.asyncio
async def test_async_update_all_states_logs_error(data_handler):
    """Test async_update_all_states handles errors."""
    data_handler.home = MagicMock()
    data_handler.home.update_all_states = AsyncMock(side_effect=RuntimeError("boom"))

    await data_handler.async_update_all_states(datetime.now(pytz.UTC))

    data_handler.home.update_all_states.assert_called_once()


@pytest.mark.asyncio
async def test_async_update_forecast_probers_skips_when_locked(data_handler):
    """Test async_update_forecast_probers skips when lock is held."""
    data_handler.home = MagicMock()
    data_handler.home.update_forecast_probers = AsyncMock()

    await data_handler._update_forecast_probers_lock.acquire()
    try:
        await data_handler.async_update_forecast_probers(datetime.now(pytz.UTC))
    finally:
        data_handler._update_forecast_probers_lock.release()

    data_handler.home.update_forecast_probers.assert_not_called()


@pytest.mark.asyncio
async def test_async_update_forecast_probers_logs_error(data_handler):
    """Test async_update_forecast_probers handles errors."""
    data_handler.home = MagicMock()
    data_handler.home.update_forecast_probers = AsyncMock(side_effect=RuntimeError("boom"))

    await data_handler.async_update_forecast_probers(datetime.now(pytz.UTC))

    data_handler.home.update_forecast_probers.assert_called_once()