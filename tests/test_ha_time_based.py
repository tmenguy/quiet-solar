"""Time-based tests with real HA.

This test file verifies time-based coordinator updates, intervals,
and scheduling behavior using real Home Assistant time control.
"""
from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, patch

import pytest
from freezegun.api import FrozenDateTimeFactory
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.quiet_solar.const import DATA_HANDLER, DOMAIN


pytestmark = pytest.mark.asyncio


async def test_load_update_interval(
    hass: HomeAssistant,
    real_home_config_entry,
) -> None:
    """Test that load updates fire at correct intervals (7 seconds)."""
    from pytest_homeassistant_custom_component.common import async_fire_time_changed
    
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_loads", new_callable=AsyncMock) as mock_update_loads:
            await hass.config_entries.async_setup(config_entry.entry_id)
            await hass.async_block_till_done()
            
            # Get data handler
            data_handler = hass.data[DOMAIN][DATA_HANDLER]
            assert data_handler is not None
            
            # Initial call count
            initial_count = mock_update_loads.call_count
            
            # Advance time by 7 seconds (load update interval)
            future = dt_util.utcnow() + datetime.timedelta(seconds=7)
            async_fire_time_changed(hass, future)
            await hass.async_block_till_done()
            
            # Update should have been called at least once more
            assert mock_update_loads.call_count > initial_count


async def test_state_update_interval(
    hass: HomeAssistant,
    real_home_config_entry,
) -> None:
    """Test that state updates fire at correct intervals (4 seconds)."""
    from pytest_homeassistant_custom_component.common import async_fire_time_changed
    
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_all_states", new_callable=AsyncMock) as mock_update_states:
            await hass.config_entries.async_setup(config_entry.entry_id)
            await hass.async_block_till_done()
            
            # Get data handler
            data_handler = hass.data[DOMAIN][DATA_HANDLER]
            assert data_handler is not None
            
            # Initial call count
            initial_count = mock_update_states.call_count
            
            # Advance time by 4 seconds (state update interval)
            future = dt_util.utcnow() + datetime.timedelta(seconds=4)
            async_fire_time_changed(hass, future)
            await hass.async_block_till_done()
            
            # Update should have been called at least once more
            assert mock_update_states.call_count > initial_count


async def test_forecast_update_interval(
    hass: HomeAssistant,
    real_home_config_entry,
) -> None:
    """Test that forecast probers update at correct intervals (30 seconds)."""
    from pytest_homeassistant_custom_component.common import async_fire_time_changed
    
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock) as mock_update_forecast:
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
        
        # Get data handler
        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        assert data_handler is not None
        
        # Initial call count
        initial_count = mock_update_forecast.call_count
        
        # Advance time by 30 seconds (forecast update interval)
        future = dt_util.utcnow() + datetime.timedelta(seconds=30)
        async_fire_time_changed(hass, future)
        await hass.async_block_till_done()
        
        # Update should have been called at least once more
        assert mock_update_forecast.call_count > initial_count


async def test_multiple_interval_updates(
    hass: HomeAssistant,
    real_home_config_entry,
) -> None:
    """Test that all intervals work together correctly."""
    from pytest_homeassistant_custom_component.common import async_fire_time_changed
    
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock) as mock_forecast:
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_loads", new_callable=AsyncMock) as mock_loads:
            with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_all_states", new_callable=AsyncMock) as mock_states:
                await hass.config_entries.async_setup(config_entry.entry_id)
                await hass.async_block_till_done()
                
                # Track initial counts
                initial_loads = mock_loads.call_count
                initial_states = mock_states.call_count
                initial_forecast = mock_forecast.call_count
                
                # Advance time by 30 seconds (covers all intervals)
                future = dt_util.utcnow() + datetime.timedelta(seconds=30)
                async_fire_time_changed(hass, future)
                await hass.async_block_till_done()
                
                # All should have been called
                assert mock_loads.call_count > initial_loads
                assert mock_states.call_count > initial_states
                assert mock_forecast.call_count > initial_forecast


async def test_time_not_advanced_no_updates(
    hass: HomeAssistant,
    real_home_config_entry,
) -> None:
    """Test that updates don't fire if time doesn't advance."""
    from pytest_homeassistant_custom_component.common import async_fire_time_changed
    
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_loads", new_callable=AsyncMock) as mock_update_loads:
            await hass.config_entries.async_setup(config_entry.entry_id)
            await hass.async_block_till_done()
            
            # Get initial call count
            initial_count = mock_update_loads.call_count
            
            # Advance time by only 1 second (less than interval)
            future = dt_util.utcnow() + datetime.timedelta(seconds=1)
            async_fire_time_changed(hass, future)
            await hass.async_block_till_done()
            
            # Update should not have been called again
            assert mock_update_loads.call_count == initial_count


async def test_unload_stops_updates(
    hass: HomeAssistant,
    real_home_config_entry,
) -> None:
    """Test that unloading entry stops periodic updates."""
    from pytest_homeassistant_custom_component.common import async_fire_time_changed
    
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_loads", new_callable=AsyncMock) as mock_update_loads:
            # Setup
            await hass.config_entries.async_setup(config_entry.entry_id)
            await hass.async_block_till_done()
            
            # Unload
            unload_ok = await hass.config_entries.async_unload(config_entry.entry_id)
            await hass.async_block_till_done()
            assert unload_ok is True
            
            # Get call count after unload
            count_after_unload = mock_update_loads.call_count
            
            # Try to advance time
            future = dt_util.utcnow() + datetime.timedelta(seconds=10)
            async_fire_time_changed(hass, future)
            await hass.async_block_till_done()
            
            # Update should not have been called (unload should cancel timers)
            assert mock_update_loads.call_count == count_after_unload


async def test_data_handler_intervals_configurable(
    hass: HomeAssistant,
    real_home_config_entry,
) -> None:
    """Test that data handler intervals are set correctly."""
    config_entry = real_home_config_entry
    
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
        
        # Get data handler
        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        
        # Verify intervals are set
        assert data_handler._load_update_scan_interval == 7
        assert data_handler._refresh_states_interval == 4
        assert data_handler._refresh_forecast_probers_interval == 30


