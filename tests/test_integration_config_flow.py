"""Tests for quiet_solar config flow."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import FlowResultType

from custom_components.quiet_solar import __init__ as qs_init
from custom_components.quiet_solar.config_flow import (
    QSFlowHandler,
    QSOptionsFlowHandler,
    LOAD_TYPES_MENU,
)
from custom_components.quiet_solar.const import (
    DOMAIN,
    DEVICE_TYPE,
    CONF_HOME_VOLTAGE,
    CONF_IS_3P,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CAR_BATTERY_CAPACITY,
    CONF_GRID_POWER_SENSOR,
    DATA_HANDLER,
)
from custom_components.quiet_solar.ha_model.home import QSHome
from custom_components.quiet_solar.ha_model.battery import QSBattery
from custom_components.quiet_solar.ha_model.solar import QSSolar
from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric
from custom_components.quiet_solar.ha_model.car import QSCar


@pytest.mark.asyncio
async def test_flow_user_init_no_home(fake_hass, mock_data_handler):
    """Test user flow when no home exists - should only show home option."""
    mock_data_handler.home = None
    
    flow = QSFlowHandler()
    flow.hass = fake_hass
    
    result = await flow.async_step_user()
    
    assert result["type"] == FlowResultType.MENU
    assert result["step_id"] == "user"
    assert QSHome.conf_type_name in result["menu_options"]
    assert len(result["menu_options"]) == 1


@pytest.mark.asyncio
async def test_flow_user_init_with_home(fake_hass, mock_data_handler):
    """Test user flow when home exists - should show all device types except home."""
    mock_home = MagicMock()
    mock_home._battery = None
    mock_home._solar_plant = None
    mock_data_handler.home = mock_home
    
    flow = QSFlowHandler()
    flow.hass = fake_hass
    
    result = await flow.async_step_user()
    
    assert result["type"] == FlowResultType.MENU
    assert result["step_id"] == "user"
    assert QSHome.conf_type_name not in result["menu_options"]
    assert "charger" in result["menu_options"]  # Charger is a submenu
    assert QSCar.conf_type_name in result["menu_options"]


@pytest.mark.asyncio
async def test_flow_user_with_battery_installed(fake_hass, mock_data_handler):
    """Test that battery option is hidden when battery already exists."""
    mock_home = MagicMock()
    mock_home._battery = MagicMock()  # Battery installed
    mock_home._solar_plant = None
    mock_data_handler.home = mock_home
    
    flow = QSFlowHandler()
    flow.hass = fake_hass
    
    result = await flow.async_step_user()
    
    assert QSBattery.conf_type_name not in result["menu_options"]


@pytest.mark.asyncio
async def test_flow_user_with_solar_installed(fake_hass, mock_data_handler):
    """Test that solar option is hidden when solar already exists."""
    mock_home = MagicMock()
    mock_home._battery = None
    mock_home._solar_plant = MagicMock()  # Solar installed
    mock_data_handler.home = mock_home
    
    flow = QSFlowHandler()
    flow.hass = fake_hass
    
    result = await flow.async_step_user()
    
    assert QSSolar.conf_type_name not in result["menu_options"]


@pytest.mark.asyncio
async def test_flow_home_step_shows_form(fake_hass):
    """Test home configuration step shows form."""
    flow = QSFlowHandler()
    flow.hass = fake_hass
    
    result = await flow.async_step_home()
    
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == QSHome.conf_type_name


@pytest.mark.asyncio
async def test_flow_home_step_creates_entry(fake_hass):
    """Test home configuration creates entry with correct data."""
    flow = QSFlowHandler()
    flow.hass = fake_hass
    
    user_input = {
        CONF_NAME: "My Home",
        CONF_HOME_VOLTAGE: 230,
        CONF_IS_3P: True,
    }
    
    with patch.object(flow, 'async_set_unique_id', new_callable=AsyncMock):
        result = await flow.async_step_home(user_input)
    
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "home: My Home"
    assert result["data"][CONF_NAME] == "My Home"
    assert result["data"][CONF_HOME_VOLTAGE] == 230
    assert result["data"][DEVICE_TYPE] == QSHome.conf_type_name


@pytest.mark.asyncio
async def test_flow_charger_menu(fake_hass):
    """Test charger submenu shows charger types."""
    flow = QSFlowHandler()
    flow.hass = fake_hass
    
    result = await flow.async_step_charger()
    
    assert result["type"] == FlowResultType.MENU
    assert result["step_id"] == "charger"
    assert len(result["menu_options"]) > 0


@pytest.mark.asyncio
async def test_flow_charger_generic_creates_entry(fake_hass, mock_data_handler):
    """Test generic charger configuration creates entry."""
    mock_data_handler.home = MagicMock()
    mock_data_handler.home._all_dynamic_groups = [MagicMock()]
    
    flow = QSFlowHandler()
    flow.hass = fake_hass
    
    user_input = {
        CONF_NAME: "Test Charger",
        CONF_CHARGER_MIN_CHARGE: 6,
        CONF_CHARGER_MAX_CHARGE: 16,
        CONF_IS_3P: False,
    }
    
    with patch.object(flow, 'async_set_unique_id', new_callable=AsyncMock):
        result = await flow.async_step_charger_generic(user_input)
    
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_NAME] == "Test Charger"
    assert result["data"][CONF_CHARGER_MIN_CHARGE] == 6


@pytest.mark.asyncio
async def test_flow_car_creates_entry(fake_hass, mock_data_handler):
    """Test car configuration creates entry."""
    mock_data_handler.home = MagicMock()
    
    flow = QSFlowHandler()
    flow.hass = fake_hass
    
    user_input = {
        CONF_NAME: "Test Car",
        CONF_CAR_BATTERY_CAPACITY: 50000,
    }
    
    with patch.object(flow, 'async_set_unique_id', new_callable=AsyncMock):
        result = await flow.async_step_car(user_input)
    
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_NAME] == "Test Car"
    assert result["data"][CONF_CAR_BATTERY_CAPACITY] == 50000


@pytest.mark.asyncio
async def test_flow_cleans_none_values(fake_hass):
    """Test that None values are cleaned from data."""
    flow = QSFlowHandler()
    flow.hass = fake_hass
    
    data = {
        CONF_NAME: "Test",
        CONF_GRID_POWER_SENSOR: None,
        CONF_IS_3P: True,
    }
    
    flow.clean_data(data)
    
    assert CONF_NAME in data
    assert CONF_IS_3P in data
    assert CONF_GRID_POWER_SENSOR not in data


@pytest.mark.asyncio
async def test_options_flow_is_creation_flow(fake_hass, mock_config_entry):
    """Test options flow is_creation_flow returns False."""
    from awesomeversion import AwesomeVersion
    from homeassistant.const import __version__ as HAVERSION
    from custom_components.quiet_solar.config_flow import HA_OPTIONS_FLOW_VERSION_THRESHOLD
    
    # Set up the handler property for newer HA versions
    flow = QSOptionsFlowHandler(mock_config_entry)
    
    # For newer HA versions, we need to set the handler manually
    if AwesomeVersion(HAVERSION) >= HA_OPTIONS_FLOW_VERSION_THRESHOLD:
        flow.handler = mock_config_entry.entry_id
        # Register the entry in FakeHass so async_get_known_entry can find it
        fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_config_entry
    
    flow.hass = fake_hass
    
    # Test the method directly
    assert flow.is_creation_flow() is False


@pytest.mark.asyncio
async def test_get_entry_title_formats_correctly(fake_hass):
    """Test entry title formatting."""
    flow = QSFlowHandler()
    flow.hass = fake_hass
    
    data = {
        CONF_NAME: "My Device",
        DEVICE_TYPE: QSHome.conf_type_name
    }
    
    title = flow.get_entry_title(data)
    
    assert "My Device" in title
    assert "home" in title.lower()


@pytest.mark.asyncio
async def test_flow_unique_id_set(fake_hass):
    """Test that unique ID is set for new entries."""
    flow = QSFlowHandler()
    flow.hass = fake_hass
    
    user_input = {
        CONF_NAME: "Unique Device",
        CONF_IS_3P: True,
    }
    
    with patch.object(flow, 'async_set_unique_id', new_callable=AsyncMock) as mock_set_unique:
        with patch.object(flow, 'async_create_entry', return_value={"type": FlowResultType.CREATE_ENTRY}):
            await flow._async_entry_next({**user_input, DEVICE_TYPE: QSHome.conf_type_name})
    
    mock_set_unique.assert_called_once()
    call_arg = mock_set_unique.call_args[0][0]
    assert "Unique Device" in call_arg
    assert QSHome.conf_type_name in call_arg
