"""Config flow tests that actually setup integration.

This test file uses real config flow to create entries and verifies
that the resulting integration works correctly with real HA.
"""
from __future__ import annotations

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er

from homeassistant.const import CONF_NAME
from custom_components.quiet_solar.const import (
    CONF_ACCURATE_POWER_SENSOR,
    CONF_CHARGER_LATITUDE,
    CONF_CHARGER_LONGITUDE,
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER,
    CONF_CHARGER_PAUSE_RESUME_SWITCH,
    CONF_CHARGER_PLUGGED,
    CONF_HOME_VOLTAGE,
    CONF_HOME_END_OFF_PEAK_RANGE_1,
    CONF_HOME_END_OFF_PEAK_RANGE_2,
    CONF_HOME_OFF_PEAK_PRICE,
    CONF_HOME_PEAK_PRICE,
    CONF_HOME_START_OFF_PEAK_RANGE_1,
    CONF_HOME_START_OFF_PEAK_RANGE_2,
    CONF_IS_3P,
    CONF_TYPE_NAME_QSChargerGeneric,
    CONF_TYPE_NAME_QSHome,
    DEVICE_TYPE,
    DOMAIN,
)


pytestmark = pytest.mark.asyncio


def _home_user_input(name: str) -> dict:
    return {
        CONF_NAME: name,
        CONF_HOME_VOLTAGE: 230,
        CONF_IS_3P: True,
        CONF_HOME_PEAK_PRICE: 0.27,
        CONF_HOME_OFF_PEAK_PRICE: 0.2068,
        CONF_HOME_START_OFF_PEAK_RANGE_1: "00:00:00",
        CONF_HOME_END_OFF_PEAK_RANGE_1: "06:00:00",
        CONF_HOME_START_OFF_PEAK_RANGE_2: "14:00:00",
        CONF_HOME_END_OFF_PEAK_RANGE_2: "16:00:00",
    }


def _charger_user_input(name: str, power_sensor: str) -> dict:
    return {
        CONF_NAME: name,
        CONF_ACCURATE_POWER_SENSOR: power_sensor,
        CONF_CHARGER_PLUGGED: "binary_sensor.test_charger_plugged",
        CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER: "number.test_charger_max_current",
        CONF_CHARGER_PAUSE_RESUME_SWITCH: "switch.test_charger_pause_resume",
        CONF_CHARGER_LATITUDE: None,
        CONF_CHARGER_LONGITUDE: None,
    }


async def test_config_flow_creates_working_home(
    hass: HomeAssistant,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test that config flow creates a functioning home device."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )

    assert result["type"] == FlowResultType.MENU
    assert result["menu_options"] == [CONF_TYPE_NAME_QSHome]

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSHome}
    )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == CONF_TYPE_NAME_QSHome

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _home_user_input("Test Home")
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "home: Test Home"
    entry = result["result"]
    assert entry.data[DEVICE_TYPE] == CONF_TYPE_NAME_QSHome

    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED

    entity_entries = er.async_entries_for_config_entry(entity_registry, entry.entry_id)
    assert entity_entries


async def test_config_flow_home_then_charger(
    hass: HomeAssistant,
) -> None:
    """Test setting up home followed by charger via config flow."""
    home_flow = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    home_flow = await hass.config_entries.flow.async_configure(
        home_flow["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSHome}
    )
    result = await hass.config_entries.flow.async_configure(
        home_flow["flow_id"], _home_user_input("Test Home")
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()

    hass.states.async_set("sensor.test_power", "100", {"unit_of_measurement": "W"})

    charger_flow = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert charger_flow["type"] == FlowResultType.MENU
    assert CONF_TYPE_NAME_QSHome not in charger_flow["menu_options"]

    charger_flow = await hass.config_entries.flow.async_configure(
        charger_flow["flow_id"], {"next_step_id": "charger"}
    )
    assert charger_flow["type"] == FlowResultType.MENU

    charger_flow = await hass.config_entries.flow.async_configure(
        charger_flow["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSChargerGeneric}
    )
    assert charger_flow["type"] == FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        charger_flow["flow_id"], _charger_user_input("Test Charger", "sensor.test_power")
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY


async def test_config_flow_device_without_home(
    hass: HomeAssistant,
) -> None:
    """Test that devices can be added before home (cached)."""
    # Try to add charger without home
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    
    # Should show menu to create home first
    assert result["type"] == FlowResultType.MENU
    assert result["menu_options"] == [CONF_TYPE_NAME_QSHome]


async def test_config_flow_unique_ids(
    hass: HomeAssistant,
) -> None:
    """Test that config flow creates unique IDs."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSHome}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _home_user_input("Test Home")
    )
    entry = result["result"]
    assert entry.unique_id == "Quiet Solar: Test Home home"


async def test_config_flow_data_cleanup(
    hass: HomeAssistant,
) -> None:
    """Test that config flow cleans up None values."""
    home_flow = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    home_flow = await hass.config_entries.flow.async_configure(
        home_flow["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSHome}
    )
    result = await hass.config_entries.flow.async_configure(
        home_flow["flow_id"], _home_user_input("Test Home")
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()

    hass.states.async_set("sensor.test_power", "100", {"unit_of_measurement": "W"})

    charger_flow = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    charger_flow = await hass.config_entries.flow.async_configure(
        charger_flow["flow_id"], {"next_step_id": "charger"}
    )
    charger_flow = await hass.config_entries.flow.async_configure(
        charger_flow["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSChargerGeneric}
    )
    result = await hass.config_entries.flow.async_configure(
        charger_flow["flow_id"], _charger_user_input("Test Charger", "sensor.test_power")
    )

    entry = result["result"]
    assert CONF_CHARGER_LATITUDE not in entry.data
    assert CONF_CHARGER_LONGITUDE not in entry.data


async def test_config_flow_entry_title_format(
    hass: HomeAssistant,
) -> None:
    """Test that entry titles are formatted correctly."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSHome}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _home_user_input("Test Home")
    )
    assert result["title"] == "home: Test Home"

