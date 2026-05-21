"""Config flow tests that actually setup integration.

This test file uses real config flow to create entries and verifies
that the resulting integration works correctly with real HA.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components.climate import HVACMode
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    CONF_ACCURATE_POWER_SENSOR,
    CONF_BATTERY_CAPACITY,
    CONF_BATTERY_MAX_CHARGE_PERCENT,
    CONF_BATTERY_MAX_CHARGE_POWER_VALUE,
    CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE,
    CONF_BATTERY_MIN_CHARGE_PERCENT,
    CONF_CHARGER_DEVICE_OCPP,
    CONF_CHARGER_DEVICE_WALLBOX,
    CONF_CHARGER_LATITUDE,
    CONF_CHARGER_LONGITUDE,
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER,
    CONF_CHARGER_PAUSE_RESUME_SWITCH,
    CONF_CHARGER_PLUGGED,
    CONF_CLIMATE,
    CONF_CLIMATE_HVAC_MODE_OFF,
    CONF_CLIMATE_HVAC_MODE_ON,
    CONF_HOME_END_OFF_PEAK_RANGE_1,
    CONF_HOME_END_OFF_PEAK_RANGE_2,
    CONF_HOME_OFF_PEAK_PRICE,
    CONF_HOME_PEAK_PRICE,
    CONF_HOME_START_OFF_PEAK_RANGE_1,
    CONF_HOME_START_OFF_PEAK_RANGE_2,
    CONF_HOME_VOLTAGE,
    CONF_IS_3P,
    CONF_POWER,
    CONF_SOLAR_FORECAST_PROVIDERS,
    CONF_SWITCH,
    DEVICE_TYPE,
    DOMAIN,
    POOL_TEMP_STEPS,
    SOLCAST_SOLAR_DOMAIN,
    CONF_TYPE_NAME_QSBattery,
    CONF_TYPE_NAME_QSChargerGeneric,
    CONF_TYPE_NAME_QSChargerOCPP,
    CONF_TYPE_NAME_QSChargerWallbox,
    CONF_TYPE_NAME_QSClimateDuration,
    CONF_TYPE_NAME_QSDynamicGroup,
    CONF_TYPE_NAME_QSHome,
    CONF_TYPE_NAME_QSOnOffDuration,
    CONF_TYPE_NAME_QSPool,
    CONF_TYPE_NAME_QSRadiator,
    CONF_TYPE_NAME_QSSolar,
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


async def _create_home_entry(hass: HomeAssistant, name: str = "Test Home"):
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSHome})
    result = await hass.config_entries.flow.async_configure(result["flow_id"], _home_user_input(name))
    assert result["type"] == FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    return result["result"]


async def _start_flow_to_step(hass: HomeAssistant, step_id: str):
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    assert result["type"] == FlowResultType.MENU
    return await hass.config_entries.flow.async_configure(result["flow_id"], {"next_step_id": step_id})


async def test_config_flow_creates_working_home(
    hass: HomeAssistant,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test that config flow creates a functioning home device."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})

    assert result["type"] == FlowResultType.MENU
    assert result["menu_options"] == [CONF_TYPE_NAME_QSHome]

    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSHome})

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == CONF_TYPE_NAME_QSHome

    result = await hass.config_entries.flow.async_configure(result["flow_id"], _home_user_input("Test Home"))

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
    home_flow = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    home_flow = await hass.config_entries.flow.async_configure(
        home_flow["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSHome}
    )
    result = await hass.config_entries.flow.async_configure(home_flow["flow_id"], _home_user_input("Test Home"))
    assert result["type"] == FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()

    hass.states.async_set("sensor.test_power", "100", {"unit_of_measurement": "W"})

    charger_flow = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    assert charger_flow["type"] == FlowResultType.MENU
    assert CONF_TYPE_NAME_QSHome not in charger_flow["menu_options"]

    charger_flow = await hass.config_entries.flow.async_configure(charger_flow["flow_id"], {"next_step_id": "charger"})
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
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})

    # Should show menu to create home first
    assert result["type"] == FlowResultType.MENU
    assert result["menu_options"] == [CONF_TYPE_NAME_QSHome]


async def test_config_flow_unique_ids(
    hass: HomeAssistant,
) -> None:
    """Test that config flow creates unique IDs."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSHome})
    result = await hass.config_entries.flow.async_configure(result["flow_id"], _home_user_input("Test Home"))
    entry = result["result"]
    assert entry.unique_id == "Quiet Solar: Test Home home"


async def test_config_flow_data_cleanup(
    hass: HomeAssistant,
) -> None:
    """Test that config flow cleans up None values."""
    home_flow = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    home_flow = await hass.config_entries.flow.async_configure(
        home_flow["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSHome}
    )
    result = await hass.config_entries.flow.async_configure(home_flow["flow_id"], _home_user_input("Test Home"))
    assert result["type"] == FlowResultType.CREATE_ENTRY

    await hass.async_block_till_done()

    hass.states.async_set("sensor.test_power", "100", {"unit_of_measurement": "W"})

    charger_flow = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    charger_flow = await hass.config_entries.flow.async_configure(charger_flow["flow_id"], {"next_step_id": "charger"})
    charger_flow = await hass.config_entries.flow.async_configure(
        charger_flow["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSChargerGeneric}
    )
    result = await hass.config_entries.flow.async_configure(
        charger_flow["flow_id"], _charger_user_input("Test Charger", "sensor.test_power")
    )

    entry = result["result"]
    assert CONF_CHARGER_LATITUDE not in entry.data
    assert CONF_CHARGER_LONGITUDE not in entry.data


async def test_config_flow_solar_with_forecast_provider(
    hass: HomeAssistant,
) -> None:
    """Test solar flow with forecast provider selection."""
    await _create_home_entry(hass)

    hass.states.async_set("sensor.solar_power", "100", {"unit_of_measurement": "W"})
    hass.data[SOLCAST_SOLAR_DOMAIN] = {"entry": object()}

    solar_flow = await _start_flow_to_step(hass, CONF_TYPE_NAME_QSSolar)
    assert solar_flow["type"] == FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        solar_flow["flow_id"],
        {
            CONF_NAME: "Solar",
            CONF_SOLAR_FORECAST_PROVIDERS: [SOLCAST_SOLAR_DOMAIN],
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    providers = result["data"][CONF_SOLAR_FORECAST_PROVIDERS]
    assert len(providers) == 1
    assert providers[0]["provider_domain"] == SOLCAST_SOLAR_DOMAIN


async def test_config_flow_battery_creates_entry(
    hass: HomeAssistant,
) -> None:
    """Test battery flow creation."""
    await _create_home_entry(hass)

    battery_flow = await _start_flow_to_step(hass, CONF_TYPE_NAME_QSBattery)
    assert battery_flow["type"] == FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        battery_flow["flow_id"],
        {
            CONF_NAME: "Battery",
            CONF_BATTERY_CAPACITY: 12000,
            CONF_BATTERY_MIN_CHARGE_PERCENT: 10,
            CONF_BATTERY_MAX_CHARGE_PERCENT: 95,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 3000,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 3500,
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_BATTERY_CAPACITY] == 12000


async def test_config_flow_charger_ocpp_creates_entry(
    hass: HomeAssistant,
) -> None:
    """Test OCPP charger flow creation."""
    await _create_home_entry(hass)

    charger_flow = await _start_flow_to_step(hass, "charger")
    charger_flow = await hass.config_entries.flow.async_configure(
        charger_flow["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSChargerOCPP}
    )
    assert charger_flow["type"] == FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        charger_flow["flow_id"],
        {
            CONF_NAME: "OCPP Charger",
            CONF_CHARGER_DEVICE_OCPP: "device_ocpp_1",
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_CHARGER_DEVICE_OCPP] == "device_ocpp_1"


async def test_config_flow_charger_wallbox_creates_entry(
    hass: HomeAssistant,
) -> None:
    """Test Wallbox charger flow creation."""
    await _create_home_entry(hass)

    charger_flow = await _start_flow_to_step(hass, "charger")
    charger_flow = await hass.config_entries.flow.async_configure(
        charger_flow["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSChargerWallbox}
    )
    assert charger_flow["type"] == FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        charger_flow["flow_id"],
        {
            CONF_NAME: "Wallbox Charger",
            CONF_CHARGER_DEVICE_WALLBOX: "device_wallbox_1",
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_CHARGER_DEVICE_WALLBOX] == "device_wallbox_1"


async def test_config_flow_pool_defaults(
    hass: HomeAssistant,
) -> None:
    """Test pool flow fills default temperature steps."""
    await _create_home_entry(hass)

    pool_flow = await _start_flow_to_step(hass, CONF_TYPE_NAME_QSPool)
    assert pool_flow["type"] == FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        pool_flow["flow_id"],
        {
            CONF_NAME: "Pool",
            CONF_POWER: 1500,
            CONF_SWITCH: "switch.pool",
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    for _min_temp, max_temp, default in POOL_TEMP_STEPS:
        assert result["data"][f"water_temp_{max_temp}"] == default


async def test_config_flow_on_off_duration_creates_entry(
    hass: HomeAssistant,
) -> None:
    """Test on/off duration flow creation."""
    await _create_home_entry(hass)

    duration_flow = await _start_flow_to_step(hass, CONF_TYPE_NAME_QSOnOffDuration)
    assert duration_flow["type"] == FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        duration_flow["flow_id"],
        {
            CONF_NAME: "Heater",
            CONF_POWER: 1000,
            CONF_SWITCH: "switch.heater",
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY


async def test_config_flow_climate_forces_hvac_modes(
    hass: HomeAssistant,
) -> None:
    """Test climate flow forces HVAC mode selection."""
    await _create_home_entry(hass)

    climate_flow = await _start_flow_to_step(hass, CONF_TYPE_NAME_QSClimateDuration)
    assert climate_flow["type"] == FlowResultType.FORM

    with patch(
        "custom_components.quiet_solar.config_flow.get_hvac_modes",
        return_value=[HVACMode.OFF, HVACMode.HEAT],
    ):
        climate_flow = await hass.config_entries.flow.async_configure(
            climate_flow["flow_id"],
            {
                CONF_NAME: "Climate",
                CONF_POWER: 1000,
                CONF_CLIMATE: "climate.living_room",
            },
        )
        assert climate_flow["type"] == FlowResultType.FORM

        result = await hass.config_entries.flow.async_configure(
            climate_flow["flow_id"],
            {
                CONF_NAME: "Climate",
                CONF_POWER: 1000,
                CONF_CLIMATE: "climate.living_room",
                CONF_CLIMATE_HVAC_MODE_OFF: HVACMode.OFF,
                CONF_CLIMATE_HVAC_MODE_ON: HVACMode.HEAT,
            },
        )
    assert result["type"] == FlowResultType.CREATE_ENTRY


async def test_config_flow_radiator_switch_creates_entry(
    hass: HomeAssistant,
) -> None:
    """AC-6 — switch-backed radiator happy path."""
    await _create_home_entry(hass)

    radiator_flow = await _start_flow_to_step(hass, CONF_TYPE_NAME_QSRadiator)
    assert radiator_flow["type"] == FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        radiator_flow["flow_id"],
        {
            CONF_NAME: "Bathroom Radiator",
            CONF_POWER: 800,
            CONF_SWITCH: "switch.bathroom_radiator",
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_SWITCH] == "switch.bathroom_radiator"


async def test_config_flow_radiator_climate_creates_entry(
    hass: HomeAssistant,
) -> None:
    """AC-6 — climate-backed radiator happy path (two-pass HVAC selection)."""
    await _create_home_entry(hass)

    radiator_flow = await _start_flow_to_step(hass, CONF_TYPE_NAME_QSRadiator)
    assert radiator_flow["type"] == FlowResultType.FORM

    with patch(
        "custom_components.quiet_solar.config_flow.get_hvac_modes",
        return_value=[HVACMode.OFF, HVACMode.HEAT, HVACMode.AUTO],
    ):
        # Pass 1 — just the climate entity (HVAC dropdowns not yet shown)
        radiator_flow = await hass.config_entries.flow.async_configure(
            radiator_flow["flow_id"],
            {
                CONF_NAME: "Living Room Radiator",
                CONF_POWER: 1500,
                CONF_CLIMATE: "climate.living_room",
            },
        )
        assert radiator_flow["type"] == FlowResultType.FORM

        # Pass 2 — HVAC dropdowns now present, pick `heat` / `off`
        result = await hass.config_entries.flow.async_configure(
            radiator_flow["flow_id"],
            {
                CONF_NAME: "Living Room Radiator",
                CONF_POWER: 1500,
                CONF_CLIMATE: "climate.living_room",
                CONF_CLIMATE_HVAC_MODE_OFF: HVACMode.OFF,
                CONF_CLIMATE_HVAC_MODE_ON: HVACMode.HEAT,
            },
        )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_CLIMATE] == "climate.living_room"
    assert result["data"][CONF_CLIMATE_HVAC_MODE_ON] == HVACMode.HEAT


async def test_config_flow_radiator_both_backings_rejected(
    hass: HomeAssistant,
) -> None:
    """AC-6 — submitting BOTH switch and climate triggers the XOR error."""
    await _create_home_entry(hass)

    radiator_flow = await _start_flow_to_step(hass, CONF_TYPE_NAME_QSRadiator)
    assert radiator_flow["type"] == FlowResultType.FORM

    with patch(
        "custom_components.quiet_solar.config_flow.get_hvac_modes",
        return_value=[HVACMode.OFF, HVACMode.HEAT],
    ):
        result = await hass.config_entries.flow.async_configure(
            radiator_flow["flow_id"],
            {
                CONF_NAME: "Misconfigured Radiator",
                CONF_POWER: 1000,
                CONF_SWITCH: "switch.r",
                CONF_CLIMATE: "climate.r",
            },
        )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {"base": "exactly_one_backing_required"}


async def test_config_flow_radiator_neither_backing_rejected(
    hass: HomeAssistant,
) -> None:
    """AC-6 — submitting NEITHER switch nor climate triggers the XOR error."""
    await _create_home_entry(hass)

    radiator_flow = await _start_flow_to_step(hass, CONF_TYPE_NAME_QSRadiator)
    assert radiator_flow["type"] == FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        radiator_flow["flow_id"],
        {
            CONF_NAME: "Backingless Radiator",
            CONF_POWER: 1000,
        },
    )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {"base": "exactly_one_backing_required"}


async def test_config_flow_radiator_options_swap_to_climate(
    hass: HomeAssistant,
) -> None:
    """AC-13 — start with switch backing, then swap to climate via options flow.

    The reload-on-options-update path re-instantiates the QSRadiator with
    the new transport — `device_id` is stable thanks to the slug-based
    identity, so persisted constraint history survives the swap.
    """
    await _create_home_entry(hass)

    radiator_flow = await _start_flow_to_step(hass, CONF_TYPE_NAME_QSRadiator)
    initial = await hass.config_entries.flow.async_configure(
        radiator_flow["flow_id"],
        {
            CONF_NAME: "Swap Radiator",
            CONF_POWER: 1000,
            CONF_SWITCH: "switch.swap",
        },
    )
    assert initial["type"] == FlowResultType.CREATE_ENTRY
    entry = initial["result"]
    assert entry.data[CONF_SWITCH] == "switch.swap"
    assert CONF_CLIMATE not in entry.data or entry.data.get(CONF_CLIMATE) is None

    # Now swap to climate via options flow.
    with (
        patch(
            "custom_components.quiet_solar.config_flow.async_reload_quiet_solar",
            new_callable=AsyncMock,
        ) as mock_reload,
        patch(
            "custom_components.quiet_solar.config_flow.get_hvac_modes",
            return_value=[HVACMode.OFF, HVACMode.HEAT],
        ),
    ):
        result = await hass.config_entries.options.async_init(entry.entry_id)
        # Pass 1 — clear the switch, set the climate
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            {
                CONF_NAME: "Swap Radiator",
                CONF_POWER: 1000,
                CONF_CLIMATE: "climate.swap",
            },
        )
        assert result["type"] == FlowResultType.FORM
        # Pass 2 — provide HVAC modes
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            {
                CONF_NAME: "Swap Radiator",
                CONF_POWER: 1000,
                CONF_CLIMATE: "climate.swap",
                CONF_CLIMATE_HVAC_MODE_OFF: HVACMode.OFF,
                CONF_CLIMATE_HVAC_MODE_ON: HVACMode.HEAT,
            },
        )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert entry.data.get(CONF_CLIMATE) == "climate.swap"
    # N13 + CR1 — after the swap, the old `switch` backing must be
    # cleared. `not in` is stricter than `is None` (the latter passes
    # whether the key is absent OR present with value `None`; the
    # latter is exactly the stale-key shape we want to rule out).
    assert CONF_SWITCH not in entry.data
    mock_reload.assert_called_once()


async def test_config_flow_dynamic_group_creates_entry(
    hass: HomeAssistant,
) -> None:
    """Test dynamic group flow creation."""
    await _create_home_entry(hass)

    group_flow = await _start_flow_to_step(hass, CONF_TYPE_NAME_QSDynamicGroup)
    assert group_flow["type"] == FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        group_flow["flow_id"],
        {CONF_NAME: "Group"},
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY


async def test_options_flow_updates_entry_and_reloads(
    hass: HomeAssistant,
) -> None:
    """Test options flow updates entry and reloads integration."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_NAME: "Options Home",
            DEVICE_TYPE: CONF_TYPE_NAME_QSHome,
            CONF_HOME_VOLTAGE: 230,
            CONF_IS_3P: True,
        },
        title="home: Options Home",
    )
    entry.add_to_hass(hass)

    with patch(
        "custom_components.quiet_solar.config_flow.async_reload_quiet_solar",
        new_callable=AsyncMock,
    ) as mock_reload:
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            _home_user_input("Updated Home"),
        )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert entry.data[CONF_NAME] == "Updated Home"
    assert entry.title == "home: Updated Home"
    mock_reload.assert_called_once()


async def test_options_flow_with_missing_device_type(
    hass: HomeAssistant,
) -> None:
    """Test options flow exits when device type is missing."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_NAME: "No Type"},
        title="home: No Type",
    )
    entry.add_to_hass(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] == FlowResultType.CREATE_ENTRY


async def test_config_flow_entry_title_format(
    hass: HomeAssistant,
) -> None:
    """Test that entry titles are formatted correctly."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"next_step_id": CONF_TYPE_NAME_QSHome})
    result = await hass.config_entries.flow.async_configure(result["flow_id"], _home_user_input("Test Home"))
    assert result["title"] == "home: Test Home"
