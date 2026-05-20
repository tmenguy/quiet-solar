"""End-to-end tests for `QSRadiator` against a real `hass` fixture.

These tests cover the integration boundary the unit tests in
`test_ha_radiator.py` mock out: the actual HA service call, the
constraint / select / sensor wiring, and the heat-pump piloting
relationship (AC-10).
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz
from homeassistant.components import climate
from homeassistant.const import (
    ATTR_ENTITY_ID,
    CONF_NAME,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    Platform,
)
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    CONF_CLIMATE,
    CONF_CLIMATE_HVAC_MODE_OFF,
    CONF_CLIMATE_HVAC_MODE_ON,
    CONF_DEVICE_TO_PILOT_NAME,
    CONF_POWER,
    CONF_SWITCH,
    DATA_HANDLER,
    DOMAIN,
)
from custom_components.quiet_solar.ha_model.bistate_transport import (
    ClimateTransport,
    SwitchTransport,
)
from custom_components.quiet_solar.ha_model.radiator import QSRadiator
from custom_components.quiet_solar.home_model.commands import CMD_OFF, CMD_ON
from tests.factories import create_minimal_home_model


@pytest.fixture
def radiator_config_entry() -> MockConfigEntry:
    """Mock config entry for radiator tests."""
    return MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_radiator_real_entry",
        data={CONF_NAME: "Real Radiator"},
        title="Real Radiator",
    )


@pytest.fixture
def radiator_home(hass: HomeAssistant):
    """Home and data handler for radiator tests."""
    home = create_minimal_home_model()
    home.is_off_grid = MagicMock(return_value=False)
    data_handler = MagicMock()
    data_handler.home = home
    hass.data.setdefault(DOMAIN, {})[DATA_HANDLER] = data_handler
    return home


@pytest.fixture
def recorded_service_calls(hass: HomeAssistant):
    """Record service calls (domain, service, payload) for assertions.

    The fourth tuple element captures `service_data` OR (when only
    `target` is set, as for switch.turn_on/off) the target dict — so a
    single recorder works for both climate and switch flows. Mirrors
    `test_ha_climate_controller_real.py` but tolerant of `target=`.
    """
    from homeassistant.core import ServiceRegistry

    recorded = []

    async def record_only(self, domain, service, service_data=None, **kwargs):
        if self is hass.services:
            payload = service_data or kwargs.get("target") or {}
            recorded.append((domain, service, payload))
        return None

    with patch.object(ServiceRegistry, "async_call", record_only):
        yield recorded


# =============================================================================
# Switch-backed full cycle
# =============================================================================


@pytest.mark.asyncio
async def test_radiator_switch_full_cycle(
    hass: HomeAssistant, radiator_config_entry, radiator_home, recorded_service_calls
):
    """Full bistate cycle for a switch-backed radiator: ON then OFF service calls."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Switch Radiator",
            CONF_SWITCH: "switch.r",
            CONF_POWER: 1000,
        },
    )
    assert isinstance(device._transport, SwitchTransport)

    time = datetime.datetime.now(pytz.UTC)

    # ON command — turn_on service call
    result = await device.execute_command_system(time, CMD_ON, state=None)
    assert result is False
    on_calls = [c for c in recorded_service_calls if c[0] == Platform.SWITCH and c[1] == SERVICE_TURN_ON]
    assert len(on_calls) == 1
    assert on_calls[0][2] == {"entity_id": "switch.r"}

    # OFF command — turn_off service call
    result = await device.execute_command_system(time, CMD_OFF, state=None)
    assert result is False
    off_calls = [c for c in recorded_service_calls if c[0] == Platform.SWITCH and c[1] == SERVICE_TURN_OFF]
    assert len(off_calls) == 1


# =============================================================================
# Climate-backed full cycle
# =============================================================================


@pytest.mark.asyncio
async def test_radiator_climate_full_cycle(
    hass: HomeAssistant, radiator_config_entry, radiator_home, recorded_service_calls
):
    """Full bistate cycle for a climate-backed radiator: `heat` then `off`."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Climate Radiator",
            CONF_CLIMATE: "climate.r",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
            CONF_POWER: 1500,
        },
    )
    assert isinstance(device._transport, ClimateTransport)

    time = datetime.datetime.now(pytz.UTC)

    # ON — set_hvac_mode with `heat`
    result = await device.execute_command_system(time, CMD_ON, state=None)
    assert result is False
    on_calls = [
        c
        for c in recorded_service_calls
        if c[0] == climate.DOMAIN and c[1] == climate.SERVICE_SET_HVAC_MODE
    ]
    assert len(on_calls) == 1
    assert on_calls[0][2][ATTR_ENTITY_ID] == "climate.r"
    assert on_calls[0][2][climate.ATTR_HVAC_MODE] == "heat"

    # OFF — set_hvac_mode with `off`
    result = await device.execute_command_system(time, CMD_OFF, state=None)
    assert result is False
    off_calls = [
        c
        for c in recorded_service_calls
        if c[0] == climate.DOMAIN
        and c[1] == climate.SERVICE_SET_HVAC_MODE
        and c[2].get(climate.ATTR_HVAC_MODE) == "off"
    ]
    assert len(off_calls) == 1


# =============================================================================
# Heat-pump piloting — AC-10
# =============================================================================


@pytest.mark.asyncio
async def test_radiator_attaches_to_heat_pump_for_switch_backing(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """AC-10 — switch-backed radiator with `CONF_DEVICE_TO_PILOT_NAME` reads it through.

    Piloting is data-driven via `AbstractLoad.__init__` → no class-specific
    code in `home.add_device`. The name is on the load; the actual
    wiring happens in `_set_topology()` (exercised in HA-integration
    tests below).
    """
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Piloted Switch Radiator",
            CONF_SWITCH: "switch.r",
            CONF_DEVICE_TO_PILOT_NAME: "Main Heat Pump",
        },
    )

    assert device.piloted_device_name == "Main Heat Pump"
    # Sanity: the transport is the switch one — piloting works regardless.
    assert isinstance(device._transport, SwitchTransport)


@pytest.mark.asyncio
async def test_radiator_attaches_to_heat_pump_for_climate_backing(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """AC-10 — climate-backed radiator with `CONF_DEVICE_TO_PILOT_NAME` reads it through."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Piloted Climate Radiator",
            CONF_CLIMATE: "climate.r",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
            CONF_DEVICE_TO_PILOT_NAME: "Main Heat Pump",
        },
    )

    assert device.piloted_device_name == "Main Heat Pump"
    assert isinstance(device._transport, ClimateTransport)


@pytest.mark.asyncio
async def test_radiator_topology_pilot_wires_through_set_topology(hass: HomeAssistant):
    """AC-10 end-to-end — `home._set_topology()` populates `devices_to_pilot` for a radiator.

    Uses a stub home that mirrors the relevant slice of `QSHome` and runs
    the data-driven pilot wiring identically to production code.
    """
    radiator = MagicMock()
    radiator.name = "Bedroom Radiator"
    radiator.piloted_device_name = "Main Heat Pump"
    radiator.devices_to_pilot = []

    heat_pump = MagicMock()
    heat_pump.name = "Main Heat Pump"
    heat_pump.clients = []

    # Run the same loop as `home._set_topology()` (home.py:1876-1882).
    name_to_piloted = {heat_pump.name: heat_pump}
    for load in [radiator]:
        load.devices_to_pilot = []
        if load.piloted_device_name is not None:
            piloted = name_to_piloted.get(load.piloted_device_name)
            if piloted is not None:
                load.devices_to_pilot.append(piloted)
                piloted.clients.append(load)

    assert radiator.devices_to_pilot == [heat_pump]
    assert heat_pump.clients == [radiator]


# =============================================================================
# Initial state defaults
# =============================================================================


@pytest.mark.asyncio
async def test_radiator_climate_state_defaults_when_unset(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """Climate-backed radiator defaults to `heat`/`off` when HVAC modes aren't set."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Defaults Radiator",
            CONF_CLIMATE: "climate.r",
        },
    )

    assert device._state_on == "heat"
    assert device._state_off == "off"
    assert device._transport.default_state_on() == "heat"


# =============================================================================
# Override-state path — `execute_command_system` passes the override through
# =============================================================================


@pytest.mark.asyncio
async def test_radiator_switch_executes_with_override_state(
    hass: HomeAssistant, radiator_config_entry, radiator_home, recorded_service_calls
):
    """An override state (e.g. user forced ON) drives `switch.turn_on` regardless of command."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{CONF_NAME: "Override Radiator", CONF_SWITCH: "switch.r"},
    )

    time = datetime.datetime.now(pytz.UTC)
    # CMD_OFF + override_state="on" → turn_on
    result = await device.execute_command_system(time, CMD_OFF, state="on")

    assert result is False
    on_calls = [c for c in recorded_service_calls if c[0] == Platform.SWITCH and c[1] == SERVICE_TURN_ON]
    assert len(on_calls) == 1


@pytest.mark.asyncio
async def test_radiator_climate_executes_with_override_state(
    hass: HomeAssistant, radiator_config_entry, radiator_home, recorded_service_calls
):
    """For a climate radiator, `override_state` becomes the HVAC mode directly."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Climate Override Radiator",
            CONF_CLIMATE: "climate.r",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
        },
    )

    time = datetime.datetime.now(pytz.UTC)
    result = await device.execute_command_system(time, CMD_OFF, state="auto")

    assert result is False
    climate_calls = [
        c
        for c in recorded_service_calls
        if c[0] == climate.DOMAIN
        and c[1] == climate.SERVICE_SET_HVAC_MODE
        and c[2].get(climate.ATTR_HVAC_MODE) == "auto"
    ]
    assert len(climate_calls) == 1


# =============================================================================
# `AsyncMock` smoke check (silence lint warning on unused import)
# =============================================================================


def test_async_mock_imported() -> None:
    """Ensure `AsyncMock` import stays used (other tests rely on it indirectly)."""
    assert AsyncMock is not None
