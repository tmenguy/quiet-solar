"""Tests for `ha_model/bistate_transport.py` — switch and climate transports.

The transport strategy extracts per-backing details (which HA entity is
observed and which service is called) out of the bistate-duration
classes so a single host (`QSBiStateDuration`) can delegate to a
`SwitchTransport`, a `ClimateTransport`, or any future variant without
duplicating the bistate logic.
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz
from homeassistant.components import climate
from homeassistant.const import (
    ATTR_ENTITY_ID,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    Platform,
)

from custom_components.quiet_solar.ha_model.bistate_transport import (
    BistateTransport,
    ClimateTransport,
    SwitchTransport,
    get_hvac_modes,
)
from custom_components.quiet_solar.home_model.commands import (
    CMD_IDLE,
    CMD_OFF,
    CMD_ON,
    LoadCommand,
)


# =============================================================================
# Switch transport
# =============================================================================


def test_switch_transport_default_states():
    """SwitchTransport defaults are always `on` / `off`."""
    transport = SwitchTransport("switch.kitchen")

    assert transport.entity == "switch.kitchen"
    assert transport.default_state_on() == "on"
    assert transport.default_state_off() == "off"


def test_switch_transport_mode_options_static():
    """SwitchTransport mode options are static; `hass` argument is unused."""
    transport = SwitchTransport("switch.kitchen")
    hass = MagicMock()

    options = transport.mode_options(hass)

    assert options == ["on", "off"]


def test_switch_transport_power_from_state():
    """`power_from_state` returns `None` for unknown, `power_use` for on, `0` for off."""
    transport = SwitchTransport("switch.kitchen")

    assert transport.power_from_state(None, power_use=1500.0) is None
    assert transport.power_from_state(STATE_UNKNOWN, power_use=1500.0) is None
    assert transport.power_from_state(STATE_UNAVAILABLE, power_use=1500.0) is None
    assert transport.power_from_state("on", power_use=1500.0) == 1500.0
    assert transport.power_from_state("off", power_use=1500.0) == 0.0


@pytest.mark.asyncio
async def test_switch_transport_execute_turn_on():
    """Calling `execute` with `CMD_ON` issues a `switch.turn_on` service call."""
    transport = SwitchTransport("switch.kitchen")
    hass = MagicMock()
    hass.services.async_call = AsyncMock()

    time = datetime.datetime.now(pytz.UTC)
    result = await transport.execute(hass, CMD_ON, None, "on", "off")

    assert result is False
    hass.services.async_call.assert_awaited_once_with(
        domain=Platform.SWITCH,
        service=SERVICE_TURN_ON,
        target={"entity_id": "switch.kitchen"},
    )
    # silence unused-time lint
    assert time is not None


@pytest.mark.asyncio
async def test_switch_transport_execute_turn_off():
    """Calling `execute` with `CMD_OFF` issues a `switch.turn_off` service call."""
    transport = SwitchTransport("switch.kitchen")
    hass = MagicMock()
    hass.services.async_call = AsyncMock()

    result = await transport.execute(hass, CMD_OFF, None, "on", "off")

    assert result is False
    hass.services.async_call.assert_awaited_once_with(
        domain=Platform.SWITCH,
        service=SERVICE_TURN_OFF,
        target={"entity_id": "switch.kitchen"},
    )


@pytest.mark.asyncio
async def test_switch_transport_execute_uses_override_state():
    """When `override_state` is set, `execute` follows it (override on idle → turn ON)."""
    transport = SwitchTransport("switch.kitchen")
    hass = MagicMock()
    hass.services.async_call = AsyncMock()

    # CMD_IDLE alone would call turn_off; override_state="on" forces turn_on.
    result = await transport.execute(hass, CMD_IDLE, "on", "on", "off")

    assert result is False
    hass.services.async_call.assert_awaited_once_with(
        domain=Platform.SWITCH,
        service=SERVICE_TURN_ON,
        target={"entity_id": "switch.kitchen"},
    )


@pytest.mark.asyncio
async def test_switch_transport_execute_override_only_off_when_state_matches_off():
    """M1 regression — legacy semantics: only TURN_OFF when override matches state_off.

    Pre-refactor `execute_command_system` (legacy) did:
        if state == expected_state_from_command(CMD_IDLE):  # i.e. state_off
            action = SERVICE_TURN_OFF
        else:
            action = SERVICE_TURN_ON

    The transport must mirror this. An override state that is **neither**
    exactly `state_on` nor `state_off` (e.g. a custom HA helper state)
    must default to TURN_ON, not silently flip to TURN_OFF.
    """
    transport = SwitchTransport("switch.kitchen")
    hass = MagicMock()
    hass.services.async_call = AsyncMock()

    # Override state is neither "on" nor "off" — legacy says TURN_ON.
    result = await transport.execute(hass, CMD_IDLE, "custom_state", "on", "off")

    assert result is False
    hass.services.async_call.assert_awaited_once_with(
        domain=Platform.SWITCH,
        service=SERVICE_TURN_ON,
        target={"entity_id": "switch.kitchen"},
    )


@pytest.mark.asyncio
async def test_switch_transport_execute_override_off_calls_turn_off():
    """M1 regression — override_state matching state_off triggers TURN_OFF."""
    transport = SwitchTransport("switch.kitchen")
    hass = MagicMock()
    hass.services.async_call = AsyncMock()

    result = await transport.execute(hass, CMD_ON, "off", "on", "off")

    assert result is False
    hass.services.async_call.assert_awaited_once_with(
        domain=Platform.SWITCH,
        service=SERVICE_TURN_OFF,
        target={"entity_id": "switch.kitchen"},
    )


@pytest.mark.asyncio
async def test_switch_transport_execute_invalid_command_raises():
    """An unrecognised command (neither on, off, nor idle) raises `ValueError`."""
    transport = SwitchTransport("switch.kitchen")
    hass = MagicMock()
    hass.services.async_call = AsyncMock()

    invalid = LoadCommand(command="invalid", power_consign=0)

    with pytest.raises(ValueError, match="Invalid command"):
        await transport.execute(hass, invalid, None, "on", "off")


# =============================================================================
# Climate transport
# =============================================================================


def test_climate_transport_default_states():
    """ClimateTransport `default_*` reflect the configured HVAC modes."""
    transport = ClimateTransport("climate.living_room", "heat", "off")

    assert transport.entity == "climate.living_room"
    assert transport.default_state_on() == "heat"
    assert transport.default_state_off() == "off"
    assert transport.state_on == "heat"
    assert transport.state_off == "off"


def test_climate_transport_state_on_off_setters_mutate_attributes():
    """Mutating `state_on` / `state_off` updates the transport in place."""
    transport = ClimateTransport("climate.living_room", "heat", "off")

    transport.state_on = "auto"
    transport.state_off = "fan_only"

    assert transport.state_on == "auto"
    assert transport.state_off == "fan_only"
    assert transport.default_state_on() == "auto"
    assert transport.default_state_off() == "fan_only"


def test_climate_transport_mode_options_calls_get_hvac_modes():
    """`mode_options` delegates to `get_hvac_modes` against the climate entity."""
    transport = ClimateTransport("climate.living_room", "heat", "off")
    hass = MagicMock()

    with patch(
        "custom_components.quiet_solar.ha_model.bistate_transport.get_hvac_modes",
        return_value=["off", "heat", "cool"],
    ) as mock_get:
        result = transport.mode_options(hass)

    assert result == ["off", "heat", "cool"]
    mock_get.assert_called_once_with(hass, "climate.living_room")


@pytest.mark.asyncio
async def test_climate_transport_execute_set_hvac_mode():
    """Calling `execute` issues a `climate.set_hvac_mode` service call."""
    transport = ClimateTransport("climate.living_room", "heat", "off")
    hass = MagicMock()
    hass.services.async_call = AsyncMock()

    result = await transport.execute(hass, CMD_ON, None, "heat", "off")

    assert result is False
    hass.services.async_call.assert_awaited_once_with(
        climate.DOMAIN,
        climate.SERVICE_SET_HVAC_MODE,
        {ATTR_ENTITY_ID: "climate.living_room", climate.ATTR_HVAC_MODE: "heat"},
    )


@pytest.mark.asyncio
async def test_climate_transport_execute_uses_state_off_for_off_command():
    """`CMD_OFF` selects the `state_off` value (e.g. `off`)."""
    transport = ClimateTransport("climate.living_room", "heat", "off")
    hass = MagicMock()
    hass.services.async_call = AsyncMock()

    result = await transport.execute(hass, CMD_OFF, None, "heat", "off")

    assert result is False
    call_args = hass.services.async_call.await_args.args
    assert call_args[0] == climate.DOMAIN
    assert call_args[1] == climate.SERVICE_SET_HVAC_MODE
    assert call_args[2][climate.ATTR_HVAC_MODE] == "off"


@pytest.mark.asyncio
async def test_climate_transport_execute_override_state_wins():
    """When `override_state` is provided it becomes the HVAC mode."""
    transport = ClimateTransport("climate.living_room", "heat", "off")
    hass = MagicMock()
    hass.services.async_call = AsyncMock()

    result = await transport.execute(hass, CMD_ON, "cool", "heat", "off")

    assert result is False
    call_args = hass.services.async_call.await_args.args
    assert call_args[2][climate.ATTR_HVAC_MODE] == "cool"


@pytest.mark.asyncio
async def test_climate_transport_execute_invalid_command_raises():
    """An invalid command on the climate transport raises `ValueError`."""
    transport = ClimateTransport("climate.living_room", "heat", "off")
    hass = MagicMock()
    hass.services.async_call = AsyncMock()

    invalid = LoadCommand(command="invalid", power_consign=0)

    with pytest.raises(ValueError, match="Invalid command"):
        await transport.execute(hass, invalid, None, "heat", "off")


# =============================================================================
# get_hvac_modes helper (re-exported through `bistate_transport`)
# =============================================================================


def test_get_hvac_modes_uses_entity_registry():
    """`get_hvac_modes` reads capabilities from the entity registry."""
    hass = MagicMock()
    entry = MagicMock()
    entry.capabilities = {"hvac_modes": ["off", "heat", "cool", "auto"]}
    registry = MagicMock()
    registry.async_get.return_value = entry

    with patch(
        "custom_components.quiet_solar.ha_model.bistate_transport.er.async_get",
        return_value=registry,
    ):
        modes = get_hvac_modes(hass, "climate.living_room")

    assert modes == ["off", "heat", "cool", "auto"]


def test_get_hvac_modes_defaults_when_capabilities_missing():
    """N11 — when the entity has no `hvac_modes` capability, defaults to BOTH `auto` AND `off`."""
    hass = MagicMock()
    entry = MagicMock()
    entry.capabilities = {}
    registry = MagicMock()
    registry.async_get.return_value = entry

    with patch(
        "custom_components.quiet_solar.ha_model.bistate_transport.er.async_get",
        return_value=registry,
    ):
        modes = get_hvac_modes(hass, "climate.living_room")

    # N11 — pin the full fallback set rather than the loose "either" check.
    assert {"auto", "off"}.issubset(set(modes))


def test_get_hvac_modes_returns_defaults_when_entry_is_none():
    """M6 — `registry.async_get` returning `None` (stale entity) falls back cleanly."""
    hass = MagicMock()
    registry = MagicMock()
    registry.async_get.return_value = None

    with patch(
        "custom_components.quiet_solar.ha_model.bistate_transport.er.async_get",
        return_value=registry,
    ):
        modes = get_hvac_modes(hass, "climate.missing")

    assert {"auto", "off"}.issubset(set(modes))


def test_get_hvac_modes_returns_defaults_when_capabilities_is_none():
    """M6 — `entry.capabilities` being `None` (some climates) falls back cleanly."""
    hass = MagicMock()
    entry = MagicMock()
    entry.capabilities = None
    registry = MagicMock()
    registry.async_get.return_value = entry

    with patch(
        "custom_components.quiet_solar.ha_model.bistate_transport.er.async_get",
        return_value=registry,
    ):
        modes = get_hvac_modes(hass, "climate.no_capabilities")

    assert {"auto", "off"}.issubset(set(modes))


@pytest.mark.asyncio
async def test_switch_transport_execute_none_command_raises_value_error():
    """N5 — `command=None` raises `ValueError`, not `AttributeError`."""
    transport = SwitchTransport("switch.kitchen")
    hass = MagicMock()
    hass.services.async_call = AsyncMock()

    with pytest.raises(ValueError, match="Invalid command"):
        await transport.execute(hass, None, None, "on", "off")


@pytest.mark.asyncio
async def test_climate_transport_execute_none_command_raises_value_error():
    """N5 — same for `ClimateTransport.execute(None, None, …)`."""
    transport = ClimateTransport("climate.living_room", "heat", "off")
    hass = MagicMock()
    hass.services.async_call = AsyncMock()

    with pytest.raises(ValueError, match="Invalid command"):
        await transport.execute(hass, None, None, "heat", "off")


# =============================================================================
# Layer-rule sanity checks (the abstract base is public)
# =============================================================================


def test_bistate_transport_is_public_class():
    """`BistateTransport` is the public name (no leading underscore)."""
    assert BistateTransport.__name__ == "BistateTransport"


def test_switch_transport_is_subclass_of_bistate_transport():
    """`SwitchTransport` extends `BistateTransport`."""
    assert issubclass(SwitchTransport, BistateTransport)


def test_climate_transport_is_subclass_of_bistate_transport():
    """`ClimateTransport` extends `BistateTransport`."""
    assert issubclass(ClimateTransport, BistateTransport)
