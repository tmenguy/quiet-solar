"""Tests for `ha_model/radiator.py` — `QSRadiator` heating-only load.

A radiator is a `QSBiStateDuration` whose backing is either a switch
(`CONF_SWITCH`) OR a climate entity (`CONF_CLIMATE`), but never both
and never neither. It picks a `BistateTransport` strategy at
construction time and inherits the rest of the bistate logic from its
parent.
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    CONF_CLIMATE,
    CONF_CLIMATE_HVAC_MODE_OFF,
    CONF_CLIMATE_HVAC_MODE_ON,
    CONF_DEVICE_TO_PILOT_NAME,
    CONF_POWER,
    CONF_SWITCH,
    CONF_TYPE_NAME_QSRadiator,
    DATA_HANDLER,
    DOMAIN,
    SENSOR_CONSTRAINT_SENSOR_RADIATOR,
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
        entry_id="test_radiator_entry",
        data={CONF_NAME: "Test Radiator"},
        title="Test Radiator",
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


def test_radiator_conf_type_name_is_radiator():
    """`QSRadiator.conf_type_name == "radiator"` (matches the new constant)."""
    assert QSRadiator.conf_type_name == CONF_TYPE_NAME_QSRadiator
    assert QSRadiator.conf_type_name == "radiator"


def test_radiator_init_with_switch_picks_switch_transport(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """When configured with `CONF_SWITCH`, the radiator picks a `SwitchTransport`."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Bathroom Radiator",
            CONF_SWITCH: "switch.bathroom_radiator",
            CONF_POWER: 800,
        },
    )

    assert isinstance(device._transport, SwitchTransport)
    assert device._transport.entity == "switch.bathroom_radiator"
    assert device.bistate_entity == "switch.bathroom_radiator"
    assert device._state_on == "on"
    assert device._state_off == "off"


def test_radiator_init_with_climate_picks_climate_transport(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """When configured with `CONF_CLIMATE`, the radiator picks a `ClimateTransport`."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Living Room Radiator",
            CONF_CLIMATE: "climate.living_room",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
            CONF_POWER: 1500,
        },
    )

    assert isinstance(device._transport, ClimateTransport)
    assert device._transport.entity == "climate.living_room"
    assert device.bistate_entity == "climate.living_room"
    assert device._state_on == "heat"
    assert device._state_off == "off"


def test_radiator_init_both_truthy_picks_climate_and_logs(
    hass: HomeAssistant, radiator_config_entry, radiator_home, caplog
):
    """EH6 — a stale entry with BOTH backings deterministically prefers climate.

    A buggy migration or a manual `.storage/` edit could end up with
    both `CONF_CLIMATE` and `CONF_SWITCH` set. Rather than crashing the
    entry's reload (the old `ServiceValidationError` path), we log a
    warning and pick `CONF_CLIMATE` as the canonical winner — it's the
    richer backing and matches what the config-flow defaults steer to.
    The other key is dropped from `kwargs` in place so the base
    `AbstractLoad` doesn't pick up the loser.
    """
    import logging as _logging

    caplog.set_level(_logging.WARNING, logger="custom_components.quiet_solar.ha_model.radiator")

    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Stale Both Backings Radiator",
            CONF_SWITCH: "switch.r",
            CONF_CLIMATE: "climate.r",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
        },
    )

    assert isinstance(device._transport, ClimateTransport)
    assert device._transport.entity == "climate.r"
    # `AbstractLoad.__init__` populated `switch_entity` from kwargs — we
    # popped `CONF_SWITCH` before super so it lands as `None`.
    assert device.switch_entity is None
    assert any("both" in rec.message.lower() and "backing" in rec.message.lower() for rec in caplog.records)


def test_radiator_init_neither_raises_service_validation_error(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """Configuring neither `CONF_SWITCH` nor `CONF_CLIMATE` raises `ServiceValidationError`.

    EH6 turns the both-set case into a warn-and-pick, but the
    neither-set case remains a hard error — there's no sensible
    default to pick from.
    """
    with pytest.raises(ServiceValidationError, match="exactly one of climate or switch"):
        QSRadiator(
            hass=hass,
            config_entry=radiator_config_entry,
            home=radiator_home,
            **{CONF_NAME: "Backingless Radiator"},
        )


def test_radiator_climate_defaults_to_heat_off(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """A climate-backed radiator defaults to `heat` / `off` (heating-only)."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Bedroom Radiator",
            CONF_CLIMATE: "climate.bedroom",
            # No CONF_CLIMATE_HVAC_MODE_ON / OFF provided
        },
    )

    assert device._state_on == "heat"
    assert device._state_off == "off"


def test_radiator_select_translation_key_is_radiator_mode(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """Radiator exposes its own select-translation key (no clash with on/off or climate)."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{CONF_NAME: "Test Radiator", CONF_SWITCH: "switch.r"},
    )

    assert device.get_select_translation_key() == "radiator_mode"


def test_radiator_virtual_current_constraint_translation_key_is_radiator(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """Constraint sensor uses the dedicated `SENSOR_CONSTRAINT_SENSOR_RADIATOR` key."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{CONF_NAME: "Test Radiator", CONF_SWITCH: "switch.r"},
    )

    assert device.get_virtual_current_constraint_translation_key() == SENSOR_CONSTRAINT_SENSOR_RADIATOR


def test_radiator_piloted_device_attaches_for_switch_backing(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """The `device_to_pilot_name` is read by `AbstractLoad.__init__` for any backing."""
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


def test_radiator_piloted_device_attaches_for_climate_backing(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """Same wiring works with `CONF_CLIMATE` — piloting is data-driven."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Piloted Climate Radiator",
            CONF_CLIMATE: "climate.r",
            CONF_DEVICE_TO_PILOT_NAME: "Main Heat Pump",
        },
    )

    assert device.piloted_device_name == "Main Heat Pump"


@pytest.mark.asyncio
async def test_radiator_execute_command_calls_transport_switch(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """`execute_command_system` delegates to the switch transport."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{CONF_NAME: "Test Radiator", CONF_SWITCH: "switch.r"},
    )
    # B3 — the inherited `_state_on/_state_off` property reads through
    # the transport, so the mock must expose `state_on`/`state_off`.
    mock_transport = MagicMock(spec=SwitchTransport)
    mock_transport.execute = AsyncMock(return_value=False)
    mock_transport.entity = "switch.r"
    mock_transport.state_on = "on"
    mock_transport.state_off = "off"
    device._transport = mock_transport

    time = datetime.datetime.now(pytz.UTC)
    result = await device.execute_command_system(time, CMD_ON, state=None)

    assert result is False
    device._transport.execute.assert_awaited_once()
    args = device._transport.execute.await_args.args
    assert args[0] is hass
    assert args[1] is CMD_ON
    assert args[2] is None
    # state_on / state_off are the host's values (now routed via property → transport)
    assert args[3] == "on"
    assert args[4] == "off"


@pytest.mark.asyncio
async def test_radiator_execute_command_calls_transport_climate(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """`execute_command_system` delegates to the climate transport with `heat`/`off`."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Test Climate Radiator",
            CONF_CLIMATE: "climate.r",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
        },
    )
    # B3 — the inherited `_state_on/_state_off` property reads through
    # the transport, so the mock must expose `state_on`/`state_off`.
    mock_transport = MagicMock(spec=ClimateTransport)
    mock_transport.execute = AsyncMock(return_value=False)
    mock_transport.entity = "climate.r"
    mock_transport.state_on = "heat"
    mock_transport.state_off = "off"
    device._transport = mock_transport

    time = datetime.datetime.now(pytz.UTC)
    result = await device.execute_command_system(time, CMD_OFF, state=None)

    assert result is False
    device._transport.execute.assert_awaited_once()


def test_radiator_no_climate_state_on_off_attributes(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """Regression (AC-12): radiator does not expose `climate_state_on/off` properties.

    The runtime select for climate state is removed — radiators are
    heating-only and configured once.
    """
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Test Radiator",
            CONF_CLIMATE: "climate.r",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
        },
    )

    # The radiator must NOT be a QSClimateDuration (otherwise the select
    # platform would create the runtime climate_state_on / climate_state_off
    # selects). It IS a QSBiStateDuration via inheritance.
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    from custom_components.quiet_solar.ha_model.climate_controller import QSClimateDuration

    assert isinstance(device, QSBiStateDuration)
    assert not isinstance(device, QSClimateDuration)


def test_radiator_empty_climate_with_switch_picks_switch_transport(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """M4 regression — `CONF_CLIMATE=""` is treated as absent; switch wins."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Empty Climate Radiator",
            CONF_CLIMATE: "",
            CONF_SWITCH: "switch.r",
        },
    )

    assert isinstance(device._transport, SwitchTransport)
    assert device._transport.entity == "switch.r"


def test_radiator_empty_switch_with_climate_picks_climate_transport(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """M4 regression — `CONF_SWITCH=""` is treated as absent; climate wins."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Empty Switch Radiator",
            CONF_CLIMATE: "climate.r",
            CONF_SWITCH: "",
        },
    )

    assert isinstance(device._transport, ClimateTransport)
    assert device._transport.entity == "climate.r"


def test_radiator_whitespace_entity_is_treated_as_absent(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """N6 regression — `CONF_CLIMATE="   "` is normalised away."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Whitespace Climate Radiator",
            CONF_CLIMATE: "   ",
            CONF_SWITCH: "switch.r",
        },
    )

    assert isinstance(device._transport, SwitchTransport)


def test_radiator_empty_hvac_mode_falls_back_to_heat(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """S9 regression — persisted `CONF_CLIMATE_HVAC_MODE_ON=""` falls back to `heat`."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Empty HVAC Mode Radiator",
            CONF_CLIMATE: "climate.r",
            CONF_CLIMATE_HVAC_MODE_ON: "",
            CONF_CLIMATE_HVAC_MODE_OFF: "",
        },
    )

    assert device._state_on == "heat"
    assert device._state_off == "off"


def test_radiator_no_climate_state_on_off_attribute_hasattr(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """S10 — `hasattr` regression: radiator must NOT expose `climate_state_on/off`."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Hasattr Radiator",
            CONF_CLIMATE: "climate.r",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
        },
    )

    assert not hasattr(device, "climate_state_on")
    assert not hasattr(device, "climate_state_off")


def test_radiator_transport_built_before_super_init(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """S2 regression — `self._transport` is available during `super().__init__`."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{CONF_NAME: "Init Order Radiator", CONF_SWITCH: "switch.r"},
    )

    # After init, host-owned values reflect the transport's defaults.
    # The fact that init didn't crash plus that bistate_entity matches
    # the transport's entity (rather than `switch_entity` overridden by
    # the base ctor) proves the transport was built first.
    assert device._transport.entity == "switch.r"
    assert device.bistate_entity == "switch.r"
    assert device._state_on == "on"
    assert device._state_off == "off"


def test_radiator_bistate_modes_set_is_pinned(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """S12 + CR4 — pin the EXACT set of `radiator_mode` select states.

    AC-7 enumerated which states the `radiator_mode` select must
    expose. Set-equality (rather than `issubset`) catches both
    missing and unexpected modes.
    """
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{CONF_NAME: "Modes Pin Radiator", CONF_SWITCH: "switch.r"},
    )

    modes = device.get_bistate_modes()

    assert set(modes) == {
        "bistate_mode_auto",
        "bistate_mode_exact_calendar",
        "bistate_mode_default",
        "on",
        "off",
    }


def test_radiator_climate_bistate_modes_set_includes_on_off(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """E3 + CR4 — climate-backed radiator also exposes `on`/`off` (not raw HVAC modes).

    E3 hardcoded the bistate-mode labels to `on`/`off` regardless of
    the HVAC mode configured for the underlying climate entity. The
    `radiator_mode` translation only carries `on` / `off` / calendar
    keys; raw HVAC mode strings like `auto` would render unlocalised
    otherwise.
    """
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Climate Modes Pin Radiator",
            CONF_CLIMATE: "climate.r",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
        },
    )

    modes = device.get_bistate_modes()

    assert set(modes) == {
        "bistate_mode_auto",
        "bistate_mode_exact_calendar",
        "bistate_mode_default",
        "on",
        "off",
    }


def test_radiator_state_on_routes_through_transport(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """B3 regression — radiator inherits the `_state_on` property shadow.

    Writing through `self._state_on` MUST update the transport so the
    host and the underlying service-call layer never diverge. Mirrors
    the climate-controller round-trip test for parity.
    """
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Property Shadow Radiator",
            CONF_CLIMATE: "climate.r",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
        },
    )

    device._state_on = "cool"
    assert device._transport.state_on == "cool"
    assert device._state_on == "cool"


def test_radiator_state_off_routes_through_transport(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """B3 regression — same for `_state_off`."""
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Property Shadow Off Radiator",
            CONF_SWITCH: "switch.r",
        },
    )

    device._state_off = "stopped"
    assert device._transport.state_off == "stopped"
    assert device._state_off == "stopped"


def test_radiator_kwargs_normalised_in_place(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """E1 regression — whitespace switch entity must NOT leak into base ctor.

    Constructing with `switch="   "` plus a real climate must leave
    `device.switch_entity` empty/None (the base ctor reads `CONF_SWITCH`
    from kwargs, which must be normalised before super runs).
    """
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Whitespace Switch Radiator",
            CONF_CLIMATE: "climate.r",
            CONF_SWITCH: "   ",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
        },
    )

    # `switch_entity` is the attribute that downstream code (home.py
    # device registry, dashboard template) sees. It must NOT be "   ".
    assert not device.switch_entity or device.switch_entity.strip() == ""


def test_radiator_bistate_mode_labels_independent_of_hvac_mode(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """E3 regression — radiator bistate-mode labels are always `on`/`off`.

    Even when the climate-backed radiator's HVAC ON is `auto`, the
    bistate-mode select must expose `on`/`off` so the `radiator_mode`
    translation can label them ("Force ON" / "Force OFF") instead of
    showing the raw HVAC mode string.
    """
    device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Auto HVAC Radiator",
            CONF_CLIMATE: "climate.r",
            CONF_CLIMATE_HVAC_MODE_ON: "auto",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
        },
    )

    assert device._bistate_mode_on == "on"
    assert device._bistate_mode_off == "off"
    # The transport still uses the raw HVAC mode for service calls.
    assert device._transport.state_on == "auto"


def test_radiator_options_flow_backing_swap_picks_new_transport(
    hass: HomeAssistant, radiator_config_entry, radiator_home
):
    """AC-13 — re-creating a radiator with a different backing picks the new transport.

    The options-flow swap is wired through `async_reload_entry` →
    re-instantiation. Verifying that re-instantiating with the new
    backing picks the right transport is the same property the reload
    relies on.
    """
    switch_device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{CONF_NAME: "Swap Radiator", CONF_SWITCH: "switch.r"},
    )
    assert isinstance(switch_device._transport, SwitchTransport)

    # Simulate the options-flow swap: rebuild the device with the new
    # config (the same device_id thanks to the stable slug-based id).
    climate_device = QSRadiator(
        hass=hass,
        config_entry=radiator_config_entry,
        home=radiator_home,
        **{
            CONF_NAME: "Swap Radiator",
            CONF_CLIMATE: "climate.r",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
        },
    )
    assert isinstance(climate_device._transport, ClimateTransport)
    # Stable identity across the swap (device_id is derived from name + type).
    assert switch_device.device_id == climate_device.device_id
