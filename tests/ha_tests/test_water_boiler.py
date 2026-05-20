"""Tests for quiet_solar ha_model/water_boiler.py.

QS-194: new `water_boiler` (cumulus) load type — a thin subclass of
`QSOnOffDuration` with a dedicated config step, dashboard section,
select-mode translation key, and an optional water-tank temperature
sensor (plumbing only — no constraint/solver logic acts on it).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import DOMAIN
from custom_components.quiet_solar.ha_model.device import HADeviceMixin
from custom_components.quiet_solar.ha_model.on_off_duration import QSOnOffDuration
from custom_components.quiet_solar.ha_model.water_boiler import QSWaterBoiler

from .const import (
    MOCK_WATER_BOILER_CONFIG,
    MOCK_WATER_BOILER_CONFIG_NO_TEMP,
    MOCK_WATER_BOILER_ENTRY_ID,
    MOCK_WATER_BOILER_NO_TEMP_ENTRY_ID,
)

pytestmark = pytest.mark.usefixtures("mock_sensor_states")

_TEMP_SENSOR_ID = "sensor.test_water_boiler_temperature"


def _matching_probe_calls(mock_probe, entity_id: str | None) -> list:
    """Filter the recorded probe calls down to those for a given entity id.

    `attach_ha_state_to_probe` is called many times during device init
    (power sensor, amps sensors, etc.). This helper isolates the call(s)
    for a specific entity id so the test can make precise assertions
    without coupling to the unrelated calls.

    All production callers in this repo pass the entity id positionally
    (see pool.py, car.py, battery.py, etc.); a keyword-form fallback
    would be dead code, so only positional matching is supported.
    With ``autospec=True`` the first positional arg is ``self``; the
    entity id is therefore at index 1.
    """
    return [
        call
        for call in mock_probe.call_args_list
        if len(call.args) >= 2 and call.args[1] == entity_id
    ]


async def test_water_boiler_device_type_registered(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """A water_boiler config entry creates a QSWaterBoiler reachable via the home."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_WATER_BOILER_CONFIG,
        entry_id=MOCK_WATER_BOILER_ENTRY_ID,
        title=f"water_boiler: {MOCK_WATER_BOILER_CONFIG['name']}",
        unique_id=f"quiet_solar_{MOCK_WATER_BOILER_ENTRY_ID}",
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED

    device = hass.data[DOMAIN].get(entry.entry_id)
    assert device is not None
    assert isinstance(device, QSWaterBoiler)
    # QSWaterBoiler is a subclass of QSOnOffDuration — isinstance check
    # preserved for existing call-sites.
    assert isinstance(device, QSOnOffDuration)
    assert device.device_type == "water_boiler"
    assert QSWaterBoiler.conf_type_name == "water_boiler"

    # Reachable via the home's dashboard-section lookup
    home = hass.data[DOMAIN].get(home_config_entry.entry_id)
    assert home is not None
    devices_in_section = home.get_devices_for_dashboard_section("water_boilers")
    assert device in devices_in_section


async def test_water_boiler_with_temperature_sensor_attaches_probe(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When configured with a temp sensor, the entity id is stored and probed.

    Spies on `HADeviceMixin.attach_ha_state_to_probe` via `wraps=` so the
    real probe-attachment side-effects still happen (state machine,
    history bootstrap, etc.) while the call list is captured for
    assertion. Filtering by `_TEMP_SENSOR_ID` isolates the boiler's
    temperature-sensor call from the many unrelated calls made during
    parent-class init (power sensor, amps sensors, etc.).
    """
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_WATER_BOILER_CONFIG,
        entry_id=MOCK_WATER_BOILER_ENTRY_ID,
        title=f"water_boiler: {MOCK_WATER_BOILER_CONFIG['name']}",
        unique_id=f"quiet_solar_{MOCK_WATER_BOILER_ENTRY_ID}",
    )
    entry.add_to_hass(hass)

    real_attach = HADeviceMixin.attach_ha_state_to_probe
    with patch.object(
        HADeviceMixin,
        "attach_ha_state_to_probe",
        autospec=True,
        wraps=real_attach,
    ) as mock_probe:
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    device = hass.data[DOMAIN].get(entry.entry_id)
    assert device is not None

    assert device.water_boiler_temperature_sensor == _TEMP_SENSOR_ID
    # attach_ha_state_to_probe was invoked once with the temperature sensor
    # entity id and is_numerical=True, mirroring the pool.py pattern.
    matches = _matching_probe_calls(mock_probe, _TEMP_SENSOR_ID)
    assert len(matches) == 1, (
        f"Expected exactly one attach_ha_state_to_probe call for "
        f"{_TEMP_SENSOR_ID}, got {len(matches)}: {matches}"
    )
    assert matches[0].kwargs.get("is_numerical") is True


async def test_water_boiler_without_temperature_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When no temp sensor is configured, the field is None and no probe is attached."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_WATER_BOILER_CONFIG_NO_TEMP,
        entry_id=MOCK_WATER_BOILER_NO_TEMP_ENTRY_ID,
        title=f"water_boiler: {MOCK_WATER_BOILER_CONFIG_NO_TEMP['name']}",
        unique_id=f"quiet_solar_{MOCK_WATER_BOILER_NO_TEMP_ENTRY_ID}",
    )
    entry.add_to_hass(hass)

    real_attach = HADeviceMixin.attach_ha_state_to_probe
    with patch.object(
        HADeviceMixin,
        "attach_ha_state_to_probe",
        autospec=True,
        wraps=real_attach,
    ) as mock_probe:
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    device = hass.data[DOMAIN].get(entry.entry_id)
    assert device is not None
    assert device.water_boiler_temperature_sensor is None
    # No call to attach_ha_state_to_probe was made with the (absent) temp
    # sensor entity id.
    assert _matching_probe_calls(mock_probe, _TEMP_SENSOR_ID) == []
    # The short-circuit pathway WAS exercised: QSWaterBoiler.__init__
    # unconditionally calls attach_ha_state_to_probe(None, ...) which
    # early-returns inside the implementation. Mirrors pool.py:34.
    none_calls = _matching_probe_calls(mock_probe, None)
    assert len(none_calls) >= 1, (
        "Expected at least one attach_ha_state_to_probe(None, ...) call "
        "to exercise the short-circuit pathway"
    )


async def test_water_boiler_empty_string_temperature_sensor_normalised_to_none(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """WF-3: an empty-string temp sensor value is normalised to None.

    The options-flow form can store `""` when an EntitySelector is
    cleared; without normalisation that empty string would propagate
    into `attach_ha_state_to_probe("", ...)` and register `""` as a
    probe entity id. Water_boiler is the only optional instance of
    this pattern (pool's field is required).
    """
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    config = {
        **MOCK_WATER_BOILER_CONFIG_NO_TEMP,
        # Inject the empty-string degenerate case
        "water_boiler_temperature_sensor": "",
    }
    entry = MockConfigEntry(
        domain=DOMAIN,
        data=config,
        entry_id="water_boiler_empty_str_test",
        title=f"water_boiler: {config['name']}",
        unique_id="quiet_solar_water_boiler_empty_str_test",
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    device = hass.data[DOMAIN].get(entry.entry_id)
    assert device is not None
    assert device.water_boiler_temperature_sensor is None, (
        "Empty string must be normalised to None to avoid registering "
        '"" as a probe entity id'
    )


async def test_water_boiler_select_translation_key(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """QSWaterBoiler returns the dedicated `water_boiler_mode` translation key."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_WATER_BOILER_CONFIG,
        entry_id=MOCK_WATER_BOILER_ENTRY_ID,
        title=f"water_boiler: {MOCK_WATER_BOILER_CONFIG['name']}",
        unique_id=f"quiet_solar_{MOCK_WATER_BOILER_ENTRY_ID}",
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    device = hass.data[DOMAIN].get(entry.entry_id)
    assert device is not None
    assert device.get_select_translation_key() == "water_boiler_mode"
