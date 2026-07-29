"""QS-304: HA-layer behaviour when QS loses control of a load.

Covers the parts of the story that only exist above the domain boundary: the
`qs_load_uncontrollable` binary sensor, the mobile push, the absence of
collateral damage on power accounting, the off-grid contract, and the fact that
QS never touches the user's `qs_enable_device` switch.
"""

from __future__ import annotations

import inspect
import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.binary_sensor import create_ha_binary_sensor
from custom_components.quiet_solar.const import (
    BINARY_SENSOR_LOAD_UNCONTROLLABLE,
    DATA_HANDLER,
    DOMAIN,
)
from custom_components.quiet_solar.ha_model.home import QSHome, QSHomeMode
from custom_components.quiet_solar.home_model.commands import CMD_IDLE, CMD_ON, copy_command
from custom_components.quiet_solar.home_model.load import (
    NUM_MAX_COMMAND_RELAUNCH,
    AbstractDevice,
)

from .const import MOCK_BATTERY_CONFIG, MOCK_CHARGER_CONFIG

pytestmark = pytest.mark.usefixtures("mock_sensor_states")

T0 = datetime(2026, 7, 27, 12, 12, 19, tzinfo=pytz.UTC)
CYCLE_S = 7
LADDER_WALL_S = 1050 + NUM_MAX_COMMAND_RELAUNCH * CYCLE_S


async def _get_home(hass: HomeAssistant, entry: ConfigEntry) -> QSHome:
    """Set up the home entry and return the QSHome object."""
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return hass.data[DOMAIN][DATA_HANDLER].home


async def _add_charger(hass: HomeAssistant, entry_id: str):
    """Add a real QSChargerGeneric from MOCK_CHARGER_CONFIG and return it."""
    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id=entry_id,
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id=f"qs_{entry_id}",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()
    return hass.data[DOMAIN][charger_entry.entry_id], charger_entry


def _make_never_ack(load) -> None:
    """Make a real device answer its probe but never confirm the command."""
    load.probe_if_command_set = AsyncMock(return_value=False)
    load.execute_command = AsyncMock(return_value=False)


async def _drive(load, start: datetime, duration_s: float) -> datetime:
    """Run the load-management cycle at the observed ~7 s cadence."""
    time = start
    end = start + timedelta(seconds=duration_s)
    while time <= end:
        await load.check_and_relaunch_command(time)
        time = time + timedelta(seconds=CYCLE_S)
    return time


# =============================================================================
# AC7 — no collateral damage on power accounting
# =============================================================================

IS_LOAD_COMMAND_SET_SOURCE = """    def is_load_command_set(self, time: datetime):
        if self.qs_enable_device is False:
            return False

        return self.running_command is None and self.current_command is not None
"""


def test_is_load_command_set_body_is_unmodified():
    """AC7: widening `is_load_command_set` would move the persisted forecast."""
    assert inspect.getsource(AbstractDevice.is_load_command_set) == IS_LOAD_COMMAND_SET_SOURCE


def _accounting_snapshot(home: QSHome, load, time: datetime) -> dict:
    """Capture every power-accounting value an uncontrollable load could move."""
    return {
        "is_load_command_set": load.is_load_command_set(time),
        "is_user_overridden": load.is_user_overridden(),
        "command_power_state": load.command_power_state_getter(load.command_based_power_sensor, time),
        "device_power": load.get_device_power_latest_possible_valid_value(
            tolerance_seconds=None, time=time, ignore_auto_and_user_overridden_load=True
        ),
        "group_power": home.get_device_power_latest_possible_valid_value(
            tolerance_seconds=None, time=time, ignore_auto_and_user_overridden_load=True
        ),
    }


async def test_uncontrollable_load_does_not_perturb_power_accounting(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """AC7: crossing the lost-control threshold moves no accounting value."""
    home = await _get_home(hass, home_config_entry)
    charger, charger_entry = await _add_charger(hass, "charger_ac7")
    _make_never_ack(charger)

    # The device is confirmed `on` at 3000 W, then asked to go `idle` and
    # refuses. This is the shape both before and after QS-304.
    charger._ack_command(T0 - timedelta(seconds=60), copy_command(CMD_ON, power_consign=3000.0))
    await charger.launch_command(T0, CMD_IDLE)
    assert charger.running_command is not None

    time = await _drive(charger, T0 + timedelta(seconds=CYCLE_S), 3 * 60)
    assert charger.is_uncontrollable is False
    before = _accounting_snapshot(home, charger, time)

    time = await _drive(charger, time, LADDER_WALL_S)
    assert charger.is_uncontrollable is True
    after = _accounting_snapshot(home, charger, time)

    assert after == before
    # And the concrete values, pinned, so a future change cannot move both sides
    # together and stay green.
    assert after["is_load_command_set"] is False
    assert after["is_user_overridden"] is False
    assert after["command_power_state"] is None
    assert after["device_power"] == 0.0
    assert after["group_power"] == 0.0
    # `current_command` is the last CONFIRMED command and stays truthful.
    assert charger.current_command == copy_command(CMD_ON, power_consign=3000.0)

    await hass.config_entries.async_unload(charger_entry.entry_id)
    await hass.async_block_till_done()


# =============================================================================
# AC8 — the off-grid contract is unchanged
# =============================================================================


async def test_permanently_uncontrollable_load_still_blocks_off_grid_switch(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """AC8: `all_ok` stays False, exactly as for a merely-unacked command."""
    home = await _get_home(hass, home_config_entry)
    charger, charger_entry = await _add_charger(hass, "charger_ac8")
    _make_never_ack(charger)

    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home._init_completed = True
    home.physical_battery = None

    await charger.launch_command(T0, CMD_IDLE)
    time = await _drive(charger, T0 + timedelta(seconds=CYCLE_S), 3 * 60)
    assert await home.check_loads_commands(time) is False

    time = await _drive(charger, time, LADDER_WALL_S)
    assert charger.is_uncontrollable is True
    assert await home.check_loads_commands(time) is False

    home._switch_to_off_grid_launched = time - timedelta(seconds=30)
    finished, just_switched = await home.finish_off_grid_switch(time)
    assert finished is False
    assert just_switched is False

    await hass.config_entries.async_unload(charger_entry.entry_id)
    await hass.async_block_till_done()


# =============================================================================
# AC9 — one line in, one push in, one line out, no push out
# =============================================================================


async def test_entry_pushes_once_and_recovery_pushes_never(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC9: exactly one ERROR + one push on entry, one INFO and no push out."""
    home = await _get_home(hass, home_config_entry)
    charger, charger_entry = await _add_charger(hass, "charger_ac9")
    _make_never_ack(charger)
    charger.mobile_app = "mobile_app_test_phone"

    notify_calls = []
    original_async_call = hass.services.async_call

    async def recording_async_call(self, domain, service, *args, **kwargs):
        if domain == Platform.NOTIFY:
            notify_calls.append((service, kwargs.get("service_data")))
            return None
        return await original_async_call(domain, service, *args, **kwargs)

    with (
        patch.object(type(hass.services), "async_call", recording_async_call),
        caplog.at_level(logging.INFO),
    ):
        await charger.launch_command(T0, CMD_IDLE)
        time = await _drive(charger, T0 + timedelta(seconds=CYCLE_S), LADDER_WALL_S)
        assert charger.is_uncontrollable is True
        assert len(notify_calls) == 1
        assert "lost control" in notify_calls[0][1]["message"]

        # Superseding while uncontrollable must not re-notify.
        await charger.launch_command(time, copy_command(CMD_ON, power_consign=1761.0))
        time = await _drive(charger, time + timedelta(seconds=CYCLE_S), 600)
        assert len(notify_calls) == 1

        # The device finally complies: one INFO out, and no push at all.
        charger.probe_if_command_set = AsyncMock(return_value=True)
        await _drive(charger, time, 600)

    assert charger.is_uncontrollable is False
    assert _count(caplog, "Lost control of load") == 1
    assert _count(caplog, "Regained control of load") == 1
    assert len(notify_calls) == 1

    await hass.config_entries.async_unload(charger_entry.entry_id)
    await hass.async_block_till_done()


# =============================================================================
# AC10 — QS never touches the user's enable switch
# =============================================================================


async def test_qs_never_flips_the_enable_switch(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """AC10: no auto-disable on the way in, no auto-re-enable on the way out."""
    home = await _get_home(hass, home_config_entry)
    charger, charger_entry = await _add_charger(hass, "charger_ac10")
    _make_never_ack(charger)

    home.remove_device = MagicMock(wraps=home.remove_device)
    home.add_disabled_device = MagicMock(wraps=home.add_disabled_device)
    home.remove_disabled_device = MagicMock(wraps=home.remove_disabled_device)

    await charger.launch_command(T0, CMD_IDLE)
    time = await _drive(charger, T0 + timedelta(seconds=CYCLE_S), LADDER_WALL_S)
    assert charger.is_uncontrollable is True
    assert charger.qs_enable_device is True

    charger.probe_if_command_set = AsyncMock(return_value=True)
    await _drive(charger, time, 600)
    assert charger.is_uncontrollable is False

    home.remove_device.assert_not_called()
    home.add_disabled_device.assert_not_called()
    home.remove_disabled_device.assert_not_called()

    await hass.config_entries.async_unload(charger_entry.entry_id)
    await hass.async_block_till_done()


async def test_manually_disabled_load_is_never_auto_re_enabled(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """AC10: `qs_enable_device` stays the user's manual off-switch for retries."""
    home = await _get_home(hass, home_config_entry)
    charger, charger_entry = await _add_charger(hass, "charger_ac10b")
    _make_never_ack(charger)

    await charger.launch_command(T0, CMD_IDLE)
    time = await _drive(charger, T0 + timedelta(seconds=CYCLE_S), LADDER_WALL_S)
    assert charger.is_uncontrollable is True

    charger.qs_enable_device = False
    # Disabling calls `reset()`, which empties the command slot — so there is
    # nothing in flight and the load is no longer "uncontrollable".
    assert charger.running_command is None
    assert charger.is_uncontrollable is False

    await _drive(charger, time, LADDER_WALL_S)
    assert charger.qs_enable_device is False
    assert charger.is_uncontrollable is False
    assert charger.execute_command.await_count == 1 + NUM_MAX_COMMAND_RELAUNCH

    await hass.config_entries.async_unload(charger_entry.entry_id)
    await hass.async_block_till_done()


# =============================================================================
# AC11 — the qs_load_uncontrollable binary sensor
# =============================================================================


async def test_uncontrollable_binary_sensor_is_created_for_a_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """AC11: the sensor exists for a load and reads `is_uncontrollable`."""
    await _get_home(hass, home_config_entry)
    charger, charger_entry = await _add_charger(hass, "charger_ac11")

    entities = create_ha_binary_sensor(charger)
    matches = [e for e in entities if e.entity_description.key == BINARY_SENSOR_LOAD_UNCONTROLLABLE]
    assert len(matches) == 1

    description = matches[0].entity_description
    # It must NOT rely on the `getattr(device, key, False)` fallback: `key` is
    # the translation key, not the property name.
    assert description.value_fn is not None
    assert getattr(charger, description.key, "missing") == "missing"

    assert description.value_fn(charger, description.key) is False
    charger.unresponsive_since = T0
    charger.running_command = copy_command(CMD_IDLE)
    assert description.value_fn(charger, description.key) is True

    await hass.config_entries.async_unload(charger_entry.entry_id)
    await hass.async_block_till_done()


async def test_uncontrollable_binary_sensor_is_absent_for_home_and_battery(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """AC11: `QSHome` and `QSBattery` are not loads and get no such sensor."""
    home = await _get_home(hass, home_config_entry)

    battery_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_BATTERY_CONFIG,
        entry_id="battery_ac11",
        title=f"battery: {MOCK_BATTERY_CONFIG['name']}",
        unique_id="qs_battery_ac11",
    )
    battery_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(battery_entry.entry_id)
    await hass.async_block_till_done()
    battery = hass.data[DOMAIN][battery_entry.entry_id]

    for device in (home, battery):
        keys = [e.entity_description.key for e in create_ha_binary_sensor(device)]
        assert BINARY_SENSOR_LOAD_UNCONTROLLABLE not in keys

    await hass.config_entries.async_unload(battery_entry.entry_id)
    await hass.async_block_till_done()


# =============================================================================
# AC14b — the driver holds no ladder arithmetic
# =============================================================================


def test_check_loads_commands_is_a_thin_driver():
    """AC14b: the relaunch cadence is decided in the pure layer, not here."""
    source = inspect.getsource(QSHome.check_loads_commands)

    assert "50" not in source
    assert "6" not in source
    assert "force_relaunch_command" not in source
    assert "check_and_relaunch_command" in source

    # The delay itself is derived from `command_relaunch_delay_s()`, one call
    # away, in `home_model/load.py`.
    lifecycle = inspect.getsource(AbstractDevice.check_and_relaunch_command) + inspect.getsource(
        AbstractDevice._relaunch_stale_command
    )
    assert "command_relaunch_delay_s()" in lifecycle


# =============================================================================
# Helpers
# =============================================================================


def _count(caplog: pytest.LogCaptureFixture, needle: str) -> int:
    """Count log records whose formatted message contains `needle`."""
    return len([r for r in caplog.records if needle in r.getMessage()])
