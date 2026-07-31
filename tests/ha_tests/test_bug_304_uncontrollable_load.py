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

from custom_components.quiet_solar.const import (
    DATA_HANDLER,
    DOMAIN,
)
from custom_components.quiet_solar.ha_model.home import QSHome, QSHomeMode
from custom_components.quiet_solar.home_model.commands import CMD_IDLE, CMD_ON, copy_command
from custom_components.quiet_solar.home_model.load import (
    NUM_MAX_COMMAND_RELAUNCH,
    AbstractDevice,
)
from tests.factories import (
    AlwaysReplansLoad,
    NeverAcksLoad,
    RaisingCheckLoad,
    attach_minimal_load_to_home,
)
from tests.qs304_helpers import CYCLE_S, LADDER_WALL_S, count_log

from .const import MOCK_CHARGER_CONFIG

pytestmark = pytest.mark.usefixtures("mock_sensor_states")

T0 = datetime(2026, 7, 27, 12, 12, 19, tzinfo=pytz.UTC)


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


def test_is_load_command_set_truth_table_is_unmodified():
    """AC7: widening `is_load_command_set` would move the persisted forecast.

    Review fix #01/15: this used to assert `inspect.getsource(...) == "<hardcoded
    body>"`, which broke on any reformat or comment while letting a real semantic
    change through if it happened to reformat identically. The behavioural truth
    table over the three inputs it actually reads is both stricter and stable.
    """
    load = NeverAcksLoad(name="accounting_probe")
    time = T0

    # (enabled, running_command, current_command, expected)
    cases = [
        (True, None, None, False),
        (True, None, CMD_ON, True),
        (True, CMD_IDLE, None, False),
        (True, CMD_IDLE, CMD_ON, False),
        (False, None, CMD_ON, False),
        (False, CMD_IDLE, CMD_ON, False),
    ]

    for enabled, running, current, expected in cases:
        load._enabled = enabled
        load.running_command = None if running is None else copy_command(running)
        load.current_command = None if current is None else copy_command(current)
        assert load.is_load_command_set(time) is expected, (enabled, running, current)


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
    """AC7: crossing the lost-control threshold moves no accounting value.

    Review fix #01/11 — provenance of the pinned literals. They were verified
    against pre-QS-304 code during implementation: the production diff was stashed
    and the equivalent probe run against `main`'s `_ack_command(time, None)` +
    spent-ladder state (`running_command_num_relaunch = 7`), which produced the
    same five values — `is_load_command_set False`, `is_user_overridden False`,
    `command_power_state None`, `device_power 0.0`, `group_power 0.0`. That run was
    throwaway, so the temporal claim is not reconstructible from branch history.

    What IS structurally guaranteed here, and is the durable part of the pin: all
    five values are forced by `is_load_command_set` returning False whenever
    `running_command is not None`, and that body is unmodified (pinned by
    `test_is_load_command_set_truth_table_is_unmodified`). The before/after
    comparison below is therefore the load-bearing assertion; the literals are a
    secondary guard against both sides moving together.
    """
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
# AC8 — an unconfirmed load blocks the off-grid switch
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

    # And the same holds for a load whose cycle RAISES rather than reporting
    # not-good: it is equally unconfirmed, so it must equally hold the switch back.
    # `attach_minimal_load_to_home` appends, so the uncontrollable charger stays in
    # the sweep and this assertion keeps covering it too. Clobbering `_all_loads`
    # here also broke the charger-entry teardown.
    attach_minimal_load_to_home(home, name="raising_load", load_class=RaisingCheckLoad)
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
    assert count_log(caplog, "Lost control of load") == 1
    assert count_log(caplog, "Lost-control state cleared for load") == 1
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
# Review fix #01/1 + #01/2 — the driver's fault isolation and the force-solve
# =============================================================================


async def test_one_broken_load_does_not_stop_the_sweep(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Review fix #01/1: `check_loads_commands` isolates a per-load failure.

    Story §5 keeps "the `all_ok` aggregation and the per-load `try/except`" as a
    behaviour. Before this test the branch was only ever reached by accident, via
    the swallowed `TypeError` from a `MagicMock` load — so task 7's conversion to
    real loads turned the R1 hazard into a coverage hole.
    """
    home = await _get_home(hass, home_config_entry)
    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home._init_completed = True
    home.physical_battery = None

    broken = attach_minimal_load_to_home(home, name="broken_load", load_class=RaisingCheckLoad)
    healthy = attach_minimal_load_to_home(home, name="healthy_load")
    healthy.check_and_relaunch_command = AsyncMock(wraps=healthy.check_and_relaunch_command)
    # Order matters: the broken load must come first to prove the sweep continues.
    home._all_loads = [broken, healthy]

    with caplog.at_level(logging.ERROR):
        all_ok = await home.check_loads_commands(T0)

    # (a) the sweep continued past the broken load
    healthy.check_and_relaunch_command.assert_awaited_once()
    # (b) the failure was logged, naming the load
    assert count_log(caplog, "check_loads_commands: Error checking load commands broken_load") == 1
    # (c) a load whose cycle raised is NOT confirmed-good, so it falsifies `all_ok`.
    #     Reporting True here let `finish_off_grid_switch` complete the transition
    #     while that load's command had never landed, violating AC8 for exactly the
    #     device whose probe raises.
    assert all_ok is False


async def test_a_load_reporting_a_constraint_change_forces_a_solve(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Review fix #01/2: `update_live_constraints` returning True forces a re-plan.

    Collateral from the de-Mock conversion: every real double returned `False`, so
    nothing drove `do_force_solve = True` any more. The line is untouched by this
    story's diff — the coverage loss was introduced by the test rewrite.
    """
    home = await _get_home(hass, home_config_entry)
    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home._init_completed = True
    home._switch_to_off_grid_launched = None
    home.physical_battery = None
    home.finish_setup = AsyncMock(return_value=True)
    home.update_loads_constraints = AsyncMock()
    home.compute_non_controlled_forecast = AsyncMock(return_value=[])
    home.get_solar_from_current_forecast = MagicMock(return_value=[])

    load = attach_minimal_load_to_home(
        home, name="replanning_load", time=T0, with_constraint=True, load_class=AlwaysReplansLoad
    )
    home._chargers = []
    home._commands = []
    home._battery_commands = []
    # A recent solve, so ONLY the constraint change can trigger the re-plan.
    home._last_solve_done = T0 - timedelta(seconds=30)

    solver = MagicMock()
    solver.solve = MagicMock(return_value=([], []))
    with patch("custom_components.quiet_solar.ha_model.home.PeriodSolver", return_value=solver):
        await home.update_loads(T0)

    solver.solve.assert_called_once()
    assert home._last_solve_done == T0
    assert load.is_load_active(T0) is True


# =============================================================================
# AC14b — the driver holds no ladder arithmetic
# =============================================================================


def test_check_loads_commands_is_a_thin_driver():
    """AC14b: the relaunch cadence is decided in the pure layer, not here.

    Review fix #01/15: the digit substring checks (`"50"`, `"6"`) were brittle —
    any line number, count or unrelated literal in a comment could trip them. The
    meaningful assertion is the absence of the ladder *symbols* from the driver.
    """
    source = inspect.getsource(QSHome.check_loads_commands)
    # Comments are prose about the design and may legitimately name the pure-layer
    # helpers; the assertion is about the executable code.
    code = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))

    for ladder_symbol in (
        "COMMAND_RELAUNCH_BASE_DELAY_S",
        "NUM_MAX_COMMAND_RELAUNCH",
        AbstractDevice.command_relaunch_delay_s.__name__,
        AbstractDevice.force_relaunch_command.__name__,
        "running_command_num_relaunch",
    ):
        assert ladder_symbol not in code, ladder_symbol

    assert AbstractDevice.check_and_relaunch_command.__name__ in code

    # The delay itself is derived from `command_relaunch_delay_s()`, one call
    # away, in `home_model/load.py`.
    lifecycle = inspect.getsource(AbstractDevice.check_and_relaunch_command) + inspect.getsource(
        AbstractDevice._relaunch_stale_command
    )
    assert f"{AbstractDevice.command_relaunch_delay_s.__name__}()" in lifecycle


# =============================================================================
# Helpers
# =============================================================================
