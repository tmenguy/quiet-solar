"""QS-304: HA-layer behaviour when QS loses control of a load.

Covers the parts of the story that only exist above the domain boundary: the
mobile push, the absence of collateral damage on power accounting, the off-grid
contract, and the fact that QS never touches the user's `qs_enable_device`
switch.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from collections.abc import Iterator
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz
import slugify
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME, Platform
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.binary_sensor import create_ha_binary_sensor
from custom_components.quiet_solar.const import (
    BINARY_SENSOR_LOAD_LOST_CONTROL,
    CONF_TYPE_NAME_QSChargerGeneric,
    DATA_HANDLER,
    DEVICE_STATUS_CHANGE_CONSTRAINT,
    DOMAIN,
    NOTIFICATION_TAG_LOST_CONTROL_PREFIX,
)
from custom_components.quiet_solar.ha_model.battery import QSBattery
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
from tests.qs304_helpers import CYCLE_S, LADDER_WALL_S, LOST_CONTROL_LOG, count_log

from .const import MOCK_BATTERY_CONFIG, MOCK_CHARGER_CONFIG

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


@contextlib.contextmanager
def _recording_notify(hass: HomeAssistant) -> Iterator[list[tuple[str, dict | None]]]:
    """Capture every `Platform.NOTIFY` call as `(service, service_data)`.

    Everything else still reaches the real service registry, so a test that also
    exercises the load-management cycle is not otherwise sandboxed.
    """
    notify_calls: list[tuple[str, dict | None]] = []
    original_async_call = hass.services.async_call

    async def recording_async_call(self, domain, service, *args, **kwargs):
        if domain == Platform.NOTIFY:
            notify_calls.append((service, kwargs.get("service_data")))
            return None
        return await original_async_call(domain, service, *args, **kwargs)

    with patch.object(type(hass.services), "async_call", recording_async_call):
        yield notify_calls


def _expected_lost_control_tag(device_id: str) -> str:
    """Return the collapsing tag a lost-control push for `device_id` must carry."""
    return f"{NOTIFICATION_TAG_LOST_CONTROL_PREFIX}{device_id}"


# =============================================================================
# AC7 — no collateral damage on power accounting
# =============================================================================


def test_is_load_command_set_truth_table_is_unmodified():
    """AC7: widening `is_load_command_set` would move the persisted forecast.

    Review fix #01/15: this used to assert `inspect.getsource(...) == "<hardcoded
    body>"`, which broke on any reformat or comment while letting a real semantic
    change through if it happened to reformat identically. The behavioural truth
    table over the three inputs it actually reads is both stricter and stable.

    QS-307 widened the table with an `unresponsive_since` column. That story moved
    this predicate INWARD in `check_load_activity_and_constraints` — from the whole
    user-override block down to detection only — and the tempting shortcut was to
    make it lost-control-aware instead. It must stay indifferent: it is read by
    three power-accounting call sites that reach the persisted consumption
    forecast, so widening it there would move billed energy.
    """
    load = NeverAcksLoad(name="accounting_probe")
    time = T0

    # (enabled, running_command, current_command, unresponsive_since, expected)
    cases = [
        (True, None, None, None, False),
        (True, None, None, T0, False),
        (True, None, CMD_ON, None, True),
        (True, None, CMD_ON, T0, True),
        (True, CMD_IDLE, None, None, False),
        (True, CMD_IDLE, None, T0, False),
        (True, CMD_IDLE, CMD_ON, None, False),
        (True, CMD_IDLE, CMD_ON, T0, False),
        (False, None, CMD_ON, None, False),
        (False, None, CMD_ON, T0, False),
        (False, CMD_IDLE, CMD_ON, None, False),
        (False, CMD_IDLE, CMD_ON, T0, False),
    ]

    for enabled, running, current, unresponsive_since, expected in cases:
        load._enabled = enabled
        load.running_command = None if running is None else copy_command(running)
        load.current_command = None if current is None else copy_command(current)
        load.unresponsive_since = unresponsive_since
        context = (enabled, running, current, unresponsive_since)
        assert load.is_load_command_set(time) is expected, context


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

    with (
        _recording_notify(hass) as notify_calls,
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
    assert count_log(caplog, LOST_CONTROL_LOG) == 1
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
# QS-319 — the push collapses on the phone, and nothing else changes shape
# =============================================================================


async def test_the_lost_control_push_carries_a_stable_collapsing_tag(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """AC9 + AC11: one tag per load, so the series becomes one notification.

    HA's mobile-app notify platform treats `data.tag` as a replace-key on both
    Android and iOS: a notification with the same tag replaces the previous one
    rather than stacking. Without it, ~380 undismissable pushes piled up on the
    phone before the device was fixed.

    This also covers AC11: `QSChargerGeneric.on_device_state_change` selects the
    recipient into locals and then makes a single helper call, so this one case
    exercises the whole of the charger's changed forwarding.

    Note `mobile_app_url` is unset here — the tag must survive on its own, which is
    the case the pre-existing `test_on_device_state_change_with_url` fixture cannot
    reach.
    """
    await _get_home(hass, home_config_entry)
    charger, charger_entry = await _add_charger(hass, "charger_qs319_tag")
    _make_never_ack(charger)
    charger.mobile_app = "mobile_app_test_phone"
    assert charger.car is None
    assert charger.mobile_app_url is None

    with _recording_notify(hass) as notify_calls:
        await charger.launch_command(T0, CMD_IDLE)
        await _drive(charger, T0 + timedelta(seconds=CYCLE_S), LADDER_WALL_S)

    assert charger.is_uncontrollable is True
    assert len(notify_calls) == 1
    service_data = notify_calls[0][1]
    assert service_data is not None
    assert "lost control" in service_data["message"]
    assert service_data["data"] == {"tag": _expected_lost_control_tag(charger.device_id)}

    # The tag is a pure function of the CONFIG-derived `device_id`, never of runtime
    # state, so it is stable across a restart and the replace semantics survive one.
    # Recomputed from `MOCK_CHARGER_CONFIG` rather than read back off the object.
    from_config = (
        f"qs_{slugify.slugify(MOCK_CHARGER_CONFIG[CONF_NAME], separator='_')}_{CONF_TYPE_NAME_QSChargerGeneric}"
    )
    assert service_data["data"]["tag"] == _expected_lost_control_tag(from_config)

    await hass.config_entries.async_unload(charger_entry.entry_id)
    await hass.async_block_till_done()


async def test_the_tag_coexists_with_the_click_through_url(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """AC9: adding the tag must not displace the existing `url` / `clickAction`."""
    await _get_home(hass, home_config_entry)
    charger, charger_entry = await _add_charger(hass, "charger_qs319_tag_url")
    _make_never_ack(charger)
    charger.mobile_app = "mobile_app_test_phone"
    charger.mobile_app_url = "https://ha.local/lovelace/quiet-solar"

    with _recording_notify(hass) as notify_calls:
        await charger.launch_command(T0, CMD_IDLE)
        await _drive(charger, T0 + timedelta(seconds=CYCLE_S), LADDER_WALL_S)

    assert len(notify_calls) == 1
    assert notify_calls[0][1]["data"] == {
        "url": "https://ha.local/lovelace/quiet-solar",
        "clickAction": "https://ha.local/lovelace/quiet-solar",
        "tag": _expected_lost_control_tag(charger.device_id),
    }

    await hass.config_entries.async_unload(charger_entry.entry_id)
    await hass.async_block_till_done()


async def test_an_untagged_notification_keeps_its_exact_shape(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """AC10: no other notification changes shape. Pre-change pin.

    Decision 2 of the story is that ONLY lost-control pushes are tagged. The nested
    `data` dict must therefore stay conditional: creating it unconditionally would
    ship `"data": {}` on every previously-bare notification, a shape change to a
    shared path. This passes before the tag exists and must still pass after.
    """
    await _get_home(hass, home_config_entry)
    charger, charger_entry = await _add_charger(hass, "charger_qs319_shape")
    charger.mobile_app = "mobile_app_test_phone"
    assert charger.mobile_app_url is None

    with _recording_notify(hass) as notify_calls:
        await charger.on_device_state_change(
            T0, DEVICE_STATUS_CHANGE_CONSTRAINT, message="the pool pump will run at 14:00"
        )

    assert len(notify_calls) == 1
    assert notify_calls[0][1] == {
        "title": f"What will happen for {MOCK_CHARGER_CONFIG[CONF_NAME]}?",
        "message": "the pool pump will run at 14:00",
    }

    await hass.config_entries.async_unload(charger_entry.entry_id)
    await hass.async_block_till_done()


async def test_a_charger_with_no_recipient_still_latches_the_episode(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC12: no car, no `mobile_app` — nobody is paged, but the episode still latches.

    Decision 6: the helper only sends when `mobile_app is not None`, so this charger
    announces to nobody and will not re-announce later. Accepted, because the
    `qs_load_lost_control` binary sensor is the surface for exactly this case.

    The ERROR needle is asserted rather than the latch alone, because the QS-307
    invalid-probe give-up ALSO latches and ALSO pushes nothing — only the announce
    branch emits this line, so this pins §2's path specifically.
    """
    await _get_home(hass, home_config_entry)
    charger, charger_entry = await _add_charger(hass, "charger_qs319_norecipient")
    _make_never_ack(charger)
    assert charger.car is None
    assert charger.mobile_app is None

    with _recording_notify(hass) as notify_calls, caplog.at_level(logging.ERROR):
        await charger.launch_command(T0, CMD_IDLE)
        await _drive(charger, T0 + timedelta(seconds=CYCLE_S), LADDER_WALL_S)

    assert notify_calls == []
    assert count_log(caplog, LOST_CONTROL_LOG) == 1
    assert charger.has_unacknowledged_lost_control is True

    await hass.config_entries.async_unload(charger_entry.entry_id)
    await hass.async_block_till_done()


# =============================================================================
# QS-319 — the episode is visible in Home Assistant, not just in the logs
# =============================================================================


def _lost_control_sensor(device):
    """Return the device's `qs_load_lost_control` entity, or None."""
    return device.ha_entities.get(BINARY_SENSOR_LOAD_LOST_CONTROL)


async def _refresh(hass: HomeAssistant, sensor, time: datetime) -> str:
    """Drive one update cycle for `sensor` and return its resulting HA state."""
    # Sync HA `@callback` despite the `async_` name — do not `await` it.
    sensor.async_update_callback(time)
    await hass.async_block_till_done()
    return hass.states.get(sensor.entity_id).state


async def test_the_lost_control_binary_sensor_tracks_the_episode(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """AC13: after the alert, Home Assistant says the device is still broken.

    The push is fire-and-forget; before this the lost-control state was purely
    internal, so once the notification scrolled away there was no entity, no
    dashboard state and no automation hook. The sensor exposes the per-EPISODE
    latch — NOT `is_uncontrollable`, which is per-command and flickers False every
    time the slot empties.
    """
    await _get_home(hass, home_config_entry)
    charger, charger_entry = await _add_charger(hass, "charger_qs319_sensor")
    _make_never_ack(charger)

    sensor = _lost_control_sensor(charger)
    assert sensor is not None
    assert charger.has_unacknowledged_lost_control is False
    assert await _refresh(hass, sensor, T0) == "off"

    # The ladder wall is crossed: the episode is announced and the sensor lights up.
    await charger.launch_command(T0, CMD_IDLE)
    time = await _drive(charger, T0 + timedelta(seconds=CYCLE_S), LADDER_WALL_S)
    assert charger.has_unacknowledged_lost_control is True
    assert await _refresh(hass, sensor, time) == "on"

    # The device finally answers: a real ack ends the episode.
    charger.probe_if_command_set = AsyncMock(return_value=True)
    time = await _drive(charger, time, 600)
    assert charger.has_unacknowledged_lost_control is False
    assert await _refresh(hass, sensor, time) == "off"

    # ...and so does explicit user remediation, even mid-episode. A DIFFERING
    # command: `idle` is now the confirmed state, so re-asking for it is absorbed by
    # the equal-command early return and never reaches the ladder at all.
    charger.probe_if_command_set = AsyncMock(return_value=False)
    await charger.launch_command(time, copy_command(CMD_ON, power_consign=1761.0))
    time = await _drive(charger, time + timedelta(seconds=CYCLE_S), LADDER_WALL_S)
    assert await _refresh(hass, sensor, time) == "on"
    await charger.user_clean_and_reset()
    assert charger.has_unacknowledged_lost_control is False
    assert await _refresh(hass, sensor, time) == "off"

    await hass.config_entries.async_unload(charger_entry.entry_id)
    await hass.async_block_till_done()


async def test_the_lost_control_sensor_is_unavailable_on_a_disabled_load(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """AC13: a disabled load reports `unavailable`, never a reassuring `off`.

    QS was told to leave the load alone, so it has nothing to say about it — and
    `off` would be a claim. A separate instance is used deliberately: §3b makes
    disabling an ACKNOWLEDGEMENT, so a fixture that reused the episode above would
    be asserting the wrong thing.
    """
    await _get_home(hass, home_config_entry)
    charger, charger_entry = await _add_charger(hass, "charger_qs319_disabled")

    sensor = _lost_control_sensor(charger)
    assert sensor is not None
    assert await _refresh(hass, sensor, T0) == "off"

    charger.qs_enable_device = False
    await hass.async_block_till_done()

    assert await _refresh(hass, sensor, T0) == "unavailable"

    await hass.config_entries.async_unload(charger_entry.entry_id)
    await hass.async_block_till_done()


async def test_only_commandable_loads_gain_the_lost_control_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """AC13: a charger gains exactly one; a `Battery` and the `QSHome` gain none.

    `QSChargerGeneric` matched no existing dispatch arm, so it goes from zero binary
    sensors to one. The dispatcher is a series of independent `if isinstance(...)`
    arms that each `extend` — adding the load arm must not turn it into an `elif`
    chain, or devices matching two arms would silently lose entities.
    """
    home = await _get_home(hass, home_config_entry)
    charger, charger_entry = await _add_charger(hass, "charger_qs319_dispatch")

    battery_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_BATTERY_CONFIG,
        entry_id="battery_qs319_dispatch",
        title=f"battery: {MOCK_BATTERY_CONFIG[CONF_NAME]}",
        unique_id="qs_battery_qs319_dispatch",
    )
    battery_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(battery_entry.entry_id)
    await hass.async_block_till_done()
    battery = hass.data[DOMAIN][battery_entry.entry_id]

    charger_keys = [e.entity_description.key for e in create_ha_binary_sensor(charger)]
    assert charger_keys == [BINARY_SENSOR_LOAD_LOST_CONTROL]

    assert isinstance(battery, QSBattery)
    assert create_ha_binary_sensor(battery) == []

    home_keys = [e.entity_description.key for e in create_ha_binary_sensor(home)]
    assert home_keys
    assert BINARY_SENSOR_LOAD_LOST_CONTROL not in home_keys

    await hass.config_entries.async_unload(battery_entry.entry_id)
    await hass.config_entries.async_unload(charger_entry.entry_id)
    await hass.async_block_till_done()


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
