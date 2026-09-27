"""QS-376 regression tests.

After an HA restart, lbbrhzn/ocpp v0.12.0 held `switch.<cpid>_charge_control`
unavailable on `wallbox 2 parking` (a StopTransaction it could not attribute). QS
then ran out of start retries on that charger and, because the budgeting group
returns no actionable chargers while any member is not in its expected state, no
other charger of the group was ever budgeted again: no car charged all afternoon.

Tests 1-3 use a generic stuck start (the switch accepts `turn_on`, the charger just
never reports enabled); tests 4-5 use the production OCPP state (switch
`unavailable`, status `Finishing`).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz

from custom_components.quiet_solar.ha_model.charger import (
    CHARGER_FAULT_NOTIFY_DEBOUNCE_S,
    TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_OFF_TO_ON_S,
    QSChargerStatus,
)
from custom_components.quiet_solar.home_model.commands import CMD_AUTO_GREEN_ONLY, copy_command
from tests.test_charger_coverage_deep import (
    _create_charger,
    _create_ocpp_charger,
    _make_charger_group,
    _make_hass,
    _make_home,
    _make_real_car,
)

# first start attempt of the production log (14:05:34 local)
T0 = datetime(2026, 9, 27, 12, 5, 34, tzinfo=pytz.UTC)
STEP = timedelta(seconds=7)

STUCK_NAME = "wallbox 2 parking"
DEV = "wallbox_2_parking"
SWITCH = f"switch.{DEV}_charge_control"
STATUS = f"sensor.{DEV}_status_connector"

# 7 s grid: launches every 91 s (13 cycles > CHARGER_START_STOP_RETRY_S = 90)
LAUNCH_OFFSETS_S = [0, 91, 182, 273]
FOURTH_LAUNCH = T0 + timedelta(seconds=273)
# first 7 s cycle >= 4th launch + 15 min (T0 + 19:33)
REARM = T0 + timedelta(seconds=1176)


class _States:
    """Per-entity HA state store behind `hass.states.get`."""

    def __init__(self):
        self.values: dict[str, SimpleNamespace] = {}

    def set(self, entity_id: str, value: str, t: datetime) -> None:
        self.values[entity_id] = SimpleNamespace(entity_id=entity_id, state=value, last_updated=t, attributes={})

    def remove(self, entity_id: str) -> None:
        self.values.pop(entity_id, None)

    def get(self, entity_id):
        return self.values.get(entity_id)


def _build_stuck_charger(switch_state: str, status: str = "Finishing", hass=None, home=None, states=None):
    """Real QSChargerOCPP 'wallbox 2 parking' with a real car, stuck wanting to start."""
    if hass is None:
        hass = _make_hass()
    if home is None:
        home = _make_home()
        home.async_notify_all_mobile_apps = AsyncMock()
    if states is None:
        states = _States()
        hass.states.get = MagicMock(side_effect=states.get)

    charger = _create_ocpp_charger(hass, home, name=STUCK_NAME)
    car = _make_real_car(hass, home, name="ID.buzz")

    t_seed = T0 - timedelta(minutes=10)
    states.set(STATUS, status, t_seed)
    states.set(SWITCH, switch_state, t_seed)
    # plug-probe history >= CHARGER_CHECK_STATE_WINDOW_S of a plugged, not-enabled value
    charger.add_to_history(STATUS, t_seed)
    charger.add_to_history(charger._internal_fake_is_plugged_id, t_seed)
    charger.attach_car(car, t_seed)

    charger.current_command = copy_command(CMD_AUTO_GREEN_ONLY)
    charger.running_command = None
    charger.qs_enable_device = True
    charger._boot_time = None
    charger._asked_for_reboot_at_time = None

    t_idle = T0 - timedelta(minutes=2)
    charger._expected_charge_state.set(False, t_idle)
    charger._expected_amperage.set(6, t_idle)
    charger._expected_num_active_phases.set(1, t_idle)
    charger._expected_charge_state.set(True, T0)
    return hass, home, states, charger


def _make_healthy(hass, home):
    healthy = _create_charger(hass, home, name="healthy")
    cs_healthy = QSChargerStatus(healthy)
    healthy.is_charger_faulted = MagicMock(return_value=False)
    healthy.ensure_correct_state = AsyncMock(return_value=(True, False, T0))
    healthy.get_stable_dynamic_charge_status = MagicMock(return_value=cs_healthy)
    return healthy, cs_healthy


def _turn_on_calls(hass) -> int:
    return sum(1 for c in hass.services.async_call.await_args_list if c.kwargs.get("service") == "turn_on")


def _status_nudges(hass) -> int:
    return sum(
        1
        for c in hass.services.async_call.await_args_list
        if c.args[:2] == ("ocpp", "trigger_custom_message")
        and c.args[2].get("requested_message") == "StatusNotification"
    )


def _reserved(charger, amps: int) -> list[float]:
    ret = [0.0, 0.0, 0.0]
    ret[charger.mono_phase_index] = amps
    return ret


async def _drive_group_with_healthy(order_stuck_first: bool):
    hass, home, _states, stuck = _build_stuck_charger(switch_state="off")
    healthy, cs_healthy = _make_healthy(hass, home)
    members = [stuck, healthy] if order_stuck_first else [healthy, stuck]
    group = _make_charger_group(home, members)
    assert stuck.is_charger_faulted(T0) is False  # V-generic: F3 cannot flag this one

    t = T0
    end = T0 + timedelta(minutes=60)
    while t <= end:
        actionable, _ = await group.ensure_correct_state(t)
        if t >= FOURTH_LAUNCH:
            assert cs_healthy in actionable, f"healthy starved at {t - T0}"
            stuck_cs = [cs for cs in actionable if cs.charger is stuck]
            if t < REARM:
                assert stuck.is_start_stuck(t) is True
                assert stuck_cs == [], f"stuck charger not isolated at {t - T0}"
                assert group._isolated_reserved_amps == _reserved(stuck, 6)
            else:
                assert len(stuck_cs) == 1, f"stuck charger did not rejoin at {t - T0}"
                assert stuck_cs[0].current_real_max_charging_amp == 0
                assert group._isolated_reserved_amps == [0.0, 0.0, 0.0]
        t += STEP


@pytest.mark.asyncio
async def test_stuck_start_charger_does_not_starve_group():
    """QS-376 F1: a start-stuck charger iterated first must not starve the group."""
    await _drive_group_with_healthy(order_stuck_first=True)


@pytest.mark.asyncio
async def test_stuck_start_charger_does_not_starve_group_when_iterated_last():
    """QS-376 F1: same, with the stuck charger iterated last."""
    await _drive_group_with_healthy(order_stuck_first=False)


@pytest.mark.asyncio
async def test_stuck_start_rearms_through_the_group():
    """QS-376 F2: exhausted start retries re-arm through the group, never outside the budget."""
    hass, home, _states, stuck = _build_stuck_charger(switch_state="off")
    group = _make_charger_group(home, [stuck])

    launches = []
    rearmed_at = None
    t = T0
    end = T0 + timedelta(minutes=60)
    spacing = timedelta(seconds=TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_OFF_TO_ON_S)
    while t <= end:
        before = _turn_on_calls(hass)
        res, handled, _ = await stuck.ensure_correct_state(t)
        if _turn_on_calls(hass) > before:
            launches.append(int((t - T0).total_seconds()))

        if t < REARM:
            assert stuck._expected_charge_state.value is True, f"re-armed too early at {t - T0}"
        elif rearmed_at is None:
            # (b) the re-arm cycle settles in the same call
            assert stuck._expected_charge_state.value is False
            assert res is True
            assert handled is False
            rearmed_at = t
            # (c) it rejoins the group as an idle member right away
            actionable, _ = await group.ensure_correct_state(t)
            assert [cs.charger for cs in actionable] == [stuck]
            assert actionable[0].current_real_max_charging_amp == 0

        if rearmed_at is not None:
            # (d) the off->on spacing is enforced before a new start can be budgeted
            cs = stuck.get_stable_dynamic_charge_status(t)
            if t < rearmed_at + spacing:
                assert cs.possible_amps == [0], f"start allowed too early at {t - T0}"
            else:
                assert 6 in cs.possible_amps
        t += STEP

    # (a) exactly the 4 budgeted launches, none outside the budget afterwards
    assert launches == LAUNCH_OFFSETS_S
    assert rearmed_at == REARM

    # (d) a new start comes only from the group budget, with a fresh retry counter
    cs = stuck.get_stable_dynamic_charge_status(t)
    cs.budgeted_amp = 6
    cs.budgeted_num_phases = 1
    t_apply = t
    before = _turn_on_calls(hass)
    await group.apply_budgets([cs], [cs], t_apply)
    assert stuck._expected_charge_state.value is True
    assert stuck._expected_charge_state.can_launch() is True
    new_launches = [0] if _turn_on_calls(hass) > before else []

    t = t_apply + STEP
    while t <= t_apply + timedelta(minutes=10):
        before = _turn_on_calls(hass)
        await stuck.ensure_correct_state(t)
        if _turn_on_calls(hass) > before:
            new_launches.append(int((t - t_apply).total_seconds()))
        t += STEP

    assert new_launches == LAUNCH_OFFSETS_S
    assert stuck.is_start_stuck(t) is True


@pytest.mark.asyncio
async def test_ocpp_unavailable_charge_control_is_a_fault():
    """QS-376 F3: an unavailable OCPP charge_control on a plugged, idle charger is a fault."""
    hass, home, states, charger = _build_stuck_charger(switch_state="unavailable")
    charger.on_device_state_change = AsyncMock()
    # keep the plugged car attached (car selection is not under test here)
    charger.get_best_car = MagicMock(return_value=charger.car)

    notified_at = []
    t = T0
    while t <= T0 + timedelta(minutes=3):
        assert charger.is_charger_faulted(t) is True, f"not faulted at {t - T0}"
        before = home.async_notify_all_mobile_apps.await_count
        await charger.check_load_activity_and_constraints(t)
        if home.async_notify_all_mobile_apps.await_count > before:
            notified_at.append(int((t - T0).total_seconds()))
        t += STEP

    # first 7 s cycle >= the QS-346 debounce
    first_ok = -(-CHARGER_FAULT_NOTIFY_DEBOUNCE_S // 7) * 7
    assert notified_at == [first_ok]
    title, message = home.async_notify_all_mobile_apps.await_args.args
    assert message == (
        f"{STUCK_NAME}: charge control unavailable, ID.buzz cannot be started — unplug and replug the car"
    )

    # switch available again -> no fault
    states.set(SWITCH, "off", t)
    assert charger.is_charger_faulted(t) is False
    # switch entity absent (status sensor still present) -> no fault
    states.remove(SWITCH)
    assert charger.is_charger_faulted(t) is False
    # charging -> never leaves the group, even with the switch unavailable
    states.set(SWITCH, "unavailable", t)
    states.set(STATUS, "Charging", t)
    assert charger.is_charger_faulted(t) is False
    # no pause/resume switch configured -> no fault
    states.set(STATUS, "Finishing", t)
    assert charger.is_charger_faulted(t) is True
    charger.charger_pause_resume_switch = None
    assert charger.is_charger_faulted(t) is False


@pytest.mark.asyncio
async def test_ocpp_fault_message_without_car_and_on_plain_fault():
    """QS-376 F3: no-car OCPP text, and a real Faulted status keeps the QS-346 text."""
    hass, home, states, charger = _build_stuck_charger(switch_state="unavailable")
    assert charger._charger_fault_message(T0, "Finishing", None) == (
        f"{STUCK_NAME}: charge control unavailable — please unplug and replug the car"
    )
    states.set(STATUS, "Faulted", T0)
    msg = charger._charger_fault_message(T0, "Faulted", charger.car)
    assert msg == (
        f"{STUCK_NAME} is in error (Faulted) and cannot charge. Please go unplug and replug ID.buzz on {STUCK_NAME}."
    )


@pytest.mark.asyncio
async def test_ocpp_unavailable_charge_control_sends_status_nudge():
    """QS-376 F4: a connector-less StatusNotification nudge, at most every 5 min."""
    hass, home, states, charger = _build_stuck_charger(switch_state="unavailable")
    charger.on_device_state_change = AsyncMock()

    nudged_at = []
    t = T0
    while t <= T0 + timedelta(minutes=12):
        before = _status_nudges(hass)
        await charger.check_load_activity_and_constraints(t)
        if _status_nudges(hass) > before:
            nudged_at.append(int((t - T0).total_seconds()))
        t += STEP

    assert nudged_at == [0, 301, 602]
    nudge = next(
        c
        for c in hass.services.async_call.await_args_list
        if c.args[:2] == ("ocpp", "trigger_custom_message")
        and c.args[2].get("requested_message") == "StatusNotification"
    )
    assert nudge.args[2] == {"devid": STUCK_NAME, "requested_message": "StatusNotification"}

    # recovery resets the rate limit: the next detection nudges immediately
    states.set(SWITCH, "off", t)
    await charger.check_load_activity_and_constraints(t)
    assert charger._last_status_nudge_time is None
    states.set(SWITCH, "unavailable", t + STEP)
    before = _status_nudges(hass)
    await charger.check_load_activity_and_constraints(t + STEP)
    assert _status_nudges(hass) == before + 1


@pytest.mark.asyncio
async def test_ocpp_no_status_nudge_when_control_available_or_plain_fault():
    """QS-376 F4: no nudge when the switch is available, nor on a plain Faulted status."""
    for status in ("Finishing", "Faulted"):
        hass, home, states, charger = _build_stuck_charger(switch_state="off", status=status)
        charger.on_device_state_change = AsyncMock()
        t = T0
        while t <= T0 + timedelta(minutes=6):
            await charger.check_load_activity_and_constraints(t)
            t += STEP
        assert _status_nudges(hass) == 0


@pytest.mark.asyncio
async def test_ocpp_status_nudge_failure_is_logged_not_raised():
    """QS-376 F4: a failing nudge service call must not break the cycle."""
    hass, home, states, charger = _build_stuck_charger(switch_state="unavailable")

    async def _boom(*args, **kwargs):
        if args[:2] == ("ocpp", "trigger_custom_message") and args[2].get("requested_message") == "StatusNotification":
            raise RuntimeError("ocpp down")

    hass.services.async_call = AsyncMock(side_effect=_boom)
    await charger._on_charger_fault_cycle(T0)
    assert charger._last_status_nudge_time == T0


@pytest.mark.asyncio
async def test_isolated_charger_amps_are_reserved_in_group_current_checks():
    """QS-376 F1: while a member is isolated, every group current check reserves its amps."""
    hass, home, _states, stuck = _build_stuck_charger(switch_state="off")
    healthy, _cs_healthy = _make_healthy(hass, home)
    group = _make_charger_group(home, [stuck, healthy])
    dg = group.dynamic_group

    t = T0
    while t <= FOURTH_LAUNCH:
        await group.ensure_correct_state(t)
        t += STEP
    reserved = _reserved(stuck, 6)
    assert group._isolated_reserved_amps == reserved

    group._is_current_acceptable([1, 2, 3], [4, 5, 6], t)
    assert dg.is_current_acceptable.call_args.kwargs == {
        "new_amps": [1 + reserved[0], 2 + reserved[1], 3 + reserved[2]],
        "estimated_current_amps": [4 + reserved[0], 5 + reserved[1], 6 + reserved[2]],
        "time": t,
    }
    # an unknown estimate stays unknown
    group._is_current_acceptable_and_diff([1, 2, 3], None, t)
    assert dg.is_current_acceptable_and_diff.call_args.kwargs == {
        "new_amps": [1 + reserved[0], 2 + reserved[1], 3 + reserved[2]],
        "estimated_current_amps": None,
        "time": t,
    }

    # no isolated member: amps pass through untouched
    group._isolated_reserved_amps = [0.0, 0.0, 0.0]
    new_amps = [1, 2, 3]
    group._is_current_acceptable(new_amps, None, t)
    assert dg.is_current_acceptable.call_args.kwargs["new_amps"] is new_amps


async def _isolate_stuck_in_group():
    hass, home, states, stuck = _build_stuck_charger(switch_state="off")
    healthy, cs_healthy = _make_healthy(hass, home)
    group = _make_charger_group(home, [stuck, healthy])
    t = T0
    while t <= FOURTH_LAUNCH:
        await group.ensure_correct_state(t)
        t += STEP
    assert group._isolated_reserved_amps == _reserved(stuck, 6)
    return hass, home, states, stuck, group, cs_healthy, t


@pytest.mark.asyncio
async def test_reservation_never_prevents_a_current_reduction():
    """QS-376 review EC1: if shaving cannot fit with the reservation, reductions win."""
    _hass, _home, _states, _stuck, group, cs_healthy, t = await _isolate_stuck_in_group()

    seen_reserved = []

    async def _shave(actionable, do_reset_allocation, time):
        seen_reserved.append(list(group._isolated_reserved_amps))
        return actionable, group._isolated_reserved_amps == [0.0, 0.0, 0.0], False

    group._do_prepare_and_shave_budgets = AsyncMock(side_effect=_shave)
    group.get_budget_diffs = MagicMock(return_value=(0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
    res = await group.budgeting_algorithm_minimize_diffs([], 0.0, 0.0, False, t)

    assert res[0] is not False
    assert len(seen_reserved) == 3
    assert seen_reserved[0] == seen_reserved[1] != [0.0, 0.0, 0.0]
    assert seen_reserved[2] == [0.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_shave_failure_without_reservation_still_fails():
    """QS-376 review EC1: with nothing reserved the failure path is unchanged (no extra retry)."""
    hass, home, _states, _stuck = _build_stuck_charger(switch_state="off")
    group = _make_charger_group(home, [])
    group._do_prepare_and_shave_budgets = AsyncMock(return_value=([], False, False))
    res = await group.budgeting_algorithm_minimize_diffs([], None, None, False, T0)
    assert res[0] is False
    assert group._do_prepare_and_shave_budgets.await_count == 2


def test_start_stuck_is_false_in_state_reset():
    """QS-376 review EC4: a charger in state reset is never 'start stuck' (no None amps reserved)."""
    _hass, _home, _states, stuck = _build_stuck_charger(switch_state="off")
    stuck._expected_charge_state._num_launched = 4
    assert stuck.is_start_stuck(T0) is True
    stuck._inner_amperage = None
    assert stuck.is_start_stuck(T0) is False


def test_charge_control_check_without_status_sensor():
    """QS-376 review EC5: no status sensor -> not the held-control condition, no HA lookup of None."""
    hass, _home, _states, charger = _build_stuck_charger(switch_state="unavailable")
    assert charger._is_charge_control_unavailable_while_plugged(T0) is True
    charger.charger_status_sensor = None
    assert charger._is_charge_control_unavailable_while_plugged(T0) is False
    assert all(c.args != (None,) for c in hass.states.get.call_args_list)


@pytest.mark.asyncio
async def test_group_probe_does_not_claim_the_start_stuck_info_line():
    """QS-376 review EC7: a group-level probe logs the isolation at DEBUG, not via the INFO throttle."""
    _hass, _home, _states, stuck, group, _cs, t = await _isolate_stuck_in_group()
    group._log_on_change_state = None
    await group.ensure_correct_state(t, probe_only=True)
    assert group._log_on_change_state is None or f"start_stuck:{stuck.name}" not in group._log_on_change_state
    assert group._isolated_reserved_amps == _reserved(stuck, 6)
