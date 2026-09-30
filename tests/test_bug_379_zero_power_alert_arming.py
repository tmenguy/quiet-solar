"""QS-379: arm the zero-power charger alert from the first start launch.

The "There is no power being delivered to the car ... while charging was expected"
alert needs `_expected_charge_state.last_ping_time_success`. Before QS-379 only
`QSStateCmd.success()` set it, so a start that never succeeds (the QS-376 stuck
start) could never arm the alert and the household got no signal.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.quiet_solar.const import CAR_CHARGE_NO_POWER_ERROR, DEVICE_STATUS_CHANGE_ERROR
from custom_components.quiet_solar.ha_model.charger import CHARGER_CHECK_REAL_POWER_WINDOW_S
from tests.factories import (
    create_charger,
    create_state_cmd,
    make_charger_group,
    make_hass,
    make_home,
    make_real_car,
)
from tests.test_bug_376_stuck_charger_group import (
    REARM,
    STATUS,
    STEP,
    SWITCH,
    T0,
    build_stuck_charger,
)

WINDOW = timedelta(seconds=CHARGER_CHECK_REAL_POWER_WINDOW_S)

# First point on the 7 s grid from `T0` strictly past the 600 s re-check window — the
# first cycle at which the zero-power alert becomes eligible (N3: derived, not literal).
_CYCLES_PAST_WINDOW = int(CHARGER_CHECK_REAL_POWER_WINDOW_S // STEP.total_seconds()) + 1
FIRST_ALERT_AT = T0 + _CYCLES_PAST_WINDOW * STEP

# Substring of the production f-string in
# `QSChargerGeneric.constraint_update_value_callback_soc`:
# f"There is no power being delivered to the car ({self.car.name}) while charging was expected"
NO_POWER_MSG_FRAGMENT = "no power being delivered"


class _SocConstraint:
    """A SOC constraint 30 % below its target, anchored before `T0`."""

    def __init__(self) -> None:
        self.current_value = 50.0
        self.target_value = 80.0
        self.first_value_update = T0 - timedelta(hours=1)
        self.last_value_update = T0 - timedelta(minutes=1)
        self.last_value_change_update = T0 - timedelta(minutes=10)

    def is_constraint_met(self, time: datetime, current_value: float) -> bool:
        return False


def _fake_zero_power_readings(charger, car) -> None:
    """Fake only the sensor readings (SOC flat at 50 %, zero charging power) and the sinks."""
    car.get_car_charge_percent_raw_sensor = MagicMock(return_value=50.0)
    car.is_in_soc_estimation_mode = MagicMock(return_value=False)
    car.is_soc_sensor_distrusted = MagicMock(return_value=False)
    car.is_car_charge_growing = MagicMock(return_value=False)
    car.setup_car_charge_target_if_needed = AsyncMock()
    charger._compute_added_charge_update = MagicMock(return_value=0.0)
    charger.is_charging_power_zero = MagicMock(return_value=True)
    charger.on_device_state_change = AsyncMock()
    charger.charger_group.dyn_handle = AsyncMock()


def _no_power_alerts(charger) -> int:
    return sum(
        1
        for c in charger.on_device_state_change.await_args_list
        if c.kwargs.get("device_change_type") == DEVICE_STATUS_CHANGE_ERROR
        and NO_POWER_MSG_FRAGMENT in c.kwargs.get("message", "")
    )


async def _drive_stuck_to_first_alert(stuck, ct) -> datetime:
    """Drive the stuck fixture on the 7 s grid until the first zero-power alert; return its time."""
    t = T0
    while _no_power_alerts(stuck) == 0:
        await stuck.ensure_correct_state(t)
        await stuck.constraint_update_value_callback_percent_soc(ct, t)
        if t > T0 + WINDOW + 20 * STEP:
            raise AssertionError("no zero-power alert fired within the expected window")
        t += STEP
    return t


# --------------------------------------------------------------------------------------
# AC1 / AC2 — QSStateCmd
# --------------------------------------------------------------------------------------


def test_register_launch_arms_last_ping_on_first_launch():
    """AC1: the first launch arms the reference, retries keep it, a retarget re-arms it."""
    cmd = create_state_cmd()
    t0 = T0
    assert cmd.last_ping_time_success is None

    cmd.register_launch(True, t0)
    assert cmd.last_ping_time_success == t0

    cmd.register_launch(True, t0 + timedelta(seconds=91))
    assert cmd.last_ping_time_success == t0

    t2 = t0 + timedelta(seconds=182)
    cmd.register_launch(False, t2)
    assert cmd.last_ping_time_success == t2

    cmd.set(True, t0 + timedelta(seconds=273))
    assert cmd.last_ping_time_success is None

    t4 = t0 + timedelta(seconds=364)
    cmd.register_launch(True, t4)
    assert cmd.last_ping_time_success == t4


@pytest.mark.parametrize("value", [True, 16], ids=["charge_state", "amperage"])
@pytest.mark.asyncio
async def test_success_after_launch_overwrites_armed_ping(value):
    """AC2: the first success still moves the reference to the success time."""
    cmd = create_state_cmd()
    cmd.register_launch(value, T0)
    t1 = T0 + STEP
    await cmd.success(t1)
    assert cmd.first_time_success == t1
    assert cmd.last_ping_time_success == t1


# --------------------------------------------------------------------------------------
# AC3 — the QS-376 stuck start now raises the alert, once per stuck episode
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stuck_start_raises_zero_power_alert_once_per_episode():
    """AC3: armed at the first launch, alert after 600 s, no second notification after the re-arm."""
    _hass, home, _states, stuck = build_stuck_charger(switch_state="off")
    group = make_charger_group(home, [stuck])
    car = stuck.car
    _fake_zero_power_readings(stuck, car)
    ct = _SocConstraint()
    # A live SOC constraint on the load: this is what makes `is_load_active` True (the
    # real gate `Home` checks before the SOC callback), so S3 can assert the gate itself
    # rather than the `is_charger_faulted` proxy.
    stuck._constraints = [ct]

    first_alert_at = None
    t = T0
    while t < REARM:
        await stuck.ensure_correct_state(t)
        assert stuck._expected_charge_state.value is True
        # not faulted: `is_load_active` (the gate `Home` uses before the SOC callback)
        # stays True, so the callback keeps running every load-management cycle (QS-346).
        assert stuck.is_load_active(t) is True
        assert stuck.is_charger_faulted(t) is False
        # The reference sits at `T0` until the block first runs; once the block has run
        # (at `first_alert_at`) it advances, and the *next* re-check is 600 s later —
        # which lands past REARM, so it never re-notifies this episode.
        assert stuck._expected_charge_state.last_ping_time_success == (
            T0 if first_alert_at is None else first_alert_at
        )

        await stuck.constraint_update_value_callback_percent_soc(ct, t)

        if t <= T0 + WINDOW:
            assert _no_power_alerts(stuck) == 0, f"alerted too early at {t - T0}"
            assert stuck.possible_charge_error_start_time is None
        elif first_alert_at is None:
            first_alert_at = t
            assert _no_power_alerts(stuck) == 1
            assert stuck.possible_charge_error_start_time == t
            assert stuck.get_charge_type()[0] == CAR_CHARGE_NO_POWER_ERROR
            # S4: the alert message names the car
            assert car.name in stuck.on_device_state_change.await_args.kwargs["message"]
            # the block ran: its re-check reference advanced
            assert stuck._expected_charge_state.last_ping_time_success == t
        t += STEP

    assert first_alert_at == FIRST_ALERT_AT

    # F2 re-arm: the target goes back to idle; the error stays visible on the car card.
    # S5: bound the wait so a re-arm regression fails instead of hanging.
    while stuck._expected_charge_state.value is True and t < REARM + 10 * STEP:
        await stuck.ensure_correct_state(t)
        t += STEP
    assert stuck._expected_charge_state.value is not True
    assert stuck.get_charge_type()[0] == CAR_CHARGE_NO_POWER_ERROR

    # The `is_load_active` assertions above needed a live constraint on the load; drop it
    # now so the lightweight dummy does not reach the constraint-iterating group/status
    # helpers exercised by the round-2 re-arm below (the NO_POWER card stays latched).
    stuck._constraints = []

    # second stuck round, started by the group budget
    t += timedelta(minutes=20)
    cs = stuck.get_stable_dynamic_charge_status(t)
    cs.budgeted_amp = 6
    cs.budgeted_num_phases = 1
    t_apply = t
    await group.apply_budgets([cs], [cs], t_apply)
    assert stuck._expected_charge_state.value is True

    round_two_checked = False
    t = t_apply
    while t <= t_apply + WINDOW + 2 * STEP:
        await stuck.ensure_correct_state(t)
        if t == t_apply:
            assert stuck._expected_charge_state.last_ping_time_success == t_apply
        before = stuck._expected_charge_state.last_ping_time_success
        await stuck.constraint_update_value_callback_percent_soc(ct, t)
        if t > t_apply + WINDOW and not round_two_checked:
            # the round-2 check runs (its reference advances) ...
            assert before == t_apply
            assert stuck._expected_charge_state.last_ping_time_success == t
            round_two_checked = True
        assert stuck.get_charge_type()[0] == CAR_CHARGE_NO_POWER_ERROR
        t += STEP

    # ... but does not notify the household a second time
    assert round_two_checked is True
    assert _no_power_alerts(stuck) == 1
    assert stuck.on_device_state_change.await_count == 1


# --------------------------------------------------------------------------------------
# AC3b — the same on a non-OCPP charger
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generic_charger_stuck_start_raises_zero_power_alert():
    """AC3b: a generic charger whose start never takes effect alerts 600 s after the first launch."""
    hass = make_hass()
    home = make_home()
    charger = create_charger(hass, home, name="generic stuck")
    car = make_real_car(hass, home, name="Zoe")
    charger.attach_car(car, T0 - timedelta(minutes=10))
    charger.current_command = MagicMock()
    charger.current_command.is_off_or_idle = MagicMock(return_value=False)
    charger._do_update_charger_state = AsyncMock()
    charger.is_not_plugged = MagicMock(return_value=False)
    charger.is_charge_enabled = MagicMock(return_value=False)
    charger.is_charge_disabled = MagicMock(return_value=True)
    charger.low_level_start_charge = AsyncMock(return_value=True)
    _fake_zero_power_readings(charger, car)
    ct = _SocConstraint()

    await charger.start_charge(T0)
    await charger.start_charge(T0 + timedelta(seconds=91))
    assert charger._expected_charge_state.last_ping_time_success == T0

    await charger.constraint_update_value_callback_percent_soc(ct, T0 + WINDOW)
    assert _no_power_alerts(charger) == 0

    t_alert = T0 + WINDOW + timedelta(seconds=1)
    await charger.constraint_update_value_callback_percent_soc(ct, t_alert)
    assert _no_power_alerts(charger) == 1
    assert charger.on_device_state_change.await_count == 1
    assert charger.possible_charge_error_start_time == t_alert
    assert charger.get_charge_type()[0] == CAR_CHARGE_NO_POWER_ERROR
    # S4: the alert message names the car
    assert car.name in charger.on_device_state_change.await_args.kwargs["message"]


# --------------------------------------------------------------------------------------
# AC4 — a confirmed start keeps the success time as reference
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirmed_start_keeps_success_time_as_reference():
    """AC4: a start confirmed on the next cycle is checked 600 s after the success, as before."""
    _hass, _home, states, charger = build_stuck_charger(switch_state="off")
    car = charger.car
    _fake_zero_power_readings(charger, car)
    ct = _SocConstraint()

    await charger.ensure_correct_state(T0)
    assert charger._expected_charge_state.last_ping_time_success == T0

    t_ok = T0 + STEP
    states.set(SWITCH, "on", t_ok)
    states.set(STATUS, "Charging", t_ok)
    charger.is_charge_enabled = MagicMock(return_value=True)
    charger.is_charge_disabled = MagicMock(return_value=False)
    await charger.ensure_correct_state(t_ok)
    assert charger._expected_charge_state.first_time_success == t_ok
    assert charger._expected_charge_state.last_ping_time_success == t_ok

    await charger.constraint_update_value_callback_percent_soc(ct, t_ok + WINDOW)
    assert _no_power_alerts(charger) == 0
    assert charger.possible_charge_error_start_time is None


# --------------------------------------------------------------------------------------
# S1 — the zero-power latch does not outlive the stuck episode
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_power_latch_clears_when_charge_no_longer_wanted():
    """S1: once QS stops wanting charge for a full window, the latch clears and the card recovers."""
    _hass, _home, _states, stuck = build_stuck_charger(switch_state="off")
    _fake_zero_power_readings(stuck, stuck.car)
    ct = _SocConstraint()

    await _drive_stuck_to_first_alert(stuck, ct)
    assert stuck.possible_charge_error_start_time is not None
    assert stuck.get_charge_type()[0] == CAR_CHARGE_NO_POWER_ERROR

    # QS stops wanting charge and keeps not wanting it for longer than one re-check window
    t = FIRST_ALERT_AT + STEP
    stuck._expected_charge_state.set(False, t)
    end = t + WINDOW + 3 * STEP
    while t <= end:
        await stuck.ensure_correct_state(t)
        t += STEP

    assert stuck._expected_charge_state.value is not True
    assert stuck.possible_charge_error_start_time is None
    assert stuck.get_charge_type()[0] != CAR_CHARGE_NO_POWER_ERROR


@pytest.mark.asyncio
async def test_second_stuck_start_in_same_session_notifies_again():
    """S1: after the latch clears, a genuine later stuck start in the same plug session notifies again."""
    _hass, home, _states, stuck = build_stuck_charger(switch_state="off")
    group = make_charger_group(home, [stuck])
    _fake_zero_power_readings(stuck, stuck.car)
    ct = _SocConstraint()

    await _drive_stuck_to_first_alert(stuck, ct)
    assert _no_power_alerts(stuck) == 1

    # let the episode end (target False for a full window) so the latch clears
    t = FIRST_ALERT_AT + STEP
    stuck._expected_charge_state.set(False, t)
    end = t + WINDOW + 3 * STEP
    while t <= end:
        await stuck.ensure_correct_state(t)
        t += STEP
    assert stuck.possible_charge_error_start_time is None

    # a fresh budgeted start later in the SAME plug session
    t += timedelta(minutes=5)
    cs = stuck.get_stable_dynamic_charge_status(t)
    cs.budgeted_amp = 6
    cs.budgeted_num_phases = 1
    t_apply = t
    await group.apply_budgets([cs], [cs], t_apply)
    assert stuck._expected_charge_state.value is True

    t = t_apply
    while _no_power_alerts(stuck) == 1:
        await stuck.ensure_correct_state(t)
        await stuck.constraint_update_value_callback_percent_soc(ct, t)
        if t > t_apply + WINDOW + 20 * STEP:
            raise AssertionError("second stuck start never re-notified")
        t += STEP

    assert _no_power_alerts(stuck) == 2
    assert stuck.on_device_state_change.await_count == 2


# --------------------------------------------------------------------------------------
# S2 — fault recovery does not trigger an immediate zero-power alert
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("call_order", ["fault_state_first", "callback_first"])
@pytest.mark.asyncio
async def test_fault_recovery_rearms_zero_power_reference(call_order):
    """S2: a charger back from a fault gets its own window; no zero-power alert before recovery + 600 s."""
    hass = make_hass()
    home = make_home()
    home.async_notify_all_mobile_apps = AsyncMock()
    charger = create_charger(hass, home, name="faulting")
    car = make_real_car(hass, home, name="Zoe")
    charger.attach_car(car, T0 - timedelta(minutes=10))
    charger.current_command = MagicMock()
    charger.current_command.is_off_or_idle = MagicMock(return_value=False)
    charger._do_update_charger_state = AsyncMock()
    charger.is_not_plugged = MagicMock(return_value=False)
    charger.is_charge_enabled = MagicMock(return_value=False)
    charger.is_charge_disabled = MagicMock(return_value=True)
    charger.low_level_start_charge = AsyncMock(return_value=True)
    charger._notify_charger_fault = AsyncMock()
    _fake_zero_power_readings(charger, car)
    ct = _SocConstraint()

    fault_start = T0 + timedelta(seconds=100)
    recovery = T0 + timedelta(seconds=900)

    def _faulted(t: datetime) -> bool:
        return fault_start <= t < recovery

    charger.is_charger_faulted = MagicMock(side_effect=_faulted)

    await charger.start_charge(T0)
    assert charger._expected_charge_state.last_ping_time_success == T0

    async def _run_cycle(t: datetime) -> None:
        # the SOC callback is gated off while faulted (`is_load_active` is False),
        # so mirror that here; exercise both relative orders of the two writers.
        if call_order == "fault_state_first":
            await charger._update_charger_fault_state(t)
            if not _faulted(t):
                await charger.constraint_update_value_callback_percent_soc(ct, t)
        else:
            if not _faulted(t):
                await charger.constraint_update_value_callback_percent_soc(ct, t)
            await charger._update_charger_fault_state(t)

    first_alert_at = None
    t = T0
    while first_alert_at is None:
        await _run_cycle(t)
        if _no_power_alerts(charger) >= 1:
            first_alert_at = t
        if t > recovery + WINDOW + 30 * STEP:
            break
        t += STEP

    assert first_alert_at is not None, "the restarted charger never got its zero-power alert"
    assert first_alert_at > recovery + WINDOW, "the restarted charger was alerted before its own window"


# --------------------------------------------------------------------------------------
# N1 — a car swap without unplug does not inherit the previous car's latch
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_car_swap_clears_latch_but_allocation_churn_keeps_it():
    """N1: a genuine car swap clears the latch; detach/re-attach of the same car keeps it."""
    hass, home, _states, stuck = build_stuck_charger(switch_state="off")
    car_a = stuck.car
    _fake_zero_power_readings(stuck, car_a)
    ct = _SocConstraint()

    await _drive_stuck_to_first_alert(stuck, ct)
    assert stuck.possible_charge_error_start_time is not None
    assert stuck.car is car_a

    # allocation churn: detach then re-attach the SAME car → latch kept
    t = FIRST_ALERT_AT + STEP
    stuck.detach_car()
    stuck.attach_car(car_a, t)
    assert stuck.possible_charge_error_start_time is not None

    # genuine swap: a different car is selected → the stale latch is cleared
    car_b = make_real_car(hass, home, name="Other car")
    stuck.attach_car(car_b, t + STEP)
    assert stuck.car is car_b
    assert stuck.possible_charge_error_start_time is None
