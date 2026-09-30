"""QS-379: arm the zero-power charger alert from the first start launch.

The "There is no power being delivered to the car ... while charging was expected"
alert needs `_expected_charge_state.last_ping_time_success`. Before QS-379 only
`QSStateCmd.success()` set it, so a start that never succeeds (the QS-376 stuck
start) could never arm the alert and the household got no signal.

The episode bookkeeping (end-of-episode latch clear) runs every cycle on the real
per-load path `check_load_activity_and_constraints` (QS-379 M1), which
`Home.update_loads_constraints` calls regardless of `is_load_active` or whether the
SOC callback fires — so these tests drive that real path, not a helper in isolation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.quiet_solar.const import CAR_CHARGE_NO_POWER_ERROR, DEVICE_STATUS_CHANGE_ERROR
from custom_components.quiet_solar.ha_model.charger import (
    CHARGER_CHECK_REAL_POWER_WINDOW_S,
    CHARGER_FAULT_NOTIFY_DEBOUNCE_S,
    CHARGER_NO_POWER_EPISODE_END_S,
    TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_OFF_TO_ON_S,
)
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
    States,
    build_stuck_charger,
)

WINDOW = timedelta(seconds=CHARGER_CHECK_REAL_POWER_WINDOW_S)
# QS-379 S1: how long QS must not want charge before the latch is dropped.
EPISODE_END = timedelta(seconds=CHARGER_NO_POWER_EPISODE_END_S)
# QS-379 S1: the soonest the group may re-set the target True after a F2 re-arm — the
# realistic gap the latch must survive (see `test_stuck_start_rearms_through_the_group`).
OFF_TO_ON_SPACING = timedelta(seconds=TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_OFF_TO_ON_S)

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


async def _real_cycle(charger, ct, t: datetime, soc: bool = True) -> None:
    """Drive one real load-management cycle on the 7 s grid.

    Mirrors production: the per-load path (`check_load_activity_and_constraints`, which
    carries the fault and zero-power-episode bookkeeping), then the state machine
    (`ensure_correct_state`), then — when the load is active — the SOC callback the
    solver drives via `update_live_constraints`.
    """
    await charger.check_load_activity_and_constraints(t)
    await charger.ensure_correct_state(t)
    if soc:
        await charger.constraint_update_value_callback_percent_soc(ct, t)


async def _drive_stuck_to_first_alert(stuck, ct) -> datetime:
    """Drive the stuck fixture via the real per-cycle path until the first zero-power
    alert; assert it lands at FIRST_ALERT_AT and return that alert time (S6)."""
    t = T0
    while True:
        await _real_cycle(stuck, ct, t)
        if _no_power_alerts(stuck) >= 1:
            assert t == FIRST_ALERT_AT
            return t
        if t > FIRST_ALERT_AT + 5 * STEP:
            raise AssertionError("no zero-power alert fired within the expected window")
        t += STEP


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
    """AC3: armed at the first launch, alert after 600 s, no second notification after the
    re-arm — driven every cycle through the *real* per-cycle path with a realistic
    (off->on spacing) re-arm gap (QS-379 S1/M1)."""
    _hass, home, _states, stuck = build_stuck_charger(switch_state="off")
    group = make_charger_group(home, [stuck])
    car = stuck.car
    _fake_zero_power_readings(stuck, car)
    ct = _SocConstraint()
    # A live SOC constraint on the load: this is what makes `is_load_active` True (the
    # real gate `Home` checks before the SOC callback), so we can assert the gate itself.
    stuck._constraints = [ct]

    first_alert_at = None
    t = T0
    while t < REARM:
        await stuck.check_load_activity_and_constraints(t)
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
    while stuck._expected_charge_state.value is True and t < REARM + 10 * STEP:
        await stuck.check_load_activity_and_constraints(t)
        await stuck.ensure_correct_state(t)
        t += STEP
    assert stuck._expected_charge_state.value is not True
    assert stuck.get_charge_type()[0] == CAR_CHARGE_NO_POWER_ERROR
    rearm_t = t

    # The `is_load_active` assertions above needed a live constraint on the load; drop it
    # now so the lightweight dummy does not reach the constraint-iterating group/status
    # helpers exercised by the round-2 re-arm below (the NO_POWER card stays latched).
    stuck._constraints = []

    # QS-379 S1: drive EVERY cycle through the *realistic* re-arm gap (the off->on
    # spacing, the soonest the group may re-start) via the real per-cycle path. The gap
    # is shorter than the end-of-episode threshold, so the latch is kept: no re-notify.
    assert OFF_TO_ON_SPACING < EPISODE_END
    while t < rearm_t + OFF_TO_ON_SPACING:
        await stuck.check_load_activity_and_constraints(t)
        assert stuck.possible_charge_error_start_time is not None, f"latch dropped too early at {t - rearm_t}"
        t += STEP

    # second stuck round, started by the group budget once the spacing has elapsed
    cs = stuck.get_stable_dynamic_charge_status(t)
    cs.budgeted_amp = 6
    cs.budgeted_num_phases = 1
    t_apply = t
    await group.apply_budgets([cs], [cs], t_apply)
    assert stuck._expected_charge_state.value is True
    stuck._constraints = [ct]

    round_two_checked = False
    t = t_apply
    while t <= t_apply + WINDOW + 2 * STEP:
        await stuck.check_load_activity_and_constraints(t)
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
# M1 / S1 — the zero-power latch does not outlive the stuck episode, on the real path
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_latch_clears_on_real_per_cycle_path_when_constraint_removed():
    """M1: once the SOC constraint is gone (the callback path no longer runs), the latch
    still clears — the end-of-episode bookkeeping runs on the every-cycle path
    `check_load_activity_and_constraints`, not only inside the SOC-gated block."""
    _hass, _home, _states, stuck = build_stuck_charger(switch_state="off")
    _fake_zero_power_readings(stuck, stuck.car)
    ct = _SocConstraint()

    t_alert = await _drive_stuck_to_first_alert(stuck, ct)
    assert stuck.possible_charge_error_start_time is not None

    # The constraint ends / is removed and QS stops wanting charge. Only the per-cycle
    # path runs from here (the SOC callback no longer fires) — the latch must still clear.
    stuck._constraints = []
    t = t_alert + STEP
    t_start = t
    stuck._expected_charge_state.set(False, t)

    # not yet: less than the end-of-episode threshold has elapsed
    while t < t_start + EPISODE_END:
        await stuck.check_load_activity_and_constraints(t)
        t += STEP
    assert stuck.possible_charge_error_start_time is not None

    # past the threshold: the latch clears and the card recovers
    end = t + 3 * STEP
    while t <= end:
        await stuck.check_load_activity_and_constraints(t)
        t += STEP
    assert stuck.possible_charge_error_start_time is None
    assert stuck.get_charge_type()[0] != CAR_CHARGE_NO_POWER_ERROR


@pytest.mark.asyncio
async def test_latch_clears_after_state_machine_reset_drops_command_object():
    """S1: the missing-command-object rule. When `_reset_state_machine()` drops the
    charge-state command object (`_inner_expected_charge_state is None`, as after a state
    reset / OCPP comm-error), `_update_no_power_episode` treats the charger as *not*
    wanting charge (the `_inner_expected_charge_state is not None` guard short-circuits
    before the lazy `_expected_charge_state` property runs), so the latch still clears
    after the episode threshold — and the bookkeeping does not lazily recreate the object."""
    _hass, _home, _states, stuck = build_stuck_charger(switch_state="off")
    _fake_zero_power_readings(stuck, stuck.car)
    ct = _SocConstraint()

    t_alert = await _drive_stuck_to_first_alert(stuck, ct)
    assert stuck.possible_charge_error_start_time is not None

    # Drop the command objects (post `_reset_state_machine` / OCPP comm-error): QS has no
    # charge-state command, which the episode bookkeeping counts as "not wanting charge".
    stuck._constraints = []
    stuck._reset_state_machine()
    assert stuck._inner_expected_charge_state is None

    t = t_alert + STEP
    t_start = t
    # below the threshold: latch kept
    while t < t_start + EPISODE_END:
        await stuck.check_load_activity_and_constraints(t)
        t += STEP
    assert stuck.possible_charge_error_start_time is not None

    # past the threshold: the latch clears even though the command object is gone
    end = t + 3 * STEP
    while t <= end:
        await stuck.check_load_activity_and_constraints(t)
        t += STEP
    assert stuck.possible_charge_error_start_time is None
    # the episode bookkeeping never touched the lazy `_expected_charge_state` property, so
    # the dropped charge-state command object was not recreated by this path
    assert stuck._inner_expected_charge_state is None


@pytest.mark.asyncio
async def test_per_member_zero_power_bookkeeping_is_independent():
    """M1: `_update_no_power_episode` keys only on each charger's *own* command object and
    latch, so two chargers in one group keep independent episode clocks — a member left
    wanting charge keeps its latch while a sibling that stopped wanting charge clears its
    own on schedule.

    Scope: this proves *per-member bookkeeping independence*, driving each member's real
    per-cycle path (`check_load_activity_and_constraints`) directly. It does not drive the
    whole-home `update_loads_constraints`/group-solve loop — the mock `home` fixture cannot
    run it — so it does not claim end-to-end group behaviour (QS-379 S3)."""
    hass = make_hass()
    home = make_home()
    # two DISTINCT stuck chargers (distinct names / entity ids) sharing ONE state store and
    # ONE group that is kept and actually references them.
    states = States()
    hass.states.get = MagicMock(side_effect=states.get)
    _h, _ho, _s, stuck_a = build_stuck_charger(
        switch_state="off", hass=hass, home=home, states=states, name="wallbox A", car_name="Car A"
    )
    _h2, _ho2, _s2, stuck_b = build_stuck_charger(
        switch_state="off", hass=hass, home=home, states=states, name="wallbox B", car_name="Car B"
    )
    group = make_charger_group(home, [stuck_a, stuck_b])
    assert group._chargers == [stuck_a, stuck_b]
    for c in (stuck_a, stuck_b):
        _fake_zero_power_readings(c, c.car)

    # both members hold a zero-power latch; member A is left wanting charge (still stuck,
    # its per-cycle path keeps returning without ending the episode), member B stops.
    stuck_a.possible_charge_error_start_time = T0
    stuck_b.possible_charge_error_start_time = T0
    stuck_a._constraints = []
    stuck_b._constraints = []
    t = T0
    stuck_b._expected_charge_state.set(False, t)

    end = t + EPISODE_END + 3 * STEP
    while t <= end:
        # every member's own per-cycle path runs, in order, regardless of the other's result.
        res_a = await stuck_a.check_load_activity_and_constraints(t)
        assert res_a is False
        await stuck_b.check_load_activity_and_constraints(t)
        t += STEP

    # member A still wants charge → its latch stays; member B's latch cleared on schedule
    assert stuck_a.possible_charge_error_start_time is not None
    assert stuck_b.possible_charge_error_start_time is None


@pytest.mark.asyncio
async def test_latch_kept_below_threshold_but_cleared_above_it():
    """S1 boundary pair: driven every cycle, not-wanting charge for less than the
    end-of-episode threshold keeps the latch; strictly more clears it."""
    _hass, _home, _states, stuck = build_stuck_charger(switch_state="off")
    _fake_zero_power_readings(stuck, stuck.car)
    ct = _SocConstraint()

    t_alert = await _drive_stuck_to_first_alert(stuck, ct)
    assert stuck.possible_charge_error_start_time is not None

    stuck._constraints = []
    t0 = t_alert + STEP
    stuck._expected_charge_state.set(False, t0)

    # just below the threshold: latch kept
    t = t0
    while (t - t0) <= EPISODE_END - STEP:
        await stuck.check_load_activity_and_constraints(t)
        t += STEP
    assert stuck.possible_charge_error_start_time is not None

    # cross the threshold: latch cleared
    while (t - t0) <= EPISODE_END + STEP:
        await stuck.check_load_activity_and_constraints(t)
        t += STEP
    assert stuck.possible_charge_error_start_time is None


@pytest.mark.asyncio
async def test_second_stuck_start_in_same_session_notifies_again():
    """S1: after the latch clears, a genuine later stuck start in the same plug session notifies again."""
    _hass, home, _states, stuck = build_stuck_charger(switch_state="off")
    group = make_charger_group(home, [stuck])
    _fake_zero_power_readings(stuck, stuck.car)
    ct = _SocConstraint()

    t_alert = await _drive_stuck_to_first_alert(stuck, ct)
    assert _no_power_alerts(stuck) == 1

    # let the episode end (target False for the full threshold) so the latch clears
    stuck._constraints = []
    t = t_alert + STEP
    stuck._expected_charge_state.set(False, t)
    end = t + EPISODE_END + 3 * STEP
    while t <= end:
        await stuck.check_load_activity_and_constraints(t)
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
    stuck._constraints = [ct]

    t = t_apply
    while _no_power_alerts(stuck) == 1:
        await _real_cycle(stuck, ct, t)
        if t > t_apply + WINDOW + 20 * STEP:
            raise AssertionError("second stuck start never re-notified")
        t += STEP

    assert _no_power_alerts(stuck) == 2
    assert stuck.on_device_state_change.await_count == 2


@pytest.mark.asyncio
async def test_fresh_latch_resets_end_of_episode_clock():
    """S5: setting a fresh zero-power latch resets the end-of-episode clock, so a stale
    'not wanting charge since' from an earlier period cannot clear the new latch on the
    very next not-wanted observation (which would break AC3 round 2)."""
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
    # a stale "not wanting charge since" from an earlier period, older than the threshold
    charger._no_charge_wanted_since = T0 - timedelta(seconds=CHARGER_NO_POWER_EPISODE_END_S + 100)

    t_alert = T0 + WINDOW + timedelta(seconds=1)
    await charger.constraint_update_value_callback_percent_soc(ct, t_alert)
    assert charger.possible_charge_error_start_time == t_alert
    # the fresh latch reset the stale clock ...
    assert charger._no_charge_wanted_since is None

    # ... so the first not-wanted observation does NOT immediately clear the fresh latch
    charger._expected_charge_state.set(False, t_alert + STEP)
    charger._update_no_power_episode(t_alert + STEP)
    assert charger.possible_charge_error_start_time is not None


# --------------------------------------------------------------------------------------
# S2 — fault recovery does not trigger an immediate zero-power alert
# --------------------------------------------------------------------------------------


def _make_faulting_charger(hass, home):
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
    return charger


@pytest.mark.parametrize("call_order", ["fault_state_first", "callback_first"])
@pytest.mark.asyncio
async def test_fault_recovery_rearms_zero_power_reference(call_order):
    """S2: a charger back from a fault gets its own window; no zero-power alert before recovery + 600 s."""
    hass = make_hass()
    home = make_home()
    home.async_notify_all_mobile_apps = AsyncMock()
    charger = _make_faulting_charger(hass, home)
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


@pytest.mark.asyncio
async def test_short_fault_recovery_alerts_exactly_one_window_later():
    """S3: a short (but real) fault re-arms to the recovery time, not `time`, so the stuck
    round alerts on the first grid cycle just past recovery + 600 s (tight upper bound)."""
    hass = make_hass()
    home = make_home()
    home.async_notify_all_mobile_apps = AsyncMock()
    charger = _make_faulting_charger(hass, home)
    ct = _SocConstraint()

    # a real fault (longer than the notify debounce) that clears quickly
    fault_start = T0 + timedelta(seconds=100)
    recovery = T0 + timedelta(seconds=300)

    def _faulted(t: datetime) -> bool:
        return fault_start <= t < recovery

    # the fault holds at least the notify debounce as *actually observed* on the 7 s grid —
    # the span between the first and last grid cycle for which `_faulted` is True, which is
    # what the debounce machine sees (not the nominal `recovery - fault_start`).
    _faulted_cycles = [T0 + i * STEP for i in range(int((recovery - T0) / STEP) + 2) if _faulted(T0 + i * STEP)]
    _observed_fault_held = _faulted_cycles[-1] - _faulted_cycles[0]
    assert _observed_fault_held.total_seconds() >= CHARGER_FAULT_NOTIFY_DEBOUNCE_S

    charger.is_charger_faulted = MagicMock(side_effect=_faulted)

    await charger.start_charge(T0)

    first_alert_at = None
    t = T0
    while first_alert_at is None:
        await charger._update_charger_fault_state(t)
        if not _faulted(t):
            await charger.constraint_update_value_callback_percent_soc(ct, t)
        if _no_power_alerts(charger) >= 1:
            first_alert_at = t
        if t > recovery + WINDOW + 10 * STEP:
            break
        t += STEP

    assert first_alert_at is not None, "the restarted charger never got its zero-power alert"
    # exactly one full window after recovery — not a whole extra window from `time`
    assert recovery + WINDOW < first_alert_at <= recovery + WINDOW + 2 * STEP


@pytest.mark.asyncio
async def test_subdebounce_status_blips_do_not_postpone_zero_power_alert():
    """S2: a stuck charger with a flaky status entity (sub-debounce `unavailable` blips
    every ~300 s) is still alerted on schedule — a brief blip must not keep refreshing the
    fault-recovery grace."""
    hass = make_hass()
    home = make_home()
    home.async_notify_all_mobile_apps = AsyncMock()
    charger = _make_faulting_charger(hass, home)
    ct = _SocConstraint()

    # a one-cycle fault blip every ~300 s, each far shorter than the notify debounce
    blip_period = timedelta(seconds=301)
    blip_starts = [T0 + i * blip_period for i in range(1, 10)]

    def _faulted(t: datetime) -> bool:
        return any(bs <= t < bs + STEP for bs in blip_starts)

    charger.is_charger_faulted = MagicMock(side_effect=_faulted)

    await charger.start_charge(T0)

    first_alert_at = None
    t = T0
    while first_alert_at is None:
        await charger._update_charger_fault_state(t)
        if not _faulted(t):
            await charger.constraint_update_value_callback_percent_soc(ct, t)
        if _no_power_alerts(charger) >= 1:
            first_alert_at = t
        if t > T0 + WINDOW + 20 * STEP:
            break
        t += STEP

    assert first_alert_at is not None, "a flaky status entity suppressed the zero-power alert entirely"
    # the blips never granted a fresh grace, so the alert still fires ~one window in
    assert first_alert_at <= T0 + WINDOW + 5 * STEP
    charger._notify_charger_fault.assert_not_awaited()


# --------------------------------------------------------------------------------------
# N1 / S4 — car-identity changes and the no-power latch
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_car_swap_clears_latch_but_allocation_churn_keeps_it():
    """N1: a genuine car swap clears the latch; detach/re-attach of the same car keeps it."""
    hass, home, _states, stuck = build_stuck_charger(switch_state="off")
    car_a = stuck.car
    _fake_zero_power_readings(stuck, car_a)
    ct = _SocConstraint()

    t_alert = await _drive_stuck_to_first_alert(stuck, ct)
    assert stuck.possible_charge_error_start_time is not None
    assert stuck.car is car_a

    # allocation churn: detach then re-attach the SAME car → latch kept
    t = t_alert + STEP
    stuck.detach_car()
    stuck.attach_car(car_a, t)
    assert stuck.possible_charge_error_start_time is not None

    # genuine swap: a different real car is selected → the stale latch is cleared, and
    # the new car gets its own zero-power window (the reference is re-armed).
    car_b = make_real_car(hass, home, name="Other car")
    t_swap = t + STEP
    stuck.attach_car(car_b, t_swap)
    assert stuck.car is car_b
    assert stuck.possible_charge_error_start_time is None
    assert stuck._expected_charge_state.last_ping_time_success == t_swap


@pytest.mark.asyncio
async def test_same_name_refresh_and_generic_transitions_keep_latch():
    """S4: comparing by identity would treat a config-reload refresh of the same car, or a
    generic->real identification, as a swap. Comparing by name (and ignoring the default
    generic car) keeps the latch in those cases."""
    hass, home, _states, stuck = build_stuck_charger(switch_state="off")
    car_a = stuck.car
    _fake_zero_power_readings(stuck, car_a)
    ct = _SocConstraint()

    t_alert = await _drive_stuck_to_first_alert(stuck, ct)
    assert stuck.possible_charge_error_start_time is not None

    # a config reload recreates the QSCar object with the SAME name → not a real swap
    t = t_alert + STEP
    car_a_reloaded = make_real_car(hass, home, name=car_a.name)
    stuck.attach_car(car_a_reloaded, t)
    assert stuck.car is car_a_reloaded
    assert stuck.possible_charge_error_start_time is not None

    # The real car drops to the per-charger generic fallback and later the real car is
    # identified again — both legs driven through the real attach/detach API (`detach_car`
    # records `_last_attached_car`; `attach_car` reads it), no hand-set of the private
    # field. A transition to or from the default generic car is not a real identity change,
    # so the latch is kept on both legs.
    stuck.detach_car()
    stuck.attach_car(stuck._default_generic_car, t + STEP)
    assert stuck.car is stuck._default_generic_car
    assert stuck.possible_charge_error_start_time is not None

    stuck.detach_car()
    car_real = make_real_car(hass, home, name="Zoe")
    stuck.attach_car(car_real, t + 2 * STEP)
    assert stuck.car is car_real
    assert stuck.possible_charge_error_start_time is not None
