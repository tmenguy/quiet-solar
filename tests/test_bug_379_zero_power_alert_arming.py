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
import pytz

from custom_components.quiet_solar.const import CAR_CHARGE_NO_POWER_ERROR, DEVICE_STATUS_CHANGE_ERROR
from custom_components.quiet_solar.ha_model.charger import CHARGER_CHECK_REAL_POWER_WINDOW_S
from tests.factories import create_state_cmd
from tests.test_bug_376_stuck_charger_group import (
    REARM,
    STATUS,
    STEP,
    SWITCH,
    T0,
    _build_stuck_charger,
    _make_charger_group,
)
from tests.test_charger_coverage_deep import _create_charger, _make_hass, _make_home, _make_real_car

WINDOW = timedelta(seconds=CHARGER_CHECK_REAL_POWER_WINDOW_S)


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
        and "no power being delivered" in c.kwargs.get("message", "")
    )


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
    hass, home, _states, stuck = _build_stuck_charger(switch_state="off")
    group = _make_charger_group(home, [stuck])
    car = stuck.car
    _fake_zero_power_readings(stuck, car)
    ct = _SocConstraint()
    state_cmd = lambda: stuck._expected_charge_state  # noqa: E731 — rebuilt on reset

    first_alert_at = None
    t = T0
    while t < REARM:
        await stuck.ensure_correct_state(t)
        assert stuck._expected_charge_state.value is True
        # not faulted: `is_load_active` keeps the SOC callback running (QS-346 gate)
        assert stuck.is_charger_faulted(t) is False
        assert state_cmd().last_ping_time_success == (T0 if first_alert_at is None else first_alert_at)

        await stuck.constraint_update_value_callback_percent_soc(ct, t)

        if t <= T0 + WINDOW:
            assert _no_power_alerts(stuck) == 0, f"alerted too early at {t - T0}"
            assert stuck.possible_charge_error_start_time is None
        elif first_alert_at is None:
            first_alert_at = t
            assert _no_power_alerts(stuck) == 1
            assert stuck.possible_charge_error_start_time == t
            assert stuck.get_charge_type()[0] == CAR_CHARGE_NO_POWER_ERROR
            # the block ran: its re-check reference advanced
            assert state_cmd().last_ping_time_success == t
        t += STEP

    assert first_alert_at == T0 + timedelta(seconds=602)

    # F2 re-arm: the target goes back to idle; the error stays visible on the car card
    while stuck._expected_charge_state.value is True:
        await stuck.ensure_correct_state(t)
        t += STEP
    assert stuck.get_charge_type()[0] == CAR_CHARGE_NO_POWER_ERROR

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
    hass = _make_hass()
    home = _make_home()
    charger = _create_charger(hass, home, name="generic stuck")
    car = _make_real_car(hass, home, name="Zoe")
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
    assert charger.possible_charge_error_start_time == t_alert
    assert charger.get_charge_type()[0] == CAR_CHARGE_NO_POWER_ERROR


# --------------------------------------------------------------------------------------
# AC4 — a confirmed start keeps the success time as reference
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirmed_start_keeps_success_time_as_reference():
    """AC4: a start confirmed on the next cycle is checked 600 s after the success, as before."""
    hass, home, states, charger = _build_stuck_charger(switch_state="off")
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
