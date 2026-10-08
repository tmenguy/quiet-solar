"""QS-403: a car that stops asking current is not "charged" when its trusted SOC is far below target.

Night of 2026-10-07/08: the Twingo stopped drawing power at 51 % (car-side fault) under a
manual "100 % at 05:30" constraint. After `SuspendedEV` for 1200 s,
`is_car_stopped_asking_current()` turned True and `QSChargerGeneric.is_car_charged`
forced `result = target_charge`, ignoring a healthy SOC sensor reading 51 %. Two callers
then treated the target as reached:

- `constraint_update_value_callback_soc` → constraint COMPLETED and notified (03:49);
- `_person_constraint_ends_this_cycle` → person constraint removed (04:56, 05:17).

Fixtures derive from the production log: SOC 51.0, target 100 (manual) or 95.69 (person).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz

from custom_components.quiet_solar.const import (
    CHARGE_TIME_CONSTRAINTS_CLEARED,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    DEVICE_STATUS_CHANGE_ERROR,
    USER_ORIGINATED_CHARGE_TIME,
)
from custom_components.quiet_solar.home_model.commands import CMD_AUTO_GREEN_ONLY, CMD_IDLE, copy_command
from custom_components.quiet_solar.home_model.constraints import MultiStepsPowerLoadConstraintChargePercent
from tests.factories import create_charger, make_charger_group, make_hass, make_home, make_real_car
from tests.test_bug_376_stuck_charger_group import build_stuck_charger
from tests.test_bug_379_zero_power_alert_arming import (
    _drive_stuck_to_first_alert,
    _fake_zero_power_readings,
    _SocConstraint,
)

NOW = datetime(2026, 10, 8, 1, 49, 11, tzinfo=pytz.UTC)  # 03:49:11 local
SOC_AT_FAULT = 51.0
MANUAL_TARGET = 100
PERSON_TARGET = 95.6896551724138


def _charger_with_car(*, distrusted: bool = False):
    hass = make_hass()
    home = make_home()
    charger = create_charger(hass, home, name="wallbox 3 portail", is_3p=True)
    car = make_real_car(hass, home, name="Twingo")
    charger.attach_car(car, NOW - timedelta(hours=3))
    car.is_soc_sensor_distrusted = MagicMock(return_value=distrusted)
    charger.is_car_stopped_asking_current = MagicMock(return_value=True)
    return charger, car


class _ManualSocConstraint:
    """The manual 100 % constraint, with a real `current >= target` met check."""

    def __init__(self, current_value: float, target_value: float) -> None:
        self.current_value = current_value
        self.target_value = target_value
        # short constraint history: stay on the plain "use the sensor" path
        self.first_value_update = NOW - timedelta(minutes=5)
        self.last_value_update = NOW - timedelta(minutes=1)
        self.last_value_change_update = NOW - timedelta(minutes=5)

    def is_constraint_met(self, time: datetime, current_value: float) -> bool:
        return current_value >= self.target_value


def _setup_soc_callback(charger, car, sensor_value: float | None) -> None:
    """Fake only the readings and sinks the SOC callback touches."""
    charger._do_update_charger_state = AsyncMock()
    charger.is_not_plugged = MagicMock(return_value=False)
    charger.current_command = copy_command(CMD_AUTO_GREEN_ONLY)
    car.get_car_charge_percent_raw_sensor = MagicMock(return_value=sensor_value)
    car.is_in_soc_estimation_mode = MagicMock(return_value=False)
    car.is_car_charge_growing = MagicMock(return_value=False)
    car.setup_car_charge_target_if_needed = AsyncMock()
    charger._compute_added_charge_update = MagicMock(return_value=0.0)
    # neutralise the zero-power hardware check: not what this bug is about
    charger.is_charging_power_zero = MagicMock(return_value=False)
    charger.on_device_state_change = AsyncMock()
    charger.charger_group.dyn_handle = AsyncMock()


# --------------------------------------------------------------------------------------
# Red tests: reproduce the diagnosed cause
# --------------------------------------------------------------------------------------


def test_stopped_asking_with_trusted_soc_far_below_target_is_not_charged():
    """1: trusted SOC 51 % vs target 100 % — stopped asking must not force 'charged'."""
    charger, _car = _charger_with_car()

    is_charged, result = charger.is_car_charged(
        NOW, current_charge=SOC_AT_FAULT, target_charge=MANUAL_TARGET, is_target_percent=True
    )

    assert (is_charged, result) == (False, SOC_AT_FAULT)


@pytest.mark.asyncio
async def test_soc_callback_keeps_manual_100_constraint_when_car_stops_at_51():
    """2: the 03:49 SOC callback must keep the manual 100 % constraint alive at 51 %."""
    charger, car = _charger_with_car()
    _setup_soc_callback(charger, car, sensor_value=SOC_AT_FAULT)
    # stay on the sensor path (no manual base-SOC estimation)
    assert car._user_base_soc_value is None
    ct = _ManualSocConstraint(current_value=SOC_AT_FAULT, target_value=MANUAL_TARGET)

    result, do_continue_constraint = await charger.constraint_update_value_callback_percent_soc(ct, NOW)

    assert result == SOC_AT_FAULT
    assert do_continue_constraint is True


def test_person_constraint_not_removed_when_car_stops_at_51():
    """3: the 04:56 / 05:17 person path must not see the person as covered at 51 %."""
    charger, car = _charger_with_car()
    assert car.get_user_originated(USER_ORIGINATED_CHARGE_TIME) != CHARGE_TIME_CONSTRAINTS_CLEARED
    person = MagicMock()
    person.name = "Arthur Menguy"

    ends = charger._person_constraint_ends_this_cycle(
        person=person,
        next_usage_time=NOW + timedelta(hours=3),
        person_min_target_charge=PERSON_TARGET,
        is_person_covered=False,
        car_current_charge_value=SOC_AT_FAULT,
        car_charge_agenda=None,
        is_target_percent=True,
        time=NOW,
    )

    assert ends is False


def test_stopped_asking_gap_just_above_threshold_is_not_charged():
    """4: boundary — a 6 % gap (> 5 %) is still 'far below target'."""
    charger, _car = _charger_with_car()

    is_charged, _result = charger.is_car_charged(
        NOW, current_charge=94, target_charge=MANUAL_TARGET, is_target_percent=True
    )

    assert is_charged is False


@pytest.mark.asyncio
async def test_soc_callback_idle_command_does_not_complete_when_car_stops_at_51():
    """Review fix #01 EC-1: an idle/off command feeds `None` to `is_car_charged`. The raw
    trusted SOC (51 %) must still block the force, or the callback completes the constraint."""
    charger, car = _charger_with_car()
    _setup_soc_callback(charger, car, sensor_value=SOC_AT_FAULT)
    charger.current_command = copy_command(CMD_IDLE)
    ct = _ManualSocConstraint(current_value=SOC_AT_FAULT, target_value=MANUAL_TARGET)

    result, do_continue_constraint = await charger.constraint_update_value_callback_percent_soc(ct, NOW)

    assert (result, do_continue_constraint) == (None, True)


@pytest.mark.asyncio
async def test_time_constraint_not_killed_when_car_stops_far_below_target():
    """Review fix #01 EC-4: the time-constraint kill caller. With a recent completed constraint
    and an agenda event 10 h out, the legacy force kills the agenda constraint; a trusted
    SOC far below target must keep it."""
    charger, car = _charger_with_car()
    now = datetime.now(pytz.UTC)
    charger.is_charger_unavailable = MagicMock(return_value=False)
    charger.probe_for_possible_needed_reboot = MagicMock(return_value=False)
    charger.is_not_plugged = MagicMock(return_value=False)
    charger.is_plugged = MagicMock(return_value=True)
    charger.set_charging_num_phases = AsyncMock(return_value=False)
    charger.set_max_charging_current = AsyncMock(return_value=True)
    charger.reboot = AsyncMock()
    charger.get_best_car = MagicMock(return_value=car)
    car.do_force_next_charge = False
    car.do_next_charge_time = None
    car.get_car_charge_percent = lambda time=None, *a, **kw: 50.0
    charger._last_completed_constraint = MultiStepsPowerLoadConstraintChargePercent(
        total_capacity_wh=60000,
        type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        time=now - timedelta(hours=3),
        load=charger,
        load_param=car.name,
        from_user=False,
        end_of_constraint=now - timedelta(hours=1),
        initial_value=30.0,
        target_value=80.0,
        power_steps=charger._power_steps,
    )
    start_time = now + timedelta(hours=10)
    car.get_next_scheduled_event = AsyncMock(return_value=(start_time, start_time + timedelta(hours=2)))
    car.get_best_person_next_need = AsyncMock(return_value=(None, None, None, None))

    await charger.check_load_activity_and_constraints(now)

    agenda_cts = [c for c in charger._constraints if c is not None and c.is_mandatory and not c.from_user]
    assert [c.end_of_constraint for c in agenda_cts] == [start_time]


# --------------------------------------------------------------------------------------
# Non-regression: green before and after
# --------------------------------------------------------------------------------------


def test_stopped_asking_in_soc_estimation_mode_keeps_legacy_force():
    """Review fix #01 EC-2: a manual-override estimate is not a sensor-backed SOC — keep the
    legacy force so a really-full car with a lagging estimate still completes."""
    charger, car = _charger_with_car()
    car.is_in_soc_estimation_mode = MagicMock(return_value=True)

    is_charged, result = charger.is_car_charged(
        NOW, current_charge=90, target_charge=MANUAL_TARGET, is_target_percent=True
    )

    assert (is_charged, result) == (True, MANUAL_TARGET)


def test_stopped_asking_with_distrusted_soc_still_forces_charged():
    """5: a distrusted SOC keeps the legacy 'stopped asking = charged' force."""
    charger, _car = _charger_with_car(distrusted=True)

    is_charged, result = charger.is_car_charged(
        NOW, current_charge=SOC_AT_FAULT, target_charge=MANUAL_TARGET, is_target_percent=True
    )

    assert (is_charged, result) == (True, MANUAL_TARGET)


def test_stopped_asking_gap_at_threshold_still_forces_charged():
    """6: a 5 % gap is NOT strictly above the threshold — 95 % for a 100 % target is charged."""
    charger, _car = _charger_with_car()

    is_charged, result = charger.is_car_charged(
        NOW, current_charge=95, target_charge=MANUAL_TARGET, is_target_percent=True
    )

    assert (is_charged, result) == (True, MANUAL_TARGET)


def test_stopped_asking_energy_target_unchanged():
    """7: energy (Wh) targets keep the legacy force."""
    charger, _car = _charger_with_car()

    is_charged, result = charger.is_car_charged(NOW, current_charge=10000, target_charge=40000, is_target_percent=False)

    assert (is_charged, result) == (True, 40000)


@pytest.mark.asyncio
async def test_stopped_asking_value_from_calculus_known_limit():
    """8: known limit (story "Limite connue") — frozen on purpose.

    With the sensor at None the callback evaluates the calculus value (95.69, the filler's
    start value) and the legacy force still applies: option A trusts the sensor *status*,
    not the origin of the value.
    """
    charger, car = _charger_with_car()
    _setup_soc_callback(charger, car, sensor_value=None)
    ct = _ManualSocConstraint(current_value=PERSON_TARGET, target_value=MANUAL_TARGET)

    result, do_continue_constraint = await charger.constraint_update_value_callback_percent_soc(ct, NOW)

    assert (result, do_continue_constraint) == (MANUAL_TARGET, False)


def test_stopped_asking_no_car_unchanged():
    """9: defensive — no production caller reaches here without a car; legacy force kept."""
    charger = create_charger(make_hass(), make_home(), name="wallbox 3 portail", is_3p=True)
    charger.is_car_stopped_asking_current = MagicMock(return_value=True)
    assert charger.car is None

    is_charged, result = charger.is_car_charged(
        NOW, current_charge=SOC_AT_FAULT, target_charge=MANUAL_TARGET, is_target_percent=True
    )

    assert (is_charged, result) == (True, MANUAL_TARGET)


# --------------------------------------------------------------------------------------
# UX addition (not a reproduction of the cause)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_power_notification_suggests_replug():
    """10: the zero-power notification tells the household to re-plug the car."""
    _hass, home, _states, stuck = build_stuck_charger(switch_state="off")
    make_charger_group(home, [stuck])
    _fake_zero_power_readings(stuck, stuck.car)
    ct = _SocConstraint()
    stuck._constraints = [ct]

    await _drive_stuck_to_first_alert(stuck, ct)

    error_calls = [
        c
        for c in stuck.on_device_state_change.await_args_list
        if c.kwargs.get("device_change_type") == DEVICE_STATUS_CHANGE_ERROR
    ]
    assert error_calls
    assert error_calls[0].kwargs["message"] == (
        f"There is no power being delivered to the car ({stuck.car.name}) while charging was expected"
        " — try unplugging and re-plugging the car"
    )
