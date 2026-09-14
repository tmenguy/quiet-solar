"""QS-352 — person-constraint target must not leak into the car's
``_next_charge_target`` once the person constraint's life ends.

Regression tests for the no-snapshot path (the path the 2026-09-14 incident
provably took, log-verified). The snapshot path is tracked separately as #353.

Real ``QSChargerGeneric`` + real ``QSCar`` on the public charger harness. The
only oracle that discriminates the fix is a spy wrapping
``QSCar.set_next_charge_target_percent``: in the step-2 cycle the person branch
is unreachable (person is ``None`` before the push site), so the restore is the
only possible caller of that method.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz

from custom_components.quiet_solar.const import (
    CONSTRAINT_FORECASTED_PERSON_KEY,
    CONSTRAINT_ORIGINATOR_KEY,
    CONSTRAINT_ORIGINATOR_PERSON,
    CONSTRAINT_TYPE_FILLER,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
)
from custom_components.quiet_solar.home_model.constraints import (
    MultiStepsPowerLoadConstraintChargePercent,
)
from tests.utils.charger_harness import (
    create_charger,
    init_charger_states,
    make_hass,
    make_home,
    make_real_car,
    plug_car,
)

QS_LOGGER = "custom_components.quiet_solar"

PERSON_TARGET = 41.6314620759743  # the person's float min-target (int-cast to 41 on the car)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _car_cts(charger):
    """Live constraints belonging to the charger's car."""
    return [c for c in charger._constraints if c is not None and c.load_param == charger.car.name]


def _person_cts(charger, name):
    """Car constraints tagged for the given forecasted person."""
    return [
        c
        for c in _car_cts(charger)
        if c.load_info is not None and c.load_info.get(CONSTRAINT_FORECASTED_PERSON_KEY) == name
    ]


def _non_person_cts(charger):
    """Car constraints with no forecasted-person tag (the fillers)."""
    return [
        c
        for c in _car_cts(charger)
        if c.load_info is None or CONSTRAINT_FORECASTED_PERSON_KEY not in c.load_info
    ]


def _base_fixture():
    """Real Tesla-M3 car plugged into a real charger, mocked HA I/O only."""
    hass = make_hass()
    home = make_home()
    charger = create_charger(hass, home)
    car = make_real_car(hass, home, name="Tesla M3", default_charge=80.0, minimum_ok_charge=20.0)
    now = datetime.now(pytz.UTC)

    init_charger_states(charger)
    charger.is_charger_unavailable = MagicMock(return_value=False)
    charger.probe_for_possible_needed_reboot = MagicMock(return_value=False)
    charger.is_not_plugged = MagicMock(return_value=False)
    charger.is_plugged = MagicMock(return_value=True)
    charger.is_car_stopped_asking_current = MagicMock(return_value=False)
    charger.set_charging_num_phases = AsyncMock(return_value=False)
    charger.set_max_charging_current = AsyncMock(return_value=True)
    charger.reboot = AsyncMock()

    plug_car(charger, car, now)
    charger.get_best_car = MagicMock(return_value=car)

    car.get_car_charge_percent = lambda time=None, *a, **kw: 37.0
    car.get_next_scheduled_event = AsyncMock(return_value=(None, None))
    car.do_next_charge_time = None
    car.do_force_next_charge = False

    magali = MagicMock()
    magali.name = "Magali Menguy"
    magali.notify_of_forecast_if_needed = AsyncMock()
    thomas = MagicMock()
    thomas.name = "Thomas Menguy"
    thomas.notify_of_forecast_if_needed = AsyncMock()

    return hass, home, charger, car, now, magali, thomas


def _preseed_filler(charger, car, now):
    """Push the pre-existing 37 -> 80 filler the incident had live (the `replacing` line)."""
    filler = MultiStepsPowerLoadConstraintChargePercent(
        total_capacity_wh=car.car_battery_capacity,
        type=CONSTRAINT_TYPE_FILLER,
        time=now - timedelta(hours=1),
        load=charger,
        load_param=car.name,
        from_user=False,
        initial_value=37.0,
        target_value=80.0,
        power_steps=charger._power_steps,
        support_auto=True,
    )
    charger.push_live_constraint(now - timedelta(hours=1), filler)


async def _run_step1(charger, car, now, magali):
    """Cycle N: Magali present and the car not charged enough -> person constraint
    created and the person's target int-cast into ``_next_charge_target``."""
    car.get_best_person_next_need = AsyncMock(return_value=(False, now + timedelta(hours=7), PERSON_TARGET, magali))
    car.current_forecasted_person = magali
    await charger.check_load_activity_and_constraints(now)

    # Preconditions: a step-2 failure can only be the diagnosed cause.
    person_cts = _person_cts(charger, "Magali Menguy")
    assert len(person_cts) == 1
    assert person_cts[0].target_value == pytest.approx(PERSON_TARGET)
    assert car._next_charge_target == 41
    non_person = _non_person_cts(charger)
    assert len(non_person) == 1
    assert non_person[0].initial_value == 37.0
    assert non_person[0].target_value == 80.0


def _install_spy(car):
    spy = AsyncMock(wraps=car.set_next_charge_target_percent)
    car.set_next_charge_target_percent = spy
    return spy


# --------------------------------------------------------------------------- #
# Test 1 — the person target is restored to default once the person leaves
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "step2_return, forecasted",
    [
        pytest.param((True, "thomas", 30.0), "thomas", id="reallocated_covered"),
        pytest.param((False, "thomas", 36.0), "thomas", id="need_already_met"),
        pytest.param((None, None, None), None, id="person_less"),
    ],
)
@pytest.mark.asyncio
async def test_person_target_restored_to_default_after_person_constraint_removed(
    step2_return, forecasted, caplog
):
    hass, home, charger, car, now, magali, thomas = _base_fixture()
    people = {"thomas": thomas, "magali": magali, None: None}

    _preseed_filler(charger, car, now)
    await _run_step1(charger, car, now, magali)

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    is_covered, who, target = step2_return
    person2 = people[who]
    car.get_best_person_next_need = AsyncMock(
        return_value=(is_covered, (now + timedelta(hours=7)) if person2 else None, target, person2)
    )
    car.current_forecasted_person = people[forecasted]

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    # Fails today:
    spy.assert_awaited_once_with(car.car_default_charge)
    assert car._next_charge_target == car.car_default_charge
    assert all(c.target_value == car.car_default_charge for c in _car_cts(charger))
    assert "restoring default charge target" in caplog.text

    # Passes today (scope guards):
    assert _person_cts(charger, "Magali Menguy") == []
    assert not car.has_user_originated("charge_target_percent")


# --------------------------------------------------------------------------- #
# Test 2 — a genuine user target is never clobbered by the restore
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_user_originated_target_survives_person_constraint_removal(caplog):
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    _preseed_filler(charger, car, now)
    await _run_step1(charger, car, now, magali)

    # Seed a user target via the two primitives (order matters), NOT
    # user_set_next_charge_target (which would create a user-timed constraint).
    await car.set_next_charge_target_percent(60)
    car.set_user_originated("charge_target_percent", 60)

    # Lock the snapshot side effects the branch relies on being inert.
    assert car.get_user_originated("charge_time") is None
    assert car.get_user_originated("force_charge") is False
    assert car.get_user_originated("bump_solar") is False
    assert car._next_charge_target == 60
    assert not car.has_user_originated("person_name")

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    car.get_best_person_next_need = AsyncMock(return_value=(True, now + timedelta(hours=7), 30.0, thomas))
    car.current_forecasted_person = thomas

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    spy.assert_not_awaited()
    assert car.get_user_originated("charge_target_percent") == 60
    assert car.get_car_target_SOC() == 60
    assert _person_cts(charger, "Magali Menguy") == []
    assert not any(c.from_user is True for c in _car_cts(charger))
    non_person = _non_person_cts(charger)
    assert len(non_person) == 1
    assert non_person[0].initial_value == 37.0
    assert non_person[0].target_value == 60
    assert "restoring default charge target" not in caplog.text


# --------------------------------------------------------------------------- #
# Test 3 — a present-but-None user target does not block the restore (C4 hole)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_none_user_target_does_not_block_restore(caplog):
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    # State left by a snapshot after home.py:2773 cleared the field.
    car._user_originated["charge_target_percent"] = None

    _preseed_filler(charger, car, now)
    await _run_step1(charger, car, now, magali)

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    car.get_best_person_next_need = AsyncMock(return_value=(True, now + timedelta(hours=7), 30.0, thomas))
    car.current_forecasted_person = thomas

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    spy.assert_awaited_once_with(car.car_default_charge)
    assert car._next_charge_target == car.car_default_charge
    assert all(c.target_value == car.car_default_charge for c in _car_cts(charger))
    assert "restoring default charge target" in caplog.text
    assert _person_cts(charger, "Magali Menguy") == []


# --------------------------------------------------------------------------- #
# Test 4 — guard-false path: no restore (and no churn) when already at default
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_no_restore_when_target_already_default(caplog):
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    # No Magali-need cycle: the field is at its lazy default.
    assert car.get_car_target_SOC() == 80.0

    # A live Magali person constraint to remove this cycle.
    person_ct = MultiStepsPowerLoadConstraintChargePercent(
        total_capacity_wh=car.car_battery_capacity,
        type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        time=now - timedelta(hours=1),
        load=charger,
        load_param=car.name,
        load_info={
            CONSTRAINT_FORECASTED_PERSON_KEY: "Magali Menguy",
            CONSTRAINT_ORIGINATOR_KEY: CONSTRAINT_ORIGINATOR_PERSON,
        },
        from_user=False,
        end_of_constraint=now + timedelta(hours=7),
        initial_value=37.0,
        target_value=50.0,
        power_steps=charger._power_steps,
        support_auto=True,
    )
    charger.push_live_constraint(now - timedelta(hours=1), person_ct)

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    car.get_best_person_next_need = AsyncMock(return_value=(True, now + timedelta(hours=7), 30.0, thomas))
    car.current_forecasted_person = thomas

    await charger.check_load_activity_and_constraints(now)

    assert _person_cts(charger, "Magali Menguy") == []
    spy.assert_not_awaited()
    assert car._next_charge_target == car.car_default_charge
    assert "restoring default charge target" not in caplog.text
