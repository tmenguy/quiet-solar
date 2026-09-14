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
    CHARGE_TIME_CONSTRAINTS_CLEARED,
    CONSTRAINT_FORECASTED_PERSON_KEY,
    CONSTRAINT_ORIGINATOR_AGENDA,
    CONSTRAINT_ORIGINATOR_KEY,
    CONSTRAINT_ORIGINATOR_PERSON,
    CONSTRAINT_TYPE_FILLER,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
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


def _base_fixture(default_charge=80.0):
    """Real Tesla-M3 car plugged into a real charger, mocked HA I/O only."""
    hass = make_hass()
    home = make_home()
    charger = create_charger(hass, home)
    car = make_real_car(hass, home, name="Tesla M3", default_charge=default_charge, minimum_ok_charge=20.0)
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


def _preseed_filler(charger, car, now, target=80.0):
    """Push the pre-existing 37 -> `target` filler the incident had live (the `replacing` line)."""
    filler = MultiStepsPowerLoadConstraintChargePercent(
        total_capacity_wh=car.car_battery_capacity,
        type=CONSTRAINT_TYPE_FILLER,
        time=now - timedelta(hours=1),
        load=charger,
        load_param=car.name,
        from_user=False,
        initial_value=37.0,
        target_value=target,
        power_steps=charger._power_steps,
        support_auto=True,
    )
    charger.push_live_constraint(now - timedelta(hours=1), filler)


async def _run_step1(charger, car, now, magali, default=80.0):
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
    assert non_person[0].target_value == default


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


# --------------------------------------------------------------------------- #
# Finding #2 — a user target present must also be re-applied to _next_charge_target
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_user_target_set_before_person_restores_user_value(caplog):
    """user-set 60 -> person leak (41) -> person removed -> _next_charge_target back to 60."""
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    _preseed_filler(charger, car, now)

    await car.set_next_charge_target_percent(60)
    car.set_user_originated("charge_target_percent", 60)

    await _run_step1(charger, car, now, magali)
    assert car._next_charge_target == 41  # the push site overwrote the user's 60

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    car.get_best_person_next_need = AsyncMock(return_value=(True, now + timedelta(hours=7), 30.0, thomas))
    car.current_forecasted_person = thomas

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    spy.assert_awaited_once_with(60)
    assert car._next_charge_target == 60
    assert car.get_car_target_SOC() == 60
    assert car.get_user_originated("charge_target_percent") == 60
    assert _person_cts(charger, "Magali Menguy") == []
    non_person = _non_person_cts(charger)
    assert len(non_person) == 1
    assert non_person[0].initial_value == 37.0
    assert non_person[0].target_value == 60
    assert "restoring user charge target" in caplog.text


@pytest.mark.asyncio
async def test_post_reboot_user_default_restores_after_person_removed(caplog):
    """Post-reboot select-restore writes user_target=80 (int); the person leak (41) must
    not linger in the select — the user-target branch restores 80."""
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    _preseed_filler(charger, car, now)

    # The select restore writes an int via user_set_next_charge_target.
    await car.set_next_charge_target_percent(80)
    car.set_user_originated("charge_target_percent", 80)

    await _run_step1(charger, car, now, magali)
    assert car._next_charge_target == 41

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    car.get_best_person_next_need = AsyncMock(return_value=(True, now + timedelta(hours=7), 30.0, thomas))
    car.current_forecasted_person = thomas

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    spy.assert_awaited_once_with(80)
    assert car._next_charge_target == 80
    assert "restoring user charge target" in caplog.text


# --------------------------------------------------------------------------- #
# Finding #3 — an unplug before re-allocation must not leave the leaked target
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_unplug_with_live_person_constraint_clears_leaked_target(caplog):
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    _preseed_filler(charger, car, now)
    await _run_step1(charger, car, now, magali)
    assert car._next_charge_target == 41

    caplog.set_level(logging.INFO, logger=QS_LOGGER)
    charger.is_not_plugged = MagicMock(return_value=True)
    charger.is_plugged = MagicMock(return_value=False)

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    assert car._next_charge_target is None
    assert car.get_car_target_SOC() == car.car_default_charge


@pytest.mark.asyncio
async def test_unplug_with_live_person_constraint_clears_target():
    """Review #02 finding #1: the clear must fire even with a user target marker present
    (unplug wipes all user-originated state anyway), so the person leak must not linger."""
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    _preseed_filler(charger, car, now)

    await car.set_next_charge_target_percent(60)
    car.set_user_originated("charge_target_percent", 60)

    await _run_step1(charger, car, now, magali)
    assert car._next_charge_target == 41  # the person push clobbered the user's 60

    charger.is_not_plugged = MagicMock(return_value=True)
    charger.is_plugged = MagicMock(return_value=False)

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    # The field must not carry the person leak (nor the wiped user marker's value).
    assert car._next_charge_target is None
    assert car.get_car_target_SOC() == car.car_default_charge


@pytest.mark.asyncio
async def test_unplug_present_but_none_user_target_clears_leaked_target():
    """Review #02 finding #12(d): a present-but-None user marker is moot under the
    finding-#1 design — the clear fires whenever a person constraint is live."""
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    car._user_originated["charge_target_percent"] = None
    _preseed_filler(charger, car, now)
    await _run_step1(charger, car, now, magali)
    assert car._next_charge_target == 41

    charger.is_not_plugged = MagicMock(return_value=True)
    charger.is_plugged = MagicMock(return_value=False)

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    assert car._next_charge_target is None


@pytest.mark.asyncio
async def test_unplug_no_person_constraint_preserves_target():
    """Review #02 finding #12(e): unplug with NO live person constraint must not touch
    `_next_charge_target` — the clear is scoped to person-derived leaks only.

    NOTE: the value survives *at unplug* only; the first plugged cycle after replug
    reverts a marker-less non-default value to the default — see
    `test_replug_without_marker_reverts_to_default` (review #03 finding #3, intended)."""
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    # A non-person target with no person constraint live.
    await car.set_next_charge_target_percent(55)
    assert car._next_charge_target == 55

    charger.is_not_plugged = MagicMock(return_value=True)
    charger.is_plugged = MagicMock(return_value=False)

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    assert car._next_charge_target == 55


@pytest.mark.asyncio
async def test_has_live_person_constraint_no_car():
    """The person-constraint marker helper is safe when no car is attached."""
    hass, home, charger, car, now, magali, thomas = _base_fixture()
    charger.car = None
    assert charger._has_live_person_constraint() is False


# --------------------------------------------------------------------------- #
# Review #02 #4 / #03 #4/#8 — force/timed session must not inherit the leaked target
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
async def test_force_charge_after_person_removed_uses_default(step2_return, forecasted, caplog):
    hass, home, charger, car, now, magali, thomas = _base_fixture()
    people = {"thomas": thomas, "magali": magali, None: None}

    _preseed_filler(charger, car, now)
    await _run_step1(charger, car, now, magali)
    assert car._next_charge_target == 41

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    # Person no longer needs the car (covered / already-met / gone) AND the user
    # presses "charge now" on the same cycle.
    car.do_force_next_charge = True
    is_covered, who, target = step2_return
    person2 = people[who]
    car.get_best_person_next_need = AsyncMock(
        return_value=(is_covered, (now + timedelta(hours=7)) if person2 else None, target, person2)
    )
    car.current_forecasted_person = people[forecasted]

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    spy.assert_awaited_once_with(car.car_default_charge)
    assert car._next_charge_target == car.car_default_charge
    asap_cts = [c for c in _car_cts(charger) if c.type == CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE]
    assert len(asap_cts) == 1
    assert asap_cts[0].target_value == car.car_default_charge


@pytest.mark.asyncio
async def test_timed_charge_after_person_removed_uses_default(caplog):
    """#8(i): the user-timed variant (do_next_charge_time) also realigns."""
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    _preseed_filler(charger, car, now)
    await _run_step1(charger, car, now, magali)
    assert car._next_charge_target == 41

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    car.do_next_charge_time = now + timedelta(hours=6)
    car.get_best_person_next_need = AsyncMock(return_value=(True, now + timedelta(hours=7), 30.0, thomas))
    car.current_forecasted_person = thomas

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    spy.assert_awaited_once_with(car.car_default_charge)
    assert car._next_charge_target == car.car_default_charge
    timed_cts = [c for c in _car_cts(charger) if c.from_user is True and c.type == CONSTRAINT_TYPE_MANDATORY_END_TIME]
    assert len(timed_cts) == 1
    assert timed_cts[0].target_value == car.car_default_charge


@pytest.mark.asyncio
async def test_force_charge_after_person_removed_uses_user_target(caplog):
    """#8(iii): the early restore realigns to a user target when one is set."""
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    _preseed_filler(charger, car, now)
    await car.set_next_charge_target_percent(60)
    car.set_user_originated("charge_target_percent", 60)
    await _run_step1(charger, car, now, magali)
    assert car._next_charge_target == 41

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    # Setting the user target snapshotted a force_charge=False marker; clear it so the
    # do_force_next_charge press is honoured (else the person block, not the early
    # restore, would run).
    car.clear_user_originated("force_charge")
    car.do_force_next_charge = True
    car.get_best_person_next_need = AsyncMock(return_value=(True, now + timedelta(hours=7), 30.0, thomas))
    car.current_forecasted_person = thomas

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    spy.assert_awaited_once_with(60)
    assert car._next_charge_target == 60
    asap_cts = [c for c in _car_cts(charger) if c.type == CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE]
    assert len(asap_cts) == 1
    assert asap_cts[0].target_value == 60


@pytest.mark.asyncio
async def test_force_charge_cleared_charge_time_restores_default(caplog):
    """#4: the CLEARED-charge-time sentinel ends the person constraint even while the
    person is still assigned+uncovered — the early restore must fire under force."""
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    _preseed_filler(charger, car, now)
    await _run_step1(charger, car, now, magali)
    assert car._next_charge_target == 41

    # Sentinel set directly (bypasses the user-originated snapshot side effects).
    car._user_originated["charge_time"] = CHARGE_TIME_CONSTRAINTS_CLEARED

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    car.do_force_next_charge = True
    # Magali still assigned + range-uncovered, but the CLEARED sentinel ends the ct.
    car.get_best_person_next_need = AsyncMock(return_value=(False, now + timedelta(hours=7), PERSON_TARGET, magali))
    car.current_forecasted_person = magali

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    spy.assert_awaited_once_with(car.car_default_charge)
    assert car._next_charge_target == car.car_default_charge


@pytest.mark.asyncio
async def test_force_charge_person_still_needs_car_no_early_restore(caplog):
    """#8(ii): the person still needs the car -> the early restore must NOT fire."""
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    _preseed_filler(charger, car, now)
    await _run_step1(charger, car, now, magali)
    assert car._next_charge_target == 41

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    # Magali still assigned and range-uncovered, not charged enough.
    car.do_force_next_charge = True
    car.get_best_person_next_need = AsyncMock(return_value=(False, now + timedelta(hours=7), PERSON_TARGET, magali))
    car.current_forecasted_person = magali

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    spy.assert_not_awaited()


# --------------------------------------------------------------------------- #
# Review #03 finding #2 — car-switch / "no car" exits must also clear the leak
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_car_switch_with_live_person_constraint_clears_target():
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    _preseed_filler(charger, car, now)
    await _run_step1(charger, car, now, magali)
    assert car._next_charge_target == 41

    # The charger's car select now resolves to a different car.
    other = make_real_car(hass, home, name="ID.buzz", default_charge=80.0, minimum_ok_charge=20.0)
    charger.get_best_car = MagicMock(return_value=other)

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    assert car._next_charge_target is None


@pytest.mark.asyncio
async def test_no_car_selected_with_live_person_constraint_clears_target():
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    _preseed_filler(charger, car, now)
    await _run_step1(charger, car, now, magali)
    assert car._next_charge_target == 41

    charger.get_best_car = MagicMock(return_value=None)

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    assert car._next_charge_target is None


# --------------------------------------------------------------------------- #
# Review #03 finding #3 — marker-less non-default target reverts on replug (intended)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_replug_without_marker_reverts_to_default(caplog):
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    # User picks 55 (both primitives); no person constraint involved.
    await car.set_next_charge_target_percent(55)
    car.set_user_originated("charge_target_percent", 55)

    # Unplug wipes the user marker but leaves _next_charge_target = 55.
    charger.is_not_plugged = MagicMock(return_value=True)
    charger.is_plugged = MagicMock(return_value=False)
    await charger.check_load_activity_and_constraints(now)
    assert not car.has_user_originated("charge_target_percent")
    assert car._next_charge_target == 55

    # Replug: the first plugged cycle reverts the marker-less non-default value.
    charger.is_not_plugged = MagicMock(return_value=False)
    charger.is_plugged = MagicMock(return_value=True)
    charger.get_best_car = MagicMock(return_value=car)
    car.get_best_person_next_need = AsyncMock(return_value=(None, None, None, None))
    car.current_forecasted_person = None
    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    spy.assert_awaited_once_with(car.car_default_charge)
    assert car._next_charge_target == car.car_default_charge


# --------------------------------------------------------------------------- #
# Finding #6 — the same-cycle agenda constraint must follow the restored target
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_agenda_constraint_target_refreshed_after_person_removed(caplog):
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    _preseed_filler(charger, car, now)
    await _run_step1(charger, car, now, magali)

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)
    charger._last_completed_constraint = None

    start_time = now + timedelta(hours=20)
    car.get_next_scheduled_event = AsyncMock(return_value=(start_time, None))
    car.get_best_person_next_need = AsyncMock(return_value=(True, now + timedelta(hours=7), 30.0, thomas))
    car.current_forecasted_person = thomas

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    spy.assert_awaited_once_with(car.car_default_charge)
    agenda_cts = [
        c
        for c in _non_person_cts(charger)
        if c.type == CONSTRAINT_TYPE_MANDATORY_END_TIME and c.end_of_constraint == start_time
    ]
    assert len(agenda_cts) == 1
    assert agenda_cts[0].target_value == car.car_default_charge
    # Review #02 finding #2: the serialized field must be realigned too.
    assert agenda_cts[0].requested_target_value == car.car_default_charge


@pytest.mark.asyncio
async def test_agenda_live_constraint_realigned_after_person_removed(caplog):
    """Review #02 finding #2: when an eq_no_current-equal agenda constraint is already
    live, `push_agenda_constraints` discards the freshly built object; the realign must
    reach the LIVE constraint by identity, not mutate the discarded orphan."""
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    # Agenda end must be > 25 h beyond the person's next usage, else the person
    # constraint is not pushed and the leak would already clear in cycle 1.
    start_time = now + timedelta(hours=40)
    car.get_next_scheduled_event = AsyncMock(return_value=(start_time, None))
    car._next_charge_target = 41  # leaked state from a prior person push
    charger._last_completed_constraint = None

    # Cycle 1: Magali present & not covered -> a live agenda carrying the leaked 41,
    # plus a live person constraint.
    car.get_best_person_next_need = AsyncMock(return_value=(False, now + timedelta(hours=5), PERSON_TARGET, magali))
    car.current_forecasted_person = magali
    await charger.check_load_activity_and_constraints(now)

    agendas = [
        c
        for c in _non_person_cts(charger)
        if c.load_info is not None and c.load_info.get(CONSTRAINT_ORIGINATOR_KEY) == CONSTRAINT_ORIGINATOR_AGENDA
    ]
    assert len(agendas) == 1
    assert agendas[0].target_value == 41
    live_agenda = agendas[0]  # the object we expect to be realigned in place

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    # Cycle 2: person gone -> removal + restore. The fresh agenda build is an orphan
    # (eq_no_current-equal to `live_agenda`), so only the identity lookup can fix it.
    car.get_best_person_next_need = AsyncMock(return_value=(True, now + timedelta(hours=5), 30.0, thomas))
    car.current_forecasted_person = thomas
    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    spy.assert_awaited_once_with(car.car_default_charge)
    assert live_agenda.target_value == car.car_default_charge
    assert live_agenda.requested_target_value == car.car_default_charge
    # still exactly one agenda constraint (no duplicate pushed)
    agendas2 = [
        c
        for c in _non_person_cts(charger)
        if c.load_info is not None and c.load_info.get(CONSTRAINT_ORIGINATOR_KEY) == CONSTRAINT_ORIGINATOR_AGENDA
    ]
    assert len(agendas2) == 1


def _push_person_ct(charger, car, now, name, target, end):
    ct = MultiStepsPowerLoadConstraintChargePercent(
        total_capacity_wh=car.car_battery_capacity,
        type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        time=now - timedelta(hours=1),
        load=charger,
        load_param=car.name,
        load_info={CONSTRAINT_FORECASTED_PERSON_KEY: name, CONSTRAINT_ORIGINATOR_KEY: CONSTRAINT_ORIGINATOR_PERSON},
        from_user=False,
        end_of_constraint=end,
        initial_value=37.0,
        target_value=target,
        power_steps=charger._power_steps,
        support_auto=True,
    )
    charger.push_live_constraint(now - timedelta(hours=1), ct)


def _push_agenda_ct(charger, car, now, target, end):
    ct = MultiStepsPowerLoadConstraintChargePercent(
        total_capacity_wh=car.car_battery_capacity,
        type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        time=now - timedelta(hours=1),
        load=charger,
        load_param=car.name,
        load_info={CONSTRAINT_ORIGINATOR_KEY: CONSTRAINT_ORIGINATOR_AGENDA},
        from_user=False,
        end_of_constraint=end,
        initial_value=37.0,
        target_value=target,
        power_steps=charger._power_steps,
        support_auto=True,
    )
    charger.push_live_constraint(now - timedelta(hours=1), ct)
    return ct


@pytest.mark.asyncio
async def test_agenda_realigned_when_not_rebuilt_this_cycle(caplog):
    """Review #03 finding #7: a live agenda carrying the leaked value is realigned even
    when this cycle does not rebuild one (a completed person ct suppresses the rebuild)."""
    hass, home, charger, car, now, magali, thomas = _base_fixture()
    car._next_charge_target = 41
    _push_person_ct(charger, car, now, "Magali Menguy", 41, now + timedelta(hours=5))
    live_agenda = _push_agenda_ct(charger, car, now, 41, now + timedelta(hours=30))

    # No scheduled event this cycle -> the agenda block does not rebuild/replace it.
    car.get_next_scheduled_event = AsyncMock(return_value=(None, None))
    charger._last_completed_constraint = None

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)
    car.get_best_person_next_need = AsyncMock(return_value=(True, now + timedelta(hours=5), 30.0, thomas))
    car.current_forecasted_person = thomas

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    spy.assert_awaited_once_with(car.car_default_charge)
    assert live_agenda.target_value == car.car_default_charge
    assert live_agenda.requested_target_value == car.car_default_charge


@pytest.mark.asyncio
async def test_realized_refresh_avoids_extra_filler(caplog):
    """Review #03 finding #9: refreshing realized_charge_target after the restore keeps
    the filler block from pushing a filler alongside the (now-default) agenda."""
    hass, home, charger, car, now, magali, thomas = _base_fixture()
    car._next_charge_target = 41
    _push_person_ct(charger, car, now, "Magali Menguy", 41, now + timedelta(hours=7))

    start_time = now + timedelta(hours=20)
    car.get_next_scheduled_event = AsyncMock(return_value=(start_time, None))
    charger._last_completed_constraint = None

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)
    car.get_best_person_next_need = AsyncMock(return_value=(True, now + timedelta(hours=7), 30.0, thomas))
    car.current_forecasted_person = thomas

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    spy.assert_awaited_once_with(car.car_default_charge)
    # realized refreshed to 80 -> `80 < 80` is False -> no filler pushed alongside the agenda.
    assert [c for c in _car_cts(charger) if c.type == CONSTRAINT_TYPE_FILLER] == []
    agendas = [
        c
        for c in _non_person_cts(charger)
        if c.load_info is not None and c.load_info.get(CONSTRAINT_ORIGINATOR_KEY) == CONSTRAINT_ORIGINATOR_AGENDA
    ]
    assert len(agendas) == 1
    assert agendas[0].target_value == car.car_default_charge


# --------------------------------------------------------------------------- #
# Finding #7 — churn / user_target == 0 / fractional-default boundaries
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_restore_does_not_churn_across_cycles(caplog):
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    _preseed_filler(charger, car, now)
    await _run_step1(charger, car, now, magali)

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    car.get_best_person_next_need = AsyncMock(return_value=(None, None, None, None))
    car.current_forecasted_person = None

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))
    spy.assert_awaited_once_with(car.car_default_charge)

    # No churn on the next cycle: target already at default.
    await charger.check_load_activity_and_constraints(now + timedelta(minutes=6))
    spy.assert_awaited_once()

    # home.py:2773 clears it to None between cycles -> lazy default, still no churn.
    car._next_charge_target = None
    await charger.check_load_activity_and_constraints(now + timedelta(minutes=9))
    spy.assert_awaited_once()


@pytest.mark.asyncio
async def test_zero_user_target_blocks_default_restore(caplog):
    """user_target == 0 is an explicit override: the default-restore branch is skipped;
    with finding #2 in place the user-target branch re-applies 0."""
    hass, home, charger, car, now, magali, thomas = _base_fixture()

    _preseed_filler(charger, car, now)

    await car.set_next_charge_target_percent(0)
    car.set_user_originated("charge_target_percent", 0)
    assert car.get_user_originated("charge_target_percent") == 0

    await _run_step1(charger, car, now, magali)
    assert car._next_charge_target == 41

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    car.get_best_person_next_need = AsyncMock(return_value=(True, now + timedelta(hours=7), 30.0, thomas))
    car.current_forecasted_person = thomas

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    spy.assert_awaited_once_with(0)
    assert car._next_charge_target == 0
    assert "restoring default charge target" not in caplog.text


@pytest.mark.asyncio
async def test_fractional_default_no_churn(caplog):
    hass, home, charger, car, now, magali, thomas = _base_fixture(default_charge=80.5)

    _preseed_filler(charger, car, now, target=80.5)
    await _run_step1(charger, car, now, magali, default=80.5)

    spy = _install_spy(car)
    caplog.set_level(logging.INFO, logger=QS_LOGGER)

    car.get_best_person_next_need = AsyncMock(return_value=(None, None, None, None))
    car.current_forecasted_person = None

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))
    spy.assert_awaited_once_with(80.5)

    # int() on both sides: int(get_car_target_SOC()==80) == int(80.5) -> no second restore.
    await charger.check_load_activity_and_constraints(now + timedelta(minutes=6))
    spy.assert_awaited_once()
