"""QS-353 — a system/person-derived charge target must never be promoted to
user intent by a *system* action.

Three slices:

- **A′** the daily notification places a time-boxed *system* person hold on the
  car (honoured by allocation until the announced leave time) instead of writing
  ``person_name`` as user-originated;
- **B** the freeze never stores ``None`` under the target keys and stamps only
  the key matching the car's percent capability;
- **C** a user person change resets the car's person-bound state before freezing
  the new choice.

Car-only criteria use a real ``QSCar`` via the public charger harness
(``make_home`` + ``make_real_car``), with ``home._persons`` / a
``get_person_by_name`` stub so authorization resolves (the raw ``MagicMock``
home fails it otherwise). All ``time`` / ``until`` values are aware UTC.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz

from custom_components.quiet_solar.const import (
    CHARGE_TIME_CONSTRAINTS_CLEARED,
    CONF_MOBILE_APP,
    CONF_PERSON_AUTHORIZED_CARS,
    CONF_PERSON_NOTIFICATION_TIME,
    CONF_PERSON_PERSON_ENTITY,
    FORCE_CAR_NO_PERSON_ATTACHED,
    USER_ORIGINATED_CHARGE_TARGET_PERCENT,
    USER_ORIGINATED_CHARGE_TIME,
    USER_ORIGINATED_CHARGER_NAME,
)
from tests.utils.charger_harness import make_hass, make_home, make_real_car

QS_LOGGER = "custom_components.quiet_solar"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _authorized_person(name: str, car_name: str) -> MagicMock:
    person = MagicMock()
    person.name = name
    person.authorized_cars = [car_name]
    return person


def _wire_persons(home, persons) -> None:
    home._persons = list(persons)
    home.get_person_by_name = lambda n: next((p for p in home._persons if p.name == n), None)


def _future(hours: float = 2.0) -> datetime:
    return datetime.now(tz=pytz.UTC) + timedelta(hours=hours)


def _held_car(name: str = "HeldCar"):
    """A real percent car with an authorized forecasted person and no user pin."""
    hass = make_hass()
    home = make_home()
    car = make_real_car(hass, home, name=name)
    person = _authorized_person("Forecast", car.name)
    car.current_forecasted_person = person
    _wire_persons(home, [person])
    return hass, home, car, person


# =========================================================================== #
# AC 1 — hold_forecasted_person_until normalises `until` to aware UTC
# =========================================================================== #
@pytest.mark.parametrize("tz_mode", ["utc", "non_utc", "naive"])
def test_ac1_hold_until_stored_aware_utc(tz_mode):
    hass, home, car, person = _held_car()
    until_utc = _future()
    if tz_mode == "utc":
        until = until_utc
    elif tz_mode == "non_utc":
        until = until_utc.astimezone(pytz.timezone("Europe/Paris"))
    else:  # naive — assumed to already be UTC
        until = until_utc.replace(tzinfo=None)

    car.hold_forecasted_person_until(until)

    assert car._system_person_hold_name == "Forecast"
    assert car._system_person_hold_until == until_utc
    assert car._system_person_hold_until.tzinfo is not None
    # the freeze did not run — no user intent created
    assert not car.has_user_originated("person_name")
    assert not car.has_user_originated(USER_ORIGINATED_CHARGE_TARGET_PERCENT)


def test_ac1_hold_no_op_when_until_already_passed():
    hass, home, car, person = _held_car()
    car.hold_forecasted_person_until(datetime.now(tz=pytz.UTC) - timedelta(minutes=1))
    assert car._system_person_hold_name is None
    assert car._system_person_hold_until is None


# =========================================================================== #
# AC 3 — get_pinned_person_name resolver
# =========================================================================== #
def test_ac3_user_pin_wins_over_hold():
    hass, home, car, person = _held_car()
    until = _future()
    car.hold_forecasted_person_until(until)
    car._user_originated["person_name"] = FORCE_CAR_NO_PERSON_ATTACHED
    # user pin (even the sentinel) wins over the hold
    assert car.get_pinned_person_name(_future(1)) == FORCE_CAR_NO_PERSON_ATTACHED


def test_ac3_hold_returned_before_until_then_expires():
    hass, home, car, person = _held_car()
    until = _future()
    car.hold_forecasted_person_until(until)

    assert car.get_pinned_person_name(until - timedelta(seconds=1)) == "Forecast"
    # at `until` the hold has expired: None and both fields cleared
    assert car.get_pinned_person_name(until) is None
    assert car._system_person_hold_name is None
    assert car._system_person_hold_until is None
    # idempotent second call
    assert car.get_pinned_person_name(until) is None


def test_ac3_resolver_normalises_naive_time():
    """The resolver defensively normalises a naive `time` (assumed UTC) instead
    of raising when compared with the aware `until` (review: blind/edge hunters)."""
    hass, home, car, person = _held_car()
    until = _future()
    car.hold_forecasted_person_until(until)
    naive_before = (until - timedelta(seconds=1)).replace(tzinfo=None)
    assert car.get_pinned_person_name(naive_before) == "Forecast"
    naive_after = until.replace(tzinfo=None)
    assert car.get_pinned_person_name(naive_after) is None


# =========================================================================== #
# AC 6 — persistence round-trip
# =========================================================================== #
def test_ac6_persistence_round_trip():
    hass, home, car, person = _held_car()
    until = _future()
    car.hold_forecasted_person_until(until)

    data: dict = {}
    car.update_to_be_saved_extra_device_info(data)
    assert data["system_person_hold_name"] == "Forecast"
    assert data["system_person_hold_until"] == until.isoformat()

    other = make_real_car(make_hass(), make_home(), name="Other")
    other.use_saved_extra_device_info(data)
    assert other._system_person_hold_name == "Forecast"
    assert other._system_person_hold_until == until
    assert other._system_person_hold_until.tzinfo is not None


def test_ac6_persisted_expired_hold_resolves_none():
    hass, home, car, person = _held_car()
    past = datetime.now(tz=pytz.UTC) - timedelta(hours=1)
    data = {"system_person_hold_name": "Forecast", "system_person_hold_until": past.isoformat()}
    car.use_saved_extra_device_info(data)
    assert car.get_pinned_person_name(datetime.now(tz=pytz.UTC)) is None


def test_ac6_absent_keys_no_hold_no_warning(caplog):
    car = make_real_car(make_hass(), make_home(), name="Legacy")
    with caplog.at_level(logging.WARNING, logger=QS_LOGGER):
        car.use_saved_extra_device_info({"current_forecasted_person_name_from_boot": None})
    assert car._system_person_hold_name is None
    assert car._system_person_hold_until is None
    assert "system person hold" not in caplog.text


def test_ac6_malformed_until_drops_hold_with_warning(caplog):
    car = make_real_car(make_hass(), make_home(), name="Corrupt")
    with caplog.at_level(logging.WARNING, logger=QS_LOGGER):
        car.use_saved_extra_device_info(
            {"system_person_hold_name": "Forecast", "system_person_hold_until": "not-a-date"}
        )
    assert car._system_person_hold_name is None
    assert car._system_person_hold_until is None
    assert "invalid until" in caplog.text


def test_ac6_half_hold_name_only_no_hold_no_warning(caplog):
    car = make_real_car(make_hass(), make_home(), name="Half")
    with caplog.at_level(logging.WARNING, logger=QS_LOGGER):
        car.use_saved_extra_device_info(
            {"system_person_hold_name": "Forecast", "system_person_hold_until": None}
        )
    assert car._system_person_hold_name is None
    assert car._system_person_hold_until is None
    assert "invalid until" not in caplog.text


# =========================================================================== #
# AC 7 — clear sites
# =========================================================================== #
def test_ac7a_freeze_converts_held_person_to_user_pin():
    hass, home, car, person = _held_car()
    car.hold_forecasted_person_until(_future())
    # a genuine user tap (bump solar) with no user pin: the freeze converts the
    # held (forecasted) person into a user pin AND drops the hold (D5).
    car.set_user_originated("bump_solar", True)
    assert car.get_user_originated("person_name") == "Forecast"
    assert car._system_person_hold_name is None
    assert car._system_person_hold_until is None


@pytest.mark.asyncio
async def test_ac7b_other_car_holding_person_only_hold_cleared():
    hass = make_hass()
    home = make_home()
    car1 = make_real_car(hass, home, name="Car1")
    car2 = make_real_car(hass, home, name="Car2")
    magali = _authorized_person("Magali", car2.name)
    magali.authorized_cars = [car1.name, car2.name]
    _wire_persons(home, [magali])

    # car2 merely HOLDS Magali and carries a bump_solar marker in its store.
    car2.current_forecasted_person = magali
    car2.hold_forecasted_person_until(_future())
    car2._user_originated["bump_solar"] = True

    await car1.user_set_person_for_car("Magali")

    # car2 loses only its hold; the store survives (source-aware clear).
    assert car2._system_person_hold_name is None
    assert car2.get_user_originated("bump_solar") is True


@pytest.mark.asyncio
async def test_ac7bprime_other_car_user_pinned_person_whole_store_cleared():
    hass = make_hass()
    home = make_home()
    car1 = make_real_car(hass, home, name="Car1")
    car2 = make_real_car(hass, home, name="Car2")
    magali = _authorized_person("Magali", car2.name)
    magali.authorized_cars = [car1.name, car2.name]
    _wire_persons(home, [magali])

    # car2 is USER-PINNED to Magali (bump marker rides along). D5: no hold coexists.
    car2._user_originated["person_name"] = "Magali"
    car2._user_originated["bump_solar"] = True

    await car1.user_set_person_for_car("Magali")

    assert car2.get_user_originated("person_name") is None
    assert car2.get_user_originated("bump_solar") is None
    assert car2._system_person_hold_name is None


@pytest.mark.asyncio
async def test_ac7c_physical_unplug_clears_hold():
    """AC 7(c): a physical unplug clears the hold (charger.py, next to
    clear_all_user_originated). Fixture shape of the QS-352 unplug test."""
    from unittest.mock import MagicMock as _MM

    from tests.test_qs352_person_target_leak import _base_fixture, _preseed_filler, _run_step1

    hass, home, charger, car, now, magali, _thomas = _base_fixture()
    _preseed_filler(charger, car, now)
    await _run_step1(charger, car, now, magali)

    # seed a live system hold, then unplug
    car._system_person_hold_name = "Magali Menguy"
    car._system_person_hold_until = now + timedelta(hours=7)
    charger.is_not_plugged = _MM(return_value=True)
    charger.is_plugged = _MM(return_value=False)

    await charger.check_load_activity_and_constraints(now + timedelta(minutes=3))

    assert car._system_person_hold_name is None
    assert car._system_person_hold_until is None


@pytest.mark.asyncio
async def test_ac7d_user_clean_and_reset_clears_hold():
    hass, home, car, person = _held_car()
    car.hold_forecasted_person_until(_future())
    car.charger = None
    await car.user_clean_and_reset()
    assert car._system_person_hold_name is None


def test_ac7e_allocation_unauthorized_held_person_clears_hold_keeps_store():
    from custom_components.quiet_solar.ha_model.home import QSHome

    hass = make_hass()
    home = make_home()
    car = make_real_car(hass, home, name="Car")
    # a held person that is NOT authorized for the car
    ghost = MagicMock()
    ghost.name = "Ghost"
    ghost.authorized_cars = []
    ghost.preferred_car = None
    ghost.update_person_forecast = MagicMock(return_value=(None, None))
    _wire_persons(home, [ghost])
    car._system_person_hold_name = "Ghost"
    car._system_person_hold_until = _future()
    car._user_originated["bump_solar"] = True

    import asyncio

    asyncio.run(
        QSHome.compute_and_set_best_persons_cars_allocations(
            home, time=datetime.now(tz=pytz.UTC), force_update=True, do_notify=False
        )
    )

    assert car._system_person_hold_name is None
    assert car.get_user_originated("bump_solar") is True  # store untouched (hold-only car)


def test_ac7f_allocation_unauthorized_user_pin_clears_store_and_hold():
    from custom_components.quiet_solar.ha_model.home import QSHome

    hass = make_hass()
    home = make_home()
    car = make_real_car(hass, home, name="Car")
    ghost = MagicMock()
    ghost.name = "Ghost"
    ghost.authorized_cars = []
    ghost.preferred_car = None
    ghost.update_person_forecast = MagicMock(return_value=(None, None))
    _wire_persons(home, [ghost])
    car._user_originated["person_name"] = "Ghost"
    car._user_originated["bump_solar"] = True

    import asyncio

    asyncio.run(
        QSHome.compute_and_set_best_persons_cars_allocations(
            home, time=datetime.now(tz=pytz.UTC), force_update=True, do_notify=False
        )
    )

    assert car.get_user_originated("person_name") is None  # whole store cleared
    assert car._system_person_hold_name is None


# =========================================================================== #
# AC 8 — next-day re-hold updates `until`
# =========================================================================== #
def test_ac8_second_hold_updates_until():
    hass, home, car, person = _held_car()
    t1 = _future(1)
    car.hold_forecasted_person_until(t1)
    assert car._system_person_hold_until == t1
    t2 = _future(5)
    car.hold_forecasted_person_until(t2)
    assert car._system_person_hold_until == t2


def test_ac8_expired_hold_can_be_reheld():
    hass, home, car, person = _held_car()
    # seed an already-expired hold directly, then re-hold in the future
    car._system_person_hold_name = "Forecast"
    car._system_person_hold_until = datetime.now(tz=pytz.UTC) - timedelta(hours=1)
    t2 = _future(5)
    car.hold_forecasted_person_until(t2)
    assert car._system_person_hold_name == "Forecast"
    assert car._system_person_hold_until == t2


# =========================================================================== #
# AC 9 / 10 — B: the freeze never stores None; stamps only the matching key
# =========================================================================== #
def test_ac9_freeze_percent_car_none_target_stamps_nothing():
    hass = make_hass()
    home = make_home()
    car = make_real_car(hass, home, name="PercentCar")
    assert car.can_use_charge_percent_constraints() is True
    car._next_charge_target = None
    car._qs_bump_solar_priority = True  # field the freeze re-stamps from
    car.set_user_originated("bump_solar", True)  # fires the freeze
    assert not car.has_user_originated(USER_ORIGINATED_CHARGE_TARGET_PERCENT)
    assert not car.has_user_originated("charge_target_energy")
    assert car.get_user_originated("bump_solar") is True


def test_ac10_freeze_energy_car_stamps_only_energy():
    hass = make_hass()
    home = make_home()
    car = make_real_car(hass, home, name="EnergyCar", is_invited=True)
    assert car.can_use_charge_percent_constraints() is False
    car._next_charge_target_energy = 30000
    car.set_user_originated("bump_solar", True)  # fires the freeze
    assert car.get_user_originated("charge_target_energy") == 30000
    assert not car.has_user_originated(USER_ORIGINATED_CHARGE_TARGET_PERCENT)


# =========================================================================== #
# AC 11 / 12 / 13 — C: a user person change resets the person-bound state
# =========================================================================== #
async def _c_scenario(seed_via_field: bool):
    hass = make_hass()
    home = make_home()
    car = make_real_car(hass, home, name="Car")
    other = make_real_car(hass, home, name="Other")
    magali = _authorized_person("Magali", car.name)
    thomas = _authorized_person("Thomas", car.name)
    _wire_persons(home, [magali, thomas])

    # a second car user-pinned to the sentinel (seeded directly to avoid the freeze)
    other._user_originated["person_name"] = FORCE_CAR_NO_PERSON_ATTACHED

    car._qs_bump_solar_priority = True  # direct field — the property setter fires the freeze
    car._user_originated.update(
        {
            "person_name": "Magali",
            USER_ORIGINATED_CHARGER_NAME: "wallbox",
            USER_ORIGINATED_CHARGE_TIME: CHARGE_TIME_CONSTRAINTS_CLEARED,
        }
    )
    if seed_via_field:
        # snapshot provenance: the 41 was frozen from the field into the store
        car._next_charge_target = 41
        car._user_originated.update({USER_ORIGINATED_CHARGE_TARGET_PERCENT: 41, "bump_solar": True})
    else:
        # user-typed provenance (D4 / Blast radius): the 41 came from the select entity
        await car.user_set_next_charge_target(41)
        assert car.get_user_originated(USER_ORIGINATED_CHARGE_TARGET_PERCENT) == 41
    car._system_person_hold_name = "Magali"
    car._system_person_hold_until = _future()
    return hass, home, car, other


@pytest.mark.asyncio
@pytest.mark.parametrize("seed_via_field", [True, False])
async def test_ac11_person_change_resets_person_bound_state(seed_via_field):
    hass, home, car, other = await _c_scenario(seed_via_field)

    await car.user_set_person_for_car("Thomas")

    assert car._user_originated == {
        "person_name": "Thomas",
        "bump_solar": True,
        "force_charge": False,
        USER_ORIGINATED_CHARGE_TIME: CHARGE_TIME_CONSTRAINTS_CLEARED,
        USER_ORIGINATED_CHARGER_NAME: "wallbox",
    }
    # fields reset BEFORE the lazy getter re-materialises the default
    assert car._next_charge_target is None
    assert car._next_charge_target_energy is None
    assert car.get_car_target_SOC() == car.car_default_charge
    # hold cleared; the other car's store untouched
    assert car._system_person_hold_name is None
    assert other.get_user_originated("person_name") == FORCE_CAR_NO_PERSON_ATTACHED


@pytest.mark.asyncio
async def test_ac12_person_change_to_force_no_person():
    hass, home, car, other = await _c_scenario(seed_via_field=True)

    await car.user_set_person_for_car(FORCE_CAR_NO_PERSON_ATTACHED)

    assert car._user_originated == {
        "person_name": FORCE_CAR_NO_PERSON_ATTACHED,
        "bump_solar": True,
        "force_charge": False,
        USER_ORIGINATED_CHARGE_TIME: CHARGE_TIME_CONSTRAINTS_CLEARED,
        USER_ORIGINATED_CHARGER_NAME: "wallbox",
    }
    assert car._next_charge_target is None
    assert car._system_person_hold_name is None


@pytest.mark.asyncio
async def test_ac13_person_change_noop_when_already_pinned():
    hass, home, car, other = await _c_scenario(seed_via_field=True)
    before = dict(car._user_originated)

    await car.user_set_person_for_car("Magali")  # already pinned to Magali

    assert car._user_originated == before
    assert car._next_charge_target == 41  # fields untouched (early return)
    assert car._system_person_hold_name == "Magali"


# =========================================================================== #
# AC 5 — end-to-end no-tap snapshot path (#352 finding C2): the daily
# notification sets a hold, never a user pin, so the #352 restore fires once the
# person leaves.
# =========================================================================== #
@pytest.mark.asyncio
async def test_ac5_notification_holds_then_restore_fires_after_person_leaves(caplog):

    from custom_components.quiet_solar.ha_model.person import QSPerson
    from tests.test_qs352_person_target_leak import (
        PERSON_TARGET,
        _base_fixture,
        _car_cts,
        _install_spy,
        _person_cts,
        _preseed_filler,
    )

    hass, home, charger, car, now, _magali_mock, thomas = _base_fixture()

    person = QSPerson(
        hass=hass,
        home=home,
        config_entry=None,
        name="Magali",
        **{
            CONF_PERSON_PERSON_ENTITY: "person.magali",
            CONF_MOBILE_APP: "notify",
            CONF_PERSON_NOTIFICATION_TIME: "00:00:00",
            CONF_PERSON_AUTHORIZED_CARS: [car.name],
        },
    )
    until = now + timedelta(hours=7)
    person.predicted_mileage = 200.0
    person.predicted_leave_time = until
    person._last_request_prediction_time = now  # stabilise: no recompute
    person._last_forecast_notification_call_time = now - timedelta(days=1)
    person.on_device_state_change = AsyncMock()  # isolate the notify payload path
    _wire_persons(home, [person])

    car.current_forecasted_person = person
    # not-covered → the notify branch that sets a hold
    car.get_adapt_target_percent_soc_to_reach_range_km = lambda *a, **k: (False, 37.0, 80.0, None)

    _preseed_filler(charger, car, now)

    # Step 1: one cycle pushes the person constraint at 41 AND runs the notification.
    car.get_best_person_next_need = AsyncMock(return_value=(False, until, PERSON_TARGET, person))
    caplog.set_level(logging.INFO, logger=QS_LOGGER)
    await charger.check_load_activity_and_constraints(now)

    assert len(_person_cts(charger, "Magali")) == 1
    assert car._next_charge_target == 41
    # the no-tap snapshot path did NOT promote the person minimum to user intent
    assert not car.has_user_originated(USER_ORIGINATED_CHARGE_TARGET_PERCENT)
    assert not car.has_user_originated("person_name")
    assert car._system_person_hold_name == "Magali"

    # Step 2: advance past `until` and re-allocate (person left → Thomas covered).
    spy = _install_spy(car)
    later = until + timedelta(minutes=3)
    car.get_best_person_next_need = AsyncMock(return_value=(True, until, 30.0, thomas))
    car.current_forecasted_person = thomas
    await charger.check_load_activity_and_constraints(later)

    # the #352 restore fires (user_target is None because nothing was promoted)
    spy.assert_awaited_once_with(car.car_default_charge, do_update_charger=False)
    assert car._next_charge_target == car.car_default_charge
    assert _person_cts(charger, "Magali") == []
    assert all(c.target_value == car.car_default_charge for c in _car_cts(charger))
