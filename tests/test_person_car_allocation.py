"""End-to-end tests for the person-car allocation algorithm.

These tests exercise the real allocation pipeline (Hungarian algorithm,
manual overrides via user_set_person_for_car) with lightweight fakes
instead of full HA integration.
"""

import itertools
import logging
from datetime import UTC, datetime, timedelta

import pytest
import pytz

from custom_components.quiet_solar.const import (
    FORCE_CAR_NO_PERSON_ATTACHED,
    PASS1_PREFERRED_CAR_PENALTY_WH,
    PLUGGED_COVERED_CAR_PENALTY_WH,
    PREFERRED_CAR_ENERGY_THRESHOLD_WH,
)
from custom_components.quiet_solar.ha_model.car import QSCar
from custom_components.quiet_solar.ha_model.home import QSHome

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight fakes – just enough state for the real allocation methods
# ---------------------------------------------------------------------------

# Production energy units are Wh (car_battery_capacity is configured in Wh and
# car.py:1996 computes diff_energy in Wh); the fakes mirror that so the real
# cost-matrix thresholds (Wh) are exercised as in production.
WH_PER_KM = 150.0


class _FakeCharger:
    """Stub charger so car.charger is truthy and user_set_person_for_car works."""

    async def update_charger_for_user_change(self):
        pass


class _FakeCar:
    """Minimal car with autonomy-based coverage computation."""

    def __init__(
        self,
        name,
        remaining_km,
        has_charger,
        is_invited=False,
        is_plugged=False,
        default_charge=100.0,
        data_error=False,
    ):
        self.name = name
        self._remaining_km = remaining_km
        self.charger = _FakeCharger() if has_charger else None
        self.car_is_invited = is_invited
        # data_error=True → unreadable SOC / efficiency: coverage is None (the -2
        # "car data error" sentinel in _build_raw_energy_matrix), independent of
        # whether the person has a forecast.
        self._data_error = data_error
        self._user_originated: dict = {}
        self.current_forecasted_person = None
        self.home = None
        self.ha_entities = {}
        self._is_plugged = is_plugged
        self.car_default_charge = default_charge
        self._next_charge_target = None

    # Mirror AbstractDevice user_originated API
    def set_user_originated(self, key, value):
        self._user_originated[key] = value

    def get_user_originated(self, key, default=None):
        return self._user_originated.get(key, default)

    def has_user_originated(self, key):
        return key in self._user_originated

    def clear_user_originated(self, key):
        self._user_originated.pop(key, None)

    def clear_all_user_originated(self):
        self._user_originated.clear()

    def get_adapt_target_percent_soc_to_reach_range_km(self, mileage, time):
        """Return (is_covered, current_soc, needed_soc, diff_energy)."""
        if self._data_error:
            return (None, None, None, None)
        if mileage is None:
            return (None, None, None, None)
        if self._remaining_km >= mileage:
            surplus = (self._remaining_km - mileage) * WH_PER_KM
            return (True, 80.0, 60.0, -surplus)
        deficit = (mileage - self._remaining_km) * WH_PER_KM
        return (False, 40.0, 80.0, deficit)

    def is_car_plugged(self, time=None, for_duration=None):
        """Return plugged status from test configuration."""
        return self._is_plugged

    # Bind real methods from QSCar so we exercise the actual logic.
    user_set_person_for_car = QSCar.user_set_person_for_car
    _is_person_authorized_for_car = QSCar._is_person_authorized_for_car
    _fix_user_selected_person_from_forecast = QSCar._fix_user_selected_person_from_forecast


class _FakePerson:
    """Minimal person with a fixed forecast and car authorizations."""

    def __init__(self, name, preferred_car, authorized_car_names, forecast_leave_time=None, forecast_mileage=None):
        self.name = name
        self.preferred_car = preferred_car
        self.authorized_cars = list(authorized_car_names)
        self._forecast_leave = forecast_leave_time
        self._forecast_mileage = forecast_mileage
        self.home = None

    def get_authorized_cars(self):
        if self.home is None:
            return []
        return [c for c in self.home._cars if c.name in self.authorized_cars]

    def update_person_forecast(self, time=None, force_update=False):
        return self._forecast_leave, self._forecast_mileage

    async def notify_of_forecast_if_needed(self, **kwargs):
        pass


class _FakeHome:
    """Home that uses the REAL QSHome allocation methods."""

    def __init__(self, cars, persons):
        self._cars = list(cars)
        self._persons = list(persons)
        self._last_persons_car_allocation = {}
        self._last_persons_car_allocation_time = None

        for car in self._cars:
            car.home = self
        for person in self._persons:
            person.home = self
            if person.preferred_car and person.preferred_car not in person.authorized_cars:
                person.authorized_cars.append(person.preferred_car)

    # Bind all the real QSHome methods needed by the allocation pipeline.
    compute_and_set_best_persons_cars_allocations = QSHome.compute_and_set_best_persons_cars_allocations
    _build_raw_energy_matrix = staticmethod(QSHome._build_raw_energy_matrix)
    _finalize_cost_matrix = staticmethod(QSHome._finalize_cost_matrix)
    _compute_assignment_energy = staticmethod(QSHome._compute_assignment_energy)
    get_person_by_name = QSHome.get_person_by_name
    get_car_by_name = QSHome.get_car_by_name
    get_preferred_person_for_car = QSHome.get_preferred_person_for_car


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _person_name(car):
    """Return the person name assigned to a car, or None."""
    if car.current_forecasted_person is None:
        return None
    return car.current_forecasted_person.name


def _build_scenario():
    """Build the 4-car / 4-person test scenario.

    Cars:
      Tesla   – no charger, 200 km remaining
      Twingo  – charger,     95 km remaining
      Zoe     – charger,    150 km remaining
      IDBuzz  – charger,     10 km remaining

    Persons (all depart at 07:30):
      Arthur  – drives Zoe & Twingo, prefers Twingo, needs 100 km
      Magali  – drives all four,     prefers Zoe,    needs  20 km
      Thomas  – drives all four,     prefers Tesla,  needs  30 km
      Brice   – drives Twingo & Zoe, no forecast
    """
    tesla = _FakeCar("Tesla", remaining_km=200, has_charger=False)
    # 95 km: Arthur's 100 km trip needs 750 Wh here — within the preferred
    # threshold, so his preferred Twingo wins for the right reason (QS-351).
    twingo = _FakeCar("Twingo", remaining_km=95, has_charger=True)
    zoe = _FakeCar("Zoe", remaining_km=150, has_charger=True)
    idbuzz = _FakeCar("IDBuzz", remaining_km=10, has_charger=True)

    leave = datetime.now(UTC).replace(hour=7, minute=30, second=0) + timedelta(days=1)

    arthur = _FakePerson(
        "Arthur",
        preferred_car="Twingo",
        authorized_car_names=["Zoe", "Twingo"],
        forecast_leave_time=leave,
        forecast_mileage=100.0,
    )
    magali = _FakePerson(
        "Magali",
        preferred_car="Zoe",
        authorized_car_names=["Tesla", "Twingo", "Zoe", "IDBuzz"],
        forecast_leave_time=leave,
        forecast_mileage=20.0,
    )
    thomas = _FakePerson(
        "Thomas",
        preferred_car="Tesla",
        authorized_car_names=["Tesla", "Twingo", "Zoe", "IDBuzz"],
        forecast_leave_time=leave,
        forecast_mileage=30.0,
    )
    brice = _FakePerson(
        "Brice",
        preferred_car=None,
        authorized_car_names=["Twingo", "Zoe"],
        forecast_leave_time=None,
        forecast_mileage=None,
    )

    cars = [tesla, twingo, zoe, idbuzz]
    persons = [arthur, magali, thomas, brice]
    home = _FakeHome(cars, persons)

    return home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSetUserPersonEdgeCases:
    """Cover edge-case branches in user_set_person_for_car."""

    @pytest.mark.asyncio
    async def test_invalid_person_name_becomes_force_no_person(self):
        """Passing an unknown person name should log an error and convert to
        FORCE_CAR_NO_PERSON_ATTACHED (car.py lines 316-317)."""
        home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice = _build_scenario()
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        await twingo.user_set_person_for_car("GhostPerson")

        assert twingo.get_user_originated("person_name") == FORCE_CAR_NO_PERSON_ATTACHED

    @pytest.mark.asyncio
    async def test_same_value_noop(self):
        """Calling user_set_person_for_car with the already-set value should
        return immediately without touching the allocation (car.py line 322)."""
        home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice = _build_scenario()
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        await twingo.user_set_person_for_car("Arthur")
        assert twingo.get_user_originated("person_name") == "Arthur"

        # Call again with the same value -- should be a no-op
        await twingo.user_set_person_for_car("Arthur")
        assert twingo.get_user_originated("person_name") == "Arthur"

    @pytest.mark.asyncio
    async def test_forecasted_matches_skips_reallocation(self):
        """When the manual selection matches the already-forecasted person,
        no reallocation is needed (car.py line 331)."""
        home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice = _build_scenario()
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        # Arthur is auto-assigned to Twingo (his preferred car);
        # manually confirming should skip realloc
        assert _person_name(twingo) == "Arthur"
        await twingo.user_set_person_for_car("Arthur")
        assert twingo.get_user_originated("person_name") == "Arthur"
        assert _person_name(twingo) == "Arthur"


class TestPersonSwapNotification:
    """Cover lines 2350-2352: both old and new person tracked when a car's
    assigned person changes from A to B."""

    @pytest.mark.asyncio
    async def test_swap_tracks_both_old_and_new_person(self):
        """When a forecast change forces two cars to swap their assigned
        persons, both the outgoing and incoming person on each car must be
        added to the notification list.

        Step 1 -- PersonX needs 70km (only CarA covers), PersonY needs 30km:
          CarA(80km)->PersonX, CarB(40km)->PersonY

        Step 2 -- swap mileages so PersonY needs 70km, PersonX needs 30km:
          CarA(80km)->PersonY, CarB(40km)->PersonX

        This exercises lines 2350-2352 (new person on a swapped car).
        """
        time_now = datetime(2026, 3, 15, 23, 00, tzinfo=pytz.UTC)
        leave = datetime(2026, 3, 16, 7, 30, tzinfo=pytz.UTC)
        # leave = datetime.now(timezone.utc).replace(hour=7, minute=30) + timedelta(days=1)

        car_a = _FakeCar("CarA", remaining_km=80, has_charger=False)
        car_b = _FakeCar("CarB", remaining_km=40, has_charger=False)

        person_x = _FakePerson(
            "PersonX",
            preferred_car=None,
            authorized_car_names=["CarA", "CarB"],
            forecast_leave_time=leave,
            forecast_mileage=70.0,
        )
        person_y = _FakePerson(
            "PersonY",
            preferred_car=None,
            authorized_car_names=["CarA", "CarB"],
            forecast_leave_time=leave,
            forecast_mileage=30.0,
        )

        home = _FakeHome([car_a, car_b], [person_x, person_y])

        # Step 1: PersonX needs 70km -> only CarA covers -> CarA->PersonX, CarB->PersonY
        await home.compute_and_set_best_persons_cars_allocations(time=time_now, force_update=True)
        assert _person_name(car_a) == "PersonX"
        assert _person_name(car_b) == "PersonY"

        # Step 2: swap the mileages
        person_x._forecast_mileage = 30.0
        person_y._forecast_mileage = 70.0

        notified = []
        orig_notify = _FakePerson.notify_of_forecast_if_needed

        async def _capture_notify(self_person, **kwargs):
            notified.append(self_person.name)
            return await orig_notify(self_person, **kwargs)

        _FakePerson.notify_of_forecast_if_needed = _capture_notify
        try:
            await home.compute_and_set_best_persons_cars_allocations(time=time_now, force_update=True)
        finally:
            _FakePerson.notify_of_forecast_if_needed = orig_notify

        # Allocation should have swapped
        assert _person_name(car_a) == "PersonY", f"CarA should now be PersonY, got {_person_name(car_a)}"
        assert _person_name(car_b) == "PersonX", f"CarB should now be PersonX, got {_person_name(car_b)}"
        assert "PersonX" in notified, "PersonX should be notified of the swap"
        assert "PersonY" in notified, "PersonY should be notified of the swap"


class TestNoPersonMultipleCars:
    """Regression tests for issue #260: setting "no person" on a second car
    must not clear the first car's user-originated "no person" selection."""

    @pytest.mark.asyncio
    async def test_no_person_on_two_cars_preserves_both(self):
        """AC1: setting FORCE_CAR_NO_PERSON_ATTACHED on Twingo and then Zoe
        must keep BOTH cars pinned to "no person" — the second call must not
        wipe the first car's user-originated state."""
        home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice = _build_scenario()
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        # Both cars must hold a forecasted person, otherwise the early
        # return in user_set_person_for_car skips the clearing loop and the
        # test would pass even on unfixed code.
        assert _person_name(twingo) == "Arthur"
        assert _person_name(zoe) == "Magali"

        await twingo.user_set_person_for_car(FORCE_CAR_NO_PERSON_ATTACHED)

        # Intermediate state: Twingo is pinned to "no person".
        assert twingo.get_user_originated("person_name") == FORCE_CAR_NO_PERSON_ATTACHED
        assert twingo.current_forecasted_person is None

        await zoe.user_set_person_for_car(FORCE_CAR_NO_PERSON_ATTACHED)

        # Final state: Twingo's manual "no person" survived the second call
        # (the forced reallocation must keep it pinned), and Zoe is also
        # pinned to "no person".
        assert twingo.get_user_originated("person_name") == FORCE_CAR_NO_PERSON_ATTACHED
        assert twingo.current_forecasted_person is None
        assert zoe.get_user_originated("person_name") == FORCE_CAR_NO_PERSON_ATTACHED
        assert zoe.current_forecasted_person is None

    @pytest.mark.asyncio
    async def test_no_person_preserves_other_manual_person(self):
        """AC2: setting "no person" on Zoe must not touch Twingo's manual
        real-person assignment."""
        home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice = _build_scenario()
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        await twingo.user_set_person_for_car("Arthur")
        assert twingo.get_user_originated("person_name") == "Arthur"

        await zoe.user_set_person_for_car(FORCE_CAR_NO_PERSON_ATTACHED)

        assert twingo.get_user_originated("person_name") == "Arthur"
        assert _person_name(twingo) == "Arthur"
        assert zoe.get_user_originated("person_name") == FORCE_CAR_NO_PERSON_ATTACHED
        assert zoe.current_forecasted_person is None

    @pytest.mark.asyncio
    async def test_no_person_frees_person_for_reallocation(self):
        """AC4: setting "no person" on Twingo must still trigger a full
        reallocation of the other cars. Arthur (displaced from Twingo) lands
        on Zoe deterministically: with Twingo pinned out, Zoe is Arthur's
        only remaining authorized car. Magali, displaced from Zoe, lands on
        IDBuzz."""
        home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice = _build_scenario()
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        assert _person_name(twingo) == "Arthur"
        assert _person_name(zoe) == "Magali"

        await twingo.user_set_person_for_car(FORCE_CAR_NO_PERSON_ATTACHED)

        # Twingo stays pinned to "no person".
        assert twingo.get_user_originated("person_name") == FORCE_CAR_NO_PERSON_ATTACHED
        assert twingo.current_forecasted_person is None
        # Arthur was reallocated to Zoe (his only remaining authorized car).
        assert _person_name(zoe) == "Arthur"
        # Magali, displaced from Zoe, landed on IDBuzz.
        assert _person_name(idbuzz) == "Magali"


class TestCacheHitReApply:
    """Cover the cache-hit re-apply branch (home.py lines 2358-2367)."""

    @pytest.mark.asyncio
    async def test_corrupted_person_restored_from_cache(self):
        """If current_forecasted_person is overwritten between allocation runs,
        a cache-hit call should restore it from the cached result."""
        home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice = _build_scenario()

        await home.compute_and_set_best_persons_cars_allocations(force_update=True)
        assert _person_name(tesla) == "Thomas"

        # Corrupt Tesla's assignment (simulates what car.reset() used to do)
        tesla.current_forecasted_person = None

        # Call again without force_update -- cache hit should re-apply
        await home.compute_and_set_best_persons_cars_allocations(force_update=False)
        assert _person_name(tesla) == "Thomas"


class TestPersonCarAllocationScenario:
    """Reproduce the 4-car / 4-person scenario from the bug report."""

    @pytest.mark.asyncio
    async def test_automatic_assignment(self):
        """First automatic allocation should produce:

        Thomas → Tesla  (preferred, no charger, covered)
        Arthur → Twingo (preferred, plugged, needs 750 Wh — within the preferred threshold)
        Magali → Zoe    (preferred, plugged, covered)
        IDBuzz → nobody (10 km, nobody left needs it)
        """
        home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice = _build_scenario()

        result = await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        assert _person_name(tesla) == "Thomas", f"Tesla should be Thomas, got {_person_name(tesla)}"
        assert _person_name(twingo) == "Arthur", f"Twingo should be Arthur, got {_person_name(twingo)}"
        assert _person_name(zoe) == "Magali", f"Zoe should be Magali, got {_person_name(zoe)}"
        assert _person_name(idbuzz) is None, f"IDBuzz should be unassigned, got {_person_name(idbuzz)}"

    @pytest.mark.asyncio
    async def test_manual_override_triggers_reassignment(self):
        """After manually assigning Arthur to Twingo the system should
        redistribute:

        Arthur → Twingo  (manual override)
        Magali → Zoe     (was on Twingo, moves to her preferred car)
        Thomas → Tesla   (unchanged)
        IDBuzz → nobody
        """
        home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice = _build_scenario()

        # --- step 1: run the initial automatic allocation ---
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        # --- step 2: manually override Arthur → Twingo ---
        await twingo.user_set_person_for_car("Arthur")

        # --- verify the reassignment ---
        assert twingo.get_user_originated("person_name") == "Arthur"
        assert _person_name(twingo) == "Arthur", f"Twingo should be Arthur (manual), got {_person_name(twingo)}"
        assert _person_name(zoe) == "Magali", f"Zoe should be Magali after reassignment, got {_person_name(zoe)}"
        assert _person_name(tesla) == "Thomas", f"Tesla should still be Thomas, got {_person_name(tesla)}"
        assert _person_name(idbuzz) is None, f"IDBuzz should be unassigned, got {_person_name(idbuzz)}"


class TestDefaultChargeNoPersonAssigned:
    """Cover Story 4.1 / Issue #30: when no person is assigned to a plugged car,
    set the target charge to car_default_charge."""

    @pytest.mark.asyncio
    async def test_plugged_no_person_gets_default_charge(self):
        """AC1: plugged car with no person → target set to car_default_charge."""
        time_now = datetime(2026, 3, 15, 23, 0, tzinfo=pytz.UTC)
        leave = datetime(2026, 3, 16, 7, 30, tzinfo=pytz.UTC)

        # CarA: plugged, no charger-person match expected
        car_a = _FakeCar("CarA", remaining_km=200, has_charger=True, is_plugged=True, default_charge=80.0)
        # Only one person, authorized only for CarB
        car_b = _FakeCar("CarB", remaining_km=50, has_charger=True, is_plugged=True, default_charge=90.0)

        person = _FakePerson(
            "Alice",
            preferred_car="CarB",
            authorized_car_names=["CarB"],
            forecast_leave_time=leave,
            forecast_mileage=100.0,
        )

        home = _FakeHome([car_a, car_b], [person])
        await home.compute_and_set_best_persons_cars_allocations(time=time_now, force_update=True)

        # Alice → CarB (only authorized car)
        assert _person_name(car_b) == "Alice"
        # CarA has no person, is plugged → should get default charge
        assert _person_name(car_a) is None
        assert car_a._next_charge_target == 80.0

    @pytest.mark.asyncio
    async def test_force_no_person_plugged_gets_default_charge(self):
        """AC2: force-no-person on plugged car → target set to car_default_charge."""
        time_now = datetime(2026, 3, 15, 23, 0, tzinfo=pytz.UTC)
        leave = datetime(2026, 3, 16, 7, 30, tzinfo=pytz.UTC)

        car = _FakeCar("MyCar", remaining_km=100, has_charger=True, is_plugged=True, default_charge=75.0)
        person = _FakePerson(
            "Bob",
            preferred_car="MyCar",
            authorized_car_names=["MyCar"],
            forecast_leave_time=leave,
            forecast_mileage=50.0,
        )

        home = _FakeHome([car], [person])

        # Set force-no-person before allocation
        car.set_user_originated("person_name", FORCE_CAR_NO_PERSON_ATTACHED)
        await home.compute_and_set_best_persons_cars_allocations(time=time_now, force_update=True)

        assert _person_name(car) is None
        assert car._next_charge_target == 75.0

    @pytest.mark.asyncio
    async def test_user_originated_charge_target_preserved(self):
        """AC3: user already set a charge target → system must NOT overwrite it."""
        time_now = datetime(2026, 3, 15, 23, 0, tzinfo=pytz.UTC)

        car = _FakeCar("MyCar", remaining_km=100, has_charger=True, is_plugged=True, default_charge=100.0)
        home = _FakeHome([car], [])

        # User explicitly set a charge target of 60%
        car._next_charge_target = 60
        await home.compute_and_set_best_persons_cars_allocations(time=time_now, force_update=True)

        assert car._next_charge_target == 60, "User-set target must be preserved"

    @pytest.mark.asyncio
    async def test_user_originated_charge_time_preserved(self):
        """AC4: user set a charge time → system must NOT overwrite it.
        The system still sets the default target if none exists, but the
        charge time is untouched (this code never sets charge time)."""
        time_now = datetime(2026, 3, 15, 23, 0, tzinfo=pytz.UTC)

        car = _FakeCar("MyCar", remaining_km=100, has_charger=True, is_plugged=True, default_charge=90.0)
        home = _FakeHome([car], [])

        # User set a charge time (stored as user_originated)
        car.set_user_originated("charge_time", "2026-03-16T07:00:00+00:00")
        await home.compute_and_set_best_persons_cars_allocations(time=time_now, force_update=True)

        # Charge time must be untouched
        assert car.get_user_originated("charge_time") == "2026-03-16T07:00:00+00:00"
        # Default target still applies since _next_charge_target was None
        assert car._next_charge_target == 90.0

    @pytest.mark.asyncio
    async def test_person_later_assigned_replaces_default(self):
        """AC5: default charge was applied → person later assigned → default replaced."""
        time_now = datetime(2026, 3, 15, 23, 0, tzinfo=pytz.UTC)
        leave = datetime(2026, 3, 16, 7, 30, tzinfo=pytz.UTC)

        car = _FakeCar("MyCar", remaining_km=50, has_charger=True, is_plugged=True, default_charge=100.0)
        # Initially no person with forecast
        person = _FakePerson(
            "Alice",
            preferred_car="MyCar",
            authorized_car_names=["MyCar"],
            forecast_leave_time=None,
            forecast_mileage=None,
        )

        home = _FakeHome([car], [person])

        # Round 1: no forecast → no person → default charge applied
        await home.compute_and_set_best_persons_cars_allocations(time=time_now, force_update=True)
        assert _person_name(car) is None
        assert car._next_charge_target == 100.0

        # Round 2: person now has a forecast → gets assigned
        person._forecast_leave = leave
        person._forecast_mileage = 80.0
        await home.compute_and_set_best_persons_cars_allocations(time=time_now, force_update=True)
        assert _person_name(car) == "Alice"
        # The system-set default is replaced: _next_charge_target reset to None
        # so the person's forecast drives charging instead
        assert car._next_charge_target is None

    @pytest.mark.asyncio
    async def test_not_plugged_no_person_no_default(self):
        """AC6: car NOT plugged, no person → no default charge target set."""
        time_now = datetime(2026, 3, 15, 23, 0, tzinfo=pytz.UTC)

        car = _FakeCar("MyCar", remaining_km=100, has_charger=True, is_plugged=False, default_charge=80.0)
        home = _FakeHome([car], [])

        await home.compute_and_set_best_persons_cars_allocations(time=time_now, force_update=True)

        assert _person_name(car) is None
        assert car._next_charge_target is None, "Unplugged car should not get default charge"

    @pytest.mark.asyncio
    async def test_plugged_unknown_gets_default_charge(self):
        """D2: is_car_plugged returns None (sensor unavailable) → treat as
        potentially plugged and set default charge."""
        time_now = datetime(2026, 3, 15, 23, 0, tzinfo=pytz.UTC)

        car = _FakeCar("MyCar", remaining_km=100, has_charger=True, is_plugged=None, default_charge=80.0)
        home = _FakeHome([car], [])

        await home.compute_and_set_best_persons_cars_allocations(time=time_now, force_update=True)

        assert car._next_charge_target == 80.0, "Unknown plug state should be treated as plugged"

    @pytest.mark.asyncio
    async def test_energy_user_originated_prevents_clear(self):
        """D3: user set charge_target_energy → person assigned → system must
        NOT clear _next_charge_target even if it matches default."""
        time_now = datetime(2026, 3, 15, 23, 0, tzinfo=pytz.UTC)
        leave = datetime(2026, 3, 16, 7, 30, tzinfo=pytz.UTC)

        car = _FakeCar("MyCar", remaining_km=50, has_charger=True, is_plugged=True, default_charge=100.0)
        person = _FakePerson(
            "Alice",
            preferred_car="MyCar",
            authorized_car_names=["MyCar"],
            forecast_leave_time=None,
            forecast_mileage=None,
        )

        home = _FakeHome([car], [person])

        # Round 1: no forecast → no person → default charge applied
        await home.compute_and_set_best_persons_cars_allocations(time=time_now, force_update=True)
        assert car._next_charge_target == 100.0

        # User sets energy target (which snapshots current percent target)
        car.set_user_originated("charge_target_energy", 50.0)

        # Round 2: person now has forecast → gets assigned
        person._forecast_leave = leave
        person._forecast_mileage = 80.0
        await home.compute_and_set_best_persons_cars_allocations(time=time_now, force_update=True)

        assert _person_name(car) == "Alice"
        # The target must NOT be cleared because charge_target_energy is user-originated
        assert car._next_charge_target == 100.0


class TestPluggedCoveredPenalty:
    """QS-351 (event 2): a covered pair on a plugged car must be priced as an
    absolute tie-break (PLUGGED_COVERED_CAR_PENALTY_WH), never E_max-relative.

    Otherwise a person whose preferred plugged car already covers their trip is
    swapped onto a car that *needs* charging, manufacturing a grid charge (the
    incident of 2026-09-13 → 14, Tesla M3 at 11 kW).
    """

    @pytest.mark.asyncio
    async def test_covered_plugged_preferred_car_not_abandoned_for_car_needing_charge(self):
        """Red test A — derived from the 07:05:04 incident.

        With the old E_max-relative plugged penalty the energy-optimal pass
        swaps Magali off her covered, plugged IDBuzz onto the 37 % Tesla that
        needs charging. With the fix she stays on the IDBuzz.
        """
        zoe = _FakeCar("Zoe", remaining_km=175, has_charger=False)  # just unplugged; covers Arthur (99) and Brice (24)
        tesla = _FakeCar("Tesla", remaining_km=80, has_charger=True)  # plugged; covers Thomas (13), NOT Magali (113)
        idbuzz = _FakeCar("IDBuzz", remaining_km=200, has_charger=True)  # plugged; covers Magali and Thomas
        twingo = _FakeCar("Twingo", remaining_km=20, has_charger=False)  # Arthur's preferred; needs charging

        leave = datetime.now(UTC) + timedelta(hours=2)
        arthur = _FakePerson("Arthur", "Twingo", ["Zoe", "Twingo"], leave, 99.0)
        magali = _FakePerson("Magali", "IDBuzz", ["IDBuzz", "Tesla"], leave, 113.0)
        thomas = _FakePerson("Thomas", "Tesla", ["Tesla", "IDBuzz"], leave, 13.0)
        brice = _FakePerson("Brice", "Zoe", ["Zoe", "Twingo"], leave, 24.0)

        home = _FakeHome([zoe, tesla, idbuzz, twingo], [arthur, magali, thomas, brice])
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        # The first two assertions make the test discriminating: without them
        # the last two are exactly the *preferred* pass's output.
        assert _person_name(zoe) == "Arthur"  # energy-optimal pass WON …
        assert _person_name(twingo) == "Brice"
        assert _person_name(idbuzz) == "Magali"  # … and kept Magali on her plugged, already-covered car
        assert _person_name(tesla) == "Thomas"  # the car that needs charging is not handed to Magali

    @pytest.mark.asyncio
    async def test_preferred_plugged_car_needing_over_threshold_is_left_for_covered_car(self):
        """Red test A2 — the intended behaviour change made explicit.

        A preferred plugged car needing more than the threshold is left for a
        covered car: Arthur's Twingo needs 4.5 kWh (> 1 kWh), so the
        energy-optimal pass moves him onto the covered Zoe and Magali onto the
        covered Twingo. Masked before the fix by the inverted penalty.
        """
        home, tesla, twingo, zoe, idbuzz, *_ = _build_scenario()
        twingo._remaining_km = 70  # pre-fix premise: Arthur needs 4500 Wh on his preferred Twingo
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        assert _person_name(zoe) == "Arthur"  # 4.5 kWh > 1 kWh: energy-optimal wins, Zoe covers him
        assert _person_name(twingo) == "Magali"  # Twingo covers her 20 km
        assert _person_name(tesla) == "Thomas"
        assert _person_name(idbuzz) is None

    @pytest.mark.asyncio
    async def test_all_covered_preferred_plugged_car_kept_when_e_max_is_zero(self):
        """Invariant pin (not a red test): when everyone is covered (E_max == 0)
        a person is not moved off their preferred plugged car.

        Fixture order matters — the unplugged car is listed first so the
        E_max == 0 tie at penalty 1.0 breaks against the plugged car and the pin
        is strict (pins PLUGGED_COVERED_CAR_PENALTY_WH < 1.0). The binding
        constraint is pass 2's ``n·(E_max + 1.0 + PLUGGED) + eps`` offset (at
        E_max == 0, n == 1 that is 1.25 + 1.0 = 2.25).
        """
        leave = datetime.now(UTC) + timedelta(hours=2)

        y = _FakeCar("Y", remaining_km=200, has_charger=False)  # non-preferred, unplugged, covers, free
        x = _FakeCar("X", remaining_km=200, has_charger=True)  # P's preferred car, plugged, covers
        home = _FakeHome([y, x], [_FakePerson("P", "X", ["X", "Y"], leave, 50.0)])
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)
        assert _person_name(x) == "P"  # E_max == 0: not moved off the preferred plugged car (pins penalty < 1.0)

        # lower bound: with NO preference the unplugged covered car must win (pins penalty > 0)
        home2 = _FakeHome(
            [_FakeCar("X", 200, True), _FakeCar("Y", 200, False)],
            [_FakePerson("P", None, ["X", "Y"], leave, 50.0)],
        )
        await home2.compute_and_set_best_persons_cars_allocations(force_update=True)
        assert _person_name(home2._cars[1]) == "P"


class TestAllocationUnits:
    """QS-351 (defect B): the two-pass gate and tie-break are in Wh, not kWh.

    diff_energy is Wh, so the threshold must be 1000 Wh — a disagreement of a
    few hundred Wh between the passes must keep the preferred assignment.
    """

    @pytest.mark.asyncio
    async def test_preferred_car_kept_when_energy_gap_below_threshold_wh(self, caplog):
        """Red test B — gap 750 Wh <= 1000 Wh must keep the preferred pass."""
        leave = datetime.now(UTC) + timedelta(hours=2)

        c1 = _FakeCar("C1", remaining_km=60, has_charger=True)  # P1 needs 6000 Wh, P2 needs 4500 Wh
        c2 = _FakeCar("C2", remaining_km=95, has_charger=True)  # P1 needs 750 Wh, P2 covered
        p1 = _FakePerson("P1", "C1", ["C1", "C2"], leave, 100.0)
        p2 = _FakePerson("P2", "C2", ["C1", "C2"], leave, 90.0)
        home = _FakeHome([c1, c2], [p1, p2])

        with caplog.at_level(logging.INFO):
            await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        # preferred pass = 6000 Wh, energy-optimal = 5250 Wh: gap 750 Wh <= 1000 Wh -> preferred must win
        assert _person_name(c1) == "P1"
        assert _person_name(c2) == "P2"
        assert "energy diff 750.00 Wh <= threshold 1000.00 Wh" in caplog.text

    @pytest.mark.asyncio
    async def test_energy_optimal_wins_when_gap_above_threshold_wh(self):
        """Companion — brackets the constant from above: gap 1050 Wh > 1000 Wh
        must flip to the energy-optimal pass (green before and after the fix)."""
        leave = datetime.now(UTC) + timedelta(hours=2)

        c1 = _FakeCar("C1", remaining_km=60, has_charger=True)  # P1 needs 6000 Wh, P2 needs 4500 Wh
        c2 = _FakeCar("C2", remaining_km=97, has_charger=True)  # P1 needs 450 Wh, P2 covered
        p1 = _FakePerson("P1", "C1", ["C1", "C2"], leave, 100.0)
        p2 = _FakePerson("P2", "C2", ["C1", "C2"], leave, 90.0)
        home = _FakeHome([c1, c2], [p1, p2])

        await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        # gap 6000 - (450 + 4500) = 1050 Wh > 1000 Wh -> energy-optimal
        assert _person_name(c1) == "P2"
        assert _person_name(c2) == "P1"


class TestPass1TieBreak:
    """QS-351 review-fix #01 (should-fix): the pass-1 preferred-car bias must be
    a genuine ordering tie-break, not an energy-scale bias.

    If the bias (PASS1_PREFERRED_CAR_PENALTY_WH) exceeds a per-person real need,
    pass 1 keeps everyone on their preferred car, corrupting
    total_energy_optimal; the gate then under-measures the real aggregate saving
    and adopts the preferred assignment, spreading charging the true optimum
    would avoid. The bias must stay well below any real need (n·PASS1 <<
    THRESHOLD for any realistic n).
    """

    @pytest.mark.asyncio
    async def test_pass1_bias_does_not_corrupt_energy_optimum_for_small_per_person_needs(self):
        """12 persons, each preferring a plugged car needing 90 Wh with a covered
        non-preferred alternative. The true optimum charges 0 Wh (everyone on
        their covered car); a per-person bias above 90 Wh would instead keep
        everyone on their preferred car and charge 12×90 = 1080 Wh.
        """
        leave = datetime.now(UTC) + timedelta(hours=2)

        n = 12
        cars = []
        persons = []
        for i in range(n):
            # Pref{i}: plugged, needs (100 - 99.4)*150 = 90 Wh (below a 100 Wh bias)
            pref = _FakeCar(f"Pref{i}", remaining_km=99.4, has_charger=True)
            # Cov{i}: unplugged, covers the 100 km trip, non-preferred, free
            cov = _FakeCar(f"Cov{i}", remaining_km=500, has_charger=False)
            cars += [pref, cov]
            persons.append(_FakePerson(f"P{i}", f"Pref{i}", [f"Pref{i}", f"Cov{i}"], leave, 100.0))

        home = _FakeHome(cars, persons)
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        # True energy optimum (0 Wh total): everyone lands on their covered car,
        # every plugged preferred car is left unused.
        for i in range(n):
            assert _person_name(cars[2 * i]) is None, f"Pref{i} must be unused, got {_person_name(cars[2 * i])}"
            assert _person_name(cars[2 * i + 1]) == f"P{i}"

    def test_pass1_penalty_constants_are_ordering_tie_breaks(self):
        """QS-351 review-fix #02 (NTH-B): pin the constant relations directly so a
        future bump cannot silently re-introduce the energy-bias / tie classes.

        - PLUGGED < PASS1 keeps pass 1 deterministic at the covered tie (SF-A);
        - PASS1 + PLUGGED < 1.0 keeps both epsilons below the E_max+1 sentinel /
          pass-2 offset floor at E_max == 0;
        - n·(PASS1+PLUGGED) < THRESHOLD (relation (ii)): both epsilons apply to a
          non-preferred covered-plugged cell, so their *sum* is the per-person
          corruption bound; the documented max n is 1333 (N-5).
        """
        summed_epsilon = PASS1_PREFERRED_CAR_PENALTY_WH + PLUGGED_COVERED_CAR_PENALTY_WH
        assert PLUGGED_COVERED_CAR_PENALTY_WH < PASS1_PREFERRED_CAR_PENALTY_WH
        assert summed_epsilon < 1.0
        # relation (ii): the documented max n where n·(PASS1+PLUGGED) < THRESHOLD.
        assert 1333 * summed_epsilon < PREFERRED_CAR_ENERGY_THRESHOLD_WH <= 1334 * summed_epsilon

    @pytest.mark.asyncio
    async def test_covered_plugged_preferred_car_is_order_independent_with_energy_pass_adopted(self, caplog):
        """QS-351 review-fix #02 (SF-A): a preferred, covered, plugged car must not
        tie a non-preferred, covered, unplugged car in pass 1 — otherwise the pick
        depends on the Hungarian zero-scan order (car list order).

        Run with the energy pass adopted (person Q saves ~13.5 kWh > 1 kWh, flipping
        the gate) so pass 1's choice for P is what actually ships, and assert P
        stays on their preferred plugged car X for BOTH car orders.
        """
        leave = datetime.now(UTC) + timedelta(hours=2)

        def _build(car_order):
            catalogue = {
                "X": _FakeCar("X", remaining_km=300, has_charger=True),  # P's preferred, plugged, covers
                "Y": _FakeCar("Y", remaining_km=300, has_charger=False),  # non-preferred, unplugged, covers
                "Wpref": _FakeCar("Wpref", remaining_km=10, has_charger=True),  # Q's preferred, needs ~13.5 kWh
                "Zcov": _FakeCar("Zcov", remaining_km=300, has_charger=False),  # covers Q, unplugged
            }
            cars = [catalogue[name] for name in car_order]
            p = _FakePerson("P", "X", ["X", "Y"], leave, 100.0)
            q = _FakePerson("Q", "Wpref", ["Wpref", "Zcov"], leave, 100.0)
            return _FakeHome(cars, [p, q]), catalogue["X"], catalogue["Y"]

        for order in (["X", "Y", "Wpref", "Zcov"], ["Y", "X", "Wpref", "Zcov"]):
            home, x, y = _build(order)
            with caplog.at_level(logging.INFO):
                caplog.clear()
                await home.compute_and_set_best_persons_cars_allocations(force_update=True)
            assert _person_name(x) == "P", f"order {order}: P must stay on the preferred plugged car"
            assert _person_name(y) is None, f"order {order}: the non-preferred unplugged car must be free"
            # N-1: pin that pass 1 (energy) really is the adopted pass — otherwise
            # X->P would silently degrade into a pass-2 assertion.
            assert "using energy-optimal assignment" in caplog.text, f"order {order}: energy pass must be adopted"


class TestSentinelAndPass2Ordering:
    """QS-351 review-fix #03/#05 (SF-1, N-4): the pass-2 offset
    ``n·(E_max + 1.0 + PLUGGED) + eps`` must dominate the *aggregate* base spread
    for all n >= 1 and E_max >= 0 (so pass 2 maximises preferred-car count), and
    the plugged-covered nudge must apply to the no-need sentinel branches (-1/-2)
    too — so no allocation that ships depends on ``self._cars`` order.
    """

    @pytest.mark.asyncio
    async def test_pass2_no_tie_between_preferred_sentinel_and_covered_when_e_max_zero(self, caplog):
        """SF-1 repro 1 — E_max == 0, multi-person. P1's preferred car A is a -2
        data-error sentinel; a covered non-preferred car B ties it in pass 2
        (both E_max+1 == n*E_max+1 at E_max == 0). The pass-2 offset must break the
        tie toward the preferred car for every car order."""
        leave = datetime.now(UTC) + timedelta(hours=2)

        def _build(car_order):
            catalogue = {
                "A": _FakeCar("A", remaining_km=300, has_charger=False, data_error=True),  # P1 preferred, -2
                "B": _FakeCar("B", remaining_km=300, has_charger=False),  # covered, unplugged, non-preferred
                "C": _FakeCar("C", remaining_km=300, has_charger=False),  # covers P2
            }
            cars = [catalogue[name] for name in car_order]
            p1 = _FakePerson("P1", "A", ["A", "B"], leave, 100.0)
            p2 = _FakePerson("P2", "C", ["C"], leave, 100.0)
            return _FakeHome(cars, [p1, p2]), catalogue["A"]

        for order in (["A", "B", "C"], ["B", "A", "C"], ["C", "B", "A"]):
            home, a = _build(order)
            with caplog.at_level(logging.INFO):
                caplog.clear()
                await home.compute_and_set_best_persons_cars_allocations(force_update=True)
            assert _person_name(a) == "P1", f"order {order}: P1 must stay on the preferred car"
            assert "using preferred-car assignment" in caplog.text, f"order {order}: pass 2 must be adopted"

    @pytest.mark.asyncio
    async def test_pass2_no_tie_single_person_data_error_preferred_car_e_max_positive(self, caplog):
        """SF-1 repro 2 — E_max > 0, single person. P's preferred car X is a -2
        data-error sentinel (E_max+1); a covered non-preferred car Y ties it in
        pass 2 when n == 1 (E_max+1 == 1*E_max+1). A third car Z carries a real
        need so E_max > 0. The pass-2 offset must break the tie toward X."""
        leave = datetime.now(UTC) + timedelta(hours=2)

        def _build(car_order):
            catalogue = {
                "X": _FakeCar("X", remaining_km=300, has_charger=False, data_error=True),  # preferred, -2
                "Y": _FakeCar("Y", remaining_km=300, has_charger=False),  # covered, unplugged, non-preferred
                "Z": _FakeCar("Z", remaining_km=10, has_charger=True),  # real need -> E_max > 0
            }
            cars = [catalogue[name] for name in car_order]
            p = _FakePerson("P", "X", ["X", "Y", "Z"], leave, 100.0)
            return _FakeHome(cars, [p]), catalogue["X"]

        for order in (["X", "Y", "Z"], ["Y", "X", "Z"], ["Z", "Y", "X"]):
            home, x = _build(order)
            with caplog.at_level(logging.INFO):
                caplog.clear()
                await home.compute_and_set_best_persons_cars_allocations(force_update=True)
            assert _person_name(x) == "P", f"order {order}: P must stay on the preferred data-error car"
            assert "using preferred-car assignment" in caplog.text, f"order {order}: pass 2 must be adopted"

    @pytest.mark.asyncio
    async def test_far_future_forecast_person_prefers_unplugged_car_regardless_of_order(self):
        """N-4 — a far-future-forecast person (-1 on every authorised car) must not
        be handed a plugged car while an unplugged one sits free. With the plugged
        nudge applied to the -1 branch the person lands on an unplugged car for
        every ``self._cars`` order.

        The forecast carries a mileage (so the person enters the optimisation) but
        a leave time beyond FAR_FUTURE_FORECAST_THRESHOLD_S, which is exactly the
        -1 "no/far-future forecast" sentinel branch.
        """
        leave_far = datetime.now(UTC) + timedelta(hours=48)

        def _build(car_order):
            catalogue = {
                "Pa": _FakeCar("Pa", remaining_km=300, has_charger=True),  # plugged
                "Ua": _FakeCar("Ua", remaining_km=300, has_charger=False),  # unplugged
                "Ub": _FakeCar("Ub", remaining_km=300, has_charger=False),  # unplugged
                "Pb": _FakeCar("Pb", remaining_km=300, has_charger=True),  # plugged
            }
            cars = [catalogue[name] for name in car_order]
            # no preferred car, far-future forecast -> every authorised car is a -1 sentinel
            p = _FakePerson("P", None, ["Pa", "Ua", "Ub", "Pb"], leave_far, 100.0)
            return _FakeHome(cars, [p])

        for order in (
            ["Pa", "Ua", "Ub", "Pb"],
            ["Pb", "Ua", "Ub", "Pa"],
            ["Ua", "Pa", "Pb", "Ub"],
        ):
            home = _build(order)
            await home.compute_and_set_best_persons_cars_allocations(force_update=True)
            assigned = [c for c in home._cars if _person_name(c) == "P"]
            assert len(assigned) == 1, f"order {order}: P must be assigned exactly one car"
            assert assigned[0].charger is None, f"order {order}: P must land on an unplugged car, got {assigned[0].name}"

    @pytest.mark.asyncio
    async def test_pass2_offset_dominates_aggregate_spread(self, caplog):
        """QS-351 review-fix #05 (SF-1): the pass-2 offset must dominate the
        *aggregate* base spread, not just one cell's spread — Hungarian minimises
        total cost. Two perfect matchings can differ in all n cells, so one
        assignment's base cost exceeds another's by up to n·(E_max + 1.0 + PLUGGED)
        = n·M (#06 SF-1: the multiplier is n, not n-1). The per-cell offset
        (n·E_max + 1.0 + eps) left preferred matches on the table; the path-A
        offset (n·M + eps) maximises preferred-car count.

        Realisable E_max == 0 counterexample (found by sweep): with the shipped
        per-cell offset the pipeline returned only 1 preferred match where 2 are
        achievable. N-4: assert it for every ``self._cars`` order.
        """
        near = datetime.now(UTC) + timedelta(hours=2)
        far = datetime.now(UTC) + timedelta(hours=48)

        def _build(order):
            catalogue = {
                # c0/c3 unreadable-SOC (-2) plugged; c1 covered unplugged; c2 covered plugged.
                "c0": _FakeCar("c0", remaining_km=1000, has_charger=True, data_error=True),
                "c1": _FakeCar("c1", remaining_km=1000, has_charger=False),
                "c2": _FakeCar("c2", remaining_km=1000, has_charger=True),
                "c3": _FakeCar("c3", remaining_km=1000, has_charger=True, data_error=True),
            }
            cars = [catalogue[name] for name in order]
            # p0 far-future (its whole row is -1 sentinels); p1/p2 normal (covered).
            p0 = _FakePerson("p0", "c2", ["c2", "c3"], far, 100.0)
            p1 = _FakePerson("p1", "c2", ["c0", "c2"], near, 100.0)
            p2 = _FakePerson("p2", "c3", ["c1", "c3"], near, 100.0)
            return _FakeHome(cars, [p0, p1, p2]), cars

        for order in itertools.permutations(["c0", "c1", "c2", "c3"]):
            home, cars = _build(order)
            with caplog.at_level(logging.INFO):
                caplog.clear()
                await home.compute_and_set_best_persons_cars_allocations(force_update=True)
            preferred_matches = sum(
                1
                for c in cars
                if c.current_forecasted_person is not None and c.current_forecasted_person.preferred_car == c.name
            )
            # p1->c2 and p2->c3 are both achievable (max preferred-car count == 2).
            assert preferred_matches == 2, f"order {order}: pass 2 must maximise preferred count, got {preferred_matches}"
            # E_max == 0 -> gate diff 0 -> the preferred pass ships.
            assert "using preferred-car assignment" in caplog.text, f"order {order}: preferred pass must be adopted"

    @pytest.mark.asyncio
    async def test_pass2_offset_multiplier_is_n_not_n_minus_1(self, caplog):
        """QS-351 review-fix #06 (SF-1b): pin that the pass-2 offset multiplier is
        ``len(p_s)`` (n), not (n-1). Single-person witness: the person's preferred
        car c_pref is a plugged data-error sentinel (base E_max + 1.0 + PLUGGED),
        a non-preferred car needs the whole E_max (base E_max). Only an offset
        with the n multiplier (n == 1 -> 1·M + eps) keeps the person on the
        preferred car; forcing the multiplier to (n-1) -> eps = 1.0 would move
        them onto the needy car (recorded red under that mutation in the progress
        note). Order-independent (N-4).
        """
        near = datetime.now(UTC) + timedelta(hours=2)

        def _build(order):
            catalogue = {
                # needy: 0 km left for a 100 km trip -> needs 15000 Wh (E_max).
                "needy": _FakeCar("needy", remaining_km=0, has_charger=False),
                # preferred: plugged, unreadable SOC (-2 sentinel).
                "pref": _FakeCar("pref", remaining_km=1000, has_charger=True, data_error=True),
            }
            cars = [catalogue[name] for name in order]
            p = _FakePerson("solo", "pref", ["needy", "pref"], near, 100.0)
            return _FakeHome(cars, [p]), catalogue["pref"]

        for order in (["needy", "pref"], ["pref", "needy"]):
            home, pref = _build(order)
            with caplog.at_level(logging.INFO):
                caplog.clear()
                await home.compute_and_set_best_persons_cars_allocations(force_update=True)
            assert _person_name(pref) == "solo", f"order {order}: single person must keep the preferred car"
            assert "using preferred-car assignment" in caplog.text, f"order {order}: preferred pass must be adopted"
