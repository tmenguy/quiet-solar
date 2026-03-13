"""End-to-end tests for the person-car allocation algorithm.

These tests exercise the real allocation pipeline (Hungarian algorithm,
pre-allocation of unplugged cars, manual overrides via set_user_person_for_car)
with lightweight fakes instead of full HA integration.
"""

import logging
from datetime import datetime, timedelta, timezone

import pytest

from custom_components.quiet_solar.ha_model.home import QSHome
from custom_components.quiet_solar.ha_model.car import QSCar
from custom_components.quiet_solar.const import (
    FORCE_CAR_NO_PERSON_ATTACHED,
    SENSOR_CAR_PERSON_FORECAST,
)

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight fakes – just enough state for the real allocation methods
# ---------------------------------------------------------------------------

KWH_PER_KM = 0.15


class _FakeCharger:
    """Stub charger so car.charger is truthy and set_user_person_for_car works."""

    async def update_charger_for_user_change(self):
        pass


class _FakeCar:
    """Minimal car with autonomy-based coverage computation."""

    def __init__(self, name, remaining_km, has_charger, is_invited=False):
        self.name = name
        self._remaining_km = remaining_km
        self.charger = _FakeCharger() if has_charger else None
        self.car_is_invited = is_invited
        self.user_selected_person_name_for_car = None
        self.current_forecasted_person = None
        self.home = None
        self.ha_entities = {}

    def get_adapt_target_percent_soc_to_reach_range_km(self, mileage, time):
        """Return (is_covered, current_soc, needed_soc, diff_energy)."""
        if mileage is None:
            return (None, None, None, None)
        if self._remaining_km >= mileage:
            surplus = (self._remaining_km - mileage) * KWH_PER_KM
            return (True, 80.0, 60.0, -surplus)
        deficit = (mileage - self._remaining_km) * KWH_PER_KM
        return (False, 40.0, 80.0, deficit)

    # Bind the real set_user_person_for_car from QSCar so we exercise the
    # actual override + reallocation logic (not a simplified version).
    set_user_person_for_car = QSCar.set_user_person_for_car


class _FakePerson:
    """Minimal person with a fixed forecast and car authorizations."""

    def __init__(self, name, preferred_car, authorized_car_names,
                 forecast_leave_time=None, forecast_mileage=None):
        self.name = name
        self.preferred_car = preferred_car
        self.authorized_cars = list(authorized_car_names)
        self._forecast_leave = forecast_leave_time
        self._forecast_mileage = forecast_mileage
        self.home = None

    def get_authorized_cars(self):
        if self.home is None:
            return []
        return [
            c for c in self.home._cars
            if c.name in self.authorized_cars
        ]

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
      Twingo  – charger,     70 km remaining
      Zoe     – charger,    150 km remaining
      IDBuzz  – charger,     10 km remaining

    Persons (all depart at 07:30):
      Arthur  – drives Zoe & Twingo, prefers Twingo, needs 100 km
      Magali  – drives all four,     prefers Zoe,    needs  20 km
      Thomas  – drives all four,     prefers Tesla,  needs  30 km
      Brice   – drives Twingo & Zoe, no forecast
    """
    tesla = _FakeCar("Tesla", remaining_km=200, has_charger=False)
    twingo = _FakeCar("Twingo", remaining_km=70, has_charger=True)
    zoe = _FakeCar("Zoe", remaining_km=150, has_charger=True)
    idbuzz = _FakeCar("IDBuzz", remaining_km=10, has_charger=True)

    leave = datetime.now(timezone.utc).replace(hour=7, minute=30, second=0) + timedelta(days=1)

    arthur = _FakePerson(
        "Arthur", preferred_car="Twingo",
        authorized_car_names=["Zoe", "Twingo"],
        forecast_leave_time=leave, forecast_mileage=100.0,
    )
    magali = _FakePerson(
        "Magali", preferred_car="Zoe",
        authorized_car_names=["Tesla", "Twingo", "Zoe", "IDBuzz"],
        forecast_leave_time=leave, forecast_mileage=20.0,
    )
    thomas = _FakePerson(
        "Thomas", preferred_car="Tesla",
        authorized_car_names=["Tesla", "Twingo", "Zoe", "IDBuzz"],
        forecast_leave_time=leave, forecast_mileage=30.0,
    )
    brice = _FakePerson(
        "Brice", preferred_car=None,
        authorized_car_names=["Twingo", "Zoe"],
        forecast_leave_time=None, forecast_mileage=None,
    )

    cars = [tesla, twingo, zoe, idbuzz]
    persons = [arthur, magali, thomas, brice]
    home = _FakeHome(cars, persons)

    return home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSetUserPersonEdgeCases:
    """Cover edge-case branches in set_user_person_for_car."""

    @pytest.mark.asyncio
    async def test_invalid_person_name_becomes_force_no_person(self):
        """Passing an unknown person name should log an error and convert to
        FORCE_CAR_NO_PERSON_ATTACHED (car.py lines 316-317)."""
        home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice = (
            _build_scenario()
        )
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        await twingo.set_user_person_for_car("GhostPerson")

        assert twingo.user_selected_person_name_for_car == FORCE_CAR_NO_PERSON_ATTACHED

    @pytest.mark.asyncio
    async def test_same_value_noop(self):
        """Calling set_user_person_for_car with the already-set value should
        return immediately without touching the allocation (car.py line 322)."""
        home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice = (
            _build_scenario()
        )
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        await twingo.set_user_person_for_car("Arthur")
        assert twingo.user_selected_person_name_for_car == "Arthur"

        # Call again with the same value -- should be a no-op
        await twingo.set_user_person_for_car("Arthur")
        assert twingo.user_selected_person_name_for_car == "Arthur"

    @pytest.mark.asyncio
    async def test_forecasted_matches_skips_reallocation(self):
        """When the manual selection matches the already-forecasted person,
        no reallocation is needed (car.py line 331)."""
        home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice = (
            _build_scenario()
        )
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        # Arthur is auto-assigned to Twingo (his preferred car);
        # manually confirming should skip realloc
        assert _person_name(twingo) == "Arthur"
        await twingo.set_user_person_for_car("Arthur")
        assert twingo.user_selected_person_name_for_car == "Arthur"
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
        leave = datetime.now(timezone.utc).replace(hour=7, minute=30) + timedelta(days=1)

        car_a = _FakeCar("CarA", remaining_km=80, has_charger=False)
        car_b = _FakeCar("CarB", remaining_km=40, has_charger=False)

        person_x = _FakePerson(
            "PersonX", preferred_car=None,
            authorized_car_names=["CarA", "CarB"],
            forecast_leave_time=leave, forecast_mileage=70.0,
        )
        person_y = _FakePerson(
            "PersonY", preferred_car=None,
            authorized_car_names=["CarA", "CarB"],
            forecast_leave_time=leave, forecast_mileage=30.0,
        )

        home = _FakeHome([car_a, car_b], [person_x, person_y])

        # Step 1: PersonX needs 70km -> only CarA covers -> CarA->PersonX, CarB->PersonY
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)
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
            await home.compute_and_set_best_persons_cars_allocations(force_update=True)
        finally:
            _FakePerson.notify_of_forecast_if_needed = orig_notify

        # Allocation should have swapped
        assert _person_name(car_a) == "PersonY", f"CarA should now be PersonY, got {_person_name(car_a)}"
        assert _person_name(car_b) == "PersonX", f"CarB should now be PersonX, got {_person_name(car_b)}"
        assert "PersonX" in notified, "PersonX should be notified of the swap"
        assert "PersonY" in notified, "PersonY should be notified of the swap"


class TestCacheHitReApply:
    """Cover the cache-hit re-apply branch (home.py lines 2358-2367)."""

    @pytest.mark.asyncio
    async def test_corrupted_person_restored_from_cache(self):
        """If current_forecasted_person is overwritten between allocation runs,
        a cache-hit call should restore it from the cached result."""
        home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice = (
            _build_scenario()
        )

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
        Arthur → Twingo (preferred, plugged, needs charging but preferred wins)
        Magali → Zoe    (preferred, plugged, covered)
        IDBuzz → nobody (10 km, nobody left needs it)
        """
        home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice = (
            _build_scenario()
        )

        result = await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        assert _person_name(tesla) == "Thomas", (
            f"Tesla should be Thomas, got {_person_name(tesla)}"
        )
        assert _person_name(twingo) == "Arthur", (
            f"Twingo should be Arthur, got {_person_name(twingo)}"
        )
        assert _person_name(zoe) == "Magali", (
            f"Zoe should be Magali, got {_person_name(zoe)}"
        )
        assert _person_name(idbuzz) is None, (
            f"IDBuzz should be unassigned, got {_person_name(idbuzz)}"
        )

    @pytest.mark.asyncio
    async def test_manual_override_triggers_reassignment(self):
        """After manually assigning Arthur to Twingo the system should
        redistribute:

        Arthur → Twingo  (manual override)
        Magali → Zoe     (was on Twingo, moves to her preferred car)
        Thomas → Tesla   (unchanged)
        IDBuzz → nobody
        """
        home, tesla, twingo, zoe, idbuzz, arthur, magali, thomas, brice = (
            _build_scenario()
        )

        # --- step 1: run the initial automatic allocation ---
        await home.compute_and_set_best_persons_cars_allocations(force_update=True)

        # --- step 2: manually override Arthur → Twingo ---
        await twingo.set_user_person_for_car("Arthur")

        # --- verify the reassignment ---
        assert twingo.user_selected_person_name_for_car == "Arthur"
        assert _person_name(twingo) == "Arthur", (
            f"Twingo should be Arthur (manual), got {_person_name(twingo)}"
        )
        assert _person_name(zoe) == "Magali", (
            f"Zoe should be Magali after reassignment, got {_person_name(zoe)}"
        )
        assert _person_name(tesla) == "Thomas", (
            f"Tesla should still be Thomas, got {_person_name(tesla)}"
        )
        assert _person_name(idbuzz) is None, (
            f"IDBuzz should be unassigned, got {_person_name(idbuzz)}"
        )
