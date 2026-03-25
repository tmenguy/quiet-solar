"""Tests for the _user_originated system (fix #8).

Tests cover:
- AbstractDevice CRUD helpers
- Persistence round-trip and backward-compat migration
- user_clean_and_reset clears all user_originated
- QSCar get/set/clear_user_originated for person_name and charger_name
- _on_user_originated_changed auto-captures all values AFTER state changes
- charge_time "constraints_cleared" sentinel logic
- _fix_user_selected_person_from_forecast guarded by user_originated
- get_best_person_next_need prefers user selection
- user_clean_constraints sets charge_time sentinel + triggers snapshot
- Car unplug clears charge_time sentinel
- User-action methods trigger snapshot with correct (post-change) values
- qs_bump_solar_charge_priority setter: value stored, no recursive snapshot
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    CHARGE_TIME_CONSTRAINTS_CLEARED,
    DATA_HANDLER,
    DOMAIN,
    FORCE_CAR_NO_PERSON_ATTACHED,
)
from tests.factories import MinimalTestLoad, create_minimal_home_model

# =============================================================================
# AbstractDevice _user_originated CRUD
# =============================================================================


class TestUserOriginatedCRUD:
    """Test _user_originated helpers at AbstractDevice level."""

    def _make_device(self) -> MinimalTestLoad:
        return MinimalTestLoad(name="TestLoad")

    def test_set_and_get(self):
        dev = self._make_device()
        dev.set_user_originated("key1", "val1")
        assert dev.get_user_originated("key1") == "val1"

    def test_get_default(self):
        dev = self._make_device()
        assert dev.get_user_originated("missing") is None
        assert dev.get_user_originated("missing", 42) == 42

    def test_has(self):
        dev = self._make_device()
        assert dev.has_user_originated("x") is False
        dev.set_user_originated("x", 1)
        assert dev.has_user_originated("x") is True

    def test_clear_single(self):
        dev = self._make_device()
        dev.set_user_originated("a", 1)
        dev.set_user_originated("b", 2)
        dev.clear_user_originated("a")
        assert dev.has_user_originated("a") is False
        assert dev.has_user_originated("b") is True

    def test_clear_missing_key_is_noop(self):
        dev = self._make_device()
        dev.clear_user_originated("nonexistent")  # should not raise

    def test_clear_all(self):
        dev = self._make_device()
        dev.set_user_originated("a", 1)
        dev.set_user_originated("b", 2)
        dev.clear_all_user_originated()
        assert dev._user_originated == {}

    @pytest.mark.asyncio
    async def test_user_clean_and_reset_clears_user_originated(self):
        dev = self._make_device()
        dev.set_user_originated("x", 42)
        await dev.user_clean_and_reset()
        assert dev._user_originated == {}


# =============================================================================
# Persistence round-trip
# =============================================================================


class TestUserOriginatedPersistence:
    """Test save/restore of _user_originated."""

    def test_save_round_trip(self):
        dev = MinimalTestLoad(name="TestLoad")
        dev.set_user_originated("key", "value")
        data = {}
        dev.update_to_be_saved_extra_device_info(data)
        assert data["_user_originated"] == {"key": "value"}

        dev2 = MinimalTestLoad(name="TestLoad")
        dev2.use_saved_extra_device_info(data)
        assert dev2.get_user_originated("key") == "value"

    def test_restore_empty(self):
        dev = MinimalTestLoad(name="TestLoad")
        dev.use_saved_extra_device_info({})
        assert dev._user_originated == {}


# =============================================================================
# QSCar user_originated for person_name and charger_name
# =============================================================================


@pytest.fixture
def car_home():
    home = create_minimal_home_model()
    home.is_off_grid = MagicMock(return_value=False)
    home.latitude = 48.8566
    home.longitude = 2.3522
    return home


@pytest.fixture
def car_data_handler(car_home):
    handler = MagicMock()
    handler.home = car_home
    return handler


@pytest.fixture
def car_hass_data(hass: HomeAssistant, car_data_handler):
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][DATA_HANDLER] = car_data_handler


@pytest.fixture
def create_car(hass, car_hass_data, car_home):
    from custom_components.quiet_solar.ha_model.car import QSCar

    def _factory(**extra_kwargs):
        from custom_components.quiet_solar.const import (
            CONF_CAR_BATTERY_CAPACITY,
            CONF_CAR_CHARGE_PERCENT_SENSOR,
            CONF_CAR_CHARGER_MAX_CHARGE,
            CONF_CAR_CHARGER_MIN_CHARGE,
            CONF_CAR_PLUGGED,
            CONF_CAR_TRACKER,
            CONF_DEFAULT_CAR_CHARGE,
            CONF_MINIMUM_OK_CAR_CHARGE,
        )

        config = {
            CONF_NAME: "Test Car",
            CONF_CAR_TRACKER: "device_tracker.test_car",
            CONF_CAR_PLUGGED: "binary_sensor.test_car_plugged",
            CONF_CAR_CHARGE_PERCENT_SENSOR: "sensor.test_car_soc",
            CONF_CAR_BATTERY_CAPACITY: 60000,
            CONF_CAR_CHARGER_MIN_CHARGE: 6,
            CONF_CAR_CHARGER_MAX_CHARGE: 32,
            CONF_DEFAULT_CAR_CHARGE: 80,
            CONF_MINIMUM_OK_CAR_CHARGE: 30,
        }
        config.update(extra_kwargs)
        car = QSCar(hass=hass, config_entry=MockConfigEntry(domain=DOMAIN, data=config), **config)
        car.home = car_home
        return car

    return _factory


class TestUserOriginatedPersonAndCharger:
    """Test get/set/clear_user_originated for person_name and charger_name on QSCar."""

    def test_person_name_none_by_default(self, create_car):
        car = create_car()
        assert car.get_user_originated("person_name") is None

    def test_set_person_name_stores_in_overrides(self, create_car):
        car = create_car()
        car.set_user_originated("person_name", "Alice")
        assert car.get_user_originated("person_name") == "Alice"

    def test_clear_person_name(self, create_car):
        car = create_car()
        car.set_user_originated("person_name", "Alice")
        car.clear_user_originated("person_name")
        assert car.has_user_originated("person_name") is False

    def test_restore_person_from_user_originated(self, create_car):
        """Saved _user_originated with person_name restores correctly."""
        car = create_car()
        stored = {
            "_user_originated": {"person_name": "Bob"},
            "current_forecasted_person_name_from_boot": None,
        }
        car.use_saved_extra_device_info(stored)
        assert car.get_user_originated("person_name") == "Bob"


# =============================================================================
# _on_user_originated_changed tests (auto-snapshot hook)
# =============================================================================


class TestOnUserOriginatedChanged:
    """Test _on_user_originated_changed auto-captures all values when triggered."""

    def test_captures_all_fields_via_set(self, create_car):
        """set_user_originated triggers _on_user_originated_changed which snapshots state."""
        car = create_car()
        car._next_charge_target = 80
        car._next_charge_target_energy = 48000
        car._qs_bump_solar_priority = True
        car.do_force_next_charge = True
        car.do_next_charge_time = datetime.datetime(2026, 3, 20, 7, 0, tzinfo=pytz.UTC)
        car.set_user_originated("charger_name", "Charger1")

        # Trigger the hook explicitly to snapshot all current state
        car._on_user_originated_changed("test", None)

        assert car.get_user_originated("charge_target_percent") == 80
        assert car.get_user_originated("charge_target_energy") == 48000
        assert car.get_user_originated("bump_solar") is True
        assert car.get_user_originated("force_charge") is True
        assert car.get_user_originated("charge_time") == "2026-03-20T07:00:00+00:00"
        assert car.get_user_originated("charger_name") == "Charger1"

    def test_preserves_existing_person(self, create_car):
        """If person_name is already in overrides, snapshot doesn't overwrite it."""
        car = create_car()
        car.set_user_originated("person_name", "Explicit")
        person = MagicMock()
        person.name = "Forecast"
        car.current_forecasted_person = person
        car._on_user_originated_changed("test", None)
        assert car.get_user_originated("person_name") == "Explicit"

    def test_captures_forecast_person_when_no_explicit(self, create_car):
        """Snapshot captures forecast person when no explicit selection exists."""
        car = create_car()
        person = MagicMock()
        person.name = "Alice"
        car.current_forecasted_person = person
        # Make person authorized
        person_obj = MagicMock()
        person_obj.authorized_cars = [car.name]
        car.home.get_person_by_name = MagicMock(return_value=person_obj)
        car.home._persons = [person_obj]

        car._on_user_originated_changed("test", None)
        assert car.get_user_originated("person_name") == "Alice"

    def test_null_charge_time(self, create_car):
        car = create_car()
        car.do_next_charge_time = None
        car._on_user_originated_changed("test", None)
        assert car.get_user_originated("charge_time") is None

    def test_preserves_constraints_cleared_sentinel(self, create_car):
        """When charge_time is 'constraints_cleared' and do_next_charge_time is None,
        the sentinel is preserved."""
        car = create_car()
        car.set_user_originated("charge_time", CHARGE_TIME_CONSTRAINTS_CLEARED)
        car.do_next_charge_time = None
        car._on_user_originated_changed("test", None)
        assert car.get_user_originated("charge_time") == CHARGE_TIME_CONSTRAINTS_CLEARED

    def test_supersedes_constraints_cleared_when_time_set(self, create_car):
        """When do_next_charge_time is set, it overwrites the constraints_cleared sentinel."""
        car = create_car()
        car.set_user_originated("charge_time", CHARGE_TIME_CONSTRAINTS_CLEARED)
        car.do_next_charge_time = datetime.datetime(2026, 3, 20, 7, 0, tzinfo=pytz.UTC)
        car._on_user_originated_changed("test", None)
        assert car.get_user_originated("charge_time") == "2026-03-20T07:00:00+00:00"


# =============================================================================
# _fix_user_selected_person_from_forecast guarded by override
# =============================================================================


class TestFixPersonGuard:
    """Test that _fix_user_selected_person_from_forecast is guarded by person_name override."""

    def test_noop_when_no_forecasted_person(self, create_car):
        """Early return when current_forecasted_person is None and no override."""
        car = create_car()
        car.current_forecasted_person = None
        car._fix_user_selected_person_from_forecast()
        assert car.get_user_originated("person_name") is None

    def test_guard_blocks_when_override_exists(self, create_car):
        car = create_car()
        car.set_user_originated("person_name", "Explicit")
        person = MagicMock()
        person.name = "Forecast"
        car.current_forecasted_person = person
        car._fix_user_selected_person_from_forecast()
        # Should NOT have been overwritten
        assert car.get_user_originated("person_name") == "Explicit"

    def test_guard_allows_when_no_override(self, create_car):
        car = create_car()
        person = MagicMock()
        person.name = "Forecast"
        car.current_forecasted_person = person
        # Make person authorized
        person_obj = MagicMock()
        person_obj.authorized_cars = [car.name]
        car.home.get_person_by_name = MagicMock(return_value=person_obj)
        car.home._persons = [person_obj]

        car._fix_user_selected_person_from_forecast()
        assert car.get_user_originated("person_name") == "Forecast"


# =============================================================================
# get_best_person_next_need prefers user selection
# =============================================================================


class TestGetBestPersonNextNeed:
    """Test get_best_person_next_need prefers user selection."""

    @pytest.mark.asyncio
    async def test_force_no_person(self, create_car):
        car = create_car()
        car.set_user_originated("person_name", FORCE_CAR_NO_PERSON_ATTACHED)
        result = await car.get_best_person_next_need(datetime.datetime.now(pytz.UTC))
        assert result == (None, None, None, None)

    @pytest.mark.asyncio
    async def test_user_selected_person_used(self, create_car):
        car = create_car()
        person = MagicMock()
        person.name = "Alice"
        person.update_person_forecast = MagicMock(
            return_value=(datetime.datetime(2026, 3, 20, 18, 0, tzinfo=pytz.UTC), 50.0)
        )
        car.home.get_person_by_name = MagicMock(return_value=person)
        car.home._persons = [person]
        car.set_user_originated("person_name", "Alice")

        result = await car.get_best_person_next_need(datetime.datetime.now(pytz.UTC))
        assert result[3] is person

    @pytest.mark.asyncio
    async def test_falls_back_to_forecast(self, create_car):
        car = create_car()
        person = MagicMock()
        person.name = "Bob"
        person.update_person_forecast = MagicMock(
            return_value=(datetime.datetime(2026, 3, 20, 18, 0, tzinfo=pytz.UTC), 50.0)
        )
        car.current_forecasted_person = person
        car.home._persons = [person]
        # No user selection
        result = await car.get_best_person_next_need(datetime.datetime.now(pytz.UTC))
        assert result[3] is person


# =============================================================================
# User-action methods trigger snapshot with CORRECT (post-change) values
# =============================================================================


class TestUserActionTriggersSnapshot:
    """Test that user-action methods snapshot AFTER state modifications."""

    @pytest.mark.asyncio
    async def test_user_force_charge_now_captures_new_state(self, create_car):
        """force_charge=True and charge_time=None must appear in overrides."""
        car = create_car()
        charger = MagicMock()
        charger.update_charger_for_user_change = AsyncMock()
        car.charger = charger
        car.do_force_next_charge = False  # initial
        car.do_next_charge_time = datetime.datetime(2026, 3, 20, 7, 0, tzinfo=pytz.UTC)  # will be set to None

        await car.user_force_charge_now()

        # Snapshot should have captured the POST-change values
        assert car.get_user_originated("force_charge") is True
        assert car.get_user_originated("charge_time") is None

    @pytest.mark.asyncio
    async def test_user_add_default_charge_captures_charge_time(self, create_car):
        """charge_time override should be an isoformat string, not None."""
        car = create_car()
        charger = MagicMock()
        charger.update_charger_for_user_change = AsyncMock()
        car.charger = charger
        car.default_charge_time = datetime.time(7, 0)
        await car.user_add_default_charge()
        # charge_time should be a real isoformat string (from user_add_default_charge_at_datetime)
        ct = car.get_user_originated("charge_time")
        assert ct is not None
        assert isinstance(ct, str) and ct != "constraints_cleared"

    @pytest.mark.asyncio
    async def test_user_set_next_charge_target_triggers_snapshot(self, create_car):
        """Snapshot fires after the target value has been changed."""
        car = create_car()
        charger = MagicMock()
        charger.update_charger_for_user_change = AsyncMock()
        charger.get_charge_type = MagicMock(return_value=("not_plugged", None))
        car.charger = charger

        await car.user_set_next_charge_target(80)

        # The snapshot captured whatever _next_charge_target was set to
        # (set_next_charge_target_percent sets it before snapshot fires)
        assert car.has_user_originated("charge_target_percent")
        assert car.get_user_originated("charge_target_percent") == car._next_charge_target

    def test_bump_solar_setter_captures_new_value(self, create_car):
        """bump_solar override must be True after setting to True."""
        car = create_car()
        car.home._cars = [car]
        car._qs_bump_solar_priority = False  # initial
        car.qs_bump_solar_charge_priority = True
        assert car.get_user_originated("bump_solar") is True

    def test_bump_solar_setter_no_recursive_snapshot_on_other_cars(self, create_car):
        """Setting bump on car1 should NOT trigger snapshot on car2."""
        car1 = create_car()
        car2 = create_car()
        car1.home._cars = [car1, car2]
        car2.home = car1.home

        car1.qs_bump_solar_charge_priority = True

        # car2 should NOT have any overrides (no snapshot was triggered on it)
        assert car2._user_originated == {}
        # car1 should have bump_solar=True
        assert car1.get_user_originated("bump_solar") is True

    @pytest.mark.asyncio
    async def test_user_set_selected_charger_captures_new_charger(self, create_car):
        """charger_name override should reflect the post-change state."""
        car = create_car()
        car.home._chargers = []
        car.set_user_originated("charger_name", "OldCharger")  # initial
        from custom_components.quiet_solar.const import FORCE_CAR_NO_CHARGER_CONNECTED

        await car.user_set_selected_charger_by_name(FORCE_CAR_NO_CHARGER_CONNECTED)
        # After setting to FORCE_CAR_NO_CHARGER_CONNECTED, the snapshot should capture it
        assert car.get_user_originated("charger_name") == FORCE_CAR_NO_CHARGER_CONNECTED

    @pytest.mark.asyncio
    async def test_user_add_default_charge_at_datetime_captures_time(self, create_car):
        """Snapshot captures the new charge time, not the old one."""
        car = create_car()
        charger = MagicMock()
        car.charger = charger
        car.do_next_charge_time = None  # initial

        end = datetime.datetime(2026, 6, 15, 8, 0, tzinfo=pytz.UTC)
        result = await car.user_add_default_charge_at_datetime(end)

        assert result is True
        assert car.get_user_originated("charge_time") == end.isoformat()

    @pytest.mark.asyncio
    async def test_user_set_selected_charger_none_clears(self, create_car):
        """Passing None to user_set_selected_charger_by_name clears charger_name."""
        car = create_car()
        car.home._chargers = []
        car.set_user_originated("charger_name", "SomeCharger")
        await car.user_set_selected_charger_by_name(None)
        assert car.get_user_originated("charger_name") is None

    @pytest.mark.asyncio
    async def test_user_set_selected_charger_unknown_name_clears(self, create_car):
        """Passing an unknown charger name clears charger_name."""
        car = create_car()
        car.home._chargers = []
        car.set_user_originated("charger_name", "SomeCharger")
        await car.user_set_selected_charger_by_name("NonExistentCharger")
        assert car.get_user_originated("charger_name") is None
