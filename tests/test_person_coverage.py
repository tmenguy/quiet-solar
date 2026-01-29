"""Coverage-focused tests for QSPerson."""
from __future__ import annotations

from datetime import datetime, timedelta, time as dt_time
from types import SimpleNamespace

import pytest
import pytz

from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    DOMAIN,
    CONF_MOBILE_APP,
    CONF_MOBILE_APP_URL,
    CONF_PERSON_AUTHORIZED_CARS,
    CONF_PERSON_NOTIFICATION_TIME,
    CONF_PERSON_PREFERRED_CAR,
    CONF_PERSON_PERSON_ENTITY,
    MAX_PERSON_MILEAGE_HISTORICAL_DATA_DAYS,
    PERSON_NOTIFY_REASON_DAILY_REMINDER_FOR_CAR_NO_CHARGER,
    PERSON_NOTIFY_REASON_CHANGED_CAR,
)
from custom_components.quiet_solar.ha_model.car import QSCar
from custom_components.quiet_solar.ha_model.device import HADeviceMixin
from custom_components.quiet_solar.ha_model.home import QSHome
from custom_components.quiet_solar.ha_model.person import QSPerson
from custom_components.quiet_solar.home_model.constraints import (
    LoadConstraint,
    MultiStepsPowerLoadConstraintChargePercent,
)
from homeassistant.const import Platform


def _make_person(hass: HomeAssistant, **overrides) -> tuple[HomeAssistant, QSHome, QSPerson]:
    """Create hass, home, and person for tests."""
    hass.data.setdefault(DOMAIN, {})
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "Test Home"},
        title="Test Home",
    )
    home = QSHome(hass=hass, config_entry=config_entry, name="Test Home")
    person = QSPerson(
        hass=hass,
        home=home,
        config_entry=None,
        name="Test Person",
        **{
            CONF_PERSON_PERSON_ENTITY: "person.test_person",
            **overrides,
        },
    )
    return hass, home, person


def _make_car(hass: HomeAssistant, home: QSHome, name: str = "Test Car") -> QSCar:
    car = QSCar(
        hass=hass,
        home=home,
        config_entry=None,
        name=name,
        car_tracker="device_tracker.test_car",
    )
    home.add_device(car)
    return car


def test_init_normalizes_mobile_app_and_notification_time(hass: HomeAssistant) -> None:
    """Test config normalization for mobile app and notification time."""
    hass, home, person = _make_person(hass,
        **{
            CONF_MOBILE_APP: "notify",
            CONF_MOBILE_APP_URL: "qs",
            CONF_PERSON_NOTIFICATION_TIME: "08:30:00",
            CONF_PERSON_AUTHORIZED_CARS: ["Car A"],
            CONF_PERSON_PREFERRED_CAR: "Car B",
        }
    )

    assert person.mobile_app == "notify"
    assert person.mobile_app_url == "/qs"
    assert person.notification_dt_time == dt_time(8, 30, 0)
    assert "Car B" in person.authorized_cars

    _, _, person_empty = _make_person(hass,
        **{
            CONF_MOBILE_APP_URL: "",
        }
    )
    assert person_empty.mobile_app_url is None

    _, _, person_root = _make_person(hass,
        **{
            CONF_MOBILE_APP_URL: "/",
        }
    )
    assert person_root.mobile_app_url is None

    _, _, person_prefixed = _make_person(hass,
        **{
            CONF_MOBILE_APP_URL: "/already",
        }
    )
    assert person_prefixed.mobile_app_url == "/already"


def test_add_to_mileage_history_insert_update_and_trim(hass: HomeAssistant) -> None:
    """Test history insertion/update behavior and trimming."""
    _, _, person = _make_person(hass)

    base = datetime(2026, 1, 1, 8, 0, tzinfo=pytz.UTC)
    day0 = base
    day1 = base + timedelta(days=1)
    day2 = base + timedelta(days=2)

    person.add_to_mileage_history(day0, 10.0, day0 + timedelta(hours=9))
    person.add_to_mileage_history(day2, 12.0, day2 + timedelta(hours=9))
    person.add_to_mileage_history(day1, 11.0, day1 + timedelta(hours=9))

    assert [entry[0].date() for entry in person.historical_mileage_data] == [
        day0.date(),
        day1.date(),
        day2.date(),
    ]

    person.add_to_mileage_history(day1, 15.0, day1 + timedelta(hours=10))
    assert len(person.historical_mileage_data) == 3
    assert person.historical_mileage_data[1][1] == 15.0

    for idx in range(MAX_PERSON_MILEAGE_HISTORICAL_DATA_DAYS + 1):
        day = base + timedelta(days=3 + idx)
        person.add_to_mileage_history(day, 1.0, day + timedelta(hours=8))

    assert len(person.historical_mileage_data) == MAX_PERSON_MILEAGE_HISTORICAL_DATA_DAYS
    assert len(person.serializable_historical_data) == MAX_PERSON_MILEAGE_HISTORICAL_DATA_DAYS


def test_should_recompute_history_empty_authorized_cars(hass: HomeAssistant) -> None:
    """Test recompute history when no cars are authorized."""
    _, _, person = _make_person(hass, **{CONF_PERSON_AUTHORIZED_CARS: []})
    now = datetime.now(tz=pytz.UTC)
    assert person.should_recompute_history(now) is False


def test_add_to_mileage_history_missing_leave_time(hass: HomeAssistant) -> None:
    """Test that missing leave time is ignored."""
    _, _, person = _make_person(hass)
    day = datetime(2026, 1, 1, 8, 0, tzinfo=pytz.UTC)
    person.add_to_mileage_history(day, 12.0, None)
    assert person.historical_mileage_data == []


def test_get_best_week_day_guess_uses_last_two_entries(hass: HomeAssistant) -> None:
    """Test weekday best-guess selection from last two entries."""
    _, _, person = _make_person(hass)

    base = datetime(2026, 1, 1, 8, 0, tzinfo=pytz.UTC)
    day_old = base
    day_mid = base + timedelta(days=7)
    day_new = base + timedelta(days=14)

    person.add_to_mileage_history(day_old, 120.0, day_old + timedelta(hours=9))
    person.add_to_mileage_history(day_mid, 20.0, day_mid + timedelta(hours=9))
    person.add_to_mileage_history(day_new, 60.0, day_new + timedelta(hours=8))

    week_day = day_new.replace(tzinfo=pytz.UTC).astimezone(tz=None).weekday()
    mileage, leave_time = person._get_best_week_day_guess(week_day)
    expected_leave_time = (day_new + timedelta(hours=8)).replace(tzinfo=pytz.UTC).astimezone(tz=None).time()

    assert mileage == 60.0
    assert leave_time == expected_leave_time


def test_compute_person_next_need_today_vs_tomorrow(hass: HomeAssistant) -> None:
    """Test forecast selection for today vs tomorrow."""
    _, _, person = _make_person(hass)

    now = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    today_leave = now + timedelta(hours=10)
    tomorrow_leave = now + timedelta(days=1, hours=1)

    person.add_to_mileage_history(now, 30.0, today_leave)
    person.add_to_mileage_history(now + timedelta(days=1), 50.0, tomorrow_leave)

    predicted_leave, predicted_mileage = person._compute_person_next_need(now)
    assert predicted_leave is not None
    assert predicted_mileage == 30.0

    now_after_today = now + timedelta(hours=12)
    predicted_leave, predicted_mileage = person._compute_person_next_need(now_after_today)
    assert predicted_leave is not None
    assert predicted_mileage == 50.0


def test_compute_person_next_need_tomorrow_only(hass: HomeAssistant) -> None:
    """Test forecast when only tomorrow data exists."""
    _, _, person = _make_person(hass)
    now = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    tomorrow = now + timedelta(days=1)
    person.add_to_mileage_history(tomorrow, 75.0, tomorrow + timedelta(hours=8))

    predicted_leave, predicted_mileage = person._compute_person_next_need(now)
    assert predicted_leave is not None
    assert predicted_mileage == 75.0


def test_compute_person_next_need_empty_prediction(hass: HomeAssistant) -> None:
    """Test no prediction when weekday data is missing."""
    _, _, person = _make_person(hass)

    now = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    other_day = now + timedelta(days=3)
    person.add_to_mileage_history(other_day, 42.0, other_day + timedelta(hours=9))

    predicted_leave, predicted_mileage = person._compute_person_next_need(now)
    assert predicted_leave is None
    assert predicted_mileage is None


@pytest.mark.asyncio
async def test_device_post_home_init_restores_history(hass: HomeAssistant) -> None:
    """Test state restoration from stored historical data."""
    hass, _, person = _make_person(hass)
    person.ha_entities["person_mileage_prediction"] = SimpleNamespace(
        entity_id="sensor.person_mileage_prediction"
    )

    base = datetime(2026, 1, 10, 8, 0, tzinfo=pytz.UTC)
    entries = [
        {
            "day": base.isoformat(),
            "mileage": 20.0,
            "leave_time": (base + timedelta(hours=9)).isoformat(),
        }
    ]
    hass.states.async_set(
        "sensor.person_mileage_prediction",
        "ok",
        {"historical_data": entries, "has_been_initialized": True},
    )

    person.device_post_home_init(base)

    assert len(person.historical_mileage_data) == 1
    assert person.has_been_initialized is True


def test_device_post_home_init_missing_sensor_entity(hass: HomeAssistant) -> None:
    """Test state restore when no HA entity is present."""
    _, _, person = _make_person(hass)
    person.device_post_home_init(datetime(2026, 1, 2, tzinfo=pytz.UTC))
    assert person.historical_mileage_data == []


@pytest.mark.asyncio
async def test_device_post_home_init_handles_malformed_entries(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test malformed historical entries are skipped."""
    hass, _, person = _make_person(hass)
    person.ha_entities["person_mileage_prediction"] = SimpleNamespace(
        entity_id="sensor.person_mileage_prediction"
    )
    now = datetime(2026, 1, 10, 8, 0, tzinfo=pytz.UTC)

    entries = [
        {"day": None, "mileage": 10.0, "leave_time": now.isoformat()},
        {"day": "bad-date", "mileage": 10.0, "leave_time": "bad-time"},
        {
            "day": now.isoformat(),
            "mileage": 15.0,
            "leave_time": (now + timedelta(hours=1)).isoformat(),
        },
    ]

    def _add_to_history(day, mileage, _leave_time):
        person.historical_mileage_data.append((day, mileage, None, day.weekday()))

    monkeypatch.setattr(person, "add_to_mileage_history", _add_to_history)
    monkeypatch.setattr(person, "update_person_forecast", lambda *_, **__: None)
    hass.states.async_set(
        "sensor.person_mileage_prediction",
        "ok",
        {"historical_data": entries, "has_been_initialized": False},
    )

    person.device_post_home_init(now)
    assert len(person.historical_mileage_data) == 1
    assert person.historical_mileage_data[0][2] is None


@pytest.mark.asyncio
async def test_device_post_home_init_warns_when_not_initialized(hass: HomeAssistant) -> None:
    """Test no-history state keeps initialization false."""
    hass, _, person = _make_person(hass)
    person.ha_entities["person_mileage_prediction"] = SimpleNamespace(
        entity_id="sensor.person_mileage_prediction"
    )
    hass.states.async_set(
        "sensor.person_mileage_prediction",
        "ok",
        {"historical_data": [], "has_been_initialized": False},
    )
    person.device_post_home_init(datetime(2026, 1, 2, tzinfo=pytz.UTC))
    assert person.has_been_initialized is False


@pytest.mark.asyncio
async def test_device_post_home_init_handles_unknown_state(hass: HomeAssistant) -> None:
    """Test state restore when sensor is unknown."""
    hass, _, person = _make_person(hass)
    person.ha_entities["person_mileage_prediction"] = SimpleNamespace(
        entity_id="sensor.person_mileage_prediction"
    )

    person.historical_mileage_data = [
        (datetime(2026, 1, 1, tzinfo=pytz.UTC), 10.0, datetime(2026, 1, 1, 9, tzinfo=pytz.UTC), 3)
    ]
    hass.states.async_set("sensor.person_mileage_prediction", "unknown", {})

    person.device_post_home_init(datetime(2026, 1, 2, tzinfo=pytz.UTC))

    assert person.historical_mileage_data == []
    assert person.predicted_leave_time is None
    assert person.predicted_mileage is None


def test_get_forecast_readable_string_no_forecast(hass: HomeAssistant) -> None:
    """Test readable string when there is no forecast."""
    _, _, person = _make_person(hass)
    assert person.get_forecast_readable_string() == "No forecast"


def test_get_authorized_and_preferred_cars(hass: HomeAssistant) -> None:
    """Test retrieving authorized and preferred cars."""
    hass, home, person = _make_person(hass,
        **{
            CONF_PERSON_AUTHORIZED_CARS: ["Car A", "Missing"],
            CONF_PERSON_PREFERRED_CAR: "Car A",
        }
    )
    car = _make_car(hass, home, name="Car A")
    authorized = person.get_authorized_cars()
    assert authorized == [car]
    assert person.get_preferred_car() == car

    person.preferred_car = "Missing"
    assert person.get_preferred_car() is None

    person.preferred_car = None
    assert person.get_preferred_car() is None


@pytest.mark.asyncio
async def test_notify_no_allocated_car(hass: HomeAssistant) -> None:
    """Test notification when no car is allocated."""
    _, home, person = _make_person(hass, **{CONF_MOBILE_APP: "notify"})
    person.predicted_mileage = 40.0
    person.predicted_leave_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    notify_time = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    person._last_request_prediction_time = notify_time

    notifications: list[tuple[str, str]] = []

    async def _capture(*_, title: str | None = None, message: str | None = None, **__):
        notifications.append((title, message))

    person.on_device_state_change = _capture

    await person.notify_of_forecast_if_needed(
        time=notify_time,
        notify_reason=PERSON_NOTIFY_REASON_CHANGED_CAR,
    )

    assert notifications
    title, message = notifications[0]
    assert title == "No allocated car !"
    assert "40km" in message


@pytest.mark.asyncio
async def test_notify_daily_reminder_no_charger_triggers_allocation(hass: HomeAssistant) -> None:
    """Test daily reminder triggers allocation when no charger."""
    hass, home, person = _make_person(hass,
        **{
            CONF_MOBILE_APP: "notify",
            CONF_PERSON_NOTIFICATION_TIME: "08:00:00",
        }
    )
    person._last_forecast_notification_call_time = datetime(2026, 1, 14, 8, 0, tzinfo=pytz.UTC)

    allocation_called = {"value": False}

    async def _alloc(*_, **__):
        allocation_called["value"] = True
        return {}

    home.get_best_persons_cars_allocations = _alloc

    await person.notify_of_forecast_if_needed(
        time=datetime(2026, 1, 15, 8, 5, tzinfo=pytz.UTC),
        notify_reason=PERSON_NOTIFY_REASON_DAILY_REMINDER_FOR_CAR_NO_CHARGER,
    )

    assert allocation_called["value"] is True


@pytest.mark.asyncio
async def test_notify_returns_without_mobile_app(hass: HomeAssistant) -> None:
    """Test notify returns when no mobile app is configured."""
    _, _, person = _make_person(hass)
    await person.notify_of_forecast_if_needed()


@pytest.mark.asyncio
async def test_notify_uses_default_time(hass: HomeAssistant) -> None:
    """Test notify uses default time when not provided."""
    _, _, person = _make_person(hass, **{CONF_MOBILE_APP: "notify"})
    notifications: list[tuple[str | None, str | None]] = []

    async def _capture(*_, title: str | None = None, message: str | None = None, **__):
        notifications.append((title, message))

    person.on_device_state_change = _capture
    await person.notify_of_forecast_if_needed(
        time=None,
        notify_reason=PERSON_NOTIFY_REASON_CHANGED_CAR,
    )

    assert notifications
    assert person._last_forecast_notification_call_time is not None


@pytest.mark.asyncio
async def test_notify_car_ready(hass: HomeAssistant) -> None:
    """Test notification when car already covers the trip."""
    hass, home, person = _make_person(hass, **{CONF_MOBILE_APP: "notify"})
    car = _make_car(hass, home)
    car.current_forecasted_person = person
    car.charger = object()
    car.get_adapt_target_percent_soc_to_reach_range_km = lambda *_: (True, 70.0, 60.0, None)

    person.predicted_mileage = 60.0
    person.predicted_leave_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    notify_time = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    person._last_request_prediction_time = notify_time

    notifications: list[tuple[str, str]] = []

    async def _capture(*_, title: str | None = None, message: str | None = None, **__):
        notifications.append((title, message))

    person.on_device_state_change = _capture

    await person.notify_of_forecast_if_needed(
        time=notify_time,
        notify_reason=PERSON_NOTIFY_REASON_CHANGED_CAR,
    )

    title, message = notifications[0]
    assert title == "Test Car: OK, it is already ready!"
    assert "current charge is 70%" in message


@pytest.mark.asyncio
async def test_notify_constraint_target_not_enough(hass: HomeAssistant) -> None:
    """Test notification when scheduled charge will not cover trip."""
    hass, home, person = _make_person(hass, **{CONF_MOBILE_APP: "notify"})
    car = _make_car(hass, home)
    car.current_forecasted_person = person
    car.charger = object()
    car.get_adapt_target_percent_soc_to_reach_range_km = lambda *_: (False, 40.0, 80.0, None)

    person.predicted_mileage = 80.0
    person.predicted_leave_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    notify_time = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    person._last_request_prediction_time = notify_time

    constraint = object.__new__(MultiStepsPowerLoadConstraintChargePercent)
    constraint.target_value = 50.0
    constraint.end_of_constraint = person.predicted_leave_time + timedelta(hours=1)

    notifications: list[tuple[str, str]] = []

    async def _capture(*_, title: str | None = None, message: str | None = None, **__):
        notifications.append((title, message))

    person.on_device_state_change = _capture

    await person.notify_of_forecast_if_needed(
        time=notify_time,
        notify_reason=PERSON_NOTIFY_REASON_CHANGED_CAR,
        force_ct=constraint,
    )

    title, message = notifications[0]
    assert "WON'T cover your trip" in title
    assert "scheduled charge" in message
    assert car._user_selected_person_name_for_car == person.name


@pytest.mark.asyncio
async def test_notify_scheduled_charge_ok(hass: HomeAssistant) -> None:
    """Test notification when scheduled charge covers the trip."""
    hass, home, person = _make_person(hass, **{CONF_MOBILE_APP: "notify"})
    car = _make_car(hass, home)
    car.current_forecasted_person = person
    car.charger = object()
    car.get_adapt_target_percent_soc_to_reach_range_km = lambda *_: (False, 40.0, 60.0, None)

    person.predicted_mileage = 60.0
    person.predicted_leave_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    notify_time = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    person._last_request_prediction_time = notify_time

    constraint = object.__new__(MultiStepsPowerLoadConstraintChargePercent)
    constraint.target_value = 80.0
    constraint.end_of_constraint = person.predicted_leave_time - timedelta(minutes=30)

    notifications: list[tuple[str, str]] = []

    async def _capture(*_, title: str | None = None, message: str | None = None, **__):
        notifications.append((title, message))

    person.on_device_state_change = _capture

    await person.notify_of_forecast_if_needed(
        time=notify_time,
        notify_reason=PERSON_NOTIFY_REASON_CHANGED_CAR,
        force_ct=constraint,
    )

    title, message = notifications[0]
    assert "scheduled charge that works" in title
    assert "cover your trip" in message


@pytest.mark.asyncio
async def test_notify_person_constraint_charging(hass: HomeAssistant) -> None:
    """Test notification when person constraint is adequate."""
    hass, home, person = _make_person(hass, **{CONF_MOBILE_APP: "notify"})
    car = _make_car(hass, home)
    car.current_forecasted_person = person
    car.charger = object()
    car.get_adapt_target_percent_soc_to_reach_range_km = lambda *_: (False, 20.0, 50.0, None)

    person.predicted_mileage = 50.0
    person.predicted_leave_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    notify_time = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    person._last_request_prediction_time = notify_time

    person_constraint = LoadConstraint(
        time=datetime(2026, 1, 15, 7, 0, tzinfo=pytz.UTC),
        end_of_constraint=person.predicted_leave_time - timedelta(minutes=30),
        target_value=60.0,
    )

    notifications: list[tuple[str, str]] = []

    async def _capture(*_, title: str | None = None, message: str | None = None, **__):
        notifications.append((title, message))

    person.on_device_state_change = _capture

    await person.notify_of_forecast_if_needed(
        time=notify_time,
        notify_reason=PERSON_NOTIFY_REASON_CHANGED_CAR,
        person_ct=person_constraint,
    )

    title, message = notifications[0]
    assert "I'll charge it for you" in title
    assert "charge it to 50%" in message.lower()


@pytest.mark.asyncio
async def test_notify_person_constraint_insufficient(hass: HomeAssistant) -> None:
    """Test notification when person constraint is insufficient."""
    hass, home, person = _make_person(hass, **{CONF_MOBILE_APP: "notify"})
    car = _make_car(hass, home)
    car.current_forecasted_person = person
    car.charger = object()
    car.get_adapt_target_percent_soc_to_reach_range_km = lambda *_: (False, 30.0, 80.0, None)

    person.predicted_mileage = 80.0
    person.predicted_leave_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    notify_time = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    person._last_request_prediction_time = notify_time

    person_constraint = LoadConstraint(
        time=datetime(2026, 1, 15, 7, 0, tzinfo=pytz.UTC),
        end_of_constraint=person.predicted_leave_time + timedelta(minutes=30),
        target_value=60.0,
    )

    notifications: list[tuple[str, str]] = []

    async def _capture(*_, title: str | None = None, message: str | None = None, **__):
        notifications.append((title, message))

    person.on_device_state_change = _capture

    await person.notify_of_forecast_if_needed(
        time=notify_time,
        notify_reason=PERSON_NOTIFY_REASON_CHANGED_CAR,
        person_ct=person_constraint,
    )

    title, message = notifications[0]
    assert "check what i've done" in title.lower()
    assert "charge it to 80%" in message.lower()


@pytest.mark.asyncio
async def test_notify_user_constraint_unknown_soc(hass: HomeAssistant) -> None:
    """Test user constraint selection with unknown SOC."""
    hass, home, person = _make_person(hass, **{CONF_MOBILE_APP: "notify"})
    car = _make_car(hass, home)
    car.current_forecasted_person = person
    car.charger = object()
    car.get_adapt_target_percent_soc_to_reach_range_km = lambda *_: (False, None, None, None)

    person.predicted_mileage = 70.0
    person.predicted_leave_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    notify_time = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    person._last_request_prediction_time = notify_time

    user_constraint = LoadConstraint(
        time=datetime(2026, 1, 15, 7, 0, tzinfo=pytz.UTC),
        end_of_constraint=person.predicted_leave_time - timedelta(minutes=15),
        target_value=60.0,
    )

    notifications: list[tuple[str, str]] = []

    async def _capture(*_, title: str | None = None, message: str | None = None, **__):
        notifications.append((title, message))

    person.on_device_state_change = _capture

    await person.notify_of_forecast_if_needed(
        time=notify_time,
        notify_reason=PERSON_NOTIFY_REASON_CHANGED_CAR,
        user_ct=user_constraint,
    )

    title, message = notifications[0]
    assert "trip is not covered" in title
    assert "UNKNOWN" in message
    assert car._user_selected_person_name_for_car == person.name


def test_get_person_mileage_serialized_prediction(hass: HomeAssistant) -> None:
    """Test serialization of forecast state."""
    _, _, person = _make_person(hass)
    now = datetime.now(tz=pytz.UTC)
    person.predicted_mileage = 25.0
    person.predicted_leave_time = now + timedelta(hours=2)
    person._last_request_prediction_time = now
    person.serializable_historical_data = [
        {
            "day": now.isoformat(),
            "mileage": 25.0,
            "leave_time": (now + timedelta(hours=2)).isoformat(),
        }
    ]

    state, data = person.get_person_mileage_serialized_prediction()

    assert isinstance(state, str)
    assert data["historical_data"]
    assert data["predicted_mileage"] == person.predicted_mileage
    assert data["predicted_leave_time"] is not None


def test_get_platforms_parent_none(hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test platforms when parent returns None."""
    _, _, person = _make_person(hass)

    monkeypatch.setattr(HADeviceMixin, "get_platforms", lambda self: None)
    platforms = person.get_platforms()
    assert Platform.SENSOR in platforms


def test_person_reset(hass: HomeAssistant) -> None:
    """Test reset passes through."""
    _, _, person = _make_person(hass)
    person.reset()
