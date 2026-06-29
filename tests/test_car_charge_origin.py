"""QS-274 — coverage for ``QSCar.get_car_charge_origin_readable_string()``.

The helper produces the origin-responsive context line rendered in the
car card's ``.forecast-row``. It is pure / CPU-only: it reads the active
constraint origin via ``charger.get_charge_type(return_charge_errors=False)``
and the car's live ``current_forecasted_person`` state, and never surfaces
a charger-error string.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
import pytz
from freezegun import freeze_time
from homeassistant.const import CONF_NAME

from custom_components.quiet_solar.const import (
    CAR_CHARGE_TYPE_AS_FAST_AS_POSSIBLE,
    CAR_CHARGE_TYPE_CALENDAR,
    CAR_CHARGE_TYPE_MANUAL,
    CAR_CHARGE_TYPE_MANUAL_AS_FAST_AS_POSSIBLE,
    CAR_CHARGE_TYPE_PERSON_AUTOMATED,
    CAR_CHARGE_TYPE_SOLAR_AFTER_BATTERY,
    CONF_PERSON_AUTHORIZED_CARS,
    CONF_PERSON_PERSON_ENTITY,
)
from custom_components.quiet_solar.ha_model.car import QSCar
from custom_components.quiet_solar.ha_model.person import QSPerson
from custom_components.quiet_solar.home_model.constraints import get_readable_date_string
from tests.test_load_model import create_real_constraint


def _make_car(fake_hass, mock_data_handler, **overrides) -> QSCar:
    from tests.ha_tests.const import MOCK_CAR_CONFIG

    config = {
        **MOCK_CAR_CONFIG,
        "hass": fake_hass,
        "config_entry": MagicMock(),
        "data_handler": mock_data_handler,
        "home": mock_data_handler.home,
    }
    config.update(overrides)
    return QSCar(**config)


@pytest.fixture
def origin_car(fake_hass, mock_data_handler, current_time):
    return _make_car(fake_hass, mock_data_handler)


def _set_charge_type(car: QSCar, charge_type: str, ct=None) -> None:
    car.charger = MagicMock()
    car.charger.get_charge_type = MagicMock(return_value=(charge_type, ct))


def _make_person(fake_hass, mock_data_handler, *, mileage, leave_time) -> QSPerson:
    """A real QSPerson with a deterministic forecast (no mocked forecast string)."""
    person = QSPerson(
        hass=fake_hass,
        config_entry=MagicMock(),
        home=mock_data_handler.home,
        **{
            CONF_NAME: "Magali",
            CONF_PERSON_PERSON_ENTITY: "person.magali",
            CONF_PERSON_AUTHORIZED_CARS: ["Test Car"],
        },
    )
    person.predicted_mileage = mileage
    person.predicted_leave_time = leave_time
    # keep the prediction stable — get_forecast_readable_string() runs for real
    person.update_person_forecast = MagicMock(return_value=(leave_time, mileage))
    return person


def _near_term_constraint():
    """A real constraint with a near-term target so the real formatter yields the
    normal ``today HH:MM`` form (QS-278: origin line uses normal formatting). The
    expected value is derived from the formatter — not a literal — so the
    today/tomorrow boundary at midnight cannot cause a flaky assertion."""
    end_time = datetime.now(pytz.UTC) + timedelta(hours=2)
    ct = create_real_constraint(load=None, end_time=end_time)
    expected = get_readable_date_string(end_time, for_small_standalone=False)
    return ct, expected


def _far_term_constraint():
    """A real constraint >24h out. QS-278: the origin line renders the full
    ``%Y-%m-%d %H:%M`` date on a single line. Returns the expected full-date
    substring derived from the normal formatter."""
    end_time = datetime.now(pytz.UTC) + timedelta(days=3)
    ct = create_real_constraint(load=None, end_time=end_time)
    expected = get_readable_date_string(end_time, for_small_standalone=False)
    return ct, expected


# ── Person (real method, no mock — review-fix #01 finding 4) ─────────────────


def test_person_origin(origin_car, fake_hass, mock_data_handler):
    person = _make_person(
        fake_hass,
        mock_data_handler,
        mileage=30.0,
        leave_time=datetime.now(pytz.UTC) + timedelta(hours=3),
    )
    origin_car.current_forecasted_person = person
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_PERSON_AUTOMATED)

    # review-fix #03: all person branches delegate to the single helper
    # get_car_person_readable_forecast_mileage() → "<name>: <forecast>"
    result = origin_car.get_car_charge_origin_readable_string()
    assert result.startswith("Magali: ")
    assert "30km" in result


@freeze_time("2026-01-15 12:00:00")
def test_person_origin_far_out_leave_time_is_single_line(origin_car, fake_hass, mock_data_handler):
    """AC3 (review-fix #01-3): the person fall-through with a >24h leave time
    renders the full ``%Y-%m-%d %H:%M`` date on a single line — directly locking
    the "leave times" clause, not just the Calendar/Manual target path. Frozen
    clock makes the full-date branch deterministic."""
    leave_time = datetime.now(pytz.UTC) + timedelta(days=3)
    person = _make_person(fake_hass, mock_data_handler, mileage=30.0, leave_time=leave_time)
    origin_car.current_forecasted_person = person
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_PERSON_AUTOMATED)

    result = origin_car.get_car_charge_origin_readable_string()
    expected = get_readable_date_string(leave_time, for_small_standalone=False)
    assert "\n" not in result
    assert result == f"Magali: 30km {expected}"
    assert expected[:4].isdigit()  # full-date form "%Y-%m-%d %H:%M", not compact


def test_orphaned_person_origin(origin_car):
    """Person-tagged constraint outlives a now-None current_forecasted_person."""
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_PERSON_AUTOMATED)
    origin_car.current_forecasted_person = None
    assert origin_car.get_car_charge_origin_readable_string() == "No forecasted person"


def test_empty_forecast_person_origin(origin_car, fake_hass, mock_data_handler):
    """A person attached with an empty forecast surfaces the person line as-is —
    "<name>: No forecast" (review-fix #03 — accepted, unified behavior)."""
    person = _make_person(fake_hass, mock_data_handler, mileage=None, leave_time=None)
    origin_car.current_forecasted_person = person
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_PERSON_AUTOMATED)

    assert origin_car.get_car_charge_origin_readable_string() == "Magali: No forecast"


# ── Calendar / Manual (real formatter — review-fix #01 finding 6) ────────────


def test_calendar_origin(origin_car):
    ct, expected = _near_term_constraint()
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_CALENDAR, ct)
    assert origin_car.get_car_charge_origin_readable_string() == f"Calendar · {expected}"


def test_manual_origin(origin_car):
    ct, expected = _near_term_constraint()
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_MANUAL, ct)
    assert (
        origin_car.get_car_charge_origin_readable_string() == f"Manually set to {expected}"
    )


@freeze_time("2026-01-15 12:00:00")
def test_calendar_origin_far_out_target_is_single_line(origin_car):
    """A >24h target renders the full ``%Y-%m-%d %H:%M`` date on one line
    (QS-278: normal formatting, no raw newline). Frozen clock (review-fix
    #01-4) keeps the full-date branch deterministic regardless of run timing."""
    ct, expected = _far_term_constraint()
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_CALENDAR, ct)
    result = origin_car.get_car_charge_origin_readable_string()
    assert "\n" not in result
    assert result == f"Calendar · {expected}"
    assert expected[:4].isdigit()  # full-date form "%Y-%m-%d %H:%M"


@freeze_time("2026-01-15 12:00:00")
def test_manual_origin_far_out_target_is_single_line(origin_car):
    """A >24h manual target renders the full ``%Y-%m-%d %H:%M`` date on one line."""
    ct, expected = _far_term_constraint()
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_MANUAL, ct)
    result = origin_car.get_car_charge_origin_readable_string()
    assert "\n" not in result
    assert result == f"Manually set to {expected}"
    assert expected[:4].isdigit()  # full-date form "%Y-%m-%d %H:%M"


# ── QS-278 wording locks (normal today/tomorrow formatting) ──────────────────


@freeze_time("2026-01-15 12:00:00")
def test_calendar_origin_uses_tomorrow_wording(origin_car):
    """QS-278 AC1: a Calendar target one day out renders ``tomorrow HH:MM`` —
    the normal formatting, not the compact ``%m-%d %H:%M`` form."""
    target = datetime.now(pytz.UTC) + timedelta(days=1)
    ct = create_real_constraint(load=None, end_time=target)
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_CALENDAR, ct)
    result = origin_car.get_car_charge_origin_readable_string()
    expected = get_readable_date_string(target, for_small_standalone=False)
    assert result == f"Calendar · {expected}"
    assert result.startswith("Calendar · tomorrow ")


@freeze_time("2026-01-15 12:00:00")
def test_default_forecast_mileage_keeps_compact_form(origin_car, fake_hass, mock_data_handler):
    """QS-278 AC4: direct callers (default ``for_small_standalone=True``) keep the
    compact form — a one-day-out leave time stays ``%m-%d %H:%M``, never
    ``tomorrow``. Locks the True branch of both threaded methods."""
    leave_time = datetime.now(pytz.UTC) + timedelta(days=1)
    person = _make_person(fake_hass, mock_data_handler, mileage=30.0, leave_time=leave_time)
    origin_car.current_forecasted_person = person
    result = origin_car.get_car_person_readable_forecast_mileage()
    expected = get_readable_date_string(leave_time, for_small_standalone=True, allow_cr=False)
    assert result == f"Magali: 30km {expected}"
    assert "tomorrow" not in result


@freeze_time("2026-01-15 12:00:00")
def test_person_origin_uses_tomorrow_wording(origin_car, fake_hass, mock_data_handler):
    """QS-278 AC2: the person fall-through renders the leave time with the normal
    ``tomorrow HH:MM`` formatting (was the compact ``HH:MM`` form)."""
    leave_time = datetime.now(pytz.UTC) + timedelta(days=1)
    person = _make_person(fake_hass, mock_data_handler, mileage=30.0, leave_time=leave_time)
    origin_car.current_forecasted_person = person
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_PERSON_AUTOMATED)
    result = origin_car.get_car_charge_origin_readable_string()
    expected = get_readable_date_string(leave_time, for_small_standalone=False)
    assert result == f"Magali: 30km {expected}"
    assert result.startswith("Magali: 30km tomorrow ")


# ── ct-is-None fall-through (review-fix #01 finding 2) ───────────────────────


def test_calendar_origin_ct_none_falls_through(origin_car):
    """CALENDAR type with ct=None must not raise — falls through to fallback."""
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_CALENDAR, None)
    origin_car.current_forecasted_person = None
    assert origin_car.get_car_charge_origin_readable_string() == "No forecasted person"


def test_manual_origin_ct_none_falls_through(origin_car):
    """MANUAL type with ct=None must not raise — falls through to fallback."""
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_MANUAL, None)
    origin_car.current_forecasted_person = None
    assert origin_car.get_car_charge_origin_readable_string() == "No forecasted person"


# ── As-fast (automation vs user) ────────────────────────────────────────────


def test_as_fast_origin(origin_car):
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_AS_FAST_AS_POSSIBLE)
    assert (
        origin_car.get_car_charge_origin_readable_string()
        == "Automatically forced to charge as fast as possible"
    )


def test_manual_as_fast_origin(origin_car):
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_MANUAL_AS_FAST_AS_POSSIBLE)
    assert (
        origin_car.get_car_charge_origin_readable_string()
        == "Manual as fast as possible charge"
    )


# ── Other type fallback ────────────────────────────────────────────────────


def test_other_type_with_person_returns_forecast(origin_car, fake_hass, mock_data_handler):
    person = _make_person(
        fake_hass,
        mock_data_handler,
        mileage=30.0,
        leave_time=datetime.now(pytz.UTC) + timedelta(hours=3),
    )
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_SOLAR_AFTER_BATTERY)
    origin_car.current_forecasted_person = person
    result = origin_car.get_car_charge_origin_readable_string()
    assert result.startswith("Magali: ")
    assert "30km" in result


def test_other_type_without_person_returns_no_forecast(origin_car):
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_SOLAR_AFTER_BATTERY)
    origin_car.current_forecasted_person = None
    assert origin_car.get_car_charge_origin_readable_string() == "No forecasted person"


# ── No charger ─────────────────────────────────────────────────────────────


def test_no_charger_with_person_returns_forecast(origin_car, fake_hass, mock_data_handler):
    person = _make_person(
        fake_hass,
        mock_data_handler,
        mileage=30.0,
        leave_time=datetime.now(pytz.UTC) + timedelta(hours=3),
    )
    origin_car.charger = None
    origin_car.current_forecasted_person = person
    result = origin_car.get_car_charge_origin_readable_string()
    assert result.startswith("Magali: ")


def test_no_charger_without_person_returns_no_forecast(origin_car):
    origin_car.charger = None
    origin_car.current_forecasted_person = None
    assert origin_car.get_car_charge_origin_readable_string() == "No forecasted person"
