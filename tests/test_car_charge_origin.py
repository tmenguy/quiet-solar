"""QS-274 — coverage for ``QSCar.get_car_charge_origin_readable_string()``.

The helper produces the origin-responsive context line rendered in the
car card's ``.forecast-row``. It is pure / CPU-only: it reads the active
constraint origin via ``charger.get_charge_type(return_charge_errors=False)``
and the car's live ``current_forecasted_person`` state, and never surfaces
a charger-error string.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.quiet_solar.const import (
    CAR_CHARGE_TYPE_AS_FAST_AS_POSSIBLE,
    CAR_CHARGE_TYPE_CALENDAR,
    CAR_CHARGE_TYPE_MANUAL,
    CAR_CHARGE_TYPE_MANUAL_AS_FAST_AS_POSSIBLE,
    CAR_CHARGE_TYPE_PERSON_AUTOMATED,
    CAR_CHARGE_TYPE_SOLAR_AFTER_BATTERY,
)
from custom_components.quiet_solar.ha_model.car import QSCar


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


def _make_person(name: str, forecast: str) -> MagicMock:
    person = MagicMock()
    person.name = name
    person.get_forecast_readable_string = MagicMock(return_value=forecast)
    return person


def _make_target_constraint(target: str) -> MagicMock:
    ct = MagicMock()
    ct.get_readable_next_target_date_string = MagicMock(return_value=target)
    return ct


# ── Person ───────────────────────────────────────────────────────────────


def test_person_origin(origin_car):
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_PERSON_AUTOMATED)
    origin_car.current_forecasted_person = _make_person("Magali", "30km 14:30")
    assert (
        origin_car.get_car_charge_origin_readable_string()
        == "Forecasted from Magali: 30km 14:30"
    )


def test_orphaned_person_origin(origin_car):
    """Person-tagged constraint outlives a now-None current_forecasted_person."""
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_PERSON_AUTOMATED)
    origin_car.current_forecasted_person = None
    assert origin_car.get_car_charge_origin_readable_string() == "No proper Forecast"


# ── Calendar / Manual ──────────────────────────────────────────────────────


def test_calendar_origin(origin_car):
    ct = _make_target_constraint("14:30")
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_CALENDAR, ct)
    assert origin_car.get_car_charge_origin_readable_string() == "Calendar · 14:30"


def test_manual_origin(origin_car):
    ct = _make_target_constraint("14:30")
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_MANUAL, ct)
    assert (
        origin_car.get_car_charge_origin_readable_string() == "Manually set to 14:30"
    )


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


def test_other_type_with_person_returns_forecast(origin_car):
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_SOLAR_AFTER_BATTERY)
    origin_car.current_forecasted_person = _make_person("Magali", "30km 14:30")
    assert origin_car.get_car_charge_origin_readable_string() == "Magali: 30km 14:30"


def test_other_type_without_person_returns_no_forecast(origin_car):
    _set_charge_type(origin_car, CAR_CHARGE_TYPE_SOLAR_AFTER_BATTERY)
    origin_car.current_forecasted_person = None
    assert origin_car.get_car_charge_origin_readable_string() == "No proper Forecast"


# ── No charger ─────────────────────────────────────────────────────────────


def test_no_charger_with_person_returns_forecast(origin_car):
    origin_car.charger = None
    origin_car.current_forecasted_person = _make_person("Magali", "30km 14:30")
    assert origin_car.get_car_charge_origin_readable_string() == "Magali: 30km 14:30"


def test_no_charger_without_person_returns_no_forecast(origin_car):
    origin_car.charger = None
    origin_car.current_forecasted_person = None
    assert origin_car.get_car_charge_origin_readable_string() == "No proper Forecast"
