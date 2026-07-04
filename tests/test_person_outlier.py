"""Outlier-resistant trip-prediction tests (QS-298).

These exercise ``QSPerson._is_mileage_outlier`` and the reworked
``QSPerson._get_best_week_day_guess`` directly by building
``historical_mileage_data`` tuples with full control over the weekday
buckets. Tuples are ``(day, mileage_km, leave_time, weekday)`` in local
time, date-ordered ascending — the same schema the ingestion path
produces.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytz
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    CONF_PERSON_PERSON_ENTITY,
    DOMAIN,
    MAX_PERSON_MILEAGE_HISTORICAL_DATA_DAYS,
    PERSON_MILEAGE_RECURRENCE_WINDOW_DAYS,
)
from custom_components.quiet_solar.ha_model.home import QSHome
from custom_components.quiet_solar.ha_model.person import QSPerson

# Reference weekdays (2026-01-01 is a Thursday).
WED = datetime(2026, 1, 7, 12, 0, tzinfo=pytz.UTC)  # Wednesday
TUE = datetime(2026, 1, 6, 12, 0, tzinfo=pytz.UTC)  # Tuesday
FRI = datetime(2026, 1, 9, 12, 0, tzinfo=pytz.UTC)  # Friday
SUN = datetime(2026, 1, 11, 12, 0, tzinfo=pytz.UTC)  # Sunday


def _make_person(hass: HomeAssistant, **overrides) -> QSPerson:
    hass.data.setdefault(DOMAIN, {})
    config_entry = MockConfigEntry(domain=DOMAIN, data={"name": "Test Home"}, title="Test Home")
    home = QSHome(hass=hass, config_entry=config_entry, name="Test Home")
    return QSPerson(
        hass=hass,
        home=home,
        config_entry=None,
        name="Test Person",
        **{CONF_PERSON_PERSON_ENTITY: "person.test_person", **overrides},
    )


def _entry(base: datetime, weeks: int, km: float, leave_hour: int = 8):
    """Build one same-weekday record ``weeks`` after ``base``."""
    day = base + timedelta(weeks=weeks)
    leave = day.replace(hour=leave_hour, minute=0)
    return (day, float(km), leave, day.weekday())


def _set_history(person: QSPerson, entries: list) -> None:
    person.historical_mileage_data = sorted(entries, key=lambda e: e[0])


def test_one_off_long_trip_rejected(hass: HomeAssistant) -> None:
    """AC 1: a single 400 km record among ~12 weeks of 30-60 km days is
    rejected; prediction is the max of the last 2 non-outlier records."""
    person = _make_person(hass)
    entries = [_entry(WED, w, 30 + (w % 2) * 30) for w in range(11)]  # 30 / 60 km
    entries.append(_entry(WED, 11, 400.0))  # one-off last week
    _set_history(person, entries)

    mileage, leave = person._get_best_week_day_guess(WED.weekday())
    assert mileage is not None and mileage <= 60.0
    assert leave is not None


def test_live_recurring_weekly_long_trip_kept(hass: HomeAssistant) -> None:
    """AC 2 (minority regime): last 2 same-weekday records both ~400 km on
    consecutive weeks, within tolerance, with a normal-day majority → kept."""
    person = _make_person(hass)
    entries = [_entry(WED, w, 40.0) for w in range(6)]
    entries.append(_entry(WED, 6, 400.0))
    entries.append(_entry(WED, 7, 390.0))  # newest, within 20% of 400
    _set_history(person, entries)

    mileage, _leave = person._get_best_week_day_guess(WED.weekday())
    assert mileage is not None and mileage >= 390.0


def test_pattern_stop_reverts_prediction_immediately(hass: HomeAssistant) -> None:
    """AC 2: a normal day landing after a live pattern reverts at once."""
    person = _make_person(hass)
    entries = [_entry(WED, w, 40.0) for w in range(6)]
    entries.append(_entry(WED, 6, 400.0))
    entries.append(_entry(WED, 7, 390.0))
    entries.append(_entry(WED, 8, 40.0))  # normal day breaks liveness
    _set_history(person, entries)

    mileage, _leave = person._get_best_week_day_guess(WED.weekday())
    assert mileage is not None and mileage <= 60.0


def test_different_weekday_big_record_does_not_corroborate(hass: HomeAssistant) -> None:
    """AC 2: a big record on a different weekday never rescues an outlier."""
    person = _make_person(hass)
    entries = [_entry(WED, w, 40.0) for w in range(6)]
    entries.append(_entry(WED, 6, 400.0))  # one-off Wednesday
    # a big Tuesday record that must not corroborate the Wednesday one-off
    entries += [_entry(TUE, w, 40.0) for w in range(6)]
    entries.append(_entry(TUE, 6, 400.0))
    _set_history(person, entries)

    mileage, _leave = person._get_best_week_day_guess(WED.weekday())
    assert mileage is not None and mileage <= 60.0


def test_out_and_back_weekend_trip_rejected(hass: HomeAssistant) -> None:
    """AC 3: a one-off ~200 km Friday-out / Sunday-back trip is rejected on
    both weekdays (different buckets cannot corroborate)."""
    person = _make_person(hass)
    entries = []
    for base in (FRI, SUN):
        entries += [_entry(base, w, 40.0) for w in range(6)]
        entries.append(_entry(base, 6, 200.0))
    _set_history(person, entries)

    fri_mileage, _ = person._get_best_week_day_guess(FRI.weekday())
    sun_mileage, _ = person._get_best_week_day_guess(SUN.weekday())
    assert fri_mileage is not None and fri_mileage <= 60.0
    assert sun_mileage is not None and sun_mileage <= 60.0


def test_absolute_floor_keeps_normal_trip_for_low_mileage_person(hass: HomeAssistant) -> None:
    """AC 4: a 40 km record for a ~10 km-median person is not an outlier."""
    person = _make_person(hass)
    entries = [_entry(WED, w, 10.0) for w in range(6)]
    entries.append(_entry(WED, 6, 40.0))  # 4x median but under the 100 km floor
    _set_history(person, entries)

    mileage, _leave = person._get_best_week_day_guess(WED.weekday())
    assert mileage == 40.0


def test_sparse_weekday_bucket_disables_outlier_detection(hass: HomeAssistant) -> None:
    """AC 5: 3 leave-one-out samples → detection off (char.); exactly 4 → on."""
    person = _make_person(hass)

    # 4 records total → LOO others for the 400 record = 3 → detection off.
    sparse = [_entry(WED, w, 40.0) for w in range(3)] + [_entry(WED, 3, 400.0)]
    _set_history(person, sparse)
    mileage_off, leave_off = person._get_best_week_day_guess(WED.weekday())
    assert mileage_off == 400.0  # identical to today's max-of-last-2
    assert leave_off is not None

    # 5 records total → LOO others = 4 → detection on → 400 rejected.
    dense = [_entry(WED, w, 40.0) for w in range(4)] + [_entry(WED, 4, 400.0)]
    _set_history(person, dense)
    mileage_on, _leave = person._get_best_week_day_guess(WED.weekday())
    assert mileage_on == 40.0


def test_median_interpolated_on_even_bucket(hass: HomeAssistant) -> None:
    """AC 5: baseline uses statistics.median (interpolated) — on an even
    leave-one-out bucket [40,40,400,400] the baseline is 220, so a 500 km
    record is NOT suspicious (500 <= 2.5 * 220)."""
    person = _make_person(hass)
    others = [
        _entry(WED, 0, 40.0),
        _entry(WED, 1, 40.0),
        _entry(WED, 2, 400.0),
        _entry(WED, 3, 400.0),
    ]
    target = _entry(WED, 4, 500.0)
    _set_history(person, others + [target])

    # baseline = median([40,40,400,400]) = 220 ; 500 <= 2.5*220 = 550
    assert person._is_mileage_outlier(target) is False


def test_older_suspicious_record_never_rescued(hass: HomeAssistant) -> None:
    """Design §2 step 3 truth table: rescue only applies to the last 2 bucket
    records; an older suspicious record is always an outlier."""
    person = _make_person(hass)
    entries = [
        _entry(WED, 0, 40.0),
        _entry(WED, 1, 400.0),  # older big pair (consecutive + similar)
        _entry(WED, 2, 400.0),
        _entry(WED, 3, 40.0),
        _entry(WED, 4, 40.0),  # last 2 are normal
    ]
    _set_history(person, entries)

    older_big = entries[1]
    assert person._is_mileage_outlier(older_big) is True


def test_first_occurrence_rejected_pattern_live_after_second(hass: HomeAssistant) -> None:
    """AC 6: first occurrence of a new recurring trip rejected; second
    occurrence within 21 days makes the pattern live and kept."""
    person = _make_person(hass)
    normals = [_entry(WED, w, 40.0) for w in range(5)]

    _set_history(person, normals + [_entry(WED, 5, 400.0)])
    mileage_first, _ = person._get_best_week_day_guess(WED.weekday())
    assert mileage_first is not None and mileage_first <= 60.0

    _set_history(person, normals + [_entry(WED, 5, 400.0), _entry(WED, 6, 400.0)])
    mileage_second, _ = person._get_best_week_day_guess(WED.weekday())
    assert mileage_second == 400.0


def test_recurrence_tolerance_exceeded_rejects(hass: HomeAssistant) -> None:
    """AC 6: a 400/250 last-2 pair fails the similarity test → both rejected."""
    person = _make_person(hass)
    entries = [_entry(WED, w, 40.0) for w in range(5)]
    entries.append(_entry(WED, 5, 400.0))
    entries.append(_entry(WED, 6, 250.0))  # |400-250|=150 > 0.2*250=50
    _set_history(person, entries)

    mileage, _ = person._get_best_week_day_guess(WED.weekday())
    assert mileage is not None and mileage <= 60.0


def test_recurrence_window_exceeded_rejects(hass: HomeAssistant) -> None:
    """AC 6: window is inclusive at exactly 21 days; 28 days apart rejects."""
    assert PERSON_MILEAGE_RECURRENCE_WINDOW_DAYS == 21
    person = _make_person(hass)
    normals = [_entry(WED, w, 40.0) for w in range(5)]

    # exactly 21 days apart (3 weeks) → inclusive → live pattern kept
    _set_history(person, normals + [_entry(WED, 5, 400.0), _entry(WED, 8, 400.0)])
    mileage_in, _ = person._get_best_week_day_guess(WED.weekday())
    assert mileage_in == 400.0

    # 28 days apart (4 weeks) → outside window → rejected
    _set_history(person, normals + [_entry(WED, 5, 400.0), _entry(WED, 9, 400.0)])
    mileage_out, _ = person._get_best_week_day_guess(WED.weekday())
    assert mileage_out is not None and mileage_out <= 60.0


def test_leave_time_ignores_outlier_records(hass: HomeAssistant) -> None:
    """AC 1 (leave-time half): an outlier contributes neither mileage nor
    leave time — the predicted leave time is the earliest of the good ones."""
    person = _make_person(hass)
    entries = [_entry(WED, w, 40.0, leave_hour=8 + (w % 2)) for w in range(6)]  # 08:00 / 09:00
    entries.append(_entry(WED, 6, 400.0, leave_hour=5))  # anomalous early departure
    _set_history(person, entries)

    mileage, leave = person._get_best_week_day_guess(WED.weekday())
    assert mileage is not None and mileage <= 60.0
    assert leave is not None and leave.hour == 8  # never the 05:00 outlier


def test_serialized_history_90_records_under_recorder_limit(hass: HomeAssistant) -> None:
    """AC 8 (char., permanent guard): 90 realistic serialized records plus the
    sensor's other attributes stay under the recorder's 16 KB limit."""
    person = _make_person(hass)
    day = datetime(2026, 1, 1, 6, 30, 15, tzinfo=pytz.UTC)
    for i in range(MAX_PERSON_MILEAGE_HISTORICAL_DATA_DAYS):
        d = day + timedelta(days=i)
        person.add_to_mileage_history(d, 123.45, d + timedelta(hours=13, minutes=27))

    assert len(person.serializable_historical_data) == 90

    attributes = {
        "historical_data": person.serializable_historical_data,
        "predicted_mileage": 123.45,
        "predicted_leave_time": (day + timedelta(hours=13, minutes=27)).isoformat(),
        "has_been_initialized": True,
    }
    payload = json.dumps(attributes)
    assert len(payload) < 16 * 1024


def test_restored_old_records_survive_when_initialized(hass: HomeAssistant) -> None:
    """AC 7 (restart half, char.): a serialize→restore round-trip preserves
    records older than the backfill window and reproduces the prediction; an
    initialized person never triggers recompute."""
    person = _make_person(hass, person_authorized_cars=["Car A"])
    day = datetime(2026, 1, 1, 6, 0, tzinfo=pytz.UTC)
    for i in range(MAX_PERSON_MILEAGE_HISTORICAL_DATA_DAYS):
        d = day + timedelta(days=i)
        person.add_to_mileage_history(d, 40.0 + (i % 3) * 5.0, d + timedelta(hours=8 + (i % 2)))

    serialized = list(person.serializable_historical_data)
    assert len(serialized) == 90
    original = person._get_best_week_day_guess(WED.weekday())

    restored = _make_person(hass, person_authorized_cars=["Car A"])
    for e in serialized:
        rec_day = datetime.fromisoformat(e["day"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        rec_leave = datetime.fromisoformat(e["leave_time"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        restored.add_to_mileage_history(rec_day, float(e["mileage"]), rec_leave)

    assert len(restored.historical_mileage_data) == len(person.historical_mileage_data)
    assert restored._get_best_week_day_guess(WED.weekday()) == original

    restored.has_been_initialized = True
    assert restored.should_recompute_history(datetime.now(tz=pytz.UTC)) is False
