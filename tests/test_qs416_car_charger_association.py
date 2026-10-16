"""QS-416 — car↔charger association.

- D1: score tiers rebalanced (proximity ≤ 10 m, then plug sensor, then sync_ok,
  then fine distance, then fine sync). The ≤ 3 m bonus is a REPLACEMENT plug bonus.
- D2: the absolute 1-hour lock is replaced by a persisted per-attach floor.
- D3: a user action on the attached car pins the car↔charger association.

Built on `tests.utils.charger_harness` (real `QSChargerGeneric` / `QSCar`), on the
model of `tests/test_charger_allocation_tiebreak.py`.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from datetime import datetime, timedelta
from datetime import time as dt_time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz
from homeassistant.core import State

from custom_components.quiet_solar.const import (
    CHARGE_TIME_CONSTRAINTS_CLEARED,
    CHARGER_NO_CAR_CONNECTED,
    CONF_CHARGER_LATITUDE,
    CONF_CHARGER_LONGITUDE,
    FORCE_CAR_NO_CHARGER_CONNECTED,
    STORAGE_KEY_ATTACHED_CAR_BEST_SCORE,
    STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_CAR_NAME,
    STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_PLUG_STATE,
    STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_SAVED_AT,
    STORAGE_KEY_USER_ORIGINATED,
    STORAGE_KEY_USER_PIN_PLUG_STATE,
    STORAGE_KEY_USER_PIN_SAVED_AT,
    USER_ORIGINATED_CAR_NAME,
    USER_ORIGINATED_CHARGER_NAME,
)
from custom_components.quiet_solar.ha_model.charger import (
    CAR_CHARGER_PLUG_PROXY_RADIUS_M,
    CAR_CHARGER_PROXIMITY_RADIUS_M,
    CAR_SCORE_MAX,
)
from tests.utils.charger_harness import (
    create_charger,
    make_hass,
    make_home,
    make_real_car,
)

CHARGER_LOGGER = "custom_components.quiet_solar.ha_model.charger"

# 2026-10-11 15:14:50 local (UTC+2): the incident instant.
T0 = datetime(2026, 10, 11, 13, 14, 50, tzinfo=pytz.UTC)

W1_COORDS = (43.635300, 6.989300)
# `haversine` mean Earth radius: 1 degree of latitude in metres.
M_PER_DEG_LAT = 111195.08

# car plug sensor -> (windowed, instant) `is_car_plugged` answers
SENSOR = {
    "on": (True, True),
    "fallback": (False, True),
    "off": (False, False),
    "unknown": (None, None),
}

SYNC_228 = 20.0  # car plugged 20 s vs charger 400 s → |Δ| = 380 s → sync 228
NO_SYNC = 7200.0  # car plugged for 2 h: no sync with a 400 s charger

F_NIGHT = 1_510_870_229  # A "on" + sync 228 at 6.75 m, attached > 20 min (+1 incumbent)


def _at(base: tuple[float, float], metres: float) -> tuple[float, float]:
    return (base[0] + metres / M_PER_DEG_LAT, base[1])


def _set_car(car, sensor="on", dist=None, plug_duration=None, gps=True, home=True) -> None:
    windowed, instant = SENSOR[sensor]
    car.is_car_plugged = MagicMock(
        side_effect=lambda time=None, for_duration=None: windowed if for_duration else instant
    )
    if gps and dist is not None:
        car.get_car_coordinates = MagicMock(return_value=_at(W1_COORDS, dist))
    else:
        car.get_car_coordinates = MagicMock(return_value=(None, None))
    car.get_continuous_plug_duration = MagicMock(return_value=plug_duration)
    car.is_car_home = MagicMock(return_value=home)


def _make_charger(hass, home, name="wallbox_1", coords=W1_COORDS, plug_duration=400.0):
    charger = create_charger(
        hass,
        home,
        name=name,
        **{CONF_CHARGER_LATITUDE: coords[0], CONF_CHARGER_LONGITUDE: coords[1]},
    )
    charger.is_plugged = MagicMock(return_value=True)
    charger.get_continuous_plug_duration = MagicMock(return_value=plug_duration)
    charger.update_charger_for_user_change = AsyncMock()
    return charger


def _add_plug_sensor(car) -> None:
    car.car_plugged = f"binary_sensor.{car.name.lower()}_plugged"
    car.attach_ha_state_to_probe(car.car_plugged, is_numerical=False)
    car._entity_probed_state[car.car_plugged] = []


def _make_car(hass, home, name, with_plug_sensor=False):
    car = make_real_car(hass, home, name=name)
    if with_plug_sensor:
        _add_plug_sensor(car)
    return car


def _fx(car_names=("A", "B"), with_w2=False, with_plug_sensor=False) -> SimpleNamespace:
    hass = make_hass()
    home = make_home()
    w1 = _make_charger(hass, home)
    w2 = None
    if with_w2:
        w2 = _make_charger(hass, home, name="wallbox_2", coords=_at(W1_COORDS, 7.5), plug_duration=340.0)
    cars = {n: _make_car(hass, home, n, with_plug_sensor=with_plug_sensor) for n in car_names}
    return SimpleNamespace(hass=hass, home=home, w1=w1, w2=w2, cars=cars)


# =============================================================================
# T1 — D1 score equation
# =============================================================================

# (case, {car: (sensor, dist, plug_duration, gps)}, winner, {car: exact score})
D1_CASES = [
    (
        1,
        {"IDBuzz": ("off", 1.25, None, True), "Twingo": ("on", 7.6, SYNC_228, True)},
        "Twingo",
        {"IDBuzz": 1_100_980_000, "Twingo": 1_510_850_228},
    ),
    (
        2,
        {"IDBuzz": ("off", 1.25, None, True), "Twingo": ("off", 7.6, None, True)},
        "IDBuzz",
        {"IDBuzz": 1_100_980_000, "Twingo": 0},
    ),
    (
        3,
        {"X": ("off", 2.75, None, True), "Y": ("on", 39.75, NO_SYNC, True)},
        "X",
        {"X": 1_100_950_000, "Y": 500_210_000},
    ),
    (
        4,
        {"X": ("off", 2.75, None, True), "Y": ("on", 8.75, NO_SYNC, True)},
        "Y",
        {"X": 1_100_950_000, "Y": 1_500_830_000},
    ),
    (
        5,
        {"IDBuzz": ("on", 1.25, NO_SYNC, True), "Twingo": ("on", 7.6, SYNC_228, True)},
        "Twingo",
        {"IDBuzz": 1_500_980_000, "Twingo": 1_510_850_228},
    ),
    (
        6,
        {"A": ("on", 3.75, NO_SYNC, True), "B": ("on", 6.75, SYNC_228, True)},
        "B",
        {"A": 1_500_930_000, "B": 1_510_870_228},
    ),
    (
        7,
        {"A": ("on", None, SYNC_228, False), "B": ("off", 2.75, None, True)},
        "B",
        {"A": 510_010_228, "B": 1_100_950_000},
    ),
    (
        8,
        {"Twingo": ("on", 11.2, SYNC_228, True), "IDBuzz": ("off", 1.25, None, True)},
        "IDBuzz",
        {"Twingo": 510_780_228, "IDBuzz": 1_100_980_000},
    ),
]


@pytest.mark.parametrize(("case", "cars", "winner", "scores"), D1_CASES, ids=[f"case{c[0]}" for c in D1_CASES])
def test_d1_case_winner(case, cars, winner, scores):
    fx = _fx(car_names=tuple(cars))
    for name, (sensor, dist, plug_duration, gps) in cars.items():
        _set_car(fx.cars[name], sensor=sensor, dist=dist, plug_duration=plug_duration, gps=gps)

    for name, expected in scores.items():
        assert fx.w1.get_car_score(fx.cars[name], T0, {}) == expected, name

    assert fx.w1.get_best_car(T0).name == winner


def test_d1_case9_two_chargers_distance_breaks_sync_tie():
    # W1 plugged at t (400 s), W2 at t+60 s (340 s); A plugged on W1 at t+300 s
    # (100 s) and only reported "on" late. A sits on the W1→W2 line, 1.25 m from W1.
    fx = _fx(car_names=("A",), with_w2=True)
    _set_car(fx.cars["A"], sensor="on", dist=1.25, plug_duration=100.0)

    # sync_ok on both: 251 on W1 (|Δ| 300 s), 267 on W2 (|Δ| 240 s) — distance decides
    assert fx.w1.get_car_score(fx.cars["A"], T0, {}) == 1_510_980_251
    assert fx.w2.get_car_score(fx.cars["A"], T0, {}) == 1_510_880_267

    assert fx.w1.get_best_car(T0).name == "A"
    assert fx.w2.get_best_car(T0).name != "A"


def test_case1_exact_scores():
    fx = _fx(car_names=("IDBuzz", "Twingo"))
    _set_car(fx.cars["IDBuzz"], sensor="off", dist=1.25)
    _set_car(fx.cars["Twingo"], sensor="on", dist=7.6, plug_duration=SYNC_228)

    # Twingo: proximity 1 (10^9) + plug 5 (5·10^8) + sync_ok (10^7) + dist 85 (85·10^4) + sync 228
    assert fx.w1.get_car_score(fx.cars["Twingo"], T0, {}) == 1_510_850_228
    # ID.buzz: proximity 1 (10^9) + ≤ 3 m bonus 1 (10^8) + dist 98 (98·10^4), no sync
    assert fx.w1.get_car_score(fx.cars["IDBuzz"], T0, {}) == 1_100_980_000


def test_plug_proxy_bump_not_additive():
    fx = _fx(car_names=("A",))
    _set_car(fx.cars["A"], sensor="on", dist=1.25, plug_duration=NO_SYNC)
    # plug_bump 5, not 5 + 1
    assert fx.w1.get_car_score(fx.cars["A"], T0, {}) == 1_500_980_000


def test_plug_instant_fallback_weight():
    fx = _fx(car_names=("A",))
    _set_car(fx.cars["A"], sensor="fallback", dist=6.75, plug_duration=NO_SYNC)
    assert fx.w1.get_car_score(fx.cars["A"], T0, {}) == 1_200_870_000


def test_off_car_never_gets_sync_or_incumbent_bonus():
    # guard: an "off" car within 3 m, attached for 30 min, with a plug duration that
    # WOULD sync: no sync, no sync_ok, no +1 incumbent.
    fx = _fx(car_names=("A",))
    car = fx.cars["A"]
    _set_car(car, sensor="off", dist=1.25, plug_duration=SYNC_228)
    fx.w1.attach_car(car, T0 - timedelta(minutes=30))
    assert fx.w1.get_car_score(car, T0, {}) == 1_100_980_000


def test_incumbent_bonus_does_not_set_sync_ok():
    fx = _fx(car_names=("A",))
    car = fx.cars["A"]
    _set_car(car, sensor="on", dist=6.75, plug_duration=NO_SYNC)
    fx.w1.attach_car(car, T0 - timedelta(minutes=30))
    assert fx.w1.get_car_score(car, T0, {}) == 1 + 87 * 10**4 + 5 * 10**8 + 10**9


@pytest.mark.parametrize(
    ("dist", "sensor", "expected"),
    [
        # off at exactly 3 m: ≤ 3 m bonus AND proximity granted
        (CAR_CHARGER_PLUG_PROXY_RADIUS_M, "off", 10**9 + 10**8 + 95 * 10**4),
        # on at exactly 10 m: proximity granted
        (CAR_CHARGER_PROXIMITY_RADIUS_M, "on", 10**9 + 5 * 10**8 + 81 * 10**4),
    ],
)
def test_radius_boundaries_inclusive(dist, sensor, expected):
    fx = _fx(car_names=("A",))
    _set_car(fx.cars["A"], sensor=sensor, dist=1.0, plug_duration=NO_SYNC)
    with patch("custom_components.quiet_solar.ha_model.charger.haversine", return_value=dist):
        assert fx.w1.get_car_score(fx.cars["A"], T0, {}) == expected


def test_entry_condition_unchanged():
    # guard
    fx = _fx(car_names=("A", "B"))
    _set_car(fx.cars["A"], sensor="off", dist=5.0)
    _set_car(fx.cars["B"], sensor="on", gps=False, home=False, plug_duration=NO_SYNC)
    assert fx.w1.get_car_score(fx.cars["A"], T0, {}) == 0
    assert fx.w1.get_car_score(fx.cars["B"], T0, {}) == 0


# =============================================================================
# T2 — D2 floor
# =============================================================================


@pytest.mark.parametrize("attach_age", [timedelta(minutes=1), timedelta(hours=2)], ids=["1min", "2h"])
def test_wrong_car_displaced_whatever_attach_age(attach_age):
    fx = _fx(car_names=("IDBuzz", "Twingo"))
    buzz, twingo = fx.cars["IDBuzz"], fx.cars["Twingo"]
    _set_car(buzz, sensor="off", dist=1.25)
    _set_car(twingo, sensor="off", dist=7.6)
    fx.w1.attach_car(buzz, T0 - attach_age)
    assert fx.w1.get_car_score(buzz, T0 - timedelta(seconds=7), {}) == 1_100_980_000

    # the Twingo API finally reports "on", with a clean plug sync
    _set_car(twingo, sensor="on", dist=7.6, plug_duration=SYNC_228)
    assert fx.w1.get_best_car(T0).name == "Twingo"
    # no absolute lock any more
    assert fx.w1.get_car_score(buzz, T0, {}) == 1_100_980_000


@pytest.mark.parametrize("dropped", ["unknown", "off"])
def test_attached_car_keeps_best_score_when_sensor_drops(caplog, dropped):
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    fx = _fx(car_names=("A", "B"))
    a, b = fx.cars["A"], fx.cars["B"]
    _set_car(a, sensor="on", dist=6.75, plug_duration=SYNC_228)
    _set_car(b, sensor="off", dist=40.0)
    fx.w1.attach_car(a, T0 - timedelta(minutes=30))
    assert fx.w1.get_car_score(a, T0 - timedelta(seconds=7), {}) == F_NIGHT

    # night: A's API falls asleep (or reports a false "off"), B (plugged for hours)
    # is "on" at 2.25 m
    _set_car(a, sensor=dropped, dist=6.75)
    _set_car(b, sensor="on", dist=2.25, plug_duration=NO_SYNC)
    assert fx.w1.get_car_score(a, T0, {}) == F_NIGHT
    assert fx.w1.get_car_score(b, T0, {}) == 1_500_960_000
    assert fx.w1.get_best_car(T0).name == "A"

    floored_lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("get_car_score: A for wallbox_1 ") and "floored: True" in r.getMessage()
    ]
    assert floored_lines


@pytest.mark.parametrize("variant", ["equal", "strictly_better"])
def test_floored_car_vs_challenger(variant):
    fx = _fx(car_names=("A", "B"))
    a, b = fx.cars["A"], fx.cars["B"]
    if variant == "equal":
        _set_car(a, sensor="on", dist=6.75, plug_duration=SYNC_228)
        fx.w1.attach_car(a, T0 - timedelta(minutes=5))
    else:
        _set_car(a, sensor="on", dist=6.75, plug_duration=NO_SYNC)
        fx.w1.attach_car(a, T0 - timedelta(hours=2))
    _set_car(b, sensor="off", dist=40.0)
    floor = fx.w1.get_car_score(a, T0 - timedelta(seconds=7), {})

    _set_car(a, sensor="unknown", dist=6.75)
    if variant == "equal":
        _set_car(b, sensor="on", dist=6.75, plug_duration=SYNC_228)
        assert fx.w1.get_car_score(b, T0, {}) == floor
        assert fx.w1.get_best_car(T0).name == "A"
    else:
        _set_car(b, sensor="on", dist=1.25, plug_duration=SYNC_228)
        assert fx.w1.get_car_score(b, T0, {}) > floor
        assert fx.w1.get_best_car(T0).name == "B"


@pytest.mark.parametrize("exit_path", ["detach", "unplug_reset", "car_change"])
def test_floor_reset_on_detach_unplug_and_car_change(exit_path):
    fx = _fx(car_names=("A", "B"))
    a, b = fx.cars["A"], fx.cars["B"]
    _set_car(a, sensor="on", dist=6.75, plug_duration=SYNC_228)
    fx.w1.attach_car(a, T0 - timedelta(minutes=30))
    fx.w1.get_car_score(a, T0, {})
    assert fx.w1._attached_car_best_score == F_NIGHT

    if exit_path == "detach":
        fx.w1.detach_car()
    elif exit_path == "unplug_reset":
        fx.w1.reset(keep_commands=True)
    else:
        fx.w1.attach_car(b, T0)
    assert fx.w1._attached_car_best_score is None


@pytest.mark.parametrize("path", ["manual_other_car", "invited_car"])
def test_floor_excluded_paths_scores(path):
    # guard (scores only)
    fx = _fx(car_names=("A", "B"))
    a = fx.cars["A"]
    if path == "manual_other_car":
        _set_car(a, sensor="on", dist=6.75, plug_duration=SYNC_228)
        fx.w1.attach_car(a, T0 - timedelta(minutes=30))
        # A is scored FIRST, so it has a real floor the manual branch could leak
        assert fx.w1.get_car_score(a, T0 - timedelta(seconds=7), {}) == F_NIGHT
        assert fx.w1._attached_car_best_score == F_NIGHT
        fx.w1.set_user_originated(USER_ORIGINATED_CAR_NAME, "B")
        # the floor is BYPASSED by the manual choice of another car (not reset: the
        # car is still attached)
        assert fx.w1.get_car_score(a, T0, {}) == 0
        assert fx.w1._attached_car_best_score == F_NIGHT
        return
    guest = make_real_car(fx.hass, fx.home, name="Guest", is_invited=True)
    fx.w1.attach_car(guest, T0 - timedelta(minutes=30))
    assert fx.w1.get_car_score(guest, T0, {}) == -1
    assert fx.w1._attached_car_best_score is None


def test_floor_excluded_paths_attribute():
    fx = _fx(car_names=("A",), with_w2=True)
    a = fx.cars["A"]
    _set_car(a, sensor="on", dist=1.25, plug_duration=SYNC_228)

    # scored on ANOTHER charger than its own: no floor anywhere
    fx.w1.attach_car(a, T0 - timedelta(minutes=30))
    assert fx.w2.get_car_score(a, T0, {}) > 0
    assert fx.w2._attached_car_best_score is None
    assert fx.w1._attached_car_best_score is None

    # the manual selection of the attached car never feeds the floor
    fx.w1.set_user_originated(USER_ORIGINATED_CAR_NAME, "A")
    assert fx.w1.get_car_score(a, T0, {}) == CAR_SCORE_MAX
    assert fx.w1._attached_car_best_score is None


# ---- persistence ------------------------------------------------------------


def _first_life(shutdown_s=600.0, plug_at_save=None):
    """Charger W1 with A attached and floored at F_NIGHT; return the saved payload.

    `plug_at_save`: a REAL reading of A's plug sensor just before the save.
    """
    fx = _fx(car_names=("A", "B"), with_plug_sensor=True)
    a = fx.cars["A"]
    _set_car(a, sensor="on", dist=6.75, plug_duration=SYNC_228)
    fx.w1.attach_car(a, T0 - timedelta(hours=1))
    assert fx.w1.get_car_score(a, T0 - timedelta(minutes=20), {}) == F_NIGHT
    data: dict = {}
    saved_at = T0 - timedelta(seconds=shutdown_s)
    if plug_at_save is not None:
        a._add_state_history(a.car_plugged, plug_at_save, saved_at - timedelta(seconds=5), None, {})
    fx.w1.update_to_be_saved_extra_device_info(data, saved_at)
    assert data[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_CAR_NAME] == "A"
    assert data[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE] == F_NIGHT
    assert data[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_SAVED_AT] == saved_at.isoformat()
    assert data[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_PLUG_STATE] == plug_at_save
    return copy.deepcopy(data)


def _reboot(data, boot_time=T0 - timedelta(seconds=60), post_init=True) -> SimpleNamespace:
    """A NEW home with a NEW charger of the same name, restored from `data`."""
    fx = _fx(car_names=("A", "B"), with_plug_sensor=True)
    for car in fx.cars.values():
        _set_car(car, sensor="unknown", dist=6.75)
    fx.w1.use_saved_extra_device_info(copy.deepcopy(data))
    if post_init:
        fx.w1.device_post_home_init(boot_time)
    return fx


def _history(car, entries) -> None:
    for offset_s, value in entries:
        car._add_state_history(car.car_plugged, value, T0 + timedelta(seconds=offset_s), None, {})


@pytest.mark.parametrize(
    ("scenario", "restored"),
    [
        ("same_car_on_hole_on", True),
        ("same_car_no_history", True),
        ("same_car_off_after_restart", False),
        ("off_before_saved_at_only", True),
        ("other_car", False),
        ("generic_first_then_car", True),
        ("generic_first_then_expiry", False),
        ("detach_then_reattach", False),
    ],
)
def test_floor_persist_restore(scenario, restored):
    data = _first_life()
    fx = _reboot(data)
    a = fx.cars["A"]

    if scenario == "same_car_on_hole_on":
        _history(a, [(-1200, "on"), (-60, None), (-5, "on")])
        fx.w1.attach_car(a, T0)
    elif scenario == "same_car_no_history":
        fx.w1.attach_car(a, T0)
    elif scenario == "same_car_off_after_restart":
        _history(a, [(-1200, "on"), (-60, None), (-30, "off")])
        fx.w1.attach_car(a, T0)
    elif scenario == "off_before_saved_at_only":
        # an "off" older than saved_at is no proof of an unplug during the shutdown
        _history(a, [(-1200, "off")])
        fx.w1.attach_car(a, T0)
    elif scenario == "other_car":
        fx.w1.attach_car(fx.cars["B"], T0)
        assert fx.w1._attached_car_best_score is None
        # nothing consumed for a non-matching car
        assert fx.w1._boot_attached_car_best_score is not None
        return
    elif scenario == "generic_first_then_car":
        fx.w1.attach_car(fx.w1._default_generic_car, T0 - timedelta(seconds=10))
        assert fx.w1._boot_attached_car_best_score is not None
        fx.w1.attach_car(a, T0)
    elif scenario == "generic_first_then_expiry":
        fx.w1.attach_car(fx.w1._default_generic_car, T0 - timedelta(seconds=10))
        fx.w1.reset_boot_data()
        fx.w1.attach_car(a, T0)
    elif scenario == "detach_then_reattach":
        fx.w1.attach_car(a, T0)
        assert fx.w1._attached_car_best_score == F_NIGHT
        fx.w1.detach_car()
        fx.w1.attach_car(a, T0 + timedelta(seconds=10))

    if restored:
        assert fx.w1._attached_car_best_score == F_NIGHT
        assert fx.w1._boot_attached_car_best_score is None
        # the floor is now what protects the sleeping-API car
        assert fx.w1.get_car_score(a, T0 + timedelta(seconds=10), {}) == F_NIGHT
    else:
        assert fx.w1._attached_car_best_score is None


def test_floor_not_restored_when_saved_at_in_the_future(caplog):
    # host clock behind saved_at (no RTC before NTP sync): no "off" since saved_at
    # can ever be seen, so refuse and consume the pending value
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    data = _first_life()
    data[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_SAVED_AT] = (T0 + timedelta(hours=1)).isoformat()
    fx = _reboot(data)
    fx.w1.attach_car(fx.cars["A"], T0)
    assert fx.w1._attached_car_best_score is None
    assert fx.w1._boot_attached_car_best_score is None
    assert any("in the future" in r.getMessage() for r in caplog.records)


def test_floor_restore_ignores_car_without_plug_sensor():
    data = _first_life()
    fx = _reboot(data)
    a = fx.cars["A"]
    a.car_plugged = None
    fx.w1.attach_car(a, T0)
    assert fx.w1._attached_car_best_score == F_NIGHT


def _bad(mutation):
    data = _first_life()
    mutation(data)
    return data


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.pop(STORAGE_KEY_ATTACHED_CAR_BEST_SCORE),
        lambda d: [d.pop(k) for k in list(d) if k.startswith("attached_car_best_score")],
        lambda d: d.__setitem__(STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_CAR_NAME, 42),
        lambda d: d.__setitem__(STORAGE_KEY_ATTACHED_CAR_BEST_SCORE, True),
        lambda d: d.__setitem__(STORAGE_KEY_ATTACHED_CAR_BEST_SCORE, "1000"),
        lambda d: d.__setitem__(STORAGE_KEY_ATTACHED_CAR_BEST_SCORE, float("nan")),
        lambda d: d.__setitem__(STORAGE_KEY_ATTACHED_CAR_BEST_SCORE, float("inf")),
        lambda d: d.__setitem__(STORAGE_KEY_ATTACHED_CAR_BEST_SCORE, 0),
        lambda d: d.__setitem__(STORAGE_KEY_ATTACHED_CAR_BEST_SCORE, -5.0),
        lambda d: d.__setitem__(STORAGE_KEY_ATTACHED_CAR_BEST_SCORE, CAR_SCORE_MAX),
        lambda d: d.__setitem__(STORAGE_KEY_ATTACHED_CAR_BEST_SCORE, 10**400),
        lambda d: d.__setitem__(STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_SAVED_AT, "not a date"),
        lambda d: d.__setitem__(STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_SAVED_AT, 12),
        lambda d: d.__setitem__(STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_SAVED_AT, "2026-10-11T13:00:00"),
    ],
    ids=[
        "score_key_missing",
        "all_keys_missing",
        "name_not_str",
        "score_bool",
        "score_str",
        "score_nan",
        "score_inf",
        "score_zero",
        "score_negative",
        "score_max",
        "score_huge_int",
        "date_unparsable",
        "date_not_str",
        "date_naive",
    ],
)
def test_floor_persist_invalid_storage(mutation):
    fx = _reboot(_bad(mutation), post_init=False)
    assert fx.w1._saved_attached_car_best_score is None
    fx.w1.device_post_home_init(T0 - timedelta(seconds=60))
    fx.w1.attach_car(fx.cars["A"], T0)
    assert fx.w1._attached_car_best_score is None


def test_floor_persist_int_score_accepted():
    data = _first_life()
    data[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE] = 1_000_000
    fx = _reboot(data, post_init=False)
    assert fx.w1._saved_attached_car_best_score[1] == 1_000_000.0


def _assert_saved_none(d: dict) -> None:
    assert d[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_CAR_NAME] is None
    assert d[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE] is None
    assert d[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_SAVED_AT] is None


@pytest.mark.parametrize("stage", ["never_transferred", "boot_window_pending"])
def test_floor_save_while_not_attached_writes_none(stage):
    # only the LIVE floor is ever written: a restored value waiting in `_saved_*`
    # (never transferred, e.g. a charger rebuilt after the home init) or in the boot
    # slot is never re-written
    data = _first_life()
    fx = _reboot(data, post_init=stage == "boot_window_pending")
    if stage == "never_transferred":
        assert fx.w1._saved_attached_car_best_score is not None
    else:
        assert fx.w1._boot_attached_car_best_score is not None

    d: dict = {}
    fx.w1.update_to_be_saved_extra_device_info(d, T0 - timedelta(seconds=30))
    _assert_saved_none(d)

    if stage == "boot_window_pending":
        # the pending value lives in memory: a save before the attach does not lose it
        fx.w1.attach_car(fx.cars["A"], T0)
        assert fx.w1._attached_car_best_score == F_NIGHT


def test_floor_saved_as_none_after_unplug():
    data = _first_life()
    fx = _reboot(data)
    fx.w1.attach_car(fx.cars["A"], T0)
    assert fx.w1._attached_car_best_score == F_NIGHT
    fx.w1.reset(keep_commands=True)
    d: dict = {}
    fx.w1.update_to_be_saved_extra_device_info(d, T0)
    _assert_saved_none(d)


def test_floor_saved_with_injected_time_while_attached():
    fx = _fx(car_names=("A",))
    a = fx.cars["A"]
    _set_car(a, sensor="on", dist=6.75, plug_duration=SYNC_228)
    fx.w1.attach_car(a, T0 - timedelta(hours=1))
    d: dict = {}
    # attached but never scored: no floor yet → nothing to save
    fx.w1.update_to_be_saved_extra_device_info(d, T0)
    _assert_saved_none(d)

    fx.w1.get_car_score(a, T0, {})
    save_time = T0 + timedelta(seconds=5)
    fx.w1.update_to_be_saved_extra_device_info(d, save_time)
    assert d[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_CAR_NAME] == "A"
    assert d[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE] == F_NIGHT
    assert datetime.fromisoformat(d[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_SAVED_AT]) == save_time


# =============================================================================
# T3 — D3 user action pins the association
# =============================================================================


def _attached_fx(car_names=("A", "B"), with_w2=False) -> SimpleNamespace:
    fx = _fx(car_names=car_names, with_w2=with_w2)
    a = fx.cars["A"]
    _set_car(a, sensor="off", dist=5.0)
    for name in car_names[1:]:
        _set_car(fx.cars[name], sensor="off", dist=40.0)
    fx.w1.attach_car(a, T0 - timedelta(minutes=30))
    a.default_charge_time = dt_time(hour=7, minute=0)
    return fx


async def _car_set_target(fx):
    await fx.cars["A"].user_set_next_charge_target(70)
    assert fx.cars["A"].get_user_originated("charge_target_percent") == 70


async def _car_force(fx):
    await fx.cars["A"].user_force_charge_now()
    assert fx.cars["A"].do_force_next_charge is True


async def _car_default_charge(fx):
    await fx.cars["A"].user_add_default_charge()
    assert fx.cars["A"].do_next_charge_time is not None


async def _car_bump(fx):
    fx.cars["A"].qs_bump_solar_charge_priority = True
    assert fx.cars["A"].qs_bump_solar_charge_priority is True


async def _car_clean_constraints(fx):
    await fx.cars["A"].user_clean_constraints()
    assert fx.cars["A"].get_user_originated("charge_time") == CHARGE_TIME_CONSTRAINTS_CLEARED


async def _car_person(fx):
    fx.home.get_person_by_name = MagicMock(return_value=SimpleNamespace(name="Alice"))
    fx.cars["A"]._is_person_authorized_for_car = MagicMock(return_value=True)
    await fx.cars["A"].user_set_person_for_car("Alice")
    assert fx.cars["A"].get_user_originated("person_name") == "Alice"


async def _charger_bump(fx):
    fx.w1.qs_bump_solar_charge_priority = True
    assert fx.cars["A"].qs_bump_solar_charge_priority is True


async def _charger_default_charge(fx):
    await fx.w1.user_add_default_charge()
    assert fx.cars["A"].do_next_charge_time is not None


async def _charger_force(fx):
    await fx.w1.user_force_charge_now()
    assert fx.cars["A"].do_force_next_charge is True


PIN_ACTIONS = {
    "car_set_next_charge_target": _car_set_target,
    "car_force_charge_now": _car_force,
    "car_add_default_charge": _car_default_charge,
    "car_bump_solar": _car_bump,
    "car_clean_constraints": _car_clean_constraints,
    "car_set_person": _car_person,
    "charger_bump_solar": _charger_bump,
    "charger_add_default_charge": _charger_default_charge,
    "charger_force_charge_now": _charger_force,
}


@pytest.mark.parametrize("action", list(PIN_ACTIONS))
async def test_user_action_on_attached_car_pins_association(action):
    fx = _attached_fx()
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None
    assert fx.cars["A"]._charger_assignment_is_user_originated() is False

    await PIN_ACTIONS[action](fx)  # also checks the action's own effect is intact

    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"
    assert fx.cars["A"]._charger_assignment_is_user_originated() is True


async def test_same_value_action_after_attach_pins():
    fx = _fx(car_names=("A",))
    a = fx.cars["A"]
    a.qs_bump_solar_charge_priority = True  # not attached: no pin
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None
    fx.w1.attach_car(a, T0)
    a.qs_bump_solar_charge_priority = True  # same value, now attached
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"


async def test_pinned_car_not_stolen_by_synced_challenger():
    fx = _attached_fx()
    _set_car(fx.cars["B"], sensor="on", dist=1.25, plug_duration=SYNC_228)
    assert fx.w1.get_best_car(T0).name == "B"

    fx.cars["A"].qs_bump_solar_charge_priority = True

    assert fx.w1.get_best_car(T0).name == "A"


def _drive_real_cycle_path(chargers, cars) -> None:
    """Local copy of the tiebreak helper (that file must not change — AC11)."""
    for charger in chargers:
        charger.is_charger_unavailable = MagicMock(return_value=False)
        charger.probe_for_possible_needed_reboot = MagicMock(return_value=False)
        charger.is_not_plugged = MagicMock(return_value=False)
        charger.set_charging_num_phases = AsyncMock(return_value=False)
        charger.set_max_charging_current = AsyncMock(return_value=True)
        charger.reboot = AsyncMock()
        charger.is_car_stopped_asking_current = MagicMock(return_value=False)
    for car in cars:
        car.get_car_charge_percent = lambda time=None, *a, **kw: 40.0
        car.get_best_person_next_need = AsyncMock(return_value=(False, None, None, None))
        car.get_next_scheduled_event = AsyncMock(return_value=(None, None))
        car.setup_car_charge_target_if_needed = AsyncMock(return_value=80.0)


@pytest.mark.parametrize("lift", ["unplug_cycle", "car_clean_and_reset", "charger_clean_and_reset"])
async def test_pin_lifted(lift):
    fx = _attached_fx()
    a = fx.cars["A"]
    # give A a real floor first (before the pin: a pinned car never feeds it), so the
    # unplug provably resets it
    _set_car(a, sensor="on", dist=5.0, plug_duration=SYNC_228)
    fx.w1.get_car_score(a, T0 - timedelta(seconds=10), {})
    assert fx.w1._attached_car_best_score > 0
    await a.user_set_next_charge_target(70)
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"

    if lift == "unplug_cycle":
        _drive_real_cycle_path([fx.w1], list(fx.cars.values()))
        fx.w1.is_not_plugged = MagicMock(return_value=True)
        fx.w1.is_plugged = MagicMock(return_value=False)
        await fx.w1.check_load_activity_and_constraints(T0)
        assert fx.w1.car is None
        assert fx.w1._attached_car_best_score is None
    elif lift == "car_clean_and_reset":
        await a.user_clean_and_reset()
    else:
        await fx.w1.user_clean_and_reset()

    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None


async def _oos_unattached(fx):
    fx.w1.detach_car()
    fx.cars["A"].qs_bump_solar_charge_priority = True
    await fx.cars["A"].user_set_next_charge_target(70)


async def _oos_charger_key(fx):
    fx.cars["A"].set_user_originated(USER_ORIGINATED_CHARGER_NAME, fx.w1.name)


async def _oos_charger_clean_constraints(fx):
    await fx.w1.user_clean_constraints()


async def _oos_system_person_hold(fx):
    a = fx.cars["A"]
    a.current_forecasted_person = SimpleNamespace(name="Alice")
    a._is_person_authorized_for_car = MagicMock(return_value=True)
    a.hold_forecasted_person_until(datetime.now(tz=pytz.UTC) + timedelta(hours=3))
    assert a._system_person_hold_name == "Alice"


OUT_OF_SCOPE = {
    "unattached_car": _oos_unattached,
    "charger_name_key": _oos_charger_key,
    "charger_clean_constraints": _oos_charger_clean_constraints,
    "system_person_hold": _oos_system_person_hold,
}


@pytest.mark.parametrize("action", list(OUT_OF_SCOPE))
async def test_no_pin_out_of_scope(action):
    # guard
    fx = _attached_fx()
    await OUT_OF_SCOPE[action](fx)
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None


async def test_action_on_a_never_pins_b():
    # guard: the exclusive bump and the person re-assignment touch B without its hook
    fx = _attached_fx(with_w2=True)
    a, b = fx.cars["A"], fx.cars["B"]
    b.set_user_originated("person_name", "Alice")  # B not attached yet: no pin
    b.qs_bump_solar_charge_priority = True
    fx.w2.attach_car(b, T0 - timedelta(minutes=30))
    assert fx.w2.get_user_originated(USER_ORIGINATED_CAR_NAME) is None

    fx.home.get_person_by_name = MagicMock(return_value=SimpleNamespace(name="Alice"))
    a._is_person_authorized_for_car = MagicMock(return_value=True)
    a.qs_bump_solar_charge_priority = True
    await a.user_set_person_for_car("Alice")

    assert b.qs_bump_solar_charge_priority is False
    assert b.get_user_originated("person_name") is None
    assert fx.w2.get_user_originated(USER_ORIGINATED_CAR_NAME) is None
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"


async def test_boot_replay_before_attach_does_not_pin():
    # guard: characterises the order (entity restore replays BEFORE the first attach)
    fx = _fx(car_names=("A",))
    a = fx.cars["A"]
    _set_car(a, sensor="on", dist=1.25, plug_duration=SYNC_228)
    _drive_real_cycle_path([fx.w1], [a])
    fx.w1.device_post_home_init(T0)

    # restored entities replay their user setters (switch bump, select target)
    a.qs_bump_solar_charge_priority = True
    await a.user_set_next_charge_target(80)
    await fx.w1.check_load_activity_and_constraints(T0 + timedelta(seconds=5))
    assert fx.w1.car is None

    await fx.w1.check_load_activity_and_constraints(T0 + timedelta(seconds=40))
    assert fx.w1.car is a
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None


@pytest.mark.parametrize("choice", ["wallbox_2", FORCE_CAR_NO_CHARGER_CONNECTED, "wallbox_1"])
async def test_car_card_charger_change(choice):
    fx = _attached_fx(car_names=("A",), with_w2=True)
    a = fx.cars["A"]
    _set_car(a, sensor="on", dist=1.25, plug_duration=SYNC_228)
    a.qs_bump_solar_charge_priority = True
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"

    await a.user_set_selected_charger_by_name(choice)

    if choice == "wallbox_1":
        assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"
        assert fx.w1.get_best_car(T0) is a
        return

    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None
    assert fx.w1.get_best_car(T0) is not a
    if choice == "wallbox_2":
        assert fx.w2.get_best_car(T0) is a


async def test_user_action_on_generic_car_pins_it():
    fx = _fx(car_names=("A",), with_w2=True)
    a = fx.cars["A"]
    generic = fx.w1._default_generic_car
    fx.w1.attach_car(generic, T0 - timedelta(minutes=5))

    generic.qs_bump_solar_charge_priority = True
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == generic.name

    # a real eligible car right at W1: 0 on W1, allocated on W2
    _set_car(a, sensor="on", dist=1.25, plug_duration=SYNC_228)
    assert fx.w1.get_car_score(a, T0, {}) == 0
    assert fx.w1.get_best_car(T0) is generic
    assert fx.w2.get_best_car(T0) is a


# ---- pin lifetime (review fix #01 S1) ------------------------------------------


async def test_unplug_clears_pin_even_when_pinned_car_left():
    # S1a: the pinned car was taken by another charger, so the pinned charger has no
    # car any more — its physical unplug must still drop the pin
    fx = _attached_fx(car_names=("A",), with_w2=True)
    a = fx.cars["A"]
    await a.user_set_next_charge_target(70)
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"
    fx.w1.detach_car()
    fx.w2.attach_car(a, T0 - timedelta(minutes=1))
    assert fx.w1.car is None
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"

    _drive_real_cycle_path([fx.w1], [a])
    fx.w1.is_not_plugged = MagicMock(return_value=True)
    fx.w1.is_plugged = MagicMock(return_value=False)
    await fx.w1.check_load_activity_and_constraints(T0)

    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None


@pytest.mark.parametrize("w1_state", ["unavailable", "disabled"])
async def test_car_pinned_on_inactive_charger_not_won_elsewhere(w1_state):
    # S1b: W1 out of W2's active chargers (unavailable > 15 s) must not lose its
    # pinned car to W2; a DISABLED charger's pin protects nothing
    fx = _attached_fx(car_names=("A",), with_w2=True)
    a = fx.cars["A"]
    _set_car(a, sensor="on", dist=1.25, plug_duration=SYNC_228)
    a.qs_bump_solar_charge_priority = True
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"

    if w1_state == "unavailable":
        fx.w1.is_plugged = MagicMock(return_value=False)
    else:
        fx.w1._enabled = False
    assert fx.w2.get_car_score(a, T0, {}) > 0

    best = fx.w2.get_best_car(T0)

    if w1_state == "unavailable":
        assert best is not a
        assert fx.w1.car is a
        assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"
    else:
        assert best is a


@pytest.mark.parametrize("choice", ["wallbox_2", FORCE_CAR_NO_CHARGER_CONNECTED])
async def test_car_card_charger_choice_clears_stale_pin_of_unattached_car(choice):
    # S1c: the car is attached nowhere, yet W1 still pins it: choosing another
    # charger (or "no charger") from the car card must drop that stale pin
    fx = _fx(car_names=("A", "B"), with_w2=True)
    a = fx.cars["A"]
    _set_car(a, sensor="on", dist=1.25, plug_duration=SYNC_228)
    _set_car(fx.cars["B"], sensor="off", dist=40.0)
    fx.w1.set_user_originated(USER_ORIGINATED_CAR_NAME, "A")
    assert a.charger is None

    await a.user_set_selected_charger_by_name(choice)

    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None
    if choice == "wallbox_2":
        assert fx.w2.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"


async def test_car_card_charger_choice_keeps_other_cars_pins():
    # guard: only the pins naming THIS car are dropped
    fx = _fx(car_names=("A", "B"), with_w2=True)
    fx.w1.set_user_originated(USER_ORIGINATED_CAR_NAME, "B")
    await fx.cars["A"].user_set_selected_charger_by_name(FORCE_CAR_NO_CHARGER_CONNECTED)
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "B"


# ---- entity restore replay (review fix #01 S2) ---------------------------------


def _description(entities, key):
    return next(e.entity_description for e in entities if e.entity_description.key == key)


@pytest.mark.parametrize("for_init", [True, False], ids=["restore_replay", "user_action"])
@pytest.mark.parametrize("entity", ["car_target_select", "car_bump_switch", "charger_bump_switch"])
async def test_entity_restore_replay_does_not_pin(entity, for_init):
    from custom_components.quiet_solar.const import SWITCH_CAR_BUMP_SOLAR_CHARGE_PRIORITY
    from custom_components.quiet_solar.select import create_ha_select_for_QSCar
    from custom_components.quiet_solar.switch import create_ha_switch_for_QSCar, create_ha_switch_for_QSCharger

    fx = _attached_fx()
    a = fx.cars["A"]

    if entity == "car_target_select":
        desc = _description(create_ha_select_for_QSCar(a), "selected_next_charge_limit_for_car")
        await desc.async_set_current_option_fn(a, desc.key, 70, for_init)
        assert a.get_user_originated("charge_target_percent") == 70
    elif entity == "car_bump_switch":
        desc = _description(create_ha_switch_for_QSCar(a), SWITCH_CAR_BUMP_SOLAR_CHARGE_PRIORITY)
        await desc.async_switch(a, True, for_init)
        assert a.qs_bump_solar_charge_priority is True
    else:
        desc = _description(create_ha_switch_for_QSCharger(fx.w1), SWITCH_CAR_BUMP_SOLAR_CHARGE_PRIORITY)
        await desc.async_switch(fx.w1, True, for_init)
        assert a.qs_bump_solar_charge_priority is True

    # the car's own user-originated replay is untouched; only the charger pin is skipped
    assert a.get_user_originated("bump_solar") is (entity != "car_target_select")
    expected = None if for_init else "A"
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == expected


# ---- review fix #02 ------------------------------------------------------------


async def test_replay_for_init_does_not_leak_across_awaits():
    # R2-1: the replay marker is per asyncio task (ContextVar), not a car attribute
    # held across awaits: a genuine user action DURING a replay still pins, and two
    # interleaved calls can never leave the marker stuck
    fx = _attached_fx()
    a = fx.cars["A"]
    entered = [asyncio.Event(), asyncio.Event()]
    gates = [asyncio.Event(), asyncio.Event()]
    calls: list[int] = []

    async def _blocking_convert():
        i = len(calls)
        calls.append(i)
        entered[i].set()
        await gates[i].wait()
        return False

    a.convert_auto_constraint_to_manual_if_needed = _blocking_convert

    replay = asyncio.create_task(a.user_set_next_charge_target(80, for_init=True))
    await entered[0].wait()

    # (a) a user action on the same car while the replay is suspended pins
    await a.user_force_charge_now()
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"
    fx.w1.clear_user_originated(USER_ORIGINATED_CAR_NAME)

    # (b) a user call interleaved with the replay: the replay ends FIRST
    user = asyncio.create_task(a.user_set_next_charge_target(70))
    await entered[1].wait()
    gates[0].set()
    await replay
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None  # replay: no pin
    gates[1].set()
    await user
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"

    # not stuck: a later user action still pins
    fx.w1.clear_user_originated(USER_ORIGINATED_CAR_NAME)
    await a.user_force_charge_now()
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"


def _pinned_first_life(saved_at, plug_at_save=None) -> dict:
    """W1 with A attached and PINNED by a user action; return the saved payload.

    `plug_at_save`: a REAL reading of A's plug sensor just before the save (A has no
    plug sensor at all when None).
    """
    fx = _attached_fx()
    a = fx.cars["A"]
    if plug_at_save is not None:
        _add_plug_sensor(a)
        a._add_state_history(a.car_plugged, plug_at_save, saved_at - timedelta(seconds=5), None, {})
    a.qs_bump_solar_charge_priority = True
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"
    data: dict = {}
    fx.w1.update_to_be_saved_extra_device_info(data, saved_at)
    assert data[STORAGE_KEY_USER_PIN_SAVED_AT] == saved_at.isoformat()
    assert data[STORAGE_KEY_USER_PIN_PLUG_STATE] == plug_at_save
    return copy.deepcopy(data)


def _pinned_reboot(data, a_history) -> SimpleNamespace:
    """New home / charger restored from `data`, booted at T0; B plugged and synced on W1."""
    fx = _fx(car_names=("A", "B"), with_plug_sensor=True)
    a, b = fx.cars["A"], fx.cars["B"]
    _set_car(a, sensor="unknown", dist=40.0)
    _set_car(b, sensor="on", dist=1.25, plug_duration=SYNC_228)
    _history(a, a_history)
    _drive_real_cycle_path([fx.w1], [a, b])
    fx.w1.use_saved_extra_device_info(copy.deepcopy(data))
    fx.w1.device_post_home_init(T0)
    assert fx.w1._boot_car is a
    assert a.get_user_originated(USER_ORIGINATED_CHARGER_NAME) == fx.w1.name
    return fx


SAVED_AT = T0 - timedelta(seconds=600)


@pytest.mark.parametrize(
    ("scenario", "lifted"),
    [
        ("off_after_save", True),
        ("on_after_save", False),
        ("hole_after_save", False),
        ("off_before_save_only", False),
        ("saved_at_missing", False),
        ("saved_at_unparsable", False),
        ("saved_at_naive", False),
        ("saved_at_not_str", False),
        ("saved_at_in_the_future", False),
        ("no_plug_sensor", False),
    ],
)
async def test_persisted_pin_lifted_when_pinned_car_unplugged_during_downtime(caplog, scenario, lifted):
    # R2-2: A pinned on W1, HA stops, A is unplugged and B plugged on W1 meanwhile.
    # A valid "off" of A's plug sensor since the pin's saved time lifts the
    # persisted pin; anything else (no proof) keeps it, as today.
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    data = _pinned_first_life(SAVED_AT)
    history = {
        "off_after_save": [(-1200, "on"), (10, "off")],
        "on_after_save": [(-1200, "on"), (10, "on")],
        "hole_after_save": [(-1200, "on"), (10, None)],
        "off_before_save_only": [(-1200, "off")],
    }.get(scenario, [(-1200, "on"), (10, "off")])
    if scenario == "saved_at_missing":
        data.pop(STORAGE_KEY_USER_PIN_SAVED_AT)
    elif scenario == "saved_at_unparsable":
        data[STORAGE_KEY_USER_PIN_SAVED_AT] = "not a date"
    elif scenario == "saved_at_naive":
        data[STORAGE_KEY_USER_PIN_SAVED_AT] = "2026-10-11T13:00:00"
    elif scenario == "saved_at_not_str":
        data[STORAGE_KEY_USER_PIN_SAVED_AT] = 12
    elif scenario == "saved_at_in_the_future":
        # R3-4: host clock behind the save — unlike the floor, the pin is KEPT and
        # its guard stays armed (it is re-checked at every boot-window decision)
        data[STORAGE_KEY_USER_PIN_SAVED_AT] = (T0 + timedelta(hours=1)).isoformat()

    fx = _pinned_reboot(data, history)
    a, b = fx.cars["A"], fx.cars["B"]
    if scenario == "no_plug_sensor":
        a.car_plugged = None

    await fx.w1.check_load_activity_and_constraints(T0 + timedelta(seconds=40))

    if lifted:
        assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None
        assert a.get_user_originated(USER_ORIGINATED_CHARGER_NAME) is None
        assert fx.w1._boot_car is None
        assert fx.w1.car is b
        # R5-3: neutral prefix (emitted from both the decision and the unplugged branch)
        assert any(r.getMessage().startswith("boot_user_pin: A on ") for r in caplog.records)
    else:
        assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"
        assert fx.w1.car is a
        if scenario == "saved_at_in_the_future":
            assert fx.w1._boot_user_pin is not None


async def test_persisted_pin_lift_keeps_a_pin_on_another_charger_of_the_car():
    # R2-2: the car's own charger choice is only cleared when it names THIS charger
    data = _pinned_first_life(SAVED_AT)
    fx = _pinned_reboot(data, [(-1200, "on"), (10, "off")])
    a = fx.cars["A"]
    a._user_originated[USER_ORIGINATED_CHARGER_NAME] = "wallbox_9"
    fx.w1.get_best_car(T0 + timedelta(seconds=40))
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None
    assert a.get_user_originated(USER_ORIGINATED_CHARGER_NAME) == "wallbox_9"


async def test_persisted_pin_guard_ends_with_the_boot_window():
    # R2-2: the guard lives exactly as long as the other boot data
    data = _pinned_first_life(SAVED_AT)
    fx = _pinned_reboot(data, [(-1200, "on")])
    assert fx.w1._boot_user_pin is not None
    fx.w1.reset_boot_data()
    assert fx.w1._boot_user_pin is None
    _history(fx.cars["A"], [(10, "off")])
    fx.w1.get_best_car(T0 + timedelta(seconds=40))
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"


async def test_persisted_pin_guard_dropped_by_a_new_user_pin():
    # R2-2: a pin the user sets AFTER the restart postdates every "off": never lifted
    data = _pinned_first_life(SAVED_AT)
    fx = _pinned_reboot(data, [(-1200, "on"), (10, "off")])
    await fx.w1.user_set_selected_car_by_name("A")
    assert fx.w1._boot_user_pin is None
    fx.w1.get_best_car(T0 + timedelta(seconds=40))
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"


async def test_persisted_generic_car_pin_kept():
    # R2-2: a pinned car unknown to the home (the generic car, a removed car) has no
    # plug sensor to prove anything: the pin is kept
    fx = _fx(car_names=("A",))
    generic = fx.w1._default_generic_car
    fx.w1.attach_car(generic, T0 - timedelta(minutes=5))
    generic.qs_bump_solar_charge_priority = True
    data: dict = {}
    fx.w1.update_to_be_saved_extra_device_info(data, SAVED_AT)

    fx2 = _fx(car_names=("A",))
    fx2.w1.use_saved_extra_device_info(copy.deepcopy(data))
    fx2.w1.device_post_home_init(T0)
    assert fx2.w1._boot_user_pin == (generic.name, SAVED_AT, None)
    fx2.w1.get_best_car(T0 + timedelta(seconds=40))
    assert fx2.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == generic.name
    assert fx2.w1._boot_user_pin is not None


async def test_persisted_pin_guard_ignores_a_changed_pin():
    # R2-2: the guard only applies to the very pin that was persisted
    data = _pinned_first_life(SAVED_AT)
    fx = _pinned_reboot(data, [(-1200, "on"), (10, "off")])
    fx.w1._user_originated[USER_ORIGINATED_CAR_NAME] = "B"  # restore-style write, no hook
    fx.w1.get_best_car(T0 + timedelta(seconds=40))
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "B"


@pytest.mark.parametrize(
    ("pin", "attached", "expect_saved_at"),
    [
        (None, False, False),
        (CHARGER_NO_CAR_CONNECTED, False, False),
        ("A", False, False),
        ("A", True, True),
        ("B", True, False),
    ],
    ids=["no_pin", "no_car_choice", "preselection", "attached_car_pin", "pin_on_another_car"],
)
def test_pin_saved_at_written_only_for_an_attached_car_pin(pin, attached, expect_saved_at):
    # R5-1: only a pin on the car attached HERE is stamped (arms the boot guard); a
    # pre-selection (pinned car not attached here) writes None for both keys
    fx = _fx(car_names=("A", "B"), with_plug_sensor=True)
    a = fx.cars["A"]
    if attached:
        fx.w1.attach_car(a, T0 - timedelta(minutes=5))
    a._add_state_history(a.car_plugged, "on", T0 - timedelta(seconds=5), None, {})
    if pin is not None:
        fx.w1.set_user_originated(USER_ORIGINATED_CAR_NAME, pin)
    d: dict = {}
    fx.w1.update_to_be_saved_extra_device_info(d, T0)
    assert d[STORAGE_KEY_USER_PIN_SAVED_AT] == (T0.isoformat() if expect_saved_at else None)
    assert d[STORAGE_KEY_USER_PIN_PLUG_STATE] == ("on" if expect_saved_at else None)


def test_pin_saved_at_not_handed_to_boot_without_a_car_pin():
    # a saved time left without its pin (e.g. hand-edited store) arms nothing
    data = _pinned_first_life(SAVED_AT)
    data[STORAGE_KEY_USER_ORIGINATED] = {}
    fx = _fx(car_names=("A", "B"), with_plug_sensor=True)
    fx.w1.use_saved_extra_device_info(copy.deepcopy(data))
    fx.w1.device_post_home_init(T0)
    assert fx.w1._boot_user_pin is None


async def test_disabling_a_pinned_charger_drops_its_pin():
    # R2-3: disable -> reset + remove from the home: the pin must not survive and
    # pull the car back from its new charger at the re-enable
    fx = _attached_fx(car_names=("A",), with_w2=True)
    a = fx.cars["A"]
    fx.home.remove_device = MagicMock(side_effect=fx.home._chargers.remove)
    fx.home.add_device = MagicMock(side_effect=fx.home._chargers.append)
    a.qs_bump_solar_charge_priority = True
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"

    fx.w1.qs_enable_device = False
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None
    assert fx.w1.car is None

    # A now right at W2
    _set_car(a, sensor="on", dist=7.5, plug_duration=SYNC_228)
    assert fx.w2.get_best_car(T0) is a
    fx.w2.attach_car(a, T0)

    fx.w1.qs_enable_device = True
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None
    assert fx.w1.get_best_car(T0) is not a
    assert fx.w2.get_best_car(T0) is a


async def test_pin_on_inactive_charger_not_holding_the_car_protects_nothing():
    # R2-4: W1 pinned to A but holding nothing (e.g. a car-card pick never attached),
    # W1 inactive: A is NOT blocked everywhere — W2 wins it
    fx = _fx(car_names=("A",), with_w2=True)
    a = fx.cars["A"]
    _set_car(a, sensor="on", dist=1.25, plug_duration=SYNC_228)
    fx.w1.set_user_originated(USER_ORIGINATED_CAR_NAME, "A")
    fx.w1.is_plugged = MagicMock(return_value=False)
    assert fx.w1.car is None

    assert fx.w2.get_best_car(T0) is a


async def test_pinned_elsewhere_skip_logged_at_info_once(caplog):
    # R2-4: the S1b skip is visible at INFO, once per (charger, car) episode
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    fx = _attached_fx(car_names=("A",), with_w2=True)
    a = fx.cars["A"]
    _set_car(a, sensor="on", dist=1.25, plug_duration=SYNC_228)
    a.qs_bump_solar_charge_priority = True
    fx.w1.is_plugged = MagicMock(return_value=False)

    fx.w2.get_best_car(T0)
    fx.w2.get_best_car(T0 + timedelta(seconds=7))

    lines = [r for r in caplog.records if "pinned on inactive charger" in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO
    assert "wallbox_1" in lines[0].getMessage()


@pytest.mark.parametrize("with_challenger", [False, True], ids=["boot", "boot_over_computed"])
async def test_boot_car_pinned_elsewhere_not_taken(with_challenger):
    # R2-6: W2's boot car A is held and pinned by the inactive W1: neither boot
    # branch may hand it to W2 (one ping-pong otherwise)
    fx = _attached_fx(car_names=("A", "B"), with_w2=True)
    a, b = fx.cars["A"], fx.cars["B"]
    _set_car(a, sensor="on", dist=1.25, plug_duration=SYNC_228)
    a.qs_bump_solar_charge_priority = True
    fx.w1.is_plugged = MagicMock(return_value=False)
    fx.w2._boot_car = a
    if with_challenger:
        _set_car(b, sensor="on", dist=7.5, plug_duration=SYNC_228)

    best = fx.w2.get_best_car(T0)

    assert best is not a
    assert fx.w1.car is a
    if with_challenger:
        assert best is b
    else:
        assert best is fx.w2._default_generic_car


async def test_charger_card_choice_clears_the_same_car_pin_on_other_chargers():
    # R2-7: mirror of the car-card S1c on the charger-card path
    fx = _fx(car_names=("A", "B"), with_w2=True)
    fx.w1.set_user_originated(USER_ORIGINATED_CAR_NAME, "A")
    fx.w2.set_user_originated(USER_ORIGINATED_CAR_NAME, "B")

    await fx.w2.user_set_selected_car_by_name("A")

    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None
    assert fx.w2.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"


@pytest.mark.parametrize("choice", ["generic", CHARGER_NO_CAR_CONNECTED])
async def test_charger_card_non_car_choice_keeps_other_pins(choice):
    # R2-7 guard: only a REAL car choice clears that car's pin elsewhere
    fx = _fx(car_names=("A",), with_w2=True)
    generic_name = fx.w2._default_generic_car.name
    fx.w1.set_user_originated(USER_ORIGINATED_CAR_NAME, generic_name if choice == "generic" else "A")
    await fx.w2.user_set_selected_car_by_name(generic_name if choice == "generic" else choice)
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is not None


async def test_unplug_clears_a_no_car_choice():
    # R2-8: a charger choice (even "no car connected") lives until the physical
    # unplug — a REAL plugged -> unplugged edge (review fix #03 R3-2)
    fx = _fx(car_names=("A",))
    await fx.w1.user_set_selected_car_by_name(CHARGER_NO_CAR_CONNECTED)
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == CHARGER_NO_CAR_CONNECTED

    _drive_real_cycle_path([fx.w1], list(fx.cars.values()))
    fx.w1.is_plugged = MagicMock(return_value=True)
    await fx.w1.check_load_activity_and_constraints(T0)
    assert fx.w1.car is None
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == CHARGER_NO_CAR_CONNECTED

    fx.w1.is_not_plugged = MagicMock(return_value=True)
    fx.w1.is_plugged = MagicMock(return_value=False)
    await fx.w1.check_load_activity_and_constraints(T0 + timedelta(seconds=7))

    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None


async def test_bump_switch_setter_rejects_a_foreign_device():
    # R2-9: an explicit type check (an `assert` is stripped under `python -O`)
    from custom_components.quiet_solar.switch import _async_set_bump_solar_priority

    with pytest.raises(TypeError):
        await _async_set_bump_solar_priority(object(), True, False)


# ---- review fix #03 ------------------------------------------------------------


def _poll_plug(car, value, time) -> None:
    """A REAL poll of the car plug sensor, as `update_states` does at every cycle.

    `add_to_history(entity, time)` stamps the sample with the poll `time`: a sensor
    stuck at `off` since before the save gets a fresh `off` sample after the boot.
    """
    car.hass.states.get = MagicMock(side_effect=lambda eid: State(eid, value) if eid == car.car_plugged else None)
    car.add_to_history(car.car_plugged, time)


# (id, real reading at save, history loaded from the recorder, post-boot polls, unplug proven)
R3_1_ROWS = [
    ("stuck_off_saved_off", "off", [(-1200, "off")], ["off"], False),
    ("stuck_off_not_saved", None, [(-1200, "off")], ["off"], False),
    ("on_then_off", None, [(-1200, "on")], ["off"], True),
    ("saved_on_then_off", "on", [(-1200, "on")], ["off"], True),
    ("on_hole_then_off", None, [(-1200, "on")], ["unavailable", "off"], True),
    ("no_history_saved_on", "on", [], ["off"], True),
    ("no_history_saved_off", "off", [], ["off"], False),
    ("no_history_not_saved", None, [], ["off"], False),
    ("saved_off_over_older_on", "off", [(-1200, "on")], ["off"], False),
]


@pytest.mark.parametrize(
    ("row_id", "plug_at_save", "history", "polls", "proven"), R3_1_ROWS, ids=[r[0] for r in R3_1_ROWS]
)
async def test_unplug_proof_is_an_on_to_off_transition_pin(row_id, plug_at_save, history, polls, proven):
    # R3-1: a re-polled stale `off` (the ID.buzz reports `off` while plugged) is no
    # proof of an unplug: only an `on -> off` transition since the save is
    data = _pinned_first_life(SAVED_AT, plug_at_save=plug_at_save)
    fx = _pinned_reboot(data, history)
    a, b = fx.cars["A"], fx.cars["B"]
    if a.car_plugged is None:
        _add_plug_sensor(a)
        _history(a, history)
    for i, value in enumerate(polls):
        _poll_plug(a, value, T0 + timedelta(seconds=i))

    await fx.w1.check_load_activity_and_constraints(T0 + timedelta(seconds=40))

    if proven:
        assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None
        assert fx.w1.car is b
    else:
        assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"
        assert fx.w1.car is a
        assert fx.w1._boot_user_pin is not None  # still armed: a later real unplug lifts it


@pytest.mark.parametrize(
    ("row_id", "plug_at_save", "history", "polls", "proven"), R3_1_ROWS, ids=[r[0] for r in R3_1_ROWS]
)
def test_unplug_proof_is_an_on_to_off_transition_floor(row_id, plug_at_save, history, polls, proven):
    data = _first_life(plug_at_save=plug_at_save)
    fx = _reboot(data)
    a = fx.cars["A"]
    _history(a, history)
    for i, value in enumerate(polls):
        _poll_plug(a, value, T0 - timedelta(seconds=30 - i))

    fx.w1.attach_car(a, T0)

    assert fx.w1._attached_car_best_score == (None if proven else F_NIGHT)


async def test_pin_lifted_by_a_real_unplug_seen_later_in_the_boot_window():
    # R3-1: no proof at the first decision keeps the pin armed; the transition seen
    # at a later decision of the boot window lifts it
    data = _pinned_first_life(SAVED_AT, plug_at_save="on")
    fx = _pinned_reboot(data, [])
    a = fx.cars["A"]
    fx.w1.get_best_car(T0 + timedelta(seconds=30))
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"

    _poll_plug(a, "off", T0 + timedelta(seconds=35))
    fx.w1.get_best_car(T0 + timedelta(seconds=40))
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None


@pytest.mark.parametrize("value", ["garbage", 12, None, "unavailable"])
def test_saved_plug_reading_invalid_in_store_is_unknown(value):
    # R3-1: anything but "on" / "off" is an unknown reading (the history decides)
    data = _first_life(plug_at_save="on")
    data[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_PLUG_STATE] = value
    fx = _reboot(data, post_init=False)
    assert fx.w1._saved_attached_car_best_score[3] is None

    pin = _pinned_first_life(SAVED_AT, plug_at_save="on")
    pin[STORAGE_KEY_USER_PIN_PLUG_STATE] = value
    fx2 = _fx(car_names=("A", "B"), with_plug_sensor=True)
    fx2.w1.use_saved_extra_device_info(copy.deepcopy(pin))
    fx2.w1.device_post_home_init(T0)
    assert fx2.w1._boot_user_pin == ("A", SAVED_AT, None)


def test_saved_plug_reading_none_without_a_reading():
    # R3-1: a pinned car with a plug sensor but no valid sample yet saves None
    fx = _attached_fx()
    a = fx.cars["A"]
    _add_plug_sensor(a)
    a._add_state_history(a.car_plugged, None, T0 - timedelta(seconds=5), None, {})
    a.qs_bump_solar_charge_priority = True
    d: dict = {}
    fx.w1.update_to_be_saved_extra_device_info(d, T0)
    assert d[STORAGE_KEY_USER_PIN_SAVED_AT] == T0.isoformat()
    assert d[STORAGE_KEY_USER_PIN_PLUG_STATE] is None


@pytest.mark.parametrize("pin", [None, CHARGER_NO_CAR_CONNECTED])
def test_saved_plug_reading_none_without_a_car_pin(pin):
    fx = _fx(car_names=("A",), with_plug_sensor=True)
    if pin is not None:
        fx.w1.set_user_originated(USER_ORIGINATED_CAR_NAME, pin)
    d: dict = {}
    fx.w1.update_to_be_saved_extra_device_info(d, T0)
    assert d[STORAGE_KEY_USER_PIN_PLUG_STATE] is None
    assert d[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_PLUG_STATE] is None


@pytest.mark.parametrize("choice", ["A", CHARGER_NO_CAR_CONNECTED])
async def test_choice_on_an_unplugged_charger_survives_until_the_unplug_edge(choice):
    # R3-2: the unplug clear is edge-triggered (plugged -> unplugged): a choice made
    # on an unplugged charger survives the unplugged cycles, is honoured at the plug,
    # and is cleared by the next physical unplug
    fx = _fx(car_names=("A",))
    a = fx.cars["A"]
    _set_car(a, sensor="off", dist=40.0)
    _drive_real_cycle_path([fx.w1], [a])
    fx.w1.is_not_plugged = MagicMock(return_value=True)
    fx.w1.is_plugged = MagicMock(return_value=False)
    await fx.w1.check_load_activity_and_constraints(T0)

    await fx.w1.user_set_selected_car_by_name(choice)
    for i in range(1, 4):
        await fx.w1.check_load_activity_and_constraints(T0 + timedelta(seconds=7 * i))
        assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == choice

    fx.w1.is_not_plugged = MagicMock(return_value=False)
    fx.w1.is_plugged = MagicMock(return_value=True)
    await fx.w1.check_load_activity_and_constraints(T0 + timedelta(seconds=60))
    assert fx.w1.car is (a if choice == "A" else None)
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == choice

    fx.w1.is_not_plugged = MagicMock(return_value=True)
    fx.w1.is_plugged = MagicMock(return_value=False)
    await fx.w1.check_load_activity_and_constraints(T0 + timedelta(seconds=70))
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None
    assert fx.w1.car is None

    # the edge is consumed: a new choice while still unplugged survives again
    await fx.w1.user_set_selected_car_by_name(choice)
    await fx.w1.check_load_activity_and_constraints(T0 + timedelta(seconds=77))
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == choice


async def test_unplugged_at_boot_is_no_edge():
    # R3-2: a charger seen unplugged at its first cycle was never seen plugged:
    # a persisted choice is kept (as before QS-416 when no car was attached)
    fx = _fx(car_names=("A",))
    fx.w1._user_originated[USER_ORIGINATED_CAR_NAME] = "A"
    _drive_real_cycle_path([fx.w1], list(fx.cars.values()))
    fx.w1.is_not_plugged = MagicMock(return_value=True)
    fx.w1.is_plugged = MagicMock(return_value=False)
    await fx.w1.check_load_activity_and_constraints(T0)
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"


def _entity(entity_cls, device, description):
    ent = entity_cls(data_handler=MagicMock(), device=device, description=description)
    ent.async_write_ha_state = MagicMock()
    return ent


async def _replay_target_select(fx, for_init):
    from custom_components.quiet_solar.select import QSSimpleSelectRestore, create_ha_select_for_QSCar

    a = fx.cars["A"]
    desc = _description(create_ha_select_for_QSCar(a), "selected_next_charge_limit_for_car")
    await _entity(QSSimpleSelectRestore, a, desc).async_select_option(70, for_init=for_init)
    assert a.get_user_originated("charge_target_percent") == 70


async def _replay_bump_switch(fx, for_init, device_name):
    from custom_components.quiet_solar.const import SWITCH_CAR_BUMP_SOLAR_CHARGE_PRIORITY
    from custom_components.quiet_solar.switch import (
        QSSwitchEntityChargerOrCar,
        create_ha_switch_for_QSCar,
        create_ha_switch_for_QSCharger,
    )

    device = fx.cars["A"] if device_name == "car" else fx.w1
    create = create_ha_switch_for_QSCar if device_name == "car" else create_ha_switch_for_QSCharger
    desc = _description(create(device), SWITCH_CAR_BUMP_SOLAR_CHARGE_PRIORITY)
    await _entity(QSSwitchEntityChargerOrCar, device, desc).async_turn_on(for_init=for_init)
    assert fx.cars["A"].qs_bump_solar_charge_priority is True


async def _generic_select(fx, for_init):
    # the shared select dispatch: a setter that does NOT thread `for_init` itself
    from custom_components.quiet_solar.select import QSBaseSelect, create_ha_select_for_QSCar

    a = fx.cars["A"]
    fx.home.get_person_by_name = MagicMock(return_value=SimpleNamespace(name="Alice"))
    a._is_person_authorized_for_car = MagicMock(return_value=True)
    desc = _description(create_ha_select_for_QSCar(a), "selected_person_for_car")
    await _entity(QSBaseSelect, a, desc).set_current_option("Alice", for_init=for_init)
    assert a.get_user_originated("person_name") == "Alice"


async def _generic_switch(fx, for_init):
    # the shared switch dispatch, plain `setattr` path (no `async_switch`)
    from custom_components.quiet_solar.switch import QSSwitchEntity, QSSwitchEntityDescription

    desc = QSSwitchEntityDescription(key="qs_bump_solar_charge_priority")
    for turn_on in (False, True):
        ent = _entity(QSSwitchEntity, fx.cars["A"], desc)
        await (ent.async_turn_on if turn_on else ent.async_turn_off)(for_init=for_init)
        if for_init:
            assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None
    assert fx.cars["A"].qs_bump_solar_charge_priority is True


async def _generic_number(fx, for_init):
    # the shared number dispatch: a setter that does NOT thread `for_init` itself
    from custom_components.quiet_solar.number import QSBaseNumber, QSNumberEntityDescription

    desc = QSNumberEntityDescription(
        key="next_charge_target",
        async_set_fn=lambda device, value, _for_init: device.user_set_next_charge_target(int(value)),
    )
    await _entity(QSBaseNumber, fx.cars["A"], desc).async_set_native_value(70.0, for_init=for_init)
    assert fx.cars["A"].get_user_originated("charge_target_percent") == 70


async def _generic_time(fx, for_init):
    # the shared time dispatch (plain `setattr`): any car setter it reaches
    from custom_components.quiet_solar.time import QSBaseTime, QSTimeEntityDescription

    desc = QSTimeEntityDescription(key="qs_bump_solar_charge_priority")
    await _entity(QSBaseTime, fx.cars["A"], desc).async_set_value(dt_time(hour=7), for_init=for_init)
    assert fx.cars["A"].qs_bump_solar_charge_priority is True


REPLAYS = {
    "car_target_select": _replay_target_select,
    "car_bump_switch": lambda fx, for_init: _replay_bump_switch(fx, for_init, "car"),
    "charger_bump_switch": lambda fx, for_init: _replay_bump_switch(fx, for_init, "charger"),
    "generic_select_dispatch": _generic_select,
    "generic_switch_dispatch": _generic_switch,
    "generic_number_dispatch": _generic_number,
    "generic_time_dispatch": _generic_time,
}


@pytest.mark.parametrize("for_init", [True, False], ids=["restore_replay", "user_action"])
@pytest.mark.parametrize("entity", list(REPLAYS))
async def test_entity_level_restore_replay_never_pins(entity, for_init):
    # R3-3: the replay marker is set once, at the shared restore dispatch of every
    # entity platform, whatever the setter behind it
    fx = _attached_fx()
    fx.home.force_update_all = AsyncMock()
    await REPLAYS[entity](fx, for_init)
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == (None if for_init else "A")


async def test_time_restore_replays_with_for_init():
    # R3-3: the time restore (`async_added_to_hass`) passes `for_init=True`
    from custom_components.quiet_solar.time import QSBaseTime, QSTimeEntityDescription

    fx = _attached_fx()
    desc = QSTimeEntityDescription(key="qs_bump_solar_charge_priority")
    ent = _entity(QSBaseTime, fx.cars["A"], desc)
    ent.async_get_last_state = AsyncMock(return_value=State("time.x", "07:30:00"))
    with patch("custom_components.quiet_solar.entity.QSDeviceEntity.async_added_to_hass", AsyncMock()):
        await ent.async_added_to_hass()
    assert fx.cars["A"].qs_bump_solar_charge_priority is True
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None


async def test_user_replay_context_never_clears_an_outer_replay():
    # R3-3: a nested `for_init=False` is a no-op, it does not end an outer replay
    from custom_components.quiet_solar.ha_model.car import _USER_REPLAY_FOR_INIT, user_replay_context

    assert _USER_REPLAY_FOR_INIT.get() is False
    with user_replay_context(True):
        with user_replay_context(False):
            assert _USER_REPLAY_FOR_INIT.get() is True
        assert _USER_REPLAY_FOR_INIT.get() is True
    assert _USER_REPLAY_FOR_INIT.get() is False


# ---- review fix #04 ------------------------------------------------------------


def _w1_unplugged(fx) -> None:
    fx.w1.is_not_plugged = MagicMock(return_value=True)
    fx.w1.is_plugged = MagicMock(return_value=False)


def _w1_plugged(fx) -> None:
    fx.w1.is_not_plugged = MagicMock(return_value=False)
    fx.w1.is_plugged = MagicMock(return_value=True)


@pytest.mark.parametrize("stale_off", [False, True], ids=["on_then_off", "stale_off"])
async def test_persisted_pin_guard_runs_while_the_charger_is_unplugged_at_boot(stale_off):
    # R4-1: W1 unplugged at boot never reaches `_compute_best_car` during the boot
    # window: the guard must also run on the unplugged branch, or the pin outlives
    # its proof (the next car plugged on W1 would be ignored for the absent A)
    if stale_off:
        data = _pinned_first_life(SAVED_AT, plug_at_save="off")
        fx = _pinned_reboot(data, [(-1200, "off")])
    else:
        data = _pinned_first_life(SAVED_AT, plug_at_save="on")
        fx = _pinned_reboot(data, [(-1200, "on")])
    a, b = fx.cars["A"], fx.cars["B"]
    _poll_plug(a, "off", T0 + timedelta(seconds=1))
    _set_car(a, sensor="off", dist=40.0)
    _w1_unplugged(fx)
    for i in range(5):
        await fx.w1.check_load_activity_and_constraints(T0 + timedelta(seconds=10 + 10 * i))
    fx.w1.reset_boot_data()

    if stale_off:
        # a re-polled stale `off` proves nothing: the pin is kept, as before
        assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"
        return
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None
    assert a.get_user_originated(USER_ORIGINATED_CHARGER_NAME) is None

    # hours later B plugs on W1 (on, synced, 1.25 m): B is attached, not the absent A
    _set_car(b, sensor="on", dist=1.25, plug_duration=SYNC_228)
    _w1_plugged(fx)
    await fx.w1.check_load_activity_and_constraints(T0 + timedelta(hours=5))
    assert fx.w1.car is b


async def test_charger_card_choice_right_after_a_physical_unplug_is_kept():
    # R4-2: the charger is instantly unplugged but not yet for the check window
    # (`_seen_plugged` still True); a choice made then is a pre-selection, not wiped
    # by the unplug edge the window confirms a few seconds later
    fx = _attached_fx()
    a, b = fx.cars["A"], fx.cars["B"]
    _drive_real_cycle_path([fx.w1], [a, b])
    assert fx.w1.car is a
    assert fx.w1._seen_plugged is True  # set by the attach

    def _not_plugged(time, for_duration=None):
        return True if for_duration is None else window_elapsed[0]

    def _plugged(time, for_duration=None):
        return False

    window_elapsed = [False]
    fx.w1.is_not_plugged = MagicMock(side_effect=_not_plugged)
    fx.w1.is_plugged = MagicMock(side_effect=_plugged)
    await fx.w1.user_set_selected_car_by_name("B", time=T0)
    # R5-2: the instant-unplug check uses the injected time, not the wall clock
    fx.w1.is_not_plugged.assert_any_call(T0)
    assert fx.w1.car is None

    window_elapsed[0] = True
    for i in range(1, 4):
        await fx.w1.check_load_activity_and_constraints(T0 + timedelta(seconds=7 * i))
        assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "B"

    _w1_plugged(fx)
    await fx.w1.check_load_activity_and_constraints(T0 + timedelta(seconds=60))
    assert fx.w1.car is b


async def test_charger_card_choice_while_still_plugged_keeps_the_unplug_edge():
    # R4-2 guard: a choice made while the charger is plugged is still cleared by the
    # next physical unplug
    fx = _attached_fx()
    a, b = fx.cars["A"], fx.cars["B"]
    _drive_real_cycle_path([fx.w1], [a, b])
    _w1_plugged(fx)
    await fx.w1.user_set_selected_car_by_name("B", time=T0)
    fx.w1.is_not_plugged.assert_any_call(T0)
    assert fx.w1._seen_plugged is True
    _w1_unplugged(fx)
    await fx.w1.check_load_activity_and_constraints(T0 + timedelta(seconds=7))
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) is None


def test_pin_save_without_a_home_writes_no_plug_reading():
    # R4-4: the persistence hook never dereferences a missing home
    fx = _fx(car_names=("A",), with_plug_sensor=True)
    fx.w1.attach_car(fx.cars["A"], T0 - timedelta(minutes=5))
    fx.w1.set_user_originated(USER_ORIGINATED_CAR_NAME, "A")
    fx.w1.home = None
    d: dict = {}
    fx.w1.update_to_be_saved_extra_device_info(d, T0)
    assert d[STORAGE_KEY_USER_PIN_SAVED_AT] == T0.isoformat()
    assert d[STORAGE_KEY_USER_PIN_PLUG_STATE] is None


def _unplugged(w) -> None:
    w.is_not_plugged = MagicMock(return_value=True)
    w.is_plugged = MagicMock(return_value=False)


async def test_preselection_without_restart_survives_car_unplug_elsewhere():
    # R5-1 contrast: a pre-selection on an unplugged W1 (A away) is kept when A is
    # unplugged elsewhere — no boot guard involved
    fx = _fx(car_names=("A", "B"), with_plug_sensor=True)
    a, b = fx.cars["A"], fx.cars["B"]
    _drive_real_cycle_path([fx.w1], [a, b])
    _unplugged(fx.w1)
    await fx.w1.user_set_selected_car_by_name("A", time=T0)
    _set_car(a, sensor="off", dist=40000.0)
    for i in range(5):
        await fx.w1.check_load_activity_and_constraints(T0 + timedelta(seconds=10 + 10 * i))
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"


async def test_preselection_across_restart_kept_when_car_unplugged_elsewhere():
    # R5-1: a pre-selection (A pinned on the unplugged W1, never attached there) is
    # persisted WITHOUT a pin stamp, so the restart arms no boot guard and an
    # on -> off of A's plug sensor elsewhere does not lift it (same as no restart)
    fx = _fx(car_names=("A", "B"), with_plug_sensor=True)
    a, b = fx.cars["A"], fx.cars["B"]
    _drive_real_cycle_path([fx.w1], [a, b])
    a._add_state_history(a.car_plugged, "on", SAVED_AT - timedelta(seconds=5), None, {})
    _unplugged(fx.w1)
    await fx.w1.user_set_selected_car_by_name("A", time=SAVED_AT)
    assert fx.w1.car is None
    data: dict = {}
    fx.w1.update_to_be_saved_extra_device_info(data, SAVED_AT)
    assert data[STORAGE_KEY_USER_PIN_SAVED_AT] is None
    assert data[STORAGE_KEY_USER_PIN_PLUG_STATE] is None

    fx = _fx(car_names=("A", "B"), with_plug_sensor=True)
    a, b = fx.cars["A"], fx.cars["B"]
    _set_car(a, sensor="unknown", dist=40000.0)
    _set_car(b, sensor="off", dist=1.25)
    _history(a, [(-1200, "on")])
    _drive_real_cycle_path([fx.w1], [a, b])
    fx.w1.use_saved_extra_device_info(copy.deepcopy(data))
    fx.w1.device_post_home_init(T0)
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"
    assert fx.w1._boot_user_pin is None  # no guard armed for a pre-selection
    _poll_plug(a, "off", T0 + timedelta(seconds=1))
    _set_car(a, sensor="off", dist=40000.0)
    _unplugged(fx.w1)
    for i in range(5):
        await fx.w1.check_load_activity_and_constraints(T0 + timedelta(seconds=10 + 10 * i))
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"


async def test_charger_card_choice_without_a_home_skips_the_other_chargers():
    # R5-3: the R2-7 "drop this car's pin elsewhere" loop never dereferences a
    # missing home (mirror of the car-side loop)
    fx = _fx(car_names=("A",))
    fx.w1.home = None
    fx.w1.update_charger_for_user_change = AsyncMock()
    await fx.w1.user_set_selected_car_by_name("A", time=T0)
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"
    fx.w1.update_charger_for_user_change.assert_awaited_once()


async def test_charger_card_choice_defaults_to_the_wall_clock():
    # R5-2: the entity entry point passes no time; the method resolves "now"
    fx = _fx(car_names=("A",))
    fx.w1.update_charger_for_user_change = AsyncMock()
    fx.w1.is_not_plugged = MagicMock(return_value=False)
    await fx.w1.user_set_selected_car_by_name("A")
    (called_time,), _ = fx.w1.is_not_plugged.call_args
    assert called_time.tzinfo is not None
