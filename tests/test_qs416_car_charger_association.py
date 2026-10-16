"""QS-416 — car↔charger association.

- D1: score tiers rebalanced (proximity ≤ 10 m, then plug sensor, then sync_ok,
  then fine distance, then fine sync). The ≤ 3 m bonus is a REPLACEMENT plug bonus.
- D2: the absolute 1-hour lock is replaced by a persisted per-attach floor.
- D3: a user action on the attached car pins the car↔charger association.

Built on `tests.utils.charger_harness` (real `QSChargerGeneric` / `QSCar`), on the
model of `tests/test_charger_allocation_tiebreak.py`.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timedelta
from datetime import time as dt_time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from custom_components.quiet_solar.const import (
    CHARGE_TIME_CONSTRAINTS_CLEARED,
    CONF_CHARGER_LATITUDE,
    CONF_CHARGER_LONGITUDE,
    FORCE_CAR_NO_CHARGER_CONNECTED,
    STORAGE_KEY_ATTACHED_CAR_BEST_SCORE,
    STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_CAR_NAME,
    STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_SAVED_AT,
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


def _make_car(hass, home, name, with_plug_sensor=False):
    car = make_real_car(hass, home, name=name)
    if with_plug_sensor:
        car.car_plugged = f"binary_sensor.{name.lower()}_plugged"
        car.attach_ha_state_to_probe(car.car_plugged, is_numerical=False)
        car._entity_probed_state[car.car_plugged] = []
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


def test_attached_car_keeps_best_score_when_sensor_drops(caplog):
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    fx = _fx(car_names=("A", "B"))
    a, b = fx.cars["A"], fx.cars["B"]
    _set_car(a, sensor="on", dist=6.75, plug_duration=SYNC_228)
    _set_car(b, sensor="off", dist=40.0)
    fx.w1.attach_car(a, T0 - timedelta(minutes=30))
    assert fx.w1.get_car_score(a, T0 - timedelta(seconds=7), {}) == F_NIGHT

    # night: A's API falls asleep, B (plugged for hours) is "on" at 2.25 m
    _set_car(a, sensor="unknown", dist=6.75)
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
        fx.w1.set_user_originated(USER_ORIGINATED_CAR_NAME, "B")
        assert fx.w1.get_car_score(a, T0, {}) == 0
    else:
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


def _first_life(shutdown_s=600.0):
    """Charger W1 with A attached and floored at F_NIGHT; return the saved payload."""
    fx = _fx(car_names=("A", "B"), with_plug_sensor=True)
    a = fx.cars["A"]
    _set_car(a, sensor="on", dist=6.75, plug_duration=SYNC_228)
    fx.w1.attach_car(a, T0 - timedelta(hours=1))
    assert fx.w1.get_car_score(a, T0 - timedelta(minutes=20), {}) == F_NIGHT
    data: dict = {}
    fx.w1.update_to_be_saved_extra_device_info(data)
    assert data[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_CAR_NAME] == "A"
    assert data[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE] == F_NIGHT
    data[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_SAVED_AT] = (T0 - timedelta(seconds=shutdown_s)).isoformat()
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


def test_floor_pending_value_rewritten_and_survives_second_reboot():
    data = _first_life()
    saved_at = data[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_SAVED_AT]

    # save between use_saved_extra_device_info and device_post_home_init
    fx = _reboot(data, post_init=False)
    d1: dict = {}
    fx.w1.update_to_be_saved_extra_device_info(d1)
    assert d1[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_CAR_NAME] == "A"
    assert d1[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE] == F_NIGHT
    assert d1[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_SAVED_AT] == saved_at

    # save in the boot window, before any attach
    fx.w1.device_post_home_init(T0 - timedelta(seconds=60))
    d2: dict = {}
    fx.w1.update_to_be_saved_extra_device_info(d2)
    assert d2[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_SAVED_AT] == saved_at

    # 2nd restart inside the boot window: still restorable
    fx2 = _reboot(copy.deepcopy(d2))
    fx2.w1.attach_car(fx2.cars["A"], T0)
    assert fx2.w1._attached_car_best_score == F_NIGHT


def test_floor_saved_as_none_after_unplug():
    data = _first_life()
    fx = _reboot(data)
    fx.w1.attach_car(fx.cars["A"], T0)
    assert fx.w1._attached_car_best_score == F_NIGHT
    fx.w1.reset(keep_commands=True)
    d: dict = {}
    fx.w1.update_to_be_saved_extra_device_info(d)
    assert d[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_CAR_NAME] is None
    assert d[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE] is None
    assert d[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_SAVED_AT] is None


def test_floor_saved_with_current_clock_while_attached():
    fx = _fx(car_names=("A",))
    a = fx.cars["A"]
    _set_car(a, sensor="on", dist=6.75, plug_duration=SYNC_228)
    fx.w1.attach_car(a, T0 - timedelta(hours=1))
    d: dict = {}
    # attached but never scored: no floor yet → nothing to save
    fx.w1.update_to_be_saved_extra_device_info(d)
    assert d[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE] is None

    fx.w1.get_car_score(a, T0, {})
    before = datetime.now(tz=pytz.UTC)
    fx.w1.update_to_be_saved_extra_device_info(d)
    saved_at = datetime.fromisoformat(d[STORAGE_KEY_ATTACHED_CAR_BEST_SCORE_SAVED_AT])
    assert saved_at.tzinfo is not None
    assert saved_at >= before


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
    await a.user_set_next_charge_target(70)
    assert fx.w1.get_user_originated(USER_ORIGINATED_CAR_NAME) == "A"

    if lift == "unplug_cycle":
        _drive_real_cycle_path([fx.w1], list(fx.cars.values()))
        fx.w1.is_not_plugged = MagicMock(return_value=True)
        fx.w1.is_plugged = MagicMock(return_value=False)
        await fx.w1.check_load_activity_and_constraints(T0)
        assert fx.w1.car is None
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
