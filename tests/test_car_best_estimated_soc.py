"""Tests for the live best-estimated SOC display model (Story QS-281).

Covers the pure-read accessor `get_best_estimated_car_charge_percent` (every
AC2/AC7 branch), and the per-cycle re-anchor refactor of
`_capture_last_valid_base_soc`: genuine-change up/down, same-integer heartbeat,
`None`-raw no-op, manual-override guard, idle hold, and the stale→healthy
recovery transition.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from custom_components.quiet_solar.const import CONF_CAR_CHARGE_PERCENT_SENSOR
from custom_components.quiet_solar.ha_model.car import QSCar, _finite_soc_or_none


def _make_car(fake_hass, mock_data_handler, **overrides) -> QSCar:
    # A REAL `QSCar` domain object built from the standard `MOCK_CAR_CONFIG`
    # (no MagicMock for the domain object, per project rules). Only
    # `config_entry` is a MagicMock — it is genuine HA infrastructure (a config
    # entry handle), not a domain object, and the same convention is used by the
    # sibling `test_car_soc_estimation.py` fixtures.
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
def est_car(fake_hass, mock_data_handler, current_time):
    """A normal car: SOC sensor + battery capacity, not invited."""
    return _make_car(fake_hass, mock_data_handler)


@pytest.fixture
def est_car_no_sensor(fake_hass, mock_data_handler, current_time):
    """A real car with a true capacity but no SOC sensor."""
    return _make_car(fake_hass, mock_data_handler, **{CONF_CAR_CHARGE_PERCENT_SENSOR: None})


def _set_soc(car: QSCar, value, time):
    car._entity_probed_last_valid_state[car.car_charge_percent_sensor] = (time, value, {})


# ── get_best_estimated_car_charge_percent — AC2 / AC7 branches ───────────────


def test_best_estimate_estimation_mode_returns_estimate(est_car, current_time):
    """Branch 1 — while estimating, parity with `get_car_charge_percent`."""
    est_car.car_api_stale_percent_mode = True  # → estimating
    est_car._last_valid_base_soc_value = 40.0
    est_car._computed_added_delta_soc_percent = 7.0
    assert est_car.is_in_soc_estimation_mode(current_time) is True
    assert est_car.get_best_estimated_car_charge_percent(current_time) == pytest.approx(47.0)
    assert est_car.get_best_estimated_car_charge_percent(current_time) == est_car.get_car_charge_percent(current_time)


def test_best_estimate_estimation_mode_pure_delta_none(est_car_no_sensor, current_time):
    """Branch 1, pure-delta — no base while estimating → None (card falls back)."""
    assert est_car_no_sensor.is_in_soc_estimation_mode(current_time) is True
    est_car_no_sensor._last_valid_base_soc_value = None
    est_car_no_sensor._computed_added_delta_soc_percent = 5.0
    assert est_car_no_sensor.get_best_estimated_car_charge_percent(current_time) is None


def test_best_estimate_base_plus_delta_when_not_estimating(est_car, current_time):
    """Branch 2 — healthy sensor, a base exists → clamp(base + delta)."""
    _set_soc(est_car, 45.0, current_time)
    assert est_car.is_in_soc_estimation_mode(current_time) is False
    est_car._last_valid_base_soc_value = 45.0
    est_car._computed_added_delta_soc_percent = 2.5
    assert est_car.get_best_estimated_car_charge_percent(current_time) == pytest.approx(47.5)


def test_best_estimate_clamped_at_100(est_car, current_time):
    """Branch 2 — base + delta clamps to 100 (reuses `_estimated_soc_percent`)."""
    _set_soc(est_car, 98.0, current_time)
    assert est_car.is_in_soc_estimation_mode(current_time) is False  # pin branch 2, not 1
    est_car._last_valid_base_soc_value = 98.0
    est_car._computed_added_delta_soc_percent = 10.0
    assert est_car.get_best_estimated_car_charge_percent(current_time) == 100.0


def test_best_estimate_never_below_raw_on_sub_integer_increase(est_car, current_time):
    """fix-plan #04 #02 — base re-anchored at 45.0 (delta 0); a genuine slow-API
    advance to 45.9 is swallowed by the integer de-bounce (no re-anchor), but
    the display accessor must clamp `>= raw` so the estimate never lags the live
    API (and no behind-the-API `*`)."""
    est_car._last_valid_base_soc_value = 45.0
    est_car._computed_added_delta_soc_percent = 0.0
    _set_soc(est_car, 45.9, current_time)
    # the per-cycle re-anchor does NOT fire: int(45.9) == int(45.0)
    est_car._capture_last_valid_base_soc(current_time)
    assert est_car._last_valid_base_soc_value == 45.0
    # …but the display accessor clamps up to the genuine raw value
    best = est_car.get_best_estimated_car_charge_percent(current_time)
    assert best == 45.9
    assert best >= est_car.get_car_charge_percent_raw_sensor(current_time)


def test_best_estimate_ahead_of_raw_when_api_lags(est_car, current_time):
    """The complement of the clamp: when the estimate genuinely LEADS the
    (lagging) raw API, the estimate stands (no down-clamp to raw)."""
    est_car._last_valid_base_soc_value = 45.0
    est_car._computed_added_delta_soc_percent = 2.5  # charge added while API stuck
    _set_soc(est_car, 45.0, current_time)
    assert est_car.is_in_soc_estimation_mode(current_time) is False
    assert est_car.get_best_estimated_car_charge_percent(current_time) == pytest.approx(47.5)


def test_best_estimate_base_with_non_numeric_raw_returns_estimate(est_car, current_time):
    """Branch 2 with a non-numeric/unavailable raw → the estimate stands (the
    raw clamp is skipped and no `str` leaks into the comparison)."""
    _set_soc(est_car, "unknown", current_time)
    est_car._last_valid_base_soc_value = 50.0
    est_car._computed_added_delta_soc_percent = 4.0
    assert est_car.is_in_soc_estimation_mode(current_time) is False
    assert est_car.get_best_estimated_car_charge_percent(current_time) == pytest.approx(54.0)


def test_best_estimate_no_base_returns_raw(est_car, current_time):
    """Branch 3 — healthy sensor, no base → the plain raw sensor value."""
    _set_soc(est_car, 53.0, current_time)
    est_car._last_valid_base_soc_value = None
    assert est_car.get_best_estimated_car_charge_percent(current_time) == 53.0


def test_best_estimate_no_base_str_raw_returns_none(est_car, current_time):
    """Branch 3 — fix-plan #04 #05: a non-numeric raw live read is sanitized to
    `None`, so the BATTERY/MEASUREMENT sensor never receives a `str`."""
    _set_soc(est_car, "unavailable", current_time)
    est_car._last_valid_base_soc_value = None
    assert est_car.get_best_estimated_car_charge_percent(current_time) is None


def test_best_estimate_no_base_raw_none(est_car, current_time):
    """Branch 3 — no base and the raw sensor itself is None → None."""
    est_car._entity_probed_last_valid_state[est_car.car_charge_percent_sensor] = None
    est_car._last_valid_base_soc_value = None
    assert est_car.get_best_estimated_car_charge_percent(current_time) is None


def test_best_estimate_time_none_resolves_to_now(est_car):
    """`time=None` resolves to now inside the accessor (no exception, returns raw)."""
    import pytz

    now = __import__("datetime").datetime.now(tz=pytz.UTC)
    _set_soc(est_car, 61.0, now)
    est_car._last_valid_base_soc_value = None
    assert est_car.get_best_estimated_car_charge_percent() == 61.0


def test_best_estimate_is_a_pure_read(est_car, current_time):
    """AC2 — the accessor must NOT mutate any estimate field."""
    _set_soc(est_car, 50.0, current_time)
    est_car._last_valid_base_soc_value = 50.0
    est_car._computed_added_delta_soc_percent = 3.0
    est_car._delta_soc_last_integration_time = current_time
    snapshot = (
        est_car._last_valid_base_soc_value,
        est_car._computed_added_delta_soc_percent,
        est_car._delta_soc_last_integration_time,
        est_car._user_base_soc_value,
    )
    est_car.get_best_estimated_car_charge_percent(current_time)
    assert (
        est_car._last_valid_base_soc_value,
        est_car._computed_added_delta_soc_percent,
        est_car._delta_soc_last_integration_time,
        est_car._user_base_soc_value,
    ) == snapshot


# ── per-cycle re-anchor (_capture_last_valid_base_soc) ───────────────────────


def test_reanchor_genuine_change_up(est_car, current_time):
    """A genuinely higher raw value re-anchors the base and resets the delta."""
    est_car._last_valid_base_soc_value = 40.0
    est_car._computed_added_delta_soc_percent = 6.0
    est_car._delta_soc_last_integration_time = current_time
    _set_soc(est_car, 45.0, current_time)
    est_car._capture_last_valid_base_soc(current_time)
    assert est_car._last_valid_base_soc_value == 45.0
    assert est_car._computed_added_delta_soc_percent == 0.0
    assert est_car._delta_soc_last_integration_time is None


def test_reanchor_genuine_change_down(est_car, current_time):
    """A genuine API change is trusted and may move the value DOWN (estimate ran
    ahead, API corrects it)."""
    est_car._last_valid_base_soc_value = 47.0
    est_car._computed_added_delta_soc_percent = 3.0
    _set_soc(est_car, 45.0, current_time)
    est_car._capture_last_valid_base_soc(current_time)
    assert est_car._last_valid_base_soc_value == 45.0
    assert est_car._computed_added_delta_soc_percent == 0.0


def test_reanchor_same_integer_heartbeat_no_reset(est_car, current_time):
    """A same-integer reading (float noise / heartbeat) must NOT re-anchor."""
    est_car._last_valid_base_soc_value = 50.0
    est_car._computed_added_delta_soc_percent = 4.0
    est_car._delta_soc_last_integration_time = current_time
    _set_soc(est_car, 50.4, current_time)  # int(50.4) == int(50.0)
    est_car._capture_last_valid_base_soc(current_time)
    assert est_car._last_valid_base_soc_value == 50.0
    assert est_car._computed_added_delta_soc_percent == 4.0
    assert est_car._delta_soc_last_integration_time == current_time


def test_reanchor_first_base_from_none(est_car, current_time):
    """With no base yet, the first raw reading sets the base (delta cleared)."""
    est_car._last_valid_base_soc_value = None
    est_car._computed_added_delta_soc_percent = 9.0
    _set_soc(est_car, 33.0, current_time)
    est_car._capture_last_valid_base_soc(current_time)
    assert est_car._last_valid_base_soc_value == 33.0
    assert est_car._computed_added_delta_soc_percent == 0.0


def test_reanchor_noop_when_raw_none(est_car, current_time):
    """No reading yet → no-op: base stays, delta untouched (AC3)."""
    est_car._last_valid_base_soc_value = 40.0
    est_car._computed_added_delta_soc_percent = 6.0
    est_car._entity_probed_last_valid_state[est_car.car_charge_percent_sensor] = None
    est_car._capture_last_valid_base_soc(current_time)
    assert est_car._last_valid_base_soc_value == 40.0
    assert est_car._computed_added_delta_soc_percent == 6.0


@pytest.mark.parametrize("bad_raw", [float("nan"), float("inf"), float("-inf")])
def test_reanchor_noop_when_raw_non_finite(est_car, current_time, bad_raw):
    """A non-finite raw (`nan`/`inf`) must be a no-op, never `int()`-crash the
    per-cycle re-anchor (must-fix #01)."""
    est_car._last_valid_base_soc_value = 40.0
    est_car._computed_added_delta_soc_percent = 6.0
    _set_soc(est_car, bad_raw, current_time)
    # Must not raise (int(nan) → ValueError, int(inf) → OverflowError).
    est_car._capture_last_valid_base_soc(current_time)
    assert est_car._last_valid_base_soc_value == 40.0
    assert est_car._computed_added_delta_soc_percent == 6.0


def test_reanchor_noop_when_raw_non_numeric_string(est_car, current_time):
    """A non-numeric sensor state (raw is typed `str | float | None`) is a
    no-op — `int("foo")` must never crash the per-cycle re-anchor."""
    est_car._last_valid_base_soc_value = 40.0
    est_car._computed_added_delta_soc_percent = 6.0
    _set_soc(est_car, "unavailable", current_time)
    est_car._capture_last_valid_base_soc(current_time)
    assert est_car._last_valid_base_soc_value == 40.0
    assert est_car._computed_added_delta_soc_percent == 6.0


def test_reanchor_recovers_when_base_non_finite(est_car, current_time):
    """A previously-stored non-finite base re-anchors to the next good raw
    rather than crashing the compare."""
    est_car._last_valid_base_soc_value = float("nan")
    est_car._computed_added_delta_soc_percent = 3.0
    _set_soc(est_car, 50.0, current_time)
    est_car._capture_last_valid_base_soc(current_time)
    assert est_car._last_valid_base_soc_value == 50.0
    assert est_car._computed_added_delta_soc_percent == 0.0


def test_reanchor_skips_when_user_override(est_car, current_time):
    """Manual override owns its delta — the re-anchor must NOT zero it."""
    est_car._user_base_soc_value = 70.0
    est_car._last_valid_base_soc_value = 40.0
    est_car._computed_added_delta_soc_percent = 6.0
    est_car._delta_soc_last_integration_time = current_time
    _set_soc(est_car, 55.0, current_time)  # would re-anchor if unguarded
    est_car._capture_last_valid_base_soc(current_time)
    assert est_car._last_valid_base_soc_value == 40.0
    assert est_car._computed_added_delta_soc_percent == 6.0
    assert est_car._delta_soc_last_integration_time == current_time


def test_reanchor_idle_holds_value(est_car, current_time):
    """Idle (no charge → delta does not grow): repeated equal readings hold the
    base + delta unchanged; only the charger callback grows the delta."""
    est_car._last_valid_base_soc_value = 60.0
    est_car._computed_added_delta_soc_percent = 5.0  # accrued during the prior charge
    _set_soc(est_car, 60.0, current_time)
    for _ in range(3):
        est_car._capture_last_valid_base_soc(current_time)
    # held: no re-anchor (same int), and the re-anchor never grows the delta
    assert est_car._last_valid_base_soc_value == 60.0
    assert est_car._computed_added_delta_soc_percent == 5.0
    assert est_car.get_best_estimated_car_charge_percent(current_time) == pytest.approx(65.0)


def test_reanchor_stale_to_healthy_recovery(est_car, current_time):
    """Stale→healthy: a retained delta survives until the first DIFFERENT
    healthy reading re-anchors; an equal reading does not (AC3a)."""
    est_car._last_valid_base_soc_value = 60.0
    est_car._computed_added_delta_soc_percent = 4.0
    # equal reading on recovery → delta survives
    _set_soc(est_car, 60.0, current_time)
    est_car._capture_last_valid_base_soc(current_time)
    assert est_car._computed_added_delta_soc_percent == 4.0
    # first genuinely different healthy reading → re-anchor
    later = current_time + timedelta(seconds=30)
    _set_soc(est_car, 63.0, later)
    est_car._capture_last_valid_base_soc(later)
    assert est_car._last_valid_base_soc_value == 63.0
    assert est_car._computed_added_delta_soc_percent == 0.0


# ── Legacy/corrupt persisted-base sanitization (fix-plan #03 #01) ─────────────


@pytest.mark.parametrize(
    "value,expected",
    [
        (55.0, 55.0),
        (40, 40.0),
        (None, None),
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
        ("55.0", None),  # numeric-looking string is still rejected
        ("unavailable", None),
        (True, None),  # bool is not a valid SOC numeric
    ],
)
def test_finite_soc_or_none(value, expected):
    """`_finite_soc_or_none` keeps only finite real numbers, else `None`."""
    assert _finite_soc_or_none(value) == expected


@pytest.mark.parametrize("bad_base", [float("nan"), float("inf"), "corrupt", "55.0"])
def test_load_sanitizes_non_finite_persisted_base(est_car, current_time, bad_base):
    """A legacy/corrupt persisted base is coerced to `None` on load, so the
    healthy-path accessor can never crash or emit nan/inf (fix-plan #03 #01)."""
    est_car.use_saved_extra_device_info(
        {
            "last_valid_base_soc_value": bad_base,
            "computed_added_delta_soc_percent": float("inf"),
            "user_base_soc_value": "bad",
            "user_base_soc_entry_sensor_value": float("nan"),
        }
    )
    assert est_car._last_valid_base_soc_value is None
    assert est_car._computed_added_delta_soc_percent is None
    assert est_car._user_base_soc_value is None
    assert est_car._user_base_soc_entry_sensor_value is None
    # The accessor (and thus the sensor value_fn) must not raise — a healthy raw
    # sensor is present, so it falls back to the raw value.
    _set_soc(est_car, 53.0, current_time)
    assert est_car.get_best_estimated_car_charge_percent(current_time) == 53.0


def test_reset_soc_estimate_clears_unified_base_accumulator_cursor(est_car):
    """AC6 — `reset_soc_estimate` zeroes the unified base + accumulator + cursor
    in one shot (fix-plan #03 #02 traceability). The "persisted base survives a
    reboot re-attach" half of AC6 is covered by the QS-243 serialize round-trip
    test (`test_persistence_round_trip` in `test_car_soc_estimation.py`)."""
    est_car._last_valid_base_soc_value = 47.0
    est_car._computed_added_delta_soc_percent = 5.0
    est_car._delta_soc_last_integration_time = "cursor"
    est_car.reset_soc_estimate()
    assert est_car._last_valid_base_soc_value is None
    assert est_car._computed_added_delta_soc_percent is None
    assert est_car._delta_soc_last_integration_time is None
