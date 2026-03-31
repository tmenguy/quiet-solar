"""Tests for Bug #86: Numpy type serialization.

Covers:
- serialize_stored_values() produces native Python types (AC1, AC2, AC3)
- push_and_get() stores native Python types (AC2)
- compute_now_forecast() returns native Python types (AC1)
- Values are numerically identical after conversion (AC5)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytz

from custom_components.quiet_solar.ha_model.home import QSforecastValueSensor


# ============================================================================
# F1: serialize_stored_values() — safety net at serialization boundary
# ============================================================================


def test_serialize_stored_values_converts_numpy_int():
    """serialize_stored_values converts numpy.int32 values to native Python int."""
    prober = QSforecastValueSensor("test", 3600, lambda t: (t, 100.0))
    t1 = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    t2 = datetime(2026, 3, 30, 13, 0, 0, tzinfo=pytz.UTC)
    # Inject numpy scalars directly into _stored_values
    prober._stored_values = [(t1, np.int32(500)), (t2, np.int32(750))]

    result = prober.serialize_stored_values()

    assert len(result) == 2
    # Values must be native Python types, not numpy
    assert type(result[0][1]) is int
    assert type(result[1][1]) is int
    assert result[0][1] == 500
    assert result[1][1] == 750


def test_serialize_stored_values_converts_numpy_float():
    """serialize_stored_values converts numpy.float64 values to native Python float."""
    prober = QSforecastValueSensor("test", 3600, lambda t: (t, 100.0))
    t1 = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    prober._stored_values = [(t1, np.float64(123.456))]

    result = prober.serialize_stored_values()

    assert type(result[0][1]) is float
    assert result[0][1] == 123.456


def test_serialize_stored_values_preserves_native_types():
    """serialize_stored_values passes through native Python types unchanged."""
    prober = QSforecastValueSensor("test", 3600, lambda t: (t, 100.0))
    t1 = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    prober._stored_values = [(t1, 500), (t1, 42.5)]

    result = prober.serialize_stored_values()

    assert type(result[0][1]) is int
    assert type(result[1][1]) is float


def test_serialize_stored_values_round_trip_with_numpy():
    """Serialize numpy values, restore, values are numerically identical."""
    prober = QSforecastValueSensor("test", 3600, lambda t: (t, 100.0))
    t1 = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    t2 = datetime(2026, 3, 30, 13, 0, 0, tzinfo=pytz.UTC)
    prober._stored_values = [(t1, np.int32(466)), (t2, np.int32(347))]

    serialized = prober.serialize_stored_values()

    new_prober = QSforecastValueSensor("test", 3600, lambda t: (t, 100.0))
    new_prober.restore_stored_values(serialized)

    assert len(new_prober._stored_values) == 2
    assert new_prober._stored_values[0][1] == 466.0
    assert new_prober._stored_values[1][1] == 347.0


# ============================================================================
# F3: push_and_get() — converts values entering _stored_values
# ============================================================================


def test_push_and_get_converts_numpy_forecast_values():
    """push_and_get stores native Python float, not numpy scalar, in _stored_values."""
    # Getter returns a numpy scalar value
    def numpy_getter(t):
        return (t, np.int32(265))

    prober = QSforecastValueSensor("test", 3600, numpy_getter)
    time = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)

    prober.push_and_get(time)

    # Check that stored values contain native Python types
    assert len(prober._stored_values) > 0
    for _, val in prober._stored_values:
        assert not isinstance(val, (np.integer, np.floating)), (
            f"Expected native Python type, got {type(val)}"
        )


def test_push_and_get_returns_native_type():
    """push_and_get returns native Python float, not numpy scalar."""
    call_count = 0

    def numpy_getter(t):
        nonlocal call_count
        call_count += 1
        return (t, np.int32(500))

    prober = QSforecastValueSensor("test", 0, None, current_getter=lambda t: (t, np.float64(42.5)))
    time = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)

    result = prober.push_and_get(time)

    assert result is not None
    assert not isinstance(result, (np.integer, np.floating)), (
        f"Expected native Python type, got {type(result)}"
    )


# ============================================================================
# F2: compute_now_forecast() — converts on forecast list construction
# ============================================================================


def test_compute_now_forecast_returns_native_types():
    """compute_now_forecast returns forecast tuples with native Python int values."""
    from unittest.mock import MagicMock, patch

    from custom_components.quiet_solar.ha_model.home import (
        BUFFER_SIZE_IN_INTERVALS,
        INTERVALS_MN,
        NUM_INTERVAL_PER_HOUR,
        QSSolarHistoryVals,
    )

    forecast_obj = QSSolarHistoryVals.__new__(QSSolarHistoryVals)
    forecast_obj.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)
    forecast_obj._current_forecast = None
    forecast_obj._last_forecast_update_time = None
    forecast_obj.home = None

    time_now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    now_idx, now_days = forecast_obj.get_index_from_time(time_now)

    # Fill values for 48 hours of history
    num_intervals = 48 * NUM_INTERVAL_PER_HOUR
    for i in range(num_intervals):
        idx = (now_idx - num_intervals + i) % BUFFER_SIZE_IN_INTERVALS
        forecast_obj.values[0][idx] = np.int32(100 + i)
        forecast_obj.values[1][idx] = np.int32(now_days)

    # Mock _get_possible_now_val_for_forcast to return a numpy scalar
    forecast_obj._get_possible_now_val_for_forcast = lambda t: np.int32(200)
    # Mock _get_possible_past_consumption_for_forecast to return scores pointing to our data
    forecast_obj._get_possible_past_consumption_for_forecast = lambda *args, **kwargs: [(num_intervals, 1.0)]

    result = forecast_obj.compute_now_forecast(time_now, 24, 24)

    assert len(result) > 0
    for time_val, power_val in result:
        assert isinstance(time_val, datetime), f"Expected datetime, got {type(time_val)}"
        assert not isinstance(power_val, (np.integer, np.floating)), (
            f"Expected native Python type for power value, got {type(power_val)}"
        )
