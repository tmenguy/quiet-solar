"""Thorough tests for time series utility functions in home_utils.py."""

from datetime import UTC, datetime, timedelta

import pytest

from custom_components.quiet_solar.home_model.home_utils import (
    align_time_series_and_values,
    align_time_series_on_time_slots,
    get_slots_from_time_series,
    get_value_from_time_series,
    slot_value_from_time_series,
)

# Also verify backward-compatible import paths
from custom_components.quiet_solar.home_model.load import (
    align_time_series_and_values as load_align,
)
from custom_components.quiet_solar.home_model.load import (
    get_slots_from_time_series as load_get_slots,
)
from custom_components.quiet_solar.home_model.load import (
    get_value_from_time_series as load_get_value,
)

T0 = datetime(2025, 6, 15, 8, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# slot_value_from_time_series
# ---------------------------------------------------------------------------
class TestSlotValueFromTimeSeries:
    def test_empty_forecast(self):
        last, val = slot_value_from_time_series([], T0, T0 + timedelta(minutes=30), -1)
        assert last == -1
        assert val == 0.0

    def test_single_point(self):
        fc = [(T0, 100.0)]
        last, val = slot_value_from_time_series(fc, T0, T0 + timedelta(minutes=30), -1)
        assert val == 100.0

    def test_exact_boundaries(self):
        fc = [
            (T0, 100.0),
            (T0 + timedelta(minutes=30), 200.0),
            (T0 + timedelta(minutes=60), 300.0),
        ]
        # Slot from T0 to T0+30m: step-and-hold averaging gives 100 for entire period
        last, val = slot_value_from_time_series(fc, T0, T0 + timedelta(minutes=30), -1)
        assert last == 1
        assert val == pytest.approx(100.0)

    def test_geometric_smoothing(self):
        fc = [
            (T0, 100.0),
            (T0 + timedelta(minutes=60), 200.0),
        ]
        # With geometric smoothing, boundary interpolation applies
        last, val = slot_value_from_time_series(
            fc, T0 + timedelta(minutes=15), T0 + timedelta(minutes=45), -1, geometric_smoothing=True
        )
        assert val > 0.0

    def test_gap_in_forecast(self):
        fc = [
            (T0, 100.0),
            (T0 + timedelta(hours=2), 300.0),
        ]
        # Slot that falls in a gap
        last, val = slot_value_from_time_series(fc, T0 + timedelta(minutes=30), T0 + timedelta(minutes=60), -1)
        assert val >= 0.0

    def test_forecast_ends_within_slot(self):
        fc = [
            (T0, 100.0),
            (T0 + timedelta(minutes=15), 200.0),
        ]
        # Slot extends past forecast end
        last, val = slot_value_from_time_series(fc, T0, T0 + timedelta(minutes=30), -1)
        assert val > 0.0

    def test_sequential_slots(self):
        fc = [
            (T0, 100.0),
            (T0 + timedelta(minutes=30), 200.0),
            (T0 + timedelta(minutes=60), 300.0),
        ]
        last = -1
        values = []
        for i in range(2):
            begin = T0 + timedelta(minutes=30 * i)
            end = T0 + timedelta(minutes=30 * (i + 1))
            last, val = slot_value_from_time_series(fc, begin, end, last)
            values.append(val)
        assert len(values) == 2
        assert all(v > 0 for v in values)

    def test_prev_end_before_begin_with_interpolation(self):
        fc = [
            (T0, 100.0),
            (T0 + timedelta(minutes=60), 200.0),
        ]
        # prev_end at 0 (T0), slot starts at T0+15
        last, val = slot_value_from_time_series(fc, T0 + timedelta(minutes=15), T0 + timedelta(minutes=45), 0)
        assert val > 0.0

    def test_prev_end_before_begin_with_geometric(self):
        fc = [
            (T0, 100.0),
            (T0 + timedelta(minutes=60), 200.0),
        ]
        last, val = slot_value_from_time_series(
            fc, T0 + timedelta(minutes=15), T0 + timedelta(minutes=45), 0, geometric_smoothing=True
        )
        assert val > 0.0


# ---------------------------------------------------------------------------
# align_time_series_on_time_slots
# ---------------------------------------------------------------------------
class TestAlignTimeSeriesOnTimeSlots:
    def test_empty_series(self):
        result = align_time_series_on_time_slots([], [T0, T0 + timedelta(hours=1)])
        assert result == []

    def test_single_boundary(self):
        fc = [(T0, 100.0)]
        result = align_time_series_on_time_slots(fc, [T0])
        assert result == []

    def test_single_slot(self):
        fc = [(T0, 100.0), (T0 + timedelta(hours=1), 200.0)]
        result = align_time_series_on_time_slots(fc, [T0, T0 + timedelta(hours=1)])
        assert len(result) == 1
        assert result[0][0] == T0
        assert result[0][1] > 0.0

    def test_multiple_slots(self):
        fc = [
            (T0, 100.0),
            (T0 + timedelta(minutes=30), 200.0),
            (T0 + timedelta(minutes=60), 300.0),
            (T0 + timedelta(minutes=90), 400.0),
        ]
        boundaries = [
            T0,
            T0 + timedelta(minutes=30),
            T0 + timedelta(minutes=60),
            T0 + timedelta(minutes=90),
        ]
        result = align_time_series_on_time_slots(fc, boundaries)
        assert len(result) == 3

    def test_misaligned_boundaries(self):
        fc = [
            (T0, 100.0),
            (T0 + timedelta(minutes=60), 200.0),
        ]
        # Boundaries don't match forecast points
        boundaries = [
            T0 + timedelta(minutes=15),
            T0 + timedelta(minutes=45),
        ]
        result = align_time_series_on_time_slots(fc, boundaries)
        assert len(result) == 1

    def test_different_resolutions(self):
        # High-res forecast, low-res boundaries
        fc = [(T0 + timedelta(minutes=5 * i), float(100 + i * 10)) for i in range(13)]
        boundaries = [T0, T0 + timedelta(minutes=30), T0 + timedelta(minutes=60)]
        result = align_time_series_on_time_slots(fc, boundaries)
        assert len(result) == 2

    def test_with_geometric_smoothing(self):
        fc = [(T0, 100.0), (T0 + timedelta(hours=1), 200.0)]
        result = align_time_series_on_time_slots(fc, [T0, T0 + timedelta(hours=1)], geometric_smoothing=True)
        assert len(result) == 1
        assert result[0][1] > 0.0


# ---------------------------------------------------------------------------
# Backward compatibility: functions accessible from both import paths
# ---------------------------------------------------------------------------
class TestBackwardCompatibility:
    def test_align_same_function(self):
        assert load_align is align_time_series_and_values

    def test_get_slots_same_function(self):
        assert load_get_slots is get_slots_from_time_series

    def test_get_value_same_function(self):
        assert load_get_value is get_value_from_time_series


# ---------------------------------------------------------------------------
# get_slots_from_time_series (moved function)
# ---------------------------------------------------------------------------
class TestGetSlotsFromTimeSeries:
    def test_empty(self):
        assert get_slots_from_time_series([], T0) == []

    def test_single_point_no_end(self):
        ts = [(T0, 100.0)]
        result = get_slots_from_time_series(ts, T0)
        assert len(result) == 1

    def test_range_extraction(self):
        ts = [
            (T0, 100.0),
            (T0 + timedelta(minutes=30), 200.0),
            (T0 + timedelta(minutes=60), 300.0),
        ]
        result = get_slots_from_time_series(ts, T0, T0 + timedelta(minutes=60))
        assert len(result) == 3

    def test_start_before_series(self):
        ts = [(T0 + timedelta(minutes=30), 200.0)]
        result = get_slots_from_time_series(ts, T0, T0 + timedelta(hours=1))
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# get_value_from_time_series (moved function)
# ---------------------------------------------------------------------------
class TestGetValueFromTimeSeries:
    def test_empty(self):
        result = get_value_from_time_series([], T0)
        assert result == (None, None, False, -1)

    def test_none_input(self):
        result = get_value_from_time_series(None, T0)
        assert result == (None, None, False, -1)

    def test_exact_match_first(self):
        ts = [(T0, 100.0), (T0 + timedelta(hours=1), 200.0)]
        t, v, exact, idx = get_value_from_time_series(ts, T0)
        assert t == T0
        assert v == 100.0
        assert exact is True
        assert idx == 0

    def test_exact_match_last(self):
        ts = [(T0, 100.0), (T0 + timedelta(hours=1), 200.0)]
        t, v, exact, idx = get_value_from_time_series(ts, T0 + timedelta(hours=1))
        assert t == T0 + timedelta(hours=1)
        assert v == 200.0
        assert exact is True

    def test_nearest_match(self):
        ts = [(T0, 100.0), (T0 + timedelta(hours=1), 200.0)]
        t, v, exact, idx = get_value_from_time_series(ts, T0 + timedelta(minutes=10))
        assert exact is False
        assert t == T0  # closer to T0

    def test_after_series(self):
        ts = [(T0, 100.0)]
        t, v, exact, idx = get_value_from_time_series(ts, T0 + timedelta(hours=5))
        assert v == 100.0

    def test_before_series(self):
        ts = [(T0 + timedelta(hours=1), 100.0)]
        t, v, exact, idx = get_value_from_time_series(ts, T0)
        assert v == 100.0

    def test_with_interpolation(self):
        ts = [(T0, 100.0), (T0 + timedelta(hours=1), 200.0)]

        def interp(v1, v2, t):
            frac = (t - v1[0]).total_seconds() / (v2[0] - v1[0]).total_seconds()
            return (t, v1[1] + frac * (v2[1] - v1[1]))

        t, v, exact, idx = get_value_from_time_series(ts, T0 + timedelta(minutes=30), interpolation_operation=interp)
        assert v == pytest.approx(150.0)

    def test_interpolation_with_none_values(self):
        ts = [(T0, None), (T0 + timedelta(hours=1), None)]

        def interp(v1, v2, t):
            return (t, 0.0)

        t, v, exact, idx = get_value_from_time_series(ts, T0 + timedelta(minutes=30), interpolation_operation=interp)
        assert v is None
        assert idx == -1

    def test_interpolation_first_none(self):
        ts = [(T0, None), (T0 + timedelta(hours=1), 200.0)]

        def interp(v1, v2, t):
            return (t, 0.0)

        t, v, exact, idx = get_value_from_time_series(ts, T0 + timedelta(minutes=30), interpolation_operation=interp)
        assert v == 200.0

    def test_interpolation_second_none(self):
        ts = [(T0, 100.0), (T0 + timedelta(hours=1), None)]

        def interp(v1, v2, t):
            return (t, 0.0)

        t, v, exact, idx = get_value_from_time_series(ts, T0 + timedelta(minutes=30), interpolation_operation=interp)
        assert v == 100.0

    def test_exact_match_middle(self):
        ts = [
            (T0, 100.0),
            (T0 + timedelta(minutes=30), 200.0),
            (T0 + timedelta(minutes=60), 300.0),
        ]
        t, v, exact, idx = get_value_from_time_series(ts, T0 + timedelta(minutes=30))
        assert exact is True
        assert v == 200.0


# ---------------------------------------------------------------------------
# align_time_series_and_values (moved function) - additional edge tests
# ---------------------------------------------------------------------------
class TestAlignTimeSeriesAndValues:
    def test_both_empty_with_operation(self):
        result = align_time_series_and_values([], [], operation=lambda x, y: (x or 0) + (y or 0))
        assert result == []

    def test_both_empty_no_operation(self):
        r1, r2 = align_time_series_and_values([], [])
        assert r1 == []
        assert r2 == []

    def test_first_empty_with_operation_2tuple(self):
        ts2 = [(T0, 100.0)]
        result = align_time_series_and_values([], ts2, operation=lambda x, y: (x or 0) + (y or 0))
        assert len(result) == 1

    def test_second_empty_with_operation_2tuple(self):
        ts1 = [(T0, 100.0)]
        result = align_time_series_and_values(ts1, [], operation=lambda x, y: (x or 0) + (y or 0))
        assert len(result) == 1

    def test_first_empty_no_operation_2tuple(self):
        ts2 = [(T0, 100.0)]
        r1, r2 = align_time_series_and_values([], ts2)
        assert len(r1) == 1
        assert r1[0][1] is None
        assert r2 == ts2

    def test_second_empty_no_operation_2tuple(self):
        ts1 = [(T0, 100.0)]
        r1, r2 = align_time_series_and_values(ts1, [])
        assert r1 == ts1
        assert len(r2) == 1
        assert r2[0][1] is None

    def test_3tuple_with_operation(self):
        ts1 = [(T0, 100.0, {"a": 1})]
        ts2 = [(T0, 200.0, {"b": 2})]
        result = align_time_series_and_values(ts1, ts2, operation=lambda x, y: x + y)
        assert len(result) == 1
        assert result[0][1] == 300.0

    def test_3tuple_empty_first_with_op(self):
        ts2 = [(T0, 100.0, {"b": 2})]
        result = align_time_series_and_values([], ts2, operation=lambda x, y: (x or 0) + (y or 0))
        assert len(result) == 1

    def test_3tuple_empty_second_with_op(self):
        ts1 = [(T0, 100.0, {"a": 1})]
        result = align_time_series_and_values(ts1, [], operation=lambda x, y: (x or 0) + (y or 0))
        assert len(result) == 1

    def test_3tuple_empty_first_no_op(self):
        ts2 = [(T0, 100.0, {"b": 2})]
        r1, r2 = align_time_series_and_values([], ts2)
        assert r1[0][1] is None

    def test_3tuple_empty_second_no_op(self):
        ts1 = [(T0, 100.0, {"a": 1})]
        r1, r2 = align_time_series_and_values(ts1, [])
        assert r2[0][1] is None

    def test_interpolation_between_points(self):
        ts1 = [(T0, 100.0), (T0 + timedelta(hours=1), 200.0)]
        ts2 = [(T0 + timedelta(minutes=30), 50.0)]
        r1, r2 = align_time_series_and_values(ts1, ts2)
        # ts1 should have interpolated value at T0+30m
        assert len(r1) == 3  # T0, T0+30m, T0+60m
        assert r1[1][1] == pytest.approx(150.0)

    def test_out_of_order_source_triggers_d2_nonpositive_fallback(self):
        """When source entries are out of order, interpolation between adjacent
        entries can produce d2 <= 0, falling back to vcur."""
        # tsv1 is intentionally out of order: index 0 has T0+1h, indices 1-2 have T0
        ts1 = [
            (T0 + timedelta(hours=1), 200.0),
            (T0, 100.0),
            (T0, 300.0),
            (T0 + timedelta(hours=2), 400.0),
        ]
        # Query at T0+90min: between T0+1h and T0+2h in sorted timings.
        # last_real_idx=0 (set at T0+1h), tsv1[1]=(T0,100) so d2<0 → fallback to vcur=200
        ts2 = [(T0 + timedelta(minutes=90), 50.0)]
        r1, _r2 = align_time_series_and_values(ts1, ts2)
        interpolated = [v for t, v in r1 if t == T0 + timedelta(minutes=90)]
        assert len(interpolated) == 1
        assert interpolated[0] == 200.0  # vcur fallback, no division
