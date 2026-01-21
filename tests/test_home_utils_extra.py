"""Extra tests for home_utils helpers."""

from datetime import datetime, timedelta

import pytz

from custom_components.quiet_solar.home_model.home_utils import (
    add_amps,
    diff_amps,
    get_average_time_series,
    is_amps_zero,
    max_amps,
    min_amps,
)


def test_amp_helpers_with_none() -> None:
    """Test amps helpers handle None inputs."""
    assert is_amps_zero(None) is True
    assert add_amps(None, [1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    assert add_amps([1.0, 2.0, 3.0], None) == [1.0, 2.0, 3.0]
    assert min_amps(None, [4.0, 5.0, 6.0]) == [4.0, 5.0, 6.0]
    assert max_amps([1.0, 2.0, 3.0], None) == [1.0, 2.0, 3.0]
    assert diff_amps([3.0, 2.0, 1.0], [1.0, 1.0, 1.0]) == [2.0, 1.0, 0.0]


def test_get_average_time_series_with_bounds() -> None:
    """Test average time series with explicit bounds."""
    now = datetime(2026, 1, 15, 12, 0, tzinfo=pytz.UTC)
    data = [
        (now, 10.0, {}),
        (now + timedelta(minutes=10), 20.0, {}),
        (now + timedelta(minutes=20), None, {}),
        (now + timedelta(minutes=30), 30.0, {}),
    ]
    avg = get_average_time_series(
        data, first_timing=now - timedelta(minutes=5), last_timing=now + timedelta(minutes=40)
    )
    assert avg == 39000 / 2100
    assert avg > 0
