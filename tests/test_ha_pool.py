"""Tests for ha_model/pool.py - Pool device functionality."""

from __future__ import annotations

from datetime import datetime, timedelta
from datetime import time as dt_time
from unittest.mock import MagicMock

import pytz

from custom_components.quiet_solar.const import (
    CONF_POOL_DEFAULT_IDX,
    CONF_POOL_TEMPERATURE_SENSOR,
    CONF_POOL_WINTER_IDX,
    POOL_TEMP_STEPS,
)


class FakeQSPool:
    """A testable version of QSPool that doesn't require full HA setup."""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "Test Pool")
        self.pool_temperature_sensor = kwargs.get(CONF_POOL_TEMPERATURE_SENSOR, "sensor.pool_temp")
        self.power_use = kwargs.get("power", 1500)
        self.hass = kwargs.get("hass", MagicMock())
        self.home = kwargs.get("home", MagicMock())
        self.bistate_mode = kwargs.get("bistate_mode", "bistate_mode_auto")
        self.default_on_finish_time = kwargs.get("default_on_finish_time", dt_time(hour=0, minute=0, second=0))
        self.is_load_time_sensitive = False

        # Pool steps configuration
        self.pool_steps = []
        for min_temp, max_temp, default in POOL_TEMP_STEPS:
            val = kwargs.get(f"water_temp_{max_temp}", default)
            self.pool_steps.append([min_temp, max_temp, val])

        # Mock methods
        self._temp_history = kwargs.get("temp_history", [])
        self._current_temp = kwargs.get("current_temp", None)

    def get_sensor_latest_possible_valid_value(self, entity_id):
        """Mock temperature reading."""
        return self._current_temp

    def get_state_history_data(self, entity_id, num_seconds_before, to_ts, keep_invalid_states):
        """Mock temperature history."""
        return self._temp_history

    def is_best_effort_only_load(self):
        return False

    def get_pool_filter_time_s(self, force_winter: bool, time: datetime) -> float:
        idx = 0
        if force_winter:
            idx = CONF_POOL_WINTER_IDX
        else:
            data = self._temp_history
            if data is None or len(data) == 0:
                idx = CONF_POOL_DEFAULT_IDX
            else:
                temps = [x[1] for x in data]
                temp = min(temps)
                idx = CONF_POOL_DEFAULT_IDX
                for id, t in enumerate(self.pool_steps):
                    min_temp, max_temp, val = t
                    if temp >= min_temp and temp <= max_temp:
                        idx = id
                        break

        return self.pool_steps[idx][2] * 3600.0


def test_pool_current_water_temperature_with_value():
    """Test getting current water temperature."""
    pool = FakeQSPool(current_temp=25.5)
    temp = pool.get_sensor_latest_possible_valid_value(pool.pool_temperature_sensor)
    assert temp == 25.5


def test_pool_current_water_temperature_none():
    """Test getting current water temperature when sensor unavailable."""
    pool = FakeQSPool(current_temp=None)
    temp = pool.get_sensor_latest_possible_valid_value(pool.pool_temperature_sensor)
    assert temp is None


def test_pool_get_filter_time_winter_mode():
    """Test pool filter time calculation in winter mode."""
    pool = FakeQSPool()
    dt = datetime(year=2024, month=1, day=15, hour=12, tzinfo=pytz.UTC)

    filter_time_s = pool.get_pool_filter_time_s(force_winter=True, time=dt)

    expected_hours = pool.pool_steps[CONF_POOL_WINTER_IDX][2]
    assert filter_time_s == expected_hours * 3600.0


def test_pool_get_filter_time_auto_no_history():
    """Test pool filter time when no temperature history available."""
    pool = FakeQSPool(temp_history=None)
    dt = datetime(year=2024, month=6, day=15, hour=12, tzinfo=pytz.UTC)

    filter_time_s = pool.get_pool_filter_time_s(force_winter=False, time=dt)

    expected_hours = pool.pool_steps[CONF_POOL_DEFAULT_IDX][2]
    assert filter_time_s == expected_hours * 3600.0


def test_pool_get_filter_time_auto_empty_history():
    """Test pool filter time when temperature history is empty."""
    pool = FakeQSPool(temp_history=[])
    dt = datetime(year=2024, month=6, day=15, hour=12, tzinfo=pytz.UTC)

    filter_time_s = pool.get_pool_filter_time_s(force_winter=False, time=dt)

    expected_hours = pool.pool_steps[CONF_POOL_DEFAULT_IDX][2]
    assert filter_time_s == expected_hours * 3600.0


def test_pool_get_filter_time_auto_with_history():
    """Test pool filter time based on temperature history."""
    history = [
        (datetime.now(pytz.UTC) - timedelta(hours=12), 22.0),
        (datetime.now(pytz.UTC) - timedelta(hours=6), 21.0),
        (datetime.now(pytz.UTC), 20.0),
    ]
    pool = FakeQSPool(temp_history=history)
    dt = datetime(year=2024, month=6, day=15, hour=12, tzinfo=pytz.UTC)

    filter_time_s = pool.get_pool_filter_time_s(force_winter=False, time=dt)

    assert filter_time_s >= 0


def test_pool_steps_configuration():
    """Test that pool steps are properly configured."""
    pool = FakeQSPool()

    assert len(pool.pool_steps) > 0

    for step in pool.pool_steps:
        assert len(step) == 3
        min_temp, max_temp, hours = step
        assert max_temp >= min_temp
        assert hours >= 0


def test_pool_bistate_mode():
    """Test that bistate_mode is stored correctly."""
    pool_auto = FakeQSPool(bistate_mode="bistate_mode_auto")
    assert pool_auto.bistate_mode == "bistate_mode_auto"

    pool_winter = FakeQSPool(bistate_mode="pool_winter_mode")
    assert pool_winter.bistate_mode == "pool_winter_mode"


def test_pool_default_on_finish_time():
    """Test pool default finish time."""
    pool = FakeQSPool(default_on_finish_time=dt_time(hour=6, minute=30))
    assert pool.default_on_finish_time == dt_time(hour=6, minute=30)


def test_pool_is_not_time_sensitive():
    """Test pool is not marked as time sensitive."""
    pool = FakeQSPool()
    assert pool.is_load_time_sensitive is False


def test_pool_power_use():
    """Test pool power consumption setting."""
    pool = FakeQSPool(power=2000)
    assert pool.power_use == 2000


def _make_pool_with_day_bounds(now, end_day_offset_h=24, start_day_offset_h=0):
    """Create a FakeQSPool with get_next_time_from_hours mocked for DST-safe day bounds.

    Returns (pool, end_day, start_day).
    """
    end_day = now + timedelta(hours=end_day_offset_h)
    start_day = now - timedelta(hours=start_day_offset_h)
    pool = FakeQSPool()
    pool.qs_bistate_current_on_h = 0.0
    pool.qs_bistate_current_duration_h = 0.0
    # First call: end_day; second call: start_day (DST-safe boundary)
    pool.get_next_time_from_hours = MagicMock(side_effect=[end_day, start_day])
    return pool, end_day, start_day


def test_pool_update_current_metrics_completed_and_active_shows_only_active():
    """Test that active constraints exclude completed constraint from metrics."""
    from custom_components.quiet_solar.ha_model.pool import QSPool

    now = datetime.now(tz=pytz.UTC)
    end = now + timedelta(hours=1)

    completed_ct = MagicMock()
    completed_ct.end_of_constraint = end
    completed_ct.start_of_constraint = now - timedelta(hours=1)
    completed_ct.target_value = 3600.0
    completed_ct.current_value = 1800.0

    active_ct = MagicMock()
    active_ct.end_of_constraint = end
    active_ct.start_of_constraint = now
    active_ct.target_value = 7200.0
    active_ct.current_value = 3600.0

    pool, _, _ = _make_pool_with_day_bounds(now)
    pool._last_completed_constraint = completed_ct
    pool._constraints = [active_ct]

    QSPool.update_current_metrics(pool, now)

    assert pool.qs_bistate_current_on_h == 3600.0 / 3600.0
    assert pool.qs_bistate_current_duration_h == 7200.0 / 3600.0


def test_pool_update_current_metrics_completed_only_shows_completed():
    """Test that completed constraint values display when no active constraints."""
    from custom_components.quiet_solar.ha_model.pool import QSPool

    now = datetime.now(tz=pytz.UTC)
    end = now + timedelta(hours=1)

    completed_ct = MagicMock()
    completed_ct.end_of_constraint = end
    completed_ct.start_of_constraint = now - timedelta(hours=1)
    completed_ct.target_value = 3600.0
    completed_ct.current_value = 3600.0

    pool, _, _ = _make_pool_with_day_bounds(now)
    pool._last_completed_constraint = completed_ct
    pool._constraints = []

    QSPool.update_current_metrics(pool, now)

    assert pool.qs_bistate_current_on_h == 3600.0 / 3600.0
    assert pool.qs_bistate_current_duration_h == 3600.0 / 3600.0


def test_pool_update_current_metrics_after_reset_shows_zero():
    """Test that metrics are zero when no constraints and no completed constraint."""
    from custom_components.quiet_solar.ha_model.pool import QSPool

    now = datetime.now(tz=pytz.UTC)

    pool, _, _ = _make_pool_with_day_bounds(now)
    pool._last_completed_constraint = None
    pool._constraints = []
    pool.qs_bistate_current_on_h = 99.0
    pool.qs_bistate_current_duration_h = 99.0

    QSPool.update_current_metrics(pool, now)

    assert pool.qs_bistate_current_on_h == 0.0
    assert pool.qs_bistate_current_duration_h == 0.0


def test_pool_update_current_metrics_partial_completion():
    """Test metrics with partial completion (target_value != current_value)."""
    from custom_components.quiet_solar.ha_model.pool import QSPool

    now = datetime.now(tz=pytz.UTC)

    ct = MagicMock()
    ct.end_of_constraint = now + timedelta(hours=2)
    ct.start_of_constraint = now - timedelta(hours=1)
    ct.target_value = 7200.0  # 2h target
    ct.current_value = 2700.0  # 0.75h done

    pool, _, _ = _make_pool_with_day_bounds(now)
    pool._last_completed_constraint = None
    pool._constraints = [ct]

    QSPool.update_current_metrics(pool, now)

    assert pool.qs_bistate_current_on_h == 2700.0 / 3600.0
    assert pool.qs_bistate_current_duration_h == 7200.0 / 3600.0


def test_pool_update_current_metrics_day_boundary_excludes_old():
    """Test that constraints outside the current day window are excluded."""
    from custom_components.quiet_solar.ha_model.pool import QSPool
    from custom_components.quiet_solar.home_model.constraints import DATETIME_MIN_UTC

    now = datetime.now(tz=pytz.UTC)
    end_day = now + timedelta(hours=12)
    start_day = end_day - timedelta(hours=24)

    # Constraint entirely before the day window
    old_ct = MagicMock()
    old_ct.end_of_constraint = start_day - timedelta(hours=1)
    old_ct.start_of_constraint = start_day - timedelta(hours=5)

    pool = FakeQSPool()
    pool.qs_bistate_current_on_h = 0.0
    pool.qs_bistate_current_duration_h = 0.0
    pool.get_next_time_from_hours = MagicMock(side_effect=[end_day, start_day])
    pool._last_completed_constraint = old_ct
    pool._constraints = []

    QSPool.update_current_metrics(pool, now)

    assert pool.qs_bistate_current_on_h == 0.0
    assert pool.qs_bistate_current_duration_h == 0.0


def test_pool_update_current_metrics_multiple_active_constraints_sum():
    """Test that multiple active constraints are all summed (by design)."""
    from custom_components.quiet_solar.ha_model.pool import QSPool

    now = datetime.now(tz=pytz.UTC)

    ct1 = MagicMock()
    ct1.end_of_constraint = now + timedelta(hours=4)
    ct1.start_of_constraint = now
    ct1.target_value = 3600.0
    ct1.current_value = 1800.0

    ct2 = MagicMock()
    ct2.end_of_constraint = now + timedelta(hours=8)
    ct2.start_of_constraint = now + timedelta(hours=4)
    ct2.target_value = 7200.0
    ct2.current_value = 0.0

    pool, _, _ = _make_pool_with_day_bounds(now)
    pool._last_completed_constraint = None
    pool._constraints = [ct1, ct2]

    QSPool.update_current_metrics(pool, now)

    assert pool.qs_bistate_current_on_h == (1800.0 + 0.0) / 3600.0
    assert pool.qs_bistate_current_duration_h == (3600.0 + 7200.0) / 3600.0


def test_pool_update_current_metrics_with_end_range_parameter():
    """Test update_current_metrics with explicit end_range parameter."""
    from custom_components.quiet_solar.ha_model.pool import QSPool

    now = datetime.now(tz=pytz.UTC)

    ct = MagicMock()
    ct.end_of_constraint = now + timedelta(hours=2)
    ct.start_of_constraint = now
    ct.target_value = 5400.0
    ct.current_value = 1800.0

    pool, _, _ = _make_pool_with_day_bounds(now)
    pool._last_completed_constraint = None
    pool._constraints = [ct]

    # Call with explicit end_range (6:30 AM)
    QSPool.update_current_metrics(pool, now, end_range=dt_time(hour=6, minute=30))

    assert pool.qs_bistate_current_on_h == 1800.0 / 3600.0
    assert pool.qs_bistate_current_duration_h == 5400.0 / 3600.0
    # Verify get_next_time_from_hours was called with the custom end_range
    first_call_args = pool.get_next_time_from_hours.call_args_list[0]
    assert first_call_args[1]["local_hours"] == dt_time(hour=6, minute=30)


def test_pool_update_current_metrics_dst_safe_day_boundary():
    """Test that DST-safe day boundary uses get_next_time_from_hours, not raw 24h."""
    from custom_components.quiet_solar.ha_model.pool import QSPool

    now = datetime.now(tz=pytz.UTC)
    # Simulate spring-forward: 23h local day
    end_day = now + timedelta(hours=12)
    start_day = end_day - timedelta(hours=23)  # 23h day, not 24h

    ct = MagicMock()
    ct.end_of_constraint = now + timedelta(hours=1)
    ct.start_of_constraint = now
    ct.target_value = 3600.0
    ct.current_value = 1800.0

    pool = FakeQSPool()
    pool.qs_bistate_current_on_h = 0.0
    pool.qs_bistate_current_duration_h = 0.0
    pool.get_next_time_from_hours = MagicMock(side_effect=[end_day, start_day])
    pool._last_completed_constraint = None
    pool._constraints = [ct]

    QSPool.update_current_metrics(pool, now)

    assert pool.qs_bistate_current_on_h == 1800.0 / 3600.0
    # Verify the second call uses end_day - 26h (DST-safe offset)
    second_call = pool.get_next_time_from_hours.call_args_list[1]
    assert second_call[1]["time_utc_now"] == end_day - timedelta(hours=26)
