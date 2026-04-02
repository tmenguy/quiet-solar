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

        self.calendar = None

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

    def _is_calendar_based_mode(self, bistate_mode):
        """Pool never uses calendar-based mode for auto/winter."""
        return False

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


def _make_pool_with_day_bounds(now, today_utc=None, tomorrow_utc=None):
    """Create a FakeQSPool with mocked day boundaries for update_current_metrics.

    Returns (pool, today_utc, tomorrow_utc).
    """
    if today_utc is None:
        today_utc = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if tomorrow_utc is None:
        tomorrow_utc = today_utc + timedelta(days=1)
    pool = FakeQSPool()
    pool.qs_bistate_current_on_h = 0.0
    pool.qs_bistate_current_duration_h = 0.0
    pool._get_today_boundaries = MagicMock(return_value=(today_utc, tomorrow_utc))
    return pool, today_utc, tomorrow_utc


async def test_pool_update_current_metrics_completed_and_active_sums_both():
    """Test that active + completed constraints from today are both counted."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration

    now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)

    completed_ct = MagicMock()
    completed_ct.end_of_constraint = datetime(2026, 3, 30, 8, 0, 0, tzinfo=pytz.UTC)
    completed_ct.start_of_constraint = datetime(2026, 3, 30, 7, 0, 0, tzinfo=pytz.UTC)
    completed_ct.target_value = 3600.0
    completed_ct.current_value = 1800.0

    active_ct = MagicMock()
    active_ct.end_of_constraint = datetime(2026, 3, 30, 17, 0, 0, tzinfo=pytz.UTC)
    active_ct.start_of_constraint = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    active_ct.target_value = 7200.0
    active_ct.current_value = 3600.0

    pool, _, _ = _make_pool_with_day_bounds(now, today_utc, tomorrow_utc)
    pool._last_completed_constraint = completed_ct
    pool._constraints = [active_ct]

    await QSBiStateDuration.update_current_metrics(pool, now)

    # Both constraints from today are summed
    assert pool.qs_bistate_current_on_h == (3600.0 + 1800.0) / 3600.0
    assert pool.qs_bistate_current_duration_h == (7200.0 + 3600.0) / 3600.0


async def test_pool_update_current_metrics_completed_only_shows_completed():
    """Test that completed constraint values display when no active constraints."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration

    now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)

    completed_ct = MagicMock()
    completed_ct.end_of_constraint = datetime(2026, 3, 30, 8, 0, 0, tzinfo=pytz.UTC)
    completed_ct.start_of_constraint = datetime(2026, 3, 30, 7, 0, 0, tzinfo=pytz.UTC)
    completed_ct.target_value = 3600.0
    completed_ct.current_value = 3600.0

    pool, _, _ = _make_pool_with_day_bounds(now, today_utc, tomorrow_utc)
    pool._last_completed_constraint = completed_ct
    pool._constraints = []

    await QSBiStateDuration.update_current_metrics(pool, now)

    assert pool.qs_bistate_current_on_h == 3600.0 / 3600.0
    assert pool.qs_bistate_current_duration_h == 3600.0 / 3600.0


async def test_pool_update_current_metrics_after_reset_shows_zero():
    """Test that metrics are zero when no constraints and no completed constraint."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration

    now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)

    pool, _, _ = _make_pool_with_day_bounds(now)
    pool._last_completed_constraint = None
    pool._constraints = []
    pool.qs_bistate_current_on_h = 99.0
    pool.qs_bistate_current_duration_h = 99.0

    await QSBiStateDuration.update_current_metrics(pool, now)

    assert pool.qs_bistate_current_on_h == 0.0
    assert pool.qs_bistate_current_duration_h == 0.0


async def test_pool_update_current_metrics_partial_completion():
    """Test metrics with partial completion (target_value != current_value)."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration

    now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)

    ct = MagicMock()
    ct.end_of_constraint = datetime(2026, 3, 30, 14, 0, 0, tzinfo=pytz.UTC)
    ct.start_of_constraint = datetime(2026, 3, 30, 11, 0, 0, tzinfo=pytz.UTC)
    ct.target_value = 7200.0  # 2h target
    ct.current_value = 2700.0  # 0.75h done

    pool, _, _ = _make_pool_with_day_bounds(now, today_utc, tomorrow_utc)
    pool._last_completed_constraint = None
    pool._constraints = [ct]

    await QSBiStateDuration.update_current_metrics(pool, now)

    assert pool.qs_bistate_current_on_h == 2700.0 / 3600.0
    assert pool.qs_bistate_current_duration_h == 7200.0 / 3600.0


async def test_pool_update_current_metrics_day_boundary_excludes_old():
    """Test that constraints outside the current day window are excluded."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    from custom_components.quiet_solar.home_model.constraints import DATETIME_MAX_UTC

    now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)

    # Completed constraint from 2 days ago
    old_ct = MagicMock()
    old_ct.end_of_constraint = datetime(2026, 3, 28, 10, 0, 0, tzinfo=pytz.UTC)
    old_ct.start_of_constraint = datetime(2026, 3, 28, 7, 0, 0, tzinfo=pytz.UTC)

    pool, _, _ = _make_pool_with_day_bounds(now, today_utc, tomorrow_utc)
    pool._last_completed_constraint = old_ct
    pool._constraints = []

    await QSBiStateDuration.update_current_metrics(pool, now)

    assert pool.qs_bistate_current_on_h == 0.0
    assert pool.qs_bistate_current_duration_h == 0.0


async def test_pool_update_current_metrics_multiple_active_constraints_sum():
    """Test that multiple active constraints within today are all summed."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration

    now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)

    ct1 = MagicMock()
    ct1.end_of_constraint = datetime(2026, 3, 30, 16, 0, 0, tzinfo=pytz.UTC)
    ct1.start_of_constraint = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    ct1.target_value = 3600.0
    ct1.current_value = 1800.0

    ct2 = MagicMock()
    ct2.end_of_constraint = datetime(2026, 3, 30, 20, 0, 0, tzinfo=pytz.UTC)
    ct2.start_of_constraint = datetime(2026, 3, 30, 16, 0, 0, tzinfo=pytz.UTC)
    ct2.target_value = 7200.0
    ct2.current_value = 0.0

    pool, _, _ = _make_pool_with_day_bounds(now, today_utc, tomorrow_utc)
    pool._last_completed_constraint = None
    pool._constraints = [ct1, ct2]

    await QSBiStateDuration.update_current_metrics(pool, now)

    assert pool.qs_bistate_current_on_h == (1800.0 + 0.0) / 3600.0
    assert pool.qs_bistate_current_duration_h == (3600.0 + 7200.0) / 3600.0


async def test_pool_update_current_metrics_tomorrow_constraint_excluded():
    """Test that constraints ending tomorrow are excluded from today's metrics."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration

    now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)

    ct = MagicMock()
    ct.end_of_constraint = datetime(2026, 3, 31, 7, 0, 0, tzinfo=pytz.UTC)  # tomorrow
    ct.start_of_constraint = datetime(2026, 3, 31, 6, 0, 0, tzinfo=pytz.UTC)
    ct.target_value = 3600.0
    ct.current_value = 0.0

    pool, _, _ = _make_pool_with_day_bounds(now, today_utc, tomorrow_utc)
    pool._last_completed_constraint = None
    pool._constraints = [ct]

    await QSBiStateDuration.update_current_metrics(pool, now)

    assert pool.qs_bistate_current_on_h == 0.0
    assert pool.qs_bistate_current_duration_h == 0.0


async def test_pool_update_current_metrics_completed_from_today_included():
    """Test that completed constraint from today is counted in metrics."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration

    now = datetime(2026, 3, 30, 18, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)

    ct = MagicMock()
    ct.end_of_constraint = datetime(2026, 3, 30, 14, 0, 0, tzinfo=pytz.UTC)
    ct.start_of_constraint = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    ct.target_value = 5400.0
    ct.current_value = 1800.0

    pool, _, _ = _make_pool_with_day_bounds(now, today_utc, tomorrow_utc)
    pool._last_completed_constraint = ct
    pool._constraints = []

    await QSBiStateDuration.update_current_metrics(pool, now)

    assert pool.qs_bistate_current_on_h == 1800.0 / 3600.0
    assert pool.qs_bistate_current_duration_h == 5400.0 / 3600.0


async def test_pool_update_current_metrics_yesterday_active_constraint_excluded():
    """Test that an active constraint from yesterday does not leak into today's metrics.

    Bug #95: The active constraints loop only had an upper bound (end <= tomorrow),
    missing a lower bound (end > today), so yesterday's constraints leaked through.
    """
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration

    now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)

    # Active constraint from yesterday (end < today) — should be excluded
    yesterday_ct = MagicMock()
    yesterday_ct.end_of_constraint = datetime(2026, 3, 29, 17, 0, 0, tzinfo=pytz.UTC)
    yesterday_ct.start_of_constraint = datetime(2026, 3, 29, 12, 0, 0, tzinfo=pytz.UTC)
    yesterday_ct.target_value = 3600.0
    yesterday_ct.current_value = 3600.0

    # Active constraint from today — should be included
    today_ct = MagicMock()
    today_ct.end_of_constraint = datetime(2026, 3, 30, 17, 0, 0, tzinfo=pytz.UTC)
    today_ct.start_of_constraint = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    today_ct.target_value = 7200.0
    today_ct.current_value = 1800.0

    pool, _, _ = _make_pool_with_day_bounds(now, today_utc, tomorrow_utc)
    pool._last_completed_constraint = None
    pool._constraints = [yesterday_ct, today_ct]

    await QSBiStateDuration.update_current_metrics(pool, now)

    # Only today's constraint should count
    assert pool.qs_bistate_current_duration_h == 7200.0 / 3600.0
    assert pool.qs_bistate_current_on_h == 1800.0 / 3600.0


async def test_pool_update_current_metrics_same_end_date_no_double_count():
    """Test that lcc with same end date as active constraint does not double-count.

    Bug #95: When push_live_constraint replaces a completed constraint with a new one
    sharing the same end date, the completed constraint's runtime is carried over into
    the new constraint. Counting both would double-count.
    """
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    from custom_components.quiet_solar.home_model.constraints import TimeBasedSimplePowerLoadConstraint

    now = datetime(2026, 3, 30, 14, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)

    # Completed constraint ending at 17:00 — runtime was carried into new constraint
    completed_ct = TimeBasedSimplePowerLoadConstraint(
        type=1, time=now, power=1500, initial_value=0,
        target_value=28800.0, current_value=28800.0,
        start_of_constraint=datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC),
        end_of_constraint=datetime(2026, 3, 30, 17, 0, 0, tzinfo=pytz.UTC),
    )

    # New active constraint with SAME end date — already absorbed lcc's runtime
    active_ct = TimeBasedSimplePowerLoadConstraint(
        type=1, time=now, power=1500, initial_value=0,
        target_value=36000.0, current_value=28800.0,
        start_of_constraint=datetime(2026, 3, 30, 8, 0, 0, tzinfo=pytz.UTC),
        end_of_constraint=datetime(2026, 3, 30, 17, 0, 0, tzinfo=pytz.UTC),
    )

    pool, _, _ = _make_pool_with_day_bounds(now, today_utc, tomorrow_utc)
    pool._last_completed_constraint = completed_ct
    pool._constraints = [active_ct]

    await QSBiStateDuration.update_current_metrics(pool, now)

    # Only the active constraint should be counted — lcc is already absorbed
    assert pool.qs_bistate_current_duration_h == 36000.0 / 3600.0  # 10h
    assert pool.qs_bistate_current_on_h == 28800.0 / 3600.0  # 8h


async def test_pool_update_current_metrics_extended_lcc_absorbed_via_initial_end():
    """Test that lcc absorbed via initial_end_of_constraint is not double-counted.

    Bug #95 review: When lcc's end was extended (initial_end != end_of_constraint),
    an active constraint matching the initial end should still trigger absorption.
    """
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    from custom_components.quiet_solar.home_model.constraints import TimeBasedSimplePowerLoadConstraint

    now = datetime(2026, 3, 30, 14, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)

    # lcc created with end=15:00, then extended to 17:00
    completed_ct = TimeBasedSimplePowerLoadConstraint(
        type=1, time=now, power=1500, initial_value=0,
        target_value=14400.0, current_value=14400.0,
        start_of_constraint=datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC),
        end_of_constraint=datetime(2026, 3, 30, 15, 0, 0, tzinfo=pytz.UTC),
    )
    # Simulate extension: end moved to 17:00 but initial_end stays 15:00
    completed_ct.end_of_constraint = datetime(2026, 3, 30, 17, 0, 0, tzinfo=pytz.UTC)

    # Active constraint matching lcc's INITIAL end (15:00), not current end (17:00)
    active_ct = TimeBasedSimplePowerLoadConstraint(
        type=1, time=now, power=1500, initial_value=0,
        target_value=21600.0, current_value=14400.0,
        start_of_constraint=datetime(2026, 3, 30, 8, 0, 0, tzinfo=pytz.UTC),
        end_of_constraint=datetime(2026, 3, 30, 15, 0, 0, tzinfo=pytz.UTC),
    )

    pool, _, _ = _make_pool_with_day_bounds(now, today_utc, tomorrow_utc)
    pool._last_completed_constraint = completed_ct
    pool._constraints = [active_ct]

    await QSBiStateDuration.update_current_metrics(pool, now)

    # lcc absorbed via initial_end match — only active counted
    assert pool.qs_bistate_current_duration_h == 21600.0 / 3600.0  # 6h
    assert pool.qs_bistate_current_on_h == 14400.0 / 3600.0  # 4h


async def test_pool_update_current_metrics_day_rollover_lcc_not_double_counted():
    """Test that lcc from previous day rollover is not double-counted with today's active.

    Bug #101: Pool with default_on_finish_time=00:00 has yesterday's completed
    constraint ending at today_utc (midnight) and today's active constraint ending
    at tomorrow_utc (next midnight). The lcc passes the day-window check but
    represents yesterday's cycle — counting it inflates target by lcc.target and
    actual by lcc.current.
    """
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    from custom_components.quiet_solar.home_model.constraints import TimeBasedSimplePowerLoadConstraint

    now = datetime(2026, 3, 30, 14, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)

    # Yesterday's completed constraint ending exactly at today_utc (midnight rollover)
    # This simulates a pool that ran yesterday with default_on_finish_time=00:00
    lcc = TimeBasedSimplePowerLoadConstraint(
        type=1, time=now, power=1500, initial_value=0,
        target_value=28800.0,   # 8h target (yesterday)
        current_value=28800.0,  # 8h actual (yesterday)
        start_of_constraint=datetime(2026, 3, 29, 16, 0, 0, tzinfo=pytz.UTC),
        end_of_constraint=today_utc,  # midnight — exact rollover boundary
    )

    # Today's active constraint ending at tomorrow_utc (next midnight)
    # User set slider to 10h target, pool has run 3h so far today
    active_ct = TimeBasedSimplePowerLoadConstraint(
        type=1, time=now, power=1500, initial_value=0,
        target_value=36000.0,   # 10h target (slider value)
        current_value=10800.0,  # 3h actual (today's runtime)
        start_of_constraint=datetime(2026, 3, 30, 8, 0, 0, tzinfo=pytz.UTC),
        end_of_constraint=tomorrow_utc,  # next midnight
    )

    pool, _, _ = _make_pool_with_day_bounds(now, today_utc, tomorrow_utc)
    pool._last_completed_constraint = lcc
    pool._constraints = [active_ct]

    await QSBiStateDuration.update_current_metrics(pool, now)

    # Only today's active constraint should count — lcc is a previous-day rollover
    assert pool.qs_bistate_current_duration_h == 36000.0 / 3600.0  # 10h (AC1)
    assert pool.qs_bistate_current_on_h == 10800.0 / 3600.0  # 3h (AC2)
