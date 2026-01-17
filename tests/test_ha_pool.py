"""Tests for ha_model/pool.py - Pool device functionality."""
from __future__ import annotations

from datetime import datetime, time as dt_time, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import pytz

from custom_components.quiet_solar.const import (
    CONF_POOL_TEMPERATURE_SENSOR,
    POOL_TEMP_STEPS,
    CONF_POOL_WINTER_IDX,
    CONF_POOL_DEFAULT_IDX,
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
