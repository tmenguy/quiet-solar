"""Tests for bug #78: bistate/pool target and actual hours display today-only.

Verifies that update_current_metrics in bistate_duration.py:
- Calendar mode: uses pre-computed calendar metrics for today only
- Default/pool mode: day-filters _constraints and _last_completed_constraint to today
- Pool _is_calendar_based_mode returns False for auto/winter modes
- _get_today_boundaries returns local midnight boundaries in UTC
- Sanity-check warning when _last_completed_constraint diverges from calendar actuals
"""

from __future__ import annotations

from datetime import datetime, timedelta
from datetime import time as dt_time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
from custom_components.quiet_solar.home_model.constraints import (
    DATETIME_MAX_UTC,
    DATETIME_MIN_UTC,
    TimeBasedSimplePowerLoadConstraint,
)


# =============================================================================
# Test helpers
# =============================================================================


def _create_bistate_device(calendar=None):
    """Create a minimal concrete bistate device for testing metrics."""
    from custom_components.quiet_solar.const import CONF_CALENDAR, CONF_SWITCH

    hass = MagicMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    home = MagicMock()
    config_entry = MagicMock()
    config_entry.entry_id = "test_bug78"
    config_entry.data = {}

    kwargs = {
        "hass": hass,
        "config_entry": config_entry,
        "home": home,
        "name": "Test Climate",
        CONF_SWITCH: "switch.test_device",
    }
    if calendar is not None:
        kwargs[CONF_CALENDAR] = calendar

    class ConcreteBiState(QSBiStateDuration):
        async def execute_command_system(self, time, command, state):
            return True

        def get_virtual_current_constraint_translation_key(self):
            return "test_key"

        def get_select_translation_key(self):
            return "test_select_key"

    device = ConcreteBiState(**kwargs)
    device.power_use = 2000.0
    device.default_on_finish_time = dt_time(hour=0, minute=0, second=0)
    return device


def _make_constraint(device, time, target_s, current_s, start, end):
    """Create a TimeBasedSimplePowerLoadConstraint with explicit start/end."""
    return TimeBasedSimplePowerLoadConstraint(
        type=1,
        time=time,
        load=device,
        power=device.power_use,
        initial_value=0,
        target_value=target_s,
        current_value=current_s,
        start_of_constraint=start,
        end_of_constraint=end,
    )


def _set_day_boundaries(device, today_utc, tomorrow_utc):
    """Mock _get_today_boundaries to return explicit day boundaries."""
    device._get_today_boundaries = MagicMock(return_value=(today_utc, tomorrow_utc))


# =============================================================================
# _is_calendar_based_mode tests
# =============================================================================


def test_is_calendar_based_mode_auto_with_calendar():
    """Auto mode with calendar attached is calendar-based."""
    device = _create_bistate_device(calendar="calendar.test")
    assert device._is_calendar_based_mode("bistate_mode_auto") is True


def test_is_calendar_based_mode_exact_calendar_with_calendar():
    """Exact calendar mode with calendar attached is calendar-based."""
    device = _create_bistate_device(calendar="calendar.test")
    assert device._is_calendar_based_mode("bistate_mode_exact_calendar") is True


def test_is_calendar_based_mode_default_returns_false():
    """Default mode is never calendar-based."""
    device = _create_bistate_device(calendar="calendar.test")
    assert device._is_calendar_based_mode("bistate_mode_default") is False


def test_is_calendar_based_mode_auto_without_calendar_returns_false():
    """Auto mode without calendar is not calendar-based."""
    device = _create_bistate_device()
    assert device._is_calendar_based_mode("bistate_mode_auto") is False


# =============================================================================
# Pool _is_calendar_based_mode override tests
# =============================================================================


def test_pool_is_calendar_based_mode_auto_returns_false():
    """Pool auto mode is not calendar-based (pool overrides auto)."""
    from custom_components.quiet_solar.ha_model.pool import QSPool

    assert QSPool._is_calendar_based_mode(MagicMock(), "bistate_mode_auto") is False


def test_pool_is_calendar_based_mode_winter_returns_false():
    """Pool winter mode is not calendar-based."""
    from custom_components.quiet_solar.ha_model.pool import QSPool

    assert QSPool._is_calendar_based_mode(MagicMock(), "pool_winter_mode") is False


# =============================================================================
# _get_today_boundaries tests
# =============================================================================


def test_get_today_boundaries_returns_valid_utc_window():
    """Day boundaries span from local midnight to next local midnight in UTC."""
    device = _create_bistate_device()
    now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    today_utc, tomorrow_utc = device._get_today_boundaries(now)

    assert tomorrow_utc > today_utc
    assert today_utc <= now
    assert tomorrow_utc > now


def test_get_today_boundaries_today_before_tomorrow():
    """Start of today must be strictly before start of tomorrow."""
    device = _create_bistate_device()
    now = datetime(2026, 6, 15, 8, 0, 0, tzinfo=pytz.UTC)
    today_utc, tomorrow_utc = device._get_today_boundaries(now)
    assert today_utc < tomorrow_utc


# =============================================================================
# update_current_metrics: calendar path
# =============================================================================


def test_calendar_mode_shows_today_only_target():
    """Calendar mode target = pre-computed today calendar target only."""
    device = _create_bistate_device(calendar="calendar.test")
    now = datetime(2026, 3, 30, 11, 30, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)
    _set_day_boundaries(device, today_utc, tomorrow_utc)

    device._is_current_calendar_mode = True
    device._today_calendar_target_s = 7200.0  # 2h
    device._today_calendar_past_actual_s = 3600.0  # 1h past completed
    device._constraints = []

    device.update_current_metrics(now)

    assert device.qs_bistate_current_duration_h == pytest.approx(2.0)
    assert device.qs_bistate_current_on_h == pytest.approx(1.0)


def test_calendar_mode_actual_includes_active_constraint_current_value():
    """Calendar mode actual = past events + active constraint current_value."""
    device = _create_bistate_device(calendar="calendar.test")
    now = datetime(2026, 3, 30, 16, 30, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)
    _set_day_boundaries(device, today_utc, tomorrow_utc)

    device._is_current_calendar_mode = True
    device._today_calendar_target_s = 7200.0  # 2h total
    device._today_calendar_past_actual_s = 3600.0  # 1h past event

    # Active constraint for 16:00-17:00, 30 min done
    ct = _make_constraint(
        device,
        now,
        target_s=3600.0,
        current_s=1800.0,
        start=datetime(2026, 3, 30, 16, 0, 0, tzinfo=pytz.UTC),
        end=datetime(2026, 3, 30, 17, 0, 0, tzinfo=pytz.UTC),
    )
    device._constraints = [ct]

    device.update_current_metrics(now)

    assert device.qs_bistate_current_duration_h == pytest.approx(2.0)
    assert device.qs_bistate_current_on_h == pytest.approx(1.5)


def test_calendar_mode_excludes_tomorrow_active_constraint():
    """Calendar mode excludes active constraints ending tomorrow."""
    device = _create_bistate_device(calendar="calendar.test")
    now = datetime(2026, 3, 30, 23, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)
    _set_day_boundaries(device, today_utc, tomorrow_utc)

    device._is_current_calendar_mode = True
    device._today_calendar_target_s = 3600.0
    device._today_calendar_past_actual_s = 3600.0

    # Constraint ending tomorrow — should NOT add current_value
    ct = _make_constraint(
        device,
        now,
        target_s=7200.0,
        current_s=900.0,
        start=datetime(2026, 3, 30, 23, 0, 0, tzinfo=pytz.UTC),
        end=datetime(2026, 3, 31, 1, 0, 0, tzinfo=pytz.UTC),
    )
    device._constraints = [ct]

    device.update_current_metrics(now)

    assert device.qs_bistate_current_duration_h == pytest.approx(1.0)
    assert device.qs_bistate_current_on_h == pytest.approx(1.0)


# =============================================================================
# update_current_metrics: default path
# =============================================================================


def test_default_mode_filters_constraints_to_today():
    """Default mode only sums constraints ending within today."""
    device = _create_bistate_device()
    now = datetime(2026, 3, 30, 11, 30, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)
    _set_day_boundaries(device, today_utc, tomorrow_utc)

    ct_today = _make_constraint(
        device,
        now,
        target_s=3600.0,
        current_s=0.0,
        start=datetime(2026, 3, 30, 16, 0, 0, tzinfo=pytz.UTC),
        end=datetime(2026, 3, 30, 17, 0, 0, tzinfo=pytz.UTC),
    )
    ct_tomorrow = _make_constraint(
        device,
        now,
        target_s=3600.0,
        current_s=0.0,
        start=datetime(2026, 3, 31, 6, 0, 0, tzinfo=pytz.UTC),
        end=datetime(2026, 3, 31, 7, 0, 0, tzinfo=pytz.UTC),
    )
    device._constraints = [ct_today, ct_tomorrow]
    device._last_completed_constraint = None

    device.update_current_metrics(now)

    assert device.qs_bistate_current_duration_h == pytest.approx(1.0)
    assert device.qs_bistate_current_on_h == pytest.approx(0.0)


def test_default_mode_includes_last_completed_from_today():
    """Default mode adds last_completed_constraint if from today."""
    device = _create_bistate_device()
    now = datetime(2026, 3, 30, 11, 30, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)
    _set_day_boundaries(device, today_utc, tomorrow_utc)

    # Active constraint for afternoon
    ct = _make_constraint(
        device,
        now,
        target_s=3600.0,
        current_s=0.0,
        start=datetime(2026, 3, 30, 16, 0, 0, tzinfo=pytz.UTC),
        end=datetime(2026, 3, 30, 17, 0, 0, tzinfo=pytz.UTC),
    )
    device._constraints = [ct]

    # Completed morning constraint
    lcc = _make_constraint(
        device,
        now,
        target_s=3600.0,
        current_s=3600.0,
        start=datetime(2026, 3, 30, 6, 0, 0, tzinfo=pytz.UTC),
        end=datetime(2026, 3, 30, 7, 0, 0, tzinfo=pytz.UTC),
    )
    device._last_completed_constraint = lcc

    device.update_current_metrics(now)

    assert device.qs_bistate_current_duration_h == pytest.approx(2.0)
    assert device.qs_bistate_current_on_h == pytest.approx(1.0)


def test_default_mode_excludes_last_completed_from_yesterday():
    """Default mode ignores last_completed_constraint from previous days."""
    device = _create_bistate_device()
    now = datetime(2026, 3, 30, 11, 30, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)
    _set_day_boundaries(device, today_utc, tomorrow_utc)

    device._constraints = []
    lcc = _make_constraint(
        device,
        now,
        target_s=3600.0,
        current_s=3600.0,
        start=datetime(2026, 3, 29, 6, 0, 0, tzinfo=pytz.UTC),
        end=datetime(2026, 3, 29, 7, 0, 0, tzinfo=pytz.UTC),
    )
    device._last_completed_constraint = lcc

    device.update_current_metrics(now)

    assert device.qs_bistate_current_duration_h == pytest.approx(0.0)
    assert device.qs_bistate_current_on_h == pytest.approx(0.0)


def test_default_mode_excludes_last_completed_with_max_sentinel():
    """Default mode ignores last_completed_constraint with DATETIME_MAX_UTC sentinel."""
    device = _create_bistate_device()
    now = datetime(2026, 3, 30, 11, 30, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)
    _set_day_boundaries(device, today_utc, tomorrow_utc)

    device._constraints = []
    lcc = _make_constraint(
        device,
        now,
        target_s=3600.0,
        current_s=3600.0,
        start=DATETIME_MIN_UTC,
        end=DATETIME_MAX_UTC,
    )
    device._last_completed_constraint = lcc

    device.update_current_metrics(now)

    assert device.qs_bistate_current_duration_h == pytest.approx(0.0)
    assert device.qs_bistate_current_on_h == pytest.approx(0.0)


# =============================================================================
# Edge cases
# =============================================================================


def test_no_constraints_no_completed_shows_zero():
    """No data at all shows 0h for both metrics."""
    device = _create_bistate_device()
    now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)
    _set_day_boundaries(device, today_utc, tomorrow_utc)

    device._constraints = []
    device._last_completed_constraint = None

    device.update_current_metrics(now)

    assert device.qs_bistate_current_duration_h == 0.0
    assert device.qs_bistate_current_on_h == 0.0


def test_tomorrow_only_constraints_show_zero_target():
    """Tomorrow-only constraints show 0h target in default mode."""
    device = _create_bistate_device()
    now = datetime(2026, 3, 30, 10, 23, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)
    _set_day_boundaries(device, today_utc, tomorrow_utc)

    ct = _make_constraint(
        device,
        now,
        target_s=12600.0,
        current_s=0.0,
        start=datetime(2026, 3, 31, 6, 30, 0, tzinfo=pytz.UTC),
        end=datetime(2026, 3, 31, 10, 0, 0, tzinfo=pytz.UTC),
    )
    device._constraints = [ct]
    device._last_completed_constraint = None

    device.update_current_metrics(now)

    assert device.qs_bistate_current_duration_h == pytest.approx(0.0)
    assert device.qs_bistate_current_on_h == pytest.approx(0.0)


def test_calendar_mode_no_active_constraints_shows_precomputed():
    """Calendar mode with no active constraints uses only pre-computed values."""
    device = _create_bistate_device(calendar="calendar.test")
    now = datetime(2026, 3, 30, 8, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)
    _set_day_boundaries(device, today_utc, tomorrow_utc)

    device._is_current_calendar_mode = True
    device._today_calendar_target_s = 10800.0  # 3h
    device._today_calendar_past_actual_s = 3600.0  # 1h past
    device._constraints = []

    device.update_current_metrics(now)

    assert device.qs_bistate_current_duration_h == pytest.approx(3.0)
    assert device.qs_bistate_current_on_h == pytest.approx(1.0)


# =============================================================================
# Calendar pre-computation and sync warning (AC5)
# =============================================================================


async def test_calendar_precomputation_computes_today_metrics():
    """check_load_activity_and_constraints pre-computes calendar metrics for today."""
    device = _create_bistate_device(calendar="calendar.test")
    device.bistate_mode = "bistate_mode_auto"
    now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)

    _set_day_boundaries(device, today_utc, tomorrow_utc)
    device.is_load_command_set = MagicMock(return_value=False)

    # Calendar returns two today events + one tomorrow event
    async def mock_get_events(time=None, give_currently_running_event=True, max_number_of_events=None):
        return [
            (datetime(2026, 3, 30, 6, 0, 0, tzinfo=pytz.UTC), datetime(2026, 3, 30, 7, 0, 0, tzinfo=pytz.UTC)),
            (datetime(2026, 3, 30, 16, 0, 0, tzinfo=pytz.UTC), datetime(2026, 3, 30, 17, 0, 0, tzinfo=pytz.UTC)),
            (datetime(2026, 3, 31, 6, 0, 0, tzinfo=pytz.UTC), datetime(2026, 3, 31, 7, 0, 0, tzinfo=pytz.UTC)),
        ]

    device.get_next_scheduled_events = mock_get_events

    async def mock_build(time, bistate_mode, do_push_constraint_after):
        return []

    device._build_mode_constraint_items = mock_build
    device._last_completed_constraint = None

    await device.check_load_activity_and_constraints(now)

    assert device._is_current_calendar_mode is True
    # Target: two 1h events = 7200s
    assert device._today_calendar_target_s == pytest.approx(7200.0)
    # Past actual: 6:00-7:00 ended before 12:00 = 3600s, 16:00-17:00 not past yet
    assert device._today_calendar_past_actual_s == pytest.approx(3600.0)
    # Metrics should reflect pre-computed values
    assert device.qs_bistate_current_duration_h == pytest.approx(2.0)
    assert device.qs_bistate_current_on_h == pytest.approx(1.0)


async def test_calendar_sync_warning_on_divergent_last_completed():
    """Warning emitted when last_completed current_value diverges from past actual."""
    device = _create_bistate_device(calendar="calendar.test")
    device.bistate_mode = "bistate_mode_auto"
    now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)

    _set_day_boundaries(device, today_utc, tomorrow_utc)
    device.is_load_command_set = MagicMock(return_value=False)

    # Calendar: one past 1h event
    async def mock_get_events(time=None, give_currently_running_event=True, max_number_of_events=None):
        return [(datetime(2026, 3, 30, 6, 0, 0, tzinfo=pytz.UTC), datetime(2026, 3, 30, 7, 0, 0, tzinfo=pytz.UTC))]

    device.get_next_scheduled_events = mock_get_events

    async def mock_build(time, bistate_mode, do_push_constraint_after):
        return []

    device._build_mode_constraint_items = mock_build

    # Divergent last completed: current_value=7200 vs past_actual=3600 (>300s diff)
    lcc = _make_constraint(
        device,
        now,
        target_s=3600.0,
        current_s=7200.0,
        start=datetime(2026, 3, 30, 6, 0, 0, tzinfo=pytz.UTC),
        end=datetime(2026, 3, 30, 7, 0, 0, tzinfo=pytz.UTC),
    )
    device._last_completed_constraint = lcc

    with patch("custom_components.quiet_solar.ha_model.bistate_duration._LOGGER") as mock_logger:
        await device.check_load_activity_and_constraints(now)
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "sync-check" in warning_msg


async def test_no_sync_warning_when_values_close():
    """No warning when last_completed current_value is close to past actual."""
    device = _create_bistate_device(calendar="calendar.test")
    device.bistate_mode = "bistate_mode_auto"
    now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    today_utc = datetime(2026, 3, 30, 0, 0, 0, tzinfo=pytz.UTC)
    tomorrow_utc = datetime(2026, 3, 31, 0, 0, 0, tzinfo=pytz.UTC)

    _set_day_boundaries(device, today_utc, tomorrow_utc)
    device.is_load_command_set = MagicMock(return_value=False)

    async def mock_get_events(time=None, give_currently_running_event=True, max_number_of_events=None):
        return [(datetime(2026, 3, 30, 6, 0, 0, tzinfo=pytz.UTC), datetime(2026, 3, 30, 7, 0, 0, tzinfo=pytz.UTC))]

    device.get_next_scheduled_events = mock_get_events

    async def mock_build(time, bistate_mode, do_push_constraint_after):
        return []

    device._build_mode_constraint_items = mock_build

    # Close to past actual (3600 vs 3650, diff=50 < 300)
    lcc = _make_constraint(
        device,
        now,
        target_s=3600.0,
        current_s=3650.0,
        start=datetime(2026, 3, 30, 6, 0, 0, tzinfo=pytz.UTC),
        end=datetime(2026, 3, 30, 7, 0, 0, tzinfo=pytz.UTC),
    )
    device._last_completed_constraint = lcc

    with patch("custom_components.quiet_solar.ha_model.bistate_duration._LOGGER") as mock_logger:
        await device.check_load_activity_and_constraints(now)
        mock_logger.warning.assert_not_called()


async def test_non_calendar_mode_sets_flag_false():
    """Default mode sets _is_current_calendar_mode to False."""
    device = _create_bistate_device()
    device.bistate_mode = "bistate_mode_default"
    now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)

    device.is_load_command_set = MagicMock(return_value=False)
    device.default_on_duration = 2.0
    device.default_on_finish_time = dt_time(hour=23, minute=0)
    device.get_next_time_from_hours = MagicMock(
        return_value=datetime(2026, 3, 30, 23, 0, 0, tzinfo=pytz.UTC)
    )
    device._constraints = []

    await device.check_load_activity_and_constraints(now)

    assert device._is_current_calendar_mode is False
