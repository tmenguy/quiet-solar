"""Tests for bug #74: exact-calendar constraint beyond day-window shows 0h target.

Verifies that update_current_metrics in bistate_duration.py:
- Always includes active constraints (no day-window filter)
- Only applies day-window filter to _last_completed_constraint fallback
- Preserves existing behavior for pool constraints (regression guard)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from datetime import time as dt_time
from unittest.mock import MagicMock

import pytest
import pytz

from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
from custom_components.quiet_solar.home_model.constraints import (
    DATETIME_MIN_UTC,
    TimeBasedSimplePowerLoadConstraint,
)


# =============================================================================
# Test helpers
# =============================================================================


def _create_bistate_device():
    """Create a minimal concrete bistate device for testing metrics."""
    from custom_components.quiet_solar.const import CONF_SWITCH

    hass = MagicMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    home = MagicMock()
    config_entry = MagicMock()
    config_entry.entry_id = "test_bug74"
    config_entry.data = {}

    class ConcreteBiState(QSBiStateDuration):
        def __init__(self, **kwargs):
            if CONF_SWITCH not in kwargs:
                kwargs[CONF_SWITCH] = "switch.test_device"
            super().__init__(**kwargs)

        async def execute_command_system(self, time, command, state):
            return True

        def get_virtual_current_constraint_translation_key(self):
            return "test_key"

        def get_select_translation_key(self):
            return "test_select_key"

    device = ConcreteBiState(
        hass=hass,
        config_entry=config_entry,
        home=home,
        name="Test Climate",
    )
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


# =============================================================================
# Task 3.1: Active constraint beyond day-window boundary (THE BUG)
# =============================================================================


def test_active_constraint_beyond_day_window_shows_target():
    """Active calendar constraint starting after day-window boundary must appear in metrics.

    Scenario from bug #74:
    - Current time: 10:23
    - default_on_finish_time: midnight (00:00)
    - end_day = next midnight = tomorrow 00:00
    - Calendar constraint: tomorrow 06:30-10:00 (3h30 target)
    - BUG: both start and end are after end_day, so the filter excluded it
    - FIX: active constraints skip the day-window filter entirely
    """
    device = _create_bistate_device()
    # March 30 at 10:23 UTC
    now = datetime(2026, 3, 30, 10, 23, 0, tzinfo=pytz.UTC)

    # Calendar constraint: tomorrow 06:30 to 10:00
    tomorrow_start = datetime(2026, 3, 31, 6, 30, 0, tzinfo=pytz.UTC)
    tomorrow_end = datetime(2026, 3, 31, 10, 0, 0, tzinfo=pytz.UTC)
    target_s = (tomorrow_end - tomorrow_start).total_seconds()  # 3h30 = 12600s

    ct = _make_constraint(device, now, target_s=target_s, current_s=0.0, start=tomorrow_start, end=tomorrow_end)

    device._constraints = [ct]
    device._last_completed_constraint = None

    device.update_current_metrics(now)

    assert device.qs_bistate_current_duration_h == pytest.approx(target_s / 3600.0), (
        f"Should show 3.5h target, got {device.qs_bistate_current_duration_h}"
    )
    assert device.qs_bistate_current_on_h == 0.0


# =============================================================================
# Task 3.2: Active constraint within day-window (regression guard)
# =============================================================================


def test_active_constraint_within_day_window_still_works():
    """Active constraint within the day window should still show in metrics."""
    device = _create_bistate_device()
    now = datetime(2026, 3, 30, 10, 0, 0, tzinfo=pytz.UTC)

    # Constraint today: 14:00-17:00 (well within next midnight boundary)
    start = datetime(2026, 3, 30, 14, 0, 0, tzinfo=pytz.UTC)
    end = datetime(2026, 3, 30, 17, 0, 0, tzinfo=pytz.UTC)
    target_s = (end - start).total_seconds()  # 3h = 10800s

    ct = _make_constraint(device, now, target_s=target_s, current_s=1800.0, start=start, end=end)

    device._constraints = [ct]
    device._last_completed_constraint = None

    device.update_current_metrics(now)

    assert device.qs_bistate_current_duration_h == pytest.approx(3.0)
    assert device.qs_bistate_current_on_h == pytest.approx(0.5)


# =============================================================================
# Task 3.3: No active constraints, last_completed within window
# =============================================================================


def test_last_completed_within_day_window_shows_completed():
    """Completed constraint within day window should show as fallback."""
    device = _create_bistate_device()
    now = datetime(2026, 3, 30, 18, 0, 0, tzinfo=pytz.UTC)

    # Completed constraint: started 6h ago, ends in 6h (within day window)
    start = now - timedelta(hours=6)
    end = now + timedelta(hours=6)

    ct = _make_constraint(device, now, target_s=3 * 3600.0, current_s=3 * 3600.0, start=start, end=end)

    device._constraints = []
    device._last_completed_constraint = ct

    device.update_current_metrics(now)

    assert device.qs_bistate_current_duration_h == pytest.approx(3.0)
    assert device.qs_bistate_current_on_h == pytest.approx(3.0)


# =============================================================================
# Task 3.4: No active constraints, last_completed outside window
# =============================================================================


def test_last_completed_outside_day_window_shows_zero():
    """Completed constraint outside day window should show 0 (stale data filtered)."""
    device = _create_bistate_device()
    now = datetime(2026, 3, 30, 18, 0, 0, tzinfo=pytz.UTC)

    # Completed constraint from 2 days ago — entirely outside the day window
    old_start = now - timedelta(days=2)
    old_end = now - timedelta(days=2) + timedelta(hours=3)

    ct = _make_constraint(device, now, target_s=3 * 3600.0, current_s=3 * 3600.0, start=old_start, end=old_end)

    device._constraints = []
    device._last_completed_constraint = ct

    device.update_current_metrics(now)

    assert device.qs_bistate_current_duration_h == 0.0
    assert device.qs_bistate_current_on_h == 0.0
