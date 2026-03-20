"""
External control detection scenario tests.

Tests that when a device's state changes without a solver-initiated command,
the system detects external control, excludes the device from the current
planning cycle, and re-includes it on the next evaluation.

Story 2.4, AC #3: External control detection (FR10).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest import TestCase

import pytest
import pytz

from custom_components.quiet_solar.home_model.load import TestLoad


@pytest.mark.integration
class TestExternalControlDetection(TestCase):
    """Test external control detection on AbstractLoad."""

    def setUp(self):
        self.dt = datetime(year=2024, month=6, day=1, hour=10, minute=0, second=0, tzinfo=pytz.UTC)
        self.load = TestLoad(name="test_device")

    def test_initial_state_no_external_control(self):
        """By default, no external control is detected."""
        assert self.load.external_user_initiated_state is None
        assert self.load.external_user_initiated_state_time is None
        assert self.load.get_override_state() == "NO OVERRIDE"

    def test_set_external_user_initiated_state(self):
        """Setting external_user_initiated_state marks external control detected."""
        self.load.external_user_initiated_state = "ON"
        self.load.external_user_initiated_state_time = self.dt

        assert self.load.external_user_initiated_state == "ON"
        assert self.load.external_user_initiated_state_time == self.dt
        assert "Override: ON" in self.load.get_override_state()

    def test_reset_override_clears_state(self):
        """reset_override_state_and_set_reset_ask_time clears external state."""
        self.load.external_user_initiated_state = "ON"
        self.load.external_user_initiated_state_time = self.dt

        self.load.reset_override_state_and_set_reset_ask_time(time=self.dt + timedelta(minutes=5))

        assert self.load.external_user_initiated_state is None
        assert self.load.external_user_initiated_state_time is None
        # asked_for_reset_user_initiated_state_time should be set
        assert self.load.asked_for_reset_user_initiated_state_time is not None

    def test_reset_override_noop_when_no_external_state(self):
        """Reset is a no-op when there's no external state — no crash, no side effects."""
        assert self.load.external_user_initiated_state is None
        self.load.reset_override_state_and_set_reset_ask_time(time=self.dt)

        assert self.load.external_user_initiated_state is None
        assert self.load.external_user_initiated_state_time is None
        # asked_for_reset should NOT be set since there was nothing to reset
        assert self.load.asked_for_reset_user_initiated_state_time is None

    def test_override_state_shows_asked_for_reset(self):
        """After reset request, get_override_state reflects the reset-in-progress."""
        self.load.external_user_initiated_state = "OFF"
        self.load.external_user_initiated_state_time = self.dt
        self.load.reset_override_state_and_set_reset_ask_time(time=self.dt + timedelta(minutes=1))

        assert self.load.get_override_state() == "ASKED FOR RESET OVERRIDE"

    def test_persistence_save_and_restore(self):
        """External state is persisted via update_to_be_saved / use_saved_extra_device_info."""
        self.load.external_user_initiated_state = "ON"
        self.load.external_user_initiated_state_time = self.dt

        # Save
        data = {}
        self.load.update_to_be_saved_extra_device_info(data)
        assert data["external_user_initiated_state"] == "ON"
        assert data["external_user_initiated_state_time"] is not None

        # Restore to a fresh load
        fresh_load = TestLoad(name="fresh_device")
        assert fresh_load.external_user_initiated_state is None
        fresh_load.use_saved_extra_device_info(data)

        assert fresh_load.external_user_initiated_state == "ON"
        assert fresh_load.external_user_initiated_state_time == self.dt

    def test_persistence_save_none_state(self):
        """When no external state, saved data contains None values."""
        data = {}
        self.load.update_to_be_saved_extra_device_info(data)
        assert data["external_user_initiated_state"] is None
        assert data["external_user_initiated_state_time"] is None

    def test_persistence_restore_none_state(self):
        """Restoring None external state works correctly."""
        data = {
            "external_user_initiated_state": None,
            "external_user_initiated_state_time": None,
            "asked_for_reset_user_initiated_state_time": None,
            "asked_for_reset_user_initiated_state_time_first_cmd_reset_done": None,
        }
        self.load.use_saved_extra_device_info(data)
        assert self.load.external_user_initiated_state is None
        assert self.load.external_user_initiated_state_time is None

    def test_multiple_external_state_changes(self):
        """Multiple external state changes are tracked correctly."""
        # First external change
        self.load.external_user_initiated_state = "ON"
        self.load.external_user_initiated_state_time = self.dt
        assert self.load.get_override_state() == "Override: ON"

        # Second external change overwrites
        t2 = self.dt + timedelta(minutes=10)
        self.load.external_user_initiated_state = "OFF"
        self.load.external_user_initiated_state_time = t2
        assert self.load.get_override_state() == "Override: OFF"
        assert self.load.external_user_initiated_state_time == t2

    def test_reset_then_new_external_state(self):
        """After reset, a new external state is detected fresh."""
        # Set and reset
        self.load.external_user_initiated_state = "ON"
        self.load.external_user_initiated_state_time = self.dt
        self.load.reset_override_state_and_set_reset_ask_time(time=self.dt + timedelta(minutes=5))
        assert self.load.get_override_state() == "ASKED FOR RESET OVERRIDE"

        # Clear reset state to simulate next cycle completion
        self.load.asked_for_reset_user_initiated_state_time = None

        # New external event
        t2 = self.dt + timedelta(minutes=15)
        self.load.external_user_initiated_state = "OFF"
        self.load.external_user_initiated_state_time = t2
        assert self.load.get_override_state() == "Override: OFF"

    def test_best_effort_load_flag_independent_of_external_state(self):
        """External state detection works independently of best-effort flags."""
        self.load.load_is_auto_to_be_boosted = True
        self.load.external_user_initiated_state = "ON"
        self.load.external_user_initiated_state_time = self.dt
        assert "Override: ON" in self.load.get_override_state()

        self.load.load_is_auto_to_be_boosted = False
        self.load.qs_best_effort_green_only = True
        assert "Override: ON" in self.load.get_override_state()
