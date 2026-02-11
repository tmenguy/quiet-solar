"""Additional tests for QSBiStateDuration to reach 95% coverage.

Targets the specific uncovered lines:
70-73, 77-78, 81-84, 91, 265, 269, 283, 286, 303, 373-379, 426, 429, 433, 435
"""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import time as dt_time, timedelta

import pytz

from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_NAME, STATE_UNKNOWN, STATE_UNAVAILABLE

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.ha_model.bistate_duration import (
    QSBiStateDuration,
    bistate_modes,
    MAX_USER_OVERRIDE_DURATION_S,
    USER_OVERRIDE_STATE_BACK_DURATION_S,
)
from custom_components.quiet_solar.home_model.commands import (
    CMD_ON,
    CMD_OFF,
    CMD_IDLE,
)
from custom_components.quiet_solar.home_model.constraints import (
    TimeBasedSimplePowerLoadConstraint,
    DATETIME_MAX_UTC,
)
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    CONSTRAINT_TYPE_FILLER_AUTO,
    CONF_SWITCH,
)

from tests.factories import create_minimal_home_model


# ---------------------------------------------------------------------------
# Concrete test subclass
# ---------------------------------------------------------------------------

class ConcreteBiState(QSBiStateDuration):
    """Concrete implementation for testing."""

    def __init__(self, **kwargs):
        if "switch_entity" in kwargs:
            kwargs[CONF_SWITCH] = kwargs.pop("switch_entity")
        elif CONF_SWITCH not in kwargs:
            kwargs[CONF_SWITCH] = "switch.test_device"
        super().__init__(**kwargs)

    async def execute_command_system(self, time, command, state):
        return True

    def get_virtual_current_constraint_translation_key(self):
        return "test_key"

    def get_select_translation_key(self):
        return "test_select_key"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg_entry(hass):
    """Config entry added to hass."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        source=SOURCE_USER,
        entry_id="test_bistate_cov",
        data={CONF_NAME: "Cov BiState"},
        title="Cov BiState",
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def home():
    """Minimal home model mock."""
    h = create_minimal_home_model()
    h.force_next_solve = MagicMock()
    h.update_all_states = AsyncMock()
    return h


@pytest.fixture
def setup(hass, cfg_entry, home):
    """Wire up domain data handler."""
    dh = MagicMock()
    dh.home = home
    dh.hass = hass
    hass.data.setdefault(DOMAIN, {})[DATA_HANDLER] = dh
    return {"entry": cfg_entry, "home": home}


@pytest.fixture
def device(hass, setup):
    """Standard device with power=1000, mode_off support, constraints enabled."""
    dev = ConcreteBiState(
        hass=hass,
        config_entry=setup["entry"],
        home=setup["home"],
        switch_entity="switch.test_device",
        **{CONF_NAME: "Test Device"},
    )
    dev.power_use = 1000.0
    dev.load_is_auto_to_be_boosted = False
    dev.externally_initialized_constraints = True
    dev._constraints = []
    return dev


# ===================================================================
# Lines 91: power_use returns None when _power_use_conf is None
# ===================================================================

class TestPowerUseNone:

    def test_power_use_returns_none_when_conf_is_none(self, hass, setup):
        """When _power_use_conf is None, power_use should return None."""
        dev = ConcreteBiState(
            hass=hass,
            config_entry=setup["entry"],
            home=setup["home"],
            **{CONF_NAME: "NoPower"},
        )
        # Don't set power_use -> _power_use_conf remains None
        assert dev.power_use is None


# ===================================================================
# Lines 70-73: user_set_default_on_duration with for_init=False
# ===================================================================

class TestUserSetDefaultOnDuration:

    @pytest.mark.asyncio
    async def test_sets_duration_and_triggers_solve(self, device):
        """Calling user_set_default_on_duration triggers constraint check and force_next_solve.

        Covers lines 70-73 including line 72 (force_next_solve call).
        """
        # Use mode_on so that check_load_activity_and_constraints returns True
        device.bistate_mode = "bistate_mode_on"
        device.is_load_command_set = MagicMock(return_value=False)
        device.get_proper_local_adapted_tomorrow = MagicMock(
            return_value=datetime.datetime.now(pytz.UTC) + timedelta(hours=24)
        )

        await device.user_set_default_on_duration(3.5, for_init=False)

        assert device.default_on_duration == 3.5
        # force_next_solve should have been called (line 72)
        device.home.force_next_solve.assert_called_once()
        # update_all_states should have been awaited (line 73)
        device.home.update_all_states.assert_awaited_once()


# ===================================================================
# Lines 77-78: user_set_bistate_mode with invalid mode
# Lines 81-84: user_set_bistate_mode with valid mode, for_init=False
# ===================================================================

class TestUserSetBistateMode:

    @pytest.mark.asyncio
    async def test_invalid_mode_logs_error_and_returns(self, device):
        """Invalid mode should be rejected without changing bistate_mode."""
        original = device.bistate_mode
        await device.user_set_bistate_mode("totally_invalid_mode", for_init=False)
        assert device.bistate_mode == original  # unchanged

    @pytest.mark.asyncio
    async def test_valid_mode_triggers_solve(self, device):
        """Valid mode should set bistate_mode and trigger a solve."""
        device.bistate_mode = "bistate_mode_auto"
        device.is_load_command_set = MagicMock(return_value=False)

        await device.user_set_bistate_mode("bistate_mode_default", for_init=False)

        assert device.bistate_mode == "bistate_mode_default"
        device.home.update_all_states.assert_awaited_once()


# ===================================================================
# Line 265: override state unchanged (current == external_user_initiated)
# Line 269: override state changed (current != external_user_initiated)
# ===================================================================

class TestOverrideStateChangedDetection:

    @pytest.mark.asyncio
    async def test_override_state_same_keeps_override(self, hass, device):
        """When HA state matches external override, no state change detected."""
        device.is_load_command_set = MagicMock(return_value=True)
        device.external_user_initiated_state = "on"
        device.external_user_initiated_state_time = datetime.datetime.now(pytz.UTC)
        device.asked_for_reset_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        # HA state matches the override
        hass.states.async_set("switch.test_device", "on")

        # Create an existing override constraint so the code finds it
        override_ct = TimeBasedSimplePowerLoadConstraint(
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            degraded_type=CONSTRAINT_TYPE_FILLER_AUTO,
            time=datetime.datetime.now(pytz.UTC),
            load=device,
            load_param="on",
            load_info={"originator": "user_override"},
            from_user=True,
            end_of_constraint=datetime.datetime.now(pytz.UTC) + timedelta(hours=2),
            power=1000.0,
            initial_value=0,
            target_value=7200.0,
        )
        device._constraints = [override_ct]

        device.bistate_mode = "bistate_mode_auto"
        device.get_next_scheduled_events = AsyncMock(return_value=[])

        time = datetime.datetime.now(pytz.UTC)
        result = await device.check_load_activity_and_constraints(time)

        # Override state didn't change, so no override detection logic fires
        assert device.external_user_initiated_state == "on"  # unchanged

    @pytest.mark.asyncio
    async def test_override_state_changed_triggers_new_override(self, hass, device):
        """When HA state differs from external override, new override is created."""
        time = datetime.datetime.now(pytz.UTC)
        device.is_load_command_set = MagicMock(return_value=True)
        device.external_user_initiated_state = "on"
        device.external_user_initiated_state_time = time
        device.asked_for_reset_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        # HA state differs from override
        hass.states.async_set("switch.test_device", "off")

        device._constraints = []
        device.bistate_mode = "bistate_mode_auto"
        device.get_next_scheduled_events = AsyncMock(return_value=[])

        result = await device.check_load_activity_and_constraints(time)

        # Override should now be "off" (the new state)
        assert device.external_user_initiated_state == "off"


# ===================================================================
# Line 283: expected_state is None -> fallback to expected_state_running
# Line 286: expected_state_running is None -> fallback to expected_state
# ===================================================================

class TestExpectedStateFallbacks:

    @pytest.mark.asyncio
    async def test_fallback_expected_state_to_running(self, hass, device):
        """When current_command is None, expected_state falls back to running_command's state.

        Covers line 283: expected_state = expected_state_running
        """
        device.is_load_command_set = MagicMock(return_value=True)
        device.external_user_initiated_state = None
        device.external_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        # current_command None -> expected_state stays None
        # running_command set -> expected_state_running gets set
        device.current_command = None
        device.running_command = CMD_ON

        # HA state matches expected_state_running ("on")
        hass.states.async_set("switch.test_device", "on")
        device.bistate_mode = "bistate_mode_auto"
        device.get_next_scheduled_events = AsyncMock(return_value=[])

        time = datetime.datetime.now(pytz.UTC)
        result = await device.check_load_activity_and_constraints(time)

        # Should NOT trigger override because state matches after fallback
        assert device.external_user_initiated_state is None

    @pytest.mark.asyncio
    async def test_fallback_running_to_expected(self, hass, device):
        """When running_command is None, expected_state_running falls back to current_command's state.

        Covers line 286: expected_state_running = expected_state
        """
        device.is_load_command_set = MagicMock(return_value=True)
        device.external_user_initiated_state = None
        device.external_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        # current_command set -> expected_state gets set to "off"
        # running_command None -> expected_state_running stays None, falls back
        device.current_command = CMD_OFF
        device.running_command = None

        # HA state matches expected_state ("off")
        hass.states.async_set("switch.test_device", "off")
        device.bistate_mode = "bistate_mode_auto"
        device.get_next_scheduled_events = AsyncMock(return_value=[])

        time = datetime.datetime.now(pytz.UTC)
        result = await device.check_load_activity_and_constraints(time)

        # Should NOT trigger override because state matches after fallback
        assert device.external_user_initiated_state is None


# ===================================================================
# Line 303: asked_for_reset timeout expires -> clear reset time
# ===================================================================

class TestResetTimeoutExpires:

    @pytest.mark.asyncio
    async def test_reset_window_expired_clears_asked_time(self, hass, device):
        """After reset window expires, asked_for_reset_user_initiated_state_time is cleared.

        Covers line 303: self.asked_for_reset_user_initiated_state_time = None
        """
        time = datetime.datetime.now(pytz.UTC)
        device.is_load_command_set = MagicMock(return_value=True)
        device.external_user_initiated_state = None
        device.external_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        # Set asked_for_reset time far enough in the past (> USER_OVERRIDE_STATE_BACK_DURATION_S)
        device.asked_for_reset_user_initiated_state_time = (
            time - timedelta(seconds=USER_OVERRIDE_STATE_BACK_DURATION_S + 10)
        )

        device.current_command = CMD_OFF
        device.running_command = None
        # HA state differs from expected -> would trigger override
        hass.states.async_set("switch.test_device", "on")

        device.bistate_mode = "bistate_mode_auto"
        device.get_next_scheduled_events = AsyncMock(return_value=[])

        result = await device.check_load_activity_and_constraints(time)

        # The reset window has expired, so asked_for_reset time should be cleared
        assert device.asked_for_reset_user_initiated_state_time is None
        # And override should have been detected
        assert device.external_user_initiated_state == "on"


# ===================================================================
# Lines 373-379: mode_off with do_push_constraint_after, no found_override
# ===================================================================

class TestModeOffOverrideBranches:

    @pytest.mark.asyncio
    async def test_mode_off_push_override_constraint_not_found_in_list(self, hass, device):
        """In mode_off, when override constraint was just built but not in _constraints.

        Covers lines 373-375: push override_constraint via set_live_constraints.
        """
        time = datetime.datetime.now(pytz.UTC)

        device.bistate_mode = "bistate_mode_off"
        device.is_load_command_set = MagicMock(return_value=True)
        device.external_user_initiated_state = "on"
        device.external_user_initiated_state_time = time
        device.asked_for_reset_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        # HA state changed from override -> triggers new override detection
        hass.states.async_set("switch.test_device", "off")

        # No existing constraints
        device._constraints = []
        device.set_live_constraints = MagicMock()

        result = await device.check_load_activity_and_constraints(time)

        # Override was detected (state changed off != on), external state updated
        # Then mode_off logic: do_push_constraint_after is set from idle override
        # The idle override path resets constraints (command_and_constraint_reset)
        # and do_push_constraint_after is set
        assert result is True

    @pytest.mark.asyncio
    async def test_mode_off_no_override_constraint_resets(self, hass, device):
        """In mode_off with do_push_constraint_after but no override_constraint.

        Covers lines 376-379: else branch when override_constraint is None.
        """
        time = datetime.datetime.now(pytz.UTC)

        device.bistate_mode = "bistate_mode_off"
        device.is_load_command_set = MagicMock(return_value=True)
        # Simulate an idle override: external state == idle state ("off")
        device.external_user_initiated_state = "off"
        device.external_user_initiated_state_time = time
        device.asked_for_reset_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        # HA state matches override (no state change)
        hass.states.async_set("switch.test_device", "off")

        # Put a non-override constraint in the list
        non_override_ct = TimeBasedSimplePowerLoadConstraint(
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            degraded_type=CONSTRAINT_TYPE_FILLER_AUTO,
            time=time,
            load=device,
            load_param="on",
            load_info={"originator": "system"},
            from_user=False,
            end_of_constraint=time + timedelta(hours=1),
            power=1000.0,
            initial_value=0,
            target_value=1800.0,
        )
        device._constraints = [non_override_ct]
        # Don't mock command_and_constraint_reset - let it actually clear constraints
        # to avoid None entries in _constraints when update_current_metrics runs.

        result = await device.check_load_activity_and_constraints(time)

        # Since idle override -> do_push_constraint_after is set,
        # but no user_override constraint found and override_constraint is None
        # -> lines 377-379 path executes
        assert result is True


# ===================================================================
# Lines 426, 429, 433, 435: calendar event filtering in auto mode
# ===================================================================

class TestCalendarEventFiltering:

    @pytest.mark.asyncio
    async def test_event_end_before_push_constraint_after_skipped(self, hass, device):
        """Calendar events ending before do_push_constraint_after are skipped.

        Covers line 426: continue when end_schedule < do_push_constraint_after.
        """
        time = datetime.datetime.now(pytz.UTC)

        device.bistate_mode = "bistate_mode_auto"
        device.is_load_command_set = MagicMock(return_value=True)

        # Set up an idle override to create do_push_constraint_after
        device.external_user_initiated_state = "off"
        device.external_user_initiated_state_time = time
        device.asked_for_reset_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        # HA state matches override
        hass.states.async_set("switch.test_device", "off")
        device._constraints = []

        push_after = device.external_user_initiated_state_time + timedelta(
            seconds=3600.0 * device.override_duration
        )

        # Event that ends BEFORE do_push_constraint_after -> skipped (line 426)
        early_event = (
            time + timedelta(hours=1),
            push_after - timedelta(seconds=10),
        )
        # Valid event that ends after push_after
        valid_event = (
            push_after + timedelta(seconds=1),
            push_after + timedelta(hours=3),
        )
        device.get_next_scheduled_events = AsyncMock(
            return_value=[early_event, valid_event]
        )

        result = await device.check_load_activity_and_constraints(time)

        # At least the valid event should produce a constraint
        assert len(device._constraints) >= 1

    @pytest.mark.asyncio
    async def test_event_end_before_current_time_skipped(self, hass, device):
        """Calendar events ending at or before current time are skipped.

        Covers line 429: continue when end_schedule <= time.
        """
        time = datetime.datetime.now(pytz.UTC)

        device.bistate_mode = "bistate_mode_auto"
        device.is_load_command_set = MagicMock(return_value=False)
        device._constraints = []

        # Event that already ended
        past_event = (
            time - timedelta(hours=2),
            time - timedelta(seconds=1),
        )
        # Valid future event
        future_event = (
            time + timedelta(hours=1),
            time + timedelta(hours=3),
        )
        device.get_next_scheduled_events = AsyncMock(
            return_value=[past_event, future_event]
        )

        result = await device.check_load_activity_and_constraints(time)

        # Only the future event should create a constraint
        assert len(device._constraints) == 1

    @pytest.mark.asyncio
    async def test_event_start_adjusted_by_push_constraint_after(self, hass, device):
        """Calendar event start_schedule is adjusted when do_push_constraint_after is set.

        Covers line 433: start_schedule = max(do_push_constraint_after, start_schedule).
        """
        time = datetime.datetime.now(pytz.UTC)

        device.bistate_mode = "bistate_mode_exact_calendar"
        device.is_load_command_set = MagicMock(return_value=True)

        # Idle override -> creates do_push_constraint_after
        device.external_user_initiated_state = "off"
        device.external_user_initiated_state_time = time
        device.asked_for_reset_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        hass.states.async_set("switch.test_device", "off")
        device._constraints = []

        push_after = device.external_user_initiated_state_time + timedelta(
            seconds=3600.0 * device.override_duration
        )

        # Valid event after push_after (start is clamped but still < end)
        event_valid = (
            time + timedelta(hours=1),  # original start before push_after
            push_after + timedelta(hours=3),  # end well after push_after
        )

        device.get_next_scheduled_events = AsyncMock(
            return_value=[event_valid]
        )

        result = await device.check_load_activity_and_constraints(time)
        assert result is True

    @pytest.mark.asyncio
    async def test_event_start_ge_end_after_clamping_skipped(self, hass, device):
        """Event is skipped when start_schedule >= end_schedule after do_push_constraint_after clamp.

        Covers line 435: continue when start_schedule >= end_schedule.
        """
        time = datetime.datetime.now(pytz.UTC)

        # Use auto mode - start_schedule gets set to do_push_constraint_after at line 439
        device.bistate_mode = "bistate_mode_auto"
        device.is_load_command_set = MagicMock(return_value=True)

        # Idle override -> creates do_push_constraint_after
        device.external_user_initiated_state = "off"
        device.external_user_initiated_state_time = time
        device.asked_for_reset_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        hass.states.async_set("switch.test_device", "off")
        device._constraints = []

        push_after = device.external_user_initiated_state_time + timedelta(
            seconds=3600.0 * device.override_duration
        )

        # Event where end == do_push_constraint_after exactly:
        # - Passes line 426 (end NOT < push_after, it's equal)
        # - Passes line 429 (end > time since push_after is in the future)
        # - Line 433: start = max(push_after, original_start) = push_after
        # - Line 439 (auto mode): start = do_push_constraint_after = push_after
        # - Line 435: start (push_after) >= end (push_after) -> True -> continue!
        event_exact_boundary = (
            time + timedelta(hours=1),  # original start
            push_after,  # end exactly at push_after
        )

        # Add a valid event too so we can distinguish
        event_valid = (
            push_after + timedelta(seconds=10),
            push_after + timedelta(hours=3),
        )

        device.get_next_scheduled_events = AsyncMock(
            return_value=[event_exact_boundary, event_valid]
        )

        result = await device.check_load_activity_and_constraints(time)

        # Only the valid event should produce a constraint (boundary event skipped)
        assert len(device._constraints) == 1
        assert result is True


# ===================================================================
# Lines 373-375 specifically: mode_off with override just detected
# ===================================================================

class TestModeOffWithFreshOverride:

    @pytest.mark.asyncio
    async def test_mode_off_fresh_on_override_pushes_via_set_live(self, hass, device):
        """In mode_off, when push_live_constraint fails but override_constraint exists.

        Specifically hits lines 373-375 by:
        1. Override detected (state changed) -> override_constraint is created
        2. push_live_constraint returns False -> constraint NOT in _constraints
        3. do_push_constraint_after is set from override_constraint.end_of_constraint
        4. Mode is off -> mode_off branch
        5. _constraints is empty -> found_override = False
        6. override_constraint is not None -> lines 373-375 execute
        """
        time = datetime.datetime.now(pytz.UTC)

        device.bistate_mode = "bistate_mode_off"
        device.is_load_command_set = MagicMock(return_value=True)

        # No prior override; current command is OFF but HA state is ON -> override detected
        device.external_user_initiated_state = None
        device.external_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None
        device.current_command = CMD_OFF
        device.running_command = None

        hass.states.async_set("switch.test_device", "on")
        device._constraints = []

        # Mock push_live_constraint to return False so override is NOT added to _constraints
        # but override_constraint variable is still set
        device.push_live_constraint = MagicMock(return_value=False)
        device.set_live_constraints = MagicMock()

        result = await device.check_load_activity_and_constraints(time)

        # Override detected: external_user_initiated_state should now be "on"
        assert device.external_user_initiated_state == "on"
        assert result is True
        # set_live_constraints should have been called with override_constraint (lines 374-375)
        device.set_live_constraints.assert_called_once()


# ===================================================================
# Additional: mode_on and mode_default constraint building (lines 391-417)
# These are already covered but let's ensure push_live_constraint path
# ===================================================================

class TestConstraintPushPaths:

    @pytest.mark.asyncio
    async def test_mode_on_pushes_live_constraint(self, hass, device):
        """Mode_on should push constraint via push_live_constraint (non-agenda).

        Ensures the push_live_constraint path at line 475 is hit.
        """
        device.bistate_mode = "bistate_mode_on"
        device.is_load_command_set = MagicMock(return_value=False)
        device.get_proper_local_adapted_tomorrow = MagicMock(
            return_value=datetime.datetime.now(pytz.UTC) + timedelta(hours=24)
        )
        device._constraints = []

        time = datetime.datetime.now(pytz.UTC)
        result = await device.check_load_activity_and_constraints(time)

        assert result is True
        assert len(device._constraints) >= 1
        ct = device._constraints[0]
        assert ct.target_value == 25 * 3600.0

    @pytest.mark.asyncio
    async def test_mode_default_pushes_live_constraint(self, hass, device):
        """Mode_default should push constraint via push_live_constraint (non-agenda).

        Ensures the push_live_constraint path at line 475 is hit for default mode.
        """
        device.bistate_mode = "bistate_mode_default"
        device.default_on_duration = 2.0
        device.default_on_finish_time = dt_time(hour=7, minute=0, second=0)
        device.is_load_command_set = MagicMock(return_value=False)
        device.get_next_time_from_hours = MagicMock(
            return_value=datetime.datetime.now(pytz.UTC) + timedelta(hours=8)
        )
        device._constraints = []

        time = datetime.datetime.now(pytz.UTC)
        result = await device.check_load_activity_and_constraints(time)

        assert result is True
        assert len(device._constraints) >= 1
        ct = device._constraints[0]
        assert ct.target_value == 2.0 * 3600.0
