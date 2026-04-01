"""Tests for bug #99: idle command not physically enforced on OCPP charger.

Verifies that probe_if_command_set correctly declares desired state before
probing, so that transitioning from active charging to idle triggers
execute_command instead of being silently skipped.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
import pytz

from custom_components.quiet_solar.ha_model.charger import (
    QSChargerGeneric,
    QSStateCmd,
)
from custom_components.quiet_solar.home_model.commands import (
    CMD_AUTO_FROM_CONSIGN,
    CMD_AUTO_GREEN_ONLY,
    CMD_IDLE,
    CMD_OFF,
    CMD_ON,
    copy_command,
)


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_charger_coverage_deep.py)
# ---------------------------------------------------------------------------

from tests.test_charger_coverage_deep import (
    _create_charger,
    _init_charger_states,
    _make_hass,
    _make_home,
    _make_real_car,
    _plug_car,
)


def _setup_charging_charger(
    amperage: int = 28, num_phases: int = 3
) -> tuple[QSChargerGeneric, datetime]:
    """Create a charger that is actively charging at *amperage* A.

    Returns (charger, now).  The charger's expected state mirrors a real
    charger in the middle of an ON constraint: charge_state=True,
    amperage=28A, phases=3.
    """
    hass = _make_hass()
    home = _make_home()
    charger = _create_charger(hass, home, name="Bug99Charger", is_3p=True)
    car = _make_real_car(hass, home, name="Bug99Car")
    now = datetime.now(pytz.UTC)

    _init_charger_states(charger, charge_state=True, amperage=amperage, num_phases=num_phases)
    _plug_car(charger, car, now - timedelta(hours=1))

    # Mock HA-level I/O only
    charger._do_update_charger_state = AsyncMock()
    charger.is_optimistic_plugged = MagicMock(return_value=True)
    charger.is_in_state_reset = MagicMock(return_value=False)
    charger.is_car_stopped_asking_current = MagicMock(return_value=False)
    charger._asked_for_reboot_at_time = None

    # Charger hardware reports: charging at *amperage* A
    charger.is_charge_enabled = MagicMock(return_value=True)
    charger.is_charge_disabled = MagicMock(return_value=False)
    charger.get_charging_current = MagicMock(return_value=amperage)
    charger.update_data_request = AsyncMock()
    charger.stop_charge = AsyncMock()
    charger.set_charging_current = AsyncMock()
    charger.set_charging_num_phases = AsyncMock()

    return charger, now


# ===========================================================================
# Task 3: probe_if_command_set with idle command
# ===========================================================================


class TestProbeIdleWhileCharging:
    """AC2 + AC5: probe idle on an active charger -> returns False."""

    @pytest.mark.asyncio
    async def test_probe_idle_returns_false_when_charging(self):
        """The bug scenario: charging at 28A, idle arrives, probe must return False."""
        charger, now = _setup_charging_charger(amperage=28, num_phases=3)

        with patch.object(
            type(charger), "current_num_phases", new_callable=PropertyMock, return_value=3
        ):
            result = await charger.probe_if_command_set(now, copy_command(CMD_IDLE))

        assert result is False, "probe must detect mismatch and return False so execute_command runs"

    @pytest.mark.asyncio
    async def test_expected_state_declared_before_probing(self):
        """AC5: after probe, expected state reflects idle values."""
        charger, now = _setup_charging_charger(amperage=28, num_phases=3)

        with patch.object(
            type(charger), "current_num_phases", new_callable=PropertyMock, return_value=3
        ):
            await charger.probe_if_command_set(now, copy_command(CMD_IDLE))

        assert charger._expected_charge_state.value is False
        assert charger._expected_amperage.value == charger.charger_default_idle_charge
        assert charger._expected_num_active_phases.value == 3

    @pytest.mark.asyncio
    async def test_probe_off_returns_false_when_charging(self):
        """CMD_OFF is also off_or_idle and should behave identically."""
        charger, now = _setup_charging_charger(amperage=28)

        with patch.object(
            type(charger), "current_num_phases", new_callable=PropertyMock, return_value=3
        ):
            result = await charger.probe_if_command_set(now, copy_command(CMD_OFF))

        assert result is False


class TestProbeIdleWhenAlreadyIdle:
    """AC3: probe idle when charger is already idle -> returns True."""

    @pytest.mark.asyncio
    async def test_probe_idle_returns_true_when_already_idle(self):
        """Optimization preserved: if charger is already idle, skip execute_command."""
        charger, now = _setup_charging_charger(amperage=6, num_phases=3)

        # Charger is already stopped
        idle_charge = charger.charger_default_idle_charge
        _init_charger_states(charger, charge_state=False, amperage=idle_charge, num_phases=3)
        charger.is_charge_enabled = MagicMock(return_value=False)
        charger.is_charge_disabled = MagicMock(return_value=True)
        charger.get_charging_current = MagicMock(return_value=idle_charge)
        charger.is_in_state_reset = MagicMock(return_value=False)

        with patch.object(
            type(charger), "current_num_phases", new_callable=PropertyMock, return_value=3
        ):
            result = await charger.probe_if_command_set(now, copy_command(CMD_IDLE))

        assert result is True, "charger already idle — probe should confirm command is set"


class TestProbeOnAutoUnchanged:
    """AC4: ON/AUTO probe behavior is unchanged (handled=False path)."""

    @pytest.mark.asyncio
    async def test_probe_auto_command_no_state_update(self):
        """AUTO command: handled=False, guarded block never executes."""
        charger, now = _setup_charging_charger(amperage=28)

        # Capture state before probe
        charge_state_before = charger._expected_charge_state.value
        amperage_before = charger._expected_amperage.value

        with patch.object(
            type(charger), "current_num_phases", new_callable=PropertyMock, return_value=3
        ):
            await charger.probe_if_command_set(now, copy_command(CMD_AUTO_GREEN_ONLY))

        # Expected state should NOT have been changed to idle values
        assert charger._expected_charge_state.value == charge_state_before
        assert charger._expected_amperage.value == amperage_before

    @pytest.mark.asyncio
    async def test_probe_on_command_no_state_update(self):
        """ON command: handled=False, no idle-state declaration."""
        charger, now = _setup_charging_charger(amperage=28)

        charge_state_before = charger._expected_charge_state.value
        amperage_before = charger._expected_amperage.value

        with patch.object(
            type(charger), "current_num_phases", new_callable=PropertyMock, return_value=3
        ):
            await charger.probe_if_command_set(now, copy_command(CMD_ON))

        assert charger._expected_charge_state.value == charge_state_before
        assert charger._expected_amperage.value == amperage_before


class TestRepeatedIdleProbeCounters:
    """AC6: repeated idle probes do NOT reset retry counters."""

    @pytest.mark.asyncio
    async def test_second_idle_probe_preserves_retry_counters(self):
        """If _expected_charge_state is already False, set() is a no-op.

        Amps/phases should NOT be re-set (which would call reset() and
        wipe _num_launched counters).
        """
        charger, now = _setup_charging_charger(amperage=6, num_phases=3)

        # Simulate state after first idle declaration: charge=False, amps=idle
        idle_charge = charger.charger_default_idle_charge
        _init_charger_states(charger, charge_state=False, amperage=idle_charge, num_phases=3)
        charger.is_in_state_reset = MagicMock(return_value=False)

        # Simulate retry counters from prior enforcement attempts
        charger._expected_amperage._num_launched = 2
        charger._expected_num_active_phases._num_launched = 1
        amp_num_set_before = charger._expected_amperage._num_set
        phase_num_set_before = charger._expected_num_active_phases._num_set

        # Second idle probe — _expected_charge_state.set(False) returns False (no-op)
        charger._probe_and_enforce_stopped_charge_command_state(now, command=copy_command(CMD_IDLE))

        # Retry counters must be preserved
        assert charger._expected_amperage._num_launched == 2, "amp retry counter must not be wiped"
        assert charger._expected_num_active_phases._num_launched == 1, "phase retry counter must not be wiped"
        assert charger._expected_amperage._num_set == amp_num_set_before, "amp set count must not change"
        assert charger._expected_num_active_phases._num_set == phase_num_set_before, "phase set count must not change"


class TestProbeFollowedByExecute:
    """AC1 partial: after probe returns False, execute_command sends stop."""

    @pytest.mark.asyncio
    async def test_execute_after_failed_probe_sends_stop(self):
        """Full flow: probe -> False -> execute_command -> charger stops."""
        charger, now = _setup_charging_charger(amperage=28, num_phases=3)
        idle_cmd = copy_command(CMD_IDLE)

        with patch.object(
            type(charger), "current_num_phases", new_callable=PropertyMock, return_value=3
        ):
            probe_result = await charger.probe_if_command_set(now, idle_cmd)
            assert probe_result is False

            # Now execute_command (what launch_command does after probe returns False)
            exec_result = await charger.execute_command(now, idle_cmd)

        # execute_command calls stop_charge (or set_charging_current to idle)
        assert exec_result is not None, "execute_command should not return None (car is plugged)"


# ===========================================================================
# Task 4: Integration test — SOC reached -> idle -> charger stops
# ===========================================================================


class TestFullIdleEnforcementFlow:
    """AC1: end-to-end — idle command arrives while charging, charger physically stops."""

    @pytest.mark.asyncio
    async def test_idle_command_physically_stops_charger(self):
        """Simulate the bug scenario end-to-end through probe + execute."""
        charger, now = _setup_charging_charger(amperage=28, num_phases=3)
        idle_cmd = copy_command(CMD_IDLE)

        with patch.object(
            type(charger), "current_num_phases", new_callable=PropertyMock, return_value=3
        ):
            # Step 1: probe (what launch_command calls first)
            is_set = await charger.probe_if_command_set(now, idle_cmd)
            assert is_set is False, "idle NOT already set — charger is still charging at 28A"

            # Step 2: execute_command (what launch_command calls when probe returns False)
            # After execute, expected state is cleared by _reset_state_machine
            # and re-declared by _probe_and_enforce_stopped_charge_command_state(probe_only=False)
            # then _ensure_correct_state(probe_only=False) sends the actual stop.
            await charger.execute_command(now, idle_cmd)

        # Verify a hardware command was sent (stop_charge or set_charging_current)
        hardware_cmds_sent = (
            charger.stop_charge.call_count + charger.set_charging_current.call_count
        )
        assert hardware_cmds_sent > 0, "OCPP stop or current-reduce command must be sent"
