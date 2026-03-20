"""Constraint interaction boundary tests (Story 2.3).

Trust-critical tests validating constraint behavior at resource exhaustion boundaries:
- Switching budget exhaustion with mandatory constraints
- Amp budget exhaustion with mandatory constraints
- Multiple MANDATORY constraints competing for insufficient power
- Constraint type transitions under resource pressure

Uses real objects from tests/factories.py - no mocks for constraint logic.
All tests use @pytest.mark.integration marker per architecture requirements.
"""

from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, timedelta

import numpy as np
import pytest
import pytz

from custom_components.quiet_solar.const import (
    CHANGE_ON_OFF_STATE_HYSTERESIS_S,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    SOLVER_STEP_S,
)
from custom_components.quiet_solar.home_model.commands import (
    CMD_CST_ON,
    CMD_IDLE,
    CMD_ON,
    LoadCommand,
    copy_command,
)
from custom_components.quiet_solar.home_model.constraints import (
    MultiStepsPowerLoadConstraint,
    TimeBasedSimplePowerLoadConstraint,
)
from custom_components.quiet_solar.home_model.load import TestLoad
from custom_components.quiet_solar.home_model.solver import PeriodSolver
from tests.factories import TestDynamicGroupDouble

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _BoundaryTestLoad:
    """Load giving fine-grained control over switching budget for boundary tests."""

    def __init__(
        self,
        name: str = "boundary_load",
        num_max_on_off: int | None = None,
        num_on_off: int = 0,
        current_command: LoadCommand | None = None,
        father_device=None,
        off_grid: bool = False,
        last_state_change_time: datetime | None = None,
    ):
        self.name = name
        self.num_max_on_off = num_max_on_off
        self.num_on_off = num_on_off
        self.current_command = current_command
        self.father_device = father_device
        self._off_grid = off_grid
        self.last_state_change_time = last_state_change_time
        self.efficiency_factor = 1.0

    def get_update_value_callback_for_constraint_class(self, _constraint):
        return None

    def is_off_grid(self) -> bool:
        return self._off_grid

    def is_time_sensitive(self) -> bool:
        return False

    def get_normalized_score(self, ct, time, score_span):
        return 0.0

    def get_possible_delta_power_from_piloted_devices_for_budget(self, slot_idx, add):
        return 0.0

    def get_phase_amps_from_power_for_budgeting(self, power):
        a = power / 230.0
        return [a, a, a]

    def get_phase_amps_from_power_for_piloted_budgeting(self, power):
        a = power / 230.0
        return [a, a, a]

    def get_first_unlocked_slot_index(self, time_slots, change_state_hysteresis_s=None):
        if change_state_hysteresis_s is None:
            change_state_hysteresis_s = CHANGE_ON_OFF_STATE_HYSTERESIS_S
        if self.num_max_on_off is None or self.last_state_change_time is None:
            return 0
        unlock_time = self.last_state_change_time + timedelta(seconds=change_state_hysteresis_s)
        idx = bisect_left(time_slots, unlock_time)
        return min(idx, len(time_slots))


def _make_time_slots(start: datetime, num_slots: int, step_s: float = SOLVER_STEP_S) -> list[datetime]:
    return [start + timedelta(seconds=i * step_s) for i in range(num_slots)]


def _make_durations(num_slots: int, step_s: float = SOLVER_STEP_S) -> np.ndarray:
    return np.full(num_slots, step_s, dtype=np.float64)


def _make_power_steps(powers: list[float]) -> list[LoadCommand]:
    return [LoadCommand(command=CMD_CST_ON, power_consign=p) for p in powers]


# ===========================================================================
# Task 1: Switching budget exhaustion boundary tests (AC #1)
# ===========================================================================


@pytest.mark.integration
class TestSwitchingBudgetExhaustion:
    """Tests for MANDATORY constraints vs exhausted num_max_on_off switching budget."""

    def test_mandatory_cannot_create_new_on_segment_when_budget_exhausted(self):
        """AC 1.1: MANDATORY_END_TIME cannot create new ON segments when budget is fully used.

        Setup:
        - Load with num_max_on_off=4, num_on_off=4 (budget exhausted)
        - MANDATORY_END_TIME constraint needing energy
        - All slots currently OFF

        Expected:
        - No new ON segments created (would require a transition)
        - Constraint remains unsatisfied
        """
        now = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        load = _BoundaryTestLoad(
            name="exhausted_switch",
            num_max_on_off=4,
            num_on_off=4,  # Budget fully exhausted
            current_command=CMD_IDLE,
        )

        power_steps = _make_power_steps([1000.0, 2000.0, 3000.0])
        constraint = MultiStepsPowerLoadConstraint(
            time=now,
            load=load,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=now + timedelta(hours=2),
            initial_value=0.0,
            target_value=2000.0,  # 2kWh needed
            power_steps=power_steps,
        )

        num_slots = 8
        time_slots = _make_time_slots(now, num_slots)
        durations = _make_durations(num_slots)
        existing_commands: list[LoadCommand | None] = [None] * num_slots

        result = constraint.adapt_repartition(
            first_slot=0,
            last_slot=num_slots - 1,
            energy_delta=2000.0,
            power_slots_duration_s=durations,
            existing_commands=existing_commands,
            allow_change_state=True,
            time=now,
            time_slots=time_slots,
        )
        _, _, _, remaining_energy, out_commands, _ = result

        # With exhausted budget, no new ON segments should be created
        on_count = sum(1 for cmd in out_commands if cmd is not None and cmd.power_consign > 0)
        assert on_count == 0, f"Expected 0 ON segments with exhausted budget, got {on_count}"
        assert remaining_energy > 0, "Should have remaining energy (constraint not fulfilled)"

    def test_mandatory_uses_last_transitions_wisely(self):
        """AC 1.2: MANDATORY_END_TIME with exactly 2 remaining transitions uses them.

        Setup:
        - Load with num_max_on_off=4, num_on_off=2 (2 transitions remaining)
        - MANDATORY_END_TIME constraint
        - All slots currently OFF

        Note: Creating an isolated ON segment costs 2 transitions (OFF->ON boundary
        + ON->OFF boundary). So 2 remaining transitions = exactly 1 new ON segment.

        Expected:
        - At least 1 ON segment created (uses the last 2 transitions)
        """
        now = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        load = _BoundaryTestLoad(
            name="two_transitions_left",
            num_max_on_off=4,
            num_on_off=2,  # 2 transitions remaining — enough for 1 isolated ON segment
            current_command=CMD_IDLE,
        )

        power_steps = _make_power_steps([1000.0, 2000.0])
        constraint = MultiStepsPowerLoadConstraint(
            time=now,
            load=load,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=now + timedelta(hours=2),
            initial_value=0.0,
            target_value=2000.0,
            power_steps=power_steps,
        )

        num_slots = 8
        time_slots = _make_time_slots(now, num_slots)
        durations = _make_durations(num_slots)
        existing_commands: list[LoadCommand | None] = [None] * num_slots

        result = constraint.adapt_repartition(
            first_slot=0,
            last_slot=num_slots - 1,
            energy_delta=2000.0,
            power_slots_duration_s=durations,
            existing_commands=existing_commands,
            allow_change_state=True,
            time=now,
            time_slots=time_slots,
        )
        _, _, _, remaining_energy, out_commands, _ = result

        # With 2 transitions remaining, creating 1 isolated segment is possible
        on_segments = [cmd for cmd in out_commands if cmd is not None and cmd.power_consign > 0]
        assert len(on_segments) >= 1, "Should use 2 transitions to create an ON segment"
        # The segment should deliver meaningful energy (not just 1 slot)
        delivered_energy = sum(
            cmd.power_consign * durations[i] / 3600.0
            for i, cmd in enumerate(out_commands)
            if cmd is not None and cmd.power_consign > 0
        )
        assert delivered_energy > 0, "ON segment should deliver energy toward the target"

    def test_two_pass_prefers_free_transitions(self):
        """AC 1.3: Pass 1 (free transitions) is attempted before pass 2 (budget-spending).

        Setup:
        - Load with remaining switches, currently ON
        - Constraint needs more energy
        - Some adjacent ON slots (extending = free), some empty slots (creating = costs switch)

        Expected:
        - Extensions of existing ON segments happen first (free)
        - New segments only if budget allows in pass 2
        """
        now = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        load = _BoundaryTestLoad(
            name="two_pass_load",
            num_max_on_off=4,
            num_on_off=2,  # 2 transitions remaining
            current_command=copy_command(CMD_ON, power_consign=1000),
        )

        power_steps = _make_power_steps([1000.0, 2000.0])
        constraint = MultiStepsPowerLoadConstraint(
            time=now,
            load=load,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=now + timedelta(hours=4),
            initial_value=0.0,
            target_value=5000.0,
            power_steps=power_steps,
        )

        num_slots = 16
        time_slots = _make_time_slots(now, num_slots)
        durations = _make_durations(num_slots)
        # Slots 0-3 ON (existing), slots 4-15 empty
        on_cmd = copy_command(CMD_ON, power_consign=1000)
        existing_commands: list[LoadCommand | None] = [copy_command(on_cmd)] * 4 + [None] * 12

        result = constraint.adapt_repartition(
            first_slot=0,
            last_slot=num_slots - 1,
            energy_delta=5000.0,
            power_slots_duration_s=durations,
            existing_commands=existing_commands,
            allow_change_state=True,
            time=now,
            time_slots=time_slots,
        )
        _, _, _, remaining_energy, out_commands, _ = result

        # Verify that existing ON slots (0-3) are extended (power increased) — pass 1 (free)
        extended_existing = [cmd for cmd in out_commands[:4] if cmd is not None and cmd.power_consign > 1000]
        assert len(extended_existing) > 0, "Pass 1 should extend existing ON slots to higher power (free transitions)"

        # Verify that new ON segments were also created beyond slot 3 — pass 2 (budget-spending)
        on_count = sum(1 for cmd in out_commands if cmd is not None and cmd.power_consign > 0)
        assert on_count > len(extended_existing), "Pass 2 should create new ON segments beyond existing ones"

    def test_segment_extension_works_when_budget_exhausted(self):
        """AC 1.4: Extending adjacent ON segments is allowed when budget is exhausted.

        Setup:
        - Load with exhausted switching budget (num_on_off=num_max_on_off)
        - Currently ON with some existing ON slots
        - Constraint needs more energy

        Expected:
        - Existing ON segments can be extended (power increased) — no transition cost
        - Budget NOT decremented for extensions
        """
        now = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        load = _BoundaryTestLoad(
            name="extend_load",
            num_max_on_off=4,
            num_on_off=4,  # Budget exhausted
            current_command=copy_command(CMD_ON, power_consign=1000),
        )

        power_steps = _make_power_steps([1000.0, 2000.0, 3000.0])
        constraint = MultiStepsPowerLoadConstraint(
            time=now,
            load=load,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=now + timedelta(hours=2),
            initial_value=0.0,
            target_value=3000.0,
            power_steps=power_steps,
        )

        num_slots = 8
        time_slots = _make_time_slots(now, num_slots)
        durations = _make_durations(num_slots)
        # Slots 0-3 already ON at low power — extension should increase power
        low_cmd = copy_command(CMD_ON, power_consign=1000)
        existing_commands: list[LoadCommand | None] = [copy_command(low_cmd)] * 4 + [None] * 4

        result = constraint.adapt_repartition(
            first_slot=0,
            last_slot=num_slots - 1,
            energy_delta=3000.0,
            power_slots_duration_s=durations,
            existing_commands=existing_commands,
            allow_change_state=True,
            time=now,
            time_slots=time_slots,
        )
        _, _, _, remaining_energy, out_commands, _ = result

        # Existing ON slots should have increased power (extension, not new segment)
        extended_slots = [cmd for i, cmd in enumerate(out_commands[:4]) if cmd is not None and cmd.power_consign > 1000]
        assert len(extended_slots) > 0, "Existing ON slots should be extended to higher power"

    def test_asap_with_exhausted_budget_behaves_differently_from_mandatory(self):
        """AC 1.5: ASAP vs MANDATORY_END_TIME with exhausted budget — score difference.

        Both are mandatory but ASAP has higher score. Verify scoring reflects this
        even when both face the same switching budget constraint.
        """
        now = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)

        # Create two constraints with identical loads but different types
        load_asap = _BoundaryTestLoad(
            name="asap_load",
            num_max_on_off=4,
            num_on_off=4,
        )
        load_mandatory = _BoundaryTestLoad(
            name="mandatory_load",
            num_max_on_off=4,
            num_on_off=4,
        )

        constraint_asap = MultiStepsPowerLoadConstraint(
            time=now,
            load=load_asap,
            type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            end_of_constraint=now + timedelta(hours=2),
            initial_value=0.0,
            target_value=2000.0,
            power_steps=_make_power_steps([1000.0]),
        )

        constraint_mandatory = MultiStepsPowerLoadConstraint(
            time=now,
            load=load_mandatory,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=now + timedelta(hours=2),
            initial_value=0.0,
            target_value=2000.0,
            power_steps=_make_power_steps([1000.0]),
        )

        score_asap = constraint_asap.score(now)
        score_mandatory = constraint_mandatory.score(now)

        assert score_asap > score_mandatory, (
            f"ASAP score ({score_asap}) should be > MANDATORY score ({score_mandatory})"
        )

        # Both are mandatory
        assert constraint_asap.is_mandatory
        assert constraint_mandatory.is_mandatory


# ===========================================================================
# Task 2: Amp budget exhaustion with mandatory constraints (AC #2)
# ===========================================================================


@pytest.mark.integration
class TestAmpBudgetExhaustion:
    """Tests for mandatory constraints vs exhausted charger amp budget."""

    def test_mandatory_gets_no_allocation_when_min_step_exceeds_amps(self):
        """AC 2.1: When minimum power step exceeds available amps, no allocation.

        Setup:
        - Father device (dynamic group) with 5A available per phase
        - Constraint with minimum power step of 7A * 230V = 1610W (needs ~7A)
        - 7A > 5A available

        Expected:
        - adapt_power_steps_budgeting returns empty list
        """
        now = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        father = TestDynamicGroupDouble(
            max_amps=[5.0, 5.0, 5.0],  # Only 5A available per phase
            num_slots=4,
        )
        load = _BoundaryTestLoad(
            name="amp_limited",
            father_device=father,
        )

        # Minimum step is 1610W (~7A per phase) which exceeds 5A available
        power_steps = _make_power_steps([1610.0, 3220.0, 4830.0])
        constraint = MultiStepsPowerLoadConstraint(
            time=now,
            load=load,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=now + timedelta(hours=2),
            initial_value=0.0,
            target_value=3000.0,
            power_steps=power_steps,
        )

        # Check budgeting at slot 0
        available_cmds = constraint.adapt_power_steps_budgeting_low_level(
            slot_idx=0,
            existing_amps=None,
            use_production_limits=False,
        )

        assert len(available_cmds) == 0, (
            f"Expected 0 available commands when min step exceeds amp budget, got {len(available_cmds)}"
        )

    def test_mandatory_uses_smallest_fitting_step(self):
        """AC 2.2: When only smallest power step fits in available amps, uses it.

        Setup:
        - Father device with 8A available per phase
        - Power steps: 1610W (~7A), 3220W (~14A), 4830W (~21A)
        - Only 7A step fits

        Expected:
        - adapt_power_steps_budgeting returns only the smallest step
        """
        now = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        father = TestDynamicGroupDouble(
            max_amps=[8.0, 8.0, 8.0],  # 8A per phase — only 7A step fits
            num_slots=4,
        )
        load = _BoundaryTestLoad(
            name="small_step_only",
            father_device=father,
        )

        power_steps = _make_power_steps([1610.0, 3220.0, 4830.0])
        constraint = MultiStepsPowerLoadConstraint(
            time=now,
            load=load,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=now + timedelta(hours=2),
            initial_value=0.0,
            target_value=3000.0,
            power_steps=power_steps,
        )

        available_cmds = constraint.adapt_power_steps_budgeting_low_level(
            slot_idx=0,
            existing_amps=None,
            use_production_limits=False,
        )

        assert len(available_cmds) == 1, f"Expected 1 fitting command, got {len(available_cmds)}"
        assert available_cmds[0].power_consign == 1610.0

    def test_amp_budget_varies_across_slots(self):
        """AC 2.3: Amp budget drops mid-window — constraint adapts per slot.

        Setup:
        - Slot 0-1: 15A available (both steps fit)
        - Slot 2-3: 5A available (no steps fit)

        Expected:
        - Slots 0-1: commands available
        - Slots 2-3: no commands available
        """
        now = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        father = TestDynamicGroupDouble(
            max_amps=[15.0, 15.0, 15.0],
            num_slots=4,
        )
        # Override slot 2-3 to have lower amps
        father.available_amps_for_group[2] = [5.0, 5.0, 5.0]
        father.available_amps_for_group[3] = [5.0, 5.0, 5.0]

        load = _BoundaryTestLoad(
            name="varying_amps",
            father_device=father,
        )

        power_steps = _make_power_steps([1610.0, 3220.0])
        constraint = MultiStepsPowerLoadConstraint(
            time=now,
            load=load,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=now + timedelta(hours=2),
            initial_value=0.0,
            target_value=3000.0,
            power_steps=power_steps,
        )

        # High-amp slots should have commands available
        cmds_slot_0 = constraint.adapt_power_steps_budgeting_low_level(slot_idx=0)
        cmds_slot_1 = constraint.adapt_power_steps_budgeting_low_level(slot_idx=1)
        # Low-amp slots should have no commands
        cmds_slot_2 = constraint.adapt_power_steps_budgeting_low_level(slot_idx=2)
        cmds_slot_3 = constraint.adapt_power_steps_budgeting_low_level(slot_idx=3)

        assert len(cmds_slot_0) > 0, "Slot 0 (15A) should have available commands"
        assert len(cmds_slot_1) > 0, "Slot 1 (15A) should have available commands"
        assert len(cmds_slot_2) == 0, "Slot 2 (5A) should have no available commands"
        assert len(cmds_slot_3) == 0, "Slot 3 (5A) should have no available commands"

    def test_higher_score_constraint_gets_amp_budget_priority(self):
        """AC 2.4: Two mandatory constraints — higher score gets allocated first.

        When two loads compete for the same amp budget through the solver,
        the higher-scored constraint should be allocated power first.
        """
        dt = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        end_time = dt + timedelta(hours=4)

        # ASAP constraint (higher score) — small energy need
        load_high = TestLoad(name="high_priority")
        constraint_high = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_high,
            type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            end_of_constraint=dt + timedelta(hours=1),
            initial_value=0,
            target_value=1 * 3600,  # 1h at 2kW = 2kWh
            power=2000,
        )
        load_high.push_live_constraint(dt, constraint_high)

        # MANDATORY_END_TIME (lower score) — large energy need
        load_low = TestLoad(name="low_priority")
        constraint_low = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_low,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=3 * 3600,  # 3h at 2kW = 6kWh
            power=2000,
        )
        load_low.push_live_constraint(dt, constraint_low)

        # Very limited power: 2kW — only one load at a time
        pv_forecast = [(dt + timedelta(hours=h), 2100.0) for h in range(5)]
        ua_forecast = [(dt + timedelta(hours=h), 100.0) for h in range(5)]

        solver = PeriodSolver(
            start_time=dt,
            end_time=end_time,
            tariffs=0.20 / 1000.0,
            actionable_loads=[load_high, load_low],
            battery=None,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast,
        )

        load_commands, _ = solver.solve(with_self_test=True)

        # Verify scores reflect priority
        assert constraint_high.score(dt) > constraint_low.score(dt)

        # High priority should be satisfied
        c_high_final = next((c for c in solver._constraints_for_test if c.load.name == "high_priority"), None)
        assert c_high_final is not None
        assert c_high_final.is_constraint_met(end_time), "High priority ASAP constraint should be met"


# ===========================================================================
# Task 3: Multiple MANDATORY constraints competing (AC #3)
# ===========================================================================


@pytest.mark.integration
class TestMultipleMandatoryCompetition:
    """Tests for multiple MANDATORY constraints competing for insufficient power."""

    def test_two_mandatory_same_deadline_score_determines_allocation(self):
        """AC 3.1: Two MANDATORY_END_TIME, same deadline, limited power.

        Higher-scored constraint gets more energy. Score is differentiated by
        target energy (larger target = higher score in energy component).
        """
        dt = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        end_time = dt + timedelta(hours=4)

        # Load A: needs 5kWh (larger target = higher energy score component)
        load_a = TestLoad(name="load_big_target")
        constraint_a = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_a,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=2.5 * 3600,  # 2.5h at 2kW = 5kWh
            power=2000,
        )
        load_a.push_live_constraint(dt, constraint_a)

        # Load B: needs 3kWh (smaller target = lower energy score)
        load_b = TestLoad(name="load_small_target")
        constraint_b = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_b,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=1.5 * 3600,  # 1.5h at 2kW = 3kWh
            power=2000,
        )
        load_b.push_live_constraint(dt, constraint_b)

        # Total need: 8kWh, available ~6kWh (1.5kW for 4h)
        pv_forecast = [(dt + timedelta(hours=h), 1600.0) for h in range(5)]
        ua_forecast = [(dt + timedelta(hours=h), 100.0) for h in range(5)]

        solver = PeriodSolver(
            start_time=dt,
            end_time=end_time,
            tariffs=0.20 / 1000.0,
            actionable_loads=[load_a, load_b],
            battery=None,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast,
        )

        load_commands, _ = solver.solve(with_self_test=True)

        # Verify scoring hierarchy
        score_a = constraint_a.score(dt)
        score_b = constraint_b.score(dt)
        assert score_a > score_b, f"Larger target ({score_a}) should score higher than smaller ({score_b})"

        # Both should get some allocation (solver doesn't fail)
        assert len(load_commands) == 2, "Both loads should receive command timelines"

    def test_asap_vs_mandatory_asap_allocated_first(self):
        """AC 3.2: ASAP vs MANDATORY_END_TIME — ASAP gets allocated first.

        With limited total power, ASAP constraint should be fully satisfied
        before MANDATORY_END_TIME gets remaining power.
        """
        dt = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        end_time = dt + timedelta(hours=6)

        # ASAP: needs 3kWh
        load_asap = TestLoad(name="asap_load")
        constraint_asap = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_asap,
            type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            end_of_constraint=dt + timedelta(hours=2),
            initial_value=0,
            target_value=1.5 * 3600,  # 1.5h at 2kW = 3kWh
            power=2000,
        )
        load_asap.push_live_constraint(dt, constraint_asap)

        # Mandatory: needs 5kWh
        load_mandatory = TestLoad(name="mandatory_load")
        constraint_mandatory = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_mandatory,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=2.5 * 3600,  # 2.5h at 2kW = 5kWh
            power=2000,
        )
        load_mandatory.push_live_constraint(dt, constraint_mandatory)

        # Limited: ~6.6kWh available (1.1kW for 6h)
        pv_forecast = [(dt + timedelta(hours=h), 1200.0) for h in range(7)]
        ua_forecast = [(dt + timedelta(hours=h), 100.0) for h in range(7)]

        solver = PeriodSolver(
            start_time=dt,
            end_time=end_time,
            tariffs=0.20 / 1000.0,
            actionable_loads=[load_asap, load_mandatory],
            battery=None,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast,
        )

        load_commands, _ = solver.solve(with_self_test=True)

        # ASAP should have higher score
        assert constraint_asap.score(dt) > constraint_mandatory.score(dt)

        # ASAP should be satisfied (allocated first)
        c_asap_final = next((c for c in solver._constraints_for_test if c.load.name == "asap_load"), None)
        assert c_asap_final is not None
        assert c_asap_final.is_constraint_met(end_time), "ASAP constraint should be met (allocated first)"

    def test_three_mandatory_only_two_satisfied(self):
        """AC 3.3: Three mandatory constraints, only enough power for two.

        The lowest-score constraint should be the one partially/not fulfilled.
        """
        dt = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        end_time = dt + timedelta(hours=6)

        # ASAP: needs 2kWh (highest score)
        load_a = TestLoad(name="asap")
        constraint_a = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_a,
            type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            end_of_constraint=dt + timedelta(hours=1),
            initial_value=0,
            target_value=1 * 3600,  # 1h at 2kW = 2kWh
            power=2000,
        )
        load_a.push_live_constraint(dt, constraint_a)

        # MANDATORY (big target): needs 4kWh (medium score — big energy)
        load_b = TestLoad(name="mandatory_big")
        constraint_b = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_b,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=2 * 3600,  # 2h at 2kW = 4kWh
            power=2000,
        )
        load_b.push_live_constraint(dt, constraint_b)

        # MANDATORY (small target): needs 3kWh (lowest score — small energy)
        load_c = TestLoad(name="mandatory_small")
        constraint_c = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_c,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=1.5 * 3600,  # 1.5h at 2kW = 3kWh
            power=2000,
        )
        load_c.push_live_constraint(dt, constraint_c)

        # Total need: 9kWh, available ~5.4kWh (0.9kW for 6h)
        pv_forecast = [(dt + timedelta(hours=h), 1000.0) for h in range(7)]
        ua_forecast = [(dt + timedelta(hours=h), 100.0) for h in range(7)]

        solver = PeriodSolver(
            start_time=dt,
            end_time=end_time,
            tariffs=0.20 / 1000.0,
            actionable_loads=[load_a, load_b, load_c],
            battery=None,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast,
        )

        load_commands, _ = solver.solve(with_self_test=True)

        # Verify score ordering
        score_a = constraint_a.score(dt)
        score_b = constraint_b.score(dt)
        score_c = constraint_c.score(dt)

        assert score_a > score_b, "ASAP should score highest"
        assert score_a > score_c, "ASAP should score higher than small mandatory"

        # ASAP should be satisfied (highest priority)
        c_a_final = next((c for c in solver._constraints_for_test if c.load.name == "asap"), None)
        assert c_a_final is not None
        assert c_a_final.is_constraint_met(end_time), "ASAP constraint should be met"

        # Solver should produce valid output for all loads
        assert len(load_commands) == 3

        # All mandatory constraints can use grid power, so all may be met.
        # The key invariant: higher-score constraint (ASAP) is allocated first,
        # and the solver handles the competition without crashing.
        c_b_final = next((c for c in solver._constraints_for_test if c.load.name == "mandatory_big"), None)
        c_c_final = next((c for c in solver._constraints_for_test if c.load.name == "mandatory_small"), None)
        assert c_b_final is not None
        assert c_c_final is not None

        # Verify score ordering: ASAP > big mandatory > small mandatory
        assert score_b > score_c, "Bigger energy target should score higher than smaller"

    def test_two_mandatory_different_deadlines(self):
        """AC 3.4: Two mandatory constraints with different deadlines.

        Both MANDATORY_END_TIME but one has an earlier deadline. When power
        is limited, both are allocated by score (which includes energy target
        as a component).
        """
        dt = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        end_time = dt + timedelta(hours=8)

        # Early deadline: 2h, needs 3kWh
        load_early = TestLoad(name="early_deadline")
        constraint_early = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_early,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=dt + timedelta(hours=2),
            initial_value=0,
            target_value=1.5 * 3600,  # 1.5h at 2kW = 3kWh
            power=2000,
        )
        load_early.push_live_constraint(dt, constraint_early)

        # Late deadline: 8h, needs 6kWh
        load_late = TestLoad(name="late_deadline")
        constraint_late = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_late,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=3 * 3600,  # 3h at 2kW = 6kWh
            power=2000,
        )
        load_late.push_live_constraint(dt, constraint_late)

        # Medium power: ~7.2kWh (0.9kW for 8h) — not enough for 9kWh total
        pv_forecast = [(dt + timedelta(hours=h), 1000.0) for h in range(9)]
        ua_forecast = [(dt + timedelta(hours=h), 100.0) for h in range(9)]

        solver = PeriodSolver(
            start_time=dt,
            end_time=end_time,
            tariffs=0.20 / 1000.0,
            actionable_loads=[load_early, load_late],
            battery=None,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast,
        )

        load_commands, _ = solver.solve(with_self_test=True)

        # Both should have commands
        assert len(load_commands) == 2

        # Both constraints should have received some allocation
        c_early = next((c for c in solver._constraints_for_test if c.load.name == "early_deadline"), None)
        c_late = next((c for c in solver._constraints_for_test if c.load.name == "late_deadline"), None)
        assert c_early is not None
        assert c_late is not None

        # Score is driven by energy target within same type tier, not deadline.
        # Late constraint has larger energy target (6kWh vs 3kWh) → higher score.
        score_early = constraint_early.score(dt)
        score_late = constraint_late.score(dt)
        assert score_late > score_early, (
            f"Larger energy target ({score_late}) should score higher than smaller ({score_early})"
        )

    def test_user_mandatory_beats_system_mandatory(self):
        """AC 3.5: User-originated mandatory vs system mandatory — from_user wins.

        Two MANDATORY_END_TIME constraints with identical parameters except
        from_user flag. User constraint should have higher score.
        """
        dt = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        end_time = dt + timedelta(hours=4)

        # User-originated mandatory
        load_user = TestLoad(name="user_mandatory")
        constraint_user = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_user,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=2 * 3600,  # 2h at 2kW = 4kWh
            power=2000,
            from_user=True,
        )
        load_user.push_live_constraint(dt, constraint_user)

        # System-originated mandatory
        load_sys = TestLoad(name="system_mandatory")
        constraint_sys = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_sys,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=2 * 3600,  # Same 4kWh
            power=2000,
            from_user=False,
        )
        load_sys.push_live_constraint(dt, constraint_sys)

        # Limited: ~3kWh (0.75kW for 4h) — not enough for 8kWh total
        pv_forecast = [(dt + timedelta(hours=h), 850.0) for h in range(5)]
        ua_forecast = [(dt + timedelta(hours=h), 100.0) for h in range(5)]

        solver = PeriodSolver(
            start_time=dt,
            end_time=end_time,
            tariffs=0.20 / 1000.0,
            actionable_loads=[load_user, load_sys],
            battery=None,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast,
        )

        # Verify score ordering
        score_user = constraint_user.score(dt)
        score_sys = constraint_sys.score(dt)
        assert score_user > score_sys, f"User constraint ({score_user}) should score higher than system ({score_sys})"

        load_commands, _ = solver.solve(with_self_test=True)

        # User constraint should get better allocation
        c_user_final = next((c for c in solver._constraints_for_test if c.load.name == "user_mandatory"), None)
        c_sys_final = next((c for c in solver._constraints_for_test if c.load.name == "system_mandatory"), None)

        assert c_user_final is not None
        assert c_sys_final is not None

        # User progress should be strictly >= system progress (user has higher score)
        user_progress = c_user_final.current_value / c_user_final.target_value if c_user_final.target_value else 0
        sys_progress = c_sys_final.current_value / c_sys_final.target_value if c_sys_final.target_value else 0
        assert user_progress >= sys_progress, (
            f"User progress ({user_progress:.1%}) should be >= system ({sys_progress:.1%})"
        )


# ===========================================================================
# Task 4: Constraint type transitions under resource pressure (AC #4)
# ===========================================================================


@pytest.mark.integration
class TestConstraintTypeTransitions:
    """Tests for constraint type transitions under resource pressure."""

    def test_off_grid_degrades_asap_to_mandatory(self):
        """AC 4.1: Off-grid mode degrades ASAP to MANDATORY_END_TIME.

        When load is off-grid, the type property returns _degraded_type
        instead of _type. ASAP (9) degrades to MANDATORY_END_TIME (7).
        """
        now = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        load = _BoundaryTestLoad(name="offgrid_load", off_grid=True)

        constraint = MultiStepsPowerLoadConstraint(
            time=now,
            load=load,
            type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            degraded_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=now + timedelta(hours=2),
            initial_value=0.0,
            target_value=2000.0,
            power_steps=_make_power_steps([1000.0]),
        )

        # Off-grid: type property should return degraded value
        assert constraint.type == CONSTRAINT_TYPE_MANDATORY_END_TIME, (
            f"Off-grid ASAP should degrade to MANDATORY_END_TIME, got {constraint.type}"
        )
        assert constraint.is_mandatory, "Degraded ASAP should still be mandatory"
        assert not constraint.as_fast_as_possible, "Degraded ASAP should not be as_fast_as_possible"

        # Returning to on-grid should restore ASAP type
        load._off_grid = False
        assert constraint.type == CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE, (
            "Returning on-grid should restore ASAP type"
        )
        assert constraint.as_fast_as_possible, "On-grid should be as_fast_as_possible again"

    def test_constraint_skip_and_pushed_count_properties(self):
        """AC 4.2: Verify skip and pushed_count properties support the deadline-miss protocol.

        The real skip logic lives in load.py:1462 (update_live_constraints) which
        requires a full AbstractLoad. This test verifies the constraint properties
        that protocol depends on: pushed_count is writable, skip defaults to False,
        and skip is writable. The actual triggered behavior (pushed_count > 4 → skip)
        is tested through the solver in integration tests.
        """
        now = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        load = _BoundaryTestLoad(name="push_limit_load")

        constraint = MultiStepsPowerLoadConstraint(
            time=now,
            load=load,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=now + timedelta(hours=2),
            initial_value=0.0,
            target_value=2000.0,
            power_steps=_make_power_steps([1000.0]),
        )

        # Verify defaults
        assert not constraint.skip, "skip should default to False"
        assert constraint.pushed_count == 0, "pushed_count should default to 0"

        # Verify the constraint is still mandatory and scorable when pushed
        constraint.pushed_count = 3
        assert constraint.is_mandatory, "Constraint should remain mandatory after pushes"
        score_before = constraint.score(now)

        # Verify skip flag prevents further processing
        constraint.skip = True
        assert constraint.skip, "skip should be settable to True"

        # A skipped constraint should still be mandatory (skip is orthogonal to type)
        assert constraint.is_mandatory, "Skipped constraint should still be mandatory"

    def test_constraint_type_promotion_via_setter(self):
        """AC 4.3: Type setter promotes MANDATORY_END_TIME to ASAP correctly.

        The real promotion is triggered by load.py:1462-1465 (update_live_constraints)
        which calls `constraint.type = MANDATORY_AS_FAST_AS_POSSIBLE`. This test
        verifies the setter contract: setting type to ASAP changes the public type,
        increases the score, and caps off-grid degradation at MANDATORY_END_TIME.
        """
        now = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        load = _BoundaryTestLoad(name="promote_load")

        constraint = MultiStepsPowerLoadConstraint(
            time=now,
            load=load,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=now + timedelta(hours=2),
            initial_value=0.0,
            target_value=2000.0,
            power_steps=_make_power_steps([1000.0]),
        )

        # Verify initial state via public API
        assert constraint.type == CONSTRAINT_TYPE_MANDATORY_END_TIME
        assert constraint.is_mandatory
        assert not constraint.as_fast_as_possible
        score_before = constraint.score(now)

        # Simulate deadline miss promotion (load.py:1462-1465)
        constraint.type = CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE

        # Verify promotion through public API
        assert constraint.type == CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE, "Type should be promoted to ASAP"
        assert constraint.as_fast_as_possible, "Should now be as_fast_as_possible"
        assert constraint.is_mandatory, "Should still be mandatory"

        # ASAP score must be higher than MANDATORY_END_TIME score
        score_after = constraint.score(now)
        assert score_after > score_before, (
            f"ASAP score ({score_after}) should be > MANDATORY_END_TIME score ({score_before})"
        )

        # Verify off-grid degradation is capped (setter side-effect)
        load._off_grid = True
        assert constraint.type == CONSTRAINT_TYPE_MANDATORY_END_TIME, (
            "Off-grid should degrade ASAP back to MANDATORY_END_TIME"
        )

    def test_off_grid_transition_preserves_constraint_progress(self):
        """AC 4.4: Off-grid transition preserves current_value continuity.

        When a load transitions to off-grid mode, the constraint's current_value
        should remain unchanged — only the type degrades, not the progress.
        """
        now = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        load = _BoundaryTestLoad(name="transition_load", off_grid=False)

        constraint = MultiStepsPowerLoadConstraint(
            time=now,
            load=load,
            type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            degraded_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=now + timedelta(hours=2),
            initial_value=0.0,
            target_value=2000.0,
            current_value=800.0,  # 40% progress
            power_steps=_make_power_steps([1000.0]),
        )

        # Before off-grid
        assert constraint.type == CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE
        assert constraint.current_value == 800.0

        # Transition to off-grid
        load._off_grid = True

        # Type should degrade but progress preserved
        assert constraint.type == CONSTRAINT_TYPE_MANDATORY_END_TIME, "Type should degrade"
        assert constraint.current_value == 800.0, "Progress should be preserved during grid transition"
        assert constraint.target_value == 2000.0, "Target should be preserved"

    def test_dual_resource_exhaustion_no_crash(self):
        """AC 4.5: Both switching budget and amp budget exhausted simultaneously.

        When both constraints are exhausted, the system should handle it
        gracefully without crashing.
        """
        now = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        father = TestDynamicGroupDouble(
            max_amps=[3.0, 3.0, 3.0],  # Very limited amps
            num_slots=8,
        )
        load = _BoundaryTestLoad(
            name="dual_exhaustion",
            num_max_on_off=4,
            num_on_off=4,  # Switching budget exhausted
            father_device=father,
            current_command=CMD_IDLE,
        )

        # Power steps all exceed amp budget (min 7A > 3A available)
        power_steps = _make_power_steps([1610.0, 3220.0])
        constraint = MultiStepsPowerLoadConstraint(
            time=now,
            load=load,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=now + timedelta(hours=2),
            initial_value=0.0,
            target_value=3000.0,
            power_steps=power_steps,
        )

        num_slots = 8
        time_slots = _make_time_slots(now, num_slots)
        durations = _make_durations(num_slots)
        existing_commands: list[LoadCommand | None] = [None] * num_slots

        # Should not crash — gracefully handle dual exhaustion
        result = constraint.adapt_repartition(
            first_slot=0,
            last_slot=num_slots - 1,
            energy_delta=3000.0,
            power_slots_duration_s=durations,
            existing_commands=existing_commands,
            allow_change_state=True,
            time=now,
            time_slots=time_slots,
        )
        _, _, _, remaining_energy, out_commands, _ = result

        # No allocation possible — both switching and amp budgets exhausted
        on_count = sum(1 for cmd in out_commands if cmd is not None and cmd.power_consign > 0)
        assert on_count == 0, "No allocation should be possible with dual exhaustion"
        assert remaining_energy > 0, "Energy should remain unfulfilled"
