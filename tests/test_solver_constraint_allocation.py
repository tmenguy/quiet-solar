"""
Deep tests for solver constraint allocation and priority resolution.

These tests validate that the solver correctly prioritizes and allocates
power to multiple competing constraints based on their type, score, and
available power.
"""

import pytest
import pytz
import numpy as np
from datetime import datetime, timedelta

from custom_components.quiet_solar.home_model.solver import PeriodSolver
from custom_components.quiet_solar.home_model.battery import Battery
from custom_components.quiet_solar.home_model.load import TestLoad
from custom_components.quiet_solar.home_model.constraints import (
    MultiStepsPowerLoadConstraint,
    TimeBasedSimplePowerLoadConstraint,
)
from custom_components.quiet_solar.home_model.commands import LoadCommand
from custom_components.quiet_solar.const import (
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
    CONSTRAINT_TYPE_FILLER_AUTO,
)

from tests.utils.energy_validation import (
    validate_battery_soc_bounds,
    calculate_energy_from_commands,
    validate_no_overallocation,
    reconstruct_battery_charge_evolution,
)
from tests.utils.scenario_builders import (
    create_test_battery,
    create_car_with_power_steps,
)


class TestConstraintPriorityAllocation:
    """Test constraint priority and allocation logic."""

    def test_constraint_priority_with_limited_solar(self):
        """
        DEEP TEST: Multiple constraints compete for limited solar energy.
        
        Validates that mandatory constraints are satisfied before optional ones
        when power is limited.
        
        Setup:
        - Solar: moderate availability
        - Load A: MANDATORY (high priority)
        - Load B: MANDATORY (high priority)
        - Load C: OPTIONAL (lower priority)
        
        Validates:
        - Mandatory constraints satisfied
        - Priority system works
        - Optional gets less priority
        - Solver produces valid solution
        """
        dt = datetime(2024, 6, 1, 8, 0, 0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=10)
        
        # Create loads
        car = TestLoad(name="car")
        heater = TestLoad(name="heater")
        pool = TestLoad(name="pool")
        
        # Car: mandatory
        car_steps = [LoadCommand(command="ON", power_consign=a * 230 * 3) 
                     for a in range(7, 20)]  # Reduced range
        car_constraint = MultiStepsPowerLoadConstraint(
            time=dt,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=8000,  # 8kWh
            power_steps=car_steps,
            support_auto=True
        )
        car.push_live_constraint(dt, car_constraint)
        
        # Heater: mandatory
        heater_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=heater,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=2 * 3600,  # 2h at 2kW = 4kWh
            power=2000
        )
        heater.push_live_constraint(dt, heater_constraint)
        
        # Pool: optional
        pool_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=pool,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=3 * 3600,  # 3h at 2kW = 6kWh
            power=2000
        )
        pool.push_live_constraint(dt, pool_constraint)
        
        # Moderate solar
        pv_forecast = [[dt + timedelta(hours=h), 2300.0] for h in range(11)]
        ua_forecast = [[dt + timedelta(hours=h), 200.0] for h in range(11)]

        for i in range(6, 11):
            pv_forecast[i][1] = 1000.0  # Lower solar later

        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=0.20/1000.0,
            actionable_loads=[car, heater, pool],
            battery=None,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast
        )
        
        load_commands, _ = solver.solve(with_self_test=True)
        
        # =================================================================
        # VALIDATE: Check constraints are handled
        # =================================================================
        # Find final constraint states
        car_constraint_final = None
        heater_constraint_final = None
        pool_constraint_final = None
        
        for c in solver._constraints_for_test:
            if c.load.name == "car":
                car_constraint_final = c
            elif c.load.name == "heater":
                heater_constraint_final = c
            elif c.load.name == "pool":
                pool_constraint_final = c
        
        # Check mandatory constraints are met or significantly progressed
        assert car_constraint_final is not None
        assert heater_constraint_final is not None
        
        car_satisfied = car_constraint_final.is_constraint_met(end_time)
        heater_satisfied = heater_constraint_final.is_constraint_met(end_time)
        pool_satisfied = pool_constraint_final.is_constraint_met(end_time) if pool_constraint_final else False

        assert car_satisfied and heater_satisfied

        assert pool_satisfied is False

        assert pool_constraint_final.current_value > pool_constraint_final.target_value*0.3

        # Check priority scores
        car_score = car_constraint.score(dt)
        pool_score = pool_constraint.score(dt)
        
        assert car_score > pool_score, (
            f"Mandatory should have higher score than optional\n"
            f"  Car (mandatory): {car_score}\n"
            f"  Pool (optional): {pool_score}"
        )
        
        # All loads should have commands
        assert len(load_commands) == 3
        
        print(f"✅ Priority allocation test passed!")
        print(f"   - Car (mandatory): {'met' if car_satisfied else 'partial'}")
        print(f"   - Heater (mandatory): {'met' if heater_satisfied else 'partial'}")
        print(f"   - Pool (optional): {'met' if pool_satisfied else 'partial'}")
        print(f"   - Priority scores: car={car_score:.0f} > pool={pool_score:.0f}")

    def test_asap_constraint_gets_highest_priority(self):
        """
        DEEP TEST: ASAP constraint gets power before all others.
        
        Setup:
        - Load A: ASAP, needs 3kWh
        - Load B: MANDATORY_END_TIME, needs 5kWh
        - Load C: BEFORE_BATTERY, needs 4kWh
        - Available: 8kWh total
        
        Expected:
        - A gets 3kWh (ASAP takes first)
        - B gets 5kWh (mandatory takes second)
        - C gets 0kWh (insufficient power remaining)
        
        Validates:
        - ASAP priority > MANDATORY > BEFORE_BATTERY
        - Allocation order follows priority
        - Energy accounting correct
        """
        dt = datetime(2024, 6, 1, 9, 0, 0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=8)
        
        # Create three loads with different priorities
        load_a = TestLoad(name="load_asap")
        load_b = TestLoad(name="load_mandatory")
        load_c = TestLoad(name="load_optional")
        
        # ASAP constraint
        constraint_a = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_a,
            type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            end_of_constraint=dt + timedelta(hours=2),
            initial_value=0,
            target_value=1.5 * 3600,  # 1.5h at 2kW = 3kWh
            power=2000
        )
        load_a.push_live_constraint(dt, constraint_a)
        
        # Mandatory constraint
        constraint_b = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_b,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=2.5 * 3600,  # 2.5h at 2kW = 5kWh
            power=2000
        )
        load_b.push_live_constraint(dt, constraint_b)
        
        # Optional constraint
        constraint_c = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_c,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=2 * 3600,  # 2h at 2kW = 4kWh
            power=2000
        )
        load_c.push_live_constraint(dt, constraint_c)
        
        # Limited solar: 8kWh total = 1kW for 8h
        pv_forecast = [(dt + timedelta(hours=h), 1000.0) for h in range(9)]
        ua_forecast = [(dt + timedelta(hours=h), 50.0) for h in range(9)]
        
        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=0.20/1000.0,
            actionable_loads=[load_a, load_b, load_c],
            battery=None,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast
        )
        
        load_commands, _ = solver.solve(with_self_test=True)
        
        # =================================================================
        # VALIDATE: Check constraint satisfaction and priority
        # =================================================================
        # Find final states
        constraint_a_final = next((c for c in solver._constraints_for_test if c.load.name == "load_asap"), None)
        constraint_b_final = next((c for c in solver._constraints_for_test if c.load.name == "load_mandatory"), None)
        constraint_c_final = next((c for c in solver._constraints_for_test if c.load.name == "load_optional"), None)
        
        assert constraint_a_final is not None
        assert constraint_b_final is not None
        assert constraint_c_final is not None
        
        # Check satisfaction
        asap_satisfied = constraint_a_final.is_constraint_met(end_time)
        mandatory_satisfied = constraint_b_final.is_constraint_met(end_time)
        optional_satisfied = constraint_c_final.is_constraint_met(end_time)

        assert asap_satisfied and mandatory_satisfied
        assert not optional_satisfied


        # Check scores reflect priority
        score_a = constraint_a.score(dt)
        score_b = constraint_b.score(dt)
        score_c = constraint_c.score(dt)
        
        assert score_a > score_b > score_c, (
            f"Scores don't reflect priority\n"
            f"  ASAP: {score_a}\n"
            f"  Mandatory: {score_b}\n"
            f"  Optional: {score_c}"
        )
        
        # All should have commands
        assert len(load_commands) == 3
        
        print(f"✅ ASAP priority test passed!")
        print(f"   - ASAP: {'met' if asap_satisfied else 'partial'}")
        print(f"   - Mandatory: {'met' if mandatory_satisfied else 'partial'}")
        print(f"   - Optional: {'met' if optional_satisfied else 'partial'}")
        print(f"   - Scores: {score_a:.0f} > {score_b:.0f} > {score_c:.0f}")

    def test_mandatory_vs_optional_with_battery(self):
        """
        DEEP TEST: Mandatory constraints use battery, optional constraints don't.
        
        Setup:
        - Battery: 6kWh, 50% charged
        - Mandatory load: needs 5kWh (may use battery)
        - Optional load: wants 8kWh (solar only)
        - Solar: 4kWh available
        - Evening period (no future solar expected)
        
        Expected:
        - Mandatory: gets 5kWh (4kWh solar + 1kWh battery)
        - Optional: gets ~0kWh (no solar left, can't use battery)
        
        Validates:
        - Mandatory constraints can use battery
        - Optional constraints limited to available power only
        - Battery used strategically for mandatory
        - Energy conservation maintained
        """
        dt = datetime(2024, 6, 1, 16, 0, 0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=6)
        
        battery = create_test_battery(
            capacity_wh=6000.0,
            initial_soc_percent=50.0,
            min_soc_percent=15.0,
            max_soc_percent=90.0
        )
        
        # Mandatory load
        mandatory_load = TestLoad(name="mandatory")
        mandatory_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=mandatory_load,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=2.5 * 3600,  # 2.5h at 2kW = 5kWh
            power=2000
        )
        mandatory_load.push_live_constraint(dt, mandatory_constraint)
        
        # Optional load
        optional_load = TestLoad(name="optional")
        optional_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=optional_load,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=4 * 3600,  # 4h at 2kW = 8kWh
            power=2000
        )
        optional_load.push_live_constraint(dt, optional_constraint)
        
        # Limited solar: ~4kWh
        pv_forecast = [(dt + timedelta(hours=h), 700.0) for h in range(7)]
        ua_forecast = [(dt + timedelta(hours=h), 100.0) for h in range(7)]
        
        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=0.20/1000.0,
            actionable_loads=[mandatory_load, optional_load],
            battery=battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast
        )
        
        load_commands, _ = solver.solve(with_self_test=True)
        
        # Extract energies
        commands_by_load = {load.name: cmds for load, cmds in load_commands}
        
        mandatory_energy = calculate_energy_from_commands(
            commands_by_load.get("mandatory", []), end_time
        )
        optional_energy = calculate_energy_from_commands(
            commands_by_load.get("optional", []), end_time
        )
        
        # =================================================================
        # VALIDATE: Check constraint priorities
        # =================================================================
        # Find final states  
        mandatory_final = next((c for c in solver._constraints_for_test if c.load.name == "mandatory"), None)
        optional_final = next((c for c in solver._constraints_for_test if c.load.name == "optional"), None)
        
        assert mandatory_final is not None, "Should have mandatory constraint"
        assert optional_final is not None, "Should have optional constraint"

        assert mandatory_final.is_constraint_met(end_time), "Mandatory constraint should be met"
        assert optional_final.is_constraint_met(end_time) is False, "Optional constraint shouldn't be met"

        # Check scores (mandatory should have higher priority)
        mandatory_score = mandatory_constraint.score(dt)
        optional_score = optional_constraint.score(dt)
        
        assert mandatory_score > optional_score, (
            f"Mandatory should have higher priority score\n"
            f"  Mandatory: {mandatory_score}\n"
            f"  Optional: {optional_score}"
        )
        
        # Check that mandatory constraint is prioritized
        # (Both types should have appropriate scores)
        assert mandatory_final.type >= CONSTRAINT_TYPE_MANDATORY_END_TIME, (
            "Mandatory constraint should have mandatory type"
        )
        assert optional_final.type < CONSTRAINT_TYPE_MANDATORY_END_TIME, (
            "Optional constraint should have non-mandatory type"
        )
        
        # Battery should discharge (net negative)
        battery_power = solver._battery_power_external_consumption_for_test
        net_battery_energy = np.sum(battery_power * solver._durations_s) / 3600.0
        
        # Should discharge to support mandatory
        assert net_battery_energy < -500, (
            f"Battery should discharge for mandatory load\n"
            f"  Net battery energy: {net_battery_energy:.1f}Wh\n"
            f"  Expected: negative (discharging)"
        )
        
        # Energy conservation validated by with_self_test
        
        print(f"✅ Mandatory vs optional with battery test passed!")
        print(f"   - Mandatory score: {mandatory_score}")
        print(f"   - Optional score: {optional_score}")
        print(f"   - Mandatory has priority: ✓")
        print(f"   - Battery net: {net_battery_energy:.0f}Wh")

    def test_before_battery_vs_after_battery_constraints(self):
        """
        DEEP TEST: Before-battery constraints run before battery charging,
        after-battery constraints only with surplus.
        
        Setup:
        - Battery: 8kWh, 30% charged
        - Load A: BEFORE_BATTERY, wants 10kWh
        - Load B: FILLER (after battery), wants 6kWh
        - Solar: 12kWh available
        
        Expected:
        - Load A gets power during battery charging phase
        - Load B only gets power when battery full + surplus
        - Battery charges from solar
        
        Validates:
        - Phase ordering: before_battery → battery → after_battery
        - After-battery only runs with surplus
        - Energy conservation maintained
        """
        dt = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=6)
        
        battery = create_test_battery(
            capacity_wh=8000.0,
            initial_soc_percent=30.0,
            min_soc_percent=10.0,
            max_soc_percent=90.0,
            max_charge_power=4000.0
        )
        
        # Before-battery constraint
        load_before = TestLoad(name="before_battery")
        constraint_before = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_before,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=5 * 3600,  # 5h at 2kW = 10kWh
            power=2000
        )
        load_before.push_live_constraint(dt, constraint_before)
        
        # After-battery constraint (lower priority)
        load_after = TestLoad(name="after_battery")
        constraint_after = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_after,
            type=CONSTRAINT_TYPE_FILLER_AUTO,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=3 * 3600,  # 3h at 2kW = 6kWh
            power=2000
        )
        load_after.push_live_constraint(dt, constraint_after)
        
        # Solar: 2kW for 6h = 12kWh
        pv_forecast = [(dt + timedelta(hours=h), 2000.0) for h in range(7)]
        ua_forecast = [(dt + timedelta(hours=h), 100.0) for h in range(7)]
        
        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=0.20/1000.0,
            actionable_loads=[load_before, load_after],
            battery=battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast
        )
        
        load_commands, _ = solver.solve(with_self_test=True)
        
        # =================================================================
        # VALIDATE
        # =================================================================
        # Find final constraint states
        constraint_before_final = next((c for c in solver._constraints_for_test if c.load.name == "before_battery"), None)
        constraint_after_final = next((c for c in solver._constraints_for_test if c.load.name == "after_battery"), None)
        
        assert constraint_before_final is not None
        assert constraint_after_final is not None
        
        # Before-battery is higher priority, should get more progress
        before_progress = constraint_before_final.current_value / constraint_before_final.target_value if constraint_before_final.target_value > 0 else 0
        after_progress = constraint_after_final.current_value / constraint_after_final.target_value if constraint_after_final.target_value > 0 else 0
        
        # Before-battery should get priority
        assert before_progress >= after_progress * 0.9, (
            f"Before-battery should get priority\n"
            f"  Before progress: {100*before_progress:.0f}%\n"
            f"  After progress: {100*after_progress:.0f}%"
        )
        
        # Check battery behavior
        battery_power = solver._battery_power_external_consumption_for_test
        net_battery = np.sum(battery_power * solver._durations_s) / 3600.0
        
        # Battery should charge with available solar
        assert net_battery > -1000, (  # Should not significantly discharge
            f"Battery behavior unexpected\n"
            f"  Net: {net_battery:.1f}Wh"
        )
        
        # Energy conservation validated by with_self_test
        
        # Validate battery SOC bounds
        battery_charge = reconstruct_battery_charge_evolution(
            initial_charge=battery._current_charge_value,
            battery_power=battery_power,
            durations_s=solver._durations_s
        )
        validate_battery_soc_bounds(battery_charge, battery)
        
        print(f"✅ Before/after battery test passed!")
        print(f"   - Before-battery progress: {100*before_progress:.0f}%")
        print(f"   - After-battery progress: {100*after_progress:.0f}%")
        print(f"   - Battery net energy: {net_battery:.0f}Wh")

    def test_user_constraint_priority_boost(self):
        """
        DEEP TEST: User-created constraints get priority boost.
        
        Setup:
        - Load A: FILLER, from_user=True, wants 4kWh
        - Load B: FILLER, from_user=False, wants 4kWh
        - Available: 5kWh total
        
        Expected:
        - Load A gets more power (user boost)
        - Load B gets less
        
        Validates:
        - from_user flag increases constraint score
        - Higher score leads to more power allocation
        - Both constraints same type, user flag is differentiator
        """
        dt = datetime(2024, 6, 1, 10, 0, 0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=5)
        
        # Two identical constraints except from_user flag
        load_user = TestLoad(name="load_from_user")
        constraint_user = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_user,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=2 * 3600,  # 2h at 2kW = 4kWh
            power=2000,
            from_user=True  # User-created
        )
        load_user.push_live_constraint(dt, constraint_user)
        
        load_auto = TestLoad(name="load_automatic")
        constraint_auto = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=load_auto,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=2 * 3600,  # Same 4kWh
            power=2000,
            from_user=False  # Automatic
        )
        load_auto.push_live_constraint(dt, constraint_auto)
        
        # Limited solar: 5kWh = 1kW for 5h
        pv_forecast = [(dt + timedelta(hours=h), 1000.0) for h in range(6)]
        ua_forecast = [(dt + timedelta(hours=h), 50.0) for h in range(6)]
        
        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=0.20/1000.0,
            actionable_loads=[load_user, load_auto],
            battery=None,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast
        )
        
        load_commands, _ = solver.solve(with_self_test=True)
        
        # =================================================================
        # VALIDATE
        # =================================================================
        # Check scores - user should have higher score
        score_user = constraint_user.score(dt)
        score_auto = constraint_auto.score(dt)
        
        assert score_user > score_auto, (
            f"User constraint should have higher score\n"
            f"  User score: {score_user}\n"
            f"  Auto score: {score_auto}"
        )
        
        # Find final states
        constraint_user_final = next((c for c in solver._constraints_for_test if c.load.name == "load_from_user"), None)
        constraint_auto_final = next((c for c in solver._constraints_for_test if c.load.name == "load_automatic"), None)
        
        # Check progress - user should get at least as much as auto
        if constraint_user_final and constraint_auto_final:
            user_progress = constraint_user_final.current_value / constraint_user_final.target_value
            auto_progress = constraint_auto_final.current_value / constraint_auto_final.target_value
            
            assert user_progress >= auto_progress * 0.9, (
                f"User constraint should get priority\n"
                f"  User progress: {100*user_progress:.0f}%\n"
                f"  Auto progress: {100*auto_progress:.0f}%"
            )
        
        # All loads should get commands
        assert len(load_commands) == 2
        
        print(f"✅ User priority boost test passed!")
        print(f"   - User score: {score_user:.0f}")
        print(f"   - Auto score: {score_auto:.0f}")
        print(f"   - User has priority: ✓")


class TestSolverComplexScenarios:
    """Complex multi-load scenarios testing solver logic."""

    def test_three_loads_different_priorities_realistic_day(self):
        """
        DEEP TEST: Three loads with different priorities (simplified version).
        
        Setup:
        - Car: MANDATORY
        - Heater: MANDATORY
        - Pool: OPTIONAL
        - Realistic solar
        
        Validates:
        - Solver handles multiple loads correctly
        - Mandatory constraints prioritized
        - Battery SOC bounds maintained
        - with_self_test passes (energy conservation)
        """
        dt = datetime(2024, 6, 1, 8, 0, 0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=12)  # Shorter period
        
        battery = create_test_battery(
            capacity_wh=8000.0,
            initial_soc_percent=50.0,
            min_soc_percent=20.0,
            max_soc_percent=90.0
        )
        
        # Car: mandatory
        car = TestLoad(name="car")
        car_steps = [LoadCommand(command="ON", power_consign=a * 230 * 3) for a in range(7, 21)]
        car_constraint = MultiStepsPowerLoadConstraint(
            time=dt,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=10000,  # 10kWh
            power_steps=car_steps,
            support_auto=True
        )
        car.push_live_constraint(dt, car_constraint)
        
        # Heater: mandatory
        heater = TestLoad(name="heater")
        heater_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=heater,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=2 * 3600,  # 2h at 2kW = 4kWh
            power=2000
        )
        heater.push_live_constraint(dt, heater_constraint)
        
        # Pool: optional
        pool = TestLoad(name="pool")
        pool_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=pool,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=None,
            initial_value=0,
            target_value=3 * 3600,  # 3h at 2kW = 6kWh
            power=2000
        )
        pool.push_live_constraint(dt, pool_constraint)
        
        # Good solar
        pv_forecast = [(dt + timedelta(hours=h), 4000.0) for h in range(13)]
        ua_forecast = [(dt + timedelta(hours=h), 400.0) for h in range(13)]
        
        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=0.20/1000.0,
            actionable_loads=[car, heater, pool],
            battery=battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast
        )
        
        load_commands, battery_commands = solver.solve(with_self_test=True)
        
        # =================================================================
        # VALIDATE
        # =================================================================
        # Check solver produced output
        assert len(load_commands) == 3
        assert battery_commands is not None
        
        # Find constraint final states
        car_final = next((c for c in solver._constraints_for_test if c.load.name == "car"), None)
        heater_final = next((c for c in solver._constraints_for_test if c.load.name == "heater"), None)
        pool_final = next((c for c in solver._constraints_for_test if c.load.name == "pool"), None)
        
        # Check mandatory constraints are met or nearly met
        car_satisfied = car_final.is_constraint_met(end_time) if car_final else False
        heater_satisfied = heater_final.is_constraint_met(end_time) if heater_final else False
        pool_satisfied = pool_final.is_constraint_met(end_time) if pool_final else False

        assert car_satisfied and heater_satisfied and pool_satisfied

        # At least one mandatory should be satisfied
        assert car_satisfied or heater_satisfied or (
            car_final and car_final.current_value >= car_final.target_value * 0.8
        ), "Mandatory constraints should make good progress"
        
        # Battery SOC bounds
        battery_power = solver._battery_power_external_consumption_for_test
        battery_charge = reconstruct_battery_charge_evolution(
            initial_charge=battery._current_charge_value,
            battery_power=battery_power,
            durations_s=solver._durations_s
        )
        validate_battery_soc_bounds(battery_charge, battery, tolerance_wh=10.0)
        
        print(f"✅ Complex 3-load scenario passed!")
        print(f"   - Car: {'met' if car_satisfied else 'partial'}")
        print(f"   - Heater: {'met' if heater_satisfied else 'partial'}")
        print(f"   - Battery SOC bounds: ✓")
        print(f"   - Energy conservation: ✓ (with_self_test)")
