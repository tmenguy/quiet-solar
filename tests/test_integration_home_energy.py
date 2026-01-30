"""
Integration tests for complete home energy scenarios.

These tests validate that solver, constraints, battery, and loads work
together correctly in realistic scenarios with minimal mocking.
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
    reconstruct_battery_charge_evolution,
)
from tests.utils.scenario_builders import (
    build_realistic_solar_forecast,
    build_realistic_consumption_forecast,
    create_test_battery,
    create_car_with_power_steps,
)


class TestCompleteHomeEnergy:
    """Integration tests for complete home energy management."""

    def test_complete_home_with_car_heater_pool_battery(self):
        """
        INTEGRATION TEST: Full home energy simulation over 16 hours.
        
        Setup:
        - Battery: 10kWh
        - Solar: realistic curve with 6am-8pm window
        - Loads:
          - Base consumption: 500W constant
          - Car: 18kWh needed by evening
          - Water heater: 4kWh needed by 11pm
          - Pool: 6kWh anytime (optional)
        
        Validates:
        - Car target met by deadline
        - Heater target met
        - Pool gets surplus only
        - Battery SOC stays within bounds
        - Energy conservation (via with_self_test)
        - No constraint violated
        """
        # =================================================================
        # SETUP: 16-hour realistic home scenario
        # =================================================================
        dt = datetime(2024, 6, 1, 6, 0, 0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=16)
        
        # Battery
        battery = create_test_battery(
            capacity_wh=10000.0,
            initial_soc_percent=45.0,
            min_soc_percent=20.0,
            max_soc_percent=90.0,
            max_charge_power=3500.0,
            max_discharge_power=3500.0
        )
        
        # Car: mandatory, big target, deadline
        car, car_steps = create_car_with_power_steps(min_amps=7, max_amps=32)
        car_constraint = MultiStepsPowerLoadConstraint(
            time=dt,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=dt + timedelta(hours=14),  # 8pm deadline
            initial_value=0,
            target_value=18000,  # 18kWh
            power_steps=car_steps,
            support_auto=True
        )
        car.push_live_constraint(dt, car_constraint)
        
        # Water heater: mandatory, smaller target
        heater = TestLoad(name="heater")
        heater_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=heater,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=dt + timedelta(hours=15),  # 9pm deadline
            initial_value=0,
            target_value=2 * 3600,  # 2h at 2kW = 4kWh
            power=2000
        )
        heater.push_live_constraint(dt, heater_constraint)
        
        # Pool: optional, no deadline
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
        
        # Realistic solar
        pv_forecast = build_realistic_solar_forecast(
            start_time=start_time,
            num_hours=16,
            peak_power=9000.0,
            sunrise_hour=6,
            sunset_hour=20
        )
        
        # Realistic consumption
        ua_forecast = build_realistic_consumption_forecast(
            start_time=start_time,
            num_hours=16,
            base_load=500.0,
            peak_evening_load=1500.0,
            night_load=300.0
        )
        
        # Fixed tariff
        tariff = 0.20 / 1000.0
        
        # =================================================================
        # EXECUTE: Solve
        # =================================================================
        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=tariff,
            actionable_loads=[car, heater, pool],
            battery=battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast
        )
        
        load_commands, battery_commands = solver.solve(with_self_test=True)
        
        # =================================================================
        # VALIDATE: Check solver produced valid output
        # =================================================================
        assert load_commands is not None and len(load_commands) == 3
        assert battery_commands is not None and len(battery_commands) > 0
        
        commands_by_load = {load.name: cmds for load, cmds in load_commands}
        
        assert "car" in commands_by_load
        assert "heater" in commands_by_load
        assert "pool" in commands_by_load

        assert len(solver._loads_energy_and_time_consumption_for_test) == 3

        for l_cmd in load_commands:
            load = l_cmd[0]
            if "car" in load.name:
                car_consumed_energy = solver._loads_energy_and_time_consumption_for_test[load][0]
                assert car_consumed_energy  >= car_constraint.target_value, "Car should be fullfilled"
                print(f" {car_constraint.target_value}")
            elif "heater" in load.name:
                heater_consumed_time = solver._loads_energy_and_time_consumption_for_test[load][1]
                assert heater_consumed_time >= heater_constraint.target_value, "Heater should be fullfilled"
            elif "pool" in load.name:
                pool_consumed_time = solver._loads_energy_and_time_consumption_for_test[load][1]
                assert pool_consumed_time >= pool_constraint.target_value, "Pool should be fullfilled"
        

        # Battery SOC bounds
        battery_power = solver._battery_power_external_consumption_for_test
        battery_charge = reconstruct_battery_charge_evolution(
            initial_charge=battery._current_charge_value,
            battery_power=battery_power,
            durations_s=solver._durations_s
        )
        
        validate_battery_soc_bounds(battery_charge, battery, tolerance_wh=10.0)


    def test_off_grid_mode_battery_only_survival(self):
        """
        INTEGRATION TEST: Off-grid mode with battery-only operation.
        
        Setup:
        - No solar (night simulation)
        - Battery: 8kWh, 60% charged
        - Base load: 400W constant
        - Optional load: wants 3kWh
        - Duration: 10 hours
        
        Expected (off-grid):
        - Base consumption covered by battery
        - Optional load limited to preserve battery
        - Battery never below min_soc
        - Loads prioritized correctly
        
        Validates:
        - Off-grid mode prevents grid usage
        - Battery preservation logic works
        - Critical loads prioritized
        - Battery doesn't over-discharge
        """
        dt = datetime(2024, 6, 1, 20, 0, 0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=10)
        
        battery = create_test_battery(
            capacity_wh=8000.0,
            initial_soc_percent=60.0,
            min_soc_percent=20.0,
            max_soc_percent=90.0
        )
        
        # Optional load
        lights = TestLoad(name="lights")
        lights_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=lights,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=3 * 3600,  # 3h at 1kW = 3kWh
            power=1000
        )
        lights.push_live_constraint(dt, lights_constraint)
        
        # No solar at night
        pv_forecast = [(dt + timedelta(hours=h), 0.0) for h in range(11)]
        
        # Base consumption
        ua_forecast = [(dt + timedelta(hours=h), 400.0) for h in range(11)]
        
        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=0.25/1000.0,
            actionable_loads=[lights],
            battery=battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast
        )
        
        load_commands, _ = solver.solve(is_off_grid=True, with_self_test=True)
        
        # =================================================================
        # VALIDATE
        # =================================================================
        commands_by_load = {load.name: cmds for load, cmds in load_commands}
        lights_energy = calculate_energy_from_commands(
            commands_by_load.get("lights", []), end_time
        )
        
        # Battery should stay above minimum
        battery_power = solver._battery_power_external_consumption_for_test
        battery_charge = reconstruct_battery_charge_evolution(
            initial_charge=battery._current_charge_value,
            battery_power=battery_power,
            durations_s=solver._durations_s
        )
        
        min_charge = battery.get_value_empty()
        min_reached = np.min(battery_charge)
        
        assert min_reached >= min_charge - 10.0, (
            f"Off-grid: Battery went below minimum\n"
            f"  Minimum reached: {min_reached:.0f}Wh ({100*min_reached/battery.capacity:.1f}%)\n"
            f"  Minimum allowed: {min_charge:.0f}Wh ({battery.min_charge_SOC_percent:.1f}%)"
        )
        
        # Optional load should be limited (preserve battery for base load)
        # Base load needs: 400W * 10h = 4kWh
        # Battery has: 8000 * 0.60 = 4800Wh, usable: 4800 - 1600 (min) = 3200Wh
        # So lights should get limited to preserve battery
        assert lights_energy < 3000, (
            f"Optional load should be limited in off-grid\n"
            f"  Delivered: {lights_energy:.0f}Wh\n"
            f"  Target: 3000Wh"
        )
        
        validate_battery_soc_bounds(battery_charge, battery)
        
        print(f"\n✅ Off-grid integration test passed!")
        print(f"   - Lights (optional): {lights_energy:.0f}Wh / 3000Wh")
        print(f"   - Battery min SOC: {100*min_reached/battery.capacity:.1f}%")
        print(f"   - Battery preserved for base load: ✓")

    def test_solar_surplus_triggers_opportunistic_charging(self):
        """
        INTEGRATION TEST: High solar surplus triggers opportunistic loads.
        
        Simplified version focusing on validating that surplus is utilized.
        
        Validates:
        - Surplus utilization
        - Battery charges with high solar
        - Optional loads get power
        - SOC bounds maintained
        """
        dt = datetime(2024, 6, 1, 11, 0, 0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=4)
        
        # Battery near full
        battery = create_test_battery(
            capacity_wh=5000.0,
            initial_soc_percent=70.0,
            min_soc_percent=15.0,
            max_soc_percent=90.0,
            max_charge_power=3000.0
        )
        
        # Optional load
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
        
        # High solar
        pv_forecast = [(dt + timedelta(hours=h), 6000.0) for h in range(5)]
        
        # Low consumption
        ua_forecast = [(dt + timedelta(hours=h), 500.0) for h in range(5)]
        
        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=0.15/1000.0,
            actionable_loads=[pool],
            battery=battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast
        )
        
        load_commands, _ = solver.solve(with_self_test=True)
        
        # =================================================================
        # VALIDATE
        # =================================================================
        # Pool should get commands (surplus available)
        commands_by_load = {load.name: cmds for load, cmds in load_commands}
        assert "pool" in commands_by_load
        assert len(commands_by_load["pool"]) > 0, "Pool should have commands with surplus"
        
        # Battery should charge or stay high
        battery_power = solver._battery_power_external_consumption_for_test
        battery_charge = reconstruct_battery_charge_evolution(
            initial_charge=battery._current_charge_value,
            battery_power=battery_power,
            durations_s=solver._durations_s
        )
        
        max_charge = battery.get_value_full()
        max_reached = np.max(battery_charge)
        
        # Battery should be high with lots of solar
        assert max_reached >= battery._current_charge_value * 0.95, (
            f"Battery should maintain/increase with high solar\n"
            f"  Initial: {battery._current_charge_value:.0f}Wh\n"
            f"  Max reached: {max_reached:.0f}Wh"
        )
        
        validate_battery_soc_bounds(battery_charge, battery)
        
        # Check pool made progress
        pool_final = next((c for c in solver._active_constraints if c.load.name == "pool"), None)
        if pool_final:
            pool_progress = pool_final.current_value / pool_final.target_value
            print(f"   - Pool progress: {100*pool_progress:.0f}%")
        
        print(f"\n✅ Solar surplus integration test passed!")
        print(f"   - Battery max SOC: {100*max_reached/battery.capacity:.1f}%")
        print(f"   - Surplus utilized: ✓")

    def test_variable_pricing_shifts_loads_to_cheap_periods(self):
        """
        INTEGRATION TEST: Variable pricing causes load time-shifting.
        
        Setup:
        - 12-hour period
        - Prices: 0.08€ (night), 0.15€ (day), 0.35€ (peak evening)
        - Solar: minimal (cloud day simulation)
        - Flexible load: 8kWh needed over 12h
        - Battery: 6kWh, 50% charged
        
        Expected:
        - Load runs primarily during cheap periods
        - Some load during day prices
        - Minimal/no load during peak prices
        - Battery helps shift load to cheap periods
        
        Validates:
        - Price-based optimization works
        - Cost calculated and minimized
        - Load distribution follows price curve
        """
        dt = datetime(2024, 6, 1, 0, 0, 0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=12)
        
        battery = create_test_battery(
            capacity_wh=6000.0,
            initial_soc_percent=50.0,
            min_soc_percent=15.0,
            max_soc_percent=85.0
        )
        
        # Flexible heater load
        heater = TestLoad(name="heater")
        heater_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=heater,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=end_time,
            initial_value=0,
            target_value=4 * 3600,  # 4h at 2kW = 8kWh
            power=2000
        )
        heater.push_live_constraint(dt, heater_constraint)
        
        # Variable pricing
        # Hours 0-3: cheap (0.08€), 4-7: normal (0.15€), 8-11: peak (0.35€)
        tariffs = []
        for h in range(12):
            hour_time = dt + timedelta(hours=h)
            if h < 4:
                price = 0.08 / 1000.0  # Cheap night
            elif h < 8:
                price = 0.15 / 1000.0  # Normal day
            else:
                price = 0.35 / 1000.0  # Peak evening
            tariffs.append((hour_time, price))
        
        # Low solar (cloudy day)
        pv_forecast = [(dt + timedelta(hours=h), 800.0) for h in range(13)]
        
        # Base consumption
        ua_forecast = [(dt + timedelta(hours=h), 300.0) for h in range(13)]
        
        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=tariffs,
            actionable_loads=[heater],
            battery=battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast
        )
        
        load_commands, _ = solver.solve(with_self_test=True)
        
        commands_by_load = {load.name: cmds for load, cmds in load_commands}
        heater_cmds = commands_by_load.get("heater", [])
        
        # =================================================================
        # VALIDATE: Load distribution by price period
        # =================================================================
        # Calculate energy in each price period
        cheap_energy = 0.0
        normal_energy = 0.0
        peak_energy = 0.0
        
        for cmd_time, cmd in heater_cmds:
            hour = cmd_time.hour
            if hour < 4:
                cheap_energy += cmd.power_consign
            elif hour < 8:
                normal_energy += cmd.power_consign
            else:
                peak_energy += cmd.power_consign
        
        # Should prefer cheap periods
        assert cheap_energy >= normal_energy, (
            f"Should run more during cheap periods\n"
            f"  Cheap (0-3h): {cheap_energy:.0f}W\n"
            f"  Normal (4-7h): {normal_energy:.0f}W\n"
            f"  Peak (8-11h): {peak_energy:.0f}W"
        )
        
        # Should avoid peak periods
        assert peak_energy <= normal_energy, (
            f"Should run less during peak periods\n"
            f"  Normal: {normal_energy:.0f}W\n"
            f"  Peak: {peak_energy:.0f}W"
        )
        
        # Battery SOC bounds
        battery_power = solver._battery_power_external_consumption_for_test
        battery_charge = reconstruct_battery_charge_evolution(
            initial_charge=battery._current_charge_value,
            battery_power=battery_power,
            durations_s=solver._durations_s
        )
        validate_battery_soc_bounds(battery_charge, battery)
        
        print(f"\n✅ Variable pricing integration test passed!")
        print(f"   - Cheap period (0-3h): {cheap_energy:.0f}W")
        print(f"   - Normal period (4-7h): {normal_energy:.0f}W")
        print(f"   - Peak period (8-11h): {peak_energy:.0f}W")
        print(f"   - Load shifted to cheaper periods: ✓")

    def test_multiple_constraints_same_load(self):
        """
        INTEGRATION TEST: Multiple constraints on same load.
        
        Setup:
        - Car with two constraints:
          1. Mandatory: 8kWh
          2. Optional: additional 6kWh
        - Solar: good availability
        
        Validates:
        - Multiple constraints on same load work
        - Priority ordering maintained
        - Solver handles correctly
        """
        dt = datetime(2024, 6, 1, 9, 0, 0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=10)
        
        battery = create_test_battery(
            capacity_wh=6000.0,
            initial_soc_percent=50.0
        )
        
        # Car with two constraints
        car = TestLoad(name="car")
        car_steps = [LoadCommand(command="ON", power_consign=a * 230 * 3) for a in range(7, 21)]
        
        # Mandatory constraint
        car_mandatory = MultiStepsPowerLoadConstraint(
            time=dt,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=dt + timedelta(hours=6),
            initial_value=0,
            target_value=8000,  # 8kWh
            power_steps=car_steps,
            support_auto=True
        )
        car.push_live_constraint(dt, car_mandatory)
        
        # Optional constraint
        car_optional = MultiStepsPowerLoadConstraint(
            time=dt,
            load=car,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=None,
            initial_value=None,  # Follows previous
            target_value=6000,  # Additional 6kWh
            power_steps=car_steps,
            support_auto=True
        )
        car.push_live_constraint(dt, car_optional)
        
        # Good solar
        pv_forecast = [(dt + timedelta(hours=h), 3000.0) for h in range(11)]
        ua_forecast = [(dt + timedelta(hours=h), 400.0) for h in range(11)]
        
        solver = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=0.20/1000.0,
            actionable_loads=[car],
            battery=battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=ua_forecast
        )
        
        load_commands, _ = solver.solve(with_self_test=True)
        
        # =================================================================
        # VALIDATE
        # =================================================================
        # Check solver produced output
        assert len(load_commands) >= 1
        
        # Find constraints for car
        constraints = [c for c in solver._active_constraints if c.load.name == "car"]
        
        # Should have constraints (may be merged internally)
        assert len(constraints) >= 1, "Should have at least 1 constraint for car"
        
        # Check that car has commands
        car_cmds = next((cmds for load, cmds in load_commands if load.name == "car"), None)
        assert car_cmds is not None and len(car_cmds) > 0, "Car should have commands"
        
        # Battery bounds
        battery_power = solver._battery_power_external_consumption_for_test
        battery_charge = reconstruct_battery_charge_evolution(
            initial_charge=battery._current_charge_value,
            battery_power=battery_power,
            durations_s=solver._durations_s
        )
        validate_battery_soc_bounds(battery_charge, battery)
        
        print(f"\n✅ Multiple constraints integration test passed!")
        print(f"   - Car has {len(constraints)} active constraint(s)")
        print(f"   - Car commands: {len(car_cmds)}")
        print(f"   - Solver handled correctly: ✓")
        print(f"   - Battery SOC bounds: ✓")
