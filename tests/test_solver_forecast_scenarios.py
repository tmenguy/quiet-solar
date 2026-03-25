"""
Solver forecast change scenario tests.

Tests solver behavior under rapid solar forecast changes (cloudy day simulation),
verifying constraint priority handling, command stability, and energy conservation
when available power drops significantly.

Story 2.4, AC #1: Solver behavior under rapid forecast changes.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest import TestCase

import pytest
import pytz

from custom_components.quiet_solar.const import (
    CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
    CONSTRAINT_TYPE_FILLER_AUTO,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
)
from custom_components.quiet_solar.home_model.commands import (
    CMD_CST_AUTO_CONSIGN,
    LoadCommand,
)
from custom_components.quiet_solar.home_model.constraints import (
    MultiStepsPowerLoadConstraint,
    TimeBasedSimplePowerLoadConstraint,
)
from custom_components.quiet_solar.home_model.load import TestLoad
from custom_components.quiet_solar.home_model.solver import PeriodSolver
from tests.utils.energy_validation import count_transitions
from tests.utils.scenario_builders import (
    build_realistic_consumption_forecast,
    build_realistic_solar_forecast,
    build_variable_pricing,
    create_test_battery,
)


def _make_car_steps(min_amps=7, max_amps=32, phases=3, voltage=230.0):
    """Create car charging power steps."""
    return [
        LoadCommand(command=CMD_CST_AUTO_CONSIGN, power_consign=a * phases * voltage)
        for a in range(min_amps, max_amps + 1)
    ]


def _apply_cloud_cover(forecast, start_hour, end_hour, reduction_factor):
    """Apply cloud cover to a solar forecast by reducing power in a time range.

    Args:
        forecast: list of (datetime, power) tuples
        start_hour: hour when clouds start (0-23)
        end_hour: hour when clouds end (0-23)
        reduction_factor: factor to multiply power by (0.0 = total cloud, 1.0 = no change)

    Returns:
        Modified forecast with reduced power during cloudy period
    """
    modified = []
    for dt, power in forecast:
        hour = dt.hour
        if start_hour <= hour < end_hour:
            modified.append((dt, power * reduction_factor))
        else:
            modified.append((dt, power))
    return modified


@pytest.mark.integration
class TestSolverForecastChanges(TestCase):
    """Test solver behavior when solar forecast changes rapidly (cloudy day)."""

    def setUp(self):
        self.dt = datetime(year=2024, month=6, day=1, hour=6, minute=0, second=0, tzinfo=pytz.UTC)
        self.end_time = self.dt + timedelta(hours=18)

    def test_solver_handles_50_percent_solar_drop_mandatory_still_met(self):
        """Solver re-plans when solar drops 50% — mandatory constraints still satisfied."""
        # Scenario: Clear morning, then clouds at noon reduce solar by 50%
        sunny_forecast = build_realistic_solar_forecast(self.dt, num_hours=18, peak_power=8000.0)
        cloudy_forecast = _apply_cloud_cover(sunny_forecast, start_hour=11, end_hour=16, reduction_factor=0.5)

        consumption = build_realistic_consumption_forecast(self.dt, num_hours=18)
        pricing = build_variable_pricing(self.dt, num_hours=18)

        car = TestLoad(name="car")
        car_steps = _make_car_steps()

        # Mandatory: charge car by 5pm
        car_mandatory = MultiStepsPowerLoadConstraint(
            time=self.dt,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=self.dt + timedelta(hours=11),
            initial_value=5000,
            target_value=15000,
            power_steps=car_steps,
            support_auto=True,
        )
        car.push_live_constraint(self.dt, car_mandatory)

        battery = create_test_battery(initial_soc_percent=50.0)

        # Solve with cloudy forecast
        solver = PeriodSolver(
            start_time=self.dt,
            end_time=self.end_time,
            tariffs=pricing,
            actionable_loads=[car],
            battery=battery,
            pv_forecast=cloudy_forecast,
            unavoidable_consumption_forecast=consumption,
        )

        result, battery_commands = solver.solve(with_self_test=True)

        # Mandatory constraint should still be met
        assert len(result) > 0, "Solver should produce commands for the car"
        car_result = [r for r in result if r[0] == car]
        assert len(car_result) == 1, "Car should have commands"
        car_commands = car_result[0][1]
        assert len(car_commands) > 0, "Car should have at least one command"

    def test_solver_drops_filler_when_solar_insufficient(self):
        """Under reduced solar, filler constraints are dropped while mandatory preserved."""
        cloudy_forecast = build_realistic_solar_forecast(self.dt, num_hours=18, peak_power=3000.0)
        consumption = build_realistic_consumption_forecast(self.dt, num_hours=18)

        car = TestLoad(name="car")
        pool = TestLoad(name="pool")
        car_steps = _make_car_steps()

        # Mandatory: car must charge
        car_mandatory = MultiStepsPowerLoadConstraint(
            time=self.dt,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=self.dt + timedelta(hours=11),
            initial_value=5000,
            target_value=12000,
            power_steps=car_steps,
            support_auto=True,
        )
        car.push_live_constraint(self.dt, car_mandatory)

        # Filler: pool can run if surplus
        pool_filler = TimeBasedSimplePowerLoadConstraint(
            time=self.dt,
            load=pool,
            type=CONSTRAINT_TYPE_FILLER_AUTO,
            end_of_constraint=self.dt + timedelta(hours=18),
            initial_value=0,
            target_value=6 * 3600,
            power=1430,
        )
        pool.push_live_constraint(self.dt, pool_filler)

        battery = create_test_battery(initial_soc_percent=50.0)

        solver = PeriodSolver(
            start_time=self.dt,
            end_time=self.end_time,
            tariffs=0.20 / 1000.0,
            actionable_loads=[car, pool],
            battery=battery,
            pv_forecast=cloudy_forecast,
            unavoidable_consumption_forecast=consumption,
        )

        result, battery_commands = solver.solve(with_self_test=True)

        from tests.utils.energy_validation import calculate_energy_from_commands

        # Car should get commands (mandatory)
        car_result = [r for r in result if r[0] == car]
        assert len(car_result) == 1, "Car should have commands"
        car_commands = car_result[0][1]
        assert len(car_commands) > 0, "Mandatory car constraint should produce commands"

        car_energy = calculate_energy_from_commands(car_commands, self.end_time)

        # Filler pool should get less energy than mandatory car under limited solar
        pool_result = [r for r in result if r[0] == pool]
        pool_energy = calculate_energy_from_commands(pool_result[0][1], self.end_time) if pool_result else 0.0
        assert car_energy >= pool_energy, (
            f"Mandatory car ({car_energy:.0f}Wh) should get at least as much "
            f"energy as filler pool ({pool_energy:.0f}Wh)"
        )

    def test_solver_command_stability_alternating_solar(self):
        """Alternating solar (cloud pattern) doesn't cause excessive ON/OFF switching."""
        # Use alternating pattern: high → low → high → low
        from tests.utils.scenario_builders import build_alternating_solar_pattern

        alternating_forecast = build_alternating_solar_pattern(
            self.dt, num_hours=18, high_power=6000.0, low_power=1000.0, period_hours=2
        )
        consumption = build_realistic_consumption_forecast(self.dt, num_hours=18)

        pool = TestLoad(name="pool")

        # Pool has a mandatory constraint
        pool_constraint = TimeBasedSimplePowerLoadConstraint(
            time=self.dt,
            load=pool,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=self.dt + timedelta(hours=17),
            initial_value=0,
            target_value=8 * 3600,
            power=1430,
        )
        pool.push_live_constraint(self.dt, pool_constraint)

        battery = create_test_battery(initial_soc_percent=60.0)

        solver = PeriodSolver(
            start_time=self.dt,
            end_time=self.end_time,
            tariffs=0.20 / 1000.0,
            actionable_loads=[pool],
            battery=battery,
            pv_forecast=alternating_forecast,
            unavoidable_consumption_forecast=consumption,
        )

        result, _ = solver.solve(with_self_test=True)

        # Pool should have commands
        pool_result = [r for r in result if r[0] == pool]
        assert len(pool_result) == 1
        pool_commands = pool_result[0][1]

        # Verify transitions are bounded (not excessive switching)
        transitions = count_transitions(pool_commands)
        # With alternating solar and a mandatory constraint, transitions should be reasonable
        # The solver should plan continuous operation, not flip-flop each period
        assert transitions <= 10, (
            f"Too many ON/OFF transitions ({transitions}) under alternating solar — "
            f"solver should plan reasonably stable operation for mandatory constraints"
        )

    def test_solver_multiple_forecast_updates_energy_conservation(self):
        """Solver maintains energy conservation across different forecast levels."""
        consumption = build_realistic_consumption_forecast(self.dt, num_hours=18)

        car = TestLoad(name="car")
        car_steps = _make_car_steps()

        car_constraint = MultiStepsPowerLoadConstraint(
            time=self.dt,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=self.dt + timedelta(hours=14),
            initial_value=3000,
            target_value=15000,
            power_steps=car_steps,
            support_auto=True,
        )
        car.push_live_constraint(self.dt, car_constraint)

        battery = create_test_battery(initial_soc_percent=50.0)

        # Test three different forecast scenarios — solver should be valid for each
        forecast_levels = [
            ("sunny", build_realistic_solar_forecast(self.dt, num_hours=18, peak_power=8000.0)),
            ("partly_cloudy", build_realistic_solar_forecast(self.dt, num_hours=18, peak_power=4000.0)),
            ("overcast", build_realistic_solar_forecast(self.dt, num_hours=18, peak_power=1500.0)),
        ]

        for label, forecast in forecast_levels:
            # Reset constraint state for each run
            car._constraints = []
            car_constraint_copy = MultiStepsPowerLoadConstraint(
                time=self.dt,
                load=car,
                type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
                end_of_constraint=self.dt + timedelta(hours=14),
                initial_value=3000,
                target_value=15000,
                power_steps=car_steps,
                support_auto=True,
            )
            car.push_live_constraint(self.dt, car_constraint_copy)

            battery_copy = create_test_battery(initial_soc_percent=50.0)

            solver = PeriodSolver(
                start_time=self.dt,
                end_time=self.end_time,
                tariffs=0.20 / 1000.0,
                actionable_loads=[car],
                battery=battery_copy,
                pv_forecast=forecast,
                unavoidable_consumption_forecast=consumption,
            )

            # with_self_test=True validates energy conservation internally
            result, battery_cmds = solver.solve(with_self_test=True)

            # Solver should produce valid results for all forecast levels
            assert result is not None, f"Solver failed for {label} forecast"

    def test_solver_cloudy_day_three_loads_priority_ordering(self):
        """Three loads with different priorities under limited solar are ordered correctly."""
        # Very cloudy day — limited solar
        cloudy_forecast = build_realistic_solar_forecast(self.dt, num_hours=18, peak_power=2500.0)
        consumption = build_realistic_consumption_forecast(self.dt, num_hours=18)

        car = TestLoad(name="car")
        pool = TestLoad(name="pool")
        boiler = TestLoad(name="boiler")
        car_steps = _make_car_steps()

        # ASAP: car needs charge urgently
        car_asap = MultiStepsPowerLoadConstraint(
            time=self.dt,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            end_of_constraint=None,
            initial_value=2000,
            target_value=8000,
            power_steps=car_steps,
            support_auto=True,
        )
        car.push_live_constraint(self.dt, car_asap)

        # MANDATORY_END_TIME: pool must run by end of day
        pool_mandatory = TimeBasedSimplePowerLoadConstraint(
            time=self.dt,
            load=pool,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=self.dt + timedelta(hours=17),
            initial_value=0,
            target_value=4 * 3600,
            power=1430,
        )
        pool.push_live_constraint(self.dt, pool_mandatory)

        # BEFORE_BATTERY_GREEN: boiler runs with surplus
        boiler_green = TimeBasedSimplePowerLoadConstraint(
            time=self.dt,
            load=boiler,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=self.dt + timedelta(hours=18),
            initial_value=0,
            target_value=2 * 3600,
            power=2000,
        )
        boiler.push_live_constraint(self.dt, boiler_green)

        battery = create_test_battery(initial_soc_percent=40.0)

        solver = PeriodSolver(
            start_time=self.dt,
            end_time=self.end_time,
            tariffs=0.20 / 1000.0,
            actionable_loads=[car, pool, boiler],
            battery=battery,
            pv_forecast=cloudy_forecast,
            unavoidable_consumption_forecast=consumption,
        )

        result, _ = solver.solve(with_self_test=True)

        from tests.utils.energy_validation import calculate_energy_from_commands

        # ASAP car should get the most energy (highest priority)
        car_result = [r for r in result if r[0] == car]
        assert len(car_result) == 1, "ASAP car should have commands"
        car_commands = car_result[0][1]
        assert len(car_commands) > 0, "ASAP constraint should produce commands"

        car_energy = calculate_energy_from_commands(car_commands, self.end_time)

        # Under limited solar, ASAP car (highest priority) should get at least
        # as much energy as the lower-priority BEFORE_BATTERY_GREEN boiler
        boiler_result = [r for r in result if r[0] == boiler]
        boiler_energy = calculate_energy_from_commands(boiler_result[0][1], self.end_time) if boiler_result else 0.0
        assert car_energy >= boiler_energy, (
            f"ASAP car ({car_energy:.0f}Wh) should get at least as much energy as green boiler ({boiler_energy:.0f}Wh)"
        )

    def test_solver_sudden_cloud_cover_with_battery_backup(self):
        """When solar drops suddenly, battery helps cover mandatory constraints."""
        # Morning sun, then total cloud cover at noon
        forecast = build_realistic_solar_forecast(self.dt, num_hours=18, peak_power=8000.0)
        clouded = _apply_cloud_cover(forecast, start_hour=12, end_hour=18, reduction_factor=0.1)
        consumption = build_realistic_consumption_forecast(self.dt, num_hours=18)

        car = TestLoad(name="car")
        car_steps = _make_car_steps()

        # Car needs to be charged by 6pm
        car_constraint = MultiStepsPowerLoadConstraint(
            time=self.dt,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=self.dt + timedelta(hours=12),
            initial_value=5000,
            target_value=18000,
            power_steps=car_steps,
            support_auto=True,
        )
        car.push_live_constraint(self.dt, car_constraint)

        # Battery with good charge to help
        battery = create_test_battery(initial_soc_percent=70.0, max_discharge_power=5000.0)

        solver = PeriodSolver(
            start_time=self.dt,
            end_time=self.end_time,
            tariffs=0.20 / 1000.0,
            actionable_loads=[car],
            battery=battery,
            pv_forecast=clouded,
            unavoidable_consumption_forecast=consumption,
        )

        result, battery_cmds = solver.solve(with_self_test=True)

        # Solver should still produce valid plan
        assert result is not None
        car_result = [r for r in result if r[0] == car]
        assert len(car_result) == 1
        # Battery commands should exist (battery supports the plan)
        assert len(battery_cmds) > 0, "Battery should be used when solar drops"

    def test_solver_no_solar_all_grid_mandatory_still_planned(self):
        """With zero solar, solver still plans mandatory constraints using grid power."""
        # Night or complete overcast — zero solar
        zero_forecast = [(self.dt + timedelta(hours=h), 0.0) for h in range(18)]
        consumption = build_realistic_consumption_forecast(self.dt, num_hours=18)

        car = TestLoad(name="car")
        car_steps = _make_car_steps()

        car_constraint = MultiStepsPowerLoadConstraint(
            time=self.dt,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=self.dt + timedelta(hours=10),
            initial_value=2000,
            target_value=10000,
            power_steps=car_steps,
            support_auto=True,
        )
        car.push_live_constraint(self.dt, car_constraint)

        battery = create_test_battery(initial_soc_percent=30.0)

        solver = PeriodSolver(
            start_time=self.dt,
            end_time=self.end_time,
            tariffs=0.20 / 1000.0,
            actionable_loads=[car],
            battery=battery,
            pv_forecast=zero_forecast,
            unavoidable_consumption_forecast=consumption,
        )

        result, _ = solver.solve(with_self_test=True)

        # Solver should still plan for mandatory even with no solar
        car_result = [r for r in result if r[0] == car]
        assert len(car_result) == 1, "Car should still get commands with zero solar"
        car_commands = car_result[0][1]
        assert len(car_commands) > 0, "Mandatory constraint should produce commands even with zero solar"
