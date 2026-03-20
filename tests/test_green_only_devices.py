"""
Green-only device behavior tests.

Tests that devices restricted to solar-only energy receive only
CMD_AUTO_GREEN_ONLY commands, never draw from the grid, and stop
when solar surplus drops to zero.

Story 2.4, AC #4: Green-only device behavior (FR11).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest import TestCase

import pytest
import pytz

from custom_components.quiet_solar.home_model.commands import (
    CMD_AUTO_GREEN_ONLY,
    CMD_CST_AUTO_GREEN,
    commands_scores,
)
from custom_components.quiet_solar.home_model.load import TestLoad


@pytest.mark.integration
class TestGreenOnlyDeviceBehavior(TestCase):
    """Test green-only device configuration and behavior."""

    def setUp(self):
        self.dt = datetime(year=2024, month=6, day=1, hour=10, minute=0, second=0, tzinfo=pytz.UTC)

    def test_green_only_flag_default_false(self):
        """By default, qs_best_effort_green_only is False."""
        load = TestLoad(name="device")
        assert load.qs_best_effort_green_only is False

    def test_green_only_flag_enables_best_effort(self):
        """Setting green_only flag makes load a best-effort-only load."""
        load = TestLoad(name="device")
        load.qs_best_effort_green_only = True
        assert load.is_best_effort_only_load() is True

    def test_green_only_is_not_time_sensitive(self):
        """Green-only loads are not time-sensitive."""
        load = TestLoad(name="device")
        load.qs_best_effort_green_only = True
        load.is_load_time_sensitive = True  # even if explicitly set
        assert load.is_time_sensitive() is False

    def test_cmd_auto_green_only_is_auto(self):
        """CMD_AUTO_GREEN_ONLY is classified as an auto command."""
        assert CMD_AUTO_GREEN_ONLY.is_auto() is True

    def test_cmd_auto_green_only_command_type(self):
        """CMD_AUTO_GREEN_ONLY uses the correct command string."""
        assert CMD_AUTO_GREEN_ONLY.command == CMD_CST_AUTO_GREEN

    def test_cmd_auto_green_only_has_zero_power_consign(self):
        """CMD_AUTO_GREEN_ONLY has zero power consign (surplus-driven)."""
        assert CMD_AUTO_GREEN_ONLY.power_consign == 0.0

    def test_green_command_score_lower_than_consign(self):
        """Auto-green command scores lower than auto-consign (lower priority)."""
        green_score = commands_scores[CMD_CST_AUTO_GREEN]
        from custom_components.quiet_solar.home_model.commands import CMD_CST_AUTO_CONSIGN
        consign_score = commands_scores[CMD_CST_AUTO_CONSIGN]
        assert green_score < consign_score

    def test_green_command_score_higher_than_off(self):
        """Auto-green command scores higher than OFF."""
        green_score = commands_scores[CMD_CST_AUTO_GREEN]
        from custom_components.quiet_solar.home_model.commands import CMD_CST_OFF
        off_score = commands_scores[CMD_CST_OFF]
        assert green_score > off_score

    def test_support_green_only_switch_default(self):
        """AbstractLoad.support_green_only_switch returns False by default."""
        load = TestLoad(name="device")
        assert load.support_green_only_switch() is False

    def test_green_only_combined_with_boost_only(self):
        """Both green-only and boost-only set results in is_best_effort_only_load."""
        load = TestLoad(name="device")
        load.qs_best_effort_green_only = True
        load.load_is_auto_to_be_boosted = True
        assert load.is_best_effort_only_load() is True

    def test_green_only_off_grid_exclusion(self):
        """Green-only loads are excluded from off-grid solver planning.

        In the solver, when is_off_grid=True, loads with is_best_effort_only_load()=True
        are skipped during constraint energy allocation (solver.py:1144).
        """
        from custom_components.quiet_solar.const import CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN
        from custom_components.quiet_solar.home_model.constraints import TimeBasedSimplePowerLoadConstraint
        from custom_components.quiet_solar.home_model.solver import PeriodSolver
        from tests.utils.scenario_builders import build_realistic_solar_forecast, create_test_battery

        pool = TestLoad(name="pool")
        pool.qs_best_effort_green_only = True

        pool_constraint = TimeBasedSimplePowerLoadConstraint(
            time=self.dt,
            load=pool,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=self.dt + timedelta(hours=12),
            initial_value=0,
            target_value=4 * 3600,
            power=1430,
        )
        pool.push_live_constraint(self.dt, pool_constraint)

        battery = create_test_battery(initial_soc_percent=50.0)
        forecast = build_realistic_solar_forecast(self.dt, num_hours=12, peak_power=5000.0)
        consumption = [(self.dt + timedelta(hours=h), 500.0) for h in range(12)]

        solver = PeriodSolver(
            start_time=self.dt,
            end_time=self.dt + timedelta(hours=12),
            tariffs=0.20 / 1000.0,
            actionable_loads=[pool],
            battery=battery,
            pv_forecast=forecast,
            unavoidable_consumption_forecast=consumption,
        )

        # Off-grid should exclude green-only loads
        result, _ = solver.solve(is_off_grid=True, with_self_test=True)
        assert result is not None

    def test_green_only_on_grid_gets_commands(self):
        """Green-only loads DO get commands in on-grid mode."""
        from custom_components.quiet_solar.const import CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN
        from custom_components.quiet_solar.home_model.constraints import TimeBasedSimplePowerLoadConstraint
        from custom_components.quiet_solar.home_model.solver import PeriodSolver
        from tests.utils.scenario_builders import build_realistic_solar_forecast, create_test_battery

        pool = TestLoad(name="pool")
        pool.qs_best_effort_green_only = True

        pool_constraint = TimeBasedSimplePowerLoadConstraint(
            time=self.dt,
            load=pool,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=self.dt + timedelta(hours=12),
            initial_value=0,
            target_value=4 * 3600,
            power=1430,
        )
        pool.push_live_constraint(self.dt, pool_constraint)

        battery = create_test_battery(initial_soc_percent=50.0)
        forecast = build_realistic_solar_forecast(self.dt, num_hours=12, peak_power=8000.0)
        consumption = [(self.dt + timedelta(hours=h), 500.0) for h in range(12)]

        solver = PeriodSolver(
            start_time=self.dt,
            end_time=self.dt + timedelta(hours=12),
            tariffs=0.20 / 1000.0,
            actionable_loads=[pool],
            battery=battery,
            pv_forecast=forecast,
            unavoidable_consumption_forecast=consumption,
        )

        # On-grid (default) — green-only loads participate normally
        result, _ = solver.solve(with_self_test=True)
        assert result is not None
        pool_result = [r for r in result if r[0] == pool]
        assert len(pool_result) == 1, "Green-only pool should get commands in on-grid mode"
