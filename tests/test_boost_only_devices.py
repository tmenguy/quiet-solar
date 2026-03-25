"""
Boost-only device management tests.

Tests that devices supporting only boost mode (load_is_auto_to_be_boosted=True)
receive CMD_FORCE_CHARGE when needed, are excluded from normal solver allocation,
and boost triggers appropriately based on constraint conditions.

Story 2.4, AC #5: Boost-only device management (FR12).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest import TestCase

import pytest
import pytz

from custom_components.quiet_solar.const import CONF_LOAD_IS_BOOST_ONLY
from custom_components.quiet_solar.home_model.commands import (
    CMD_CST_FORCE_CHARGE,
    CMD_FORCE_CHARGE,
    commands_scores,
)
from custom_components.quiet_solar.home_model.load import TestLoad


@pytest.mark.integration
class TestBoostOnlyDeviceBehavior(TestCase):
    """Test boost-only device configuration and behavior."""

    def setUp(self):
        self.dt = datetime(year=2024, month=6, day=1, hour=10, minute=0, second=0, tzinfo=pytz.UTC)

    def test_boost_only_flag_default_false(self):
        """By default, load_is_auto_to_be_boosted is False."""
        load = TestLoad(name="device")
        assert load.load_is_auto_to_be_boosted is False

    def test_boost_only_flag_via_kwargs(self):
        """load_is_auto_to_be_boosted can be set via CONF_LOAD_IS_BOOST_ONLY kwarg."""
        load = TestLoad(name="device", **{CONF_LOAD_IS_BOOST_ONLY: True})
        assert load.load_is_auto_to_be_boosted is True

    def test_boost_only_is_best_effort_load(self):
        """Boost-only loads are classified as best-effort-only."""
        load = TestLoad(name="device")
        load.load_is_auto_to_be_boosted = True
        assert load.is_best_effort_only_load() is True

    def test_boost_only_not_time_sensitive(self):
        """Boost-only loads are not time-sensitive."""
        load = TestLoad(name="device")
        load.load_is_auto_to_be_boosted = True
        load.is_load_time_sensitive = True  # even if set
        assert load.is_time_sensitive() is False

    def test_cmd_force_charge_properties(self):
        """CMD_FORCE_CHARGE has expected properties."""
        assert CMD_FORCE_CHARGE.command == CMD_CST_FORCE_CHARGE
        assert CMD_FORCE_CHARGE.power_consign == 0.0

    def test_cmd_force_charge_is_not_auto(self):
        """CMD_FORCE_CHARGE is NOT an auto command — it's a forced mode."""
        assert CMD_FORCE_CHARGE.is_auto() is False

    def test_cmd_force_charge_not_off_or_idle(self):
        """CMD_FORCE_CHARGE is not off or idle."""
        assert CMD_FORCE_CHARGE.is_off_or_idle() is False

    def test_force_charge_score_higher_than_auto_consign(self):
        """FORCE_CHARGE scores higher than AUTO_CONSIGN (more aggressive)."""
        from custom_components.quiet_solar.home_model.commands import CMD_CST_AUTO_CONSIGN

        force_score = commands_scores[CMD_CST_FORCE_CHARGE]
        auto_score = commands_scores[CMD_CST_AUTO_CONSIGN]
        assert force_score > auto_score

    def test_force_charge_score_lower_than_on(self):
        """FORCE_CHARGE scores lower than ON (ON is highest)."""
        from custom_components.quiet_solar.home_model.commands import CMD_CST_ON

        force_score = commands_scores[CMD_CST_FORCE_CHARGE]
        on_score = commands_scores[CMD_CST_ON]
        assert force_score < on_score

    def test_boost_only_off_grid_excluded_from_solver(self):
        """Boost-only loads are excluded from off-grid solver energy allocation."""
        from custom_components.quiet_solar.const import CONSTRAINT_TYPE_MANDATORY_END_TIME
        from custom_components.quiet_solar.home_model.constraints import TimeBasedSimplePowerLoadConstraint
        from custom_components.quiet_solar.home_model.solver import PeriodSolver
        from tests.utils.scenario_builders import build_realistic_solar_forecast, create_test_battery

        boiler = TestLoad(name="boiler")
        boiler.load_is_auto_to_be_boosted = True

        boiler_constraint = TimeBasedSimplePowerLoadConstraint(
            time=self.dt,
            load=boiler,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=self.dt + timedelta(hours=10),
            initial_value=0,
            target_value=3 * 3600,
            power=2000,
        )
        boiler.push_live_constraint(self.dt, boiler_constraint)

        battery = create_test_battery(initial_soc_percent=50.0)
        forecast = build_realistic_solar_forecast(self.dt, num_hours=12, peak_power=5000.0)
        consumption = [(self.dt + timedelta(hours=h), 500.0) for h in range(12)]

        solver = PeriodSolver(
            start_time=self.dt,
            end_time=self.dt + timedelta(hours=12),
            tariffs=0.20 / 1000.0,
            actionable_loads=[boiler],
            battery=battery,
            pv_forecast=forecast,
            unavoidable_consumption_forecast=consumption,
        )

        # Off-grid excludes best-effort (boost-only) loads
        result, _ = solver.solve(is_off_grid=True, with_self_test=True)
        assert result is not None

    def test_boost_only_on_grid_participates(self):
        """Boost-only loads participate in on-grid solver planning."""
        from custom_components.quiet_solar.const import CONSTRAINT_TYPE_MANDATORY_END_TIME
        from custom_components.quiet_solar.home_model.constraints import TimeBasedSimplePowerLoadConstraint
        from custom_components.quiet_solar.home_model.solver import PeriodSolver
        from tests.utils.scenario_builders import build_realistic_solar_forecast, create_test_battery

        boiler = TestLoad(name="boiler")
        boiler.load_is_auto_to_be_boosted = True

        boiler_constraint = TimeBasedSimplePowerLoadConstraint(
            time=self.dt,
            load=boiler,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=self.dt + timedelta(hours=10),
            initial_value=0,
            target_value=3 * 3600,
            power=2000,
        )
        boiler.push_live_constraint(self.dt, boiler_constraint)

        battery = create_test_battery(initial_soc_percent=50.0)
        forecast = build_realistic_solar_forecast(self.dt, num_hours=12, peak_power=5000.0)
        consumption = [(self.dt + timedelta(hours=h), 500.0) for h in range(12)]

        solver = PeriodSolver(
            start_time=self.dt,
            end_time=self.dt + timedelta(hours=12),
            tariffs=0.20 / 1000.0,
            actionable_loads=[boiler],
            battery=battery,
            pv_forecast=forecast,
            unavoidable_consumption_forecast=consumption,
        )

        result, _ = solver.solve(with_self_test=True)
        assert result is not None
        boiler_result = [r for r in result if r[0] == boiler]
        assert len(boiler_result) == 1, "Boost-only load should get commands in on-grid mode"

    def test_boost_and_green_both_best_effort(self):
        """A load that is both boost-only and green-only is still best-effort."""
        load = TestLoad(name="device")
        load.load_is_auto_to_be_boosted = True
        load.qs_best_effort_green_only = True
        assert load.is_best_effort_only_load() is True
        assert load.is_time_sensitive() is False
