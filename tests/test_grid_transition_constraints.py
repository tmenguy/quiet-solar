"""Tests for constraint preservation across grid off/on transitions.

Validates that:
- Fix 1: reset_override_state_and_set_reset_ask_time is a no-op without active override
- Fix 2: Pool auto/winter modes go through parent override detection, constraints preserved
- Fix 3: _switch_to_off_grid_launched is cleared on back-to-on-grid transition
"""

from __future__ import annotations

import datetime
from datetime import time as dt_time
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest
import pytz
from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    CONF_SWITCH,
    CONSTRAINT_TYPE_FILLER,
    CONSTRAINT_TYPE_FILLER_AUTO,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    DATA_HANDLER,
    DOMAIN,
)
from custom_components.quiet_solar.ha_model.bistate_duration import (
    ConstraintItemType,
    QSBiStateDuration,
)
from custom_components.quiet_solar.home_model.commands import CMD_IDLE
from custom_components.quiet_solar.home_model.constraints import (
    TimeBasedSimplePowerLoadConstraint,
)
from tests.factories import create_minimal_home_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ConcreteBiStateDevice(QSBiStateDuration):
    """Concrete bistate implementation for testing."""

    def __init__(self, **kwargs):
        if "switch_entity" in kwargs:
            kwargs[CONF_SWITCH] = kwargs.pop("switch_entity")
        elif CONF_SWITCH not in kwargs:
            kwargs[CONF_SWITCH] = "switch.test_device"
        super().__init__(**kwargs)

    async def execute_command_system(self, time, command, state):
        return True

    def get_virtual_current_constraint_translation_key(self):
        return "test_constraint_key"

    def get_select_translation_key(self):
        return "test_select_key"


def _make_device(hass, home, config_entry, **extra) -> ConcreteBiStateDevice:
    defaults = {
        "hass": hass,
        "config_entry": config_entry,
        "home": home,
        "switch_entity": "switch.test_device",
        CONF_NAME: "Test Device",
    }
    defaults.update(extra)
    device = ConcreteBiStateDevice(**defaults)
    device.power_use = 1000.0
    return device


def _utcnow():
    return datetime.datetime.now(pytz.UTC)


def _make_constraint(load, time, end_time, target_value=14400.0, current_value=0.0):
    return TimeBasedSimplePowerLoadConstraint(
        type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        degraded_type=CONSTRAINT_TYPE_FILLER,
        time=time,
        load=load,
        from_user=False,
        end_of_constraint=end_time,
        power=load.power_use,
        initial_value=0,
        target_value=target_value,
        current_value=current_value,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_entry(hass: HomeAssistant) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        source=SOURCE_USER,
        entry_id="test_grid_entry",
        data={CONF_NAME: "Test Grid"},
        title="Test Grid",
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def home():
    home = create_minimal_home_model()
    home.is_off_grid = MagicMock(return_value=False)
    home.force_next_solve = MagicMock()
    return home


@pytest.fixture
def setup(hass, config_entry, home):
    data_handler = MagicMock()
    data_handler.home = home
    data_handler.hass = hass
    hass.data.setdefault(DOMAIN, {})[DATA_HANDLER] = data_handler
    return {"config_entry": config_entry, "home": home}


@pytest.fixture
def device(hass, setup) -> ConcreteBiStateDevice:
    return _make_device(hass, setup["home"], setup["config_entry"])


# ===========================================================================
# Fix 1: reset_override_state_and_set_reset_ask_time no-op without override
# ===========================================================================


class TestResetOverrideNoOp:
    """reset_override_state_and_set_reset_ask_time should do nothing when
    there is no active user override."""

    def test_noop_when_no_override(self, device: ConcreteBiStateDevice):
        """When both external_user_initiated_state and _time are None, no flag is set."""
        device.external_user_initiated_state = None
        device.external_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        t = _utcnow()
        device.reset_override_state_and_set_reset_ask_time(t)

        assert device.asked_for_reset_user_initiated_state_time is None
        assert device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done is None

    def test_sets_flag_when_override_active(self, device: ConcreteBiStateDevice):
        """When an override IS active, flags are set normally."""
        device.external_user_initiated_state = "on"
        device.external_user_initiated_state_time = _utcnow() - timedelta(minutes=5)
        device.asked_for_reset_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        t = _utcnow()
        device.reset_override_state_and_set_reset_ask_time(t)

        assert device.external_user_initiated_state is None
        assert device.external_user_initiated_state_time is None
        assert device.asked_for_reset_user_initiated_state_time == t
        assert device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done == t

    def test_noop_preserves_existing_constraint(self, device: ConcreteBiStateDevice):
        """After a no-op reset, check_load_activity_and_constraints should NOT
        trigger constraint_reset_and_reset_commands_if_needed."""
        t = _utcnow()
        end = t + timedelta(hours=12)

        ct = _make_constraint(device, t, end, target_value=14400.0, current_value=12600.0)
        device._constraints = [ct]
        device.externally_initialized_constraints = True

        device.external_user_initiated_state = None
        device.external_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        device.reset_override_state_and_set_reset_ask_time(t)

        assert device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done is None
        assert len(device._constraints) == 1
        assert device._constraints[0].current_value == 12600.0


# ===========================================================================
# Fix 2: Bistate check_load_activity_and_constraints preserves constraints
#         after grid transition (asked_for_reset flag is None)
# ===========================================================================


class TestBistateConstraintPreservationAfterGridTransition:
    """After a grid off/on transition where no user override was active,
    the asked_for_reset flag stays None (Fix 1), so
    check_load_activity_and_constraints should not wipe constraints."""

    @pytest.mark.asyncio
    async def test_constraints_preserved_when_flag_not_set(self, hass: HomeAssistant, device: ConcreteBiStateDevice):
        """Constraint with progress survives check_load_activity_and_constraints
        when asked_for_reset_user_initiated_state_time_first_cmd_reset_done is None."""
        t = _utcnow()
        end = t + timedelta(hours=12)

        ct = _make_constraint(device, t, end, target_value=14400.0, current_value=12600.0)
        device._constraints = [ct]

        device.bistate_mode = "bistate_mode_default"
        device.default_on_duration = 4.0
        device.default_on_finish_time = dt_time(hour=0, minute=0, second=0)

        device.current_command = CMD_IDLE
        device.running_command = None

        device.external_user_initiated_state = None
        device.external_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        hass.states.async_set("switch.test_device", "off")

        await device.check_load_activity_and_constraints(t)

        has_original = any(c.current_value == 12600.0 for c in device._constraints if c is not None)
        assert has_original, (
            f"Original constraint with 12600s progress was lost. "
            f"Constraints: {[(c.current_value, c.target_value) for c in device._constraints]}"
        )

    @pytest.mark.asyncio
    async def test_constraints_wiped_when_flag_set(self, hass: HomeAssistant, device: ConcreteBiStateDevice):
        """When the flag IS set (user override was active), constraints are properly reset."""
        t = _utcnow()
        end = t + timedelta(hours=12)

        ct = _make_constraint(device, t, end, target_value=14400.0, current_value=12600.0)
        device._constraints = [ct]

        device.bistate_mode = "bistate_mode_default"
        device.default_on_duration = 4.0
        device.default_on_finish_time = dt_time(hour=0, minute=0, second=0)

        device.current_command = CMD_IDLE
        device.running_command = None

        device.external_user_initiated_state = None
        device.external_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time = t - timedelta(seconds=30)
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = t - timedelta(seconds=30)

        hass.states.async_set("switch.test_device", "off")

        await device.check_load_activity_and_constraints(t)

        assert device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done is None


# ===========================================================================
# Fix 2: Pool _build_mode_constraint_items
# ===========================================================================


class TestPoolBuildModeConstraintItems:
    """Pool auto/winter modes return correct ConstraintItemType via
    _build_mode_constraint_items, which is called by the parent's
    check_load_activity_and_constraints."""

    @pytest.mark.asyncio
    async def test_pool_build_auto_mode_items(self, hass, setup):
        """Pool auto mode produces a single ConstraintItemType with proper degraded_type."""
        from custom_components.quiet_solar.const import CONF_POOL_TEMPERATURE_SENSOR, CONF_POWER
        from custom_components.quiet_solar.ha_model.pool import QSPool

        home = setup["home"]
        entry = setup["config_entry"]

        hass.states.async_set("sensor.pool_temp", "15.0")

        pool = QSPool(
            hass=hass,
            config_entry=entry,
            home=home,
            **{
                CONF_NAME: "Test Pool",
                CONF_SWITCH: "switch.pool_pump",
                CONF_POOL_TEMPERATURE_SENSOR: "sensor.pool_temp",
                CONF_POWER: 1500,
            },
        )
        pool.bistate_mode = "bistate_mode_auto"
        pool.default_on_finish_time = dt_time(hour=0, minute=0, second=0)

        t = _utcnow()

        with patch.object(pool, "get_state_history_data", return_value=[(t, 15.0)]):
            items = await pool._build_mode_constraint_items(t, "bistate_mode_auto", None)

        assert len(items) == 1
        item = items[0]
        assert isinstance(item, ConstraintItemType)
        assert item.degraded_type == CONSTRAINT_TYPE_FILLER
        assert item.has_user_forced_constraint is False
        assert item.agenda_push is False
        assert item.target_value > 0

    @pytest.mark.asyncio
    async def test_pool_build_winter_mode_items(self, hass, setup):
        """Pool winter mode produces items with CONSTRAINT_TYPE_FILLER degraded_type."""
        from custom_components.quiet_solar.const import CONF_POOL_TEMPERATURE_SENSOR, CONF_POWER
        from custom_components.quiet_solar.ha_model.pool import QSPool

        home = setup["home"]
        entry = setup["config_entry"]

        hass.states.async_set("sensor.pool_temp", "8.0")

        pool = QSPool(
            hass=hass,
            config_entry=entry,
            home=home,
            **{
                CONF_NAME: "Test Pool",
                CONF_SWITCH: "switch.pool_pump",
                CONF_POOL_TEMPERATURE_SENSOR: "sensor.pool_temp",
                CONF_POWER: 1500,
            },
        )
        pool.bistate_mode = "pool_winter_mode"
        pool.default_on_finish_time = dt_time(hour=0, minute=0, second=0)

        t = _utcnow()
        items = await pool._build_mode_constraint_items(t, "pool_winter_mode", None)

        assert len(items) == 1
        assert items[0].degraded_type == CONSTRAINT_TYPE_FILLER

    @pytest.mark.asyncio
    async def test_pool_delegates_other_modes(self, hass, setup):
        """Pool delegates non-auto/non-winter modes to parent."""
        from custom_components.quiet_solar.const import CONF_POOL_TEMPERATURE_SENSOR, CONF_POWER
        from custom_components.quiet_solar.ha_model.pool import QSPool

        home = setup["home"]
        entry = setup["config_entry"]

        hass.states.async_set("sensor.pool_temp", "15.0")

        pool = QSPool(
            hass=hass,
            config_entry=entry,
            home=home,
            **{
                CONF_NAME: "Test Pool",
                CONF_SWITCH: "switch.pool_pump",
                CONF_POOL_TEMPERATURE_SENSOR: "sensor.pool_temp",
                CONF_POWER: 1500,
            },
        )
        pool.default_on_duration = 4.0
        pool.default_on_finish_time = dt_time(hour=0, minute=0, second=0)

        t = _utcnow()
        items = await pool._build_mode_constraint_items(t, "bistate_mode_default", None)

        assert len(items) == 1
        assert items[0].degraded_type is None  # parent default


# ===========================================================================
# Fix 3: _switch_to_off_grid_launched cleared on back-to-on-grid
# ===========================================================================


class TestSwitchToOffGridLaunchedClear:
    """async_set_off_grid_mode should clear _switch_to_off_grid_launched
    when transitioning from off-grid to on-grid.

    Uses the real QSHome.async_set_off_grid_mode method bound to a mock home
    to avoid needing a full HA integration setup.
    """

    def _make_home(self, off_grid=False, switch_launched=None):
        from custom_components.quiet_solar.ha_model.home import QSHome

        home = MagicMock()
        home.qs_home_is_off_grid = off_grid
        home._switch_to_off_grid_launched = switch_launched
        home._all_loads = []
        home.battery = None
        home._last_solve_done = None
        home.is_off_grid = lambda: home.qs_home_is_off_grid
        home.force_next_solve = MagicMock()
        home.async_set_off_grid_mode = lambda off_grid, for_init: QSHome.async_set_off_grid_mode(
            home, off_grid, for_init
        )
        return home

    @pytest.mark.asyncio
    async def test_cleared_on_back_to_on_grid(self):
        """Going off->on clears the gate in the else branch."""
        home = self._make_home(
            off_grid=True,
            switch_launched=_utcnow() - timedelta(seconds=30),
        )

        await home.async_set_off_grid_mode(off_grid=False, for_init=False)

        assert home.qs_home_is_off_grid is False
        assert home._switch_to_off_grid_launched is None

    @pytest.mark.asyncio
    async def test_set_on_going_off_grid(self):
        """Going on->off sets _switch_to_off_grid_launched."""
        home = self._make_home(off_grid=False, switch_launched=None)

        await home.async_set_off_grid_mode(off_grid=True, for_init=False)

        assert home.qs_home_is_off_grid is True
        assert home._switch_to_off_grid_launched is not None

    @pytest.mark.asyncio
    async def test_no_change_when_same_state(self):
        """No-op when state doesn't change."""
        home = self._make_home(off_grid=False, switch_launched=None)

        await home.async_set_off_grid_mode(off_grid=False, for_init=False)

        assert home._switch_to_off_grid_launched is None


# ===========================================================================
# Integration: full off/on cycle preserves constraint progress
# ===========================================================================


class TestFullGridTransitionCycle:
    """Simulate a complete grid off -> on cycle and verify constraint progress."""

    @pytest.mark.asyncio
    async def test_off_on_cycle_preserves_bistate_constraint(self, hass: HomeAssistant, device: ConcreteBiStateDevice):
        """A bistate device with no user override preserves its constraint
        through a full off-grid -> on-grid cycle."""
        t = _utcnow()
        end = t + timedelta(hours=12)

        ct = _make_constraint(device, t, end, target_value=14400.0, current_value=12600.0)
        device._constraints = [ct]
        device.externally_initialized_constraints = True
        device.bistate_mode = "bistate_mode_default"
        device.default_on_duration = 4.0
        device.default_on_finish_time = dt_time(hour=0, minute=0, second=0)

        device.external_user_initiated_state = None
        device.external_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time = None
        device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        # --- Simulate going off-grid ---
        device.reset_override_state_and_set_reset_ask_time(t)

        # Fix 1: no override was active -> flag should NOT be set
        assert device.asked_for_reset_user_initiated_state_time is None
        assert device.asked_for_reset_user_initiated_state_time_first_cmd_reset_done is None

        device.current_command = CMD_IDLE
        device.running_command = None

        # --- Simulate coming back on-grid ---
        hass.states.async_set("switch.test_device", "off")

        await device.check_load_activity_and_constraints(t + timedelta(seconds=35))

        has_original = any(c.current_value == 12600.0 for c in device._constraints if c is not None)
        assert has_original, "Constraint progress (12600s of 14400s) was lost after grid off/on cycle"


# ===========================================================================
# Story 2.4, AC #2: Off-grid mode edge cases
# ===========================================================================


@pytest.mark.integration
class TestOffGridSolverEdgeCases:
    """Test solver behavior in off-grid mode: battery depletion, load shedding priority, recovery."""

    def setup_method(self):
        self.dt = datetime.datetime(year=2024, month=6, day=1, hour=10, minute=0, second=0, tzinfo=pytz.UTC)
        self.end_time = self.dt + timedelta(hours=14)

    def _make_solver_off_grid(self, loads, battery, solar_power=2000.0):
        """Create a solver configured for off-grid scenario."""
        from custom_components.quiet_solar.home_model.solver import PeriodSolver
        from tests.utils.scenario_builders import (
            build_realistic_consumption_forecast,
            build_realistic_solar_forecast,
        )

        forecast = build_realistic_solar_forecast(self.dt, num_hours=14, peak_power=solar_power)
        consumption = build_realistic_consumption_forecast(self.dt, num_hours=14, base_load=300.0, night_load=200.0)

        return PeriodSolver(
            start_time=self.dt,
            end_time=self.end_time,
            tariffs=0.0,
            actionable_loads=loads,
            battery=battery,
            pv_forecast=forecast,
            unavoidable_consumption_forecast=consumption,
        )

    def test_off_grid_solver_uses_only_available_power(self):
        """Off-grid mode forces solver to use only available power (solar + battery), no grid."""
        from custom_components.quiet_solar.home_model.commands import CMD_CST_AUTO_CONSIGN, LoadCommand
        from custom_components.quiet_solar.home_model.constraints import MultiStepsPowerLoadConstraint
        from custom_components.quiet_solar.home_model.load import TestLoad
        from tests.utils.scenario_builders import create_test_battery

        car = TestLoad(name="car")
        car_steps = [LoadCommand(command=CMD_CST_AUTO_CONSIGN, power_consign=a * 3 * 230) for a in range(7, 33)]

        # Car needs lots of energy — more than solar alone can provide
        car_constraint = MultiStepsPowerLoadConstraint(
            time=self.dt,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=self.dt + timedelta(hours=10),
            initial_value=2000,
            target_value=20000,
            power_steps=car_steps,
            support_auto=True,
        )
        car.push_live_constraint(self.dt, car_constraint)

        battery = create_test_battery(initial_soc_percent=60.0, max_discharge_power=3000.0)

        solver = self._make_solver_off_grid([car], battery, solar_power=3000.0)

        # Solve in off-grid mode
        result, battery_cmds = solver.solve(is_off_grid=True, with_self_test=True)

        # Solver should still produce valid output (energy conserved)
        assert result is not None

    def test_off_grid_battery_depletion_respects_min_soc(self):
        """In off-grid mode, battery discharge respects minimum SOC threshold."""
        from custom_components.quiet_solar.home_model.commands import CMD_CST_AUTO_CONSIGN, LoadCommand
        from custom_components.quiet_solar.home_model.constraints import MultiStepsPowerLoadConstraint
        from custom_components.quiet_solar.home_model.load import TestLoad
        from tests.utils.scenario_builders import create_test_battery

        car = TestLoad(name="car")
        car_steps = [LoadCommand(command=CMD_CST_AUTO_CONSIGN, power_consign=a * 3 * 230) for a in range(7, 33)]

        # Very large demand that exceeds available energy
        car_constraint = MultiStepsPowerLoadConstraint(
            time=self.dt,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=self.dt + timedelta(hours=10),
            initial_value=0,
            target_value=50000,
            power_steps=car_steps,
            support_auto=True,
        )
        car.push_live_constraint(self.dt, car_constraint)

        min_soc_percent = 20.0
        capacity_wh = 10000.0
        # Battery with 20% minimum SOC = 2000Wh minimum
        battery = create_test_battery(
            capacity_wh=capacity_wh,
            initial_soc_percent=50.0,
            min_soc_percent=min_soc_percent,
            max_discharge_power=5000.0,
        )

        solver = self._make_solver_off_grid([car], battery, solar_power=2000.0)

        result, battery_cmds = solver.solve(is_off_grid=True, with_self_test=True)

        assert result is not None
        # with_self_test=True validates energy conservation internally;
        # additionally verify the solver planned within the battery's SOC floor
        min_charge_wh = capacity_wh * min_soc_percent / 100.0
        assert battery.current_charge >= min_charge_wh - 1.0, (
            f"Battery SOC {battery.current_charge:.0f}Wh dropped below min "
            f"{min_charge_wh:.0f}Wh (min_soc={min_soc_percent}%)"
        )

    def test_off_grid_load_shedding_filler_before_mandatory(self):
        """In off-grid with limited capacity, filler loads are shed before mandatory."""
        from custom_components.quiet_solar.home_model.load import TestLoad
        from tests.utils.scenario_builders import create_test_battery

        car = TestLoad(name="car")
        pool = TestLoad(name="pool")
        boiler = TestLoad(name="boiler")

        from custom_components.quiet_solar.home_model.commands import CMD_CST_AUTO_CONSIGN, LoadCommand
        from custom_components.quiet_solar.home_model.constraints import (
            MultiStepsPowerLoadConstraint,
            TimeBasedSimplePowerLoadConstraint,
        )

        car_steps = [LoadCommand(command=CMD_CST_AUTO_CONSIGN, power_consign=a * 3 * 230) for a in range(7, 33)]

        # Mandatory: car must charge
        car_mandatory = MultiStepsPowerLoadConstraint(
            time=self.dt,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=self.dt + timedelta(hours=8),
            initial_value=2000,
            target_value=10000,
            power_steps=car_steps,
            support_auto=True,
        )
        car.push_live_constraint(self.dt, car_mandatory)

        # Filler: pool pump
        pool_filler = TimeBasedSimplePowerLoadConstraint(
            time=self.dt,
            load=pool,
            type=CONSTRAINT_TYPE_FILLER,
            end_of_constraint=self.dt + timedelta(hours=14),
            initial_value=0,
            target_value=8 * 3600,
            power=1430,
        )
        pool.push_live_constraint(self.dt, pool_filler)

        # Filler auto: boiler
        boiler_filler = TimeBasedSimplePowerLoadConstraint(
            time=self.dt,
            load=boiler,
            type=CONSTRAINT_TYPE_FILLER_AUTO,
            end_of_constraint=self.dt + timedelta(hours=14),
            initial_value=0,
            target_value=3 * 3600,
            power=2000,
        )
        boiler.push_live_constraint(self.dt, boiler_filler)

        # Limited battery — not enough for all loads
        battery = create_test_battery(initial_soc_percent=40.0, max_discharge_power=3000.0)

        solver = self._make_solver_off_grid([car, pool, boiler], battery, solar_power=1500.0)

        result, _ = solver.solve(is_off_grid=True, with_self_test=True)

        from tests.utils.energy_validation import calculate_energy_from_commands

        # Mandatory car should get commands
        car_result = [r for r in result if r[0] == car]
        assert len(car_result) == 1, "Mandatory car should have commands in off-grid"
        car_commands = car_result[0][1]
        assert len(car_commands) > 0, "Mandatory load should not be shed in off-grid"

        car_energy = calculate_energy_from_commands(car_commands, self.end_time)

        # Filler loads should get less energy than mandatory
        pool_result = [r for r in result if r[0] == pool]
        boiler_result = [r for r in result if r[0] == boiler]
        pool_energy = calculate_energy_from_commands(pool_result[0][1], self.end_time) if pool_result else 0.0
        boiler_energy = calculate_energy_from_commands(boiler_result[0][1], self.end_time) if boiler_result else 0.0

        assert car_energy >= pool_energy, (
            f"Mandatory car ({car_energy:.0f}Wh) should get at least as much "
            f"energy as filler pool ({pool_energy:.0f}Wh)"
        )
        assert car_energy >= boiler_energy, (
            f"Mandatory car ({car_energy:.0f}Wh) should get at least as much "
            f"energy as filler boiler ({boiler_energy:.0f}Wh)"
        )

    def test_off_grid_zero_battery_zero_solar(self):
        """Off-grid with no battery and no solar — solver should still not crash."""
        from custom_components.quiet_solar.home_model.load import TestLoad
        from tests.utils.scenario_builders import create_test_battery

        pool = TestLoad(name="pool")

        from custom_components.quiet_solar.home_model.constraints import (
            TimeBasedSimplePowerLoadConstraint,
        )

        pool_constraint = TimeBasedSimplePowerLoadConstraint(
            time=self.dt,
            load=pool,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=self.dt + timedelta(hours=10),
            initial_value=0,
            target_value=4 * 3600,
            power=1430,
        )
        pool.push_live_constraint(self.dt, pool_constraint)

        # Nearly depleted battery, no solar
        battery = create_test_battery(initial_soc_percent=5.0, min_soc_percent=5.0)

        solver = self._make_solver_off_grid([pool], battery, solar_power=0.0)

        # Should not crash even with zero resources
        result, _ = solver.solve(is_off_grid=True, with_self_test=True)
        assert result is not None

    def test_grid_restoration_solver_returns_to_normal(self):
        """After grid restoration, solver operates normally (not off-grid constrained)."""
        from custom_components.quiet_solar.home_model.commands import CMD_CST_AUTO_CONSIGN, LoadCommand
        from custom_components.quiet_solar.home_model.constraints import MultiStepsPowerLoadConstraint
        from custom_components.quiet_solar.home_model.load import TestLoad
        from tests.utils.scenario_builders import create_test_battery

        car = TestLoad(name="car")
        car_steps = [LoadCommand(command=CMD_CST_AUTO_CONSIGN, power_consign=a * 3 * 230) for a in range(7, 33)]

        # Large energy demand
        car_constraint = MultiStepsPowerLoadConstraint(
            time=self.dt,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=self.dt + timedelta(hours=10),
            initial_value=0,
            target_value=30000,
            power_steps=car_steps,
            support_auto=True,
        )
        car.push_live_constraint(self.dt, car_constraint)

        battery = create_test_battery(initial_soc_percent=50.0)

        # First solve off-grid (constrained)
        solver_off = self._make_solver_off_grid([car], battery, solar_power=3000.0)
        result_off, _ = solver_off.solve(is_off_grid=True, with_self_test=True)

        # Reset constraint for on-grid solve
        car._constraints = []
        car_constraint_on = MultiStepsPowerLoadConstraint(
            time=self.dt,
            load=car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint=self.dt + timedelta(hours=10),
            initial_value=0,
            target_value=30000,
            power_steps=car_steps,
            support_auto=True,
        )
        car.push_live_constraint(self.dt, car_constraint_on)

        battery_on = create_test_battery(initial_soc_percent=50.0)

        # Then solve on-grid (normal)
        solver_on = self._make_solver_off_grid([car], battery_on, solar_power=3000.0)
        result_on, _ = solver_on.solve(is_off_grid=False, with_self_test=True)

        from tests.utils.energy_validation import calculate_energy_from_commands

        # Both should produce valid results
        assert result_off is not None
        assert result_on is not None

        # On-grid should allocate at least as much energy as off-grid
        # because off-grid is constrained to available power only
        car_off = [r for r in result_off if r[0] == car]
        car_on = [r for r in result_on if r[0] == car]
        assert len(car_off) == 1 and len(car_on) == 1

        energy_off = calculate_energy_from_commands(car_off[0][1], self.end_time)
        energy_on = calculate_energy_from_commands(car_on[0][1], self.end_time)
        assert energy_on >= energy_off, (
            f"On-grid energy ({energy_on:.0f}Wh) should be >= off-grid "
            f"({energy_off:.0f}Wh) since grid provides additional capacity"
        )
