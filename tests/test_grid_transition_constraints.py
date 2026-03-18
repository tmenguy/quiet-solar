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
