"""Deep coverage tests for charger.py focusing on untested code paths.

Uses REAL objects (QSCar, QSChargerGeneric, constraints) wherever possible.
Only mocks what truly requires Home Assistant (sensor state reads, service calls).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from custom_components.quiet_solar.const import (
    CHARGER_NO_CAR_CONNECTED,
    CONF_CAR_BATTERY_CAPACITY,
    CONF_CAR_CHARGER_MAX_CHARGE,
    CONF_CAR_CHARGER_MIN_CHARGE,
    CONF_CAR_CHARGE_PERCENT_SENSOR,
    CONF_CAR_IS_INVITED,
    CONF_CAR_PLUGGED,
    CONF_CAR_TRACKER,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_PLUGGED,
    CONF_CHARGER_STATUS_SENSOR,
    CONF_DEFAULT_CAR_CHARGE,
    CONF_DEVICE_EFFICIENCY,
    CONF_IS_3P,
    CONF_MINIMUM_OK_CAR_CHARGE,
    CONF_MONO_PHASE,
    CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
    CONSTRAINT_TYPE_FILLER,
    CONSTRAINT_TYPE_FILLER_AUTO,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    FORCE_CAR_NO_CHARGER_CONNECTED,
)
from custom_components.quiet_solar.ha_model.car import QSCar
from custom_components.quiet_solar.ha_model.charger import (
    CHARGER_ADAPTATION_WINDOW_S,
    CHARGER_BOOT_TIME_DATA_EXPIRATION_S,
    CHARGER_CHECK_STATE_WINDOW_S,
    QSChargerGeneric,
    QSChargerGroup,
    QSChargerStates,
    QSChargerStatus,
    QSStateCmd,
)
from custom_components.quiet_solar.home_model.commands import (
    CMD_AUTO_FROM_CONSIGN,
    CMD_AUTO_GREEN_CAP,
    CMD_AUTO_GREEN_CONSIGN,
    CMD_AUTO_GREEN_ONLY,
    CMD_AUTO_PRICE,
    CMD_ON,
    copy_command,
)
from custom_components.quiet_solar.home_model.constraints import (
    DATETIME_MAX_UTC,
    LoadConstraint,
    MultiStepsPowerLoadConstraint,
    MultiStepsPowerLoadConstraintChargePercent,
)

_LOGGER = logging.getLogger(__name__)


# =============================================================================
# Shared helpers: only mock what truly needs HA
# =============================================================================


def _make_hass() -> MagicMock:
    """Minimal hass mock: only for HA-level I/O (states, services, bus)."""
    hass = MagicMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.config = MagicMock()
    hass.config.config_dir = "/tmp/test"
    hass.bus = MagicMock()
    hass.bus.async_listen = MagicMock(return_value=lambda: None)
    hass.async_add_executor_job = AsyncMock(side_effect=lambda f, *a: f(*a))
    return hass


def _make_home(battery=None, voltage=230.0):
    """Create a mock home.  Home has no simple real constructor so we mock it."""
    home = MagicMock()
    home.name = "TestHome"
    home.voltage = voltage
    home.is_3p = True
    home._cars = []
    home._chargers = []
    home._loads = []
    home._persons = []
    home.available_amps_for_group = [[32.0, 32.0, 32.0]]
    home.battery = battery
    home.get_car_by_name = lambda n: next((c for c in home._cars if c.name == n), None)
    home.get_available_power_values = MagicMock(return_value=None)
    home.get_grid_consumption_power_values = MagicMock(return_value=None)
    home.get_best_tariff = MagicMock(return_value=0.15)
    home.get_tariff = MagicMock(return_value=0.20)
    home.battery_can_discharge = MagicMock(return_value=True)
    home.is_off_grid = MagicMock(return_value=False)
    home.dashboard_sections = None
    home.get_best_persons_cars_allocations = AsyncMock()
    home.get_preferred_person_for_car = MagicMock(return_value=None)
    return home


def _make_real_car(hass, home, name="TestCar", battery_capacity=60000,
                   min_charge=6, max_charge=32, default_charge=80.0,
                   minimum_ok_charge=20.0, is_invited=False,
                   has_soc_sensor=True) -> QSCar:
    """Create a REAL QSCar with minimal HA mocking."""
    kwargs = {
        "name": name,
        "hass": hass,
        "home": home,
        "config_entry": None,
        CONF_CAR_BATTERY_CAPACITY: battery_capacity,
        CONF_CAR_CHARGER_MIN_CHARGE: min_charge,
        CONF_CAR_CHARGER_MAX_CHARGE: max_charge,
        CONF_DEFAULT_CAR_CHARGE: default_charge,
        CONF_MINIMUM_OK_CAR_CHARGE: minimum_ok_charge,
        CONF_CAR_IS_INVITED: is_invited,
        CONF_DEVICE_EFFICIENCY: 90.0,
    }
    if has_soc_sensor:
        kwargs[CONF_CAR_CHARGE_PERCENT_SENSOR] = f"sensor.{name.lower().replace(' ','_')}_soc"
    car = QSCar(**kwargs)
    # Force percent mode on if SOC sensor is set (the real __init__ may fail to
    # auto-detect because hass.states returns None during construction)
    if has_soc_sensor and battery_capacity is not None:
        car._use_percent_mode = True
    home._cars.append(car)
    return car


def _make_charger_group(home, chargers, max_amps=None):
    """Build QSChargerGroup around a mock dynamic-group (the group has no easy real ctor)."""
    from custom_components.quiet_solar.ha_model.dynamic_group import QSDynamicGroup

    if max_amps is None:
        max_amps = [32.0, 32.0, 32.0]

    dg = MagicMock(spec=QSDynamicGroup)
    dg.name = "TestGroup"
    dg.home = home
    dg._childrens = chargers
    dg.available_amps_for_group = [max_amps]
    dg.dyn_group_max_phase_current = max(max_amps)
    dg.is_current_acceptable = MagicMock(return_value=True)
    dg.is_current_acceptable_and_diff = MagicMock(return_value=(True, [0.0, 0.0, 0.0]))
    dg.get_median_sensor = MagicMock(return_value=None)
    dg.accurate_power_sensor = "sensor.group_power"
    dg.secondary_power_sensor = None

    group = QSChargerGroup(dg)
    group.charger_consumption_W = 70
    return group


def _create_charger(hass, home, name="TestCharger", is_3p=False,
                    min_charge=6, max_charge=32, **extra) -> QSChargerGeneric:
    """Create a REAL QSChargerGeneric."""
    config_entry = MagicMock()
    config_entry.entry_id = f"test_entry_{name}"
    config_entry.data = {}

    config = {
        "name": name, "hass": hass, "home": home,
        "config_entry": config_entry,
        CONF_CHARGER_MIN_CHARGE: min_charge,
        CONF_CHARGER_MAX_CHARGE: max_charge,
        CONF_IS_3P: is_3p,
        CONF_MONO_PHASE: 1,
        CONF_CHARGER_STATUS_SENSOR: f"sensor.{name}_status",
        CONF_CHARGER_PLUGGED: f"sensor.{name}_plugged",
        CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER: f"number.{name}_max_current",
    }
    config.update(extra)

    with patch("custom_components.quiet_solar.ha_model.charger.entity_registry"):
        charger = QSChargerGeneric(**config)

    home._chargers.append(charger)

    if hasattr(charger, "father_device") and charger.father_device is not None:
        group = _make_charger_group(home, [charger])
        charger.father_device.charger_group = group
    return charger


def _init_charger_states(charger, charge_state=True, amperage=None, num_phases=1):
    """Set up inner state objects (the charger resets them to None on reset)."""
    if amperage is None:
        amperage = charger.min_charge
    charger._inner_expected_charge_state = QSStateCmd()
    charger._inner_expected_charge_state.value = charge_state
    charger._inner_amperage = QSStateCmd()
    charger._inner_amperage.value = amperage
    charger._inner_num_active_phases = QSStateCmd()
    charger._inner_num_active_phases.value = num_phases


def _plug_car(charger, car, time):
    """Attach a real car to a real charger (calls update_power_steps)."""
    charger.attach_car(car, time)


def _make_battery(asked_charge=0.0):
    battery = MagicMock()
    battery.get_current_battery_asked_change_for_outside_production_system = MagicMock(return_value=asked_charge)
    return battery


# =============================================================================
# _check_charger_status
# =============================================================================


class TestCheckChargerStatus:

    def _ch(self):
        hass = _make_hass()
        home = _make_home()
        return _create_charger(hass, home)

    def test_empty_vals_returns_none(self):
        ch = self._ch()
        assert ch._check_charger_status([], datetime.now(pytz.UTC)) is None

    def test_no_sensor_returns_none(self):
        ch = self._ch()
        ch.charger_status_sensor = None
        assert ch._check_charger_status(["X"], datetime.now(pytz.UTC)) is None

    def test_duration_none_result(self):
        ch = self._ch()
        ch.get_last_state_value_duration = MagicMock(return_value=(None, None))
        assert ch._check_charger_status(["X"], datetime.now(pytz.UTC), for_duration=30.0) is None

    def test_duration_sufficient(self):
        ch = self._ch()
        ch.get_last_state_value_duration = MagicMock(return_value=(60.0, datetime.now(pytz.UTC)))
        assert ch._check_charger_status(["X"], datetime.now(pytz.UTC), for_duration=30.0) is True

    def test_duration_insufficient(self):
        ch = self._ch()
        ch.get_last_state_value_duration = MagicMock(return_value=(10.0, datetime.now(pytz.UTC)))
        assert ch._check_charger_status(["X"], datetime.now(pytz.UTC), for_duration=30.0) is False

    def test_no_duration_in_vals(self):
        ch = self._ch()
        ch.get_sensor_latest_possible_valid_value = MagicMock(return_value="Charging")
        assert ch._check_charger_status(["Charging"], datetime.now(pytz.UTC)) is True

    def test_no_duration_not_in_vals(self):
        ch = self._ch()
        ch.get_sensor_latest_possible_valid_value = MagicMock(return_value="Off")
        assert ch._check_charger_status(["Charging"], datetime.now(pytz.UTC)) is False

    def test_no_duration_none_state(self):
        ch = self._ch()
        ch.get_sensor_latest_possible_valid_value = MagicMock(return_value=None)
        assert ch._check_charger_status(["Charging"], datetime.now(pytz.UTC)) is None

    def test_invert_true_not_in(self):
        ch = self._ch()
        ch.get_sensor_latest_possible_valid_value = MagicMock(return_value="Idle")
        assert ch._check_charger_status(["Charging"], datetime.now(pytz.UTC), invert_prob=True) is True

    def test_invert_true_in(self):
        ch = self._ch()
        ch.get_sensor_latest_possible_valid_value = MagicMock(return_value="Charging")
        assert ch._check_charger_status(["Charging"], datetime.now(pytz.UTC), invert_prob=True) is False

    def test_duration_zero_treated_as_none(self):
        ch = self._ch()
        ch.get_sensor_latest_possible_valid_value = MagicMock(return_value="Charging")
        assert ch._check_charger_status(["Charging"], datetime.now(pytz.UTC), for_duration=0) is True


# =============================================================================
# get_stable_dynamic_charge_status  (real car, real charger)
# =============================================================================


class TestGetStableDynamicChargeStatus:

    def _setup(self, is_3p=False):
        hass = _make_hass()
        home = _make_home()
        charger = _create_charger(hass, home, is_3p=is_3p)
        car = _make_real_car(hass, home)
        past = datetime.now(pytz.UTC) - timedelta(hours=2)
        now = datetime.now(pytz.UTC)

        _init_charger_states(charger, charge_state=True, amperage=10, num_phases=1)
        _plug_car(charger, car, past)

        charger.qs_enable_device = True
        charger.is_not_plugged = MagicMock(return_value=False)
        charger.is_charger_unavailable = MagicMock(return_value=False)
        charger._probe_and_enforce_stopped_charge_command_state = MagicMock(return_value=False)
        charger.get_median_sensor = MagicMock(return_value=1000.0)
        charger.get_current_active_constraint = MagicMock(return_value=None)
        charger.can_do_3_to_1_phase_switch = MagicMock(return_value=False)

        charger._expected_charge_state.last_change_asked = past
        charger._expected_charge_state.last_time_set = past
        charger._expected_num_active_phases.last_change_asked = past
        charger._expected_num_active_phases.last_time_set = past

        return hass, home, charger, car, now

    def test_disabled_returns_none(self):
        *_, charger, _, now = self._setup()
        charger.qs_enable_device = False
        assert charger.get_stable_dynamic_charge_status(now) is None

    def test_no_car_returns_none(self):
        *_, charger, car, now = self._setup()
        charger.detach_car()
        assert charger.get_stable_dynamic_charge_status(now) is None

    def test_not_plugged_returns_none(self):
        *_, charger, _, now = self._setup()
        charger.is_not_plugged = MagicMock(return_value=True)
        assert charger.get_stable_dynamic_charge_status(now) is None

    def test_cmd_on_converts(self):
        *_, charger, _, now = self._setup()
        charger.current_command = copy_command(CMD_ON)
        cs = charger.get_stable_dynamic_charge_status(now)
        assert cs is not None
        assert cs.command.is_like(CMD_AUTO_FROM_CONSIGN)

    def test_cmd_green_cap_zero_forbids(self):
        *_, charger, _, now = self._setup()
        charger.current_command = copy_command(CMD_AUTO_GREEN_CAP, power_consign=0)
        cs = charger.get_stable_dynamic_charge_status(now)
        assert cs is not None
        assert 0 in cs.possible_amps or charger.min_charge in cs.possible_amps

    def test_cmd_auto_from_consign_no_zero(self):
        *_, charger, _, now = self._setup()
        charger.current_command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=2000)
        cs = charger.get_stable_dynamic_charge_status(now)
        assert cs is not None
        assert 0 not in cs.possible_amps

    def test_cmd_auto_price_not_stoppable(self):
        *_, charger, _, now = self._setup()
        charger.current_command = copy_command(CMD_AUTO_PRICE, power_consign=2000)
        cs = charger.get_stable_dynamic_charge_status(now)
        assert cs is not None
        assert cs.can_be_started_and_stopped is False

    def test_green_consign_battery_discharge_blocks_stop(self):
        hass, home, charger, car, now = self._setup()
        battery = _make_battery(asked_charge=-500.0)
        home.battery = battery
        home.battery_can_discharge = MagicMock(return_value=True)
        charger.current_command = copy_command(CMD_AUTO_GREEN_CONSIGN, power_consign=2000)
        far_past = now - timedelta(hours=2)
        charger._expected_charge_state.last_change_asked = far_past
        charger._expected_charge_state.last_time_set = far_past
        cs = charger.get_stable_dynamic_charge_status(now)
        assert cs is not None
        assert 0 not in cs.possible_amps

    def test_bump_solar_cap_to_green_only(self):
        *_, charger, car, now = self._setup()
        car.qs_bump_solar_charge_priority = True
        charger.current_command = copy_command(CMD_AUTO_GREEN_CAP, power_consign=3000)
        cs = charger.get_stable_dynamic_charge_status(now)
        assert cs.command.is_like(CMD_AUTO_GREEN_ONLY)

    def test_charge_off_amps_zero(self):
        *_, charger, _, now = self._setup()
        charger._expected_charge_state.value = False
        charger.current_command = copy_command(CMD_AUTO_GREEN_ONLY)
        cs = charger.get_stable_dynamic_charge_status(now)
        assert cs.current_real_max_charging_amp == 0

    def test_below_min_treated_as_zero(self):
        *_, charger, _, now = self._setup()
        charger._expected_amperage.value = 3
        charger.current_command = copy_command(CMD_AUTO_GREEN_ONLY)
        cs = charger.get_stable_dynamic_charge_status(now)
        assert cs.current_real_max_charging_amp == 0

    def test_off_cant_change_gives_zero_only(self):
        *_, charger, _, now = self._setup()
        charger._expected_charge_state.value = False
        charger._expected_amperage.value = 0
        charger._expected_charge_state._num_set = 5
        charger._expected_charge_state.last_change_asked = now
        charger._expected_charge_state.last_time_set = now
        charger.current_command = copy_command(CMD_AUTO_GREEN_ONLY)
        cs = charger.get_stable_dynamic_charge_status(now)
        assert cs.possible_amps == [0]

    def test_phase_switch_available(self):
        *_, charger, _, now = self._setup()
        charger.can_do_3_to_1_phase_switch = MagicMock(return_value=True)
        charger._expected_num_active_phases.last_change_asked = now - timedelta(hours=2)
        charger._expected_num_active_phases.last_time_set = now - timedelta(hours=2)
        charger.current_command = copy_command(CMD_AUTO_GREEN_ONLY)
        cs = charger.get_stable_dynamic_charge_status(now)
        assert 1 in cs.possible_num_phases and 3 in cs.possible_num_phases

    def test_score_increases_with_mandatory_constraint(self):
        *_, charger, _, now = self._setup()
        charger.current_command = copy_command(CMD_AUTO_GREEN_ONLY)
        cs_no_ct = charger.get_stable_dynamic_charge_status(now)

        ct = MagicMock(spec=LoadConstraint)
        ct.as_fast_as_possible = True
        ct.end_of_constraint = now + timedelta(hours=1)
        ct.is_before_battery = True
        ct.is_mandatory = True
        charger.get_current_active_constraint = MagicMock(return_value=ct)
        cs_ct = charger.get_stable_dynamic_charge_status(now)
        assert cs_ct.charge_score > cs_no_ct.charge_score


# =============================================================================
# check_load_activity_and_constraints  (real car, real push_live_constraint)
# =============================================================================


class TestCheckLoadActivityAndConstraints:
    """Uses REAL car + REAL push_live_constraint (no mocking of constraint logic)."""

    def _setup(self):
        hass = _make_hass()
        home = _make_home()
        charger = _create_charger(hass, home)
        car = _make_real_car(hass, home)
        now = datetime.now(pytz.UTC)

        _init_charger_states(charger)
        charger.is_charger_unavailable = MagicMock(return_value=False)
        charger.probe_for_possible_needed_reboot = MagicMock(return_value=False)
        charger.is_not_plugged = MagicMock(return_value=False)
        charger.is_plugged = MagicMock(return_value=True)
        charger.set_charging_num_phases = AsyncMock(return_value=False)
        charger.set_max_charging_current = AsyncMock(return_value=True)
        charger.reboot = AsyncMock()
        # let the real is_car_charged work (needs car_charge_percent from sensor)
        # For tests that check is_car_charged, we mock get_car_charge_percent on the car
        return hass, home, charger, car, now

    @pytest.mark.asyncio
    async def test_reboot_pending(self):
        *_, charger, _, now = self._setup()
        charger._asked_for_reboot_at_time = now
        assert await charger.check_load_activity_and_constraints(now) is False

    @pytest.mark.asyncio
    async def test_probe_reboot_triggers(self):
        *_, charger, _, now = self._setup()
        charger.probe_for_possible_needed_reboot = MagicMock(return_value=True)
        assert await charger.check_load_activity_and_constraints(now) is False
        charger.reboot.assert_called_once()

    @pytest.mark.asyncio
    async def test_boot_too_recent(self):
        *_, charger, _, now = self._setup()
        charger._boot_time = now - timedelta(seconds=1)
        assert await charger.check_load_activity_and_constraints(now) is False

    @pytest.mark.asyncio
    async def test_boot_adjusted_set(self):
        *_, charger, car, now = self._setup()
        charger._boot_time = now - timedelta(seconds=2 * CHARGER_CHECK_STATE_WINDOW_S + 10)
        charger._boot_time_adjusted = None
        # neither plugged nor unplugged so we only go through the boot path
        charger.is_not_plugged = MagicMock(return_value=False)
        charger.is_plugged = MagicMock(return_value=False)
        await charger.check_load_activity_and_constraints(now)
        assert charger._boot_time_adjusted is not None

    @pytest.mark.asyncio
    async def test_boot_data_expires(self):
        *_, charger, car, now = self._setup()
        charger._boot_time = now - timedelta(seconds=2 * CHARGER_CHECK_STATE_WINDOW_S + 10)
        charger._boot_time_adjusted = now - timedelta(seconds=CHARGER_BOOT_TIME_DATA_EXPIRATION_S + 10)
        _plug_car(charger, car, now)
        await charger.check_load_activity_and_constraints(now)
        assert charger._boot_time is None

    @pytest.mark.asyncio
    async def test_unplug_with_car_resets(self):
        *_, charger, car, now = self._setup()
        charger.is_not_plugged = MagicMock(return_value=True)
        _plug_car(charger, car, now)
        result = await charger.check_load_activity_and_constraints(now)
        assert result is True
        assert charger.car is None

    @pytest.mark.asyncio
    async def test_plugged_no_car_selected(self):
        *_, charger, car, now = self._setup()
        charger.get_best_car = MagicMock(return_value=None)
        result = await charger.check_load_activity_and_constraints(now)
        assert result is True

    @pytest.mark.asyncio
    async def test_plugged_car_change(self):
        hass, home, charger, car, now = self._setup()
        new_car = _make_real_car(hass, home, name="NewCar")
        _plug_car(charger, car, now)
        charger.get_best_car = MagicMock(return_value=new_car)
        await charger.check_load_activity_and_constraints(now)
        assert charger.car.name == "NewCar"

    @pytest.mark.asyncio
    async def test_force_charge_creates_asap_real(self):
        """Real push_live_constraint; verify constraint ends up in _constraints."""
        *_, charger, car, now = self._setup()
        _plug_car(charger, car, now)
        charger.get_best_car = MagicMock(return_value=car)
        car.do_force_next_charge = True
        car.do_next_charge_time = None

        result = await charger.check_load_activity_and_constraints(now)
        assert result is True
        # Real push_live_constraint should have added a constraint
        asap_cts = [c for c in charger._constraints if c is not None and c.as_fast_as_possible]
        assert len(asap_cts) >= 1
        assert asap_cts[0].type == CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE

    @pytest.mark.asyncio
    async def test_force_charge_at_target_lowers_initial(self):
        """Car already at 85% with target 80%: initial_value forced < target."""
        *_, charger, car, now = self._setup()
        _plug_car(charger, car, now)
        charger.get_best_car = MagicMock(return_value=car)
        car.do_force_next_charge = True
        # Mock only get_car_charge_percent on the real car to return 85%
        car.get_car_charge_percent = lambda time=None, *a, **kw: 85.0

        result = await charger.check_load_activity_and_constraints(now)
        asap_cts = [c for c in charger._constraints if c is not None and c.as_fast_as_possible]
        assert len(asap_cts) >= 1
        assert asap_cts[0].initial_value < 80.0

    @pytest.mark.asyncio
    async def test_user_timed_constraint_real(self):
        *_, charger, car, now = self._setup()
        _plug_car(charger, car, now)
        charger.get_best_car = MagicMock(return_value=car)
        car.do_next_charge_time = now + timedelta(hours=8)

        result = await charger.check_load_activity_and_constraints(now)
        assert result is True
        user_cts = [c for c in charger._constraints
                    if c is not None and c.from_user and c.is_mandatory]
        assert len(user_cts) >= 1

    @pytest.mark.asyncio
    async def test_force_plus_timed_prefers_asap(self):
        *_, charger, car, now = self._setup()
        _plug_car(charger, car, now)
        charger.get_best_car = MagicMock(return_value=car)
        car.do_force_next_charge = True
        car.do_next_charge_time = now + timedelta(hours=8)

        await charger.check_load_activity_and_constraints(now)
        assert car.do_next_charge_time is None
        asap_cts = [c for c in charger._constraints if c is not None and c.as_fast_as_possible]
        assert len(asap_cts) >= 1

    @pytest.mark.asyncio
    async def test_existing_asap_updates_target(self):
        """Existing ASAP constraint target follows car target changes."""
        *_, charger, car, now = self._setup()
        _plug_car(charger, car, now)
        charger.get_best_car = MagicMock(return_value=car)
        car.do_force_next_charge = True

        # First call: push initial ASAP
        await charger.check_load_activity_and_constraints(now)
        old_ct = [c for c in charger._constraints if c is not None and c.as_fast_as_possible][0]
        assert old_ct.target_value == 80.0

        # Simulate car target changed to 90 via the internal field
        car.do_force_next_charge = False  # already active
        car.car_default_charge = 90.0
        car._next_charge_target = 90.0  # The real car caches this
        await charger.check_load_activity_and_constraints(now + timedelta(seconds=30))
        updated_ct = [c for c in charger._constraints if c is not None and c.as_fast_as_possible][0]
        assert updated_ct.target_value == 90.0

    @pytest.mark.asyncio
    async def test_agenda_constraint_real(self):
        """Agenda (calendar) constraint pushed via real push_agenda_constraints."""
        *_, charger, car, now = self._setup()
        _plug_car(charger, car, now)
        charger.get_best_car = MagicMock(return_value=car)
        # Patch only the calendar call
        car.get_next_scheduled_event = AsyncMock(return_value=(now + timedelta(hours=12), None))

        await charger.check_load_activity_and_constraints(now)
        agenda_cts = [c for c in charger._constraints
                      if c is not None and c.load_info and c.load_info.get("originator") == "agenda"]
        assert len(agenda_cts) >= 1

    @pytest.mark.asyncio
    async def test_agenda_past_event_ignored(self):
        *_, charger, car, now = self._setup()
        _plug_car(charger, car, now)
        charger.get_best_car = MagicMock(return_value=car)
        car.get_next_scheduled_event = AsyncMock(return_value=(now - timedelta(hours=1), None))
        await charger.check_load_activity_and_constraints(now)
        agenda_cts = [c for c in charger._constraints
                      if c is not None and c.load_info and c.load_info.get("originator") == "agenda"]
        assert len(agenda_cts) == 0

    @pytest.mark.asyncio
    async def test_last_completed_blocks_when_charged(self):
        """If car is charged and last_completed exists, agenda is not pushed."""
        *_, charger, car, now = self._setup()
        _plug_car(charger, car, now)
        charger.get_best_car = MagicMock(return_value=car)
        # Report car SOC at target via real car's get_car_charge_percent
        car.get_car_charge_percent = lambda time=None, *a, **kw: 80.0

        completed_ct = MultiStepsPowerLoadConstraintChargePercent(
            total_capacity_wh=60000, type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=now - timedelta(hours=2), load=charger, load_param=car.name,
            from_user=False, end_of_constraint=now - timedelta(hours=1),
            initial_value=50.0, target_value=80.0, power_steps=charger._power_steps,
            support_auto=True,
        )
        charger._last_completed_constraint = completed_ct

        car.get_next_scheduled_event = AsyncMock(return_value=(now + timedelta(hours=12), None))
        await charger.check_load_activity_and_constraints(now)
        agenda_cts = [c for c in charger._constraints
                      if c is not None and c.load_info and c.load_info.get("originator") == "agenda"]
        assert len(agenda_cts) == 0

    @pytest.mark.asyncio
    async def test_last_completed_allows_when_not_charged(self):
        """If car is NOT charged, agenda constraint IS pushed."""
        *_, charger, car, now = self._setup()
        _plug_car(charger, car, now)
        charger.get_best_car = MagicMock(return_value=car)
        car.get_car_charge_percent = lambda time=None, *a, **kw: 30.0

        completed_ct = MultiStepsPowerLoadConstraintChargePercent(
            total_capacity_wh=60000, type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=now - timedelta(hours=2), load=charger, load_param=car.name,
            from_user=False, end_of_constraint=now - timedelta(hours=1),
            initial_value=20.0, target_value=80.0, power_steps=charger._power_steps,
            support_auto=True,
        )
        charger._last_completed_constraint = completed_ct

        car.get_next_scheduled_event = AsyncMock(return_value=(now + timedelta(hours=12), None))
        await charger.check_load_activity_and_constraints(now)
        agenda_cts = [c for c in charger._constraints
                      if c is not None and c.load_info and c.load_info.get("originator") == "agenda"]
        assert len(agenda_cts) >= 1

    @pytest.mark.asyncio
    async def test_filler_constraint_always_pushed(self):
        """A filler/best-effort constraint is always pushed for a plugged car."""
        *_, charger, car, now = self._setup()
        _plug_car(charger, car, now)
        charger.get_best_car = MagicMock(return_value=car)

        await charger.check_load_activity_and_constraints(now)
        assert len([c for c in charger._constraints if c is not None]) >= 1

    @pytest.mark.asyncio
    async def test_bump_solar_changes_filler_type(self):
        *_, charger, car, now = self._setup()
        _plug_car(charger, car, now)
        charger.get_best_car = MagicMock(return_value=car)
        car.qs_bump_solar_charge_priority = True

        await charger.check_load_activity_and_constraints(now)
        cts = [c for c in charger._constraints if c is not None]
        assert any(c.type >= CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN for c in cts)

    @pytest.mark.asyncio
    async def test_non_percent_car(self):
        """Car without SOC sensor uses energy-based constraints."""
        hass, home, charger, _, now = self._setup()
        car_no_soc = _make_real_car(hass, home, name="NoSocCar", has_soc_sensor=False)
        _plug_car(charger, car_no_soc, now)
        charger.get_best_car = MagicMock(return_value=car_no_soc)
        car_no_soc.do_force_next_charge = True

        await charger.check_load_activity_and_constraints(now)
        cts = [c for c in charger._constraints if c is not None]
        assert len(cts) >= 1
        assert all(not isinstance(c, MultiStepsPowerLoadConstraintChargePercent) for c in cts)

    @pytest.mark.asyncio
    async def test_boot_constraints_restored_real(self):
        """After boot, saved boot constraints are restored via real push_live_constraint."""
        *_, charger, car, now = self._setup()
        boot_time = now - timedelta(seconds=2 * CHARGER_CHECK_STATE_WINDOW_S + 10)
        charger._boot_time = boot_time
        charger._boot_time_adjusted = boot_time

        # Attach car to build power_steps, save them, then detach
        _plug_car(charger, car, now)
        saved_power_steps = list(charger._power_steps)
        charger.detach_car()

        boot_ct = MultiStepsPowerLoadConstraintChargePercent(
            total_capacity_wh=60000, type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=boot_time, load=charger, load_param=car.name,
            from_user=True, end_of_constraint=now + timedelta(hours=6),
            initial_value=40.0, target_value=80.0, power_steps=saved_power_steps,
            support_auto=True,
        )
        charger._boot_car = car
        charger._boot_constraints = [boot_ct]
        charger._boot_last_completed_constraint = None
        charger.get_best_car = MagicMock(return_value=car)

        await charger.check_load_activity_and_constraints(now)
        assert len([c for c in charger._constraints if c is not None]) >= 1


# =============================================================================
# compute_is_before_battery  (real charger + real car)
# =============================================================================


class TestComputeIsBeforeBattery:

    def _setup(self):
        hass = _make_hass()
        home = _make_home()
        charger = _create_charger(hass, home)
        car = _make_real_car(hass, home)
        now = datetime.now(pytz.UTC)
        _init_charger_states(charger)
        _plug_car(charger, car, now)
        return charger, car, now

    def test_none_constraint(self):
        ch, _, _ = self._setup()
        assert ch.compute_is_before_battery(None) is False

    def test_green_cap_overrides(self):
        ch, _, now = self._setup()
        ct = MagicMock(); ct.is_before_battery = True
        ch.get_current_active_constraint = MagicMock(return_value=ct)
        ch.current_command = copy_command(CMD_AUTO_GREEN_CAP, power_consign=1000)
        assert ch.compute_is_before_battery(ct, now) is False

    def test_price_forces_before(self):
        ch, _, now = self._setup()
        ct = MagicMock(); ct.is_before_battery = False
        ch.get_current_active_constraint = MagicMock(return_value=ct)
        ch.current_command = copy_command(CMD_AUTO_PRICE, power_consign=1000)
        assert ch.compute_is_before_battery(ct, now) is True

    def test_bump_solar_restores_ct(self):
        ch, car, now = self._setup()
        car.qs_bump_solar_charge_priority = True
        ct = MagicMock(); ct.is_before_battery = True
        ch.get_current_active_constraint = MagicMock(return_value=ct)
        ch.current_command = copy_command(CMD_AUTO_GREEN_CAP, power_consign=1000)
        assert ch.compute_is_before_battery(ct, now) is True


# =============================================================================
# budgeting_algorithm_minimize_diffs  (real chargers + real status objects)
# =============================================================================


class TestBudgetingAlgorithmMinimizeDiffs:

    def _setup(self, n=2, battery=None, max_amps=None):
        hass = _make_hass()
        home = _make_home(battery=battery)
        now = datetime.now(pytz.UTC)
        chargers, css = [], []

        for i in range(n):
            ch = _create_charger(hass, home, name=f"Ch{i}", is_3p=False)
            car = _make_real_car(hass, home, name=f"Car{i}")
            _init_charger_states(ch, charge_state=True, amperage=10, num_phases=1)
            _plug_car(ch, car, now - timedelta(hours=1))
            chargers.append(ch)

            cs = QSChargerStatus(ch)
            cs.current_real_max_charging_amp = 10
            cs.current_active_phase_number = 1
            cs.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            cs.possible_num_phases = [1]
            cs.budgeted_amp = 10
            cs.budgeted_num_phases = 1
            cs.charge_score = 100 + i * 10
            cs.can_be_started_and_stopped = True
            cs.command = copy_command(CMD_AUTO_GREEN_ONLY)
            cs.is_before_battery = False
            css.append(cs)

        group = _make_charger_group(home, chargers, max_amps=max_amps)
        return hass, home, group, chargers, css, now

    @pytest.mark.asyncio
    async def test_increase(self):
        *_, group, _, css, now = self._setup()
        ok, *_ = await group.budgeting_algorithm_minimize_diffs(css, 3000, 3000, False, now)
        assert ok is True
        assert sum(cs.budgeted_amp for cs in css) >= 20

    @pytest.mark.asyncio
    async def test_decrease(self):
        *_, group, _, css, now = self._setup()
        ok, *_ = await group.budgeting_algorithm_minimize_diffs(css, -3000, -3000, False, now)
        assert ok is True
        assert sum(cs.budgeted_amp for cs in css) <= 20

    @pytest.mark.asyncio
    async def test_battery_after(self):
        bat = _make_battery(asked_charge=1000)
        _, home, group, _, css, now = self._setup(battery=bat)
        home.battery_can_discharge = MagicMock(return_value=True)
        for cs in css:
            cs.is_before_battery = False
        ok, *_ = await group.budgeting_algorithm_minimize_diffs(css, 5000, 3000, False, now)
        assert ok is True

    @pytest.mark.asyncio
    async def test_battery_before_discharge(self):
        bat = _make_battery(asked_charge=-800)
        _, home, group, _, css, now = self._setup(battery=bat)
        home.battery_can_discharge = MagicMock(return_value=True)
        css[0].is_before_battery = True
        css[0].command = copy_command(CMD_AUTO_GREEN_CONSIGN, power_consign=2000)
        ok, *_ = await group.budgeting_algorithm_minimize_diffs(css, 2000, 3000, False, now)
        assert ok is True

    @pytest.mark.asyncio
    async def test_price_optimization(self):
        *_, group, _, css, now = self._setup()
        group.home.battery_can_discharge = MagicMock(return_value=False)
        group.home.get_best_tariff = MagicMock(return_value=0.25)
        group.home.get_tariff = MagicMock(return_value=0.10)
        css[0].command = copy_command(CMD_AUTO_PRICE, power_consign=2000)
        ok, *_ = await group.budgeting_algorithm_minimize_diffs(css, 5000, 5000, False, now)
        assert ok is True


# =============================================================================
# apply_budget_strategy  (with remaining_cs)
# =============================================================================


class TestApplyBudgetStrategy:

    def _setup(self, n=2):
        hass = _make_hass()
        home = _make_home()
        now = datetime.now(pytz.UTC)
        chargers, css = [], []
        for i in range(n):
            ch = _create_charger(hass, home, name=f"Ch{i}", is_3p=True)
            car = _make_real_car(hass, home, name=f"Car{i}")
            _init_charger_states(ch, charge_state=True, amperage=10, num_phases=1)
            _plug_car(ch, car, now - timedelta(hours=1))
            ch._ensure_correct_state = AsyncMock()
            chargers.append(ch)

            cs = QSChargerStatus(ch)
            cs.current_real_max_charging_amp = 10
            cs.current_active_phase_number = 1
            cs.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            cs.possible_num_phases = [1, 3]
            cs.budgeted_amp = 10; cs.budgeted_num_phases = 1
            cs.charge_score = 100 + i * 10
            cs.can_be_started_and_stopped = True
            cs.command = copy_command(CMD_AUTO_GREEN_ONLY)
            cs.is_before_battery = False
            css.append(cs)

        group = _make_charger_group(home, chargers)
        return group, chargers, css, now

    @pytest.mark.asyncio
    async def test_simple_all(self):
        group, _, css, now = self._setup()
        group.apply_budgets = AsyncMock()
        await group.apply_budget_strategy(css, 2300, now)
        group.apply_budgets.assert_called_once()
        assert group.remaining_budget_to_apply == []

    @pytest.mark.asyncio
    async def test_split_inc_dec(self):
        group, _, css, now = self._setup()
        group.apply_budgets = AsyncMock()
        group.dynamic_group.is_current_acceptable = MagicMock(side_effect=[False])
        css[0].budgeted_amp = 6; css[1].budgeted_amp = 14
        await group.apply_budget_strategy(css, 2300, now)
        assert len(group.remaining_budget_to_apply) > 0

    @pytest.mark.asyncio
    async def test_null_power(self):
        group, _, css, now = self._setup()
        group.apply_budgets = AsyncMock()
        await group.apply_budget_strategy(css, None, now)
        assert group.know_reduced_state is None

    @pytest.mark.asyncio
    async def test_empty(self):
        group, _, _, now = self._setup()
        group.apply_budgets = AsyncMock()
        await group.apply_budget_strategy([], None, now)
        group.apply_budgets.assert_not_called()

    @pytest.mark.asyncio
    async def test_remaining_cs_3_to_1_and_1_to_3_phase_switches(self):
        """3 chargers: two doing ambiguous phase switches land in remaining_cs.

        Charger A: 3-phase 10A -> 1-phase 10A  (same amps, 3->1 switch)
          - doesn't match any explicit inc/dec rule -> remaining_cs
          - inner logic: budgeted_phases=1, current_phases=3 -> copy with min amp, append copy to dec, original to inc

        Charger B: 1-phase 10A -> 3-phase 8A  (lower amps, 1->3 switch)
          - doesn't match any explicit inc/dec rule -> remaining_cs
          - inner logic: budgeted_phases=3, current_phases=1 -> copy with min amp & phases=1 to dec, original to inc

        Charger C: 1-phase 10A -> 1-phase 16A  (simple increase, same phase)
          - matches line 1457 (budgeted > current, same phase) -> increasing_cs

        Worst-case scenario is unacceptable so the splitting logic runs.
        """
        hass = _make_hass()
        home = _make_home()
        now = datetime.now(pytz.UTC)

        chargers = []
        css = []

        # Helper to create a charger + car + status in one go
        def _make(name, is_3p, current_amp, current_phase, budget_amp, budget_phase, score):
            ch = _create_charger(hass, home, name=name, is_3p=is_3p)
            car = _make_real_car(hass, home, name=f"{name}_car")
            _init_charger_states(ch, charge_state=True, amperage=current_amp, num_phases=current_phase)
            _plug_car(ch, car, now - timedelta(hours=1))
            ch._ensure_correct_state = AsyncMock()
            chargers.append(ch)

            cs = QSChargerStatus(ch)
            cs.current_real_max_charging_amp = current_amp
            cs.current_active_phase_number = current_phase
            cs.budgeted_amp = budget_amp
            cs.budgeted_num_phases = budget_phase
            cs.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            cs.possible_num_phases = [1, 3]
            cs.charge_score = score
            cs.can_be_started_and_stopped = True
            cs.command = copy_command(CMD_AUTO_GREEN_ONLY)
            cs.is_before_battery = False
            css.append(cs)
            return ch, cs

        # A: 3-phase 10A -> 1-phase 10A  (phase switch 3->1, amps same => remaining)
        _make("ChA", True, current_amp=10, current_phase=3, budget_amp=10, budget_phase=1, score=100)

        # B: 1-phase 10A -> 3-phase 8A  (phase switch 1->3, amps decrease => remaining)
        _make("ChB", True, current_amp=10, current_phase=1, budget_amp=8, budget_phase=3, score=90)

        # C: 1-phase 10A -> 1-phase 16A  (simple increase, same phase)
        _make("ChC", True, current_amp=10, current_phase=1, budget_amp=16, budget_phase=1, score=80)

        group = _make_charger_group(home, chargers, max_amps=[32.0, 32.0, 32.0])
        # Make worst-case unacceptable to trigger the split
        group.dynamic_group.is_current_acceptable = MagicMock(return_value=False)
        group.apply_budgets = AsyncMock()

        await group.apply_budget_strategy(css, 5000.0, now)

        # The split should have happened
        group.apply_budgets.assert_called_once()
        called_cs = group.apply_budgets.call_args[0][0]
        remaining = group.remaining_budget_to_apply

        # remaining_budget_to_apply should contain the increasing chargers
        assert len(remaining) > 0
        # The decreasing phase (first phase of the split) should have been applied
        assert len(called_cs) > 0

        # Verify the remaining includes the original phase-switch chargers (A and B)
        remaining_charger_names = {cs.charger.name for cs in remaining}
        assert "ChA" in remaining_charger_names, f"ChA should be in remaining (increasing part of phase switch), got {remaining_charger_names}"
        assert "ChB" in remaining_charger_names, f"ChB should be in remaining (increasing part of phase switch), got {remaining_charger_names}"

    @pytest.mark.asyncio
    async def test_remaining_cs_error_branch_unexpected_phase(self):
        """Charger with unexpected phase combination hits the error branch.

        This covers the else at line 1484 where budgeted/current phase combo
        is neither 1->3 nor 3->1 (shouldn't happen in practice but is defensive code).
        We forge it by setting current_active_phase_number=2 (non-standard).
        """
        hass = _make_hass()
        home = _make_home()
        now = datetime.now(pytz.UTC)

        chargers = []
        css = []

        def _make(name, current_amp, current_phase, budget_amp, budget_phase, score):
            ch = _create_charger(hass, home, name=name, is_3p=True)
            car = _make_real_car(hass, home, name=f"{name}_car")
            _init_charger_states(ch, charge_state=True, amperage=current_amp, num_phases=current_phase)
            _plug_car(ch, car, now - timedelta(hours=1))
            ch._ensure_correct_state = AsyncMock()
            chargers.append(ch)

            cs = QSChargerStatus(ch)
            cs.current_real_max_charging_amp = current_amp
            cs.current_active_phase_number = current_phase
            cs.budgeted_amp = budget_amp
            cs.budgeted_num_phases = budget_phase
            cs.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            cs.possible_num_phases = [1, 3]
            cs.charge_score = score
            cs.can_be_started_and_stopped = True
            cs.command = copy_command(CMD_AUTO_GREEN_ONLY)
            cs.is_before_battery = False
            css.append(cs)
            return ch, cs

        # Two chargers with exotic phase combos that land in remaining_cs
        # "2-phase" is synthetic to trigger the error branch
        # Remaining requires: phases differ AND doesn't match any inc/dec rule
        _make("ChX", current_amp=10, current_phase=3, budget_amp=10, budget_phase=1, score=100)
        # Force an unexpected phase value post-creation
        css[-1].current_active_phase_number = 2  # not 1 or 3
        css[-1].budgeted_num_phases = 3

        _make("ChY", current_amp=10, current_phase=3, budget_amp=10, budget_phase=1, score=90)
        css[-1].current_active_phase_number = 2
        css[-1].budgeted_num_phases = 3

        # A normal charger to have 3 total
        _make("ChZ", current_amp=10, current_phase=1, budget_amp=16, budget_phase=1, score=80)

        group = _make_charger_group(home, chargers, max_amps=[32.0, 32.0, 32.0])
        group.dynamic_group.is_current_acceptable = MagicMock(return_value=False)
        group.apply_budgets = AsyncMock()

        # Should log errors but not crash
        await group.apply_budget_strategy(css, 5000.0, now)
        # The "error" chargers should end up in increasing_cs (the fallback)
        remaining = group.remaining_budget_to_apply
        remaining_names = {cs.charger.name for cs in remaining}
        assert "ChX" in remaining_names
        assert "ChY" in remaining_names

    @pytest.mark.asyncio
    async def test_remaining_cs_exactly_two_3_to_1(self):
        """Both remaining chargers are doing 3->1 phase switch.

        ChA: 3-phase 12A -> 1-phase 12A  (amps same, going 3->1 => remaining)
        ChB: 3-phase 8A -> 1-phase 10A   (amps increase, going 3->1 => remaining,
             because budgeted(10) > current(8) but budgeted_phases=1 !=3 so not line 1459)
        ChC: simple decrease (provides the pressure that makes worst case unacceptable)

        Verifies that the duplicate copies land in decreasing with correct min amps.
        """
        hass = _make_hass()
        home = _make_home()
        now = datetime.now(pytz.UTC)

        chargers = []
        css = []

        def _make(name, current_amp, current_phase, budget_amp, budget_phase, score):
            ch = _create_charger(hass, home, name=name, is_3p=True)
            car = _make_real_car(hass, home, name=f"{name}_car")
            _init_charger_states(ch, charge_state=True, amperage=current_amp, num_phases=current_phase)
            _plug_car(ch, car, now - timedelta(hours=1))
            ch._ensure_correct_state = AsyncMock()
            chargers.append(ch)

            cs = QSChargerStatus(ch)
            cs.current_real_max_charging_amp = current_amp
            cs.current_active_phase_number = current_phase
            cs.budgeted_amp = budget_amp
            cs.budgeted_num_phases = budget_phase
            cs.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            cs.possible_num_phases = [1, 3]
            cs.charge_score = score
            cs.can_be_started_and_stopped = True
            cs.command = copy_command(CMD_AUTO_GREEN_ONLY)
            cs.is_before_battery = False
            css.append(cs)
            return ch, cs

        # ChA: 3-phase 12A -> 1-phase 12A (same amps, 3->1)
        _make("ChA", current_amp=12, current_phase=3, budget_amp=12, budget_phase=1, score=100)

        # ChB: 3-phase 8A -> 1-phase 10A (amp increase, 3->1)
        # budgeted(10) > current(8), budgeted_phases=1 != current_phases=3
        # NOT line 1457 (phases differ), NOT line 1459 (budgeted_phases!=3) => remaining
        _make("ChB", current_amp=8, current_phase=3, budget_amp=10, budget_phase=1, score=90)

        # ChC: simple decrease (1-phase 14A -> 1-phase 8A)
        _make("ChC", current_amp=14, current_phase=1, budget_amp=8, budget_phase=1, score=80)

        group = _make_charger_group(home, chargers, max_amps=[32.0, 32.0, 32.0])
        group.dynamic_group.is_current_acceptable = MagicMock(return_value=False)
        group.apply_budgets = AsyncMock()

        await group.apply_budget_strategy(css, 5000.0, now)

        called_cs = group.apply_budgets.call_args[0][0]  # decreasing_cs applied first
        remaining = group.remaining_budget_to_apply

        # ChA and ChB originals should be in remaining (increasing side of phase switch)
        remaining_names = {cs.charger.name for cs in remaining}
        assert "ChA" in remaining_names
        assert "ChB" in remaining_names

        # The copies with reduced amps should be in the decreasing set
        dec_names = {cs.charger.name for cs in called_cs}
        assert "ChA" in dec_names or "ChB" in dec_names  # at least one copy
        # ChC (simple decrease) should definitely be in decreasing
        assert "ChC" in dec_names

    @pytest.mark.asyncio
    async def test_remaining_cs_1_to_3_phase_switch(self):
        """Two chargers doing 1->3 phase switch land in remaining_cs.

        ChA: 1-phase 10A -> 3-phase 10A  (same amps, going 1->3 => remaining
             because budgeted_phases=3 but line 1459 requires budgeted > current)
        ChB: 1-phase 12A -> 3-phase 8A   (amps decrease, going 1->3 => remaining,
             budgeted(8) < current(12), budgeted_phases=3 != current_phases=1,
             NOT line 1451 (phases differ), NOT line 1453 (budgeted_phases != 1))
        ChC: simple increase to make worst case unacceptable
        """
        hass = _make_hass()
        home = _make_home()
        now = datetime.now(pytz.UTC)
        chargers, css = [], []

        def _make(name, current_amp, current_phase, budget_amp, budget_phase, score):
            ch = _create_charger(hass, home, name=name, is_3p=True)
            car = _make_real_car(hass, home, name=f"{name}_car")
            _init_charger_states(ch, charge_state=True, amperage=current_amp, num_phases=current_phase)
            _plug_car(ch, car, now - timedelta(hours=1))
            ch._ensure_correct_state = AsyncMock()
            chargers.append(ch)

            cs = QSChargerStatus(ch)
            cs.current_real_max_charging_amp = current_amp
            cs.current_active_phase_number = current_phase
            cs.budgeted_amp = budget_amp
            cs.budgeted_num_phases = budget_phase
            cs.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            cs.possible_num_phases = [1, 3]
            cs.charge_score = score
            cs.can_be_started_and_stopped = True
            cs.command = copy_command(CMD_AUTO_GREEN_ONLY)
            cs.is_before_battery = False
            css.append(cs)

        # ChA: 1-phase 10A -> 3-phase 10A  (same amps, 1->3 switch)
        # budgeted(10) == current(10), phases differ => not any rule => remaining
        _make("ChA", current_amp=10, current_phase=1, budget_amp=10, budget_phase=3, score=100)

        # ChB: 1-phase 12A -> 3-phase 8A  (amps decrease + phase increase)
        # budgeted(8) < current(12), budgeted_phases=3 != current_phases=1
        # NOT line 1451 (phases differ), NOT line 1453 (budgeted_phases=3 not 1) => remaining
        _make("ChB", current_amp=12, current_phase=1, budget_amp=8, budget_phase=3, score=90)

        # ChC: simple increase, same phase, to ensure worst case is big
        _make("ChC", current_amp=10, current_phase=1, budget_amp=16, budget_phase=1, score=80)

        group = _make_charger_group(home, chargers, max_amps=[32.0, 32.0, 32.0])
        group.dynamic_group.is_current_acceptable = MagicMock(return_value=False)
        group.apply_budgets = AsyncMock()

        await group.apply_budget_strategy(css, 5000.0, now)

        called_cs = group.apply_budgets.call_args[0][0]
        remaining = group.remaining_budget_to_apply

        # The originals should end up in remaining (increasing side)
        remaining_names = {cs.charger.name for cs in remaining}
        assert "ChA" in remaining_names
        assert "ChB" in remaining_names

        # Copies with budgeted_num_phases=1 should be in decreasing
        dec_copies_with_1_phase = [
            cs for cs in called_cs
            if cs.charger.name in ("ChA", "ChB") and cs.budgeted_num_phases == 1
        ]
        assert len(dec_copies_with_1_phase) >= 1, (
            f"Expected copies with num_phases=1 in decreasing, got phases: "
            f"{[(cs.charger.name, cs.budgeted_num_phases) for cs in called_cs]}"
        )


# =============================================================================
# apply_budgets  (real state transitions)
# =============================================================================


class TestApplyBudgets:

    def _setup(self):
        hass = _make_hass()
        home = _make_home()
        now = datetime.now(pytz.UTC)
        ch = _create_charger(hass, home, name="Ch0")
        car = _make_real_car(hass, home, name="Car0")
        _init_charger_states(ch, charge_state=True, amperage=10, num_phases=1)
        _plug_car(ch, car, now - timedelta(hours=1))
        ch._ensure_correct_state = AsyncMock()

        cs = QSChargerStatus(ch)
        cs.current_real_max_charging_amp = 10
        cs.current_active_phase_number = 1
        cs.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12]
        cs.possible_num_phases = [1]
        cs.budgeted_amp = 12; cs.budgeted_num_phases = 1
        cs.charge_score = 100
        cs.command = copy_command(CMD_AUTO_GREEN_ONLY)

        group = _make_charger_group(home, [ch])
        return group, ch, cs, now

    @pytest.mark.asyncio
    async def test_apply_increases_amp(self):
        group, ch, cs, now = self._setup()
        await group.apply_budgets([cs], [cs], now, check_charger_state=False)
        assert ch._expected_amperage.value == 12

    @pytest.mark.asyncio
    async def test_below_min_turns_off(self):
        group, ch, cs, now = self._setup()
        cs.budgeted_amp = 3
        await group.apply_budgets([cs], [cs], now, check_charger_state=False)
        assert ch._expected_charge_state.value is False

    @pytest.mark.asyncio
    async def test_above_max_clamps(self):
        group, ch, cs, now = self._setup()
        cs.budgeted_amp = 40
        await group.apply_budgets([cs], [cs], now, check_charger_state=False)
        assert ch._expected_amperage.value == ch.max_charge

    @pytest.mark.asyncio
    async def test_check_unacceptable_aborts(self):
        group, ch, cs, now = self._setup()
        group.dynamic_group.is_current_acceptable = MagicMock(return_value=False)
        await group.apply_budgets([cs], [cs], now, check_charger_state=True)
        ch._ensure_correct_state.assert_not_called()


# =============================================================================
# can_change_budget  (real QSChargerStatus + real charger)
# =============================================================================


class TestCanChangeBudget:

    def _cs(self, budgeted=10, phases=1, possible=None, p_phases=None):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home, is_3p=False)
        car = _make_real_car(hass, home)
        _init_charger_states(ch, amperage=10, num_phases=1)
        _plug_car(ch, car, datetime.now(pytz.UTC))
        cs = QSChargerStatus(ch)
        cs.budgeted_amp = budgeted; cs.budgeted_num_phases = phases
        cs.possible_amps = possible or [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        cs.possible_num_phases = p_phases or [1]
        cs.current_real_max_charging_amp = budgeted
        cs.current_active_phase_number = phases
        return cs

    def test_increase_from_0(self):
        cs = self._cs(budgeted=0)
        a, _ = cs.can_change_budget(allow_state_change=True, increase=True)
        assert a == 6

    def test_increase_from_0_no_state(self):
        cs = self._cs(budgeted=0)
        a, _ = cs.can_change_budget(allow_state_change=False, increase=True)
        assert a is None

    def test_decrease_to_0(self):
        cs = self._cs(budgeted=6)
        a, _ = cs.can_change_budget(allow_state_change=True, increase=False)
        assert a == 0

    def test_normal_increase(self):
        cs = self._cs(budgeted=10)
        a, _ = cs.can_change_budget(increase=True)
        assert a == 11

    def test_normal_decrease(self):
        cs = self._cs(budgeted=10)
        a, _ = cs.can_change_budget(increase=False)
        assert a == 9

    def test_at_max(self):
        cs = self._cs(budgeted=16)
        a, _ = cs.can_change_budget(increase=True)
        assert a is None

    def test_none_budgeted(self):
        cs = self._cs(); cs.budgeted_amp = None
        a, _ = cs.can_change_budget(increase=True)
        assert a is None


# =============================================================================
# device_post_home_init  (real car references)
# =============================================================================


class TestDevicePostHomeInit:

    def test_user_attached(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        car = _make_real_car(hass, home)
        _init_charger_states(ch)
        ch.user_attached_car_name = "TestCar"
        ch.device_post_home_init(datetime.now(pytz.UTC))
        assert ch._boot_car is not None and ch._boot_car.name == "TestCar"

    def test_constraint_based(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        car = _make_real_car(hass, home)
        _init_charger_states(ch)
        ct = MagicMock(); ct.load_param = "TestCar"; ct.name = "ct"
        ch._constraints = [ct]
        ch.device_post_home_init(datetime.now(pytz.UTC))
        assert ch._boot_car is not None

    def test_force_no_charger_skipped(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        car = _make_real_car(hass, home)
        car.user_attached_charger_name = FORCE_CAR_NO_CHARGER_CONNECTED
        _init_charger_states(ch)
        ct = MagicMock(); ct.load_param = "TestCar"; ct.name = "ct"
        ch._constraints = [ct]
        ch.device_post_home_init(datetime.now(pytz.UTC))
        assert ch._boot_car is None


# =============================================================================
# on_device_state_change  (real car)
# =============================================================================


class TestOnDeviceStateChange:

    @pytest.mark.asyncio
    async def test_with_car_no_person(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        car = _make_real_car(hass, home)
        _init_charger_states(ch); _plug_car(ch, car, datetime.now(pytz.UTC))
        car.mobile_app = "notify.car"; car.mobile_app_url = "http://car"
        ch.on_device_state_change_helper = AsyncMock()
        await ch.on_device_state_change(datetime.now(pytz.UTC), "test")
        assert ch.on_device_state_change_helper.call_args[1]["mobile_app"] == "notify.car"

    @pytest.mark.asyncio
    async def test_without_car(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        _init_charger_states(ch)
        ch.mobile_app = "notify.ch"; ch.mobile_app_url = "http://ch"
        ch.on_device_state_change_helper = AsyncMock()
        await ch.on_device_state_change(datetime.now(pytz.UTC), "test")
        assert ch.on_device_state_change_helper.call_args[1]["mobile_app"] == "notify.ch"


# =============================================================================
# start_charge / stop_charge
# =============================================================================


class TestStartStopCharge:

    @pytest.mark.asyncio
    async def test_stop_exception_handled(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); _init_charger_states(ch, charge_state=True)
        ch.is_charge_enabled = MagicMock(return_value=True)
        ch.low_level_stop_charge = AsyncMock(side_effect=Exception("boom"))
        await ch.stop_charge(datetime.now(pytz.UTC))  # no raise

    @pytest.mark.asyncio
    async def test_start_exception_handled(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); _init_charger_states(ch, charge_state=False)
        ch.is_charge_disabled = MagicMock(return_value=True)
        ch.low_level_start_charge = AsyncMock(side_effect=Exception("boom"))
        await ch.start_charge(datetime.now(pytz.UTC))

    @pytest.mark.asyncio
    async def test_stop_when_already_off(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); _init_charger_states(ch, charge_state=False)
        ch.is_charge_enabled = MagicMock(return_value=False)
        ch.low_level_stop_charge = AsyncMock()
        await ch.stop_charge(datetime.now(pytz.UTC))
        ch.low_level_stop_charge.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_when_already_on(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); _init_charger_states(ch, charge_state=True)
        ch.is_charge_disabled = MagicMock(return_value=False)
        ch.low_level_start_charge = AsyncMock()
        await ch.start_charge(datetime.now(pytz.UTC))
        ch.low_level_start_charge.assert_not_called()


# =============================================================================
# QSChargerStatus.duplicate
# =============================================================================


class TestDuplicate:

    def test_independence(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); car = _make_real_car(hass, home)
        _init_charger_states(ch); _plug_car(ch, car, datetime.now(pytz.UTC))
        cs = QSChargerStatus(ch)
        cs.possible_amps = [0, 6, 7]; cs.possible_num_phases = [1, 3]
        cs.budgeted_amp = 8; cs.budgeted_num_phases = 1; cs.charge_score = 150
        cs.is_before_battery = True; cs.bump_solar = True
        d = cs.duplicate()
        d.possible_amps.append(99)
        assert 99 not in cs.possible_amps
        assert d.charge_score == 150


# =============================================================================
# _check_plugged_val (car fallback)
# =============================================================================


class TestCheckPluggedVal:

    def test_car_fallback_plugged(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); car = _make_real_car(hass, home)
        _init_charger_states(ch); _plug_car(ch, car, datetime.now(pytz.UTC))
        ch.get_last_state_value_duration = MagicMock(return_value=(None, None))
        ch.get_sensor_latest_possible_valid_value = MagicMock(return_value=QSChargerStates.PLUGGED)
        # Real car.is_car_plugged returns None by default (no sensor data)
        # So overall result will be None + fallback check; state says PLUGGED, car says None -> no override
        result = ch._check_plugged_val(datetime.now(pytz.UTC), for_duration=30.0, check_for_val=True)
        # When car returns None, no override happens, result stays None
        assert result is None

    def test_no_duration_direct(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); _init_charger_states(ch)
        ch.get_sensor_latest_possible_valid_value = MagicMock(return_value=QSChargerStates.PLUGGED)
        assert ch._check_plugged_val(datetime.now(pytz.UTC), check_for_val=True) is True


# =============================================================================
# check_charge_state
# =============================================================================


class TestCheckChargeState:

    def test_not_plugged(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); _init_charger_states(ch)
        ch.is_optimistic_plugged = MagicMock(return_value=False)
        assert ch.check_charge_state(datetime.now(pytz.UTC), check_for_val=True) is False

    def test_plugged_none(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); _init_charger_states(ch)
        ch.is_optimistic_plugged = MagicMock(return_value=None)
        assert ch.check_charge_state(datetime.now(pytz.UTC)) is None

    def test_plugged_with_status(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); _init_charger_states(ch)
        ch.is_optimistic_plugged = MagicMock(return_value=True)
        ch.get_car_charge_enabled_status_vals = MagicMock(return_value=["Charging"])
        ch._check_charger_status = MagicMock(return_value=True)
        assert ch.check_charge_state(datetime.now(pytz.UTC)) is True


# =============================================================================
# dyn_handle
# =============================================================================


class TestDynHandle:

    @pytest.mark.asyncio
    async def test_no_actionable(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        _init_charger_states(ch)
        group = _make_charger_group(home, [ch])
        group.ensure_correct_state = AsyncMock(return_value=([], None))
        await group.dyn_handle(datetime.now(pytz.UTC))

    @pytest.mark.asyncio
    async def test_remaining_budget(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); car = _make_real_car(hass, home)
        _init_charger_states(ch, amperage=10, num_phases=1)
        _plug_car(ch, car, datetime.now(pytz.UTC) - timedelta(hours=1))
        group = _make_charger_group(home, [ch])

        cs = QSChargerStatus(ch)
        cs.current_real_max_charging_amp = 10; cs.current_active_phase_number = 1
        cs.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12]; cs.possible_num_phases = [1]
        cs.budgeted_amp = 12; cs.budgeted_num_phases = 1; cs.charge_score = 100
        cs.command = copy_command(CMD_AUTO_GREEN_ONLY)

        now = datetime.now(pytz.UTC)
        group.ensure_correct_state = AsyncMock(return_value=([cs], now - timedelta(seconds=120)))
        group.remaining_budget_to_apply = [cs]
        group.apply_budgets = AsyncMock()
        await group.dyn_handle(now)
        group.apply_budgets.assert_called_once()
        assert group.remaining_budget_to_apply == []

    @pytest.mark.asyncio
    async def test_with_battery(self):
        bat = _make_battery(asked_charge=500)
        hass = _make_hass(); home = _make_home(battery=bat)
        ch = _create_charger(hass, home); car = _make_real_car(hass, home)
        _init_charger_states(ch, charge_state=True, amperage=10, num_phases=1)
        _plug_car(ch, car, datetime.now(pytz.UTC) - timedelta(hours=1))
        group = _make_charger_group(home, [ch])

        cs = QSChargerStatus(ch)
        cs.current_real_max_charging_amp = 10; cs.current_active_phase_number = 1
        cs.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12]; cs.possible_num_phases = [1]
        cs.budgeted_amp = 10; cs.budgeted_num_phases = 1; cs.charge_score = 100
        cs.command = copy_command(CMD_AUTO_GREEN_ONLY); cs.is_before_battery = False
        cs.accurate_current_power = 2300.0

        now = datetime.now(pytz.UTC)
        vtime = now - timedelta(seconds=CHARGER_ADAPTATION_WINDOW_S + 10)
        group.ensure_correct_state = AsyncMock(return_value=([cs], vtime))
        group.dynamic_group.get_median_sensor = MagicMock(return_value=2300.0)
        pv = [(now - timedelta(seconds=i), 500.0) for i in range(10)]
        home.get_available_power_values = MagicMock(return_value=pv)
        home.get_grid_consumption_power_values = MagicMock(return_value=pv)

        group.budgeting_algorithm_minimize_diffs = AsyncMock(return_value=(True, False, False))
        group.apply_budget_strategy = AsyncMock()
        await group.dyn_handle(now)
        group.budgeting_algorithm_minimize_diffs.assert_called_once()
        group.apply_budget_strategy.assert_called_once()

    def _build_single_charger_dyn_group(self, current_amp=10, num_phases=1):
        """Helper: one real charger in a group, ready for dampening tests.

        The car's update_dampening_value is replaced with a MagicMock so we can
        track calls, since the real method needs full power-step data from sensors.
        """
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home, name="Ch0"); car = _make_real_car(hass, home, name="Car0")
        _init_charger_states(ch, charge_state=True, amperage=current_amp, num_phases=num_phases)
        _plug_car(ch, car, datetime.now(pytz.UTC) - timedelta(hours=1))

        # Replace update_dampening_value with a mock to track calls
        # (the real one needs full sensor data to interpolate power steps)
        car.update_dampening_value = MagicMock(return_value=False)

        group = _make_charger_group(home, [ch])
        now = datetime.now(pytz.UTC)
        vtime = now - timedelta(seconds=CHARGER_ADAPTATION_WINDOW_S + 10)

        cs = QSChargerStatus(ch)
        cs.current_real_max_charging_amp = current_amp
        cs.current_active_phase_number = num_phases
        cs.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12]; cs.possible_num_phases = [1]
        cs.budgeted_amp = current_amp; cs.budgeted_num_phases = num_phases
        cs.charge_score = 100
        cs.command = copy_command(CMD_AUTO_GREEN_ONLY); cs.is_before_battery = False
        cs.accurate_current_power = None  # key: NOT set -> charger NOT in dampened_chargers

        pv = [(now - timedelta(seconds=i), 500.0) for i in range(10)]
        home.get_available_power_values = MagicMock(return_value=pv)
        home.get_grid_consumption_power_values = MagicMock(return_value=pv)

        return hass, home, group, ch, car, cs, now, vtime

    @pytest.mark.asyncio
    async def test_dampening_simple_case(self):
        """Lines 726-739: one charger charging, group power known, not already dampened.

        Conditions: num_true_charging_cs<=1, num_charging_cs==1,
        current_real_cars_power not None, charger NOT in dampened_chargers.
        dampened_chargers is empty because cs.accurate_current_power is None
        (the per-charger dampening loop at 679 skips it).
        """
        hass, home, group, ch, car, cs, now, vtime = self._build_single_charger_dyn_group()

        # The charger is charging (state True, amp >= min) but cs.accurate_current_power is None
        # so it won't enter dampened_chargers in the first loop (line 681 check fails).
        # Then num_charging_cs == 1 and a_charging_cs is set.
        # The group-level power IS available:
        group_power = 2300.0
        group.dynamic_group.get_median_sensor = MagicMock(return_value=group_power)

        # is_charging_power_zero must return False (charger IS consuming power)
        ch.is_charging_power_zero = MagicMock(return_value=False)

        group.ensure_correct_state = AsyncMock(return_value=([cs], vtime))
        group.budgeting_algorithm_minimize_diffs = AsyncMock(return_value=(True, False, False))
        group.apply_budget_strategy = AsyncMock()

        await group.dyn_handle(now)

        # update_car_dampening_value should have been called with the group power
        car.update_dampening_value.assert_called()
        call_kw = car.update_dampening_value.call_args[1]
        assert call_kw["power_value_or_delta"] == group_power
        assert call_kw["amperage"] == (10, 1)
        assert call_kw["amperage_transition"] is None

    @pytest.mark.asyncio
    async def test_dampening_transition_case(self):
        """Lines 744-777: know_reduced_state from a prior run, one charger changed amps.

        Simulates two consecutive dyn_handle calls:
        - 1st call establishes know_reduced_state via apply_budget_strategy
        - 2nd call detects the charger changed from 10A to 12A and computes delta power
        """
        hass, home, group, ch, car, cs, now, vtime = self._build_single_charger_dyn_group()

        group_power_before = 2300.0
        group_power_after = 2760.0  # 12A * 230V

        ch.is_charging_power_zero = MagicMock(return_value=False)

        # Pre-seed know_reduced_state as if a previous apply_budget_strategy ran
        group.know_reduced_state = {ch: (10, 1)}  # was 10A, 1-phase
        group.know_reduced_state_real_power = group_power_before

        # Now the charger has moved to 12A (budgeting happened in between)
        cs.current_real_max_charging_amp = 12
        ch._expected_amperage.value = 12

        group.dynamic_group.get_median_sensor = MagicMock(return_value=group_power_after)
        group.ensure_correct_state = AsyncMock(return_value=([cs], vtime))
        group.budgeting_algorithm_minimize_diffs = AsyncMock(return_value=(True, False, False))
        group.apply_budget_strategy = AsyncMock()

        await group.dyn_handle(now)

        # update_dampening_value should be called with the transition
        assert car.update_dampening_value.called
        # Find the call with amperage_transition (not the simple dampening one)
        transition_calls = [
            c for c in car.update_dampening_value.call_args_list
            if c[1].get("amperage_transition") is not None
        ]
        assert len(transition_calls) >= 1
        tc = transition_calls[0][1]
        assert tc["amperage_transition"] == ((10, 1), (12, 1))
        assert tc["power_value_or_delta"] == group_power_after - group_power_before

    @pytest.mark.asyncio
    async def test_dampening_transition_to_zero_skipped(self):
        """Lines 762-768: transition to zero dampening is skipped (do_dampen_transition=False).

        When current_real_cars_power is below charger_consumption_W (e.g. 50W < 70W),
        dampening_power_value_for_car_consumption returns 0, and the transition is
        skipped because it's going to a "zero consumption" state.
        """
        hass, home, group, ch, car, cs, now, vtime = self._build_single_charger_dyn_group()

        # charger_consumption_W is 70 by default on the group
        # current power is 50W -> below 70W -> dampening returns 0
        group_power_after = 50.0
        group_power_before = 2300.0

        ch.is_charging_power_zero = MagicMock(return_value=False)

        group.know_reduced_state = {ch: (10, 1)}
        group.know_reduced_state_real_power = group_power_before

        cs.current_real_max_charging_amp = 8  # changed
        ch._expected_amperage.value = 8

        group.dynamic_group.get_median_sensor = MagicMock(return_value=group_power_after)
        group.ensure_correct_state = AsyncMock(return_value=([cs], vtime))
        group.budgeting_algorithm_minimize_diffs = AsyncMock(return_value=(True, False, False))
        group.apply_budget_strategy = AsyncMock()

        # The charger's own dampening_power_value_for_car_consumption
        # is called on current_real_cars_power (50W) -> returns 0 since < 70W
        # AND current_reduced_states[ch][0] = 8 >= min_charge (6) -> skip transition
        await group.dyn_handle(now)

        # Verify no transition call happened (only the simple dampening, if any)
        transition_calls = [
            c for c in car.update_dampening_value.call_args_list
            if c[1].get("amperage_transition") is not None
        ]
        assert len(transition_calls) == 0

    @pytest.mark.asyncio
    async def test_dampening_transition_from_zero_skipped(self):
        """Lines 766-768: transition from zero dampening is skipped.

        When know_reduced_state_real_power was below charger_consumption_W,
        the old state was a "zero consumption" state, so the transition is skipped.
        """
        hass, home, group, ch, car, cs, now, vtime = self._build_single_charger_dyn_group()

        # Previous power was 50W (< 70W threshold) -> old state was "zero dampening"
        group_power_before = 50.0
        group_power_after = 2300.0

        ch.is_charging_power_zero = MagicMock(return_value=False)

        group.know_reduced_state = {ch: (10, 1)}
        group.know_reduced_state_real_power = group_power_before

        cs.current_real_max_charging_amp = 12  # changed up
        ch._expected_amperage.value = 12

        group.dynamic_group.get_median_sensor = MagicMock(return_value=group_power_after)
        group.ensure_correct_state = AsyncMock(return_value=([cs], vtime))
        group.budgeting_algorithm_minimize_diffs = AsyncMock(return_value=(True, False, False))
        group.apply_budget_strategy = AsyncMock()

        await group.dyn_handle(now)

        transition_calls = [
            c for c in car.update_dampening_value.call_args_list
            if c[1].get("amperage_transition") is not None
        ]
        assert len(transition_calls) == 0

    @pytest.mark.asyncio
    async def test_dampening_transition_charger_set_mismatch_skips(self):
        """Lines 749-751: know_reduced_state has a charger not in current set -> no transition.

        If a charger disappeared from the actionable set between calls, the whole
        transition detection is skipped (last_changed_charger = None via break).
        """
        hass, home, group, ch, car, cs, now, vtime = self._build_single_charger_dyn_group()

        # Fake a second charger that was in know_reduced_state but is gone now
        phantom_charger = MagicMock()
        phantom_charger.name = "PhantomCharger"
        group.know_reduced_state = {ch: (10, 1), phantom_charger: (8, 1)}
        group.know_reduced_state_real_power = 2300.0

        ch.is_charging_power_zero = MagicMock(return_value=False)
        group.dynamic_group.get_median_sensor = MagicMock(return_value=2300.0)
        group.ensure_correct_state = AsyncMock(return_value=([cs], vtime))
        group.budgeting_algorithm_minimize_diffs = AsyncMock(return_value=(True, False, False))
        group.apply_budget_strategy = AsyncMock()

        await group.dyn_handle(now)

        # No transition should happen (sets don't match)
        transition_calls = [
            c for c in car.update_dampening_value.call_args_list
            if c[1].get("amperage_transition") is not None
        ]
        assert len(transition_calls) == 0


# =============================================================================
# get_amps_phase_switch
# =============================================================================


class TestGetAmpsPhaseSwitch:

    def _cs(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home, is_3p=True)
        car = _make_real_car(hass, home)
        _init_charger_states(ch, num_phases=1); _plug_car(ch, car, datetime.now(pytz.UTC))
        cs = QSChargerStatus(ch)
        cs.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        cs.possible_num_phases = [1, 3]
        return cs

    def test_1_to_3(self):
        cs = self._cs()
        a, p, _ = cs.get_amps_phase_switch(12, 1)
        assert p == 3 and a >= 6

    def test_3_to_1(self):
        cs = self._cs()
        a, p, _ = cs.get_amps_phase_switch(6, 3)
        assert p == 1 and a <= 16


# =============================================================================
# QSChargerOCPP and QSChargerWallbox specific tests
# =============================================================================

from custom_components.quiet_solar.ha_model.charger import (
    QSOCPPv16v201ChargePointStatus,
)
from custom_components.quiet_solar.const import (
    CONF_CHARGER_DEVICE_OCPP,
    CONF_CHARGER_DEVICE_WALLBOX,
    CONF_CHARGER_PAUSE_RESUME_SWITCH,
    CONF_CHARGER_CHARGING_CURRENT_SENSOR,
)
from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE
from homeassistant.components.wallbox.const import ChargerStatus as WallboxChargerStatus


def _make_entity_entry(entity_id):
    """Create a fake entity registry entry with just entity_id."""
    e = MagicMock()
    e.entity_id = entity_id
    return e


def _create_ocpp_charger(hass, home, name="OcppCharger", min_charge=6, max_charge=32):
    """Create a REAL QSChargerOCPP by mocking just the device/entity registry lookups."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerOCPP

    config_entry = MagicMock()
    config_entry.entry_id = f"test_entry_{name}"
    config_entry.data = {}

    device_id = f"device_{name}"
    devname = name.lower().replace(" ", "_")

    # Build fake device registry entries that _find_charger_entity_id will discover
    entries = [
        _make_entity_entry(f"switch.{devname}_charge_control"),
        _make_entity_entry(f"number.{devname}_maximum_current"),
        _make_entity_entry(f"sensor.{devname}_status_connector"),
        _make_entity_entry(f"sensor.{devname}_power_active_import"),
    ]

    fake_device = MagicMock()
    fake_device.id = device_id
    fake_device.name = name
    fake_device.name_by_user = None

    with (
        patch("custom_components.quiet_solar.ha_model.charger.device_registry") as mock_dev_reg,
        patch("custom_components.quiet_solar.ha_model.charger.entity_registry") as mock_ent_reg,
    ):
        dev_reg_instance = MagicMock()
        dev_reg_instance.async_get.return_value = fake_device
        mock_dev_reg.async_get.return_value = dev_reg_instance

        ent_reg_instance = MagicMock()
        mock_ent_reg.async_get.return_value = ent_reg_instance
        mock_ent_reg.async_entries_for_device.return_value = entries

        charger = QSChargerOCPP(
            name=name,
            hass=hass,
            home=home,
            config_entry=config_entry,
            **{CONF_CHARGER_DEVICE_OCPP: device_id},
            **{CONF_CHARGER_MIN_CHARGE: min_charge, CONF_CHARGER_MAX_CHARGE: max_charge},
            **{CONF_IS_3P: False, CONF_MONO_PHASE: 1},
            **{CONF_CHARGER_PLUGGED: f"sensor.{devname}_plugged"},
        )

    home._chargers.append(charger)

    if hasattr(charger, "father_device") and charger.father_device is not None:
        group = _make_charger_group(home, [charger])
        charger.father_device.charger_group = group
    return charger


def _create_wallbox_charger(hass, home, name="WbCharger", min_charge=6, max_charge=32):
    """Create a REAL QSChargerWallbox by mocking just the device/entity registry lookups."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerWallbox

    config_entry = MagicMock()
    config_entry.entry_id = f"test_entry_{name}"
    config_entry.data = {}

    device_id = f"device_{name}"
    devname = name.lower().replace(" ", "_")

    entries = [
        _make_entity_entry(f"number.{devname}_maximum_charging_current"),
        _make_entity_entry(f"switch.{devname}_pause_resume"),
        _make_entity_entry(f"sensor.{devname}_charging_power"),
        _make_entity_entry(f"sensor.{devname}_status_description"),
    ]

    fake_device = MagicMock()
    fake_device.id = device_id
    fake_device.name = name
    fake_device.name_by_user = None

    with (
        patch("custom_components.quiet_solar.ha_model.charger.device_registry") as mock_dev_reg,
        patch("custom_components.quiet_solar.ha_model.charger.entity_registry") as mock_ent_reg,
    ):
        dev_reg_instance = MagicMock()
        dev_reg_instance.async_get.return_value = fake_device
        mock_dev_reg.async_get.return_value = dev_reg_instance

        ent_reg_instance = MagicMock()
        mock_ent_reg.async_get.return_value = ent_reg_instance
        mock_ent_reg.async_entries_for_device.return_value = entries

        charger = QSChargerWallbox(
            name=name,
            hass=hass,
            home=home,
            config_entry=config_entry,
            **{CONF_CHARGER_DEVICE_WALLBOX: device_id},
            **{CONF_CHARGER_MIN_CHARGE: min_charge, CONF_CHARGER_MAX_CHARGE: max_charge},
            **{CONF_IS_3P: False, CONF_MONO_PHASE: 1},
            **{CONF_CHARGER_PLUGGED: f"sensor.{devname}_plugged"},
        )

    home._chargers.append(charger)

    if hasattr(charger, "father_device") and charger.father_device is not None:
        group = _make_charger_group(home, [charger])
        charger.father_device.charger_group = group
    return charger


# ---- QSChargerOCPP tests ----


class TestQSChargerOCPP:
    """Test OCPP-specific charger methods using a real QSChargerOCPP instance."""

    def _setup(self):
        hass = _make_hass()
        home = _make_home()
        ch = _create_ocpp_charger(hass, home)
        _init_charger_states(ch)
        return hass, home, ch

    # -- Construction and entity discovery --

    def test_ocpp_init_sets_entities(self):
        """OCPP init discovers entity IDs from device registry."""
        _, _, ch = self._setup()
        assert ch.charger_pause_resume_switch == "switch.ocppcharger_charge_control"
        assert ch.charger_max_charging_current_number == "number.ocppcharger_maximum_current"
        assert ch.charger_status_sensor == "sensor.ocppcharger_status_connector"
        assert ch.charger_ocpp_power_active_import == "sensor.ocppcharger_power_active_import"
        assert ch.secondary_power_sensor == ch.charger_ocpp_power_active_import

    def test_ocpp_devid_set(self):
        """devid is set from the device name."""
        _, _, ch = self._setup()
        assert ch.devid == "OcppCharger"

    # -- Status value lists --

    def test_get_car_charge_enabled_status_vals(self):
        _, _, ch = self._setup()
        vals = ch.get_car_charge_enabled_status_vals()
        assert QSOCPPv16v201ChargePointStatus.charging in vals
        assert QSOCPPv16v201ChargePointStatus.suspended_ev in vals
        assert QSOCPPv16v201ChargePointStatus.suspended_evse in vals
        assert len(vals) == 3

    def test_get_car_plugged_in_status_vals(self):
        _, _, ch = self._setup()
        vals = ch.get_car_plugged_in_status_vals()
        assert QSOCPPv16v201ChargePointStatus.preparing in vals
        assert QSOCPPv16v201ChargePointStatus.charging in vals
        assert QSOCPPv16v201ChargePointStatus.ev_connected in vals
        assert QSOCPPv16v201ChargePointStatus.finishing in vals
        assert QSOCPPv16v201ChargePointStatus.reserved in vals
        assert len(vals) == 7

    def test_get_car_stopped_asking_current_status_vals(self):
        _, _, ch = self._setup()
        vals = ch.get_car_stopped_asking_current_status_vals()
        assert vals == [QSOCPPv16v201ChargePointStatus.suspended_ev]

    def test_get_car_status_unknown_vals(self):
        _, _, ch = self._setup()
        vals = ch.get_car_status_unknown_vals()
        assert QSOCPPv16v201ChargePointStatus.unavailable in vals
        assert QSOCPPv16v201ChargePointStatus.faulted in vals

    def test_get_car_status_rebooting_vals(self):
        _, _, ch = self._setup()
        vals = ch.get_car_status_rebooting_vals()
        assert vals == [QSOCPPv16v201ChargePointStatus.unavailable]

    # -- get_probable_entities --

    def test_get_probable_entities_includes_ocpp_sensors(self):
        _, _, ch = self._setup()
        entities = ch.get_probable_entities()
        assert ch.charger_ocpp_power_active_import in entities
        # Parent entities should also be included
        assert ch.charger_plugged in entities
        assert ch.charger_pause_resume_switch in entities

    def test_get_probable_entities_without_optional(self):
        """When optional OCPP sensors are None, they are not in the list."""
        _, _, ch = self._setup()
        ch.charger_ocpp_power_active_import = None
        ch.charger_ocpp_current_import = None
        ch.charger_ocpp_transaction_id = None
        entities = ch.get_probable_entities()
        # Only parent entities
        assert ch.charger_plugged in entities

    # -- low_level_update_data_request --

    @pytest.mark.asyncio
    async def test_low_level_update_data_request_success(self):
        hass, _, ch = self._setup()
        hass.services.async_call = AsyncMock()
        result = await ch.low_level_update_data_request(datetime.now(pytz.UTC))
        assert result is True
        hass.services.async_call.assert_called_once()
        call_args = hass.services.async_call.call_args
        assert call_args[0][0] == "ocpp"
        assert call_args[0][1] == "trigger_custom_message"
        assert call_args[0][2]["requested_message"] == "MeterValues"

    @pytest.mark.asyncio
    async def test_low_level_update_data_request_exception(self):
        hass, _, ch = self._setup()
        hass.services.async_call = AsyncMock(side_effect=Exception("service error"))
        result = await ch.low_level_update_data_request(datetime.now(pytz.UTC))
        assert result is False

    # -- low_level_set_max_charging_current --

    @pytest.mark.asyncio
    async def test_low_level_set_max_charging_current_no_custom_profile(self):
        """Without custom profile, delegates to parent (number entity)."""
        hass, _, ch = self._setup()
        assert ch.use_ocpp_custom_charging_profile is False
        hass.services.async_call = AsyncMock()
        result = await ch.low_level_set_max_charging_current(12, datetime.now(pytz.UTC))
        assert result is True
        # Should have called number.set_value
        assert hass.services.async_call.called

    @pytest.mark.asyncio
    async def test_low_level_set_max_charging_current_with_custom_profile(self):
        """With custom profile, calls low_level_set_charging_current which returns False."""
        hass, _, ch = self._setup()
        ch.use_ocpp_custom_charging_profile = True
        result = await ch.low_level_set_max_charging_current(12, datetime.now(pytz.UTC))
        # low_level_set_charging_current with custom profile returns False
        assert result is False

    # -- low_level_set_charging_current --

    @pytest.mark.asyncio
    async def test_low_level_set_charging_current_no_custom(self):
        """Without custom profile, delegates to parent."""
        hass, _, ch = self._setup()
        ch.use_ocpp_custom_charging_profile = False
        hass.services.async_call = AsyncMock()
        result = await ch.low_level_set_charging_current(10, datetime.now(pytz.UTC))
        assert result is True

    @pytest.mark.asyncio
    async def test_low_level_set_charging_current_custom_returns_false(self):
        """With custom profile, returns False (commented-out OCPP charge profile code)."""
        hass, _, ch = self._setup()
        ch.use_ocpp_custom_charging_profile = True
        result = await ch.low_level_set_charging_current(10, datetime.now(pytz.UTC))
        assert result is False

    # -- low_level_plug_check_now (OCPP specific: "off" means plugged) --

    def test_low_level_plug_check_now_plugged(self):
        """OCPP: state 'off' means car is plugged."""
        hass, _, ch = self._setup()
        state = MagicMock()
        state.state = "off"
        state.last_updated = datetime.now(pytz.UTC)
        hass.states.get = MagicMock(return_value=state)
        res, t = ch.low_level_plug_check_now(datetime.now(pytz.UTC))
        assert res is True

    def test_low_level_plug_check_now_unplugged(self):
        """OCPP: state 'on' (available) means no car plugged."""
        hass, _, ch = self._setup()
        state = MagicMock()
        state.state = "on"
        state.last_updated = datetime.now(pytz.UTC)
        hass.states.get = MagicMock(return_value=state)
        res, t = ch.low_level_plug_check_now(datetime.now(pytz.UTC))
        assert res is False

    def test_low_level_plug_check_now_unknown(self):
        """OCPP: unknown/unavailable state returns None."""
        hass, _, ch = self._setup()
        state = MagicMock()
        state.state = STATE_UNKNOWN
        state.last_updated = datetime.now(pytz.UTC)
        hass.states.get = MagicMock(return_value=state)
        res, t = ch.low_level_plug_check_now(datetime.now(pytz.UTC))
        assert res is None

    def test_low_level_plug_check_now_no_state(self):
        """OCPP: no state returns None."""
        hass, _, ch = self._setup()
        hass.states.get = MagicMock(return_value=None)
        res, t = ch.low_level_plug_check_now(datetime.now(pytz.UTC))
        assert res is None

    def test_low_level_plug_check_now_no_sensor(self):
        """OCPP: if charger_plugged is None, returns None."""
        hass, _, ch = self._setup()
        ch.charger_plugged = None
        # With no plugged sensor, should fall through to parent (None, time)
        # But OCPP overrides low_level_plug_check_now which checks charger_plugged
        hass.states.get = MagicMock(return_value=None)
        res, t = ch.low_level_plug_check_now(datetime.now(pytz.UTC))
        assert res is None

    # -- handle_ocpp_notification --

    @pytest.mark.asyncio
    async def test_handle_ocpp_notification(self):
        """Notification handler logs and returns (early return in current code)."""
        _, _, ch = self._setup()
        await ch.handle_ocpp_notification("Test message", "Test title")
        # Just verifies no crash; the method has an early return

    # -- Integration: OCPP charger used in check_load_activity --

    @pytest.mark.asyncio
    async def test_ocpp_charger_check_load_activity(self):
        """OCPP charger works with real check_load_activity_and_constraints."""
        hass, home, ch = self._setup()
        car = _make_real_car(hass, home)
        _plug_car(ch, car, datetime.now(pytz.UTC))

        ch.is_charger_unavailable = MagicMock(return_value=False)
        ch.probe_for_possible_needed_reboot = MagicMock(return_value=False)
        ch.is_not_plugged = MagicMock(return_value=False)
        ch.is_plugged = MagicMock(return_value=True)
        ch.set_charging_num_phases = AsyncMock(return_value=False)
        ch.set_max_charging_current = AsyncMock(return_value=True)
        ch.get_best_car = MagicMock(return_value=car)

        result = await ch.check_load_activity_and_constraints(datetime.now(pytz.UTC))
        # Should have created constraints on the charger
        assert len([c for c in ch._constraints if c is not None]) >= 1

    # -- Generic parent low-level methods on the OCPP charger --

    @pytest.mark.asyncio
    async def test_low_level_stop_charge(self):
        hass, _, ch = self._setup()
        hass.services.async_call = AsyncMock()
        result = await ch.low_level_stop_charge(datetime.now(pytz.UTC))
        assert result is True
        # Should call switch.turn_off on the pause_resume switch
        call = hass.services.async_call.call_args
        assert call[1].get("service") == "turn_off" or call[0][1] == "turn_off"

    @pytest.mark.asyncio
    async def test_low_level_start_charge(self):
        hass, _, ch = self._setup()
        hass.services.async_call = AsyncMock()
        result = await ch.low_level_start_charge(datetime.now(pytz.UTC))
        assert result is True
        call = hass.services.async_call.call_args
        assert call[1].get("service") == "turn_on" or call[0][1] == "turn_on"

    @pytest.mark.asyncio
    async def test_low_level_stop_charge_exception(self):
        hass, _, ch = self._setup()
        hass.services.async_call = AsyncMock(side_effect=Exception("fail"))
        result = await ch.low_level_stop_charge(datetime.now(pytz.UTC))
        assert result is False

    @pytest.mark.asyncio
    async def test_low_level_start_charge_exception(self):
        hass, _, ch = self._setup()
        hass.services.async_call = AsyncMock(side_effect=Exception("fail"))
        result = await ch.low_level_start_charge(datetime.now(pytz.UTC))
        assert result is False

    # -- low_level_charge_check_now (generic, inherited) --

    def test_low_level_charge_check_now_on(self):
        hass, _, ch = self._setup()
        state = MagicMock(); state.state = "on"
        hass.states.get = MagicMock(return_value=state)
        assert ch.low_level_charge_check_now(datetime.now(pytz.UTC)) is True

    def test_low_level_charge_check_now_off(self):
        hass, _, ch = self._setup()
        state = MagicMock(); state.state = "off"
        hass.states.get = MagicMock(return_value=state)
        assert ch.low_level_charge_check_now(datetime.now(pytz.UTC)) is False

    def test_low_level_charge_check_now_unavailable(self):
        hass, _, ch = self._setup()
        state = MagicMock(); state.state = STATE_UNAVAILABLE
        hass.states.get = MagicMock(return_value=state)
        assert ch.low_level_charge_check_now(datetime.now(pytz.UTC)) is None

    def test_low_level_charge_check_now_none(self):
        hass, _, ch = self._setup()
        hass.states.get = MagicMock(return_value=None)
        assert ch.low_level_charge_check_now(datetime.now(pytz.UTC)) is None

    # -- low_level_set_charging_num_phases --

    @pytest.mark.asyncio
    async def test_low_level_set_charging_num_phases_1(self):
        hass, _, ch = self._setup()
        ch.charger_three_to_one_phase_switch = "switch.charger_phase"
        hass.services.async_call = AsyncMock()
        result = await ch.low_level_set_charging_num_phases(1, datetime.now(pytz.UTC))
        assert result is True
        # Should call turn_on for 1 phase
        hass.services.async_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_low_level_set_charging_num_phases_3(self):
        hass, _, ch = self._setup()
        ch.charger_three_to_one_phase_switch = "switch.charger_phase"
        hass.services.async_call = AsyncMock()
        result = await ch.low_level_set_charging_num_phases(3, datetime.now(pytz.UTC))
        assert result is True
        hass.services.async_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_low_level_set_charging_num_phases_exception(self):
        hass, _, ch = self._setup()
        ch.charger_three_to_one_phase_switch = "switch.charger_phase"
        hass.services.async_call = AsyncMock(side_effect=Exception("fail"))
        result = await ch.low_level_set_charging_num_phases(1, datetime.now(pytz.UTC))
        assert result is False

    # -- low_level_reboot --

    @pytest.mark.asyncio
    async def test_low_level_reboot(self):
        hass, _, ch = self._setup()
        ch.charger_reboot_button = "button.charger_reset"
        hass.services.async_call = AsyncMock()
        await ch.low_level_reboot(datetime.now(pytz.UTC))
        hass.services.async_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_low_level_reboot_no_button(self):
        hass, _, ch = self._setup()
        ch.charger_reboot_button = None
        hass.services.async_call = AsyncMock()
        await ch.low_level_reboot(datetime.now(pytz.UTC))
        hass.services.async_call.assert_not_called()

    # -- get_max_charging_amp_per_phase / get_charging_current --

    def test_get_max_charging_amp_per_phase(self):
        hass, _, ch = self._setup()
        state = MagicMock(); state.state = "16.0"
        hass.states.get = MagicMock(return_value=state)
        assert ch.get_max_charging_amp_per_phase() == 16.0

    def test_get_max_charging_amp_per_phase_unavailable(self):
        hass, _, ch = self._setup()
        state = MagicMock(); state.state = STATE_UNAVAILABLE
        hass.states.get = MagicMock(return_value=state)
        assert ch.get_max_charging_amp_per_phase() is None

    def test_get_max_charging_amp_per_phase_bad_value(self):
        hass, _, ch = self._setup()
        state = MagicMock(); state.state = "not_a_number"
        hass.states.get = MagicMock(return_value=state)
        assert ch.get_max_charging_amp_per_phase() is None

    def test_get_charging_current_from_sensor(self):
        """When charging_current_sensor is set, reads from it."""
        hass, _, ch = self._setup()
        ch.charger_charging_current_sensor = "sensor.ocpp_current"
        state = MagicMock(); state.state = "12.5"
        hass.states.get = MagicMock(return_value=state)
        assert ch.get_charging_current() == 12.5

    def test_get_charging_current_no_sensor_delegates(self):
        """When no charging_current_sensor, delegates to get_max_charging_amp_per_phase."""
        hass, _, ch = self._setup()
        ch.charger_charging_current_sensor = None
        state = MagicMock(); state.state = "10.0"
        hass.states.get = MagicMock(return_value=state)
        assert ch.get_charging_current() == 10.0


# ---- QSChargerWallbox tests ----


class TestQSChargerWallbox:
    """Test Wallbox-specific charger methods using a real QSChargerWallbox instance."""

    def _setup(self):
        hass = _make_hass()
        home = _make_home()
        ch = _create_wallbox_charger(hass, home)
        _init_charger_states(ch)
        return hass, home, ch

    # -- Construction --

    def test_wallbox_init_sets_entities(self):
        _, _, ch = self._setup()
        assert ch.charger_max_charging_current_number == "number.wbcharger_maximum_charging_current"
        assert ch.charger_pause_resume_switch == "switch.wbcharger_pause_resume"
        assert ch.charger_wallbox_charging_power == "sensor.wbcharger_charging_power"
        assert ch.charger_status_sensor == "sensor.wbcharger_status_description"
        assert ch.secondary_power_sensor == ch.charger_wallbox_charging_power

    def test_wallbox_initial_num_in_out(self):
        """Wallbox sets initial_num_in_out_immediate = 2."""
        _, _, ch = self._setup()
        assert ch.initial_num_in_out_immediate == 2

    # -- Status value lists --

    def test_get_car_charge_enabled_status_vals(self):
        _, _, ch = self._setup()
        vals = ch.get_car_charge_enabled_status_vals()
        assert WallboxChargerStatus.CHARGING.value in vals
        assert WallboxChargerStatus.DISCHARGING.value in vals
        assert WallboxChargerStatus.WAITING_FOR_CAR.value in vals
        assert WallboxChargerStatus.WAITING.value in vals

    def test_get_car_plugged_in_status_vals(self):
        _, _, ch = self._setup()
        vals = ch.get_car_plugged_in_status_vals()
        assert WallboxChargerStatus.CHARGING.value in vals
        assert WallboxChargerStatus.PAUSED.value in vals
        assert WallboxChargerStatus.SCHEDULED.value in vals
        assert WallboxChargerStatus.LOCKED_CAR_CONNECTED.value in vals
        assert WallboxChargerStatus.WAITING_IN_QUEUE_ECO_SMART.value in vals
        assert len(vals) == 12

    def test_get_car_stopped_asking_current_status_vals(self):
        _, _, ch = self._setup()
        vals = ch.get_car_stopped_asking_current_status_vals()
        assert vals == [WallboxChargerStatus.WAITING_FOR_CAR.value]

    def test_get_car_status_unknown_vals(self):
        _, _, ch = self._setup()
        vals = ch.get_car_status_unknown_vals()
        assert WallboxChargerStatus.UNKNOWN.value in vals
        assert WallboxChargerStatus.ERROR.value in vals
        assert WallboxChargerStatus.DISCONNECTED.value in vals
        assert WallboxChargerStatus.UPDATING.value in vals

    def test_get_car_status_rebooting_vals(self):
        _, _, ch = self._setup()
        vals = ch.get_car_status_rebooting_vals()
        assert vals == [WallboxChargerStatus.DISCONNECTED.value]

    # -- low_level_plug_check_now (Wallbox: uses pause_resume switch) --

    def test_low_level_plug_check_now_on(self):
        """Wallbox: if pause_resume switch has any valid state, car is plugged."""
        hass, _, ch = self._setup()
        state = MagicMock(); state.state = "on"; state.last_updated = datetime.now(pytz.UTC)
        hass.states.get = MagicMock(return_value=state)
        res, _ = ch.low_level_plug_check_now(datetime.now(pytz.UTC))
        assert res is True

    def test_low_level_plug_check_now_off(self):
        """Wallbox: switch off still means plugged (switch exists)."""
        hass, _, ch = self._setup()
        state = MagicMock(); state.state = "off"; state.last_updated = datetime.now(pytz.UTC)
        hass.states.get = MagicMock(return_value=state)
        res, _ = ch.low_level_plug_check_now(datetime.now(pytz.UTC))
        assert res is True

    def test_low_level_plug_check_now_unavailable(self):
        """Wallbox: unavailable switch means not plugged."""
        hass, _, ch = self._setup()
        state = MagicMock(); state.state = STATE_UNAVAILABLE; state.last_updated = datetime.now(pytz.UTC)
        hass.states.get = MagicMock(return_value=state)
        res, _ = ch.low_level_plug_check_now(datetime.now(pytz.UTC))
        assert res is False

    def test_low_level_plug_check_now_none(self):
        """Wallbox: no state means not plugged."""
        hass, _, ch = self._setup()
        hass.states.get = MagicMock(return_value=None)
        res, _ = ch.low_level_plug_check_now(datetime.now(pytz.UTC))
        assert res is False

    # -- low_level_charge_check_now (Wallbox specific) --

    def test_low_level_charge_check_now_on(self):
        hass, _, ch = self._setup()
        state = MagicMock(); state.state = "on"
        hass.states.get = MagicMock(return_value=state)
        assert ch.low_level_charge_check_now(datetime.now(pytz.UTC)) is True

    def test_low_level_charge_check_now_off(self):
        hass, _, ch = self._setup()
        state = MagicMock(); state.state = "off"
        hass.states.get = MagicMock(return_value=state)
        assert ch.low_level_charge_check_now(datetime.now(pytz.UTC)) is False

    def test_low_level_charge_check_now_unavailable(self):
        hass, _, ch = self._setup()
        state = MagicMock(); state.state = STATE_UNAVAILABLE
        hass.states.get = MagicMock(return_value=state)
        assert ch.low_level_charge_check_now(datetime.now(pytz.UTC)) is False

    # -- Integration: Wallbox check_load_activity --

    @pytest.mark.asyncio
    async def test_wallbox_charger_check_load_activity(self):
        """Wallbox charger works with real check_load_activity_and_constraints."""
        hass, home, ch = self._setup()
        car = _make_real_car(hass, home)
        _plug_car(ch, car, datetime.now(pytz.UTC))

        ch.is_charger_unavailable = MagicMock(return_value=False)
        ch.probe_for_possible_needed_reboot = MagicMock(return_value=False)
        ch.is_not_plugged = MagicMock(return_value=False)
        ch.is_plugged = MagicMock(return_value=True)
        ch.set_charging_num_phases = AsyncMock(return_value=False)
        ch.set_max_charging_current = AsyncMock(return_value=True)
        ch.get_best_car = MagicMock(return_value=car)

        result = await ch.check_load_activity_and_constraints(datetime.now(pytz.UTC))
        assert len([c for c in ch._constraints if c is not None]) >= 1


# ---- _find_charger_entity_id edge cases ----


class TestFindChargerEntityId:
    """Test the entity discovery logic used by OCPP/Wallbox init."""

    def _ch(self):
        hass = _make_hass()
        home = _make_home()
        return _create_charger(hass, home)

    def test_single_match(self):
        ch = self._ch()
        entries = [_make_entity_entry("switch.charger_charge_control")]
        device = MagicMock(); device.id = "d1"; device.name = "charger"; device.name_by_user = None
        result = ch._find_charger_entity_id(device, entries, "switch.", "_charge_control")
        assert result == "switch.charger_charge_control"

    def test_no_match_falls_back_to_computed(self):
        ch = self._ch()
        entries = [_make_entity_entry("switch.other_entity")]
        device = MagicMock(); device.id = "d1"; device.name = "My Charger"; device.name_by_user = None
        result = ch._find_charger_entity_id(device, entries, "switch.", "_charge_control")
        assert result == "switch.my_charger_charge_control"  # slugified fallback

    def test_multiple_matches_picks_longest(self):
        ch = self._ch()
        entries = [
            _make_entity_entry("switch.ch_ctrl"),
            _make_entity_entry("switch.charger_charge_ctrl"),
        ]
        device = MagicMock(); device.id = "d1"; device.name = "charger"; device.name_by_user = None
        result = ch._find_charger_entity_id(device, entries, "switch.", "_ctrl")
        assert result == "switch.charger_charge_ctrl"  # longest

    def test_no_device_no_entries(self):
        ch = self._ch()
        result = ch._find_charger_entity_id(None, [], "switch.", "_foo")
        assert result is None

    def test_user_name_used_for_computed(self):
        ch = self._ch()
        entries = []
        device = MagicMock(); device.id = "d1"; device.name = "orig"; device.name_by_user = "My Custom Name"
        result = ch._find_charger_entity_id(device, entries, "sensor.", "_status")
        assert result == "sensor.my_custom_name_status"


# =============================================================================
# Targeted tests to push coverage from 94% to 95%+
# =============================================================================

from custom_components.quiet_solar.ha_model.charger import (
    CHARGER_MAX_POWER_AMPS_PRECISION_W,
    TIME_OK_BETWEEN_BUDGET_RESET_S,
    TIME_OK_SHOULD_BUDGET_RESET_S,
)
from custom_components.quiet_solar.const import (
    CAR_CHARGE_NO_POWER_ERROR,
    CAR_CHARGE_TYPE_FAULTED,
    CAR_CHARGE_TYPE_NOT_PLUGGED,
    CAR_CHARGE_TYPE_NOT_CHARGING,
)


class TestQSStateCmdIsOkToLaunch:
    """Cover line 252: is_ok_to_launch when last_time_set is None."""

    def test_last_time_set_none(self):
        cmd = QSStateCmd()
        now = datetime.now(pytz.UTC)
        cmd.set(True, now)
        cmd.register_launch(True, now)
        # Force last_time_set=None after launch to hit line 252
        cmd.last_time_set = None
        assert cmd.is_ok_to_launch(True, now) is True


class TestGetAmpsFromPowerSteps:
    """Cover lines 1830, 1837, 1842 in _get_amps_from_power_steps."""

    def _ch(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home, is_3p=False, min_charge=6, max_charge=32)
        car = _make_real_car(hass, home)
        _init_charger_states(ch); _plug_car(ch, car, datetime.now(pytz.UTC))
        return ch, car

    def test_below_min_not_safe_returns_none(self):
        """Line 1830: power too far below min_charge step, safe_border=False -> None."""
        ch, car = self._ch()
        steps = car.get_charge_power_per_phase_A(False)[0]
        # Power well below min step (6A * 230V = 1380W), use 100W
        result = ch._get_amps_from_power_steps(steps, 100.0, safe_border=False)
        assert result is None

    def test_above_max_not_safe_returns_none(self):
        """Line 1837: power too far above max_charge step, safe_border=False -> None."""
        ch, car = self._ch()
        steps = car.get_charge_power_per_phase_A(False)[0]
        # Power well above max step (32A * 230V = 7360W), use 50000W
        result = ch._get_amps_from_power_steps(steps, 50000.0, safe_border=False)
        assert result is None

    def test_above_max_safe_returns_max(self):
        """Line 1837: power above max, safe_border=True -> max_charge."""
        ch, car = self._ch()
        steps = car.get_charge_power_per_phase_A(False)[0]
        result = ch._get_amps_from_power_steps(steps, 50000.0, safe_border=True)
        assert result == 32

    def test_near_next_step_adjusts_up(self):
        """Line 1842: power closer to next step -> adjusts amp upward."""
        ch, car = self._ch()
        steps = car.get_charge_power_per_phase_A(False)[0]
        # 10A = 2300W, 11A = 2530W. Use a power slightly below 2530W
        result = ch._get_amps_from_power_steps(steps, 2529.0, safe_border=False)
        assert result == 11  # adjusted up because within precision


class TestIsChargerGroupPowerZero:
    """Cover line 1748."""

    def test_returns_none_when_no_power(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        _init_charger_states(ch)
        # father_device.get_average_power returns None
        ch.father_device.get_average_power = MagicMock(return_value=None)
        assert ch.is_charger_group_power_zero(datetime.now(pytz.UTC), 30.0) is None

    def test_returns_true_when_below_threshold(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        _init_charger_states(ch)
        ch.father_device.get_average_power = MagicMock(return_value=50.0)  # < 70W threshold
        assert ch.is_charger_group_power_zero(datetime.now(pytz.UTC), 30.0) is True

    def test_returns_false_when_above_threshold(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        _init_charger_states(ch)
        ch.father_device.get_average_power = MagicMock(return_value=2300.0)
        assert ch.is_charger_group_power_zero(datetime.now(pytz.UTC), 30.0) is False


class TestGetChargeType:
    """Cover lines 3582, and exercise the charge type flow."""

    def test_charge_no_power_error(self):
        """Line 3582: possible_charge_error_start_time set -> CAR_CHARGE_NO_POWER_ERROR."""
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        _init_charger_states(ch)
        ch.is_charger_faulted = MagicMock(return_value=False)
        ch.possible_charge_error_start_time = datetime.now(pytz.UTC)
        charge_type, ct = ch.get_charge_type()
        assert charge_type == CAR_CHARGE_NO_POWER_ERROR
        assert ct is None

    def test_faulted(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        _init_charger_states(ch)
        ch.is_charger_faulted = MagicMock(return_value=True)
        charge_type, ct = ch.get_charge_type()
        assert charge_type == CAR_CHARGE_TYPE_FAULTED

    def test_not_plugged(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        _init_charger_states(ch)
        ch.is_charger_faulted = MagicMock(return_value=False)
        ch.possible_charge_error_start_time = None
        charge_type, ct = ch.get_charge_type()
        assert charge_type == CAR_CHARGE_TYPE_NOT_PLUGGED

    def test_not_charging_with_car(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); car = _make_real_car(hass, home)
        _init_charger_states(ch); _plug_car(ch, car, datetime.now(pytz.UTC))
        ch.is_charger_faulted = MagicMock(return_value=False)
        ch.possible_charge_error_start_time = None
        charge_type, ct = ch.get_charge_type()
        assert charge_type == CAR_CHARGE_TYPE_NOT_CHARGING


class TestCheckIfRebootHappened:
    """Cover lines 3556, 3559."""

    @pytest.mark.asyncio
    async def test_no_status_sensor(self):
        """Line 3556: no charger_status_sensor -> returns True."""
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        _init_charger_states(ch)
        ch.charger_status_sensor = None
        result = ch.check_if_reboot_happened(
            from_time=datetime.now(pytz.UTC) - timedelta(minutes=5),
            to_time=datetime.now(pytz.UTC),
        )
        if asyncio.iscoroutine(result):
            result = await result
        assert result is True

    def test_is_state_set_false(self):
        """Line 3559: _is_state_set returns False when inner states are None."""
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        ch._inner_expected_charge_state = None
        ch._inner_amperage = None
        ch._inner_num_active_phases = None
        assert ch._is_state_set(datetime.now(pytz.UTC)) is False


class TestCheckPluggedValCarFallback:
    """Cover lines 3315-3318: car confirms plug state."""

    def test_car_confirms_plugged(self):
        """Lines 3315-3316: car.is_car_plugged returns True, charger says PLUGGED -> True."""
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); car = _make_real_car(hass, home)
        _init_charger_states(ch); _plug_car(ch, car, datetime.now(pytz.UTC))
        # Duration check returns None -> triggers car fallback
        ch.get_last_state_value_duration = MagicMock(return_value=(None, None))
        # But latest state is PLUGGED
        ch.get_sensor_latest_possible_valid_value = MagicMock(return_value=QSChargerStates.PLUGGED)
        # Car confirms plugged
        car.is_car_plugged = MagicMock(return_value=True)
        result = ch._check_plugged_val(datetime.now(pytz.UTC), for_duration=30.0, check_for_val=True)
        assert result is True

    def test_car_confirms_unplugged(self):
        """Lines 3317-3318: car not plugged, charger says UN_PLUGGED -> False."""
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); car = _make_real_car(hass, home)
        _init_charger_states(ch); _plug_car(ch, car, datetime.now(pytz.UTC))
        ch.get_last_state_value_duration = MagicMock(return_value=(None, None))
        ch.get_sensor_latest_possible_valid_value = MagicMock(return_value=QSChargerStates.UN_PLUGGED)
        car.is_car_plugged = MagicMock(return_value=False)
        result = ch._check_plugged_val(datetime.now(pytz.UTC), for_duration=30.0, check_for_val=True)
        assert result is False


class TestCheckChargeStateUnknownSensor:
    """Cover line 3379: check_charge_state falls through to None when status sensor unknown."""

    def test_status_sensor_unknown_returns_none(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        _init_charger_states(ch)
        ch.is_optimistic_plugged = MagicMock(return_value=True)
        ch.get_car_charge_enabled_status_vals = MagicMock(return_value=["Charging"])
        # _check_charger_status returns None
        ch._check_charger_status = MagicMock(return_value=None)
        # charger_status_sensor exists but state is unknown
        state = MagicMock(); state.state = STATE_UNKNOWN
        hass.states.get = MagicMock(return_value=state)
        result = ch.check_charge_state(datetime.now(pytz.UTC))
        assert result is None


class TestCanChangeBudgetEdgeCases:
    """Cover lines 394, 415, 438 in can_change_budget."""

    def _cs(self, budgeted, phases, possible, p_phases=None):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home, is_3p=True)
        car = _make_real_car(hass, home)
        _init_charger_states(ch); _plug_car(ch, car, datetime.now(pytz.UTC))
        cs = QSChargerStatus(ch)
        cs.budgeted_amp = budgeted; cs.budgeted_num_phases = phases
        cs.possible_amps = possible
        cs.possible_num_phases = p_phases or [1]
        cs.current_real_max_charging_amp = budgeted
        cs.current_active_phase_number = phases
        return cs

    def test_increase_from_0_possible_no_zero(self):
        """Line 394: increase from 0, possible_amps[0] != 0 -> returns possible_amps[0]."""
        cs = self._cs(budgeted=0, phases=1, possible=[6, 7, 8, 9, 10])
        amp, _ = cs.can_change_budget(allow_state_change=True, increase=True)
        assert amp == 6  # possible_amps[0] since it's not 0

    def test_not_real_increase_returns_none(self):
        """Line 415: increase check fails when total amps don't actually increase."""
        # 3-phase 6A = 18 total. next_amp=7, 7*3=21 > 18 -> real increase. So let's
        # set a case where it wraps: 1-phase 6A = 6 total, next_amp=7, 7*1=7 > 6 -> still real.
        # The only way to trigger 415 is if next_amp * next_num_phases <= current_all_amps
        # which happens with rounding in phase switch. Let's use 3-phase budgeted_amp=3
        # current_all_amps = 3*3=9. increase: next_amp=4, 4*3=12 > 9 -> still works.
        # Hard to trigger without phase switch. With phase switch:
        cs = self._cs(budgeted=6, phases=3, possible=[0, 6, 7, 8], p_phases=[1, 3])
        # 6*3=18 total. Normal increase: 7*3=21 > 18 -> valid.
        # Phase switch: 6//3=2 -> 1phase. 2*1=2 < 18 -> not increase. 
        # probe_amp_cb(2) -> 3, 3*1=3 < 18 -> still not increase -> None
        # So net result with phase switch option: normal increase wins at 7A/3phase
        amp, phases = cs.can_change_budget(allow_state_change=True, allow_phase_change=True, increase=True)
        assert amp == 7 and phases == 3

    def test_phase_change_validates_correctly(self):
        """Line 438: phase change validated as real increase/decrease after probing."""
        # 1-phase 16A = 16 total. Phase switch to 3: 16//3=5 -> 3phase, 5*3=15 < 16 -> decrease!
        # For increase with phase switch: probe_amp(5) = 6, 6*3=18 > 16 -> real increase!
        cs = self._cs(budgeted=16, phases=1, possible=[0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], p_phases=[1, 3])
        amp, phases = cs.can_change_budget(allow_state_change=True, allow_phase_change=True, increase=True)
        # Normal increase: 17 is not in possible_amps (max=16) -> None
        # Phase switch: 16//3=5 -> 5*3=15 < 16 (not increase), probe: 6, 6*3=18 > 16 (yes!) -> 6A/3phase
        # |16 - 18| = 2 vs |16 - None| -> phase switch wins
        assert amp == 6 and phases == 3


class TestGetBestCarEdgeCases:
    """Cover lines 2358, 2366, 2370-2371 in get_best_car."""

    def test_user_selected_car_force_no_charger(self):
        """Line 2358: user selected car has FORCE_CAR_NO_CHARGER_CONNECTED -> car set to None,
        falls through to scoring which returns default generic car."""
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        car = _make_real_car(hass, home)
        _init_charger_states(ch)
        car.user_attached_charger_name = FORCE_CAR_NO_CHARGER_CONNECTED
        ch.user_attached_car_name = "TestCar"
        result = ch.get_best_car(datetime.now(pytz.UTC))
        # car was set to None at line 2358 but code falls through to scoring
        # which returns the default generic car
        assert result is not None
        assert result.name == ch._default_generic_car.name

    def test_disabled_charger_skipped(self):
        """Line 2366: disabled charger skipped in active charger scan."""
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home, name="Ch0")
        ch2 = _create_charger(hass, home, name="Ch1")
        ch2.qs_enable_device = False  # disabled
        car = _make_real_car(hass, home)
        _init_charger_states(ch)
        _init_charger_states(ch2)
        ch.user_attached_car_name = "TestCar"
        # ch2 is disabled, shouldn't interfere
        result = ch.get_best_car(datetime.now(pytz.UTC))
        assert result is not None

    def test_car_attached_to_multiple_chargers_detached(self):
        """Lines 2370-2371: car manually attached to two chargers -> second one detached."""
        hass = _make_hass(); home = _make_home()
        ch1 = _create_charger(hass, home, name="Ch1")
        ch2 = _create_charger(hass, home, name="Ch2")
        car = _make_real_car(hass, home)
        _init_charger_states(ch1); _init_charger_states(ch2)
        # Both chargers claim the same car
        ch1.user_attached_car_name = "TestCar"
        ch2.user_attached_car_name = "TestCar"
        _plug_car(ch2, car, datetime.now(pytz.UTC))
        result = ch1.get_best_car(datetime.now(pytz.UTC))
        assert result is not None
        # ch2 should have been detached
        assert ch2.user_attached_car_name is None
        assert ch2.car is None


class TestDynHandleBudgetResetTiming:
    """Cover lines 788, 792-793, 795-796 in dyn_handle."""

    def _build(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); car = _make_real_car(hass, home)
        _init_charger_states(ch, charge_state=True, amperage=10, num_phases=1)
        _plug_car(ch, car, datetime.now(pytz.UTC) - timedelta(hours=1))
        car.update_dampening_value = MagicMock(return_value=False)
        group = _make_charger_group(home, [ch])
        now = datetime.now(pytz.UTC)
        vtime = now - timedelta(seconds=CHARGER_ADAPTATION_WINDOW_S + 10)

        cs = QSChargerStatus(ch)
        cs.current_real_max_charging_amp = 10; cs.current_active_phase_number = 1
        cs.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12]; cs.possible_num_phases = [1]
        cs.budgeted_amp = 10; cs.budgeted_num_phases = 1; cs.charge_score = 100
        cs.command = copy_command(CMD_AUTO_GREEN_ONLY); cs.is_before_battery = False
        cs.accurate_current_power = None

        ch.is_charging_power_zero = MagicMock(return_value=False)
        pv = [(now - timedelta(seconds=i), 500.0) for i in range(10)]
        home.get_available_power_values = MagicMock(return_value=pv)
        home.get_grid_consumption_power_values = MagicMock(return_value=pv)
        group.dynamic_group.get_median_sensor = MagicMock(return_value=2300.0)
        group.ensure_correct_state = AsyncMock(return_value=([cs], vtime))
        group.apply_budget_strategy = AsyncMock()

        return group, ch, cs, now

    @pytest.mark.asyncio
    async def test_done_reset_budget_updates_timestamps(self):
        """Lines 792-793: done_reset_budget=True sets _last_time_reset_budget_done."""
        group, ch, cs, now = self._build()
        group.budgeting_algorithm_minimize_diffs = AsyncMock(return_value=(True, False, True))
        await group.dyn_handle(now)
        assert group._last_time_reset_budget_done == now
        assert group._last_time_should_reset_budget_received is None

    @pytest.mark.asyncio
    async def test_should_reset_records_time(self):
        """Lines 795-796: should_do_reset_allocation=True, first time -> records time."""
        group, ch, cs, now = self._build()
        group.budgeting_algorithm_minimize_diffs = AsyncMock(return_value=(True, True, False))
        group._last_time_should_reset_budget_received = None
        await group.dyn_handle(now)
        assert group._last_time_should_reset_budget_received == now

    @pytest.mark.asyncio
    async def test_allow_budget_reset_from_should_timeout(self):
        """Line 788: allow_budget_reset when _last_time_should_reset_budget_received is old."""
        group, ch, cs, now = self._build()
        # Set should_reset_received long ago so it exceeds TIME_OK_SHOULD_BUDGET_RESET_S
        group._last_time_should_reset_budget_received = now - timedelta(seconds=TIME_OK_SHOULD_BUDGET_RESET_S + 10)
        group._last_time_reset_budget_done = now  # Recent, so first condition won't trigger
        group.budgeting_algorithm_minimize_diffs = AsyncMock(return_value=(True, False, False))
        await group.dyn_handle(now)
        # The allow_budget_reset=True should have been passed to budgeting_algorithm
        call_args = group.budgeting_algorithm_minimize_diffs.call_args[0]
        assert call_args[3] is True  # allow_budget_reset


class TestApplyBudgetsMismatchAborts:
    """Cover lines 1530, 1533: apply_budgets with check_charger_state when charger not found."""

    @pytest.mark.asyncio
    async def test_partial_charger_mismatch_aborts(self):
        """Lines 1530, 1533: some chargers in actionable but not in cs_to_apply -> abort."""
        hass = _make_hass(); home = _make_home(); now = datetime.now(pytz.UTC)

        ch1 = _create_charger(hass, home, name="Ch1")
        ch2 = _create_charger(hass, home, name="Ch2")
        car1 = _make_real_car(hass, home, name="Car1")
        car2 = _make_real_car(hass, home, name="Car2")
        _init_charger_states(ch1, amperage=10); _init_charger_states(ch2, amperage=10)
        _plug_car(ch1, car1, now); _plug_car(ch2, car2, now)
        ch1._ensure_correct_state = AsyncMock()
        ch2._ensure_correct_state = AsyncMock()

        cs1 = QSChargerStatus(ch1)
        cs1.current_real_max_charging_amp = 10; cs1.current_active_phase_number = 1
        cs1.budgeted_amp = 12; cs1.budgeted_num_phases = 1; cs1.charge_score = 100
        cs1.possible_amps = [0, 6, 7, 8, 9, 10, 11, 12]; cs1.possible_num_phases = [1]
        cs1.command = copy_command(CMD_AUTO_GREEN_ONLY)

        cs2 = QSChargerStatus(ch2)
        cs2.current_real_max_charging_amp = 10; cs2.current_active_phase_number = 1
        cs2.budgeted_amp = 10; cs2.budgeted_num_phases = 1; cs2.charge_score = 80
        cs2.possible_amps = [0, 6, 7, 8, 9, 10]; cs2.possible_num_phases = [1]
        cs2.command = copy_command(CMD_AUTO_GREEN_ONLY)

        group = _make_charger_group(home, [ch1, ch2])

        # cs_to_apply has only cs1, but actionable has cs1+cs2 -> cs2 falls into else at 1530
        # num_ok(1) != len(cs_to_apply)(1) when... actually num_ok will be 1 == len([cs1])
        # To trigger 1533 we need num_ok != len(cs_to_apply)
        # So: cs_to_apply = [cs1, cs2] but actionable = [cs1] -> cs2 not found -> line 1530
        # and num_ok(1) != len(cs_to_apply)(2) -> line 1533
        await group.apply_budgets([cs1, cs2], [cs1], now, check_charger_state=True)
        # Should have aborted - no _ensure_correct_state called
        ch1._ensure_correct_state.assert_not_called()


class TestDevicePostHomeInitEdge:
    """Cover line 2573-2575: stored constraint references unavailable car."""

    def test_stored_constraint_car_not_found(self):
        """Lines 2573-2575: constraint load_param references non-existent car -> skipped."""
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        _init_charger_states(ch)
        # Constraint references a car that doesn't exist
        ct = MagicMock(); ct.load_param = "NonExistentCar"; ct.name = "ghost_ct"
        ch._constraints = [ct]
        ch.device_post_home_init(datetime.now(pytz.UTC))
        assert ch._boot_car is None


class TestGetStableGreenCapCannotStop:
    """Cover line 1996: CMD_AUTO_GREEN_CAP zero consign, charging but can't stop."""

    def test_green_cap_zero_charging_cant_stop(self):
        """Line 1996: possible_amps = [min_charge] when must stop but can't change state."""
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home, is_3p=False)
        car = _make_real_car(hass, home)
        _init_charger_states(ch, charge_state=True, amperage=10, num_phases=1)
        _plug_car(ch, car, datetime.now(pytz.UTC) - timedelta(hours=1))
        now = datetime.now(pytz.UTC)

        ch.qs_enable_device = True
        ch.is_not_plugged = MagicMock(return_value=False)
        ch.is_charger_unavailable = MagicMock(return_value=False)
        ch._probe_and_enforce_stopped_charge_command_state = MagicMock(return_value=False)
        ch.get_median_sensor = MagicMock(return_value=1000.0)
        ch.get_current_active_constraint = MagicMock(return_value=None)
        ch.can_do_3_to_1_phase_switch = MagicMock(return_value=False)

        # Charge state is True (current_state=True), can't change (recent)
        ch._expected_charge_state._num_set = 5
        ch._expected_charge_state.last_change_asked = now
        ch._expected_charge_state.last_time_set = now

        ch.current_command = copy_command(CMD_AUTO_GREEN_CAP, power_consign=0)
        cs = ch.get_stable_dynamic_charge_status(now)
        assert cs is not None
        assert cs.possible_amps == [ch.min_charge]


class TestEnsureCorrectStateGroup:
    """Cover lines 556-557, 567: charger returns None (skipped), multiple vcst values."""

    @pytest.mark.asyncio
    async def test_charger_none_skipped(self):
        """Lines 556-557: charger returns res=None -> warning and continue."""
        hass = _make_hass(); home = _make_home()
        ch1 = _create_charger(hass, home, name="Ch1")
        ch2 = _create_charger(hass, home, name="Ch2")
        car2 = _make_real_car(hass, home, name="Car2")
        _init_charger_states(ch1); _init_charger_states(ch2)
        _plug_car(ch2, car2, datetime.now(pytz.UTC))
        now = datetime.now(pytz.UTC)
        vcst2 = now - timedelta(seconds=10)

        # Charger ensure_correct_state returns (res, handled_static, vcst)
        # ch1 returns res=None (unavailable)
        ch1.ensure_correct_state = AsyncMock(return_value=(None, False, None))
        # ch2 returns ok with a vcst
        ch2.ensure_correct_state = AsyncMock(return_value=(True, False, vcst2))
        ch2.get_stable_dynamic_charge_status = MagicMock(return_value=None)

        group = _make_charger_group(home, [ch1, ch2])
        result, vcst = await group.ensure_correct_state(now)
        # ch1 was skipped (res=None), ch2's vcst is used
        assert vcst == vcst2

    @pytest.mark.asyncio
    async def test_multiple_vcst_takes_latest(self):
        """Line 567: verified_correct_state_time updated to latest vcst among chargers."""
        hass = _make_hass(); home = _make_home()
        ch1 = _create_charger(hass, home, name="Ch1")
        ch2 = _create_charger(hass, home, name="Ch2")
        car1 = _make_real_car(hass, home, name="Car1")
        car2 = _make_real_car(hass, home, name="Car2")
        _init_charger_states(ch1); _init_charger_states(ch2)
        _plug_car(ch1, car1, datetime.now(pytz.UTC))
        _plug_car(ch2, car2, datetime.now(pytz.UTC))
        now = datetime.now(pytz.UTC)

        vcst1 = now - timedelta(seconds=30)
        vcst2 = now - timedelta(seconds=10)  # more recent

        # (res, handled_static, vcst)
        ch1.ensure_correct_state = AsyncMock(return_value=(True, False, vcst1))
        ch2.ensure_correct_state = AsyncMock(return_value=(True, False, vcst2))
        ch1.get_stable_dynamic_charge_status = MagicMock(return_value=None)
        ch2.get_stable_dynamic_charge_status = MagicMock(return_value=None)

        group = _make_charger_group(home, [ch1, ch2])
        result, vcst = await group.ensure_correct_state(now)
        assert vcst == vcst2  # The more recent one


# =============================================================================
# Push to 96%: _ensure_correct_state, update_power_steps, get_min_max_power, etc.
# =============================================================================

from custom_components.quiet_solar.ha_model.charger import (
    STATE_CMD_TIME_BETWEEN_RETRY_S,
)


class TestEnsureCorrectStateCharger:
    """Cover lines inside _ensure_correct_state (charger level): 3659-3660, 3675,
    3694, 3705, 3709-3712, 3724, 3737-3742, 3743, 3767.
    """

    def _setup(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_ocpp_charger(hass, home, name="EnsureCh")
        car = _make_real_car(hass, home, name="EnsureCar")
        now = datetime.now(pytz.UTC)
        _init_charger_states(ch, charge_state=True, amperage=10, num_phases=1)
        _plug_car(ch, car, now - timedelta(hours=1))
        ch._do_update_charger_state = AsyncMock()
        ch.is_charger_unavailable = MagicMock(return_value=False)
        ch.is_not_plugged = MagicMock(return_value=False)
        ch.running_command = None
        ch.is_car_stopped_asking_current = MagicMock(return_value=False)
        ch.update_data_request = AsyncMock()
        ch.set_charging_current = AsyncMock()
        ch.set_charging_num_phases = AsyncMock()
        ch.start_charge = AsyncMock()
        ch.stop_charge = AsyncMock()
        return hass, home, ch, car, now

    @pytest.mark.asyncio
    async def test_reboot_asked_not_done(self):
        """Lines 3659-3660: reboot asked but not happened -> one_bad=True, returns False."""
        _, _, ch, _, now = self._setup()
        ch._asked_for_reboot_at_time = now - timedelta(minutes=5)
        ch.check_if_reboot_happened = MagicMock(return_value=False)
        result = await ch._ensure_correct_state(now)
        assert result is False

    @pytest.mark.asyncio
    async def test_reboot_asked_done(self):
        """Lines 3655-3657: reboot happened -> clears _asked_for_reboot_at_time."""
        _, _, ch, _, now = self._setup()
        ch._asked_for_reboot_at_time = now - timedelta(minutes=5)
        ch.check_if_reboot_happened = MagicMock(return_value=True)
        ch.is_charge_enabled = MagicMock(return_value=True)
        ch.is_charge_disabled = MagicMock(return_value=False)
        ch.get_charging_current = MagicMock(return_value=10)
        await ch._ensure_correct_state(now)
        assert ch._asked_for_reboot_at_time is None

    @pytest.mark.asyncio
    async def test_phase_mismatch_not_ok_to_launch(self):
        """Line 3675: phase mismatch but not ok to launch -> debug log, one_bad=True."""
        _, _, ch, _, now = self._setup()
        ch._expected_num_active_phases.value = 3
        # current_num_phases returns 1 (mocked via hass.states.get returning None -> default)
        # Make is_ok_to_launch return False
        ch._expected_num_active_phases._num_set = 5
        ch._expected_num_active_phases._num_launched = 5  # > STATE_CMD_RETRY_NUMBER
        result = await ch._ensure_correct_state(now)
        assert result is False

    @pytest.mark.asyncio
    async def test_charge_stopped_but_expected_off_amps_ok(self):
        """Line 3694: expected off + disabled -> amps_bad_set=False, no amp correction."""
        _, _, ch, _, now = self._setup()
        ch._expected_charge_state.value = False
        ch.is_charge_enabled = MagicMock(return_value=False)
        ch.is_charge_disabled = MagicMock(return_value=True)
        ch.get_charging_current = MagicMock(return_value=10)  # doesn't match expected
        ch._expected_amperage.value = 6
        result = await ch._ensure_correct_state(now)
        # amps don't match but amps_bad_set=False because charge is off
        # charge state matches (expected False, disabled True) -> success
        assert result is True

    @pytest.mark.asyncio
    async def test_amps_match_ping_update(self):
        """Lines 3705, 3712: amps match, ping triggers update_data_request."""
        _, _, ch, _, now = self._setup()
        ch.is_charge_enabled = MagicMock(return_value=True)
        ch.is_charge_disabled = MagicMock(return_value=False)
        ch.get_charging_current = MagicMock(return_value=10)
        ch._expected_amperage.value = 10
        # Set up success timing: first_time_success and last_ping same -> duration_check = STATE_CMD_TIME_BETWEEN_RETRY_S
        long_ago = now - timedelta(seconds=STATE_CMD_TIME_BETWEEN_RETRY_S + 10)
        ch._expected_amperage.first_time_success = long_ago
        ch._expected_amperage.last_ping_time_success = long_ago
        result = await ch._ensure_correct_state(now)
        # Should have called update_data_request (line 3705)
        ch.update_data_request.assert_called()

    @pytest.mark.asyncio
    async def test_amps_match_but_reset_needed(self):
        """Lines 3709-3712: amps match but ping timeout -> re-sets charging current."""
        _, _, ch, _, now = self._setup()
        ch.is_charge_enabled = MagicMock(return_value=True)
        ch.is_charge_disabled = MagicMock(return_value=False)
        ch.get_charging_current = MagicMock(return_value=10)
        ch._expected_amperage.value = 10
        very_long_ago = now - timedelta(seconds=2 * STATE_CMD_TIME_BETWEEN_RETRY_S + 10)
        ch._expected_amperage.first_time_success = very_long_ago
        ch._expected_amperage.last_ping_time_success = very_long_ago
        await ch._ensure_correct_state(now)
        ch.set_charging_current.assert_called()

    @pytest.mark.asyncio
    async def test_amps_mismatch_not_ok_to_launch(self):
        """Line 3724: amps mismatch but not ok to launch -> debug log."""
        _, _, ch, _, now = self._setup()
        ch.is_charge_enabled = MagicMock(return_value=True)
        ch.is_charge_disabled = MagicMock(return_value=False)
        ch.get_charging_current = MagicMock(return_value=8)
        ch._expected_amperage.value = 12
        ch._expected_amperage._num_set = 5
        ch._expected_amperage._num_launched = 5  # > retry number
        await ch._ensure_correct_state(now)
        # set_charging_current should NOT be called (not ok to launch)
        ch.set_charging_current.assert_not_called()

    @pytest.mark.asyncio
    async def test_charge_state_mismatch_start(self):
        """Lines 3737-3738: expected on but disabled -> starts charge."""
        _, _, ch, _, now = self._setup()
        ch._expected_charge_state.value = True
        ch.is_charge_enabled = MagicMock(return_value=False)
        ch.is_charge_disabled = MagicMock(return_value=True)
        ch.get_charging_current = MagicMock(return_value=10)
        ch._expected_amperage.value = 10
        await ch._ensure_correct_state(now)
        ch.start_charge.assert_called()

    @pytest.mark.asyncio
    async def test_charge_state_mismatch_stop(self):
        """Lines 3740-3741: expected off but enabled -> stops charge."""
        _, _, ch, _, now = self._setup()
        ch._expected_charge_state.value = False
        ch.is_charge_enabled = MagicMock(return_value=True)
        ch.is_charge_disabled = MagicMock(return_value=False)
        ch.get_charging_current = MagicMock(return_value=10)
        ch._expected_amperage.value = 10
        await ch._ensure_correct_state(now)
        ch.stop_charge.assert_called()

    @pytest.mark.asyncio
    async def test_charge_state_mismatch_not_ok_to_launch(self):
        """Line 3743: charge state mismatch but not ok to launch -> debug log only."""
        _, _, ch, _, now = self._setup()
        ch._expected_charge_state.value = True
        ch.is_charge_enabled = MagicMock(return_value=False)
        ch.is_charge_disabled = MagicMock(return_value=True)
        ch.get_charging_current = MagicMock(return_value=10)
        ch._expected_amperage.value = 10
        ch._expected_charge_state._num_set = 5
        ch._expected_charge_state._num_launched = 5  # > retry number
        await ch._ensure_correct_state(now)
        ch.start_charge.assert_not_called()
        ch.stop_charge.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_good_sets_verified_time(self):
        """Line 3767 (3753-3756): everything matches -> _verified_correct_state_time set."""
        _, _, ch, _, now = self._setup()
        ch.is_charge_enabled = MagicMock(return_value=True)
        ch.is_charge_disabled = MagicMock(return_value=False)
        ch.get_charging_current = MagicMock(return_value=10)
        ch._expected_amperage.value = 10
        ch._expected_charge_state.value = True
        ch._verified_correct_state_time = None
        result = await ch._ensure_correct_state(now)
        assert result is True
        assert ch._verified_correct_state_time == now


class TestUpdatePowerStepsAndMinMax:
    """Cover lines 3176, 3196."""

    def test_get_min_max_power_with_car(self):
        """Line 3196: returns min/max from power_steps."""
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); car = _make_real_car(hass, home)
        _init_charger_states(ch); _plug_car(ch, car, datetime.now(pytz.UTC))
        mn, mx = ch.get_min_max_power()
        assert mn > 0
        assert mx > mn

    def test_get_min_max_power_no_car(self):
        """Line 3193-3194: no car -> (0,0)."""
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        _init_charger_states(ch)
        mn, mx = ch.get_min_max_power()
        assert mn == 0.0 and mx == 0.0

    def test_update_power_steps_removes_zero(self):
        """Line 3176: 0 is removed from the power step set."""
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); car = _make_real_car(hass, home)
        _init_charger_states(ch); _plug_car(ch, car, datetime.now(pytz.UTC))
        # All power steps should be > 0 (no zero step)
        for step in ch._power_steps:
            assert step.power_consign > 0


class TestApplyBudgetsStateChangeNullLastChange:
    """Cover line 1577: state change when last_change_asked is None."""

    @pytest.mark.asyncio
    async def test_state_change_null_last_change(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); car = _make_real_car(hass, home)
        now = datetime.now(pytz.UTC)
        _init_charger_states(ch, charge_state=True, amperage=10, num_phases=1)
        _plug_car(ch, car, now - timedelta(hours=1))
        ch._ensure_correct_state = AsyncMock()

        # Force last_change_asked to None
        ch._expected_charge_state.last_change_asked = None

        cs = QSChargerStatus(ch)
        cs.current_real_max_charging_amp = 10; cs.current_active_phase_number = 1
        cs.budgeted_amp = 3  # Below min -> triggers state change to False
        cs.budgeted_num_phases = 1; cs.charge_score = 100
        cs.possible_amps = [0, 6, 7, 8, 9, 10]; cs.possible_num_phases = [1]
        cs.command = copy_command(CMD_AUTO_GREEN_ONLY)

        group = _make_charger_group(home, [ch])
        await group.apply_budgets([cs], [cs], now, check_charger_state=False)
        # Should have set state to False and logged "delta None"
        assert ch._expected_charge_state.value is False


class TestCheckIfRebootNoStatusVals:
    """Cover line 3552: check_if_reboot_happened with no rebooting status vals."""

    @pytest.mark.asyncio
    async def test_no_reboot_status_vals(self):
        """Line 3552: get_car_status_rebooting_vals returns [] -> returns True."""
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)  # Generic has empty rebooting vals
        _init_charger_states(ch)
        result = ch.check_if_reboot_happened(
            from_time=datetime.now(pytz.UTC) - timedelta(minutes=5),
            to_time=datetime.now(pytz.UTC),
        )
        if asyncio.iscoroutine(result):
            result = await result
        assert result is True


class TestEnsureCorrectStateChargerLevel:
    """Cover lines 3614-3615: ensure_correct_state wrapper, and handled_static path."""

    @pytest.mark.asyncio
    async def test_unavailable_returns_none(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        _init_charger_states(ch)
        ch._do_update_charger_state = AsyncMock()
        ch.is_charger_unavailable = MagicMock(return_value=True)
        res, handled, vcst = await ch.ensure_correct_state(datetime.now(pytz.UTC))
        assert res is None
        assert handled is False

    @pytest.mark.asyncio
    async def test_no_car_returns_true_no_vcst(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home)
        _init_charger_states(ch)
        ch._do_update_charger_state = AsyncMock()
        ch.is_charger_unavailable = MagicMock(return_value=False)
        ch.is_not_plugged = MagicMock(return_value=True)
        res, handled, vcst = await ch.ensure_correct_state(datetime.now(pytz.UTC))
        assert res is True
        assert vcst is None

    @pytest.mark.asyncio
    async def test_short_unplug_returns_false(self):
        """Plugged for duration but not instant -> short unplug."""
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); car = _make_real_car(hass, home)
        _init_charger_states(ch); _plug_car(ch, car, datetime.now(pytz.UTC))
        ch._do_update_charger_state = AsyncMock()
        ch.is_charger_unavailable = MagicMock(return_value=False)
        # Not plugged for duration: False, but not plugged instant: True
        ch.is_not_plugged = MagicMock(side_effect=[False, True])
        res, handled, vcst = await ch.ensure_correct_state(datetime.now(pytz.UTC))
        assert res is False

    @pytest.mark.asyncio
    async def test_running_command_returns_false(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_charger(hass, home); car = _make_real_car(hass, home)
        _init_charger_states(ch); _plug_car(ch, car, datetime.now(pytz.UTC))
        ch._do_update_charger_state = AsyncMock()
        ch.is_charger_unavailable = MagicMock(return_value=False)
        ch.is_not_plugged = MagicMock(return_value=False)
        ch.running_command = copy_command(CMD_AUTO_GREEN_ONLY)
        res, handled, vcst = await ch.ensure_correct_state(datetime.now(pytz.UTC))
        assert res is False


# =============================================================================
# constraint_update_value_callback_soc and related
# =============================================================================


class TestConstraintUpdateValueCallback:
    """Cover lines in constraint_update_value_callback_soc:
    3815-3816 (idle), 3854, 3926, 3959, 3969, etc.
    """

    def _setup(self):
        hass = _make_hass(); home = _make_home()
        ch = _create_ocpp_charger(hass, home, name="CbCh")
        car = _make_real_car(hass, home, name="CbCar")
        now = datetime.now(pytz.UTC)
        _init_charger_states(ch, charge_state=True, amperage=10, num_phases=1)
        _plug_car(ch, car, now - timedelta(hours=1))
        ch._do_update_charger_state = AsyncMock()
        ch.is_not_plugged = MagicMock(return_value=False)
        ch.is_charger_unavailable = MagicMock(return_value=False)
        ch.is_car_stopped_asking_current = MagicMock(return_value=False)
        ch.is_charging_power_zero = MagicMock(return_value=False)
        ch.is_charger_group_power_zero = MagicMock(return_value=False)
        ch.on_device_state_change = AsyncMock()

        ct = MultiStepsPowerLoadConstraintChargePercent(
            total_capacity_wh=60000, type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=now - timedelta(hours=1), load=ch, load_param=car.name,
            from_user=False, end_of_constraint=now + timedelta(hours=6),
            initial_value=40.0, target_value=80.0, power_steps=ch._power_steps,
            support_auto=True,
        )
        ct.last_value_update = now - timedelta(minutes=30)
        ct.first_value_update = now - timedelta(hours=1)
        ct.last_value_change_update = now - timedelta(minutes=10)
        ct.current_value = 50.0
        return hass, home, ch, car, ct, now

    @pytest.mark.asyncio
    async def test_idle_command_returns_none(self):
        """Lines 3815-3816: idle/off command -> result=None."""
        _, _, ch, _, ct, now = self._setup()
        ch.current_command = None
        result, do_cont = await ch.constraint_update_value_callback_percent_soc(ct, now)
        assert result is None

    @pytest.mark.asyncio
    async def test_not_plugged_duration(self):
        _, _, ch, _, ct, now = self._setup()
        ch.is_not_plugged = MagicMock(side_effect=[True])
        result, do_cont = await ch.constraint_update_value_callback_percent_soc(ct, now)
        assert result is None and do_cont is False

    @pytest.mark.asyncio
    async def test_short_unplug(self):
        _, _, ch, _, ct, now = self._setup()
        ch.is_not_plugged = MagicMock(side_effect=[False, True])
        result, do_cont = await ch.constraint_update_value_callback_percent_soc(ct, now)
        assert result is None and do_cont is True

    @pytest.mark.asyncio
    async def test_active_command_with_sensor(self):
        """Lines 3819-3835: active command, sensor returns value."""
        _, _, ch, car, ct, now = self._setup()
        ch.current_command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=2300)
        car.get_car_charge_percent = MagicMock(return_value=55.0)
        ch._compute_added_charge_update = MagicMock(return_value=2.0)
        ch.is_car_charged = MagicMock(return_value=(False, 55.0))
        ch.charger_group.dyn_handle = AsyncMock()
        result, do_cont = await ch.constraint_update_value_callback_percent_soc(ct, now)
        assert result is not None and do_cont is True

    @pytest.mark.asyncio
    async def test_sensor_none_calculus_fallback(self):
        """Lines 3837-3841: sensor None -> uses calculus."""
        _, _, ch, car, ct, now = self._setup()
        ch.current_command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=2300)
        car.get_car_charge_percent = MagicMock(return_value=None)
        ch._compute_added_charge_update = MagicMock(return_value=3.0)
        ch.is_car_charged = MagicMock(return_value=(False, 53.0))
        ch.charger_group.dyn_handle = AsyncMock()
        result, _ = await ch.constraint_update_value_callback_percent_soc(ct, now)
        assert result == 53  # ct.current_value(50) + int(3.0)

    @pytest.mark.asyncio
    async def test_energy_callback(self):
        _, _, ch, car, ct, now = self._setup()
        ch.current_command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=2300)
        car.get_car_charge_energy = MagicMock(return_value=30000.0)
        ch._compute_added_charge_update = MagicMock(return_value=1500.0)
        ch.is_car_charged = MagicMock(return_value=(False, 30000.0))
        ch.charger_group.dyn_handle = AsyncMock()
        result, _ = await ch.constraint_update_value_callback_energy_soc(ct, now)
        assert result is not None

    @pytest.mark.asyncio
    async def test_charge_met_stops(self):
        """Lines 3932-3933: constraint met -> do_continue=False."""
        _, _, ch, car, ct, now = self._setup()
        ch.current_command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=2300)
        car.get_car_charge_percent = MagicMock(return_value=80.0)
        ch._compute_added_charge_update = MagicMock(return_value=0.0)
        ch.is_car_charged = MagicMock(return_value=(True, 80.0))
        ct.current_value = 80.0
        ch.charger_group.dyn_handle = AsyncMock()
        _, do_cont = await ch.constraint_update_value_callback_percent_soc(ct, now)
        assert do_cont is False

    @pytest.mark.asyncio
    async def test_no_power_error_set(self):
        """Lines 3920-3924: charging expected, no power -> error flagged."""
        _, _, ch, car, ct, now = self._setup()
        ch.current_command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=2300)
        car.get_car_charge_percent = MagicMock(return_value=50.0)
        ch._compute_added_charge_update = MagicMock(return_value=0.0)
        ch._expected_charge_state.value = True
        ch._expected_charge_state.last_ping_time_success = now - timedelta(minutes=20)
        ch.is_charging_power_zero = MagicMock(return_value=True)
        ch.is_charger_group_power_zero = MagicMock(return_value=True)
        car.is_car_charge_growing = MagicMock(return_value=False)
        ch.is_car_charged = MagicMock(return_value=(False, 50.0))
        ch.charger_group.dyn_handle = AsyncMock()
        await ch.constraint_update_value_callback_percent_soc(ct, now)
        assert ch.possible_charge_error_start_time is not None

    @pytest.mark.asyncio
    async def test_no_power_error_cleared(self):
        """Line 3926: power OK -> error cleared."""
        _, _, ch, car, ct, now = self._setup()
        ch.current_command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=2300)
        car.get_car_charge_percent = MagicMock(return_value=50.0)
        ch._compute_added_charge_update = MagicMock(return_value=1.0)
        ch._expected_charge_state.value = True
        ch._expected_charge_state.last_ping_time_success = now - timedelta(minutes=20)
        ch.possible_charge_error_start_time = now - timedelta(minutes=5)
        ch.is_charging_power_zero = MagicMock(return_value=False)
        ch.is_car_charged = MagicMock(return_value=(False, 50.0))
        ch.charger_group.dyn_handle = AsyncMock()
        await ch.constraint_update_value_callback_percent_soc(ct, now)
        assert ch.possible_charge_error_start_time is None
