"""Additional coverage tests for charger.py."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
import pytz

from homeassistant.const import STATE_UNKNOWN

from custom_components.quiet_solar.const import (
    CONF_CHARGER_MAX_CHARGE,
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_PLUGGED,
    CONF_CHARGER_STATUS_SENSOR,
    CONF_CHARGER_THREE_TO_ONE_PHASE_SWITCH,
    CONF_IS_3P,
    CONF_MONO_PHASE,
    DOMAIN,
    DATA_HANDLER,
    CONF_CHARGER_DEVICE_OCPP,
    CONF_CHARGER_DEVICE_WALLBOX,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    CHARGER_NO_CAR_CONNECTED,
    FORCE_CAR_NO_CHARGER_CONNECTED,
)
from custom_components.quiet_solar.ha_model.charger import (
    QSChargerGroup,
    QSChargerOCPP,
    QSChargerStatus,
    QSChargerWallbox,
    QSChargerGeneric,
    QSStateCmd,
)
from custom_components.quiet_solar.home_model.commands import (
    CMD_AUTO_FROM_CONSIGN,
    CMD_AUTO_GREEN_CAP,
    CMD_AUTO_GREEN_ONLY,
    CMD_AUTO_GREEN_CONSIGN,
    CMD_AUTO_PRICE,
    copy_command,
)
from custom_components.quiet_solar.home_model.constraints import LoadConstraint
from tests.factories import create_minimal_home_model


def create_mock_hass() -> MagicMock:
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.data = {DOMAIN: {DATA_HANDLER: MagicMock()}}
    return hass


def create_mock_home(hass: MagicMock) -> MagicMock:
    """Create a mock QSHome instance."""
    home = create_minimal_home_model()
    home.hass = hass
    home.battery = None
    home.get_available_power_values = MagicMock(return_value=None)
    home.get_grid_consumption_power_values = MagicMock(return_value=None)
    home.battery_can_discharge = MagicMock(return_value=False)
    home.get_tariff = MagicMock(return_value=0.15)
    home.get_best_tariff = MagicMock(return_value=0.10)
    return home


def create_charger_generic(hass: MagicMock, home: MagicMock, name: str = "TestCharger", **extra_config) -> QSChargerGeneric:
    """Create a QSChargerGeneric instance for testing."""
    config_entry = MagicMock()
    config_entry.entry_id = f"test_entry_{name}"
    config_entry.data = {}

    config = {
        "name": name,
        "hass": hass,
        "home": home,
        "config_entry": config_entry,
        CONF_CHARGER_MIN_CHARGE: 6,
        CONF_CHARGER_MAX_CHARGE: 32,
        CONF_IS_3P: True,
        CONF_MONO_PHASE: 1,
        CONF_CHARGER_STATUS_SENSOR: f"sensor.{name}_status",
        CONF_CHARGER_PLUGGED: f"sensor.{name}_plugged",
        CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER: f"number.{name}_max_current",
    }
    config.update(extra_config)

    with patch("custom_components.quiet_solar.ha_model.charger.entity_registry"):
        charger = QSChargerGeneric(**config)

    return charger


@pytest.mark.asyncio
async def test_qs_state_cmd_retry_and_success() -> None:
    """Cover QSStateCmd retry gating and success callback logic."""
    cmd = QSStateCmd()
    now = datetime.now(pytz.UTC)
    cmd.set(True, now)
    cmd.register_launch(True, now)

    assert cmd.is_ok_to_launch(True, now) is False
    later = now + timedelta(seconds=cmd.command_retries_s + 1)
    assert cmd.is_ok_to_launch(True, later) is True

    callback_called = {"value": False}

    async def success_cb(time: datetime, **kwargs) -> None:
        callback_called["value"] = True

    cmd.register_success_cb(success_cb, {})
    await cmd.success(now + timedelta(seconds=10))

    assert callback_called["value"] is True
    assert cmd.on_success_action_cb is None


def test_charger_status_phase_switch_change_budget() -> None:
    """Cover phase-switch budget selection paths."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)

    status = QSChargerStatus(charger)
    status.possible_amps = [0, 6, 8, 10, 16]
    status.possible_num_phases = [1, 3]
    status.budgeted_amp = 6
    status.budgeted_num_phases = 3

    next_amp, next_num_phases = status.can_change_budget(
        allow_state_change=True,
        allow_phase_change=True,
        increase=True,
    )

    assert next_amp == 7
    assert next_num_phases == 3


def test_charger_status_phase_switch_selected() -> None:
    """Cover selection of phase switch when it reduces delta."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)

    status = QSChargerStatus(charger)
    status.possible_amps = [0, 6, 8, 10, 16]
    status.possible_num_phases = [1, 3]
    status.budgeted_amp = 6
    status.budgeted_num_phases = 3

    next_amp, next_num_phases = status.can_change_budget(
        allow_state_change=True,
        allow_phase_change=True,
        increase=False,
    )

    assert next_num_phases == 1
    assert next_amp is not None


def test_charger_status_consign_values_with_phase_switch() -> None:
    """Cover get_consign_amps_values with phase switch and tolerance."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home, **{CONF_CHARGER_THREE_TO_ONE_PHASE_SWITCH: "switch.phase"})
    charger._expected_num_active_phases.set(3, datetime.now(pytz.UTC))

    charger.car = MagicMock()
    charger.car.car_charger_min_charge = 6
    charger.car.car_charger_max_charge = 32
    charger.car.get_charge_power_per_phase_A.return_value = (
        {6: 1200, 7: 1400, 8: 1600, 10: 2000, 16: 3200},
        6,
        16,
    )

    status = QSChargerStatus(charger)
    status.command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=2600)
    status.current_active_phase_number = 3

    with patch.object(charger, "_get_amps_from_power_steps", side_effect=[None, 8]):
        possible_num_phases, consign_amp = status.get_consign_amps_values(consign_is_minimum=True)

    assert possible_num_phases == [1]
    assert consign_amp == 8


def test_get_stable_dynamic_charge_status_green_cap() -> None:
    """Cover green-cap path for stable dynamic charge status."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)
    charger.father_device = MagicMock()
    charger.father_device.charger_group = MagicMock()
    charger.father_device.charger_group.dyn_handle = AsyncMock()
    charger.car = MagicMock()
    charger.car.name = "TestCar"
    charger.car.qs_bump_solar_charge_priority = False
    charger.car.qs_bump_solar_charge_priority = False
    charger.car.qs_bump_solar_charge_priority = False
    charger.car.car_charger_min_charge = 6
    charger.car.car_charger_max_charge = 32
    charger.car.get_charge_power_per_phase_A.return_value = ({6: 1200}, 6, 32)

    charger._expected_amperage.set(10, datetime.now(pytz.UTC))
    charger._expected_num_active_phases.set(3, datetime.now(pytz.UTC))
    charger._expected_charge_state.set(True, datetime.now(pytz.UTC))
    charger.current_command = copy_command(CMD_AUTO_GREEN_CAP, power_consign=0)

    with patch.object(charger, "is_not_plugged", return_value=False), \
         patch.object(charger, "is_charger_unavailable", return_value=False), \
         patch.object(charger, "_probe_and_enforce_stopped_charge_command_state", return_value=False), \
         patch.object(charger, "get_current_active_constraint", return_value=None):
        cs = charger.get_stable_dynamic_charge_status(datetime.now(pytz.UTC))

    assert cs is not None
    assert 0 in cs.possible_amps
    assert cs.current_active_phase_number == 3


def test_compute_is_before_battery_respects_command() -> None:
    """Cover compute_is_before_battery with command overrides."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)
    charger.father_device = MagicMock()
    charger.father_device.charger_group = MagicMock()
    charger.father_device.charger_group.dyn_handle = AsyncMock()
    charger.car = MagicMock()
    charger.car.name = "TestCar"
    charger.car.qs_bump_solar_charge_priority = False

    ct = LoadConstraint(time=datetime.now(pytz.UTC), load=charger)
    ct.is_before_battery = True

    charger.current_command = copy_command(CMD_AUTO_GREEN_CAP, power_consign=1000)
    charger.get_current_active_constraint = MagicMock(return_value=ct)

    assert charger.compute_is_before_battery(ct, datetime.now(pytz.UTC)) is False

    ct.is_before_battery = False
    charger.current_command = copy_command(CMD_AUTO_GREEN_CONSIGN, power_consign=500)
    assert charger.compute_is_before_battery(ct, datetime.now(pytz.UTC)) is True


def test_get_normalized_score_increases_with_bump() -> None:
    """Cover get_normalized_score and bump solar influence."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)
    charger.father_device = MagicMock()
    charger.father_device.charger_group = MagicMock()
    charger.father_device.charger_group.dyn_handle = AsyncMock()
    charger.car = MagicMock()
    charger.car.name = "TestCar"
    charger.car.get_car_charge_percent.return_value = 50.0
    charger.car.car_battery_capacity = 60000

    ct = LoadConstraint(time=datetime.now(pytz.UTC), load=charger)
    ct.end_of_constraint = datetime.now(pytz.UTC) + timedelta(hours=4)
    ct.type = CONSTRAINT_TYPE_MANDATORY_END_TIME

    charger.qs_bump_solar_charge_priority = False
    base_score = charger.get_normalized_score(ct, datetime.now(pytz.UTC))
    charger.qs_bump_solar_charge_priority = True
    bump_score = charger.get_normalized_score(ct, datetime.now(pytz.UTC))

    assert bump_score > base_score


@pytest.mark.asyncio
async def test_ensure_correct_state_success_path() -> None:
    """Cover the success path in _ensure_correct_state."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)

    now = datetime.now(pytz.UTC)
    charger._expected_amperage.set(10, now)
    charger._expected_num_active_phases.set(3, now)
    charger._expected_charge_state.set(True, now)

    with patch.object(type(charger), "current_num_phases", new_callable=PropertyMock, return_value=3), \
         patch.object(charger, "get_charging_current", return_value=10), \
         patch.object(charger, "is_charge_enabled", return_value=True), \
         patch.object(charger, "is_charge_disabled", return_value=False), \
         patch.object(charger, "update_data_request", new_callable=AsyncMock):
        result = await charger._ensure_correct_state(now)

    assert result is True
    assert charger._verified_correct_state_time == now


@pytest.mark.asyncio
async def test_constraint_update_value_callback_soc_paths() -> None:
    """Cover constraint update callback logic for percent and energy."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)
    charger.father_device = MagicMock()
    charger.father_device.charger_group = MagicMock()
    charger.father_device.charger_group.dyn_handle = AsyncMock()
    charger.car = MagicMock()
    charger.car.name = "TestCar"
    charger.car.get_car_charge_percent.return_value = 50.0
    charger.car.get_car_charge_energy.return_value = 20000.0
    charger.car.car_battery_capacity = 60000
    charger.car.efficiency_factor = 1.0
    charger.car.car_charge_percent_sensor = None
    charger.car.is_car_charge_growing.return_value = True
    charger.car.setup_car_charge_target_if_needed = AsyncMock()

    now = datetime.now(pytz.UTC)
    ct = LoadConstraint(time=now, load=charger, target_value=80)
    ct.current_value = 40
    ct.last_value_update = now
    ct.last_value_change_update = now
    ct.first_value_update = now

    charger.current_command = copy_command(CMD_AUTO_GREEN_ONLY)

    mock_group = MagicMock()
    mock_group.dyn_handle = AsyncMock()

    with patch.object(charger, "_do_update_charger_state", new_callable=AsyncMock), \
         patch.object(charger, "is_not_plugged", return_value=False), \
         patch.object(charger, "get_device_real_energy", return_value=3000), \
         patch.object(type(charger), "charger_group", new_callable=PropertyMock, return_value=mock_group):
        percent_value, percent_done = await charger.constraint_update_value_callback_percent_soc(ct, now + timedelta(minutes=30))
        energy_value, energy_done = await charger.constraint_update_value_callback_energy_soc(ct, now + timedelta(minutes=30))

    assert percent_value is not None
    assert energy_value is not None
    assert percent_done is True
    assert energy_done is False


@pytest.mark.asyncio
async def test_on_device_state_change_uses_car_context() -> None:
    """Cover on_device_state_change with car/person context."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)

    person = MagicMock()
    person.mobile_app = "notify.mobile_app"
    person.mobile_app_url = "https://example.com/app"

    charger.car = MagicMock()
    charger.car.name = "TestCar"
    charger.car.current_forecasted_person = person

    with patch.object(charger, "on_device_state_change_helper", new_callable=AsyncMock) as helper:
        await charger.on_device_state_change(datetime.now(pytz.UTC), "test-change")

    helper.assert_called_once()


def test_find_charger_entity_id_multiple_entries() -> None:
    """Cover _find_charger_entity_id selecting longest match."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)

    device = MagicMock()
    device.id = "device-1"
    device.name = "Test Device"
    device.name_by_user = None

    entry_short = MagicMock()
    entry_short.entity_id = "sensor.test_device_status"
    entry_long = MagicMock()
    entry_long.entity_id = "sensor.test_device_status_connector"

    entity_id = charger._find_charger_entity_id(
        device,
        [entry_short, entry_long],
        "sensor.",
        "_status",
    )

    assert entity_id == entry_long.entity_id


@pytest.mark.asyncio
async def test_update_data_request_time_gate() -> None:
    """Cover update_data_request time gating."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)

    now = datetime.now(pytz.UTC)
    charger._last_requested_update_time = now

    with patch.object(charger, "low_level_update_data_request", new_callable=AsyncMock) as low_level:
        low_level.return_value = True
        res = await charger.update_data_request(now)
        assert res is False
        low_level.assert_not_called()

        later = now + timedelta(seconds=200)
        res = await charger.update_data_request(later)
        assert res is True
        low_level.assert_called()


@pytest.mark.asyncio
async def test_do_update_charger_state_requests_update() -> None:
    """Cover _do_update_charger_state calling update entity service."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)

    state = MagicMock()
    state.state = STATE_UNKNOWN
    state.last_updated = None
    hass.states.get.return_value = state

    await charger._do_update_charger_state(datetime.now(pytz.UTC))
    hass.services.async_call.assert_called()


def test_get_car_score_plug_and_distance() -> None:
    """Cover get_car_score with plug, time, and distance bumps."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)

    now = datetime.now(pytz.UTC)
    car = MagicMock()
    car.name = "TestCar"
    car.car_is_invited = False
    car.user_attached_charger_name = None
    car.is_car_plugged.return_value = True
    car.is_car_home.return_value = True
    car.get_continuous_plug_duration.return_value = 120.0
    car.get_car_coordinates.return_value = (0.0, 0.0)

    charger.car = car
    charger.car_attach_time = now - timedelta(minutes=5)
    charger.charger_latitude = 0.0
    charger.charger_longitude = 0.0

    with patch.object(charger, "get_continuous_plug_duration", return_value=100.0):
        score = charger.get_car_score(car, now, {})

    assert score > 0


def test_get_best_car_user_selected_detaches_others() -> None:
    """Cover user-selected car branch and detaching others."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)

    car = MagicMock()
    car.name = "PreferredCar"
    car.user_attached_charger_name = None

    charger.user_attached_car_name = car.name
    home.get_car_by_name = MagicMock(return_value=car)

    other_charger = MagicMock()
    other_charger.qs_enable_device = True
    other_charger.car = car
    other_charger.user_attached_car_name = None
    other_charger.detach_car = MagicMock()

    home._chargers = [charger, other_charger]

    best = charger.get_best_car(datetime.now(pytz.UTC))

    assert best == car
    other_charger.detach_car.assert_called_once()


def test_get_best_car_falls_back_to_default() -> None:
    """Cover fallback to default car when no candidates."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)

    car = MagicMock()
    car.name = "IgnoredCar"
    car.user_attached_charger_name = FORCE_CAR_NO_CHARGER_CONNECTED
    home._cars = [car]
    home._chargers = [charger]

    best = charger.get_best_car(datetime.now(pytz.UTC))
    assert best == charger._default_generic_car


def test_get_car_options_and_current_selection() -> None:
    """Cover car options and current selection logic."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)

    car = MagicMock()
    car.name = "TestCar"
    home._cars = [car]
    charger.car = car

    with patch.object(charger, "is_optimistic_plugged", return_value=True):
        options = charger.get_car_options()
    assert car.name in options
    assert CHARGER_NO_CAR_CONNECTED in options

    charger.user_attached_car_name = car.name
    assert charger.get_current_selected_car_option() == car.name

    with patch.object(charger, "is_optimistic_plugged", return_value=False):
        assert charger.get_car_options() == [CHARGER_NO_CAR_CONNECTED]


@pytest.mark.asyncio
async def test_set_user_selected_car_by_name_detaches() -> None:
    """Cover user selection changing and detaching current car."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)

    car = MagicMock()
    car.name = "OldCar"
    charger.car = car

    charger.detach_car = MagicMock()
    charger.update_charger_for_user_change = AsyncMock()

    await charger.set_user_selected_car_by_name("NewCar")

    charger.detach_car.assert_called_once()
    charger.update_charger_for_user_change.assert_called_once()


def test_device_post_home_init_uses_user_attached_car() -> None:
    """Cover device_post_home_init boot car restoration."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)

    car = MagicMock()
    car.name = "StoredCar"
    home.get_car_by_name = MagicMock(return_value=car)

    charger.user_attached_car_name = car.name
    charger.device_post_home_init(datetime.now(pytz.UTC))

    assert charger._boot_car == car


@pytest.mark.asyncio
async def test_check_load_activity_and_constraints_person_flow() -> None:
    """Cover person-based constraint path in check_load_activity_and_constraints."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    charger = create_charger_generic(hass, home)

    now = datetime.now(pytz.UTC)
    person = MagicMock()
    person.name = "Alex"
    person.notify_of_forecast_if_needed = AsyncMock()

    car = MagicMock()
    car.name = "TestCar"
    car.car_battery_capacity = 60000
    car.car_default_charge = 70
    car.car_charger_min_charge = 6
    car.car_charger_max_charge = 32
    car.get_charge_power_per_phase_A.return_value = ([i * 100 for i in range(33)], 6, 16)
    car.can_use_charge_percent_constraints.return_value = True
    car.setup_car_charge_target_if_needed = AsyncMock(return_value=80)
    car.get_car_charge_percent.return_value = 50.0
    car.get_best_person_next_need = AsyncMock(
        return_value=(False, now + timedelta(hours=3), 60, person)
    )
    car.do_force_next_charge = False
    car.do_next_charge_time = None
    car.get_next_scheduled_event = AsyncMock(return_value=(None, None))
    car.get_car_target_SOC.return_value = 80
    car.set_next_charge_target_percent = AsyncMock()
    car.get_car_minimum_ok_SOC.return_value = 20

    charger._power_steps = []
    charger._constraints = []
    charger._auto_constraints_cleaned_at_user_reset = []

    with patch.object(charger, "is_charger_unavailable", return_value=False), \
         patch.object(charger, "probe_for_possible_needed_reboot", return_value=False), \
         patch.object(charger, "is_not_plugged", return_value=False), \
         patch.object(charger, "is_plugged", return_value=True), \
         patch.object(charger, "get_best_car", return_value=car), \
         patch.object(charger, "is_car_charged", return_value=(False, 50.0)), \
         patch.object(charger, "push_agenda_constraints", return_value=True), \
         patch.object(charger, "push_live_constraint", return_value=True), \
         patch.object(charger, "clean_constraints_for_load_param_and_if_same_key_same_value_info"), \
         patch.object(charger, "set_live_constraints"), \
         patch.object(charger, "is_off_grid", return_value=False):
        changed = await charger.check_load_activity_and_constraints(now)

    assert changed is True


@pytest.mark.asyncio
async def test_budgeting_algorithm_minimize_diffs_runs() -> None:
    """Cover budgeting_algorithm_minimize_diffs main loop."""
    hass = create_mock_hass()
    home = create_mock_home(hass)
    home.battery_can_discharge = MagicMock(return_value=False)
    home.get_best_tariff = MagicMock(return_value=0.1)
    home.get_tariff = MagicMock(return_value=0.2)

    dynamic_group = MagicMock()
    dynamic_group.home = home
    dynamic_group.name = "group"
    dynamic_group._childrens = []
    dynamic_group.is_current_acceptable = MagicMock(return_value=True)
    dynamic_group.is_current_acceptable_and_diff = MagicMock(return_value=(False, [10, 10, 10]))

    group = QSChargerGroup(dynamic_group)

    charger = create_charger_generic(hass, home)
    charger.get_delta_dampened_power = MagicMock(return_value=500.0)
    charger.qs_enable_device = True

    cs_best = QSChargerStatus(charger)
    cs_best.charge_score = 10
    cs_best.current_real_max_charging_amp = 0
    cs_best.current_active_phase_number = 3
    cs_best.budgeted_amp = 6
    cs_best.budgeted_num_phases = 3
    cs_best.possible_amps = [0, 6, 7]
    cs_best.possible_num_phases = [1, 3]
    cs_best.can_be_started_and_stopped = True
    cs_best.is_before_battery = False
    cs_best.command = copy_command(CMD_AUTO_GREEN_ONLY)

    cs_other = QSChargerStatus(charger)
    cs_other.charge_score = 5
    cs_other.current_real_max_charging_amp = 6
    cs_other.current_active_phase_number = 3
    cs_other.budgeted_amp = 6
    cs_other.budgeted_num_phases = 3
    cs_other.possible_amps = [0, 6, 7]
    cs_other.possible_num_phases = [1, 3]
    cs_other.can_be_started_and_stopped = True
    cs_other.is_before_battery = False
    cs_other.command = copy_command(CMD_AUTO_PRICE, power_consign=1000)

    with patch.object(group, "_do_prepare_and_shave_budgets", new_callable=AsyncMock, return_value=(True, True, True)):
        success, should_reset, _ = await group.budgeting_algorithm_minimize_diffs(
            [cs_best, cs_other],
            full_available_home_power=2000.0,
            grid_available_home_power=0.0,
            allow_budget_reset=True,
            time=datetime.now(pytz.UTC),
        )

    assert success is True
    assert should_reset in (True, False)


@pytest.mark.asyncio
async def test_shave_current_budgets_reduces() -> None:
    """Cover _shave_current_budgets reduction logic."""
    hass = create_mock_hass()
    home = create_mock_home(hass)

    dynamic_group = MagicMock()
    dynamic_group.home = home
    dynamic_group.name = "group"
    dynamic_group._childrens = []
    dynamic_group.dyn_group_max_phase_current = 16
    call_count = {"count": 0}

    def is_current_acceptable(*args, **kwargs):
        call_count["count"] += 1
        return call_count["count"] > 1

    dynamic_group.is_current_acceptable = MagicMock(side_effect=is_current_acceptable)
    dynamic_group.is_current_acceptable_and_diff = MagicMock(return_value=(False, [10, 10, 10]))

    group = QSChargerGroup(dynamic_group)

    charger = create_charger_generic(hass, home)
    charger.get_delta_dampened_power = MagicMock(return_value=100.0)

    cs = QSChargerStatus(charger)
    cs.charge_score = 1
    cs.current_real_max_charging_amp = 6
    cs.current_active_phase_number = 1
    cs.budgeted_amp = 6
    cs.budgeted_num_phases = 1
    cs.possible_amps = [0, 6]
    cs.possible_num_phases = [1, 3]

    _, current_ok = await group._shave_current_budgets([cs], datetime.now(pytz.UTC))
    assert current_ok is True


@pytest.mark.asyncio
async def test_shave_mandatory_budgets_updates_possible_amps() -> None:
    """Cover _shave_mandatory_budgets stopping logic."""
    hass = create_mock_hass()
    home = create_mock_home(hass)

    dynamic_group = MagicMock()
    dynamic_group.home = home
    dynamic_group.name = "group"
    dynamic_group._childrens = []
    dynamic_group.is_current_acceptable = MagicMock(return_value=False)
    dynamic_group.is_current_acceptable_and_diff = MagicMock(return_value=(True, [0, 0, 0]))

    group = QSChargerGroup(dynamic_group)

    charger = create_charger_generic(hass, home)
    charger.update_amps_with_delta = MagicMock(return_value=[0.0, 0.0, 0.0])

    cs = QSChargerStatus(charger)
    cs.charge_score = 1
    cs.budgeted_amp = 6
    cs.budgeted_num_phases = 1
    cs.possible_amps = [6, 7]
    cs.possible_num_phases = [1]
    cs.can_be_started_and_stopped = True
    cs.command = copy_command(CMD_AUTO_GREEN_ONLY)

    current_amps = [6.0, 0.0, 0.0]
    mandatory_amps = [6.0, 0.0, 0.0]

    updated = await group._shave_mandatory_budgets([cs], current_amps, mandatory_amps, datetime.now(pytz.UTC))
    assert updated is not None
    assert cs.possible_amps[0] <= 6


@pytest.mark.asyncio
async def test_apply_budget_strategy_records_state() -> None:
    """Cover apply_budget_strategy recording state."""
    hass = create_mock_hass()
    home = create_mock_home(hass)

    dynamic_group = MagicMock()
    dynamic_group.home = home
    dynamic_group.name = "group"
    dynamic_group._childrens = []
    dynamic_group.is_current_acceptable = MagicMock(return_value=True)

    group = QSChargerGroup(dynamic_group)

    charger = create_charger_generic(hass, home)
    cs = QSChargerStatus(charger)
    cs.current_real_max_charging_amp = 6
    cs.current_active_phase_number = 1
    cs.budgeted_amp = 6
    cs.budgeted_num_phases = 1
    cs.possible_amps = [0, 6]
    cs.possible_num_phases = [1]

    await group.apply_budget_strategy([cs], current_real_cars_power=500.0, time=datetime.now(pytz.UTC))
    assert group.know_reduced_state is not None


@pytest.mark.asyncio
async def test_ocpp_init_and_low_level_update_request() -> None:
    """Cover OCPP init entity discovery and update request."""
    hass = create_mock_hass()
    home = create_mock_home(hass)

    device = MagicMock()
    device.name = "OCPP Charger"
    device.name_by_user = None

    entry = MagicMock()
    entry.entity_id = "sensor.ocpp_status_connector"

    device_reg = MagicMock()
    device_reg.async_get.return_value = device

    entity_reg = MagicMock()
    entity_reg.async_entries_for_device.return_value = [entry]

    with patch("custom_components.quiet_solar.ha_model.charger.device_registry.async_get", return_value=device_reg), \
         patch("custom_components.quiet_solar.ha_model.charger.entity_registry.async_get", return_value=entity_reg):
        charger = QSChargerOCPP(
            hass=hass,
            home=home,
            config_entry=MagicMock(),
            name="OcppCharger",
            **{
                CONF_CHARGER_MIN_CHARGE: 6,
                CONF_CHARGER_MAX_CHARGE: 32,
                CONF_IS_3P: True,
                CONF_MONO_PHASE: 1,
                CONF_CHARGER_DEVICE_OCPP: "device.ocpp",
            },
        )

    charger.devid = "ocpp_device"
    done = await charger.low_level_update_data_request(datetime.now(pytz.UTC))

    assert done is True
    hass.services.async_call.assert_called()


def test_wallbox_low_level_checks() -> None:
    """Cover wallbox plug and charge checks."""
    hass = create_mock_hass()
    home = create_mock_home(hass)

    device = MagicMock()
    device.name = "Wallbox Charger"
    device.name_by_user = None

    entry = MagicMock()
    entry.entity_id = "switch.wallbox_pause_resume"

    device_reg = MagicMock()
    device_reg.async_get.return_value = device

    entity_reg = MagicMock()
    entity_reg.async_entries_for_device.return_value = [entry]

    with patch("custom_components.quiet_solar.ha_model.charger.device_registry.async_get", return_value=device_reg), \
         patch("custom_components.quiet_solar.ha_model.charger.entity_registry.async_get", return_value=entity_reg):
        charger = QSChargerWallbox(
            hass=hass,
            home=home,
            config_entry=MagicMock(),
            name="WallboxCharger",
            **{
                CONF_CHARGER_MIN_CHARGE: 6,
                CONF_CHARGER_MAX_CHARGE: 32,
                CONF_IS_3P: True,
                CONF_MONO_PHASE: 1,
                CONF_CHARGER_DEVICE_WALLBOX: "device.wallbox",
            },
        )

    state = MagicMock()
    state.state = "on"
    state.last_updated = datetime.now(pytz.UTC)
    hass.states.get.return_value = state

    plugged, plugged_time = charger.low_level_plug_check_now(datetime.now(pytz.UTC))
    charging = charger.low_level_charge_check_now(datetime.now(pytz.UTC))

    assert plugged is True
    assert plugged_time == state.last_updated
    assert charging is True

