"""Tests for quiet_solar home.py functionality."""

import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytz
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import device_registry as dr

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_HOME_VOLTAGE,
    CONF_IS_3P,
    CONF_GRID_POWER_SENSOR,
)
from custom_components.quiet_solar.home_model.commands import CMD_IDLE
import numpy as np

from custom_components.quiet_solar.ha_model.home import (
    QSHomeMode,
    QSHomeConsumptionHistoryAndForecast,
    QSSolarHistoryVals,
    QSforecastValueSensor,
    _segments_weak_sub_on_main_overlap,
    _segments_strong_overlap,
    BUFFER_SIZE_IN_INTERVALS,
    NUM_INTERVALS_PER_DAY,
    BEGINING_OF_TIME,
)

pytestmark = pytest.mark.usefixtures("mock_sensor_states")


async def test_home_initialization(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home device initialization."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    assert home_config_entry.state is ConfigEntryState.LOADED
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    assert data_handler is not None
    assert data_handler.home is not None


async def test_home_voltage_configuration(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home voltage configuration."""
    from .const import MOCK_HOME_CONFIG
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home
    assert home.voltage == MOCK_HOME_CONFIG[CONF_HOME_VOLTAGE]


async def test_home_grid_power_sensor(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home grid power sensor configuration."""
    from .const import MOCK_HOME_CONFIG
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home
    assert home.grid_active_power_sensor == MOCK_HOME_CONFIG[CONF_GRID_POWER_SENSOR]


async def test_home_sensor_entities(hass: HomeAssistant, home_config_entry: ConfigEntry, entity_registry: er.EntityRegistry) -> None:
    """Test home sensor entities are created."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    entity_entries = er.async_entries_for_config_entry(entity_registry, home_config_entry.entry_id)
    sensor_entries = [e for e in entity_entries if e.domain == "sensor"]
    assert len(sensor_entries) >= 10


async def test_home_select_entities(hass: HomeAssistant, home_config_entry: ConfigEntry, entity_registry: er.EntityRegistry) -> None:
    """Test home select entities are created."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    entity_entries = er.async_entries_for_config_entry(entity_registry, home_config_entry.entry_id)
    select_entries = [e for e in entity_entries if e.domain == "select"]
    assert len(select_entries) >= 1


async def test_home_switch_entities(hass: HomeAssistant, home_config_entry: ConfigEntry, entity_registry: er.EntityRegistry) -> None:
    """Test home switch entities are created."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    entity_entries = er.async_entries_for_config_entry(entity_registry, home_config_entry.entry_id)
    switch_entries = [e for e in entity_entries if e.domain == "switch"]
    assert len(switch_entries) >= 1


async def test_home_button_entities(hass: HomeAssistant, home_config_entry: ConfigEntry, entity_registry: er.EntityRegistry) -> None:
    """Test home button entities are created."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    entity_entries = er.async_entries_for_config_entry(entity_registry, home_config_entry.entry_id)
    button_entries = [e for e in entity_entries if e.domain == "button"]
    assert len(button_entries) >= 4


async def test_home_with_solar(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home with solar inverter configured."""
    from .const import MOCK_SOLAR_CONFIG
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    solar_entry = MockConfigEntry(domain=DOMAIN, data=MOCK_SOLAR_CONFIG, entry_id="solar_test",
                                  title=f"solar: {MOCK_SOLAR_CONFIG['name']}", unique_id="quiet_solar_solar_test")
    solar_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(solar_entry.entry_id)
    await hass.async_block_till_done()
    assert solar_entry.state is ConfigEntryState.LOADED


async def test_home_with_battery(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home with battery configured."""
    from .const import MOCK_BATTERY_CONFIG
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    battery_entry = MockConfigEntry(domain=DOMAIN, data=MOCK_BATTERY_CONFIG, entry_id="battery_test",
                                    title=f"battery: {MOCK_BATTERY_CONFIG['name']}", unique_id="quiet_solar_battery_test")
    battery_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(battery_entry.entry_id)
    await hass.async_block_till_done()
    assert battery_entry.state is ConfigEntryState.LOADED


async def test_home_mode_selection(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home mode selection entity."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    state = hass.states.get("select.qs_test_home_home_home_mode")
    assert state is not None
    assert state.state == "home_mode_sensors_only"


async def test_home_off_grid_switch(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home off-grid switch entity."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    state = hass.states.get("switch.qs_test_home_home_qs_home_is_off_grid")
    assert state is not None
    assert state.state == "off"


async def test_home_reset_button(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home reset button entity exists."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    state = hass.states.get("button.qs_test_home_home_qs_home_reset_history")
    assert state is not None


async def test_home_forecast_sensors(hass: HomeAssistant, home_config_entry: ConfigEntry, entity_registry: er.EntityRegistry) -> None:
    """Test home forecast sensor entities are created."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    entity_entries = er.async_entries_for_config_entry(entity_registry, home_config_entry.entry_id)
    sensor_entries = [e for e in entity_entries if e.domain == "sensor"]
    forecast_sensors = [e for e in sensor_entries if "forecast" in e.entity_id.lower()]
    assert len(forecast_sensors) >= 6


async def test_home_consumption_sensors(hass: HomeAssistant, home_config_entry: ConfigEntry, entity_registry: er.EntityRegistry) -> None:
    """Test home consumption sensor entities are created."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    entity_entries = er.async_entries_for_config_entry(entity_registry, home_config_entry.entry_id)
    sensor_entries = [e for e in entity_entries if e.domain == "sensor"]
    consumption_sensors = [e for e in sensor_entries if "consumption" in e.entity_id.lower()]
    assert len(consumption_sensors) >= 2


async def test_home_available_power_sensor(hass: HomeAssistant, home_config_entry: ConfigEntry, entity_registry: er.EntityRegistry) -> None:
    """Test home available power sensor is created."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    entity_entries = er.async_entries_for_config_entry(entity_registry, home_config_entry.entry_id)
    sensor_entries = [e for e in entity_entries if e.domain == "sensor"]
    available_power_sensors = [e for e in sensor_entries if "available_power" in e.entity_id.lower()]
    assert len(available_power_sensors) >= 1


async def test_home_unload_and_reload(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home unload and reload functionality."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    assert home_config_entry.state is ConfigEntryState.LOADED
    await hass.config_entries.async_unload(home_config_entry.entry_id)
    await hass.async_block_till_done()
    assert home_config_entry.state is ConfigEntryState.NOT_LOADED
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    assert home_config_entry.state is ConfigEntryState.LOADED


async def test_home_device_registry(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home device is registered in device registry."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    device_registry = dr.async_get(hass)
    devices = dr.async_entries_for_config_entry(device_registry, home_config_entry.entry_id)
    assert len(devices) >= 1


async def test_home_get_platforms(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home returns correct platforms."""
    from homeassistant.const import Platform
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home
    platforms = home.get_platforms()
    assert Platform.SENSOR in platforms
    assert Platform.SELECT in platforms
    assert Platform.SWITCH in platforms
    assert Platform.BUTTON in platforms


async def test_home_with_full_setup(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home with full setup (solar, battery, charger, car, person)."""
    from .const import (
        MOCK_SOLAR_CONFIG, MOCK_BATTERY_CONFIG,
        MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG, MOCK_PERSON_CONFIG
    )
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create solar
    solar_entry = MockConfigEntry(domain=DOMAIN, data=MOCK_SOLAR_CONFIG, entry_id="solar_full_test",
                                  title=f"solar: {MOCK_SOLAR_CONFIG['name']}", unique_id="quiet_solar_solar_full_test")
    solar_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(solar_entry.entry_id)
    await hass.async_block_till_done()

    # Create battery
    battery_entry = MockConfigEntry(domain=DOMAIN, data=MOCK_BATTERY_CONFIG, entry_id="battery_full_test",
                                    title=f"battery: {MOCK_BATTERY_CONFIG['name']}", unique_id="quiet_solar_battery_full_test")
    battery_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(battery_entry.entry_id)
    await hass.async_block_till_done()

    # Create charger
    charger_entry = MockConfigEntry(domain=DOMAIN, data=MOCK_CHARGER_CONFIG, entry_id="charger_full_test",
                                    title=f"charger: {MOCK_CHARGER_CONFIG['name']}", unique_id="quiet_solar_charger_full_test")
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(domain=DOMAIN, data=MOCK_CAR_CONFIG, entry_id="car_full_test",
                                title=f"car: {MOCK_CAR_CONFIG['name']}", unique_id="quiet_solar_car_full_test")
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create person
    person_entry = MockConfigEntry(domain=DOMAIN, data=MOCK_PERSON_CONFIG, entry_id="person_full_test",
                                   title=f"person: {MOCK_PERSON_CONFIG['name']}", unique_id="quiet_solar_person_full_test")
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    # Verify all entries are loaded
    assert home_config_entry.state is ConfigEntryState.LOADED
    assert solar_entry.state is ConfigEntryState.LOADED
    assert battery_entry.state is ConfigEntryState.LOADED
    assert charger_entry.state is ConfigEntryState.LOADED
    assert car_entry.state is ConfigEntryState.LOADED
    assert person_entry.state is ConfigEntryState.LOADED


async def test_home_data_handler_reference(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test data handler holds reference to home."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    assert data_handler.home is not None
    assert data_handler.hass is hass


async def test_home_get_car_by_name_none(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home get_car_by_name returns None for unknown car."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home
    # Non-existent car
    car = home.get_car_by_name("NonExistentCar")
    assert car is None


async def test_home_get_person_by_name_none(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home get_person_by_name returns None for unknown person."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home
    # Non-existent person
    person = home.get_person_by_name("NonExistentPerson")
    assert person is None


async def test_home_cars_list_empty(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home cars list is empty when no cars are configured."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home
    # Initially no cars
    assert len(home._cars) == 0


async def test_home_chargers_list_empty(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home chargers list is empty when no chargers are configured."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home
    # Initially no chargers
    assert len(home._chargers) == 0


async def test_home_persons_list_empty(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home persons list is empty when no persons are configured."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home
    # Initially no persons
    assert len(home._persons) == 0


async def test_home_with_car_populated(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home cars list is populated when car is added."""
    from .const import MOCK_CAR_CONFIG
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Add car
    car_entry = MockConfigEntry(domain=DOMAIN, data=MOCK_CAR_CONFIG, entry_id="car_pop_test",
                                title=f"car: {MOCK_CAR_CONFIG['name']}", unique_id="quiet_solar_car_pop_test")
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home
    # Now should have 1 car
    assert len(home._cars) == 1

    # Get car by name
    car = home.get_car_by_name(MOCK_CAR_CONFIG['name'])
    assert car is not None
    assert car.name == MOCK_CAR_CONFIG['name']


async def test_home_off_grid_limits_and_reset(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test off-grid mode updates limits and resets loads."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.physical_solar_plant = MagicMock(
        solar_production=3000.0,
        solar_max_output_power_value=2500.0,
        solar_max_phase_amps=10.0,
    )
    home.physical_battery = MagicMock(
        battery_can_discharge=MagicMock(return_value=True),
        get_max_discharging_power=MagicMock(return_value=2000.0),
    )

    load = MagicMock()
    load.qs_enable_device = True
    load.launch_command = AsyncMock()
    home._all_loads = [load]

    await home.async_set_off_grid_mode(True, for_init=False)

    load.launch_command.assert_awaited()
    assert load.launch_command.call_args.kwargs["command"] == CMD_IDLE

    max_phase_amps = home.get_home_max_phase_amps()
    assert max_phase_amps <= home.dyn_group_max_phase_current_conf

    dyn_limits = home.dyn_group_max_phase_current
    assert len(dyn_limits) == 3
    assert all(val <= home.dyn_group_max_phase_current_conf for val in dyn_limits)

    budget_limits = home.dyn_group_max_phase_current_for_budget
    assert len(budget_limits) == 3


async def test_home_devices_for_dashboard_section_includes_virtual_devices(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test dashboard section includes attached virtual devices."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_dashboard_section_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_dashboard_section_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home
    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    devices = home.get_devices_for_dashboard_section(charger_device.dashboard_section)
    device_names = {device.name for device in devices}

    assert charger_device.name in device_names
    assert charger_device._default_generic_car.name in device_names


async def test_home_best_tariff_selection(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test best tariff selection logic."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.price_peak = 0.30
    home.price_off_peak = 0.12
    assert home.get_best_tariff(datetime.now(tz=pytz.UTC)) == 0.12

    home.price_off_peak = 0.0
    assert home.get_best_tariff(datetime.now(tz=pytz.UTC)) == 0.30


async def test_home_battery_helpers_and_topology(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test battery helpers and topology setup."""
    from .const import MOCK_DYNAMIC_GROUP_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    battery = MagicMock()
    battery.get_state_history_data = MagicMock(return_value=[(datetime.now(tz=pytz.UTC), 100.0, {})])
    battery.set_max_discharging_power = AsyncMock()
    battery.set_max_charging_power = AsyncMock()
    home.physical_battery = battery

    values = home.get_battery_charge_values(3600.0, datetime.now(tz=pytz.UTC))
    assert values

    await home.set_max_discharging_power(1200.0, blocking=True)
    battery.set_max_discharging_power.assert_awaited_with(1200.0, True)

    await home.set_max_charging_power(500.0, blocking=False)
    battery.set_max_charging_power.assert_awaited_with(500.0, False)

    dynamic_group_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_DYNAMIC_GROUP_CONFIG,
        entry_id="dynamic_group_topology_test",
        title=f"dynamic_group: {MOCK_DYNAMIC_GROUP_CONFIG['name']}",
        unique_id="quiet_solar_dynamic_group_topology_test",
    )
    dynamic_group_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(dynamic_group_entry.entry_id)
    await hass.async_block_till_done()

    dynamic_group_device = hass.data[DOMAIN].get(dynamic_group_entry.entry_id)
    dynamic_group_device.dynamic_group_name = home.name

    mock_load = MagicMock()
    home._all_dynamic_groups = [dynamic_group_device]
    home._all_loads = [dynamic_group_device, mock_load]

    home._set_topology()

    assert dynamic_group_device.father_device is home


async def test_home_update_all_states(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_all_states updates devices and forecasts."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home
    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home.finish_setup = AsyncMock(return_value=True)

    solar = MagicMock()
    solar.update_forecast = AsyncMock()
    home.physical_solar_plant = solar

    device = MagicMock()
    device.update_states = AsyncMock()
    home._all_devices = [device]

    consumption = MagicMock()
    consumption.add_value = MagicMock(return_value=True)
    consumption.save_values = AsyncMock()
    consumption.update_current_forecast_if_needed = MagicMock(return_value=True)

    home._consumption_forecast = MagicMock()
    home._consumption_forecast.init_forecasts = AsyncMock(return_value=True)
    home._consumption_forecast.home_non_controlled_consumption = consumption
    home._compute_non_controlled_forecast_intl = MagicMock()

    time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    await home.update_all_states(time)

    solar.update_forecast.assert_awaited()
    device.update_states.assert_awaited()
    home._compute_non_controlled_forecast_intl.assert_called_once()


async def test_home_update_loads_solver_path(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_loads runs solver and launches commands."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home
    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home.finish_setup = AsyncMock(return_value=True)
    home.update_loads_constraints = AsyncMock()
    home.compute_non_controlled_forecast = AsyncMock(return_value=[])
    home.get_solar_from_current_forecast = MagicMock(return_value=[])

    load = MagicMock()
    load.name = "Load 1"
    load.check_commands = AsyncMock(return_value=timedelta(seconds=0))
    load.running_command_num_relaunch = 0
    load.force_relaunch_command = AsyncMock()
    load.is_load_active = MagicMock(return_value=True)
    load.update_live_constraints = AsyncMock(return_value=True)
    load.get_phase_amps_from_power_for_budgeting = MagicMock(return_value=[0, 0, 0])
    load.launch_command = AsyncMock()
    load.is_load_has_a_command_now_or_coming = MagicMock(return_value=False)
    load.get_current_active_constraint = MagicMock(return_value=None)
    load.do_probe_state_change = AsyncMock()
    load.current_command = None
    load.father_device = SimpleNamespace(is_delta_current_acceptable=MagicMock(return_value=True))

    home._all_loads = [load]
    home._chargers = [load]

    battery = MagicMock()
    battery.launch_command = AsyncMock()
    home.physical_battery = battery

    class DummySolver:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def solve(self, is_off_grid: bool):
            cmd_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
            return ([(load, [(cmd_time, CMD_IDLE)])], [(cmd_time, CMD_IDLE)])

    with patch("custom_components.quiet_solar.ha_model.home.PeriodSolver", DummySolver):
        time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
        await home.update_loads(time)

    load.launch_command.assert_awaited()
    battery.launch_command.assert_awaited()


async def test_home_best_persons_cars_allocations_basic(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car/person allocation flow."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    car_a = SimpleNamespace(
        name="Car A",
        current_forecasted_person=None,
        user_selected_person_name_for_car=None,
        car_is_invited=False,
        get_adapt_target_percent_soc_to_reach_range_km=MagicMock(
            return_value=(False, 40.0, 80.0, 10.0)
        ),
    )
    car_b = SimpleNamespace(
        name="Car B",
        current_forecasted_person=None,
        user_selected_person_name_for_car=None,
        car_is_invited=False,
        get_adapt_target_percent_soc_to_reach_range_km=MagicMock(
            return_value=(True, 90.0, 60.0, 0.0)
        ),
    )

    person_a = SimpleNamespace(
        name="Person A",
        preferred_car="Car A",
        update_person_forecast=MagicMock(
            return_value=(datetime(2026, 1, 16, 8, 0, tzinfo=pytz.UTC), 30.0)
        ),
        get_authorized_cars=MagicMock(return_value=[car_a, car_b]),
    )
    person_b = SimpleNamespace(
        name="Person B",
        preferred_car="Car B",
        update_person_forecast=MagicMock(
            return_value=(datetime(2026, 1, 16, 9, 0, tzinfo=pytz.UTC), 20.0)
        ),
        get_authorized_cars=MagicMock(return_value=[car_a, car_b]),
    )

    home._cars = [car_a, car_b]
    home._persons = [person_a, person_b]

    result = await home.get_best_persons_cars_allocations(
        time=datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC),
        force_update=True,
        do_notify=False,
    )

    assert result
    assert car_a.current_forecasted_person is not None
    assert car_b.current_forecasted_person is not None


async def test_home_recompute_people_historical_data(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test recompute_people_historical_data flow."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    person = SimpleNamespace(
        name="Person A",
        historical_mileage_data=[("x", 1, "y", 0)],
        has_been_initialized=False,
        update_person_forecast=MagicMock(),
    )
    home._persons = [person]

    home._compute_and_store_person_car_forecasts = AsyncMock()

    await home.recompute_people_historical_data(datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC))

    assert person.has_been_initialized is True
    person.update_person_forecast.assert_called()


async def test_home_compute_and_store_person_car_forecasts(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test storing mileage forecasts for people."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    person = MagicMock()
    person.name = "Person A"
    person.add_to_mileage_history = MagicMock()
    home._persons = [person]

    leave_time = datetime(2026, 1, 14, 18, 0, tzinfo=pytz.UTC)
    home._compute_mileage_for_period_per_person = AsyncMock(
        return_value={person: (25.0, leave_time)}
    )

    await home._compute_and_store_person_car_forecasts(
        datetime(2026, 1, 15, 0, 0, tzinfo=pytz.UTC), day_shift=1
    )

    person.add_to_mileage_history.assert_called()


async def test_home_update_loads_relaunch_and_forbid(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_loads relaunch and forbid command paths."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home
    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home.finish_setup = AsyncMock(return_value=True)
    home.update_loads_constraints = AsyncMock()

    load = MagicMock()
    load.name = "Load 2"
    load.check_commands = AsyncMock(return_value=timedelta(seconds=60))
    load.running_command_num_relaunch = 0
    load.force_relaunch_command = AsyncMock()
    load.is_load_active = MagicMock(return_value=True)
    load.update_live_constraints = AsyncMock(return_value=False)
    load.get_phase_amps_from_power_for_budgeting = MagicMock(return_value=[1, 1, 1])
    load.launch_command = AsyncMock()
    load.is_load_has_a_command_now_or_coming = MagicMock(return_value=False)
    load.get_current_active_constraint = MagicMock(return_value=None)
    load.do_probe_state_change = AsyncMock()
    load.current_command = CMD_IDLE
    load.father_device = SimpleNamespace(is_delta_current_acceptable=MagicMock(return_value=False))

    home._all_loads = [load]
    home._commands = [(load, [(datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC), CMD_IDLE)])]
    home._battery_commands = []
    home._last_solve_done = datetime(2026, 1, 15, 8, 59, tzinfo=pytz.UTC)

    await home.update_loads(datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))

    load.force_relaunch_command.assert_awaited()


def test_solar_history_vals_forecast_scores(tmp_path) -> None:
    """Test history values forecast scoring."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")

    vals.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS))
    vals.values[0][:] = 10.0
    vals.values[1][:] = 1

    now_idx = NUM_INTERVALS_PER_DAY * 2
    scores = vals._get_possible_past_consumption_for_forecast(
        now_idx=now_idx,
        now_days=2,
        history_in_hours=4,
    )

    assert scores


def test_solar_history_vals_add_value(tmp_path) -> None:
    """Test adding values and reading current non-stored value."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")
    vals.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS))

    time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    added = vals.add_value(time, 50.0)
    assert added is False

    current_val, current_duration = vals.get_current_non_stored_val_at_time(time)
    assert current_val is not None


def test_solar_history_vals_range_score(tmp_path) -> None:
    """Test range scoring and correlation helpers."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")

    current_values = np.array([1.0, 2.0, 3.0, 4.0])
    current_ok = np.array([1, 1, 1, 1])
    vals.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS))
    vals.values[0][:] = 1.0
    vals.values[1][:] = 1

    score = vals._get_range_score(current_values, current_ok, start_idx=10, past_delta=NUM_INTERVALS_PER_DAY)
    assert score

    corr = vals.xcorr_max_pearson([1, 1, 1], [2, 2, 2], Lmax=0)
    assert corr[2] == 1


def test_solar_history_vals_time_helpers(tmp_path) -> None:
    """Test time index helpers."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")

    time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    idx, days = vals.get_index_from_time(time)
    next_idx, next_days = vals.get_index_with_delta(idx, days, delta_idx=4)
    assert next_idx != idx or next_days != days

    utc_time = vals.get_utc_time_from_index(idx, days)
    assert utc_time is not None


@pytest.mark.asyncio
async def test_home_consumption_reset_forecasts(tmp_path) -> None:
    """Test reset_forecasts with mocked history values."""
    home = SimpleNamespace(
        hass=None,
        battery=SimpleNamespace(charge_discharge_sensor="sensor.battery_power"),
        solar_plant=SimpleNamespace(solar_inverter_active_power="sensor.solar_power"),
        grid_active_power_sensor="sensor.grid_power",
        grid_active_power_sensor_inverted=False,
        _childrens=[],
        _all_loads=[],
    )
    forecast = QSHomeConsumptionHistoryAndForecast(home=home, storage_path=str(tmp_path))
    forecast._all_loads = []

    class DummyHistory:
        def __init__(self, entity_id, forecast):
            self.entity_id = entity_id
            self.forecast = forecast
            self.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS))
            self.file_name = f"{entity_id}.npy"

        async def init(self, time, for_reset=False, reset_for_switch_device=None):
            return BEGINING_OF_TIME, time

        async def save_values(self, *args, **kwargs):
            return None

        def get_index_from_time(self, time):
            return 0, 0

    with patch(
        "custom_components.quiet_solar.ha_model.home.QSSolarHistoryVals",
        DummyHistory,
    ):
        result = await forecast.reset_forecasts(
            datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC),
            light_reset=False,
        )

    assert result is True


def test_forecast_value_sensor_and_segments() -> None:
    """Test forecast value sensor and segment overlap helpers."""
    now = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)

    def getter(time):
        return time, 10.0

    def getter_now(time):
        return time, 5.0

    probers = QSforecastValueSensor.get_probers(
        getter=getter,
        getter_now=getter_now,
        names_and_duration={"now": 0, "later": 3600},
    )
    assert probers["now"].push_and_get(now) == 5.0
    assert probers["later"].push_and_get(now) == 10.0

    segs_main = [(now, now + timedelta(hours=2))]
    segs_sub = [(now + timedelta(minutes=30), now + timedelta(hours=1))]
    weak_overlap = _segments_weak_sub_on_main_overlap(segs_sub, segs_main, min_overlap=60)
    assert weak_overlap

    segs_1 = [(now, now + timedelta(hours=1))]
    segs_2 = [(now + timedelta(minutes=15), now + timedelta(minutes=45))]
    strong_overlap = _segments_strong_overlap(segs_1, segs_2, min_overlap=60)
    assert strong_overlap
