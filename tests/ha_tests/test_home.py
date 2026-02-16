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
    FORCE_CAR_NO_PERSON_ATTACHED,
)
from custom_components.quiet_solar.home_model.commands import CMD_IDLE
from custom_components.quiet_solar.home_model.load import AbstractDevice
from custom_components.quiet_solar.ha_model.device import HADeviceMixin
import numpy as np

from custom_components.quiet_solar.ha_model.home import (
    QSHomeMode,
    QSHomeConsumptionHistoryAndForecast,
    QSSolarHistoryVals,
    QSforecastValueSensor,
    get_time_from_state,
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
    """Test home has no switch entities (off-grid switch was replaced by select)."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    entity_entries = er.async_entries_for_config_entry(entity_registry, home_config_entry.entry_id)
    switch_entries = [e for e in entity_entries if e.domain == "switch"]
    assert len(switch_entries) == 0


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


async def test_home_off_grid_mode_select(hass: HomeAssistant, home_config_entry: ConfigEntry) -> None:
    """Test home off-grid mode select entity exists with default auto option."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    state = hass.states.get("select.qs_test_home_home_off_grid_mode")
    assert state is not None
    assert state.state == "off_grid_mode_auto"


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

    # Set home_mode to allow solar_plant and battery properties to return the mocked values
    home.home_mode = QSHomeMode.HOME_MODE_ON.value

    home.physical_solar_plant = MagicMock(
        solar_production=3000.0,
        solar_max_output_power_value=2500.0,
        solar_max_phase_amps=10.0,
    )
    home.physical_battery = MagicMock(
        battery_can_discharge=MagicMock(return_value=True),
        get_max_discharging_power=MagicMock(return_value=2000.0),
        launch_command=AsyncMock(),
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
    load.check_commands = AsyncMock(return_value=(timedelta(seconds=0), True))
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

    with patch(
        "custom_components.quiet_solar.ha_model.home.hungarian_algorithm",
        return_value={0: 0, 1: 1},
    ):
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
    load.check_commands = AsyncMock(return_value=(timedelta(seconds=60), False))
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


def test_get_time_from_state_parses_attribute_and_string() -> None:
    """Test get_time_from_state handles attribute overrides and strings."""
    base_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    attr_time = datetime(2026, 1, 15, 10, 0, tzinfo=pytz.UTC)
    state = SimpleNamespace(
        last_updated=base_time,
        attributes={"last_updated": attr_time.isoformat()},
        state="home",
    )

    result = get_time_from_state(state)
    assert result == attr_time

    bad_state = SimpleNamespace(
        last_updated="invalid",
        attributes={"last_updated": "not-a-date"},
        state="home",
    )
    assert get_time_from_state(bad_state) is None


def test_forecast_value_sensor_handles_missing_values() -> None:
    """Test forecast value sensor returns None without stored data."""
    now = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)

    def getter(time):
        return time, None

    sensor = QSforecastValueSensor("test", 3600, getter)
    assert sensor.push_and_get(now) is None


async def test_home_map_location_path_segments(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test map_location_path builds segments for home and away."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    start = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    end = datetime(2026, 1, 15, 10, 0, tzinfo=pytz.UTC)

    def make_state(time, state, lat, lon):
        return SimpleNamespace(
            entity_id="device_tracker.test",
            state=state,
            last_updated=time,
            attributes={"latitude": lat, "longitude": lon, "source": "gps"},
        )

    person_states = [
        make_state(start, "home", 48.8566, 2.3522),
        make_state(start + timedelta(minutes=30), "not_home", 48.857, 2.36),
        make_state(start + timedelta(hours=1, minutes=30), "home", 48.8566, 2.3522),
    ]
    car_states = [
        make_state(start + timedelta(minutes=40), "not_home", 48.8575, 2.361),
        make_state(start + timedelta(hours=1, minutes=10), "home", 48.8566, 2.3522),
    ]

    gps_segments, person_not_home, car_not_home = home.map_location_path(
        person_states, car_states, start=start, end=end
    )

    assert gps_segments
    assert len(person_not_home) == 1
    assert len(car_not_home) == 1


async def test_home_compute_mileage_for_period_per_person_basic(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test mileage computation assigns distance and leave time."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    start = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    end = datetime(2026, 1, 15, 10, 0, tzinfo=pytz.UTC)

    car = MagicMock()
    car.name = "Car A"
    car.car_is_invited = False
    car.car_tracker = "device_tracker.car_a"
    car.get_car_mileage_on_period_km = AsyncMock(return_value=12.0)

    person = MagicMock()
    person.name = "Person A"
    person.authorized_cars = ["Car A"]
    person.preferred_car = "Car A"
    person.get_tracker_id = MagicMock(return_value="device_tracker.person_a")

    home._cars = [car]
    home._persons = [person]

    def make_state(entity_id, time, state, lat, lon):
        return SimpleNamespace(
            entity_id=entity_id,
            state=state,
            last_updated=time,
            attributes={"latitude": lat, "longitude": lon, "source": "gps"},
        )

    person_states = [
        make_state("device_tracker.person_a", start, "home", 48.8566, 2.3522),
        make_state("device_tracker.person_a", start + timedelta(minutes=20), "not_home", 48.857, 2.36),
        make_state("device_tracker.person_a", start + timedelta(hours=1, minutes=20), "home", 48.8566, 2.3522),
    ]
    car_states = [
        make_state("device_tracker.car_a", start + timedelta(minutes=25), "not_home", 48.8575, 2.361),
        make_state("device_tracker.car_a", start + timedelta(hours=1, minutes=10), "home", 48.8566, 2.3522),
    ]

    async def load_history_side_effect(hass_ref, entity_id, start_time, end_time, no_attributes=True):
        if entity_id == "device_tracker.person_a":
            return person_states
        if entity_id == "device_tracker.car_a":
            return car_states
        return []

    with patch(
        "custom_components.quiet_solar.ha_model.home.load_from_history",
        new=AsyncMock(side_effect=load_history_side_effect),
    ):
        result = await home._compute_mileage_for_period_per_person(start, end)

    assert person in result
    assert result[person][0] == 12.0
    assert result[person][1] is not None


async def test_home_tariff_helpers(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test tariff range and pricing helpers."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.price_off_peak = 0.1
    home.price_peak = 0.3
    home.tariff_start_1 = "22:00"
    home.tariff_end_1 = "06:00"
    home.tariff_start_2 = None
    home.tariff_end_2 = None

    now = datetime(2026, 1, 15, 21, 0, tzinfo=pytz.UTC)
    ranges = home._get_today_off_peak_ranges(now)
    assert len(ranges) == 1
    assert ranges[0][1] > ranges[0][0]

    start_peak = ranges[0][0] - timedelta(hours=1)
    end_peak = ranges[0][0] - timedelta(minutes=30)
    assert home.get_tariff(start_peak, end_peak) == home.price_peak

    start_off_peak = ranges[0][0] + timedelta(minutes=5)
    end_off_peak = ranges[0][0] + timedelta(hours=1)
    assert home.get_tariff(start_off_peak, end_off_peak) == home.price_off_peak

    tariffs = home.get_tariffs(start_peak, end_off_peak)
    assert isinstance(tariffs, list)
    assert tariffs

    with patch.object(home, "is_off_grid", return_value=True):
        assert home.get_tariffs(start_peak, end_off_peak) == 0.0


async def test_home_forecast_getters_and_compute(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test forecast getter helpers and compute_non_controlled_forecast."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    forecast_vals = SimpleNamespace(
        get_value_from_current_forecast=MagicMock(return_value=(None, 12.0)),
        get_closest_stored_value=MagicMock(return_value=(None, 8.0)),
    )
    home._consumption_forecast = SimpleNamespace(
        home_non_controlled_consumption=forecast_vals,
        init_forecasts=AsyncMock(return_value=True),
    )

    assert home.get_non_controlled_consumption_from_current_forecast_getter(
        datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    ) == (None, 12.0)
    assert home.get_non_controlled_consumption_best_stored_value_getter(
        datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    ) == (None, 8.0)

    with patch.object(home, "_compute_non_controlled_forecast_intl", return_value=[(None, 5.0)]):
        forecast = await home.compute_non_controlled_forecast(
            datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
        )

    assert forecast == [(None, 5.0)]


async def test_home_non_controlled_consumption_calculation(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test non-controlled consumption calculation with solar and battery."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.home_mode = QSHomeMode.HOME_MODE_ON.value

    battery = MagicMock()
    battery.charge_discharge_sensor = "sensor.battery_power"
    battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=100.0)
    battery.battery_get_current_possible_max_discharge_power = MagicMock(return_value=0.0)
    battery.is_dc_coupled = False

    solar = MagicMock()
    solar.solar_inverter_active_power = "sensor.solar_power"
    solar.solar_inverter_input_active_power = None
    solar.get_sensor_latest_possible_valid_value = MagicMock(return_value=500.0)
    solar.solar_max_output_power_value = 2000.0
    solar.solar_production = 0.0

    home.physical_battery = battery
    home.physical_solar_plant = solar
    home.accurate_power_sensor = None
    home._childrens = []
    home._all_loads = []

    def sensor_value(entity_id, tolerance_seconds=None, time=None):
        if entity_id == home.grid_active_power_sensor:
            return 200.0
        return None

    home.get_sensor_latest_possible_valid_value = MagicMock(side_effect=sensor_value)

    time_now = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    result = home.home_non_controlled_consumption_sensor_state_getter(
        home.home_non_controlled_consumption_sensor, time_now
    )

    assert result is not None
    assert result[1] == 300.0
    assert home.home_available_power == 300.0
    assert home.grid_consumption_power == 200.0


async def test_home_update_all_states_basic_flow(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_all_states with forecast and device updates."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home.finish_setup = AsyncMock(return_value=True)

    solar = MagicMock()
    solar.update_forecast = AsyncMock()
    home.physical_solar_plant = solar

    device_ok = MagicMock()
    device_ok.update_states = AsyncMock()
    device_ok.name = "ok"
    device_error = MagicMock()
    device_error.update_states = AsyncMock(side_effect=RuntimeError("boom"))
    device_error.name = "err"
    home._all_devices = [device_ok, device_error]

    forecast_vals = SimpleNamespace(
        add_value=MagicMock(return_value=True),
        save_values=AsyncMock(),
        update_current_forecast_if_needed=MagicMock(return_value=True),
    )
    home._consumption_forecast = SimpleNamespace(
        init_forecasts=AsyncMock(return_value=True),
        home_non_controlled_consumption=forecast_vals,
    )
    home.home_non_controlled_consumption = 123.0

    with patch.object(home, "_compute_non_controlled_forecast_intl") as mock_compute:
        await home.update_all_states(datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))

    solar.update_forecast.assert_called()
    device_ok.update_states.assert_called()
    forecast_vals.save_values.assert_called()
    mock_compute.assert_called()


async def test_home_update_loads_no_solver(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_loads runs command checks and idle launch without solver."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home.finish_setup = AsyncMock(return_value=True)
    home.update_loads_constraints = AsyncMock()
    home._last_solve_done = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)

    load = MagicMock()
    load.name = "load1"
    load.running_command_num_relaunch = 0
    load.check_commands = AsyncMock(return_value=(timedelta(seconds=60), False))
    load.force_relaunch_command = AsyncMock()
    load.is_load_active = MagicMock(return_value=True)
    load.update_live_constraints = AsyncMock(return_value=False)
    load.is_load_has_a_command_now_or_coming = MagicMock(return_value=False)
    load.get_current_active_constraint = MagicMock(return_value=None)
    load.launch_command = AsyncMock()
    load.do_probe_state_change = AsyncMock()
    load.current_command = CMD_IDLE

    home._all_loads = [load]
    home._chargers = []
    home.physical_battery = None
    home._commands = []
    home._battery_commands = []

    await home.update_loads(datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))

    load.force_relaunch_command.assert_called()
    load.launch_command.assert_called()
    load.do_probe_state_change.assert_called()


async def test_home_update_loads_finish_setup_false(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_loads returns when setup incomplete."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home.finish_setup = AsyncMock(return_value=False)

    await home.update_loads(datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))


async def test_home_update_loads_constraints_runs_loads(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_loads_constraints triggers load checks."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home.finish_setup = AsyncMock(return_value=True)
    home.force_next_solve = MagicMock()

    load = MagicMock()
    load.qs_enable_device = True
    load.do_run_check_load_activity_and_constraints = AsyncMock(return_value=True)
    disabled_load = MagicMock()
    disabled_load.qs_enable_device = False
    disabled_load.do_run_check_load_activity_and_constraints = AsyncMock(return_value=True)
    home._all_loads = [load, disabled_load]

    await home.update_loads_constraints(datetime(2026, 1, 15, 10, 0, tzinfo=pytz.UTC))

    home.force_next_solve.assert_called()


async def test_home_update_loads_constraints_early_return(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_loads_constraints returns when home mode off."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.home_mode = QSHomeMode.HOME_MODE_OFF.value
    await home.update_loads_constraints(datetime(2026, 1, 15, 10, 0, tzinfo=pytz.UTC))


async def test_home_update_loads_constraints_finish_setup_false(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_loads_constraints returns when setup incomplete."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home.finish_setup = AsyncMock(return_value=False)
    await home.update_loads_constraints(datetime(2026, 1, 15, 11, 0, tzinfo=pytz.UTC))


async def test_home_prepare_data_for_dump(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test prepare data for debug dump."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    car = SimpleNamespace(
        name="Car A",
        car_tracker="device_tracker.car_a",
        car_odometer_sensor="sensor.car_a_odo",
    )
    car_no_tracker = SimpleNamespace(
        name="Car B",
        car_tracker=None,
        car_odometer_sensor=None,
    )
    person = SimpleNamespace(
        person_entity_id="person.a",
        get_tracker_id=MagicMock(return_value="device_tracker.person_a"),
    )
    home._cars = [car, car_no_tracker]
    home._persons = [person]

    async def load_history_side_effect(hass_ref, entity_id, start_time, end_time, no_attributes=True):
        if entity_id == "device_tracker.person_a":
            return None
        return [SimpleNamespace(entity_id=entity_id, state="home", last_updated=start_time, attributes={})]

    with patch(
        "custom_components.quiet_solar.ha_model.home.load_from_history",
        new=AsyncMock(side_effect=load_history_side_effect),
    ):
        result = await home._prepare_data_for_dump(
            datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC),
            datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC),
        )

    assert result[2][0][0] == "Car A"
    assert result[2][1][0] == "Car B"
    assert result[3][0][0] == "person.a"


async def test_home_dump_person_car_data_for_debug(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    tmp_path,
) -> None:
    """Test debug dump writes via executor."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home._prepare_data_for_dump = AsyncMock(return_value=["start", "end", [], []])
    home._compute_person_needed_time_and_date = MagicMock(
        return_value=(datetime(2026, 1, 15), datetime(2026, 1, 15), datetime(2026, 1, 15, tzinfo=pytz.UTC), False)
    )
    home.hass.async_add_executor_job = AsyncMock()

    await home.dump_person_car_data_for_debug(
        datetime(2026, 1, 16, 8, 0, tzinfo=pytz.UTC),
        storage_path=str(tmp_path),
    )

    home.hass.async_add_executor_job.assert_called()


async def test_home_update_forecast_probers_flow(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test forecast probers update and recompute flow."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home.finish_setup = AsyncMock(return_value=True)
    home._consumption_forecast = SimpleNamespace(init_forecasts=AsyncMock(return_value=True))

    home.home_non_controlled_power_forecast_sensor_values_providers = {
        "now": SimpleNamespace(push_and_get=MagicMock(return_value=1.0))
    }
    home.home_solar_forecast_sensor_values_providers = {
        "now": SimpleNamespace(push_and_get=MagicMock(return_value=2.0))
    }
    home.home_non_controlled_power_forecast_sensor_values = {}
    home.home_solar_forecast_sensor_values = {}

    person = SimpleNamespace(
        name="Person A",
        should_recompute_history=MagicMock(return_value=True),
    )
    home._persons = [person]

    home.recompute_people_historical_data = AsyncMock()
    home._compute_and_store_person_car_forecasts = AsyncMock()
    home.get_best_persons_cars_allocations = AsyncMock()

    prev_time = datetime(2026, 1, 15, 2, 0, tzinfo=pytz.UTC)
    home._last_forecast_probe_time = prev_time

    prev_local_day = datetime(2026, 1, 14)
    next_local_day = datetime(2026, 1, 15)
    home._compute_person_needed_time_and_date = MagicMock(
        side_effect=[
            (prev_local_day, prev_local_day, datetime(2026, 1, 14, tzinfo=pytz.UTC), True),
            (next_local_day, next_local_day, datetime(2026, 1, 15, tzinfo=pytz.UTC), True),
        ]
    )

    await home.update_forecast_probers(datetime(2026, 1, 15, 6, 0, tzinfo=pytz.UTC))

    home.recompute_people_historical_data.assert_called()
    home._compute_and_store_person_car_forecasts.assert_called()
    home.get_best_persons_cars_allocations.assert_called()


async def test_home_update_all_states_handles_solar_exception(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_all_states handles solar forecast exceptions."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home.finish_setup = AsyncMock(return_value=True)

    solar = MagicMock()
    solar.update_forecast = AsyncMock(side_effect=RuntimeError("boom"))
    home.physical_solar_plant = solar
    home._all_devices = []

    home._consumption_forecast = SimpleNamespace(init_forecasts=AsyncMock(return_value=False))

    await home.update_all_states(datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))


async def test_home_update_forecast_probers_early_returns(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test forecast probers early return paths."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.home_mode = QSHomeMode.HOME_MODE_OFF.value
    await home.update_forecast_probers(datetime(2026, 1, 15, 6, 0, tzinfo=pytz.UTC))

    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home.finish_setup = AsyncMock(return_value=False)
    await home.update_forecast_probers(datetime(2026, 1, 15, 7, 0, tzinfo=pytz.UTC))

    home.finish_setup = AsyncMock(return_value=True)
    home._consumption_forecast = SimpleNamespace(init_forecasts=AsyncMock(return_value=False))
    await home.update_forecast_probers(datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC))


async def test_home_best_persons_cars_allocations_fallbacks_and_notify(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test allocation fallbacks and notification path."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    class HashablePerson(SimpleNamespace):
        __hash__ = object.__hash__

    person_selected = HashablePerson(
        name="Person Selected",
        preferred_car=None,
        update_person_forecast=MagicMock(return_value=(None, None)),
        get_authorized_cars=MagicMock(return_value=[]),
        notify_of_forecast_if_needed=AsyncMock(),
    )
    person_preferred = HashablePerson(
        name="Person Preferred",
        preferred_car="Car Preferred",
        update_person_forecast=MagicMock(return_value=(None, None)),
        get_authorized_cars=MagicMock(return_value=[]),
        notify_of_forecast_if_needed=AsyncMock(),
    )
    car_authorized = SimpleNamespace(name="Car Authorized")
    person_authorized = HashablePerson(
        name="Person Authorized",
        preferred_car=None,
        update_person_forecast=MagicMock(return_value=(None, None)),
        get_authorized_cars=MagicMock(return_value=[car_authorized]),
        notify_of_forecast_if_needed=AsyncMock(),
    )

    car_selected = SimpleNamespace(
        name="Car Selected",
        current_forecasted_person=None,
        user_selected_person_name_for_car="Person Selected",
        car_is_invited=False,
    )
    car_force_none = SimpleNamespace(
        name="Car Force None",
        current_forecasted_person=person_preferred,
        user_selected_person_name_for_car=FORCE_CAR_NO_PERSON_ATTACHED,
        car_is_invited=False,
    )
    car_preferred = SimpleNamespace(
        name="Car Preferred",
        current_forecasted_person=person_selected,
        user_selected_person_name_for_car=None,
        car_is_invited=False,
    )
    car_authorized = SimpleNamespace(
        name="Car Authorized",
        current_forecasted_person=None,
        user_selected_person_name_for_car=None,
        car_is_invited=False,
    )

    home._cars = [car_selected, car_force_none, car_preferred, car_authorized]
    home._persons = [person_selected, person_preferred, person_authorized]

    result = await home.get_best_persons_cars_allocations(
        time=datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC),
        force_update=True,
        do_notify=True,
    )

    assert result
    assert car_selected.current_forecasted_person == person_selected
    assert car_preferred.current_forecasted_person == person_preferred
    assert car_authorized.current_forecasted_person == person_authorized

    person_selected.notify_of_forecast_if_needed.assert_called()
    person_preferred.notify_of_forecast_if_needed.assert_called()
    person_authorized.notify_of_forecast_if_needed.assert_called()


async def test_home_update_loads_error_paths(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_loads error and branch handling."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home.finish_setup = AsyncMock(return_value=True)
    home.update_loads_constraints = AsyncMock(side_effect=RuntimeError("boom"))
    time_now = datetime(2026, 1, 15, 9, 30, tzinfo=pytz.UTC)
    home._last_solve_done = time_now
    home._commands = []
    home._battery_commands = []
    home.physical_battery = None
    home.compute_non_controlled_forecast = AsyncMock(return_value=[])
    home.get_solar_from_current_forecast = MagicMock(return_value=[])

    inactive_load = MagicMock()
    inactive_load.name = "inactive"
    inactive_load.running_command_num_relaunch = 3
    inactive_load.check_commands = AsyncMock(return_value=(timedelta(seconds=60), False))
    inactive_load.is_load_active = MagicMock(return_value=False)
    inactive_load.is_load_has_a_command_now_or_coming = MagicMock(return_value=False)
    inactive_load.get_current_active_constraint = MagicMock(return_value=None)
    inactive_load.launch_command = AsyncMock()
    inactive_load.do_probe_state_change = AsyncMock()

    active_load = MagicMock()
    active_load.name = "active"
    active_load.running_command_num_relaunch = 3
    active_load.check_commands = AsyncMock(return_value=(timedelta(seconds=60), False))
    active_load.is_load_active = MagicMock(return_value=True)
    active_load.update_live_constraints = AsyncMock(side_effect=RuntimeError("boom"))
    active_load.is_load_has_a_command_now_or_coming = MagicMock(return_value=True)
    active_load.get_current_active_constraint = MagicMock(return_value=True)
    active_load.launch_command = AsyncMock()
    active_load.do_probe_state_change = AsyncMock()
    active_load.current_command = CMD_IDLE

    home._all_loads = [inactive_load, active_load]
    home._chargers = []

    await home.update_loads(time_now)


async def test_home_update_loads_charger_only(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_loads uses charger-only mode list."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.home_mode = QSHomeMode.HOME_MODE_CHARGER_ONLY.value
    home.finish_setup = AsyncMock(return_value=True)
    home.update_loads_constraints = AsyncMock()
    home._last_solve_done = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)

    charger = MagicMock()
    charger.name = "charger"
    charger.running_command_num_relaunch = 0
    charger.check_commands = AsyncMock(return_value=(timedelta(seconds=0), True))
    charger.is_load_active = MagicMock(return_value=False)
    charger.is_load_has_a_command_now_or_coming = MagicMock(return_value=False)
    charger.get_current_active_constraint = MagicMock(return_value=None)
    charger.launch_command = AsyncMock()
    charger.do_probe_state_change = AsyncMock()
    charger.current_command = CMD_IDLE

    home._chargers = [charger]
    home._all_loads = []
    home.physical_battery = None
    home._commands = []
    home._battery_commands = []

    await home.update_loads(datetime(2026, 1, 15, 9, 45, tzinfo=pytz.UTC))


async def test_home_finish_setup_initializes_devices(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test finish_setup initializes devices and updates states."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    class DummyDevice(HADeviceMixin, AbstractDevice):
        def __init__(self):
            self._enabled = True
            self.home = None
            self.config_entry_initialized = True
            self.root_device_post_home_init = MagicMock()
            self.update_states = AsyncMock()
            self.name = "Dummy"

    dummy_device = DummyDevice()
    error_device = DummyDevice()
    error_device.name = "Boom"
    error_device.update_states = AsyncMock(side_effect=RuntimeError("boom"))
    home._all_devices = [dummy_device]
    home._all_loads = [SimpleNamespace(externally_initialized_constraints=True)]
    home._disabled_devices = []
    home._init_completed = False

    home._all_devices.append(error_device)
    result = await home.finish_setup(datetime(2026, 1, 15, 10, 0, tzinfo=pytz.UTC))

    assert result is True
    dummy_device.root_device_post_home_init.assert_called()
    dummy_device.update_states.assert_called()


async def test_home_recompute_people_historical_data_delta_branch(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test recompute_people_historical_data when not passed limit."""
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
    home._compute_person_needed_time_and_date = MagicMock(
        return_value=(datetime(2026, 1, 15), datetime(2026, 1, 15), datetime(2026, 1, 15, tzinfo=pytz.UTC), False)
    )

    await home.recompute_people_historical_data(datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC))

    assert person.has_been_initialized is True


async def test_home_best_persons_cars_allocations_cost_matrix_branches(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test allocation cost matrix branches."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    car_main = SimpleNamespace(
        name="Car Main",
        current_forecasted_person=None,
        user_selected_person_name_for_car=None,
        car_is_invited=False,
        get_adapt_target_percent_soc_to_reach_range_km=MagicMock(
            return_value=(False, 40.0, 80.0, 10.0)
        ),
    )
    car_unused = SimpleNamespace(
        name="Car Unused",
        current_forecasted_person=None,
        user_selected_person_name_for_car=None,
        car_is_invited=False,
        get_adapt_target_percent_soc_to_reach_range_km=MagicMock(
            return_value=(False, 40.0, 80.0, 10.0)
        ),
    )

    person = SimpleNamespace(
        name="Person A",
        preferred_car=None,
        update_person_forecast=MagicMock(
            return_value=(None, 30.0)
        ),
        get_authorized_cars=MagicMock(return_value=[car_main, SimpleNamespace(name="Other")]),
    )

    home._cars = [car_main, car_unused]
    home._persons = [person]

    with patch(
        "custom_components.quiet_solar.ha_model.home.hungarian_algorithm",
        return_value={0: 0},
    ):
        await home.get_best_persons_cars_allocations(
            time=datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC),
            force_update=True,
            do_notify=False,
        )

    assert car_main.current_forecasted_person is person


async def test_home_best_persons_cars_allocations_no_allocations(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test allocation with no persons or cars."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home._cars = []
    home._persons = []

    result = await home.get_best_persons_cars_allocations(
        time=None,
        force_update=True,
        do_notify=False,
    )

    assert result == {}


async def test_home_update_all_states_device_error_logging(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_all_states handles AbstractDevice errors."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home.finish_setup = AsyncMock(return_value=True)
    home.physical_solar_plant = None

    class DummyAbstract(AbstractDevice):
        async def update_states(self, time):  # type: ignore[override]
            raise RuntimeError("boom")

    device = DummyAbstract(name="Device", device_type=None, home=home)
    home._all_devices = [device]
    home._consumption_forecast = SimpleNamespace(init_forecasts=AsyncMock(return_value=False))

    await home.update_all_states(datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))


async def test_home_update_loads_home_mode_sensors_only(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_loads returns when in sensors-only mode."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home.home_mode = QSHomeMode.HOME_MODE_SENSORS_ONLY.value
    await home.update_loads(datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))


async def test_home_best_persons_cars_allocations_skip_cases(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test allocation skip branches for invited and preset cars."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    invited_car = SimpleNamespace(
        name="Invited",
        current_forecasted_person=None,
        user_selected_person_name_for_car=None,
        car_is_invited=True,
    )
    preset_car = SimpleNamespace(
        name="Preset",
        current_forecasted_person=SimpleNamespace(name="PresetPerson"),
        user_selected_person_name_for_car=None,
        car_is_invited=False,
    )
    home._cars = [invited_car, preset_car]
    home._persons = []

    result = await home.get_best_persons_cars_allocations(
        time=datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC),
        force_update=True,
        do_notify=False,
    )

    assert result == {}


async def test_home_finish_setup_returns_false_on_uninitialized_load(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test finish_setup returns False when constraints uninitialized."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    home._all_devices = [SimpleNamespace(config_entry_initialized=True)]
    home._all_loads = [SimpleNamespace(externally_initialized_constraints=False)]
    home._disabled_devices = []
    home._init_completed = False

    result = await home.finish_setup(datetime(2026, 1, 15, 10, 0, tzinfo=pytz.UTC))
    assert result is False


async def test_home_sensor_state_getters_and_battery_helpers(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test sensor state getters and battery helpers."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home
    home.home_mode = QSHomeMode.HOME_MODE_ON.value

    time_now = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    home.home_consumption = 50.0
    home.home_available_power = 10.0
    home.grid_consumption_power = 5.0

    assert home.home_consumption_sensor_state_getter("x", time_now)[1] == 50.0
    assert home.home_available_power_sensor_state_getter("x", time_now)[1] == 10.0
    assert home.grid_consumption_power_sensor_state_getter("x", time_now)[1] == 5.0

    battery = MagicMock()
    battery.charge_discharge_sensor = "sensor.battery_power"
    battery.get_state_history_data = MagicMock(return_value=[(time_now, 1.0, {})])
    battery.battery_can_discharge = MagicMock(return_value=True)
    home.physical_battery = battery

    assert home.battery_can_discharge() is True
    assert home.get_battery_charge_values(60, time_now)


async def test_home_power_helpers(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test power helper computations."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home
    home.home_mode = QSHomeMode.HOME_MODE_ON.value

    solar = SimpleNamespace(
        solar_production=1500.0,
        solar_max_output_power_value=1000.0,
    )
    home.physical_solar_plant = solar
    assert home.get_current_over_clamp_production_power() == 500.0

    battery = SimpleNamespace(
        is_dc_coupled=True,
        battery_get_current_possible_max_discharge_power=MagicMock(return_value=200.0),
    )
    home.physical_battery = battery
    solar.solar_production = 300.0
    solar.solar_max_output_power_value = 2000.0
    assert home.get_current_maximum_production_output_power() == 500.0


async def test_home_forecast_getters_without_sources(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test forecast getters when no forecast sources exist."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    assert home.get_non_controlled_consumption_from_current_forecast_getter(
        datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    ) == (None, None)
    assert home.get_non_controlled_consumption_best_stored_value_getter(
        datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    ) == (None, None)
    assert home.get_solar_from_current_forecast_getter(
        datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    ) == (None, None)
    assert home.get_solar_from_current_forecast(
        datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    ) == []
