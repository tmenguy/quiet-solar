"""Tests for quiet_solar car.py functionality."""

import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock

import pytz
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_CAR_BATTERY_CAPACITY,
    CONF_DEFAULT_CAR_CHARGE,
    CONF_MINIMUM_OK_CAR_CHARGE,
    CONF_PERSON_AUTHORIZED_CARS,
    CONF_PERSON_PREFERRED_CAR,
    CAR_CHARGE_TYPE_PERSON_AUTOMATED,
    FORCE_CAR_NO_PERSON_ATTACHED,
)
from custom_components.quiet_solar.home_model.constraints import DATETIME_MAX_UTC


pytestmark = pytest.mark.usefixtures("mock_sensor_states")


async def test_car_initialization(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car device initialization."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_init_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_init_test",
    )
    car_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    assert car_entry.state is ConfigEntryState.LOADED

    # Verify car device was created
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    assert data_handler is not None

    # Find the car device
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    assert car_device is not None
    assert car_device.name == MOCK_CAR_CONFIG['name']


async def test_car_battery_capacity(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car battery capacity configuration."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_capacity_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_capacity_test",
    )
    car_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    assert car_device.car_battery_capacity == MOCK_CAR_CONFIG[CONF_CAR_BATTERY_CAPACITY]


async def test_car_default_charge_percent(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car default charge percent configuration."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_default_charge_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_default_charge_test",
    )
    car_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    assert car_device.car_default_charge == MOCK_CAR_CONFIG[CONF_DEFAULT_CAR_CHARGE]
    assert car_device.car_minimum_ok_charge == MOCK_CAR_CONFIG[CONF_MINIMUM_OK_CAR_CHARGE]


async def test_car_charger_amp_limits(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car charger amperage limits configuration."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_amp_limits_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_amp_limits_test",
    )
    car_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    assert car_device.car_charger_min_charge == 6
    assert car_device.car_charger_max_charge == 32


async def test_car_soc_sensor_state(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car SOC sensor state reading."""
    from .const import MOCK_CAR_CONFIG

    # Set up mock SOC sensor
    hass.states.async_set("sensor.test_car_soc", "75", {"unit_of_measurement": "%"})

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_soc_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_soc_test",
    )
    car_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    assert car_device is not None


async def test_car_tracker_state(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car tracker state reading."""
    from .const import MOCK_CAR_CONFIG

    # Set up mock tracker
    hass.states.async_set("device_tracker.test_car", "home", {"latitude": 48.8566, "longitude": 2.3522})

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_tracker_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_tracker_test",
    )
    car_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    assert car_device.car_tracker == "device_tracker.test_car"


async def test_car_plugged_state(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car plugged state reading."""
    from .const import MOCK_CAR_CONFIG

    # Set up mock plugged sensor
    hass.states.async_set("binary_sensor.test_car_plugged", "on")

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_plugged_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_plugged_test",
    )
    car_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    assert car_device.car_plugged == "binary_sensor.test_car_plugged"


async def test_car_amp_to_power_calculation(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car amp to power conversion calculation."""
    from .const import MOCK_CAR_CONFIG, MOCK_HOME_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_amp_power_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_amp_power_test",
    )
    car_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    # Test theoretical power calculation (P = V * I)
    # For single phase at 230V:
    voltage = MOCK_HOME_CONFIG.get("home_voltage", 230)
    expected_power_10a_1p = voltage * 10  # 2300W at 10A
    expected_power_10a_3p = 3 * voltage * 10  # 6900W at 10A 3-phase

    assert car_device.theoretical_amp_to_power_1p[10] == expected_power_10a_1p
    assert car_device.theoretical_amp_to_power_3p[10] == expected_power_10a_3p


async def test_car_with_charger_connection(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car connection to charger."""
    from .const import MOCK_CAR_CONFIG, MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create charger first
    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_for_car_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_for_car_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_with_charger_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_with_charger_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    assert car_entry.state is ConfigEntryState.LOADED
    assert charger_entry.state is ConfigEntryState.LOADED


async def test_car_select_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test car select entities are created."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_select_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_select_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Check select entities
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, car_entry.entry_id
    )

    select_entries = [e for e in entity_entries if e.domain == "select"]

    # Should have at least: charger selection, charge limit, person selection
    assert len(select_entries) >= 2, f"Expected at least 2 select entities, got {len(select_entries)}"


async def test_car_button_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test car button entities are created."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_button_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_button_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Check button entities
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, car_entry.entry_id
    )

    button_entries = [e for e in entity_entries if e.domain == "button"]

    # Should have buttons: force charge, add charge, reset, clean constraints
    assert len(button_entries) >= 2, f"Expected at least 2 button entities, got {len(button_entries)}"


async def test_car_switch_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test car switch entities are created."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_switch_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_switch_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Check switch entities
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, car_entry.entry_id
    )

    switch_entries = [e for e in entity_entries if e.domain == "switch"]

    # Should have at least bump solar charge priority switch
    assert len(switch_entries) >= 1, f"Expected at least 1 switch entity, got {len(switch_entries)}"


async def test_car_time_entity(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test car time entity is created."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_time_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_time_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Check time entities
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, car_entry.entry_id
    )

    time_entries = [e for e in entity_entries if e.domain == "time"]

    # Should have default charge time
    assert len(time_entries) >= 1, f"Expected at least 1 time entity, got {len(time_entries)}"


async def test_multiple_cars(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test multiple cars can be created."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create first car
    car1_config = {**MOCK_CAR_CONFIG, "name": "Car 1"}
    car1_entry = MockConfigEntry(
        domain=DOMAIN,
        data=car1_config,
        entry_id="car1_multi_test",
        title="car: Car 1",
        unique_id="quiet_solar_car1_multi_test",
    )
    car1_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car1_entry.entry_id)
    await hass.async_block_till_done()

    # Create second car
    car2_config = {**MOCK_CAR_CONFIG, "name": "Car 2"}
    car2_entry = MockConfigEntry(
        domain=DOMAIN,
        data=car2_config,
        entry_id="car2_multi_test",
        title="car: Car 2",
        unique_id="quiet_solar_car2_multi_test",
    )
    car2_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car2_entry.entry_id)
    await hass.async_block_till_done()

    assert car1_entry.state is ConfigEntryState.LOADED
    assert car2_entry.state is ConfigEntryState.LOADED

    # Verify both cars exist in data handler
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    assert data_handler.home is not None


async def test_car_get_platforms(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car returns correct platforms."""
    from .const import MOCK_CAR_CONFIG
    from homeassistant.const import Platform

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_platforms_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_platforms_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    platforms = car_device.get_platforms()

    assert Platform.SENSOR in platforms
    assert Platform.SELECT in platforms
    assert Platform.SWITCH in platforms
    assert Platform.BUTTON in platforms
    assert Platform.TIME in platforms


async def test_car_get_charge_type_no_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car charge type when no charger is connected."""
    from .const import MOCK_CAR_CONFIG
    from custom_components.quiet_solar.const import CAR_CHARGE_TYPE_NOT_PLUGGED

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_charge_type_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_charge_type_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    # Without charger connected, should return NOT_PLUGGED
    charge_type = car_device.get_car_charge_type()
    assert charge_type == CAR_CHARGE_TYPE_NOT_PLUGGED


async def test_car_charge_time_no_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car charge time when no charger is connected."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_charge_time_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_charge_time_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    # Without charger, should return default time
    charge_time = car_device.get_car_charge_time_readable_name()
    assert charge_time == "--:--"


async def test_car_persons_options_empty(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car persons options when no persons are configured."""
    from .const import MOCK_CAR_CONFIG
    from custom_components.quiet_solar.const import FORCE_CAR_NO_PERSON_ATTACHED

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_persons_options_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_persons_options_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    options = car_device.get_car_persons_options()
    # Should at least have the "no person" option
    assert FORCE_CAR_NO_PERSON_ATTACHED in options


async def test_car_dashboard_sort_string(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car dashboard sort string."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Test regular car (not invited)
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_sort_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_sort_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    # Regular car should sort first
    assert car_device.dashboard_sort_string_in_type == "AAA"


async def test_car_person_forecast_no_person(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car person forecast when no person is assigned."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_forecast_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_forecast_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    # Without person, should return no forecast message
    forecast = car_device.get_car_person_readable_forecast_mileage()
    assert forecast == "No forecasted person"


async def test_car_unload(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car unload."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_unload_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_unload_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    assert car_entry.state is ConfigEntryState.LOADED

    # Unload
    await hass.config_entries.async_unload(car_entry.entry_id)
    await hass.async_block_till_done()

    assert car_entry.state is ConfigEntryState.NOT_LOADED


async def test_car_is_3_phase(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car 3 phase configuration."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_3phase_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_3phase_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    # Check device exists and has required attributes
    assert car_device is not None
    assert hasattr(car_device, 'car_battery_capacity')


async def test_car_conf_type_name(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car conf_type_name class attribute."""
    from custom_components.quiet_solar.ha_model.car import QSCar
    from custom_components.quiet_solar.const import CONF_TYPE_NAME_QSCar

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert QSCar.conf_type_name == CONF_TYPE_NAME_QSCar


async def test_car_min_charge(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car minimum charge configuration."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_min_charge_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_min_charge_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    # Car should have minimum ok charge value
    assert hasattr(car_device, 'car_minimum_ok_charge')


async def test_car_charger_connection_property(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car charger connection property."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_charger_prop_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_charger_prop_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    # Car should have charger property (may be None when not connected)
    assert hasattr(car_device, 'charger')


async def test_car_current_forecasted_person(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car current_forecasted_person property."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_forecast_person_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_forecast_person_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    # Without person, should be None
    assert car_device.current_forecasted_person is None


async def test_car_user_selected_person(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car user_selected_person_name_for_car property."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_user_person_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_user_person_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    # Initially should be None
    assert car_device.user_selected_person_name_for_car is None


async def test_car_is_invited(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car_is_invited property."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_invited_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_invited_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    # Default should be False
    assert car_device.car_is_invited == MOCK_CAR_CONFIG.get("car_is_invited", False)


async def test_car_person_options_filters_authorized_persons(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car person options include only authorized persons."""
    from .const import MOCK_CAR_CONFIG, MOCK_PERSON_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_person_options_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_person_options_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    person1_config = {**MOCK_PERSON_CONFIG, "name": "Person A"}
    person2_config = {
        **MOCK_PERSON_CONFIG,
        "name": "Person B",
        CONF_PERSON_AUTHORIZED_CARS: ["Other Car"],
        CONF_PERSON_PREFERRED_CAR: "Other Car",
    }

    person1_entry = MockConfigEntry(
        domain=DOMAIN,
        data=person1_config,
        entry_id="person_a_for_car_options",
        title="person: Person A",
        unique_id="quiet_solar_person_a_for_car_options",
    )
    person1_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person1_entry.entry_id)
    await hass.async_block_till_done()

    person2_entry = MockConfigEntry(
        domain=DOMAIN,
        data=person2_config,
        entry_id="person_b_for_car_options",
        title="person: Person B",
        unique_id="quiet_solar_person_b_for_car_options",
    )
    person2_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person2_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    options = car_device.get_car_persons_options()

    assert "Person A" in options
    assert "Person B" not in options
    assert FORCE_CAR_NO_PERSON_ATTACHED in options


async def test_car_set_user_person_for_car_updates_other_cars(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test setting a person for one car clears others."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car1_entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_CAR_CONFIG, "name": "Car One"},
        entry_id="car_one_person_test",
        title="car: Car One",
        unique_id="quiet_solar_car_one_person_test",
    )
    car1_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car1_entry.entry_id)
    await hass.async_block_till_done()

    car2_entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_CAR_CONFIG, "name": "Car Two"},
        entry_id="car_two_person_test",
        title="car: Car Two",
        unique_id="quiet_solar_car_two_person_test",
    )
    car2_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car2_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    data_handler.home.get_best_persons_cars_allocations = AsyncMock(return_value={})

    car1_device = hass.data[DOMAIN].get(car1_entry.entry_id)
    car2_device = hass.data[DOMAIN].get(car2_entry.entry_id)
    car2_device._user_selected_person_name_for_car = "Person A"

    await car1_device.set_user_person_for_car("Person A")

    assert car1_device.user_selected_person_name_for_car == "Person A"
    assert car2_device.user_selected_person_name_for_car is None


async def test_car_device_post_home_init_restores_person(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test device_post_home_init restores selected and forecasted person."""
    from .const import MOCK_CAR_CONFIG, MOCK_PERSON_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_post_init_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_post_init_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_PERSON_CONFIG, "name": "Person Restore"},
        entry_id="person_restore_test",
        title="person: Person Restore",
        unique_id="quiet_solar_person_restore_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    car_device.use_saved_extra_device_info(
        {
            "user_selected_person_name_for_car": "Person Restore",
            "current_forecasted_person_name_from_boot": "Person Restore",
        }
    )

    car_device.device_post_home_init(datetime.now(tz=pytz.UTC))

    assert car_device.current_forecasted_person is not None
    assert car_device.current_forecasted_person.name == "Person Restore"


async def test_car_charge_time_readable_name_with_constraint(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car charge time readable name from charger constraint."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_charge_time_constraint",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_charge_time_constraint",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    constraint = SimpleNamespace(
        get_readable_next_target_date_string=lambda for_small_standalone: "12:30"
    )
    car_device.charger = SimpleNamespace(
        get_current_active_constraint=lambda time: constraint
    )

    assert car_device.get_car_charge_time_readable_name() == "12:30"


async def test_car_convert_auto_constraint_to_manual(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test auto constraint conversion triggers manual charge."""
    from .const import MOCK_CAR_CONFIG, MOCK_PERSON_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_auto_constraint_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_auto_constraint_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_auto_constraint_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_auto_constraint_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    car_device.current_forecasted_person = hass.data[DOMAIN].get(person_entry.entry_id)

    constraint = SimpleNamespace(end_of_constraint=datetime.now(tz=pytz.UTC))
    car_device.charger = SimpleNamespace(
        get_charge_type=lambda: (CAR_CHARGE_TYPE_PERSON_AUTOMATED, constraint)
    )
    car_device.user_add_default_charge_at_datetime = AsyncMock(return_value=True)

    result = await car_device.convert_auto_constraint_to_manual_if_needed()

    assert result is True
    assert car_device.user_selected_person_name_for_car == MOCK_PERSON_CONFIG["name"]


async def test_car_efficiency_from_soc_and_odometer(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test efficiency calculation from SOC and odometer."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_efficiency_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_efficiency_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    car_device.car_odometer_sensor = "sensor.car_odometer"
    car_device.car_charge_percent_sensor = "sensor.test_car_soc"
    car_device.car_estimated_range_sensor = None

    time1 = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    car_device.get_car_charge_percent = MagicMock(return_value=80.0)
    car_device.get_car_odometer_km = MagicMock(return_value=1000.0)

    car_device.car_efficiency_km_per_kwh_sensor_state_getter(
        "sensor.car_efficiency", time1
    )

    time2 = datetime(2026, 1, 15, 18, 0, tzinfo=pytz.UTC)
    car_device.get_car_charge_percent = MagicMock(return_value=70.0)
    car_device.get_car_odometer_km = MagicMock(return_value=1010.0)

    result = car_device.car_efficiency_km_per_kwh_sensor_state_getter(
        "sensor.car_efficiency", time2
    )

    assert result is not None
    assert result[1] is not None


async def test_car_add_soc_odo_segments(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test SOC/odometer segment tracking branches."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_segments_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_segments_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    car_device._decreasing_segments = []
    car_device._dec_seg_count = 0

    base_time = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    car_device._add_soc_odo_value_to_segments(80.0, 1000.0, base_time)
    car_device._add_soc_odo_value_to_segments(79.0, 1001.0, base_time + timedelta(hours=1))
    car_device._add_soc_odo_value_to_segments(79.0, 1001.0, base_time + timedelta(hours=2))
    car_device._add_soc_odo_value_to_segments(81.0, 1002.0, base_time + timedelta(hours=3))
    car_device._add_soc_odo_value_to_segments(80.0, 1003.0, base_time + timedelta(hours=4))

    assert len(car_device._decreasing_segments) >= 1


async def test_car_mileage_from_odometer_history(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test mileage computation using odometer history."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_mileage_odo_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_mileage_odo_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    car_device.car_odometer_sensor = "sensor.car_odometer"

    from_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    to_time = datetime(2026, 1, 15, 12, 0, tzinfo=pytz.UTC)

    states = [
        SimpleNamespace(state="1000", last_changed=from_time - timedelta(hours=1), attributes={}),
        SimpleNamespace(state="1010", last_changed=from_time + timedelta(minutes=30), attributes={}),
        SimpleNamespace(state="1020", last_changed=to_time, attributes={}),
    ]

    with patch(
        "custom_components.quiet_solar.ha_model.car.load_from_history",
        new=AsyncMock(return_value=states),
    ):
        res = await car_device.get_car_mileage_on_period_km(from_time, to_time)

    assert res == 20.0


async def test_car_mileage_from_tracker_history(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test mileage computation using tracker history."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_mileage_tracker_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_mileage_tracker_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    car_device.car_odometer_sensor = None
    car_device.car_tracker = "device_tracker.test_car"

    from_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    to_time = datetime(2026, 1, 15, 12, 0, tzinfo=pytz.UTC)

    positions = [
        SimpleNamespace(state="home", last_changed=from_time, attributes={"latitude": 48.8566, "longitude": 2.3522}),
        SimpleNamespace(state="away", last_changed=to_time, attributes={"latitude": 48.8666, "longitude": 2.3622}),
    ]

    with patch(
        "custom_components.quiet_solar.ha_model.car.load_from_history",
        new=AsyncMock(return_value=positions),
    ):
        res = await car_device.get_car_mileage_on_period_km(from_time, to_time)

    assert res is not None
    assert res > 0.0


async def test_car_person_forecast_readable_variants(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test readable forecast string branches."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_forecast_readable_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_forecast_readable_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    person = SimpleNamespace(
        name="Forecast Person",
        predicted_mileage=50.0,
        predicted_leave_time=datetime(2026, 1, 16, 8, 0, tzinfo=pytz.UTC),
        get_forecast_readable_string=lambda: "50km tomorrow",
    )
    car_device.current_forecasted_person = person
    car_device.get_adapt_target_percent_soc_to_reach_range_km = MagicMock(
        return_value=(True, 70.0, 60.0, None)
    )
    assert "OK!" in car_device.get_car_person_readable_forecast_mileage()

    car_device.get_adapt_target_percent_soc_to_reach_range_km = MagicMock(
        return_value=(False, 40.0, 80.0, None)
    )
    assert "Need charge" in car_device.get_car_person_readable_forecast_mileage()

    car_device.get_adapt_target_percent_soc_to_reach_range_km = MagicMock(
        return_value=(None, None, None, None)
    )
    assert "UNKNOWN" in car_device.get_car_person_readable_forecast_mileage()


async def test_car_bootstrap_efficiency_from_history(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test bootstrap efficiency calculation from history."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_efficiency_bootstrap_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_efficiency_bootstrap_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    car_device.car_odometer_sensor = "sensor.car_odometer"
    car_device.car_charge_percent_sensor = "sensor.car_soc"
    car_device.car_battery_capacity = 60000

    base_time = datetime(2026, 1, 10, 8, 0, tzinfo=pytz.UTC)
    odos = [
        SimpleNamespace(state="1000", last_changed=base_time, attributes={}),
        SimpleNamespace(state="1010", last_changed=base_time + timedelta(days=1), attributes={}),
        SimpleNamespace(state="1020", last_changed=base_time + timedelta(days=2), attributes={}),
    ]
    socs = [
        SimpleNamespace(state="80", last_changed=base_time, attributes={}),
        SimpleNamespace(state="70", last_changed=base_time + timedelta(days=1), attributes={}),
        SimpleNamespace(state="60", last_changed=base_time + timedelta(days=2), attributes={}),
    ]

    def fake_state_changes(hass_obj, start_time, end_time, entity_id, **kwargs):
        if entity_id == "sensor.car_odometer":
            return {entity_id: odos}
        if entity_id == "sensor.car_soc":
            return {entity_id: socs}
        return {entity_id: []}

    class DummyRecorder:
        async def async_add_executor_job(self, func, *args):
            return func(*args)

    with patch(
        "custom_components.quiet_solar.ha_model.car.state_changes_during_period",
        new=fake_state_changes,
    ), patch(
        "custom_components.quiet_solar.ha_model.car.recorder_get_instance",
        return_value=DummyRecorder(),
    ):
        await car_device._async_bootstrap_efficiency_from_history(
            datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
        )

    assert car_device._km_per_kwh is not None


async def test_car_user_charge_actions(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test user charge actions and options."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_user_actions_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_user_actions_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    car_device.charger = SimpleNamespace(update_charger_for_user_change=AsyncMock())
    car_device.default_charge_time = datetime(2026, 1, 15, 7, 30, tzinfo=pytz.UTC).time()
    car_device.get_next_time_from_hours = MagicMock(
        return_value=datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    )

    await car_device.user_add_default_charge()
    assert car_device.do_force_next_charge is False
    car_device.charger.update_charger_for_user_change.assert_awaited()

    car_device.current_forecasted_person = SimpleNamespace(name="Person A")
    await car_device.user_force_charge_now()
    assert car_device.do_force_next_charge is True


async def test_car_charge_power_per_phase(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charge power steps for 1p/3p."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_power_steps_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_power_steps_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    steps_3p, min_charge, max_charge = car_device.get_charge_power_per_phase_A(True)
    steps_1p, _, _ = car_device.get_charge_power_per_phase_A(False)

    assert steps_3p[min_charge] >= 0
    assert steps_1p[max_charge] >= 0


async def test_car_charge_percent_constraints_flags(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charge percent constraints availability."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_percent_constraints_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_percent_constraints_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    car_device.car_battery_capacity = None
    assert car_device.can_use_charge_percent_constraints() is False

    car_device.car_battery_capacity = 50000
    car_device.car_charge_percent_sensor = None
    assert car_device.can_use_charge_percent_constraints() is False

    car_device.car_charge_percent_sensor = "sensor.car_soc"
    assert car_device.can_use_charge_percent_constraints() is True


async def test_car_user_add_default_charge_time_validation(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test default charge time validation."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_default_charge_time_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_default_charge_time_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    car_device.charger = SimpleNamespace(update_charger_for_user_change=AsyncMock())

    result = await car_device.user_add_default_charge_at_dt_time(None)
    assert result is False

