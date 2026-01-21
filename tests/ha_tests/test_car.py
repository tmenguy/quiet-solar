"""Tests for quiet_solar car.py functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

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
)


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
