"""Tests for sensor platform."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import pytz

from homeassistant.const import STATE_UNAVAILABLE

from custom_components.quiet_solar.sensor import (
    create_ha_sensor,
    create_ha_sensor_for_QSHome,
    create_ha_sensor_for_QSCar,
    create_ha_sensor_for_Load,
    QSBaseSensor,
    QSLoadSensorCurrentConstraints,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.quiet_solar.const import DOMAIN
from tests.factories import create_minimal_home_model
from tests.test_helpers import create_mock_device


def test_create_ha_sensor_for_home():
    """Test creating sensors for home device."""
    mock_home = create_mock_device("home", name="Test Home")
    mock_home.data_handler = MagicMock()
    mock_home.home_non_controlled_power_forecast_sensor_values = {}
    mock_home.home_solar_forecast_sensor_values = {}
    
    entities = create_ha_sensor_for_QSHome(mock_home)
    
    assert len(entities) > 0
    # Should create home consumption, available power, and forecast sensors
    assert any("home_non_controlled_consumption" in e.entity_description.key for e in entities)
    assert any("home_consumption" in e.entity_description.key for e in entities)
    assert any("home_available_power" in e.entity_description.key for e in entities)


def test_create_ha_sensor_for_car():
    """Test creating sensors for car device."""
    mock_car = create_mock_device("car", name="Test Car")
    mock_car.data_handler = MagicMock()
    mock_car.charger = None
    mock_car.get_car_charge_percent = MagicMock(return_value=80)
    mock_car.get_car_charge_type = MagicMock(return_value="Solar")
    mock_car.get_car_charge_time_readable_name = MagicMock(return_value="2 hours")
    
    mock_car.get_estimated_range_km = MagicMock(return_value=260)
    mock_car.get_autonomy_to_target_soc_km = MagicMock(return_value=50)

    entities = create_ha_sensor_for_QSCar(mock_car)
    
    assert len(entities) > 0
    # Should create SOC, charge type, charge time, and range sensors
    assert any("car_soc_percentage" in e.entity_description.key for e in entities)
    assert any("car_charge_type" in e.entity_description.key for e in entities)
    assert any("car_estimated_range_km" in e.entity_description.key for e in entities)
    assert any("car_autonomy_to_target_soc_km" in e.entity_description.key for e in entities)


def test_create_ha_sensor_for_load():
    """Test creating sensors for load device."""
    from custom_components.quiet_solar.home_model.load import AbstractLoad
    from custom_components.quiet_solar.ha_model.device import HADeviceMixin
    
    mock_load = MagicMock(spec=[AbstractLoad, HADeviceMixin])
    mock_load.data_handler = MagicMock()
    mock_load.device_id = "test_load_123"
    mock_load.device_type = "load"
    mock_load.name = "Test Load"
    mock_load.qs_enable_device = True
    mock_load.current_command = None
    mock_load.support_user_override = MagicMock(return_value=True)
    mock_load.get_override_state = MagicMock(return_value="off")
    mock_load.get_virtual_current_constraint_translation_key = MagicMock(return_value="constraint")
    
    entities = create_ha_sensor_for_Load(mock_load)
    
    assert len(entities) > 0
    # Should create  override state sensor
    assert any("load_override_state" in e.entity_description.key for e in entities)


def test_create_ha_sensor_car_without_charger():
    """Test car sensor with no charger attached."""
    mock_car = create_mock_device("car", name="Test Car")
    mock_car.data_handler = MagicMock()
    mock_car.charger = None
    mock_car.get_car_charge_percent = MagicMock(return_value=50)
    mock_car.get_car_charge_type = MagicMock(return_value="Not Charging")
    mock_car.get_car_charge_time_readable_name = MagicMock(return_value="N/A")
    
    entities = create_ha_sensor_for_QSCar(mock_car)
    
    # Should still create entities even without charger
    assert len(entities) > 0


def test_qs_base_sensor_init():
    """Test QSBaseSensor initialization."""
    mock_handler = MagicMock()
    mock_device = create_mock_device("test")

    from custom_components.quiet_solar.sensor import QSSensorEntityDescription

    mock_description = QSSensorEntityDescription(
        key="test_sensor",
        translation_key="test",
        qs_is_none_unavailable=False,
    )
    
    sensor = QSBaseSensor(mock_handler, mock_device, mock_description)
    
    assert sensor.device == mock_device
    assert sensor.entity_description == mock_description


def test_qs_base_sensor_update_with_value():
    """Test sensor update with valid value."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.test_value = 42.5


    from custom_components.quiet_solar.sensor import QSSensorEntityDescription

    mock_description = QSSensorEntityDescription(
        key="test_value",
        translation_key="test",
        qs_is_none_unavailable=False,
        value_fn=None
    )

    
    sensor = QSBaseSensor(mock_handler, mock_device, mock_description)
    sensor.async_write_ha_state = MagicMock()
    
    test_time = datetime.now(pytz.UTC)
    sensor.async_update_callback(test_time)
    
    assert sensor._attr_native_value == 42.5
    sensor.async_write_ha_state.assert_called_once()


def test_qs_base_sensor_update_with_none_not_unavailable():
    """Test sensor update with None value when not marked unavailable."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.test_value = None

    from custom_components.quiet_solar.sensor import QSSensorEntityDescription

    mock_description = QSSensorEntityDescription(
        key="test_value",
        translation_key="test",
        qs_is_none_unavailable=False,
        value_fn=None
    )
    
    sensor = QSBaseSensor(mock_handler, mock_device, mock_description)
    sensor.async_write_ha_state = MagicMock()
    
    test_time = datetime.now(pytz.UTC)
    sensor.async_update_callback(test_time)
    
    sensor.async_write_ha_state.assert_called_once()


def test_qs_base_sensor_update_with_none_unavailable():
    """Test sensor becomes unavailable when value is None and marked unavailable."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.test_value = None

    from custom_components.quiet_solar.sensor import QSSensorEntityDescription

    mock_description = QSSensorEntityDescription(
        key="test_value",
        translation_key="test",
        qs_is_none_unavailable=True,
        value_fn=None
    )

    
    sensor = QSBaseSensor(mock_handler, mock_device, mock_description)
    sensor.async_write_ha_state = MagicMock()
    
    test_time = datetime.now(pytz.UTC)
    sensor.async_update_callback(test_time)
    
    assert sensor._attr_available is False
    assert sensor._attr_native_value == STATE_UNAVAILABLE


def test_qs_base_sensor_update_with_value_fn():
    """Test sensor update using value function."""

    from custom_components.quiet_solar.sensor import QSSensorEntityDescription

    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")

    description = QSSensorEntityDescription(
        key="computed_value",
        translation_key="test",
        qs_is_none_unavailable=False,
        value_fn=lambda device, key: device.name.upper()
    )
    
    sensor = QSBaseSensor(mock_handler, mock_device, description)
    sensor.async_write_ha_state = MagicMock()
    
    test_time = datetime.now(pytz.UTC)
    sensor.async_update_callback(test_time)
    
    assert sensor._attr_native_value == "MOCK DEVICE"


@pytest.mark.asyncio
async def test_async_setup_entry(fake_hass, mock_config_entry):
    """Test sensor platform setup."""
    mock_device = create_mock_device("home")
    mock_device.data_handler = MagicMock()
    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device
    
    mock_add_entities = MagicMock()
    
    with patch('custom_components.quiet_solar.sensor.create_ha_sensor', return_value=[MagicMock()]):
        await async_setup_entry(fake_hass, mock_config_entry, mock_add_entities)
        
        mock_add_entities.assert_called_once()


@pytest.mark.asyncio
async def test_async_setup_entry_no_device(fake_hass, mock_config_entry):
    """Test sensor platform setup with no device."""
    mock_add_entities = MagicMock()
    
    await async_setup_entry(fake_hass, mock_config_entry, mock_add_entities)
    
    mock_add_entities.assert_not_called()


@pytest.mark.asyncio
async def test_async_unload_entry(fake_hass, mock_config_entry):
    """Test sensor platform unload."""
    mock_device = create_mock_device("test")
    mock_home = create_minimal_home_model()
    mock_device.home = mock_home
    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device
    
    result = await async_unload_entry(fake_hass, mock_config_entry)
    
    assert result is True
    mock_home.remove_device.assert_called_once_with(mock_device)


@pytest.mark.asyncio
async def test_async_unload_entry_handles_exception(fake_hass, mock_config_entry):
    """Test sensor platform unload handles exceptions gracefully."""
    mock_device = create_mock_device("test")
    mock_home = create_minimal_home_model()
    mock_home.remove_device = MagicMock(side_effect=Exception("Test error"))
    mock_device.home = mock_home
    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device
    
    result = await async_unload_entry(fake_hass, mock_config_entry)
    
    # Should still return True even with exception
    assert result is True


def test_qs_load_sensor_current_constraints_update():
    """Test QSLoadSensorCurrentConstraints update callback."""
    from custom_components.quiet_solar.home_model.load import AbstractLoad
    
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = MagicMock(spec=AbstractLoad)
    mock_device.device_id = "test_load"
    mock_device.device_type = "load"
    mock_device.name = "Test Load"
    mock_device.qs_enable_device = True
    mock_device.get_active_readable_name = MagicMock(return_value="Active Constraint")
    mock_device.get_active_constraints = MagicMock(return_value=[])
    mock_device._last_completed_constraint = None
    mock_device.update_to_be_saved_info = MagicMock(return_value={})
    
    mock_description = MagicMock()
    mock_description.key = "current_constraint"
    mock_description.name = None
    mock_description.translation_key = "test"
    
    sensor = QSLoadSensorCurrentConstraints(mock_handler, mock_device, mock_description)
    sensor.async_write_ha_state = MagicMock()
    
    test_time = datetime.now(pytz.UTC)
    sensor.async_update_callback(test_time)
    
    assert sensor._attr_native_value == "Active Constraint"
    sensor.async_write_ha_state.assert_called_once()
