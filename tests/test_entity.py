"""Tests for entity creation and base classes."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest
import pytz

from homeassistant.const import CONF_NAME, Platform

from custom_components.quiet_solar.const import DOMAIN, DEVICE_TYPE
from custom_components.quiet_solar.entity import (
    create_device_from_type,
    QSBaseEntity,
    QSDeviceEntity,
    LOAD_TYPE__DICT,
    LOAD_NAMES,
)
from custom_components.quiet_solar.ha_model.home import QSHome
from custom_components.quiet_solar.ha_model.car import QSCar
from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric
from custom_components.quiet_solar.ha_model.battery import QSBattery
from tests.test_helpers import create_mock_device


def test_create_device_from_type_home(fake_hass, mock_home_config_entry):
    """Test creating home device from config."""
    mock_home_config_entry.data[DEVICE_TYPE] = QSHome.conf_type_name
    mock_home_config_entry.data[CONF_NAME] = "Test Home"
    
    # Create device - should succeed with FakeHass since we have all needed infrastructure
    device = create_device_from_type(
        hass=fake_hass,
        home=None,
        type=QSHome.conf_type_name,
        config_entry=mock_home_config_entry
    )
    
    # Verify device was created successfully
    assert device is not None
    assert isinstance(device, QSHome)
    assert device.name == "Test Home"


def test_create_device_from_type_unknown_type(fake_hass, mock_config_entry):
    """Test creating device with unknown type returns None."""
    device = create_device_from_type(
        hass=fake_hass,
        home=None,
        type="unknown_type",
        config_entry=mock_config_entry
    )
    
    assert device is None


def test_create_device_from_type_no_config_entry(fake_hass):
    """Test creating device without config entry uses empty data."""
    device = create_device_from_type(
        hass=fake_hass,
        home=None,
        type="unknown",
        config_entry=None
    )
    
    assert device is None


def test_load_type_dict_contains_all_types():
    """Test LOAD_TYPE__DICT contains all expected device types."""
    assert QSHome.conf_type_name in LOAD_TYPE__DICT
    assert QSCar.conf_type_name in LOAD_TYPE__DICT
    assert QSChargerGeneric.conf_type_name in LOAD_TYPE__DICT
    assert QSBattery.conf_type_name in LOAD_TYPE__DICT


def test_load_names_maps_types_correctly():
    """Test LOAD_NAMES provides readable names."""
    assert LOAD_NAMES[QSHome.conf_type_name] == "home"
    assert LOAD_NAMES[QSCar.conf_type_name] == "car"
    assert "charger" in LOAD_NAMES.values()


def test_qs_base_entity_init():
    """Test QSBaseEntity initialization."""
    mock_handler = MagicMock()
    mock_description = MagicMock()
    mock_description.name = "Test Entity"
    mock_description.translation_key = None
    
    entity = QSBaseEntity(mock_handler, mock_description)
    
    assert entity.data_handler == mock_handler
    assert entity.entity_description == mock_description
    assert entity._attr_extra_state_attributes == {}


def test_qs_base_entity_availability():
    """Test QSBaseEntity availability setting."""
    mock_handler = MagicMock()
    mock_description = MagicMock()
    mock_description.name = "Test"
    mock_description.translation_key = None
    
    entity = QSBaseEntity(mock_handler, mock_description)
    entity._set_availabiltiy()
    
    assert entity._attr_available is True


def test_qs_base_entity_update_callback():
    """Test QSBaseEntity async_update_callback."""
    mock_handler = MagicMock()
    mock_description = MagicMock()
    mock_description.name = "Test"
    mock_description.translation_key = None
    
    entity = QSBaseEntity(mock_handler, mock_description)
    test_time = datetime.now(pytz.UTC)
    
    entity.async_update_callback(test_time)
    
    assert entity._attr_available is True


def test_qs_device_entity_init():
    """Test QSDeviceEntity initialization."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test", name="Test Device")
    mock_description = MagicMock()
    mock_description.key = "test_key"
    mock_description.name = None
    mock_description.translation_key = "test_translation"
    
    entity = QSDeviceEntity(mock_handler, mock_device, mock_description)
    
    assert entity.device == mock_device
    assert entity._attr_device_info is not None
    assert "Test Device" in entity._attr_device_info["name"]
    assert (DOMAIN, mock_device.device_id) in entity._attr_device_info["identifiers"]


def test_qs_device_entity_unique_id():
    """Test QSDeviceEntity generates unique ID."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test", name="Test Device")
    mock_device.device_id = "test_device_123"
    mock_description = MagicMock()
    mock_description.key = "test_sensor"
    mock_description.name = None
    mock_description.translation_key = "test"
    
    entity = QSDeviceEntity(mock_handler, mock_device, mock_description)
    
    assert entity._attr_unique_id == "test_device_123-test_sensor"


def test_qs_device_entity_device_type_property():
    """Test QSDeviceEntity device_type property."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test_type", name="Test")
    mock_description = MagicMock()
    mock_description.key = "key"
    mock_description.name = None
    mock_description.translation_key = "test"
    
    entity = QSDeviceEntity(mock_handler, mock_device, mock_description)
    
    assert entity.device_type == "test_type"


@pytest.mark.asyncio
async def test_qs_device_entity_async_added_to_hass():
    """Test QSDeviceEntity async_added_to_hass."""
    from custom_components.quiet_solar.ha_model.device import HADeviceMixin
    
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    
    # Create a mock device that is HADeviceMixin
    mock_device = MagicMock(spec=HADeviceMixin)
    mock_device.device_id = "test_123"
    mock_device.device_type = "test"
    mock_device.name = "Test"
    mock_device.qs_enable_device = True
    mock_device.attach_exposed_has_entity = MagicMock()
    
    mock_description = MagicMock()
    mock_description.key = "key"
    mock_description.name = None
    mock_description.translation_key = "test"
    
    entity = QSDeviceEntity(mock_handler, mock_device, mock_description)
    
    await entity.async_added_to_hass()
    
    mock_device.attach_exposed_has_entity.assert_called_once_with(entity)
    assert entity._attr_available is True


def test_qs_device_entity_availability_disabled_device():
    """Test QSDeviceEntity becomes unavailable when device disabled."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test", qs_enable_device=False)
    mock_description = MagicMock()
    mock_description.key = "key"
    mock_description.name = None
    mock_description.translation_key = "test"
    
    entity = QSDeviceEntity(mock_handler, mock_device, mock_description)
    entity._set_availabiltiy()
    
    assert entity._attr_available is False


def test_qs_device_entity_availability_enabled_device():
    """Test QSDeviceEntity available when device enabled."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test", qs_enable_device=True)
    mock_description = MagicMock()
    mock_description.key = "key"
    mock_description.name = None
    mock_description.translation_key = "test"
    
    entity = QSDeviceEntity(mock_handler, mock_device, mock_description)
    entity._set_availabiltiy()
    
    assert entity._attr_available is True
