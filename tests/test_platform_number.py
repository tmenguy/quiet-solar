"""Tests for number platform."""
from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN

from custom_components.quiet_solar.number import (
    create_ha_number,
    create_ha_number_for_QSBiStateDuration,
    QSBaseNumber,
    QSNumberEntityDescription,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.quiet_solar.const import DOMAIN
from tests.conftest import create_mock_device


def test_create_ha_number_for_bistate_duration():
    """Test creating number entities for bistate duration device."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    
    mock_device = MagicMock(spec=QSBiStateDuration)
    mock_device.data_handler = MagicMock()
    mock_device.device_id = "test_bistate"
    mock_device.device_type = "bistate"
    mock_device.name = "Test BiState"
    mock_device.qs_enable_device = True
    mock_device.default_on_duration = 2.0
    
    entities = create_ha_number_for_QSBiStateDuration(mock_device)
    
    # Should create 2 number entity for default_on_duration and override_duration
    assert len(entities) == 2
    for i in range(2):
        assert isinstance(entities[i], QSBaseNumber)
        assert entities[i].entity_description.key in ["default_on_duration", "override_duration"]

def test_create_ha_number_general():
    """Test general number creation function."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    
    # BiState device should get number entities - use full mock with all attributes
    mock_bistate = create_mock_device("bistate", name="Test BiState")
    mock_bistate.__class__ = QSBiStateDuration
    mock_bistate.data_handler = MagicMock()
    mock_bistate.default_on_duration = 1.5
    
    entities = create_ha_number(mock_bistate)
    assert len(entities) == 2
    
    # Non-BiState device should get no number entities
    mock_home = create_mock_device("home")
    entities = create_ha_number(mock_home)
    assert len(entities) == 0


def test_qs_base_number_init():
    """Test QSBaseNumber initialization."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    
    mock_handler = MagicMock()
    mock_device = MagicMock(spec=QSBiStateDuration)
    mock_device.device_id = "test_device"
    mock_device.device_type = "bistate"
    mock_device.name = "Test Device"
    mock_device.qs_enable_device = True
    
    mock_description = QSNumberEntityDescription(
        key="default_on_duration",
        translation_key="default_on_duration",
        native_max_value=24,
        native_min_value=0,
        native_step=0.5,
    )
    
    number = QSBaseNumber(mock_handler, mock_device, mock_description)
    
    assert number.device == mock_device
    assert number.entity_description == mock_description
    assert number.entity_description.native_max_value == 24
    assert number.entity_description.native_min_value == 0


@pytest.mark.asyncio
async def test_qs_base_number_set_value():
    """Test setting number value."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = MagicMock(spec=QSBiStateDuration)
    mock_device.device_id = "test_device"
    mock_device.device_type = "bistate"
    mock_device.name = "Test Device"
    mock_device.qs_enable_device = True
    mock_device.default_on_duration = 2.0
    
    mock_description = QSNumberEntityDescription(
        key="default_on_duration",
        translation_key="default_on_duration",
        native_max_value=24,
        native_min_value=0,
        native_step=0.5,
    )
    
    number = QSBaseNumber(mock_handler, mock_device, mock_description)
    number.async_write_ha_state = MagicMock()
    
    # Set new value
    await number.async_set_native_value(3.5)
    
    # Value should be updated
    assert number._attr_native_value == 3.5
    # Device attribute should be updated
    assert mock_device.default_on_duration == 3.5
    # State should be written
    number.async_write_ha_state.assert_called_once()


@pytest.mark.asyncio
async def test_qs_base_number_restore_state():
    """Test restoring number state."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    from homeassistant.core import State
    
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = MagicMock(spec=QSBiStateDuration)
    mock_device.device_id = "test_device"
    mock_device.device_type = "bistate"
    mock_device.name = "Test Device"
    mock_device.qs_enable_device = True
    mock_device.default_on_duration = 2.0
    
    mock_description = QSNumberEntityDescription(
        key="default_on_duration",
        translation_key="default_on_duration",
        native_max_value=24,
        native_min_value=0,
        native_step=0.5,
    )
    
    number = QSBaseNumber(mock_handler, mock_device, mock_description)
    number.async_write_ha_state = MagicMock()
    
    # Mock last state
    last_state = State("number.test", "5.5")
    
    with patch.object(number, 'async_get_last_state', return_value=last_state):
        await number.async_added_to_hass()
    
    # Should restore to last value
    assert number._attr_native_value == 5.5
    assert mock_device.default_on_duration == 5.5


@pytest.mark.asyncio
async def test_qs_base_number_restore_no_state():
    """Test number when no previous state exists."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = MagicMock(spec=QSBiStateDuration)
    mock_device.device_id = "test_device"
    mock_device.device_type = "bistate"
    mock_device.name = "Test Device"
    mock_device.qs_enable_device = True
    mock_device.default_on_duration = 3.0
    
    mock_description = QSNumberEntityDescription(
        key="default_on_duration",
        translation_key="default_on_duration",
        native_max_value=24,
        native_min_value=0,
        native_step=0.5,
    )
    
    number = QSBaseNumber(mock_handler, mock_device, mock_description)
    number.async_write_ha_state = MagicMock()
    
    # Mock no last state
    with patch.object(number, 'async_get_last_state', return_value=None):
        await number.async_added_to_hass()
    
    # Should use device's current value
    assert number._attr_native_value == 3.0


@pytest.mark.asyncio
async def test_qs_base_number_restore_invalid_state():
    """Test number when previous state is invalid."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    from homeassistant.core import State
    
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = MagicMock(spec=QSBiStateDuration)
    mock_device.device_id = "test_device"
    mock_device.device_type = "bistate"
    mock_device.name = "Test Device"
    mock_device.qs_enable_device = True
    mock_device.default_on_duration = 2.5
    
    mock_description = QSNumberEntityDescription(
        key="default_on_duration",
        translation_key="default_on_duration",
        native_max_value=24,
        native_min_value=0,
        native_step=0.5,
    )
    
    number = QSBaseNumber(mock_handler, mock_device, mock_description)
    number.async_write_ha_state = MagicMock()
    
    # Mock invalid last state
    last_state = State("number.test", STATE_UNAVAILABLE)
    
    with patch.object(number, 'async_get_last_state', return_value=last_state):
        await number.async_added_to_hass()
    
    # Should use device's current value when state is unavailable
    assert number._attr_native_value == 2.5


@pytest.mark.asyncio
async def test_qs_base_number_set_value_error():
    """Test setting number value when setattr fails."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    
    # Create a device where setting attribute will fail
    mock_device = MagicMock(spec=QSBiStateDuration)
    mock_device.device_id = "test_device"
    mock_device.device_type = "bistate"
    mock_device.name = "Test Device"
    mock_device.qs_enable_device = True
    # Make setting attribute raise exception
    type(mock_device).default_on_duration = property(lambda self: 1.0, lambda self, val: (_ for _ in ()).throw(Exception("Can't set")))
    
    mock_description = QSNumberEntityDescription(
        key="default_on_duration",
        translation_key="default_on_duration",
        native_max_value=24,
        native_min_value=0,
        native_step=0.5,
    )
    
    number = QSBaseNumber(mock_handler, mock_device, mock_description)
    number.async_write_ha_state = MagicMock()
    
    # Setting value should not raise exception (catches internally)
    await number.async_set_native_value(5.0)
    
    # Value should still be updated in entity
    assert number._attr_native_value == 5.0


@pytest.mark.asyncio
async def test_async_setup_entry():
    """Test number platform setup."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    
    fake_hass = MagicMock()
    mock_config_entry = MagicMock()
    mock_config_entry.entry_id = "test_entry"
    
    # Use create_mock_device for proper mocking
    mock_device = create_mock_device("bistate", name="Test BiState")
    mock_device.__class__ = QSBiStateDuration
    mock_device.data_handler = MagicMock()
    mock_device.default_on_duration = 2.0
    
    fake_hass.data = {DOMAIN: {mock_config_entry.entry_id: mock_device}}
    
    mock_add_entities = MagicMock()
    
    await async_setup_entry(fake_hass, mock_config_entry, mock_add_entities)
    
    # Should add entities
    mock_add_entities.assert_called_once()
    added_entities = mock_add_entities.call_args[0][0]
    assert len(added_entities) == 2
    assert isinstance(added_entities[0], QSBaseNumber)
    assert isinstance(added_entities[1], QSBaseNumber)


@pytest.mark.asyncio
async def test_async_setup_entry_no_device():
    """Test number platform setup with no device."""
    fake_hass = MagicMock()
    mock_config_entry = MagicMock()
    mock_config_entry.entry_id = "test_entry"
    
    fake_hass.data = {DOMAIN: {}}
    
    mock_add_entities = MagicMock()
    
    await async_setup_entry(fake_hass, mock_config_entry, mock_add_entities)
    
    # Should not add entities
    mock_add_entities.assert_not_called()


@pytest.mark.asyncio
async def test_async_setup_entry_device_without_numbers():
    """Test number platform setup for device without number entities."""
    fake_hass = MagicMock()
    mock_config_entry = MagicMock()
    mock_config_entry.entry_id = "test_entry"
    
    # Home device doesn't have number entities
    mock_home = create_mock_device("home")
    fake_hass.data = {DOMAIN: {mock_config_entry.entry_id: mock_home}}
    
    mock_add_entities = MagicMock()
    
    await async_setup_entry(fake_hass, mock_config_entry, mock_add_entities)
    
    # Should not add entities for home device
    mock_add_entities.assert_not_called()


@pytest.mark.asyncio
async def test_async_unload_entry():
    """Test number platform unload."""
    fake_hass = MagicMock()
    mock_config_entry = MagicMock()
    mock_config_entry.entry_id = "test_entry"
    
    mock_device = create_mock_device("bistate")
    mock_home = MagicMock()
    mock_device.home = mock_home
    
    fake_hass.data = {DOMAIN: {mock_config_entry.entry_id: mock_device}}
    
    result = await async_unload_entry(fake_hass, mock_config_entry)
    
    assert result is True
    mock_home.remove_device.assert_called_once_with(mock_device)


@pytest.mark.asyncio
async def test_async_unload_entry_no_device():
    """Test number platform unload with no device."""
    fake_hass = MagicMock()
    mock_config_entry = MagicMock()
    mock_config_entry.entry_id = "nonexistent"
    
    fake_hass.data = {DOMAIN: {}}
    
    result = await async_unload_entry(fake_hass, mock_config_entry)
    
    assert result is True


@pytest.mark.asyncio
async def test_async_unload_entry_handles_exception():
    """Test number platform unload handles exceptions gracefully."""
    fake_hass = MagicMock()
    mock_config_entry = MagicMock()
    mock_config_entry.entry_id = "test_entry"
    
    mock_device = create_mock_device("test")
    mock_home = MagicMock()
    mock_home.remove_device = MagicMock(side_effect=Exception("Test error"))
    mock_device.home = mock_home
    
    fake_hass.data = {DOMAIN: {mock_config_entry.entry_id: mock_device}}
    
    result = await async_unload_entry(fake_hass, mock_config_entry)
    
    # Should still return True even with exception
    assert result is True


def test_number_entity_description():
    """Test custom number entity description."""
    description = QSNumberEntityDescription(
        key="test_number",
        translation_key="test",
        native_max_value=100,
        native_min_value=0,
        native_step=1,
        qs_default_option="default",
    )
    
    assert description.key == "test_number"
    assert description.native_max_value == 100
    assert description.native_min_value == 0
    assert description.native_step == 1
    assert description.qs_default_option == "default"


def test_number_entity_min_max_step():
    """Test number entity respects min, max, and step values."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    
    mock_handler = MagicMock()
    mock_device = MagicMock(spec=QSBiStateDuration)
    mock_device.device_id = "test"
    mock_device.device_type = "bistate"
    mock_device.name = "Test"
    mock_device.qs_enable_device = True
    
    description = QSNumberEntityDescription(
        key="default_on_duration",
        translation_key="default_on_duration",
        native_max_value=24,
        native_min_value=0,
        native_step=0.5,
    )
    
    number = QSBaseNumber(mock_handler, mock_device, description)
    
    # Check min, max, step are set
    assert number.native_max_value == 24
    assert number.native_min_value == 0
    assert number.native_step == 0.5


@pytest.mark.asyncio
async def test_number_with_disabled_device():
    """Test number entity availability when device is disabled."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    
    mock_handler = MagicMock()
    mock_device = MagicMock(spec=QSBiStateDuration)
    mock_device.device_id = "test"
    mock_device.device_type = "bistate"
    mock_device.name = "Test"
    mock_device.qs_enable_device = False  # Disabled
    
    description = QSNumberEntityDescription(
        key="default_on_duration",
        translation_key="default_on_duration",
        native_max_value=24,
        native_min_value=0,
        native_step=0.5,
    )
    
    number = QSBaseNumber(mock_handler, mock_device, description)
    
    # Number should be unavailable when device is disabled
    assert number._attr_available is False


@pytest.mark.asyncio
async def test_number_availability_updates():
    """Test that number availability updates when device state changes."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    
    mock_handler = MagicMock()
    mock_device = MagicMock(spec=QSBiStateDuration)
    mock_device.device_id = "test"
    mock_device.device_type = "bistate"
    mock_device.name = "Test"
    mock_device.qs_enable_device = True
    mock_device.default_on_duration = 2.0
    
    description = QSNumberEntityDescription(
        key="default_on_duration",
        translation_key="default_on_duration",
        native_max_value=24,
        native_min_value=0,
        native_step=0.5,
    )
    
    number = QSBaseNumber(mock_handler, mock_device, description)
    number.async_write_ha_state = MagicMock()
    
    # Initially available
    assert number._attr_available is True
    
    # Disable device
    mock_device.qs_enable_device = False
    number._set_availabiltiy()
    
    # Should now be unavailable
    assert number._attr_available is False

