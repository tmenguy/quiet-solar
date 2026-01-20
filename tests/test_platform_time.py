"""Tests for time platform."""
from __future__ import annotations

from datetime import datetime, time as dt_time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import pytz

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN

from custom_components.quiet_solar.time import (
    create_ha_time,
    create_ha_time_for_QSCharger,
    create_ha_time_for_QSCar,
    create_ha_time_for_QSBiStateDuration,
    QSBaseTime,
    QSTimeEntityDescription,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.quiet_solar.const import DOMAIN
from tests.test_helpers import create_mock_device


def test_create_ha_time_for_charger():
    """Test creating time entities for charger device."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric

    mock_device = MagicMock(spec=QSChargerGeneric)
    mock_device.name = "Test Charger"
    mock_device.device_id = "charger_test"
    mock_device.device_type = "charger"
    mock_device.qs_enable_device = True
    mock_device.data_handler = MagicMock()
    mock_device.data_handler.hass = MagicMock()
    mock_device.default_charge_time = dt_time(hour=7, minute=0)

    entities = create_ha_time_for_QSCharger(mock_device)

    assert len(entities) == 1
    assert isinstance(entities[0], QSBaseTime)
    assert entities[0].entity_description.key == "default_charge_time"
    assert entities[0].entity_description.translation_key == "default_charge_time"


def test_create_ha_time_for_car():
    """Test creating time entities for car device."""
    from custom_components.quiet_solar.ha_model.car import QSCar

    mock_device = MagicMock(spec=QSCar)
    mock_device.name = "Test Car"
    mock_device.device_id = "car_test"
    mock_device.device_type = "car"
    mock_device.qs_enable_device = True
    mock_device.data_handler = MagicMock()
    mock_device.data_handler.hass = MagicMock()
    mock_device.default_charge_time = dt_time(hour=8, minute=30)

    entities = create_ha_time_for_QSCar(mock_device)

    assert len(entities) == 1
    assert isinstance(entities[0], QSBaseTime)
    assert entities[0].entity_description.key == "default_charge_time"


def test_create_ha_time_for_bistate_duration():
    """Test creating time entities for BiStateDuration device."""
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration

    mock_device = MagicMock(spec=QSBiStateDuration)
    mock_device.name = "Test BiState"
    mock_device.device_id = "bistate_test"
    mock_device.device_type = "bistate_duration"
    mock_device.qs_enable_device = True
    mock_device.data_handler = MagicMock()
    mock_device.data_handler.hass = MagicMock()
    mock_device.default_on_finish_time = dt_time(hour=6, minute=0)

    entities = create_ha_time_for_QSBiStateDuration(mock_device)

    assert len(entities) == 1
    assert isinstance(entities[0], QSBaseTime)
    assert entities[0].entity_description.key == "default_on_finish_time"
    assert entities[0].entity_description.translation_key == "default_on_finish_time"


def test_create_ha_time_for_charger_device():
    """Test create_ha_time for charger type device."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric

    mock_device = MagicMock(spec=QSChargerGeneric)
    mock_device.name = "Test Charger"
    mock_device.device_id = "charger_test"
    mock_device.device_type = "charger"
    mock_device.qs_enable_device = True
    mock_device.data_handler = MagicMock()
    mock_device.data_handler.hass = MagicMock()
    mock_device.default_charge_time = dt_time(hour=7, minute=0)

    entities = create_ha_time(mock_device)

    assert len(entities) == 1


def test_create_ha_time_for_car_device():
    """Test create_ha_time for car type device."""
    from custom_components.quiet_solar.ha_model.car import QSCar

    mock_device = MagicMock(spec=QSCar)
    mock_device.name = "Test Car"
    mock_device.device_id = "car_test"
    mock_device.device_type = "car"
    mock_device.qs_enable_device = True
    mock_device.data_handler = MagicMock()
    mock_device.data_handler.hass = MagicMock()
    mock_device.default_charge_time = dt_time(hour=8, minute=30)

    entities = create_ha_time(mock_device)

    assert len(entities) == 1


def test_create_ha_time_for_non_time_device():
    """Test create_ha_time returns empty list for non-time devices."""
    from custom_components.quiet_solar.ha_model.home import QSHome

    mock_device = MagicMock(spec=QSHome)
    mock_device.name = "Test Home"
    mock_device.device_id = "home_test"

    entities = create_ha_time(mock_device)

    assert len(entities) == 0


def test_qs_time_entity_description():
    """Test QSTimeEntityDescription dataclass."""
    desc = QSTimeEntityDescription(
        key="test_time",
        translation_key="test",
        qs_default_option="default_value"
    )

    assert desc.key == "test_time"
    assert desc.translation_key == "test"
    assert desc.qs_default_option == "default_value"


def test_qs_time_entity_description_defaults():
    """Test QSTimeEntityDescription default values."""
    desc = QSTimeEntityDescription(
        key="test",
        translation_key="test_key",
    )

    assert desc.qs_default_option is None


def test_qs_base_time_init():
    """Test QSBaseTime initialization."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.qs_enable_device = True
    mock_device.test_time = dt_time(hour=10, minute=30)

    mock_description = QSTimeEntityDescription(
        key="test_time",
        translation_key="test",
    )

    time_entity = QSBaseTime(mock_handler, mock_device, mock_description)

    assert time_entity.device == mock_device
    assert time_entity.entity_description == mock_description


@pytest.mark.asyncio
async def test_qs_base_time_set_value():
    """Test setting time value on QSBaseTime."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.qs_enable_device = True
    mock_device.test_time = dt_time(hour=10, minute=30)

    mock_description = QSTimeEntityDescription(
        key="test_time",
        translation_key="test",
    )

    time_entity = QSBaseTime(mock_handler, mock_device, mock_description)
    time_entity.async_write_ha_state = MagicMock()

    new_time = dt_time(hour=14, minute=45)
    await time_entity.async_set_value(new_time)

    assert time_entity._attr_native_value == new_time
    assert mock_device.test_time == new_time
    time_entity.async_write_ha_state.assert_called_once()


@pytest.mark.asyncio
async def test_qs_base_time_set_value_exception():
    """Test setting time value handles exception gracefully."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.qs_enable_device = True

    # Make setting attribute raise an exception
    def raise_error(value):
        raise ValueError("Test error")

    type(mock_device).test_time = property(lambda s: dt_time(hour=10, minute=0), raise_error)

    mock_description = QSTimeEntityDescription(
        key="test_time",
        translation_key="test",
    )

    time_entity = QSBaseTime(mock_handler, mock_device, mock_description)
    time_entity.async_write_ha_state = MagicMock()

    new_time = dt_time(hour=14, minute=45)
    # Should not raise exception
    await time_entity.async_set_value(new_time)

    assert time_entity._attr_native_value == new_time
    time_entity.async_write_ha_state.assert_called_once()


def test_qs_base_time_availability_enabled():
    """Test time entity available when device is enabled."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.qs_enable_device = True
    mock_device.test_time = dt_time(hour=10, minute=0)

    mock_description = QSTimeEntityDescription(
        key="test_time",
        translation_key="test",
    )

    time_entity = QSBaseTime(mock_handler, mock_device, mock_description)
    time_entity._set_availabiltiy()

    assert time_entity._attr_available is True


def test_qs_base_time_availability_disabled():
    """Test time entity unavailable when device is disabled."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.qs_enable_device = False
    mock_device.test_time = dt_time(hour=10, minute=0)

    mock_description = QSTimeEntityDescription(
        key="test_time",
        translation_key="test",
    )

    time_entity = QSBaseTime(mock_handler, mock_device, mock_description)
    time_entity._set_availabiltiy()

    assert time_entity._attr_available is False


@pytest.mark.asyncio
async def test_qs_base_time_restore_state():
    """Test QSBaseTime restores last state on add to hass."""
    from homeassistant.core import State

    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.qs_enable_device = True
    mock_device.test_time = dt_time(hour=7, minute=0)

    mock_description = QSTimeEntityDescription(
        key="test_time",
        translation_key="test",
    )

    time_entity = QSBaseTime(mock_handler, mock_device, mock_description)
    time_entity.async_write_ha_state = MagicMock()

    # Mock the restore state mechanism
    mock_last_state = State("time.test", "14:30:00")

    with patch.object(time_entity, 'async_get_last_state', return_value=mock_last_state):
        await time_entity.async_added_to_hass()

    assert time_entity._attr_native_value == dt_time(hour=14, minute=30, second=0)
    assert mock_device.test_time == dt_time(hour=14, minute=30, second=0)


@pytest.mark.asyncio
async def test_qs_base_time_restore_state_unknown():
    """Test QSBaseTime uses default when last state is unknown."""
    from homeassistant.core import State

    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.qs_enable_device = True
    mock_device.test_time = dt_time(hour=9, minute=30)

    mock_description = QSTimeEntityDescription(
        key="test_time",
        translation_key="test",
    )

    time_entity = QSBaseTime(mock_handler, mock_device, mock_description)
    time_entity.async_write_ha_state = MagicMock()

    # Mock unknown state
    mock_last_state = State("time.test", STATE_UNKNOWN)

    with patch.object(time_entity, 'async_get_last_state', return_value=mock_last_state):
        await time_entity.async_added_to_hass()

    # Should use device default
    assert time_entity._attr_native_value == dt_time(hour=9, minute=30)


@pytest.mark.asyncio
async def test_qs_base_time_restore_state_unavailable():
    """Test QSBaseTime uses default when last state is unavailable."""
    from homeassistant.core import State

    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.qs_enable_device = True
    mock_device.test_time = dt_time(hour=8, minute=0)

    mock_description = QSTimeEntityDescription(
        key="test_time",
        translation_key="test",
    )

    time_entity = QSBaseTime(mock_handler, mock_device, mock_description)
    time_entity.async_write_ha_state = MagicMock()

    # Mock unavailable state
    mock_last_state = State("time.test", STATE_UNAVAILABLE)

    with patch.object(time_entity, 'async_get_last_state', return_value=mock_last_state):
        await time_entity.async_added_to_hass()

    # Should use device default
    assert time_entity._attr_native_value == dt_time(hour=8, minute=0)


@pytest.mark.asyncio
async def test_qs_base_time_restore_state_none():
    """Test QSBaseTime uses fallback when no last state."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.qs_enable_device = True
    mock_device.test_time = None  # No default on device

    mock_description = QSTimeEntityDescription(
        key="test_time",
        translation_key="test",
    )

    time_entity = QSBaseTime(mock_handler, mock_device, mock_description)
    time_entity.async_write_ha_state = MagicMock()

    with patch.object(time_entity, 'async_get_last_state', return_value=None):
        await time_entity.async_added_to_hass()

    # Should use fallback 7:00:00
    assert time_entity._attr_native_value == dt_time(hour=7, minute=0, second=0)


@pytest.mark.asyncio
async def test_qs_base_time_update_callback():
    """Test QSBaseTime update callback."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.qs_enable_device = True
    mock_device.test_time = dt_time(hour=15, minute=45)

    mock_description = QSTimeEntityDescription(
        key="test_time",
        translation_key="test",
    )

    time_entity = QSBaseTime(mock_handler, mock_device, mock_description)
    time_entity.async_write_ha_state = MagicMock()

    # Trigger update callback
    time_entity.async_update_callback(datetime.now(pytz.UTC))

    assert time_entity._attr_native_value == dt_time(hour=15, minute=45)
    time_entity.async_write_ha_state.assert_called_once()


@pytest.mark.asyncio
async def test_qs_base_time_update_callback_none_value():
    """Test QSBaseTime update callback with None value."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.qs_enable_device = True
    mock_device.test_time = None  # Device value is None

    mock_description = QSTimeEntityDescription(
        key="test_time",
        translation_key="test",
    )

    time_entity = QSBaseTime(mock_handler, mock_device, mock_description)
    time_entity.async_write_ha_state = MagicMock()

    # Trigger update callback
    time_entity.async_update_callback(datetime.now(pytz.UTC))

    # Should use fallback 7:00:00
    assert time_entity._attr_native_value == dt_time(hour=7, minute=0, second=0)
    time_entity.async_write_ha_state.assert_called_once()


@pytest.mark.asyncio
async def test_async_setup_entry_charger(fake_hass, mock_config_entry):
    """Test time platform setup for charger."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric

    mock_device = MagicMock(spec=QSChargerGeneric)
    mock_device.name = "Test Charger"
    mock_device.device_id = "charger_test"
    mock_device.device_type = "charger"
    mock_device.qs_enable_device = True
    mock_device.data_handler = MagicMock()
    mock_device.data_handler.hass = fake_hass
    mock_device.default_charge_time = dt_time(hour=7, minute=0)
    mock_device.get_attached_virtual_devices = MagicMock(return_value=[])

    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device

    mock_add_entities = MagicMock()

    await async_setup_entry(fake_hass, mock_config_entry, mock_add_entities)

    mock_add_entities.assert_called_once()
    entities = mock_add_entities.call_args[0][0]
    assert len(entities) == 1


@pytest.mark.asyncio
async def test_async_setup_entry_car(fake_hass, mock_config_entry):
    """Test time platform setup for car."""
    from custom_components.quiet_solar.ha_model.car import QSCar

    mock_device = MagicMock(spec=QSCar)
    mock_device.name = "Test Car"
    mock_device.device_id = "car_test"
    mock_device.device_type = "car"
    mock_device.qs_enable_device = True
    mock_device.data_handler = MagicMock()
    mock_device.data_handler.hass = fake_hass
    mock_device.default_charge_time = dt_time(hour=8, minute=0)
    mock_device.get_attached_virtual_devices = MagicMock(return_value=[])

    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device

    mock_add_entities = MagicMock()

    await async_setup_entry(fake_hass, mock_config_entry, mock_add_entities)

    mock_add_entities.assert_called_once()
    entities = mock_add_entities.call_args[0][0]
    assert len(entities) == 1


@pytest.mark.asyncio
async def test_async_setup_entry_no_device(fake_hass, mock_config_entry):
    """Test time platform setup with no device."""
    mock_add_entities = MagicMock()

    await async_setup_entry(fake_hass, mock_config_entry, mock_add_entities)

    mock_add_entities.assert_not_called()


@pytest.mark.asyncio
async def test_async_setup_entry_with_attached_devices(fake_hass, mock_config_entry):
    """Test time platform setup with attached virtual devices."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric
    from custom_components.quiet_solar.ha_model.car import QSCar

    # Main device (charger)
    mock_device = MagicMock(spec=QSChargerGeneric)
    mock_device.name = "Main Charger"
    mock_device.device_id = "charger_main"
    mock_device.device_type = "charger"
    mock_device.qs_enable_device = True
    mock_device.data_handler = MagicMock()
    mock_device.data_handler.hass = fake_hass
    mock_device.default_charge_time = dt_time(hour=7, minute=0)

    # Attached device (car)
    attached_device = MagicMock(spec=QSCar)
    attached_device.name = "Attached Car"
    attached_device.device_id = "car_attached"
    attached_device.device_type = "car"
    attached_device.qs_enable_device = True
    attached_device.data_handler = MagicMock()
    attached_device.data_handler.hass = fake_hass
    attached_device.default_charge_time = dt_time(hour=8, minute=0)

    mock_device.get_attached_virtual_devices = MagicMock(return_value=[attached_device])

    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device

    mock_add_entities = MagicMock()

    await async_setup_entry(fake_hass, mock_config_entry, mock_add_entities)

    mock_add_entities.assert_called_once()
    entities = mock_add_entities.call_args[0][0]
    # 1 from charger + 1 from car
    assert len(entities) == 2


@pytest.mark.asyncio
async def test_async_unload_entry(fake_hass, mock_config_entry):
    """Test time platform unload."""
    mock_device = create_mock_device("test")
    mock_device.home = MagicMock()
    mock_device.home.remove_device = MagicMock()

    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device

    result = await async_unload_entry(fake_hass, mock_config_entry)

    assert result is True
    mock_device.home.remove_device.assert_called_once_with(mock_device)


@pytest.mark.asyncio
async def test_async_unload_entry_no_device(fake_hass, mock_config_entry):
    """Test time platform unload with no device."""
    result = await async_unload_entry(fake_hass, mock_config_entry)

    assert result is True


@pytest.mark.asyncio
async def test_async_unload_entry_no_home(fake_hass, mock_config_entry):
    """Test time platform unload with device but no home."""
    mock_device = create_mock_device("test")
    mock_device.home = None

    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device

    result = await async_unload_entry(fake_hass, mock_config_entry)

    assert result is True


@pytest.mark.asyncio
async def test_async_unload_entry_exception(fake_hass, mock_config_entry):
    """Test time platform unload handles exceptions."""
    mock_device = create_mock_device("test")
    mock_device.home = MagicMock()
    mock_device.home.remove_device = MagicMock(side_effect=Exception("Test error"))

    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device

    # Should not raise, should return True
    result = await async_unload_entry(fake_hass, mock_config_entry)

    assert result is True


def test_create_ha_time_multiple_types():
    """Test create_ha_time for device that is both Charger and BiStateDuration."""
    # In practice this wouldn't happen, but test the logic
    from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration

    # Create a mock that is only QSChargerGeneric
    mock_device = MagicMock(spec=QSChargerGeneric)
    mock_device.name = "Test Charger"
    mock_device.device_id = "charger_test"
    mock_device.device_type = "charger"
    mock_device.qs_enable_device = True
    mock_device.data_handler = MagicMock()
    mock_device.data_handler.hass = MagicMock()
    mock_device.default_charge_time = dt_time(hour=7, minute=0)

    entities = create_ha_time(mock_device)

    # Should get 1 entity from charger
    assert len(entities) == 1


def test_qs_base_time_availability_updates():
    """Test that time availability updates when device state changes."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.qs_enable_device = True
    mock_device.test_time = dt_time(hour=10, minute=0)

    mock_description = QSTimeEntityDescription(
        key="test_time",
        translation_key="test",
    )

    time_entity = QSBaseTime(mock_handler, mock_device, mock_description)

    # Initially available
    assert time_entity._attr_available is True

    # Disable device
    mock_device.qs_enable_device = False
    time_entity._set_availabiltiy()

    # Should now be unavailable
    assert time_entity._attr_available is False

    # Re-enable device
    mock_device.qs_enable_device = True
    time_entity._set_availabiltiy()

    # Should be available again
    assert time_entity._attr_available is True
