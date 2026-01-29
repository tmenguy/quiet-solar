"""Tests for switch platform."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import pytz

from custom_components.quiet_solar.switch import (
    create_ha_switch,
    create_ha_switch_for_QSCharger,
    create_ha_switch_for_QSCar,
    create_ha_switch_for_QSPool,
    create_ha_switch_for_AbstractLoad,
    QSSwitchEntity,
    QSSwitchEntityChargerOrCar,
    QSSwitchEntityWithRestore,
    QSExtraStoredData,
    async_setup_entry,
    async_unload_entry,
    QSSwitchEntityDescription,
)
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    SWITCH_BEST_EFFORT_GREEN_ONLY,
    SWITCH_ENABLE_DEVICE,
)
from custom_components.quiet_solar.ha_model.device import HADeviceMixin
from custom_components.quiet_solar.home_model.load import TestLoad
from tests.test_helpers import create_mock_device
from tests.test_helpers import FakeConfigEntry, FakeHass


def test_create_ha_switch_for_charger():
    """Test creating switches for charger."""
    mock_charger = create_mock_device("charger")
    mock_charger.data_handler = MagicMock()
    
    entities = create_ha_switch_for_QSCharger(mock_charger)
    
    assert len(entities) == 1  # Solar priority switch
    assert all(isinstance(e, QSSwitchEntityChargerOrCar) for e in entities)


def test_create_ha_switch_for_car():
    """Test creating switches for car."""
    mock_car = create_mock_device("car")
    mock_car.data_handler = MagicMock()
    
    entities = create_ha_switch_for_QSCar(mock_car)
    
    assert len(entities) == 1  # Solar priority switch


class _TestLoadHA(HADeviceMixin, TestLoad):
    """Test load supporting HADeviceMixin for switch creation."""

    def __init__(self, *args, support_green_only_switch: bool = False, **kwargs):
        self._support_green_only_switch = support_green_only_switch
        super().__init__(*args, **kwargs)

    def support_green_only_switch(self) -> bool:
        return self._support_green_only_switch


def _create_load_with_handler(support_green_only_switch: bool) -> _TestLoadHA:
    fake_hass = FakeHass()
    fake_hass.states.async_available = MagicMock(return_value=True)
    data_handler = MagicMock()
    data_handler.hass = fake_hass
    fake_hass.data[DOMAIN][DATA_HANDLER] = data_handler
    config_entry = FakeConfigEntry(entry_id="load_entry", data={})
    return _TestLoadHA(
        hass=fake_hass,
        config_entry=config_entry,
        home=MagicMock(),
        name="Test Load",
        device_type="load",
        support_green_only_switch=support_green_only_switch,
    )


def test_create_ha_switch_for_load():
    """Test creating switches for load with green-only support."""
    load = _create_load_with_handler(support_green_only_switch=True)
    entities = create_ha_switch_for_AbstractLoad(load)

    keys = {entity.entity_description.key for entity in entities}
    assert keys == {SWITCH_BEST_EFFORT_GREEN_ONLY, SWITCH_ENABLE_DEVICE}


def test_create_ha_switch_for_load_no_green_only():
    """Test creating switches for load without green-only support."""
    load = _create_load_with_handler(support_green_only_switch=False)
    entities = create_ha_switch_for_AbstractLoad(load)

    keys = {entity.entity_description.key for entity in entities}
    assert keys == {SWITCH_ENABLE_DEVICE}


@pytest.mark.asyncio
async def test_qs_switch_entity_turn_on():
    """Test turning switch on."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.test_switch = False


    mock_description = QSSwitchEntityDescription(
        key="test_switch",
        translation_key="test",
    )
    
    switch = QSSwitchEntity(mock_handler, mock_device, mock_description)
    switch.async_write_ha_state = MagicMock()
    
    await switch.async_turn_on()
    
    assert mock_device.test_switch is True
    assert switch._attr_is_on is True
    switch.async_write_ha_state.assert_called_once()


@pytest.mark.asyncio
async def test_qs_switch_entity_turn_off():
    """Test turning switch off."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.test_switch = True

    mock_description = QSSwitchEntityDescription(
        key="test_switch",
        translation_key="test",
    )


    
    switch = QSSwitchEntity(mock_handler, mock_device, mock_description)
    switch.async_write_ha_state = MagicMock()
    
    await switch.async_turn_off()
    
    assert mock_device.test_switch is False
    assert switch._attr_is_on is False


@pytest.mark.asyncio
async def test_qs_switch_entity_async_switch_callback():
    """Test async_switch callback is used."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.home = MagicMock()
    mock_device.home.force_update_all = AsyncMock()

    async_switch = AsyncMock()
    mock_description = QSSwitchEntityDescription(
        key="test_switch",
        translation_key="test",
        async_switch=async_switch,
    )

    switch = QSSwitchEntity(mock_handler, mock_device, mock_description)
    switch.async_write_ha_state = MagicMock()

    await switch.async_turn_on(for_init=True)
    async_switch.assert_called_with(mock_device, True, True)

    await switch.async_turn_off(for_init=True)
    async_switch.assert_called_with(mock_device, False, True)


@pytest.mark.asyncio
async def test_qs_switch_restore_uses_last_state():
    """Test switch restore uses last stored value."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.test_switch = False

    mock_description = QSSwitchEntityDescription(
        key="test_switch",
        translation_key="test",
    )

    switch = QSSwitchEntityWithRestore(mock_handler, mock_device, mock_description)
    switch.async_write_ha_state = MagicMock()
    switch.async_get_last_extra_data = AsyncMock(return_value=QSExtraStoredData(True))

    await switch.async_added_to_hass()

    assert switch._attr_is_on is True


@pytest.mark.asyncio
async def test_qs_switch_restore_defaults_when_no_attr():
    """Test switch restore falls back to False when attribute missing."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = SimpleNamespace(
        device_id="test_device",
        device_type="test",
        name="Test Device",
        qs_enable_device=True,
        home=None,
    )

    mock_description = QSSwitchEntityDescription(
        key="missing_attr",
        translation_key="test",
    )

    switch = QSSwitchEntityWithRestore(mock_handler, mock_device, mock_description)
    switch.async_write_ha_state = MagicMock()
    switch.async_get_last_extra_data = AsyncMock(return_value=None)

    await switch.async_added_to_hass()

    assert switch._attr_is_on is False


def test_switch_extra_restore_data_from_dict_error():
    """Test QSExtraStoredData.from_dict handles errors."""
    assert QSExtraStoredData.from_dict({}) is None


def test_qs_switch_entity_availability_disabled_device():
    """Test switch unavailable when device disabled (except enable switch itself)."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test", qs_enable_device=False)
    mock_description = MagicMock()
    mock_description.key = "other_switch"
    mock_description.name = None
    mock_description.translation_key = "test"
    mock_description.set_val = None
    
    switch = QSSwitchEntity(mock_handler, mock_device, mock_description)
    switch._set_availabiltiy()
    
    assert switch._attr_available is False


def test_qs_switch_entity_enable_switch_always_available():
    """Test enable switch is available even when device disabled."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test", qs_enable_device=False)
    mock_description = MagicMock()
    mock_description.key = SWITCH_ENABLE_DEVICE
    mock_description.name = None
    mock_description.translation_key = "test"
    mock_description.set_val = None
    
    switch = QSSwitchEntity(mock_handler, mock_device, mock_description)
    switch._set_availabiltiy()
    
    assert switch._attr_available is True


def test_qs_switch_entity_charger_or_car_availability_no_connection():
    """Test charger/car switch unavailable when not connected."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric
    
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    
    # Create a mock that looks like QSChargerGeneric
    mock_charger = MagicMock(spec=QSChargerGeneric)
    mock_charger.car = None
    mock_charger.qs_enable_device = True
    mock_charger.device_id = "charger_123"
    mock_charger.device_type = "charger"
    mock_charger.name = "Charger"
    
    mock_description = MagicMock()
    mock_description.key = "test_switch"
    mock_description.name = None
    mock_description.translation_key = "test"
    mock_description.set_val = None
    
    switch = QSSwitchEntityChargerOrCar(mock_handler, mock_charger, mock_description)
    switch._set_availabiltiy()
    
    assert switch._attr_available is False


def test_qs_switch_entity_charger_or_car_availability_with_car():
    """Test charger switch available when car connected."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric
    
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_charger = MagicMock(spec=QSChargerGeneric)
    mock_charger.car = MagicMock()
    mock_charger.qs_enable_device = True
    mock_charger.device_id = "charger_123"
    mock_charger.device_type = "charger"
    mock_charger.name = "Charger"
    mock_description = MagicMock()
    mock_description.key = "test_switch"
    mock_description.name = None
    mock_description.translation_key = "test"
    mock_description.set_val = None
    
    switch = QSSwitchEntityChargerOrCar(mock_handler, mock_charger, mock_description)
    switch._set_availabiltiy()
    
    assert switch._attr_available is True


@pytest.mark.asyncio
async def test_async_setup_entry(fake_hass, mock_config_entry):
    """Test switch platform setup."""
    mock_device = create_mock_device("test")
    mock_device.data_handler = MagicMock()
    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device
    
    mock_add_entities = MagicMock()
    
    with patch('custom_components.quiet_solar.switch.create_ha_switch', return_value=[MagicMock()]):
        await async_setup_entry(fake_hass, mock_config_entry, mock_add_entities)
        
        mock_add_entities.assert_called_once()


@pytest.mark.asyncio
async def test_async_unload_entry_handles_exception():
    """Test switch platform unload handles errors."""
    fake_hass = FakeHass()
    mock_config_entry = FakeConfigEntry(entry_id="test_entry", data={})

    mock_device = create_mock_device("test")
    mock_home = MagicMock()
    mock_home.remove_device = MagicMock(side_effect=RuntimeError("boom"))
    mock_device.home = mock_home

    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device

    result = await async_unload_entry(fake_hass, mock_config_entry)

    assert result is True
