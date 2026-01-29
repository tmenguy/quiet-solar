"""Tests for switch platform."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import pytz

from homeassistant.config_entries import SOURCE_USER
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

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
from tests.factories import create_minimal_home_model


@pytest.fixture
def load_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """MockConfigEntry for load device, added to hass."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        source=SOURCE_USER,
        data={},
        entry_id="load_entry",
        title="Test Load",
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def switch_platform_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """MockConfigEntry for switch platform setup/unload tests."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        source=SOURCE_USER,
        data={},
        entry_id="test_entry",
        title="Test Device",
    )
    entry.add_to_hass(hass)
    return entry


def _create_load_with_handler(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    support_green_only_switch: bool,
) -> _TestLoadHA:
    """Create a load with data handler and minimal home (using factory)."""
    hass.data.setdefault(DOMAIN, {})[DATA_HANDLER] = MagicMock(hass=hass)
    home = create_minimal_home_model()
    return _TestLoadHA(
        hass=hass,
        config_entry=config_entry,
        home=home,
        name="Test Load",
        device_type="load",
        support_green_only_switch=support_green_only_switch,
    )


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


def test_create_ha_switch_for_pool():
    """Test creating switches for pool."""
    from custom_components.quiet_solar.ha_model.pool import QSPool

    mock_pool = create_mock_device("pool")
    mock_pool.__class__ = QSPool
    entities = create_ha_switch_for_QSPool(mock_pool)

    assert entities == []


def test_create_ha_switch_includes_pool_branch():
    """Test create_ha_switch covers pool branch."""
    from custom_components.quiet_solar.ha_model.pool import QSPool

    mock_pool = create_mock_device("pool")
    mock_pool.__class__ = QSPool
    with patch("custom_components.quiet_solar.switch.create_ha_switch_for_QSPool", return_value=[] ) as pool_mock, \
        patch("custom_components.quiet_solar.switch.create_ha_switch_for_AbstractLoad", return_value=[]):
        entities = create_ha_switch(mock_pool)
        pool_mock.assert_called_once()
        assert entities == []


class _TestLoadHA(HADeviceMixin, TestLoad):
    """Test load supporting HADeviceMixin for switch creation."""

    def __init__(self, *args, support_green_only_switch: bool = False, **kwargs):
        self._support_green_only_switch = support_green_only_switch
        super().__init__(*args, **kwargs)

    def support_green_only_switch(self) -> bool:
        return self._support_green_only_switch


def test_create_ha_switch_for_load(
    hass: HomeAssistant, load_config_entry: MockConfigEntry
):
    """Test creating switches for load with green-only support."""
    load = _create_load_with_handler(hass, load_config_entry, support_green_only_switch=True)
    entities = create_ha_switch_for_AbstractLoad(load)

    keys = {entity.entity_description.key for entity in entities}
    assert keys == {SWITCH_BEST_EFFORT_GREEN_ONLY, SWITCH_ENABLE_DEVICE}


def test_create_ha_switch_for_load_no_green_only(
    hass: HomeAssistant, load_config_entry: MockConfigEntry
):
    """Test creating switches for load without green-only support."""
    load = _create_load_with_handler(hass, load_config_entry, support_green_only_switch=False)
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
    mock_device.home = create_minimal_home_model()
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
    mock_device = MagicMock(spec=[])  # Empty spec - only has explicit attrs
    mock_device.device_id = "test_device"
    mock_device.device_type = "test"
    mock_device.name = "Test Device"
    mock_device.qs_enable_device = True
    mock_device.home = None

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


def test_qs_switch_entity_charger_or_car_helpers():
    """Test charger/car helper accessors."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric
    from custom_components.quiet_solar.ha_model.car import QSCar

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
    assert switch.car() is mock_charger.car
    assert switch.charger() is mock_charger

    mock_car = MagicMock(spec=QSCar)
    mock_car.charger = mock_charger
    mock_car.qs_enable_device = True
    mock_car.device_id = "car_123"
    mock_car.device_type = "car"
    mock_car.name = "Car"

    switch = QSSwitchEntityChargerOrCar(mock_handler, mock_car, mock_description)
    assert switch.car() is mock_car
    assert switch.charger() is mock_car.charger

    unknown_device = MagicMock()
    switch = QSSwitchEntityChargerOrCar(mock_handler, unknown_device, mock_description)
    assert switch.car() is None
    assert switch.charger() is None


def test_qs_switch_entity_car_availability_charger_disabled():
    """Test car switch unavailable when charger disabled."""
    from custom_components.quiet_solar.ha_model.car import QSCar

    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()

    mock_charger = MagicMock()
    mock_charger.qs_enable_device = False

    mock_car = MagicMock(spec=QSCar)
    mock_car.charger = mock_charger
    mock_car.qs_enable_device = True
    mock_car.device_id = "car_123"
    mock_car.device_type = "car"
    mock_car.name = "Car"

    mock_description = MagicMock()
    mock_description.key = "test_switch"
    mock_description.name = None
    mock_description.translation_key = "test"
    mock_description.set_val = None

    switch = QSSwitchEntityChargerOrCar(mock_handler, mock_car, mock_description)
    switch._set_availabiltiy()
    assert switch._attr_available is False


@pytest.mark.asyncio
async def test_async_setup_entry(
    hass: HomeAssistant, switch_platform_config_entry: MockConfigEntry
):
    """Test switch platform setup."""
    mock_device = create_mock_device("test")
    mock_device.data_handler = MagicMock()
    hass.data.setdefault(DOMAIN, {})[switch_platform_config_entry.entry_id] = mock_device

    mock_add_entities = MagicMock()

    with patch(
        "custom_components.quiet_solar.switch.create_ha_switch",
        return_value=[MagicMock()],
    ):
        await async_setup_entry(hass, switch_platform_config_entry, mock_add_entities)

        mock_add_entities.assert_called_once()


@pytest.mark.asyncio
async def test_async_unload_entry_handles_exception(
    hass: HomeAssistant, switch_platform_config_entry: MockConfigEntry
):
    """Test switch platform unload handles errors."""
    mock_device = create_mock_device("test")
    mock_home = create_minimal_home_model()
    mock_home.remove_device = MagicMock(side_effect=RuntimeError("boom"))
    mock_device.home = mock_home

    hass.data.setdefault(DOMAIN, {})[
        switch_platform_config_entry.entry_id
    ] = mock_device

    result = await async_unload_entry(hass, switch_platform_config_entry)

    assert result is True
