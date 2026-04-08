"""Tests for button platform."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from custom_components.quiet_solar.button import (
    QSButtonEntity,
    QSButtonEntityDescription,
    async_setup_entry,
    async_unload_entry,
    create_ha_button,
    create_ha_button_for_AbstractLoad,
    create_ha_button_for_QSCar,
    create_ha_button_for_QSChargerGeneric,
    create_ha_button_for_QSHome,
    create_ha_button_for_QSSolar,
)
from custom_components.quiet_solar.const import (
    BUTTON_SOLAR_COMPUTE_DAMPENING_1DAY,
    BUTTON_SOLAR_COMPUTE_DAMPENING_7DAY,
    BUTTON_SOLAR_RECOMPUTE_FORECAST_SCORES,
    BUTTON_SOLAR_RESET_DAMPENING,
    DOMAIN,
)
from tests.factories import create_minimal_home_model
from tests.test_helpers import create_mock_device


def test_create_ha_button_for_home():
    """Test creating buttons for home device."""
    mock_home = create_mock_device("home")
    mock_home.data_handler = MagicMock()
    mock_home.reset_forecasts = AsyncMock()
    mock_home.dump_for_debug = AsyncMock()
    mock_home.generate_yaml_for_dashboard = AsyncMock()

    entities = create_ha_button_for_QSHome(mock_home)

    assert len(entities) == 5  # Reset history, serialize debug, generate yaml
    assert all(isinstance(e, QSButtonEntity) for e in entities)


def test_create_ha_button_for_charger():
    """Test creating buttons for charger device."""
    mock_charger = create_mock_device("charger")
    mock_charger.data_handler = MagicMock()
    mock_charger.user_force_charge_now = AsyncMock()
    mock_charger.user_add_default_charge = AsyncMock()
    mock_charger.can_force_a_charge_now = MagicMock(return_value=True)
    mock_charger.can_add_default_charge = MagicMock(return_value=True)

    entities = create_ha_button_for_QSChargerGeneric(mock_charger)

    assert len(entities) == 2  # Force charge now, add default charge
    assert all(isinstance(e, QSButtonEntity) for e in entities)


def test_create_ha_button_for_car():
    """Test creating buttons for car device."""
    mock_car = create_mock_device("car")
    mock_car.data_handler = MagicMock()
    mock_car.user_force_charge_now = AsyncMock()
    mock_car.user_add_default_charge = AsyncMock()
    mock_car.user_clean_and_reset = AsyncMock()
    mock_car.can_force_a_charge_now = MagicMock(return_value=True)
    mock_car.can_add_default_charge = MagicMock(return_value=True)

    entities = create_ha_button_for_QSCar(mock_car)

    assert len(entities) == 2  # Force charge, add default


def test_create_ha_button_for_load():
    """Test creating buttons for load device."""
    from custom_components.quiet_solar.home_model.load import AbstractLoad

    mock_load = MagicMock(spec=AbstractLoad)
    mock_load.data_handler = MagicMock()
    mock_load.device_id = "test_load"
    mock_load.device_type = "load"
    mock_load.name = "Test Load"
    mock_load.qs_enable_device = True
    mock_load.mark_current_constraint_has_done = AsyncMock()
    mock_load.user_clean_and_reset = AsyncMock()
    mock_load.async_reset_override_state = AsyncMock()
    mock_load.support_user_override = MagicMock(return_value=True)

    entities = create_ha_button_for_AbstractLoad(mock_load)

    assert len(entities) == 2  # Mark done,  reset override


def test_create_ha_button_for_load_no_override():
    """Test creating buttons for load without override support."""
    from custom_components.quiet_solar.home_model.load import AbstractLoad

    mock_load = MagicMock(spec=AbstractLoad)
    mock_load.data_handler = MagicMock()
    mock_load.device_id = "test_load"
    mock_load.device_type = "load"
    mock_load.name = "Test Load"
    mock_load.qs_enable_device = True
    mock_load.mark_current_constraint_has_done = AsyncMock()
    mock_load.user_clean_and_reset = AsyncMock()
    mock_load.support_user_override = MagicMock(return_value=False)

    entities = create_ha_button_for_AbstractLoad(mock_load)

    assert len(entities) == 1  # Only mark done and clean/reset


def test_qs_button_entity_init():
    """Test QSButtonEntity initialization."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_description = MagicMock()
    mock_description.key = "test_button"
    mock_description.name = None
    mock_description.translation_key = "test"
    mock_description.is_available = None
    mock_description.async_press = AsyncMock()

    button = QSButtonEntity(mock_handler, mock_device, mock_description)

    assert button.device == mock_device
    assert button.entity_description == mock_description


@pytest.mark.asyncio
async def test_qs_button_entity_press():
    """Test button press calls async_press function."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_handler.force_update_all = AsyncMock()
    mock_device = create_mock_device("test")

    press_called = False

    async def mock_press(entity):
        nonlocal press_called
        press_called = True

    mock_description = QSButtonEntityDescription(key="test_button", translation_key="test", async_press=mock_press)

    button = QSButtonEntity(mock_handler, mock_device, mock_description)

    await button.async_press()

    assert press_called is True


@pytest.mark.asyncio
async def test_qs_button_entity_press_with_home():
    """Test button press triggers force_update_all when device.home is set."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_home = MagicMock()
    mock_home.force_update_all = AsyncMock()
    mock_device = create_mock_device("test", home=mock_home)

    mock_description = QSButtonEntityDescription(
        key="test_button",
        translation_key="test",
        async_press=AsyncMock(),
    )

    button = QSButtonEntity(mock_handler, mock_device, mock_description)
    await button.async_press()

    mock_home.force_update_all.assert_awaited_once()


def test_qs_button_entity_availability_disabled_device():
    """Test button unavailable when device disabled."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test", qs_enable_device=False)
    mock_description = MagicMock()
    mock_description.key = "test_button"
    mock_description.name = None
    mock_description.translation_key = "test"
    mock_description.is_available = None
    mock_description.async_press = AsyncMock()

    button = QSButtonEntity(mock_handler, mock_device, mock_description)
    button._set_availabiltiy()

    assert button._attr_available is False


def test_qs_button_entity_availability_custom_function():
    """Test button availability with custom function."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test", qs_enable_device=True)
    mock_description = MagicMock()
    mock_description.key = "test_button"
    mock_description.name = None
    mock_description.translation_key = "test"
    mock_description.is_available = lambda entity: entity.device.name == "Mock Device"
    mock_description.async_press = AsyncMock()

    button = QSButtonEntity(mock_handler, mock_device, mock_description)
    button._set_availabiltiy()

    assert button._attr_available is True


def test_qs_button_entity_availability_custom_function_false():
    """Test button unavailable when custom function returns False."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test", qs_enable_device=True)
    mock_description = MagicMock()
    mock_description.key = "test_button"
    mock_description.name = None
    mock_description.translation_key = "test"
    mock_description.is_available = lambda entity: False
    mock_description.async_press = AsyncMock()

    button = QSButtonEntity(mock_handler, mock_device, mock_description)
    button._set_availabiltiy()

    assert button._attr_available is False


def test_qs_button_entity_update_callback():
    """Test button update callback."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_description = MagicMock()
    mock_description.key = "test_button"
    mock_description.name = None
    mock_description.translation_key = "test"
    mock_description.is_available = None
    mock_description.async_press = AsyncMock()

    button = QSButtonEntity(mock_handler, mock_device, mock_description)
    button.hass = mock_handler.hass  # Set hass on the button entity
    button.async_write_ha_state = MagicMock()

    test_time = datetime.now(pytz.UTC)
    button.async_update_callback(test_time)

    button.async_write_ha_state.assert_called_once()


@pytest.mark.asyncio
async def test_async_setup_entry(fake_hass, mock_config_entry):
    """Test button platform setup."""
    mock_device = create_mock_device("home")
    mock_device.data_handler = MagicMock()
    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device

    mock_add_entities = MagicMock()

    with patch("custom_components.quiet_solar.button.create_ha_button", return_value=[MagicMock()]):
        await async_setup_entry(fake_hass, mock_config_entry, mock_add_entities)

        mock_add_entities.assert_called_once()


@pytest.mark.asyncio
async def test_async_setup_entry_no_device(fake_hass, mock_config_entry):
    """Test button platform setup with no device."""
    mock_add_entities = MagicMock()

    await async_setup_entry(fake_hass, mock_config_entry, mock_add_entities)

    mock_add_entities.assert_not_called()


@pytest.mark.asyncio
async def test_async_unload_entry(fake_hass, mock_config_entry):
    """Test button platform unload."""
    mock_device = create_mock_device("test")
    mock_home = create_minimal_home_model()
    mock_device.home = mock_home
    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device

    result = await async_unload_entry(fake_hass, mock_config_entry)

    assert result is True
    mock_home.remove_device.assert_called_once()


@pytest.mark.asyncio
async def test_async_unload_entry_handles_exception(fake_hass, mock_config_entry):
    """Test button platform unload handles exceptions."""
    mock_device = create_mock_device("test")
    mock_home = create_minimal_home_model()
    mock_home.remove_device = MagicMock(side_effect=Exception("Test error"))
    mock_device.home = mock_home
    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device

    result = await async_unload_entry(fake_hass, mock_config_entry)

    assert result is True


@pytest.mark.asyncio
async def test_qs_button_entity_press_force_update_all_exception():
    """Test button press logs error when force_update_all raises."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_home = MagicMock()
    mock_home.force_update_all = AsyncMock(side_effect=RuntimeError("update failed"))
    mock_device = create_mock_device("test", home=mock_home)

    mock_description = QSButtonEntityDescription(
        key="test_button",
        translation_key="test",
        async_press=AsyncMock(),
    )

    button = QSButtonEntity(mock_handler, mock_device, mock_description)

    with patch("custom_components.quiet_solar.button._LOGGER") as mock_logger:
        await button.async_press()

    mock_home.force_update_all.assert_awaited_once()
    mock_logger.error.assert_called_once()
    assert "force_update_all failed" in mock_logger.error.call_args[0][0]


def test_create_ha_button_for_solar():
    """Test creating buttons for solar device."""
    from custom_components.quiet_solar.ha_model.solar import QSSolar

    mock_solar = MagicMock(spec=QSSolar)
    mock_solar.data_handler = MagicMock()
    mock_solar.device_id = "test_solar"
    mock_solar.device_type = "solar"
    mock_solar.name = "Test Solar"
    mock_solar.qs_enable_device = True
    mock_solar.force_scoring_cycle = AsyncMock()

    entities = create_ha_button_for_QSSolar(mock_solar)

    assert len(entities) == 4
    assert all(isinstance(e, QSButtonEntity) for e in entities)
    keys = [e.entity_description.key for e in entities]
    assert BUTTON_SOLAR_RECOMPUTE_FORECAST_SCORES in keys
    assert BUTTON_SOLAR_COMPUTE_DAMPENING_1DAY in keys
    assert BUTTON_SOLAR_COMPUTE_DAMPENING_7DAY in keys
    assert BUTTON_SOLAR_RESET_DAMPENING in keys


def test_create_ha_button_dispatcher_includes_solar():
    """Test that create_ha_button dispatcher includes QSSolar buttons."""
    from custom_components.quiet_solar.ha_model.solar import QSSolar

    mock_solar = MagicMock(spec=QSSolar)
    mock_solar.data_handler = MagicMock()
    mock_solar.device_id = "test_solar"
    mock_solar.device_type = "solar"
    mock_solar.name = "Test Solar"
    mock_solar.qs_enable_device = True
    mock_solar.force_scoring_cycle = AsyncMock()

    buttons = create_ha_button(mock_solar)

    # Should include the solar button plus the AbstractDevice buttons
    solar_buttons = [b for b in buttons if b.entity_description.key == BUTTON_SOLAR_RECOMPUTE_FORECAST_SCORES]
    assert len(solar_buttons) == 1


@pytest.mark.asyncio
async def test_solar_button_press_calls_force_scoring_cycle():
    """Test that pressing the solar button calls force_scoring_cycle on the device."""
    from custom_components.quiet_solar.ha_model.solar import QSSolar

    mock_solar = MagicMock(spec=QSSolar)
    mock_solar.data_handler = MagicMock()
    mock_solar.data_handler.hass = MagicMock()
    mock_solar.device_id = "test_solar"
    mock_solar.device_type = "solar"
    mock_solar.name = "Test Solar"
    mock_solar.qs_enable_device = True
    mock_solar.home = None
    mock_solar.force_scoring_cycle = AsyncMock()

    entities = create_ha_button_for_QSSolar(mock_solar)
    button = entities[0]

    await button.entity_description.async_press(button)

    mock_solar.force_scoring_cycle.assert_awaited_once()
