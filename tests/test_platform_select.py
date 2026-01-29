"""Tests for select platform."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.select import DOMAIN as SELECT_DOMAIN, SERVICE_SELECT_OPTION
from homeassistant.const import ATTR_ENTITY_ID, ATTR_OPTION
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import entity_registry as er

from custom_components.quiet_solar.select import (
    QSBaseSelect,
    QSUserOverrideSelectRestore,
    QSExtraStoredDataSelect,
    QSSimpleSelectRestore,
    QSSelectEntityDescription,
    async_unload_entry,
)
from custom_components.quiet_solar.const import DOMAIN


class _DummyHome:
    def __init__(self) -> None:
        self.force_update_all = AsyncMock()


def _make_device() -> MagicMock:
    device = MagicMock()
    device.device_id = "device_1"
    device.device_type = "home"
    device.name = "Device"
    device.qs_enable_device = True
    device.home = _DummyHome()
    return device


@pytest.mark.asyncio
async def test_select_updates_device_attribute():
    """Test QSBaseSelect updates device attribute and state."""
    handler = MagicMock()
    handler.hass = MagicMock()
    device = _make_device()
    device.mode = "option_1"

    description = QSSelectEntityDescription(
        key="mode",
        translation_key="mode",
        options=["option_1", "option_2"],
    )

    entity = QSBaseSelect(handler, device, description)
    entity.async_write_ha_state = MagicMock()

    await entity.async_select_option("option_2")

    assert device.mode == "option_2"
    assert entity.current_option == "option_2"
    device.home.force_update_all.assert_awaited_once()
    entity.async_write_ha_state.assert_called_once()


@pytest.mark.asyncio
async def test_select_uses_custom_setter():
    """Test QSBaseSelect uses custom setter when provided."""
    handler = MagicMock()
    handler.hass = MagicMock()
    device = _make_device()

    setter = AsyncMock()
    description = QSSelectEntityDescription(
        key="mode",
        translation_key="mode",
        options=["option_1", "option_2"],
        async_set_current_option_fn=setter,
    )

    entity = QSBaseSelect(handler, device, description)
    entity.async_write_ha_state = MagicMock()

    await entity.async_select_option("option_1")

    setter.assert_awaited_once_with(device, "mode", "option_1")
    device.home.force_update_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_simple_select_restore_uses_default_option():
    """Test QSSimpleSelectRestore picks default when state is invalid."""
    handler = MagicMock()
    handler.hass = MagicMock()
    device = _make_device()
    device.mode = None

    description = QSSelectEntityDescription(
        key="mode",
        translation_key="mode",
        options=["option_1", "option_2"],
        qs_default_option="option_2",
    )

    entity = QSSimpleSelectRestore(handler, device, description)
    entity.async_select_option = AsyncMock()

    invalid_state = State("select.test", "unknown")
    with patch.object(entity, "async_get_last_state", return_value=invalid_state):
        await entity.async_added_to_hass()

    entity.async_select_option.assert_awaited_once_with("option_2")


@pytest.mark.asyncio
async def test_simple_select_restore_uses_first_option_when_missing():
    """Test QSSimpleSelectRestore falls back to first option."""
    handler = MagicMock()
    handler.hass = MagicMock()
    device = _make_device()
    device.mode = None

    description = QSSelectEntityDescription(
        key="mode",
        translation_key="mode",
        options=["option_1", "option_2"],
    )

    entity = QSSimpleSelectRestore(handler, device, description)
    entity.async_select_option = AsyncMock()

    with patch.object(entity, "async_get_last_state", return_value=None):
        await entity.async_added_to_hass()

    entity.async_select_option.assert_awaited_once_with("option_1")


@pytest.mark.asyncio
async def test_select_update_callback_without_hass():
    """Test update callback exits when hass is missing."""
    handler = MagicMock()
    handler.hass = MagicMock()
    device = _make_device()
    device.mode = "option_1"

    description = QSSelectEntityDescription(
        key="mode",
        translation_key="mode",
        options=["option_1", "option_2"],
    )

    entity = QSBaseSelect(handler, device, description)
    entity.hass = None
    entity.async_write_ha_state = MagicMock()

    entity.async_update_callback(None)
    entity.async_write_ha_state.assert_not_called()


@pytest.mark.asyncio
async def test_select_setter_handles_attribute_error():
    """Test QSBaseSelect handles attribute errors."""
    handler = MagicMock()
    handler.hass = MagicMock()

    class _Device:
        device_id = "device_2"
        device_type = "home"
        name = "Device"
        qs_enable_device = True
        home = _DummyHome()

        @property
        def mode(self):
            return "option_1"

        @mode.setter
        def mode(self, _val):
            raise AttributeError("fail")

    description = QSSelectEntityDescription(
        key="mode",
        translation_key="mode",
        options=["option_1", "option_2"],
    )

    entity = QSBaseSelect(handler, _Device(), description)
    entity.async_write_ha_state = MagicMock()

    await entity.async_select_option("option_2")
    entity.async_write_ha_state.assert_called_once()


@pytest.mark.asyncio
async def test_user_override_select_restore():
    """Test QSUserOverrideSelectRestore restores user selection."""
    handler = MagicMock()
    handler.hass = MagicMock()
    device = _make_device()
    device.mode = "option_1"

    description = QSSelectEntityDescription(
        key="mode",
        translation_key="mode",
        options=["option_1", "option_2"],
    )

    entity = QSUserOverrideSelectRestore(handler, device, description)
    entity.async_write_ha_state = MagicMock()

    extra = QSExtraStoredDataSelect("option_2")
    with patch.object(entity, "async_get_last_extra_data", return_value=extra):
        await entity.async_added_to_hass()

    assert entity.user_selected_option == "option_2"


@pytest.mark.asyncio
async def test_user_override_select_no_restore_data():
    """Test user override select with no restore data."""
    handler = MagicMock()
    handler.hass = MagicMock()
    device = _make_device()

    description = QSSelectEntityDescription(
        key="test_key",
        translation_key="test_key",
        options=["option_1", "option_2"],
    )

    entity = QSUserOverrideSelectRestore(handler, device, description)
    entity.async_get_last_extra_data = AsyncMock(return_value=None)
    entity.async_select_option = AsyncMock()

    await entity.async_added_to_hass()

    assert entity.user_selected_option is None
    entity.async_select_option.assert_called_once_with(None)


def test_user_override_select_extra_restore_data():
    """Test user override select extra restore data."""
    handler = MagicMock()
    handler.hass = MagicMock()
    device = _make_device()

    description = QSSelectEntityDescription(
        key="test_key",
        translation_key="test_key",
        options=["option_1", "option_2"],
    )

    entity = QSUserOverrideSelectRestore(handler, device, description)
    extra = entity.extra_restore_state_data

    assert isinstance(extra, QSExtraStoredDataSelect)

def test_extra_stored_data_from_dict_error():
    """Test QSExtraStoredDataSelect.from_dict handles errors."""
    assert QSExtraStoredDataSelect.from_dict({}) is None


@pytest.mark.asyncio
async def test_select_unload_handles_exception():
    """Test select unload handles exceptions."""
    hass = MagicMock()
    entry = MagicMock()
    entry.entry_id = "entry_1"

    device = _make_device()
    device.home.remove_device = MagicMock(side_effect=RuntimeError("boom"))
    hass.data = {DOMAIN: {entry.entry_id: device}}

    result = await async_unload_entry(hass, entry)
    assert result is True


@pytest.mark.asyncio
async def test_select_entities_in_ha(
    hass: HomeAssistant,
    real_home_config_entry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test select entities are created and selectable in HA."""
    with patch(
        "custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(real_home_config_entry.entry_id)
        await hass.async_block_till_done()

    entities = er.async_entries_for_config_entry(entity_registry, real_home_config_entry.entry_id)
    select_entities = [entry.entity_id for entry in entities if entry.entity_id.startswith("select.")]

    assert select_entities, "Expected select entities to be created"

    select_entity_id = select_entities[0]
    state = hass.states.get(select_entity_id)
    assert state is not None

    options = state.attributes.get("options", [])
    assert options, "Expected select entity options"

    await hass.services.async_call(
        SELECT_DOMAIN,
        SERVICE_SELECT_OPTION,
        {ATTR_ENTITY_ID: select_entity_id, ATTR_OPTION: options[0]},
        blocking=True,
    )

    updated_state = hass.states.get(select_entity_id)
    assert updated_state is not None
