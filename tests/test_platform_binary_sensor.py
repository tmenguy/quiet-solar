"""Tests for binary sensor platform."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.quiet_solar.binary_sensor import (
    QSBaseBinarySensor,
    QSBinarySensorEntityDescription,
    async_unload_entry,
)
from custom_components.quiet_solar.const import DOMAIN


def _make_device() -> MagicMock:
    device = MagicMock()
    device.device_id = "device_1"
    device.device_type = "car"
    device.name = "Device"
    device.qs_enable_device = True
    device.home = MagicMock()
    return device


@pytest.mark.asyncio
async def test_binary_sensor_update_callback_value_fn():
    """Test binary sensor updates with value_fn."""
    handler = MagicMock()
    handler.hass = MagicMock()
    device = _make_device()

    description = QSBinarySensorEntityDescription(
        key="flag",
        translation_key="flag",
        value_fn=lambda d, _k: d.qs_enable_device,
    )

    entity = QSBaseBinarySensor(handler, device, description)
    entity.async_write_ha_state = MagicMock()

    entity.async_update_callback(None)
    assert entity.is_on is True
    entity.async_write_ha_state.assert_called_once()


@pytest.mark.asyncio
async def test_binary_sensor_update_callback_attribute():
    """Test binary sensor updates from device attribute."""
    handler = MagicMock()
    handler.hass = MagicMock()
    device = _make_device()
    device.flag = False

    description = QSBinarySensorEntityDescription(
        key="flag",
        translation_key="flag",
    )

    entity = QSBaseBinarySensor(handler, device, description)
    entity.async_write_ha_state = MagicMock()

    entity.async_update_callback(None)
    assert entity.is_on is False
    entity.async_write_ha_state.assert_called_once()


@pytest.mark.asyncio
async def test_binary_sensor_unload_entry_handles_exception():
    """Test unload handles device removal errors."""
    hass = MagicMock()
    entry = MagicMock()
    entry.entry_id = "entry_1"

    device = _make_device()
    device.home.remove_device = MagicMock(side_effect=RuntimeError("boom"))
    hass.data = {DOMAIN: {entry.entry_id: device}}

    result = await async_unload_entry(hass, entry)
    assert result is True
