"""Test that WallboxChargerStatus fallback works when wallbox package is absent."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch


def test_wallbox_charger_status_fallback():
    """WallboxChargerStatus fallback enum is used when wallbox is not installed."""
    charger_mod = "custom_components.quiet_solar.ha_model.charger"
    wallbox_const = "homeassistant.components.wallbox.const"
    wallbox_init = "homeassistant.components.wallbox"

    saved_modules = {}
    for key in list(sys.modules):
        if key == charger_mod or key.startswith(charger_mod + "."):
            saved_modules[key] = sys.modules.pop(key)

    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def mock_import(name, *args, **kwargs):
        if name in (wallbox_const, wallbox_init) or name == "wallbox":
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    try:
        with patch("builtins.__import__", side_effect=mock_import):
            mod = importlib.import_module(charger_mod)
            fallback_cls = mod.WallboxChargerStatus
            assert fallback_cls.CHARGING.value == "Charging"
            assert fallback_cls.DISCONNECTED.value == "Disconnected"
            assert fallback_cls.UNKNOWN.value == "Unknown"
            assert fallback_cls.WAITING_FOR_CAR.value == "Waiting for car demand"
            assert fallback_cls.PAUSED.value == "Paused"
            assert fallback_cls.SCHEDULED.value == "Scheduled"
            assert fallback_cls.ERROR.value == "Error"
            assert fallback_cls.UPDATING.value == "Updating"
            assert fallback_cls.WAITING_IN_QUEUE_POWER_SHARING.value == "Waiting in queue by Power Sharing"
            assert fallback_cls.WAITING_IN_QUEUE_POWER_BOOST.value == "Waiting in queue by Power Boost"
            assert fallback_cls.WAITING_MID_FAILED.value == "Waiting MID failed"
            assert fallback_cls.WAITING_MID_SAFETY.value == "Waiting MID safety margin exceeded"
            assert fallback_cls.WAITING_IN_QUEUE_ECO_SMART.value == "Waiting in queue by Eco-Smart"
            assert fallback_cls.DISCHARGING.value == "Discharging"
            assert fallback_cls.WAITING.value == "Waiting"
            assert fallback_cls.READY.value == "Ready"
            assert fallback_cls.LOCKED.value == "Locked"
            assert fallback_cls.LOCKED_CAR_CONNECTED.value == "Locked, car connected"
    finally:
        sys.modules.pop(charger_mod, None)
        sys.modules.update(saved_modules)
