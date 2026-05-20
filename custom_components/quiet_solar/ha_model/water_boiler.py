"""Quiet Solar water boiler (cumulus) load type."""

from __future__ import annotations

from typing import Any

from ..const import (
    CONF_WATER_BOILER_TEMPERATURE_SENSOR,
    CONF_TYPE_NAME_QSWaterBoiler,
)
from .on_off_duration import QSOnOffDuration


class QSWaterBoiler(QSOnOffDuration):
    """Water boiler (cumulus or thermodynamic boiler) load.

    Thin subclass of QSOnOffDuration with a dedicated config step,
    dashboard section, select-mode translation key, and an optional
    water-tank temperature sensor. Temperature-aware control logic is
    deferred to a follow-up story; this PR only attaches the probe
    so the data is available in the device-mixin state ring buffer.
    """

    conf_type_name = CONF_TYPE_NAME_QSWaterBoiler

    def __init__(self, **kwargs: Any) -> None:
        self.water_boiler_temperature_sensor: str | None = kwargs.pop(CONF_WATER_BOILER_TEMPERATURE_SENSOR, None)
        super().__init__(**kwargs)
        # attach_ha_state_to_probe early-returns on None; no guard
        # needed here. Mirrors pool.py:34.
        self.attach_ha_state_to_probe(self.water_boiler_temperature_sensor, is_numerical=True)

    def get_select_translation_key(self) -> str | None:
        """Return the dedicated translation key for the bistate-mode select."""
        return "water_boiler_mode"
