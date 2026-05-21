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
        raw = kwargs.pop(CONF_WATER_BOILER_TEMPERATURE_SENSOR, None)
        # Normalise empty string → None so downstream consumers
        # (templates, probe storage) only ever see a real entity id
        # or None. The option-flow form stores "" when the user clears
        # the optional field; pool.py's required field never hits this
        # case, which is why the pattern diverges by design.
        self.water_boiler_temperature_sensor: str | None = raw or None
        # super().__init__ initialises HADeviceMixin state (the probe
        # dicts/sets); attach_ha_state_to_probe below depends on it,
        # so the order here is load-bearing.
        super().__init__(**kwargs)
        # attach_ha_state_to_probe early-returns on None; no guard
        # needed here. Mirrors pool.py:34.
        self.attach_ha_state_to_probe(self.water_boiler_temperature_sensor, is_numerical=True)

    def get_select_translation_key(self) -> str | None:
        """Return the dedicated translation key for the bistate-mode select."""
        return "water_boiler_mode"
