"""`QSRadiator` — heating-only bistate-duration load with dual backing.

A radiator is a `QSBiStateDuration` that flips its heater on/off for a
target duration each day. Unlike `QSOnOffDuration` (switch only) or
`QSClimateDuration` (climate only), a radiator can sit on EITHER a
switch entity OR a climate entity — the user picks one in the config
flow, and the constructor picks the matching `BistateTransport`.

The class is the dedicated host for future radiator-specific logic
(e.g. temperature-aware duration, scheduled heating profiles). For
this initial story it only adds:
  - the heating-only HVAC defaults (`heat` / `off`)
  - the exactly-one-backing validation (raises
    `ServiceValidationError`)
  - the dedicated translation keys (`radiator_mode`,
    `qs_constraint_sensor_radiator`)
  - the dedicated dashboard card (`qs-radiator-card`)

It does NOT expose `climate_state_on` / `climate_state_off` properties —
those run-time selects are removed for radiators (config-time only). A
user who wants seasonal mode flipping should use the `climate` load
type instead.
"""

from __future__ import annotations

import logging

from homeassistant.exceptions import ServiceValidationError

from ..const import (
    CONF_CLIMATE,
    CONF_CLIMATE_HVAC_MODE_OFF,
    CONF_CLIMATE_HVAC_MODE_ON,
    CONF_SWITCH,
    SENSOR_CONSTRAINT_SENSOR_RADIATOR,
    CONF_TYPE_NAME_QSRadiator,
)
from .bistate_duration import QSBiStateDuration
from .bistate_transport import BistateTransport, ClimateTransport, SwitchTransport

_LOGGER = logging.getLogger(__name__)


class QSRadiator(QSBiStateDuration):
    """Heating-only bistate radiator backed by a switch OR a climate entity."""

    conf_type_name = CONF_TYPE_NAME_QSRadiator

    def __init__(self, **kwargs):
        # N6 — normalise whitespace-only entity_ids so they're treated as
        # absent (YAML imports occasionally yield `"   "`).
        raw_climate = kwargs.get(CONF_CLIMATE)
        raw_switch = kwargs.get(CONF_SWITCH)
        climate_entity = raw_climate.strip() if isinstance(raw_climate, str) else raw_climate
        switch_entity = raw_switch.strip() if isinstance(raw_switch, str) else raw_switch

        # M4 — use truthy checks consistently with the transport-picker
        # below; `bool("")` is `False`, so an empty-string backing is
        # treated as "not set". This keeps XOR and transport-selection
        # in agreement.
        if bool(climate_entity) == bool(switch_entity):
            raise ServiceValidationError("Radiator requires exactly one of climate or switch backing")

        # S2 — Build the transport BEFORE calling `super().__init__` so
        # the base class can observe a fully-formed `_transport` during
        # any future initialisation hook. The host-owned `_state_on/off`
        # are re-pinned AFTER super() because the base ctor
        # unconditionally seeds them to `"on"` / `"off"`.
        self._transport: BistateTransport
        if climate_entity:
            # S9 — `kwargs.get(..., default)` only fires the default when
            # the key is MISSING; a persisted `""` would otherwise slip
            # through and crash `set_hvac_mode("")`. Use `or` to also
            # fall back on empty/None.
            hvac_on = kwargs.get(CONF_CLIMATE_HVAC_MODE_ON) or "heat"
            hvac_off = kwargs.get(CONF_CLIMATE_HVAC_MODE_OFF) or "off"
            self._transport = ClimateTransport(climate_entity, hvac_on, hvac_off)
        else:
            self._transport = SwitchTransport(switch_entity)

        super().__init__(**kwargs)

        # Re-pin host-owned state AFTER `super().__init__` because the
        # base constructor seeds `_state_on/off`, `_bistate_mode_*`, and
        # `bistate_entity` to switch defaults — they must reflect the
        # radiator's chosen transport instead.
        self._state_on = self._transport.default_state_on()
        self._state_off = self._transport.default_state_off()
        self._bistate_mode_on = self._state_on
        self._bistate_mode_off = self._state_off
        self.bistate_entity = self._transport.entity

    def get_virtual_current_constraint_translation_key(self) -> str | None:
        return SENSOR_CONSTRAINT_SENSOR_RADIATOR

    def get_select_translation_key(self) -> str | None:
        """Return the translation key for the bistate-mode select."""
        return "radiator_mode"

    # `execute_command_system` is inherited from `QSBiStateDuration` and
    # delegates to `self._transport.execute(...)` (N2 review-fix).
