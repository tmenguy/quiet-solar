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
        # N6 + E1 — normalise whitespace-only entity_ids IN PLACE so
        # `super().__init__(**kwargs)` (which stores `switch_entity` on
        # `AbstractLoad`) doesn't see a stale `"   "` value the base
        # would treat as a valid entity id.
        for key in (CONF_CLIMATE, CONF_SWITCH):
            raw = kwargs.get(key)
            if isinstance(raw, str):
                stripped = raw.strip()
                kwargs[key] = stripped if stripped else None

        climate_entity = kwargs.get(CONF_CLIMATE)
        switch_entity = kwargs.get(CONF_SWITCH)

        # M4 + B8 — explicit XOR for readability: exactly one of the
        # two backings must be set. `bool("") is False` so an empty
        # entity id is treated as "not set", keeping XOR and the
        # transport-selection branch below in agreement.
        both_set = bool(climate_entity) and bool(switch_entity)
        neither_set = not climate_entity and not switch_entity
        if both_set or neither_set:
            raise ServiceValidationError("Radiator requires exactly one of climate or switch backing")

        # S2 — Build the transport BEFORE calling `super().__init__` so
        # the inherited `_state_on/_state_off` property shadow can
        # delegate to it during the base ctor's seed assignments. We
        # capture the intended state values now because `super()` is
        # about to overwrite the transport's `state_on/state_off` via
        # `self._state_on = "on"` — and we want to restore them after.
        self._transport: BistateTransport
        if climate_entity:
            # S9 — `kwargs.get(..., default)` only fires the default
            # when the key is MISSING; a persisted `""` would otherwise
            # slip through and crash `set_hvac_mode("")`. Use `or` to
            # also fall back on empty/None.
            intended_state_on = kwargs.get(CONF_CLIMATE_HVAC_MODE_ON) or "heat"
            intended_state_off = kwargs.get(CONF_CLIMATE_HVAC_MODE_OFF) or "off"
            self._transport = ClimateTransport(climate_entity, intended_state_on, intended_state_off)
        else:
            intended_state_on = "on"
            intended_state_off = "off"
            self._transport = SwitchTransport(switch_entity)

        super().__init__(**kwargs)

        # Restore the chosen transport modes AFTER `super().__init__`
        # (the base ctor wrote `"on"` / `"off"` into the transport via
        # the inherited property shadow; for climate radiators we need
        # to bring back the real HVAC modes).
        self._state_on = intended_state_on
        self._state_off = intended_state_off
        # E3 — `_bistate_mode_on/off` drive the `radiator_mode` select
        # UI. Hard-code them to `"on"`/`"off"` regardless of the HVAC
        # mode so the translation always has matching state keys (the
        # `radiator_mode` translation defines `on` and `off` but not
        # arbitrary HVAC modes like `auto`, `fan_only`, `dry`). The
        # transport's `state_on/state_off` continue to carry the
        # actual HVAC mode strings for service-call dispatch.
        self._bistate_mode_on = "on"
        self._bistate_mode_off = "off"
        self.bistate_entity = self._transport.entity

    def get_virtual_current_constraint_translation_key(self) -> str | None:
        return SENSOR_CONSTRAINT_SENSOR_RADIATOR

    def get_select_translation_key(self) -> str | None:
        """Return the translation key for the bistate-mode select."""
        return "radiator_mode"

    # `execute_command_system` is inherited from `QSBiStateDuration` and
    # delegates to `self._transport.execute(...)` (N2 review-fix).
