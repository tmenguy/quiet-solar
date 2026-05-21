import logging

from homeassistant.components.climate import HVACMode

from ..const import (
    CONF_CLIMATE,
    CONF_CLIMATE_HVAC_MODE_OFF,
    CONF_CLIMATE_HVAC_MODE_ON,
    SENSOR_CONSTRAINT_SENSOR_CLIMATE,
    CONF_TYPE_NAME_QSClimateDuration,
)
from .bistate_duration import QSBiStateDuration
from .bistate_transport import ClimateTransport

_LOGGER = logging.getLogger(__name__)


class QSClimateDuration(QSBiStateDuration):
    conf_type_name = CONF_TYPE_NAME_QSClimateDuration

    def __init__(self, **kwargs):

        # S2/S5/B3 — build the transport BEFORE `super().__init__` so
        # the inherited `_state_on/_state_off` property shadow (in
        # `QSBiStateDuration`) can delegate to it during the base
        # ctor's seed assignments. The transport is the single source
        # of truth for the HVAC modes; the base ctor's
        # `self._state_on = "on"` lands as
        # `self._transport.state_on = "on"` and gets overwritten with
        # the actual HVAC mode below.
        climate_entity = kwargs.pop(CONF_CLIMATE, None)
        state_off = kwargs.pop(CONF_CLIMATE_HVAC_MODE_OFF, str(HVACMode.OFF.value))
        state_on = kwargs.pop(CONF_CLIMATE_HVAC_MODE_ON, str(HVACMode.AUTO.value))
        self.climate_entity = climate_entity
        self._transport: ClimateTransport = ClimateTransport(climate_entity, state_on, state_off)

        super().__init__(**kwargs)

        # B7 — route the post-super HVAC-mode re-pin through the
        # inherited property shadow (`self._state_on = …` updates the
        # transport). The base ctor seeded the transport with `"on"`/
        # `"off"` via the same path; we now restore the actual HVAC
        # modes the user picked.
        self._state_on = state_on
        self._state_off = state_off
        self._bistate_mode_on = state_on
        self._bistate_mode_off = state_off
        self.bistate_entity = climate_entity
        self.is_load_time_sensitive = True

    # `_state_on` / `_state_off` properties are inherited from
    # `QSBiStateDuration` (B3 review-fix moved them up so both
    # `QSClimateDuration` AND `QSRadiator` route writes through the
    # transport from a single definition).

    @property
    def climate_state_on(self):
        return self._state_on

    @climate_state_on.setter
    def climate_state_on(self, value):
        self._state_on = value  # routes through the inherited shadow
        self._bistate_mode_on = value

    @property
    def climate_state_off(self):
        return self._state_off

    @climate_state_off.setter
    def climate_state_off(self, value):
        self._state_off = value  # routes through the inherited shadow
        self._bistate_mode_off = value

    def get_possibles_modes(self):
        """return the possible modes for the climate entity"""
        return self._transport.mode_options(self.hass)

    def get_virtual_current_constraint_translation_key(self) -> str | None:
        return SENSOR_CONSTRAINT_SENSOR_CLIMATE

    def get_select_translation_key(self) -> str | None:
        """return the translation key for the select"""
        return "climate_mode"

    # `execute_command_system` is inherited from `QSBiStateDuration` and
    # delegates to `self._transport.execute(...)` (N2 review-fix). The
    # base passes `self._state_on/_state_off`, which thanks to the S5
    # property shadows defined above resolve to the transport's modes —
    # so this stays in sync with `climate_state_on/off` mutations.
