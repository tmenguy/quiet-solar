import logging

from ..const import CONF_SWITCH, SENSOR_CONSTRAINT_SENSOR_ON_OFF, CONF_TYPE_NAME_QSOnOffDuration
from .bistate_duration import QSBiStateDuration
from .bistate_transport import SwitchTransport

_LOGGER = logging.getLogger(__name__)


class QSOnOffDuration(QSBiStateDuration):
    conf_type_name = CONF_TYPE_NAME_QSOnOffDuration

    def __init__(self, **kwargs):
        # Build the transport BEFORE `super().__init__` so the inherited
        # `_state_on/_state_off` property shadow can delegate to it
        # during the base ctor's seed assignments — symmetric with
        # `QSClimateDuration` and `QSRadiator`.
        # NOTE: we `kwargs.get` (NOT `.pop`) because `AbstractLoad.__init__`
        # itself pops `CONF_SWITCH` to populate `self.switch_entity`. If
        # we popped here, `AbstractLoad`'s pop would default to `None`
        # and clobber the attribute downstream code (e.g. `bistate_entity`)
        # relies on. Climate's `kwargs.pop(CONF_CLIMATE)` is safe because
        # `AbstractLoad` doesn't know about that key.
        switch_entity = kwargs.get(CONF_SWITCH)
        self._transport: SwitchTransport = SwitchTransport(switch_entity)

        super().__init__(**kwargs)

        self._state_on = "on"
        self._state_off = "off"
        self._bistate_mode_on = "on_off_mode_on"
        self._bistate_mode_off = "on_off_mode_off"
        self.bistate_entity = self.switch_entity

    def get_virtual_current_constraint_translation_key(self) -> str | None:
        return SENSOR_CONSTRAINT_SENSOR_ON_OFF

    def get_select_translation_key(self) -> str | None:
        """return the translation key for the select"""
        return "on_off_mode"

    # `execute_command_system` is inherited from `QSBiStateDuration` and
    # delegates to `self._transport.execute(...)` (N2 review-fix).
