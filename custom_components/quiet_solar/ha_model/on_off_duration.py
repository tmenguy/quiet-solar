import logging

from ..const import SENSOR_CONSTRAINT_SENSOR_ON_OFF, CONF_TYPE_NAME_QSOnOffDuration
from .bistate_duration import QSBiStateDuration
from .bistate_transport import SwitchTransport

_LOGGER = logging.getLogger(__name__)


class QSOnOffDuration(QSBiStateDuration):
    conf_type_name = CONF_TYPE_NAME_QSOnOffDuration

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._state_on = "on"
        self._state_off = "off"
        self._bistate_mode_on = "on_off_mode_on"
        self._bistate_mode_off = "on_off_mode_off"
        self.bistate_entity = self.switch_entity
        self._transport: SwitchTransport = SwitchTransport(self.switch_entity)

    def get_virtual_current_constraint_translation_key(self) -> str | None:
        return SENSOR_CONSTRAINT_SENSOR_ON_OFF

    def get_select_translation_key(self) -> str | None:
        """return the translation key for the select"""
        return "on_off_mode"

    # `execute_command_system` is inherited from `QSBiStateDuration` and
    # delegates to `self._transport.execute(...)` (N2 review-fix).
