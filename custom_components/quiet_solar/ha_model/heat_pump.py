import logging

from homeassistant.const import Platform

from ..const import CONF_TYPE_NAME_QSHeatPump
from ..ha_model.device import HADeviceMixin
from ..home_model.load import PilotedDevice

_LOGGER = logging.getLogger(__name__)


class QSHeatPump(HADeviceMixin, PilotedDevice):
    """
    Class to manage a heat pump device.
    This is a placeholder for future implementation.
    """

    conf_type_name = CONF_TYPE_NAME_QSHeatPump

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Add heat pump specific initialization here

    def get_platforms(self):
        parent = super().get_platforms()
        parent.append(Platform.BINARY_SENSOR)
        return parent
