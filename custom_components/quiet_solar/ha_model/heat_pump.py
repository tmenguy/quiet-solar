import logging

from homeassistant.const import Platform

from ..const import CONF_TYPE_NAME_QSHeatPump
from ..ha_model.device import HADeviceMixin
from ..home_model.load import PilotedDevice

_LOGGER = logging.getLogger(__name__)


class QSHeatPump(HADeviceMixin, PilotedDevice):
    """
    Class to manage a heat pump device.
    """
    conf_type_name = CONF_TYPE_NAME_QSHeatPump

    def get_platforms(self):

        parent = super().get_platforms()
        if parent is None:
            parent = set()
        else:
            parent = set(parent)
        parent.update([Platform.BINARY_SENSOR])
        return list(parent)
