from homeassistant.const import Platform

from ..ha_model.device import HADeviceMixin
from ..home_model.load import AbstractLoad


class QSFPHeater(HADeviceMixin, AbstractLoad):

    def get_platforms(self):
        return [ Platform.SENSOR, Platform.SELECT ]