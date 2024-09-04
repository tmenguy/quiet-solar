from homeassistant.const import Platform

from ..ha_model.device import HADeviceMixin
from ..home_model.load import AbstractLoad


class QSFPHeater(HADeviceMixin, AbstractLoad):

    def get_platforms(self):
        parent = super().get_platforms()
        if parent is None:
            parent = set()
        else:
            parent = set(parent)
        parent.update([ Platform.SENSOR, Platform.SELECT ])
        return list(parent)