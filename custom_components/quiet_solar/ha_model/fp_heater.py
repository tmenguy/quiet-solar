from quiet_solar.ha_model.device import HADeviceMixin
from quiet_solar.home_model.load import AbstractLoad
from homeassistant.const import Platform

class QSFPHeater(HADeviceMixin, AbstractLoad):

    def get_platforms(self):
        return [ Platform.SENSOR, Platform.SELECT ]