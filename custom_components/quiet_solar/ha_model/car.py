from quiet_solar.ha_model.device import HADeviceMixin
from quiet_solar.home_model.load import AbstractDevice
from homeassistant.const import Platform

class QSCar(HADeviceMixin, AbstractDevice):

    def get_platforms(self):
        return [ Platform.SENSOR, Platform.SELECT ]