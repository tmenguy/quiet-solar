from quiet_solar.ha_model.device import HADeviceMixin
from quiet_solar.home_model.battery import Battery
from homeassistant.const import Platform

class QSBattery(HADeviceMixin, Battery):

    def get_platforms(self):
        return [ Platform.SENSOR]
