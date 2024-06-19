from quiet_solar.ha_model.device import HADeviceMixin
from quiet_solar.home_model.load import AbstractLoad
from homeassistant.const import Platform

class QSCharger(HADeviceMixin, AbstractLoad):

    def get_platforms(self):
        return [ Platform.SENSOR, Platform.SELECT ]

    def attach_car(self, car):
        pass

    def detach_car(self):
        pass



class QSChargerOCPP(QSCharger):
    pass

class QSChargerGeneric(QSCharger):
    pass

class QSChargerWallbox(QSCharger):
    pass