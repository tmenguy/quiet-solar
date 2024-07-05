import copy

from quiet_solar.ha_model.device import HADeviceMixin
from quiet_solar.home_model.load import AbstractDevice
from homeassistant.const import Platform

class QSCar(HADeviceMixin, AbstractDevice):

    def __init__(self, **kwargs):
        self.car_plugged = kwargs.pop("car_plugged")
        self.car_tracker = kwargs.pop("car_tracker")
        self.car_charge_percent_sensor = kwargs.pop("car_charge_percent_sensor")
        self.car_battery_capacity = kwargs.pop( "car_battery_capacity")
        self.car_charger_min_charge : int = int(max(0,kwargs.pop("car_charger_min_charge", 6)))
        self.car_charger_max_charge : int = int(max(0,kwargs.pop("car_charger_max_charge",32)))
        self.car_use_custom_power_charge_values = kwargs.pop("car_use_custom_power_charge_values", False)
        self.car_is_custom_power_charge_values_3p = kwargs.pop("car_is_custom_power_charge_values_3p", False)

        self.amp_to_power_1p = [-1]*(self.car_charger_max_charge + 1 - self.car_charger_min_charge)
        self.amp_to_power_3p = [-1]*(self.car_charger_max_charge + 1 - self.car_charger_min_charge)

        if self.car_use_custom_power_charge_values:

            for a in range(self.car_charger_min_charge, self.car_charger_max_charge + 1):
                val_1p = val_3p = float(kwargs.pop(f"charge_{a}", -1))
                if self.car_is_custom_power_charge_values_3p:
                    val_1p = val_3p / 3.0
                else:
                    val_3p = val_1p * 3.0
                self.amp_to_power_1p[a - self.car_charger_min_charge] = val_1p
                self.amp_to_power_3p[a - self.car_charger_min_charge] = val_3p

        super().__init__(**kwargs)

    def get_charge_power_per_phase_A(self, for_3p:bool) -> tuple[list[float], int, int]:
        if for_3p:
            return self.amp_to_power_3p, self.car_charger_min_charge, self.car_charger_max_charge
        else:
            return self.amp_to_power_1p, self.car_charger_min_charge, self.car_charger_max_charge


    def get_platforms(self):
        return [ Platform.SENSOR, Platform.SELECT ]