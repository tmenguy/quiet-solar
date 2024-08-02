import copy
from datetime import datetime


from quiet_solar.const import CONF_CAR_PLUGGED, CONF_CAR_TRACKER, CONF_CAR_CHARGE_PERCENT_SENSOR, \
    CONF_CAR_BATTERY_CAPACITY, CONF_CAR_CHARGER_MIN_CHARGE, CONF_CAR_CHARGER_MAX_CHARGE, \
    CONF_CAR_CUSTOM_POWER_CHARGE_VALUES, CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P
from quiet_solar.ha_model.device import HADeviceMixin
from quiet_solar.home_model.load import AbstractDevice
from homeassistant.const import Platform

MIN_CHARGE_POWER_W = 150

class QSCar(HADeviceMixin, AbstractDevice):

    def __init__(self, **kwargs):
        self.car_plugged = kwargs.pop(CONF_CAR_PLUGGED, None)
        self.car_tracker = kwargs.pop(CONF_CAR_TRACKER, None)
        self.car_charge_percent_sensor = kwargs.pop(CONF_CAR_CHARGE_PERCENT_SENSOR, None)
        self.car_battery_capacity = kwargs.pop( CONF_CAR_BATTERY_CAPACITY, None)

        self.car_charger_min_charge : int = int(max(0,kwargs.pop(CONF_CAR_CHARGER_MIN_CHARGE, 6)))
        self.car_charger_max_charge : int = int(max(0,kwargs.pop(CONF_CAR_CHARGER_MAX_CHARGE,32)))
        self.car_use_custom_power_charge_values = kwargs.pop(CONF_CAR_CUSTOM_POWER_CHARGE_VALUES, False)
        self.car_is_custom_power_charge_values_3p = kwargs.pop(CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P, False)
        self.amp_to_power_1p = [-1] * (self.car_charger_max_charge + 1 - self.car_charger_min_charge)
        self.amp_to_power_3p = [-1] * (self.car_charger_max_charge + 1 - self.car_charger_min_charge)
        self._last_dampening_update = None

        super().__init__(**kwargs)


        for a in range(self.car_charger_min_charge, self.car_charger_max_charge + 1):

            val_1p = float(self.home.voltage * a)
            val_3p = 3*val_1p

            self.amp_to_power_1p[a - self.car_charger_min_charge] = val_1p
            self.amp_to_power_3p[a - self.car_charger_min_charge] = val_3p


        if self.car_use_custom_power_charge_values:

            for a in range(self.car_charger_min_charge, self.car_charger_max_charge + 1):
                val_1p = val_3p = float(kwargs.pop(f"charge_{a}", -1))
                if val_1p >= 0:
                    if self.car_is_custom_power_charge_values_3p:
                        val_1p = val_3p / 3.0
                    else:
                        val_3p = val_1p * 3.0
                    self.amp_to_power_1p[a - self.car_charger_min_charge] = val_1p
                    self.amp_to_power_3p[a - self.car_charger_min_charge] = val_3p


        self.attach_ha_state_to_probe(self.car_charge_percent_sensor, is_numerical=True)

    async def _save_dampening_values(self):

        data = dict(self.config_entry.data)

        data[CONF_CAR_CHARGER_MIN_CHARGE] = self.car_charger_min_charge
        data[CONF_CAR_CHARGER_MAX_CHARGE] = self.car_charger_max_charge
        data[CONF_CAR_CUSTOM_POWER_CHARGE_VALUES] = self.car_use_custom_power_charge_values
        data[CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P] = self.car_is_custom_power_charge_values_3p

        for a in range(0, 33):
            data[f"charge_{a}"] = -1

        for a in range(self.car_charger_min_charge, self.car_charger_max_charge + 1):
            if self.car_is_custom_power_charge_values_3p:
                val = self.amp_to_power_3p[a - self.car_charger_min_charge]
            else:
                val = self.amp_to_power_1p[a - self.car_charger_min_charge]

            data[f"charge_{a}"] = val

        self.hass.config_entries.async_update_entry(self.config_entry, data=data)


    def update_dampening_value(self, amperage:int|float, power_value:int|float, for_3p:bool, time:datetime):

        if amperage < self.car_charger_min_charge or amperage > self.car_charger_max_charge:
            return



        if power_value < MIN_CHARGE_POWER_W:
            # we may have a 0 value for a given amperage actually it could change the min and max amperage
            power_value = 0

        if for_3p:
            old_val = self.amp_to_power_3p[amperage - self.car_charger_min_charge]
        else:
            old_val = self.amp_to_power_1p[amperage - self.car_charger_min_charge]

        do_update = False
        if (power_value == 0 or old_val == 0):
            if power_value == old_val:
                do_update = False
            elif max(power_value, old_val) < MIN_CHARGE_POWER_W:
                if power_value == 0:
                    do_update = True
            else:
                do_update = True

        elif abs(old_val - power_value) > 0.1*max(old_val,power_value):
            do_update = True

        if do_update:

            self.car_is_custom_power_charge_values_3p = for_3p

            if for_3p:
                self.amp_to_power_3p[amperage - self.car_charger_min_charge] = power_value
                self.amp_to_power_1p[amperage - self.car_charger_min_charge] = power_value / 3.0
            else:
                self.amp_to_power_1p[amperage - self.car_charger_min_charge] = power_value
                self.amp_to_power_3p[amperage - self.car_charger_min_charge] = power_value * 3.0

            if power_value == 0:
                d = 0
                for i, val in enumerate(self.amp_to_power_3p):
                    if val == 0:
                        d += 1
                    else:
                        break

                self.car_charger_min_charge += d
                self.amp_to_power_1p = self.amp_to_power_1p[d:]
                self.amp_to_power_3p = self.amp_to_power_3p[d:]


            if self._last_dampening_update is None or (time - self._last_dampening_update).total_seconds() > 300:
                self._last_dampening_update = time
                self.hass.add_job(self._save_dampening_values)

    def get_charge_power_per_phase_A(self, for_3p:bool) -> tuple[list[float], int, int]:
        if for_3p:
            return self.amp_to_power_3p, self.car_charger_min_charge, self.car_charger_max_charge
        else:
            return self.amp_to_power_1p, self.car_charger_min_charge, self.car_charger_max_charge


    def get_platforms(self):
        return [ Platform.SENSOR, Platform.SELECT ]

