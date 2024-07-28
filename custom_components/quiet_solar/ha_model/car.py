import copy

from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.helpers.entity import Entity

from quiet_solar.ha_model.device import HADeviceMixin
from quiet_solar.home_model.load import AbstractDevice
from homeassistant.const import Platform

from quiet_solar.sensor import QSSensorEntityDescription, QSBaseSensorRestore


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

        super().__init__(**kwargs)

        self.amp_to_power_1p = [-1] * (self.car_charger_max_charge + 1 - self.car_charger_min_charge)
        self.amp_to_power_3p = [-1] * (self.car_charger_max_charge + 1 - self.car_charger_min_charge)

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


    def get_charge_power_per_phase_A(self, for_3p:bool) -> tuple[list[float], int, int]:
        if for_3p:
            return self.amp_to_power_3p, self.car_charger_min_charge, self.car_charger_max_charge
        else:
            return self.amp_to_power_1p, self.car_charger_min_charge, self.car_charger_max_charge


    def get_platforms(self):
        return [ Platform.SENSOR, Platform.SELECT ]

    def create_ha_entities(self, platform: str) -> list[Entity]:

        entities = []


        store_sensor = QSSensorEntityDescription(
            key="charge_state",
            device_class=SensorDeviceClass.ENUM,
            options=[
                "not_in_charge",
                "waiting_for_a_planned_charge",
                "charge_ended",
                "waiting_for_current_charge",
                "energy_flap_opened",
                "charge_in_progress",
                "charge_error"
            ],
        )

        entities.append(QSBaseSensorRestore(data_handler=self.data_handler, qs_device=self, description=store_sensor))

        return entities
