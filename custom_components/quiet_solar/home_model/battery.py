import logging

from ..home_model.load import AbstractDevice
from ..const import CONF_BATTERY_CAPACITY, CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE, CONF_BATTERY_MAX_CHARGE_POWER_VALUE, \
    CONF_BATTERY_MIN_CHARGE_PERCENT, CONF_BATTERY_MAX_CHARGE_PERCENT, MAX_POWER_INFINITE, CONF_BATTERY_IS_DC_COUPLED

_LOGGER = logging.getLogger(__name__)

class Battery(AbstractDevice):

    def __init__(self, **kwargs):

        self.capacity = kwargs.pop(CONF_BATTERY_CAPACITY, 7000)
        self.max_discharging_power  = kwargs.pop(CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE, 1500)
        self.max_charging_power =  kwargs.pop(CONF_BATTERY_MAX_CHARGE_POWER_VALUE, 1500)
        self.min_charge_SOC_percent = kwargs.pop(CONF_BATTERY_MIN_CHARGE_PERCENT, 0.0)
        self.max_charge_SOC_percent = kwargs.pop(CONF_BATTERY_MAX_CHARGE_PERCENT, 100.0)


        super().__init__(**kwargs)

        self._current_charge_value = None
        self.max_soc = self.max_charge_SOC_percent / 100.0 # %percentage of battery capacity between 0 and 1
        self.min_soc = self.min_charge_SOC_percent / 100.0
        self.min_charging_power = 0.0
        self.min_discharging_power = 0.0

    @property
    def current_charge(self) -> float | None:
        return self._current_charge_value

    @property
    def charge_from_grid(self) -> bool:
        return False


    @property
    def max_inverter_dc_to_ac_power(self) -> float:
        return MAX_POWER_INFINITE

    def get_best_charge_power(self, power_in: float, solar_production: float, duration_s: float, current_charge: float | None = None):

        if power_in < self.min_charging_power:
            return 0.0

        if current_charge is None:
            current_charge = self.current_charge

        if current_charge is None:
            current_charge = 0.0

        charging_power = min(power_in, self.max_charging_power)

        if self.charge_from_grid is False:
            if solar_production < charging_power:
                _LOGGER.warning(
                    f"get_best_charge_power: clamping charging_power:{charging_power} > solar_production:{solar_production}")

            charging_power = min(solar_production, charging_power)

        available_power = max(0.0, ((self.get_value_full()) - current_charge) * 3600 / duration_s)

        charging_power = min(charging_power, available_power)

        return charging_power

    def is_value_full(self, energy_value_wh: float | None) -> bool:
        if energy_value_wh is None:
            return False
        return energy_value_wh >= self.get_value_full()

    def get_value_full(self):
        return self.max_soc * self.capacity

    def is_value_empty(self, energy_value_wh: float | None) -> bool:
        if energy_value_wh is None:
            return True
        return energy_value_wh <= self.get_value_empty()

    def get_value_empty(self):
        return self.min_soc * self.capacity


    def get_best_discharge_power(self, power_out: float, solar_production: float, duration_s: float, current_charge: float | None = None):

        if power_out < self.min_discharging_power:
            return 0.0

        if current_charge is None:
            current_charge = self.current_charge

        if current_charge is None:
            current_charge = 0.0

        discharging_power = power_out

        # if there is too much solar production whatever we do we can't discharge more than the inverter capacity
        if solar_production > 0.0:
            if power_out + solar_production > self.max_inverter_dc_to_ac_power:
                discharging_power = max(0.0, self.max_inverter_dc_to_ac_power - solar_production)
                _LOGGER.warning(f"get_best_discharge_power: clamping power_out:{power_out} + solar_production:{solar_production} to max_inverter_dc_to_ac_power:{self.max_inverter_dc_to_ac_power}, so discharging_power={discharging_power}")


        discharging_power = min(discharging_power, self.max_discharging_power)

        available_power = max(0.0, (current_charge - (self.min_soc * self.capacity)) * 3600 / duration_s)

        discharging_power = min(discharging_power, available_power)

        return discharging_power


