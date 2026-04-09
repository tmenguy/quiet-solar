import logging

from ..const import (
    CONF_BATTERY_CAPACITY,
    CONF_BATTERY_IS_DC_COUPLED,
    CONF_BATTERY_MAX_CHARGE_PERCENT,
    CONF_BATTERY_MAX_CHARGE_POWER_VALUE,
    CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE,
    CONF_BATTERY_MIN_CHARGE_PERCENT,
    MAX_POWER_INFINITE,
)
from ..home_model.load import AbstractDevice

_LOGGER = logging.getLogger(__name__)


class Battery(AbstractDevice):
    def __init__(self, **kwargs):

        self.capacity = kwargs.pop(CONF_BATTERY_CAPACITY, 7000)
        self.max_discharging_power = kwargs.pop(CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE, 1500)
        self.max_charging_power = kwargs.pop(CONF_BATTERY_MAX_CHARGE_POWER_VALUE, 1500)
        self.min_charge_SOC_percent = kwargs.pop(CONF_BATTERY_MIN_CHARGE_PERCENT, 0.0)
        self.max_charge_SOC_percent = kwargs.pop(CONF_BATTERY_MAX_CHARGE_PERCENT, 100.0)
        self.is_dc_coupled = kwargs.pop(CONF_BATTERY_IS_DC_COUPLED, False)

        super().__init__(**kwargs)

        self._current_charge_value = None
        self.max_soc = self.max_charge_SOC_percent / 100.0  # %percentage of battery capacity between 0 and 1
        self.min_soc = self.min_charge_SOC_percent / 100.0
        self.min_charging_power = 0.0
        self.min_discharging_power = 0.0

    @property
    def current_charge(self) -> float | None:
        return self._current_charge_value

    @property
    def charge_from_grid(self) -> bool:
        return False


    def get_charger_power(
        self,
        available_power: float,
        clamped_over_dc_power: float,
        max_inverter_dc_to_ac_power: float | None,
        duration_s: float,
        current_charge: float | None = None,
    ):
        # available_power = ua + p - pv + cp
        # cp : the clamped dc power
        # ua : the non-controllable load consumption
        # p  : load power
        # pv : solar panel production

        if current_charge is None:
            current_charge = self.current_charge if self.current_charge is not None else 0.0

        inverter_ac_limit = float(max_inverter_dc_to_ac_power) if max_inverter_dc_to_ac_power is not None else float("inf")

        battery_ac_out = 0.0
        battery_ac_in = 0.0

        if available_power > 0.0:
            battery_ac_out = min(self.max_discharging_power, min(available_power, inverter_ac_limit))
        else:
            battery_ac_in = min(self.max_charging_power, min(0.0 - available_power, inverter_ac_limit))

        battery_dc_in = min(self.max_charging_power, clamped_over_dc_power + battery_ac_in)
        battery_dc_out = battery_ac_out

        charging_power = battery_dc_in - battery_dc_out

        possible_charge = min(self.max_charging_power, max(0.0, (self.get_value_full() - current_charge) * 3600 / duration_s))
        possible_discharge = min(self.max_discharging_power, max(0.0, (current_charge - self.get_value_empty()) * 3600 / duration_s))

        battery_ac_in = min(battery_ac_in, max(0.0, possible_charge - clamped_over_dc_power))
        battery_ac_out = min(battery_ac_out, possible_discharge)

        if charging_power > 0:
            charging_power = min(charging_power, possible_charge)
        else:
            charging_power = max(charging_power, -possible_discharge)

        return charging_power, battery_ac_in - battery_ac_out, possible_discharge


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

    def get_available_energy(self):
        current_charge = self.current_charge

        if current_charge is None:
            return 0.0

        return max(0.0, (current_charge - (self.min_soc * self.capacity)))

    def battery_get_current_possible_max_discharge_power(self) -> float:

        current_charge = self.current_charge

        # unknown ... return max discharge by default
        if current_charge is None:
            return self.max_discharging_power

        if self.is_value_empty(current_charge):
            return 0.0

        max_discharge = self.get_max_discharging_power()

        if max_discharge is None:
            return self.max_discharging_power

        if max_discharge == 0.0:
            return 0.0

        return max_discharge

    def get_max_discharging_power(self) -> float | None:
        if self.max_discharging_power == MAX_POWER_INFINITE:
            return None
        return self.max_discharging_power
