from ..home_model.load import AbstractLoad
from .commands import LoadCommand
from ..const import CONF_BATTERY_CAPACITY, CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE, CONF_BATTERY_MAX_CHARGE_POWER_VALUE, \
    CONF_BATTERY_MIN_CHARGE_PERCENT, CONF_BATTERY_MAX_CHARGE_PERCENT

CMD_FORCE_CHARGE = LoadCommand(command="charge", power_consign=0.0)
CMD_FORCE_DISCHARGE = LoadCommand(command="discharge", power_consign=0.0)

class Battery(AbstractLoad):

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


    def battery_can_discharge(self):

        current_charge = self.current_charge

        if current_charge is None:
            return False

        if current_charge > max(self.min_soc * self.capacity, 500): # 500Wh is a threshold
            return True

        return False


    def get_best_charge_power(self, power_in: float, duration_s: float, current_charge: float | None = None):

        if power_in < self.min_charging_power:
            return 0.0

        if current_charge is None:
            current_charge = self.current_charge

        if current_charge is None:
            current_charge = 0.0

        charging_power = min(power_in, self.max_charging_power)

        available_power = max(0.0, ((self.max_soc * self.capacity) - current_charge) * 3600 / duration_s)

        charging_power = min(charging_power, available_power)

        return charging_power

    def get_best_discharge_power(self, power_in: float, duration_s: float, current_charge: float | None = None):

        if power_in < self.min_discharging_power:
            return 0.0

        if current_charge is None:
            current_charge = self.current_charge

        if current_charge is None:
            current_charge = 0.0

        charging_power = min(power_in, self.max_discharging_power)

        available_power = max(0.0, (current_charge - (self.min_soc * self.capacity)) * 3600 / duration_s)

        charging_power = min(charging_power, available_power)

        return charging_power


