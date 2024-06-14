from home_model.load import AbstractLoad
from home_model.commands import LoadCommand, CMD_AUTO_ECO

CMD_FORCE_CHARGE = LoadCommand(command="charge", power_consign=0.0, param="only")
CMD_FORCE_DISCHARGE = LoadCommand(command="discharge", power_consign=0.0, param="only")

class Battery(AbstractLoad):

    def __init__(self, name:str, capacity_kWh :int, **kwargs):

        super().__init__(name, **kwargs)
        self._capacity = capacity_kWh #in kWh
        self.default_cmd : LoadCommand = CMD_AUTO_ECO


        self._state = 0.0
        self._constraints = []
        self.current_charge = 0.0
        self._discharge = 0.0

        self.max_soc = 1.0 # %percentage of battery capacity between 0 and 1
        self.min_soc = 0.0
        self.min_charging_power = 0.0
        self.max_charging_power = 0.0
        self.min_discharging_power = 0.0
        self.max_discharging_power = 0.0

    def get_best_charge_power(self, power_in: float, duration_s: float, current_charge: float | None = None):

        if power_in < self.min_charging_power:
            return 0.0

        if current_charge is None:
            current_charge = self.current_charge

        charging_power = min(power_in, self.max_charging_power)

        available_power = max(0.0, ((self.max_soc*self._capacity) - current_charge)*3600 / duration_s)

        charging_power = min(charging_power, available_power)

        return charging_power

    def get_best_discharge_power(self, power_in: float, duration_s: float, current_charge: float | None = None):

        if power_in < self.min_discharging_power:
            return 0.0

        if current_charge is None:
            current_charge = self.current_charge

        charging_power = min(power_in, self.max_discharging_power)

        available_power = max(0.0, (current_charge - (self.min_soc * self._capacity)) * 3600 / duration_s)

        charging_power = min(charging_power, available_power)

        return charging_power


