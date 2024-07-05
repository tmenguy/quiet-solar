from .load import AbstractLoad
from .commands import LoadCommand, CMD_AUTO_ECO
from ..const import CONF_BATTERY_CAPACITY, CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE, CONF_BATTERY_MAX_CHARGE_POWER_VALUE

CMD_FORCE_CHARGE = LoadCommand(command="charge", power_consign=0.0, param="only")
CMD_FORCE_DISCHARGE = LoadCommand(command="discharge", power_consign=0.0, param="only")

class Battery(AbstractLoad):

    def __init__(self, battery_capacity :float, **kwargs):

        self.capacity = kwargs.pop(CONF_BATTERY_CAPACITY, 230)

        self.max_discharging_power  = kwargs.pop(CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE, 0)
        self.max_charging_power =  kwargs.pop(CONF_BATTERY_MAX_CHARGE_POWER_VALUE, 0)

        super().__init__(**kwargs)

        self.default_cmd : LoadCommand = CMD_AUTO_ECO


        self._state = 0.0
        self._constraints = []
        self.current_charge = 0.0
        self._discharge = 0.0

        self.max_soc = 1.0 # %percentage of battery capacity between 0 and 1
        self.min_soc = 0.0
        self.min_charging_power = 0.0
        self.min_discharging_power = 0.0


    def is_battery_in_auto_mode(self):
        return self.current_command is None or self.current_command.command == CMD_AUTO_ECO.command

    def get_best_charge_power(self, power_in: float, duration_s: float, current_charge: float | None = None):

        if power_in < self.min_charging_power:
            return 0.0

        if current_charge is None:
            current_charge = self.current_charge

        charging_power = min(power_in, self.max_charging_power)

        available_power = max(0.0, ((self.max_soc * self.capacity) - current_charge) * 3600 / duration_s)

        charging_power = min(charging_power, available_power)

        return charging_power

    def get_best_discharge_power(self, power_in: float, duration_s: float, current_charge: float | None = None):

        if power_in < self.min_discharging_power:
            return 0.0

        if current_charge is None:
            current_charge = self.current_charge

        charging_power = min(power_in, self.max_discharging_power)

        available_power = max(0.0, (current_charge - (self.min_soc * self.capacity)) * 3600 / duration_s)

        charging_power = min(charging_power, available_power)

        return charging_power


