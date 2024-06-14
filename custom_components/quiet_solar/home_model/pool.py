from home_model.load import AbstractLoad

class Pool(AbstractLoad):

    def __init__(self, name, power, power_factor, capacity, max_power, efficiency):

        super().__init__(name, power, power_factor)
        self._capacity = capacity
        self._max_power = max_power
        self._efficiency = efficiency
        self._state = 0.0
        self._constraints = []
        self._charge = 0.0
        self._discharge = 0.0
        self._charge_power = 0.0
        self._discharge_power = 0.0
