import copy
from datetime import datetime
from datetime import timedelta
from abc import abstractmethod
from typing import Callable, Self, Awaitable

from home_model.commands import LoadCommand, CMD_ON, CMD_AUTO_GREEN_ONLY, CMD_AUTO_ECO
import numpy.typing as npt
import numpy as np
from bisect import bisect_left



class LoadConstraint(object):

    def __init__(self,
                 load,
                 mandatory: bool = True,
                 start_of_constraint: datetime | None = None,
                 end_of_constraint: datetime | None = None,
                 initial_value: float = 0.0,
                 target_value: float = 0.0,
                 update_value_callback: Callable[[Self, datetime], Awaitable[float]]| None = None,
                 update_command_callback: Callable[[Self, datetime], Awaitable[LoadCommand|None]] | None = None
                 ):

        """
        :param load: the load that the constraint is applied to
        :param mandatory: is it mandatory to meet the constraint
        :param start_of_constraint: constraint start time if None the constraint start asap
        :param end_of_constraint: constraint end time if None the constraint is always active
        :param initial_value: the constraint start value
        :param target_value: the constraint target value if None it means that the constraint is always active
        """

        self.load = load
        self.is_mandatory = mandatory

        if end_of_constraint is None:
            end_of_constraint = datetime.max

        if start_of_constraint is None:
            start_of_constraint = datetime.min

        self.end_of_constraint: datetime = end_of_constraint

        self.user_start_of_constraint: datetime = start_of_constraint
        self.start_of_constraint: datetime = start_of_constraint

        self.initial_value = initial_value
        self.target_value = target_value
        self.current_value = initial_value

        self._update_value_callback = update_value_callback
        self._update_command_callback = update_command_callback


        nt = datetime.now()
        self.last_value_update = nt
        self.last_state_update = nt

        self.current_command : LoadCommand | None = None

        self.skip = False
        self.pushed_count = 0
        self.start_of_constraint: datetime = datetime.min


    def is_constraint_active_for_time_period(self, start_time: datetime, end_time: datetime) -> bool:

        if self.is_constraint_met():
            return False

        if self.end_of_constraint == datetime.max:
            if self.start_of_constraint > end_time:
                return False
            else:
                return True

        #only active if the constraint finish before the end of the given time period
        return  self.end_of_constraint >= start_time and self.end_of_constraint <= end_time


    def is_constraint_met(self) -> bool:
        """ is the constraint met in its current form? """

        if self.target_value is None:
            return False

        if self.target_value > self.initial_value and self.current_value >= self.target_value:
            return True

        if self.target_value < self.initial_value and self.current_value <= self.target_value:
            return True

        return False

    def _update_current_value(self, time: datetime, value: float):
        """ Update the constraint with the new value. to be called by a load that can compute the value based or sensors or external data"""
        self.current_value = value
        self.last_value_update = time


    async def _compute_value(self, time: datetime):
        """ Compute the value of the constraint at the given time. to be implemented by the load"""
        if time >= self.last_value_update:
            if self._update_value_callback is not None:
                value = await self._update_value_callback(self, time)
            else:
                value = self.compute_value(time)
            self._update_current_value(time, value)


    async def update(self, time: datetime):
        """ Update the constraint with the new value. to be called by a load that can compute the value based or sensors or external data"""

        #check first if there is not an external change of state
        if self._update_command_callback is not None:
            new_cmd = await self._update_command_callback(self, time)
            if new_cmd is not None:
                await self.update_current_command(time, new_cmd)
                return

        await self._compute_value(time)


    async def update_current_command(self, time: datetime, cmd: LoadCommand | None = None):
        """ Update the current Command of the constraint."""
        # update the value if needed
        await self._compute_value(time)

        if cmd is not None and (self.current_command is None or self.current_command != cmd):
            self.current_command = cmd
            self.last_state_update = time

    @abstractmethod
    def compute_value(self, time: datetime) -> float:
        """ Compute the value of the constraint when the state change."""

    @abstractmethod
    def best_duration_to_meet(self) -> timedelta:
        """ Return the best duration to meet the constraint."""

    @abstractmethod
    def compute_best_period_repartition(self,
                                        power_available_power : npt.NDArray[np.float64],
                                        power_slots_duration_s : npt.NDArray[np.float64],
                                        prices: npt.NDArray[np.float64],
                                        prices_ordered_values: list[float],
                                        time_slots: list[datetime]) -> tuple[bool, list[LoadCommand|None], npt.NDArray[np.float64]]:
        """ Compute the best repartition of the constraint over the given period."""


class MultiStepsPowerLoadConstraint(LoadConstraint):

    def __init__(self,
                 power_steps: dict[str, float] = None,
                 support_auto: bool = False,
                 **kwargs):

        super().__init__(**kwargs)
        self._power_steps = power_steps
        self._power_vals = [c for c in power_steps.items() if c[1] > 0.0]
        self._power_vals = sorted(self._power_vals, key=lambda x: x[1])
        self._max_power = self._power_vals[-1][1]
        self._min_power = self._power_vals[0][1]
        self._support_auto = support_auto


    def best_duration_to_meet(self) -> timedelta:
        """ Return the best duration to meet the constraint."""
        seconds = (3600.0 * (self.target_value - self.initial_value)) / self._max_power
        return timedelta(seconds=seconds)

    def compute_value(self, time: datetime) -> float:
        """ Compute the value of the constraint whenever it is called changed state or not,
        hence use the old state and the last value change to add the consummed energy """
        if self.current_command is None:
            return self.current_value

        return (((time - self.last_value_update).total_seconds()) * self.current_command.value / 3600.0) + self.current_value


    def compute_best_period_repartition(self,
                                        power_available_power : npt.NDArray[np.float64],
                                        power_slots_duration_s : npt.NDArray[np.float64],
                                        prices: npt.NDArray[np.float64],
                                        prices_ordered_values: list[float],
                                        time_slots: list[datetime]) -> tuple[bool, list[LoadCommand|None], npt.NDArray[np.float64]]:
        nrj_to_be_added = self.target_value - self.current_value
        return self._compute_best_period_repartition(power_available_power, power_slots_duration_s, prices, prices_ordered_values, time_slots, nrj_to_be_added)

    def _compute_best_period_repartition(self,
                                        power_available_power : npt.NDArray[np.float64],
                                        power_slots_duration_s : npt.NDArray[np.float64],
                                        prices: npt.NDArray[np.float64],
                                        prices_ordered_values: list[float],
                                        time_slots: list[datetime],
                                        nrj_to_be_added: float) -> tuple[bool, list[LoadCommand|None], npt.NDArray[np.float64]]:

        out_power = np.zeros(len(power_available_power), dtype=np.float64)
        out_power_idxs = np.zeros(len(power_available_power), dtype=np.int64)
        out_commands : list[LoadCommand|None] = [None] * len(power_available_power)

        #first get to the available power slots (ie with negative power available, fill it at best in a greedy way
        min_power = self._min_power

        first_slot = 0
        last_slot = len(power_available_power) - 1

        #find the constraint last slot
        if self.end_of_constraint != datetime.max:
            #by construction the end of the constraints IS ending on a slot
            last_slot = bisect_left(time_slots, self.end_of_constraint)
            last_slot = max(0, last_slot - 1) # -1 as the last_slot index is the index of the slot not the time anchor

        if self.start_of_constraint != datetime.min:
            first_slot = bisect_left(time_slots, self.start_of_constraint)

        for i in range(first_slot, last_slot + 1):

            if power_available_power[i] <= -min_power:
                j = 0
                while j < len(self._power_vals) - 1 and self._power_vals[j+1][1] < -power_available_power[i]:
                    j += 1

                if self._support_auto:
                    new_cmd = copy.copy(CMD_AUTO_GREEN_ONLY)
                    new_cmd.value = self._power_vals[j][1]
                    out_commands[i] = new_cmd
                else:
                    out_commands[i] = LoadCommand(command=self._power_vals[j][0], value=self._power_vals[j][1], param=self._power_vals[j][0])

                out_power[i] = self._power_vals[j][1]
                out_power_idxs[i] = j

                nrj_to_be_added -= (out_power[i] * power_slots_duration_s[i]) / 3600.0

                if nrj_to_be_added <= 0.0:
                    break

        if nrj_to_be_added <= 0.0 or self.is_mandatory is False:
            return nrj_to_be_added <= 0.0, out_commands, out_power

        #pure solar was not enough, we will try to see if we can get a bit more solar energy directly if price is better
        costs_optimizers = []

        for i in range(first_slot, last_slot + 1):

            if power_available_power[i] + out_power[i] <= 0.0:
                #there is still some solar to get
                power_to_add_idx  = -1
                if out_commands[i] is None:
                    power_to_add_idx = 0
                elif out_power_idxs[i] < len(self._power_vals) - 1:
                    power_to_add_idx = out_power_idxs[i] + 1

                if power_to_add_idx >= 0:
                    power_to_add : float = self._power_vals[power_to_add_idx][1]
                    energy_to_add : float = (power_to_add * power_slots_duration_s[i]) / 3600.0
                    cost: float  = ((power_to_add + power_available_power[i]) * power_slots_duration_s[i] * prices[i])/3600.0
                    if self._support_auto:
                        new_cmd = copy.copy(CMD_AUTO_ECO)
                        new_cmd.value = self._power_vals[power_to_add_idx][1]
                    else:
                        new_cmd = LoadCommand(command=self._power_vals[power_to_add_idx][0], value=self._power_vals[power_to_add_idx][1], param=self._power_vals[power_to_add_idx][0])

                    costs_optimizers.append((cost, energy_to_add, i, power_to_add, new_cmd))

        costs_optimizers: list[tuple[float, float, int, float, LoadCommand]] = sorted(costs_optimizers, key=lambda x: x[0])


        for price in prices_ordered_values:

            price_span_h = (np.sum(power_slots_duration_s[first_slot:last_slot+1], where = prices[first_slot:last_slot+1] == price))/3600.0

            #to try to fill as smoothly as possible: is it possible to fill the slot with the maximum power value?
            fill_power_idx = 0
            for fill_power_idx in range(len(self._power_vals)):
                if (nrj_to_be_added / self._power_vals[fill_power_idx][1]) < price_span_h:
                    break

            price_cmd = LoadCommand(command=self._power_vals[fill_power_idx][0], value=self._power_vals[fill_power_idx][1], param=self._power_vals[fill_power_idx][0])
            price_power = self._power_vals[fill_power_idx][1]

            for i in range(first_slot, last_slot + 1):

                if prices[i] == price and out_commands[i] is None:

                    current_energy : float = (price_power * power_slots_duration_s[i])/3600.0
                    current_cost = current_energy * price

                    c_cmd = price_cmd
                    c_power = price_power
                    slot_idx = i

                    found_opti_idx = -1
                    for j in range(len(costs_optimizers)):
                        if costs_optimizers[j][0] >= current_cost:
                            break
                        if costs_optimizers[j][1] >= current_energy:
                            found_opti_idx = j
                            break

                    if found_opti_idx >= 0:
                        _, _, slot_idx, c_power, c_cmd = costs_optimizers.pop(found_opti_idx)


                    out_commands[slot_idx] = c_cmd
                    out_power[slot_idx] = c_power
                    delta_energy = (c_power * power_slots_duration_s[slot_idx]) / 3600.0

                    nrj_to_be_added -= delta_energy

                    if nrj_to_be_added <= 0.0:
                        break

            if nrj_to_be_added <= 0.0:
                break

        return nrj_to_be_added <= 0.0, out_commands, out_power


class SimplePowerLoadConstraint(MultiStepsPowerLoadConstraint):


    def __init__(self,
                 power: float = 0.0,
                 **kwargs
                 ):
        power_steps = {CMD_ON.command: power}
        self._power = power
        super().__init__(power_steps=power_steps, support_auto=False, **kwargs)


class TimeBasedSimplePowerLoadConstraint(SimplePowerLoadConstraint):

    def __init__(self,
                 power: float = 0.0,
                 **kwargs
                 ):
        super().__init__(power=power, **kwargs)


    def best_duration_to_meet(self) -> timedelta:
        """ Return the best duration to meet the constraint."""
        return timedelta(seconds=self.target_value - self.initial_value)

    def compute_value_on_state_change(self, from_state: str, to_state: str, from_time: datetime, to_time: datetime):
        """ Compute the value of the constraint whenever it is called changed state or not,
        hence use the old state and the last value change to add teh consummed energy """

        if from_state == CMD_ON:
            return (to_time - self.last_value_update).total_seconds() + self.current_value
        else:
            return self.current_value


    def compute_best_period_repartition(self,
                                        power_available_power : npt.NDArray[np.float64],
                                        power_slots_duration_s : npt.NDArray[np.float64],
                                        prices: npt.NDArray[np.float64],
                                        prices_ordered_values: list[float],
                                        time_slots: list[datetime])  -> tuple[bool, list[LoadCommand|None], npt.NDArray[np.float64]]:

        nrj_to_be_added = (self._power * (self.target_value - self.current_value) / 3600.0)
        return self._compute_best_period_repartition(power_available_power, power_slots_duration_s, prices, prices_ordered_values, time_slots, nrj_to_be_added)







