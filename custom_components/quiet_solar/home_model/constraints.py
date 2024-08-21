from datetime import datetime
from datetime import timedelta
from abc import abstractmethod
from typing import Callable, Self, Awaitable
import pytz

from .commands import LoadCommand, CMD_ON, CMD_AUTO_GREEN_ONLY, CMD_AUTO_FROM_CONSIGN, copy_command
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
                 ):

        """
        :param load: the load that the constraint is applied to
        :param mandatory: is it mandatory to meet the constraint
        :param start_of_constraint: constraint start time if None the constraint start asap
        :param end_of_constraint: constraint end time if None the constraint is always active
        :param initial_value: the constraint start value
        :param target_value: the constraint target value if None it means that the constraint is always active
        :param update_value_callback: a callback to compute the value of the constraint, typically passed by the load
        """

        self.load = load
        self.is_mandatory = mandatory


        self.name = f"Constraint for {load.name} ({initial_value}/{target_value}/{mandatory})"

        if end_of_constraint is None:
            end_of_constraint = DATETIME_MAX_UTC

        if start_of_constraint is None:
            start_of_constraint = DATETIME_MIN_UTC

        self.end_of_constraint: datetime = end_of_constraint

        self.user_start_of_constraint: datetime = start_of_constraint
        self.start_of_constraint: datetime = start_of_constraint

        self.initial_value = initial_value
        self.target_value = target_value
        self.current_value = initial_value

        self._update_value_callback = update_value_callback


        nt = datetime.now(pytz.UTC)
        self.last_value_update = nt
        self.last_state_update = nt

        self.skip = False
        self.pushed_count = 0
        self.start_of_constraint: datetime = DATETIME_MIN_UTC


    def convert_target_value_to_energy(self, value: float) -> float:
        return value

    def convert_energy_to_target_value(self, energy: float) -> float:
        return energy

    def is_constraint_active_for_time_period(self, start_time: datetime, end_time: datetime) -> bool:

        if self.is_constraint_met():
            return False

        if self.end_of_constraint == DATETIME_MAX_UTC:
            if self.start_of_constraint > end_time:
                return False
            else:
                return True

        #only active if the constraint finish before the end of the given time period
        return  self.end_of_constraint >= start_time and self.end_of_constraint <= end_time


    def is_constraint_met(self, current_value=None) -> bool:
        """ is the constraint met in its current form? """
        if current_value is None:
            current_value = self.current_value

        if self.target_value is None:
            return False

        if self.target_value > self.initial_value and current_value >= self.target_value:
            return True

        if self.target_value < self.initial_value and current_value <= self.target_value:
            return True

        return False

    async def update(self, time: datetime):
        """ Update the constraint with the new value. to be called by a load that can compute the value based or sensors or external data"""
        if time >= self.last_value_update:
            if self._update_value_callback is not None:
                value = await self._update_value_callback(self, time)
                if value is None:
                    value = self.current_value
            else:
                value = self.compute_value(time)
            self.current_value = value
            self.last_value_update = time

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
                 power_steps: list[LoadCommand] = None,
                 support_auto: bool = False,
                 **kwargs):

        super().__init__(**kwargs)
        self._power_cmds = power_steps
        self._power_sorted_cmds = [c for c in power_steps if c.power_consign > 0.0]
        self._power_sorted_cmds = sorted(self._power_sorted_cmds, key=lambda x: x.power_consign)
        self._max_power = self._power_sorted_cmds[-1].power_consign
        self._min_power = self._power_sorted_cmds[0].power_consign
        self._support_auto = support_auto


    def best_duration_to_meet(self) -> timedelta:
        """ Return the best duration to meet the constraint."""
        seconds = (3600.0 * (self.target_value - self.initial_value)) / self._max_power
        return timedelta(seconds=seconds)

    def compute_value(self, time: datetime) -> float:
        """ Compute the value of the constraint whenever it is called changed state or not """
        if self.load.current_command is None:
            return self.current_value

        return (((time - self.last_value_update).total_seconds()) * self.load.current_command.power_consign / 3600.0) + self.current_value


    def compute_best_period_repartition(self,
                                        power_available_power : npt.NDArray[np.float64],
                                        power_slots_duration_s : npt.NDArray[np.float64],
                                        prices: npt.NDArray[np.float64],
                                        prices_ordered_values: list[float],
                                        time_slots: list[datetime]) -> tuple[bool, list[LoadCommand|None], npt.NDArray[np.float64]]:
        nrj_to_be_added = self.convert_target_value_to_energy(self.target_value) - self.convert_target_value_to_energy(self.current_value)
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
        if self.end_of_constraint is not None:
            #by construction the end of the constraints IS ending on a slot
            last_slot = bisect_left(time_slots, self.end_of_constraint)
            last_slot = max(0, last_slot - 1) # -1 as the last_slot index is the index of the slot not the time anchor

        if self.start_of_constraint != DATETIME_MIN_UTC:
            first_slot = bisect_left(time_slots, self.start_of_constraint)

        for i in range(first_slot, last_slot + 1):

            if power_available_power[i] <= -min_power:
                j = 0
                while j < len(self._power_sorted_cmds) - 1 and self._power_sorted_cmds[j + 1].power_consign < -power_available_power[i]:
                    j += 1

                if self._support_auto:
                    out_commands[i] = copy_command(CMD_AUTO_GREEN_ONLY, power_consign=self._power_sorted_cmds[j].power_consign)
                else:
                    out_commands[i] = copy_command(self._power_sorted_cmds[j])

                out_power[i] = out_commands[i].power_consign
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
                elif out_power_idxs[i] < len(self._power_sorted_cmds) - 1:
                    power_to_add_idx = out_power_idxs[i] + 1

                if power_to_add_idx >= 0:
                    # all command in ECO mode now....
                    power_to_add : float = self._power_sorted_cmds[power_to_add_idx].power_consign
                    energy_to_add : float = (power_to_add * float(power_slots_duration_s[i])) / 3600.0
                    cost: float  = ((power_to_add + float(power_available_power[i])) * float(power_slots_duration_s[i]) * float(prices[i]))/3600.0
                    cost_per_watt = cost / energy_to_add
                    if self._support_auto:
                        new_cmd = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=self._power_sorted_cmds[power_to_add_idx].power_consign)
                    else:
                        new_cmd = copy_command(self._power_sorted_cmds[power_to_add_idx])

                    costs_optimizers.append((cost_per_watt, cost, energy_to_add, i, power_to_add, new_cmd))

        costs_optimizers: list[tuple[float, float, float, int, float, LoadCommand]] = sorted(costs_optimizers, key=lambda x: x[0])


        for price in prices_ordered_values:

            # first add the optimizers if their cost per watt is lower than the current price
            while costs_optimizers[0][0] <= price:
                _, _, _, slot_idx, c_power, c_cmd = costs_optimizers.pop(0)
                prev_energy = 0
                if out_commands[slot_idx] is not None:
                    prev_energy = (out_commands[slot_idx].power_consign * power_slots_duration_s[slot_idx]) / 3600.0
                delta_energy = (c_power * power_slots_duration_s[slot_idx]) / 3600.0 - prev_energy
                if delta_energy > 0.0:
                    out_commands[slot_idx] = c_cmd
                    out_power[slot_idx] = c_power
                    nrj_to_be_added -= delta_energy

                    if nrj_to_be_added <= 0.0:
                        break

            if nrj_to_be_added <= 0.0:
                break

            price_span_h = (np.sum(power_slots_duration_s[first_slot:last_slot+1], where = prices[first_slot:last_slot+1] == price))/3600.0

            #to try to fill as smoothly as possible: is it possible to fill the slot with the maximum power value?
            fill_power_idx = 0
            for fill_power_idx in range(len(self._power_sorted_cmds)):
                if (nrj_to_be_added / self._power_sorted_cmds[fill_power_idx].power_consign) < price_span_h:
                    break

            if self._support_auto:
                price_cmd = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=self._power_sorted_cmds[fill_power_idx].power_consign)
            else:
                price_cmd = copy_command(self._power_sorted_cmds[fill_power_idx])

            price_power = price_cmd.power_consign

            for i in range(first_slot, last_slot + 1):

                if prices[i] == price and out_commands[i] is None:

                    current_energy : float = (price_power * float(power_slots_duration_s[i]))/3600.0
                    out_commands[i] = price_cmd
                    out_power[i] = price_power
                    nrj_to_be_added -= current_energy

                    if nrj_to_be_added <= 0.0:
                        break

            if nrj_to_be_added <= 0.0:
                break

        return nrj_to_be_added <= 0.0, out_commands, out_power


class MultiStepsPowerLoadConstraintChargePercent(MultiStepsPowerLoadConstraint):

    def __init__(self,
                 total_capacity_wh: float,
                 power_steps: list[LoadCommand] = None,
                 support_auto: bool = False,
                 **kwargs):

        super().__init__(power_steps=power_steps, support_auto=support_auto, **kwargs)
        self.total_capacity_wh = total_capacity_wh


    def convert_target_value_to_energy(self, value: float) -> float:
        return (value * self.total_capacity_wh) / 100.0

    def convert_energy_to_target_value(self, energy: float) -> float:
        return (100.0*energy)/self.total_capacity_wh


    def best_duration_to_meet(self) -> timedelta:
        """ Return the best duration to meet the constraint."""
        seconds = (3600.0 * (((self.target_value - self.current_value)*self.total_capacity_wh)/100.0)) / self._max_power
        return timedelta(seconds=seconds)

    def compute_value(self, time: datetime) -> float:
        """ Compute the value of the constraint whenever it is called changed state or not """
        if self.load.current_command is None:
            return self.current_value

        return 100.0*((((time - self.last_value_update).total_seconds()) * self.load.current_command.power_consign / 3600.0)/self.total_capacity_wh)+ self.current_value



class SimplePowerLoadConstraint(MultiStepsPowerLoadConstraint):


    def __init__(self,
                 power: float = 0.0,
                 **kwargs
                 ):
        power_steps = [copy_command(CMD_ON, power_consign=power)]
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


    def compute_value(self, time: datetime) -> float:
        """ Compute the value of the constraint whenever it is called changed state or not,
        hence use the old state and the last value change to add the consummed energy """
        if self.load.current_command == CMD_ON:
            return (time - self.last_value_update).total_seconds() + self.current_value
        else:
            return self.current_value

    def convert_target_value_to_energy(self, value: float) -> float:
        return (self._power * self.target_value)/3600.0

    def convert_energy_to_target_value(self, energy: float) -> float:
        return (energy*3600.0)/self._power


DATETIME_MAX_UTC = datetime.max.replace(tzinfo=pytz.UTC)
DATETIME_MIN_UTC = datetime.min.replace(tzinfo=pytz.UTC)
