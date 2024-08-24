import copy
from datetime import datetime
from datetime import date
from datetime import timedelta
from abc import abstractmethod
from typing import Callable, Self, Awaitable
import pytz

from .commands import LoadCommand, CMD_ON, CMD_AUTO_GREEN_ONLY, CMD_AUTO_FROM_CONSIGN, copy_command
import numpy.typing as npt
import numpy as np
from bisect import bisect_left

import importlib




class LoadConstraint(object):

    def __init__(self,
                 load,
                 load_param:str |None = None,
                 from_user: bool = False,
                 mandatory: bool = True,
                 start_of_constraint: datetime | None = None,
                 end_of_constraint: datetime | None = None,
                 initial_value: float = 0.0,
                 target_value: float = 0.0,
                 **kwargs
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
        self.load_param = load_param
        self.from_user = from_user
        self._update_value_callback = load.get_update_value_callback_for_constraint_class(self.__class__.__name__)


        # ok fill form the args
        if end_of_constraint is None:
            end_of_constraint = DATETIME_MAX_UTC

        if start_of_constraint is None:
            start_of_constraint = DATETIME_MIN_UTC

        self.end_of_constraint: datetime = end_of_constraint
        self.is_mandatory = mandatory
        self.start_of_constraint: datetime = start_of_constraint
        self.initial_value = initial_value
        self.target_value = target_value

        self._internal_initial_value = self.initial_value
        self.current_value = self._internal_initial_value
        self.name = f"Constraint for {self.load.name} ({self.initial_value}/{self.target_value}/{self.is_mandatory})"
        self._internal_start_of_constraint: datetime = self.start_of_constraint # this one can be overriden by the prev onstraint of the load
        nt = datetime.now(pytz.UTC)
        self.last_value_update = nt
        self.last_state_update = nt
        self.skip = False
        self.pushed_count = 0


    def __eq__(self, other):
        if other.to_dict() == self.to_dict():
            return other.name == self.name
        return False

    @classmethod
    def new_from_saved_dict(cls, load, data: dict) -> Self:
        try:
            module = importlib.import_module("quiet_solar.home_model.constraints")
        except:
            return None
        if module is None:
            return None
        class_name = data.pop("qs_class_type", None)
        if class_name is None:
            return None
        my_class = getattr(module,class_name)
        kwargs = my_class.from_dict_to_kwargs(data)
        my_instance = my_class(load = load, **kwargs)
        return my_instance

    def to_dict(self) -> dict:
        return {
            "qs_class_type": self.__class__.__name__,
            "load_param": self.load_param,
            "from_user": self.from_user,
            "mandatory": self.is_mandatory,
            "initial_value": self.initial_value,
            "target_value": self.target_value,
            "start_of_constraint" : f"{self.start_of_constraint}",
            "end_of_constraint": f"{self.end_of_constraint}"
        }

    @classmethod
    def from_dict_to_kwargs(cls, data: dict) -> dict:
        res = copy.deepcopy(data)
        res["start_of_constraint"] = datetime.fromisoformat(data["start_of_constraint"])
        res["end_of_constraint"] = datetime.fromisoformat(data["end_of_constraint"])
        return res


    def convert_target_value_to_energy(self, value: float) -> float:
        return value

    def convert_energy_to_target_value(self, energy: float) -> float:
        return energy


    def _get_target_date_string(self) -> str:
        if self.end_of_constraint == DATETIME_MAX_UTC or self.end_of_constraint is None:
            target = "no time constraint"
        else:
            local_target_date = self.end_of_constraint.replace(tzinfo=pytz.UTC).astimezone(tz=None)
            local_constraint_day = datetime(local_target_date.year, local_target_date.month, local_target_date.day)
            local_today = date.today()
            local_tomorrow = date.today() + timedelta(days=1)
            if local_constraint_day == local_today:
                target = "by today: " + local_target_date.strftime("%Y-%m-%d %H:%M")
            elif local_constraint_day == local_tomorrow:
                target = "by tomorrow: " + local_target_date.strftime("%Y-%m-%d %H:%M")
            else:
                target = local_target_date.strftime("%Y-%m-%d %H:%M")

        return target

    def _get_readable_target_value_string(self) -> str:
        target_string =  f"{int(self.target_value)} Wh"
        if self.target_value >= 2000:
            target_string = f"{int(self.target_value / 1000)} kWh"
        return target_string

    def get_readable_name_for_load(self) -> str:
        target_date = self._get_target_date_string()
        target_string = self._get_readable_target_value_string()
        if self.is_mandatory:
            strength = "for sure"
        else:
            strength = "best effort"

        return f"{target_string} {target_date} ({strength})"



    def is_constraint_active_for_time_period(self, start_time: datetime, end_time: datetime | None = None) -> bool:

        if self.is_constraint_met():
            return False

        if end_time is None:
            end_time = DATETIME_MAX_UTC

        if self.end_of_constraint == DATETIME_MAX_UTC:
            if self._internal_start_of_constraint > end_time:
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

        if self.target_value > self._internal_initial_value and current_value >= self.target_value:
            return True

        if self.target_value < self._internal_initial_value and current_value <= self.target_value:
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
                 power: float | None = None,
                 support_auto: bool = False,
                 **kwargs):

        super().__init__(**kwargs)

        if power_steps is None and power is not None:
            power_steps = [copy_command(CMD_ON, power_consign=power)]

        self._power_cmds = power_steps
        self._power_sorted_cmds = [c for c in power_steps if c.power_consign > 0.0]
        self._power_sorted_cmds = sorted(self._power_sorted_cmds, key=lambda x: x.power_consign)
        self._power = self._power_sorted_cmds[-1].power_consign
        self._max_power = self._power_sorted_cmds[-1].power_consign
        self._min_power = self._power_sorted_cmds[0].power_consign
        self._support_auto = support_auto

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["power_steps"] = [c.to_dict() for c in self._power_cmds]
        data["support_auto"] = self._support_auto
        return data

    @classmethod
    def from_dict_to_kwargs(cls, data: dict) -> dict:
        res = super().from_dict_to_kwargs(data)
        res["power_steps"] = [LoadCommand(**c) for c in data["power_steps"]]
        return res

    def best_duration_to_meet(self) -> timedelta:
        """ Return the best duration to meet the constraint."""
        seconds = (3600.0 * (self.target_value - self._internal_initial_value)) / self._max_power
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

        if self._internal_start_of_constraint != DATETIME_MIN_UTC:
            first_slot = bisect_left(time_slots, self._internal_start_of_constraint)

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
                 **kwargs):

        super().__init__(**kwargs)
        self.total_capacity_wh = total_capacity_wh


    def to_dict(self) -> dict:
        data = super().to_dict()
        data["total_capacity_wh"] = self.total_capacity_wh
        return data

    def _get_readable_target_value_string(self) -> str:
        target_string =  f"{int(round(self.target_value))} %"
        return target_string

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


class TimeBasedSimplePowerLoadConstraint(MultiStepsPowerLoadConstraint):


    def _get_readable_target_value_string(self) -> str:
        target_string =  f"{int(self.target_value)} s"
        if self.target_value >= 4*3600:
            target_string = f"{int(round(self.target_value / 3600))} h"
        elif self.target_value >= 3600:
            target_string = f"{int(self.target_value / 3600)} h {int((self.target_value % 3600)/60)} mn"
        elif self.target_value >= 60:
            target_string = f"{int(self.target_value / 60)} mn"
        return target_string


    def best_duration_to_meet(self) -> timedelta:
        """ Return the best duration to meet the constraint."""
        return timedelta(seconds=self.target_value - self._internal_initial_value)


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
