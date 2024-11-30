import copy
import logging
from datetime import datetime
from datetime import date
from datetime import timedelta
from abc import abstractmethod
from typing import Self
import pytz

from .commands import LoadCommand, CMD_ON, CMD_AUTO_GREEN_ONLY, CMD_AUTO_FROM_CONSIGN, copy_command, CMD_OFF, CMD_IDLE
import numpy.typing as npt
import numpy as np
from bisect import bisect_left

import importlib

from ..const import CONSTRAINT_TYPE_FILLER_AUTO, CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE, \
    CONSTRAINT_TYPE_MANDATORY_END_TIME, CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN

_LOGGER = logging.getLogger(__name__)


class LoadConstraint(object):

    def __init__(self,
                 time: datetime,
                 load,
                 load_param: str | None = None,
                 from_user: bool = False,
                 type: int = CONSTRAINT_TYPE_FILLER_AUTO,
                 start_of_constraint: datetime | None = None,
                 end_of_constraint: datetime | None = None,
                 initial_value: float | None = 0.0,
                 current_value: float | None = None,
                 target_value: float = 0.0,
                 **kwargs
                 ):

        """
        :param time: the time at constraint creation
        :param load: the load that the constraint is applied to
        :param load_param: the load param that the constraint is applied to
        :param type: of CONSTRAINT_TYPE_* behaviour of the constraint
        :param start_of_constraint: constraint start time if None the constraint start asap, depends on the type
        :param end_of_constraint: constraint end time if None the constraint is always active, depends on the type
        :param initial_value: the constraint start value if None it means the constraints will adapt from the previous
               constraint, or be computed automatically
        :param target_value: the constraint target value, growing number from initial_value to target_value
        """

        self.load = load
        self.load_param = load_param
        self.from_user = from_user
        self.type = type

        self._update_value_callback = load.get_update_value_callback_for_constraint_class(self)

        # ok fill form the args
        if end_of_constraint is None:
            end_of_constraint = DATETIME_MAX_UTC

        if start_of_constraint is None:
            start_of_constraint = DATETIME_MIN_UTC

        self.end_of_constraint: datetime = end_of_constraint
        self.initial_end_of_constraint: datetime = end_of_constraint

        self.start_of_constraint: datetime = start_of_constraint

        # this one can be overriden by the prev constraint of the load:
        self._internal_start_of_constraint: datetime = self.start_of_constraint

        self.initial_value = initial_value
        self._internal_initial_value = self.initial_value
        if initial_value is None:
            self._internal_initial_value = 0.0

        self.target_value = target_value

        if current_value is None:
            self.current_value = self._internal_initial_value
        else:
            self.current_value = current_value

        self._compute_constraint_name()

        # will correct end of constraint if needed best_duration_to_meet uses the values above
        if self.as_fast_as_possible:
            dt_to_finish = self.best_duration_to_meet()
            self.end_of_constraint = time + dt_to_finish

        self.last_value_update = time
        self.last_state_update = time
        self.skip = False
        self.pushed_count = 0


    def _compute_constraint_name(self):
        self.name = (f"Constraint for {self.load.name} ({self.load_param} "
                     f"{self.initial_value}/{self.target_value}/{self.type})")

    def reset_load_param(self, new_param):
        self.load_param = new_param
        self._compute_constraint_name()


    def __eq__(self, other):
        if other is None:
            return None
        if other.to_dict() == self.to_dict():
            return other.name == self.name
        return False

    @property
    def as_fast_as_possible(self) -> bool:
        return self.type >= CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE

    @property
    def is_mandatory(self) -> bool:
        return self.type >= CONSTRAINT_TYPE_MANDATORY_END_TIME

    @property
    def is_before_battery(self) -> bool:
        return self.type >= CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN

    def score(self):
        score_offset = 10000000000
        score = int(100.0*self.convert_target_value_to_energy(self.target_value))
        score += self.type * score_offset
        if self.from_user:
            score += 100*score_offset
        return score

    @classmethod
    def new_from_saved_dict(cls, time, load, data: dict) -> Self:
        try:
            module = importlib.import_module(__name__)
        except:
            module = None

        if module is None:
            try:
                module = importlib.import_module("quiet_solar.home_model.constraints")
            except:
                module = None

        if module is None:
            _LOGGER.error(f"Cannot import the module to load constraint {__name__} {data}")
            return None

        class_name: str | None = data.get("qs_class_type", None)
        if class_name is None:
            _LOGGER.error(f"Cannot import the load constraint: no class {data}")
            return None
        my_class = getattr(module, class_name)
        kwargs = my_class.from_dict_to_kwargs(data)
        my_instance = my_class(time=time, load=load, **kwargs)
        return my_instance

    def to_dict(self) -> dict:
        return {
            "qs_class_type": self.__class__.__name__,
            "type": self.type,
            "load_param": self.load_param,
            "from_user": self.from_user,
            "initial_value": self.initial_value,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "start_of_constraint": f"{self.start_of_constraint}",
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
            target = "no limit"
        else:
            local_target_date = self.end_of_constraint.replace(tzinfo=pytz.UTC).astimezone(tz=None)
            local_constraint_day = datetime(local_target_date.year, local_target_date.month, local_target_date.day)
            local_today = date.today()
            local_today = datetime(local_today.year, local_today.month, local_today.day)
            local_tomorrow = local_today + timedelta(days=1)
            if local_constraint_day == local_today:
                target = "by today " + local_target_date.strftime("%H:%M")
            elif local_constraint_day == local_tomorrow:
                target = "by tomorrow " + local_target_date.strftime("%H:%M")
            else:
                target = local_target_date.strftime("%Y-%m-%d %H:%M")

        return target

    def _get_readable_target_value_string(self) -> str:
        target_string = f"{int(self.target_value)} Wh"
        if self.target_value >= 2000:
            target_string = f"{int(self.target_value / 1000)} kWh"
        return target_string

    def get_readable_name_for_load(self) -> str:
        target_date = self._get_target_date_string()
        target_string = self._get_readable_target_value_string()

        postfix = ""
        if self.type < CONSTRAINT_TYPE_MANDATORY_END_TIME:
            postfix = " best effort"
        elif self.type >= CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE:
            postfix = " ASAP"

        prefix = ""
        if self.load_param:
            prefix = f"{self.load_param}:"

        return f"{prefix}{target_string} {target_date}{postfix}"

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

        # only active if the constraint finish before the end of the given time period
        return start_time <= self.end_of_constraint <= end_time

    def is_constraint_met(self, current_value=None) -> bool:
        """ is the constraint met in its current form? """
        if current_value is None:
            current_value = self.current_value

        if self.target_value is None:
            return False

        if current_value >= (0.995*self.target_value): #0.5% tolerance
            return True

        return False

    def reset_initial_value_to_follow_prev_if_needed(self, time: datetime, prev_constraint: Self):

        if self.initial_value is not None:
            return

        prev_constraint_energy_value = prev_constraint.convert_target_value_to_energy(prev_constraint.target_value)
        prev_constraint_value = prev_constraint.convert_energy_to_target_value(prev_constraint_energy_value)
        self._internal_initial_value = prev_constraint_value
        self.current_value = prev_constraint_value

    async def update(self, time: datetime) -> bool:
        """ Update the constraint with the new value. to be called by a load that
            can compute the value based or sensors or external data"""
        if time >= self.last_value_update:
            do_continue_constraint = True
            if self._update_value_callback is not None:
                value, do_continue_constraint = await self._update_value_callback(self, time)
                if do_continue_constraint is False:
                    _LOGGER.info(f"{self.name} update callback asked for stop")
            else:
                value = self.compute_value(time)

            if value is None:
                value = self.current_value

            self.last_value_update = time
            self.current_value = value
            if do_continue_constraint is False or self.is_constraint_met():
                return False
        return True

    @abstractmethod
    def compute_value(self, time: datetime) -> float | None:
        """ Compute the value of the constraint when the state change."""

    @abstractmethod
    def best_duration_to_meet(self) -> timedelta:
        """ Return the best duration to meet the constraint."""

    @abstractmethod
    def compute_best_period_repartition(self,
                                        do_use_available_power_only: bool,
                                        power_available_power: npt.NDArray[np.float64],
                                        power_slots_duration_s: npt.NDArray[np.float64],
                                        prices: npt.NDArray[np.float64],
                                        prices_ordered_values: list[float],
                                        time_slots: list[datetime]) -> tuple[
                                        bool, list[LoadCommand | None], npt.NDArray[np.float64]]:
        """ Compute the best repartition of the constraint over the given period."""


class MultiStepsPowerLoadConstraint(LoadConstraint):

    def __init__(self,
                 power_steps: list[LoadCommand] = None,
                 power: float | None = None,
                 support_auto: bool = False,
                 **kwargs):

        # do this before super as best_duration_to_meet is used in the super call
        if kwargs.get("type", CONSTRAINT_TYPE_FILLER_AUTO) >= CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE:
            support_auto = False
            mx_power = 0.0
            if power is not None:
                mx_power = power
            elif power_steps is not None:
                sorted_cmds = [c for c in power_steps if c.power_consign > 0.0]
                sorted_cmds = sorted(sorted_cmds, key=lambda x: x.power_consign)
                mx_power = sorted_cmds[-1].power_consign

            power_steps = [copy_command(CMD_ON, power_consign=mx_power)]
        else:
            if power_steps is None and power is not None:
                power_steps = [copy_command(CMD_ON, power_consign=power)]

        self._power_cmds = power_steps
        self._power_sorted_cmds = [c for c in power_steps if c.power_consign > 0.0]
        self._power_sorted_cmds = sorted(self._power_sorted_cmds, key=lambda x: x.power_consign)
        self._power = self._power_sorted_cmds[-1].power_consign
        self._max_power = self._power_sorted_cmds[-1].power_consign
        self._min_power = self._power_sorted_cmds[0].power_consign
        self.support_auto = support_auto

        super().__init__(**kwargs)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["power_steps"] = [c.to_dict() for c in self._power_cmds]
        data["support_auto"] = self.support_auto
        return data

    @classmethod
    def from_dict_to_kwargs(cls, data: dict) -> dict:
        res = super().from_dict_to_kwargs(data)
        res["power_steps"] = [LoadCommand(**c) for c in data["power_steps"]]
        return res

    def best_duration_to_meet(self) -> timedelta:
        """ Return the best duration to meet the constraint."""
        seconds = (3600.0 * (self.target_value - self.current_value)) / self._max_power
        return timedelta(seconds=seconds)

    def compute_value(self, time: datetime) -> float | None:
        """ Compute the value of the constraint whenever it is called changed state or not """
        if self.load.current_command is None or self.load.current_command.is_off_or_idle():
            return None

        return (((time - self.last_value_update).total_seconds()) *
                self.load.current_command.power_consign / 3600.0) + self.current_value

    def compute_best_period_repartition(self,
                                        do_use_available_power_only: bool,
                                        power_available_power: npt.NDArray[np.float64],
                                        power_slots_duration_s: npt.NDArray[np.float64],
                                        prices: npt.NDArray[np.float64],
                                        prices_ordered_values: list[float],
                                        time_slots: list[datetime]) -> tuple[
                                        bool, list[LoadCommand | None], npt.NDArray[np.float64]]:
        nrj_to_be_added = self.convert_target_value_to_energy(self.target_value) - self.convert_target_value_to_energy(
            self.current_value)
        return self._compute_best_period_repartition(do_use_available_power_only, power_available_power,
                                                     power_slots_duration_s, prices, prices_ordered_values, time_slots,
                                                     nrj_to_be_added)

    def _num_command_state_change(self, out_commands: list[LoadCommand | None]):
        num = 0
        prev_cmd = None
        empty_cmds = []
        current_empty = None

        if len(out_commands) > 0:

            #do not add the first empty : no merge for first and end
            #if out_commands[0] is None:
            #    current_empty = [0,0]

            for i, cmd in enumerate(out_commands):
                if (cmd is None and prev_cmd is not None) or (cmd is not None and prev_cmd is None):
                    num += 1
                    if cmd is None:
                        current_empty = [i,i]
                    else:
                        if current_empty is not None:
                            current_empty[1] = i
                            empty_cmds.append(current_empty)
                            current_empty = None

                prev_cmd = cmd
            # do not add the last empty : no merge for first and end
            #if current_empty is not None:
            #    current_empty[1] = len(out_commands)
            #    empty_cmds.append(current_empty)

        return num, empty_cmds


    def _adapt_commands(self, out_commands, out_power, power_slots_duration_s, nrj_to_be_added):

        if self.load.num_max_on_off is not None and self.support_auto is False:
            num_command_state_change, inner_empty_cmds = self._num_command_state_change(out_commands)
            num_allowed_switch = self.load.num_max_on_off - self.load.num_on_off
            if num_command_state_change > num_allowed_switch:
                # too many state changes .... need to merge some commands
                # keep only the main one as it is solar only


                for empty_cmd in inner_empty_cmds:
                    empty_cmd.append(0)
                    for i in range(empty_cmd[0], empty_cmd[1]):
                        empty_cmd[2] += power_slots_duration_s[i]

                #removed the smallest holes first
                inner_empty_cmds = sorted(inner_empty_cmds, key=lambda x: x[2])

                num_removed = 0

                for empty_cmd in inner_empty_cmds:
                    num_removed += 1
                    for i in range(empty_cmd[0], empty_cmd[1]):
                        out_commands[i] = copy_command(self._power_sorted_cmds[0])
                        out_power[i] = out_commands[i].power_consign
                        nrj_to_be_added -= (out_power[i] * power_slots_duration_s[i]) / 3600.0
                        num_command_state_change -= 2  # 2 changes removed
                        if num_command_state_change <= num_allowed_switch:
                            break

                _LOGGER.info(f"Adapted command for on_off num/max: {self.load.num_on_off}/{self.load.num_max_on_off} Removed empty segments {num_removed}")

        return nrj_to_be_added

    def _compute_best_period_repartition(self,
                                         do_use_available_power_only: bool,
                                         power_available_power: npt.NDArray[np.float64],
                                         power_slots_duration_s: npt.NDArray[np.float64],
                                         prices: npt.NDArray[np.float64],
                                         prices_ordered_values: list[float],
                                         time_slots: list[datetime],
                                         nrj_to_be_added: float) -> tuple[
                                         bool, list[LoadCommand | None], npt.NDArray[np.float64]]:

        out_power = np.zeros(len(power_available_power), dtype=np.float64)
        out_power_idxs = np.zeros(len(power_available_power), dtype=np.int64)
        out_commands: list[LoadCommand | None] = [None] * len(power_available_power)


        # first get to the available power slots (ie with negative power available, fill it at best in a greedy way
        min_power = self._min_power

        first_slot = 0
        last_slot = len(power_available_power) - 1

        # find the constraint last slot
        if self.end_of_constraint is not None:
            # by construction the end of the constraints IS ending on a slot
            last_slot = bisect_left(time_slots, self.end_of_constraint)
            last_slot = max(0, last_slot - 1)  # -1 as the last_slot index is the index of the slot not the time anchor
            if last_slot >= len(power_available_power):
                last_slot = len(power_available_power) - 1

        if self._internal_start_of_constraint != DATETIME_MIN_UTC:
            first_slot = bisect_left(time_slots, self._internal_start_of_constraint)


        sub_power_available_power = power_available_power[first_slot:last_slot + 1]
        min_power_idx = np.argmin(sub_power_available_power)
        left = min_power_idx
        right = min_power_idx
        sorted_available_power = [min_power_idx]
        for i in range(len(sub_power_available_power) - 1):
            if left == 0:
                right = right + 1
                if right >= len(sub_power_available_power):
                    # should never happen
                    break
                sorted_available_power.append(right)
            elif right == len(sub_power_available_power) - 1:
                left = left - 1
                if left < 0:
                    # should never happen
                    break
                sorted_available_power.append(left)
            else:
                left_val = sub_power_available_power[left - 1]
                right_val = sub_power_available_power[right + 1]

                if left_val < right_val:
                    left = left - 1
                    sorted_available_power.append(left)
                else:
                    right = right + 1
                    sorted_available_power.append(right)

        if len(sorted_available_power) != len(sub_power_available_power):
            _LOGGER.error(f"ordered_exploration is not the same size as sub_power_available_power {len(sorted_available_power)} != {len(sub_power_available_power)}")
            sorted_available_power = sub_power_available_power.argsort() # power_available_power negative value means free power


        # for i in range(first_slot, last_slot + 1):

        # try to shave first the biggest free slots
        for i_sorted in sorted_available_power:

            i = i_sorted + first_slot

        #for i in range(first_slot, last_slot + 1):
            if power_available_power[i] <= -min_power:
                j = 0
                while j < len(self._power_sorted_cmds) - 1 and self._power_sorted_cmds[j + 1].power_consign < - \
                        power_available_power[i]:
                    j += 1

                if self.support_auto:
                    out_commands[i] = copy_command(CMD_AUTO_GREEN_ONLY,
                                                   power_consign=self._power_sorted_cmds[j].power_consign)
                else:
                    out_commands[i] = copy_command(self._power_sorted_cmds[j])

                out_power[i] = out_commands[i].power_consign
                out_power_idxs[i] = j

                nrj_to_be_added -= (out_power[i] * power_slots_duration_s[i]) / 3600.0

                if nrj_to_be_added <= 0.0:
                    break

        if (nrj_to_be_added <= 0.0 or
                do_use_available_power_only or
                self.type <= CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN):

            nrj_to_be_added = self._adapt_commands(out_commands, out_power, power_slots_duration_s, nrj_to_be_added)

            if self.support_auto:
                # fill all with green only command, to fill everything with the best power available
                for i in range(first_slot, last_slot + 1):
                    if out_commands[i] is None:
                        out_commands[i] = copy_command(CMD_AUTO_GREEN_ONLY, power_consign=0.0)
                        out_power[i] = 0.0

            return nrj_to_be_added <= 0.0, out_commands, out_power

        # pure solar was not enough, we will try to see if we can get a more solar energy directly if price is better
        costs_optimizers = []

        for i in range(first_slot, last_slot + 1):

            if power_available_power[i] + out_power[i] < -min_power/4:
                # there is still some solar to get
                power_to_add_idx = -1
                if out_commands[i] is None:
                    power_to_add_idx = 0
                elif out_power_idxs[i] < len(self._power_sorted_cmds) - 1:
                    power_to_add_idx = out_power_idxs[i] + 1

                if power_to_add_idx >= 0:
                    # all command in ECO mode now....
                    power_to_add: float = self._power_sorted_cmds[power_to_add_idx].power_consign
                    energy_to_add: float = (power_to_add * float(power_slots_duration_s[i])) / 3600.0
                    cost: float = ((power_to_add + float(power_available_power[i])) * float(
                        power_slots_duration_s[i]) * float(prices[i])) / 3600.0
                    cost_per_watt = cost / energy_to_add
                    if self.support_auto:
                        new_cmd = copy_command(CMD_AUTO_FROM_CONSIGN,
                                               power_consign=self._power_sorted_cmds[power_to_add_idx].power_consign)
                    else:
                        new_cmd = copy_command(self._power_sorted_cmds[power_to_add_idx])

                    costs_optimizers.append((cost_per_watt, cost, energy_to_add, i, power_to_add, new_cmd))

        costs_optimizers: list[tuple[float, float, float, int, float, LoadCommand]] = sorted(costs_optimizers,
                                                                                             key=lambda x: x[0])

        for price in prices_ordered_values:

            if costs_optimizers:
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
                    if len(costs_optimizers) == 0:
                        break

                if nrj_to_be_added <= 0.0:
                    break



            if len(self._power_sorted_cmds) > 1:

                price_span_h = ((np.sum(power_slots_duration_s[first_slot:last_slot + 1],
                                        where=prices[first_slot:last_slot + 1] == price)) / 3600.0)

                # to try to fill as smoothly as possible: is it possible to fill the slot with the maximum power value?
                fill_power_idx = 0
                for fill_power_idx in range(len(self._power_sorted_cmds)):
                    if nrj_to_be_added <= price_span_h * self._power_sorted_cmds[fill_power_idx].power_consign:
                        break
            else:
                fill_power_idx = 0

            # used to spread the commands : be a bit conservative on teh spanning and use fill_power_aggressive_idx for the commands
            price_power = self._power_sorted_cmds[fill_power_idx].power_consign

            explore_range = range(last_slot, first_slot - 1, -1)
            if self.support_auto:
                price_cmd = copy_command(CMD_AUTO_FROM_CONSIGN,
                                         power_consign=self._power_sorted_cmds[fill_power_idx].power_consign)

                if self.load and self.load.current_command and self.load.current_command.is_like(CMD_AUTO_FROM_CONSIGN):
                    # we are already in auto consign mode for this load : we want to keep the continuity of the command
                    if first_slot == 0 and prices[first_slot] == price:
                        # in this particular case : we will go from now to end to keep the continuity with the current command
                        explore_range = range(first_slot, last_slot + 1)
                        # if not done : force the first slot to be a consign!
                        if out_commands[first_slot] is not None and out_commands[first_slot].is_like(CMD_AUTO_GREEN_ONLY):
                            out_commands[first_slot] = price_cmd
                            # put back the removed energy
                            nrj_to_be_added += (out_power[first_slot] * power_slots_duration_s[first_slot]) / 3600.0
                            out_power[first_slot] = price_power
                            nrj_to_be_added -= (price_power * power_slots_duration_s[first_slot]) / 3600.0

            else:
                price_cmd = copy_command(self._power_sorted_cmds[fill_power_idx])


            # go reverse to respect the end constraint the best we can? or at the contrary fill it as soon as possible?
            # may depend on the load type for a boiler you want to be closer, for a car it is more the asap? let's do reverse

            for i in explore_range:

                if prices[i] == price and out_commands[i] is None:

                    current_energy: float = (price_power * float(power_slots_duration_s[i])) / 3600.0
                    out_commands[i] = price_cmd
                    out_power[i] = price_power
                    nrj_to_be_added -= current_energy

                    if nrj_to_be_added <= 0.0:
                        break

            if nrj_to_be_added <= 0.0:
                break

        nrj_to_be_added = self._adapt_commands(out_commands, out_power, power_slots_duration_s, nrj_to_be_added)

        if self.support_auto:
            # fill all with green only command, to fill everything with the best power available
            for i in range(first_slot, last_slot + 1):
                if out_commands[i] is None:
                    out_commands[i] = copy_command(CMD_AUTO_GREEN_ONLY, power_consign=0.0)
                    out_power[i] = 0.0

            if self.load and self.load.current_command and self.load.current_command.command == CMD_AUTO_FROM_CONSIGN.command and (out_commands[0].command == CMD_AUTO_GREEN_ONLY.command or out_commands[first_slot].command == CMD_AUTO_GREEN_ONLY.command):
                _LOGGER.info(f"We will switch from a consign to a green only command for {self.load.name} {self.load_param} {first_slot}/{last_slot}")
                for i in range(first_slot, last_slot + 1):
                    _LOGGER.info(f" ==> CMD: {time_slots[i]} {out_commands[i]} ")


        return nrj_to_be_added <= 0.0, out_commands, out_power


class MultiStepsPowerLoadConstraintChargePercent(MultiStepsPowerLoadConstraint):

    def __init__(self,
                 total_capacity_wh: float,
                 **kwargs):
        # do this before super as best_duration_to_meet is used in the super call
        self.total_capacity_wh = total_capacity_wh

        super().__init__(**kwargs)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["total_capacity_wh"] = self.total_capacity_wh
        return data

    def _get_readable_target_value_string(self) -> str:
        target_string = f"{int(round(self.target_value))} %"
        return target_string

    def convert_target_value_to_energy(self, value: float) -> float:
        return (value * self.total_capacity_wh) / 100.0

    def convert_energy_to_target_value(self, energy: float) -> float:
        return (100.0 * energy) / self.total_capacity_wh

    def best_duration_to_meet(self) -> timedelta:
        """ Return the best duration to meet the constraint."""
        seconds = (3600.0 * (
                ((self.target_value - self.current_value) * self.total_capacity_wh) / 100.0)) / self._max_power
        return timedelta(seconds=seconds)

    def compute_value(self, time: datetime) -> float | None:
        """ Compute the value of the constraint whenever it is called changed state or not """
        if self.load.current_command is None or self.load.current_command.is_off_or_idle():
            return None

        return 100.0 * ((((time - self.last_value_update).total_seconds()) *
                         self.load.current_command.power_consign / 3600.0) /
                        self.total_capacity_wh) + self.current_value


class TimeBasedSimplePowerLoadConstraint(MultiStepsPowerLoadConstraint):

    def _get_readable_target_value_string(self) -> str:
        minutes = int((self.target_value % 3600) / 60)
        hours = int(self.target_value / 3600)
        target_string = f"{int(self.target_value)}s"
        if self.target_value >= 4 * 3600:
            target_string = f"{hours}h"
        elif self.target_value >= 3600:
            if minutes > 0:
                target_string = f"{hours}h{minutes}mn"
            else:
                target_string = f"{hours}h"
        elif self.target_value >= 60:
            target_string = f"{int(self.target_value / 60)}mn"
        return target_string

    def best_duration_to_meet(self) -> timedelta:
        """ Return the best duration to meet the constraint."""
        return timedelta(seconds=self.target_value - self.current_value)

    def compute_value(self, time: datetime) -> float | None:
        """ Compute the value of the constraint whenever it is called changed state or not,
        hence use the old state and the last value change to add the consummed energy """
        if self.load.current_command is not None and self.load.current_command.is_like(CMD_ON):
            return (time - self.last_value_update).total_seconds() + self.current_value
        else:
            return None

    def convert_target_value_to_energy(self, value: float) -> float:
        return (self._power * value) / 3600.0

    def convert_energy_to_target_value(self, energy: float) -> float:
        return (energy * 3600.0) / self._power


DATETIME_MAX_UTC = datetime.max.replace(tzinfo=pytz.UTC)
DATETIME_MIN_UTC = datetime.min.replace(tzinfo=pytz.UTC)
