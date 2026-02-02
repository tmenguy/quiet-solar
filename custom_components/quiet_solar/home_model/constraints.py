import copy
import logging
from datetime import datetime
from datetime import date
from datetime import timedelta
from abc import abstractmethod
from typing import Self
import pytz

from .commands import LoadCommand, CMD_ON, CMD_AUTO_GREEN_ONLY, CMD_AUTO_FROM_CONSIGN, copy_command, \
    copy_command_and_change_type, CMD_IDLE, \
    CMD_AUTO_PRICE, CMD_AUTO_GREEN_CAP, CMD_AUTO_GREEN_CONSIGN
import numpy.typing as npt
import numpy as np
from bisect import bisect_left

import importlib

from .home_utils import is_amps_greater, min_amps, add_amps, is_amps_zero
from ..const import CONSTRAINT_TYPE_FILLER_AUTO, CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE, \
    CONSTRAINT_TYPE_MANDATORY_END_TIME, CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN, CONSTRAINT_TYPE_FILLER, SOLVER_STEP_S

_LOGGER = logging.getLogger(__name__)


def get_readable_date_string(time:datetime | None, for_small_standalone: bool = False) -> str:
    if time is None or time == DATETIME_MAX_UTC or time == DATETIME_MIN_UTC:
        if for_small_standalone:
            return "--:--"
        else:
            return ""
    else:
        local_target_date = time.replace(tzinfo=pytz.UTC).astimezone(tz=None)
        local_constraint_day = datetime(local_target_date.year, local_target_date.month, local_target_date.day)
        local_today = date.today()
        local_today = datetime(local_today.year, local_today.month, local_today.day)
        local_tomorrow = local_today + timedelta(days=1)
        local_now = datetime.now(tz=pytz.UTC).replace(tzinfo=pytz.UTC).astimezone(tz=None)

        if for_small_standalone:
            if local_now + timedelta(days=1) > local_target_date:
                # the target hour/mn/ss is enough to describe unambiguously the target date vs now
                target = local_target_date.strftime("%H:%M")
            else:
                target = local_target_date.strftime("%m-%d\n%H:%M")
        elif local_constraint_day == local_today:
            target = "today " + local_target_date.strftime("%H:%M")
        elif local_constraint_day == local_tomorrow:
            target = "tomorrow " + local_target_date.strftime("%H:%M")
        else:
            target = local_target_date.strftime("%Y-%m-%d %H:%M")

    return target

class LoadConstraint(object):

    def __init__(self,
                 time: datetime | None = None,
                 load = None,
                 load_param: str | None = None,
                 load_info: dict | None = None,
                 from_user: bool = False,
                 artificial_step_to_final_value: None|int|float = None,
                 type: int = CONSTRAINT_TYPE_FILLER_AUTO,
                 degraded_type: int = CONSTRAINT_TYPE_FILLER_AUTO,
                 start_of_constraint: datetime | None = None,
                 end_of_constraint: datetime | None = None,
                 initial_value: float | None = 0.0,
                 current_value: float | None = None,
                 target_value: float = 0.0,
                 support_auto: bool = False,
                 always_end_at_end_of_constraint = False,
                 **kwargs
                 ):

        """
        :param time: the time at constraint creation
        :param load: the load that the constraint is applied to
        :param load_param: the load param that the constraint is applied to
        :param load_info: additional load information dictionary, SIMPLE dict ... direct HA serialization safe
        :param type: of CONSTRAINT_TYPE_* behaviour of the constraint
        :param start_of_constraint: constraint start time if None the constraint start asap, depends on the type
        :param end_of_constraint: constraint end time if None the constraint is always active, depends on the type
        :param initial_value: the constraint start value if None it means the constraints will adapt from the previous
               constraint, or be computed automatically
        :param target_value: the constraint target value, growing number from initial_value to target_value
        """

        self.load = load
        self.load_param = load_param
        self.load_info = load_info
        self.from_user = from_user
        self.artificial_step_to_final_value = artificial_step_to_final_value
        self._type = type
        self._degraded_type = degraded_type
        self.always_end_at_end_of_constraint = always_end_at_end_of_constraint

        self.support_auto = support_auto

        self._update_value_callback = None

        if load is not None:
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
        self.current_start_of_constraint: datetime = self.start_of_constraint

        self.initial_value = initial_value
        self._internal_initial_value = self.initial_value
        if initial_value is None:
            self._internal_initial_value = 0.0

        self.target_value = target_value
        self.requested_target_value = target_value

        if current_value is None:
            self.current_value = self._internal_initial_value
        else:
            self.current_value = current_value

        # will correct end of constraint if needed best_duration_to_meet uses the values above
        if self.as_fast_as_possible or self._type >= CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE:
            dt_to_finish = self.best_duration_to_meet()
            self.end_of_constraint = time + dt_to_finish

        self.last_value_update = None
        self.last_value_change_update = None
        self.first_value_update = None

        self.skip = False
        self.pushed_count = 0

        self.real_constraint = self

    @property
    def use_time_for_budgeting(self):
        return False

    def carry_info_from_other_constraint(self, other: Self):
        pass

    @property
    def type(self)-> int:
        if self.is_off_grid():
            return self._degraded_type
        return self._type

    @type.setter
    def type(self, value: int):
        self._type = value
        self._degraded_type = min(CONSTRAINT_TYPE_MANDATORY_END_TIME, value)

    def is_off_grid(self):
        if self.load is None:
            return False
        return self.load.is_off_grid()


    def reset_load_param(self, new_param):
        self.load_param = new_param

    @property
    def name(self):
        target_s = f"{self.target_value}"
        if self.artificial_step_to_final_value is not None:
            target_s = f"{self.target_value} -> {self.artificial_step_to_final_value}"

        extra_s = ""
        if self.load_param:
            extra_s += f"{self.load_param} "
        if self.load_info:
            extra_s += f"{self.load_info} "

        return f"Constraint for {self.load.name} ({extra_s}{self.initial_value}/{target_s}/{self._type}/{self._degraded_type})"

    @property
    def stable_name(self):
        target_value = self._get_target_value_for_readable()
        extra_s = ""
        if self.load_param:
            extra_s += f"{self.load_param} "
        if self.load_info:
            extra_s += f"{self.load_info} "
        return f"Constraint for {self.load.name} ({extra_s}{target_value}/{self._type}/{self._degraded_type})"


    def add_or_update_load_info(self, key: str, value):
        if self.load_info is None:
            self.load_info = {}
        self.load_info[key] = value

    def __eq__(self, other):
        if other is None:
            return False
        if other.to_dict() == self.to_dict():
            return other.stable_name == self.stable_name
        return False

    def eq_no_current(self, other):
        if other is None:
            return False
        od = other.to_dict()
        od["current_value"] = 0
        od["power_steps"] = 0
        d = self.to_dict()
        d["current_value"] = 0
        d["power_steps"] = 0

        return od == d

    def __hash__(self):
        # Create a hash based on the same attributes used in __eq__
        # We need to hash the dictionary representation and the name
        dict_str = str(sorted(self.to_dict().items()))
        return hash((dict_str, self.name))

    @property
    def as_fast_as_possible(self) -> bool:
        return self.type >= CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE

    @property
    def is_mandatory(self) -> bool:
        return self.type >= CONSTRAINT_TYPE_MANDATORY_END_TIME

    @property
    def is_before_battery(self) -> bool:
        return self.type >= CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN

    @is_before_battery.setter
    def is_before_battery(self, value: bool):
        if value:
            if self._type < CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN:
                self._type = CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN
            if self._degraded_type < CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN:
                self._degraded_type = CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN
        else:
            if self._type >= CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN:
                self._type = CONSTRAINT_TYPE_FILLER
            if self._degraded_type >= CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN:
                self._degraded_type = CONSTRAINT_TYPE_FILLER

    def score(self, time:datetime):

        energy_score_span = 1000000.0 # 1000kwh
        type_score_span = 10.0

        reserved_load_score_span = 1000000.0
        if self.load is not None:
            load_score = float(self.load.get_normalized_score(ct=self, time=time, score_span=reserved_load_score_span))

        energy_score = float(min(max(0, int(self.convert_target_value_to_energy(self.target_value))), int(energy_score_span) - 1))
        type_score = self.type
        user_score = 0.0
        if self.from_user:
            user_score = 1.0

        return energy_score + energy_score_span*load_score + energy_score_span*reserved_load_score_span*type_score + energy_score_span*reserved_load_score_span*type_score_span*user_score


    @classmethod
    def new_from_saved_dict(cls, time, load, data: dict) -> Self | None:
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

    def copy_to_other_type(self, time, other_constraint_class, additional_kwargs:dict) -> Self | None:
        data = self.to_dict()
        # adapt if needed the class specific parameters

        kwargs = other_constraint_class.from_dict_to_kwargs(data)
        kwargs.update(additional_kwargs)
        output = other_constraint_class(time=time, load=self.load, **kwargs)

        output.target_value = output.convert_energy_to_target_value(self.convert_target_value_to_energy(self.target_value))
        output.requested_target_value = output.convert_energy_to_target_value(self.convert_target_value_to_energy(self.requested_target_value))
        output.initial_value = output.convert_energy_to_target_value(self.convert_target_value_to_energy(self.initial_value))
        output.current_value = output.convert_energy_to_target_value(self.convert_target_value_to_energy(self.current_value))

        if self.artificial_step_to_final_value is not None:
            output.artificial_step_to_final_value = output.convert_energy_to_target_value(self.convert_target_value_to_energy(self.artificial_step_to_final_value))

        # redo a clean update so the constructor can properly do its job
        data = output.to_dict()
        kwargs = other_constraint_class.from_dict_to_kwargs(data)
        output = other_constraint_class(time=time, load=self.load, **kwargs)

        return output


    def to_dict(self) -> dict:
        return {
            "qs_class_type": self.__class__.__name__,
            "type": self._type,
            "degraded_type": self._degraded_type,
            "load_param": self.load_param,
            "load_info": self.load_info,
            "from_user": self.from_user,
            "artificial_step_to_final_value": self.artificial_step_to_final_value,
            "initial_value": self.initial_value,
            "current_value": self.current_value,
            "target_value": self.requested_target_value,
            "start_of_constraint": f"{self.start_of_constraint}",
            "end_of_constraint": f"{self.end_of_constraint}",
            "support_auto": self.support_auto,
            "always_end_at_end_of_constraint": self.always_end_at_end_of_constraint
        }

    @classmethod
    def from_dict_to_kwargs(cls, data: dict) -> dict:
        res = copy.deepcopy(data)
        res["start_of_constraint"] = datetime.fromisoformat(data["start_of_constraint"])
        res["end_of_constraint"] = datetime.fromisoformat(data["end_of_constraint"])
        return res







    def convert_target_value_to_energy(self, value: float) -> float:
        return self.load.efficiency_factor*value

    def convert_energy_to_target_value(self, energy: float) -> float:
        return max(0.0, energy)/self.load.efficiency_factor

    @abstractmethod
    def convert_target_value_to_time(self, value: float) -> float:
        '''  Convert the target value to time in seconds'''


    @abstractmethod
    def convert_time_to_target_value(self, time_s: float) -> float:
        ''' Convert time in seconds to target value'''


    def get_percent_completion(self, time) -> float | None:

        if self.current_value is None:
            return None

        target_val = self.target_value

        if target_val is None:
            return None

        init_val = self._internal_initial_value
        if init_val is None:
            return None

        if init_val == target_val:
            return None

        if target_val < init_val:
            return 100.0 * (self.current_value - target_val) / (init_val - target_val)
        else:
            return 100.0 * (self.current_value - init_val) / (target_val - init_val)

    def get_readable_next_target_date_string(self, for_small_standalone:bool=False) -> str:
        return get_readable_date_string(self.end_of_constraint, for_small_standalone=for_small_standalone)


    def _get_target_value_for_readable(self) -> int|float:
        target_value = self.target_value
        if self.artificial_step_to_final_value is not None:
            target_value = self.artificial_step_to_final_value
        return target_value


    def _get_readable_target_value_string(self) -> str:
        target_value = self._get_target_value_for_readable()
        target_string = f"{int(target_value)} Wh"
        if target_value >= 2000:
            target_string = f"{int(target_value / 1000)} kWh"
        return target_string

    def get_readable_name_for_load(self) -> str:
        target_date = self.get_readable_next_target_date_string()
        target_string = self._get_readable_target_value_string()

        postfix = ""
        if self.type < CONSTRAINT_TYPE_MANDATORY_END_TIME:
            postfix = " best effort"
        elif self.type >= CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE:
            postfix = " ASAP"

        prefix = ""
        if self.load_param:
            prefix = f"{self.load_param}"
        if self.load_info:
            prefix += " ("
            i = 0
            for k,v in self.load_info.items():
                if i > 0:
                    prefix += ", "
                prefix += f"{k}:{v}"
                i += 1
            prefix += ")"
        if prefix != "":
            prefix = f"{prefix}: "

        if target_date:
            target_date = f" {target_date}"

        return f"{prefix}{target_string}{target_date}{postfix}"

    def is_constraint_active_for_time_period(self, start_time: datetime, end_time: datetime | None = None) -> bool:

        if self.is_constraint_met(time=start_time):
            return False

        if end_time is None:
            end_time = DATETIME_MAX_UTC

        if self.current_start_of_constraint > end_time:
            return False

        if self.end_of_constraint == DATETIME_MAX_UTC:
            return True

        # only active if the constraint finish before the end of the given time period
        ret = start_time <= self.end_of_constraint <= end_time

        if ret is False and end_time != DATETIME_MAX_UTC:
            # it wil be active in the period anyway as it supports automatic consumtpion, and is not time sensitive
            if self.load.is_time_sensitive() is False and self.support_auto:
                ret = self.is_constraint_active_for_time_period(start_time)

        return ret



    def is_constraint_met(self, time:datetime | None=None, current_value=None) -> bool:
        """ is the constraint met in its current form? """

        if time is not None and self.always_end_at_end_of_constraint and time >= self.end_of_constraint:
            return True

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
        if self.last_value_update is None or time >= self.last_value_update:

            if self.last_value_update is None:
                self.last_value_update = time

            if self.first_value_update is None:
                self.first_value_update = time

            if self.last_value_change_update is None:
                self.last_value_change_update = time

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

            if self.current_value != value:
                self.last_value_change_update = time

            self.current_value = value
            if do_continue_constraint is False or self.is_constraint_met(time=time):
                return False
        return True


    def shallow_copy_for_budget_delta_quantity(self, added_budget_quantity: float) -> Self:
        # no deep copy to not copy inner objects
        out = copy.copy(self)
        if self.use_time_for_budgeting:
            out.current_value = self.convert_added_time_to_target_value(added_budget_quantity)
        else:
            out.current_value = self.convert_added_energy_to_target_value(added_budget_quantity)

        # if we are only simulating for budgeting, we don't want to always end at end of constraint
        out.always_end_at_end_of_constraint = False

        return out

    def get_quantity_to_be_added_for_budgeting(self) -> float:
        """ return the energy or time to be added to the load to meet the constraint"""
        if self.use_time_for_budgeting:
            return self.convert_target_value_to_time(self.target_value - self.current_value)
        else:
            return self.convert_target_value_to_energy(self.target_value  - self.current_value)

    def convert_added_energy_to_target_value(self, added_energy:float) -> float:
        # added energy can be negative, so we can reduce the current value
        return self.convert_energy_to_target_value(max(0.0, self.convert_target_value_to_energy(self.current_value) + added_energy))

    def convert_added_time_to_target_value(self, added_time:float) -> float:
        # added time can be negative, so we can reduce the current value
        return self.convert_time_to_target_value(max(0.0, self.convert_target_value_to_time(self.current_value) + added_time))

    @abstractmethod
    def compute_value(self, time: datetime) -> float | None:
        """ Compute the value of the constraint when the state change."""

    def best_duration_to_meet(self) -> timedelta:
        """ Return the best duration to meet the constraint."""
        seconds = self.convert_target_value_to_time(self.target_value - self.current_value)
        return timedelta(seconds=seconds)


    def get_delta_budget_quantity(self, power_w:float, duration_s:float) -> float:
        if self.use_time_for_budgeting:
            if power_w == 0:
                return 0.0
            if power_w < 0:
                return 0.0 - duration_s
            return duration_s
        else:
            return (power_w * duration_s) / 3600.0


    def best_duration_extension_to_push_constraint(self, time:datetime, end_constraint_min_tolerancy: timedelta) -> timedelta:

        duration_s = self.best_duration_to_meet() + end_constraint_min_tolerancy
        duration_s = max(timedelta(seconds=1200),
                         duration_s * (1.0 + self.pushed_count * 0.2))  # extend if we continue to push it

        return duration_s


    @abstractmethod
    def compute_best_period_repartition(self,
                                        do_use_available_power_only: bool,
                                        power_available_power: npt.NDArray[np.float64],
                                        power_slots_duration_s: npt.NDArray[np.float64],
                                        prices: npt.NDArray[np.float64],
                                        prices_ordered_values: list[float],
                                        time_slots: list[datetime],
                                        additional_available_energy_to_deplete:float=0.0,
                                        max_power_to_deplete:float=0.0
                                        ) -> tuple[
        Self, float,  bool, list[LoadCommand | None], npt.NDArray[np.float64], int, int, int, int, float]:
        """ Compute the best repartition of the constraint over the given period."""

    @abstractmethod
    def adapt_repartition(self,
                          first_slot:int,
                          last_slot:int,
                          energy_delta: float,
                          power_slots_duration_s: npt.NDArray[np.float64],
                          existing_commands: list[LoadCommand | None],
                          allow_change_state: bool,
                          time: datetime) -> tuple[Self, bool, bool, float, list[LoadCommand | None], npt.NDArray[np.float64]]:
        """ Adapt the power repartition of the constraint over the given period."""

class MultiStepsPowerLoadConstraint(LoadConstraint):

    def __init__(self,
                 power_steps: list[LoadCommand] = None,
                 power: float | None = None,
                 **kwargs):

        if power_steps is None and power is not None:
            power_steps = [copy_command(CMD_ON, power_consign=power)]

        self.update_power_steps(power_steps)

        super().__init__(**kwargs)

    def carry_info_from_other_constraint(self, other: LoadConstraint):
        super().carry_info_from_other_constraint(other)
        if isinstance(other, MultiStepsPowerLoadConstraint):
            self.update_power_steps(other._power_cmds)

    def update_power_steps(self, power_steps: list[LoadCommand]):
        self._power_cmds = power_steps
        self._power_sorted_cmds = [c for c in power_steps]
        self._power_sorted_cmds = sorted(self._power_sorted_cmds, key=lambda x: x.power_consign)
        self._power = self._power_sorted_cmds[-1].power_consign
        self._max_power = self._power_sorted_cmds[-1].power_consign
        self._min_power = self._power_sorted_cmds[0].power_consign

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["power_steps"] = [c.to_dict() for c in self._power_cmds]
        return data

    @classmethod
    def from_dict_to_kwargs(cls, data: dict) -> dict:
        res = super().from_dict_to_kwargs(data)

        res["power_steps"] = []
        for c in data["power_steps"]:
            if "phase_current" in c:
                # this is a legacy constraint command, remove the phase_current
                del c["phase_current"]
            if "load_param" in c:
                # this is a legacy constraint command, remove the load_param
                del c["load_param"]
            res["power_steps"].append(LoadCommand(**c))
        return res

    def convert_target_value_to_time(self, value: float) -> float:
        return self.load.efficiency_factor * ((3600.0 * (value)) / self._max_power)

    def convert_time_to_target_value(self, time_s: float) -> float:
        return (time_s * self._max_power) / (3600.0 * self.load.efficiency_factor)


    def compute_value(self, time: datetime) -> float | None:
        """ Compute the value of the constraint whenever it is called changed state or not """
        if self.load.current_command is None or self.load.current_command.is_off_or_idle():
            return None

        return (((time - self.last_value_update).total_seconds()) *
                self.load.current_command.power_consign / 3600.0) + self.current_value


    def _num_command_state_change_and_empty_inner_commands(self, out_commands: list[LoadCommand | None], first_slot:int, last_slot:int) -> tuple[int, list[list[int]], bool, LoadCommand | None]:

        # we give back the first empty command only if just before the first slot there was a non empty command, else we don't give it back, it is by design

        num = 0
        prev_cmd = None
        empty_cmds = []
        current_empty = None

        start_witch_switch = False
        start_cmd = None

        if len(out_commands) > 0:

            if first_slot == 0:
                if self.load and self.load.current_command:
                    start_cmd = self.load.current_command
            else:
                start_cmd = out_commands[first_slot - 1]

            if start_cmd and not start_cmd.is_off_or_idle():
                prev_cmd = start_cmd

            for i in range(first_slot, last_slot + 1):
                cmd = out_commands[i]
                if (cmd is None and prev_cmd is not None) or (cmd is not None and prev_cmd is None):
                    # this is a state change, so it we started with a non switch and empty : we won't output the first empty segment
                    if i == first_slot:
                        start_witch_switch = True
                    num += 1
                    if cmd is None:
                        current_empty = [i,i]
                    else:
                        if current_empty is not None:
                            current_empty[1] = i - 1
                            empty_cmds.append(current_empty)
                            current_empty = None

                prev_cmd = cmd

            # do not add the last one as empty : it is only for empty commands
            # if current_empty is not None:
            #    current_empty[1] = last_slot
            #    empty_cmds.append(current_empty)

        return num, empty_cmds, start_witch_switch, start_cmd




    def _replace_by_command_in_slots(self,
                                     out_commands,
                                     out_power,
                                     power_slots_duration_s,
                                     start_idx,
                                     end_idx,
                                     cmd_to_push,
                                     quantity_to_forbid_if_sign_changed=None):
        delta_quantity = 0.0
        durations_s = 0
        for i in range(start_idx, end_idx + 1):
            durations_s += power_slots_duration_s[i]
            new_power = out_power[i]
            new_d = 0.0

            if cmd_to_push is None:
                power_piloted_delta = self.load.get_possible_delta_power_from_piloted_devices_for_budget(slot_idx=i,
                                                                                                         add=False)
                prev_power = out_power[i]
                old_cmd = out_commands[i]
                if old_cmd is None:
                    old_cmd = self._power_sorted_cmds[0]
                old_cmd = copy_command(old_cmd)
                new_power = max(0, prev_power - old_cmd.power_consign - power_piloted_delta)
                new_d = self.get_delta_budget_quantity(old_cmd.power_consign, power_slots_duration_s[i])
            else:
                power_piloted_delta = self.load.get_possible_delta_power_from_piloted_devices_for_budget(slot_idx=i,
                                                                                                         add=True)
                new_power = cmd_to_push.power_consign + power_piloted_delta

                old_cmd = out_commands[i]
                if old_cmd is not None:
                    new_d = self.get_delta_budget_quantity(old_cmd.power_consign, power_slots_duration_s[i])

                new_d = new_d - self.get_delta_budget_quantity(cmd_to_push.power_consign, power_slots_duration_s[i])

            if quantity_to_forbid_if_sign_changed is None or (quantity_to_forbid_if_sign_changed + delta_quantity + new_d)*quantity_to_forbid_if_sign_changed >= 0.0:
                out_commands[i] = copy_command(cmd_to_push)
                out_power[i] = new_power
                delta_quantity += new_d
            else:
                return delta_quantity, durations_s, True

        return delta_quantity, durations_s, False


    def _adapt_commands(self, out_commands, out_power, power_slots_duration_s, budgeting_quantity_to_be_added, first_slot: int, last_slot: int) -> float:

        if self.load.num_max_on_off is not None and self.support_auto is False:
            num_command_state_change, inner_empty_cmds, start_witch_switch, start_cmd = self._num_command_state_change_and_empty_inner_commands(out_commands, first_slot, last_slot)
            num_allowed_switch = self.load.num_max_on_off - self.load.num_on_off
            num_removed = 0

            _LOGGER.info(f"Probe commands for on_off num/max: {self.load.num_on_off}/{self.load.num_max_on_off}")

            quantity_to_recover = 0.0

            if num_command_state_change > 1 and num_command_state_change > num_allowed_switch - 3:
                # too many state changes .... need to merge some commands
                # keep only the main one as it is solar only



                if start_witch_switch and len(inner_empty_cmds) > 0:
                    # we are starting with a switch...can we try to not do it?
                    if inner_empty_cmds[0][0] == first_slot:
                        # we start by a empty : do we need to stay on?
                        rge = [first_slot, inner_empty_cmds[0][1]]
                        if start_cmd is None:
                            start_cmd = self._power_sorted_cmds[0]
                        cmd_to_push = copy_command(start_cmd)
                    else:
                        # we start by a true command, do we want to continue to be stopped?
                        rge = [first_slot, max(first_slot, inner_empty_cmds[0][0]-1)]
                        cmd_to_push = None


                    durations_s = 0
                    for i in range(rge[0], rge[1]+1):
                        durations_s += power_slots_duration_s[i]

                    if durations_s <= SOLVER_STEP_S + 1:
                        # we can remove the first empty command, small enough

                        _LOGGER.info(f"_adapt_commands: removed start with switch command for {durations_s}s by {cmd_to_push}")
                        num_command_state_change -= 1
                        num_removed += 1

                        delta_quantity, durations_s, _ = self._replace_by_command_in_slots(out_commands,
                                                                                        out_power,
                                                                                        power_slots_duration_s,
                                                                                        rge[0],
                                                                                        rge[1],
                                                                                        cmd_to_push)
                        budgeting_quantity_to_be_added += delta_quantity

                        if cmd_to_push is None:
                            # extend the first inner empty command
                            inner_empty_cmds[0][0] = first_slot
                        else:
                            # ok done ... we have removed the first empty command
                            inner_empty_cmds.pop(0)


                        # delta_quantity is positive ... we have removed a command, we have more quantity to add
                        # if negative ... the contrary we have added a command we will need to remove some quantity
                        quantity_to_recover = delta_quantity

                        # switch has been removed at the start ....
                        start_witch_switch = False




            if num_command_state_change > 1 and num_command_state_change > num_allowed_switch - 3:
                do_for_num_switch_reduction = True
            else:
                do_for_num_switch_reduction = False

            sorted_by_size_empty_cmds = []
            for empty_cmd in inner_empty_cmds:
                empty_cmd.append(empty_cmd[1] +1 - empty_cmd[0])
                empty_cmd.append(False)
                if empty_cmd[2] == 1:
                    # only single slots
                    sorted_by_size_empty_cmds.append(empty_cmd)
                    empty_cmd[3] = True
                elif do_for_num_switch_reduction:
                    sorted_by_size_empty_cmds.append(empty_cmd)

            #remove the smallest holes first
            sorted_by_size_empty_cmds = sorted(inner_empty_cmds, key=lambda x: x[2])

            if len(sorted_by_size_empty_cmds) > 0:

                for empty_cmd in sorted_by_size_empty_cmds:
                    # if we start with a non switch and the first is empty : it means the previous was empty so replacing it will actually NOT remove any switch
                    if empty_cmd[0] == first_slot and start_witch_switch is False:
                        continue

                    # if we do have already removed enough switches ... stop
                    if empty_cmd[3] is False and num_command_state_change <= num_allowed_switch:
                        break

                    if empty_cmd[0] == first_slot:
                        start_witch_switch = False

                    num_removed += 1
                    num_command_state_change -= 2  # 2 changes removed

                    delta_quantity, durations_s, _ = self._replace_by_command_in_slots(out_commands,
                                                                                       out_power,
                                                                                       power_slots_duration_s,
                                                                                       empty_cmd[0],
                                                                                       empty_cmd[1],
                                                                                       self._power_sorted_cmds[0])

                    budgeting_quantity_to_be_added += delta_quantity
                    quantity_to_recover += delta_quantity

            if quantity_to_recover != 0.0:
                # quantity_to_recover is positive ... we have removed a command, we have more quantity to add
                # if negative ... the contrary we have added a command we will need to remove some quantity
                init_quantity_to_recover = quantity_to_recover

                def is_cmd_to_be_replaced(i):
                    if init_quantity_to_recover > 0.0:
                        # need to add some quantity
                        return out_commands[i] is None
                    else:
                        # need to remove some quantity
                        return out_commands[i] is not None

                if init_quantity_to_recover > 0.0:
                    cmd_to_push = self._power_sorted_cmds[-1]
                else:
                    cmd_to_push = None



                # well just a xor :)
                if start_witch_switch:
                    prev_cmd_is_to_be_replaced = not is_cmd_to_be_replaced(first_slot)
                else:
                    prev_cmd_is_to_be_replaced = is_cmd_to_be_replaced(first_slot)



                for i in range(first_slot, last_slot + 1):

                    limit_reached = False

                    # to not add any additional switch... only "flip" commands at transition points
                    if prev_cmd_is_to_be_replaced is False and is_cmd_to_be_replaced(i):

                        if init_quantity_to_recover < 0.0:
                            limit_to_not_change_sign = quantity_to_recover
                        else:
                            limit_to_not_change_sign = None

                        delta_quantity, durations_s, limit_reached = self._replace_by_command_in_slots( out_commands,
                                                                                                        out_power,
                                                                                                        power_slots_duration_s,
                                                                                                        i,
                                                                                                        i,
                                                                                                        cmd_to_push,
                                                                                                        quantity_to_forbid_if_sign_changed=limit_to_not_change_sign)
                        quantity_to_recover += delta_quantity
                        budgeting_quantity_to_be_added += delta_quantity

                    if quantity_to_recover*init_quantity_to_recover <= 0.0 or limit_reached:
                        # we changed sign between the two ...  or we have no more quantity to be added
                        break

                    # could have been changed by the flipping cmd code above
                    prev_cmd_is_to_be_replaced = is_cmd_to_be_replaced(i)


            _LOGGER.info(f"Adapted command for on_off num/max:{self.load.name} {self.load.num_on_off}/{self.load.num_max_on_off} Removed empty segments {num_removed}")

        return budgeting_quantity_to_be_added

    def adapt_power_steps_budgeting_low_level(self, slot_idx: int | None=None, existing_amps: list[float|int]| None=None , additional_command_piloted_power : float = 0.0) -> list[LoadCommand]:

        out_sorted_commands = []

        if self.load is None or self.load.father_device is None or self.load.father_device.available_amps_for_group is None or slot_idx is None:
            return self._power_sorted_cmds

        # first compute the minium available amps on the slots
        smaller_budget = self.load.father_device.available_amps_for_group[slot_idx]

        if existing_amps is not None and not is_amps_zero(existing_amps):
            smaller_budget = add_amps(smaller_budget, existing_amps)


        last_cmd_idx_ok = len(self._power_sorted_cmds)  - 1
        for i in range(len(self._power_sorted_cmds)-1, -1, -1):
            cmd = self._power_sorted_cmds[i]
            cmd_amps = self.load.get_phase_amps_from_power_for_budgeting(cmd.power_consign)
            if additional_command_piloted_power > 0.0:
                piloted_amps = self.load.get_phase_amps_from_power_for_piloted_budgeting(additional_command_piloted_power)
                cmd_amps = add_amps(cmd_amps, piloted_amps)
            if is_amps_greater(cmd_amps, smaller_budget):
                # not ok ... continue to remove commands
                last_cmd_idx_ok = i - 1
            else:
                break

        if last_cmd_idx_ok == len(self._power_sorted_cmds)  - 1:
            # all commands are ok, no need to adapt
            return self._power_sorted_cmds
        elif last_cmd_idx_ok >= 0:
            # we have some commands that are ok, so we can use them
            out_sorted_commands = self._power_sorted_cmds[:last_cmd_idx_ok + 1]

        return out_sorted_commands

    def adapt_power_steps_budgeting(self, slot_idx: int | None = None, commands: list[LoadCommand | None] | None =None,  for_add=True):

        if slot_idx is None:
            return self.adapt_power_steps_budgeting_low_level(), 0.0, 0.0

        existing_cmd_amps = None
        is_current_slot_empty = False

        if commands is None or commands[slot_idx] is None or commands[slot_idx].is_off_or_idle() or commands[slot_idx].power_consign == 0:
            if for_add:
                # current is empty and we will have : we may have a transition triggering piloted devices to add power
                power_piloted_delta = self.load.get_possible_delta_power_from_piloted_devices_for_budget(slot_idx=slot_idx,
                                                                                                         add=True)
            else:
                # we are already at bottom ... can't reduce
                power_piloted_delta = 0.0

            is_current_slot_empty = True
        else:
            existing_cmd_amps = self.load.get_phase_amps_from_power_for_budgeting(commands[slot_idx].power_consign)
            # see if the existing command was a "transition" triggering the piloted devices to add power
            power_piloted_delta = self.load.get_possible_delta_power_from_piloted_devices_for_budget(slot_idx=slot_idx, add=False)
            if power_piloted_delta > 0:
                piloted_cmd_amps = self.load.get_phase_amps_from_power_for_piloted_budgeting(power_piloted_delta)
                existing_cmd_amps = add_amps(existing_cmd_amps, piloted_cmd_amps)

        if for_add is False and is_current_slot_empty:
            power_sorted_cmds = []
            power_piloted_delta = 0.0
        else:
            power_sorted_cmds = self.adapt_power_steps_budgeting_low_level(slot_idx=slot_idx, existing_amps=existing_cmd_amps, additional_command_piloted_power=power_piloted_delta)
        
        return power_sorted_cmds, is_current_slot_empty, power_piloted_delta


    def adapt_repartition(self,
                          first_slot:int,
                          last_slot:int,
                          energy_delta: float,
                          power_slots_duration_s: npt.NDArray[np.float64],
                          existing_commands: list[LoadCommand | None],
                          allow_change_state: bool,
                          time: datetime) -> tuple[Self, bool, bool, float, list[LoadCommand | None], npt.NDArray[np.float64]]:

        """ Adapt the repartition of the constraint over the given period."""
        out_commands: list[LoadCommand | None] = [None] * len(power_slots_duration_s)
        out_delta_power = np.zeros(len(power_slots_duration_s), dtype=np.float64)

        log_msg = f"{self.name} from {first_slot} to {last_slot} ({int(np.sum(power_slots_duration_s[:first_slot]))}s to {int(np.sum(power_slots_duration_s[:last_slot]))}s)"
        if energy_delta >= 0.0:
            _LOGGER.info(
                f"adapt_repartition: for {self.name} consume more energy {energy_delta}Wh for {log_msg}")
            # start from the end as future is more uncertain, so take actions far from now, and we will see later if it was worth it
            # sorted_available_power = range(last_slot, first_slot - 1, -1)

            # well not sure of the statement above as the solver will assess stuff differently depending on what we consumed
            sorted_available_power = range(first_slot, last_slot + 1)
        else:
            _LOGGER.info(
                f"adapt_repartition: for {self.name} reclaim energy {energy_delta}Wh from {log_msg}")
            # try to reclaim energy to fill the battery as requested by the energy delta
            # start as soon as possible to get a chance to fill the battery as needed 
            sorted_available_power = range(first_slot, last_slot + 1)

        init_energy_delta = energy_delta

        num_changes = 0

        delta_budget_quantity = 0.0

        out_constraint = self

        is_current_constraint_met = self.is_constraint_met()

        num_non_zero_existing_commands = 0

        first_modified_slot = None

        start_solved_frontier = first_slot

        if init_energy_delta < 0.0 and self.is_mandatory is True:
            # do nothing
            _LOGGER.info(f"adapt_repartition: no adaptation for {self.name} as it is mandatory and we can't reduce its consumption")
        else:

            do_not_touch_commands = []
            default_cmd = None
            empty_cmd = None

            if self.support_auto:

                if init_energy_delta >= 0.0:
                    do_not_touch_commands = [CMD_AUTO_FROM_CONSIGN, CMD_AUTO_PRICE, CMD_AUTO_GREEN_CAP]
                    default_cmd = copy_command(CMD_AUTO_GREEN_CONSIGN)
                    empty_cmd = copy_command(CMD_AUTO_GREEN_ONLY)
                else:
                    do_not_touch_commands = [CMD_AUTO_FROM_CONSIGN, CMD_AUTO_PRICE]
                    default_cmd = copy_command(CMD_AUTO_GREEN_CAP)
                    empty_cmd = copy_command(CMD_AUTO_GREEN_CAP)

            for i in sorted_available_power:

                current_command_power = 0.0

                if existing_commands and existing_commands[i] is not None:

                    if existing_commands[i].is_like_one_of_cmds(do_not_touch_commands):
                        continue

                    current_command_power = existing_commands[i].power_consign

                if current_command_power > 0.0:
                    num_non_zero_existing_commands += 1

                j = None

                power_sorted_cmds, is_current_empty_command, possible_power_piloted_delta = self.adapt_power_steps_budgeting(slot_idx=i, commands=existing_commands, for_add=(init_energy_delta >= 0.0))

                if init_energy_delta >= 0.0:

                    if current_command_power == 0 and allow_change_state is False:
                        # we do not want to change the state of the load, so we cannot add energy
                        continue

                    if current_command_power == 0:
                        # j == 0 : lowest possible ON value
                        j = 0
                    else:
                        j = self._get_lower_consign_idx_for_power(power_sorted_cmds, current_command_power)

                        if j is None:
                            # lowest possible ON value
                            j = 0
                        elif j == len(power_sorted_cmds) - 1:
                            # we are already at the max power, we cannot add more, force to stay at this power
                            continue
                        else:
                            j += 1

                        if j is not None and power_sorted_cmds[j].power_consign <= current_command_power:
                           if j == len(power_sorted_cmds) - 1:
                               continue #do nothing, we are already at the max power
                           else:
                               j += 1

                else: # init_energy_delta < 0.0:

                    # for reduction: reduce strongly
                    if current_command_power == 0 or is_current_empty_command:
                        # we won't be able to reduce...cap at 0 to force stay this way
                        continue
                    else:
                        j = self._get_lower_consign_idx_for_power(power_sorted_cmds, current_command_power)

                        if (j is None or j == 0):
                            if allow_change_state is False:
                                # we won't be able to go below
                                continue
                            else:
                                j = -1
                        else:
                            # go to the minimum power load
                            j = 0

                        if j is not None and j >= 0 and power_sorted_cmds[j].power_consign >= current_command_power:
                           if j == 0:
                               if allow_change_state is False:
                                   continue #do nothing, we are already at the min power
                               else:
                                   j = -1
                           else:
                               j -= 1

                        if current_command_power == 0 and j < 0:
                            # we are already consuming nothing, we cannot reduce more
                            continue


                if j is not None:

                    if j >= 0:
                        base_cmd = power_sorted_cmds[j]
                    else:
                        # can't be init_energy_delta >= 0.0 in this case, so CAP only ....
                        if self.support_auto:
                            base_cmd = copy_command(CMD_AUTO_GREEN_CAP)
                        else:
                            base_cmd = copy_command(CMD_IDLE)

                    if not self.support_auto:
                        pass_through_command = base_cmd
                    else:
                        pass_through_command = copy_command(default_cmd)

                    delta_power = base_cmd.power_consign - current_command_power # should be same sign as init_energy_delta

                    if init_energy_delta >= 0.0:
                        if is_current_empty_command:
                            #we move from no client to add one: count the piloted_delta
                            delta_power += possible_power_piloted_delta
                    elif j < 0:
                        # we are reducing consumption to 0, so we can also reduce the piloted devices
                        if not is_current_empty_command:
                            delta_power -= possible_power_piloted_delta

                    if delta_power == 0.0:
                        continue # no change in power, nothing to do

                    d_energy = (delta_power * power_slots_duration_s[i]) / 3600.0 # should be same sign as init_energy_delta

                    d_budget_quantity = self.get_delta_budget_quantity(base_cmd.power_consign, power_slots_duration_s[i]) - self.get_delta_budget_quantity(current_command_power, power_slots_duration_s[i])

                    if (energy_delta - d_energy)* init_energy_delta <= 0:
                        # sign has changed ... we are no more in over consume or under consume
                        if init_energy_delta >= 0.0:
                            #we are overconsuming if we do that: don't allow this change
                            # for under consume it is ok to underconsume a bit more
                            break

                    if init_energy_delta >= 0.0:
                        # we want to consume a bit more if possible so we should add energy
                        if j < len(power_sorted_cmds) - 1:
                            new_base_cmd = power_sorted_cmds[j+1]
                            new_delta_power = new_base_cmd.power_consign - current_command_power

                            #if init_energy_delta >= 0.0:
                            if is_current_empty_command:
                                new_delta_power += possible_power_piloted_delta

                            new_d_energy = (new_delta_power* power_slots_duration_s[i]) / 3600.0
                            if (energy_delta - new_d_energy) * init_energy_delta >= 0:
                                j += 1
                                delta_power = new_delta_power
                                d_energy = new_d_energy
                                d_budget_quantity = self.get_delta_budget_quantity(new_base_cmd.power_consign, power_slots_duration_s[i]) - self.get_delta_budget_quantity(current_command_power, power_slots_duration_s[i])
                                base_cmd = new_base_cmd

                    out_commands[i] = copy_command_and_change_type(cmd=base_cmd,
                                                                   new_type=pass_through_command.command)
                    _LOGGER.info(
                        f"adapt_repartition: adapted {self.name} with command {out_commands[i]} / {i} effective in {int(np.sum(power_slots_duration_s[:i]))}s")

                    out_delta_power[i] = delta_power
                    delta_budget_quantity += d_budget_quantity
                    energy_delta -= d_energy

                    if first_modified_slot is None:
                        first_modified_slot = i
                    else:
                        first_modified_slot = min(first_modified_slot, i)

                    num_changes += 1

                    if init_energy_delta > 0.0:
                        if is_current_constraint_met:
                            # we should reclaim some power or time "from the future" to meet the constraint, we need to reclaim d_energy
                            budget_quantity_to_be_reclaimed = d_budget_quantity
                            has_reclaimed = False
                            for k in range(len(power_slots_duration_s) - 1, last_slot, -1 ):

                                if out_commands[k] is not None:
                                    # it has been reclaimed already
                                    continue

                                cmd = existing_commands[k]
                                if cmd is None or cmd.power_consign <= 0.0:
                                    continue

                                # we go to 0 : we reclaim all
                                possible_power_piloted_delta = self.load.get_possible_delta_power_from_piloted_devices_for_budget(slot_idx=k, add=False)

                                reclaimed_budget_quantity = self.get_delta_budget_quantity(cmd.power_consign, power_slots_duration_s[k])

                                do_reclaim = False
                                reclaim_cmd = None
                                if reclaimed_budget_quantity <= budget_quantity_to_be_reclaimed:
                                    do_reclaim = True
                                elif self.is_mandatory is False:
                                    do_reclaim = True
                                elif len(self._power_sorted_cmds) > 1:
                                    # we can use directly as we want to go lower than cmd.power_consign
                                    j = self._get_lower_consign_idx_for_power(self._power_sorted_cmds, cmd.power_consign)
                                    if j is not None and j > 0:
                                        # we won't "stop" the load, but we will reduce its power consumption
                                        possible_power_piloted_delta = 0.0
                                        reclaimed_budget_quantity = self.get_delta_budget_quantity(cmd.power_consign, power_slots_duration_s[k]) - self.get_delta_budget_quantity(self._power_sorted_cmds[0].power_consign, power_slots_duration_s[k])
                                        if reclaimed_budget_quantity <= budget_quantity_to_be_reclaimed:
                                            do_reclaim = True
                                            reclaim_cmd = copy_command(cmd, power_consign=self._power_sorted_cmds[0].power_consign)


                                if do_reclaim:
                                    has_reclaimed = True
                                    budget_quantity_to_be_reclaimed -= reclaimed_budget_quantity

                                    if reclaim_cmd is None:
                                        if self.support_auto:
                                            reclaim_cmd = copy_command(CMD_AUTO_GREEN_CAP)
                                        else:
                                            reclaim_cmd = copy_command(CMD_IDLE)

                                    out_delta_power[k] -= (cmd.power_consign - reclaim_cmd.power_consign + possible_power_piloted_delta)
                                    out_commands[k] = reclaim_cmd


                                    if budget_quantity_to_be_reclaimed < 0:
                                        break

                            if has_reclaimed:
                                # cool we do have successfully reclaimed some energy for a met constraint...
                                delta_budget_quantity -= (d_budget_quantity - budget_quantity_to_be_reclaimed)
                                _LOGGER.info(f"adapt_repartition: adapted {self.name} reclaimed met constraint {budget_quantity_to_be_reclaimed}Wh or s from the future")
                            else:
                                # the constraint was met and we can't reduce the futur: stop here
                                start_solved_frontier = i
                                break


                    if energy_delta * init_energy_delta <= 0.0:
                        # it means they are not of the same sign, we are done
                        start_solved_frontier = i
                        break

            out_constraint = self.shallow_copy_for_budget_delta_quantity(delta_budget_quantity)

            if num_non_zero_existing_commands == 0:
                _LOGGER.info(
                    f"adapt_repartition: for {self.name} THERE WERE NO NON ZERO EXISTING COMMANDS")

            if self.support_auto and num_changes > 0:

                # CAP the modified segment to what has been computed ...or force some consumption
                # we may in fact do that "all the time" but we are not sure of the futue, so limit
                # the forced caps to where they are important? use first_modified_slot instead?
                for i in range(start_solved_frontier, last_slot + 1):

                    if out_commands[i] is None:
                        if existing_commands[i] is None:
                            out_commands[i] = copy_command(empty_cmd)
                        else:
                            if existing_commands[i].is_like_one_of_cmds(do_not_touch_commands):
                                # we do not want to change the state of the load, so we cannot change energy
                                out_commands[i] = existing_commands[i]
                            else:
                                if existing_commands[i].power_consign == 0:
                                    out_commands[i] = copy_command(empty_cmd)
                                else:
                                    out_commands[i] = copy_command_and_change_type(cmd=existing_commands[i],
                                                                                new_type=default_cmd.command)

        return out_constraint, energy_delta * init_energy_delta <= 0.0, num_changes > 0, energy_delta, out_commands, out_delta_power


    def _get_lower_consign_idx_for_power(self, power_sorted_cmds:list[LoadCommand], power: float) -> int | None:

        j = None
        if power >= power_sorted_cmds[0].power_consign:
            j = 0
            while j < len(power_sorted_cmds) - 1 and power_sorted_cmds[j + 1].power_consign <= power:
                j += 1

        return j



    def compute_best_period_repartition(self,
                                        do_use_available_power_only: bool,
                                        power_available_power: npt.NDArray[np.float64],
                                        power_slots_duration_s: npt.NDArray[np.float64],
                                        prices: npt.NDArray[np.float64],
                                        prices_ordered_values: list[float],
                                        time_slots: list[datetime],
                                        additional_available_energy_to_deplete: float = 0.0,
                                        max_power_to_deplete: float = 0.0
                                        ) -> tuple[
        Self, bool, list[LoadCommand | None], npt.NDArray[np.float64], int, int, int, int, float]:

        # force to only use solar power for non mandatory constraints
        if self.is_mandatory is False:
            do_use_available_power_only = True

        remaining_additional_available_energy_to_deplete = additional_available_energy_to_deplete

        has_a_proper_end_time = False
        if self.end_of_constraint != DATETIME_MAX_UTC:
            if self.end_of_constraint <= time_slots[-1]:
                has_a_proper_end_time = True
            else:
                do_use_available_power_only = True

        initial_quantity_to_be_added = quantity_to_be_added = self.get_quantity_to_be_added_for_budgeting()

        out_power = np.zeros(len(power_available_power), dtype=np.float64)
        out_power_idxs = np.zeros(len(power_available_power), dtype=np.int64)
        out_commands: list[LoadCommand | None] = [None] * len(power_available_power)

        first_slot = 0
        last_slot = len(power_available_power) - 1

        max_idx_with_energy_impact = -1
        min_idx_with_energy_impact = len(power_available_power)


        default_cmd = None
        if self.support_auto:
            # fill all with green only command, to fill everything with the best power available
            if self.is_before_battery:
                default_cmd = CMD_AUTO_GREEN_ONLY
            else:
                default_cmd = CMD_AUTO_GREEN_CAP

        # find the constraint last slot
        if has_a_proper_end_time:
            # by construction the end of the constraints IS ending on a slot
            last_slot = bisect_left(time_slots, self.end_of_constraint)
            last_slot = max(0, last_slot - 1)  # -1 as the last_slot index is the index of the slot not the time anchor
            if last_slot >= len(power_available_power):
                last_slot = len(power_available_power) - 1

        if self.as_fast_as_possible:

            # fill with the best (more power) possible commands
            _LOGGER.info(
                f"compute_best_period_repartition: as fast as possible constraint {self.name} nrj/duration: {quantity_to_be_added}")

            has_a_cmd = False
            for i in range(first_slot, last_slot + 1):

                # we will add a command from 0 command so by construction power_piloted_delta will happen
                power_sorted_cmds, is_current_empty_command, power_piloted_delta = self.adapt_power_steps_budgeting(slot_idx=i, commands=None, for_add=True)

                if len(power_sorted_cmds) == 0:
                    continue

                has_a_cmd = True

                additional_available_power = 0.0
                if remaining_additional_available_energy_to_deplete < 0:
                    additional_available_power = min(max_power_to_deplete,
                                                     (0.0 - remaining_additional_available_energy_to_deplete) * 3600.0 /
                                                     power_slots_duration_s[i])

                if power_available_power[i] < 0:
                    additional_available_power -= power_available_power[i]

                if do_use_available_power_only:
                    if power_sorted_cmds[0].power_consign + power_piloted_delta > additional_available_power:
                        # even the smallest command is too big for the available power
                        continue

                if do_use_available_power_only:
                    # remove the power_piloted_delta from the available power, as it will be counted in the command we will add
                    j = self._get_lower_consign_idx_for_power(power_sorted_cmds, additional_available_power - power_piloted_delta)
                    if j is None:
                        as_fast_cmd_idx = 0
                    else:
                        as_fast_cmd_idx = j
                else:
                    as_fast_cmd_idx = len(power_sorted_cmds) - 1

                as_fast_power = power_sorted_cmds[as_fast_cmd_idx].power_consign
                as_fast_cmd_idx = as_fast_cmd_idx

                if self.support_auto:
                    as_fast_cmd = copy_command_and_change_type(cmd=power_sorted_cmds[as_fast_cmd_idx],
                                                               new_type=CMD_AUTO_FROM_CONSIGN.command)
                else:
                    as_fast_cmd = copy_command(power_sorted_cmds[as_fast_cmd_idx])


                out_commands[i] = as_fast_cmd
                out_power[i] = as_fast_power + power_piloted_delta

                quantity_to_be_added -= self.get_delta_budget_quantity(as_fast_cmd.power_consign, power_slots_duration_s[i])

                max_idx_with_energy_impact = max(max_idx_with_energy_impact, i)
                min_idx_with_energy_impact = min(min_idx_with_energy_impact, i)

                truly_consumed_power = power_available_power[i] + out_power[i]

                if truly_consumed_power >= 0:
                    remaining_additional_available_energy_to_deplete += (truly_consumed_power * power_slots_duration_s[i]) / 3600.0

                if quantity_to_be_added <= 0.0:
                    break

            if has_a_cmd is False:
                _LOGGER.error(f"compute_best_period_repartition: no power sorted commands in as fast as possible {self.name}")
                final_ret = False
            else:
                final_ret = quantity_to_be_added <= 0.0
        else:

            if self.current_start_of_constraint != DATETIME_MIN_UTC:
                first_slot = bisect_left(time_slots, self.current_start_of_constraint)
                if first_slot > 0 and time_slots[first_slot] > self.current_start_of_constraint:
                    # to get the proper slot we are in
                    first_slot -= 1

                if first_slot >= len(power_available_power):
                    first_slot = len(power_available_power) - 1

            # no need to reduce in case of OFF GRID Mode
            if has_a_proper_end_time and self.load.is_time_sensitive() and not self.load.is_off_grid():
                # we are in a time sensitive constraint, we will try to limit the number of slots to the last ones
                # a 6 hours windows, of course if the constraint is timed we get bigger

                best_s = max(3600.0*6, self.best_duration_to_meet().total_seconds()*1.5)
                start_reduction = self.end_of_constraint - timedelta(seconds=int(best_s) + 1)

                _LOGGER.info(
                    f"compute_best_period_repartition: reduce slots for time sensitive constraint {self.load.name} {self.end_of_constraint} to {start_reduction}")

                new_first_slots = bisect_left(time_slots, start_reduction)
                first_slot = max(first_slot, new_first_slots)

            if time_slots[min(last_slot+1, len(time_slots) - 1)] - time_slots[first_slot] < self.best_duration_to_meet():
                # expend a bit the window to have enough time to meet the constraint
                if first_slot > 0:
                    first_slot -= 1
                elif last_slot < len(power_available_power) - 1:
                    last_slot += 1

            if self.support_auto:
                # if we do support automatic filling: we will get all the available power, as soon as we get it:
                # so consume it greedily
                sorted_available_power = range(last_slot + 1 - first_slot)
            else:
                sub_power_available_power = power_available_power[first_slot:last_slot + 1]

                # We do want to order the available power in a way that we can use the best available power first
                # ...but not too far from now: else, especially in case of

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
                    _LOGGER.error(f"compute_best_period_repartition: ordered_exploration is not the same size as sub_power_available_power {len(sorted_available_power)} != {len(sub_power_available_power)}")
                    sorted_available_power = sub_power_available_power.argsort() # power_available_power negative value means free power


            has_a_cmd = False

            allocate_available_algos = ["available_only"]
            if remaining_additional_available_energy_to_deplete < 0:
                allocate_available_algos .append("available_and_to_deplete")

            # two passes: first only available power, second try to deplete battery if asked through the additional_available_energy_to_deplete
            for algo in allocate_available_algos:

                if algo == "available_only":
                    local_remaining_additional_available_energy_to_deplete = 0.0
                else:
                    local_remaining_additional_available_energy_to_deplete = remaining_additional_available_energy_to_deplete

                local_quantity_to_be_added = quantity_to_be_added

                for i_sorted in sorted_available_power:

                    i = i_sorted + first_slot
                    available_power = float(power_available_power[i])

                    # we will add a command from 0 command so by construction power_piloted_delta will happen
                    power_sorted_cmds, is_current_empty_command, power_piloted_delta = self.adapt_power_steps_budgeting(slot_idx=i, commands=None, for_add=True)

                    if len(power_sorted_cmds) == 0:
                        continue

                    has_a_cmd = True

                    additional_available_power = 0.0
                    if local_remaining_additional_available_energy_to_deplete < 0:
                        additional_available_power = min(max_power_to_deplete,
                                                         (0.0 - local_remaining_additional_available_energy_to_deplete) * 3600.0 /
                                                         float(power_slots_duration_s[i]))

                    # remove the power_piloted_delta from the available power, as it will be counted in the command we will add
                    j = self._get_lower_consign_idx_for_power(power_sorted_cmds, additional_available_power - available_power - power_piloted_delta)
                    if additional_available_power > 0.0:
                        j_naked = self._get_lower_consign_idx_for_power(power_sorted_cmds, 0.0 - available_power - power_piloted_delta)
                    else:
                        j_naked = j


                    if j is not None:

                        if self.support_auto:
                            if j == j_naked:
                                # no need to deplete the battery here
                                out_commands[i] = copy_command_and_change_type(cmd=power_sorted_cmds[j],
                                                                                new_type=default_cmd.command)
                            else:
                                # force a bit on consumption here to deplete battery
                                out_commands[i] = copy_command_and_change_type(cmd=power_sorted_cmds[j],
                                                                               new_type=CMD_AUTO_FROM_CONSIGN.command)
                        else:
                            out_commands[i] = copy_command(power_sorted_cmds[j])

                        out_power[i] = out_commands[i].power_consign + power_piloted_delta
                        out_power_idxs[i] = j

                        local_quantity_to_be_added -= self.get_delta_budget_quantity(out_commands[i].power_consign, float(power_slots_duration_s[i]))

                        max_idx_with_energy_impact = max(max_idx_with_energy_impact, i)
                        min_idx_with_energy_impact = min(min_idx_with_energy_impact, i)

                        truly_consumed_power = available_power + out_power[i]

                        if truly_consumed_power >= 0:
                            local_remaining_additional_available_energy_to_deplete += (truly_consumed_power *
                                                                                 power_slots_duration_s[i]) / 3600.0

                        if local_quantity_to_be_added <= 0.0:
                            break

                if algo == "available_only":
                    # we filled everything correctly!
                    if local_quantity_to_be_added <= 0.0 or has_a_cmd is False or len(allocate_available_algos) == 1:
                        quantity_to_be_added = local_quantity_to_be_added
                        break
                else:
                    remaining_additional_available_energy_to_deplete = local_remaining_additional_available_energy_to_deplete
                    quantity_to_be_added = local_quantity_to_be_added

            if has_a_cmd is False:
                _LOGGER.error(f"compute_best_period_repartition: no power sorted commands for green energy {self.name}")
                final_ret = False
            elif (quantity_to_be_added <= 0.0 or do_use_available_power_only):
                final_ret = quantity_to_be_added <= 0.0
            else:

                # pure solar (or pure solar plus allowed battery depletion)  was not enough, we will try to see if we can get more solar energy directly if price is better
                # but actually we should have depleted the battery on the "non controlled" part first, to see what is really possible,
                # this is checked in the charger budgeting algorithm, with self.home.battery_can_discharge() is False
                for i in range(first_slot, last_slot + 1):

                    if quantity_to_be_added <= 0.0:
                        break

                    if power_available_power[i] + out_power[i] < 0:
                        # there is still some solar to get
                        power_sorted_cmds, is_current_slot_empty, power_piloted_delta = self.adapt_power_steps_budgeting(slot_idx=i, commands=out_commands, for_add=True)

                        if len(power_sorted_cmds) == 0:
                            continue

                        power_to_add_idx = -1
                        init_power_to_add_idx = out_power_idxs[i]

                        if is_current_slot_empty or init_power_to_add_idx is None:
                            # no command yet, we can start from the lowest power command
                            is_current_slot_empty = True
                            init_power_to_add_idx = -1
                            power_to_add_idx = 0
                        elif out_power_idxs[i] < len(power_sorted_cmds) - 1:
                            power_to_add_idx = out_power_idxs[i] + 1

                        if power_to_add_idx > init_power_to_add_idx:
                            # all command in ECO mode now....
                            prev_quantity = 0.0
                            if not is_current_slot_empty:
                                if out_commands[i] :
                                    prev_quantity = self.get_delta_budget_quantity(float(out_commands[i].power_consign), power_slots_duration_s[i])

                            # check if the price of the extra energy we add is better than the best price we have
                            power_to_add: float = power_sorted_cmds[power_to_add_idx].power_consign + power_piloted_delta
                            # out_power[i] has already the prev_power_piloted_delta in it
                            energy_to_add: float = ((power_to_add - float(out_power[i]))* float(power_slots_duration_s[i])) / 3600.0
                            cost: float = ((power_to_add + float(power_available_power[i])) * float(power_slots_duration_s[i]) * float(prices[i])) / 3600.0
                            cost_per_watt_hour = cost / energy_to_add
                            quantity_to_add = self.get_delta_budget_quantity(power_sorted_cmds[power_to_add_idx].power_consign, power_slots_duration_s[i])

                            if cost_per_watt_hour > prices_ordered_values[0]:
                                # not interesting to add this energy
                                power_to_add_idx = -1

                            # if we have a lower cost per watt_hours, than the lowest available price, use this bit of extra solar production
                            if power_to_add_idx > init_power_to_add_idx:
                                if self.support_auto:
                                    new_cmd = copy_command_and_change_type(cmd=power_sorted_cmds[power_to_add_idx],
                                                                           new_type=CMD_AUTO_PRICE.command)
                                else:
                                    new_cmd = copy_command(power_sorted_cmds[power_to_add_idx])

                                _LOGGER.info(
                                        f"compute_best_period_repartition: price optimizer from {out_commands[i]} to {new_cmd}")

                                out_commands[i] = new_cmd

                                truly_consumed_power_before = power_available_power[i] + out_power[i]
                                if truly_consumed_power_before >= 0:
                                    remaining_additional_available_energy_to_deplete -= (truly_consumed_power_before *
                                                                                         power_slots_duration_s[
                                                                                             i]) / 3600.0

                                out_power[i] = power_to_add
                                out_power_idxs[i] = power_to_add_idx

                                truly_consumed_power = power_available_power[i] + out_power[i]
                                if truly_consumed_power >= 0:
                                    remaining_additional_available_energy_to_deplete += (truly_consumed_power *
                                                                                         power_slots_duration_s[
                                                                                             i]) / 3600.0


                                quantity_to_be_added += prev_quantity - quantity_to_add
                                max_idx_with_energy_impact = max(max_idx_with_energy_impact, i)
                                min_idx_with_energy_impact = min(min_idx_with_energy_impact, i)

                if quantity_to_be_added > 0.0:

                    has_a_cmd = False
                    for price in prices_ordered_values:

                        explore_range = range(last_slot, first_slot - 1, -1)

                        # will give back all commands, with no limits due to amps consumption
                        power_sorted_cmds, _, _ = self.adapt_power_steps_budgeting()

                        num_commands = len(power_sorted_cmds)

                        if  num_commands > 1:

                            #price_span_h = ((np.sum(power_slots_duration_s[first_slot:last_slot + 1],
                            #                        where=prices[first_slot:last_slot + 1] == price)) / 3600.0)

                            if self.load and self.load.current_command and self.load.current_command.is_like(CMD_AUTO_FROM_CONSIGN):
                                # we are already in auto consign mode for this load : we want to keep the continuity of the command
                                if first_slot == 0 and prices[first_slot] == price:
                                    # in this particular case : we will go from now to end to keep the continuity with the current command
                                    _LOGGER.info(
                                        f"compute_best_period_repartition:adapt constraint {self.name} to match current command {self.load.current_command}")
                                    # it will force the first slot to be a consign by construction
                                    explore_range = range(first_slot, last_slot + 1)

                            quantity_to_replace = 0.0

                            for i in explore_range:
                                if prices[i] == price and out_commands[i] is not None:
                                    quantity_to_replace += self.get_delta_budget_quantity(out_commands[i].power_consign, power_slots_duration_s[i])

                            # to try to fill as smoothly as possible: is it possible to fill the slot with the maximum power value?
                            fill_power_idx = 0
                            for fill_power_idx in range(num_commands):
                                filling_quantity_on_price = 0.0
                                for i in explore_range:
                                    if prices[i] == price:
                                        power_sorted_cmds, is_current_slot_empty, power_piloted_delta = self.adapt_power_steps_budgeting(slot_idx=i, commands=out_commands, for_add=True)
                                        if power_sorted_cmds is not None and len(power_sorted_cmds) > fill_power_idx:
                                            filling_quantity_on_price += self.get_delta_budget_quantity(power_sorted_cmds[fill_power_idx].power_consign, power_slots_duration_s[i])

                                if (quantity_to_be_added + quantity_to_replace) <= filling_quantity_on_price: #price_span_h * power_sorted_cmds[fill_power_idx].power_consign:
                                    break

                            # boost a bit to speed up a bit the filling, only if not off grid mode
                            if not self.load.is_off_grid():
                                fill_power_idx = min(fill_power_idx + 1, len(power_sorted_cmds) - 1)
                        else:
                            fill_power_idx = 0

                        # used to spread the commands : be a bit conservative on the spanning and use fill_power_aggressive_idx for the commands
                        # go reverse to respect the end constraint the best we can? or at the contrary fill it as soon as possible?
                        # may depend on the load type for a boiler you want to be closer, for a car it is more the asap? let's do reverse

                        for i in explore_range:

                            if prices[i] == price:

                                power_sorted_cmds, is_prev_empty, power_piloted_delta = self.adapt_power_steps_budgeting(slot_idx=i, commands=out_commands, for_add=True)

                                if len(power_sorted_cmds) == 0:
                                    continue

                                has_a_cmd = True

                                if is_prev_empty is False or out_power[i] > 0.0:
                                    # add back the previously removed energy or time
                                    if out_commands[i]:
                                        quantity_to_be_added += self.get_delta_budget_quantity(out_commands[i].power_consign, power_slots_duration_s[i])

                                    truly_consumed_power_before = power_available_power[i] + out_power[i]
                                    if truly_consumed_power_before >= 0:
                                        remaining_additional_available_energy_to_deplete -= (truly_consumed_power_before *power_slots_duration_s[i]) / 3600.0

                                # reduce command according to the amps budgeted consumption
                                power_cmd_idx = min(fill_power_idx, len(power_sorted_cmds) - 1)

                                if self.support_auto:
                                    price_cmd = copy_command_and_change_type(cmd=power_sorted_cmds[power_cmd_idx],
                                                                             new_type=CMD_AUTO_FROM_CONSIGN.command)
                                else:
                                    price_cmd = copy_command(power_sorted_cmds[power_cmd_idx])

                                out_commands[i] = price_cmd
                                out_power[i] = price_cmd.power_consign + power_piloted_delta
                                quantity_to_be_added -= self.get_delta_budget_quantity(price_cmd.power_consign, power_slots_duration_s[i])

                                truly_consumed_power = power_available_power[i] + out_power[i]
                                if truly_consumed_power >= 0:
                                    remaining_additional_available_energy_to_deplete += (truly_consumed_power *
                                                                                         power_slots_duration_s[
                                                                                             i]) / 3600.0

                                max_idx_with_energy_impact = max(max_idx_with_energy_impact, i)
                                min_idx_with_energy_impact = min(min_idx_with_energy_impact, i)

                                if quantity_to_be_added <= 0.0:
                                    break

                        if quantity_to_be_added <= 0.0:
                            break

                    if has_a_cmd is False:
                        _LOGGER.error(
                            f"compute_best_period_repartition: no power sorted commands for mandatory per price repartition {self.name}")

            quantity_to_be_added = self._adapt_commands(out_commands, out_power, power_slots_duration_s, quantity_to_be_added, first_slot, last_slot)
            final_ret = quantity_to_be_added <= 0.0


        if self.support_auto:
            # fill all with green only command, to fill everything with the best power available    
            for i in range(first_slot, last_slot + 1):
                if out_commands[i] is None:
                    out_commands[i] = copy_command(default_cmd)
                    out_power[i] = 0.0

        added_quantity = initial_quantity_to_be_added - quantity_to_be_added
        out_constraint = self.shallow_copy_for_budget_delta_quantity(added_quantity)

        _LOGGER.info(f"compute_best_period_repartition: {self.load.name} {added_quantity}Wh or s {self.get_readable_name_for_load()} use_available_only: {do_use_available_power_only} allocated is fulfilled: {final_ret}")

        if min_idx_with_energy_impact > max_idx_with_energy_impact or max_idx_with_energy_impact < 0 or min_idx_with_energy_impact >= len(power_available_power):
            min_idx_with_energy_impact = max_idx_with_energy_impact = -1

        return out_constraint, final_ret, out_commands, out_power, first_slot, last_slot, min_idx_with_energy_impact, max_idx_with_energy_impact, remaining_additional_available_energy_to_deplete


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
        target_string = f"{int(round(self._get_target_value_for_readable()))} %"
        return target_string

    def convert_target_value_to_energy(self, value: float) -> float:
        return self.load.efficiency_factor*(value * self.total_capacity_wh) / 100.0

    def convert_energy_to_target_value(self, energy: float) -> float:
        return (100.0 * max(0.0, energy)) / (self.total_capacity_wh*self.load.efficiency_factor)

    def convert_target_value_to_time(self, value: float) -> float:
        return self.load.efficiency_factor*((3600.0 * ((value * self.total_capacity_wh) / 100.0)) / self._max_power)

    def convert_time_to_target_value(self, time_s: float) -> float:
        return (time_s * self._max_power * 100.0 )/ (3600.0 * self.load.efficiency_factor * self.total_capacity_wh)

    def compute_value(self, time: datetime) -> float | None:
        """ Compute the value of the constraint whenever it is called changed state or not """
        if self.load.current_command is None or self.load.current_command.is_off_or_idle():
            return None

        return 100.0 * ((((time - self.last_value_update).total_seconds()) *
                         self.load.current_command.power_consign / 3600.0) /
                        self.total_capacity_wh) + self.current_value


class TimeBasedSimplePowerLoadConstraint(MultiStepsPowerLoadConstraint):


    def __init__(self, **kwargs):

        # in fact the code below is not usefull really it extend artificially the need of the constraint, and it is a bad behavior
        # I leave it here as a trace of the thought process

        # exact_start_end_coverage = kwargs.get("exact_start_end_coverage", False)
        # if exact_start_end_coverage:
        #     kwargs["always_end_at_end_of_constraint"] = True
        #     start_of_constraint = kwargs.get("start_of_constraint", None)
        #     if start_of_constraint is not None:
        #         end_of_constraint = kwargs.get("end_of_constraint", None)
        #         if end_of_constraint is not None:
        #             duration_s = (end_of_constraint - start_of_constraint).total_seconds()
        #             # force complete time coverage, has to exceed the end-constraint - start time by at least one solver step
        #             kwargs["target_value"] = (duration_s//SOLVER_STEP_S)*SOLVER_STEP_S + SOLVER_STEP_S + 2

        # well in  fact for TimeBased do that alsways ...
        kwargs["always_end_at_end_of_constraint"] = True

        super().__init__(**kwargs)

    @property
    def use_time_for_budgeting(self):
        return True

    def _get_readable_target_value_string(self) -> str:
        target_value = self._get_target_value_for_readable()
        minutes = int((target_value % 3600) / 60)
        hours = int(target_value / 3600)
        target_string = f"{int(target_value)}s"
        if target_value >= 4 * 3600:
            target_string = f"{hours}h"
        elif target_value >= 3600:
            if minutes > 0:
                target_string = f"{hours}h{minutes}mn"
            else:
                target_string = f"{hours}h"
        elif target_value >= 60:
            target_string = f"{int(target_value / 60)}mn"
        return target_string

    def best_duration_extension_to_push_constraint(self, time: datetime, end_constraint_min_tolerancy: timedelta) -> timedelta:
        return self.best_duration_to_meet()

    def is_constraint_met(self, time:datetime|None = None, current_value=None) -> bool:

        if time is not None and self.always_end_at_end_of_constraint and  time >= self.end_of_constraint:
            return True

        if current_value is None:
            current_value = self.current_value

        if self.target_value is None:
            return False

        if current_value >= (0.995*self.target_value): #0.5% tolerance
            return True

        if time is not None and self.end_of_constraint <= time and current_value >= (0.9*self.target_value): # 10% tolerance if close to end
            return True

        return False

    def convert_target_value_to_time(self, value: float) -> float:
        return value

    def convert_time_to_target_value(self, time_s: float) -> float:
        return time_s

    def compute_value(self, time: datetime) -> float | None:
        """ Compute the value of the constraint whenever it is called changed state or not,
        hence use the old state and the last value change to add the consummed energy """
        if self.load.current_command is not None and not self.load.current_command.is_off_or_idle():
            return (time - self.last_value_update).total_seconds() + self.current_value
        else:
            return None

    def convert_target_value_to_energy(self, value: float) -> float:
        return self.load.efficiency_factor*(self._power * value) / 3600.0

    def convert_energy_to_target_value(self, energy: float) -> float:
        return (max(0.0, energy) * 3600.0) / (self._power * self.load.efficiency_factor)


DATETIME_MAX_UTC = datetime.max.replace(tzinfo=pytz.UTC)
DATETIME_MIN_UTC = datetime.min.replace(tzinfo=pytz.UTC)
