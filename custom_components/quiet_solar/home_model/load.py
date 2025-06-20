import copy
import logging
import math
from bisect import bisect_left
from datetime import datetime, timedelta
from collections.abc import Generator
from operator import itemgetter
import random

import pytz

from .commands import LoadCommand, copy_command, CMD_OFF, CMD_IDLE, CMD_ON
from .constraints import LoadConstraint, DATETIME_MAX_UTC, DATETIME_MIN_UTC

from typing import TYPE_CHECKING, Any, Mapping, Callable, Awaitable

from ..const import CONF_POWER, CONF_SWITCH, CONF_LOAD_IS_BOOST_ONLY, CONF_MOBILE_APP, CONF_MOBILE_APP_NOTHING, \
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE, CONF_DEVICE_EFFICIENCY, DEVICE_CHANGE_CONSTRAINT, \
    DEVICE_CHANGE_CONSTRAINT_COMPLETED, CONF_IS_3P, CONF_MONO_PHASE, CONF_DEVICE_DYNAMIC_GROUP_NAME, CONF_NUM_MAX_ON_OFF

import slugify

if TYPE_CHECKING:
    import QSDynamicGroup

NUM_MAX_INVALID_PROBES_COMMANDS = 3


_LOGGER = logging.getLogger(__name__)

def is_amps_zero(amps: list[float | int]) -> bool:
    if amps is None:
        return True

    for a in amps:
        if a != 0.0:
            return False

    return True

def is_amps_greater(left_amps: list[float | int], right_amps: list[float | int]):
    for i in range(3):
        if left_amps[i] > right_amps[i]:
            return True
    return False


def add_amps(left_amps: list[float | int], right_amps: list[float | int]) -> list[float | int]:
    if left_amps is None and right_amps is None:
        return [0.0, 0.0, 0.0]
    elif left_amps is None:
        return copy.copy(right_amps)
    elif right_amps is None:
        return copy.copy(left_amps)

    adds = [left_amps[i] + right_amps[i] for i in range(3)]
    return adds


def diff_amps(left_amps: list[float | int], right_amps: list[float | int]) -> list[float | int]:
    if left_amps is None or right_amps is None:
        return [0.0, 0.0, 0.0]

    diff = [left_amps[i] - right_amps[i] for i in range(3)]
    return diff

def min_amps(left_amps: list[float | int], right_amps: list[float | int]) -> list[float | int]:
    if left_amps is None and right_amps is None:
        return [0.0, 0.0, 0.0]
    elif left_amps is None:
        return copy.copy(right_amps)
    elif right_amps is None:
        return copy.copy(left_amps)

    mins = [min(left_amps[i], right_amps[i]) for i in range(3)]
    return mins

def max_amps(left_amps: list[float | int], right_amps: list[float | int]) -> list[float | int]:
    if left_amps is None and right_amps is None:
        return [0.0, 0.0, 0.0]
    elif left_amps is None:
        return copy.copy(right_amps)
    elif right_amps is None:
        return copy.copy(left_amps)

    maxs = [max(left_amps[i], right_amps[i]) for i in range(3)]
    return maxs


class AbstractDevice(object):
    def __init__(self, name:str, device_type:str|None = None, **kwargs):
        super().__init__()
        self._enabled = True
        self.efficiency = float(min(kwargs.pop(CONF_DEVICE_EFFICIENCY, 100.0), 100.0))
        self._device_is_3p_conf = kwargs.pop(CONF_IS_3P, False)
        self.dynamic_group_name = kwargs.pop(CONF_DEVICE_DYNAMIC_GROUP_NAME, None)
        self._mono_phase_conf = kwargs.pop(CONF_MONO_PHASE, None)
        if self._mono_phase_conf is None:
            # at random allocate phase on 0, 1, or 2
            self._mono_phase_default = random.randint(0,2)
        else:
            self._mono_phase_default = int(self._mono_phase_conf) - 1

        self.name = name
        self._device_type = device_type
        self.device_id = f"qs_{slugify.slugify(name, separator="_")}_{self.device_type}"
        self.home = kwargs.pop("home", None)

        self._local_reset()

        self._ack_command(None, None)

        self.num_max_on_off = kwargs.pop(CONF_NUM_MAX_ON_OFF, None)
        if self.num_max_on_off is not None:
            self.num_max_on_off = int(self.num_max_on_off)
            if self.num_max_on_off % 2 == 1:
                self.num_max_on_off += 1

        self.father_device : QSDynamicGroup = self.home

    def _local_reset(self):
        _LOGGER.info(f"Reset device {self.name}")
        self._constraints: list[LoadConstraint | None] = []
        self.current_command : LoadCommand | None = None
        self.prev_command: LoadCommand | None = None
        self.running_command : LoadCommand | None = None # a command that has been launched but not yet finished, wait for its resolution
        self._stacked_command: LoadCommand | None = None # a command (keep only the last one) that has been pushed to be executed later when running command is free
        self.running_command_first_launch: datetime | None = None
        self.running_command_last_launch: datetime | None = None
        self.running_command_num_relaunch : int = 0
        self.running_command_num_relaunch_after_invalid: int = 0
        self.num_on_off : int = 0
        self.device_phase_amps_budget : list[float|int] | None = None
        self.to_budget: bool = False
        self.reset_daily_load_datas()

    # for class overcharging reset
    def reset(self):
        self._local_reset()

    @property
    def qs_enable_device(self) -> bool:
        return self._enabled

    @qs_enable_device.setter
    def qs_enable_device(self, enabled:bool):
        if enabled != self._enabled:
            self._enabled = enabled
            if enabled:
                self.reset()
                self.home.remove_device(self)

            if hasattr(self, "_exposed_entities"):
                time = datetime.now(pytz.utc)
                for ha_object in self._exposed_entities:
                    ha_object.async_update_callback(time)


    def allocate_phase_amps_budget(self, time:datetime, from_father_budget: list[float|int]|None) -> list[float|int]:

        if self.qs_enable_device is False:
            return [0.0, 0.0, 0.0]

        if from_father_budget is None:
            allocate_budget, _ = self.get_min_max_phase_amps_for_budgeting()
        else:
            allocate_budget = from_father_budget

        self.device_phase_amps_budget = copy.copy(allocate_budget)

        _LOGGER.info(f"allocate_phase_amps_budget for load {self.name} from_father_budget {from_father_budget} => {self.device_phase_amps_budget}")

        return self.device_phase_amps_budget

    @property
    def device_type(self):
        if self._device_type is None:
            return self.__class__.__name__
        return self._device_type



    @property
    def physical_num_phases(self) -> int:
        if self._device_is_3p_conf:
            return 3
        return 1

    @property
    def physical_3p(self) -> bool:
        return self.physical_num_phases == 3

    @property
    def current_num_phases(self) -> int:
        return self.physical_num_phases

    @property
    def current_3p(self) -> bool:
        return self.current_num_phases == 3

    def can_do_3_to_1_phase_switch(self):
        return False

    @property
    def mono_phase_index(self) -> int:

        if self._mono_phase_conf is not None:
            return self._mono_phase_default

        if self.father_device is not None and self.father_device != self.home and not self.father_device.physical_3p:
            return self.father_device.mono_phase_index

        return self._mono_phase_default

    def update_amps_with_delta(self, from_amps:list[float|int], delta:int|float, is_3p:bool) -> list[float|int]:
        amps = copy.copy(from_amps)
        if is_3p is False:
            amps[self.mono_phase_index] += delta
        else:
            amps[0] += delta
            amps[1] += delta
            amps[2] += delta
        return amps

    def __repr__(self):
        return self.device_id

    #it is a property as it has to be overchargeable (ex: charger for its car)
    @property
    def efficiency_factor(self):
        return 100.0 / self.efficiency

    def get_to_be_saved_info(self) -> dict:
        return {"num_on_off": self.num_on_off}

    def reset_daily_load_datas(self, time:datetime | None = None):
        self.num_on_off = 0


    def get_min_max_power(self) -> (float, float):
        return 0.0, 0.0

    def get_min_max_phase_amps_for_budgeting(self) -> ( list[float|int],  list[float|int]):
        min_p, max_p = self.get_min_max_power()
        return self.get_phase_amps_from_power_for_budgeting(min_p), self.get_phase_amps_from_power_for_budgeting(max_p)

    def get_evaluated_needed_phase_amps_for_budgeting(self, time: datetime) -> list[float|int]:
        return [0.0, 0.0, 0.0]


    def get_phase_amps_from_power_for_budgeting(self, power:float) -> list[float | int]:
        return self.get_phase_amps_from_power(power, is_3p=self.physical_3p)

    def get_phase_amps_from_power(self, power:float, is_3p=False) -> list[float | int]:

        if is_3p:
            power = power / 3.0
        p = power / self.home.voltage
        if is_3p:
            return [p, p, p]
        else:
            ret = [0, 0, 0]
            ret[self.mono_phase_index] = p
            return ret


    def get_current_active_constraint(self, time:datetime) -> LoadConstraint | None:
        if self.qs_enable_device is False:
            self._constraints = []

        if not self._constraints:
            self._constraints = []
        for c in self._constraints:
            if c.is_constraint_active_for_time_period(time):
                return c
        return None

    def is_as_fast_as_possible_constraint_active_for_budgeting(self, time:datetime) -> bool:
        if self.qs_enable_device is False:
            self._constraints = []

        if self.to_budget is False:
            return False

        if not self._constraints:
            self._constraints = []
        for c in self._constraints:
            if c.is_constraint_active_for_time_period(time) and c.as_fast_as_possible:
                return True
        return False

    def is_consumption_optional_for_budgeting(self, time:datetime) -> bool:
        ct = self.get_current_active_constraint(time)
        if ct is not None and ct.end_of_constraint != DATETIME_MAX_UTC:
            return False
        return True

    def _ack_command(self, time:datetime|None,  command:LoadCommand|None):

        if command is not None:
            _LOGGER.info(f"ack command {command.command} for load {self.name}")
        else:
            _LOGGER.info(f"ack command None for load {self.name}")

        self.prev_command = self.current_command
        self.current_command = command
        self.running_command = None
        self.running_command_num_relaunch = 0
        self.running_command_num_relaunch_after_invalid = 0
        self.running_command_first_launch = None
        self.running_command_last_launch = None

        if command is not None and time is not None and self.prev_command is not None:
            do_count = False
            if command.is_off_or_idle() and not self.prev_command.is_off_or_idle():
                do_count = True
            elif not command.is_off_or_idle() and self.prev_command.is_off_or_idle():
                do_count = True

            if do_count:
                self.num_on_off += 1
                _LOGGER.info(f"Change load: {self.name} state increment num_on_off:{self.num_on_off} ({command.command})")

    def is_load_has_a_command_now_or_coming(self, time:datetime) -> bool:
        if self.qs_enable_device is False:
            return False

        if self.current_command is not None:
            return True
        if self.running_command is not None:
            return True
        if self._stacked_command is not None:
            return True
        return False

    async def launch_qs_command_back(self, time: datetime, ctxt="NO CTXT"):
        current = self.running_command
        if current is None:
            current = self.current_command
        if current is None:
            current = CMD_IDLE

        self.current_command = None
        self.running_command = None
        await self.launch_command(time=time, command=current, ctxt=ctxt)

    async def launch_command(self, time:datetime, command: LoadCommand, ctxt="NO CTXT"):
        if self.qs_enable_device is False:
            return

        command = copy_command(command)

        if self.running_command is not None:
            # another command has been launched, stack this one (we replace the previous stacked one)
            self._stacked_command = command
            _LOGGER.info(f"launch_command: stack command {command} for this load {self.name}), ctxt: {ctxt}")
            return

        # there is no running : whatever we will not execute the stacked one but only the last one
        self._stacked_command = None

        if self.current_command is not None and self.current_command == command:
            # We kill the stacked one and keep the current one like the choice above
            self.current_command = command # needed as command == may have been overcharged to not test everything
            return


        self.running_command = command
        self.running_command_first_launch = time
        self.running_command_last_launch = time

        _LOGGER.info(f"launch_command: {command} for this load {self.name}), ctxt: {ctxt}")

        is_command_set = await self.probe_if_command_set(time, self.running_command)
        if is_command_set is True:
            _LOGGER.info(f"launch_command: Command already set {command} for this load {self.name}, ctxt: {ctxt}")
        else:
            try:
                is_command_set = await self.execute_command(time, command)
            except Exception as err:
                _LOGGER.error(f"Error while executing command {command.command} for load {self.name} : {err}, ctxt: {ctxt}", exc_info=err)
                is_command_set = None

        if is_command_set is None:
            # hum we may have an impossibility to launch this command
            _LOGGER.info(f"launch_command: Impossible to launch this command {command.command} on this load {self.name}, ctxt: {ctxt}")
        elif is_command_set is True:
            _LOGGER.info(f"launch_command: ack command {command} for this load {self.name}), ctxt: {ctxt}")
            self._ack_command(time, self.running_command)

        return

    def is_load_command_set(self, time:datetime):
        if self.qs_enable_device is False:
            return True

        return self.running_command is None and self.current_command is not None

    async def check_commands(self, time: datetime) -> timedelta:

        res = timedelta(seconds=0)

        if self.qs_enable_device is False:
            return res

        if self.running_command is not None:
            _LOGGER.info(
                f"check command {self.running_command.command} for this load {self.name}) (#{self.running_command_num_relaunch_after_invalid})")

            is_command_set = await self.probe_if_command_set(time, self.running_command)
            if is_command_set is None:
                # impossible to run this command for this load ...
                self.running_command_num_relaunch_after_invalid += 1
                _LOGGER.info(f"impossible to check command {self.running_command.command} for this load {self.name}) (#{self.running_command_num_relaunch_after_invalid})")
                if self.running_command_num_relaunch_after_invalid >= NUM_MAX_INVALID_PROBES_COMMANDS:
                    # will kill completely the command ....
                    self._ack_command(time, None)

            if is_command_set is True:
                self._ack_command(time, self.running_command)
            elif self.running_command_last_launch is not None:
                res = time - self.running_command_last_launch


        if self.running_command is None and self._stacked_command is not None:
            await self.launch_command(time, self._stacked_command, ctxt="check_commands, launch stacked command")

        return res

    async def force_relaunch_command(self, time: datetime):
        if self.qs_enable_device is False:
            self.running_command = None

        if self.running_command is not None:
            _LOGGER.info(f"force launch command {self.running_command.command} for this load {self.name} (#{self.running_command_num_relaunch})")
            self.running_command_num_relaunch += 1
            is_command_set = await self.execute_command(time, self.running_command)
            self.running_command_last_launch = time
            if is_command_set is None:
                _LOGGER.info(f"impossible to force command {self.running_command.command} for this load {self.name})")
            elif is_command_set is True:
                self._ack_command(time, self.running_command)
            else:
                await self.check_commands(time)

    async def execute_command(self, time: datetime, command: LoadCommand) -> bool | None:
        print(f"Executing command {command}")
        return False

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool | None:
        return True



class AbstractLoad(AbstractDevice):

    def __init__(self, **kwargs):
        self.switch_entity = kwargs.pop(CONF_SWITCH, None)
        self.power_use = kwargs.pop(CONF_POWER, None)
        self.load_is_auto_to_be_boosted = kwargs.pop(CONF_LOAD_IS_BOOST_ONLY, False)
        self.external_user_initiated_state: str | None = None
        self.external_user_initiated_state_time : datetime | None = None
        self.asked_for_reset_user_initiated_state_time : datetime | None = None


        super().__init__(**kwargs)


        self._last_completed_constraint: LoadConstraint | None = None
        self.current_constraint_current_value: float | None = None
        self.current_constraint_current_energy: float | None = None
        self.current_constraint_current_percent_completion: float | None = None
        self._externally_initialized_constraints = False

        self.qs_best_effort_green_only = False

        self._last_constraint_update: datetime|None = None
        self._last_pushed_end_constraint_from_agenda = None
        self._last_hash_state = None

        self.is_load_time_sensitive = False

    def get_override_state(self):
        if self.asked_for_reset_user_initiated_state_time is not None:
            return "ASKED FOR RESET OVERRIDE"
        if self.external_user_initiated_state is None:
            return "NO OVERRIDE"
        return f"Override: {self.external_user_initiated_state}"

    def is_time_sensitive(self):

        if self.load_is_auto_to_be_boosted or self.qs_best_effort_green_only:
            return False

        return self.is_load_time_sensitive

    def get_for_solver_constraints(self, start_time:datetime, end_time:datetime) -> list[Any]:
        if self.qs_enable_device is False:
            self._constraints = []

        res = []

        for c in self._constraints:
            if c.is_constraint_active_for_time_period(start_time, end_time):
                res.append(c)

        return res


    def is_consumption_optional_for_budgeting(self, time:datetime) -> bool:
        if self.load_is_auto_to_be_boosted or  self.qs_best_effort_green_only:
            return True
        return super().is_consumption_optional_for_budgeting(time)

    def get_min_max_power(self) -> (float, float):
        if self.power_use is None:
            return 0.0, 0.0
        return self.power_use, self.power_use

    def get_evaluated_needed_phase_amps_for_budgeting(self, time: datetime) -> list[float|int]:
        device_needed_amp = [0.0, 0.0, 0.0]
        if self.to_budget:
            ct = self.get_current_active_constraint(time)
            min_a, max_a = self.get_min_max_phase_amps_for_budgeting()
            if ct:
                load_power = ct.evaluate_needed_mean_power(time)
                device_needed_amp = self.get_phase_amps_from_power_for_budgeting(load_power)
                ret = copy.copy(device_needed_amp)
                for i in range(3):
                    if ret[i] < min_a[i]:
                        ret[i] = min_a[i]
                    elif ret[i] > max_a[i]:
                        ret[i] = max_a[i]
                device_needed_amp = ret
            else:
                device_needed_amp = copy.copy(max_a)

        return device_needed_amp

    def support_green_only_switch(self) -> bool:
        return False

    def support_user_override(self) -> bool:
        return False

    def push_unique_and_current_end_of_constraint_from_agenda(self, time: datetime, new_ct: LoadConstraint):

        new_end_constraint = new_ct.end_of_constraint

        if new_end_constraint is None or new_end_constraint == DATETIME_MAX_UTC or new_end_constraint == DATETIME_MIN_UTC:
            return False

        if self._last_pushed_end_constraint_from_agenda is None:
            self._last_pushed_end_constraint_from_agenda = new_end_constraint
        else:
            # if the agenda has changed ... we should remove an existing uneeded constraint
            if self._last_pushed_end_constraint_from_agenda != new_end_constraint:
                for i, ct in enumerate(self._constraints):
                    if (isinstance(ct, new_ct.__class__)
                            and ct.type == new_ct.type
                            and ct.end_of_constraint == self._last_pushed_end_constraint_from_agenda):
                        self._constraints[i] = None
                        break

                self._constraints = [c for c in self._constraints if c is not None]

        res = self.push_live_constraint(time, new_ct)
        self._last_pushed_end_constraint_from_agenda = new_end_constraint

        return res

    def get_power_from_switch_state(self, state : str | None) -> float | None:
        if state is None:
            return None
        if state == "on":
            return self.power_use
        else:
            return 0.0

    async def do_run_check_load_activity_and_constraints(self, time: datetime)-> bool:
        if self._externally_initialized_constraints is False:
            return False
        return  await self.check_load_activity_and_constraints(time)



    def load_constraints_from_storage(self, time:datetime, constraints_dicts: list[dict], stored_executed: dict | None, stored_load_info: dict | None):
        self.reset()
        for c_dict in constraints_dicts:
            cs_load = LoadConstraint.new_from_saved_dict(time, self, c_dict)
            if cs_load is not None:
                # only restore constraints that can still be active
                if cs_load.is_constraint_active_for_time_period(time):
                    self.push_live_constraint(time, cs_load)

        if stored_executed is not None:
            self._last_completed_constraint = LoadConstraint.new_from_saved_dict(time, self, stored_executed)
        else:
            self._last_completed_constraint = None

        if stored_load_info:
            self.num_on_off =  stored_load_info.get("num_on_off", 0)

            if self.num_on_off > 0 and self.num_on_off % 2 == 1:
                # because of a reboot we may need a bit more ...
                self.num_on_off -= 1

            if self.num_max_on_off is not None:
                if self.num_max_on_off - self.num_on_off <= 2:
                    self.num_on_off = self.num_max_on_off - 2

        self._externally_initialized_constraints = True

    async def check_load_activity_and_constraints(self, time: datetime) -> bool:
        return False

    async def do_probe_state_change(self, time: datetime):

        new_hash = self.get_active_state_hash(time)

        if new_hash is not None:
            # do not notify just after a reset (self._last_hash_state None)
            if (self._last_hash_state is not None and self._last_hash_state != new_hash):
                _LOGGER.info(f"Hash state change for load {self.name} from {self._last_hash_state} to {new_hash}")
                await self.on_device_state_change(time, DEVICE_CHANGE_CONSTRAINT)

            self._last_hash_state = new_hash


    async def on_device_state_change(self, time: datetime, device_change_type:str):
        pass


    def is_cmd_compatible_with_load_budget(self, cmd : LoadCommand) -> bool:

        if  self.device_phase_amps_budget is None:
            return True

        if cmd.power_consign is not None and cmd.power_consign > 0:
            # it is for budgeting device_phase_amps_budget has been comouted with "static" 3p devices
            # compare only budgeting value together
            amps = self.get_phase_amps_from_power_for_budgeting(cmd.power_consign)
            # shave by one amp to allow a bit more commands
            amps = [ max(0, a-1) for a in amps]
            return not is_amps_greater(amps, self.device_phase_amps_budget)

        return True


    def get_update_value_callback_for_constraint_class(self, constraint:LoadConstraint) -> Callable[[LoadConstraint, datetime], Awaitable[tuple[float | None, bool]]] | None:
        return None

    def is_load_active(self, time: datetime):
        if not self._constraints:
            return False
        return True

    def reset(self):
        _LOGGER.info(f"Reset load {self.name}")
        super().reset()
        self._last_completed_constraint = None
        self._last_pushed_end_constraint_from_agenda = None

    async def ack_completed_constraint(self, time:datetime, constraint:LoadConstraint|None):
        self._last_completed_constraint = constraint
        await self.on_device_state_change(time, DEVICE_CHANGE_CONSTRAINT_COMPLETED)


    def get_active_readable_name(self, time:datetime, filter_for_human_notification=False) -> str | None:

        current_constraint = self.get_current_active_constraint(time)

        new_val = None

        if current_constraint is None:
            if filter_for_human_notification is False:
                if self._last_completed_constraint is not None:
                    new_val = "COMPLETED: " + self._last_completed_constraint.get_readable_name_for_load()
                else:
                    new_val = "NOTHING PLANNED"

        else:
            new_val = current_constraint.get_readable_name_for_load()

        return new_val

    def get_active_state_hash(self, time:datetime) -> str:

        current_constraint = self.get_current_active_constraint(time)

        if current_constraint is None:
            if self._last_completed_constraint is not None:
                load_param = "NO"
                if self._last_completed_constraint.load_param is not None:
                    load_param = self._last_completed_constraint.load_param
                new_val = ("COMPLETED:" +
                           self._last_completed_constraint.name +
                           "-" +
                           load_param +
                           "-" +
                           self._last_completed_constraint.end_of_constraint.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                new_val = "NOTHING PLANNED"
        else:
            load_param = "NO"
            if current_constraint.load_param is not None:
                load_param = current_constraint.load_param
            new_val = ("RUNNING:" +
                       current_constraint.name +
                       "-" +
                       load_param +
                       "-" +
                       current_constraint.end_of_constraint.strftime("%Y-%m-%d %H:%M:%S"))

        return new_val

    def get_active_constraints_for_storage(self, time:datetime) -> list[LoadConstraint]:
        return [c for c in self._constraints if c.is_constraint_active_for_time_period(time) and c.end_of_constraint < DATETIME_MAX_UTC]

    def set_live_constraints(self, time: datetime, constraints: list[LoadConstraint]):

        if not constraints:
            constraints = []

        self._constraints = constraints
        if not constraints:
            return

        self._constraints = [c for c in self._constraints if c is not None]
        self._constraints.sort(key=lambda x: x.end_of_constraint)

        #remove all the infinite constraints but the last one
        if self._constraints[-1].end_of_constraint == DATETIME_MAX_UTC:
            removed_infinits : list[LoadConstraint] = []
            while self._constraints[-1].end_of_constraint == DATETIME_MAX_UTC:
                removed_infinits.append(self._constraints.pop())
                if len(self._constraints) == 0:
                    break

            #only one infinite is allowed!
            if removed_infinits:
                keep : LoadConstraint = removed_infinits[0]
                for k in removed_infinits:
                    if k.is_constraint_met():
                        continue
                    if k.score() > keep.score():
                        keep = k

                self._constraints.append(keep)

        # only one as fast as possible constraint can be active at a time.... and has to be first
        removed_as_fast = [(i,c) for i, c in enumerate(self._constraints) if c.as_fast_as_possible]
        if len(removed_as_fast) == 0 or (len(removed_as_fast) == 1 and removed_as_fast[0][0] == 0):
            # ok if there is a as fast constraint it should be the first one
            pass
        else:
            new_constraints = []
            for i, c in enumerate(self._constraints):
                if i < removed_as_fast[0][0]:
                    continue
                if c.as_fast_as_possible:
                    continue
                new_constraints.append(c)

            keep = removed_as_fast[0][1]
            end_ctr = keep.end_of_constraint
            for (_, k) in removed_as_fast:
                if k.is_constraint_met():
                    continue
                if k.score() > keep.score():
                    keep = k
            keep.end_of_constraint = end_ctr
            self._constraints = [keep].extend(new_constraints)

        #check all the constraints that have teh same end time, keep the highest score
        current_end = DATETIME_MIN_UTC

        current_cluster : list[tuple[int, LoadConstraint]] = []
        clusters : list[list[tuple[int, LoadConstraint]]] = []
        if self._constraints is None:
            self._constraints = []
        for i, c in enumerate(self._constraints):
            if c.end_of_constraint == DATETIME_MAX_UTC or c.end_of_constraint == DATETIME_MIN_UTC:
                continue

            if c.end_of_constraint == current_end:
                current_cluster.append((i,c))
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [(i,c)]
                current_end = c.end_of_constraint

        if len(current_cluster) > 1:
            clusters.append(current_cluster)

        if len(clusters) > 0:
            for current_cluster in clusters:
                keep_ic : tuple[int, LoadConstraint] = current_cluster[0]
                for i, c in current_cluster:
                    if c.score() > keep_ic[1].score():
                        keep_ic = (i,c)

                for i, c in current_cluster:
                    if i == keep_ic[0]:
                        continue
                    else:
                        self._constraints[i] = None

            self._constraints = [c for c in self._constraints if c is not None]

        #and now we may have to recompute the start values of the constraints
        prev_ct = None
        for c in self._constraints:
            if prev_ct is not None:
                c.reset_initial_value_to_follow_prev_if_needed(time, prev_ct)
                if c.is_constraint_met():
                    # keep the prev energy as it was possibly higher to meet this constraint
                    continue
            prev_ct = c

        self._constraints = [c for c in self._constraints if c.is_constraint_met() is False]

        #recompute the constraint start:
        kept = []
        current_start = DATETIME_MIN_UTC
        for c in self._constraints:
            c._internal_start_of_constraint = max(current_start, c.start_of_constraint)
            current_start = c.end_of_constraint
            kept.append(c)
            if current_start >= DATETIME_MAX_UTC:
                break

        self._constraints = kept
        if not self._constraints:
            self._constraints = []


    def push_live_constraint(self, time:datetime, constraint: LoadConstraint| None = None) -> bool:

        if self.qs_enable_device is False:
            self._constraints = []
            return True


        if not self._constraints:
            self._constraints = []

        if constraint is not None:

            if (self._last_completed_constraint is not None and
                self._last_completed_constraint.end_of_constraint == constraint.end_of_constraint and
                self._last_completed_constraint.score() >= constraint.score()):
                _LOGGER.debug(f"Constraint {constraint.name} not pushed because same end date as last completed one")
                return False



            for i, c in enumerate(self._constraints):
                if c == constraint:
                    return False
                if  c.end_of_constraint == constraint.end_of_constraint:
                    if c.score() == constraint.score():
                        _LOGGER.debug(f"Constraint {constraint.name} not pushed because same end date as another one, and same score")
                        return False
                    else:
                        self._constraints[i] = None
                        _LOGGER.info(f"Constraint {constraint.name} replacing {c.name} one with same end date, different score (last one force replace the new one")

            self._constraints.append(constraint)
            self.set_live_constraints(time, self._constraints)
            return True

    async def update_live_constraints(self, time:datetime, period: timedelta, end_constraint_min_tolerancy: timedelta = timedelta(seconds=2)) -> bool:

        if self.qs_enable_device is False:
            self._constraints = []
            return True

        # there should be ONLY ONE ACTIVE CONSTRAINT AT A TIME!
        # they are sorted in time order, the first one we find should be executed (could be a constraint with no end date
        # if it is the last and the one before are for the next days)
        if self._last_constraint_update is None:
            self._last_constraint_update = time

        prev_local_date = self._last_constraint_update.replace(tzinfo=pytz.UTC).astimezone(tz=None)
        now_local_date = time.replace(tzinfo=pytz.UTC).astimezone(tz=None)

        if prev_local_date.day != now_local_date.day:
            # we should reset some stuffs
            self.reset_daily_load_datas(time)


        current_constraint = None
        #if self.running_command is not None:
        #    force_solving =  False
        #elif
        # to update any constraint the load must be in a state with the right command working...do not update constraints during its execution
        # well don't like it ... running command will be gracefully handled by launch command
        if not self._constraints:
            self._constraints = []
            force_solving =  False
        else:
            force_solving = False

            # be sure we don't forget one ...
            for c in self._constraints:
                c.skip = False

            for i, c in enumerate(self._constraints):

                if c.skip:
                    continue

                do_update_c = False

                if c.is_constraint_met():
                    c.skip = True
                    force_solving = True
                    await self.ack_completed_constraint(time, c)
                    _LOGGER.info(f"{c.name} skipped because met")
                elif c.end_of_constraint <= time  and c.is_mandatory is False:
                    _LOGGER.info(f"{c.name} skipped because not mandatory")
                    c.skip = True
                    force_solving = True
                elif c.is_mandatory and c.end_of_constraint <  time + end_constraint_min_tolerancy:
                    # a not met mandatory one! we should expand it or force it
                    duration_s = c.best_duration_to_meet() + end_constraint_min_tolerancy
                    duration_s = max(timedelta(seconds=1200), duration_s*(1.0 + c.pushed_count*0.2)) # extend if we continue to push it
                    new_constraint_end = time + duration_s
                    handled_constraint_force = False
                    c.skip = True

                    if i < len(self._constraints) - 1:

                        for j in range(i+1, len(self._constraints)):

                            nc = self._constraints[j]

                            if nc.skip:
                                continue

                            if nc.end_of_constraint < time:
                                c.skip = True
                                continue

                            if nc.end_of_constraint >= new_constraint_end:
                                break

                            if nc.end_of_constraint < new_constraint_end:
                                if nc.is_constraint_met():
                                    nc.skip = True
                                else:
                                    force_solving = True
                                    # nc constraint may need to be forced or not
                                    if nc.score() > c.score():
                                        # we should skip the current one
                                        c.skip = True
                                        handled_constraint_force = True
                                        # make the current constraint the next important one
                                        # to break below after if handled_constraint_force:
                                        c = nc
                                        break
                                    else:
                                        nc.skip = True

                    if handled_constraint_force is False:

                        if c.pushed_count > 4:
                            # TODO: we should send a push notification to the one attached to the constraint!
                            # As it is not met and pushed too many times
                            c.skip = True
                            _LOGGER.info(f"{c.name} not met and pushed too many times")
                        else:

                            # unskip the current one
                            c.skip = False
                            c.pushed_count += 1
                            _LOGGER.info(f"{c.name} pushed because mandatory and not met (#pushed {c.pushed_count}) from {c.end_of_constraint} to {new_constraint_end}")
                            handled_constraint_force = True
                            c.end_of_constraint = new_constraint_end

                    if handled_constraint_force:
                        force_solving = True
                        # ok we have pushed or made a target the next important constraint
                        do_update_c = True
                        c.type = CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE  # force as much as we can....
                        _LOGGER.info(f"{c.name} handled_constraint_force")
                else:
                    do_update_c = True

                if do_update_c and c.is_constraint_active_for_time_period(time, time + period):
                    do_continue_ct = await c.update(time)
                    if do_continue_ct is False:
                        if c.is_constraint_met():
                            await self.ack_completed_constraint(time, c)
                            _LOGGER.info(f"{c.name} skipped because met (just after update)")
                        else:
                            _LOGGER.info(f"{c.name} stopped by callback (just after update)")
                        c.skip = True
                    break

            constraints = [c for c in self._constraints if c.skip is False]

            if len(constraints) != len(self._constraints):
                force_solving = True

            self.set_live_constraints(time, constraints)

            current_constraint = self.get_current_active_constraint(time)


        if current_constraint is not None:
            self.current_constraint_current_value = current_constraint.current_value
            self.current_constraint_current_energy = current_constraint.convert_target_value_to_energy(current_constraint.current_value)
            self.current_constraint_current_percent_completion = current_constraint.get_percent_completion(time)

        else:
            self.current_constraint_current_value = None
            self.current_constraint_current_energy = None
            self.current_constraint_current_percent_completion = None

        self._last_constraint_update = time
        return force_solving



    async def mark_current_constraint_has_done(self):
        time = datetime.now(tz=pytz.UTC)
        c = self.get_current_active_constraint(time)
        if c:
            # for it has met, will be properly handled in the update constraint for the load
            c.current_value = c.target_value
            await self.update_live_constraints(time, self.home._period)
            if self.is_load_active(time) is False or self.get_current_active_constraint(time) is None:
                await self.launch_command(time=time, command=CMD_IDLE, ctxt=f"mark_current_constraint_has_done constraint {self.get_current_active_constraint(time)} is active {self.is_load_active(time)}")

    async def async_reset_override_state(self):

        self.external_user_initiated_state = None
        self.external_user_initiated_state_time = None

        if self.asked_for_reset_user_initiated_state_time is None:
            # set the ask to now
            self.asked_for_reset_user_initiated_state_time = datetime.now(tz=pytz.UTC)
            await self.launch_qs_command_back(time=self.asked_for_reset_user_initiated_state_time, ctxt="async_reset_override_state get back to current command")


class TestLoad(AbstractLoad):

    def __init__(self, min_p=1500, max_p=1500, min_a=7, max_a=7, **kwargs):
        super().__init__(**kwargs)
        self.min_a = min_a
        self.max_a = max_a
        self.min_p = min_p
        self.max_p = max_p

    def get_min_max_power(self) -> (float, float):
        return self.min_p, self.max_p

    def get_min_max_phase_amps_for_budgeting(self)-> ( list[float|int],  list[float|int]):
        return [self.min_a, 0, 0], [self.max_a, 0, 0]


def align_time_series_and_values(
        tsv1: list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]] | list[tuple[datetime | None, str | float | None]],
        tsv2: list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]] | list[tuple[datetime | None, str | float | None]]| None,
        operation: Callable[[Any, Any], Any] | None = None):

    if not tsv1:
        if not tsv2:
            if operation is not None:
                return []
            else:
                return [], []
        else:
            if operation is not None:
                if len(tsv2[0]) == 3:
                    return [(t, operation(None, v), a) for t, v, a in tsv2]
                else:
                    return [(t, operation(None, v)) for t, v in tsv2]
            else:
                if len(tsv2[0]) == 3:
                    return [(t, None, None) for t, _, _ in tsv2], tsv2
                else:
                    return [(t, None) for t, _ in tsv2], tsv2

    if not tsv2:
        if operation is not None:
            if len(tsv1[0]) == 3:
                return [(t, operation(v, None), a) for t, v, a in tsv1]
            else:
                return [(t, operation(v, None)) for t, v in tsv1]
        else:
            if len(tsv1[0]) == 3:
                return tsv1, [(t, None, None) for t, _ in tsv1]
            else:
                return tsv1, [(t, None) for t, _ in tsv1]

    timings = {}

    for i, tv in enumerate(tsv1):
        timings[tv[0]] = [i, None]
    for i, tv in enumerate(tsv2):
        if tv[0] in timings:
            timings[tv[0]][1] = i
        else:
            timings[tv[0]] = [None, i]

    timings = [(k, v) for k, v in timings.items()]
    timings.sort(key=lambda x: x[0])
    t_only = [t for t, _ in timings]

    object_len = 3
    object_len = min(object_len, len(tsv1[0]), len(tsv2[0]))

    #compute all values for each time
    new_v1: list[float | str | None] = [0] * len(t_only)
    new_v2: list[float | str | None] = [0] * len(t_only)

    new_attr_1 = []
    new_attr_2 = []
    if object_len == 3:
        new_attr_1: list[dict | None] = [None] * len(t_only)
        new_attr_2: list[dict | None] = [None] * len(t_only)

    for vi in range(2):

        new_v = new_v1
        new_attr = new_attr_1
        tsv = tsv1
        if vi == 1:
            if operation is None:
                new_v = new_v2
                new_attr = new_attr_2
            tsv = tsv2

        last_real_idx = None
        for i, (t, idxs) in enumerate(timings):
            attr_to_put = None
            if idxs[vi] is not None:
                #ok an exact value
                last_real_idx = idxs[vi]
                val_to_put = (tsv[last_real_idx][1])
                if object_len == 3:
                    attr_to_put = (tsv[last_real_idx][2])
            else:
                if last_real_idx is None:
                    #we have new values "before" the first real value"
                    val_to_put = (tsv[0][1])
                    if object_len == 3:
                        attr_to_put = (tsv[0][2])
                elif last_real_idx == len(tsv) - 1:
                    #we have new values "after" the last real value"
                    val_to_put = (tsv[-1][1])
                    if object_len == 3:
                        attr_to_put = (tsv[-1][2])
                else:
                    # we have new values "between" two real values"
                    # interpolate
                    vcur = tsv[last_real_idx][1]
                    vnxt = tsv[last_real_idx + 1][1]

                    if vnxt is None:
                        val_to_put = vcur
                    elif vcur is None:
                        val_to_put = None
                    else:
                        d1 = float((t - tsv[last_real_idx][0]).total_seconds())
                        d2 = float((tsv[last_real_idx + 1][0] - tsv[last_real_idx][0]).total_seconds())
                        nv = (d1 / d2) * (vnxt - vcur) + vcur
                        val_to_put = float(nv)
                    if object_len == 3:
                        attr_to_put = (tsv[last_real_idx][2])

            if object_len == 3 and attr_to_put is not None:
                attr_to_put = dict(attr_to_put)

            if vi == 0 or operation is None:
                new_v[i] = val_to_put
                if object_len == 3:
                    new_attr[i] = attr_to_put
            else:
                if new_v[i] is None or val_to_put is None:
                    new_v[i] = None
                else:
                    new_v[i] = operation(new_v[i], val_to_put)
                if object_len == 3:
                    if new_attr[i] is None:
                        new_attr[i] = attr_to_put
                    elif attr_to_put is not None:
                        new_attr[i].update(attr_to_put)

    #ok so we do have values and timings for 1 and 2
    if operation is not None:
        if object_len == 3:
            return list(zip(t_only, new_v1, new_attr_1))
        else:
            return list(zip(t_only, new_v1))
    if object_len == 3:
        return list(zip(t_only, new_v1, new_attr_1)), list(zip(t_only, new_v2, new_attr_2))
    else:
        return list(zip(t_only, new_v1)), list(zip(t_only, new_v2))


def get_slots_from_time_serie(time_serie, start_time: datetime, end_time: datetime | None) -> list[
    tuple[datetime | None, str | float | None]]:
    if not time_serie:
        return []

    start_idx = bisect_left(time_serie, start_time, key=itemgetter(0))
    # get one before
    if start_idx > 0:
        if time_serie[start_idx][0] != start_time:
            start_idx -= 1

    if end_time is None:
        return time_serie[start_idx:start_idx + 1]

    end_idx = bisect_left(time_serie, end_time, key=itemgetter(0))
    if end_idx >= len(time_serie):
        end_idx = len(time_serie) - 1
    elif end_idx < len(time_serie) - 1:
        # take one after
        if time_serie[end_idx][0] != end_time:
            end_idx += 1

    return time_serie[start_idx:end_idx + 1]
