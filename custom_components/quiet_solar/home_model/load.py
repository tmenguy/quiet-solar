import logging
from bisect import bisect_left
from datetime import datetime, timedelta
from collections.abc import Generator
from operator import itemgetter

from .commands import LoadCommand, copy_command
from .constraints import LoadConstraint, DATETIME_MAX_UTC, DATETIME_MIN_UTC

from typing import TYPE_CHECKING, Any, Mapping, Callable, Awaitable

from ..const import CONF_POWER, CONF_SWITCH, CONF_LOAD_IS_BOOST_ONLY

import slugify

if TYPE_CHECKING:
    pass

NUM_MAX_INVALID_PROBES_COMMANDS = 3

_LOGGER = logging.getLogger(__name__)
class AbstractDevice(object):
    def __init__(self, name:str, device_type:str|None = None, **kwargs):
        super().__init__()
        self.name = name
        self._device_type = device_type
        self.device_id = f"qs_{slugify.slugify(name, separator="_")}_{self.device_type}"
        self.home = kwargs.pop("home", None)

    @property
    def device_type(self):
        if self._device_type is None:
            return self.__class__.__name__
        return self._device_type

    def __repr__(self):
        return self.device_id


class AbstractLoad(AbstractDevice):

    def __init__(self, **kwargs):
        self.switch_entity = kwargs.pop(CONF_SWITCH, None)
        self.power_use = kwargs.pop(CONF_POWER, None)
        self.load_is_auto_to_be_boosted = kwargs.pop(CONF_LOAD_IS_BOOST_ONLY, False)
        super().__init__(**kwargs)

        self._constraints: list[LoadConstraint] = []
        self._last_completed_constraint: LoadConstraint | None = None
        self.current_command : LoadCommand | None = None
        self.running_command : LoadCommand | None = None # a command that has been launched but not yet finished, wait for its resolution
        self._stacked_command: LoadCommand | None = None # a command (keep only the last one) that has been pushed to be executed later when running command is free
        self.running_command_first_launch: datetime | None = None
        self.running_command_num_relaunch : int = 0
        self.running_command_num_relaunch_after_invalid: int = 0
        self.current_constraint_current_value: float | None = None
        self.current_constraint_current_energy: float | None = None
        self._externally_initialized_constraints = False

        self._ack_command(None)

        self.qs_best_effort_green_only = False

    def support_green_only_switch(self) -> bool:
        return False

    def get_power_from_switch_state(self, state : str | None) -> float | None:
        if state is None:
            return None
        if state == "on":
            return self.power_use
        else:
            return 0

    async def do_run_check_load_activity_and_constraints(self, time: datetime)-> bool:
        if self._externally_initialized_constraints is False:
            return False
        return  await self.check_load_activity_and_constraints(time)



    def load_constraints_from_storage(self, time:datetime, constraints_dicts: list[dict], stored_executed: dict | None):
        self.reset()
        for c_dict in constraints_dicts:
            cs_load = LoadConstraint.new_from_saved_dict(time, self, c_dict)
            if cs_load is not None:
                if not cs_load.is_constraint_active_for_time_period(time):
                    cs_load.end_of_constraint = time + timedelta(seconds=5*60)
                if cs_load.is_constraint_active_for_time_period(time):
                    self.push_live_constraint(time, cs_load)

        if stored_executed is not None:
            self._last_completed_constraint = LoadConstraint.new_from_saved_dict(time, self, stored_executed)

        self._externally_initialized_constraints = True

    async def check_load_activity_and_constraints(self, time: datetime) -> bool:
        return False

    def get_update_value_callback_for_constraint_class(self, constraint:LoadConstraint) -> Callable[[LoadConstraint, datetime], Awaitable[tuple[float | None, bool]]] | None:
        return None

    def is_load_active(self, time: datetime):
        if not self._constraints:
            return False
        return True

    def reset(self):
        self.current_command = None
        self._constraints = []
        self._last_completed_constraint = None

    def get_active_constraint_generator(self, start_time:datetime, end_time) -> Generator[Any, None, None]:
        for c in self._constraints:
            if c.is_constraint_active_for_time_period(start_time, end_time):
                yield c

    def get_current_active_constraint(self, time:datetime) -> LoadConstraint | None:
        for c in self._constraints:
            if c.is_constraint_active_for_time_period(time):
                return c
        return None

    def get_active_constraints_for_storage(self, time:datetime) -> list[LoadConstraint]:
        return [c for c in self._constraints if c.is_constraint_active_for_time_period(time) and c.end_of_constraint < DATETIME_MAX_UTC]

    def set_live_constraints(self, time: datetime, constraints: list[LoadConstraint]):

        self._constraints = constraints
        if not constraints:
            return

        self._constraints.sort(key=lambda x: x.end_of_constraint)

        #remove all the infinite constraints but the last one
        if self._constraints[-1].end_of_constraint == DATETIME_MAX_UTC:
            removed_infinits = []
            while self._constraints[-1].end_of_constraint == DATETIME_MAX_UTC:
                removed_infinits.append(self._constraints.pop())
                if len(self._constraints) == 0:
                    break

            #only one infinite is allowed!
            keep = removed_infinits[0]
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


    def push_live_constraint(self, time:datetime, constraint: LoadConstraint| None = None, check_end_constraint_exists:bool = False) -> bool:

        if constraint is not None:

            if (check_end_constraint_exists and
                    self._last_completed_constraint is not None and
                    self._last_completed_constraint.end_of_constraint == constraint.end_of_constraint):
                _LOGGER.info(f"Constraint {constraint.name} not pushed because same end date as last completed one")
                return False

            for c in self._constraints:
                if c == constraint:
                    return False
                if check_end_constraint_exists and constraint is not None and c.end_of_constraint == constraint.end_of_constraint:
                    _LOGGER.info(f"Constraint {constraint.name} not pushed because same end date as another one")
                    return False
            self._constraints.append(constraint)
            self.set_live_constraints(time, self._constraints)
            return True


    async def update_live_constraints(self, time:datetime, period: timedelta) -> bool:

        # there should be ONLY ONE ACTIVE CONSTRAINT AT A TIME!
        # they are sorted in time order, the first one we find should be executed (could be a constraint with no end date
        # if it is the last and the one before are for the next days)

        # to update any constraint the load must be in a state with the right command working...do not update constraints during its execution
        current_constraint = None
        if self.running_command is not None:
            force_solving =  False

        elif not self._constraints:
            force_solving =  False
        else:
            force_solving = False
            for i, c in enumerate(self._constraints):

                if c.skip:
                    continue

                if c.end_of_constraint < time:

                    # it means this constraint should be finished by now
                    if c.is_constraint_met() or c.is_mandatory is False:
                        _LOGGER.info(f"{c.name} skipped because met or not mandatory")
                        c.skip = True
                    else:
                        # a not met mandatory one! we should expand it or force it
                        duration_s = c.best_duration_to_meet()
                        duration_s = duration_s*(1.0 + c.pushed_count*0.2) # extend if we continue to push it
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
                                c.end_of_constraint = new_constraint_end
                                # unskip the current one
                                c.skip = False
                                c.pushed_count += 1
                                _LOGGER.info(f"{c.name} pushed because not mandatory and not met")
                                handled_constraint_force = True

                        if handled_constraint_force:
                            force_solving = True
                            # ok we have pushed or made a target the next important constraint
                            do_continue_ct = await c.update(time)
                            if do_continue_ct is False:
                                if c.is_constraint_met():
                                    self._last_completed_constraint = c
                                    _LOGGER.info(f"{c.name} skipped because met (just after update)")
                                else:
                                    _LOGGER.info(f"{c.name} stopped by callback (just after update)")
                                c.skip = True
                            break

                elif c.is_constraint_met():
                    c.skip = True
                    _LOGGER.info(f"{c.name} skipped because met")
                elif c.is_constraint_active_for_time_period(time, time + period):
                    do_continue_ct = await c.update(time)
                    if do_continue_ct is False:
                        if c.is_constraint_met():
                            self._last_completed_constraint = c
                            _LOGGER.info(f"{c.name} skipped because met (just after update)")
                        else:
                            _LOGGER.info(f"{c.name} stopped by callback (just after update)")
                        c.skip = True
                    break

            constraints = [c for c in self._constraints if c.skip is False]

            if len(constraints) != len(self._constraints):
                force_solving = True

            self.set_live_constraints(time, constraints)

            for c in self._constraints:
                if c.is_constraint_active_for_time_period(time):
                    current_constraint = c
                    break


        if current_constraint is not None:
            self.current_constraint_current_value = current_constraint.current_value
            self.current_constraint_current_energy = current_constraint.convert_target_value_to_energy(current_constraint.current_value)
        else:
            self.current_constraint_current_value = None
            self.current_constraint_current_energy = None


        return force_solving

    def _ack_command(self, command):
        self.current_command = command
        self.running_command = None
        self.running_command_num_relaunch = 0
        self.running_command_num_relaunch_after_invalid = 0
        self.running_command_first_launch = None

    async def launch_command(self, time:datetime, command: LoadCommand):

        command = copy_command(command)

        if self.running_command is not None:
            if self.running_command == command:
                self._stacked_command = None #no need of it anymore
                return
            else:
                self._stacked_command = command
                return

        # there is no running : whatever we wil not execute the stacked one but only the last one
        self._stacked_command = None

        if self.current_command == command:
            # We kill the stacked one and keep the current one like the choice above
            self._stacked_command = None
            self.current_command = command # needed as command == has been overcharged to not test everything
            return


        self.running_command = command
        self.running_command_first_launch = time
        is_command_set = await self.execute_command(time, command)
        if is_command_set is None:
            # hum we may have an impossibility to launch this command
            _LOGGER.info(f"Impossible to launch this command {command.command} on this load {self.name}")
        else:
            await self.check_commands(time)

    def is_load_command_set(self, time:datetime):
        return self.running_command is None and self.current_command is not None

    async def check_commands(self, time: datetime) -> timedelta:

        res = timedelta(seconds=0)

        if self.running_command is not None:
            is_command_set = await self.probe_if_command_set(time, self.running_command)
            if is_command_set is None:
                # impossible to run this command for this load ...
                self.running_command_num_relaunch_after_invalid += 1
                _LOGGER.info(f"impossible to check command {self.running_command.command} for this load {self.name}) (#{self.running_command_num_relaunch_after_invalid})")
                if self.running_command_num_relaunch_after_invalid >= NUM_MAX_INVALID_PROBES_COMMANDS:
                    # will kill completely the command ....
                    self._ack_command(None)

            if is_command_set is True:
                self._ack_command(self.running_command)
            else:
                res = time - self.running_command_first_launch


        if self.running_command is None and self._stacked_command is not None:
            await self.launch_command(time, self._stacked_command)

        return res

    async def force_relaunch_command(self, time: datetime):
        if self.running_command is not None:
            self.running_command_num_relaunch += 1
            is_command_set = await self.execute_command(time, self.running_command)
            if is_command_set is None:
                _LOGGER.info(f"impossible to force command {self.running_command.command} for this load {self.name})")
            else:
                await self.check_commands(time)



    async def execute_command(self, time: datetime, command: LoadCommand) -> bool | None:
        print(f"Executing command {command}")
        return False

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool | None:
        return True



class TestLoad(AbstractLoad):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


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
