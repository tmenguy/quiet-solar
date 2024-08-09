import logging
from abc import ABC
from datetime import datetime, timedelta
from collections.abc import Generator

from .commands import LoadCommand, CMD_OFF, copy_command
from .constraints import LoadConstraint, DATETIME_MAX_UTC, DATETIME_MIN_UTC

from typing import TYPE_CHECKING, Any, Mapping, Callable

if TYPE_CHECKING:
    pass

FLOATING_PERIOD = 48*3600

_LOGGER = logging.getLogger(__name__)
class AbstractDevice(object):
    def __init__(self, name:str, device_type:str|None = None, **kwargs):
        super().__init__()
        self.name = name
        self._device_type = device_type
        self.device_id = f"qs_device_{name}_{self.device_type}"
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
        super().__init__(**kwargs)

        self._constraints: list[LoadConstraint] = []
        self.current_command : LoadCommand | None = None
        self.running_command : LoadCommand | None = None # a command that has been launched but not yet finished, wait for its resolution
        self._stacked_command: LoadCommand | None = None # a command (keep only the last one) that has been pushed to be executed later when running command is free
        self.default_cmd : LoadCommand = CMD_OFF
        self.running_command_first_launch: datetime | None = None
        self.running_command_num_relaunch : int = 0


    async def check_load_activity_and_constraints(self, time: datetime):
        return

    def is_load_active(self, time: datetime):
        if not self._constraints:
            return False
        return True


    def reset(self):
        self.current_command = None
        self._constraints = []

    def get_active_constraint_generator(self, start_time:datetime, end_time) -> Generator[Any, None, None]:
        for c in self._constraints:
            if c.is_constraint_active_for_time_period(start_time, end_time):
                yield c

    def set_live_constraints(self, constraints: list[LoadConstraint]):
        self._constraints = constraints
        if not constraints:
            return

        self._constraints.sort(key=lambda x: x.end_of_constraint)
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
                if k.is_mandatory:
                    keep = k
                    break

            self._constraints.append(keep)

        #recompute the contraint start:
        current_start = DATETIME_MIN_UTC
        for c in self._constraints:
            c.start_of_constraint = max(current_start, c.user_start_of_constraint)
            current_start = c.end_of_constraint


    def push_live_constraint(self, constraint: LoadConstraint| None = None):
        if constraint is not None:
            if constraint.end_of_constraint == DATETIME_MAX_UTC and len(self._constraints) > 0:
                #only one infinite is allowed!
                while self._constraints[-1].end_of_constraint == DATETIME_MAX_UTC:
                    self._constraints.pop()
                    if len(self._constraints) == 0:
                        break
            self._constraints.append(constraint)
        self.set_live_constraints(self._constraints)


    async def update_live_constraints(self, dt:datetime, period: timedelta) -> bool:

        # there should be ONLY ONE ACTIVE CONSTRAINT AT A TIME!
        # they are sorted in time order, the first one we find should be executed (could be a constraint with no end date
        # if it is the last and the one before are for the next days)

        # to update any constraint the load must be in a state with the right command working...do not update constraints during its execution
        if self.running_command is not None:
            return False

        if not self._constraints:
            return False


        force_solving = False
        for i, c in enumerate(self._constraints):

            if c.skip:
                continue

            if c.end_of_constraint < dt:

                if c.is_constraint_met() or c.is_mandatory is False:
                    _LOGGER.info(f"{c.name} skipped because met or not mandatory")
                    c.skip = True
                else:
                    # a not met mandatory one! we should expand it or force it
                    duration_s = c.best_duration_to_meet()
                    new_constraint_end = dt + duration_s
                    handled_constraint_force = False
                    c.skip = True
                    if i < len(self._constraints) - 1:

                        for j in range(i+1, len(self._constraints)):

                            nc = self._constraints[j]

                            if nc.skip:
                                continue

                            if nc.end_of_constraint < dt:
                                c.skip = True
                                continue

                            if nc.end_of_constraint >= new_constraint_end:
                                break

                            if nc.end_of_constraint < new_constraint_end:
                                if nc.is_constraint_met():
                                    nc.skip = True
                                else:
                                    nc.is_mandatory = True
                                    nc.target_value = c.target_value
                                    force_solving = True
                                    handled_constraint_force = True
                                    break

                    if handled_constraint_force is False:

                        if c.pushed_count > 1:
                            #TODO: we should send a push notification to the one attached to the constraint!
                            c.skip = True
                            _LOGGER.info(f"{c.name} not met and pushed too many times")
                        else:
                            c.end_of_constraint = new_constraint_end
                            force_solving = True
                            c.skip = False
                            c.pushed_count += 1
                            _LOGGER.info(f"{c.name} pushed because not mandatory and not met")

            elif c.is_constraint_met():
                c.skip = True
                _LOGGER.info(f"{c.name} skipped because met")
            elif c.is_constraint_active_for_time_period(dt, dt + period):
                await c.update(dt)
                if c.is_constraint_met():
                    _LOGGER.info(f"{c.name} skipped because met (just after update)")
                    c.skip = True
                break

        constraints = [c for c in self._constraints if c.skip is False]

        if len(constraints) != len(self._constraints):
            force_solving = True
        
        self.set_live_constraints(constraints)

        return force_solving

    async def update_value_callback_example(self, ct: LoadConstraint, time: datetime) -> float:
        """ Example of a value compute callback for a load constraint. like get a sensor state, compute energy available for a car from battery charge etc
        it could also update the current command of the load if needed
        """
        pass

    async def launch_command(self, time:datetime, command: LoadCommand):

        command = copy_command(command)

        if self.running_command is not None:
            if self.running_command == command:
                self._stacked_command = None #no need of it anymore
                return
            else:
                self._stacked_command = command
                return

        elif self.current_command == command:
            # We kill the stacked one and keep the current one like the choice above
            self._stacked_command = None
            return
        else:
            #no running command ... kill the stacked one and execute this one
            self._stacked_command = None

        self.running_command = command
        self.running_command_first_launch = time
        await self.execute_command(time, command)
        is_command_set = await self.probe_if_command_set(time, command)
        if is_command_set:
            self.current_command = command
            self.running_command = None
            self.running_command_num_relaunch = 0
            self.running_command_first_launch = None

    def is_load_command_set(self, time:datetime):
        return self.running_command is None and self.current_command is not None
    async def check_commands(self, time: datetime):

        res = timedelta(seconds=0)

        if self.running_command is not None:
            is_command_set = await self.probe_if_command_set(time, self.running_command)
            if is_command_set:
                self.current_command = self.running_command
                self.running_command = None
                self.running_command_num_relaunch = 0
                self.running_command_first_launch = None
            else:
                res = time - self.running_command_first_launch


        if self.running_command is None and self._stacked_command is not None:
            await self.launch_command(time, self._stacked_command)

        return res

    async def force_relaunch_command(self, time: datetime):
        if self.running_command is not None:
            self.running_command_num_relaunch += 1
            await self.execute_command(time, self.running_command)
            is_command_set = await self.probe_if_command_set(time, self.running_command)
            if is_command_set:
                self.current_command = self.running_command
                self.running_command = None
                self.running_command_num_relaunch = 0
                self.running_command_first_launch = None



    async def execute_command(self, time: datetime, command: LoadCommand):
        print(f"Executing command {command}")

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool:
        return True



class TestLoad(AbstractLoad):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def align_time_series_and_values(
        tsv1: list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]],
        tsv2: list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]] | None,
        operation: Callable[[Any, Any], Any] | None = None):
    if not tsv1:
        if not tsv2:
            if operation is not None:
                return []
            else:
                return [], []
        else:
            if operation is not None:
                return [(t, operation(0, v), a) for t, v, a in tsv2]
            else:
                return [(t, 0, None) for t, _, a in tsv2], tsv2

    if not tsv2:
        if operation is not None:
            return [(t, operation(v, 0), a) for t, v, a in tsv1]
        else:
            return tsv1, [(t, 0, None) for t, _ in tsv1]

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

    #compute all values for each time
    new_v1: list[float | str | None] = [0] * len(t_only)
    new_v2: list[float | str | None] = [0] * len(t_only)

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
            if idxs[vi] is not None:
                #ok an exact value
                last_real_idx = idxs[vi]
                val_to_put = (tsv[last_real_idx][1])
                attr_to_put = (tsv[last_real_idx][2])
            else:
                if last_real_idx is None:
                    #we have new values "before" the first real value"
                    val_to_put = (tsv[0][1])
                    attr_to_put = (tsv[0][2])
                elif last_real_idx == len(tsv) - 1:
                    #we have new values "after" the last real value"
                    val_to_put = (tsv[-1][1])
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

                    attr_to_put = (tsv[last_real_idx][2])

            if attr_to_put is not None:
                attr_to_put = dict(attr_to_put)
            if vi == 0 or operation is None:
                new_v[i] = val_to_put
                new_attr[i] = attr_to_put
            else:
                if new_v[i] is None or val_to_put is None:
                    new_v[i] = None
                else:
                    new_v[i] = operation(new_v[i], val_to_put)

                if new_attr[i] is None:
                    new_attr[i] = attr_to_put
                elif attr_to_put is not None:
                    new_attr[i].update(attr_to_put)

    #ok so we do have values and timings for 1 and 2
    if operation is not None:
        return list(zip(t_only, new_v1, new_attr_1))

    return list(zip(t_only, new_v1, new_attr_1)), list(zip(t_only, new_v2, new_attr_2))
