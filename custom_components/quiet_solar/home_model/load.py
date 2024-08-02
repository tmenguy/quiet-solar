
from abc import ABC
from datetime import datetime, timedelta
from collections.abc import Generator

from .commands import LoadCommand, CMD_OFF, copy_command
from .constraints import LoadConstraint

from typing import TYPE_CHECKING, Any

from ..const import DATETIME_MAX_UTC, DATETIME_MIN_UTC

if TYPE_CHECKING:
    pass

FLOATING_PERIOD = 24*3600

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
                        else:
                            c.end_of_constraint = new_constraint_end
                            force_solving = True
                            c.skip = False
                            c.pushed_count += 1

            elif c.is_constraint_met():
                c.skip = True
            elif c.is_constraint_active_for_time_period(dt, dt + period):
                await c.update(dt)
                if c.is_constraint_met():
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







