
from abc import abstractmethod
from datetime import datetime, timedelta
from collections.abc import Generator
from bisect import bisect_left

from home_model.commands import LoadCommand, CMD_OFF
from home_model.constraints import LoadConstraint

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    pass

FLOATING_PERIOD = 24*3600

class AbstractLoad(object):

    def __init__(self, name:str, **kwargs):
        super().__init__()
        self._name = name
        self._constraints: list[LoadConstraint] = []
        #self._timed_commands: list[tuple[datetime, LoadConstraint, LoadCommand]] = []
        self._current_command : dict | None = None
        self.default_cmd : LoadCommand = CMD_OFF


    def __repr__(self):
        return f"{self._name} load"

    def get_active_constraint_generator(self, start_time:datetime, end_time) -> Generator[Any, None, None]:
        for c in self._constraints:
            if c.is_constraint_active_for_time_period(start_time, end_time):
                yield c

    def set_live_constraints(self, constraints: list[LoadConstraint]):
        self._constraints = constraints
        self._constraints.sort(key=lambda x: x.end_of_constraint)
        if self._constraints[-1].end_of_constraint == datetime.max:
            removed_infinits = []
            while self._constraints[-1].end_of_constraint == datetime.max:
                removed_infinits.append(self._constraints.pop())

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
        current_start = datetime.min
        for c in self._constraints:
            c.start_of_constraint = max(current_start, c.user_start_of_constraint)
            current_start = c.end_of_constraint


    def push_live_constraint(self, constraint: LoadConstraint| None = None):
        if constraint is not None:
            if constraint.end_of_constraint == datetime.max:
                #only one infinite is allowed!
                while self._constraints[-1].end_of_constraint == datetime.max:
                    self._constraints.pop()
            self._constraints.append(constraint)
        self.set_live_constraints(self._constraints)

    def reset_commands_after_time(self, dt:datetime):

        last_slot = bisect_left(self._timed_commands, dt, key=lambda x: x[0])
        if last_slot >= len(self._timed_commands):
            return

        self._timed_commands = self._timed_commands[:last_slot]

    def push_timed_command(self, dt:datetime, constraint:LoadConstraint, command: LoadCommand):
        self._timed_commands.append((dt, constraint, command))
        self._timed_commands.sort(key=lambda x: x[0])


    async def update_live_constraints(self, dt:datetime, period: timedelta) -> bool:

        # there should be ONLY ONE ACTIVE CONSTRAINT AT A TIME!
        # they are sorted in time order, the first one we find should be executed (could be a constraint with no end date
        # if it is the last and the one before are for the next days)

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



    def execute_time(self, dt:datetime):
        self.update_live_constraints(dt)
        while self._timed_commands[0][0] >= dt:
            ct, constraint, command = self._timed_commands.pop(0)
            constraint.update_current_command(dt, command)
            self.execute_command(command)



    async def update_value_callback_example(self, ct: LoadConstraint, time: datetime) -> float:
        """ Example of a value compute callback for a load constraint. like get a sensor state, compute energy available for a car from battery charge etc
        it could also update the current command of the load if needed
        """
        pass


    async def update_command_callback_example(self, ct: LoadConstraint, time: datetime) -> LoadCommand:
        """ Example of a command compute callback for a load constraint. like get a switch state, and update the real state of a constraint and load
        """
        pass

    @abstractmethod
    async def execute_command(self, command: LoadCommand):
        """ Execute a command on the load."""



class TestLoad(AbstractLoad):

    def __init__(self, name:str, **kwargs):
        super().__init__(name, **kwargs)

    async def execute_command(self, command: LoadCommand):
        print(f"Executing command {command} on {self._name}")

    async def update_value_callback_example(self, ct: LoadConstraint, time: datetime) -> float:
        return 0.0

    async def update_command_callback_example(self, ct: LoadConstraint, time: datetime) -> LoadCommand:
        return self.default_cmd







