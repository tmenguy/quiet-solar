import logging
from datetime import datetime

from ..const import CONSTRAINT_TYPE_MANDATORY_END_TIME
from ..ha_model.device import HADeviceMixin
from ..home_model.commands import LoadCommand, CMD_ON, CMD_OFF, CMD_IDLE
from ..home_model.constraints import TimeBasedSimplePowerLoadConstraint
from ..home_model.load import AbstractLoad
from homeassistant.const import Platform, SERVICE_TURN_ON, SERVICE_TURN_OFF, STATE_UNKNOWN, STATE_UNAVAILABLE

_LOGGER = logging.getLogger(__name__)
class QSOnOffDuration(HADeviceMixin, AbstractLoad):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_pushed_end_constraint = None

    def reset(self):
        super().reset()
        self._last_pushed_end_constraint = None

    def get_platforms(self):
        return [ Platform.SENSOR, Platform.SWITCH ]

    async def execute_command(self, time: datetime, command:LoadCommand) -> bool | None:
        if command.is_like(CMD_ON):
            action = SERVICE_TURN_ON
        elif command.is_like(CMD_OFF) or command.is_like(CMD_IDLE):
            action = SERVICE_TURN_OFF
        else:
            raise ValueError("Invalid command")

        await self.hass.services.async_call(
            domain=Platform.SWITCH,
            service=action,
            target={"entity_id": self.switch_entity},
        )
        return False

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool | None:
        """ check the states of the switch to see if the command is set """
        state = self.hass.states.get(self.switch_entity) # may be faster to get the python entity object no?

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return None

        if command.is_like(CMD_ON):
            return state.state == "on"
        else:
            return state.state == "off"

    async def check_load_activity_and_constraints(self, time: datetime):
        # check that we have a connected car, and which one, or that it is completely disconnected
        #  if there is no more car ... just reset

        start_schedule, end_schedule = self.get_next_scheduled_event(time)

        if start_schedule is not None and end_schedule is not None:

            if self._last_pushed_end_constraint is not None:
                if self._last_pushed_end_constraint.end_of_constraint == end_schedule:
                    # we already have a constraint for this end time
                    # this is a small optimisation to avoid creating a constraint object just to
                    # let push_live_constraint check that it is already in the list or completed
                    return

            # schedule the load to be launched
            load_mandatory = TimeBasedSimplePowerLoadConstraint(
                    type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
                    time=time,
                    load=self,
                    from_user=False,
                    end_of_constraint=end_schedule,
                    power=self.power_use,
                    target=(end_schedule - start_schedule).total_seconds()
            )
            # check_end_constraint_exists will check that the constraint is not already in the list
            # or have not been done already after a restart
            self.push_live_constraint(time, load_mandatory, check_end_constraint_exists=True)
            self._last_pushed_end_constraint.end_of_constraint = end_schedule