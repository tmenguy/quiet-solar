from datetime import datetime

from ..const import CONSTRAINT_TYPE_MANDATORY_END_TIME
from ..ha_model.device import HADeviceMixin
from ..home_model.commands import LoadCommand, CMD_ON, CMD_OFF, CMD_IDLE
from ..home_model.constraints import TimeBasedSimplePowerLoadConstraint
from ..home_model.load import AbstractLoad
from homeassistant.const import Platform, SERVICE_TURN_ON, SERVICE_TURN_OFF, STATE_UNKNOWN, STATE_UNAVAILABLE


class QSOnOffDuration(HADeviceMixin, AbstractLoad):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
            return False

        if command.is_like(CMD_ON):
            return state.state == "on"
        else:
            return state.state == "off"

    async def check_load_activity_and_constraints(self, time: datetime):
        # check that we have a connected car, and which one, or that it is completely disconnected
        #  if there is no more car ... just reset

        start_schedule, end_schedule = self.get_next_scheduled_event(time)

        if start_schedule is not None and end_schedule is not None:

            do_schedule = True

            if self._last_completed_constraint is not None and self._last_completed_constraint.end_of_constraint >= end_schedule:
                # we have already scheduled this event
                do_schedule = False

            if do_schedule:
                for ct in self._constraints:
                    if ct.end_of_constraint >= end_schedule:
                        do_schedule = False
                        break

            if do_schedule:
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
                self.push_live_constraint(time, load_mandatory)

        return

