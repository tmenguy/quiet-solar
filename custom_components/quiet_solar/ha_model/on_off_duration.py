import logging
from datetime import datetime

from ..const import CONSTRAINT_TYPE_MANDATORY_END_TIME, CONSTRAINT_TYPE_FILLER, SENSOR_CONSTRAINT_SENSOR_ON_OFF
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

    def support_green_only_switch(self) -> bool:
        if self.load_is_auto_to_be_boosted:
            return False
        return True

    def get_platforms(self):
        parent = super().get_platforms()
        if parent is None:
            parent = set()
        else:
            parent = set(parent)
        parent.update([ Platform.SENSOR, Platform.SWITCH ])
        return list(parent)

    def get_virtual_current_constraint_translation_key(self) -> str | None:
        return SENSOR_CONSTRAINT_SENSOR_ON_OFF

    async def execute_command(self, time: datetime, command:LoadCommand) -> bool | None:
        if command.is_like(CMD_ON):
            action = SERVICE_TURN_ON
        elif command.is_off_or_idle():
            action = SERVICE_TURN_OFF
        else:
            raise ValueError("Invalid command")

        _LOGGER.info(f"Executing on/off command {action} on {self.switch_entity}")
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

    async def check_load_activity_and_constraints(self, time: datetime) -> bool:
        # check that we have a connected car, and which one, or that it is completely disconnected
        #  if there is no more car ... just reset

        start_schedule, end_schedule = self.get_next_scheduled_event(time)

        if start_schedule is not None and end_schedule is not None:

            if self._last_pushed_end_constraint is not None:
                if self._last_pushed_end_constraint == end_schedule:
                    # we already have a constraint for this end time
                    # this is a small optimisation to avoid creating a constraint object just to
                    # let push_live_constraint check that it is already in the list or completed
                    return False

            # schedule the load to be launched
            type = CONSTRAINT_TYPE_MANDATORY_END_TIME
            if self.load_is_auto_to_be_boosted:
                type = CONSTRAINT_TYPE_FILLER
            elif self.qs_best_effort_green_only is True:
                type = CONSTRAINT_TYPE_FILLER # will be after battery filling

            load_mandatory = TimeBasedSimplePowerLoadConstraint(
                    type=type,
                    time=time,
                    load=self,
                    from_user=False,
                    end_of_constraint=end_schedule,
                    power=self.power_use,
                    initial_value=0,
                    target_value=(end_schedule - start_schedule).total_seconds()
            )
            # check_end_constraint_exists will check that the constraint is not already in the list
            # or have not been done already after a restart
            res = self.push_live_constraint(time, load_mandatory, check_end_constraint_exists=True)
            self._last_pushed_end_constraint = end_schedule
            return res

        return False