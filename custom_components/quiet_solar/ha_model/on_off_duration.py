import logging
from datetime import datetime, timedelta
from enum import StrEnum
from datetime import time as dt_time

import pytz

from ..const import CONSTRAINT_TYPE_MANDATORY_END_TIME, CONSTRAINT_TYPE_FILLER, SENSOR_CONSTRAINT_SENSOR_ON_OFF
from ..ha_model.device import HADeviceMixin
from ..home_model.commands import LoadCommand, CMD_ON, CMD_OFF, CMD_IDLE
from ..home_model.constraints import TimeBasedSimplePowerLoadConstraint
from ..home_model.load import AbstractLoad
from homeassistant.const import Platform, SERVICE_TURN_ON, SERVICE_TURN_OFF, STATE_UNKNOWN, STATE_UNAVAILABLE


class QSOnOffMode(StrEnum):
    ON_OFF_MODE_OFF = "on_off_mode_off"
    ON_OFF_MODE_AUTO = "on_off_mode_auto"
    ON_OFF_MODE_DEFAULT = "on_off_mode_default"
    ON_OFF_MODE_ON = "on_off_mode_on"



_LOGGER = logging.getLogger(__name__)
class QSOnOffDuration(HADeviceMixin, AbstractLoad):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.on_off_mode = QSOnOffMode.ON_OFF_MODE_AUTO

        self.default_on_duration: float | None = 1.0
        self.default_on_finish_time: dt_time | None = dt_time(hour=0, minute=0, second=0)

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
        parent.update([ Platform.SENSOR, Platform.SWITCH, Platform.SELECT, Platform.TIME, Platform.NUMBER ])
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
            target={"entity_id": self.switch_entity}
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

    def get_proper_local_adapted_tomorrow(self, time: datetime) -> datetime:
        local_target_date = time.replace(tzinfo=pytz.UTC).astimezone(tz=None)
        local_constraint_day = datetime(local_target_date.year, local_target_date.month, local_target_date.day)
        local_tomorrow = local_constraint_day + timedelta(days=1)
        return local_tomorrow.replace(tzinfo=None).astimezone(tz=pytz.UTC)

    async def check_load_activity_and_constraints(self, time: datetime) -> bool:

        if self.on_off_mode == QSOnOffMode.ON_OFF_MODE_OFF.value:
            # remove all constraints if any
            self.reset()
            return False
        else:
            do_add_constraint = False
            target_value = 0
            from_user = False
            end_schedule = None
            if self.on_off_mode == QSOnOffMode.ON_OFF_MODE_ON.value:
                end_schedule = self.get_proper_local_adapted_tomorrow(time)
                target_value = 25*3600.0 # 25 hours, more than a day will force the load to be on
                do_add_constraint = True
                from_user = True # not sure if it is needed
            elif self.on_off_mode == QSOnOffMode.ON_OFF_MODE_DEFAULT.value:
                if self.default_on_duration is not None and  self.default_on_finish_time is not None:
                    dt_now = time.replace(tzinfo=pytz.UTC).astimezone(tz=None)
                    next_time = datetime(year=dt_now.year,
                                         month=dt_now.month,
                                         day=dt_now.day,
                                         hour=self.default_on_finish_time.hour,
                                         minute=self.default_on_finish_time.minute,
                                         second=self.default_on_finish_time.second)
                    next_time = next_time.astimezone(tz=None)
                    if next_time < dt_now:
                        next_time = next_time + timedelta(days=1)

                    end_schedule = next_time.replace(tzinfo=None).astimezone(tz=pytz.UTC)

                    target_value = self.default_on_duration*3600.0

                    do_add_constraint = True


            else:
                start_schedule, end_schedule = await self.get_next_scheduled_event(time)

                if start_schedule is not None and end_schedule is not None:
                    do_add_constraint = True
                    target_value = (end_schedule - start_schedule).total_seconds()

            if do_add_constraint:

                type = CONSTRAINT_TYPE_MANDATORY_END_TIME
                if self.load_is_auto_to_be_boosted:
                    type = CONSTRAINT_TYPE_FILLER
                elif self.qs_best_effort_green_only is True:
                    type = CONSTRAINT_TYPE_FILLER  # will be after battery filling

                load_mandatory = TimeBasedSimplePowerLoadConstraint(
                        type=type,
                        time=time,
                        load=self,
                        from_user=from_user,
                        end_of_constraint=end_schedule,
                        power=self.power_use,
                        initial_value=0,
                        target_value=target_value
                )

                res = self.push_unique_and_current_end_of_constraint_from_agenda(time, load_mandatory)

                return res

        return False