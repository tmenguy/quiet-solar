import logging
from abc import abstractmethod
from datetime import datetime, timedelta
from enum import StrEnum
from datetime import time as dt_time

import pytz

from ..const import CONSTRAINT_TYPE_MANDATORY_END_TIME, CONSTRAINT_TYPE_FILLER, CONSTRAINT_TYPE_FILLER_AUTO
from ..ha_model.device import HADeviceMixin
from ..home_model.commands import LoadCommand, CMD_ON, CMD_OFF, CMD_IDLE
from ..home_model.constraints import TimeBasedSimplePowerLoadConstraint
from ..home_model.load import AbstractLoad
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE



bistate_modes = [
"bistate_mode_auto",
"bistate_mode_default",
]

MAX_USER_OVERRIDE_DURATION_S = 8*3600
USER_OVERRIDE_STATE_BACK_DURATION_S = 90

_LOGGER = logging.getLogger(__name__)
class QSBiStateDuration(HADeviceMixin, AbstractLoad):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bistate_mode = "bistate_mode_auto"

        self.default_on_duration: float | None = 1.0
        self.default_on_finish_time: dt_time | None = dt_time(hour=0, minute=0, second=0)


        # to be overcharged by the child class
        self._state_on = "on"
        self._state_off = "off"
        self._bistate_mode_on = "bistate_mode_on"
        self._bistate_mode_off = "bistate_mode_off"
        self.bistate_entity = self.switch_entity

        self.is_load_time_sensitive = True

    async def user_clean_and_reset(self):
        await super().user_clean_and_reset()
        await self.clean_and_reset()

    def get_power_from_switch_state(self, state : str | None) -> float | None:
        if state is None:
            return None
        if state == self._state_on:
            return self.power_use
        else:
            return 0.0

    def get_bistate_modes(self) -> list[str]:
        if not self.support_user_override():
            # do not allow the user to force the bistate mode in any case
            return bistate_modes
        return bistate_modes + [self._bistate_mode_on, self._bistate_mode_off]

    def support_green_only_switch(self) -> bool:
        if self.load_is_auto_to_be_boosted:
            return False
        return True

    def support_user_override(self) -> bool:
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

    @abstractmethod
    def get_virtual_current_constraint_translation_key(self) -> str | None:
       """ return the translation key for the current constraint """

    @abstractmethod
    def get_select_translation_key(self) -> str | None:
        """ return the translation key for the select """

    def expected_state_from_command(self, command: LoadCommand):
        if command is None:
            return None

        if command.is_like(CMD_ON):
            return self._state_on
        else:
            return self._state_off

    def expected_state_from_command_or_user(self, command: LoadCommand):
        if self.external_user_initiated_state is not None:
            return self.external_user_initiated_state
        return self.expected_state_from_command(command)

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool | None:
        """ check the states of the switch to see if the command is set """
        state = self.hass.states.get(self.bistate_entity)

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return None
        else:
            return state.state == self.expected_state_from_command_or_user(command)


    def get_proper_local_adapted_tomorrow(self, time: datetime) -> datetime:
        local_target_date = time.replace(tzinfo=pytz.UTC).astimezone(tz=None)
        local_constraint_day = datetime(local_target_date.year, local_target_date.month, local_target_date.day)
        local_tomorrow = local_constraint_day + timedelta(days=1)
        return local_tomorrow.replace(tzinfo=None).astimezone(tz=pytz.UTC)

    async def execute_command(self, time: datetime, command: LoadCommand) -> bool | None:
        if self.external_user_initiated_state is not None:
            # we say all is good ... but we don't do anything
            _LOGGER.info(
                f"External state set... forbid execute_command {command} for load {self.name} to stay in state {self.external_user_initiated_state}")
            return True
        return await self.execute_command_system(time, command)

    @abstractmethod
    async def execute_command_system(self, time: datetime, command: LoadCommand) -> bool | None:
        """ execute the command on the system """

    async def check_load_activity_and_constraints(self, time: datetime) -> bool:

        do_force_next_solve = False

        bistate_mode = self.bistate_mode

        # we want to check that the load hasn't been changed externally from the system:
        if self.is_load_command_set(time) and self.support_user_override():
            # we need to know if the state we have is compatible with the current command
            # well more if it has been set ON or any other stuff externally so that we don't want to reset it to OFF
            # because the user wanted to force the state of the load
            # Ex : I have an HVAC that I manually open at 8pm and I don't want the system to close it because he thinks
            # it should be closed because of electricity price or any other stuffs

            if self.external_user_initiated_state_time is not None and (time - self.external_user_initiated_state_time).total_seconds() > MAX_USER_OVERRIDE_DURATION_S:
                _LOGGER.info(
                    f"External state time is long, reset from {self.external_user_initiated_state} for load {self.name} ")
                # we need to reset the external user initiated state
                await self.async_reset_override_state()
                do_force_next_solve = True
            else:
                state = self.hass.states.get(self.bistate_entity)

                if state is not None and state.state not in [STATE_UNKNOWN, STATE_UNAVAILABLE]:

                    expected_state_running = expected_state = None
                    if self.current_command is None and self.running_command is None:
                        expected_state_running = expected_state = self.expected_state_from_command(CMD_IDLE)
                    else:
                        if self.current_command is not None:
                           expected_state = self.expected_state_from_command(self.current_command)

                        if self.running_command is not None:
                            expected_state_running = self.expected_state_from_command(self.running_command)

                        if expected_state is None:
                            expected_state = expected_state_running

                        if expected_state_running is None:
                            expected_state_running = expected_state

                    do_override_set = False
                    if (expected_state is not None and state.state == expected_state) or (expected_state_running is not None and state.state == expected_state_running):
                        do_override_set = False
                    elif self.external_user_initiated_state is None or self.external_user_initiated_state != state.state:
                        do_override_set = True


                    if self.asked_for_reset_user_initiated_state_time is not None:
                        if (time - self.asked_for_reset_user_initiated_state_time).total_seconds() < USER_OVERRIDE_STATE_BACK_DURATION_S:

                            if do_override_set is False:
                                # great no more override already after the last ask to stop override, reset the timer
                                self.asked_for_reset_user_initiated_state_time = None

                            # in all case still in a "short window" to ask for an override anyway
                            do_override_set = False
                        else:
                            self.asked_for_reset_user_initiated_state_time = None


                    # if the user did something different ... just OVERRIDE the automation for a given time
                    if do_override_set:
                        # we need to remember the state and the time
                        _LOGGER.info(
                            f"check_load_activity_and_constraints: OVERRIDE BY USER {state.state} for load {self.name} instead of {expected_state} ")

                        self.external_user_initiated_state = state.state
                        self.external_user_initiated_state_time = time

                        # remove all constraints if any, will be added again below
                        self.reset()
                        do_force_next_solve = True


                if self.external_user_initiated_state is not None:

                    # we do have forced "from the outside" a constraint that may be "infinite" so we way want to
                    # change the current constraint to expand it if needed to infinity (at minimum for the override
                    # time) or on the contrary: remove any constraint that may be set to put the bistate on if the
                    # user has set it to off, it is as if the user put the bistate_mode in a state
                    # execute_command and probe_if_command set has been "shunted" to do touch the state of the load
                    if self.external_user_initiated_state == self.expected_state_from_command_or_user(CMD_OFF):
                        bistate_mode = self._bistate_mode_off
                    else:
                        bistate_mode = self._bistate_mode_on


        if bistate_mode == self._bistate_mode_off:
            # remove all constraints if any
            self.reset()
        else:
            do_add_constraint = False
            target_value = 0
            end_schedule = None
            has_user_forced_constraint = False
            if bistate_mode == self._bistate_mode_on:
                has_user_forced_constraint = True
                end_schedule = self.get_proper_local_adapted_tomorrow(time)
                target_value = 25*3600.0 # 25 hours, more than a day will force the load to be on
                do_add_constraint = True
            elif bistate_mode == "bistate_mode_default":
                if self.default_on_duration is not None and  self.default_on_finish_time is not None:
                    end_schedule = self.get_next_time_from_hours(local_hours=self.default_on_finish_time, time_utc_now=time, output_in_utc=True)
                    target_value = self.default_on_duration * 3600.0
                    do_add_constraint = True
            else:
                start_schedule, end_schedule = await self.get_next_scheduled_event(time)

                if start_schedule is not None and end_schedule is not None:
                    do_add_constraint = True
                    target_value = (end_schedule - start_schedule).total_seconds()

            if do_add_constraint:

                type = CONSTRAINT_TYPE_MANDATORY_END_TIME
                if has_user_forced_constraint is False and (self.load_is_auto_to_be_boosted or self.qs_best_effort_green_only is True):
                    type = CONSTRAINT_TYPE_FILLER_AUTO # will be after battery filling lowest priority

                load_mandatory = TimeBasedSimplePowerLoadConstraint(
                        type=type,
                        time=time,
                        load=self,
                        from_user=has_user_forced_constraint,
                        end_of_constraint=end_schedule,
                        power=self.power_use,
                        initial_value=0,
                        target_value=target_value
                )

                do_force_next_solve = self.push_unique_and_current_end_of_constraint_from_agenda(time, load_mandatory) or do_force_next_solve

        return do_force_next_solve