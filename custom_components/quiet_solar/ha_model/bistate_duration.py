import logging
from abc import abstractmethod
from datetime import datetime
from datetime import time as dt_time


from ..const import CONSTRAINT_TYPE_MANDATORY_END_TIME, CONSTRAINT_TYPE_FILLER_AUTO
from ..ha_model.device import HADeviceMixin
from ..home_model.commands import LoadCommand, CMD_ON, CMD_OFF, CMD_IDLE
from ..home_model.constraints import TimeBasedSimplePowerLoadConstraint
from ..home_model.load import AbstractLoad
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE



bistate_modes = [
"bistate_mode_auto",
"bistate_mode_exact_calendar",
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

        self.override_duration: float | None  = MAX_USER_OVERRIDE_DURATION_S // 3600

        # to be overcharged by the child class
        self._state_on = "on"
        self._state_off = "off"
        self._bistate_mode_on = "bistate_mode_on"
        self._bistate_mode_off = "bistate_mode_off"
        self.bistate_entity = self.switch_entity

        self.is_load_time_sensitive = True

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

        if command.is_off_or_idle():
            return self._state_off
        else:
            return self._state_on

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

    async def execute_command(self, time: datetime, command: LoadCommand) -> bool | None:

        override_state = self.external_user_initiated_state
        ct = self.get_current_active_constraint(time)
        if ct is not None and ct.load_param is not None and self.external_user_initiated_state is not None:
            if not ct.is_mandatory:
                # if the override constraint is no more mandatory
                if command.is_off_or_idle():
                    override_state = None

        if override_state is not None:
            _LOGGER.info(
                f"External state set...intercept execute_command {command} for load {self.name} to stay in state {override_state}")

            state = self.hass.states.get(self.bistate_entity)

            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                pass
            elif state.state == override_state:
                return True

        return await self.execute_command_system(time, command, override_state)

    @abstractmethod
    async def execute_command_system(self, time: datetime, command: LoadCommand, state:str|None) -> bool | None:
        """ execute the command on the system """

    async def check_load_activity_and_constraints(self, time: datetime) -> bool:

        do_force_next_solve = False

        bistate_mode = self.bistate_mode
        has_a_running_override = False

        # we want to check that the load hasn't been changed externally from the system:
        if self.is_load_command_set(time) and self.support_user_override():
            # we need to know if the state we have is compatible with the current command
            # well more if it has been set ON or any other stuff externally so that we don't want to reset it to OFF
            # because the user wanted to force the state of the load
            # Ex : I have an HVAC that I manually open at 8pm and I don't want the system to close it because he thinks
            # it should be closed because of electricity price or any other stuffs
            # to maintain that:
            # - we detect that the current command is not one that has been set by the system
            # - we store this command and state change time
            # if not done we create a constraint,marked as user, with the proper command detected parameter (ex for an HVAC it could be multiple)
            if self.external_user_initiated_state_time is not None and (time - self.external_user_initiated_state_time).total_seconds() > (3600.0*self.self.override_duration):
                _LOGGER.info(
                    f"External state time is long, reset from {self.external_user_initiated_state} for load {self.name} ")
                # we need to reset the external user initiated state
                self.reset_override_state_and_set_reset_ask_time(time)
                do_force_next_solve = True
                self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None # a proper constraint re-evaluation for th eload will be done "normally"
            else:

                if self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done is not None:
                    # do nothing below, just ask for a proper constraint evaluation as an asked to not be overriden one:
                    has_a_running_override = False
                    do_force_next_solve = True
                    self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None
                else:
                    override_constraint = None
                    for i, ct in enumerate(self._constraints):
                        if ct.from_user and ct.load_param is not None:  # and ct.load_param == state.state:
                            # we do have already a constraint for this override state
                            override_constraint = ct
                            has_a_running_override = True
                            break

                    state = self.hass.states.get(self.bistate_entity)
                    is_command_overridden = False
                    expected_state = "UNKNOWN"
                    expected_state_running = "UNKNOWN"
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

                        if (expected_state is not None and state.state == expected_state) or (
                                expected_state_running is not None and state.state == expected_state_running):
                            is_command_overridden = False
                        else:
                            is_command_overridden = True


                    if self.asked_for_reset_user_initiated_state_time is not None:
                        if (time - self.asked_for_reset_user_initiated_state_time).total_seconds() < min(
                                USER_OVERRIDE_STATE_BACK_DURATION_S, (3600.0 * self.self.override_duration) / 2.0):
                            # small time window after asking for reset, do not consider the command overridden
                            if is_command_overridden is False:
                                # great no more override already after the last ask to stop override, reset the timer
                                self.asked_for_reset_user_initiated_state_time = None

                            # too soon to launch an override again
                            is_command_overridden = False
                        else:
                            # long enough ask to check the fact that the override should be finished
                            self.asked_for_reset_user_initiated_state_time = None


                    if is_command_overridden and (self.external_user_initiated_state is None or self.external_user_initiated_state != state.state):

                        _LOGGER.info(
                            f"check_load_activity_and_constraints: bistate OVERRIDE BY USER {state.state} for load {self.name} instead of {expected_state} {expected_state_running}")

                        # the user did something different ... just OVERRIDE the automation for a given time
                        has_a_running_override = True

                        self.external_user_initiated_state = state.state
                        self.external_user_initiated_state_time = time

                        # remove any overriden constraint if any
                        for i, ct in enumerate(self._constraints):
                            if ct.from_user and ct.load_param is not None:
                                # we do have already a constraint for this override state, kill it!
                                self._constraints[i] = None
                                self.set_live_constraints(time, self._constraints)
                                break


                        # we will create a constraint if the asked state is not idle ...
                        if self.expected_state_from_command(CMD_IDLE) == self.external_user_initiated_state:
                            # idle command
                            has_a_running_override = True
                            # push all constraints if feasible: ie move their start, keep their values? no override is override : user decision
                            self.reset()
                            do_force_next_solve = True
                        else:
                            end_schedule = time + datetime.timedelta(seconds=(3600.0*self.self.override_duration))
                            override_constraint = TimeBasedSimplePowerLoadConstraint(
                                type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
                                degraded_type=CONSTRAINT_TYPE_FILLER_AUTO,
                                time=time,
                                load=self,
                                load_param=self.external_user_initiated_state,
                                from_user=True,
                                end_of_constraint=end_schedule,
                                power=self.power_use,
                                initial_value=0,
                                target_value=3600.0*self.self.override_duration,
                                always_end_at_end_of_constraint=True
                            )

                            if self.push_live_constraint(time, override_constraint):
                                _LOGGER.info(
                                    f"check_load_activity_and_constraints: bistate load {self.name} pushed user override constraint")
                                do_force_next_solve = True

        if has_a_running_override is False:

            if bistate_mode == self._bistate_mode_off:
                # remove all constraints if any
                if len(self._constraints) > 0:
                    do_force_next_solve = True
                self.reset()
            else:
                do_add_constraint = False
                target_value = 0
                end_schedule = None
                has_user_forced_constraint = False
                start_schedule = None
                always_end_at_end_of_constraint = False
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
                        if bistate_mode == "bistate_mode_exact_calendar":
                            always_end_at_end_of_constraint = True
                            target_value += 600 # 10 mn of leg room to be sure the planner will fill all of this


                if do_add_constraint:

                    type = CONSTRAINT_TYPE_MANDATORY_END_TIME
                    if has_user_forced_constraint is False and self.is_best_effort_only_load():
                        type = CONSTRAINT_TYPE_FILLER_AUTO # will be after battery filling lowest priority

                    load_mandatory = TimeBasedSimplePowerLoadConstraint(
                            type=type,
                            degraded_type=CONSTRAINT_TYPE_FILLER_AUTO,
                            time=time,
                            load=self,
                            from_user=has_user_forced_constraint,
                            end_of_constraint=end_schedule,
                            power=self.power_use,
                            initial_value=0,
                            target_value=target_value,
                            always_end_at_end_of_constraint=always_end_at_end_of_constraint
                    )

                    do_force_next_solve = self.push_unique_and_current_end_of_constraint_from_agenda(time, load_mandatory) or do_force_next_solve

        return do_force_next_solve