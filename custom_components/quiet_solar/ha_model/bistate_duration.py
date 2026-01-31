import logging
from abc import abstractmethod
from collections import namedtuple
from datetime import datetime, timedelta
from datetime import time as dt_time

import pytz

from ..const import CONSTRAINT_TYPE_MANDATORY_END_TIME, CONSTRAINT_TYPE_FILLER_AUTO, SOLVER_STEP_S
from ..ha_model.device import HADeviceMixin
from ..home_model.commands import LoadCommand, CMD_ON, CMD_OFF, CMD_IDLE
from ..home_model.constraints import TimeBasedSimplePowerLoadConstraint, DATETIME_MAX_UTC
from ..home_model.load import AbstractLoad
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE



bistate_modes = [
"bistate_mode_auto",
"bistate_mode_exact_calendar",
"bistate_mode_default",
]

MAX_USER_OVERRIDE_DURATION_S = 8*3600
USER_OVERRIDE_STATE_BACK_DURATION_S = 60

ConstraintItemType = namedtuple("ConstraintItem", ["start_schedule", "end_schedule", "target_value", "has_user_forced_constraint", "agenda_push"], defaults=(None, None, 0.0, False, False))

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

        self._last_power_use_computation_time: datetime | None = None

    async def user_set_default_on_duration(self, float_value: float, for_init:bool = False):
        self.default_on_duration = float_value
        if for_init is False:
            time = datetime.now(pytz.UTC)
            if await self.do_run_check_load_activity_and_constraints(time):
                self.home.force_next_solve()
            await self.home.update_all_states(time)

    async def user_set_bistate_mode(self, option: str, for_init:bool = False):
        if option not in self.get_bistate_modes():
            _LOGGER.error(f"bistate_mode: {option} is not a valid bistate_mode")
            return
        self.bistate_mode = option
        if for_init is False:
            time = datetime.now(pytz.UTC)
            if await self.do_run_check_load_activity_and_constraints(time):
                self.home.force_next_solve()
            await self.home.update_all_states(time)

    @property
    def power_use(self):
        power = self._power_use_conf

        if power is None:
            return None

        power = float(power)

        if self.accurate_power_sensor is not None:
            time = datetime.now(tz=pytz.UTC)
            if self._last_power_use_computation_time is None or (time - self._last_power_use_computation_time).total_seconds() > SOLVER_STEP_S:

                power = self.get_average_sensor(self.accurate_power_sensor,num_seconds=4*3600, time=time, min_val=self._power_use_conf/3.0)
                if power is None or power == 0.0:
                    power = self.get_average_sensor(self.accurate_power_sensor, num_seconds=8 * 3600, time=time,
                                                    min_val=self._power_use_conf / 3.0)
                if power is None or power == 0.0:
                    power = self.get_average_sensor(self.accurate_power_sensor, num_seconds=24 * 3600, time=time,
                                                    min_val=self._power_use_conf / 3.0)
                if power is None or power == 0.0:
                    power = self._power_use_conf

                self._last_power_use_computation_time = time

                _LOGGER.info(f"power_use: recomputation for {self.name} to {power} (conf:{self._power_use_conf})")


        return float(power)

    @power_use.setter
    def power_use(self, power: float | None):
        self._power_use_conf = power

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
        """ check the load activity and set proper constraints """
        do_force_next_solve = False

        bistate_mode = self.bistate_mode
        override_constraint = None
        do_push_constraint_after = None

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
            if self.external_user_initiated_state_time is not None and (time - self.external_user_initiated_state_time).total_seconds() > (3600.0*self.override_duration):
                _LOGGER.info(
                    f"External state time is long, reset from {self.external_user_initiated_state} for load {self.name} ")
                # we need to reset the external user initiated state
                self.reset_override_state_and_set_reset_ask_time(time)
                do_force_next_solve = True
                self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None # a proper constraint re-evaluation for the load will be done "normally"
                self.command_and_constraint_reset() #remove any constraint if any we will add it back if needed below
            else:

                if self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done is not None:
                    # do nothing below, just ask for a proper constraint evaluation to kill the current override::
                    has_a_running_override = False
                    do_force_next_solve = True
                    self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None
                    self.command_and_constraint_reset()  # remove any constraint if any we will add it back if needed below
                else:

                    for i, ct in enumerate(self._constraints):
                        if ct.load_param is not None and ct.load_info is not None and ct.load_info.get("originator","") == "user_override":
                            # we do have already a constraint for this override state
                            override_constraint = ct
                            break

                    state = self.hass.states.get(self.bistate_entity)
                    is_command_overridden_state_changed = False
                    expected_state = "UNKNOWN"
                    expected_state_running = "UNKNOWN"
                    current_state = None

                    if state is not None and state.state not in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                        current_state = state.state

                    if current_state is not None:

                        if self.external_user_initiated_state is not None and current_state == self.external_user_initiated_state:
                            # we are still in the same override ... all is good
                            is_command_overridden_state_changed = False
                        elif self.external_user_initiated_state is not None and current_state != self.external_user_initiated_state:
                            # hum : we changed the state from the override ... it is like a new override
                            # should we wait a bit to see if the user is changing back the state ?
                            is_command_overridden_state_changed = True
                        else:
                            # no current override ... check if the command is overridden
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

                            if (expected_state is not None and current_state == expected_state) or (
                                    expected_state_running is not None and current_state == expected_state_running):
                                is_command_overridden_state_changed = False
                            else:
                                is_command_overridden_state_changed = True


                    if self.asked_for_reset_user_initiated_state_time is not None:
                        if (time - self.asked_for_reset_user_initiated_state_time).total_seconds() < min(
                                float(USER_OVERRIDE_STATE_BACK_DURATION_S), (3600.0 * self.override_duration) / 2.0):
                            # small time window after asking for reset, do not consider the command overridden
                            # too soon to launch an override again
                            is_command_overridden_state_changed = False
                        else:
                            # long enough ask to check the fact that the override should be finished
                            self.asked_for_reset_user_initiated_state_time = None

                    if is_command_overridden_state_changed:

                        _LOGGER.info(
                            f"check_load_activity_and_constraints: bistate OVERRIDE BY USER {state.state} for load {self.name} instead of {expected_state} {expected_state_running}")

                        # the user did something different ... just OVERRIDE the automation for a given time
                        self.external_user_initiated_state = current_state
                        self.external_user_initiated_state_time = time

                        # remove any overridden constraint if any
                        self.command_and_constraint_reset()  # remove any constraint if any we will add it back if needed below

                        # we will create a constraint if the asked state is not idle ...
                        if self.expected_state_from_command(CMD_IDLE) == self.external_user_initiated_state:
                            # idle command
                            do_force_next_solve = True
                            # all constraint removed above : command_and_constraint_reset
                        else:
                            end_schedule = time + timedelta(seconds=(3600.0*self.override_duration))
                            override_constraint = TimeBasedSimplePowerLoadConstraint(
                                type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
                                degraded_type=CONSTRAINT_TYPE_FILLER_AUTO,
                                time=time,
                                load=self,
                                load_param=self.external_user_initiated_state,
                                load_info={"originator":"user_override"},
                                from_user=True,
                                end_of_constraint=end_schedule,
                                power=self.power_use,
                                initial_value=0,
                                target_value=3600.0*self.override_duration
                            )

                            if self.push_live_constraint(time, override_constraint):
                                _LOGGER.info(
                                    f"check_load_activity_and_constraints: bistate load {self.name} pushed user override constraint")
                                do_force_next_solve = True

                    if self.external_user_initiated_state is not None and self.expected_state_from_command(CMD_IDLE) == self.external_user_initiated_state:
                        do_push_constraint_after = self.external_user_initiated_state_time + timedelta(seconds=(3600.0*self.override_duration))

                    if override_constraint is not None and override_constraint.end_of_constraint != DATETIME_MAX_UTC:
                        do_push_constraint_after = override_constraint.end_of_constraint + timedelta(seconds=1)

        if bistate_mode == self._bistate_mode_off:
            # remove all constraints if any ... except if we have a running override
            if do_push_constraint_after is not None:
                # keep ONLY the override
                found_override = False
                removed_one = False
                for i, ct in enumerate(self._constraints):
                    if ct.load_param is not None and ct.load_info is not None and ct.load_info.get("originator",
                                                                                                   "") == "user_override":
                        # we do have already a constraint for this override state
                        found_override = True
                    else:
                        # remove this constraint that is not an override
                        self._constraints[i] = None
                        removed_one = True

                if found_override:
                    if removed_one:
                        do_force_next_solve = True
                        self._constraints = [c for c in self._constraints if c is not None]
                        self.set_live_constraints(time, self._constraints)
                else:

                    if override_constraint is not None:
                        do_force_next_solve = True
                        self.set_live_constraints(time, [override_constraint])
                    else:
                        if len(self._constraints) > 0:
                            do_force_next_solve = True
                        self.command_and_constraint_reset()
            else:
                if len(self._constraints) > 0:
                    do_force_next_solve = True
                self.command_and_constraint_reset()

            _LOGGER.debug(
                f"check_load_activity_and_constraints: bistate _bistate_mode_off {self._bistate_mode_off} for load {self.name}")
        else:

            constraints = []

            if bistate_mode == self._bistate_mode_on:
                end_schedule = self.get_proper_local_adapted_tomorrow(time)
                start_schedule = do_push_constraint_after
                ct = None
                if start_schedule is None or start_schedule < end_schedule:
                    ct = ConstraintItemType(start_schedule=start_schedule,
                                             end_schedule=end_schedule,
                                             target_value=25*3600.0, # 25 hours, more than a day will force the load to be on
                                             has_user_forced_constraint=True,
                                             agenda_push=False)
                    constraints.append(ct)
                _LOGGER.debug(
                    f"check_load_activity_and_constraints: bistate _bistate_mode_on {self._bistate_mode_on} for load {self.name} {ct}")
            elif bistate_mode == "bistate_mode_default":
                if self.default_on_duration is not None and  self.default_on_finish_time is not None:
                    end_schedule = self.get_next_time_from_hours(local_hours=self.default_on_finish_time, time_utc_now=time, output_in_utc=True)
                    start_schedule = do_push_constraint_after
                    ct = None
                    if start_schedule is None or start_schedule < end_schedule:
                        ct = ConstraintItemType(start_schedule=start_schedule,
                                                end_schedule=end_schedule,
                                                target_value=self.default_on_duration * 3600.0,
                                                has_user_forced_constraint=False,
                                                agenda_push=True)
                        constraints.append(ct)
                    _LOGGER.debug(
                        f"check_load_activity_and_constraints: bistate bistate_mode_default for load {self.name} {ct}")
            else:
                events = await self.get_next_scheduled_events(time=time, give_currently_running_event=True)

                for ev in events:
                    start_schedule, end_schedule = ev
                    if start_schedule is not None and end_schedule is not None:

                        if do_push_constraint_after is not None and end_schedule < do_push_constraint_after:
                            continue

                        if end_schedule <= time:
                            continue

                        # start_schedule = max(time, start_schedule) don't do that the constraint has to be stable! for comparison in the push constraints
                        if do_push_constraint_after is not None:
                            start_schedule = max(do_push_constraint_after, start_schedule)
                        if start_schedule >= end_schedule:
                            continue
                        target_value = (end_schedule - start_schedule).total_seconds()
                        if bistate_mode != "bistate_mode_exact_calendar":
                            # bistate mode auto, start should be None or after the overridden time if any
                            start_schedule = do_push_constraint_after

                        ct = ConstraintItemType(start_schedule=start_schedule,
                                                end_schedule=end_schedule,
                                                target_value=target_value,
                                                has_user_forced_constraint=False,
                                                agenda_push=True)
                        constraints.append(ct)
                        _LOGGER.debug(
                            f"check_load_activity_and_constraints: bistate calendar {bistate_mode} for load {self.name} {ct}")


            if len(constraints) > 0:

                agend_cts = []
                for ct in constraints:

                    type = CONSTRAINT_TYPE_MANDATORY_END_TIME
                    if ct.has_user_forced_constraint is False and self.is_best_effort_only_load():
                        type = CONSTRAINT_TYPE_FILLER_AUTO # will be after battery filling lowest priority

                    load_mandatory = TimeBasedSimplePowerLoadConstraint(
                            type=type,
                            degraded_type=CONSTRAINT_TYPE_FILLER_AUTO,
                            time=time,
                            load=self,
                            from_user=ct.has_user_forced_constraint,
                            start_of_constraint=ct.start_schedule,
                            end_of_constraint=ct.end_schedule,
                            power=self.power_use,
                            initial_value=0,
                            target_value=ct.target_value
                    )
                    if ct.agenda_push:
                        agend_cts.append(load_mandatory)
                    else:
                        push_res = self.push_live_constraint(time, load_mandatory)
                        do_force_next_solve = push_res or do_force_next_solve
                        if push_res:
                            _LOGGER.info(
                                f"check_load_activity_and_constraints: bistate load {self.name} pushed non-agenda constraint {load_mandatory}")

                if len(agend_cts) > 0:
                    push_res = self.push_agenda_constraints(time, agend_cts)
                    do_force_next_solve = push_res or do_force_next_solve
                    if push_res:
                        _LOGGER.info(
                            f"check_load_activity_and_constraints: bistate load {self.name} pushed agenda constraints {agend_cts}")

        return do_force_next_solve