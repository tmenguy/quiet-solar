import bisect
import copy
import logging
from datetime import datetime, timedelta
from enum import StrEnum

from typing import Any, Callable, Awaitable

import pytz
from datetime import time as dt_time

from homeassistant.components.button import SERVICE_PRESS
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry, device_registry
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE, SERVICE_TURN_OFF, SERVICE_TURN_ON, \
    ATTR_ENTITY_ID, ATTR_UNIT_OF_MEASUREMENT, UnitOfPower
from homeassistant.components import number, homeassistant
from homeassistant.util import slugify
from haversine import haversine, Unit

from .dynamic_group import QSDynamicGroup

try:
    from homeassistant.components.wallbox.const import ChargerStatus as WallboxChargerStatus
except:
    class WallboxChargerStatus(StrEnum):
        """Charger Status Description."""

        CHARGING = "Charging"
        DISCHARGING = "Discharging"
        PAUSED = "Paused"
        SCHEDULED = "Scheduled"
        WAITING_FOR_CAR = "Waiting for car demand"
        WAITING = "Waiting"
        DISCONNECTED = "Disconnected"
        ERROR = "Error"
        READY = "Ready"
        LOCKED = "Locked"
        LOCKED_CAR_CONNECTED = "Locked, car connected"
        UPDATING = "Updating"
        WAITING_IN_QUEUE_POWER_SHARING = "Waiting in queue by Power Sharing"
        WAITING_IN_QUEUE_POWER_BOOST = "Waiting in queue by Power Boost"
        WAITING_MID_FAILED = "Waiting MID failed"
        WAITING_MID_SAFETY = "Waiting MID safety margin exceeded"
        WAITING_IN_QUEUE_ECO_SMART = "Waiting in queue by Eco-Smart"
        UNKNOWN = "Unknown"


class QSOCPPv16ChargePointStatus(StrEnum):
    """
    Status reported in StatusNotification.req. A status can be reported for
    the Charge Point main controller (connectorId = 0) or for a specific
    connector. Status for the Charge Point main controller is a subset of the
    enumeration: Available, Unavailable or Faulted.

    States considered Operative are: Available, Preparing, Charging,
    SuspendedEVSE, SuspendedEV, Finishing, Reserved.
    States considered Inoperative are: Unavailable, Faulted.
    """

    available = "Available"
    preparing = "Preparing"
    charging = "Charging"
    suspended_evse = "SuspendedEVSE"
    suspended_ev = "SuspendedEV"
    finishing = "Finishing"
    reserved = "Reserved"
    unavailable = "Unavailable"
    faulted = "Faulted"


from ..const import CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, CONF_CHARGER_PAUSE_RESUME_SWITCH, \
    CONF_CHARGER_PLUGGED, CONF_CHARGER_MAX_CHARGE, CONF_CHARGER_MIN_CHARGE, \
    CONF_CHARGER_DEVICE_OCPP, CONF_CHARGER_DEVICE_WALLBOX, CONF_CHARGER_CONSUMPTION, CONF_CAR_CHARGER_MIN_CHARGE, \
    CONF_CAR_CHARGER_MAX_CHARGE, CONF_CHARGER_STATUS_SENSOR, CONF_CAR_BATTERY_CAPACITY, CONF_CALENDAR, \
    CHARGER_NO_CAR_CONNECTED, CONSTRAINT_TYPE_MANDATORY_END_TIME, \
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE, \
    SENSOR_CONSTRAINT_SENSOR_CHARGE, CONF_DEVICE_EFFICIENCY, \
    CONF_CHARGER_LONGITUDE, CONF_CHARGER_LATITUDE, CONF_DEFAULT_CAR_CHARGE, \
    CONSTRAINT_TYPE_FILLER, CONF_CHARGER_THREE_TO_ONE_PHASE_SWITCH, CONF_CHARGER_REBOOT_BUTTON, FORCE_CAR_NO_CHARGER_CONNECTED
from ..home_model.constraints import LoadConstraint, MultiStepsPowerLoadConstraintChargePercent, \
    MultiStepsPowerLoadConstraint, DATETIME_MAX_UTC
from ..ha_model.car import QSCar
from ..ha_model.device import HADeviceMixin, get_average_sensor, get_median_sensor
from ..home_model.commands import LoadCommand, CMD_AUTO_GREEN_ONLY, CMD_ON, copy_command, \
    CMD_AUTO_FROM_CONSIGN, CMD_AUTO_PRICE, CMD_AUTO_GREEN_CAP
from ..home_model.load import AbstractLoad, diff_amps, add_amps, is_amps_zero, are_amps_equal

_LOGGER = logging.getLogger(__name__)



CHARGER_MAX_POWER_AMPS_PRECISION_W = 100  # 100W precision for power
CHARGER_MIN_REBOOT_DURATION_S = 120


CHARGER_STATE_REFRESH_INTERVAL_S = 7
CHARGER_ADAPTATION_WINDOW_S = 30
CHARGER_CHECK_STATE_WINDOW_S = 15

CHARGER_STOP_CAR_ASKING_FOR_CURRENT_TO_STOP_S = 5 * 60 # to be sure the car is not asking for current anymore
CHARGER_LONG_CONNECTION_S = 60 * 10

CAR_CHARGER_LONG_RELATIONSHIP_S = 60 * 60

STATE_CMD_RETRY_NUMBER = 3
STATE_CMD_TIME_BETWEEN_RETRY_S = CHARGER_STATE_REFRESH_INTERVAL_S * 3


TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_OFF_TO_ON_S = 60 * 10
TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_ON_TO_OFF_S = 60 * 30
TIME_OK_BETWEEN_BUDGET_RESET_S = 30 * 60 # to check if a car is now really more important than others and is not charging
TIME_OK_SHOULD_BUDGET_RESET_S = min(TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_OFF_TO_ON_S, TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_ON_TO_OFF_S)



TIME_OK_BETWEEN_CHANGING_CHARGER_PHASES = 60*30
CHARGER_START_STOP_RETRY_S = 90




class QSChargerStates(StrEnum):
    PLUGGED = "plugged"
    UN_PLUGGED = "unplugged"


class QSStateCmd():

    def __init__(self, initial_num_in_out_immediate:int=0, command_retries_s:float=STATE_CMD_TIME_BETWEEN_RETRY_S):
        self.reset()
        self.initial_num_in_out_immediate = initial_num_in_out_immediate
        self.command_retries_s = command_retries_s

    def reset(self):
        self.value = None
        self._num_launched = 0
        self._num_set = 0
        self.last_time_set = None
        self.last_change_asked = None
        self.on_success_action_cb : Callable[[datetime, dict], Awaitable] | None = None
        self.on_success_action_cb_kwargs : dict | None = None

    def register_success_cb(self, on_success_action_cb : Callable[[datetime, dict], Awaitable], on_success_action_cb_kwargs : dict | None):
        self.on_success_action_cb = on_success_action_cb
        if on_success_action_cb_kwargs is None:
            on_success_action_cb_kwargs = {}
        self.on_success_action_cb_kwargs = on_success_action_cb_kwargs


    def set(self, value, time: datetime | None):
        if time is None:
            self.last_change_asked = None

        if self.value != value:
            _LOGGER.info(f"QSStateCmd set with change from {self.value} to {value} at {time}")
            num_set = self._num_set
            self.reset()
            self.value = value
            self.last_change_asked = time
            self._num_set = num_set + 1
            return True

        return False

    def is_ok_to_set(self, time: datetime, min_change_time: float):

        if self.last_change_asked is None or time is None:
            return True

        # for initial in/out!
        if self._num_set <= self.initial_num_in_out_immediate:
            return True

        if (time - self.last_change_asked).total_seconds() > min_change_time:
            return True

        return False

    async def success(self, time):
        self._num_launched = 0
        self.last_time_set = None

        if self.on_success_action_cb is not None:
            await self.on_success_action_cb(time=time, **self.on_success_action_cb_kwargs)

        self.on_success_action_cb = None
        self.on_success_action_cb_kwargs = None

    def is_ok_to_launch(self, value, time: datetime):

        self.set(value, time)

        if self._num_launched == 0:
            return True

        if self._num_launched > STATE_CMD_RETRY_NUMBER:
            return False

        if self.last_time_set is None:
            return True

        if self.last_time_set is not None and (
                time - self.last_time_set).total_seconds() > self.command_retries_s:
            return True

        return False

    def register_launch(self, value, time: datetime):
        self.set(value, time)
        self._num_launched += 1
        self.last_time_set = time


class QSChargerStatus(object):

    def __init__(self, charger):
        self.charger : QSChargerGeneric = charger
        self.accurate_current_power = None
        self.secondary_current_power = None
        self.command = None
        self.current_real_max_charging_amp = None
        self.current_active_phase_number = None
        self.possible_amps = None
        self.possible_num_phases = None
        self.budgeted_amp = None
        self.budgeted_num_phases = None
        self.charge_score = 0
        self.can_be_started_and_stopped = False

    def duplicate(self):
        d = QSChargerStatus(self.charger)
        d.accurate_current_power = self.accurate_current_power
        d.secondary_current_power = self.secondary_current_power
        d.command = self.command
        d.current_real_max_charging_amp = self.current_real_max_charging_amp
        d.current_active_phase_number = self.current_active_phase_number
        d.possible_amps = copy.copy(self.possible_amps)
        d.possible_num_phases = copy.copy(self.possible_num_phases)
        d.budgeted_amp = self.budgeted_amp
        d.budgeted_num_phases = self.budgeted_num_phases
        d.charge_score = self.charge_score
        d.can_be_started_and_stopped = self.can_be_started_and_stopped

    @property
    def name(self) -> str:
        car_name = "NO CAR"
        if self.charger.car:
            car_name = self.charger.car.name

        return f"{self.charger.name}/{car_name}"



    def get_amps_from_values(self, amp: float|int, num_phases:int) -> list[float|int]:

        if num_phases == 1:
            ret = [0.0, 0.0, 0.0]
            ret[self.charger.mono_phase_index] = amp
        else:
            ret = [amp, amp, amp]

        return ret

    def get_current_charging_amps(self) -> list[float|int]:
        return self.get_amps_from_values(self.current_real_max_charging_amp, self.current_active_phase_number)


    def get_budget_amps(self) -> list[float|int]:
        return self.get_amps_from_values(self.budgeted_amp, self.budgeted_num_phases)

    def update_amps_with_delta(self, from_amps:list[float|int],  num_phases:int, delta:int|float) -> list[float|int]:
        return self.charger.update_amps_with_delta(from_amps=from_amps, delta=delta, is_3p=num_phases==3)

    def get_diff_power(self, old_amp, old_num_phases, new_amp, new_num_phases) -> float | None:

        diff_power = self.charger.get_delta_dampened_power(old_amp, old_num_phases, new_amp, new_num_phases)

        if diff_power is None:
            _LOGGER.error(f"get_diff_power: diff_power is None for {self.name} old_amp {old_amp} old_num_phases {old_num_phases} new_amp {new_amp} new_num_phases {new_num_phases}")

        return diff_power


    def get_amps_phase_switch(self, from_amp:int | float, from_num_phase:int, delta_for_borders=0) -> tuple[float|int, int, list[float|int]]:


        if from_num_phase == 1:
            try_amps = from_amp // 3
            to_phase = 3
        else:
            try_amps = from_amp * 3
            to_phase = 1

        try_amps = min(try_amps, self.possible_amps[-1] + delta_for_borders)
        if from_amp > 0:
            if self.possible_amps[0] == 0:
                try_amps = max(try_amps,
                               self.possible_amps[1] + delta_for_borders)  # will be decreased/increased later
            else:
                try_amps = max(try_amps,
                               self.possible_amps[0] + delta_for_borders)  # will be decreased/increased later

        return try_amps, to_phase, self.get_amps_from_values(try_amps, to_phase)


    # it will try to get to the smallest consumption increase possible, in number of amps in the circuit
    # by juggling between phase switch if needed
    def can_change_budget(self, allow_state_change=True, allow_phase_change=False, increase=True) -> tuple[float|None, int]:

        if self.budgeted_amp is None:
            return None, self.possible_num_phases[-1]


        def _try_amp_decrease(self, amp, allow_state_change):
            next_amp = None
            if self.possible_amps[0] == 0:
                if len(self.possible_amps) > 1:
                    if amp > self.possible_amps[1]:
                        next_amp = amp - 1
                    else:
                        if amp != 0 and allow_state_change:
                            next_amp = 0
            else:
                if amp > self.possible_amps[0]:
                    next_amp = amp - 1
            return next_amp

        def _try_amp_increase(self, amp, allow_state_change):
            next_amp = None
            if self.possible_amps[-1] > 0:
                if amp == 0:
                    if allow_state_change:
                        if self.possible_amps[0] == 0:
                            if len(self.possible_amps) > 1:
                                next_amp = self.possible_amps[1]
                        else:
                            next_amp = self.possible_amps[0]
                elif amp < self.possible_amps[-1]:
                    next_amp = amp + 1
            return next_amp

        if increase:
            probe_amp_cb = _try_amp_increase
            delta_first_adaptive_amp = -1
        else:
            probe_amp_cb = _try_amp_decrease
            delta_first_adaptive_amp = 1


        next_amp = probe_amp_cb(self, self.budgeted_amp, allow_state_change)
        next_num_phases = self.budgeted_num_phases

        current_all_amps = self.budgeted_amp * self.budgeted_num_phases

        # double check it is a real increase/decrease
        if next_amp is not None and ((increase and next_amp * next_num_phases <= current_all_amps)
                or (increase is False and next_amp * next_num_phases >= current_all_amps)):
            next_amp = None

        if allow_phase_change and len(self.possible_num_phases) > 1:
            # we can go up or down in phase change to get the minimum increment/decrease

            next_amp_with_phase_change, next_num_phases_with_phase_change, _ = self.get_amps_phase_switch(from_amp=self.budgeted_amp,
                                                                                     from_num_phase=self.budgeted_num_phases,
                                                                                     delta_for_borders=delta_first_adaptive_amp)

            all_amps_with_change = next_amp_with_phase_change * next_num_phases_with_phase_change

            if ((increase and all_amps_with_change > current_all_amps)
                    or (increase is False and all_amps_with_change < current_all_amps)):  # it is a real decrease/increase
                # we never know with the rounding to 3 phases, we may have actually a true increase or decrease
                pass
            else:
                next_amp_with_phase_change = probe_amp_cb(self, next_amp_with_phase_change, allow_state_change)
                if next_amp_with_phase_change is not None:

                    all_amps_with_change = next_amp_with_phase_change * next_num_phases_with_phase_change

                    if ((increase and all_amps_with_change > current_all_amps)
                            or (increase is False and all_amps_with_change < current_all_amps)): # it is a real decrease/increase
                        pass
                    else:
                        next_amp_with_phase_change = None

            if next_amp_with_phase_change is not None and (next_amp is None or abs(current_all_amps - all_amps_with_change) < abs(current_all_amps - (next_amp * next_num_phases))):
                    # we can phase switch as the amps change will be globally smaller
                    next_amp = next_amp_with_phase_change
                    next_num_phases = next_num_phases_with_phase_change

        return next_amp, next_num_phases

    def get_consign_amps_values(self, consign_is_minimum=True, add_tolerance=0.0) -> (list[int]|None, int|float):

        possible_num_phases = None

        if self.command.power_consign is not None and self.command.power_consign > 0:

            power = self.command.power_consign

            if consign_is_minimum:
                power = power*(1.0 - add_tolerance)
            else:
                power = power*(1.0 + add_tolerance)

            if self.charger.can_do_3_to_1_phase_switch():

                # avoid phase switch if not needed
                possible_num_phases = [self.current_active_phase_number]

                # if we can have "both" 1 and 3 phases, we will try to see if the current phase setup is compatible, if so
                # no need to phase switch and we keep the current need
                current_steps = self.charger.car.get_charge_power_per_phase_A(self.current_active_phase_number == 3)

                res_current = self.charger._get_amps_from_power_steps(current_steps, power, safe_border=False)

                if res_current is not None:
                    # we can keep the current phase setup
                    consign_amp = res_current
                else:
                    # need to phase switch to get the minimum asked power (either up or down)
                    switch_steps = self.charger.car.get_charge_power_per_phase_A(self.current_active_phase_number != 3)
                    res_switch = self.charger._get_amps_from_power_steps(switch_steps, power, safe_border=False)

                    if res_switch is None:
                        # well we stay as we are ... no phase switch
                        consign_amp = self.charger._get_amps_from_power_steps(current_steps, power, safe_border=False)
                    else:
                        # we can switch to the other phase setup
                        consign_amp = res_switch
                        if self.current_active_phase_number == 3:
                            possible_num_phases = [1]
                        else:
                            possible_num_phases = [3]

            else:
                native_power_steps = self.charger.car.get_charge_power_per_phase_A(self.charger.physical_3p)
                consign_amp = self.charger._get_amps_from_power_steps(native_power_steps, power, safe_border=True)
        else:
            if consign_is_minimum:
                consign_amp = self.charger.min_charge
            else:
                consign_amp = self.charger.max_charge

        consign_amp = int(max(consign_amp, self.charger.min_charge))
        consign_amp = int(min(consign_amp, self.charger.max_charge))

        return possible_num_phases, consign_amp


class QSChargerGroup(object):

    def __init__(self, dynamic_group: QSDynamicGroup ):
        self.dynamic_group : QSDynamicGroup = dynamic_group
        self._chargers : list[QSChargerGeneric] = []
        self.home = dynamic_group.home
        self.name = dynamic_group.name
        self.remaining_budget_to_apply = []
        self.know_reduced_state = None
        self.know_reduced_state_real_power = None
        self._last_time_reset_budget_done : datetime | None = None
        self._last_time_should_reset_budget_received: datetime | None = None

        self.charger_consumption_W = 0.0
        for device in dynamic_group._childrens:
            if isinstance(device, QSChargerGeneric):
                self._chargers.append(device)
                self.charger_consumption_W += device.charger_consumption_W

        self.dync_group_chargers_only = False
        if len(self._chargers) == len(dynamic_group._childrens):
            self.dync_group_chargers_only = True

    def dampening_power_value_for_car_consumption(self, value: float | None) -> float | None:
        if value is None:
            return None

        if abs(value) < self.charger_consumption_W:
            return 0.0
        else:
            return value


    async def ensure_correct_state(self, time: datetime, probe_only=False) -> (list[QSChargerStatus], datetime|None):

        verified_correct_state_time = None
        actionable_chargers = []
        for charger in self._chargers:

            if charger.qs_enable_device is False:
                continue

            res, handled_static, vcst = await charger.ensure_correct_state(time, probe_only=probe_only)

            _LOGGER.info(f"ensure_correct_state dyn group: {charger.name} {res}/{handled_static}")

            if handled_static:
                continue

            if res is None:
                _LOGGER.warning(f"ensure_correct_state dyn group: {charger.name} res is None, so not available, ignore it")
                continue

            if res is False:
                return [], None

            # if vcst is None it means the charger is probably unplugged for a long enough time: it is not a problem
            if vcst is not None:
                if verified_correct_state_time is None:
                    verified_correct_state_time = vcst
                elif vcst > verified_correct_state_time:
                    verified_correct_state_time = vcst


            cs = charger.get_stable_dynamic_charge_status(time)
            if cs is not None:
                actionable_chargers.append(cs)

        actionable_chargers = sorted(actionable_chargers, key=lambda cs: cs.charge_score, reverse=True)

        # we will know check the ones that had a possible state change blocked or approved, and be sure to align a bit better
        # those possible changes to not allow a more important one to be stopped because of a negative power budget
        # if a less important one will continue charging

        for i, cs in enumerate(actionable_chargers):

            if (cs.can_be_started_and_stopped
                    and i < len(actionable_chargers) - 1
                    and cs.possible_amps[0] == 0
                    and len(cs.possible_amps) > 1
                    and cs.current_real_max_charging_amp > 0):
                # ok this one may be stopped  .... check if all the "lower ones" can be stopped too, if not forbid its possible stop
                can_stop = True
                for j in range(i + 1, len(actionable_chargers)):
                    next_cs = actionable_chargers[j]
                    if next_cs.can_be_started_and_stopped and next_cs.possible_amps[0] > 0:
                        can_stop = False
                        break

                if can_stop is False:
                    # we can stop this one, so we will allow it to be stopped
                    # remove the 0 to not allow to stop it
                    cs.possible_amps = cs.possible_amps[1:]




        return actionable_chargers, verified_correct_state_time

    def get_budget_diffs(self, actionable_chargers: list[QSChargerStatus]):


        new_sum_amps = [0.0, 0.0, 0.0]
        current_amps = [0.0, 0.0, 0.0]
        diff_power = 0
        for cs in actionable_chargers:
            current_amps = add_amps(current_amps, cs.get_current_charging_amps())
            new_sum_amps = add_amps(new_sum_amps, cs.get_budget_amps())
            diff_power += cs.get_diff_power(cs.current_real_max_charging_amp, cs.current_active_phase_number, cs.budgeted_amp, cs.budgeted_num_phases)

        return diff_power, new_sum_amps, current_amps



    async def dyn_handle(self, time: datetime):

        # here we check all the chargers and they all need to be in a good state
        # could be plugged, unplugged whatever but in a good state
        _LOGGER.info(
            f"dyn_handle: START")

        actionable_chargers, verified_correct_state_time = await self.ensure_correct_state(time)

        if len(actionable_chargers) == 0:
            _LOGGER.info(
                f"dyn_handle: no actionable chargers, do nothing")
        else:
            if self.remaining_budget_to_apply:
                # in case we would have had increases that could go above the limits because done before the decrease
                _LOGGER.info(
                    f"dyn_handle: handling increasing budgets in a second phase")
                await self.apply_budgets(self.remaining_budget_to_apply, actionable_chargers, time, check_charger_state=True)
                self.remaining_budget_to_apply = []
            # only take decision if the state is "good" for a while CHARGER_ADAPTATION_WINDOW, for all active chargers
            elif verified_correct_state_time is not None and (time - verified_correct_state_time).total_seconds() > CHARGER_ADAPTATION_WINDOW_S:

                # all chargers are now in a correct state, stable enough to do a computation

                # let's compute the current power consummed by all the chargers : we may have it on the group itself ... if it is made of the
                # chargers only ... or we will have to compute it by asking each chargers
                current_real_cars_power = self.dynamic_group.get_median_sensor(self.dynamic_group.accurate_power_sensor,
                                                                               CHARGER_ADAPTATION_WINDOW_S, time)

                available_power = self.home.get_available_power_values(CHARGER_ADAPTATION_WINDOW_S, time)
                # the battery is normally adapting itself to the solar production, so if it is charging ... we will say that this power is available to the car

                if available_power:

                    last_p_mean = get_average_sensor(available_power[-len(available_power) // 2:], last_timing=time)
                    all_p_mean = get_average_sensor(available_power, last_timing=time)
                    last_p_median = get_median_sensor(available_power[-len(available_power) // 2:], last_timing=time)
                    all_p_median = get_median_sensor(available_power, last_timing=time)

                    full_available_home_power = min(last_p_mean, all_p_mean, last_p_median, all_p_median) #if positive we are exporting solar , negative we are importing from the grid

                    _LOGGER.info(
                        f"dyn_handle: full_available_home_power {full_available_home_power}W, {last_p_mean}, {all_p_mean}, {last_p_median}, {all_p_median}")

                    dampened_chargers = {}

                    for cs in actionable_chargers:

                        charger = cs.charger

                        if (charger._expected_charge_state.value is True and
                            cs.current_real_max_charging_amp >= charger.min_charge and
                            cs.accurate_current_power is not None ):

                            charger.update_car_dampening_value(time=time,
                                                               amperage=(cs.current_real_max_charging_amp, cs.current_active_phase_number),
                                                               amperage_transition=None,
                                                               power_value_or_delta=cs.accurate_current_power,
                                                               can_be_saved=((time - verified_correct_state_time).total_seconds() > 2 * CHARGER_ADAPTATION_WINDOW_S))
                            dampened_chargers[charger] = cs


                    a_charging_cs = None
                    num_charging_cs = 0

                    current_reduced_states= {}
                    for cs in actionable_chargers:
                        # get all the minimums
                        current_reduced_states[cs.charger] = (cs.current_real_max_charging_amp, cs.current_active_phase_number)

                        if cs.charger._expected_charge_state.value is True and cs.current_real_max_charging_amp >= cs.charger.min_charge:
                            a_charging_cs = cs
                            num_charging_cs += 1

                    # we can update a bit the dampening for the only one charging if we do have a global counter for the group
                    num_true_charging_cs = 0
                    charging = []
                    reason = ""
                    for c in self._chargers:

                        if c.qs_enable_device is False:
                            num_true_charging_cs = 0
                            # can't dampen if the charger is not enabled
                            reason = f"{c.name} qs disabled"
                            break

                        is_charger_zero = c.is_charging_power_zero(time=time, for_duration=CHARGER_ADAPTATION_WINDOW_S)
                        if is_charger_zero is None:
                            num_true_charging_cs = 0
                            # can't dampen if the charger is not enabled
                            reason = f"{c.name} is_charger_zero None"
                            break
                        elif is_charger_zero is False:
                            num_true_charging_cs += 1
                            charging.append(c.name)


                    if num_true_charging_cs <= 1 and num_charging_cs == 1 and current_real_cars_power is not None and a_charging_cs.charger not in dampened_chargers:
                        charger = a_charging_cs.charger
                        # num_true_charging_cs <= because the power could be 0 for the charger, so we can dampen it to change the min_charge of the car ...
                        _LOGGER.info(
                            f"dyn_handle: dampening simple case {charger.name} {current_real_cars_power}W for {a_charging_cs.current_real_max_charging_amp}A #phases{ a_charging_cs.current_active_phase_number}")
                        a_charging_cs.charger.update_car_dampening_value(time=time,
                                                                         amperage=(a_charging_cs.current_real_max_charging_amp, a_charging_cs.current_active_phase_number),
                                                                         amperage_transition=None,
                                                                         power_value_or_delta=current_real_cars_power,
                                                                         can_be_saved=((time - verified_correct_state_time).total_seconds() > 2 * CHARGER_ADAPTATION_WINDOW_S))
                        dampened_chargers[charger] = a_charging_cs
                    else:
                        _LOGGER.info(
                            f"dyn_handle: can't dampen simple case {num_true_charging_cs} {charging} {reason}")



                    # check the current state of the chargers to see if we can try to map the delta power properly
                    if current_real_cars_power is not None and self.know_reduced_state is not None and len(self.know_reduced_state) == len(current_reduced_states):
                        num_changes = 0
                        last_changed_charger = None
                        for c in self.know_reduced_state:
                            if c not in current_reduced_states:
                                last_changed_charger = None
                                break
                            else:
                                if self.know_reduced_state[c] != current_reduced_states[c]:
                                    num_changes += 1
                                    last_changed_charger = c

                        if last_changed_charger and num_changes == 1:
                            # great we do have a single change in the chargers, and we do have the previous cars power
                            # we can save the transition from self.know_reduced_state[c] to current_reduced_states[c]
                            delta_power = current_real_cars_power - self.know_reduced_state_real_power

                            do_dampen_transition = True
                            if last_changed_charger.dampening_power_value_for_car_consumption(current_real_cars_power) == 0 and current_reduced_states[last_changed_charger][0] >= last_changed_charger.min_charge:
                                # this is a transition to a 0 dampening
                                do_dampen_transition = False
                            elif last_changed_charger.dampening_power_value_for_car_consumption(self.know_reduced_state_real_power) == 0 and self.know_reduced_state[last_changed_charger][0] >= last_changed_charger.min_charge:
                                # this is a transition from a 0 dampening
                                do_dampen_transition = False

                            if do_dampen_transition:
                                _LOGGER.info(
                                    f"dyn_handle: dampening transition case {last_changed_charger.name} from {self.know_reduced_state[last_changed_charger]} to {current_reduced_states[last_changed_charger]} delta {delta_power}W ({current_real_cars_power} - {self.know_reduced_state_real_power})")
                                last_changed_charger.update_car_dampening_value(time=time,
                                                                                amperage=None,
                                                                                amperage_transition=(self.know_reduced_state[last_changed_charger], current_reduced_states[last_changed_charger]),
                                                                                power_value_or_delta=delta_power,
                                                                                can_be_saved=((time - verified_correct_state_time).total_seconds() > 2 * CHARGER_ADAPTATION_WINDOW_S))
                            else:
                                _LOGGER.info(
                                    f"dyn_handle: can't dampening {last_changed_charger.name} current_real_cars_power {current_real_cars_power}W, know_reduced_state_real_power {self.know_reduced_state_real_power}W, from {self.know_reduced_state[last_changed_charger]} to {current_reduced_states[last_changed_charger]} so no transition")


                    allow_budget_reset = False
                    if self._last_time_reset_budget_done is None or (time - self._last_time_reset_budget_done).total_seconds() > TIME_OK_BETWEEN_BUDGET_RESET_S:
                        allow_budget_reset = True

                    if self._last_time_should_reset_budget_received is not None and (time - self._last_time_should_reset_budget_received).total_seconds() > TIME_OK_SHOULD_BUDGET_RESET_S:
                        allow_budget_reset = True

                    success, should_do_reset_allocation, done_reset_budget = await self.budgeting_algorithm_minimize_diffs(actionable_chargers, full_available_home_power, allow_budget_reset, time)
                    if done_reset_budget:
                        self._last_time_reset_budget_done = time
                        self._last_time_should_reset_budget_received = None
                    elif should_do_reset_allocation:
                        if self._last_time_should_reset_budget_received is None:
                            self._last_time_should_reset_budget_received = time
                    else:
                        self._last_time_should_reset_budget_received = None

                    if success:
                        await self.apply_budget_strategy(actionable_chargers, current_real_cars_power, time)

                else:
                    _LOGGER.info(
                        f"dyn_handle: NO VALID AVAILABLE POWER, can't compute budgets")


    def _update_and_prob_for_amps_reduction(self, old_amps, new_amps, estimated_current_amps, time) -> tuple[bool, bool]:

        old_res, prev_diff_amps = self.dynamic_group.is_current_acceptable_and_diff(
            new_amps=old_amps,
            estimated_current_amps=estimated_current_amps,
            time=time
        )

        if old_res:
            return True, False

        new_res, diff_amps = self.dynamic_group.is_current_acceptable_and_diff(
                                                                                new_amps=new_amps,
                                                                                estimated_current_amps=estimated_current_amps,
                                                                                time=time
                                                                               )
        # if the reduction went in the right direction
        if max(diff_amps) < max(prev_diff_amps) or new_res:
            return new_res, True
        else:
            return False, False


    async def budgeting_algorithm_minimize_diffs(self, actionable_chargers, full_available_home_power, allow_budget_reset, time:datetime) -> (bool, bool, bool):

        actionable_chargers = sorted(actionable_chargers, key=lambda cs: cs.charge_score, reverse=True)

        do_reset_allocation = False
        should_do_reset_allocation = False

        # try to check if the "best" charger is in fact not charging when another one is charging
        # allow to stop another one to allow the best one to charge, do that only every hour or so or less
        if len(actionable_chargers) > 1:

            cs_to_stop_can_now = None
            cs_to_stop_by_forcing_it = None

            do_try_to_stop_other_chargers = False
            # the best one is not charging
            # if actionable_chargers[0].possible_amps[0] is not 0: ... it will be started by nature right after in the normal _do_prepare_budgets_for_algo
            if actionable_chargers[0].current_real_max_charging_amp == 0 and (actionable_chargers[0].possible_amps[0] == 0 and len(actionable_chargers[0].possible_amps) > 1):
                do_try_to_stop_other_chargers = True
                _LOGGER.info(
                    f"budgeting_algorithm_minimize_diffs: DO TRY RESET ALLOCATION for not charging best charger {actionable_chargers[0].name}")

            elif actionable_chargers[0].current_real_max_charging_amp > 0 and actionable_chargers[0].charger.qs_bump_solar_charge_priority:
                # if there is a bump solar charge priority, we may want to stop all the other chargers to allow the best one to charge
                do_try_to_stop_other_chargers = True
                _LOGGER.info(
                    f"budgeting_algorithm_minimize_diffs: DO TRY RESET ALLOCATION for bump solar best charger {actionable_chargers[0].name}")

            if do_try_to_stop_other_chargers:
                for i in range(1, len(actionable_chargers)):
                    if actionable_chargers[i].current_real_max_charging_amp > 0 and \
                            actionable_chargers[i].can_be_started_and_stopped:
                        # pick the last possible ones
                        if actionable_chargers[i].possible_amps[0] == 0:
                            # we can stop it now
                            cs_to_stop_can_now = actionable_chargers[i]
                        else:
                            cs_to_stop_by_forcing_it = actionable_chargers[i]

            if cs_to_stop_can_now is not None or cs_to_stop_by_forcing_it is not None:

                # ok we may have an opportunity to stop a charger to allow the best one to charge
                # check that the last time we did check that was more than an hour ago or so
                should_do_reset_allocation = True
                if allow_budget_reset:

                    if cs_to_stop_can_now is None and cs_to_stop_by_forcing_it is not None:

                        time_to_check = int(min(TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_OFF_TO_ON_S, TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_ON_TO_OFF_S)*0.9)

                        can_change_state = cs_to_stop_by_forcing_it.charger._expected_charge_state.is_ok_to_set(time, time_to_check)

                        if can_change_state:
                            _LOGGER.info(
                                f"budgeting_algorithm_minimize_diffs: DO RESET ALLOCATION, light forcing {cs_to_stop_by_forcing_it.name} to be allowed to stop")
                            cs_to_stop_can_now = cs_to_stop_by_forcing_it
                            cs_to_stop_by_forcing_it.possible_amps.insert(0, 0)
                        else:
                            _LOGGER.info(
                                f"budgeting_algorithm_minimize_diffs: DO RESET ALLOCATION FAILED FOR NOW, best charger {actionable_chargers[0].name} is not charging, while {cs_to_stop_by_forcing_it.name} is charging, but we cannot stop it now")

                    if cs_to_stop_can_now:
                        _LOGGER.info(
                            f"budgeting_algorithm_minimize_diffs: DO RESET ALLOCATION, best charger {actionable_chargers[0].name} is not charging, while {cs_to_stop_can_now.name} is")
                        do_reset_allocation = True

        current_amps, has_phase_changes, mandatory_amps = await self._do_prepare_budgets_for_algo(actionable_chargers, do_reset_allocation)

        # first bad case of amps overly booked by the solver for example...
        new_mandatory_amps = await self._shave_mandatory_budgets(actionable_chargers, current_amps, mandatory_amps, time)

        # ok in case of change redo the budgets allocations
        if are_amps_equal(new_mandatory_amps, mandatory_amps) is False:
            current_amps, has_phase_changes, mandatory_amps = await self._do_prepare_budgets_for_algo(actionable_chargers, do_reset_allocation)

        current_ok = await self._shave_current_budgets(actionable_chargers, time)

        if current_ok is False:
            _LOGGER.error(
                f"budgeting_algorithm_minimize_diffs: CAN'T SHAVE BUDGETS !!!!")
            return False, should_do_reset_allocation, False

        # ok we do have the "best" possible base for the chargers
        diff_power_budget, alloted_amps, current_amps = self.get_budget_diffs(actionable_chargers)

        power_budget = full_available_home_power - diff_power_budget
        # in case of "no reset" allocation, we will try to minimize the diffs
        # this algorithm will only try to move "a bit" the chargers to reach the power budget
        # if power_budget is negative : we need to go down and find the best charger to go down
        # if power_budget is positive : we need to go up to consume extra solar and find the best charger to go up
        # else:
        # try to get as close as possible to the power budget, without going over it

        if do_reset_allocation:
            allow_state_changes = [True]
        else:
            allow_state_changes = [False, True]

        if has_phase_changes:
            if do_reset_allocation:
                check_phase_change = [True, False]
            else:
                check_phase_change = [False, True]
        else:
            check_phase_change = [False]


        if power_budget < 0:
            increase = False
            stop_on_first_change = False
        else:
            increase = True
            if do_reset_allocation:
                stop_on_first_change = False
            else:
                stop_on_first_change = True


        # sort the charger according to their score, if increase put the most important to finish the charge first
        # if decrease: remove charging from less important first (lower score)
        if do_reset_allocation:
            actionable_chargers = sorted(actionable_chargers, key=lambda cs: cs.charge_score, reverse=True)
        else:
            actionable_chargers = sorted(actionable_chargers, key=lambda cs: cs.charge_score, reverse=increase)


        _LOGGER.info(
            f"budgeting_algorithm_minimize_diffs: {[cs.name for cs in actionable_chargers]} full_available_home_power {full_available_home_power} diff_power_budget {diff_power_budget} power_budget {power_budget}, increase {increase}, budget_alloted_amps {alloted_amps}")

        do_stop = False
        for allow_state_change in allow_state_changes:
            for allow_phase_change in check_phase_change:
                for cs in actionable_chargers:

                    local_stop_on_first_change = stop_on_first_change
                    if cs.command.is_like_one_of_cmds([CMD_AUTO_PRICE, CMD_AUTO_FROM_CONSIGN]):
                        local_stop_on_first_change = False

                    num_changes = 0
                    while True:

                        next_possible_budgeted_amp, next_possible_num_phases = cs.can_change_budget(allow_state_change=allow_state_change,
                                                                                                    allow_phase_change=allow_phase_change,
                                                                                                    increase=increase)
                        if next_possible_budgeted_amp is None:
                            _LOGGER.info(
                                f"budgeting_algorithm_minimize_diffs ({cs.name}): forbid change because of can_change_budget possible_amps {cs.possible_amps} current_amps {cs.get_current_charging_amps()} budgeted_amp {cs.get_budget_amps()} increase {increase}")
                        else:

                            diff_power = cs.get_diff_power(cs.budgeted_amp, cs.budgeted_num_phases, next_possible_budgeted_amp, next_possible_num_phases)

                            if diff_power is None:
                                next_possible_budgeted_amp = None
                                _LOGGER.info(
                                    f"budgeting_algorithm_minimize_diffs ({cs.name}): forbid change because of diff_power None power_budget {power_budget} diff_power {diff_power} increase {increase} from {cs.get_budget_amps()} to next_possible_budgeted_amp {next_possible_budgeted_amp} next_possible_num_phases {next_possible_num_phases}")
                            else:
                                new_alloted_amps = diff_amps(alloted_amps, cs.get_budget_amps())
                                new_alloted_amps = add_amps(new_alloted_amps, cs.get_amps_from_values(next_possible_budgeted_amp, next_possible_num_phases))

                                if increase:
                                    if power_budget - diff_power >= 0:
                                        # ok good change, we still have some power to give
                                        pass
                                    else:
                                        # no we exhausted too far the available solar budget
                                        _LOGGER.info(
                                            f"budgeting_algorithm_minimize_diffs ({cs.name}): forbid change because of power_budget {power_budget} diff_power {diff_power} increase {increase} from {cs.get_budget_amps()} to next_possible_budgeted_amp {next_possible_budgeted_amp} next_possible_num_phases {next_possible_num_phases}")

                                        next_possible_budgeted_amp = None

                                if next_possible_budgeted_amp is not None:
                                    if self.dynamic_group.is_current_acceptable(
                                                new_amps=new_alloted_amps,
                                                estimated_current_amps=current_amps,
                                                time=time
                                    ) is False:
                                        next_possible_budgeted_amp = None
                                        _LOGGER.info(
                                            f"budgeting_algorithm_minimize_diffs ({cs.name}): forbid change because of dynamic_group new_amps {new_alloted_amps} estimated_current_amps {current_amps}")

                                if next_possible_budgeted_amp is not None:

                                    power_budget -= diff_power
                                    alloted_amps = new_alloted_amps

                                    _LOGGER.info(
                                        f"budgeting_algorithm_minimize_diffs ({cs.name}): allowing change from {cs.budgeted_amp}A to {next_possible_budgeted_amp}A, new power_budget {power_budget}, diff_power {diff_power}, increase {increase}, new alloted_amps {alloted_amps}")
                                    cs.budgeted_amp = next_possible_budgeted_amp
                                    cs.budgeted_num_phases = next_possible_num_phases

                                    num_changes += 1

                                    if local_stop_on_first_change:
                                        do_stop = True

                                    if increase is False and power_budget >= 0:
                                        # we are back on track for solar or we reduced enough
                                        do_stop = True

                        if do_stop or next_possible_budgeted_amp is None:
                            # we can't change this charger anymore ... just stop here
                            break

                    if stop_on_first_change and num_changes > 0:
                        do_stop = True

                    if do_stop:
                        break

                if do_stop:
                    break

            if do_stop:
                break

        # optimize cost usage in case battery won't be used
        if full_available_home_power > 0:

            best_global_command = CMD_AUTO_GREEN_ONLY
            for cs in actionable_chargers:
                if cs.command.is_like_one_of_cmds([CMD_AUTO_PRICE,CMD_AUTO_FROM_CONSIGN]):
                    best_global_command = CMD_AUTO_PRICE
                    break

            if best_global_command == CMD_AUTO_PRICE and self.home.battery_can_discharge() is False:

                diff_power_budget, alloted_amps, current_amps = self.get_budget_diffs(actionable_chargers)

                # we will compute here if the price to take "more" power is better than the best
                # electricity rate we may have

                # check we have room to expand, and allocate amps
                if self.dynamic_group.is_current_acceptable(
                        new_amps=add_amps(alloted_amps, [1,1,1]),
                        estimated_current_amps=current_amps,
                        time=time
                ):

                    _LOGGER.info(f"dyn_handle: auto-price case")

                    best_price = self.home.get_best_tariff(time)
                    durations_eval_s = 2 * CHARGER_ADAPTATION_WINDOW_S
                    current_price = self.home.get_tariff(time, time + timedelta(seconds=durations_eval_s))

                    # find the smallest possible increment in power
                    smallest_power_increment = None
                    cs_to_update = None
                    new_amp_budget = None
                    next_num_phases_budget = None

                    # try to augment first the most important ones
                    actionable_chargers = sorted(actionable_chargers,
                                                 key=lambda cs: cs.charge_score, reverse=True)


                    for cs in actionable_chargers:

                        if cs.command.is_like_one_of_cmds([CMD_AUTO_GREEN_ONLY, CMD_AUTO_GREEN_CAP]):
                            # no need to try to augment on the green only: reserve it to the price ones
                            continue

                        next_budgeted_amp, next_budgeted_num_phases = cs.can_change_budget(allow_state_change=True,
                                                                                           allow_phase_change=True,
                                                                                           increase=True)

                        if next_budgeted_amp is not None:

                            diff_power = cs.get_diff_power(cs.budgeted_amp, cs.budgeted_num_phases,
                                                           next_budgeted_amp, next_budgeted_num_phases)

                            new_alloted_amps = diff_amps(alloted_amps, cs.get_budget_amps())
                            new_alloted_amps = add_amps(new_alloted_amps,
                                                                           cs.get_amps_from_values(
                                                                               next_budgeted_amp,
                                                                               next_budgeted_num_phases))

                            if self.dynamic_group.is_current_acceptable(
                                    new_amps=new_alloted_amps,
                                    estimated_current_amps=current_amps,
                                    time=time
                            ) and diff_power > 0:
                                if smallest_power_increment is None or diff_power < smallest_power_increment:
                                    smallest_power_increment = diff_power
                                    cs_to_update = cs
                                    new_amp_budget = next_budgeted_amp
                                    next_num_phases_budget = next_budgeted_num_phases

                    if smallest_power_increment is not None:

                        _LOGGER.info(
                            f"dyn_handle: auto-price extended charge {smallest_power_increment}")
                        additional_added_energy = (smallest_power_increment * durations_eval_s) / 3600.0
                        cost = (((diff_power_budget + smallest_power_increment - full_available_home_power) * durations_eval_s) / 3600.0) * current_price
                        cost_per_watt_h = cost / additional_added_energy

                        if cost_per_watt_h < best_price:
                            cs_to_update.budgeted_amp = new_amp_budget
                            cs_to_update.budgeted_num_phases = next_num_phases_budget

        return True, should_do_reset_allocation, do_reset_allocation

    async def _do_prepare_budgets_for_algo(self, actionable_chargers, do_reset_allocation):
        current_amps = [0.0, 0.0, 0.0]
        mandatory_amps = [0.0, 0.0, 0.0]
        has_phase_changes = False
        for cs in actionable_chargers:

            if do_reset_allocation:
                # put the minmum values for the amps
                cs.budgeted_amp = cs.possible_amps[0]
                cs.budgeted_num_phases = min(cs.possible_num_phases)
            else:
                # keep the current values as much as we can to minimize the diffs
                cs.budgeted_amp = max(cs.current_real_max_charging_amp, cs.possible_amps[0])
                if cs.current_active_phase_number in cs.possible_num_phases:
                    cs.budgeted_num_phases = cs.current_active_phase_number
                else:
                    cs.budgeted_num_phases = min(cs.possible_num_phases)  # get to one phase if unknown

            if len(cs.possible_num_phases) > 1:
                has_phase_changes = True

            current_amps = add_amps(current_amps, cs.get_current_charging_amps())
            mandatory_amps = add_amps(mandatory_amps,
                                      cs.get_amps_from_values(cs.possible_amps[0], cs.budgeted_num_phases))
        return current_amps, has_phase_changes, mandatory_amps

    async def _shave_current_budgets(self, actionable_chargers, time):

        current_ok = True
        diff_power_budget, alloted_amps, current_amps = self.get_budget_diffs(actionable_chargers)
        # check first if we are not already in a bad amps situation with the current amps
        if self.dynamic_group.is_current_acceptable(
                new_amps=alloted_amps,
                estimated_current_amps=current_amps,
                time=time
        ) is False:
            # ok we know it is possible to shave to reach the max phase current, due to the preparation before
            _LOGGER.info(
                f"_shave_current_budgets: group too much {self.name} {alloted_amps} > {self.dynamic_group.dyn_group_max_phase_current}")
            # start decrease the lower scores
            actionable_chargers = sorted(actionable_chargers, key=lambda cs: cs.charge_score)
            current_ok = False

            # Try to switch phase if possible to reach budget
            for cs in actionable_chargers:
                if cs.budgeted_num_phases == 1 and len(cs.possible_num_phases) > 1:

                    # we can lower the circuit amps by going 3 phases
                    # but we need to check if the phase of the current charger is at problem or not : else we will lower the wrong phase and increases the others

                    budget = cs.get_budget_amps()
                    new_alloted_amps = diff_amps(alloted_amps, budget)
                    try_amp, _, try_amps = cs.get_amps_phase_switch(from_amp=cs.budgeted_amp, from_num_phase=1)
                    new_alloted_amps = add_amps(new_alloted_amps, try_amps)

                    res_probe, do_update = self._update_and_prob_for_amps_reduction(old_amps=alloted_amps,
                                                                                    new_amps=new_alloted_amps,
                                                                                    estimated_current_amps=current_amps,
                                                                                    time=time)
                    if do_update:
                        alloted_amps = new_alloted_amps
                        cs.budgeted_amp = try_amp
                        cs.budgeted_num_phases = 3

                    if res_probe:
                        current_ok = True
                        break

            if current_ok is False:
                for allow_state_change in [False, True]:
                    while True:
                        has_shaved = False
                        for cs in actionable_chargers:
                            next_amp, next_num_phases = cs.can_change_budget(allow_state_change=allow_state_change,
                                                                             allow_phase_change=False,
                                                                             increase=False)
                            if next_amp is not None:
                                _LOGGER.info(
                                    f"_shave_current_budgets:  shaving {cs.name} from {cs.budgeted_amp} to {next_amp}")
                                alloted_amps = diff_amps(alloted_amps, cs.get_budget_amps())
                                cs.budgeted_amp = next_amp
                                cs.budgeted_num_phases = next_num_phases
                                alloted_amps = add_amps(alloted_amps, cs.get_budget_amps())
                                has_shaved = True

                            if self.dynamic_group.is_current_acceptable(
                                    new_amps=alloted_amps,
                                    estimated_current_amps=current_amps,
                                    time=time
                            ):
                                current_ok = True
                                break
                        if current_ok:
                            break

                        if has_shaved is False:
                            # nothing to be removed ....
                            break
                    if current_ok:
                        break
        return actionable_chargers, current_ok

    async def _shave_mandatory_budgets(self, actionable_chargers, current_amps, mandatory_amps, time):

        if self.dynamic_group.is_current_acceptable(
                new_amps=mandatory_amps,
                estimated_current_amps=current_amps,
                time=time
        ) is False:
            # ouch ... we have to lower the charge of some cars
            _LOGGER.warning(
                f"_shave_mandatory_budgets: need to shave some auto consign"
            )

            # first try to stop auto only
            possible_allotment_reached = False

            # first shave amps as much as we can to get to a proper level, starting with
            # less important chargers
            actionable_chargers = sorted(actionable_chargers, key=lambda cs: cs.charge_score)


            really_stoppable_shave_cs = [cs for cs in actionable_chargers if cs.can_be_started_and_stopped]
            first_stoppable_shave_cs = [cs for cs in actionable_chargers if
                                        ( cs.command.is_like_one_of_cmds([CMD_AUTO_GREEN_ONLY, CMD_AUTO_GREEN_CAP, CMD_AUTO_PRICE]))]
            shave_only_amps_cs = [cs for cs in actionable_chargers if cs.possible_amps[0] > cs.charger.min_charge]

            for cs_s in [really_stoppable_shave_cs, first_stoppable_shave_cs, shave_only_amps_cs, actionable_chargers]:

                if possible_allotment_reached:
                    break

                while possible_allotment_reached is False:
                    has_shaved = False
                    for cs in cs_s:

                        if cs.possible_amps[0] > 0:
                            # we can lower the charge by stopping it if needed

                            if cs.possible_amps[0] == cs.charger.min_charge:
                                insert_front = 0
                                delta_remove = cs.possible_amps[0]
                            else:
                                insert_front = cs.possible_amps[0] - 1
                                delta_remove = 1

                            updated_amps = cs.update_amps_with_delta(mandatory_amps, num_phases=cs.budgeted_num_phases,
                                                                     delta=-delta_remove)

                            res_probe, do_update = self._update_and_prob_for_amps_reduction(old_amps=mandatory_amps,
                                                                                            new_amps=updated_amps,
                                                                                            estimated_current_amps=current_amps,
                                                                                            time=time)
                            if do_update:
                                cs.possible_amps.insert(0, insert_front)
                                has_shaved = True
                                mandatory_amps = updated_amps

                            if res_probe is True:
                                possible_allotment_reached = True
                                break

                    if has_shaved is False:
                        # no more possibilities ...
                        break

                    if possible_allotment_reached:
                        break

        return mandatory_amps

    async def apply_budget_strategy(self, actionable_chargers, current_real_cars_power,time):


        if len(actionable_chargers) == 0:
            return

        if current_real_cars_power is None:
            self.know_reduced_state = None
            self.know_reduced_state_real_power = None
        else:
            if len(actionable_chargers) > 0:
                self.know_reduced_state = {}
                for cs in actionable_chargers:
                    self.know_reduced_state[cs.charger] = (cs.current_real_max_charging_amp, cs.current_active_phase_number)
                self.know_reduced_state_real_power = current_real_cars_power

        max_amps_in_worst_case_scenario = [0.0, 0.0, 0.0]
        current_amps = [0.0, 0.0, 0.0]

        for cs in actionable_chargers:
            curr_amps = cs.get_current_charging_amps()
            budget_amps = cs.get_budget_amps()
            current_amps = add_amps(current_amps, curr_amps)
            # it is really pessimistic here: was the worse of the worse when switching phase, it may add up
            max_curr_amps = [ max(curr_amps[i], budget_amps[i]) for i in range(3) ]
            max_amps_in_worst_case_scenario = add_amps(max_amps_in_worst_case_scenario, max_curr_amps)

        if self.dynamic_group.is_current_acceptable(
                    new_amps=max_amps_in_worst_case_scenario,
                    estimated_current_amps=current_amps,
                    time=time
            ) is False:

            # even if the sum of amps is ok, we may have the sum of charger amps during the change that is too high
            # we will split the chargers in 2 groups, the one that are decreasing in amps and the one that are increasing

            increasing_cs = []
            decreasing_cs = []
            remaining_cs = []

            # with phase switch beware, not so easy to know which of the charger will really decrease stuff
            # the easy case is if amps are decreasing while remaining on the same phase
            for cs in actionable_chargers:
                if cs.budgeted_amp == cs.current_real_max_charging_amp and cs.budgeted_num_phases == cs.current_active_phase_number:
                    decreasing_cs.append(cs)
                elif cs.budgeted_amp == 0 and  cs.current_real_max_charging_amp > 0:
                    decreasing_cs.append(cs)
                elif cs.budgeted_amp < cs.current_real_max_charging_amp and cs.budgeted_num_phases == cs.current_active_phase_number:
                    decreasing_cs.append(cs)
                elif cs.budgeted_amp < cs.current_real_max_charging_amp and cs.budgeted_num_phases == 1:
                    decreasing_cs.append(cs)
                elif cs.budgeted_amp > 0 and  cs.current_real_max_charging_amp == 0:
                    increasing_cs.append(cs)
                elif cs.budgeted_amp > cs.current_real_max_charging_amp and cs.budgeted_num_phases == cs.current_active_phase_number:
                    increasing_cs.append(cs)
                elif cs.budgeted_amp > cs.current_real_max_charging_amp and cs.budgeted_num_phases == 3:
                    increasing_cs.append(cs)
                else:
                    # the one to be allocated for first or second phase
                    remaining_cs.append(cs)

            # in remaining we have phase changes that may lead to an overhcarge of one circuit ex : a budget one phase on phase 1 with 32
            # and we have another one that was on phase 2 with 27 ... that is going 3 phases to will add 9 to the phase 1 that is already in 32,
            # so this phase change could't happen in the first one... do the one in remaining in the first phase ONLY

            # by construction the budgeted_num_phases and current_active_phase_number are different
            # start by adding the "reduction" part of a phase switch first, then do the increase partso we are sure all is well
            if len(remaining_cs) > 1:

                for cs in remaining_cs:
                    cs_copy = cs.duplicate()
                    if cs.budgeted_num_phases == 1 and cs.current_active_phase_number == 3:
                        cs_copy.budget_amp = min(cs.budgeted_amp, cs.current_real_max_charging_amp)
                        decreasing_cs.append(cs_copy)
                        increasing_cs.append(cs)
                    elif cs.budgeted_num_phases == 3 and cs.current_active_phase_number == 1:
                        cs_copy.budget_amp = min(cs.budgeted_amp, cs.current_real_max_charging_amp)
                        cs_copy.budgeted_num_phases = 1
                        decreasing_cs.append(cs_copy)
                        increasing_cs.append(cs)
                    else:
                        _LOGGER.error(
                            f"apply_budget_strategy: can't add remaining budget/current: {cs.budgeted_amp}/{cs.current_real_max_charging_amp} {cs.budgeted_num_phases}/{cs.current_active_phase_number}")
                        increasing_cs.append(cs)


            for cs in actionable_chargers:
                if cs.budgeted_amp > cs.current_real_max_charging_amp:
                    increasing_cs.append(cs)
                elif cs.budgeted_amp < cs.current_real_max_charging_amp:
                    decreasing_cs.append(cs)

            cs_to_apply = decreasing_cs
            self.remaining_budget_to_apply = increasing_cs

            _LOGGER.info(
                f"apply_budget_strategy: need to split updates {len(decreasing_cs)}/{len(increasing_cs)}")
        else:
            cs_to_apply = actionable_chargers
            self.remaining_budget_to_apply = []

        if cs_to_apply:
            await self.apply_budgets(cs_to_apply, actionable_chargers, time, check_charger_state=False)

    async def apply_budgets(self, cs_to_apply, actionable_chargers, time: datetime, check_charger_state=False):

        if check_charger_state:
            # we need to check again if the charger are still active !!!
            chargers = {}
            do_apply = False
            for cs in cs_to_apply:
                chargers[cs.charger] = cs

            num_ok = 0
            current_amps  = [0.0, 0.0, 0.0]
            new_amps = [0.0, 0.0, 0.0]
            for cs in actionable_chargers:

                curr_amps = cs.get_current_charging_amps()
                current_amps = add_amps(current_amps, curr_amps)

                if cs.charger in chargers:
                    new_amps = add_amps(new_amps,  chargers[cs.charger].get_current_charging_amps())
                    num_ok += 1
                    do_apply = True
                else:
                    new_amps = add_amps(new_amps, curr_amps)

            if num_ok != len(cs_to_apply):
                do_apply = False

            # check now that the new power won't be higher than the amp limit
            if self.dynamic_group.is_current_acceptable(new_amps=new_amps,
                                                        estimated_current_amps=current_amps,
                                                        time=time) is False:
                do_apply = False

            if do_apply is False:
                self.remaining_budget_to_apply = None
                self.know_reduced_state = None
                return

        for cs in cs_to_apply:

            init_state = cs.charger._expected_charge_state.value
            init_amp = cs.current_real_max_charging_amp
            init_phase_num = cs.current_active_phase_number

            new_amp = cs.budgeted_amp
            if new_amp < cs.charger.min_charge:
                new_amp = cs.charger.charger_default_idle_charge  # do not use charger min charge so next time we plug ...it may work
                new_state = False
            elif new_amp > cs.charger.max_charge:
                new_amp = cs.charger.max_charge
                new_state = True
            else:
                new_state = True

            new_num_phases = cs.budgeted_num_phases

            if init_state != new_state or (new_amp != init_amp and new_state) or new_num_phases != init_phase_num:
                _LOGGER.info(
                    f"{cs.name} new_amp {new_amp} / init_amp {init_amp} new_state {new_state} / init_state {init_state} new_num_phases {new_num_phases} / init_phase_num {init_phase_num}")
                _LOGGER.info(f"{cs.name} min charge {cs.charger.min_charge} max charge {cs.charger.max_charge}")

            if init_state != new_state:
                # normally the state change has been checked already to allow or not a change of state in the
                # get_stable_current_charge_status , and the allowed amps
                # if it is done anyway it is because of too high amps for exp...
                cs.charger._expected_charge_state.set(new_state, time)
                cs.charger.num_on_off += 1

                if cs.charger._expected_charge_state.last_change_asked is None:
                    _LOGGER.info(
                        f"Change State: new_state {new_state} delta None")
                else:
                    _LOGGER.info(
                        f"Change State: new_state {new_state} delta {(time - cs.charger._expected_charge_state.last_change_asked).total_seconds()}s")


            if new_amp is not None:
                cs.charger._expected_amperage.set(int(new_amp), time)

            if new_num_phases is not None:
                cs.charger._expected_num_active_phases.set(new_num_phases, time)

            await cs.charger._ensure_correct_state(time)


class QSChargerGeneric(HADeviceMixin, AbstractLoad):

    def __init__(self, **kwargs):
        self.charger_plugged = kwargs.pop(CONF_CHARGER_PLUGGED, None)
        self.charger_max_charging_current_number = kwargs.pop(CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, None)
        self.charger_pause_resume_switch = kwargs.pop(CONF_CHARGER_PAUSE_RESUME_SWITCH, None)
        self.charger_three_to_one_phase_switch = kwargs.pop(CONF_CHARGER_THREE_TO_ONE_PHASE_SWITCH, None)
        self.charger_max_charge = kwargs.pop(CONF_CHARGER_MAX_CHARGE, 32)
        self.charger_min_charge = kwargs.pop(CONF_CHARGER_MIN_CHARGE, 6)
        self._charger_default_idle_charge = min(self.charger_max_charge, self.charger_min_charge)
        self.charger_reboot_button = kwargs.pop(CONF_CHARGER_REBOOT_BUTTON, None)

        self.charger_latitude = kwargs.pop(CONF_CHARGER_LATITUDE, None)
        self.charger_longitude = kwargs.pop(CONF_CHARGER_LONGITUDE, None)

        self.charger_consumption_W = kwargs.pop(CONF_CHARGER_CONSUMPTION, 70)

        self.charger_status_sensor = kwargs.pop(CONF_CHARGER_STATUS_SENSOR, None)

        self._internal_fake_is_plugged_id = "is_there_a_car_plugged"

        self.car: QSCar | None = None
        self.user_attached_car_name: str | None = None
        self.car_attach_time: datetime | None = None

        self.charge_state = STATE_UNKNOWN

        self._last_charger_state_prob_time = None

        self._asked_for_reboot_at_time :datetime | None = None





        self.minimum_reboot_duration_s = CHARGER_MIN_REBOOT_DURATION_S

        super().__init__(**kwargs)

        data = {
            CONF_CAR_CHARGER_MIN_CHARGE: self.charger_min_charge,
            CONF_CAR_CHARGER_MAX_CHARGE: self.charger_max_charge,
            CONF_CAR_BATTERY_CAPACITY: 100000,
            CONF_CALENDAR: self.calendar,
            CONF_DEVICE_EFFICIENCY: self.efficiency,
            CONF_DEFAULT_CAR_CHARGE: 80.0
        }

        self._default_generic_car = QSCar(hass=self.hass, home=self.home, config_entry=None,
                                          name=f"{self.name} generic car", **data)

        self._inner_expected_charge_state: QSStateCmd | None = None
        self._inner_amperage: QSStateCmd | None = None
        self._inner_num_active_phases: QSStateCmd | None = None
        self.reset()

        self.initial_num_in_out_immediate = 0
        self.charge_start_stop_retry_s = CHARGER_START_STOP_RETRY_S


        self._unknown_state_vals = set()
        self._unknown_state_vals.update([STATE_UNKNOWN, STATE_UNAVAILABLE])
        self._unknown_state_vals.update(self.get_car_status_unknown_vals())

        self.attach_ha_state_to_probe(self.charger_status_sensor,
                                      is_numerical=False,
                                      state_invalid_values=self._unknown_state_vals,
                                      attach_unfiltered=True)

        self.charger_status_sensor_unfiltered = self.get_unfiltered_entity_name(self.charger_status_sensor)

        self.attach_ha_state_to_probe(self._internal_fake_is_plugged_id,
                                      is_numerical=False,
                                      non_ha_entity_get_state=self.is_plugged_state_getter)

        _LOGGER.info(f"Creating Charger: {self.name}")

        self._power_steps = []


    @property
    def charger_group(self) -> QSChargerGroup:

        charger_group = self.father_device.charger_group

        if charger_group is None:
            charger_group = self.father_device.charger_group = QSChargerGroup(self.father_device)

        return charger_group

    @property
    def qs_bump_solar_charge_priority(self) -> bool:
        if self.car is not None:
            return self.car.qs_bump_solar_charge_priority
        return False

    @qs_bump_solar_charge_priority.setter
    def qs_bump_solar_charge_priority(self, value: bool):
        if self.car is not None:
            self.car.qs_bump_solar_charge_priority = value

    @property
    def current_num_phases(self) -> int:
        if self.can_do_3_to_1_phase_switch() and self.physical_3p:
            state = self.hass.states.get(self.charger_three_to_one_phase_switch)

            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                res = 3
            else:
                if state.state == "on":
                    res = 1
                else:
                    res = 3
            return res
        else:
            res = self.physical_num_phases

        return res

    def get_phase_amps_from_power(self, power:float, is_3p=False) -> list[float | int]:

        steps = self.car.get_charge_power_per_phase_A(is_3p)
        resp = self._get_amps_from_power_steps(steps, power, safe_border=True)

        if is_3p:
            return [resp,resp,resp]
        else:
            ret = [0,0,0]
            ret[self.mono_phase_index] = resp
            return ret

    def get_device_amps_consumption(self, tolerance_seconds: float | None, time:datetime) -> list[float|int] | None:

        if self.is_charge_enabled(time=time):
            current_amp = self.get_max_charging_amp_per_phase()
            if current_amp is not None and current_amp >= self.min_charge and current_amp <= self.max_charge:
                if self.current_3p:
                    return [current_amp, current_amp, current_amp]
                else:
                    ret = [0.0, 0.0, 0.0]
                    ret[self.mono_phase_index] = current_amp
                    return ret

        return super().get_device_amps_consumption(tolerance_seconds, time)


    def _get_amps_from_power_steps(self, steps, power, safe_border=False) -> int|float|None:

        power = self.charger_group.dampening_power_value_for_car_consumption(power)

        if power is None or power == 0.0:
            return 0.0

        amp = self.min_charge + bisect.bisect_left(steps[self.min_charge:self.max_charge + 1], power)
        if amp <= self.min_charge:
            if safe_border is False and abs(steps[self.min_charge] - power) > CHARGER_MAX_POWER_AMPS_PRECISION_W:
                amp = None
            else:
                amp = self.min_charge
        elif amp > self.max_charge:
            if safe_border is False and abs(steps[self.max_charge] - power) > CHARGER_MAX_POWER_AMPS_PRECISION_W:
                amp = None
            else:
                amp = self.max_charge
        elif steps[amp] != power:
            if amp > self.min_charge and abs(steps[amp - 1] - power) < CHARGER_MAX_POWER_AMPS_PRECISION_W:
                amp = amp - 1
            elif amp < self.max_charge and abs(steps[amp + 1] - power) < CHARGER_MAX_POWER_AMPS_PRECISION_W:
                amp = amp + 1

        return amp

    def get_virtual_current_constraint_translation_key(self) -> str | None:
        return SENSOR_CONSTRAINT_SENSOR_CHARGE


    def get_update_value_callback_for_constraint_class(self, constraint:LoadConstraint) -> Callable[[LoadConstraint, datetime], Awaitable[tuple[float | None, bool]]] | None:

        class_name = constraint.__class__.__name__

        if class_name == MultiStepsPowerLoadConstraintChargePercent.__name__:
            return  self.constraint_update_value_callback_percent_soc

        return None

    def dampening_power_value_for_car_consumption(self, value: float) -> float | None:
        if value is None:
            return None

        if abs(value) < self.charger_consumption_W:
            return 0.0
        else:
            return value

    def is_plugged_state_getter(self, entity_id: str, time: datetime | None) -> (
            tuple[datetime | None, float | str | None, dict | None] | None):

        is_plugged, state_time = self.is_charger_plugged_now(time)

        if is_plugged is None:
            state = None
        else:
            if is_plugged:
                state = QSChargerStates.PLUGGED
            else:
                state = QSChargerStates.UN_PLUGGED

        return (state_time, state, {})

    @property
    def _expected_charge_state(self):
        if self._inner_expected_charge_state is None:
            self._inner_expected_charge_state = QSStateCmd(initial_num_in_out_immediate=self.initial_num_in_out_immediate, command_retries_s=self.charge_start_stop_retry_s)
        return self._inner_expected_charge_state

    @property
    def _expected_amperage(self):
        if self._inner_amperage is None:
            self._inner_amperage = QSStateCmd()
        return self._inner_amperage

    @property
    def _expected_num_active_phases(self):
        if self._inner_num_active_phases is None:
            self._inner_num_active_phases = QSStateCmd()
        return self._inner_num_active_phases

    def get_stable_dynamic_charge_status(self, time: datetime)-> QSChargerStatus | None:

        if self.qs_enable_device is False:
            _LOGGER.info(f"get_stable_dynamic_charge_status: {self.name} not enabled in qs")
            return None

        if self.car is None or self.is_not_plugged(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW_S):
            _LOGGER.info(f"get_stable_dynamic_charge_status: {self.name} no car or no plugged {self.car}")
            return None

        if self.is_charger_unavailable(time=time):
            _LOGGER.info(f"get_stable_dynamic_charge_status: {self.name} unavailable")
            return None

        handled = self._probe_and_enforce_stopped_charge_command_state(time, command=self.current_command, probe_only=True)

        if handled:
            # the charger is in a "static" state and is not consumming any current
            _LOGGER.info(f"get_stable_dynamic_charge_status: {self.name} None as _probe_and_enforce_stopped_charge_command_state")
            return None

        cs = QSChargerStatus(self)

        cs.accurate_current_power = self.get_median_sensor(self.accurate_power_sensor, CHARGER_ADAPTATION_WINDOW_S, time)

        cs.secondary_current_power = self.get_median_sensor(self.secondary_power_sensor, CHARGER_ADAPTATION_WINDOW_S, time)

        cs.current_real_max_charging_amp = self._expected_amperage.value

        cs.current_active_phase_number = self._expected_num_active_phases.value

        native_power_steps, _, _ = self.car.get_charge_power_per_phase_A(self.physical_3p)

        cs.command = self.current_command
        if cs.command.is_like(CMD_ON):
            cs.command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=native_power_steps[self.min_charge])


        # if the car charge is in state "not charging" mark the real charge as 0
        if self._expected_charge_state.value is False:
            cs.current_real_max_charging_amp = 0
        elif cs.current_real_max_charging_amp < self.min_charge:
            cs.current_real_max_charging_amp = 0


        current_state = True
        if cs.current_real_max_charging_amp == 0:
            current_state = False

        cs.can_be_started_and_stopped = False

        # we need to force a charge here from a given minimum
        if cs.command.is_like(CMD_AUTO_FROM_CONSIGN):

            possible_num_phases, min_amps = cs.get_consign_amps_values(consign_is_minimum=True)
            if possible_num_phases is None:
                possible_num_phases = [cs.current_active_phase_number]

            # 0 is not an option to not allow to stop the charge while we got this consign, in "minimum" consign or "price" consign
            possible_amps = [i for i in range(min_amps, self.max_charge + 1)]

        else:
            max_charge = self.max_charge
            possible_num_phases = None
            possible_amps = None

            if cs.command.is_like(CMD_AUTO_GREEN_CAP):

                if cs.command.power_consign == 0:
                    # forbid charge in that case ...
                    possible_num_phases = [cs.current_active_phase_number]
                    possible_amps = [0]

                else:
                    possible_num_phases, max_charge = cs.get_consign_amps_values(consign_is_minimum=False, add_tolerance=0.2)

            if possible_num_phases is None:
                possible_num_phases = [cs.current_active_phase_number]
                # check if we have the right to change phase number
                if self.can_do_3_to_1_phase_switch():
                    if self._expected_num_active_phases.is_ok_to_set(time, TIME_OK_BETWEEN_CHANGING_CHARGER_PHASES):
                        # we can change the number of phases
                        possible_num_phases = [1, 3]

            if possible_amps is None:
                run_list = [i for i in range(self.min_charge, max_charge + 1)]
                # check if it is on or off : can we change from off to on or on to off because of allowed changes
                # check if we need to wait a bit before changing the state of the charger

                cs.can_be_started_and_stopped = True
                if cs.command.is_like(CMD_AUTO_PRICE):
                    cs.can_be_started_and_stopped = False

                if current_state is False:
                    time_to_check = TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_OFF_TO_ON_S
                else:
                    time_to_check = TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_ON_TO_OFF_S

                can_change_state = self._expected_charge_state.is_ok_to_set(time, time_to_check)

                # that may force a change if can_stop False and Current_state is False
                possible_amps = run_list
                if cs.can_be_started_and_stopped:
                    if can_change_state:
                        possible_amps = [0]
                        possible_amps.extend(run_list)
                    elif current_state is False:
                        # we are off and we can't change the state, so we can only stay off
                        possible_amps = [0]


        cs.possible_amps = possible_amps
        cs.possible_num_phases = possible_num_phases

        ct = self.get_current_active_constraint(time)
        score = 0
        # if there is a time constraint that is not too far away : score boost
        # if this constraint is mandatory of as fast as possible : score boost
        score_boost_standard_ct = 1000
        score_boost_mandatory = 100*score_boost_standard_ct
        score_boost_as_fast_as_possible = 100*score_boost_mandatory
        if ct is not None:

            time_to_complete_h = None
            score_boost = score_boost_standard_ct
            if ct.as_fast_as_possible:
                score_boost = score_boost_as_fast_as_possible
            elif ct.is_mandatory:
                score_boost = score_boost_mandatory

            if ct.as_fast_as_possible:
                time_to_complete_h = 0
            elif ct.end_of_constraint < DATETIME_MAX_UTC:
                # there is a true end constraint
                time_to_complete_h = (ct.end_of_constraint - time).total_seconds()/3600.0

            if time_to_complete_h is not None and time_to_complete_h <= 24:
                score += (25 - time_to_complete_h) * score_boost


        if self.qs_bump_solar_charge_priority:
            score += score_boost_standard_ct * 50 # to be below any mandatory constraint for solar and as fast as possible constraint

        # give more to the ones with the lower car SOC percentage to reach their default target charge
        car_percent = self.car.get_car_charge_percent(time)
        if car_percent is None:
            _LOGGER.warning(f"get_stable_dynamic_charge_status: charging score: {self.name} for {self.car.name} car_percent is None")
            car_percent = 0.0

        car_battery_capacity = self.car.car_battery_capacity
        if car_battery_capacity is None or car_battery_capacity == 0:
            car_battery_capacity = 100000 # (100kWh)

        #convert in kwh
        car_battery_capacity = car_battery_capacity / 1000.0
        max_battery = 300
        score += max_battery - ((car_battery_capacity*car_percent)/100.0)


        _LOGGER.info(f"get_stable_dynamic_charge_status: {self.name} for {self.car.name} score:{score} possible_amps:{cs.possible_amps} possible_num_phases:{cs.possible_num_phases} current_amps:{cs.get_current_charging_amps()} command:{cs.command}")

        cs.charge_score = score

        return cs

    def get_normalized_score(self, ct:LoadConstraint, time:datetime, score_span:int) -> float:
        if self.car is None:
            return 0.0

        native_score_span_duration = 100
        native_score_span_battery = 1000
        native_score_span = native_score_span_duration * native_score_span_battery

        native_score_duration = 0
        max_duration_h = 23 # 4*max_duration_h + 3 must be < 100
        if ct is not None:

            time_to_complete_h = None


            if ct.as_fast_as_possible:
                time_to_complete_h = 0
            elif ct.end_of_constraint < DATETIME_MAX_UTC:
                # there is a true end constraint
                time_to_complete_h = (ct.end_of_constraint - time).total_seconds()/3600.0

            if time_to_complete_h is not None and time_to_complete_h <= max_duration_h - 1:
                native_score_duration = int(max_duration_h - time_to_complete_h)

                if ct.is_mandatory:
                    native_score_duration += 2*max_duration_h + 2


        if self.qs_bump_solar_charge_priority:
            native_score_duration += max_duration_h + 1 # to be above non mandatory but below in case of mandatory constraints

        native_score_duration = float(min(native_score_span_duration-1, int(native_score_duration)))

        # give more to the ones with the lower car SOC percentage to reach their default target charge
        car_percent = self.car.get_car_charge_percent(time)
        if car_percent is None:
            _LOGGER.warning(f"get_stable_dynamic_charge_status: charging score: {self.name} for {self.car.name} car_percent is None")
            car_percent = 0.0

        car_battery_capacity = self.car.car_battery_capacity
        if car_battery_capacity is None or car_battery_capacity == 0:
            car_battery_capacity = 100000 # (100kWh)

        #convert in kwh
        car_battery_capacity = car_battery_capacity / 1000.0
        max_battery = 400
        native_score_battery = float(int(max_battery - ((car_battery_capacity*car_percent)/100.0)))


        if score_span != native_score_span:

            score_span_duration = int((float(score_span) * float(native_score_span_duration))/float(native_score_span))
            score_duration = int((float(score_span_duration) * float(native_score_duration))/float(native_score_span_duration))

            score_span_battery = int((float(score_span) * float(native_score_span_battery))/float(native_score_span))
            score_battery = int((float(score_span_battery) * float(native_score_battery))/float(native_score_span_battery))
        else:
            score_span_duration = native_score_span_duration
            score_duration = native_score_duration

            score_span_battery = native_score_span_battery
            score_battery = native_score_battery

        score_duration = int(max(0, min(score_duration, score_span_duration-1)))
        score_battery = int(max(0, min(score_battery, score_span_battery-1)))

        score = score_duration * score_span_battery + score_battery
        return score






    def reset(self):
        _LOGGER.info(f"Charger reset {self.name}")
        super().reset()
        self.detach_car()
        self._reset_state_machine()
        self._asked_for_reboot_at_time = None
        self.qs_bump_solar_priority = False

    def reset_load_only(self):
        _LOGGER.info(f"Charger reset only load {self.name}")
        super().reset()

    def _reset_state_machine(self):
        self._verified_correct_state_time = None
        self._inner_expected_charge_state = None
        self._inner_amperage = None
        self._inner_num_active_phases = None

    def is_in_state_reset(self) -> bool:
        return (self._inner_expected_charge_state is None or
                self._inner_amperage is None or
                self._inner_num_active_phases is None or
                self._expected_charge_state.value is None or
                self._expected_amperage.value is None or
                self._expected_num_active_phases.value is None)

    async def on_device_state_change(self, time: datetime, device_change_type:str):

        if self.car:
            load_name = self.car.name
            mobile_app = self.car.mobile_app
            mobile_app_url = self.car.mobile_app_url
        else:
            load_name = self.name
            mobile_app = self.mobile_app
            mobile_app_url = self.mobile_app_url

        await self.on_device_state_change_helper(time, device_change_type, load_name=load_name, mobile_app=mobile_app, mobile_app_url=mobile_app_url)

    def get_car_score(self, car: QSCar,  time: datetime, cache:dict) -> float:

        # 0 to 9: part of the score for the plug only
        # 10 to 10*10000 - 1: part of the score for the plug_time only: 10000 span
        # 100000 to 1000*100000: part of the score for the distance to the charger 1000span
        plug_span = 10.0
        plug_time_span = 10000.0
        dist_span = 1000.0
        max_sore = plug_span*plug_time_span*dist_span*10.0

        score = None

        is_long_time_attached = False
        connected_time_delta = None
        if self.car is not None and self.car.name == car.name:
            connected_time_delta = time - self.car_attach_time
            is_long_time_attached = connected_time_delta > timedelta(seconds=CAR_CHARGER_LONG_RELATIONSHIP_S)



        if self.user_attached_car_name is not None:
            if self.user_attached_car_name != CHARGER_NO_CAR_CONNECTED:
                attached_car = self.home.get_car_by_name(self.user_attached_car_name)
                if attached_car is not None and car is not None:
                    if attached_car.name != car.name:
                        score = 0.0
                        _LOGGER.info(f"get_car_score: {car.name} for {self.name} score: {score} when {attached_car.name} user attached to charger")

                    else:
                        score = max_sore
                        _LOGGER.info(f"get_car_score: {car.name} for {self.name} score: {score}, Best Car from user selection")


        if is_long_time_attached:
            score = max_sore - 1.0
            _LOGGER.info(f"get_car_score: {car.name} for {self.name} score: {score}, is_long_time_attached to charger")


        if score is None and car.car_is_invited is False:

            score = 0.0

            score_plug_bump = 0
            car_plug_res = cache.get(car, {}).get("car_plug_res", "NOT_FOUND")
            if car_plug_res == "NOT_FOUND":
                car_plug_res = car.is_car_plugged(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW_S)
                if car_plug_res is None:
                    car_plug_res = car.is_car_plugged(time=time)
                cache.setdefault(car, {})["car_plug_res"] = car_plug_res

            if car_plug_res:
                score_plug_bump = 5

            score_plug_time_bump = 0
            if score_plug_bump > 0:
                charger_plugged_duration = cache.get(self,{}).get("charger_plugged_duration")
                if  charger_plugged_duration is None:
                    charger_plugged_duration = self.get_continuous_plug_duration(time)
                    if charger_plugged_duration is None:
                        charger_plugged_duration = -1.0
                    cache.setdefault(self,{})["charger_plugged_duration"] = charger_plugged_duration

                car_plugged_duration = cache.get(car,{}).get("car_plugged_duration")
                if car_plugged_duration is None:
                    car_plugged_duration = car.get_continuous_plug_duration(time)
                    if car_plugged_duration is None:
                        car_plugged_duration = -1.0
                    cache.setdefault(self,{})["car_plugged_duration"] = car_plugged_duration

                if charger_plugged_duration >= 0 and car_plugged_duration >= 0:
                    # check they have been roughly connected at the same time
                    if abs(charger_plugged_duration - car_plugged_duration) < CHARGER_LONG_CONNECTION_S:
                        score_plug_time_bump = 1.0 + int((plug_time_span/10.0) * ((CHARGER_LONG_CONNECTION_S - abs(charger_plugged_duration - car_plugged_duration)) / CAR_CHARGER_LONG_RELATIONSHIP_S))

                if connected_time_delta is not None :
                    if connected_time_delta > timedelta(seconds=CAR_CHARGER_LONG_RELATIONSHIP_S):
                        # not usefull seeing the return above on is_long_time_attached
                        score_plug_time_bump += 2*(plug_time_span/10.0)
                    elif connected_time_delta > timedelta(seconds=CHARGER_LONG_CONNECTION_S):
                        # the current charger has been connected to this car for a long time, we can keep it?
                        score_plug_time_bump += 1.0

            score_dist_bump = 0
            dist = -1
            if self.charger_latitude is not None and self.charger_longitude is not None:
                max_dist = 50.0  # 50 meters
                car_lat, car_long = car.get_car_coordinates(time)
                if car_lat is not None and car_long is not None:
                    dist = haversine((self.charger_latitude, self.charger_longitude), (car_lat, car_long), unit=Unit.METERS)
                    if dist <= max_dist:
                        score_dist_bump = 1.0 + int((dist_span/10.0)*(max_dist - dist)/max_dist)

                    if dist <= 3.0:
                        # we are very very close to the charger, we can assume it is plugged
                        score_plug_bump += 1

            car_home_res = cache.get(car, {}).get("car_home_res", "NOT_FOUND")
            if car_home_res == "NOT_FOUND":
                # check if the car is home
                car_home_res = car.is_car_home(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW_S)
                if car_home_res is None:
                    car_home_res = car.is_car_home(time=time)
                cache.setdefault(car, {})["car_home_res"] = car_home_res

            if car_home_res and score_dist_bump == 0:
                score_dist_bump = 1.0

            if score_plug_bump > 0 and (score_dist_bump > 0 or score_plug_time_bump > 0):
                # only if plugged .... then if home or a very compatible plug time
                score = score_plug_bump + plug_span*score_plug_time_bump + plug_span*plug_time_span*score_dist_bump

            _LOGGER.info(f"get_car_score: {car.name} for {self.name} score: {score} dist_bump: {score_dist_bump} dist: {int(dist*100)/100.0}m plug_bump: {score_plug_bump} plug_time_bump {score_plug_time_bump} connected {connected_time_delta}")


        if score is None:
            score = -1.0
            _LOGGER.info(f"get_car_score: {car.name} for {self.name} score: {score}, score was None ... no chance to be selected")

        return score


    def get_best_car(self, time: datetime) -> QSCar | None:
        # find the best car ...

        # cleanly handle the case where it has been user forced to a car
        if self.user_attached_car_name is not None:
            if self.user_attached_car_name != CHARGER_NO_CAR_CONNECTED:
                if self.user_attached_car_name == self._default_generic_car.name:
                    car = self._default_generic_car
                else:
                    car = self.home.get_car_by_name(self.user_attached_car_name)

                if car is not None:
                    _LOGGER.info(f"get_best_car:Best Car from user selection: {car.name}")

                    for charger in self.home._chargers:

                        if charger.qs_enable_device is False:
                            continue

                        if charger != self and charger.car is not None and charger.car.name == car.name:
                            if charger.user_attached_car_name is not None and charger.user_attached_car_name == car.name:
                                _LOGGER.error(f"get_best_car: {car.name} manually attached to multiple chargers: {self.name} and {charger.name}, detaching from {charger.name}")
                                charger.user_attached_car_name = None
                            else:
                                _LOGGER.info(f"get_best_car: {car.name} manually attached to charger {self.name}, detaching from {charger.name}")
                            charger.detach_car()

                    return car
            else:
                _LOGGER.info(f"get_best_car: NO GOOD CAR BECAUSE:CHARGER_NO_CAR_CONNECTED")
                return None

        cars_to_charger = {}
        active_chargers = [self]
        for charger in self.home._chargers:

            if charger.qs_enable_device is False:
                continue

            if charger.is_plugged(time, for_duration=CHARGER_CHECK_STATE_WINDOW_S):

                car = charger.car
                if car is not None and car.name != charger._default_generic_car.name:
                    cars_to_charger[car] = charger

                if charger != self:
                    active_chargers.append(charger)


        cache = {}

        chargers_scores = {}

        assigned_chargers = {}
        assigned_chargers_score = {}

        for charger in active_chargers:

            chargers_scores[charger] = []

            for car in self.home._cars:

                if car.user_attached_charger_name == FORCE_CAR_NO_CHARGER_CONNECTED:
                    continue

                score = charger.get_car_score(car, time, cache)
                # a score of 0 can't be added : it means no chance to be selected
                if score > 0:
                    chargers_scores[charger].append((score,car))

        # now we do know the scores and the cars :
        for charger in active_chargers:
            # sort the scores
            chargers_scores[charger].sort(reverse=True, key=lambda x: x[0])

        while True:
            # assign the biggest score of the bunch

            best_cur_score = None
            best_cur_charger = None
            best_cur_car = None

            for charger in active_chargers:
                if charger in assigned_chargers:
                    continue

                if len(chargers_scores[charger]) == 0:
                    continue

                score, car = chargers_scores[charger][0]
                # score > 0 by construction (see the get_car_score loop above)
                if best_cur_score is None or score > best_cur_score:
                    best_cur_score = score
                    best_cur_car = car
                    best_cur_charger = charger


            if best_cur_car is None:
                # no more changes to do
                break

            assigned_chargers[best_cur_charger] = best_cur_car
            assigned_chargers_score[best_cur_charger] = best_cur_score

            #remove best_cur_car from all the lists
            for charger in active_chargers:
                if len(chargers_scores[charger]) == 0:
                    continue

                cur_scores = chargers_scores[charger]
                chargers_scores[charger] = [s for s in cur_scores if s[1].name != best_cur_car.name]

        # ok now we have the best car for each charger ..... simply use the one for the current charger
        # and detach if it was used elsewhere, will be re-attached by the other charger directly
        best_car = assigned_chargers.get(self)

        if best_car is None:
            # there is no good car for this charger: get an invited car that is not already assigned to another charger
            if self.car is not None and self.car.car_is_invited:
                best_car = self.car
            else:
                for car in self.home._cars:
                    if car.car_is_invited and cars_to_charger.get(car) is None:
                        best_car = car
                        _LOGGER.info(f"get_best_car: Best invited car used: {best_car.name}")
                        break
            if best_car is None:
                best_car = self._default_generic_car
                _LOGGER.info(f"get_best_car: Default car used: {best_car.name}")
        else:
            _LOGGER.info(f"Best Car: {best_car.name} with score {assigned_chargers_score.get(self)} for charger {self.name}")

        existing_charger = cars_to_charger.get(best_car)
        if existing_charger is not None and existing_charger != self:
            # will force a reset of everything
            _LOGGER.info(f"Best Car for charger {self.name}: removed from another charger {existing_charger.name}")
            existing_charger.detach_car()
            # hoping we won't have back and forth between chargers

        return best_car

    def get_car_options(self)  -> list[str]:

        time = datetime.now(pytz.UTC)

        if self.is_optimistic_plugged(time):
            options = []
            for car in self.home._cars:
                options.append(car.name)

            options.extend([self._default_generic_car.name, CHARGER_NO_CAR_CONNECTED])
            return options
        else:
            self.user_attached_car_name = None
            return [CHARGER_NO_CAR_CONNECTED]

    def get_current_selected_car_option(self) -> str|None:
        if self.user_attached_car_name is not None:
            return self.user_attached_car_name

        if self.car is None:
            return None
        else:
            return self.car.name

    async def set_user_selected_car_by_name(self, car_name: str | None):
        self.user_attached_car_name = car_name
        if self.car is not None and self.car.name != car_name:
            self.detach_car()
            time = datetime.now(pytz.UTC)
            if await self.check_load_activity_and_constraints(time):
                self.home.force_next_solve()

    @property
    def default_charge_time(self) -> dt_time | None:
        if self.car is not None:
            return self.car.default_charge_time
        return None

    @default_charge_time.setter
    def default_charge_time(self, value: dt_time | None):
        if self.car is not None:
            self.car.default_charge_time = value


    async def add_default_charge(self):
        if self.can_add_default_charge():
            await self.car.add_default_charge()

    def can_add_default_charge(self) -> bool:
        if self.car is not None and self.car.can_add_default_charge():
            return True
        return False

    def can_force_a_charge_now(self) -> bool:
        if self.car is not None and self.car.can_force_a_charge_now():
            return True
        return False

    async def force_charge_now(self):
        if self.can_force_a_charge_now():
            await self.car.force_charge_now()

    def get_and_adapt_existing_constraints(self, time: datetime) -> list[LoadConstraint]:
        existing_constraints = []
        for ct in self._constraints:

            if ct.from_user is False or ct.as_fast_as_possible is False:
                continue

            _LOGGER.info(f"Found a stored car constraint to be kept with {ct.load_param}  {ct.name}")

            existing_constraints.append(ct)

        return existing_constraints



    async def check_load_activity_and_constraints(self, time: datetime) -> bool:
        # check that we have a connected car, and which one, or that it is completely disconnected
        #  if there is no more car ... just reset

        do_force_solve = False
        if not self._constraints:
            self._constraints = []

        if self._asked_for_reboot_at_time is not None:
            return False

        if self.is_charger_unavailable(time) is False:
            if self.probe_for_possible_needed_reboot(time):
                # we need to launch a rebbot
                await self.reboot(time)
                return False

        if self.is_not_plugged(time, for_duration=CHARGER_CHECK_STATE_WINDOW_S):

            if self.car:
                _LOGGER.info(f"check_load_activity_and_constraints: unplugged connected car {self.car.name}: reset")
                existing_constraints = self.get_and_adapt_existing_constraints(time)
                self.reset()
                self.user_attached_car_name = None
                do_force_solve = True
                for ct in existing_constraints:
                    self.push_live_constraint(time, ct)

            # set_charging_num_phases will check that this switch is possible
            # force single phase charge by default
            if await self.set_charging_num_phases(num_phases=1, time=time, for_default_when_unplugged=True) is False:
                # only do that if the prev one has done nothing
                await self.set_max_charging_current(current=self.charger_default_idle_charge, time=time, for_default_when_unplugged=True)

        elif self.is_plugged(time, for_duration=CHARGER_CHECK_STATE_WINDOW_S):

            existing_constraints = []
            best_car = self.get_best_car(time)

            # user forced a "deconnection"
            if best_car is None:
                _LOGGER.info("check_load_activity_and_constraints: plugged car with CHARGER_NO_CAR_CONNECTED selected option")
                existing_constraints = self.get_and_adapt_existing_constraints(time)
                self.reset()
                for ct in existing_constraints:
                    self.push_live_constraint(time, ct)
                return True

            elif self.car:
                # there is already a car connected, check if it is the right one
                #change car we have a better one ...
                if best_car.name != self.car.name:
                    _LOGGER.info("check_load_activity_and_constraints: CHANGE CONNECTED CAR!")
                    self.detach_car()

            if not self.car:
                # we may have some saved constraints that have been loaded already from the storage at init
                # so we need to check if they are still valid

                car = best_car
                _LOGGER.info(f"check_load_activity_and_constraints: plugging car {car.name} not connected: reset and attach car")

                # only keep user constraints...well in fact getting the as fast as possible one is enough
                existing_constraints = self.get_and_adapt_existing_constraints(time)

                # this reset is key it removes the constraint, the self._last_completed_constraint, etc
                # it will detach the car, etc
                self.reset()

                # find the best car ... for now
                self.attach_car(car, time)

            target_charge = await self.car.setup_car_charge_target_if_needed()

            car_current_charge_percent = car_initial_percent = self.car.get_car_charge_percent(time)
            if car_initial_percent is None: # for possible percent issue
                car_initial_percent = 0.0
                _LOGGER.info(f"check_load_activity_and_constraints: plugged car {self.car.name} has a None car_initial_percent... force init at 0")

            realized_charge_target = None
            # add a constraint ... for now just fill the car as much as possible
            force_constraint = None

            # in case a user pressed the button ....clean everything and force the charge
            if self.car.do_force_next_charge is True:
                do_force_solve = True
                self.reset_load_only() # cleanup any previous constraints to force this one!

                if car_initial_percent >= target_charge:
                    _LOGGER.info(
                        f"check_load_activity_and_constraints: plugged car {self.car.name} ins as fast as possible and has a car_initial_percent {car_initial_percent} >= target_charge {target_charge}... force init at {max(0, target_charge - 5)}")
                    car_initial_percent = max(0, target_charge - 5)


                force_constraint = MultiStepsPowerLoadConstraintChargePercent(
                    total_capacity_wh=self.car.car_battery_capacity,
                    type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
                    time=time,
                    load=self,
                    load_param=self.car.name,
                    from_user=True,
                    initial_value=car_initial_percent,
                    target_value=target_charge,
                    power_steps=self._power_steps,
                    support_auto=True,
                )
                if self.push_live_constraint(time, force_constraint):
                    _LOGGER.info(
                        f"check_load_activity_and_constraints: plugged car {self.car.name}  target_charge {target_charge} /  next target {self.car.get_car_target_charge()} pushed forces constraint {force_constraint.name}")
                    do_force_solve = True
            else:
                for ct in existing_constraints:
                    if ct.target_value != target_charge:
                        do_force_solve = True
                    if ct.load_param != self.car.name:
                        _LOGGER.info(
                            f"check_load_activity_and_constraints: Found a stored car constraint with {ct.load_param} for {self.car.name}, forcing to current car")
                        ct.reset_load_param(self.car.name)
                    ct.target_value = target_charge
                    self.push_live_constraint(time, ct)

                # we may have a as fast as possible constraint still active ... if so we need to update it
                for ct in self._constraints:
                    if ct.is_constraint_active_for_time_period(time):
                        if ct.as_fast_as_possible:
                            force_constraint = ct
                            ct.reset_load_param(self.car.name)
                            if force_constraint.target_value != target_charge:
                                do_force_solve = True
                            force_constraint.target_value = target_charge
                            break

            if force_constraint is not None:
                # reset the next charge force state
                self.car.do_force_next_charge = False
                realized_charge_target = target_charge
            else:
                # if we do have a last completed one it means there was no plug / un plug or reset in between
                # in case of a car schedule we take the start of the event as the target
                start_time, end_time = await self.car.get_next_scheduled_event(time, after_end_time=True)
                if start_time is not None and time > start_time:
                    # not need at all to push any time constraint ... we passed it
                    _LOGGER.info(
                        f"check_load_activity_and_constraints: plugged car {self.car.name} NOT pushing time mandatory constraint: {start_time} end:{end_time} time:{time}")
                    start_time = None

                if self._last_completed_constraint is not None:

                    do_check_timed_constraint = False
                    is_car_charged, _ = self.is_car_charged(time, current_charge=car_current_charge_percent, target_charge=target_charge)

                    if is_car_charged is False:
                        do_check_timed_constraint = True
                    elif start_time is not None:
                        if time - self._last_completed_constraint.end_of_constraint > timedelta(hours=12) or start_time - time < timedelta(hours=8):
                            # we may want to try to push a time based constraint that may happen sooner just in case ...
                            do_check_timed_constraint = True

                    if do_check_timed_constraint is False:
                        if start_time is not None:
                            _LOGGER.info(
                                f"check_load_activity_and_constraints: plugged car {self.car.name} removed a time constraint: {start_time} completed ct end: {self._last_completed_constraint.end_of_constraint} time:{time} is_car_charged:{is_car_charged}")
                        start_time = None
                    else:
                        # we will check for a time constraint here if there is one (ie start_time is not None)
                        # a time constraint will be created, else we will create a filler one at the end
                        # in the "if realized_charge_target is None" pass below
                        pass

                # only add it if it is "after" the end of the forced constraint
                if start_time is not None:

                    car_charge_mandatory = MultiStepsPowerLoadConstraintChargePercent(
                        total_capacity_wh=self.car.car_battery_capacity,
                        type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
                        time=time,
                        load=self,
                        load_param=self.car.name,
                        from_user=False,
                        end_of_constraint=start_time,
                        initial_value=car_initial_percent,
                        target_value=target_charge,
                        power_steps=self._power_steps,
                        support_auto=True
                    )

                    if self.push_unique_and_current_end_of_constraint_from_agenda(time, car_charge_mandatory):
                        do_force_solve = True
                        _LOGGER.info(
                            f"check_load_activity_and_constraints: plugged car {self.car.name} pushed mandatory constraint {car_charge_mandatory.name}")

                    realized_charge_target = target_charge

            if realized_charge_target is None:

                # make car charging after the battery, as it is a bets effort one
                # well no really after battery in fact
                type = CONSTRAINT_TYPE_FILLER # slightly higher priority than CONSTRAINT_TYPE_FILLER_AUTO, not CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN
                realized_charge_target = car_initial_percent

                car_charge_best_effort = MultiStepsPowerLoadConstraintChargePercent(
                    total_capacity_wh=self.car.car_battery_capacity,
                    type=type,
                    time=time,
                    load=self,
                    load_param=self.car.name,
                    from_user=False,
                    initial_value=realized_charge_target,
                    target_value=target_charge,
                    power_steps=self._power_steps,
                    support_auto=True
                )

                if self.push_live_constraint(time, car_charge_best_effort):
                    do_force_solve = True
                    _LOGGER.info(
                        f"check_load_activity_and_constraints: plugged car {self.car.name} default charge: {self.car.car_default_charge}% pushed filler constraint {car_charge_best_effort.name}")


        return do_force_solve

    @property
    def min_charge(self):
        if self.car:
            return int(max(self.charger_min_charge, self.car.car_charger_min_charge))
        else:
            return int(self.charger_min_charge)

    @property
    def charger_default_idle_charge(self):
        return int(max(self._charger_default_idle_charge, self.min_charge))

    @property
    def efficiency_factor(self):
        if self.car:
            return self.car.efficiency_factor
        else:
            return super().efficiency_factor

    @property
    def max_charge(self):
        if self.car:
            return int(min(self.charger_max_charge, self.car.car_charger_max_charge))
        else:
            return self.charger_max_charge

    def get_platforms(self):
        parent = super().get_platforms()
        if parent is None:
            parent = set()
        else:
            parent = set(parent)
        parent.update([Platform.SENSOR, Platform.SELECT, Platform.SWITCH,Platform.BUTTON, Platform.TIME])
        return list(parent)

    def attach_car(self, car, time: datetime):

        if self.car is not None:
            if self.car.name == car.name:
                return
            self.detach_car()

        self.car = car

        # reset dampening to conf values, and some states
        car.reset()

        if car.calendar is None:
            car.calendar = self.calendar

        self.update_power_steps()
        self.car_attach_time = time

        car.charger = self

    def detach_car(self):
        if self.car is not None:
            self.car.charger = None

        self.car = None
        self._power_steps = []
        self.car_attach_time = None

    # update in place the power steps
    def update_power_steps(self):
        if self.car:
            if self.can_do_3_to_1_phase_switch():
                power_steps_3p, min_charge, max_charge = self.car.get_charge_power_per_phase_A(True)
                power_steps_1p, min_charge_1p, max_charge_1p = self.car.get_charge_power_per_phase_A(False)
                s = set(power_steps_3p[min_charge: max_charge + 1])
                s.update(power_steps_1p[min_charge_1p: max_charge_1p + 1])
            else:
                power_steps, min_charge, max_charge = self.car.get_charge_power_per_phase_A(self.physical_3p)
                s = set(power_steps[self.min_charge: self.max_charge + 1])

            if 0 in s:
                s.remove(0)

            power_steps = sorted(s)

            _LOGGER.info(f"update_power_steps: {self.car.name} {power_steps} {min_charge}/{max_charge}")
            steps = []

            for power in power_steps:
                steps.append(copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=power))

            self._power_steps = steps

            for ct in self._constraints:
                if isinstance(ct, MultiStepsPowerLoadConstraint):
                    ct.update_power_steps(steps)

    def get_min_max_power(self) -> (float, float):
        if self._power_steps is None or len(self._power_steps) == 0 or self.car is None:
            return 0.0, 0.0

        return self._power_steps[0].power_consign, self._power_steps[-1].power_consign

    def get_min_max_phase_amps_for_budgeting(self) -> ( list[float|int],  list[float|int]):
        if self.physical_3p:
            return [self.min_charge, self.min_charge, self.min_charge], [self.max_charge, self.max_charge, self.max_charge]
        else:
            min_c = [0.0, 0.0, 0.0]
            max_c = [0.0, 0.0, 0.0]
            min_c[self.mono_phase_index] = self.min_charge
            max_c[self.mono_phase_index] = self.max_charge

            return min_c, max_c

    async def stop_charge(self, time: datetime):

        self._expected_charge_state.register_launch(value=False, time=time)
        charge_state = self.is_charge_enabled(time)

        if charge_state or charge_state is None:
            _LOGGER.info("STOP CHARGE LAUNCHED")
            try:
                await self.low_level_stop_charge(time)
            except Exception as e:
                _LOGGER.warning(f"stop_charge EXCEPTION {e}")

    async def start_charge(self, time: datetime):
        self._expected_charge_state.register_launch(value=True, time=time)
        charge_state = self.is_charge_disabled(time)
        if charge_state or charge_state is None:
            _LOGGER.info("START CHARGE LAUNCHED")
            try:
                await self.low_level_start_charge(time)
            except Exception as e:
                _LOGGER.warning(f"start_charge EXCEPTION {e}")

    def _check_charger_status(self,
                              status_vals: list[str],
                              time: datetime,
                              for_duration: float | None = None,
                              invert_prob=False) -> bool | None:

        if not status_vals or self.charger_status_sensor is None:
            return None

        if for_duration is None or for_duration < 0:
            for_duration = 0

        contiguous_status = self.get_last_state_value_duration(self.charger_status_sensor,
                                                               states_vals=status_vals,
                                                               num_seconds_before=8*for_duration,
                                                               time=time,
                                                               invert_val_probe=invert_prob)[0]
        if contiguous_status is None:
            return None
        else:
            return contiguous_status >= for_duration and contiguous_status > 0


    def get_continuous_plug_duration(self, time: datetime) -> float | None:

        return self.get_last_state_value_duration(self._internal_fake_is_plugged_id,
                                                  states_vals=[QSChargerStates.PLUGGED],
                                                  num_seconds_before=None,
                                                  time=time)[0]

    def _check_plugged_val(self,
                           time: datetime,
                           for_duration: float | None = None,
                           check_for_val=True) -> bool | None:

        if for_duration is None or for_duration < 0:
            for_duration = 0

        contiguous_status = self.get_last_state_value_duration(self._internal_fake_is_plugged_id,
                                                               states_vals=[QSChargerStates.PLUGGED],
                                                               num_seconds_before=8*for_duration,
                                                               time=time,
                                                               invert_val_probe=not check_for_val)[0]
        if contiguous_status is None:
            res = contiguous_status
        else:
            res = contiguous_status >= for_duration and contiguous_status > 0


        if res is None and self.car:

            if check_for_val:
                ok_value = QSChargerStates.PLUGGED
                not_ok_value = QSChargerStates.UN_PLUGGED
            else:
                ok_value = QSChargerStates.UN_PLUGGED
                not_ok_value = QSChargerStates.PLUGGED

            latest_charger_valid_state = self.get_sensor_latest_possible_valid_value(self._internal_fake_is_plugged_id)

            if latest_charger_valid_state is None:
                res = None
            else:
                # only check car if we are very sure of the charger state
                res_car =  self.car.is_car_plugged(time, for_duration)
                if res_car is not None:
                    if res_car is check_for_val and latest_charger_valid_state == ok_value:
                        res = True
                    if res_car is not check_for_val and latest_charger_valid_state == not_ok_value:
                        res = False

        return res


    def is_plugged(self, time: datetime, for_duration: float | None = None) -> bool | None:
        return self._check_plugged_val(time, for_duration, check_for_val=True)

    def is_optimistic_plugged(self, time: datetime) -> bool | None:
        is_plugged = self.is_plugged(time=time)
        if is_plugged is None:
            is_plugged = self.is_plugged(time, for_duration=CHARGER_CHECK_STATE_WINDOW_S)

        return is_plugged

    def is_not_plugged(self, time: datetime, for_duration: float | None = None) -> bool | None:
        return self._check_plugged_val(time, for_duration, check_for_val=False)

    def is_charger_plugged_now(self, time: datetime) -> tuple[bool|None, datetime]:

        if self.charger_status_sensor:
            state_wallbox = self.hass.states.get(self.charger_status_sensor)
            if state_wallbox is None or state_wallbox.state in self._unknown_state_vals:
                #if other are not available, we can't know if the charger is plugged at all
                if state_wallbox is not None:
                    state_time : datetime = state_wallbox.last_updated
                else:
                    state_time = time
                return None, state_time

            plugged_state_vals = self.get_car_plugged_in_status_vals()
            if plugged_state_vals:
                state_time: datetime = state_wallbox.last_updated
                if state_wallbox.state in plugged_state_vals:
                    return True, state_time
                else:
                    return False, state_time


        return self.low_level_plug_check_now(time)

    def check_charge_state(self, time: datetime, for_duration: float | None = None, check_for_val=True) -> bool | None:

        result = not check_for_val
        if self.is_plugged(time=time, for_duration=for_duration):

            status_vals = self.get_car_charge_enabled_status_vals()

            result = self._check_charger_status(status_vals, time, for_duration, invert_prob=not check_for_val)

            if result is not None:
                return result

            if self.charger_status_sensor:
                state_wallbox = self.hass.states.get(self.charger_status_sensor)
                if state_wallbox is None or state_wallbox.state in self._unknown_state_vals:
                    return None

            result =  self.low_level_charge_check_now(time)

            if result is not None and check_for_val is False:
                result = not result

        return result

    def is_charger_unavailable(self, time: datetime, for_duration: float | None = None) -> bool | None:

        if self.charger_status_sensor is not None:

            state_wallbox = self.hass.states.get(self.charger_status_sensor)
            if state_wallbox is None or state_wallbox == STATE_UNAVAILABLE:
                return True

        return False

    def is_charge_enabled(self, time: datetime, for_duration: float | None = None) -> bool | None:
        return self.check_charge_state(time, for_duration, check_for_val=True)

    def is_charge_disabled(self, time: datetime, for_duration: float | None = None) -> bool | None:
        return self.check_charge_state(time, for_duration, check_for_val=False)

    def is_car_stopped_asking_current(self, time: datetime) -> bool | None:

        result = False
        if self.is_plugged(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW_S) or self.is_plugged(time=time):

            result = None

            max_charging_power = self.get_max_charging_amp_per_phase()
            if max_charging_power is None:
                return None

            # check that in fact the car could receive something...if not it may be waiting for power but cant get it
            if max_charging_power < self.min_charge:
                return None

            status_vals = self.get_car_stopped_asking_current_status_vals()

            for_duration: float = CHARGER_STOP_CAR_ASKING_FOR_CURRENT_TO_STOP_S

            if status_vals and len(status_vals) > 0:
                result = self._check_charger_status(status_vals, time, for_duration)
                if result:
                    _LOGGER.info(
                        f"is_car_stopped_asking_current: because charger state in {status_vals} for  {for_duration} seconds")

            if result is None:
                if self.is_charge_enabled(time=time, for_duration=for_duration):
                    result =  self.is_charging_power_zero(time=time, for_duration=for_duration)
                    if result:
                        _LOGGER.info(f"is_car_stopped_asking_current: because Charging power is zero for {for_duration} seconds")

        return result

    def is_charging_power_zero(self, time: datetime, for_duration: float) -> bool | None:
        val = self.get_median_power(for_duration, time, use_fallback_command=False)
        if val is None:
            return None

        return self.dampening_power_value_for_car_consumption(val) == 0.0  # 70 W of consumption for the charger for ex


    async def set_max_charging_current(self, current, time: datetime, for_default_when_unplugged=False):

        has_done_change = False
        if for_default_when_unplugged is False:
            self._expected_amperage.register_launch(value=current, time=time)

        if self.get_max_charging_amp_per_phase() != current:
            has_done_change = await self.low_level_set_max_charging_current(current, time)

        # await self.hass.services.async_call(
        #    domain=domain, service=service, service_data={number.ATTR_VALUE:int(min(max_value, max(min_value, range_value)))}, target={ATTR_ENTITY_ID: self.charger_max_charging_current_number}, blocking=blocking
        # )
        return has_done_change



    def get_max_charging_amp_per_phase(self):

        state = self.hass.states.get(self.charger_max_charging_current_number)

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            result = None
        else:
            try:
                result = float(state.state)
            except TypeError:
                result = None
            except ValueError:
                result = None

        return result

    def can_do_3_to_1_phase_switch(self):
        if self.charger_three_to_one_phase_switch and self.physical_3p:
            return True
        return False

    def can_reboot(self):
        if self.charger_reboot_button:
            return True
        return False


    async def set_charging_num_phases(self, num_phases:int, time: datetime, for_default_when_unplugged=False) -> bool:

        has_done_change = False
        if self.can_do_3_to_1_phase_switch() and self.physical_3p:
            if for_default_when_unplugged is False:
                self._expected_num_active_phases.register_launch(value=num_phases, time=time)

            if self.current_num_phases != num_phases:

                has_done_change = await self.low_level_set_charging_num_phases(num_phases, time)

                if for_default_when_unplugged is False and self.do_reboot_on_phase_switch and has_done_change:

                    is_plugged = self.is_optimistic_plugged(time=time)

                    if is_plugged is True:
                        # we will need a restart of the charger to take the new phase switch into account
                        # we need to launch the rebbot only when the _expected_num_active_phases is met
                        self._expected_num_active_phases.register_success_cb(self.reboot, None)
        else:
            # success, direct
            self._expected_num_active_phases.register_launch(value=self.physical_num_phases, time=time)
            await self._expected_num_active_phases.success(time=time)
            has_done_change = True

        return has_done_change


    async def reboot(self, time: datetime):
        if self.can_reboot():
            self._asked_for_reboot_at_time = time
            _LOGGER.warning(f"reboot: {self.name}")
            await self.low_level_reboot(time)



    def probe_for_possible_needed_reboot(self, time):
        if self.can_reboot() is False:
            return False
        # for now no clear way of defining when a reboot is needed
        return False

    async def check_if_reboot_happened(self, from_time: datetime, to_time: datetime) -> bool:

        if self.can_reboot() is False:
            return True

        reboot_duration_s = (to_time - from_time).total_seconds()

        if reboot_duration_s <= self.minimum_reboot_duration_s:
            return False

        status_vals = self.get_car_status_rebooting_vals()
        if self.charger_status_sensor_unfiltered is not None and status_vals is not None and len(status_vals) > 0:


            contiguous_status = self.get_last_state_value_duration(self.charger_status_sensor_unfiltered,
                                                                   states_vals=status_vals,
                                                                   num_seconds_before=reboot_duration_s,
                                                                   time=to_time,
                                                                   invert_val_probe=False,
                                                                   count_only_duration=True)[0]
            if contiguous_status is None or contiguous_status < 2*CHARGER_STATE_REFRESH_INTERVAL_S:
                return False
            else:
                # long enough recognition of a state associated with a reboot, now check that we are no more in this state
                no_reboot_time = self.get_last_state_value_duration(self.charger_status_sensor_unfiltered,
                                                                       states_vals=status_vals,
                                                                       num_seconds_before=reboot_duration_s,
                                                                       time=to_time,
                                                                       invert_val_probe=True)[0]
                if no_reboot_time is not None and no_reboot_time > 2*CHARGER_STATE_REFRESH_INTERVAL_S:
                    return True

        return True

    def _is_state_set(self, time: datetime) -> bool:
        return self._expected_amperage.value is not None and self._expected_charge_state.value is not None and self._expected_num_active_phases.value is not None


    async def ensure_correct_state(self, time: datetime, probe_only:bool = False) -> (bool | None, bool, datetime | None):

        await self._do_update_charger_state(time)

        if self.is_charger_unavailable(time=time):
            _LOGGER.info(f"ensure_correct_state: {self.name} not available")
            # return None in ths particular case as nothing frm this charger will be actionable
            return None, False, None

        if self.car is None or self.is_not_plugged(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW_S):
            # if we reset here it will remove the current constraint list from the load!!!!
            _LOGGER.info(f"ensure_correct_state: {self.name} no car or not plugged")
            return True, False, None

        if self.is_not_plugged(time=time):
            # could be a "short" unplug
            _LOGGER.info(f"ensure_correct_state:{self.name} short unplug")
            return False, False, None # we don't know if final

        if self.running_command is not None:
            _LOGGER.info(f"ensure_correct_state:{self.name} running command {self.running_command}")
            return False, False, None

        handled = self._probe_and_enforce_stopped_charge_command_state(time, command=self.current_command, probe_only=probe_only)
        return await self._ensure_correct_state(time, probe_only), handled, self._verified_correct_state_time

    async def _ensure_correct_state(self, time, probe_only:bool = False) -> bool:

        one_bad = False

        if self.is_in_state_reset():
            _LOGGER.info(f"Ensure State:{self.name} no correct expected state")
            one_bad = True

        if one_bad is False:
            if self._asked_for_reboot_at_time is not None:
                is_reboot_done = self.check_if_reboot_happened(from_time=self._asked_for_reboot_at_time, to_time=time)
                if is_reboot_done:
                    _LOGGER.info(f"Ensure State:{self.name} reboot asked and now restart happened")
                    self._asked_for_reboot_at_time = None
                else:
                    _LOGGER.info(f"Ensure State:{self.name} reboot asked but still not happened")
                    one_bad = True

        if one_bad is False:
            current_active_phases = self.current_num_phases
            if current_active_phases != self._expected_num_active_phases.value:
                one_bad = True
                # check first if amperage setting is ok
                if probe_only is False:
                    if self._expected_num_active_phases.is_ok_to_launch(value=self._expected_num_active_phases.value, time=time):
                        _LOGGER.info(f"Ensure State:{self.name} num_phases {current_active_phases} expected {self._expected_num_active_phases.value}")
                        await self.set_charging_num_phases(num_phases=self._expected_num_active_phases.value, time=time)
                    else:
                        _LOGGER.debug(f"Ensure State:{self.name} NOT OK TO LAUNCH num phases {current_active_phases} expected {self._expected_num_active_phases.value}")
            else:
                await self._expected_num_active_phases.success(time=time)


        #if we expect a charge stop, if it is stopped the amps is not a mandatory check should only be done if we expect a charge start
        if one_bad is False:

            is_charge_enabled = self.is_charge_enabled(time)
            is_charge_disabled = self.is_charge_disabled(time)

            if is_charge_enabled is None:
                _LOGGER.info(f"Ensure State:{self.name} is_charge_enabled state unknown")
            if is_charge_disabled is None:
                _LOGGER.info(f"Ensure State:{self.name} is_charge_disabled state unknown")

            amps_bad_set = True
            if self._expected_charge_state.value is False and is_charge_disabled:
                # ok we are in a good stopped state : even if the amperage is not set, we can say that the result will be good
                amps_bad_set = False

            max_charging_current = self.get_max_charging_amp_per_phase()
            if max_charging_current == self._expected_amperage.value:
                await self._expected_amperage.success(time=time)
            else:
                one_bad = amps_bad_set
                _LOGGER.info(
                    f"Ensure State:{self.name} current {max_charging_current}A expected {self._expected_amperage.value}A")
                if probe_only is False:
                    if self._expected_amperage.is_ok_to_launch(value=self._expected_amperage.value, time=time):
                        _LOGGER.info(f"Ensure State:{self.name} current {max_charging_current}A expected {self._expected_amperage.value}A LAUNCH set_max_charging_current")
                        await self.set_max_charging_current(current=self._expected_amperage.value, time=time)
                    else:
                        _LOGGER.debug(f"Ensure State:{self.name} NOT OK TO LAUNCH current {max_charging_current}A expected {self._expected_amperage.value}A")

            if not ((self._expected_charge_state.value is True and is_charge_enabled) or (
                self._expected_charge_state.value is False and is_charge_disabled)):
                one_bad = True
                _LOGGER.info(
                    f"Ensure State:{self.name} expected {self._expected_charge_state.value} is_charge_enabled {is_charge_enabled} is_charge_disabled {is_charge_disabled}")
                if probe_only is False:
                    # if amperage is ok check if charge state is ok
                    if self._expected_charge_state.is_ok_to_launch(value=self._expected_charge_state.value, time=time):
                        if self._expected_charge_state.value:
                            _LOGGER.info(f"Ensure State:{self.name} start_charge")
                            await self.start_charge(time=time)
                        else:
                            _LOGGER.info(f"Ensure State:{self.name} stop_charge")
                            await self.stop_charge(time=time)
                    else:
                        _LOGGER.debug(f"Ensure State:{self.name} NOT OK TO LAUNCH expected {self._expected_charge_state.value} is_charge_enabled {is_charge_enabled} is_charge_disabled {is_charge_disabled}")
            else:
                await self._expected_charge_state.success(time=time)

        if one_bad is False:
            _LOGGER.debug(f"Ensure State:{self.name} success amp {self._expected_amperage.value} (#phases: {self._expected_num_active_phases.value})")

            if self._verified_correct_state_time is None:
                # ok we enter the state knowing where we are
                self._verified_correct_state_time = time
            return True
        else:
            self._verified_correct_state_time = None

        return False


    def _compute_added_percent_charge_update(self, start_time: datetime, end_time: datetime) -> float | None:
        """ compute the percent charge update for a given time period
        """
        if self.car is None or self.car.car_battery_capacity is None:
            return None

        added_percent = None

        added_nrj = self.get_device_real_energy(start_time=start_time,
                                                end_time=end_time,
                                                clip_to_zero_under_power=self.charger_consumption_W)

        if added_nrj is not None and self.car.car_battery_capacity > 0:
            added_nrj = added_nrj/self.efficiency_factor
            added_percent = (100.0 * added_nrj) / self.car.car_battery_capacity

        return added_percent

    async def constraint_update_value_callback_percent_soc(self, ct: LoadConstraint, time: datetime) -> tuple[float | None, bool]:
        """ Example of a value compute callback for a load constraint. like get a sensor state, compute energy available for a car from battery charge etc
        it could also update the current command of the load if needed
        """
        await self._do_update_charger_state(time)

        if self.car is None or self.is_not_plugged(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW_S):
            # if we reset here it will remove the current constraint list from the load!!!!
            _LOGGER.info(f"update_value_callback: {self.name} reset because no car or not plugged")
            return (None, False)

        if self.is_not_plugged(time=time):
            # could be a "short" unplug
            _LOGGER.info(f"update_value_callback: {self.name} short unplug")
            return (None, True)

        result_calculus = None
        sensor_result = None

        if self.current_command is None or self.current_command.is_off_or_idle():
            _LOGGER.info(f"update_value_callback: {self.name} no command or idle/off")
            result = None
        else:

            probe_charge_window = 30*60
            sensor_result = self.car.get_car_charge_percent(time, tolerance_seconds=probe_charge_window)

            total_charge_duration = (ct.last_value_update - ct.first_value_update).total_seconds()

            computed_percent_added_last = self._compute_added_percent_charge_update(start_time=ct.last_value_change_update, end_time=time)
            if computed_percent_added_last is not None:
                result_calculus = ct.current_value + int(computed_percent_added_last) # round it to int ... so stay the same for small updates

            result = sensor_result

            if sensor_result is None:
                if self.car.car_charge_percent_sensor:
                    _LOGGER.info(
                        f"update_value_callback:{self.name} {self.car.name} use calculus because sensor None {result_calculus}")
                result = result_calculus
            else:
                if sensor_result > 99:
                    if computed_percent_added_last is not None:
                        result = min(sensor_result, ct.current_value + computed_percent_added_last)
                elif total_charge_duration >= probe_charge_window:
                    # keep the sensor result we are at the begining
                    computed_percent_added_begin = self._compute_added_percent_charge_update(
                        start_time=ct.first_value_update, end_time=time)

                    if computed_percent_added_begin is None or computed_percent_added_begin >= 1:
                        # we should have a growing one probe the last probe_window
                        computed_percent_probe_window = self._compute_added_percent_charge_update(start_time=time - timedelta(seconds=probe_charge_window), end_time=time)

                        if computed_percent_probe_window is not None and computed_percent_probe_window >= 1:
                            # we are growing in the last probe window
                            is_growing = self.car.is_car_charge_growing(num_seconds=computed_percent_probe_window, time=time)
                            if is_growing is None:
                                _LOGGER.info(
                                    f"update_value_callback:{self.name} {self.car.name} use calculus because sensor growing unknown (expected growth:{computed_percent_probe_window}%)  {result_calculus}")
                                result = result_calculus
                            elif is_growing is False:
                                # we are not growing and we should ...
                                if computed_percent_probe_window > 5:
                                    _LOGGER.info(f"update_value_callback:{self.name} {self.car.name} use calculus because sensor not growing (expected growth:{computed_percent_probe_window}%)  {result_calculus}")
                                    result = result_calculus
                            # else : the sensor is growing ... keep it


        is_car_charged, result = self.is_car_charged(time, current_charge=result, target_charge=ct.target_value)

        if result is not None and ct.is_constraint_met(time=time, current_value=result):
            do_continue_constraint = False
        else:
            do_continue_constraint = True
            await self.car.setup_car_charge_target_if_needed(ct.target_value)
            # await self._dynamic_compute_and_launch_new_charge_state(time)
            await self.charger_group.dyn_handle(time)

        _LOGGER.info(f"update_value_callback:{self.name} {self.car.name}  {do_continue_constraint}/{result} ({sensor_result}/{result_calculus}) is_car_charged {is_car_charged} cmd {self.current_command}")

        return (result, do_continue_constraint)

    def is_car_charged(self, time: datetime,  current_charge: float | int | None,  target_charge: float | int) -> (bool, int|float):

        is_car_stopped_asked_current = self.is_car_stopped_asking_current(time=time)
        result = current_charge

        if is_car_stopped_asked_current:
            # force met constraint: car is charged in all cases
            result = target_charge
        elif current_charge is not None:
            if target_charge >= 100:
                # for a car to be fully charged it has to have a stopped asking charge at minimum
                result = min(current_charge, 99)
            else:
                ct = LoadConstraint()
                ct.target_value = target_charge
                ct.current_value = current_charge
                if ct.is_constraint_met(time=time, current_value=current_charge):
                    # force met constraint
                    result = ct.target_value

        return result == target_charge, result

    def get_delta_dampened_power(self, from_amp: int | float, from_num_phase: int, to_amp: int | float, to_num_phase: int) -> float | None:
        if self.car:
            return self.car.get_delta_dampened_power(from_amp=from_amp, from_num_phase=from_num_phase, to_amp=to_amp, to_num_phase=to_num_phase)
        else:
            return None

    def update_car_dampening_value(self, time : datetime, amperage:None|tuple[float,int]|tuple[int,int], amperage_transition: None|tuple[tuple[int,int]|tuple[float,int], tuple[int,int]|tuple[float,int]], power_value_or_delta: float, can_be_saved:bool=False):
        if self.car:
            if self.car.update_dampening_value(amperage=amperage, amperage_transition=amperage_transition, power_value_or_delta=power_value_or_delta, time=time, can_be_saved=can_be_saved):
                self.update_power_steps()

    def _probe_and_enforce_stopped_charge_command_state(self, time, command: LoadCommand, probe_only: bool = False) -> bool:

        if self.car is None:
            return True

        handled = False
        if self.is_car_stopped_asking_current(time):
            _LOGGER.info(f"_probe_and_enforce_stopped_charge_command_state: {self.name} car {self.car.name} stopped asking current ... do nothing")
            handled = True
        elif command is None or command.is_off_or_idle():
            handled = True
            _LOGGER.info(f"_probe_and_enforce_stopped_charge_command_state: {self.name} car {self.car.name} not a running command {command}")
        elif command.is_auto() or command.is_like(CMD_ON):
            handled = False

        if self.is_in_state_reset():
            _LOGGER.info(f"_probe_and_enforce_stopped_charge_command_state: {self.name} car {self.car.name}in state reset at the end .. force an idle like state")
            handled = True

        if probe_only is False and handled is True:
            self._expected_charge_state.set(False, time)
            self._expected_amperage.set(self.charger_default_idle_charge, time)
            self._expected_num_active_phases.set(self.current_num_phases, time)

        return handled

    async def execute_command(self, time: datetime, command: LoadCommand) -> bool | None:

        # force a homeassistant.update_entity service on the charger entity?
        if command is None:
            return True

        await self._do_update_charger_state(time)
        is_plugged = self.is_optimistic_plugged(time=time)

        if is_plugged and self.car is not None:

            do_reset = True
            if (self.current_command is not None and
                    ((command.is_like(CMD_ON) or command.is_auto())
                     and (self.current_command.is_auto() or self.current_command.is_like(CMD_ON)))
                    and not self.is_in_state_reset()):
                # well by construction , command and self.current_command are different
                # we are in the middle of the execution of probably the same on or command constraint (or another one) but in a continuity of commands
                do_reset = False

            if do_reset:
                self._reset_state_machine()
                _LOGGER.info(f"DO RESET Execute command {command.command}/{command.power_consign} on charger {self.name}")
                self._probe_and_enforce_stopped_charge_command_state(time, command=command)
                res = await self._ensure_correct_state(time)
            else:
                # we where already in automatic command ... no need to reset anything : just let the callback plays its role
                _LOGGER.info(f"Recieved and acknowledged command {command.command}/{command.power_consign} on charger {self.name}")
                res = True

            return res
        else:
            if command.is_off_or_idle():
                return True
            return None

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool | None:
        await self._do_update_charger_state(time)
        is_plugged = self.is_optimistic_plugged(time=time)

        result = None
        if is_plugged and self.car is not None:
            _LOGGER.info(f"called again compute_and_launch_new_charge_state command {command}")
            self._probe_and_enforce_stopped_charge_command_state(time, command=command)
            result = await self._ensure_correct_state(time, probe_only=True)
        else:
            if command.is_off_or_idle():
                result = True
            else:
                if self.car is None:
                    _LOGGER.info(f"Bad prob command set: plugged {is_plugged} NO CAR")
                else:
                    _LOGGER.info(f"Bad prob command set: plugged {is_plugged} Car: {self.car.name}")

        return result

    async def _do_update_charger_state(self, time):
        if self._last_charger_state_prob_time is None or (time - self._last_charger_state_prob_time).total_seconds() > CHARGER_STATE_REFRESH_INTERVAL_S:

            state = self.hass.states.get(self.charger_max_charging_current_number)

            if state is not None:
                state_time = state.last_updated

                if (time - state_time).total_seconds() <= CHARGER_STATE_REFRESH_INTERVAL_S:
                    self._last_charger_state_prob_time = state_time
                else:
                    _LOGGER.warning(f"Forcing a charger state update!")

                    entity_to_probe = []
                    if self.charger_plugged is not None:
                        entity_to_probe.append(self.charger_plugged)
                    if self.charger_pause_resume_switch is not None:
                        entity_to_probe.append(self.charger_pause_resume_switch)
                    if self.charger_max_charging_current_number is not None:
                        entity_to_probe.append(self.charger_max_charging_current_number)

                    await self.hass.services.async_call(
                        homeassistant.DOMAIN,
                        homeassistant.SERVICE_UPDATE_ENTITY,
                        {ATTR_ENTITY_ID: entity_to_probe},
                        blocking=False
                    )
                    self._last_charger_state_prob_time = time

    def _find_charger_entity_id(self, device, entries, prefix, suffix):

        found = None
        for entry in entries:
            if entry.entity_id.startswith(prefix) and entry.entity_id.endswith(suffix):
                found = entry.entity_id
                break

        if device is not None:
            device_name = device.name_by_user or device.name
            computed = prefix + slugify(device_name) + suffix

            if found is not None and found != computed:
                _LOGGER.warning(f"Entity ID {found} does not match expected {computed} for device {device.id}")
                # we could rename it here if needed

            if found is None:
                _LOGGER.warning(
                    f"Entity ID for {device_name} not found with prefix {prefix} and suffix {suffix}, expected {computed}")
                found = computed

        return found


    # ============================ INTERFACE TO BE OVERCHARGED ===================================== #

    @property
    def do_reboot_on_phase_switch(self):
        return False

    def low_level_charge_check_now(self, time: datetime) -> bool | None:

        state = self.hass.states.get(self.charger_pause_resume_switch)
        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            res_charge_check = None
        else:
            res_charge_check = state.state == "on"

        return res_charge_check

    def low_level_plug_check_now(self, time: datetime) -> tuple[None, datetime] | tuple[bool, datetime]:

        state = self.hass.states.get(self.charger_plugged)
        if state is not None:
            state_time = state.last_updated
        else:
            state_time = time

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            res_plugged = None
        else:
            res_plugged = state.state == "on"

        return res_plugged, state_time

    async def low_level_reboot(self, time):
        if self.can_reboot() and self.charger_reboot_button is not None:
            try:
                await self.hass.services.async_call(
                    domain=Platform.BUTTON,
                    service=SERVICE_PRESS,
                    target={ATTR_ENTITY_ID: self.charger_reboot_button},
                    blocking=False
                )
                _LOGGER.info(f"Rebooting charger {self.name} at {time}")
            except Exception as e:
                _LOGGER.error(f"low_level_reboot: Error {e}")


    async def low_level_set_charging_num_phases(self, num_phases: int, time: datetime) -> bool:
        if num_phases == 1:
            service = SERVICE_TURN_ON
        else:
            service = SERVICE_TURN_OFF

        try:
            await self.hass.services.async_call(
                domain=Platform.SWITCH,
                service=service,
                target={ATTR_ENTITY_ID: self.charger_three_to_one_phase_switch},
                blocking=False
            )
            return True
        except Exception as e:
            _LOGGER.warning(f"low_level_set_charging_num_phases: num_phases {num_phases} Error {e}")
            return False

    async def low_level_set_max_charging_current(self, current, time: datetime) -> bool:
        try:
            data: dict[str, Any] = {ATTR_ENTITY_ID: self.charger_max_charging_current_number}
            range_value = float(current)
            service = number.SERVICE_SET_VALUE
            min_value = float(self.charger_min_charge)
            max_value = float(self.charger_max_charge)
            data[number.ATTR_VALUE] = int(min(max_value, max(min_value, range_value)))
            domain = number.DOMAIN

            await self.hass.services.async_call(
                domain, service, data, blocking=False
            )
            done = True
        except Exception as e:
            _LOGGER.warning(f"low_level_set_max_charging_current: Error {e}")
            done = False
        return done


    async def low_level_stop_charge(self, time: datetime):

        await self.hass.services.async_call(
            domain=Platform.SWITCH,
            service=SERVICE_TURN_OFF,
            target={ATTR_ENTITY_ID: self.charger_pause_resume_switch},
            blocking=False
        )

    async def low_level_start_charge(self, time: datetime):

        await self.hass.services.async_call(
            domain=Platform.SWITCH,
            service=SERVICE_TURN_ON,
            target={ATTR_ENTITY_ID: self.charger_pause_resume_switch},
            blocking=False
        )

    def get_car_charge_enabled_status_vals(self) -> list[str]:
        return []

    def get_car_plugged_in_status_vals(self) -> list[str]:
        return []

    def get_car_status_unknown_vals(self) -> list[str]:
        return []

    def get_car_stopped_asking_current_status_vals(self) -> list[str]:
        return []

    def get_car_status_rebooting_vals(self) -> list[str]:
        return []



class QSChargerOCPP(QSChargerGeneric):

    def __init__(self, **kwargs):
        self.charger_device_ocpp = kwargs.pop(CONF_CHARGER_DEVICE_OCPP, None)
        self.charger_ocpp_current_import = None
        self.charger_ocpp_power_active_import = None

        hass: HomeAssistant | None = kwargs.get("hass", None)

        if self.charger_device_ocpp is not None and hass is not None:

            device_registry_instance = device_registry.async_get(hass)
            device = device_registry_instance.async_get(self.charger_device_ocpp)

            entity_reg = entity_registry.async_get(hass)
            entries = entity_registry.async_entries_for_device(entity_reg, self.charger_device_ocpp)


            kwargs[CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER] = self._find_charger_entity_id(device, entries, "number.", "_maximum_current")
            kwargs[CONF_CHARGER_PAUSE_RESUME_SWITCH] = self._find_charger_entity_id(device, entries, "switch.", "_charge_control")
            kwargs[CONF_CHARGER_PLUGGED] = self._find_charger_entity_id(device, entries, "switch.", "_availability")

            # if entry.entity_id.startswith("sensor.") and entry.entity_id.endswith("_power_active_import"):
            #    self.charger_ocpp_power_active_import = entry.entity_id
            kwargs[CONF_CHARGER_STATUS_SENSOR] = self._find_charger_entity_id(device, entries, "sensor.", "_status_connector")
            # kwargs[CONF_CHARGER_REBOOT_BUTTON] = self._find_charger_entity_id(device, entries, "button.", "_reset")

            self.charger_ocpp_current_import = self._find_charger_entity_id(device, entries, "sensor.", "_current_import")

        super().__init__(**kwargs)

        self.secondary_power_sensor = self.charger_ocpp_current_import # it is total amps (3 phases sum)
        self.attach_power_to_probe(self.secondary_power_sensor, transform_fn=self.convert_ocpp_current_import_amps_to_W)
        # self.attach_power_to_probe(self.charger_ocpp_power_active_import)

    async def handle_ocpp_notification(self, message: str, title: str = "OCPP Charger Notification"):
        """Handle notifications from the OCPP integration and take automated actions."""
        _LOGGER.info(f"Received OCPP notification for charger {self.name}: {title} - {message}")
        # check message: Warning: Start transaction failed with response
        return

        try:
            _LOGGER.info(f"Received OCPP notification for charger {self.name}: {title} - {message}")
            
            # Analyze the notification content and take appropriate actions
            message_lower = message.lower()
            title_lower = title.lower()
            
            # Check for conditions that require a reboot
            reboot_triggers = [
                "firmware upload status",
                "configuration changed", 
                "reset required",
                "reboot required",
                "restart needed",
                "configuration update",
                "profile updated"
            ]
            
            should_reboot = False
            for trigger in reboot_triggers:
                if trigger in message_lower:
                    should_reboot = True
                    _LOGGER.info(f"OCPP notification indicates reboot needed: {trigger}")
                    break
            
            # Check for error conditions that might need attention
            error_triggers = [
                "error",
                "failed", 
                "fault",
                "warning",
                "exception",
                "timeout"
            ]
            
            is_error = False
            for error_trigger in error_triggers:
                if error_trigger in message_lower:
                    is_error = True
                    _LOGGER.warning(f"OCPP notification indicates error condition: {error_trigger}")
                    break
            
            # Take automated actions based on notification content
            current_time = datetime.now(pytz.UTC)
            
            if should_reboot and self.can_reboot():
                _LOGGER.info(f"Automatically rebooting charger {self.name} due to OCPP notification: {message}")
                await self.reboot(current_time)
                
            elif is_error:
                # For error conditions, we might want to reset the charger state or take other actions
                _LOGGER.info(f"Handling error condition for charger {self.name}: {message}")
                
                # Reset the charger's internal state if it's having issues
                if "connection" in message_lower or "communication" in message_lower:
                    self._reset_state_machine()
                    _LOGGER.info(f"Reset state machine for charger {self.name} due to communication issues")
                
                # If the charger reports being unavailable, mark it as such
                if "unavailable" in message_lower or "offline" in message_lower:
                    self.status = "unavailable"
                    _LOGGER.info(f"Marked charger {self.name} as unavailable due to OCPP notification")
            
            # Log successful operations  
            elif any(keyword in message_lower for keyword in ["success", "completed", "ready", "available"]):
                _LOGGER.info(f"OCPP operation successful for charger {self.name}: {message}")
                
                # If charger is back online, ensure it's marked as available
                if "available" in message_lower or "ready" in message_lower:
                    self.status = "ok"
                    _LOGGER.info(f"Marked charger {self.name} as available due to OCPP notification")
            
            # Handle firmware update notifications
            elif "firmware" in message_lower:
                _LOGGER.info(f"Firmware-related notification for charger {self.name}: {message}")
                
                if "upload status" in message_lower and ("completed" in message_lower or "success" in message_lower):
                    _LOGGER.info(f"Firmware upload completed for charger {self.name}, scheduling reboot")
                    await self.reboot(current_time)
            
            # Handle diagnostic upload notifications
            elif "diagnostic" in message_lower:
                _LOGGER.info(f"Diagnostic-related notification for charger {self.name}: {message}")
            
            # General notification logging
            else:
                _LOGGER.info(f"General OCPP notification for charger {self.name}: {message}")
        except Exception as e:
            _LOGGER.error(f"Error handling OCPP notification for charger {self.name}: {e}")

    def convert_ocpp_current_import_amps_to_W(self, amps: float, attr:dict) -> (float, dict):

        val = amps * self.home.voltage

        new_attr = {}
        if attr is not None:
            new_attr = dict(attr)

        new_attr[ATTR_UNIT_OF_MEASUREMENT] = UnitOfPower.WATT

        return val, new_attr

    def low_level_plug_check_now(self, time: datetime) -> (bool|None, datetime):

        state = self.hass.states.get(self.charger_plugged)
        if state is not None:
            state_time = state.last_updated
        else:
            state_time = time

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            res_plugged = None
        else:
            res_plugged = state.state == "off" # if the car is plugged it won't be seen as "available" in OCPP, so we use "off" to mean "plugged in"

        return res_plugged, state_time

    def get_car_charge_enabled_status_vals(self) -> list[str]:
        return [
            QSOCPPv16ChargePointStatus.suspended_ev,
            QSOCPPv16ChargePointStatus.charging,
            QSOCPPv16ChargePointStatus.suspended_evse
        ]

    def get_car_plugged_in_status_vals(self) -> list[str]:
        return [
            QSOCPPv16ChargePointStatus.preparing,
            QSOCPPv16ChargePointStatus.charging,
            QSOCPPv16ChargePointStatus.suspended_ev,
            QSOCPPv16ChargePointStatus.suspended_evse,
            QSOCPPv16ChargePointStatus.finishing
        ]

    def get_car_status_unknown_vals(self) -> list[str]:
        return [QSOCPPv16ChargePointStatus.unavailable, QSOCPPv16ChargePointStatus.faulted]

    def get_car_stopped_asking_current_status_vals(self) -> list[str]:
        return [QSOCPPv16ChargePointStatus.suspended_ev]


    def get_car_status_rebooting_vals(self) -> list[str]:
        return [QSOCPPv16ChargePointStatus.unavailable]


class QSChargerWallbox(QSChargerGeneric):



    def __init__(self, **kwargs):
        self.charger_device_wallbox = kwargs.pop(CONF_CHARGER_DEVICE_WALLBOX, None)
        self.charger_wallbox_charging_power = None
        hass: HomeAssistant | None = kwargs.get("hass", None)

        if self.charger_device_wallbox is not None and hass is not None:

            try:
                device_registry_instance = device_registry.async_get(hass)
                device = device_registry_instance.async_get(self.charger_device_wallbox)
            except:
                device = None

            entity_reg = entity_registry.async_get(hass)
            entries = entity_registry.async_entries_for_device(entity_reg, self.charger_device_wallbox)

            kwargs[CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER] = self._find_charger_entity_id(device, entries, "number.", "_maximum_charging_current")
            kwargs[CONF_CHARGER_PAUSE_RESUME_SWITCH] = self._find_charger_entity_id(device, entries, "switch.", "_pause_resume")
            self.charger_wallbox_charging_power = self._find_charger_entity_id(device, entries, "sensor.", "_charging_power")
            kwargs[CONF_CHARGER_STATUS_SENSOR] = self._find_charger_entity_id(device, entries, "sensor.", "_status_description")
            # kwargs[CONF_CHARGER_THREE_TO_ONE_PHASE_SWITCH] = self._find_charger_entity_id(device, entries, "switch.", "_phase_switch")

        super().__init__(**kwargs)

        self.secondary_power_sensor = self.charger_wallbox_charging_power
        self.attach_power_to_probe(self.secondary_power_sensor)

        # the wallbox are starting charging right away
        self.initial_num_in_out_immediate = 2
        # self.do_reboot_on_phase_switch = True

    def low_level_plug_check_now(self, time: datetime) -> (bool|None, datetime):

        state = self.hass.states.get(self.charger_pause_resume_switch)
        if state is not None:
            state_time = state.last_updated
        else:
            state_time = time

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            res_plugged = False
        else:
            res_plugged = True

        return res_plugged, state_time


    def low_level_charge_check_now(self, time: datetime) -> bool | None:

        state = self.hass.states.get(self.charger_pause_resume_switch)
        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            res_charge_check = False
        else:
            res_charge_check = state.state == "on"

        return res_charge_check

    def get_car_charge_enabled_status_vals(self) -> list[str]:
        return [
            WallboxChargerStatus.CHARGING.value,
            WallboxChargerStatus.DISCHARGING.value,
            WallboxChargerStatus.WAITING_FOR_CAR.value,
            WallboxChargerStatus.WAITING.value
        ]

    def get_car_plugged_in_status_vals(self) -> list[str]:
        return [
            WallboxChargerStatus.CHARGING.value,
            WallboxChargerStatus.DISCHARGING.value,
            WallboxChargerStatus.PAUSED.value,
            WallboxChargerStatus.SCHEDULED.value,
            WallboxChargerStatus.WAITING_FOR_CAR.value,
            WallboxChargerStatus.WAITING.value,
            WallboxChargerStatus.LOCKED_CAR_CONNECTED.value,
            WallboxChargerStatus.WAITING_IN_QUEUE_POWER_SHARING.value,
            WallboxChargerStatus.WAITING_IN_QUEUE_POWER_BOOST.value,
            WallboxChargerStatus.WAITING_MID_FAILED.value,
            WallboxChargerStatus.WAITING_MID_SAFETY.value,
            WallboxChargerStatus.WAITING_IN_QUEUE_ECO_SMART.value]

    def get_car_stopped_asking_current_status_vals(self) -> list[str]:
        return [
            WallboxChargerStatus.WAITING_FOR_CAR.value
        ]

    def get_car_status_unknown_vals(self) -> list[str]:
        return [
            WallboxChargerStatus.UNKNOWN.value,
            WallboxChargerStatus.ERROR.value,
            WallboxChargerStatus.DISCONNECTED.value,
            WallboxChargerStatus.UPDATING.value
        ]

    def get_car_status_rebooting_vals(self) -> list[str]:
        return [
            WallboxChargerStatus.DISCONNECTED.value,
        ]
