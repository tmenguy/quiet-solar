import bisect
import logging
from datetime import datetime, timedelta
from enum import StrEnum
from functools import total_ordering
from tokenize import group

from typing import Any, Callable, Awaitable

import pytz
from datetime import time as dt_time
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE, SERVICE_TURN_OFF, SERVICE_TURN_ON, ATTR_ENTITY_ID
from homeassistant.components import number, homeassistant
from haversine import haversine, Unit
from sqlalchemy import false

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
    CONF_CHARGER_PLUGGED, CONF_CHARGER_MAX_CHARGE, CONF_CHARGER_MIN_CHARGE, CONF_IS_3P, \
    CONF_CHARGER_DEVICE_OCPP, CONF_CHARGER_DEVICE_WALLBOX, CONF_CHARGER_CONSUMPTION, CONF_CAR_CHARGER_MIN_CHARGE, \
    CONF_CAR_CHARGER_MAX_CHARGE, CONF_CHARGER_STATUS_SENSOR, CONF_CAR_BATTERY_CAPACITY, CONF_CALENDAR, \
    CHARGER_NO_CAR_CONNECTED, CONSTRAINT_TYPE_MANDATORY_END_TIME, CONSTRAINT_TYPE_FILLER_AUTO, \
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE, CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN, \
    SENSOR_CONSTRAINT_SENSOR_CHARGE, CONF_DEVICE_EFFICIENCY, DEVICE_CHANGE_CONSTRAINT, \
    DEVICE_CHANGE_CONSTRAINT_COMPLETED, CONF_CHARGER_LONGITUDE, CONF_CHARGER_LATITUDE
from ..home_model.constraints import DATETIME_MIN_UTC, LoadConstraint, MultiStepsPowerLoadConstraintChargePercent, \
    MultiStepsPowerLoadConstraint
from ..ha_model.car import QSCar
from ..ha_model.device import HADeviceMixin, get_average_sensor, get_median_sensor
from ..home_model.commands import LoadCommand, CMD_AUTO_GREEN_ONLY, CMD_ON, CMD_OFF, copy_command, \
    CMD_AUTO_FROM_CONSIGN, CMD_IDLE, CMD_AUTO_PRICE
from ..home_model.load import AbstractLoad

_LOGGER = logging.getLogger(__name__)

CHARGER_STATE_REFRESH_INTERVAL = 3
CHARGER_ADAPTATION_WINDOW = 30
CHARGER_CHECK_STATE_WINDOW = 12

STATE_CMD_RETRY_NUMBER = 3
STATE_CMD_TIME_BETWEEN_RETRY = CHARGER_STATE_REFRESH_INTERVAL * 3

TIME_OK_BETWEEN_CHANGING_CHARGER_STATE = 60*10





class QSChargerStates(StrEnum):
    PLUGGED = "plugged"
    UN_PLUGGED = "unplugged"


class QSStateCmd():

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = None
        self._num_launched = 0
        self._num_set = 0
        self.last_time_set = None
        self.last_change_asked = None

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

    def is_ok_to_set(self, time: datetime, min_change_time: float):

        if self.last_change_asked is None or time is None:
            return True

        # for initial in/out!
        if self._num_set <= 2:
            return True

        if (time - self.last_change_asked).total_seconds() > min_change_time:
            return True

        return False

    def success(self):
        self._num_launched = 0
        self.last_time_set = None

    def is_ok_to_launch(self, value, time: datetime):

        self.set(value, time)

        if self._num_launched == 0:
            return True

        if self._num_launched > STATE_CMD_RETRY_NUMBER:
            return False

        if self.last_time_set is None:
            return True

        if self.last_time_set is not None and (
                time - self.last_time_set).total_seconds() > STATE_CMD_TIME_BETWEEN_RETRY:
            return True

        return False

    def register_launch(self, value, time: datetime):
        self.set(value, time)
        self._num_launched += 1
        self.last_time_set = time


class QSChargerStatus(object):

    def __init__(self, charger):
        self.charger = charger
        self.power_steps = None
        self.accurate_current_power = None
        self.secondary_current_power =None
        self.command = None
        self.current_real_max_charging_amp = None
        self.state_change_allowed = None
        self.possible_amps = None
        self.best_power_measure = None
        self._budgeted_amp = None
        self.budgeted_power = None

    @property
    def budgeted_amp(self):
        return self._budgeted_amp

    @budgeted_amp.setter
    def budgeted_amp(self, value):
        self._budgeted_amp = value
        if value == self.current_real_max_charging_amp:
            self.budgeted_power = self.best_power_measure
        else:
            self.budgeted_power = self.power_steps[value]

    def get_diff_power(self, old_amp, new_amp):

        diff_power = self.charger.get_delta_dampened_power(old_amp, new_amp)

        if diff_power is None:
            # ok we need to use the power_steps or anyother stuff
            old_power = self.power_steps[old_amp]
            if old_amp == self.current_real_max_charging_amp:
                old_power = self.best_power_measure
            new_power = self.power_steps[new_amp]
            if new_amp == self.current_real_max_charging_amp:
                new_power = self.best_power_measure

            diff_power = new_power - old_power

        return diff_power




    def can_increase_budget(self, allow_state_change=True):

        next_amp = None
        if self.budgeted_amp is None:
            return self.possible_amps[0]

        if self.budgeted_amp < self.possible_amps[-1] and self.possible_amps[-1] > 0:

            if self.budgeted_amp == 0:
                if allow_state_change:
                    if self.possible_amps[0] == 0:
                        next_amp = self.possible_amps[1]
                    else:
                        next_amp = self.possible_amps[0]
            else:
                next_amp = self.budgeted_amp + 1

        return next_amp

    def can_decrease_budget(self, allow_state_change=True):

        next_amp = None
        if self.budgeted_amp is None:
            return self.possible_amps[0]

        if self.budgeted_amp > self.possible_amps[0]:
            # self.budgeted_amp is so > 0 here
            if self.possible_amps[0] == 0:
                if self.budgeted_amp > self.possible_amps[1]:
                    next_amp = self.budgeted_amp - 1
                else:
                    if allow_state_change:
                        next_amp = 0
            else:
                # here self.budgeted_amp > self.possible_amps[0] and self.possible_amps[0] not 0
                next_amp = self.budgeted_amp - 1

        return next_amp


class QSChargerGroup(object):

    def __init__(self, dynamic_group: QSDynamicGroup ):
        self.dynamic_group : QSDynamicGroup = dynamic_group
        self._chargers : list[QSChargerGeneric] = []
        self.home = dynamic_group.home
        self.remaining_budget_to_apply = []
        self.know_reduced_state = None
        self.know_reduced_state_real_power = None

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

        if value < self.charger_consumption_W:
            return 0.0
        else:
            return value


    async def ensure_correct_state(self, time: datetime, probe_only=False) -> (bool, datetime|None):

        verified_correct_state_time = DATETIME_MIN_UTC
        for charger in self._chargers:
            res, vcst = await charger.ensure_correct_state(time, probe_only=probe_only)
            if res is False:
                return False, None
            if vcst is None:
                verified_correct_state_time = None
            else:
                if verified_correct_state_time is None:
                    pass
                elif vcst > verified_correct_state_time:
                    verified_correct_state_time = vcst

        return True, verified_correct_state_time

    def get_budget_diffs(self, actionable_chargers: list[QSChargerStatus]):

        initial_sum_amps = 0
        new_sum_amps = 0
        diff_power = 0
        for cs in actionable_chargers:
            initial_sum_amps += cs.current_real_max_charging_amp
            new_sum_amps += cs.budgeted_amp
            diff_power += cs.get_diff_power(cs.current_real_max_charging_amp, cs.budgeted_amp)

        return diff_power, new_sum_amps - initial_sum_amps, new_sum_amps



    async def dyn_handle(self, time: datetime):

        res_state, verified_correct_state_time = await self.ensure_correct_state(time)

        if res_state and self.remaining_budget_to_apply:
            # in case we would have had increases that could go above the limits because done before the decrease
            await self.apply_budgets(self.remaining_budget_to_apply, time)
            self.remaining_budget_to_apply = []
        # only take decision if the state is "good" for a while CHARGER_ADAPTATION_WINDOW, for all active chargers
        elif res_state and verified_correct_state_time is not None and (time - verified_correct_state_time).total_seconds() > CHARGER_ADAPTATION_WINDOW:

            # all chargers are now in a correct state, stable enough to do a computation

            # let's compute the current power consummed by all the chargers : we may have it on the group itself ... if it is made of the
            # chargers only ... or we will have to compute it by asking each chargers
            current_real_cars_power = self.dynamic_group.get_median_sensor(self.dynamic_group.accurate_power_sensor,
                                                                            CHARGER_ADAPTATION_WINDOW / 2.0, time)
            current_real_cars_power = self.dampening_power_value_for_car_consumption(current_real_cars_power)

            # time to update some dampening car values:
            if current_real_cars_power is not None:
                # ok we do have a value, can we infer teh dampening of the cars themselves?
                # we will see in teh "charger by charger" probe below if we can dampen or not
                # do that when in fact there was only one change versus the previous state to know what was the
                # incremental change due to the increase ....store this somewhere
                pass


            available_power = self.home.get_available_power_values(CHARGER_ADAPTATION_WINDOW, time)
            # the battery is normally adapting itself to the solar production, so if it is charging ... we will say that this power is available to the car

            if available_power:


                last_p_mean = get_average_sensor(available_power[-len(available_power) // 2:], last_timing=time)
                all_p_mean = get_average_sensor(available_power, last_timing=time)
                last_p_median = get_median_sensor(available_power[-len(available_power) // 2:], last_timing=time)
                all_p_median = get_median_sensor(available_power, last_timing=time)

                full_home_power = min(last_p_mean, all_p_mean, last_p_median, all_p_median) #if positive we are exporting solar , negative we are importing from the grid

                current_sum_amps = 0
                actionable_chargers = []

                sum_power_from_chargers = 0.0

                for charger in self._chargers:
                    cs = charger.get_stable_dynamic_charge_status(time)
                    if cs is not None:

                        if (charger._expected_charge_state.value is True and
                            cs.current_real_max_charging_amp >= charger.min_charge and
                            cs.accurate_current_power is not None and
                            cs.accurate_current_power > 0):
                            charger.update_car_dampening_value(time=time,
                                                               amperage_transition=cs.current_real_max_charging_amp,
                                                               power_value_or_delta=cs.accurate_current_power,
                                                               can_be_saved=((time - verified_correct_state_time).total_seconds() > 2 * CHARGER_ADAPTATION_WINDOW))

                        actionable_chargers.append(cs)
                        current_sum_amps += cs.current_real_max_charging_amp
                        sum_power_from_chargers += cs.best_power_measure

                if len(actionable_chargers) == 0:
                    _LOGGER.info(
                        f"dyn_handle: no actionable chargers")
                else:

                    mandatory_amps = 0
                    a_charging_cs = None
                    num_charging_cs = 0

                    current_reduced_states= {}
                    for cs in actionable_chargers:
                        # get all the minimums
                        mandatory_amps += cs.possible_amps[0]

                        current_reduced_states[cs.charger] = cs.current_real_max_charging_amp

                        if cs.charger._expected_charge_state.value is True and cs.current_real_max_charging_amp >= cs.charger.min_charge:
                            a_charging_cs = cs
                            num_charging_cs += 1

                    # we can update a bit the dampening for the only one charging if we do have a global counter for the group
                    if num_charging_cs == 1 and current_real_cars_power is not None:
                        a_charging_cs.charger.update_car_dampening_value(time=time,
                                                                         amperage_transition=a_charging_cs.current_real_max_charging_amp,
                                                                         power_value_or_delta=current_real_cars_power,
                                                                         can_be_saved=((time - verified_correct_state_time).total_seconds() > 2 * CHARGER_ADAPTATION_WINDOW))


                    # check the current state of teh chargers to see if we can try to map the delta power properly
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
                            # we can save the transition from self.know_reduced_state[c] to  current_reduced_states[c]
                            delta_power = current_real_cars_power - self.know_reduced_state_real_power
                            last_changed_charger.charger.update_car_dampening_value(time=time,
                                                                                    amperage_transition=(self.know_reduced_state[last_changed_charger], current_reduced_states[last_changed_charger]),
                                                                                    power_value_or_delta=delta_power,
                                                                                    can_be_saved=((time - verified_correct_state_time).total_seconds() > 2 * CHARGER_ADAPTATION_WINDOW))

                    # first bad case of amps too asked by the solver for example
                    if mandatory_amps > self.dynamic_group.dyn_group_max_phase_current:
                        # ouch .... we have to lower the charge of some cars
                        _LOGGER.warning(
                            f"dyn_handle: need to shave some auto consign")
                        shaved_amps = 0
                        # first shave amps as much as we can to get to a proper level
                        while True:
                            has_shaved = False
                            for cs in actionable_chargers:
                                if cs.possible_amps[0] > cs.charger.min_charge:
                                    # we can lower the charge
                                    cs.possible_amps.insert(0, cs.possible_amps[0] - 1)
                                    shaved_amps += 1
                                    has_shaved = True

                                if mandatory_amps - shaved_amps <= self.dynamic_group.dyn_group_max_phase_current:
                                    break

                            if has_shaved is False:
                                break

                            if mandatory_amps - shaved_amps <= self.dynamic_group.dyn_group_max_phase_current:
                                break
                        mandatory_amps -= shaved_amps

                        if mandatory_amps > self.dynamic_group.dyn_group_max_phase_current:
                            _LOGGER.warning(
                                f"dyn_handle: need to shave a lot more and force some quick down")

                            for step in ["stop_auto_only", "can_stop_all"]:

                                shaved_amps = 0
                                for cs in actionable_chargers:
                                    can_stop = True
                                    if step == "stop_auto_only":
                                        can_stop = cs.command.is_auto()
                                    if cs.possible_amps[0] >= cs.charger.min_charge and can_stop:
                                        # we can lower the charge
                                        shaved_amps += cs.possible_amps[0]
                                        cs.possible_amps.insert(0,0)

                                    if mandatory_amps - shaved_amps <= self.dynamic_group.dyn_group_max_phase_current:
                                        break

                                mandatory_amps -= shaved_amps
                                if mandatory_amps <= self.dynamic_group.dyn_group_max_phase_current:
                                    break

                    # ok now we know we are sure we can allocate
                    if current_real_cars_power is not None:
                        current_power = current_real_cars_power
                    else:
                        current_power = sum_power_from_chargers



                    best_global_command = CMD_AUTO_GREEN_ONLY
                    for cs in actionable_chargers:
                        if cs.command.is_like(CMD_AUTO_PRICE) or cs.command.is_like(CMD_AUTO_FROM_CONSIGN):
                            best_global_command = CMD_AUTO_PRICE
                            break

                    _LOGGER.info(f"full_home_power {full_home_power}, current_power {current_power}, power_budget {full_home_power+current_power}")

                    # await self.budgeting_algorithm_predictive(actionable_chargers, best_global_command, current_power, full_home_power)
                    await self.budgeting_algorithm_minimize_diffs(actionable_chargers, best_global_command, current_power, full_home_power)

                    diff_power_budget, diff_amp_budget, alloted_amps = self.get_budget_diffs(actionable_chargers)

                    if best_global_command == CMD_AUTO_PRICE:
                        if self.home.battery_can_discharge() is False and alloted_amps < self.dynamic_group.dyn_group_max_phase_current and full_home_power > 0:
                            # we will compute here is the price to take "more" power is better than the best
                            # electricity rate we may have

                            best_price = self.home.get_best_tariff(time)
                            durations_eval_s = 2 * CHARGER_ADAPTATION_WINDOW
                            current_price = self.home.get_tariff(time, time + timedelta(seconds=durations_eval_s))

                            # find the smallest possible increment in power
                            smallest_power_increment = None
                            cs_to_update = None
                            new_amp_budget = None
                            for cs in actionable_chargers:

                                next_budgeted_amp = cs.can_increase_budget()
                                if next_budgeted_amp is not None:

                                    diff_power = cs.get_diff_power(cs.budgeted_amp, next_budgeted_amp)
                                    diff_amp = next_budgeted_amp - cs.budgeted_amp

                                    if alloted_amps + diff_amp <= self.dynamic_group.dyn_group_max_phase_current:

                                        if smallest_power_increment is None or diff_power < smallest_power_increment:
                                            smallest_power_increment = diff_power
                                            cs_to_update = cs
                                            new_amp_budget = next_budgeted_amp

                            if smallest_power_increment is not None:

                                additional_added_energy = (smallest_power_increment * durations_eval_s) / 3600.0
                                cost = ((diff_power_budget - full_home_power) * durations_eval_s) / 3600.0 * current_price
                                cost_per_watt_h = cost / additional_added_energy

                                if cost_per_watt_h < best_price:
                                    cs_to_update.budgeted_amp = new_amp_budget

                    await self.apply_budget_strategy(actionable_chargers, current_real_cars_power, time)



    async def budgeting_algorithm_minimize_diffs(self, actionable_chargers, best_global_command, current_power, full_home_power):

        alloted_amps = 0

        for cs in actionable_chargers:
            cs.budgeted_amp = max(cs.current_real_max_charging_amp, cs.possible_amps[0])
            alloted_amps += cs.budgeted_amp

        if alloted_amps > self.dynamic_group.dyn_group_max_phase_current:
            # ok we know it is possible to shave to reach the max phase current, due to the preparation before
            for allow_state_change in [False, True]:
                for cs in actionable_chargers:
                    next_amp = cs.can_decrease_budget(allow_state_change=allow_state_change)
                    if next_amp is not None:
                        alloted_amps += next_amp - cs.budgeted_amp
                        cs.budgeted_amp = next_amp

                    if alloted_amps <= self.dynamic_group.dyn_group_max_phase_current:
                        break
                if alloted_amps <= self.dynamic_group.dyn_group_max_phase_current:
                    break
            if alloted_amps > self.dynamic_group.dyn_group_max_phase_current:
                return False


        # ok we do have the "best" possible base for the chargers
        diff_power_budget, diff_amp_budget, alloted_amps = self.get_budget_diffs(actionable_chargers)

        power_budget = full_home_power - diff_power_budget

        # this algorithm will only try to move "a bit" the chargers to reach the power budget
        # if power_budget is negative : we need to go down and find the best charger to go down
        # if power_budget is positive : we need to go up to consume extra solar and find the best charger to go up

        stop_on_first_change = True
        if power_budget < 0:
            increase = False
            if diff_amp_budget > 1:
                stop_on_first_change = False
        else:
            increase = True

        do_stop = False
        for allow_state_change in [False, True]:
            for cs in actionable_chargers:
                if increase:
                    next_budgeted_amp = cs.can_increase_budget(allow_state_change=allow_state_change)
                else:
                    next_budgeted_amp = cs.can_decrease_budget(allow_state_change=allow_state_change)

                if next_budgeted_amp is not None:

                    diff_power = cs.get_diff_power(cs.budgeted_amp, next_budgeted_amp)
                    diff_amp = next_budgeted_amp - cs.budgeted_amp

                    if increase:
                        if power_budget - diff_power >= 0:
                            # ok good change we still have some power to give
                            pass
                        else:
                            next_budgeted_amp = None

                    if alloted_amps + diff_amp > self.dynamic_group.dyn_group_max_phase_current:
                        next_budgeted_amp = None

                    if next_budgeted_amp is not None:
                        power_budget -= diff_power
                        cs.budgeted_amp = next_budgeted_amp
                        alloted_amps += diff_amp

                        if stop_on_first_change:
                            do_stop = True
                            break

                        if increase is False and power_budget >= 0:
                            do_stop = True
                            break

                if do_stop:
                    break

        return True


    async def budgeting_algorithm_predictive(self, actionable_chargers, best_global_command, current_power,
                                             full_home_power):
        alloted_power = 0.0
        alloted_amps = 0
        current_sum_amps = 0
        for cs in actionable_chargers:
            cs.budgeted_amp = cs.possible_amps[0]
            alloted_amps += cs.budgeted_amp
            alloted_power += cs.budgeted_power
            current_sum_amps += cs.current_real_max_charging_amp
        power_budget = full_home_power + current_power
        if alloted_power < power_budget and alloted_amps < self.dynamic_group.dyn_group_max_phase_current:
            # there is more power to be given!
            # to who we can give first ... obviously teh one that are not charging in the current budget now

            for step in ["budgeted_stopped_chargers_that_were_on_already", "already_stopped", "charging"]:
                for cs in actionable_chargers:
                    next_budgeted_amp = None
                    if step == "budgeted_stopped_chargers_that_were_on_already":
                        # try to stay in a charging state the ones already charging to reduce number or charge/discharge
                        if cs.budgeted_amp == 0 and cs.current_real_max_charging_amp > 0:
                            next_budgeted_amp = cs.can_increase_budget()
                    elif step == "already_stopped":
                        if cs.budgeted_amp == 0 and cs.current_real_max_charging_amp == 0:
                            next_budgeted_amp = cs.can_increase_budget()
                    else:
                        if cs.budgeted_amp > 0:
                            next_budgeted_amp = cs.can_increase_budget()

                    if next_budgeted_amp is not None:

                        diff_power = cs.get_diff_power(cs.budgeted_amp, next_budgeted_amp)
                        diff_amp = next_budgeted_amp - cs.budgeted_amp

                        if (alloted_power + diff_power <= power_budget and
                                alloted_amps + diff_amp <= self.dynamic_group.dyn_group_max_phase_current):

                            cs.budgeted_amp = next_budgeted_amp

                            alloted_power += diff_power
                            alloted_amps += diff_amp

                            if alloted_power >= power_budget:
                                break
                            if alloted_amps >= self.dynamic_group.dyn_group_max_phase_current:
                                break

                    if alloted_power >= power_budget:
                        break
                    if alloted_amps >= self.dynamic_group.dyn_group_max_phase_current:
                        break
                if alloted_power >= power_budget:
                    break
                if alloted_amps >= self.dynamic_group.dyn_group_max_phase_current:
                    break

            # boom we do have a new budget for the chargers ... cs.budgeted_amp
            # Now try to get a bit more for the AUTO_PRICE if we can (if there is ONE AUTO_PRICE ... do as if we were all auto price
            # same if only auto_green  if we have a too big up in number of added amps : reduce it, and shave the one that are growing the more

            if best_global_command == CMD_AUTO_GREEN_ONLY:
                # we do have some solar power to be used, lower up speed a bit if we grew too fast the amps
                if full_home_power > 0:
                    delta_amp = alloted_amps - current_sum_amps
                    if delta_amp > 1:
                        # shave a bit at least one amp
                        for cs in actionable_chargers:
                            if cs.budgeted_amp > 0:

                                next_budgeted_amp = cs.can_decrease_budget(allow_state_change=False)
                                if next_budgeted_amp is not None:
                                    diff_power = cs.get_diff_power(cs.budgeted_amp, next_budgeted_amp)
                                    diff_amp = next_budgeted_amp - cs.budgeted_amp

                                    cs.budgeted_amp = next_budgeted_amp

                                    alloted_power += diff_power
                                    alloted_amps += diff_amp

                                    _LOGGER.info(f"Lower charge up speed for solar")
                                    break

        return True

    async def apply_budget_strategy(self, actionable_chargers, current_real_cars_power, time):


        if len(actionable_chargers) == 0:
            return

        if current_real_cars_power is None:
            self.know_reduced_state = None
            self.know_reduced_state_real_power = None
        else:
            if len(actionable_chargers) > 0:
                self.know_reduced_state = {}
                for cs in actionable_chargers:
                    self.know_reduced_state[cs.charger] = cs.current_real_max_charging_amp
                self.know_reduced_state_real_power = current_real_cars_power

        max_amps_in_worst_case_scenario = 0
        for cs in actionable_chargers:
            max_amps_in_worst_case_scenario += max(cs.current_real_max_charging_amp, cs.budgeted_amp)

        if max_amps_in_worst_case_scenario > self.dynamic_group.dyn_group_max_phase_current:
            # even if the sum of amps is ok, we may have the sum of charger amps during the change that is too high
            # we will split the chargers in 2 groups, the one that are decreasing in amps and the one that are increasing

            increasing_cs = []
            decreasing_cs = []

            for cs in actionable_chargers:
                if cs.budgeted_amp > cs.current_real_max_charging_amp:
                    increasing_cs.append(cs)
                elif cs.budgeted_amp < cs.current_real_max_charging_amp:
                    decreasing_cs.append(cs)

            cs_to_apply = decreasing_cs
            self.remaining_budget_to_apply = increasing_cs
        else:
            cs_to_apply = actionable_chargers
            self.remaining_budget_to_apply = []

        if cs_to_apply:
            await self.apply_budgets(cs_to_apply, time)

    async def apply_budgets(self, actionable_chargers, time: datetime):
        for cs in actionable_chargers:

            init_state = cs.charger._expected_charge_state.value
            init_amp = cs.current_real_max_charging_amp
            new_amp = cs.budgeted_amp
            if new_amp < cs.charger.min_charge:
                new_amp = cs.charger.charger_default_idle_charge  # do not use charger min charge so next time we plug ...it may work
                new_state = False
            elif new_amp > cs.charger.max_charge:
                new_amp = cs.charger.max_charge
                new_state = True
            else:
                new_state = True


            if init_state != new_state or new_amp != init_amp:
                _LOGGER.info(
                    f"new_amp {new_amp} / init_amp {init_amp} new_state {new_state} / init_state {init_state}")
                _LOGGER.info(f"car: {cs.charger.car.name} min charge {cs.charger.min_charge} max charge {cs.charger.max_charge}")

            if init_state != new_state:
                # normally the state change has been checked already to allow or not a change of state in the
                # get_stable_current_charge_status with cs.state_change_allowed, and teh allowed amps
                # if it is done anyway it is because of too high amps for exp...
                cs.charger._expected_charge_state.set(new_state, time)
                cs.charger.num_on_off += 1

                if cs.charger._expected_charge_state.last_change_asked is None:
                    _LOGGER.info(
                        f"Change State: new_state {new_state} delta None > {TIME_OK_BETWEEN_CHANGING_CHARGER_STATE}s")
                else:
                    _LOGGER.info(
                        f"Change State: new_state {new_state} delta {(time - cs.charger._expected_charge_state.last_change_asked).total_seconds()}s >= {TIME_OK_BETWEEN_CHANGING_CHARGER_STATE}s")


            if new_amp is not None:
                cs.charger._expected_amperage.set(int(new_amp), time)

            await cs.charger._ensure_correct_state(time)





class QSChargerGeneric(HADeviceMixin, AbstractLoad):

    def __init__(self, **kwargs):
        self.charger_plugged = kwargs.pop(CONF_CHARGER_PLUGGED, None)
        self.charger_max_charging_current_number = kwargs.pop(CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, None)
        self.charger_pause_resume_switch = kwargs.pop(CONF_CHARGER_PAUSE_RESUME_SWITCH, None)
        self.charger_max_charge = kwargs.pop(CONF_CHARGER_MAX_CHARGE, 32)
        self.charger_min_charge = kwargs.pop(CONF_CHARGER_MIN_CHARGE, 6)
        self._charger_default_idle_charge = min(self.charger_max_charge, max(self.charger_min_charge, 8))

        self.charger_latitude = kwargs.pop(CONF_CHARGER_LATITUDE, None)
        self.charger_longitude = kwargs.pop(CONF_CHARGER_LONGITUDE, None)


        self.charger_consumption_W = kwargs.pop(CONF_CHARGER_CONSUMPTION, 50)

        self.charger_status_sensor = kwargs.pop(CONF_CHARGER_STATUS_SENSOR, None)

        self._internal_fake_is_plugged_id = "is_there_a_car_plugged"

        self._is_next_charge_full = False
        self._do_force_next_charge = False

        self.car: QSCar | None = None
        self._user_attached_car_name: str | None = None
        self.car_attach_time: datetime | None = None

        self.charge_state = STATE_UNKNOWN

        self._last_charger_state_prob_time = None

        self.default_charge_time : dt_time | None = None

        super().__init__(**kwargs)

        data = {
            CONF_CAR_CHARGER_MIN_CHARGE: self.charger_min_charge,
            CONF_CAR_CHARGER_MAX_CHARGE: self.charger_max_charge,
            CONF_CAR_BATTERY_CAPACITY: 100000,
            CONF_CALENDAR: self.calendar,
            CONF_DEVICE_EFFICIENCY: self.efficiency
        }

        self._default_generic_car = QSCar(hass=self.hass, home=self.home, config_entry=None,
                                          name=f"{self.name} generic car", **data)

        self._inner_expected_charge_state: QSStateCmd | None = None
        self._inner_amperage: QSStateCmd | None = None
        self.reset()


        self._unknown_state_vals = set()
        self._unknown_state_vals.update([STATE_UNKNOWN, STATE_UNAVAILABLE])
        self._unknown_state_vals.update(self.get_car_status_unknown_vals())

        self.attach_ha_state_to_probe(self.charger_status_sensor,
                                      is_numerical=False,
                                      state_invalid_values=self._unknown_state_vals)

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


    def get_virtual_current_constraint_translation_key(self) -> str | None:
        return SENSOR_CONSTRAINT_SENSOR_CHARGE

    async def set_next_charge_full_or_not(self, value: bool):
        self._is_next_charge_full = value

        new_target = await self.setup_car_charge_target_if_needed()

        if new_target and self._constraints:
            for ct in self._constraints:
                if isinstance(ct, MultiStepsPowerLoadConstraintChargePercent) and ct.is_mandatory:
                    ct.target_value = new_target

    async def setup_car_charge_target_if_needed(self, asked_target_charge=None):

        target_charge = asked_target_charge

        if target_charge is None:
            if self.is_next_charge_full():
                target_charge = 100
            else:
                if self.car:
                    target_charge = self.car.car_default_charge

            if self.car and target_charge is not None:
                await self.car.set_max_charge_limit(target_charge)

        return target_charge

    def is_next_charge_full(self) -> bool:
        return self._is_next_charge_full

    def get_update_value_callback_for_constraint_class(self, constraint:LoadConstraint) -> Callable[[LoadConstraint, datetime], Awaitable[tuple[float | None, bool]]] | None:

        class_name = constraint.__class__.__name__

        if class_name == MultiStepsPowerLoadConstraintChargePercent.__name__:
            return  self.constraint_update_value_callback_percent_soc

        return None

    def dampening_power_value_for_car_consumption(self, value: float) -> float | None:
        if value is None:
            return None

        if value < self.charger_consumption_W:
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
            self._inner_expected_charge_state = QSStateCmd()
        return self._inner_expected_charge_state

    @property
    def _expected_amperage(self):
        if self._inner_amperage is None:
            self._inner_amperage = QSStateCmd()
        return self._inner_amperage

    def get_stable_dynamic_charge_status(self, time: datetime):

        if self.car is None or self.is_not_plugged(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW):
            return None

        handled = self._probe_and_enforce_stopped_charge_command_state(time, command=self.current_command, probe_only=True)

        if handled:
            # the charger is in a "static" state and is not consumming any current
            return None

        cs = QSChargerStatus(self)

        cs.power_steps, _, _ = self.car.get_charge_power_per_phase_A(self.device_is_3p)
        cs.command = self.current_command
        if cs.command.is_like(CMD_ON):
            cs.command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=cs.power_steps[self.max_charge],
                                      phase_current=self.max_charge)

        cs.accurate_current_power = self.get_median_sensor(self.accurate_power_sensor, CHARGER_ADAPTATION_WINDOW / 2.0,
                                                           time)
        cs.accurate_current_power = self.dampening_power_value_for_car_consumption(cs.accurate_current_power)

        cs.secondary_current_power = self.get_median_sensor(self.secondary_power_sensor, CHARGER_ADAPTATION_WINDOW,
                                                            time)
        cs.secondary_current_power = self.dampening_power_value_for_car_consumption(cs.secondary_current_power)

        cs.current_real_max_charging_amp = self._expected_amperage.value

        if self._expected_charge_state.value is False:
            cs.current_real_max_charging_amp = 0

        cs.best_power_measure = 0.0
        if cs.accurate_current_power is not None:
            cs.best_power_measure =  cs.accurate_current_power
        elif cs.secondary_current_power is not None:
            cs.best_power_measure = cs.secondary_current_power
        elif self._expected_charge_state.value is True and cs.current_real_max_charging_amp >= self.min_charge:
            cs.best_power_measure = cs.power_steps[cs.current_real_max_charging_amp]


        current_state = True
        if cs.current_real_max_charging_amp == 0:
            current_state = False

        if cs.command.is_like(CMD_AUTO_FROM_CONSIGN):
            min_amps = self.min_charge
            if cs.command.phase_current is not None:
                min_amps = int(max(min_amps, round(cs.command.phase_current)))
            elif cs.command.power_consign is not None:
                min_amps = self.min_charge + bisect.bisect_left(cs.power_steps[self.min_charge:self.max_charge+1], cs.command.power_consign)

            min_amps = int(max(min_amps, self.min_charge))
            min_amps = int(min(min_amps, self.max_charge))

            possible_amps = [i for i in range(min_amps, self.max_charge + 1)]
        else:
            run_list = [i for i in range(self.min_charge, self.max_charge + 1)]
            # check if it is on or off : can we change from off to on or on to off because of allowed changes
            # check if we need to wait a bit before changing the state of the charger
            if self._expected_charge_state.is_ok_to_set(time, TIME_OK_BETWEEN_CHANGING_CHARGER_STATE):
                cs.state_change_allowed = True
                possible_amps = [0]
                possible_amps.extend(run_list)
            else:
                cs.state_change_allowed = False
                if current_state is False:
                    possible_amps = [0]
                else:
                    possible_amps = run_list

        cs.possible_amps = possible_amps
        return cs


    def reset(self):
        _LOGGER.info(f"Charger reset")
        super().reset()
        self.detach_car()
        self._reset_state_machine()
        self._do_force_next_charge = False

    def reset_load_only(self):
        _LOGGER.info(f"Charger reset only load")
        super().reset()

    def _reset_state_machine(self):
        self._verified_correct_state_time = None
        self._inner_expected_charge_state = None
        self._inner_amperage = None

    def is_in_state_reset(self) -> bool:
        return self._inner_expected_charge_state is None or self._inner_amperage is None or self._expected_charge_state.value is None or self._expected_amperage.value is None

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




    def get_best_car(self, time: datetime) -> QSCar | None:
        # find the best car ....


        cars = {}
        for charger in self.home._chargers:
            car = charger.car
            if car is not None and car != charger._default_generic_car:
                cars[car]= charger


        if self._user_attached_car_name is not None:
            if self._user_attached_car_name != CHARGER_NO_CAR_CONNECTED:
                car = self.home.get_car_by_name(self._user_attached_car_name)
                if car is not None:
                    _LOGGER.info(f"Best Car from user selection: {car.name}")

                    existing_charger = cars.get(car)

                    if existing_charger is not None and existing_charger != self:
                        # will force a reset of everyhting
                        existing_charger.detach_car()

                    return car
            else:
                _LOGGER.info(f"NO GOOD CAR BECAUSE: CHARGER_NO_CAR_CONNECTED")
                return None


        best_score = 0
        best_car = None


        for car in self.home._cars:

            score = 0

            existing_charger = cars.get(car)

            if existing_charger and existing_charger != self and existing_charger._user_attached_car_name == car.name:
                # this car is already attached to another charger, with a manual selection, can't be a candidate here
                continue

            if (existing_charger and
                    time - existing_charger.car_attach_time > timedelta(seconds=4*CHARGER_ADAPTATION_WINDOW)):
                # this car is already attached to another charger, for a long enough time, it is not a candidate
                if existing_charger == self:
                    # the current charger has been connected to this car for a long time, we can keep it ?
                    best_car = car
                    best_score = score
                    break
                else:
                    # we will let it on the other charger, this car is not a candidate anymore
                    continue


            score_plug_bump = 0
            car_plug_res = car.is_car_plugged(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW)
            if car_plug_res:
                score_plug_bump = 3
            elif car_plug_res is None:
                car_plug_res = car.is_car_plugged(time=time)
                if car_plug_res:
                    score_plug_bump = 2

            score_dist_bump = 0
            if self.charger_latitude is not None and self.charger_longitude is not None:
                max_dist = 50.0
                car_lat, car_long = car.get_car_coordinates(time)
                if car_lat is not None and car_long is not None:
                    dist = haversine((self.charger_latitude, self.charger_longitude), (car_lat, car_long), unit=Unit.METERS)
                    _LOGGER.info(f"Car {car.name} distance to charger {self.name}: {dist}m")
                    if dist <= max_dist:
                        score_dist_bump = 0.5*((max_dist - dist)/max_dist)

            score_home_bump = 0

            car_home_res = car.is_car_home(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW)
            if car_home_res:
                score_home_bump = 3
            elif car_home_res is None:
                car_home_res = car.is_car_home(time=time)
                if car_home_res:
                    score_home_bump= 2

            if score_dist_bump > 0 and score_home_bump == 0:
                car_home_res = True
                score_home_bump = 2


            if car_plug_res and car_home_res:
                # only if plugged .... then if home
                score = score_plug_bump + score_home_bump + score_dist_bump

            if score > best_score:
                best_car = car
                best_score = score


        if best_car is None:
            best_car = self.get_default_car()
            _LOGGER.info(f"Default best car used: {best_car.name}")
        else:
            _LOGGER.info(f"Best Car: {best_car.name} with score {best_score}")
            existing_charger = cars.get(best_car)

            if existing_charger is not None and existing_charger != self:
                # will force a reset of everyhting
                _LOGGER.info(f"Best Car for charger {self.name}: removed from another charger {existing_charger.name}")
                existing_charger.detach_car()
                # hoping we won't have back and forth between chargers
        return best_car

    def get_default_car(self):

        for car in self.home._cars:
            if car.car_is_default:
                return car

        return self._default_generic_car

    def get_car_options(self, time: datetime)  -> list[str]:

        if self.is_plugged(time, for_duration=CHARGER_CHECK_STATE_WINDOW):
            options = []
            for car in self.home._cars:
                options.append(car.name)

            options.extend([self._default_generic_car.name, CHARGER_NO_CAR_CONNECTED])
            return options
        else:
            self._user_attached_car_name = None
            return [CHARGER_NO_CAR_CONNECTED]

    def get_current_selected_car_option(self):
        if self.car is None:
            return CHARGER_NO_CAR_CONNECTED
        else:
            return self.car.name

    async def set_user_selected_car_by_name(self, time:datetime, car_name: str):
        self._user_attached_car_name = car_name
        if self._user_attached_car_name != self.get_current_selected_car_option():
            if await self.check_load_activity_and_constraints(time):
                self.home.force_next_solve()


    async def add_default_charge(self):
        if self.can_add_default_charge():
            if self.default_charge_time is not None:
                # compute the next occurency of the default charge time
                dt_now = datetime.now(tz=None)
                next_time = datetime(year=dt_now.year, month=dt_now.month, day=dt_now.day, hour=self.default_charge_time.hour, minute=self.default_charge_time.minute, second=self.default_charge_time.second)
                if next_time < dt_now:
                    next_time = next_time + timedelta(days=1)

                next_time = next_time.replace(tzinfo=None).astimezone(tz=pytz.UTC)

                await self.car.add_default_charge(next_time)

    def can_add_default_charge(self) -> bool:
        if self.car is not None and self.car.calendar is not None:
            return True
        return False

    def can_force_a_charge_now(self) -> bool:
        if self.car is not None:
            return True
        return False

    async def force_charge_now(self):
        self._do_force_next_charge = True


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

        if self.is_not_plugged(time, for_duration=CHARGER_CHECK_STATE_WINDOW) and self.car:
            _LOGGER.info(f"check_load_activity_and_constraints: unplugged connected car {self.car.name}: reset")
            existing_constraints = self.get_and_adapt_existing_constraints(time)
            self.reset()
            self._user_attached_car_name = None
            do_force_solve = True
            for ct in existing_constraints:
                self.push_live_constraint(time, ct)
        elif self.is_plugged(time, for_duration=CHARGER_CHECK_STATE_WINDOW):

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

                # find the best car .... for now
                self.attach_car(car, time)

            target_charge = await self.setup_car_charge_target_if_needed()

            car_current_charge_percent = car_initial_percent = self.car.get_car_charge_percent(time)
            if car_initial_percent is None: # for possible percent issue
                car_initial_percent = 0.0
                _LOGGER.info(f"check_load_activity_and_constraints: plugged car {self.car.name} as a None car_initial_percent... force init at 0")
            elif car_initial_percent >= target_charge:
                _LOGGER.info(f"check_load_activity_and_constraints: plugged car {self.car.name} as a car_initial_percent {car_initial_percent} >= target_charge {target_charge}... force init at {max(0, target_charge - 5)}")
                car_initial_percent = max(0, target_charge - 5)


            realized_charge_target = None
            # add a constraint ... for now just fill the car as much as possible
            force_constraint = None

            # in case a user pressed the button ....clean everything and force the charge
            if self._do_force_next_charge is True:
                do_force_solve = True
                self.reset_load_only() # cleanup any previous constraints to force this one!
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
                        f"check_load_activity_and_constraints: plugged car {self.car.name} pushed forces constraint {force_constraint.name}")
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
                self._do_force_next_charge = False
                realized_charge_target = target_charge
            else:
                # if we do have a last completed one it means there was no plug / un plug or reset in between
                start_time, end_time = await self.car.get_next_scheduled_event(time, after_end_time=True)
                is_passed_forced = force_constraint is not None and start_time is not None and start_time <= force_constraint.end_of_constraint
                if (start_time is not None and time > start_time) or is_passed_forced:
                    # not need at all to push any time constraint
                    _LOGGER.info(
                        f"check_load_activity_and_constraints: plugged car {self.car.name} NOT pushing time mandatory constraint: {start_time} end:{end_time} time:{time} is_passed_forced:{is_passed_forced}")
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
                        #as if there was a running one ... so the last filler is a full eco one
                        realized_charge_target = target_charge

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

            if realized_charge_target is None or realized_charge_target <= 99.9:

                if realized_charge_target is None:
                    # make car charging bigger than the battery filling if it is the only car constraint
                    type = CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN
                    realized_charge_target = car_initial_percent
                else:
                    #there was already a charge before and it is not full ... try solar only
                    type = CONSTRAINT_TYPE_FILLER_AUTO
                    target_charge = 100
                    realized_charge_target = 0

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
                        f"check_load_activity_and_constraints: plugged car {self.car.name} pushed filler constraint {car_charge_best_effort.name}")


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

    def detach_car(self):
        self.car = None
        self._power_steps = []
        self.car_attach_time = None

    # update in place the power steps
    def update_power_steps(self):
        if self.car:
            power_steps, min_charge, max_charge = self.car.get_charge_power_per_phase_A(self.device_is_3p)
            steps = []
            for a in range(min_charge, max_charge + 1):
                steps.append(copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=power_steps[a], phase_current=float(a)))
            self._power_steps = steps

            for ct in self._constraints:
                if isinstance(ct, MultiStepsPowerLoadConstraint):
                    ct.update_power_steps(steps)

    def get_min_max_power(self) -> (float, float):
        if len(self._power_steps) == 0 or self.car is None:
            return 0.0, 0.0

        return self._power_steps[0].power_consign, self._power_steps[-1].power_consign

    def get_min_max_phase_amps(self) -> (float, float):
        return self.min_charge, self.max_charge

    async def stop_charge(self, time: datetime):

        self._expected_charge_state.register_launch(value=False, time=time)
        charge_state = self.is_charge_enabled(time)

        if charge_state or charge_state is None:
            _LOGGER.info("STOP CHARGE LAUNCHED")
            try:
                await self.hass.services.async_call(
                    domain=Platform.SWITCH,
                    service=SERVICE_TURN_OFF,
                    target={ATTR_ENTITY_ID: self.charger_pause_resume_switch},
                    blocking=False
                )
            except:
                _LOGGER.info("STOP CHARGE LAUNCHED, EXCEPTION")

    async def start_charge(self, time: datetime):
        self._expected_charge_state.register_launch(value=True, time=time)
        charge_state = self.is_charge_disabled(time)
        if charge_state or charge_state is None:
            _LOGGER.info("START CHARGE LAUNCHED")
            try:
                await self.hass.services.async_call(
                    domain=Platform.SWITCH,
                    service=SERVICE_TURN_ON,
                    target={ATTR_ENTITY_ID: self.charger_pause_resume_switch},
                    blocking=False
                )
            except:
                _LOGGER.info("START CHARGE LAUNCHED, EXCEPTION")

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
                                                               num_seconds_before=2*for_duration,
                                                               time=time,
                                                               invert_val_probe=invert_prob)
        if contiguous_status is None:
            return None
        else:
            return contiguous_status >= for_duration and contiguous_status > 0

    def _check_plugged_val(self,
                           time: datetime,
                           for_duration: float | None = None,
                           check_for_val=True) -> bool | None:

        if for_duration is None or for_duration < 0:
            for_duration = 0

        contiguous_status = self.get_last_state_value_duration(self._internal_fake_is_plugged_id,
                                                               states_vals=[QSChargerStates.PLUGGED],
                                                               num_seconds_before=2 * for_duration,
                                                               time=time,
                                                               invert_val_probe=not check_for_val)
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


    def is_not_plugged(self, time: datetime, for_duration: float | None = None) -> bool | None:
        return self._check_plugged_val(time, for_duration, check_for_val=False)

    def is_charger_plugged_now(self, time: datetime) -> [bool|None, datetime]:

        if self.charger_status_sensor:
            state_wallbox = self.hass.states.get(self.charger_status_sensor)
            if state_wallbox is None or state_wallbox.state in self._unknown_state_vals:
                #if other are not available, we can't know if the charger is plugged at all
                if state_wallbox is not None:
                    state_time = state_wallbox.last_updated
                else:
                    state_time = time
                return None, state_time

            plugged_state_vals = self.get_car_plugged_in_status_vals()
            if plugged_state_vals:
                if state_wallbox.state in plugged_state_vals:
                    return True, state_wallbox.last_updated
                else:
                    return False, state_wallbox.last_updated


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



    def is_charge_enabled(self, time: datetime, for_duration: float | None = None) -> bool | None:
        return self.check_charge_state(time, for_duration, check_for_val=True)

    def is_charge_disabled(self, time: datetime, for_duration: float | None = None) -> bool | None:
        return self.check_charge_state(time, for_duration, check_for_val=False)

    def is_car_stopped_asking_current(self, time: datetime,
                                      for_duration: float | None = CHARGER_ADAPTATION_WINDOW) -> bool | None:

        result = False
        if self.is_plugged(time=time, for_duration=for_duration):

            result = None

            max_charging_power = self.get_max_charging_power()
            if max_charging_power is None:
                return None

            # check that in fact the car could receive something...if not it may be waiting for power but cant get it
            if max_charging_power < self.min_charge:
                return None

            status_vals = self.get_car_stopped_asking_current_status_vals()

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

    def is_charging_power_zero(self, time: datetime, for_duration: float) -> bool:
        val = self.get_median_power(for_duration, time)
        if val is None:
            return False

        return self.dampening_power_value_for_car_consumption(val) == 0.0  # 50 W of consumption for the charger for ex


    async def set_max_charging_current(self, current, time: datetime):

        self._expected_amperage.register_launch(value=current, time=time)

        if self.get_max_charging_power() != current:
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

        # await self.hass.services.async_call(
        #    domain=domain, service=service, service_data={number.ATTR_VALUE:int(min(max_value, max(min_value, range_value)))}, target={ATTR_ENTITY_ID: self.charger_max_charging_current_number}, blocking=blocking
        # )

    def get_max_charging_power(self):

        state = self.hass.states.get(self.charger_max_charging_current_number)

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            result = None
        else:
            try:
                result = int(state.state)
            except TypeError:
                result = None
            except ValueError:
                result = None

        return result

    def _is_state_set(self, time: datetime) -> bool:
        return self._expected_amperage.value is not None and self._expected_charge_state.value is not None


    async def ensure_correct_state(self, time: datetime, probe_only:bool = False) -> (bool, datetime | None):

        await self._do_update_charger_state(time)

        if self.car is None or self.is_not_plugged(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW):
            # if we reset here it will remove the current constraint list from the load!!!!
            _LOGGER.info(f"ensure_correct_state: no car or not plugged")
            return True, None

        if self.is_not_plugged(time=time):
            # could be a "short" unplug
            _LOGGER.info(f"ensure_correct_state: short unplug")
            return False, None # we don't know if final

        return await self._ensure_correct_state(time, probe_only), self._verified_correct_state_time




    async def _ensure_correct_state(self, time, probe_only:bool = False) -> bool:

        if self.is_in_state_reset():
            _LOGGER.info(f"Ensure State: no correct expected state")
            return False

        do_success = False
        max_charging_power = self.get_max_charging_power()
        if max_charging_power != self._expected_amperage.value:
            # check first if amperage setting is ok
            if probe_only is False:
                if self._expected_amperage.is_ok_to_launch(value=self._expected_amperage.value, time=time):
                    _LOGGER.info(f"Ensure State: current {max_charging_power}A expected {self._expected_amperage.value}A")
                    await self.set_max_charging_current(current=self._expected_amperage.value, time=time)
                else:
                    _LOGGER.debug(f"Ensure State: NOT OK TO LAUNCH current {max_charging_power}A expected {self._expected_amperage.value}A")
        else:
            is_charge_enabled = self.is_charge_enabled(time)
            is_charge_disabled = self.is_charge_disabled(time)

            if is_charge_enabled is None:
                _LOGGER.info(f"Ensure State: is_charge_enabled state unknown")
            if is_charge_disabled is None:
                _LOGGER.info(f"Ensure State: is_charge_disabled state unknown")

            if not ((self._expected_charge_state.value is True and is_charge_enabled) or (
                self._expected_charge_state.value is False and is_charge_disabled)):
                # acknowledge the charging power success above
                self._expected_amperage.success()

                if probe_only is False:
                    _LOGGER.info(f"Ensure State: expected {self._expected_charge_state.value} is_charge_enabled {is_charge_enabled} is_charge_disabled {is_charge_disabled}")
                    # if amperage is ok check if charge state is ok
                    if self._expected_charge_state.is_ok_to_launch(value=self._expected_charge_state.value, time=time):
                        if self._expected_charge_state.value:
                            _LOGGER.info(f"Ensure State: start_charge")
                            await self.start_charge(time=time)
                        else:
                            _LOGGER.info(f"Ensure State: stop_charge")
                            await self.stop_charge(time=time)
                    else:
                        _LOGGER.debug(f"Ensure State: NOT OK TO LAUNCH expected {self._expected_charge_state.value} is_charge_enabled {is_charge_enabled} is_charge_disabled {is_charge_disabled}")


            else:
                do_success = True

        if do_success:
            _LOGGER.debug(f"Ensure State: success amp {self._expected_amperage.value}")
            self._expected_charge_state.success()
            self._expected_amperage.success()
            if self._verified_correct_state_time is None:
                # ok we enter the state knowing where we are
                self._verified_correct_state_time = time
            return True
        else:
            self._verified_correct_state_time = None

        return False

    async def constraint_update_value_callback_percent_soc(self, ct: LoadConstraint, time: datetime) -> tuple[float | None, bool]:
        """ Example of a value compute callback for a load constraint. like get a sensor state, compute energy available for a car from battery charge etc
        it could also update the current command of the load if needed
        """
        await self._do_update_charger_state(time)

        if self.car is None or self.is_not_plugged(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW):
            # if we reset here it will remove the current constraint list from the load!!!!
            _LOGGER.info(f"update_value_callback: reset because no car or not plugged")
            return (None, False)

        if self.is_not_plugged(time=time):
            # could be a "short" unplug
            _LOGGER.info(f"update_value_callback:short unplug")
            return (None, True)

        result_calculus = None
        sensor_result = None

        if self.current_command is None or self.current_command.is_off_or_idle():
            _LOGGER.info(f"update_value_callback:no command or idle/off")
            result = None
        else:
            sensor_result = self.car.get_car_charge_percent(time)

            added_nrj = self.get_device_real_energy(start_time=ct.last_value_update, end_time=time,
                                                    clip_to_zero_under_power=self.charger_consumption_W)
            if added_nrj is not None and self.car.car_battery_capacity is not None and self.car.car_battery_capacity > 0:
                added_nrj = added_nrj/self.efficiency_factor # divide by efficiency factor as here we want to know what will be the impact in percent
                added_percent = (100.0 * added_nrj) / self.car.car_battery_capacity
                result_calculus = ct.current_value + added_percent

            result = sensor_result

            if result_calculus is not None:
                if sensor_result is None:
                    result = result_calculus
                else:
                    if sensor_result > result_calculus + 10 and sensor_result <= 99:
                        # in case the initial value was 0 for example because of a bad car % value
                        # reset the current value if now the result is valid
                        result = sensor_result
                    else:
                        result = min(result_calculus, sensor_result)

        is_car_charged, result = self.is_car_charged(time, current_charge=result, target_charge=ct.target_value)

        if result is not None and ct.is_constraint_met(result):
            do_continue_constraint = False
        else:
            do_continue_constraint = True
            await self.setup_car_charge_target_if_needed(ct.target_value)
            # await self._dynamic_compute_and_launch_new_charge_state(time)
            await self.charger_group.dyn_handle(time)

        _LOGGER.info(f"update_value_callback: {do_continue_constraint}/{result} ({sensor_result}/{result_calculus}) is_car_charged {is_car_charged} cmd {self.current_command}")

        return (result, do_continue_constraint)

    def is_car_charged(self, time: datetime,  current_charge: float | int | None,  target_charge: float | int) -> (bool, int|float):

        is_car_stopped_asked_current = self.is_car_stopped_asking_current(time=time,
                                                                          for_duration=CHARGER_ADAPTATION_WINDOW)
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
                if ct.is_constraint_met(current_charge):
                    # force met constraint
                    result = ct.target_value

        return result == target_charge, result

    def get_delta_dampened_power(self, from_amp: int | float, to_amp: int | float) -> float | None:
        power = None
        if self.car:
            power = self.car.get_delta_dampened_power(from_amp=from_amp, to_amp=to_amp, for_3p=self.device_is_3p)
        return power

    def update_car_dampening_value(self, time : datetime, amperage_transition: int|float|tuple[int,int]|tuple[float,float], power_value_or_delta: float, can_be_saved:bool=False):
        if self.car:
            if self.car.update_dampening_value(amperage_transition=amperage_transition, power_value_or_delta=power_value_or_delta, for_3p=self.device_is_3p, time=time, can_be_saved=can_be_saved):
                self.update_power_steps()

    def _probe_and_enforce_stopped_charge_command_state(self, time, command: LoadCommand, probe_only: bool = False) -> bool:

        if self.car is None:
            return True

        handled = False
        if self.is_car_stopped_asking_current(time, for_duration=CHARGER_CHECK_STATE_WINDOW):
            # this is wrong actually : we fix the car for CHARGER_ADAPTATION_WINDOW minimum ...
            # so the battery will adapt itself, let it do its job ... no need to touch its state at all!
            _LOGGER.info(f"_compute_and_launch_new_charge_state:car stopped asking current ... do nothing")
            handled = True
            if probe_only is False:
                self._expected_amperage.set(self.charger_default_idle_charge, time) # do not set charger_min_charge as it can be lower than what the car is asking only do that when stopping the charge
                self._expected_charge_state.set(True, time) # is it really needed? ... seems so to keep the box in the right state ?
        elif command is None or command.is_off_or_idle():
            handled = True
            if probe_only is False:
                self._expected_charge_state.set(False, time)
                self._expected_amperage.set(self.charger_default_idle_charge, time) # do not use charger min charge so next time we plug ...it may work
        elif command.is_auto() or command.is_like(CMD_ON):
            handled = False

        if self.is_in_state_reset():
            _LOGGER.info(f"_compute_and_launch_new_charge_state: in state reset at the end .. force an idle like state")
            handled = True
            if probe_only is False:
                self._expected_charge_state.set(False, time)
                self._expected_amperage.set(self.charger_default_idle_charge, time)

        return handled

    async def execute_command(self, time: datetime, command: LoadCommand) -> bool | None:

        # force a homeassistant.update_entity service on the charger entity?
        if command is None:
            return True

        await self._do_update_charger_state(time)
        is_plugged = self.is_plugged(time=time)
        if is_plugged is None:
            is_plugged = self.is_plugged(time, for_duration=CHARGER_CHECK_STATE_WINDOW)

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
        is_plugged = self.is_plugged(time=time)
        if is_plugged is None:
            is_plugged = self.is_plugged(time, for_duration=CHARGER_CHECK_STATE_WINDOW)

        result = None
        if is_plugged and self.car is not None:
            # need to call again _compute_and_launch_new_charge_state in case the car stopped asking for current in the middle
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
        if self._last_charger_state_prob_time is None or (time - self._last_charger_state_prob_time).total_seconds() > CHARGER_STATE_REFRESH_INTERVAL:
            await self.hass.services.async_call(
                homeassistant.DOMAIN,
                homeassistant.SERVICE_UPDATE_ENTITY,
                {ATTR_ENTITY_ID: [self.charger_pause_resume_switch, self.charger_max_charging_current_number]},
                blocking=False
            )
            self._last_charger_state_prob_time = time

    # ============================ INTERFACE TO BE OVERCHARGED ===================================== #

    def low_level_charge_check_now(self, time: datetime) -> bool | None:

        state = self.hass.states.get(self.charger_pause_resume_switch)
        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            res_charge_check = None
        else:
            res_charge_check = state.state == "on"

        return res_charge_check

    def low_level_plug_check_now(self, time: datetime) -> [bool | None, datetime]:

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

    def get_car_charge_enabled_status_vals(self) -> list[str]:
        return []

    def get_car_plugged_in_status_vals(self) -> list[str]:
        return []

    def get_car_status_unknown_vals(self) -> list[str]:
        return []

    def get_car_stopped_asking_current_status_vals(self) -> list[str]:
        return []



class QSChargerOCPP(QSChargerGeneric):

    def __init__(self, **kwargs):
        self.charger_device_ocpp = kwargs.pop(CONF_CHARGER_DEVICE_OCPP, None)
        self.charger_ocpp_current_import = None
        self.charger_ocpp_power_active_import = None

        hass: HomeAssistant | None = kwargs.get("hass", None)

        if self.charger_device_ocpp is not None and hass is not None:
            entity_reg = entity_registry.async_get(hass)
            entries = entity_registry.async_entries_for_device(entity_reg, self.charger_device_ocpp)

            for entry in entries:
                if entry.entity_id.startswith("number.") and entry.entity_id.endswith("_maximum_current"):
                    kwargs[CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER] = entry.entity_id

                if entry.entity_id.startswith("switch.") and entry.entity_id.endswith("_charge_control"):
                    kwargs[CONF_CHARGER_PAUSE_RESUME_SWITCH] = entry.entity_id

                if entry.entity_id.startswith("switch.") and entry.entity_id.endswith("_availability"):
                    kwargs[CONF_CHARGER_PLUGGED] = entry.entity_id

                # if entry.entity_id.startswith("sensor.") and entry.entity_id.endswith("_power_active_import"):
                #    self.charger_ocpp_power_active_import = entry.entity_id

                # OCPP only sensors :
                if entry.entity_id.startswith("sensor.") and entry.entity_id.endswith("_status_connector"):
                    self.charger_ocpp_status_connector = entry.entity_id

                if entry.entity_id.startswith("sensor.") and entry.entity_id.endswith("_status_connector"):
                    kwargs[CONF_CHARGER_STATUS_SENSOR] = entry.entity_id

                if entry.entity_id.startswith("sensor.") and entry.entity_id.endswith("_current_import"):
                    self.charger_ocpp_current_import = entry.entity_id

        super().__init__(**kwargs)

        self.secondary_power_sensor = self.charger_ocpp_current_import
        self.attach_power_to_probe(self.secondary_power_sensor, transform_fn=self.convert_amps_to_W)
        # self.attach_power_to_probe(self.charger_ocpp_power_active_import)


    def convert_amps_to_W(self, amps: float, attr:dict) -> float:
        # mult = 1.0
        # if self.charger_is_3p:
        #    mult = 3.0
        val = amps * self.home.voltage
        return val

    def low_level_plug_check_now(self, time: datetime) -> [bool|None, datetime]:

        state = self.hass.states.get(self.charger_plugged)
        if state is not None:
            state_time = state.last_updated
        else:
            state_time = time

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            res_plugged = None
        else:
            res_plugged = state.state == "off"

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


class QSChargerWallbox(QSChargerGeneric):
    def __init__(self, **kwargs):
        self.charger_device_wallbox = kwargs.pop(CONF_CHARGER_DEVICE_WALLBOX, None)
        self.charger_wallbox_charging_power = None
        hass: HomeAssistant | None = kwargs.get("hass", None)

        if self.charger_device_wallbox is not None and hass is not None:
            entity_reg = entity_registry.async_get(hass)
            entries = entity_registry.async_entries_for_device(entity_reg, self.charger_device_wallbox)

            for entry in entries:
                if entry.entity_id.startswith("number.") and entry.entity_id.endswith("_maximum_charging_current"):
                    kwargs[CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER] = entry.entity_id

                if entry.entity_id.startswith("switch.") and entry.entity_id.endswith("_pause_resume"):
                    kwargs[CONF_CHARGER_PAUSE_RESUME_SWITCH] = entry.entity_id

                if entry.entity_id.startswith("sensor.") and entry.entity_id.endswith("_charging_power"):
                    self.charger_wallbox_charging_power = entry.entity_id

                if entry.entity_id.startswith("sensor.") and entry.entity_id.endswith("_status_description"):
                    kwargs[CONF_CHARGER_STATUS_SENSOR] = entry.entity_id

        super().__init__(**kwargs)

        self.secondary_power_sensor = self.charger_wallbox_charging_power
        self.attach_power_to_probe(self.secondary_power_sensor)

    def low_level_plug_check_now(self, time: datetime) -> [bool|None, datetime]:

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

