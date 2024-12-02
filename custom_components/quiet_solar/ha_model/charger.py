import logging
from datetime import datetime, timedelta
from enum import StrEnum

from typing import Any, Callable, Awaitable

import pytz
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE, SERVICE_TURN_OFF, SERVICE_TURN_ON, ATTR_ENTITY_ID
from homeassistant.components import number, homeassistant
from homeassistant.components.wallbox.const import ChargerStatus

from ..const import CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, CONF_CHARGER_PAUSE_RESUME_SWITCH, \
    CONF_CHARGER_PLUGGED, CONF_CHARGER_MAX_CHARGE, CONF_CHARGER_MIN_CHARGE, CONF_CHARGER_IS_3P, \
    CONF_CHARGER_DEVICE_OCPP, CONF_CHARGER_DEVICE_WALLBOX, CONF_CHARGER_CONSUMPTION, CONF_CAR_CHARGER_MIN_CHARGE, \
    CONF_CAR_CHARGER_MAX_CHARGE, CONF_CHARGER_STATUS_SENSOR, CONF_CAR_BATTERY_CAPACITY, CONF_CALENDAR, \
    CHARGER_NO_CAR_CONNECTED, CONSTRAINT_TYPE_MANDATORY_END_TIME, CONSTRAINT_TYPE_FILLER_AUTO, \
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE, CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN, SENSOR_CONSTRAINT_SENSOR_CHARGE
from ..home_model.constraints import DATETIME_MIN_UTC, LoadConstraint, MultiStepsPowerLoadConstraintChargePercent
from ..ha_model.car import QSCar
from ..ha_model.device import HADeviceMixin, get_average_sensor, get_median_sensor
from ..home_model.commands import LoadCommand, CMD_AUTO_GREEN_ONLY, CMD_ON, CMD_OFF, copy_command, \
    CMD_AUTO_FROM_CONSIGN, CMD_IDLE
from ..home_model.load import AbstractLoad

_LOGGER = logging.getLogger(__name__)

CHARGER_STATE_REFRESH_INTERVAL = 3
CHARGER_ADAPTATION_WINDOW = 30
CHARGER_CHECK_STATE_WINDOW = 12

STATE_CMD_RETRY_NUMBER = 3
STATE_CMD_TIME_BETWEEN_RETRY = CHARGER_STATE_REFRESH_INTERVAL * 3

TIME_OK_BETWEEN_CHANGING_CHARGER_STATE = 60*10

class QSOCPPChargePointStatus(StrEnum):
    """ OCPP Charger Status Description."""
    AVAILABLE = "Available"
    PREPARING = "Preparing"
    CHARGING = "Charging"
    SUSPENDED_EVSE = "SuspendedEVSE"
    SUSPENDED_EV = "SuspendedEV"
    FINISHING = "Finishing"
    RESERVED = "Reserved"

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




class QSChargerGeneric(HADeviceMixin, AbstractLoad):

    def __init__(self, **kwargs):
        self.charger_plugged = kwargs.pop(CONF_CHARGER_PLUGGED, None)
        self.charger_max_charging_current_number = kwargs.pop(CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, None)
        self.charger_pause_resume_switch = kwargs.pop(CONF_CHARGER_PAUSE_RESUME_SWITCH, None)
        self.charger_max_charge = kwargs.pop(CONF_CHARGER_MAX_CHARGE, 32)
        self.charger_min_charge = kwargs.pop(CONF_CHARGER_MIN_CHARGE, 6)
        self.charger_default_idle_charge = min(self.charger_max_charge, max(self.charger_min_charge, 8))

        self.charger_is_3p = kwargs.pop(CONF_CHARGER_IS_3P, False)
        self.charger_consumption_W = kwargs.pop(CONF_CHARGER_CONSUMPTION, 50)

        self.charger_status_sensor = kwargs.pop(CONF_CHARGER_STATUS_SENSOR, None)

        self._internal_fake_is_plugged_id = "is_there_a_car_plugged"

        self._is_next_charge_full = False
        self._do_force_next_charge = False

        self.car: QSCar | None = None
        self._user_attached_car_name: str | None = None

        self.charge_state = STATE_UNKNOWN

        self._last_charger_state_prob_time = None

        super().__init__(**kwargs)

        data = {
            CONF_CAR_CHARGER_MIN_CHARGE: self.charger_min_charge,
            CONF_CAR_CHARGER_MAX_CHARGE: self.charger_max_charge,
            CONF_CAR_BATTERY_CAPACITY: 22000,
            CONF_CALENDAR: self.calendar
        }

        self._default_generic_car = QSCar(hass=self.hass, home=self.home, config_entry=None,
                                          name=f"{self.name} generic car", **data)

        self._inner_expected_charge_state: QSStateCmd | None = None
        self._inner_amperage: QSStateCmd | None = None
        self.reset()

        self.attach_ha_state_to_probe(self.charger_status_sensor,
                                      is_numerical=False)

        self.attach_ha_state_to_probe(self._internal_fake_is_plugged_id,
                                      is_numerical=False,
                                      non_ha_entity_get_state=self.is_plugged_state_getter)

        _LOGGER.info(f"Creating Charger: {self.name}")

        self._power_steps = []

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



    async def on_hash_state_change(self, time: datetime):

        if self.car:
            load_name = self.car.name
            mobile_app = self.car.mobile_app
            mobile_app_url = self.car.mobile_app_url
        else:
            load_name = self.name
            mobile_app = self.mobile_app
            mobile_app_url = self.mobile_app_url


        if isinstance(self, AbstractLoad):
            readable_state = self.get_active_readable_name(time, filter_for_human_notification=True)
        else:
            readable_state = "WRONG STATE"

        _LOGGER.info(f"Sending notification for Charger, car {load_name} app: {mobile_app} with: {readable_state}")

        if mobile_app is not None and readable_state is not None:

            data={
                "title": f"What will happen for {load_name}?",
                "message": f"{readable_state}",
            }
            if mobile_app_url is not None:
                data["data"] = {}
                data["data"]["url"] = mobile_app_url
                data["data"]["clickAction"] = mobile_app_url

            _LOGGER.info(f"Full Sending notification for Charger, car {load_name} app: {mobile_app} with: {data}")

            await self.hass.services.async_call(
                domain=Platform.NOTIFY,
                service=mobile_app,
                service_data=data,
            )


    def get_best_car(self, time: datetime) -> QSCar | None:
        # find the best car ....

        if self._user_attached_car_name is not None:
            if self._user_attached_car_name != CHARGER_NO_CAR_CONNECTED:
                car = self.home.get_car_by_name(self._user_attached_car_name)
                if car is not None:
                    _LOGGER.info(f"Best Car from user selection: {car.name}")
                    return car
            else:
                _LOGGER.info(f"NO GOOD CAR BECAUSE: CHARGER_NO_CAR_CONNECTED")
                return None

        best_car = self.get_default_car()
        best_score = 0

        for car in self.home._cars:

            score = 0
            if car.is_car_plugged(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW) and car.is_car_home(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW):
                # only if plugged .... then if home
                score += 1
                if car.car_is_default:
                    score += 1

            if score > best_score:
                best_car = car
                best_score = score

        if best_score == 0:
            _LOGGER.debug(f"Default best car used: {best_car.name}")
        else:
            _LOGGER.debug(f"Best Car: {best_car.name} with score {best_score}")
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



    async def force_charge_now(self):
        self._do_force_next_charge = True


    async def check_load_activity_and_constraints(self, time: datetime) -> bool:
        # check that we have a connected car, and which one, or that it is completely disconnected
        #  if there is no more car ... just reset

        do_force_solve = False
        if self.is_not_plugged(time, for_duration=CHARGER_CHECK_STATE_WINDOW) and self.car:
            _LOGGER.info(f"unplugged connected car {self.car.name}: reset")
            self.reset()
            self._user_attached_car_name = None
            do_force_solve = True
        elif self.is_plugged(time, for_duration=CHARGER_CHECK_STATE_WINDOW):

            existing_constraints = []
            best_car = self.get_best_car(time)

            # user forced a "deconnection"
            if best_car is None:
                _LOGGER.info("plugged car with CHARGER_NO_CAR_CONNECTED selected option")
                self.reset()
                return True

            elif self.car:
                # there is already a car connected, check if it is the right one
                #change car we have a better one ...
                if best_car.name != self.car.name:
                    _LOGGER.info("CHANGE CONNECTED CAR!")
                    self.detach_car()

            if not self.car:
                # we may have some saved constraints that have been loaded already from the storage at init
                # so we need to check if they are still valid

                car = best_car
                _LOGGER.info(f"plugging car {car.name} not connected: reset and attach car")

                if self._constraints and self._do_force_next_charge is False:
                    # only keep user constraints...well in fact getting the as fast as possible one is enough
                    for ct in self._constraints:

                        if ct.from_user is False or ct.as_fast_as_possible is False:
                            continue

                        _LOGGER.info(f"Found a stored car constraint with {ct.load_param} for {car.name}")

                        if ct.load_param != car.name:
                            _LOGGER.info(f"Found a stored car constraint with {ct.load_param} for {car.name}, forcing to current car")
                            ct.reset_load_param(car.name)

                        existing_constraints.append(ct)

                # this reset is key it removes the constraint, the self._last_completed_constraint, etc
                # it will detach the car, etc
                self.reset()

                _LOGGER.info(f"plugged and no connected car: reset and attach car {car.name}")
                # find the best car .... for now
                self.attach_car(car)


            car_initial_percent = self.car.get_car_charge_percent(time)
            if car_initial_percent is None: # for possible percent issue
                car_initial_percent = 0.0
                _LOGGER.info(f"plugged car {self.car.name} as a None car_initial_percent... force init at 0")

            target_charge = await self.setup_car_charge_target_if_needed()

            if existing_constraints:
                for ct in existing_constraints:
                    if ct.target_value != target_charge:
                        do_force_solve = True
                    ct.target_value = target_charge
                    self.push_live_constraint(time, ct)

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
                        f"plugged car {self.car.name} pushed forces constraint {force_constraint.name}")
                    do_force_solve = True
            else:
                # we may have a as fast as possible constraint still active ... if so we need to update it
                for ct in self._constraints:
                    if ct.is_constraint_active_for_time_period(time):
                        if ct.as_fast_as_possible:
                            force_constraint = ct
                            if force_constraint.target_value != target_charge:
                                do_force_solve = True
                            force_constraint.target_value = target_charge
                            break

            if force_constraint is not None:
                self._do_force_next_charge = False
                realized_charge_target = target_charge

            start_time, end_time = await self.car.get_next_scheduled_event(time, after_end_time=True)
            # only add it if it is "after" the end of the forced constraint
            if start_time is not None and (force_constraint is None or start_time > force_constraint.end_of_constraint):

                if time > start_time:
                    # it means .... we are in the middle of the scheduled event, in theory it is passed, do nothing
                    # if there is a mandatory one unfinished it should have been pushed already to try to complete it
                    _LOGGER.info(
                        f"plugged car {self.car.name} NOT pushing mandatory constraint as we are in the middle of a scheduled event start:{start_time} end:{end_time} time:{time}")
                else:
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
                            f"plugged car {self.car.name} pushed mandatory constraint {car_charge_mandatory.name}")

                    realized_charge_target = target_charge

            if realized_charge_target is None or realized_charge_target <= 99.9:

                if realized_charge_target is None:
                    realized_charge_target = car_initial_percent
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
                        f"plugged car {self.car.name} pushed filler constraint {car_charge_best_effort.name}")


        return do_force_solve

    @property
    def min_charge(self):
        if self.car:
            return int(max(self.charger_min_charge, self.car.car_charger_min_charge))
        else:
            return self.charger_min_charge

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
        parent.update([Platform.SENSOR, Platform.SELECT, Platform.SWITCH,Platform.BUTTON])
        return list(parent)

    def attach_car(self, car):

        if self.car is not None:
            if self.car.name == car.name:
                return
            self.detach_car()

        self.car = car

        # reset dampening to conf values, and some states
        car.reset()

        if car.calendar is None:
            car.calendar = self.calendar

        power_steps, min_charge, max_charge = self.car.get_charge_power_per_phase_A(self.charger_is_3p)

        steps = []
        for a in range(min_charge, max_charge + 1):
            steps.append(copy_command(CMD_AUTO_GREEN_ONLY, power_consign=power_steps[a]))

        self._power_steps = steps

    def detach_car(self):
        self.car = None
        self._power_steps = []


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
            res_car =  self.car.is_car_plugged(time, for_duration)
            if res_car is None:
                return latest_charger_valid_state == ok_value
            else:
                if latest_charger_valid_state is not None:
                    if res_car is check_for_val and latest_charger_valid_state == ok_value:
                        return True
                    if res_car is not check_for_val and latest_charger_valid_state == not_ok_value:
                        return False
            return None

        return res


    def is_plugged(self, time: datetime, for_duration: float | None = None) -> bool | None:
        return self._check_plugged_val(time, for_duration, check_for_val=True)


    def is_not_plugged(self, time: datetime, for_duration: float | None = None) -> bool | None:
        return self._check_plugged_val(time, for_duration, check_for_val=False)

    def is_charger_plugged_now(self, time: datetime) -> [bool|None, datetime]:

        state = self.hass.states.get(self.charger_plugged)
        if state is not None:
            state_time = state.last_updated
        else:
            state_time = time

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return None, state_time

        return state.state == "on", state_time

    def get_car_charge_enabled_status_vals(self) -> list[str]:
        return []

    def check_charge_state(self, time: datetime, for_duration: float | None = None, check_for_val=True) -> bool | None:

        result = not check_for_val
        if self.is_plugged(time=time, for_duration=for_duration):

            status_vals = self.get_car_charge_enabled_status_vals()

            result = self._check_charger_status(status_vals, time, for_duration, invert_prob=not check_for_val)

            if result is not None:
                return result

            state = self.hass.states.get(self.charger_pause_resume_switch)
            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                result = None
            else:
                result = state.state == "on"

                if not check_for_val:
                    result = not result

        return result

    def is_charge_enabled(self, time: datetime, for_duration: float | None = None) -> bool | None:
        return self.check_charge_state(time, for_duration, check_for_val=True)

    def is_charge_disabled(self, time: datetime, for_duration: float | None = None) -> bool | None:
        return self.check_charge_state(time, for_duration, check_for_val=False)

    def get_car_stopped_asking_current_status_vals(self) -> list[str]:
        return []

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

        if self.current_command is None or self.car is None or self.is_not_plugged(time=time,
                                                                                   for_duration=CHARGER_CHECK_STATE_WINDOW):
            # if we reset here it will remove the current constraint list from the load!!!!
            _LOGGER.info(f"update_value_callback: reset because no car or not plugged")
            return (None, False)

        if self.is_not_plugged(time=time):
            # could be a "short" unplug
            _LOGGER.info(f"update_value_callback:short unplug")
            return (None, True)

        if self.current_command is None or self.current_command.is_off_or_idle():
            _LOGGER.info(f"update_value_callback:no command or idle/off")
            result = None
        else:
            result = self.car.get_car_charge_percent(time)
            result_calculus = None

            added_nrj = self.get_device_real_energy(start_time=ct.last_value_update, end_time=time,
                                                    clip_to_zero_under_power=self.charger_consumption_W)
            if added_nrj is not None and self.car.car_battery_capacity is not None and self.car.car_battery_capacity > 0:
                added_percent = (100.0 * added_nrj) / self.car.car_battery_capacity
                result_calculus = ct.current_value + added_percent

            _LOGGER.info(f"update_value_callback: sensor {result} and calculus {result_calculus}")

            if result_calculus is not None:
                if result is None:
                    result = result_calculus
                else:
                    if result > result_calculus + 10 and result < 98:
                        # in case the initial value was 0 for example because of a bad car % value
                        # reset the current value if now the result is valid
                        pass
                    else:
                        result = min(result_calculus, result)

        is_car_stopped_asked_current = self.is_car_stopped_asking_current(time=time, for_duration=CHARGER_ADAPTATION_WINDOW)

        if ct.target_value >= 100:
            # the only way when at a 100 to stop is that the car will stop asking current
            if is_car_stopped_asked_current:
                _LOGGER.info(f"update_value_callback: FINISH 100% target and car stop asking current")
                # force met constraint
                result = ct.target_value
            elif result is not None:
                # force to not finish until the car stopped asking current
                result = min(result, 99)
        elif ct.is_constraint_met(result) or is_car_stopped_asked_current:
                _LOGGER.info(f"update_value_callback: FINISH under 100%>{ct.target_value}% met:{ct.is_constraint_met(result)} stopped_asking: {is_car_stopped_asked_current}")
                # force met constraint
                result = ct.target_value

        if result is not None and ct.is_constraint_met(result):
            do_continue_constraint = False
        else:
            do_continue_constraint = True
            await self._compute_and_launch_new_charge_state(time, command=self.current_command, constraint=ct)

        return (result, do_continue_constraint)


    async def _compute_and_launch_new_charge_state(self, time, command: LoadCommand, constraint: LoadConstraint | None = None, probe_only: bool = False) -> bool:

        if self.car is None:
            return True

        init_amp = self._expected_amperage.value
        init_state = self._expected_charge_state.value

        if probe_only:
            constraint = None

        if self.is_car_stopped_asking_current(time, for_duration=CHARGER_CHECK_STATE_WINDOW):
            # this is wrong actually : we fix the car for CHARGER_ADAPTATION_WINDOW minimum ...
            # so the battery will adapt itself, let it do its job ... no need to touch its state at all!
            _LOGGER.info(f"_compute_and_launch_new_charge_state:car stopped asking current ... do nothing")
            if probe_only is False:
                self._expected_amperage.set(int(max(self.min_charge, self.charger_default_idle_charge)), time) # do not set charger_min_charge as it can be lower than what the car is asking only do that when stopping the charge
                self._expected_charge_state.set(True, time) # is it really needed? ... seems so to keep the box in the right state ?
        elif command.is_off_or_idle():
            self._expected_charge_state.set(False, time)
            self._expected_amperage.set(int(self.charger_default_idle_charge), time) # do not use charger min charge so next time we plug ...it may work
        elif command.is_like(CMD_ON):
            self._expected_amperage.set(self.max_charge, time)
            self._expected_charge_state.set(True, time)
        elif command.is_auto():

            if constraint is None:
                # so we are in execution for a command more than for a constraint update : we need stability
                # if nothing was set properly force it

                if self.is_in_state_reset():
                    # set default values
                    self._expected_charge_state.set(False, time)
                    self._expected_amperage.set(int(self.charger_default_idle_charge), time) # do not use charger min charge so next time we plug ...it may work
            else:

                res_ensure_state = await self._ensure_correct_state(time)

                # only take decision if the state is "good" for a while CHARGER_ADAPTATION_WINDOW
                if res_ensure_state and self._verified_correct_state_time is not None and (time - self._verified_correct_state_time).total_seconds() > CHARGER_ADAPTATION_WINDOW:

                    await self.setup_car_charge_target_if_needed(constraint.target_value)

                    # _LOGGER.info(f"update_value_callback compute")
                    current_power = 0.0

                    current_real_max_charging_power = self._expected_amperage.value
                    if self._expected_charge_state.value:
                        current_real_car_power = self.get_median_sensor(self.accurate_power_sensor, CHARGER_ADAPTATION_WINDOW / 2.0, time)
                        current_real_car_power = self.dampening_power_value_for_car_consumption(current_real_car_power)

                        # time to update some dampening car values:
                        if current_real_car_power is not None:
                            if constraint is not None:
                                _LOGGER.info( f"update_value_callback: dampening {current_real_max_charging_power}:{current_real_car_power}")
                                # this following function can change the power steps of the car
                                self.car.update_dampening_value(amperage=current_real_max_charging_power,
                                                                power_value=current_real_car_power,
                                                                for_3p=self.charger_is_3p,
                                                                time=time,
                                                                can_be_saved = (time - self._verified_correct_state_time).total_seconds() > 2*CHARGER_ADAPTATION_WINDOW)


                        else:
                            current_real_car_power = self.get_median_sensor(self.secondary_power_sensor,
                                                                            CHARGER_ADAPTATION_WINDOW, time)
                            current_real_car_power = self.dampening_power_value_for_car_consumption(current_real_car_power)

                        # we will compare now if the current need to be adapted compared to solar production
                        if current_real_max_charging_power >= self.min_charge:
                            current_power = current_real_car_power
                            if current_power is None:
                                power_steps, _, _ = self.car.get_charge_power_per_phase_A(self.charger_is_3p)
                                current_power = power_steps[int(current_real_max_charging_power)]
                        else:
                            current_power = 0.0

                    power_steps, _, _ = self.car.get_charge_power_per_phase_A(self.charger_is_3p)

                    available_power = self.home.get_available_power_values(CHARGER_ADAPTATION_WINDOW, time)
                    # the battery is normally adapting itself to the solar production, so if it is charging ... we will say that this powe is available to the car

                    # do we need a bit of a PID ? (proportional integral derivative? or keep it simple for now) or a convex hul with min / max?
                    # very rough estimation for now:

                    if available_power:
                        last_p_mean = get_average_sensor(available_power[-len(available_power) // 2:], last_timing=time)
                        all_p_mean = get_average_sensor(available_power, last_timing=time)
                        last_p_median = get_median_sensor(available_power[-len(available_power) // 2:], last_timing=time)
                        all_p_median = get_median_sensor(available_power, last_timing=time)

                        if command.is_like(CMD_AUTO_GREEN_ONLY):
                            target_delta_power = min(last_p_mean, all_p_mean, last_p_median, all_p_median)
                        else:
                            # mode CMD_AUTO_FROM_CONSIGN
                            target_delta_power = max(last_p_mean, all_p_mean, last_p_median, all_p_median)


                        auto_target_power = current_power + target_delta_power

                        higher_do_not_cross_power_boundary = auto_target_power
                        lower_do_not_go_below_power_boundary = None

                        if command.is_like(CMD_AUTO_FROM_CONSIGN):
                            if command.power_consign is not None and command.power_consign > 0:
                                # in CMD_AUTO_FROM_CONSIGN mode take always max possible power if available vs the computed consign
                                target_charge = max(power_steps[self.min_charge], command.power_consign)
                                lower_do_not_go_below_power_boundary = target_charge
                                if auto_target_power <= target_charge :
                                    # we do not have a do not cross boundary at all
                                    higher_do_not_cross_power_boundary = None
                                else:
                                    # we will try to not cross it
                                    higher_do_not_cross_power_boundary = auto_target_power


                        safe_powers_steps = power_steps[self.min_charge:self.max_charge + 1]

                        if higher_do_not_cross_power_boundary is not None and higher_do_not_cross_power_boundary <= 0:
                            #only relevant in case of CMD_AUTO_GREEN_ONLY
                            new_amp = 0
                        else:
                            #safe_powers_steps are now necessarily ordered,
                            best_lower = None
                            if lower_do_not_go_below_power_boundary:
                                for i, p in enumerate(safe_powers_steps):
                                    if p >= lower_do_not_go_below_power_boundary:
                                        best_lower = i
                                        break

                                if best_lower is None:
                                    best_lower = len(safe_powers_steps) - 1

                            best_higher = None
                            if higher_do_not_cross_power_boundary:
                                for i in range(len(safe_powers_steps) -1, -1, -1):
                                    if safe_powers_steps[i] <= higher_do_not_cross_power_boundary:
                                        best_higher = i
                                    else:
                                        break

                                if best_higher is None:
                                    best_higher = -1

                            # both can't be None
                            if best_lower is None:
                                if best_higher < 0:
                                    # nothing works, case o green only case ... stop charge
                                    new_amp = 0
                                else:
                                    new_amp = best_higher + self.min_charge
                            elif best_higher is None:
                                new_amp = best_lower + self.min_charge
                            else:
                                # best_lower can't be < 0 and as a good "max" at len(safe_powers_steps) - 1
                                # best_higher can be < 0
                                new_amp = max(best_higher, best_lower) + self.min_charge

                            # to smooth a bit going up for nothing
                            if command.is_like(CMD_AUTO_GREEN_ONLY):
                                if target_delta_power < 0:
                                    # it means no available power ... force to go down if not done
                                    if current_real_max_charging_power <= new_amp and current_power > 0 and self._expected_charge_state.value:
                                        new_amp = min(current_real_max_charging_power-1, new_amp-1)
                                        _LOGGER.info(f"Correct new_amp du to negative available power: {new_amp}")
                                else:
                                    delta_amp = new_amp - current_real_max_charging_power
                                    if delta_amp > 1:
                                        new_amp -= 1
                                        _LOGGER.info(f"Lower charge up speed: {new_amp}")


                        new_state = init_state
                        if new_amp < self.min_charge:
                            new_amp = self.charger_default_idle_charge # do not use charger min charge so next time we plug ...it may work
                            new_state = False
                        elif new_amp > self.max_charge:
                            new_amp = self.max_charge
                            new_state = True
                        else:
                            new_state = True

                        if init_state != new_state or new_amp != init_amp:
                            _LOGGER.info(f"target_delta_power {target_delta_power} target_powers {higher_do_not_cross_power_boundary}/{lower_do_not_go_below_power_boundary}, current_power {current_power} ")
                            _LOGGER.info(f"new_amp {new_amp} / init_amp {init_amp} new_state {new_state} / init_state {init_state}")
                            _LOGGER.info(f"car: {self.car.name} min charge {self.min_charge} max charge {self.max_charge}")
                            _LOGGER.info(f"power steps {safe_powers_steps}")

                        if init_state != new_state:
                           # we need to wait a bit before changing the state of the charger
                           if constraint is None or self._expected_charge_state.is_ok_to_set(time, TIME_OK_BETWEEN_CHANGING_CHARGER_STATE):
                                self._expected_charge_state.set(new_state, time)
                                if constraint:
                                    self.num_on_off += 1

                                if self._expected_charge_state.last_change_asked is None:
                                    _LOGGER.info(
                                        f"Change State: new_state {new_state} delta None > {TIME_OK_BETWEEN_CHANGING_CHARGER_STATE}s")
                                else:
                                    _LOGGER.info(
                                        f"Change State: new_state {new_state} delta {(time - self._expected_charge_state.last_change_asked).total_seconds()}s >= {TIME_OK_BETWEEN_CHANGING_CHARGER_STATE}s")

                           else:
                               new_amp = self.min_charge #force to minimum charge
                               _LOGGER.info(f"Forbid: new_state {new_state} delta {(time - self._expected_charge_state.last_change_asked).total_seconds()}s < {TIME_OK_BETWEEN_CHANGING_CHARGER_STATE}s")

                        if new_amp is not None:
                            self._expected_amperage.set(int(new_amp), time)

                    else:
                        _LOGGER.info(f"Available power invalid")


        if self.is_in_state_reset():
            _LOGGER.info(f"_compute_and_launch_new_charge_state: in state reset at the end .. force an idle like state")
            self._expected_charge_state.set(False, time)
            self._expected_amperage.set(int(self.charger_default_idle_charge), time)

        if init_amp != self._expected_amperage.value or init_state != self._expected_charge_state.value:
            _LOGGER.info(
                f"Change inner states values: change state to {int(self._expected_amperage.value)}A - charge:{self._expected_charge_state.value}")


        # do it all the time
        return await self._ensure_correct_state(time, probe_only=probe_only)

    async def execute_command(self, time: datetime, command: LoadCommand) -> bool | None:

        # force a homeassistant.update_entity service on the charger entity?
        await self._do_update_charger_state(time)
        is_plugged = self.is_plugged(time=time)
        if is_plugged is None:
            is_plugged = self.is_plugged(time, for_duration=CHARGER_CHECK_STATE_WINDOW)

        if is_plugged and self.car is not None:

            do_reset = True
            if self.current_command is None:
                do_reset = True
            elif command.is_auto() and self.current_command.is_auto():
                # well by construction , command and self.current_command are different
                if self.is_in_state_reset():
                    do_reset = True
                else:
                    # we are in the middle of the execution of probably the same constraint (or another one) but in a continuity of commands
                    do_reset = False

            if do_reset:
                self._reset_state_machine()
                _LOGGER.info(f"DO RESET Execute command {command.command}/{command.power_consign} on charger {self.name}")
                res = await self._compute_and_launch_new_charge_state(time, command=command, constraint=None)
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
            result = await self._compute_and_launch_new_charge_state(time, command=command, probe_only=True)
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

    def is_charger_plugged_now(self, time: datetime) -> [bool|None, datetime]:

        state = self.hass.states.get(self.charger_plugged)
        if state is not None:
            state_time = state.last_updated
        else:
            state_time = time

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return None, state_time

        return state.state == "off", state_time


    def get_car_charge_enabled_status_vals(self) -> list[str]:
        return [
            QSOCPPChargePointStatus.SUSPENDED_EV,
            QSOCPPChargePointStatus.CHARGING,
            QSOCPPChargePointStatus.SUSPENDED_EVSE
        ]

    def get_car_stopped_asking_current_status_vals(self) -> list[str]:
        return [QSOCPPChargePointStatus.SUSPENDED_EV]


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

    def is_charger_plugged_now(self, time: datetime) -> [bool|None, datetime]:

        state_wallbox = self.hass.states.get(self.charger_status_sensor)
        if state_wallbox is None or state_wallbox.state in [STATE_UNAVAILABLE]:
            #if other are not available, we can't know if the charger is plugged at all
            if state_wallbox is not None:
                state_time = state_wallbox.last_updated
            else:
                state_time = time
            return None, state_time
        else:
            state = self.hass.states.get(self.charger_pause_resume_switch)
            if state is not None:
                state_time = state.last_updated
            else:
                state_time = time

            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                return False, state_time
            return True, state_time

    def check_charge_state(self, time: datetime, for_duration: float | None = None, check_for_val=True) -> bool | None:

        result = not check_for_val
        if self.is_plugged(time=time, for_duration=for_duration):

            status_vals = self.get_car_charge_enabled_status_vals()

            result = self._check_charger_status(status_vals, time, for_duration, invert_prob=not check_for_val)

            if result is not None:
                return result

            state_wallbox = self.hass.states.get(self.charger_status_sensor)
            if state_wallbox is None or state_wallbox.state in [STATE_UNAVAILABLE]:
                return None

            state = self.hass.states.get(self.charger_pause_resume_switch)
            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                result = False
            else:
                result = state.state == "on"

                if not check_for_val:
                    result = not result

        return result

    def get_car_charge_enabled_status_vals(self) -> list[str]:
        return [
            ChargerStatus.CHARGING,
            ChargerStatus.DISCHARGING,
            ChargerStatus.WAITING_FOR_CAR,
            ChargerStatus.WAITING
        ]

    def get_car_stopped_asking_current_status_vals(self) -> list[str]:
        return [ChargerStatus.WAITING_FOR_CAR]

