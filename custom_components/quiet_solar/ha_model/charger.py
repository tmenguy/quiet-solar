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
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE, CONSTRAINT_TYPE_BEFORE_BATTERY_AUTO_GREEN
from ..home_model.constraints import MultiStepsPowerLoadConstraint, DATETIME_MIN_UTC, LoadConstraint, \
    MultiStepsPowerLoadConstraintChargePercent
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
        self.last_time_set = None
        self.last_change_asked = None

    def set(self, value, time: datetime):
        if time is None:
            self.last_change_asked = None

        if self.value != value:
            _LOGGER.info(f"QSStateCmd set with change from {self.value} to {value} at {time}")
            self.reset()
            self.value = value
            self.last_change_asked = time
            return True

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
        self.charger_is_3p = kwargs.pop(CONF_CHARGER_IS_3P, False)
        self.charger_consumption_W = kwargs.pop(CONF_CHARGER_CONSUMPTION, 50)

        self.charger_status_sensor = kwargs.pop(CONF_CHARGER_STATUS_SENSOR, None)

        self._internal_fake_is_plugged_id = "is_there_a_car_plugged"

        self._is_next_charge_full = False
        self._do_force_next_charge = False

        self.car: QSCar | None = None
        self._user_attached_car_name: str | None = None

        self.charge_state = STATE_UNKNOWN

        self._last_charger_state_prob_time = DATETIME_MIN_UTC

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



    def get_virtual_current_constraint_entity_name(self) -> str | None:
        if not isinstance(self, AbstractLoad):
            return None
        return f"Next Charge ({self.name})"

    def set_next_charge_full_or_not(self, value: bool):
        self._is_next_charge_full = value

        if value:
            new_target = 100
        else:
            new_target = self.car.car_default_charge

        for ct in self._constraints:
            if isinstance(ct, MultiStepsPowerLoadConstraintChargePercent) and ct.is_mandatory:
                ct.target_value = new_target


    def is_next_charge_full(self):
        if self.car is not None:
            if self.car.car_default_charge == 100:
                self.set_next_charge_full_or_not(True)

        return self._is_next_charge_full
    def get_update_value_callback_for_constraint_class(self, class_name:str) -> Callable[[LoadConstraint, datetime], Awaitable[float]] | None:

        if str == MultiStepsPowerLoadConstraintChargePercent.__name__:
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
        _LOGGER.info(f"charger reset")
        super().reset()
        self.detach_car()
        self._reset_state_machine()
        self._do_force_next_charge = False
        self._retry_check_constraint = 0

    def _reset_state_machine(self):
        self._verified_correct_state_time = None
        self._inner_expected_charge_state = None
        self._inner_amperage = None

    def get_best_car(self, time: datetime) -> QSCar:
        # find the best car .... for now default one

        best_car = self._default_generic_car
        best_score = 0

        for car in self.home._cars:

            score = 0
            if car.is_car_plugged(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW):
                score += 1
            if car.is_car_home(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW):
                score += 2

            if score > best_score:
                best_car = car
                best_score = score

        _LOGGER.info(f"Best Car: {best_car.name} with score {best_score}")

        return best_car

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
            await self.check_load_activity_and_constraints(time)



    async def force_charge_now(self):
        self._do_force_next_charge = True


    async def check_load_activity_and_constraints(self, time: datetime):
        # check that we have a connected car, and which one, or that it is completely disconnected
        #  if there is no more car ... just reset

        if self.is_not_plugged(time, for_duration=CHARGER_CHECK_STATE_WINDOW) and self.car:
            _LOGGER.info(f"unplugged connected car {self.car.name}: reset")
            self.reset()
            self._user_attached_car_name = None
        elif self.is_plugged(time, for_duration=CHARGER_CHECK_STATE_WINDOW):

            do_initial_constraints = False
            existing_constraints = []

            if self._user_attached_car_name is not None:
                if self._user_attached_car_name == CHARGER_NO_CAR_CONNECTED:
                    _LOGGER.info("plugged car with CHARGER_NO_CAR_CONNECTED selected option")
                    self.reset()
                    return

                if self.car is not None and self.car.name != self._user_attached_car_name:
                    self.detach_car()

            if self.car:
                car = self.get_best_car(time)
                if car.name != self.car.name:
                    _LOGGER.info("CHANGE CONNECTED CAR!")
                    self.detach_car()


            if not self.car:
                # we may have some saved constraints that have been loaded already from the storage at init
                # so we need to check if they are still valid

                if self._user_attached_car_name is not None and self._user_attached_car_name != CHARGER_NO_CAR_CONNECTED:
                    car = self.home.get_car_by_name(self._user_attached_car_name)
                else:
                    car = self.get_best_car(time)

                if self._constraints:
                    for ct in self._constraints:
                        if ct.from_user:
                            if ct.load_param == car.name:
                                existing_constraints.append(ct)
                            else:
                                # we may want to try again to check again the car if it is teh right one
                                self._retry_check_constraint +=1
                                if self._retry_check_constraint < 2:
                                    _LOGGER.info(f"retrying car attachement due to constraint mismatch {ct.load_param} != {car.name}")
                                    return # we will try again later

                self.reset()
                _LOGGER.info(f"plugged and no connected car: reset and attach car {car.name}")
                # find the best car .... for now
                self.attach_car(car)
                do_initial_constraints = True




            car_initial_percent = self.car.get_car_charge_percent(time)
            if car_initial_percent is None:
                car_initial_percent = 0.0

            if car_initial_percent < 99.5:

                if do_initial_constraints or self._do_force_next_charge:

                    # add a constraint ... for now just fill the car as much as possible
                    power_steps, min_charge, max_charge = self.car.get_charge_power_per_phase_A(self.charger_is_3p)

                    steps = []
                    for a in range(min_charge, max_charge + 1):
                        steps.append(copy_command(CMD_AUTO_GREEN_ONLY, power_consign=power_steps[a]))

                    realized_charge_target = None
                    if self._do_force_next_charge:

                        target_charge = self.car.car_default_charge
                        if self.is_next_charge_full():
                            target_charge = 100

                        realized_charge_target = target_charge
                        car_charge_mandatory = MultiStepsPowerLoadConstraintChargePercent(
                            total_capacity_wh=self.car.car_battery_capacity,
                            type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
                            time=time,
                            load=self,
                            load_param=self.car.name,
                            from_user=True,
                            initial_value=car_initial_percent,
                            target_value=target_charge,
                            power_steps=steps,
                            support_auto=True
                        )
                        self.push_live_constraint(time, car_charge_mandatory)
                        _LOGGER.info(
                            f"plugged car {self.car.name} pushed forces constraint {car_charge_mandatory.name}")
                        self._do_force_next_charge = False

                    if do_initial_constraints:

                        if realized_charge_target is None:
                            for ct in existing_constraints:
                                self.push_live_constraint(time, ct)

                        target_charge = self.car.car_default_charge

                        if self.is_next_charge_full():
                            target_charge = 100

                        if realized_charge_target is None or realized_charge_target < target_charge:

                            start_time, end_time = self.car.get_next_scheduled_event(time)

                            if start_time is not None and end_time is not None:
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
                                    power_steps=steps,
                                    support_auto=True
                                )
                                _LOGGER.info(
                                    f"plugged car {self.car.name} pushed mandatory constraint {car_charge_mandatory.name}")
                                self.push_live_constraint(time, car_charge_mandatory)
                                realized_charge_target = target_charge

                        if realized_charge_target is None or realized_charge_target < 100:

                            type = CONSTRAINT_TYPE_FILLER_AUTO
                            if realized_charge_target is None:
                                realized_charge_target = car_initial_percent
                                # make car charging bigger than the battery filling if it is the only car constraint
                                type = CONSTRAINT_TYPE_BEFORE_BATTERY_AUTO_GREEN

                            car_charge_best_effort = MultiStepsPowerLoadConstraintChargePercent(
                                total_capacity_wh=self.car.car_battery_capacity,
                                type=type,
                                time=time,
                                load=self,
                                load_param=self.car.name,
                                from_user=False,
                                initial_value=realized_charge_target,
                                target_value=100,
                                power_steps=steps,
                                support_auto=True
                            )
                            _LOGGER.info(f"plugged car {self.car.name} pushed filler constraint {car_charge_best_effort.name}")
                            self.push_live_constraint(time, car_charge_best_effort)

            else:
                _LOGGER.info(f"plugged car {self.car.name} already full: reset and detach car")
                self.reset()

        return

    @property
    def min_charge(self):
        return int(max(self.charger_min_charge, self.car.car_charger_min_charge))

    @property
    def max_charge(self):
        return int(min(self.charger_max_charge, self.car.car_charger_max_charge))

    def get_platforms(self):
        return [Platform.SENSOR, Platform.SELECT, Platform.SWITCH,Platform.BUTTON]

    def attach_car(self, car):

        if self.car is not None:
            if self.car.name == car.name:
                return
            self.detach_car()

        self.car = car
        if car.calendar is None:
            car.calendar = self.calendar

    def detach_car(self):
        self.car = None

    async def stop_charge(self, time: datetime):
        self._expected_charge_state.register_launch(value=False, time=time)
        if self.is_charge_enabled(time):
            _LOGGER.info("STOP CHARGE LAUNCHED")
            await self.hass.services.async_call(
                domain=Platform.SWITCH,
                service=SERVICE_TURN_OFF,
                target={ATTR_ENTITY_ID: self.charger_pause_resume_switch},
                blocking=False
            )

    async def start_charge(self, time: datetime):
        self._expected_charge_state.register_launch(value=True, time=time)
        if self.is_charge_disabled(time):
            _LOGGER.info("START CHARGE LAUNCHED")
            await self.hass.services.async_call(
                domain=Platform.SWITCH,
                service=SERVICE_TURN_ON,
                target={ATTR_ENTITY_ID: self.charger_pause_resume_switch},
                blocking=False
            )

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

        return contiguous_status >= for_duration and contiguous_status > 0

    def _check_plugged_val(self,
                           time: datetime,
                           for_duration: float | None = None,
                           check_for_val=True) -> bool:

        if for_duration is None or for_duration < 0:
            for_duration = 0

        contiguous_status = self.get_last_state_value_duration(self._internal_fake_is_plugged_id,
                                                               states_vals=[QSChargerStates.PLUGGED],
                                                               num_seconds_before=2 * for_duration,
                                                               time=time,
                                                               invert_val_probe=not check_for_val)

        return contiguous_status >= for_duration and contiguous_status > 0

    def is_plugged(self, time: datetime, for_duration: float | None = None) -> bool:
        return self._check_plugged_val(time, for_duration, check_for_val=True)

    def is_not_plugged(self, time: datetime, for_duration: float | None = None) -> bool:
        return self._check_plugged_val(time, for_duration, check_for_val=False)

    def is_charger_plugged_now(self, time: datetime) -> [bool, datetime]:

        state = self.hass.states.get(self.charger_plugged)
        if state is not None:
            state_time = state.last_updated
        else:
            state_time = time

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return False, state_time
        return state.state == "on", state_time

    def get_car_charge_enabled_status_vals(self) -> list[str]:
        return []

    def _check_charge_state(self, time: datetime, for_duration: float | None = None, check_for_val=True) -> bool:

        result = not check_for_val
        if self.is_plugged(time=time, for_duration=for_duration):

            status_vals = self.get_car_charge_enabled_status_vals()

            result = self._check_charger_status(status_vals, time, for_duration, invert_prob=not check_for_val)

            if result is not None:
                return result

            state = self.hass.states.get(self.charger_pause_resume_switch)
            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                result = False
            else:
                result = state.state == "on"

            if not check_for_val:
                result = not result

        return result

    def is_charge_enabled(self, time: datetime, for_duration: float | None = None) -> bool:
        return self._check_charge_state(time, for_duration, check_for_val=True)

    def is_charge_disabled(self, time: datetime, for_duration: float | None = None) -> bool:
        return self._check_charge_state(time, for_duration, check_for_val=False)

    def get_car_stopped_asking_current_status_vals(self) -> list[str]:
        return []

    def is_car_stopped_asking_current(self, time: datetime,
                                      for_duration: float | None = CHARGER_ADAPTATION_WINDOW) -> bool:

        result = False
        if self.is_plugged(time=time, for_duration=for_duration):

            status_vals = self.get_car_stopped_asking_current_status_vals()

            result = self._check_charger_status(status_vals, time, for_duration)

            if result is not None:
                return result

            if self.is_charge_enabled(time=time, for_duration=for_duration):
                return self.is_charging_power_zero(time=time, for_duration=for_duration)

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
            min_value = float(self.min_charge)
            max_value = float(self.max_charge)
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

    async def _ensure_correct_state(self, time) -> bool:

        if self._expected_amperage.value is None or self._expected_charge_state.value is None:
            _LOGGER.info(f"Ensure State: no correct expected state")
            return False

        do_success = False
        max_charging_power = self.get_max_charging_power()
        if max_charging_power != self._expected_amperage.value:
            # check first if amperage setting is ok
            if self._expected_amperage.is_ok_to_launch(value=self._expected_amperage.value, time=time):
                _LOGGER.info(f"Ensure State: current {max_charging_power}A expected {self._expected_amperage.value}A")
                await self.set_max_charging_current(current=self._expected_amperage.value, time=time)
            else:
                _LOGGER.debug(f"Ensure State: NOT OK TO LAUNCH current {max_charging_power}A expected {self._expected_amperage.value}A")

            self._verified_correct_state_time = None
        else:
            is_charge_enabled = self.is_charge_enabled(time)
            is_charge_disabled = self.is_charge_disabled(time)

            if is_charge_enabled is None:
                _LOGGER.info(f"Ensure State: is_charge_enabled state unknown")
            if is_charge_disabled is None:
                _LOGGER.info(f"Ensure State: is_charge_disabled state unknown")

            if not ((self._expected_charge_state.value is True and is_charge_enabled) or (
                self._expected_charge_state.value is False and is_charge_disabled)):
                # acknowledge the chariging power success above
                self._expected_amperage.success()
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

                self._verified_correct_state_time = None
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

        return False

    async def constraint_update_value_callback_percent_soc(self, ct: LoadConstraint, time: datetime) -> float | None:
        """ Example of a value compute callback for a load constraint. like get a sensor state, compute energy available for a car from battery charge etc
        it could also update the current command of the load if needed
        """

        await self._do_update_charger_state(time)

        if self.current_command is None or self.car is None or self.is_not_plugged(time=time,
                                                                                   for_duration=CHARGER_CHECK_STATE_WINDOW):
            self.reset()
            _LOGGER.info(f"update_value_callback: reset because no car or not plugged")
            return None

        if self.is_not_plugged(time=time):
            # could be a "short" unplug
            _LOGGER.info(f"update_value_callback:short unplug")
            return None

        if self.car.car_charge_percent_sensor is None:
            result = 0.0
        else:
            result = self.car.get_car_charge_percent(time)

        if self.is_car_stopped_asking_current(time=time, for_duration=CHARGER_ADAPTATION_WINDOW):
            # do we need to say that the car is not charging anymore? ... and so the constraint is ok?
            _LOGGER.info(f"update_value_callback:stop asking, set ct as target")
            result = ct.target_value

        if result is not None:
            if result > 99.8:
                result = ct.target_value
                _LOGGER.info(f"update_value_callback: a 100% reached")
            elif result >= ct.target_value:
                _LOGGER.info(f"update_value_callback: more than target {result} >= {ct.target_value}")


        await self._compute_and_launch_new_charge_state(time, command=self.current_command)

        return result

    async def _compute_and_launch_new_charge_state(self, time, command: LoadCommand, for_auto_command_init=False) -> bool:
        init_amp = self._expected_amperage.value
        init_state = self._expected_charge_state.value

        if for_auto_command_init:
            probe_duration = 0
            res_ensure_state = True
        else:
            probe_duration = CHARGER_ADAPTATION_WINDOW
            res_ensure_state = await self._ensure_correct_state(time)

        if self.is_car_stopped_asking_current(time, for_duration=probe_duration):
            # we can put back the battery as possibly discharging! as the car won't consume anymore soon ...

            # this is wrong actually : we fix the car for CHARGER_ADAPTATION_WINDOW minimum ...
            # so the battery will adapt itself, let it do its job ... no need to touch its state at all!

            _LOGGER.info(f"update_value_callback:car stopped asking current")
            self._expected_amperage.set(int(self.min_charge), time)
        elif command == CMD_OFF  or command == CMD_IDLE:
            self._expected_charge_state.set(False, time)
            self._expected_amperage.set(int(self.charger_min_charge), time)
        elif command == CMD_ON:
            self._expected_amperage.set(self.max_charge, time)
            self._expected_charge_state.set(True, time)
        elif command == CMD_AUTO_GREEN_ONLY or command == CMD_AUTO_FROM_CONSIGN:
            # only take decision if teh state is "good" for a while CHARGER_ADAPTATION_WINDOW
            if for_auto_command_init or (res_ensure_state and self._verified_correct_state_time is not None and (time - self._verified_correct_state_time).total_seconds() > CHARGER_ADAPTATION_WINDOW):


                current_power = 0.0

                current_real_max_charging_power = self._expected_amperage.value
                if self._expected_charge_state.value:
                    current_real_car_power = self.get_median_sensor(self.accurate_power_sensor, probe_duration / 2.0, time)
                    current_real_car_power = self.dampening_power_value_for_car_consumption(current_real_car_power)

                    # time to update some dampening car values:
                    if current_real_car_power is not None:
                        if not for_auto_command_init:
                            _LOGGER.info( f"update_value_callback: dampening {current_real_max_charging_power}:{current_real_car_power}")
                            # this following function can change the power steps of the car
                            self.car.update_dampening_value(amperage=current_real_max_charging_power,
                                                            power_value=current_real_car_power,
                                                            for_3p=self.charger_is_3p,
                                                            time=time,
                                                            can_be_saved = (time - self._verified_correct_state_time).total_seconds() > 2*CHARGER_ADAPTATION_WINDOW)


                    else:
                        current_real_car_power = self.get_median_sensor(self.secondary_power_sensor,
                                                                        probe_duration, time)
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

                available_power = self.home.get_available_power_values(probe_duration, time)
                # the battery is normally adapting itself to the solar production, so if it is charging ... we will say that this powe is available to the car

                # do we need a bit of a PID ? (proportional integral derivative? or keep it simple for now) or a convex hul with min / max?
                # very rough estimation for now:

                if available_power:
                    last_p_mean = get_average_sensor(available_power[-len(available_power) // 2:], last_timing=time)
                    all_p_mean = get_average_sensor(available_power, last_timing=time)
                    last_p_median = get_median_sensor(available_power[-len(available_power) // 2:], last_timing=time)
                    all_p_median = get_median_sensor(available_power, last_timing=time)

                    if command == CMD_AUTO_GREEN_ONLY:
                        target_delta_power = min(last_p_mean, all_p_mean, last_p_median, all_p_median)
                    else:
                        # mode CMD_AUTO_FROM_CONSIGN
                        target_delta_power = max(last_p_mean, all_p_mean, last_p_median, all_p_median)


                    target_power = current_power + target_delta_power

                    if command == CMD_AUTO_FROM_CONSIGN:
                        if command.power_consign is not None and command.power_consign >= 0:
                            # in CMD_AUTO_FROM_CONSIGN mode take always max possible power if available vs teh computed consign
                            target_power = max(power_steps[self.min_charge], max(command.power_consign, target_power))


                    safe_powers_steps = power_steps[self.min_charge:self.max_charge + 1]

                    if target_power <= 0:
                        #only relevant in case of CMD_AUTO_GREEN_ONLY
                        new_amp = 0
                    else:
                        #they are not necessarily ordered depends on the measures, etc
                        best_dist = 1000000000 # (1 GW :) )
                        best_i = -1
                        p_min_non_0 = 1000000000
                        p_max = -1
                        found_equal = False
                        for i,p in enumerate(safe_powers_steps):

                            if p > 0:
                                if p < p_min_non_0:
                                    p_min_non_0 = p
                            if p > p_max:
                                p_max = p

                            if p == target_power:
                                best_i = i
                                found_equal = True
                                break
                            elif target_power > p and abs(p - target_power) < best_dist:
                                best_dist = abs(p - target_power)
                                best_i = i


                        if best_i < 0:
                            if  target_power > p_max:
                                new_amp = self.max_charge
                            elif target_power < p_min_non_0:
                                if command == CMD_AUTO_GREEN_ONLY:
                                    new_amp = 0
                                else:
                                    # for CMD_AUTO_FROM_CONSIGN
                                    new_amp = self.min_charge
                            else:
                                new_amp = self.min_charge

                        elif found_equal:
                            new_amp = best_i + self.min_charge
                        else:
                            if command == CMD_AUTO_GREEN_ONLY:
                                new_amp = best_i + self.min_charge
                            else:
                                new_amp = min(self.max_charge, best_i + self.min_charge + 1)

                        # to smooth a bit going up for nothing
                        if command == CMD_AUTO_GREEN_ONLY:
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
                        new_amp = self.min_charge
                        new_state = False
                    elif new_amp > self.max_charge:
                        new_amp = self.max_charge
                        new_state = True
                    else:
                        new_state = True


                    _LOGGER.info(f"target_delta_power {target_delta_power} target_power {target_power}, current_power {current_power} ")
                    _LOGGER.info(f"new_amp {new_amp} / init_amp {init_amp} new_state {new_state} / init_state {init_state}")
                    _LOGGER.info(f"car: {self.car.name} min charge {self.min_charge} max charge {self.max_charge}")
                    _LOGGER.info(f"power steps {safe_powers_steps}")

                    if init_state != new_state:
                           if (self._expected_charge_state.last_change_asked is None or
                             (time - self._expected_charge_state.last_change_asked).total_seconds() >= TIME_OK_BETWEEN_CHANGING_CHARGER_STATE):
                                self._expected_charge_state.set(new_state, time)
                           else:
                               _LOGGER.info(f"Forbid: new_state {new_state} delta {(time - self._expected_charge_state.last_change_asked).total_seconds()}s < {TIME_OK_BETWEEN_CHANGING_CHARGER_STATE}s")

                    if new_amp is not None:
                        self._expected_amperage.set(int(new_amp), time)

                else:
                    _LOGGER.info(f"Available power invalid")



        if init_amp != self._expected_amperage.value or init_state != self._expected_charge_state.value:
            _LOGGER.info(
                f"Change inner states values: change state to {int(self._expected_amperage.value)}A - charge:{self._expected_charge_state.value}")

        # do it all the time
        return await self._ensure_correct_state(time)


    def set_state_machine_to_current_state(self, time: datetime):
        if self.is_charge_disabled(time):
            self._expected_charge_state.set(False, None)
        else:
            self._expected_charge_state.set(True, None)
        self._expected_amperage.set(self.get_max_charging_power(), time)

    async def execute_command(self, time: datetime, command: LoadCommand):

        # force a homeassistant.update_entity service on the charger entity?
        if self.is_plugged(time=time):
            # set us in a correct current state
            self._reset_state_machine()
            _LOGGER.info(f"Execute command {command.command} on charger {self.name}")
            self.set_state_machine_to_current_state(time)
            await self._compute_and_launch_new_charge_state(time, command, for_auto_command_init=True)
            self._last_charger_state_prob_time = time

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool:
        await self._do_update_charger_state(time)
        result = await self._compute_and_launch_new_charge_state(time, command, for_auto_command_init=False)
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

    def is_charger_plugged_now(self, time: datetime) -> [bool, datetime]:

        state = self.hass.states.get(self.charger_plugged)
        if state is not None:
            state_time = state.last_updated
        else:
            state_time = time

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return False, state_time
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

    def is_charger_plugged_now(self, time: datetime) -> [bool, datetime]:

        state = self.hass.states.get(self.charger_pause_resume_switch)
        if state is not None:
            state_time = state.last_updated
        else:
            state_time = time

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return False, state_time
        return True, state_time



    def get_car_charge_enabled_status_vals(self) -> list[str]:
        return [
            ChargerStatus.CHARGING,
            ChargerStatus.DISCHARGING,
            ChargerStatus.WAITING_FOR_CAR,
            ChargerStatus.WAITING
        ]

    def get_car_stopped_asking_current_status_vals(self) -> list[str]:
        return [ChargerStatus.WAITING_FOR_CAR]
