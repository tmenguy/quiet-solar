import bisect
import logging
from datetime import datetime
from enum import StrEnum
import random
from typing import Any

from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import entity_registry
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE, SERVICE_TURN_OFF, SERVICE_TURN_ON, ATTR_ENTITY_ID
from homeassistant.components import number, homeassistant
from homeassistant.components.wallbox.const import ChargerStatus

from ..const import CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, CONF_CHARGER_PAUSE_RESUME_SWITCH, \
    CONF_CHARGER_PLUGGED, CONF_CHARGER_MAX_CHARGE, CONF_CHARGER_MIN_CHARGE, CONF_CHARGER_IS_3P, \
    CONF_CHARGER_DEVICE_OCPP, CONF_CHARGER_DEVICE_WALLBOX, CONF_CHARGER_CONSUMPTION, CONF_CAR_CHARGER_MIN_CHARGE, CONF_CAR_CHARGER_MAX_CHARGE, CONF_CHARGER_STATUS_SENSOR
from ..home_model.constraints import MultiStepsPowerLoadConstraint, DATETIME_MIN_UTC, LoadConstraint
from ..ha_model.car import QSCar
from ..ha_model.device import HADeviceMixin, align_time_series_and_values, get_average_sensor
from ..home_model.commands import LoadCommand, CMD_AUTO_GREEN_ONLY, CMD_ON, CMD_OFF, copy_command
from ..home_model.load import AbstractLoad




_LOGGER = logging.getLogger(__name__)

CHARGER_STATE_REFRESH_INTERVAL = 4
CHARGER_ADAPTATION_WINDOW = 20


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


STATE_CMD_RETRY_NUMBER = 3
STATE_CMD_TIME_BETWEEN_RETRY = CHARGER_STATE_REFRESH_INTERVAL * 3


class QSStateCmd():

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = None
        self._num_launched = 0
        self._last_time_set = None

    def set(self, value):
        if self.value != value:
            self.reset()
            self.value = value
            return True

    def success(self):
        self._num_launched = 0
        self._last_time_set = None

    def is_ok_to_launch(self, value, time: datetime):

        self.set(value)

        if self._num_launched == 0:
            return True

        if self._num_launched > STATE_CMD_RETRY_NUMBER:
            return False

        if self._last_time_set is None:
            return True

        if self._last_time_set is not None and (
                time - self._last_time_set).total_seconds() > STATE_CMD_TIME_BETWEEN_RETRY:
            return True

        return False

    def register_launch(self, value, time: datetime):
        self.set(value)
        self._num_launched += 1
        self._last_time_set = time




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

        self.car: QSCar | None = None

        self.charge_state = STATE_UNKNOWN

        self._last_charger_state_prob_time = DATETIME_MIN_UTC

        super().__init__(**kwargs)

        data = {
            CONF_CAR_CHARGER_MIN_CHARGE: self.charger_min_charge,
            CONF_CAR_CHARGER_MAX_CHARGE: self.charger_max_charge,
        }

        self._default_generic_car = QSCar(hass=self.hass, home=self.home, config_entry=None,
                                          name=f"{self.name}_generic_car", **data)

        self._inner_expected_charge_state: QSStateCmd | None = None
        self._inner_amperage: QSStateCmd | None = None
        self.reset()

        self.attach_ha_state_to_probe(self.charger_status_sensor,
                                      is_numerical=False)

        self.attach_ha_state_to_probe(self._internal_fake_is_plugged_id,
                                      is_numerical=False,
                                      non_ha_entity_get_state=self.is_plugged_state_getter)

        _LOGGER.info(f"Creating Charger: {self.name}")

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
            if car.is_car_plugged(time=time, for_duration=CHARGER_ADAPTATION_WINDOW):
                score += 1
            if car.is_car_home(time=time, for_duration=CHARGER_ADAPTATION_WINDOW):
                score += 2

            if score > best_score:
                best_car = car
                best_score = score

        _LOGGER.info(f"Best Car: {best_car.name} with score {best_score}")

        return best_car

    async def check_load_activity_and_constraints(self, time: datetime):
        # check that we have a connected car, and which one, or that it is completely disconnected

        #  if there is no more car ... just reset
        if self.is_not_plugged(time, for_duration=CHARGER_ADAPTATION_WINDOW) and self.car:
            self.reset()
            _LOGGER.info(f"unplugged connected car: reset because no car or not plugged")
        elif self.is_plugged(time, for_duration=CHARGER_ADAPTATION_WINDOW) and not self.car:
            self.reset()
            _LOGGER.info(f"plugged and no connected car: reset and attach car")
            # find the best car .... for now
            c = self.get_best_car(time)
            self.attach_car(c)

            # add a constraint ... for now just fill the car as much as possible
            power_steps, min_charge, max_charge = self.car.get_charge_power_per_phase_A(self.charger_is_3p)

            steps = []
            for a in range(min_charge, max_charge + 1):
                steps.append(copy_command(CMD_AUTO_GREEN_ONLY, power_consign=power_steps[a]))

            car_charge_mandatory = MultiStepsPowerLoadConstraint(
                load=self,
                mandatory=True,
                end_of_constraint=None,
                initial_value=0,
                target_value=100,
                power_steps=steps,
                support_auto=True,
                update_value_callback=self.constraint_update_value_callback_percent_soc
            )
            self.push_live_constraint(car_charge_mandatory)
            await self.launch_command(time=time, command=CMD_AUTO_GREEN_ONLY)

        return

    @property
    def min_charge(self):
        return int(max(self.charger_min_charge, self.car.car_charger_min_charge))

    @property
    def max_charge(self):
        return int(min(self.charger_max_charge, self.car.car_charger_max_charge))

    def get_platforms(self):
        return [Platform.SENSOR, Platform.SELECT]

    def attach_car(self, car):
        self.car = car

    def detach_car(self):
        self.car = None

    async def stop_charge(self, time: datetime):
        self._expected_charge_state.register_launch(value=False, time=time)
        if self.is_charge_enabled(time):
            _LOGGER.info(f"STOP CHARGE LAUNCHED")
            await self.hass.services.async_call(
                domain=Platform.SWITCH,
                service=SERVICE_TURN_OFF,
                target={ATTR_ENTITY_ID: self.charger_pause_resume_switch},
                blocking=False
            )

    async def start_charge(self, time: datetime):
        self._expected_charge_state.register_launch(value=True, time=time)
        if self.is_charge_disabled(time):
            _LOGGER.info(f"START CHARGE LAUNCHED")
            await self.hass.services.async_call(
                domain=Platform.SWITCH,
                service=SERVICE_TURN_ON,
                target={ATTR_ENTITY_ID: self.charger_pause_resume_switch},
                blocking=False
            )

    def _check_charger_status(self, status_vals: list[str], time: datetime, for_duration: float | None = None,
                              invert_prob=False) -> bool | None:
        if not status_vals or self.charger_status_sensor is None:
            return None

        if for_duration is None or for_duration < 0:
            for_duration = 0

        contiguous_status = self.get_last_state_value_duration(self.charger_status_sensor,
                                                               states_vals=status_vals,
                                                               num_seconds_before=2 * for_duration,
                                                               time=time,
                                                               invert_val_probe=invert_prob)

        return contiguous_status >= for_duration and contiguous_status > 0

    def _check_plugged_val(self, time: datetime, for_duration: float | None = None, check_for_val=True) -> bool:

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

    async def _ensure_correct_state(self, time):

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
            _LOGGER.info(f"Ensure State: success amp {self._expected_amperage.value}")
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
                                                                                   for_duration=CHARGER_ADAPTATION_WINDOW):
            self.reset()
            _LOGGER.info(f"update_value_callback: reset because no car or not plugged")
            return None

        if self.is_not_plugged(time=time):
            # could be a "short" unplug
            _LOGGER.info(f"update_value_callback:short unplug")
            return None

        if self.car.car_charge_percent_sensor is None:
            result = 0.0
            if self.is_car_stopped_asking_current(time):
                # do we need to say that the car is not charging anymore? ... and so the constraint is ok?
                _LOGGER.info(f"update_value_callback:stop asking, set ct as target")
                result = ct.target_value

        else:
            state = self.hass.states.get(self.car.car_charge_percent_sensor)

            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                result = None
            else:
                result = float(state.state)

        if result is not None:
            if result > 99.8:
                result = ct.target_value
                _LOGGER.info(f"update_value_callback: a 100% reached")
            elif result >= ct.target_value:
                _LOGGER.info(f"update_value_callback: more than target {result} >= {ct.target_value}")


        await self._compute_and_launch_new_charge_state(time, command=self.current_command)

        return result

    async def _compute_and_launch_new_charge_state(self, time, command: LoadCommand, for_auto_command_init=False):
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
            self._expected_amperage.set(int(self.min_charge))
        elif command.command == CMD_OFF.command:
            self._expected_charge_state.set(False)
            self._expected_amperage.set(int(self.charger_min_charge))
        elif command.command == CMD_ON.command:
            self._expected_amperage.set(self.max_charge)
            self._expected_charge_state.set(True)
        elif command.command == CMD_AUTO_GREEN_ONLY.command:
            # only take decision if teh state is "good" for a while CHARGER_ADAPTATION_WINDOW
            if for_auto_command_init or (res_ensure_state and self._verified_correct_state_time is not None and (time - self._verified_correct_state_time).total_seconds() > CHARGER_ADAPTATION_WINDOW):


                current_power = 0.0

                current_real_max_charging_power = self._expected_amperage.value
                if self._expected_charge_state.value:
                    current_real_car_power = self.get_median_sensor(self.accurate_power_sensor, probe_duration,
                                                                    time)
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
                                                            can_be_saved = (time - self._verified_correct_state_time).total_seconds() > 3*CHARGER_ADAPTATION_WINDOW)


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
                    last_p = get_average_sensor(available_power[-len(available_power) // 2:])
                    all_p = get_average_sensor(available_power)

                    if command.param == CMD_AUTO_GREEN_ONLY.param:
                        target_delta_power = min(last_p, all_p)
                    else:
                        target_delta_power = max(last_p, all_p)


                    target_power = current_power + target_delta_power
                    safe_powers_steps = power_steps[self.min_charge:self.max_charge + 1]

                    if target_power <= 0:
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
                                if command.param == CMD_AUTO_GREEN_ONLY.param:
                                    new_amp = 0
                                else:
                                    # for ecomode
                                    new_amp = self.min_charge
                            else:
                                new_amp = self.min_charge

                        elif found_equal:
                            new_amp = best_i + self.min_charge
                        else:
                            if command.param == CMD_AUTO_GREEN_ONLY.param:
                                new_amp = best_i + self.min_charge
                            else:
                                new_amp = min(self.max_charge, best_i + self.min_charge + 1)




                    _LOGGER.info(f"Compute: target_delta_power {target_delta_power} current_power {current_power} ")
                    _LOGGER.info(f"target_power {target_power} new_amp {new_amp} current amp {current_real_max_charging_power}")
                    _LOGGER.info(f"min charge {self.min_charge} max charge {self.max_charge}")
                    _LOGGER.info(f"power steps {safe_powers_steps}")

                    if current_real_max_charging_power <= new_amp and current_power > 0 and target_delta_power < 0 and command.param == CMD_AUTO_GREEN_ONLY.param:
                        new_amp = min(current_real_max_charging_power -1, new_amp -1)
                        _LOGGER.info(f"Correct new_amp du to negative available power: {new_amp}")

                    if new_amp < self.min_charge:
                        new_amp = self.min_charge
                        self._expected_charge_state.set(False)
                    elif new_amp > self.max_charge:
                        new_amp = self.max_charge
                        self._expected_charge_state.set(True)
                    else:
                        self._expected_charge_state.set(True)

                    if new_amp is not None:
                        self._expected_amperage.set(int(new_amp))

                else:
                    _LOGGER.info(f"Available power invalid")
                    # new_amp = random.randint(int(self.min_charge) - 2, int(self.max_charge))



        if init_amp != self._expected_amperage.value or init_state != self._expected_charge_state.value:
            _LOGGER.info(
                f"Change inner states values: change state to {int(self._expected_amperage.value)}A - charge:{self._expected_charge_state.value}")

        # do it all the time
        await self._ensure_correct_state(time)


    def set_state_machine_to_current_state(self, time: datetime):
        if self.is_charge_disabled(time):
            self._expected_charge_state.set(False)
        else:
            self._expected_charge_state.set(True)
        self._expected_amperage.set(self.get_max_charging_power())

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
        result = False
        if self.is_plugged(time=time):
            if command.command == CMD_ON.command:
                if self.is_charge_enabled(time):
                    result = True
            elif command.command == CMD_OFF.command:
                if self.is_car_stopped_asking_current(time):
                    # we don't go off if it happens
                    return True
                elif self.is_charge_disabled(time):
                    result = True
            elif command.command == CMD_AUTO_GREEN_ONLY.command:
                result = True
        elif self.is_not_plugged(time=time):
            result = True

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
        self.attach_power_to_probe(self.charger_ocpp_current_import, transform_fn=self.convert_amps_to_W)
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
        self.attach_power_to_probe(self.charger_wallbox_charging_power)

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
