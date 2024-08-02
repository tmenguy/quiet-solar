import bisect
from datetime import datetime
from enum import StrEnum
import random
from typing import Any


from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry
from quiet_solar.const import CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, CONF_CHARGER_PAUSE_RESUME_SWITCH, \
    CONF_CHARGER_PLUGGED, CONF_CHARGER_MAX_CHARGE, CONF_CHARGER_MIN_CHARGE, CONF_CHARGER_IS_3P, \
    CONF_CHARGER_DEVICE_OCPP, CONF_CHARGER_DEVICE_WALLBOX, CONF_CHARGER_CONSUMPTION, DATETIME_MIN_UTC
from quiet_solar.home_model.constraints import MultiStepsPowerLoadConstraint

from quiet_solar.ha_model.car import QSCar
from quiet_solar.ha_model.device import HADeviceMixin, align_time_series_and_values, get_average_power, \
    get_median_power, convert_power_to_w
from quiet_solar.home_model.commands import LoadCommand, CMD_AUTO_GREEN_ONLY, CMD_ON, CMD_OFF, copy_command
from quiet_solar.home_model.load import AbstractLoad
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE, SERVICE_TURN_OFF, SERVICE_TURN_ON, \
    ATTR_ENTITY_ID, ATTR_UNIT_OF_MEASUREMENT, UnitOfPower

from homeassistant.components import number, homeassistant


CHARGER_STATE_REFRESH_INTERVAL = 3

CHARGER_ADAPTATION_WINDOW = 15


class QSOCPPChargePointStatus(StrEnum):
    """ OCPP Charger Status Description."""
    AVAILABLE = "Available"
    PREPARING = "Preparing"
    CHARGING = "Charging"
    SUSPENDED_EVSE = "SuspendedEVSE"
    SUSPENDED_EV = "SuspendedEV"
    FINISHING = "Finishing"
    RESERVED = "Reserved"


class QSWallboxChargerStatus(StrEnum):
    """Wallbox Charger Status Description."""

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

class QSChargerGeneric(HADeviceMixin, AbstractLoad):

    def __init__(self, **kwargs):
        self.charger_plugged = kwargs.pop(CONF_CHARGER_PLUGGED, None)
        self.charger_max_charging_current_number = kwargs.pop(CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, None)
        self.charger_pause_resume_switch = kwargs.pop(CONF_CHARGER_PAUSE_RESUME_SWITCH, None)
        self.charger_max_charge = kwargs.pop(CONF_CHARGER_MAX_CHARGE, 32)
        self.charger_min_charge = kwargs.pop(CONF_CHARGER_MIN_CHARGE, 6)
        self.charger_is_3p = kwargs.pop(CONF_CHARGER_IS_3P, False)
        self.charger_consumption = kwargs.pop(CONF_CHARGER_CONSUMPTION, 50)
        self.car : QSCar | None = None

        self.charge_state = STATE_UNKNOWN

        self._last_command_prob_time = DATETIME_MIN_UTC

        super().__init__(**kwargs)

        self._default_generic_car = QSCar(hass=self.hass, home=self.home, config_entry=self.config_entry, name=f"{self.name}_generic_car")

        self._inner_expected_charge_state = None
        self._inner_amperage = None
        self.reset()

    @property
    def _expected_charge_state(self):
        return self._inner_expected_charge_state

    @_expected_charge_state.setter
    def _expected_charge_state(self, value):
        if self._inner_expected_charge_state != value:
            self._inner_expected_charge_state = value
            self._asked_charge_state = None

    @property
    def _expected_amperage(self):
        return self._inner_amperage

    @_expected_amperage.setter
    def _expected_amperage(self, value):
        if self._inner_amperage != value:
            self._inner_amperage = value
            self._asked_amperage = None

    def reset(self):
        super().reset()
        self.detach_car()
        self._reset_state_machine()


    async def get_best_car(self , time: datetime) -> QSCar:
        # find the best car .... for now default one
        return self._default_generic_car

    async def check_load_activity_and_constraints(self, time: datetime):

        # check that we have a connected car, and which one, or that it is completely disconnected

        #  if there is no more car ... just reset
        is_plugged = self.is_plugged()
        if not is_plugged and self.car:
            self.reset()
        elif is_plugged and not self.car:
            self.reset()
            # find the best car .... for now
            c = await self.get_best_car(time)
            self.attach_car(c)

            # add a constraint ... for now just fill the car as much as possible
            power_steps, min_charge, max_charge = self.car.get_charge_power_per_phase_A(self.charger_is_3p)

            steps = []
            for v in power_steps:
                steps.append(copy_command(CMD_AUTO_GREEN_ONLY, power_consign=v))

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



    def _reset_state_machine(self):
        self._verified_amperage_command_time = None
        self._expected_amperage = None
        self._asked_amperage = None
        self._expected_charge_state = None
        self._asked_charge_state = None

    @property
    def min_charge(self):
        return max(self.charger_min_charge, self.car.car_charger_min_charge)

    @property
    def max_charge(self):
        return min(self.charger_max_charge, self.car.car_charger_max_charge)

    def get_platforms(self):
        return [ Platform.SENSOR, Platform.SELECT ]


    def attach_car(self, car):
        self.car = car

    def detach_car(self):
        self.car = None

    def is_plugged(self):
        state = self.hass.states.get(self.charger_plugged)
        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return False
        return state.state == "on"

    async def stop_charge(self):
        self._asked_charge_state = False
        if self.is_charge_enabled() and not self.is_charge_stopped():
            await self.hass.services.async_call(
                domain=Platform.SWITCH,
                service=SERVICE_TURN_OFF,
                target={ATTR_ENTITY_ID: self.charger_pause_resume_switch},
                blocking=False
            )

    async def start_charge(self):
        self._asked_charge_state = True
        if not self.is_charge_enabled() and  self.is_charge_stopped():
            await self.hass.services.async_call(
                domain=Platform.SWITCH,
                service=SERVICE_TURN_ON,
                target={ATTR_ENTITY_ID: self.charger_pause_resume_switch},
                blocking=False
            )

    def is_charge_enabled(self):
        result = False
        if self.is_plugged():
            state = self.hass.states.get(self.charger_pause_resume_switch)
            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                result = False
            else:
                result = state.state == "on"
        return result

    def is_charge_stopped(self):
        result = True
        if self.is_plugged():
            state = self.hass.states.get(self.charger_pause_resume_switch)
            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                result = True
            else:
                result = state.state == "off"
        return result

    def get_charging_power(self):
        if self.is_plugged() is False:
            return 0.0
        state = self.hass.states.get(self.power_sensor)
        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return 0.0
        val_power =  float(state.state)
        val_power = convert_power_to_w(val_power, state.attributes)
        return val_power

    def is_car_stopped_asking_current(self, time:datetime):
        return self.is_charging_power_zero(CHARGER_ADAPTATION_WINDOW, time)

    def is_charging_power_zero(self, num_seconds, time) -> bool:
        val =  self.get_median_charging_power(num_seconds, time)
        if val is None:
            return False

        return val < self.charger_consumption # 50 W of consumption for the charger for ex

    def _internal_get_median_charging_power(self, num_seconds, time) -> float | None:
        if self.power_sensor:
            charge_power_values = self.get_state_history_data(self.power_sensor, num_seconds, time)
            if not charge_power_values:
                return None
            all_p = get_median_power(charge_power_values)
            if all_p < self.charger_consumption: #50 W of consumption for the charger for ex
                all_p = 0.0
            return all_p
        return None

    def get_median_charging_power(self, num_seconds, time) -> float | None:
        return self._internal_get_median_charging_power(num_seconds, time)

    async def set_battery_discharge_max_power_if_needed(self, power: float|None = None):
        if self.current_command.command == CMD_AUTO_GREEN_ONLY.command and self.home.is_battery_in_auto_mode():
                await self.home.set_max_discharging_power(power=power)

    async def set_max_charging_current(self, current, blocking=False):

        self._asked_amperage = current

        data: dict[str, Any] = {ATTR_ENTITY_ID: self.charger_max_charging_current_number}
        range_value = float(current)
        service = number.SERVICE_SET_VALUE
        min_value = float(self.min_charge)
        max_value = float(self.max_charge)
        data[number.ATTR_VALUE] = int(min(max_value, max(min_value, range_value)))
        domain = number.DOMAIN

        await self.hass.services.async_call(
            domain, service, data, blocking=blocking
        )

        #await self.hass.services.async_call(
        #    domain=domain, service=service, service_data={number.ATTR_VALUE:int(min(max_value, max(min_value, range_value)))}, target={ATTR_ENTITY_ID: self.charger_max_charging_current_number}, blocking=blocking
        #)

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


    async def _ensure_correct_state(self, time, current_real_max_charging_power):

        if current_real_max_charging_power != self._expected_amperage:
            # check first if amperage setting is ok
            if self._asked_amperage != self._expected_amperage:
                await self.set_max_charging_current(self._expected_amperage, blocking=False)
                self._asked_amperage = self._expected_amperage
            else:
                await self._do_update_charger_state(time)
            self._verified_amperage_command_time = None
        elif self.is_charge_enabled() != self._expected_charge_state:
            # if amperage is ok check if charge state is ok
            if self._asked_charge_state != self._expected_charge_state:
                if self._expected_charge_state:
                    await self.start_charge()
                else:
                    await self.stop_charge()
                self._asked_charge_state = self._expected_charge_state
            else:
                await self._do_update_charger_state(time)
            self._verified_amperage_command_time = None
        else:
            self._asked_amperage = None
            self._asked_charge_state = None
            return True
        return False

    async def constraint_update_value_callback_percent_soc(self, ct: AbstractLoad, time: datetime) -> float | None:
        """ Example of a value compute callback for a load constraint. like get a sensor state, compute energy available for a car from battery charge etc
        it could also update the current command of the load if needed
        """

        if self.car.car_charge_percent_sensor is None:
            result = 0.0
        else:
            state = self.hass.states.get(self.car.car_charge_percent_sensor)

            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                result = None
            else:
                result = float(state.state)


        # check also if the current state of the automation has been properly set
        if self.current_command is None or self.is_plugged() is False:
            self.current_command = None
            self._verified_amperage_command_time = None
        elif self.current_command.command == CMD_OFF.command:
            self._verified_amperage_command_time = None
        else:
            # init the expected command if needed
            self._init_expected_state()
            current_real_max_charging_power = self.get_max_charging_power()

            if await self._ensure_correct_state(time, current_real_max_charging_power):
                # we are in a "good" state

                if self._verified_amperage_command_time is None:
                    #ok we enter the state knowing where we are
                    self._verified_amperage_command_time = time

                    # we are all set after a change of state, handle battery
                    if self._expected_charge_state:
                        # no battery discharge in case of car is charging in auto mode
                        await  self.set_battery_discharge_max_power_if_needed(power=0.0)
                    else:
                        await  self.set_battery_discharge_max_power_if_needed()

                # we will compare now if the current need to be adapted compared to solar production
                if (time - self._verified_amperage_command_time).total_seconds() > CHARGER_ADAPTATION_WINDOW:

                    if self._expected_charge_state and self.is_charging_power_zero(CHARGER_ADAPTATION_WINDOW, time) and self.is_car_stopped_asking_current(time):
                        # we can put back the battery as possibly discharging! as the car won't consume anymore soon ...
                        await self.set_battery_discharge_max_power_if_needed()

                    elif self.current_command.command == CMD_AUTO_GREEN_ONLY.command:

                        power_steps, min_charge, max_charge = self.car.get_charge_power_per_phase_A(self.charger_is_3p)

                        current_power = 0.0

                        if self._expected_charge_state:
                            current_real_car_power = self._internal_get_median_charging_power(CHARGER_ADAPTATION_WINDOW, time)

                            # time to update some dampening car values:
                            if current_real_car_power is not None and (time - self._verified_amperage_command_time).total_seconds() > 2*CHARGER_ADAPTATION_WINDOW:
                                self.car.update_dampening_value(amperage=current_real_max_charging_power,
                                                                power_value=current_real_car_power,
                                                                for_3p=self.charger_is_3p,
                                                                time=time)


                            # we will compare now if the current need to be adapted compared to solar production
                            if current_real_max_charging_power >= min_charge:
                                current_power = current_real_car_power
                                if current_power is None:
                                    current_power = power_steps[int(current_real_max_charging_power - min_charge)]
                            else:
                                current_power = 0.0

                        grid_active_power_values = self.home.get_grid_active_power_values(CHARGER_ADAPTATION_WINDOW, time)
                        charging_values = []

                        if self.home.is_battery_in_auto_mode():
                            charging_values = self.home.get_battery_charge_values(CHARGER_ADAPTATION_WINDOW, time)


                        available_power = align_time_series_and_values(grid_active_power_values, charging_values, operation=lambda v1,v2: v1+v2)
                        # the battery is normally adapting itself to the solar production, so if it is charging ... we will say that this powe is available to the car

                        # do we need a bit of a PID ? (proportional integral derivative? or keep it simple for now) or a convex hul with min / max?
                        # very rough estimation for now:

                        if available_power:
                            last_p = get_average_power(available_power[-len(available_power)//2:])
                            all_p = get_average_power(available_power)

                            if self.current_command.param == CMD_AUTO_GREEN_ONLY.param:
                                target_delta_power = min(last_p, all_p)
                            else:
                                target_delta_power = max(last_p, all_p)

                            target_power = current_power + target_delta_power

                            i = bisect.bisect_left(power_steps, target_power)

                            if i == len(power_steps):
                                new_amp = self.max_charge
                            elif i == 0 and power_steps[0] > target_power:
                                if self.current_command.param == CMD_AUTO_GREEN_ONLY.param:
                                    new_amp = 0
                                else:
                                    new_amp = self.min_charge
                            elif power_steps[i] == target_power:
                                new_amp = i + min_charge
                            else:
                                if self.current_command.param == CMD_AUTO_GREEN_ONLY.param:
                                    new_amp = i + min_charge
                                else:
                                    new_amp = min(max_charge, i + min_charge + 1)
                        else:
                            new_amp = random.randint(min_charge, max_charge)

                        if new_amp < self.min_charge:
                            self._expected_amperage = self.min_charge
                            self._expected_charge_state = False
                        elif new_amp > self.max_charge:
                            self._expected_amperage = self.max_charge
                            self._expected_charge_state = True
                        else:
                            self._expected_amperage = new_amp
                            self._expected_charge_state = True

                        await self._ensure_correct_state(time, current_real_max_charging_power)

                        self.current_command.private = self._expected_amperage

        return result

    def _init_expected_state(self):
        self._set_command_params(self.current_command)
        if self._expected_amperage is None:
            self._asked_amperage = None
            if self.current_command.command == CMD_ON.command:
                self._expected_amperage = self.current_command.private
            else:
                self._expected_amperage = self.min_charge
        if self._expected_charge_state is None:
            self._asked_charge_state = None
            if self.current_command.command == CMD_ON.command:
                self._expected_charge_state = True
            else:
                self._expected_charge_state = False

    def _set_command_params(self, command: LoadCommand):
        if command.command == CMD_ON.command:
            if command.power_consign is None or command.power_consign == 0.0:
                command.private = self.charger_max_charge
            else:
                mult = 1.0
                if self.charger_is_3p:
                    mult = 3.0
                command.private = int(min(self.max_charge, max(self.min_charge, float(command.power_consign) / (mult*self.home.voltage))))

    async def execute_command(self, time: datetime, command : LoadCommand):

        # force a homeassistant.update_entity service on the charger entity?
        self._set_command_params(command)
        self._reset_state_machine()

        if command.command == CMD_ON.command:
            self._expected_charge_state = True
            await self.start_charge()
        elif command.command == "off" or command.command == "auto":
            self._expected_charge_state = False
            await self.stop_charge()

        self._last_command_prob_time = time

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool:

        result = False
        if self.is_plugged():
            if command.command == CMD_ON.command:
                result = self.is_charge_enabled()
            elif command.command == CMD_OFF.command:
                result = self.is_charge_stopped()
            elif command.command == CMD_AUTO_GREEN_ONLY.command:
                result =  True
        elif command.command == CMD_OFF.command:
            result = True

        if result is False:
            await self._do_update_charger_state(time)

        return result

    async def _do_update_charger_state(self, time):
        if (time - self._last_command_prob_time).total_seconds() > CHARGER_STATE_REFRESH_INTERVAL:
            await self.hass.services.async_call(
                homeassistant.DOMAIN,
                homeassistant.SERVICE_UPDATE_ENTITY,
                {ATTR_ENTITY_ID: [self.charger_pause_resume_switch, self.charger_max_charging_current_number]},
                blocking=False
            )
            self._last_command_prob_time = time


class QSChargerOCPP(QSChargerGeneric):

    def __init__(self, **kwargs):
        self.charger_device_ocpp = kwargs.pop(CONF_CHARGER_DEVICE_OCPP, None)
        self.charger_ocpp_status_connector = None
        self.charger_ocpp_current_import  = None
        self.charger_ocpp_stop_reason = None
        self.charger_ocpp_power_active_import = None


        hass : HomeAssistant | None = kwargs.get("hass", None)

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

                if entry.entity_id.startswith("sensor.") and entry.entity_id.endswith("_power_active_import"):
                    self.charger_ocpp_power_active_import = entry.entity_id

                # OCPP only sensors :
                if entry.entity_id.startswith("sensor.") and entry.entity_id.endswith("_status_connector"):
                    self.charger_ocpp_status_connector = entry.entity_id

                if entry.entity_id.startswith("sensor.") and entry.entity_id.endswith("_status_connector"):
                    self.charger_ocpp_status_connector = entry.entity_id

                if entry.entity_id.startswith("sensor.") and entry.entity_id.endswith("_stop_reason"):
                    self.charger_ocpp_stop_reason = entry.entity_id

                if entry.entity_id.startswith("sensor.") and entry.entity_id.endswith("_current_import"):
                    self.charger_ocpp_current_import = entry.entity_id



        super().__init__(**kwargs)

        self.attach_ha_state_to_probe(self.charger_ocpp_current_import, is_numerical=True,
                                      update_on_change_only=True)
        self.attach_ha_state_to_probe(self.charger_ocpp_power_active_import, is_numerical=True,
                                      update_on_change_only=True)
        self.attach_ha_state_to_probe(self.charger_ocpp_stop_reason, is_numerical=False,
                                      update_on_change_only=False)
        self.attach_ha_state_to_probe(self.charger_ocpp_status_connector, is_numerical=False,
                                      update_on_change_only=False)


    def is_plugged(self):
        state = self.hass.states.get(self.charger_plugged)

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return False

        return state.state == "off"

    def get_median_charging_power(self, num_seconds, time) -> float | None:
        val = super().get_median_charging_power(num_seconds, time)
        if val is not None:
            return val

        val = None

        if self.charger_ocpp_current_import:
            charge_power_values = self.get_state_history_data(self.charger_ocpp_current_import, num_seconds, time)
            if not charge_power_values:
                return None
            all_p = get_median_power(charge_power_values)
            # mult = 1.0
            # if self.charger_is_3p:
            #    mult = 3.0
            val =  all_p * self.home.voltage

        return val

    def is_car_stopped_asking_current(self, time:datetime):


        contiguous_last_disconnect = 0
        if self.charger_ocpp_stop_reason is not None:

            contiguous_last_disconnect = self.get_last_state_value_duration(self.charger_ocpp_stop_reason,
                                               states_vals=["EVDisconnected"],
                                               num_seconds_before=CHARGER_ADAPTATION_WINDOW, time=time)






        if self.charger_ocpp_status_connector is not None:

            contiguous_last_disconnect = max(contiguous_last_disconnect, self.get_last_state_value_duration(self.charger_ocpp_status_connector,
                                                                            states_vals=["SuspendedEVSE", "SuspendedEV"],
                                                                            num_seconds_before=CHARGER_ADAPTATION_WINDOW, time=time))




        return contiguous_last_disconnect >= CHARGER_ADAPTATION_WINDOW//2


    def get_charging_power(self):

        if self.power_sensor:
            return super().get_charging_power()

        if self.is_plugged() is False:
            return 0.0

        state = None
        if self.charger_ocpp_power_active_import is not None:
            state = self.hass.states.get(self.charger_ocpp_power_active_import)

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            val_power = 0.0
        else:
            val_power = float(state.state)
            val_power = convert_power_to_w(val_power, state.attributes)

        if val_power == 0.0 and self.charger_ocpp_current_import is not None:
            state = self.hass.states.get(self.charger_ocpp_current_import)
            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                val_power = 0.0
            else:
                #mult = 1.0
                #if self.charger_is_3p:
                #    mult = 3.0
                val_power = float(state.state) * self.home.voltage # ok in W

        return val_power



class QSChargerWallbox(QSChargerGeneric):
    def __init__(self, **kwargs):
        self.charger_device_wallbox = kwargs.pop(CONF_CHARGER_DEVICE_WALLBOX, None)
        self.charger_wallbox_status_description = None
        self.charger_wallbox_charging_power = None
        hass : HomeAssistant | None = kwargs.get("hass", None)


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
                    self.charger_wallbox_status_description = entry.entity_id


        super().__init__(**kwargs)

        self.attach_ha_state_to_probe(self.charger_wallbox_charging_power, is_numerical=True,
                                      update_on_change_only=True)

    def is_plugged(self):
        state = self.hass.states.get(self.charger_pause_resume_switch)

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return False

        return True

    def get_charging_power(self):

        if self.power_sensor:
            return super().get_charging_power()

        if self.is_plugged() is False:
            return 0.0

        state = self.hass.states.get(self.charger_wallbox_charging_power)

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            val_power = 0.0
        else:
            val_power = float(state.state)
            val_power = convert_power_to_w(val_power, state.attributes)


        return val_power


    def get_median_charging_power(self, num_seconds, time) -> float | None:
        val = super().get_median_charging_power(num_seconds, time)
        if val is not None:
            return val

        val = None

        if self.charger_wallbox_charging_power:
            charge_power_values = self.get_state_history_data(self.charger_wallbox_charging_power, num_seconds, time)
            if not charge_power_values:
                return None
            all_p = get_median_power(charge_power_values)

            # mult = 1.0
            # if self.charger_is_3p:
            #    mult = 3.0
            val =  all_p

        return val

    def is_car_stopped_asking_current(self, time:datetime):

        contiguous_last_disconnect = 0
        if self.charger_wallbox_status_description is not None:

            contiguous_last_disconnect = self.get_last_state_value_duration(self.charger_wallbox_status_description,
                                               states_vals=["Waiting for car demand"],
                                               num_seconds_before=CHARGER_ADAPTATION_WINDOW, time=time)


        return contiguous_last_disconnect >= CHARGER_ADAPTATION_WINDOW//2






