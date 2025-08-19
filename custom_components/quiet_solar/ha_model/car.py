import bisect
import logging
from datetime import datetime, timedelta
from typing import Any

import pytz
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE, ATTR_ENTITY_ID
from homeassistant.components import number

from ..const import CONF_CAR_PLUGGED, CONF_CAR_TRACKER, CONF_CAR_CHARGE_PERCENT_SENSOR, \
    CONF_CAR_CHARGE_PERCENT_MAX_NUMBER, \
    CONF_CAR_BATTERY_CAPACITY, CONF_CAR_CHARGER_MIN_CHARGE, CONF_CAR_CHARGER_MAX_CHARGE, \
    CONF_CAR_CUSTOM_POWER_CHARGE_VALUES, CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P, MAX_POSSIBLE_AMPERAGE, \
    CONF_DEFAULT_CAR_CHARGE, CONF_CAR_IS_INVITED, FORCE_CAR_NO_CHARGER_CONNECTED, \
    CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS, CONF_MINIMUM_OK_CAR_CHARGE
from ..ha_model.device import HADeviceMixin
from ..home_model.constraints import MultiStepsPowerLoadConstraintChargePercent
from ..home_model.load import AbstractDevice
from datetime import time as dt_time


_LOGGER = logging.getLogger(__name__)

MIN_CHARGE_POWER_W = 70


class QSCar(HADeviceMixin, AbstractDevice):

    def __init__(self, **kwargs):
        self.car_plugged = kwargs.pop(CONF_CAR_PLUGGED, None)
        self.car_tracker = kwargs.pop(CONF_CAR_TRACKER, None)
        self.car_charge_percent_sensor = kwargs.pop(CONF_CAR_CHARGE_PERCENT_SENSOR, None)
        self.car_charge_percent_max_number = kwargs.pop(CONF_CAR_CHARGE_PERCENT_MAX_NUMBER, None)
        self._conf_car_charge_percent_max_number_steps = kwargs.pop(CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS, None)
        if self._conf_car_charge_percent_max_number_steps == "":
            self._conf_car_charge_percent_max_number_steps = None
        self.car_battery_capacity = kwargs.pop( CONF_CAR_BATTERY_CAPACITY, None)
        self.car_default_charge = kwargs.pop(CONF_DEFAULT_CAR_CHARGE, 100.0)
        self.car_minimum_ok_charge = kwargs.pop(CONF_MINIMUM_OK_CAR_CHARGE, 50.0)


        self.car_is_invited = kwargs.pop(CONF_CAR_IS_INVITED, False)

        self.car_charger_min_charge : int = int(max(0,kwargs.pop(CONF_CAR_CHARGER_MIN_CHARGE, 6)))
        self._conf_car_charger_min_charge = self.car_charger_min_charge
        self.car_charger_max_charge : int = min(MAX_POSSIBLE_AMPERAGE, int(max(0, kwargs.pop(CONF_CAR_CHARGER_MAX_CHARGE, 32))))
        self._conf_car_charger_max_charge = self.car_charger_max_charge
        self.car_use_custom_power_charge_values = kwargs.pop(CONF_CAR_CUSTOM_POWER_CHARGE_VALUES, False)
        if self.car_use_custom_power_charge_values is False:
            self.car_is_custom_power_charge_values_3p = None
        else:
            self.car_is_custom_power_charge_values_3p = kwargs.pop(CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P, False)

        self.amp_to_power_1p = [-1] * (MAX_POSSIBLE_AMPERAGE)
        self.amp_to_power_3p = [-1] * (MAX_POSSIBLE_AMPERAGE)
        self._last_dampening_update = None


        self.theoretical_amp_to_power_1p = [-1] * (MAX_POSSIBLE_AMPERAGE)
        self.theoretical_amp_to_power_3p = [-1] * (MAX_POSSIBLE_AMPERAGE)

        self.customized_amp_to_power_1p = [-1] * (MAX_POSSIBLE_AMPERAGE)
        self.customized_amp_to_power_3p = [-1] * (MAX_POSSIBLE_AMPERAGE)

        self.conf_customized_amp_to_power_1p = [-1] * (MAX_POSSIBLE_AMPERAGE)
        self.conf_customized_amp_to_power_3p = [-1] * (MAX_POSSIBLE_AMPERAGE)

        self.car_charge_percent_max_number_steps = []
        if self._conf_car_charge_percent_max_number_steps and isinstance(self._conf_car_charge_percent_max_number_steps, str):
            vals_str = self._conf_car_charge_percent_max_number_steps.split(",")
            self.car_charge_percent_max_number_steps = []
            for val in vals_str:
                try:
                    v = int(val.strip())
                    if v >= 0 and v <= 100:
                        self.car_charge_percent_max_number_steps.append(v)
                except ValueError:
                    _LOGGER.error(f"Invalid value {val} for car charge percent max number steps, must be an integer")
                    self.car_charge_percent_max_number_steps = []
                    break

            if len(self.car_charge_percent_max_number_steps) > 0:
                self.car_charge_percent_max_number_steps.sort()
                if self.car_charge_percent_max_number_steps[-1] != 100:
                  self.car_charge_percent_max_number_steps.append(100)

        super().__init__(**kwargs)

        self._conf_calendar = self.calendar

        for a in range(len(self.theoretical_amp_to_power_1p)):

            val_1p = float(self.voltage * a)
            val_3p = 3*val_1p

            self.amp_to_power_1p[a] = self.theoretical_amp_to_power_1p[a] = val_1p
            self.amp_to_power_3p[a] = self.theoretical_amp_to_power_3p[a] = val_3p

        self.can_dampen_strongly_dynamically = True
        if self.car_use_custom_power_charge_values:

            self.can_dampen_strongly_dynamically = False

            for a in range(self.car_charger_min_charge, self.car_charger_max_charge + 1):
                val = float(kwargs.pop(f"charge_{a}", -1))
                if val >= 0:
                    if self.car_is_custom_power_charge_values_3p:
                        self.conf_customized_amp_to_power_3p[a] = self.customized_amp_to_power_3p[a] = self.amp_to_power_3p[a] = val
                        if a*3 >= self.car_charger_min_charge and a*3 <= self.car_charger_max_charge:
                            self.conf_customized_amp_to_power_1p[a*3] = self.customized_amp_to_power_1p[a*3] = self.amp_to_power_1p[a*3] = val

                    else:
                        self.conf_customized_amp_to_power_1p[a] = self.customized_amp_to_power_1p[a] = self.amp_to_power_1p[a] = val
                        if a % 3 == 0 and a // 3 >= self.car_charger_min_charge and a // 3 <= self.car_charger_max_charge:
                            self.conf_customized_amp_to_power_3p[a//3] = self.customized_amp_to_power_3p[a//3] = self.amp_to_power_3p[a//3] = val


        self.attach_ha_state_to_probe(self.car_charge_percent_sensor,
                                      is_numerical=True)

        self.attach_ha_state_to_probe(self.car_plugged,
                                      is_numerical=False)

        self.attach_ha_state_to_probe(self.car_tracker,
                                      is_numerical=False)

        self._salvable_dampening = {}

        self._dampening_deltas = {}
        self._dampening_deltas_graph = {}

        self.charger = None
        self.do_force_next_charge = False

        self._next_charge_target = None
        self.user_attached_charger_name : str | None = None

        self.default_charge_time: dt_time | None = None

        self._qs_bump_solar_priority = False

        self.reset()

    def get_platforms(self):
        parent = super().get_platforms()
        if parent is None:
            parent = set()
        else:
            parent = set(parent)
        parent.update([Platform.SENSOR, Platform.SELECT, Platform.SWITCH, Platform.BUTTON, Platform.TIME])
        return list(parent)

    def reset(self):
        super().reset()
        self.interpolate_power_steps(do_recompute_min_charge=True, use_conf_values=True)
        self._dampening_deltas = {}
        self._dampening_deltas_graph = {}
        self.calendar = self._conf_calendar
        self.charger = None
        self.do_force_next_charge = False
        self.user_attached_charger_name = None


    def attach_charger(self, charger):
        _LOGGER.info(f"Car {self.name} attached charger {self.charger.name}")
        charger.attach_car(self)

    def detach_charger(self):
        if self.charger is not None:
            _LOGGER.info(f"Car {self.name} detached charger {self.charger.name}")
            self.charger.detach_car()


    def get_continuous_plug_duration(self, time:datetime) -> float | None:

        if self.car_plugged is None:
            return None

        return self.get_last_state_value_duration(self.car_plugged,
                                                  states_vals=["on"],
                                                  num_seconds_before=None,
                                                  time=time)[0]

    def is_car_plugged(self, time:datetime, for_duration:float|None = None) -> bool | None:

        if self.car_plugged is None:
            return None

        if for_duration is not None:

            contiguous_status = self.get_last_state_value_duration(self.car_plugged,
                                                                   states_vals=["on"],
                                                                   num_seconds_before=8*for_duration,
                                                                   time=time)[0]
            if contiguous_status is not None:
                return contiguous_status >= for_duration and contiguous_status > 0
            else:
                return None

        else:
            latest_state = self.get_sensor_latest_possible_valid_value(entity_id=self.car_plugged, time=time)

            if latest_state is None:
                return None

            return latest_state == "on"

    def get_car_coordinates(self, time:datetime) -> tuple[float, float] | tuple[None,None]:

            if self.car_tracker is None:
                return None, None

            state, state_attr = self.get_sensor_latest_possible_valid_value_and_attr(entity_id=self.car_tracker, time=time)
            if state is None:
                return None, None

            if state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                return None, None

            if state_attr is None:
                return None, None

            latitude: str | None  = state_attr.get("latitude", None)
            longitude: str | None  = state_attr.get("longitude", None)

            if latitude is None or longitude is None:
                return None, None

            try:
                return float(latitude), float(longitude)
            except ValueError:
                return None, None

    def is_car_home(self, time:datetime, for_duration:float|None = None) -> bool | None:

        if self.car_tracker is None:
            return None

        if for_duration is not None:

            contiguous_status = self.get_last_state_value_duration(self.car_tracker,
                                                                   states_vals=["home"],
                                                                   num_seconds_before=8*for_duration,
                                                                   time=time)[0]
            if contiguous_status is not None:
                return contiguous_status >= for_duration and contiguous_status > 0
            else:
                return None
        else:
            latest_state = self.get_sensor_latest_possible_valid_value(entity_id=self.car_tracker, time=time)

            if latest_state is None:
                return None

            return latest_state == "home"

    def get_car_charge_percent(self, time: datetime | None = None, tolerance_seconds: float=4*3600 ) -> float | None:
        return self.get_sensor_latest_possible_valid_value(entity_id=self.car_charge_percent_sensor, time=time, tolerance_seconds=tolerance_seconds)

    def is_car_charge_growing(self,
                              num_seconds: float,
                              time: datetime) -> bool | None:

        return self.is_sensor_growing(entity_id=self.car_charge_percent_sensor,
                               num_seconds=num_seconds,
                              time=time)

    def get_car_current_capacity(self, time: datetime) -> float | None:
        res = self.get_car_charge_percent(time)
        if res is None:
            return None

        if self.car_battery_capacity is None or self.car_battery_capacity == 0:
            return None

        try:
            return float(res) * self.car_battery_capacity / 100.0
        except TypeError:
            return None

    async def adapt_max_charge_limit(self, asked_percent):

        if self.car_charge_percent_max_number is None:
            return

        percent= asked_percent
        # in fact stop the charge only at the "default charge" of the carelse ... continue to charge
        if asked_percent <= self.car_default_charge:
            percent = self.car_default_charge

        if self.car_charge_percent_max_number_steps and len(self.car_charge_percent_max_number_steps) >= 1:
            p_idx = bisect.bisect_left(self.car_charge_percent_max_number_steps, percent)
            if p_idx >= len(self.car_charge_percent_max_number_steps):
                percent = self.car_charge_percent_max_number_steps[-1]
            else:
                # get the one that is the closest to the percent but bigger or equal
                percent = self.car_charge_percent_max_number_steps[p_idx]

        current_charge_limit = self.get_max_charge_limit()
        if current_charge_limit != percent:
            _LOGGER.info(f"Car {self.name} set max charge limit from {current_charge_limit}% to {percent}%")

            data: dict[str, Any] = {ATTR_ENTITY_ID: self.car_charge_percent_max_number}
            service = number.SERVICE_SET_VALUE
            data[number.ATTR_VALUE] = int(percent)
            domain = number.DOMAIN

            await self.hass.services.async_call(
                domain, service, data
            )

    def car_can_limit_its_soc(self):
        if self.car_charge_percent_max_number is None:
            return False
        return True

    def get_max_charge_limit(self):

        result = None
        if self.car_charge_percent_max_number is not None:

            state = self.hass.states.get(self.car_charge_percent_max_number)

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



    def find_path(self, graph, start, end, path=None):
        if path is None:
            path = []

        path = path + [start]

        if start == end:
            return path

        if start not in graph:
            return None

        for node in graph[start]:
            if node not in path:
                new_path = self.find_path(graph, node, end, path)
                if new_path:
                    return new_path

        return None

    def _get_power_from_stored_amps(self, from_amp: int | float, from_num_phase: int ) -> None|float:
        if from_amp < self.car_charger_min_charge:
            from_power = 0.0
        else:
            from_amp = max(min(self.car_charger_max_charge, from_amp), self.car_charger_min_charge)
            if from_num_phase == 1:
                from_power = self.amp_to_power_1p[from_amp]
            else:
                from_power = self.amp_to_power_3p[from_amp]
        return from_power


    def get_delta_dampened_power(self, from_amp: int | float, from_num_phase: int, to_amp: int | float, to_num_phase: int) -> float | None:

        if from_amp*from_num_phase == to_amp*to_num_phase:
            return 0.0

        power = None

        from_power = self._get_power_from_stored_amps(from_amp, from_num_phase)
        to_power = self._get_power_from_stored_amps(to_amp, to_num_phase)

        if len(self._dampening_deltas) > 0:

            from_amp = from_amp*from_num_phase
            to_amp = to_amp*to_num_phase

            power = self._dampening_deltas.get((from_amp, to_amp))

            if power is None and len(self._dampening_deltas) > 2:
                # not direct try a path:
                path = self.find_path(self._dampening_deltas_graph, from_amp, to_amp)

                if path and len(path) > 1 and path[0] == from_amp and path[-1] == to_amp:
                    power = 0
                    for i in range(1, len(path)):
                        p = self._dampening_deltas.get((path[i-1], path[i]))
                        if p is None:
                            _LOGGER.error(f"get_delta_dampened_power path in error: Car {self.name} deltas {self._dampening_deltas} graph {self._dampening_deltas_graph} from_amp {from_amp} to_amp {to_amp} path[i-1] {path[i-1]} path[i] {path[i]}")
                            power = None
                            break

                        power += p
                elif path:
                    _LOGGER.error(
                        f"get_delta_dampened_power path error: Car {self.name} deltas {self._dampening_deltas} graph {self._dampening_deltas_graph} from_amp {from_amp} to_amp {to_amp}")

        if power is None:
            if from_power is not None and to_power is not None:
                power = to_power - from_power

        return power



    def _theoretical_max_power(self, amperage:tuple[float,int] | tuple[int,int], delta_amp:float) -> float:
        if amperage[0] == 0:
            return 0.0
        theoretical_power = float(self.voltage * max(0.0, amperage[0] + delta_amp))
        if amperage[1] == 3:
            theoretical_power = theoretical_power * 3
        return theoretical_power



    def _add_to_amps_power_graph(self, from_a: tuple[float,int], to_a:tuple[float,int], power_delta: int | float) -> bool:
        from_amp = int(from_a[0] * from_a[1])
        to_amp = int(to_a[0] * to_a[1])

        if from_amp == to_amp:
            return False


        if from_amp > to_amp:

            if power_delta > 0.0:
                # we do not allow to add a delta that is positive from a higher amperage to a lower amperage
                _LOGGER.warning(f"_add_to_amps_power_graph: {self.name}  from_amp {from_a} > to_amp {to_a} with power_delta {power_delta} - ignoring this value")
                return False

            from_amp, to_amp = to_amp, from_amp
            from_a, to_a = to_a, from_a
            power_delta = -power_delta
        else:
            if power_delta < 0.0:
                # we do not allow to add a delta that is positive from a higher amperage to a lower amperage
                _LOGGER.warning(f"_add_to_amps_power_graph: {self.name}  from_amp {from_a} < to_amp {to_a} with power_delta {power_delta} - ignoring this value")
                return False

        self._dampening_deltas[(from_amp, to_amp)] = power_delta
        self._dampening_deltas[(to_amp, from_amp)] = -power_delta

        fs = self._dampening_deltas_graph.setdefault(from_amp, set())
        fs.add(to_amp)
        ts = self._dampening_deltas_graph.setdefault(to_amp, set())
        ts.add(from_amp)

        return True

    def _can_accept_new_dampen_values(self, old_val:float, new_val:float) -> bool:

        if old_val * new_val < 0:
            # should be same sign
            return False

        old_val = abs(old_val)
        new_val = abs(new_val)

        if old_val < MIN_CHARGE_POWER_W and new_val < MIN_CHARGE_POWER_W:
            return False



        if new_val < MIN_CHARGE_POWER_W:
            # it means we are setting to 0 for the new transition
            if self.can_dampen_strongly_dynamically is False:
                return False
            else:
                return True

        if old_val < MIN_CHARGE_POWER_W:
            # wow it was a 0 something and now it has a value ....
            if self.can_dampen_strongly_dynamically is False:
                return False
            else:
                return True


        ratio = new_val / old_val

        if ratio > 1.10:
            #if growing too much ... we do nothing vs what was there before
            return False

        lower_ratio = 0.2
        if self.can_dampen_strongly_dynamically is False:
            lower_ratio = 0.7

        if ratio < lower_ratio:
            #if going down too much ... we do nothing vs what was there before
            return False

        return True


    def update_dampening_value(self, amperage: None | tuple[float,int] | tuple[int,int], amperage_transition: None | tuple[tuple[int,int] | tuple[float,int], tuple[int,int] | tuple[float,int]], power_value_or_delta: int | float, time:datetime, can_be_saved:bool = False) -> bool:

        do_update = False

       # if self.can_dampen_strongly_dynamically is False:
       #     _LOGGER.info(f"Car {self.name} cannot dampen dynamically, ignoring amperage {amperage} and amperage_transition {amperage_transition}")
       #     return False

        if amperage_transition is None and amperage is None:
            return False

        if amperage_transition is not None:

            if amperage is None:
                if amperage_transition[0][0] == 0:
                    amperage = amperage_transition[1]
                elif amperage_transition[1][0] == 0:
                    amperage = amperage_transition[0]
                    power_value_or_delta = -power_value_or_delta

            if amperage is None:

                orig_delta = self.get_delta_dampened_power(amperage_transition[0][0], amperage_transition[0][1], amperage_transition[1][0], amperage_transition[1][1])

                if orig_delta is not None:

                    if self._can_accept_new_dampen_values(orig_delta, power_value_or_delta) is False:
                        _LOGGER.info(f"Car {self.name} cannot accept new dampening value for amperage_transition {amperage_transition} with power_value_or_delta {power_value_or_delta} orig_delta {orig_delta} - ignoring this value")
                        return False

                    if self._add_to_amps_power_graph(amperage_transition[0], amperage_transition[1], power_value_or_delta) is False:
                        return False

                    do_update = True


        if amperage is not None:

            if amperage[0] < self.car_charger_min_charge or amperage[0] > self.car_charger_max_charge:
                return False

            if amperage[1] == 3:
                for_3p = True
            else:
                for_3p = False

            amps_val = int(amperage[0])

            old_val =  self._get_power_from_stored_amps(amperage[0], amperage[1])


            if power_value_or_delta < -MIN_CHARGE_POWER_W:
                return False

            if abs(power_value_or_delta) < MIN_CHARGE_POWER_W:
                power_value_or_delta = abs(power_value_or_delta)

            if self._can_accept_new_dampen_values(old_val, power_value_or_delta) is False:
                _LOGGER.info(
                    f"Car {self.name} cannot accept new dampening value for amperage {amperage} with power_value_or_delta {power_value_or_delta} orig_delta {old_val} - ignoring this value")
                return False


            if power_value_or_delta >= MIN_CHARGE_POWER_W and self._add_to_amps_power_graph((0.0, amperage[1]),
                                                                                            (amperage[0],
                                                                                             amperage[1]),
                                                                                            power_value_or_delta) is False:
                return False

            can_be_saved = False

            do_recompute_min_charge = can_be_saved

            car_percent = self.get_car_charge_percent(time)

            if power_value_or_delta >= MIN_CHARGE_POWER_W:
                if for_3p:
                    self.customized_amp_to_power_3p[amps_val] = float(power_value_or_delta)
                    if 3*amps_val <= self.car_charger_max_charge and 3*amps_val >= self.car_charger_min_charge:
                        self.customized_amp_to_power_1p[3*amps_val] = float(power_value_or_delta)
                else:
                    self.customized_amp_to_power_1p[amps_val] = float(power_value_or_delta)
                    if amps_val % 3 == 0 and amps_val//3 >= self.car_charger_min_charge and amps_val//3 <= self.car_charger_max_charge:
                        self.customized_amp_to_power_3p[amps_val//3] = float(power_value_or_delta)
            elif amps_val <= self._conf_car_charger_min_charge + 2:
                power_value_or_delta = 0.0
                # limite the possibility to have amps 0
                for i in range(0, amps_val+1):
                    # no need to do per phase for 0: the car won't take current on amps values only
                    self.customized_amp_to_power_3p[i] = 0.0
                    self.customized_amp_to_power_1p[i] = 0.0

                if car_percent is None or car_percent < 90.0:
                    do_recompute_min_charge = True

            self.interpolate_power_steps(do_recompute_min_charge=do_recompute_min_charge)
            do_update = True
            if can_be_saved and self.config_entry and car_percent is not None and car_percent > 10 and car_percent < 70:

                if self.car_is_custom_power_charge_values_3p is None:
                    self.car_is_custom_power_charge_values_3p = for_3p

                # only save what was set as conf
                if for_3p == self.car_is_custom_power_charge_values_3p:

                    self.car_use_custom_power_charge_values = True

                    #ok this value can be saved ... we see above for now we force to not save it
                    self._salvable_dampening[CONF_CAR_CUSTOM_POWER_CHARGE_VALUES] = self.car_use_custom_power_charge_values
                    self._salvable_dampening[CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P] = self.car_is_custom_power_charge_values_3p
                    self._salvable_dampening[f"charge_{amps_val}"] = power_value_or_delta

                    if self._last_dampening_update is None or (time - self._last_dampening_update).total_seconds() > 300:
                        self._last_dampening_update = time
                        data = dict(self.config_entry.data)
                        data.update(self._salvable_dampening)
                        self.hass.config_entries.async_update_entry(self.config_entry, data=data)

        return do_update

    def _interpolate_power_steps(self, customized_amp_to_power, theoretical_amp_to_power, amp_to_power) -> int|float:


        min_charge = self._conf_car_charger_min_charge

        prev_measured_val = customized_amp_to_power[min_charge]

        if prev_measured_val == 0.0 or prev_measured_val > 0 and prev_measured_val < MIN_CHARGE_POWER_W:
            orig_min_charge = min_charge
            for i in range(orig_min_charge, self.car_charger_max_charge):
                prev_measured_val = customized_amp_to_power[i]
                if prev_measured_val == 0.0 or prev_measured_val > 0 and prev_measured_val < MIN_CHARGE_POWER_W:
                    min_charge = i + 1
                    customized_amp_to_power[i] = 0.0
                else:
                    break

        new_vals: list[float] = [0.0] * (len(theoretical_amp_to_power))

        prev_measured_a = min_charge
        prev_measured_val = customized_amp_to_power[prev_measured_a]

        # -1 is by default, so we can use it to detect if no value was set
        if prev_measured_val < 0:
            # compute a best possible first
            prev_measured_val = theoretical_amp_to_power[min_charge]
            first = None
            second = None
            for a in range(min_charge+1, self.car_charger_max_charge + 1):
                if customized_amp_to_power[a] > 0:
                    if first is None:
                        first = a
                    elif second is None:
                        second = a
                    else:
                        break

            # interpolate the first possible value if we do have some measures, else it will be the theoretical value
            if first is not None and second is not None:
                first_possible_val = (min_charge - first) * ((customized_amp_to_power[second] -  customized_amp_to_power[first])/(second - first)) + customized_amp_to_power[first]
                if first_possible_val > 0:
                    prev_measured_val = min(first_possible_val, prev_measured_val)


        new_vals[min_charge] = prev_measured_val

        for a in range(min_charge+1, self.car_charger_max_charge + 1):

            measured = customized_amp_to_power[a]

            if a == self.car_charger_max_charge:
                if measured < 0 or (prev_measured_val > 0 and measured > 0 and measured < prev_measured_val):
                    measured = max(prev_measured_val, theoretical_amp_to_power[self.car_charger_max_charge])


            if measured > prev_measured_val or a == self.car_charger_max_charge:
                # only increasing values allowed
                new_vals[a] = measured
                if a > prev_measured_a + 1:
                    for ap in range(prev_measured_a+1, a):
                        new_vals[ap] = prev_measured_val + ((measured - prev_measured_val) * (ap - prev_measured_a) / (a - prev_measured_a))
                prev_measured_a = a
                prev_measured_val = measured



        for a in range(0, len(theoretical_amp_to_power)):
            amp_to_power[a] = new_vals[a]

        return min_charge




    def interpolate_power_steps(self, do_recompute_min_charge=False, use_conf_values=False):

        if use_conf_values:
            customized_amp_to_power_3p = self.conf_customized_amp_to_power_3p
            customized_amp_to_power_1p = self.conf_customized_amp_to_power_1p
        else:
            customized_amp_to_power_3p = self.customized_amp_to_power_3p
            customized_amp_to_power_1p = self.customized_amp_to_power_1p

        min_charge_3p = self._interpolate_power_steps(customized_amp_to_power_3p,
                                      self.theoretical_amp_to_power_3p,
                                      self.amp_to_power_3p)
        min_charge_1p = self._interpolate_power_steps(customized_amp_to_power_1p,
                                      self.theoretical_amp_to_power_1p,
                                      self.amp_to_power_1p)

        if use_conf_values:
            self.car_charger_min_charge = self._conf_car_charger_min_charge

        if do_recompute_min_charge:
            init_car_min_charge = self.car_charger_min_charge
            self.car_charger_min_charge = max(min_charge_3p, min_charge_1p)
            if init_car_min_charge != self.car_charger_min_charge:
                _LOGGER.info(
                    f"interpolate_power_steps: Car {self.name} updated min charge from {init_car_min_charge} to {self.car_charger_min_charge}")


    def get_charge_power_per_phase_A(self, for_3p:bool) -> tuple[list[float], int, int]:
        if for_3p:
            return self.amp_to_power_3p, self.car_charger_min_charge, self.car_charger_max_charge
        else:
            return self.amp_to_power_1p, self.car_charger_min_charge, self.car_charger_max_charge


    async def add_default_charge_at_datetime(self, end_charge:datetime):
        if self.calendar is None:
            return
        start_time = end_charge
        end_time = end_charge + timedelta(seconds=60*30)
        await self.set_next_scheduled_event(start_time, end_time, f"Charge {self.name}")


    async def add_default_charge_at_dt_time(self, default_charge_time:dt_time | None):
        if self.calendar is None:
            return

        if default_charge_time is None:
            _LOGGER.error(f"Car {self.name} cannot add default charge at None time")
            return

        # compute the next occurrence of the default charge time
        dt_now = datetime.now(tz=None)
        next_time = datetime(year=dt_now.year, month=dt_now.month, day=dt_now.day, hour=default_charge_time.hour,
                             minute=default_charge_time.minute, second=default_charge_time.second)
        if next_time < dt_now:
            next_time = next_time + timedelta(days=1)

        next_time = next_time.replace(tzinfo=None).astimezone(tz=pytz.UTC)

        await self.add_default_charge_at_datetime(next_time)


    async def add_default_charge(self):
        if self.can_add_default_charge():
            await self.add_default_charge_at_dt_time(self.default_charge_time)

    def can_add_default_charge(self) -> bool:
        if self.charger is not None and self.calendar is not None:
            return True
        return False

    def can_force_a_charge_now(self) -> bool:
        if self.charger is not None:
            return True
        return False

    async def force_charge_now(self):
        if self.can_force_a_charge_now():
            self.do_force_next_charge = True

    async def setup_car_charge_target_if_needed(self, asked_target_charge=None):

        target_charge = asked_target_charge

        if target_charge is None:
            target_charge = self.get_car_target_SOC()

        if target_charge is not None:
            await self.adapt_max_charge_limit(target_charge)

        return target_charge




    def get_car_next_charge_values_options(self):
        time = datetime.now(pytz.UTC)
        current_soc = self.get_car_charge_percent(time)
        if current_soc is None:
            current_soc = 0

        options = set()
        options.add(100)

        if self.car_charge_percent_max_number_steps and len(self.car_charge_percent_max_number_steps) >= 1:
            for v in self.car_charge_percent_max_number_steps:
                if v > current_soc:
                    options.add(v)
        else:
            if current_soc < 90:
                first = int(current_soc // 10 + 1)
                if first < 10:
                    for i in range(first, 10):
                        options.add(i * 10)

            # 85 is a special case, as it is usefull for NMC cars
            if current_soc < 85:
                options.add(85)

        # if current_soc < self.car_default_charge:
        # always add the default
        options.add(int(self.car_default_charge))

        v = int(self.get_car_target_SOC())
        #always add the current set
        options.add(v)

        options = list(options)
        options.sort()

        for i in range(len(options)):
            options[i] = self.get_car_option_charge_from_value(options[i])

        return options

    def get_car_option_charge_from_value(self, value:int|float):
        value = int(float(value))
        if value > 100:
            value = 100

        if value == self.car_default_charge:
            return f"{value}% - {self.name} default"
        elif value == 100:
            return "100% - full"
        else:
            return f"{value}%"


    async def set_next_charge_target(self, value:int|float|str):

        if isinstance(value, str):
            if "default" in value:
                value = self.car_default_charge
            elif "full" in value:
                value = 100
            else:
                try:
                    value = value.strip("%")
                    value = float(value)
                except ValueError:
                    _LOGGER.error(f"Car {self.name} set_next_charge_target: invalid value {value}, must be an integer or 'default' or 'full'")
                    return

        value = int(value)

        self._next_charge_target = value

        new_target = await self.setup_car_charge_target_if_needed()

        if self.charger:
            if new_target and self.charger._constraints:
                for ct in self.charger._constraints:
                    if isinstance(ct, MultiStepsPowerLoadConstraintChargePercent):
                        ct.target_value = new_target
                        break

    def get_car_target_charge_option(self):
        return self.get_car_option_charge_from_value(self.get_car_target_SOC())

    def get_car_target_SOC(self) -> int | float:
        if self._next_charge_target is None:
            self._next_charge_target = self.car_default_charge
        return self._next_charge_target

    def get_car_minimum_ok_SOC(self) -> int | float:
        return self.car_minimum_ok_charge



    @property
    def qs_bump_solar_charge_priority(self) -> bool:
        return self._qs_bump_solar_priority

    @qs_bump_solar_charge_priority.setter
    def qs_bump_solar_charge_priority(self, value: bool):
        if value is False:
            self._qs_bump_solar_priority = False
        else:
            # only one can have a bump of the entire chargers list
            for c in self.home._cars:
                c.qs_bump_solar_charge_priority = False
            self._qs_bump_solar_priority = True

    def get_charger_options(self)  -> list[str]:

        time = datetime.now(pytz.UTC)

        options = []
        for charger in self.home._chargers:
            if charger.is_optimistic_plugged(time):
                options.append(charger.name)
            else:
                charger.user_attached_car_name = None

        options.append(FORCE_CAR_NO_CHARGER_CONNECTED)
        return options


    def get_current_selected_charger_option(self) -> str | None:
        if self.user_attached_charger_name is not None:
            return self.user_attached_charger_name

        if self.charger is None:
            return None
        else:
            return self.charger.name

    async def set_user_selected_charger_by_name(self, charger_name: str | None):

        # if the car is already attached to a charger, we detach it
        if self.charger is not None and self.charger.name != charger_name:
            self.detach_charger()

        if charger_name == FORCE_CAR_NO_CHARGER_CONNECTED:
            self.user_attached_charger_name = FORCE_CAR_NO_CHARGER_CONNECTED
            return

        self.user_attached_charger_name = None

        if charger_name is not None:
            charger = None
            for c in self.home._chargers:
                if c.name == charger_name:
                    charger = c
                    break

            if charger is not None:
                await charger.set_user_selected_car_by_name(car_name=self.name)

    async def clean_and_reset(self):
        if self.charger is not None:
            await self.charger.clean_and_reset()
        self.reset()
