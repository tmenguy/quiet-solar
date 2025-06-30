import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Awaitable
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE, ATTR_ENTITY_ID
from homeassistant.components import number, homeassistant
from ..const import CONF_CAR_PLUGGED, CONF_CAR_TRACKER, CONF_CAR_CHARGE_PERCENT_SENSOR, CONF_CAR_CHARGE_PERCENT_MAX_NUMBER, \
    CONF_CAR_BATTERY_CAPACITY, CONF_CAR_CHARGER_MIN_CHARGE, CONF_CAR_CHARGER_MAX_CHARGE, \
    CONF_CAR_CUSTOM_POWER_CHARGE_VALUES, CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P, MAX_POSSIBLE_APERAGE, \
    CONF_DEFAULT_CAR_CHARGE, CONF_CAR_IS_DEFAULT, CONF_MOBILE_APP, CONF_MOBILE_APP_NOTHING
from ..ha_model.device import HADeviceMixin
from ..home_model.load import AbstractDevice


_LOGGER = logging.getLogger(__name__)

MIN_CHARGE_POWER_W = 150


class QSCar(HADeviceMixin, AbstractDevice):

    def __init__(self, **kwargs):
        self.car_plugged = kwargs.pop(CONF_CAR_PLUGGED, None)
        self.car_tracker = kwargs.pop(CONF_CAR_TRACKER, None)
        self.car_charge_percent_sensor = kwargs.pop(CONF_CAR_CHARGE_PERCENT_SENSOR, None)
        self.car_charge_percent_max_number = kwargs.pop(CONF_CAR_CHARGE_PERCENT_MAX_NUMBER, None)
        self.car_battery_capacity = kwargs.pop( CONF_CAR_BATTERY_CAPACITY, None)
        self.car_default_charge = kwargs.pop(CONF_DEFAULT_CAR_CHARGE, 100.0)
        self.car_is_default = kwargs.pop(CONF_CAR_IS_DEFAULT, False)

        self.car_charger_min_charge : int = int(max(0,kwargs.pop(CONF_CAR_CHARGER_MIN_CHARGE, 6)))
        self._conf_car_charger_min_charge = self.car_charger_min_charge
        self.car_charger_max_charge : int = min(MAX_POSSIBLE_APERAGE, int(max(0,kwargs.pop(CONF_CAR_CHARGER_MAX_CHARGE,32))))
        self._conf_car_charger_max_charge = self.car_charger_max_charge
        self.car_use_custom_power_charge_values = kwargs.pop(CONF_CAR_CUSTOM_POWER_CHARGE_VALUES, False)
        if self.car_use_custom_power_charge_values is False:
            self.car_is_custom_power_charge_values_3p = None
        else:
            self.car_is_custom_power_charge_values_3p = kwargs.pop(CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P, False)

        self.amp_to_power_1p = [-1] * (MAX_POSSIBLE_APERAGE)
        self.amp_to_power_3p = [-1] * (MAX_POSSIBLE_APERAGE)
        self._last_dampening_update = None


        self.theoretical_amp_to_power_1p = [-1] * (MAX_POSSIBLE_APERAGE)
        self.theoretical_amp_to_power_3p = [-1] * (MAX_POSSIBLE_APERAGE)

        self.customized_amp_to_power_1p = [-1] * (MAX_POSSIBLE_APERAGE)
        self.customized_amp_to_power_3p = [-1] * (MAX_POSSIBLE_APERAGE)

        self.conf_customized_amp_to_power_1p = [-1] * (MAX_POSSIBLE_APERAGE)
        self.conf_customized_amp_to_power_3p = [-1] * (MAX_POSSIBLE_APERAGE)

        super().__init__(**kwargs)

        self._conf_calendar = self.calendar

        for a in range(len(self.theoretical_amp_to_power_1p)):

            val_1p = float(self.home.voltage * a)
            val_3p = 3*val_1p

            self.amp_to_power_1p[a] = self.theoretical_amp_to_power_1p[a] = val_1p
            self.amp_to_power_3p[a] = self.theoretical_amp_to_power_3p[a] = val_3p

        if self.car_use_custom_power_charge_values:

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

        self.reset()

    def reset(self):
        super().reset()
        self.interpolate_power_steps(do_recompute_min_charge=True, use_conf_values=True)
        self._dampening_deltas = {}
        self._dampening_deltas_graph = {}
        self.calendar = self._conf_calendar

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

    def get_car_charge_percent(self, time: datetime, tolerance_seconds: float=4*3600 ) -> float | None:
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

    async def set_max_charge_limit(self, percent):

        if self.car_charge_percent_max_number is None:
            return

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
        from_power = None
        if from_amp == 0:
            from_power = 0.0
        elif from_amp >= self.car_charger_min_charge and from_amp <= self.car_charger_max_charge:
            if from_num_phase == 1:
                from_power = self.amp_to_power_1p[from_amp]
            else:
                from_power = self.amp_to_power_3p[from_amp]
        return from_power



    def get_delta_dampened_power(self, from_amp: int | float, from_num_phase: int, to_amp: int | float, to_num_phase: int) -> tuple[float | None, float | None, float | None]:

        from_power = self._get_power_from_stored_amps(from_amp, from_num_phase)
        to_power = self._get_power_from_stored_amps(to_amp, to_num_phase)


        if from_amp*from_num_phase == to_amp*to_num_phase:
            return 0.0, from_power, to_power

        if from_power is None or to_power is None:
            return None, from_power, to_power

        power = None

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
                            return None, from_power, to_power

                        power += p
                elif path:
                    _LOGGER.error(
                        f"get_delta_dampened_power path error: Car {self.name} deltas {self._dampening_deltas} graph {self._dampening_deltas_graph} from_amp {from_amp} to_amp {to_amp}")

        return power, from_power, to_power



    def _theoretical_max_power(self, amperage:tuple[float,int] | tuple[int,int], delta_amp:float) -> float:
        if amperage[0] == 0:
            return 0.0
        theoretical_power = float(self.home.voltage * max(0.0, amperage[0] + delta_amp))
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


        from_theoretical = self._theoretical_max_power(from_a, -0.6)
        to_theoretical = self._theoretical_max_power(to_a, 0.6)

        if power_delta > to_theoretical - from_theoretical:
            _LOGGER.warning(
                f"_add_to_amps_power_graph: {self.name} power_delta {power_delta} > theoretical_delta {to_theoretical - from_theoretical} from {from_a} to {to_a} - ignoring this value"
            )
            return False


        self._dampening_deltas[(from_amp, to_amp)] = power_delta
        self._dampening_deltas[(to_amp, from_amp)] = -power_delta

        fs = self._dampening_deltas_graph.setdefault(from_amp, set())
        fs.add(to_amp)
        ts = self._dampening_deltas_graph.setdefault(to_amp, set())
        ts.add(from_amp)

        return True


    def update_dampening_value(self, amperage: None | tuple[float,int] | tuple[int,int], amperage_transition: None | tuple[tuple[int,int] | tuple[float,int], tuple[int,int] | tuple[float,int]], power_value_or_delta: int | float, time:datetime, can_be_saved:bool = False) -> bool:

        do_update = False

        if amperage_transition is None and amperage is None:
            return do_update

        if amperage_transition is not None:

            if amperage is None:
                if amperage_transition[0][0] == 0:
                    amperage = amperage_transition[1]
                elif amperage_transition[1][0] == 0:
                    amperage = amperage_transition[0]
                    power_value_or_delta = -power_value_or_delta

            if amperage is None:

                if self._add_to_amps_power_graph(amperage_transition[0], amperage_transition[1], power_value_or_delta) is False:
                    return do_update


        if amperage is not None:

            if amperage[0] < self.car_charger_min_charge or amperage[0] > self.car_charger_max_charge:
                return False


            if amperage[1] == 3:
                for_3p = True
            else:
                for_3p = False

            amps_val = int(amperage[0])



            if self._add_to_amps_power_graph((0.0, amperage[1]), (amperage[0], amperage[1]), power_value_or_delta) is False:
                return do_update

            if power_value_or_delta < MIN_CHARGE_POWER_W:
                # we may have a 0 value for a given amperage actually it could change the min and max amperage
                power_value_or_delta = 0

            if for_3p:
                old_val = self.amp_to_power_3p[amps_val]
            else:
                old_val = self.amp_to_power_1p[amps_val]

            do_update = False
            if (power_value_or_delta == 0 or old_val == 0):
                if power_value_or_delta == old_val:
                    do_update = False
                else:
                    do_update = True
            elif abs(old_val - power_value_or_delta) > 0.1*max(old_val, power_value_or_delta):
                do_update = True

            if do_update:

                can_be_saved = False

                if for_3p:
                    self.customized_amp_to_power_3p[amps_val] = float(power_value_or_delta)
                    if 3*amps_val <= self.car_charger_max_charge and 3*amps_val >= self.car_charger_min_charge:
                        self.customized_amp_to_power_1p[3*amps_val] = float(power_value_or_delta)
                else:
                    self.customized_amp_to_power_1p[amps_val] = float(power_value_or_delta)
                    if amps_val % 3 == 0 and amps_val//3 >= self.car_charger_min_charge and amps_val//3 <= self.car_charger_max_charge:
                        self.customized_amp_to_power_3p[amps_val//3] = float(power_value_or_delta)

                do_recompute_min_charge = False
                if can_be_saved and power_value_or_delta == 0:
                    do_recompute_min_charge = True

                self.interpolate_power_steps(do_recompute_min_charge=do_recompute_min_charge)

                car_percent = self.get_car_charge_percent(time)

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

    def _interpolate_power_steps(self, customized_amp_to_power, theoretical_amp_to_power, amp_to_power):


        min_charge = self._conf_car_charger_min_charge

        prev_measured_val = customized_amp_to_power[min_charge]

        new_vals: list[float] = [0.0] * (len(theoretical_amp_to_power))

        prev_measured_a = min_charge

        if prev_measured_val <= 0:
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

            if first is not None and second is not None:
                first_possible_val = (min_charge - first) * ((customized_amp_to_power[second] -  customized_amp_to_power[first])/(second - first)) + customized_amp_to_power[first]
                if first_possible_val > 0:
                    prev_measured_val = min(first_possible_val, prev_measured_val)


        new_vals[min_charge] = prev_measured_val

        for a in range(min_charge+1, self.car_charger_max_charge + 1):

            measured = customized_amp_to_power[a]

            if a == self.car_charger_max_charge:
                if measured <= 0 or (prev_measured_val > 0 and measured > 0 and measured < prev_measured_val):
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




    def interpolate_power_steps(self, do_recompute_min_charge=False, use_conf_values=False):

        if use_conf_values:
            customized_amp_to_power_3p = self.conf_customized_amp_to_power_3p
            customized_amp_to_power_1p = self.conf_customized_amp_to_power_1p
        else:
            customized_amp_to_power_3p = self.customized_amp_to_power_3p
            customized_amp_to_power_1p = self.customized_amp_to_power_1p

        self._interpolate_power_steps(customized_amp_to_power_3p,
                                      self.theoretical_amp_to_power_3p,
                                      self.amp_to_power_3p)
        self._interpolate_power_steps(customized_amp_to_power_1p,
                                      self.theoretical_amp_to_power_1p,
                                      self.amp_to_power_1p)

        if do_recompute_min_charge or use_conf_values:
            self.car_charger_min_charge = self._conf_car_charger_min_charge
            if use_conf_values is False:
                for i, val in enumerate(self.amp_to_power_3p):
                    if i < self._conf_car_charger_min_charge:
                        continue
                    if 0 <= val < MIN_CHARGE_POWER_W:
                        self.car_charger_min_charge = i + 1
                    else:
                        break

                for i, val in enumerate(self.amp_to_power_1p):
                    if i < self.car_charger_min_charge: # use self.car_charger_min_charge instead of self._conf_car_charger_min_charge as it has been updated by the loop above
                        continue
                    if 0 <= val < MIN_CHARGE_POWER_W:
                        self.car_charger_min_charge = i + 1
                    else:
                        break


    def get_charge_power_per_phase_A(self, for_3p:bool) -> tuple[list[float], int, int]:
        if for_3p:
            return self.amp_to_power_3p, self.car_charger_min_charge, self.car_charger_max_charge
        else:
            return self.amp_to_power_1p, self.car_charger_min_charge, self.car_charger_max_charge


    def get_platforms(self):
        parent = super().get_platforms()
        if parent is None:
            parent = set()
        else:
            parent = set(parent)
        parent.update([ Platform.SENSOR, Platform.SELECT ])
        return list(parent)

    async def add_default_charge(self, end_charge:datetime):
        if self.calendar is None:
            return
        start_time = end_charge
        end_time = end_charge + timedelta(seconds=60*30)
        await self.set_next_scheduled_event(start_time, end_time, f"Charge {self.name}")


