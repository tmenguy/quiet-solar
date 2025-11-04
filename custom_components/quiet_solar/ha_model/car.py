import bisect
import logging
from operator import itemgetter
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING

import pytz
from haversine import haversine, Unit
from homeassistant.components.recorder.models import LazyState
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE, ATTR_ENTITY_ID
from homeassistant.components.recorder.history import state_changes_during_period
from homeassistant.components.recorder import get_instance as recorder_get_instance
from homeassistant.components import number

from .device import convert_distance_to_km, load_from_history

from ..const import CONF_CAR_PLUGGED, CONF_CAR_TRACKER, CONF_CAR_CHARGE_PERCENT_SENSOR, \
    CONF_CAR_CHARGE_PERCENT_MAX_NUMBER, \
    CONF_CAR_BATTERY_CAPACITY, CONF_CAR_CHARGER_MIN_CHARGE, CONF_CAR_CHARGER_MAX_CHARGE, \
    CONF_CAR_CUSTOM_POWER_CHARGE_VALUES, CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P, MAX_POSSIBLE_AMPERAGE, \
    CONF_DEFAULT_CAR_CHARGE, CONF_CAR_IS_INVITED, FORCE_CAR_NO_CHARGER_CONNECTED, \
    CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS, CONF_MINIMUM_OK_CAR_CHARGE, CONF_TYPE_NAME_QSCar, \
    CAR_CHARGE_TYPE_NOT_PLUGGED, CONF_CAR_ODOMETER_SENSOR, CONF_CAR_ESTIMATED_RANGE_SENSOR, \
    CAR_EFFICIENCY_KM_PER_KWH, CAR_HARD_WIRED_CHARGER, FORCE_CAR_NO_PERSON_ATTACHED
from ..ha_model.device import HADeviceMixin
from ..home_model.load import AbstractDevice
from datetime import time as dt_time

if TYPE_CHECKING:
    from ..ha_model.person import QSPerson


_LOGGER = logging.getLogger(__name__)

MIN_CHARGE_POWER_W = 70

CAR_MAX_EFFICIENCY_HISTORY_S = 3600*24*31

CAR_DEFAULT_CAPACITY = 100000 # 100 kWh

CAR_MINIMUM_LEFT_RANGE_KM = 30.0

class QSCar(HADeviceMixin, AbstractDevice):

    conf_type_name = CONF_TYPE_NAME_QSCar

    def __init__(self, **kwargs):

        self.car_hard_wired_charger = kwargs.pop(CAR_HARD_WIRED_CHARGER, None)

        self.car_plugged = kwargs.pop(CONF_CAR_PLUGGED, None)
        self.car_tracker = kwargs.pop(CONF_CAR_TRACKER, None)
        self.car_charge_percent_sensor = kwargs.pop(CONF_CAR_CHARGE_PERCENT_SENSOR, None)
        self.car_charge_percent_max_number = kwargs.pop(CONF_CAR_CHARGE_PERCENT_MAX_NUMBER, None)
        self.car_odometer_sensor = kwargs.pop(CONF_CAR_ODOMETER_SENSOR, None)
        self.car_estimated_range_sensor = kwargs.pop(CONF_CAR_ESTIMATED_RANGE_SENSOR, None)
        self._conf_car_charge_percent_max_number_steps = kwargs.pop(CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS, None)
        if self._conf_car_charge_percent_max_number_steps == "":
            self._conf_car_charge_percent_max_number_steps = None
        self.car_battery_capacity = kwargs.pop( CONF_CAR_BATTERY_CAPACITY, None)
        self.car_default_charge = kwargs.pop(CONF_DEFAULT_CAR_CHARGE, 100.0)
        self.car_minimum_ok_charge = kwargs.pop(CONF_MINIMUM_OK_CAR_CHARGE, 50.0)

        self.car_efficiency_km_per_kwh_sensor : str = CAR_EFFICIENCY_KM_PER_KWH

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

        self._salvable_dampening = {}

        self._dampening_deltas = {}
        self._dampening_deltas_graph = {}

        self.charger = None

        self.do_force_next_charge = False
        self.do_next_charge_time : datetime | None = None

        self._next_charge_target = None
        self._next_charge_target_energy = None
        self.user_attached_charger_name : str | None = None

        self.default_charge_time: dt_time | None = None

        self._qs_bump_solar_priority = False

        # Efficiency learning state
        self._km_per_kwh: float | None = None
        self._efficiency_segments: list[tuple[float, float, float, float, datetime]] = []  # delta_km, delta_soc, soc_from, soc_to, time of finishing

        # self._efficiency_deltas = {}
        # self._efficiency_deltas_graph = {}

        self._decreasing_segments: list[list[tuple[float, float]|None|int]] = []
        self._dec_seg_count = 0

        self.current_forecasted_person : QSPerson | None = None

        self.reset()

        self._user_selected_person_name_for_car : str | None = None

        self.attach_ha_state_to_probe(self.car_charge_percent_sensor,
                                      is_numerical=True)

        self.attach_ha_state_to_probe(self.car_plugged,
                                      is_numerical=False)

        self.attach_ha_state_to_probe(self.car_tracker,
                                      is_numerical=False)

        self.attach_ha_state_to_probe(self.car_odometer_sensor,
                                      conversion_fn=convert_distance_to_km,
                                      is_numerical=True)

        self.attach_ha_state_to_probe(self.car_estimated_range_sensor,
                                      conversion_fn=convert_distance_to_km,
                                      is_numerical=True)

        self.attach_ha_state_to_probe(self.car_efficiency_km_per_kwh_sensor,
                                      is_numerical=True,
                                      non_ha_entity_get_state=self.car_efficiency_km_per_kwh_sensor_state_getter)

    # make a property out of user_selected_person_name_for_car
    @property
    def user_selected_person_name_for_car(self) -> str | None:
        return self._user_selected_person_name_for_car

    @user_selected_person_name_for_car.setter
    def user_selected_person_name_for_car(self, value: str | None):
        do_update = False
        if value != self._user_selected_person_name_for_car:
            do_update = True
        self._user_selected_person_name_for_car = value
        if do_update and self.home is not None:
            self.home.get_best_persons_cars_allocations(force_update=True)

    def _car_person_option(self, person_name: str):
        return person_name

    def update_to_be_saved_extra_device_info(self, data_to_update:dict):
        super().update_to_be_saved_extra_device_info(data_to_update)
        # do not use the property, but teh underlying model
        data_to_update["user_selected_person_name_for_car"] = self._user_selected_person_name_for_car

    def use_saved_extra_device_info(self, stored_load_info: dict):
        super().use_saved_extra_device_info(stored_load_info)
        # do not use the property to not trigger an unnecessary compute of the people allocation
        self._user_selected_person_name_for_car = stored_load_info.get("user_selected_person_name_for_car", None)


    def get_car_persons_options(self) -> list[str]:
        options = []
        if self.home:

            # get the possible persons for the house and their needed charge
            for person in self.home._persons:

                if self.name not in person.authorized_cars:
                    continue

                opt_person = self._car_person_option(person.name)
                if opt_person is not None:
                    options.append(opt_person)

        options.append(FORCE_CAR_NO_PERSON_ATTACHED)
        return options


    def get_car_person_option(self) -> str | None:

        p_name = None
        if self.user_selected_person_name_for_car is not None:
            p_name = self.user_selected_person_name_for_car
        elif self.current_forecasted_person is not None:
            p_name = self.current_forecasted_person.name

        # check the attributed one if any by the home
        if p_name is not None:
            return self._car_person_option(p_name)

        return None


    async def set_user_person_for_car(self, option:str):

        if option is None:
            # user_selected_person_name_for_car will trigger an update if needed
            self.user_selected_person_name_for_car = FORCE_CAR_NO_PERSON_ATTACHED
        elif option == FORCE_CAR_NO_PERSON_ATTACHED:
            # user_selected_person_name_for_car will trigger an update if needed
            self.user_selected_person_name_for_car = FORCE_CAR_NO_PERSON_ATTACHED
        else:
            # do not use the property to not trigger an unnecessary compute of the people allocation
            do_need_update = False
            new_value = option

            if new_value != self.user_selected_person_name_for_car:
                do_need_update = True

            self._user_selected_person_name_for_car = new_value

            if self.home:
                for car in self.home._cars:
                    if car.name == self.name:
                        continue

                    if car.user_selected_person_name_for_car == new_value:
                        # do not use the property to not trigger an unnecessary compute of the people allocation
                        car._user_selected_person_name_for_car = None
                        do_need_update = True

            # now we should recompute all car assignment with this new one, and update everything that should be updated
            if do_need_update and self.home:
                self.home.get_best_persons_cars_allocations(force_update=True)

        return None


    async def get_car_mileage_on_period_km(self, from_time: datetime, to_time: datetime) -> float | None:

        res = None

        if self.car_odometer_sensor is not None:
            car_odometers : list[LazyState] = await load_from_history(self.hass, self.car_odometer_sensor, from_time - timedelta(days=2), to_time, no_attributes=False)

            prev_state = None
            from_state = None
            for odo_state in car_odometers:
                if odo_state is None or odo_state.state == STATE_UNKNOWN or odo_state.state == STATE_UNAVAILABLE:
                    continue
                try:
                    v = float(odo_state.state)
                except (TypeError, ValueError):
                    continue

                if odo_state.last_changed > from_time:
                    if prev_state is not None:
                        from_state = prev_state
                    else:
                        from_state = odo_state
                    break
                else:
                    prev_state = odo_state

            to_state = None
            for odo_state in reversed(car_odometers):
                if odo_state is None or odo_state.state == STATE_UNKNOWN or odo_state.state == STATE_UNAVAILABLE:
                    continue
                try:
                    v = float(odo_state.state)
                except (TypeError, ValueError):
                    continue

                to_state = odo_state
                break

            if from_state is None or to_state is None:
                res =  None
            else:

                from_km, _ = convert_distance_to_km(float(from_state.state), from_state.attributes)
                to_km, _ = convert_distance_to_km(float(to_state.state), to_state.attributes)

                return to_km - from_km

        if res is None and self.car_tracker is not None:
            car_positions = await load_from_history(self.hass, self.car_tracker, from_time, to_time, no_attributes=False)

            prev_pos = None
            for car_position in car_positions:
                if car_position is None or car_position.state == STATE_UNKNOWN or car_position.state == STATE_UNAVAILABLE:
                    continue

                state_attr: dict[str, Any] = car_position.attributes

                if state_attr is None:
                    continue

                latitude = state_attr.get("latitude", None)
                longitude = state_attr.get("longitude", None)

                if latitude is None or longitude is None:
                    continue

                cur_pos = (float(latitude), float(longitude))

                if prev_pos is not None:
                    res_add = haversine(prev_pos, cur_pos, unit=Unit.KILOMETERS)
                    if res is None:
                        res = 0.0
                    res += res_add

                prev_pos = cur_pos

        return res

    async def get_best_person_next_need(self, time:datetime) -> tuple[bool | None, datetime | None, float | None, Any | None]:
        if self.home:
            self.home.get_best_persons_cars_allocations(time)

            person = self.current_forecasted_person

            if person is not None:

                next_usage_time, p_mileage = person.update_person_forecast(time)
                is_person_covered, current_soc, person_min_target_charge, diff_energy = self.get_adapt_target_percent_soc_to_reach_range_km(p_mileage, time)

                return is_person_covered, next_usage_time, person_min_target_charge, person

        return (None, None, None, None)



    def car_efficiency_km_per_kwh_sensor_state_getter(self, entity_id: str,  time: datetime | None) -> (tuple[datetime | None, float | str | None, dict | None] | None):

        # Learn efficiency only when car is unplugged and we have both sensors
        if self.car_odometer_sensor is None or self.car_charge_percent_sensor is None:
            return None

        soc = self.get_car_charge_percent(time)
        odo = self.get_car_odometer_km(time)
        if soc is None or odo is None:
            return None

        # check what was before this sample
        prev_sample = None
        prev_seg_idx = None

        if len(self._decreasing_segments) > 0:
            prev_sample = self._decreasing_segments[-1][1]
            if prev_sample is None:
                prev_sample = self._decreasing_segments[-1][0]
            prev_seg_idx = self._decreasing_segments[-1][2]

        self._add_soc_odo_value_to_segments(soc, odo, time)

        sample_eff = None
        if self.car_estimated_range_sensor is not None:
            current_soc = self.get_car_charge_percent(time)
            car_estimate = self.get_car_estimated_range_km_from_sensor(time)
            if current_soc is not None and car_estimate is not None and current_soc > 0.0:
                sample_eff = float(car_estimate) / ((float(current_soc) / 100.0) * (float(self.car_battery_capacity) / 1000.0))

        if sample_eff is None:
            if prev_sample is not None and self._decreasing_segments[-1][2] == prev_seg_idx and prev_sample[0] > soc:
                # we have added the new point to the last segment
                prev_soc, prev_odo, prev_time = prev_sample
                if soc < prev_soc and odo > prev_odo:
                    # progressive EMA from segment start to current
                    if self.car_battery_capacity is not None and self.car_battery_capacity > 0:
                        seg_soc0, seg_odo0, seg_t0 = self._decreasing_segments[-1][0]
                        distance_km = float(odo - seg_odo0)
                        delta_soc = float(seg_soc0 - soc)
                        energy_kwh = (delta_soc / 100.0) * (float(self.car_battery_capacity) / 1000.0)
                        if energy_kwh > 0.0 and distance_km > 0.0:
                            sample_eff = distance_km / energy_kwh

        if sample_eff is not None:
            if self._km_per_kwh is None:
                self._km_per_kwh = sample_eff
            else:
                alpha = 0.2
                self._km_per_kwh = alpha * sample_eff + (1.0 - alpha) * self._km_per_kwh

        return (time, self._km_per_kwh, {})


    def get_car_charge_type(self) -> str:
        if self.charger is None:
            return CAR_CHARGE_TYPE_NOT_PLUGGED
        else:
            return self.charger.get_charge_type()

    def get_car_charge_time_readable_name(self):

        if self.charger is None:
            return "--:--"

        # set time as now
        time = datetime.now(pytz.UTC)

        current_constraint = self.charger.get_current_active_constraint(time)

        if current_constraint is None:
            return "--:--"

        return current_constraint.get_readable_next_target_date_string(for_small_standalone=True)

    @property
    def dashboard_sort_string_in_type(self) -> str:
        if self.car_is_invited:
            return "ZZZ"
        return "AAA"

    def get_platforms(self):
        parent = super().get_platforms()
        if parent is None:
            parent = set()
        else:
            parent = set(parent)
        parent.update([Platform.SENSOR, Platform.SELECT, Platform.SWITCH, Platform.BUTTON, Platform.TIME])
        return list(parent)

    def device_post_home_init(self, time: datetime):
        # Try to bootstrap efficiency from history at startup (best-effort, non-blocking)
        super().device_post_home_init(time)
        try:
            # Asynchronously try to compute an initial km/kWh from HA history
            if self.hass is not None:
                self.hass.async_create_task(self._async_bootstrap_efficiency_from_history(time))
        except Exception:
            pass


    def _add_soc_odo_value_to_segments(self, soc:float, odo:float, time:datetime):

        current_vals = (soc, odo, time)

        if len(self._decreasing_segments) > 0:
            current_segment = self._decreasing_segments[-1]
        else:
            current_segment = [current_vals, None, self._dec_seg_count]
            self._decreasing_segments.append(current_segment)
            self._dec_seg_count += 1
            return

        if current_segment[1] is None:
            if soc < current_segment[0][0]:
                # decreasing open really the segment
                current_segment[1] = current_vals
            elif soc == current_segment[0][0]:
                # do nothing keep the segment open
                pass
            else:
                # upper than segment start ... segment no closed, start a new one
                current_segment = [current_vals, None, self._dec_seg_count]
                self._decreasing_segments[-1] = current_segment
                self._dec_seg_count += 1
        else:
            if soc <= current_segment[1][0]:
                # continue decreasing segment
                current_segment[1] = current_vals
            else:
                # close the current segment:
                new_segment = [current_vals, None, self._dec_seg_count]
                self._dec_seg_count += 1
                if current_segment[1][0] < current_segment[0][0] and current_segment[1][1] > current_segment[0][1]:
                    # good soc and odo values, keep it, add the new one
                    self._decreasing_segments.append(new_segment)

                    from_soc = current_segment[0][0]
                    to_soc = current_segment[1][0]

                    # first segment, just add it
                    delta_soc = float(from_soc - to_soc)
                    delta_km = float(current_segment[1][1] - current_segment[0][1])

                    to_be_stored_efficiency_segment = (delta_km, delta_soc, from_soc, to_soc, time)

                    if len(self._efficiency_segments) == 0 or time > self._efficiency_segments[-1][4]:
                        self._efficiency_segments.append(to_be_stored_efficiency_segment)
                    else:
                        # insert it in the time ordered list
                        idx = bisect.bisect_left(self._efficiency_segments, to_be_stored_efficiency_segment, key=itemgetter(4))
                        # always insert
                        self._efficiency_segments.insert(idx, to_be_stored_efficiency_segment)

                    if (time - self._efficiency_segments[0][4]).total_seconds() > CAR_MAX_EFFICIENCY_HISTORY_S:
                        self._efficiency_segments.pop(0)


                    # to play a bit with graphs with known soc deltas ... not sure it is really useful
                    # self._efficiency_deltas[(from_soc, to_soc)] = delta_km
                    # self._efficiency_deltas[(to_soc, from_soc)] = -delta_km
                    #
                    # fs = self._efficiency_deltas_graph.setdefault(from_soc, set())
                    # fs.add(to_soc)
                    # ts = self._efficiency_deltas_graph.setdefault(to_soc, set())
                    # ts.add(from_soc)


                else:
                    # bad segment, replace it with the new one
                    self._decreasing_segments[-1] = new_segment


    async def _async_bootstrap_efficiency_from_history(self, time: datetime):
        # pull last 14 days; compute efficiency only from segments where SOC decreases
        if self.hass is None:
            return
        if self.car_odometer_sensor is None or self.car_charge_percent_sensor is None:
            return
        if self.car_battery_capacity is None or self.car_battery_capacity <= 0:
            return

        start_time = time - timedelta(days=31)
        end_time = time

        def _load_hist(entity_id: str):
            return state_changes_during_period(
                self.hass, start_time, end_time, entity_id, include_start_time_state=True, no_attributes=True
            ).get(entity_id, [])

        try:
            odos = await recorder_get_instance(self.hass).async_add_executor_job(_load_hist, self.car_odometer_sensor)
            socs = await recorder_get_instance(self.hass).async_add_executor_job(_load_hist, self.car_charge_percent_sensor)
        except Exception:
            return

        if not odos or not socs:
            return

        # Build time series (time, value) with floats, ordered
        def _series(lst):
            res = []
            for s in lst:
                if s is None or s.state in [STATE_UNKNOWN, STATE_UNAVAILABLE, None, "None", "unknown", "unavailable"]:
                    continue
                try:
                    v = float(s.state)
                except (TypeError, ValueError):
                    continue
                res.append((s.last_changed, v))
            res.sort(key=lambda x: x[0])
            return res

        odo_series = _series(odos)
        soc_series = _series(socs)
        if len(odo_series) < 2 or len(soc_series) < 2:
            return

        # Helper to get odometer value at or before time (using bisect with key)
        def _odo_at(ts):
            idx = bisect.bisect_right(odo_series, ts, key=itemgetter(0)) - 1
            if idx < 0:
                return odo_series[0][1]
            if idx >= len(odo_series):
                return odo_series[-1][1]
            return odo_series[idx][1]

        total_energy_kwh = 0.0
        total_distance_km = 0.0
        cap_kwh = float(self.car_battery_capacity) / 1000.0

        self._decreasing_segments = []
        self._efficiency_segments = []

        # Iterate SOC segments; only count when SOC decreases

        for t, soc in soc_series:
            odo = _odo_at(t)
            self._add_soc_odo_value_to_segments(soc, odo, t)

        for d_seg in self._decreasing_segments:

            if d_seg[1] is None:
                continue

            soc0, odo0, _ = d_seg[0]
            soc1, odo1, _ = d_seg[1]

            delta_soc = float(soc0 - soc1)

            if delta_soc <= 0.0:
                continue

            if odo0 is None or odo1 is None:
                continue

            delta_km = float(odo1 - odo0)

            if delta_km <= 0.0:
                continue

            energy_kwh = (delta_soc / 100.0) * cap_kwh
            total_energy_kwh += energy_kwh
            total_distance_km += delta_km


        if total_energy_kwh <= 0.0 or total_distance_km <= 1.0:
            return

        eff = total_distance_km / total_energy_kwh
        # no clamping per user request
        self._km_per_kwh = eff

    def reset(self):
        super().reset()
        self.interpolate_power_steps(do_recompute_min_charge=True, use_conf_values=True)
        self._dampening_deltas = {}
        self._dampening_deltas_graph = {}
        self.calendar = self._conf_calendar
        self.charger = None
        self.do_force_next_charge = False
        self.do_next_charge_time= None
        self.user_attached_charger_name = None
        self._qs_bump_solar_priority = False
        if self.home:
            self.current_forecasted_person = self.home.get_preferred_person_for_car(self)
        else:
            self.current_forecasted_person = None


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

    def get_car_charge_energy(self, time: datetime, tolerance_seconds: float=4*3600) -> float | None:
        res = self.get_car_charge_percent(time, tolerance_seconds)
        if res is None:
            return None

        if self.car_battery_capacity is None or self.car_battery_capacity == 0:
            return None

        try:
            return float(res) * self.car_battery_capacity / 100.0
        except TypeError:
            return None

    def get_car_odometer_km(self, time: datetime | None = None, tolerance_seconds: float=24*3600 ) -> float | None:
        return self.get_sensor_latest_possible_valid_value(entity_id=self.car_odometer_sensor, time=time, tolerance_seconds=tolerance_seconds)

    def get_car_estimated_range_km_from_sensor(self, time: datetime | None = None, tolerance_seconds: float=24*3600 ) -> float | None:
        return self.get_sensor_latest_possible_valid_value(entity_id=self.car_estimated_range_sensor, time=time, tolerance_seconds=tolerance_seconds)

    def is_car_charge_growing(self,
                              num_seconds: float,
                              time: datetime) -> bool | None:

        return self.is_sensor_growing(entity_id=self.car_charge_percent_sensor,
                               num_seconds=num_seconds,
                              time=time)


    def _get_delta_from_graph(self, deltas, deltas_graph, from_v, to_v):

        delta = None

        if len(deltas) > 0:

            delta = deltas.get((from_v, to_v))

            if delta is None and len(deltas) > 2:
                # not direct try a path:
                path = self.find_path(deltas_graph, from_v, to_v)

                if path and len(path) > 1 and path[0] == from_v and path[-1] == to_v:
                    delta = 0
                    for i in range(1, len(path)):
                        d = deltas.get((path[i-1], path[i]))
                        if d is None:
                            _LOGGER.error(f"_get_delta_from_graph path in error: Car {self.name} deltas {deltas} graph {deltas_graph} from_v {from_v} to_v {to_v} path[i-1] {path[i-1]} path[i] {path[i]}")
                            delta = None
                            break

                        delta += d
                elif path:
                    _LOGGER.error(
                        f"_get_delta_from_graph path error: Car {self.name} deltas {deltas} graph {deltas_graph} from_v {from_v} to_v {to_v}")

        return delta



    def get_computed_range_efficiency_km_per_percent(self, time:datetime, delta_soc:float=0.0) -> float | None:

        if time is None:
            time = datetime.now(pytz.UTC)

        current_soc = self.get_car_charge_percent(time)
        car_estimate = self.get_car_estimated_range_km_from_sensor(time)

        if current_soc is not None and car_estimate is not None and current_soc > 0.0:
            return car_estimate / float(current_soc)

        best_segment = None
        # from the more recent to the older
        for i in range(len(self._efficiency_segments) - 1, -1, -1):
            seg = self._efficiency_segments[i]
            if (time - seg[4]).total_seconds() > CAR_MAX_EFFICIENCY_HISTORY_S:
                break

            if seg[0] == 0 or seg[1] == 0:
                continue

            if best_segment is None or (abs(seg[1] - delta_soc) < abs(best_segment[1] - delta_soc)):
                best_segment = seg

        if best_segment is not None:
            return (best_segment[0] / best_segment[1])

        if self._km_per_kwh is not None:
            return (self._km_per_kwh * (float(self.car_battery_capacity) / 1000.0)) / 100.0

        return None


    def get_adapt_target_percent_soc_to_reach_range_km(self, target_range_km: float | None, time: datetime | None = None) -> tuple[bool | None, float | None, float | None, float | None]:

        km_per_percent = current_soc = current_range_km = None

        if target_range_km is not None:
            target_range_km = target_range_km + CAR_MINIMUM_LEFT_RANGE_KM
            current_range_km = self.get_estimated_range_km(time)

            current_soc = self.get_car_charge_percent(time)
            km_per_percent = self.get_computed_range_efficiency_km_per_percent(time)

        if km_per_percent is None or current_soc is None or current_range_km is None or target_range_km is None:
            return None, None, None, None

        needed_soc = min(100.0, target_range_km / km_per_percent)

        diff_energy = (abs(needed_soc - current_soc)*self.car_battery_capacity)/100.0

        if current_range_km >= target_range_km:
            return True, current_soc, needed_soc, diff_energy
        else:
            return False, current_soc, needed_soc, diff_energy





    def get_car_estimated_range_km(self, from_soc=100.0, to_soc=0.0, time: datetime | None = None) -> float | None:

        # not really useful no?
        # graph_distance = self._get_delta_from_graph(self._efficiency_deltas, self._efficiency_deltas_graph, from_soc, to_soc)

        if time is None:
            time = datetime.now(pytz.UTC)

        delta_soc = abs(from_soc - to_soc)

        eff_perc = self.get_computed_range_efficiency_km_per_percent(time, delta_soc)

        result = None
        if eff_perc is not None:
            result = eff_perc * delta_soc

        return result


    def get_estimated_range_km(self, time: datetime | None = None) -> float | None:

        res = self.get_car_estimated_range_km_from_sensor(time)
        if res is not None:
            return res

        soc = self.get_car_charge_percent(time)
        if soc is None:
            return None
        return self.get_car_estimated_range_km(from_soc=soc, to_soc=0.0, time=time)


    def get_autonomy_to_target_soc_km(self, time: datetime | None = None) -> float | None:

        soc = self.get_car_target_SOC()
        if soc is None:
            return None
        return self.get_car_estimated_range_km(from_soc=soc, to_soc=0.0, time=time)


    async def adapt_max_charge_limit(self, asked_percent):

        if self.car_charge_percent_max_number is None:
            return

        percent= asked_percent
        # in fact stop the charge only at the "default charge" of the car else ... continue to charge
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

            try:
                await self.hass.services.async_call(
                    domain, service, data
                )
            except Exception as exc:
                _LOGGER.error(f"Car {self.name} failed to set max charge limit to {percent}%: {exc}")

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

        power = self._get_delta_from_graph(deltas=self._dampening_deltas, deltas_graph=self._dampening_deltas_graph, from_v=from_amp*from_num_phase, to_v=to_amp*to_num_phase)

        if power is None:
            from_power = self._get_power_from_stored_amps(from_amp, from_num_phase)
            to_power = self._get_power_from_stored_amps(to_amp, to_num_phase)

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


    async def add_default_charge_at_datetime(self, end_charge:datetime) -> bool:
        if self.can_add_default_charge() is False:
            return False
        self.do_next_charge_time = end_charge
        # start_time = end_charge
        # end_time = end_charge + timedelta(seconds=60*30)
        # time = datetime.now(pytz.UTC)
        # await self.set_next_scheduled_event(time, start_time, end_time, f"Charge {self.name}")
        return True


    async def add_default_charge_at_dt_time(self, default_charge_time:dt_time | None) -> bool:
        if self.can_add_default_charge() is False:
            return False

        if default_charge_time is None:
            _LOGGER.error(f"Car {self.name} cannot add default charge at None time")
            return False

        # compute the next occurrence of the default charge time
        next_time = self.get_next_time_from_hours(local_hours=default_charge_time, output_in_utc=True)

        return await self.add_default_charge_at_datetime(next_time)


    async def user_add_default_charge(self):
        if self.can_add_default_charge():

            res = await self.add_default_charge_at_dt_time(self.default_charge_time)

            if res and self.charger:
                self.do_force_next_charge = False
                await self.charger.update_charger_for_user_change()

    def can_add_default_charge(self) -> bool:
        if self.charger is not None:
            return True
        return False

    def can_force_a_charge_now(self) -> bool:
        if self.charger is not None:
            return True
        return False

    async def user_force_charge_now(self):
        if self.can_force_a_charge_now():
            self.do_force_next_charge = True
            self.do_next_charge_time = None
            if self.charger:
                await self.charger.update_charger_for_user_change()

    def can_use_charge_percent_constraints(self):

        if self.car_battery_capacity is None:
            return False
        if self.car_charge_percent_sensor is None:
            return False

        return True


    async def setup_car_charge_target_if_needed(self, asked_target_charge=None):

        target_charge = asked_target_charge

        if target_charge is None:
            target_charge = self.get_car_target_SOC()

        if target_charge is not None:
            await self.adapt_max_charge_limit(target_charge)

        return target_charge

    def get_car_next_charge_values_options(self):

        if self.can_use_charge_percent_constraints():
            return self.get_car_next_charge_values_options_percent()
        else:
            return self.get_car_next_charge_values_options_energy()

    async def set_next_charge_target(self, value:int|float|str):

        if self.can_use_charge_percent_constraints():
            await self.set_next_charge_target_percent(value)
        else:
            await self.set_next_charge_target_energy(value)

    def get_car_target_charge_option(self):
        if self.can_use_charge_percent_constraints():
            return self.get_car_target_charge_option_percent()
        else:
            return self.get_car_target_charge_option_energy()

    def get_car_next_charge_values_options_percent(self):
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
            options[i] = self.get_car_option_charge_from_value_percent(options[i])

        return options

    def get_car_option_charge_from_value_percent(self, value:int|float):
        value = int(float(value))
        if value > 100:
            value = 100

        if value == self.car_default_charge:
            return f"{value}% - {self.name} default"
        elif value == 100:
            return "100% - full"
        else:
            return f"{value}%"

    async def set_next_charge_target_percent(self, value:int|float|str, do_update_charger:bool=True):

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

        if do_update_charger and self.charger and new_target:
            await self.charger.update_charger_for_user_change()

    def get_car_target_charge_option_percent(self):
        return self.get_car_option_charge_from_value_percent(self.get_car_target_SOC())

    def get_car_target_SOC(self) -> int | float:
        if self._next_charge_target is None:
            self._next_charge_target = self.car_default_charge
        return self._next_charge_target

    def get_car_minimum_ok_SOC(self) -> int | float:
        return self.car_minimum_ok_charge


    def get_car_next_charge_values_options_energy(self):

        max_battery_energy = self.car_battery_capacity
        if max_battery_energy is None or max_battery_energy <= 0:
            max_battery_energy = CAR_DEFAULT_CAPACITY

        options = set()
        options.add(max_battery_energy)

        for v in range(0, max_battery_energy, 5000):
            options.add(v)

        options = list(options)
        options.sort()

        for i in range(len(options)):
            options[i] = self.get_car_option_charge_from_value_energy(options[i])

        return options

    def get_car_option_charge_from_value_energy(self, value:int|float):
        value = int(float(value))//1000 # kwh
        return f"{value}kWh"

    async def set_next_charge_target_energy(self, value:int|float|str):

        if isinstance(value, str):
            try:
                value = value.strip("kWh")
                value = float(value)*1000.0
            except ValueError:
                _LOGGER.error(f"Car {self.name} set_next_charge_target_energy: invalid value {value}")
                return
        else:
            value = float(value) * 1000.0

        self._next_charge_target_energy = value

        if self.charger:
            await self.charger.update_charger_for_user_change()

    def get_car_target_charge_energy(self) -> int | float:
        if self._next_charge_target_energy is None:
            if self.car_battery_capacity is not None and self.car_battery_capacity > 0:
                self._next_charge_target_energy = self.car_battery_capacity
            else:
                self._next_charge_target_energy = CAR_DEFAULT_CAPACITY
        return self._next_charge_target_energy

    def get_car_target_charge_option_energy(self):
        return self.get_car_option_charge_from_value_energy(self.get_car_target_charge_energy())


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
            if self.car_hard_wired_charger:
                if charger is not self.car_hard_wired_charger:
                    continue
            if charger.is_optimistic_plugged(time):
                options.append(charger.name)

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
        orig_charger = self.charger
        new_charger = None

        if self.charger is not None and self.charger.name != charger_name:
            self.detach_charger()

        if charger_name == FORCE_CAR_NO_CHARGER_CONNECTED:
            self.user_attached_charger_name = FORCE_CAR_NO_CHARGER_CONNECTED
        else:
            self.user_attached_charger_name = None

            if charger_name is not None:
                charger = None
                for c in self.home._chargers:
                    if c.name == charger_name:
                        charger = c
                        break

                if charger is not None:
                    # this one will update the charger
                    await charger.set_user_selected_car_by_name(car_name=self.name)

        if orig_charger is not None:
            await orig_charger.update_charger_for_user_change()

    async def user_clean_and_reset(self):
        charger = self.charger
        await super().user_clean_and_reset()

        self.user_attached_charger_name = None
        self.user_selected_person_name_for_car = None  # asked full reset, reset the user selected person,will trigger person allocation

        self.reset()  # will detach the car
        if charger is not None:
            await charger.user_clean_and_reset()


    async def user_clean_constraints(self):
        charger = self.charger
        await super().user_clean_constraints()
        if charger is not None:
            await charger.user_clean_constraints()


    @property
    def current_constraint_current_energy(self):
        if self.charger is None:
            return None
        return self.charger.current_constraint_current_energy