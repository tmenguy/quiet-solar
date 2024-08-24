from abc import abstractmethod
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from operator import itemgetter
from typing import Mapping, Any, Callable

import pytz
from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE, UnitOfPower, ATTR_UNIT_OF_MEASUREMENT
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, State, callback, Event, EventStateChangedData
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util.unit_conversion import PowerConverter

from ..const import CONF_ACCURATE_POWER_SENSOR, DOMAIN, DATA_HANDLER, COMMAND_BASED_POWER_SENSOR, \
    CONF_CALENDAR
from ..home_model.commands import CMD_OFF, CMD_IDLE
from ..home_model.load import AbstractLoad

import numpy as np


def compute_energy_Wh_rieman_sum(
        power_data: list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]] | list[tuple[datetime | None, str | float | None]],
        conservative: bool = False):
    """Compute energy from power with a rieman sum."""

    energy = 0
    duration_h = 0
    if power_data and len(power_data) > 1:

        # compute a rieman sum, as best as possible , trapezoidal, taking pessimistic asumption
        # as we don't want to artifically go up the previous one
        # (except in rare exceptions like reset, 0 , etc)

        for i in range(len(power_data) - 1):

            dt_h = float((power_data[i + 1][0] - power_data[i][0]).total_seconds()) / 3600.0
            duration_h += dt_h

            if conservative:
                d_p_w = 0
            else:
                d_p_w = abs(float(power_data[i + 1][1] - power_data[i][1]))

            d_nrj_wh = dt_h * (
                    min(power_data[i + 1][1], power_data[i][1]) + 0.5 * d_p_w
            )

            energy += d_nrj_wh

    return energy, duration_h


def convert_power_to_w(value: float, attributes: dict | None = None) -> float:
    default_unit: str = UnitOfPower.WATT
    if attributes is None:
        sensor_unit = default_unit
    else:
        sensor_unit = attributes.get(ATTR_UNIT_OF_MEASUREMENT, default_unit)

    if sensor_unit in UnitOfPower:
        value = PowerConverter.convert(value=value, from_unit=sensor_unit, to_unit=UnitOfPower.WATT)

    return value


def get_average_power_energy_based(
        power_data: list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]] | list[tuple[datetime | None, str | float | None]]):
    if len(power_data) == 0:
        return 0
    elif len(power_data) == 1:
        val = power_data[0][1]
        if val is None:
            return 0.0
    else:
        power_data = [(entry[0], entry[1]) for entry in power_data if entry[1] is not None]
        if power_data:
            nrj, dh = compute_energy_Wh_rieman_sum(power_data)
            val = nrj / dh
        else:
            return 0.0

    # do not change units
    return val


def get_average_sensor(sensor_data: list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]] | list[tuple[datetime | None, str | float | None]],
                       last_timing: datetime | None = None):
    if len(sensor_data) == 0:
        return 0
    elif len(sensor_data) == 1:
        val = sensor_data[0][1]
        if val is None:
            val = 0.0
    else:
        sum_time = 0
        sum_vals = 0
        add_last = 0
        if last_timing is not None:
            add_last = 1
        for i in range(1, len(sensor_data) + add_last):
            value = sensor_data[i - 1][1]
            if value is None:
                continue
            if i == len(sensor_data):
                dt = (last_timing - sensor_data[i - 1][0]).total_seconds()
            else:
                dt = (sensor_data[i][0] - sensor_data[i-1][0]).total_seconds()

            if dt == 0:
                dt = 1

            sum_time += dt
            sum_vals += dt*float(value)

        if sum_time > 0:
            return sum_vals / sum_time
        else:
            return 0.0
    # do not change units
    return val


def get_median_sensor(sensor_data: list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]] | list[tuple[datetime | None, str | float | None]],
                      last_timing: datetime | None = None):
    if len(sensor_data) == 0:
        return 0
    elif len(sensor_data) == 1:
        val = sensor_data[0][1]
        if val is None:
            val = 0.0
    else:
        vals : list[float] = []
        add_last = 0
        if last_timing is not None:
            add_last = 1
        for i in range(1, len(sensor_data) + add_last):
            value = sensor_data[i - 1][1]
            if value is None:
                continue
            if i == len(sensor_data):
                dt = (last_timing - sensor_data[i - 1][0]).total_seconds()
            else:
                dt = (sensor_data[i][0] - sensor_data[i-1][0]).total_seconds()

            if dt == 0:
                dt = 1

            num_add = int(dt) + 1
            vals.extend([float(value)]*num_add)

        if vals:
            val = np.median(vals)
        else:
            val = 0.0
    # do not change units
    return val

MAX_STATE_HISTORY_S = 7200

class HADeviceMixin:

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry, **kwargs):

        self.calendar = kwargs.pop(CONF_CALENDAR, None)
        self.accurate_power_sensor = kwargs.pop(CONF_ACCURATE_POWER_SENSOR, None)
        self.secondary_power_sensor = None
        self.best_power_value = None

        self.command_based_power_sensor = COMMAND_BASED_POWER_SENSOR

        super().__init__(**kwargs)
        self.hass = hass
        if hass:
            self.data_handler = hass.data[DOMAIN].get(DATA_HANDLER)
        else:
            self.data_handler = None

        self.config_entry = config_entry

        self._entity_probed_state_is_numerical: dict[str, bool] = {}
        self._entity_probed_state_conversion_fn: dict[str, Callable[[float, dict], float] | None] = {}
        self._entity_probed_state_transform_fn: dict[str, Callable[[float, dict], float] | None] = {}
        self._entity_probed_state_non_ha_entity_get_state: dict[str, Callable[[str, datetime | None], tuple[
                                                                                                          datetime | None, float | str | None, dict | None] | None] | None] = {}
        self._entity_probed_state: dict[
            str, list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]]] = {}
        self._entity_probed_last_valid_state: dict[
            str, tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict] | None] = {}
        self._entity_probed_auto = set()
        self._entity_on_change = set()

        self._exposed_entities = set()

        self.attach_power_to_probe(self.accurate_power_sensor)

        self.attach_power_to_probe(self.command_based_power_sensor,
                                   non_ha_entity_get_state=self.command_power_state_getter)

        self._unsub = None


    def get_next_scheduled_event(self, time:datetime) -> tuple[datetime|None,datetime|None]:
        if self.calendar is None:
            return None, None

        state = self.hass.states.get(self.calendar)
        state_attr = {}
        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            state = None

        if state is not None:
            state_attr = state.attributes

        return state_attr.get("start_time", None), state_attr.get("end_time", None)


    def attach_exposed_has_entity(self, ha_object):
        self._exposed_entities.add(ha_object)

    def command_power_state_getter(self, entity_id: str, time: datetime | None) -> (
            tuple[datetime | None, float | str | None, dict | None] | None):

        if not isinstance(self, AbstractLoad):
            return None

        command_value = None

        do_return_None = False
        if self.is_load_command_set(time) is False:
            do_return_None = True
        else:
            if self.current_command == CMD_OFF or self.current_command == CMD_IDLE:
                command_value = 0.0
            else:
                command_value = self.current_command.power_consign
                if command_value is None:
                    command_value = self.power_use
        value = None
        if self.accurate_power_sensor is not None:
            hist_f = self._entity_probed_state.get(self.accurate_power_sensor, [])
            if hist_f:
                value = hist_f[-1][1]
        elif self.secondary_power_sensor is not None:
            hist_f = self._entity_probed_state.get(self.secondary_power_sensor, [])
            if hist_f:
                value = hist_f[-1][1]
        else:
            value = command_value

        self.best_power_value = value

        if do_return_None:
            return None
        return (time, str(command_value), {})


    def get_virtual_load_HA_power_entity_name(self) -> str | None:
        if not isinstance(self, AbstractLoad):
            return None
        return f"{self.device_id}_power"

    def get_virtual_current_constraint_entity_name(self) -> str | None:
        if not isinstance(self, AbstractLoad):
            return None
        return f"{self.device_id}_constraint"


    def get_best_power_HA_entity(self):
        if self.accurate_power_sensor is not None:
            return self.accurate_power_sensor
        elif self.secondary_power_sensor is not None:
            return self.secondary_power_sensor
        else:
            return None


    def get_sensor_latest_possible_valid_value(self, entity_id, tolerance_seconds: float | None = None,
                                               time = None) -> str | float | None:
        if entity_id is None:
            return None

        last_valid = self._entity_probed_last_valid_state[entity_id]
        if last_valid is None:
            return None

        if time is None or time >= last_valid[0]:

            if tolerance_seconds is None or tolerance_seconds == 0:
                return last_valid[1]

            if time is not None and (time - last_valid[0]).total_seconds() > tolerance_seconds:
                return None

            return last_valid[1]
        else:
            vals = self.get_state_history_data(entity_id, tolerance_seconds, time)
            if not vals:
                return None
            return vals[-1][1]

    def get_device_power_latest_possible_valid_value(self, tolerance_seconds: float | None, time) -> float | None:
        val = self.get_sensor_latest_possible_valid_value(self.accurate_power_sensor, tolerance_seconds, time)
        if not val:
            val = self.get_sensor_latest_possible_valid_value(self.secondary_power_sensor, tolerance_seconds, time)
        if not val:
            val = self.get_sensor_latest_possible_valid_value(self.command_based_power_sensor, tolerance_seconds, time)
        return val

    def get_median_sensor(self, entity_id: str | None, num_seconds: float | None, time: datetime) -> float | None:
        if entity_id is None:
            return None
        entity_id_values = self.get_state_history_data(entity_id, num_seconds, time)
        if not entity_id_values:
            return None
        return get_median_sensor(entity_id_values, time)

    def get_average_sensor(self, entity_id: str | None, num_seconds: float | None, time: datetime) -> float | None:
        if entity_id is None:
            return None
        entity_id_values = self.get_state_history_data(entity_id, num_seconds, time)
        if not entity_id_values:
            return None
        return get_average_sensor(entity_id_values, time)

    def get_median_power(self, num_seconds: float | None, time) -> float | None:
        val = self.get_median_sensor(self.accurate_power_sensor, num_seconds, time)
        if not val:
            val = self.get_median_sensor(self.secondary_power_sensor, num_seconds, time)
        if not val:
            val = self.get_median_sensor(self.command_based_power_sensor, num_seconds, time)
        return val

    def get_average_power(self, num_seconds: float | None, time) -> float | None:
        val = self.get_average_sensor(self.accurate_power_sensor, num_seconds, time)
        if not val:
            val = self.get_average_sensor(self.secondary_power_sensor, num_seconds, time)
        if not val:
            val = self.get_average_sensor(self.command_based_power_sensor, num_seconds, time)
        return val

    def get_device_power_values(self, duration_before_s: float, time: datetime) -> list[
        tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]]:
        val = self.get_state_history_data(self.accurate_power_sensor, duration_before_s, time)
        if not val:
            val = self.get_state_history_data(self.secondary_power_sensor, duration_before_s, time)
        if not val:
            val = self.get_state_history_data(self.command_based_power_sensor, duration_before_s, time)
        return val

    def get_last_state_value_duration(self, entity_id: str, states_vals: list[str], num_seconds_before: float | None,
                                      time: datetime, invert_val_probe=False, allowed_max_holes_s: float = 3) -> float:

        states_vals = set(states_vals)
        if entity_id in self._entity_probed_state:
            # get latest values
            self.add_to_history(entity_id, time)

        values = self.get_state_history_data(entity_id, num_seconds_before=num_seconds_before, to_ts=time,
                                             keep_invalid_states=True)

        if not values:
            return 0.0

        values.sort(key=itemgetter(0), reverse=True)

        state_status_duration = 0

        # check the last states
        current_hole = 0.0
        first_is_met = False

        for i, (ts, state, attr) in enumerate(values):
            if i > 0:
                next_ts = values[i - 1][0]
            else:
                next_ts = time

            delta_t = (next_ts - ts).total_seconds()

            val_prob_ok = False
            if state is not None:
                if invert_val_probe:
                    val_prob_ok = state not in states_vals
                else:
                    val_prob_ok = state in states_vals

            if val_prob_ok:
                state_status_duration += delta_t
                current_hole = 0.0
                first_is_met = True
            else:
                if state is not None and first_is_met is False:
                    # if we have an incompatible non None state (ie not unknown or not availabe) first we do not count anything
                    # but we could start with a small unavailable hole of a None State
                    break
                current_hole += delta_t
                if current_hole > allowed_max_holes_s:
                    break

        return state_status_duration

    def register_all_on_change_states(self):

        if len(self._entity_on_change) > 0:
            @callback
            def async_threshold_sensor_state_listener(
                    event: Event[EventStateChangedData],
            ) -> None:
                """Handle sensor state changes."""
                new_state = event.data["new_state"]
                time = new_state.last_updated
                self.add_to_history(new_state.entity_id, time, state=new_state)

            self._unsub = async_track_state_change_event(
                self.hass,
                list(self._entity_on_change),
                async_threshold_sensor_state_listener,
            )

    async def update_states(self, time: datetime):

        for entity_id in self._entity_probed_auto:
            self.add_to_history(entity_id, time)

        for ha_object in self._exposed_entities:
            ha_object.async_update_callback(time)

    def attach_power_to_probe(self, entity_id: str | None, transform_fn: Callable[[float, dict], float] | None = None,
                              non_ha_entity_get_state: Callable[[str, datetime | None], tuple[
                                                                                            float | str | None, datetime | None, dict | None] | None] = None):
        self.attach_ha_state_to_probe(entity_id=entity_id, is_numerical=True, transform_fn=transform_fn,
                                      conversion_fn=convert_power_to_w, update_on_change_only=True,
                                      non_ha_entity_get_state=non_ha_entity_get_state)

    def attach_ha_state_to_probe(self, entity_id: str | None, is_numerical: bool = False,
                                 transform_fn: Callable[[float, dict], float] | None = None,
                                 conversion_fn: Callable[[float, dict], float] | None = None,
                                 update_on_change_only: bool = True,
                                 non_ha_entity_get_state: Callable[[str, datetime | None], tuple[
                                                                                               float | str | None, datetime | None, dict | None] | None] = None):
        if entity_id is None:
            return

        self._entity_probed_state[entity_id] = []
        self._entity_probed_last_valid_state[entity_id] = None
        self._entity_probed_state_is_numerical[entity_id] = is_numerical
        self._entity_probed_state_transform_fn[entity_id] = transform_fn
        self._entity_probed_state_conversion_fn[entity_id] = conversion_fn
        self._entity_probed_state_non_ha_entity_get_state[entity_id] = non_ha_entity_get_state

        if non_ha_entity_get_state is not None:
            self._entity_probed_auto.add(entity_id)
        else:
            if update_on_change_only:
                self._entity_on_change.add(entity_id)
            else:
                self._entity_on_change.add(entity_id)
                self._entity_probed_auto.add(entity_id)

        # store a first version
        if self.hass:
            self.add_to_history(entity_id)

    def _clean_times_arrays(self, current_time: datetime, time_array: list[datetime], value_arrays: list[list]) -> list[
        datetime]:
        if len(time_array) == 0:
            return time_array

        while len(time_array) > 0 and (current_time - time_array[0]).total_seconds() > MAX_STATE_HISTORY_S:
            time_array.pop(0)
            for v in value_arrays:
                v.pop(0)

        return time_array

    def add_to_history(self, entity_id: str, time: datetime = None, state: State = None):

        state_getter = self._entity_probed_state_non_ha_entity_get_state[entity_id]
        state_time: datetime | None = None

        if state is not None or state_getter is None:
            if state is None:
                state = self.hass.states.get(entity_id)
            state_attr = {}
            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                value = None
                if state is not None:
                    state_time = state.last_updated
            else:
                value = state.state
                state_time = state.last_updated

            if state is not None:
                state_attr = state.attributes
        else:
            fake_state = state_getter(entity_id, time)
            if fake_state is not None:
                state_time, value, state_attr = fake_state
            else:
                value = None
                state_attr = {}

        if state_attr is None:
            state_attr = {}

        if state_time is None:
            if time is None:
                state_time = datetime.now(tz=pytz.UTC)
            else:
                state_time = time

        if self._entity_probed_state_is_numerical[entity_id]:
            try:
                value = float(value)
            except ValueError:
                value = None
            except TypeError:
                value = None

        if value is not None:
            conversion_fn = self._entity_probed_state_conversion_fn[entity_id]
            if conversion_fn is not None:
                value = conversion_fn(value, state_attr)
            transform_fn = self._entity_probed_state_transform_fn[entity_id]
            if transform_fn is not None:
                value = transform_fn(value, state_attr)

        val_array = self._entity_probed_state[entity_id]

        if state is None:
            to_add = (state_time, value, None)
        else:
            to_add = (state_time, value, state.attributes)

        if value is not None:
            prev_valid = self._entity_probed_last_valid_state[entity_id]
            if prev_valid is None or prev_valid[0] <= state_time:
                self._entity_probed_last_valid_state[entity_id] = to_add

        if not val_array:
            val_array.append(to_add)
        else:
            # small optim to add additional values at the end
            if state_time > val_array[-1][0]:
                val_array.append(to_add)
            elif state_time == val_array[-1][0]:
                val_array[-1] = to_add
            else:
                insert_idx = bisect_left(val_array, state_time, key=itemgetter(0))
                if insert_idx == len(val_array):
                    val_array.append(to_add)
                else:
                    if val_array[insert_idx][0] == state_time:
                        val_array[insert_idx] = to_add
                    else:
                        val_array.insert(insert_idx, to_add)

        while len(val_array) > 1 and (val_array[-1][0] - val_array[0][0]).total_seconds() > MAX_STATE_HISTORY_S:
            val_array.pop(0)

    def get_state_history_data(self, entity_id: str, num_seconds_before: float | None, to_ts: datetime,
                               keep_invalid_states=False) -> list[
        tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]]:
        hist_f = self._entity_probed_state.get(entity_id, [])

        if not hist_f:
            return []

        if to_ts is None:
            to_ts = datetime.now(tz=pytz.UTC)

        # the last value is in fact still valid now (by construction : either it was through polling or through a state change)

        if num_seconds_before is None:
            from_ts = hist_f[0][0] - timedelta(seconds=10)
        else:
            from_ts = to_ts - timedelta(seconds=num_seconds_before)

        ret = None

        if from_ts >= hist_f[-1][0]:
            ret = hist_f[-1:]
        elif to_ts < hist_f[0][0]:
            ret = []
        elif to_ts == hist_f[0][0]:
            ret = hist_f[:1]
        else:

            in_s = bisect_left(hist_f, from_ts, key=itemgetter(0))

            # the state is "valid" for its whole duration so pick the one before
            if in_s > 0:
                if hist_f[in_s][0] > from_ts:
                    in_s -= 1

            if to_ts >= hist_f[-1][0]:
                out_s = len(hist_f)
            else:
                out_s = bisect_right(hist_f, to_ts, key=itemgetter(0))

            if in_s == out_s:
                if out_s == len(hist_f):
                    ret = hist_f[-1:]

            if ret is None:
                ret = hist_f[in_s:out_s]

        if ret is None:
            ret = []

        if keep_invalid_states is False and ret:
            return [v for v in ret if v[1] is not None]
        else:
            return ret

    @abstractmethod
    def get_platforms(self) -> list[str]:
        """ returns associated platforms for this device """
