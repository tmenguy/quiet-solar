from abc import abstractmethod
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from enum import Enum
from operator import itemgetter
from typing import Mapping, Any

from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, State, callback, Event, EventStateChangedData
from homeassistant.helpers.event import async_track_state_change_event

from quiet_solar.const import CONF_POWER_SENSOR, DOMAIN, DATA_HANDLER

import numpy as np
import numpy.typing as npt

def compute_energy_Wh_rieman_sum(power_data, conservative: bool = False):
    """Compute energy from power with a rieman sum."""

    energy = 0
    duration_h = 0
    if power_data and len(power_data) > 1:

        # compute a rieman sum, as best as possible , trapezoidal, taking pessimistic asumption
        # as we don't want to artifically go up the previous one
        # (except in rare exceptions like reset, 0 , etc)

        for i in range(len(power_data) - 1):

            dt_h = float(power_data[i + 1][0] - power_data[i][0]) / 3600.0
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


def get_average_power(power_data: list[tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]]):

    if len(power_data) == 0:
        return 0
    elif len(power_data) == 1:
        return power_data[0][0]

    nrj, dh = compute_energy_Wh_rieman_sum(power_data)
    return nrj / dh


def get_median_power(power_data: list[tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]]):

    if len(power_data) == 0:
        return 0
    elif len(power_data) == 1:
        return power_data[0][0]

    return np.median([float(v) for _, v, _ in power_data])


def align_time_series_and_values(tsv1:list[tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]], tsv2:list[tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]] | None, do_sum:bool = False):


    if tsv2 is None or len(tsv2) == 0:
        if do_sum:
            return tsv1
        return tsv1, [(0,t) for t, _ in tsv1]


    timings= {}

    for i, tv in enumerate(tsv1):
        timings[tv[0]] = [i, None]
    for i, tv in enumerate(tsv2):
        if tv[0] in timings:
            timings[tv[0]][1] = i
        timings[tv[0]] = [None, i]

    timings = [(k, v) for k, v in timings.items()]
    timings.sort(key=lambda x: x[0])
    t_only = [t for t, _ in timings]


    #compute all values for each time
    new_v1  = [0]*len(t_only)
    new_v2 = [0]*len(t_only)

    for vi in range(2):

        new_v = new_v1
        tsv = tsv1
        if vi > 0:
            if do_sum is False:
                new_v = new_v2
            tsv = tsv2

        last_real_idx = None
        for i, t, idxs in enumerate(timings):

            if idxs[vi] is not None:
                #ok an exact value
                last_real_idx = idxs[vi]
                new_v[i] += (tsv[last_real_idx][1])
            else:
                if last_real_idx is None:
                    #we have new values "before" the first real value"
                    new_v[i] += (tsv[0][1])
                elif last_real_idx  == len(tsv) - 1:
                    #we have new values "after" the last real value"
                    new_v[i] += (tsv[-1][1])
                else:
                    # we have new values "between" two real values"
                    # interpolate
                    d1 = float((t - tsv[last_real_idx][0]).total_seconds())
                    d2 = float((tsv[last_real_idx + 1][0] - tsv[last_real_idx][0]).total_seconds())
                    nv = (d1 / d2) * (tsv[last_real_idx + 1][1] - tsv[last_real_idx][1]) + tsv[last_real_idx][1]
                    new_v[i] += (nv)

    #ok so we do have values and timings for 1 and 2
    if do_sum:
        return zip(t_only, new_v1)

    return zip(t_only, new_v1), zip(t_only, new_v2)


MAX_STATE_HISTORY_S = 3600

class StoredStateStatus(Enum):
    
    VALID = "valid"
    ALL = "all"

class HADeviceMixin:

    def __init__(self, hass:HomeAssistant, config_entry:ConfigEntry, **kwargs):
        super().__init__(**kwargs)
        self.hass = hass
        self.data_handler = hass.data[DOMAIN].get(DATA_HANDLER)
        self.config_entry = config_entry

        self.power_sensor = kwargs.pop(CONF_POWER_SENSOR, None)

        self._entity_probed_state_is_numerical : dict[str, bool] = {}
        self._entity_probed_state_transform_fn: dict[str, Any | None] = {}
        self._entity_probed_state : dict[str, dict[StoredStateStatus, list[tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]] ]] = {}
        self._entity_probed_auto = set()
        self._entity_on_change = set()
        self.attach_ha_state_to_probe(self.power_sensor, is_numerical=True, update_on_change_only=True)
        self._unsub = None

    def get_last_state_value_duration(self, entity_id: str, states_vals:list[str], num_seconds_before, time:datetime) -> float:

        states_vals = set(states_vals)
        values = self.get_state_history_data(entity_id, num_seconds_before=num_seconds_before, to_ts=time)

        values.sort(key=itemgetter(0), reverse=True)

        state_status_duration = 0

        # check the last states
        for i, (ts, state, attr) in enumerate(values):
            if i > 0:
                next_ts = values[i - 1][0]
            else:
                next_ts = time

            if state in states_vals:
                values += (next_ts - ts).total_seconds()
            else:
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

    def update_states(self, time:datetime):

        for entity_id in self._entity_probed_auto:
            self.add_to_history(entity_id, time)


    def attach_ha_state_to_probe(self, entity_id:str|None, is_numerical:bool=False, transform_fn=None, update_on_change_only:bool=False):
        if entity_id is None:
            return

        self._entity_probed_state[entity_id] = {StoredStateStatus.ALL: [], StoredStateStatus.VALID: []}
        self._entity_probed_state_is_numerical[entity_id] = is_numerical
        self._entity_probed_state_transform_fn[entity_id] = transform_fn

        if update_on_change_only:
            self._entity_on_change.add(entity_id)
        else:
            self._entity_probed_auto.add(entity_id)


    def _clean_times_arrays(self, current_time:datetime, time_array : list[datetime], value_arrays : list[list]) -> list[datetime]:
        if len(time_array) == 0:
            return time_array

        while len(time_array) > 0 and (current_time - time_array[0]).total_seconds() > MAX_STATE_HISTORY_S:
            time_array.pop(0)
            for v in value_arrays:
                v.pop(0)

        return time_array

    def add_to_history(self, entity_id:str, time:datetime, state:State = None):

        if state is None:
            state = self.hass.states.get(entity_id)


        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            value = None
        else:
            value = state.state

        if self._entity_probed_state_is_numerical[entity_id]:
            try:
                value = float(value)
            except ValueError:
                value = None
            except TypeError:
                value = None

        if value is None:
            to_update = [StoredStateStatus.ALL]
        else:
            transform_fn = self._entity_probed_state_transform_fn[entity_id]
            if transform_fn is not None:
                value = transform_fn(value)

            to_update = [StoredStateStatus.ALL, StoredStateStatus.VALID]


        for to_change in to_update:
            val_array = self._entity_probed_state[entity_id][to_change]
            val_array.append((time, value, state.attributes))
            val_array.sort(key=itemgetter(0))
            while len(val_array) > 0 and (val_array[-1][0] - val_array[0][0]).total_seconds() > MAX_STATE_HISTORY_S:
                val_array.pop(0)


    def _get_last_state_value(self, entity_id:str, valid_type:StoredStateStatus) -> tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]:

        if len(self._entity_probed_state[entity_id][valid_type]) == 0:
            return None, None, None

        return self._entity_probed_state[entity_id][valid_type][-1]



    def get_current_state_value(self, entity_id:str) -> tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]:
        return self._get_last_state_value(entity_id, StoredStateStatus.ALL)

    def get_last_valid_state_value(self, entity_id: str) -> tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]:
        return self._get_last_state_value(entity_id, StoredStateStatus.VALID)

    def get_state_history_data(self, entity_id:str, num_seconds_before:float, to_ts:datetime, valid_type:StoredStateStatus=StoredStateStatus.VALID) -> list[tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]]:
        hist_f = self._entity_probed_state.get(entity_id, {}).get(valid_type, [])

        if not hist_f:
            return []

        from_ts = to_ts - timedelta(seconds=num_seconds_before)

        in_s = bisect_left(hist_f, from_ts, key=itemgetter(0))

        if to_ts is None or to_ts > hist_f[-1][0]:
            out_s = len(hist_f)
        else:
            out_s = bisect_right(hist_f, to_ts, key=itemgetter(0))

        return hist_f[in_s:out_s]




    @abstractmethod
    def get_platforms(self) -> list[str]:
        """ returns associated platforms for this device """

