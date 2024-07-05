from abc import abstractmethod
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from operator import itemgetter

from homeassistant.core import HomeAssistant


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


def get_average_power(power_data:list[tuple[datetime, float]]):

    if len(power_data) == 0:
        return 0
    elif len(power_data) == 1:
        return power_data[0][0]

    nrj, dh = compute_energy_Wh_rieman_sum(power_data)
    return nrj / dh



def align_time_series_and_values(tsv1:list[tuple[datetime, float]], tsv2:list[tuple[datetime, float]] | None, do_sum:bool = False):


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
































class HADeviceMixin:

    def __init__(self, hass:HomeAssistant, **kwargs ):
        super().__init__(**kwargs)
        self.hass = hass
        self.historical_data : dict[str, list[tuple[datetime, float]]] = {}

    def add_to_history(self, entity_id:str, time:datetime, value:float):
        if entity_id not in self.historical_data:
            self.historical_data[entity_id] = []

        if len(self.historical_data[entity_id]) == 0 or self.historical_data[entity_id][-1][0] < time:
            self.historical_data[entity_id].append((time, value))
        else:
            self.historical_data[entity_id].append((time, value))
            self.historical_data[entity_id].sort(key=lambda x: x[0])

        if len(self.historical_data[entity_id]) > 200:
            self.historical_data[entity_id].pop(0)

    def get_history_data(self, entity_id:str, num_seconds_before:float, to_ts:datetime):
        hist_f = self.historical_data.get(entity_id, [])

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
    def get_platforms(self):
        """ returns associated platforms for this device """



