import copy
from datetime import datetime
from typing import Mapping, Any


def is_amps_zero(amps: list[float | int]) -> bool:
    if amps is None:
        return True

    for a in amps:
        if a != 0.0:
            return False

    return True


def are_amps_equal(left_amps: list[float | int], right_amps: list[float | int]) -> bool:
    for i in [0,1,2]:
        if left_amps[i] != right_amps[i]:
            return False
    return True


def is_amps_greater(left_amps: list[float | int], right_amps: list[float | int]):
    for i in range(3):
        if left_amps[i] > right_amps[i]:
            return True
    return False


def add_amps(left_amps: list[float | int], right_amps: list[float | int]) -> list[float | int]:
    if left_amps is None and right_amps is None:
        return [0.0, 0.0, 0.0]
    elif left_amps is None:
        return copy.copy(right_amps)
    elif right_amps is None:
        return copy.copy(left_amps)

    adds = [left_amps[i] + right_amps[i] for i in range(3)]
    return adds


def diff_amps(left_amps: list[float | int], right_amps: list[float | int]) -> list[float | int]:
    if left_amps is None or right_amps is None:
        return [0.0, 0.0, 0.0]

    diff = [left_amps[i] - right_amps[i] for i in range(3)]
    return diff


def min_amps(left_amps: list[float | int], right_amps: list[float | int]) -> list[float | int]:
    if left_amps is None and right_amps is None:
        return [0.0, 0.0, 0.0]
    elif left_amps is None:
        return copy.copy(right_amps)
    elif right_amps is None:
        return copy.copy(left_amps)

    mins = [min(left_amps[i], right_amps[i]) for i in range(3)]
    return mins


def max_amps(left_amps: list[float | int], right_amps: list[float | int]) -> list[float | int]:
    if left_amps is None and right_amps is None:
        return [0.0, 0.0, 0.0]
    elif left_amps is None:
        return copy.copy(right_amps)
    elif right_amps is None:
        return copy.copy(left_amps)

    maxs = [max(left_amps[i], right_amps[i]) for i in range(3)]
    return maxs

def get_average_time_series(sensor_data: list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]] | list[tuple[datetime | None, str | float | None]],
                       first_timing: datetime | None = None, last_timing: datetime | None = None, geometric_mean : bool = False,):

    # remove None values, will actually use None as gaps, do the mean of the data we know
    if geometric_mean:
        sensor_data = [x for x in sensor_data if x[1] is not None and x[0] is not None]

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
        first_idx = 1
        if first_timing is not None:
            first_idx = 0
        for i in range(first_idx, len(sensor_data) + add_last):
            if i == 0:
                value = sensor_data[0][1]
            else:
                value = sensor_data[i - 1][1]

            if value is None:
                continue

            if i == len(sensor_data):
                dt = (last_timing - sensor_data[i - 1][0]).total_seconds()
            elif i == 0:
                dt = (sensor_data[i][0] - first_timing).total_seconds()
            else:
                dt = (sensor_data[i][0] - sensor_data[i-1][0]).total_seconds()
                if geometric_mean:
                    value += sensor_data[i][1]
                    value /= 2.0

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
