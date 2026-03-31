import copy
from bisect import bisect_left
from collections.abc import Callable, Mapping
from datetime import datetime
from operator import itemgetter
from typing import Any

import numpy as np


def is_amps_zero(amps: list[float | int]) -> bool:
    if amps is None:
        return True

    for a in amps:
        if a != 0.0:
            return False

    return True


def are_amps_equal(left_amps: list[float | int], right_amps: list[float | int]) -> bool:
    for i in [0, 1, 2]:
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


def get_average_time_series(
    sensor_data: list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]]
    | list[tuple[datetime | None, str | float | None]],
    first_timing: datetime | None = None,
    last_timing: datetime | None = None,
    geometric_mean: bool = False,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:

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

            if min_val is not None and value < min_val:
                continue

            if max_val is not None and value > max_val:
                continue

            if i == len(sensor_data):
                dt = (last_timing - sensor_data[i - 1][0]).total_seconds()
            elif i == 0:
                dt = (sensor_data[i][0] - first_timing).total_seconds()
            else:
                dt = (sensor_data[i][0] - sensor_data[i - 1][0]).total_seconds()
                if geometric_mean:
                    value += sensor_data[i][1]
                    value /= 2.0

            if dt == 0:
                dt = 1

            sum_time += dt
            sum_vals += dt * float(value)

        if sum_time > 0:
            return sum_vals / sum_time
        else:
            return 0.0
    # do not change units
    return val


def hungarian_algorithm(cost_matrix: np.ndarray) -> dict[int, int]:
    """
    Pure numpy implementation of the Hungarian algorithm for minimum cost assignment.

    Args:
        cost_matrix: n×n cost matrix (will be padded if rectangular)

    Returns:
        Dictionary mapping row indices to column indices (only original rows)
    """
    # Ensure square matrix
    n_rows, n_cols = cost_matrix.shape
    n = max(n_rows, n_cols)
    original_n_rows = n_rows
    original_n_cols = n_cols

    if n_rows != n or n_cols != n:
        padded = np.full((n, n), np.amax(cost_matrix) + 1e6)
        padded[:n_rows, :n_cols] = cost_matrix
        cost_matrix = padded

    # Step 1: Subtract row minimums
    cost_matrix = cost_matrix - np.amin(cost_matrix, axis=1, keepdims=True)

    # Step 2: Subtract column minimums
    cost_matrix = cost_matrix - np.amin(cost_matrix, axis=0, keepdims=True)

    # Step 3-6: Iterative covering and augmenting
    max_iterations = n * n
    for _ in range(max_iterations):
        # Find assignment using minimum number of lines
        assignment = _try_assign(cost_matrix)

        if assignment is not None:
            # Only return assignments for original rows and columns
            result = {}
            for i in range(original_n_rows):
                if assignment[i] >= 0 and assignment[i] < original_n_cols:
                    result[i] = int(assignment[i])
            return result

        # Find minimum uncovered value
        covered_rows, covered_cols = _find_minimum_cover(cost_matrix)

        # Get uncovered elements
        uncovered_mask = ~covered_rows[:, None] & ~covered_cols[None, :]
        uncovered_values = cost_matrix[uncovered_mask]

        if len(uncovered_values) == 0:
            # All elements are covered, force break
            break

        min_uncovered = np.amin(uncovered_values)

        # Adjust matrix
        cost_matrix[~covered_rows[:, None] & ~covered_cols[None, :]] -= min_uncovered
        cost_matrix[covered_rows[:, None] & covered_cols[None, :]] += min_uncovered

    # Fallback greedy assignment
    full_assignment = _greedy_assignment(cost_matrix)
    # Filter to original dimensions
    return {
        i: full_assignment[i]
        for i in range(original_n_rows)
        if i in full_assignment and full_assignment[i] < original_n_cols
    }


def _try_assign(cost_matrix: np.ndarray) -> np.ndarray | None:
    """Try to find a complete assignment using zeros in the cost matrix."""
    n = len(cost_matrix)
    zero_positions = np.argwhere(cost_matrix == 0)

    if len(zero_positions) == 0:
        return None

    # Try to assign greedily
    assignment = np.full(n, -1, dtype=int)
    assigned_cols = set()

    # Sort by row (prefer rows with fewer zeros)
    row_zero_counts = np.bincount(zero_positions[:, 0], minlength=n)
    sorted_rows = np.argsort(row_zero_counts)

    for row in sorted_rows:
        if row_zero_counts[row] == 0:
            continue
        available_cols = zero_positions[zero_positions[:, 0] == row, 1]
        available_cols = available_cols[~np.isin(available_cols, list(assigned_cols))]

        if len(available_cols) > 0:
            col = available_cols[0]
            assignment[row] = col
            assigned_cols.add(col)

    if np.all(assignment >= 0):
        return assignment

    # Try augmenting paths for unassigned rows
    for row in range(n):
        if assignment[row] == -1:
            if not _augment(row, cost_matrix, assignment, assigned_cols):
                return None

    return assignment if np.all(assignment >= 0) else None


def _augment(row: int, cost_matrix: np.ndarray, assignment: np.ndarray, assigned_cols: set) -> bool:
    """Try to find an augmenting path for an unassigned row."""
    n = len(cost_matrix)
    visited = set()

    def dfs(r: int) -> bool:
        zero_cols = np.where(cost_matrix[r] == 0)[0]

        for c in zero_cols:
            if c in visited:
                continue
            visited.add(c)

            # If column unassigned, we found an augmenting path
            if c not in assigned_cols:
                assignment[r] = c
                assigned_cols.add(c)
                return True

            # Try to reassign the row currently assigned to this column
            conflicting_row = np.where(assignment == c)[0][0]
            if dfs(conflicting_row):
                assignment[r] = c
                return True

        return False

    return dfs(row)


def _find_minimum_cover(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find minimum line cover of zeros using König's theorem.

    Builds a maximum partial matching from zeros (greedy + augmentation)
    and then applies König's theorem via BFS from unassigned rows.
    Unlike _try_assign, partial results are kept even when a perfect
    matching does not exist, which is required for the cover to be correct.
    """
    n = len(cost_matrix)

    zero_positions = np.argwhere(cost_matrix == 0)
    assignment = np.full(n, -1, dtype=int)
    assigned_cols: set[int] = set()

    if len(zero_positions) > 0:
        row_zero_counts = np.bincount(zero_positions[:, 0], minlength=n)
        sorted_rows = np.argsort(row_zero_counts)

        for row in sorted_rows:
            if row_zero_counts[row] == 0:
                continue
            available_cols = zero_positions[zero_positions[:, 0] == row, 1]
            available_cols = available_cols[~np.isin(available_cols, list(assigned_cols))]
            if len(available_cols) > 0:
                assignment[row] = available_cols[0]
                assigned_cols.add(int(available_cols[0]))

        for row in range(n):
            if assignment[row] == -1:
                _augment(row, cost_matrix, assignment, assigned_cols)

    # Find unassigned rows
    unassigned_rows = np.where(assignment == -1)[0]

    # BFS/DFS to find alternating paths
    marked_rows = np.zeros(n, dtype=bool)
    marked_cols = np.zeros(n, dtype=bool)

    queue = list(unassigned_rows)
    marked_rows[unassigned_rows] = True

    while queue:
        row = queue.pop(0)
        zero_cols = np.where(cost_matrix[row] == 0)[0]

        for col in zero_cols:
            if not marked_cols[col]:
                marked_cols[col] = True

                # Find row assigned to this column
                assigned_row = np.where(assignment == col)[0]
                if len(assigned_row) > 0:
                    assigned_row = assigned_row[0]
                    if not marked_rows[assigned_row]:
                        marked_rows[assigned_row] = True
                        queue.append(assigned_row)

    # Cover: unmarked rows and marked columns
    covered_rows = ~marked_rows
    covered_cols = marked_cols

    return covered_rows, covered_cols


def _greedy_assignment(cost_matrix: np.ndarray) -> dict[int, int]:
    """Greedy fallback assignment."""
    n = len(cost_matrix)
    assignment = {}
    assigned_cols = set()

    for row in range(n):
        available = [c for c in range(n) if c not in assigned_cols]
        if available:
            col = min(available, key=lambda c: cost_matrix[row, c])
            assignment[row] = col
            assigned_cols.add(col)

    return assignment


def align_time_series_and_values(
    tsv1: list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]]
    | list[tuple[datetime | None, str | float | None]],
    tsv2: list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]]
    | list[tuple[datetime | None, str | float | None]]
    | None,
    operation: Callable[[Any, Any], Any] | None = None,
):

    if not tsv1:
        if not tsv2:
            if operation is not None:
                return []
            else:
                return [], []
        else:
            if operation is not None:
                if len(tsv2[0]) == 3:
                    return [(t, operation(None, v), a) for t, v, a in tsv2]
                else:
                    return [(t, operation(None, v)) for t, v in tsv2]
            else:
                if len(tsv2[0]) == 3:
                    return [(t, None, None) for t, _, _ in tsv2], tsv2
                else:
                    return [(t, None) for t, _ in tsv2], tsv2

    if not tsv2:
        if operation is not None:
            if len(tsv1[0]) == 3:
                return [(t, operation(v, None), a) for t, v, a in tsv1]
            else:
                return [(t, operation(v, None)) for t, v in tsv1]
        else:
            if len(tsv1[0]) == 3:
                return tsv1, [(t, None, None) for t, _, _ in tsv1]
            else:
                return tsv1, [(t, None) for t, _ in tsv1]

    timings = {}

    for i, tv in enumerate(tsv1):
        timings[tv[0]] = [i, None]
    for i, tv in enumerate(tsv2):
        if tv[0] in timings:
            timings[tv[0]][1] = i
        else:
            timings[tv[0]] = [None, i]

    sorted_timings = sorted(timings.items(), key=lambda x: x[0])
    t_only = [t for t, _ in sorted_timings]

    object_len = 3
    object_len = min(object_len, len(tsv1[0]), len(tsv2[0]))

    # compute all values for each time
    new_v1: list[float | str | None] = [0] * len(t_only)
    new_v2: list[float | str | None] = [0] * len(t_only)

    new_attr_1 = []
    new_attr_2 = []
    if object_len == 3:
        new_attr_1: list[dict | None] = [None] * len(t_only)
        new_attr_2: list[dict | None] = [None] * len(t_only)

    for vi in range(2):
        new_v = new_v1
        new_attr = new_attr_1
        tsv = tsv1
        if vi == 1:
            if operation is None:
                new_v = new_v2
                new_attr = new_attr_2
            tsv = tsv2

        last_real_idx = None
        for i, (t, idxs) in enumerate(sorted_timings):
            attr_to_put = None
            if idxs[vi] is not None:
                # ok an exact value
                last_real_idx = idxs[vi]
                val_to_put = tsv[last_real_idx][1]
                if object_len == 3:
                    attr_to_put = tsv[last_real_idx][2]
            else:
                if last_real_idx is None:
                    # we have new values "before" the first real value"
                    val_to_put = tsv[0][1]
                    if object_len == 3:
                        attr_to_put = tsv[0][2]
                elif last_real_idx == len(tsv) - 1:
                    # we have new values "after" the last real value"
                    val_to_put = tsv[-1][1]
                    if object_len == 3:
                        attr_to_put = tsv[-1][2]
                else:
                    # we have new values "between" two real values"
                    # interpolate
                    vcur = tsv[last_real_idx][1]
                    vnxt = tsv[last_real_idx + 1][1]

                    if vnxt is None:
                        val_to_put = vcur
                    elif vcur is None:
                        val_to_put = None
                    else:
                        d1 = float((t - tsv[last_real_idx][0]).total_seconds())
                        d2 = float((tsv[last_real_idx + 1][0] - tsv[last_real_idx][0]).total_seconds())
                        if d2 > 0:
                            nv = (d1 / d2) * (vnxt - vcur) + vcur
                            val_to_put = float(nv)
                        else:
                            val_to_put = vcur
                    if object_len == 3:
                        attr_to_put = tsv[last_real_idx][2]

            if object_len == 3 and attr_to_put is not None:
                attr_to_put = dict(attr_to_put)

            if vi == 0 or operation is None:
                new_v[i] = val_to_put
                if object_len == 3:
                    new_attr[i] = attr_to_put
            else:
                if new_v[i] is None or val_to_put is None:
                    new_v[i] = None
                else:
                    new_v[i] = operation(new_v[i], val_to_put)
                if object_len == 3:
                    if new_attr[i] is None:
                        new_attr[i] = attr_to_put
                    elif attr_to_put is not None:
                        new_attr[i].update(attr_to_put)

    # ok so we do have values and timings for 1 and 2
    if operation is not None:
        if object_len == 3:
            return list(zip(t_only, new_v1, new_attr_1))
        else:
            return list(zip(t_only, new_v1))
    if object_len == 3:
        return list(zip(t_only, new_v1, new_attr_1)), list(zip(t_only, new_v2, new_attr_2))
    else:
        return list(zip(t_only, new_v1)), list(zip(t_only, new_v2))


def get_slots_from_time_series(
    time_serie, start_time: datetime, end_time: datetime | None = None
) -> list[tuple[datetime | None, str | float | None]]:
    if not time_serie:
        return []

    start_idx = bisect_left(time_serie, start_time, key=itemgetter(0))
    # get one before to have the timing just before
    if start_idx > 0:
        if time_serie[start_idx][0] != start_time:
            start_idx -= 1

    if end_time is None:
        return time_serie[start_idx : start_idx + 1]

    end_idx = bisect_left(time_serie, end_time, key=itemgetter(0))
    if end_idx >= len(time_serie):
        end_idx = len(time_serie) - 1
    elif end_idx < len(time_serie) - 1:
        # take one after
        if time_serie[end_idx][0] != end_time:
            end_idx += 1

    return time_serie[start_idx : end_idx + 1]


def get_value_from_time_series(
    time_series, time: datetime, interpolation_operation: Callable[[Any, Any, datetime], Any] | None = None
) -> tuple[datetime | None, str | float | None, bool, int]:

    # find the closest time in the time serie
    if time_series is None or len(time_series) == 0:
        return (None, None, False, -1)

    # small optim:
    if time_series[-1][0] == time:
        res = time_series[-1]
        res_idx = len(time_series) - 1
    elif time_series[0][0] == time:
        res = time_series[0]
        res_idx = 0
    else:
        idx = bisect_left(time_series, time, key=itemgetter(0))

        if idx >= len(time_series):
            res = time_series[-1]
            res_idx = len(time_series) - 1
        elif idx <= 0:
            res = time_series[0]
            res_idx = 0
        elif time_series[idx][0] == time:
            res = time_series[idx]
            res_idx = idx
        else:
            # we have multiple scenarios here : we can take the closest one or compute an interpolated value
            if interpolation_operation is None:
                # a time serie is normally by construction: the point gives its value to the whole
                # step
                res = time_series[idx - 1]
                res_idx = idx - 1

                # if time - time_series[idx - 1][0] <= time_series[idx][0] - time:
                #     res = time_series[idx - 1]
                #     res_idx = idx - 1
                # else:
                #     res = time_series[idx]
                #     res_idx = idx
            else:
                v1 = time_series[idx - 1]
                v2 = time_series[idx]

                if v1[1] is None and v2[1] is None:
                    res = (None, None)
                    res_idx = -1
                elif v1[1] is None:
                    res = v2
                    res_idx = idx
                elif v2[1] is None:
                    res = v1
                    res_idx = idx - 1
                else:
                    res = interpolation_operation(v1, v2, time)
                    res_idx = idx - 1

    return res[0], res[1], res[0] == time, res_idx


def slot_value_from_time_series(
    forecast: list[tuple[datetime, float]],
    begin_slot: datetime,
    end_slot: datetime,
    last_end: int,
    geometric_smoothing: bool = False,
) -> tuple[int, float]:
    """Extract an averaged power value from a forecast time series for a given slot.

    Mirrors the logic of ``PeriodSolver._power_slot_from_forecast`` but as a
    standalone function (no ``self`` dependency).

    Args:
        forecast: sorted list of (datetime, value) pairs.
        begin_slot: start of the time slot.
        end_slot: end of the time slot.
        last_end: index of the last entry used in the previous slot (-1 initially).
        geometric_smoothing: if True, linearly interpolate at slot boundaries.

    Returns:
        (new_last_end, averaged_value) tuple.
    """
    if not forecast:
        return last_end, 0.0

    prev_end = last_end
    while last_end < len(forecast) - 1 and forecast[last_end + 1][0] <= end_slot:
        last_end += 1

    power_series: list[tuple[datetime, float]] = []
    if prev_end >= 0:
        if forecast[prev_end][0] == begin_slot:
            power_series.append((begin_slot, forecast[prev_end][1]))
        elif forecast[prev_end][0] < begin_slot and prev_end < len(forecast) - 1:
            adapted_power = forecast[prev_end][1]
            if geometric_smoothing:
                dt = (forecast[prev_end + 1][0] - forecast[prev_end][0]).total_seconds()
                if dt > 0:
                    adapted_power += (
                        (forecast[prev_end + 1][1] - forecast[prev_end][1])
                        * (begin_slot - forecast[prev_end][0]).total_seconds()
                        / dt
                    )
            power_series.append((begin_slot, adapted_power))
    for j in range(prev_end + 1, last_end + 1):
        power_series.append((forecast[j][0], forecast[j][1]))

    if last_end < len(forecast) - 1 and forecast[last_end][0] < end_slot:
        adapted_power = forecast[last_end][1]
        if geometric_smoothing:
            dt = (forecast[last_end + 1][0] - forecast[last_end][0]).total_seconds()
            if dt > 0:
                adapted_power += (
                    (forecast[last_end + 1][1] - forecast[last_end][1])
                    * (end_slot - forecast[last_end][0]).total_seconds()
                    / dt
                )
        power_series.append((end_slot, adapted_power))

    if len(power_series) == 0 and prev_end == last_end and last_end == len(forecast) - 1:
        power_series.append((forecast[prev_end][0], forecast[prev_end][1]))

    return last_end, get_average_time_series(power_series, geometric_mean=geometric_smoothing)


def align_time_series_on_time_slots(
    time_series: list[tuple[datetime, float]],
    slot_boundaries: list[datetime],
    geometric_smoothing: bool = False,
) -> list[tuple[datetime, float]]:
    """Align a time series onto time slots defined by consecutive boundary pairs.

    For each consecutive pair ``(slot_boundaries[i], slot_boundaries[i+1])``,
    computes the time-weighted average of *time_series* within that interval
    using :func:`slot_value_from_time_series`.

    Args:
        time_series: sorted list of (datetime, value) pairs.
        slot_boundaries: ordered list of N boundary datetimes producing N-1 slots.
        geometric_smoothing: passed through to ``slot_value_from_time_series``.

    Returns:
        List of (slot_start, averaged_value) with one entry per slot.
    """
    if not time_series or len(slot_boundaries) < 2:
        return []

    result: list[tuple[datetime, float]] = []
    last_end = -1
    if time_series[0][0] < slot_boundaries[0]:
        while last_end < len(time_series) - 1 and time_series[last_end + 1][0] <= slot_boundaries[0]:
            last_end += 1

    for i in range(len(slot_boundaries) - 1):
        last_end, value = slot_value_from_time_series(
            time_series, slot_boundaries[i], slot_boundaries[i + 1], last_end, geometric_smoothing
        )
        result.append((slot_boundaries[i], value))
    return result
