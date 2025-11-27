import copy
from datetime import datetime
from typing import Mapping, Any, Dict, Tuple
import numpy as np


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


def hungarian_algorithm(cost_matrix: np.ndarray) -> Dict[int, int]:
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
        padded = np.full((n, n), cost_matrix.max() + 1e6)
        padded[:n_rows, :n_cols] = cost_matrix
        cost_matrix = padded

    # Step 1: Subtract row minimums
    cost_matrix = cost_matrix - cost_matrix.min(axis=1, keepdims=True)

    # Step 2: Subtract column minimums
    cost_matrix = cost_matrix - cost_matrix.min(axis=0, keepdims=True)

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

        min_uncovered = uncovered_values.min()

        # Adjust matrix
        cost_matrix[~covered_rows[:, None] & ~covered_cols[None, :]] -= min_uncovered
        cost_matrix[covered_rows[:, None] & covered_cols[None, :]] += min_uncovered

    # Fallback greedy assignment
    full_assignment = _greedy_assignment(cost_matrix)
    # Filter to original dimensions
    return {i: full_assignment[i] for i in range(original_n_rows) if i in full_assignment and full_assignment[i] < original_n_cols}


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


def _augment(row: int, cost_matrix: np.ndarray, assignment: np.ndarray,
             assigned_cols: set) -> bool:
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


def _find_minimum_cover(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find minimum line cover of zeros using König's theorem."""
    n = len(cost_matrix)

    # Build matching from zeros
    assignment = _try_assign(cost_matrix)
    if assignment is None:
        assignment = np.full(n, -1, dtype=int)

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


def _greedy_assignment(cost_matrix: np.ndarray) -> Dict[int, int]:
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
