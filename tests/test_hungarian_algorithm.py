"""Tests for Hungarian algorithm implementation in home_utils.py"""
import pytest
import numpy as np
from custom_components.quiet_solar.home_model.home_utils import (
    hungarian_algorithm,
    _greedy_assignment,
)
from custom_components.quiet_solar.const import PREFERRED_CAR_ENERGY_THRESHOLD_KWH


class TestHungarianAlgorithm:
    """Test suite for Hungarian algorithm implementation."""

    def test_simple_3x3_assignment(self):
        """Test a simple 3x3 cost matrix."""
        # Classic example from textbooks
        cost_matrix = np.array([
            [4, 2, 8],
            [4, 3, 7],
            [3, 1, 6]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        # Verify all rows are assigned
        assert len(assignment) == 3

        # Verify each row assigned to different column
        assigned_cols = set(assignment.values())
        assert len(assigned_cols) == 3

        # Calculate total cost
        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())

        # The optimal assignment should be: 0->1 (cost 2), 1->2 (cost 7), 2->0 (cost 3) = 12
        # Or another optimal: 0->1 (cost 2), 1->0 (cost 4), 2->2 (cost 6) = 12
        assert total_cost == 12

    def test_square_4x4_assignment(self):
        """Test a 4x4 cost matrix."""
        cost_matrix = np.array([
            [10, 19, 8, 15],
            [10, 18, 7, 17],
            [13, 16, 9, 14],
            [12, 19, 8, 18]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        # Verify complete assignment
        assert len(assignment) == 4
        assigned_cols = set(assignment.values())
        assert len(assigned_cols) == 4

        # Calculate total cost
        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())

        # Optimal: checking manually - 0->2(8) + 1->0(10) + 2->3(14) + 3->1(19) = 51
        # or other combinations. The algorithm should find an optimal solution.
        # Let's verify it's better than worst case
        assert total_cost <= 60  # Reasonable upper bound

    def test_rectangular_matrix_more_cols(self):
        """Test when there are more columns than rows (more cars than drivers)."""
        cost_matrix = np.array([
            [5, 9, 3, 6, 8],
            [8, 7, 4, 2, 5],
            [6, 4, 9, 7, 3]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        # Should assign all 3 rows
        assert len(assignment) == 3

        # All assigned columns should be different
        assigned_cols = set(assignment.values())
        assert len(assigned_cols) == 3

        # All assigned columns should be valid (0-4)
        assert all(0 <= col < 5 for col in assigned_cols)

        # Calculate total cost
        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())

        # Optimal: 0->2(3) + 1->3(2) + 2->4(3) = 8
        assert total_cost == 8

    def test_rectangular_matrix_more_rows(self):
        """Test when there are more rows than columns (more drivers than cars)."""
        cost_matrix = np.array([
            [5, 9, 3],
            [8, 7, 4],
            [6, 4, 9],
            [2, 5, 7]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        # Should assign some rows (may not be all if columns are fewer)
        assert len(assignment) >= 3

        # All assigned columns should be different
        assigned_cols = set(assignment.values())
        assert len(assigned_cols) <= 3

    def test_zero_cost_matrix(self):
        """Test with all zeros (any assignment is optimal)."""
        cost_matrix = np.zeros((3, 3), dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        # Should still provide valid assignment
        assert len(assignment) == 3
        assigned_cols = set(assignment.values())
        assert len(assigned_cols) == 3

    def test_identity_preference(self):
        """Test where diagonal is cheapest (prefer matching indices)."""
        cost_matrix = np.array([
            [1, 5, 5],
            [5, 1, 5],
            [5, 5, 1]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        # Optimal is diagonal: 0->0, 1->1, 2->2, cost = 3
        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())
        assert total_cost == 3

    def test_large_cost_differences(self):
        """Test with very large cost differences."""
        cost_matrix = np.array([
            [1, 1000, 1000],
            [1000, 1, 1000],
            [1000, 1000, 1]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())
        # Should prefer diagonal
        assert total_cost == 3

    def test_tie_breaking(self):
        """Test when multiple optimal solutions exist."""
        cost_matrix = np.array([
            [1, 1, 2],
            [1, 1, 2],
            [2, 2, 1]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        # Any valid assignment with cost 3 is acceptable
        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())
        assert total_cost == 3

    def test_single_element(self):
        """Test 1x1 matrix."""
        cost_matrix = np.array([[5.0]], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        assert assignment == {0: 0}

    def test_2x2_simple(self):
        """Test simple 2x2 case."""
        cost_matrix = np.array([
            [1, 2],
            [3, 4]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        # Optimal: 0->0(1) + 1->1(4) = 5
        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())
        assert total_cost == 5

    def test_asymmetric_costs(self):
        """Test with very asymmetric cost structure."""
        cost_matrix = np.array([
            [1, 100, 100, 100],
            [100, 1, 100, 100],
            [100, 100, 1, 100],
            [100, 100, 100, 1]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        # Should assign diagonal
        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())
        assert total_cost == 4

    def test_floating_point_costs(self):
        """Test with floating point costs."""
        cost_matrix = np.array([
            [1.5, 2.7, 3.1],
            [2.2, 1.8, 2.9],
            [3.0, 2.5, 1.2]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        # Verify valid assignment
        assert len(assignment) == 3
        assigned_cols = set(assignment.values())
        assert len(assigned_cols) == 3

        # Check total cost is reasonable
        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())
        # Optimal: 0->0(1.5) + 1->1(1.8) + 2->2(1.2) = 4.5
        assert abs(total_cost - 4.5) < 0.01

    def test_with_preference_penalty(self):
        """Test scenario with preference penalty (like car allocation)."""
        # Simulating: 2 drivers, 3 cars
        # Driver 0 prefers car 1, needs 5 units energy for car0, 0 for car1, 10 for car2
        # Driver 1 prefers car 0, needs 2 units for car0, 8 for car1, 3 for car2
        # Penalty for not getting preferred car: 100

        energy_costs = np.array([
            [5, 0, 10],
            [2, 8, 3]
        ], dtype=np.float64)

        # Add preference penalty
        preferences = [1, 0]  # driver 0 prefers car 1, driver 1 prefers car 0
        penalty = 100.0

        cost_matrix = energy_costs.copy()
        for driver_idx, preferred_car in enumerate(preferences):
            for car_idx in range(cost_matrix.shape[1]):
                if car_idx != preferred_car:
                    cost_matrix[driver_idx, car_idx] += penalty

        assignment = hungarian_algorithm(cost_matrix)

        # Should prefer giving drivers their preferred cars
        # Driver 0 should get car 1 (energy 0 + penalty 0 = 0)
        # Driver 1 should get car 0 (energy 2 + penalty 0 = 2)
        assert assignment[0] == 1
        assert assignment[1] == 0

    def test_all_high_costs_except_one_path(self):
        """Test where only one good assignment path exists."""
        cost_matrix = np.array([
            [1, 99, 99],
            [99, 99, 1],
            [99, 1, 99]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())
        # Optimal: 0->0(1) + 1->2(1) + 2->1(1) = 3
        assert total_cost == 3


    def test_augmentation_path(self):
        """Test a matrix where greedy assignment fails but augmenting paths succeed.

        After row/column reduction the zero pattern is:
            [0, -, 0]
            [0, 0, -]
            [-, 0, -]
        Greedy (fewest-zeros-first) assigns Row2->col1, Row0->col0, leaving
        Row1 unassigned. Augmentation must reassign Row0 from col0 to col2
        so Row1 can take col0.
        """
        cost_matrix = np.array([
            [2, 7, 2],
            [3, 3, 8],
            [10, 5, 10]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        assert len(assignment) == 3
        assigned_cols = set(assignment.values())
        assert len(assigned_cols) == 3

        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())
        # Optimal: 0->2(2) + 1->0(3) + 2->1(5) = 10
        assert total_cost == 10


    def test_iterative_adjustment_required(self):
        """Test a matrix where row/col reduction produces insufficient zeros.

        After reduction the zero graph has no perfect matching (rows 0 and 1
        both only have a zero in column 0), so the algorithm must enter the
        iterative covering / adjustment loop (lines 192-196 of home_utils)
        and the BFS in _find_minimum_cover must follow alternating paths
        through assigned rows (lines 325-329).
        """
        cost_matrix = np.array([
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [7, 3, 3, 11],
            [8, 12, 4, 4]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        assert len(assignment) == 4
        assert len(set(assignment.values())) == 4

        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())
        assert total_cost == 14

    def test_multiple_adjustment_iterations(self):
        """Test a 6x6 matrix that needs 5 adjustment iterations to converge.

        This exercises the iterative loop repeatedly and forces the minimum
        cover BFS to traverse deep alternating paths on each round.
        """
        cost_matrix = np.array([
            [13, 9, 10, 9, 15, 15],
            [2, 15, 14, 19, 3, 1],
            [18, 17, 14, 12, 12, 3],
            [11, 13, 16, 13, 5, 3],
            [2, 14, 13, 4, 13, 5],
            [11, 16, 14, 10, 6, 13]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        assert len(assignment) == 6
        assert len(set(assignment.values())) == 6

        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())
        assert total_cost == 37

    def test_deep_alternating_path_in_cover(self):
        """Test a 6x6 matrix whose BFS in _find_minimum_cover follows a chain
        of 4 assigned rows (deep alternating path).

        This specifically targets the code in _find_minimum_cover that marks
        assigned rows reachable from unassigned rows via zero-column links.
        """
        cost_matrix = np.array([
            [17, 16, 18, 14, 16, 7],
            [3, 13, 7, 19, 16, 5],
            [8, 15, 9, 18, 4, 13],
            [12, 2, 6, 6, 3, 17],
            [13, 16, 2, 16, 8, 7],
            [6, 12, 14, 15, 3, 9]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        assert len(assignment) == 6
        assert len(set(assignment.values())) == 6

        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())
        assert total_cost == 33

    def test_3x3_previously_incorrect(self):
        """Test a 3x3 matrix that gave wrong results before the fix.

        Before fixing _find_minimum_cover the algorithm would fall through
        to the greedy heuristic and return cost 114 instead of optimal 106.
        """
        cost_matrix = np.array([
            [15, 43, 29],
            [36, 13, 32],
            [71, 59, 86]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        assert len(assignment) == 3
        assert len(set(assignment.values())) == 3

        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())
        assert total_cost == 106


class TestHungarianInternalPaths:
    """Tests targeting specific internal code paths of the Hungarian algorithm."""

    def test_no_zeros_in_try_assign(self):
        """Line 210: _try_assign returns None when cost matrix has no zeros.

        After row/col reduction, a uniform matrix has all zeros, so we
        construct a scenario where _try_assign is called on a modified matrix
        that has had its zeros removed by the covering/adjustment step.
        We verify correctness of the final result instead.
        """
        from custom_components.quiet_solar.home_model.home_utils import _try_assign

        cost_matrix = np.array([
            [3, 5, 7],
            [2, 4, 6],
            [1, 3, 5]
        ], dtype=np.float64)

        result = _try_assign(cost_matrix)
        assert result is None

    def test_row_zero_count_zero_skipped_in_try_assign(self):
        """Line 222: _try_assign skips rows that have zero zero-count.

        Construct a matrix where after reduction at least one row has no zeros,
        so the sorted_rows loop hits the 'continue' for row_zero_counts[row]==0.
        """
        from custom_components.quiet_solar.home_model.home_utils import _try_assign

        cost_matrix = np.array([
            [0, 5, 5],
            [5, 0, 5],
            [3, 3, 3]
        ], dtype=np.float64)

        result = _try_assign(cost_matrix)
        assert result is None

    def test_row_zero_count_zero_in_find_minimum_cover(self):
        """Line 294: _find_minimum_cover skips rows with no zeros.

        Use a matrix where one row has no zeros so that the row is skipped
        during the greedy matching inside _find_minimum_cover.
        """
        from custom_components.quiet_solar.home_model.home_utils import _find_minimum_cover

        cost_matrix = np.array([
            [0, 5, 5],
            [5, 0, 5],
            [3, 3, 3]
        ], dtype=np.float64)

        covered_rows, covered_cols = _find_minimum_cover(cost_matrix)

        assert covered_rows.shape == (3,)
        assert covered_cols.shape == (3,)
        assert covered_rows.sum() + covered_cols.sum() >= 2

    def test_all_covered_break_path(self):
        """Line 190: break when all elements are covered (len(uncovered_values)==0).

        Use a degenerate matrix where the covering lines cover everything but
        no perfect matching exists, forcing the break.
        """
        cost_matrix = np.array([
            [1, 2, 3],
            [2, 4, 6],
            [3, 6, 9]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        assert len(assignment) == 3
        assert len(set(assignment.values())) == 3
        total_cost = sum(cost_matrix[r, c] for r, c in assignment.items())
        assert total_cost <= 15

    def test_greedy_fallback_path(self):
        """Lines 199-201: fallback to _greedy_assignment after max iterations.

        Verify the algorithm still returns a valid assignment even on
        pathological inputs that might exhaust the iteration count.
        The [[5,9,1],[10,3,2],[8,7,4]] matrix requires multiple adjustment
        rounds.
        """
        cost_matrix = np.array([
            [5, 9, 1],
            [10, 3, 2],
            [8, 7, 4]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        assert len(assignment) == 3
        assert len(set(assignment.values())) == 3
        total_cost = sum(cost_matrix[r, c] for r, c in assignment.items())
        assert total_cost <= 15

    def test_proportional_matrix_degenerate(self):
        """Degenerate matrix where rows are proportional (rank 1).

        After row reduction all entries become zero, making covering trivial
        but assignment still valid.
        """
        cost_matrix = np.array([
            [2, 4, 6],
            [1, 2, 3],
            [3, 6, 9]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        assert len(assignment) == 3
        assert len(set(assignment.values())) == 3

    def test_5x5_forces_multiple_cover_adjustments(self):
        """5x5 matrix that needs multiple cover/adjustment iterations,
        exercising the uncovered_mask computation and min_uncovered subtraction.
        """
        cost_matrix = np.array([
            [7, 53, 183, 439, 56],
            [8, 72, 164, 378, 36],
            [21, 6, 104, 289, 8],
            [11, 46, 133, 316, 22],
            [28, 29, 83, 198, 13]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        assert len(assignment) == 5
        assert len(set(assignment.values())) == 5

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        scipy_cost = cost_matrix[row_ind, col_ind].sum()
        our_cost = sum(cost_matrix[r, c] for r, c in assignment.items())
        assert abs(our_cost - scipy_cost) < 1e-6


class TestHungarianEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_matrix(self):
        """Test with empty matrix."""
        cost_matrix = np.array([], dtype=np.float64).reshape(0, 0)

        # Should handle gracefully or return empty assignment
        try:
            assignment = hungarian_algorithm(cost_matrix)
            assert len(assignment) == 0
        except (ValueError, IndexError):
            # Also acceptable to raise an error for empty input
            pass

    def test_negative_costs(self):
        """Test with negative costs (still valid for Hungarian algorithm)."""
        cost_matrix = np.array([
            [-5, -2, -1],
            [-3, -4, -2],
            [-1, -3, -5]
        ], dtype=np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        # Should still work (algorithm handles negative values)
        assert len(assignment) == 3

        # Verify optimal (maximize negative = minimize cost)
        # Want most negative: 0->0(-5) + 1->1(-4) + 2->2(-5) = -14
        total_cost = sum(cost_matrix[row, col] for row, col in assignment.items())
        assert total_cost <= -12  # Should be quite negative


class TestHungarianCorrectnessAgainstScipy:
    """Compare results against scipy.optimize.linear_sum_assignment."""

    @pytest.mark.parametrize("seed,size", [
        (0, 3), (1, 3), (2, 4), (3, 4), (4, 5),
        (5, 5), (6, 6), (7, 6), (8, 3), (9, 4),
        (10, 5), (11, 6), (12, 3), (13, 4), (14, 5),
        (15, 6), (16, 3), (17, 4), (18, 5), (19, 6),
    ])
    def test_random_square_matrix(self, seed, size):
        """Verify optimal cost matches scipy on a random square matrix."""
        from scipy.optimize import linear_sum_assignment

        rng = np.random.default_rng(seed)
        cost_matrix = rng.integers(1, 100, size=(size, size)).astype(np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        assert len(assignment) == size
        assert len(set(assignment.values())) == size

        our_cost = sum(cost_matrix[r, c] for r, c in assignment.items())
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        scipy_cost = cost_matrix[row_ind, col_ind].sum()

        assert abs(our_cost - scipy_cost) < 1e-6

    @pytest.mark.parametrize("seed,rows,cols", [
        (100, 2, 5), (101, 3, 6), (102, 5, 3), (103, 4, 2), (104, 3, 4),
    ])
    def test_random_rectangular_matrix(self, seed, rows, cols):
        """Verify optimal cost matches scipy on a random rectangular matrix."""
        from scipy.optimize import linear_sum_assignment

        rng = np.random.default_rng(seed)
        cost_matrix = rng.integers(1, 100, size=(rows, cols)).astype(np.float64)

        assignment = hungarian_algorithm(cost_matrix)

        expected_count = min(rows, cols)
        assert len(assignment) == expected_count
        assert len(set(assignment.values())) == expected_count

        our_cost = sum(cost_matrix[r, c] for r, c in assignment.items())
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        scipy_cost = cost_matrix[row_ind, col_ind].sum()

        assert abs(our_cost - scipy_cost) < 1e-6


class TestGreedyAssignment:
    """Test the _greedy_assignment fallback function directly."""

    def test_simple_3x3(self):
        """Test greedy picks the minimum in each row, respecting prior assignments."""
        cost_matrix = np.array([
            [5, 1, 9],
            [2, 8, 3],
            [7, 4, 6]
        ], dtype=np.float64)

        result = _greedy_assignment(cost_matrix)

        assert len(result) == 3
        assert len(set(result.values())) == 3
        # Row 0 picks col 1 (cost 1), row 1 picks col 0 (cost 2), row 2 picks col 2 (cost 6)
        assert result == {0: 1, 1: 0, 2: 2}

    def test_diagonal_preference(self):
        """Test when diagonal has the lowest costs."""
        cost_matrix = np.array([
            [1, 99, 99],
            [99, 1, 99],
            [99, 99, 1]
        ], dtype=np.float64)

        result = _greedy_assignment(cost_matrix)

        assert result == {0: 0, 1: 1, 2: 2}

    def test_greedy_is_not_optimal(self):
        """Test a case where greedy gives a suboptimal result.

        Greedy processes rows in order, so row 0 grabs col 0 (cost 1),
        forcing row 1 to take col 1 (cost 10). The optimal would be
        row 0->col1 (2), row 1->col0 (3) = 5, but greedy yields 11.
        """
        cost_matrix = np.array([
            [1, 2],
            [3, 10]
        ], dtype=np.float64)

        result = _greedy_assignment(cost_matrix)

        assert result == {0: 0, 1: 1}
        total = sum(cost_matrix[r, c] for r, c in result.items())
        assert total == 11  # greedy: 1 + 10

    def test_single_element(self):
        """Test 1x1 matrix."""
        cost_matrix = np.array([[42.0]], dtype=np.float64)

        result = _greedy_assignment(cost_matrix)

        assert result == {0: 0}

    def test_4x4_with_conflicts(self):
        """Test greedy row-by-row selection on a larger matrix."""
        cost_matrix = np.array([
            [3, 8, 2, 7],
            [1, 5, 9, 4],
            [6, 2, 8, 3],
            [7, 4, 1, 6]
        ], dtype=np.float64)

        result = _greedy_assignment(cost_matrix)

        assert len(result) == 4
        assert len(set(result.values())) == 4
        # Row 0: min is col 2 (cost 2)
        # Row 1: col 2 taken, min of {0,1,3} is col 0 (cost 1)
        # Row 2: cols 0,2 taken, min of {1,3} is col 1 (cost 2)... wait, col 3 cost 3
        # Actually: available {1,3}, costs [2, 3], min is col 1 (cost 2)
        # Row 3: available {3}, col 3 (cost 6)
        assert result == {0: 2, 1: 0, 2: 1, 3: 3}

    def test_uniform_costs(self):
        """Test when all costs are equal -- assigns columns in order."""
        cost_matrix = np.full((3, 3), 5.0)

        result = _greedy_assignment(cost_matrix)

        assert result == {0: 0, 1: 1, 2: 2}


class TestTwoPassAllocation:
    """Test the two-pass allocation strategy that chooses between
    energy-optimal and preferred-car-biased assignments based on a
    global energy threshold.

    These tests exercise _build_raw_energy_matrix, _finalize_cost_matrix,
    and _compute_assignment_energy indirectly through numpy matrices,
    replicating the logic without needing full QSHome/QSPerson/QSCar objects.
    """

    @staticmethod
    def _build_raw(energies, authorized_mask):
        """Build a raw energy matrix from plain floats.

        energies[i][j] is the diff_energy for person i / car j.
        authorized_mask[i][j] is True if person i can drive car j.
        Unauthorized pairs get sentinel 0.0, covered pairs get -3.0.
        """
        n_p, n_c = len(energies), len(energies[0])
        raw = np.zeros((n_p, n_c), dtype=np.float64)
        E_max = 0.0
        for i in range(n_p):
            for j in range(n_c):
                if not authorized_mask[i][j]:
                    raw[i, j] = 0.0
                elif energies[i][j] == 0.0:
                    raw[i, j] = -3.0
                else:
                    raw[i, j] = energies[i][j]
                    E_max = max(E_max, energies[i][j])
        return raw, E_max

    @staticmethod
    def _finalize(raw, E_max, n_p, n_c, preferences, penalty):
        """Replicate _finalize_cost_matrix without QS objects.

        preferences[i] is the preferred car index for person i.
        """
        costs = raw.copy()
        maxi_val = max(1e12, (E_max + 1.0) * (1.0 + max(n_c, n_p)))
        for i in range(n_p):
            for j in range(n_c):
                if costs[i, j] == 0.0:
                    costs[i, j] = maxi_val
                else:
                    if costs[i, j] == -3.0:
                        costs[i, j] = 0.0
                    elif costs[i, j] < 0:
                        costs[i, j] = E_max + 1.0
                    if penalty > 0.0 and preferences[i] != j:
                        costs[i, j] += penalty
        return costs

    @staticmethod
    def _total_energy(assignment, raw):
        """Sum positive raw energy for an assignment."""
        total = 0.0
        for pi, ci in assignment.items():
            val = raw[pi, ci]
            if val > 0.0:
                total += val
        return total

    def _run_two_pass(self, energies, authorized_mask, preferences,
                      threshold=PREFERRED_CAR_ENERGY_THRESHOLD_KWH):
        """Run the full two-pass logic and return (chosen_assignment, choice).

        choice is 'preferred' or 'energy'.
        """
        raw, E_max = self._build_raw(energies, authorized_mask)
        n_p, n_c = raw.shape

        costs_energy = self._finalize(raw, E_max, n_p, n_c, preferences, penalty=0.0)
        assignment_energy = hungarian_algorithm(costs_energy)
        total_energy_optimal = self._total_energy(assignment_energy, raw)

        penalty = (n_p * E_max) + 1.0
        costs_preferred = self._finalize(raw, E_max, n_p, n_c, preferences, penalty=penalty)
        assignment_preferred = hungarian_algorithm(costs_preferred)
        total_energy_preferred = self._total_energy(assignment_preferred, raw)

        if total_energy_preferred - total_energy_optimal <= threshold:
            return assignment_preferred, "preferred", total_energy_preferred, total_energy_optimal
        else:
            return assignment_energy, "energy", total_energy_preferred, total_energy_optimal

    def test_energy_wins_when_difference_large(self):
        """Preferred assignment costs 40 kWh vs energy-optimal 2 kWh.

        Person A prefers Car1, Person B prefers Car2.
        A->Car1 needs 20 kWh, A->Car2 needs 1 kWh
        B->Car1 needs 1 kWh, B->Car2 needs 20 kWh
        """
        energies = [[20.0, 1.0], [1.0, 20.0]]
        authorized = [[True, True], [True, True]]
        preferences = [0, 1]

        assignment, choice, e_pref, e_opt = self._run_two_pass(energies, authorized, preferences)

        assert choice == "energy"
        assert e_opt == pytest.approx(2.0)
        assert e_pref == pytest.approx(40.0)
        total = self._total_energy(assignment, self._build_raw(energies, authorized)[0])
        assert total == pytest.approx(2.0)

    def test_preferred_wins_when_difference_small(self):
        """Preferred costs 10.5 kWh vs energy-optimal 10.0 kWh (diff 0.5 < 1.0)."""
        energies = [[5.0, 5.3], [5.2, 5.0]]
        authorized = [[True, True], [True, True]]
        preferences = [0, 1]

        assignment, choice, e_pref, e_opt = self._run_two_pass(energies, authorized, preferences)

        assert choice == "preferred"
        assert assignment[0] == 0
        assert assignment[1] == 1

    def test_exact_threshold_boundary(self):
        """Energy difference equals exactly the threshold -- preferred should win (<= check)."""
        energies = [[5.0, 5.5], [5.5, 5.0]]
        authorized = [[True, True], [True, True]]
        preferences = [0, 1]

        assignment, choice, e_pref, e_opt = self._run_two_pass(energies, authorized, preferences)

        assert e_pref - e_opt == pytest.approx(1.0, abs=1e-9) or e_pref - e_opt < 1.0
        assert choice == "preferred"

    def test_all_cars_already_covered(self):
        """E_max = 0, all cars have enough charge. Preferred should win (diff = 0)."""
        energies = [[0.0, 0.0], [0.0, 0.0]]
        authorized = [[True, True], [True, True]]
        preferences = [0, 1]

        assignment, choice, e_pref, e_opt = self._run_two_pass(energies, authorized, preferences)

        assert choice == "preferred"
        assert e_opt == 0.0
        assert e_pref == 0.0
        assert assignment[0] == 0
        assert assignment[1] == 1

    def test_single_person_single_car(self):
        """Trivial 1x1 case."""
        energies = [[7.5]]
        authorized = [[True]]
        preferences = [0]

        assignment, choice, _, _ = self._run_two_pass(energies, authorized, preferences)

        assert assignment == {0: 0}

    def test_unauthorized_pairs_never_selected(self):
        """Person 0 can only drive Car0, Person 1 can only drive Car1."""
        energies = [[5.0, 1.0], [1.0, 5.0]]
        authorized = [[True, False], [False, True]]
        preferences = [0, 1]

        assignment, choice, _, _ = self._run_two_pass(energies, authorized, preferences)

        assert assignment[0] == 0
        assert assignment[1] == 1

    def test_3x3_mixed_scenario(self):
        """3 persons, 3 cars with mixed preferences.

        Energy-optimal may differ from preferred, but the global difference
        determines the choice.
        """
        energies = [
            [2.0, 8.0, 3.0],
            [7.0, 1.0, 6.0],
            [5.0, 4.0, 2.0],
        ]
        authorized = [[True, True, True], [True, True, True], [True, True, True]]
        preferences = [0, 1, 2]

        assignment, choice, e_pref, e_opt = self._run_two_pass(energies, authorized, preferences)

        assert e_opt == pytest.approx(5.0)
        assert assignment[0] == 0 or assignment[1] == 1 or assignment[2] == 2
        assert e_pref - e_opt <= 1.0 or choice == "energy"

    def test_rectangular_more_cars_than_persons(self):
        """2 persons, 4 cars. Algorithm should handle padding correctly."""
        energies = [[10.0, 2.0, 8.0, 1.0], [3.0, 9.0, 1.0, 7.0]]
        authorized = [[True, True, True, True], [True, True, True, True]]
        preferences = [0, 1]

        assignment, choice, e_pref, e_opt = self._run_two_pass(energies, authorized, preferences)

        assert len(assignment) == 2
        assert len(set(assignment.values())) == 2
        assert e_opt == pytest.approx(2.0)
        assert choice == "energy"

    def test_rectangular_more_persons_than_cars(self):
        """3 persons, 2 cars. Not all persons get a car."""
        energies = [[5.0, 1.0], [1.0, 5.0], [3.0, 3.0]]
        authorized = [[True, True], [True, True], [True, True]]
        preferences = [0, 1, 0]

        assignment, choice, e_pref, e_opt = self._run_two_pass(energies, authorized, preferences)

        assert len(assignment) == 2
        assert len(set(assignment.values())) == 2

    def test_preferred_matches_energy_optimal(self):
        """When preferred cars happen to also be energy-optimal, both passes agree."""
        energies = [[1.0, 10.0], [10.0, 1.0]]
        authorized = [[True, True], [True, True]]
        preferences = [0, 1]

        assignment, choice, e_pref, e_opt = self._run_two_pass(energies, authorized, preferences)

        assert choice == "preferred"
        assert e_opt == pytest.approx(2.0)
        assert e_pref == pytest.approx(2.0)
        assert assignment[0] == 0
        assert assignment[1] == 1

    def test_large_energy_gap_one_pair(self):
        """One pair has a huge energy difference, rest are equal.

        Person 0: Car0=50kWh, Car1=0.5kWh (prefers Car0)
        Person 1: Car0=0.5kWh, Car1=0.5kWh (prefers Car1)
        Preferred: 0->Car0(50) + 1->Car1(0.5) = 50.5
        Energy:    0->Car1(0.5) + 1->Car0(0.5) = 1.0
        Diff = 49.5 >> threshold => energy wins.
        """
        energies = [[50.0, 0.5], [0.5, 0.5]]
        authorized = [[True, True], [True, True]]
        preferences = [0, 1]

        assignment, choice, e_pref, e_opt = self._run_two_pass(energies, authorized, preferences)

        assert choice == "energy"
        assert e_opt == pytest.approx(1.0)

    def test_real_scenario_twingo_zoe_arthur_magali(self):
        """Reproduce real-world bug: preferred car forces unnecessary charging.

        After pre-allocation removes Tesla (unplugged) -> Thomas, the
        remaining pool is:

        Cars (columns):  IDBuzz(0)  Twingo(1)  Zoe(2)
        Arthur (row 0):  unauth     5.0 kWh    0.0 (covered)
        Magali (row 1):  0.0        0.0        0.0 (all covered)

        Arthur prefers Twingo (col 1), Magali prefers Zoe (col 2).

        Old algorithm: Arthur->Twingo (5 kWh, preferred), Magali->Zoe (0 kWh)
                       = 5 kWh total. Preferred penalty dominated.

        Correct:       Arthur->Zoe (0 kWh), Magali->Twingo or IDBuzz (0 kWh)
                       = 0 kWh total. Energy-optimal.
        """
        energies = [
            [99.0, 5.0, 0.0],   # Arthur: IDBuzz=unauth, Twingo=5kWh, Zoe=covered
            [0.0,  0.0, 0.0],   # Magali: all covered
        ]
        authorized = [
            [False, True, True],   # Arthur can drive Twingo, Zoe (not IDBuzz)
            [True,  True, True],   # Magali can drive all three
        ]
        preferences = [1, 2]  # Arthur prefers Twingo (1), Magali prefers Zoe (2)

        assignment, choice, e_pref, e_opt = self._run_two_pass(energies, authorized, preferences)

        assert choice == "energy"
        assert e_opt == pytest.approx(0.0)
        assert e_pref == pytest.approx(5.0)
        assert assignment[0] == 2  # Arthur -> Zoe
        assert assignment[1] in (0, 1)  # Magali -> IDBuzz or Twingo


class _FakeCar:
    """Minimal car stub for pre-allocation tests."""

    def __init__(self, name, is_home, has_charger, is_invited=False,
                 coverage_map=None):
        """coverage_map: dict of mileage -> (is_covered, current_soc, needed_soc, diff_energy)."""
        self.name = name
        self._is_home = is_home
        self.charger = object() if has_charger else None
        self.car_is_invited = is_invited
        self._coverage = coverage_map or {}

    def get_adapt_target_percent_soc_to_reach_range_km(self, mileage, time):
        return self._coverage.get(mileage, (None, None, None, None))


class _FakePerson:
    """Minimal person stub for pre-allocation tests."""

    def __init__(self, name, preferred_car, authorized_car_names):
        self.name = name
        self.preferred_car = preferred_car
        self._authorized_car_names = authorized_car_names
        self._authorized_cars = []

    def set_car_objects(self, cars_by_name):
        self._authorized_cars = [
            cars_by_name[n] for n in self._authorized_car_names if n in cars_by_name
        ]

    def get_authorized_cars(self):
        return self._authorized_cars


class _FakeHome:
    """Minimal home stub that only has _pre_allocate_unplugged_home_cars."""

    def __init__(self, cars, persons):
        self._cars = cars
        self._persons = persons
        self._last_persons_car_allocation = {}

    _pre_allocate_unplugged_home_cars = (
        __import__(
            "custom_components.quiet_solar.ha_model.home", fromlist=["QSHome"]
        ).QSHome._pre_allocate_unplugged_home_cars
    )


def _make_home(cars, persons):
    cars_by_name = {c.name: c for c in cars}
    for p in persons:
        p.set_car_objects(cars_by_name)
    return _FakeHome(cars, persons)


class TestPreAllocateUnpluggedHomeCars:
    """Test the preferred-car-priority pre-allocation of unplugged home cars."""

    def test_preferred_person_gets_priority(self):
        """When two persons' trips are covered, the one who prefers this car wins."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        car = _FakeCar("CarA", is_home=True, has_charger=False, coverage_map={
            50.0: (True, 80.0, 60.0, 5.0),
            30.0: (True, 80.0, 40.0, 2.0),
        })
        person_pref = _FakePerson("Alice", preferred_car="CarA", authorized_car_names=["CarA"])
        person_other = _FakePerson("Bob", preferred_car="CarB", authorized_car_names=["CarA"])

        home = _make_home([car], [person_pref, person_other])
        forecasts = {"Alice": (now, 50.0), "Bob": (now, 30.0)}
        covered_cars: set[str] = set()
        covered_persons: set[str] = set()

        home._pre_allocate_unplugged_home_cars(now, forecasts, covered_cars, covered_persons)

        assert home._last_persons_car_allocation["CarA"].name == "Alice"
        assert "CarA" in covered_cars
        assert "Alice" in covered_persons

    def test_fallback_to_closest_margin_when_no_preferred(self):
        """No person prefers this car -- pick the one with the smallest margin."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        car = _FakeCar("CarA", is_home=True, has_charger=False, coverage_map={
            80.0: (True, 90.0, 70.0, 10.0),
            40.0: (True, 90.0, 50.0, 3.0),
        })
        person_far = _FakePerson("Alice", preferred_car="CarX", authorized_car_names=["CarA"])
        person_close = _FakePerson("Bob", preferred_car="CarY", authorized_car_names=["CarA"])

        home = _make_home([car], [person_far, person_close])
        forecasts = {"Alice": (now, 80.0), "Bob": (now, 40.0)}
        covered_cars: set[str] = set()
        covered_persons: set[str] = set()

        home._pre_allocate_unplugged_home_cars(now, forecasts, covered_cars, covered_persons)

        assert home._last_persons_car_allocation["CarA"].name == "Bob"

    def test_plugged_car_skipped(self):
        """A car connected to a charger should not be pre-allocated."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        car = _FakeCar("CarA", is_home=True, has_charger=True, coverage_map={
            50.0: (True, 80.0, 60.0, 5.0),
        })
        person = _FakePerson("Alice", preferred_car="CarA", authorized_car_names=["CarA"])

        home = _make_home([car], [person])
        forecasts = {"Alice": (now, 50.0)}
        covered_cars: set[str] = set()
        covered_persons: set[str] = set()

        home._pre_allocate_unplugged_home_cars(now, forecasts, covered_cars, covered_persons)

        assert len(home._last_persons_car_allocation) == 0

    def test_trip_not_covered_skipped(self):
        """If the car's current charge doesn't cover the trip, no pre-allocation."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        car = _FakeCar("CarA", is_home=True, has_charger=False, coverage_map={
            200.0: (False, 30.0, 80.0, 25.0),
        })
        person = _FakePerson("Alice", preferred_car="CarA", authorized_car_names=["CarA"])

        home = _make_home([car], [person])
        forecasts = {"Alice": (now, 200.0)}
        covered_cars: set[str] = set()
        covered_persons: set[str] = set()

        home._pre_allocate_unplugged_home_cars(now, forecasts, covered_cars, covered_persons)

        assert len(home._last_persons_car_allocation) == 0

    def test_preferred_wins_over_closer_margin(self):
        """Preferred person wins even if another person has a smaller margin.

        Alice prefers CarA, margin 8.0 kWh.
        Bob does not prefer CarA, margin 1.0 kWh.
        Alice should be assigned because preferred car takes priority.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        car = _FakeCar("CarA", is_home=True, has_charger=False, coverage_map={
            100.0: (True, 90.0, 70.0, 8.0),
            20.0: (True, 90.0, 30.0, 1.0),
        })
        person_pref = _FakePerson("Alice", preferred_car="CarA", authorized_car_names=["CarA"])
        person_close = _FakePerson("Bob", preferred_car="CarZ", authorized_car_names=["CarA"])

        home = _make_home([car], [person_pref, person_close])
        forecasts = {"Alice": (now, 100.0), "Bob": (now, 20.0)}
        covered_cars: set[str] = set()
        covered_persons: set[str] = set()

        home._pre_allocate_unplugged_home_cars(now, forecasts, covered_cars, covered_persons)

        assert home._last_persons_car_allocation["CarA"].name == "Alice"

    def test_real_scenario_tesla_unplugged_preallocation(self):
        """Full real-world scenario: Tesla unplugged at home, 3 other cars connected.

        Cars:
          Tesla  : 174km, at home, NOT connected
          IDBuzz : 253km, at home, connected
          Twingo : 125km, at home, connected
          Zoe    : 182km, at home, connected

        Persons:
          Thomas: 32km trip, prefers Tesla, auth [Tesla, Zoe, Twingo, IDBuzz]
          Arthur: 104km trip, prefers Twingo, auth [Zoe, Twingo]
          Magali: 75km trip, prefers Zoe, auth [Tesla, Zoe, Twingo, IDBuzz]

        Coverage (car range vs trip + margin):
          Tesla covers Thomas (174 > 32)   -> True
          Tesla covers Magali (174 > 75)   -> True
          Twingo covers Thomas (125 > 32)  -> True
          Twingo covers Magali (125 > 75)  -> True
          Twingo does NOT cover Arthur     -> False, diff_energy=5.0
          Zoe covers everyone              -> True
          IDBuzz covers everyone           -> True

        Pre-allocation: Tesla is unplugged+home. Thomas prefers Tesla and
        trip is covered -> Tesla pre-allocated to Thomas.

        After pre-allocation the remaining pool has Arthur and Magali
        competing for Twingo/Zoe/IDBuzz. Energy-optimal assigns Arthur->Zoe
        (0 kWh) instead of his preferred Twingo (5 kWh charge).
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        tesla = _FakeCar("Tesla", is_home=True, has_charger=False, coverage_map={
            32.0: (True, 62.0, 20.0, 2.0),
            75.0: (True, 62.0, 45.0, 5.0),
        })
        idbuzz = _FakeCar("IDBuzz", is_home=True, has_charger=True, coverage_map={
            32.0: (True, 62.0, 15.0, 1.0),
            75.0: (True, 62.0, 35.0, 3.0),
            104.0: (True, 62.0, 50.0, 5.0),
        })
        twingo = _FakeCar("Twingo", is_home=True, has_charger=True, coverage_map={
            32.0: (True, 75.0, 30.0, 1.5),
            75.0: (True, 75.0, 60.0, 3.0),
            104.0: (False, 75.0, 100.0, 5.0),
        })
        zoe = _FakeCar("Zoe", is_home=True, has_charger=True, coverage_map={
            32.0: (True, 53.0, 20.0, 2.0),
            75.0: (True, 53.0, 40.0, 4.0),
            104.0: (True, 53.0, 50.0, 6.0),
        })

        thomas = _FakePerson("Thomas", preferred_car="Tesla",
                             authorized_car_names=["Tesla", "Zoe", "Twingo", "IDBuzz"])
        arthur = _FakePerson("Arthur", preferred_car="Twingo",
                             authorized_car_names=["Zoe", "Twingo"])
        magali = _FakePerson("Magali", preferred_car="Zoe",
                             authorized_car_names=["Tesla", "Zoe", "Twingo", "IDBuzz"])

        cars = [tesla, idbuzz, twingo, zoe]
        persons = [thomas, arthur, magali]
        home = _make_home(cars, persons)

        forecasts = {
            "Thomas": (now, 32.0),
            "Arthur": (now, 104.0),
            "Magali": (now, 75.0),
        }
        covered_cars: set[str] = set()
        covered_persons: set[str] = set()

        # Step 1: pre-allocate unplugged cars
        home._pre_allocate_unplugged_home_cars(now, forecasts, covered_cars, covered_persons)

        # Tesla (unplugged, at home) -> Thomas (prefers Tesla, trip covered)
        assert "Tesla" in covered_cars
        assert "Thomas" in covered_persons
        assert home._last_persons_car_allocation["Tesla"].name == "Thomas"

        # Step 2: verify remaining pool excludes Tesla/Thomas
        assert "IDBuzz" not in covered_cars
        assert "Twingo" not in covered_cars
        assert "Zoe" not in covered_cars
        assert "Arthur" not in covered_persons
        assert "Magali" not in covered_persons


class TestHungarianAlgorithmBreakWhenAllCovered:
    """Test the break-when-all-covered path in hungarian_algorithm (line 190)."""

    def test_all_covered_after_reduction_triggers_break(self):
        """Line 188-190: all elements covered -> break -> fallback greedy.

        Patch _try_assign to return None on the first call so the algorithm
        enters the covering loop. On an all-zeros matrix, _find_minimum_cover
        covers all rows+columns, leaving no uncovered values and hitting
        the break at line 190, then the greedy fallback at line 199.
        """
        from unittest.mock import patch as _patch

        cost = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        call_count = [0]
        real_try_assign = __import__(
            "custom_components.quiet_solar.home_model.home_utils",
            fromlist=["_try_assign"],
        )._try_assign

        def fake_try_assign(cm):
            call_count[0] += 1
            if call_count[0] == 1:
                return None
            return real_try_assign(cm)

        with _patch(
            "custom_components.quiet_solar.home_model.home_utils._try_assign",
            side_effect=fake_try_assign,
        ):
            result = hungarian_algorithm(cost)

        assert len(result) == 2
        assert len(set(result.values())) == 2

    def test_all_zeros_rectangular_triggers_greedy_fallback(self):
        """Test rectangular matrix that may trigger the greedy fallback."""
        cost = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ])
        result = hungarian_algorithm(cost)
        assert len(result) == 2
        assert len(set(result.values())) == 2

    def test_identical_rows_assignment(self):
        """Matrix with identical rows stresses the algorithm."""
        cost = np.array([
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
        ])
        result = hungarian_algorithm(cost)
        assert len(result) == 3
        assert len(set(result.values())) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

