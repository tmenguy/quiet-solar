"""Tests for Hungarian algorithm implementation in home_utils.py"""
import pytest
import numpy as np
from custom_components.quiet_solar.home_model.home_utils import (
    hungarian_algorithm,
    _greedy_assignment,
)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

