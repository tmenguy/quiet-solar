"""Tests for LoadConstraint.has_a_following_constraint() method."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytz

from custom_components.quiet_solar.const import (
    CONSTRAINT_TYPE_FILLER_AUTO,
    SOLVER_STEP_S,
)
from custom_components.quiet_solar.home_model.commands import LoadCommand
from custom_components.quiet_solar.home_model.constraints import (
    DATETIME_MAX_UTC,
    DATETIME_MIN_UTC,
    TimeBasedSimplePowerLoadConstraint,
)


class _FakeLoad:
    def __init__(self) -> None:
        self.name = "fake_load"
        self.efficiency_factor = 1.0
        self.current_command = LoadCommand(command="on", power_consign=1000)
        self.num_max_on_off = 4
        self.num_on_off = 0
        self.father_device = None
        self._constraints: list = []

    def get_possible_delta_power_from_piloted_devices_for_budget(self, slot_idx: int, add: bool) -> float:
        return 0.0

    def get_update_value_callback_for_constraint_class(self, _constraint):
        return None

    def is_off_grid(self) -> bool:
        return False

    def get_phase_amps_from_power_for_budgeting(self, power: float) -> list[float]:
        return [0.0, 0.0, 0.0]

    def get_phase_amps_from_power_for_piloted_budgeting(self, power: float) -> list[float]:
        return [0.0, 0.0, 0.0]

    def get_normalized_score(self, ct, time: datetime, score_span: float) -> float:
        return 0.5

    def is_time_sensitive(self) -> bool:
        return False


NOW = datetime(2026, 3, 11, 12, 0, 0, tzinfo=pytz.UTC)


def _make_constraint(load, start=None, end=None, type=CONSTRAINT_TYPE_FILLER_AUTO):
    """Helper to create a TimeBasedSimplePowerLoadConstraint with given start/end."""
    ct = TimeBasedSimplePowerLoadConstraint(
        time=NOW,
        load=load,
        type=type,
        start_of_constraint=start,
        end_of_constraint=end,
        initial_value=0.0,
        target_value=3600.0,
        power=1000.0,
    )
    return ct


# =============================================================================
# Single constraint - no following
# =============================================================================


def test_single_constraint_returns_false():
    """A lone constraint has no follower."""
    load = _FakeLoad()
    c1 = _make_constraint(load, start=NOW, end=NOW + timedelta(hours=2))
    load._constraints = [c1]
    assert c1.has_a_following_constraint() is False


# =============================================================================
# Constraint not found in load's list
# =============================================================================


def test_constraint_not_in_load_list_returns_false():
    """A constraint not in the load's _constraints returns False."""
    load = _FakeLoad()
    c1 = _make_constraint(load, start=NOW, end=NOW + timedelta(hours=2))
    c_orphan = _make_constraint(load, start=NOW, end=NOW + timedelta(hours=3))
    load._constraints = [c1]
    assert c_orphan.has_a_following_constraint() is False


# =============================================================================
# Last constraint in list - no following
# =============================================================================


def test_last_constraint_returns_false():
    """The last constraint in the list has no follower."""
    load = _FakeLoad()
    c1 = _make_constraint(load, start=NOW, end=NOW + timedelta(hours=1))
    c2 = _make_constraint(load, start=NOW + timedelta(hours=1), end=NOW + timedelta(hours=2))
    load._constraints = [c1, c2]
    assert c2.has_a_following_constraint() is False


# =============================================================================
# Two constraints with direct continuity (no gap)
# =============================================================================


def test_two_constraints_continuous():
    """Two constraints where next starts exactly when current ends -> True."""
    load = _FakeLoad()
    c1 = _make_constraint(load, start=NOW, end=NOW + timedelta(hours=1))
    c2 = _make_constraint(load, start=NOW + timedelta(hours=1), end=NOW + timedelta(hours=2))
    load._constraints = [c1, c2]
    assert c1.has_a_following_constraint() is True


def test_two_constraints_small_gap_within_solver_step():
    """Two constraints with a gap smaller than SOLVER_STEP_S -> True (continuous)."""
    load = _FakeLoad()
    gap = SOLVER_STEP_S - 10
    c1 = _make_constraint(load, start=NOW, end=NOW + timedelta(hours=1))
    c2 = _make_constraint(load, start=NOW + timedelta(hours=1, seconds=gap), end=NOW + timedelta(hours=2))
    load._constraints = [c1, c2]
    assert c1.has_a_following_constraint() is True


def test_two_constraints_overlap():
    """Two constraints where next starts BEFORE current ends -> True (overlap = continuous)."""
    load = _FakeLoad()
    c1 = _make_constraint(load, start=NOW, end=NOW + timedelta(hours=2))
    c2 = _make_constraint(load, start=NOW + timedelta(hours=1), end=NOW + timedelta(hours=3))
    load._constraints = [c1, c2]
    assert c1.has_a_following_constraint() is True


# =============================================================================
# Two constraints with a gap larger than SOLVER_STEP_S
# =============================================================================


def test_two_constraints_large_gap():
    """Two constraints with a gap larger than SOLVER_STEP_S -> False."""
    load = _FakeLoad()
    gap = SOLVER_STEP_S + 60
    c1 = _make_constraint(load, start=NOW, end=NOW + timedelta(hours=1))
    c2 = _make_constraint(load, start=NOW + timedelta(hours=1, seconds=gap), end=NOW + timedelta(hours=3))
    load._constraints = [c1, c2]
    assert c1.has_a_following_constraint() is False


# =============================================================================
# Special datetime values: DATETIME_MAX_UTC and DATETIME_MIN_UTC
# =============================================================================


def test_current_has_infinite_end():
    """Current constraint has DATETIME_MAX_UTC end (no end) -> True."""
    load = _FakeLoad()
    c1 = _make_constraint(load, start=NOW, end=DATETIME_MAX_UTC)
    c2 = _make_constraint(load, start=NOW + timedelta(hours=5), end=NOW + timedelta(hours=6))
    load._constraints = [c1, c2]
    assert c1.has_a_following_constraint() is True


def test_current_has_none_end():
    """Current constraint has None end (defaults to DATETIME_MAX_UTC) -> True."""
    load = _FakeLoad()
    c1 = _make_constraint(load, start=NOW, end=None)
    c2 = _make_constraint(load, start=NOW + timedelta(hours=5), end=NOW + timedelta(hours=6))
    load._constraints = [c1, c2]
    assert c1.has_a_following_constraint() is True


def test_next_has_min_utc_start():
    """Next constraint has DATETIME_MIN_UTC start (always active) -> True."""
    load = _FakeLoad()
    c1 = _make_constraint(load, start=NOW, end=NOW + timedelta(hours=1))
    c2 = _make_constraint(load, start=DATETIME_MIN_UTC, end=NOW + timedelta(hours=3))
    load._constraints = [c1, c2]
    assert c1.has_a_following_constraint() is True


def test_next_has_none_start():
    """Next constraint has None start (defaults to DATETIME_MIN_UTC) -> True."""
    load = _FakeLoad()
    c1 = _make_constraint(load, start=NOW, end=NOW + timedelta(hours=1))
    c2 = _make_constraint(load, start=None, end=NOW + timedelta(hours=3))
    load._constraints = [c1, c2]
    assert c1.has_a_following_constraint() is True


# =============================================================================
# Three constraints - middle one checks
# =============================================================================


def test_three_constraints_middle_has_following():
    """Middle constraint has a direct follower."""
    load = _FakeLoad()
    c1 = _make_constraint(load, start=NOW, end=NOW + timedelta(hours=1))
    c2 = _make_constraint(load, start=NOW + timedelta(hours=1), end=NOW + timedelta(hours=2))
    c3 = _make_constraint(load, start=NOW + timedelta(hours=2), end=NOW + timedelta(hours=3))
    load._constraints = [c1, c2, c3]
    assert c1.has_a_following_constraint() is True
    assert c2.has_a_following_constraint() is True
    assert c3.has_a_following_constraint() is False


def test_three_constraints_gap_after_middle():
    """Middle constraint has a gap before the third -> False for middle."""
    load = _FakeLoad()
    gap = SOLVER_STEP_S + 120
    c1 = _make_constraint(load, start=NOW, end=NOW + timedelta(hours=1))
    c2 = _make_constraint(load, start=NOW + timedelta(hours=1), end=NOW + timedelta(hours=2))
    c3 = _make_constraint(load, start=NOW + timedelta(hours=2, seconds=gap), end=NOW + timedelta(hours=4))
    load._constraints = [c1, c2, c3]
    assert c1.has_a_following_constraint() is True
    assert c2.has_a_following_constraint() is False
    assert c3.has_a_following_constraint() is False


# =============================================================================
# Empty constraints list
# =============================================================================


def test_empty_constraints_list():
    """Load has no constraints at all."""
    load = _FakeLoad()
    c1 = _make_constraint(load, start=NOW, end=NOW + timedelta(hours=1))
    load._constraints = []
    assert c1.has_a_following_constraint() is False


def test_none_constraints_list():
    """Load has _constraints = None (edge case)."""
    load = _FakeLoad()
    c1 = _make_constraint(load, start=NOW, end=NOW + timedelta(hours=1))
    load._constraints = None
    assert c1.has_a_following_constraint() is False
