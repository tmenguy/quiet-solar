"""Tests for home_model/constraints.py - Load constraint functionality."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import pytz

from custom_components.quiet_solar.home_model.constraints import (
    LoadConstraint,
    get_readable_date_string,
    DATETIME_MAX_UTC,
)
from custom_components.quiet_solar.const import (
    CONSTRAINT_TYPE_FILLER_AUTO,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
    CONSTRAINT_TYPE_FILLER,
)


# =============================================================================
# Test get_readable_date_string
# =============================================================================

def test_get_readable_date_string_none():
    """Test readable date string for None."""
    result = get_readable_date_string(None)
    assert result == ""


def test_get_readable_date_string_none_standalone():
    """Test readable date string for None in standalone mode."""
    result = get_readable_date_string(None, for_small_standalone=True)
    assert result == "--:--"


def test_get_readable_date_string_max_utc():
    """Test readable date string for DATETIME_MAX_UTC."""
    result = get_readable_date_string(DATETIME_MAX_UTC)
    assert result == ""


def test_get_readable_date_string_max_utc_standalone():
    """Test readable date string for DATETIME_MAX_UTC in standalone mode."""
    result = get_readable_date_string(DATETIME_MAX_UTC, for_small_standalone=True)
    assert result == "--:--"


def test_get_readable_date_string_today():
    """Test readable date string for today."""
    now = datetime.now(tz=pytz.UTC)
    # Add a few hours to now so it's still today
    test_time = now + timedelta(hours=2)
    result = get_readable_date_string(test_time)
    assert "today" in result.lower() or ":" in result


def test_get_readable_date_string_standalone_near():
    """Test readable date string in standalone mode for near time."""
    now = datetime.now(tz=pytz.UTC)
    test_time = now + timedelta(hours=2)
    result = get_readable_date_string(test_time, for_small_standalone=True)
    # Should be just HH:MM format
    assert ":" in result


# =============================================================================
# Test LoadConstraint initialization
# =============================================================================

def test_load_constraint_init_defaults():
    """Test LoadConstraint with default values."""
    time = datetime.now(tz=pytz.UTC)
    constraint = LoadConstraint(time=time)

    assert constraint.load is None
    assert constraint.load_param is None
    assert constraint.load_info is None
    assert constraint.from_user is False
    assert constraint._type == CONSTRAINT_TYPE_FILLER_AUTO
    assert constraint.support_auto is False


def test_load_constraint_init_with_type():
    """Test LoadConstraint with specific type."""
    time = datetime.now(tz=pytz.UTC)
    constraint = LoadConstraint(
        time=time,
        type=CONSTRAINT_TYPE_MANDATORY_END_TIME
    )

    assert constraint._type == CONSTRAINT_TYPE_MANDATORY_END_TIME


def test_load_constraint_init_with_load():
    """Test LoadConstraint with mock load."""
    time = datetime.now(tz=pytz.UTC)
    mock_load = MagicMock()
    mock_load.name = "TestLoad"

    constraint = LoadConstraint(
        time=time,
        load=mock_load,
        load_param="power"
    )

    assert constraint.load is mock_load
    assert constraint.load_param == "power"


def test_load_constraint_init_with_values():
    """Test LoadConstraint with initial and target values."""
    time = datetime.now(tz=pytz.UTC)

    constraint = LoadConstraint(
        time=time,
        initial_value=10.0,
        target_value=100.0
    )

    assert constraint.target_value == 100.0


def test_load_constraint_init_with_times():
    """Test LoadConstraint with start and end times."""
    time = datetime.now(tz=pytz.UTC)
    start = time + timedelta(hours=1)
    end = time + timedelta(hours=5)

    constraint = LoadConstraint(
        time=time,
        start_of_constraint=start,
        end_of_constraint=end
    )

    assert constraint.start_of_constraint == start
    assert constraint.end_of_constraint == end


def test_load_constraint_from_user():
    """Test LoadConstraint created from user."""
    time = datetime.now(tz=pytz.UTC)

    constraint = LoadConstraint(
        time=time,
        from_user=True
    )

    assert constraint.from_user is True


def test_load_constraint_support_auto():
    """Test LoadConstraint with auto support."""
    time = datetime.now(tz=pytz.UTC)

    constraint = LoadConstraint(
        time=time,
        support_auto=True
    )

    assert constraint.support_auto is True


def test_load_constraint_degraded_type():
    """Test LoadConstraint with degraded type."""
    time = datetime.now(tz=pytz.UTC)

    constraint = LoadConstraint(
        time=time,
        type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        degraded_type=CONSTRAINT_TYPE_FILLER
    )

    assert constraint._type == CONSTRAINT_TYPE_MANDATORY_END_TIME
    assert constraint._degraded_type == CONSTRAINT_TYPE_FILLER


def test_load_constraint_artificial_step():
    """Test LoadConstraint with artificial step."""
    time = datetime.now(tz=pytz.UTC)

    constraint = LoadConstraint(
        time=time,
        artificial_step_to_final_value=5
    )

    assert constraint.artificial_step_to_final_value == 5


def test_load_constraint_always_end_at_end():
    """Test LoadConstraint with always_end_at_end_of_constraint."""
    time = datetime.now(tz=pytz.UTC)

    constraint = LoadConstraint(
        time=time,
        always_end_at_end_of_constraint=True
    )

    assert constraint.always_end_at_end_of_constraint is True


def test_load_constraint_load_info():
    """Test LoadConstraint with load_info dictionary."""
    time = datetime.now(tz=pytz.UTC)
    info = {"key": "value", "number": 42}

    constraint = LoadConstraint(
        time=time,
        load_info=info
    )

    assert constraint.load_info == info
    assert constraint.load_info["key"] == "value"
    assert constraint.load_info["number"] == 42


# =============================================================================
# Test Constraint Types
# =============================================================================

def test_constraint_type_filler_auto():
    """Test CONSTRAINT_TYPE_FILLER_AUTO."""
    assert CONSTRAINT_TYPE_FILLER_AUTO is not None


def test_constraint_type_mandatory_asap():
    """Test CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE."""
    assert CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE is not None


def test_constraint_type_mandatory_end():
    """Test CONSTRAINT_TYPE_MANDATORY_END_TIME."""
    assert CONSTRAINT_TYPE_MANDATORY_END_TIME is not None


def test_constraint_type_before_battery():
    """Test CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN."""
    assert CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN is not None


def test_constraint_type_filler():
    """Test CONSTRAINT_TYPE_FILLER."""
    assert CONSTRAINT_TYPE_FILLER is not None
