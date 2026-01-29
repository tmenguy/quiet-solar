"""Tests for home_model/constraints.py - Load constraint functionality."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import pytz

from custom_components.quiet_solar.home_model.constraints import (
    LoadConstraint,
    MultiStepsPowerLoadConstraint,
    MultiStepsPowerLoadConstraintChargePercent,
    TimeBasedSimplePowerLoadConstraint,
    get_readable_date_string,
    DATETIME_MAX_UTC,
)
from custom_components.quiet_solar.home_model.commands import LoadCommand
from custom_components.quiet_solar.const import (
    CONSTRAINT_TYPE_FILLER_AUTO,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
    CONSTRAINT_TYPE_FILLER,
)


class _FakeLoad:
    def __init__(self) -> None:
        self.efficiency_factor = 1.0
        self.current_command = LoadCommand(command="on", power_consign=1000)
        self.num_max_on_off = 4
        self.num_on_off = 0

    def get_possible_delta_power_from_piloted_devices_for_budget(self, slot_idx: int, add: bool) -> float:
        return 0.0

    def get_update_value_callback_for_constraint_class(self, _constraint):
        return None

    def is_off_grid(self) -> bool:
        return False


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


def test_constraint_best_duration_extension():
    """Test best duration extension with push count."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    constraint = MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        power=1000,
        initial_value=0.0,
        target_value=1000.0,
    )
    constraint.pushed_count = 1
    extension = constraint.best_duration_extension_to_push_constraint(time, timedelta(seconds=10))
    assert extension >= timedelta(seconds=1200)


def test_multisteps_from_dict_strips_legacy_fields():
    """Test legacy fields are removed from power steps."""
    now = datetime.now(tz=pytz.UTC)
    data = {
        "start_of_constraint": now.isoformat(),
        "end_of_constraint": now.isoformat(),
        "power_steps": [
            {"command": "on", "power_consign": 1000, "phase_current": 10, "load_param": "x"},
        ]
    }
    kwargs = MultiStepsPowerLoadConstraint.from_dict_to_kwargs(data)
    assert kwargs["power_steps"][0].power_consign == 1000


def test_num_command_state_change_and_empty_segments():
    """Test command state change detection and empty segments."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    constraint = MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        power_steps=[LoadCommand(command="on", power_consign=1000)],
    )
    out_commands = [
        LoadCommand(command="on", power_consign=1000),
        None,
        None,
        LoadCommand(command="on", power_consign=1000),
    ]
    num, empty_cmds, start_switch, start_cmd = constraint._num_command_state_change_and_empty_inner_commands(
        out_commands, 0, 3
    )
    assert num >= 1
    assert empty_cmds
    assert start_cmd is not None
    assert start_switch is False


def test_replace_by_command_in_slots():
    """Test replacing commands in slots."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    constraint = MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        power_steps=[LoadCommand(command="on", power_consign=1000)],
    )
    out_commands = [None, None]
    out_power = [0.0, 0.0]
    durations = [900, 900]

    delta, dur, aborted = constraint._replace_by_command_in_slots(
        out_commands,
        out_power,
        durations,
        0,
        1,
        constraint._power_sorted_cmds[0],
    )
    assert aborted is False
    assert dur == sum(durations)

    delta, dur, aborted = constraint._replace_by_command_in_slots(
        out_commands,
        out_power,
        durations,
        0,
        1,
        None,
    )
    assert aborted is False


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


# =============================================================================
# Test constraint copy_to_other_type conversions
# =============================================================================

def _make_test_load() -> MagicMock:
    load = MagicMock()
    load.name = "TestLoad"
    load.efficiency_factor = 0.9
    load.get_update_value_callback_for_constraint_class.return_value = None
    load.is_off_grid.return_value = False
    return load


def _make_power_steps() -> list[LoadCommand]:
    return [
        LoadCommand(command="on", power_consign=500.0),
        LoadCommand(command="on", power_consign=1200.0),
    ]


def _make_constraint(
    constraint_cls,
    time_now: datetime,
    load: MagicMock,
    total_capacity_wh: float,
):
    common_kwargs = {
        "time": time_now,
        "load": load,
        "load_param": "power",
        "load_info": {"phase": "A", "index": 1},
        "from_user": True,
        "artificial_step_to_final_value": None,
        "type": CONSTRAINT_TYPE_MANDATORY_END_TIME,
        "degraded_type": CONSTRAINT_TYPE_FILLER,
        "start_of_constraint": time_now - timedelta(hours=2),
        "end_of_constraint": time_now + timedelta(hours=4),
        "support_auto": True,
        "always_end_at_end_of_constraint": False,
        "power_steps": _make_power_steps(),
    }

    if constraint_cls is MultiStepsPowerLoadConstraintChargePercent:
        common_kwargs.update(
            {
                "total_capacity_wh": total_capacity_wh,
                "initial_value": 10.0,
                "current_value": 25.0,
                "target_value": 80.0,
            }
        )
    elif constraint_cls is TimeBasedSimplePowerLoadConstraint:
        common_kwargs.update(
            {
                "initial_value": 600.0,
                "current_value": 1200.0,
                "target_value": 3600.0,
            }
        )
    else:
        common_kwargs.update(
            {
                "initial_value": 100.0,
                "current_value": 250.0,
                "target_value": 800.0,
            }
        )

    return constraint_cls(**common_kwargs)


@pytest.mark.parametrize(
    ("source_cls", "target_cls"),
    [
        (MultiStepsPowerLoadConstraint, MultiStepsPowerLoadConstraintChargePercent),
        (MultiStepsPowerLoadConstraint, TimeBasedSimplePowerLoadConstraint),
        (MultiStepsPowerLoadConstraintChargePercent, MultiStepsPowerLoadConstraint),
        (MultiStepsPowerLoadConstraintChargePercent, TimeBasedSimplePowerLoadConstraint),
        (TimeBasedSimplePowerLoadConstraint, MultiStepsPowerLoadConstraint),
        (TimeBasedSimplePowerLoadConstraint, MultiStepsPowerLoadConstraintChargePercent),
    ],
)
def test_copy_to_other_type_round_trip_fields(source_cls, target_cls):
    """Test copy_to_other_type between constraint types."""
    time_now = datetime(2026, 1, 23, 12, 0, tzinfo=pytz.UTC)
    load = _make_test_load()
    common_additional_kwargs = {"total_capacity_wh": 50000.0}

    source = _make_constraint(source_cls, time_now, load, common_additional_kwargs["total_capacity_wh"])
    converted = source.copy_to_other_type(time_now, target_cls, common_additional_kwargs)

    assert isinstance(converted, target_cls)
    assert converted.load is load

    assert converted._type == source._type
    assert converted._degraded_type == source._degraded_type
    assert converted.load_param == source.load_param
    assert converted.load_info == source.load_info
    assert converted.from_user == source.from_user
    assert converted.artificial_step_to_final_value == source.artificial_step_to_final_value
    assert converted.support_auto == source.support_auto
    assert converted.start_of_constraint == source.start_of_constraint
    assert converted.end_of_constraint == source.end_of_constraint

    expected_always_end = (
        True if target_cls is TimeBasedSimplePowerLoadConstraint else source.always_end_at_end_of_constraint
    )
    assert converted.always_end_at_end_of_constraint == expected_always_end

    if target_cls is MultiStepsPowerLoadConstraintChargePercent:
        assert converted.total_capacity_wh == common_additional_kwargs["total_capacity_wh"]

    assert converted.to_dict()["power_steps"] == source.to_dict()["power_steps"]

    assert converted.convert_target_value_to_energy(converted.target_value) == pytest.approx(
        source.convert_target_value_to_energy(source.target_value)
    )
    assert converted.convert_target_value_to_energy(converted.initial_value) == pytest.approx(
        source.convert_target_value_to_energy(source.initial_value)
    )
    assert converted.convert_target_value_to_energy(converted.current_value) == pytest.approx(
        source.convert_target_value_to_energy(source.current_value)
    )
