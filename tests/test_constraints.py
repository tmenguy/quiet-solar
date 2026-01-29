"""Tests for home_model/constraints.py - Load constraint functionality."""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import pytz
import numpy as np

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
    SOLVER_STEP_S,
)


class _FakeLoad:
    def __init__(self) -> None:
        self.name = "fake_load"
        self.efficiency_factor = 1.0
        self.current_command = LoadCommand(command="on", power_consign=1000)
        self.num_max_on_off = 4
        self.num_on_off = 0
        self.father_device = None

    def get_possible_delta_power_from_piloted_devices_for_budget(self, slot_idx: int, add: bool) -> float:
        return 100.0 if add else 50.0

    def get_update_value_callback_for_constraint_class(self, _constraint):
        return None

    def is_off_grid(self) -> bool:
        return False

    def get_phase_amps_from_power_for_budgeting(self, power: float) -> list[float]:
        amps = power / 230.0
        return [amps, amps, amps]

    def get_phase_amps_from_power_for_piloted_budgeting(self, power: float) -> list[float]:
        amps = power / 230.0
        return [amps, amps, amps]

    def get_normalized_score(self, ct, time: datetime, score_span: float) -> float:
        return 0.5

    def is_time_sensitive(self) -> bool:
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


def test_get_readable_date_string_tomorrow():
    """Test readable date string for tomorrow."""
    local_now = datetime.now(tz=pytz.UTC).astimezone(tz=None)
    local_today = datetime(local_now.year, local_now.month, local_now.day, tzinfo=local_now.tzinfo)
    target_local = local_today + timedelta(days=1, hours=1)
    target_utc = target_local.astimezone(pytz.UTC)
    result = get_readable_date_string(target_utc)
    assert "tomorrow" in result.lower()


def test_get_readable_date_string_future_standalone():
    """Test readable date string for future standalone format."""
    local_now = datetime.now(tz=pytz.UTC).astimezone(tz=None)
    local_today = datetime(local_now.year, local_now.month, local_now.day, tzinfo=local_now.tzinfo)
    target_local = local_today + timedelta(days=2, hours=3)
    target_utc = target_local.astimezone(pytz.UTC)
    result = get_readable_date_string(target_utc, for_small_standalone=True)
    assert "\n" in result


def test_constraint_names_and_hashing():
    """Test constraint naming, load info, and hashing."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    constraint = MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        power=1000,
        load_param="power",
        load_info={"device": "test"},
        artificial_step_to_final_value=1200,
    )

    assert "power" in constraint.name
    assert "device" in constraint.name
    assert "1200" in constraint.name
    assert "device" in constraint.stable_name
    constraint.add_or_update_load_info("extra", "value")
    assert "extra" in constraint.load_info
    assert isinstance(hash(constraint), int)


def test_constraint_reset_load_param_and_offgrid_type():
    """Test reset_load_param and off-grid type behavior."""
    time = datetime.now(tz=pytz.UTC)

    class _OffGridLoad(_FakeLoad):
        def is_off_grid(self) -> bool:
            return True

    load = _OffGridLoad()
    constraint = MultiStepsPowerLoadConstraint(time=time, load=load, power=1000)
    constraint.type = CONSTRAINT_TYPE_MANDATORY_END_TIME
    assert constraint.type == constraint._degraded_type

    constraint.reset_load_param("amps")
    assert constraint.load_param == "amps"


def test_constraint_equality_and_no_current():
    """Test equality and eq_no_current comparisons."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    constraint = MultiStepsPowerLoadConstraint(time=time, load=load, power=1000)
    other = MultiStepsPowerLoadConstraint(time=time, load=load, power=1000)
    other.current_value = constraint.current_value
    assert constraint == other
    other.current_value = constraint.current_value + 1
    assert constraint.eq_no_current(other)
    assert constraint.eq_no_current(None) is False


def test_constraint_type_and_before_battery():
    """Test type setter and before battery toggling."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    constraint = MultiStepsPowerLoadConstraint(time=time, load=load, power=1000)

    constraint.type = CONSTRAINT_TYPE_MANDATORY_END_TIME
    assert constraint.type == CONSTRAINT_TYPE_MANDATORY_END_TIME

    constraint.is_before_battery = True
    assert constraint.is_before_battery is True
    constraint.is_before_battery = False
    assert constraint.is_before_battery is False


def test_constraint_score_and_percent_completion():
    """Test score and percent completion branches."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    constraint = MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        power=1000,
        initial_value=10.0,
        target_value=5.0,
        current_value=7.0,
        from_user=True,
    )
    assert constraint.score(time) > 0
    assert constraint.get_percent_completion(time) > 0

    constraint.current_value = None
    assert constraint.get_percent_completion(time) is None

    constraint.current_value = 1.0
    constraint.target_value = None
    assert constraint.get_percent_completion(time) is None

    constraint.target_value = 1.0
    constraint._internal_initial_value = None
    assert constraint.get_percent_completion(time) is None

    constraint._internal_initial_value = 1.0
    assert constraint.get_percent_completion(time) is None


def test_constraint_readable_name_and_active_period():
    """Test readable name and active period logic."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    constraint = MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        power=2500,
        target_value=2500,
        load_param="energy",
        load_info={"source": "solar"},
        support_auto=True,
        end_of_constraint=time + timedelta(hours=2),
    )

    readable = constraint.get_readable_name_for_load()
    assert "kWh" in readable
    assert "energy" in readable
    assert "," not in readable

    assert constraint.is_constraint_active_for_time_period(time) is True
    assert constraint.is_constraint_active_for_time_period(time, time - timedelta(minutes=1)) is True


def test_constraint_readable_name_multiple_info():
    """Test readable name with multiple load_info entries."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    constraint = MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        power=1000,
        load_info={"a": "1", "b": "2"},
    )
    readable = constraint.get_readable_name_for_load()
    assert "," in readable


def test_constraint_met_conditions():
    """Test constraint met logic branches."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    constraint = MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        power=1000,
        target_value=100.0,
        current_value=100.0,
        always_end_at_end_of_constraint=True,
        end_of_constraint=time - timedelta(seconds=1),
    )
    assert constraint.is_constraint_met(time) is True

    constraint.always_end_at_end_of_constraint = False
    constraint.target_value = None
    assert constraint.is_constraint_met(time) is False

    constraint.target_value = 100.0
    constraint.current_value = 99.5
    assert constraint.is_constraint_met(time) is True


def test_constraint_active_period_branches():
    """Test active period with various end conditions."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    constraint = MultiStepsPowerLoadConstraint(time=time, load=load, power=1000)
    constraint.current_start_of_constraint = time + timedelta(hours=1)
    assert constraint.is_constraint_active_for_time_period(time, time) is False

    constraint.current_start_of_constraint = time
    constraint.end_of_constraint = DATETIME_MAX_UTC
    constraint.target_value = 100.0
    constraint.current_value = 0.0
    assert constraint.is_constraint_active_for_time_period(time) is True


@pytest.mark.asyncio
async def test_constraint_update_with_callback_and_compute_value():
    """Test update with callback and compute_value."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    constraint = MultiStepsPowerLoadConstraint(time=time, load=load, power=1000)
    constraint.last_value_update = time - timedelta(hours=1)
    constraint.current_value = 0.0

    async def _cb(_self, _time):
        return 5.0, False

    constraint._update_value_callback = _cb
    assert await constraint.update(time) is False

    constraint._update_value_callback = None
    assert await constraint.update(time) is False


def test_constraint_new_from_saved_dict_import_failure():
    """Test new_from_saved_dict handles import failures."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    data = {"qs_class_type": "DoesNotExist"}

    with patch("custom_components.quiet_solar.home_model.constraints.importlib.import_module", side_effect=ImportError):
        assert LoadConstraint.new_from_saved_dict(time, load, data) is None


def test_constraint_new_from_saved_dict_and_copy():
    """Test new_from_saved_dict and copy_to_other_type."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    constraint = MultiStepsPowerLoadConstraint(time=time, load=load, power=1000)
    data = constraint.to_dict()
    data.pop("qs_class_type", None)
    assert LoadConstraint.new_from_saved_dict(time, load, data) is None

    copied = constraint.copy_to_other_type(time, MultiStepsPowerLoadConstraint, {})
    assert copied is not None


def test_time_based_target_string_and_compute_value():
    """Test time-based readable target values and compute_value."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    constraint = TimeBasedSimplePowerLoadConstraint(
        time=time,
        load=load,
        power=1000,
        target_value=5 * 3600,
    )
    assert "h" in constraint._get_readable_target_value_string()

    constraint.target_value = 3600 + 120
    assert "h" in constraint._get_readable_target_value_string()

    constraint.target_value = 120
    assert "mn" in constraint._get_readable_target_value_string()

    constraint.last_value_update = time - timedelta(seconds=10)
    constraint.current_value = 0.0
    assert constraint.compute_value(time) is not None

    load.current_command = None
    assert constraint.compute_value(time) is None


def test_time_based_is_constraint_met_thresholds():
    """Test time-based constraint met thresholds."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    constraint = TimeBasedSimplePowerLoadConstraint(
        time=time,
        load=load,
        power=1000,
        target_value=100.0,
        current_value=90.0,
        end_of_constraint=time - timedelta(seconds=1),
    )
    assert constraint.is_constraint_met(time) is True

    constraint.target_value = None
    constraint.end_of_constraint = time + timedelta(hours=1)
    assert constraint.is_constraint_met(time) is False


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


def test_adapt_commands_removes_short_start():
    """Test adapt_commands removes short initial empty period."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    load.num_max_on_off = 2
    load.num_on_off = 0

    constraint = MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        power_steps=[LoadCommand(command="on", power_consign=1000)],
    )

    out_commands = [
        None,
        LoadCommand(command="on", power_consign=1000),
        None,
        LoadCommand(command="on", power_consign=1000),
    ]
    out_power = [500.0, 1000.0, 0.0, 1000.0]
    power_slots_duration_s = [SOLVER_STEP_S] * len(out_commands)

    constraint._adapt_commands(out_commands, out_power, power_slots_duration_s, 0.0, 0, len(out_commands) - 1)

    assert out_commands[0] is not None
    assert out_commands[0].power_consign == 1000


def test_adapt_power_steps_budgeting_low_level_limits_commands():
    """Test command filtering with available amps."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    load.father_device = SimpleNamespace(available_amps_for_group=[[6.0, 6.0, 6.0]])

    constraint = MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        power_steps=[
            LoadCommand(command="on", power_consign=1150),  # 5A
            LoadCommand(command="on", power_consign=4600),  # 20A
        ],
    )

    commands = constraint.adapt_power_steps_budgeting_low_level(slot_idx=0)
    assert len(commands) == 1
    assert commands[0].power_consign == 1150


def test_adapt_power_steps_budgeting_empty_and_existing():
    """Test adapt_power_steps_budgeting for empty and existing commands."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    load.father_device = SimpleNamespace(available_amps_for_group=[[20.0, 20.0, 20.0]])

    constraint = MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        power_steps=[LoadCommand(command="on", power_consign=2300)],
    )

    commands = [None]
    power_sorted_cmds, is_empty, piloted_delta = constraint.adapt_power_steps_budgeting(
        slot_idx=0, commands=commands, for_add=True
    )
    assert is_empty is True
    assert piloted_delta > 0
    assert power_sorted_cmds

    power_sorted_cmds, is_empty, piloted_delta = constraint.adapt_power_steps_budgeting(
        slot_idx=0, commands=commands, for_add=False
    )
    assert is_empty is True
    assert piloted_delta == 0.0
    assert power_sorted_cmds == []

    commands = [LoadCommand(command="on", power_consign=2300)]
    power_sorted_cmds, is_empty, piloted_delta = constraint.adapt_power_steps_budgeting(
        slot_idx=0, commands=commands, for_add=True
    )
    assert is_empty is False
    assert piloted_delta > 0


def test_adapt_repartition_add_energy_support_auto():
    """Test adapt_repartition adds energy with support_auto."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()
    load.father_device = SimpleNamespace(available_amps_for_group=[[20.0, 20.0, 20.0]] * 3)

    constraint = MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        power_steps=[
            LoadCommand(command="on", power_consign=500),
            LoadCommand(command="on", power_consign=1000),
        ],
        support_auto=True,
        current_value=100.0,
        target_value=100.0,
    )

    power_slots_duration_s = np.array([SOLVER_STEP_S, SOLVER_STEP_S, SOLVER_STEP_S], dtype=np.float64)
    existing_commands = [
        LoadCommand(command="on", power_consign=500),
        None,
        LoadCommand(command="on", power_consign=500),
    ]

    out_constraint, done, changed, energy_delta, out_commands, out_delta_power = constraint.adapt_repartition(
        first_slot=0,
        last_slot=1,
        energy_delta=200.0,
        power_slots_duration_s=power_slots_duration_s,
        existing_commands=existing_commands,
        allow_change_state=True,
        time=time,
    )

    assert out_constraint is not None
    assert changed is True
    assert any(cmd is not None for cmd in out_commands)


def test_adapt_repartition_mandatory_reduce_no_change():
    """Test adapt_repartition does nothing for mandatory reduce."""
    time = datetime.now(tz=pytz.UTC)
    load = _FakeLoad()

    constraint = MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        power_steps=[LoadCommand(command="on", power_consign=500)],
        type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
    )

    power_slots_duration_s = np.array([SOLVER_STEP_S, SOLVER_STEP_S], dtype=np.float64)
    existing_commands = [LoadCommand(command="on", power_consign=500), None]

    _, done, changed, _, out_commands, _ = constraint.adapt_repartition(
        first_slot=0,
        last_slot=1,
        energy_delta=-100.0,
        power_slots_duration_s=power_slots_duration_s,
        existing_commands=existing_commands,
        allow_change_state=True,
        time=time,
    )

    assert done is False
    assert changed is False
    assert all(cmd is None for cmd in out_commands)


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
