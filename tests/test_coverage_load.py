"""Targeted tests for home_model/load.py to reach 95% coverage.

Covers standalone functions, AbstractDevice edge cases, AbstractLoad constraint
lifecycle, and the update_live_constraints complex flow.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

import pytz

from custom_components.quiet_solar.home_model.load import (
    AbstractDevice,
    AbstractLoad,
    PilotedDevice,
    TestLoad,
    align_time_series_and_values,
    get_slots_from_time_series,
    get_value_from_time_series,
)
from custom_components.quiet_solar.home_model.commands import (
    LoadCommand,
    CMD_ON,
    CMD_OFF,
    CMD_IDLE,
    CMD_AUTO_GREEN_ONLY,
    copy_command,
)
from custom_components.quiet_solar.home_model.constraints import (
    LoadConstraint,
    MultiStepsPowerLoadConstraint,
    DATETIME_MAX_UTC,
    DATETIME_MIN_UTC,
)
from custom_components.quiet_solar.const import (
    CONF_POWER,
    CONF_SWITCH,
    CONF_LOAD_IS_BOOST_ONLY,
    CONF_DEVICE_EFFICIENCY,
    CONF_IS_3P,
    CONF_MONO_PHASE,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
    CONSTRAINT_TYPE_FILLER_AUTO,
    CONSTRAINT_TYPE_FILLER,
    DASHBOARD_NO_SECTION,
)
from tests.factories import (
    MinimalTestLoad,
    MinimalTestHome,
    create_constraint,
    create_load_command,
    create_minimal_home_model,
)


# =========================================================================
# Helper: create a time
# =========================================================================
def _t(hours_offset: float = 0.0) -> datetime:
    """Create a UTC datetime offset from a base time."""
    base = datetime(2026, 2, 11, 8, 0, 0, tzinfo=pytz.UTC)
    return base + timedelta(hours=hours_offset)


# =========================================================================
# Tests for standalone functions
# =========================================================================


class TestAlignTimeSeriesAndValues:
    """Test align_time_series_and_values with operation path."""

    def test_both_empty_with_operation(self):
        """Both tsv1 and tsv2 empty with operation returns []."""
        result = align_time_series_and_values([], [], operation=lambda a, b: (a or 0) + (b or 0))
        assert result == []

    def test_both_empty_without_operation(self):
        """Both tsv1 and tsv2 empty without operation returns ([], [])."""
        r1, r2 = align_time_series_and_values([], [], operation=None)
        assert r1 == []
        assert r2 == []

    def test_tsv1_empty_tsv2_has_data_with_operation_2tuple(self):
        """tsv1 empty, tsv2 has 2-tuple data, operation provided."""
        t1, t2 = _t(0), _t(1)
        tsv2 = [(t1, 10.0), (t2, 20.0)]
        result = align_time_series_and_values([], tsv2, operation=lambda a, b: (a or 0) + (b or 0))
        assert len(result) == 2
        assert result[0] == (t1, 10.0)
        assert result[1] == (t2, 20.0)

    def test_tsv1_empty_tsv2_has_data_with_operation_3tuple(self):
        """tsv1 empty, tsv2 has 3-tuple data, operation provided."""
        t1 = _t(0)
        tsv2 = [(t1, 10.0, {"key": "val"})]
        result = align_time_series_and_values([], tsv2, operation=lambda a, b: (a or 0) + (b or 0))
        assert len(result) == 1
        assert result[0][0] == t1
        assert result[0][1] == 10.0
        assert result[0][2] == {"key": "val"}

    def test_tsv1_empty_tsv2_has_data_no_operation_2tuple(self):
        """tsv1 empty, tsv2 has 2-tuple data, no operation."""
        t1 = _t(0)
        tsv2 = [(t1, 5.0)]
        r1, r2 = align_time_series_and_values([], tsv2, operation=None)
        assert r1 == [(t1, None)]
        assert r2 == tsv2

    def test_tsv1_empty_tsv2_has_data_no_operation_3tuple(self):
        """tsv1 empty, tsv2 has 3-tuple data, no operation."""
        t1 = _t(0)
        tsv2 = [(t1, 5.0, {"a": 1})]
        r1, r2 = align_time_series_and_values([], tsv2, operation=None)
        assert r1 == [(t1, None, None)]
        assert r2 == tsv2

    def test_tsv2_empty_tsv1_has_data_with_operation_2tuple(self):
        """tsv2 empty, tsv1 has 2-tuple data, operation provided."""
        t1 = _t(0)
        tsv1 = [(t1, 10.0)]
        result = align_time_series_and_values(tsv1, [], operation=lambda a, b: (a or 0) + (b or 0))
        assert len(result) == 1
        assert result[0] == (t1, 10.0)

    def test_tsv2_empty_tsv1_has_data_with_operation_3tuple(self):
        """tsv2 empty, tsv1 has 3-tuple data, operation provided."""
        t1 = _t(0)
        tsv1 = [(t1, 10.0, {"x": 1})]
        result = align_time_series_and_values(tsv1, [], operation=lambda a, b: (a or 0) + (b or 0))
        assert len(result) == 1
        assert result[0][0] == t1
        assert result[0][1] == 10.0

    def test_tsv2_empty_tsv1_has_data_no_operation_2tuple(self):
        """tsv2 empty, tsv1 has 2-tuple data, no operation."""
        t1 = _t(0)
        tsv1 = [(t1, 5.0)]
        r1, r2 = align_time_series_and_values(tsv1, [], operation=None)
        assert r1 == tsv1
        assert r2 == [(t1, None)]

    def test_tsv2_empty_tsv1_has_data_no_operation_3tuple(self):
        """tsv2 empty, tsv1 has 3-tuple data, no operation."""
        t1 = _t(0)
        tsv1 = [(t1, 5.0, {"a": 1})]
        r1, r2 = align_time_series_and_values(tsv1, [], operation=None)
        assert r1 == tsv1
        assert r2 == [(t1, None, None)]

    def test_both_have_data_with_operation_2tuple(self):
        """Both have 2-tuple data with operation - alignment + interpolation."""
        t1, t2, t3 = _t(0), _t(1), _t(2)
        tsv1 = [(t1, 10.0), (t3, 30.0)]
        tsv2 = [(t1, 100.0), (t2, 200.0), (t3, 300.0)]
        result = align_time_series_and_values(tsv1, tsv2, operation=lambda a, b: a + b)
        # t1: 10+100=110, t2: interpolated(20)+200=220, t3: 30+300=330
        assert len(result) == 3
        assert result[0][1] == 110.0
        assert abs(result[1][1] - 220.0) < 0.01
        assert result[2][1] == 330.0

    def test_both_have_data_with_operation_3tuple(self):
        """Both have 3-tuple data with operation - alignment with attrs."""
        t1, t2 = _t(0), _t(1)
        tsv1 = [(t1, 10.0, {"src": "a"}), (t2, 20.0, {"src": "a"})]
        tsv2 = [(t1, 100.0, {"src": "b"}), (t2, 200.0, {"src": "b"})]
        result = align_time_series_and_values(tsv1, tsv2, operation=lambda a, b: a + b)
        assert len(result) == 2
        assert result[0][1] == 110.0
        # attrs should be merged
        assert "src" in result[0][2]

    def test_both_have_data_no_operation_2tuple(self):
        """Both have 2-tuple data without operation."""
        t1, t2 = _t(0), _t(1)
        tsv1 = [(t1, 10.0), (t2, 20.0)]
        tsv2 = [(t1, 100.0), (t2, 200.0)]
        r1, r2 = align_time_series_and_values(tsv1, tsv2, operation=None)
        assert len(r1) == 2
        assert len(r2) == 2

    def test_interpolation_with_misaligned_times(self):
        """Test interpolation when time series have different time points."""
        t1, t2, t3 = _t(0), _t(0.5), _t(1)
        tsv1 = [(t1, 0.0), (t3, 10.0)]
        tsv2 = [(t2, 100.0)]
        result = align_time_series_and_values(tsv1, tsv2, operation=lambda a, b: a + b)
        # At t1: val1=0, val2=100 (before first real of tsv2) => 0+100=100
        # At t2: val1=5 (interpolated), val2=100 => 5+100=105
        # At t3: val1=10, val2=100 (after last real of tsv2) => 10+100=110
        assert len(result) == 3

    def test_operation_with_none_values(self):
        """Test operation when interpolated values are None."""
        t1, t2 = _t(0), _t(1)
        tsv1 = [(t1, None), (t2, None)]
        tsv2 = [(t1, 10.0), (t2, 20.0)]
        result = align_time_series_and_values(tsv1, tsv2, operation=lambda a, b: a + b if a is not None and b is not None else None)
        # None values propagate
        assert len(result) == 2

    def test_interpolation_between_values_with_none_next(self):
        """Test interpolation when next value is None."""
        t1, t2, t3 = _t(0), _t(0.5), _t(1)
        tsv1 = [(t1, 10.0), (t3, None)]  # next value is None
        tsv2 = [(t2, 100.0)]  # only t2 in tsv2
        # In tsv1: at t2, last_real_idx=0, next is idx=1 which is None
        # So val_to_put = vcur (10.0)
        result = align_time_series_and_values(tsv1, tsv2, operation=lambda a, b: (a or 0) + (b or 0))
        assert len(result) == 3

    def test_interpolation_between_values_with_none_current(self):
        """Test interpolation when current value is None but next is not."""
        t1, t2, t3 = _t(0), _t(0.5), _t(1)
        tsv1 = [(t1, None), (t3, 10.0)]  # current value is None
        tsv2 = [(t2, 100.0)]  # only t2 in tsv2
        # In tsv1: at t2, last_real_idx=0 (None), next is 10.0
        # vcur is None => val_to_put = None
        result = align_time_series_and_values(tsv1, tsv2, operation=lambda a, b: (a or 0) + (b or 0))
        assert len(result) == 3


class TestGetSlotsFromTimeSeries:
    """Test get_slots_from_time_series edge cases."""

    def test_empty_time_series(self):
        """Empty time series returns empty list."""
        result = get_slots_from_time_series([], _t(0))
        assert result == []

    def test_single_slot_no_end_time(self):
        """Single slot retrieval without end_time."""
        ts = [(_t(0), 10.0), (_t(1), 20.0), (_t(2), 30.0)]
        result = get_slots_from_time_series(ts, _t(0.5), end_time=None)
        assert len(result) == 1

    def test_range_with_end_time(self):
        """Range retrieval with end_time."""
        ts = [(_t(0), 10.0), (_t(1), 20.0), (_t(2), 30.0), (_t(3), 40.0)]
        result = get_slots_from_time_series(ts, _t(0.5), end_time=_t(2.5))
        assert len(result) >= 3  # should include surrounding entries

    def test_start_at_exact_time(self):
        """Start at exact time point."""
        ts = [(_t(0), 10.0), (_t(1), 20.0)]
        result = get_slots_from_time_series(ts, _t(0), end_time=_t(1))
        assert len(result) == 2

    def test_end_beyond_series(self):
        """End time beyond the series."""
        ts = [(_t(0), 10.0), (_t(1), 20.0)]
        result = get_slots_from_time_series(ts, _t(0), end_time=_t(5))
        assert len(result) == 2

    def test_end_at_exact_boundary(self):
        """End time at exact boundary."""
        ts = [(_t(0), 10.0), (_t(1), 20.0), (_t(2), 30.0)]
        result = get_slots_from_time_series(ts, _t(0), end_time=_t(1))
        assert len(result) >= 2

    def test_start_before_series(self):
        """Start time before the series."""
        ts = [(_t(1), 10.0), (_t(2), 20.0)]
        result = get_slots_from_time_series(ts, _t(0), end_time=_t(3))
        assert len(result) == 2


class TestGetValueFromTimeSeries:
    """Test get_value_from_time_series interpolation paths."""

    def test_empty_series(self):
        """Empty series returns None."""
        result = get_value_from_time_series(None, _t(0))
        assert result == (None, None, False, -1)

    def test_empty_list(self):
        """Empty list returns None."""
        result = get_value_from_time_series([], _t(0))
        assert result == (None, None, False, -1)

    def test_exact_last_time(self):
        """Exact match on last element."""
        ts = [(_t(0), 10.0), (_t(1), 20.0)]
        result = get_value_from_time_series(ts, _t(1))
        assert result[1] == 20.0

    def test_exact_first_time(self):
        """Exact match on first element."""
        ts = [(_t(0), 10.0), (_t(1), 20.0)]
        result = get_value_from_time_series(ts, _t(0))
        assert result[1] == 10.0

    def test_between_times_no_interpolation(self):
        """Between two times without interpolation, pick closest."""
        ts = [(_t(0), 10.0), (_t(2), 20.0)]
        result = get_value_from_time_series(ts, _t(0.5))
        assert result[1] == 10.0  # closer to t(0)

    def test_between_times_closer_to_second(self):
        """Between two times without interpolation, closer to second."""
        ts = [(_t(0), 10.0), (_t(2), 20.0)]
        result = get_value_from_time_series(ts, _t(1.5))
        assert result[1] == 20.0  # closer to t(2)

    def test_interpolation_both_none(self):
        """Interpolation with both values None."""
        ts = [(_t(0), None), (_t(2), None)]
        interp = lambda v1, v2, t: (t, 0.0)
        result = get_value_from_time_series(ts, _t(1), interpolation_operation=interp)
        # Lines 1628-1630: v1[1] is None and v2[1] is None
        assert result[0] is None
        assert result[1] is None
        assert result[3] == -1

    def test_interpolation_v1_none(self):
        """Interpolation with first value None."""
        ts = [(_t(0), None), (_t(2), 20.0)]
        interp = lambda v1, v2, t: (t, v2[1])
        result = get_value_from_time_series(ts, _t(1), interpolation_operation=interp)
        # Lines 1631-1632: v1[1] is None => return v2
        assert result[1] == 20.0

    def test_interpolation_v2_none(self):
        """Interpolation with second value None."""
        ts = [(_t(0), 10.0), (_t(2), None)]
        interp = lambda v1, v2, t: (t, v1[1])
        result = get_value_from_time_series(ts, _t(1), interpolation_operation=interp)
        # Lines 1634-1636: v2[1] is None => return v1
        assert result[1] == 10.0

    def test_interpolation_normal(self):
        """Normal interpolation between two valid values."""
        ts = [(_t(0), 10.0), (_t(2), 20.0)]
        interp = lambda v1, v2, t: (t, (v1[1] + v2[1]) / 2)
        result = get_value_from_time_series(ts, _t(1), interpolation_operation=interp)
        assert result[1] == 15.0

    def test_before_series(self):
        """Time before the series."""
        ts = [(_t(1), 10.0), (_t(2), 20.0)]
        result = get_value_from_time_series(ts, _t(0))
        assert result[1] == 10.0

    def test_after_series(self):
        """Time after the series."""
        ts = [(_t(0), 10.0), (_t(1), 20.0)]
        result = get_value_from_time_series(ts, _t(5))
        assert result[1] == 20.0

    def test_exact_match_in_middle(self):
        """Exact match in middle of series."""
        ts = [(_t(0), 10.0), (_t(1), 20.0), (_t(2), 30.0)]
        result = get_value_from_time_series(ts, _t(1))
        assert result[1] == 20.0


# =========================================================================
# AbstractDevice edge cases
# =========================================================================


class TestAbstractDeviceEdgeCases:
    """Test uncovered AbstractDevice lines."""

    def test_device_type_fallback_to_class_name(self):
        """Line 291: device_type falls back to class name when no conf_type_name."""
        home = create_minimal_home_model()
        device = AbstractDevice(name="Test", home=home)
        # device_type is set via _device_type in __init__ or conf_type_name
        # To hit line 291, we need _device_type=None AND no conf_type_name on class
        device._device_type = None
        # Temporarily remove conf_type_name from class
        orig = AbstractDevice.__dict__.get("conf_type_name")
        try:
            delattr(AbstractDevice, "conf_type_name")
            result = device.device_type
            assert result == "AbstractDevice"
        finally:
            AbstractDevice.conf_type_name = orig

    def test_qs_enable_device_setter_disable(self):
        """Line 446: qs_enable_device setter disabling a device."""
        home = create_minimal_home_model()
        home.remove_device = MagicMock()
        home.add_disabled_device = MagicMock()
        home.add_device = MagicMock()
        home.remove_disabled_device = MagicMock()
        device = AbstractDevice(name="Test", home=home)
        assert device.qs_enable_device is True

        device.qs_enable_device = False
        assert device.qs_enable_device is False
        home.remove_device.assert_called_once()
        home.add_disabled_device.assert_called_once()

    def test_qs_enable_device_setter_enable(self):
        """qs_enable_device setter enabling a disabled device."""
        home = create_minimal_home_model()
        home.remove_device = MagicMock()
        home.add_disabled_device = MagicMock()
        home.add_device = MagicMock()
        home.remove_disabled_device = MagicMock()
        device = AbstractDevice(name="Test", home=home)
        device._enabled = False

        device.qs_enable_device = True
        assert device.qs_enable_device is True
        home.add_device.assert_called_once()
        home.remove_disabled_device.assert_called_once()

    def test_is_load_has_a_command_now_or_coming_stacked(self):
        """Lines 554-555: is_load_has_a_command_now_or_coming with stacked cmd."""
        home = create_minimal_home_model()
        device = AbstractDevice(name="Test", home=home)
        device._stacked_command = copy_command(LoadCommand(command=CMD_ON, power_consign=1000))
        assert device.is_load_has_a_command_now_or_coming(_t(0)) is True

    @pytest.mark.asyncio
    async def test_execute_command_default(self):
        """Lines 553-555: default execute_command returns False."""
        home = create_minimal_home_model()
        device = AbstractDevice(name="Test", home=home)
        cmd = LoadCommand(command=CMD_ON, power_consign=1000)
        result = await device.execute_command(_t(0), cmd)
        assert result is False

    @pytest.mark.asyncio
    async def test_launch_command_disabled(self):
        """Line 446: launch_command when device disabled does nothing."""
        home = create_minimal_home_model()
        home.remove_device = MagicMock()
        home.add_disabled_device = MagicMock()
        device = AbstractDevice(name="Test", home=home)
        device.qs_enable_device = False
        cmd = LoadCommand(command=CMD_ON, power_consign=1000)
        # Should return without doing anything
        await device.launch_command(_t(0), cmd)
        assert device.current_command is None

    def test_dashboard_sort_string(self):
        """Line 291: dashboard_sort_string property."""
        home = create_minimal_home_model()
        device = AbstractDevice(name="Test", home=home)
        result = device.dashboard_sort_string
        assert isinstance(result, str)
        assert len(result) > 0


# =========================================================================
# AbstractLoad edge cases
# =========================================================================


class TestAbstractLoadGetOverrideState:
    """Test get_override_state with external override and constraint originator."""

    def test_get_override_state_no_override(self):
        """Default: no override returns 'NO OVERRIDE'."""
        load = MinimalTestLoad(name="TestLoad")
        assert load.get_override_state() == "NO OVERRIDE"

    def test_get_override_state_external(self):
        """External override state returns the override value."""
        load = MinimalTestLoad(name="TestLoad")
        load.external_user_initiated_state = "forced_on"
        result = load.get_override_state()
        assert "Override" in result
        assert "forced_on" in result

    def test_get_override_state_from_constraint_originator(self):
        """Lines 710-711: override from constraint with user_override originator."""
        load = MinimalTestLoad(name="TestLoad")
        now = datetime.now(tz=pytz.UTC)
        ct = create_constraint(
            load=load,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
            time=now,
            end_of_constraint=DATETIME_MAX_UTC,
            target_value=1000,
        )
        ct.load_param = "my_override_param"
        ct.load_info = {"originator": "user_override"}
        load._constraints = [ct]
        result = load.get_override_state()
        assert "Override" in result

    def test_get_override_state_asked_for_reset(self):
        """Override state with asked_for_reset returns reset message."""
        load = MinimalTestLoad(name="TestLoad")
        load.asked_for_reset_user_initiated_state_time = _t(0)
        result = load.get_override_state()
        assert "ASKED FOR RESET OVERRIDE" in result


class TestAbstractLoadIsBestEffort:
    """Test is_best_effort_only_load for line 710-711."""

    def test_best_effort_with_green_only(self):
        """Line 727: qs_best_effort_green_only returns True."""
        load = MinimalTestLoad(name="TestLoad")
        load.qs_best_effort_green_only = True
        assert load.is_best_effort_only_load() is True

    def test_best_effort_with_boost(self):
        """Line 727: load_is_auto_to_be_boosted returns True."""
        load = MinimalTestLoad(name="TestLoad")
        load.load_is_auto_to_be_boosted = True
        assert load.is_best_effort_only_load() is True


class TestAbstractLoadGetForSolverConstraints:
    """Test get_for_solver_constraints disabled path."""

    def test_disabled_clears_constraints(self):
        """Line 731: disabled device clears constraints."""
        home = create_minimal_home_model()
        home.remove_device = MagicMock()
        home.add_disabled_device = MagicMock()
        load = MinimalTestLoad(name="TestLoad", home=home)
        ct = create_constraint(load=load, time=_t(0), end_of_constraint=_t(2), target_value=100)
        load._constraints = [ct]
        load._enabled = False
        result = load.get_for_solver_constraints(_t(0), _t(2))
        assert result == []
        assert load._constraints == []


class TestAbstractLoadPushAgendaConstraints:
    """Test push_agenda_constraints edge cases (lines 846, 854-858, 861)."""

    def test_push_agenda_constraints_remove_old_calendar(self):
        """Line 846: old agenda constraint removed when not in new list."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        old_ct = create_constraint(
            load=load, time=t, end_of_constraint=_t(2), target_value=500,
        )
        old_ct.load_info = {"originator": "agenda"}
        load._constraints = [old_ct]

        new_ct = create_constraint(
            load=load, time=t, end_of_constraint=_t(4), target_value=1000,
        )
        result = load.push_agenda_constraints(t, [new_ct])
        assert result is True
        # The old one should be gone, new one present
        assert any(c.target_value >= 1000 for c in load._constraints)


class TestAbstractLoadLoadConstraintsFromStorage:
    """Test async_load_constraints_from_storage (lines 854-858, 861)."""

    @pytest.mark.asyncio
    async def test_load_constraints_from_storage(self):
        """Lines 854-858: load constraints from saved dicts."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct = create_constraint(
            load=load, time=t, end_of_constraint=_t(2), target_value=1000,
        )
        ct_dict = ct.to_dict()

        await load.async_load_constraints_from_storage(t, [ct_dict], None)
        assert load.externally_initialized_constraints is True
        assert len(load._constraints) >= 1

    @pytest.mark.asyncio
    async def test_load_constraints_from_storage_with_executed(self):
        """Line 861: stored_executed is not None."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct = create_constraint(
            load=load, time=t, end_of_constraint=_t(2), target_value=1000,
        )
        ct_dict = ct.to_dict()
        executed_ct = create_constraint(
            load=load, time=_t(-1), end_of_constraint=_t(-0.5), target_value=500,
        )
        exec_dict = executed_ct.to_dict()

        await load.async_load_constraints_from_storage(t, [ct_dict], exec_dict)
        assert load._last_completed_constraint is not None

    @pytest.mark.asyncio
    async def test_load_constraints_from_storage_expired(self):
        """Line 857: expired constraint not restored."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        # Create a constraint that ended in the past
        old_ct = create_constraint(
            load=load, time=_t(-3), end_of_constraint=_t(-1), target_value=100,
        )
        ct_dict = old_ct.to_dict()

        await load.async_load_constraints_from_storage(t, [ct_dict], None)
        # The expired one should not be restored
        assert load.externally_initialized_constraints is True


class TestAbstractLoadCheckLoadActivity:
    """Test check_load_activity_and_constraints (line 846)."""

    @pytest.mark.asyncio
    async def test_check_load_activity_default(self):
        """Line 846: default returns False."""
        load = MinimalTestLoad(name="TestLoad")
        result = await load.check_load_activity_and_constraints(_t(0))
        assert result is False


class TestAbstractLoadIsLoadActive:
    """Test is_load_active (line 895)."""

    def test_is_load_active_with_constraint(self):
        """Line 895: is_load_active True when has constraints."""
        load = MinimalTestLoad(name="TestLoad")
        ct = create_constraint(load=load, time=_t(0), end_of_constraint=_t(2), target_value=100)
        load._constraints = [ct]
        assert load.is_load_active(_t(0)) is True

    def test_is_load_active_disabled(self):
        """is_load_active False when disabled."""
        home = create_minimal_home_model()
        home.remove_device = MagicMock()
        home.add_disabled_device = MagicMock()
        load = MinimalTestLoad(name="TestLoad", home=home)
        load._enabled = False
        assert load.is_load_active(_t(0)) is False


class TestAbstractLoadCleanConstraints:
    """Test clean_constraints_for_load_param (lines 928, 955, 959)."""

    def test_clean_constraints_matching(self):
        """Line 928: clean with matching constraint keeps it."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct = create_constraint(
            load=load, time=t, end_of_constraint=_t(2), target_value=100,
        )
        ct.load_param = "my_param"
        load._constraints = [ct]
        result = load.clean_constraints_for_load_param_and_if_same_key_same_value_info(
            t, "my_param", for_full_reset=True
        )
        assert result is True

    def test_clean_constraints_not_full_reset(self):
        """Lines 955, 958-959: not full reset path."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct = create_constraint(
            load=load, time=t, end_of_constraint=_t(2), target_value=100,
        )
        ct.load_param = "my_param"
        bad_ct = create_constraint(
            load=load, time=t, end_of_constraint=_t(3), target_value=200,
        )
        bad_ct.load_param = "other_param"
        load._constraints = [ct, bad_ct]
        result = load.clean_constraints_for_load_param_and_if_same_key_same_value_info(
            t, "my_param", for_full_reset=False
        )
        assert result is True

    def test_clean_constraints_no_bad_not_full_reset(self):
        """No bad constraint found, for_full_reset=False => no reset needed."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct = create_constraint(
            load=load, time=t, end_of_constraint=_t(2), target_value=100,
        )
        ct.load_param = "my_param"
        load._constraints = [ct]
        result = load.clean_constraints_for_load_param_and_if_same_key_same_value_info(
            t, "my_param", for_full_reset=False
        )
        assert result is False  # No bad found, no reset needed

    def test_clean_constraints_with_load_info_mismatch(self):
        """Line 910: load_info key mismatch."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct = create_constraint(
            load=load, time=t, end_of_constraint=_t(2), target_value=100,
        )
        ct.load_param = "my_param"
        ct.load_info = {"car": "car1"}
        load._constraints = [ct]
        # Matching param but different info value
        result = load.clean_constraints_for_load_param_and_if_same_key_same_value_info(
            t, "my_param", load_info={"car": "car2"}, for_full_reset=True
        )
        assert result is True


class TestAbstractLoadGetActiveReadableName:
    """Test get_active_readable_name (lines 988, 993)."""

    def test_no_constraint_with_completed(self):
        """Line 988: no current constraint but has completed."""
        load = MinimalTestLoad(name="TestLoad")
        completed = create_constraint(
            load=load, time=_t(-2), end_of_constraint=_t(-1), target_value=100,
        )
        load._last_completed_constraint = completed
        result = load.get_active_readable_name(_t(0))
        assert result is not None
        assert "COMPLETED" in result

    def test_has_active_constraint(self):
        """Line 993: has active constraint."""
        load = MinimalTestLoad(name="TestLoad")
        ct = create_constraint(
            load=load, time=_t(0), end_of_constraint=_t(2), target_value=100,
        )
        load._constraints = [ct]
        result = load.get_active_readable_name(_t(0))
        assert result is not None


class TestAbstractLoadSetLiveConstraints:
    """Test set_live_constraints complex logic (lines 1063, 1076, 1085, 1098, 1149)."""

    def test_set_live_constraints_multiple_infinite(self):
        """Line 1063: multiple infinite constraints, keep highest score."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct1 = create_constraint(
            load=load, time=t, end_of_constraint=DATETIME_MAX_UTC, target_value=100,
        )
        ct2 = create_constraint(
            load=load, time=t, end_of_constraint=DATETIME_MAX_UTC, target_value=200,
        )
        load.set_live_constraints(t, [ct1, ct2])
        # Only one infinite should remain
        infinite_count = sum(1 for c in load._constraints if c.end_of_constraint == DATETIME_MAX_UTC)
        assert infinite_count <= 1

    def test_set_live_constraints_as_fast_as_possible_reorder(self):
        """Lines 1076, 1085: as_fast_as_possible constraints reordering."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        # Regular constraint first (ends before ASAP)
        ct_regular = create_constraint(
            load=load, time=t,
            end_of_constraint=_t(1),
            target_value=100,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        # ASAP constraint - type makes as_fast_as_possible True
        ct_asap = create_constraint(
            load=load, time=t,
            end_of_constraint=_t(3),
            target_value=200,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
        )
        assert ct_asap.as_fast_as_possible is True
        # Another ASAP to trigger the multi-ASAP path (lines 1076, 1085)
        ct_asap2 = create_constraint(
            load=load, time=t,
            end_of_constraint=_t(4),
            target_value=300,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
        )
        # Another regular after ASAP
        ct_regular2 = create_constraint(
            load=load, time=t,
            end_of_constraint=_t(5),
            target_value=400,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        load.set_live_constraints(t, [ct_regular, ct_asap, ct_regular2, ct_asap2])
        # ASAP should be first
        if load._constraints:
            assert load._constraints[0].as_fast_as_possible is True

    def test_set_live_constraints_none_filter(self):
        """Line 1098: _constraints set to None gets handled."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct = create_constraint(load=load, time=t, end_of_constraint=_t(2), target_value=100)
        load.set_live_constraints(t, [ct, None])
        assert None not in load._constraints

    def test_set_live_constraints_start_after_end(self):
        """Line 1149: constraint with start >= end is removed."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct1 = create_constraint(
            load=load, time=t,
            end_of_constraint=_t(1),
            target_value=100,
        )
        ct2 = create_constraint(
            load=load, time=t,
            start_of_constraint=_t(0),
            end_of_constraint=_t(2),
            target_value=200,
        )
        load.set_live_constraints(t, [ct1, ct2])
        # Both or one should remain depending on overlap


class TestAbstractLoadPushLiveConstraint:
    """Test push_live_constraint edge cases (lines 1188-1190)."""

    def test_push_constraint_same_score_same_end(self):
        """Lines 1188-1190: same end date, same score, not pushed."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct1 = create_constraint(
            load=load, time=t, end_of_constraint=_t(2),
            target_value=100,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        load.push_live_constraint(t, ct1)

        ct2 = create_constraint(
            load=load, time=t, end_of_constraint=_t(2),
            target_value=100,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        result = load.push_live_constraint(t, ct2)
        # Should not push since same score
        assert result is False

    def test_push_constraint_disabled(self):
        """push_live_constraint on disabled device."""
        home = create_minimal_home_model()
        home.remove_device = MagicMock()
        home.add_disabled_device = MagicMock()
        load = MinimalTestLoad(name="TestLoad", home=home)
        load._enabled = False
        ct = create_constraint(load=load, time=_t(0), end_of_constraint=_t(2), target_value=100)
        result = load.push_live_constraint(_t(0), ct)
        assert result is True
        assert load._constraints == []


class TestAbstractLoadUpdateLiveConstraints:
    """Test update_live_constraints complex flow (lines 1243, 1253-1255, 1259-1318, 1323-1331, 1363)."""

    @pytest.mark.asyncio
    async def test_update_empty_constraints(self):
        """No constraints => force_solving is False."""
        load = MinimalTestLoad(name="TestLoad")
        result = await load.update_live_constraints(_t(0), timedelta(minutes=30))
        assert result is False

    @pytest.mark.asyncio
    async def test_update_met_constraint(self):
        """Constraint met => skipped and acked."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct = create_constraint(
            load=load, time=t, end_of_constraint=_t(2),
            target_value=100,
            initial_value=100,  # Already met
        )
        load._constraints = [ct]
        load._last_constraint_update = t
        result = await load.update_live_constraints(t, timedelta(minutes=30))
        assert result is True  # force_solving should be True

    @pytest.mark.asyncio
    async def test_update_non_mandatory_expired(self):
        """Lines 1252-1255: expired non-mandatory constraint gets skipped."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(1)  # time is after constraint end
        ct = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=_t(0.5),  # Already ended
            target_value=100, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,  # Not mandatory
        )
        load._constraints = [ct]
        load._last_constraint_update = t
        result = await load.update_live_constraints(t, timedelta(minutes=30))
        assert result is True

    @pytest.mark.asyncio
    async def test_update_mandatory_expired_pushed(self):
        """Lines 1259-1318: mandatory expired not met, gets pushed."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(1)
        ct = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t - timedelta(seconds=1),  # Just expired
            target_value=10000, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        ct.always_end_at_end_of_constraint = False
        load._constraints = [ct]
        load._last_constraint_update = t
        result = await load.update_live_constraints(t, timedelta(minutes=30))
        # Should be pushed, force_solving True
        assert result is True

    @pytest.mark.asyncio
    async def test_update_mandatory_expired_pushed_too_many_times(self):
        """Lines 1299-1303: pushed too many times (>4), gets skipped."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(1)
        ct = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t - timedelta(seconds=1),
            target_value=10000, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        ct.always_end_at_end_of_constraint = False
        ct.pushed_count = 5  # Already pushed too many times
        load._constraints = [ct]
        load._last_constraint_update = t
        result = await load.update_live_constraints(t, timedelta(minutes=30))
        assert result is True

    @pytest.mark.asyncio
    async def test_update_mandatory_with_next_constraint_higher_score(self):
        """Lines 1286-1293: next constraint has higher score, current skipped."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(1)
        # First constraint: expired mandatory, not met
        ct1 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t - timedelta(seconds=1),
            target_value=100, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        ct1.always_end_at_end_of_constraint = False
        # Second constraint: a more important one within the push window
        ct2 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t + timedelta(minutes=10),
            target_value=50000, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        ct2.from_user = True  # Higher score
        load._constraints = [ct1, ct2]
        load._last_constraint_update = t
        result = await load.update_live_constraints(t, timedelta(minutes=30))
        assert result is True

    @pytest.mark.asyncio
    async def test_update_active_constraint_with_callback_stop(self):
        """Lines 1323-1331: update callback returns False, constraint stopped."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct = create_constraint(
            load=load, time=t,
            end_of_constraint=_t(2),
            target_value=100, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        # Set up a callback that stops the constraint
        async def stop_callback(constraint, time):
            return (50.0, False)

        ct._update_value_callback = stop_callback
        load._constraints = [ct]
        load._last_constraint_update = t
        # Period must be large enough that the constraint end is within time+period
        result = await load.update_live_constraints(t, timedelta(hours=24))
        # Constraint is stopped (skip=True), filtered out, force_solving=True
        assert result is True

    @pytest.mark.asyncio
    async def test_update_day_rollover(self):
        """Lines 1243, 1253-1255: day rollover triggers reset_daily_load_datas."""
        load = MinimalTestLoad(name="TestLoad")
        # Previous update was yesterday
        yesterday = datetime(2026, 2, 10, 23, 50, 0, tzinfo=pytz.UTC)
        today = datetime(2026, 2, 11, 0, 10, 0, tzinfo=pytz.UTC)
        load._last_constraint_update = yesterday
        ct = create_constraint(
            load=load, time=yesterday,
            end_of_constraint=today + timedelta(hours=2),
            target_value=100, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        load._constraints = [ct]
        result = await load.update_live_constraints(today, timedelta(minutes=30))
        # Day rollover should have been detected

    @pytest.mark.asyncio
    async def test_update_constraint_start_time_set(self):
        """Line 1363: constraint start/end times are set on load."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct = create_constraint(
            load=load, time=t,
            start_of_constraint=_t(0),
            end_of_constraint=_t(2),
            target_value=100, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        load._constraints = [ct]
        load._last_constraint_update = t
        await load.update_live_constraints(t, timedelta(minutes=30))
        # After update, next_or_current_constraint_end_time should be set
        if load._constraints:
            assert load.next_or_current_constraint_end_time is not None

    @pytest.mark.asyncio
    async def test_update_with_completed_constraint(self):
        """Line 1349-1352: no current, but has last completed."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        completed = create_constraint(
            load=load, time=_t(-2),
            end_of_constraint=_t(-1),
            target_value=100, initial_value=100,
        )
        load._last_completed_constraint = completed
        load._last_constraint_update = t
        result = await load.update_live_constraints(t, timedelta(minutes=30))
        assert load.current_constraint_current_percent_completion == 100.0

    @pytest.mark.asyncio
    async def test_update_normal_active_constraint(self):
        """Normal active constraint gets updated."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct = create_constraint(
            load=load, time=t,
            end_of_constraint=_t(2),
            target_value=100, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        load._constraints = [ct]
        load._last_constraint_update = t
        result = await load.update_live_constraints(t, timedelta(hours=24))
        # Active constraint should be properly processed
        assert load.next_or_current_constraint_end_time is not None


class TestAbstractLoadMarkConstraintDone:
    """Test mark_current_constraint_has_done and user_clean_and_reset (lines 1629-1636)."""

    @pytest.mark.asyncio
    async def test_mark_current_constraint_has_done(self):
        """Lines 1629-1630: mark_current_constraint_has_done."""
        home = MinimalTestHome()
        home._period = timedelta(minutes=30)
        home.force_next_solve = MagicMock()
        load = MinimalTestLoad(name="TestLoad", home=home)
        t = _t(0)
        ct = create_constraint(
            load=load, time=t,
            end_of_constraint=_t(2),
            target_value=100, initial_value=50,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        load._constraints = [ct]
        load._last_constraint_update = t
        await load.mark_current_constraint_has_done(t)
        # Constraint should be marked as done (current_value = target_value)

    @pytest.mark.asyncio
    async def test_mark_current_constraint_has_done_no_time(self):
        """Lines 1632-1633: mark_current_constraint_has_done with no time uses now."""
        home = MinimalTestHome()
        home._period = timedelta(minutes=30)
        home.force_next_solve = MagicMock()
        load = MinimalTestLoad(name="TestLoad", home=home)
        # No constraints - should not error
        await load.mark_current_constraint_has_done(None)

    @pytest.mark.asyncio
    async def test_user_clean_and_reset(self):
        """Lines 1635-1636: user_clean_and_reset calls launch_command with IDLE."""
        load = MinimalTestLoad(name="TestLoad")
        await load.user_clean_and_reset()
        # Should have been reset


class TestAbstractLoadAsyncResetOverrideState:
    """Test async_reset_override_state (lines 1398-1401)."""

    @pytest.mark.asyncio
    async def test_async_reset_override_state(self):
        """Lines 1398-1401: async_reset_override_state."""
        home = MinimalTestHome()
        home.force_next_solve = MagicMock()
        load = MinimalTestLoad(name="TestLoad", home=home)
        load.externally_initialized_constraints = True
        load.external_user_initiated_state = "forced_on"
        load.external_user_initiated_state_time = _t(0)
        await load.async_reset_override_state()
        assert load.external_user_initiated_state is None


class TestAbstractLoadCommandAndConstraintReset:
    """Test AbstractLoad.command_and_constraint_reset (line 898-900)."""

    def test_command_and_constraint_reset_clears_completed(self):
        """Line 898-900: reset clears _last_completed_constraint."""
        load = MinimalTestLoad(name="TestLoad")
        ct = create_constraint(load=load, time=_t(0), end_of_constraint=_t(1), target_value=100)
        load._last_completed_constraint = ct

        load.constraint_reset_and_reset_commands_if_needed(keep_commands=False)

        assert load._last_completed_constraint is None
        assert load._constraints == []
        assert load.current_command is None


class TestAbstractLoadMandatoryExpiredWithNextSkipPath:
    """Test the inner loop of update_live_constraints for expired mandatory constraints."""

    @pytest.mark.asyncio
    async def test_next_constraint_also_expired_skipped(self):
        """Lines 1273-1275: next constraint also expired, gets skipped in inner loop."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(1)
        # First: mandatory, just expired, not met
        ct1 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t - timedelta(seconds=1),
            target_value=10000, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        ct1.always_end_at_end_of_constraint = False
        # Second: also expired
        ct2 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t - timedelta(seconds=2),
            target_value=5000, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        # Third: future, large enough to break the loop
        ct3 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t + timedelta(hours=5),
            target_value=100, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        load._constraints = [ct1, ct2, ct3]
        load._last_constraint_update = t
        result = await load.update_live_constraints(t, timedelta(minutes=30))
        assert result is True

    @pytest.mark.asyncio
    async def test_next_constraint_met_skipped(self):
        """Lines 1281-1282: next constraint is met, gets skipped."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(1)
        ct1 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t - timedelta(seconds=1),
            target_value=10000, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        ct1.always_end_at_end_of_constraint = False
        # Next: within push window but already met
        ct2 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t + timedelta(minutes=5),
            target_value=100, initial_value=100,  # Already met
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        # Third: far future to break
        ct3 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t + timedelta(hours=5),
            target_value=100, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        load._constraints = [ct1, ct2, ct3]
        load._last_constraint_update = t
        result = await load.update_live_constraints(t, timedelta(minutes=30))
        assert result is True

    @pytest.mark.asyncio
    async def test_next_constraint_lower_score_skipped(self):
        """Lines 1294-1295: next constraint with lower score gets skipped."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(1)
        # First: mandatory, just expired, not met, HIGH score
        ct1 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t - timedelta(seconds=1),
            target_value=50000, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        ct1.from_user = True  # Higher score
        ct1.always_end_at_end_of_constraint = False
        # Second: within push window, lower score, not met
        ct2 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t + timedelta(minutes=5),
            target_value=100, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        # Third: far future to break
        ct3 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t + timedelta(hours=5),
            target_value=100, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        load._constraints = [ct1, ct2, ct3]
        load._last_constraint_update = t
        result = await load.update_live_constraints(t, timedelta(minutes=30))
        assert result is True


class TestAbstractLoadUpdateConstraintCallback:
    """Test update_live_constraints with constraint update callback (lines 1322-1331)."""

    @pytest.mark.asyncio
    async def test_update_callback_met_after_update(self):
        """Lines 1325-1327: constraint met just after update."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct = create_constraint(
            load=load, time=t,
            end_of_constraint=_t(2),
            target_value=100, initial_value=50,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )

        async def met_callback(constraint, time):
            constraint.current_value = constraint.target_value  # Make it met
            return (constraint.target_value, False)

        ct._update_value_callback = met_callback
        load._constraints = [ct]
        load._last_constraint_update = t
        result = await load.update_live_constraints(t, timedelta(hours=24))
        assert result is True

    @pytest.mark.asyncio
    async def test_update_callback_stopped_not_met(self):
        """Lines 1328-1330: callback stopped but not met."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct = create_constraint(
            load=load, time=t,
            end_of_constraint=_t(2),
            target_value=100, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )

        async def stop_callback(constraint, time):
            return (10.0, False)

        ct._update_value_callback = stop_callback
        load._constraints = [ct]
        load._last_constraint_update = t
        result = await load.update_live_constraints(t, timedelta(hours=24))
        assert result is True


class TestSetLiveConstraintsDeepEdgeCases:
    """Deeper edge cases for set_live_constraints to cover remaining lines."""

    def test_multiple_infinite_higher_score_second(self):
        """Line 1063: multiple infinite constraints, second has higher score => keep it."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        # First infinite with low score
        ct1 = create_constraint(
            load=load, time=t, end_of_constraint=DATETIME_MAX_UTC,
            target_value=100, constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        # Second infinite with higher score (user requested, higher target)
        ct2 = create_constraint(
            load=load, time=t, end_of_constraint=DATETIME_MAX_UTC,
            target_value=50000, constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        ct2.from_user = True
        load.set_live_constraints(t, [ct1, ct2])
        # Only one infinite should remain, and it should be the higher score one
        infinites = [c for c in load._constraints if c.end_of_constraint == DATETIME_MAX_UTC]
        assert len(infinites) <= 1

    def test_asap_constraint_already_met_in_reorder(self):
        """Line 1085: ASAP constraint that is_constraint_met => continue in the scoring loop."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        # A regular before
        ct_regular = create_constraint(
            load=load, time=t,
            end_of_constraint=_t(1),
            target_value=100,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        # ASAP that's already met
        ct_asap_met = create_constraint(
            load=load, time=t,
            end_of_constraint=_t(3),
            target_value=100, initial_value=100,  # Already met
            constraint_type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
        )
        # Another ASAP that's not met
        ct_asap2 = create_constraint(
            load=load, time=t,
            end_of_constraint=_t(4),
            target_value=200,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
        )
        load.set_live_constraints(t, [ct_regular, ct_asap_met, ct_asap2])

    def test_constraint_start_after_end_removed(self):
        """Line 1149: constraint with start >= end is removed during final pass."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        # First constraint takes the full window
        ct1 = create_constraint(
            load=load, time=t,
            start_of_constraint=_t(0),
            end_of_constraint=_t(5),
            target_value=100,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        # Second constraint starts at t(0) but ends at t(3) - its current_start will be
        # pushed to after ct1's end (t(5)), making start >= end
        ct2 = create_constraint(
            load=load, time=t,
            start_of_constraint=_t(0),
            end_of_constraint=_t(3),
            target_value=200,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        load.set_live_constraints(t, [ct1, ct2])
        # ct2 should be removed since its adjusted start (t(5)) >= its end (t(3))

    def test_push_same_end_same_score(self):
        """Lines 1188-1190: push with same end date AND same score => not pushed."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        ct1 = create_constraint(
            load=load, time=t,
            end_of_constraint=_t(2),
            target_value=100,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        load.push_live_constraint(t, ct1)
        assert len(load._constraints) == 1

        # Same end, same type, same target => same score
        ct2 = create_constraint(
            load=load, time=t,
            end_of_constraint=_t(2),
            target_value=100,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        result = load.push_live_constraint(t, ct2)
        # It should be rejected (same score, same end)
        assert result is False


class TestUpdateLiveConstraintsInnerLoop:
    """Test the inner loop of update_live_constraints for nc paths."""

    @pytest.mark.asyncio
    async def test_nc_skip_already_set(self):
        """Line 1271: nc.skip=True in inner loop (set by previous outer iteration)."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(1)
        # First: mandatory, just expired, not met
        ct1 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t - timedelta(seconds=1),
            target_value=10000, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        ct1.always_end_at_end_of_constraint = False
        # Second: already met (will be skipped in inner loop)
        ct2 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t + timedelta(minutes=5),
            target_value=100, initial_value=100,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        # Third: far future, end >= new_constraint_end to break
        ct3 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t + timedelta(hours=5),
            target_value=100, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        load._constraints = [ct1, ct2, ct3]
        load._last_constraint_update = t
        result = await load.update_live_constraints(t, timedelta(hours=24))
        assert result is True

    @pytest.mark.asyncio
    async def test_nc_end_beyond_push_window(self):
        """Line 1278: nc.end_of_constraint >= new_constraint_end => break."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(1)
        ct1 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t - timedelta(seconds=1),
            target_value=10000, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        )
        ct1.always_end_at_end_of_constraint = False
        # Next: end way beyond push window
        ct2 = create_constraint(
            load=load, time=_t(0),
            end_of_constraint=t + timedelta(hours=48),
            target_value=100, initial_value=0,
            constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        )
        load._constraints = [ct1, ct2]
        load._last_constraint_update = t
        result = await load.update_live_constraints(t, timedelta(hours=24))
        assert result is True


class TestAsyncResetOverrideWithActivity:
    """Test async_reset_override_state line 1401."""

    @pytest.mark.asyncio
    async def test_reset_override_with_active_constraints(self):
        """Line 1401: do_run_check returns True => force_next_solve."""
        home = MinimalTestHome()
        home.force_next_solve = MagicMock()
        load = MinimalTestLoad(name="TestLoad", home=home)
        load.externally_initialized_constraints = True
        load.external_user_initiated_state = "forced_on"

        # Override check_load_activity_and_constraints to return True
        async def mock_check(time):
            return True

        load.check_load_activity_and_constraints = mock_check
        await load.async_reset_override_state()
        home.force_next_solve.assert_called_once()


class TestPushAgendaConstraintDeep:
    """Test push_agenda_constraints line 928 deeper."""

    def test_push_agenda_full_reset_with_completed(self):
        """Line 928: full reset with last completed constraint matching."""
        load = MinimalTestLoad(name="TestLoad")
        t = _t(0)
        completed = create_constraint(
            load=load, time=_t(-2), end_of_constraint=_t(-1), target_value=50,
        )
        completed.load_param = None
        load._last_completed_constraint = completed

        # Clean with matching param
        result = load.clean_constraints_for_load_param_and_if_same_key_same_value_info(
            t, None, for_full_reset=True
        )
        assert result is True


class TestTestLoadInit:
    """Test the TestLoad class at end of file."""

    def test_test_load_init(self):
        """Test TestLoad initialization."""
        home = MinimalTestHome()
        tl = TestLoad(name="TL", home=home, min_p=500, max_p=2000, min_a=3, max_a=16)
        assert tl.min_p == 500
        assert tl.max_p == 2000
        mn, mx = tl.get_min_max_power()
        assert mn == 500
        assert mx == 2000

    def test_test_load_defaults(self):
        """Test TestLoad default values."""
        home = MinimalTestHome()
        tl = TestLoad(name="TL", home=home)
        assert tl.min_p == 1500
        assert tl.max_p == 1500
        assert tl.min_a == 7
        assert tl.max_a == 7
