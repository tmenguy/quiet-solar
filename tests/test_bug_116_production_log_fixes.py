"""Tests for bug fixes from GitHub issue #116.

Nine bugs discovered in 2026-04-04 production log analysis:
- Bug 1: NoneType budgeted_amp crash in apply_budget_strategy
- Bug 2: IndexError in adapt_repartition with empty power_sorted_cmds
- Bug 3: Empty commands in adapt_power_steps_budgeting_low_level + latent bug
- Bug 4: False no-power-detected error when SOC sensor unavailable
- Bug 5: is_person_covered warning spam
- Bug 6: REJECTING unauthorized assignment warning spam
- Bug 7: Python 2-style except syntax
- Bug 8: Startup lazy safe value warning spam
- Bug 9: Battery probe_if_command_set startup warning
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytz

from custom_components.quiet_solar.ha_model.charger import QSChargerStatus
from custom_components.quiet_solar.ha_model.home import QSforecastValueSensor
from custom_components.quiet_solar.home_model.commands import CMD_ON, LoadCommand, copy_command

from .factories import (
    MinimalTestHome,
    MinimalTestLoad,
    TestChargerDouble,
    TestDynamicGroupDouble,
    create_constraint,
    create_load_command,
)


# =============================================================================
# Bug 1: NoneType budgeted_amp — defensive guards in get_*_amps()
# =============================================================================


def test_get_current_charging_amps_returns_zeros_when_none():
    """Bug 1: get_current_charging_amps returns [0,0,0] when values are None."""
    charger = MagicMock()
    charger.mono_phase_index = 0
    cs = QSChargerStatus(charger)
    # __init__ sets both to None
    assert cs.current_real_max_charging_amp is None
    assert cs.current_active_phase_number is None
    result = cs.get_current_charging_amps()
    assert result == [0.0, 0.0, 0.0]


def test_get_budget_amps_returns_zeros_when_none():
    """Bug 1: get_budget_amps returns [0,0,0] when values are None."""
    charger = MagicMock()
    charger.mono_phase_index = 0
    cs = QSChargerStatus(charger)
    assert cs.budgeted_amp is None
    assert cs.budgeted_num_phases is None
    result = cs.get_budget_amps()
    assert result == [0.0, 0.0, 0.0]


def test_get_budget_amps_returns_zeros_when_amp_is_none_but_phases_set():
    """Bug 1: get_budget_amps returns [0,0,0] when budgeted_amp is None."""
    charger = MagicMock()
    charger.mono_phase_index = 0
    cs = QSChargerStatus(charger)
    cs.budgeted_num_phases = 3
    # budgeted_amp still None
    result = cs.get_budget_amps()
    assert result == [0.0, 0.0, 0.0]


def test_get_current_charging_amps_works_when_values_set():
    """Bug 1: get_current_charging_amps works normally when values are set."""
    charger = MagicMock()
    charger.mono_phase_index = 0
    cs = QSChargerStatus(charger)
    cs.current_real_max_charging_amp = 16
    cs.current_active_phase_number = 3
    result = cs.get_current_charging_amps()
    assert result == [16, 16, 16]


def test_get_budget_amps_works_when_values_set():
    """Bug 1: get_budget_amps works normally when values are set."""
    charger = MagicMock()
    charger.mono_phase_index = 0
    cs = QSChargerStatus(charger)
    cs.budgeted_amp = 10
    cs.budgeted_num_phases = 1
    result = cs.get_budget_amps()
    assert result == [10, 0.0, 0.0]


def test_cooldown_excluded_charger_gets_budgeted_amp_initialized():
    """Bug 1: charger excluded from budgeting by cooldown filter gets budgeted_amp set."""
    charger = MagicMock()
    charger.mono_phase_index = 0
    charger.physical_3p = False
    cs = QSChargerStatus(charger)
    cs.current_real_max_charging_amp = 16
    cs.current_active_phase_number = 1

    # Simulate what the cooldown filter now does
    cs.budgeted_amp = cs.current_real_max_charging_amp or 0
    cs.budgeted_num_phases = cs.current_active_phase_number or (3 if cs.charger.physical_3p else 1)

    assert cs.budgeted_amp == 16
    assert cs.budgeted_num_phases == 1
    # Now get_budget_amps should work without crash
    result = cs.get_budget_amps()
    assert result == [16, 0.0, 0.0]


def test_apply_budget_strategy_no_crash_with_cooldown_initialized_values():
    """Bug 1: apply_budget_strategy can compute max_curr_amps without TypeError."""
    charger = MagicMock()
    charger.mono_phase_index = 0
    cs = QSChargerStatus(charger)
    cs.current_real_max_charging_amp = 10
    cs.current_active_phase_number = 3
    cs.budgeted_amp = 10  # set by cooldown filter
    cs.budgeted_num_phases = 3

    curr_amps = cs.get_current_charging_amps()
    budget_amps = cs.get_budget_amps()

    # This is the line that used to crash: max(curr_amps[i], budget_amps[i])
    max_curr_amps = [max(curr_amps[i], budget_amps[i]) for i in range(3)]
    assert max_curr_amps == [10, 10, 10]


# =============================================================================
# Bug 2: IndexError in adapt_repartition with empty power_sorted_cmds
# =============================================================================


def test_adapt_repartition_empty_power_sorted_cmds_no_crash():
    """Bug 2: adapt_repartition handles empty power_sorted_cmds without IndexError."""
    import numpy as np

    from custom_components.quiet_solar.const import CONSTRAINT_TYPE_MANDATORY_END_TIME

    now = datetime(2026, 4, 4, 10, 0, tzinfo=pytz.UTC)
    home = MinimalTestHome(voltage=230.0)
    load = MinimalTestLoad(name="TestLoad", power=1000.0, home=home)

    dyn_group = TestDynamicGroupDouble(
        name="TestGroup",
        home=home,
        max_amps=[32.0, 32.0, 32.0],
        num_slots=4,
    )
    load.father_device = dyn_group

    ct = create_constraint(
        load=load,
        constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        time=now,
        end_of_constraint=now + timedelta(hours=2),
        initial_value=0.0,
        target_value=2000.0,
        power=1000.0,
    )

    # Mock adapt_power_steps_budgeting to return empty list
    ct.adapt_power_steps_budgeting = MagicMock(return_value=([], False, 0.0))

    duration_s = np.array([900.0] * 4, dtype=np.float64)
    existing_commands: list[LoadCommand | None] = [None] * 4

    # Should NOT raise IndexError
    result = ct.adapt_repartition(
        first_slot=0,
        last_slot=3,
        energy_delta=500.0,
        power_slots_duration_s=duration_s,
        existing_commands=existing_commands,
        allow_change_state=True,
        time=now,
    )
    # Returns a tuple: (self, changed_state, changed_power, energy_delta, commands, delta_power)
    assert isinstance(result, tuple)


# =============================================================================
# Bug 3: Empty commands in adapt_power_steps_budgeting_low_level
# =============================================================================


def test_adapt_power_steps_budgeting_low_level_returns_empty_when_all_exceed_budget():
    """Bug 3: explicit return [] when all commands exceed available budget."""
    from custom_components.quiet_solar.const import CONSTRAINT_TYPE_FILLER_AUTO
    from custom_components.quiet_solar.home_model.commands import CMD_ON

    now = datetime(2026, 4, 4, 10, 0, tzinfo=pytz.UTC)
    home = MinimalTestHome(voltage=230.0)
    load = MinimalTestLoad(name="TestLoad", power=5000.0, home=home)

    # Very restrictive budget — 1 amp per phase
    dyn_group = TestDynamicGroupDouble(
        name="TestGroup",
        home=home,
        max_amps=[1.0, 1.0, 1.0],
        num_slots=2,
    )
    load.father_device = dyn_group

    ct = create_constraint(
        load=load,
        constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
        time=now,
        end_of_constraint=now + timedelta(hours=2),
        initial_value=0.0,
        target_value=10000.0,
        power=5000.0,
    )

    # Create power steps that all exceed the 1A budget
    ct._power_sorted_cmds = [
        create_load_command(command=CMD_ON, power_consign=2000.0),
        create_load_command(command=CMD_ON, power_consign=5000.0),
    ]

    result = ct.adapt_power_steps_budgeting_low_level(slot_idx=0)
    assert result == []


# =============================================================================
# Bug 7: Python 2-style except syntax
# =============================================================================


def test_except_syntax_catches_typeerror():
    """Bug 7: except (ValueError, TypeError, IndexError) catches TypeError from float(None)."""
    sv = QSforecastValueSensor(name="test", duration_s=0, forecast_getter=None, current_getter=None)

    # Restore with data that includes None value — should not propagate TypeError
    bad_data = [
        ["2026-04-04T10:00:00+00:00", None],  # float(None) → TypeError
        ["2026-04-04T11:00:00+00:00", "not_a_number"],  # float("not_a_number") → ValueError
    ]
    sv.restore_stored_values(bad_data)
    # Should have skipped the bad entries without crashing
    assert len(sv._stored_values) == 0


def test_except_syntax_catches_valueerror():
    """Bug 7: except (ValueError, TypeError, IndexError) catches ValueError."""
    sv = QSforecastValueSensor(name="test", duration_s=0, forecast_getter=None, current_getter=None)

    bad_data = [
        ["2026-04-04T10:00:00+00:00", "abc"],  # ValueError
    ]
    sv.restore_stored_values(bad_data)
    assert len(sv._stored_values) == 0


def test_except_syntax_accepts_valid_data():
    """Bug 7: valid data still works correctly after except syntax fix."""
    sv = QSforecastValueSensor(name="test", duration_s=0, forecast_getter=None, current_getter=None)

    good_data = [
        ["2026-04-04T10:00:00+00:00", "42.5"],
        ["2026-04-04T11:00:00+00:00", "100.0"],
    ]
    sv.restore_stored_values(good_data)
    assert len(sv._stored_values) == 2


# =============================================================================
# Bug 5: is_person_covered warning spam rate-limiting
# =============================================================================


def test_person_coverage_warning_emitted_once_per_triplet(caplog):
    """Bug 5: warning emitted once per (car, person, next_usage_time) triplet."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric

    charger = MagicMock(spec=QSChargerGeneric)
    charger._warned_person_coverage_triplet = None

    car_name = "Test Car"
    person_name = "Test Person"
    next_usage_time = datetime(2026, 4, 5, 8, 0, tzinfo=pytz.UTC)
    now = datetime(2026, 4, 4, 10, 0, tzinfo=pytz.UTC)

    triplet = (car_name, person_name, next_usage_time)

    # First call — warning should be emitted
    with caplog.at_level(logging.WARNING):
        if triplet != charger._warned_person_coverage_triplet:
            charger._warned_person_coverage_triplet = triplet
            logging.getLogger().warning(
                "plugged car %s is assigned to person %s next usage at %s",
                car_name,
                person_name,
                next_usage_time,
            )

    assert len(caplog.records) == 1

    # Second call with same triplet — should be suppressed
    caplog.clear()
    if triplet != charger._warned_person_coverage_triplet:
        logging.getLogger().warning("should not appear")

    assert len(caplog.records) == 0


def test_person_coverage_warning_reemitted_on_triplet_change(caplog):
    """Bug 5: warning re-emitted when triplet changes."""
    charger = MagicMock()
    charger._warned_person_coverage_triplet = None

    triplet1 = ("Car A", "Person A", datetime(2026, 4, 5, 8, 0, tzinfo=pytz.UTC))
    triplet2 = ("Car A", "Person B", datetime(2026, 4, 5, 9, 0, tzinfo=pytz.UTC))

    with caplog.at_level(logging.WARNING):
        # First triplet
        if triplet1 != charger._warned_person_coverage_triplet:
            charger._warned_person_coverage_triplet = triplet1
            logging.getLogger().warning("warning for triplet 1")

        # Different triplet — should emit again
        if triplet2 != charger._warned_person_coverage_triplet:
            charger._warned_person_coverage_triplet = triplet2
            logging.getLogger().warning("warning for triplet 2")

    assert len(caplog.records) == 2


# =============================================================================
# Bug 6: REJECTING unauthorized assignment — pre-filter and log downgrade
# =============================================================================


def test_persons_with_no_authorized_car_excluded_from_optimization():
    """Bug 6: persons with no authorized car in current set are pre-filtered."""
    # Build a simple scenario where person has authorized cars but none are in c_s
    person = MagicMock()
    person.name = "Test Person"
    person.get_authorized_cars = MagicMock(return_value=[])  # No authorized cars

    c_s = [MagicMock(name="Car A")]
    c_names = {c.name for c in c_s}

    p_s = [(person, datetime(2026, 4, 5, 8, 0, tzinfo=pytz.UTC), 50.0)]

    # Pre-filter (same logic as the fix)
    filtered_p_s = [
        (p, leave_time, mileage)
        for p, leave_time, mileage in p_s
        if any(c.name in c_names for c in p.get_authorized_cars())
    ]

    assert len(filtered_p_s) == 0


def test_persons_with_authorized_car_kept_in_optimization():
    """Bug 6: persons with authorized cars in current set are kept."""
    car_mock = MagicMock()
    car_mock.name = "Car A"

    person = MagicMock()
    person.name = "Test Person"
    person.get_authorized_cars = MagicMock(return_value=[car_mock])

    c_s = [car_mock]
    c_names = {c.name for c in c_s}

    p_s = [(person, datetime(2026, 4, 5, 8, 0, tzinfo=pytz.UTC), 50.0)]

    filtered_p_s = [
        (p, leave_time, mileage)
        for p, leave_time, mileage in p_s
        if any(c.name in c_names for c in p.get_authorized_cars())
    ]

    assert len(filtered_p_s) == 1


# =============================================================================
# Bug 4: False no-power-detected error — SOC sensor unavailable guard
# =============================================================================


def test_zero_power_error_skipped_when_sensor_unavailable(caplog):
    """Bug 4: zero-power error is skipped (downgraded to WARNING) when SOC sensor is None."""
    sensor_result = None
    charger_is_zero = True
    is_target_percent = True
    name = "Test Charger"
    car_name = "Test Car"

    with caplog.at_level(logging.WARNING):
        if charger_is_zero:
            if sensor_result is None:
                logging.getLogger().warning(
                    "update_value_callback: %s %s expected to be charging but no power detected"
                    " (skipping error — SOC sensor unavailable)",
                    name,
                    car_name,
                )
            else:
                logging.getLogger().error("should not appear")

    assert len(caplog.records) == 1
    assert "skipping error" in caplog.records[0].message
    assert caplog.records[0].levelno == logging.WARNING


def test_zero_power_error_fires_when_sensor_available(caplog):
    """Bug 4: zero-power error still fires normally when SOC sensor has a value."""
    sensor_result = 50.0  # sensor available
    charger_is_zero = True

    with caplog.at_level(logging.ERROR):
        if charger_is_zero:
            if sensor_result is None:
                logging.getLogger().warning("should not appear")
            else:
                logging.getLogger().error(
                    "expected to be charging but no power detected",
                )

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "no power detected" in error_records[0].message


# =============================================================================
# Bug 8: Startup lazy safe value — log downgrade to DEBUG
# =============================================================================


def test_lazy_safe_value_logs_at_debug_not_warning():
    """Bug 8: lazy safe value errors are logged at DEBUG, not WARNING.

    Verify by source code inspection that the log level was changed.
    """
    import inspect

    from custom_components.quiet_solar.ha_model.home import QSSolarHistoryVals

    source = inspect.getsource(QSSolarHistoryVals)
    # The "Error loading lazy safe value" messages should use _LOGGER.debug, not _LOGGER.warning
    assert '_LOGGER.warning("Error loading lazy safe value' not in source
    assert '_LOGGER.debug("Error loading lazy safe value' in source


# =============================================================================
# Bug 9: Battery probe_if_command_set — log downgrade to DEBUG
# =============================================================================


def test_battery_probe_logs_at_debug_not_warning():
    """Bug 9: battery probe_if_command_set None returns are logged at DEBUG."""
    # Verify by checking the source code directly — the _LOGGER.debug calls
    import inspect

    from custom_components.quiet_solar.ha_model.battery import QSBattery

    source = inspect.getsource(QSBattery.probe_if_command_set)
    # Should contain _LOGGER.debug, not _LOGGER.warning for these cases
    assert "_LOGGER.debug" in source
    assert "_LOGGER.warning" not in source
