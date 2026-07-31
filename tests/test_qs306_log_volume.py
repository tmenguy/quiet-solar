"""QS-306: INFO log volume reduction.

Covers the `LogOnChangeMixin` helper (B3) and the 13 call sites the story
enumerates (B1, B1b, B2, B4, B5).
"""

from __future__ import annotations

import copy
import logging
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz

import custom_components.quiet_solar as quiet_solar
from custom_components.quiet_solar.const import (
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    USER_ORIGINATED_CAR_NAME,
)
from custom_components.quiet_solar.ha_model.charger import (
    _POWER_LOG_DEADBAND_W,
    _RELOG_UNCHANGED_AFTER_S,
    CHARGER_ADAPTATION_WINDOW_S,
    LogOnChangeMixin,
    QSChargerStatus,
)
from custom_components.quiet_solar.home_model.commands import (
    CMD_AUTO_FROM_CONSIGN,
    CMD_AUTO_GREEN_ONLY,
    CMD_ON,
    copy_command,
)
from custom_components.quiet_solar.home_model.constraints import (
    LoadConstraint,
    MultiStepsPowerLoadConstraintChargePercent,
)
from tests.factories import MinimalTestLoad, create_constraint
from tests.utils.charger_harness import (
    create_charger,
    init_charger_states,
    make_battery,
    make_charger_group,
    make_hass,
    make_home,
    make_real_car,
    plug_car,
)

CHARGER_LOGGER = "custom_components.quiet_solar.ha_model.charger"
LOAD_LOGGER = "custom_components.quiet_solar.home_model.load"

T0 = datetime(2026, 7, 29, 8, 0, 0, tzinfo=pytz.UTC)


class _Host(LogOnChangeMixin):
    """Minimal mixin host: the mixin defines no `__init__`, so this suffices."""


def _messages(
    caplog: pytest.LogCaptureFixture,
    fragment: str,
    level: int,
    logger_name: str | None = None,
) -> list[logging.LogRecord]:
    """Return the captured records at `level` whose message contains `fragment`.

    `logger_name` pins the emitting logger. B2 requires it, and the story records
    the coupling deliberately: extracting the mixin out of `charger.py` would
    change the logger name, and this assertion is the guard that notices. Every
    call site in this module passes it; it stays optional so a future caller that
    genuinely does not care is not forced to lie.
    """
    return [
        r
        for r in caplog.records
        if r.levelno == level and fragment in r.getMessage() and (logger_name is None or r.name == logger_name)
    ]


# =============================================================================
# B3 — helper unit tests
# =============================================================================


def _same_mock_twice() -> tuple[object, object]:
    """One FRESH `MagicMock` in both positions — equal by identity, not a number.

    A module-scope mock would be shared mutable state across parametrized runs, so
    call history would leak between cases.
    """
    mock = MagicMock()
    return (mock,), (mock,)


_DB = _POWER_LOG_DEADBAND_W

# (label, values_factory, delta_seconds, deadband, second_call_logs)
# `values_factory` returns `(first_value, second_value)`. It is a factory rather
# than a pair of literals so any case needing a fresh mutable object gets one per
# parametrized run.
_B3_CASES = [
    ("no_deadband_equal_inside_window", lambda: (1, 1), 7, None, False),
    ("no_deadband_changed_inside_window", lambda: (1, 2), 7, None, True),
    ("no_deadband_unchanged_after_window", lambda: (1, 1), _RELOG_UNCHANGED_AFTER_S, None, True),
    ("small_backward_jump_stays_silent", lambda: (1, 1), -5, None, False),
    ("large_backward_jump_relogs", lambda: (1, 1), -1000, None, True),
    ("deadband_sub_threshold_silent", lambda: ((1000.0, 500.0, 0.0), (1049.0, 500.0, 0.0)), 7, _DB, False),
    ("deadband_one_element_over_threshold_logs", lambda: ((1000.0, 500.0, 0.0), (1000.0, 650.0, 0.0)), 7, _DB, True),
    ("deadband_non_tuple_current_logs", lambda: ((1000.0,), 1000.0), 7, _DB, True),
    ("deadband_non_tuple_prev_logs", lambda: (1000.0, (1000.0,)), 7, _DB, True),
    ("deadband_length_mismatch_logs", lambda: ((1000.0,), (1000.0, 2.0)), 7, _DB, True),
    ("deadband_bool_element_logs", lambda: ((True,), (True,)), 7, _DB, True),
    ("deadband_none_element_logs", lambda: ((None,), (None,)), 7, _DB, True),
    ("deadband_mock_element_logs", _same_mock_twice, 7, _DB, True),
    # NH1: a persistently-NaN sensor must NOT re-inflate the log to full cycle rate.
    ("deadband_nan_to_nan_silent", lambda: ((math.nan, 500.0, 0.0), (math.nan, 500.0, 0.0)), 7, _DB, False),
    ("deadband_nan_to_finite_logs", lambda: ((math.nan, 500.0, 0.0), (1000.0, 500.0, 0.0)), 7, _DB, True),
    ("deadband_finite_to_nan_logs", lambda: ((1000.0, 500.0, 0.0), (math.nan, 500.0, 0.0)), 7, _DB, True),
    # A stuck-at-inf sensor is the same failure mode (`inf - inf` is NaN).
    ("deadband_inf_to_inf_silent", lambda: ((math.inf, 500.0, 0.0), (math.inf, 500.0, 0.0)), 7, _DB, False),
    ("deadband_inf_to_finite_logs", lambda: ((math.inf, 500.0, 0.0), (1000.0, 500.0, 0.0)), 7, _DB, True),
    # NH4/MF1: a sign flip is the operationally meaningful boundary (export vs
    # import) and beats the deadband — but ONLY above a magnitude floor. An unbounded
    # flip term makes near-zero dither log every cycle, which is the same
    # full-cycle-rate re-inflation the NaN guard above exists to prevent.
    ("deadband_sub_threshold_sign_flip_silent", lambda: ((20.0, 500.0, 0.0), (-20.0, 500.0, 0.0)), 7, _DB, False),
    ("deadband_supra_threshold_sign_flip_logs", lambda: ((150.0, 500.0, 0.0), (-150.0, 500.0, 0.0)), 7, _DB, True),
    # One side outside the floor is enough: a real export -> import transition
    # necessarily crosses the deadband on one side.
    ("deadband_sign_flip_one_side_over_floor_logs", lambda: ((10.0, 500.0, 0.0), (-120.0, 500.0, 0.0)), 7, _DB, True),
    ("deadband_sub_threshold_same_sign_silent", lambda: ((20.0, 500.0, 0.0), (40.0, 500.0, 0.0)), 7, _DB, False),
    # NH-J: zero sits on the negative side of `> 0`, so the sign test alone is
    # asymmetric about zero. The magnitude floor must make both directions behave
    # identically — small moves silent, large moves logged.
    ("deadband_zero_to_small_negative_silent", lambda: ((0.0, 500.0, 0.0), (-50.0, 500.0, 0.0)), 7, _DB, False),
    ("deadband_zero_to_small_positive_silent", lambda: ((0.0, 500.0, 0.0), (50.0, 500.0, 0.0)), 7, _DB, False),
    ("deadband_zero_to_large_negative_logs", lambda: ((0.0, 500.0, 0.0), (-150.0, 500.0, 0.0)), 7, _DB, True),
    ("deadband_zero_to_large_positive_logs", lambda: ((0.0, 500.0, 0.0), (150.0, 500.0, 0.0)), 7, _DB, True),
]


@pytest.mark.parametrize(
    ("values_factory", "delta_seconds", "deadband", "second_call_logs"),
    [pytest.param(*case[1:], id=case[0]) for case in _B3_CASES],
)
def test_log_info_on_change_emission_predicate(
    caplog: pytest.LogCaptureFixture,
    values_factory,
    delta_seconds: int,
    deadband: float | None,
    second_call_logs: bool,
) -> None:
    """B3 cases 1-11 plus the NaN/inf and sign-flip cases."""
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    host = _Host()
    first_value, second_value = values_factory()

    # Case 1: lazy init — `_log_on_change_state` is `None`, so the first call logs.
    host.log_info_on_change("k", first_value, T0, "probe %s", "one", deadband=deadband)
    assert len(_messages(caplog, "probe", logging.INFO, CHARGER_LOGGER)) == 1

    # Case 2: the state is now initialized — the second call takes the other arc.
    host.log_info_on_change(
        "k", second_value, T0 + timedelta(seconds=delta_seconds), "probe %s", "two", deadband=deadband
    )
    expected = 2 if second_call_logs else 1
    assert len(_messages(caplog, "probe", logging.INFO, CHARGER_LOGGER)) == expected


def test_log_info_on_change_survives_a_naive_datetime(caplog: pytest.LogCaptureFixture) -> None:
    """NH2: a logging helper must never be load-bearing.

    Mixing naive and aware datetimes under one key raises `TypeError` on the
    subtraction; that exception would propagate out of `dyn_handle` and kill the
    whole budgeting cycle. Bail to logging instead.
    """
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    host = _Host()

    host.log_info_on_change("k", 1, T0, "naive %s", "aware")
    host.log_info_on_change("k", 1, T0.replace(tzinfo=None), "naive %s", "naive")

    assert len(_messages(caplog, "naive", logging.INFO, CHARGER_LOGGER)) == 2
    # The key is re-stamped with the naive value, so the reverse direction is safe too.
    host.log_info_on_change("k", 1, T0, "naive %s", "aware again")
    assert len(_messages(caplog, "naive", logging.INFO, CHARGER_LOGGER)) == 3


def test_log_info_on_change_keeps_state_per_instance(caplog: pytest.LogCaptureFixture) -> None:
    """B3 case 12: an immutable class default cannot leak state between instances."""
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    first = _Host()
    second = _Host()

    first.log_info_on_change("shared_key", "value", T0, "per instance %s", "a")
    second.log_info_on_change("shared_key", "value", T0, "per instance %s", "b")

    assert len(_messages(caplog, "per instance", logging.INFO, CHARGER_LOGGER)) == 2
    assert LogOnChangeMixin._log_on_change_state is None


def test_log_info_on_change_only_mutates_its_own_state(caplog: pytest.LogCaptureFixture) -> None:
    """B3 case 13: returns `None` and adds exactly one attribute to the instance."""
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    host = _Host()
    before = copy.deepcopy(vars(host))
    assert "_log_on_change_state" not in before

    assert host.log_info_on_change("k", "v", T0, "only state %s", "x") is None

    after = vars(host)
    assert set(after) - set(before) == {"_log_on_change_state"}
    assert set(before) - set(after) == set()
    assert host._log_on_change_state == {"k": ("v", T0)}


def test_log_info_on_change_resets_the_timer_on_every_emission(caplog: pytest.LogCaptureFixture) -> None:
    """The 900 s constant bounds the gap between emissions, not a fixed cadence."""
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    host = _Host()

    host.log_info_on_change("k", 1, T0, "timer %s", 1)
    # A change re-stamps the timer, so 899 s after the change is still inside the window.
    changed_at = T0 + timedelta(seconds=_RELOG_UNCHANGED_AFTER_S - 1)
    host.log_info_on_change("k", 2, changed_at, "timer %s", 2)
    host.log_info_on_change("k", 2, changed_at + timedelta(seconds=7), "timer %s", 3)

    assert len(_messages(caplog, "timer", logging.INFO, CHARGER_LOGGER)) == 2


def test_log_info_on_change_keys_are_independent(caplog: pytest.LogCaptureFixture) -> None:
    """Two keys on one instance memo independently."""
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    host = _Host()

    host.log_info_on_change("a", 1, T0, "keyed %s", "a")
    host.log_info_on_change("b", 1, T0, "keyed %s", "b")
    host.log_info_on_change("a", 1, T0 + timedelta(seconds=7), "keyed %s", "a")

    assert len(_messages(caplog, "keyed", logging.INFO, CHARGER_LOGGER)) == 2


# =============================================================================
# dyn_handle drivers — S1, S2, S3, S4, S13 (B1) and S10 (B2)
# =============================================================================


def _power_series(value: float, time: datetime) -> list[tuple[datetime, float, dict]]:
    """A non-empty, CONSTANT series: a varying one drifts the weighted mean."""
    return [(time - timedelta(seconds=30), value, {}), (time, value, {})]


def _make_dyn_handle_group(
    num_chargers: int = 2,
    cooldown_seconds: float | None = None,
    battery=None,
    available_power_w: float | None = None,
    grid_power_w: float = 500.0,
):
    """Build a group whose `dyn_handle` reaches the budgeting branch.

    `ensure_correct_state` is stubbed (the S9 driver below uses the real one).
    """
    hass = make_hass()
    home = make_home(battery=battery)
    past = T0 - timedelta(seconds=CHARGER_ADAPTATION_WINDOW_S + 100)

    chargers = []
    statuses = []
    for index in range(num_chargers):
        charger = create_charger(hass, home, name=f"Ch{index}")
        car = make_real_car(hass, home, name=f"Car{index}")
        init_charger_states(charger, charge_state=True, amperage=10, num_phases=1)
        plug_car(charger, car, past)
        charger.is_charging_power_zero = MagicMock(return_value=False)
        charger.qs_enable_device = True
        charger.update_car_dampening_value = MagicMock()
        if cooldown_seconds is not None:
            charger._last_amp_change_time = T0 - timedelta(seconds=cooldown_seconds)
        chargers.append(charger)

        status = QSChargerStatus(charger)
        status.current_real_max_charging_amp = 10
        status.current_active_phase_number = 1
        status.accurate_current_power = 2300.0
        statuses.append(status)

    group = make_charger_group(home, chargers)
    group.ensure_correct_state = AsyncMock(return_value=(statuses, past))
    group.dynamic_group.get_median_sensor = MagicMock(return_value=4600.0)
    group.apply_budgets = AsyncMock()
    group.apply_budget_strategy = AsyncMock()
    group.budgeting_algorithm_minimize_diffs = AsyncMock(return_value=(True, False, False))
    group.remaining_budget_to_apply = []

    if available_power_w is not None:
        home.get_available_power_values = MagicMock(side_effect=lambda _w, t: _power_series(available_power_w, t))
        home.get_grid_consumption_power_values = MagicMock(side_effect=lambda _w, t: _power_series(grid_power_w, t))

    return group, chargers, home


async def test_s1_dyn_handle_start_is_debug(caplog: pytest.LogCaptureFixture) -> None:
    """B1/S1: `dyn_handle: START` is unconditional — it must never be INFO."""
    caplog.set_level(logging.DEBUG, logger=CHARGER_LOGGER)
    group, _, _ = _make_dyn_handle_group()

    await group.dyn_handle(T0)

    assert _messages(caplog, "dyn_handle: START", logging.INFO, CHARGER_LOGGER) == []
    assert len(_messages(caplog, "dyn_handle: START", logging.DEBUG, CHARGER_LOGGER)) == 1


async def test_s4_cannot_dampen_simple_case_is_debug(caplog: pytest.LogCaptureFixture) -> None:
    """B1/S4: two truly-charging chargers means the simple case does not apply."""
    caplog.set_level(logging.DEBUG, logger=CHARGER_LOGGER)
    group, _, _ = _make_dyn_handle_group(num_chargers=2)

    await group.dyn_handle(T0)

    assert _messages(caplog, "can't dampen simple case", logging.INFO, CHARGER_LOGGER) == []
    assert len(_messages(caplog, "can't dampen simple case", logging.DEBUG, CHARGER_LOGGER)) == 1


async def test_s2_and_s3_cooldown_are_debug(caplog: pytest.LogCaptureFixture) -> None:
    """B1/S2+S3: one actionable charger in cooldown means one skip plus one all-in-cooldown."""
    caplog.set_level(logging.DEBUG, logger=CHARGER_LOGGER)
    group, _, _ = _make_dyn_handle_group(num_chargers=1, cooldown_seconds=10)

    await group.dyn_handle(T0)

    assert _messages(caplog, "amp change cooldown", logging.INFO, CHARGER_LOGGER) == []
    assert len(_messages(caplog, "amp change cooldown", logging.DEBUG, CHARGER_LOGGER)) == 1
    assert _messages(caplog, "all chargers in cooldown", logging.INFO, CHARGER_LOGGER) == []
    assert len(_messages(caplog, "all chargers in cooldown", logging.DEBUG, CHARGER_LOGGER)) == 1
    group.budgeting_algorithm_minimize_diffs.assert_not_awaited()


async def test_s13_no_actionable_chargers_is_debug(caplog: pytest.LogCaptureFixture) -> None:
    """B1/S13: the `do nothing` branch is mutually exclusive with S9/S10's."""
    caplog.set_level(logging.DEBUG, logger=CHARGER_LOGGER)
    group, _, _ = _make_dyn_handle_group(num_chargers=1)
    group.ensure_correct_state = AsyncMock(return_value=([], None))

    await group.dyn_handle(T0)

    assert _messages(caplog, "no actionable chargers, do nothing", logging.INFO, CHARGER_LOGGER) == []
    assert len(_messages(caplog, "no actionable chargers, do nothing", logging.DEBUG, CHARGER_LOGGER)) == 1
    assert _messages(caplog, "battery_asked_charge", logging.INFO, CHARGER_LOGGER) == []


async def test_s10_available_power_logged_once_per_window(caplog: pytest.LogCaptureFixture) -> None:
    """B2/S10: unchanged power over 10 cycles emits once; +900 s and a change re-emit."""
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    group, _, home = _make_dyn_handle_group(num_chargers=2, battery=make_battery(0.0), available_power_w=1000.0)

    for cycle in range(10):
        await group.dyn_handle(T0 + timedelta(seconds=7 * cycle))
    assert len(_messages(caplog, "battery_asked_charge", logging.INFO, CHARGER_LOGGER)) == 1

    await group.dyn_handle(T0 + timedelta(seconds=_RELOG_UNCHANGED_AFTER_S + 63))
    assert len(_messages(caplog, "battery_asked_charge", logging.INFO, CHARGER_LOGGER)) == 2

    home.get_available_power_values = MagicMock(side_effect=lambda _w, t: _power_series(5000.0, t))
    await group.dyn_handle(T0 + timedelta(seconds=_RELOG_UNCHANGED_AFTER_S + 70))
    assert len(_messages(caplog, "battery_asked_charge", logging.INFO, CHARGER_LOGGER)) == 3


async def test_s10_deadband_absorbs_sub_threshold_jitter(caplog: pytest.LogCaptureFixture) -> None:
    """B2/S10: a walk whose every step is <100 W emits once; a >=100 W step emits again."""
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    group, _, home = _make_dyn_handle_group(num_chargers=2, battery=make_battery(0.0), available_power_w=1000.0)

    jitter = [1000.0, 1049.0, 1051.0, 1049.0, 1055.0, 1049.0, 1051.0, 1060.0, 1049.0, 1051.0]
    for cycle, watts in enumerate(jitter):
        home.get_available_power_values = MagicMock(side_effect=lambda _w, t, v=watts: _power_series(v, t))
        await group.dyn_handle(T0 + timedelta(seconds=7 * cycle))
    assert len(_messages(caplog, "battery_asked_charge", logging.INFO, CHARGER_LOGGER)) == 1

    home.get_available_power_values = MagicMock(side_effect=lambda _w, t: _power_series(1051.0 + 100.0, t))
    await group.dyn_handle(T0 + timedelta(seconds=7 * len(jitter)))
    assert len(_messages(caplog, "battery_asked_charge", logging.INFO, CHARGER_LOGGER)) == 2


async def _run_oscillating_power_cycles(group, home, watts: float, cycles: int = 10) -> None:
    """Drive `dyn_handle` with the power series alternating +watts / -watts."""
    for cycle in range(cycles):
        value = watts if cycle % 2 == 0 else -watts
        home.get_available_power_values = MagicMock(side_effect=lambda _w, t, v=value: _power_series(v, t))
        home.get_grid_consumption_power_values = MagicMock(side_effect=lambda _w, t, v=value: _power_series(v, t))
        await group.dyn_handle(T0 + timedelta(seconds=7 * cycle))


async def test_mf1_near_zero_sign_dither_does_not_re_inflate_the_log(caplog: pytest.LogCaptureFixture) -> None:
    """MF1: an UNBOUNDED sign-flip term logs every ~7 s forever near zero.

    Near-zero grid power is the steady state of a well-regulated self-consumption
    home, so this is the common case, not an exotic one. |delta| is 40 W here, well
    inside the 100 W deadband, yet every cycle flips sign. Without a magnitude floor
    this is exactly the full-cycle-rate volume the PR exists to remove — on its
    highest-volume site.
    """
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    group, _, home = _make_dyn_handle_group(num_chargers=2, battery=make_battery(0.0), available_power_w=1000.0)

    await _run_oscillating_power_cycles(group, home, watts=20.0)

    assert len(_messages(caplog, "battery_asked_charge", logging.INFO, CHARGER_LOGGER)) == 1


async def test_mf1_real_export_import_transition_is_still_logged(caplog: pytest.LogCaptureFixture) -> None:
    """MF1: the floor must not cost us a genuine export <-> import transition."""
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    group, _, home = _make_dyn_handle_group(num_chargers=2, battery=make_battery(0.0), available_power_w=1000.0)

    await _run_oscillating_power_cycles(group, home, watts=150.0)

    # Every cycle crosses zero with both sides outside the deadband, so every cycle
    # is a real transition and every cycle is reported.
    assert len(_messages(caplog, "battery_asked_charge", logging.INFO, CHARGER_LOGGER)) == 10


# =============================================================================
# S5, S6 — charger-side rule-2 demotions (B1)
# =============================================================================


def _make_stable_status_charger():
    """A plugged, enabled, available charger — S5's branch precondition."""
    hass = make_hass()
    home = make_home()
    charger = create_charger(hass, home)
    car = make_real_car(hass, home)
    past = T0 - timedelta(hours=2)

    init_charger_states(charger, charge_state=True, amperage=10, num_phases=1)
    plug_car(charger, car, past)

    charger.qs_enable_device = True
    charger.current_command = copy_command(CMD_AUTO_GREEN_ONLY)
    charger.is_not_plugged = MagicMock(return_value=False)
    charger.is_charger_unavailable = MagicMock(return_value=False)
    charger._probe_and_enforce_stopped_charge_command_state = MagicMock(return_value=False)
    charger.get_median_sensor = MagicMock(return_value=1000.0)
    charger.get_current_active_constraint = MagicMock(return_value=None)
    charger.can_do_3_to_1_phase_switch = MagicMock(return_value=False)
    charger._expected_charge_state.last_change_asked = past
    charger._expected_charge_state.last_time_set = past
    charger._expected_num_active_phases.last_change_asked = past
    charger._expected_num_active_phases.last_time_set = past

    return hass, home, charger, car


def test_s5_stable_status_dump_is_debug(caplog: pytest.LogCaptureFixture) -> None:
    """B1/S5: the per-cycle state dump moves to DEBUG."""
    caplog.set_level(logging.DEBUG, logger=CHARGER_LOGGER)
    *_, charger, _ = _make_stable_status_charger()

    assert charger.get_stable_dynamic_charge_status(T0) is not None

    assert _messages(caplog, "possible_amps:", logging.INFO, CHARGER_LOGGER) == []
    assert len(_messages(caplog, "possible_amps:", logging.DEBUG, CHARGER_LOGGER)) == 1


async def _drive_remove_all_person_constraints() -> None:
    """Reach S6: a plugged car charged past a person's minimum target."""
    hass = make_hass()
    home = make_home()
    charger = create_charger(hass, home)
    car = make_real_car(hass, home)
    init_charger_states(charger)
    charger.is_charger_unavailable = MagicMock(return_value=False)
    charger.probe_for_possible_needed_reboot = MagicMock(return_value=False)
    charger.is_not_plugged = MagicMock(return_value=False)
    charger.is_plugged = MagicMock(return_value=True)
    charger.set_charging_num_phases = AsyncMock(return_value=False)
    charger.set_max_charging_current = AsyncMock(return_value=True)
    charger.reboot = AsyncMock()
    plug_car(charger, car, T0)
    charger.get_best_car = MagicMock(return_value=car)
    car.get_car_charge_percent = lambda time=None, *a, **kw: 80.0
    charger.is_car_stopped_asking_current = MagicMock(return_value=True)

    person = MagicMock()
    person.name = "Dave"
    person.notify_of_forecast_if_needed = AsyncMock()

    next_usage_time = T0 + timedelta(hours=10)
    person_min_target_charge = 50.0
    old_person_ct = MultiStepsPowerLoadConstraintChargePercent(
        total_capacity_wh=60000,
        type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        time=T0 - timedelta(hours=1),
        load=charger,
        load_param=car.name,
        from_user=False,
        end_of_constraint=next_usage_time,
        initial_value=30.0,
        target_value=person_min_target_charge,
        power_steps=charger._power_steps,
        support_auto=True,
    )
    old_person_ct.load_info = {"person": "Dave"}
    charger.push_live_constraint(T0 - timedelta(hours=1), old_person_ct)

    car.get_best_person_next_need = AsyncMock(return_value=(False, next_usage_time, person_min_target_charge, person))
    car.get_next_scheduled_event = AsyncMock(return_value=(None, None))
    car.do_next_charge_time = None
    car.do_force_next_charge = False
    charger._auto_constraints_cleaned_at_user_reset = []

    await charger.check_load_activity_and_constraints(T0)


async def test_s6_remove_all_person_constraints_is_debug(caplog: pytest.LogCaptureFixture) -> None:
    """B1/S6: the announcement of the person-constraint cleanup moves to DEBUG."""
    caplog.set_level(logging.DEBUG, logger=CHARGER_LOGGER)

    await _drive_remove_all_person_constraints()

    assert _messages(caplog, "do_remove_all_person_constraints", logging.INFO, CHARGER_LOGGER) == []
    assert len(_messages(caplog, "do_remove_all_person_constraints", logging.DEBUG, CHARGER_LOGGER)) == 1


# =============================================================================
# S7, S8 — home_model/load.py (B1 and B1b)
# =============================================================================

_S8_FRAGMENT = "Constraint Reset device"


def test_s7_no_bad_constraint_found_is_debug(caplog: pytest.LogCaptureFixture) -> None:
    """B1/S7: `no reset needed` is a no-op announcement."""
    caplog.set_level(logging.DEBUG, logger=LOAD_LOGGER)
    load = MinimalTestLoad(name="S7Load")
    caplog.clear()

    assert (
        load.clean_constraints_for_load_param_and_if_same_key_same_value_info(
            T0, load_param="CarA", load_info=None, for_full_reset=False
        )
        is False
    )

    assert _messages(caplog, "No bad constraint found", logging.INFO, LOAD_LOGGER) == []
    assert len(_messages(caplog, "No bad constraint found", logging.DEBUG, LOAD_LOGGER)) == 1


def test_b1b_fresh_load_construction_logs_the_no_op_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    """B1b: constructing a load raises nothing and takes case (d) — nothing to reset."""
    caplog.set_level(logging.DEBUG, logger=LOAD_LOGGER)

    load = MinimalTestLoad(name="FreshLoad")

    assert load.name == "FreshLoad"
    assert _messages(caplog, _S8_FRAGMENT, logging.INFO, LOAD_LOGGER) == []
    assert len(_messages(caplog, _S8_FRAGMENT, logging.DEBUG, LOAD_LOGGER)) == 1
    assert "nothing to reset" in _messages(caplog, _S8_FRAGMENT, logging.DEBUG, LOAD_LOGGER)[0].getMessage()


@pytest.mark.parametrize(
    ("case", "keep_commands", "expected_level"),
    [
        pytest.param("constraints", True, logging.INFO, id="a_non_empty_constraints"),
        pytest.param("last_completed", True, logging.INFO, id="b_last_completed_constraint"),
        pytest.param("command", False, logging.INFO, id="c_command_dropped"),
        pytest.param("nothing", True, logging.DEBUG, id="d_nothing_to_reset"),
        pytest.param("command", True, logging.DEBUG, id="e_command_kept"),
        # NH5: `keep_commands=False` also drops an in-flight `running_command`.
        pytest.param("running_command", False, logging.INFO, id="f_running_command_dropped"),
        pytest.param("running_command", True, logging.DEBUG, id="g_running_command_kept"),
        # NH-A: ... and a queued `_stacked_command`, on the same reasoning.
        pytest.param("stacked_command", False, logging.INFO, id="h_stacked_command_dropped"),
        pytest.param("stacked_command", True, logging.DEBUG, id="i_stacked_command_kept"),
    ],
)
def test_b1b_s8_logs_info_only_when_work_is_performed(
    caplog: pytest.LogCaptureFixture, case: str, keep_commands: bool, expected_level: int
) -> None:
    """B1b: INFO iff the reset actually had something to reset."""
    caplog.set_level(logging.DEBUG, logger=LOAD_LOGGER)
    load = MinimalTestLoad(name=f"S8Load{case}")
    load._constraints = []
    load._last_completed_constraint = None
    load.current_command = None
    load.running_command = None
    load._stacked_command = None

    if case == "constraints":
        load._constraints = [create_constraint(load=load, time=T0)]
    elif case == "last_completed":
        load._last_completed_constraint = create_constraint(load=load, time=T0)
    elif case == "command":
        load.current_command = copy_command(CMD_ON, power_consign=1000.0)
    elif case == "running_command":
        load.running_command = copy_command(CMD_ON, power_consign=1000.0)
    elif case == "stacked_command":
        load._stacked_command = copy_command(CMD_ON, power_consign=1000.0)

    caplog.clear()
    load.constraint_reset_and_reset_commands_if_needed(keep_commands=keep_commands)

    other_level = logging.DEBUG if expected_level == logging.INFO else logging.INFO
    assert len(_messages(caplog, _S8_FRAGMENT, expected_level, LOAD_LOGGER)) == 1
    assert _messages(caplog, _S8_FRAGMENT, other_level, LOAD_LOGGER) == []


@pytest.mark.parametrize(
    ("force_next_charge", "next_charge_time", "expected_level"),
    [
        pytest.param(True, None, logging.INFO, id="force_next_charge_flag"),
        pytest.param(False, T0 + timedelta(hours=3), logging.INFO, id="next_charge_time"),
        pytest.param(False, None, logging.DEBUG, id="nothing_set"),
    ],
)
def test_sf2_charger_reset_logs_info_when_it_clears_a_user_flag(
    caplog: pytest.LogCaptureFixture,
    force_next_charge: bool,
    next_charge_time: datetime | None,
    expected_level: int,
) -> None:
    """SF2: `QSChargerGeneric`'s override destroys user-initiated flags.

    The base predicate must see them, else a user-initiated action is wiped while
    the reset reports "nothing to reset" at DEBUG.
    """
    caplog.set_level(logging.DEBUG, logger=LOAD_LOGGER)
    hass = make_hass()
    home = make_home()
    charger = create_charger(hass, home, name="Sf2Ch")
    car = make_real_car(hass, home, name="Sf2Car")
    init_charger_states(charger)
    plug_car(charger, car, T0 - timedelta(hours=1))

    charger._constraints = []
    charger._last_completed_constraint = None
    charger.current_command = None
    charger.running_command = None
    car.do_force_next_charge = force_next_charge
    car.do_next_charge_time = next_charge_time

    caplog.clear()
    charger.constraint_reset_and_reset_commands_if_needed(keep_commands=True)

    other_level = logging.DEBUG if expected_level == logging.INFO else logging.INFO
    assert len(_messages(caplog, _S8_FRAGMENT, expected_level, LOAD_LOGGER)) == 1
    assert _messages(caplog, _S8_FRAGMENT, other_level, LOAD_LOGGER) == []
    # The override still clears the flags either way.
    assert car.do_force_next_charge is False
    assert car.do_next_charge_time is None


def test_sf2_base_hook_survives_a_carless_charger(caplog: pytest.LogCaptureFixture) -> None:
    """SF2: the charger override must use `getattr` — `car` is absent during __init__."""
    caplog.set_level(logging.DEBUG, logger=LOAD_LOGGER)
    hass = make_hass()
    home = make_home()

    charger = create_charger(hass, home, name="Sf2NoCar")

    assert charger.car is None
    assert charger._has_state_to_reset(keep_commands=True) is False


# =============================================================================
# S9 — ensure_correct_state, per-charger key (B2)
# =============================================================================


def _make_ensure_correct_state_group(num_chargers: int = 2):
    """Drive the REAL `QSChargerGroup.ensure_correct_state`; stub the children only."""
    hass = make_hass()
    home = make_home()
    chargers = []
    for index in range(num_chargers):
        charger = create_charger(hass, home, name=f"EcsCh{index}")
        charger.qs_enable_device = True
        charger.ensure_correct_state = AsyncMock(return_value=(True, False, T0))
        charger.get_stable_dynamic_charge_status = MagicMock(return_value=None)
        chargers.append(charger)

    return make_charger_group(home, chargers), chargers


async def test_s9_correct_state_logged_once_per_charger_per_window(caplog: pytest.LogCaptureFixture) -> None:
    """B2/S9: one record PER CHARGER over 10 cycles; +900 s and a change re-emit."""
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    group, chargers = _make_ensure_correct_state_group(num_chargers=2)

    for cycle in range(10):
        await group.ensure_correct_state(T0 + timedelta(seconds=7 * cycle))
    records = _messages(caplog, "ensure_correct_state dyn group", logging.INFO, CHARGER_LOGGER)
    assert len(records) == 2
    for charger in chargers:
        assert len([r for r in records if charger.name in r.getMessage()]) == 1

    await group.ensure_correct_state(T0 + timedelta(seconds=_RELOG_UNCHANGED_AFTER_S + 63))
    assert len(_messages(caplog, "ensure_correct_state dyn group", logging.INFO, CHARGER_LOGGER)) == 4

    # A change is not deferred to the next tick, even well inside the window.
    for charger in chargers:
        charger.ensure_correct_state = AsyncMock(return_value=(True, True, T0))
    await group.ensure_correct_state(T0 + timedelta(seconds=_RELOG_UNCHANGED_AFTER_S + 70))
    assert len(_messages(caplog, "ensure_correct_state dyn group", logging.INFO, CHARGER_LOGGER)) == 6


async def test_s9_probe_only_never_logs_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """B2/S9: the defensive `probe_only` early-out reports at DEBUG only."""
    caplog.set_level(logging.DEBUG, logger=CHARGER_LOGGER)
    group, chargers = _make_ensure_correct_state_group(num_chargers=2)

    for cycle in range(3):
        await group.ensure_correct_state(T0 + timedelta(seconds=7 * cycle), probe_only=True)

    assert _messages(caplog, "ensure_correct_state dyn group", logging.INFO, CHARGER_LOGGER) == []
    assert len(_messages(caplog, "ensure_correct_state dyn group", logging.DEBUG, CHARGER_LOGGER)) == 6


async def test_nhb_re_enabling_a_charger_inside_the_window_emits_a_marker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """NH-B: the `qs_enable_device is False` early-out must EVICT the memo key.

    Otherwise the entry goes stale, and a charger re-enabled inside 900 s with the
    same `(res, handled_static)` pair produces no line — the log carries no marker
    that it came back under management. `detach_car()` gives the charger-level memo
    this cleanup; the group needs its own.
    """
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    group, chargers = _make_ensure_correct_state_group(num_chargers=1)
    charger = chargers[0]

    await group.ensure_correct_state(T0)
    assert len(_messages(caplog, "ensure_correct_state dyn group", logging.INFO, CHARGER_LOGGER)) == 1

    charger.qs_enable_device = False
    await group.ensure_correct_state(T0 + timedelta(seconds=7))
    # The disabled charger reports nothing at all, by design.
    assert len(_messages(caplog, "ensure_correct_state dyn group", logging.INFO, CHARGER_LOGGER)) == 1

    charger.qs_enable_device = True
    await group.ensure_correct_state(T0 + timedelta(seconds=14))
    assert len(_messages(caplog, "ensure_correct_state dyn group", logging.INFO, CHARGER_LOGGER)) == 2


# =============================================================================
# S11 — get_best_car (B2)
# =============================================================================


def _make_best_car_charger():
    """A charger whose `get_best_car` lands on the computed-best-car branch."""
    hass = make_hass()
    home = make_home()
    charger = create_charger(hass, home, name="BestCh")
    car_a = make_real_car(hass, home, name="CarA")
    car_b = make_real_car(hass, home, name="CarB")
    init_charger_states(charger, charge_state=True, amperage=10)
    plug_car(charger, car_a, T0 - timedelta(hours=1))
    charger.clear_user_originated(USER_ORIGINATED_CAR_NAME)
    charger._boot_car = None
    charger.is_plugged = MagicMock(return_value=True)
    return hass, home, charger, car_a, car_b


async def test_s11_best_car_logged_once_per_window(caplog: pytest.LogCaptureFixture) -> None:
    """B2/S11: the key is the car name only — a per-cycle score must not re-emit."""
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    _, _, charger, car_a, car_b = _make_best_car_charger()

    scores = {"CarA": 10.0, "CarB": 5.0}
    drift = {"n": 0}

    def _score(car, time, cache):
        # The score drifts every cycle: keying on it would preserve all 71k lines.
        drift["n"] += 1
        return scores[car.name] + drift["n"] * 0.001

    charger.get_car_score = MagicMock(side_effect=_score)

    for cycle in range(10):
        assert charger.get_best_car(T0 + timedelta(seconds=7 * cycle)).name == "CarA"
    assert len(_messages(caplog, "with score", logging.INFO, CHARGER_LOGGER)) == 1

    assert charger.get_best_car(T0 + timedelta(seconds=_RELOG_UNCHANGED_AFTER_S + 63)).name == "CarA"
    assert len(_messages(caplog, "with score", logging.INFO, CHARGER_LOGGER)) == 2

    scores["CarB"] = 100.0
    assert charger.get_best_car(T0 + timedelta(seconds=_RELOG_UNCHANGED_AFTER_S + 70)).name == car_b.name
    records = _messages(caplog, "with score", logging.INFO, CHARGER_LOGGER)
    assert len(records) == 3
    # NH7: the records must name the car that actually won each time.
    assert [car_a.name, car_a.name, car_b.name] == [r.args[0] for r in records]


async def test_sf1_car_swap_is_not_silenced_by_the_memo(caplog: pytest.LogCaptureFixture) -> None:
    """SF1: `detach_car()` must invalidate the memo, else a swap names the wrong car.

    Car A unplugged and car B plugged into the same charger inside the 900 s window
    yields the same memoized tuple, so without the clear the line is suppressed and
    the last INFO record names the WRONG car.
    """
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    charger, constraint = _make_soc_callback_charger()
    car_a = charger.car
    hass, home = charger.hass, charger.home

    await charger.constraint_update_value_callback_percent_soc(constraint, T0)
    first = _messages(caplog, _S12_FRAGMENT, logging.INFO, CHARGER_LOGGER)
    assert len(first) == 1
    assert car_a.name in first[0].getMessage()

    charger.detach_car()
    car_b = make_real_car(hass, home, name="SwapCar")
    plug_car(charger, car_b, T0)
    car_b.get_car_charge_percent_raw_sensor = MagicMock(return_value=52.0)
    car_b.is_car_charge_growing = MagicMock(return_value=True)
    car_b.setup_car_charge_target_if_needed = AsyncMock()

    # Well inside the window, and the reported tuple is identical to car A's.
    await charger.constraint_update_value_callback_percent_soc(constraint, T0 + timedelta(seconds=7))

    records = _messages(caplog, _S12_FRAGMENT, logging.INFO, CHARGER_LOGGER)
    assert len(records) == 2
    assert car_b.name in records[1].getMessage()


def test_sf1_detach_car_clears_the_memo() -> None:
    """SF1: the clear is unconditional — it also marks a new session on replug."""
    hass = make_hass()
    home = make_home()
    charger = create_charger(hass, home, name="DetachCh")
    car = make_real_car(hass, home, name="DetachCar")
    init_charger_states(charger)
    plug_car(charger, car, T0)
    charger.log_info_on_change("get_best_car", car.name, T0, "seed %s", car.name)
    assert charger._log_on_change_state

    charger.detach_car()

    assert charger._log_on_change_state is None


# =============================================================================
# S12 — constraint_update_value_callback_soc (B2)
# =============================================================================

_S12_FRAGMENT = "is_car_charged"


def _make_soc_callback_charger():
    """The real-charger SOC-callback driver: constant inputs across invocations."""
    hass = make_hass()
    home = make_home()
    charger = create_charger(hass, home, name="SocCh")
    car = make_real_car(hass, home, name="SocCar")
    init_charger_states(charger, charge_state=True, amperage=10, num_phases=1)
    plug_car(charger, car, T0 - timedelta(hours=2))

    charger.current_command = copy_command(CMD_AUTO_GREEN_ONLY)
    charger.is_not_plugged = MagicMock(return_value=False)
    # `return_value`, never `side_effect`: a finite list raises StopIteration on cycle 2.
    charger._compute_added_charge_update = MagicMock(return_value=6.0)
    charger._do_update_charger_state = AsyncMock()
    charger.charger_group.dyn_handle = AsyncMock()

    car.car_charge_percent_sensor = "sensor.car_soc"
    car.get_car_charge_percent_raw_sensor = MagicMock(return_value=52.0)
    car.is_car_charge_growing = MagicMock(return_value=True)
    car.setup_car_charge_target_if_needed = AsyncMock()

    probe_charge_window = 30 * 60
    constraint = MagicMock(spec=LoadConstraint)
    constraint.first_value_update = T0 - timedelta(seconds=probe_charge_window + 100)
    constraint.last_value_update = T0
    constraint.last_value_change_update = T0 - timedelta(seconds=60)
    constraint.current_value = 50.0
    constraint.target_value = 80.0
    constraint.is_constraint_met = MagicMock(return_value=False)

    return charger, constraint


async def test_s12_soc_callback_logged_once_per_mode_per_window(caplog: pytest.LogCaptureFixture) -> None:
    """B2/S12: percent and energy modes memo separately; +900 s and a change re-emit."""
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    charger, constraint = _make_soc_callback_charger()

    for cycle in range(10):
        await charger.constraint_update_value_callback_percent_soc(constraint, T0 + timedelta(seconds=7 * cycle))
    assert len(_messages(caplog, _S12_FRAGMENT, logging.INFO, CHARGER_LOGGER)) == 1

    # The energy mode is a different key: one key would let the two callbacks thrash.
    await charger.constraint_update_value_callback_energy_soc(constraint, T0 + timedelta(seconds=70))
    assert len(_messages(caplog, _S12_FRAGMENT, logging.INFO, CHARGER_LOGGER)) == 2

    await charger.constraint_update_value_callback_percent_soc(
        constraint, T0 + timedelta(seconds=_RELOG_UNCHANGED_AFTER_S + 63)
    )
    assert len(_messages(caplog, _S12_FRAGMENT, logging.INFO, CHARGER_LOGGER)) == 3

    constraint.is_constraint_met = MagicMock(return_value=True)
    await charger.constraint_update_value_callback_percent_soc(
        constraint, T0 + timedelta(seconds=_RELOG_UNCHANGED_AFTER_S + 70)
    )
    assert len(_messages(caplog, _S12_FRAGMENT, logging.INFO, CHARGER_LOGGER)) == 4


async def test_sfg_soc_progress_numbers_are_memoized_but_quantised(caplog: pytest.LogCaptureFixture) -> None:
    """SF-G: the printed charge-progress numbers must be in the memo — "it can't be
    invisible" — but QUANTISED, or the site degenerates into logging every cycle
    (the trap MF1 fell into).
    """
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    charger, constraint = _make_soc_callback_charger()

    # Sub-unit drift: every reading rounds to 52, so the operator learns nothing new.
    for cycle, soc in enumerate([52.0, 52.1, 52.4, 52.2, 51.8, 52.3, 51.6, 52.4, 52.0, 51.9]):
        charger.car.get_car_charge_percent_raw_sensor = MagicMock(return_value=soc)
        await charger.constraint_update_value_callback_percent_soc(constraint, T0 + timedelta(seconds=7 * cycle))
    assert len(_messages(caplog, _S12_FRAGMENT, logging.INFO, CHARGER_LOGGER)) == 1

    # A whole unit of real progress is worth a line, well inside the 900 s window.
    charger.car.get_car_charge_percent_raw_sensor = MagicMock(return_value=53.0)
    await charger.constraint_update_value_callback_percent_soc(constraint, T0 + timedelta(seconds=77))
    assert len(_messages(caplog, _S12_FRAGMENT, logging.INFO, CHARGER_LOGGER)) == 2


async def test_sfg_consign_change_under_the_same_command_name_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SF-G: the memo stored only `command.command`, so a consign change was invisible.

    `LoadCommand.__eq__` compares `power_consign` too, and the message prints the whole
    command — so `auto_green` at 3000 W and at 7000 W must not share a memo entry.
    """
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    charger, constraint = _make_soc_callback_charger()

    charger.current_command = copy_command(CMD_AUTO_GREEN_ONLY, power_consign=3000.0)
    await charger.constraint_update_value_callback_percent_soc(constraint, T0)
    assert len(_messages(caplog, _S12_FRAGMENT, logging.INFO, CHARGER_LOGGER)) == 1

    charger.current_command = copy_command(CMD_AUTO_GREEN_ONLY, power_consign=7000.0)
    assert charger.current_command.command == CMD_AUTO_GREEN_ONLY.command
    await charger.constraint_update_value_callback_percent_soc(constraint, T0 + timedelta(seconds=7))

    records = _messages(caplog, _S12_FRAGMENT, logging.INFO, CHARGER_LOGGER)
    assert len(records) == 2
    assert "7000" in records[1].getMessage()


async def test_sfg_soc_volume_scales_with_amp_changes_not_cycles(caplog: pytest.LogCaptureFixture) -> None:
    """SF-G's COST, pinned: S12 volume is proportional to amp changes, not cycles.

    The story deliberately kept `power_consign` OUT of this memo (F8 calls it "a
    per-cycle budgeted float"), and review fix plan #02 SF-G put it back. That is the
    same shape as the MF1 defect — a volatile value entering a memo — so the saving
    needs an explicit assertion, not an argument. `power_consign` is safe because it
    comes from the DISCRETE power-step table, so a hold is silent and only the step
    logs.
    """
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    charger, constraint = _make_soc_callback_charger()

    # Three discrete power steps, each held for four ~7 s cycles.
    consigns = [1380.0] * 4 + [1610.0] * 4 + [1840.0] * 4
    for cycle, consign in enumerate(consigns):
        charger.current_command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=consign)
        await charger.constraint_update_value_callback_percent_soc(constraint, T0 + timedelta(seconds=7 * cycle))

    records = _messages(caplog, _S12_FRAGMENT, logging.INFO, CHARGER_LOGGER)
    assert len(records) == len(set(consigns)) == 3, "one line per amp change, not one per cycle"


async def test_sfg_soc_volume_ceiling_is_one_line_per_consign_change(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The documented CEILING of SF-G, so the cost is explicit rather than implied.

    If the consign genuinely changed on every cycle, every cycle would log. In
    production that is bounded by the 45 s amp-change cooldown (the S2 site), not by
    this memo — so the worst realistic case is ~1 line per 45 s, still far below the
    ~7 s cycle rate. If a future change adds smoothing, this test should fail and
    force a conscious decision.
    """
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    charger, constraint = _make_soc_callback_charger()

    for cycle in range(10):
        consign = 1380.0 if cycle % 2 == 0 else 1610.0
        charger.current_command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=consign)
        await charger.constraint_update_value_callback_percent_soc(constraint, T0 + timedelta(seconds=7 * cycle))

    assert len(_messages(caplog, _S12_FRAGMENT, logging.INFO, CHARGER_LOGGER)) == 10


async def test_s12_literal_percent_is_escaped(caplog: pytest.LogCaptureFixture) -> None:
    """The literal `%` must be `%%`, else `record.getMessage()` raises ValueError."""
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    charger, constraint = _make_soc_callback_charger()

    await charger.constraint_update_value_callback_percent_soc(constraint, T0)

    record = _messages(caplog, _S12_FRAGMENT, logging.INFO, CHARGER_LOGGER)[0]
    assert "(is %:True)" in record.getMessage()
    assert "%%" in record.msg


# =============================================================================
# Harness contract (NH-C, NH-D)
# =============================================================================


def test_harness_hass_config_dir_is_caller_controlled_and_unique(tmp_path: Path) -> None:
    """NH-C: a fixed shared config dir is collision-prone under parallel runs."""
    assert make_hass(config_dir=str(tmp_path)).config.config_dir == str(tmp_path)
    assert make_hass().config.config_dir != make_hass().config.config_dir


async def test_harness_executor_job_forwards_keyword_arguments() -> None:
    """NH-D: dropping kwargs would raise TypeError instead of running the job."""
    hass = make_hass()

    assert await hass.async_add_executor_job(lambda a, b=0: a + b, 1, b=2) == 3


# =============================================================================
# B4 — the keepers are still at INFO
# =============================================================================

# Regex rather than `inspect.getsource` of an enclosing function: ruff may split a
# call across lines, and this keeps the check independent of function naming.
_B4_SOURCE_ANCHORS = [
    pytest.param("ha_model/charger.py", "STOP CHARGE LAUNCHED", id="charger_stop_charge"),
    pytest.param("ha_model/charger.py", "START CHARGE LAUNCHED", id="charger_start_charge"),
    pytest.param("home_model/load.py", "ack command %s for load %s", id="load_ack_command"),
    pytest.param("home_model/load.py", "launch_command: %s for this load %s), ctxt: %s", id="load_launch_command"),
    pytest.param("home_model/constraints.py", "%s update callback asked for stop", id="constraints_asked_for_stop"),
    pytest.param(
        "home_model/solver.py",
        "_constraints_delta: trying to consume more: %sWh from %s to %s for loads %s",
        id="solver_constraints_delta",
    ),
]


@pytest.mark.parametrize(("module_path", "anchor"), _B4_SOURCE_ANCHORS)
def test_b4_keeper_lines_are_still_logged_at_info(module_path: str, anchor: str) -> None:
    r"""B4: each keeper appears exactly once, inside an INFO call.

    Deliberately NOT a single clever regex. The previous form
    (`_LOGGER\.info\(\s*[^)]*` + anchor) had `[^)]*` stop at the first `)`, so a
    keeper wrapped in an inner call would silently stop matching, and a cosmetic ruff
    reformat could fail the test while behavior was intact. Instead: assert the anchor
    is unique in the module, then check the nearest preceding logger call is `.info`.
    The sibling runtime `caplog` assertions below carry the behavioral weight.
    """
    source = (Path(quiet_solar.__file__).parent / module_path).read_text(encoding="utf-8")

    occurrences = [m.start() for m in re.finditer(re.escape(anchor), source)]
    assert len(occurrences) == 1, f"expected exactly one occurrence of {anchor!r}"

    preceding_calls = list(re.finditer(r"_LOGGER\.(\w+)\(", source[: occurrences[0]]))
    assert preceding_calls, f"no logger call precedes {anchor!r}"
    assert preceding_calls[-1].group(1) == "info", (
        f"{anchor!r} is emitted by _LOGGER.{preceding_calls[-1].group(1)}, expected info"
    )


async def test_b4_charger_start_stop_charge_still_log_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """B4 runtime: the real action lines survive in the file this change set edits."""
    caplog.set_level(logging.DEBUG, logger=CHARGER_LOGGER)
    *_, charger, _ = _make_stable_status_charger()
    charger.low_level_stop_charge = AsyncMock()
    charger.low_level_start_charge = AsyncMock()
    charger.is_charge_enabled = MagicMock(return_value=True)
    charger.is_charge_disabled = MagicMock(return_value=True)

    await charger.stop_charge(T0)
    await charger.start_charge(T0)

    assert len(_messages(caplog, "STOP CHARGE LAUNCHED", logging.INFO, CHARGER_LOGGER)) == 1
    assert len(_messages(caplog, "START CHARGE LAUNCHED", logging.INFO, CHARGER_LOGGER)) == 1


async def test_b4_load_command_lines_still_log_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """B4 runtime: `launch_command` and `_ack_command` stay at INFO in `load.py`."""
    caplog.set_level(logging.DEBUG, logger=LOAD_LOGGER)
    load = MinimalTestLoad(name="B4Load")
    caplog.clear()

    load._ack_command(T0, copy_command(CMD_ON, power_consign=1000.0))
    await load.launch_command(T0, copy_command(CMD_ON, power_consign=2000.0), ctxt="B4")

    acks = [r for r in caplog.records if r.msg == "ack command %s for load %s" and r.levelno == logging.INFO]
    launches = [
        r
        for r in caplog.records
        if r.msg == "launch_command: %s for this load %s), ctxt: %s" and r.levelno == logging.INFO
    ]
    # Exact counts: `>=` would pass on a duplication regression, which is the
    # direction this assertion exists to guard. `launch_command` acks its own
    # command once the probe confirms it, so the explicit call plus that ack is 2.
    assert len(acks) == 2
    assert len(launches) == 1


# =============================================================================
# B5 — every touched site logs lazily, with the expected literal
# =============================================================================

# site id -> (fragment used to find the record, expected `record.msg` literal)
_B5_SITES = {
    "S1": ("dyn_handle: START", "dyn_handle: START"),
    "S2": (
        "amp change cooldown",
        "dyn_handle: skipping %s for budgeting, amp change cooldown (%ss since last change)",
    ),
    "S3": ("all chargers in cooldown", "dyn_handle: all chargers in cooldown, skipping budgeting"),
    "S4": ("can't dampen simple case", "dyn_handle: can't dampen simple case %s %s %s"),
    "S5": (
        "possible_amps:",
        "get_stable_dynamic_charge_status: %s for %s score:%s possible_amps:%s "
        "possible_num_phases:%s current_amps:%s command:%s",
    ),
    "S6": (
        "do_remove_all_person_constraints",
        "check_load_activity_and_constraints: plugged car %s do_remove_all_person_constraints",
    ),
    "S7": (
        "No bad constraint found",
        "clean_constraints_for_load_param: No bad constraint found for %s, no reset needed",
    ),
    "S8": ("Constraint Reset device", "Constraint Reset device %s"),
    "S8d": ("nothing to reset", "Constraint Reset device %s, nothing to reset"),
    "S9": (
        "ensure_correct_state dyn group",
        "ensure_correct_state dyn group: %s  correct_state: %s handled_static: %s",
    ),
    "S10": (
        "battery_asked_charge",
        "dyn_handle: full_available_home_power %sW, grid_available_home_power %sW battery_asked_charge %sW",
    ),
    "S11": ("with score", "get_best_car: %s with score %s for charger %s"),
    "S12": (
        "is_car_charged",
        "update_value_callback (is %%:%s):%s %s  %s/%s (%s/%s) is_car_charged %s cmd %s",
    ),
    "S13": ("no actionable chargers, do nothing", "dyn_handle: no actionable chargers, do nothing"),
}

# S1, S3 and S13 are argless literals; every other touched site interpolates.
_B5_ARGLESS_SITES = {"S1", "S3", "S13"}


async def _drive_every_touched_site() -> None:
    """Drive all 13 sites once, in the branch each one needs."""
    group, _, _ = _make_dyn_handle_group(num_chargers=2, battery=make_battery(0.0), available_power_w=1000.0)
    await group.dyn_handle(T0)  # S1, S4, S10

    cooldown_group, _, _ = _make_dyn_handle_group(num_chargers=1, cooldown_seconds=10)
    await cooldown_group.dyn_handle(T0)  # S2, S3

    idle_group, _, _ = _make_dyn_handle_group(num_chargers=1)
    idle_group.ensure_correct_state = AsyncMock(return_value=([], None))
    await idle_group.dyn_handle(T0)  # S13

    *_, stable_charger, _ = _make_stable_status_charger()
    stable_charger.get_stable_dynamic_charge_status(T0)  # S5

    await _drive_remove_all_person_constraints()  # S6

    ecs_group, _ = _make_ensure_correct_state_group(num_chargers=1)
    await ecs_group.ensure_correct_state(T0)  # S9

    _, _, best_charger, _, _ = _make_best_car_charger()
    best_charger.get_car_score = MagicMock(return_value=10.0)
    best_charger.get_best_car(T0)  # S11

    soc_charger, constraint = _make_soc_callback_charger()
    await soc_charger.constraint_update_value_callback_percent_soc(constraint, T0)  # S12

    load = MinimalTestLoad(name="B5Load")  # S8d (nothing to reset)
    load.clean_constraints_for_load_param_and_if_same_key_same_value_info(
        T0, load_param="CarA", load_info=None, for_full_reset=False
    )  # S7
    load._constraints = [create_constraint(load=load, time=T0)]
    load.constraint_reset_and_reset_commands_if_needed(keep_commands=True)  # S8


async def test_b5_every_touched_site_logs_lazily(caplog: pytest.LogCaptureFixture) -> None:
    """B5: no surviving f-string, no trailing period, and `%s` args where expected.

    No gate can catch this: `pyproject.toml` ignores `G004` with 384 existing
    occurrences, and `per-file-ignores` cannot scope below file granularity.
    """
    caplog.set_level(logging.DEBUG, logger=CHARGER_LOGGER)
    caplog.set_level(logging.DEBUG, logger=LOAD_LOGGER)

    await _drive_every_touched_site()

    for site, (fragment, expected_msg) in _B5_SITES.items():
        matching = [r for r in caplog.records if fragment in r.getMessage() and r.msg == expected_msg]
        assert matching, f"{site}: no record whose msg is exactly {expected_msg!r}"
        record = matching[0]
        # An f-string would have been interpolated, so an exact-literal match is the
        # detector even for the argless sites.
        assert record.msg == expected_msg
        assert not record.msg.rstrip().endswith("."), f"{site}: log messages take no trailing period"
        if site in _B5_ARGLESS_SITES:
            assert not record.args
        else:
            assert record.args, f"{site}: converted site must interpolate lazily"
            assert "%" in record.msg
