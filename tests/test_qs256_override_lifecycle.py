"""QS-256 — bistate user-override lifecycle regression tests.

Replays the June 4 → June 5 pool-pump incident (false "user override →
off" self-perpetuating for ~18h) and verifies every fix of the story:

- AC2: storage restore drops an already-expired override (and keeps a
  still-valid one, and anchors the causality guard on a restored
  command).
- AC3: no phantom ack — a command suppressed by an active override is
  DROPPED in ``launch_command`` (no service call, no counters mutation).
- AC5: post-override re-execution — once the hold-off constraint
  completes, the daily CMD_ON physically executes.
- AC6: causality guard — a state mismatch only classifies as a user
  override when the entity state is NEWER than the load's last real
  command execution.
- AC12: boundary-slot command stamped before the constraint start is
  dropped at dispatch time and re-issued after the override ends.
- AC13: flagship full-timeline replay.
- AC14: timer fallback for overrides without a constraint, and
  both-mechanisms-armed convergence.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from datetime import time as dt_time
from unittest.mock import MagicMock

import pytz
from freezegun import freeze_time
from homeassistant.const import CONF_NAME

from custom_components.quiet_solar.const import (
    CONF_POWER,
    CONF_SWITCH,
    CONSTRAINT_ORIGINATOR_KEY,
    CONSTRAINT_ORIGINATOR_USER_OVERRIDE,
)
from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
from custom_components.quiet_solar.home_model.commands import CMD_IDLE, CMD_ON, LoadCommand
from custom_components.quiet_solar.home_model.constraints import (
    TimeBasedHoldOffConstraint,
    TimeBasedSimplePowerLoadConstraint,
)
from tests.conftest import FakeHass
from tests.factories import create_minimal_home_model

# Normative incident anchors (log local times replayed as UTC).
RESTART_TIME = datetime(2026, 6, 4, 15, 57, 38, tzinfo=pytz.UTC)
PUMP_ENTITY = "switch.pool_pump"

_LOGGER = logging.getLogger(__name__)


class _ReplayPump(QSBiStateDuration):
    """Concrete bistate device with a recording transport layer."""

    def __init__(self, **kwargs):
        kwargs.setdefault(CONF_SWITCH, PUMP_ENTITY)
        super().__init__(**kwargs)
        # every (time, command, override_state) that reached the transport
        self.transport_calls: list[tuple[datetime, str, str | None]] = []
        # when True the fake entity follows the transport instantly
        self.auto_apply_state = True

    async def execute_command_system(self, time, command, state):
        self.transport_calls.append((time, command.command, state))
        if self.auto_apply_state:
            target = state if state is not None else self.expected_state_from_command(command)
            self.hass.states.set(self.bistate_entity, target, last_changed=time)
        return True

    def get_virtual_current_constraint_translation_key(self):
        return "test_constraint_key"

    def get_select_translation_key(self):
        return "test_select_key"


def _next_daily_end(local_hours: dt_time, time_utc_now: datetime | None = None, output_in_utc=True):
    """Deterministic, timezone-independent stand-in for get_next_time_from_hours."""
    assert output_in_utc is True
    assert time_utc_now is not None
    candidate = time_utc_now.replace(
        hour=local_hours.hour, minute=local_hours.minute, second=local_hours.second, microsecond=0
    )
    if candidate < time_utc_now:
        candidate += timedelta(days=1)
    return candidate


def _make_pump(override_duration_h: float = 8.0) -> _ReplayPump:
    hass = FakeHass()
    home = create_minimal_home_model()
    home.is_off_grid = MagicMock(return_value=False)
    pump = _ReplayPump(
        hass=hass,
        config_entry=None,
        home=home,
        **{CONF_NAME: "Piscine", CONF_POWER: 1259.0},
    )
    pump.override_duration = override_duration_h
    pump.bistate_mode = "bistate_mode_default"
    pump.default_on_duration = 1.0
    pump.default_on_finish_time = dt_time(hour=6, minute=0, second=0)
    pump.get_next_time_from_hours = _next_daily_end  # type: ignore[method-assign]
    pump.externally_initialized_constraints = True
    return pump


def _override_constraints(pump: _ReplayPump) -> list:
    return [
        c
        for c in pump._constraints
        if c.load_info is not None
        and c.load_info.get(CONSTRAINT_ORIGINATOR_KEY, "") == CONSTRAINT_ORIGINATOR_USER_OVERRIDE
    ]


def _daily_constraints(pump: _ReplayPump) -> list:
    return [
        c
        for c in pump._constraints
        if c.load_info is None or c.load_info.get(CONSTRAINT_ORIGINATOR_KEY, "") != CONSTRAINT_ORIGINATOR_USER_OVERRIDE
    ]


# =========================================================================
# AC13 — flagship June 4 → 5 replay
# =========================================================================


async def test_ac13_flagship_june_4_5_replay():
    """Full normative-timeline replay: no false override pair, hold-off
    lifecycle, no phantom ack, no re-arm loop, exact physical transitions."""
    with freeze_time(RESTART_TIME) as frozen:
        pump = _make_pump(override_duration_h=8.0)

        # --- (a) restart: restored current_command + stale facade -------
        pump.use_saved_extra_device_info(
            {
                "num_on_off": 0,
                "current_command": {"command": "idle", "power_consign": 0.0},
                "last_state_change_time": None,
                "last_check_update": None,
            }
        )
        restore_anchor = pump.last_command_execution_time
        assert restore_anchor is not None
        assert restore_anchor == RESTART_TIME

        # stale template-switch mirror: still 'on', last_changed BEFORE restore
        pump.hass.states.set(PUMP_ENTITY, "on", last_changed=RESTART_TIME - timedelta(seconds=60))

        t_check_1 = RESTART_TIME + timedelta(minutes=5, seconds=9)  # 16:02:47
        frozen.move_to(t_check_1)
        await pump.check_load_activity_and_constraints(t_check_1)

        # the stale state must NOT classify as a user override (AC6 / AC13a)
        assert pump.external_user_initiated_state is None
        assert _override_constraints(pump) == []

        # facade catches up at 16:03:50 → matches expected idle, still quiet
        t_facade_off = RESTART_TIME + timedelta(minutes=6, seconds=12)
        frozen.move_to(t_facade_off)
        pump.hass.states.set(PUMP_ENTITY, "off", last_changed=t_facade_off)
        await pump.check_load_activity_and_constraints(t_facade_off)
        assert pump.external_user_initiated_state is None
        assert _override_constraints(pump) == []

        # --- daytime: system physically turns the pump ON ---------------
        t_on = RESTART_TIME.replace(hour=18, minute=0, second=0)
        frozen.move_to(t_on)
        await pump.launch_command(t_on, CMD_ON, ctxt="solver dispatch")
        assert len(pump.transport_calls) == 1
        assert pump.num_on_off == 1
        assert pump.last_command_execution_time == t_on
        assert pump.hass.states.get(PUMP_ENTITY).state == "on"

        # --- (b) genuine OFF override: user flips the pump off ----------
        t_user_off = RESTART_TIME.replace(hour=20, minute=0, second=0)
        frozen.move_to(t_user_off)
        pump.hass.states.set(PUMP_ENTITY, "off", last_changed=t_user_off)

        t_detect = t_user_off + timedelta(seconds=60)
        frozen.move_to(t_detect)
        await pump.check_load_activity_and_constraints(t_detect)

        assert pump.external_user_initiated_state == "off"
        hold_offs = _override_constraints(pump)
        assert len(hold_offs) == 1
        hold_off = hold_offs[0]
        assert isinstance(hold_off, TimeBasedHoldOffConstraint)
        assert hold_off.end_of_constraint == t_detect + timedelta(hours=8)

        # the daily constraint is chained strictly after the override end
        dailies = _daily_constraints(pump)
        assert len(dailies) == 1
        assert dailies[0].start_of_constraint == t_detect + timedelta(hours=8, seconds=1)

        # solver pins the hold-off window to idle; idle is NOT suppressed
        t_idle = t_detect + timedelta(seconds=30)
        frozen.move_to(t_idle)
        await pump.launch_command(t_idle, CMD_IDLE, ctxt="solver dispatch hold-off")
        assert pump.current_command is not None and pump.current_command.is_off_or_idle()
        assert pump.num_on_off == 2  # on → idle, the user's physical transition
        assert len(pump.transport_calls) == 1  # probe-acked, no service call

        # --- midnight: daily constraint pushed; solver 'on' must NOT
        # phantom-ack (AC3 / AC13b) ---------------------------------------
        t_midnight = datetime(2026, 6, 5, 0, 0, 3, tzinfo=pytz.UTC)
        frozen.move_to(t_midnight)
        await pump.check_load_activity_and_constraints(t_midnight)
        assert pump.external_user_initiated_state == "off"

        transports_before = len(pump.transport_calls)
        num_on_off_before = pump.num_on_off
        current_before = pump.current_command
        last_change_before = pump.last_state_change_time

        await pump.launch_command(t_midnight, CMD_ON, ctxt="midnight solve dispatch")

        assert len(pump.transport_calls) == transports_before  # no service call
        assert pump.num_on_off == num_on_off_before
        assert pump.current_command == current_before
        assert pump.last_state_change_time == last_change_before
        assert pump.running_command is None

        # --- (d) UI metrics: daily progress never accrues while off -----
        t_mid_override = t_midnight + timedelta(minutes=5)
        frozen.move_to(t_mid_override)
        await pump.update_live_constraints(t_mid_override, timedelta(hours=24))
        for daily in _daily_constraints(pump):
            assert daily.current_value == 0.0

        # --- override end: constraint-driven reset (AC10 path) ----------
        t_override_end = t_detect + timedelta(hours=8, seconds=2)
        frozen.move_to(t_override_end)
        await pump.update_live_constraints(t_override_end, timedelta(hours=24))
        assert pump.external_user_initiated_state is None
        assert pump.asked_for_reset_user_initiated_state_time is not None
        assert _override_constraints(pump) == []

        # --- no re-arm loop after the cooldown ---------------------------
        t_after_cooldown = t_override_end + timedelta(seconds=200)
        frozen.move_to(t_after_cooldown)
        await pump.check_load_activity_and_constraints(t_after_cooldown)
        await pump.check_load_activity_and_constraints(t_after_cooldown + timedelta(seconds=5))
        assert pump.external_user_initiated_state is None
        assert _override_constraints(pump) == []

        # --- morning: the daily 'on' physically executes (AC5) ----------
        t_morning = datetime(2026, 6, 5, 5, 0, 0, tzinfo=pytz.UTC)
        frozen.move_to(t_morning)
        await pump.launch_command(t_morning, CMD_ON, ctxt="morning solve dispatch")

        assert len(pump.transport_calls) == 2
        assert pump.transport_calls[-1][1] == "on"
        assert pump.hass.states.get(PUMP_ENTITY).state == "on"
        assert pump.current_command == CMD_ON

        # --- (c) NET physical transitions: on, user-off, on → exactly 3 -
        assert pump.num_on_off == 3


# =========================================================================
# AC3 — no phantom ack (focused unit)
# =========================================================================


async def test_ac3_suppressed_command_is_dropped_without_any_side_effect(caplog):
    caplog.set_level(logging.INFO, logger="custom_components.quiet_solar.home_model.load")
    with freeze_time(RESTART_TIME):
        pump = _make_pump()
        pump.external_user_initiated_state = "off"
        pump.external_user_initiated_state_time = RESTART_TIME
        pump.current_command = LoadCommand(command="idle", power_consign=0.0)
        pump.hass.states.set(PUMP_ENTITY, "off", last_changed=RESTART_TIME)
        baseline_last_change = pump.last_state_change_time

        await pump.launch_command(RESTART_TIME + timedelta(minutes=1), CMD_ON, ctxt="test")

        assert pump.transport_calls == []
        assert pump.current_command == LoadCommand(command="idle", power_consign=0.0)
        assert pump.num_on_off == 0
        assert pump.last_state_change_time == baseline_last_change
        assert pump.running_command is None
        assert any("suppressed by user override" in rec.getMessage() for rec in caplog.records)


# =========================================================================
# AC6 — causality guard, all three branches
# =========================================================================


async def test_ac6_stale_state_older_than_last_execution_is_not_an_override():
    with freeze_time(RESTART_TIME):
        pump = _make_pump()
        pump.current_command = LoadCommand(command="idle", power_consign=0.0)
        pump.last_command_execution_time = RESTART_TIME
        pump.hass.states.set(PUMP_ENTITY, "on", last_changed=RESTART_TIME - timedelta(seconds=1))

        t = RESTART_TIME + timedelta(minutes=2)
        await pump.check_load_activity_and_constraints(t)

        assert pump.external_user_initiated_state is None


async def test_ac6_fresh_state_newer_than_last_execution_is_an_override():
    with freeze_time(RESTART_TIME):
        pump = _make_pump()
        pump.current_command = LoadCommand(command="idle", power_consign=0.0)
        pump.last_command_execution_time = RESTART_TIME
        pump.hass.states.set(PUMP_ENTITY, "on", last_changed=RESTART_TIME + timedelta(seconds=30))

        t = RESTART_TIME + timedelta(minutes=2)
        await pump.check_load_activity_and_constraints(t)

        assert pump.external_user_initiated_state == "on"


async def test_ac6_no_anchor_classification_proceeds_on_state_mismatch():
    with freeze_time(RESTART_TIME):
        pump = _make_pump()
        pump.current_command = LoadCommand(command="idle", power_consign=0.0)
        assert pump.last_command_execution_time is None
        pump.hass.states.set(PUMP_ENTITY, "on", last_changed=RESTART_TIME - timedelta(days=30))

        t = RESTART_TIME + timedelta(minutes=2)
        await pump.check_load_activity_and_constraints(t)

        assert pump.external_user_initiated_state == "on"


# =========================================================================
# AC2 — storage restore staleness guard
# =========================================================================


async def test_ac2_expired_stored_override_is_dropped_at_restore(caplog):
    caplog.set_level(logging.INFO, logger="custom_components.quiet_solar.ha_model.bistate_duration")
    with freeze_time(RESTART_TIME):
        pump = _make_pump(override_duration_h=8.0)
        stale_time = RESTART_TIME - timedelta(hours=9)
        pump.use_saved_extra_device_info(
            {
                "num_on_off": 0,
                "current_command": None,
                "last_state_change_time": None,
                "last_check_update": None,
                "external_user_initiated_state": "off",
                "external_user_initiated_state_time": stale_time.isoformat(),
                "asked_for_reset_user_initiated_state_time": None,
                "asked_for_reset_user_initiated_state_time_first_cmd_reset_done": None,
            }
        )

        assert pump.external_user_initiated_state is None
        assert pump.external_user_initiated_state_time is None
        assert pump.asked_for_reset_user_initiated_state_time is None
        assert pump.asked_for_reset_user_initiated_state_time_first_cmd_reset_done is None
        assert any("expired" in rec.getMessage() for rec in caplog.records)


async def test_ac2_still_valid_stored_override_is_kept():
    with freeze_time(RESTART_TIME):
        pump = _make_pump(override_duration_h=8.0)
        valid_time = RESTART_TIME - timedelta(hours=1)
        pump.use_saved_extra_device_info(
            {
                "num_on_off": 0,
                "current_command": None,
                "last_state_change_time": None,
                "last_check_update": None,
                "external_user_initiated_state": "off",
                "external_user_initiated_state_time": valid_time.isoformat(),
            }
        )

        assert pump.external_user_initiated_state == "off"
        assert pump.external_user_initiated_state_time == valid_time
        # no restored command → no causality anchor
        assert pump.last_command_execution_time is None


async def test_ac2_restored_command_sets_causality_anchor():
    with freeze_time(RESTART_TIME):
        pump = _make_pump()
        pump.use_saved_extra_device_info(
            {
                "num_on_off": 0,
                "current_command": {"command": "on", "power_consign": 1259.0},
                "last_state_change_time": None,
                "last_check_update": None,
            }
        )
        assert pump.last_command_execution_time == RESTART_TIME


# =========================================================================
# AC5 — post-override re-execution (focused)
# =========================================================================


async def test_ac5_daily_on_physically_executes_after_hold_off_completes():
    with freeze_time(RESTART_TIME) as frozen:
        pump = _make_pump(override_duration_h=1.0)
        pump.current_command = LoadCommand(command="on", power_consign=1259.0)
        pump.last_command_execution_time = RESTART_TIME - timedelta(hours=1)
        pump.hass.states.set(PUMP_ENTITY, "off", last_changed=RESTART_TIME)

        t_detect = RESTART_TIME + timedelta(seconds=30)
        frozen.move_to(t_detect)
        await pump.check_load_activity_and_constraints(t_detect)
        assert pump.external_user_initiated_state == "off"
        assert len(_override_constraints(pump)) == 1

        # align the system command with the held state
        await pump.launch_command(t_detect, CMD_IDLE, ctxt="hold-off dispatch")
        num_on_off_before = pump.num_on_off

        # hold-off completes → constraint-driven override reset (AC10)
        t_end = t_detect + timedelta(hours=1, seconds=2)
        frozen.move_to(t_end)
        force_solve = await pump.update_live_constraints(t_end, timedelta(hours=24))
        assert force_solve is True
        assert pump.external_user_initiated_state is None
        assert _override_constraints(pump) == []

        # the re-solve dispatch physically executes the daily CMD_ON
        transports_before = len(pump.transport_calls)
        t_dispatch = t_end + timedelta(minutes=5)
        frozen.move_to(t_dispatch)
        await pump.launch_command(t_dispatch, CMD_ON, ctxt="re-solve dispatch")

        assert len(pump.transport_calls) == transports_before + 1
        assert pump.transport_calls[-1][1] == "on"
        assert pump.current_command == CMD_ON
        assert pump.num_on_off == num_on_off_before + 1


# =========================================================================
# AC12 — boundary-slot command dropped at dispatch, re-issued after end
# =========================================================================


async def test_ac12_boundary_slot_command_dropped_then_reissued():
    with freeze_time(RESTART_TIME) as frozen:
        pump = _make_pump(override_duration_h=1.0)
        pump.external_user_initiated_state = "off"
        pump.external_user_initiated_state_time = RESTART_TIME
        pump.current_command = LoadCommand(command="idle", power_consign=0.0)
        pump.hass.states.set(PUMP_ENTITY, "off", last_changed=RESTART_TIME)

        hold_off = TimeBasedHoldOffConstraint(
            time=RESTART_TIME,
            load=pump,
            load_param="off",
            load_info={CONSTRAINT_ORIGINATOR_KEY: CONSTRAINT_ORIGINATOR_USER_OVERRIDE},
            from_user=True,
            type=7,
            start_of_constraint=RESTART_TIME,
            end_of_constraint=RESTART_TIME + timedelta(hours=1),
            initial_value=0,
            target_value=3600.0,
        )
        pump.push_live_constraint(RESTART_TIME, hold_off)
        daily = TimeBasedSimplePowerLoadConstraint(
            time=RESTART_TIME,
            load=pump,
            type=7,
            start_of_constraint=RESTART_TIME + timedelta(hours=1),
            end_of_constraint=RESTART_TIME + timedelta(hours=4),
            initial_value=0,
            target_value=3600.0,
            power=1259.0,
        )
        pump.push_live_constraint(RESTART_TIME, daily)

        # boundary-slot command, stamped one solver slot BEFORE the daily start
        t_boundary = RESTART_TIME + timedelta(minutes=45)
        frozen.move_to(t_boundary)
        await pump.launch_command(t_boundary, CMD_ON, ctxt="boundary slot dispatch")

        # dropped at the drop site: no execution, no ack, no counters
        assert pump.transport_calls == []
        assert pump.running_command is None
        assert pump.num_on_off == 0
        assert pump.current_command == LoadCommand(command="idle", power_consign=0.0)

        # override ends → re-solve re-issues the command, which executes
        t_end = RESTART_TIME + timedelta(hours=1, seconds=2)
        frozen.move_to(t_end)
        await pump.update_live_constraints(t_end, timedelta(hours=24))
        assert pump.external_user_initiated_state is None

        await pump.launch_command(t_end + timedelta(seconds=1), CMD_ON, ctxt="re-solve dispatch")
        assert len(pump.transport_calls) == 1
        assert pump.transport_calls[-1][1] == "on"
        assert pump.num_on_off == 1


# =========================================================================
# AC14 — timer fallback and both-mechanisms-armed convergence
# =========================================================================


async def test_ac14_override_without_constraint_expires_via_legacy_timer():
    with freeze_time(RESTART_TIME) as frozen:
        pump = _make_pump(override_duration_h=8.0)
        # restored from storage: override valid at restore, no constraint
        pump.use_saved_extra_device_info(
            {
                "num_on_off": 0,
                "current_command": {"command": "idle", "power_consign": 0.0},
                "last_state_change_time": None,
                "last_check_update": None,
                "external_user_initiated_state": "off",
                "external_user_initiated_state_time": (RESTART_TIME - timedelta(hours=1)).isoformat(),
            }
        )
        assert pump.external_user_initiated_state == "off"
        assert _override_constraints(pump) == []
        pump.hass.states.set(PUMP_ENTITY, "off", last_changed=RESTART_TIME - timedelta(hours=1))

        # past the 8h window: the legacy timer path resets the override
        t_expired = RESTART_TIME + timedelta(hours=7, minutes=1)
        frozen.move_to(t_expired)
        force_solve = await pump.check_load_activity_and_constraints(t_expired)

        assert force_solve is True
        assert pump.external_user_initiated_state is None
        assert pump.external_user_initiated_state_time is None


async def test_ac14_both_mechanisms_armed_converge_whichever_fires_first():
    """Timer-first and constraint-first orders both converge to the same state."""
    for timer_first in (True, False):
        with freeze_time(RESTART_TIME) as frozen:
            pump = _make_pump(override_duration_h=1.0)
            pump.current_command = LoadCommand(command="on", power_consign=1259.0)
            pump.last_command_execution_time = RESTART_TIME - timedelta(hours=1)
            pump.hass.states.set(PUMP_ENTITY, "off", last_changed=RESTART_TIME)

            t_detect = RESTART_TIME + timedelta(seconds=30)
            frozen.move_to(t_detect)
            await pump.check_load_activity_and_constraints(t_detect)
            assert pump.external_user_initiated_state == "off"
            assert len(_override_constraints(pump)) == 1

            t_expired = t_detect + timedelta(hours=1, seconds=5)
            frozen.move_to(t_expired)
            if timer_first:
                await pump.check_load_activity_and_constraints(t_expired)
                await pump.update_live_constraints(t_expired, timedelta(hours=24))
            else:
                await pump.update_live_constraints(t_expired, timedelta(hours=24))
                await pump.check_load_activity_and_constraints(t_expired)

            assert pump.external_user_initiated_state is None, f"timer_first={timer_first}"
            assert pump.external_user_initiated_state_time is None, f"timer_first={timer_first}"
            assert _override_constraints(pump) == [], f"timer_first={timer_first}"


# =========================================================================
# Review fix #01 — naive-datetime storage tolerance (finding 1)
# =========================================================================


async def test_fix01_naive_expired_stored_override_dropped_without_crash(caplog):
    """A legacy tz-naive stored timestamp must not abort the restore."""
    caplog.set_level(logging.INFO, logger="custom_components.quiet_solar.ha_model.bistate_duration")
    with freeze_time(RESTART_TIME):
        pump = _make_pump(override_duration_h=8.0)
        naive_stale = (RESTART_TIME - timedelta(hours=9)).replace(tzinfo=None)
        pump.use_saved_extra_device_info(
            {
                "num_on_off": 0,
                "current_command": None,
                "last_state_change_time": None,
                "last_check_update": None,
                "external_user_initiated_state": "off",
                "external_user_initiated_state_time": naive_stale.isoformat(),
            }
        )

        assert pump.external_user_initiated_state is None
        assert pump.external_user_initiated_state_time is None
        assert any("expired" in rec.getMessage() for rec in caplog.records)


async def test_fix01_naive_valid_stored_override_kept_and_normalized():
    """A still-valid naive stored override is kept, coerced to tz-aware UTC."""
    with freeze_time(RESTART_TIME):
        pump = _make_pump(override_duration_h=8.0)
        naive_valid = (RESTART_TIME - timedelta(hours=1)).replace(tzinfo=None)
        pump.use_saved_extra_device_info(
            {
                "num_on_off": 0,
                "current_command": None,
                "last_state_change_time": None,
                "last_check_update": None,
                "external_user_initiated_state": "off",
                "external_user_initiated_state_time": naive_valid.isoformat(),
            }
        )

        assert pump.external_user_initiated_state == "off"
        assert pump.external_user_initiated_state_time is not None
        assert pump.external_user_initiated_state_time.tzinfo is not None
        assert pump.external_user_initiated_state_time == RESTART_TIME - timedelta(hours=1)


async def test_fix01_naive_override_time_at_check_site_does_not_crash():
    """The legacy timer check coerces a naive timestamp instead of raising."""
    with freeze_time(RESTART_TIME):
        pump = _make_pump(override_duration_h=8.0)
        pump.current_command = LoadCommand(command="idle", power_consign=0.0)
        pump.external_user_initiated_state = "off"
        # naive timestamp injected straight onto the field (legacy storage path)
        pump.external_user_initiated_state_time = (RESTART_TIME - timedelta(hours=9)).replace(tzinfo=None)
        pump.hass.states.set(PUMP_ENTITY, "off", last_changed=RESTART_TIME)

        force_solve = await pump.check_load_activity_and_constraints(RESTART_TIME)

        # expired override reset via the legacy timer path, no TypeError
        assert force_solve is True
        assert pump.external_user_initiated_state is None
        assert pump.external_user_initiated_state_time is None


# =========================================================================
# Review fix #02 — must-fix and should-fix findings
# =========================================================================


async def test_fix02_naive_asked_for_reset_time_restored_is_coerced():
    """F1: a tz-naive stored reset-ask timestamp is coerced at restore."""
    with freeze_time(RESTART_TIME):
        pump = _make_pump(override_duration_h=8.0)
        naive_ask = (RESTART_TIME - timedelta(seconds=30)).replace(tzinfo=None)
        pump.use_saved_extra_device_info(
            {
                "num_on_off": 0,
                "current_command": None,
                "last_state_change_time": None,
                "last_check_update": None,
                "asked_for_reset_user_initiated_state_time": naive_ask.isoformat(),
            }
        )
        assert pump.asked_for_reset_user_initiated_state_time is not None
        assert pump.asked_for_reset_user_initiated_state_time.tzinfo is not None
        assert pump.asked_for_reset_user_initiated_state_time == RESTART_TIME - timedelta(seconds=30)


async def test_fix02_naive_asked_for_reset_time_at_cooldown_site_no_crash():
    """F1: the cooldown check coerces a naive timestamp instead of raising."""
    with freeze_time(RESTART_TIME):
        pump = _make_pump(override_duration_h=8.0)
        pump.current_command = LoadCommand(command="idle", power_consign=0.0)
        # naive timestamp injected straight onto the field (legacy path)
        pump.asked_for_reset_user_initiated_state_time = (RESTART_TIME - timedelta(seconds=30)).replace(tzinfo=None)
        # mismatching fresh state would classify a new override without cooldown
        pump.hass.states.set(PUMP_ENTITY, "on", last_changed=RESTART_TIME)

        await pump.check_load_activity_and_constraints(RESTART_TIME)

        # inside the 180s cooldown: suppressed, no TypeError, field coerced
        assert pump.external_user_initiated_state is None
        assert pump.asked_for_reset_user_initiated_state_time is not None
        assert pump.asked_for_reset_user_initiated_state_time.tzinfo is not None


async def test_fix02_force_relaunch_drops_suppressed_command_state_matches_override():
    """F2(a): stale running command under an override → no phantom ack."""
    with freeze_time(RESTART_TIME):
        pump = _make_pump()
        pump.external_user_initiated_state = "off"
        pump.external_user_initiated_state_time = RESTART_TIME
        pump.current_command = LoadCommand(command="idle", power_consign=0.0)
        pump.running_command = LoadCommand(command="on", power_consign=1259.0)
        pump.hass.states.set(PUMP_ENTITY, "off", last_changed=RESTART_TIME)

        await pump.force_relaunch_command(RESTART_TIME + timedelta(minutes=1))

        assert pump.transport_calls == []  # no service call
        assert pump.running_command is None  # dropped, nothing to resurrect
        assert pump.current_command == LoadCommand(command="idle", power_consign=0.0)  # no ack
        assert pump.num_on_off == 0  # no counter increment
        assert pump.last_command_execution_time is None  # no re-anchor


async def test_fix02_force_relaunch_no_transport_call_when_state_unavailable_under_override():
    """F2(b): entity unavailable under an active override → no transport call."""
    with freeze_time(RESTART_TIME):
        pump = _make_pump()
        pump.external_user_initiated_state = "off"
        pump.external_user_initiated_state_time = RESTART_TIME
        pump.running_command = LoadCommand(command="on", power_consign=1259.0)
        pump.hass.states.set(PUMP_ENTITY, "unavailable", last_changed=RESTART_TIME)

        await pump.force_relaunch_command(RESTART_TIME + timedelta(minutes=1))

        assert pump.transport_calls == []
        assert pump.running_command is None
        assert pump.num_on_off == 0


async def test_fix02_naive_state_last_changed_in_causality_guard_no_crash():
    """F3: a tz-naive entity last_changed is coerced, not raised on."""
    with freeze_time(RESTART_TIME):
        pump = _make_pump()
        pump.current_command = LoadCommand(command="idle", power_consign=0.0)
        pump.last_command_execution_time = RESTART_TIME
        naive_stale = (RESTART_TIME - timedelta(seconds=1)).replace(tzinfo=None)
        pump.hass.states.set(PUMP_ENTITY, "on", last_changed=naive_stale)

        await pump.check_load_activity_and_constraints(RESTART_TIME + timedelta(minutes=2))

        # stale (once coerced) → suppressed, no TypeError
        assert pump.external_user_initiated_state is None


async def test_fix02_future_dated_stored_override_is_dropped(caplog):
    """F4: a stored override timestamp ahead of now is poison → dropped."""
    caplog.set_level(logging.INFO, logger="custom_components.quiet_solar.ha_model.bistate_duration")
    with freeze_time(RESTART_TIME):
        pump = _make_pump(override_duration_h=8.0)
        future_time = RESTART_TIME + timedelta(hours=1)
        pump.use_saved_extra_device_info(
            {
                "num_on_off": 0,
                "current_command": None,
                "last_state_change_time": None,
                "last_check_update": None,
                "external_user_initiated_state": "off",
                "external_user_initiated_state_time": future_time.isoformat(),
            }
        )

        assert pump.external_user_initiated_state is None
        assert pump.external_user_initiated_state_time is None
        assert pump.asked_for_reset_user_initiated_state_time is None
        assert pump.asked_for_reset_user_initiated_state_time_first_cmd_reset_done is None


async def test_fix02_small_clock_skew_in_stored_override_is_tolerated():
    """F4: a small (< 60s) future skew is NOT treated as poison."""
    with freeze_time(RESTART_TIME):
        pump = _make_pump(override_duration_h=8.0)
        slight_future = RESTART_TIME + timedelta(seconds=30)
        pump.use_saved_extra_device_info(
            {
                "num_on_off": 0,
                "current_command": None,
                "last_state_change_time": None,
                "last_check_update": None,
                "external_user_initiated_state": "off",
                "external_user_initiated_state_time": slight_future.isoformat(),
            }
        )

        assert pump.external_user_initiated_state == "off"
        assert pump.external_user_initiated_state_time == slight_future


async def test_fix02_last_changed_none_with_anchor_suppresses_classification():
    """F8: last_changed=None while an anchor exists → cannot prove freshness
    → conservative: no override classified."""
    with freeze_time(RESTART_TIME):
        pump = _make_pump()
        pump.current_command = LoadCommand(command="idle", power_consign=0.0)
        pump.last_command_execution_time = RESTART_TIME
        pump.hass.states.set(PUMP_ENTITY, "on", last_changed=RESTART_TIME)
        pump.hass.states.get(PUMP_ENTITY).last_changed = None

        await pump.check_load_activity_and_constraints(RESTART_TIME + timedelta(minutes=2))

        assert pump.external_user_initiated_state is None
