"""QS-307 — an override on an unresponsive load must still expire.

`QSBiStateDuration.check_load_activity_and_constraints` used to gate its
**whole** user-override block on `is_load_command_set(time)`, which requires
`running_command is None`. Gating *detection* that way is correct — while a
command is landing, observed state != wanted state is expected. But the same
gate also suppressed the branches of the override lifecycle that never read
entity state at all:

- override **expiry** — pure clock arithmetic on
  `external_user_initiated_state_time`;
- the **reset-ask follow-up** — a pure flag check;
- the post-override **cooldown expiry** — pure clock arithmetic on
  `asked_for_reset_user_initiated_state_time`.

Since QS-304 made the relaunch backoff saturate and retry forever, the gate can
stay shut indefinitely, so those branches never ran: an override on a load that
stopped obeying was pinned **forever** and the load never came back into
controlled consumption.

Both death modes are covered, because they leave different command state behind:

- **visible but disobeying** — the entity still reports `on`/`off`, it just
  ignores commands, so `running_command` stays in flight forever
  (`probe_if_command_set` keeps returning `False`);
- **unavailable** — the entity drops off the network, the invalid-probe give-up
  (`NUM_MAX_INVALID_PROBES_COMMANDS`) nulls `current_command` through
  `_ack_command(time, None)`, so `is_load_command_set` is `False` on its
  `current_command is None` arm instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from datetime import time as dt_time
from unittest.mock import MagicMock

import pytz
from homeassistant.const import CONF_NAME, STATE_UNAVAILABLE

from custom_components.quiet_solar.const import (
    CONF_POWER,
    CONF_SWITCH,
    CONSTRAINT_ORIGINATOR_KEY,
    CONSTRAINT_ORIGINATOR_USER_OVERRIDE,
    CONSTRAINT_TYPE_FILLER_AUTO,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    OVERRIDE_STATE_NO_OVERRIDE,
)
from custom_components.quiet_solar.ha_model.bistate_duration import (
    USER_OVERRIDE_STATE_BACK_DURATION_S,
    QSBiStateDuration,
)
from custom_components.quiet_solar.home_model.commands import CMD_IDLE, CMD_ON, copy_command
from custom_components.quiet_solar.home_model.constraints import TimeBasedSimplePowerLoadConstraint
from custom_components.quiet_solar.home_model.load import NUM_MAX_INVALID_PROBES_COMMANDS
from tests.conftest import FakeHass
from tests.factories import create_minimal_home_model
from tests.qs304_helpers import CYCLE_S, LADDER_WALL_S, drive

PUMP_ENTITY = "switch.pool_pump"

# 18:00 — the worked example in the story: the user forces the pool pump ON with
# an 8 h window that should end at 02:00.
T_OVERRIDE = datetime(2026, 7, 31, 18, 0, 0, tzinfo=pytz.UTC)
OVERRIDE_DURATION_H = 8.0

# Past the window whether the override was armed at `T_OVERRIDE` directly or
# classified one cycle later by the detection branch.
T_EXPIRED = T_OVERRIDE + timedelta(hours=OVERRIDE_DURATION_H, seconds=2 * CYCLE_S)


class _StuckPump(QSBiStateDuration):
    """Bistate pump whose transport can be made to stop obeying.

    `obeys = False` reproduces the QS-304 pool-house shape: the service call
    goes out and returns "dispatched, unconfirmed" (the real
    `BistateTransport.execute` contract) but the entity never follows, so the
    command can never be acked.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault(CONF_SWITCH, PUMP_ENTITY)
        super().__init__(**kwargs)
        self.transport_calls: list[tuple[datetime, str, str | None]] = []
        self.obeys = True

    async def execute_command_system(self, time, command, state):
        """Record the service call, and only move the entity when obeying."""
        self.transport_calls.append((time, command.command, state))
        if self.obeys:
            target = state if state is not None else self.expected_state_from_command(command)
            self.hass.states.set(self.bistate_entity, target, last_changed=time)
            return True
        return False

    def get_virtual_current_constraint_translation_key(self):
        """Provide the translation key the entity layer would supply."""
        return "test_constraint_key"

    def get_select_translation_key(self):
        """Provide the translation key the entity layer would supply."""
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


def _make_pump(override_duration_h: float = OVERRIDE_DURATION_H) -> _StuckPump:
    """Build a switch-backed bistate pump on FakeHass, with no config entry."""
    hass = FakeHass()
    home = create_minimal_home_model()
    home.is_off_grid = MagicMock(return_value=False)
    pump = _StuckPump(
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


def _override_constraints(pump: _StuckPump) -> list:
    """Return the USER_OVERRIDE-originated constraints currently live."""
    return [
        c
        for c in pump._constraints
        if c.load_info is not None
        and c.load_info.get(CONSTRAINT_ORIGINATOR_KEY, "") == CONSTRAINT_ORIGINATOR_USER_OVERRIDE
    ]


def _arm_override(pump: _StuckPump, state: str = "on", time: datetime = T_OVERRIDE) -> None:
    """Put the pump in the state the detection branch would have left behind."""
    pump.external_user_initiated_state = state
    pump.external_user_initiated_state_time = time


def _push_override_constraint(pump: _StuckPump, time: datetime = T_OVERRIDE) -> None:
    """Push the hold-off/power constraint an ON override carries."""
    ct = TimeBasedSimplePowerLoadConstraint(
        type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        degraded_type=CONSTRAINT_TYPE_FILLER_AUTO,
        time=time,
        load=pump,
        load_param=pump.external_user_initiated_state,
        load_info={CONSTRAINT_ORIGINATOR_KEY: CONSTRAINT_ORIGINATOR_USER_OVERRIDE},
        from_user=True,
        start_of_constraint=time,
        end_of_constraint=time + timedelta(hours=OVERRIDE_DURATION_H),
        power=pump.power_use,
        initial_value=0,
        target_value=3600.0 * OVERRIDE_DURATION_H,
    )
    pump.push_live_constraint(time, ct)


# =============================================================================
# AC1 — expiry fires while a command is in flight
# =============================================================================


async def test_ac1_expiry_fires_while_a_command_is_in_flight():
    """A clock comparison has no business being gated on command state.

    `running_command` set + `current_command` set is the visible-but-disobeying
    death mode: `is_load_command_set` is `False` on its `running_command is
    None` arm, and on `main` that froze the expiry with it.
    """
    pump = _make_pump()
    _arm_override(pump)
    _push_override_constraint(pump)
    pump.current_command = copy_command(CMD_IDLE)
    pump.running_command = copy_command(CMD_ON)
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)

    # the precondition: the old gate is shut
    assert pump.is_load_command_set(T_EXPIRED) is False
    assert _override_constraints(pump) != []

    await pump.check_load_activity_and_constraints(T_EXPIRED)

    assert pump.external_user_initiated_state is None
    assert pump.external_user_initiated_state_time is None
    assert pump.asked_for_reset_user_initiated_state_time == T_EXPIRED
    assert _override_constraints(pump) == []
    # `keep_commands=True`: the reset must not touch the command slot
    assert pump.current_command is not None
    assert pump.running_command is not None


# =============================================================================
# AC2 — expiry fires after the invalid-probe give-up
# =============================================================================


async def _drive_to_invalid_probe_give_up(pump: _StuckPump, start: datetime) -> datetime:
    """Spin the cycle until the invalid-probe give-up empties the slot."""
    time = start
    for _ in range(NUM_MAX_INVALID_PROBES_COMMANDS + 2):
        await pump.check_and_relaunch_command(time)
        time = time + timedelta(seconds=CYCLE_S)
    return time


async def test_ac2_expiry_fires_after_the_invalid_probe_give_up():
    """The *unavailable* death mode empties the slot, so the ladder never climbs.

    `_ack_command(time, None)` nulls `current_command`, so
    `is_load_command_set` is `False` on its other arm — no relaunch escalation
    can ever reopen the gate. This is the case a "detection is allowed once the
    load is uncontrollable" predicate could not have fixed.
    """
    pump = _make_pump()
    _arm_override(pump)
    _push_override_constraint(pump)

    t_dead = T_OVERRIDE + timedelta(hours=1)
    pump.obeys = False
    pump.current_command = copy_command(CMD_IDLE)
    pump.running_command = copy_command(CMD_ON)
    pump.running_command_first_launch = t_dead
    pump.running_command_last_launch = t_dead
    pump.hass.states.set(PUMP_ENTITY, STATE_UNAVAILABLE, last_changed=t_dead)

    await _drive_to_invalid_probe_give_up(pump, t_dead)

    assert pump.current_command is None
    assert pump.running_command is None
    assert pump.is_load_command_set(T_EXPIRED) is False

    await pump.check_load_activity_and_constraints(T_EXPIRED)

    assert pump.external_user_initiated_state is None
    assert pump.external_user_initiated_state_time is None
    assert pump.asked_for_reset_user_initiated_state_time == T_EXPIRED
    assert _override_constraints(pump) == []


# =============================================================================
# AC3 — healthy-load parity
# =============================================================================


async def test_ac3_a_live_override_is_never_reset_early():
    """Nothing resets an override that has not aged out, in any command state.

    The safety net for the one admitted behaviour change: expiry can now land
    one cycle earlier (during the command flight rather than just after it).
    That is only sound if the branches themselves are inert, so a not-yet-aged
    override must survive every combination of command state.
    """
    for running in (None, CMD_ON):
        for current in (None, CMD_ON):
            pump = _make_pump()
            _arm_override(pump)
            _push_override_constraint(pump)
            pump.running_command = None if running is None else copy_command(running)
            pump.current_command = None if current is None else copy_command(current)
            pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)

            # half-way through the 8 h window: nowhere near expiry
            await pump.check_load_activity_and_constraints(T_OVERRIDE + timedelta(hours=4))

            context = (running, current)
            assert pump.external_user_initiated_state == "on", context
            assert pump.external_user_initiated_state_time == T_OVERRIDE, context
            assert pump.asked_for_reset_user_initiated_state_time is None, context
            assert pump.is_user_overridden() is True, context
            assert len(_override_constraints(pump)) == 1, context


# =============================================================================
# AC4 — disabled loads unchanged
# =============================================================================


async def test_ac4_a_disabled_load_runs_no_lifecycle_branch():
    """`is_load_command_set` is `False` for a disabled load, and stays decisive.

    The hoist replaces the gate with an explicit `qs_enable_device` guard rather
    than dropping it: QS was told to leave this load alone, so it must not
    silently start expiring overrides on it.
    """
    pump = _make_pump()
    _arm_override(pump)
    pump.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = T_OVERRIDE
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)
    pump._enabled = False

    await pump.check_load_activity_and_constraints(T_EXPIRED)

    assert pump.external_user_initiated_state == "on"
    assert pump.external_user_initiated_state_time == T_OVERRIDE
    assert pump.asked_for_reset_user_initiated_state_time is None
    assert pump.asked_for_reset_user_initiated_state_time_first_cmd_reset_done == T_OVERRIDE


# =============================================================================
# AC5 — the reset-ask follow-up runs unconditionally
# =============================================================================


async def test_ac5_reset_ask_follow_up_runs_with_a_command_in_flight():
    """The follow-up cleanup cycle is a flag check, so it must not be gated."""
    pump = _make_pump()
    _arm_override(pump)
    _push_override_constraint(pump)
    pump.external_user_initiated_state = None
    pump.external_user_initiated_state_time = None
    pump.asked_for_reset_user_initiated_state_time = T_OVERRIDE
    pump.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = T_OVERRIDE
    pump.current_command = copy_command(CMD_IDLE)
    pump.running_command = copy_command(CMD_ON)
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)

    t = T_OVERRIDE + timedelta(seconds=CYCLE_S)
    assert pump.is_load_command_set(t) is False
    assert _override_constraints(pump) != []

    do_force_next_solve = await pump.check_load_activity_and_constraints(t)

    assert pump.asked_for_reset_user_initiated_state_time_first_cmd_reset_done is None
    assert do_force_next_solve is True
    # `constraint_reset_and_reset_commands_if_needed(keep_commands=True)` ran:
    # the stale override constraint is gone, the commands are untouched.
    assert _override_constraints(pump) == []
    assert pump.current_command is not None
    assert pump.running_command is not None


# =============================================================================
# AC6 — detection is unchanged
# =============================================================================


async def test_ac6_detection_stays_shut_while_a_command_is_in_flight():
    """The regression pin for the non-goal: no new detection, ever.

    A state matching neither `expected_state` nor `expected_state_running` is
    exactly what "the user acted" looks like — and it is also what a landing
    command looks like. `is_load_command_set` still guards that decision.
    """
    pump = _make_pump()
    pump.current_command = copy_command(CMD_IDLE)
    pump.running_command = copy_command(CMD_IDLE)
    pump.last_command_execution_time = T_OVERRIDE
    # "on" matches neither expectation (both commands expect "off"), and the
    # state is newer than the last execution, so the causality guard allows it
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE + timedelta(seconds=30))

    await pump.check_load_activity_and_constraints(T_OVERRIDE + timedelta(minutes=2))

    assert pump.external_user_initiated_state is None
    assert pump.external_user_initiated_state_time is None
    assert _override_constraints(pump) == []


async def test_ac6_detection_still_works_once_the_command_has_landed():
    """...and the gate opening on an acked command still classifies the user."""
    pump = _make_pump()
    pump.current_command = copy_command(CMD_IDLE)
    pump.last_command_execution_time = T_OVERRIDE
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE + timedelta(seconds=30))

    t = T_OVERRIDE + timedelta(minutes=2)
    assert pump.is_load_command_set(t) is True

    await pump.check_load_activity_and_constraints(t)

    assert pump.external_user_initiated_state == "on"
    assert pump.external_user_initiated_state_time == t


# =============================================================================
# AC8 — the end-to-end bug, in both death modes
# =============================================================================


async def _detect_user_override_on(pump: _StuckPump) -> None:
    """Replay the story's 18:00: the user flips the pump ON by hand."""
    pump.current_command = copy_command(CMD_IDLE)
    pump.last_command_execution_time = T_OVERRIDE - timedelta(hours=1)
    pump.hass.states.set(PUMP_ENTITY, "off", last_changed=T_OVERRIDE - timedelta(hours=1))
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)

    await pump.check_load_activity_and_constraints(T_OVERRIDE + timedelta(seconds=CYCLE_S))

    assert pump.external_user_initiated_state == "on"
    assert pump.is_user_overridden() is True
    assert len(_override_constraints(pump)) == 1


async def _assert_override_self_heals(pump: _StuckPump) -> None:
    """The override expires on time, then the cooldown releases the load."""
    await pump.check_load_activity_and_constraints(T_EXPIRED)

    assert pump.external_user_initiated_state is None
    assert _override_constraints(pump) == []

    # the post-override cooldown is clock arithmetic too, so it also drains on
    # a load QS can no longer talk to — otherwise the load stayed pinned in
    # ASKED FOR RESET forever, which is the same bug one step later
    t_back = T_EXPIRED + timedelta(seconds=USER_OVERRIDE_STATE_BACK_DURATION_S + CYCLE_S)
    await pump.check_load_activity_and_constraints(t_back)

    assert pump.asked_for_reset_user_initiated_state_time is None
    assert pump.get_override_state() == OVERRIDE_STATE_NO_OVERRIDE
    assert pump.is_user_overridden() is False
    # ...and the load is back in the solver with its ordinary daily constraint
    assert pump._constraints != []


async def test_ac8_override_expires_on_a_visible_but_disobeying_load():
    """The #304 pool-house shape: reachable, answering, and ignoring us."""
    pump = _make_pump()
    await _detect_user_override_on(pump)

    # 19:00 — the relay dies: the pump falls back to "off" and stops obeying
    t_dead = T_OVERRIDE + timedelta(hours=1)
    pump.obeys = False
    pump.hass.states.set(PUMP_ENTITY, "off", last_changed=t_dead)
    await pump.launch_command(t_dead, CMD_ON, ctxt="override hold dispatch")
    assert pump.running_command is not None

    # QS-304: the ladder saturates and retries forever, so the gate stays shut
    await drive(pump, t_dead + timedelta(seconds=CYCLE_S), LADDER_WALL_S)
    assert pump.is_uncontrollable is True
    assert pump.is_load_command_set(t_dead) is False

    await _assert_override_self_heals(pump)


async def test_ac8_override_expires_on_an_unavailable_load():
    """The other death mode: the entity drops off the network entirely."""
    pump = _make_pump()
    await _detect_user_override_on(pump)

    t_dead = T_OVERRIDE + timedelta(hours=1)
    pump.obeys = False
    pump.hass.states.set(PUMP_ENTITY, STATE_UNAVAILABLE, last_changed=t_dead)
    await pump.launch_command(t_dead, CMD_ON, ctxt="override hold dispatch")
    assert pump.running_command is not None

    await _drive_to_invalid_probe_give_up(pump, t_dead + timedelta(seconds=CYCLE_S))
    assert pump.current_command is None
    assert pump.running_command is None
    assert pump.is_load_command_set(t_dead) is False

    await _assert_override_self_heals(pump)
