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

import pytest
import pytz
from freezegun import freeze_time
from homeassistant.const import CONF_NAME, STATE_UNAVAILABLE

from custom_components.quiet_solar.const import (
    CONF_POWER,
    CONF_SWITCH,
    CONSTRAINT_ORIGINATOR_KEY,
    CONSTRAINT_ORIGINATOR_USER_OVERRIDE,
    CONSTRAINT_TYPE_FILLER_AUTO,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    OVERRIDE_STATE_ASKED_FOR_RESET,
    OVERRIDE_STATE_NO_OVERRIDE,
    OVERRIDE_STATE_PREFIX,
    STORAGE_KEY_ASKED_FOR_RESET_TIME,
)
from custom_components.quiet_solar.ha_model.bistate_duration import (
    USER_OVERRIDE_STATE_BACK_DURATION_S,
    QSBiStateDuration,
)
from custom_components.quiet_solar.home_model.commands import CMD_IDLE, CMD_ON, LoadCommand, copy_command
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


class _RacingPump(_StuckPump):
    """A pump whose service call is interrupted by a user action mid-flight.

    `check_load_activity_and_constraints` gained a command-slot mutation (the
    expiry-time drop of an override-aligned command), and three of its four callers
    run OUTSIDE `_update_loads_lock` — `user_set_default_on_duration`,
    `user_set_bistate_mode` and `AbstractLoad.async_reset_override_state`. So a user
    touching the mode select, the on-duration number or the reset button while a
    service call is in flight really can empty the slot across the await. Setting
    `race_at` replays that deterministically, from inside the transport.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.race_at: datetime | None = None
        # what the interrupted service call reports back: `True` reaches the ack
        # branch, `None` reaches the "impossible to force" log — both dereference
        # the command slot after the await
        self.race_result: bool | None = True

    async def execute_command_system(self, time, command, state):
        """Let a user action land in the middle of the service call."""
        if self.race_at is None:
            return await super().execute_command_system(time, command, state)

        self.transport_calls.append((time, command.command, state))
        race_at, self.race_at = self.race_at, None
        await self.check_load_activity_and_constraints(race_at)
        return self.race_result


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


def _make_pump(
    override_duration_h: float = OVERRIDE_DURATION_H,
    pump_class: type[_StuckPump] = _StuckPump,
) -> _StuckPump:
    """Build a switch-backed bistate pump on FakeHass, with no config entry."""
    hass = FakeHass()
    home = create_minimal_home_model()
    home.is_off_grid = MagicMock(return_value=False)
    pump = pump_class(
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

    Review fix #02: the in-flight command here is the override's OWN service call
    (it drives the entity to the override state), so expiry must drop it. Left
    alone it would keep being relaunched for up to a full ladder *after* the
    override ended, because nulling the override state also disables
    `force_relaunch_command`'s suppression drop.
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
    assert pump.expected_state_from_command(pump.running_command) == pump.external_user_initiated_state

    await pump.check_load_activity_and_constraints(T_EXPIRED)

    assert pump.external_user_initiated_state is None
    assert pump.external_user_initiated_state_time is None
    assert pump.asked_for_reset_user_initiated_state_time == T_EXPIRED
    assert _override_constraints(pump) == []
    # the override's own command is gone...
    assert pump.running_command is None
    # ...but `keep_commands=True` still protects the last CONFIRMED command
    assert pump.current_command is not None


async def test_ac1_expiry_leaves_a_command_the_override_was_not_serving_alone():
    """Review fix #02 must not over-reach: only the override's own call is dropped.

    A command whose expected state differs from the override state was never the
    override's service call — it is a genuine solver intent that got through (the
    degraded-override nuance lets off/idle commands past). Expiry calls
    `constraint_reset_and_reset_commands_if_needed(keep_commands=True)`, so it must
    survive.
    """
    pump = _make_pump()
    _arm_override(pump)
    _push_override_constraint(pump)
    pump.current_command = copy_command(CMD_ON)
    pump.running_command = copy_command(CMD_IDLE)
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)

    assert pump.expected_state_from_command(pump.running_command) != pump.external_user_initiated_state

    await pump.check_load_activity_and_constraints(T_EXPIRED)

    assert pump.external_user_initiated_state is None
    assert pump.running_command == CMD_IDLE
    assert pump.current_command == CMD_ON


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


def _armed_pump(running: LoadCommand | None, current: LoadCommand | None) -> _StuckPump:
    """A pump carrying a live ON override plus the given command-slot state."""
    pump = _make_pump()
    _arm_override(pump)
    _push_override_constraint(pump)
    pump.running_command = None if running is None else copy_command(running)
    pump.current_command = None if current is None else copy_command(current)
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)
    return pump


@pytest.mark.parametrize("current", [None, CMD_ON], ids=["no-current", "current-on"])
@pytest.mark.parametrize("running", [None, CMD_ON], ids=["no-running", "running-on"])
async def test_ac3_a_live_override_is_never_reset_early(running, current):
    """Nothing resets an override that has not aged out, in any command state.

    The negative half of the safety net for the one admitted behaviour change:
    expiry can now land one cycle earlier (during the command flight rather than
    just after it). That is only sound if the branches themselves are inert, so a
    not-yet-aged override must survive every combination of command state.
    """
    pump = _armed_pump(running, current)

    # half-way through the 8 h window: nowhere near expiry
    await pump.check_load_activity_and_constraints(T_OVERRIDE + timedelta(hours=4))

    assert pump.external_user_initiated_state == "on"
    assert pump.external_user_initiated_state_time == T_OVERRIDE
    assert pump.asked_for_reset_user_initiated_state_time is None
    assert pump.is_user_overridden() is True
    assert len(_override_constraints(pump)) == 1


@pytest.mark.parametrize("current", [None, CMD_ON], ids=["no-current", "current-on"])
@pytest.mark.parametrize("running", [None, CMD_ON], ids=["no-running", "running-on"])
async def test_ac3_an_aged_override_resets_on_the_same_cycle_as_on_main(running, current):
    """...and the POSITIVE half, which review fix #07 asked to stop delegating.

    The window boundary is `> override_duration`, so the last cycle at exactly
    8 h must NOT reset and the next one must — identically in all four
    command-state combinations, which is what "same cycle as on `main`" means.
    """
    pump = _armed_pump(running, current)

    # exactly at the window edge: still alive (`>` not `>=`)
    await pump.check_load_activity_and_constraints(T_OVERRIDE + timedelta(hours=OVERRIDE_DURATION_H))
    assert pump.external_user_initiated_state == "on"
    assert pump.asked_for_reset_user_initiated_state_time is None

    # the very next cycle resets, whatever the command slot holds
    await pump.check_load_activity_and_constraints(T_EXPIRED)
    assert pump.external_user_initiated_state is None
    assert pump.external_user_initiated_state_time is None
    assert pump.asked_for_reset_user_initiated_state_time == T_EXPIRED
    assert _override_constraints(pump) == []


# =============================================================================
# AC4 — disabled loads unchanged
# =============================================================================


async def test_ac4_a_disabled_load_runs_no_lifecycle_branch():
    """`is_load_command_set` is `False` for a disabled load, and stays decisive.

    The hoist replaces the gate with an explicit `qs_enable_device` guard rather
    than dropping it: QS was told to leave this load alone, so it must not
    silently start expiring overrides on it.

    All **three** lifecycle branches are armed here (review fix #02/05): an aged
    override, a pending reset-ask follow-up flag, and a drained-past cooldown timer.
    AC4 was reworded from "neither" to "no hoisted branch" when the third one was
    added, but its test was not extended with it.
    """
    pump = _make_pump()
    _arm_override(pump)
    pump.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = T_OVERRIDE
    pump.asked_for_reset_user_initiated_state_time = T_OVERRIDE
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)
    pump._enabled = False

    await pump.check_load_activity_and_constraints(T_EXPIRED)

    assert pump.external_user_initiated_state == "on"
    assert pump.external_user_initiated_state_time == T_OVERRIDE
    assert pump.asked_for_reset_user_initiated_state_time_first_cmd_reset_done == T_OVERRIDE
    # the cooldown is long past its window, and still must not be drained
    assert pump.asked_for_reset_user_initiated_state_time == T_OVERRIDE


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
# AC5b — the post-override cooldown, the third hoisted branch
#
# Review fix #04: the story's D1 designed two hoisted branches and this one was
# added during implementation because AC8 is unreachable without it —
# `get_override_state()` returns ASKED FOR RESET while the timestamp is set, so
# the load would expire its override and still never come back. It was covered
# only transitively, through AC8's 8-hour end-to-end replay; these two pin it
# directly.
# =============================================================================


async def test_the_post_override_cooldown_drains_with_a_command_in_flight():
    """A cooldown timer is clock arithmetic, so a stuck command must not freeze it.

    This drain used to live *inside* the detection branch, which meant a load that
    stopped answering stayed `ASKED FOR RESET` forever: `is_user_overridden()`
    stayed `True`, and the load stayed out of controlled consumption.
    """
    pump = _make_pump()
    pump.asked_for_reset_user_initiated_state_time = T_OVERRIDE
    pump.current_command = copy_command(CMD_IDLE)
    pump.running_command = copy_command(CMD_ON)
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)

    t = T_OVERRIDE + timedelta(seconds=USER_OVERRIDE_STATE_BACK_DURATION_S + CYCLE_S)
    assert pump.is_load_command_set(t) is False
    assert pump.get_override_state() == OVERRIDE_STATE_ASKED_FOR_RESET

    await pump.check_load_activity_and_constraints(t)

    assert pump.asked_for_reset_user_initiated_state_time is None
    assert pump.get_override_state() == OVERRIDE_STATE_NO_OVERRIDE
    assert pump.is_user_overridden() is False


async def test_the_post_override_cooldown_is_not_drained_early():
    """...and the window is still honoured: an open cooldown is left alone.

    Hoisting must not shorten the 180 s "too soon to re-override" window, which is
    what stops a just-ended override immediately re-arming itself.
    """
    pump = _make_pump()
    pump.asked_for_reset_user_initiated_state_time = T_OVERRIDE
    pump.current_command = copy_command(CMD_IDLE)
    pump.running_command = copy_command(CMD_ON)
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)

    await pump.check_load_activity_and_constraints(
        T_OVERRIDE + timedelta(seconds=USER_OVERRIDE_STATE_BACK_DURATION_S - CYCLE_S)
    )

    assert pump.asked_for_reset_user_initiated_state_time == T_OVERRIDE
    assert pump.get_override_state() == OVERRIDE_STATE_ASKED_FOR_RESET


async def test_a_rewound_clock_does_not_freeze_the_cooldown_drain():
    """A backwards clock step must not pin the load overridden.

    The drain is the ONLY release of the cooldown, so a future-dated anchor beyond
    the skew band means "treat as fully elapsed" rather than "negative, therefore
    below the window, therefore never".
    """
    pump = _make_pump()
    # NTP corrects the clock backwards AFTER restore, so the future-dated drop in
    # `use_saved_extra_device_info` never got a chance to help
    pump.asked_for_reset_user_initiated_state_time = T_OVERRIDE + timedelta(hours=3)

    await pump.check_load_activity_and_constraints(T_OVERRIDE)

    assert pump.asked_for_reset_user_initiated_state_time is None
    assert pump.is_user_overridden() is False


async def test_a_rewound_clock_does_not_freeze_the_override_expiry():
    """The same hazard on the expiry comparison, which shares the helper."""
    pump = _make_pump()
    _arm_override(pump, time=T_OVERRIDE + timedelta(hours=3))
    _push_override_constraint(pump, time=T_OVERRIDE)
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)

    await pump.check_load_activity_and_constraints(T_OVERRIDE)

    assert pump.external_user_initiated_state is None
    assert _override_constraints(pump) == []


async def test_a_slightly_future_dated_override_is_not_destroyed():
    """Review fix #02 (must-fix): benign skew must not cancel a BRAND-NEW override.

    `_seconds_since` reads every negative delta as "fully elapsed", so a few seconds
    of future-dating expired a fresh 8 h override on the spot — the override state
    nulled, its command dropped, its constraint wiped, and the load parked in
    ASKED FOR RESET. QS fighting the user, arriving from the other direction.

    And it needs no clock anomaly at all: `user_set_bistate_mode`,
    `user_set_default_on_duration` and `async_reset_override_state` each take their
    own unlocked `datetime.now(pytz.UTC)`, while an `async_update_loads` cycle
    already in flight is still carrying an earlier `event_time` — so a user-stamped
    anchor can legitimately sit ahead of the cycle that evaluates it.
    """
    pump = _make_pump()
    _arm_override(pump)
    _push_override_constraint(pump)
    pump.current_command = copy_command(CMD_IDLE)
    pump.running_command = copy_command(CMD_ON)
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)

    # the cycle evaluating the override is 5 s BEHIND the user action that armed it
    await pump.check_load_activity_and_constraints(T_OVERRIDE - timedelta(seconds=5))

    assert pump.external_user_initiated_state == "on"
    assert pump.external_user_initiated_state_time == T_OVERRIDE
    assert pump.asked_for_reset_user_initiated_state_time is None
    assert pump.get_override_state() == f"{OVERRIDE_STATE_PREFIX}on"
    assert pump.is_user_overridden() is True
    # ...and the override's own in-flight command is not dropped either
    assert pump.running_command == CMD_ON
    assert len(_override_constraints(pump)) == 1


async def test_a_slightly_future_dated_cooldown_is_not_drained_early():
    """The symmetric half: the same band protects the cooldown timer."""
    pump = _make_pump()
    pump.asked_for_reset_user_initiated_state_time = T_OVERRIDE
    pump.current_command = copy_command(CMD_IDLE)
    pump.running_command = copy_command(CMD_ON)
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)

    await pump.check_load_activity_and_constraints(T_OVERRIDE - timedelta(seconds=5))

    assert pump.asked_for_reset_user_initiated_state_time == T_OVERRIDE
    assert pump.get_override_state() == OVERRIDE_STATE_ASKED_FOR_RESET


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
# AC4b — `support_user_override()` is not a second freeze (review fix #03)
#
# Unlike the `qs_enable_device` arm, this one is not self-healing: disabling and
# re-enabling a load runs the lifecycle again, but flipping a load to boost-only
# never does. With the lifecycle behind it, a load reconfigured to boost-only
# kept a live override or a pending reset-ask FOREVER, across restarts — exactly
# the harm this story exists to fix.
# =============================================================================


async def test_a_load_flipped_to_boost_only_still_expires_its_override():
    """The lifecycle is not gated on `support_user_override()` — detection is."""
    pump = _make_pump()
    _arm_override(pump)
    _push_override_constraint(pump)
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)

    # the user reconfigures the load as boost-only
    pump.load_is_auto_to_be_boosted = True
    assert pump.support_user_override() is False

    await pump.check_load_activity_and_constraints(T_EXPIRED)
    assert pump.external_user_initiated_state is None
    assert _override_constraints(pump) == []

    t_back = T_EXPIRED + timedelta(seconds=USER_OVERRIDE_STATE_BACK_DURATION_S + CYCLE_S)
    await pump.check_load_activity_and_constraints(t_back)

    assert pump.asked_for_reset_user_initiated_state_time is None
    assert pump.get_override_state() == OVERRIDE_STATE_NO_OVERRIDE
    assert pump.is_user_overridden() is False


async def test_a_boost_only_load_drains_a_reset_ask_restored_from_storage():
    """...and the persisted half: restore keeps a PAST-dated ask, so it must drain.

    `use_saved_extra_device_info` only drops a *future*-dated reset-ask, and the
    reset button was the sole other clearer, so a stored timestamp on a boost-only
    load survived every restart and kept `is_user_overridden()` True for good.
    """
    pump = _make_pump()
    pump.load_is_auto_to_be_boosted = True

    with freeze_time(T_EXPIRED):
        pump.use_saved_extra_device_info({STORAGE_KEY_ASKED_FOR_RESET_TIME: f"{T_OVERRIDE}"})

    # restore deliberately keeps it — nothing there expires a past-dated ask
    assert pump.asked_for_reset_user_initiated_state_time == T_OVERRIDE
    assert pump.get_override_state() == OVERRIDE_STATE_ASKED_FOR_RESET

    await pump.check_load_activity_and_constraints(T_EXPIRED)

    assert pump.asked_for_reset_user_initiated_state_time is None
    assert pump.is_user_overridden() is False


async def test_detection_stays_off_for_a_boost_only_load():
    """The conjunct still does its real job: no override detection when unsupported."""
    pump = _make_pump()
    pump.load_is_auto_to_be_boosted = True
    pump.current_command = copy_command(CMD_IDLE)
    pump.last_command_execution_time = T_OVERRIDE
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE + timedelta(seconds=30))

    t = T_OVERRIDE + timedelta(minutes=2)
    assert pump.is_load_command_set(t) is True  # the OTHER detection conjunct is open

    await pump.check_load_activity_and_constraints(t)

    assert pump.external_user_initiated_state is None
    assert _override_constraints(pump) == []


# =============================================================================
# AC25 — the expiry drop must not strand a command slot across an await
# =============================================================================


@pytest.mark.parametrize("race_result", [True, None], ids=["service-call-lands", "service-call-impossible"])
async def test_a_user_action_during_a_service_call_cannot_strand_the_command_slot(race_result):
    """Review fix #03 (must-fix): the expiry drop broke a documented invariant.

    D1d's drop is the first command-slot mutation ever performed inside
    `check_load_activity_and_constraints`, and three of that method's callers run
    outside `_update_loads_lock`. So "each command-slot mutation happens between
    `await`s" (`check_and_relaunch_command`'s docstring) stopped being true, and
    `force_relaunch_command` dereferences the slot after its `await` — either
    `_ack_command(time, None)`, which nulls `current_command` and silently drops the
    load out of controlled-consumption accounting, or a `None` deref that aborts the
    cycle.

    The fix is a post-await re-check, not a wider lock. Both post-await branches are
    exercised here: a service call that lands, and one that reports impossible.
    """
    pump = _make_pump(pump_class=_RacingPump)
    _arm_override(pump)
    _push_override_constraint(pump)
    pump.race_result = race_result
    pump.hass.states.set(PUMP_ENTITY, "off", last_changed=T_OVERRIDE)
    # a confirmed command of record, which must survive whatever happens
    pump._ack_command(T_OVERRIDE, copy_command(CMD_IDLE))
    confirmed = pump.current_command

    # the override's own ON command goes in flight and is never confirmed
    pump.obeys = False
    await pump.launch_command(T_OVERRIDE + timedelta(hours=1), CMD_ON, ctxt="override hold dispatch")
    assert pump.running_command == CMD_ON

    # the relaunch's service call is interrupted by the user, and the expiry that
    # runs on that path drops the override-aligned command underneath it
    pump.race_at = T_EXPIRED
    await pump.force_relaunch_command(T_EXPIRED)

    assert pump.running_command is None
    assert pump.external_user_initiated_state is None
    # no phantom ack of `None`: the confirmed command of record is intact...
    assert pump.current_command == confirmed
    # ...so the load is still inside the power-accounting that feeds the forecast
    assert pump.is_load_command_set(T_EXPIRED) is True
    # and the counters do not outlive the command they describe
    assert pump.running_command_num_relaunch == 0
    assert pump.running_command_last_launch is None


# =============================================================================
# AC8 — the end-to-end bug, in both death modes
# =============================================================================


def _controlled_consumption(pump: _StuckPump, time: datetime) -> float:
    """The power the home accounting attributes to this load.

    Review fix #06: this is the observable the story's Problem section actually
    blames — "permanently excluded from controlled consumption … flows into the
    persisted forecast". `get_device_power_latest_possible_valid_value` returns
    `0.0` for a user-overridden load, and that zero is what reaches
    `home.py`'s consumption accounting and the persisted forecast.
    """
    return pump.get_device_power_latest_possible_valid_value(
        tolerance_seconds=None, time=time, ignore_auto_and_user_overridden_load=True
    )


async def _detect_user_override_on(pump: _StuckPump) -> None:
    """Replay the story's 18:00: the user flips the pump ON by hand."""
    pump.current_command = copy_command(CMD_IDLE)
    pump.last_command_execution_time = T_OVERRIDE - timedelta(hours=1)
    pump.hass.states.set(PUMP_ENTITY, "off", last_changed=T_OVERRIDE - timedelta(hours=1))
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)
    # a real, measurable draw for the accounting to attribute (or refuse to)
    pump._entity_probed_last_valid_state[pump._get_power_measure()] = (
        T_OVERRIDE,
        pump.power_use,
        {},
    )

    await pump.check_load_activity_and_constraints(T_OVERRIDE + timedelta(seconds=CYCLE_S))

    assert pump.external_user_initiated_state == "on"
    assert pump.is_user_overridden() is True
    assert len(_override_constraints(pump)) == 1
    # ...and while the override stands, the load is OUT of controlled consumption
    assert _controlled_consumption(pump, T_OVERRIDE + timedelta(seconds=CYCLE_S)) == 0.0


async def _assert_override_self_heals(pump: _StuckPump) -> None:
    """The override expires on time, then the cooldown releases the load."""
    await pump.check_load_activity_and_constraints(T_EXPIRED)

    assert pump.external_user_initiated_state is None
    assert _override_constraints(pump) == []
    # review fix #02: the override's own command must not outlive the override
    assert pump.running_command is None
    transport_calls_after_expiry = len(pump.transport_calls)

    # Review fix #03: drive a FULL relaunch ladder across the post-expiry window
    # before re-reading the counter. Without it the "no further service call"
    # assertion below was vacuous — nothing could have re-issued the call in zero
    # cycles, and the auditor re-introduced the regression with the test still green.
    await drive(pump, T_EXPIRED + timedelta(seconds=CYCLE_S), LADDER_WALL_S)

    # the post-override cooldown is clock arithmetic too, so it also drains on
    # a load QS can no longer talk to — otherwise the load stayed pinned in
    # ASKED FOR RESET forever, which is the same bug one step later
    t_back = T_EXPIRED + timedelta(seconds=LADDER_WALL_S + USER_OVERRIDE_STATE_BACK_DURATION_S + CYCLE_S)
    await pump.check_load_activity_and_constraints(t_back)

    assert pump.asked_for_reset_user_initiated_state_time is None
    assert pump.get_override_state() == OVERRIDE_STATE_NO_OVERRIDE
    assert pump.is_user_overridden() is False
    # ...and the load is back in the solver with its ordinary daily constraint
    assert pump._constraints != []
    # ...and back in controlled consumption: its real draw is billed again,
    # instead of the 0.0 an override forces into the persisted forecast
    assert _controlled_consumption(pump, t_back) == pump.power_use
    # nothing kept re-issuing the ended override's service call
    assert len(pump.transport_calls) == transport_calls_after_expiry


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
