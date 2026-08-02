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

`_RacingPump` is *the* mid-await race harness, so this module hosts both stories'
race cases: QS-307's **emptied** slot (AC25) and QS-320's **replaced** slot — one
family, one view.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Any
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
    STORAGE_KEY_EXTERNAL_USER_INITIATED_STATE,
    STORAGE_KEY_EXTERNAL_USER_INITIATED_STATE_TIME,
)
from custom_components.quiet_solar.ha_model.bistate_duration import (
    USER_OVERRIDE_STATE_BACK_DURATION_S,
    QSBiStateDuration,
)
from custom_components.quiet_solar.home_model.commands import CMD_IDLE, CMD_OFF, CMD_ON, LoadCommand, copy_command
from custom_components.quiet_solar.home_model.constraints import TimeBasedSimplePowerLoadConstraint
from custom_components.quiet_solar.home_model import load as load_module
from custom_components.quiet_solar.home_model.load import NUM_MAX_COMMAND_RELAUNCH, NUM_MAX_INVALID_PROBES_COMMANDS
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

    QS-320: `race_action` makes *which* user action lands pluggable, so the same
    harness covers the slot being **replaced** (a button press that supersedes or
    resets-then-reinstalls) as well as QS-307's **emptied** case. It must always go
    through a real `launch_command` / `user_clean_and_reset` /
    `_drop_running_command` — a race action that assigns `running_command` by hand
    replaces the slot WITHOUT bumping the generation, so the guard under test would
    never trip and the test would pass vacuously.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.race_at: datetime | None = None
        # what the interrupted service call reports back: `True` reaches the ack
        # branch, `None` reaches the "impossible to force" log — both dereference
        # the command slot after the await
        self.race_result: bool | None = True
        # QS-320: the user action that lands mid-await. QS-307's override-expiry
        # drop is the default, so the two #316 tests keep their exact behaviour.
        self.race_action: Callable[[datetime], Awaitable[Any]] = self.check_load_activity_and_constraints

    async def execute_command_system(self, time, command, state):
        """Let a user action land in the middle of the service call."""
        if self.race_at is None:
            return await super().execute_command_system(time, command, state)

        self.transport_calls.append((time, command.command, state))
        race_at, self.race_at = self.race_at, None
        await self.race_action(race_at)
        return self.race_result


class _ProbeRacingPump(_RacingPump):
    """A pump whose *probe* — not its service call — is interrupted mid-flight.

    QS-320 needs this for `check_commands`, whose only await is
    `probe_if_command_set`. Real subclasses do await I/O there:
    `QSChargerGeneric.probe_if_command_set` awaits `_do_update_charger_state`, a
    genuine `hass.services.async_call`, and `QSBattery`'s awaits
    `is_charge_from_grid()`. Only the bistate probe this test file uses is
    synchronous, hence the injection.

    One-shot, like `race_at`: the injected press re-enters
    `probe_if_command_set` through `launch_command`, so a re-arming hook would
    recurse.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.probe_race_at: datetime | None = None
        self.probe_race_action: Callable[[datetime], Awaitable[Any]] = self.check_load_activity_and_constraints
        # what the interrupted probe reports back: `True` reaches the ack branch,
        # `None` the invalid-probe branch, `False` the staleness arm
        self.probe_race_result: bool | None = True

    async def probe_if_command_set(self, time, command):
        """Let a user action land in the middle of the probe."""
        if self.probe_race_at is None:
            return await super().probe_if_command_set(time, command)

        probe_race_at, self.probe_race_at = self.probe_race_at, None
        await self.probe_race_action(probe_race_at)
        return self.probe_race_result


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
    """...and the POSITIVE half, which review fix #01/07 asked to stop delegating.

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


async def test_a_confused_clock_keeps_the_override_rather_than_destroying_it():
    """AC26 (review fix #04): `main`'s plain subtraction, restored.

    Review fix #01 routed this comparison through a "clock-safe" helper whose
    "future anchor ⇒ fully elapsed" rule is right for the retry ladder it was written
    for — there it means *do not freeze the retry*. On an override it means *destroy
    what the user just set*: state nulled, the aligned command dropped, constraints
    wiped, the load parked in ASKED FOR RESET. Review fix #02's ±60 s band narrowed
    the window without closing it.

    Plain subtraction gives a negative age, which is never past the threshold, so a
    confused clock simply makes the override last a little longer. That is the benign
    direction, and it is what `main` did all along. There was never a bug here.
    """
    pump = _make_pump()
    # far beyond the tolerance band review fix #02 added, to pin that the band is gone
    _arm_override(pump, time=T_OVERRIDE + timedelta(seconds=90))
    _push_override_constraint(pump, time=T_OVERRIDE)
    pump.current_command = copy_command(CMD_IDLE)
    pump.running_command = copy_command(CMD_ON)
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)

    await pump.check_load_activity_and_constraints(T_OVERRIDE)

    assert pump.external_user_initiated_state == "on"
    assert pump.external_user_initiated_state_time == T_OVERRIDE + timedelta(seconds=90)
    assert pump.asked_for_reset_user_initiated_state_time is None
    assert pump.get_override_state() == f"{OVERRIDE_STATE_PREFIX}on"
    assert pump.is_user_overridden() is True
    # ...and the override's own in-flight command survives with it
    assert pump.running_command == CMD_ON
    assert len(_override_constraints(pump)) == 1


async def test_a_confused_clock_keeps_the_cooldown_rather_than_draining_it():
    """AC26, symmetric half — and the worse of the two.

    Draining the cooldown early also lets detection re-classify the user's manual
    action on the same cycle, so QS re-arms the override the user just cancelled.
    """
    pump = _make_pump()
    pump.asked_for_reset_user_initiated_state_time = T_OVERRIDE + timedelta(seconds=90)
    pump.current_command = copy_command(CMD_IDLE)
    pump.running_command = copy_command(CMD_ON)
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)

    await pump.check_load_activity_and_constraints(T_OVERRIDE)

    assert pump.asked_for_reset_user_initiated_state_time == T_OVERRIDE + timedelta(seconds=90)
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
# AC17-AC19 — `support_user_override()` gates the whole block, as on `main`
#
# Review fix #01 moved this conjunct down beside `is_load_command_set` so the
# lifecycle would tear down an override on a load reconfigured to boost-only.
# Review fix #04 reverted that: it ran the whole override lifecycle every cycle
# for every boost-only `QSBiStateDuration` load to serve one reconfiguration
# case, which is now handled at the restore boundary instead.
# =============================================================================


async def test_a_boost_only_load_runs_no_override_lifecycle():
    """AC17 (reworked, review fix #04): `support_user_override()` gates the block.

    Plan #02 moved this conjunct down to the detection gate so the lifecycle would
    tear down an override on a load reconfigured to boost-only. That fixed a narrow
    case by running the whole override lifecycle, every cycle, for every load that
    cannot have an override — chargers included. Reverted: the conjunct answers "can
    this load have an override at all", which is not a command-state question, so it
    belongs on the outer gate exactly as on `main`. The boost-only case is handled at
    restore instead (see the next test).
    """
    pump = _make_pump()
    _arm_override(pump)
    _push_override_constraint(pump)
    pump.asked_for_reset_user_initiated_state_time = T_OVERRIDE
    pump.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = T_OVERRIDE
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_OVERRIDE)

    pump.load_is_auto_to_be_boosted = True
    assert pump.support_user_override() is False

    await pump.check_load_activity_and_constraints(T_EXPIRED)

    # not one lifecycle branch ran: every field is exactly as it was
    assert pump.external_user_initiated_state == "on"
    assert pump.external_user_initiated_state_time == T_OVERRIDE
    assert pump.asked_for_reset_user_initiated_state_time == T_OVERRIDE
    assert pump.asked_for_reset_user_initiated_state_time_first_cmd_reset_done == T_OVERRIDE


async def test_a_boost_only_load_drops_a_stored_override_at_restore():
    """AC18 (reworked, review fix #04): the teardown belongs at the restore boundary.

    A reconfigure to boost-only reloads the config entry, which runs
    `use_saved_extra_device_info` — where stale and future-dated overrides are
    already dropped. Extending that existing condition is enough, and it costs
    nothing per cycle. Without it a stored override would be orphaned: the only code
    that could ever clear it is now gated off.
    """
    pump = _make_pump()
    pump.load_is_auto_to_be_boosted = True

    with freeze_time(T_OVERRIDE + timedelta(minutes=1)):
        pump.use_saved_extra_device_info(
            {
                STORAGE_KEY_EXTERNAL_USER_INITIATED_STATE: "on",
                STORAGE_KEY_EXTERNAL_USER_INITIATED_STATE_TIME: f"{T_OVERRIDE}",
                STORAGE_KEY_ASKED_FOR_RESET_TIME: f"{T_OVERRIDE}",
            }
        )

    # young enough that neither the staleness nor the future-dating arm would fire
    assert pump.external_user_initiated_state is None
    assert pump.external_user_initiated_state_time is None
    assert pump.asked_for_reset_user_initiated_state_time is None
    assert pump.asked_for_reset_user_initiated_state_time_first_cmd_reset_done is None
    assert pump.get_override_state() == OVERRIDE_STATE_NO_OVERRIDE
    assert pump.is_user_overridden() is False


async def test_a_load_that_supports_overrides_keeps_a_stored_override_at_restore():
    """...and the guard does not over-reach: a normal load still restores its override."""
    pump = _make_pump()

    with freeze_time(T_OVERRIDE + timedelta(minutes=1)):
        pump.use_saved_extra_device_info(
            {
                STORAGE_KEY_EXTERNAL_USER_INITIATED_STATE: "on",
                STORAGE_KEY_EXTERNAL_USER_INITIATED_STATE_TIME: f"{T_OVERRIDE}",
                STORAGE_KEY_ASKED_FOR_RESET_TIME: f"{T_OVERRIDE}",
            }
        )

    assert pump.external_user_initiated_state == "on"
    assert pump.external_user_initiated_state_time == T_OVERRIDE
    assert pump.asked_for_reset_user_initiated_state_time == T_OVERRIDE


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


@pytest.mark.parametrize("race_result", [True, None], ids=["service-call-lands", "service-call-impossible"])
async def test_a_user_action_during_the_first_launch_cannot_strand_the_command_slot(race_result):
    """AC25, second call site (review fix #04): `launch_command` has the same shape.

    Review fix #03 guarded `force_relaunch_command` and claimed the invariant break
    was closed, but `launch_command` acks after its own `await execute_command(...)`
    too — and it is the FIRST place a command reaches, so it is at least as exposed.
    Same guard, copied rather than redesigned.

    Only the ack branch was actually broken here (this site's "impossible" log already
    used a local, unlike `force_relaunch_command`'s). The impossible case is
    parametrized anyway so both post-await branches stay pinned at both call sites.
    """
    pump = _make_pump(pump_class=_RacingPump)
    _arm_override(pump)
    _push_override_constraint(pump)
    pump.race_result = race_result
    pump.hass.states.set(PUMP_ENTITY, "off", last_changed=T_OVERRIDE)
    pump._ack_command(T_OVERRIDE, copy_command(CMD_IDLE))
    confirmed = pump.current_command

    # the very first dispatch of the override's ON command is interrupted by the user
    pump.obeys = False
    pump.race_at = T_EXPIRED
    await pump.launch_command(T_EXPIRED, CMD_ON, ctxt="override hold dispatch")

    assert pump.running_command is None
    assert pump.external_user_initiated_state is None
    assert pump.current_command == confirmed
    assert pump.is_load_command_set(T_EXPIRED) is True


# =============================================================================
# QS-320 — the replaced slot
#
# #316 (above) guarded the slot being EMPTIED across an await. A slot that was
# REPLACED passes an `is None` test just as well, so a button press that
# supersedes — or resets and reinstalls — the in-flight command mid-await could
# still write the previous command's answer onto the new occupant: a phantom ack
# of a command no service call ever confirmed.
#
# Two distinct instants throughout, so a stamp written by the stale resumer is
# always distinguishable from the press's own.
# =============================================================================

T_RELAUNCH = T_OVERRIDE + timedelta(hours=1)
T_PRESS = T_RELAUNCH + timedelta(seconds=1)

_LOAD_LOGGER = "custom_components.quiet_solar.home_model.load"


def _spy_on_acks(pump: _StuckPump, monkeypatch: pytest.MonkeyPatch) -> list[LoadCommand | None]:
    """Record every command `_ack_command` is called with from now on.

    A phantom ack is the corruption this story is about, and its visible traces
    (`current_command`, `num_on_off`, the counters) are all reachable by other
    paths too. Recording the call itself is what makes "did not ack" an assertion
    about the bug rather than about its footprint. `monkeypatch` so the patch is
    restored at teardown rather than leaking off the instance.
    """
    acked: list[LoadCommand | None] = []
    real_ack = pump._ack_command

    def _record(time, command):
        acked.append(command)
        return real_ack(time, command)

    monkeypatch.setattr(pump, "_ack_command", _record)
    return acked


def _sync_race(action: Callable[[], Any]) -> Callable[[datetime], Awaitable[None]]:
    """Adapt a synchronous user action to the awaitable race-action hook."""

    async def _run(_time: datetime) -> None:
        action()

    return _run


def _race_log(caplog: pytest.LogCaptureFixture, site: str, verdict: str) -> bool:
    """True when `site` reported the slot `dropped`/`replaced` across its await.

    The dropped-vs-replaced axis IS the subject of this story, so both arms of the
    guard's one message are pinned — an operator staring at a misbehaving load needs
    to know which of the two happened.
    """
    needle = f"was {verdict} while a call to the device was in flight"
    return any(needle in message and message.startswith(f"{site}:") for message in caplog.messages)


def _arm_for_supersede(pump: _StuckPump, time: datetime = T_RELAUNCH) -> None:
    """Put the pump in the exact state where a `CMD_IDLE` press SUPERSEDES (route A).

    Every line is load-bearing, because the obvious setup reaches a *different*
    branch and the test then passes even unfixed:

    - no override armed — otherwise the press's `CMD_IDLE` is override-suppressed
      and DROPPED instead of installed;
    - `_ack_command` provides the confirmed `CMD_ON` of record. It nulls the slot as
      a side effect, hence the direct slot write after it — setup only, never inside
      a race action, where it would replace the slot without bumping the generation
      and the test would pass vacuously;
    - `unresponsive_since` set — otherwise `is_uncontrollable` is False and the
      press STACKS;
    - `_last_supersede_time` clear — otherwise the supersede is throttled and stacks;
    - the confirmed command is `CMD_ON`, not `CMD_IDLE` — otherwise the press hits
      the "already confirmed" early return and drops the stale command instead.

    The entity reads `on` and the transport stops obeying, so the press's own
    `CMD_IDLE` probe returns False and its service call leaves the replacement in
    flight and *unacked* — which is precisely what makes a phantom ack of it visible.
    """
    pump.external_user_initiated_state = None
    pump._ack_command(time, copy_command(CMD_ON))
    pump.running_command = copy_command(CMD_ON)
    pump.running_command_first_launch = time
    pump.running_command_last_launch = time
    pump.unresponsive_since = time
    pump._last_supersede_time = None
    pump.obeys = False
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=time)


async def test_the_dispatch_generation_only_ever_moves_forward():
    """AC13: the tag that makes "is the slot still MINE?" answerable.

    The guard cannot compare the command value — route B reinstalls a
    field-identical `CMD_IDLE`, so an equality test passes on the very race this
    exists to catch — nor object identity, because absorb legitimately swaps in an
    equal object for the SAME dispatch. Only a monotonic per-install tag
    distinguishes them, and it is monotonic ONLY as long as nothing ever rewinds
    it: a `reset()` that zeroed it would let a tag captured before the reset
    compare equal to one issued after it, reopening QS-320 on exactly route B.
    """
    pump = _make_pump(pump_class=_RacingPump)
    pump.hass.states.set(PUMP_ENTITY, "off", last_changed=T_RELAUNCH)
    pump.obeys = False
    start = pump._running_command_generation

    await pump.launch_command(T_RELAUNCH, CMD_ON, ctxt="first install")
    first = pump._running_command_generation
    assert first > start

    # absorb: the same command is already in flight, so this is not a new dispatch
    # and the caller already awaiting still owns its paperwork
    await pump.launch_command(T_RELAUNCH, CMD_ON, ctxt="absorbed")
    assert pump.running_command == CMD_ON
    assert pump._running_command_generation == first

    # route B's `reset()` wipes the whole command state — it must NOT rewind the tag
    pump.reset()
    assert pump.running_command is None
    assert pump._running_command_generation == first

    await pump.launch_command(T_PRESS, CMD_ON, ctxt="reinstall after reset")
    assert pump._running_command_generation > first


def test_every_running_command_write_site_is_sanctioned():
    """Review fix #01/6: the "one way in" install invariant, enforced.

    The generation guard is sound only while every NEW-dispatch slot write bumps
    the tag — i.e. goes through `_install_running_command`. That invariant was
    held by a docstring alone, and QS-320 exists precisely because a future
    install site can silently reopen it. This scans the whole package for direct
    `self.running_command = ...` assignments and fails on any site outside the
    sanctioned five, matched by enclosing function (not line number, so the test
    does not rot on unrelated edits):

    - `constraint_reset_and_reset_commands_if_needed` — the reset wipe (`None`)
    - `_ack_command` — the ack clear (`None`)
    - `abandon_running_command` — the abandon clear (`None`)
    - `_install_running_command` — THE install, the only generation bump
    - `launch_command` — absorb ONLY: an equal-valued object for the SAME
      dispatch, deliberately generation-neutral
    """
    package = Path(load_module.__file__).resolve().parents[1]
    # Review fix #02/2: keys are CLASS-QUALIFIED. `(basename, bare-function)` was
    # ambiguous — `constraint_reset_and_reset_commands_if_needed` exists on both
    # `AbstractDevice` and `AbstractLoad`, so a write added to an override of any
    # sanctioned name would have been silently sanctioned by name-match.
    # Review fix #03/5: paths are PACKAGE-RELATIVE, so a second `load.py` anywhere
    # in the package cannot collide with the sanctioned entries.
    sanctioned = {
        ("home_model/load.py", "AbstractDevice.constraint_reset_and_reset_commands_if_needed"),
        ("home_model/load.py", "AbstractDevice._ack_command"),
        ("home_model/load.py", "AbstractDevice.abandon_running_command"),
        ("home_model/load.py", "AbstractDevice._install_running_command"),
        ("home_model/load.py", "AbstractDevice.launch_command"),
    }

    def flattened(target: ast.expr):
        """Yield leaf targets through tuple/list/starred unpacking (#02/6)."""
        if isinstance(target, ast.Tuple | ast.List):
            for element in target.elts:
                yield from flattened(element)
        elif isinstance(target, ast.Starred):
            yield from flattened(target.value)
        else:
            yield target

    def binding_targets(node: ast.AST) -> list[ast.expr]:
        """The ACCIDENTAL syntactic forms that can bind `self.running_command`.

        #02/6 + #03/1: assignment (plain, annotated, augmented), `for` loops,
        `with ... as`, and comprehension for-targets (`[0 for self.running_command
        in cmds]` parses AND binds at runtime; `iter_child_nodes` reaches the
        `comprehension` node under all four comprehension kinds). Deliberate
        evasion (aliased `setattr`, `object.__setattr__`, exec, C extensions) is
        OUTSIDE this lint's threat model — see `is_sneaky_write_of_running_command`.
        """
        if isinstance(node, ast.Assign):
            return node.targets
        if isinstance(node, ast.AnnAssign | ast.AugAssign):
            return [node.target]
        if isinstance(node, ast.For | ast.AsyncFor):
            return [node.target]
        if isinstance(node, ast.withitem) and node.optional_vars is not None:
            return [node.optional_vars]
        if isinstance(node, ast.comprehension):
            return [node.target]
        return []

    def is_sneaky_write_of_running_command(node: ast.AST) -> bool:
        """Call/subscript forms a target-based scan misses (#02/6, #03/3).

        Covers the cheap accidental spellings: bare `setattr(...)`,
        `builtins.setattr(...)`. The `self.__dict__["running_command"] = x`
        subscript-store is caught by the leaf matcher below. Deliberate evasion
        (aliasing `setattr`, `object.__setattr__`, exec) is out of scope.
        """
        if not (isinstance(node, ast.Call) and len(node.args) >= 2):
            return False
        is_setattr = (isinstance(node.func, ast.Name) and node.func.id == "setattr") or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "setattr"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "builtins"
        )
        return is_setattr and isinstance(node.args[1], ast.Constant) and node.args[1].value == "running_command"

    def is_running_command_leaf(leaf: ast.expr) -> bool:
        """`self.running_command` or `self.__dict__["running_command"]` (#03/3)."""
        if (
            isinstance(leaf, ast.Attribute)
            and leaf.attr == "running_command"
            and isinstance(leaf.value, ast.Name)
            and leaf.value.id == "self"
        ):
            return True
        return (
            isinstance(leaf, ast.Subscript)
            and isinstance(leaf.slice, ast.Constant)
            and leaf.slice.value == "running_command"
            and isinstance(leaf.value, ast.Attribute)
            and leaf.value.attr == "__dict__"
            and isinstance(leaf.value.value, ast.Name)
            and leaf.value.value.id == "self"
        )

    found: set[tuple[str, str]] = set()
    for py_file in sorted(package.rglob("*.py")):
        # `filename=` so a syntax error names the file, not `<unknown>` (#02/5)
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        rel_path = py_file.relative_to(package).as_posix()
        # (node, dotted qualifier) worklist: ClassDef and function defs both extend
        # the qualifier, so each write is keyed by its class-qualified location
        stack: list[tuple[ast.AST, str]] = [(tree, "")]
        while stack:
            node, qual = stack.pop()
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                qual = f"{qual}.{node.name}" if qual else node.name
            if is_sneaky_write_of_running_command(node):
                found.add((rel_path, qual or "<module>"))
            for target in binding_targets(node):
                for leaf in flattened(target):
                    if is_running_command_leaf(leaf):
                        found.add((rel_path, qual or "<module>"))
            stack.extend((child, qual) for child in ast.iter_child_nodes(node))

    # Both directions reported (#02/7): an extra site is a QS-320 reopening; a
    # missing one means a sanctioned function was renamed/moved and the sanction
    # list must follow it.
    assert found == sanctioned, (
        f"unsanctioned `self.running_command` write site(s): {sorted(found - sanctioned)}; "
        f"sanctioned site(s) no longer found (renamed/moved?): {sorted(sanctioned - found)} — "
        "a new dispatch install MUST go through `_install_running_command` so the "
        "generation tag is bumped, or QS-320 silently reopens"
    )


async def test_a_press_that_supersedes_mid_relaunch_cannot_ack_the_command_it_replaced(caplog, monkeypatch):
    """AC1/AC8/AC9/AC10/AC12 — the headline bug, at `force_relaunch_command`.

    The relaunch ladder's service call for `CMD_ON` is in flight when the user
    presses "mark current constraint done". That press supersedes the slot with its
    own `CMD_IDLE`, and on `main` the resumer then finishes `CMD_ON`'s paperwork
    against the new occupant: it stamps `CMD_IDLE`'s launch time with the *relaunch*
    instant and acks `CMD_IDLE` off `CMD_ON`'s result. QS records "the device
    confirmed idle" though no `idle` service call ever completed, the load keeps
    running, and nothing retries because the slot is empty and `current_command`
    says idle.
    """
    pump = _make_pump(pump_class=_RacingPump)
    _arm_for_supersede(pump)
    assert pump.is_uncontrollable is True  # pre-race: the supersede branch is armed
    rung_before = pump.running_command_num_relaunch
    num_on_off_before = pump.num_on_off
    acked = _spy_on_acks(pump, monkeypatch)

    pump.race_at = T_PRESS
    pump.race_action = lambda t: pump.launch_command(t, CMD_IDLE, ctxt="button: mark constraint done")

    with caplog.at_level(logging.INFO, logger=_LOAD_LOGGER):
        await pump.force_relaunch_command(T_RELAUNCH)

    # the supersede really ran: this is the REPLACED case, not #316's emptied one
    assert pump.running_command == CMD_IDLE
    assert pump.current_command == CMD_ON

    # none of `CMD_ON`'s outcome was written onto the press's `CMD_IDLE`
    assert acked == []
    assert pump.running_command_last_launch == T_PRESS  # the press's stamp, not T_RELAUNCH
    assert pump.num_on_off == num_on_off_before  # no phantom transition on the daily budget
    assert pump.unresponsive_since == T_RELAUNCH  # the lost-control episode is not over

    # AC8: the load is not stranded — the replacement is in flight and unacked, so
    # the lifecycle still has something to drive
    assert pump.is_load_command_set(T_PRESS) is False

    # AC9 (D7): the ladder rung is deliberately INHERITED across a supersede — that
    # is QS-304's saturated cadence, not a defect this story introduces
    assert pump.running_command_num_relaunch == rung_before + 1

    # AC10: a service call really landed, so the causality anchor must fire even
    # though the slot changed hands
    assert pump.last_command_execution_time == T_RELAUNCH

    # AC12: the "replaced" arm of the guard's message
    assert _race_log(caplog, "force_relaunch_command", "replaced")


async def test_a_superseded_dispatch_cannot_rewind_the_causality_anchor():
    """Review fix #01/1: the anchor is monotonic — it never moves backwards.

    `_anchor_causality_guard_if_executed` sits ABOVE the ownership guard on
    purpose (a service call physically landed, whoever owns the slot). But the
    replacing dispatch may have already anchored a NEWER instant: here the press
    obeys, so its `CMD_IDLE` really lands, anchors `T_PRESS` and acks. When the
    superseded dispatch then resumes and anchors its older `T_RELAUNCH`, a plain
    assignment rewinds the causality floor by one second — and the freshness test
    in `check_load_activity_and_constraints` then classifies the entity state QS
    itself just caused as an external user override, freezing the load for
    `override_duration` hours on QS's own command.
    """
    pump = _make_pump(pump_class=_RacingPump)
    _arm_for_supersede(pump)
    # the press must really land: its execute moves the entity and anchors T_PRESS
    pump.obeys = True

    pump.race_at = T_PRESS
    pump.race_action = lambda t: pump.launch_command(t, CMD_IDLE, ctxt="button: mark constraint done")

    await pump.force_relaunch_command(T_RELAUNCH)

    # the press's dispatch went the whole way: executed, anchored, acked
    assert pump.current_command == CMD_IDLE
    assert pump.running_command is None
    # and the stale resumer did not drag the anchor one second into the past
    assert pump.last_command_execution_time == T_PRESS


async def test_a_press_that_supersedes_the_first_dispatch_cannot_be_acked_by_it(monkeypatch):
    """AC2/AC8 — the same bug at `launch_command`, the first place a command reaches.

    The entity is unavailable, so both probes return `None`: the outer dispatch falls
    through to `execute_command` (where the press lands) instead of self-acking, and
    the press's own dispatch executes and reports "dispatched, unconfirmed".
    """
    pump = _make_pump(pump_class=_RacingPump)
    pump.external_user_initiated_state = None
    pump._ack_command(T_RELAUNCH, copy_command(CMD_ON))  # the confirmed command of record
    pump.unresponsive_since = T_RELAUNCH  # so the press supersedes rather than stacks
    pump._last_supersede_time = None
    pump.obeys = False
    pump.hass.states.set(PUMP_ENTITY, STATE_UNAVAILABLE, last_changed=T_RELAUNCH)
    acked = _spy_on_acks(pump, monkeypatch)

    pump.race_at = T_PRESS
    pump.race_action = lambda t: pump.launch_command(t, CMD_IDLE, ctxt="button: clean and reset")

    # `CMD_OFF`: the outer dispatch must differ from the confirmed `CMD_ON` (else it
    # early-returns without ever installing) AND from the press's `CMD_IDLE` (else
    # the press absorbs instead of superseding)
    await pump.launch_command(T_RELAUNCH, CMD_OFF, ctxt="solver dispatch")

    assert pump.running_command == CMD_IDLE  # the press's command, still in flight
    assert pump.current_command == CMD_ON  # NOT the phantom `CMD_IDLE`
    assert acked == []
    assert pump.is_load_command_set(T_RELAUNCH) is False
    # review fix #01/2: the execute returned True, so the device ANSWERED — proven
    # contact must end the lost-control episode even though the ack is withheld
    assert pump.unresponsive_since is None


async def test_a_reset_that_reinstalls_an_identical_command_is_not_acked_by_the_old_one(monkeypatch):
    """AC3 — route B, the case a value-equality guard silently misses.

    "Clean and reset" `reset()`s the whole command state and then installs a fresh
    `CMD_IDLE`. When the in-flight command was already `CMD_IDLE`, that replacement
    compares **equal on every field** to the one being awaited — so a guard that
    compared commands instead of dispatches would wave the stale resumer through and
    let it stamp and ack a dispatch whose service call is still in flight. This test
    must FAIL under value equality and PASS under the generation tag.

    No preconditions: `reset()` is unconditional, which is why route B is the
    disproportionately dangerous one. Stamps are deliberately not asserted here —
    `user_clean_and_reset` uses `datetime.now()`, not the test clock.
    """
    pump = _make_pump(pump_class=_RacingPump)
    pump.external_user_initiated_state = None
    pump.running_command = copy_command(CMD_IDLE)  # setup-only direct write
    pump.running_command_first_launch = T_RELAUNCH
    pump.running_command_last_launch = T_RELAUNCH
    pump.obeys = False
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_RELAUNCH)
    acked = _spy_on_acks(pump, monkeypatch)

    pump.race_at = T_PRESS
    pump.race_action = lambda t: pump.user_clean_and_reset()

    await pump.force_relaunch_command(T_RELAUNCH)

    # a field-identical `CMD_IDLE` now occupies the slot, from a different dispatch
    assert pump.running_command == CMD_IDLE
    # the stale resumer must not ack it: `current_command is None` is the `reset()`'s
    # own doing, and would read `CMD_IDLE` under the phantom ack
    assert acked == []
    assert pump.current_command is None


@pytest.mark.parametrize("race_result", [True, None], ids=["service-call-lands", "service-call-impossible"])
async def test_an_absorbed_command_still_belongs_to_the_caller_in_flight(race_result, monkeypatch):
    """AC4 — absorb must NOT trip the guard. The mirror of AC1.

    The absorb branch fires when the incoming command equals the one in flight and
    swaps in a **new object of equal value** for the SAME dispatch, leaving the
    launch stamps and counters alone. Both buttons launch `CMD_IDLE`, so a press
    against an in-flight `CMD_IDLE` takes absorb — and an identity-based guard would
    read that as "replaced" and bail out *before* `running_command_last_launch =
    time`. The stamp would stay stale, the backoff gate would then be permanently
    open, and the load would make a service call every cycle while the rung climbed
    to a false lost-control escalation.

    Green both pre- and post-fix: it pins the design against the identity variant.
    """
    pump = _make_pump(pump_class=_RacingPump)
    pump.external_user_initiated_state = None
    pump.running_command = copy_command(CMD_IDLE)  # setup-only direct write
    pump.running_command_first_launch = T_RELAUNCH - timedelta(seconds=1)
    # review fix #01/4: the setup stamp must DIFFER from the relaunch instant, or
    # the "still stamped" assertion below holds whether or not the stamp was
    # rewritten and the clause AC4 exists to prove is unobservable
    pump.running_command_last_launch = T_RELAUNCH - timedelta(seconds=1)
    pump.obeys = False
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_RELAUNCH)
    pump.race_result = race_result
    slot_before = pump.running_command
    generation_before = pump._running_command_generation
    acked = _spy_on_acks(pump, monkeypatch)

    pump.race_at = T_PRESS
    pump.race_action = lambda t: pump.launch_command(t, CMD_IDLE, ctxt="button: mark constraint done")

    await pump.force_relaunch_command(T_RELAUNCH)

    # absorb is the one non-`None` slot write that does NOT bump the generation
    assert pump._running_command_generation == generation_before

    if race_result is True:
        # the caller in flight still owns its paperwork, so it still acks
        assert acked == [CMD_IDLE]
        assert pump.current_command == CMD_IDLE
    else:
        # the "impossible" arm: no ack, but the stamp is still ours to write, and
        # the swapped-in object proves absorb really ran
        assert acked == []
        assert pump.running_command_last_launch == T_RELAUNCH
        assert pump.running_command == CMD_IDLE
        assert pump.running_command is not slot_before


async def test_check_commands_cannot_ack_a_command_the_probe_never_saw(monkeypatch):
    """AC5 — the third site. `check_commands` had no post-await re-check at all.

    Its await is the probe, and real subclasses do await I/O there —
    `QSChargerGeneric.probe_if_command_set` awaits a genuine service call. A probe
    that answers `True` about the command that WAS in the slot then acks whatever is
    in the slot now.
    """
    pump = _make_pump(pump_class=_ProbeRacingPump)
    _arm_for_supersede(pump)
    acked = _spy_on_acks(pump, monkeypatch)

    pump.probe_race_at = T_PRESS
    pump.probe_race_action = lambda t: pump.launch_command(t, CMD_IDLE, ctxt="button: mark constraint done")
    pump.probe_race_result = True

    res, command_acked_or_good = await pump.check_commands(T_RELAUNCH)

    assert pump.running_command == CMD_IDLE  # the replacement, unacked
    assert acked == []
    assert pump.current_command == CMD_ON
    # "not confirmed this cycle, re-check next cycle" — the safe answer for a probe
    # that told us nothing about the current occupant
    assert command_acked_or_good is False
    assert res == timedelta(0)


async def test_a_confirming_probe_ends_the_lost_control_episode_even_when_the_slot_changed_hands(monkeypatch):
    """Review fix #01/2: contact evidence must not ride on the ack.

    A `True` probe proves the device ANSWERED — that is dispatch-independent
    contact (`ContactEvidence.CONFIRMED` is documented as "an ack is contact
    whether or not a clock was live"). On `main` the phantom ack cleared the
    episode as a side effect; QS-320 correctly removed the ack, but the contact
    signal must survive on its own, or a live lost-control episode outlasts
    demonstrated contact and the next solver command supersedes (an extra service
    call) instead of stacking.

    The ack, the stamp and the counters stay withheld — only the episode ends.
    """
    pump = _make_pump(pump_class=_ProbeRacingPump)
    _arm_for_supersede(pump)
    assert pump.unresponsive_since == T_RELAUNCH
    acked = _spy_on_acks(pump, monkeypatch)

    pump.probe_race_at = T_PRESS
    pump.probe_race_action = lambda t: pump.launch_command(t, CMD_IDLE, ctxt="button: mark constraint done")
    pump.probe_race_result = True

    _res, command_acked_or_good = await pump.check_commands(T_RELAUNCH)

    # proven contact: the episode is over, the shout latch dropped with it
    assert pump.unresponsive_since is None
    assert pump._unresponsive_needs_ack is False
    # but the ack is still withheld — the probe confirmed the PREVIOUS occupant
    assert acked == []
    assert pump.current_command == CMD_ON
    assert command_acked_or_good is False


# The instant of the "following cycle" in the two re-announce tests below: past
# `T_PRESS` so the successor's launch stamp is in the past, and well inside even
# the rung-0 backoff (50 s) so the cycle relaunches nothing.
T_NEXT = T_PRESS + timedelta(seconds=5)


def _arm_announced_episode(pump: _StuckPump) -> None:
    """Saturate the ladder and mark the episode as already announced (QS-319).

    `unresponsive_since` is only ever set once the rung reaches
    `NUM_MAX_COMMAND_RELAUNCH`, and announcing sets the latch — so this is the
    ONLY state in which the disowned-slot CONFIRMED clear can fire on a replaced
    slot in production (a press can only supersede while `is_uncontrollable`).
    """
    pump.running_command_num_relaunch = NUM_MAX_COMMAND_RELAUNCH
    pump._unresponsive_needs_ack = True


async def test_proven_contact_on_a_replaced_slot_does_not_reannounce_the_episode(caplog, monkeypatch):
    """Review fix #02/1, `check_commands` site: contact must END the episode, not restart it.

    The #01/2 CONFIRMED clear drops both the clock and the QS-319 announce latch
    — but the successor installed by the racing press inherits a saturated rung
    it never earned. Unfixed, the very next `_escalate_or_recover` pass re-marks
    the clock and, with the latch down, takes the ANNOUNCE branch: a fresh
    "Lost control of load" ERROR with an inherited relaunch count, a second push,
    and an on→off→on flap of the `qs_load_lost_control` PROBLEM sensor —
    milliseconds after the device demonstrably answered. The ladder's premise is
    "device not answering"; a confirming probe disproves it, so the successor
    starts at rung 0. The clear also wiped `_last_supersede_time` — here the
    stamp the successor's OWN supersede wrote a second earlier, so the
    one-per-window throttle invariant lost its anchor; it is preserved.
    """
    pump = _make_pump(pump_class=_ProbeRacingPump)
    _arm_for_supersede(pump)
    _arm_announced_episode(pump)
    pushes: list[LoadCommand] = []

    async def _spy_push(time, command):
        pushes.append(command)

    monkeypatch.setattr(pump, "_notify_unresponsive", _spy_push)

    pump.probe_race_at = T_PRESS
    pump.probe_race_action = lambda t: pump.launch_command(t, CMD_IDLE, ctxt="button: mark constraint done")
    pump.probe_race_result = True

    with caplog.at_level(logging.INFO, logger=_LOAD_LOGGER):
        await pump.check_commands(T_RELAUNCH)

        # the successor performed zero relaunches of its own: rung reset, episode
        # over, and its own supersede stamp survives the clear
        assert pump.running_command_num_relaunch == 0
        assert pump._last_supersede_time == T_PRESS
        assert pump.unresponsive_since is None
        assert pump.has_unacknowledged_lost_control is False

        # the following driver cycle must not re-mark, re-announce or re-push
        await pump.check_and_relaunch_command(T_NEXT)

    assert not any("Lost control of load" in message for message in caplog.messages)
    assert pushes == []
    # the QS-319 pin: no PROBLEM-sensor re-flap — False, and it STAYS False
    assert pump.has_unacknowledged_lost_control is False
    assert pump.unresponsive_since is None
    assert pump.running_command == CMD_IDLE  # the successor is intact and in flight


async def test_proven_contact_on_a_replaced_first_dispatch_does_not_reannounce_the_episode(caplog, monkeypatch):
    """Review fix #02/1, `launch_command` site: same scenario, one cycle later.

    Identical mechanics, but the disowned-slot clear fires in `launch_command`,
    so the spurious re-announce would land on the NEXT `check_and_relaunch_command`
    cycle — with the PROBLEM sensor reading "resolved" across a full cycle
    boundary in between, a real off→on edge for any automation on it.
    """
    pump = _make_pump(pump_class=_RacingPump)
    pump.external_user_initiated_state = None
    pump._ack_command(T_RELAUNCH, copy_command(CMD_ON))  # confirmed of record, slot -> None
    pump.unresponsive_since = T_RELAUNCH
    _arm_announced_episode(pump)
    pump._last_supersede_time = None
    pump.obeys = False
    pump.hass.states.set(PUMP_ENTITY, "on", last_changed=T_RELAUNCH)
    pushes: list[LoadCommand] = []

    async def _spy_push(time, command):
        pushes.append(command)

    monkeypatch.setattr(pump, "_notify_unresponsive", _spy_push)

    pump.race_at = T_PRESS
    pump.race_action = lambda t: pump.launch_command(t, CMD_IDLE, ctxt="button: mark constraint done")

    with caplog.at_level(logging.INFO, logger=_LOAD_LOGGER):
        # `CMD_OFF`: differs from the confirmed `CMD_ON` (no early return) and from
        # the press's `CMD_IDLE` (no absorb), and the entity reads `on` so the probe
        # returns False and the dispatch reaches the execute where the race lands
        await pump.launch_command(T_RELAUNCH, CMD_OFF, ctxt="solver dispatch")

        assert pump.running_command_num_relaunch == 0
        assert pump._last_supersede_time == T_PRESS
        assert pump.unresponsive_since is None
        assert pump.has_unacknowledged_lost_control is False

        await pump.check_and_relaunch_command(T_NEXT)

    assert not any("Lost control of load" in message for message in caplog.messages)
    assert pushes == []
    assert pump.has_unacknowledged_lost_control is False
    assert pump.unresponsive_since is None
    assert pump.running_command == CMD_IDLE


async def test_check_commands_hands_no_invalid_probe_strike_to_the_successor():
    """AC6 — a deliberate behaviour change: the successor inherits no strike.

    An invalid probe about the *previous* occupant says nothing about the new one, so
    `running_command_num_relaunch_after_invalid` must not be charged. Pre-fix the
    counter reads 1 — the supersede zeroed it and the stale resumer then incremented
    it — and the successor starts life one strike closer to being killed off.
    """
    pump = _make_pump(pump_class=_ProbeRacingPump)
    _arm_for_supersede(pump)

    pump.probe_race_at = T_PRESS
    pump.probe_race_action = lambda t: pump.launch_command(t, CMD_IDLE, ctxt="button: mark constraint done")
    pump.probe_race_result = None

    res, command_acked_or_good = await pump.check_commands(T_RELAUNCH)

    assert pump.running_command == CMD_IDLE
    assert pump.running_command_num_relaunch_after_invalid == 0
    assert command_acked_or_good is False
    # and no staleness computed from the previous dispatch's stamp (pre-fix: -1 s,
    # because the press's stamp is LATER than the relaunch instant)
    assert res == timedelta(0)
    # review fix #01/2: a `None` probe is NOT contact — the episode-ending clear is
    # gated on `is True`, so the lost-control episode survives here
    assert pump.unresponsive_since == T_RELAUNCH


async def test_check_commands_cannot_ack_none_onto_an_emptied_slot(caplog, monkeypatch):
    """AC7/AC12 — `check_commands` also carried an unguarded EMPTIED-slot hole.

    With no post-await check, an emptied slot reached `_ack_command(time, None)`,
    which nulls `current_command` — wiping the confirmed command of record and
    dropping the load out of controlled-consumption accounting.

    The bail-out is a flag rather than an early `return` precisely so the
    stacked-promotion tail still runs in the same cycle; an early return would delay
    an emptied-slot promotion by a whole cycle, a behaviour change on an unrelated
    path.
    """
    pump = _make_pump(pump_class=_ProbeRacingPump)
    pump.external_user_initiated_state = None
    pump._ack_command(T_RELAUNCH, copy_command(CMD_ON))
    pump.running_command = copy_command(CMD_ON)  # setup-only direct write
    pump.running_command_first_launch = T_RELAUNCH
    pump.running_command_last_launch = T_RELAUNCH
    pump.obeys = False
    # `off`: the promoted stacked command must not be able to self-ack and mask the
    # corruption this test is looking for
    pump.hass.states.set(PUMP_ENTITY, "off", last_changed=T_RELAUNCH)
    # `abandon_running_command` leaves the stack alone, so the tail must still see it
    pump._stacked_command = copy_command(CMD_ON)
    acked = _spy_on_acks(pump, monkeypatch)

    pump.probe_race_at = T_PRESS
    pump.probe_race_action = _sync_race(lambda: pump._drop_running_command("test: the user emptied the slot"))
    pump.probe_race_result = True

    with caplog.at_level(logging.INFO, logger=_LOAD_LOGGER):
        _res, command_acked_or_good = await pump.check_commands(T_RELAUNCH)

    assert acked == []  # pre-fix: `_ack_command(time, None)`
    assert pump.current_command == CMD_ON  # the confirmed command of record survives
    assert command_acked_or_good is False
    assert pump._stacked_command is None  # the promotion tail still ran this cycle
    assert _race_log(caplog, "check_commands", "dropped")


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
    # review fix #01/02: the override's own command must not outlive the override
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
