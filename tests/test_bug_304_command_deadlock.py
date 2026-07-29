"""QS-304: a load whose command never lands must never deadlock its command slot.

Pure-layer regression tests. This module deliberately imports **no**
`homeassistant.*` module (AC14a): the whole relaunch/supersede/escalate
lifecycle is decided in `home_model/load.py` and must be provable without
Home Assistant.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytz

from custom_components.quiet_solar.const import DEVICE_STATUS_CHANGE_ERROR
from custom_components.quiet_solar.home_model.commands import (
    CMD_IDLE,
    CMD_OFF,
    CMD_ON,
    LoadCommand,
    copy_command,
)
from custom_components.quiet_solar.home_model.load import (
    COMMAND_RELAUNCH_BASE_DELAY_S,
    NUM_MAX_COMMAND_RELAUNCH,
    NUM_MAX_INVALID_PROBES_COMMANDS,
    SUPERSEDE_MIN_INTERVAL_S,
)
from tests.factories import (
    NeverAcksDevice,
    NeverAcksLoad,
    RaisingExecuteLoad,
    RaisingProbeLoad,
)

T0 = datetime(2026, 7, 27, 12, 12, 19, tzinfo=pytz.UTC)

# The observed quiet-solar load-management cycle period.
CYCLE_S = 7

# Cumulative wall time of the full 50/100/150/200/250/300 ladder.
LADDER_TOTAL_S = 1050

# Each rung fires on the first load-management cycle AFTER its deadline, so the
# observed wall time runs up to one cycle per rung longer — 17 m 52 s in the
# incident log against a nominal 17 m 30 s.
LADDER_WALL_S = LADDER_TOTAL_S + NUM_MAX_COMMAND_RELAUNCH * CYCLE_S

LOST_CONTROL_LOG = "Lost control of load"
REGAINED_CONTROL_LOG = "Regained control of load"


async def drive(load, start: datetime, duration_s: float, step_s: float = CYCLE_S) -> datetime:
    """Run the load-management cycle over `duration_s`, returning the next cycle time."""
    time = start
    end = start + timedelta(seconds=duration_s)
    while time <= end:
        await load.check_and_relaunch_command(time)
        time = time + timedelta(seconds=step_s)
    return time


async def drive_until_uncontrollable(load, start: datetime = T0) -> datetime:
    """Launch `idle`, spin the ladder out, and return the next cycle time."""
    await load.launch_command(start, CMD_IDLE)
    time = await drive(load, start + timedelta(seconds=CYCLE_S), LADDER_WALL_S)
    assert load.is_uncontrollable is True
    return time


# =============================================================================
# AC2 — the ladder saturates instead of terminating
# =============================================================================


def test_relaunch_delay_saturates_at_rung_six():
    """The backoff grows to 300 s and then stays there, forever."""
    load = NeverAcksLoad(name="pool_house")

    expected = {0: 50.0, 1: 100.0, 2: 150.0, 3: 200.0, 4: 250.0, 5: 300.0, 6: 300.0, 7: 300.0}
    for rung, delay in expected.items():
        load.running_command_num_relaunch = rung
        assert load.command_relaunch_delay_s() == delay

    # The ladder's total wall time is unchanged from the pre-QS-304 behaviour.
    load.running_command_num_relaunch = 0
    total = sum(COMMAND_RELAUNCH_BASE_DELAY_S * (n + 1) for n in range(NUM_MAX_COMMAND_RELAUNCH))
    assert total == LADDER_TOTAL_S
    assert SUPERSEDE_MIN_INTERVAL_S == 300


# =============================================================================
# AC3 — one durable clock, and it cannot stick
# =============================================================================


def test_is_uncontrollable_requires_in_flight_command():
    """`is_uncontrollable` needs both the clock AND something still in flight."""
    load = NeverAcksLoad(name="pool_house")

    assert load.unresponsive_since is None
    assert load.is_uncontrollable is False

    load.unresponsive_since = T0
    assert load.is_uncontrollable is False  # nothing in flight: not stuck

    load.running_command = copy_command(CMD_IDLE)
    assert load.is_uncontrollable is True

    load.unresponsive_since = None
    assert load.is_uncontrollable is False


async def test_unresponsive_since_survives_a_constraint_reset():
    """A constraint reset must not silently clear the lost-control clock."""
    load = await _uncontrollable_load()
    load.constraint_reset_and_reset_commands_if_needed(keep_commands=True)
    assert load.unresponsive_since is not None


async def test_recovery_clears_and_logs_once(caplog: pytest.LogCaptureFixture):
    """A real ack clears the clock and logs the recovery exactly once."""
    load = NeverAcksLoad(name="pool_house")
    time = await drive_until_uncontrollable(load)

    with caplog.at_level(logging.INFO):
        load.probe_result = True
        time = await drive(load, time, 4 * SUPERSEDE_MIN_INTERVAL_S)

    assert load.unresponsive_since is None
    assert load.is_uncontrollable is False
    assert _count(caplog, REGAINED_CONTROL_LOG) == 1


async def test_recovery_after_equal_command_early_return(caplog: pytest.LogCaptureFixture):
    """The `current_command == command` early-return empties the slot: recover once."""
    load = NeverAcksLoad(name="pool_house")
    # The device is confirmed `on`, then asked to go `idle` and never obeys.
    await load.launch_command(T0 - timedelta(seconds=60), CMD_ON)
    load._ack_command(T0 - timedelta(seconds=60), copy_command(CMD_ON))
    time = await drive_until_uncontrollable(load)

    with caplog.at_level(logging.INFO):
        # The solver wants `on` again — exactly what the device already is.
        await load.launch_command(time, CMD_ON)
        assert load.running_command is None
        assert load.current_command == CMD_ON
        await load.check_and_relaunch_command(time + timedelta(seconds=CYCLE_S))

    assert load.unresponsive_since is None
    assert load.is_uncontrollable is False
    assert _count(caplog, REGAINED_CONTROL_LOG) == 1


async def test_recovery_after_override_suppression_drop(caplog: pytest.LogCaptureFixture):
    """An override-suppression drop empties the slot: recover once, not forever stuck."""
    load = NeverAcksLoad(name="pool_house")
    load._ack_command(T0 - timedelta(seconds=60), copy_command(CMD_ON))
    time = await drive_until_uncontrollable(load)

    with caplog.at_level(logging.INFO):
        load.suppress_override = True
        time = await drive(load, time, 2 * SUPERSEDE_MIN_INTERVAL_S)

    assert load.running_command is None
    assert load.unresponsive_since is None
    assert load.is_uncontrollable is False
    assert _count(caplog, REGAINED_CONTROL_LOG) == 1


async def test_re_arm_resets_the_ladder_rung_so_the_next_command_starts_fresh():
    """An emptied slot must not poison the next command with a stale rung.

    Review finding: `abandon_running_command` preserves
    `running_command_num_relaunch` so a supersede keeps the saturated 300 s
    cadence — but the equal-command early-return abandons with NO successor.
    Leaving the rung at 6 made the next, entirely unrelated command cross the
    threshold on its first cycle, with zero relaunches and a false
    "after 6 relaunches" push.
    """
    load = NeverAcksLoad(name="pool_house")
    load._ack_command(T0 - timedelta(seconds=60), copy_command(CMD_ON))
    time = await drive_until_uncontrollable(load)
    assert load.running_command_num_relaunch == NUM_MAX_COMMAND_RELAUNCH

    # The solver reverts to `on`, which equals the preserved `current_command`,
    # so the supersede lands on the early-return and leaves the slot empty.
    await load.launch_command(time, CMD_ON)
    assert load.running_command is None

    # One cycle re-arms the detection AND resets the rung.
    await load.check_and_relaunch_command(time + timedelta(seconds=CYCLE_S))
    assert load.unresponsive_since is None
    assert load.running_command_num_relaunch == 0

    # A genuinely new, differing command must get a full fresh ladder.
    time = time + timedelta(seconds=2 * CYCLE_S)
    await load.launch_command(time, CMD_OFF)
    assert load.running_command == CMD_OFF

    await load.check_and_relaunch_command(time + timedelta(seconds=CYCLE_S))
    assert load.is_uncontrollable is False
    assert len([n for n in load.state_change_notifications if n[1] == DEVICE_STATUS_CHANGE_ERROR]) == 1

    # ...and it only crosses the threshold after the real 1050 s of evidence.
    await drive(load, time + timedelta(seconds=CYCLE_S), LADDER_WALL_S)
    assert load.is_uncontrollable is True
    assert len([n for n in load.state_change_notifications if n[1] == DEVICE_STATUS_CHANGE_ERROR]) == 2


async def test_relaunch_counters_never_outlive_the_command_they_describe():
    """At every cycle boundary, an empty command slot implies both counters are zero.

    Regression guard for a whole bug class rather than one instance.
    `abandon_running_command` deliberately preserves
    `running_command_num_relaunch` to keep the saturated supersede cadence, which
    made it possible to leave a spent rung behind on a path with no successor —
    and a later, unrelated command then crossed the lost-control threshold with
    zero relaunches of its own. Any NEW slot-emptying path that forgets to reset
    either counter fails here rather than in production.

    Two properties, and the second is the one that actually matters:

    - **P1 (cycle boundary)** empty slot => rung 0 and after-invalid 0. This is
      deliberately NOT asserted instantaneously: `launch_command`'s equal-command
      early-return leaves the slot empty with the rung intact until the next
      `_escalate_or_recover`. That transient is harmless because nothing reads the
      rung while the slot is empty — every reader is guarded on
      `running_command is not None`.
    - **P2 (the safety property)** a command in flight is NEVER simultaneously at
      or past the threshold AND un-escalated, unless it earned the rung itself.
      This holds because the rung reset and the re-arm are performed together in
      one block, so the transient in P1 can never be observed as "fresh command,
      spent rung, clock cleared" — which is precisely the shape of the bug.
    """
    load = NeverAcksLoad(name="pool_house")
    p1_violations: list[tuple[str, datetime, int, int]] = []
    p2_violations: list[tuple[str, datetime, int]] = []
    empty_slot_observations = 0
    earned_rung_cycles = 0

    def audit(stage: str, time: datetime) -> None:
        """Check both properties at a cycle boundary."""
        nonlocal empty_slot_observations, earned_rung_cycles
        if load.running_command is None:
            empty_slot_observations += 1
            if load.running_command_num_relaunch or load.running_command_num_relaunch_after_invalid:
                p1_violations.append(
                    (
                        stage,
                        time,
                        load.running_command_num_relaunch,
                        load.running_command_num_relaunch_after_invalid,
                    )
                )
            return

        if load.running_command_num_relaunch >= NUM_MAX_COMMAND_RELAUNCH:
            earned_rung_cycles += 1
            if load.unresponsive_since is None:
                p2_violations.append((stage, time, load.running_command_num_relaunch))

    async def audited_drive(stage: str, start: datetime, duration_s: float) -> datetime:
        time = start
        while time <= start + timedelta(seconds=duration_s):
            await load.check_and_relaunch_command(time)
            audit(stage, time)
            time = time + timedelta(seconds=CYCLE_S)
        return time

    # 1. the device is confirmed `on`, then refuses `idle` until we give up on it
    load._ack_command(T0 - timedelta(seconds=60), copy_command(CMD_ON))
    await load.launch_command(T0, CMD_IDLE)
    time = await audited_drive("ladder", T0 + timedelta(seconds=CYCLE_S), LADDER_WALL_S)
    assert load.running_command_num_relaunch >= NUM_MAX_COMMAND_RELAUNCH

    # 2. equal-command supersede: abandons with NO successor
    await load.launch_command(time, CMD_ON)
    time = await audited_drive("post-equal-supersede", time + timedelta(seconds=CYCLE_S), 3 * CYCLE_S)

    # 3. override-suppression drop
    load.suppress_override = True
    await load.launch_command(time, CMD_OFF)
    time = await audited_drive("override-drop", time, 2 * SUPERSEDE_MIN_INTERVAL_S)
    load.suppress_override = False

    # 4. differing-command supersede: abandons WITH a successor
    await load.launch_command(time, CMD_IDLE)
    time = await audited_drive("ladder-2", time + timedelta(seconds=CYCLE_S), LADDER_WALL_S)
    await load.launch_command(time, copy_command(CMD_ON, power_consign=99.0))
    time = await audited_drive("post-differing-supersede", time + timedelta(seconds=CYCLE_S), 3 * CYCLE_S)

    # 5. the probe goes unavailable: the invalid-probe give-up empties the slot too
    load.probe_result = None
    time = await audited_drive("give-up", time, CYCLE_S * (NUM_MAX_INVALID_PROBES_COMMANDS + 4))
    assert load.current_command is None

    # 6. and a real ack, the ordinary way out
    load.probe_result = True
    await load.launch_command(time, CMD_ON)
    time = await audited_drive("recovered", time + timedelta(seconds=CYCLE_S), 3 * CYCLE_S)

    assert p1_violations == []
    assert p2_violations == []
    # Guard against a vacuous pass: both properties must actually have been
    # exercised, in both of their interesting states.
    assert empty_slot_observations > 10
    assert earned_rung_cycles > 10


async def test_unreachable_entity_keeps_shouting_through_the_sensor():
    """A permanently unreachable entity must keep the PROBLEM sensor on, not go quiet.

    **Read this before "cleaning up" `unresponsive_since` — see #308.**

    When a load's probe returns `None` forever (the entity is unavailable), the
    `NUM_MAX_INVALID_PROBES_COMMANDS` give-up empties the command slot every ~70 s
    while `unresponsive_since` is deliberately kept. That lingering clock is what
    keeps `is_uncontrollable` True for the large majority of cycles, which is the
    correct user-facing answer for a permanently broken entity.

    It is currently achieved by an *ownerless* clock rather than by design (#308).
    The tempting tidy-up — clearing the clock at the give-up — silently drops the
    signal from ~88% of cycles to ~0%, because the give-up fires at ~70 s while the
    relaunch threshold needs 1050 s, so the clock could never be re-earned. This
    test exists so that regression fails loudly here instead of shipping.

    When #308 lands, rewrite this to assert the new `qs_load_unreachable` sensor.
    """
    load = NeverAcksLoad(name="broken_entity")
    load._ack_command(T0 - timedelta(seconds=60), copy_command(CMD_ON))
    await load.launch_command(T0, CMD_IDLE)

    # Earn the first escalation honestly.
    time = await drive(load, T0 + timedelta(seconds=CYCLE_S), LADDER_WALL_S)
    assert load.is_uncontrollable is True
    pushes_after_first_episode = len(load.state_change_notifications)
    assert pushes_after_first_episode == 1

    # The entity now goes permanently unavailable, while QS keeps commanding it.
    load.probe_result = None
    sensor_on = total = 0
    end = time + timedelta(seconds=6 * 3600)
    while time <= end:
        if load.running_command is None:
            await load.launch_command(time, CMD_IDLE if total % 2 else CMD_ON)
        await load.check_and_relaunch_command(time)
        total += 1
        sensor_on += 1 if load.is_uncontrollable else 0
        time = time + timedelta(seconds=CYCLE_S)

    # Measured at ~88%; the regression being guarded takes this to ~0%.
    assert sensor_on > total // 2, f"sensor only on for {sensor_on}/{total} cycles"
    # And it stays one push per episode — no per-cycle notification storm.
    assert len(load.state_change_notifications) == pushes_after_first_episode


async def test_unavailable_probe_give_up_is_not_a_recovery(caplog: pytest.LogCaptureFixture):
    """`_ack_command(time, None)` empties the slot, but the device is failing harder."""
    load = NeverAcksLoad(name="pool_house")
    time = await drive_until_uncontrollable(load)

    with caplog.at_level(logging.INFO):
        # The probe now returns None: the entity went unavailable.
        load.probe_result = None
        time = await drive(load, time, CYCLE_S * (NUM_MAX_INVALID_PROBES_COMMANDS + 2))

    assert load.current_command is None
    assert load.running_command is None
    assert load.unresponsive_since is not None
    assert _count(caplog, REGAINED_CONTROL_LOG) == 0


# =============================================================================
# AC6 — abandon without faking an ack
# =============================================================================


def test_abandon_preserves_current_command_clock_and_rung():
    """Abandoning keeps the confirmed truth, the lost-control clock and the rung."""
    load = NeverAcksLoad(name="pool_house")
    load.current_command = copy_command(CMD_ON, power_consign=1761.0)
    load.prev_command = copy_command(CMD_IDLE)
    load.num_on_off = 3
    load.running_command = copy_command(CMD_IDLE)
    load.running_command_num_relaunch = NUM_MAX_COMMAND_RELAUNCH
    load.running_command_num_relaunch_after_invalid = 2
    load.running_command_first_launch = T0
    load.running_command_last_launch = T0 + timedelta(seconds=10)
    load.unresponsive_since = T0
    load._last_supersede_time = T0
    load._stacked_command = copy_command(CMD_OFF)

    load.abandon_running_command(T0 + timedelta(seconds=20), reason="superseded")

    assert load.running_command is None
    assert load.running_command_num_relaunch_after_invalid == 0
    assert load.running_command_first_launch is None
    assert load.running_command_last_launch is None
    # The rung is PRESERVED: zeroing it would restart the ladder at 50 s after
    # every supersede and turn the 300 s cadence into a service-call storm.
    assert load.running_command_num_relaunch == NUM_MAX_COMMAND_RELAUNCH
    assert load.command_relaunch_delay_s() == float(SUPERSEDE_MIN_INTERVAL_S)
    # `current_command` means "last CONFIRMED command" and stays truthful.
    assert load.current_command == copy_command(CMD_ON, power_consign=1761.0)
    assert load.prev_command == CMD_IDLE
    assert load.num_on_off == 3
    assert load.unresponsive_since == T0
    assert load._last_supersede_time == T0
    assert load._stacked_command == CMD_OFF


# =============================================================================
# AC1 / AC9 — perpetual retry, one shout in
# =============================================================================


async def test_retry_is_perpetual(caplog: pytest.LogCaptureFixture):
    """Relaunches never stop, and they settle on the saturated 300 s cadence."""
    load = NeverAcksLoad(name="pool_house")

    with caplog.at_level(logging.ERROR):
        await load.launch_command(T0, CMD_IDLE)
        assert len(load.executed_commands) == 1

        time = await drive(load, T0 + timedelta(seconds=CYCLE_S), LADDER_WALL_S)
        # 1 original launch + the 6 ladder rungs.
        assert len(load.executed_commands) == 1 + NUM_MAX_COMMAND_RELAUNCH
        assert load.is_uncontrollable is True

        # Three more hours of a device that never obeys.
        at_threshold = len(load.executed_commands)
        await drive(load, time, 3 * 3600)

    extra = len(load.executed_commands) - at_threshold
    # 300 s cadence over 3 h, allowing cycle quantisation.
    assert 33 <= extra <= 36
    assert load.running_command is not None
    assert load.is_uncontrollable is True
    # One line in — not one per load-management cycle.
    assert _count(caplog, LOST_CONTROL_LOG) == 1


async def test_threshold_notifies_once():
    """Entry pushes exactly one notification, and superseding does not re-notify."""
    load = NeverAcksLoad(name="pool_house")
    time = await drive_until_uncontrollable(load)

    await load.launch_command(time, copy_command(CMD_ON, power_consign=1761.0))
    await drive(load, time + timedelta(seconds=CYCLE_S), 2 * SUPERSEDE_MIN_INTERVAL_S)

    errors = [n for n in load.state_change_notifications if n[1] == DEVICE_STATUS_CHANGE_ERROR]
    assert len(errors) == 1
    assert "pool_house" in errors[0][2]


async def test_notify_is_a_no_op_for_a_non_load_device(caplog: pytest.LogCaptureFixture):
    """A plain `AbstractDevice` (e.g. the battery) shouts in the log but has no push channel."""
    device = NeverAcksDevice(name="home_battery")

    with caplog.at_level(logging.ERROR):
        await device.launch_command(T0, CMD_IDLE)
        await drive(device, T0 + timedelta(seconds=CYCLE_S), LADDER_WALL_S)

    assert device.is_uncontrollable is True
    assert _count(caplog, LOST_CONTROL_LOG) == 1
    assert await device._notify_unresponsive(T0, CMD_IDLE) is None


# =============================================================================
# AC4 — supersession, two cases, one throttle
# =============================================================================


async def test_equal_command_makes_no_service_call_at_any_rung():
    """Re-requesting the very same command never produces a service call."""
    load = NeverAcksLoad(name="pool_house")
    await load.launch_command(T0, CMD_IDLE)

    for rung in range(NUM_MAX_COMMAND_RELAUNCH + 2):
        load.running_command_num_relaunch = rung
        before = len(load.executed_commands)
        load._stacked_command = copy_command(CMD_OFF)
        await load.launch_command(T0 + timedelta(seconds=rung), CMD_IDLE)
        assert len(load.executed_commands) == before
        assert load._stacked_command is None


async def test_differing_command_on_a_healthy_load_is_stacked():
    """Today's behaviour is untouched while QS still has control."""
    load = NeverAcksLoad(name="pool_house")
    await load.launch_command(T0, CMD_IDLE)
    before = len(load.executed_commands)

    await load.launch_command(T0 + timedelta(seconds=CYCLE_S), copy_command(CMD_ON, power_consign=1761.0))

    assert load.is_uncontrollable is False
    assert len(load.executed_commands) == before
    assert load._stacked_command == copy_command(CMD_ON, power_consign=1761.0)
    assert load.running_command == CMD_IDLE


async def test_supersede_throttled_across_jitter_sequence():
    """The real Cumulus Enfants jitter must produce ONE service call, not four."""
    load = NeverAcksLoad(name="cumulus_enfants")
    time = await drive_until_uncontrollable(load)
    before = len(load.executed_commands)

    jitter = [1761.0, 521.0, 534.0, 536.0]
    for index, consign in enumerate(jitter):
        await load.launch_command(
            time + timedelta(seconds=CYCLE_S * index), copy_command(CMD_ON, power_consign=consign)
        )

    assert len(load.executed_commands) - before == 1
    assert load.executed_commands[-1] == copy_command(CMD_ON, power_consign=1761.0)
    # Last-wins: the newest throttled command is what gets retried next window.
    assert load._stacked_command == copy_command(CMD_ON, power_consign=536.0)
    assert load.num_on_off == 0


async def test_supersede_cadence_survives_repeated_supersession():
    """After a supersede the rung is preserved, so the 300 s cadence holds."""
    load = NeverAcksLoad(name="cumulus_enfants")
    time = await drive_until_uncontrollable(load)

    await load.launch_command(time, copy_command(CMD_ON, power_consign=1761.0))
    after_first = len(load.executed_commands)
    assert load.running_command_num_relaunch == NUM_MAX_COMMAND_RELAUNCH

    # A whole throttle window minus one cycle: a reset rung would relaunch at
    # +50 / +150 / +300 s inside it. A preserved rung relaunches at none.
    await drive(load, time + timedelta(seconds=CYCLE_S), SUPERSEDE_MIN_INTERVAL_S - 2 * CYCLE_S)
    assert len(load.executed_commands) == after_first

    # A differing command inside the window is throttled too.
    await load.launch_command(
        time + timedelta(seconds=SUPERSEDE_MIN_INTERVAL_S - CYCLE_S), copy_command(CMD_ON, power_consign=521.0)
    )
    assert len(load.executed_commands) == after_first

    # Past the window, the next differing command supersedes for real.
    await load.launch_command(
        time + timedelta(seconds=SUPERSEDE_MIN_INTERVAL_S + CYCLE_S), copy_command(CMD_ON, power_consign=534.0)
    )
    assert len(load.executed_commands) == after_first + 1


async def test_supersede_needs_an_in_flight_command():
    """With an empty slot a command is launched immediately, exactly as today."""
    load = NeverAcksLoad(name="pool_house")
    assert load.running_command is None

    await load.launch_command(T0, CMD_IDLE)

    assert len(load.executed_commands) == 1
    assert load.running_command == CMD_IDLE


# =============================================================================
# AC5 — a raising device still climbs the ladder
# =============================================================================


async def test_raising_execute_still_reaches_the_threshold():
    """`execute_command` blowing up must not stall the ladder."""
    load = RaisingExecuteLoad(name="pool_house")
    await load.launch_command(T0, CMD_IDLE)
    assert load.running_command == CMD_IDLE

    await drive(load, T0 + timedelta(seconds=CYCLE_S), LADDER_WALL_S)

    assert load.running_command_num_relaunch >= NUM_MAX_COMMAND_RELAUNCH
    assert load.is_uncontrollable is True


async def test_raising_probe_still_reaches_the_threshold():
    """`probe_if_command_set` blowing up must not stall the ladder either."""
    load = RaisingProbeLoad(name="pool_house")
    load.running_command = copy_command(CMD_IDLE)
    load.running_command_first_launch = T0
    load.running_command_last_launch = T0

    time = T0 + timedelta(seconds=CYCLE_S)
    end = T0 + timedelta(seconds=LADDER_WALL_S + CYCLE_S)
    while time <= end:
        with pytest.raises(RuntimeError):
            await load.check_and_relaunch_command(time)
        time = time + timedelta(seconds=CYCLE_S)

    assert load.running_command_num_relaunch >= NUM_MAX_COMMAND_RELAUNCH
    assert load.is_uncontrollable is True


# =============================================================================
# AC13 — the Pool House incident, end to end
# =============================================================================


async def test_pool_house_incident_regression():
    """Replay 2026-07-27: constraint met, `idle` never acked, load kept commandable."""
    load = NeverAcksLoad(name="cumulus_pool_house")

    # 12:01 — `on` is launched and acked.
    boot = T0 - timedelta(minutes=11)
    load.probe_result = True
    await load.launch_command(boot, CMD_ON)
    assert load.current_command == CMD_ON

    # 12:12:19 — `idle` is dispatched and the probe never returns True again.
    load.probe_result = False
    await load.launch_command(T0, CMD_IDLE)

    # 12:23 — the solver wants `on` again while QS is still healthy: stacked.
    time = await drive(load, T0 + timedelta(seconds=CYCLE_S), 11 * 60)
    assert load.is_uncontrollable is False
    await load.launch_command(time, CMD_ON)
    assert load._stacked_command == CMD_ON
    assert load.running_command == CMD_IDLE

    # 12:30 — the ladder is spent. Before QS-304 this was the last service
    # call ever sent to the entity.
    time = await drive(load, time + timedelta(seconds=CYCLE_S), LADDER_WALL_S - 11 * 60)
    assert load.is_uncontrollable is True
    at_threshold = len(load.executed_commands)

    # 15:00 — the constraint completes and `idle` is asked for again. It is
    # absorbed by the equal-command branch: no stale `on` promotion.
    await load.launch_command(time, CMD_IDLE)
    assert load._stacked_command is None
    assert load.running_command == CMD_IDLE
    assert len(load.executed_commands) == at_threshold

    # 19:06 — still retrying, still shouting, and still commandable.
    await drive(load, time + timedelta(seconds=CYCLE_S), 4 * 3600)
    assert len(load.executed_commands) > at_threshold
    assert load.is_uncontrollable is True
    errors = [n for n in load.state_change_notifications if n[1] == DEVICE_STATUS_CHANGE_ERROR]
    assert len(errors) == 1


# =============================================================================
# AC14a / AC15 — the decision lives in the pure layer
# =============================================================================

HA_IMPORT_RE = re.compile(r"^\s*(from|import)\s+homeassistant", re.MULTILINE)


def test_relaunch_timing_is_decided_without_home_assistant():
    """AC14a: this very module proves the lifecycle without importing HA."""
    source = Path(__file__).read_text(encoding="utf-8")
    assert HA_IMPORT_RE.search(source) is None


def test_home_model_never_imports_home_assistant():
    """AC15: the domain layer stays pure Python."""
    package = Path(__file__).resolve().parents[1] / "custom_components" / "quiet_solar" / "home_model"
    modules = sorted(package.glob("*.py"))
    assert modules, "home_model package not found"

    offenders = [m.name for m in modules if HA_IMPORT_RE.search(m.read_text(encoding="utf-8"))]
    assert offenders == []


# =============================================================================
# Helpers
# =============================================================================


def _count(caplog: pytest.LogCaptureFixture, needle: str) -> int:
    """Count log records whose formatted message contains `needle`."""
    return len([r for r in caplog.records if needle in r.getMessage()])


async def _uncontrollable_load(name: str = "pool_house") -> NeverAcksLoad:
    """Return a load that has already crossed the uncontrollable threshold."""
    load = NeverAcksLoad(name=name)
    await drive_until_uncontrollable(load)
    return load


def test_load_command_equality_is_consign_sensitive():
    """The throttle exists because a consign jitter really is a new command."""
    assert copy_command(CMD_ON, power_consign=521.0) != copy_command(CMD_ON, power_consign=536.0)
    assert isinstance(CMD_ON, LoadCommand)
