"""QS-381 regression tests.

`QSChargerGroup.ensure_correct_state` returns no actionable charger while any member is
not in its expected state. QS-376 contained the start-stuck case; these tests pin the
other non-converging members:

- A1: a phase switch that never follows is adopted after its retries (bounded block);
- A2: an amps mismatch while charging keeps blocking by design, re-sending every cycle;
- A3: a requested reboot that never happens stops blocking after a timeout (and the
  reboot check is really awaited);
- A4: a start-stuck member stuck behind one of those checks is re-armed in bounded time.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.quiet_solar.const import (
    CONF_CHARGER_PAUSE_RESUME_SWITCH,
    CONF_CHARGER_THREE_TO_ONE_PHASE_SWITCH,
)
from custom_components.quiet_solar.ha_model import charger as charger_module
from custom_components.quiet_solar.ha_model.charger import (
    STATE_CMD_TIME_BETWEEN_RETRY_S,
    TIME_OK_BETWEEN_CHANGING_CHARGER_PHASES,
    QSChargerStatus,
)
from custom_components.quiet_solar.home_model.commands import CMD_AUTO_GREEN_ONLY, copy_command
from tests.test_bug_376_stuck_charger_group import (
    FOURTH_LAUNCH,
    STEP,
    T0,
    _build_stuck_charger,
    _make_healthy,
    _States,
)
from tests.test_charger_coverage_deep import (
    _create_charger,
    _create_ocpp_charger,
    _init_charger_states,
    _make_charger_group,
    _make_hass,
    _make_home,
    _make_real_car,
)

# 4 launches spaced > STATE_CMD_TIME_BETWEEN_RETRY_S (42 s) on the 7 s grid
PHASE_LAUNCH_OFFSETS_S = [0, 49, 98, 147]
# first 7 s cycle > 147 + 42 s
PHASE_ADOPT_OFFSET_S = 196
# longest wait for a requested reboot (QS-381)
REBOOT_TIMEOUT_S = 10 * 60


def _base_mocks(charger, amps: int = 10, charging: bool = True) -> None:
    """Plugged, available, auto command; the charge / amps readings are mocks."""
    charger._do_update_charger_state = AsyncMock()
    charger.is_charger_unavailable = MagicMock(return_value=False)
    charger.is_charger_faulted = MagicMock(return_value=False)
    charger.is_not_plugged = MagicMock(return_value=False)
    charger.running_command = None
    charger.current_command = copy_command(CMD_AUTO_GREEN_ONLY)
    charger.update_data_request = AsyncMock()
    charger.is_charge_enabled = MagicMock(return_value=charging)
    charger.is_charge_disabled = MagicMock(return_value=not charging)
    charger.get_charging_current = MagicMock(return_value=amps)
    charger.get_stable_dynamic_charge_status = MagicMock(return_value=QSChargerStatus(charger))


def _build_phase_charger(name="broken", switch_state="off", charging=True):
    """Real 3-phase charger with a 3->1 phase switch, expected on 1 phase."""
    hass = _make_hass()
    home = _make_home()
    home.async_notify_all_mobile_apps = AsyncMock()
    states = _States()
    hass.states.get = MagicMock(side_effect=states.get)
    phase_sw = f"switch.{name}_phase"
    charger = _create_charger(
        hass,
        home,
        name=name,
        is_3p=True,
        **{
            CONF_CHARGER_THREE_TO_ONE_PHASE_SWITCH: phase_sw,
            CONF_CHARGER_PAUSE_RESUME_SWITCH: f"switch.{name}_charge",
        },
    )
    car = _make_real_car(hass, home, name=f"{name} car")
    charger.attach_car(car, T0 - timedelta(hours=1))
    _init_charger_states(charger, charge_state=True, amperage=10, num_phases=1)
    _base_mocks(charger, charging=charging)
    states.set(phase_sw, switch_state, T0 - timedelta(hours=1))
    return hass, home, states, charger, phase_sw


def _calls_on(hass, entity_id: str, service: str | None = None) -> int:
    return sum(
        1
        for c in hass.services.async_call.await_args_list
        if (c.kwargs.get("target") or {}).get("entity_id") == entity_id
        and (service is None or c.kwargs.get("service") == service)
    )


def _off(t) -> int:
    return int((t - T0).total_seconds())


@pytest.mark.asyncio
@pytest.mark.parametrize("broken_first", [True, False])
async def test_phase_switch_never_converging_is_adopted(broken_first, caplog):
    """A1: after 4 launches + the retry delay, the observed phase count is adopted."""
    hass, home, _states, broken, phase_sw = _build_phase_charger()
    healthy, cs_healthy = _make_healthy(hass, home)
    group = _make_charger_group(home, [broken, healthy] if broken_first else [healthy, broken])
    assert broken.current_num_phases == 3  # switch "off" reads 3 phases, expected 1

    launches = []
    adopted_at = None
    t = T0
    with caplog.at_level(logging.WARNING):
        while t <= T0 + timedelta(minutes=10):
            before = _calls_on(hass, phase_sw, "turn_on")
            actionable, _ = await group.ensure_correct_state(t)
            if _calls_on(hass, phase_sw, "turn_on") > before:
                launches.append(_off(t))
            if adopted_at is None and broken._expected_num_active_phases.value == 3:
                adopted_at = t
            if adopted_at is None or t == adopted_at:
                assert actionable == [], f"group unblocked too early at {_off(t)}"
            else:
                assert cs_healthy in actionable, f"group still blocked at {_off(t)}"
            t += STEP

    assert launches == PHASE_LAUNCH_OFFSETS_S
    assert adopted_at is not None, "the observed phase count was never adopted"
    assert _off(adopted_at) == PHASE_ADOPT_OFFSET_S
    assert await broken._ensure_correct_state(adopted_at + STEP) is True
    assert "phase switch never converged" in caplog.text

    # the adoption restarts the 30 min phase-change spacing
    cmd = broken._expected_num_active_phases
    spacing = timedelta(seconds=TIME_OK_BETWEEN_CHANGING_CHARGER_PHASES)
    assert cmd.is_ok_to_set(adopted_at + spacing - STEP, TIME_OK_BETWEEN_CHANGING_CHARGER_PHASES) is False
    assert cmd.is_ok_to_set(adopted_at + spacing + STEP, TIME_OK_BETWEEN_CHANGING_CHARGER_PHASES) is True


@pytest.mark.asyncio
async def test_phase_mismatch_not_yet_due_is_not_adopted():
    """A1: exhausted retries whose last launch is still within its retry delay: no adoption."""
    _hass, _home, _states, broken, _sw = _build_phase_charger()
    cmd = broken._expected_num_active_phases
    for i in range(4):
        cmd.register_launch(1, T0 + timedelta(seconds=50 * i))
    t = T0 + timedelta(seconds=150 + STATE_CMD_TIME_BETWEEN_RETRY_S)
    assert cmd.can_launch() is False
    assert await broken._ensure_correct_state(t) is False
    assert cmd.value == 1
    # a probe never adopts
    assert await broken._ensure_correct_state(t + timedelta(minutes=5), probe_only=True) is False
    assert cmd.value == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reported", [16, None])
async def test_amps_mismatch_while_charging_keeps_blocking_and_resending(reported):
    """A2 (by design): a charging member whose set-point differs keeps blocking the group
    and gets its set-point re-sent on every cycle (amps retries are not limited)."""
    hass, home, _states, member, _sw = _build_phase_charger(name="amps", switch_state="on")
    member.get_charging_current = MagicMock(return_value=reported)
    healthy, cs_healthy = _make_healthy(hass, home)
    group = _make_charger_group(home, [member, healthy])

    t = T0
    while t <= T0 + timedelta(minutes=5):
        actionable, _ = await group.ensure_correct_state(t)
        assert actionable == [], f"group unblocked at {_off(t)} with {reported}A reported"
        t += STEP
    assert member._expected_amperage.can_launch() is True

    member.get_charging_current = MagicMock(return_value=10)
    actionable, _ = await group.ensure_correct_state(t)
    assert cs_healthy in actionable


@pytest.mark.asyncio
async def test_amps_mismatch_resends_every_cycle():
    """A2 (by design): the amps command is re-sent on every ensure cycle, never capped at 4."""
    _hass, _home, _states, member, _sw = _build_phase_charger(name="amps", switch_state="on")
    member.get_charging_current = MagicMock(return_value=16)
    member.set_charging_current = AsyncMock(return_value=True)
    t = T0
    cycles = 0
    while t <= T0 + timedelta(minutes=5):
        assert await member._ensure_correct_state(t) is False
        cycles += 1
        t += STEP
    assert member.set_charging_current.await_count == cycles


def _build_rebooting_ocpp(name="rebooter"):
    """Real OCPP charger with a reboot button, otherwise in its expected state (charging)."""
    hass = _make_hass()
    home = _make_home()
    states = _States()
    hass.states.get = MagicMock(side_effect=states.get)
    charger = _create_ocpp_charger(hass, home, name=name)
    car = _make_real_car(hass, home, name=f"{name} car")
    charger.attach_car(car, T0 - timedelta(hours=1))
    _init_charger_states(charger, charge_state=True, amperage=10, num_phases=1)
    _base_mocks(charger)
    charger.charger_reboot_button = f"button.{name}_reboot"
    return hass, home, states, charger


def test_reboot_wait_timeout_constant():
    """A3: the reboot wait is bounded at 10 min."""
    assert getattr(charger_module, "CHARGER_REBOOT_WAIT_TIMEOUT_S", None) == REBOOT_TIMEOUT_S


@pytest.mark.asyncio
async def test_reboot_check_is_awaited():
    """A3: `check_if_reboot_happened` is awaited: a reboot 30 s old is not done yet."""
    _hass, _home, _states, charger = _build_rebooting_ocpp()
    assert charger.can_reboot() is True
    await charger.reboot(T0)
    assert charger._asked_for_reboot_at_time == T0

    assert await charger._ensure_correct_state(T0 + timedelta(seconds=30)) is False
    assert charger._asked_for_reboot_at_time == T0


@pytest.mark.asyncio
async def test_reboot_never_happening_times_out(caplog):
    """A3: a reboot that never happens blocks the group for at most the timeout."""
    hass, home, _states, charger = _build_rebooting_ocpp()
    healthy, cs_healthy = _make_healthy(hass, home)
    group = _make_charger_group(home, [charger, healthy])
    await charger.reboot(T0)

    timeout = timedelta(seconds=REBOOT_TIMEOUT_S)
    t = T0 + STEP
    with caplog.at_level(logging.WARNING):
        while t <= T0 + timeout + timedelta(minutes=2):
            actionable, _ = await group.ensure_correct_state(t)
            if t < T0 + timeout:
                assert actionable == [], f"group unblocked before the timeout at {_off(t)}"
                assert charger._asked_for_reboot_at_time == T0
            else:
                assert cs_healthy in actionable, f"group still blocked at {_off(t)}"
                assert charger._asked_for_reboot_at_time is None
            t += STEP
    assert caplog.text.count("never happened, giving up the wait") == 1


@pytest.mark.asyncio
async def test_reboot_guard_times_out_without_car():
    """A3: the load's own reboot guard is bounded too, including with no car plugged."""
    _hass, _home, _states, charger = _build_rebooting_ocpp()
    await charger.reboot(T0)
    charger.detach_car()
    charger.is_not_plugged = MagicMock(return_value=True)
    charger._on_charger_fault_cycle = AsyncMock()

    timeout = timedelta(seconds=REBOOT_TIMEOUT_S)
    assert await charger.check_load_activity_and_constraints(T0 + timeout - STEP) is False
    charger._on_charger_fault_cycle.assert_not_awaited()
    assert charger._asked_for_reboot_at_time == T0

    await charger.check_load_activity_and_constraints(T0 + timeout)
    charger._on_charger_fault_cycle.assert_awaited()
    assert charger._asked_for_reboot_at_time is None


async def _drive_until_rearmed(group, stuck, healthy_cs, t_end, on_cycle=None):
    """Drive the group; return (4th start launch time, re-arm time) of the stuck member."""
    fourth = None
    rearmed_at = None
    t = T0
    while t <= t_end:
        if fourth is None and not stuck._expected_charge_state.can_launch():
            fourth = stuck._expected_charge_state.last_time_set
        if on_cycle is not None:
            await on_cycle(t, fourth)
        actionable, _ = await group.ensure_correct_state(t)
        if fourth is not None:
            assert healthy_cs in actionable, f"healthy starved at {_off(t)}"
        if rearmed_at is None and stuck._expected_charge_state.value is False:
            rearmed_at = t
        if rearmed_at is not None:
            assert group._isolated_reserved_amps == [0.0, 0.0, 0.0]
        t += STEP
    return fourth, rearmed_at


@pytest.mark.asyncio
async def test_start_stuck_with_phase_mismatch_rearms():
    """A4(a): a start-stuck member whose phase switch then stops following is re-armed."""
    hass, home, states, stuck, phase_sw = _build_phase_charger(name="stuck3p", switch_state="on", charging=False)
    stuck._expected_charge_state.set(False, T0 - timedelta(minutes=2))
    stuck._expected_charge_state.set(True, T0)
    healthy, cs_healthy = _make_healthy(hass, home)
    group = _make_charger_group(home, [stuck, healthy])

    adopted = {}

    async def on_cycle(t, fourth):
        if fourth is not None and t >= fourth + timedelta(minutes=14):
            if states.get(phase_sw).state != "unavailable":
                states.set(phase_sw, "unavailable", t)
        if "at" not in adopted and stuck._expected_num_active_phases.value == 3:
            adopted["at"] = t

    fourth, rearmed_at = await _drive_until_rearmed(
        group, stuck, cs_healthy, T0 + timedelta(minutes=30), on_cycle
    )

    assert _calls_on(hass, "switch.stuck3p_charge", "turn_on") == 4  # the 4 start launches
    assert fourth is not None
    assert "at" in adopted, "the observed phase count was never adopted"
    # the phase check, not the F2 threshold, was the last thing holding the re-arm
    assert adopted["at"] > fourth + timedelta(minutes=15)
    # the adoption is seen at the start of the cycle after it happened; the re-arm runs
    # in that same call, once the phases match
    assert rearmed_at == adopted["at"]


@pytest.mark.asyncio
async def test_start_stuck_with_pending_reboot_rearms():
    """A4(b): a start-stuck member waiting for a reboot that never happens is re-armed."""
    hass, home, _states, stuck = _build_stuck_charger(switch_state="off")
    stuck.charger_reboot_button = "button.wallbox_2_parking_reboot"
    healthy, cs_healthy = _make_healthy(hass, home)
    group = _make_charger_group(home, [stuck, healthy])

    reboot_at = FOURTH_LAUNCH + timedelta(minutes=10)
    timeout = timedelta(seconds=REBOOT_TIMEOUT_S)

    async def on_cycle(t, _fourth):
        if stuck._asked_for_reboot_at_time is None and reboot_at <= t < reboot_at + STEP:
            await stuck.reboot(t)

    fourth, rearmed_at = await _drive_until_rearmed(
        group, stuck, cs_healthy, T0 + timedelta(minutes=40), on_cycle
    )

    assert fourth == FOURTH_LAUNCH
    assert rearmed_at is not None
    assert rearmed_at >= reboot_at + timeout
    assert rearmed_at < reboot_at + timeout + 2 * STEP
