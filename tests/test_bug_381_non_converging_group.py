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
    QSChargerGeneric,
    QSChargerStatus,
)
from custom_components.quiet_solar.home_model.commands import (
    CMD_AUTO_FROM_CONSIGN,
    CMD_AUTO_GREEN_ONLY,
    copy_command,
)
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
    assert _calls_on(hass, phase_sw, "turn_on") == 4  # AC 1a: exactly 4, none after the adoption
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
    # one step past the delay, a non-probe cycle DOES adopt: pins both sides of the strict `>`
    assert await broken._ensure_correct_state(t + STEP) is False
    assert cmd.value == 3


async def _drive_to_adoption(group_or_charger, charger, t=T0):
    """Run ensure cycles until the observed phase count is adopted; return the next time."""
    while charger._expected_num_active_phases.value != 3:
        if group_or_charger is charger:
            await charger._ensure_correct_state(t)
        else:
            await group_or_charger.ensure_correct_state(t)
        t += STEP
        assert _off(t) <= 600, "the phase count was never adopted"
    return t


@pytest.mark.asyncio
async def test_late_phase_flip_after_adoption_is_not_reverted(caplog):
    """A1 / fix #01: once the observed count is adopted, a later flip of the switch to the
    requested phase follows the observed value: no reboot, no re-drive, group stays free."""
    hass, home, states, broken, phase_sw = _build_phase_charger()
    healthy, cs_healthy = _make_healthy(hass, home)
    group = _make_charger_group(home, [broken, healthy])
    broken.reboot = AsyncMock()

    t = await _drive_to_adoption(group, broken)
    adopted_at = t - STEP
    assert broken._phases_adopted_at == adopted_at
    phase_calls_at_adoption = _calls_on(hass, phase_sw)
    # spy on the re-drive entry point from the adoption cycle on
    broken.set_charging_num_phases = AsyncMock(wraps=broken.set_charging_num_phases)

    # the phase switch finally flips to the requested phase, one step after adoption
    states.set(phase_sw, "on", adopted_at + STEP)
    assert broken.current_num_phases == 1  # now reads 1 phase, adopted value was 3

    with caplog.at_level(logging.WARNING):
        caplog.clear()  # only look at what happens after the flip
        end = adopted_at + timedelta(minutes=5)
        while t <= end:
            await group.ensure_correct_state(t)
            t += STEP

    # the expected value followed the observed one: no re-drive, no new phase command, no reboot
    assert broken._expected_num_active_phases.value == 1
    broken.set_charging_num_phases.assert_not_awaited()
    assert _calls_on(hass, phase_sw) == phase_calls_at_adoption
    broken.reboot.assert_not_awaited()
    assert "never converged" not in caplog.text  # no second adoption warning
    # and the group is free again once the member follows
    actionable, _ = await group.ensure_correct_state(t)
    assert cs_healthy in actionable
    assert broken.get_stable_dynamic_charge_status.return_value in actionable


@pytest.mark.asyncio
async def test_new_budget_phase_request_after_adoption_still_launches():
    """A1 / fix #01: after adoption, a genuine new budget phase request clears the adoption
    state and launches the phase switch again."""
    hass, _home, _states, broken, phase_sw = _build_phase_charger()
    t = await _drive_to_adoption(broken, broken)
    assert broken._phases_adopted_at is not None
    assert broken._expected_num_active_phases.value == 3
    launches_at_adoption = _calls_on(hass, phase_sw, "turn_on")

    # the budget/constraint side asks for 1 phase again (the real request path in apply_budgets)
    broken.set_expected_num_active_phases(1, t)
    assert broken._phases_adopted_at is None  # a real request ends the adoption

    assert await broken._ensure_correct_state(t) is False
    assert _calls_on(hass, phase_sw, "turn_on") == launches_at_adoption + 1


@pytest.mark.asyncio
async def test_same_phase_request_after_adoption_keeps_adoption():
    """A1 / fix #01: re-asking for the already-adopted value is not a new request: the
    adoption marker stays so a late flip still follows the observed count."""
    _hass, _home, _states, broken, _phase_sw = _build_phase_charger()
    t = await _drive_to_adoption(broken, broken)
    marker = broken._phases_adopted_at
    assert marker is not None

    # re-asking for the adopted value (3) is a no-op: the adoption marker is kept
    broken.set_expected_num_active_phases(3, t)
    assert broken._phases_adopted_at == marker


@pytest.mark.asyncio
async def test_stale_replayed_budget_after_follow_does_not_revert_adoption():
    """A1 / fix #02 item 1: a split-budget cycle keeps a stale increasing snapshot in
    `remaining_budget_to_apply`. After the switch flips late and the follow sets the expected
    count to 1, replaying that stale 3-phase snapshot through `apply_budgets`
    (`check_charger_state=True`) must not re-drive the switch back to 3 or clear the adoption."""
    hass, home, states, broken, phase_sw = _build_phase_charger()
    healthy, _cs_healthy = _make_healthy(hass, home)
    group = _make_charger_group(home, [broken, healthy])

    # target was 1 phase, switch stuck "off" (reads 3): the observed 3 gets adopted
    t = await _drive_to_adoption(broken, broken)
    assert broken._expected_num_active_phases.value == 3
    assert broken._phases_adopted_at is not None

    # cycle N: the split kept a stale increasing snapshot for `broken` (budgeted 3 phases,
    # current 3 phases as measured at the split, before the switch ever followed)
    stale = broken.get_stable_dynamic_charge_status.return_value
    stale.current_real_max_charging_amp = 6
    stale.current_active_phase_number = 3
    stale.budgeted_amp = 6
    stale.budgeted_num_phases = 3
    # fix #05: staleness is now TIME-based. This snapshot was taken before the follow below
    # changes the observed phase count, so stamp it before any observation-side change.
    stale.snapshot_time = T0

    # cycle N+1: the switch finally flips on -> the follow sets the expected count to 1
    states.set(phase_sw, "on", t)
    await broken._ensure_correct_state(t)
    assert broken._expected_num_active_phases.value == 1
    assert broken._phases_adopted_at is not None

    broken.set_charging_num_phases = AsyncMock(wraps=broken.set_charging_num_phases)
    phase_calls_before = _calls_on(hass, phase_sw)

    # cycle N+2: replay the stale increasing snapshot (the real `remaining_budget_to_apply` path)
    await group.apply_budgets([stale], [stale], t, check_charger_state=True)

    # fix #03: the phase request now compares against the LIVE expected count (1), so the stale
    # 3-phase snapshot WOULD re-drive the switch — the replay-drop guard is what stops it. The
    # drop returns early (remaining_budget_to_apply cleared), so no re-drive, adoption kept.
    assert group.remaining_budget_to_apply is None  # the early return fired (drop guard)
    assert broken._expected_num_active_phases.value == 1
    assert broken._phases_adopted_at is not None
    broken.set_charging_num_phases.assert_not_awaited()
    assert _calls_on(hass, phase_sw) == phase_calls_before


@pytest.mark.asyncio
async def test_stale_replay_budgeted_phase_differs_is_stale():
    """A1 / fix #04 item 1: under a LIVE adoption, a replay whose BUDGETED phase count differs
    from the live expected value is stale too — not only one whose CURRENT count differs. The
    3->1 split applies 1 first (expected 1), the switch stays off so D1 adopts 3, then the
    original snapshot (current 3, budget 1) is replayed: current now equals the adopted 3 so the
    old check missed it, but the budget still asks for the pre-adoption 1 and would clear the
    adoption / re-drive the switch. It must be dropped."""
    hass, home, _states, broken, phase_sw = _build_phase_charger()
    healthy, _cs = _make_healthy(hass, home)
    group = _make_charger_group(home, [broken, healthy])
    t = await _drive_to_adoption(broken, broken)  # expected 3, switch "off" (reads 3)
    assert broken._expected_num_active_phases.value == 3
    assert broken._phases_adopted_at is not None

    # the replayed snapshot: its CURRENT count matches the adopted 3, but it still BUDGETS 1
    stale = broken.get_stable_dynamic_charge_status.return_value
    stale.current_real_max_charging_amp = 6
    stale.current_active_phase_number = 3  # equals the live expected
    stale.budgeted_amp = 6
    stale.budgeted_num_phases = 1  # but it budgets the pre-adoption 1
    # fix #05: TIME-based staleness — this snapshot predates the adoption's observation change
    stale.snapshot_time = T0

    assert broken.is_phase_snapshot_stale(stale) is True

    broken.set_charging_num_phases = AsyncMock(wraps=broken.set_charging_num_phases)
    phase_calls_before = _calls_on(hass, phase_sw)
    await group.apply_budgets([stale], [stale], t, check_charger_state=True)

    assert group.remaining_budget_to_apply is None  # the drop guard fired (early return)
    assert broken._expected_num_active_phases.value == 3  # adoption kept
    assert broken._phases_adopted_at is not None
    broken.set_charging_num_phases.assert_not_awaited()
    assert _calls_on(hass, phase_sw) == phase_calls_before


@pytest.mark.asyncio
async def test_fresh_post_adoption_split_replay_is_not_dropped():
    """A1 / fix #05 item 1: staleness is now TIME-based, not value-based. A FRESH split budget
    built AFTER an adoption is field-for-field identical to the stale one (current 3 / budget 1),
    but it must NOT be dropped — dropping the whole replay would starve an unrelated healthy
    charger's legitimate amps increase every cycle for up to 30 min. Replay the fresh snapshots
    through the real `apply_budgets(check_charger_state=True)` path (the same entry the group uses
    for `remaining_budget_to_apply`)."""
    hass, home, _states, broken, _sw = _build_phase_charger()
    healthy, cs_h = _make_healthy(hass, home)
    group = _make_charger_group(home, [broken, healthy])
    t = await _drive_to_adoption(broken, broken)
    assert broken._expected_num_active_phases.value == 3 and broken._phases_adopted_at is not None
    t += STEP

    # a FRESH post-adoption snapshot (current 3 / budget 1 — identical to the stale one). Its
    # snapshot_time is AFTER the adoption's observation change, so it is NOT stale. The value-based
    # check wrongly flagged it (budget 1 != expected 3) and dropped the whole replay.
    cs_b = broken.get_stable_dynamic_charge_status.return_value
    cs_b.current_real_max_charging_amp = 6
    cs_b.current_active_phase_number = 3
    cs_b.budgeted_amp = 6
    cs_b.budgeted_num_phases = 1
    cs_b.snapshot_time = t  # fresh, taken after the adoption
    assert broken.is_phase_snapshot_stale(cs_b) is False

    # an unrelated healthy charger with a legitimate amps increase (6 -> 10), also fresh
    healthy._expected_amperage.set(6, t)
    cs_h.current_real_max_charging_amp = 6
    cs_h.current_active_phase_number = 3
    cs_h.budgeted_amp = 10
    cs_h.budgeted_num_phases = 3
    cs_h.snapshot_time = t

    broken._ensure_correct_state = AsyncMock()
    healthy._ensure_correct_state = AsyncMock()
    group._is_current_acceptable = MagicMock(return_value=True)

    await group.apply_budgets([cs_b, cs_h], [cs_b, cs_h], t, check_charger_state=True)

    # the fresh replay is NOT dropped (no early return), and the healthy charger's increase lands
    assert group.remaining_budget_to_apply is not None
    assert healthy._expected_amperage.value == 10


@pytest.mark.asyncio
async def test_phantom_consign_offers_only_current_phase_count():
    """fix #05 item 2: `get_consign_amps_values` must not offer the other phase count on a phantom
    reading. A phantom switch can never succeed, so a consign that can't be met on the current
    phases keeps the current count and clamps the amps instead of asking for a switch. A
    real-reading consign is unchanged."""
    _hass, _home, states, ch, phase_sw = _build_phase_charger(name="consign", switch_state="off")
    cs = QSChargerStatus(ch)
    cs.current_active_phase_number = 1  # current/expected is 1 phase
    # a consign the 1-phase setup cannot meet -> the real path would switch to 3 phases
    cs.command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=15000)

    # real reading (switch `off` is a real state): the other phase count IS offered
    assert ch._has_real_phase_reading() is True
    phases_real, amp_real = cs.get_consign_amps_values(consign_is_minimum=True)
    assert phases_real == [3]
    assert ch.min_charge <= amp_real <= ch.max_charge

    # phantom reading (switch unavailable): only the current count, amps clamped to current phase
    states.set(phase_sw, "unavailable", T0)
    assert ch._has_real_phase_reading() is False
    phases_phantom, amp_phantom = cs.get_consign_amps_values(consign_is_minimum=True)
    assert phases_phantom == [cs.current_active_phase_number]
    assert ch.min_charge <= amp_phantom <= ch.max_charge


@pytest.mark.asyncio
async def test_reset_state_machine_clears_phase_markers():
    """fix #05 polish: `_reset_state_machine` clears both phase adoption markers."""
    _hass, _home, _states, broken, _sw = _build_phase_charger()
    await _drive_to_adoption(broken, broken)
    assert broken._phases_adopted_at is not None
    assert broken._phases_observed_change_at is not None
    broken._reset_state_machine()
    assert broken._phases_adopted_at is None
    assert broken._phases_observed_change_at is None


@pytest.mark.asyncio
async def test_d1_readoption_inside_live_window_keeps_original_stamp():
    """A1 / fix #04 item 2: after a follow, a phantom reading falls through to launch and D1
    adopts again INSIDE the live window. The re-adoption must NOT restart the window:
    `_phases_adopted_at` keeps the ORIGINAL adoption time, so the window still counts from the
    first adoption (fix #02's rule)."""
    _hass, _home, states, broken, phase_sw = _build_phase_charger()
    t = await _drive_to_adoption(broken, broken)  # expected 3, switch "off"
    adopted_at = broken._phases_adopted_at
    assert adopted_at is not None

    # a real reading flips on (1 phase) inside the window: followed -> expected 1, marker kept
    states.set(phase_sw, "on", t)
    await broken._ensure_correct_state(t)
    assert broken._expected_num_active_phases.value == 1
    assert broken._phases_adopted_at == adopted_at
    t += STEP

    # now it goes unavailable (phantom 3): it falls through to launch, and 4 launches later D1
    # re-adopts 3 — still inside the ORIGINAL 30 min window
    states.set(phase_sw, "unavailable", t)
    spacing = TIME_OK_BETWEEN_CHANGING_CHARGER_PHASES
    while broken._expected_num_active_phases.value != 3:
        await broken._ensure_correct_state(t)
        t += STEP
        assert (t - adopted_at).total_seconds() < spacing, "the re-adoption left the window"

    # the re-adoption kept the ORIGINAL stamp; it did not open a fresh window
    assert broken._phases_adopted_at == adopted_at


@pytest.mark.asyncio
async def test_real_budget_phase_change_through_apply_budgets_after_adoption_launches():
    """A1 / fix #02 item 1: a genuine phase change in a budget snapshot (budgeted != current)
    still goes through `set_expected_num_active_phases`, clears the adoption and launches."""
    hass, home, _states, broken, _phase_sw = _build_phase_charger()
    healthy, _cs_healthy = _make_healthy(hass, home)
    group = _make_charger_group(home, [broken, healthy])
    t = await _drive_to_adoption(broken, broken)  # expected 3, adopted, switch "off" (reads 3)
    assert broken._phases_adopted_at is not None
    broken.set_charging_num_phases = AsyncMock(wraps=broken.set_charging_num_phases)

    cs = broken.get_stable_dynamic_charge_status.return_value
    cs.current_real_max_charging_amp = 6
    cs.current_active_phase_number = 3  # observed/adopted 3
    cs.budgeted_amp = 6
    cs.budgeted_num_phases = 1  # a genuine request for 1 phase

    await group.apply_budgets([cs], [cs], t)

    assert broken._phases_adopted_at is None  # a real phase change ends the adoption
    broken.set_charging_num_phases.assert_awaited()
    assert broken.set_charging_num_phases.await_args.kwargs["num_phases"] == 1


@pytest.mark.asyncio
async def test_idle_path_expected_change_still_applies_budget_phase():
    """A1 / fix #03 (should-fix): apply_budgets must gate the phase request on the LIVE expected
    count, not the frozen snapshot. If the expected count changed underneath (e.g. the idle
    probe adopts the observed count) with NO adoption marker, a budget whose phase count equals
    the stale snapshot but differs from the live expected must still be applied — otherwise the
    charger gets amps budgeted for the other phase count."""
    hass, home, _states, broken, _sw = _build_phase_charger(switch_state="off")  # reads 3
    healthy, _cs = _make_healthy(hass, home)
    group = _make_charger_group(home, [broken, healthy])
    t = T0

    # the expected count changed underneath to 1, with NO adoption marker (idle-path style)
    broken._expected_num_active_phases.set(1, t)
    assert broken._phases_adopted_at is None

    cs = broken.get_stable_dynamic_charge_status.return_value
    cs.current_real_max_charging_amp = 6
    cs.current_active_phase_number = 3  # frozen snapshot value (stale)
    cs.budgeted_amp = 6
    cs.budgeted_num_phases = 3  # asks for 3 == snapshot init, but != live expected (1)
    broken.set_expected_num_active_phases = MagicMock(wraps=broken.set_expected_num_active_phases)

    await group.apply_budgets([cs], [cs], t)

    # applied against the live expected (1), not silently skipped by the frozen snapshot compare
    broken.set_expected_num_active_phases.assert_called_once_with(3, t)
    assert broken._expected_num_active_phases.value == 3


@pytest.mark.asyncio
async def test_flapping_switch_does_not_extend_follow_window():
    """A1 / fix #02 item 2: a follow must not refresh `_phases_adopted_at`. Otherwise a
    flapping switch keeps the 30 min window open forever and blocks the group one cycle per
    flap. The window counts from the original adoption."""
    _hass, _home, states, broken, phase_sw = _build_phase_charger()
    t = await _drive_to_adoption(broken, broken)
    adopted_at = t - STEP
    assert broken._phases_adopted_at == adopted_at
    spacing = timedelta(seconds=TIME_OK_BETWEEN_CHANGING_CHARGER_PHASES)

    # an intermediate flip well inside the window is followed, but must NOT refresh the marker
    mid = adopted_at + spacing // 2
    states.set(phase_sw, "on", mid)
    await broken._ensure_correct_state(mid)
    assert broken._expected_num_active_phases.value == 1
    assert broken._phases_adopted_at == adopted_at  # window still counts from the adoption

    # a later mismatch PAST the original window takes the normal launch path, not another follow
    late = adopted_at + spacing + STEP
    states.set(phase_sw, "off", late)  # reads 3 again, expected is 1
    broken.set_charging_num_phases = AsyncMock(wraps=broken.set_charging_num_phases)
    await broken._ensure_correct_state(late)
    broken.set_charging_num_phases.assert_awaited()  # re-driven, the follow window has expired
    assert broken._phases_adopted_at is None  # the expiry clear ran before the launch path


@pytest.mark.asyncio
@pytest.mark.parametrize("within_window", [True, False])
async def test_follow_window_edges(within_window):
    """A1 / fix #02 item 4: a flip at adoption + 30 min - 1 step is still followed; a flip at
    adoption + 30 min (or later) has left the window and takes the normal launch path."""
    _hass, _home, states, broken, phase_sw = _build_phase_charger()
    t = await _drive_to_adoption(broken, broken)
    adopted_at = t - STEP
    spacing = timedelta(seconds=TIME_OK_BETWEEN_CHANGING_CHARGER_PHASES)
    flip_at = adopted_at + spacing - STEP if within_window else adopted_at + spacing

    states.set(phase_sw, "on", flip_at)  # reads 1, expected is still the adopted 3
    broken.set_charging_num_phases = AsyncMock(wraps=broken.set_charging_num_phases)
    await broken._ensure_correct_state(flip_at)

    if within_window:
        # followed: the expected count tracks the observed one, no re-drive
        assert broken._expected_num_active_phases.value == 1
        broken.set_charging_num_phases.assert_not_awaited()
    else:
        # window expired: the mismatch is re-driven back toward the still-expected 3
        assert broken._expected_num_active_phases.value == 3
        broken.set_charging_num_phases.assert_awaited()
        assert broken._phases_adopted_at is None  # the expiry clear ran at the window edge


@pytest.mark.asyncio
async def test_unavailable_phase_switch_within_window_falls_through_to_launch():
    """A1 / fix #03: an unavailable phase switch reads a phantom 3. Within the follow window it
    is NOT *followed* (that would lock in a phantom the charger never reported), but it must not
    block the group forever either: the phantom falls through to the normal launch path, which
    then drives / adopts in bounded time. The gate is now the follow branch only."""
    _hass, _home, states, broken, phase_sw = _build_phase_charger()
    t = await _drive_to_adoption(broken, broken)  # expected 3, switch "off"
    adopted_at = t - STEP

    # the switch genuinely flips on -> follow -> expected 1
    states.set(phase_sw, "on", adopted_at + STEP)
    await broken._ensure_correct_state(adopted_at + STEP)
    assert broken._expected_num_active_phases.value == 1

    # now it goes unavailable within the window: current_num_phases reads a phantom 3
    states.set(phase_sw, "unavailable", adopted_at + 2 * STEP)
    assert broken.current_num_phases == 3
    broken.set_charging_num_phases = AsyncMock(wraps=broken.set_charging_num_phases)
    assert await broken._ensure_correct_state(adopted_at + 2 * STEP) is False
    # the phantom is NOT followed back to 3 (expected stays 1)...
    assert broken._expected_num_active_phases.value == 1
    # ...but the normal launch path IS taken: the phantom does not block the group forever
    broken.set_charging_num_phases.assert_awaited()


@pytest.mark.asyncio
async def test_unavailable_phase_switch_within_window_unblocks_group_in_bounded_time(caplog):
    """A1 / fix #04 item 3: the within-window phantom fall-through must also UNBLOCK the group in
    bounded time, not merely launch. After a follow to 1, a phantom 3 inside the window is
    re-launched and then adopted, so the group frees the healthy member within ~196 s."""
    hass, home, states, broken, phase_sw = _build_phase_charger()
    healthy, cs_healthy = _make_healthy(hass, home)
    group = _make_charger_group(home, [broken, healthy])
    t = await _drive_to_adoption(group, broken)  # expected 3, switch "off"

    # a real reading flips on (1 phase) inside the window: followed -> expected 1
    states.set(phase_sw, "on", t)
    await group.ensure_correct_state(t)
    assert broken._expected_num_active_phases.value == 1
    t += STEP

    # now it goes permanently unavailable (phantom 3) inside the window
    states.set(phase_sw, "unavailable", t)
    phantom_start = t
    unblocked_at = None
    with caplog.at_level(logging.WARNING):
        while t <= phantom_start + timedelta(minutes=10):
            actionable, _ = await group.ensure_correct_state(t)
            if unblocked_at is None and cs_healthy in actionable:
                unblocked_at = t
            t += STEP
    assert unblocked_at is not None, "the within-window phantom blocked the group forever"
    assert (unblocked_at - phantom_start).total_seconds() <= PHASE_ADOPT_OFFSET_S + 2 * STEP.total_seconds()
    assert broken._expected_num_active_phases.value == 3
    assert "phase switch never converged" in caplog.text


@pytest.mark.asyncio
async def test_unavailable_phase_switch_past_delay_is_adopted():
    """A1 / fix #03 (must-fix): an unavailable phase switch reads a phantom 3. It is never
    *followed*, but past the launch retries it IS adopted (bounded block), exactly like a real
    `off` reading. The story says adopting the phantom 3 is safe (it over-counts per-phase
    current, which is conservative)."""
    _hass, _home, states, broken, phase_sw = _build_phase_charger()
    states.set(phase_sw, "unavailable", T0)
    cmd = broken._expected_num_active_phases
    for i in range(4):
        cmd.register_launch(1, T0 + timedelta(seconds=50 * i))
    t = T0 + timedelta(seconds=150 + STATE_CMD_TIME_BETWEEN_RETRY_S + STEP.total_seconds())
    assert cmd.can_launch() is False
    assert broken.current_num_phases == 3  # phantom read
    assert await broken._ensure_correct_state(t) is False
    assert cmd.value == 3  # adopted: the phantom is safe to adopt once the retries run out
    assert broken._phases_adopted_at == t  # marker set so a real reading is then followed


@pytest.mark.asyncio
async def test_phantom_adoption_then_real_reading_is_followed():
    """A1 / fix #03: a permanently-unavailable switch (phantom 3) is adopted after the retries;
    when a real `on` (1 phase) returns inside the window it is *followed* with no switch command
    and no reboot (the adoption marker set on the phantom adoption makes that possible)."""
    hass, _home, states, broken, phase_sw = _build_phase_charger()
    states.set(phase_sw, "unavailable", T0)
    broken.reboot = AsyncMock()
    t = await _drive_to_adoption(broken, broken)  # phantom 3 adopted
    adopted_at = t - STEP
    assert broken._expected_num_active_phases.value == 3
    assert broken._phases_adopted_at == adopted_at
    phase_calls = _calls_on(hass, phase_sw)

    # a real reading returns inside the window: the switch is really on 1 phase
    states.set(phase_sw, "on", adopted_at + STEP)
    assert broken.current_num_phases == 1
    broken.set_charging_num_phases = AsyncMock(wraps=broken.set_charging_num_phases)
    await broken._ensure_correct_state(adopted_at + STEP)

    assert broken._expected_num_active_phases.value == 1  # followed, no switch command
    broken.set_charging_num_phases.assert_not_awaited()
    assert _calls_on(hass, phase_sw) == phase_calls
    broken.reboot.assert_not_awaited()


@pytest.mark.asyncio
async def test_permanently_unavailable_switch_unblocks_group_in_bounded_time(caplog):
    """MUST-FIX (fix #03): a permanently unavailable phase switch with 1 phase expected must NOT
    block the whole budgeting group forever. The phantom 3 is adopted after the launch retries
    and the group is unblocked in bounded time."""
    hass, home, states, broken, phase_sw = _build_phase_charger()
    states.set(phase_sw, "unavailable", T0 - timedelta(hours=1))
    assert broken.current_num_phases == 3  # phantom (unavailable reads 3), expected 1
    healthy, cs_healthy = _make_healthy(hass, home)
    group = _make_charger_group(home, [broken, healthy])

    unblocked_at = None
    t = T0
    with caplog.at_level(logging.WARNING):
        while t <= T0 + timedelta(minutes=10):
            actionable, _ = await group.ensure_correct_state(t)
            if unblocked_at is None and cs_healthy in actionable:
                unblocked_at = t
            t += STEP
    assert unblocked_at is not None, "the unavailable switch blocked the group forever"
    assert _off(unblocked_at) <= PHASE_ADOPT_OFFSET_S + 2 * int(STEP.total_seconds())
    assert broken._expected_num_active_phases.value == 3
    assert "phase switch never converged" in caplog.text


@pytest.mark.asyncio
async def test_phantom_reading_offers_only_current_phase_count():
    """A1 / fix #04 item 4: `get_stable_dynamic_charge_status` must not offer `[1, 3]` to a phase
    switch with no real reading. A dead switch would otherwise be asked for 1 phase again once
    the 30 min spacing expires, repeating the adoption cycle (4 launches, ~196 s group block) and
    a re-adoption every 30 min without end. A real reading still offers both counts."""
    hass, home, states, broken, phase_sw = _build_phase_charger(switch_state="on")  # real, reads 1
    # restore the real method (the fixture mocks it) and let the phase-change spacing pass
    broken.get_stable_dynamic_charge_status = QSChargerGeneric.get_stable_dynamic_charge_status.__get__(broken)
    broken._expected_num_active_phases.set(1, T0 - timedelta(hours=1))

    # a real reading -> the phase switch is offered both phase counts once the spacing allows
    cs = broken.get_stable_dynamic_charge_status(T0)
    assert cs is not None
    assert cs.possible_num_phases == [1, 3]

    # a phantom reading (switch unavailable) -> only the current count, no new switch request
    states.set(phase_sw, "unavailable", T0)
    assert broken._has_real_phase_reading() is False
    cs = broken.get_stable_dynamic_charge_status(T0)
    assert cs is not None
    assert cs.possible_num_phases == [cs.current_active_phase_number]

    # a charger with no phase switch always has a real reading, so it is unaffected by the gate
    plain = _create_charger(hass, home, name="plain")
    assert plain._has_real_phase_reading() is True


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
    # the group returns the healthy member AND the recovering member itself (AC2c): the
    # recovering member must not be silently dropped from the budget on the way back.
    assert cs_healthy in actionable
    assert member.get_stable_dynamic_charge_status.return_value in actionable


@pytest.mark.asyncio
@pytest.mark.parametrize("reported", [16, None])
async def test_amps_mismatch_resends_every_cycle(reported):
    """A2 (by design): the amps command is re-sent on every ensure cycle, never capped at 4,
    both when the charger over-reports (16 A) and when it reports nothing (None)."""
    _hass, _home, _states, member, _sw = _build_phase_charger(name="amps", switch_state="on")
    member.get_charging_current = MagicMock(return_value=reported)
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
    # AC3: hold the status connector at `Finishing` so a rebooting status is never seen and the
    # reboot check only clears on the 10 min timeout (not on a phantom rebooting reading).
    states.set(f"sensor.{name}_status_connector", "Finishing", T0 - timedelta(hours=1))
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
async def test_reboot_give_up_runs_in_probe_mode():
    """A3: clearing a stale reboot wait is not a command, so the give-up fires in probe mode too
    (like the reboot-done branch), past the timeout."""
    _hass, _home, _states, charger = _build_rebooting_ocpp()
    await charger.reboot(T0)
    assert charger._asked_for_reboot_at_time == T0

    await charger._ensure_correct_state(T0 + timedelta(seconds=REBOOT_TIMEOUT_S), probe_only=True)
    assert charger._asked_for_reboot_at_time is None


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
@pytest.mark.parametrize("phase_read", ["unavailable", "off"])
async def test_start_stuck_with_phase_mismatch_rearms(phase_read):
    """A4(a): a start-stuck member whose phase switch then stops following is re-armed.

    AC 4(a) uses an `unavailable` switch (a phantom 3). fix #03: the phantom no longer blocks
    the group forever — it is adopted after the launch retries, exactly like a real `off`
    reading — so the re-arm still fires. The real `off` variant is kept as a second case."""
    hass, home, states, stuck, phase_sw = _build_phase_charger(name="stuck3p", switch_state="on", charging=False)
    stuck._expected_charge_state.set(False, T0 - timedelta(minutes=2))
    stuck._expected_charge_state.set(True, T0)
    healthy, cs_healthy = _make_healthy(hass, home)
    group = _make_charger_group(home, [stuck, healthy])

    adopted = {}

    async def on_cycle(t, fourth):
        # the phase switch stops following the expected 1 back: either it goes `unavailable`
        # (a phantom 3, AC 4(a)) or it physically flips to 3 phases (a real `off` reading).
        # Both are adopted after the launch retries, resolving the mismatch in bounded time.
        if fourth is not None and t >= fourth + timedelta(minutes=14):
            if states.get(phase_sw).state != phase_read:
                states.set(phase_sw, phase_read, t)
        if "at" not in adopted and stuck._expected_num_active_phases.value == 3:
            adopted["at"] = t

    fourth, rearmed_at = await _drive_until_rearmed(group, stuck, cs_healthy, T0 + timedelta(minutes=30), on_cycle)

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

    fourth, rearmed_at = await _drive_until_rearmed(group, stuck, cs_healthy, T0 + timedelta(minutes=40), on_cycle)

    assert fourth == FOURTH_LAUNCH
    assert rearmed_at is not None
    assert rearmed_at >= reboot_at + timeout
    assert rearmed_at < reboot_at + timeout + 2 * STEP
