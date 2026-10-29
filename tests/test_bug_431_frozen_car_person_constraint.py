"""QS-431 — a user-frozen car gets no person constraint; the person ASAP no longer
oscillates on a target marker.

Incident (2026-10-23, Zoe): a person ASAP constraint was rewritten to the car's
``charge_target_percent`` marker (25 %, at or below the 33 % SOC) by the #48 escape,
silently dropped as met, then re-pushed by the person block the next cycle — a 14-s
push/drop loop and a notification storm. A ~7-min secondary storm came from the
person block rewriting the target on sub-percent drift.

Real ``QSChargerGeneric`` + real ``QSCar`` on the public charger harness, same
plugged-charger mocks as ``test_qs352_person_target_leak._base_fixture``.

The freeze flag is read as the string literal ``"user_frozen"`` (no new import, no
``is_user_frozen()`` call) so the guard tests are meaningfully green before the fix.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz

from custom_components.quiet_solar.const import (
    CONSTRAINT_FORECASTED_PERSON_KEY,
    CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
    CONSTRAINT_TYPE_FILLER,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
)
from custom_components.quiet_solar.home_model.constraints import (
    DATETIME_MAX_UTC,
    MultiStepsPowerLoadConstraintChargePercent,
)
from tests.utils.charger_harness import (
    create_charger,
    init_charger_states,
    make_hass,
    make_home,
    make_real_car,
    plug_car,
)

PERSON_TARGET = 44.05502122842387  # the incident's person need for Thomas
STEP = timedelta(seconds=7)
PERIOD = timedelta(minutes=15)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _car_cts(charger):
    """Live constraints belonging to the charger's car."""
    return [c for c in charger._constraints if c is not None and c.load_param == charger.car.name]


def _person_cts(charger):
    """Car constraints tagged with a forecasted person."""
    return [
        c
        for c in _car_cts(charger)
        if c.load_info is not None and c.load_info.get(CONSTRAINT_FORECASTED_PERSON_KEY) is not None
    ]


def _non_person_cts(charger):
    """Car constraints with no forecasted-person tag."""
    return [
        c
        for c in _car_cts(charger)
        if c.load_info is None or c.load_info.get(CONSTRAINT_FORECASTED_PERSON_KEY) is None
    ]


def _fixture(soc=33.0):
    """Real Zoe (default 100 %) plugged into a real charger, mocked HA I/O only."""
    hass = make_hass()
    home = make_home()
    charger = create_charger(hass, home)
    car = make_real_car(hass, home, name="Zoe", default_charge=100.0, minimum_ok_charge=20.0)
    now = datetime.now(pytz.UTC)

    init_charger_states(charger)
    charger.is_charger_unavailable = MagicMock(return_value=False)
    charger.probe_for_possible_needed_reboot = MagicMock(return_value=False)
    charger.is_not_plugged = MagicMock(return_value=False)
    charger.is_plugged = MagicMock(return_value=True)
    charger.is_car_stopped_asking_current = MagicMock(return_value=False)
    charger.set_charging_num_phases = AsyncMock(return_value=False)
    charger.set_max_charging_current = AsyncMock(return_value=True)
    charger.reboot = AsyncMock()

    plug_car(charger, car, now)
    charger.get_best_car = MagicMock(return_value=car)

    state = {"soc": soc}
    car.get_car_charge_percent = lambda time=None, *a, **kw: state["soc"]
    car.get_next_scheduled_event = AsyncMock(return_value=(None, None))
    car.do_next_charge_time = None
    car.do_force_next_charge = False

    thomas = MagicMock()
    thomas.name = "Thomas Menguy"
    thomas.notify_of_forecast_if_needed = AsyncMock()

    return charger, car, now, thomas, state


def _set_need(car, deadline, target, person):
    car.get_best_person_next_need = AsyncMock(return_value=(False, deadline, target, person))
    car.current_forecasted_person = person


async def _cycle(charger, t):
    """One solver cycle; the hash is read after ``update_live_constraints`` at the same t."""
    await charger.check_load_activity_and_constraints(t)
    await charger.update_live_constraints(t, PERIOD)
    return charger.get_active_state_hash(t)


def _count_changes(hashes):
    return sum(1 for a, b in zip(hashes, hashes[1:], strict=False) if a != b)


# --------------------------------------------------------------------------- #
# Fix A′ — mechanism
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_replayed_marker_does_not_rewrite_person_asap():
    """A boot-replayed 25 % marker must not lower an unfrozen person ASAP (the loop)."""
    charger, car, now, thomas, _ = _fixture()

    # boot replay BEFORE cycle 1: no live constraint, so no auto->manual conversion
    await car.user_set_next_charge_target(25, for_init=True)
    assert car.get_user_originated("charge_time") is None
    assert car.get_user_originated("user_frozen") is None

    _set_need(car, now + timedelta(seconds=20), PERSON_TARGET, thomas)

    t = now
    baseline = None
    hashes = []
    for _ in range(10):
        h = await _cycle(charger, t)
        if baseline is None and _person_cts(charger):
            baseline = h
        if baseline is not None:
            hashes.append(h)
        if t > now + timedelta(seconds=20):
            person = _person_cts(charger)
            assert len(person) == 1
            assert person[0].type == CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE
            assert person[0].target_value == pytest.approx(PERSON_TARGET)
        t += STEP

    assert baseline is not None
    assert _count_changes(hashes) <= 1


# --------------------------------------------------------------------------- #
# Fix C — product rule
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_user_frozen_car_gets_no_person_constraint():
    """The incident: a genuine 25 % user target freezes the car — no person constraint."""
    charger, car, now, thomas, _ = _fixture()

    await car.user_set_next_charge_target(25)
    _set_need(car, now + timedelta(seconds=20), PERSON_TARGET, thomas)

    t = now
    hashes = []
    for _ in range(10):
        hashes.append(await _cycle(charger, t))
        assert _person_cts(charger) == []
        t += STEP

    assert _count_changes(hashes) == 0
    # scope guard: the need is still computed (drives allocation / notifications)
    car.get_best_person_next_need.assert_awaited()


@pytest.mark.asyncio
async def test_user_action_freezes_and_removes_live_person_constraint():
    """A user action (bump solar) on a car with a live person constraint removes it."""
    charger, car, now, thomas, _ = _fixture()
    deadline = now + timedelta(hours=6)
    _set_need(car, deadline, PERSON_TARGET, thomas)

    t = now
    await _cycle(charger, t)
    assert len(_person_cts(charger)) == 1  # precondition: not frozen, pushed

    car.set_qs_bump_solar_charge_priority(True)
    # scope check (QS-353 D1): the visible target was captured, int-cast
    assert car.get_car_target_SOC() == 44

    t += STEP
    await _cycle(charger, t)
    assert _person_cts(charger) == []
    replacement = _non_person_cts(charger)
    assert len(replacement) == 1
    assert replacement[0].type == CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN
    assert replacement[0].initial_value == pytest.approx(33.0)
    assert replacement[0].target_value == pytest.approx(44.0)
    # the rebuilt chain starts from the real SOC: the old filler's stale start (the
    # person target, 44.055) must not be carried into it (it would be met at push)
    assert replacement[0].current_value == pytest.approx(33.0)
    assert charger._last_completed_constraint is None


@pytest.mark.asyncio
async def test_person_not_repushed_after_forced_session_on_frozen_car():
    """After a "charge now" session completes, the person constraint is not re-pushed."""
    charger, car, now, thomas, state = _fixture()
    deadline = now + timedelta(hours=6)
    _set_need(car, deadline, PERSON_TARGET, thomas)

    t = now
    await _cycle(charger, t)
    assert len(_person_cts(charger)) == 1  # precondition

    await car.user_force_charge_now()
    t += STEP
    await _cycle(charger, t)
    asap = [c for c in _car_cts(charger) if c.as_fast_as_possible]
    assert len(asap) == 1
    force_ct = asap[0]
    assert force_ct.target_value == pytest.approx(44.0)
    assert _person_cts(charger) == []

    # complete it explicitly: the harness never sets current_command, so the SOC
    # callback returns None and keeps current_value (constraints.py update)
    force_ct.current_value = 50.0
    state["soc"] = 50.0
    t += STEP
    await charger.update_live_constraints(t, PERIOD)
    assert not [c for c in _car_cts(charger) if c.as_fast_as_possible or c.from_user]
    assert charger._last_completed_constraint is force_ct

    _set_need(car, deadline, 60.0, thomas)
    for _ in range(5):
        t += STEP
        await _cycle(charger, t)
        assert _person_cts(charger) == []


@pytest.mark.asyncio
async def test_restart_replay_does_not_freeze():
    """Guard: an HA restart replay is not a user action — no freeze, person pushed."""
    charger, car, now, thomas, _ = _fixture()

    await car.user_set_next_charge_target(100, for_init=True)
    car.set_qs_bump_solar_charge_priority(False, for_init=True)
    assert car.get_user_originated("user_frozen") is None

    _set_need(car, now + timedelta(hours=6), PERSON_TARGET, thomas)
    await _cycle(charger, now)
    person = _person_cts(charger)
    assert len(person) == 1
    assert person[0].target_value == pytest.approx(PERSON_TARGET)


# --------------------------------------------------------------------------- #
# Fix B — secondary
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_person_target_sub_percent_drift_does_not_rewrite():
    """Sub-percent drift of the person need must not rewrite the target (same display)."""
    charger, car, now, thomas, _ = _fixture()
    deadline = now + timedelta(hours=6)

    _set_need(car, deadline, 44.00, thomas)
    t = now
    baseline = await _cycle(charger, t)
    person = _person_cts(charger)
    assert len(person) == 1
    assert person[0].target_value == pytest.approx(44.00)

    for drift in (44.04, 44.055):
        _set_need(car, deadline, drift, thomas)
        t += STEP
        h = await _cycle(charger, t)
        person = _person_cts(charger)
        assert len(person) == 1
        assert person[0].target_value == pytest.approx(44.00)
        assert h == baseline

    _set_need(car, deadline, 44.6, thomas)
    t += STEP
    h = await _cycle(charger, t)
    person = _person_cts(charger)
    assert len(person) == 1
    assert person[0].target_value == pytest.approx(44.6)
    assert h != baseline


# --------------------------------------------------------------------------- #
# guards — ordering carve-out and the refined A′ condition
# --------------------------------------------------------------------------- #
async def _reach_person_asap(charger, car, now, thomas):
    """Run cycles until the person constraint is promoted to ASAP; return next t."""
    _set_need(car, now + timedelta(seconds=20), PERSON_TARGET, thomas)
    t = now
    for _ in range(6):
        await _cycle(charger, t)
        t += STEP
    person = _person_cts(charger)
    assert len(person) == 1
    assert person[0].type == CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE
    assert person[0].target_value == pytest.approx(PERSON_TARGET)
    return t


@pytest.mark.asyncio
async def test_freeze_while_person_asap_live_runs_to_completion():
    """Guard: a freeze while a person ASAP is live retargets it to the user marker, no loop."""
    charger, car, now, thomas, _ = _fixture()
    t = await _reach_person_asap(charger, car, now, thomas)

    car.set_qs_bump_solar_charge_priority(True)  # D1 captures marker 44
    assert car.get_user_originated("charge_target_percent") == 44

    hashes = []
    for _ in range(5):
        hashes.append(await _cycle(charger, t))
        person = _person_cts(charger)
        assert len(person) == 1
        assert person[0].type == CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE
        assert person[0].target_value == pytest.approx(44.0)
        t += STEP

    assert _count_changes(hashes) <= 1


@pytest.mark.asyncio
async def test_frozen_without_marker_keeps_person_asap_target():
    """Guard: a frozen car with no target marker keeps the person ASAP target (not 100)."""
    charger, car, now, thomas, _ = _fixture()
    t = await _reach_person_asap(charger, car, now, thomas)

    # the state `user_set_person_for_car` leaves: pin + freeze, target markers cleared
    car._user_originated = {"person_name": "Thomas Menguy", "user_frozen": True}
    car._reset_charge_targets()

    for _ in range(3):
        await _cycle(charger, t)
        person = _person_cts(charger)
        assert len(person) == 1
        assert person[0].target_value == pytest.approx(PERSON_TARGET)
        t += STEP


# --------------------------------------------------------------------------- #
# Generic: the filler is artificial — tagged, and dropped + rebuilt from the real
# chain whenever another constraint of the car is touched in the cycle (QS-431)
# --------------------------------------------------------------------------- #
FILLER_TAG = {"originator": "filler"}


def _fillers(charger):
    """Car constraints with no deadline that are not user-originated (the best-effort chain)."""
    return [c for c in _non_person_cts(charger) if not c.from_user and c.end_of_constraint == DATETIME_MAX_UTC]


@pytest.mark.asyncio
async def test_filler_is_tagged():
    """The best-effort filler carries an explicit filler tag."""
    charger, car, now, thomas, _ = _fixture()
    _set_need(car, now + timedelta(hours=6), PERSON_TARGET, thomas)

    await _cycle(charger, now)

    fillers = _fillers(charger)
    assert len(fillers) == 1
    assert fillers[0].load_info == FILLER_TAG


@pytest.mark.asyncio
async def test_filler_rebuilt_from_real_soc_when_person_covered():
    """Person covered -> its constraint is removed: the filler chained behind it (start =
    the person target) must be rebuilt from the real SOC, not kept with a stale start."""
    charger, car, now, thomas, _ = _fixture()
    deadline = now + timedelta(hours=6)
    _set_need(car, deadline, PERSON_TARGET, thomas)

    t = now
    await _cycle(charger, t)
    assert len(_person_cts(charger)) == 1
    assert _fillers(charger)[0].initial_value == pytest.approx(PERSON_TARGET)

    car.get_best_person_next_need = AsyncMock(return_value=(True, deadline, PERSON_TARGET, thomas))
    t += STEP
    await _cycle(charger, t)

    assert _person_cts(charger) == []
    fillers = _fillers(charger)
    assert len(fillers) == 1
    assert fillers[0].initial_value == pytest.approx(33.0)
    assert fillers[0].current_value == pytest.approx(33.0)
    assert fillers[0].target_value == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_filler_follows_person_asap_retarget():
    """A frozen car's person ASAP is retargeted to the user marker (44.055 -> 44): the
    filler behind it must start from the new target, not the stale 44.055."""
    charger, car, now, thomas, _ = _fixture()
    t = await _reach_person_asap(charger, car, now, thomas)
    assert _fillers(charger)[0].initial_value == pytest.approx(PERSON_TARGET)

    car.set_qs_bump_solar_charge_priority(True)  # D1 captures marker 44
    await _cycle(charger, t)

    fillers = _fillers(charger)
    assert len(fillers) == 1
    assert fillers[0].initial_value == pytest.approx(44.0)


@pytest.mark.asyncio
async def test_filler_untouched_when_nothing_changes():
    """Guard: steady cycles keep the very same filler object and the same hash."""
    charger, car, now, thomas, _ = _fixture()
    _set_need(car, now + timedelta(hours=6), PERSON_TARGET, thomas)

    t = now
    baseline = await _cycle(charger, t)
    filler = _fillers(charger)[0]
    for _ in range(5):
        t += STEP
        assert await _cycle(charger, t) == baseline
        fillers = _fillers(charger)
        assert len(fillers) == 1
        assert fillers[0] is filler


@pytest.mark.asyncio
async def test_legacy_untagged_filler_replaced_on_change():
    """A filler persisted before QS-431 (no tag) is dropped like a tagged one when the
    chain changes: exactly one, tagged, filler remains."""
    charger, car, now, thomas, _ = _fixture()
    legacy = MultiStepsPowerLoadConstraintChargePercent(
        total_capacity_wh=car.car_battery_capacity,
        type=CONSTRAINT_TYPE_FILLER,
        time=now - timedelta(hours=1),
        load=charger,
        load_param=car.name,
        from_user=False,
        initial_value=50.0,
        target_value=100.0,
        power_steps=charger._power_steps,
        support_auto=True,
    )
    charger.push_live_constraint(now - timedelta(hours=1), legacy)
    assert legacy.load_info is None

    _set_need(car, now + timedelta(hours=6), PERSON_TARGET, thomas)
    await _cycle(charger, now)

    assert len(_person_cts(charger)) == 1
    fillers = _fillers(charger)
    assert len(fillers) == 1
    assert fillers[0] is not legacy
    assert fillers[0].load_info == FILLER_TAG
    assert fillers[0].initial_value == pytest.approx(PERSON_TARGET)


# --------------------------------------------------------------------------- #
# Review fix #01 — the charger-card reset unfreezes the car, the filler survives
# an identical-shape rebuild, the rounded compare has hysteresis
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_charger_card_reset_unfreezes_car():
    """S2: "charge now" from the charger card freezes the car; the charger-card reset
    must unfreeze it, so the person constraint is pushed again."""
    charger, car, now, thomas, _ = _fixture()
    deadline = now + timedelta(hours=6)
    _set_need(car, deadline, PERSON_TARGET, thomas)

    t = now
    await _cycle(charger, t)
    assert len(_person_cts(charger)) == 1  # precondition: not frozen, pushed

    await charger.user_force_charge_now()  # charger-card "charge now"
    assert car.get_user_originated("user_frozen") is True

    await charger.user_clean_and_reset()  # charger-card reset
    assert car.get_user_originated("user_frozen") is None

    t += STEP
    plug_car(charger, car, t)
    await _cycle(charger, t)
    assert len(_person_cts(charger)) == 1


@pytest.mark.asyncio
async def test_filler_kept_when_agenda_rebuilt_with_same_shape():
    """N1: the agenda constraint is rebuilt each cycle with the live SOC as its start;
    a replaced-but-identical-shape constraint must keep the very same filler."""
    charger, car, now, thomas, state = _fixture()
    await car.user_set_next_charge_target(80, for_init=True)  # agenda target < default
    car.get_next_scheduled_event = AsyncMock(return_value=(now + timedelta(hours=5), now + timedelta(hours=6)))
    _set_need(car, None, None, None)

    t = now
    await _cycle(charger, t)
    agenda = [c for c in _non_person_cts(charger) if c.end_of_constraint != DATETIME_MAX_UTC]
    assert len(agenda) == 1  # precondition: the agenda constraint is live
    fillers = _fillers(charger)
    assert len(fillers) == 1
    filler = fillers[0]

    for _ in range(6):
        state["soc"] += 1.0
        t += STEP
        await _cycle(charger, t)
        fillers = _fillers(charger)
        assert len(fillers) == 1
        assert fillers[0] is filler


@pytest.mark.asyncio
async def test_person_target_rounding_jitter_does_not_rewrite():
    """N2: a person need jittering around a .5 boundary must not flip the target on
    each poll (downward half-percent hysteresis); a genuine move still rewrites.

    Starts above the boundary: an upward crossing always rewrites (R2-N2), so the
    jitter exercises the downward hysteresis — at most one flip per crossing."""
    charger, car, now, thomas, _ = _fixture()
    deadline = now + timedelta(hours=6)

    _set_need(car, deadline, 44.51, thomas)
    t = now
    baseline = await _cycle(charger, t)
    assert _person_cts(charger)[0].target_value == pytest.approx(44.51)

    for jitter in (44.49, 44.48, 44.52):
        _set_need(car, deadline, jitter, thomas)
        t += STEP
        h = await _cycle(charger, t)
        person = _person_cts(charger)
        assert len(person) == 1
        assert person[0].target_value == pytest.approx(44.51)
        assert h == baseline

    _set_need(car, deadline, 43.9, thomas)
    t += STEP
    h = await _cycle(charger, t)
    person = _person_cts(charger)
    assert len(person) == 1
    assert person[0].target_value == pytest.approx(43.9)
    assert h != baseline


# --------------------------------------------------------------------------- #
# Review fix #02 — the charger-card reset mirrors the car-card reset; an upward
# person move always rewrites
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_charger_card_reset_mirrors_car_card_reset():
    """R2-N1: the charger-card reset also drops the car's forecasted person, system hold
    and charge targets, and re-runs the person allocation, like the car-card reset."""
    charger, car, now, thomas, _ = _fixture()
    _set_need(car, now + timedelta(hours=6), PERSON_TARGET, thomas)
    await _cycle(charger, now)
    assert len(_person_cts(charger)) == 1  # precondition: person constraint live

    await car.user_set_next_charge_target(80)  # user: 80 %, freezes the car
    car._system_person_hold_name = "Thomas Menguy"
    car._system_person_hold_until = now + timedelta(hours=1)
    charger.home.compute_and_set_best_persons_cars_allocations.reset_mock()

    await charger.user_clean_and_reset()  # charger-card reset
    assert car.get_user_originated("user_frozen") is None
    assert car.current_forecasted_person is None
    assert car._system_person_hold_name is None
    assert car.get_car_target_SOC() == pytest.approx(car.car_default_charge)
    charger.home.compute_and_set_best_persons_cars_allocations.assert_awaited_once_with(
        force_update=True, do_notify=True
    )


@pytest.mark.asyncio
async def test_car_card_reset_runs_the_allocation_once():
    """R2-N1 guard: the car-card reset drives the charger reset; the car-side steps
    (and the notifying allocation) must not run twice."""
    charger, car, now, thomas, _ = _fixture()
    charger.home.compute_and_set_best_persons_cars_allocations.reset_mock()
    await car.user_clean_and_reset()
    charger.home.compute_and_set_best_persons_cars_allocations.assert_awaited_once_with(
        force_update=True, do_notify=True
    )


@pytest.mark.asyncio
async def test_person_target_upward_move_rewrites():
    """R2-N2: a need that rises past a rounded % rewrites the person target at once (no
    rounded-target lag below the rounded need), even when it moves by less than half a
    percent."""
    charger, car, now, thomas, _ = _fixture()
    deadline = now + timedelta(hours=6)

    _set_need(car, deadline, 80.4, thomas)
    t = now
    await _cycle(charger, t)
    assert _person_cts(charger)[0].target_value == pytest.approx(80.4)

    _set_need(car, deadline, 80.6, thomas)
    t += STEP
    await _cycle(charger, t)
    person = _person_cts(charger)
    assert len(person) == 1
    assert person[0].target_value == pytest.approx(80.6)


# --------------------------------------------------------------------------- #
# Review fix #04 — no boot-time car-store wipe (user decision, documented
# limitation): the boot guard only lifts the QS-416 pins; the freeze survives a
# restart, whatever the car's cloud plug sensor reports at boot
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_boot_guard_stale_off_keeps_frozen_car_store():
    """R4-M1: a plain restart, the car never left its (plugged) charger, and the car's
    cloud plug sensor reports a stale "off" around the boot: the freeze and the user's
    target marker survive (no car-store wipe at boot — an unplug while HA is down
    does not end the freeze, the next live unplug or a reset does)."""
    charger, car, now, thomas, _ = _fixture()
    saved_at = now - timedelta(minutes=30)

    # HA restart: the persisted stores are restored (no user action in this process),
    # the car is not attached at boot, the persisted charger pin arms the guard
    charger.detach_car()
    car._user_originated = {"charge_target_percent": 80, "user_frozen": True}
    charger._user_originated["car_name"] = car.name
    charger._boot_user_pin = (car.name, saved_at, "on")
    car.car_plugged = "binary_sensor.zoe_plugged"
    car.attach_ha_state_to_probe(car.car_plugged, is_numerical=False)
    car._entity_probed_state[car.car_plugged] = []
    car._add_state_history(car.car_plugged, "off", now - timedelta(seconds=3), None, {})  # stale boot poll
    del charger.get_best_car  # the real car selection (the charger stays plugged)
    charger.home._chargers = [charger]

    t = now
    await charger.check_load_activity_and_constraints(t)
    for state in ["off", "off", "on", "on"]:  # stale "off" re-polled >= 15 s, then fresh "on"
        t += STEP
        car._add_state_history(car.car_plugged, state, t, None, {})
        await charger.check_load_activity_and_constraints(t)

    assert charger._boot_user_pin is None  # the QS-416 guard fired: pin lift, unchanged
    assert car.get_user_originated("user_frozen") is True
    assert car.get_user_originated("charge_target_percent") == 80
