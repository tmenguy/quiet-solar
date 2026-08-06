"""QS-342 — deterministic car↔charger allocation tie-break (regression net).

Replays the 2026-08-04/05 field incident: ID.buzz on wallbox 2 "parking" and
Zoe on wallbox 3 "portail", with the Zoe physically parked between the two
wallboxes. The verified tracker/charger coordinates produce an EXACT 3-way
score tie (`dist_bump` 0.5 m buckets), which the pre-fix greedy loop resolved
with a caller-wins bias: infinite attach/detach ping-pong on the Zoe and the
ID.buzz never attached anywhere.

Fixture helpers are imported from `tests.utils.charger_harness` (the shared
copy of the `tests/test_charger_coverage_deep.py` module-level helpers the
story points at).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz

from custom_components.quiet_solar.const import (
    CHARGER_NO_CAR_CONNECTED,
    CONF_CHARGER_LATITUDE,
    CONF_CHARGER_LONGITUDE,
    FORCE_CAR_NO_CHARGER_CONNECTED,
    USER_ORIGINATED_CAR_NAME,
    USER_ORIGINATED_CHARGER_NAME,
)
from custom_components.quiet_solar.ha_model.charger import (
    _LOG_VALUES_PER_WINDOW,
    _RELOG_UNCHANGED_AFTER_S,
)
from tests.utils.charger_harness import (
    create_charger,
    make_hass,
    make_home,
    make_real_car,
)

LOAD_LOGGER = "custom_components.quiet_solar.home_model.load"

# Per key the helper emits at most `budget` lines per window plus one overflow
# disclosure — see `LogOnChangeMixin.log_info_on_change`.
_PER_KEY_CEILING = _LOG_VALUES_PER_WINDOW + 1

CHARGER_LOGGER = "custom_components.quiet_solar.ha_model.charger"

T0 = datetime(2026, 8, 4, 17, 0, 0, tzinfo=pytz.UTC)

# Verified against the production recorder DB (story, Problem section).
WALLBOX_PARKING_COORDS = (43.635211, 6.989175)  # wallbox 2 "parking"
WALLBOX_PORTAIL_COORDS = (43.635238, 6.989261)  # wallbox 3 "portail"
ZOE_COORDS = (43.6352197222222, 6.98922277777778)  # 3.97 m / 3.69 m
IDBUZZ_COORDS = (43.635236, 6.989147)  # 3.58 m / 9.18 m

# The fixed point the incident matrix must converge to (via regret).
FIXED_POINT = {"wallbox_parking": "IDBuzz", "wallbox_portail": "Zoe"}


def _pin_car(car, coords) -> None:
    """Pin a car to the incident matrix: fixed GPS + `is_car_plugged` → True.

    Patching `is_car_plugged` covers both call forms (`for_duration` set and
    unset), so `plug_bump` is 5 rather than the instant-fallback 2.
    """
    car.get_car_coordinates = MagicMock(return_value=coords)
    car.is_car_plugged = MagicMock(return_value=True)


def _make_incident_fixture(pin_buzz: bool = True) -> SimpleNamespace:
    """Two plugged chargers + two cars at the verified incident coordinates.

    `get_continuous_plug_duration` stays unmocked (returns None → -1 →
    `plug_time_bump == 0`, as in the field after an HA restart).
    """
    hass = make_hass()
    home = make_home()
    parking = create_charger(
        hass,
        home,
        name="wallbox_parking",
        **{
            CONF_CHARGER_LATITUDE: WALLBOX_PARKING_COORDS[0],
            CONF_CHARGER_LONGITUDE: WALLBOX_PARKING_COORDS[1],
        },
    )
    portail = create_charger(
        hass,
        home,
        name="wallbox_portail",
        **{
            CONF_CHARGER_LATITUDE: WALLBOX_PORTAIL_COORDS[0],
            CONF_CHARGER_LONGITUDE: WALLBOX_PORTAIL_COORDS[1],
        },
    )
    zoe = make_real_car(hass, home, name="Zoe")
    buzz = make_real_car(hass, home, name="IDBuzz")

    _pin_car(zoe, ZOE_COORDS)
    if pin_buzz:
        _pin_car(buzz, IDBUZZ_COORDS)

    for charger in (parking, portail):
        charger.is_plugged = MagicMock(return_value=True)
        # Preconditions from the story: no boot car (it would preempt the
        # generic fallback) and no user-originated car selection.
        assert charger._boot_car is None
        assert charger.get_user_originated(USER_ORIGINATED_CAR_NAME) is None

    return SimpleNamespace(
        hass=hass,
        home=home,
        parking=parking,
        portail=portail,
        zoe=zoe,
        buzz=buzz,
        chargers={"wallbox_parking": parking, "wallbox_portail": portail},
        cars={"Zoe": zoe, "IDBuzz": buzz},
    )


def _run_allocation_round(chargers, time) -> None:
    """One full allocation round: each charger picks then applies its own row.

    Per the story recipe: attach only when the returned car actually changed.
    """
    for charger in chargers:
        car = charger.get_best_car(time)
        if car is not None and charger.car is not car:
            charger.attach_car(car, time)


def _allocation(fx) -> dict:
    return {
        name: (charger.car.name if charger.car is not None else None)
        for name, charger in fx.chargers.items()
    }


def _spy_detach(charger) -> list:
    """Wrap `detach_car` so post-convergence oscillation is observable.

    Forwards any arguments: a real oscillation must surface as a non-empty
    `calls` list, never as a `TypeError` inside the spy.
    """
    original = charger.detach_car
    calls: list[str] = []

    def spy(*args, **kwargs):
        calls.append(charger.name)
        original(*args, **kwargs)

    charger.detach_car = spy
    return calls


# Attachment states of the 2×2 incident fixture: each charger's car in
# {None, Zoe, IDBuzz}, minus double-attachment of the same car.
_ATTACHMENT_STATES = [
    (parking_car, portail_car)
    for parking_car in (None, "Zoe", "IDBuzz")
    for portail_car in (None, "Zoe", "IDBuzz")
    if parking_car is None or parking_car != portail_car
]

_EXECUTION_ORDERS = [
    ("wallbox_parking", "wallbox_portail"),
    ("wallbox_portail", "wallbox_parking"),
]


def _apply_attachment_state(fx, parking_car, portail_car, time) -> None:
    for charger_name, car_name in (
        ("wallbox_parking", parking_car),
        ("wallbox_portail", portail_car),
    ):
        if car_name is not None:
            # Within the last 20 min: the attach-duration score bumps must NOT
            # kick in — the tests exercise the new tie-break, not the old bumps.
            fx.chargers[charger_name].attach_car(fx.cars[car_name], time - timedelta(seconds=60))


# =============================================================================
# Incident replay (historical TDD note: was red on pre-A1 code, green since A1)
# =============================================================================


def test_incident_replay_idbuzz_gets_parking():
    fx = _make_incident_fixture()
    order = [fx.parking, fx.portail]
    time = T0

    _run_allocation_round(order, time)

    # The regret tie-break resolves the 3-way tie: ID.buzz→parking, Zoe→portail.
    assert _allocation(fx) == FIXED_POINT

    # Convergence: the next round changes nothing…
    time += timedelta(seconds=7)
    _run_allocation_round(order, time)
    assert _allocation(fx) == FIXED_POINT

    # …and zero detach_car calls over ≥3 further rounds (no ping-pong).
    detach_calls = [_spy_detach(fx.parking), _spy_detach(fx.portail)]
    for _ in range(3):
        time += timedelta(seconds=7)
        _run_allocation_round(order, time)
        assert _allocation(fx) == FIXED_POINT
    assert detach_calls == [[], []]


# =============================================================================
# Start-state × execution-order sweep (historical TDD note: was red pre-A1)
# =============================================================================


@pytest.mark.parametrize("order_names", _EXECUTION_ORDERS, ids=lambda o: "->".join(o))
@pytest.mark.parametrize(
    ("parking_car", "portail_car"),
    _ATTACHMENT_STATES,
    ids=lambda v: str(v),
)
def test_convergence_from_all_attachment_states(parking_car, portail_car, order_names):
    fx = _make_incident_fixture()
    _apply_attachment_state(fx, parking_car, portail_car, T0)
    order = [fx.chargers[name] for name in order_names]
    time = T0

    # Converges to the same fixed point within ONE full round, whatever the
    # starting attachment state (including the crossed pairing inherited from
    # the pre-fix ping-pong) and whatever the execution order.
    _run_allocation_round(order, time)
    assert _allocation(fx) == FIXED_POINT

    time += timedelta(seconds=7)
    _run_allocation_round(order, time)
    assert _allocation(fx) == FIXED_POINT

    detach_calls = [_spy_detach(fx.parking), _spy_detach(fx.portail)]
    for _ in range(3):
        time += timedelta(seconds=7)
        _run_allocation_round(order, time)
        assert _allocation(fx) == FIXED_POINT
    assert detach_calls == [[], []]


# =============================================================================
# Regret tie-break (selection-level, patched score maps)
# =============================================================================


def _patch_scores(charger, mapping) -> None:
    """Replace `get_car_score` with a fixed map (0.0 = excluded from the pool)."""
    charger.get_car_score = MagicMock(side_effect=lambda car, time, cache: mapping.get(car.name, 0.0))


def test_regret_prefers_charger_specific_car():
    # Incident matrix (real scores): ID.buzz's alternative on portail is
    # strictly worse (8 200 005 < 9 300 005) → regret 1 100 000 beats the
    # Zoe's regret 0 → ID.buzz wins the contested parking charger.
    fx = _make_incident_fixture()
    assert fx.parking.get_best_car(T0) is fx.buzz
    assert fx.portail.get_best_car(T0) is fx.zoe

    # No-alternative case: the car is absent from every other unassigned
    # charger's list → its alternative is 0 → regret = its own score.
    fx = _make_incident_fixture()
    _patch_scores(fx.parking, {"Zoe": 100.0, "IDBuzz": 100.0})
    _patch_scores(fx.portail, {"Zoe": 100.0})  # IDBuzz scores 0 on portail
    assert fx.parking.get_best_car(T0) is fx.buzz
    assert fx.portail.get_best_car(T0) is fx.zoe

    # All-equal case (equal scores, equal regrets, no attachment): falls
    # through to stable (charger name, car name) order — parking < portail,
    # IDBuzz < Zoe → (parking, IDBuzz) first, then (portail, Zoe).
    fx = _make_incident_fixture()
    _patch_scores(fx.parking, {"Zoe": 100.0, "IDBuzz": 100.0})
    _patch_scores(fx.portail, {"Zoe": 100.0, "IDBuzz": 100.0})
    assert (fx.parking.get_best_car(T0), fx.portail.get_best_car(T0)) == (fx.buzz, fx.zoe)


# =============================================================================
# Stickiness and steal semantics (AC3)
# =============================================================================


def test_equal_score_does_not_steal_attached_car():
    # Symmetric tie, equal regrets: the incumbent keeps the car.
    fx = _make_incident_fixture()
    _patch_scores(fx.parking, {"Zoe": 100.0, "IDBuzz": 100.0})
    _patch_scores(fx.portail, {"Zoe": 100.0, "IDBuzz": 100.0})
    fx.portail.attach_car(fx.zoe, T0 - timedelta(seconds=60))

    assert fx.portail.get_best_car(T0) is fx.zoe
    assert fx.parking.get_best_car(T0) is fx.buzz
    # The equal-score, equal-regret competitor never stole the Zoe.
    assert fx.portail.car is fx.zoe

    # A strictly higher score DOES steal an attached car.
    fx = _make_incident_fixture()
    _patch_scores(fx.parking, {"Zoe": 100.0, "IDBuzz": 100.0})
    _patch_scores(fx.portail, {"Zoe": 200.0})
    fx.parking.attach_car(fx.zoe, T0 - timedelta(seconds=60))
    assert fx.portail.get_best_car(T0) is fx.zoe  # stolen from parking
    assert fx.parking.car is None  # get_best_car detached it from the incumbent

    # An equal score with strictly higher regret DOES steal (crossed-pairing
    # recovery, real incident scores): the Zoe sits on parking but ID.buzz's
    # regret there is strictly higher → ID.buzz evicts it.
    fx = _make_incident_fixture()
    fx.parking.attach_car(fx.zoe, T0 - timedelta(seconds=60))
    fx.portail.attach_car(fx.buzz, T0 - timedelta(seconds=60))
    assert fx.parking.get_best_car(T0) is fx.buzz
    assert fx.portail.get_best_car(T0) is fx.zoe


# =============================================================================
# Generic-car fallback: a losing plugged charger is never orphaned (AC4)
# =============================================================================


def test_losing_charger_falls_back_to_generic_car():
    # Incident matrix but the second car scores 0 (not plugged, no coords, not
    # home): the Zoe ties on both chargers, stable order gives it to parking,
    # and the losing portail must fall back to its own generic car.
    fx = _make_incident_fixture(pin_buzz=False)
    order = [fx.parking, fx.portail]
    time = T0

    best = fx.portail.get_best_car(time)
    assert best is fx.portail._default_generic_car

    _run_allocation_round(order, time)
    assert fx.parking.car is fx.zoe
    assert fx.portail.car is fx.portail._default_generic_car

    # One more round: the generic attachment is stable (the generic car is not
    # in home._cars, so it can never become a steal candidate) and no plugged
    # charger ends the round with car is None.
    time += timedelta(seconds=7)
    _run_allocation_round(order, time)
    assert fx.parking.car is fx.zoe
    assert fx.portail.car is fx.portail._default_generic_car
    assert all(charger.car is not None for charger in order)


# =============================================================================
# Caller independence (AC1)
# =============================================================================


@pytest.mark.parametrize("caller_name", ["wallbox_parking", "wallbox_portail"])
@pytest.mark.parametrize(
    ("parking_car", "portail_car"),
    _ATTACHMENT_STATES,
    ids=lambda v: str(v),
)
def test_allocation_is_caller_independent(parking_car, portail_car, caller_name):
    # Given identical attachment state, every charger's own computed row agrees
    # with the same global allocation, whichever charger runs the computation.
    fx = _make_incident_fixture()
    _apply_attachment_state(fx, parking_car, portail_car, T0)
    caller = fx.chargers[caller_name]
    assert caller.get_best_car(T0) is fx.cars[FIXED_POINT[caller_name]]


def _setup_scenario(fx, scenario):
    """Set up one scenario family; return the expected global allocation."""
    if scenario == "regret_no_alternative":
        _patch_scores(fx.parking, {"Zoe": 100.0, "IDBuzz": 100.0})
        _patch_scores(fx.portail, {"Zoe": 100.0})  # IDBuzz has no alternative
        return {"wallbox_parking": fx.buzz, "wallbox_portail": fx.zoe}
    if scenario == "all_equal_stable_order":
        _patch_scores(fx.parking, {"Zoe": 100.0, "IDBuzz": 100.0})
        _patch_scores(fx.portail, {"Zoe": 100.0, "IDBuzz": 100.0})
        return {"wallbox_parking": fx.buzz, "wallbox_portail": fx.zoe}
    if scenario == "stickiness":
        _patch_scores(fx.parking, {"Zoe": 100.0, "IDBuzz": 100.0})
        _patch_scores(fx.portail, {"Zoe": 100.0, "IDBuzz": 100.0})
        fx.portail.attach_car(fx.zoe, T0 - timedelta(seconds=60))
        return {"wallbox_parking": fx.buzz, "wallbox_portail": fx.zoe}
    assert scenario == "generic_fallback"  # real incident scores, Zoe only
    return {"wallbox_parking": fx.zoe, "wallbox_portail": fx.portail._default_generic_car}


@pytest.mark.parametrize("caller_name", ["wallbox_parking", "wallbox_portail"])
@pytest.mark.parametrize(
    "scenario",
    ["regret_no_alternative", "all_equal_stable_order", "stickiness", "generic_fallback"],
)
def test_scenario_families_are_caller_independent(scenario, caller_name):
    # Story section B: caller independence holds "for each scenario above" —
    # a fresh fixture per caller gives every charger the IDENTICAL attachment
    # state, and each caller's own row must agree with the same global
    # allocation.
    fx = _make_incident_fixture(pin_buzz=(scenario != "generic_fallback"))
    expected = _setup_scenario(fx, scenario)
    caller = fx.chargers[caller_name]
    assert caller.get_best_car(T0) is expected[caller_name]


# =============================================================================
# C1 — score-decomposition INFO-on-change logging (AC5)
# =============================================================================


def _messages(caplog, fragment: str) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == CHARGER_LOGGER and fragment in record.getMessage()
    ]


def _decomposition_lines(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == CHARGER_LOGGER
        and record.levelno == logging.INFO
        and record.getMessage().startswith("get_car_score:")
    ]


def test_get_car_score_decomposition_dedups_by_value(caplog):
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    time = T0

    # First evaluation: exactly one INFO decomposition line.
    fx.parking.get_car_score(fx.zoe, time, {})
    assert len(_decomposition_lines(caplog)) == 1
    assert "plug_bump" in _decomposition_lines(caplog)[0]

    # GPS jitter INSIDE the same 0.5 m distance bucket: the quantised tuple is
    # unchanged, so the value is still remembered → zero new lines.
    fx.zoe.get_car_coordinates = MagicMock(return_value=(ZOE_COORDS[0] + 0.0000005, ZOE_COORDS[1]))
    time += timedelta(seconds=10)
    fx.parking.get_car_score(fx.zoe, time, {})
    assert len(_decomposition_lines(caplog)) == 1

    # KEY-COMPOSITION ORACLE (do not fold into the step above): the same in-bucket
    # jitter far later, still inside the 900 s TTL. Only the change key can explain
    # the silence, so this fails if the key ever grows a raw (unquantised)
    # component — raw distance here.
    time = T0 + timedelta(seconds=600)
    fx.parking.get_car_score(fx.zoe, time, {})
    assert len(_decomposition_lines(caplog)) == 1

    # Crossing a bucket edge (≈11 m) is a value never seen before → one new line.
    fx.zoe.get_car_coordinates = MagicMock(return_value=(ZOE_COORDS[0] + 0.0001, ZOE_COORDS[1]))
    fx.parking.get_car_score(fx.zoe, time, {})
    assert len(_decomposition_lines(caplog)) == 2

    # Reverting to the FIRST bucket stays silent — that value is still remembered.
    # This is the whole point of dedup-by-value: the old changed-value time floor
    # re-emitted here, so an A->B->A oscillation cost a line per TRANSITION. It now
    # costs one line per DISTINCT VALUE per window.
    fx.zoe.get_car_coordinates = MagicMock(return_value=ZOE_COORDS)
    fx.parking.get_car_score(fx.zoe, time + timedelta(seconds=7), {})
    assert len(_decomposition_lines(caplog)) == 2

    # Once the per-value TTL lapses, the 900 s heartbeat re-emits it.
    time = T0 + timedelta(seconds=_RELOG_UNCHANGED_AFTER_S + 7)
    fx.parking.get_car_score(fx.zoe, time, {})
    assert len(_decomposition_lines(caplog)) == 3


def test_get_car_score_change_key_ignores_the_attach_delta(caplog):
    # Second key-composition oracle: `connected_time_delta` grows on every call
    # while the car stays attached. It is payload, not part of the change key —
    # scoring 65 s later (floor elapsed, quantised tuple identical) must stay
    # silent. Fails if the attach delta ever enters the key.
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)

    fx.parking.attach_car(fx.zoe, T0)
    fx.parking.get_car_score(fx.zoe, T0, {})
    assert len(_decomposition_lines(caplog)) == 1

    later = T0 + timedelta(seconds=65)
    fx.parking.get_car_score(fx.zoe, later, {})
    assert len(_decomposition_lines(caplog)) == 1


def test_naive_time_is_normalised(caplog):
    # Replaces the old naive/aware "non-comparable clock" tests. That branch used to
    # emit unconditionally, so a caller alternating naive and aware datetimes under
    # one key escaped the bound ENTIRELY. `time` is now normalised to UTC up front:
    # a naive input must neither raise nor become an escape hatch.
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    charger = fx.parking

    charger.log_info_on_change("k", "a", T0, "tz probe %s", "aware")
    assert len(_messages(caplog, "tz probe")) == 1

    # Same instant, same value, expressed naively: still throttled, still no raise.
    charger.log_info_on_change("k", "a", T0.replace(tzinfo=None), "tz probe %s", "naive")
    assert len(_messages(caplog, "tz probe")) == 1

    # Alternating clock kinds cannot re-inflate the key either.
    for offset in range(1, 20):
        stamp = T0 + timedelta(seconds=7 * offset)
        charger.log_info_on_change(
            "k", "a", stamp if offset % 2 else stamp.replace(tzinfo=None), "tz probe %s", "mixed"
        )
    assert len(_messages(caplog, "tz probe")) == 1


def test_get_car_score_decomposition_bounded_when_tuple_oscillates(caplog):
    # A car GPS-jittering ACROSS a 0.5 m bucket edge alternates the change key on
    # every ~7 s allocation cycle (the incident's churn class). Under dedup this
    # collapses to ONE line per distinct bucket — two lines total, then silence for
    # the rest of the window.
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)

    in_bucket = ZOE_COORDS
    other_bucket = (ZOE_COORDS[0] + 0.0001, ZOE_COORDS[1])  # ≈11 m: different bucket

    for index in range(128):
        fx.zoe.get_car_coordinates = MagicMock(return_value=other_bucket if index % 2 else in_bucket)
        fx.parking.get_car_score(fx.zoe, T0 + timedelta(seconds=7 * index), {})

    # Exactly the two distinct states, not one line per transition, and well under
    # the per-key ceiling — so the budget never even engages for an oscillation.
    assert len(_decomposition_lines(caplog)) == 2


def test_monotone_value_drift_is_budget_capped(caplog):
    """The test that makes the BUDGET non-optional. Do not weaken it.

    Dedup-by-value kills oscillation perfectly and drift not at all: a car creeping
    0.55 m per cycle lands in a new 0.5 m bucket almost every cycle, so every cycle
    is a value never seen before. A dedup-only implementation emits ~87 lines here.
    """
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)

    # ~0.55 m per cycle, due north of the wallbox.
    metres_per_degree_lat = 111_320.0
    for index in range(128):
        drift = index * 0.55 / metres_per_degree_lat
        fx.zoe.get_car_coordinates = MagicMock(return_value=(ZOE_COORDS[0] + drift, ZOE_COORDS[1]))
        fx.parking.get_car_score(fx.zoe, T0 + timedelta(seconds=7 * index), {})

    lines = _decomposition_lines(caplog)
    assert len(lines) <= _PER_KEY_CEILING, lines
    # The drift is genuinely distinct-valued, so the budget (not dedup) is what
    # bounded it — if this ever drops to 1 the fixture stopped drifting.
    assert len(lines) >= 2, lines


def test_log_state_survives_detach_car(caplog):
    """Replaces `test_sf1_detach_car_clears_the_memo`, which pinned the defect.

    `detach_car()` sits ON the churn path — every change of the `get_best_car` value
    routes through it — so wiping the log state there dropped the key, left the next
    call with nothing to compare against, and made every throttle on this path a
    no-op in production. The state must now survive.
    """
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)

    fx.parking.attach_car(fx.zoe, T0)
    fx.parking.get_car_score(fx.zoe, T0 + timedelta(seconds=7), {})
    fx.parking.log_info_on_change("session", "seeded", T0, "session probe")
    assert len(_decomposition_lines(caplog)) == 1
    assert len(_messages(caplog, "session probe")) == 1

    fx.parking.detach_car()

    # Nothing was dropped, so neither key re-emits.
    assert "session" in fx.parking._log_on_change_state
    assert any(key.startswith("get_car_score:") for key in fx.parking._log_on_change_state)

    fx.parking.get_car_score(fx.zoe, T0 + timedelta(seconds=14), {})
    fx.parking.log_info_on_change("session", "seeded", T0 + timedelta(seconds=14), "session probe")
    assert len(_decomposition_lines(caplog)) == 1
    assert len(_messages(caplog, "session probe")) == 1


# =============================================================================
# Production-path volume (review #03 / D8)
#
# These drive `check_load_activity_and_constraints`, NOT `get_best_car` or
# `get_car_score` in isolation. That distinction is the whole reason round #02's
# throttle shipped green and did nothing: the isolated call never applies the
# result, so `detach_car()` never runs — and `detach_car()` was wiping the very
# state the throttle depended on. Isolated, emissions looked ~63 s apart; on the
# real path they were at the full 7 s cycle rate.
# =============================================================================


def _drive_real_cycle_path(fx) -> None:
    """Stub only HA I/O so `check_load_activity_and_constraints` runs end to end."""
    for charger in (fx.parking, fx.portail):
        charger.is_charger_unavailable = MagicMock(return_value=False)
        charger.probe_for_possible_needed_reboot = MagicMock(return_value=False)
        charger.is_not_plugged = MagicMock(return_value=False)
        charger.set_charging_num_phases = AsyncMock(return_value=False)
        charger.set_max_charging_current = AsyncMock(return_value=True)
        charger.reboot = AsyncMock()
        charger.is_car_stopped_asking_current = MagicMock(return_value=False)
    for car in (fx.zoe, fx.buzz):
        car.get_car_charge_percent = lambda time=None, *a, **kw: 40.0
        car.get_best_person_next_need = AsyncMock(return_value=(False, None, None, None))
        car.get_next_scheduled_event = AsyncMock(return_value=(None, None))
        car.setup_car_charge_target_if_needed = AsyncMock(return_value=80.0)


def _pin_scores(fx, per_charger: dict) -> None:
    """Give each charger a fixed score for the Zoe (0 for anything else)."""
    for charger in (fx.parking, fx.portail):
        charger.get_car_score = MagicMock(
            side_effect=lambda car, time, cache, _c=charger: (
                per_charger[_c.name] if car.name == "Zoe" else 0.0
            )
        )


def _info_volume(caplog) -> list:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.INFO and record.name in (CHARGER_LOGGER, LOAD_LOGGER)
    ]


async def _run_cycles(fx, caplog, cycles: int = 129, step_s: int = 7) -> None:
    for index in range(cycles):
        time = T0 + timedelta(seconds=step_s * index)
        for charger in (fx.parking, fx.portail):
            await charger.check_load_activity_and_constraints(time)


async def test_allocation_churn_total_log_volume_over_a_full_window(caplog):
    """AGGREGATE volume over a full 900 s window, not a per-prefix count.

    Every escaped site is invisible to a prefix-scoped assertion, which is how five
    of the six winner-announcement branches ran unthrottled through three review
    rounds. The scenario is a STABLE allocation — parking keeps the Zoe, portail
    falls back to its generic car — so no session churn is involved and every line
    above the floor is pure logging defect. Measured 149 lines before this round,
    129 of them from the single unthrottled `Default car used` branch.
    """
    fx = _make_incident_fixture(pin_buzz=False)
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    caplog.set_level(logging.INFO, logger=LOAD_LOGGER)
    fx.home._cars = [fx.zoe]
    _drive_real_cycle_path(fx)
    _pin_scores(fx, {"wallbox_parking": 100.0, "wallbox_portail": 90.0})

    await _run_cycles(fx, caplog)

    # N chargers, M real cars: get_car_score (N*M) + get_best_car (N) +
    # detach_from_other_charger (N) + soc (2N) + group (N+1) keys, each bounded by
    # `budget + 1` lines per window.
    num_chargers, num_cars = 2, 1
    ceiling = _PER_KEY_CEILING * (num_chargers * num_cars + 5 * num_chargers + 1)
    volume = _info_volume(caplog)
    assert len(volume) <= ceiling, f"{len(volume)} lines > {ceiling}:\n" + "\n".join(
        sorted({record.getMessage()[:110] for record in volume})
    )

    # The allocation itself is untouched by any of this.
    assert fx.parking.car is fx.zoe
    assert fx.portail.car is fx.portail._default_generic_car


async def test_generic_car_fallback_line_is_throttled(caplog):
    # `Default car used` fired 129/129 cycles: a charger with no scoring car says so
    # every single cycle, forever, while nothing whatsoever is changing.
    fx = _make_incident_fixture(pin_buzz=False)
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    fx.home._cars = [fx.zoe]
    _drive_real_cycle_path(fx)
    _pin_scores(fx, {"wallbox_parking": 100.0, "wallbox_portail": 90.0})

    await _run_cycles(fx, caplog)

    fallback = [r for r in caplog.records if "generic_fallback" in r.getMessage()]
    assert len(fallback) <= _PER_KEY_CEILING, fallback


async def test_removed_from_another_charger_is_throttled(caplog):
    # THE incident message — the literal line that appeared 17 791 times. It sits
    # inside the ping-pong condition and was still a raw unthrottled f-string.
    fx = _make_incident_fixture(pin_buzz=False)
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    fx.home._cars = [fx.zoe]
    _drive_real_cycle_path(fx)

    # The winner flips between the two chargers every cycle, so the loser's
    # `best_car.charger is not self` branch is taken on every cycle.
    per_charger = {"wallbox_parking": 100.0, "wallbox_portail": 90.0}
    _pin_scores(fx, per_charger)
    for index in range(129):
        per_charger["wallbox_parking"], per_charger["wallbox_portail"] = (
            (90.0, 100.0) if index % 2 else (100.0, 90.0)
        )
        time = T0 + timedelta(seconds=7 * index)
        for charger in (fx.parking, fx.portail):
            await charger.check_load_activity_and_constraints(time)

    removed = [r for r in caplog.records if "removed from another charger" in r.getMessage()]
    assert len(removed) <= _PER_KEY_CEILING, removed


async def test_user_pinned_car_line_is_throttled(caplog):
    # `Best Car from user selection` fired 20/20 cycles for a setting that never
    # changes. It is now one branch of the single merged `get_best_car` key.
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    _drive_real_cycle_path(fx)
    fx.parking.set_user_originated(USER_ORIGINATED_CAR_NAME, fx.zoe.name)

    await _run_cycles(fx, caplog, cycles=20)

    pinned = [r for r in caplog.records if "user_selection" in r.getMessage()]
    assert len(pinned) <= _PER_KEY_CEILING, pinned


async def test_no_car_selected_state_is_idempotent(caplog):
    """The single largest volume source: ~74 000 lines/day from one charger.

    Selecting "no car connected" ran a full `reset(keep_commands=True)` EVERY cycle,
    emitting six INFO lines each time — four times the original incident. Once there
    is nothing left to reset the state must go completely quiet.
    """
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    caplog.set_level(logging.INFO, logger=LOAD_LOGGER)
    _drive_real_cycle_path(fx)
    for charger in (fx.parking, fx.portail):
        charger.set_user_originated(USER_ORIGINATED_CAR_NAME, CHARGER_NO_CAR_CONNECTED)

    await _run_cycles(fx, caplog, cycles=20)

    # Two chargers x (one merged `get_best_car` key + at most one settling reset).
    volume = _info_volume(caplog)
    assert len(volume) <= 2 * (_PER_KEY_CEILING + 6), "\n".join(
        record.getMessage()[:110] for record in volume
    )

    # And the reset genuinely stops running rather than merely logging less.
    reset_lines = [r for r in caplog.records if "CHARGER_NO_CAR_CONNECTED selected option" in r.getMessage()]
    assert len(reset_lines) <= 2, reset_lines
    assert fx.parking.car is None


async def test_force_car_no_charger_is_not_logged_from_the_inner_loop(caplog):
    # This reports a STATIC user setting from inside a charger x car double loop
    # that itself runs once per charger per cycle: N^2*M lines per cycle, measured
    # 101 over 20 cycles (~62k/day). It belongs at DEBUG.
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    _drive_real_cycle_path(fx)
    for car in (fx.zoe, fx.buzz):
        car.set_user_originated(USER_ORIGINATED_CHARGER_NAME, FORCE_CAR_NO_CHARGER_CONNECTED)

    await _run_cycles(fx, caplog, cycles=20)

    assert [r for r in caplog.records if "FORCE_CAR_NO_CHARGER_CONNECTED" in r.getMessage()] == []


def test_same_cycle_duplicate_calls_emit_once(caplog):
    """One cycle, N chargers, ONE line per `get_car_score:{car}` key.

    `time` is HA's single `event_time` for the whole cycle, and each key is hit once
    per active charger's `get_best_car`. Under the old floor those N same-instant
    calls inflated the suppressed-change bank by a factor of N; under dedup the
    first emits and the rest are the same value at the same instant.
    """
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)

    for charger in (fx.parking, fx.portail):
        charger.get_best_car(T0)

    for car in (fx.zoe, fx.buzz):
        for charger in (fx.parking, fx.portail):
            per_key = [
                record
                for record in caplog.records
                if record.getMessage().startswith(f"get_car_score: {car.name} for {charger.name} ")
            ]
            assert len(per_key) == 1, per_key
