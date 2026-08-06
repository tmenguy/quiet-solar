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
from unittest.mock import MagicMock

import pytest
import pytz

from custom_components.quiet_solar.const import (
    CONF_CHARGER_LATITUDE,
    CONF_CHARGER_LONGITUDE,
    USER_ORIGINATED_CAR_NAME,
)
from custom_components.quiet_solar.ha_model.charger import _CHANGED_RELOG_MIN_INTERVAL_S
from tests.utils.charger_harness import (
    create_charger,
    make_hass,
    make_home,
    make_real_car,
)

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


def test_get_car_score_decomposition_logs_once_then_on_change(caplog):
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    time = T0

    # First evaluation: exactly one INFO decomposition line.
    fx.parking.get_car_score(fx.zoe, time, {})
    assert len(_decomposition_lines(caplog)) == 1
    assert "plug_bump" in _decomposition_lines(caplog)[0]

    # GPS jitter INSIDE the same 0.5 m distance bucket, within the 900 s
    # window: the quantised tuple is unchanged → zero new lines.
    fx.zoe.get_car_coordinates = MagicMock(return_value=(ZOE_COORDS[0] + 0.0000005, ZOE_COORDS[1]))
    time += timedelta(seconds=10)
    fx.parking.get_car_score(fx.zoe, time, {})
    assert len(_decomposition_lines(caplog)) == 1

    # KEY-COMPOSITION ORACLE (do not fold into the step above): in-bucket jitter
    # with the last emission OLDER than the rate floor. The floor can no longer
    # explain the silence, so this fails if the change key ever grows a raw
    # (unquantised) component — raw distance here.
    fx.zoe.get_car_coordinates = MagicMock(return_value=(ZOE_COORDS[0] + 0.0000005, ZOE_COORDS[1]))
    time = T0 + timedelta(seconds=_CHANGED_RELOG_MIN_INTERVAL_S + 5)
    fx.parking.get_car_score(fx.zoe, time, {})
    assert len(_decomposition_lines(caplog)) == 1

    # Crossing a bucket edge (≈11 m shift) changes the quantised tuple → one
    # new line (the floor has elapsed since the only emission, at T0).
    fx.zoe.get_car_coordinates = MagicMock(return_value=(ZOE_COORDS[0] + 0.0001, ZOE_COORDS[1]))
    fx.parking.get_car_score(fx.zoe, time, {})
    assert len(_decomposition_lines(caplog)) == 2

    # A change WITHIN the floor window is rate-limited: no new line.
    fx.zoe.get_car_coordinates = MagicMock(return_value=ZOE_COORDS)
    fx.parking.get_car_score(fx.zoe, time + timedelta(seconds=7), {})
    assert len(_decomposition_lines(caplog)) == 2

    # Unchanged value re-emits once the 900 s heartbeat elapses (pins the
    # bounded-volume claim).
    fx.zoe.get_car_coordinates = MagicMock(return_value=(ZOE_COORDS[0] + 0.0001, ZOE_COORDS[1]))
    time += timedelta(seconds=901)
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

    later = T0 + timedelta(seconds=_CHANGED_RELOG_MIN_INTERVAL_S + 5)
    fx.parking.get_car_score(fx.zoe, later, {})
    assert len(_decomposition_lines(caplog)) == 1


def test_get_car_score_excursion_reverting_inside_the_floor_is_disclosed(caplog):
    # QS-342 review #02 / S1: a tuple excursion that begins AND ends inside the
    # floor window must not vanish. The memo is not restamped on suppression, so
    # a revert would otherwise leave the operator a flat trace while the 900 s
    # heartbeat re-emits the pre-excursion tuple.
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)

    fx.parking.get_car_score(fx.zoe, T0, {})
    assert len(_decomposition_lines(caplog)) == 1

    # Excursion to another bucket for three cycles, all inside the floor.
    fx.zoe.get_car_coordinates = MagicMock(return_value=(ZOE_COORDS[0] + 0.0001, ZOE_COORDS[1]))
    for offset in (7, 14, 21):
        fx.parking.get_car_score(fx.zoe, T0 + timedelta(seconds=offset), {})
    assert len(_decomposition_lines(caplog)) == 1

    # Reverts to the pre-excursion bucket, still inside the floor: nothing yet.
    fx.zoe.get_car_coordinates = MagicMock(return_value=ZOE_COORDS)
    fx.parking.get_car_score(fx.zoe, T0 + timedelta(seconds=28), {})
    assert len(_decomposition_lines(caplog)) == 1

    # Once the floor expires the banked excursion is disclosed, even though the
    # value now equals the last emitted one.
    fx.parking.get_car_score(fx.zoe, T0 + timedelta(seconds=_CHANGED_RELOG_MIN_INTERVAL_S + 5), {})
    lines = _decomposition_lines(caplog)
    assert len(lines) == 2
    assert "suppressed" in lines[1]
    assert "3" in lines[1]  # the three suppressed changes are counted


def test_log_info_on_change_floor_survives_a_tzinfo_mismatch(caplog):
    # S4: the naive/aware mismatch path cannot compute `elapsed`, so it emits
    # (a logging helper must never raise). Pin that this is ONE-SHOT: the
    # emission restamps the key, so the floor governs again immediately after.
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    charger = fx.parking
    floor = {"min_interval_s": _CHANGED_RELOG_MIN_INTERVAL_S}

    charger.log_info_on_change("k", "a", T0, "tz probe %s", "aware", **floor)
    charger.log_info_on_change("k", "b", T0.replace(tzinfo=None), "tz probe %s", "naive", **floor)
    assert len(_messages(caplog, "tz probe")) == 2

    # Same (naive) clock kind again, changed value inside the floor: suppressed.
    charger.log_info_on_change(
        "k", "c", T0.replace(tzinfo=None) + timedelta(seconds=7), "tz probe %s", "naive again", **floor
    )
    assert len(_messages(caplog, "tz probe")) == 2


def _gaps(offsets: list[int]) -> list[int]:
    """Simulated-time distances between consecutive emissions."""
    return [later - earlier for earlier, later in zip(offsets[:-1], offsets[1:], strict=True)]


def _emission_offsets(caplog, lines_fn, cycles, step_s, run_cycle) -> list[int]:
    """Simulated-time offsets (s) at which `lines_fn` gained a line."""
    offsets: list[int] = []
    seen = len(lines_fn(caplog))
    for index in range(cycles):
        offset = index * step_s
        run_cycle(index, T0 + timedelta(seconds=offset))
        current = len(lines_fn(caplog))
        if current > seen:
            offsets.append(offset)
            seen = current
    return offsets


def test_get_car_score_decomposition_rate_limited_when_tuple_oscillates(caplog):
    # A car GPS-jittering ACROSS a 0.5 m bucket edge alternates the change key
    # on every ~7 s allocation cycle (the incident's churn class): emissions
    # must stay bounded by the changed-value floor, not track the cycle rate.
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)

    in_bucket = ZOE_COORDS
    other_bucket = (ZOE_COORDS[0] + 0.0001, ZOE_COORDS[1])  # ≈11 m: different bucket

    def _cycle(index, time):
        fx.zoe.get_car_coordinates = MagicMock(return_value=other_bucket if index % 2 else in_bucket)
        fx.parking.get_car_score(fx.zoe, time, {})

    offsets = _emission_offsets(caplog, _decomposition_lines, cycles=40, step_s=7, run_cycle=_cycle)

    # Pin the MECHANISM, not a loose count: consecutive emissions are at least a
    # floor apart (a floor half as effective would fail), and churn still gets
    # reported rather than silenced entirely.
    assert len(offsets) >= 2
    assert all(gap >= _CHANGED_RELOG_MIN_INTERVAL_S for gap in _gaps(offsets)), offsets


def test_get_best_car_log_is_rate_limited_when_the_winner_flips(caplog):
    # S2: the sibling INFO-on-change site in `get_best_car` is keyed on the
    # winning car name. The documented residual mode (a car jittering across a
    # 0.5 m bucket edge flips the strict-score winner) drives it at the full
    # cycle rate unless it is floored too.
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)
    _patch_scores(fx.portail, {})  # never competes: keeps the flip on `parking`
    scores = {"Zoe": 100.0, "IDBuzz": 90.0}
    fx.parking.get_car_score = MagicMock(side_effect=lambda car, time, cache: scores.get(car.name, 0.0))

    def _lines(cap):
        return [
            record.getMessage()
            for record in cap.records
            if record.name == CHARGER_LOGGER and record.getMessage().startswith("get_best_car: ")
        ]

    def _cycle(index, time):
        # Strict-score winner flips every cycle — no tie, so the cascade never
        # sees it (this is the documented bucket-edge residual).
        scores["Zoe"], scores["IDBuzz"] = (90.0, 100.0) if index % 2 else (100.0, 90.0)
        fx.parking.get_best_car(time)

    offsets = _emission_offsets(caplog, _lines, cycles=20, step_s=7, run_cycle=_cycle)

    assert len(offsets) >= 2
    assert all(gap >= _CHANGED_RELOG_MIN_INTERVAL_S for gap in _gaps(offsets)), offsets


def test_detach_car_preserves_get_car_score_memo(caplog):
    # `detach_car` wipes the QS-306 memos describing the departed session, but
    # the `get_car_score:{car}` keys are car-qualified and cover EVERY candidate
    # car: they must survive, or attach/detach churn re-emits one decomposition
    # line per car per detach even with unchanged tuples.
    fx = _make_incident_fixture()
    caplog.set_level(logging.INFO, logger=CHARGER_LOGGER)

    fx.parking.attach_car(fx.zoe, T0)
    fx.parking.get_car_score(fx.zoe, T0 + timedelta(seconds=7), {})
    assert len(_decomposition_lines(caplog)) == 1

    # Seed a NON-score memo so the wipe assertion below is not vacuous: without
    # it the prefix-only check passes even if the wipe stops clearing anything.
    fx.parking.log_info_on_change("session", "seeded", T0, "session probe")
    assert "session" in fx.parking._log_on_change_state

    fx.parking.detach_car()

    # Unchanged tuple right after the detach: no re-emission burst.
    fx.parking.get_car_score(fx.zoe, T0 + timedelta(seconds=14), {})
    assert len(_decomposition_lines(caplog)) == 1

    # The non-get_car_score memos ARE still cleared (QS-306 fresh-session rule).
    assert "session" not in fx.parking._log_on_change_state
    assert all(key.startswith("get_car_score:") for key in fx.parking._log_on_change_state)
