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
    """Pin a car to the incident matrix: plugged (both call forms) + fixed GPS."""
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
    """One full allocation round: each charger picks then applies its own row."""
    for charger in chargers:
        car = charger.get_best_car(time)
        if car is not None:
            charger.attach_car(car, time)


def _allocation(fx) -> dict:
    return {
        name: (charger.car.name if charger.car is not None else None)
        for name, charger in fx.chargers.items()
    }


def _spy_detach(charger) -> list:
    """Wrap `detach_car` so post-convergence oscillation is observable."""
    original = charger.detach_car
    calls: list[str] = []

    def spy():
        calls.append(charger.name)
        original()

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
# Incident replay (expected red on pre-fix code)
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
# Start-state × execution-order sweep (expected red on pre-fix code)
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


# =============================================================================
# C1 — score-decomposition INFO-on-change logging (AC5)
# =============================================================================


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

    # Crossing a bucket edge (≈11 m shift) changes the quantised tuple → one
    # new line.
    fx.zoe.get_car_coordinates = MagicMock(return_value=(ZOE_COORDS[0] + 0.0001, ZOE_COORDS[1]))
    time += timedelta(seconds=10)
    fx.parking.get_car_score(fx.zoe, time, {})
    assert len(_decomposition_lines(caplog)) == 2

    # Unchanged value re-emits once the 900 s heartbeat elapses (pins the
    # bounded-volume claim).
    time += timedelta(seconds=901)
    fx.parking.get_car_score(fx.zoe, time, {})
    assert len(_decomposition_lines(caplog)) == 3
