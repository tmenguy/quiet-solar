"""Liveness and the watchdog (QS-406 §9).

The ``liveness_watchdog`` hook (every ``LIVENESS_EVERY_S``) takes one listing,
``claude.try_agents(timeout=HOOK_SUBPROCESS_S)`` (``None`` means unknown), and beats right after.

- **Nodes:** ``nodes.refresh``. ``node_dead``: a task in ``tasks.NODE_RANGE`` whose current generation
  is ``reaped``, seen on ``LIVENESS_CONFIRM`` listings in a row.
- **Orchestrators:** ``orchestrator_dead``: an open run whose lease session is absent from
  ``LIVENESS_CONFIRM`` successful listings in a row.
- A ``None`` listing leaves every liveness kind out of the sync and keeps the counters.
- **Cold state** (a new daemon): each sighting counter is seeded from the open ``alerts`` rows (an open
  subject counts as confirmed), and the kinds are left out until this daemon has taken
  ``LIVENESS_CONFIRM`` successful listings — so a restart is never a recurrence.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from . import activeloop, alerts, daemon, liveness, nodes, tasks, ticks
from . import clock as clock_mod

LIVENESS_WATCHDOG = "liveness_watchdog"
LIVENESS_EVERY_S = 30.0
LIVENESS_CONFIRM = 2

_THROTTLE = ticks.Throttle(LIVENESS_EVERY_S)


@dataclass
class _State:
    listings: int = 0
    seeded: bool = False
    node_seen: dict[str, int] = field(default_factory=dict)
    orch_seen: dict[str, int] = field(default_factory=dict)


_state = _State()


def _seed(conn: sqlite3.Connection) -> None:
    for row in conn.execute(
        "SELECT kind, subject FROM alerts WHERE cleared_at IS NULL AND kind IN (?, ?)",
        (alerts.NODE_DEAD, alerts.ORCHESTRATOR_DEAD),
    ).fetchall():
        seen = _state.node_seen if row["kind"] == alerts.NODE_DEAD else _state.orch_seen
        seen[row["subject"]] = LIVENESS_CONFIRM
    _state.seeded = True


def _count(seen: dict[str, int], present: dict[str, tuple[str, dict[str, object]]]) -> dict[str, int]:
    return {subject: seen.get(subject, 0) + 1 for subject in present}


def _dead_nodes(conn: sqlite3.Connection) -> dict[str, tuple[str, dict[str, object]]]:
    marks = ", ".join("?" for _ in tasks.NODE_RANGE)
    rows = conn.execute(
        f"SELECT n.id, n.run_id, n.task_id, n.generation FROM nodes n JOIN tasks t ON t.id = n.task_id"
        f" WHERE t.state IN ({marks}) AND n.state = 'reaped'"
        f" AND n.generation = (SELECT max(generation) FROM nodes WHERE task_id = n.task_id) ORDER BY n.id",
        tuple(sorted(tasks.NODE_RANGE)),
    ).fetchall()
    return {
        r["id"]: (r["run_id"], {"node_id": r["id"], "task_id": r["task_id"], "generation": r["generation"]})
        for r in rows
    }


def _dead_orchestrators(
    conn: sqlite3.Connection, listing: list[liveness.Agent]
) -> dict[str, tuple[str, dict[str, object]]]:
    rows = conn.execute(
        "SELECT r.id, l.session_id FROM runs r JOIN run_leases l ON l.run_id = r.id WHERE r.state = 'open' ORDER BY r.rowid"
    ).fetchall()
    return {
        f"{r['id']}:{r['session_id']}": (r["id"], {"run_id": r["id"], "session_id": r["session_id"]})
        for r in rows
        if liveness.find(listing, session_id=r["session_id"]) is None
    }


def liveness_watchdog_hook(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
    if not _THROTTLE.due(clock):
        return
    seams = activeloop.seams()
    listing = seams.claude.try_agents(timeout=ticks.HOOK_SUBPROCESS_S)
    daemon.beat(conn, clock)
    if listing is None:
        return  # unknown: nothing is synced, the counters and ladders are kept
    nodes.refresh(conn, clock, listing)
    if not _state.seeded:
        _seed(conn)
    _state.listings += 1
    dead_nodes = _dead_nodes(conn)
    dead_orchs = _dead_orchestrators(conn, listing)
    _state.node_seen = _count(_state.node_seen, dead_nodes)
    _state.orch_seen = _count(_state.orch_seen, dead_orchs)
    if _state.listings < LIVENESS_CONFIRM:
        return  # cold: not enough sightings by this daemon to clear or raise anything
    active = [
        alerts.Condition(alerts.NODE_DEAD, subject, (run,), payload)
        for subject, (run, payload) in dead_nodes.items()
        if _state.node_seen[subject] >= LIVENESS_CONFIRM
    ] + [
        alerts.Condition(alerts.ORCHESTRATOR_DEAD, subject, (run,), payload)
        for subject, (run, payload) in dead_orchs.items()
        if _state.orch_seen[subject] >= LIVENESS_CONFIRM
    ]
    alerts.sync(conn, clock, kinds={alerts.NODE_DEAD, alerts.ORCHESTRATOR_DEAD}, active=active)


def _reset_for_tests() -> None:
    global _state
    _state = _State()
