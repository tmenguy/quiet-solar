"""QS-406 T10/T11b: liveness and the watchdog (§9, AC 11, AC 12, and AC 3's liveness restart)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from control_plane import activeloop, alerts, clock, ticks, watchdog

from .conftest import ORCH, FakeClaude, agent, insert_node, insert_task, open_run, sql


def _tick(conn: Any, fake_clock: clock.FakeClock) -> None:
    watchdog.liveness_watchdog_hook(conn, fake_clock)
    fake_clock.advance(watchdog.LIVENESS_EVERY_S)


def _open(path: Path, kind: str | None = None) -> list[tuple[str, str, str]]:
    rows = sql(path, "SELECT run_id, kind, subject FROM alerts WHERE cleared_at IS NULL ORDER BY id")
    return [(r[0], r[1], r[2]) for r in rows if kind is None or r[1] == kind]


def _messages(path: Path) -> int:
    return int(sql(path, "SELECT count(*) FROM messages WHERE sender = 'cp:daemon'")[0][0])


# --------------------------------------------------------------------------- liveness (AC 11)


class TestLiveness:
    def test_node_dead_after_two_sightings(self, conn, migrated, fake_claude: FakeClaude, fake_clock) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1, state="building")
        insert_node(
            migrated, "N1", r1, "T1", state="running", session_id="S-n1", launch_at="2020-01-01T00:00:00.000000Z"
        )
        fake_claude.listing = [agent(ORCH)]  # the node's session is gone: refresh reaps it
        _tick(conn, fake_clock)
        assert _open(migrated) == []  # a single sighting raises nothing
        _tick(conn, fake_clock)
        assert _open(migrated) == [(r1, "node_dead", "N1")]
        assert fake_claude.last_timeout == ticks.HOOK_SUBPROCESS_S

    def test_a_newer_generation_or_a_finished_task_is_not_dead(self, conn, migrated, fake_claude, fake_clock) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1, state="building")
        insert_node(migrated, "N1", r1, "T1", state="reaped")
        insert_node(migrated, "N2", r1, "T1", generation=2, state="spawning")
        insert_task(migrated, "T2", r1, state="merged")
        insert_node(migrated, "N3", r1, "T2", state="reaped")
        fake_claude.listing = [agent(ORCH)]
        for _ in range(3):
            _tick(conn, fake_clock)
        assert _open(migrated) == []

    def test_orchestrator_dead_after_two_misses(self, conn, migrated, fake_claude, fake_clock) -> None:
        r1, _ = open_run()
        fake_claude.listing = []
        _tick(conn, fake_clock)
        assert _open(migrated) == []
        fake_claude.listing = [agent(ORCH)]  # back: the count starts over
        _tick(conn, fake_clock)
        fake_claude.listing = []
        _tick(conn, fake_clock)
        assert _open(migrated) == []
        _tick(conn, fake_clock)
        assert _open(migrated) == [(r1, "orchestrator_dead", f"{r1}:{ORCH}")]
        fake_claude.listing = [agent(ORCH)]
        _tick(conn, fake_clock)
        assert _open(migrated) == []

    def test_a_failed_listing_changes_nothing(self, conn, migrated, fake_claude, fake_clock) -> None:
        r1, _ = open_run()
        fake_claude.listing = []
        _tick(conn, fake_clock)
        _tick(conn, fake_clock)
        assert len(_open(migrated)) == 1
        fake_claude.listing = None
        for _ in range(3):
            _tick(conn, fake_clock)
        assert len(_open(migrated)) == 1 and watchdog._state.orch_seen == {f"{r1}:{ORCH}": 2}

    def test_a_restart_is_not_a_recurrence(self, conn, migrated, fake_claude, fake_clock) -> None:
        r1, _ = open_run()
        insert_task(migrated, "T1", r1, state="building")
        insert_node(migrated, "N1", r1, "T1", state="reaped")
        fake_claude.listing = []
        _tick(conn, fake_clock)
        _tick(conn, fake_clock)
        assert {k for _, k, _ in _open(migrated)} == {"node_dead", "orchestrator_dead"}
        sent = _messages(migrated)
        ticks._reset_for_tests()
        activeloop._reset_for_tests()  # a new daemon: cold counters
        _tick(conn, fake_clock)  # one listing: the kinds are left out, nothing is cleared
        assert len(_open(migrated)) == 2
        _tick(conn, fake_clock)  # the seeded subjects count as confirmed
        assert len(_open(migrated)) == 2 and _messages(migrated) == sent

    def test_throttled(self, conn, migrated, fake_claude, fake_clock) -> None:
        watchdog.liveness_watchdog_hook(conn, fake_clock)
        watchdog.liveness_watchdog_hook(conn, fake_clock)
        assert fake_claude.listings == 1


def test_kinds_are_known() -> None:
    assert {alerts.NODE_DEAD, alerts.ORCHESTRATOR_DEAD} <= alerts.KINDS


@pytest.mark.usefixtures("active_loop")
def test_the_hook_is_built_in() -> None:
    assert watchdog.LIVENESS_WATCHDOG in dict(ticks.registered())
