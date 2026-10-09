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


# --------------------------------------------------------------------------- the ladder (AC 12)

import json  # noqa: E402

import models  # noqa: E402
from control_plane import paths  # noqa: E402
from control_plane.runner import RunResult  # noqa: E402

from .conftest import FakeProbe, FakeRunner, run_cli  # noqa: E402


def _stall(path: Path, run_id: str, fake_clock: clock.FakeClock, age: float = watchdog.WATCHDOG_S + 1) -> int:
    old = clock.stamp(fake_clock, plus=-age)
    sql(
        path,
        "INSERT INTO messages (run_id, recipient, kind, sender, payload, state, visible_at, created_at)"
        " VALUES (?, 'orchestrator', 'k', 'node:T1', '{}', 'queued', ?, ?)",
        [run_id, old, old],
    )
    return int(sql(path, "SELECT max(id) FROM messages")[0][0])


def _messenger_rows(path: Path) -> list[dict[str, Any]]:
    return [dict(r) for r in sql(path, "SELECT * FROM tool_calls WHERE tool = 'watchdog-messenger' ORDER BY key")]


def _launches(runner: FakeRunner) -> list[Any]:
    return runner.matching("claude", "--bg")


@pytest.fixture
def stalled(conn, migrated, fake_claude: FakeClaude, fake_clock) -> dict[str, Any]:
    run_id, token = open_run()
    head = _stall(migrated, run_id, fake_clock)
    fake_claude.listing = [agent(ORCH, "r1", status="idle")]
    return {"run": run_id, "token": token, "head": head}


class TestLadder:
    def test_a_stalled_idle_run_launches_the_messenger(
        self, conn, migrated, stalled, fake_runner, fake_clock, fake_main, tmp_path
    ) -> None:
        _tick(conn, fake_clock)
        [call] = _launches(fake_runner)
        directory = paths.messenger_dir()
        assert call.cwd == str(directory) and directory == (tmp_path / "messenger").resolve()
        assert directory.stat().st_mode & 0o777 == 0o700 and not directory.is_relative_to(fake_main)
        argv = call.argv
        assert argv[argv.index("--model") + 1] == models.model_for("claude", "fast")
        assert argv[argv.index("--permission-mode") + 1] == "auto" and "--allowedTools=SendMessage" in argv
        assert argv[argv.index("-n") + 1] == f"qs-wake-{stalled['run']}-m{stalled['head']}"
        prompt = argv[-1]
        assert "session named `r1`" in prompt and f"msg pop --run {stalled['run']} --as orchestrator" in prompt
        assert stalled["token"] not in prompt and "<your run token>" in prompt
        [row] = _messenger_rows(migrated)
        assert row["key"] == f"msg:{stalled['head']}" and row["state"] == "succeeded" and row["exit_code"] == 0
        assert row["run_id"] == stalled["run"] and row["actor"] == "cp:daemon" and row["args_hash"] == "-"
        assert json.loads(row["args"]) == argv[argv.index("--bg") + 1 :] and row["holder_pid"] and row["finished_at"]
        assert _open(migrated) == []

    def test_a_replay_launches_nothing_then_not_listening(
        self, conn, migrated, stalled, fake_runner, fake_clock
    ) -> None:
        _tick(conn, fake_clock)
        _tick(conn, fake_clock)
        assert len(_launches(fake_runner)) == 1 and _open(migrated) == []
        fake_clock.advance(watchdog.WAKE_RETRY_S)
        _tick(conn, fake_clock)
        assert _open(migrated) == [(stalled["run"], "orchestrator_not_listening", stalled["run"])]
        assert len(_launches(fake_runner)) == 1
        snap = run_cli("snapshot")[1]
        assert snap["alerts"][0]["kind"] == "orchestrator_not_listening"
        assert snap["alerts"][0]["payload"]["reason"] == "still_stalled"

    def test_a_restart_adopts_the_heads_row(self, conn, migrated, stalled, fake_runner, fake_clock) -> None:
        _tick(conn, fake_clock)
        ticks._reset_for_tests()
        activeloop._reset_for_tests()
        _tick(conn, fake_clock)
        assert len(_launches(fake_runner)) == 1 and _open(migrated) == []
        fake_clock.advance(watchdog.WAKE_RETRY_S)
        _tick(conn, fake_clock)
        assert len(_open(migrated)) == 1 and len(_launches(fake_runner)) == 1

    def test_a_failed_launch_is_not_listening_at_once(self, conn, migrated, stalled, fake_runner, fake_clock) -> None:
        fake_runner.on(("claude", "--bg"), RunResult(1, "", "boom"))
        _tick(conn, fake_clock)
        assert _messenger_rows(migrated)[0]["state"] == "failed"
        assert _open(migrated) == [(stalled["run"], "orchestrator_not_listening", stalled["run"])]
        [alert] = run_cli("snapshot")[1]["alerts"]  # AC 12: visible to the maintainer in `snapshot`
        assert alert["kind"] == "orchestrator_not_listening" and alert["payload"]["reason"] == "messenger_failed"
        ticks._reset_for_tests()
        activeloop._reset_for_tests()
        _tick(conn, fake_clock)  # a new daemon: the failed row means rung 3 at once, still no relaunch
        assert len(_open(migrated)) == 1 and len(_launches(fake_runner)) == 1

    def test_the_cap_makes_the_next_run_wait_and_a_failed_row_does_not_take_it(
        self, conn, migrated, fake_claude, fake_runner, fake_clock
    ) -> None:
        r1, _ = open_run("r1", "S-1")
        r2, _ = open_run("r2", "S-2")
        _stall(migrated, r1, fake_clock)
        _stall(migrated, r2, fake_clock)
        fake_claude.listing = [agent("S-1", "r1", status="idle"), agent("S-2", "r2", status="idle")]
        _tick(conn, fake_clock)
        assert [r["run_id"] for r in _messenger_rows(migrated)] == [r1]  # r2 waits for the cap
        fake_clock.advance(watchdog.MESSENGER_TTL_S)
        _tick(conn, fake_clock)
        assert sorted(r["run_id"] for r in _messenger_rows(migrated)) == sorted([r1, r2])

    def test_a_failed_row_does_not_take_the_cap(self, conn, migrated, fake_claude, fake_runner, fake_clock) -> None:
        r1, _ = open_run("r1", "S-1")
        r2, _ = open_run("r2", "S-2")
        _stall(migrated, r1, fake_clock)
        _stall(migrated, r2, fake_clock)
        fake_claude.listing = [agent("S-1", "r1", status="idle"), agent("S-2", "r2", status="idle")]
        fake_runner.on(("claude", "--bg"), RunResult(1, "", "boom"))
        _tick(conn, fake_clock)
        assert len(_launches(fake_runner)) == 2  # the failed first launch left the cap free

    def test_an_unbound_run_name_is_not_listening(self, conn, migrated, stalled, fake_runner, fake_clock) -> None:
        sql(migrated, "UPDATE run_leases SET name_bound_session_id = 'S-other'")
        _tick(conn, fake_clock)
        assert _launches(fake_runner) == [] and len(_open(migrated)) == 1

    @pytest.mark.parametrize("case", ["busy", "absent", "waiter", "dead", "unknown"])
    def test_nothing_is_raised_or_cleared(
        self, conn, migrated, stalled, fake_claude, fake_runner, fake_clock, case: str
    ) -> None:
        fake_runner.on(("claude", "--bg"), RunResult(1, "", "boom"))
        _tick(conn, fake_clock)
        assert len(_open(migrated)) == 1  # rung 3 reached
        if case == "busy":
            fake_claude.listing = [agent(ORCH, "r1", status="busy")]
        elif case == "absent":
            fake_claude.listing = []
        elif case == "waiter":
            sql(
                migrated,
                "INSERT INTO waiters (run_id, pid, pid_start, started_at, heartbeat_at) VALUES (?, 7, NULL, 'x', 'x')",
                [stalled["run"]],
            )
        elif case == "dead":
            sql(
                migrated,
                "INSERT INTO alerts (run_id, kind, subject, fingerprint, payload, first_seen, last_seen) VALUES (?, 'orchestrator_dead', 's', 'f', '{}', 'x', 'x')",
                [stalled["run"]],
            )
        else:
            fake_claude.listing = None
        _tick(conn, fake_clock)
        assert "orchestrator_not_listening" in {k for _, k, _ in _open(migrated)}

    def test_a_drained_or_fresh_head_clears(self, conn, migrated, stalled, fake_runner, fake_clock) -> None:
        fake_runner.on(("claude", "--bg"), RunResult(1, "", "boom"))
        _tick(conn, fake_clock)
        sql(migrated, "UPDATE messages SET state = 'acked'")
        _tick(conn, fake_clock)
        assert _open(migrated) == []
        _stall(migrated, stalled["run"], fake_clock, age=10)  # a fresh head is not stalled yet
        _tick(conn, fake_clock)
        assert _open(migrated) == [] and len(_launches(fake_runner)) == 1

    def test_a_started_row_whose_holder_is_dead_is_failed(
        self, conn, migrated, stalled, fake_probe: FakeProbe, fake_runner, fake_clock
    ) -> None:
        sql(
            migrated,
            "INSERT INTO tool_calls (tool, key, args_hash, run_id, actor, args, state, holder_pid, holder_pid_start, started_at)"
            " VALUES ('watchdog-messenger', ?, '-', ?, 'cp:daemon', '[]', 'started', 4242, 'start-4242', ?)",
            [f"msg:{stalled['head']}", stalled["run"], clock.stamp(fake_clock)],
        )
        _tick(conn, fake_clock)  # alive: the launch is in progress, nothing happens
        assert _messenger_rows(migrated)[0]["state"] == "started" and _open(migrated) == []
        fake_probe.kill(4242)
        _tick(conn, fake_clock)
        assert _messenger_rows(migrated)[0]["state"] == "failed" and len(_open(migrated)) == 1
        assert _launches(fake_runner) == []

    def test_a_lost_claim_launches_nothing(
        self, conn, migrated, stalled, fake_runner, fake_clock, fake_probe, monkeypatch
    ) -> None:
        real = fake_probe.me

        def racing() -> Any:
            sql(
                migrated,
                "INSERT INTO tool_calls (tool, key, args_hash, run_id, actor, args, state, started_at)"
                " VALUES ('watchdog-messenger', ?, '-', ?, 'cp:daemon', '[]', 'succeeded', 'x')",
                [f"msg:{stalled['head']}", stalled["run"]],
            )
            return real()

        monkeypatch.setattr(fake_probe, "me", racing)
        _tick(conn, fake_clock)
        assert _launches(fake_runner) == []

    def test_an_unreadable_started_at_counts_as_due(self, conn, migrated, stalled, fake_clock) -> None:
        sql(
            migrated,
            "INSERT INTO tool_calls (tool, key, args_hash, run_id, actor, args, state, started_at)"
            " VALUES ('watchdog-messenger', ?, '-', ?, 'cp:daemon', '[]', 'succeeded', 'x')",
            [f"msg:{stalled['head']}", stalled["run"]],
        )
        _tick(conn, fake_clock)
        assert len(_open(migrated)) == 1


# --------------------------------------------------------------------------- review fix #01 (F1, F12, F20, F21)


class TestLaunchFailures:
    def test_a_bad_messenger_dir_leaves_no_started_row_and_escalates(
        self, conn, migrated, stalled, fake_runner, fake_clock, monkeypatch, tmp_path
    ) -> None:
        bad = tmp_path / "not-a-dir"
        bad.write_text("x")
        monkeypatch.setenv("QS_CP_MESSENGER_DIR", str(bad))
        _tick(conn, fake_clock)
        assert [r for r in _messenger_rows(migrated) if r["state"] == "started"] == []
        assert _launches(fake_runner) == []
        assert _open(migrated) == [(stalled["run"], "orchestrator_not_listening", stalled["run"])]
        _tick(conn, fake_clock)  # still failing: the alert stays, one occurrence
        assert len(_open(migrated)) == 1 and _messages(migrated) == 1

    def test_a_raising_spawn_fails_the_row_then_escalates(
        self, conn, migrated, stalled, fake_claude, fake_runner, fake_clock, monkeypatch
    ) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("spawn exploded")

        monkeypatch.setattr(fake_claude, "spawn_bg", boom)
        _tick(conn, fake_clock)
        [row] = _messenger_rows(migrated)
        assert row["state"] == "failed" and row["finished_at"]
        assert "spawn exploded" in json.loads(row["result"])["error"]
        assert _open(migrated) == [(stalled["run"], "orchestrator_not_listening", stalled["run"])]
        snap = run_cli("snapshot")[1]
        assert snap["alerts"][0]["kind"] == "orchestrator_not_listening"
        assert snap["alerts"][0]["payload"]["reason"] == "messenger_failed"

    def test_an_aged_started_row_is_reaped_even_with_a_live_holder(
        self, conn, migrated, stalled, fake_runner, fake_clock
    ) -> None:
        sql(
            migrated,
            "INSERT INTO tool_calls (tool, key, args_hash, run_id, actor, args, state, holder_pid, holder_pid_start, started_at)"
            " VALUES ('watchdog-messenger', ?, '-', ?, 'cp:daemon', '[]', 'started', 4242, 'start-4242', ?)",
            [f"msg:{stalled['head']}", stalled["run"], clock.stamp(fake_clock, plus=-watchdog.MESSENGER_TTL_S - 1)],
        )
        _tick(conn, fake_clock)
        assert _messenger_rows(migrated)[0]["state"] == "failed" and len(_open(migrated)) == 1
        assert _launches(fake_runner) == []


class TestReviewFix02:
    def test_unbuildable_messenger_args_claim_nothing_and_escalate(
        self, conn, migrated, stalled, fake_runner, fake_clock, monkeypatch, capsys
    ) -> None:
        def broken() -> str:
            raise KeyError("fast")

        monkeypatch.setattr(watchdog, "_messenger_model", broken)
        _tick(conn, fake_clock)
        assert _messenger_rows(migrated) == [] and _launches(fake_runner) == []
        assert _open(migrated) == [(stalled["run"], "orchestrator_not_listening", stalled["run"])]
        assert capsys.readouterr().err.count("messenger arguments unusable") == 1

    def test_a_failing_beat_after_a_spawn_keeps_the_launch(
        self, conn, migrated, stalled, fake_runner, fake_clock, monkeypatch, capsys
    ) -> None:
        real = watchdog.daemon.beat
        calls: list[int] = []

        def beat(c: Any, k: Any) -> None:
            calls.append(1)
            if len(calls) == 2:  # the beat right after the spawn
                raise RuntimeError("beat failed")
            real(c, k)

        monkeypatch.setattr(watchdog.daemon, "beat", beat)
        _tick(conn, fake_clock)
        [row] = _messenger_rows(migrated)
        assert row["state"] == "succeeded" and "stdout_tail" in json.loads(row["result"])
        err = capsys.readouterr().err
        assert "messenger launch failed" not in err and "the beat after a messenger launch raised" in err


def test_the_messenger_never_inherits_the_session_identity(
    conn, migrated, stalled, fake_runner, fake_clock, monkeypatch
) -> None:
    for name in ("CLAUDE_CODE_SESSION_ID", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_MESSAGING_SOCKET"):
        monkeypatch.setenv(name, "inherited")
    monkeypatch.setenv("QS_CP_KEEP_ME", "1")
    _tick(conn, fake_clock)
    [call] = _launches(fake_runner)
    removed = set(call.env_remove)
    assert {"QS_CP_TOKEN", "CLAUDE_CODE_SESSION_ID", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"} <= removed
    assert "CLAUDE_CODE_MESSAGING_SOCKET" in removed and "QS_CP_KEEP_ME" not in removed


def test_cp_command_falls_back_to_this_python_without_the_main_venv(fake_main) -> None:
    import sys

    seams = activeloop.seams()
    assert watchdog.cp_command(seams) == f"{sys.executable} {fake_main}/scripts/qs/cp.py"
    venv = fake_main / "venv" / "bin" / "python"
    venv.parent.mkdir(parents=True)
    venv.write_text("")
    assert watchdog.cp_command(seams) == f"{venv} {fake_main}/scripts/qs/cp.py"


def test_cp_command_quotes_its_paths(tmp_path: Path, monkeypatch) -> None:
    import shlex
    import sys

    main = tmp_path / "my main"
    seams = activeloop.Seams(runner=None, probe=None, claude=None, main=main, github=None)  # type: ignore[arg-type]
    monkeypatch.setattr(sys, "executable", "/opt/py thon/bin/python")
    assert shlex.split(watchdog.cp_command(seams)) == ["/opt/py thon/bin/python", f"{main}/scripts/qs/cp.py"]
    assert f"`{watchdog.cp_command(seams)} msg pop" in watchdog.wake_text(seams, "R1", 1)
