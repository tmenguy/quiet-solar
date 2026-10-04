"""Checkpoint 6: ``wait`` (AC6)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
from control_plane import clock, wait

from .conftest import ORCH, insert_node, insert_task, open_run, run_cli, sql


def _hook(fake_clock: clock.FakeClock, fn: Callable[[int], None]) -> None:
    count = {"n": 0}
    original = fake_clock.sleep

    def sleep(seconds: float) -> None:
        original(seconds)
        count["n"] += 1
        fn(count["n"])

    fake_clock.sleep = sleep  # type: ignore[method-assign]


def _post(run_id: str, token: str, tmp_path: Path) -> None:
    f = tmp_path / "m.json"
    f.write_text(json.dumps({"x": 1}))
    assert (
        run_cli(
            "msg",
            "post",
            "--run",
            run_id,
            "--to",
            "orchestrator",
            "--kind",
            "k",
            "--payload-file",
            str(f),
            "--token",
            token,
        )[0]
        == 0
    )


def _waiters(path: Path) -> int:
    return sql(path, "SELECT count(*) FROM waiters")[0][0]


@pytest.fixture
def run(migrated: Path, fake_popen) -> tuple[str, str]:
    r = open_run()
    fake_popen.calls.clear()
    return r


def test_stale_token_exits_3_without_side_effects(migrated, run, fake_popen, fake_setup) -> None:
    run_id, token = run
    run_cli("run", "claim", run_id, "--session-id", ORCH)
    fake_popen.calls.clear()
    code, out = run_cli("wait", "--run", run_id, "--token", token)
    assert code == 3 and out["error"] == "STALE_TOKEN"
    assert _waiters(migrated) == 0 and fake_popen.calls == []
    assert fake_setup.calls == []


def test_returns_pending_without_popping(migrated, run, fake_popen, fake_setup, tmp_path) -> None:
    run_id, token = run
    _post(run_id, token, tmp_path)
    code, out = run_cli("wait", "--run", run_id, "--token", token)
    assert (code, out) == (0, {"ok": True, "pending": 1})
    assert sql(migrated, "SELECT state FROM messages")[0][0] == "queued"
    assert _waiters(migrated) == 0
    assert len(fake_popen.calls) == 1  # ensure() before blocking
    assert fake_setup.calls == [("sigterm", wait._on_sigterm)]


def test_waits_until_a_message_arrives(migrated, run, fake_clock, tmp_path) -> None:
    run_id, token = run
    rows: list[int] = []

    def on_sleep(n: int) -> None:
        rows.append(_waiters(migrated))
        if n == 3:
            _post(run_id, token, tmp_path)

    _hook(fake_clock, on_sleep)
    assert run_cli("wait", "--run", run_id, "--token", token, "--poll", "2")[1]["pending"] == 1
    assert rows == [1, 1, 1] and fake_clock.sleeps == [2, 2, 2]
    assert _waiters(migrated) == 0


def test_timeout(migrated, run, fake_clock) -> None:
    run_id, token = run
    code, out = run_cli("wait", "--run", run_id, "--token", token, "--timeout", "3", "--poll", "1")
    assert (code, out) == (0, {"ok": True, "timeout": True})
    assert fake_clock.sleeps == [1, 1, 1]


def test_reensures_a_stale_daemon_at_most_once_per_window(migrated, run, fake_popen) -> None:
    run_id, token = run
    run_cli("wait", "--run", run_id, "--token", token, "--timeout", "95", "--poll", "1")
    assert len(fake_popen.calls) == 4  # t=0, 30, 60, 90


def test_fresh_daemon_is_not_reensured(migrated, run, fake_popen, fake_clock) -> None:
    run_id, token = run

    def beat(n: int = 0) -> None:
        sql(
            migrated,
            "INSERT OR REPLACE INTO daemon_lease (id, pid, schema_version, started_at, heartbeat_at) VALUES (1, 7, 1, 'x', ?)",
            [clock.stamp(fake_clock)],
        )

    beat()
    _hook(fake_clock, beat)
    run_cli("wait", "--run", run_id, "--token", token, "--timeout", "95", "--poll", "1")
    assert fake_popen.calls == []


def test_migration_mid_wait_exits_5_restart_wait(migrated, run, fake_clock) -> None:
    run_id, token = run
    _hook(fake_clock, lambda n: sql(migrated, "PRAGMA user_version = 2"))
    code, out = run_cli("wait", "--run", run_id, "--token", token)
    assert code == 5 and out["restart_wait"] is True
    assert _waiters(migrated) == 0


def test_token_going_stale_mid_wait(migrated, run, fake_clock) -> None:
    run_id, token = run
    _hook(fake_clock, lambda n: run_cli("run", "claim", run_id, "--session-id", ORCH))
    assert run_cli("wait", "--run", run_id, "--token", token)[0] == 3
    assert _waiters(migrated) == 0


def test_sigterm_removes_the_waiter(migrated, run, fake_clock) -> None:
    run_id, token = run
    _hook(fake_clock, lambda n: wait._on_sigterm(15, None))
    with pytest.raises(SystemExit) as exc:
        run_cli("wait", "--run", run_id, "--token", token)
    assert exc.value.code == 143
    assert _waiters(migrated) == 0


def test_cleanup_survives_a_changed_table(migrated, run, fake_clock) -> None:
    run_id, token = run

    def drop(n: int) -> None:
        sql(migrated, "DROP TABLE waiters")
        raise SystemExit(1)

    _hook(fake_clock, drop)
    with pytest.raises(SystemExit):
        run_cli("wait", "--run", run_id, "--token", token)


def test_only_the_run_token_of_that_run(migrated, run) -> None:
    run_id, token = run
    insert_task(migrated, "T1", run_id)
    node = insert_node(migrated, "N1", run_id, "T1")
    assert run_cli("wait", "--run", run_id, "--token", node)[1]["error"] == "CONFLICT"
    open_run("r2", "S-2")
    assert run_cli("wait", "--run", "r2", "--token", token)[1]["error"] == "CONFLICT"


def test_live_waiter(migrated, run, fake_probe) -> None:
    run_id, _ = run
    c = sqlite3.connect(migrated)
    c.row_factory = sqlite3.Row
    assert not wait.live_waiter(c, fake_probe, run_id)
    c.execute(
        "INSERT INTO waiters (run_id, pid, pid_start, started_at, heartbeat_at) VALUES (?, 7, 'start-7', 'x', 'x')",
        (run_id,),
    )
    assert wait.live_waiter(c, fake_probe, run_id)
    fake_probe.kill(7)
    assert not wait.live_waiter(c, fake_probe, run_id)
    c.close()
