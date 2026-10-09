"""Checkpoint 4: the daemon skeleton and ``ensure`` (AC7)."""

from __future__ import annotations

import fcntl
import io
import json
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from control_plane import cli, clock, daemon, db, errors, migrations, paths

from .conftest import CUR, NEXT, FakeKill, FakePopen, FakeProbe

V2 = migrations.Migration(NEXT, "test v2", ("CREATE TABLE extra (a INTEGER)",))


def _lease(path: Path) -> dict[str, Any] | None:
    return daemon.read_lease(path)


class Singleton:
    """The daemon's singleton flock, held as a live old daemon holds it; ``release()`` is that daemon exiting."""

    def __init__(self, db_path: Path) -> None:
        self.fd: int | None = os.open(paths.sidecar(db_path, ".daemon.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


@pytest.fixture
def held(migrated: Path) -> Any:
    lock = Singleton(migrated)
    yield lock
    lock.release()


def _open_run(path: Path) -> None:
    c = db.connect(path)
    with db.write(c):
        c.execute("INSERT INTO runs (id, name, title, state, created_at) VALUES ('R1', 'r', 't', 'open', 'x')")
    c.close()


class TestRun:
    def test_migrates_heartbeats_ticks_and_clears_its_lease(self, db_path, fake_clock, fake_probe, fake_setup) -> None:
        seen: list[tuple[Any, Any]] = []
        result = daemon.run(
            fake_clock, probe=fake_probe, db_path=db_path, tick_hooks=[lambda c, k: seen.append((c, k))], max_ticks=3
        )
        assert result["ticks"] == 3 and result["exit"] == "max_ticks"
        assert result["migrated"]["to"] == CUR
        assert len(seen) == 3 and seen[0][1] is fake_clock
        assert fake_clock.sleeps == [daemon.TICK_S] * 2
        lease = _lease(db_path)
        assert lease is not None and lease["pid"] is None and lease["heartbeat_at"] is None
        assert lease["schema_version"] == CUR and lease["pid_start"].startswith("start-")
        assert fake_setup.calls == [("sigterm", daemon._on_sigterm)]

    def test_second_daemon_exits_held_elsewhere(self, db_path, fake_clock, fake_probe) -> None:
        with db.file_lock(paths.sidecar(db_path, ".daemon.lock"), exclusive=True, timeout=0):
            assert daemon.run(fake_clock, probe=fake_probe, db_path=db_path) == {"singleton": "held_elsewhere"}
        assert not db_path.exists()

    def test_idle_exit(self, db_path, fake_clock, fake_probe) -> None:
        result = daemon.run(fake_clock, probe=fake_probe, db_path=db_path, tick_s=5, idle_exit_s=20)
        assert result["exit"] == "idle" and result["ticks"] == 5

    def test_open_run_keeps_it_alive(self, migrated, fake_clock, fake_probe) -> None:
        _open_run(migrated)
        result = daemon.run(fake_clock, probe=fake_probe, db_path=migrated, idle_exit_s=1, max_ticks=4)
        assert result["exit"] == "max_ticks" and result["ticks"] == 4

    def test_sigterm_and_a_broken_hook(self, db_path, fake_clock, fake_probe, capsys) -> None:
        def broken(c: Any, k: Any) -> None:
            raise RuntimeError("bad hook")

        def stop(c: Any, k: Any) -> None:
            daemon._on_sigterm(signal.SIGTERM, None)

        result = daemon.run(fake_clock, probe=fake_probe, db_path=db_path, tick_hooks=[broken, stop])
        assert result["exit"] == "sigterm" and result["ticks"] == 1
        assert "bad hook" in capsys.readouterr().err
        # a new run starts with a clear stop flag
        assert daemon.run(fake_clock, probe=fake_probe, db_path=db_path, max_ticks=1)["exit"] == "max_ticks"

    def test_failing_migration_writes_the_sidecar(self, migrated, fake_clock, fake_probe, monkeypatch) -> None:
        bad = migrations.Migration(NEXT, "bad", ("NOT SQL",))
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, bad))
        with pytest.raises(errors.CpError) as exc:
            daemon.run(fake_clock, probe=fake_probe, db_path=migrated)
        assert exc.value.code == "INTERNAL" and exc.value.exit_code == 1
        side = json.loads(paths.sidecar(migrated, ".migrate-error.json").read_text())
        assert side["at"] == clock.iso(fake_clock.now()) and "OperationalError" in side["error"]
        # success removes it
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS[:-1], V2))
        daemon.run(fake_clock, probe=fake_probe, db_path=migrated, max_ticks=1)
        assert not paths.sidecar(migrated, ".migrate-error.json").exists()

    def test_refusal_against_a_missing_live_db(self, fake_main, fake_clock, fake_probe, monkeypatch) -> None:
        monkeypatch.setattr(paths, "main_head_branch", lambda m: None)
        live = fake_main / "harness_state.db"
        with pytest.raises(errors.CpError):
            daemon.run(fake_clock, probe=fake_probe, db_path=live)
        side = json.loads(paths.sidecar(live, ".migrate-error.json").read_text())
        assert "POLICY_REFUSED" in side["error"]
        assert not live.exists()

    def test_lease_is_cleared_only_if_still_ours(self, db_path, fake_clock, fake_probe) -> None:
        def steal(c: sqlite3.Connection, k: Any) -> None:
            with db.write(c):
                c.execute("UPDATE daemon_lease SET pid = 4242")

        daemon.run(fake_clock, probe=fake_probe, db_path=db_path, tick_hooks=[steal], max_ticks=1)
        assert _lease(db_path)["pid"] == 4242

    def test_selects_the_db_by_default(self, db_path, fake_clock, fake_probe) -> None:
        assert daemon.run(fake_clock, probe=fake_probe, max_ticks=1)["ticks"] == 1
        assert db_path.exists()


def _set_lease(path: Path, *, pid: int | None, version: int, heartbeat: str | None) -> None:
    c = db.connect(path)
    c.execute(
        "INSERT OR REPLACE INTO daemon_lease (id, pid, pid_start, schema_version, started_at, heartbeat_at)"
        " VALUES (1, ?, ?, ?, 'x', ?)",
        (pid, None if pid is None else f"start-{pid}", version, heartbeat),
    )
    c.close()


def _ensure(**kw: Any) -> dict[str, Any]:
    return daemon.ensure(**kw)


class TestEnsure:
    def test_fresh_lease_same_version_is_a_noop(self, migrated, fake_clock, fake_probe, fake_popen) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(10)
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated)
        assert res == {"status": "already_running", "pid": 7}
        assert fake_popen.calls == []

    @pytest.mark.parametrize("lease", ["missing-db", "no-table", "absent", "stale", "cleared"])
    def test_starts_a_detached_daemon(self, lease, db_path, fake_clock, fake_probe, fake_popen, fake_main) -> None:
        if lease == "no-table":
            sqlite3.connect(db_path).close()
        elif lease != "missing-db":
            migrations.migrate(db_path, role="test")
            if lease == "stale":
                _set_lease(db_path, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
                fake_clock.advance(31)
                fake_probe.kill(7)  # a stale lease of a dead daemon (alive and stale: G18)
            elif lease == "cleared":
                _set_lease(db_path, pid=None, version=CUR, heartbeat=None)
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=db_path)
        assert res == {"status": "started"}
        [(argv, kwargs)] = fake_popen.calls
        assert argv == [sys.executable, str(fake_main / "scripts" / "qs" / "cp.py"), "daemon"]
        assert kwargs["start_new_session"] is True and kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["cwd"] == str(fake_main) and kwargs["close_fds"] is True
        assert kwargs["stdout"].name == str(fake_main / "harness_state.daemon.log")
        assert kwargs["stderr"] is kwargs["stdout"]

    def test_uses_the_main_venv_python(self, db_path, fake_clock, fake_probe, fake_popen, fake_main) -> None:
        venv_python = fake_main / "venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("")
        daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=db_path)
        assert fake_popen.calls[0][0][0] == str(venv_python)

    def test_corrupt_file_reads_as_stale(self, db_path, fake_clock, fake_probe, fake_popen) -> None:
        db_path.write_text("not a database")
        assert daemon.read_lease(db_path) is None
        db_path.unlink()
        db_path.mkdir()  # cannot even be opened
        assert daemon.read_lease(db_path) is None

    def test_lower_schema_daemon_is_restarted_and_migrates(
        self, held, migrated, fake_clock, fake_probe: FakeProbe, fake_kill: FakeKill, monkeypatch
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))

        def stopped(pid: int, sig: int) -> None:  # the old daemon exits: clears its lease, drops its flock
            c = sqlite3.connect(migrated)
            c.execute("UPDATE daemon_lease SET pid = NULL, heartbeat_at = NULL")
            c.commit()
            c.close()
            held.release()

        fake_kill.on_call = stopped
        popen = FakePopen(on_call=lambda a, k: daemon.run(fake_clock, probe=fake_probe, db_path=migrated, max_ticks=1))
        res = daemon.ensure(popen=popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "restarted"}
        assert fake_kill.calls == [(7, signal.SIGTERM)]
        assert db.probe_version(migrated) == NEXT  # the pending migration completed

    def test_restart_waits_then_starts_anyway(
        self, migrated, fake_clock, fake_probe, fake_kill, fake_popen, monkeypatch
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))
        fake_probe.kill(7)  # already dead: no signal sent
        res = daemon.ensure(
            popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill, restart_wait_s=2
        )
        assert res == {"status": "restarted"} and fake_kill.calls == []
        assert fake_clock.sleeps == [] and len(fake_popen.calls) == 1  # G3: a dead old daemon is not waited for

    def test_migrate_error_backoff(self, migrated, fake_clock, fake_probe, fake_popen) -> None:
        side = paths.sidecar(migrated, ".migrate-error.json")
        side.write_text(json.dumps({"at": clock.stamp(fake_clock), "error": "boom"}))
        fake_clock.advance(10)
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated)
        assert res == {"status": "migrate_failed", "error": "boom"} and fake_popen.calls == []
        fake_clock.advance(300)
        assert (
            daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated)["status"] == "started"
        )
        side.write_text(json.dumps({"error": "no at"}))
        assert (
            daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated)["status"] == "started"
        )

    def test_live_db_from_non_main_code_is_refused(
        self, fake_main, tmp_path, fake_clock, fake_probe, fake_popen, monkeypatch
    ) -> None:
        monkeypatch.setattr(paths, "code_root", lambda: tmp_path / "wt")
        with pytest.raises(errors.CpError) as exc:
            daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=fake_main / "harness_state.db")
        assert exc.value.code == "POLICY_REFUSED" and fake_popen.calls == []

    def test_real_popen_through_the_argv_seam(self, db_path, fake_clock, fake_probe) -> None:
        launched: list[subprocess.Popen[bytes]] = []

        def popen(argv: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
            assert kwargs["start_new_session"] is True and kwargs["stdin"] is subprocess.DEVNULL
            proc = subprocess.Popen(argv, **kwargs)
            launched.append(proc)
            return proc

        res = daemon.ensure(
            popen=popen, clock=fake_clock, probe=fake_probe, db_path=db_path, argv=[sys.executable, "-c", "pass"]
        )
        assert res == {"status": "started"}
        assert launched[0].wait(timeout=10) == 0

    def test_selects_the_db_by_default(self, fake_clock, fake_probe, fake_popen) -> None:
        assert daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe)["status"] == "started"


class TestCli:
    def test_version_has_the_schema(self, invoke) -> None:
        assert invoke("version")[1]["schema_version"] == CUR

    def test_ensure_command(self, invoke, fake_popen) -> None:
        code, out = invoke("ensure")
        assert code == 0 and out["status"] == "started" and len(fake_popen.calls) == 1

    def test_daemon_command(self, invoke, monkeypatch, db_path) -> None:
        monkeypatch.setattr(daemon, "IDLE_EXIT_S", 10.0)
        code, out = invoke("daemon")
        assert code == 0 and out["exit"] == "idle" and out["ticks"] == 3
        assert db.probe_version(db_path) == CUR


class TestConnection:
    def _io(self, deps) -> cli.Io:
        return cli.Io(stdin=io.StringIO(), stdout=io.StringIO(), deps=deps)

    def test_read_on_a_missing_db(self, deps, fake_popen) -> None:
        with cli.connection(self._io(deps), "read") as c:
            assert c is None
        assert fake_popen.calls == []

    def test_read_is_read_only(self, deps, migrated) -> None:
        with cli.connection(self._io(deps), "read") as c:
            assert c is not None
            with pytest.raises(sqlite3.OperationalError):
                c.execute("INSERT INTO meta (key, value) VALUES ('a', 'b')")

    def test_write_waits_for_the_daemon(self, deps, fake_popen, db_path) -> None:
        fake_popen.on_call = lambda a, k: migrations.migrate(db_path, role="test")
        with cli.connection(self._io(deps), "write") as c:
            assert c is not None and db.user_version(c) == CUR
        assert len(fake_popen.calls) == 1

    def test_real_deps(self) -> None:
        from control_plane import liveness, runner

        from .conftest import REAL_MAKE_DEPS

        d = REAL_MAKE_DEPS()
        assert isinstance(d.clock, clock.SystemClock) and isinstance(d.runner, runner.Runner)
        assert isinstance(d.probe, liveness.ProcessProbe) and isinstance(d.claude, liveness.ClaudeCli)
        assert d.popen is subprocess.Popen


# --------------------------------------------------------------------------- review fix #01 (F6, F7, F8, F9)


class TestReviewFix01:
    def test_starts_on_a_feature_branch_when_no_migration_is_needed(
        self, fake_main, fake_clock, fake_probe, monkeypatch
    ) -> None:
        """F6: a current live DB needs no authorisation; the daemon starts."""
        live = fake_main / "harness_state.db"
        migrations.migrate(live, role="daemon")
        monkeypatch.setattr(paths, "main_head_branch", lambda m: None)
        result = daemon.run(fake_clock, probe=fake_probe, db_path=live, max_ticks=1)
        assert result["migrated"]["result"] == "noop" and result["ticks"] == 1
        assert not paths.sidecar(live, ".migrate-error.json").exists()

    def test_a_busy_migration_is_retried_without_the_sidecar(
        self, migrated, fake_clock, fake_probe, monkeypatch, capsys
    ) -> None:
        """F7: contention never becomes a MIGRATE_BACKOFF_S outage."""
        real = migrations.migrate
        calls = {"n": 0}

        def flaky(path: Path, *, role: str) -> dict[str, Any]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise errors.CpError("BUSY", "database busy")
            return real(path, role=role)

        monkeypatch.setattr(migrations, "migrate", flaky)
        result = daemon.run(fake_clock, probe=fake_probe, db_path=migrated, max_ticks=1)
        assert result["ticks"] == 1 and calls["n"] == 2
        assert not paths.sidecar(migrated, ".migrate-error.json").exists()
        assert "busy" in capsys.readouterr().err

    def test_a_persistently_busy_migration_exits_busy_without_the_sidecar(
        self, migrated, fake_clock, fake_probe, monkeypatch
    ) -> None:
        def busy(path: Path, *, role: str) -> dict[str, Any]:
            raise errors.CpError("BUSY", "database busy")

        monkeypatch.setattr(migrations, "migrate", busy)
        with pytest.raises(errors.CpError) as exc:
            daemon.run(fake_clock, probe=fake_probe, db_path=migrated)
        assert exc.value.code == "BUSY"
        assert not paths.sidecar(migrated, ".migrate-error.json").exists()
        assert fake_clock.sleeps == [daemon.TICK_S] * daemon.MIGRATE_BUSY_RETRIES

    def test_an_auto_rolled_back_step_lands_in_the_sidecar(self, migrated, fake_clock, fake_probe, monkeypatch) -> None:
        rb = migrations.Migration(
            NEXT,
            "rb",
            ("CREATE TABLE t (x UNIQUE)", "INSERT INTO t VALUES (1)", "INSERT OR ROLLBACK INTO t VALUES (1)"),
        )
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, rb))
        with pytest.raises(errors.CpError):
            daemon.run(fake_clock, probe=fake_probe, db_path=migrated)
        side = json.loads(paths.sidecar(migrated, ".migrate-error.json").read_text())
        assert "IntegrityError" in side["error"]

    def test_a_busy_heartbeat_does_not_kill_the_daemon(
        self, migrated, fake_clock, fake_probe, monkeypatch, capsys
    ) -> None:
        """F8: log it and go on to the next tick."""
        monkeypatch.setattr(db, "BUSY_TIMEOUT_MS", 20)
        holder = sqlite3.connect(migrated, isolation_level=None)
        ticks = {"n": 0}

        def contend(c: sqlite3.Connection, k: Any) -> None:
            ticks["n"] += 1
            if ticks["n"] == 1:
                holder.execute("BEGIN IMMEDIATE")  # the next heartbeat hits BUSY
            elif ticks["n"] == 2:
                holder.execute("ROLLBACK")

        try:
            result = daemon.run(fake_clock, probe=fake_probe, db_path=migrated, tick_hooks=[contend], max_ticks=3)
        finally:
            if holder.in_transaction:
                holder.execute("ROLLBACK")
            holder.close()
        assert result["ticks"] == 3
        assert "heartbeat" in capsys.readouterr().err

    def test_a_stale_but_alive_older_daemon_is_terminated(
        self, held, migrated, fake_clock, fake_probe, fake_kill, monkeypatch
    ) -> None:
        """F9: SIGTERM whenever the older daemon is alive, fresh or not."""
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(120)  # stale: hung, but alive
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))

        def stopped(pid: int, sig: int) -> None:
            c = sqlite3.connect(migrated)
            c.execute("UPDATE daemon_lease SET pid = NULL, heartbeat_at = NULL")
            c.commit()
            c.close()

        fake_kill.on_call = stopped
        popen = FakePopen()
        res = daemon.ensure(popen=popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "restarted"} and fake_kill.calls == [(7, signal.SIGTERM)]
        assert len(popen.calls) == 1

    def test_an_old_daemon_still_alive_after_the_deadline_is_restart_pending(
        self, held, migrated, fake_clock, fake_probe, fake_kill, fake_popen, monkeypatch
    ) -> None:
        """F9: never spawn a new daemon that would exit at once on the singleton lock."""
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(daemon.STALE_AFTER_S)  # J4: a beat that will be stale at the deadline
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))
        res = daemon.ensure(
            popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill, restart_wait_s=2
        )
        assert res == {"status": "restart_pending", "pid": 7}
        assert fake_kill.calls == [(7, signal.SIGTERM), (7, signal.SIGKILL)] and fake_popen.calls == []  # G4

    def test_unknown_liveness_sends_no_signal_and_waits(
        self, held, migrated, fake_clock, fake_probe, fake_kill, fake_popen, monkeypatch
    ) -> None:
        """F4 + F9: the pid exists but its start time is unknown — maybe reused: no SIGTERM, no spawn."""
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(120)
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))
        monkeypatch.setattr(fake_probe, "alive", lambda pid, start: None)
        res = daemon.ensure(
            popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill, restart_wait_s=2
        )
        assert res == {"status": "restart_pending", "pid": 7}
        assert fake_kill.calls == [] and fake_popen.calls == []

    def test_a_stale_dead_older_daemon_is_just_started_over(
        self, migrated, fake_clock, fake_probe, fake_kill, fake_popen, monkeypatch
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(120)
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))
        fake_probe.kill(7)
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "started"} and fake_kill.calls == [] and fake_clock.sleeps == []
        assert len(fake_popen.calls) == 1

    def test_a_heartbeat_on_a_migrated_db_still_stops_the_daemon(self, migrated, fake_clock, fake_probe) -> None:
        def migrate_away(c: sqlite3.Connection, k: Any) -> None:
            sqlite3.connect(migrated).execute(f"PRAGMA user_version = {NEXT}").connection.close()

        with pytest.raises(errors.CpError) as exc:
            daemon.run(fake_clock, probe=fake_probe, db_path=migrated, tick_hooks=[migrate_away], max_ticks=3)
        assert exc.value.code == "SCHEMA_TOO_NEW"


# --------------------------------------------------------------------------- review fix #02 (G3, G4, G13, G18)


def _clear_lease(path: Path, pid: int | None = None) -> None:
    c = sqlite3.connect(path)
    c.execute("UPDATE daemon_lease SET pid = ?, heartbeat_at = NULL", (pid,))
    c.commit()
    c.close()


def _no_start(path: Path) -> None:
    c = sqlite3.connect(path)
    c.execute("UPDATE daemon_lease SET pid_start = NULL")
    c.commit()
    c.close()


def _beat(path: Path, stamp: str) -> None:
    c = sqlite3.connect(path)
    c.execute("UPDATE daemon_lease SET heartbeat_at = ?", (stamp,))
    c.commit()
    c.close()


class TestReviewFix02:
    @pytest.fixture
    def v2(self, monkeypatch) -> None:
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))

    # G3
    def test_a_stale_lease_without_pid_start_is_never_signalled(
        self, migrated, fake_clock, fake_probe, fake_kill, fake_popen, v2
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        _no_start(migrated)
        fake_clock.advance(120)
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "started"} and fake_kill.calls == [] and fake_clock.sleeps == []
        assert len(fake_popen.calls) == 1

    def test_a_fresh_lease_without_pid_start_is_signalled(
        self, held, migrated, fake_clock, fake_probe, fake_kill, fake_popen, v2
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        _no_start(migrated)
        fake_kill.on_call = lambda pid, sig: _clear_lease(migrated)
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "restarted"} and fake_kill.calls == [(7, signal.SIGTERM)]

    def test_a_dead_old_daemon_is_not_waited_for(
        self, migrated, fake_clock, fake_probe, fake_kill, fake_popen, v2
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))  # fresh, but proven dead
        fake_probe.kill(7)
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "restarted"} and fake_kill.calls == [] and fake_clock.sleeps == []
        assert len(fake_popen.calls) == 1

    def test_the_wait_ends_when_another_daemon_took_the_lease(
        self, held, migrated, fake_clock, fake_probe, fake_kill, fake_popen, v2
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_kill.on_call = lambda pid, sig: _clear_lease(migrated, pid=99)
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "restarted"} and fake_clock.sleeps == [] and len(fake_popen.calls) == 1

    def test_a_vanished_pid_at_sigterm_is_fine(
        self, held, migrated, fake_clock, fake_probe, fake_kill, fake_popen, v2
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))

        def gone(pid: int, sig: int) -> None:
            raise ProcessLookupError(pid)

        fake_kill.on_call = gone
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "restarted"} and fake_clock.sleeps == [] and len(fake_popen.calls) == 1

    def test_the_daemon_retries_its_start_time_then_starts(self, db_path, fake_clock, fake_probe, monkeypatch) -> None:
        from control_plane import liveness

        answers = iter([None, None, "start-1"])
        monkeypatch.setattr(fake_probe, "me", lambda: liveness.Holder(1, next(answers), 1))
        result = daemon.run(fake_clock, probe=fake_probe, db_path=db_path, max_ticks=1)
        assert result["ticks"] == 1 and fake_clock.sleeps[:2] == [daemon.PID_START_RETRY_S] * 2
        assert _lease(db_path)["pid_start"] == "start-1"

    def test_the_daemon_refuses_to_run_without_a_start_time(
        self, db_path, fake_clock, fake_probe, monkeypatch, capsys
    ) -> None:
        from control_plane import liveness

        monkeypatch.setattr(fake_probe, "me", lambda: liveness.Holder(1, None, 1))
        result = daemon.run(fake_clock, probe=fake_probe, db_path=db_path, max_ticks=1)
        assert result == {"ticks": 0, "exit": "no_pid_start"}
        assert _lease(db_path) is None and "start time" in capsys.readouterr().err

    # G4
    def test_a_hung_old_daemon_is_killed_then_replaced(
        self, held, migrated, fake_clock, fake_probe, fake_kill, fake_popen, v2
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(daemon.STALE_AFTER_S)  # J4: a beat that will be stale at the deadline

        def on_kill(pid: int, sig: int) -> None:
            if sig == signal.SIGKILL:
                fake_probe.kill(7)  # SIGKILL always works; SIGTERM is ignored by the hung daemon

        fake_kill.on_call = on_kill
        res = daemon.ensure(
            popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill, restart_wait_s=2
        )
        assert res == {"status": "restarted"}
        assert fake_kill.calls == [(7, signal.SIGTERM), (7, signal.SIGKILL)] and len(fake_popen.calls) == 1

    def test_a_kill_that_does_not_take_is_restart_pending(
        self, held, migrated, fake_clock, fake_probe, fake_kill, fake_popen, v2
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(daemon.STALE_AFTER_S)  # J4: a beat that will be stale at the deadline
        res = daemon.ensure(
            popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill, restart_wait_s=2
        )
        assert res == {"status": "restart_pending", "pid": 7}
        assert fake_kill.calls == [(7, signal.SIGTERM), (7, signal.SIGKILL)] and fake_popen.calls == []

    def test_a_kill_on_a_vanished_pid_starts_the_new_daemon(
        self, held, migrated, fake_clock, fake_probe, fake_kill, fake_popen, v2
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(daemon.STALE_AFTER_S)  # J4: a beat that will be stale at the deadline

        def on_kill(pid: int, sig: int) -> None:
            if sig == signal.SIGKILL:
                raise ProcessLookupError(pid)

        fake_kill.on_call = on_kill
        res = daemon.ensure(
            popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill, restart_wait_s=2
        )
        assert res == {"status": "restarted"} and len(fake_popen.calls) == 1

    def test_an_old_daemon_still_beating_is_not_killed(
        self, held, migrated, fake_clock, fake_probe, fake_kill, fake_popen, v2
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_kill.on_call = lambda pid, sig: _beat(migrated, "2099-01-01T00:00:00.000000Z")
        res = daemon.ensure(
            popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill, restart_wait_s=2
        )
        assert res == {"status": "restart_pending", "pid": 7} and fake_kill.calls == [(7, signal.SIGTERM)]

    def test_an_old_daemon_dead_at_the_deadline_is_replaced(
        self, held, migrated, fake_clock, fake_probe, fake_kill, fake_popen, v2, monkeypatch
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        answers = iter([True, False])
        monkeypatch.setattr(fake_probe, "alive", lambda pid, start: next(answers))
        res = daemon.ensure(
            popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill, restart_wait_s=2
        )
        assert res == {"status": "restarted"} and fake_kill.calls == [(7, signal.SIGTERM)]

    def test_a_lease_unreadable_during_the_wait_ends_it(
        self, held, migrated, fake_clock, fake_probe, fake_kill, fake_popen, v2, monkeypatch
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        real = daemon.read_lease
        answers = iter([real(migrated), None])
        monkeypatch.setattr(daemon, "read_lease", lambda path: next(answers))
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "restarted"} and len(fake_popen.calls) == 1

    # G18
    def test_a_stale_alive_same_schema_daemon_is_killed_then_replaced(
        self, held, migrated, fake_clock, fake_probe, fake_kill, fake_popen
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(120)

        def on_kill(pid: int, sig: int) -> None:
            if sig == signal.SIGKILL:
                fake_probe.kill(7)

        fake_kill.on_call = on_kill
        res = daemon.ensure(
            popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill, restart_wait_s=2
        )
        assert res == {"status": "restarted"} and fake_kill.calls == [(7, signal.SIGTERM), (7, signal.SIGKILL)]
        assert len(fake_popen.calls) == 1

    def test_a_stale_alive_same_schema_daemon_that_resumes_is_stale_alive(
        self, held, migrated, fake_clock, fake_probe, fake_kill, fake_popen
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(120)
        fake_kill.on_call = lambda pid, sig: _beat(migrated, "2099-01-01T00:00:00.000000Z")
        res = daemon.ensure(
            popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill, restart_wait_s=2
        )
        assert res == {"status": "stale_alive", "pid": 7} and fake_popen.calls == []

    def test_a_stale_same_schema_daemon_of_unknown_liveness_is_just_started_over(
        self, migrated, fake_clock, fake_probe, fake_kill, fake_popen, monkeypatch
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(120)
        monkeypatch.setattr(fake_probe, "alive", lambda pid, start: None)
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "started"} and fake_kill.calls == []

    # G13
    def test_the_final_lease_clear_never_masks_the_result(
        self, migrated, fake_clock, fake_probe, capsys, monkeypatch
    ) -> None:
        def migrate_away(c: sqlite3.Connection, k: Any) -> None:
            sqlite3.connect(migrated).execute(f"PRAGMA user_version = {NEXT}").connection.close()
            monkeypatch.setattr(daemon, "beat", lambda conn, clock: None)  # only the final clear meets v2

        result = daemon.run(fake_clock, probe=fake_probe, db_path=migrated, tick_hooks=[migrate_away], max_ticks=1)
        assert result["exit"] == "max_ticks"
        assert "lease" in capsys.readouterr().err


# --------------------------------------------------------------------------- review fix #03 (H2)


class TestReviewFix03:
    @pytest.fixture
    def v2(self, monkeypatch) -> None:
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))

    # (a)
    def test_a_newer_schema_daemon_is_never_signalled(
        self, migrated, held, fake_clock, fake_probe, fake_kill, fake_popen
    ) -> None:
        _set_lease(migrated, pid=7, version=NEXT, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(120)  # stale and alive: newer code, never ours to stop
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "newer_running", "pid": 7} and fake_kill.calls == [] and fake_clock.sleeps == []
        assert fake_popen.calls == [] and not paths.sidecar(migrated, ".migrate-error.json").exists()  # J5

    # (b)
    def test_the_lease_is_beaten_around_every_tick_hook(self, migrated, fake_clock, fake_probe) -> None:
        seen: list[tuple[str, str | None, str]] = []

        def slow(c: sqlite3.Connection, k: Any) -> None:
            seen.append(("slow", _lease(migrated)["heartbeat_at"], clock.stamp(k)))
            k.advance(daemon.STALE_AFTER_S - 5)  # each hook within the window; the tick as a whole is not

        def check(c: sqlite3.Connection, k: Any) -> None:
            seen.append(("check", _lease(migrated)["heartbeat_at"], clock.stamp(k)))

        daemon.run(fake_clock, probe=fake_probe, db_path=migrated, tick_hooks=[slow, slow, check], max_ticks=1)
        assert [beat == now for _, beat, now in seen] == [True, True, True]

    def test_a_daemon_inside_long_hooks_stays_fresh_for_ensure(self, migrated, fake_clock, fake_probe) -> None:
        kill_calls: list[tuple[int, int]] = []
        results: list[dict[str, Any]] = []

        def slow(c: sqlite3.Connection, k: Any) -> None:
            k.advance(daemon.STALE_AFTER_S - 5)

        def probe_ensure(c: sqlite3.Connection, k: Any) -> None:
            results.append(
                daemon.ensure(
                    popen=FakePopen(),
                    clock=k,
                    probe=fake_probe,
                    db_path=migrated,
                    kill=lambda pid, sig: kill_calls.append((pid, sig)),
                )
            )

        daemon.run(fake_clock, probe=fake_probe, db_path=migrated, tick_hooks=[slow, slow, probe_ensure], max_ticks=1)
        assert results[0]["status"] == "already_running" and kill_calls == []

    def test_a_hook_may_beat_the_lease_itself(self, migrated, fake_clock, fake_probe) -> None:
        stamps: list[str | None] = []

        def long_hook(c: sqlite3.Connection, k: Any) -> None:
            k.advance(daemon.STALE_AFTER_S - 5)
            daemon.beat(c, k)
            stamps.append(_lease(migrated)["heartbeat_at"])
            stamps.append(clock.stamp(k))

        daemon.run(fake_clock, probe=fake_probe, db_path=migrated, tick_hooks=[long_hook], max_ticks=1)
        assert stamps[0] == stamps[1]

    # (c)
    def test_a_zombie_old_daemon_is_replaced_without_any_signal(
        self, migrated, fake_clock, fake_probe, fake_kill, fake_popen, v2
    ) -> None:
        """The probe says alive (a zombie), but nothing holds the singleton flock: it is gone."""
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "restarted"} and fake_kill.calls == [] and fake_clock.sleeps == []
        assert len(fake_popen.calls) == 1

    def test_a_zombie_stale_same_schema_daemon_is_replaced_without_any_signal(
        self, migrated, fake_clock, fake_probe, fake_kill, fake_popen
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(120)
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "restarted"} and fake_kill.calls == [] and len(fake_popen.calls) == 1

    def test_the_flock_freed_after_sigterm_ends_the_wait(
        self, migrated, held, fake_clock, fake_probe, fake_kill, fake_popen, v2
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_kill.on_call = lambda pid, sig: held.release()  # exited; its parent has not reaped it yet
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "restarted"} and fake_kill.calls == [(7, signal.SIGTERM)]
        assert fake_clock.sleeps == []

    def test_a_proven_dead_daemon_after_sigterm_ends_the_wait(
        self, migrated, held, fake_clock, fake_probe, fake_kill, fake_popen, v2
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_kill.on_call = lambda pid, sig: fake_probe.kill(7)  # died without clearing its lease
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "restarted"} and fake_clock.sleeps == []

    def test_the_flock_freed_after_sigkill_ends_the_wait(
        self, migrated, held, fake_clock, fake_probe, fake_kill, fake_popen, v2
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(daemon.STALE_AFTER_S)  # J4: a beat that will be stale at the deadline

        def on_kill(pid: int, sig: int) -> None:
            if sig == signal.SIGKILL:
                held.release()  # killed: a zombie until its parent reaps it, but its flock is gone

        fake_kill.on_call = on_kill
        res = daemon.ensure(
            popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill, restart_wait_s=2
        )
        assert res == {"status": "restarted"} and fake_kill.calls == [(7, signal.SIGTERM), (7, signal.SIGKILL)]
        assert len(fake_popen.calls) == 1

    def test_unknown_liveness_at_the_deadline_is_never_killed(
        self, migrated, held, fake_clock, fake_probe, fake_kill, fake_popen, v2, monkeypatch
    ) -> None:
        """SIGKILL needs proof: the flock held is not enough (a newer daemon may hold it before its lease)."""
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        answers = iter([True])
        monkeypatch.setattr(fake_probe, "alive", lambda pid, start: next(answers, None))
        res = daemon.ensure(
            popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill, restart_wait_s=2
        )
        assert res == {"status": "restart_pending", "pid": 7} and fake_kill.calls == [(7, signal.SIGTERM)]


class TestReviewFix04:
    @pytest.fixture
    def v2(self, monkeypatch) -> None:
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))

    # J4
    def test_an_older_daemon_with_a_young_beat_at_the_deadline_is_not_killed(
        self, migrated, held, fake_clock, fake_probe, fake_kill, fake_popen, v2
    ) -> None:
        """Inside a tick hook (within STALE_AFTER_S): healthy, so SIGTERM only, never SIGKILL."""
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        res = daemon.ensure(
            popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill, restart_wait_s=2
        )
        assert res == {"status": "restart_pending", "pid": 7} and fake_kill.calls == [(7, signal.SIGTERM)]
        assert fake_popen.calls == []

    def test_an_older_daemon_with_a_beat_stale_at_the_deadline_is_killed(
        self, migrated, held, fake_clock, fake_probe, fake_kill, fake_popen, v2
    ) -> None:
        _set_lease(migrated, pid=7, version=CUR, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(daemon.STALE_AFTER_S - 2)  # the beat turns STALE_AFTER_S old exactly at the deadline
        res = daemon.ensure(
            popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill, restart_wait_s=2
        )
        assert res == {"status": "restart_pending", "pid": 7}
        assert fake_kill.calls == [(7, signal.SIGTERM), (7, signal.SIGKILL)]

    # J5
    def test_a_stale_live_newer_daemon_is_newer_running_without_a_spawn(
        self, migrated, held, fake_clock, fake_probe, fake_kill, fake_popen
    ) -> None:
        _set_lease(migrated, pid=7, version=NEXT, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(120)
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "newer_running", "pid": 7}
        assert fake_kill.calls == [] and fake_popen.calls == [] and fake_clock.sleeps == []

    def test_a_stale_dead_newer_daemon_is_started_over(
        self, migrated, fake_clock, fake_probe, fake_kill, fake_popen
    ) -> None:
        _set_lease(migrated, pid=7, version=NEXT, heartbeat=clock.stamp(fake_clock))
        fake_clock.advance(120)
        fake_probe.kill(7)
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "started"} and fake_kill.calls == [] and len(fake_popen.calls) == 1
