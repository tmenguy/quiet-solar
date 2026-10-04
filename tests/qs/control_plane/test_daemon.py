"""Checkpoint 4: the daemon skeleton and ``ensure`` (AC7)."""

from __future__ import annotations

import io
import json
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from control_plane import cli, clock, daemon, db, errors, migrations, paths

from .conftest import FakeKill, FakePopen, FakeProbe

V2 = migrations.Migration(2, "test v2", ("CREATE TABLE extra (a INTEGER)",))


def _lease(path: Path) -> dict[str, Any] | None:
    return daemon.read_lease(path)


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
        assert result["migrated"]["to"] == 1
        assert len(seen) == 3 and seen[0][1] is fake_clock
        assert fake_clock.sleeps == [daemon.TICK_S] * 2
        lease = _lease(db_path)
        assert lease is not None and lease["pid"] is None and lease["heartbeat_at"] is None
        assert lease["schema_version"] == 1 and lease["pid_start"].startswith("start-")
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
        bad = migrations.Migration(2, "bad", ("NOT SQL",))
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
        _set_lease(migrated, pid=7, version=1, heartbeat=clock.stamp(fake_clock))
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
                _set_lease(db_path, pid=7, version=1, heartbeat=clock.stamp(fake_clock))
                fake_clock.advance(31)
            elif lease == "cleared":
                _set_lease(db_path, pid=None, version=1, heartbeat=None)
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
        self, migrated, fake_clock, fake_probe: FakeProbe, fake_kill: FakeKill, monkeypatch
    ) -> None:
        _set_lease(migrated, pid=7, version=1, heartbeat=clock.stamp(fake_clock))
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))

        def stopped(pid: int, sig: int) -> None:  # the old daemon exits and clears its lease
            c = sqlite3.connect(migrated)
            c.execute("UPDATE daemon_lease SET pid = NULL, heartbeat_at = NULL")
            c.commit()
            c.close()

        fake_kill.on_call = stopped
        popen = FakePopen(on_call=lambda a, k: daemon.run(fake_clock, probe=fake_probe, db_path=migrated, max_ticks=1))
        res = daemon.ensure(popen=popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill)
        assert res == {"status": "restarted"}
        assert fake_kill.calls == [(7, signal.SIGTERM)]
        assert db.probe_version(migrated) == 2  # the pending migration completed

    def test_restart_waits_then_starts_anyway(
        self, migrated, fake_clock, fake_probe, fake_kill, fake_popen, monkeypatch
    ) -> None:
        _set_lease(migrated, pid=7, version=1, heartbeat=clock.stamp(fake_clock))
        monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, V2))
        fake_probe.kill(7)  # already dead: no signal sent
        res = daemon.ensure(
            popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated, kill=fake_kill, restart_wait_s=2
        )
        assert res == {"status": "restarted"} and fake_kill.calls == []
        assert sum(fake_clock.sleeps) >= 2 and len(fake_popen.calls) == 1

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
        assert invoke("version")[1]["schema_version"] == 1

    def test_ensure_command(self, invoke, fake_popen) -> None:
        code, out = invoke("ensure")
        assert code == 0 and out["status"] == "started" and len(fake_popen.calls) == 1

    def test_daemon_command(self, invoke, monkeypatch, db_path) -> None:
        monkeypatch.setattr(daemon, "IDLE_EXIT_S", 10.0)
        code, out = invoke("daemon")
        assert code == 0 and out["exit"] == "idle" and out["ticks"] == 3
        assert db.probe_version(db_path) == 1


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
            assert c is not None and db.user_version(c) == 1
        assert len(fake_popen.calls) == 1

    def test_real_deps(self) -> None:
        from control_plane import liveness, runner

        from .conftest import REAL_MAKE_DEPS

        d = REAL_MAKE_DEPS()
        assert isinstance(d.clock, clock.SystemClock) and isinstance(d.runner, runner.Runner)
        assert isinstance(d.probe, liveness.ProcessProbe) and isinstance(d.claude, liveness.ClaudeCli)
        assert d.popen is subprocess.Popen
