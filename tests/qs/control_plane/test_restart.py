"""QS-406 T6a: the code version, the restart on new code, the env strip (§4.1, D7, D8, D16, AC 2, AC 4, AC 5)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import control_plane
import pytest
from control_plane import cli, clock, codever, daemon, db, liveness, paths, runner, selfcheck, ticks

from .conftest import CUR, SCRIPTS_QS, FakePopen, open_run, run_cli, sql


def _cp_file(main: Path, name: str = "a.py", text: str = "x = 1\n") -> Path:
    f = main / "scripts" / "qs" / "control_plane" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    return f


def _grow(f: Path) -> None:
    f.write_text(f.read_text() + "# changed\n")  # a size change, whatever the mtime resolution


def _editor(f: Path, at_call: int = 1) -> Any:
    calls = {"n": 0}

    def hook(conn: Any, clk: clock.Clock) -> None:
        calls["n"] += 1
        if calls["n"] == at_call:
            _grow(f)

    return hook


# --------------------------------------------------------------------------- codever (D8, AC 2)


class TestCodeVersion:
    def test_a_missing_file_is_hashed_as_absent(self, fake_main: Path) -> None:
        bare = codever.code_version(fake_main)  # no scripts/ at all: never raises
        (fake_main / "scripts" / "qs").mkdir(parents=True)
        (fake_main / "scripts" / "qs" / "cp.py").write_text("")
        assert codever.code_version(fake_main) != bare

    def test_cached_on_the_signature(self, fake_main: Path, monkeypatch) -> None:
        f = _cp_file(fake_main)
        v = codever.code_version(fake_main)
        reads: list[Path] = []
        real = Path.read_bytes
        monkeypatch.setattr(Path, "read_bytes", lambda self: reads.append(self) or real(self))
        assert codever.code_version(fake_main) == v and reads == []
        _grow(f)
        assert codever.code_version(fake_main) != v and reads

    def test_hashed_files(self, fake_main: Path) -> None:
        f = _cp_file(fake_main, "sub/b.py")
        qs = fake_main / "scripts" / "qs"
        assert codever.hashed_files(fake_main) == sorted([f, qs / "cp.py", qs / "models.py", qs / "targets.py"])

    def test_the_hashed_set_covers_every_loaded_module(self) -> None:
        """In a real ``python -I`` process: the loaded ``scripts/qs`` modules ⊆ the D8 set (relative paths)."""
        code = f"""
import json, pathlib, sys
qs = pathlib.Path({str(SCRIPTS_QS)!r}).resolve()
sys.path.insert(0, str(qs))
import control_plane.cli
from control_plane import activeloop, codever, watchdog
activeloop.register_builtin()
watchdog._messenger_model()  # the lazy `import models` (QS-405 D6): loaded at run time, so load it here too
assert "models" in sys.modules
loaded = set()
for m in list(sys.modules.values()):
    f = getattr(m, "__file__", None)
    if f and pathlib.Path(f).resolve().is_relative_to(qs):
        loaded.add(pathlib.Path(f).resolve().relative_to(qs).as_posix())
hashed = [p.resolve().relative_to(qs).as_posix() for p in codever.hashed_files(qs.parents[1])]
print(json.dumps({{"loaded": sorted(loaded), "hashed": sorted(hashed)}}))
"""
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
        sets = json.loads(out.stdout)
        assert set(sets["loaded"]) <= set(sets["hashed"]), set(sets["loaded"]) - set(sets["hashed"])
        assert {h for h in sets["hashed"] if not h.startswith("control_plane/")} == {"cp.py", "models.py", "targets.py"}

    def test_git_busy(self, fake_main: Path) -> None:
        assert codever.git_busy(fake_main) is None
        (fake_main / ".git" / "MERGE_HEAD").write_text("x")
        assert codever.git_busy(fake_main) == str(fake_main / ".git" / "MERGE_HEAD")

    def test_a_stale_index_lock_is_not_busy(self, fake_main: Path, capsys) -> None:
        lock = fake_main / ".git" / "index.lock"
        lock.write_text("")
        assert codever.git_busy(fake_main) == str(lock)  # fresh: a git command is running
        old = time.time() - codever.GIT_LOCK_STALE_S - 1
        os.utime(lock, (old, old))
        assert codever.git_busy(fake_main) is None  # a crashed git's leftover
        assert codever.git_busy(fake_main) is None
        assert capsys.readouterr().err.count("ignored as a stale lock") == 1  # logged once
        assert codever.git_busy(fake_main, ignore_stale=False) == str(lock)  # the merge gate still names it
        (fake_main / ".git" / "MERGE_HEAD").write_text("x")
        os.utime(fake_main / ".git" / "MERGE_HEAD", (old, old))
        assert codever.git_busy(fake_main) == str(fake_main / ".git" / "MERGE_HEAD")  # a long merge stays busy


@pytest.mark.usefixtures("real_loaded_version")
class TestLoadedVersion:
    def test_plain_hash_and_cached(self, fake_main: Path, monkeypatch) -> None:
        _cp_file(fake_main)
        monkeypatch.setattr(control_plane, "IMPORTED_AT_NS", time.time_ns() + 10**9)  # files predate the import
        v = codever.loaded_version()
        assert v == codever.code_version(fake_main)
        _grow(fake_main / "scripts" / "qs" / "control_plane" / "a.py")
        assert codever.loaded_version() == v  # computed once

    def test_a_file_modified_since_import_is_unverified(self, fake_main: Path, monkeypatch) -> None:
        f = _cp_file(fake_main)
        stamp = time.time_ns() - 10**9
        monkeypatch.setattr(control_plane, "IMPORTED_AT_NS", stamp)
        os.utime(f, ns=(stamp + 10**6, stamp + 10**6))
        v = codever.loaded_version()
        assert v == codever.UNVERIFIED + codever.code_version(fake_main)

    def test_a_future_mtime_is_ignored_and_logged_once(self, fake_main: Path, monkeypatch, capsys) -> None:
        f = _cp_file(fake_main)
        g = _cp_file(fake_main, "g.py")
        future = time.time_ns() + 3600 * 10**9
        os.utime(f, ns=(future, future))
        os.utime(g, ns=(future, future))
        monkeypatch.setattr(control_plane, "IMPORTED_AT_NS", time.time_ns())
        assert codever.loaded_version() == codever.code_version(fake_main)
        codever._loaded_done = False
        codever.loaded_version()
        assert capsys.readouterr().err.count(f"{f} has an mtime in the future") == 1

    def test_no_main_checkout_is_none(self, monkeypatch) -> None:
        def no_main() -> Path:
            raise codever.errors.CpError("PATH_GUARD", "not a checkout")

        monkeypatch.setattr(paths, "main", no_main)
        assert codever.loaded_version() is None


# --------------------------------------------------------------------------- the restart (D7, AC 4)


def _run(migrated: Path, fake_clock, fake_probe, **kw: Any) -> dict[str, Any]:
    return daemon.run(fake_clock, probe=fake_probe, db_path=migrated, tick_hooks=ticks.hooks(), **kw)


@pytest.mark.usefixtures("active_loop")
class TestRestart:
    def test_a_change_seen_on_two_ticks_exits_new_code(self, migrated, fake_main, fake_clock, fake_probe) -> None:
        open_run()
        f = _cp_file(fake_main)
        ticks.register("t-edit", _editor(f))
        out = _run(migrated, fake_clock, fake_probe, loaded_version=codever.code_version(fake_main), max_ticks=10)
        assert out["exit"] == "new_code" and out["ticks"] == 3  # edited in tick 1, seen in ticks 2 and 3
        assert daemon.read_lease(migrated)["pid"] is None
        with db.file_lock(paths.sidecar(migrated, ".daemon.lock"), exclusive=True, timeout=0) as got:
            assert got  # the singleton is free before `ensure`
        again = _run(migrated, fake_clock, fake_probe, loaded_version=codever.code_version(fake_main), max_ticks=1)
        assert again["exit"] == "max_ticks" and again["ticks"] == 1  # a second run() does not exit at once

    def test_seen_once_triggers_nothing(self, migrated, fake_main, fake_clock, fake_probe) -> None:
        f = _cp_file(fake_main)
        ticks.register("t-edit", _editor(f))
        out = _run(migrated, fake_clock, fake_probe, loaded_version=codever.code_version(fake_main), max_ticks=2)
        assert out["exit"] == "max_ticks" and daemon.restart_candidate() is not None

    def test_a_git_operation_in_progress_triggers_nothing(self, migrated, fake_main, fake_clock, fake_probe) -> None:
        f = _cp_file(fake_main)
        loaded = codever.code_version(fake_main)
        _grow(f)
        (fake_main / ".git" / "MERGE_HEAD").write_text("x")
        out = _run(migrated, fake_clock, fake_probe, loaded_version=loaded, max_ticks=5)
        assert out["exit"] == "max_ticks" and daemon.restart_candidate() is None

    def test_a_reverted_change_starts_the_debounce_over(self, migrated, fake_main, fake_clock, fake_probe) -> None:
        f = _cp_file(fake_main)
        loaded = codever.code_version(fake_main)
        original = f.read_text()
        calls = {"n": 0}

        def flip(conn: Any, clk: clock.Clock) -> None:
            calls["n"] += 1
            f.write_text(original + "# changed\n" if calls["n"] % 2 else original)

        ticks.register("t-flip", flip)
        out = _run(migrated, fake_clock, fake_probe, loaded_version=loaded, max_ticks=6)
        assert out["exit"] == "max_ticks"

    def test_loaded_version_none_never_restarts(self, migrated, fake_main, fake_clock, fake_probe) -> None:
        f = _cp_file(fake_main)
        ticks.register("t-edit", _editor(f))
        out = _run(migrated, fake_clock, fake_probe, max_ticks=5)
        assert out["exit"] == "max_ticks" and daemon.started_code_version() is None

    @pytest.mark.usefixtures("real_loaded_version")
    def test_an_unverified_load_restarts_two_ticks_later(
        self, migrated, fake_main, fake_clock, fake_probe, monkeypatch
    ) -> None:
        f = _cp_file(fake_main)
        stamp = time.time_ns() - 10**9
        monkeypatch.setattr(control_plane, "IMPORTED_AT_NS", stamp)
        os.utime(f, ns=(stamp + 10**6, stamp + 10**6))
        loaded = codever.loaded_version()
        assert loaded is not None and loaded.startswith(codever.UNVERIFIED)
        out = _run(migrated, fake_clock, fake_probe, loaded_version=loaded, max_ticks=10)
        assert out["exit"] == "new_code" and out["ticks"] == 2

    def test_a_sigterm_during_a_hook_ends_the_loop_after_it(self, migrated, fake_clock, fake_probe) -> None:
        after: list[int] = []
        ticks.register("t-term", lambda conn, clk: daemon._on_sigterm(signal.SIGTERM, None))
        ticks.register("t-after", lambda conn, clk: after.append(1))
        out = _run(migrated, fake_clock, fake_probe, max_ticks=5)
        assert out == {**out, "exit": "sigterm", "ticks": 1} and after == []

    def test_a_restart_requested_mid_tick_takes_effect_at_its_end(self, migrated, fake_clock, fake_probe) -> None:
        after: list[int] = []
        ticks.register("t-restart", lambda conn, clk: daemon.request_restart("new_code"))
        ticks.register("t-after", lambda conn, clk: after.append(1))
        out = _run(migrated, fake_clock, fake_probe, max_ticks=5)
        assert out["exit"] == "new_code" and out["ticks"] == 1 and after == [1]


@pytest.mark.usefixtures("active_loop")
class TestDaemonCommand:
    def test_cp_daemon_respawns_on_new_code(self, invoke, migrated, fake_main, fake_popen) -> None:
        open_run()
        fake_popen.calls.clear()
        f = _cp_file(fake_main)
        ticks.register("t-edit", _editor(f))
        code, out = invoke("daemon")
        assert code == 0 and out["exit"] == "new_code" and out["respawn"] == "started"
        assert len(fake_popen.calls) == 1

    def test_a_failing_respawn_is_reported(self, invoke, migrated, fake_main, monkeypatch) -> None:
        open_run()
        f = _cp_file(fake_main)
        ticks.register("t-edit", _editor(f))

        def boom(io: Any, path: Any = None) -> dict[str, Any]:
            raise OSError("no fork")

        monkeypatch.setattr(cli, "ensure_daemon", boom)
        code, out = invoke("daemon")
        assert code == 0 and out["respawn"] == "error" and "OSError: no fork" in out["respawn_error"]

    def test_started_code_version_is_the_value_at_entry(self, invoke, fake_main, monkeypatch) -> None:
        f = _cp_file(fake_main)
        monkeypatch.setattr(codever, "loaded_version", lambda: "sentinel")
        real_migrate = daemon._migrate

        def migrate_then_pull(*a: Any, **k: Any) -> Any:
            _grow(f)  # a pull lands while the daemon migrates
            return real_migrate(*a, **k)

        monkeypatch.setattr(daemon, "_migrate", migrate_then_pull)
        seen: list[str | None] = []
        ticks.register("t-seen", lambda conn, clk: seen.append(daemon.started_code_version()))
        monkeypatch.setattr(daemon, "IDLE_EXIT_S", 1.0)
        assert invoke("daemon")[0] == 0 and seen and set(seen) == {"sentinel"}


# --------------------------------------------------------------------------- bootstrap, env strip


class TestEnsure:
    def test_bootstrap_an_older_schema_daemon_is_replaced(self, migrated, fake_clock, fake_probe, fake_popen) -> None:
        sql(
            migrated,
            "INSERT OR REPLACE INTO daemon_lease (id, pid, pid_start, schema_version, started_at, heartbeat_at)"
            " VALUES (1, 7, 'start-7', 1, 'x', ?)",
            [clock.stamp(fake_clock)],
        )
        assert CUR > 1
        res = daemon.ensure(popen=fake_popen, clock=fake_clock, probe=fake_probe, db_path=migrated)
        assert res == {"status": "restarted"} and len(fake_popen.calls) == 1

    def test_the_spawned_env_carries_no_session_secret(self, migrated, fake_clock, fake_probe, monkeypatch) -> None:
        for name in ("CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CODE_SESSION_ID"):
            monkeypatch.setenv(name, "secret")
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
        monkeypatch.setenv("QS_KEEP_ME", "1")
        monkeypatch.setenv("QS_CP_TOKEN", "run-token")  # a run token in the caller's env never reaches the daemon (J7)
        popen = FakePopen()
        daemon.ensure(popen=popen, clock=fake_clock, probe=fake_probe, db_path=migrated)
        env = popen.calls[0][1]["env"]
        assert env["QS_KEEP_ME"] == "1" and "PATH" in env
        assert not [k for k in env if k.startswith("CLAUDE_CODE_MESSAGING_")]
        assert not {"CLAUDE_CODE_SESSION_ID", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "QS_CP_TOKEN"} & set(env)


# --------------------------------------------------------------------------- killed during wait (AC 5)


def test_a_daemon_killed_during_wait_is_restarted(
    migrated, deps, fake_clock, fake_popen, tmp_path, monkeypatch
) -> None:
    run_id, token = open_run()
    fake_popen.calls.clear()
    proc = subprocess.Popen(["sleep", "60"])
    try:
        probe = liveness.ProcessProbe(runner.Runner())
        start = probe.start_of(proc.pid)
        assert start is not None
        sql(
            migrated,
            "INSERT OR REPLACE INTO daemon_lease (id, pid, pid_start, schema_version, started_at, heartbeat_at)"
            " VALUES (1, ?, ?, ?, 'x', ?)",
            [proc.pid, start, CUR, clock.stamp(fake_clock)],
        )
        monkeypatch.setattr(cli, "make_deps", lambda: cli.Deps(**{**deps.__dict__, "probe": probe}))
        original = fake_clock.sleep
        killed = {"done": False}

        def sleep(seconds: float) -> None:
            original(seconds)
            if not killed["done"]:
                killed["done"] = True
                proc.kill()
                proc.wait()  # reaped: no zombie
                fake_clock.advance(daemon.STALE_AFTER_S)

        fake_clock.sleep = sleep  # type: ignore[method-assign]
        msg = tmp_path / "m.json"
        msg.write_text("{}")

        def on_spawn(argv: list[str], kw: dict[str, Any]) -> None:
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
                    str(msg),
                    "--token",
                    token,
                )[0]
                == 0
            )

        fake_popen.on_call = on_spawn
        code, out = run_cli("wait", "--run", run_id, "--token", token, "--poll", "1")
        assert (code, out) == (0, {"ok": True, "pending": 1})
        assert len(fake_popen.calls) == 1  # `ensure` started a new daemon
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_the_code_version_hook_is_built_in() -> None:
    assert ("code_version", selfcheck.code_version_hook) in ticks.registered()
