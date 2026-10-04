"""The daemon skeleton (§5.2) and ``ensure`` (§5.3) — the core of child 14's active loop.

``run``: a singleton ``flock``, its own start time (retried; never a NULL
``pid_start``), ``migrate(role="daemon")`` (retried on ``BUSY``, which never
writes the backoff sidecar), a ``daemon_lease`` row, a heartbeat (a ``BUSY``
beat is logged and skipped), the tick hooks, and an idle exit. ``ensure``:
start the daemon when its lease is stale; stop an older-schema daemon, or a
stale same-schema one proven alive, with SIGTERM, then SIGKILL when it is
proven alive, the same process and its heartbeat frozen past the wait;
``restart_pending`` / ``stale_alive`` while it cannot be proven gone.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import clock as clock_mod
from . import db, errors, liveness, migrations, paths, procsetup

TICK_S = 5.0
IDLE_EXIT_S = 1800.0
STALE_AFTER_S = 30.0
DAEMON_RESTART_WAIT_S = 15.0
MIGRATE_BACKOFF_S = 300.0
MIGRATE_BUSY_RETRIES = 6
PID_START_RETRIES = 3
PID_START_RETRY_S = 1.0
KILL_WAIT_S = 2.0
LOG_NAME = "harness_state.daemon.log"

TickHook = Callable[[Any, clock_mod.Clock], None]

_stop = threading.Event()


def _on_sigterm(signum: int, frame: Any) -> None:
    """The daemon's SIGTERM handler: finish the current tick, then exit."""
    _stop.set()


def _log(message: str) -> None:
    sys.stderr.write(f"[cp-daemon] {message}\n")
    sys.stderr.flush()


def _write_migrate_error(db_path: Path, clock: clock_mod.Clock, error: str) -> None:
    payload = {"at": clock_mod.iso(clock.now()), "error": error}
    paths.sidecar(db_path, ".migrate-error.json").write_text(json.dumps(payload, sort_keys=True))


def _migrate(db_path: Path, clock: clock_mod.Clock, tick_s: float) -> dict[str, Any]:
    """``migrate(role="daemon")``, retrying contention; ``BUSY`` at the end is raised without the sidecar."""
    attempt = 0
    while True:
        try:
            return migrations.migrate(db_path, role="daemon")
        except errors.CpError as exc:
            if exc.code != "BUSY" or attempt >= MIGRATE_BUSY_RETRIES:
                raise
            attempt += 1
            _log(f"migration busy, retrying: {exc.detail}")
            clock.sleep(tick_s)


def _any_open_run(conn: Any) -> bool:
    return conn.execute("SELECT 1 FROM runs WHERE state = 'open' LIMIT 1").fetchone() is not None


def run(
    clock: clock_mod.Clock,
    *,
    probe: liveness.ProcessProbe,
    db_path: Path | None = None,
    tick_hooks: Sequence[TickHook] = (),
    max_ticks: int | None = None,
    tick_s: float = TICK_S,
    idle_exit_s: float = IDLE_EXIT_S,
) -> dict[str, Any]:
    db_path = db_path or paths.select_db()
    with db.file_lock(paths.sidecar(db_path, ".daemon.lock"), exclusive=True, timeout=0) as got:
        if not got:
            return {"singleton": "held_elsewhere"}
        me = _me_with_start(probe, clock)
        if me is None:
            _log("cannot read this process's start time: refusing to write a lease nobody could verify")
            return {"ticks": 0, "exit": "no_pid_start"}
        try:
            migrated = _migrate(db_path, clock, tick_s)
        except Exception as exc:  # noqa: BLE001 — every failure but contention is recorded for `ensure`
            if isinstance(exc, errors.CpError) and exc.code == "BUSY":
                _log(f"migration still busy, exiting without the backoff sidecar: {exc.detail}")
                raise
            error = (
                exc.payload() if isinstance(exc, errors.CpError) else {"error": type(exc).__name__, "detail": str(exc)}
            )
            _write_migrate_error(db_path, clock, json.dumps(error, sort_keys=True))
            _log(f"migration failed: {error}")
            raise errors.CpError("INTERNAL", "migration failed", migrate_error=error) from exc
        paths.sidecar(db_path, ".migrate-error.json").unlink(missing_ok=True)
        conn = db.connect(db_path)
        try:
            with db.write(conn):
                conn.execute(
                    "INSERT INTO daemon_lease (id, pid, pid_start, schema_version, started_at, heartbeat_at)"
                    " VALUES (1, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET pid = excluded.pid,"
                    " pid_start = excluded.pid_start, schema_version = excluded.schema_version,"
                    " started_at = excluded.started_at, heartbeat_at = excluded.heartbeat_at",
                    (me.pid, me.pid_start, migrations.current_schema_version(), db.now(clock), db.now(clock)),
                )
            _stop.clear()
            procsetup.get().install_sigterm(_on_sigterm)
            ticks = 0
            last_open = clock.now()
            reason = "max_ticks"
            while True:
                try:
                    with db.write(conn):
                        conn.execute("UPDATE daemon_lease SET heartbeat_at = ? WHERE id = 1", (db.now(clock),))
                except errors.CpError as exc:
                    if exc.code != "BUSY":
                        raise
                    _log(f"heartbeat skipped (busy): {exc.detail}")
                for hook in tick_hooks:
                    try:
                        hook(conn, clock)
                    except Exception as exc:  # noqa: BLE001 — a broken hook must not kill the loop
                        _log(f"tick hook {getattr(hook, '__name__', hook)!r} failed: {exc!r}")
                ticks += 1
                if _any_open_run(conn):
                    last_open = clock.now()
                if _stop.is_set():
                    reason = "sigterm"
                    break
                if (clock.now() - last_open).total_seconds() >= idle_exit_s:
                    reason = "idle"
                    break
                if max_ticks is not None and ticks >= max_ticks:
                    break
                clock.sleep(tick_s)
            return {"ticks": ticks, "exit": reason, "migrated": migrated}
        finally:
            try:
                with db.write(conn):
                    conn.execute(
                        "UPDATE daemon_lease SET pid = NULL, heartbeat_at = NULL WHERE id = 1 AND pid = ?", (me.pid,)
                    )
            except errors.CpError as exc:  # never mask the loop's own result or error
                _log(f"could not clear the daemon lease: {exc}")
            finally:
                conn.close()


def _me_with_start(probe: liveness.ProcessProbe, clock: clock_mod.Clock) -> liveness.Holder | None:
    """This process, with its start time read (retried); ``None`` when ``ps`` keeps failing."""
    for attempt in range(PID_START_RETRIES):
        me = probe.me()
        if me.pid_start is not None:
            return me
        if attempt + 1 < PID_START_RETRIES:
            clock.sleep(PID_START_RETRY_S)
    return None


def read_lease(db_path: Path) -> dict[str, Any] | None:
    """The ``daemon_lease`` row, read defensively: a missing file or table is ``None``."""
    if not db_path.exists():
        return None
    try:
        conn = db.connect(db_path, mode="ro")
    except errors.CpError:
        return None
    try:
        return db.as_dict(conn.execute("SELECT * FROM daemon_lease WHERE id = 1").fetchone())
    except Exception:  # noqa: BLE001 — no table yet, a corrupt file: "stale"
        return None
    finally:
        conn.close()


def read_lease_conn(conn: Any) -> dict[str, Any] | None:
    """The ``daemon_lease`` row through an open connection."""
    return db.as_dict(conn.execute("SELECT * FROM daemon_lease WHERE id = 1").fetchone())


def _fresh(lease: dict[str, Any] | None, clock: clock_mod.Clock, stale_after_s: float) -> bool:
    if lease is None or lease["pid"] is None:
        return False
    elapsed = clock_mod.age(clock, lease["heartbeat_at"])
    return elapsed is not None and elapsed < stale_after_s


def _signal(kill: Callable[[int, int], None], pid: int, sig: int) -> bool:
    """Send ``sig``; ``False`` when the process is already gone."""
    try:
        kill(pid, sig)
    except ProcessLookupError:
        return False
    return True


def _stop_old(
    lease: dict[str, Any],
    *,
    alive: bool | None,
    fresh: bool,
    pending: str,
    db_path: Path,
    clock: clock_mod.Clock,
    probe: liveness.ProcessProbe,
    kill: Callable[[int, int], None],
    restart_wait_s: float,
) -> dict[str, Any] | None:
    """Stop the daemon of ``lease`` → ``None`` once it is gone, else ``{"status": pending, "pid": …}``.

    SIGTERM only when it is proven alive and its identity is checkable (a recorded start time, or a fresh
    lease); SIGKILL once the wait passes only when it is still proven alive, the same process (its start
    time matches) and its heartbeat has not moved since the SIGTERM — a hung daemon. Unknown liveness
    never gets a signal: ``pending``.
    """
    old_pid, start = lease["pid"], lease["pid_start"]
    signalled = alive is True and (start is not None or fresh)
    if signalled and not _signal(kill, old_pid, signal.SIGTERM):
        return None
    deadline = clock.now().timestamp() + restart_wait_s
    while True:
        current = read_lease(db_path)
        if current is None or current["pid"] != old_pid:
            return None  # gone, or replaced by another daemon
        if clock.now().timestamp() >= deadline:
            break
        clock.sleep(db.POLL_S)
    now_alive = probe.alive(old_pid, start)
    if now_alive is False:
        return None
    hung = signalled and now_alive is True and start is not None and current["heartbeat_at"] == lease["heartbeat_at"]
    if hung:
        if not _signal(kill, old_pid, signal.SIGKILL):
            return None
        kill_deadline = clock.now().timestamp() + KILL_WAIT_S
        while probe.alive(old_pid, start) is not False:
            if clock.now().timestamp() >= kill_deadline:
                return {"status": pending, "pid": old_pid}
            clock.sleep(db.POLL_S)
        return None
    return {"status": pending, "pid": old_pid}


def ensure(
    *,
    popen: Callable[..., Any],
    clock: clock_mod.Clock,
    probe: liveness.ProcessProbe,
    db_path: Path | None = None,
    kill: Callable[[int, int], None] = os.kill,
    argv: Sequence[str] | None = None,
    stale_after_s: float = STALE_AFTER_S,
    restart_wait_s: float = DAEMON_RESTART_WAIT_S,
    migrate_backoff_s: float = MIGRATE_BACKOFF_S,
) -> dict[str, Any]:
    """Make sure a daemon running this code's schema is up → ``{"status": …}``."""
    db_path = db_path or paths.select_db()
    root = paths.code_root()
    main_dir = paths.main_checkout(root)
    if paths.is_live_db_path(db_path.resolve()) and main_dir.resolve() != root.resolve():
        raise errors.CpError("POLICY_REFUSED", "only the main checkout's code may start the daemon of the live DB")
    target = migrations.current_schema_version()
    lease = read_lease(db_path)
    fresh = _fresh(lease, clock, stale_after_s)
    if fresh and lease is not None and lease["schema_version"] >= target:
        return {"status": "already_running", "pid": lease["pid"]}
    error = db.migrate_error(db_path)
    if error is not None:
        elapsed = clock_mod.age(clock, error.get("at")) if isinstance(error.get("at"), str) else None
        if elapsed is not None and elapsed < migrate_backoff_s:
            return {"status": "migrate_failed", "error": error.get("error")}
    status = "started"
    if lease is not None and lease["pid"] is not None:
        older = lease["schema_version"] < target
        # A stale lease with no start time cannot be verified (its pid may be reused): never signalled or
        # waited on — a new daemon that finds the singleton held simply exits.
        checkable = lease["pid_start"] is not None or fresh
        alive = probe.alive(lease["pid"], lease["pid_start"]) if checkable else False
        stuck = not older and alive is True and lease["pid_start"] is not None  # stale, same schema, alive
        if (older and alive is not False) or stuck:
            outcome = _stop_old(
                lease,
                alive=alive,
                fresh=fresh,
                pending="restart_pending" if older else "stale_alive",
                db_path=db_path,
                clock=clock,
                probe=probe,
                kill=kill,
                restart_wait_s=restart_wait_s,
            )
            if outcome is not None:
                return outcome
            status = "restarted"
        elif older and fresh:
            status = "restarted"  # a fresh older lease whose pid is proven dead: nothing to wait for
    if argv is None:
        venv_python = main_dir / "venv" / "bin" / "python"
        python = str(venv_python) if venv_python.exists() else sys.executable
        argv = [python, str(root / "scripts" / "qs" / "cp.py"), "daemon"]
    with open(main_dir / LOG_NAME, "a") as log:
        popen(
            list(argv),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            cwd=str(main_dir),
            close_fds=True,
        )
    return {"status": status}
