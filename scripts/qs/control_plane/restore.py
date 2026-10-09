"""Periodic backups and the restore (QS-406 §10, D14).

**The ``backup`` hook.** While a run is open it copies the DB every ``BACKUP_EVERY_S`` into
``backups.db_dir(db_path)`` (0600 files, 0700 directories) and rotates the periodic copies. A failure
(``CpError``, ``OSError``, ``sqlite3.Error``) is logged once and kept in ``meta.last_backup_error``:
``backup_failed`` is synced to every open run while that key exists (so it survives a restart), the next
attempt waits ``BACKUP_EVERY_S``, and a success deletes the key.

**``restore``** (``cp.py restore --confirm``; exempt — the DB may be lost — and guarded by the
``PreToolUse`` ``ask`` prompt) copies the newest ``periodic`` or ``v<N>`` backup **into the live file in
place**, with SQLite's backup API: the file is never replaced and its WAL never deleted, so every open
connection stays coherent and a concurrent writer meets the busy timeout. The live DB is kept first as
``harness_state.replaced.<stamp>.db``. The live ``waiters`` rows and the ``selfcheck`` / ``selfcheck_override``
/ ``selfcheck_pending`` meta keys (they describe the code on disk, not the data) are carried across.
A ``<db>.restoring`` marker keeps ``ensure`` and any starting daemon off the DB meanwhile; every exit
path after it removes it, releases the locks and calls ``ensure``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import alerts, backups, daemon, db, errors, liveness, migrations, paths
from . import clock as clock_mod

BACKUP = "backup"
LAST_AT = "last_backup_at"
LAST_ERROR = "last_backup_error"
RESTORE_LOCK_WAIT_S = 60.0
CARRIED_META = ("selfcheck", "selfcheck_override", "selfcheck_pending")
FIXED_UP_TABLES = ("waiters", "daemon_lease", "meta")  # the restore's fix-up writes these in the copied data

_logged_error: str | None = None


def _log(message: str) -> None:
    sys.stderr.write(f"[cp-backup] {message}\n")
    sys.stderr.flush()


def _last_error(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """``meta.last_backup_error``; a value that is not an object (hand-edited, truncated) as ``{at: None, error}``."""
    value = _meta(conn, LAST_ERROR)
    if value is None or isinstance(value, dict):
        return value
    return {"at": None, "error": str(value)}


def _meta(conn: sqlite3.Connection, key: str) -> Any:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except TypeError, ValueError:
        return row[0]


def _older_than(clock: clock_mod.Clock, stamp: Any, seconds: float) -> bool:
    """``stamp`` is at least ``seconds`` old (an unreadable stamp counts as old)."""
    try:
        return (clock_mod.age(clock, str(stamp)) or 0.0) >= seconds
    except ValueError:
        return True


def _open_runs(conn: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(r[0] for r in conn.execute("SELECT id FROM runs WHERE state = 'open' ORDER BY rowid"))


def _db_path(conn: sqlite3.Connection) -> Path:
    return Path(next(r[2] for r in conn.execute("PRAGMA database_list") if r[1] == "main"))


def backup_hook(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
    global _logged_error
    runs = _open_runs(conn)
    error = _last_error(conn)
    due = (
        bool(runs)
        and (_meta(conn, LAST_AT) is None or _older_than(clock, _meta(conn, LAST_AT), backups.BACKUP_EVERY_S))
        and (error is None or _older_than(clock, error.get("at"), backups.BACKUP_EVERY_S))
    )
    if due:
        now = db.now(clock)
        try:
            db_path = _db_path(conn)
            backups.take(conn, clock.now(), db_path)
        except (errors.CpError, OSError, sqlite3.Error) as exc:
            error = {"at": now, "error": f"{type(exc).__name__}: {exc}"}
            if error["error"] != _logged_error:
                _logged_error = error["error"]
                _log(f"backup failed: {error['error']}")
            with db.write(conn):
                conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (LAST_ERROR, json.dumps(error)))
        else:
            error, _logged_error = None, None
            with db.write(conn):
                conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (LAST_AT, json.dumps(now)))
                conn.execute("DELETE FROM meta WHERE key = ?", (LAST_ERROR,))
            try:  # the copy is taken: a failed rotation is logged, never a `backup_failed`
                backups.rotate(backups.db_dir(db_path), clock.now())
            except (OSError, ValueError) as exc:  # ValueError: a file name with an impossible stamp
                _log(f"rotation failed: {type(exc).__name__}: {exc}")
    active = [] if error is None else [alerts.Condition(alerts.BACKUP_FAILED, "backup", runs, dict(error))]
    alerts.sync(conn, clock, kinds={alerts.BACKUP_FAILED}, active=active)


# --------------------------------------------------------------------------- restore


def _live_state(path: Path) -> tuple[int, int] | None | str:
    """``(user_version, page_size)``; ``None`` when missing (or never migrated); ``"corrupt"`` when unreadable."""
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return "corrupt"
    return None if version == 0 else (version, page_size)


def _live_tool_call(conn: sqlite3.Connection, probe: liveness.ProcessProbe) -> bool:
    rows = conn.execute(
        "SELECT holder_pid, holder_pid_start, holder_pgid FROM tool_calls WHERE state = 'started'"
    ).fetchall()
    return any(probe.holder_alive(r[0], r[1], r[2]) for r in rows)


def _check_live_calls(path: Path, probe: liveness.ProcessProbe) -> None:
    conn = db.connect(path, mode="ro")
    try:
        if _live_tool_call(conn, probe):
            raise errors.CpError("BUSY", "a tool call is in flight; retry when it has finished")
    finally:
        conn.close()


def _check_source(src: Path, live: tuple[int, int] | None) -> tuple[int, int]:
    try:
        conn = db.connect(src, mode="ro")
        try:
            check = conn.execute("PRAGMA quick_check").fetchone()[0]
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
            version = db.user_version(conn)
            page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        finally:
            conn.close()
    except (sqlite3.DatabaseError, errors.CpError) as exc:
        raise errors.CpError("CONFLICT", f"the backup {src.name} is unreadable: {exc}") from exc
    if check != "ok":
        raise errors.CpError("CONFLICT", f"the backup {src.name} fails quick_check: {check}")
    missing = [t for t in FIXED_UP_TABLES if t not in tables]
    if missing:
        raise errors.CpError("CONFLICT", f"the backup {src.name} has no {', '.join(missing)} table")
    if version > migrations.current_schema_version():
        raise errors.CpError("CONFLICT", f"the backup {src.name} is at schema v{version}, newer than this code")
    if live is not None and page_size != live[1]:
        raise errors.CpError("CONFLICT", f"the backup's page size {page_size} differs from the live DB's {live[1]}")
    return version, page_size


def _write_marker(path: Path, me: liveness.Holder, clock: clock_mod.Clock) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"pid": me.pid, "pid_start": me.pid_start, "at": db.now(clock)}))
    os.replace(tmp, path)


def _remaining(deadline: float, clock: clock_mod.Clock) -> float:
    return max(0.0, deadline - clock.now().timestamp())


def _carry(conn: sqlite3.Connection) -> tuple[list[tuple[Any, ...]], dict[str, str]]:
    waiters = [
        tuple(r) for r in conn.execute("SELECT id, run_id, pid, pid_start, started_at, heartbeat_at FROM waiters")
    ]
    marks = ", ".join("?" for _ in CARRIED_META)
    meta = {r[0]: r[1] for r in conn.execute(f"SELECT key, value FROM meta WHERE key IN ({marks})", CARRIED_META)}
    return waiters, meta


def _fix_up(conn: sqlite3.Connection, waiters: list[tuple[Any, ...]], meta: dict[str, str]) -> None:
    """Re-apply the carried rows on the copied data and clear the copied daemon lease (one transaction)."""
    db.begin(conn, "BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM waiters")
        conn.executemany(
            "INSERT INTO waiters (id, run_id, pid, pid_start, started_at, heartbeat_at) VALUES (?, ?, ?, ?, ?, ?)",
            waiters,
        )
        conn.execute("UPDATE daemon_lease SET pid = NULL, heartbeat_at = NULL")
        conn.executemany("DELETE FROM meta WHERE key = ?", [(k,) for k in CARRIED_META])
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", list(meta.items()))
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _restore_into(path: Path, src: Path, live: tuple[int, int] | None, clock: clock_mod.Clock) -> str | None:
    """Keep the live DB, copy the backup into it in place, re-apply the carried rows → the ``replaced`` path.

    A failure before the copy is a ``CONFLICT`` (the live DB is untouched); a failure of the fix-up after
    the copy is an ``INTERNAL`` error saying so. Both carry ``replaced`` once the live DB was kept.
    """
    stamp = clock.now().strftime(backups.STAMP_FORMAT)
    replaced: str | None = None

    def kept() -> dict[str, str]:
        return {} if replaced is None else {"replaced": replaced}

    try:
        conn = db.connect(path, mode="rwc")
        try:
            waiters: list[tuple[Any, ...]] = []
            meta: dict[str, str] = {}
            if live is not None:
                dest = backups.db_dir(path) / f"harness_state.replaced.{stamp}.db"
                replaced = str(backups.write_copy(conn, dest))
                waiters, meta = _carry(conn)
            source = db.connect(src, mode="ro")
            try:
                source.backup(conn)
            finally:
                source.close()
            try:
                _fix_up(conn, waiters, meta)
            except sqlite3.Error as exc:
                where = "" if replaced is None else f"; the pre-restore DB is kept at {replaced}"
                raise errors.CpError(
                    "INTERNAL",
                    f"the backup {src.name} was copied into the live DB, but the fix-up (waiters, daemon lease,"
                    f" self-check keys) failed: {exc}",
                    hint=f"the live DB holds the backup's data{where}; retry the restore",
                    **kept(),
                ) from exc
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        where = "" if replaced is None else f" (the pre-restore DB is kept at {replaced})"
        raise errors.CpError(
            "CONFLICT",
            f"the live DB cannot be kept or restored: {exc}",
            hint=f"move it aside, then retry{where}",
            **kept(),
        ) from exc
    return replaced


def restore(
    db_path: Path,
    *,
    clock: clock_mod.Clock,
    probe: liveness.ProcessProbe,
    kill: Callable[[int, int], None],
    popen: Callable[..., Any],
    lock_wait_s: float = RESTORE_LOCK_WAIT_S,
) -> dict[str, Any]:
    """Restore the newest backup of ``db_path`` in place (see the module docstring) → the report."""
    # 1. check, before stopping anything: a refusal here writes no marker and calls no `ensure`
    state = _live_state(db_path)
    if isinstance(state, str):  # "corrupt": refused before anything is stopped
        raise errors.CpError("CONFLICT", f"the live DB {db_path} is unreadable", hint="move it aside, then retry")
    live = state
    if live is not None and live[0] > migrations.current_schema_version():
        raise errors.CpError("SCHEMA_TOO_NEW", f"the live DB is at schema v{live[0]}, newer than this code")
    src = backups.choose_source(backups.db_dir(db_path))
    if src is None:
        raise errors.CpError("NOT_FOUND", f"no backup of {db_path} in {backups.db_dir(db_path)}")
    _check_source(src, live)
    if live is not None:
        _check_live_calls(db_path, probe)
    # 2. the marker; from here on every exit path removes it, releases the locks and calls `ensure`
    marker = daemon.marker_path(db_path)
    _write_marker(marker, probe.me(), clock)
    deadline = clock.now().timestamp() + lock_wait_s
    try:
        daemon.stop(db_path, clock=clock, probe=probe, kill=kill, restart_wait_s=_remaining(deadline, clock))
        busy = errors.CpError("BUSY", "the daemon may be finishing a tick; retry")
        with db.file_lock(
            paths.sidecar(db_path, ".daemon.lock"), exclusive=True, timeout=_remaining(deadline, clock)
        ) as got:
            if not got:
                raise busy
            with db.file_lock(
                paths.sidecar(db_path, ".migrate.lock"), exclusive=True, timeout=_remaining(deadline, clock)
            ) as got_migrate:
                if not got_migrate:
                    raise busy
                if live is not None:
                    _check_live_calls(db_path, probe)
                replaced = _restore_into(db_path, src, live, clock)
    finally:
        marker.unlink(missing_ok=True)
        try:
            daemon.ensure(popen=popen, clock=clock, probe=probe, db_path=db_path, kill=kill)
        except Exception as exc:  # noqa: BLE001 — the next `wait` or command ensures again
            _log(f"ensure after the restore failed: {exc!r}")
    backup_at = backups.stamp_of(src.name)
    at = clock_mod.iso(backup_at) if backup_at is not None else None
    return {
        "restored_from": str(src),
        "backup_at": at,
        "replaced": replaced,
        "message": (
            f"Everything recorded after {at} is lost. Run tokens issued since then are stale: each orchestrator"
            " must `run claim` again. The next ticks re-sync CI and liveness."
        ),
    }
