"""Connections, write transactions and the schema checks (§3).

* ``connect`` never creates the DB (``mode=rw``); only ``migrate`` does.
* ``write`` is ``BEGIN IMMEDIATE`` + a ``user_version`` re-check, so a
  long-lived process running older code can never write to a migrated DB.
* No subprocess ever runs inside a write transaction.
"""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import clock as clock_mod
from . import errors, migrations, paths

BUSY_TIMEOUT_MS = 5000
MIGRATE_WAIT_S = 60.0
POLL_S = 0.5
_MIGRATING = -1


def connect(path: Path, *, mode: str = "rw") -> sqlite3.Connection:
    """Autocommit connection (explicit transactions only); ``mode`` is ``rw``, ``rwc`` or ``ro``."""
    uri = Path(path).resolve().as_uri() + f"?mode={mode}"
    try:
        conn = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=BUSY_TIMEOUT_MS / 1000)
    except sqlite3.OperationalError as exc:
        raise errors.CpError("INTERNAL", f"cannot open {path}: {exc}") from exc
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    if mode != "ro":
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


_BUSY_CODES = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})


def begin(conn: sqlite3.Connection, statement: str) -> None:
    """Start a transaction: contention is ``BUSY``, any other SQLite failure ``INTERNAL``."""
    try:
        conn.execute(statement)
    except sqlite3.OperationalError as exc:
        if (exc.sqlite_errorcode & 0xFF) in _BUSY_CODES:
            raise errors.CpError("BUSY", f"database busy: {exc}") from exc
        raise errors.CpError("INTERNAL", f"cannot start a transaction: {exc}") from exc


@contextmanager
def write(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` → schema re-check → body → ``COMMIT`` (``ROLLBACK`` on any exception, a failed COMMIT included)."""
    begin(conn, "BEGIN IMMEDIATE")
    try:
        found = user_version(conn)
        expected = migrations.current_schema_version()
        if found != expected:
            raise errors.CpError(
                "SCHEMA_TOO_NEW",
                f"the DB is at schema v{found}, this code writes v{expected}",
                db_version=found,
                code_version=expected,
            )
        yield conn
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    try:
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:  # e.g. a deferred constraint: the transaction is still open
            conn.execute("ROLLBACK")
        raise


@contextmanager
def read(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """A consistent read snapshot."""
    begin(conn, "BEGIN")
    try:
        yield conn
    finally:
        if conn.in_transaction:
            conn.execute("COMMIT")


@contextmanager
def file_lock(path: Path, *, exclusive: bool, timeout: float | None) -> Iterator[bool]:
    """An ``flock`` on ``path``: yields whether it was obtained.

    ``timeout=None`` blocks; ``0`` tries once.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, flag | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    acquired = False
                    break
                time.sleep(0.02)
        try:
            yield acquired
        finally:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def probe_version(path: Path) -> int | None:
    """``None`` for a missing DB, ``-1`` while a migration holds the lock, else ``user_version``."""
    if not path.exists():
        return None
    with file_lock(paths.sidecar(path, ".migrate.lock"), exclusive=False, timeout=0) as got:
        if not got:
            return _MIGRATING
        conn = connect(path, mode="ro")
        try:
            return user_version(conn)
        finally:
            conn.close()


def migrate_error(path: Path) -> dict[str, Any] | None:
    """The daemon's ``<db>.migrate-error.json`` sidecar, if present and readable."""
    try:
        data = json.loads(paths.sidecar(path, ".migrate-error.json").read_text())
    except OSError, ValueError:
        return None
    return data if isinstance(data, dict) else None


def check_schema(
    path: Path,
    *,
    wait: bool,
    clock: clock_mod.Clock,
    ensure: Callable[[], dict[str, Any]] | None = None,
    wait_s: float = MIGRATE_WAIT_S,
    poll_s: float = POLL_S,
) -> str:
    """The entry schema check: ``"ok"``, ``"missing"`` (``wait=False`` only), or an error.

    Equal → proceed. Newer DB → ``SCHEMA_TOO_NEW``. Older or missing: with
    ``wait``, start the daemon (``ensure``) and poll until it migrated, else
    ``SCHEMA_PENDING``; without ``wait`` (read-only commands, hooks), a
    missing DB is ``"missing"`` and an older one is ``SCHEMA_PENDING`` at once.
    """
    target = migrations.current_schema_version()
    found = probe_version(path)
    if found == target:
        return "ok"
    if found is not None and found > target:
        raise errors.CpError("SCHEMA_TOO_NEW", f"the DB is at schema v{found}, this code knows v{target}")
    if not wait:
        if found is None:
            return "missing"
        if found == _MIGRATING:
            raise errors.CpError("SCHEMA_PENDING", f"a migration is in progress, waiting for v{target}")
        raise errors.CpError("SCHEMA_PENDING", f"the DB is at schema v{found}, waiting for v{target}")
    status = ensure() if ensure is not None else {}
    if status.get("status") == "migrate_failed":
        raise errors.CpError("SCHEMA_PENDING", "the daemon's migration failed", migrate_error=status.get("error"))
    deadline = clock.now().timestamp() + wait_s
    while clock.now().timestamp() < deadline:
        clock.sleep(poll_s)
        found = probe_version(path)
        if found == target:
            return "ok"
        if found is not None and found > target:
            raise errors.CpError("SCHEMA_TOO_NEW", f"the DB is at schema v{found}, this code knows v{target}")
    raise errors.CpError(
        "SCHEMA_PENDING", f"the DB did not reach schema v{target} within {wait_s:g}s", migrate_error=migrate_error(path)
    )


def next_id(conn: sqlite3.Connection, kind: str, prefix: str) -> str:
    """Allocate ``<prefix><n>`` from ``counters`` inside the caller's transaction."""
    row = conn.execute("SELECT next FROM counters WHERE kind = ?", (kind,)).fetchone()
    n = 1 if row is None else int(row[0])
    conn.execute(
        "INSERT INTO counters (kind, next) VALUES (?, ?) ON CONFLICT(kind) DO UPDATE SET next = excluded.next",
        (kind, n + 1),
    )
    return f"{prefix}{n}"


def now(clock: clock_mod.Clock) -> str:
    return clock_mod.iso(clock.now())


def as_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return None if row is None else {k: row[k] for k in row.keys()}  # noqa: SIM118 — sqlite3.Row
