"""The migration registry and ``migrate()`` (§5.1).

Seam: later children append ``Migration(n + 1, …)`` to ``MIGRATIONS``. Each
step runs in its own ``BEGIN IMMEDIATE`` with single statements
(``execute``, never ``executescript``, which commits implicitly), then
``PRAGMA user_version = n``, then ``COMMIT``. Only the daemon running the
main checkout's code, with ``main`` checked out, migrates the live DB.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import errors, faults, paths, schema_v1

MIGRATE_LOCK_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "schema v1: the task tree, queues, tools and hooks", schema_v1.STATEMENTS),
)
SCHEMA_VERSION = MIGRATIONS[-1].version


def current_schema_version() -> int:
    """The version this code reads and writes (read at call time, so tests can extend the registry)."""
    return MIGRATIONS[-1].version


def _authorise(db_path: Path, role: str) -> None:
    resolved = db_path.resolve()
    if not paths.is_live_db_path(resolved):
        return
    root = paths.code_root()
    main_dir = paths.main_checkout(root)
    if (
        role == "daemon"
        and main_dir.resolve() == root.resolve()
        and resolved == paths.live_db(main_dir).resolve()
        and paths.main_head_branch(main_dir) == "main"
    ):
        return
    raise errors.CpError(
        "POLICY_REFUSED",
        "only the daemon running the main checkout's code, with `main` checked out, migrates the live DB",
    )


def _backup(conn: sqlite3.Connection, version: int) -> Path:
    target_dir = paths.backup_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    dest = target_dir / f"harness_state.v{version}.{stamp}.db"
    copy = sqlite3.connect(dest)
    try:
        conn.backup(copy)
    finally:
        copy.close()
    fd = os.open(dest, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dest


def migrate(db_path: Path, *, role: str, lock_timeout: float = MIGRATE_LOCK_TIMEOUT_S) -> dict[str, Any]:
    """Create or migrate ``db_path`` to the current schema → ``{from, to, backup}``."""
    from . import db  # db imports this module for the version check

    db_path = Path(db_path)
    _authorise(db_path, role)
    with db.file_lock(paths.sidecar(db_path, ".migrate.lock"), exclusive=True, timeout=lock_timeout) as got:
        if not got:
            raise errors.CpError("BUSY", f"another migration of {db_path} holds the lock")
        existed = db_path.exists()
        conn = db.connect(db_path, mode="rwc")
        try:
            current = db.user_version(conn)
            target = current_schema_version()
            if current > target:
                raise errors.CpError("SCHEMA_TOO_NEW", f"the DB is at schema v{current}, this code knows v{target}")
            if current == target:
                return {"result": "noop", "from": current, "to": current, "backup": None}
            backup = _backup(conn, current) if existed and current > 0 else None
            for step in MIGRATIONS:
                if step.version <= current:
                    continue
                conn.execute("BEGIN IMMEDIATE")
                try:
                    for i, statement in enumerate(step.statements):
                        conn.execute(statement)
                        if i == 0:
                            faults.hit("migrate.mid_step")
                    conn.execute(f"PRAGMA user_version = {int(step.version)}")
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            return {
                "result": "migrated",
                "from": current,
                "to": target,
                "backup": None if backup is None else str(backup),
            }
        finally:
            conn.close()
