"""Backups of the Control Plane DB (QS-406 §10): one private subdirectory per DB, atomic copies.

A leaf module (D17): it imports only ``paths``. Every copy (periodic, migration, a restore's
``replaced``) goes through ``write_copy`` into ``db_dir(db_path)``.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from . import paths


def db_dir(db_path: Path) -> Path:
    """``backup_dir() / sha256(<resolved db path>)[:8]``: a temporary DB's backups never mix with the live DB's."""
    return paths.backup_dir() / hashlib.sha256(str(Path(db_path).resolve()).encode()).hexdigest()[:8]


def _journal_delete(copy: sqlite3.Connection) -> str:
    """Switch the copy to a rollback journal → the mode SQLite reports."""
    copy.execute("PRAGMA user_version").fetchone()  # re-read page 1, or the pragma is a no-op on a WAL header
    return str(copy.execute("PRAGMA journal_mode = DELETE").fetchone()[0])


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_copy(src_conn: sqlite3.Connection, dest: Path) -> Path:
    """Copy ``src_conn``'s DB to ``dest``: 0600, in 0700 directories, ``journal_mode = DELETE``, atomic.

    The copy is written to ``<dest>.partial`` and renamed into place; any failure removes the
    ``.partial`` and re-raises.
    """
    paths.ensure_private_dir(paths.backup_dir())
    paths.ensure_private_dir(dest.parent)
    partial = dest.with_name(dest.name + ".partial")
    os.close(os.open(partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    copy: sqlite3.Connection | None = None
    try:
        copy = sqlite3.connect(partial)
        src_conn.backup(copy)
        mode = _journal_delete(copy)
        if mode != "delete":
            raise sqlite3.OperationalError(f"the backup copy kept journal_mode {mode!r}")
        copy.close()
        copy = None
        _fsync(partial)
        os.replace(partial, dest)
    except BaseException:
        if copy is not None:
            copy.close()
        partial.unlink(missing_ok=True)
        raise
    return dest


# --------------------------------------------------------------------------- periodic backups (T12)

STAMP_FORMAT = "%Y%m%dT%H%M%S%fZ"
BACKUP_EVERY_S = 900.0
BACKUP_KEEP_ALL_S = 86400.0
BACKUP_KEEP_DAYS = 7
NAME = re.compile(r"^harness_state\.(periodic|v\d+)\.(\d{8}T\d{12}Z)\.db$")


def stamp_of(name: str) -> datetime | None:
    """The stamp of a ``periodic`` or ``v<N>`` backup file name (``replaced`` and ``.partial`` files: ``None``)."""
    m = NAME.match(name)
    return None if m is None else datetime.strptime(m.group(2), STAMP_FORMAT).replace(tzinfo=UTC)


def take(conn: sqlite3.Connection, now: datetime, db_path: Path) -> Path:
    """A periodic copy: ``db_dir(db_path)/harness_state.periodic.<stamp>.db``."""
    return write_copy(conn, db_dir(db_path) / f"harness_state.periodic.{now.strftime(STAMP_FORMAT)}.db")


def rotate(directory: Path, now: datetime) -> list[Path]:
    """Prune ``periodic`` files: all of the last 24 h, then the newest per UTC day for 7 days → the deleted."""
    kept_days: set[str] = set()
    deleted = []
    periodic = sorted(
        ((s, p) for p in directory.glob("harness_state.periodic.*.db") if (s := stamp_of(p.name)) is not None),
        reverse=True,
    )
    for stamp, path in periodic:
        age = (now - stamp).total_seconds()
        day = stamp.strftime("%Y-%m-%d")
        if age < BACKUP_KEEP_ALL_S:
            continue
        if age < BACKUP_KEEP_DAYS * 86400 and day not in kept_days:
            kept_days.add(day)
            continue
        path.unlink(missing_ok=True)
        deleted.append(path)
    return deleted


def choose_source(directory: Path) -> Path | None:
    """The newest ``periodic`` or ``v<N>`` backup by stamp; never a ``replaced`` or ``.partial`` file."""
    found = [(s, p) for p in directory.glob("harness_state.*.db") if (s := stamp_of(p.name)) is not None]
    return max(found)[1] if found else None
