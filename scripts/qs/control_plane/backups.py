"""Backups of the Control Plane DB (QS-406 §10): one private subdirectory per DB, atomic copies.

A leaf module (D17): it imports only ``paths``. Every copy (periodic, migration, a restore's
``replaced``) goes through ``write_copy`` into ``db_dir(db_path)``.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
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
