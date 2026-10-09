"""Schema v2 (QS-406 §11): the active loop's ``alerts`` table and the ``hook_events`` cursor.

Single SQL statements, run one by one with ``execute`` (see ``migrations``). ``selfcheck``,
``selfcheck_override``, ``selfcheck_pending``, ``last_backup_at`` and ``last_backup_error`` are
``meta`` keys written at runtime, so they need no statement here.
"""

from __future__ import annotations

STATEMENTS: tuple[str, ...] = (
    """CREATE TABLE alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL REFERENCES runs(id),
        kind TEXT NOT NULL CHECK (length(kind) BETWEEN 1 AND 32),
        subject TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL,
        message_id INTEGER REFERENCES messages(id),
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        cleared_at TEXT,
        UNIQUE (run_id, fingerprint)
    )""",
    "CREATE INDEX alerts_open ON alerts (run_id, kind, subject, cleared_at)",
    # Seeded past the existing history, so the router never replays it (§7).
    "INSERT OR REPLACE INTO meta (key, value)"
    " SELECT 'hook_events_cursor', CAST(COALESCE(max(id), 0) AS TEXT) FROM hook_events",
)
