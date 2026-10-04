"""``wait`` (§7.3): block until the orchestrator's queue has a visible message.

It checks the token first (a stale token exits 3 with no write), registers a
``waiters`` row, makes sure the daemon runs, then polls: heartbeat, token,
schema version (a migration → exit 5 with ``restart_wait``), queue. It never
pops. The ``waiters`` row is always removed on exit, SIGTERM included.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from . import clock as clock_mod
from . import daemon, db, errors, liveness, messages, migrations, procsetup, tokens

POLL_S = 1.0
WAIT_TIMEOUT_S = 6 * 3600.0


def _on_sigterm(signum: int, frame: Any) -> None:
    """``wait``'s SIGTERM handler: unwind through the ``finally`` that removes the waiter."""
    raise SystemExit(128 + signum)


def live_waiter(conn: sqlite3.Connection, probe: liveness.ProcessProbe, run_id: str) -> bool:
    """A ``waiters`` row whose process is alive (a dead pid counts as absent)."""
    rows = conn.execute("SELECT pid, pid_start FROM waiters WHERE run_id = ?", (run_id,)).fetchall()
    return any(probe.alive(r["pid"], r["pid_start"]) for r in rows)


def wait(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    probe: liveness.ProcessProbe,
    ensure: Callable[[], dict[str, Any]],
    *,
    token: str,
    run_ref: str,
    db_path: Any,
    timeout: float = WAIT_TIMEOUT_S,
    poll: float = POLL_S,
    stale_after_s: float = daemon.STALE_AFTER_S,
) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run"})
        run_id = messages._own_run(conn, who, run_ref)
    me = probe.me()
    now = db.now(clock)
    with db.write(conn):
        cur = conn.execute(
            "INSERT INTO waiters (run_id, pid, pid_start, started_at, heartbeat_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, me.pid, me.pid_start, now, now),
        )
        waiter_id = cur.lastrowid
    procsetup.get().install_sigterm(_on_sigterm)
    try:
        ensure()
        last_ensure = clock.now()
        deadline = clock.now().timestamp() + timeout
        while True:
            if db.user_version(conn) != migrations.current_schema_version():
                raise errors.CpError(
                    "SCHEMA_TOO_NEW", "the DB was migrated: restart `wait` with the new code", restart_wait=True
                )
            with db.write(conn):
                conn.execute("UPDATE waiters SET heartbeat_at = ? WHERE id = ?", (db.now(clock), waiter_id))
                tokens.require(conn, token, kinds={"run"})
                pending = messages.visible_count(conn, run_id, messages.ORCHESTRATOR, db.now(clock))
            if pending:
                return {"pending": pending}
            lease = daemon.read_lease(db_path)
            since = (clock.now() - last_ensure).total_seconds()
            if not daemon._fresh(lease, clock, stale_after_s) and since >= stale_after_s:
                ensure()
                last_ensure = clock.now()
            if clock.now().timestamp() >= deadline:
                return {"timeout": True}
            clock.sleep(poll)
    finally:
        try:
            conn.execute("DELETE FROM waiters WHERE id = ?", (waiter_id,))
        except sqlite3.Error:
            pass  # a migration may have changed the table; the row then counts as dead anyway
