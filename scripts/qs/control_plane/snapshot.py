"""The read-only, board-complete view (§12) — and ``task show``.

No nonce ever leaves the DB through here. It never starts the daemon and
never waits for a migration.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from . import clock as clock_mod
from . import daemon, db, errors, migrations, tasks

SECRET_COLUMNS = frozenset({"nonce"})
ALERTS_LIMIT = 50

KEYS = (
    "schema_version",
    "runs",
    "tasks",
    "task_deps",
    "run_roots",
    "work_list",
    "criteria",
    "task_history",
    "nodes",
    "queues",
    "questions",
    "decisions",
    "digests",
    "reports",
    "integrations",
    "tool_calls_in_flight",
    "locks",
    "cap_slots",
    "daemon",
    "alerts",
)


def _rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    out = []
    for row in conn.execute(sql, params).fetchall():
        item = {k: row[k] for k in row.keys() if k not in SECRET_COLUMNS}  # noqa: SIM118 — sqlite3.Row
        for key in ("payload", "fields", "detail"):
            if isinstance(item.get(key), str):
                try:
                    item[key] = json.loads(item[key])
                except ValueError:
                    pass
        out.append(item)
    return out


def empty() -> dict[str, Any]:
    out: dict[str, Any] = {k: [] for k in KEYS}
    out["schema_version"] = None
    out["daemon"] = None
    return out


def _queues(
    conn: sqlite3.Connection, clock: clock_mod.Clock, run_filter: str, params: tuple[Any, ...]
) -> list[dict[str, Any]]:
    now = clock_mod.iso(clock.now())
    rows = conn.execute(
        "SELECT run_id, recipient,"
        " sum(CASE WHEN state = 'queued' OR (state = 'in_flight' AND visible_at <= ?) THEN 1 ELSE 0 END) AS visible,"
        " sum(CASE WHEN state = 'in_flight' AND visible_at > ? THEN 1 ELSE 0 END) AS in_flight,"
        " sum(CASE WHEN state = 'dead' THEN 1 ELSE 0 END) AS dead,"
        " min(CASE WHEN state IN ('queued', 'in_flight') THEN created_at END) AS oldest"
        f" FROM messages WHERE 1 = 1 {run_filter} GROUP BY run_id, recipient ORDER BY run_id, recipient",
        (now, now, *params),
    ).fetchall()
    return [
        {
            "run_id": r["run_id"],
            "recipient": r["recipient"],
            "depth": r["visible"],
            "in_flight": r["in_flight"],
            "dead": r["dead"],
            "oldest_age_s": clock_mod.age(clock, r["oldest"]),
        }
        for r in rows
    ]


def snapshot(conn: sqlite3.Connection | None, clock: clock_mod.Clock, run_id: str | None = None) -> dict[str, Any]:
    if conn is None:
        return empty()
    flt = "" if run_id is None else "AND run_id = ?"
    p: tuple[Any, ...] = () if run_id is None else (run_id,)
    task_flt = "" if run_id is None else "AND task_id IN (SELECT id FROM tasks WHERE run_id = ?)"
    with db.read(conn):
        runs = _rows(conn, f"SELECT * FROM runs WHERE 1 = 1 {flt.replace('run_id', 'id')} ORDER BY rowid", p)
        leases = {r["run_id"]: r for r in _rows(conn, "SELECT * FROM run_leases")}
        for run in runs:
            run["lease"] = leases.get(run["id"])
        lease = daemon.read_lease_conn(conn)
        return {
            "schema_version": db.user_version(conn),
            "runs": runs,
            "tasks": _rows(conn, f"SELECT * FROM tasks WHERE 1 = 1 {flt} ORDER BY rowid", p),
            "task_deps": _rows(conn, f"SELECT * FROM task_deps WHERE 1 = 1 {task_flt} ORDER BY task_id, depends_on", p),
            "run_roots": _rows(conn, f"SELECT * FROM run_roots WHERE 1 = 1 {flt} ORDER BY run_id, task_id", p),
            "work_list": _rows(conn, f"SELECT * FROM work_list WHERE 1 = 1 {flt} ORDER BY run_id, task_id", p),
            "criteria": _rows(conn, f"SELECT * FROM criteria WHERE 1 = 1 {task_flt} ORDER BY task_id, idx", p),
            "task_history": _rows(conn, f"SELECT * FROM task_history WHERE 1 = 1 {task_flt} ORDER BY id", p),
            "nodes": _rows(conn, f"SELECT * FROM nodes WHERE 1 = 1 {flt} ORDER BY rowid", p),
            "queues": _queues(conn, clock, flt, p),
            "questions": _rows(conn, f"SELECT * FROM questions WHERE 1 = 1 {flt} ORDER BY created_at, rowid", p),
            "decisions": _rows(conn, f"SELECT * FROM decisions WHERE 1 = 1 {flt} ORDER BY id", p),
            "digests": _rows(
                conn,
                f"SELECT task_id, length(CAST(body AS BLOB)) AS bytes, updated_at FROM digests WHERE 1 = 1 {task_flt} ORDER BY task_id",
                p,
            ),
            "reports": _rows(conn, f"SELECT * FROM reports WHERE 1 = 1 {task_flt} ORDER BY id", p),
            "integrations": _rows(
                conn,
                "SELECT * FROM integrations WHERE 1 = 1 "
                + ("" if run_id is None else "AND deliverable_id IN (SELECT id FROM tasks WHERE run_id = ?)")
                + " ORDER BY id",
                p,
            ),
            "tool_calls_in_flight": [
                {**c, "age_s": clock_mod.age(clock, c["started_at"])}
                for c in _rows(
                    conn,
                    "SELECT tool, key, run_id, task_id, actor, holder_pid, holder_pgid, started_at FROM tool_calls"
                    f" WHERE state = 'started' {flt} ORDER BY started_at, tool, key",
                    p,
                )
            ],
            "locks": _rows(conn, "SELECT * FROM locks ORDER BY name"),
            "cap_slots": _rows(conn, "SELECT * FROM cap_slots ORDER BY cap, slot"),
            "daemon": None
            if lease is None
            else {
                **lease,
                "heartbeat_age_s": clock_mod.age(clock, lease["heartbeat_at"]),
                "code_schema_version": migrations.current_schema_version(),
            },
            "alerts": _rows(
                conn, "SELECT * FROM hook_events WHERE decision = 'alert' ORDER BY id DESC LIMIT ?", (ALERTS_LIMIT,)
            ),
        }


def task_show(conn: sqlite3.Connection | None, task_id: str) -> dict[str, Any]:
    if conn is None:
        raise errors.CpError("NOT_FOUND", f"unknown task {task_id} (no DB yet)")
    with db.read(conn):
        tasks.get(conn, task_id)
        task = _rows(conn, "SELECT * FROM tasks WHERE id = ?", (task_id,))[0]
        digest = conn.execute("SELECT body, updated_at FROM digests WHERE task_id = ?", (task_id,)).fetchone()
        return {
            "task": task,
            "digest": None if digest is None else {"body": digest["body"], "updated_at": digest["updated_at"]},
            "criteria": _rows(conn, "SELECT * FROM criteria WHERE task_id = ? ORDER BY idx", (task_id,)),
            "history": _rows(conn, "SELECT * FROM task_history WHERE task_id = ? ORDER BY id", (task_id,)),
            "depends_on": [
                r["depends_on"]
                for r in _rows(
                    conn, "SELECT depends_on FROM task_deps WHERE task_id = ? ORDER BY depends_on", (task_id,)
                )
            ],
            "nodes": _rows(conn, "SELECT * FROM nodes WHERE task_id = ? ORDER BY generation", (task_id,)),
            "reports": _rows(conn, "SELECT * FROM reports WHERE task_id = ? ORDER BY id", (task_id,)),
            "questions": _rows(conn, "SELECT * FROM questions WHERE task_id = ? ORDER BY created_at, id", (task_id,)),
            "decisions": _rows(conn, "SELECT * FROM decisions WHERE task_id = ? ORDER BY id", (task_id,)),
        }
