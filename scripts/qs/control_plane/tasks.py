"""The task tree (§4 "Task states", §6.5): tasks, transitions, history, deps, criteria, work items.

``TRANSITIONS`` is the only source of truth for task states. A node token may
move its own task only within ``NODE_RANGE`` (and in or out of ``blocked``
from there); every other transition needs the run token.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from . import clock as clock_mod
from . import db, errors, runs, tokens

TERMINAL = frozenset({"merged", "validated", "dropped"})
TRANSITIONS: dict[str, frozenset[str]] = {
    "proposed": frozenset({"ready"}),
    "ready": frozenset({"planning"}),
    "planning": frozenset({"contracted"}),
    "contracted": frozenset({"building"}),
    "building": frozenset({"ready_to_merge"}),
    "ready_to_merge": frozenset({"merged"}),
    "merged": frozenset({"validated"}),
    "validated": frozenset(),
    "blocked": frozenset(),
    "dropped": frozenset(),
}
NODE_RANGE = frozenset({"planning", "contracted", "building", "ready_to_merge"})
KINDS = ("epic", "feature", "bug")
UNBLOCK = "unblock"


def get(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        raise errors.CpError("NOT_FOUND", f"unknown task {task_id}")
    return row


def plan_transition(current: str, to: str, blocked_from: str | None) -> tuple[str, str | None]:
    """``(new state, new blocked_from)`` or ``INVALID_STATE``."""
    if to == UNBLOCK:
        if current != "blocked" or blocked_from is None:
            raise errors.CpError("INVALID_STATE", f"cannot unblock a task in state {current}")
        return blocked_from, None
    if current not in TERMINAL and current != to:
        if to == "blocked":
            return "blocked", current
        if to == "dropped":
            return "dropped", None
    if to in TRANSITIONS.get(current, frozenset()):
        return to, None
    raise errors.CpError("INVALID_STATE", f"forbidden transition {current} → {to}")


def node_may(current: str, new: str, blocked_from: str | None) -> bool:
    if current in NODE_RANGE and new in NODE_RANGE:
        return True
    if current in NODE_RANGE and new == "blocked":
        return True
    return current == "blocked" and blocked_from in NODE_RANGE and new == blocked_from


def apply_transition(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    task_id: str,
    to: str,
    *,
    actor: str,
    node: bool,
    note: str | None = None,
    expect: str | None = None,
) -> dict[str, Any]:
    """Inside the caller's transaction. ``expect`` makes it a compare-and-set on the current state."""
    row = get(conn, task_id)
    if expect is not None and row["state"] != expect:
        raise errors.CpError("INVALID_STATE", f"task {task_id} is {row['state']}, expected {expect}")
    new, blocked_from = plan_transition(row["state"], to, row["blocked_from"])
    if node and not node_may(row["state"], new, row["blocked_from"]):
        raise errors.CpError("CONFLICT", f"a node token may not move its task {row['state']} → {new}")
    now = db.now(clock)
    conn.execute(
        "UPDATE tasks SET state = ?, blocked_from = ?, updated_at = ? WHERE id = ?", (new, blocked_from, now, task_id)
    )
    conn.execute(
        "INSERT INTO task_history (task_id, at, actor, from_state, to_state, note) VALUES (?, ?, ?, ?, ?, ?)",
        (task_id, now, actor, row["state"], new, note),
    )
    return {"task_id": task_id, "from": row["state"], "to": new}


def set_state(
    conn: sqlite3.Connection, clock: clock_mod.Clock, *, token: str, task_id: str, to: str, note: str | None
) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run", "node"}, task_id=task_id)
        return apply_transition(conn, clock, task_id, to, actor=who.actor, node=who.kind == "node", note=note)


def allocate_item_k(conn: sqlite3.Connection, deliverable_id: str) -> int:
    """The next work-item number of a deliverable: never reused, even after a drop (#400)."""
    row = get(conn, deliverable_id)
    k = int(row["next_item_k"])
    conn.execute("UPDATE tasks SET next_item_k = ? WHERE id = ?", (k + 1, deliverable_id))
    return k


def _run_scope(conn: sqlite3.Connection, who: tokens.Principal, run_ref: str | None) -> str | None:
    if run_ref is None:
        return None
    run = runs.resolve(conn, run_ref)
    if run["id"] != who.run_id:
        raise errors.CpError("CONFLICT", f"the token is for run {who.run_id}, not {run_ref}")
    runs.require_open(conn, run["id"])
    return str(run["id"])


def _same_run(row: sqlite3.Row, who: tokens.Principal) -> sqlite3.Row:
    """A referenced task of another run is refused (the ``tokens.require`` rule: NULL or the caller's run).

    The rule is the token's run, not the new task's: a task created with no run may reference a task of no
    run or of the token's own run (``parent``, ``item_of``, a ``task dep``), never one of another run.
    """
    if row["run_id"] not in (None, who.run_id):
        raise errors.CpError("CONFLICT", f"task {row['id']} belongs to run {row['run_id']}, not {who.run_id}")
    return row


def add(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    token: str,
    run_ref: str | None,
    title: str,
    kind: str,
    target: str | None,
    parent: str | None,
    issue: int | None,
    lane: str | None,
    deliverable: bool,
    item_of: str | None,
) -> dict[str, Any]:
    if kind not in KINDS:
        raise errors.CpError("USAGE", f"--kind must be one of {', '.join(KINDS)}")
    if item_of is not None and (deliverable or issue is not None):
        raise errors.CpError("USAGE", ITEM_NOT_DELIVERABLE)
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run"})
        run_id = _run_scope(conn, who, run_ref)
        if parent is not None:
            _same_run(get(conn, parent), who)
        item_k = None
        if item_of is not None:
            owner = _same_run(get(conn, item_of), who)
            if not owner["is_deliverable"]:
                raise errors.CpError("INVALID_STATE", f"task {item_of} is not a deliverable")
            item_k = allocate_item_k(conn, item_of)
        task_id = db.next_id(conn, "task", "T")
        now = db.now(clock)
        conn.execute(
            "INSERT INTO tasks (id, run_id, parent_id, issue_number, title, kind, target, is_deliverable,"
            " deliverable_id, item_k, state, lane, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, ?, ?)",
            (task_id, run_id, parent, issue, title, kind, target, int(deliverable), item_of, item_k, lane, now, now),
        )
        conn.execute(
            "INSERT INTO task_history (task_id, at, actor, from_state, to_state) VALUES (?, ?, ?, NULL, 'proposed')",
            (task_id, now, who.actor),
        )
    return {"task_id": task_id, "item_k": item_k, "run_id": run_id}


# QS-400: a work item lands in its deliverable's PR (branch `QS_<N>_<k>`); a deliverable is its own
# issue `M` (branch `QS_<M>`, its own PR). One task is never both.
ITEM_NOT_DELIVERABLE = (
    "a work item is part of its deliverable's PR (branch QS_<N>_<k>): it has no --deliverable flag and no "
    "--issue — a deliverable is its own issue (branch QS_<M>); add it without --item-of"
)

SETTABLE = ("issue_number", "worktree", "branch", "pr_number", "pr_url", "ci_state", "ci_sha", "merge_sha")


def update_fields(conn: sqlite3.Connection, clock: clock_mod.Clock, task_id: str, fields: dict[str, Any]) -> None:
    """Inside the caller's transaction (also the tools' ``on_success``)."""
    unknown = set(fields) - set(SETTABLE)
    if unknown:
        raise errors.CpError("USAGE", f"not settable: {', '.join(sorted(unknown))}")
    if not fields:
        return
    assignments = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(
        f"UPDATE tasks SET {assignments}, updated_at = ? WHERE id = ?",
        (*fields.values(), db.now(clock), task_id),
    )


def set_fields(
    conn: sqlite3.Connection, clock: clock_mod.Clock, *, token: str, task_id: str, fields: dict[str, Any]
) -> dict[str, Any]:
    if ("pr_number" in fields) != ("pr_url" in fields):
        raise errors.CpError("USAGE", "--pr-number and --pr-url go together")
    if ("ci_state" in fields) != ("ci_sha" in fields):
        raise errors.CpError("USAGE", "--ci-state and --ci-sha go together")
    with db.write(conn):
        tokens.require(conn, token, kinds={"run"}, task_id=task_id)
        row = get(conn, task_id)
        if "issue_number" in fields and row["deliverable_id"] is not None:
            raise errors.CpError("INVALID_STATE", f"task {task_id}: {ITEM_NOT_DELIVERABLE}")
        update_fields(conn, clock, task_id, fields)
    return {"task_id": task_id, "updated": sorted(fields)}


def edit_dep(conn: sqlite3.Connection, *, token: str, action: str, task_id: str, on: str) -> dict[str, Any]:
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run"}, task_id=task_id)
        _same_run(get(conn, on), who)  # both ends of the edge: never a task of another run
        if action == "add":
            if task_id == on:
                raise errors.CpError("USAGE", "a task cannot depend on itself")
            conn.execute("INSERT OR IGNORE INTO task_deps (task_id, depends_on) VALUES (?, ?)", (task_id, on))
        else:
            conn.execute("DELETE FROM task_deps WHERE task_id = ? AND depends_on = ?", (task_id, on))
    return {"task_id": task_id, "depends_on": on, "action": action}


def edit_membership(
    conn: sqlite3.Connection, *, token: str, table: str, action: str, run_ref: str, task_id: str
) -> dict[str, Any]:
    assert table in ("run_roots", "work_list")
    with db.write(conn):
        who = tokens.require(conn, token, kinds={"run"}, task_id=task_id)
        run_id = _run_scope(conn, who, run_ref)
        if action == "add":
            conn.execute(f"INSERT OR IGNORE INTO {table} (run_id, task_id) VALUES (?, ?)", (run_id, task_id))
        else:
            conn.execute(f"DELETE FROM {table} WHERE run_id = ? AND task_id = ?", (run_id, task_id))
    return {"run_id": run_id, "task_id": task_id, "action": action}


def criteria_set(conn: sqlite3.Connection, *, token: str, task_id: str, lines: list[str]) -> dict[str, Any]:
    texts = [ln.strip() for ln in lines if ln.strip()]
    with db.write(conn):
        tokens.require(conn, token, kinds={"run"}, task_id=task_id)
        get(conn, task_id)
        if conn.execute("SELECT 1 FROM criteria WHERE task_id = ? AND validated_at IS NOT NULL", (task_id,)).fetchone():
            raise errors.CpError("INVALID_STATE", "the criteria are validated; they can no longer be replaced")
        conn.execute("DELETE FROM criteria WHERE task_id = ?", (task_id,))
        conn.executemany(
            "INSERT INTO criteria (task_id, idx, text) VALUES (?, ?, ?)",
            [(task_id, i, t) for i, t in enumerate(texts, 1)],
        )
    return {"task_id": task_id, "count": len(texts)}


def criteria_validate(conn: sqlite3.Connection, clock: clock_mod.Clock, *, token: str, task_id: str) -> dict[str, Any]:
    with db.write(conn):
        tokens.require(conn, token, kinds={"run"}, task_id=task_id)
        cur = conn.execute("UPDATE criteria SET validated_at = ? WHERE task_id = ?", (db.now(clock), task_id))
        if cur.rowcount == 0:
            raise errors.CpError("INVALID_STATE", f"task {task_id} has no criteria")
    return {"task_id": task_id, "validated": cur.rowcount}


def criteria_state(conn: sqlite3.Connection, *, token: str, task_id: str, idx: int, to: str) -> dict[str, Any]:
    if to not in ("open", "met", "waived"):
        raise errors.CpError("USAGE", "--to must be open, met or waived")
    with db.write(conn):
        tokens.require(conn, token, kinds={"run"}, task_id=task_id)
        cur = conn.execute("UPDATE criteria SET state = ? WHERE task_id = ? AND idx = ?", (to, task_id, idx))
        if cur.rowcount == 0:
            raise errors.CpError("NOT_FOUND", f"task {task_id} has no criterion {idx}")
    return {"task_id": task_id, "idx": idx, "state": to}


def record_integration(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    item_task_id: str,
    deliverable_id: str,
    item_tip: str,
    merge_commit: str | None,
    result: str,
    tool_call_key: str | None,
) -> int:
    """One ``integrations`` row (#400), inside the caller's transaction."""
    cur = conn.execute(
        "INSERT INTO integrations (item_task_id, deliverable_id, item_tip, merge_commit, result, tool_call_key, at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (item_task_id, deliverable_id, item_tip, merge_commit, result, tool_call_key, db.now(clock)),
    )
    return int(cur.lastrowid or 0)
