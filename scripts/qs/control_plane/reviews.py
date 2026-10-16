"""Review sessions (#407): the global review of a run and the cross-run coherence review.

A review is a row, not a node (D1): ``reviews`` (one per review), ``review_tasks`` (its scope, with the base
and head it reviewed and its outcome per task) and ``review_triggers`` (every trigger, and the review
pilot's queue). ``request_locked`` is the one entry point that computes a scope, skips, joins, defers or
creates (D3, D8); ``tool review`` and the pilot (``reviewpilot``) both call it.

See ``docs/workflow/control-plane.md`` "Review sessions".
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from . import alerts, daemon, db, errors, hooks, liveness, nodes, paths, runs, tokens, tools
from . import clock as clock_mod
from .schema_reviews import AGENTS, IN_FLIGHT, KINDS, REVIEW_TTL_S, SOURCES

GLOBAL = "global"
CROSS_RUN = "cross_run"
TOOL_NAME = "review"
REVIEW_PERMISSION_MODE = "dontAsk"  # D6: "auto" is the fallback (`launch_args` builds both)


# --------------------------------------------------------------------------- where a review runs (D5)


def db_tag() -> str:
    """``sha256(str(<the selected DB, resolved>))[:8]``: keeps two DBs' reviews (and launch names) apart."""
    return hashlib.sha256(str(paths.select_db().resolve()).encode()).hexdigest()[:8]


def review_dir(review_id: str, *, create: bool = True) -> Path:
    """``<QS_CP_REVIEW_DIR>/<db tag>/<review id>/`` — every level (and its ``in/``) a private 0700 directory."""
    root = paths.review_root()
    tagged = root / db_tag()
    directory = tagged / review_id
    if create:
        for d in (root, tagged, directory, directory / "in"):
            paths.ensure_private_dir(d)
    return directory


def launch_name(review_id: str) -> str:
    """The ``claude --bg -n`` name of a review session (unique per DB)."""
    return f"qs-review-{db_tag()}-{review_id}"


# --------------------------------------------------------------------------- request_locked (D3, D7, D8)


@dataclass(frozen=True)
class Skipped:
    """Nothing to review: no row written (a pilot queue row is marked ``skipped``)."""

    reason: str
    tasks: list[str]


@dataclass(frozen=True)
class Joined:
    """An in-flight review of the same key whose heads are not captured yet absorbs this trigger."""

    review_id: str


@dataclass(frozen=True)
class Deferred:
    """An in-flight review of the key already captured its heads (or, global, covers another scope)."""

    review_id: str


@dataclass(frozen=True)
class Created:
    review_id: str
    tasks: list[str]


Requested = Skipped | Joined | Deferred | Created


def _global_scope(conn: sqlite3.Connection, run_id: str) -> list[str]:
    """Every deliverable or work item of the run that is not terminal (D3), in insertion order."""
    marks = ", ".join("?" for _ in tokens.TERMINAL_TASK_STATES)
    rows = conn.execute(
        "SELECT id FROM tasks WHERE run_id = ? AND (is_deliverable = 1 OR deliverable_id IS NOT NULL)"
        f" AND state NOT IN ({marks}) ORDER BY rowid",
        (run_id, *sorted(tokens.TERMINAL_TASK_STATES)),
    ).fetchall()
    return [r["id"] for r in rows]


def _cross_run_scope(
    conn: sqlite3.Connection, raw: Collection[str], merged_task: str | None
) -> tuple[list[str], str | None]:
    """The kept tasks of a cross-run event (re-filtered at every request) → ``(kept, skip reason or None)``.

    A task of a closed run (or of no run) is dropped, except a merge event's merged task. Skipped when nothing
    left is non-terminal, when an alert event spans fewer than 2 open runs, or when a merge event has no open
    run other than the merged task's.
    """
    kept: list[sqlite3.Row] = []
    for task_id in sorted(set(raw)):
        row = conn.execute(
            "SELECT t.id, t.run_id, t.state, r.state AS run_state FROM tasks t LEFT JOIN runs r ON r.id = t.run_id"
            " WHERE t.id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            continue
        if row["run_state"] == "open" or task_id == merged_task:
            kept.append(row)
    ids = [r["id"] for r in kept]
    if not any(r["state"] not in tokens.TERMINAL_TASK_STATES for r in kept):
        return ids, "no_open_task"
    open_runs = {r["run_id"] for r in kept if r["run_state"] == "open"}
    if merged_task is None:
        return ids, None if len(open_runs) >= 2 else "single_run"
    merged_run = next((r["run_id"] for r in kept if r["id"] == merged_task), None)
    return ids, None if open_runs - {merged_run} else "no_other_run"


def coalesce_key(kind: str, *, run_id: str | None, tasks: Collection[str]) -> str:
    """``global:<run>`` or ``cross_run:<sorted kept task ids joined by "|">`` (D8)."""
    return f"{GLOBAL}:{run_id}" if kind == GLOBAL else f"{CROSS_RUN}:" + "|".join(sorted(tasks))


def _write_trigger(
    conn: sqlite3.Connection,
    now: str,
    *,
    row_id: int | None,
    review_id: str | None,
    outcome: str,
    actor: str,
    trigger: Mapping[str, Any],
    tool_key: str,
) -> None:
    if row_id is not None:  # the pilot's queue row: updated, never re-inserted (C3-4)
        conn.execute("UPDATE review_triggers SET review_id = ?, outcome = ? WHERE id = ?", (review_id, outcome, row_id))
        return
    conn.execute(
        'INSERT INTO review_triggers (review_id, actor, "trigger", tool_key, outcome, at) VALUES (?, ?, ?, ?, ?, ?)',
        (review_id, actor, _canonical(trigger), tool_key, outcome, now),
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_locked(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    kind: str,
    run_id: str | None = None,
    tasks: Collection[str] | None = None,
    trigger: Mapping[str, Any],
    actor: str,
    tool_key: str,
    grant: bool,
    trigger_row_id: int | None = None,
) -> Requested:
    """Inside the caller's write transaction: scope (D3), then skip, join, defer or create (D8).

    ``kind = global`` reviews ``run_id``; ``kind = cross_run`` reviews ``tasks`` re-filtered (a merge event
    names its ``merged_task`` in ``trigger``). On ``Created`` the row is ``launching`` with a session ``nonce``
    (``grant=False``, the run-token path) or ``requested`` with a ``grant_nonce`` (``grant=True``, the pilot).
    The trigger row is updated when ``trigger_row_id`` is given (the pilot's queue), else inserted — except on
    ``Skipped`` / ``Deferred``.
    """
    if kind not in KINDS:
        raise errors.CpError("USAGE", f"kind must be one of {', '.join(KINDS)}: {kind!r}")
    now = db.now(clock)
    if kind == GLOBAL:
        if run_id is None:
            raise errors.CpError("USAGE", "a global review needs its run")
        scope = _global_scope(conn, run_id)
        skip = None if len(scope) >= 2 else "single_task"
    else:
        merged = trigger.get("merged_task")
        scope, skip = _cross_run_scope(conn, tasks or (), merged if isinstance(merged, str) else None)

    def record(review_id: str | None, outcome: str) -> None:
        _write_trigger(
            conn,
            now,
            row_id=trigger_row_id,
            review_id=review_id,
            outcome=outcome,
            actor=actor,
            trigger=trigger,
            tool_key=tool_key,
        )

    if skip is not None:
        if trigger_row_id is not None:
            record(None, "skipped")
        return Skipped(skip, scope)
    key = coalesce_key(kind, run_id=run_id, tasks=scope)
    marks = ", ".join("?" for _ in IN_FLIGHT)
    live = conn.execute(
        f"SELECT id, heads_at FROM reviews WHERE coalesce_key = ? AND state IN ({marks})", (key, *IN_FLIGHT)
    ).fetchone()
    if live is not None:
        covered = {r[0] for r in conn.execute("SELECT task_id FROM review_tasks WHERE review_id = ?", (live["id"],))}
        if live["heads_at"] is None and covered == set(scope):
            record(live["id"], "joined")
            return Joined(live["id"])
        return Deferred(live["id"])  # it would never see the new commits: retried after it closes
    review_id = db.next_id(conn, "review", "V")
    conn.execute(
        "INSERT INTO reviews (id, kind, coalesce_key, home_run_id, state, agent, permission_mode, nonce, grant_nonce,"
        " name, tool_key, requested_by, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            review_id,
            kind,
            key,
            run_id if kind == GLOBAL else None,
            "requested" if grant else "launching",
            AGENTS[kind],
            REVIEW_PERMISSION_MODE,
            None if grant else tokens.new_nonce(),
            tokens.new_nonce() if grant else None,
            launch_name(review_id),
            f"{TOOL_NAME}/{tool_key}",
            actor,
            now,
            now,
        ),
    )
    conn.executemany(
        "INSERT INTO review_tasks (review_id, task_id, run_id) SELECT ?, id, run_id FROM tasks WHERE id = ?",
        [(review_id, task_id) for task_id in scope],
    )
    record(review_id, "created")
    return Created(review_id, scope)


# --------------------------------------------------------------------------- closing a review, and its alert


def finding_ids(conn: sqlite3.Connection, review_id: str, task_id: str | None = None) -> list[int]:
    """The findings this review opened (their actor is ``review:<id>``), optionally on one task."""
    sql = "SELECT id FROM findings WHERE actor = ?" + ("" if task_id is None else " AND task_id = ?")
    params = (f"review:{review_id}",) if task_id is None else (f"review:{review_id}", task_id)
    return [int(r[0]) for r in conn.execute(sql + " ORDER BY id", params)]


def open_runs_of(conn: sqlite3.Connection, review_id: str) -> list[str]:
    """The open runs the review has a task in, sorted."""
    rows = conn.execute(
        "SELECT DISTINCT rt.run_id FROM review_tasks rt JOIN runs r ON r.id = rt.run_id"
        " WHERE rt.review_id = ? AND r.state = 'open' ORDER BY rt.run_id",
        (review_id,),
    ).fetchall()
    return [r[0] for r in rows]


def close_locked(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    review_id: str,
    *,
    to: str,
    error: str,
    where: str,
    params: Sequence[Any] = (),
) -> bool:
    """A **guarded** move to ``failed`` / ``abandoned`` (``where`` re-evaluates its condition in SQL).

    Only when one row changed, posts one ``review_failed`` event to each open run of its scope (alerts are per
    run), with the findings it opened before it closed (C17). → whether the row moved.
    """
    now = db.now(clock)
    error = tools.redact(error)[-500:]
    cur = conn.execute(
        f"UPDATE reviews SET state = ?, error = ?, finished_at = ?, updated_at = ? WHERE id = ? AND ({where})",
        (to, error, now, now, review_id, *params),
    )
    if cur.rowcount != 1:
        return False
    kind = conn.execute("SELECT kind FROM reviews WHERE id = ?", (review_id,)).fetchone()["kind"]
    payload = {
        "review_id": review_id,
        "kind": kind,
        "state": to,
        "error": error,
        "finding_ids": finding_ids(conn, review_id),
    }
    for run_id in open_runs_of(conn, review_id):
        alerts.event_locked(
            conn, clock, kind=alerts.REVIEW_FAILED, subject=f"review:{review_id}:{to}", run_id=run_id, payload=payload
        )
    return True


# --------------------------------------------------------------------------- the launch: permissions, settings, prompt (D6)

_SAFE_PATH = re.compile(r"[A-Za-z0-9_./@%+=:,-]+")  # no whitespace, no shell metacharacter (C2-11)
CP_COMMANDS = ("snapshot", "task show", "ledger show", "review read", "finding open", "review finish")


def python_for(main: Path) -> str:
    """``<MAIN>/venv/bin/python``, else this interpreter (the watchdog's rule, D10)."""
    venv = main / "venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def cp_prefix(main: Path, python: str) -> str:
    return f"{python} {main}/scripts/qs/cp.py"


def allowed_tools(main: Path, python: str, directory: Path) -> list[str]:
    """The pinned ``dontAsk`` allowlist: ``in/`` read and write, and exactly the six ``cp.py`` commands.

    A path holding whitespace or a shell metacharacter is ``POLICY_REFUSED``: the unquoted pinned form would
    stop matching.
    """
    in_dir = directory / "in"
    for what, value in (("the main checkout", str(main)), ("the interpreter", python), ("the review dir", str(in_dir))):
        if not _SAFE_PATH.fullmatch(value):
            raise errors.CpError("POLICY_REFUSED", f"{what} path {value!r} holds whitespace or a shell metacharacter")
    prefix = cp_prefix(main, python)
    return [f"Read(/{in_dir}/**)", f"Write(/{in_dir}/**)", *(f"Bash({prefix} {cmd}:*)" for cmd in CP_COMMANDS)]


def launch_args(mode: str, main: Path, python: str, directory: Path) -> list[str]:
    """The permission arguments of a review launch, in ``dontAsk`` (pinned allowlist) or ``auto`` (the fallback)."""
    if mode == "dontAsk":
        return ["--permission-mode", "dontAsk", "--allowedTools=" + ",".join(allowed_tools(main, python, directory))]
    if mode == "auto":
        return ["--permission-mode", "auto", "--disallowedTools=Edit,NotebookEdit"]
    raise errors.CpError("USAGE", f"unknown review permission mode {mode!r}")


def settings(main: Path, effort: str | None) -> str:
    """The hook fragment (its ``PreToolUse`` DB-access deny applies to any session), the effort, and the DB."""
    fragment: dict[str, Any] = {
        **hooks.hooks_settings("node", main),
        "env": {"QS_CP_DB": str(paths.select_db().resolve())},
    }
    if effort:
        fragment["effortLevel"] = effort
    return json.dumps(fragment, sort_keys=True)


def session_token(row: sqlite3.Row) -> str:
    return tokens.mint("review", row["id"], 1, row["nonce"])


def prompt(row: sqlite3.Row, scope: Sequence[Mapping[str, Any]], *, prefix: str, deadline: str) -> str:
    """The facts a review session starts from; its agent file carries the instructions."""
    lines = [
        f"You are review {row['id']} ({row['kind']}) of the Quiet Solar Control Plane.",
        f"Control Plane command prefix (exactly this, one command per Bash call): {prefix}",
        f"Your input directory, the only place you read or write files: {Path(row['dir']) / 'in'}",
        f"Your session token (only for `finding open` and `review finish`, as `--token`): {session_token(row)}",
        f"Your finding source: {SOURCES[row['kind']]}",
        f"Deadline: {deadline} (after it your token is void).",
        "Scope (task | run | title | branch | base_sha | head_sha | head_state):",
    ]
    for t in scope:
        lines.append(
            f"- {t['task_id']} | {t['run_id']} | {json.dumps(t['title'], ensure_ascii=False)} | {t['branch'] or '-'}"
            f" | {t['base_sha'] or '-'} | {t['head_sha'] or '-'} | {t['head_state']}"
        )
    lines.append("Follow your agent instructions, call `review finish` once, then stop.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- base and head per task (D3)

GIT_TIMEOUT_S = 60.0
ABORT_CODES = frozenset({124, 127})  # the runner's timeout / exec failure: git is unusable, the step fails
_HEX = re.compile(r"[0-9a-f]{7,64}")


class _Git:
    def __init__(self, ctx: tools.StepCtx) -> None:
        self.ctx = ctx
        self._main: tuple[str | None] | None = None

    def run(self, *args: str) -> tuple[int, str]:
        res = self.ctx.runner.run(["git", "-C", str(self.ctx.main), *args], cwd=self.ctx.main, timeout=GIT_TIMEOUT_S)
        if res.returncode in ABORT_CODES:
            raise tools.StepFailed(f"git {' '.join(args[:2])} exited {res.returncode}: {res.stderr.strip()[-200:]}")
        return res.returncode, res.stdout.strip()

    def sha(self, *args: str) -> str | None:
        """The output of a successful call (only ``0`` is present or resolved), else ``None``."""
        code, out = self.run(*args)
        return out if code == 0 and out else None

    def ref(self, ref: str) -> str | None:
        return self.sha("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")

    def main_base(self) -> str | None:
        """Child 14's overlap rule: the newer of local ``main`` and ``origin/main``; ``origin/main`` if diverged."""
        if self._main is None:
            local, remote = self.ref("refs/heads/main"), self.ref("refs/remotes/origin/main")
            if local is None or remote is None or local == remote:
                found = local or remote
            elif self.run("merge-base", "--is-ancestor", local, remote)[0] == 0:
                found = remote
            elif self.run("merge-base", "--is-ancestor", remote, local)[0] == 0:
                found = local
            else:
                found = remote
            self._main = (found,)
        return self._main[0]


def _heads(git: _Git, task: sqlite3.Row) -> tuple[str | None, str | None, str]:
    """``(base_sha, head_sha, head_state)`` of one task: the first row of D3's table that applies."""
    merge = task["merge_sha"]
    if merge and _HEX.fullmatch(merge) and git.run("cat-file", "-e", f"{merge}^{{commit}}")[0] == 0:
        return git.sha("rev-parse", "--verify", "--quiet", f"{merge}^1"), merge, "merged"
    if not task["branch"]:
        return None, None, "plan"
    head = git.ref(f"refs/heads/{task['branch']}")
    if head is None:
        return None, None, "unresolved"
    base_ref = git.ref(f"refs/heads/{task['deliverable_branch']}") if task["deliverable_branch"] else None
    if base_ref is None:
        base_ref = git.main_base()
    base = git.sha("merge-base", base_ref, head) if base_ref else None
    return base, head, "branch"


# --------------------------------------------------------------------------- the `review` tool (D7)


def _row(conn: sqlite3.Connection, review_id: str) -> sqlite3.Row:
    """A review row by an id already validated (a token's, or one this call recorded): never deleted."""
    row: sqlite3.Row = conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
    return row


def _scope_rows(conn: sqlite3.Connection, review_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT rt.*, t.title, t.branch, t.merge_sha, d.branch AS deliverable_branch FROM review_tasks rt"
        " JOIN tasks t ON t.id = rt.task_id LEFT JOIN tasks d ON d.id = t.deliverable_id"
        " WHERE rt.review_id = ? ORDER BY rt.task_id",
        (review_id,),
    ).fetchall()


def _authorize(conn: sqlite3.Connection, who: tokens.Principal, args: Mapping[str, Any], key: str) -> None:
    """The token × args matrix (V21): a run token asks for its own run's global review; a launch grant launches
    its own review, under the key it was minted for (C2-9, D3-1)."""
    if who.kind == "run":
        unknown = sorted(set(args) - {"kind", "run", "trigger", "review"})
        if unknown:
            raise errors.CpError("USAGE", f"unknown args: {', '.join(unknown)}")
        if "review" in args or (args.get("kind") in KINDS and args.get("kind") != GLOBAL):
            raise errors.CpError("CONFLICT", "a run token asks only for the global review of its own run")
        if args.get("kind") != GLOBAL:
            raise errors.CpError("USAGE", f"args.kind must be {GLOBAL!r}")
        if not isinstance(args.get("run"), str) or not args["run"]:
            raise errors.CpError("USAGE", "args.run is required")
        if runs.resolve(conn, args["run"])["id"] != who.run_id:
            raise errors.CpError("CONFLICT", f"the token is for run {who.run_id}, not {args['run']}")
        if not isinstance(args.get("trigger", {}), dict):
            raise errors.CpError("USAGE", "args.trigger must be a JSON object")
        return
    if dict(args) != {"review": who.review_id}:
        raise errors.CpError("CONFLICT", f"a launch grant launches only its own review {who.review_id}")
    if _row(conn, str(who.review_id))["tool_key"] != f"{TOOL_NAME}/{key}":
        raise errors.CpError("CONFLICT", f"the launch grant of {who.review_id} is not for key {key!r}")


def _review_id(ctx: tools.StepCtx) -> str | None:
    reserved = ctx.outputs.get("reserve") or {}
    rid = reserved.get("review_id")
    return str(rid) if rid else None


def _reserve(ctx: tools.StepCtx) -> None:
    who = ctx._call.who
    with ctx.write() as conn:
        if who.kind == "run":
            out = request_locked(
                conn,
                ctx.clock,
                kind=GLOBAL,
                run_id=who.run_id,
                trigger=ctx.args.get("trigger") or {},
                actor=who.actor,
                tool_key=ctx.key,
                grant=False,
            )
            if isinstance(out, Deferred):  # released before any effect: the same key works later (D8)
                raise errors.CpError(
                    "BUSY",
                    f"a review of this key ({out.review_id}) is running; replay this key after it closes",
                    review_id=out.review_id,
                )
            if isinstance(out, Skipped):
                ctx.record_output({"skipped": out.reason, "tasks": out.tasks})
            elif isinstance(out, Joined):
                ctx.record_output({"coalesced_into": out.review_id})
            else:
                ctx.record_output({"review_id": out.review_id})
            return
        row = _row(conn, str(who.review_id))
        if row["state"] == "requested":
            now = db.now(ctx.clock)
            conn.execute(
                "UPDATE reviews SET state = 'launching', nonce = ?, updated_at = ? WHERE id = ? AND state = 'requested'",
                (tokens.new_nonce(), now, row["id"]),
            )
        elif row["state"] != "launching":  # a launching row under this key is a takeover: nothing re-minted
            raise errors.CpError("INVALID_STATE", f"review {row['id']} is {row['state']}")
        ctx.record_output({"review_id": row["id"]})


def _guarded(fn: Callable[[tools.StepCtx, str], dict[str, Any]]) -> Callable[[tools.StepCtx], dict[str, Any]]:
    """A noop after a skip or a join; any ``StepFailed`` first moves the review to ``failed`` (guarded, S2)."""

    def run(ctx: tools.StepCtx) -> dict[str, Any]:
        rid = _review_id(ctx)
        if rid is None:
            return {"noop": True}
        try:
            return fn(ctx, rid)
        except tools.StepFailed as exc:
            with ctx.write() as conn:
                close_locked(
                    conn, ctx.clock, rid, to="failed", error=exc.detail, where="state IN ('requested', 'launching')"
                )
            raise

    return run


def _prepare(ctx: tools.StepCtx, rid: str) -> dict[str, Any]:
    with ctx.write() as conn:  # the capture marker, before git runs: a join stops from this instant (C3-1)
        now = db.now(ctx.clock)
        stamped = conn.execute(
            "UPDATE reviews SET heads_at = ?, updated_at = ? WHERE id = ? AND state = 'launching' AND heads_at IS NULL",
            (now, now, rid),
        ).rowcount
        row = _row(conn, rid)
        if not stamped and row["state"] != "launching":
            raise tools.StepFailed(f"review {rid} is {row['state']}")
        pending = [r for r in _scope_rows(conn, rid) if r["head_state"] is None]
    git = _Git(ctx)
    captured = [(t["task_id"], *_heads(git, t)) for t in pending]
    resolver = ctx._call.ctx.resolve_model
    if resolver is None:
        raise tools.StepFailed("no model resolver")
    try:
        model, effort = resolver(row["agent"], None)
    except Exception as exc:  # noqa: BLE001 — an unknown agent or a broken policy: the review failed
        raise tools.StepFailed(f"model policy: {exc}") from exc
    source = ctx.main / ".claude" / "agents" / f"{row['agent']}.md"
    try:
        directory = review_dir(rid, create=True)
    except (errors.CpError, OSError) as exc:
        raise tools.StepFailed(f"review directory: {exc}") from exc
    if not source.is_file():
        raise tools.StepFailed(f"no agent file {source}: render the agents on main")
    try:
        (directory / ".claude" / "agents").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, directory / ".claude" / "agents" / source.name)
    except OSError as exc:
        raise tools.StepFailed(f"agent copy: {exc}") from exc
    with ctx.write() as conn:
        conn.executemany(
            "UPDATE review_tasks SET base_sha = ?, head_sha = ?, head_state = ?"
            " WHERE review_id = ? AND task_id = ? AND head_state IS NULL",
            [(base, head, state, rid, task_id) for task_id, base, head, state in captured],
        )
        moved = conn.execute(
            "UPDATE reviews SET model = ?, effort = ?, dir = ?, updated_at = ? WHERE id = ? AND state = 'launching'",
            (model, effort, str(directory), db.now(ctx.clock), rid),
        ).rowcount
        if not moved:
            raise tools.StepFailed(f"review {rid} is {_row(conn, rid)['state']}")
    return {"model": model, "effort": effort, "dir": str(directory)}


def _launch(ctx: tools.StepCtx, rid: str) -> dict[str, Any]:
    with ctx.write() as conn:
        now = db.now(ctx.clock)
        if not conn.execute(
            "UPDATE reviews SET launch_at = ?, updated_at = ? WHERE id = ? AND state = 'launching'", (now, now, rid)
        ).rowcount:
            raise tools.StepFailed(f"review {rid} is {_row(conn, rid)['state']}")
        row = _row(conn, rid)
        scope = [dict(r) for r in _scope_rows(conn, rid)]
    directory = Path(row["dir"])
    python = python_for(ctx.main)
    try:
        permission = launch_args(str(row["permission_mode"]), ctx.main, python, directory)
        options = settings(ctx.main, row["effort"])
    except errors.CpError as exc:
        raise tools.StepFailed(exc.detail) from exc
    deadline = clock_mod.iso(clock_mod.parse(now) + timedelta(seconds=REVIEW_TTL_S))
    text = prompt(row, scope, prefix=cp_prefix(ctx.main, python), deadline=deadline)
    argv = ["--agent", row["agent"], "-n", row["name"], "--model", str(row["model"]), *permission]
    argv += ["--settings", options, text]
    res = ctx.claude.spawn_bg(argv, cwd=directory, env_remove=daemon.stripped_names(os.environ))
    if not res.ok:
        raise tools.StepFailed(f"claude --bg exited {res.returncode}", {"exit_code": res.returncode}, res.returncode)
    return {"launched": True}


def _identify(ctx: tools.StepCtx, rid: str) -> dict[str, Any]:
    with ctx.write() as conn:
        row = _row(conn, rid)
    if row["state"] in ("failed", "abandoned"):
        raise tools.StepFailed(f"review {rid} is {row['state']}")
    deadline = clock_mod.parse(row["launch_at"]).timestamp() + nodes.LAUNCH_SETTLE_S
    while True:
        listing = ctx.claude.try_agents()
        found = None if listing is None else liveness.find(listing, name=row["name"])
        if found is not None:
            return {"session_id": found.session_id, "short_id": found.id}
        if ctx.clock.now().timestamp() >= deadline:
            raise tools.StepFailed(f"review session {row['name']} was never listed")
        ctx.clock.sleep(tools.IDENTIFY_POLL_S)


def _steps(task: tools.TaskRow, args: Mapping[str, Any]) -> Sequence[tools.Step]:
    return (
        tools.Step("reserve", _reserve),
        tools.Step("prepare", _guarded(_prepare)),
        tools.Step("launch", _guarded(_launch), detach=True),
        tools.Step("identify", _guarded(_identify)),
    )


def _probe(ctx: tools.StepCtx) -> dict[str, Any | None]:
    """A launched review (``launch_at`` set) is never launched again: a takeover only finishes ``identify``."""
    with ctx.write() as conn:
        row = conn.execute("SELECT * FROM reviews WHERE tool_key = ?", (f"{TOOL_NAME}/{ctx.key}",)).fetchone()
    if row is None or row["launch_at"] is None:
        return {}
    return {
        "prepare": {"model": row["model"], "effort": row["effort"], "dir": row["dir"], "probed": True},
        "launch": {"launched": True, "probed": True},
    }


def _on_success(conn: sqlite3.Connection, ctx: tools.StepCtx) -> dict[str, Any]:
    reserved = ctx.outputs["reserve"]
    if "skipped" in reserved:
        return {"skipped": reserved["skipped"], "tasks": reserved["tasks"]}
    if "coalesced_into" in reserved:
        return {"coalesced_into": reserved["coalesced_into"]}
    rid = str(reserved["review_id"])
    ident = ctx.outputs.get("identify") or {}
    now = db.now(ctx.clock)
    conn.execute(
        "UPDATE reviews SET state = CASE state WHEN 'launching' THEN 'running' ELSE state END, session_id = ?,"
        " short_id = ?, updated_at = ? WHERE id = ?",
        (ident.get("session_id"), ident.get("short_id"), now, rid),
    )
    row = _row(conn, rid)
    if row["state"] in ("failed", "abandoned"):
        raise tools.StepFailed(f"review {rid} is {row['state']}")
    return {
        "review_id": rid,
        "kind": row["kind"],
        "name": row["name"],
        "session_id": row["session_id"],
        "state": row["state"],
        "tasks": [
            {k: t[k] for k in ("task_id", "run_id", "base_sha", "head_sha", "head_state")}
            for t in _scope_rows(conn, rid)
        ],
    }


SPEC = tools.ToolSpec(
    TOOL_NAME,
    _steps,
    probe=_probe,
    on_success=_on_success,
    token_kinds=frozenset({"run", tokens.REVIEW_GRANT}),
    task_bound=False,
    authorize=_authorize,
)


def register_review_tool() -> None:
    """Register ``tool review``; idempotent (``cli`` calls it at import and in ``main``)."""
    if TOOL_NAME not in tools.REGISTRY:
        tools.register(SPEC)
