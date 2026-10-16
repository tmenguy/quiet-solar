"""The review pilot (#407 D10–D12): the tick hook that launches the cross-run coherence review.

Every ``PILOT_EVERY_S`` (a ``ticks.Throttle``), after ``detectors`` and before ``hook_route``:

1. **Reap** the launcher processes that finished (``poll()``): no zombies.
2. **Close dead reviews** (read, probe, beat, write — no subprocess in a ``db.write``): every move is a
   guarded ``UPDATE`` that re-evaluates its condition in SQL, and posts ``review_failed`` only when one row
   changed. ``requested`` / ``launching``: a launcher that never started, whose call failed, whose holder is
   dead, or that is older than ``REQUEST_TTL_S`` → ``failed`` — except a ``launching`` review whose session is
   listed by name, which is adopted (``running``). A launched review fails early only when a known listing does
   not show it; under an unknown listing only ``REQUEST_TTL_S`` fails it. ``running``: unlisted on ``LIVENESS_CONFIRM`` successful
   listings in a row, or past ``REVIEW_TTL_S`` → ``abandoned``.
3. **The queue and the rules** (D11): every new event (an open ``overlap_cross_run`` /
   ``duplicate_task_cross_run`` occurrence, or a merge that touches another run) is written at first sight as
   a ``review_triggers`` row ``pilot:<event>:1`` (``queued``); each tick re-requests the queued rows through
   ``reviews.request_locked`` (the same scope, coalescing and deferral as ``tool review``) and, on a new
   review, starts a **detached** ``cp.py tool review`` process with the review's launch grant. An event whose
   latest launch ``failed`` gets one more attempt (``PILOT_ATTEMPTS``); an ``abandoned`` review is not retried.

The daemon holds no token: the grant (``review:V<n>.0.<nonce>``) is the launcher's only authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import activeloop, alerts, daemon, db, errors, liveness, nodes, paths, reviews, ticks, tokens
from . import clock as clock_mod
from .schema_reviews import CURSOR_KEY, PILOT_PREFIX, REQUEST_TTL_S, REVIEW_TTL_S

if TYPE_CHECKING:
    from .activeloop import Seams

REVIEW_PILOT = "review_pilot"
PILOT_EVERY_S = 30.0
PILOT_RETRY_S = 60.0  # a `requested` review whose launcher has written no `tool_calls` row by then never started
PILOT_ATTEMPTS = 2  # a failed launch is retried once (Q7)
PILOT_KINDS = frozenset({alerts.OVERLAP_CROSS_RUN, alerts.DUPLICATE_TASK_CROSS_RUN})  # not the cycle kind
LIVENESS_CONFIRM = 2
MERGES_PER_TICK = 100
ACTOR = "cp:review_pilot"

_THROTTLE = ticks.Throttle(PILOT_EVERY_S)
_logged: set[str] = set()


def _log_once(message: str) -> None:
    if message not in _logged:
        _logged.add(message)
        sys.stderr.write(f"[cp-review-pilot] {message}\n")
        sys.stderr.flush()


@dataclass
class _State:
    children: list[Any] = field(default_factory=list)  # the launcher processes not reaped yet
    unlisted: dict[str, int] = field(default_factory=dict)  # running review → listings in a row without it


_state = _State()


def review_pilot_hook(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
    if not _THROTTLE.due(clock):
        return
    seams = activeloop.seams()
    _reap()
    close_dead(conn, clock, seams)
    if seams.popen is None:
        _log_once("no popen seam: the review pilot launches nothing")
        return
    discover(conn, clock)
    retry_failed(conn, clock)
    drain(conn, clock, seams)


# --------------------------------------------------------------------------- 1. reap


def _reap() -> None:
    _state.children = [p for p in _state.children if p.poll() is None]


# --------------------------------------------------------------------------- 2. close dead reviews


@dataclass(frozen=True)
class _Move:
    review_id: str
    to: str
    error: str
    where: str
    params: tuple[Any, ...]


_CALL = "SELECT 1 FROM tool_calls t WHERE (t.tool || '/' || t.key) = reviews.tool_key"


def _open_reviews(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT r.id, r.state, r.name, r.session_id, r.created_at, r.launch_at, COALESCE(r.launch_at, r.updated_at) AS since,"
        " t.state AS call_state, t.holder_pid, t.holder_pid_start, t.holder_pgid FROM reviews r"
        " LEFT JOIN tool_calls t ON (t.tool || '/' || t.key) = r.tool_key"
        " WHERE r.state IN ('requested', 'launching', 'running') ORDER BY r.created_at, r.id"
    ).fetchall()


def _launched(row: sqlite3.Row) -> bool:
    """A ``launching`` review whose ``claude --bg`` already ran (its session may be listed)."""
    return bool(row["state"] == "launching" and row["launch_at"] is not None)


def _launch_move(row: sqlite3.Row, clock: clock_mod.Clock, dead_holder: bool, unknown: bool) -> _Move | None:
    """A ``requested`` / ``launching`` review whose launch is over without it: ``failed``.

    ``unknown``: a launched review under an unknown listing — its session may run, so only the TTL applies.
    """
    if row["call_state"] == "failed" and not unknown:
        where = f"state = ? AND EXISTS ({_CALL} AND t.state = 'failed')"
        return _Move(row["id"], "failed", "the launcher's call failed", where, (row["state"],))
    settle = clock_mod.stamp(clock, plus=-nodes.LAUNCH_SETTLE_S)
    if row["call_state"] == "started" and dead_holder and not unknown and row["since"] <= settle:
        return _Move(
            row["id"],
            "failed",
            "the launcher died",
            f"state = ? AND COALESCE(launch_at, updated_at) <= ? AND EXISTS ({_CALL} AND t.state = 'started'"
            " AND t.holder_pid IS ?)",
            (row["state"], settle, row["holder_pid"]),
        )
    retry = clock_mod.stamp(clock, plus=-PILOT_RETRY_S)
    if row["state"] == "requested" and row["call_state"] is None and row["created_at"] <= retry:
        return _Move(
            row["id"],
            "failed",
            "the launcher never started",
            f"state = 'requested' AND created_at <= ? AND NOT EXISTS ({_CALL})",
            (retry,),
        )
    ttl = clock_mod.stamp(clock, plus=-REQUEST_TTL_S)
    if row["since"] <= ttl:
        return _Move(
            row["id"],
            "failed",
            "the launch timed out",
            "state = ? AND COALESCE(launch_at, updated_at) <= ?",
            (row["state"], ttl),
        )
    return None


def _session_move(row: sqlite3.Row, clock: clock_mod.Clock, listing: list[liveness.Agent] | None) -> _Move | None:
    """A ``running`` review past ``REVIEW_TTL_S``, or whose session is gone: ``abandoned`` (never retried)."""
    ttl = clock_mod.stamp(clock, plus=-REVIEW_TTL_S)
    if row["launch_at"] is not None and row["launch_at"] <= ttl:
        return _Move(row["id"], "abandoned", "the review expired", "state = ? AND launch_at <= ?", (row["state"], ttl))
    if listing is None:
        return None  # a failed listing changes nothing
    if liveness.find(listing, session_id=row["session_id"]) is not None:
        _state.unlisted.pop(row["id"], None)
        return None
    seen = _state.unlisted[row["id"]] = _state.unlisted.get(row["id"], 0) + 1
    if seen < LIVENESS_CONFIRM:
        return None
    return _Move(
        row["id"],
        "abandoned",
        "the review session is gone",
        "state = 'running' AND session_id IS ?",
        (row["session_id"],),
    )


def close_dead(conn: sqlite3.Connection, clock: clock_mod.Clock, seams: Seams) -> list[str]:
    """Read, probe, beat, write: the reviews that died → their ids (each moved with its ``review_failed``)."""
    rows = _open_reviews(conn)
    if not rows:
        _state.unlisted.clear()
        return []
    listing = None
    if any(r["state"] == "running" or _launched(r) for r in rows):
        listing = seams.claude.try_agents(timeout=ticks.HOOK_SUBPROCESS_S)
    dead = {
        r["id"]: not seams.probe.holder_alive(r["holder_pid"], r["holder_pid_start"], r["holder_pgid"])
        for r in rows
        if r["call_state"] == "started"
    }
    daemon.beat(conn, clock)
    moves: list[_Move] = []
    adopted: list[tuple[str, liveness.Agent]] = []
    for row in rows:
        if row["state"] == "running":
            move = _session_move(row, clock, listing)
        else:  # a launch that is over without its review: failed (so the event is retried once)
            move = _launch_move(row, clock, dead.get(row["id"], False), _launched(row) and listing is None)
            if move is not None and _launched(row) and listing is not None:  # the session may run without it
                found = liveness.find(listing, name=row["name"])
                if found is not None:
                    adopted.append((row["id"], found))
                    continue
        if move is not None:
            moves.append(move)
    live = {r["id"] for r in rows if r["state"] == "running"}
    _state.unlisted = {k: v for k, v in _state.unlisted.items() if k in live}
    moved: list[str] = []
    if moves or adopted:
        with db.write(conn):
            now = db.now(clock)
            for rid, found in adopted:  # guarded: a row that moved meanwhile is left alone
                conn.execute(
                    "UPDATE reviews SET state = 'running', session_id = ?, short_id = ?, updated_at = ?"
                    " WHERE id = ? AND state = 'launching' AND launch_at IS NOT NULL",
                    (found.session_id, found.id, now, rid),
                )
            for m in moves:
                if reviews.close_locked(
                    conn, clock, m.review_id, to=m.to, error=m.error, where=m.where, params=m.params
                ):
                    moved.append(m.review_id)
                    _state.unlisted.pop(m.review_id, None)
    return moved


# --------------------------------------------------------------------------- 3a. new events (D11)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _queue_locked(
    conn: sqlite3.Connection, clock: clock_mod.Clock, event: str, attempt: int, trigger: dict[str, Any]
) -> None:
    conn.execute(
        "INSERT INTO review_triggers (review_id, actor, \"trigger\", tool_key, outcome, at) VALUES (NULL, ?, ?, ?, 'queued', ?)",
        (ACTOR, _canonical(trigger), f"{PILOT_PREFIX}{event}:{attempt}", db.now(clock)),
    )


def _pilot_rows(table: str = "review_triggers") -> str:
    """The pilot's own rows: an exact, case-sensitive prefix (never ``LIKE``, which ignores case), its actor."""
    return f"substr({table}.tool_key, 1, length(?)) = ? AND {table}.actor = ?"


def _pilot_params(prefix: str) -> tuple[str, str, str]:
    return (prefix, prefix, ACTOR)


def _payload(raw: str | None) -> dict[str, Any]:
    """An alert payload as an object; ``NULL``, non-JSON or a non-object is ``{}`` (no tasks)."""
    try:
        value = json.loads(raw) if raw is not None else None
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _ints(value: Any) -> list[int]:
    """The ``int`` elements of a list (never a nested list, which no set holds); anything else is ``[]``."""
    return [i for i in value if isinstance(i, int)] if isinstance(value, list) else []


def _event_rows(conn: sqlite3.Connection, prefix: str) -> list[sqlite3.Row]:
    """Every pilot trigger row of events whose key starts with ``prefix`` (attempts included)."""
    return conn.execute(
        f'SELECT id, "trigger", tool_key, outcome, review_id FROM review_triggers WHERE {_pilot_rows()} ORDER BY id',
        _pilot_params(f"{PILOT_PREFIX}{prefix}"),
    ).fetchall()


def _alert_events(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
    marks = ", ".join("?" for _ in PILOT_KINDS)
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for r in conn.execute(
        f"SELECT id, kind, subject, payload FROM alerts WHERE cleared_at IS NULL AND kind IN ({marks}) ORDER BY id",
        tuple(sorted(PILOT_KINDS)),
    ).fetchall():
        group = groups.setdefault((r["kind"], r["subject"]), {"ids": [], "tasks": set()})
        group["ids"].append(r["id"])
        tasks = _payload(r["payload"]).get("tasks")
        if isinstance(tasks, list):  # never a string's characters, never a non-iterable
            group["tasks"].update(t for t in tasks if isinstance(t, str))
    for (kind, subject), group in groups.items():
        family = f"alert:{kind}:{hashlib.sha256(subject.encode()).hexdigest()[:16]}:"
        ids = set(group["ids"])
        same = None
        for row in _event_rows(conn, family):
            if ids & set(_ints(_payload(row["trigger"]).get("alert_ids"))):
                same = row  # the latest row of the same occurrence wins (rows are ordered by id)
        if same is not None:  # one occurrence, one event: newly raised rows join its identity
            trigger = _payload(same["trigger"])
            merged = sorted(ids | set(_ints(trigger["alert_ids"])))
            if merged != trigger.get("alert_ids"):
                conn.execute(
                    'UPDATE review_triggers SET "trigger" = ? WHERE id = ?',
                    (_canonical({**trigger, "alert_ids": merged}), same["id"]),
                )
            continue
        event = f"{family}{min(ids)}"
        trigger = {
            "event": event,
            "source": "alert",
            "alert_kind": kind,
            "subject": subject,
            "tasks": sorted(group["tasks"]),
            "alert_ids": sorted(ids),
        }
        _queue_locked(conn, clock, event, 1, trigger)


def _overlap_partners(conn: sqlite3.Connection, task_id: str, own_run: str | None) -> list[str]:
    """Tasks an ``overlap_cross_run`` row (open or cleared) names with ``task_id``, still non-terminal in another open run."""
    named: set[str] = set()
    for (payload,) in conn.execute("SELECT payload FROM alerts WHERE kind = ?", (alerts.OVERLAP_CROSS_RUN,)).fetchall():
        found = _payload(payload).get("tasks", ())
        if isinstance(found, list) and task_id in found:
            named.update(str(t) for t in found if isinstance(t, str) and t != task_id)
    if not named:
        return []
    marks = ", ".join("?" for _ in named)
    terminal = ", ".join("?" for _ in tokens.TERMINAL_TASK_STATES)
    rows = conn.execute(
        f"SELECT t.id FROM tasks t JOIN runs r ON r.id = t.run_id WHERE t.id IN ({marks}) AND r.state = 'open'"
        f" AND t.state NOT IN ({terminal}) AND t.run_id IS NOT ? ORDER BY t.id",
        (*sorted(named), *sorted(tokens.TERMINAL_TASK_STATES), own_run),
    ).fetchall()
    return [r[0] for r in rows]


def _merge_events(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
    cursor = int(conn.execute("SELECT value FROM meta WHERE key = ?", (CURSOR_KEY,)).fetchone()[0])
    rows = conn.execute(
        "SELECT h.id, h.task_id, t.run_id FROM task_history h JOIN tasks t ON t.id = h.task_id"
        " WHERE h.id > ? AND h.to_state = 'merged' ORDER BY h.id LIMIT ?",
        (cursor, MERGES_PER_TICK),
    ).fetchall()
    for row in rows:
        partners = _overlap_partners(conn, row["task_id"], row["run_id"])
        if partners:
            event = f"merge:{row['task_id']}:{row['id']}"
            trigger = {
                "event": event,
                "source": "merge",
                "merged_task": row["task_id"],
                "tasks": sorted([row["task_id"], *partners]),
            }
            _queue_locked(conn, clock, event, 1, trigger)
    if len(rows) == MERGES_PER_TICK:  # bounded: the rest on the next tick
        upto = int(rows[-1]["id"])
    else:  # inside the write transaction: no later merge exists yet
        upto = int(conn.execute("SELECT COALESCE(max(id), ?) FROM task_history", (cursor,)).fetchone()[0])
    # the rows and the cursor in one transaction: a crash loses nothing (C21)
    conn.execute("UPDATE meta SET value = ? WHERE key = ?", (str(upto), CURSOR_KEY))


def discover(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
    """Write every new event as a queued trigger row (alerts first, then merges past the cursor)."""
    with db.write(conn):
        _alert_events(conn, clock)
        _merge_events(conn, clock)


# --------------------------------------------------------------------------- 3b. the retry of a failed launch (Q7)


def _split(tool_key: str) -> tuple[str, int] | None:
    """``pilot:<event>:<n>`` → ``(event, n)``; ``None`` for a key of any other shape (the row is skipped)."""
    event, _, attempt = tool_key[len(PILOT_PREFIX) :].rpartition(":")
    if not event or not (attempt.isascii() and attempt.isdigit()):
        return None
    return event, int(attempt)


def retry_failed(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
    """An event whose latest attempt's review ended ``failed`` gets attempt ``n + 1`` (a new row), while ``n < PILOT_ATTEMPTS``."""
    latest: dict[str, tuple[int, sqlite3.Row]] = {}
    for row in conn.execute(
        'SELECT tr.id, tr."trigger", tr.tool_key, r.state AS review_state FROM review_triggers tr'
        f" LEFT JOIN reviews r ON r.id = tr.review_id WHERE {_pilot_rows('tr')}"
        " ORDER BY tr.id",
        _pilot_params(PILOT_PREFIX),
    ).fetchall():
        split = _split(row["tool_key"])
        if split is not None:
            latest[split[0]] = (split[1], row)
    with db.write(conn):
        for event, (attempt, row) in latest.items():
            trigger = _payload(row["trigger"])
            if row["review_state"] == "failed" and attempt < PILOT_ATTEMPTS and isinstance(trigger.get("tasks"), list):
                _queue_locked(conn, clock, event, attempt + 1, trigger)  # a trigger with no task list: skipped


# --------------------------------------------------------------------------- 3c. the queue → request → launcher (D10)


def _alerts_open(conn: sqlite3.Connection, ids: Sequence[int]) -> bool:
    marks = ", ".join("?" for _ in ids)
    return (
        bool(ids)
        and conn.execute(
            f"SELECT 1 FROM alerts WHERE cleared_at IS NULL AND id IN ({marks}) LIMIT 1", tuple(ids)
        ).fetchone()
        is not None
    )


def drain(conn: sqlite3.Connection, clock: clock_mod.Clock, seams: Seams) -> list[str]:
    """Re-request every queued row; start one launcher per review created → the created review ids."""
    created: list[str] = []
    queued = conn.execute(
        f"SELECT id, \"trigger\", tool_key FROM review_triggers WHERE outcome = 'queued' AND {_pilot_rows()} ORDER BY id",
        _pilot_params(PILOT_PREFIX),
    ).fetchall()
    for row in queued:
        if _split(row["tool_key"]) is None:
            continue  # not a pilot attempt key: never requested
        trigger = _payload(row["trigger"])
        if not isinstance(trigger.get("tasks"), list):
            continue  # a trigger with no task list: never requested
        with db.write(conn):  # the daemon is a singleton: no other writer moves a pilot row meanwhile
            if trigger.get("source") == "alert" and not _alerts_open(conn, _ints(trigger.get("alert_ids"))):
                conn.execute("UPDATE review_triggers SET outcome = 'skipped' WHERE id = ?", (row["id"],))  # C4-3
                continue
            out = reviews.request_locked(
                conn,
                clock,
                kind=reviews.CROSS_RUN,
                tasks=trigger["tasks"],
                trigger=trigger,
                actor=ACTOR,
                tool_key=row["tool_key"],
                grant=True,
                trigger_row_id=row["id"],
            )
            if not isinstance(out, reviews.Created):
                continue  # joined or skipped (the row says so), or deferred (still queued)
            nonce = conn.execute("SELECT grant_nonce FROM reviews WHERE id = ?", (out.review_id,)).fetchone()[0]
        start_launcher(
            conn, clock, seams, out.review_id, row["tool_key"], tokens.mint("review", out.review_id, 0, nonce)
        )
        created.append(out.review_id)
    return created


def launcher_argv(seams: Seams, args_file: str, key: str, grant: str) -> list[str]:
    python = reviews.python_for(seams.main)
    return [
        python,
        f"{seams.main}/scripts/qs/cp.py",
        "tool",
        "review",
        "--key",
        key,
        "--args-file",
        args_file,
        "--token",
        grant,
    ]


def start_launcher(
    conn: sqlite3.Connection, clock: clock_mod.Clock, seams: Seams, review_id: str, key: str, grant: str
) -> bool:
    """A detached ``cp.py tool review`` with the launch grant; never waited on (kept to be reaped).

    Any failure here (the directory, ``args.json``, ``Popen``) moves the review to ``failed`` (guarded), with its
    alert — so the event is retried once.
    """
    assert seams.popen is not None
    try:
        env = daemon.strip_env(os.environ)
        env["QS_CP_DB"] = str(paths.select_db().resolve())  # the daemon's DB, absolute: the launcher's cwd differs
        directory = reviews.review_dir(review_id, create=True)
        args_file = directory / "args.json"
        args_file.write_text(json.dumps({"review": review_id}))
        with open(directory / "launcher.log", "a") as log:
            proc = seams.popen(
                launcher_argv(seams, str(args_file), key, grant),
                cwd=str(seams.main),
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                env=env,
            )
    except (errors.CpError, OSError, ValueError, subprocess.SubprocessError) as exc:
        with db.write(conn):
            reviews.close_locked(
                conn,
                clock,
                review_id,
                to="failed",
                error=f"the launcher did not start: {exc}",
                where="state = 'requested'",
            )
        return False
    _state.children.append(proc)
    return True


def _reset_for_tests() -> None:
    global _state
    _state = _State()
    _logged.clear()
    _THROTTLE.rearm()
