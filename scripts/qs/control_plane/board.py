"""The board (#428): a static, read-only page of every run, kept fresh by the daemon.

No socket and no server: the page is one private HTML file per DB, under ``paths.board_dir()``,
rewritten atomically by the ``board`` tick hook and by ``cp.py board``. It is read-only by construction.

``render_to`` is the single render-and-write path: one ``snapshot`` read (released before rendering),
``model``, ``board_render.render``, ``board_render.scrub``, ``write``. The hook takes its
``ticks.Throttle(BOARD_EVERY_S)`` turn first, runs no subprocess and raises no alert: a failed write
(``OSError``, ``CpError``, ``sqlite3.Error``) is logged once per distinct message, and the page's
"last known state" banner shows it.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from . import (
    activeloop,
    alerts,
    board_render,
    codever,
    daemon,
    db,
    errors,
    mergegate,
    migrations,
    nodes,
    paths,
    snapshot,
    tasks,
    ticks,
)
from . import clock as clock_mod

BOARD = "board"
BOARD_EVERY_S = 10.0
BOARD_REFRESH_S = 10
BOARD_STALE_S = 120
BOARD_FORCE_RELOAD_S = 60
GRAPH_TITLE_MAX = 40

DONE_STATES = tasks.TERMINAL - {"dropped"}  # merged, validated
CRITERIA_STATES = frozenset({"proposed", "ready"})  # rule 3: criteria are validated at start-up (decision C)
EMPTY_LINE = "no state readable yet — the daemon refreshes this page"
EMPTY_LINE_NO_DAEMON = "no state readable yet"  # with a note: no daemon will refresh this page


def board_path(db_path: Path) -> Path:
    """``board_dir() / sha256(<resolved db path>)[:8] / "index.html"`` (the ``backups.db_dir`` split)."""
    digest = hashlib.sha256(str(Path(db_path).resolve()).encode()).hexdigest()[:8]
    return paths.board_dir() / digest / "index.html"


def write(html: str, dest: Path) -> Path:
    """Write ``html`` to ``dest`` atomically: a 0600 file of its own per writer, then ``os.replace``.

    A failure before the replace removes the temporary file and re-raises; the previous page stays.
    No fsync: the page is disposable.
    """
    paths.ensure_private_dir(paths.board_dir())
    paths.ensure_private_dir(dest.parent)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix="index.html.", suffix=".partial")
    try:
        try:
            handle = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        with handle:  # a lone surrogate (a surrogateescape'd path read from the DB) never raises
            handle.write(html.encode("utf-8", "backslashreplace"))
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return dest


# --------------------------------------------------------------------------- the view model (D4, D5)


def _age(now: datetime, text: Any) -> float | None:
    """Seconds since the stored stamp ``text``; ``None`` for ``None`` or a value that is no stamp."""
    if text is None:
        return None
    try:
        return (now - clock_mod.parse(text)).total_seconds()
    except TypeError, ValueError:
        return None


def _dict(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _list(value: Any) -> list[dict[str, Any]]:
    """The dict rows of a snapshot section (anything else is dropped)."""
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _halt(gate: Any, now: datetime, current: str | None = None) -> str | None:
    """``"stuck"``, ``"failed"`` or ``None``: read-only, for rule 4 only (D5; the authority is ``tool merge``).

    ``current`` is the code version on disk. Known, it mirrors ``mergegate.merge_allowed``: only the records
    of the current version count. ``None`` (unknown): every record counts, as of the record's own version.
    """
    gate = _dict(gate) or {}
    rec, ov, pending = (_dict(gate.get(key)) for key in ("selfcheck", "override", "pending"))
    if current is not None:
        if ov is not None and ov.get("code_version") == current:
            return None
        rec = rec if rec is not None and rec.get("code_version") == current else None
        pending = pending if pending is not None and pending.get("code_version") == current else None
    waited = None if pending is None else _age(now, pending.get("since"))
    if waited is not None and (current is None or rec is None):
        # a readable pending mark is newer than the record and the override (both delete it); an
        # unreadable one hides nothing
        return "stuck" if waited > mergegate.PENDING_ESCALATE_S else None
    if (
        rec is None
        or rec.get("ok") is not False
        or mergegate.int_or(rec.get("tries"), 1) < mergegate.SELFCHECK_ESCALATE_AFTER
    ):
        return None
    version = rec.get("code_version")  # equal to `current` when it is known
    masked = ov is not None and version is not None and ov.get("code_version") == version
    return None if masked else "failed"


def _layout(ids: list[str], edges: Iterable[tuple[str, str]]) -> tuple[list[list[str]], list[str]]:
    """Kahn's layering → ``(layers, unresolved)``; ``edges`` are ``(dependency, dependent)``.

    Layer 0 holds the tasks with no dependency; a task's layer is ``1 + max`` of its dependencies'.
    The tasks never placed (in a cycle, or downstream of one) form one extra last layer, also returned
    as ``unresolved``. Inside a layer, ``ids`` order is kept.
    """
    known = set(ids)
    deps: dict[str, set[str]] = {i: set() for i in ids}
    dependents: dict[str, list[str]] = {i: [] for i in ids}
    for dep, dependent in edges:
        if dep in known and dependent in known and dep not in deps[dependent]:
            deps[dependent].add(dep)
            dependents[dep].append(dependent)
    waiting = {i: len(deps[i]) for i in ids}
    layer: dict[str, int] = {}
    queue = [i for i in ids if waiting[i] == 0]
    for i in queue:
        layer[i] = 0
    while queue:
        current = queue.pop(0)
        for nxt in dependents[current]:
            waiting[nxt] -= 1
            if waiting[nxt] == 0:
                layer[nxt] = 1 + max(layer[d] for d in deps[nxt])
                queue.append(nxt)
    depth = 1 + max(layer.values(), default=-1)
    layers = [[i for i in ids if layer.get(i) == n] for n in range(depth)]
    unresolved = [i for i in ids if i not in layer]
    if unresolved:
        layers.append(unresolved)
    return layers, unresolved


def _chat(run: dict[str, Any]) -> str:
    lease = _dict(run.get("lease")) or {}
    return str(lease.get("session_name") or run["name"])


def _progress(run_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    by_state: dict[str, int] = {}
    for t in run_tasks:
        by_state[str(t["state"])] = by_state.get(str(t["state"]), 0) + 1
    done = sum(n for s, n in by_state.items() if s in DONE_STATES)
    dropped = by_state.get("dropped", 0)
    return {
        "by_state": dict(sorted(by_state.items())),
        "done": done,
        "dropped": dropped,
        "remaining": len(run_tasks) - done - dropped,
        "total": len(run_tasks),
    }


def _short_title(title: Any) -> str:
    text = str(title)
    return text if len(text) <= GRAPH_TITLE_MAX else text[: GRAPH_TITLE_MAX - 1] + "…"


def _graph(run_tasks: list[dict[str, Any]], all_deps: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(t["id"]) for t in run_tasks]
    known = set(ids)
    edges: list[tuple[str, str]] = []
    outside = 0
    for d in all_deps:
        task_id, dep = str(d["task_id"]), str(d["depends_on"])
        if task_id in known and dep in known:
            edges.append((dep, task_id))
        elif task_id in known or dep in known:
            outside += 1
    layers, unresolved = _layout(ids, edges)
    by_id = {str(t["id"]): t for t in run_tasks}
    return {
        "layers": [
            [
                {
                    "id": i,
                    "state": str(by_id[i]["state"]),
                    "title": str(by_id[i]["title"]),
                    "short_title": _short_title(by_id[i]["title"]),
                    "unresolved": i in unresolved,
                }
                for i in layer
            ]
            for layer in layers
        ],
        "edges": edges,
        "outside": outside,
    }


def _latest_nodes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each task's node: the row with the highest ``generation``."""
    best: dict[str, dict[str, Any]] = {}
    for n in rows:
        task_id = str(n["task_id"])
        prev = best.get(task_id)
        if prev is None or mergegate.int_or(n["generation"], 0) > mergegate.int_or(prev["generation"], 0):
            best[task_id] = n
    return best


def _task_row(
    t: dict[str, Any],
    node: dict[str, Any] | None,
    crit: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    open_findings: int,
) -> dict[str, Any]:
    task_id = str(t["id"])
    met = sum(1 for c in crit if c["state"] in ("met", "waived"))
    live = node is not None and node["state"] in nodes.LIVE_STATES
    copies: list[list[str]] = [["where_is", task_id]]
    if live and node is not None and node["short_id"] is not None:
        copies.append(["attach", str(node["short_id"])])
    return {
        "id": task_id,
        "issue": t["issue_number"],
        "title": t["title"],
        "kind": t["kind"],
        "lane": t["lane"],
        "state": t["state"],
        "blocked_from": t["blocked_from"],
        "item_of": t["deliverable_id"],
        "node": None
        if node is None
        else {
            "state": node["state"],
            "name": node["name"],
            "short_id": node["short_id"],
            "live": live,
            "taken_over": node["state"] == "taken_over",
        },
        "pr_url": t["pr_url"],
        "pr_number": t["pr_number"],
        "ci_state": t["ci_state"],
        "ci_sha7": None if t["ci_sha"] is None else str(t["ci_sha"])[:7],
        "criteria": [
            {"text": c["text"], "state": c["state"], "validated": c["validated_at"] is not None} for c in crit
        ],
        "criteria_met": met,
        "criteria_total": len(crit),
        "criteria_validated": bool(crit) and all(c["validated_at"] is not None for c in crit),
        "questions": [
            {
                "id": q["id"],
                "state": q["state"],
                "blocking": bool(mergegate.int_or(q["blocking"], 0)),
                "text": q["text"],
            }
            for q in questions
            if q["state"] in ("open", "asked")
        ],
        "open_findings": open_findings,
        "copies": copies,
    }


def _oldest(now: datetime, stamps: list[Any], *, newest: bool = False) -> Any:
    """The oldest (or newest) readable stamp of ``stamps``; the first one when none is readable."""
    aged = [(age, s) for s in stamps if (age := _age(now, s)) is not None]
    if not aged:
        return stamps[0]
    pick = min if newest else max
    return pick(aged, key=lambda pair: pair[0])[1]


def _alert_group(now: datetime, rows: list[dict[str, Any]], names: dict[str, Any]) -> dict[str, Any]:
    """One cross-run group: open if any member is open, the widest time span, each run once."""
    runs: list[Any] = []
    for a in rows:
        name = names[str(a["run_id"])]
        if name not in runs:
            runs.append(name)
    return {
        **_alert_row(rows[0]),
        "first_seen": _oldest(now, [a["first_seen"] for a in rows]),
        "last_seen": _oldest(now, [a["last_seen"] for a in rows], newest=True),
        "cleared": all(a["cleared_at"] is not None for a in rows),
        "runs": runs,
    }


def _alert_row(a: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": a["kind"],
        "subject": a["subject"],
        "first_seen": a["first_seen"],
        "last_seen": a["last_seen"],
        "cleared": a["cleared_at"] is not None,
        "severity": alerts.severity(str(a["kind"])),
    }


def _review_row(r: dict[str, Any], rts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": r["id"],
        "kind": r["kind"],
        "state": r["state"],
        "summary": r["summary"],
        "tasks": [
            {"task_id": t["task_id"], "outcome": t["outcome"], "reason": t["reason"], "recorded_at": t["recorded_at"]}
            for t in rts
        ],
    }


def _open_findings(rows: list[dict[str, Any]]) -> dict[str, int]:
    count: dict[str, int] = {}
    for f in rows:
        cls = f["classification"] if f["classification"] is not None else f["severity"]
        if f["state"] == "open" and cls != "out_of_scope":
            count[str(f["task_id"])] = count.get(str(f["task_id"]), 0) + 1
    return count


def _where_any(open_runs: list[dict[str, Any]]) -> dict[str, Any]:
    chats = [_chat(r) for r in open_runs]
    return {"to": "any_chat", "chats": chats} if chats else {"to": "new_session"}


def _waiting(
    snap: dict[str, Any], open_runs: list[dict[str, Any]], halt: str | None, run_alerts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(r["id"]): r for r in open_runs}
    out: list[dict[str, Any]] = []
    # 1. an orchestrator not listening, or gone
    for a in sorted(run_alerts, key=lambda a: str(a["first_seen"])):
        run = by_id.get(str(a["run_id"]))
        if run is None or a["cleared_at"] is not None:
            continue
        if a["kind"] == alerts.ORCHESTRATOR_NOT_LISTENING:
            where: dict[str, Any] = {"to": "chat", "chat": _chat(run)}
            text = f"The orchestrator of the run {run['name']} is not listening."
        elif a["kind"] == alerts.ORCHESTRATOR_DEAD:
            where, text = {"to": "new_session"}, f"The orchestrator of the run {run['name']} is gone."
        else:
            continue
        out.append({"rule": 1, "run": run["id"], "where": where, "text": text, "copy": ["resume", str(run["id"])]})
    # 2. a question asked of the maintainer: blocking first, then the oldest
    asked = [q for q in _list(snap.get("questions")) if q["state"] == "asked" and str(q["run_id"]) in by_id]
    asked.sort(key=lambda q: (not mergegate.int_or(q["blocking"], 0), str(q["created_at"])))
    for q in asked:
        run = by_id[str(q["run_id"])]
        out.append(
            {
                "rule": 2,
                "run": run["id"],
                "where": {"to": "chat", "chat": _chat(run)},
                "text": f"{q['id']} on {q['task_id']}" + (" (blocking)" if mergegate.int_or(q["blocking"], 0) else ""),
                "detail": q["text"],
                "copy": ["answer", str(q["id"])],
            }
        )
    # 3. criteria to validate, before a node is spawned
    crit: dict[str, list[dict[str, Any]]] = {}
    for c in _list(snap.get("criteria")):
        crit.setdefault(str(c["task_id"]), []).append(c)
    for t in _list(snap.get("tasks")):
        cs = crit.get(str(t["id"]), [])
        if str(t["run_id"]) in by_id and t["state"] in CRITERIA_STATES and any(c["validated_at"] is None for c in cs):
            run = by_id[str(t["run_id"])]
            out.append(
                {
                    "rule": 3,
                    "run": run["id"],
                    "where": {"to": "chat", "chat": _chat(run)},
                    "text": f"The acceptance criteria of {t['id']} wait for your validation.",
                    "detail": t["title"],
                    "copy": ["criteria", str(t["id"])],
                }
            )
    # 4. a merge halt, 5. a backup failure: shown with or without an open run
    if halt is not None:
        out.append(
            {
                "rule": 4,
                "run": None,
                "where": _where_any(open_runs),
                "text": f"Automatic merges are halted: the merge gate is {halt}.",
                "copy": ["halt", halt],
            }
        )
    error = (_dict(snap.get("backup")) or {}).get("last_error")
    if error is not None:
        detail = error.get("error") if isinstance(error, dict) else error
        at = error.get("at") if isinstance(error, dict) else None
        out.append(
            {
                "rule": 5,
                "run": None,
                "where": _where_any(open_runs),
                "text": "The Control Plane backup is failing" + ("." if at is None else f" (last error at {at})."),
                "detail": detail,
                "copy": ["backup"],
            }
        )
    return out


def _system(snap: dict[str, Any], now: datetime, code_schema: int) -> dict[str, Any]:
    lease = _dict(snap.get("daemon"))
    age = None if lease is None else _age(now, lease.get("heartbeat_at"))
    running = lease is not None and lease.get("pid") is not None and age is not None and age < daemon.STALE_AFTER_S
    gate = _dict(snap.get("merge_gate")) or {}
    backup = _dict(snap.get("backup")) or {}
    return {
        "daemon_running": running,
        "pid": None if lease is None else lease.get("pid"),
        "heartbeat_age_s": age,
        "db_schema": snap.get("schema_version"),
        "code_schema": code_schema,
        "merge_gate": {k: gate.get(k) for k in ("selfcheck", "override", "pending")},
        "backup": {"last_at": backup.get("last_at"), "last_error": backup.get("last_error")},
        "hook_alerts": _list(snap.get("hook_alerts"))[: alerts.ALERTS_LIMIT],
    }


def _header(now: datetime, main: Path, code_schema: int) -> dict[str, Any]:
    return {
        "generated_at": clock_mod.iso(now),
        "generated_at_s": int(now.timestamp()),
        "generated_hms": now.strftime("%H:%M:%S"),
        "refresh_s": BOARD_REFRESH_S,
        "stale_s": BOARD_STALE_S,
        "force_reload_s": BOARD_FORCE_RELOAD_S,
        "main": str(main),
        "code_schema": code_schema,
    }


def model(
    snap: dict[str, Any],
    *,
    now: datetime,
    main: Path,
    code_schema: int,
    code_version: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """The board's view model of ``snap`` (a ``snapshot`` dict): pure, and tolerant of any DB value (D4).

    ``code_version`` is the code on disk (``None``: unknown, see ``_halt``); ``note`` says why no daemon
    will refresh the page (``cp.py board`` only).
    """
    out = {**_header(now, main, code_schema), "note": note}
    if snap.get("schema_version") is None:  # `snapshot.empty()`: no DB yet, or a migration in progress
        line = EMPTY_LINE if note is None else EMPTY_LINE_NO_DAEMON
        return {**out, "empty": True, "empty_line": line, "waiting": []}
    runs = _list(snap.get("runs"))
    open_runs = [r for r in runs if r["state"] == "open"]
    open_ids = {str(r["id"]) for r in open_runs}
    all_tasks = _list(snap.get("tasks"))
    all_alerts = _list(snap.get("alerts"))
    deps = _list(snap.get("task_deps"))
    latest = _latest_nodes(_list(snap.get("nodes")))
    findings = _open_findings(_list(snap.get("findings")))
    crit: dict[str, list[dict[str, Any]]] = {}
    for c in _list(snap.get("criteria")):
        crit.setdefault(str(c["task_id"]), []).append(c)
    questions: dict[str, list[dict[str, Any]]] = {}
    for q in _list(snap.get("questions")):
        questions.setdefault(str(q["task_id"]), []).append(q)
    rts = _list(snap.get("review_tasks"))
    reviews = _list(snap.get("reviews"))
    halt = _halt(snap.get("merge_gate"), now, code_version)
    waiting = _waiting(snap, open_runs, halt, all_alerts)
    orch: dict[Any, str] = {}
    for item in waiting:
        if item["rule"] == 1 and orch.get(item["run"]) != "gone":  # gone wins over not listening
            orch[item["run"]] = "not listening" if item["where"]["to"] == "chat" else "gone"
    sections = []
    for run in open_runs:
        rid = str(run["id"])
        run_tasks = [t for t in all_tasks if str(t["run_id"]) == rid]
        lease = _dict(run.get("lease")) or {}
        sections.append(
            {
                "id": rid,
                "name": run["name"],
                "title": run["title"],
                "chat": _chat(run),
                "short_id": lease.get("short_id"),
                "orchestrator": orch.get(run["id"]),
                "progress": _progress(run_tasks),
                "graph": _graph(run_tasks, deps),
                "tasks": [
                    _task_row(
                        t,
                        latest.get(str(t["id"])),
                        crit.get(str(t["id"]), []),
                        questions.get(str(t["id"]), []),
                        findings.get(str(t["id"]), 0),
                    )
                    for t in run_tasks
                ],
                "alerts": [
                    _alert_row(a)
                    for a in sorted(all_alerts, key=lambda a: a["cleared_at"] is not None)
                    if str(a["run_id"]) == rid and a["kind"] not in alerts.CROSS_RUN_KINDS
                ],
                "reviews": [
                    _review_row(r, [t for t in rts if str(t["review_id"]) == str(r["id"]) and str(t["run_id"]) == rid])
                    for r in reviews
                    if any(str(t["review_id"]) == str(r["id"]) and str(t["run_id"]) == rid for t in rts)
                ],
            }
        )
    names = {str(r["id"]): r["name"] for r in runs}
    members: dict[str, list[dict[str, Any]]] = {}
    for a in all_alerts:
        if a["kind"] in alerts.CROSS_RUN_KINDS and str(a["run_id"]) in open_ids:
            members.setdefault(str(a["fingerprint"]), []).append(a)
    groups = {fp: _alert_group(now, rows, names) for fp, rows in members.items()}
    cross_reviews = [
        _review_row(r, [t for t in rts if str(t["review_id"]) == str(r["id"])])
        for r in reviews
        if r["kind"] == "cross_run"
        and any(str(t["review_id"]) == str(r["id"]) and str(t["run_id"]) in open_ids for t in rts)
    ]
    closed = []
    for run in runs:
        if run["state"] != "open":
            p = _progress([t for t in all_tasks if str(t["run_id"]) == str(run["id"])])
            closed.append(
                {
                    "name": run["name"],
                    "title": run["title"],
                    "closed_at": run["closed_at"],
                    "done": p["done"],
                    "dropped": p["dropped"],
                    "remaining": p["remaining"],
                }
            )
    return {
        **out,
        "empty": False,
        "waiting": waiting,
        "runs": sections,
        "cross_run": {"alerts": list(groups.values()), "reviews": cross_reviews},
        "system": _system(snap, now, code_schema),
        "closed_runs": closed,
    }


# --------------------------------------------------------------------------- render_to and the hook (D8)


def _code_version(main: Path) -> str | None:
    """The code version on disk under ``main``; ``None`` when it cannot be read."""
    try:
        return codever.code_version(main)
    except Exception:  # noqa: BLE001 — the board must render; an unknown version keeps the record's own
        return None


def render_to(
    conn: sqlite3.Connection | None, clock: clock_mod.Clock, db_path: Path, main: Path, *, note: str | None = None
) -> Path:
    """Render the board of ``conn`` (``None``: no readable state) and write it to ``board_path(db_path)``."""
    snap = snapshot.empty() if conn is None else snapshot.snapshot(conn, clock)
    view = model(
        snap,
        now=clock.now(),
        main=main,
        code_schema=migrations.current_schema_version(),
        code_version=_code_version(main),
        note=note,
    )
    return write(board_render.scrub(board_render.render(view)), board_path(db_path))


_throttle = ticks.Throttle(BOARD_EVERY_S)
_logged_error: str | None = None


def _reset_for_tests() -> None:
    """Forget the logged error (a new daemon); ``ticks._reset_for_tests`` re-arms the throttle."""
    global _logged_error
    _logged_error = None


def _log(message: str) -> None:
    sys.stderr.write(f"[cp-board] {message}\n")
    sys.stderr.flush()


def _error_key(exc: BaseException) -> str:
    """What makes two failures the same: an ``OSError``'s errno and text, never its (random temp) file name."""
    if isinstance(exc, OSError) and exc.strerror is not None:
        return f"{type(exc).__name__}: [Errno {exc.errno}] {exc.strerror}"
    return f"{type(exc).__name__}: {exc}"


def board_hook(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
    """The ``board`` tick hook: rewrite this DB's board every ``BOARD_EVERY_S`` (registered last)."""
    global _logged_error
    if not _throttle.due(clock):  # first: a raising board is retried only every period
        return
    try:
        render_to(conn, clock, db.path_of(conn), activeloop.seams().main)
    except (OSError, errors.CpError, sqlite3.Error) as exc:
        key = _error_key(exc)
        if key != _logged_error:
            _logged_error = key
            _log(f"board write failed: {type(exc).__name__}: {exc}")
        return
    _logged_error = None
