"""The detectors (QS-406 §6): overlap, stalled nodes, too many rounds, state anomalies, cycles, duplicates.

Each detector is ``detect_<x>(conn, clock, seams) -> (kinds, conditions)``; the ``detectors`` hook
(every ``DETECT_EVERY_S``) runs them all and syncs each through ``alerts.sync``. A detector whose input
is unknown this tick returns no kinds, so nothing is cleared or raised. Detectors alert; they never
decide.

Times are compared **in SQL or as strings** against ``clock_mod.stamp(clock, plus=-threshold)``: ISO
stamps sort, and a non-ISO fixture stamp (``'x'``) is never "older".

Git (overlap, leftover scratch worktrees) runs through ``seams.runner`` with a ``HOOK_SUBPROCESS_S``
timeout, a beat after every call, and never inside a ``db.write``.
"""

from __future__ import annotations

import re
import sqlite3
import sys
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePath
from typing import TYPE_CHECKING, Any

from . import activeloop, alerts, codever, daemon, nodes, tasks, ticks
from . import clock as clock_mod

if TYPE_CHECKING:
    from .activeloop import Seams

DETECTORS = "detectors"
DETECT_EVERY_S = 30.0
OVERLAP_MAX_CALLS = 20
OVERLAP_EVERY_S = 120.0
OVERLAP_MAX_FILES = 50
STALL_S = 3600.0
ORPHAN_GRACE_S = 600.0
GATE_SLOT_HELD_S = 4500.0
LOCK_HELD_DEFAULT_S = 3600.0
LOCK_HELD_PROCESS_S = 1200.0  # main-checkout, main-merge, integration:* (SCRIPT_TIMEOUT_S / gh + LOCK_WAIT_S)
LOCK_HELD_SESSION_INTEGRATION_S = 10800.0  # integrate-finish co-holds the session row

ORPHAN_STATES = ("planning", "contracted", "building")
STARTED_STATES = ("planning", "contracted", "building", "ready_to_merge")
NOT_STARTED = ("proposed", "ready")
DEP_DONE = ("merged", "validated")
SCRATCH = re.compile(r"^QS_(\d+)_(\d+)_integration$")

Result = tuple[frozenset[str], list[alerts.Condition]]
NONE: Result = (frozenset(), [])

_THROTTLE = ticks.Throttle(DETECT_EVERY_S)


def _log(message: str) -> None:
    sys.stderr.write(f"[cp-detectors] {message}\n")
    sys.stderr.flush()


def _in(values: Iterable[str]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


def _open_runs(conn: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(r[0] for r in conn.execute("SELECT id FROM runs WHERE state = 'open' ORDER BY rowid"))


class GitFailed(Exception):
    """A git call failed: the detector's input is unknown this tick."""


class OutOfBudget(Exception):
    """The overlap walk reached ``OVERLAP_MAX_CALLS`` this tick."""


@dataclass
class _Git:
    conn: sqlite3.Connection
    clock: clock_mod.Clock
    seams: Seams
    budget: int | None = None
    calls: int = 0

    def run(self, *args: str, ok: tuple[int, ...] = (0,)) -> tuple[int, str]:
        if self.budget is not None and self.calls >= self.budget:
            raise OutOfBudget
        self.calls += 1
        res = self.seams.runner.run(
            ["git", "-C", str(self.seams.main), *args], cwd=self.seams.main, timeout=ticks.HOOK_SUBPROCESS_S
        )
        daemon.beat(self.conn, self.clock)
        if res.returncode not in ok:
            raise GitFailed(f"git {' '.join(args[:3])} exited {res.returncode}: {res.stderr.strip()[-200:]}")
        return res.returncode, res.stdout


# --------------------------------------------------------------------------- 1. overlap


@dataclass
class _OverlapState:
    diffs: dict[tuple[str, str], list[str]] = field(default_factory=dict)  # (base sha, tip sha) → files
    conflicts: dict[tuple[str, str], list[str]] = field(default_factory=dict)  # (tip, tip) → files
    refs: dict[str, str | None] = field(default_factory=dict)  # resolved for the current walk only
    last: list[alerts.Condition] | None = None  # the last complete result
    last_at: datetime | None = None
    walking: bool = False


_overlap = _OverlapState()
OVERLAP_KINDS = frozenset({alerts.OVERLAP, alerts.OVERLAP_CROSS_RUN})


def _resolve(git: _Git, ref: str) -> str | None:
    if ref not in _overlap.refs:
        code, out = git.run("rev-parse", "--verify", "--quiet", ref, ok=(0, 1))
        _overlap.refs[ref] = out.strip() if code == 0 and out.strip() else None
    return _overlap.refs[ref]


def _is_ancestor(git: _Git, a: str, b: str) -> bool:
    key = f"ancestor:{a}:{b}"
    if key not in _overlap.refs:
        code, _ = git.run("merge-base", "--is-ancestor", a, b, ok=(0, 1))
        _overlap.refs[key] = "yes" if code == 0 else None
    return _overlap.refs[key] is not None


def _main_base(git: _Git) -> str | None:
    local, remote = _resolve(git, "refs/heads/main"), _resolve(git, "refs/remotes/origin/main")
    if local is None or remote is None or local == remote:
        return local or remote
    if _is_ancestor(git, local, remote):
        return remote
    if _is_ancestor(git, remote, local):
        return local
    return remote  # diverged: a stale local main would show main's own commits


def _files(git: _Git, base: str, tip: str) -> list[str]:
    if (base, tip) not in _overlap.diffs:
        _, out = git.run("diff", "--name-only", f"{base}...{tip}")
        _overlap.diffs[(base, tip)] = sorted({line for line in out.splitlines() if line.strip()})
    return _overlap.diffs[(base, tip)]


def _conflicts(git: _Git, a: str, b: str) -> list[str]:
    key = (a, b) if a <= b else (b, a)
    if key not in _overlap.conflicts:
        code, out = git.run("merge-tree", "--write-tree", "--name-only", "--no-messages", key[0], key[1], ok=(0, 1))
        _overlap.conflicts[key] = [] if code == 0 else [ln for ln in out.splitlines()[1:] if ln.strip()]
    return _overlap.conflicts[key]


@dataclass(frozen=True)
class _Branch:
    task_id: str
    run_id: str | None
    deliverable_id: str | None  # set for a work item
    tip: str
    files: tuple[str, ...]


def _walk(conn: sqlite3.Connection, git: _Git) -> list[alerts.Condition]:
    rows = conn.execute(
        f"SELECT t.id, t.run_id, t.branch, t.deliverable_id, d.branch AS base_branch FROM tasks t"
        f" LEFT JOIN tasks d ON d.id = t.deliverable_id"
        f" WHERE t.branch IS NOT NULL AND t.state NOT IN {_in(tasks.TERMINAL)} ORDER BY t.id"
    ).fetchall()
    branches: list[_Branch] = []
    for row in rows:
        tip = _resolve(git, f"refs/heads/{row['branch']}")
        if tip is None:
            continue
        if row["deliverable_id"] is None:
            base = _main_base(git)
        else:
            base = None if row["base_branch"] is None else _resolve(git, f"refs/heads/{row['base_branch']}")
        if base is None:
            continue
        files = tuple(_files(git, base, tip))
        branches.append(_Branch(row["id"], row["run_id"], row["deliverable_id"], tip, files))
    out = []
    for i, a in enumerate(branches):
        for b in branches[i + 1 :]:
            if a.deliverable_id == b.task_id or b.deliverable_id == a.task_id:
                continue  # an item never overlaps its own deliverable
            shared = sorted(set(a.files) & set(b.files))
            if not shared:
                continue
            kind = alerts.OVERLAP if a.run_id == b.run_id else alerts.OVERLAP_CROSS_RUN
            ids = sorted((a.task_id, b.task_id))
            payload = {
                "tasks": ids,
                "files": shared[:OVERLAP_MAX_FILES],
                "truncated": len(shared) > OVERLAP_MAX_FILES,
                "conflicts": _conflicts(git, a.tip, b.tip),
            }
            out.append(alerts.Condition(kind, "|".join(ids), (a.run_id, b.run_id), payload))
    return out


def _cached_overlap() -> Result:
    return NONE if _overlap.last is None else (OVERLAP_KINDS, list(_overlap.last))


def detect_overlap(conn: sqlite3.Connection, clock: clock_mod.Clock, seams: Seams) -> Result:
    now = clock.now()
    due = _overlap.walking or _overlap.last_at is None or (now - _overlap.last_at).total_seconds() >= OVERLAP_EVERY_S
    if not due or codever.git_busy(seams.main) is not None:
        return _cached_overlap()
    git = _Git(conn, clock, seams, budget=OVERLAP_MAX_CALLS)
    try:
        conditions = _walk(conn, git)
    except OutOfBudget:
        _overlap.walking = True  # continue on the next run; computed pairs hit the caches
        return _cached_overlap()
    except GitFailed as exc:
        _overlap.walking = False
        _overlap.refs.clear()
        _log(f"overlap: {exc}")
        return NONE
    _overlap.refs.clear()
    _overlap.walking = False
    _overlap.last, _overlap.last_at = conditions, now
    return OVERLAP_KINDS, conditions


# --------------------------------------------------------------------------- 2. stalled node


def detect_stalled(conn: sqlite3.Connection, clock: clock_mod.Clock, seams: Seams) -> Result:
    cutoff = clock_mod.stamp(clock, plus=-STALL_S)
    rows = conn.execute(
        "SELECT n.id, n.run_id, n.task_id, ("
        "  SELECT max(t) FROM ("
        "    SELECT n.launch_at AS t UNION ALL SELECT n.spawned_at"
        "    UNION ALL SELECT max(at) FROM reports WHERE task_id = n.task_id"
        "    UNION ALL SELECT max(created_at) FROM messages WHERE sender = 'node:' || n.task_id"
        "    UNION ALL SELECT max(started_at) FROM tool_calls WHERE actor = 'node:' || n.task_id"
        "    UNION ALL SELECT max(finished_at) FROM tool_calls WHERE actor = 'node:' || n.task_id)"
        ") AS last FROM nodes n WHERE n.state = 'running' AND NOT EXISTS ("
        "  SELECT 1 FROM tool_calls c WHERE c.state = 'started' AND c.actor = 'node:' || n.task_id)"
        " ORDER BY n.id"
    ).fetchall()
    out = [
        alerts.Condition(
            alerts.NODE_STALLED,
            r["id"],
            (r["run_id"],),
            {"node_id": r["id"], "task_id": r["task_id"], "last_activity": r["last"]},
        )
        for r in rows
        if r["last"] is not None and r["last"] < cutoff
    ]
    return frozenset({alerts.NODE_STALLED}), out


# --------------------------------------------------------------------------- 3. too many rounds


def detect_rounds(conn: sqlite3.Connection, clock: clock_mod.Clock, seams: Seams) -> Result:
    rows = conn.execute(
        f"SELECT r.task_id, t.run_id, r.phase, max(r.round) AS rnd FROM reports r JOIN tasks t ON t.id = r.task_id"
        f" WHERE t.state NOT IN {_in(tasks.TERMINAL)} GROUP BY r.task_id, r.phase HAVING rnd > ? ORDER BY r.task_id, r.phase",
        (alerts.ROUNDS_ALERT,),
    ).fetchall()
    out = [
        alerts.Condition(
            alerts.TOO_MANY_ROUNDS,
            f"{r['task_id']}:{r['phase']}:r{r['rnd']}",
            (r["run_id"],),
            {"task_id": r["task_id"], "phase": r["phase"], "round": r["rnd"], "limit": alerts.ROUNDS_ALERT},
        )
        for r in rows
    ]
    return frozenset({alerts.TOO_MANY_ROUNDS}), out


# --------------------------------------------------------------------------- 4. state anomalies


def _task_without_node(conn: sqlite3.Connection, clock: clock_mod.Clock) -> list[alerts.Condition]:
    cutoff = clock_mod.stamp(clock, plus=-ORPHAN_GRACE_S)
    rows = conn.execute(
        f"SELECT t.id, t.run_id, t.state, max("
        f"  COALESCE((SELECT max(at) FROM task_history WHERE task_id = t.id), t.created_at),"
        f"  COALESCE((SELECT max(launch_at) FROM nodes WHERE task_id = t.id), '')"
        f") AS last FROM tasks t WHERE t.state IN {_in(ORPHAN_STATES)} AND NOT EXISTS ("
        f"  SELECT 1 FROM nodes n WHERE n.task_id = t.id AND n.state IN {_in(nodes.LIVE_STATES)}) ORDER BY t.id"
    ).fetchall()
    return [
        alerts.Condition(alerts.TASK_WITHOUT_NODE, r["id"], (r["run_id"],), {"task_id": r["id"], "state": r["state"]})
        for r in rows
        if r["last"] < cutoff
    ]


def _dependency_violated(conn: sqlite3.Connection) -> list[alerts.Condition]:
    rows = conn.execute(
        f"SELECT t.id, t.run_id, d.id AS dep, d.state AS dep_state FROM task_deps x"
        f" JOIN tasks t ON t.id = x.task_id JOIN tasks d ON d.id = x.depends_on"
        f" WHERE (t.state IN {_in(STARTED_STATES)}"
        f"        OR (t.state = 'blocked' AND COALESCE(t.blocked_from, '') NOT IN {_in(NOT_STARTED)}))"
        f" AND d.state NOT IN {_in(DEP_DONE)} ORDER BY t.id, d.id"
    ).fetchall()
    return [
        alerts.Condition(
            alerts.DEPENDENCY_VIOLATED,
            f"{r['id']}->{r['dep']}",
            (r["run_id"],),
            {"task_id": r["id"], "depends_on": r["dep"], "dependency_state": r["dep_state"]},
        )
        for r in rows
    ]


def lock_threshold(name: str, holder_kind: str) -> float:
    if holder_kind == "process" and (name in ("main-checkout", "main-merge") or name.startswith("integration:")):
        return LOCK_HELD_PROCESS_S
    if holder_kind == "session" and name.startswith("integration:"):
        return LOCK_HELD_SESSION_INTEGRATION_S
    return LOCK_HELD_DEFAULT_S


def _subject_run(conn: sqlite3.Connection, token_subject: str) -> str | None:
    kind, _, ref = token_subject.partition(":")
    if kind == "run":
        return ref or None
    if kind == "node":
        row = conn.execute("SELECT run_id FROM nodes WHERE id = ?", (ref,)).fetchone()
        return None if row is None else row["run_id"]
    return None


def _lock_held_long(conn: sqlite3.Connection, clock: clock_mod.Clock) -> list[alerts.Condition]:
    out = []
    for r in conn.execute("SELECT * FROM locks ORDER BY name").fetchall():
        if r["acquired_at"] >= clock_mod.stamp(clock, plus=-lock_threshold(r["name"], r["holder_kind"])):
            continue
        run = _subject_run(conn, r["token_subject"])
        runs = (run,) if run is not None else _open_runs(conn)
        payload = {
            "name": r["name"],
            "holder_kind": r["holder_kind"],
            "holder_actor": r["holder_actor"],
            "acquired_at": r["acquired_at"],
        }
        out.append(alerts.Condition(alerts.LOCK_HELD_LONG, f"{r['name']}@{r['acquired_at']}", runs, payload))
    return out


def _gate_slot_held_long(conn: sqlite3.Connection, clock: clock_mod.Clock) -> list[alerts.Condition]:
    out = []
    rows = conn.execute(
        "SELECT * FROM cap_slots WHERE cap = 'gates' AND acquired_at < ? ORDER BY slot",
        (clock_mod.stamp(clock, plus=-GATE_SLOT_HELD_S),),
    ).fetchall()
    for r in rows:
        call = conn.execute(
            "SELECT run_id FROM tool_calls WHERE state = 'started' AND holder_pid = ? AND run_id IS NOT NULL LIMIT 1",
            (r["holder_pid"],),
        ).fetchone()
        runs = (call["run_id"],) if call is not None else _open_runs(conn)
        payload = {"slot": r["slot"], "holder_actor": r["holder_actor"], "acquired_at": r["acquired_at"]}
        out.append(alerts.Condition(alerts.GATE_SLOT_HELD_LONG, f"gates#{r['slot']}@{r['acquired_at']}", runs, payload))
    return out


_FINISHED_ITEMS = (
    "SELECT i.id, i.branch, i.worktree, d.run_id FROM tasks i JOIN tasks d ON d.id = i.deliverable_id"
    " WHERE d.state IN ('validated', 'dropped')"
)


def _leftover_item(conn: sqlite3.Connection) -> list[alerts.Condition]:
    rows = conn.execute(f"{_FINISHED_ITEMS} AND i.worktree IS NOT NULL ORDER BY i.id").fetchall()
    return [
        alerts.Condition(alerts.LEFTOVER_ITEM, r["id"], (r["run_id"],), {"task_id": r["id"], "worktree": r["worktree"]})
        for r in rows
    ]


def _leftover_scratch(conn: sqlite3.Connection, clock: clock_mod.Clock, seams: Seams) -> list[alerts.Condition] | None:
    """``None`` when the worktree listing failed (the kind is then left out)."""
    items = {r["branch"]: r for r in conn.execute(f"{_FINISHED_ITEMS} AND i.branch IS NOT NULL").fetchall()}
    if not items:
        return []
    try:
        _, listing = _Git(conn, clock, seams).run("worktree", "list", "--porcelain")
    except GitFailed as exc:
        _log(f"leftover scratch: {exc}")
        return None
    out = []
    for line in listing.splitlines():
        if not line.startswith("worktree "):
            continue
        path = line[len("worktree ") :].strip()
        m = SCRATCH.match(PurePath(path).name)
        item = None if m is None else items.get(f"QS_{m.group(1)}_{m.group(2)}")
        if item is not None:
            payload = {"task_id": item["id"], "path": path}
            out.append(alerts.Condition(alerts.LEFTOVER_SCRATCH, item["id"], (item["run_id"],), payload))
    return out


def detect_anomalies(conn: sqlite3.Connection, clock: clock_mod.Clock, seams: Seams) -> Result:
    conditions = [
        *_task_without_node(conn, clock),
        *_dependency_violated(conn),
        *_lock_held_long(conn, clock),
        *_gate_slot_held_long(conn, clock),
        *_leftover_item(conn),
    ]
    kinds = {
        alerts.TASK_WITHOUT_NODE,
        alerts.DEPENDENCY_VIOLATED,
        alerts.LOCK_HELD_LONG,
        alerts.GATE_SLOT_HELD_LONG,
        alerts.LEFTOVER_ITEM,
    }
    scratch = _leftover_scratch(conn, clock, seams)
    if scratch is not None:
        kinds.add(alerts.LEFTOVER_SCRATCH)
        conditions += scratch
    return frozenset(kinds), conditions


# --------------------------------------------------------------------------- 5. dependency cycles


def strongly_connected(nodes: Iterable[str], edges: dict[str, list[str]]) -> list[list[str]]:
    """Tarjan's algorithm (iterative) → the strongly connected components."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    out: list[list[str]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            v, i = work.pop()
            if i == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on_stack.add(v)
            succ = edges.get(v, [])
            if i < len(succ):
                work.append((v, i + 1))
                w = succ[i]
                if w not in index:
                    work.append((w, 0))
                elif w in on_stack:
                    low[v] = min(low[v], index[w])
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                out.append(comp)
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
    return out


def _grouped(
    kind: str, cross: str, ids: list[str], run_of: dict[str, str | None], payload: dict[str, Any]
) -> alerts.Condition:
    ids = sorted(ids)
    runs = tuple(dict.fromkeys(run_of[i] for i in ids))
    return alerts.Condition(cross if len(runs) > 1 else kind, "|".join(ids), runs, {"tasks": ids, **payload})


def detect_cycles(conn: sqlite3.Connection, clock: clock_mod.Clock, seams: Seams) -> Result:
    edges: dict[str, list[str]] = {}
    for r in conn.execute("SELECT task_id, depends_on FROM task_deps ORDER BY task_id, depends_on").fetchall():
        edges.setdefault(r["task_id"], []).append(r["depends_on"])
    run_of = {r["id"]: r["run_id"] for r in conn.execute("SELECT id, run_id FROM tasks").fetchall()}
    out = [
        _grouped(alerts.DEPENDENCY_CYCLE, alerts.DEPENDENCY_CYCLE_CROSS_RUN, comp, run_of, {})
        for comp in strongly_connected(sorted(edges), edges)
        if len(comp) > 1
    ]
    return frozenset({alerts.DEPENDENCY_CYCLE, alerts.DEPENDENCY_CYCLE_CROSS_RUN}), out


# --------------------------------------------------------------------------- 6. duplicates


def normalise_title(title: str) -> str:
    t = unicodedata.normalize("NFKC", title).casefold()
    t = re.sub(r"^\s*qs[-_ ]?\d+\s*[:\-–—]?\s*", "", t)
    return re.sub(r"[\W_]+", " ", t).strip()


def detect_duplicates(conn: sqlite3.Connection, clock: clock_mod.Clock, seams: Seams) -> Result:
    groups: dict[str, list[str]] = {}
    run_of: dict[str, str | None] = {}
    for r in conn.execute(f"SELECT id, run_id, title FROM tasks WHERE state NOT IN {_in(tasks.TERMINAL)} ORDER BY id"):
        key = normalise_title(r["title"])
        if key:
            groups.setdefault(key, []).append(r["id"])
            run_of[r["id"]] = r["run_id"]
    out = [
        _grouped(alerts.DUPLICATE_TASK, alerts.DUPLICATE_TASK_CROSS_RUN, ids, run_of, {"title": key})
        for key, ids in sorted(groups.items())
        if len(ids) > 1
    ]
    return frozenset({alerts.DUPLICATE_TASK, alerts.DUPLICATE_TASK_CROSS_RUN}), out


# --------------------------------------------------------------------------- the hook

ALL: tuple[Callable[[sqlite3.Connection, clock_mod.Clock, Seams], Result], ...] = (
    detect_overlap,
    detect_stalled,
    detect_rounds,
    detect_anomalies,
    detect_cycles,
    detect_duplicates,
)


def detectors_hook(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
    if not _THROTTLE.due(clock):
        return
    seams = activeloop.seams()
    for detect in ALL:
        kinds, conditions = detect(conn, clock, seams)
        alerts.sync(conn, clock, kinds=kinds, active=conditions)


def _reset_for_tests() -> None:
    global _overlap
    _overlap = _OverlapState()
