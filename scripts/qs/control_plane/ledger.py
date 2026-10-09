"""The finding ledger (#375): rounds, findings, their events, and the blast radius.

The ledger **recognises, records and flags by code**; the node judges. It
refuses only malformed input, a token used out of its scope, and a red-CI
finding that does not match the task's red sha. Nothing is refused to force
convergence: a flip-flop is flagged (``matches``, ``overlaps_fix``,
``relates_to``), the node settles it with a rationale, and an exact re-raise
on the same task is born in that settled (or rejected) state.

Every write runs in one ``db.write``; every read runs in the caller's
transaction. See ``docs/workflow/control-plane.md`` "The ledger (#375)".
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import sqlite3
import unicodedata
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Any

from . import clock as clock_mod
from . import db, errors, export, tasks, tokens
from .schema_ledger import (
    BLAST_VALUES,
    CATEGORIES,
    CLASSES,
    PHASES,
    REVIEWER_SOURCES,
    SEVERITIES,
    SOURCES,
    STATES,
)

SEP = "\x1f"  # U+001F, the fingerprint / replay-key field separator; NULL is encoded as ""
SEQ_KIND = "finding_decision"  # the counters kind behind ``decided_seq``
CLOSED_BY_MATCH = frozenset({"rejected", "settled"})  # a same-task strong match in these is inherited (D9)
FIXED = frozenset({"resolved", "settled"})  # what an ``overlaps_fix`` flag looks at
BLOCKING = ("must_fix", "should_fix")
DEFERRING = frozenset({"nice_to_have", "out_of_scope"})
MECHANICAL = frozenset({"ci", "gate"})  # stored must_fix / must_fix; the round is out of their replay key
NEED_REASON = frozenset({"rejected", "settled"})
EMPTY = "_None recorded._"
SECTION_TITLES = ("Rounds", "Findings", "Blast radius")

_ITEM_KEYS = frozenset(
    {
        "severity",
        "category",
        "title",
        "body",
        "file",
        "symbol",
        "line_start",
        "line_end",
        "reviewer",
        "unchanged_lines",
        "relates_to",
        "sha",
        "integration_id",
    }
)
_NON_ALNUM = re.compile(r"[\W_]+")
_LINE_SUFFIX = re.compile(r":\d+(-\d+)?$")
MAX_INT = 2**63 - 1  # SQLite's INTEGER range: a larger id or line is USAGE, never an INTERNAL overflow


# --------------------------------------------------------------------------- writers and normalisation


@dataclass(frozen=True)
class Writer:
    """Who writes on a task: the principal, its actor, and the ``run_id`` the rows record."""

    principal: tokens.Principal
    actor: str
    run_id: str | None


def writer(
    conn: sqlite3.Connection, token: str, task_id: str, *, kinds: Collection[str] = frozenset({"run", "node"})
) -> Writer:
    """Inside ``db.write``. A node token writes on its own task, a run token on any task of its run (D13).

    The one point child 15 extends (a reviewer-scoped token, cross-run writes).
    """
    who = tokens.require(conn, token, kinds=kinds, task_id=task_id)
    task = tasks.get(conn, task_id)
    return Writer(who, who.actor, task["run_id"] if task["run_id"] is not None else who.run_id)


def norm(text: str) -> str:
    """NFKC, case-folded; every run of non-alphanumerics becomes one space; trimmed."""
    return _NON_ALNUM.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def norm_path(file: str | None) -> str | None:
    """A normalised repo-relative path (``a//b/./c`` → ``a/b/c``); empty is ``None``.

    ``USAGE``: absolute, a ``..`` segment, a backslash, or a ``:<line>`` suffix (lines go in
    ``line_start`` / ``line_end``, or every round's line would change the fingerprint).
    """
    path = _text(file)
    if path is None:
        return None
    if path.startswith("/") or "\\" in path or ".." in path.split("/") or _LINE_SUFFIX.search(path):
        raise errors.CpError("USAGE", f"`file` must be a relative path inside the repo, with no line: {file!r}")
    path = posixpath.normpath(path)
    return None if path == "." else path


def _text(value: str | None) -> str | None:
    """Stripped; a blank string is ``None``."""
    return (value.strip() or None) if value is not None else None


def _sha(parts: Iterable[Any]) -> str:
    return hashlib.sha256(SEP.join("" if p is None else str(p) for p in parts).encode()).hexdigest()


def fingerprint(*, file: str | None, symbol: str | None, category: str, title: str) -> str:
    """``sha256(file␟symbol␟category)``, plus ``␟norm(title)`` when both ``file`` and ``symbol`` are empty (D8)."""
    parts: list[Any] = [file, symbol, category]
    if not file and not symbol:
        parts.append(norm(title))
    return _sha(parts)


def family(conn: sqlite3.Connection, task_id: str) -> list[str]:
    """The task's family: its deliverable (or itself) and every work item of that deliverable (D8)."""
    root = tasks.get(conn, task_id)["deliverable_id"] or task_id
    items = [r["id"] for r in conn.execute("SELECT id FROM tasks WHERE deliverable_id = ? ORDER BY rowid", (root,))]
    return [root, *(i for i in items if i != root)]


def _check(value: str, allowed: tuple[str, ...], what: str) -> None:
    if value not in allowed:
        raise errors.CpError("USAGE", f"{what} must be one of {', '.join(allowed)}: {value!r}")


def latest_round(conn: sqlite3.Connection, task_id: str, phase: str) -> int:
    row = conn.execute("SELECT max(round) FROM rounds WHERE task_id = ? AND phase = ?", (task_id, phase)).fetchone()
    return int(row[0] or 0)


# --------------------------------------------------------------------------- rounds


def start_round(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    token: str,
    task_id: str,
    phase: str,
    head: str | None,
    base: str | None,
) -> dict[str, Any]:
    """One review pass → ``{round, base_sha, head_sha}``; the base defaults to the previous round's head (D3, D4)."""
    _check(phase, PHASES, "--phase")
    head, base = _text(head), _text(base)
    if phase == "build" and head is None:
        raise errors.CpError("USAGE", "--head is required for a build round")
    with db.write(conn):
        w = writer(conn, token, task_id)
        prev = conn.execute(
            "SELECT round, head_sha FROM rounds WHERE task_id = ? AND phase = ? ORDER BY round DESC LIMIT 1",
            (task_id, phase),
        ).fetchone()
        number = 1 if prev is None else int(prev["round"]) + 1
        base_sha = base if base is not None or prev is None else prev["head_sha"]
        conn.execute(
            "INSERT INTO rounds (task_id, phase, round, base_sha, head_sha, actor, started_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_id, phase, number, base_sha, head, w.actor, db.now(clock)),
        )
    return {"task_id": task_id, "phase": phase, "round": number, "base_sha": base_sha, "head_sha": head}


# --------------------------------------------------------------------------- opening findings


def parse_items(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise errors.CpError("USAGE", f"--input is not JSON: {exc}") from exc


def _opt(item: dict[str, Any], key: str, kind: type | tuple[type, ...], i: int) -> Any:
    value = item.get(key)
    if value is None:
        return None
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
        raise errors.CpError("USAGE", f"item {i}: `{key}` has the wrong type")
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > MAX_INT:
        raise errors.CpError("USAGE", f"item {i}: `{key}` is out of range")
    return value


def _str(item: dict[str, Any], key: str, i: int) -> str | None:
    """An optional string field, stripped (blank is ``None``); the key separator U+001F is ``USAGE``."""
    value = _opt(item, key, str, i)
    if value is not None and SEP in value:
        raise errors.CpError("USAGE", f"item {i}: `{key}` contains the control character U+001F")
    return _text(value)


def _line(item: dict[str, Any], key: str, i: int) -> int | None:
    value = _opt(item, key, int, i)
    if value is not None and value < 1:
        raise errors.CpError("USAGE", f"item {i}: `{key}` must be >= 1")
    return value


def _validate_item(raw: Any, source: str, i: int) -> dict[str, Any]:
    if not isinstance(raw, dict) or not raw:
        raise errors.CpError("USAGE", f"item {i} must be a non-empty JSON object")
    unknown = sorted(set(raw) - _ITEM_KEYS)
    if unknown:
        raise errors.CpError("USAGE", f"item {i}: unknown key(s) {', '.join(unknown)}")
    severity = _opt(raw, "severity", str, i)
    if severity is None and source not in MECHANICAL:
        raise errors.CpError("USAGE", f"item {i}: `severity` is required")
    if severity is not None:
        _check(severity, SEVERITIES, f"item {i}: `severity`")
    category = _opt(raw, "category", str, i)
    if category is None:
        raise errors.CpError("USAGE", f"item {i}: `category` is required")
    _check(category, CATEGORIES, f"item {i}: `category`")
    title = _opt(raw, "title", str, i)
    if title is None or not norm(title):
        raise errors.CpError("USAGE", f"item {i}: `title` must hold at least one letter or digit")
    _str(raw, "title", i)  # the separator check
    body = _opt(raw, "body", str, i)
    if body is None:
        raise errors.CpError("USAGE", f"item {i}: `body` is required")
    line_start, line_end = _line(raw, "line_start", i), _line(raw, "line_end", i)
    if line_end is not None and line_start is None:
        raise errors.CpError("USAGE", f"item {i}: `line_end` without `line_start`")
    if line_start is not None and line_end is None:
        line_end = line_start
    if line_start is not None and line_end is not None and line_start > line_end:
        raise errors.CpError("USAGE", f"item {i}: `line_start` > `line_end`")
    relates = _opt(raw, "relates_to", list, i) or []
    if any(not isinstance(r, int) or isinstance(r, bool) or not 1 <= r <= MAX_INT for r in relates):
        raise errors.CpError("USAGE", f"item {i}: `relates_to` must be a list of finding ids")
    sha = _str(raw, "sha", i)
    if (raw.get("sha") is not None) != (source == "ci") or (source == "ci" and sha is None):
        raise errors.CpError("USAGE", f"item {i}: `sha` is required for a ci finding, and only there")
    integration = _opt(raw, "integration_id", int, i)
    if (integration is not None) != (source == "gate"):
        raise errors.CpError("USAGE", f"item {i}: `integration_id` is required for a gate finding, and only there")
    unchanged = _opt(raw, "unchanged_lines", bool, i)
    try:
        file = norm_path(_str(raw, "file", i))
    except errors.CpError as exc:
        raise errors.CpError("USAGE", f"item {i}: {exc.detail}") from exc
    return {
        "severity": "must_fix" if source in MECHANICAL else severity,
        "category": category,
        "title": title,
        "body": body,
        "file": file,
        "symbol": _str(raw, "symbol", i),
        "line_start": line_start,
        "line_end": line_end,
        "reviewer": _str(raw, "reviewer", i),
        "unchanged_lines": None if unchanged is None else int(unchanged),
        "relates_to": list(dict.fromkeys(relates)),
        "sha": sha,
        "integration_id": integration,
    }


def validate_items(raw: Any, source: str) -> list[dict[str, Any]]:
    """The item schema of §2: one JSON object or a non-empty list of them; anything else is ``USAGE``."""
    _check(source, SOURCES, "--source")
    batch = [raw] if isinstance(raw, dict) else raw
    if not isinstance(batch, list) or not batch:
        raise errors.CpError("USAGE", "--input must hold a JSON object or a non-empty list of objects")
    return [_validate_item(r, source, i) for i, r in enumerate(batch, 1)]


def _effective(row: sqlite3.Row) -> str:
    return str(row["classification"] or row["severity"])


def _flag(
    kind: str, row: sqlite3.Row, task_id: str, *, strong: bool = False, escalated: bool = False
) -> dict[str, Any]:
    return {
        "kind": kind,
        "id": row["id"],
        "state": row["state"],
        "reason": row["reason"],
        "commit": row["resolved_sha"],
        "source": row["source"],
        "strong": strong,
        "cross_task": row["task_id"] != task_id,
        "escalated": escalated,
    }


def _inherited_reason(row: sqlite3.Row) -> str:
    """``matches #<id>: <reason>``; a reason that already names its root is reused verbatim (never nests)."""
    reason = str(row["reason"])  # a rejected or settled row always has one (D6)
    return reason if reason.startswith("matches #") else f"matches #{row['id']}: {reason}"


def _latest(rows: list[sqlite3.Row]) -> sqlite3.Row | None:
    return max(rows, key=lambda r: r["decided_seq"], default=None)


def _born(
    conn: sqlite3.Connection,
    w: Writer,
    task_id: str,
    source: str,
    item: dict[str, Any],
    fp: str,
    title_norm: str,
) -> tuple[list[dict[str, Any]], int | None, str, str | None]:
    """The flags, ``matched_id``, born state and born reason of a new row (D8, D9)."""
    fam = family(conn, task_id)
    marks = ", ".join("?" for _ in fam)
    matches = conn.execute(
        f"SELECT * FROM findings WHERE fingerprint = ? AND task_id IN ({marks}) ORDER BY id", (fp, *fam)
    ).fetchall()
    must = item["severity"] == "must_fix"
    flags = []
    for r in matches:
        strong = r["title_norm"] == title_norm
        flags.append(
            _flag("matches", r, task_id, strong=strong, escalated=strong and must and _effective(r) != "must_fix")
        )
    if item["symbol"] is not None:
        overlaps = conn.execute(
            "SELECT * FROM findings WHERE file IS ? AND symbol = ? AND fingerprint != ?"
            f" AND state IN ({', '.join('?' for _ in FIXED)}) AND task_id IN ({marks}) ORDER BY id",
            (item["file"], item["symbol"], fp, *sorted(FIXED), *fam),
        ).fetchall()
        flags += [_flag("overlaps_fix", r, task_id) for r in overlaps]
    for rid in item["relates_to"]:
        related = conn.execute("SELECT * FROM findings WHERE id = ? AND run_id IS ?", (rid, w.run_id)).fetchone()
        if related is None:  # another run's finding is not the writer's to read (D13)
            raise errors.CpError("NOT_FOUND", f"`relates_to`: no finding {rid} in run {w.run_id}")
        flags.append(_flag("relates_to", related, task_id))

    strong_own = _latest([r for r in matches if r["title_norm"] == title_norm and r["task_id"] == task_id])
    strong_fam = _latest([r for r in matches if r["title_norm"] == title_norm])
    weak = _latest([r for r in matches if r["title_norm"] != title_norm])
    hit = strong_own or strong_fam or weak
    if source in REVIEWER_SOURCES and strong_own is not None and strong_own["state"] in CLOSED_BY_MATCH:
        return flags, strong_own["id"], strong_own["state"], _inherited_reason(strong_own)
    return flags, None if hit is None else hit["id"], "open", None


def _check_ci(task: sqlite3.Row, sha: str) -> None:
    if task["ci_state"] != "red" or task["ci_sha"] != sha:
        raise errors.CpError(
            "INVALID_STATE",
            f"a ci finding needs the task's CI red on {sha} (ci_state {task['ci_state']}, ci_sha {task['ci_sha']})",
        )


def _check_gate(conn: sqlite3.Connection, task_id: str, integration_id: int) -> None:
    row = conn.execute("SELECT * FROM integrations WHERE id = ?", (integration_id,)).fetchone()
    if row is None:
        raise errors.CpError("NOT_FOUND", f"unknown integration {integration_id}")
    if task_id not in (row["item_task_id"], row["deliverable_id"]):
        raise errors.CpError("CONFLICT", f"integration {integration_id} is not on task {task_id}")


def open_findings(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    token: str,
    task_id: str,
    phase: str,
    source: str,
    round_: int | None,
    items: Any,
) -> dict[str, Any]:
    """Open one item or a batch in one transaction → ``{"ids": [...]}`` (a replay repeats its id; D10, D16)."""
    _check(phase, PHASES, "--phase")
    batch = validate_items(items, source)
    if round_ is not None and round_ < 1:
        raise errors.CpError("USAGE", "--round must be >= 1 (round 0 is only the default before any round)")
    ids: list[int] = []
    with db.write(conn):
        w = writer(conn, token, task_id)
        if round_ is None:
            number = latest_round(conn, task_id, phase)
        elif conn.execute(
            "SELECT 1 FROM rounds WHERE task_id = ? AND phase = ? AND round = ?", (task_id, phase, round_)
        ).fetchone():
            number = round_
        else:
            raise errors.CpError("NOT_FOUND", f"task {task_id} has no {phase} round {round_}")
        task = tasks.get(conn, task_id)
        now = db.now(clock)
        for item in batch:
            fp = fingerprint(file=item["file"], symbol=item["symbol"], category=item["category"], title=item["title"])
            title_norm = norm(item["title"])
            key = _sha(
                (
                    task_id,
                    phase,
                    None if source in MECHANICAL else number,
                    source,
                    item["reviewer"],
                    fp,
                    title_norm,
                    item["line_start"],
                    item["line_end"],
                    item["sha"],
                    item["integration_id"],
                )
            )
            replay = conn.execute("SELECT id FROM findings WHERE replay_key = ?", (key,)).fetchone()
            if replay is not None:
                ids.append(int(replay["id"]))
                continue
            if source == "ci":
                _check_ci(task, item["sha"])
            if source == "gate":
                _check_gate(conn, task_id, item["integration_id"])
            flags, matched_id, state, reason = _born(conn, w, task_id, source, item, fp, title_norm)
            conn.execute(
                "INSERT INTO findings (run_id, task_id, phase, round, source, reviewer, severity, classification,"
                " category, title, body, file, symbol, line_start, line_end, fingerprint, title_norm, replay_key,"
                " matched_id, flags, unchanged_lines, state, decided_seq, reason, ci_sha, integration_id, actor,"
                " created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(replay_key) DO NOTHING",
                (
                    w.run_id,
                    task_id,
                    phase,
                    number,
                    source,
                    item["reviewer"],
                    item["severity"],
                    "must_fix" if source in MECHANICAL else None,
                    item["category"],
                    item["title"],
                    item["body"],
                    item["file"],
                    item["symbol"],
                    item["line_start"],
                    item["line_end"],
                    fp,
                    title_norm,
                    key,
                    matched_id,
                    json.dumps(flags, sort_keys=True),
                    item["unchanged_lines"],
                    state,
                    db.next_seq(conn, SEQ_KIND),
                    reason,
                    item["sha"],
                    item["integration_id"],
                    w.actor,
                    now,
                    now,
                ),
            )
            ids.append(int(conn.execute("SELECT id FROM findings WHERE replay_key = ?", (key,)).fetchone()["id"]))
    return {"ids": ids}


# --------------------------------------------------------------------------- triage


def _finding(conn: sqlite3.Connection, finding_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM findings WHERE id = ?", (finding_id,)).fetchone()
    if row is None:
        raise errors.CpError("NOT_FOUND", f"unknown finding {finding_id}")
    return row


def _scoped(
    conn: sqlite3.Connection, token: str, finding_id: int, kinds: Collection[str] = frozenset({"run", "node"})
) -> tuple[sqlite3.Row, Writer]:
    """Token first (a stale token is always ``STALE_TOKEN``), then the row, then the task scope (D13)."""
    tokens.require(conn, token, kinds=kinds)
    row = _finding(conn, finding_id)
    return row, writer(conn, token, row["task_id"], kinds=kinds)


def _event(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    at: str,
    actor: str,
    kind: str,
    from_value: str | None,
    to_value: str | None,
    reason: str | None,
    commit: str | None = None,
    cause: int | None = None,
) -> None:
    conn.execute(
        "INSERT INTO finding_events (finding_id, at, actor, round, kind, from_value, to_value, reason, commit_sha,"
        " cause_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            row["id"],
            at,
            actor,
            latest_round(conn, row["task_id"], row["phase"]),
            kind,
            from_value,
            to_value,
            reason,
            commit,
            cause,
        ),
    )


def _move(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    at: str,
    actor: str,
    to: str,
    reason: str | None,
    commit: str | None = None,
    cause: int | None = None,
) -> None:
    """A state change: the row keeps only the latest state; the history is a ``state`` event (D6)."""
    conn.execute(
        "UPDATE findings SET state = ?, reason = ?, resolved_sha = ?, decided_seq = ?, updated_at = ? WHERE id = ?",
        (to, reason, commit if to == "resolved" else None, db.next_seq(conn, SEQ_KIND), at, row["id"]),
    )
    _event(
        conn,
        row,
        at=at,
        actor=actor,
        kind="state",
        from_value=row["state"],
        to_value=to,
        reason=reason,
        commit=commit,
        cause=cause,
    )


def classify(
    conn: sqlite3.Connection, clock: clock_mod.Clock, *, token: str, finding_id: int, cls: str, note: str | None
) -> dict[str, Any]:
    """Set the class; it drives open ⇄ deferred (D7). The same class is a noop."""
    _check(cls, CLASSES, "--class")
    note = _text(note)
    with db.write(conn):
        row, w = _scoped(conn, token, finding_id)
        if row["classification"] == cls:
            return {"finding_id": finding_id, "state": row["state"], "classification": cls, "changed": False}
        at = db.now(clock)
        conn.execute("UPDATE findings SET classification = ?, updated_at = ? WHERE id = ?", (cls, at, finding_id))
        _event(
            conn,
            row,
            at=at,
            actor=w.actor,
            kind="classify",
            from_value=row["classification"],
            to_value=cls,
            reason=note,
        )
        to = None
        if row["state"] == "open" and cls in DEFERRING:
            to = "deferred"
        elif row["state"] == "deferred" and cls in BLOCKING:
            to = "open"
        if to is not None:
            _move(conn, row, at=at, actor=w.actor, to=to, reason=f"classified {cls}")
    return {"finding_id": finding_id, "state": to or row["state"], "classification": cls, "changed": True}


def set_state(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    token: str,
    finding_id: int,
    to: str,
    reason: str | None,
    commit: str | None,
    cause: int | None,
) -> dict[str, Any]:
    """Free transitions; only the arguments are checked per target (D6). The same state is a noop."""
    _check(to, STATES, "--to")
    reason, commit = _text(reason), _text(commit)
    if cause is not None and cause == finding_id:
        raise errors.CpError("USAGE", "--cause must be another finding")
    if to == "resolved" and not commit:
        raise errors.CpError("USAGE", "--to resolved needs --commit")
    if to != "resolved" and commit is not None:
        raise errors.CpError("USAGE", f"--commit is only for --to resolved, not {to}")
    if to in NEED_REASON and reason is None:
        raise errors.CpError("USAGE", f"--to {to} needs --reason")
    with db.write(conn):
        row, w = _scoped(conn, token, finding_id)
        if cause is not None:
            _finding(conn, cause)
        if row["state"] == to:
            return {"finding_id": finding_id, "state": to, "changed": False}
        _move(conn, row, at=db.now(clock), actor=w.actor, to=to, reason=reason, commit=commit, cause=cause)
    return {"finding_id": finding_id, "state": to, "changed": True}


# --------------------------------------------------------------------------- blast radius


def set_blast_radius(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    *,
    token: str,
    task_id: str,
    value: str,
    head_sha: str,
    review: str,
    reason: str | None,
) -> dict[str, Any]:
    """Append one rating; a run token only — a node never rates its own blast radius (D13, D15)."""
    _check(value, BLAST_VALUES, "--value")
    if not head_sha.strip() or not review.strip():
        raise errors.CpError("USAGE", "--head-sha and --review must not be empty")
    with db.write(conn):
        w = writer(conn, token, task_id, kinds={"run"})
        cur = conn.execute(
            "INSERT INTO blast_radius (task_id, value, head_sha, review, reason, actor, at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_id, value, head_sha, review, reason, w.actor, db.now(clock)),
        )
    return {"id": cur.lastrowid, "task_id": task_id, "value": value, "head_sha": head_sha}


def blast_radius(conn: sqlite3.Connection, task_id: str, head_sha: str | None = None) -> dict[str, Any] | None:
    """Child 7's reader: the latest rating, or the latest for ``head_sha``; ``None`` when there is none."""
    sql = "SELECT * FROM blast_radius WHERE task_id = ?" + ("" if head_sha is None else " AND head_sha = ?")
    params = (task_id,) if head_sha is None else (task_id, head_sha)
    return db.as_dict(conn.execute(sql + " ORDER BY id DESC LIMIT 1", params).fetchone())


# --------------------------------------------------------------------------- reads


def convergence(conn: sqlite3.Connection, task_id: str, phase: str) -> dict[str, Any]:
    """The node declares convergence; the ledger shows the evidence (D14). Read in the caller's transaction."""
    _check(phase, PHASES, "phase")
    tasks.get(conn, task_id)
    number = latest_round(conn, task_id, phase)
    report = conn.execute(
        "SELECT status FROM reports WHERE task_id = ? AND phase = ? AND round = ? ORDER BY id DESC LIMIT 1",
        (task_id, phase, number),
    ).fetchone()
    blocking = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM findings WHERE task_id = ? AND phase = ? AND state = 'open'"
            " AND coalesce(classification, severity) IN (?, ?) ORDER BY id",
            (task_id, phase, *BLOCKING),
        )
    ]
    counts = dict.fromkeys(STATES, 0)
    for r in conn.execute(
        "SELECT state, count(*) AS n FROM findings WHERE task_id = ? AND phase = ? GROUP BY state", (task_id, phase)
    ):
        counts[r["state"]] = r["n"]
    return {
        "converged": number >= 1 and report is not None and report["status"] == "converged",
        "consistent": blocking == [],
        "blocking": blocking,
        "latest_round": number,
        "counts": counts,
    }


def _rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [{k: r[k] for k in r.keys()} for r in conn.execute(sql, params)]  # noqa: SIM118 — sqlite3.Row


def show(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    phase: str | None = None,
    states: Collection[str] | None = None,
    include_family: bool = False,
) -> dict[str, Any]:
    """The task's ledger. ``include_family`` widens only ``findings`` and their events (§5)."""
    if phase is not None:
        _check(phase, PHASES, "--phase")
    if states is not None and not states:
        raise errors.CpError("USAGE", "--state names no state")
    for state in states or ():
        _check(state, STATES, "--state")
    tasks.get(conn, task_id)
    scope = family(conn, task_id) if include_family else [task_id]
    where = f"task_id IN ({', '.join('?' for _ in scope)})"
    params: tuple[Any, ...] = tuple(scope)
    if phase is not None:
        where += " AND phase = ?"
        params += (phase,)
    if states:
        where += f" AND state IN ({', '.join('?' for _ in states)})"
        params += tuple(states)
    findings = _rows(conn, f"SELECT * FROM findings WHERE {where} ORDER BY id", params)
    for f in findings:
        f["flags"] = json.loads(f["flags"])
    flagged = sorted({flag["id"] for f in findings for flag in f["flags"]})
    current = dict(
        conn.execute(
            f"SELECT id, state FROM findings WHERE id IN ({', '.join('?' for _ in flagged)})", flagged
        ).fetchall()
    )
    for f in findings:
        for flag in f["flags"]:
            flag["current_state"] = current[flag["id"]]
    events = _rows(
        conn,
        f"SELECT * FROM finding_events WHERE finding_id IN (SELECT id FROM findings WHERE {where}) ORDER BY id",
        params,
    )
    round_where = "task_id = ?" + ("" if phase is None else " AND phase = ?")
    round_params = (task_id,) if phase is None else (task_id, phase)
    return {
        "task_id": task_id,
        "rounds": _rows(conn, f"SELECT * FROM rounds WHERE {round_where} ORDER BY phase, round", round_params),
        "findings": findings,
        "finding_events": events,
        "blast_radius": _rows(conn, "SELECT * FROM blast_radius WHERE task_id = ? ORDER BY id", (task_id,)),
        "convergence": {p: convergence(conn, task_id, p) for p in PHASES},
    }


# --------------------------------------------------------------------------- export


def _render_rounds(conn: sqlite3.Connection, task_id: str) -> str:
    rows = conn.execute(
        "SELECT * FROM rounds WHERE task_id = ? ORDER BY CASE phase WHEN 'plan' THEN 0 ELSE 1 END, round", (task_id,)
    ).fetchall()
    if not rows:
        return EMPTY
    lines = ["| phase | round | diff |", "|---|---|---|"]
    for r in rows:
        diff = export._cell(f"{r['base_sha'] or '?'}..{r['head_sha'] or '?'}")
        lines.append(f"| {r['phase']} | {r['round']} | `{diff}` |")
    return "\n".join(lines)


def _render_findings(conn: sqlite3.Connection, task_id: str) -> str:
    rows = conn.execute("SELECT * FROM findings WHERE task_id = ? ORDER BY id", (task_id,)).fetchall()
    if not rows:
        return EMPTY
    lines = [
        "| # | phase / round | source | class | state | title | resolution | flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        resolution = r["resolved_sha"] if r["state"] == "resolved" else (r["reason"] or "")
        flags = ", ".join(f"{f['kind']} #{f['id']}" for f in json.loads(r["flags"]))
        lines.append(
            f"| {r['id']} | {r['phase']} / {r['round']} | {r['source']} | {_effective(r)} | {r['state']}"
            f" | {export._cell(r['title'])} | {export._cell(resolution)} | {flags} |"
        )
    return "\n".join(lines)


def _render_blast_radius(conn: sqlite3.Connection, task_id: str) -> str:
    row = blast_radius(conn, task_id)
    if row is None:
        return EMPTY
    reason = f" — {export._cell(row['reason'])}" if row["reason"] else ""
    return f"`{row['value']}` at `{export._cell(row['head_sha'])}` (review {export._cell(row['review'])}){reason}"


def register_export() -> None:
    """Append the three ledger sections to ``export.LEDGER_SECTIONS``, each only if its title is absent."""
    present = {title for title, _ in export.LEDGER_SECTIONS}
    for title, render in zip(SECTION_TITLES, (_render_rounds, _render_findings, _render_blast_radius), strict=True):
        if title not in present:
            export.LEDGER_SECTIONS.append((title, render))
