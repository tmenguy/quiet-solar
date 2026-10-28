"""``cp.py catchup <op>`` (#440): the node's catch-up with ``main`` as a tested state machine.

The node calls ``catchup step`` at each decision point of its catch-up and does what the answer says;
the decisions and the git effects (the fetch, the merge, the lookups) are made here. Five ops, each a
write-kind command (a token check for every op, the read-only ones included):

==========  =============================================================  ==============================
op          arguments                                                      answers (``next``)
==========  =============================================================  ==============================
``step``    ``--from conflict|done --seq N --head SHA`` + the reports       ``implement`` ``review``
            (``--implemented green|red --inv I``, ``--reviewed SHA``)       ``push`` ``wait`` ``fixes``
                                                                            ``done`` ``blocked``
``base``    —                                                              ``{"base": <sha>|null}``
``fixes``   —                                                              ``fixes``
``done``    —                                                              ``done`` or ``fixes``
``unblock`` —                                                              ``{"unblocked": <cause>|null}``
==========  =============================================================  ==============================

**The state** lives in ``<worktree>/.qs-node/catchup.json``, owned by this module and never written by the
node (``round.json`` keeps only the trigger ``{"from", "seq", "head"}``); ``pending.json`` stays the
node's and is only read here. Every write is atomic (``mkstemp`` → fsync → ``os.replace``), preceded by a
token re-check, and followed by its named fault point (``FAULTS``): a reap anywhere is absorbed by
re-issuing the same call.

**Order inside every op**: the token and the task row (one ``db.write``), the worktree and its
``.qs-node/catchup.lock`` (``BUSY`` when held), then the op. No DB transaction is held across a git call;
every git call is bounded by ``CATCHUP_GIT_TIMEOUT_S`` and the op's own deadline.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import backups, db, errors, faults, gitread, ledger, node_policy, runner, tasks, tokens, tools
from . import clock as clock_mod

CATCHUP_GIT_TIMEOUT_S = 120.0  # one git call
CATCHUP_OP_DEADLINE_S = 300.0  # one op, from the handler's entry: the lock's real upper bound
CATCHUP_MIN_LEFT_S = 60.0  # the least time left to start a fetch or a merge (a killed merge leaves an index.lock)
FAULTS = (
    "catchup.after_adopt",
    "catchup.after_intake",
    "catchup.after_reviewed",
    "catchup.after_unblock",
    "catchup.after_bound",
    "catchup.after_count",
    "catchup.after_fetch_precheck",
    "catchup.after_fetch_merge",
    "catchup.after_stale",
    "catchup.after_merge",
    "catchup.after_record",
    "catchup.after_conflict_record",
    "catchup.after_lookup_record",
    "catchup.after_push_write",
    "catchup.after_fixes_write",
    "catchup.after_done_write",
)
VERSION = 1
FROMS = ("conflict", "done")
GATES = ("green", "red")
KINDS = frozenset({"run", "node"})
FULL_SHA_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")  # a full lowercase sha (``gitread.SHA_RE`` takes abbreviations)
STATE_FILE = "catchup.json"
PENDING_FILE = "pending.json"
LOCK_FILE = "catchup.lock"
MERGE_NOT_COMMITTED = "merge not committed"
GATE_RED = "gate red"
DONE_LEFT = ("digest", "report", "state")
_REF_LOCK = ("cannot lock ref", "Unable to create")
_LAST_KINDS = ("wait", "done", "fixes")
_BLOCK_CAUSES = ("catchups", "misses")
_CATCHUP_KEYS = frozenset(
    {"from", "trigger", "head", "counted", "merge", "clean", "green", "reviewed", "misses", "intake", "block"}
)

_monotonic = time.monotonic  # the deadline's seam: tests patch ``catchup._monotonic``, never ``time.monotonic``


def empty_state() -> dict[str, Any]:
    return {
        "version": VERSION,
        "catchups": 0,
        "max_seq": 0,
        "catchup": None,
        "base_merge": None,
        "last": None,
        "done_report": None,
    }


# --------------------------------------------------------------------------- the state file


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _sha(value: Any) -> bool:
    return isinstance(value, str) and FULL_SHA_RE.fullmatch(value) is not None


def _opt_sha(value: Any) -> bool:
    return value is None or _sha(value)


def _trigger_ok(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"from", "seq", "head"}
        and value["from"] in FROMS
        and _count(value["seq"])
        and value["seq"] >= 1
        and _sha(value["head"])
    )


def _wait_ok(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"next", "head", "catchup_head"}
        and value["next"] == "wait"
        and _sha(value["head"])
        and type(value["catchup_head"]) is bool
    )


def _last_ok(value: Any) -> bool:
    if value is None:
        return True
    return (
        isinstance(value, dict)
        and set(value) == {"trigger", "kind", "answer"}
        and _trigger_ok(value["trigger"])
        and value["kind"] in _LAST_KINDS
        and (_wait_ok(value["answer"]) if value["kind"] == "wait" else value["answer"] is None)
    )


def _block_ok(value: Any) -> bool:
    return value is None or (
        isinstance(value, dict)
        and set(value) == {"cause", "detail", "at"}
        and value["cause"] in _BLOCK_CAUSES
        and isinstance(value["detail"], str)
        and isinstance(value["at"], str)
    )


def _catchup_ok(value: Any) -> bool:
    if value is None:
        return True
    return (
        isinstance(value, dict)
        and set(value) == _CATCHUP_KEYS
        and value["from"] in FROMS
        and _trigger_ok(value["trigger"])
        and _sha(value["head"])
        and type(value["counted"]) is bool
        and _opt_sha(value["merge"])
        and (value["clean"] is None or type(value["clean"]) is bool)
        and _opt_sha(value["green"])
        and _opt_sha(value["reviewed"])
        and isinstance(value["misses"], list)
        and len(value["misses"]) <= 2
        and all(m in (MERGE_NOT_COMMITTED, GATE_RED) for m in value["misses"])
        and isinstance(value["intake"], list)
        and all(_count(i) and i >= 1 for i in value["intake"])
        and _block_ok(value["block"])
    )


def _base_merge_ok(value: Any) -> bool:
    return value is None or (
        isinstance(value, dict) and set(value) == {"merge", "head"} and _sha(value["merge"]) and _sha(value["head"])
    )


def valid_state(data: Any) -> bool:
    """The one validation of ``catchup.json``: its version, every field's type, nothing more, nothing less."""
    return (
        isinstance(data, dict)
        and set(data) == set(empty_state())
        and type(data["version"]) is int
        and data["version"] == VERSION
        and _count(data["catchups"])
        and _count(data["max_seq"])
        and (data["done_report"] is None or _count(data["done_report"]))
        and _base_merge_ok(data["base_merge"])
        and _last_ok(data["last"])
        and _catchup_ok(data["catchup"])
    )


def load_state(node_dir: Path) -> dict[str, Any]:
    """``catchup.json`` → the state; missing → the empty state; anything unreadable → ``INVALID_STATE``."""
    path = node_dir / STATE_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return empty_state()
    except (OSError, ValueError) as exc:  # UnicodeDecodeError is a ValueError
        raise errors.CpError("INVALID_STATE", f"{path} is unreadable: {exc}") from exc
    if not valid_state(data):
        raise errors.CpError("INVALID_STATE", f"{path} is corrupt (repair or delete it)")
    result: dict[str, Any] = data
    return result


def save_state(node_dir: Path, state: dict[str, Any]) -> None:
    """Write ``catchup.json`` atomically: a temp file, fsync, ``os.replace``, a best-effort directory fsync."""
    text = json.dumps(state, sort_keys=True, indent=1) + "\n"
    tmp: str | None = None
    try:
        fd, tmp = tempfile.mkstemp(dir=node_dir, prefix=f".{STATE_FILE}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        backups._fsync(Path(tmp))
        os.replace(tmp, node_dir / STATE_FILE)
    except OSError as exc:
        if tmp is not None:
            Path(tmp).unlink(missing_ok=True)
        raise errors.CpError("TOOL_FAILED", f"cannot write {node_dir / STATE_FILE}: {exc}") from exc
    with contextlib.suppress(OSError):  # the rename surviving a crash is best-effort
        backups._fsync(node_dir)


def _entry_ok(entry: Any) -> bool:
    return isinstance(entry, dict) and type(entry.get("finding_id")) is int and entry.get("level") in ("code", "plan")


def load_pending(node_dir: Path) -> list[dict[str, Any]]:
    """``pending.json`` (the node's): a list of ``{"finding_id": int, "level": code|plan, …}``; missing → ``[]``."""
    path = node_dir / PENDING_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        raise errors.CpError("INVALID_STATE", f"{path} is unreadable: {exc}") from exc
    if not isinstance(data, list) or not all(_entry_ok(e) for e in data):
        raise errors.CpError("INVALID_STATE", f"{path} is not a list of {{finding_id, level}} entries")
    return data


# --------------------------------------------------------------------------- answers


def blocked(cause: str, detail: str) -> dict[str, Any]:
    return {"next": "blocked", "cause": cause, "detail": detail}


def _error(detail: str) -> dict[str, Any]:
    return blocked("error", detail)


def wait_answer(head: str, catchup_head: bool) -> dict[str, Any]:
    return {"next": "wait", "head": head, "catchup_head": catchup_head}


def fixes_answer(open_: list[dict[str, Any]], drop: list[int]) -> dict[str, Any]:
    return {"next": "fixes", "open": open_, "drop": drop}


def done_answer(left: list[str], objections: list[dict[str, Any]], drop: list[int]) -> dict[str, Any]:
    return {"next": "done", "left": left, "objections": objections, "drop": drop}


# --------------------------------------------------------------------------- git


class _Git:
    """Every git call of an op: in the worktree, bounded by min(``CATCHUP_GIT_TIMEOUT_S``, the op's time left)."""

    def __init__(self, run: runner.Runner, worktree: Path, start: float) -> None:
        self.runner = run
        self.worktree = worktree
        self.start = start

    def left(self) -> float:
        return CATCHUP_OP_DEADLINE_S - (_monotonic() - self.start)

    def call(self, *args: str, need: float = 0.0) -> runner.RunResult:
        left = self.left()
        if left <= 0 or left < need:
            raise errors.CpError("TOOL_FAILED", "op deadline", command=f"git {' '.join(args)}", left_s=max(left, 0.0))
        return self.runner.run(
            ["git", *args],
            cwd=self.worktree,
            env_extra={"GIT_TERMINAL_PROMPT": "0"},
            env_remove=gitread.ENV_REMOVED,
            timeout=min(CATCHUP_GIT_TIMEOUT_S, left),
        )

    @staticmethod
    def fail(args: tuple[str, ...], res: runner.RunResult) -> errors.CpError:
        return errors.CpError("TOOL_FAILED", f"git {' '.join(args)} exited {res.returncode}", tail=tools.tail(res))

    def out(self, *args: str) -> str:
        res = self.call(*args)
        if res.returncode != 0:
            raise self.fail(args, res)
        return res.stdout.strip()

    def branch(self) -> str:
        args = ("symbolic-ref", "--short", "HEAD")
        res = self.call(*args)
        if res.returncode in (1, 128):
            raise errors.CpError("INVALID_STATE", "detached or not on the branch")
        if res.returncode != 0:
            raise self.fail(args, res)
        return res.stdout.strip()

    def rev(self, ref: str) -> str:
        return self.out("rev-parse", ref)

    def _yes_no(self, *args: str) -> bool:
        res = self.call(*args)
        if res.returncode not in (0, 1):
            raise self.fail(args, res)
        return res.returncode == 0

    def merge_open(self) -> bool:
        return self._yes_no("rev-parse", "-q", "--verify", "MERGE_HEAD")

    def ancestor(self, a: str, b: str) -> bool:
        return self._yes_no("merge-base", "--is-ancestor", a, b)

    def known(self, sha: str) -> bool:
        args = ("cat-file", "-e", f"{sha}^{{commit}}")
        res = self.call(*args)
        if res.returncode in (124, 127):
            raise self.fail(args, res)
        return res.returncode == 0

    def fetch(self) -> None:
        args = ("fetch", "origin", "main")
        res = self.call(*args, need=CATCHUP_MIN_LEFT_S)
        if res.returncode == 0:
            return
        if any(marker in res.stderr for marker in _REF_LOCK):
            raise errors.CpError("BUSY", "git fetch origin main: a ref is locked", tail=tools.tail(res))
        raise self.fail(args, res)

    def remote(self, branch: str) -> str | None:
        out = self.out("ls-remote", "origin", f"refs/heads/{branch}")
        return out.split()[0] if out else None

    def lookup(self, head: str) -> str | None:
        """The last first-parent merge of ``head..HEAD`` (the catch-up's merge commit), or ``None``."""
        return self.out("log", "-1", "--first-parent", "--merges", "--format=%H", f"{head}..HEAD") or None

    def merge(self) -> bool:
        """``git merge --no-ff --no-edit origin/main`` → ``True`` clean, ``False`` conflicts (``MERGE_HEAD`` open)."""
        args = ("merge", "--no-ff", "--no-edit", "origin/main")
        res = self.call(*args, need=CATCHUP_MIN_LEFT_S)
        if res.returncode == 0:
            return True
        if self.merge_open():
            return False
        raise self.fail(args, res)


# --------------------------------------------------------------------------- one op


class _Op:
    def __init__(
        self, conn: sqlite3.Connection, run: runner.Runner, clock: clock_mod.Clock, token: str, task_id: str
    ) -> None:
        self.start = _monotonic()  # first: the deadline holds even while the token check waits
        self.conn = conn
        self.clock = clock
        self.token = token
        self.task_id = task_id
        with db.write(conn):
            tokens.require(conn, token, kinds=KINDS, task_id=task_id)
            task = tasks.get(conn, task_id)
        if task["state"] in tasks.TERMINAL:
            raise errors.CpError("INVALID_STATE", f"task {task_id} is {task['state']}")
        tools._not_an_item(task)
        self.worktree = Path(tools._need(task, "worktree"))
        self.branch = str(tools._need(task, "branch"))
        if not self.worktree.is_dir():
            raise errors.CpError("INVALID_STATE", f"task {task_id}'s worktree {self.worktree} is not a directory")
        self.node_dir = self.worktree / ".qs-node"
        self.node_dir.mkdir(exist_ok=True)
        self.git = _Git(run, self.worktree, self.start)
        self.fetched = False

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        with db.file_lock(self.node_dir / LOCK_FILE, exclusive=True, timeout=0) as got:
            if not got:
                raise errors.CpError("BUSY", f"another catchup call holds {self.node_dir / LOCK_FILE}")
            yield

    # ----------------------------------------------------------------- state

    def commit(self, state: dict[str, Any]) -> None:
        """Re-check the token, then write ``catchup.json`` (``max_seq`` follows every recorded trigger)."""
        for holder in (state["catchup"], state["last"]):
            if holder is not None:
                state["max_seq"] = max(state["max_seq"], holder["trigger"]["seq"])
        with db.write(self.conn):
            tokens.require(self.conn, self.token, kinds=KINDS, task_id=self.task_id)
            save_state(self.node_dir, state)

    def pending(self) -> tuple[list[dict[str, Any]], list[int]]:
        """``pending.json`` against the ledger → ``(open entries, dropped finding ids)``, de-duplicated."""
        entries = load_pending(self.node_dir)
        ids = sorted({e["finding_id"] for e in entries})
        open_ids: set[int] = set()
        if ids:
            marks = ", ".join("?" for _ in ids)
            with db.read(self.conn):
                rows = self.conn.execute(
                    f"SELECT id, state FROM findings WHERE task_id = ? AND id IN ({marks})", (self.task_id, *ids)
                ).fetchall()
            open_ids = {int(r["id"]) for r in rows if r["state"] == "open"}
        seen: set[int] = set()
        open_: list[dict[str, Any]] = []
        drop: list[int] = []
        for entry in entries:
            fid = entry["finding_id"]
            if fid in seen:
                continue
            seen.add(fid)
            if fid in open_ids:
                open_.append(entry)
            else:
                drop.append(fid)
        return open_, drop

    def consumed(self, base_merge: dict[str, Any]) -> bool:
        """A build round of the task based on the merge exists (the next round consumed it)."""
        with db.read(self.conn):
            row = self.conn.execute(
                "SELECT 1 FROM rounds WHERE task_id = ? AND phase = 'build' AND base_sha = ? LIMIT 1",
                (self.task_id, base_merge["merge"]),
            ).fetchone()
        return row is not None

    def remote_state(self, head: str) -> str:
        """``pushed`` (the remote branch is ``head``), ``ahead`` (unknown here, or not an ancestor), else ``behind``."""
        remote = self.git.remote(self.branch)
        if remote == head:
            return "pushed"
        if remote is not None and (not self.git.known(remote) or not self.git.ancestor(remote, head)):
            return "ahead"
        return "behind"

    def implement(self, mode: str) -> dict[str, Any]:
        open_, _ = self.pending()
        return {"next": "implement", "mode": mode, "fixes": [e for e in open_ if e["level"] == "code"]}

    # ----------------------------------------------------------------- step (D4)

    def step(
        self, trigger: dict[str, Any], implemented: str | None, inv: int | None, reviewed: str | None
    ) -> dict[str, Any]:
        if self.git.branch() != self.branch:
            raise errors.CpError("INVALID_STATE", "detached or not on the branch")
        state = load_state(self.node_dir)
        cu = state["catchup"]
        if cu is None:
            last = state["last"]
            if last is not None and last["trigger"] == trigger:
                if last["kind"] == "wait":
                    answer: dict[str, Any] = last["answer"]
                    return answer
                return self.done_checks(state, trigger)
            if trigger["seq"] <= state["max_seq"]:
                raise errors.CpError("INVALID_STATE", "a trigger older than the last one")
            return self.count(state, trigger)
        if cu["trigger"] != trigger:
            if trigger["seq"] <= cu["trigger"]["seq"]:
                raise errors.CpError("INVALID_STATE", "a trigger older than the catch-up in progress")
            cu["trigger"] = dict(trigger)
            self.commit(state)
            faults.hit("catchup.after_adopt")
        if cu["block"] is not None:
            return blocked(cu["block"]["cause"], cu["block"]["detail"])
        if implemented is not None:
            assert inv is not None
            answer_ = self.intake(state, implemented, inv)
            if answer_ is not None:
                return answer_
        if reviewed is not None and reviewed == self.git.rev("HEAD"):
            cu["reviewed"] = reviewed
            self.commit(state)
            faults.hit("catchup.after_reviewed")
        if not cu["counted"]:
            return self.count(state, trigger)
        return self.progress(state)

    def intake(self, state: dict[str, Any], gate: str, inv: int) -> dict[str, Any] | None:
        """Step 1's report intake: once per ``inv``; a miss counted, a green gate recorded (one write)."""
        cu = state["catchup"]
        intake = cu["intake"]
        if intake and inv == intake[-1]:
            return None  # a retried call: never counted twice
        if inv in intake or (intake and inv <= max(intake)):
            raise errors.CpError("INVALID_STATE", "inv regressed")
        if cu["merge"] is None:
            found = self.git.lookup(cu["head"])
            if found is not None:
                if not self.git.ancestor(self.git.rev(f"{found}^2"), "origin/main"):
                    return _error("a merge that is not origin/main's")
                cu["merge"], cu["clean"] = found, False
        if cu["merge"] is None:
            cu["misses"].append(MERGE_NOT_COMMITTED)
        elif gate == "red":
            cu["misses"].append(GATE_RED)
        else:
            cu["green"] = self.git.rev("HEAD")
        intake.append(inv)
        if len(cu["misses"]) >= 2:
            detail = f"{cu['misses'][0]}, then {cu['misses'][1]}"
            cu["block"] = {"cause": "misses", "detail": detail, "at": db.now(self.clock)}
        self.commit(state)
        faults.hit("catchup.after_intake")
        if cu["block"] is not None:
            return blocked("misses", cu["block"]["detail"])
        return None

    def count(self, state: dict[str, Any], trigger: dict[str, Any]) -> dict[str, Any]:
        """Step 2: a new catch-up, or the re-run of an uncounted one — the guards, the pre-check, the bound."""
        cu = state["catchup"]
        from_ = trigger["from"] if cu is None else cu["from"]
        kept = trigger if cu is None else cu["trigger"]
        head = trigger["head"]
        if self.git.merge_open():
            return _error("a merge is open outside a catch-up")
        current = self.git.rev("HEAD")
        if current != head:
            return _error("HEAD is not the pushed head")
        if from_ == "done":
            last, base_merge = state["last"], state["base_merge"]
            own_wait = (
                last is not None
                and last["kind"] == "wait"
                and last["answer"]["catchup_head"]
                and last["answer"]["head"] == current
                and base_merge is not None
                and base_merge["head"] == current
            )
            if own_wait and not self.consumed(base_merge):
                return self.done_checks(state, trigger)
            self.git.fetch()
            faults.hit("catchup.after_fetch_precheck")
            self.fetched = True
            if self.git.ancestor("origin/main", current):
                return self.done_checks(state, trigger)
        fresh = {
            "from": from_,
            "trigger": dict(kept),
            "head": head,
            "counted": False,
            "merge": None,
            "clean": None,
            "green": None,
            "reviewed": None,
            "misses": [],
            "intake": [],
            "block": None,
        }
        if state["catchups"] >= node_policy.MAX_CATCHUPS:
            detail = f"{state['catchups']} catch-ups since the last done (the bound is {node_policy.MAX_CATCHUPS})"
            fresh["block"] = {"cause": "catchups", "detail": detail, "at": db.now(self.clock)}
            state["catchup"] = fresh
            self.commit(state)
            faults.hit("catchup.after_bound")
            return blocked("catchups", detail)
        fresh["counted"] = True
        state["catchups"] += 1
        state["catchup"] = fresh
        self.commit(state)
        faults.hit("catchup.after_count")
        return self.progress(state)

    def progress(self, state: dict[str, Any]) -> dict[str, Any]:
        """Steps 3 and 4: the merge, then what is left after it (the gate, the review, the push)."""
        cu = state["catchup"]
        if cu["merge"] is None:
            resolve = "resolve" if not cu["misses"] else "finish"
            if self.git.merge_open():
                return self.implement(resolve)
            current = self.git.rev("HEAD")
            if current == cu["head"]:
                if not self.fetched:
                    self.git.fetch()
                    faults.hit("catchup.after_fetch_merge")
                if self.git.ancestor("origin/main", current):
                    if cu["from"] == "done":
                        return self.done_checks(state, cu["trigger"])
                    answer = wait_answer(current, False)  # stale: back to the wait, base_merge untouched
                    state["catchup"] = None
                    state["last"] = {"trigger": cu["trigger"], "kind": "wait", "answer": answer}
                    self.commit(state)
                    faults.hit("catchup.after_stale")
                    return answer
                clean = self.git.merge()
                faults.hit("catchup.after_merge")
                if not clean:
                    cu["clean"] = False
                    self.commit(state)
                    faults.hit("catchup.after_conflict_record")
                    return self.implement(resolve)
                cu["merge"], cu["clean"] = self.git.rev("HEAD"), True
                self.commit(state)
                faults.hit("catchup.after_record")
            else:
                found = self.git.lookup(cu["head"])
                if found is None:
                    return _error("HEAD moved without a merge")
                if not self.git.ancestor(self.git.rev(f"{found}^2"), "origin/main"):
                    return _error("a merge that is not origin/main's")
                cu["merge"], cu["clean"] = found, False  # unknown, so reviewed
                self.commit(state)
                faults.hit("catchup.after_lookup_record")
        return self.after_merge(state)

    def after_merge(self, state: dict[str, Any]) -> dict[str, Any]:
        cu = state["catchup"]
        merge = cu["merge"]
        if self.git.merge_open():
            return _error("a second merge is open")
        current = self.git.rev("HEAD")
        if not self.git.ancestor(merge, current):
            return _error("the catch-up merge is no longer in HEAD")
        if cu["green"] != current:
            return self.implement("gate")
        first = self.git.rev(f"{merge}^1")
        if (not cu["clean"] or current != merge or first != cu["head"]) and cu["reviewed"] != current:
            diffs = [f"git show --remerge-diff {merge}"]
            if first != cu["head"]:
                diffs.append(f"git diff {cu['head']}..{merge}^1")
            if current != merge:
                diffs.append(f"git diff {merge}..{current}")
            return {"next": "review", "head": current, "diffs": diffs}
        remote = self.remote_state(current)
        if remote == "ahead":
            return _error("the remote branch is ahead of HEAD")
        if remote != "pushed":
            return {"next": "push", "head": current}
        answer = wait_answer(current, True)
        state["base_merge"] = {"merge": merge, "head": current}
        state["catchup"] = None
        state["last"] = {"trigger": cu["trigger"], "kind": "wait", "answer": answer}
        self.commit(state)
        faults.hit("catchup.after_push_write")
        return answer

    def done_checks(self, state: dict[str, Any], trigger: dict[str, Any]) -> dict[str, Any]:
        """Step 5: pushed, then the pending fixes → ``fixes``, or ``done`` (the counter reset)."""
        current = self.git.rev("HEAD")
        remote = self.remote_state(current)
        if remote == "ahead":
            return _error("the remote branch is ahead of HEAD")
        if remote != "pushed":
            return _error("HEAD is not pushed")
        open_, drop = self.pending()
        state["catchup"] = None
        if open_:
            state["last"] = {"trigger": dict(trigger), "kind": "fixes", "answer": None}
            self.commit(state)
            faults.hit("catchup.after_fixes_write")
            return fixes_answer(open_, drop)
        with db.read(self.conn):
            top = self.conn.execute("SELECT max(id) FROM reports WHERE task_id = ?", (self.task_id,)).fetchone()[0]
        state["catchups"] = 0
        state["last"] = {"trigger": dict(trigger), "kind": "done", "answer": None}
        state["done_report"] = int(top or 0)
        self.commit(state)
        faults.hit("catchup.after_done_write")
        return done_answer(list(DONE_LEFT), [], drop)

    # ----------------------------------------------------------------- the other ops (D4 step 1, D5, D6)

    def base(self) -> dict[str, Any]:
        state = load_state(self.node_dir)
        cu = state["catchup"]
        if self.git.merge_open():
            raise errors.CpError("INVALID_STATE", "a merge is open: no build round starts over it")
        if cu is not None and cu["merge"] is not None:
            raise errors.CpError("INVALID_STATE", "a catch-up merge is not pushed yet")
        base_merge = state["base_merge"]
        if base_merge is None or self.consumed(base_merge):
            return {"base": None}
        if not self.git.ancestor(base_merge["merge"], self.git.rev("HEAD")):
            return {"base": None}
        return {"base": base_merge["merge"]}

    def fixes(self) -> dict[str, Any]:
        return fixes_answer(*self.pending())

    def done(self) -> dict[str, Any]:
        state = load_state(self.node_dir)
        if state["catchup"] is not None:
            raise errors.CpError("INVALID_STATE", "a catch-up is in progress")
        with db.read(self.conn):
            number = ledger.latest_round(self.conn, self.task_id, "build")
            report = self.conn.execute(
                "SELECT id, status FROM reports WHERE task_id = ? AND phase = 'build' AND round = ?"
                " ORDER BY id DESC LIMIT 1",
                (self.task_id, number),
            ).fetchone()
            task_state = tasks.get(self.conn, self.task_id)["state"]
        done_report = state["done_report"]
        converged = (
            report is not None
            and report["status"] == "converged"
            and done_report is not None
            and int(report["id"]) > done_report
        )
        open_, drop = self.pending()
        if converged:
            return done_answer([] if task_state == "ready_to_merge" else ["state"], open_, drop)
        if open_:
            return fixes_answer(open_, drop)
        return done_answer(list(DONE_LEFT), [], drop)

    def unblock(self) -> dict[str, Any]:
        state = load_state(self.node_dir)
        cu = state["catchup"]
        if cu is None or cu["block"] is None:
            return {"unblocked": None}
        cause = cu["block"]["cause"]
        if cause == "catchups":
            state["catchups"] = 0
        else:
            cu["misses"] = []
        cu["block"] = None
        self.commit(state)
        faults.hit("catchup.after_unblock")
        return {"unblocked": cause}


# --------------------------------------------------------------------------- the ops


def step(
    conn: sqlite3.Connection,
    run: runner.Runner,
    clock: clock_mod.Clock,
    *,
    token: str,
    task_id: str,
    trigger_from: str,
    seq: int,
    head: str,
    implemented: str | None = None,
    inv: int | None = None,
    reviewed: str | None = None,
) -> dict[str, Any]:
    """``catchup step``: the decision procedure (D4) for the trigger ``(from, seq, head)`` and the reports."""
    if trigger_from not in FROMS:
        raise errors.CpError("USAGE", f"--from must be one of {', '.join(FROMS)}: {trigger_from!r}")
    if (implemented is None) != (inv is None):
        raise errors.CpError("USAGE", "--implemented and --inv go together")
    if implemented is not None and implemented not in GATES:
        raise errors.CpError("USAGE", f"--implemented must be one of {', '.join(GATES)}: {implemented!r}")
    for flag, value in (("head", head), ("reviewed", reviewed)):
        if value is not None and not _sha(value):
            raise errors.CpError("USAGE", f"--{flag} must be a full lowercase sha: {value!r}")
    if type(seq) is not int or seq < 1 or (inv is not None and (type(inv) is not int or inv < 1)):
        raise errors.CpError("USAGE", "--seq and --inv must be positive integers")
    op = _Op(conn, run, clock, token, task_id)
    with op.locked():
        return op.step({"from": trigger_from, "seq": seq, "head": head}, implemented, inv, reviewed)


def _simple(
    name: str, conn: sqlite3.Connection, run: runner.Runner, clock: clock_mod.Clock, token: str, task_id: str
) -> dict[str, Any]:
    op = _Op(conn, run, clock, token, task_id)
    with op.locked():
        result: dict[str, Any] = getattr(op, name)()
        return result


def base(
    conn: sqlite3.Connection, run: runner.Runner, clock: clock_mod.Clock, *, token: str, task_id: str
) -> dict[str, Any]:
    """``catchup base``: the next build round's ``--base`` — the last pushed catch-up merge while unconsumed."""
    return _simple("base", conn, run, clock, token, task_id)


def fixes(
    conn: sqlite3.Connection, run: runner.Runner, clock: clock_mod.Clock, *, token: str, task_id: str
) -> dict[str, Any]:
    """``catchup fixes``: step 6's pending-fix check (read-only)."""
    return _simple("fixes", conn, run, clock, token, task_id)


def done(
    conn: sqlite3.Connection, run: runner.Runner, clock: clock_mod.Clock, *, token: str, task_id: str
) -> dict[str, Any]:
    """``catchup done``: step 8's resume (read-only, D6)."""
    return _simple("done", conn, run, clock, token, task_id)


def unblock(
    conn: sqlite3.Connection, run: runner.Runner, clock: clock_mod.Clock, *, token: str, task_id: str
) -> dict[str, Any]:
    """``catchup unblock``: clear a ``catchups`` or ``misses`` block (idempotent: ``{"unblocked": null}``)."""
    return _simple("unblock", conn, run, clock, token, task_id)
