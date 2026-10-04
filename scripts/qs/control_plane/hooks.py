"""Hooks (§10): ``Stop`` and ``PreToolUse`` (Claude Code) and ``pre-push`` (git).

* ``Stop`` and ``PreToolUse`` **fail open**: on an error or a schema mismatch
  they allow, record the event if the DB is usable, and write one stderr line.
* ``pre-push`` refuses work-item refs (``QS_<N>_<k>``) everywhere, without the
  DB; otherwise it fails closed only for a worktree it can prove registered.
* The hooks never wait for a migration and never start the daemon.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sqlite3
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import clock as clock_mod
from . import db, errors, liveness, messages, paths, runner, runs, tokens, wait

SHIM_MARKER_PREFIX = "# qs-control-plane pre-push shim v"
SHIM_VERSION = 1
SHIM = f"""#!/bin/sh
{SHIM_MARKER_PREFIX}{SHIM_VERSION}
# Installed by `cp.py tool worktree-create` (QS-399): routes every push from
# every worktree of this repository through the Control Plane's pre-push hook.
MAIN="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
CP="$MAIN/scripts/qs/cp.py"
[ -f "$CP" ] || exit 0
if [ -x "$MAIN/venv/bin/python" ]; then
    exec "$MAIN/venv/bin/python" "$CP" hook pre-push "$@"
fi
exec python3 "$CP" hook pre-push "$@"
"""
ITEM_REF = re.compile(r"^refs/heads/QS_\d+_\d+$")
PRE_TOOL_MATCHER = "Bash|Edit|Write|SendMessage"
HOOK_TIMEOUT_S = 10
DB_BASENAMES = frozenset({"harness_state.db", "harness_state.db-wal", "harness_state.db-shm"})
READ_ONLY_PROGRAMS = frozenset({"grep", "rg", "git", "ls", "sed", "cat", "head", "tail", "wc", "find", "echo"})
_SEGMENT_SPLIT = re.compile(r"&&|\|\||;|\||\n")
_REDIRECT_ONTO_DB = re.compile(r">\s*\S*harness_state\.db")
STOP_QUEUE = "queue"
STOP_WAIT = "wait"


def _log(message: str) -> None:
    sys.stderr.write(f"[cp-hook] {message}\n")


@contextmanager
def _db_if_current(clock: clock_mod.Clock) -> Iterator[sqlite3.Connection | None]:
    """The DB when it exists at this code's schema; ``None`` when it is missing. Never waits."""
    path = paths.select_db()
    if db.check_schema(path, wait=False, clock=clock) == "missing":
        yield None
        return
    conn = db.connect(path)
    try:
        yield conn
    finally:
        conn.close()


def _record(
    conn: sqlite3.Connection,
    clock: clock_mod.Clock,
    hook: str,
    session_id: str | None,
    decision: str,
    detail: dict[str, Any],
) -> None:
    with db.write(conn):
        conn.execute(
            "INSERT INTO hook_events (hook, session_id, decision, detail, at) VALUES (?, ?, ?, ?, ?)",
            (hook, session_id, decision, json.dumps(detail, sort_keys=True), db.now(clock)),
        )


def _try_record(clock: clock_mod.Clock, hook: str, session_id: str | None, detail: dict[str, Any]) -> None:
    """Best effort: record a fail-open event when the DB is usable."""
    try:
        with _db_if_current(clock) as conn:
            if conn is not None:
                _record(conn, clock, hook, session_id, "allow", detail)
    except Exception:  # noqa: BLE001 — failing open means never raising from here
        pass


# --------------------------------------------------------------------------- Stop (§10.1)


def _stop_events(conn: sqlite3.Connection, session_id: str, n: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT decision, detail FROM hook_events WHERE hook = 'stop' AND session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, n),
    ).fetchall()
    return [{"decision": r["decision"], **json.loads(r["detail"])} for r in rows]


def _block(reason: str) -> str:
    return json.dumps({"decision": "block", "reason": reason})


def stop_decision(
    conn: sqlite3.Connection | None, clock: clock_mod.Clock, probe: liveness.ProcessProbe, session_id: str
) -> str:
    """The ``Stop`` hook's stdout: a block decision, or ``""`` to allow."""
    if conn is None:
        return ""
    lease = conn.execute(
        "SELECT l.run_id, r.state FROM run_leases l JOIN runs r ON r.id = l.run_id WHERE l.session_id = ?",
        (session_id,),
    ).fetchone()
    if lease is None or lease["state"] != "open":
        return ""  # not an orchestrator, a superseded one, or a closed run: the self-end rule
    run_id = lease["run_id"]
    main_dir = paths.main()
    cp = f"{main_dir}/venv/bin/python {main_dir}/scripts/qs/cp.py"
    now = db.now(clock)
    head = messages.head_id(conn, run_id, messages.ORCHESTRATOR, now)
    if head is not None:
        pending = messages.visible_count(conn, run_id, messages.ORCHESTRATOR, now)
        acked = conn.execute(
            "SELECT count(*) FROM messages WHERE run_id = ? AND state = 'acked'", (run_id,)
        ).fetchone()[0]
        last = _stop_events(conn, session_id, 2)
        stuck = len(last) == 2 and all(
            e["decision"] == "block"
            and e.get("kind") == STOP_QUEUE
            and e.get("head") == head
            and e.get("acked") == acked
            for e in last
        )
        if stuck:
            _record(
                conn, clock, "stop", session_id, "alert", {"kind": "queue_not_draining", "run_id": run_id, "head": head}
            )
            return ""
        _record(
            conn,
            clock,
            "stop",
            session_id,
            "block",
            {"kind": STOP_QUEUE, "run_id": run_id, "head": head, "acked": acked},
        )
        return _block(
            f"{pending} message(s) wait in run {run_id}'s queue: pop the next with "
            f"`{cp} msg pop --run {run_id} --as orchestrator --token …`, handle it, ack it."
        )
    if not wait.live_waiter(conn, probe, run_id):
        last = _stop_events(conn, session_id, 1)
        if last and last[0]["decision"] == "block" and last[0].get("kind") == STOP_WAIT:
            _record(conn, clock, "stop", session_id, "alert", {"kind": "idle_without_wait", "run_id": run_id})
            return ""
        _record(conn, clock, "stop", session_id, "block", {"kind": STOP_WAIT, "run_id": run_id})
        return _block(
            f"Run {run_id} is open and nothing waits on its queue: start `{cp} wait --run {run_id} --token …` in the background."
        )
    return ""


def hook_stop(stdin_text: str, clock: clock_mod.Clock, probe: liveness.ProcessProbe) -> str:
    session_id = None
    try:
        payload = json.loads(stdin_text or "{}")
        session_id = payload.get("session_id")
        if not session_id:
            return ""
        with _db_if_current(clock) as conn:
            return stop_decision(conn, clock, probe, str(session_id))
    except Exception as exc:  # noqa: BLE001 — Stop fails open
        _log(f"stop hook failed open: {exc!r}")
        _try_record(clock, "stop", session_id, {"kind": "error", "error": repr(exc)})
        return ""


# --------------------------------------------------------------------------- PreToolUse (§10.2)


def _segments(command: str) -> list[str]:
    return [s.strip() for s in _SEGMENT_SPLIT.split(command) if s.strip()]


def db_access_denial(tool_name: str, tool_input: dict[str, Any]) -> str | None:
    """The DB-access rule, for every session: a reason to deny, or ``None``."""
    if tool_name in ("Edit", "Write"):
        if Path(str(tool_input.get("file_path", ""))).name in DB_BASENAMES:
            return "the Control Plane DB is written only through cp.py"
        return None
    if tool_name != "Bash":
        return None
    for seg in _segments(str(tool_input.get("command", ""))):
        if "harness_state.db" not in seg or "scripts/qs/cp.py" in seg:
            continue
        words = seg.split()
        first = words[0]
        read_only = first in READ_ONLY_PROGRAMS and (first != "sed" or "-n" in words)
        if not read_only or _REDIRECT_ONTO_DB.search(seg):
            return f"direct access to the Control Plane DB is refused ({seg!r}); use cp.py"
    return None


def _deny(reason: str) -> str:
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
    )


def registered_denial(
    conn: sqlite3.Connection | None, session_id: str, tool_name: str, tool_input: dict[str, Any]
) -> str | None:
    status = runs.session_status(conn, session_id)
    if status["role"] == "none":
        return None  # the frozen pipeline keeps `gh pr merge`
    if tool_name == "Bash":
        for seg in _segments(str(tool_input.get("command", ""))):
            if seg.split()[:3] == ["gh", "pr", "merge"]:
                return "merges go through `cp.py tool merge`"
    if tool_name == "SendMessage" and status["state"] in ("superseded", "stopped"):
        return f"this session is {status['state']}: it may not send messages"
    return None


def hook_pre_tool_use(stdin_text: str, clock: clock_mod.Clock) -> str:
    session_id = None
    try:
        payload = json.loads(stdin_text or "{}")
        session_id = payload.get("session_id")
        tool_name = str(payload.get("tool_name", ""))
        tool_input = payload.get("tool_input") or {}
        reason = db_access_denial(tool_name, tool_input)
        with _db_if_current(clock) as conn:
            if reason is None and session_id:
                reason = registered_denial(conn, str(session_id), tool_name, tool_input)
            if reason is None:
                return ""
            if conn is not None:
                _record(conn, clock, "pre-tool-use", session_id, "deny", {"tool": tool_name, "reason": reason})
        return _deny(reason)
    except Exception as exc:  # noqa: BLE001 — PreToolUse fails open
        _log(f"pre-tool-use hook failed open: {exc!r}")
        _try_record(clock, "pre-tool-use", session_id, {"kind": "error", "error": repr(exc)})
        return ""


# --------------------------------------------------------------------------- pre-push (§10.3)


def _push_lines(stdin_text: str) -> list[tuple[str, str, str, str]]:
    out = []
    for line in stdin_text.splitlines():
        parts = line.split()
        if len(parts) == 4:
            out.append((parts[0], parts[1], parts[2], parts[3]))
    return out


def _registered_task(conn: sqlite3.Connection, toplevel: Path) -> sqlite3.Row | None:
    for row in conn.execute("SELECT * FROM tasks WHERE worktree IS NOT NULL").fetchall():
        if Path(row["worktree"]).resolve() == toplevel:
            return row
    return None


def _push_token_ok(conn: sqlite3.Connection, task: sqlite3.Row, node: sqlite3.Row | None, token: str | None) -> bool:
    if not token:
        return False
    try:
        who = tokens.require(conn, token, kinds={"run", "node"})
    except errors.CpError:
        return False
    if who.kind == "run":
        return who.run_id == task["run_id"]
    return node is not None and who.node_id == node["id"]


def pre_push_decision(
    stdin_text: str, *, clock: clock_mod.Clock, run: runner.Runner, token: str | None
) -> tuple[int, str]:
    """``(exit code, message)``: non-zero refuses the push."""
    lines = _push_lines(stdin_text)
    for local_ref, _, remote_ref, _ in lines:
        if ITEM_REF.match(local_ref) or ITEM_REF.match(remote_ref):
            return 1, f"refused: work-item refs (QS_<N>_<k>) are never pushed ({remote_ref})"
    try:
        top = run.run(["git", "rev-parse", "--show-toplevel"], timeout=30)
        if not top.ok:
            return 0, ""
        toplevel = Path(top.stdout.strip()).resolve()
        with _db_if_current(clock) as conn:
            if conn is None:
                return 0, ""
            with db.read(conn):
                task = _registered_task(conn, toplevel)
                if task is None:
                    return 0, ""
                return _registered_push(conn, task, lines, token)
    except Exception as exc:  # noqa: BLE001 — not proven registered: fail open
        _log(f"pre-push hook failed open: {exc!r}")
        return 0, ""


def _registered_push(
    conn: sqlite3.Connection, task: sqlite3.Row, lines: list[tuple[str, str, str, str]], token: str | None
) -> tuple[int, str]:
    try:
        node = conn.execute(
            "SELECT * FROM nodes WHERE task_id = ? ORDER BY generation DESC LIMIT 1", (task["id"],)
        ).fetchone()
        if node is not None and node["state"] == "stopped":
            return 1, f"refused: task {task['id']}'s node is stopped"
        allowed = f"refs/heads/{task['branch']}"
        for _, _, remote_ref, _ in lines:
            if task["branch"] is None or remote_ref != allowed:
                return 1, f"refused: this worktree may push only {allowed} (not {remote_ref})"
        if not _push_token_ok(conn, task, node, token):
            return 1, "refused: missing or stale QS_CP_TOKEN — push through `cp.py tool push`"
        return 0, ""
    except Exception as exc:  # noqa: BLE001 — proven registered: fail closed
        return 1, f"refused: pre-push check failed ({exc!r})"


def hook_pre_push(stdin_text: str, clock: clock_mod.Clock, run: runner.Runner) -> tuple[int, str]:
    return pre_push_decision(stdin_text, clock=clock, run=run, token=os.environ.get("QS_CP_TOKEN"))


def _under_temp(path: Path) -> bool:
    resolved = path.resolve()
    roots = {Path(tempfile.gettempdir()).resolve(), Path("/tmp").resolve()}
    return any(resolved.is_relative_to(r) for r in roots)


def install_pre_push(common_dir: Path, run: runner.Runner) -> dict[str, Any]:
    """Write the common-dir ``pre-push`` shim (idempotent), or ``POLICY_REFUSED``."""
    common_dir = Path(common_dir)
    if os.environ.get("PYTEST_CURRENT_TEST") and not _under_temp(common_dir):
        raise errors.CpError(
            "POLICY_REFUSED", f"refusing to install a hook outside a temporary dir under pytest: {common_dir}"
        )
    hooks_path = run.run(["git", "--git-dir", str(common_dir), "config", "--get", "core.hooksPath"], timeout=30)
    if hooks_path.ok and hooks_path.stdout.strip():
        raise errors.CpError("POLICY_REFUSED", f"core.hooksPath is set ({hooks_path.stdout.strip()}); not installing")
    target = common_dir / "hooks" / "pre-push"
    if target.exists():
        text = target.read_text(errors="replace")
        match = re.search(re.escape(SHIM_MARKER_PREFIX) + r"(\d+)", text)
        if match is None:
            raise errors.CpError("POLICY_REFUSED", f"a foreign pre-push hook exists at {target}")
        if int(match[1]) >= SHIM_VERSION:
            return {"path": str(target), "status": "already_installed"}
        status = "upgraded"
    else:
        status = "installed"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SHIM)
    target.chmod(0o755)
    return {"path": str(target), "status": status}


# --------------------------------------------------------------------------- settings (§10.4)


def _hook_command(main_dir: Path, hook: str) -> str:
    python = shlex.quote(str(main_dir / "venv" / "bin" / "python"))
    cp = shlex.quote(str(main_dir / "scripts" / "qs" / "cp.py"))
    return f'PY={python}; [ -x "$PY" ] || PY=python3; exec "$PY" {cp} hook {hook}'


def hooks_settings(role: str, main_dir: Path | None = None) -> dict[str, Any]:
    """The settings fragment: ``node`` gets ``PreToolUse``; ``orchestrator`` also gets ``Stop``."""
    if role not in ("node", "orchestrator"):
        raise errors.CpError("USAGE", "--role must be node or orchestrator")
    main_dir = (main_dir or paths.main()).resolve()
    entry = {"type": "command", "command": _hook_command(main_dir, "pre-tool-use"), "timeout": HOOK_TIMEOUT_S}
    fragment: dict[str, Any] = {"hooks": {"PreToolUse": [{"matcher": PRE_TOOL_MATCHER, "hooks": [entry]}]}}
    if role == "orchestrator":
        stop = {"type": "command", "command": _hook_command(main_dir, "stop"), "timeout": HOOK_TIMEOUT_S}
        fragment["hooks"]["Stop"] = [{"hooks": [stop]}]
    return fragment
