"""The ``cp.py`` command line: a ``COMMANDS`` dispatch table, JSON out, exit codes.

Every command prints one JSON object on stdout — ``{"ok": true, …}`` or
``{"ok": false, "error": "<CODE>", "detail": "…"}`` — and exits with the
code of ``errors.EXIT_CODES``. Hooks are the exception: they print the
Claude Code hook protocol (or nothing).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, NoReturn, TextIO

from . import clock as clock_mod
from . import (
    daemon,
    db,
    decisions,
    errors,
    export,
    hooks,
    items,
    liveness,
    locks,
    messages,
    migrations,
    nodes,
    paths,
    procsetup,
    questions,
    reports,
    runner,
    runs,
    snapshot,
    tasks,
    tools,
    wait,
)

items.register_item_tools()  # #400's tools: the parser and the parametrized tests read REGISTRY at import


@dataclass(frozen=True)
class Raw:
    """A handler result printed verbatim (hooks), with its exit code."""

    text: str
    code: int = 0


@dataclass
class Deps:
    """Every seam a command uses; the tests replace ``make_deps``."""

    clock: clock_mod.Clock
    runner: runner.Runner
    probe: liveness.ProcessProbe
    claude: liveness.ClaudeCli
    popen: Callable[..., Any]
    kill: Callable[[int, int], None]


def make_deps() -> Deps:
    run = runner.Runner()
    return Deps(
        clock=clock_mod.SystemClock(),
        runner=run,
        probe=liveness.ProcessProbe(run),
        claude=liveness.ClaudeCli(run),
        popen=subprocess.Popen,
        kill=os.kill,
    )


@dataclass
class Io:
    stdin: TextIO
    stdout: TextIO
    deps: Deps


def ensure_daemon(io: Io, path: Any = None) -> dict[str, Any]:
    d = io.deps
    return daemon.ensure(popen=d.popen, clock=d.clock, probe=d.probe, db_path=path, kill=d.kill)


def _daemon_status(io: Io, result: dict[str, Any]) -> dict[str, Any]:
    """After ``run open`` / ``run claim``: the daemon's status; a start error never hides the run token."""
    try:
        result["daemon"] = ensure_daemon(io)["status"]
    except Exception as exc:  # noqa: BLE001 — the run is already opened or claimed
        result["daemon"] = "error"
        result["daemon_error"] = f"{type(exc).__name__}: {exc}"
    return result


MAX_TIME_S = 86400.0
MIN_POLL_S = 0.05


def _float_flag(*, allow_zero: bool, minimum: float = 0.0) -> Callable[[str], float]:
    """A time flag: finite, ``> 0`` (or ``>= 0``), at least ``minimum``, at most a day."""

    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
        if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            raise argparse.ArgumentTypeError(f"must be finite and {'>= 0' if allow_zero else '> 0'}: {text!r}")
        if value < minimum or value > MAX_TIME_S:
            raise argparse.ArgumentTypeError(f"must be between {minimum:g} and {MAX_TIME_S:g} seconds: {text!r}")
        return value

    return parse


POSITIVE = _float_flag(allow_zero=False)
NON_NEGATIVE = _float_flag(allow_zero=True)
POLL = _float_flag(allow_zero=False, minimum=MIN_POLL_S)


@contextmanager
def connection(io: Io, kind: str) -> Iterator[sqlite3.Connection | None]:
    """Open the selected DB after the entry schema check of ``kind`` (§3).

    ``write`` waits for the self-migration (starting the daemon); ``read``
    never waits and yields ``None`` for a missing DB.
    """
    path = paths.select_db()
    if kind == "write":
        db.check_schema(path, wait=True, clock=io.deps.clock, ensure=lambda: ensure_daemon(io, path))
    elif db.check_schema(path, wait=False, clock=io.deps.clock) == "missing":
        yield None
        return
    conn = db.connect(path, mode="rw" if kind == "write" else "ro")
    try:
        yield conn
    finally:
        conn.close()


@dataclass(frozen=True)
class Command:
    """One CLI entry. ``kind`` is ``exempt``, ``read`` or ``write`` (§3)."""

    name: str
    kind: str
    handler: Callable[[argparse.Namespace, Io], Any]
    configure: Callable[[argparse.ArgumentParser], None] = field(default=lambda p: None)
    help: str = ""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise errors.CpError("USAGE", message)


# --------------------------------------------------------------------------- handlers


def _version(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    return {"package": "control_plane", "schema_version": migrations.current_schema_version()}


def _daemon(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    return daemon.run(io.deps.clock, probe=io.deps.probe, idle_exit_s=daemon.IDLE_EXIT_S, tick_s=daemon.TICK_S)


def _ensure(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    return ensure_daemon(io)


def session_id(args: argparse.Namespace) -> str:
    """``--session-id``, defaulting to ``$CLAUDE_CODE_SESSION_ID``; ``USAGE`` when neither is set."""
    value = getattr(args, "session_id", None) or os.environ.get("CLAUDE_CODE_SESSION_ID")
    if not value:
        raise errors.CpError("USAGE", "no session id: pass --session-id or set CLAUDE_CODE_SESSION_ID")
    return str(value)


def read_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        raise errors.CpError("USAGE", f"cannot read {path}: {exc}") from exc


def _token(p: argparse.ArgumentParser) -> None:
    p.add_argument("--token", required=True)


def _sid(p: argparse.ArgumentParser) -> None:
    p.add_argument("--session-id")


def _conf_run_open(p: argparse.ArgumentParser) -> None:
    p.add_argument("--name", required=True)
    p.add_argument("--title", required=True)
    _sid(p)
    p.add_argument("--session-name")
    p.add_argument("--permission-mode")
    p.add_argument("--full-grant", action="store_true")


def _run_open(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    sid = session_id(args)
    with connection(io, "write") as conn:
        assert conn is not None
        result = runs.open_run(
            conn,
            io.deps.clock,
            name=args.name,
            title=args.title,
            session_id=sid,
            session_name=args.session_name,
            permission_mode=args.permission_mode,
            full_grant=args.full_grant,
        )
    return _daemon_status(io, result)


def _conf_run_claim(p: argparse.ArgumentParser) -> None:
    p.add_argument("run")
    _sid(p)
    p.add_argument("--session-name")
    p.add_argument("--permission-mode")
    p.add_argument("--full-grant", action="store_true")
    p.add_argument("--takeover", action="store_true")


def _run_claim(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    sid = session_id(args)
    with connection(io, "write") as conn:
        assert conn is not None
        result = runs.claim(
            conn,
            io.deps.clock,
            io.deps.claude,
            run_ref=args.run,
            session_id=sid,
            takeover=args.takeover,
            session_name=args.session_name,
            permission_mode=args.permission_mode,
            full_grant=args.full_grant,
        )
    return _daemon_status(io, result)


def _conf_run_bind(p: argparse.ArgumentParser) -> None:
    p.add_argument("run")
    _token(p)


def _run_bind_name(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    with connection(io, "write") as conn:
        assert conn is not None
        return runs.bind_name(conn, io.deps.claude, token=args.token, run_ref=args.run)


def _conf_run_set_mode(p: argparse.ArgumentParser) -> None:
    _token(p)
    p.add_argument("--permission-mode", required=True)
    p.add_argument("--full-grant", action=argparse.BooleanOptionalAction, default=None)


def _run_set_mode(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    with connection(io, "write") as conn:
        assert conn is not None
        return runs.set_mode(conn, token=args.token, permission_mode=args.permission_mode, full_grant=args.full_grant)


def _conf_run_set_plan(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", required=True)
    p.add_argument("--file", required=True)
    _token(p)


def _run_set_plan(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    plan = read_file(args.file)
    with connection(io, "write") as conn:
        assert conn is not None
        return runs.set_plan(conn, token=args.token, run_ref=args.run, plan=plan)


def _run_close(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    with connection(io, "write") as conn:
        assert conn is not None
        return runs.close(conn, io.deps.clock, token=args.token)


def _session_status(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    sid = session_id(args)
    with connection(io, "read") as conn:
        return {"session_id": sid, **runs.session_status(conn, sid)}


def _conf_msg_post(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", required=True)
    p.add_argument("--to", required=True)
    p.add_argument("--kind", required=True)
    p.add_argument("--payload-file", required=True)
    p.add_argument("--dedupe-key")
    _token(p)


def _msg_post(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    payload = messages.parse_payload(read_file(args.payload_file))
    with connection(io, "write") as conn:
        assert conn is not None
        return messages.post(
            conn,
            io.deps.clock,
            token=args.token,
            run_ref=args.run,
            to=args.to,
            kind=args.kind,
            payload=payload,
            dedupe_key=args.dedupe_key,
        )


def _conf_msg_pop(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", required=True)
    p.add_argument("--as", dest="recipient", required=True)
    p.add_argument("--visibility", type=POSITIVE, default=None)
    _token(p)


def _msg_pop(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    visibility = messages.VISIBILITY_S if args.visibility is None else args.visibility
    with connection(io, "write") as conn:
        assert conn is not None
        return messages.pop(
            conn,
            io.deps.clock,
            token=args.token,
            run_ref=args.run,
            recipient=args.recipient,
            visibility=visibility,
            max_attempts=messages.MAX_ATTEMPTS,
        )


def _conf_msg_ack(p: argparse.ArgumentParser) -> None:
    p.add_argument("id", type=int)
    p.add_argument("--receipt", required=True)
    _token(p)


def _msg_ack(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    with connection(io, "write") as conn:
        assert conn is not None
        return messages.ack(conn, io.deps.clock, token=args.token, msg_id=args.id, receipt=args.receipt)


def _conf_wait(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", required=True)
    _token(p)
    p.add_argument("--timeout", type=POSITIVE, default=None)
    p.add_argument("--poll", type=POLL, default=None)


def _wait(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    if args.timeout is not None and args.poll is not None and args.poll > args.timeout:
        raise errors.CpError("USAGE", "--poll must not exceed --timeout")
    path = paths.select_db()
    with connection(io, "write") as conn:
        assert conn is not None
        return wait.wait(
            conn,
            io.deps.clock,
            io.deps.probe,
            lambda: ensure_daemon(io, path),
            token=args.token,
            run_ref=args.run,
            db_path=path,
            timeout=wait.WAIT_TIMEOUT_S if args.timeout is None else args.timeout,
            poll=wait.POLL_S if args.poll is None else args.poll,
        )


def _write(io: Io, fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
    with connection(io, "write") as conn:
        assert conn is not None
        return fn(conn, *args, **kwargs)


def _conf_task_add(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run")
    p.add_argument("--title", required=True)
    p.add_argument("--kind", required=True, choices=tasks.KINDS)
    p.add_argument("--target")
    p.add_argument("--parent")
    p.add_argument("--issue", type=int)
    p.add_argument("--lane")
    p.add_argument("--deliverable", action="store_true")
    p.add_argument("--item-of")
    _token(p)


def _task_add(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    return _write(
        io,
        tasks.add,
        io.deps.clock,
        token=args.token,
        run_ref=args.run,
        title=args.title,
        kind=args.kind,
        target=args.target,
        parent=args.parent,
        issue=args.issue,
        lane=args.lane,
        deliverable=args.deliverable,
        item_of=args.item_of,
    )


def _conf_task_set(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True)
    p.add_argument("--issue", type=int, dest="issue_number")
    p.add_argument("--worktree")
    p.add_argument("--branch")
    p.add_argument("--pr-number", type=int)
    p.add_argument("--pr-url")
    p.add_argument("--ci-state")
    p.add_argument("--ci-sha")
    _token(p)


def _task_set(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    names = ("issue_number", "worktree", "branch", "pr_number", "pr_url", "ci_state", "ci_sha")
    fields = {n: getattr(args, n) for n in names if getattr(args, n) is not None}
    return _write(io, tasks.set_fields, io.deps.clock, token=args.token, task_id=args.task, fields=fields)


def _conf_task_state(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True)
    p.add_argument("--to", required=True)
    p.add_argument("--note")
    _token(p)


def _task_state(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    return _write(io, tasks.set_state, io.deps.clock, token=args.token, task_id=args.task, to=args.to, note=args.note)


def _conf_task_dep(p: argparse.ArgumentParser) -> None:
    p.add_argument("action", choices=("add", "remove"))
    p.add_argument("--task", required=True)
    p.add_argument("--on", required=True)
    _token(p)


def _task_dep(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    return _write(io, tasks.edit_dep, token=args.token, action=args.action, task_id=args.task, on=args.on)


def _conf_membership(p: argparse.ArgumentParser) -> None:
    p.add_argument("action", choices=("add", "remove"))
    p.add_argument("--run", required=True)
    p.add_argument("--task", required=True)
    _token(p)


def _membership(table: str) -> Callable[[argparse.Namespace, Io], dict[str, Any]]:
    def handler(args: argparse.Namespace, io: Io) -> dict[str, Any]:
        return _write(
            io,
            tasks.edit_membership,
            token=args.token,
            table=table,
            action=args.action,
            run_ref=args.run,
            task_id=args.task,
        )

    return handler


def _conf_task_token(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True)
    _token(p)


def _conf_criteria_set(p: argparse.ArgumentParser) -> None:
    _conf_task_token(p)
    p.add_argument("--file", required=True)


def _criteria_set(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    lines = read_file(args.file).splitlines()
    return _write(io, tasks.criteria_set, token=args.token, task_id=args.task, lines=lines)


def _criteria_validate(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    return _write(io, tasks.criteria_validate, io.deps.clock, token=args.token, task_id=args.task)


def _conf_criteria_state(p: argparse.ArgumentParser) -> None:
    _conf_task_token(p)
    p.add_argument("--idx", type=int, required=True)
    p.add_argument("--to", required=True)


def _criteria_state(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    return _write(io, tasks.criteria_state, token=args.token, task_id=args.task, idx=args.idx, to=args.to)


def _conf_question_open(p: argparse.ArgumentParser) -> None:
    _conf_task_token(p)
    p.add_argument("--text-file", required=True)
    p.add_argument("--blocking", action="store_true")


def _question_open(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    text = read_file(args.text_file)
    return _write(
        io,
        questions.open_question,
        io.deps.clock,
        token=args.token,
        task_id=args.task,
        text=text,
        blocking=args.blocking,
    )


def _conf_question_ask(p: argparse.ArgumentParser) -> None:
    p.add_argument("question")
    _token(p)


def _question_ask(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    return _write(io, questions.ask, token=args.token, question_id=args.question)


def _conf_question_answer(p: argparse.ArgumentParser) -> None:
    _conf_question_ask(p)
    p.add_argument("--answer-file", required=True)
    p.add_argument("--reason", required=True)


def _question_answer(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    text = read_file(args.answer_file)
    return _write(
        io,
        questions.answer,
        io.deps.clock,
        token=args.token,
        question_id=args.question,
        answer_text=text,
        reason=args.reason,
    )


def _conf_decision_add(p: argparse.ArgumentParser) -> None:
    p.add_argument("--text", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--task")
    _token(p)


def _decision_add(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    return _write(
        io,
        decisions.add,
        io.deps.clock,
        token=args.token,
        text=args.text,
        reason=args.reason,
        source=args.source,
        task_id=args.task,
    )


def _conf_report_post(p: argparse.ArgumentParser) -> None:
    _conf_task_token(p)
    p.add_argument("--phase", required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--status", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--fields-file", required=True)


def _report_post(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    fields = messages.parse_payload(read_file(args.fields_file))
    return _write(
        io,
        reports.post_report,
        io.deps.clock,
        token=args.token,
        task_id=args.task,
        phase=args.phase,
        round_=args.round,
        status=args.status,
        summary=args.summary,
        fields=fields,
    )


def _conf_digest_put(p: argparse.ArgumentParser) -> None:
    _conf_task_token(p)
    p.add_argument("--file", required=True)


def _digest_put(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    body = read_file(args.file)
    return _write(io, reports.put_digest, io.deps.clock, token=args.token, task_id=args.task, body=body)


def _node_stop(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    return _write(io, nodes.stop, io.deps.clock, token=args.token, task_id=args.task)


def _node_take_over(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    return _write(io, nodes.take_over, io.deps.clock, token=args.token, task_id=args.task)


def _conf_node_hand_back(p: argparse.ArgumentParser) -> None:
    _conf_task_token(p)
    p.add_argument("--summary-file", required=True)


def _node_hand_back(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    summary = read_file(args.summary_file)
    return _write(io, nodes.hand_back, io.deps.clock, token=args.token, task_id=args.task, summary=summary)


def _conf_lock(p: argparse.ArgumentParser) -> None:
    p.add_argument("--name", required=True)
    _token(p)
    _sid(p)


def _conf_lock_acquire(p: argparse.ArgumentParser) -> None:
    _conf_lock(p)
    p.add_argument("--purpose", required=True)
    p.add_argument("--timeout", type=NON_NEGATIVE, default=0.0)


def _lock_acquire(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    sid = session_id(args)
    d = io.deps
    return _write(
        io,
        locks.acquire_session,
        d.clock,
        d.probe,
        d.claude,
        token=args.token,
        name=args.name,
        purpose=args.purpose,
        session_id=sid,
        timeout=args.timeout,
    )


def _lock_release(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    sid = session_id(args)
    return _write(io, locks.release_session, token=args.token, name=args.name, session_id=sid)


def _hook_stop(args: argparse.Namespace, io: Io) -> Raw:
    return Raw(hooks.hook_stop(io.stdin.read(), io.deps.clock, io.deps.probe))


def _hook_pre_tool_use(args: argparse.Namespace, io: Io) -> Raw:
    return Raw(hooks.hook_pre_tool_use(io.stdin.read(), io.deps.clock))


def _conf_hook_pre_push(p: argparse.ArgumentParser) -> None:
    p.add_argument("git_args", nargs="*")


def _hook_pre_push(args: argparse.Namespace, io: Io) -> Raw:
    code, message = hooks.hook_pre_push(io.stdin.read(), io.deps.clock, io.deps.runner)
    return Raw(message, code)


def _conf_hooks_settings(p: argparse.ArgumentParser) -> None:
    p.add_argument("--role", required=True, choices=("node", "orchestrator"))


def _hooks_settings(args: argparse.Namespace, io: Io) -> Raw:
    return Raw(json.dumps(hooks.hooks_settings(args.role), sort_keys=True))


def _conf_snapshot(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run")


def _snapshot(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    with connection(io, "read") as conn:
        run_id = None
        if args.run is not None and conn is not None:
            run_id = runs.resolve(conn, args.run)["id"]
        return snapshot.snapshot(conn, io.deps.clock, run_id)


def _conf_task_show(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True)


def _task_show(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    with connection(io, "read") as conn:
        return snapshot.task_show(conn, args.task)


def _conf_export(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True)
    p.add_argument("--out-worktree", required=True)


def _export_summary(args: argparse.Namespace, io: Io) -> dict[str, Any]:
    with connection(io, "read") as conn:
        if conn is None:
            raise errors.CpError("NOT_FOUND", f"unknown task {args.task} (no DB yet)")
        return {"path": str(export.write_summary(conn, args.task, args.out_worktree))}


COMMANDS: dict[str, Command] = {
    c.name: c
    for c in (
        Command("version", "exempt", _version, help="print the package and schema version (no DB)"),
        Command("daemon", "exempt", _daemon, help="run the daemon (migrates the DB, heartbeats, ticks)"),
        Command("ensure", "exempt", _ensure, help="start or restart the daemon if needed"),
        Command("hook stop", "exempt", _hook_stop, help="the orchestrator's Stop hook (stdin: hook JSON)"),
        Command("hook pre-tool-use", "exempt", _hook_pre_tool_use, help="the PreToolUse hook (stdin: hook JSON)"),
        Command("hook pre-push", "exempt", _hook_pre_push, _conf_hook_pre_push, help="git's pre-push hook"),
        Command(
            "hooks-settings", "exempt", _hooks_settings, _conf_hooks_settings, help="print a hooks settings fragment"
        ),
        Command("run open", "write", _run_open, _conf_run_open, help="open a run; returns its run token"),
        Command(
            "run claim", "write", _run_claim, _conf_run_claim, help="claim a run's lease (re-claim, take, take over)"
        ),
        Command("run bind-name", "write", _run_bind_name, _conf_run_bind, help="bind the run name after a takeover"),
        Command("run set-mode", "write", _run_set_mode, _conf_run_set_mode, help="record the permission mode"),
        Command("run set-plan", "write", _run_set_plan, _conf_run_set_plan, help="replace the run's global plan"),
        Command("run close", "write", _run_close, _token, help="close the run"),
        Command("session status", "read", _session_status, _sid, help="this session's role and state"),
        Command("snapshot", "read", _snapshot, _conf_snapshot, help="the board-complete read-only view"),
        Command("task show", "read", _task_show, _conf_task_show, help="one task in full, digest included"),
        Command("export-summary", "read", _export_summary, _conf_export, help="write the task summary into a worktree"),
        Command("msg post", "write", _msg_post, _conf_msg_post, help="post a message to a queue"),
        Command("msg pop", "write", _msg_pop, _conf_msg_pop, help="pop the next visible message (with a receipt)"),
        Command("msg ack", "write", _msg_ack, _conf_msg_ack, help="acknowledge a popped message"),
        Command("wait", "write", _wait, _conf_wait, help="block until the orchestrator's queue is not empty"),
        Command("task add", "write", _task_add, _conf_task_add, help="add a task (state `proposed`)"),
        Command("task set", "write", _task_set, _conf_task_set, help="update a task's fields"),
        Command("task state", "write", _task_state, _conf_task_state, help="apply a state transition"),
        Command("task dep", "write", _task_dep, _conf_task_dep, help="add or remove a dependency"),
        Command("task root", "write", _membership("run_roots"), _conf_membership, help="edit the run's roots"),
        Command("task work-list", "write", _membership("work_list"), _conf_membership, help="edit the work list"),
        Command("criteria set", "write", _criteria_set, _conf_criteria_set, help="replace the acceptance criteria"),
        Command("criteria validate", "write", _criteria_validate, _conf_task_token, help="the maintainer's sign-off"),
        Command("criteria state", "write", _criteria_state, _conf_criteria_state, help="update one criterion"),
        Command("question open", "write", _question_open, _conf_question_open, help="open a question"),
        Command("question ask", "write", _question_ask, _conf_question_ask, help="mark a question asked"),
        Command("question answer", "write", _question_answer, _conf_question_answer, help="answer a question"),
        Command("decision add", "write", _decision_add, _conf_decision_add, help="record a decision"),
        Command("report post", "write", _report_post, _conf_report_post, help="post a node report"),
        Command("digest put", "write", _digest_put, _conf_digest_put, help="replace a task's digest"),
        Command("node stop", "write", _node_stop, _conf_task_token, help="stop the task's node"),
        Command(
            "node take-over", "write", _node_take_over, _conf_task_token, help="the maintainer takes the node over"
        ),
        Command("node hand-back", "write", _node_hand_back, _conf_node_hand_back, help="hand the node back"),
        Command("lock acquire", "write", _lock_acquire, _conf_lock_acquire, help="hold integration:<branch> (session)"),
        Command("lock release", "write", _lock_release, _conf_lock, help="release a session-held lock"),
    )
}


def _conf_tool(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--args-file")
    _token(p)


def _tool_handler(name: str) -> Callable[[argparse.Namespace, Io], dict[str, Any]]:
    def handler(args: argparse.Namespace, io: Io) -> dict[str, Any]:
        tool_args = messages.parse_payload(read_file(args.args_file)) if args.args_file else {}
        if not isinstance(tool_args, dict):
            raise errors.CpError("USAGE", "--args-file must hold a JSON object")
        path = paths.select_db()
        with connection(io, "write"):
            pass  # the entry schema check (waits for a pending self-migration)
        d = io.deps
        ctx = tools.Ctx(
            runner=d.runner,
            conn_factory=lambda: db.connect(path),
            clock=d.clock,
            probe=d.probe,
            claude=d.claude,
            main=paths.main(),
        )
        return tools.invoke(name, key=args.key, task_id=args.task, args=tool_args, token=args.token, actor="", ctx=ctx)

    return handler


def all_commands() -> dict[str, Command]:
    """``COMMANDS`` plus one ``tool <name>`` entry per registered tool."""
    table = dict(COMMANDS)
    for name in sorted(tools.REGISTRY):
        table[f"tool {name}"] = Command(
            f"tool {name}", "write", _tool_handler(name), _conf_tool, help=f"run the {name} tool"
        )
    return table


# --------------------------------------------------------------------------- parsing


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="cp.py", description="Quiet Solar Control Plane")
    groups: dict[str, argparse._SubParsersAction[Any]] = {}
    top = parser.add_subparsers(dest="_cmd0", required=True, parser_class=_Parser)
    table = all_commands()
    for name in sorted(table):
        cmd = table[name]
        words = name.split()
        sub_action = top
        prefix = ""
        for depth, word in enumerate(words[:-1]):
            prefix = f"{prefix} {word}".strip()
            if prefix not in groups:
                group_parser = sub_action.add_parser(word, help=f"{prefix} commands")
                groups[prefix] = group_parser.add_subparsers(
                    dest=f"_cmd{depth + 1}", required=True, parser_class=_Parser
                )
            sub_action = groups[prefix]
        leaf = sub_action.add_parser(words[-1], help=cmd.help)
        cmd.configure(leaf)
        leaf.set_defaults(_command=name)
    return parser


def _emit(out: TextIO, payload: dict[str, Any]) -> None:
    out.write(json.dumps(payload, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None, *, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    items.register_item_tools()  # cheap, and keeps a reset registry whole before build_parser()
    io = Io(stdin=stdin or sys.stdin, stdout=stdout or sys.stdout, deps=make_deps())
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(argv)
        cmd = all_commands()[args._command]
        if cmd.name.startswith("tool "):
            procsetup.get().become_group_leader()
        result = cmd.handler(args, io)
    except errors.CpError as exc:
        _emit(io.stdout, exc.payload())
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001 — every unexpected error is INTERNAL
        _emit(io.stdout, {"ok": False, "error": "INTERNAL", "detail": f"{type(exc).__name__}: {exc}"})
        return 1
    if isinstance(result, Raw):
        if result.text:
            io.stdout.write(result.text if result.text.endswith("\n") else result.text + "\n")
        return result.code
    _emit(io.stdout, {"ok": True, **result})
    return 0
