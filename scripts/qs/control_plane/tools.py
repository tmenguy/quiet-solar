"""The tools layer (§9): idempotent, recorded, fenced effects.

Every effect a session asks for goes through ``run_recorded``: the token is
verified, a finished call under the key is replayed at once (before its steps
are even built, so a deleted argument file or a cleared task column cannot
break a replay), else the call is claimed in ``tool_calls`` under PK
``(tool, key)`` **before** any effect, a probe tells which steps the real
world already holds, then each missing step runs under the tool's locks /
cap, with the token and lock ownership re-checked before every step. The
``args_hash`` covers the arguments and the content of every ``*_file``
argument: the same key with an edited file is a ``CONFLICT``.

The frozen API for #400 and later children: ``StepCtx``, ``Step``,
``ToolSpec``, ``register`` and ``invoke``. ``argv_step`` is a helper and may
evolve.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import clock as clock_mod
from . import db, errors, faults, hooks, liveness, locks, merge_policy, nodes, paths, runner, tasks, tokens

TaskRow = sqlite3.Row
TOKEN_ENV = "QS_CP_TOKEN"
TAIL_LINES = 60
IDENTIFY_POLL_S = 2.0
PHASES = ("/create-plan", "/diagnose-task", "/decompose-epic")
RELEASED_BEFORE_EFFECT = frozenset({"BUSY", "POLICY_REFUSED", "STALE_TOKEN", "STOPPED"})
NON_TERMINAL_STATES = frozenset(set(tasks.TRANSITIONS) - tasks.TERMINAL)


class StepFailed(Exception):
    """A step's own failure: the call is recorded ``failed`` and the key is spent."""

    def __init__(self, detail: str, output: Any = None, exit_code: int = 1) -> None:
        super().__init__(detail)
        self.detail = detail
        self.output = output
        self.exit_code = exit_code


@dataclass(frozen=True)
class Ctx:
    """The dependencies of one tool invocation."""

    runner: runner.Runner
    conn_factory: Callable[[], sqlite3.Connection]
    clock: clock_mod.Clock
    probe: liveness.ProcessProbe
    claude: liveness.ClaudeCli
    main: Path


class _StepRunner(runner.Runner):
    """Adds ``QS_CP_TOKEN`` for ``inject_token`` steps only (and removes it otherwise); detaches ``detach`` steps."""

    def __init__(self, base: runner.Runner, token: str, *, inject: bool = False, detach: bool = False) -> None:
        self.base = base
        self.token = token
        self.inject = inject
        self.detach = detach

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | str | None = None,
        env_extra: Mapping[str, str] | None = None,
        env_remove: Sequence[str] = (),
        timeout: float | None = None,
        detach: bool = False,
    ) -> runner.RunResult:
        extra = dict(env_extra or {})
        remove = list(env_remove)
        if self.inject:
            extra[TOKEN_ENV] = self.token
        else:
            remove.append(TOKEN_ENV)
        return self.base.run(
            argv, cwd=cwd, env_extra=extra, env_remove=tuple(remove), timeout=timeout, detach=detach or self.detach
        )


# --------------------------------------------------------------------------- the frozen API (§9.3)


@dataclass(frozen=True)
class StepCtx:
    """What a step, a probe, a guard and ``on_success`` see.

    Beyond the frozen fields, ``tool``, ``key`` and ``clock`` are provided
    for markers and timeouts.
    """

    task: TaskRow
    args: Mapping[str, Any]
    outputs: Mapping[str, Any]
    runner: runner.Runner
    claude: liveness.ClaudeCli
    main: Path
    tool: str = ""
    key: str = ""
    clock: clock_mod.Clock = field(default_factory=clock_mod.SystemClock)
    _call: Any = field(default=None, repr=False, compare=False)
    _step: str | None = field(default=None, repr=False, compare=False)

    def write(self) -> AbstractContextManager[sqlite3.Connection]:
        """A ``db.write()`` transaction (DB-only steps, guards, probes)."""
        return db.write(self._call.conn)

    def record_output(self, value: Any) -> None:
        """Inside ``write()``: persist this step's output in the same transaction as its effect."""
        self._call.record_output_locked(self._step, value)


@dataclass(frozen=True)
class Step:
    name: str
    run: Callable[[StepCtx], Any]
    inject_token: bool = False
    detach: bool = False


def argv_step(
    name: str,
    argv: Callable[[StepCtx], list[str]],
    cwd: Callable[[StepCtx], Path],
    *,
    check: Callable[[runner.RunResult], Any] | None = None,
    inject_token: bool = False,
    detach: bool = False,
    timeout: float | None = None,
) -> Step:
    """A step that runs one command (helper — not part of the frozen API)."""

    def run(ctx: StepCtx) -> Any:
        res = ctx.runner.run(argv(ctx), cwd=cwd(ctx), timeout=timeout)
        if check is not None:
            return check(res)
        if not res.ok:
            raise StepFailed(
                f"{name} exited {res.returncode}", {"exit_code": res.returncode, "tail": tail(res)}, res.returncode
            )
        return {"exit_code": 0, "tail": tail(res)}

    return Step(name, run, inject_token=inject_token, detach=detach)


def _no_locks(task: TaskRow, args: Mapping[str, Any]) -> Sequence[str]:
    return ()


@dataclass(frozen=True)
class ToolSpec:
    name: str
    steps: Callable[[TaskRow, Mapping[str, Any]], Sequence[Step]]
    locks: Callable[[TaskRow, Mapping[str, Any]], Sequence[str]] = _no_locks
    cap: str | None = None
    probe: Callable[[StepCtx], dict[str, Any | None]] | None = None
    guard: Callable[[StepCtx], None] | None = None
    on_success: Callable[[sqlite3.Connection, StepCtx], dict[str, Any]] | None = None
    refuse_when_stopped: bool = True
    token_kinds: frozenset[str] = frozenset({"run"})


REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> None:
    if spec.name in REGISTRY:
        raise errors.CpError("CONFLICT", f"a tool named {spec.name!r} is registered")
    REGISTRY[spec.name] = spec


# --------------------------------------------------------------------------- run_recorded (§9.2)


def tail(res: runner.RunResult) -> list[str]:
    return (res.stdout.splitlines() + res.stderr.splitlines())[-TAIL_LINES:]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


_CALL_IDENT = ("state", "holder_pid", "holder_pid_start", "holder_pgid", "outputs")


def _ident(row: sqlite3.Row | None) -> tuple[Any, ...] | None:
    return None if row is None else tuple(row[k] for k in _CALL_IDENT)


def _files_digest(args: Mapping[str, Any]) -> str | None:
    """sha256 over the content of every ``*_file`` argument; ``None`` when one cannot be read now."""
    digests: dict[str, str] = {}
    for name, value in args.items():
        if name.endswith("_file") and isinstance(value, str):
            try:
                digests[name] = hashlib.sha256(Path(value).read_bytes()).hexdigest()
            except OSError:
                return None
    return hashlib.sha256(_canonical(digests).encode()).hexdigest()


def _owned(cur: sqlite3.Cursor) -> None:
    """A compare-and-set on this call's claim matched nothing: another process took the key over."""
    if cur.rowcount != 1:
        raise errors.CpError("STALE_TOKEN", "this call's claim was taken over by another process")


class _Call:
    def __init__(
        self, spec: ToolSpec, ctx: Ctx, *, key: str, task_id: str, args: Mapping[str, Any], token: str, actor: str
    ) -> None:
        self.spec = spec
        self.ctx = ctx
        self.key = key
        self.task_id = task_id
        self.args = dict(args)
        self.token = token
        self.actor = actor
        self.conn = ctx.conn_factory()
        self.me = ctx.probe.me()
        self.outputs: dict[str, Any] = {}
        self.recorded: set[str] = set()
        self.who: tokens.Principal | None = None
        self.args_digest = hashlib.sha256(_canonical(self.args).encode()).hexdigest()
        self.files_digest = _files_digest(self.args)

    # -- helpers
    def task(self) -> TaskRow:
        return tasks.get(self.conn, self.task_id)

    def step_ctx(self, task: TaskRow, step: Step | None = None) -> StepCtx:
        run = _StepRunner(
            self.ctx.runner, self.token, inject=bool(step and step.inject_token), detach=bool(step and step.detach)
        )
        return StepCtx(
            task=task,
            args=self.args,
            outputs=dict(self.outputs),
            runner=run,
            claude=self.ctx.claude,
            main=self.ctx.main,
            tool=self.spec.name,
            key=self.key,
            clock=self.ctx.clock,
            _call=self,
            _step=None if step is None else step.name,
        )

    def _where(self) -> tuple[str, tuple[Any, ...]]:
        return "tool = ? AND key = ? AND holder_pid = ? AND holder_pgid IS ?", (
            self.spec.name,
            self.key,
            self.me.pid,
            self.me.pgid,
        )

    def record_output_locked(self, step: str | None, value: Any) -> None:
        assert step is not None and self.conn.in_transaction, "record_output() runs inside StepCtx.write()"
        self.outputs[step] = value
        self.recorded.add(step)
        where, params = self._where()
        _owned(
            self.conn.execute(f"UPDATE tool_calls SET outputs = ? WHERE {where}", (_canonical(self.outputs), *params))
        )

    def persist_outputs(self) -> None:
        where, params = self._where()
        with db.write(self.conn):
            _owned(
                self.conn.execute(
                    f"UPDATE tool_calls SET outputs = ? WHERE {where}", (_canonical(self.outputs), *params)
                )
            )

    def require(self) -> tokens.Principal:
        self.who = tokens.require(
            self.conn,
            self.token,
            kinds=self.spec.token_kinds,
            task_id=self.task_id,
            allow_stopped=not self.spec.refuse_when_stopped,
        )
        self.actor = self.actor or self.who.actor  # locks and cap slots record who holds them
        return self.who

    # -- 1b. replay, 2. claim
    def _check_same(self, row: sqlite3.Row) -> None:
        stored_args, _, stored_files = str(row["args_hash"]).partition(":")
        if stored_args != self.args_digest or row["task_id"] != self.task_id:
            raise errors.CpError("CONFLICT", f"key {self.key!r} was used with other arguments")
        if self.files_digest is not None and stored_files not in ("-", self.files_digest):
            raise errors.CpError("CONFLICT", f"key {self.key!r} was used with other argument file content")

    def _finished(self, row: sqlite3.Row) -> dict[str, Any]:
        """The recorded outcome of a finished call: its response, or its ``TOOL_FAILED``."""
        if row["state"] == "succeeded":
            return self.response(json.loads(row["result"] or "{}"), replayed=True)
        raise errors.CpError(
            "TOOL_FAILED",
            "this key already failed; retry with a new key",
            result=json.loads(row["result"] or "{}"),
            replayed=True,
        )

    def replay(self) -> dict[str, Any] | None:
        """Inside a transaction: a finished call under this key → its outcome; else ``None``."""
        row = self.conn.execute(
            "SELECT * FROM tool_calls WHERE tool = ? AND key = ?", (self.spec.name, self.key)
        ).fetchone()
        if row is None or row["state"] == "started":
            return None
        self._check_same(row)
        return self._finished(row)

    def claim(self) -> dict[str, Any] | None:
        args_hash = f"{self.args_digest}:{self.files_digest or '-'}"
        while True:
            row = self.conn.execute(
                "SELECT * FROM tool_calls WHERE tool = ? AND key = ?", (self.spec.name, self.key)
            ).fetchone()
            alive = (
                row is not None
                and row["state"] == "started"
                and self.ctx.probe.holder_alive(row["holder_pid"], row["holder_pid_start"], row["holder_pgid"])
            )
            with db.write(self.conn):
                who = self.require()
                current = self.conn.execute(
                    "SELECT * FROM tool_calls WHERE tool = ? AND key = ?", (self.spec.name, self.key)
                ).fetchone()
                if _ident(current) != _ident(row):
                    continue
                now = db.now(self.ctx.clock)
                if current is None:
                    self.conn.execute(
                        "INSERT INTO tool_calls (tool, key, args_hash, run_id, task_id, actor, args, state, holder_pid,"
                        " holder_pid_start, holder_pgid, outputs, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'started',"
                        " ?, ?, ?, '{}', ?)",
                        (
                            self.spec.name,
                            self.key,
                            args_hash,
                            who.run_id,
                            self.task_id,
                            self.actor or who.actor,
                            _canonical(self.args),
                            self.me.pid,
                            self.me.pid_start,
                            self.me.pgid,
                            now,
                        ),
                    )
                    return None
                self._check_same(current)
                if current["state"] != "started":
                    return self._finished(current)
                if alive:
                    raise errors.CpError("BUSY", f"an in-flight call holds {self.spec.name}/{self.key}")
                self.conn.execute(
                    "UPDATE tool_calls SET holder_pid = ?, holder_pid_start = ?, holder_pgid = ?"
                    " WHERE tool = ? AND key = ? AND holder_pid IS ? AND holder_pgid IS ?",
                    (
                        self.me.pid,
                        self.me.pid_start,
                        self.me.pgid,
                        self.spec.name,
                        self.key,
                        current["holder_pid"],
                        current["holder_pgid"],
                    ),
                )
                self.outputs = json.loads(current["outputs"])
                return None

    # -- 3. probe and guard, 4. run, 5. record
    def guard(self, task: TaskRow) -> None:
        if self.spec.guard is not None:
            self.spec.guard(self.step_ctx(task))

    def run(self, steps: Sequence[Step], lock_names: Sequence[str]) -> dict[str, Any]:
        task = self.task()
        if self.spec.probe is not None:
            for name, value in self.spec.probe(self.step_ctx(task)).items():
                if value is None:
                    self.outputs.pop(name, None)
                else:
                    self.outputs[name] = value
            self.persist_outputs()
        self.guard(task)
        with locks.hold(
            lock_names,
            conn_factory=self.ctx.conn_factory,
            token=self.token,
            actor=self.actor,
            purpose=f"tool {self.spec.name}/{self.key}",
            clock=self.ctx.clock,
            probe=self.ctx.probe,
            claude=self.ctx.claude,
            cap=self.spec.cap,
            holder=self.me,
            allow_stopped=not self.spec.refuse_when_stopped,
            timeout=locks.LOCK_WAIT_S,
            cap_timeout=locks.GATE_WAIT_S,
        ) as held:
            for step in steps:
                if step.name in self.outputs:
                    continue
                locks.recheck(self.conn, held)
                task = self.task()
                self.guard(task)
                faults.hit(f"{self.spec.name}.before_effect")
                out = step.run(self.step_ctx(task, step))
                if step.name in self.recorded:
                    out = self.outputs[step.name]
                faults.hit(f"{self.spec.name}.after_effect")
                if step.name not in self.recorded:
                    self.outputs[step.name] = out
                    self.persist_outputs()
            return self.succeed()

    def succeed(self) -> dict[str, Any]:
        where, params = self._where()
        with db.write(self.conn):
            self.require()
            sctx = self.step_ctx(self.task())
            result = self.spec.on_success(self.conn, sctx) if self.spec.on_success is not None else {}
            _owned(
                self.conn.execute(
                    f"UPDATE tool_calls SET state = 'succeeded', result = ?, exit_code = 0, finished_at = ?"
                    f" WHERE {where}",
                    (_canonical(result), db.now(self.ctx.clock), *params),
                )
            )
        return self.response(result, replayed=False)

    def fail(self, error: str, detail: str, output: Any, exit_code: int) -> errors.CpError:
        result = {"error": error, "detail": detail, "output": output}
        where, params = self._where()
        with db.write(self.conn):
            _owned(
                self.conn.execute(
                    f"UPDATE tool_calls SET state = 'failed', result = ?, exit_code = ?, finished_at = ? WHERE {where}",
                    (_canonical(result), exit_code, db.now(self.ctx.clock), *params),
                )
            )
        return errors.CpError("TOOL_FAILED", f"{self.spec.name}: {detail}", result=result)

    def release_claim(self) -> None:
        where, params = self._where()
        self.conn.execute(f"DELETE FROM tool_calls WHERE {where} AND state = 'started'", params)

    def response(self, result: dict[str, Any], *, replayed: bool) -> dict[str, Any]:
        out: dict[str, Any] = {"tool": self.spec.name, "key": self.key, "replayed": replayed, "result": result}
        extra = _RESPONSE_EXTRAS.get(self.spec.name)
        if extra is not None and self.who is not None:
            out.update(extra(self.conn, result, self.who))
        return out


def run_recorded(
    spec: ToolSpec,
    *,
    key: str,
    task: str,
    args: Mapping[str, Any],
    token: str,
    actor: str,
    ctx: Ctx,
) -> dict[str, Any]:
    """Run ``spec`` for ``task`` under the idempotency ``key`` (§9.2)."""
    if not key or len(key) > 200:
        raise errors.CpError("USAGE", "--key must be 1 to 200 characters")
    call = _Call(spec, ctx, key=key, task_id=task, args=args, token=token, actor=actor)
    try:
        with db.write(call.conn):
            call.require()
            task_row = call.task()
            replay = call.replay()  # before the steps: a replay needs neither the files nor the task columns
        if replay is not None:
            return replay
        steps = list(spec.steps(task_row, call.args))
        lock_names = list(spec.locks(task_row, call.args))
        locks.assert_sorted(lock_names)
        replay = call.claim()
        if replay is not None:
            return replay
        try:
            return call.run(steps, lock_names)
        except StepFailed as exc:
            raise call.fail("TOOL_FAILED", exc.detail, exc.output, exc.exit_code) from exc
        except errors.CpError as exc:
            if exc.code in RELEASED_BEFORE_EFFECT:
                if not call.outputs:
                    call.release_claim()
                raise
            if exc.code in ("INTERNAL", "SCHEMA_TOO_NEW", "SCHEMA_PENDING"):
                raise
            raise call.fail(exc.code, exc.detail, None, 1) from exc
    finally:
        call.conn.close()


def default_ctx() -> Ctx:
    run = runner.Runner()
    path = paths.select_db()
    return Ctx(
        runner=run,
        conn_factory=lambda: db.connect(path),
        clock=clock_mod.SystemClock(),
        probe=liveness.ProcessProbe(run),
        claude=liveness.ClaudeCli(run),
        main=paths.main(),
    )


def invoke(
    name: str,
    *,
    key: str,
    task_id: str,
    args: Mapping[str, Any],
    token: str,
    actor: str,
    ctx: Ctx | None = None,
) -> dict[str, Any]:
    spec = REGISTRY.get(name)
    if spec is None:
        raise errors.CpError("NOT_FOUND", f"no tool named {name!r}")
    return run_recorded(spec, key=key, task=task_id, args=args, token=token, actor=actor, ctx=ctx or default_ctx())


# --------------------------------------------------------------------------- shared helpers for the built-ins


def task_state_guard(allowed: Callable[[str], bool]) -> Callable[[StepCtx], None]:
    """A guard on the tool's task state (re-read at every check)."""

    def guard(ctx: StepCtx) -> None:
        with ctx.write() as conn:
            state = tasks.get(conn, ctx.task["id"])["state"]
        if not allowed(state):
            raise errors.CpError("INVALID_STATE", f"task {ctx.task['id']} is {state}")

    return guard


def compose(*guards: Callable[[StepCtx], None]) -> Callable[[StepCtx], None]:
    def guard(ctx: StepCtx) -> None:
        for g in guards:
            g(ctx)

    return guard


non_terminal = task_state_guard(lambda s: s in NON_TERMINAL_STATES)


def _need(task: TaskRow, column: str) -> Any:
    value = task[column]
    if value is None:
        raise errors.CpError("INVALID_STATE", f"task {task['id']} has no {column}")
    return value


def _python(main_dir: Path) -> str:
    return str(main_dir / "venv" / "bin" / "python")


def _script(main_dir: Path, name: str) -> str:
    return str(main_dir / "scripts" / "qs" / name)


def _script_json(res: runner.RunResult, what: str) -> dict[str, Any]:
    try:
        data = json.loads(res.stdout)
    except ValueError:
        data = None
    if res.returncode != 0 or not isinstance(data, dict) or "error" in data:
        raise StepFailed(
            f"{what} failed (exit {res.returncode})",
            {"exit_code": res.returncode, "tail": tail(res), "json": data},
            res.returncode or 1,
        )
    return data


def marker(ctx: StepCtx) -> str:
    return f"<!-- qs-cp-key: {ctx.tool}/{ctx.key} -->"


def _read_arg_file(args: Mapping[str, Any], name: str) -> str:
    path = args.get(name)
    if not isinstance(path, str) or not path:
        raise errors.CpError("USAGE", f"args.{name} is required")
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise errors.CpError("USAGE", f"cannot read args.{name}: {exc}") from exc


def _gh_list_by_marker(ctx: StepCtx, argv: list[str], cwd: Path) -> dict[str, Any] | None:
    """The listed item whose body holds this call's marker; ``BUSY`` when the listing failed (unknown)."""
    res = ctx.runner.run(argv, cwd=cwd, timeout=60)
    try:
        items = json.loads(res.stdout) if res.ok else None
    except ValueError:
        items = None
    if not isinstance(items, list):
        raise errors.CpError(
            "BUSY", f"{' '.join(argv[:3])} failed (exit {res.returncode}): probe unknown, replay the same key later"
        )
    mark = marker(ctx)
    for item in items:
        if isinstance(item, dict) and mark in str(item.get("body", "")):
            return item
    return None


# --------------------------------------------------------------------------- worktree-create / worktree-cleanup


def _worktree_create_steps(task: TaskRow, args: Mapping[str, Any]) -> Sequence[Step]:
    phase = args.get("phase")
    if phase not in PHASES:
        raise errors.CpError("USAGE", f"args.phase must be one of {', '.join(PHASES)}")
    issue = _need(task, "issue_number")
    return (
        Step("pre_push", lambda ctx: hooks.install_pre_push(ctx.main / ".git", ctx.runner)),
        argv_step(
            "setup",
            lambda ctx: [
                _python(ctx.main),
                _script(ctx.main, "setup_task.py"),
                str(issue),
                "--harness",
                "claude-code",
                "--next-cmd",
                str(phase),
            ],
            lambda ctx: ctx.main,
            check=lambda res: _script_json(res, "setup_task.py"),
            inject_token=True,
            timeout=600,
        ),
    )


def _worktree_create_success(conn: sqlite3.Connection, ctx: StepCtx) -> dict[str, Any]:
    out = ctx.outputs["setup"]
    worktree = str(Path(out["worktree_path"]).resolve()) if out.get("worktree_path") else None
    fields = {"branch": out["branch"]}
    if worktree is not None:
        fields["worktree"] = worktree
    tasks.update_fields(conn, ctx.clock, ctx.task["id"], fields)
    return {"worktree": worktree, "branch": out["branch"]}


def _registered_worktrees(ctx: StepCtx) -> set[Path] | None:
    res = ctx.runner.run(["git", "-C", str(ctx.main), "worktree", "list", "--porcelain"], cwd=ctx.main, timeout=60)
    if not res.ok:
        return None
    return {
        Path(line[len("worktree ") :]).resolve() for line in res.stdout.splitlines() if line.startswith("worktree ")
    }


def _sharing_task(ctx: StepCtx, path: Path) -> str | None:
    """Another non-terminal task registered on the same worktree path, if any."""
    marks = ", ".join("?" for _ in tasks.TERMINAL)
    with ctx.write() as conn:
        rows = conn.execute(
            f"SELECT id, worktree FROM tasks WHERE id != ? AND worktree IS NOT NULL AND state NOT IN ({marks})"
            " ORDER BY rowid",
            (ctx.task["id"], *sorted(tasks.TERMINAL)),
        ).fetchall()
    return next((str(r["id"]) for r in rows if Path(r["worktree"]).resolve() == path), None)


def _worktree_cleanup_steps(task: TaskRow, args: Mapping[str, Any]) -> Sequence[Step]:
    issue = _need(task, "issue_number")

    def remove(ctx: StepCtx) -> dict[str, Any]:
        wt = ctx.task["worktree"]
        if wt is None or not Path(wt).exists():
            return {"skipped": True}
        path = Path(wt).resolve()
        if path == ctx.main.resolve():
            raise errors.CpError("POLICY_REFUSED", f"{wt} is the main checkout: never removed")
        if not (path / ".git").is_file():
            raise errors.CpError("POLICY_REFUSED", f"{wt} is not a linked worktree (its .git is not a file)")
        shared = _sharing_task(ctx, path)
        if shared is not None:
            return {"skipped": True, "shared_with": shared}  # only this task's column is cleared
        res = ctx.runner.run(
            [
                _python(ctx.main),
                _script(ctx.main, "cleanup_worktree.py"),
                "--work-dir",
                wt,
                "--issue",
                str(issue),
                "--force",
            ],
            cwd=ctx.main,
            timeout=600,
        )
        try:
            data = json.loads(res.stdout)
        except ValueError:
            data = None
        status = data.get("status") if isinstance(data, dict) else None
        if res.returncode != 0 or status != "removed":
            raise StepFailed(
                f"cleanup_worktree.py did not remove {wt} (status {status!r}, exit {res.returncode})",
                {"exit_code": res.returncode, "tail": tail(res), "json": data},
                res.returncode or 1,
            )
        return {"status": "removed"}

    def prune(ctx: StepCtx) -> dict[str, Any]:
        wt = ctx.task["worktree"]
        if ctx.outputs["remove"].get("shared_with"):
            return {"pruned": False}
        registered = _registered_worktrees(ctx)
        if wt is not None and (registered is None or Path(wt).resolve() in registered):
            res = ctx.runner.run(["git", "-C", str(ctx.main), "worktree", "prune"], cwd=ctx.main, timeout=60)
            if not res.ok:
                raise StepFailed("git worktree prune failed", {"exit_code": res.returncode, "tail": tail(res)})
            return {"pruned": True}
        return {"pruned": False}

    return (Step("remove", remove), Step("prune", prune))


def _worktree_cleanup_probe(ctx: StepCtx) -> dict[str, Any | None]:
    wt = ctx.task["worktree"]
    if wt is None:
        return {"remove": {"skipped": True}, "prune": {"pruned": False}}
    if Path(wt).exists():
        return {}
    registered = _registered_worktrees(ctx)
    if registered is not None and Path(wt).resolve() not in registered:
        return {"remove": {"skipped": True}, "prune": {"pruned": False}}
    return {}


def _worktree_cleanup_success(conn: sqlite3.Connection, ctx: StepCtx) -> dict[str, Any]:
    conn.execute("UPDATE tasks SET worktree = NULL, updated_at = ? WHERE id = ?", (db.now(ctx.clock), ctx.task["id"]))
    shared = ctx.outputs["remove"].get("shared_with")
    return {"worktree": None} if shared is None else {"worktree": None, "shared_with": shared}


def _main_checkout(task: TaskRow, args: Mapping[str, Any]) -> Sequence[str]:
    return (locks.MAIN_CHECKOUT,)


# --------------------------------------------------------------------------- gate


def _gate_steps(task: TaskRow, args: Mapping[str, Any]) -> Sequence[Step]:
    wt = Path(_need(task, "worktree"))
    mode = args.get("mode")
    if mode == "impacted":
        flags = ["--impacted"]
    elif mode == "quick" and isinstance(args.get("paths"), list) and args["paths"]:
        flags = ["--quick", *[str(p) for p in args["paths"]]]
    else:
        raise errors.CpError("USAGE", "args.mode must be `impacted`, or `quick` with a non-empty args.paths")
    return (
        argv_step(
            "gate",
            lambda ctx: [str(wt / "venv" / "bin" / "python"), "scripts/qs/quality_gate.py", *flags],
            lambda ctx: wt,
            timeout=3600,
        ),
    )


def _gate_success(conn: sqlite3.Connection, ctx: StepCtx) -> dict[str, Any]:
    return dict(ctx.outputs["gate"])


# --------------------------------------------------------------------------- spawn / resume


def _own_node(ctx: StepCtx) -> sqlite3.Row | None:
    with ctx.write() as conn:
        row: sqlite3.Row | None = conn.execute(
            "SELECT * FROM nodes WHERE spawn_tool_key = ?", (f"{ctx.tool}/{ctx.key}",)
        ).fetchone()
        return row


def _reap_own(ctx: StepCtx, own: sqlite3.Row) -> sqlite3.Row:
    with ctx.write() as conn:
        conn.execute(
            "UPDATE nodes SET state = 'reaped', updated_at = ? WHERE id = ? AND state = 'spawning'",
            (db.now(ctx.clock), own["id"]),
        )
        row: sqlite3.Row = conn.execute("SELECT * FROM nodes WHERE id = ?", (own["id"],)).fetchone()
        return row


def _launch_probe(ctx: StepCtx, own: sqlite3.Row, *, resume: bool) -> dict[str, Any | None]:
    out: dict[str, Any | None] = {}
    listing = ctx.claude.try_agents()
    agent = None
    if listing is not None and own["state"] == "spawning":
        found = liveness.find(listing, name=own["name"])
        if found is not None and (not resume or found.started_after(own["launch_at"])):
            agent = found
        elif found is None and own["launch_at"] is not None and not nodes.recently_launched(own, ctx.clock):
            # Launched, never listed within LAUNCH_SETTLE_S, by a process proven dead (claim() took this key
            # over only because its holder was dead): reap our own row so the replay relaunches.
            own = _reap_own(ctx, own)
    if own["state"] == "spawning":
        out["reserve"] = {"node_id": own["id"], "name": own["name"]}
    else:
        out["reserve"] = None
    if agent is not None:
        out["launch"] = {"launched": True, "probed": True}
        out["identify"] = {"session_id": agent.session_id, "short_id": agent.id}
    elif own["state"] == "reaped" or own["launch_at"] is None:
        out["launch"] = None
        out["identify"] = None
    else:
        out["launch"] = {"pending": True}
    return out


def _reserve_prelude(ctx: StepCtx) -> tuple[list[liveness.Agent] | None, dict[str, bool]]:
    """Outside the transaction: the listing, a refresh, and the spawn holders' liveness."""
    call = ctx._call
    listing = ctx.claude.try_agents()
    nodes.refresh(call.conn, ctx.clock, listing)
    return listing, locks.spawn_holders_alive(call.conn, call.ctx.probe)


def _launch(ctx: StepCtx, start: Callable[[sqlite3.Row], runner.RunResult]) -> dict[str, Any]:
    node_id = ctx.outputs["reserve"]["node_id"]
    with ctx.write() as conn:
        conn.execute(
            "UPDATE nodes SET launch_at = ?, updated_at = ? WHERE id = ?",
            (db.now(ctx.clock), db.now(ctx.clock), node_id),
        )
        row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
    res = start(row)
    if not res.ok:
        raise StepFailed(f"claude --bg exited {res.returncode}", {"exit_code": res.returncode}, res.returncode)
    return {"launched": True}


def _identify(ctx: StepCtx, *, resume: bool) -> dict[str, Any]:
    node_id = ctx.outputs["reserve"]["node_id"]
    with ctx.write() as conn:
        row = conn.execute("SELECT name, launch_at FROM nodes WHERE id = ?", (node_id,)).fetchone()
    deadline = clock_mod.parse(row["launch_at"]).timestamp() + nodes.LAUNCH_SETTLE_S
    while True:
        listing = ctx.claude.try_agents()
        found = None if listing is None else liveness.find(listing, name=row["name"])
        if found is not None and (not resume or found.started_after(row["launch_at"])):
            return {"session_id": found.session_id, "short_id": found.id}
        if ctx.clock.now().timestamp() >= deadline:
            raise errors.CpError("BUSY", f"session {row['name']} is not listed yet; replay the same key")
        ctx.clock.sleep(IDENTIFY_POLL_S)


def _node_token(conn: sqlite3.Connection, node_id: str) -> str:
    row = conn.execute("SELECT id, generation, nonce FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return tokens.mint("node", row["id"], row["generation"], row["nonce"])


def _spawn_steps(task: TaskRow, args: Mapping[str, Any]) -> Sequence[Step]:
    wt = Path(_need(task, "worktree"))
    run_id = _need(task, "run_id")
    for name in ("agent", "permission_mode"):
        if not isinstance(args.get(name), str) or not args[name]:
            raise errors.CpError("USAGE", f"args.{name} is required")
    prompt = _read_arg_file(args, "prompt_file")
    model = args.get("model")
    replace = bool(args.get("replace", False))

    def reserve(ctx: StepCtx) -> None:
        listing, alive = _reserve_prelude(ctx)
        skey = f"{ctx.tool}/{ctx.key}"
        with ctx.write() as conn:
            # A `spawning` row of this key is marked done by the probe; only absent or reaped reach here.
            own = conn.execute("SELECT * FROM nodes WHERE spawn_tool_key = ?", (skey,)).fetchone()
            limit = locks.max_nodes()
            if own is not None:  # our own row, reaped meanwhile: re-take it (the guard allows only this)
                locks.admit_node(conn, ctx.clock, listing=listing, holders_alive=alive, limit=limit)
                nodes.move(conn, ctx.clock, own["id"], "spawning", expect="reaped", launch_at=None)
                ctx.record_output({"node_id": own["id"], "name": own["name"]})
                return
            live = conn.execute(
                "SELECT * FROM nodes WHERE task_id = ? AND state NOT IN ('stopped', 'superseded')"
                " ORDER BY generation DESC LIMIT 1",
                (ctx.task["id"],),
            ).fetchone()
            if live is not None and not replace:
                raise errors.CpError(
                    "CONFLICT", f"task {ctx.task['id']} has a live node {live['id']}: resume it, or --replace"
                )
            if live is not None:  # first, so the replaced node frees its place; a BUSY below rolls this back
                nodes.move(conn, ctx.clock, live["id"], "superseded")
            locks.admit_node(conn, ctx.clock, listing=listing, holders_alive=alive, limit=limit)
            generation = (
                int(
                    conn.execute(
                        "SELECT coalesce(max(generation), 0) FROM nodes WHERE task_id = ?", (ctx.task["id"],)
                    ).fetchone()[0]
                )
                + 1
            )
            run_name = conn.execute("SELECT name FROM runs WHERE id = ?", (run_id,)).fetchone()["name"]
            node_id = db.next_id(conn, "node", "N")
            name = f"{run_name}-{ctx.task['id']}-g{generation}"
            now = db.now(ctx.clock)
            conn.execute(
                "INSERT INTO nodes (id, run_id, task_id, generation, name, nonce, permission_mode, state, spawn_tool_key,"
                " spawned_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'spawning', ?, ?, ?)",
                (
                    node_id,
                    run_id,
                    ctx.task["id"],
                    generation,
                    name,
                    tokens.new_nonce(),
                    args["permission_mode"],
                    skey,
                    now,
                    now,
                ),
            )
            ctx.record_output({"node_id": node_id, "name": name})

    def launch(ctx: StepCtx) -> dict[str, Any]:
        def start(row: sqlite3.Row) -> runner.RunResult:
            with ctx.write() as conn:
                token = _node_token(conn, row["id"])
            settings = json.dumps(hooks.hooks_settings("node", ctx.main), sort_keys=True)
            text = (
                f"{prompt}\n\nControl Plane: {ctx.main}/scripts/qs/cp.py · run {row['run_id']} · task {row['task_id']}"
                f" · token {token}"
            )
            argv = ["--agent", str(args["agent"]), "-n", row["name"]]
            if model:
                argv += ["--model", str(model)]
            argv += [
                "--permission-mode",
                str(args["permission_mode"]),
                "--settings",
                settings,
                "--add-dir",
                str(ctx.main),
                text,
            ]
            return ctx.claude.spawn_bg(argv, cwd=wt)

        return _launch(ctx, start)

    return (
        Step("reserve", reserve),
        Step("launch", launch, detach=True),
        Step("identify", lambda ctx: _identify(ctx, resume=False)),
    )


def _spawn_probe(ctx: StepCtx) -> dict[str, Any | None]:
    own = _own_node(ctx)
    if own is None or own["state"] not in ("spawning", "reaped"):
        return {}
    return _launch_probe(ctx, own, resume=False)


def _spawn_guard(ctx: StepCtx) -> None:
    own = _own_node(ctx)
    if own is not None and own["state"] not in ("spawning", "reaped"):
        raise errors.CpError("INVALID_STATE", f"node {own['id']} of this call is {own['state']}")


def _start_success(conn: sqlite3.Connection, ctx: StepCtx) -> dict[str, Any]:
    node_id = ctx.outputs["reserve"]["node_id"]
    ident = ctx.outputs.get("identify") or {}
    nodes.move(
        conn,
        ctx.clock,
        node_id,
        "running",
        expect="spawning",
        session_id=ident.get("session_id"),
        short_id=ident.get("short_id"),
    )
    row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return {
        "node_id": node_id,
        "name": row["name"],
        "generation": row["generation"],
        "session_id": row["session_id"],
        "short_id": row["short_id"],
    }


def _resume_steps(task: TaskRow, args: Mapping[str, Any]) -> Sequence[Step]:
    wt = Path(_need(task, "worktree"))
    message = _read_arg_file(args, "message_file")

    def reserve(ctx: StepCtx) -> None:
        listing, alive = _reserve_prelude(ctx)
        skey = f"{ctx.tool}/{ctx.key}"
        with ctx.write() as conn:
            node = nodes.current(conn, ctx.task["id"])  # `spawning` under this key is marked done by the probe
            if node["state"] != "reaped":
                raise errors.CpError("INVALID_STATE", f"node {node['id']} is {node['state']}: wake it with SendMessage")
            if node["session_id"] is None:
                raise errors.CpError("INVALID_STATE", f"node {node['id']} never launched: re-spawn with --replace")
            if "launch" not in ctx.outputs:
                if listing is None:
                    raise errors.CpError("BUSY", "liveness unknown (the session listing failed), retry")
                if nodes.listed(listing, node) is not None:
                    raise errors.CpError("INVALID_STATE", f"node {node['id']} is live: wake it with SendMessage")
            locks.admit_node(conn, ctx.clock, listing=listing, holders_alive=alive, limit=locks.max_nodes())
            nodes.move(conn, ctx.clock, node["id"], "spawning", expect="reaped", spawn_tool_key=skey, launch_at=None)
            ctx.record_output({"node_id": node["id"], "name": node["name"]})

    def launch(ctx: StepCtx) -> dict[str, Any]:
        return _launch(ctx, lambda row: ctx.claude.resume_bg(row["session_id"], message, cwd=wt))

    return (
        Step("reserve", reserve),
        Step("launch", launch, detach=True),
        Step("identify", lambda ctx: _identify(ctx, resume=True)),
    )


def _resume_probe(ctx: StepCtx) -> dict[str, Any | None]:
    own = _own_node(ctx)
    if own is None or own["state"] not in ("spawning", "reaped"):
        return {}
    return _launch_probe(ctx, own, resume=True)


def _resume_guard(ctx: StepCtx) -> None:
    with ctx.write() as conn:
        node = nodes.current(conn, ctx.task["id"])
    mine = node["state"] == "spawning" and node["spawn_tool_key"] == f"{ctx.tool}/{ctx.key}"
    if node["state"] != "reaped" and not mine:
        raise errors.CpError("INVALID_STATE", f"node {node['id']} is {node['state']}")


def _node_token_extra(conn: sqlite3.Connection, result: dict[str, Any], who: tokens.Principal) -> dict[str, Any]:
    assert who.kind == "run"  # spawn and resume are run-token tools: only the run gets the node token back
    return {"token": _node_token(conn, result["node_id"])}


_RESPONSE_EXTRAS: dict[str, Callable[[sqlite3.Connection, dict[str, Any], tokens.Principal], dict[str, Any]]] = {
    "spawn": _node_token_extra,
    "resume": _node_token_extra,
}


# --------------------------------------------------------------------------- issue-create / pr-create / push


def _labels(args: Mapping[str, Any]) -> str:
    labels = args.get("labels", "")
    if isinstance(labels, list):
        return ",".join(str(x) for x in labels)
    return str(labels)


def _issue_create_steps(task: TaskRow, args: Mapping[str, Any]) -> Sequence[Step]:
    title = args.get("title")
    if not isinstance(title, str) or not title:
        raise errors.CpError("USAGE", "args.title is required")
    body = _read_arg_file(args, "body_file")
    labels = _labels(args)
    return (
        argv_step(
            "create",
            lambda ctx: [
                _python(ctx.main),
                _script(ctx.main, "create_issue.py"),
                "--title",
                title,
                "--body",
                f"{body}\n\n{marker(ctx)}",
                "--labels",
                labels,
            ],
            lambda ctx: ctx.main,
            check=lambda res: _script_json(res, "create_issue.py"),
            timeout=120,
        ),
    )


def _issue_create_probe(ctx: StepCtx) -> dict[str, Any | None]:
    search = f'"qs-cp-key: {ctx.tool}/{ctx.key}" in:body'
    found = _gh_list_by_marker(
        ctx, ["gh", "issue", "list", "--state", "all", "--search", search, "--json", "number,url,body"], ctx.main
    )
    return {} if found is None else {"create": {"issue_number": found["number"], "url": found["url"]}}


def _issue_create_success(conn: sqlite3.Connection, ctx: StepCtx) -> dict[str, Any]:
    out = ctx.outputs["create"]
    tasks.update_fields(conn, ctx.clock, ctx.task["id"], {"issue_number": out["issue_number"]})
    return {"issue_number": out["issue_number"], "url": out["url"]}


def _pr_create_steps(task: TaskRow, args: Mapping[str, Any]) -> Sequence[Step]:
    wt = Path(_need(task, "worktree"))
    issue = _need(task, "issue_number")
    title = args.get("title")
    if not isinstance(title, str) or not title:
        raise errors.CpError("USAGE", "args.title is required")
    summary = _read_arg_file(args, "summary_file")
    return (
        argv_step(
            "create",
            lambda ctx: [
                _python(ctx.main),
                _script(ctx.main, "create_pr.py"),
                "--title",
                title,
                "--summary",
                f"{summary}\n\n{marker(ctx)}",
                "--issue",
                str(issue),
            ],
            lambda ctx: wt,
            check=lambda res: _script_json(res, "create_pr.py"),
            inject_token=True,
            timeout=300,
        ),
    )


def _pr_create_probe(ctx: StepCtx) -> dict[str, Any | None]:
    branch = _need(ctx.task, "branch")
    found = _gh_list_by_marker(
        ctx,
        ["gh", "pr", "list", "--head", branch, "--state", "all", "--json", "number,url,body"],
        Path(_need(ctx.task, "worktree")),
    )
    return {} if found is None else {"create": {"pr_number": found["number"], "url": found["url"]}}


def _pr_create_success(conn: sqlite3.Connection, ctx: StepCtx) -> dict[str, Any]:
    out = ctx.outputs["create"]
    tasks.update_fields(conn, ctx.clock, ctx.task["id"], {"pr_number": out["pr_number"], "pr_url": out["url"]})
    return {"pr_number": out["pr_number"], "pr_url": out["url"]}


def _head(ctx: StepCtx, wt: Path) -> str:
    res = ctx.runner.run(["git", "rev-parse", "HEAD"], cwd=wt, timeout=60)
    if not res.ok:
        raise StepFailed("git rev-parse HEAD failed", {"exit_code": res.returncode, "tail": tail(res)})
    return res.stdout.strip()


def _push_steps(task: TaskRow, args: Mapping[str, Any]) -> Sequence[Step]:
    wt = Path(_need(task, "worktree"))
    branch = _need(task, "branch")

    def push(ctx: StepCtx) -> dict[str, Any]:
        sha = _head(ctx, wt)
        res = ctx.runner.run(["git", "push", "-u", "origin", branch], cwd=wt, timeout=300)
        if not res.ok:
            raise StepFailed("git push failed", {"exit_code": res.returncode, "tail": tail(res)}, res.returncode)
        return {"sha": sha}

    return (Step("push", push, inject_token=True),)


def _push_probe(ctx: StepCtx) -> dict[str, Any | None]:
    wt = Path(_need(ctx.task, "worktree"))
    remote = ctx.runner.run(
        ["git", "ls-remote", "origin", f"refs/heads/{_need(ctx.task, 'branch')}"], cwd=wt, timeout=60
    )
    local = ctx.runner.run(["git", "rev-parse", "HEAD"], cwd=wt, timeout=60)
    remote_sha = remote.stdout.split()[0] if remote.ok and remote.stdout.split() else None
    if local.ok and remote_sha is not None and remote_sha == local.stdout.strip():
        return {"push": {"sha": remote_sha}}
    return {}


def _push_success(conn: sqlite3.Connection, ctx: StepCtx) -> dict[str, Any]:
    return {"sha": ctx.outputs["push"]["sha"]}


# --------------------------------------------------------------------------- merge


def _merge_locks(task: TaskRow, args: Mapping[str, Any]) -> Sequence[str]:
    return (f"{locks.INTEGRATION}{_need(task, 'branch')}", locks.MAIN_MERGE)


def _gh_json(ctx: StepCtx, argv: list[str]) -> dict[str, Any] | None:
    res = ctx.runner.run(argv, cwd=ctx.main, timeout=120)
    try:
        data = json.loads(res.stdout) if res.ok else None
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _merge_steps(task: TaskRow, args: Mapping[str, Any]) -> Sequence[Step]:
    pr = str(_need(task, "pr_number"))

    def policy(ctx: StepCtx) -> dict[str, Any]:
        verdict = merge_policy.check(ctx.task)
        if not verdict.ok:
            raise errors.CpError("POLICY_REFUSED", verdict.reason)
        return {"ok": True, "reason": verdict.reason}

    def head(ctx: StepCtx) -> dict[str, Any]:
        data = _gh_json(ctx, ["gh", "pr", "view", pr, "--json", "headRefOid"])
        if data is None or not data.get("headRefOid"):
            raise StepFailed(f"cannot read PR #{pr}'s head")
        return {"oid": data["headRefOid"]}

    def merge(ctx: StepCtx) -> dict[str, Any]:
        oid = ctx.outputs["head"]["oid"]
        res = ctx.runner.run(
            ["gh", "pr", "merge", pr, "--merge", "--match-head-commit", oid], cwd=ctx.main, timeout=300
        )
        if not res.ok:
            raise StepFailed("gh pr merge failed", {"exit_code": res.returncode, "tail": tail(res)}, res.returncode)
        data = _gh_json(ctx, ["gh", "pr", "view", pr, "--json", "mergeCommit"]) or {}
        commit = data.get("mergeCommit") or {}
        return {"merge_sha": commit.get("oid") if isinstance(commit, dict) else None}

    return (Step("policy", policy), Step("head", head), Step("merge", merge))


def _merge_probe(ctx: StepCtx) -> dict[str, Any | None]:
    data = _gh_json(ctx, ["gh", "pr", "view", str(_need(ctx.task, "pr_number")), "--json", "state,mergeCommit"])
    if data is not None and data.get("state") == "MERGED":
        commit = data.get("mergeCommit") or {}
        sha = commit.get("oid") if isinstance(commit, dict) else None
        return {"policy": {"ok": True, "reason": "already merged"}, "head": {"oid": None}, "merge": {"merge_sha": sha}}
    return {"policy": None, "head": None}


def _merge_guard(ctx: StepCtx) -> None:
    if "merge" in ctx.outputs:
        return
    task_state_guard(lambda s: s == "ready_to_merge")(ctx)


def _merge_success(conn: sqlite3.Connection, ctx: StepCtx) -> dict[str, Any]:
    """The PR is merged: always record it; move the task only if it is still ``ready_to_merge``."""
    sha = ctx.outputs["merge"]["merge_sha"]
    tasks.update_fields(conn, ctx.clock, ctx.task["id"], {"merge_sha": sha})
    state = tasks.get(conn, ctx.task["id"])["state"]
    if state != "ready_to_merge":  # changed mid-merge: the orchestrator or the maintainer reconciles it
        return {"merge_sha": sha, "state_conflict": {"expected": "ready_to_merge", "actual": state}}
    actor = ctx._call.who.actor if ctx._call is not None and ctx._call.who is not None else "tool"
    tasks.apply_transition(conn, ctx.clock, ctx.task["id"], "merged", actor=actor, node=False, expect="ready_to_merge")
    return {"merge_sha": sha}


# --------------------------------------------------------------------------- the built-in registry

BUILTINS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "worktree-create",
        _worktree_create_steps,
        locks=_main_checkout,
        guard=non_terminal,
        on_success=_worktree_create_success,
    ),
    ToolSpec(
        "worktree-cleanup",
        _worktree_cleanup_steps,
        locks=_main_checkout,
        probe=_worktree_cleanup_probe,
        on_success=_worktree_cleanup_success,
    ),
    ToolSpec(
        "gate",
        _gate_steps,
        cap=locks.GATES,
        guard=non_terminal,
        on_success=_gate_success,
        token_kinds=frozenset({"run", "node"}),
    ),
    ToolSpec(
        "spawn", _spawn_steps, probe=_spawn_probe, guard=compose(non_terminal, _spawn_guard), on_success=_start_success
    ),
    ToolSpec(
        "resume",
        _resume_steps,
        probe=_resume_probe,
        guard=compose(non_terminal, _resume_guard),
        on_success=_start_success,
    ),
    ToolSpec(
        "issue-create",
        _issue_create_steps,
        probe=_issue_create_probe,
        guard=non_terminal,
        on_success=_issue_create_success,
    ),
    ToolSpec(
        "pr-create",
        _pr_create_steps,
        probe=_pr_create_probe,
        guard=non_terminal,
        on_success=_pr_create_success,
        token_kinds=frozenset({"run", "node"}),
    ),
    ToolSpec(
        "push",
        _push_steps,
        probe=_push_probe,
        guard=non_terminal,
        on_success=_push_success,
        token_kinds=frozenset({"run", "node"}),
    ),
    ToolSpec(
        "merge", _merge_steps, locks=_merge_locks, probe=_merge_probe, guard=_merge_guard, on_success=_merge_success
    ),
)
BUILTIN_NAMES = frozenset(s.name for s in BUILTINS)
for _spec in BUILTINS:
    register(_spec)


def reset() -> None:
    """Drop every tool registered after import (tests)."""
    for name in list(REGISTRY):
        if name not in BUILTIN_NAMES:
            del REGISTRY[name]
