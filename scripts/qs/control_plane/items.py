"""#400's tools: work-item worktrees and their integration into the deliverable (QS-400 story §9).

Five ``ToolSpec``s on top of #399's ``run_recorded``:

* ``item-create`` / ``item-cleanup`` — ``setup_task.py --item`` and
  ``cleanup_worktree.py --item`` run from ``<MAIN>`` under ``main-checkout``;
  ``item-cleanup`` then drops the item's integration scratch under
  ``integration:QS_<N>`` (process-held).
* ``integrate-start`` / ``integrate-finish`` / ``integrate-drop`` — the
  subcommands of ``integrate_item.py``, called by a session that holds
  ``integration:QS_<N>`` itself (``cp.py lock acquire``): the tools co-hold it.

The git mechanics live in the scripts; this module holds the locks, the
guards and the ``integrations`` records. It imports only from its own package
and so spells the branch names itself (``QS_<N>`` and ``QS_<N>_<k>``).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import errors, locks, tasks, tools
from .tools import Step, StepCtx, StepFailed, TaskRow, ToolSpec

ITEM_BRANCH_RE = re.compile(r"QS_([0-9]+)_([1-9][0-9]*)")
SCRIPT_TIMEOUT_S = 600
GATE_TIMEOUT_S = 3600
CLEANUP_FLAGS = (
    ("delete_branch", "--delete-branch"),
    ("discard_unintegrated", "--discard-unintegrated"),
    ("force", "--force"),
)
CLEANUP_OK = frozenset({"removed", "removed-branch-kept"})
DROP_FIELDS = ("dropped_head", "integrated", "deliverable_missing", "discarded_snapshot", "discarded_files")
START_ROWS = {"conflicts": "conflict", "already-integrated": "noop"}
NODE_KINDS = frozenset({"run", "node"})


def deliverable_branch(issue: int) -> str:
    return f"QS_{issue}"


def item_branch(issue: int, item: int) -> str:
    return f"QS_{issue}_{item}"


# --------------------------------------------------------------------------- before the claim


def _require_item(task: TaskRow) -> None:
    if task["deliverable_id"] is None or task["item_k"] is None:
        raise errors.CpError("INVALID_STATE", f"task {task['id']} is not a work item (no deliverable_id / item_k)")


def _bool_arg(args: Mapping[str, Any], name: str) -> bool:
    value = args.get(name, False)
    if not isinstance(value, bool):
        raise errors.CpError("USAGE", f"args.{name} must be true or false")
    return value


def _integration_lock(task: Mapping[str, Any]) -> str:
    """``integration:QS_<N>`` from the item's own ``branch`` (the ``locks`` callable cannot read the deliverable)."""
    match = ITEM_BRANCH_RE.fullmatch(task["branch"] or "")
    if match is None:
        raise errors.CpError("INVALID_STATE", f"task {task['id']} has no item branch QS_<N>_<k> ({task['branch']!r})")
    return f"{locks.INTEGRATION}{deliverable_branch(int(match[1]))}"


def _main_checkout(task: TaskRow, args: Mapping[str, Any]) -> Sequence[str]:
    return (locks.MAIN_CHECKOUT,)


def _cleanup_locks(task: TaskRow, args: Mapping[str, Any]) -> Sequence[str]:
    if ITEM_BRANCH_RE.fullmatch(task["branch"] or ""):
        return (_integration_lock(task), locks.MAIN_CHECKOUT)
    return (locks.MAIN_CHECKOUT,)  # a NULL branch: no scratch can exist (a malformed one is refused by the guard)


def _start_drop_locks(task: TaskRow, args: Mapping[str, Any]) -> Sequence[str]:
    return (_integration_lock(task), locks.MAIN_CHECKOUT)


def _finish_locks(task: TaskRow, args: Mapping[str, Any]) -> Sequence[str]:
    return (_integration_lock(task),)


# --------------------------------------------------------------------------- the probe: the session check


def _session_probe(ctx: StepCtx) -> dict[str, Any | None]:
    """The caller's own session must hold ``integration:QS_<N>``: never wait on it in ``hold()``."""
    name = _integration_lock(ctx.task)
    who = ctx._call.who
    with ctx.write() as conn:
        row = conn.execute("SELECT * FROM locks WHERE name = ?", (name,)).fetchone()
    if (
        row is None
        or row["holder_kind"] != "session"
        or row["token_subject"] != who.subject
        or row["holder_session_id"] != who.session_id
    ):
        raise errors.CpError(
            "POLICY_REFUSED",
            f"this session does not hold {name}: run `cp.py lock acquire --name {name}"
            f' --purpose "integrate item {ctx.task["item_k"]}"` first',
        )
    return {}


# --------------------------------------------------------------------------- guards


def _deliverable(ctx: StepCtx) -> sqlite3.Row:
    with ctx.write() as conn:
        return tasks.get(conn, ctx.task["deliverable_id"])


def _item_guard(*, branch: bool = False, allow_null: bool = False) -> Callable[[StepCtx], None]:
    """The deliverable is ``QS_<issue_number>``; with ``branch``, the item is on ``QS_<N>_<k>`` (or NULL)."""

    def guard(ctx: StepCtx) -> None:
        deliverable = _deliverable(ctx)
        issue = deliverable["issue_number"]
        if issue is None or deliverable["branch"] != deliverable_branch(issue):
            raise errors.CpError(
                "INVALID_STATE",
                f"deliverable {deliverable['id']} has issue {issue!r} and branch {deliverable['branch']!r}",
            )
        expected = item_branch(issue, ctx.task["item_k"])
        actual = ctx.task["branch"]
        if branch and actual != expected and not (allow_null and actual is None):
            raise errors.CpError("INVALID_STATE", f"task {ctx.task['id']} is on {actual!r}, not {expected}")

    return guard


def _live_guard(ctx: StepCtx) -> None:
    """The item **and** its deliverable are non-terminal (``tools.non_terminal`` checks only the item)."""
    for row in (ctx.task, _deliverable(ctx)):
        if row["state"] in tasks.TERMINAL:
            raise errors.CpError("INVALID_STATE", f"task {row['id']} is {row['state']}")


# --------------------------------------------------------------------------- step helpers


def _numbers(ctx: StepCtx) -> tuple[int, int]:
    """``(N, k)`` from the rows, read inside the step."""
    return int(_deliverable(ctx)["issue_number"]), int(ctx.task["item_k"])


def _call_key(ctx: StepCtx) -> str:
    return f"{ctx.tool}/{ctx.key}"


def _unexpected(what: str, res: Any, data: Any) -> StepFailed:
    status = data.get("status") if isinstance(data, dict) else None
    return StepFailed(
        f"{what} answered {status!r} (exit {res.returncode})",
        {"exit_code": res.returncode, "tail": tools.tail(res), "json": data},
        res.returncode or 1,
    )


def _integrate(
    ctx: StepCtx, sub: str, *extra: str, statuses: Sequence[str], timeout: float = SCRIPT_TIMEOUT_S
) -> dict[str, Any]:
    """``integrate_item.py <sub> --issue N --item k …`` from ``<MAIN>`` → its JSON, whose ``status`` is expected."""
    n, k = _numbers(ctx)
    res = ctx.runner.run(
        [
            tools._python(ctx.main),
            tools._script(ctx.main, "integrate_item.py"),
            sub,
            "--issue",
            str(n),
            "--item",
            str(k),
            *extra,
        ],
        cwd=ctx.main,
        timeout=timeout,
    )
    what = f"integrate_item.py {sub}"
    data = tools._script_json(res, what)
    if data.get("status") not in statuses:
        raise _unexpected(what, res, data)
    return data


def _scratch_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    return {"outcome": data["status"], **{name: data[name] for name in DROP_FIELDS if name in data}}


def _drop(ctx: StepCtx) -> dict[str, Any]:
    return _scratch_summary(_integrate(ctx, "drop", statuses=("dropped", "nothing-to-drop")))


# --------------------------------------------------------------------------- item-create / item-cleanup


def _item_create_steps(task: TaskRow, args: Mapping[str, Any]) -> Sequence[Step]:
    _require_item(task)

    def setup(ctx: StepCtx) -> dict[str, Any]:
        n, k = _numbers(ctx)
        res = ctx.runner.run(
            [
                tools._python(ctx.main),
                tools._script(ctx.main, "setup_task.py"),
                str(n),
                "--item",
                str(k),
                "--harness",
                "claude-code",
            ],
            cwd=ctx.main,
            timeout=SCRIPT_TIMEOUT_S,
        )
        return tools._script_json(res, "setup_task.py")

    return (Step("setup", setup),)


def _item_create_success(conn: sqlite3.Connection, ctx: StepCtx) -> dict[str, Any]:
    out = ctx.outputs["setup"]
    worktree = str(Path(out["worktree_path"]).resolve())
    tasks.update_fields(conn, ctx.clock, ctx.task["id"], {"worktree": worktree, "branch": out["branch"]})
    return {"worktree": worktree, "branch": out["branch"]}


def _item_cleanup_steps(task: TaskRow, args: Mapping[str, Any]) -> Sequence[Step]:
    _require_item(task)
    chosen = {name for name, _ in CLEANUP_FLAGS if _bool_arg(args, name)}
    if "discard_unintegrated" in chosen and "delete_branch" not in chosen:
        raise errors.CpError("USAGE", "args.discard_unintegrated requires args.delete_branch")
    flags = [flag for name, flag in CLEANUP_FLAGS if name in chosen]

    def cleanup(ctx: StepCtx) -> dict[str, Any]:
        n, k = _numbers(ctx)
        work_dir = ctx.task["worktree"] or str(ctx.main.parent / f"{ctx.main.name}-worktrees" / item_branch(n, k))
        res = ctx.runner.run(
            [
                tools._python(ctx.main),
                tools._script(ctx.main, "cleanup_worktree.py"),
                "--issue",
                str(n),
                "--item",
                str(k),
                "--work-dir",
                work_dir,
                *flags,
            ],
            cwd=ctx.main,
            timeout=SCRIPT_TIMEOUT_S,
        )
        try:
            data = json.loads(res.stdout)
        except ValueError:
            data = None
        if res.returncode != 0 or not isinstance(data, dict) or data.get("status") not in CLEANUP_OK:
            raise _unexpected("cleanup_worktree.py", res, data)
        return data

    # cleanup first: a refused cleanup (StepFailed) never drops a live scratch; a retry re-runs it idempotently
    return (Step("cleanup", cleanup), Step("drop_scratch", _drop))


def _item_cleanup_success(conn: sqlite3.Connection, ctx: StepCtx) -> dict[str, Any]:
    tasks.update_fields(conn, ctx.clock, ctx.task["id"], {"worktree": None})  # the branch stays: the item's identity
    return {**ctx.outputs["cleanup"], "scratch": ctx.outputs["drop_scratch"]}


# --------------------------------------------------------------------------- integrate-start / -finish / -drop


def _record(ctx: StepCtx, conn: sqlite3.Connection, *, item_tip: str, merge_commit: str | None, result: str) -> int:
    return tasks.record_integration(
        conn,
        ctx.clock,
        item_task_id=ctx.task["id"],
        deliverable_id=ctx.task["deliverable_id"],
        item_tip=item_tip,
        merge_commit=merge_commit,
        result=result,
        tool_call_key=_call_key(ctx),
    )


def _start_steps(task: TaskRow, args: Mapping[str, Any]) -> Sequence[Step]:
    _require_item(task)

    def prepare(ctx: StepCtx) -> dict[str, Any]:
        n, k = _numbers(ctx)
        res = ctx.runner.run(
            ["git", "rev-parse", "--verify", f"refs/heads/{item_branch(n, k)}"], cwd=ctx.main, timeout=60
        )
        if not res.ok:
            raise StepFailed(
                f"git rev-parse refs/heads/{item_branch(n, k)} failed",
                {"exit_code": res.returncode, "tail": tools.tail(res)},
                res.returncode,
            )
        tip = res.stdout.strip()
        return _integrate(ctx, "prepare", "--item-tip", tip, statuses=("already-integrated", "merged", "conflicts"))

    def record(ctx: StepCtx) -> None:
        prep = ctx.outputs["prepare"]
        result = START_ROWS.get(prep["status"])  # `merged` writes no row: the finish will
        with ctx.write() as conn:
            row = None
            if result is not None:
                row = _record(ctx, conn, item_tip=prep["item_tip"], merge_commit=None, result=result)
            ctx.record_output({"result": result, "row": row})

    return (Step("prepare", prepare), Step("record", record))


def _start_success(conn: sqlite3.Connection, ctx: StepCtx) -> dict[str, Any]:
    prep = ctx.outputs["prepare"]
    out = {"outcome": prep["status"], "item_tip": prep["item_tip"]}
    out.update({name: prep[name] for name in ("base", "files", "scratch") if name in prep})
    return out


def _green(outputs: Mapping[str, Any]) -> bool:
    return outputs["check"].get("moved") is True or outputs["gate"].get("status") == "green"


def _finish_steps(task: TaskRow, args: Mapping[str, Any]) -> Sequence[Step]:
    _require_item(task)

    def check(ctx: StepCtx) -> dict[str, Any]:
        return _integrate(ctx, "check", statuses=("ready",))

    def gate(ctx: StepCtx) -> dict[str, Any]:
        found = ctx.outputs["check"]
        if found.get("moved") is True:
            return {"status": "skipped", "reason": "moved"}
        return _integrate(
            ctx, "gate", "--expect-head", found["head"], statuses=("green", "red"), timeout=GATE_TIMEOUT_S
        )

    def move(ctx: StepCtx) -> dict[str, Any]:
        if not _green(ctx.outputs):
            return {"status": "skipped", "reason": "red"}
        found = ctx.outputs["check"]
        return _integrate(
            ctx, "move", "--new", found["head"], "--old", found["base"], statuses=("moved", "already-moved")
        )

    def record(ctx: StepCtx) -> None:
        found = ctx.outputs["check"]
        green = _green(ctx.outputs)
        with ctx.write() as conn:
            if green:  # `ok` is deduplicated by content, in this same transaction
                dup = conn.execute(
                    "SELECT id FROM integrations WHERE item_task_id = ? AND result = 'ok' AND merge_commit = ?",
                    (ctx.task["id"], found["head"]),
                ).fetchone()
                row = (
                    int(dup["id"])
                    if dup is not None
                    else _record(ctx, conn, item_tip=found["item_tip"], merge_commit=found["head"], result="ok")
                )
                ctx.record_output({"result": "ok", "row": row, "deduplicated": dup is not None})
            else:
                row = _record(ctx, conn, item_tip=found["item_tip"], merge_commit=None, result="gate_red")
                ctx.record_output({"result": "gate_red", "row": row, "deduplicated": False})

    return (Step("check", check), Step("gate", gate), Step("move", move), Step("record", record))


def _finish_success(conn: sqlite3.Connection, ctx: StepCtx) -> dict[str, Any]:
    found = ctx.outputs["check"]
    if _green(ctx.outputs):
        return {"outcome": "ok", "merge_commit": found["head"], "item_tip": found["item_tip"]}
    return {"outcome": "gate_red", "item_tip": found["item_tip"], "tail": ctx.outputs["gate"].get("tail", [])}


def _drop_steps(task: TaskRow, args: Mapping[str, Any]) -> Sequence[Step]:
    _require_item(task)
    return (Step("drop", _drop),)


def _drop_success(conn: sqlite3.Connection, ctx: StepCtx) -> dict[str, Any]:
    return dict(ctx.outputs["drop"])


# --------------------------------------------------------------------------- registration


SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "item-create",
        _item_create_steps,
        locks=_main_checkout,
        guard=tools.compose(_item_guard(), _live_guard),
        on_success=_item_create_success,
    ),
    ToolSpec(
        "item-cleanup",
        _item_cleanup_steps,
        locks=_cleanup_locks,
        guard=_item_guard(branch=True, allow_null=True),
        on_success=_item_cleanup_success,
    ),
    ToolSpec(
        "integrate-start",
        _start_steps,
        locks=_start_drop_locks,
        probe=_session_probe,
        guard=tools.compose(_item_guard(branch=True), _live_guard),
        on_success=_start_success,
        token_kinds=NODE_KINDS,
    ),
    ToolSpec(
        "integrate-finish",
        _finish_steps,
        locks=_finish_locks,
        cap=locks.GATES,
        probe=_session_probe,
        guard=tools.compose(_item_guard(branch=True), _live_guard),
        on_success=_finish_success,
        token_kinds=NODE_KINDS,
    ),
    ToolSpec(
        "integrate-drop",
        _drop_steps,
        locks=_start_drop_locks,
        probe=_session_probe,
        guard=_item_guard(branch=True),
        on_success=_drop_success,
        token_kinds=NODE_KINDS,
    ),
)
NAMES = tuple(spec.name for spec in SPECS)


def register_item_tools() -> None:
    """Register the five tools; idempotent (a name already in ``tools.REGISTRY`` is skipped)."""
    for spec in SPECS:
        if spec.name not in tools.REGISTRY:
            tools.register(spec)
