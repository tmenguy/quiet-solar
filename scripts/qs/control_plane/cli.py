"""The ``cp.py`` command line: a ``COMMANDS`` dispatch table, JSON out, exit codes.

Every command prints one JSON object on stdout — ``{"ok": true, …}`` or
``{"ok": false, "error": "<CODE>", "detail": "…"}`` — and exits with the
code of ``errors.EXIT_CODES``. Hooks are the exception: they print the
Claude Code hook protocol (or nothing).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, NoReturn, TextIO

from . import clock as clock_mod
from . import daemon, db, errors, liveness, migrations, paths, procsetup, runner


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


COMMANDS: dict[str, Command] = {
    c.name: c
    for c in (
        Command("version", "exempt", _version, help="print the package and schema version (no DB)"),
        Command("daemon", "exempt", _daemon, help="run the daemon (migrates the DB, heartbeats, ticks)"),
        Command("ensure", "exempt", _ensure, help="start or restart the daemon if needed"),
    )
}


# --------------------------------------------------------------------------- parsing


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="cp.py", description="Quiet Solar Control Plane")
    groups: dict[str, argparse._SubParsersAction[Any]] = {}
    top = parser.add_subparsers(dest="_cmd0", required=True, parser_class=_Parser)
    for name in sorted(COMMANDS):
        cmd = COMMANDS[name]
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
    io = Io(stdin=stdin or sys.stdin, stdout=stdout or sys.stdout, deps=make_deps())
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(argv)
        cmd = COMMANDS[args._command]
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
