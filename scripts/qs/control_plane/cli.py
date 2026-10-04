"""The ``cp.py`` command line: a ``COMMANDS`` dispatch table, JSON out, exit codes.

Every command prints one JSON object on stdout — ``{"ok": true, …}`` or
``{"ok": false, "error": "<CODE>", "detail": "…"}`` — and exits with the
code of ``errors.EXIT_CODES``. Hooks are the exception: they print the
Claude Code hook protocol (or nothing).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, NoReturn, TextIO

from . import errors, procsetup


@dataclass(frozen=True)
class Raw:
    """A handler result printed verbatim (hooks), with its exit code."""

    text: str
    code: int = 0


@dataclass
class Io:
    stdin: TextIO
    stdout: TextIO


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
    return {"package": "control_plane"}


COMMANDS: dict[str, Command] = {
    c.name: c for c in (Command("version", "exempt", _version, help="print the package and schema version (no DB)"),)
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
    io = Io(stdin=stdin or sys.stdin, stdout=stdout or sys.stdout)
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
