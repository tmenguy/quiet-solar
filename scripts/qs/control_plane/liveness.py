"""Liveness evidence (§13): processes, process groups and Claude sessions.

* ``ProcessProbe`` — a pid is alive iff it exists and its start time still
  matches (pid reuse); a group is alive while ``killpg(pgid, 0)`` succeeds.
  ``alive`` is tri-state: when ``ps`` itself fails (timeout, exec failure,
  unparseable output) it returns ``None`` (unknown), and every caller keeps
  the holder: a failed probe is never evidence of absence.
* ``ClaudeCli`` — ``claude agents --json`` lists interactive and background
  sessions. Any failure raises ``CpError("INTERNAL")``, which callers treat
  as *unknown*: a failed listing is never evidence of absence.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import errors, runner

TOKEN_ENV = "QS_CP_TOKEN"
UNTRUSTED = "Workspace not trusted"  # Claude Code's refusal of `--bg` in an untrusted folder (2.1.299, #421)
BG_ERROR_MAX = 500  # a failed `claude --bg`'s recorded text keeps at most this many characters (#421, #424)


def untrusted(res: runner.RunResult) -> bool:
    """A ``claude --bg`` refused because its folder is not trusted (the phrase in stderr or stdout)."""
    return UNTRUSTED in res.stderr or UNTRUSTED in res.stdout


def trust_hint(trust_dir: Path | str) -> str:
    return (
        f"the folder is not trusted: run `claude` once in `{trust_dir}` or a parent folder and accept the trust prompt"
    )


def bg_failure(res: runner.RunResult, trust_dir: Path, limit: int) -> str:
    """``claude --bg exited N: <tail>`` (+ ``; <trust hint>`` when refused for trust), at most ``limit`` characters.

    The tail is stderr's, else stdout's: what the prefix and the hint leave room for. When the prefix and the hint
    alone exceed ``limit`` (an unusually long ``trust_dir``), the hint shows the folder's end after a leading ``…``:
    the result is then exactly ``limit`` characters, with no tail. Never over ``limit``: when even the prefix and the
    hint's fixed text do not fit (unreachable at ``BG_ERROR_MAX``), the result is their start, cut at ``limit``.
    """
    head = f"claude --bg exited {res.returncode}"
    output = res.stderr.strip() or res.stdout.strip()
    hint = "; " + trust_hint(trust_dir) if untrusted(res) else ""
    excess = len(head) + len(hint) - limit
    if hint and excess > 0:  # keep the head and the hint's text whole: `close_locked` keeps only the last `limit`
        hint = "; " + trust_hint("…" + str(trust_dir)[excess + 1 :])
    budget = limit - len(head) - len(": ") - len(hint)
    tail = output[-budget:] if budget > 0 else ""  # the guard matters: `output[-0:]` is the whole output
    return (head + (f": {tail}" if tail else "") + hint)[:limit]  # the head comes first: a last-resort cut keeps it


@dataclass(frozen=True)
class Holder:
    """A process identity: pid, its start time, its process group."""

    pid: int
    pid_start: str | None
    pgid: int | None


class ProbeUnknown(Exception):
    """``ps`` could not tell a pid's start time: unknown, never "dead"."""


class ProcessProbe:
    def __init__(self, run: runner.Runner | None = None) -> None:
        self.runner = run or runner.Runner()

    def alive(self, pid: int | None, start: str | None) -> bool | None:
        """``True``, ``False``, or ``None`` when the start-time probe failed (callers keep the holder)."""
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError, OverflowError:  # OverflowError: no such pid can exist
            return False
        except PermissionError:
            pass  # exists, owned by someone else
        if start is None:
            return True
        try:
            return self.start_of(pid) == start
        except ProbeUnknown:
            return None

    def group_alive(self, pgid: int | None) -> bool:
        if pgid is None:
            return False
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError, OverflowError:
            return False
        except PermissionError:
            return True
        return True

    def start_of(self, pid: int) -> str | None:
        """The pid's start time (``ps -o lstart=``, in UTC), or ``None`` if it no longer exists.

        Raises ``ProbeUnknown`` when ``ps`` failed for any other reason.
        """
        res = self.runner.run(
            ["ps", "-o", "lstart=", "-p", str(pid)], env_extra={"LC_ALL": "C", "TZ": "UTC0"}, timeout=10
        )
        text = " ".join(res.stdout.split())
        if res.returncode == 1 and not text:
            return None  # ps: no such process
        if res.returncode != 0 or not text:
            raise ProbeUnknown(f"ps exited {res.returncode}: {res.stderr.strip()[-200:]}")
        try:
            return datetime.strptime(text, "%a %b %d %H:%M:%S %Y").strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError as exc:
            raise ProbeUnknown(f"ps: unparseable start time {text!r}") from exc

    def me(self) -> Holder:
        pid = os.getpid()
        try:
            start = self.start_of(pid)
        except ProbeUnknown:
            start = None
        return Holder(pid, start, os.getpgid(0))

    def holder_alive(self, pid: int | None, start: str | None, pgid: int | None) -> bool:
        """A holder is dead iff its pid is proven dead **and** its group is gone (unknown keeps it)."""
        return self.alive(pid, start) is not False or self.group_alive(pgid)


@dataclass(frozen=True)
class Agent:
    session_id: str
    id: str | None
    name: str | None
    cwd: str | None
    kind: str | None
    status: str | None
    state: str | None
    pid: int | None
    started_at_ms: int | None

    def started_after(self, iso_stamp: str | None) -> bool:
        if iso_stamp is None or self.started_at_ms is None:
            return False
        stamp = datetime.strptime(iso_stamp, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
        return self.started_at_ms > stamp.timestamp() * 1000


def _opt_str(entry: dict[str, Any], key: str) -> str | None:
    value = entry.get(key)
    return None if value is None else str(value)


PID_MAX = 2**31  # pid_t is 32-bit signed: anything above is garbage (and overflows os.kill)


def _opt_int(entry: dict[str, Any], key: str, upper: int | None = None) -> int | None:
    value = entry.get(key)
    ok = isinstance(value, int) and not isinstance(value, bool) and value > 0 and (upper is None or value < upper)
    return value if ok else None


class ClaudeCli:
    def __init__(self, run: runner.Runner | None = None, exe: str = "claude") -> None:
        self.runner = run or runner.Runner()
        self.exe = exe

    def agents(self, timeout: float = 30) -> list[Agent]:
        res = self.runner.run([self.exe, "agents", "--json"], timeout=timeout)
        if res.returncode != 0:
            raise errors.CpError(
                "INTERNAL", f"claude agents --json exited {res.returncode}: {res.stderr.strip()[-200:]}"
            )
        try:
            data = json.loads(res.stdout)
        except ValueError as exc:
            raise errors.CpError("INTERNAL", f"claude agents --json: unparseable output ({exc})") from exc
        if not isinstance(data, list) or not all(
            isinstance(e, dict) and isinstance(e.get("sessionId"), str) for e in data
        ):
            raise errors.CpError("INTERNAL", "claude agents --json: unexpected shape")
        return [
            Agent(
                session_id=e["sessionId"],
                id=_opt_str(e, "id"),
                name=_opt_str(e, "name"),
                cwd=_opt_str(e, "cwd"),
                kind=_opt_str(e, "kind"),
                status=_opt_str(e, "status"),
                state=_opt_str(e, "state"),
                pid=_opt_int(e, "pid", PID_MAX),
                started_at_ms=_opt_int(e, "startedAt"),
            )
            for e in data
        ]

    def try_agents(self, timeout: float = 30) -> list[Agent] | None:
        """The listing, or ``None`` when it failed (unknown — never absence)."""
        try:
            return self.agents(timeout=timeout)
        except errors.CpError:
            return None

    def spawn_bg(
        self, args: Sequence[str], *, cwd: Path, timeout: float = 120, env_remove: Sequence[str] = ()
    ) -> runner.RunResult:
        """``claude --bg <args>``, detached, with the run token (and ``env_remove``) removed from its environment."""
        return self.runner.run(
            [self.exe, "--bg", *args], cwd=cwd, env_remove=(TOKEN_ENV, *env_remove), detach=True, timeout=timeout
        )

    def resume_bg(self, session_id: str, message: str, *, cwd: Path) -> runner.RunResult:
        """The bare ``claude --bg --resume <id> "<msg>"`` — any extra flag forks a copy."""
        return self.runner.run(
            [self.exe, "--bg", "--resume", session_id, message],
            cwd=cwd,
            env_remove=(TOKEN_ENV,),
            detach=True,
            timeout=120,
        )

    def stop(self, agent_id: str) -> runner.RunResult:
        """``claude stop <id>`` — stops a background session (its conversation is kept)."""
        return self.runner.run([self.exe, "stop", agent_id], timeout=30)


def find(listing: Sequence[Agent], *, session_id: str | None = None, name: str | None = None) -> Agent | None:
    for agent in listing:
        if session_id is not None and agent.session_id == session_id:
            return agent
        if name is not None and agent.name == name:
            return agent
    return None
