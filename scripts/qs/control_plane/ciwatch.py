"""The CI watcher (QS-406 §8): every open PR's rollup, within GitHub's API budget.

- ``GitHub`` (on ``seams.runner``): ``repo()`` parses ``git remote get-url origin`` once;
  ``prs(numbers)`` makes one ``gh api graphql`` call per ``CI_BATCH`` PRs. GraphQL may return
  ``errors`` with partial ``data`` (``gh`` then exits non-zero): a ``null`` alias is an unknown PR
  (``None``), every other alias is valid; only a call with no parseable ``data`` is a ``CiFailure``.
  An alias whose answer cannot be parsed is unknown too (logged once), and a ``null`` context is skipped.
- Watched PRs: deliverables with a ``pr_number``, not terminal, in an open run (work items skipped).
- A poll is all-or-nothing for transport failures: ``tasks.ci_state`` / ``ci_sha`` are written, on a
  change only, when every batch returned data. ``MERGED`` / ``CLOSED`` → ``NULL`` (clears ``ci_red``).
- ``ci_red`` is derived from the stored rows on every tick (``must-fix``), whether or not a poll ran.
- The schedule: ``CI_FAST_S`` after a poll that saw a pending rollup or a moved head, else
  ``CI_SLOW_S``; failures back off from ``CI_BACKOFF_MIN_S`` to ``CI_BACKOFF_MAX_S``; below
  ``CI_RATE_FLOOR`` remaining calls, no poll before GitHub's ``resetAt`` (at most ``CI_RESET_CAP_S``
  away). A new daemon polls at once.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import activeloop, alerts, daemon, db, runner, tasks, ticks
from . import clock as clock_mod

if TYPE_CHECKING:
    from collections.abc import Sequence

CI_WATCH = "ci_watch"
CI_FAST_S = 30.0
CI_SLOW_S = 300.0
CI_BACKOFF_MIN_S = 60.0
CI_BACKOFF_MAX_S = 900.0
CI_BATCH = 25
CI_RATE_FLOOR = 200
CI_RESET_CAP_S = 3600.0  # GitHub's rate window is one hour: a later `resetAt` is not believed

FAILING_CONCLUSIONS = frozenset({"FAILURE", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED", "STARTUP_FAILURE"})
FAILING_STATUSES = frozenset({"FAILURE", "ERROR"})
PENDING_ROLLUPS = frozenset({"PENDING", "EXPECTED"})
CLOSED_STATES = frozenset({"MERGED", "CLOSED"})

_REMOTES = (
    re.compile(r"^https://github\.com/([^/\s]+)/([^/\s]+?)(?:\.git)?/?$"),
    re.compile(r"^git@github\.com:([^/\s]+)/([^/\s]+?)(?:\.git)?$"),
    re.compile(r"^ssh://git@github\.com/([^/\s]+)/([^/\s]+?)(?:\.git)?/?$"),
)

_PR_FIELDS = (
    "state headRefOid commits(last:1){ nodes{ commit{ statusCheckRollup{ state"
    " contexts(first:100){ pageInfo{hasNextPage} nodes{ __typename"
    " ... on CheckRun{ name conclusion detailsUrl } ... on StatusContext{ context state targetUrl } } } } } } }"
)


def _log(message: str) -> None:
    sys.stderr.write(f"[cp-ciwatch] {message}\n")
    sys.stderr.flush()


class CiFailure(Exception):
    """The poll's input is unknown (transport, auth, rate, an unparseable answer or remote)."""

    def __init__(self, message: str, *, quiet: bool = False) -> None:
        super().__init__(message)
        self.quiet = quiet  # already logged once (a permanent failure)


@dataclass(frozen=True)
class PrCi:
    state: str
    head: str | None
    rollup: str | None
    failing: tuple[str, ...]
    truncated: bool


@dataclass(frozen=True)
class Rate:
    remaining: int
    reset_at: str | None


def parse_remote(url: str) -> tuple[str, str] | None:
    for pattern in _REMOTES:
        m = pattern.match(url.strip())
        if m:
            return m.group(1), m.group(2)
    return None


def query(owner: str, name: str, numbers: Sequence[int]) -> str:
    aliases = " ".join(f"p{n}: pullRequest(number: {int(n)}) {{ {_PR_FIELDS} }}" for n in numbers)
    return (
        f"query {{ rateLimit {{ remaining resetAt }} repository(owner: {json.dumps(owner)}, name: {json.dumps(name)})"
        f" {{ {aliases} }} }}"
    )


def _pr(node: dict[str, Any]) -> PrCi:
    commits = node["commits"]["nodes"]
    rollup = commits[0]["commit"]["statusCheckRollup"] if commits else None
    if rollup is None:
        return PrCi(node["state"], node["headRefOid"], None, (), False)
    failing = []
    for ctx in rollup["contexts"]["nodes"]:
        if not isinstance(ctx, dict):
            continue  # GraphQL list items are nullable
        if ctx.get("__typename") == "CheckRun" and ctx.get("conclusion") in FAILING_CONCLUSIONS:
            failing.append(str(ctx.get("name")))
        elif ctx.get("__typename") == "StatusContext" and ctx.get("state") in FAILING_STATUSES:
            failing.append(str(ctx.get("context")))
    truncated = bool(rollup["contexts"]["pageInfo"]["hasNextPage"])
    return PrCi(node["state"], node["headRefOid"], rollup["state"], tuple(failing), truncated)


class GitHub:
    """GitHub through ``gh`` and ``git``, on the daemon's runner."""

    def __init__(self, run: runner.Runner, main: Path) -> None:
        self._runner = run
        self._main = main
        self._repo: tuple[str, str] | None = None
        self._repo_error: str | None = None
        self._unparseable: set[int] = set()  # PRs already logged as unparseable

    def repo(self) -> tuple[str, str]:
        if self._repo is not None:
            return self._repo
        if self._repo_error is not None:
            raise CiFailure(self._repo_error, quiet=True)
        res = self._runner.run(
            ["git", "-C", str(self._main), "remote", "get-url", "origin"],
            cwd=self._main,
            timeout=ticks.HOOK_SUBPROCESS_S,
        )
        parsed = parse_remote(res.stdout) if res.ok else None
        if parsed is None:
            if not res.ok:
                raise CiFailure(f"git remote get-url origin exited {res.returncode}")
            self._repo_error = f"origin is not a GitHub remote: {res.stdout.strip()!r}"
            _log(self._repo_error)
            raise CiFailure(self._repo_error, quiet=True)
        self._repo = parsed
        return parsed

    def prs(self, numbers: Sequence[int]) -> tuple[dict[int, PrCi | None], Rate]:
        owner, name = self.repo()
        res = self._runner.run(
            ["gh", "api", "graphql", "-f", f"query={query(owner, name, numbers)}"],
            cwd=self._main,
            timeout=ticks.HOOK_SUBPROCESS_S,
        )
        try:
            doc = json.loads(res.stdout)
            data = doc["data"]
            rate = Rate(int(data["rateLimit"]["remaining"]), data["rateLimit"].get("resetAt"))
            repository = data["repository"]
            out: dict[int, PrCi | None] = {}
            for n in numbers:
                out[n] = self._parse(n, repository.get(f"p{n}"))
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            raise CiFailure(
                f"gh api graphql exited {res.returncode}: {type(exc).__name__}: {res.stderr.strip()[-200:]}"
            ) from exc
        return out, rate

    def _parse(self, number: int, node: Any) -> PrCi | None:
        """One alias's answer → its ``PrCi``; ``None`` (unknown this poll) when null or unparseable."""
        if node is None:
            return None
        try:
            pr = _pr(node)
        except (KeyError, TypeError, IndexError, AttributeError) as exc:
            if number not in self._unparseable:
                self._unparseable.add(number)
                _log(f"PR #{number}: unparseable answer ({type(exc).__name__}: {exc}); unknown until it parses")
            return None
        self._unparseable.discard(number)  # parses again: a later breakage is logged again
        return pr


def ci_state(pr: PrCi) -> tuple[str | None, str | None]:
    """The ``(ci_state, ci_sha)`` to store for ``pr``."""
    if pr.state in CLOSED_STATES:
        return None, None
    if pr.rollup == "SUCCESS":
        return "green", pr.head
    if pr.rollup in ("FAILURE", "ERROR"):
        return "red", pr.head
    return "pending", pr.head


@dataclass
class CiWatcher:
    next_at: datetime | None = None  # None: due now (a new daemon polls at once)
    backoff_s: float = 0.0
    failing: dict[str, tuple[str | None, tuple[str, ...], bool]] = field(default_factory=dict)

    def _watched(self, conn: sqlite3.Connection) -> list[sqlite3.Row]:
        marks = ", ".join("?" for _ in tasks.TERMINAL)
        return conn.execute(
            f"SELECT t.id, t.run_id, t.pr_number, t.ci_state, t.ci_sha FROM tasks t JOIN runs r ON r.id = t.run_id"
            f" WHERE r.state = 'open' AND t.is_deliverable = 1 AND t.pr_number IS NOT NULL"
            f" AND t.state NOT IN ({marks}) ORDER BY t.id",
            tuple(sorted(tasks.TERMINAL)),
        ).fetchall()

    def poll(self, conn: sqlite3.Connection, clock: clock_mod.Clock, github: GitHub) -> None:
        now = clock.now()
        if self.next_at is not None and now < self.next_at:
            return
        rows = self._watched(conn)
        if not rows:
            self.next_at = now + timedelta(seconds=CI_SLOW_S)
            return
        numbers = sorted({int(r["pr_number"]) for r in rows})
        results: dict[int, PrCi | None] = {}
        rate: Rate | None = None
        try:
            for i in range(0, len(numbers), CI_BATCH):
                part, rate = github.prs(numbers[i : i + CI_BATCH])
                daemon.beat(conn, clock)
                results.update(part)
        except CiFailure as exc:
            if not exc.quiet:
                _log(f"poll failed: {exc}")
            self.backoff_s = min(max(self.backoff_s * 2, CI_BACKOFF_MIN_S), CI_BACKOFF_MAX_S)
            self.next_at = now + timedelta(seconds=self.backoff_s)
            return
        self.backoff_s = 0.0
        fast = False
        with db.write(conn):
            for row in rows:
                pr = results.get(int(row["pr_number"]))
                if pr is None:
                    continue  # unknown this poll (a NOT_FOUND alias): no write
                state, sha = ci_state(pr)
                if state is not None and (pr.rollup in PENDING_ROLLUPS or sha != row["ci_sha"]):
                    fast = True
                self.failing[row["id"]] = (sha, pr.failing, pr.truncated)
                if (state, sha) != (row["ci_state"], row["ci_sha"]):
                    tasks.update_fields(conn, clock, row["id"], {"ci_state": state, "ci_sha": sha})
        watched = {row["id"] for row in rows}
        self.failing = {k: v for k, v in self.failing.items() if k in watched}  # a task no longer watched
        self.next_at = now + timedelta(seconds=CI_FAST_S if fast else CI_SLOW_S)
        if rate is not None and rate.remaining < CI_RATE_FLOOR and rate.reset_at:
            try:
                reset = datetime.fromisoformat(rate.reset_at)
            except ValueError:
                reset = now + timedelta(seconds=CI_SLOW_S)
            self.next_at = max(self.next_at, min(reset, now + timedelta(seconds=CI_RESET_CAP_S)))

    def ci_red(self, conn: sqlite3.Connection) -> list[alerts.Condition]:
        out = []
        for row in self._watched(conn):
            if row["ci_state"] != "red":
                continue
            sha, failing, truncated = self.failing.get(row["id"], (row["ci_sha"], (), False))
            payload = {
                "task_id": row["id"],
                "pr_number": row["pr_number"],
                "sha": row["ci_sha"],
                "failing": list(failing) if sha == row["ci_sha"] else [],
                "truncated": truncated if sha == row["ci_sha"] else False,
                "severity": "must-fix",
            }
            subject = f"{row['id']}@{row['ci_sha'] or '-'}"  # `ci_sha` is nullable
            out.append(alerts.Condition(alerts.CI_RED, subject, (row["run_id"],), payload))
        return out


_watcher = CiWatcher()


def ci_watch_hook(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
    _watcher.poll(conn, clock, activeloop.seams().github)
    alerts.sync(conn, clock, kinds={alerts.CI_RED}, active=_watcher.ci_red(conn))


def _reset_for_tests() -> None:
    global _watcher
    _watcher = CiWatcher()
