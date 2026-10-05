"""The merge-policy seam (§9.5): child 7 installs the merge conditions.

Until then the default policy **refuses** every merge.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyResult:
    ok: bool
    reason: str


Policy = Callable[[sqlite3.Row], PolicyResult]


def _refuse(task: sqlite3.Row) -> PolicyResult:
    return PolicyResult(False, "merge conditions are child 7's")


_policy: Policy = _refuse


def install(fn: Policy) -> None:
    global _policy
    _policy = fn


def reset() -> None:
    install(_refuse)


def check(task: sqlite3.Row) -> PolicyResult:
    return _policy(task)
