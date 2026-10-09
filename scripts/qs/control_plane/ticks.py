"""The daemon's tick-hook registration point (QS-406 §2).

A tick hook is ``fn(conn, clock)``: ``conn`` is the daemon's single long-lived
connection. ``cli`` imports ``activeloop``, whose ``register_builtin()`` puts
every built-in hook here; ``cli._daemon`` passes ``hooks()`` to ``daemon.run``.

**The contract** every hook keeps:

- never let ``daemon.STALE_AFTER_S`` (30 s) pass without a beat;
- give every subprocess a timeout of at most ``HOOK_SUBPROCESS_S``;
- call ``daemon.beat(conn, clock)`` after every subprocess call;
- run subprocesses and ``beat`` outside any ``db.write``: read, release,
  call, beat, write;
- bound the work done per tick;
- never launch a long job in-process;
- throttle with a ``Throttle``, or with a DB timestamp.

A hook that raises is logged and skipped by the daemon.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime

from . import clock as clock_mod

HOOK_SUBPROCESS_S = 20.0

Hook = Callable[[sqlite3.Connection, clock_mod.Clock], None]

_REGISTRY: dict[str, Hook] = {}


class Throttle:
    """In memory: due at once in a new daemon, then once every ``every_s``."""

    def __init__(self, every_s: float) -> None:
        self.every_s = every_s
        self._last: datetime | None = None
        _THROTTLES.append(self)

    def due(self, clock: clock_mod.Clock) -> bool:
        """``True`` (and re-armed) when ``every_s`` has passed since the last ``True``."""
        now = clock.now()
        if self._last is not None and (now - self._last).total_seconds() < self.every_s:
            return False
        self._last = now
        return True

    def rearm(self) -> None:
        self._last = None


_THROTTLES: list[Throttle] = []


def register(name: str, fn: Hook) -> None:
    """Add a tick hook under ``name``; ``ValueError`` on a duplicate name. See the module's contract."""
    if name in _REGISTRY:
        raise ValueError(f"tick hook {name!r} is already registered")
    _REGISTRY[name] = fn


def registered() -> tuple[tuple[str, Hook], ...]:
    return tuple(_REGISTRY.items())


def _named(name: str, fn: Hook) -> Hook:
    def hook(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
        fn(conn, clock)

    hook.__name__ = hook.__qualname__ = name  # the daemon logs a failing hook by name
    return hook


def hooks() -> list[Hook]:
    """Every registered hook, in registration order, each wrapped in a closure named after it."""
    return [_named(name, fn) for name, fn in _REGISTRY.items()]


def _reset_for_tests() -> None:
    """Clear the registry and re-arm every ``Throttle`` (they stay registered)."""
    _REGISTRY.clear()
    for throttle in _THROTTLES:
        throttle.rearm()
