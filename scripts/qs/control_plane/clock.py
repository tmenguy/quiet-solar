"""The injected clock: ``SystemClock`` in production, ``FakeClock`` in tests.

Timestamps are stored as fixed-width UTC ISO-8601 strings (``iso``), so SQL
string comparisons order them correctly.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Protocol

_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


class Clock(Protocol):
    def now(self) -> datetime: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """The real clock."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class FakeClock:
    """A deterministic clock: ``sleep`` advances time instead of blocking."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 10, 3, 12, 0, 0, tzinfo=UTC)
        self._lock = threading.Lock()
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += timedelta(seconds=seconds)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)


def iso(dt: datetime) -> str:
    """Fixed-width UTC string for ``dt``."""
    return dt.astimezone(UTC).strftime(_FORMAT)


def parse(text: str) -> datetime:
    """Inverse of ``iso``."""
    return datetime.strptime(text, _FORMAT).replace(tzinfo=UTC)


def stamp(clock: Clock, *, plus: float = 0.0) -> str:
    """``iso(clock.now() + plus seconds)``."""
    return iso(clock.now() + timedelta(seconds=plus))


def age(clock: Clock, text: str | None) -> float | None:
    """Seconds elapsed since the stored timestamp ``text`` (``None`` if absent)."""
    if text is None:
        return None
    return (clock.now() - parse(text)).total_seconds()
