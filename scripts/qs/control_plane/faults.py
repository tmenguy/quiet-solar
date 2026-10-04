"""Named fault points for crash tests.

``hit(name)`` is a no-op unless a test armed ``name`` with ``arm(name, exc)``.
``FaultInjected`` derives from ``BaseException`` so that no ``except
Exception`` handler mistakes a simulated crash for an ordinary failure.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

_ARMED: dict[str, BaseException] = {}
_SKIP: dict[str, int] = {}


class FaultInjected(BaseException):
    """A simulated crash."""


def hit(name: str) -> None:
    exc = _ARMED.get(name)
    if exc is None:
        return
    if _SKIP.get(name, 0) > 0:
        _SKIP[name] -= 1
        return
    raise exc


@contextmanager
def arm(name: str, exc: BaseException | None = None, *, skip: int = 0) -> Iterator[None]:
    """Fire ``name`` on its ``skip + 1``-th hit (and every hit after) while armed."""
    _ARMED[name] = exc if exc is not None else FaultInjected(name)
    _SKIP[name] = skip
    try:
        yield
    finally:
        _ARMED.pop(name, None)
        _SKIP.pop(name, None)


def reset() -> None:
    _ARMED.clear()
    _SKIP.clear()
