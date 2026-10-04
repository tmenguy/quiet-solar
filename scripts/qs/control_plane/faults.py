"""Named fault points for crash tests.

``hit(name)`` is a no-op unless a test armed ``name`` with ``arm(name, exc)``.
``FaultInjected`` derives from ``BaseException`` so that no ``except
Exception`` handler mistakes a simulated crash for an ordinary failure.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

_ARMED: dict[str, BaseException] = {}


class FaultInjected(BaseException):
    """A simulated crash."""


def hit(name: str) -> None:
    exc = _ARMED.get(name)
    if exc is not None:
        raise exc


@contextmanager
def arm(name: str, exc: BaseException | None = None) -> Iterator[None]:
    _ARMED[name] = exc if exc is not None else FaultInjected(name)
    try:
        yield
    finally:
        _ARMED.pop(name, None)


def reset() -> None:
    _ARMED.clear()
