"""The active loop's built-in tick hooks and their seams (QS-406 §3, D9, D17).

``register_builtin()`` registers every built-in hook on ``ticks`` (idempotent; ``cli`` calls it at
import and in ``main``, so a test reset makes the registry whole again). The hook modules are
imported lazily, inside ``_builtin()``, so they can import this module at the top for ``seams()``.

A hook takes its seams from ``seams()``: ``make_seams()``, built once per daemon. The tests replace
``make_seams`` with fakes over the test ``Deps``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import liveness, paths, runner, ticks

if TYPE_CHECKING:
    from .ciwatch import GitHub


@dataclass(frozen=True)
class Seams:
    """Every side-effecting dependency a built-in hook uses."""

    runner: runner.Runner
    probe: liveness.ProcessProbe
    claude: liveness.ClaudeCli
    main: Path
    github: GitHub


def make_seams() -> Seams:
    from . import ciwatch

    run = runner.Runner()
    main = paths.main()
    return Seams(
        runner=run,
        probe=liveness.ProcessProbe(run),
        claude=liveness.ClaudeCli(run),
        main=main,
        github=ciwatch.GitHub(run, main),
    )


_seams: Seams | None = None


def seams() -> Seams:
    """``make_seams()``, cached for the life of the daemon."""
    global _seams
    if _seams is None:
        _seams = make_seams()
    return _seams


def _builtin() -> list[tuple[str, ticks.Hook]]:
    """The built-in hooks, in tick order (each task adds its own, and its name to ``BUILTIN_NAMES``)."""
    from . import ciwatch, detectors, hookroute, selfcheck, watchdog

    return [
        (selfcheck.CODE_VERSION, selfcheck.code_version_hook),
        (selfcheck.SELFCHECK, selfcheck.selfcheck_hook),
        (detectors.DETECTORS, detectors.detectors_hook),
        (hookroute.HOOK_ROUTE, hookroute.hook_route_hook),
        (ciwatch.CI_WATCH, ciwatch.ci_watch_hook),
        (watchdog.LIVENESS_WATCHDOG, watchdog.liveness_watchdog_hook),
    ]


BUILTIN_NAMES: frozenset[str] = frozenset(
    {"code_version", "selfcheck", "detectors", "hook_route", "ci_watch", "liveness_watchdog"}
)


def register_builtin() -> None:
    """Register every built-in hook not registered yet."""
    present = {name for name, _ in ticks.registered()}
    for name, fn in _builtin():
        if name not in present:
            ticks.register(name, fn)


def _reset_for_tests() -> None:
    """Drop the cached seams and every hook module's in-memory state (a new daemon)."""
    global _seams
    _seams = None
    from . import ciwatch, detectors, watchdog

    detectors._reset_for_tests()
    ciwatch._reset_for_tests()
    watchdog._reset_for_tests()
