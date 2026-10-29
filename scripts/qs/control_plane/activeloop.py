"""The active loop's built-in tick hooks and their seams (QS-406 §3, D9, D17).

``register_builtin()`` registers every built-in hook on ``ticks`` (idempotent; ``cli`` calls it at
import and in ``main``, so a test reset makes the registry whole again). The hook modules are
imported lazily, inside ``_builtin()``, so they can import this module at the top for ``seams()``.

A hook takes its seams from ``seams()``: ``make_seams()``, built once per daemon. The tests replace
``make_seams`` with fakes over the test ``Deps``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    popen: Callable[..., Any] | None = None  # #407: the review pilot's detached launcher (None: it launches nothing)


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
        popen=subprocess.Popen,
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
    from . import board, ciwatch, detectors, hookroute, restore, reviewpilot, selfcheck, watchdog

    return [
        (selfcheck.CODE_VERSION, selfcheck.code_version_hook),
        (selfcheck.SELFCHECK, selfcheck.selfcheck_hook),
        (detectors.DETECTORS, detectors.detectors_hook),
        (reviewpilot.REVIEW_PILOT, reviewpilot.review_pilot_hook),  # #407: after the detectors raised their alerts
        (hookroute.HOOK_ROUTE, hookroute.hook_route_hook),
        (ciwatch.CI_WATCH, ciwatch.ci_watch_hook),
        (watchdog.LIVENESS_WATCHDOG, watchdog.liveness_watchdog_hook),
        (restore.BACKUP, restore.backup_hook),
        (board.BOARD, board.board_hook),  # #428: last, so the page shows what every other hook wrote
    ]


# Spelled out on purpose, not derived from `_builtin()`: the self-check compares the live registry with
# this independent list (a hook lost from the registry is caught), and deriving it would import every hook
# module when this module loads — the import cycle `_builtin()`'s lazy imports avoid. A test pins the two equal.
BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "code_version",
        "selfcheck",
        "detectors",
        "review_pilot",
        "hook_route",
        "ci_watch",
        "liveness_watchdog",
        "backup",
        "board",
    }
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
    from . import backups, board, ciwatch, detectors, restore, reviewpilot, watchdog

    backups._reset_for_tests()
    board._reset_for_tests()
    reviewpilot._reset_for_tests()
    restore._reset_for_tests()
    detectors._reset_for_tests()
    ciwatch._reset_for_tests()
    watchdog._reset_for_tests()
