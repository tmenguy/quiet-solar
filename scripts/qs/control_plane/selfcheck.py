"""The lifecycle hooks (QS-406 §4): restart on new code (``code_version``).

``code_version`` (every tick): the disk version of the main checkout is
compared with the version the daemon loaded. A change must read the same on
two consecutive ticks, with no git operation in progress in the main
checkout, before the daemon is asked to restart (D7): it then finishes its
tick, exits ``new_code``, and ``cli._daemon`` starts the new code.
"""

from __future__ import annotations

import sqlite3

from . import activeloop, codever, daemon
from . import clock as clock_mod

CODE_VERSION = "code_version"


def code_version_hook(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
    loaded = daemon.started_code_version()
    if loaded is None:
        return
    main = activeloop.seams().main
    current = codever.code_version(main)
    if current == loaded or codever.git_busy(main) is not None:
        daemon.set_restart_candidate(None)  # unchanged, or mid-operation: the debounce starts over
        return
    if daemon.restart_candidate() == current:
        daemon.request_restart("new_code")
        return
    daemon.set_restart_candidate(current)
