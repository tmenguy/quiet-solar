"""The process-setup seam: process groups and signal handlers.

Only ``cli.main`` (for ``tool`` commands) and ``daemon.run`` / ``wait`` call
it, always through ``get()``, which the tests replace — so pytest's own
process never changes group and never gets a signal handler.
"""

from __future__ import annotations

import os
import signal
from collections.abc import Callable
from typing import Any


class ProcessSetup:
    def become_group_leader(self) -> None:
        """Make this process a group leader (a no-op if it already is one).

        A session leader is a group leader too, and ``setpgid`` would raise
        ``EPERM`` on it — hence the check first.
        """
        if os.getpgid(0) == os.getpid():
            return
        os.setpgid(0, 0)

    def install_sigterm(self, fn: Callable[[int, Any], None]) -> None:
        signal.signal(signal.SIGTERM, fn)


_SETUP = ProcessSetup()


def get() -> ProcessSetup:
    return _SETUP
