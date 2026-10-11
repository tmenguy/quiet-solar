"""The Control Plane (QS-399): runtime state, run queues, commands, tools and hooks.

Python, no LLM, one for all runs. Every session calls
``<MAIN>/venv/bin/python <MAIN>/scripts/qs/cp.py <command> …`` — see
``docs/workflow/control-plane.md``.
"""

import time

IMPORTED_AT_NS = time.time_ns()  # QS-406 D8: before any submodule import (codever reads it at call time)

from .migrations import SCHEMA_VERSION

PACKAGE = "control_plane"

__all__ = ["IMPORTED_AT_NS", "PACKAGE", "SCHEMA_VERSION"]
