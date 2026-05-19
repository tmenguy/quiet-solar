"""Shared test configuration for _qsprocess_opencode tests.

Adds the scripts/qs_opencode directory to sys.path so test modules can
import render_agent, next_step, cleanup_worktree, etc. directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve once relative to this file — works regardless of CWD
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts" / "qs_opencode")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
