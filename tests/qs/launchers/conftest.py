"""Shared fixtures: put scripts/qs/ on sys.path so the launcher modules can
be imported as top-level (``launchers.claude`` etc.), matching how
``next_step.py`` and ``setup_task.py`` import them at runtime.

We tear sys.path back down after each test and pop any modules that came in
from the QS scripts dir, so a single pytest session can both import the
launcher modules in-process AND subprocess-invoke ``next_step.py`` (which
runs in its own interpreter and is unaffected).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

SCRIPTS_QS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "qs"


@pytest.fixture(autouse=True)
def _add_scripts_qs_to_syspath() -> Iterator[None]:
    """Make ``launchers.*`` and ``next_step`` importable as top-level."""
    added = False
    path_str = str(SCRIPTS_QS_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        added = True

    # Snapshot modules originating from scripts/qs/ so we can purge them after
    # the test and avoid cross-test pollution from module-level state.
    before = set(sys.modules)
    try:
        yield
    finally:
        new = set(sys.modules) - before
        for name in new:
            mod = sys.modules.get(name)
            # Namespace/frozen packages may carry ``__file__ is None``;
            # skip them outright instead of falling through to the
            # substring check on a bare empty string.
            mod_file = getattr(mod, "__file__", None)
            if not mod_file:
                continue
            if path_str in mod_file:
                sys.modules.pop(name, None)
        if added:
            sys.path.remove(path_str)
