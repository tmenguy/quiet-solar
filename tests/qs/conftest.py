"""Shared fixtures for the top-level ``tests/qs/`` subtree.

Mirrors :mod:`tests.qs.launchers.conftest`: inserts ``scripts/qs/`` onto
``sys.path`` so ``spawn_session`` and ``launchers.opencode`` (etc.) can
be imported as top-level modules — matching how ``next_step.py`` and
``setup_task.py`` import them at runtime.

The inner ``tests/qs/launchers/conftest.py`` keeps its own copy of this
pattern; both fixtures are idempotent (the ``added`` flag guards
against double-insertion / double-removal), so a test under
``tests/qs/launchers/`` gets the same effective sys.path whether one
or both fixtures fire.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

SCRIPTS_QS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "qs"


@pytest.fixture(autouse=True)
def _add_scripts_qs_to_syspath() -> Iterator[None]:
    """Make ``spawn_session`` and ``launchers.*`` importable as top-level."""
    added = False
    path_str = str(SCRIPTS_QS_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        added = True

    # Snapshot modules originating from scripts/qs/ so we can purge them
    # after the test and avoid cross-test pollution from module-level
    # state. Same idempotency caveat documented in the launchers
    # conftest applies.
    before = set(sys.modules)
    try:
        yield
    finally:
        new = set(sys.modules) - before
        for name in new:
            mod = sys.modules.get(name)
            mod_file = getattr(mod, "__file__", None)
            if not mod_file:
                continue
            if path_str in mod_file:
                sys.modules.pop(name, None)
        if added:
            sys.path.remove(path_str)
