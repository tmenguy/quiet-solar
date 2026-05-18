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

Robustness (review fix #01 should-fix #16): module purge uses
``Path.is_relative_to`` rather than naive substring matching so an
unrelated module whose ``__file__`` happens to contain the
``scripts/qs/`` substring is not accidentally evicted. The
``sys.path`` removal is also guarded so a sibling fixture that
already popped the entry can't raise ``ValueError`` here.
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
    # Resolve once so the per-module path comparison is symmetric with
    # the fixture's setup. ``SCRIPTS_QS_DIR`` is already resolved at
    # module load via ``Path(__file__).resolve().parents[...]``, so
    # this is effectively a re-pin against future refactoring.
    scripts_qs_resolved = SCRIPTS_QS_DIR
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
            # Resolve the module path and use ``is_relative_to`` so a
            # naive substring match can't evict unrelated modules
            # whose path happens to contain the scripts-qs path
            # string (review fix #01 should-fix #16).
            try:
                mod_path = Path(mod_file).resolve()
            except (OSError, RuntimeError):
                continue
            if mod_path.is_relative_to(scripts_qs_resolved):
                sys.modules.pop(name, None)
        # Guard the removal in case a parallel fixture already popped
        # the entry — without the guard, ``list.remove`` raises
        # ``ValueError`` and aborts teardown.
        if added and path_str in sys.path:
            sys.path.remove(path_str)
