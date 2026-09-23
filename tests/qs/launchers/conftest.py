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
    #
    # Limitation (review-fix #03 NTH6): this only purges modules imported
    # BY OR AFTER our autouse fixture. If a higher-level conftest
    # pre-imported ``next_step`` or ``launchers.*`` before we entered
    # here, they survive teardown. The remaining cross-test pollution is
    # bounded by ``monkeypatch`` (which reverts its own setattr/setitem
    # changes on teardown), so the practical exposure is small.
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


@pytest.fixture(autouse=True)
def _neutralize_cli_floor_guard(
    _add_scripts_qs_to_syspath: None, monkeypatch: pytest.MonkeyPatch,
) -> object:
    """Stop every ``build_payload`` call spawning the real ``claude --version``.

    ``build_payload`` calls ``_warn_if_cli_below_floor`` unconditionally
    (QS-367 S4), so without this ~55 launcher tests would each spawn the
    host CLI: slow on every host, and on one whose ``claude`` is below the
    floor an unrelated stderr ``warning:`` would break assertions that
    demand a clean stderr (e.g. the BOM test). Patch the **module
    attribute** to a no-op so ``build_payload``'s own lookup sees it
    (QS-367 S1).

    Returns the real function so the ``real_cli_floor_guard`` fixture can
    restore it for the three S4 end-to-end tests that must exercise the
    guard. Depends on ``_add_scripts_qs_to_syspath`` so ``launchers.claude``
    is importable when this runs.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    original = claude_launcher._warn_if_cli_below_floor
    monkeypatch.setattr(claude_launcher, "_warn_if_cli_below_floor", lambda: None)
    return original


@pytest.fixture
def real_cli_floor_guard(
    _neutralize_cli_floor_guard: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Undo the autouse no-op so a test exercises the real floor guard.

    The QS-367 S4 end-to-end tests
    (``test_build_payload_warns_on_old_cli``, its silent siblings, the
    stream-scanning and ordering tests) route ``build_payload`` through the
    real ``_warn_if_cli_below_floor`` and intercept only ``claude --version``
    via ``_patch_claude_version``. This fixture reinstalls the real function
    (captured by the autouse fixture) before they install their own
    ``subprocess.run`` fake.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    monkeypatch.setattr(
        claude_launcher, "_warn_if_cli_below_floor", _neutralize_cli_floor_guard,
    )
