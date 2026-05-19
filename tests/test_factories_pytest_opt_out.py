"""Regression guard for pytest-collection opt-out on `tests/factories.py` helpers.

Pytest's default `python_classes = Test*` rule (set in `pytest.ini`)
collects every class named `Test...`.  Our doubles in `tests/factories.py`
match that pattern but are helpers, not test classes — they carry
`__init__` constructors.  Without an opt-out, pytest emits
`cannot collect test class '...' because it has a __init__ constructor`.

This module fails fast if any class in `tests/factories.py` whose name
starts with `Test` lacks `__test__ = False`, and also asserts the three
currently sanctioned doubles are still present and opted out — so a
silent rename or accidental removal is caught.
"""

from __future__ import annotations

import inspect

from tests import factories

# The three test-double helpers currently defined in tests/factories.py.
# Update this set when intentionally adding/removing a sanctioned double.
_SANCTIONED_DOUBLES: frozenset[str] = frozenset(
    {"TestCarDouble", "TestChargerDouble", "TestDynamicGroupDouble"}
)


def _local_test_classes() -> list[tuple[str, type]]:
    """Return [(name, cls)] of every `Test*` class defined directly in factories.py."""
    return [
        (name, obj)
        for name, obj in inspect.getmembers(factories, inspect.isclass)
        if name.startswith("Test") and inspect.getmodule(obj) is factories
    ]


def test_all_test_doubles_opt_out_of_collection() -> None:
    """Every `Test*` class in factories.py must set `__test__ = False`."""
    offenders = [
        name for name, obj in _local_test_classes()
        if getattr(obj, "__test__", True) is not False
    ]
    assert not offenders, (
        "These factories.py helpers start with `Test` but do not set "
        "`__test__ = False`.  Add it so pytest skips collection:\n"
        + "\n".join(f"  - {n}" for n in offenders)
    )


def test_sanctioned_doubles_still_present_and_opted_out() -> None:
    """The three known doubles must remain in factories.py with `__test__ = False`."""
    present = {name for name, _ in _local_test_classes()}
    missing = _SANCTIONED_DOUBLES - present
    assert not missing, (
        f"Expected doubles missing from tests/factories.py: {sorted(missing)}.  "
        f"If you intentionally renamed/removed one, update _SANCTIONED_DOUBLES."
    )
    for name in _SANCTIONED_DOUBLES & present:
        cls = getattr(factories, name)
        assert getattr(cls, "__test__", True) is False, (
            f"{name} is present but does not set `__test__ = False`."
        )
