"""In-process unit tests for ``scripts/qs/next_step.py``.

Subprocess tests in ``test_next_step_cli.py`` cover the happy path and
JSON error contract. These tests verify behaviour that's awkward to
exercise via subprocess — specifically, the narrowing of the
``except`` clause to ``UnknownPhaseError`` only (review-fix #02 SF1).

Importing ``next_step`` in-process is safe under the launcher conftest
which adds ``scripts/qs/`` to ``sys.path`` and tears it down per-test.
"""

from __future__ import annotations

import sys

import pytest


def _set_argv(monkeypatch: pytest.MonkeyPatch, harness: str, next_cmd: str = "create-plan") -> None:
    """Install a minimal valid sys.argv for ``next_step.main()``."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "next_step.py",
            "--next-cmd", next_cmd,
            "--work-dir", "/tmp/work",
            "--issue", "1",
            "--title", "Title",
            "--harness", harness,
        ],
    )


def test_next_step_propagates_non_phase_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-phase ``ValueError`` from build_payload propagates uncaught.

    Before the SF1 fix, ``next_step.py`` caught every ``ValueError`` and
    misreported it as 'unknown phase'. After the fix, only
    ``UnknownPhaseError`` should be caught; everything else should
    propagate (a non-zero exit with the original traceback).
    """
    import next_step  # type: ignore[import-not-found]

    class FaultyLauncher:
        @staticmethod
        def build_payload(*_args: object, **_kwargs: object) -> dict:
            raise ValueError("totally unrelated failure mode")

    monkeypatch.setitem(next_step.LAUNCHERS, "claude-code", FaultyLauncher)
    _set_argv(monkeypatch, harness="claude-code")

    with pytest.raises(ValueError) as excinfo:
        next_step.main()

    # The original message survives — proving we did not silently
    # rewrite it as 'unknown phase'.
    assert "unrelated failure mode" in str(excinfo.value)


def test_next_step_catches_unknown_phase_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """``UnknownPhaseError`` is caught and JSON-formatted as exit 1."""
    import json

    import next_step  # type: ignore[import-not-found]
    from launchers.phases import UnknownPhaseError  # type: ignore[import-not-found]

    class PhaseRejectingLauncher:
        @staticmethod
        def build_payload(*_args: object, **_kwargs: object) -> dict:
            raise UnknownPhaseError("synthetic-bad", ["a", "b"])

    monkeypatch.setitem(next_step.LAUNCHERS, "claude-code", PhaseRejectingLauncher)
    _set_argv(monkeypatch, harness="claude-code")

    with pytest.raises(SystemExit) as excinfo:
        next_step.main()
    assert excinfo.value.code == 1

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["error"] == "unknown phase"
    assert payload["value"] == "synthetic-bad"
    assert payload["known"] == ["a", "b"]


def test_next_step_exits_zero_on_happy_path(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Sanity: happy path still exits 0 after SF1 refactor."""
    import json

    import next_step  # type: ignore[import-not-found]

    class HappyLauncher:
        @staticmethod
        def build_payload(*_args: object, **_kwargs: object) -> dict:
            return {"tool": "fake", "same_context": "x", "new_context": "y"}

    monkeypatch.setitem(next_step.LAUNCHERS, "claude-code", HappyLauncher)
    _set_argv(monkeypatch, harness="claude-code")

    with pytest.raises(SystemExit) as excinfo:
        next_step.main()
    assert excinfo.value.code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "fake"
    assert payload["harness"] == "claude-code"
