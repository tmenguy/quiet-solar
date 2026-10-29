"""Tests for ``scripts/qs/live_probe.py`` — the throwaway live-check probe (#437 / #442).

The autouse ``_add_scripts_qs_to_syspath`` fixture in ``tests/qs/conftest.py``
puts ``scripts/qs/`` on ``sys.path``, so ``import live_probe`` works in-process
(visible to coverage). Every path of ``main`` (no argument, ``--version``, any
other argument) and the ``__main__`` guard are executed here — no ``# pragma: no cover`` anywhere in the script.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_PROBE_PATH = REPO_ROOT / "scripts" / "qs" / "live_probe.py"


def test_main_without_argument_prints_ok_and_returns_0(capsys: pytest.CaptureFixture[str]) -> None:
    import live_probe

    assert live_probe.main([]) == 0
    captured = capsys.readouterr()
    assert captured.out == "live-probe ok\n"
    assert captured.err == ""


def test_main_with_version_prints_version_and_returns_0(capsys: pytest.CaptureFixture[str]) -> None:
    import live_probe

    assert live_probe.main(["--version"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "live-probe 1\n"
    assert captured.err == ""


@pytest.mark.parametrize("argv", [["--bogus"], ["ok"], ["--version", "extra"], ["-v"]])
def test_main_with_any_other_argument_returns_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    import live_probe

    assert live_probe.main(argv) == 2
    assert capsys.readouterr().out == ""


def test_script_entry_point_exits_with_main_return_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``python scripts/qs/live_probe.py`` exits via ``sys.exit(main(sys.argv[1:]))``."""
    assert LIVE_PROBE_PATH.is_file(), f"live_probe.py not found at {LIVE_PROBE_PATH} — has the script moved?"
    monkeypatch.setattr(sys, "argv", [str(LIVE_PROBE_PATH)])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(LIVE_PROBE_PATH), run_name="__main__")

    assert excinfo.value.code == 0
    assert capsys.readouterr().out == "live-probe ok\n"
