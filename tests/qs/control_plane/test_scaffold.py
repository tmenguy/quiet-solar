"""Checkpoint 1: errors, clock, faults, the CLI table and the ``cp.py`` shim."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from control_plane import cli, clock, errors, faults

from .conftest import SCRIPTS_QS


class TestErrors:
    def test_exit_codes_table(self) -> None:
        assert errors.EXIT_CODES == {
            "INTERNAL": 1,
            "TOOL_FAILED": 1,
            "USAGE": 2,
            "STALE_TOKEN": 3,
            "STOPPED": 4,
            "SCHEMA_TOO_NEW": 5,
            "SCHEMA_PENDING": 5,
            "BUSY": 6,
            "PATH_GUARD": 7,
            "NOT_FOUND": 8,
            "CONFLICT": 8,
            "INVALID_STATE": 8,
            "POLICY_REFUSED": 9,
        }

    def test_payload_and_exit_code(self) -> None:
        exc = errors.CpError("BUSY", "held", holder="x")
        assert exc.exit_code == 6
        assert exc.payload() == {"ok": False, "error": "BUSY", "detail": "held", "holder": "x"}
        assert str(exc) == "BUSY: held"
        assert str(errors.CpError("USAGE")) == "USAGE"

    def test_unknown_code_is_a_bug(self) -> None:
        with pytest.raises(ValueError, match="unknown error code"):
            errors.CpError("NOPE")


class TestClock:
    def test_system_clock_is_utc(self) -> None:
        sc = clock.SystemClock()
        assert sc.now().tzinfo is UTC
        sc.sleep(0)

    def test_fake_clock_advances_on_sleep(self) -> None:
        fc = clock.FakeClock()
        start = fc.now()
        fc.sleep(5)
        fc.advance(1.5)
        assert (fc.now() - start).total_seconds() == 6.5
        assert fc.sleeps == [5]
        assert clock.FakeClock(datetime(2020, 1, 1, tzinfo=UTC)).now().year == 2020

    def test_iso_round_trip_and_ordering(self) -> None:
        fc = clock.FakeClock()
        a = clock.stamp(fc)
        b = clock.stamp(fc, plus=0.5)
        assert a < b
        assert clock.parse(a) == fc.now()
        assert a.endswith("Z") and len(a) == len(b)

    def test_age(self) -> None:
        fc = clock.FakeClock()
        then = clock.stamp(fc)
        fc.advance(30)
        assert clock.age(fc, then) == 30
        assert clock.age(fc, None) is None


class TestFaults:
    def test_unarmed_is_noop(self) -> None:
        faults.hit("x")

    def test_armed_raises_then_disarms(self) -> None:
        with faults.arm("x"), pytest.raises(faults.FaultInjected):
            faults.hit("x")
        faults.hit("x")

    def test_custom_exception_and_reset(self) -> None:
        with faults.arm("y", RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                faults.hit("y")
            faults.reset()
            faults.hit("y")

    def test_skip(self) -> None:
        with faults.arm("z", skip=2):
            faults.hit("z")
            faults.hit("z")
            with pytest.raises(faults.FaultInjected):
                faults.hit("z")

    def test_fault_is_not_an_exception(self) -> None:
        assert not issubclass(faults.FaultInjected, Exception)


class TestCli:
    def test_version(self, invoke) -> None:
        code, out = invoke("version")
        assert code == 0
        assert out["ok"] is True and out["package"] == "control_plane"

    def test_usage_error(self, invoke) -> None:
        code, out = invoke("nope")
        assert code == 2
        assert out["error"] == "USAGE"

    def test_internal_error(self, invoke, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(args: argparse.Namespace, io: cli.Io) -> dict:
            raise RuntimeError("kaput")

        monkeypatch.setitem(cli.COMMANDS, "version", cli.Command("version", "exempt", boom))
        code, out = invoke("version")
        assert code == 1
        assert out == {"ok": False, "error": "INTERNAL", "detail": "RuntimeError: kaput"}

    def test_cp_error(self, invoke, monkeypatch: pytest.MonkeyPatch) -> None:
        def busy(args: argparse.Namespace, io: cli.Io) -> dict:
            raise errors.CpError("BUSY", "later")

        monkeypatch.setitem(cli.COMMANDS, "version", cli.Command("version", "exempt", busy))
        assert invoke("version") == (6, {"ok": False, "error": "BUSY", "detail": "later"})

    def test_nested_commands_raw_output_and_tool_group_leader(
        self, invoke, monkeypatch: pytest.MonkeyPatch, fake_setup
    ) -> None:
        def raw(args: argparse.Namespace, io: cli.Io) -> cli.Raw:
            return cli.Raw(f"x={args.x}", 3)

        def empty(args: argparse.Namespace, io: cli.Io) -> cli.Raw:
            return cli.Raw("")

        def conf(p: argparse.ArgumentParser) -> None:
            p.add_argument("--x", required=True)

        monkeypatch.setitem(cli.COMMANDS, "tool a b", cli.Command("tool a b", "write", raw, conf))
        monkeypatch.setitem(cli.COMMANDS, "tool a c", cli.Command("tool a c", "write", empty))
        assert invoke("tool", "a", "b", "--x", "1") == (3, "x=1\n")
        assert invoke("tool", "a", "c") == (0, None)
        assert fake_setup.calls == [("group_leader", None), ("group_leader", None)]
        code, out = invoke("tool", "a", "b")
        assert code == 2 and "--x" in out["detail"]

    def test_version_does_not_become_group_leader(self, invoke, fake_setup) -> None:
        invoke("version")
        assert fake_setup.calls == []

    def test_main_defaults_to_sys_streams(self, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        monkeypatch.setattr(sys, "argv", ["cp.py", "version"])
        assert cli.main() == 0
        assert '"package": "control_plane"' in capsys.readouterr().out


def test_cp_entry_smoke() -> None:
    """``cp.py version`` in a real subprocess (the shim is outside the coverage source)."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_QS / "cp.py"), "version"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert '"ok": true' in proc.stdout


def test_gitignore_has_anchored_state_entries() -> None:
    lines = (Path(SCRIPTS_QS).parents[1] / ".gitignore").read_text().splitlines()
    for entry in (
        "/harness_state.db",
        "/harness_state.db-wal",
        "/harness_state.db-shm",
        "/harness_state.db.*.lock",
        "/harness_state.db.migrate-error.json",
    ):
        assert entry in lines, entry
    assert "*.log" in lines  # covers harness_state.daemon.log
