"""QS-406 T4: the tick-hook registration point, the seams, the guard test (§2, §3, §12, AC 1)."""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from typing import Any

import pytest
from control_plane import activeloop, clock, daemon, liveness, runner, ticks, tools

from .conftest import REAL_MAKE_SEAMS, FakeRunner


def _noop(conn: Any, clk: clock.Clock) -> None:
    return None


@pytest.mark.usefixtures("active_loop")
class TestRegistry:
    def test_register_and_list(self) -> None:
        ticks.register("t-one", _noop)
        assert ("t-one", _noop) in ticks.registered()
        named = [h for h in ticks.hooks() if h.__name__ == "t-one"]
        assert len(named) == 1

    def test_a_duplicate_name_raises(self) -> None:
        ticks.register("t-dup", _noop)
        with pytest.raises(ValueError, match="t-dup"):
            ticks.register("t-dup", _noop)

    def test_hooks_call_through(self) -> None:
        seen: list[Any] = []
        ticks.register("t-call", lambda conn, clk: seen.append((conn, clk)))
        hook = next(h for h in ticks.hooks() if h.__name__ == "t-call")
        hook("c", "k")  # type: ignore[arg-type]
        assert seen == [("c", "k")]

    def test_reset_clears(self) -> None:
        ticks.register("t-gone", _noop)
        ticks._reset_for_tests()
        assert "t-gone" not in dict(ticks.registered())


class TestThrottle:
    def test_respects_its_period(self, fake_clock: clock.FakeClock) -> None:
        t = ticks.Throttle(30)
        assert t.due(fake_clock)  # a new throttle (a new daemon) is due at once
        assert not t.due(fake_clock)
        fake_clock.advance(29.9)
        assert not t.due(fake_clock)
        fake_clock.advance(0.1)
        assert t.due(fake_clock)
        assert not t.due(fake_clock)

    def test_reset_rearms_every_throttle(self, fake_clock: clock.FakeClock) -> None:
        t = ticks.Throttle(300)
        assert t.due(fake_clock) and not t.due(fake_clock)
        ticks._reset_for_tests()
        assert t.due(fake_clock)


class TestBuiltins:
    def test_register_builtin_is_idempotent(self, monkeypatch) -> None:
        monkeypatch.setattr(activeloop, "_builtin", lambda: [("t-builtin", _noop)])
        activeloop.register_builtin()
        activeloop.register_builtin()
        assert [n for n, _ in ticks.registered()].count("t-builtin") == 1

    def test_builtin_names_match_the_registered_list(self) -> None:
        assert {n for n, _ in activeloop._builtin()} == activeloop.BUILTIN_NAMES
        assert {n for n, _ in ticks.registered()} >= activeloop.BUILTIN_NAMES


class TestUnderTheDaemon:
    def test_a_hook_registered_from_cli_runs_under_cp_daemon(self, invoke, monkeypatch, active_loop) -> None:
        seen: list[int] = []
        monkeypatch.setattr(activeloop, "_builtin", lambda: [("t-count", lambda conn, clk: seen.append(1))])
        monkeypatch.setattr(daemon, "IDLE_EXIT_S", 10.0)
        code, out = invoke("daemon")  # cli.main calls register_builtin(), _daemon passes ticks.hooks()
        assert code == 0 and out["ticks"] == 3 and len(seen) == 3

    def test_a_raising_hook_is_logged_and_the_loop_goes_on(self, invoke, monkeypatch, active_loop, capsys) -> None:
        seen: list[int] = []

        def boom(conn: Any, clk: clock.Clock) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            activeloop, "_builtin", lambda: [("t-boom", boom), ("t-after", lambda conn, clk: seen.append(1))]
        )
        monkeypatch.setattr(daemon, "IDLE_EXIT_S", 10.0)
        code, out = invoke("daemon")
        assert code == 0 and out["ticks"] == 3 and len(seen) == 3
        assert "'t-boom' failed" in capsys.readouterr().err

    def test_without_active_loop_the_daemon_runs_no_hook(self, invoke, monkeypatch) -> None:
        seen: list[int] = []
        monkeypatch.setattr(activeloop, "_builtin", lambda: [("t-off", lambda conn, clk: seen.append(1))])
        monkeypatch.setattr(daemon, "IDLE_EXIT_S", 10.0)
        assert invoke("daemon")[0] == 0 and seen == []

    def test_guard_no_real_subprocess_or_socket(self, invoke, monkeypatch, active_loop, fake_runner) -> None:
        """Every built-in hook goes through the seams: nothing real is spawned or connected."""

        def refuse(*a: Any, **k: Any) -> Any:
            raise AssertionError(f"a real subprocess or socket under cp.py daemon: {a!r}")

        monkeypatch.setattr(subprocess, "run", refuse)
        monkeypatch.setattr(subprocess, "Popen", refuse)
        monkeypatch.setattr(socket, "socket", refuse)
        monkeypatch.setattr(daemon, "IDLE_EXIT_S", 10.0)
        code, out = invoke("daemon")
        assert code == 0 and out["exit"] == "idle" and out["ticks"] == 3
        assert all(isinstance(c.argv, list) for c in fake_runner.calls)


class TestSeams:
    def test_cached_once_and_built_on_the_test_deps(self, deps, fake_main: Path, fake_runner: FakeRunner) -> None:
        s = activeloop.seams()
        assert s is activeloop.seams()
        assert s.runner is fake_runner and s.probe is deps.probe and s.claude is deps.claude and s.main == fake_main
        activeloop._reset_for_tests()
        assert activeloop.seams() is not s

    def test_the_real_make_seams(self, fake_main: Path) -> None:
        s = REAL_MAKE_SEAMS()
        assert isinstance(s.runner, runner.Runner)
        assert isinstance(s.probe, liveness.ProcessProbe) and isinstance(s.claude, liveness.ClaudeCli)
        assert s.main == fake_main


def test_version_lists_tools_and_tick_hooks(invoke, monkeypatch) -> None:
    monkeypatch.setattr(activeloop, "_builtin", lambda: [("t-listed", _noop)])
    out = invoke("version")[1]
    assert set(out["tools"]) >= tools.BUILTIN_NAMES and out["tools"] == sorted(out["tools"])
    assert "t-listed" in out["tick_hooks"]
