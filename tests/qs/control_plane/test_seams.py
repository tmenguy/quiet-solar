"""Real seams tested directly (story §14): ``Runner`` and ``ProcessSetup``."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Any

import pytest
from control_plane import procsetup, runner


class TestRunner:
    def test_runs_and_captures(self) -> None:
        res = runner.Runner().run([sys.executable, "-c", "import sys; print('hi'); sys.exit(3)"])
        assert res.returncode == 3 and res.stdout == "hi\n" and not res.ok
        assert runner.RunResult(0, "", "").ok

    def test_env_extra_and_remove(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("QS_CP_TOKEN", "secret")
        code = "import os; print(os.environ.get('A'), os.environ.get('QS_CP_TOKEN'), os.getcwd())"
        res = runner.Runner().run(
            [sys.executable, "-c", code], cwd=tmp_path, env_extra={"A": "1"}, env_remove=["QS_CP_TOKEN"]
        )
        a, tok, cwd = res.stdout.split()
        assert (a, tok) == ("1", "None")
        assert os.path.realpath(cwd) == os.path.realpath(tmp_path)

    def test_detach_starts_a_new_session(self) -> None:
        code = "import os; print(os.getsid(0) == os.getpid())"
        assert runner.Runner().run([sys.executable, "-c", code], detach=True).stdout.strip() == "True"
        assert runner.Runner().run([sys.executable, "-c", code]).stdout.strip() == "False"

    def test_exec_failure_is_127(self, tmp_path) -> None:
        res = runner.Runner().run([str(tmp_path / "missing-binary")])
        assert res.returncode == 127 and res.stderr

    def test_timeout_is_124(self) -> None:
        res = runner.Runner().run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.3)
        assert res.returncode == 124 and "timeout" in res.stderr

    def test_text_helper(self) -> None:
        assert runner._text(None) == ""
        assert runner._text(b"ab") == "ab"
        assert runner._text("cd") == "cd"


class TestProcessSetup:
    def test_get_returns_the_real_setup(self) -> None:
        from .conftest import REAL_PROCSETUP_GET  # the autouse fixture replaced `get`

        assert isinstance(REAL_PROCSETUP_GET(), procsetup.ProcessSetup)

    def test_setpgid_when_not_leader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded: list[tuple[int, int]] = []
        monkeypatch.setattr(os, "getpgid", lambda pid: os.getpid() + 1)
        monkeypatch.setattr(os, "setpgid", lambda a, b: recorded.append((a, b)))
        procsetup.ProcessSetup().become_group_leader()
        assert recorded == [(0, 0)]

    def test_noop_when_already_leader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded: list[tuple[int, int]] = []
        monkeypatch.setattr(os, "getpgid", lambda pid: os.getpid())
        monkeypatch.setattr(os, "setpgid", lambda a, b: recorded.append((a, b)))
        procsetup.ProcessSetup().become_group_leader()
        assert recorded == []

    def test_install_sigterm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded: list[tuple[Any, Any]] = []
        monkeypatch.setattr(signal, "signal", lambda sig, fn: recorded.append((sig, fn)))

        def handler(signum: int, frame: Any) -> None:
            return None

        procsetup.ProcessSetup().install_sigterm(handler)
        assert recorded == [(signal.SIGTERM, handler)]

    @pytest.mark.parametrize("new_session", [False, True])
    def test_behaviour_in_a_subprocess(self, new_session: bool) -> None:
        """A normal child becomes a leader; a session leader is left alone (no EPERM)."""
        code = (
            "import os, sys; sys.path.insert(0, sys.argv[1]);"
            "from control_plane import procsetup; procsetup.ProcessSetup().become_group_leader();"
            "print(os.getpgid(0) == os.getpid())"
        )
        from .conftest import SCRIPTS_QS

        proc = subprocess.run(
            [sys.executable, "-c", code, str(SCRIPTS_QS)],
            capture_output=True,
            text=True,
            start_new_session=new_session,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "True"
