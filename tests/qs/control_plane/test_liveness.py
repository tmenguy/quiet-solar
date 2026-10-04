"""Checkpoint 3: ``ProcessProbe`` and ``ClaudeCli`` — the real seams (§13, §14)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from control_plane import errors, liveness, runner

from .conftest import FakeRunner


@pytest.fixture
def probe() -> liveness.ProcessProbe:
    return liveness.ProcessProbe()


class TestProcessProbe:
    def test_self_is_alive_with_matching_start(self, probe: liveness.ProcessProbe) -> None:
        start = probe.start_of(os.getpid())
        assert start is not None
        assert probe.alive(os.getpid(), start)
        assert probe.alive(os.getpid(), None)
        me = probe.me()
        assert me == liveness.Holder(os.getpid(), start, os.getpgid(0))

    def test_start_time_mismatch_is_dead(self, probe: liveness.ProcessProbe) -> None:
        assert not probe.alive(os.getpid(), "1999-01-01T00:00:00")

    def test_reaped_child_is_dead(self, probe: liveness.ProcessProbe) -> None:
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        assert not probe.alive(child.pid, None)
        assert probe.start_of(child.pid) is None
        assert not probe.alive(None, None)

    def test_permission_error_counts_as_alive(self, probe, monkeypatch: pytest.MonkeyPatch) -> None:
        def deny(pid: int, sig: int) -> None:
            raise PermissionError

        monkeypatch.setattr(os, "kill", deny)
        monkeypatch.setattr(os, "killpg", deny)
        assert probe.alive(1, None)
        assert probe.group_alive(1)

    def test_group_alive_and_a_detached_child(self, probe: liveness.ProcessProbe) -> None:
        assert probe.group_alive(os.getpgid(0))
        assert not probe.group_alive(None)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            assert os.getpgid(child.pid) != os.getpgid(0)  # not in the tool's group
            assert probe.group_alive(child.pid)
            assert probe.holder_alive(None, None, child.pid)
        finally:
            child.kill()
            child.wait()
        assert not probe.group_alive(child.pid)
        assert not probe.holder_alive(child.pid, None, child.pid)

    def test_unparseable_ps_output(self) -> None:
        fake = FakeRunner()
        fake.on(["ps"], "not a date")
        assert liveness.ProcessProbe(fake).start_of(1) is None
        assert fake.calls[0].env_extra == {"LC_ALL": "C"}

    def test_default_runner(self) -> None:
        assert isinstance(liveness.ProcessProbe().runner, runner.Runner)


AGENTS = [
    {
        "pid": 14628,
        "cwd": "/repo",
        "kind": "interactive",
        "startedAt": 1790938096856,
        "sessionId": "c0fce2fd",
        "name": "orchestrator",
        "status": "idle",
    },
    {"sessionId": "bg1", "id": "abc123", "name": "R1-T2-g1", "kind": "background", "state": "running", "pid": True},
]


def _fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "claude"
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")


class TestClaudeCli:
    def test_agents_from_a_fake_executable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fixture = tmp_path / "agents.json"
        fixture.write_text(json.dumps(AGENTS))
        _fake_claude(tmp_path, monkeypatch, f'cat "{fixture}"\n')
        cli = liveness.ClaudeCli()
        agents = cli.agents()
        assert agents[0] == liveness.Agent(
            "c0fce2fd", None, "orchestrator", "/repo", "interactive", "idle", None, 14628, 1790938096856
        )
        assert agents[1].id == "abc123" and agents[1].pid is None and agents[1].started_at_ms is None
        assert cli.try_agents() == agents
        assert liveness.find(agents, session_id="bg1") is agents[1]
        assert liveness.find(agents, name="orchestrator") is agents[0]
        assert liveness.find(agents, name="nope") is None

    def test_exec_failure_is_internal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_claude(tmp_path, monkeypatch, "echo boom >&2; exit 1\n")
        with pytest.raises(errors.CpError) as exc:
            liveness.ClaudeCli().agents()
        assert exc.value.code == "INTERNAL" and "boom" in exc.value.detail
        assert liveness.ClaudeCli().try_agents() is None

    @pytest.mark.parametrize("stdout", ["not json", '{"a": 1}', '[{"id": 1}]', "[1]"])
    def test_bad_output_is_internal(self, stdout: str) -> None:
        fake = FakeRunner()
        fake.on(["claude", "agents"], stdout)
        with pytest.raises(errors.CpError) as exc:
            liveness.ClaudeCli(fake).agents()
        assert exc.value.code == "INTERNAL"

    def test_launches_are_detached_without_the_token(self, tmp_path: Path) -> None:
        fake = FakeRunner()
        cli = liveness.ClaudeCli(fake)
        cli.spawn_bg(["--agent", "x", "prompt"], cwd=tmp_path)
        cli.resume_bg("sid", "msg", cwd=tmp_path)
        assert [c.argv for c in fake.calls] == [
            ["claude", "--bg", "--agent", "x", "prompt"],
            ["claude", "--bg", "--resume", "sid", "msg"],
        ]
        for call in fake.calls:
            assert call.detach and call.env_remove == ("QS_CP_TOKEN",) and call.cwd == str(tmp_path)
        assert isinstance(liveness.ClaudeCli().runner, runner.Runner)

    def test_started_after(self) -> None:
        agent = liveness.Agent("s", None, None, None, None, None, None, None, 1_790_938_096_856)
        assert agent.started_after("2026-09-01T00:00:00.000000Z")
        assert not agent.started_after("2027-01-01T00:00:00.000000Z")
        assert not agent.started_after(None)
        assert not liveness.Agent("s", None, None, None, None, None, None, None, None).started_after(
            "2026-09-01T00:00:00.000000Z"
        )
