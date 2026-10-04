"""Checkpoint 9: the Stop / PreToolUse / pre-push hooks, the installer, the shim, hooks-settings (AC15)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from control_plane import hooks, runner

from .conftest import ORCH, FakeRunner, insert_node, insert_task, open_run, run_cli, sql


def stop(session: str) -> tuple[int, Any]:
    return run_cli("hook", "stop", stdin=json.dumps({"session_id": session, "hook_event_name": "Stop", "cwd": "/x"}))


def pre_tool(session: str | None, tool: str, **tool_input: Any) -> tuple[int, Any]:
    payload = {"session_id": session, "hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input}
    return run_cli("hook", "pre-tool-use", stdin=json.dumps(payload))


def denied(result: tuple[int, Any]) -> bool:
    code, out = result
    assert code == 0
    if out is None:
        return False
    spec = out["hookSpecificOutput"]
    assert spec["hookEventName"] == "PreToolUse" and spec["permissionDecision"] == "deny"
    assert spec["permissionDecisionReason"]
    return True


def events(path: Path) -> list[dict]:
    return [
        {"decision": r["decision"], **json.loads(r["detail"])}
        for r in sql(path, "SELECT decision, detail FROM hook_events ORDER BY id")
    ]


@pytest.fixture
def world(migrated: Path, tmp_path: Path) -> dict:
    run_id, token = open_run()
    insert_task(migrated, "T1", run_id, worktree=str(tmp_path / "wt1"), branch="QS_11")
    insert_task(migrated, "T2", run_id)
    node = insert_node(migrated, "N1", run_id, "T1", session_id="S-n1")
    payload = tmp_path / "m.json"
    payload.write_text("{}")
    return {"run": run_id, "token": token, "node": node, "db": migrated, "payload": str(payload), "tmp": tmp_path}


def post(w: dict, to: str = "orchestrator") -> int:
    out = run_cli(
        "msg",
        "post",
        "--run",
        w["run"],
        "--to",
        to,
        "--kind",
        "k",
        "--payload-file",
        w["payload"],
        "--token",
        w["token"],
    )[1]
    return int(out["id"])


def add_waiter(w: dict, pid: int = 7) -> None:
    sql(
        w["db"],
        "INSERT INTO waiters (run_id, pid, pid_start, started_at, heartbeat_at) VALUES (?, ?, ?, 'x', 'x')",
        [w["run"], pid, f"start-{pid}"],
    )


class TestStop:
    def test_allows_unregistered_and_without_db(self, db_path) -> None:
        assert stop("S-x") == (0, None)
        assert run_cli("hook", "stop", stdin="{}") == (0, None)

    def test_blocks_while_a_message_is_visible(self, world, fake_main) -> None:
        add_waiter(world)
        mid = post(world)
        code, out = stop(ORCH)
        assert code == 0 and out["decision"] == "block"
        assert "1 message(s) wait in run R1's queue" in out["reason"]
        assert f"{fake_main.resolve()}/venv/bin/python {fake_main.resolve()}/scripts/qs/cp.py msg pop" in out["reason"]
        assert events(world["db"])[-1] == {
            "decision": "block",
            "kind": "queue",
            "run_id": "R1",
            "head": mid,
            "acked": 0,
        }

    def test_queue_loop_guard_alerts_and_posts_nothing(self, world) -> None:
        add_waiter(world)
        post(world)
        assert stop(ORCH)[1]["decision"] == "block"
        assert stop(ORCH)[1]["decision"] == "block"
        assert stop(ORCH) == (0, None)
        assert (
            events(world["db"])[-1]["decision"] == "alert" and events(world["db"])[-1]["kind"] == "queue_not_draining"
        )
        assert sql(world["db"], "SELECT count(*) FROM messages")[0][0] == 1

    def test_progress_resets_the_loop_guard(self, world) -> None:
        add_waiter(world)
        post(world)
        post(world, "node:T1")
        assert stop(ORCH)[1]["decision"] == "block"
        assert stop(ORCH)[1]["decision"] == "block"
        msg = run_cli("msg", "pop", "--run", "R1", "--as", "node:T1", "--token", world["node"])[1]
        run_cli("msg", "ack", str(msg["id"]), "--receipt", msg["receipt"], "--token", world["node"])
        assert stop(ORCH)[1]["decision"] == "block"  # something was acked in between

    def test_asks_once_for_wait(self, world) -> None:
        code, out = stop(ORCH)
        assert out["decision"] == "block" and "wait --run R1" in out["reason"]
        assert stop(ORCH) == (0, None)
        assert events(world["db"])[-1] == {"decision": "alert", "kind": "idle_without_wait", "run_id": "R1"}

    def test_live_waiter_allows(self, world, fake_probe) -> None:
        add_waiter(world)
        assert stop(ORCH) == (0, None)
        fake_probe.kill(7)
        assert stop(ORCH)[1]["decision"] == "block"  # a dead waiter is absent

    def test_superseded_and_closed(self, world, fake_claude) -> None:
        from .conftest import agent

        fake_claude.listing = [agent(ORCH), agent("S-new")]
        new = run_cli("run", "claim", "R1", "--session-id", "S-new", "--takeover")[1]["token"]
        assert stop(ORCH) == (0, None)
        run_cli("run", "close", "--token", new)
        assert stop("S-new") == (0, None)

    def test_fails_open_on_a_schema_mismatch(self, world, capsys) -> None:
        post(world)
        sql(world["db"], "PRAGMA user_version = 2")
        assert stop(ORCH) == (0, None)
        assert "stop hook failed open" in capsys.readouterr().err

    def test_fails_open_on_an_error_and_records_it(self, world, monkeypatch, capsys) -> None:
        def boom(*a: Any) -> str:
            raise RuntimeError("kaput")

        monkeypatch.setattr(hooks, "stop_decision", boom)
        assert stop(ORCH) == (0, None)
        assert events(world["db"])[-1]["kind"] == "error" and "kaput" in capsys.readouterr().err
        assert run_cli("hook", "stop", stdin="not json") == (0, None)


class TestPreToolUse:
    @pytest.mark.parametrize(
        "command",
        [
            "sqlite3 harness_state.db 'select 1'",
            "rm harness_state.db",
            "python -c \"import sqlite3; sqlite3.connect('harness_state.db')\"",
            "cat x > harness_state.db",
            "echo hi >> /main/harness_state.db-wal",
            "sed -i s/a/b/ harness_state.db",
            "FOO=1 sqlite3 /m/harness_state.db",
            "ls && sqlite3 harness_state.db",
        ],
    )
    def test_denies_db_access(self, command: str, db_path) -> None:
        assert denied(pre_tool("S-x", "Bash", command=command))

    @pytest.mark.parametrize(
        "command",
        [
            "grep -n harness_state.db docs/x.md",
            "git status && cat .gitignore | grep harness_state.db",
            "cat harness_state.db",
            "sed -n 1p harness_state.db",
            "venv/bin/python scripts/qs/cp.py snapshot # harness_state.db",
            "ls -la",
        ],
    )
    def test_allows_reads_and_cp(self, command: str, db_path) -> None:
        assert not denied(pre_tool("S-x", "Bash", command=command))

    @pytest.mark.parametrize("name", ["harness_state.db", "harness_state.db-wal", "harness_state.db-shm"])
    def test_edit_write_of_the_db(self, name: str, db_path) -> None:
        assert denied(pre_tool("S-x", "Edit", file_path=f"/m/{name}"))
        assert denied(pre_tool("S-x", "Write", file_path=name))
        assert not denied(pre_tool("S-x", "Write", file_path="/m/notes.md"))
        assert not denied(pre_tool("S-x", "Read", file_path=f"/m/{name}"))

    def test_registered_sessions_cannot_merge(self, world) -> None:
        assert denied(pre_tool(ORCH, "Bash", command="gh pr merge 5 --merge"))
        assert denied(pre_tool("S-n1", "Bash", command="echo x && gh pr merge 5"))
        assert not denied(pre_tool("S-free", "Bash", command="gh pr merge 5"))  # the frozen pipeline
        assert not denied(pre_tool(ORCH, "Bash", command="gh pr view 5"))
        rows = events(world["db"])
        assert [r["decision"] for r in rows] == ["deny", "deny"]

    def test_send_message_fencing(self, world, fake_claude) -> None:
        from .conftest import agent

        assert not denied(pre_tool("S-n1", "SendMessage", to="x", message="hi"))
        sql(world["db"], "UPDATE nodes SET state = 'stopped'")
        assert denied(pre_tool("S-n1", "SendMessage", to="x", message="hi"))
        fake_claude.listing = [agent(ORCH), agent("S-new")]
        run_cli("run", "claim", "R1", "--session-id", "S-new", "--takeover")
        assert denied(pre_tool(ORCH, "SendMessage", to="x", message="hi"))
        assert not denied(pre_tool("S-new", "SendMessage", to="x", message="hi"))

    def test_without_db_and_without_session(self, db_path) -> None:
        assert not denied(pre_tool(None, "Bash", command="gh pr merge 1"))
        assert denied(pre_tool(None, "Bash", command="sqlite3 harness_state.db"))

    def test_fails_open(self, world, monkeypatch, capsys) -> None:
        assert run_cli("hook", "pre-tool-use", stdin="nope") == (0, None)
        sql(world["db"], "PRAGMA user_version = 2")
        assert not denied(pre_tool(ORCH, "Bash", command="gh pr merge 5"))
        assert "failed open" in capsys.readouterr().err


class TestPrePush:
    def _push(self, w: dict, fake_runner: FakeRunner, line: str, token: str | None = None, toplevel: str | None = None):
        fake_runner.on(["git", "rev-parse", "--show-toplevel"], (toplevel or str(w["tmp"] / "wt1")) + "\n")
        return hooks.pre_push_decision(
            line + "\n", clock=__import__("control_plane").clock.FakeClock(), run=fake_runner, token=token
        )

    def test_item_refs_refused_everywhere_even_without_db(self, db_path, fake_runner) -> None:
        for line in ("refs/heads/QS_5_2 abc refs/heads/QS_5_2 000", "refs/heads/x abc refs/heads/QS_12_1 000"):
            code, msg = hooks.pre_push_decision(line, clock=None, run=fake_runner, token=None)  # type: ignore[arg-type]
            assert code == 1 and "work-item" in msg
        assert fake_runner.calls == []

    def test_unregistered_and_unknown_allow(self, world, fake_runner) -> None:
        assert self._push(world, fake_runner, "refs/heads/x a refs/heads/x b", toplevel="/elsewhere") == (0, "")
        fake_runner.on(["git", "rev-parse"], runner.RunResult(128, "", "not a repo"))
        assert hooks.pre_push_decision("x", clock=None, run=fake_runner, token=None) == (0, "")  # type: ignore[arg-type]

    def test_missing_db_and_schema_mismatch_allow(self, db_path, fake_runner, tmp_path, capsys) -> None:
        w = {"tmp": tmp_path}
        assert self._push(w, fake_runner, "refs/heads/QS_11 a refs/heads/QS_11 b") == (0, "")
        from control_plane import migrations

        migrations.migrate(db_path, role="test")
        sql(db_path, "PRAGMA user_version = 3")
        assert self._push(w, fake_runner, "refs/heads/QS_11 a refs/heads/QS_11 b") == (0, "")
        assert "failed open" in capsys.readouterr().err

    def test_registered_worktree(self, world, fake_runner) -> None:
        good = "refs/heads/QS_11 a refs/heads/QS_11 b"
        assert self._push(world, fake_runner, good)[0] == 1  # no token
        assert "cp.py tool push" in self._push(world, fake_runner, good, "garbage")[1]
        assert self._push(world, fake_runner, good, world["token"]) == (0, "")
        assert self._push(world, fake_runner, good, world["node"]) == (0, "")
        code, msg = self._push(world, fake_runner, "refs/heads/QS_11 a refs/heads/main b", world["token"])
        assert code == 1 and "may push only refs/heads/QS_11" in msg
        other_run = open_run("r2", "S-2")[1]
        assert self._push(world, fake_runner, good, other_run)[0] == 1
        insert_node(world["db"], "N2", "R1", "T1", generation=2)
        assert self._push(world, fake_runner, good, world["node"])[0] == 1  # an older generation
        sql(world["db"], "UPDATE nodes SET state = 'stopped' WHERE id = 'N2'")
        code, msg = self._push(world, fake_runner, good, world["token"])
        assert code == 1 and "stopped" in msg

    def test_registered_without_branch_and_fail_closed(self, world, fake_runner, monkeypatch) -> None:
        sql(world["db"], "UPDATE tasks SET branch = NULL WHERE id = 'T1'")
        assert self._push(world, fake_runner, "refs/heads/QS_11 a refs/heads/QS_11 b", world["token"])[0] == 1
        sql(world["db"], "UPDATE tasks SET branch = 'QS_11' WHERE id = 'T1'")

        def boom(*a: Any) -> bool:
            raise RuntimeError("x")

        monkeypatch.setattr(hooks, "_push_token_ok", boom)
        code, msg = self._push(world, fake_runner, "refs/heads/QS_11 a refs/heads/QS_11 b", world["token"])
        assert code == 1 and "check failed" in msg

    def test_cli_reads_the_env_token(self, world, fake_runner, monkeypatch) -> None:
        fake_runner.on(["git", "rev-parse", "--show-toplevel"], str(world["tmp"] / "wt1"))
        line = "refs/heads/QS_11 a refs/heads/QS_11 b\n"
        code, out = run_cli("hook", "pre-push", "origin", "git@x:y", stdin=line)
        assert code == 1 and "QS_CP_TOKEN" in out
        monkeypatch.setenv("QS_CP_TOKEN", world["token"])
        assert run_cli("hook", "pre-push", "origin", "git@x:y", stdin=line) == (0, None)


def _git(
    *args: str, cwd: Path, env: dict[str, str] | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check, env=env)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    main = tmp_path / "repo"
    main.mkdir()
    _git("init", "-q", "-b", "main", cwd=main)
    _git("-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q", "--allow-empty", "-m", "x", cwd=main)
    return main


class TestInstaller:
    def test_install_idempotent_upgrade(self, repo: Path) -> None:
        common = repo / ".git"
        out = hooks.install_pre_push(common, runner.Runner())
        target = common / "hooks" / "pre-push"
        assert out == {"path": str(target), "status": "installed"}
        assert os.access(target, os.X_OK) and "# qs-control-plane pre-push shim v1" in target.read_text()
        assert hooks.install_pre_push(common, runner.Runner())["status"] == "already_installed"
        target.write_text("#!/bin/sh\n# qs-control-plane pre-push shim v0\n")
        assert hooks.install_pre_push(common, runner.Runner())["status"] == "upgraded"
        assert target.read_text() == hooks.SHIM

    def test_refusals(self, repo: Path) -> None:
        common = repo / ".git"
        (common / "hooks").mkdir(exist_ok=True)
        (common / "hooks" / "pre-push").write_text("#!/bin/sh\necho mine\n")
        with pytest.raises(hooks.errors.CpError) as exc:
            hooks.install_pre_push(common, runner.Runner())
        assert exc.value.code == "POLICY_REFUSED" and "foreign" in exc.value.detail
        (common / "hooks" / "pre-push").unlink()
        _git("config", "core.hooksPath", ".githooks", cwd=repo)
        with pytest.raises(hooks.errors.CpError) as exc:
            hooks.install_pre_push(common, runner.Runner())
        assert "core.hooksPath" in exc.value.detail
        with pytest.raises(hooks.errors.CpError) as exc:
            hooks.install_pre_push(Path.home() / "not-a-temp-dir" / ".git", runner.Runner())
        assert "temporary" in exc.value.detail


STUB = """import json, os, sys
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calls.jsonl")
with open(out, "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "stdin": sys.stdin.read()}) + "\\n")
sys.exit(int(os.environ.get("STUB_EXIT", "0")))
"""


class TestShimEndToEnd:
    def test_routes_every_worktree_push_through_main_cp(self, repo: Path, tmp_path: Path) -> None:
        bare = tmp_path / "origin.git"
        _git("init", "-q", "--bare", str(bare), cwd=tmp_path)
        _git("remote", "add", "origin", str(bare), cwd=repo)
        stub = repo / "scripts" / "qs" / "cp.py"
        stub.parent.mkdir(parents=True)
        stub.write_text(STUB)
        hooks.install_pre_push(repo / ".git", runner.Runner())
        _git("worktree", "add", "-q", "-b", "QS_7", str(tmp_path / "wt"), cwd=repo)
        _git("push", "-q", "origin", "QS_7", cwd=tmp_path / "wt")
        calls = [json.loads(ln) for ln in (stub.parent / "calls.jsonl").read_text().splitlines()]
        assert calls[0]["argv"] == ["hook", "pre-push", "origin", str(bare)]
        assert calls[0]["stdin"].startswith("refs/heads/QS_7 ")
        env = {**os.environ, "STUB_EXIT": "1"}
        refused = _git("push", "-q", "origin", "main", cwd=repo, env=env, check=False)
        assert refused.returncode != 0
        stub.unlink()
        assert _git("push", "-q", "origin", "main", cwd=repo, check=False).returncode == 0  # no cp.py: allow

    def test_settings_command_runs_with_the_python3_fallback(self, repo: Path) -> None:
        stub = repo / "scripts" / "qs" / "cp.py"
        stub.parent.mkdir(parents=True)
        stub.write_text(STUB)
        fragment = hooks.hooks_settings("orchestrator", repo)
        command = fragment["hooks"]["Stop"][0]["hooks"][0]["command"]
        proc = subprocess.run(
            ["sh", "-c", command], input='{"session_id": "s"}', capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr
        call = json.loads((stub.parent / "calls.jsonl").read_text())
        assert call == {"argv": ["hook", "stop"], "stdin": '{"session_id": "s"}'}


class TestHooksSettings:
    @pytest.mark.parametrize("role", ["node", "orchestrator"])
    def test_fragment(self, role: str, fake_main: Path) -> None:
        code, out = run_cli("hooks-settings", "--role", role)
        assert code == 0
        pre = out["hooks"]["PreToolUse"][0]
        assert pre["matcher"] == "Bash|Edit|Write|SendMessage"
        entry = pre["hooks"][0]
        assert entry["type"] == "command" and entry["timeout"] == 10
        assert f"{fake_main.resolve()}/scripts/qs/cp.py hook pre-tool-use" in entry["command"]
        if role == "orchestrator":
            stop_group = out["hooks"]["Stop"][0]
            assert "matcher" not in stop_group and stop_group["hooks"][0]["command"].endswith("hook stop")
        else:
            assert set(out["hooks"]) == {"PreToolUse"}

    def test_bad_role(self) -> None:
        assert run_cli("hooks-settings", "--role", "x")[0] == 2
        with pytest.raises(hooks.errors.CpError):
            hooks.hooks_settings("x")
