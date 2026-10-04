"""Checkpoint 5: run open / claim / bind-name / set-mode / set-plan / close, session status (AC9, AC10)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from control_plane import db, errors, runs

from .conftest import ORCH, agent, insert_node, insert_task, open_run, run_cli, sql


def _lease(path: Path, run_id: str) -> dict:
    return dict(sql(path, "SELECT * FROM run_leases WHERE run_id = ?", [run_id])[0])


class TestOpen:
    def test_open_creates_run_lease_and_starts_the_daemon(self, db_path, fake_popen, monkeypatch) -> None:
        fake_popen.on_call = lambda a, k: __import__("control_plane").migrations.migrate(db_path, role="test")
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", ORCH)
        code, out = run_cli(
            "run",
            "open",
            "--name",
            "r1",
            "--title",
            "T",
            "--session-name",
            "orch",
            "--permission-mode",
            "auto",
            "--full-grant",
        )
        assert code == 0, out
        assert out["run_id"] == "R1" and out["token"].startswith("run:R1.1.") and out["epoch"] == 1
        assert out["daemon"] == "started" and len(fake_popen.calls) == 2  # bootstrap, then the post-open ensure
        lease = _lease(db_path, "R1")
        assert lease["session_id"] == ORCH and lease["session_name"] == "orch" and lease["short_id"] is None
        assert lease["permission_mode"] == "auto" and lease["full_grant"] == 1
        assert lease["name_bound_session_id"] == ORCH and lease["epoch"] == 1
        assert sql(db_path, "SELECT state, title FROM runs")[0][:] == ("open", "T")

    def test_bad_name_duplicate_name_and_missing_session(self, migrated) -> None:
        assert run_cli("run", "open", "--name", "Bad Name", "--title", "t", "--session-id", "s")[1]["error"] == "USAGE"
        open_run("r1")
        assert run_cli("run", "open", "--name", "r1", "--title", "t", "--session-id", "s")[1]["error"] == "CONFLICT"
        code, out = run_cli("run", "open", "--name", "r2", "--title", "t")
        assert code == 2 and out["error"] == "USAGE" and "session" in out["detail"]


class TestClaim:
    def test_holder_reclaims_its_own_run(self, migrated, fake_claude) -> None:
        run_id, old = open_run()
        code, out = run_cli("run", "claim", "r1", "--session-id", ORCH)
        assert code == 0 and out["status"] == "reclaimed" and out["epoch"] == 2
        assert run_cli("run", "close", "--token", old)[0] == 3
        lease = _lease(migrated, run_id)
        assert lease["name_bound_session_id"] == ORCH and lease["pending_bind_session_id"] is None
        assert run_cli("session", "status", "--session-id", ORCH)[1]["state"] == "current"

    def test_absent_holder_is_replaced(self, migrated, fake_claude) -> None:
        run_id, old = open_run()
        fake_claude.listing = [agent("S-new", "new", id="short1")]
        code, out = run_cli("run", "claim", run_id, "--session-id", "S-new", "--permission-mode", "plan")
        assert code == 0 and out["status"] == "claimed"
        lease = _lease(migrated, run_id)
        assert lease["session_id"] == "S-new" and lease["name_bound_session_id"] == "S-new"
        assert lease["short_id"] == "short1" and lease["permission_mode"] == "plan"
        assert run_cli("session", "status", "--session-id", ORCH)[1]["state"] == "superseded"
        code, out = run_cli("run", "close", "--token", old)
        assert code == 3 and out["superseded_by"] == "S-new"

    @pytest.mark.parametrize("listing", ["alive", "failed"])
    def test_live_holder_without_takeover_conflicts(self, migrated, fake_claude, listing) -> None:
        run_id, _ = open_run(session_name="orch")
        fake_claude.listing = [agent(ORCH, "orch", id="sx")] if listing == "alive" else None
        code, out = run_cli("run", "claim", run_id, "--session-id", "S-new")
        assert code == 8 and out["error"] == "CONFLICT"
        assert out["holder_session_id"] == ORCH and out["holder_name"] == "orch"
        assert out["holder_short_id"] == ("sx" if listing == "alive" else None)
        assert _lease(migrated, run_id)["epoch"] == 1

    def test_takeover_leaves_the_name_binding_pending(self, migrated, fake_claude) -> None:
        run_id, old = open_run()
        fake_claude.listing = [agent(ORCH), agent("S-new")]
        code, out = run_cli("run", "claim", run_id, "--session-id", "S-new", "--takeover")
        assert code == 0 and out["status"] == "taken_over" and "archive" in out["instructions"]
        lease = _lease(migrated, run_id)
        assert lease["name_bound_session_id"] == ORCH and lease["pending_bind_session_id"] == "S-new"
        assert run_cli("run", "close", "--token", old)[0] == 3
        status = run_cli("session", "status", "--session-id", ORCH)[1]
        assert (status["role"], status["state"], status["run_id"]) == ("orchestrator", "superseded", run_id)

        new = out["token"]
        assert run_cli("run", "bind-name", run_id, "--token", new)[1]["error"] == "CONFLICT"  # still listed
        fake_claude.listing = None
        assert run_cli("run", "bind-name", run_id, "--token", new)[1]["error"] == "CONFLICT"  # unknown
        fake_claude.listing = [agent("S-new")]
        assert run_cli("run", "bind-name", "r2", "--token", new)[1]["error"] == "NOT_FOUND"
        code, out = run_cli("run", "bind-name", run_id, "--token", new)
        assert code == 0 and out["name_bound_session_id"] == "S-new"
        assert run_cli("run", "bind-name", run_id, "--token", new)[1]["error"] == "INVALID_STATE"

    def test_unknown_and_closed_runs(self, migrated) -> None:
        assert run_cli("run", "claim", "nope", "--session-id", "x")[1]["error"] == "NOT_FOUND"
        run_id, token = open_run()
        assert run_cli("run", "close", "--token", token)[1]["state"] == "closed"
        assert run_cli("run", "claim", run_id, "--session-id", ORCH)[1]["error"] == "INVALID_STATE"
        assert run_cli("run", "close", "--token", token)[1]["error"] == "INVALID_STATE"


class TestConcurrentClaims:
    def _race(self, migrated, fake_claude, sessions, takeover) -> list[str]:
        barrier = threading.Barrier(2)
        first_two = iter([True, True])

        def before_list() -> None:
            if next(first_two, False):
                barrier.wait(timeout=2)

        fake_claude.before_list = before_list
        results: list[str] = []

        def worker(sid: str) -> None:
            c = db.connect(migrated)
            try:
                runs.claim(
                    c,
                    __import__("control_plane").clock.FakeClock(),
                    fake_claude,
                    run_ref="r1",
                    session_id=sid,
                    takeover=takeover,
                    session_name=None,
                    permission_mode=None,
                    full_grant=False,
                )
                results.append("ok")
            except errors.CpError as exc:
                results.append(exc.code)
            finally:
                c.close()

        threads = [threading.Thread(target=worker, args=(s,)) for s in sessions]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2)
        return sorted(results)

    def test_two_takeovers_one_wins(self, migrated, fake_claude) -> None:
        open_run()
        fake_claude.listing = [agent(ORCH), agent("A"), agent("B")]
        assert self._race(migrated, fake_claude, ["A", "B"], takeover=True) == ["CONFLICT", "ok"]
        assert _lease(migrated, "R1")["epoch"] == 2

    def test_two_claims_on_a_dead_holder_one_wins(self, migrated, fake_claude) -> None:
        open_run()
        fake_claude.listing = [agent("A"), agent("B")]  # the holder is absent; both claimers are alive
        assert self._race(migrated, fake_claude, ["A", "B"], takeover=False) == ["CONFLICT", "ok"]
        assert _lease(migrated, "R1")["epoch"] == 2


class TestSetters:
    def test_set_mode(self, migrated) -> None:
        run_id, token = open_run(permission_mode="manual")
        out = run_cli("run", "set-mode", "--token", token, "--permission-mode", "auto", "--full-grant")[1]
        assert (out["permission_mode"], out["full_grant"]) == ("auto", True)
        out = run_cli("run", "set-mode", "--token", token, "--permission-mode", "plan")[1]
        assert (out["permission_mode"], out["full_grant"]) == ("plan", True)
        out = run_cli("run", "set-mode", "--token", token, "--permission-mode", "plan", "--no-full-grant")[1]
        assert out["full_grant"] is False

    def test_set_plan(self, migrated, tmp_path) -> None:
        run_id, token = open_run()
        open_run("r2", "S-2")
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n")
        out = run_cli("run", "set-plan", "--run", "r1", "--file", str(plan), "--token", token)[1]
        assert out == {"ok": True, "run_id": run_id, "bytes": 7}
        assert sql(migrated, "SELECT global_plan FROM runs WHERE id = ?", [run_id])[0][0] == "# plan\n"
        assert (
            run_cli("run", "set-plan", "--run", "r2", "--file", str(plan), "--token", token)[1]["error"] == "CONFLICT"
        )
        missing = str(tmp_path / "missing")
        assert run_cli("run", "set-plan", "--run", "r1", "--file", missing, "--token", token)[1]["error"] == "USAGE"


class TestSessionStatus:
    def test_missing_db_and_env_default(self, monkeypatch) -> None:
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "S-x")
        code, out = run_cli("session", "status")
        assert code == 0 and out["role"] == "none" and out["session_id"] == "S-x"
        assert run_cli("session", "status")[0] == 0
        monkeypatch.delenv("CLAUDE_CODE_SESSION_ID")
        assert run_cli("session", "status")[0] == 2

    def test_nodes_and_unknown(self, migrated) -> None:
        run_id, _ = open_run()
        insert_task(migrated, "T1", run_id)
        insert_node(migrated, "N1", run_id, "T1", session_id="S-a", state="superseded")
        insert_node(migrated, "N2", run_id, "T1", generation=2, session_id="S-b", state="idle")
        insert_node(migrated, "N3", run_id, "T1", generation=3, session_id="S-c", state="stopped")
        expected = {"S-a": "superseded", "S-b": "current", "S-c": "stopped"}
        for sid, state in expected.items():
            out = run_cli("session", "status", "--session-id", sid)[1]
            assert (out["role"], out["state"], out["task_id"]) == ("node", state, "T1")
        assert run_cli("session", "status", "--session-id", "S-zz")[1]["role"] == "none"
        assert runs.session_status(None, "x")["role"] == "none"


# --------------------------------------------------------------------------- review fix #01 (F23, F24)


@pytest.mark.parametrize("command", ["open", "claim"])
def test_a_daemon_start_error_still_prints_the_token(migrated, fake_popen, command: str) -> None:
    """F24: the run is opened or claimed; a failing ``ensure`` is reported, never the whole command's error."""
    run_id = open_run()[0] if command == "claim" else None

    def boom(argv: list[str], kwargs: dict) -> None:
        raise OSError("cannot spawn")

    fake_popen.on_call = boom
    if command == "open":
        code, out = run_cli("run", "open", "--name", "r2", "--title", "t", "--session-id", "S-2")
    else:
        code, out = run_cli("run", "claim", run_id, "--session-id", ORCH)
    assert code == 0 and out["token"] and out["daemon"] == "error"
    assert "cannot spawn" in out["daemon_error"]


@pytest.mark.parametrize(
    ("argv", "flag"),
    [
        (["msg", "pop", "--run", "R1", "--as", "orchestrator", "--token", "x"], "--visibility"),
        (["wait", "--run", "R1", "--token", "x"], "--poll"),
        (["wait", "--run", "R1", "--token", "x"], "--timeout"),
    ],
)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "abc"])
def test_float_flags_must_be_finite_and_positive(argv: list[str], flag: str, value: str) -> None:
    """F23."""
    code, out = run_cli(*argv, f"{flag}={value}")
    assert code == 2 and out["error"] == "USAGE"


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_lock_timeout_must_be_finite_and_not_negative(value: str) -> None:
    argv = ["lock", "acquire", "--name", "integration:QS_1", "--purpose", "p", "--token", "x", "--session-id", "S"]
    code, out = run_cli(*argv, f"--timeout={value}")
    assert code == 2 and out["error"] == "USAGE"
