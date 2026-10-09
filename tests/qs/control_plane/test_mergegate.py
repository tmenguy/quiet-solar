"""QS-406 T6b: the merge gate, ``halt clear``, the ``ask`` guard, the self-check (§4.2, §4.3, D2, D18, D19, AC 6, AC 7)."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from control_plane import activeloop, alerts, cli, codever, daemon, db, mergegate, runner, selfcheck, ticks, tools

from .conftest import CUR, NEXT, ORCH, FakeRunner, open_run, run_cli, sql
from .test_tools import W, _prep_merge, call_row, tool, w  # noqa: F401 — `w` is a fixture

# --------------------------------------------------------------------------- helpers


def _cp_file(main: Path, name: str = "a.py") -> Path:
    f = main / "scripts" / "qs" / "control_plane" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f"# {name}\n")
    return f


def _grow(f: Path) -> None:
    f.write_text(f.read_text() + "# changed\n")


def _seed(path: Path, key: str, value: Any) -> None:
    sql(path, "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", [key, json.dumps(value)])


def _meta(path: Path, key: str) -> Any:
    rows = sql(path, "SELECT value FROM meta WHERE key = ?", [key])
    return json.loads(rows[0][0]) if rows else None


def _record(path: Path, clock: Any, version: str, *, ok: bool, times: int = 1) -> None:
    c = db.connect(path)
    try:
        for _ in range(times):
            with db.write(c):
                mergegate.record(c, clock, version, ok=ok, failures=[] if ok else [{"step": "entry"}])
    finally:
        c.close()


def _merge(w: W, key: str = "m1") -> tuple[int, Any]:
    return tool(w, "merge", key)


# --------------------------------------------------------------------------- the gate (AC 6)


@pytest.mark.usefixtures("real_gate")
class TestGate:
    def test_no_record_is_busy_before_any_github_effect(self, w: W) -> None:
        _prep_merge(w)
        code, out = _merge(w)
        assert out["error"] == "BUSY" and "pending" in out["detail"], out
        assert call_row(w, "merge", "m1") is None  # the claim is released
        assert w.runner.matching("--json", "headRefOid") == [] and w.sim.effects("gh", "pr", "merge") == 0

    def test_end_to_end_pending_then_stuck(self, w: W) -> None:
        _prep_merge(w)
        assert _merge(w)[1]["error"] == "BUSY"
        w.clock.advance(mergegate.PENDING_ESCALATE_S + 1)
        code, out = _merge(w)  # same key
        assert out["error"] == "POLICY_REFUSED" and "has not run" in out["detail"]
        assert "ask the maintainer" in out["detail"] and "halt clear" not in out["detail"]

    def test_stuck_names_a_leftover_index_lock(self, w: W) -> None:
        _prep_merge(w)
        _merge(w)
        (w.main / ".git" / "index.lock").write_text("")
        w.clock.advance(mergegate.PENDING_ESCALATE_S + 1)
        out = _merge(w)[1]
        assert str(w.main / ".git" / "index.lock") in out["detail"] and "since 20" in out["detail"]

    def test_stuck_clause_survives_an_unstatable_file(self, w: W, monkeypatch) -> None:
        monkeypatch.setattr(codever, "git_busy", lambda main, **kw: "/nonexistent/.git/index.lock")
        assert tools._git_busy_clause(w.main).endswith("since ?: /nonexistent/.git/index.lock)")

    def test_stuck_clause_names_a_stale_index_lock_too(self, w: W) -> None:
        lock = w.main / ".git" / "index.lock"
        lock.write_text("")
        os.utime(lock, (1_000_000_000, 1_000_000_000))  # 2001: stale, ignored by the restart, named here
        assert codever.git_busy(w.main) is None and str(lock) in tools._git_busy_clause(w.main)

    @pytest.mark.parametrize("failures", [1, 2])
    def test_a_few_failures_are_busy_with_the_next_retry(self, w: W, failures: int) -> None:
        _prep_merge(w)
        _record(w.db, w.clock, codever.code_version(w.main), ok=False, times=failures)
        out = _merge(w)[1]
        assert out["error"] == "BUSY" and out["next_retry_at"]
        assert f"{failures} of {mergegate.SELFCHECK_ESCALATE_AFTER}" in out["detail"]

    def test_escalated_failures_refuse_without_naming_the_command(self, w: W) -> None:
        _prep_merge(w)
        _record(w.db, w.clock, codever.code_version(w.main), ok=False, times=mergegate.SELFCHECK_ESCALATE_AFTER)
        out = _merge(w)[1]
        assert out["error"] == "POLICY_REFUSED" and "halted" in out["detail"]
        assert "halt clear" not in out["detail"] and "ask the maintainer" in out["detail"]

    def test_a_pass_then_the_same_key_merges(self, w: W) -> None:
        _prep_merge(w)
        assert _merge(w)[1]["error"] == "BUSY"
        _record(w.db, w.clock, codever.code_version(w.main), ok=True)
        code, out = _merge(w)
        assert code == 0, out
        assert w.sim.effects("gh", "pr", "merge") == 1

    def test_after_a_revert_the_old_pending_mark_is_gone(self, w: W) -> None:
        _prep_merge(w)
        f = _cp_file(w.main)
        va = codever.code_version(w.main)
        assert _merge(w, "a1")[1]["error"] == "BUSY"  # vA seen pending at t0
        _record(w.db, w.clock, va, ok=True)  # vA passed: the mark is deleted
        _grow(f)
        _record(w.db, w.clock, codever.code_version(w.main), ok=True)  # vB passed, no tool merge
        w.clock.advance(mergegate.PENDING_ESCALATE_S + 1)
        f.write_text("# a.py\n")  # back to vA
        assert codever.code_version(w.main) == va
        assert _merge(w, "a2")[1]["error"] == "BUSY"

    def test_another_versions_pending_mark_is_replaced(self, w: W) -> None:
        _prep_merge(w)
        _seed(w.db, mergegate.PENDING, {"code_version": "vC", "since": "2026-01-01T00:00:00.000000Z"})
        out = _merge(w)[1]
        assert out["error"] == "BUSY"
        assert _meta(w.db, mergegate.PENDING) == {
            "code_version": codever.code_version(w.main),
            "since": db.now(w.clock),
        }

    def test_an_unreadable_pending_mark_is_replaced(self, w: W) -> None:
        _prep_merge(w)
        _seed(w.db, mergegate.PENDING, {"code_version": codever.code_version(w.main), "since": "x"})
        assert _merge(w)[1]["error"] == "BUSY"
        assert _meta(w.db, mergegate.PENDING)["since"] == db.now(w.clock)

    def test_a_cp_file_change_closes_the_gate_again(self, w: W) -> None:
        _prep_merge(w)
        f = _cp_file(w.main)
        _record(w.db, w.clock, codever.code_version(w.main), ok=True)
        _grow(f)
        assert _merge(w)[1]["error"] == "BUSY"


class TestMeta:
    def test_unreadable_values_are_absent(self, conn: sqlite3.Connection, migrated: Path) -> None:
        sql(migrated, "INSERT INTO meta (key, value) VALUES ('selfcheck', 'not json'), ('selfcheck_override', '[1]')")
        assert mergegate.get(conn, mergegate.RECORD) is None and mergegate.get(conn, mergegate.OVERRIDE) is None

    def test_next_retry_at_of_a_bad_stamp(self) -> None:
        assert mergegate.next_retry_at({"at": "x"}) is None
        assert mergegate.next_retry_at({"at": "2026-10-03T12:00:00.000000Z"}) == "2026-10-03T12:10:00.000000Z"


# --------------------------------------------------------------------------- halt clear


@pytest.mark.usefixtures("real_gate")
class TestHaltClear:
    def test_opens_the_gate_for_this_version_only(self, w: W) -> None:
        _prep_merge(w)
        f = _cp_file(w.main)
        version = codever.code_version(w.main)
        _record(w.db, w.clock, version, ok=False, times=mergegate.SELFCHECK_ESCALATE_AFTER)
        code, out = run_cli("halt", "clear", "--reason", "known flake")
        assert code == 0 and out["override"]["code_version"] == version and out["previous"]["ok"] is False
        assert _meta(w.db, mergegate.PENDING) is None
        assert _merge(w, "h1")[0] == 0
        _grow(f)  # other code: the override does not apply
        c = db.connect(w.db)
        try:
            with db.write(c):
                assert mergegate.merge_allowed(c, codever.code_version(w.main), w.clock).state == "pending"
        finally:
            c.close()

    def test_clears_the_selfcheck_alert(self, conn, migrated: Path, fake_clock) -> None:
        r1, _ = open_run()
        cond = alerts.Condition(alerts.SELFCHECK_FAILED, "v", (r1,), {})
        alerts.sync(conn, fake_clock, kinds={alerts.SELFCHECK_FAILED}, active=[cond])
        assert run_cli("halt", "clear", "--reason", "r")[0] == 0
        assert sql(migrated, "SELECT count(*) FROM alerts WHERE cleared_at IS NULL")[0][0] == 0

    def test_a_missing_db_is_not_found(self) -> None:
        assert run_cli("halt", "clear", "--reason", "r")[1]["error"] == "NOT_FOUND"

    @pytest.mark.parametrize("argv", [["halt", "clear"], ["halt", "clear", "--reason", "  "]])
    def test_usage(self, migrated, argv: list[str]) -> None:
        assert run_cli(*argv)[1]["error"] == "USAGE"


# --------------------------------------------------------------------------- the ask guard (D2)


def _pre(command: str, mode: str | None = "default", session: str | None = None) -> tuple[int, Any]:
    payload: dict[str, Any] = {"session_id": session, "tool_name": "Bash", "tool_input": {"command": command}}
    if mode is not None:
        payload["permission_mode"] = mode
    return run_cli("hook", "pre-tool-use", stdin=json.dumps(payload))


def _decision(result: tuple[int, Any]) -> str | None:
    code, out = result
    assert code == 0
    return None if out is None else out["hookSpecificOutput"]["permissionDecision"]


class TestAskGuard:
    @pytest.mark.parametrize(
        "command",
        [
            "/main/venv/bin/python /main/scripts/qs/cp.py halt clear --reason 'flake'",
            "cd /x && python cp.py halt clear --reason r",
            "python 'scripts/qs/cp.py' halt clear --reason r",
        ],
    )
    def test_asks_in_an_ask_mode_with_no_db(self, command: str) -> None:
        for mode in ("default", "acceptEdits", "auto", "bypassPermissions"):  # T1: a hook's `ask` prompts in all
            assert _decision(_pre(command, mode=mode)) == "ask"

    @pytest.mark.parametrize("mode", ["plan", "dontAsk"])
    def test_denies_with_a_hint_elsewhere(self, mode: str | None) -> None:
        code, out = _pre("python scripts/qs/cp.py halt clear --reason r", mode=mode)
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "switch this session to default mode" in out["hookSpecificOutput"]["permissionDecisionReason"]

    def test_a_missing_mode_denies_and_points_to_a_terminal(self) -> None:
        code, out = _pre("python scripts/qs/cp.py restore --confirm", mode=None)
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
        reason = out["hookSpecificOutput"]["permissionDecisionReason"]
        assert "terminal" in reason and "switch this session" not in reason

    def test_other_commands_pass(self) -> None:
        assert _decision(_pre("python scripts/qs/cp.py halt")) is None
        assert _decision(_pre("python cp.pyx halt clear")) is None
        assert _decision(_pre("echo 'cp.py halt clear'")) is None  # one quoted word
        assert hooks_confirm("Edit", {"command": "cp.py halt clear"}) is None

    def test_a_db_access_segment_is_still_denied(self) -> None:
        assert _decision(_pre("python scripts/qs/cp.py halt clear --reason r; sqlite3 harness_state.db")) == "deny"

    def test_asks_under_a_newer_db_and_records_on_a_current_one(self, migrated: Path) -> None:
        assert _decision(_pre("python cp.py halt clear --reason r", session=ORCH)) == "ask"
        [row] = [dict(r) for r in sql(migrated, "SELECT * FROM hook_events")]
        assert row["decision"] == "allow" and json.loads(row["detail"])["kind"] == "maintainer_ask"
        sql(migrated, f"PRAGMA user_version = {NEXT}")
        assert _decision(_pre("python cp.py halt clear --reason r", session=ORCH)) == "ask"

    def test_an_orchestrator_chaining_a_merge_is_denied(self, migrated: Path) -> None:
        open_run()
        assert _decision(_pre("python cp.py halt clear --reason x && gh pr merge 5", session=ORCH)) == "deny"


def hooks_confirm(tool_name: str, tool_input: dict[str, Any]) -> str | None:
    from control_plane import hooks

    return hooks.maintainer_confirm(tool_name, tool_input)


# --------------------------------------------------------------------------- the self-check (AC 7)


def _good_version_output() -> str:
    return json.dumps(
        {"schema_version": CUR, "tools": sorted(tools.REGISTRY), "tick_hooks": sorted(activeloop.BUILTIN_NAMES)}
    )


@pytest.fixture
def check(conn: sqlite3.Connection, migrated: Path, fake_main: Path, fake_runner: FakeRunner, monkeypatch) -> Any:
    """A daemon whose loaded code is the disk code, with a healthy `cp.py version`."""
    _cp_file(fake_main)
    monkeypatch.setattr(daemon, "_loaded_version", codever.code_version(fake_main))
    fake_runner.on(("cp.py", "version"), _good_version_output())
    return SimpleNamespace(conn=conn, db=migrated, main=fake_main, runner=fake_runner)


def _entry_calls(runner_: FakeRunner) -> int:
    return len(runner_.matching("cp.py", "version"))


def _open_alerts(path: Path) -> list[tuple[str, str]]:
    return [(r[0], r[1]) for r in sql(path, "SELECT run_id, kind FROM alerts WHERE cleared_at IS NULL ORDER BY id")]


class TestSelfCheck:
    def test_a_pass_is_recorded_once(self, check, fake_clock) -> None:
        _seed(check.db, mergegate.PENDING, {"code_version": "v", "since": "x"})
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        rec = _meta(check.db, mergegate.RECORD)
        assert rec["ok"] is True and rec["tries"] == 0 and rec["code_version"] == codever.code_version(check.main)
        assert _meta(check.db, mergegate.PENDING) is None
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        assert _entry_calls(check.runner) == 1  # a recorded pass is not re-run

    def test_an_override_is_not_run(self, check, fake_clock) -> None:
        _seed(
            check.db, mergegate.OVERRIDE, {"code_version": codever.code_version(check.main), "reason": "r", "at": "x"}
        )
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        assert _entry_calls(check.runner) == 0

    def test_the_entry_runs_the_venv_python_when_present(self, check, fake_clock) -> None:
        venv = check.main / "venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").write_text("")
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        call = check.runner.matching("cp.py", "version")[0]
        assert call.argv[0] == str(venv / "python") and call.cwd == str(check.main)
        assert call.timeout == ticks.HOOK_SUBPROCESS_S
        bare = check.runner.matching("cp.py", "version")
        assert bare and sys.executable not in bare[0].argv

    @pytest.mark.parametrize(
        "output",
        [
            runner.RunResult(1, "", "boom"),
            json.dumps({"schema_version": CUR, "tools": ["merge"], "tick_hooks": sorted(activeloop.BUILTIN_NAMES)}),
            json.dumps({"schema_version": CUR, "tools": sorted(tools.REGISTRY), "tick_hooks": ["code_version"]}),
            json.dumps({"schema_version": NEXT, "tools": [], "tick_hooks": []}),
            "not json",
        ],
    )
    def test_the_entry_step_fails_on_its_fixture(self, check, fake_clock, output: Any) -> None:
        check.runner.on(("cp.py", "version"), output)
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        rec = _meta(check.db, mergegate.RECORD)
        assert rec["ok"] is False and rec["tries"] == 1 and [f["step"] for f in rec["failures"]] == ["entry"]

    def test_the_schema_step_fails_on_a_wrong_version(self, check, fake_clock, monkeypatch) -> None:
        monkeypatch.setattr(selfcheck, "migrations", SimpleNamespace(current_schema_version=lambda: NEXT))
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        assert "schema" in [f["step"] for f in _meta(check.db, mergegate.RECORD)["failures"]]

    def test_the_schema_step_reads_quick_check(self) -> None:
        class Conn:
            def execute(self, statement: str) -> Any:
                value = CUR if "user_version" in statement else "*** corrupt page 3"
                return SimpleNamespace(fetchone=lambda: (value,))

        with pytest.raises(selfcheck.CheckFailed, match="corrupt"):
            selfcheck._step_schema(Conn(), activeloop.seams())  # type: ignore[arg-type]

    @pytest.mark.parametrize("kind", ["TypeError", "ImportError", "missing"])
    def test_the_registry_step_records_any_exception(self, check, fake_clock, monkeypatch, kind: str) -> None:
        if kind == "TypeError":

            def broken() -> Any:
                raise TypeError("bad parser")

            monkeypatch.setattr(cli, "build_parser", broken)
        elif kind == "ImportError":
            monkeypatch.setitem(sys.modules, "control_plane.cli", None)
        else:
            monkeypatch.delitem(tools.REGISTRY, "merge")
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        failures = _meta(check.db, mergegate.RECORD)["failures"]
        assert "registry" in [f["step"] for f in failures]
        if kind != "missing":
            assert {f["type"] for f in failures if f["step"] == "registry"} <= {kind, "ModuleNotFoundError"}

    def test_a_failure_alerts_every_open_run_at_once(self, check, fake_clock) -> None:
        r1, _ = open_run("r1", "S-1")
        r2, _ = open_run("r2", "S-2")
        check.runner.on(("cp.py", "version"), runner.RunResult(1, "", "boom"))
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        assert _open_alerts(check.db) == [(r1, "selfcheck_failed"), (r2, "selfcheck_failed")]
        r3, _ = open_run("r3", "S-3")
        selfcheck.selfcheck_hook(check.conn, fake_clock)  # a run opened later still gets it
        assert (r3, "selfcheck_failed") in _open_alerts(check.db)

    def test_retries_are_spaced_and_never_stop(self, check, fake_clock) -> None:
        check.runner.on(("cp.py", "version"), runner.RunResult(1, "", "boom"))
        for attempt in range(1, 6):
            selfcheck.selfcheck_hook(check.conn, fake_clock)
            assert _meta(check.db, mergegate.RECORD)["tries"] == attempt
            fake_clock.advance(mergegate.SELFCHECK_RETRY_S - 1)
            daemon._reset_for_tests()  # a daemon restart: the throttle is the DB's, not memory
            activeloop._reset_for_tests()
            daemon._loaded_version = codever.code_version(check.main)
            selfcheck.selfcheck_hook(check.conn, fake_clock)
            assert _entry_calls(check.runner) == attempt  # not before at + SELFCHECK_RETRY_S
            fake_clock.advance(1)

    def test_an_unreadable_record_stamp_is_due(self, check, fake_clock) -> None:
        version = codever.code_version(check.main)
        _seed(check.db, mergegate.RECORD, {"code_version": version, "ok": False, "at": "x", "tries": 1})
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        assert _entry_calls(check.runner) == 1

    @pytest.mark.parametrize("how", ["pass", "override", "version"])
    def test_the_alert_clears(self, check, fake_clock, how: str) -> None:
        open_run()
        check.runner.on(("cp.py", "version"), runner.RunResult(1, "", "boom"))
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        assert _open_alerts(check.db)
        if how == "pass":
            check.runner.on(("cp.py", "version"), _good_version_output())
            fake_clock.advance(mergegate.SELFCHECK_RETRY_S)
        elif how == "override":
            assert run_cli("halt", "clear", "--reason", "r")[0] == 0
            _record(check.db, fake_clock, codever.code_version(check.main), ok=False)  # still failing, overridden
        else:
            _grow(check.main / "scripts" / "qs" / "control_plane" / "a.py")
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        assert _open_alerts(check.db) == []

    def test_a_new_version_after_a_failure_is_checked_at_once(self, check, fake_clock) -> None:
        check.runner.on(("cp.py", "version"), runner.RunResult(1, "", "boom"))
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        _grow(check.main / "scripts" / "qs" / "control_plane" / "a.py")
        daemon._loaded_version = codever.code_version(check.main)  # the restarted daemon
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        rec = _meta(check.db, mergegate.RECORD)
        assert _entry_calls(check.runner) == 2 and rec["tries"] == 1 and rec["code_version"] == daemon._loaded_version

    def test_skipped_while_a_restart_is_pending(self, check, fake_clock, monkeypatch) -> None:
        monkeypatch.setattr(daemon, "_loaded_version", "older")
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        monkeypatch.setattr(daemon, "_loaded_version", None)
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        assert _entry_calls(check.runner) == 0 and _meta(check.db, mergegate.RECORD) is None

    def test_a_change_during_the_steps_records_nothing(self, check, fake_clock) -> None:
        def pull(call: Any) -> runner.RunResult:
            _grow(check.main / "scripts" / "qs" / "control_plane" / "a.py")
            return runner.RunResult(0, _good_version_output(), "")

        check.runner.on(("cp.py", "version"), pull)
        selfcheck.selfcheck_hook(check.conn, fake_clock)
        assert _meta(check.db, mergegate.RECORD) is None

    @pytest.mark.usefixtures("active_loop")
    def test_bootstrap_the_first_tick_records_a_selfcheck(
        self, invoke, migrated, fake_main, fake_runner, monkeypatch
    ) -> None:
        fake_runner.on(("cp.py", "version"), _good_version_output())
        monkeypatch.setattr(daemon, "IDLE_EXIT_S", 1.0)
        assert invoke("daemon")[0] == 0
        assert _meta(migrated, mergegate.RECORD)["ok"] is True
