"""Checkpoint 7: the task tree — transitions, history, deps, criteria, work items (AC1 part, AC17 part)."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from control_plane import clock, db, errors, schema_v1, tasks

from .conftest import insert_node, open_run, run_cli, sql

CHAIN = ["proposed", "ready", "planning", "contracted", "building", "ready_to_merge", "merged", "validated"]
TERMINAL = {"merged", "validated", "dropped"}


def _expected(cur: str, to: str) -> bool:
    if (cur, to) in zip(CHAIN, CHAIN[1:]):
        return True
    if cur not in TERMINAL and to == "dropped":
        return True
    return cur not in TERMINAL and cur != "blocked" and to == "blocked"


@pytest.mark.parametrize(("cur", "to"), list(itertools.product(schema_v1.TASK_STATES, schema_v1.TASK_STATES)))
def test_transition_table(cur: str, to: str) -> None:
    blocked_from = "building" if cur == "blocked" else None
    if _expected(cur, to):
        new, bf = tasks.plan_transition(cur, to, blocked_from)
        assert new == to and bf == (cur if to == "blocked" else None)
    else:
        with pytest.raises(errors.CpError) as exc:
            tasks.plan_transition(cur, to, blocked_from)
        assert exc.value.code == "INVALID_STATE"


def test_unblock() -> None:
    assert tasks.plan_transition("blocked", "unblock", "planning") == ("planning", None)
    for cur, bf in (("building", None), ("blocked", None)):
        with pytest.raises(errors.CpError):
            tasks.plan_transition(cur, "unblock", bf)


@pytest.mark.parametrize(
    ("cur", "new", "bf", "allowed"),
    [
        ("planning", "contracted", None, True),
        ("contracted", "building", None, True),
        ("building", "ready_to_merge", None, True),
        ("ready_to_merge", "blocked", None, True),
        ("blocked", "building", "building", True),
        ("blocked", "ready", "ready", False),
        ("ready", "planning", None, False),
        ("ready_to_merge", "merged", None, False),
        ("building", "dropped", None, False),
        ("proposed", "blocked", None, False),
    ],
)
def test_node_range(cur: str, new: str, bf: str | None, allowed: bool) -> None:
    assert tasks.node_may(cur, new, bf) is allowed


@pytest.fixture
def run(migrated: Path) -> tuple[str, str]:
    return open_run()


def add(token: str, *extra: str, title: str = "t", kind: str = "feature") -> dict:
    code, out = run_cli("task", "add", "--title", title, "--kind", kind, "--token", token, *extra)
    assert code == 0, out
    return out


class TestAddAndState:
    def test_add_with_history(self, migrated, run) -> None:
        run_id, token = run
        out = add(token, "--run", run_id, "--target", "factory", "--issue", "12", "--lane", "feature-factory")
        assert out == {"ok": True, "task_id": "T1", "item_k": None, "run_id": run_id}
        row = dict(sql(migrated, "SELECT * FROM tasks WHERE id = 'T1'")[0])
        assert (row["state"], row["issue_number"], row["lane"], row["target"]) == (
            "proposed",
            12,
            "feature-factory",
            "factory",
        )
        hist = sql(migrated, "SELECT actor, from_state, to_state FROM task_history")
        assert [tuple(h) for h in hist] == [("orchestrator", None, "proposed")]

    def test_later_task_and_parent(self, migrated, run) -> None:
        _, token = run
        assert add(token)["run_id"] is None
        assert add(token, "--parent", "T1")["task_id"] == "T2"
        assert (
            run_cli("task", "add", "--title", "x", "--kind", "feature", "--parent", "T9", "--token", token)[1]["error"]
            == "NOT_FOUND"
        )
        assert run_cli("task", "add", "--title", "x", "--kind", "nope", "--token", token)[0] == 2

    def test_add_validations(self, migrated, run) -> None:
        run_id, token = run
        c = db.connect(migrated)
        with pytest.raises(errors.CpError) as exc:
            tasks.add(
                c,
                clock.FakeClock(),
                token=token,
                run_ref=None,
                title="x",
                kind="nope",
                target=None,
                parent=None,
                issue=None,
                lane=None,
                deliverable=False,
                item_of=None,
            )
        c.close()
        assert exc.value.code == "USAGE"
        open_run("r2", "S-2")
        assert (
            run_cli("task", "add", "--run", "r2", "--title", "x", "--kind", "bug", "--token", token)[1]["error"]
            == "CONFLICT"
        )

    def test_state_transitions_record_history(self, migrated, run) -> None:
        run_id, token = run
        add(token, "--run", run_id)
        for to in ("ready", "planning", "blocked", "unblock", "contracted"):
            code, out = run_cli("task", "state", "--task", "T1", "--to", to, "--note", "n", "--token", token)
            assert code == 0, out
        assert sql(migrated, "SELECT state, blocked_from FROM tasks")[0][:] == ("contracted", None)
        hist = [tuple(r) for r in sql(migrated, "SELECT from_state, to_state FROM task_history ORDER BY id")]
        assert hist[-3:] == [("planning", "blocked"), ("blocked", "planning"), ("planning", "contracted")]
        code, out = run_cli("task", "state", "--task", "T1", "--to", "merged", "--token", token)
        assert code == 8 and out["error"] == "INVALID_STATE"

    def test_node_moves_its_own_task_within_range(self, migrated, run) -> None:
        run_id, token = run
        add(token, "--run", run_id)
        add(token, "--run", run_id)
        for to in ("ready", "planning"):
            run_cli("task", "state", "--task", "T1", "--to", to, "--token", token)
        node = insert_node(migrated, "N1", run_id, "T1")
        assert run_cli("task", "state", "--task", "T1", "--to", "contracted", "--token", node)[0] == 0
        assert run_cli("task", "state", "--task", "T1", "--to", "dropped", "--token", node)[1]["error"] == "CONFLICT"
        assert run_cli("task", "state", "--task", "T2", "--to", "ready", "--token", node)[1]["error"] == "CONFLICT"
        hist = sql(migrated, "SELECT actor FROM task_history WHERE to_state = 'contracted'")
        assert hist[0][0] == "node:T1"
        sql(migrated, "UPDATE nodes SET state = 'stopped'")
        assert run_cli("task", "state", "--task", "T1", "--to", "building", "--token", node)[0] == 4


class TestFieldsDepsMembership:
    def test_task_set(self, migrated, run) -> None:
        run_id, token = run
        add(token, "--run", run_id)
        out = run_cli(
            "task",
            "set",
            "--task",
            "T1",
            "--issue",
            "5",
            "--worktree",
            "/w",
            "--branch",
            "QS_5",
            "--pr-number",
            "9",
            "--pr-url",
            "u",
            "--ci-state",
            "green",
            "--ci-sha",
            "abc",
            "--token",
            token,
        )[1]
        assert out["updated"] == ["branch", "ci_sha", "ci_state", "issue_number", "pr_number", "pr_url", "worktree"]
        row = dict(sql(migrated, "SELECT * FROM tasks")[0])
        assert (row["pr_number"], row["ci_sha"], row["worktree"]) == (9, "abc", "/w")
        assert run_cli("task", "set", "--task", "T1", "--pr-number", "9", "--token", token)[1]["error"] == "USAGE"
        assert run_cli("task", "set", "--task", "T1", "--ci-sha", "x", "--token", token)[1]["error"] == "USAGE"
        assert run_cli("task", "set", "--task", "T1", "--token", token)[1]["updated"] == []
        c = db.connect(migrated)
        with pytest.raises(errors.CpError), db.write(c):
            tasks.update_fields(c, clock.FakeClock(), "T1", {"state": "x"})
        c.close()

    def test_deps(self, migrated, run) -> None:
        run_id, token = run
        add(token, "--run", run_id)
        add(token, "--run", run_id)
        assert run_cli("task", "dep", "add", "--task", "T1", "--on", "T2", "--token", token)[0] == 0
        assert run_cli("task", "dep", "add", "--task", "T1", "--on", "T2", "--token", token)[0] == 0
        assert sql(migrated, "SELECT count(*) FROM task_deps")[0][0] == 1
        assert run_cli("task", "dep", "add", "--task", "T1", "--on", "T1", "--token", token)[1]["error"] == "USAGE"
        assert run_cli("task", "dep", "add", "--task", "T1", "--on", "T9", "--token", token)[1]["error"] == "NOT_FOUND"
        assert run_cli("task", "dep", "remove", "--task", "T1", "--on", "T2", "--token", token)[0] == 0
        assert sql(migrated, "SELECT count(*) FROM task_deps")[0][0] == 0

    @pytest.mark.parametrize(("cmd", "table"), [("root", "run_roots"), ("work-list", "work_list")])
    def test_membership(self, migrated, run, cmd: str, table: str) -> None:
        run_id, token = run
        add(token, "--run", run_id)
        assert run_cli("task", cmd, "add", "--run", run_id, "--task", "T1", "--token", token)[0] == 0
        assert sql(migrated, f"SELECT run_id, task_id FROM {table}")[0][:] == (run_id, "T1")
        assert run_cli("task", cmd, "remove", "--run", run_id, "--task", "T1", "--token", token)[0] == 0
        assert sql(migrated, f"SELECT count(*) FROM {table}")[0][0] == 0


class TestCriteria:
    def test_set_validate_state(self, migrated, run, tmp_path) -> None:
        run_id, token = run
        add(token, "--run", run_id)
        f = tmp_path / "c.txt"
        f.write_text("first\n\nsecond\n")
        assert run_cli("criteria", "set", "--task", "T1", "--file", str(f), "--token", token)[1]["count"] == 2
        f.write_text("only\n")
        assert run_cli("criteria", "set", "--task", "T1", "--file", str(f), "--token", token)[1]["count"] == 1
        assert run_cli("criteria", "validate", "--task", "T1", "--token", token)[1]["validated"] == 1
        assert (
            run_cli("criteria", "set", "--task", "T1", "--file", str(f), "--token", token)[1]["error"]
            == "INVALID_STATE"
        )
        assert run_cli("criteria", "state", "--task", "T1", "--idx", "1", "--to", "met", "--token", token)[0] == 0
        assert sql(migrated, "SELECT text, state FROM criteria")[0][:] == ("only", "met")
        assert (
            run_cli("criteria", "state", "--task", "T1", "--idx", "2", "--to", "met", "--token", token)[1]["error"]
            == "NOT_FOUND"
        )
        assert (
            run_cli("criteria", "state", "--task", "T1", "--idx", "1", "--to", "bad", "--token", token)[1]["error"]
            == "USAGE"
        )

    def test_validate_without_criteria(self, migrated, run) -> None:
        run_id, token = run
        add(token, "--run", run_id)
        assert run_cli("criteria", "validate", "--task", "T1", "--token", token)[1]["error"] == "INVALID_STATE"


class TestWorkItems:
    def test_item_k_is_never_reused(self, migrated, run) -> None:
        run_id, token = run
        add(token, "--run", run_id, "--deliverable")
        assert add(token, "--run", run_id, "--item-of", "T1")["item_k"] == 1
        assert add(token, "--run", run_id, "--item-of", "T1")["item_k"] == 2
        run_cli("task", "state", "--task", "T3", "--to", "dropped", "--token", token)
        assert add(token, "--run", run_id, "--item-of", "T1")["item_k"] == 3
        assert sql(migrated, "SELECT deliverable_id, item_k FROM tasks WHERE id = 'T4'")[0][:] == ("T1", 3)
        code, out = run_cli("task", "add", "--title", "x", "--kind", "feature", "--item-of", "T2", "--token", token)
        assert out["error"] == "INVALID_STATE"

    def test_record_integration(self, migrated, run, conn) -> None:
        run_id, token = run
        add(token, "--run", run_id, "--deliverable")
        add(token, "--run", run_id, "--item-of", "T1")
        with db.write(conn):
            rid = tasks.record_integration(
                conn,
                clock.FakeClock(),
                item_task_id="T2",
                deliverable_id="T1",
                item_tip="abc",
                merge_commit="def",
                result="ok",
                tool_call_key="item:T2:abc",
            )
        row = dict(sql(migrated, "SELECT * FROM integrations WHERE id = ?", [rid])[0])
        assert (row["item_tip"], row["merge_commit"], row["result"]) == ("abc", "def", "ok")


def test_apply_transition_compare_and_set(migrated, run, conn) -> None:
    run_id, token = run
    add(token, "--run", run_id)
    with pytest.raises(errors.CpError) as exc, db.write(conn):
        tasks.apply_transition(conn, clock.FakeClock(), "T1", "ready", actor="x", node=False, expect="building")
    assert exc.value.code == "INVALID_STATE"
    with db.write(conn):
        out = tasks.apply_transition(conn, clock.FakeClock(), "T1", "ready", actor="x", node=False, expect="proposed")
    assert out == {"task_id": "T1", "from": "proposed", "to": "ready"}


# --------------------------------------------------------------------------- review fix #02 (G6)


class TestReviewFix02:
    def test_parent_and_item_of_are_scoped_to_the_callers_run(self, migrated, run) -> None:
        run_a, token_a = run
        add(token_a, "--run", run_a, "--deliverable")  # T1, run A's deliverable
        add(token_a)  # T2, a later task (no run): anyone may attach to it
        run_b, token_b = open_run("r2", session="S-other")
        for flag in ("--item-of", "--parent"):
            code, out = run_cli("task", "add", "--title", "x", "--kind", "feature", flag, "T1", "--token", token_b)
            assert code == 8 and out["error"] == "CONFLICT" and run_a in out["detail"], flag
        assert sql(migrated, "SELECT next_item_k FROM tasks WHERE id = 'T1'")[0][0] == 1
        assert sql(migrated, "SELECT count(*) FROM tasks")[0][0] == 2
        assert add(token_b, "--parent", "T2")["task_id"] == "T3"
        assert add(token_a, "--run", run_a, "--item-of", "T1")["item_k"] == 1
