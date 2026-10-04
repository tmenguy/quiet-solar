"""Checkpoint 8: node transitions, ``refresh``, stop / take-over / hand-back (AC9, AC12)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
from control_plane import db, errors, nodes, schema_v1

from .conftest import agent, insert_node, insert_task, open_run, run_cli, sql

ALLOWED = {
    ("spawning", "reaped"),
    ("running", "reaped"),
    ("idle", "reaped"),
    ("reaped", "spawning"),
    ("spawning", "running"),
    ("running", "idle"),
    ("idle", "running"),
    ("running", "taken_over"),
    ("idle", "taken_over"),
    ("taken_over", "running"),
}
for _s in ("spawning", "running", "idle", "taken_over", "reaped"):
    ALLOWED |= {(_s, "stopped"), (_s, "superseded")}


@pytest.mark.parametrize(("cur", "to"), list(itertools.product(schema_v1.NODE_STATES, schema_v1.NODE_STATES)))
def test_transition_table(cur: str, to: str) -> None:
    if (cur, to) in ALLOWED:
        nodes.check(cur, to)
    else:
        with pytest.raises(errors.CpError) as exc:
            nodes.check(cur, to)
        assert exc.value.code == "INVALID_STATE"


@pytest.fixture
def world(migrated: Path, tmp_path: Path) -> dict:
    run_id, token = open_run()
    insert_task(migrated, "T1", run_id, is_deliverable=1, branch="QS_5")
    insert_task(migrated, "T2", run_id, deliverable_id="T1", item_k=1)
    insert_task(migrated, "T3", run_id)
    node = insert_node(migrated, "N1", run_id, "T1", session_id="S-n1")
    summary = tmp_path / "s.md"
    summary.write_text("all good")
    return {"run": run_id, "token": token, "node": node, "db": migrated, "summary": str(summary)}


class TestMoveAndCurrent:
    def test_move(self, world, conn, fake_clock) -> None:
        with db.write(conn):
            nodes.move(conn, fake_clock, "N1", "taken_over", taken_over_at="t")
        row = dict(sql(world["db"], "SELECT state, taken_over_at FROM nodes")[0])
        assert row == {"state": "taken_over", "taken_over_at": "t"}
        for node_id, kw, code in (("N9", {}, "NOT_FOUND"), ("N1", {"expect": "idle"}, "INVALID_STATE")):
            with pytest.raises(errors.CpError) as exc, db.write(conn):
                nodes.move(conn, fake_clock, node_id, "running", **kw)
            assert exc.value.code == code

    def test_current_and_deliverable(self, world, conn) -> None:
        insert_node(world["db"], "N2", world["run"], "T1", generation=2)
        assert nodes.current(conn, "T1")["id"] == "N2"
        with pytest.raises(errors.CpError):
            nodes.current(conn, "T3")
        assert nodes.deliverable_of(conn, "T2")["id"] == "T1"
        assert nodes.deliverable_of(conn, "T1")["id"] == "T1"
        assert nodes.deliverable_of(conn, "T3")["id"] == "T3"


class TestRefresh:
    def test_listing_failure_changes_nothing(self, world, conn, fake_clock) -> None:
        assert nodes.refresh(conn, fake_clock, None) == {"refreshed": False, "changes": []}

    def test_reconciles_running_idle_reaped_only(self, world, conn, fake_clock) -> None:
        db_ = world["db"]
        insert_task(db_, "T4", world["run"])
        insert_task(db_, "T5", world["run"])
        insert_task(db_, "T6", world["run"])
        insert_node(db_, "N4", world["run"], "T4", state="idle", session_id="S-4")
        insert_node(db_, "N5", world["run"], "T5", state="spawning")
        insert_node(db_, "N6", world["run"], "T6", state="taken_over")
        listing = [agent("S-n1", status="idle"), agent("S-4", status="busy")]
        out = nodes.refresh(conn, fake_clock, listing, world["run"])
        assert sorted(out["changes"]) == [("N1", "running", "idle"), ("N4", "idle", "running")]
        out = nodes.refresh(conn, fake_clock, [agent("x", name="R1-T4-g1", status="busy")])
        assert out["changes"] == [("N1", "idle", "reaped")]  # N4 is listed by name, still running
        states = dict(sql(db_, "SELECT id, state FROM nodes"))
        assert states == {"N1": "reaped", "N4": "running", "N5": "spawning", "N6": "taken_over"}
        assert nodes.refresh(conn, fake_clock, [], "R9")["changes"] == []


class TestCommands:
    def test_stop(self, world) -> None:
        code, out = run_cli("node", "stop", "--task", "T1", "--token", world["token"])
        assert code == 0 and out["state"] == "stopped"
        msg = dict(sql(world["db"], "SELECT * FROM messages WHERE id = ?", [out["message_id"]])[0])
        assert (msg["recipient"], msg["kind"]) == ("node:T1", "stop")
        assert run_cli("node", "stop", "--task", "T1", "--token", world["token"])[1]["error"] == "INVALID_STATE"
        assert run_cli("node", "stop", "--task", "T3", "--token", world["token"])[1]["error"] == "NOT_FOUND"
        assert run_cli("task", "state", "--task", "T1", "--to", "blocked", "--token", world["node"])[0] == 4
        assert run_cli("session", "status", "--session-id", "S-n1")[1]["state"] == "stopped"

    def test_take_over_and_hand_back(self, world) -> None:
        code, out = run_cli("node", "take-over", "--task", "T1", "--token", world["token"])
        assert code == 0 and out["state"] == "taken_over"
        info = dict(sql(world["db"], "SELECT * FROM messages WHERE id = ?", [out["message_id"]])[0])
        assert (info["recipient"], info["kind"]) == ("orchestrator", "info")
        assert run_cli("node", "take-over", "--task", "T1", "--token", world["token"])[1]["error"] == "INVALID_STATE"
        code, out = run_cli(
            "node", "hand-back", "--task", "T1", "--summary-file", world["summary"], "--token", world["node"]
        )
        assert code == 0 and out["state"] == "running"
        msg = dict(sql(world["db"], "SELECT * FROM messages WHERE id = ?", [out["message_id"]])[0])
        assert msg["kind"] == "hand_back" and json.loads(msg["payload"])["summary"] == "all good"
        row = dict(sql(world["db"], "SELECT taken_over_at, handed_back_at FROM nodes")[0])
        assert row["taken_over_at"] and row["handed_back_at"]
        again = run_cli(
            "node", "hand-back", "--task", "T1", "--summary-file", world["summary"], "--token", world["token"]
        )
        assert again[1]["error"] == "INVALID_STATE"

    def test_node_cannot_stop_or_take_over(self, world) -> None:
        assert run_cli("node", "stop", "--task", "T1", "--token", world["node"])[1]["error"] == "CONFLICT"
        assert run_cli("node", "take-over", "--task", "T1", "--token", world["node"])[1]["error"] == "CONFLICT"
