"""Checkpoint 7: reports, digests, questions, decisions (§7.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import insert_node, insert_task, open_run, run_cli, sql


@pytest.fixture
def world(migrated: Path, tmp_path: Path) -> dict:
    run_id, token = open_run()
    insert_task(migrated, "T1", run_id)
    insert_task(migrated, "T2", run_id)
    node = insert_node(migrated, "N1", run_id, "T1")
    return {"run": run_id, "token": token, "node": node, "db": migrated, "tmp": tmp_path}


def _file(w: dict, name: str, text: str) -> str:
    p = w["tmp"] / name
    p.write_text(text)
    return str(p)


class TestReports:
    def test_report_row_and_message(self, world) -> None:
        fields = _file(world, "f.json", json.dumps({"tests": 3}))
        code, out = run_cli(
            "report",
            "post",
            "--task",
            "T1",
            "--phase",
            "build",
            "--round",
            "2",
            "--status",
            "converged",
            "--summary",
            "done",
            "--fields-file",
            fields,
            "--token",
            world["node"],
        )
        assert code == 0, out
        rep = dict(sql(world["db"], "SELECT * FROM reports")[0])
        assert (rep["node_id"], rep["round"], rep["fields"]) == ("N1", 2, '{"tests":3}')
        msg = dict(sql(world["db"], "SELECT * FROM messages WHERE id = ?", [out["message_id"]])[0])
        assert (msg["recipient"], msg["kind"], msg["sender"]) == ("orchestrator", "report", "node:T1")
        assert json.loads(msg["payload"])["report_id"] == out["report_id"]

    def test_report_validation(self, world) -> None:
        fields = _file(world, "f.json", "{}")
        base = ["report", "post", "--phase", "p", "--round", "1", "--summary", "s", "--fields-file", fields]
        assert run_cli(*base, "--task", "T1", "--status", "bad", "--token", world["node"])[1]["error"] == "USAGE"
        assert run_cli(*base, "--task", "T2", "--status", "blocked", "--token", world["node"])[1]["error"] == "CONFLICT"
        assert run_cli(*base, "--task", "T2", "--status", "blocked", "--token", world["token"])[0] == 0


class TestDigests:
    def test_replace_and_cap(self, world) -> None:
        a = _file(world, "a.md", "first")
        b = _file(world, "b.md", "second")
        assert run_cli("digest", "put", "--task", "T1", "--file", a, "--token", world["node"])[1]["bytes"] == 5
        assert run_cli("digest", "put", "--task", "T1", "--file", b, "--token", world["token"])[0] == 0
        assert sql(world["db"], "SELECT body FROM digests")[0][0] == "second"
        big = _file(world, "big.md", "x" * (16 * 1024 + 1))
        assert (
            run_cli("digest", "put", "--task", "T1", "--file", big, "--token", world["token"])[1]["error"] == "CONFLICT"
        )


class TestQuestions:
    def test_node_question_flow(self, world) -> None:
        text = _file(world, "q.md", "Which DB?")
        code, out = run_cli(
            "question", "open", "--task", "T1", "--text-file", text, "--blocking", "--token", world["node"]
        )
        assert code == 0 and out["question_id"] == "Q-1"
        msg = dict(sql(world["db"], "SELECT * FROM messages WHERE id = ?", [out["message_id"]])[0])
        assert (msg["kind"], json.loads(msg["payload"])["blocking"]) == ("question", True)
        assert run_cli("question", "ask", "Q-1", "--token", world["token"])[1]["state"] == "asked"
        assert run_cli("question", "ask", "Q-1", "--token", world["token"])[1]["error"] == "INVALID_STATE"
        ans = _file(world, "a.md", "SQLite")
        out = run_cli(
            "question", "answer", "Q-1", "--answer-file", ans, "--reason", "simple", "--token", world["token"]
        )[1]
        assert out["state"] == "answered"
        reply = dict(sql(world["db"], "SELECT * FROM messages WHERE id = ?", [out["message_id"]])[0])
        assert (reply["recipient"], reply["kind"]) == ("node:T1", "answer")
        row = dict(sql(world["db"], "SELECT * FROM questions")[0])
        assert (row["answer"], row["reason"], row["blocking"]) == ("SQLite", "simple", 1)
        assert (
            run_cli("question", "answer", "Q-1", "--answer-file", ans, "--reason", "r", "--token", world["token"])[1][
                "error"
            ]
            == "INVALID_STATE"
        )

    def test_orchestrator_question_gets_no_reply_message(self, world) -> None:
        text = _file(world, "q.md", "?")
        run_cli("question", "open", "--task", "T2", "--text-file", text, "--token", world["token"])
        ans = _file(world, "a.md", "!")
        out = run_cli("question", "answer", "Q-1", "--answer-file", ans, "--reason", "r", "--token", world["token"])[1]
        assert out["message_id"] is None

    def test_unknown_question_and_node_cannot_answer(self, world) -> None:
        assert run_cli("question", "ask", "Q-9", "--token", world["token"])[1]["error"] == "NOT_FOUND"
        ans = _file(world, "a.md", "!")
        assert (
            run_cli("question", "answer", "Q-9", "--answer-file", ans, "--reason", "r", "--token", world["node"])[1][
                "error"
            ]
            == "CONFLICT"
        )


class TestDecisions:
    def test_add(self, world) -> None:
        out = run_cli(
            "decision",
            "add",
            "--text",
            "use WAL",
            "--reason",
            "readers",
            "--source",
            "maintainer",
            "--task",
            "T1",
            "--token",
            world["token"],
        )[1]
        assert out["decision_id"] == 1
        run_cli("decision", "add", "--text", "x", "--reason", "y", "--source", "z", "--token", world["token"])
        rows = [tuple(r) for r in sql(world["db"], "SELECT task_id, text, source FROM decisions ORDER BY id")]
        assert rows == [("T1", "use WAL", "maintainer"), (None, "x", "z")]
        assert (
            run_cli("decision", "add", "--text", "x", "--reason", "y", "--source", "z", "--token", world["node"])[1][
                "error"
            ]
            == "CONFLICT"
        )
