"""Checkpoint 7: reports, digests, questions, decisions (§7.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from control_plane import errors, messages, reports

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
        fields = _file(world, "f.json", json.dumps({"blocking": [3]}))
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
        assert (rep["node_id"], rep["round"], rep["fields"]) == ("N1", 2, '{"blocking":[3]}')
        msg = dict(sql(world["db"], "SELECT * FROM messages WHERE id = ?", [out["message_id"]])[0])
        assert (msg["recipient"], msg["kind"], msg["sender"]) == ("orchestrator", "report", "node:T1")
        assert json.loads(msg["payload"])["report_id"] == out["report_id"]

    def test_report_validation(self, world) -> None:
        fields = _file(world, "f.json", "{}")
        base = ["report", "post", "--phase", "build", "--round", "1", "--summary", "s", "--fields-file", fields]
        assert run_cli(*base, "--task", "T1", "--status", "bad", "--token", world["node"])[1]["error"] == "USAGE"
        assert run_cli(*base, "--task", "T2", "--status", "blocked", "--token", world["node"])[1]["error"] == "CONFLICT"
        assert run_cli(*base, "--task", "T2", "--status", "blocked", "--token", world["token"])[0] == 0


def digest(**values: str) -> dict[str, str]:
    """A digest on the contract: every key, empty unless given."""
    return {**dict.fromkeys(reports.DIGEST_KEYS, ""), **values}


def _post(world: dict, fields: object, *, phase: str = "build", raw: str | None = None) -> tuple[int, dict]:
    path = _file(world, "fields.json", json.dumps(fields) if raw is None else raw)
    return run_cli(
        "report",
        "post",
        "--task",
        "T1",
        "--phase",
        phase,
        "--round",
        "1",
        "--status",
        "continuing",
        "--summary",
        "s",
        "--fields-file",
        path,
        "--token",
        world["node"],
    )


def _nothing_posted(world: dict) -> bool:
    reports_n = sql(world["db"], "SELECT count(*) FROM reports")[0][0]
    messages_n = sql(world["db"], "SELECT count(*) FROM messages WHERE kind = 'report'")[0][0]
    return bool(reports_n == 0 and messages_n == 0)


_DISCOVERED = {"title": "t", "why": "w"}


class TestReportContract:
    """#424 D4: the phase is exactly `plan` / `build`; `fields` is a closed schema."""

    @pytest.mark.parametrize("phase", ["p", "Build", "build ", " plan", "PLAN", ""])
    def test_a_phase_other_than_plan_or_build_is_refused(self, world, phase: str) -> None:
        code, out = _post(world, {}, phase=phase)
        assert code == 2 and out["error"] == "USAGE" and "--phase" in out["detail"], out
        assert _nothing_posted(world)

    @pytest.mark.parametrize(
        ("fields", "why"),
        [
            ([], "object"),
            ("blocking", "object"),
            (None, "object"),
            ({"tests": 3}, "tests"),
            ({"blocking": 3}, "blocking"),
            ({"blocking": [True]}, "blocking"),
            ({"blocking": [0]}, "blocking"),
            ({"blocking": [-1]}, "blocking"),
            ({"blocking": [1.0]}, "blocking"),
            ({"blocking": ["1"]}, "blocking"),
            ({"blocking": [2**63]}, "blocking"),  # beyond a SQLite INTEGER (fix #02 G7)
            ({"questions": "Q-1"}, "questions"),
            ({"questions": [1]}, "questions"),
            ({"questions": ["Q-1 "]}, "questions"),
            ({"questions": ["Q-0"]}, "questions"),
            ({"questions": ["q-1"]}, "questions"),
            ({"questions": ["Q-01"]}, "questions"),
            ({"discovered": {}}, "discovered"),
            ({"discovered": ["t"]}, "discovered"),
            ({"discovered": [{**_DISCOVERED, "extra": 1}]}, "extra"),
            ({"discovered": [{"why": "w"}]}, "title"),
            ({"discovered": [{"title": "t"}]}, "why"),
            ({"discovered": [{"title": " ", "why": "w"}]}, "title"),
            ({"discovered": [{"title": "t", "why": ""}]}, "why"),
            ({"discovered": [{"title": 1, "why": "w"}]}, "title"),
            ({"discovered": [{**_DISCOVERED, "shrinks": 1}]}, "shrinks"),
            ({"discovered": [{**_DISCOVERED, "shrinks": 0}]}, "shrinks"),
            ({"discovered": [{**_DISCOVERED, "shrinks": None}]}, "shrinks"),
            ({"discovered": [_DISCOVERED, {"title": "t"}]}, "discovered[1]"),
        ],
    )
    def test_fields_off_the_schema_are_refused(self, world, fields: object, why: str) -> None:
        code, out = _post(world, fields)
        assert code == 2 and out["error"] == "USAGE" and why in out["detail"], out
        assert _nothing_posted(world)

    def test_fields_that_are_not_json_are_refused(self, world) -> None:
        code, out = _post(world, None, raw="{blocking")
        assert code == 2 and out["error"] == "USAGE" and _nothing_posted(world)

    def test_the_first_violation_is_named(self) -> None:
        with pytest.raises(errors.CpError) as exc:  # an unknown key is checked before the known keys' values
            reports.validate_fields({"blocking": [0], "zz": 1})
        assert exc.value.code == "USAGE" and "zz" in exc.value.detail
        with pytest.raises(errors.CpError) as exc:  # then the keys in `FIELDS` order
            reports.validate_fields({"discovered": [1], "blocking": [0]})
        assert "blocking" in exc.value.detail

    def test_an_over_size_fields_is_a_conflict(self, world) -> None:
        fields = {"discovered": [{"title": "t", "why": "é" * (reports.FIELDS_MAX_BYTES // 2)}]}
        text = messages.canonical(fields)
        assert len(text) < reports.FIELDS_MAX_BYTES < len(text.encode())  # over only by its UTF-8 size
        code, out = _post(world, fields)
        assert out["error"] == "CONFLICT", out
        assert _nothing_posted(world)

    @pytest.mark.parametrize(
        "fields",
        [
            {},
            {"blocking": []},
            {"blocking": [1, 7, 7, 2**63 - 1]},
            {"questions": ["Q-1", "Q-12"]},
            {
                "discovered": [
                    _DISCOVERED,
                    {"title": "t2", "why": "w2", "shrinks": True},
                    {**_DISCOVERED, "shrinks": False},
                ]
            },
            {"discovered": [], "questions": [], "blocking": [2]},
        ],
    )
    def test_valid_fields_are_stored_canonically(self, world, fields: dict) -> None:
        code, out = _post(world, fields, phase="plan")
        assert code == 0, out
        assert tuple(sql(world["db"], "SELECT phase, fields FROM reports")[0]) == ("plan", messages.canonical(fields))

    def test_a_lone_surrogate_is_usage(self, world) -> None:
        code, out = _post(world, {"discovered": [{"title": "\ud800", "why": "w"}]})  # #424 fix #01 F2
        assert code == 2 and out["error"] == "USAGE" and "UTF-8" in out["detail"], out
        assert _nothing_posted(world)

    def test_a_lone_surrogate_in_the_summary_is_usage(self, world, monkeypatch) -> None:
        # #424 fix #02 G3: a non-UTF-8 argv byte arrives as a lone surrogate (`surrogateescape`)
        path = _file(world, "fields.json", "{}")
        base = ["report", "post", "--task", "T1", "--phase", "build", "--round", "1", "--status", "blocked"]
        code, out = run_cli(*base, "--summary", "\udcff", "--fields-file", path, "--token", world["node"])
        assert code == 2 and out["error"] == "USAGE" and "--summary" in out["detail"], out
        assert _nothing_posted(world)

    def test_strings_are_stored_as_given(self, world) -> None:
        fields = {"discovered": [{"title": " padded ", "why": "w "}]}
        assert _post(world, fields)[0] == 0
        assert json.loads(sql(world["db"], "SELECT fields FROM reports")[0][0]) == fields


class TestDigests:
    def test_replace_and_cap(self, world) -> None:
        first, second = digest(goal="first"), digest(goal="second", done="ça marche")
        a = _file(world, "a.json", json.dumps(first))
        b = _file(world, "b.json", json.dumps(dict(reversed(list(second.items())))))  # any key order
        out = run_cli("digest", "put", "--task", "T1", "--file", a, "--token", world["node"])[1]
        assert (out["task_id"], out["bytes"]) == ("T1", len(messages.canonical(first).encode()))
        out = run_cli("digest", "put", "--task", "T1", "--file", b, "--token", world["token"])[1]
        assert out["bytes"] == len(messages.canonical(second).encode())  # UTF-8 bytes: `ç` costs two
        assert sql(world["db"], "SELECT body FROM digests")[0][0] == messages.canonical(second)
        big = _file(world, "big.json", json.dumps(digest(done="x" * (16 * 1024))))
        code, out = run_cli("digest", "put", "--task", "T1", "--file", big, "--token", world["token"])
        assert out["error"] == "CONFLICT" and "16384" in out["detail"]
        assert sql(world["db"], "SELECT body FROM digests")[0][0] == messages.canonical(second)

    @pytest.mark.parametrize(
        ("text", "why"),
        [
            ("# Goal\n\nnot JSON", "not JSON"),
            ("[]", "JSON object"),
            ('"goal"', "JSON object"),
            (json.dumps({k: "" for k in reports.DIGEST_KEYS if k != "why"}), "missing"),
            (json.dumps({**dict.fromkeys(reports.DIGEST_KEYS, ""), "extra": ""}), "unknown"),
            (json.dumps({**dict.fromkeys(reports.DIGEST_KEYS, ""), "plan": ["a"]}), "plan"),
            (json.dumps({**dict.fromkeys(reports.DIGEST_KEYS, ""), "done": None}), "done"),
        ],
    )
    def test_a_digest_off_the_contract_is_refused_and_writes_nothing(self, world, text: str, why: str) -> None:
        ok = _file(world, "ok.json", json.dumps(digest(goal="kept")))
        assert run_cli("digest", "put", "--task", "T1", "--file", ok, "--token", world["node"])[0] == 0
        bad = _file(world, "bad.json", text)
        code, out = run_cli("digest", "put", "--task", "T1", "--file", bad, "--token", world["node"])
        assert code == 2 and out["error"] == "USAGE" and why in out["detail"], out
        assert sql(world["db"], "SELECT body FROM digests")[0][0] == messages.canonical(digest(goal="kept"))

    def test_parse_digest(self) -> None:
        d = digest(decisions="## Decisions\n- x")
        assert reports.parse_digest(messages.canonical(d)) == d
        assert reports.parse_digest("plain text") is None
        assert reports.parse_digest('{"goal": "x"}') is None
        assert reports.parse_digest("[1]") is None
        assert reports.parse_digest("[" * 100_000) is None  # too deep for `json.loads`: RecursionError

    def test_a_lone_surrogate_is_usage(self, world) -> None:
        # #424 fix #01 F2: `json.loads` accepts `\\ud800`, which no UTF-8 encoding holds
        bad = _file(world, "bad.json", json.dumps(digest(goal="\ud800")))
        code, out = run_cli("digest", "put", "--task", "T1", "--file", bad, "--token", world["node"])
        assert code == 2 and out["error"] == "USAGE" and "UTF-8" in out["detail"], out
        assert sql(world["db"], "SELECT count(*) FROM digests")[0][0] == 0


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
