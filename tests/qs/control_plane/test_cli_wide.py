"""Checkpoint 11: tests parametrized over the complete CLI table (AC3, AC11).

Every effectful command and every tool is exercised with a stale token, with
a node token used out of bounds, and against a DB newer than the code. A
command added to the table without a sample here fails
``test_every_command_has_a_sample``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from control_plane import cli, migrations, tools

from .conftest import NEXT, ORCH, agent, insert_node, insert_task, open_run, run_cli, sql

EXEMPT = {"version", "daemon", "ensure", "hook stop", "hook pre-tool-use", "hook pre-push", "hooks-settings"}
READ = {"session status", "snapshot", "task show", "export-summary", "ledger show"}
NO_TOKEN_WRITES = {"run open", "run claim"}
TOOL_ARGS: dict[str, dict[str, Any]] = {
    "worktree-create": {"phase": "/create-plan"},
    "worktree-cleanup": {},
    "gate": {"mode": "impacted"},
    "spawn": {"agent": "a", "permission_mode": "auto", "prompt_file": "@prompt"},
    "resume": {"message_file": "@prompt"},
    "issue-create": {"title": "t", "body_file": "@prompt"},
    "pr-create": {"title": "t", "summary_file": "@prompt"},
    "push": {},
    "merge": {},
    "item-create": {},
    "item-cleanup": {},
    "integrate-start": {},
    "integrate-finish": {},
    "integrate-drop": {},
}


def samples(f: dict[str, str]) -> dict[str, list[str]]:
    """argv (without ``--token``) for every token-taking command."""
    out = {
        "run bind-name": ["run", "bind-name", "R1"],
        "run set-mode": ["run", "set-mode", "--permission-mode", "auto"],
        "run set-plan": ["run", "set-plan", "--run", "R1", "--file", f["text"]],
        "run close": ["run", "close"],
        "task add": ["task", "add", "--run", "R1", "--title", "x", "--kind", "bug"],
        "task set": ["task", "set", "--task", "T1", "--branch", "b"],
        "task state": ["task", "state", "--task", "T1", "--to", "blocked"],
        "task dep": ["task", "dep", "add", "--task", "T1", "--on", "T2"],
        "task root": ["task", "root", "add", "--run", "R1", "--task", "T1"],
        "task work-list": ["task", "work-list", "add", "--run", "R1", "--task", "T1"],
        "criteria set": ["criteria", "set", "--task", "T1", "--file", f["text"]],
        "criteria validate": ["criteria", "validate", "--task", "T1"],
        "criteria state": ["criteria", "state", "--task", "T1", "--idx", "1", "--to", "met"],
        "question open": ["question", "open", "--task", "T1", "--text-file", f["text"]],
        "question ask": ["question", "ask", "Q-1"],
        "question answer": ["question", "answer", "Q-1", "--answer-file", f["text"], "--reason", "r"],
        "decision add": ["decision", "add", "--text", "t", "--reason", "r", "--source", "s"],
        "report post": [
            "report",
            "post",
            "--task",
            "T1",
            "--phase",
            "p",
            "--round",
            "1",
            "--status",
            "blocked",
            "--summary",
            "s",
            "--fields-file",
            f["json"],
        ],
        "digest put": ["digest", "put", "--task", "T1", "--file", f["text"]],
        "msg post": ["msg", "post", "--run", "R1", "--to", "orchestrator", "--kind", "k", "--payload-file", f["json"]],
        "msg pop": ["msg", "pop", "--run", "R1", "--as", "orchestrator"],
        "msg ack": ["msg", "ack", "1", "--receipt", "r"],
        "wait": ["wait", "--run", "R1"],
        "node stop": ["node", "stop", "--task", "T1"],
        "node take-over": ["node", "take-over", "--task", "T1"],
        "node hand-back": ["node", "hand-back", "--task", "T1", "--summary-file", f["text"]],
        "lock acquire": ["lock", "acquire", "--name", "integration:QS_11", "--purpose", "p", "--session-id", ORCH],
        "lock release": ["lock", "release", "--name", "integration:QS_11", "--session-id", ORCH],
        "round start": ["round", "start", "--task", "T1", "--phase", "build", "--head", "h1"],
        "finding open": [
            "finding",
            "open",
            "--task",
            "T1",
            "--phase",
            "build",
            "--source",
            "reviewer",
            "--input",
            f["finding"],
        ],
        "finding classify": ["finding", "classify", "1", "--class", "nice_to_have"],
        "finding state": ["finding", "state", "1", "--to", "rejected", "--reason", "r"],
        "blast-radius set": [
            "blast-radius",
            "set",
            "--task",
            "T1",
            "--value",
            "ok",
            "--head-sha",
            "h",
            "--review",
            "R",
        ],
    }
    for name in tools.REGISTRY:
        out[f"tool {name}"] = ["tool", name, "--task", "T1", "--key", "k", "--args-file", f[f"tool:{name}"]]
    return out


# Commands a node token may call (on its own task / queue).
NODE_OK = {
    "task state",
    "report post",
    "digest put",
    "question open",
    "node hand-back",
    "msg post",
    "msg pop",
    "msg ack",
    "lock acquire",
    "lock release",
    "tool gate",
    "tool push",
    "tool pr-create",
    "tool integrate-start",
    "tool integrate-finish",
    "tool integrate-drop",
    "round start",
    "finding open",
    "finding classify",
    "finding state",
}
# The node-token variant of a sample (own queue, own session), and an out-of-bounds variant (another task).
NODE_OWN = {
    "msg pop": ["msg", "pop", "--run", "R1", "--as", "node:T1"],
    "lock acquire": ["lock", "acquire", "--name", "integration:QS_11", "--purpose", "p", "--session-id", "S-n1"],
    "lock release": ["lock", "release", "--name", "integration:QS_11", "--session-id", "S-n1"],
}


def _other_task(argv: list[str]) -> list[str]:
    swapped = ["T2" if a == "T1" else a for a in argv]
    if swapped[:2] == ["msg", "post"]:
        swapped[swapped.index("--to") + 1] = "node:T2"
    if swapped[:2] == ["msg", "pop"]:
        swapped[swapped.index("--as") + 1] = "node:T2"
    if swapped[:2] == ["lock", "acquire"]:
        swapped[swapped.index("--name") + 1] = "integration:QS_99"
    if swapped[:2] in (["finding", "classify"], ["finding", "state"]):
        swapped[2] = "2"  # finding 2 is on T2
    return swapped


def counts(path: Path) -> dict[str, int]:
    tables = sql(path, "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
    return {t: sql(path, f"SELECT count(*) FROM {t}")[0][0] for (t,) in tables}


@pytest.fixture
def world(migrated: Path, tmp_path: Path, fake_claude, fake_runner, fake_popen) -> dict[str, Any]:
    run_id, token = open_run()
    wt = tmp_path / "wt1"
    wt.mkdir()
    insert_task(
        migrated, "T1", run_id, issue_number=11, worktree=str(wt), branch="QS_11", is_deliverable=1, pr_number=5
    )
    insert_task(migrated, "T2", run_id, worktree=str(wt), branch="QS_12", is_deliverable=1)
    node = insert_node(migrated, "N1", run_id, "T1", session_id="S-n1", name="n1")
    for fid, task in ((1, "T1"), (2, "T2")):  # #375: one finding per task, seeded below the decision counter
        sql(
            migrated,
            "INSERT INTO findings (id, run_id, task_id, phase, round, source, severity, category, title, body,"
            " fingerprint, title_norm, replay_key, state, decided_seq, actor, created_at, updated_at)"
            " VALUES (?, ?, ?, 'build', 0, 'reviewer', 'should_fix', 'test', 't', 'b', ?, 't', ?, 'open', ?, 'x', 'x', 'x')",
            [fid, run_id, task, f"fp{fid}", f"rk{fid}", fid],
        )
    sql(migrated, "INSERT INTO counters (kind, next) VALUES ('finding_decision', 3)")
    fake_claude.listing = [agent(ORCH, pid=101), agent("S-n1", "n1", pid=201)]
    files = {
        "text": tmp_path / "t.md",
        "json": tmp_path / "p.json",
        "prompt": tmp_path / "prompt.md",
        "finding": tmp_path / "finding.json",
    }
    files["finding"].write_text(json.dumps({"severity": "should_fix", "category": "test", "title": "t", "body": "b"}))
    files["text"].write_text("one\n")
    files["json"].write_text("{}")
    files["prompt"].write_text("go")
    f = {k: str(v) for k, v in files.items()}
    for name, args in TOOL_ARGS.items():
        p = tmp_path / f"tool-{name}.json"
        p.write_text(json.dumps({k: (f["prompt"] if v == "@prompt" else v) for k, v in args.items()}))
        f[f"tool:{name}"] = str(p)
    fake_runner.calls.clear()
    fake_popen.calls.clear()  # the fixture's own `run open` started the (fake) daemon
    return {"run": run_id, "token": token, "node": node, "db": migrated, "f": f}


def test_every_command_has_a_kind_and_a_sample(world) -> None:
    table = cli.all_commands()
    kinds = {name: cmd.kind for name, cmd in table.items()}
    assert {n for n, k in kinds.items() if k == "exempt"} == EXEMPT
    assert {n for n, k in kinds.items() if k == "read"} == READ
    writes = {n for n, k in kinds.items() if k == "write"}
    assert writes - NO_TOKEN_WRITES == set(samples(world["f"]))
    assert set(TOOL_ARGS) == set(tools.REGISTRY)


def _stale(world: dict[str, Any], fake_runner, fake_popen) -> str:
    assert run_cli("run", "claim", world["run"], "--session-id", ORCH)[0] == 0  # the old token is now stale
    fake_runner.calls.clear()
    fake_popen.calls.clear()
    return str(world["token"])


@pytest.mark.parametrize(
    "name", sorted(samples(dict.fromkeys(["text", "json", "finding"] + [f"tool:{t}" for t in TOOL_ARGS], "x")))
)
def test_stale_run_token_changes_nothing(world, name: str, fake_runner, fake_popen) -> None:
    stale = _stale(world, fake_runner, fake_popen)
    before = counts(world["db"])
    code, out = run_cli(*samples(world["f"])[name], "--token", stale)
    assert code == 3, out
    assert out["error"] == "STALE_TOKEN" and "Stop your event loop" in out["instructions"]
    assert counts(world["db"]) == before
    assert fake_runner.calls == [] and fake_popen.calls == []


@pytest.mark.parametrize("name", sorted(NODE_OK))
def test_stale_node_generation_changes_nothing(world, name: str, fake_runner, fake_popen) -> None:
    kind, rest = world["node"].split(":", 1)
    subject, gen, nonce = rest.split(".")
    stale = f"{kind}:{subject}.{int(gen) + 1}.{nonce}"
    fake_runner.calls.clear()
    before = counts(world["db"])
    argv = NODE_OWN.get(name, samples(world["f"])[name])
    code, out = run_cli(*argv, "--token", stale)
    assert code == 3, out
    assert counts(world["db"]) == before and fake_runner.calls == [] and fake_popen.calls == []


@pytest.mark.parametrize(
    "name", sorted(set(samples(dict.fromkeys(["text", "json", "finding"] + [f"tool:{t}" for t in TOOL_ARGS], "x"))))
)
def test_node_token_out_of_bounds_conflicts(world, name: str, fake_runner) -> None:
    if name == "msg ack":  # a message of another node's queue
        post = ["msg", "post", "--run", "R1", "--to", "node:T2", "--kind", "k", "--payload-file", world["f"]["json"]]
        assert run_cli(*post, "--token", world["token"])[1]["id"] == 1
    if name == "lock release":  # a lock held by someone else
        sql(
            world["db"],
            "INSERT INTO locks (name, holder_kind, holder_session_id, holder_actor, token_subject, acquired_at)"
            " VALUES ('integration:QS_11', 'session', ?, 'orchestrator', 'run:R1', 'x')",
            [ORCH],
        )
    fake_runner.calls.clear()
    before = counts(world["db"])
    if name in NODE_OK:
        argv = _other_task(NODE_OWN.get(name, samples(world["f"])[name]))
    else:
        argv = samples(world["f"])[name]
    code, out = run_cli(*argv, "--token", world["node"])
    assert code == 8 and out["error"] == "CONFLICT", (argv, out)
    assert counts(world["db"]) == before and fake_runner.calls == []


def _all_non_exempt(f: dict[str, str]) -> dict[str, list[str]]:
    table = dict(samples(f))
    for name, argv in list(table.items()):
        table[name] = [*argv, "--token", "run:R1.1." + "0" * 32]
    table["run open"] = ["run", "open", "--name", "r9", "--title", "t", "--session-id", "S"]
    table["run claim"] = ["run", "claim", "R1", "--session-id", "S"]
    table["session status"] = ["session", "status", "--session-id", "S"]
    table["snapshot"] = ["snapshot"]
    table["task show"] = ["task", "show", "--task", "T1"]
    table["export-summary"] = ["export-summary", "--task", "T1", "--out-worktree", "/tmp"]
    table["ledger show"] = ["ledger", "show", "--task", "T1"]
    return table


ALL_NON_EXEMPT = sorted(
    _all_non_exempt(dict.fromkeys(["text", "json", "finding"] + [f"tool:{t}" for t in TOOL_ARGS], "x"))
)


@pytest.mark.parametrize("name", ALL_NON_EXEMPT)
def test_db_newer_than_the_code_refuses_every_non_exempt_command(
    world, name: str, fake_runner, fake_popen, fake_clock
) -> None:
    sql(world["db"], f"PRAGMA user_version = {NEXT}")
    fake_runner.calls.clear()
    fake_popen.calls.clear()
    before = counts(world["db"])
    code, out = run_cli(*_all_non_exempt(world["f"])[name])
    assert code == 5 and out["error"] == "SCHEMA_TOO_NEW", out
    assert counts(world["db"]) == before
    assert fake_runner.calls == [] and fake_popen.calls == [] and fake_clock.sleeps == []


def test_exempt_commands_under_a_newer_db(world, fake_popen, fake_clock) -> None:
    sql(world["db"], f"PRAGMA user_version = {NEXT}")
    assert run_cli("version")[0] == 0
    assert run_cli("hooks-settings", "--role", "node")[0] == 0
    assert run_cli("hook", "stop", stdin=json.dumps({"session_id": ORCH})) == (0, None)
    pre = {"session_id": ORCH, "tool_name": "Bash", "tool_input": {"command": "gh pr merge 1"}}
    assert run_cli("hook", "pre-tool-use", stdin=json.dumps(pre)) == (0, None)
    assert run_cli("hook", "pre-push", stdin="refs/heads/QS_1_2 a refs/heads/QS_1_2 b\n")[0] == 1  # still refused
    assert fake_popen.calls == [] and fake_clock.sleeps == []


@pytest.mark.parametrize("name", sorted(READ))
def test_read_commands_never_wait(world, name: str, fake_popen, fake_clock) -> None:
    argv = _all_non_exempt(world["f"])[name]
    sql(world["db"], "PRAGMA user_version = 0")
    code, out = run_cli(*argv)
    assert code == 5 and out["error"] == "SCHEMA_PENDING"
    assert fake_popen.calls == [] and fake_clock.sleeps == []


def test_an_older_db_with_a_migrating_daemon_proceeds(world, fake_popen, monkeypatch) -> None:
    v2 = migrations.Migration(NEXT, "test v2", ("CREATE TABLE extra (a INTEGER)",))
    monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, v2))
    fake_popen.on_call = lambda argv, kw: migrations.migrate(world["db"], role="test")
    code, out = run_cli("decision", "add", "--text", "t", "--reason", "r", "--source", "s", "--token", world["token"])
    assert code == 0, out
    assert len(fake_popen.calls) == 1


def test_the_seeded_findings_never_tie_with_a_new_decision(world) -> None:
    """#375 AC3: a hand-seeded row's ``decided_seq`` stays below the counter, so a new one is strictly newer."""
    code, out = run_cli(*samples(world["f"])["finding open"], "--token", world["token"])
    assert code == 0 and out == {"ok": True, "ids": [3]}
    seqs = dict(sql(world["db"], "SELECT id, decided_seq FROM findings"))
    assert seqs == {1: 1, 2: 2, 3: 3}
