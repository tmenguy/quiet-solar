"""Checkpoint 5: token format and ``tokens.require`` (§6.1, §6.3)."""

from __future__ import annotations

import pytest
from control_plane import db, errors, tokens

from .conftest import ORCH, insert_node, insert_task, open_run, run_cli, sql


def test_mint_and_parse() -> None:
    nonce = tokens.new_nonce()
    assert len(nonce) == 32
    tok = tokens.parse(tokens.mint("run", "R7", 3, nonce))
    assert tok == tokens.Token("run", "R7", 3, nonce)
    assert tok.subject_ref == "run:R7"


@pytest.mark.parametrize(
    "bad", [None, "", "run:R7.3", "x:R7.1." + "a" * 32, "run:R7.1." + "g" * 32, "node:N1.x." + "a" * 32]
)
def test_malformed_is_usage(bad: str | None) -> None:
    with pytest.raises(errors.CpError) as exc:
        tokens.parse(bad)
    assert exc.value.code == "USAGE"


def _require(conn, token, **kw):
    with db.write(conn):
        return tokens.require(conn, token, **kw)


class TestRunTokens:
    def test_valid(self, migrated, conn) -> None:
        run_id, token = open_run()
        who = _require(conn, token, kinds={"run"})
        assert (who.run_id, who.session_id, who.kind, who.actor, who.subject) == (
            run_id,
            ORCH,
            "run",
            "orchestrator",
            f"run:{run_id}",
        )

    def test_stale_after_takeover_names_the_successor(self, migrated, conn, fake_claude) -> None:
        from .conftest import agent

        run_id, token = open_run()
        fake_claude.listing = [agent(ORCH)]
        code, out = run_cli("run", "claim", run_id, "--session-id", "S-new", "--takeover")
        assert code == 0, out
        with pytest.raises(errors.CpError) as exc:
            _require(conn, token, kinds={"run"})
        assert exc.value.code == "STALE_TOKEN"
        assert exc.value.extra["superseded_by"] == "S-new"
        assert exc.value.extra["instructions"] == tokens.INSTRUCTIONS

    def test_wrong_nonce_without_supersede(self, migrated, conn) -> None:
        run_id, token = open_run()
        bad = token[:-1] + ("0" if token[-1] != "0" else "1")
        with pytest.raises(errors.CpError) as exc:
            _require(conn, bad, kinds={"run"})
        assert exc.value.code == "STALE_TOKEN" and exc.value.extra["superseded_by"] == ORCH

    def test_unknown_run_and_wrong_kind(self, migrated, conn) -> None:
        with pytest.raises(errors.CpError) as exc:
            _require(conn, tokens.mint("run", "R9", 1, "a" * 32), kinds={"run"})
        assert exc.value.code == "NOT_FOUND"
        _, token = open_run()
        with pytest.raises(errors.CpError) as exc:
            _require(conn, token, kinds={"node"})
        assert exc.value.code == "CONFLICT"

    def test_task_scope(self, migrated, conn) -> None:
        run_id, token = open_run()
        other, _ = open_run("r2", "S-2")
        insert_task(migrated, "T1", run_id)
        insert_task(migrated, "T2", other)
        insert_task(migrated, "T3", None)
        assert _require(conn, token, kinds={"run"}, task_id="T1").run_id == run_id
        assert _require(conn, token, kinds={"run"}, task_id="T3").run_id == run_id
        for task, code in (("T2", "CONFLICT"), ("T9", "NOT_FOUND")):
            with pytest.raises(errors.CpError) as exc:
                _require(conn, token, kinds={"run"}, task_id=task)
            assert exc.value.code == code


class TestNodeTokens:
    def _setup(self, migrated):
        run_id, _ = open_run()
        insert_task(migrated, "T1", run_id)
        insert_task(migrated, "T2", run_id)
        return run_id, insert_node(migrated, "N1", run_id, "T1", session_id="S-n1")

    def test_valid_and_own_task(self, migrated, conn) -> None:
        run_id, tok = self._setup(migrated)
        who = _require(conn, tok, kinds={"run", "node"}, task_id="T1")
        assert (who.node_id, who.task_id, who.actor, who.subject, who.session_id) == (
            "N1",
            "T1",
            "node:T1",
            "node:N1",
            "S-n1",
        )
        with pytest.raises(errors.CpError) as exc:
            _require(conn, tok, kinds={"run", "node"}, task_id="T2")
        assert exc.value.code == "CONFLICT"

    def test_superseded_generation(self, migrated, conn) -> None:
        run_id, tok = self._setup(migrated)
        sql(migrated, "UPDATE nodes SET state = 'superseded' WHERE id = 'N1'")
        insert_node(migrated, "N2", run_id, "T1", generation=2, spawned_at="t2")
        with pytest.raises(errors.CpError) as exc:
            _require(conn, tok, kinds={"node"})
        assert exc.value.code == "STALE_TOKEN" and exc.value.extra["superseded_by"] == "N2"

    def test_bumped_generation_and_bad_nonce(self, migrated, conn) -> None:
        _, tok = self._setup(migrated)
        kind, rest = tok.split(":", 1)
        subject, gen, nonce = rest.split(".")
        for bad in (f"{kind}:{subject}.{int(gen) + 1}.{nonce}", f"{kind}:{subject}.{gen}.{'f' * 32}"):
            with pytest.raises(errors.CpError) as exc:
                _require(conn, bad, kinds={"node"})
            assert exc.value.code == "STALE_TOKEN" and exc.value.extra["superseded_by"] == "N1"

    def test_stopped(self, migrated, conn) -> None:
        _, tok = self._setup(migrated)
        sql(migrated, "UPDATE nodes SET state = 'stopped' WHERE id = 'N1'")
        with pytest.raises(errors.CpError) as exc:
            _require(conn, tok, kinds={"node"})
        assert exc.value.code == "STOPPED" and exc.value.exit_code == 4
        assert _require(conn, tok, kinds={"node"}, allow_stopped=True).node_state == "stopped"

    def test_unknown_node(self, migrated, conn) -> None:
        with pytest.raises(errors.CpError) as exc:
            _require(conn, tokens.mint("node", "N9", 1, "a" * 32), kinds={"node"})
        assert exc.value.code == "NOT_FOUND"
