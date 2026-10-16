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


# --------------------------------------------------------------------------- #407: review tokens (D2)

SESSION, GRANT = "1" * 32, "2" * 32


def _review_row(path, rid: str = "V1", *, state: str = "running", nonce=SESSION, grant=GRANT, home="R1") -> None:
    sql(
        path,
        "INSERT INTO reviews (id, kind, coalesce_key, home_run_id, state, agent, nonce, grant_nonce, name,"
        " requested_by, session_id, created_at, updated_at) VALUES (?, 'global', ?, ?, ?, 'a', ?, ?, ?, 'x', 'S-v',"
        " 'x', 'x')",
        (rid, f"k-{rid}", home, state, nonce, grant, f"n-{rid}"),
    )


@pytest.fixture
def review_world(migrated):
    run_id, _ = open_run()
    other, _ = open_run("r2", "S-2")
    insert_task(migrated, "T1", run_id)
    insert_task(migrated, "T2", other)
    insert_task(migrated, "T3", run_id)
    _review_row(migrated)
    for task, run in (("T1", run_id), ("T2", other)):
        sql(migrated, "INSERT INTO review_tasks (review_id, task_id, run_id) VALUES ('V1', ?, ?)", (task, run))
    return migrated


class TestReviewTokens:
    def test_parse_and_effective_kinds(self) -> None:
        session = tokens.parse(f"review:V12.1.{SESSION}")
        grant = tokens.parse(f"review:V12.0.{GRANT}")
        assert (session.kind, session.subject, session.epoch, session.effective_kind) == ("review", "V12", 1, "review")
        assert grant.effective_kind == "review_grant" and grant.subject_ref == "review:V12"
        assert tokens.parse(tokens.mint("run", "R1", 1, SESSION)).effective_kind == "run"

    @pytest.mark.parametrize(
        "bad",
        [
            f"review:V01.1.{SESSION}",  # a leading zero
            f"review:V+1.1.{SESSION}",
            f"review:V0.1.{SESSION}",
            f"review:V1.2.{SESSION}",  # only the forms 0 and 1
            "review:V1.1.",  # an empty nonce
            f"review:V1.1.{SESSION}\n",  # a trailing newline (fullmatch)
            f"review:R1.1.{SESSION}",
            f"review:V1.1.{'A' * 32}",  # upper-case hex
        ],
    )
    def test_grammar(self, bad: str) -> None:
        with pytest.raises(errors.CpError) as exc:
            tokens.parse(bad)
        assert exc.value.code == "USAGE"

    def test_session_token(self, review_world, conn) -> None:
        who = _require(conn, f"review:V1.1.{SESSION}", kinds={"review"})
        assert (who.kind, who.review_id, who.review_kind, who.run_id, who.actor, who.session_id) == (
            "review",
            "V1",
            "global",
            "R1",
            "review:V1",
            "S-v",
        )
        assert who.scope == frozenset({"T1", "T2"}) and who.subject == "review:V1"
        assert _require(conn, f"review:V1.1.{SESSION}", kinds={"review"}, task_id="T2").review_id == "V1"  # any run
        for task in ("T3", "T9"):  # out of scope, unknown: CONFLICT before any run check
            with pytest.raises(errors.CpError) as exc:
                _require(conn, f"review:V1.1.{SESSION}", kinds={"review"}, task_id=task)
            assert exc.value.code == "CONFLICT"

    def test_a_cross_run_review_has_no_run(self, review_world, conn) -> None:
        sql(review_world, "UPDATE reviews SET home_run_id = NULL, kind = 'cross_run'")
        who = _require(conn, f"review:V1.1.{SESSION}", kinds={"review"})
        assert who.run_id == "" and who.review_kind == "cross_run"

    @pytest.mark.parametrize(
        ("state", "session_ok", "grant_ok"),
        [
            ("requested", False, True),
            ("launching", True, True),
            ("running", True, True),
            ("finished", False, True),
            ("failed", False, False),
            ("abandoned", False, False),
        ],
    )
    def test_valid_states_per_form(self, review_world, conn, state, session_ok, grant_ok) -> None:
        sql(review_world, "UPDATE reviews SET state = ?", (state,))
        for token, kinds, ok in (
            (f"review:V1.1.{SESSION}", {"review"}, session_ok),
            (f"review:V1.0.{GRANT}", {"review_grant"}, grant_ok),
        ):
            if ok:
                assert _require(conn, token, kinds=kinds).review_id == "V1"
                continue
            with pytest.raises(errors.CpError) as exc:
                _require(conn, token, kinds=kinds)
            assert exc.value.code == "STALE_TOKEN" and exc.value.exit_code == 3
            assert exc.value.extra == {"review_id": "V1", "state": state, "instructions": tokens.REVIEW_INSTRUCTIONS}

    def test_wrong_nonce_and_swapped_forms(self, review_world, conn) -> None:
        for token, kinds in (
            (f"review:V1.1.{'f' * 32}", {"review"}),
            (f"review:V1.0.{'f' * 32}", {"review_grant"}),
            (f"review:V1.1.{GRANT}", {"review"}),  # the grant's nonce in the session form
            (f"review:V1.0.{SESSION}", {"review_grant"}),
        ):
            with pytest.raises(errors.CpError) as exc:
                _require(conn, token, kinds=kinds)
            assert exc.value.code == "STALE_TOKEN"

    @pytest.mark.parametrize("stored", [None, ""])
    def test_a_missing_stored_nonce_is_stale(self, review_world, conn, stored) -> None:
        sql(review_world, "UPDATE reviews SET nonce = ?, grant_nonce = ?", (stored, stored))
        for token, kinds in ((f"review:V1.1.{SESSION}", {"review"}), (f"review:V1.0.{GRANT}", {"review_grant"})):
            with pytest.raises(errors.CpError) as exc:
                _require(conn, token, kinds=kinds)
            assert exc.value.code == "STALE_TOKEN"

    def test_kinds_are_effective_kinds(self, review_world, conn) -> None:
        for token, kinds in (
            (f"review:V1.1.{SESSION}", {"review_grant", "run"}),
            (f"review:V1.0.{GRANT}", {"review", "run"}),
            (f"review:V1.1.{SESSION}", {"run", "node"}),
        ):
            with pytest.raises(errors.CpError) as exc:
                _require(conn, token, kinds=kinds)
            assert exc.value.code == "CONFLICT"
        with pytest.raises(errors.CpError) as exc:  # the kind is checked before the row: no review needed
            _require(conn, f"review:V9.1.{SESSION}", kinds={"run"})
        assert exc.value.code == "CONFLICT"
        with pytest.raises(errors.CpError) as exc:
            _require(conn, f"review:V9.1.{SESSION}", kinds={"review"})
        assert exc.value.code == "NOT_FOUND"

    def test_run_and_node_principals_keep_empty_review_fields(self, review_world, conn) -> None:
        _, token = open_run("r3", "S-3")
        who = _require(conn, token, kinds={"run"})
        assert (who.review_id, who.review_kind, who.scope) == (None, None, frozenset())
