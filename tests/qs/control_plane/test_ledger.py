"""The finding ledger (#375): rounds, findings, matching, flip-flops, convergence, blast radius."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from control_plane import clock, db, errors, ledger, reports

from .conftest import insert_node, insert_task, open_run, run_cli, sql

BAD = "run:R1.1." + "0" * 32  # R1's lease exists, the nonce does not match: a stale run token


def item(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "severity": "should_fix",
        "category": "correctness",
        "title": "Off by one",
        "body": "b",
        "file": "a.py",
        "symbol": "f",
    }
    base.update(kw)
    return {k: v for k, v in base.items() if v is not None}


def ci_item(sha: str = "c1", **kw: Any) -> dict[str, Any]:
    return {"category": "ci", "title": "test_x failed", "body": "tail", "sha": sha, **kw}


@dataclass
class L:
    conn: sqlite3.Connection
    clock: clock.FakeClock
    db: Path
    token: str  # R1's run token
    token2: str  # R2's run token
    n1: str  # node on T1
    n2: str  # node on T2

    def open(
        self,
        task: str,
        items: Any,
        *,
        source: str = "reviewer",
        phase: str = "build",
        round_: int | None = None,
        token: str | None = None,
    ) -> list[int]:
        out = ledger.open_findings(
            self.conn,
            self.clock,
            token=token or self.token,
            task_id=task,
            phase=phase,
            source=source,
            round_=round_,
            items=items,
        )
        return list(out["ids"])

    def one(self, task: str, **kw: Any) -> int:
        source = kw.pop("source", "reviewer")
        token = kw.pop("token", None)
        return self.open(task, item(**kw), source=source, token=token)[0]

    def state(
        self,
        fid: int,
        to: str,
        *,
        reason: str | None = None,
        commit: str | None = None,
        cause: int | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        return ledger.set_state(
            self.conn,
            self.clock,
            token=token or self.token,
            finding_id=fid,
            to=to,
            reason=reason,
            commit=commit,
            cause=cause,
        )

    def classify(self, fid: int, cls: str, note: str | None = None, token: str | None = None) -> dict[str, Any]:
        return ledger.classify(self.conn, self.clock, token=token or self.token, finding_id=fid, cls=cls, note=note)

    def round(
        self,
        task: str = "T1",
        phase: str = "build",
        head: str | None = "h1",
        base: str | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        return ledger.start_round(
            self.conn, self.clock, token=token or self.token, task_id=task, phase=phase, head=head, base=base
        )

    def report(
        self, status: str, *, task: str = "T1", phase: str = "build", round_: int = 1, token: str | None = None
    ) -> None:
        reports.post_report(
            self.conn,
            self.clock,
            token=token or self.token,
            task_id=task,
            phase=phase,
            round_=round_,
            status=status,
            summary="s",
            fields={},
        )

    def row(self, fid: int) -> dict[str, Any]:
        r = db.as_dict(self.conn.execute("SELECT * FROM findings WHERE id = ?", (fid,)).fetchone())
        assert r is not None
        r["flags"] = json.loads(r["flags"])
        return r

    def events(self, fid: int) -> list[dict[str, Any]]:
        return [
            dict(r) for r in self.conn.execute("SELECT * FROM finding_events WHERE finding_id = ? ORDER BY id", (fid,))
        ]

    def count(self) -> int:
        return int(self.conn.execute("SELECT count(*) FROM findings").fetchone()[0])

    def conv(self, task: str = "T1", phase: str = "build") -> dict[str, Any]:
        return ledger.convergence(self.conn, task, phase)


@pytest.fixture
def lw(migrated: Path, fake_clock: clock.FakeClock) -> Iterator[L]:
    """R1: deliverable T1 with work items T2, T3; T4 outside the family; T5 in no run. R2: T6."""
    run_id, token = open_run()
    run2, token2 = open_run("r2", "S-2")
    insert_task(migrated, "T1", run_id, is_deliverable=1, ci_state="red", ci_sha="c1")
    insert_task(migrated, "T2", run_id, deliverable_id="T1", item_k=1)
    insert_task(migrated, "T3", run_id, deliverable_id="T1", item_k=2)
    insert_task(migrated, "T4", run_id)
    insert_task(migrated, "T5", None)
    insert_task(migrated, "T6", run2)
    n1 = insert_node(migrated, "N1", run_id, "T1")
    n2 = insert_node(migrated, "N2", run_id, "T2")
    c = db.connect(migrated)
    try:
        yield L(c, fake_clock, migrated, token, token2, n1, n2)
    finally:
        c.close()


def code_of(fn: Any, *a: Any, **kw: Any) -> str:
    with pytest.raises(errors.CpError) as exc:
        fn(*a, **kw)
    return exc.value.code


# --------------------------------------------------------------------------- normalisation


class TestNormalisation:
    def test_norm(self) -> None:
        assert ledger.norm("  Off-by-ONE:  in `f()`! ") == "off by one in f"
        assert ledger.norm("snake_case__x") == "snake case x"

    def test_norm_path(self) -> None:
        assert ledger.norm_path(None) is None
        assert ledger.norm_path("././a/b.py") == "a/b.py"
        assert ledger.norm_path(" ") is None
        for bad in ("/etc/x", "a/../b", ".."):
            assert code_of(ledger.norm_path, bad) == "USAGE"

    def test_fingerprint(self) -> None:
        fp = ledger.fingerprint
        assert fp(file="a.py", symbol="f", category="test", title="x") == fp(
            file="a.py", symbol="f", category="test", title="y"
        )
        assert fp(file="a.py", symbol=None, category="test", title="x") != fp(
            file="a.py", symbol="f", category="test", title="x"
        )
        # no file, no symbol: the title joins the fingerprint
        assert fp(file=None, symbol=None, category="docs", title="X!") == fp(
            file=None, symbol=None, category="docs", title="x"
        )
        assert fp(file=None, symbol=None, category="docs", title="x") != fp(
            file=None, symbol=None, category="docs", title="y"
        )

    def test_family(self, lw: L) -> None:
        assert ledger.family(lw.conn, "T1") == ["T1", "T2", "T3"]
        assert ledger.family(lw.conn, "T3") == ["T1", "T2", "T3"]
        assert ledger.family(lw.conn, "T4") == ["T4"]
        assert code_of(ledger.family, lw.conn, "T9") == "NOT_FOUND"


# --------------------------------------------------------------------------- AC2: replay


class TestReplay:
    def test_the_same_item_twice_is_one_row(self, lw: L) -> None:
        a = lw.open("T1", item())
        assert lw.open("T1", item()) == a and lw.count() == 1
        b = lw.open("T1", [item(title="Two"), item(title="Two")])
        assert b[0] == b[1] != a[0] and lw.count() == 2

    def test_a_replay_keeps_the_first_write(self, lw: L) -> None:
        (a,) = lw.open("T1", item(body="first"))
        assert lw.open("T1", item(body="second", severity="must_fix")) == [a]
        assert lw.row(a)["body"] == "first" and lw.row(a)["severity"] == "should_fix"

    @pytest.mark.parametrize(
        ("change", "strong"), [({"title": "Another"}, False), ({"line_start": 3}, True), ({"reviewer": "r2"}, True)]
    )
    def test_items_differing_in_title_lines_or_reviewer(self, lw: L, change: dict[str, Any], strong: bool) -> None:
        first, second = lw.open("T1", [item(), item(**change)])  # in one batch: item 2 matches item 1
        assert first != second
        assert [(f["kind"], f["id"], f["strong"]) for f in lw.row(second)["flags"]] == [("matches", first, strong)]
        (third,) = lw.open("T1", item(**change, body="x"))  # in two calls: a replay of the second
        assert third == second

    def test_two_red_ci_on_different_shas_are_two_rows(self, lw: L) -> None:
        (a,) = lw.open("T1", ci_item("c1"), source="ci")
        sql(lw.db, "UPDATE tasks SET ci_sha = 'c2' WHERE id = 'T1'")
        (b,) = lw.open("T1", ci_item("c2"), source="ci")
        assert a != b and lw.row(b)["ci_sha"] == "c2"

    def test_a_ci_retry_after_a_new_round_or_after_green(self, lw: L) -> None:
        (a,) = lw.open("T1", ci_item(), source="ci")
        lw.round()
        assert lw.open("T1", ci_item(), source="ci") == [a]
        sql(lw.db, "UPDATE tasks SET ci_state = 'green' WHERE id = 'T1'")
        assert lw.open("T1", ci_item(), source="ci") == [a]
        assert lw.count() == 1


# --------------------------------------------------------------------------- AC3: exact re-raise


class TestExactReRaise:
    @pytest.mark.parametrize("closed", ["rejected", "settled"])
    def test_born_in_the_closed_state(self, lw: L, closed: str) -> None:
        a = lw.one("T1")
        lw.state(a, closed, reason="not a bug")
        b = lw.one("T1", reviewer="r2")
        row = lw.row(b)
        assert row["state"] == closed and row["matched_id"] == a and row["reason"] == f"matches #{a}: not a bug"
        assert row["flags"] == [
            {
                "kind": "matches",
                "id": a,
                "state": closed,
                "reason": "not a bug",
                "commit": None,
                "source": "reviewer",
                "strong": True,
                "cross_task": False,
                "escalated": False,
            }
        ]
        assert lw.events(b) == []
        assert lw.state(b, "open") == {"finding_id": b, "state": "open", "changed": True}
        assert [(e["kind"], e["from_value"], e["to_value"]) for e in lw.events(b)] == [("state", closed, "open")]
        assert lw.row(b)["reason"] is None

    def test_a_must_fix_on_a_should_fix_row_is_born_closed_and_escalated(self, lw: L) -> None:
        a = lw.one("T1")
        lw.state(a, "rejected", reason="by design")
        b = lw.one("T1", severity="must_fix", reviewer="r2")
        assert lw.row(b)["state"] == "rejected" and lw.row(b)["flags"][0]["escalated"] is True
        lw.classify(a, "must_fix")  # the flagged row's effective class is must_fix now: no escalation
        c = lw.one("T1", severity="must_fix", reviewer="r3")
        assert [f["escalated"] for f in lw.row(c)["flags"]] == [False, False]

    def test_a_reason_inherited_twice_names_the_root_once(self, lw: L) -> None:
        a = lw.one("T1")
        lw.state(a, "settled", reason="trade-off kept")
        b = lw.one("T1", reviewer="r2")
        c = lw.one("T1", reviewer="r3")
        assert lw.row(c)["matched_id"] == b and lw.row(c)["reason"] == f"matches #{a}: trade-off kept"

    def test_a_reopened_and_resettled_root_is_prefixed_normally(self, lw: L) -> None:
        a = lw.one("T1")
        lw.state(a, "settled", reason="first")
        b = lw.one("T1", reviewer="r2")
        lw.state(b, "open")
        lw.state(b, "settled", reason="fresh rationale")
        c = lw.one("T1", reviewer="r3")
        assert lw.row(c)["matched_id"] == b and lw.row(c)["reason"] == f"matches #{b}: fresh rationale"

    def test_latest_is_the_most_recent_decision(self, lw: L) -> None:
        a = lw.one("T1")
        lw.state(a, "rejected", reason="no")
        b = lw.one("T1", reviewer="r2")
        assert lw.row(b)["state"] == "rejected"
        lw.state(a, "open")
        lw.state(a, "resolved", commit="c9")
        assert lw.classify(b, "nice_to_have")["state"] == "rejected"  # recorded, the state is left alone
        c = lw.one("T1", reviewer="r3")
        assert lw.row(c)["state"] == "open" and lw.row(c)["matched_id"] == a and lw.row(c)["reason"] is None

    def test_decided_seq(self, lw: L) -> None:
        seq = lambda fid: lw.row(fid)["decided_seq"]  # noqa: E731
        a = lw.one("T1")
        b = lw.one("T1", title="Two", symbol="g")
        assert seq(b) > seq(a)
        before = seq(a)
        lw.state(a, "deferred")
        assert seq(a) > seq(b)
        moved = seq(a)
        lw.classify(a, "nice_to_have")  # deferred stays deferred
        lw.state(a, "deferred")  # a noop
        assert lw.open("T1", item()) == [a]  # a replay
        assert seq(a) == moved > before
        lw.classify(a, "must_fix")  # deferred → open: a state change
        assert seq(a) > moved

    def test_the_strong_match_wins_over_a_newer_weak_one(self, lw: L) -> None:
        a = lw.one("T1")
        lw.state(a, "rejected", reason="no")
        w = lw.one("T1", title="Other words")
        assert lw.row(w)["state"] == "open" and lw.row(w)["matched_id"] == a
        c = lw.one("T1", reviewer="r2")
        assert lw.row(c)["matched_id"] == a and lw.row(c)["state"] == "rejected"
        assert [(f["id"], f["strong"]) for f in lw.row(c)["flags"]] == [(a, True), (w, False)]


# --------------------------------------------------------------------------- AC4: weak match and flip-flop


class TestFlipFlop:
    def test_a_weak_match_on_a_rejected_row_is_born_open(self, lw: L) -> None:
        a = lw.one("T1")
        lw.state(a, "rejected", reason="no")
        b = lw.one("T1", title="Different words")
        row = lw.row(b)
        assert row["state"] == "open" and row["reason"] is None and row["matched_id"] == a
        assert [(f["kind"], f["strong"], f["reason"]) for f in row["flags"]] == [("matches", False, "no")]

    def test_same_symbol_as_a_resolved_row_is_an_overlap(self, lw: L) -> None:
        a = lw.one("T1")
        lw.state(a, "resolved", commit="c7")
        b = lw.one("T1", category="design", title="Revert the guard")
        row = lw.row(b)
        assert row["state"] == "open" and row["matched_id"] is None
        assert row["flags"] == [
            {
                "kind": "overlaps_fix",
                "id": a,
                "state": "resolved",
                "reason": None,
                "commit": "c7",
                "source": "reviewer",
                "strong": False,
                "cross_task": False,
                "escalated": False,
            }
        ]
        assert lw.row(lw.one("T1", category="design", symbol="g"))["flags"] == []  # another symbol: no overlap

    def test_reraise_of_a_resolved_row_then_settle(self, lw: L) -> None:
        a = lw.one("T1")
        lw.state(a, "resolved", commit="c7")
        b = lw.one("T1", reviewer="r2")
        assert lw.row(b)["state"] == "open" and [(f["kind"], f["state"]) for f in lw.row(b)["flags"]] == [
            ("matches", "resolved")
        ]
        lw.state(b, "settled", reason="the fix stays: the guard is needed")
        assert lw.row(b)["reason"] == "the fix stays: the guard is needed"
        c = lw.one("T1", reviewer="r3")
        assert lw.row(c)["state"] == "settled" and lw.row(c)["matched_id"] == b
        assert lw.row(c)["reason"] == f"matches #{b}: the fix stays: the guard is needed"

    def test_relates_to_a_settled_row(self, lw: L) -> None:
        a = lw.one("T1", category="docs", file=None, symbol=None, title="Plan says X")
        lw.state(a, "settled", reason="X is right")
        (b,) = lw.open(
            "T1", item(category="docs", file=None, symbol=None, title="Plan should say Y", relates_to=[a, a])
        )
        row = lw.row(b)
        assert row["state"] == "open"
        assert [(f["kind"], f["id"], f["reason"], f["strong"], f["cross_task"]) for f in row["flags"]] == [
            ("relates_to", a, "X is right", False, False)
        ]

    def test_an_unknown_relates_to_is_not_found(self, lw: L) -> None:
        assert code_of(lw.open, "T1", item(relates_to=[99])) == "NOT_FOUND"
        assert lw.count() == 0


# --------------------------------------------------------------------------- AC5: family


class TestFamily:
    def test_cross_task_matches(self, lw: L) -> None:
        on_item = lw.one("T2")
        lw.state(on_item, "rejected", reason="out of this item")
        d1 = lw.one("T1", source="global_review")
        row = lw.row(d1)
        assert row["state"] == "open" and row["reason"] is None and row["matched_id"] == on_item
        assert [(f["kind"], f["id"], f["strong"], f["cross_task"], f["reason"]) for f in row["flags"]] == [
            ("matches", on_item, True, True, "out of this item")
        ]
        sib = lw.one("T3")
        assert lw.row(sib)["state"] == "open" and lw.row(sib)["matched_id"] == d1  # the latest family strong match
        assert [f["cross_task"] for f in lw.row(sib)["flags"]] == [True, True]

        lw.state(d1, "rejected", reason="re-assessed on the deliverable", token=lw.n1)
        lw.one("T3", reviewer="r2")  # a sibling raises it again since: a newer, open, family row
        d2 = lw.one("T1", source="global_review", reviewer="g2")
        assert lw.row(d2)["state"] == "rejected" and lw.row(d2)["matched_id"] == d1
        assert lw.row(d2)["reason"] == f"matches #{d1}: re-assessed on the deliverable"

    def test_the_same_task_strong_match_comes_first(self, lw: L) -> None:
        own = lw.one("T1")  # open: nothing to inherit, but still the first hit of the D8 order
        sibling = lw.one("T2")  # a newer strong match of the family
        again = lw.one("T1", reviewer="r2")
        assert lw.row(again)["matched_id"] == own and lw.row(again)["state"] == "open"
        assert [(f["id"], f["cross_task"]) for f in lw.row(again)["flags"]] == [(own, False), (sibling, True)]

    def test_a_task_outside_the_family_is_not_matched(self, lw: L) -> None:
        a = lw.one("T2")
        lw.state(a, "rejected", reason="no")
        b = lw.one("T4")
        assert lw.row(b)["flags"] == [] and lw.row(b)["matched_id"] is None and lw.row(b)["state"] == "open"

    def test_matching_crosses_phases(self, lw: L) -> None:
        (a,) = lw.open("T1", item(), phase="plan")
        lw.state(a, "settled", reason="decided at plan time")
        b = lw.one("T1")
        assert lw.row(b)["phase"] == "build" and lw.row(b)["state"] == "settled"


# --------------------------------------------------------------------------- AC6, AC7: ci and gate


class TestCiAndGate:
    def test_a_red_ci_finding(self, lw: L) -> None:
        (a,) = lw.open("T1", ci_item(severity="nice_to_have"), source="ci")
        row = lw.row(a)
        assert (row["severity"], row["classification"], row["ci_sha"], row["state"]) == (
            "must_fix",
            "must_fix",
            "c1",
            "open",
        )
        assert row["round"] == 0

    @pytest.mark.parametrize(("ci_state", "ci_sha"), [("green", "c1"), ("pending", "c1"), (None, None), ("red", "c2")])
    def test_not_red_on_that_sha_is_invalid_state(self, lw: L, ci_state: str | None, ci_sha: str | None) -> None:
        sql(lw.db, "UPDATE tasks SET ci_state = ?, ci_sha = ? WHERE id = 'T1'", [ci_state, ci_sha])
        assert code_of(lw.open, "T1", ci_item("c1"), source="ci") == "INVALID_STATE"
        assert lw.count() == 0

    def test_born_open_on_a_strong_match_with_a_settled_row(self, lw: L) -> None:
        (a,) = lw.open("T1", item(category="ci", file=None, symbol=None, title="test_x failed"))
        lw.state(a, "settled", reason="flaky, accepted")
        (b,) = lw.open("T1", ci_item(relates_to=[a]), source="ci")
        row = lw.row(b)
        assert row["state"] == "open" and row["matched_id"] == a
        assert [(f["kind"], f["strong"], f["escalated"], f["reason"]) for f in row["flags"]] == [
            ("matches", True, True, "flaky, accepted"),
            ("relates_to", False, False, "flaky, accepted"),
        ]
        assert lw.state(a, "open", cause=b)["changed"] is True  # the node reopens the settled decision
        assert [(e["to_value"], e["cause_id"]) for e in lw.events(a)] == [("settled", None), ("open", b)]

    @pytest.mark.parametrize("source", ["orchestrator", "maintainer", "detector"])
    def test_authority_sources_are_always_born_open(self, lw: L, source: str) -> None:
        a = lw.one("T1")
        lw.state(a, "rejected", reason="no")
        b = lw.one("T1", source=source)
        assert lw.row(b)["state"] == "open" and lw.row(b)["flags"][0]["strong"] is True

    def test_a_gate_finding(self, lw: L) -> None:
        cur = lw.conn.execute(
            "INSERT INTO integrations (item_task_id, deliverable_id, item_tip, result, at)"
            " VALUES ('T2', 'T1', 't', 'gate_red', 'x')"
        )
        iid = int(cur.lastrowid or 0)
        gate = {"category": "gate", "title": "gate red", "body": "E   assert 1 == 2", "integration_id": iid}
        for task in ("T2", "T1"):
            (fid,) = lw.open(task, gate, source="gate")
            row = lw.row(fid)
            assert (row["severity"], row["classification"], row["integration_id"]) == ("must_fix", "must_fix", iid)
        assert code_of(lw.open, "T1", {**gate, "integration_id": 99}, source="gate") == "NOT_FOUND"
        assert code_of(lw.open, "T4", gate, source="gate") == "CONFLICT"


# --------------------------------------------------------------------------- AC8: tokens


class TestTokens:
    def test_run_and_node_scopes(self, lw: L) -> None:
        assert lw.one("T2") and lw.one("T4")
        assert code_of(lw.one, "T6") == "CONFLICT"  # another run's task
        assert lw.one("T6", token=lw.token2)
        assert lw.one("T1", token=lw.n1, title="node", symbol="n")
        assert code_of(lw.one, "T2", token=lw.n1) == "CONFLICT"
        assert lw.one("T2", token=lw.n2, title="node", symbol="n")

    def test_a_task_with_no_run_stores_the_principals_run(self, lw: L) -> None:
        a = lw.one("T5", token=lw.token2)
        assert lw.row(a)["run_id"] == "R2" and lw.row(a)["actor"] == "orchestrator"
        assert lw.row(lw.one("T1"))["run_id"] == "R1"

    def test_blast_radius_needs_the_run_token(self, lw: L) -> None:
        kw = {"task_id": "T1", "value": "ok", "head_sha": "h", "review": "R", "reason": None}
        assert code_of(ledger.set_blast_radius, lw.conn, lw.clock, token=lw.n1, **kw) == "CONFLICT"

    def test_id_commands_check_the_token_before_the_id(self, lw: L) -> None:
        assert code_of(lw.state, 99, "open", token=BAD) == "STALE_TOKEN"
        assert code_of(lw.classify, 99, "must_fix", token=BAD) == "STALE_TOKEN"
        assert code_of(lw.state, 99, "open") == "NOT_FOUND"
        assert code_of(lw.classify, 99, "must_fix") == "NOT_FOUND"

    def test_id_commands_check_the_task_scope(self, lw: L) -> None:
        a = lw.one("T2")
        assert code_of(lw.state, a, "deferred", token=lw.n1) == "CONFLICT"
        assert code_of(lw.classify, a, "must_fix", token=lw.n1) == "CONFLICT"
        assert lw.classify(a, "must_fix", token=lw.n2)["changed"] is True
        assert code_of(lw.state, lw.one("T6", token=lw.token2), "deferred") == "CONFLICT"


# --------------------------------------------------------------------------- AC9: convergence


class TestConvergence:
    def test_no_round(self, lw: L) -> None:
        assert lw.conv() == {
            "converged": False,
            "consistent": True,
            "blocking": [],
            "latest_round": 0,
            "counts": {"open": 0, "resolved": 0, "rejected": 0, "deferred": 0, "settled": 0},
        }
        lw.report("converged", round_=0)  # a report with no round never declares it
        assert lw.conv()["converged"] is False

    def test_the_node_declares_it(self, lw: L) -> None:
        lw.round()
        assert lw.conv()["converged"] is False  # in review
        lw.report("continuing", token=lw.n1)
        assert lw.conv()["converged"] is False
        lw.report("converged", token=lw.n1)
        assert lw.conv()["converged"] is True and lw.conv()["consistent"] is True
        lw.report("continuing", token=lw.n1)  # a later report in the same round withdraws it
        assert lw.conv()["converged"] is False
        lw.report("converged")  # the orchestrator's declaration counts too
        assert lw.conv()["converged"] is True
        lw.round()
        assert lw.conv()["converged"] is False and lw.conv()["latest_round"] == 2

    def test_a_report_of_another_phase_spelling_does_not_count(self, lw: L) -> None:
        lw.round()
        lw.report("converged", phase="Build")
        lw.report("converged", phase="build ")
        assert lw.conv()["converged"] is False

    def test_blocking_and_consistent(self, lw: L) -> None:
        zero = lw.one("T1", severity="must_fix", title="round zero", symbol="z")
        lw.round()
        must = lw.one("T1", severity="must_fix", title="m", symbol="m")
        unclassified = lw.one("T1", title="s", symbol="s")
        nice = lw.one("T1", severity="nice_to_have", title="n", symbol="n")
        deferred, settled, rejected, resolved = (lw.one("T1", title=t, symbol=t) for t in "dxrv")
        lw.state(deferred, "deferred")
        lw.state(settled, "settled", reason="r")
        lw.state(rejected, "rejected", reason="r")
        lw.state(resolved, "resolved", commit="c")
        oos = lw.one("T1", severity="must_fix", title="o", symbol="o")
        lw.classify(oos, "out_of_scope")
        lw.state(oos, "open")  # reopened by hand: still out of scope
        promoted = lw.one("T1", severity="nice_to_have", title="p", symbol="p")
        lw.classify(promoted, "should_fix")
        lw.report("converged", token=lw.n1)
        conv = lw.conv()
        assert conv["blocking"] == [zero, must, unclassified, promoted]
        assert conv["converged"] is True and conv["consistent"] is False
        assert nice not in conv["blocking"] and oos not in conv["blocking"]
        assert conv["counts"] == {"open": 6, "resolved": 1, "rejected": 1, "deferred": 1, "settled": 1}
        for fid in conv["blocking"]:
            lw.state(fid, "deferred")
        assert lw.conv()["consistent"] is True and lw.conv()["converged"] is True

    def test_plan_and_build_apart(self, lw: L) -> None:
        lw.round(phase="plan", head=None)
        lw.open("T1", item(severity="must_fix"), phase="plan")
        lw.report("converged", phase="plan")
        plan, build = lw.conv(phase="plan"), lw.conv()
        assert plan["converged"] is True and plan["consistent"] is False and plan["latest_round"] == 1
        assert build["converged"] is False and build["blocking"] == [] and build["latest_round"] == 0
        assert code_of(ledger.convergence, lw.conn, "T1", "review") == "USAGE"


# --------------------------------------------------------------------------- AC10: unchanged lines, classification


class TestClassification:
    def test_unchanged_lines_is_stored_as_given(self, lw: L) -> None:
        ids = lw.open("T1", [item(unchanged_lines=True), item(unchanged_lines=False, title="b"), item(title="c")])
        assert [lw.row(i)["unchanged_lines"] for i in ids] == [1, 0, None]

    def test_born_classification(self, lw: L) -> None:
        assert lw.row(lw.one("T1"))["classification"] is None

    def test_classify_moves_open_and_deferred(self, lw: L) -> None:
        a = lw.one("T1")
        out = lw.classify(a, "nice_to_have", note="cosmetic", token=lw.n1)
        assert out == {"finding_id": a, "state": "deferred", "classification": "nice_to_have", "changed": True}
        row = lw.row(a)
        assert row["state"] == "deferred" and row["reason"] == "classified nice_to_have"
        assert [(e["kind"], e["from_value"], e["to_value"], e["reason"], e["actor"]) for e in lw.events(a)] == [
            ("classify", None, "nice_to_have", "cosmetic", "node:T1"),
            ("state", "open", "deferred", "classified nice_to_have", "node:T1"),
        ]
        assert lw.classify(a, "out_of_scope")["state"] == "deferred"  # deferred stays deferred
        assert lw.classify(a, "must_fix")["state"] == "open"
        assert lw.row(a)["reason"] == "classified must_fix"
        assert [e["actor"] for e in lw.events(a)][-2:] == ["orchestrator", "orchestrator"]
        assert lw.classify(a, "must_fix") == {
            "finding_id": a,
            "state": "open",
            "classification": "must_fix",
            "changed": False,
        }
        assert len(lw.events(a)) == 5

    @pytest.mark.parametrize("state", ["resolved", "rejected", "settled"])
    def test_classify_leaves_other_states(self, lw: L, state: str) -> None:
        a = lw.one("T1")
        lw.state(a, state, reason="r" if state != "resolved" else None, commit="c" if state == "resolved" else None)
        for cls in ("nice_to_have", "must_fix"):
            assert lw.classify(a, cls)["state"] == state
        assert [e["kind"] for e in lw.events(a)] == ["state", "classify", "classify"]

    def test_classify_validation(self, lw: L) -> None:
        assert code_of(lw.classify, lw.one("T1"), "critical") == "USAGE"

    def test_reason_and_resolved_sha_follow_the_latest_state(self, lw: L) -> None:
        a = lw.one("T1")
        lw.state(a, "resolved", commit="c1")
        assert (lw.row(a)["resolved_sha"], lw.row(a)["reason"]) == ("c1", None)
        lw.state(a, "open", reason="it came back")
        assert (lw.row(a)["resolved_sha"], lw.row(a)["reason"]) == (None, "it came back")
        lw.state(a, "deferred")
        assert lw.row(a)["reason"] is None


# --------------------------------------------------------------------------- AC12: blast radius


class TestBlastRadius:
    def test_append_and_read(self, lw: L) -> None:
        put = lambda v, h, r: ledger.set_blast_radius(  # noqa: E731
            lw.conn, lw.clock, token=lw.token, task_id="T1", value=v, head_sha=h, review=r, reason=None
        )
        assert ledger.blast_radius(lw.conn, "T1") is None
        assert put("ok", "h1", "G1") == {"id": 1, "task_id": "T1", "value": "ok", "head_sha": "h1"}
        put("doubt", "h2", "G2")
        latest = ledger.blast_radius(lw.conn, "T1")
        assert latest is not None and (latest["value"], latest["review"], latest["actor"]) == (
            "doubt",
            "G2",
            "orchestrator",
        )
        by_sha = ledger.blast_radius(lw.conn, "T1", "h1")
        assert by_sha is not None and by_sha["value"] == "ok"
        assert ledger.blast_radius(lw.conn, "T1", "h9") is None
        assert code_of(put, "huge", "h", "G") == "USAGE"
        assert code_of(put, "ok", " ", "G") == "USAGE"


# --------------------------------------------------------------------------- AC14: states and validation


class TestStates:
    def test_free_transitions(self, lw: L) -> None:
        a = lw.one("T1")
        path = [("resolved", None, "c"), ("rejected", "r", None), ("settled", "s", None), ("deferred", None, None)]
        path += [("open", None, None), ("settled", "s2", None), ("resolved", None, "c2"), ("open", None, None)]
        for to, reason, commit in path:
            assert lw.state(a, to, reason=reason, commit=commit)["changed"] is True
        assert [e["to_value"] for e in lw.events(a)] == [p[0] for p in path]

    @pytest.mark.parametrize(
        ("to", "kw"),
        [
            ("resolved", {}),
            ("rejected", {}),
            ("settled", {"reason": "  "}),
            ("open", {"commit": "c"}),
            ("deferred", {"commit": "c"}),
            ("rejected", {"reason": "r", "commit": "c"}),
            ("closed", {}),
        ],
    )
    def test_required_arguments(self, lw: L, to: str, kw: dict[str, Any]) -> None:
        assert code_of(lw.state, lw.one("T1"), to, **kw) == "USAGE"

    def test_the_noop_and_the_cause(self, lw: L) -> None:
        a = lw.one("T1")
        assert lw.state(a, "open") == {"finding_id": a, "state": "open", "changed": False}
        assert code_of(lw.state, a, "open", cause=99) == "NOT_FOUND"  # checked before the noop
        b = lw.one("T1", title="b", symbol="b")
        lw.state(a, "deferred", reason="later", cause=b)
        ev = lw.events(a)
        assert [(e["to_value"], e["reason"], e["cause_id"], e["round"]) for e in ev] == [("deferred", "later", b, 0)]
        lw.round()
        lw.state(a, "open")
        assert lw.events(a)[-1]["round"] == 1


class TestItemSchema:
    @pytest.mark.parametrize(
        "bad",
        [
            {},
            [],
            "text",
            [item(), "x"],
            item(extra=1),
            item(severity=None),
            item(severity="blocker"),
            item(severity=3),
            item(category=None),
            item(category="vibes"),
            item(title=" "),
            item(title=None),
            item(body=None),
            item(body=4),
            item(file="/abs.py"),
            item(file="a/../b.py"),
            item(file=1),
            item(line_start=0),
            item(line_start=True),
            item(line_start=5, line_end=4),
            item(line_end=4),
            item(reviewer=["x"]),
            item(unchanged_lines="yes"),
            item(relates_to=1),
            item(relates_to=["1"]),
            item(relates_to=[True]),
            item(sha="c1"),
            item(integration_id=1),
        ],
    )
    def test_each_error_is_usage(self, lw: L, bad: Any) -> None:
        assert code_of(lw.open, "T1", bad) == "USAGE"
        assert lw.count() == 0

    @pytest.mark.parametrize(
        ("source", "bad"),
        [
            ("ci", {"category": "ci", "title": "t", "body": "b"}),
            ("ci", {"category": "ci", "title": "t", "body": "b", "sha": " "}),
            ("gate", {"category": "gate", "title": "t", "body": "b"}),
            ("ci", {"category": "ci", "title": "t", "body": "b", "sha": "c1", "integration_id": 1}),
            ("nobody", item()),
        ],
    )
    def test_source_specific_errors(self, lw: L, source: str, bad: Any) -> None:
        assert code_of(lw.open, "T1", bad, source=source) == "USAGE"

    def test_normalised_fields(self, lw: L) -> None:
        (a,) = lw.open("T1", item(file="./src/a.py", symbol="  f  ", line_start=4, reviewer=None))
        row = lw.row(a)
        assert (row["file"], row["symbol"], row["line_start"], row["line_end"]) == ("src/a.py", "f", 4, 4)
        (b,) = lw.open("T1", item(file="src/a.py", symbol="f", line_start=4, body="again"))
        assert b == a  # the normalised file and symbol replay
        (c,) = lw.open("T1", item(symbol=" ", file=None, title="t"))
        assert lw.row(c)["symbol"] is None

    def test_phase_and_round(self, lw: L) -> None:
        assert code_of(lw.open, "T1", item(), phase="review") == "USAGE"
        assert code_of(lw.open, "T1", item(), round_=0) == "USAGE"
        assert code_of(lw.open, "T1", item(), round_=1) == "NOT_FOUND"
        lw.round()
        lw.round()
        assert lw.row(lw.open("T1", item(), round_=1)[0])["round"] == 1
        assert lw.row(lw.one("T1", title="later"))["round"] == 2  # the default: the latest round


class TestRounds:
    def test_numbering_base_and_head(self, lw: L) -> None:
        assert lw.round(head="h1", base="m0") == {
            "task_id": "T1",
            "phase": "build",
            "round": 1,
            "base_sha": "m0",
            "head_sha": "h1",
        }
        assert lw.round(head="h2")["base_sha"] == "h1"  # the previous round's head
        assert lw.round(head="h3", base="merge1")["base_sha"] == "merge1"  # after a catch-up with main
        assert lw.round(phase="plan", head=None) == {
            "task_id": "T1",
            "phase": "plan",
            "round": 1,
            "base_sha": None,
            "head_sha": None,
        }
        assert lw.round(head="h4")["round"] == 4  # a plan round does not move the build counter
        assert lw.round(task="T2")["round"] == 1
        actors = [r[0] for r in lw.conn.execute("SELECT actor FROM rounds ORDER BY rowid")]
        assert actors[0] == "orchestrator"
        assert lw.round(task="T1", head="h5", token=lw.n1)["round"] == 5
        assert lw.conn.execute("SELECT actor FROM rounds WHERE round = 5").fetchone()[0] == "node:T1"

    def test_a_first_round_with_no_base(self, lw: L) -> None:
        assert lw.round(head="h1")["base_sha"] is None

    def test_validation(self, lw: L) -> None:
        assert code_of(lw.round, head=None) == "USAGE"
        assert code_of(lw.round, phase="review") == "USAGE"
        assert code_of(lw.round, task="T2", token=lw.n1) == "CONFLICT"


# --------------------------------------------------------------------------- AC16: ledger show


class TestShow:
    def test_filters_family_and_current_state(self, lw: L) -> None:
        on_item = lw.one("T2")
        lw.state(on_item, "rejected", reason="no")
        lw.round()
        (plan,) = lw.open("T1", item(title="plan only", symbol="p"), phase="plan")
        d = lw.one("T1", source="global_review")
        lw.classify(d, "must_fix")
        lw.state(on_item, "open")  # the flagged row moved since: show reports its current state
        ledger.set_blast_radius(
            lw.conn, lw.clock, token=lw.token, task_id="T1", value="ok", head_sha="h", review="G", reason=None
        )
        out = ledger.show(lw.conn, "T1")
        assert [f["id"] for f in out["findings"]] == [plan, d]
        assert [(fl["state"], fl["current_state"]) for fl in out["findings"][1]["flags"]] == [("rejected", "open")]
        assert [e["finding_id"] for e in out["finding_events"]] == [d]
        assert [r["round"] for r in out["rounds"]] == [1] and len(out["blast_radius"]) == 1
        assert set(out["convergence"]) == {"plan", "build"}
        build = ledger.show(lw.conn, "T1", phase="build")
        assert [f["id"] for f in build["findings"]] == [d]
        assert ledger.show(lw.conn, "T1", phase="plan")["rounds"] == []
        fam = ledger.show(lw.conn, "T1", include_family=True)
        assert [f["id"] for f in fam["findings"]] == [on_item, plan, d]
        assert [e["finding_id"] for e in fam["finding_events"]] == [on_item, d, on_item]
        assert fam["rounds"] == out["rounds"] and fam["blast_radius"] == out["blast_radius"]  # per task only
        only = ledger.show(lw.conn, "T1", include_family=True, states=["open"], phase="build")
        assert [f["id"] for f in only["findings"]] == [on_item, d]
        assert ledger.show(lw.conn, "T1", states=["settled"])["findings"] == []

    def test_validation(self, lw: L) -> None:
        assert code_of(ledger.show, lw.conn, "T1", states=["closed"]) == "USAGE"
        assert code_of(ledger.show, lw.conn, "T1", phase="review") == "USAGE"
        assert code_of(ledger.show, lw.conn, "T9") == "NOT_FOUND"


# --------------------------------------------------------------------------- the CLI (AC8, AC13)


class TestCli:
    def _write(self, tmp: Path, payload: Any) -> str:
        p = tmp / "items.json"
        p.write_text(json.dumps(payload))
        return str(p)

    def test_the_reviewer_contract(self, lw: L, tmp_path: Path) -> None:
        base = ["finding", "open", "--task", "T1", "--phase", "build", "--source", "reviewer"]
        code, out = run_cli(*base, "--input", self._write(tmp_path, [item(), item(title="x")]), "--token", lw.n1)
        assert (code, out) == (0, {"ok": True, "ids": [1, 2]})
        code, out = run_cli(*base, "--input", "-", "--token", lw.n1, stdin=json.dumps(item(title="y")))
        assert (code, out) == (0, {"ok": True, "ids": [3]})
        code, out = run_cli(*base, "--input", "-", "--token", lw.n1, stdin="not json")
        assert code == 2 and out["error"] == "USAGE"
        assert run_cli(*base, "--input", str(tmp_path / "missing.json"), "--token", lw.n1)[1]["error"] == "USAGE"
        code, out = run_cli(*base, "--round", "7", "--input", "-", "--token", lw.n1, stdin=json.dumps(item(title="z")))
        assert out["error"] == "NOT_FOUND"

    def test_the_commands(self, lw: L, tmp_path: Path) -> None:
        tok = ("--token", lw.token)
        code, out = run_cli("round", "start", "--task", "T1", "--phase", "build", "--head", "h1", "--base", "m", *tok)
        assert (code, out["round"], out["base_sha"], out["head_sha"]) == (0, 1, "m", "h1")
        assert run_cli("round", "start", "--task", "T1", "--phase", "build", *tok)[1]["error"] == "USAGE"
        inp = self._write(tmp_path, item())
        run_cli("finding", "open", "--task", "T1", "--phase", "build", "--source", "reviewer", "--input", inp, *tok)
        code, out = run_cli("finding", "classify", "1", "--class", "nice_to_have", "--note", "n", *tok)
        assert (code, out["state"], out["changed"]) == (0, "deferred", True)
        code, out = run_cli("finding", "state", "1", "--to", "resolved", "--commit", "c1", *tok)
        assert (code, out) == (0, {"ok": True, "finding_id": 1, "state": "resolved", "changed": True})
        assert run_cli("finding", "state", "1", "--to", "open", "--commit", "c", *tok)[1]["error"] == "USAGE"
        cause = lw.one("T1", title="the cause", symbol="c")
        lw.state(cause, "resolved", commit="c2")
        code, out = run_cli("finding", "state", "1", "--to", "open", "--cause", str(cause), "--reason", "back", *tok)
        assert code == 0 and lw.events(1)[-1]["cause_id"] == cause
        bl = ("blast-radius", "set", "--task", "T1", "--value", "doubt", "--head-sha", "h1", "--review", "G1")
        code, out = run_cli(*bl, "--reason", "wide", *tok)
        assert code == 0 and out["value"] == "doubt"
        assert run_cli(*bl, "--token", lw.n1)[1]["error"] == "CONFLICT"
        code, out = run_cli(
            "ledger", "show", "--task", "T1", "--phase", "build", "--state", "open, deferred ,", "--family"
        )
        assert code == 0 and [f["id"] for f in out["findings"]] == [1] and out["blast_radius"][0]["reason"] == "wide"
        assert [f["id"] for f in run_cli("ledger", "show", "--task", "T1", "--state", "resolved")[1]["findings"]] == [
            cause
        ]
        assert run_cli("ledger", "show", "--task", "T1", "--state", "nope")[1]["error"] == "USAGE"
        assert run_cli("ledger", "show", "--task", "T9")[1]["error"] == "NOT_FOUND"

    def test_tokens_through_the_cli(self, lw: L, tmp_path: Path) -> None:
        inp = self._write(tmp_path, item())
        base = ["finding", "open", "--phase", "build", "--source", "reviewer", "--input", inp]
        assert run_cli(*base, "--task", "T6", "--token", lw.token)[1]["error"] == "CONFLICT"
        assert run_cli(*base, "--task", "T2", "--token", lw.n1)[1]["error"] == "CONFLICT"
        assert run_cli(*base, "--task", "T5", "--token", lw.token)[0] == 0
        assert lw.row(1)["run_id"] == "R1"
        assert run_cli("finding", "state", "99", "--to", "open", "--token", BAD)[1]["error"] == "STALE_TOKEN"
        assert run_cli("finding", "classify", "99", "--class", "must_fix", "--token", BAD)[1]["error"] == "STALE_TOKEN"

    def test_ledger_show_without_a_db(self, db_path: Path) -> None:
        assert not db_path.exists()
        code, out = run_cli("ledger", "show", "--task", "T1")
        assert code == 8 and out["error"] == "NOT_FOUND"


# --------------------------------------------------------------------------- review fix #01


class TestReviewFix01:
    def test_f2_relates_to_another_runs_finding_is_not_found(self, lw: L) -> None:
        other = lw.one("T6", token=lw.token2)
        assert code_of(lw.open, "T1", item(relates_to=[other])) == "NOT_FOUND"
        assert lw.open("T6", item(title="x", relates_to=[other]), token=lw.token2)

    @pytest.mark.parametrize("title", ["???", "—", " 🔥 ", "__"])
    def test_f3_a_title_that_normalises_to_nothing(self, lw: L, title: str) -> None:
        assert code_of(lw.open, "T1", item(title=title)) == "USAGE"

    @pytest.mark.parametrize("file", ["a.py:10", "a.py:10-12", "a\\b.py", "..\\x.py"])
    def test_f4_bad_file_spellings(self, lw: L, file: str) -> None:
        assert code_of(lw.open, "T1", item(file=file)) == "USAGE"

    @pytest.mark.parametrize(("given", "stored"), [("a//b/./c.py", "a/b/c.py"), ("src/", "src"), ("./x/../y.py", None)])
    def test_f4_paths_are_normalised(self, lw: L, given: str, stored: str | None) -> None:
        if stored is None:
            assert code_of(lw.open, "T1", item(file=given)) == "USAGE"  # `..` is refused before normalising
        else:
            assert lw.row(lw.open("T1", item(file=given))[0])["file"] == stored

    def test_f5_a_gate_severity_is_overridden(self, lw: L) -> None:
        cur = lw.conn.execute(
            "INSERT INTO integrations (item_task_id, deliverable_id, item_tip, result, at)"
            " VALUES ('T2', 'T1', 't', 'gate_red', 'x')"
        )
        gate = {
            "severity": "nice_to_have",
            "category": "gate",
            "title": "g",
            "body": "b",
            "integration_id": cur.lastrowid,
        }
        row = lw.row(lw.open("T1", gate, source="gate")[0])
        assert (row["severity"], row["classification"]) == ("must_fix", "must_fix")

    def test_f7_an_empty_state_filter(self, lw: L) -> None:
        assert code_of(ledger.show, lw.conn, "T1", states=[]) == "USAGE"
        for given in (",", " ", ""):
            assert run_cli("ledger", "show", "--task", "T1", "--state", given)[1]["error"] == "USAGE"

    def test_f8_a_finding_is_not_its_own_cause(self, lw: L) -> None:
        a = lw.one("T1")
        assert code_of(lw.state, a, "deferred", cause=a) == "USAGE"

    def test_f9_blank_strings(self, lw: L) -> None:
        assert code_of(lw.round, head="  ") == "USAGE"
        lw.round(head=" h1 ")
        assert lw.round(head="h2", base="  ")["base_sha"] == "h1"  # a blank base is no base: the default applies
        a = lw.one("T1")
        assert code_of(lw.state, a, "resolved", commit=" ") == "USAGE"
        lw.state(a, "deferred", reason="  ")
        assert lw.row(a)["reason"] is None and lw.events(a)[-1]["reason"] is None
        lw.state(a, "resolved", commit=" c1 ")
        assert lw.row(a)["resolved_sha"] == "c1"
        lw.classify(a, "must_fix", note=" ")
        assert lw.events(a)[-1]["reason"] is None
        (ci,) = lw.open("T1", ci_item(" c1 "), source="ci")
        assert lw.row(ci)["ci_sha"] == "c1"
        (r,) = lw.open("T1", item(title="rv", reviewer=" qs-x "))
        assert lw.row(r)["reviewer"] == "qs-x" and lw.open("T1", item(title="rv", reviewer="qs-x")) == [r]
        (blank,) = lw.open("T1", item(title="no lens", reviewer="  "))
        assert lw.row(blank)["reviewer"] is None

    def test_f10_free_text_is_escaped_in_the_export(self, lw: L) -> None:
        lw.round(head="a|b")
        ledger.set_blast_radius(
            lw.conn, lw.clock, token=lw.token, task_id="T1", value="ok", head_sha="h", review="G|1", reason="x\n\n### y"
        )
        rounds = dict(export_sections())["Rounds"](lw.conn, "T1")
        assert "`?..a\\|b`" in rounds
        blast = dict(export_sections())["Blast radius"](lw.conn, "T1")
        assert blast == "`ok` at `h` (review G|1) — x ### y"  # a paragraph, not a table: no pipe escape (G10)

    def test_f11_convergence_of_an_unknown_task(self, lw: L) -> None:
        assert code_of(ledger.convergence, lw.conn, "T9", "build") == "NOT_FOUND"

    def test_f12_bad_input_files(self, lw: L, tmp_path: Path) -> None:
        base = ["finding", "open", "--task", "T1", "--phase", "build", "--source", "reviewer", "--token", lw.token]
        bad = tmp_path / "latin1.json"
        bad.write_bytes(b'{"title": "caf\xe9"}')
        code, out = run_cli(*base, "--input", str(bad))
        assert code == 2 and out["error"] == "USAGE", out
        code, out = run_cli(*base, "--input", "-", stdin="[" * 100_000 + "]" * 100_000)
        assert code == 2 and out["error"] == "USAGE", out

    @pytest.mark.parametrize(
        "bad",
        [item(line_start=2**63), item(line_start=1, line_end=2**63), item(relates_to=[2**63]), item(relates_to=[0])],
    )
    def test_f13_out_of_range_item_ints(self, lw: L, bad: dict[str, Any]) -> None:
        assert code_of(lw.open, "T1", bad) == "USAGE"

    def test_f13_out_of_range_cli_ints(self, lw: L, tmp_path: Path) -> None:
        huge = str(2**63)
        tok = ("--token", lw.token)
        assert run_cli("finding", "state", huge, "--to", "open", *tok)[1]["error"] == "USAGE"
        assert run_cli("finding", "classify", huge, "--class", "must_fix", *tok)[1]["error"] == "USAGE"
        assert run_cli("finding", "state", "1", "--to", "open", "--cause", huge, *tok)[1]["error"] == "USAGE"
        assert run_cli("finding", "state", "0", "--to", "open", *tok)[1]["error"] == "USAGE"
        inp = tmp_path / "i.json"
        inp.write_text(json.dumps(item()))
        argv = ("finding", "open", "--task", "T1", "--phase", "build", "--source", "reviewer", "--input", str(inp))
        assert run_cli(*argv, "--round", huge, *tok)[1]["error"] == "USAGE"
        gate = {"category": "gate", "title": "g", "body": "b", "integration_id": 2**63}
        assert code_of(lw.open, "T1", gate, source="gate") == "USAGE"

    @pytest.mark.parametrize("key", ["title", "file", "symbol", "reviewer"])
    def test_f14_the_separator_is_refused(self, lw: L, key: str) -> None:
        assert code_of(lw.open, "T1", item(**{key: "a\x1fb"})) == "USAGE"

    def test_f14_the_separator_in_a_ci_sha(self, lw: L) -> None:
        assert code_of(lw.open, "T1", ci_item("c\x1f1"), source="ci") == "USAGE"

    def test_f15_unicode_spellings(self) -> None:
        assert ledger.norm("Café") == ledger.norm("Café") == "café"
        assert ledger.norm("Straße") == ledger.norm("STRASSE")

    def test_f19_two_calls_give_two_rows_and_a_match(self, lw: L) -> None:
        a = lw.one("T1")
        b = lw.one("T1", title="Another wording")
        assert a != b and [(f["kind"], f["id"], f["strong"]) for f in lw.row(b)["flags"]] == [("matches", a, False)]


class TestReviewFix02:
    def test_g1_a_non_integer_id(self, lw: L) -> None:
        assert run_cli("finding", "state", "abc", "--to", "open", "--token", lw.token)[1]["error"] == "USAGE"

    @pytest.mark.parametrize("file", ["a.py:42/", "a.py:10-12//", "a.py:10/.", "C:/repo/x.py", "c:x.py"])
    def test_g2_normalised_then_checked(self, lw: L, file: str) -> None:
        assert code_of(lw.open, "T1", item(file=file)) == "USAGE"

    def test_g3_lone_surrogates_and_non_utf8_stdin(self, lw: L) -> None:
        for key in ("title", "body", "file", "symbol", "reviewer"):
            assert code_of(lw.open, "T1", item(**{key: "x\ud800"})) == "USAGE", key
        assert code_of(lw.open, "T1", ci_item("c\udc80"), source="ci") == "USAGE"

        class BadStdin:
            def read(self) -> str:
                return b"\xe9".decode("utf-8")  # raises UnicodeDecodeError, as a strict UTF-8 stdin does

        import io as io_mod

        from control_plane import cli

        out = io_mod.StringIO()
        argv = ["finding", "open", "--task", "T1", "--phase", "build", "--source", "reviewer", "--input", "-"]
        code = cli.main([*argv, "--token", lw.token], stdin=BadStdin(), stdout=out)  # type: ignore[arg-type]
        assert code == 2 and json.loads(out.getvalue())["error"] == "USAGE"

    def test_g4_blast_radius_is_stripped(self, lw: L) -> None:
        put = lambda h, r, why: ledger.set_blast_radius(  # noqa: E731
            lw.conn, lw.clock, token=lw.token, task_id="T1", value="ok", head_sha=h, review=r, reason=why
        )
        assert put(" h1 ", " G1 ", "  ")["head_sha"] == "h1"
        row = ledger.blast_radius(lw.conn, "T1", " h1 ")
        assert row is not None and (row["head_sha"], row["review"], row["reason"]) == ("h1", "G1", None)
        assert code_of(put, "h", "  ", None) == "USAGE"

    def test_g5_a_cause_of_another_run_is_not_found(self, lw: L) -> None:
        other = lw.one("T6", token=lw.token2)
        a = lw.one("T1")
        assert code_of(lw.state, a, "deferred", cause=other) == "NOT_FOUND"
        assert lw.row(a)["state"] == "open"

    def test_g6_a_blank_commit_on_another_target(self, lw: L) -> None:
        assert code_of(lw.state, lw.one("T1"), "open", commit=" ") == "USAGE"

    def test_g7_the_tasks_ci_sha_is_compared_stripped(self, lw: L) -> None:
        sql(lw.db, "UPDATE tasks SET ci_sha = ' c1 ' WHERE id = 'T1'")
        assert lw.open("T1", ci_item("c1"), source="ci")

    def test_g8_file_and_symbol_are_nfc(self, lw: L) -> None:
        a = lw.one("T1", file="docs/cafe\u0301.md", symbol="fe\u0301")
        b = lw.one("T1", file="docs/caf\u00e9.md", symbol="f\u00e9", reviewer="r2")
        assert lw.row(a)["fingerprint"] == lw.row(b)["fingerprint"] and lw.row(b)["matched_id"] == a

    def test_g9_a_blank_sha_is_absent(self, lw: L) -> None:
        assert lw.open("T1", item(sha="  "))
        assert code_of(lw.open, "T1", ci_item("  "), source="ci") == "USAGE"

    def test_g10_backticks_never_close_a_code_span(self, lw: L) -> None:
        lw.round(head="ab`cd", base="x`")
        ledger.set_blast_radius(
            lw.conn, lw.clock, token=lw.token, task_id="T1", value="ok", head_sha="h`1", review="G", reason=None
        )
        sections = dict(export_sections())
        assert "`x..abcd`" in sections["Rounds"](lw.conn, "T1")
        assert sections["Blast radius"](lw.conn, "T1") == "`ok` at `h1` (review G)"


class TestReviewFix03:
    def test_h1_a_blank_head_sha_reads_no_rating(self, lw: L) -> None:
        ledger.set_blast_radius(
            lw.conn, lw.clock, token=lw.token, task_id="T1", value="ok", head_sha="h1", review="G", reason=None
        )
        assert ledger.blast_radius(lw.conn, "T1", "  ") is None
        assert ledger.blast_radius(lw.conn, "T1", "") is None
        assert ledger.blast_radius(lw.conn, "T1") is not None

    @pytest.mark.parametrize("file", ["a.py:42 /", "a.py:10-12 /."])
    def test_h2_a_line_suffix_before_whitespace(self, lw: L, file: str) -> None:
        assert code_of(lw.open, "T1", item(file=file)) == "USAGE"

    def test_h3_a_lone_surrogate_in_a_flag(self, lw: L) -> None:
        a = lw.one("T1")
        assert code_of(lw.state, a, "rejected", reason="x\udcff") == "USAGE"
        assert code_of(lw.classify, a, "must_fix", note="\udcff") == "USAGE"
        assert code_of(lw.round, head="h\udcff") == "USAGE"
        assert lw.row(a)["state"] == "open"

    def test_h4_the_reviewer_is_nfc(self, lw: L) -> None:
        (a,) = lw.open("T1", item(reviewer="re\u0301"))
        assert lw.open("T1", item(reviewer="r\u00e9")) == [a]

    def test_h5_backticks_in_the_blast_radius_text(self, lw: L) -> None:
        ledger.set_blast_radius(
            lw.conn, lw.clock, token=lw.token, task_id="T1", value="ok", head_sha="h", review="G`1", reason="a `b"
        )
        assert dict(export_sections())["Blast radius"](lw.conn, "T1") == "`ok` at `h` (review G1) — a b"


def export_sections() -> list[Any]:
    from control_plane import export

    return list(export.LEDGER_SECTIONS)
