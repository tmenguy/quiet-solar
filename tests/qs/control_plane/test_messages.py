"""Checkpoint 6: queues, receipts, dead letters, concurrent pop (AC5)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from control_plane import clock, db, errors, faults, messages

from .conftest import insert_node, insert_task, open_run, run_cli, sql


@pytest.fixture
def world(migrated: Path, tmp_path: Path) -> dict:
    run_id, token = open_run()
    insert_task(migrated, "T1", run_id)
    insert_task(migrated, "T2", run_id)
    node = insert_node(migrated, "N1", run_id, "T1")
    payload = tmp_path / "p.json"
    payload.write_text(json.dumps({"hello": "world"}))
    return {"run": run_id, "token": token, "node": node, "payload": str(payload), "db": migrated}


def post(w: dict, to: str = "orchestrator", token: str | None = None, **kw: str) -> tuple[int, dict]:
    argv = ["msg", "post", "--run", w["run"], "--to", to, "--kind", kw.pop("kind", "note")]
    argv += ["--payload-file", kw.pop("payload", w["payload"]), "--token", token or w["token"]]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", v]
    return run_cli(*argv)


def pop(w: dict, as_: str = "orchestrator", token: str | None = None, *extra: str) -> tuple[int, dict]:
    return run_cli("msg", "pop", "--run", w["run"], "--as", as_, "--token", token or w["token"], *extra)


def ack(w: dict, msg_id: int, receipt: str, token: str | None = None) -> tuple[int, dict]:
    return run_cli("msg", "ack", str(msg_id), "--receipt", receipt, "--token", token or w["token"])


class TestPost:
    def test_post_and_dedupe(self, world) -> None:
        code, out = post(world)
        assert code == 0 and out["deduped"] is False
        row = dict(sql(world["db"], "SELECT * FROM messages WHERE id = ?", [out["id"]])[0])
        assert (row["sender"], row["state"], row["payload"]) == ("orchestrator", "queued", '{"hello":"world"}')
        a = post(world, dedupe_key="k1")[1]
        b = post(world, dedupe_key="k1")[1]
        assert b == {"ok": True, "id": a["id"], "deduped": True}

    def test_node_posts_to_the_orchestrator_only(self, world) -> None:
        code, out = post(world, token=world["node"])
        assert code == 0
        assert sql(world["db"], "SELECT sender FROM messages WHERE id = ?", [out["id"]])[0][0] == "node:T1"
        assert post(world, to="node:T2", token=world["node"])[1]["error"] == "CONFLICT"

    def test_validation(self, world, tmp_path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{nope")
        assert post(world, payload=str(bad))[1]["error"] == "USAGE"
        assert post(world, kind="x" * 33)[1]["error"] == "USAGE"
        assert post(world, to="nobody")[1]["error"] == "USAGE"
        assert post(world, to="node:T9")[1]["error"] == "NOT_FOUND"
        other, other_token = open_run("r2", "S-2")
        assert post(world, token=other_token)[1]["error"] == "CONFLICT"
        run_cli("run", "close", "--token", world["token"])
        assert post(world)[1]["error"] == "INVALID_STATE"


class TestPopAck:
    def test_order_of_arrival_per_recipient(self, world) -> None:
        ids = [post(world)[1]["id"], post(world, to="node:T1")[1]["id"], post(world)[1]["id"]]
        assert pop(world)[1]["id"] == ids[0]
        assert pop(world)[1]["id"] == ids[2]
        out = pop(world, "node:T1", world["node"])[1]
        assert out["id"] == ids[1] and out["payload"] == {"hello": "world"}
        assert pop(world)[1] == {"ok": True, "empty": True}

    def test_pop_sets_in_flight_with_visibility_and_receipt(self, world, fake_clock) -> None:
        mid = post(world)[1]["id"]
        out = pop(world, "orchestrator", None, "--visibility", "60")[1]
        assert out["attempt"] == 1 and len(out["receipt"]) == 16 and out["sender"] == "orchestrator"
        row = dict(sql(world["db"], "SELECT * FROM messages WHERE id = ?", [mid])[0])
        assert row["state"] == "in_flight" and row["visible_at"] == clock.stamp(fake_clock, plus=60)
        assert row["popped_by"] == "orchestrator"

    def test_expiry_redelivers_first_and_receipts_fence_acks(self, world, fake_clock) -> None:
        m1 = post(world)[1]["id"]
        m2 = post(world)[1]["id"]
        first = pop(world)[1]
        assert first["id"] == m1
        assert pop(world)[1]["id"] == m2  # m1 is invisible meanwhile
        fake_clock.advance(messages.VISIBILITY_S + 1)
        again = pop(world)[1]
        assert again["id"] == m1 and again["attempt"] == 2 and again["receipt"] != first["receipt"]
        assert ack(world, m1, first["receipt"])[1]["error"] == "CONFLICT"
        assert ack(world, m1, again["receipt"])[1] == {"ok": True, "id": m1, "acked": True, "noop": False}
        assert ack(world, m1, again["receipt"])[1]["noop"] is True
        assert ack(world, m1, first["receipt"])[1]["error"] == "CONFLICT"
        fake_clock.advance(messages.VISIBILITY_S + 1)
        assert pop(world)[1]["id"] == m2  # the acked m1 never returns

    def test_poison_message_goes_dead(self, world, fake_clock) -> None:
        m1 = post(world)[1]["id"]
        m2 = post(world)[1]["id"]
        for attempt in range(1, 6):
            out = pop(world)[1]
            assert (out["id"], out["attempt"]) == (m1, attempt)  # m1 keeps blocking m2 ...
            fake_clock.advance(messages.VISIBILITY_S + 1)
        out = pop(world)[1]
        assert out["id"] == m2  # the 6th attempt marked m1 dead and moved on
        assert sql(world["db"], "SELECT state FROM messages WHERE id = ?", [m1])[0][0] == "dead"
        assert ack(world, m1, "x")[1]["error"] == "INVALID_STATE"

    def test_ack_errors(self, world) -> None:
        mid = post(world)[1]["id"]
        assert ack(world, mid, "x")[1]["error"] == "INVALID_STATE"  # still queued
        assert ack(world, 999, "x")[1]["error"] == "NOT_FOUND"

    def test_readers_are_fenced(self, world) -> None:
        post(world, to="node:T1")
        assert pop(world, "orchestrator", world["node"])[1]["error"] == "CONFLICT"
        assert pop(world, "node:T1")[1]["error"] == "CONFLICT"
        assert pop(world, "node:T2", world["node"])[1]["error"] == "CONFLICT"

    def test_stopped_node_still_pops_and_acks_its_own_queue(self, world) -> None:
        post(world, to="node:T1")
        sql(world["db"], "UPDATE nodes SET state = 'stopped'")
        out = pop(world, "node:T1", world["node"])[1]
        assert out["empty"] is False
        assert ack(world, out["id"], out["receipt"], world["node"])[0] == 0
        assert post(world, token=world["node"])[0] == 4  # but cannot post

    def test_fault_after_select_rolls_back(self, world) -> None:
        mid = post(world)[1]["id"]
        with faults.arm("pop.after_select"), pytest.raises(faults.FaultInjected):
            pop(world)
        assert sql(world["db"], "SELECT state FROM messages WHERE id = ?", [mid])[0][0] == "queued"


def test_concurrent_pop(world) -> None:
    c = db.connect(world["db"])
    fc = clock.FakeClock()
    with db.write(c):
        ids = [
            messages.post_locked(c, fc, run_id=world["run"], recipient="orchestrator", kind="k", sender="x", payload=i)[
                "id"
            ]
            for i in range(200)
        ]
    c.close()
    barrier = threading.Barrier(8)
    seen: list[int] = []
    lock = threading.Lock()
    failures: list[BaseException] = []

    def worker() -> None:
        conn = db.connect(world["db"])
        try:
            barrier.wait(timeout=2)
            while True:
                out = messages.pop(conn, fc, token=world["token"], run_ref=world["run"], recipient="orchestrator")
                if out["empty"]:
                    return
                messages.ack(conn, fc, token=world["token"], msg_id=out["id"], receipt=out["receipt"])
                with lock:
                    seen.append(out["id"])
        except BaseException as exc:  # noqa: BLE001 — surfaced below
            failures.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not failures
    assert sorted(seen) == sorted(ids)


def test_visible_helpers(world, fake_clock) -> None:
    c = db.connect(world["db"])
    now = clock.stamp(fake_clock)
    assert messages.head_id(c, world["run"], "orchestrator", now) is None
    mid = post(world)[1]["id"]
    assert messages.head_id(c, world["run"], "orchestrator", now) == mid
    assert messages.visible_count(c, world["run"], "orchestrator", now) == 1
    c.close()
    with pytest.raises(errors.CpError):
        messages.parse_payload("{")
