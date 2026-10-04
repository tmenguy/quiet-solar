"""Checkpoint 8: locks held by process groups, LOCK_ORDER, caps, session-held integration locks (AC13, AC17)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest
from control_plane import clock, db, errors, liveness, locks

from .conftest import ORCH, agent, insert_node, insert_task, open_run, run_cli, sql


def H(pid: int) -> liveness.Holder:
    return liveness.Holder(pid, f"start-{pid}", pid)


@pytest.fixture
def world(migrated: Path, fake_claude, fake_clock, fake_probe) -> dict:
    run_id, token = open_run()
    insert_task(migrated, "T1", run_id, is_deliverable=1, branch="QS_5")
    insert_task(migrated, "T2", run_id, deliverable_id="T1", item_k=1)
    insert_task(migrated, "T3", run_id, is_deliverable=1, branch="QS_9")
    n1 = insert_node(migrated, "N1", run_id, "T1", session_id="S-n1")
    n2 = insert_node(migrated, "N2", run_id, "T2", session_id="S-n2")
    fake_claude.listing = [
        agent(ORCH, pid=101, kind="interactive"),
        agent("S-n1", pid=201, started_at_ms=1_790_938_096_856),
        agent("S-n2", pid=202),
    ]
    w = {"run": run_id, "tok": token, "n1": n1, "n2": n2, "db": migrated}
    w["clock"], w["probe"], w["claude"] = fake_clock, fake_probe, fake_claude
    return w


def hold(
    w: dict, names: list[str], token: str, holder: liveness.Holder, **kw: Any
) -> AbstractContextManager[locks.Held]:
    kw.setdefault("timeout", 3)
    kw.setdefault("cap_timeout", 3)
    return locks.hold(
        names,
        conn_factory=lambda: db.connect(w["db"]),
        token=token,
        actor="tool",
        purpose="p",
        clock=w["clock"],
        probe=w["probe"],
        claude=w["claude"],
        holder=holder,
        poll=1,
        **kw,
    )


def acquire(w: dict, token: str, sid: str, name: str = "integration:QS_5", *extra: str) -> tuple[int, Any]:
    return run_cli(
        "lock", "acquire", "--name", name, "--purpose", "integrate", "--token", token, "--session-id", sid, *extra
    )


def lock_row(w: dict, name: str = "integration:QS_5") -> dict | None:
    rows = sql(w["db"], "SELECT * FROM locks WHERE name = ?", [name])
    return dict(rows[0]) if rows else None


class TestOrder:
    def test_order(self) -> None:
        locks.assert_sorted(["integration:QS_1", "integration:QS_2", "main-merge", "main-checkout"])
        locks.assert_sorted([])
        for bad in (["main-checkout", "main-merge"], ["main-merge", "integration:QS_1"], ["main-merge", "main-merge"]):
            with pytest.raises(errors.CpError) as exc:
                locks.assert_sorted(bad)
            assert exc.value.code == "INTERNAL"
        for unknown in ("foo", "integration:"):
            with pytest.raises(errors.CpError) as exc:
                locks.order_key(unknown)
            assert exc.value.code == "USAGE"
        assert locks.LOCK_ORDER == ("integration:*", "main-merge", "main-checkout")

    def test_cap_env(self, monkeypatch) -> None:
        assert (locks.max_gates(), locks.max_nodes()) == (2, 4)
        monkeypatch.setenv("QS_CP_MAX_GATES", "3")
        monkeypatch.setenv("QS_CP_MAX_NODES", "x")
        assert (locks.max_gates(), locks.max_nodes()) == (3, 4)
        monkeypatch.setenv("QS_CP_MAX_GATES", "0")
        assert locks.max_gates() == 2


class TestProcessLocks:
    def test_three_names_are_independent(self, world) -> None:
        with (
            hold(world, ["integration:QS_5"], world["tok"], H(1)),
            hold(world, ["main-merge"], world["tok"], H(2)),
            hold(world, ["main-checkout"], world["tok"], H(3)),
        ):
            rows = sql(world["db"], "SELECT name, holder_pid, holder_kind, token_subject FROM locks ORDER BY name")
            assert [tuple(r) for r in rows] == [
                ("integration:QS_5", 1, "process", "run:R1"),
                ("main-checkout", 3, "process", "run:R1"),
                ("main-merge", 2, "process", "run:R1"),
            ]
        assert sql(world["db"], "SELECT count(*) FROM locks")[0][0] == 0

    def test_live_second_holder_waits_then_busy(self, world) -> None:
        with hold(world, ["main-checkout"], world["tok"], H(1)):
            with pytest.raises(errors.CpError) as exc, hold(world, ["main-checkout"], world["tok"], H(2)):
                pass
        assert exc.value.code == "BUSY" and exc.value.extra["holder_actor"] == "tool"
        assert exc.value.extra["purpose"] == "p" and world["clock"].sleeps == [1, 1, 1]

    def test_dead_holder_is_taken_over_but_a_live_group_keeps_it(self, world) -> None:
        sql(
            world["db"],
            "INSERT INTO locks (name, holder_kind, holder_pid, holder_pid_start, holder_pgid, holder_actor, token_subject,"
            " acquired_at) VALUES ('main-merge', 'process', 7, 'start-7', 7, 'old', 'run:R1', 'x')",
        )
        world["probe"].kill(7, group=False)  # orphaned children still run in the group
        with pytest.raises(errors.CpError) as exc, hold(world, ["main-merge"], world["tok"], H(2)):
            pass
        assert exc.value.code == "BUSY"
        world["probe"].dead_groups.add(7)
        with hold(world, ["main-merge"], world["tok"], H(2)):
            assert lock_row(world, "main-merge")["holder_pid"] == 2

    def test_stale_token_takes_nothing(self, world) -> None:
        run_cli("run", "claim", world["run"], "--session-id", ORCH)
        with pytest.raises(errors.CpError) as exc, hold(world, ["main-checkout"], world["tok"], H(1)):
            pass
        assert exc.value.code == "STALE_TOKEN"
        assert sql(world["db"], "SELECT count(*) FROM locks")[0][0] == 0

    def test_unsorted_raises_before_anything(self, world) -> None:
        with pytest.raises(errors.CpError) as exc, hold(world, ["main-checkout", "main-merge"], world["tok"], H(1)):
            pass
        assert exc.value.code == "INTERNAL"

    def test_release_frees_only_what_is_still_ours(self, world) -> None:
        with hold(world, ["main-checkout"], world["tok"], H(1)):
            sql(world["db"], "UPDATE locks SET holder_pid = 99")
        assert lock_row(world, "main-checkout")["holder_pid"] == 99

    def test_recheck(self, world) -> None:
        with hold(world, ["main-checkout"], world["tok"], H(1)) as held:
            c = db.connect(world["db"])
            locks.recheck(c, held)
            sql(world["db"], "UPDATE locks SET holder_pid = 99")
            with pytest.raises(errors.CpError) as exc:
                locks.recheck(c, held)
            assert exc.value.code == "STALE_TOKEN" and "no longer held" in exc.value.detail
            c.close()

    def test_token_going_stale_mid_operation(self, world) -> None:
        with hold(world, ["main-checkout"], world["tok"], H(1)) as held:
            run_cli("run", "claim", world["run"], "--session-id", ORCH)
            c = db.connect(world["db"])
            with pytest.raises(errors.CpError) as exc:
                locks.recheck(c, held)
            c.close()
        assert exc.value.code == "STALE_TOKEN"

    def test_row_changed_after_probe_is_reprobed(self, world, monkeypatch) -> None:
        sql(
            world["db"],
            "INSERT INTO locks (name, holder_kind, holder_pid, holder_pid_start, holder_pgid, holder_actor, token_subject,"
            " acquired_at) VALUES ('main-merge', 'process', 7, 'start-7', 7, 'old', 'run:R1', 'x')",
        )
        world["probe"].kill(7)
        calls = {"n": 0}
        real = world["probe"].holder_alive

        def racing(pid: int | None, start: str | None, pgid: int | None) -> bool:
            calls["n"] += 1
            if calls["n"] == 1:
                sql(world["db"], "UPDATE locks SET acquired_at = 'y'")
            return real(pid, start, pgid)

        monkeypatch.setattr(world["probe"], "holder_alive", racing)
        with hold(world, ["main-merge"], world["tok"], H(2)):
            assert calls["n"] == 2 and world["clock"].sleeps == []


class TestGateCap:
    def test_at_most_max_gates(self, world, monkeypatch) -> None:
        monkeypatch.setenv("QS_CP_MAX_GATES", "2")
        with hold(world, [], world["tok"], H(1), cap="gates"), hold(world, [], world["tok"], H(2), cap="gates") as two:
            assert two.slots == [("gates", 1)]
            with pytest.raises(errors.CpError) as exc, hold(world, [], world["tok"], H(3), cap="gates"):
                pass
            assert exc.value.code == "BUSY"
            world["probe"].kill(1)
            with hold(world, [], world["tok"], H(3), cap="gates") as three:
                assert three.slots == [("gates", 0)]
                c = db.connect(world["db"])
                locks.recheck(c, three)
                sql(world["db"], "UPDATE cap_slots SET holder_pid = 42 WHERE slot = 0")
                with pytest.raises(errors.CpError):
                    locks.recheck(c, three)
                c.close()

    def test_stale_token_takes_no_slot(self, world) -> None:
        run_cli("run", "claim", world["run"], "--session-id", ORCH)
        with pytest.raises(errors.CpError) as exc, hold(world, [], world["tok"], H(1), cap="gates"):
            pass
        assert exc.value.code == "STALE_TOKEN" and sql(world["db"], "SELECT count(*) FROM cap_slots")[0][0] == 0


class TestSessionLocks:
    def test_only_integration_names(self, world) -> None:
        for name in ("main-merge", "integration:"):
            code, out = acquire(world, world["n1"], "S-n1", name)
            assert code == 9 and out["error"] == "POLICY_REFUSED"

    def test_liveness_and_binding(self, world) -> None:
        world["claude"].listing = None
        assert acquire(world, world["n1"], "S-n1")[1]["error"] == "BUSY"
        world["claude"].listing = [agent("S-n1", pid=201), agent(ORCH, pid=101)]
        assert acquire(world, world["n1"], "S-zz")[1]["error"] == "USAGE"
        world["probe"].kill(201)
        assert acquire(world, world["n1"], "S-n1")[1]["error"] == "USAGE"
        world["probe"].dead_pids.clear()
        assert acquire(world, world["tok"], "S-n1")[1]["error"] == "CONFLICT"  # not the run token's session
        assert acquire(world, world["n1"], "S-n1", "integration:QS_9")[1]["error"] == "CONFLICT"  # other deliverable
        assert acquire(world, world["tok"], ORCH, "integration:QS_77")[1]["error"] == "CONFLICT"
        assert acquire(world, world["tok"], ORCH, "integration:QS_9")[1]["status"] == "acquired"

    def test_acquire_records_and_reacquire_restamps(self, world) -> None:
        code, out = acquire(world, world["n1"], "S-n1")
        assert code == 0 and out["status"] == "acquired"
        row = lock_row(world)
        assert (row["holder_kind"], row["holder_session_id"], row["holder_pid"], row["holder_epoch"]) == (
            "session",
            "S-n1",
            201,
            1,
        )
        assert row["holder_pid_start"] == "start-201"  # from ProcessProbe.start_of, never the listing's startedAt
        assert row["token_subject"] == "node:N1" and row["holder_actor"] == "node:T1"
        world["claude"].listing = [agent("S-n1", pid=301), agent("S-n2", pid=202)]  # app restart: new pid
        assert acquire(world, world["n2"], "S-n2")[1]["error"] == "BUSY"  # still held: listed
        assert acquire(world, world["n1"], "S-n1")[1]["status"] == "already_held"
        row = lock_row(world)
        assert (row["holder_pid"], row["holder_pid_start"]) == (301, "start-301")

    def test_busy_waits_for_the_timeout(self, world) -> None:
        acquire(world, world["n1"], "S-n1")
        code, out = acquire(world, world["n2"], "S-n2", "integration:QS_5", "--timeout", "2")
        assert code == 6 and out["holder_session_id"] == "S-n1" and world["clock"].sleeps == [1.0, 1.0]

    @pytest.mark.parametrize("how", ["stopped", "superseded", "absent", "run-closed", "deliverable-terminal"])
    def test_freed_when_the_holder_is_dead(self, world, how: str) -> None:
        acquire(world, world["n1"], "S-n1")
        contender = (world["tok"], ORCH)
        if how == "stopped":
            sql(world["db"], "UPDATE nodes SET state = 'stopped' WHERE id = 'N1'")
        elif how == "superseded":
            sql(world["db"], "UPDATE nodes SET state = 'superseded' WHERE id = 'N1'")
        elif how == "absent":
            world["claude"].listing = [a for a in world["claude"].listing if a.session_id != "S-n1"]
        elif how == "run-closed":
            sql(world["db"], "UPDATE runs SET state = 'closed'")
        else:
            sql(world["db"], "UPDATE tasks SET state = 'merged' WHERE id = 'T1'")
        code, out = acquire(world, *contender)
        assert code == 0 and out["status"] == "acquired"
        assert lock_row(world)["holder_session_id"] == ORCH

    def test_failed_listing_falls_back_to_the_pid(self, world) -> None:
        acquire(world, world["n1"], "S-n1")
        world["claude"].listing = None
        with pytest.raises(errors.CpError) as exc, hold(world, ["integration:QS_5", "main-merge"], world["tok"], H(5)):
            pass
        assert exc.value.code == "BUSY"  # the recorded pid is alive
        world["probe"].kill(201)
        with hold(world, ["integration:QS_5", "main-merge"], world["tok"], H(5)):
            assert lock_row(world)["holder_kind"] == "process"

    def test_takeover_by_another_session_frees_but_a_same_session_reclaim_carries(self, world) -> None:
        assert acquire(world, world["tok"], ORCH)[1]["status"] == "acquired"
        new = run_cli("run", "claim", world["run"], "--session-id", ORCH)[1]["token"]  # same session: carries
        assert acquire(world, world["n1"], "S-n1")[1]["error"] == "BUSY"
        assert acquire(world, new, ORCH)[1]["status"] == "already_held"
        assert lock_row(world)["holder_epoch"] == 2
        world["claude"].listing.append(agent("S-new", pid=401))
        run_cli("run", "claim", world["run"], "--session-id", "S-new", "--takeover")
        assert acquire(world, world["n1"], "S-n1")[1]["status"] == "acquired"

    def test_co_holder_keeps_it_through_the_session_dying(self, world) -> None:
        acquire(world, world["n1"], "S-n1")
        with hold(world, ["integration:QS_5"], world["n1"], H(11)) as tool:
            assert tool.coheld == ["integration:QS_5"] and tool.locks == []
            row = lock_row(world)
            assert (row["cohold_pid"], row["cohold_pgid"], row["holder_kind"]) == (11, 11, "session")
            c = db.connect(world["db"])
            locks.recheck(c, tool)
            world["claude"].listing = [agent(ORCH, pid=101)]  # the session died mid-step
            with (
                pytest.raises(errors.CpError) as exc,
                hold(world, ["integration:QS_5", "main-merge"], world["tok"], H(12), timeout=2),
            ):
                pass
            assert exc.value.code == "BUSY"  # child 7's merge waits
            world["probe"].kill(11)  # the tool's group exits
            with hold(world, ["integration:QS_5", "main-merge"], world["tok"], H(12)):
                assert lock_row(world)["holder_pid"] == 12
            with pytest.raises(errors.CpError) as exc:
                locks.recheck(c, tool)
            assert exc.value.code == "STALE_TOKEN"
            c.close()

    def test_second_co_holder_waits_and_each_clears_its_own_stamp(self, world) -> None:
        acquire(world, world["n1"], "S-n1")
        with hold(world, ["integration:QS_5"], world["n1"], H(11)):
            with pytest.raises(errors.CpError) as exc, hold(world, ["integration:QS_5"], world["n1"], H(13), timeout=1):
                pass
            assert exc.value.code == "BUSY"
            assert lock_row(world)["cohold_pid"] == 11
        assert lock_row(world)["cohold_pid"] is None
        with hold(world, ["integration:QS_5"], world["n1"], H(13)):
            assert lock_row(world)["cohold_pid"] == 13
            sql(world["db"], "UPDATE locks SET cohold_pid = 14, cohold_pgid = 14")  # someone else's stamp
        assert lock_row(world)["cohold_pid"] == 14  # 13 cleared only its own stamp

    def test_co_hold_restamps_the_holder_pid(self, world) -> None:
        acquire(world, world["n1"], "S-n1")
        world["claude"].listing = [agent("S-n1", pid=555)]
        with hold(world, ["integration:QS_5"], world["n1"], H(11)):
            assert (lock_row(world)["holder_pid"], lock_row(world)["holder_pid_start"]) == (555, "start-555")

    def test_co_hold_without_a_listing_keeps_the_recorded_pid(self, world) -> None:
        acquire(world, world["n1"], "S-n1")
        world["claude"].listing = None
        with hold(world, ["integration:QS_5"], world["n1"], H(11)):
            assert lock_row(world)["holder_pid"] == 201

    def test_lock_order_across_commands(self, world) -> None:
        assert acquire(world, world["tok"], ORCH, "integration:QS_9")[1]["status"] == "acquired"
        code, out = acquire(world, world["tok"], ORCH, "integration:QS_5")
        assert out["error"] == "CONFLICT" and "integration:QS_9" in out["detail"]
        with pytest.raises(errors.CpError) as exc, hold(world, ["integration:QS_5"], world["tok"], H(1)):
            pass
        assert exc.value.code == "CONFLICT"
        with hold(world, ["integration:QS_9", "main-merge"], world["tok"], H(1)) as merge:
            assert merge.coheld == ["integration:QS_9"] and merge.locks == ["main-merge"]

    def test_release(self, world) -> None:
        acquire(world, world["n1"], "S-n1")
        rel = ["lock", "release", "--name", "integration:QS_5"]
        assert run_cli(*rel, "--token", world["n2"], "--session-id", "S-n2")[1]["error"] == "CONFLICT"
        assert run_cli(*rel, "--token", world["n1"], "--session-id", "S-x")[1]["error"] == "CONFLICT"
        assert run_cli(*rel, "--token", world["n1"], "--session-id", "S-n1")[1]["released"] is True
        assert run_cli(*rel, "--token", world["n1"], "--session-id", "S-n1")[1]["released"] is False
        with hold(world, ["main-merge"], world["tok"], H(1)):
            out = run_cli("lock", "release", "--name", "main-merge", "--token", world["tok"], "--session-id", ORCH)[1]
            assert out["error"] == "CONFLICT"  # a process-held lock is not the session's


def _call(path: Path, key: str, pid: int, state: str = "started") -> None:
    sql(
        path,
        "INSERT INTO tool_calls (tool, key, args_hash, actor, args, state, holder_pid, holder_pid_start, holder_pgid,"
        " started_at) VALUES ('spawn', ?, 'h', 'a', '{}', ?, ?, ?, ?, 'x')",
        [key, state, pid, f"start-{pid}", pid],
    )


class TestNodeCap:
    @pytest.fixture
    def cap(self, migrated: Path, fake_probe, fake_clock) -> Iterator[dict]:
        run_id, _ = open_run()
        for i in range(1, 7):
            insert_task(migrated, f"T{i}", run_id)
        c = db.connect(migrated)
        yield {"db": migrated, "run": run_id, "probe": fake_probe, "clock": fake_clock, "conn": c}
        c.close()

    def _admit(self, cap: dict, listing: list[liveness.Agent] | None, limit: int = 4) -> int:
        alive = locks.spawn_holders_alive(cap["conn"], cap["probe"])
        with db.write(cap["conn"]):
            return locks.admit_node(cap["conn"], cap["clock"], listing=listing, holders_alive=alive, limit=limit)

    def test_spawning_rows(self, cap) -> None:
        insert_node(cap["db"], "N1", cap["run"], "T1", state="spawning", spawn_tool_key="spawn/k1")
        _call(cap["db"], "k1", 7)
        assert self._admit(cap, []) == 1  # holder alive: counts
        with pytest.raises(errors.CpError) as exc:
            self._admit(cap, [], limit=1)
        assert exc.value.code == "BUSY"
        cap["probe"].kill(7)
        assert self._admit(cap, [agent("x", name="R1-T1-g1")]) == 1  # listed: counts
        assert self._admit(cap, None) == 1  # unknown: counts
        sql(cap["db"], "UPDATE nodes SET launch_at = ?", [clock.stamp(cap["clock"])])
        cap["clock"].advance(30)
        assert self._admit(cap, []) == 1  # within LAUNCH_SETTLE_S: counts
        cap["clock"].advance(31)
        assert self._admit(cap, []) == 0  # dead, unlisted, settled: reaped
        assert sql(cap["db"], "SELECT state FROM nodes")[0][0] == "reaped"

    def test_running_rows_count_when_listed_or_unknown(self, cap) -> None:
        insert_node(cap["db"], "N1", cap["run"], "T1", state="running", session_id="S1")
        insert_node(cap["db"], "N2", cap["run"], "T2", state="idle", session_id="S2")
        insert_node(cap["db"], "N3", cap["run"], "T3", state="taken_over", session_id="S3")
        insert_node(cap["db"], "N4", cap["run"], "T4", state="reaped")
        insert_node(cap["db"], "N5", cap["run"], "T5", state="stopped")
        assert self._admit(cap, [agent("S1"), agent("S3")]) == 2
        assert self._admit(cap, None) == 3
        assert self._admit(cap, []) == 0


def test_session_acquire_reprobes_a_row_changed_after_the_probe(world, monkeypatch) -> None:
    sql(
        world["db"],
        "INSERT INTO locks (name, holder_kind, holder_pid, holder_pid_start, holder_pgid, holder_actor, token_subject,"
        " acquired_at) VALUES ('integration:QS_5', 'process', 7, 'start-7', 7, 'old', 'run:R1', 'x')",
    )
    world["probe"].kill(7)
    calls = {"n": 0}
    real = world["probe"].holder_alive

    def racing(pid: int | None, start: str | None, pgid: int | None) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            sql(world["db"], "UPDATE locks SET acquired_at = 'y'")
        return real(pid, start, pgid)

    monkeypatch.setattr(world["probe"], "holder_alive", racing)
    assert acquire(world, world["n1"], "S-n1")[1]["status"] == "acquired"
    assert calls["n"] == 2
