"""#407 T9: the review pilot — rules, the queue, the launcher, the retry, the closing of dead reviews (D10–D12)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from control_plane import activeloop, alerts, daemon, db, errors, reviewpilot, reviews, schema_reviews

from .conftest import FakeProc, agent, insert_task, open_run, run_cli, sql


@pytest.fixture
def pw(migrated: Path, fake_clock, fake_popen, fake_claude, fake_probe, fake_main: Path, tmp_path: Path):
    """R1: deliverables T1, T2; R2: deliverable T5; R3: deliverable T7."""
    r1, tok1 = open_run()
    r2, tok2 = open_run("r2", "S-2")
    r3, _ = open_run("r3", "S-3")
    insert_task(migrated, "T1", r1, is_deliverable=1, branch="QS_1")
    insert_task(migrated, "T2", r1, is_deliverable=1, branch="QS_2")
    insert_task(migrated, "T5", r2, is_deliverable=1, branch="QS_5")
    insert_task(migrated, "T7", r3, is_deliverable=1, branch="QS_7")
    fake_popen.calls.clear()  # `run open` started the (fake) daemon
    fake_popen.procs.clear()
    conn = db.connect(migrated)
    try:
        yield {
            "db": migrated,
            "conn": conn,
            "clock": fake_clock,
            "popen": fake_popen,
            "claude": fake_claude,
            "probe": fake_probe,
            "main": fake_main,
            "tmp": tmp_path,
            "tok1": tok1,
            "tok2": tok2,
        }
    finally:
        conn.close()


def tick(pw: dict[str, Any], seconds: float = reviewpilot.PILOT_EVERY_S) -> None:
    pw["clock"].advance(seconds)
    reviewpilot.review_pilot_hook(pw["conn"], pw["clock"])


def launcher_alive(pw: dict[str, Any], rid: str = "V1") -> None:
    """What the detached launcher writes first: its ``tool_calls`` row, held by a live process."""
    tool_key = sql(pw["db"], "SELECT tool_key FROM reviews WHERE id = ?", [rid])[0][0]
    sql(
        pw["db"],
        "INSERT INTO tool_calls (tool, key, args_hash, actor, args, state, holder_pid, holder_pid_start, holder_pgid,"
        " started_at) VALUES ('review', ?, 'h', ?, '{}', 'started', 5555, 'start-5555', 5555, 'x')",
        [tool_key[len("review/") :], f"review:{rid}"],
    )


def alert(pw: dict[str, Any], run: str, kind: str, tasks: list[str], *, cleared: bool = False) -> int:
    subject = "|".join(sorted(tasks))
    n = (
        1
        + sql(
            pw["db"], "SELECT count(*) FROM alerts WHERE run_id = ? AND kind = ? AND subject = ?", [run, kind, subject]
        )[0][0]
    )
    sql(
        pw["db"],
        "INSERT INTO alerts (run_id, kind, subject, fingerprint, payload, first_seen, last_seen, cleared_at)"
        " VALUES (?, ?, ?, ?, ?, 'x', 'x', ?)",
        [run, kind, subject, f"{kind}:{subject}:{n}", json.dumps({"tasks": sorted(tasks)}), "x" if cleared else None],
    )
    return int(sql(pw["db"], "SELECT max(id) FROM alerts")[0][0])


def cross(pw: dict[str, Any], kind: str = alerts.OVERLAP_CROSS_RUN, tasks: tuple[str, ...] = ("T1", "T5")) -> list[int]:
    """One cross-run occurrence: a row in each run of ``tasks``."""
    runs = sorted({sql(pw["db"], "SELECT run_id FROM tasks WHERE id = ?", [t])[0][0] for t in tasks})
    return [alert(pw, run, kind, list(tasks)) for run in runs]


def event_key(kind: str, tasks: tuple[str, ...], first_id: int) -> str:
    return f"alert:{kind}:{hashlib.sha256('|'.join(sorted(tasks)).encode()).hexdigest()[:16]}:{first_id}"


def triggers(pw: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [tuple(r) for r in sql(pw["db"], "SELECT tool_key, outcome, review_id FROM review_triggers ORDER BY id")]


def review_states(pw: dict[str, Any]) -> dict[str, str]:
    return dict(sql(pw["db"], "SELECT id, state FROM reviews ORDER BY id"))


def failed_alerts(pw: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        tuple(r) for r in sql(pw["db"], "SELECT run_id, subject FROM alerts WHERE kind = 'review_failed' ORDER BY id")
    ]


# --------------------------------------------------------------------------- the rules (D11)


class TestRules:
    @pytest.mark.parametrize("kind", sorted(reviewpilot.PILOT_KINDS))
    def test_each_alert_rule_fires_once(self, pw, kind: str) -> None:
        ids = cross(pw, kind)
        tick(pw)
        key = f"pilot:{event_key(kind, ('T1', 'T5'), min(ids))}:1"
        assert triggers(pw) == [(key, "created", "V1")]
        row = dict(sql(pw["db"], "SELECT * FROM reviews WHERE id = 'V1'")[0])
        assert (row["kind"], row["state"], row["tool_key"], row["requested_by"]) == (
            "cross_run",
            "requested",
            f"review/{key}",
            "cp:review_pilot",
        )
        stored = json.loads(sql(pw["db"], 'SELECT "trigger" FROM review_triggers')[0][0])
        assert stored == {
            "event": key[len("pilot:") : -2],
            "source": "alert",
            "alert_kind": kind,
            "subject": "T1|T5",
            "tasks": ["T1", "T5"],
            "alert_ids": sorted(ids),
        }
        assert len(pw["popen"].calls) == 1
        launcher_alive(pw)
        for _ in range(3):
            tick(pw)
        assert len(triggers(pw)) == 1 and len(pw["popen"].calls) == 1

    def test_the_cycle_kind_does_not_fire(self, pw) -> None:
        cross(pw, alerts.DEPENDENCY_CYCLE_CROSS_RUN)
        tick(pw)
        assert triggers(pw) == [] and pw["popen"].calls == []

    def test_a_partial_clear_keeps_the_event_and_a_recurrence_is_a_new_one(self, pw) -> None:
        a, b = cross(pw)
        tick(pw)
        sql(pw["db"], "UPDATE alerts SET cleared_at = 'x' WHERE id = ?", [a])
        tick(pw)
        assert len(triggers(pw)) == 1
        sql(pw["db"], "UPDATE alerts SET cleared_at = 'x' WHERE id = ?", [b])
        sql(pw["db"], "UPDATE reviews SET state = 'finished'")
        tick(pw)
        c, d = cross(pw)  # a recurrence: new rows
        tick(pw)
        assert [t[0] for t in triggers(pw)] == [
            f"pilot:{event_key(alerts.OVERLAP_CROSS_RUN, ('T1', 'T5'), a)}:1",
            f"pilot:{event_key(alerts.OVERLAP_CROSS_RUN, ('T1', 'T5'), c)}:1",
        ]
        assert d > c

    def test_a_row_raised_later_joins_the_events_identity(self, pw) -> None:
        a = alert(pw, "R1", alerts.OVERLAP_CROSS_RUN, ["T1", "T5"])
        tick(pw)
        launcher_alive(pw)
        b = alert(pw, "R2", alerts.OVERLAP_CROSS_RUN, ["T1", "T5"])
        tick(pw)
        sql(pw["db"], "UPDATE alerts SET cleared_at = 'x' WHERE id = ?", [a])
        tick(pw)
        assert len(triggers(pw)) == 1
        assert json.loads(sql(pw["db"], 'SELECT "trigger" FROM review_triggers')[0][0])["alert_ids"] == [a, b]

    def test_a_skipped_event_is_never_re_requested(self, pw) -> None:
        sql(pw["db"], "UPDATE runs SET state = 'closed' WHERE id = 'R2'")
        cross(pw)
        tick(pw)
        assert [t[1:] for t in triggers(pw)] == [("skipped", None)]
        sql(pw["db"], "UPDATE runs SET state = 'open' WHERE id = 'R2'")
        tick(pw)
        assert [t[1:] for t in triggers(pw)] == [("skipped", None)] and pw["popen"].calls == []

    def test_a_queued_alert_event_whose_alerts_cleared_is_skipped(self, pw) -> None:
        cross(pw)
        sql(
            pw["db"],
            "INSERT INTO reviews (id, kind, coalesce_key, state, agent, name, requested_by, heads_at,"
            " created_at, updated_at) VALUES ('V9', 'cross_run', 'cross_run:T1|T5', 'running', 'a', 'n', 'x', 'x',"
            " ?, ?)",
            [db.now(pw["clock"]), db.now(pw["clock"])],
        )
        pw["claude"].listing = None  # liveness unknown: the running V9 is kept
        tick(pw)
        assert [t[1] for t in triggers(pw)] == ["queued"]  # deferred behind V9
        sql(pw["db"], "UPDATE alerts SET cleared_at = 'x'")
        tick(pw)
        assert [t[1] for t in triggers(pw)] == ["skipped"]  # C4-3: the collision is gone


class TestQueue:
    def test_join_then_defer_then_launch_after_the_close(self, pw) -> None:
        cross(pw, alerts.OVERLAP_CROSS_RUN)
        tick(pw)
        cross(pw, alerts.DUPLICATE_TASK_CROSS_RUN)  # the same tasks, while V1's heads are not captured: joined
        tick(pw)
        assert [t[1:] for t in triggers(pw)] == [("created", "V1"), ("joined", "V1")]
        sql(pw["db"], "UPDATE reviews SET state = 'running', heads_at = 'x', launch_at = ?", [db.now(pw["clock"])])
        pw["claude"].listing = [agent("S-v1")]
        sql(pw["db"], "UPDATE reviews SET session_id = 'S-v1'")
        sql(pw["db"], "UPDATE alerts SET cleared_at = 'x'")
        ids = cross(pw, alerts.OVERLAP_CROSS_RUN)  # a recurrence while V1 runs: deferred, kept queued
        for _ in range(3):
            tick(pw)
        assert [t[1:] for t in triggers(pw)][2] == ("queued", None) and len(triggers(pw)) == 3
        assert len(pw["popen"].calls) == 1
        sql(pw["db"], "UPDATE reviews SET state = 'finished'")
        tick(pw)
        assert [t[1:] for t in triggers(pw)][2] == ("created", "V2") and len(pw["popen"].calls) == 2
        assert ids

    def test_no_popen_requests_nothing(self, pw, monkeypatch, capsys) -> None:
        seams = activeloop.seams()
        monkeypatch.setattr(
            activeloop, "_seams", activeloop.Seams(seams.runner, seams.probe, seams.claude, seams.main, seams.github)
        )
        cross(pw)
        tick(pw)
        tick(pw)
        assert triggers(pw) == [] and review_states(pw) == {}
        assert capsys.readouterr().err.count("no popen seam") == 1


# --------------------------------------------------------------------------- the launcher (D10)


class TestLauncher:
    def test_the_detached_launcher(self, pw, monkeypatch) -> None:
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "S-caller")
        monkeypatch.setenv("QS_CP_TOKEN", "run:R1.1." + "0" * 32)
        ids = cross(pw)
        tick(pw)
        [(argv, kw)] = pw["popen"].calls
        row = dict(sql(pw["db"], "SELECT * FROM reviews WHERE id = 'V1'")[0])
        directory = reviews.review_dir("V1", create=False)
        key = f"pilot:{event_key(alerts.OVERLAP_CROSS_RUN, ('T1', 'T5'), min(ids))}:1"
        assert argv == [
            sys.executable,  # no <MAIN>/venv here
            f"{pw['main']}/scripts/qs/cp.py",
            "tool",
            "review",
            "--key",
            key,
            "--args-file",
            str(directory / "args.json"),
            "--token",
            f"review:V1.0.{row['grant_nonce']}",
        ]
        assert json.loads((directory / "args.json").read_text()) == {"review": "V1"}
        assert kw["cwd"] == str(pw["main"]) and kw["start_new_session"] is True and kw["stdin"] is subprocess.DEVNULL
        assert (
            kw["stdout"] is kw["stderr"]
            and kw["stdout"].closed
            and kw["stdout"].name == str(directory / "launcher.log")
        )
        env = kw["env"]
        assert (
            env["QS_CP_DB"] == os.environ["QS_CP_DB"]
            and "CLAUDE_CODE_SESSION_ID" not in env
            and "QS_CP_TOKEN" not in env
        )
        assert row["grant_nonce"] not in (directory / "launcher.log").read_text()

    def test_a_relative_db_path_reaches_the_launcher_absolute(self, pw, monkeypatch) -> None:
        monkeypatch.chdir(pw["db"].parent)
        monkeypatch.setenv("QS_CP_DB", pw["db"].name)
        cross(pw)
        tick(pw)
        [(_, kw)] = pw["popen"].calls
        assert kw["env"]["QS_CP_DB"] == str(pw["db"].resolve())

    def test_the_venv_python(self, pw) -> None:
        venv = pw["main"] / "venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").write_text("")
        cross(pw)
        tick(pw)
        assert pw["popen"].calls[0][0][0] == str(venv / "python")

    def test_never_waited_on_and_reaped_by_poll(self, pw) -> None:
        cross(pw)
        tick(pw)
        launcher_alive(pw)
        [proc] = pw["popen"].procs
        assert proc.polls == 0 and reviewpilot._state.children == [proc]
        tick(pw)
        assert proc.polls == 1 and reviewpilot._state.children == [proc]
        proc.returncode = 0
        tick(pw)
        assert reviewpilot._state.children == []

    @pytest.mark.parametrize("where", ["popen", "dir"])
    def test_a_launcher_that_cannot_start_fails_the_review(self, pw, monkeypatch, where: str) -> None:
        if where == "popen":

            def broken(argv: list[str], kw: dict[str, Any]) -> None:
                raise OSError("no fork")

            pw["popen"].on_call = broken
        else:
            monkeypatch.setattr(
                reviews,
                "review_dir",
                lambda rid, create=True: (_ for _ in ()).throw(errors.CpError("POLICY_REFUSED", "inside")),
            )
        cross(pw)
        tick(pw)
        assert review_states(pw) == {"V1": "failed"}
        assert failed_alerts(pw) == [("R1", "review:V1:failed"), ("R2", "review:V1:failed")]
        assert "the launcher did not start" in sql(pw["db"], "SELECT error FROM reviews")[0][0]


# --------------------------------------------------------------------------- the retry of a failed launch (Q7)


class TestRetry:
    def test_one_retry_then_none(self, pw) -> None:
        def broken(argv: list[str], kw: dict[str, Any]) -> None:
            raise OSError("no fork")

        pw["popen"].on_call = broken
        ids = cross(pw)
        tick(pw)
        tick(pw)
        tick(pw)
        tick(pw)
        base = f"pilot:{event_key(alerts.OVERLAP_CROSS_RUN, ('T1', 'T5'), min(ids))}"
        assert triggers(pw) == [(f"{base}:1", "created", "V1"), (f"{base}:2", "created", "V2")]
        assert review_states(pw) == {"V1": "failed", "V2": "failed"} and len(pw["popen"].calls) == 2

    def test_a_retry_launches(self, pw) -> None:
        cross(pw)
        tick(pw)
        sql(pw["db"], "UPDATE reviews SET state = 'failed'")
        tick(pw)
        assert [t[1:] for t in triggers(pw)] == [("created", "V1"), ("created", "V2")]
        assert len(pw["popen"].calls) == 2

    def test_a_malformed_pilot_row_stops_nothing(self, pw) -> None:
        for key, actor, outcome in (
            ("pilot:foo", reviewpilot.ACTOR, "created"),  # no attempt number
            ("pilot:bad:x", reviewpilot.ACTOR, "queued"),
            ("PILOT:junk:1", "orchestrator", "queued"),  # not the pilot's: the scans are exact and case-sensitive
        ):
            sql(
                pw["db"],
                "INSERT INTO review_triggers (actor, \"trigger\", tool_key, outcome, at) VALUES (?, '{}', ?, ?, 'x')",
                [actor, key, outcome],
            )
        cross(pw)
        tick(pw)
        assert review_states(pw) == {"V1": "requested"} and len(pw["popen"].calls) == 1
        sql(pw["db"], "UPDATE reviews SET state = 'failed'")
        tick(pw)
        assert review_states(pw) == {"V1": "failed", "V2": "requested"} and len(pw["popen"].calls) == 2
        assert [t[:2] for t in triggers(pw)][:3] == [
            ("pilot:foo", "created"),
            ("pilot:bad:x", "queued"),
            ("PILOT:junk:1", "queued"),
        ]

    def test_a_bad_alert_payload_stops_nothing(self, pw) -> None:
        bad = ("[]", "x", "null", '{"tasks": null}', '{"tasks": 5}', '{"tasks": "T12"}')
        for n, payload in enumerate(bad):
            sql(
                pw["db"],
                "INSERT INTO alerts (run_id, kind, subject, fingerprint, payload, first_seen, last_seen)"
                " VALUES ('R1', ?, ?, ?, ?, 'x', 'x')",
                [alerts.OVERLAP_CROSS_RUN, f"bad{n}", f"bad{n}", payload],
            )
        cross(pw)
        tick(pw)
        assert review_states(pw) == {"V1": "requested"} and len(pw["popen"].calls) == 1
        assert [t[1] for t in triggers(pw)].count("skipped") == len(bad)  # the bad groups: no task to review
        rows = sql(pw["db"], "SELECT \"trigger\" FROM review_triggers WHERE outcome = 'skipped'")
        assert all(json.loads(r[0])["tasks"] == [] for r in rows)  # never a string's characters

    def test_a_pilot_row_with_a_non_object_trigger_stops_nothing(self, pw) -> None:
        ids = cross(pw)
        family = event_key(alerts.OVERLAP_CROSS_RUN, ("T1", "T5"), ids[0]).rpartition(":")[0]
        for n, trigger in enumerate(("[]", '{"alert_ids": 5}', '{"alert_ids": [[1], "x"]}')):
            sql(
                pw["db"],
                "INSERT INTO review_triggers (actor, \"trigger\", tool_key, outcome, at) VALUES (?, ?, ?, 'skipped', 'x')",
                [reviewpilot.ACTOR, trigger, f"pilot:{family}:{900 + n}:1"],
            )
        tick(pw)
        assert review_states(pw) == {"V1": "requested"} and len(pw["popen"].calls) == 1

    def test_a_queued_pilot_row_with_a_bad_trigger_stops_nothing(self, pw) -> None:
        for n, trigger in enumerate(
            ("[]", "x", '{"tasks": 5}', '{"source": "alert", "tasks": [], "alert_ids": [[1]]}')
        ):
            sql(
                pw["db"],
                "INSERT INTO review_triggers (actor, \"trigger\", tool_key, outcome, at) VALUES (?, ?, ?, 'queued', 'x')",
                [reviewpilot.ACTOR, trigger, f"pilot:junk{n}:1"],
            )
        cross(pw)
        tick(pw)
        assert review_states(pw) == {"V1": "requested"} and len(pw["popen"].calls) == 1
        assert [t[1] for t in triggers(pw)][:4] == ["queued", "queued", "queued", "skipped"]

    def test_a_failed_review_with_a_bad_trigger_is_not_retried(self, pw) -> None:
        cross(pw)
        tick(pw)
        sql(pw["db"], "UPDATE reviews SET state = 'failed'")
        for trigger in ("[]", "x", '{"tasks": 5}'):
            sql(pw["db"], 'UPDATE review_triggers SET "trigger" = ?', [trigger])
            reviewpilot.retry_failed(pw["conn"], pw["clock"])  # direct: discovery would re-raise the event
            assert len(triggers(pw)) == 1

    def test_payload_decoding(self) -> None:
        assert reviewpilot._payload(None) == {} and reviewpilot._payload("[1]") == {}
        assert reviewpilot._payload("{") == {} and reviewpilot._payload('{"tasks": ["T1"]}') == {"tasks": ["T1"]}

    def test_an_abandoned_review_is_not_retried(self, pw) -> None:
        cross(pw)
        tick(pw)
        sql(pw["db"], "UPDATE reviews SET state = 'abandoned'")
        tick(pw)
        tick(pw)
        assert len(triggers(pw)) == 1 and len(pw["popen"].calls) == 1

    def test_a_merge_event_is_retried_after_the_cursor_moved(self, pw) -> None:
        alert(pw, "R1", alerts.OVERLAP_CROSS_RUN, ["T1", "T5"], cleared=True)
        merge(pw, "T1")
        pw["popen"].on_call = lambda argv, kw: (_ for _ in ()).throw(OSError("no fork"))
        tick(pw)
        pw["popen"].on_call = None
        tick(pw)
        hid = sql(pw["db"], "SELECT max(id) FROM task_history")[0][0]
        assert [t[:2] for t in triggers(pw)] == [
            (f"pilot:merge:T1:{hid}:1", "created"),
            (f"pilot:merge:T1:{hid}:2", "created"),
        ]
        assert review_states(pw) == {"V1": "failed", "V2": "requested"}


# --------------------------------------------------------------------------- the merge rule and its cursor


def merge(pw: dict[str, Any], task: str) -> int:
    sql(pw["db"], "UPDATE tasks SET state = 'merged' WHERE id = ?", [task])
    sql(
        pw["db"],
        "INSERT INTO task_history (task_id, at, actor, from_state, to_state) VALUES (?, 'x', 'orchestrator', 'ready_to_merge', 'merged')",
        [task],
    )
    return int(sql(pw["db"], "SELECT max(id) FROM task_history")[0][0])


def cursor(pw: dict[str, Any]) -> int:
    return int(sql(pw["db"], "SELECT value FROM meta WHERE key = ?", [schema_reviews.CURSOR_KEY])[0][0])


class TestMergeRule:
    def test_a_merge_that_touches_another_run(self, pw) -> None:
        alert(pw, "R1", alerts.OVERLAP_CROSS_RUN, ["T1", "T5"], cleared=True)  # cleared rows count too
        alert(pw, "R1", alerts.OVERLAP_CROSS_RUN, ["T1", "T2"], cleared=True)  # same run: no partner
        hid = merge(pw, "T1")
        tick(pw)
        assert triggers(pw) == [(f"pilot:merge:T1:{hid}:1", "created", "V1")]
        stored = json.loads(sql(pw["db"], 'SELECT "trigger" FROM review_triggers')[0][0])
        assert stored == {"event": f"merge:T1:{hid}", "source": "merge", "merged_task": "T1", "tasks": ["T1", "T5"]}
        assert [r[0] for r in sql(pw["db"], "SELECT task_id FROM review_tasks ORDER BY 1")] == ["T1", "T5"]
        assert cursor(pw) == hid

    def test_no_event_without_an_overlap_or_another_live_run(self, pw) -> None:
        alert(pw, "R1", alerts.OVERLAP_CROSS_RUN, ["T2", "T7"], cleared=True)
        sql(pw["db"], "UPDATE tasks SET state = 'dropped' WHERE id = 'T7'")
        merge(pw, "T1")  # never named by an overlap
        hid = merge(pw, "T2")  # its partner is terminal
        tick(pw)
        assert triggers(pw) == [] and cursor(pw) == hid

    def test_other_history_moves_the_cursor_too(self, pw) -> None:
        sql(pw["db"], "INSERT INTO task_history (task_id, at, actor, to_state) VALUES ('T1', 'x', 'a', 'building')")
        tick(pw)
        assert cursor(pw) == sql(pw["db"], "SELECT max(id) FROM task_history")[0][0]

    def test_a_crash_between_two_merges_loses_nothing(self, pw, monkeypatch) -> None:
        alert(pw, "R1", alerts.OVERLAP_CROSS_RUN, ["T1", "T5"], cleared=True)
        alert(pw, "R1", alerts.OVERLAP_CROSS_RUN, ["T2", "T7"], cleared=True)
        merge(pw, "T1")
        merge(pw, "T2")
        real = reviewpilot._overlap_partners
        calls = {"n": 0}

        def crash_on_the_second(conn: Any, task: str, run: Any) -> list[str]:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("crash")
            return real(conn, task, run)

        monkeypatch.setattr(reviewpilot, "_overlap_partners", crash_on_the_second)
        with pytest.raises(RuntimeError):
            tick(pw)
        assert triggers(pw) == [] and cursor(pw) == 0
        monkeypatch.setattr(reviewpilot, "_overlap_partners", real)
        tick(pw)
        assert [t[0].split(":")[2] for t in triggers(pw)] == ["T1", "T2"]

    def test_bounded_per_tick(self, pw, monkeypatch) -> None:
        monkeypatch.setattr(reviewpilot, "MERGES_PER_TICK", 1)
        alert(pw, "R1", alerts.OVERLAP_CROSS_RUN, ["T1", "T5"], cleared=True)
        first = merge(pw, "T1")
        merge(pw, "T2")
        tick(pw)
        assert cursor(pw) == first and len(triggers(pw)) == 1
        tick(pw)
        assert cursor(pw) == first + 1


# --------------------------------------------------------------------------- closing dead reviews (D12 step 2)


def review(
    pw: dict[str, Any],
    state: str,
    *,
    age: float = 0.0,
    launch_age: float | None = None,
    session: str | None = None,
    key: str = "pilot:e:1",
) -> None:
    from control_plane import clock as clock_mod

    created = clock_mod.stamp(pw["clock"], plus=-age)
    launch = None if launch_age is None else clock_mod.stamp(pw["clock"], plus=-launch_age)
    sql(
        pw["db"],
        "INSERT INTO reviews (id, kind, coalesce_key, state, agent, name, tool_key, requested_by, session_id,"
        " created_at, launch_at, updated_at) VALUES ('V1', 'cross_run', 'k', ?, 'a', 'n', ?, 'x', ?, ?, ?, ?)",
        [state, f"review/{key}", session, created, launch, created],
    )
    for task in ("T1", "T5"):
        sql(
            pw["db"],
            "INSERT INTO review_tasks (review_id, task_id, run_id) SELECT 'V1', id, run_id FROM tasks WHERE id = ?",
            [task],
        )


def call(pw: dict[str, Any], state: str, pid: int = 4242, key: str = "pilot:e:1") -> None:
    sql(
        pw["db"],
        "INSERT INTO tool_calls (tool, key, args_hash, actor, args, state, holder_pid, holder_pid_start, holder_pgid,"
        " started_at) VALUES ('review', ?, 'h', 'review:V1', '{}', ?, ?, ?, ?, 'x')",
        [key, state, pid, f"start-{pid}", pid],
    )


def close(pw: dict[str, Any]) -> list[str]:
    return reviewpilot.close_dead(pw["conn"], pw["clock"], activeloop.seams())


class TestCloseDead:
    def test_a_launcher_that_never_started(self, pw) -> None:
        review(pw, "requested", age=59)
        assert close(pw) == []
        pw["clock"].advance(1)
        assert close(pw) == ["V1"] and review_states(pw) == {"V1": "failed"}
        assert failed_alerts(pw) == [("R1", "review:V1:failed"), ("R2", "review:V1:failed")]
        payload = json.loads(sql(pw["db"], "SELECT payload FROM alerts LIMIT 1")[0][0])
        assert payload["error"] == "the launcher never started" and payload["finding_ids"] == []

    @pytest.mark.parametrize("state", ["requested", "launching"])
    def test_a_failed_call(self, pw, state: str) -> None:
        review(pw, state)
        call(pw, "failed")
        assert close(pw) == ["V1"]
        assert sql(pw["db"], "SELECT error FROM reviews")[0][0] == "the launcher's call failed"

    def test_a_launched_review_whose_call_failed_under_an_unknown_listing_waits_for_the_ttl(self, pw) -> None:
        review(pw, "launching", age=10, launch_age=10)
        call(pw, "failed")
        pw["claude"].listing = None
        assert close(pw) == [] and review_states(pw) == {"V1": "launching"} and failed_alerts(pw) == []
        pw["clock"].advance(schema_reviews.REQUEST_TTL_S - 11)
        assert close(pw) == []
        pw["clock"].advance(1)
        assert close(pw) == ["V1"]
        assert sql(pw["db"], "SELECT error FROM reviews")[0][0] == "the launch timed out"

    def test_a_launched_review_whose_call_failed_under_a_known_listing_fails(self, pw) -> None:
        review(pw, "launching", age=10, launch_age=10)
        call(pw, "failed")
        pw["claude"].listing = [agent("S-other", "other")]
        assert close(pw) == ["V1"]
        assert sql(pw["db"], "SELECT error FROM reviews")[0][0] == "the launcher's call failed"

    def test_a_dead_holder_after_the_settle_time(self, pw) -> None:
        review(pw, "launching", age=30)
        call(pw, "started")
        pw["probe"].kill(4242)
        assert close(pw) == []  # not settled yet
        pw["clock"].advance(30)
        assert close(pw) == ["V1"]
        assert sql(pw["db"], "SELECT error FROM reviews")[0][0] == "the launcher died"

    def _dead_launcher_after_its_launch(self, pw) -> None:
        review(pw, "launching", age=61, launch_age=61)
        call(pw, "started")
        pw["probe"].kill(4242)

    def test_a_dead_launcher_whose_session_is_listed_is_adopted(self, pw) -> None:
        self._dead_launcher_after_its_launch(pw)
        pw["claude"].listing = [agent("S-n", "n", id="short-n")]
        assert close(pw) == [] and review_states(pw) == {"V1": "running"} and failed_alerts(pw) == []
        row = dict(sql(pw["db"], "SELECT session_id, short_id FROM reviews")[0])
        assert row == {"session_id": "S-n", "short_id": "short-n"}

    def test_a_dead_launcher_whose_session_is_not_listed_fails(self, pw) -> None:
        self._dead_launcher_after_its_launch(pw)
        pw["claude"].listing = [agent("S-other", "other")]
        assert close(pw) == ["V1"] and review_states(pw) == {"V1": "failed"}

    def test_a_dead_launcher_under_a_failed_listing_waits_for_the_ttl(self, pw) -> None:
        self._dead_launcher_after_its_launch(pw)
        pw["claude"].listing = None
        assert close(pw) == [] and review_states(pw) == {"V1": "launching"}
        pw["clock"].advance(schema_reviews.REQUEST_TTL_S)
        assert close(pw) == ["V1"]
        assert sql(pw["db"], "SELECT error FROM reviews")[0][0] == "the launch timed out"

    def test_the_adoption_is_guarded(self, pw, monkeypatch) -> None:
        self._dead_launcher_after_its_launch(pw)
        pw["claude"].listing = [agent("S-n", "n", id="short-n")]
        real = daemon.beat

        def finish_meanwhile(conn: Any, clock: Any) -> None:
            sql(pw["db"], "UPDATE reviews SET state = 'finished'")
            real(conn, clock)

        monkeypatch.setattr(daemon, "beat", finish_meanwhile)
        assert close(pw) == [] and review_states(pw) == {"V1": "finished"}
        assert sql(pw["db"], "SELECT session_id FROM reviews")[0][0] is None

    def test_a_live_holder_until_the_request_ttl(self, pw) -> None:
        review(pw, "launching", age=schema_reviews.REQUEST_TTL_S - 1)  # launch_at NULL: since = updated_at
        call(pw, "started")
        assert close(pw) == []
        pw["clock"].advance(1)
        assert close(pw) == ["V1"]
        assert sql(pw["db"], "SELECT error FROM reviews")[0][0] == "the launch timed out"

    def test_a_running_session_unlisted_twice_is_abandoned(self, pw) -> None:
        review(pw, "running", launch_age=10, session="S-v1")
        call(pw, "succeeded")
        pw["claude"].listing = []
        assert close(pw) == []
        pw["claude"].listing = None  # a failed listing changes nothing
        assert close(pw) == []
        pw["claude"].listing = [agent("S-v1")]  # seen again: the count restarts
        assert close(pw) == []
        pw["claude"].listing = []
        assert close(pw) == []
        reviewpilot._reset_for_tests()  # a new daemon starts cold
        assert close(pw) == []
        assert close(pw) == ["V1"] and review_states(pw) == {"V1": "abandoned"}
        assert sql(pw["db"], "SELECT error FROM reviews")[0][0] == "the review session is gone"

    def test_the_listing_is_taken_only_while_a_review_runs(self, pw) -> None:
        review(pw, "launching")
        call(pw, "started")
        close(pw)
        assert pw["claude"].listings == 0

    def test_a_running_review_past_its_ttl(self, pw) -> None:
        review(pw, "running", launch_age=schema_reviews.REVIEW_TTL_S, session="S-v1")
        pw["claude"].listing = [agent("S-v1")]
        assert close(pw) == ["V1"] and review_states(pw) == {"V1": "abandoned"}

    def test_the_guard_against_a_finish_between_read_and_write(self, pw, monkeypatch) -> None:
        review(pw, "running", launch_age=schema_reviews.REVIEW_TTL_S, session="S-v1")
        real = daemon.beat

        def finish_meanwhile(conn: Any, clock: Any) -> None:
            sql(pw["db"], "UPDATE reviews SET state = 'finished'")
            real(conn, clock)

        monkeypatch.setattr(daemon, "beat", finish_meanwhile)
        assert close(pw) == [] and review_states(pw) == {"V1": "finished"} and failed_alerts(pw) == []

    def test_a_closed_run_gets_no_alert(self, pw) -> None:
        review(pw, "requested", age=60)
        sql(pw["db"], "UPDATE runs SET state = 'closed' WHERE id = 'R2'")
        assert close(pw) == ["V1"] and failed_alerts(pw) == [("R1", "review:V1:failed")]


# --------------------------------------------------------------------------- end to end


def test_two_runs_end_to_end(pw, fake_runner, fake_main: Path) -> None:
    """The pilot requests, the launcher (in process) launches, the session writes, both runs hear of it."""
    from control_plane.runner import RunResult

    agents = fake_main / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "qs-cross-run-review.md").write_text("---\nname: qs-cross-run-review\n---\n")

    def launched(c: Any) -> RunResult:
        name = c.argv[c.argv.index("-n") + 1]
        pw["claude"].listing = [agent(f"S-{name}", name, id="short-1")]
        return RunResult(0, "", "")

    fake_runner.on(["claude", "--bg"], launched)
    pw["popen"].on_call = lambda argv, kw: launcher_out.append(run_cli(*argv[2:]))
    launcher_out: list[Any] = []
    cross(pw)
    tick(pw)
    [(code, out)] = launcher_out
    assert code == 0 and out["result"]["state"] == "running", out
    row = dict(sql(pw["db"], "SELECT * FROM reviews WHERE id = 'V1'")[0])
    token = f"review:V1.1.{row['nonce']}"
    item = pw["tmp"] / "f.json"
    item.write_text(
        json.dumps({"severity": "must_fix", "category": "design", "title": "T1 and T5 collide", "body": "b"})
    )
    for task in ("T1", "T5"):
        opened = run_cli(
            "finding",
            "open",
            "--task",
            task,
            "--phase",
            "build",
            "--source",
            "cross_run_review",
            "--input",
            str(item),
            "--token",
            token,
        )
        assert opened[0] == 0, opened
    done = pw["tmp"] / "done.json"
    done.write_text(
        json.dumps(
            {
                "summary": "collide",
                "tasks": {"T1": {"outcome": "fail", "reason": "x"}, "T5": {"outcome": "fail", "reason": "x"}},
            }
        )
    )
    assert run_cli("review", "finish", "--input", str(done), "--token", token)[0] == 0
    msgs = sql(pw["db"], "SELECT run_id, payload FROM messages WHERE kind = 'review_done' ORDER BY id")
    assert [m[0] for m in msgs] == ["R1", "R2"]
    assert [json.loads(m[1])["tasks"][0]["finding_ids"] for m in msgs] == [[1], [2]]
    assert [r[0] for r in sql(pw["db"], "SELECT run_id FROM findings ORDER BY id")] == ["R1", "R2"]
    tick(pw)
    assert review_states(pw) == {"V1": "finished"} and len(triggers(pw)) == 1


def test_fakeproc_is_what_fakepopen_returns(fake_popen) -> None:
    proc = fake_popen(["x"])
    assert isinstance(proc, FakeProc) and proc.poll() is None and proc.pid == 70_000
