"""#428: hand-written ``snapshot`` dicts for the board's model and render tests (AC2, AC3).

``FIXTURE_SNAPSHOT`` holds every case the board shows; ``MINIMAL_SNAPSHOT`` is a fresh daemon with
nothing pending; ``with_gate(**records)`` is a copy of the fixture with ``merge_gate`` replaced.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import Any

NOW = datetime(2026, 10, 27, 12, 0, 0, tzinfo=UTC)
NOW_ISO = "2026-10-27T12:00:00.000000Z"


def ago(seconds: int) -> str:
    """An ISO stamp ``seconds`` before ``NOW`` (the ``db.now`` format)."""
    return datetime.fromtimestamp(NOW.timestamp() - seconds, UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _lease(run_id: str, session_name: str | None, short_id: str | None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "epoch": 1,
        "session_id": f"S-{run_id}",
        "short_id": short_id,
        "session_name": session_name,
        "permission_mode": "default",
        "full_grant": 0,
        "claimed_at": ago(3600),
        "name_bound_session_id": None,
        "pending_bind_session_id": None,
    }


def _run(run_id: str, name: str, state: str, lease: dict[str, Any] | None, **kw: Any) -> dict[str, Any]:
    row = {
        "id": run_id,
        "name": name,
        "title": f"title of {name}",
        "state": state,
        "global_plan": None,
        "created_at": ago(7200),
        "closed_at": ago(60) if state == "closed" else None,
        "lease": lease,
    }
    row.update(kw)
    return row


def _task(task_id: str, run_id: str, state: str, title: str | None = None, **kw: Any) -> dict[str, Any]:
    row = {
        "id": task_id,
        "run_id": run_id,
        "parent_id": None,
        "issue_number": None,
        "title": title or f"task {task_id}",
        "kind": "feature",
        "target": "factory",
        "is_deliverable": 0,
        "deliverable_id": None,
        "item_k": None,
        "next_item_k": 1,
        "state": state,
        "blocked_from": None,
        "lane": None,
        "worktree": None,
        "branch": None,
        "pr_number": None,
        "pr_url": None,
        "ci_state": None,
        "ci_sha": None,
        "merge_sha": None,
        "created_at": ago(3600),
        "updated_at": ago(60),
    }
    row.update(kw)
    return row


def _node(node_id: str, run_id: str, task_id: str, generation: int, state: str, **kw: Any) -> dict[str, Any]:
    row = {
        "id": node_id,
        "run_id": run_id,
        "task_id": task_id,
        "generation": generation,
        "name": f"{run_id}-{task_id}-g{generation}",
        "session_id": f"S-{node_id}",
        "short_id": None,
        "permission_mode": "default",
        "state": state,
        "spawn_tool_key": None,
        "launch_at": None,
        "spawned_at": ago(600),
        "taken_over_at": None,
        "handed_back_at": None,
        "updated_at": ago(60),
    }
    row.update(kw)
    return row


def _criterion(task_id: str, idx: int, state: str, validated: bool) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "idx": idx,
        "text": f"criterion {idx} of {task_id}",
        "state": state,
        "validated_at": ago(900) if validated else None,
    }


def _question(qid: str, run_id: str, task_id: str, state: str, created: int, blocking: int = 0) -> dict[str, Any]:
    return {
        "id": qid,
        "run_id": run_id,
        "task_id": task_id,
        "message_id": None,
        "text": f"question {qid}?",
        "blocking": blocking,
        "state": state,
        "answer": "yes" if state == "answered" else None,
        "reason": None,
        "created_at": ago(created),
        "answered_at": ago(10) if state == "answered" else None,
    }


def _alert(aid: int, run_id: str, kind: str, subject: str, first: int, *, cleared: bool = False, **kw: Any) -> dict:
    row = {
        "id": aid,
        "run_id": run_id,
        "kind": kind,
        "subject": subject,
        "fingerprint": f"{kind}:{subject}:1",
        "payload": {"why": kind},
        "message_id": None,
        "first_seen": ago(first),
        "last_seen": ago(5),
        "cleared_at": ago(1) if cleared else None,
    }
    row.update(kw)
    return row


def _finding(fid: int, task_id: str, state: str, severity: str, classification: str | None = None) -> dict:
    return {
        "id": fid,
        "run_id": "R1",
        "task_id": task_id,
        "phase": "build",
        "round": 1,
        "source": "reviewer",
        "severity": severity,
        "classification": classification,
        "state": state,
        "title": f"finding {fid}",
        "body": "b",
    }


def _review(rid: str, kind: str, state: str, home: str | None, summary: str | None) -> dict[str, Any]:
    return {
        "id": rid,
        "kind": kind,
        "coalesce_key": f"{kind}:{rid}",
        "home_run_id": home,
        "state": state,
        "agent": "qs-global-review",
        "name": f"qs-review-{rid}",
        "summary": summary,
        "error": None,
        "created_at": ago(300),
        "finished_at": None,
        "updated_at": ago(30),
    }


def _review_task(rid: str, task_id: str, run_id: str, outcome: str | None, reason: str | None) -> dict[str, Any]:
    return {
        "review_id": rid,
        "task_id": task_id,
        "run_id": run_id,
        "base_sha": None,
        "head_sha": None,
        "head_state": "branch",
        "outcome": outcome,
        "reason": reason,
        "blast_radius_id": None,
        "recorded_at": ago(20) if outcome else None,
    }


GH = "https://github.com/tmenguy/quiet-solar/pull/"

FIXTURE_SNAPSHOT: dict[str, Any] = {
    "schema_version": 6,
    "runs": [
        _run("R1", "alpha", "open", _lease("R1", "alpha-chat", "abc-1")),
        _run("R2", "beta", "open", None),  # no lease, no task
        _run("R3", "gamma", "closed", _lease("R3", None, None)),
        _run("R4", "delta", "closed", None),
    ],
    "tasks": [
        _task(
            "T1",
            "R1",
            "building",
            "Build the board",
            issue_number=428,
            is_deliverable=1,
            lane="feature-factory",
            pr_number=9,
            pr_url=GH + "9",
            ci_state="failure",
            ci_sha="0123456789abcdef0123456789abcdef01234567",
        ),  # fmt: skip
        _task("T2", "R1", "ready", "Snapshot keys"),
        _task("T3", "R1", "planning", "Cycle A"),
        _task("T4", "R1", "contracted", "Cycle B"),
        _task("T5", "R1", "proposed", "Downstream of the cycle"),
        _task("T6", "R1", "merged", "Done one"),
        _task("T7", "R1", "validated", "Done two"),
        _task("T8", "R1", "dropped", "Dropped one"),
        _task(
            "T9",
            "R1",
            "blocked",
            "Blocked one",
            blocked_from="building",
            pr_number=1,
            pr_url="https://evil.example/pull/1",
        ),  # fmt: skip
        _task("T10", "R1", "building", "Item of T1", deliverable_id="T1", item_k=1, kind="bug"),
        _task("T20", "R3", "merged", "Closed done"),
        _task("T21", "R3", "dropped", "Closed dropped"),
        _task("T30", "R4", "building", "Closed remaining"),
    ],
    "task_deps": [
        {"task_id": "T1", "depends_on": "T2"},  # in-run
        {"task_id": "T1", "depends_on": "T20"},  # to another run
        {"task_id": "T3", "depends_on": "T4"},  # a 2-cycle
        {"task_id": "T4", "depends_on": "T3"},
        {"task_id": "T5", "depends_on": "T3"},  # downstream of the cycle
    ],
    "run_roots": [{"run_id": "R1", "task_id": "T1"}],
    "work_list": [{"run_id": "R1", "task_id": "T1"}],
    "criteria": [
        _criterion("T1", 0, "met", True),
        _criterion("T1", 1, "open", False),  # building with an unvalidated criterion: not rule 3
        _criterion("T2", 0, "open", False),  # ready with an unvalidated criterion: rule 3
        _criterion("T6", 0, "waived", True),
    ],
    "task_history": [],
    "nodes": [
        _node("N1", "R1", "T1", 1, "superseded", short_id="old-1"),
        _node("N2", "R1", "T1", 2, "running", short_id="n2-short"),
        _node("N3", "R1", "T4", 1, "taken_over", short_id="n3"),
        _node("N4", "R1", "T10", 1, "idle"),  # live, no short_id: no attach block
        _node("N5", "R1", "T6", 1, "reaped", short_id="gone-5"),  # not live: no attach block
    ],
    "queues": [],
    "questions": [
        _question("Q-2", "R1", "T2", "asked", 500),
        _question("Q-1", "R1", "T1", "asked", 400, blocking=1),
        _question("Q-3", "R1", "T1", "open", 300),
        _question("Q-4", "R1", "T1", "answered", 200),
        _question("Q-5", "R3", "T20", "asked", 100),  # a closed run: not listed
    ],
    "decisions": [],
    "digests": [],
    "reports": [],
    "integrations": [],
    "rounds": [],
    "findings": [
        _finding(1, "T1", "open", "must_fix"),
        _finding(2, "T1", "open", "should_fix", "out_of_scope"),  # out of scope: not counted
        _finding(3, "T1", "resolved", "must_fix"),
        _finding(4, "T10", "open", "nice_to_have"),
        _finding(5, "T1", "open", "out_of_scope", "must_fix"),  # classified: its class counts
    ],
    "finding_events": [],
    "blast_radius": [],
    "tool_calls_in_flight": [],
    "locks": [],
    "cap_slots": [],
    "daemon": {
        "id": 1,
        "pid": None,
        "pid_start": None,
        "schema_version": 6,
        "started_at": ago(7200),
        "heartbeat_at": None,
        "heartbeat_age_s": None,
        "code_schema_version": 6,
    },
    "alerts": [
        _alert(1, "R1", "orchestrator_not_listening", "R1", 50),
        _alert(2, "R2", "orchestrator_dead", "R2", 90),
        _alert(3, "R1", "ci_red", "T1", 40),
        _alert(4, "R1", "overlap_cross_run", "T1|T30", 30, fingerprint="overlap_cross_run:x:1"),
        _alert(5, "R2", "overlap_cross_run", "T1|T30", 30, fingerprint="overlap_cross_run:x:1"),
        _alert(7, "R3", "ci_red", "T20", 20),  # a closed run: not shown
        _alert(6, "R1", "node_stalled", "T3", 70, cleared=True),
    ],
    "hook_alerts": [
        {"id": 2, "hook": "stop", "session_id": "S-R1", "decision": "alert", "detail": {"kind": "x"}, "at": ago(9)},
        {"id": 1, "hook": "pre-tool-use", "session_id": None, "decision": "alert", "detail": "raw", "at": ago(99)},
    ],
    "reviews": [
        _review("V1", "global", "finished", "R1", "all good"),
        _review("V2", "cross_run", "running", None, None),
    ],
    "review_tasks": [
        _review_task("V1", "T1", "R1", "pass", "fits the plan"),
        _review_task("V2", "T1", "R1", "fail", "collides with T30"),
        _review_task("V2", "T30", "R4", None, None),
    ],
    "review_triggers": [],
    "merge_gate": {"selfcheck": None, "override": None, "pending": None},
    "backup": {"last_at": ago(1000), "last_error": {"at": ago(100), "error": "OSError: disk full"}},
}

MINIMAL_SNAPSHOT: dict[str, Any] = {
    **{key: [] for key in FIXTURE_SNAPSHOT},
    "schema_version": 6,
    "runs": [_run("R1", "solo", "open", _lease("R1", "solo-chat", "s-1"))],
    "daemon": {
        "id": 1,
        "pid": 9,
        "pid_start": "x",
        "schema_version": 6,
        "started_at": ago(60),
        "heartbeat_at": ago(5),
        "heartbeat_age_s": 5.0,
        "code_schema_version": 6,
    },
    "merge_gate": {"selfcheck": None, "override": None, "pending": None},
    "backup": {"last_at": ago(100), "last_error": None},
}


def fixture() -> dict[str, Any]:
    """A fresh deep copy of ``FIXTURE_SNAPSHOT`` (tests mutate it)."""
    return copy.deepcopy(FIXTURE_SNAPSHOT)


def with_gate(**records: Any) -> dict[str, Any]:
    """A copy of the fixture with ``merge_gate`` replaced (absent records are ``None``)."""
    snap = fixture()
    snap["merge_gate"] = {"selfcheck": None, "override": None, "pending": None, **records}
    return snap
