"""Tests for ``scripts/qs/review_ledger.py`` (QS-453 D2).

The ledger is the code-review orchestrators' durable record of findings,
decisions and state history. These tests are its coverage contract
(AC-3): the gate does not measure ``scripts/`` coverage, so every
transition-table row, every forbidden class, every validation and
reconciliation rule, every ``start-round`` branch and six end-to-end
scenarios each have a named test here.

Every test drives ``main()`` in-process against an isolated real git
repo under ``tmp_path`` (``_repo`` fixture — the env-isolation pattern of
``tests/qs/_gitrepo.py``).
"""

from __future__ import annotations

import importlib
import io
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "qs" / "review_ledger.py"
ISSUE = 7
LEDGER_REL = f"docs/stories/QS-{ISSUE}.review_ledger.json"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def _commit(repo: Path, path: str, text: str) -> str:
    """Write ``path`` (relative to ``repo``), commit it, return the new HEAD sha."""
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    _git(repo, "add", "--", path)
    _git(repo, "commit", "-q", "-m", f"edit {path}")
    return _head(repo)


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def _repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE", "GIT_OBJECT_DIRECTORY"):
        monkeypatch.delenv(key, raising=False)
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "QS Test")
    _git(repo, "config", "user.email", "qs@example.invalid")
    _commit(repo, "a.py", "one\n")
    return repo.resolve()


@pytest.fixture
def mod() -> ModuleType:
    return importlib.import_module("review_ledger")


def _finding(
    key: str = "C-1",
    *,
    source: str = "qs-review-edge-case-hunter",
    severity: str = "must-fix",
    where: str = "a.py:1",
    title: str = "a title",
    verdict: str | None = "CONFIRMED",
    evidence: str | None = "a.py:1",
    reproduces_when: str | None = "an input",
    severity_action: str = "keep",
    kind: str = "NEW",
    ref: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "source": source,
        "severity": severity,
        "where": where,
        "title": title,
        "verdict": verdict,
        "evidence": evidence,
        "reproduces_when": reproduces_when,
        "severity_action": severity_action,
        "relation": {"kind": kind, "ref": ref},
        "note": note,
    }


def _unverified(key: str = "C-1", **kw: Any) -> dict[str, Any]:
    """A fallback-mode finding (D4 step 4)."""
    return _finding(key, verdict=None, evidence=None, reproduces_when=None, note="unverified", **kw)


def _check(ref: str, status: str = "RESOLVED", evidence: str = "a.py:2") -> dict[str, Any]:
    return {"ref": ref, "status": status, "evidence": evidence}


def _obj(
    findings: list[dict[str, Any]] | None = None, checks: list[dict[str, Any]] | None = None, mode: str = "verified"
) -> dict[str, Any]:
    return {"mode": mode, "checks": checks or [], "findings": findings or []}


def _keys_of(stdin: Any) -> list[str]:
    """The consolidated keys an ``add`` object carries — what the orchestrator passes as ``--keys``."""
    findings = stdin.get("findings") if isinstance(stdin, dict) else None
    if not isinstance(findings, list):
        return []
    keys = [f["key"] for f in findings if isinstance(f, dict) and isinstance(f.get("key"), str)]
    return list(dict.fromkeys(keys))  # deduped, so _validate's own duplicate-key branch is reached (fix #03 F-033)


class Ledger:
    """In-process driver for ``review_ledger.main`` against one repo."""

    def __init__(
        self, mod: ModuleType, repo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.mod = mod
        self.repo = repo
        self.capsys = capsys
        self.monkeypatch = monkeypatch
        self.path = repo / LEDGER_REL

    def __call__(self, cmd: str, *args: str, stdin: Any = None) -> tuple[int, Any]:
        if stdin is not None:
            text = stdin if isinstance(stdin, str) else json.dumps(stdin)
            self.monkeypatch.setattr(sys, "stdin", io.StringIO(text))
        if cmd == "add" and not any(a.startswith("--keys") for a in args):
            args = (*args, f"--keys={','.join(_keys_of(stdin))}")
        self.capsys.readouterr()
        code = self.mod.main([cmd, "--issue", str(ISSUE), "--root", str(self.repo), *args])
        out = self.capsys.readouterr().out
        try:
            return code, json.loads(out)
        except ValueError:
            return code, out

    def ok(self, cmd: str, *args: str, stdin: Any = None) -> Any:
        code, out = self(cmd, *args, stdin=stdin)
        assert code == 0, out
        return out

    def fail(self, cmd: str, *args: str, stdin: Any = None, code: int = 1) -> dict[str, Any]:
        got, out = self(cmd, *args, stdin=stdin)
        assert got == code, out
        assert isinstance(out, dict) and "error" in out, out
        return out

    # -- round plumbing ------------------------------------------------------

    def start(self, *extra: str) -> dict[str, Any]:
        return self.ok("start-round", "--head", _head(self.repo), *extra)

    def add(self, obj: dict[str, Any]) -> dict[str, Any]:
        return self.ok("add", stdin=obj)

    def set(self, fid: str, to: str, by: str | None = None, note: str | None = None) -> dict[str, Any]:
        args = ["--id", fid, "--to", to]
        if by is not None:
            args += ["--by", by]
        if note is not None:
            args += ["--note", note]
        return self.ok("set", *args)

    def commit_ledger(self) -> None:
        _git(self.repo, "add", "--", LEDGER_REL)
        _git(self.repo, "commit", "-q", "-m", "ledger", "--", LEDGER_REL)

    def round(
        self, obj: dict[str, Any], decisions: dict[str, tuple[str, str | None]] | None = None, *extra: str
    ) -> dict[str, Any]:
        """start-round → add → set each decision → check-round → commit the ledger."""
        self.start(*extra)
        result = self.add(obj)
        for fid, (to, by) in (decisions or {}).items():
            self.set(fid, to, by)
        self.ok("check-round")
        self.commit_ledger()
        return result

    def data(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def finding(self, fid: str) -> dict[str, Any]:
        return next(f for f in self.data()["findings"] if f["id"] == fid)

    def write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def led(mod: ModuleType, _repo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> Ledger:
    return Ledger(mod, _repo, capsys, monkeypatch)


def _accepted_round1(led: Ledger) -> None:
    """Round 1 with F-001 accepted, committed, then a code push."""
    led.round(_obj([_finding()]), {"F-001": ("accepted", None)})
    _commit(led.repo, "a.py", "two\n")


# ---------------------------------------------------------------------------
# Transition table (D2) — one test per row
# ---------------------------------------------------------------------------


def test_transition_new_creates_open(led: Ledger) -> None:
    led.start()
    out = led.add(_obj([_finding()]))
    assert out["findings"] == {"C-1": {"id": "F-001", "effect": "new"}}
    f = led.finding("F-001")
    assert f["state"] == "open"
    assert f["relation"] is None
    assert f["history"] == [{"round": 1, "from": None, "to": "open", "note": "added"}]


def test_transition_flipflop_a_creates_new_open_finding(led: Ledger) -> None:
    _accepted_round1(led)
    led.start()
    out = led.add(_obj([_finding(kind="FLIP-FLOP", ref="F-001")], [_check("F-001")]))
    assert out["findings"]["C-1"] == {"id": "F-002", "effect": "flip-flop"}
    assert led.finding("F-002")["relation"] == {"kind": "FLIP-FLOP", "ref": "F-001"}
    assert led.finding("F-002")["state"] == "open"


def test_transition_open_to_accepted(led: Ledger) -> None:
    led.start()
    led.add(_obj([_finding()]))
    out = led.set("F-001", "accepted")
    assert out == {"id": "F-001", "state": "accepted", "superseded": None}
    f = led.finding("F-001")
    assert f["state"] == "accepted"
    assert f["decided_round"] == 1
    assert f["history"][-1] == {"round": 1, "from": "open", "to": "accepted", "note": "user: fix"}


@pytest.mark.parametrize("by", ["user", "default-skip", "verifier"])
def test_transition_open_to_rejected(led: Ledger, by: str) -> None:
    led.start()
    led.add(_obj([_finding()]))
    led.set("F-001", "rejected", by, note="not worth it")
    f = led.finding("F-001")
    assert (f["state"], f["rejected_by"], f["decided_round"]) == ("rejected", by, 1)
    assert f["history"][-1]["note"] == f"{by}: not worth it"


def test_transition_accepted_to_resolved_on_check(led: Ledger) -> None:
    _accepted_round1(led)
    led.start()
    out = led.add(_obj([], [_check("F-001")]))
    assert out["checks"] == {"F-001": "resolved"}
    assert led.finding("F-001")["state"] == "resolved"


def test_transition_accepted_to_open_on_still_present(led: Ledger) -> None:
    _accepted_round1(led)
    led.start()
    out = led.add(_obj([], [_check("F-001", "STILL-PRESENT")]))
    assert out["checks"] == {"F-001": "reopened"}
    f = led.finding("F-001")
    assert f["state"] == "open"
    assert f["history"][-1]["from"] == "accepted"


def test_transition_accepted_to_open_on_reopens(led: Ledger) -> None:
    _accepted_round1(led)
    led.start()
    out = led.add(_obj([_finding(kind="REOPENS", ref="F-001", where="a.py:9")], [_check("F-001")]))
    assert out["findings"]["C-1"] == {"id": "F-001", "effect": "reopened"}
    f = led.finding("F-001")
    assert f["state"] == "open"
    assert f["where"] == "a.py:9"  # re-targeted to the new report
    reopen = next(h for h in f["history"] if h["round"] == 2 and h["to"] == "open")
    assert reopen["from"] == "accepted"
    assert '"where": "a.py:1"' in reopen["note"]  # previous values kept in history


def test_transition_resolved_to_open_on_reopens(led: Ledger) -> None:
    _accepted_round1(led)
    led.round(_obj([], [_check("F-001")]))
    _commit(led.repo, "a.py", "three\n")
    led.start()
    out = led.add(_obj([_finding(kind="REOPENS", ref="F-001")]))
    assert out["findings"]["C-1"] == {"id": "F-001", "effect": "reopened"}
    assert led.finding("F-001")["state"] == "open"
    assert led.finding("F-001")["history"][-1]["from"] == "resolved"


def test_transition_rejected_by_user_to_open_on_challenge(led: Ledger) -> None:
    led.round(_obj([_finding()]), {"F-001": ("rejected", "user")})
    _commit(led.repo, "a.py", "two\n")
    led.start()
    out = led.add(_obj([_finding(kind="FLIP-FLOP", ref="F-001", severity="should-fix")]))
    assert out["findings"]["C-1"] == {"id": "F-001", "effect": "challenged"}
    f = led.finding("F-001")
    assert (f["state"], f["rejected_by"]) == ("open", None)
    assert f["severity"] == "should-fix"


@pytest.mark.parametrize("prior", ["accepted", "resolved"])
def test_transition_to_superseded_when_flipflop_accepted(led: Ledger, prior: str) -> None:
    _accepted_round1(led)
    checks = [_check("F-001", "RESOLVED")] if prior == "resolved" else [_check("F-001", "STILL-PRESENT")]
    led.start()
    led.add(_obj([_finding(kind="FLIP-FLOP", ref="F-001")], checks))
    if prior == "accepted":
        led.set("F-001", "accepted")
    out = led.set("F-002", "accepted")
    assert out["superseded"] == "F-001"
    f = led.finding("F-001")
    assert f["state"] == "superseded"
    assert f["history"][-1] == {"round": 2, "from": prior, "to": "superseded", "note": "superseded by F-002"}


def test_auto_supersede_is_noop_when_already_superseded(led: Ledger) -> None:
    _accepted_round1(led)
    led.start()
    led.add(
        _obj(
            [_finding("C-1", kind="FLIP-FLOP", ref="F-001"), _finding("C-2", kind="FLIP-FLOP", ref="F-001")],
            [_check("F-001")],
        )
    )
    assert led.set("F-002", "accepted")["superseded"] == "F-001"
    history_len = len(led.finding("F-001")["history"])
    assert led.set("F-003", "accepted")["superseded"] is None
    assert led.finding("F-001")["state"] == "superseded"
    assert len(led.finding("F-001")["history"]) == history_len


# ---------------------------------------------------------------------------
# Forbidden classes
# ---------------------------------------------------------------------------


def test_set_to_state_outside_table_is_rejected(led: Ledger) -> None:
    led.start()
    led.add(_obj([_finding()]))
    out = led.fail("set", "--id", "F-001", "--to", "resolved")
    assert out["error"].startswith("usage:")
    assert led.finding("F-001")["state"] == "open"


def test_set_from_non_open_fails(led: Ledger) -> None:
    led.start()
    led.add(_obj([_finding()]))
    led.set("F-001", "accepted")
    out = led.fail("set", "--id", "F-001", "--to", "rejected", "--by", "user")
    assert out["id"] == "F-001"


def test_set_rejected_without_by_fails(led: Ledger) -> None:
    led.start()
    led.add(_obj([_finding()]))
    assert "--by" in led.fail("set", "--id", "F-001", "--to", "rejected")["error"]


def test_set_accepted_with_by_fails(led: Ledger) -> None:
    led.start()
    led.add(_obj([_finding()]))
    assert "--by" in led.fail("set", "--id", "F-001", "--to", "accepted", "--by", "user")["error"]


def test_set_before_add_fails(led: Ledger) -> None:
    led.start()
    led.fail("set", "--id", "F-001", "--to", "accepted")


def test_set_outside_started_round_fails(led: Ledger) -> None:
    led.round(_obj([_finding()]), {"F-001": ("accepted", None)})
    led.fail("set", "--id", "F-001", "--to", "accepted")


def test_set_without_any_round_fails(led: Ledger) -> None:
    """No ledger at all (fix #08 F-061): ``set`` is outside a started round too."""
    assert "start-round" in led.fail("set", "--id", "F-001", "--to", "accepted")["error"]
    assert not led.path.exists()


def test_add_outside_started_round_fails(led: Ledger) -> None:
    out = led.fail("add", stdin=_obj())
    assert "start-round" in out["error"]
    assert not led.path.exists()


def test_second_add_fails(led: Ledger) -> None:
    led.start()
    led.add(_obj([_finding()]))
    led.fail("add", stdin=_obj([_finding()]))
    assert len(led.data()["findings"]) == 1


def test_set_unknown_id_fails(led: Ledger) -> None:
    led.start()
    led.add(_obj([_finding()]))
    assert led.fail("set", "--id", "F-099", "--to", "accepted")["id"] == "F-099"


# ---------------------------------------------------------------------------
# Validation (both modes) — any failure writes nothing
# ---------------------------------------------------------------------------


def _bad(**overrides: Any) -> dict[str, Any]:
    f = _finding()
    f.update(overrides)
    return f


_VALIDATION_CASES: dict[str, Any] = {
    "not_json": "{nope",
    "top_level_not_object": "[]",
    "missing_top_key": {"mode": "verified", "findings": []},
    "unknown_top_key": {**_obj(), "extra": 1},
    "duplicate_json_key": '{"mode": "verified", "mode": "fallback", "checks": [], "findings": []}',
    "bad_mode": _obj(mode="maybe"),
    "checks_not_list": {"mode": "verified", "checks": {}, "findings": []},
    "findings_not_list": {"mode": "verified", "checks": [], "findings": {}},
    "finding_not_object": _obj(["C-1"]),  # type: ignore[list-item]
    "missing_finding_key": _obj([{k: v for k, v in _finding().items() if k != "title"}]),
    "unknown_finding_key": _obj([{**_finding(), "body": "x"}]),
    "duplicate_finding_key": _obj([_finding("C-1"), _finding("C-1")]),
    "empty_title": _obj([_bad(title=" ")]),
    "bad_severity": _obj([_bad(severity="blocker")]),
    "bad_verdict": _obj([_bad(verdict="MAYBE")]),
    "bad_severity_action": _obj([_bad(severity_action="raise")]),
    "bad_relation_kind": _obj([_bad(relation={"kind": "SAME", "ref": None})]),
    "relation_unknown_key": _obj([_bad(relation={"kind": "NEW", "ref": None, "x": 1})]),
    "ref_missing_for_duplicate": _obj([_bad(relation={"kind": "DUPLICATE", "ref": None})]),
    "evidence_without_verdict": _obj([_bad(verdict=None, source="orchestrator:doc-maintenance")]),
    "verdict_without_evidence": _obj([_bad(evidence=None)]),
    "note_not_string": _obj([_bad(note=3)]),
    "promote_non_nice": _obj([_bad(severity_action="promote")]),
    "promote_unverified": _obj([_bad(severity="nice-to-have", severity_action="promote", verdict=None, evidence=None)]),
    "demote_nice": _obj([_bad(severity="nice-to-have", severity_action="demote")]),
    "verified_blocking_without_verdict": _obj([_bad(verdict=None, evidence=None)]),
    "verified_without_scenario_not_demoted": _obj([_bad(reproduces_when=None)]),
    "fallback_with_verdict": _obj([_finding()], mode="fallback"),
    "check_ref_not_accepted": _obj([], [_check("F-001")]),
    "check_ref_not_string": _obj([], [{"ref": ["F-001"], "status": "RESOLVED", "evidence": "a.py:1"}]),
    "check_not_object": _obj([], ["F-001"]),  # type: ignore[list-item]
    "promote_without_scenario": _obj([_bad(severity="nice-to-have", severity_action="promote", reproduces_when=None)]),
    "promote_refuted": _obj([_bad(severity="nice-to-have", severity_action="promote", verdict="REFUTED")]),
    # Fix #02 F-021: a fallback round verifies nothing — no checks, keep, NEW only.
    "fallback_with_checks": _obj([], [_check("F-001")], mode="fallback"),
    "fallback_demote": _obj([_unverified(severity_action="demote")], mode="fallback"),
    "fallback_relation_duplicate": _obj([_unverified(kind="DUPLICATE", ref="F-001")], mode="fallback"),
    # Fix #02 F-022: a refuted finding cannot reopen or flip anything.
    "refuted_reopens": _obj(
        [
            _bad(
                verdict="REFUTED",
                reproduces_when=None,
                severity_action="demote",
                relation={"kind": "REOPENS", "ref": "F-001"},
            )
        ]
    ),
    "refuted_flipflop": _obj(
        [
            _bad(
                verdict="REFUTED",
                reproduces_when=None,
                severity_action="demote",
                relation={"kind": "FLIP-FLOP", "ref": "F-001"},
            )
        ]
    ),
}


@pytest.mark.parametrize("case", sorted(_VALIDATION_CASES))
def test_add_validation_rejects(led: Ledger, case: str) -> None:
    led.start()
    before = led.path.read_text(encoding="utf-8")
    led.fail("add", stdin=_VALIDATION_CASES[case])
    assert led.path.read_text(encoding="utf-8") == before  # nothing written, added still false
    assert led.data()["rounds"][-1]["added"] is False


def test_add_duplicate_json_key_names_the_key(led: Ledger) -> None:
    led.start()
    assert led.fail("add", stdin=_VALIDATION_CASES["duplicate_json_key"])["key"] == "mode"


def test_add_validation_check_errors(led: Ledger) -> None:
    """Check-element rules need an accepted finding to exist first."""
    _accepted_round1(led)
    led.start()
    cases = [
        _obj([], [{"ref": "F-001", "status": "RESOLVED"}]),  # missing key
        _obj([], [_check("F-001", "GONE")]),  # bad enum
        _obj([], [_check("F-001"), _check("F-001")]),  # duplicate ref
        _obj([], [_check("F-001", evidence="")]),  # empty evidence
        _obj([], []),  # verified: checks != accepted set
    ]
    for case in cases:
        led.fail("add", stdin=case)
    assert led.data()["rounds"][-1]["added"] is False
    assert led.fail("add", stdin=_obj([], []))["missing"] == ["F-001"]


@pytest.mark.parametrize(
    "finding",
    [
        _finding(verdict=None, evidence=None, source="orchestrator:doc-maintenance"),
        _finding(reproduces_when=None, source="qs-review-acceptance-auditor"),
        _finding(reproduces_when=None, source="qs-review-regression-proof"),
        _finding(reproduces_when=None, severity_action="demote"),
        _finding(severity="nice-to-have", verdict=None, evidence=None, reproduces_when=None),
        _finding(severity="nice-to-have", severity_action="promote"),
        _finding(verdict="REFUTED", reproduces_when=None, severity_action="demote"),
    ],
    ids=[
        "doc-maintenance",
        "acceptance-exempt",
        "regression-exempt",
        "demoted",
        "nice-unverified",
        "promoted",
        "refuted-demoted",
    ],
)
def test_add_validation_accepts_verified(led: Ledger, finding: dict[str, Any]) -> None:
    led.start()
    led.add(_obj([finding]))
    assert led.data()["rounds"][-1]["added"] is True


def test_add_fallback_accepts_unverified_and_skips_check_cover(led: Ledger) -> None:
    _accepted_round1(led)
    led.start()
    out = led.add(_obj([_unverified()], mode="fallback"))
    assert out == {"findings": {"C-1": {"id": "F-002", "effect": "new"}}, "checks": {}}


def test_severity_follows_severity_action(led: Ledger) -> None:
    led.start()
    led.add(
        _obj(
            [
                _finding("C-1", reproduces_when=None, severity_action="demote"),
                _finding("C-2", severity="nice-to-have", severity_action="promote"),
            ]
        )
    )
    assert (led.finding("F-001")["reported_severity"], led.finding("F-001")["severity"]) == ("must-fix", "nice-to-have")
    assert (led.finding("F-002")["reported_severity"], led.finding("F-002")["severity"]) == (
        "nice-to-have",
        "should-fix",
    )


# ---------------------------------------------------------------------------
# Reconciliation (D2 add)
# ---------------------------------------------------------------------------


def test_reconcile_preconditions_use_snapshot(led: Ledger) -> None:
    """A RESOLVED check does not invalidate a REOPENS on the same F-x (snapshot = accepted)."""
    _accepted_round1(led)
    led.start()
    out = led.add(_obj([_finding(kind="REOPENS", ref="F-001")], [_check("F-001", "RESOLVED")]))
    assert out["findings"]["C-1"] == {"id": "F-001", "effect": "reopened"}
    assert out["checks"] == {"F-001": "merged"}
    assert led.finding("F-001")["state"] == "open"


def test_reconcile_precedence_reopens_over_still_present(led: Ledger) -> None:
    _accepted_round1(led)
    led.start()
    out = led.add(_obj([_finding(kind="REOPENS", ref="F-001")], [_check("F-001", "STILL-PRESENT")]))
    assert out["findings"]["C-1"]["effect"] == "reopened"
    assert out["checks"] == {"F-001": "merged"}


def test_reconcile_precedence_still_present_over_resolved_is_one_check(led: Ledger) -> None:
    """One check per F-x: STILL-PRESENT alone wins and transitions once."""
    _accepted_round1(led)
    led.start()
    led.add(_obj([], [_check("F-001", "STILL-PRESENT")]))
    transitions = [h for h in led.finding("F-001")["history"] if h["round"] == 2 and h["from"] != h["to"]]
    assert transitions == [{"round": 2, "from": "accepted", "to": "open", "note": "check STILL-PRESENT: a.py:2"}]


def test_reconcile_precedence_challenge_twice_merges(led: Ledger) -> None:
    led.round(_obj([_finding()]), {"F-001": ("rejected", "user")})
    _commit(led.repo, "a.py", "two\n")
    led.start()
    out = led.add(
        _obj([_finding("C-1", kind="FLIP-FLOP", ref="F-001"), _finding("C-2", kind="FLIP-FLOP", ref="F-001")])
    )
    assert out["findings"] == {
        "C-1": {"id": "F-001", "effect": "challenged"},
        "C-2": {"id": "F-001", "effect": "merged"},
    }
    notes = [h["note"] for h in led.finding("F-001")["history"] if h["round"] == 2]
    assert any(n.startswith("merged:") for n in notes)


def test_reconcile_double_reopens_one_transition_one_merged(led: Ledger) -> None:
    _accepted_round1(led)
    led.start()
    out = led.add(
        _obj(
            [_finding("C-1", kind="REOPENS", ref="F-001"), _finding("C-2", kind="REOPENS", ref="F-001")],
            [_check("F-001", "RESOLVED")],
        )
    )
    assert out["findings"]["C-1"]["effect"] == "reopened"
    assert out["findings"]["C-2"]["effect"] == "merged"
    assert out["checks"]["F-001"] == "merged"


def test_reconcile_duplicate_of_decided_does_not_transition(led: Ledger) -> None:
    led.round(_obj([_finding()]), {"F-001": ("rejected", "default-skip")})
    led.start("--force-full")
    out = led.add(_obj([_finding(kind="DUPLICATE", ref="F-001")]))
    assert out["findings"]["C-1"] == {"id": "F-001", "effect": "duplicate"}
    f = led.finding("F-001")
    assert f["state"] == "rejected"
    assert f["history"][-1]["from"] == f["history"][-1]["to"] == "rejected"


def test_reconcile_duplicate_escalation_downgrades_to_new(led: Ledger) -> None:
    led.round(
        _obj([_finding(severity="nice-to-have", verdict=None, evidence=None)]), {"F-001": ("rejected", "default-skip")}
    )
    led.start("--force-full")
    out = led.add(_obj([_finding(kind="DUPLICATE", ref="F-001", severity="should-fix")]))
    assert out["findings"]["C-1"] == {"id": "F-002", "effect": "new-downgraded"}
    f = led.finding("F-002")
    assert f["relation"] is None
    assert f["history"][-1]["note"] == "relation-invalid: DUPLICATE F-001: severity escalated"


def test_reconcile_duplicate_of_superseded_stays_duplicate(led: Ledger) -> None:
    _scenario_superseded(led)
    led.start()
    out = led.add(_obj([_finding(kind="DUPLICATE", ref="F-001")], [_check("F-002")]))
    assert out["findings"]["C-1"] == {"id": "F-001", "effect": "duplicate"}


def _scenario_superseded(led: Ledger) -> None:
    """F-001 accepted (round 1) → F-002 FLIP-FLOP accepted, F-001 superseded (round 2) → push."""
    _accepted_round1(led)
    led.round(
        _obj([_finding(kind="FLIP-FLOP", ref="F-001", where="a.py:3")], [_check("F-001")]),
        {"F-002": ("accepted", None)},
    )
    assert led.finding("F-001")["state"] == "superseded"
    _commit(led.repo, "a.py", "three\n")


@pytest.mark.parametrize("kind", ["REOPENS", "FLIP-FLOP"])
def test_reconcile_superseded_ref_redirects_to_flipflop(led: Ledger, kind: str) -> None:
    _scenario_superseded(led)
    led.start()
    out = led.add(_obj([_finding(kind=kind, ref="F-001")], [_check("F-002")]))
    assert out["findings"]["C-1"] == {"id": "F-003", "effect": "flip-flop"}
    f = led.finding("F-003")
    assert f["relation"] == {"kind": "FLIP-FLOP", "ref": "F-002"}
    assert "superseded" in f["history"][-1]["note"]


def test_reconcile_superseded_without_live_superseder_downgrades(led: Ledger) -> None:
    data = {
        "version": 1,
        "rounds": [{"n": 1, "head": _head(led.repo), "added": False}],
        "findings": [
            {
                "id": "F-001",
                "round": 1,
                "source": "s",
                "reported_severity": "must-fix",
                "severity_action": "keep",
                "severity": "must-fix",
                "where": "a.py:1",
                "title": "t",
                "note": None,
                "verdict": None,
                "evidence": None,
                "reproduces_when": None,
                "relation": None,
                "state": "superseded",
                "rejected_by": None,
                "decided_round": 1,
                "history": [],
            }
        ],
    }
    led.write(data)  # untracked ledger with a round → a started round
    out = led.add(_obj([_finding(kind="REOPENS", ref="F-001")]))
    assert out["findings"]["C-1"] == {"id": "F-002", "effect": "new-downgraded"}


@pytest.mark.parametrize(
    ("setup", "kind"),
    [
        ("unknown", "REOPENS"),
        ("rejected-user", "REOPENS"),
        ("rejected-default-skip", "FLIP-FLOP"),
        ("rejected-verifier", "FLIP-FLOP"),
        ("open", "DUPLICATE"),
        ("open", "REOPENS"),
        ("open", "FLIP-FLOP"),
    ],
)
def test_reconcile_downgrades_invalid_relation(led: Ledger, setup: str, kind: str) -> None:
    if setup == "open":
        data = {
            "version": 1,
            "rounds": [{"n": 1, "head": _head(led.repo), "added": False}],
            "findings": [
                {
                    "id": "F-001",
                    "round": 1,
                    "source": "s",
                    "reported_severity": "must-fix",
                    "severity_action": "keep",
                    "severity": "must-fix",
                    "where": "a.py:1",
                    "title": "t",
                    "note": None,
                    "verdict": None,
                    "evidence": None,
                    "reproduces_when": None,
                    "relation": None,
                    "state": "open",
                    "rejected_by": None,
                    "decided_round": None,
                    "history": [],
                }
            ],
        }
        led.write(data)
    else:
        if setup != "unknown":
            led.round(_obj([_finding()]), {"F-001": ("rejected", setup.removeprefix("rejected-"))})
        led.start("--force-full")
    ref = "F-099" if setup == "unknown" else "F-001"
    out = led.add(_obj([_finding(kind=kind, ref=ref)]))
    assert out["findings"]["C-1"]["effect"] == "new-downgraded"
    new = led.finding(out["findings"]["C-1"]["id"])
    assert new["relation"] is None
    assert new["history"][-1]["note"].startswith(f"relation-invalid: {kind} {ref}: ")


def test_reconcile_new_ignores_ref(led: Ledger) -> None:
    led.start()
    out = led.add(_obj([_finding(ref="F-123")]))
    assert out["findings"]["C-1"]["effect"] == "new"
    assert led.finding("F-001")["relation"] is None


def test_ids_are_sequential_and_widen_past_999(led: Ledger) -> None:
    data = {
        "version": 1,
        "rounds": [{"n": 1, "head": _head(led.repo), "added": False}],
        "findings": [
            {
                "id": "F-999",
                "round": 1,
                "source": "s",
                "reported_severity": "must-fix",
                "severity_action": "keep",
                "severity": "must-fix",
                "where": "a.py:1",
                "title": "t",
                "note": None,
                "verdict": None,
                "evidence": None,
                "reproduces_when": None,
                "relation": None,
                "state": "rejected",
                "rejected_by": "user",
                "decided_round": 1,
                "history": [],
            }
        ],
    }
    led.write(data)
    out = led.add(_obj([_finding()]))
    assert out["findings"]["C-1"]["id"] == "F-1000"


# ---------------------------------------------------------------------------
# start-round branches
# ---------------------------------------------------------------------------


def test_start_round_first_round_is_full(led: Ledger) -> None:
    out = led.start()
    assert out == {
        "status": "new",
        "discarded": False,
        "stale_fix_plans": [],
        "round": 1,
        "mode": "full",
        "base": None,
        "reason": None,
    }
    assert led.data()["rounds"] == [{"n": 1, "head": _head(led.repo), "added": False}]


def test_start_round_unknown_head_fails(led: Ledger) -> None:
    assert led.fail("start-round", "--head", "nope-not-a-sha")["head"] == "nope-not-a-sha"


def test_start_round_discards_untracked_ledger(led: Ledger) -> None:
    led.start()
    led.add(_obj([_finding()]))
    out = led.start()
    assert out["discarded"] is True
    assert out["round"] == 1
    assert led.data()["findings"] == []


def test_start_round_discards_dirty_tracked_ledger(led: Ledger) -> None:
    led.round(_obj([_finding()]), {"F-001": ("accepted", None)})
    committed = led.path.read_text(encoding="utf-8")
    _commit(led.repo, "a.py", "two\n")
    led.start()
    led.add(_obj([_finding()], [_check("F-001")]))
    _git(led.repo, "add", "--", LEDGER_REL)  # staged counts as dirty too
    out = led.start()
    assert out["discarded"] is True
    assert out["round"] == 2
    assert json.loads(committed)["findings"] == led.data()["findings"]


def test_start_round_lists_stale_fix_plans(led: Ledger) -> None:
    stale = led.repo / "docs" / "stories" / f"QS-{ISSUE}.story_review_fix_#01.md"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("x\n")
    out = led.start()
    assert out["stale_fix_plans"] == [f"docs/stories/QS-{ISSUE}.story_review_fix_#01.md"]
    assert stale.exists()


def test_start_round_committed_unfinished_fails(led: Ledger) -> None:
    led.start()
    led.add(_obj([_finding()]))
    led.commit_ledger()  # committed with F-001 still open
    _commit(led.repo, "a.py", "two\n")
    out = led.fail("start-round", "--head", _head(led.repo))
    assert out["error"] == "round 1 committed unfinished"
    assert out["open"] == ["F-001"]


def test_start_round_no_change_same_head(led: Ledger) -> None:
    led.round(_obj())
    out = led.start()
    assert out == {
        "status": "no-change",
        "discarded": False,
        "stale_fix_plans": [],
        "round": 1,
        "mode": None,
        "base": None,
        "reason": None,
    }
    assert len(led.data()["rounds"]) == 1


def test_start_round_no_change_after_ledger_only_commit(led: Ledger) -> None:
    led.round(_obj())  # the ledger commit itself moved HEAD
    assert led.start()["status"] == "no-change"


def test_start_round_no_change_after_fix_plan_only_commit(led: Ledger) -> None:
    led.round(_obj())
    _commit(led.repo, f"docs/stories/QS-{ISSUE}.story_review_fix_#01.md", "plan\n")
    assert led.start()["status"] == "no-change"


def test_start_round_story_only_commit_is_new(led: Ledger) -> None:
    led.round(_obj())
    _commit(led.repo, f"docs/stories/QS-{ISSUE}.story.md", "story\n")
    assert led.start()["status"] == "new"


def test_start_round_force_full_on_no_change(led: Ledger) -> None:
    led.round(_obj())
    out = led.start("--force-full")
    assert (out["status"], out["round"], out["mode"], out["reason"]) == ("new", 2, "full", "forced")


def test_start_round_git_diff_error_counts_as_changed(led: Ledger) -> None:
    led.round(_obj())
    data = led.data()
    data["rounds"][-1]["head"] = "0" * 40
    led.write(data)
    led.commit_ledger()
    out = led.start()
    assert (out["status"], out["round"], out["mode"]) == ("new", 2, "full")


def test_start_round_round_two_full_round_three_delta(led: Ledger) -> None:
    led.round(_obj())
    _commit(led.repo, "a.py", "two\n")
    out2 = led.start()
    assert (out2["round"], out2["mode"], out2["base"], out2["reason"]) == (2, "full", None, None)
    head2 = _head(led.repo)
    led.add(_obj())
    led.commit_ledger()
    _commit(led.repo, "a.py", "three\n")
    out3 = led.start()
    assert (out3["round"], out3["mode"], out3["base"], out3["reason"]) == (3, "delta", head2, None)


def _two_rounds(led: Ledger) -> None:
    led.round(_obj())
    _commit(led.repo, "a.py", "two\n")
    led.round(_obj())


def test_start_round_forced_full_in_delta_range(led: Ledger) -> None:
    _two_rounds(led)
    _commit(led.repo, "a.py", "three\n")
    out = led.start("--force-full")
    assert (out["round"], out["mode"], out["base"], out["reason"]) == (3, "full", None, "forced")


def test_start_round_base_not_ancestor_is_full(led: Ledger) -> None:
    _two_rounds(led)
    first = _git(led.repo, "rev-list", "--max-parents=0", "HEAD")
    _git(led.repo, "checkout", "-q", "-b", "side", first)
    side = _commit(led.repo, "b.py", "side\n")
    _git(led.repo, "checkout", "-q", "main")
    data = led.data()
    data["rounds"][-1]["head"] = side
    led.write(data)
    led.commit_ledger()
    _commit(led.repo, "a.py", "three\n")
    out = led.start()
    assert (out["mode"], out["reason"]) == ("full", "base not an ancestor")


def test_start_round_base_unknown_is_full(led: Ledger) -> None:
    _two_rounds(led)
    data = led.data()
    data["rounds"][-1]["head"] = "0" * 40
    led.write(data)
    led.commit_ledger()
    out = led.start()
    assert (out["status"], out["mode"], out["reason"]) == ("new", "full", "base unknown")


def test_start_round_merge_commit_since_base_is_full(led: Ledger) -> None:
    _two_rounds(led)
    _git(led.repo, "checkout", "-q", "-b", "side")
    _commit(led.repo, "b.py", "side\n")
    _git(led.repo, "checkout", "-q", "main")
    _commit(led.repo, "a.py", "three\n")
    _git(led.repo, "merge", "-q", "--no-ff", "-m", "merge side", "side")
    out = led.start()
    assert (out["mode"], out["reason"]) == ("full", "merge commit since base")


# ---------------------------------------------------------------------------
# check-round
# ---------------------------------------------------------------------------


def test_check_round_open_fails(led: Ledger) -> None:
    led.start()
    led.add(_obj([_finding(), _finding("C-2")]))
    led.set("F-001", "accepted")
    out = led.fail("check-round")
    assert out["open"] == ["F-002"]


def test_check_round_clean_and_empty(led: Ledger) -> None:
    assert led.ok("check-round") == {"accepted_unchecked": []}
    led.start()
    led.add(_obj([_finding()]))
    led.set("F-001", "accepted")
    assert led.ok("check-round") == {"accepted_unchecked": []}  # decided this round → not "unchecked"


def test_check_round_reports_accepted_unchecked(led: Ledger) -> None:
    _accepted_round1(led)
    led.start()
    led.add(_obj([], mode="fallback"))
    assert led.ok("check-round") == {"accepted_unchecked": ["F-001"]}


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def test_show_json_whole_ledger(led: Ledger) -> None:
    assert led.ok("show") == {"version": 1, "rounds": [], "findings": []}
    led.round(_obj([_finding()]), {"F-001": ("accepted", None)})
    out = led.ok("show", "--format", "json")
    assert out == led.data()
    assert "file_changed" not in out["findings"][0]


def test_show_md_table(led: Ledger) -> None:
    led.round(_obj([_finding(title="pipe | inside")]), {"F-001": ("rejected", "user")})
    out = led.ok("show", "--format", "md")
    lines = out.strip().splitlines()
    assert lines[0] == "| ID | Round | Source | Sev | Where | Title | State | By | Relation |"
    assert lines[2] == (
        "| F-001 | 1 | qs-review-edge-case-hunter | must-fix | a.py:1 | pipe \\| inside | rejected | user |  |"
    )


def test_show_md_relation_column(led: Ledger) -> None:
    _scenario_superseded(led)
    out = led.ok("show", "--format", "md")
    assert "| FLIP-FLOP F-001 |" in out


def test_show_head_file_changed_true_false_null(led: Ledger) -> None:
    _commit(led.repo, "b.py", "b\n")
    led.round(
        _obj(
            [
                _finding("C-1", where="a.py:1"),
                _finding("C-2", where="b.py:4-9"),
                _finding("C-3", where="a.py and b.py"),
                _finding("C-4", where="missing.py:3"),
            ]
        ),
        {
            "F-001": ("accepted", None),
            "F-002": ("accepted", None),
            "F-003": ("accepted", None),
            "F-004": ("rejected", "user"),
        },
    )
    head = _commit(led.repo, "a.py", "changed\n")
    out = led.ok("show", "--head", head)
    by_id = {f["id"]: f for f in out["findings"]}
    assert by_id["F-001"]["file_changed"] is True
    assert "+changed" in by_id["F-001"]["diff_since_decision"]
    assert (by_id["F-002"]["file_changed"], by_id["F-002"]["diff_since_decision"]) == (False, None)
    assert (by_id["F-003"]["file_changed"], by_id["F-003"]["diff_since_decision"]) == (None, None)
    assert (by_id["F-004"]["file_changed"], by_id["F-004"]["diff_since_decision"]) == (None, None)


def test_show_head_git_error_is_null(led: Ledger) -> None:
    led.round(_obj([_finding()]), {"F-001": ("accepted", None)})
    data = led.data()
    data["rounds"][0]["head"] = "0" * 40
    led.write(data)
    out = led.ok("show", "--head", _head(led.repo))
    assert out["findings"][0]["file_changed"] is None


def test_show_head_skips_undecided(led: Ledger) -> None:
    led.start()
    led.add(_obj([_finding()]))
    out = led.ok("show", "--head", _head(led.repo))
    assert "file_changed" not in out["findings"][0]


def test_show_diff_since_decision_is_capped(led: Ledger, mod: ModuleType) -> None:
    led.round(_obj([_finding()]), {"F-001": ("accepted", None)})
    head = _commit(led.repo, "a.py", "".join(f"line {i}\n" for i in range(500)))
    diff = led.ok("show", "--head", head)["findings"][0]["diff_since_decision"]
    lines = diff.splitlines()
    assert len(lines) == mod.DIFF_CAP
    assert lines[-1].startswith("... [truncated: ")


# ---------------------------------------------------------------------------
# Exit codes, root resolution, __main__
# ---------------------------------------------------------------------------


def test_corrupt_ledger_invalid_json_exits_2(led: Ledger) -> None:
    led.path.parent.mkdir(parents=True)
    led.path.write_text("{not json", encoding="utf-8")
    assert "corrupt" in led.fail("show", code=2)["error"]


@pytest.mark.parametrize(
    "data",
    [
        {"rounds": [], "findings": []},
        {"version": 2, "rounds": [], "findings": []},
        [],
        {"version": 1, "rounds": {}, "findings": []},
    ],
)
def test_corrupt_ledger_bad_version_or_shape_exits_2(led: Ledger, data: Any) -> None:
    led.path.parent.mkdir(parents=True)
    led.path.write_text(json.dumps(data), encoding="utf-8")
    led.fail("check-round", code=2)


@pytest.mark.parametrize("argv", [[], ["bogus"], ["show"], ["show", "--issue", "x"]])
def test_argparse_errors_exit_1_with_json(mod: ModuleType, argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert mod.main(argv) == 1
    assert json.loads(capsys.readouterr().out)["error"].startswith("usage:")


def test_not_a_repo_exits_1(mod: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert mod.main(["show", "--issue", "1", "--root", str(plain)]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "not a git repo"


def test_root_defaults_to_cwd_repo_and_is_normalised(
    mod: ModuleType, _repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sub = _repo / "docs"
    sub.mkdir()
    monkeypatch.chdir(sub)
    assert mod.main(["start-round", "--issue", str(ISSUE), "--head", "HEAD"]) == 0
    assert (_repo / LEDGER_REL).exists()
    capsys.readouterr()
    assert mod.main(["show", "--issue", str(ISSUE), "--root", str(sub)]) == 0
    assert json.loads(capsys.readouterr().out)["rounds"][0]["n"] == 1


def test_script_main_entry(_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert SCRIPT_PATH.is_file()
    monkeypatch.setattr("sys.argv", [str(SCRIPT_PATH), "check-round", "--issue", str(ISSUE), "--root", str(_repo)])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
    assert excinfo.value.code == 0


# ---------------------------------------------------------------------------
# End-to-end scenarios (AC-3)
# ---------------------------------------------------------------------------


def test_scenario_1_accept_then_resolved_plus_coderabbit_repeat(led: Ledger) -> None:
    _accepted_round1(led)
    out = led.round(
        _obj([_finding(source="qs-review-coderabbit", kind="DUPLICATE", ref="F-001")], [_check("F-001", "RESOLVED")])
    )
    assert out == {"findings": {"C-1": {"id": "F-001", "effect": "duplicate"}}, "checks": {"F-001": "resolved"}}
    assert led.finding("F-001")["state"] == "resolved"
    assert len(led.data()["findings"]) == 1


def test_scenario_2_still_present_and_reopens_repeat_one_transition(led: Ledger) -> None:
    _accepted_round1(led)
    led.start()
    out = led.add(_obj([_finding(kind="REOPENS", ref="F-001")], [_check("F-001", "STILL-PRESENT")]))
    assert out == {"findings": {"C-1": {"id": "F-001", "effect": "reopened"}}, "checks": {"F-001": "merged"}}
    round2 = [h for h in led.finding("F-001")["history"] if h["round"] == 2]
    assert [h for h in round2 if h["from"] != h["to"]] == [round2[0]]
    assert round2[0]["to"] == "open"
    assert sum(h["note"].startswith("merged:") for h in round2) == 1
    led.set("F-001", "accepted")
    led.ok("check-round")


def test_scenario_3_user_reject_challenged_vs_default_skip_new(led: Ledger) -> None:
    led.round(
        _obj([_finding("C-1"), _finding("C-2", severity="nice-to-have", verdict=None, evidence=None)]),
        {"F-001": ("rejected", "user"), "F-002": ("rejected", "default-skip")},
    )
    head = _commit(led.repo, "a.py", "changed\n")
    shown = {f["id"]: f for f in led.ok("show", "--head", head)["findings"]}
    assert shown["F-001"]["file_changed"] is True and shown["F-002"]["file_changed"] is True
    led.start()
    out = led.add(
        _obj(
            [
                _finding("C-1", kind="FLIP-FLOP", ref="F-001"),
                _finding("C-2", severity="nice-to-have", verdict=None, evidence=None),  # verifier: NEW (region changed)
                _finding("C-3", severity="nice-to-have", verdict=None, evidence=None, kind="FLIP-FLOP", ref="F-002"),
            ]
        )
    )
    assert out["findings"] == {
        "C-1": {"id": "F-001", "effect": "challenged"},
        "C-2": {"id": "F-003", "effect": "new"},
        "C-3": {"id": "F-004", "effect": "new-downgraded"},
    }
    assert led.finding("F-002")["state"] == "rejected"


def test_scenario_4_crash_after_add_then_new_push_redoes_round(led: Ledger) -> None:
    led.start()
    led.add(_obj([_finding()]))  # crash: never decided nor committed
    _commit(led.repo, "a.py", "pushed\n")
    out = led.start()
    assert (out["status"], out["discarded"], out["round"], out["mode"]) == ("new", True, 1, "full")
    led.add(_obj([_finding()]))
    led.set("F-001", "accepted")
    led.ok("check-round")
    led.commit_ledger()
    assert [f["id"] for f in led.data()["findings"]] == ["F-001"]


def test_scenario_5_fallback_round_then_verified_checks_it(led: Ledger) -> None:
    _accepted_round1(led)
    led.start()
    led.add(_obj([_unverified()], mode="fallback"))
    led.set("F-002", "accepted")
    assert led.ok("check-round") == {"accepted_unchecked": ["F-001"]}
    led.commit_ledger()
    _commit(led.repo, "a.py", "three\n")
    led.start()
    led.fail("add", stdin=_obj([], [_check("F-002")]))  # F-001 must be checked too
    out = led.add(_obj([], [_check("F-001"), _check("F-002", "STILL-PRESENT")]))
    assert out["checks"] == {"F-001": "resolved", "F-002": "reopened"}


def test_scenario_6_flipflop_accept_supersedes_and_repost_redirects(led: Ledger) -> None:
    _scenario_superseded(led)
    assert led.finding("F-001")["history"][-1]["note"] == "superseded by F-002"
    out = led.round(_obj([_finding(kind="REOPENS", ref="F-001")], [_check("F-002")]), {"F-003": ("rejected", "user")})
    assert out["findings"]["C-1"] == {"id": "F-003", "effect": "flip-flop"}
    assert out["checks"] == {"F-002": "resolved"}
    assert led.finding("F-003")["relation"] == {"kind": "FLIP-FLOP", "ref": "F-002"}


# ---------------------------------------------------------------------------
# Review fix #01 (QS-453 round 1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["undo-first", "fix-first"])
def test_fix01_f001_accepting_flipflop_supersedes_open_ref_in_either_order(led: Ledger, order: str) -> None:
    """F-001: "fix A" and "undo A" can never both end up accepted."""
    _accepted_round1(led)
    led.start()
    led.add(_obj([_finding(kind="FLIP-FLOP", ref="F-001")], [_check("F-001", "STILL-PRESENT")]))
    assert led.finding("F-001")["state"] == "open"
    if order == "undo-first":
        assert led.set("F-002", "accepted")["superseded"] == "F-001"
        assert led.finding("F-001")["history"][-1]["from"] == "open"
        led.fail("set", "--id", "F-001", "--to", "accepted")  # no longer decidable
    else:
        led.set("F-001", "accepted")
        assert led.set("F-002", "accepted")["superseded"] == "F-001"
    assert led.finding("F-001")["state"] == "superseded"
    assert led.ok("check-round") == {"accepted_unchecked": []}


def test_fix01_f002_redirect_follows_supersede_chain(led: Ledger) -> None:
    """F-002: fix A → undo A → redo A; a re-raise of A targets the live head of the chain."""
    _scenario_superseded(led)  # F-002 supersedes F-001
    led.round(
        _obj([_finding(kind="FLIP-FLOP", ref="F-002", where="a.py:4")], [_check("F-002")]),
        {"F-003": ("accepted", None)},
    )
    assert led.finding("F-002")["state"] == "superseded"
    _commit(led.repo, "a.py", "four\n")
    led.start()
    out = led.add(_obj([_finding(kind="REOPENS", ref="F-001")], [_check("F-003")]))
    assert out["findings"]["C-1"] == {"id": "F-004", "effect": "flip-flop"}
    assert led.finding("F-004")["relation"] == {"kind": "FLIP-FLOP", "ref": "F-003"}


def test_fix01_f002_chain_cycle_without_live_head_downgrades(led: Ledger) -> None:
    head = _head(led.repo)
    base = {
        "round": 1,
        "source": "s",
        "reported_severity": "must-fix",
        "severity_action": "keep",
        "severity": "must-fix",
        "where": "a.py:1",
        "title": "t",
        "note": None,
        "verdict": None,
        "evidence": None,
        "reproduces_when": None,
        "state": "superseded",
        "rejected_by": None,
        "decided_round": 1,
        "history": [],
    }
    led.write(
        {
            "version": 1,
            "rounds": [{"n": 1, "head": head, "added": False}],
            "findings": [
                {**base, "id": "F-001", "relation": {"kind": "FLIP-FLOP", "ref": "F-002"}},
                {**base, "id": "F-002", "relation": {"kind": "FLIP-FLOP", "ref": "F-001"}},
            ],
        }
    )
    out = led.add(_obj([_finding(kind="REOPENS", ref="F-001")]))
    assert out["findings"]["C-1"]["effect"] == "new-downgraded"


def test_fix01_f003_set_recovers_committed_unfinished_round(led: Ledger) -> None:
    led.start()
    led.add(_obj([_finding(), _finding("C-2")]))
    led.set("F-001", "accepted")
    led.commit_ledger()  # committed outside the protocol: F-002 still open
    _commit(led.repo, "a.py", "two\n")
    err = led.fail("start-round", "--head", _head(led.repo))
    assert err["open"] == ["F-002"]
    assert "review_ledger.py set" in err["hint"]
    led.fail("add", stdin=_obj())  # still no second add
    led.set("F-002", "rejected", "user")
    assert led.ok("check-round") == {"accepted_unchecked": []}
    led.commit_ledger()
    assert led.start()["round"] == 2


def test_fix01_f003_set_on_clean_finished_ledger_still_fails(led: Ledger) -> None:
    led.round(_obj([_finding()]), {"F-001": ("accepted", None)})
    led.fail("set", "--id", "F-001", "--to", "accepted")


@pytest.mark.parametrize("tracked", [True, False])
def test_fix01_f004_discard_failure_exits_1(led: Ledger, tracked: bool) -> None:
    if tracked:
        led.round(_obj())
        _commit(led.repo, "a.py", "two\n")
    led.start()
    lock = led.repo / ".git" / "index.lock"
    lock.write_text("")
    try:
        err = led.fail("start-round", "--head", _head(led.repo))
    finally:
        lock.unlink()
    assert err["error"] == "could not discard the unfinished round"
    assert led.data()["rounds"][-1]["added"] is False  # untouched


def test_fix01_f005_demoted_repeat_of_demoted_finding_stays_duplicate(led: Ledger) -> None:
    led.round(_obj([_finding(reproduces_when=None, severity_action="demote")]), {"F-001": ("rejected", "verifier")})
    led.start("--force-full")
    out = led.add(_obj([_finding(reproduces_when=None, severity_action="demote", kind="DUPLICATE", ref="F-001")]))
    assert out["findings"]["C-1"] == {"id": "F-001", "effect": "duplicate"}


@pytest.mark.parametrize("inject", [False, True])
def test_fix01_f008_show_unknown_head_exits_1(led: Ledger, tmp_path: Path, inject: bool) -> None:
    led.round(_obj([_finding()]), {"F-001": ("accepted", None)})
    target = tmp_path / "injected"
    head = f"--output={target}" if inject else "deadbeef"
    assert led.fail("show", f"--head={head}")["head"] == head
    assert not target.exists()  # never reaches git diff as an option


# ---------------------------------------------------------------------------
# Review fix #02 (QS-453 round 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("keys", "missing", "extra"), [("C-1,C-2", ["C-2"], []), ("", [], ["C-1"])])
def test_fix02_f015_add_rejects_a_findings_key_set_mismatch(
    led: Ledger, keys: str, missing: list[str], extra: list[str]
) -> None:
    led.start()
    err = led.fail("add", f"--keys={keys}", stdin=_obj([_finding("C-1")]))
    assert (err["missing"], err["extra"]) == (missing, extra)
    assert led.data()["rounds"][-1]["added"] is False


def test_fix02_f015_add_requires_keys_and_rejects_duplicates(led: Ledger, mod: ModuleType) -> None:
    led.start()
    assert mod.main(["add", "--issue", str(ISSUE), "--root", str(led.repo)]) == 1
    led.fail("add", "--keys=C-1,C-1", stdin=_obj([_finding("C-1")]))


def test_fix02_f015_add_accepts_matching_keys_in_any_order(led: Ledger) -> None:
    led.start()
    out = led.ok("add", "--keys= C-2 , C-1 ", stdin=_obj([_finding("C-1"), _finding("C-2")]))
    assert set(out["findings"]) == {"C-1", "C-2"}


def test_fix02_f028_stale_fix_plans_include_staged_uncommitted(led: Ledger) -> None:
    led.round(_obj())
    _commit(led.repo, "a.py", "two\n")
    led.start()
    led.add(_obj())
    plan = f"docs/stories/QS-{ISSUE}.story_review_fix_#01.md"
    (led.repo / plan).write_text("plan\n")
    _git(led.repo, "add", "--", LEDGER_REL, plan)  # step 8 staged, then the commit failed
    out = led.start()
    assert out["discarded"] is True
    assert out["stale_fix_plans"] == [plan]


def test_fix02_f029_start_round_keeps_uncommitted_recovery_decisions(led: Ledger) -> None:
    led.start()
    led.add(_obj([_finding(), _finding("C-2")]))
    led.commit_ledger()  # committed unfinished: F-001 and F-002 open
    led.set("F-001", "accepted")  # recovery decision, not committed yet
    err = led.fail("start-round", "--head", _head(led.repo))
    assert err["open"] == ["F-002"]
    assert err["discarded"] is False
    assert led.finding("F-001")["state"] == "accepted"  # kept


def test_fix02_f030_check_round_fails_before_add(led: Ledger) -> None:
    led.start()
    err = led.fail("check-round")
    assert "add" in err["error"]


# ---------------------------------------------------------------------------
# Review fix #03 (QS-453 round 3)
# ---------------------------------------------------------------------------


def test_fix03_f033_duplicate_finding_key_reaches_validate(led: Ledger) -> None:
    led.start()
    err = led.fail("add", stdin=_VALIDATION_CASES["duplicate_finding_key"])
    assert err["error"] == "findings[1]: duplicate finding key"


def test_fix03_f032_empty_keys_records_a_checks_only_round(led: Ledger) -> None:
    _accepted_round1(led)
    led.start()
    out = led.ok("add", "--keys", "", stdin=_obj([], [_check("F-001")]))
    assert out == {"findings": {}, "checks": {"F-001": "resolved"}}


def _committed_unfinished(led: Ledger) -> None:
    led.start()
    led.add(_obj([_finding(), _finding("C-2")]))
    led.commit_ledger()  # committed with F-001 and F-002 open


@pytest.mark.parametrize("working", ["deleted", "no-rounds", "corrupt"])
def test_fix03_f035_f036_unusable_working_copy_is_restored_not_kept(led: Ledger, working: str) -> None:
    _committed_unfinished(led)
    if working == "deleted":
        led.path.unlink()
    elif working == "no-rounds":
        led.write({"version": 1, "rounds": [], "findings": []})
    else:
        led.path.write_text("{truncated", encoding="utf-8")
    err = led.fail("start-round", "--head", _head(led.repo))
    assert err["error"] == "round 1 committed unfinished"  # the committed state, restored
    assert err["discarded"] is True
    assert err["open"] == ["F-001", "F-002"]


# ---------------------------------------------------------------------------
# Review fix #04 (QS-453 round 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recovering", [False, True])
def test_fix04_f043_committed_unfinished_error_carries_round(led: Ledger, recovering: bool) -> None:
    _committed_unfinished(led)
    if recovering:
        led.set("F-001", "accepted")  # uncommitted recovery decision
    else:
        _commit(led.repo, "a.py", "two\n")
    assert led.fail("start-round", "--head", _head(led.repo))["round"] == 1


# ---------------------------------------------------------------------------
# Review fix #06 (QS-453 round 6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("new_code", [False, True])
@pytest.mark.parametrize("recovering", [False, True])
def test_fix06_f050_committed_unfinished_reports_changed_since_round(
    led: Ledger, new_code: bool, recovering: bool
) -> None:
    _committed_unfinished(led)
    if new_code:
        _commit(led.repo, "a.py", "new code\n")
    if recovering:
        led.set("F-001", "accepted")
    err = led.fail("start-round", "--head", _head(led.repo))
    assert err["changed_since_round"] is new_code
    assert "start-round again" not in err["hint"]


# ---------------------------------------------------------------------------
# Review fix #08 (QS-453 round 8)
# ---------------------------------------------------------------------------


def test_fix08_f059_recovery_hint_points_to_step1_without_restating_routing(mod: ModuleType) -> None:
    hint = mod._RECOVERY_HINT
    assert "step 1" in hint
    for rule in ("finish-task", "restart", "start-round again", "fix plan"):  # routing lives in the template only
        assert rule not in hint, rule
