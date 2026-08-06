"""Tests for ``scripts/qs/setup_task.py`` — declaration validation (QS-332 B2).

The truth table lives in ``test_targets.py``; these pin only the wiring:
``setup_task`` refuses an undeclared/inconsistent issue BEFORE any
branch/worktree work, with the shape-aware backfill command carrying the
real issue number.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest


def _gh_labels_response(labels: list[str]) -> str:
    return json.dumps({"labels": [{"name": name} for name in labels]})


def _make_fake_run(labels: list[str] | None, *, gh_rc: int = 0):
    """Return ``(fake_run, seen_cmds)``; ``labels=None`` + ``gh_rc`` fakes a gh failure."""
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(list(cmd))
        if cmd[:3] == ["gh", "issue", "view"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=gh_rc,
                stdout=_gh_labels_response(labels or []),
                stderr="" if gh_rc == 0 else "boom",
            )
        raise AssertionError(f"unexpected command after refusal: {cmd}")

    return fake_run, seen


def test_complete_task_declaration_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    import setup_task
    import utils

    fake_run, _seen = _make_fake_run(["kind:feature", "target:factory", "scale:task"])
    monkeypatch.setattr(utils, "run", fake_run)
    setup_task.check_declaration(332)  # returns without exiting


def test_epic_declaration_validates_as_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """An epic-shaped declaration is not asked to grow a kind (story D3).

    Declaration validity and the epic *worktree* refusal are deliberately
    separate steps: the declaration IS valid — what an epic must not get
    is a branch/worktree (see the epic-refusal tests below).
    """
    import setup_task
    import utils

    fake_run, _seen = _make_fake_run(["scale:epic", "target:product", "pinned"])
    monkeypatch.setattr(utils, "run", fake_run)
    setup_task.check_declaration(321)


# ---------------------------------------------------------------------------
# Review-fix #04 (must-fix): epic ⇒ NO branch, NO worktree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv_extra", [[], ["--no-worktree"]], ids=["worktree", "no-worktree"]
)
def test_epic_issue_is_refused_before_any_git_work(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv_extra: list[str],
) -> None:
    """The epic model (QS-321): "No implement phase; no branch, worktree,
    or PR" — an epic's output is a rationale doc on `main` + child issues.

    Machine-enforced here rather than prompt-obeyed, and enforced for
    `--no-worktree` too: that flag still cuts a BRANCH, which the model
    also forbids. The fake raises on any non-`gh` command, so this also
    proves nothing git-side ran.
    """
    import setup_task
    import utils

    fake_run, seen = _make_fake_run(["scale:epic", "target:factory"])
    monkeypatch.setattr(utils, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["setup_task.py", "321", *argv_extra])

    with pytest.raises(SystemExit) as exc:
        setup_task.main()
    assert exc.value.code == 1
    assert all(cmd[0] == "gh" for cmd in seen)

    out = json.loads(capsys.readouterr().out)
    assert out["scale"] == "epic"
    assert "worktree" in out["error"]
    # Actionable: says what an epic DOES produce instead.
    assert "child" in out["detail"]


def test_task_issue_is_not_refused_as_an_epic(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard is scale-specific — a task passes it untouched."""
    import setup_task
    import utils

    fake_run, _seen = _make_fake_run(["kind:bug", "target:product", "scale:task"])
    monkeypatch.setattr(utils, "run", fake_run)
    labels = setup_task.check_declaration(42)
    setup_task.refuse_if_epic(42, labels)  # returns without exiting


def test_check_declaration_returns_the_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """The epic guard reuses `check_declaration`'s already-fetched labels
    — no second `gh` call on the setup path."""
    import setup_task
    import utils

    fake_run, seen = _make_fake_run(["kind:feature", "target:factory", "scale:task"])
    monkeypatch.setattr(utils, "run", fake_run)
    labels = setup_task.check_declaration(332)
    assert labels == ["kind:feature", "target:factory", "scale:task"]
    assert len([c for c in seen if c[:3] == ["gh", "issue", "view"]]) == 1


def test_undeclared_issue_refuses_with_backfill_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import setup_task
    import utils

    fake_run, _seen = _make_fake_run(["bug"])
    monkeypatch.setattr(utils, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        setup_task.check_declaration(42)
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert "error" in out
    assert set(out["missing"]) == {"kind", "target", "scale"}
    # Shape-aware, actionable, with the REAL issue number substituted.
    assert "gh issue edit 42 --add-label" in out["detail"]
    assert "<N>" not in out["detail"]


def test_conflicting_declaration_refuses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import setup_task
    import utils

    fake_run, _seen = _make_fake_run(
        ["kind:bug", "target:product", "target:factory", "scale:task"]
    )
    monkeypatch.setattr(utils, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        setup_task.check_declaration(42)
    assert exc.value.code == 1
    assert "target" in json.loads(capsys.readouterr().out)["missing"]


def test_null_labels_reports_the_declaration_error_not_invalid_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review-fix #03: a valid-JSON response with ``"labels": null`` hit
    the broad ``except TypeError`` and reported the misleading "Invalid
    JSON from gh CLI". It is valid JSON — the honest verdict is the
    ordinary missing-declaration refusal, with its actionable backfill
    command."""
    import setup_task
    import utils

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert cmd[:3] == ["gh", "issue", "view"]
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps({"labels": None}), stderr=""
        )

    monkeypatch.setattr(utils, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        setup_task.check_declaration(42)
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert "Invalid JSON" not in out["error"]
    assert "no complete lane declaration" in out["error"]
    assert "gh issue edit 42 --add-label" in out["detail"]


@pytest.mark.parametrize("raw", ["null", "[]", "42"])
def test_non_dict_json_refuses_with_the_structured_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], raw: str
) -> None:
    """Review-fix #04: a non-dict top-level value raised `AttributeError`
    out of the except tuple as a raw traceback."""
    import setup_task
    import utils

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=raw, stderr="")

    monkeypatch.setattr(utils, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        setup_task.check_declaration(42)
    assert exc.value.code == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_gh_failure_refuses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import setup_task
    import utils

    fake_run, _seen = _make_fake_run(None, gh_rc=1)
    monkeypatch.setattr(utils, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        setup_task.check_declaration(42)
    assert exc.value.code == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_main_refuses_before_any_git_work(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The declaration check is enforcement BY CONSTRUCTION: it runs before
    the fetch/branch/worktree machinery, so a refused issue never touches
    git (the fake raises on any non-``gh issue view`` command).
    """
    import setup_task
    import utils

    fake_run, seen = _make_fake_run([])
    monkeypatch.setattr(utils, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["setup_task.py", "42", "--no-worktree"])
    with pytest.raises(SystemExit) as exc:
        setup_task.main()
    assert exc.value.code == 1
    assert all(cmd[0] == "gh" for cmd in seen)
    assert "gh issue edit 42 --add-label" in json.loads(capsys.readouterr().out)["detail"]
