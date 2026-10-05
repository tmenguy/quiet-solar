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


@pytest.mark.parametrize(
    ("next_cmd", "expected"),
    [
        ("create-plan", "create-plan"),
        ("/create-plan", "create-plan"),
        ("/decompose-epic", "decompose-epic"),
        ("//decompose-epic", "/decompose-epic"),  # stays refused
    ],
)
def test_phase_strips_exactly_one_leading_slash(next_cmd: str, expected: str) -> None:
    """N7: one shared normaliser for the three former inline slices."""
    import setup_task

    assert setup_task._phase(next_cmd) == expected


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


# ---------------------------------------------------------------------------
# QS-357: the render hook fails loudly at worktree birth
# ---------------------------------------------------------------------------


def _fake_run_success(labels: list[str]):
    """A fake ``run`` that answers the gh label lookup and every git call OK."""

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gh", "issue", "view"]:
            return subprocess.CompletedProcess(cmd, 0, _gh_labels_response(labels), "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return fake_run


def test_render_import_error_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    """A missing ``jinja2`` (function-local ImportError) → JSON error + exit 1."""
    import sys

    import setup_task

    import utils

    monkeypatch.setattr(utils, "run", _fake_run_success(["kind:feature", "target:factory", "scale:task"]))
    monkeypatch.setattr(setup_task, "get_main_worktree", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "render_agents", None)  # import → ImportError
    monkeypatch.setattr("sys.argv", ["setup_task.py", "42", "--no-worktree", "--title", "T"])

    with pytest.raises(SystemExit) as exc:
        setup_task.main()
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["error"] == "agent render failed"
    assert "render_agents.py --work-dir" in out["detail"]
    assert "next_step.py --next-cmd create-plan" in out["detail"]


def test_render_render_error_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    """A ``RenderError`` from ``render_all`` → JSON error + exit 1."""
    import render_agents
    import setup_task

    import utils

    def _boom(*_a: Any, **_k: Any) -> None:
        raise render_agents.RenderError("boom")

    monkeypatch.setattr(utils, "run", _fake_run_success(["kind:feature", "target:factory", "scale:task"]))
    monkeypatch.setattr(setup_task, "get_main_worktree", lambda: tmp_path)
    monkeypatch.setattr(render_agents, "render_all", _boom)
    monkeypatch.setattr("sys.argv", ["setup_task.py", "42", "--no-worktree", "--title", "T"])

    with pytest.raises(SystemExit) as exc:
        setup_task.main()
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["error"] == "agent render failed"
    assert "boom" in out["detail"]


# ---------------------------------------------------------------------------
# QS-358: setup_task forwards the labels' lane to the launcher
# ---------------------------------------------------------------------------


def test_setup_task_passes_labels_lane(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    import render_agents
    import setup_task
    import targets

    import utils

    labels = ["kind:bug", "target:product", "scale:task"]
    seen: dict = {}

    class CapturingLauncher:
        @staticmethod
        def build_payload(*_args: Any, **kwargs: Any) -> dict:
            seen.update(kwargs)
            return {"tool": "fake"}

    monkeypatch.setattr(utils, "run", _fake_run_success(labels))
    monkeypatch.setattr(setup_task, "get_main_worktree", lambda: tmp_path)
    monkeypatch.setattr(render_agents, "render_all", lambda *a, **k: [])
    monkeypatch.setitem(setup_task.LAUNCHERS, "claude-code", CapturingLauncher)
    monkeypatch.setattr(
        "sys.argv",
        ["setup_task.py", "42", "--no-worktree", "--title", "T", "--harness", "claude-code"],
    )

    setup_task.main()
    assert seen["lane"] == targets.parse_axes(labels)["lane"] == "bug-product"
    assert json.loads(capsys.readouterr().out)["tool"] == "fake"


# ---------------------------------------------------------------------------
# QS-340: the epic × factory lane gets a short-lived docs-only worktree
# ---------------------------------------------------------------------------

_EPIC_FACTORY = ["target:factory", "scale:epic"]


@pytest.mark.parametrize("next_cmd", ["/decompose-epic", "decompose-epic"])
def test_epic_factory_with_decompose_epic_is_allowed(next_cmd: str) -> None:
    import setup_task

    setup_task.refuse_if_epic(340, _EPIC_FACTORY, next_cmd)  # returns without exiting


@pytest.mark.parametrize(
    ("labels", "next_cmd", "no_worktree"),
    [
        (_EPIC_FACTORY, "/create-plan", False),
        (_EPIC_FACTORY, "//decompose-epic", False),
        (["target:product", "scale:epic"], "/decompose-epic", False),
        (_EPIC_FACTORY, "/decompose-epic", True),
    ],
    ids=["wrong-next-cmd", "double-slash", "product-epic", "no-worktree"],
)
def test_epic_refusals(
    capsys: pytest.CaptureFixture[str],
    labels: list[str],
    next_cmd: str,
    no_worktree: bool,
) -> None:
    import setup_task

    with pytest.raises(SystemExit) as exc:
        setup_task.refuse_if_epic(340, labels, next_cmd, no_worktree=no_worktree)
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["scale"] == "epic"
    assert "worktree" in out["error"]
    assert "child" in out["detail"]
    assert "### Parent epic" in out["detail"]
    assert "#339" in out["detail"]


@pytest.mark.parametrize("next_cmd", ["/decompose-epic", "decompose-epic"])
@pytest.mark.parametrize(
    "labels",
    [
        ["kind:feature", "target:factory", "scale:task"],
        ["kind:bug", "target:product", "scale:task"],
    ],
    ids=["feature-factory", "bug-product"],
)
def test_decompose_epic_refused_for_a_non_epic_issue(
    capsys: pytest.CaptureFixture[str], labels: list[str], next_cmd: str
) -> None:
    """S6: a non-epic task routed to decompose-epic is refused before any git work."""
    import setup_task

    with pytest.raises(SystemExit) as exc:
        setup_task.refuse_decompose_epic_for_wrong_lane(42, labels, next_cmd)
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert "decompose-epic" in out["error"]
    assert out["lane"] != "epic-factory"
    assert "epic × factory" in out["detail"]


@pytest.mark.parametrize("next_cmd", ["/decompose-epic", "decompose-epic"])
def test_decompose_epic_allowed_for_epic_factory(next_cmd: str) -> None:
    """S6: the epic × factory lane still passes the non-epic guard."""
    import setup_task

    setup_task.refuse_decompose_epic_for_wrong_lane(340, _EPIC_FACTORY, next_cmd)


@pytest.mark.parametrize("next_cmd", ["/create-plan", "create-plan", "/implement-task"])
def test_non_decompose_next_cmd_is_never_touched_by_the_guard(next_cmd: str) -> None:
    """S6: the guard only concerns ``decompose-epic``; other phases pass through."""
    import setup_task

    setup_task.refuse_decompose_epic_for_wrong_lane(
        42, ["kind:feature", "target:factory", "scale:task"], next_cmd
    )


def test_epic_default_next_cmd_is_refused_through_main(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` forwards ``--next-cmd`` / ``--no-worktree`` to the guard."""
    import setup_task

    import utils

    fake_run, seen = _make_fake_run(_EPIC_FACTORY)
    monkeypatch.setattr(utils, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv", ["setup_task.py", "340", "--next-cmd", "/decompose-epic", "--no-worktree"]
    )
    with pytest.raises(SystemExit) as exc:
        setup_task.main()
    assert exc.value.code == 1
    assert all(cmd[0] == "gh" for cmd in seen)


@pytest.mark.parametrize("next_cmd", ["/decompose-epic", "decompose-epic"])
def test_fail_render_remedy_carries_the_real_next_cmd(
    capsys: pytest.CaptureFixture[str], next_cmd: str
) -> None:
    import setup_task

    with pytest.raises(SystemExit) as exc:
        setup_task._fail_render(ImportError("no jinja2"), "/wd", 340, "T", next_cmd)
    assert exc.value.code == 1
    detail = json.loads(capsys.readouterr().out)["detail"]
    assert "next_step.py --next-cmd decompose-epic --work-dir /wd" in detail
    assert "--next-cmd /decompose-epic" not in detail


def test_render_failure_remedy_uses_the_passed_next_cmd(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    import sys

    import setup_task

    import utils

    monkeypatch.setattr(utils, "run", _fake_run_success(["kind:bug", "target:product", "scale:task"]))
    monkeypatch.setattr(setup_task, "get_main_worktree", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "render_agents", None)
    monkeypatch.setattr(
        "sys.argv",
        ["setup_task.py", "42", "--no-worktree", "--title", "T", "--next-cmd", "/diagnose-task"],
    )
    with pytest.raises(SystemExit):
        setup_task.main()
    assert "next_step.py --next-cmd diagnose-task" in json.loads(capsys.readouterr().out)["detail"]


# ---------------------------------------------------------------------------
# QS-400: --item K cuts a work-item worktree (no fetch, no pin, unbound render)
# ---------------------------------------------------------------------------

_TASK = ["kind:feature", "target:factory", "scale:task"]


class _ExplodingLauncher:
    @staticmethod
    def build_payload(*_args: Any, **_kwargs: Any) -> dict:
        raise AssertionError("an item gets no launcher payload")


def _item_fakes(monkeypatch: pytest.MonkeyPatch, tmp_path, labels: list[str]) -> dict[str, Any]:
    """Fake every collaborator of ``setup_task.main()`` on the item path; return the record."""
    from pathlib import Path

    import render_agents
    import setup_task

    import utils

    rec: dict[str, Any] = {"git": [], "script": [], "wt_dir": [], "render": []}

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gh", "issue", "view"]:
            return subprocess.CompletedProcess(cmd, 0, _gh_labels_response(labels), "")
        rec["git"].append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def fake_subprocess_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        rec["script"].append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    def fake_get_worktree_dir(issue: int, item: int | None = None) -> Path:
        rec["wt_dir"].append((issue, item))
        return tmp_path / f"wt-{issue}-{item}"

    def fake_build_render_context(work_dir: str, **kwargs: Any) -> dict:
        rec["render"].append((work_dir, kwargs))
        return {"ctx": True}

    monkeypatch.setattr(utils, "run", fake_run)
    monkeypatch.setattr(setup_task, "get_main_worktree", lambda: tmp_path)
    monkeypatch.setattr(setup_task.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(setup_task, "get_worktree_dir", fake_get_worktree_dir)
    monkeypatch.setattr(render_agents, "build_render_context", fake_build_render_context)
    monkeypatch.setattr(render_agents, "render_all", lambda *a, **k: [])
    monkeypatch.setitem(setup_task.LAUNCHERS, "claude-code", _ExplodingLauncher)
    return rec


def test_item_cuts_an_item_worktree(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    import setup_task

    rec = _item_fakes(monkeypatch, tmp_path, _TASK)
    monkeypatch.setattr(
        "sys.argv",
        ["setup_task.py", "42", "--item", "3", "--harness", "claude-code", "--next-cmd", "/review-task"],
    )
    setup_task.main()

    assert not any(cmd[:2] == ["git", "fetch"] for cmd in rec["git"])
    assert rec["git"] == []
    assert len(rec["script"]) == 1
    assert rec["script"][0][0] == "bash"
    assert rec["script"][0][1].endswith("scripts/worktree-setup.sh")
    assert rec["script"][0][2:] == ["42", "3"]
    assert rec["wt_dir"] == [(42, 3)]
    assert len(rec["render"]) == 1
    assert rec["render"][0][0] == str(tmp_path / "wt-42-3")
    assert rec["render"][0][1]["bound"] is False
    out = json.loads(capsys.readouterr().out)
    assert out == {
        "issue_number": 42,
        "item": 3,
        "branch": "QS_42_3",
        "worktree_path": str(tmp_path / "wt-42-3"),
        "no_worktree": False,
        "harness": "claude-code",
    }


def test_item_worktree_setup_failure_is_a_json_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    import setup_task

    _item_fakes(monkeypatch, tmp_path, _TASK)
    monkeypatch.setattr(
        setup_task.subprocess,
        "run",
        lambda cmd, **_k: subprocess.CompletedProcess(
            cmd, 1, "Error: deliverable branch QS_42 not found\n", "warning: some git noise\n"
        ),
    )
    monkeypatch.setattr("sys.argv", ["setup_task.py", "42", "--item", "3", "--harness", "claude-code"])
    with pytest.raises(SystemExit) as exc:
        setup_task.main()
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["error"] == "Worktree setup failed"
    assert "QS_42 not found" in out["detail"]


def test_item_with_no_worktree_is_refused_before_any_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import setup_task

    import utils

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("no gh / git call expected")

    monkeypatch.setattr(utils, "run", boom)
    monkeypatch.setattr(setup_task.subprocess, "run", boom)
    monkeypatch.setattr("sys.argv", ["setup_task.py", "42", "--item", "1", "--no-worktree"])
    with pytest.raises(SystemExit) as exc:
        setup_task.main()
    assert exc.value.code == 1
    assert "--item" in json.loads(capsys.readouterr().out)["error"]


@pytest.mark.parametrize("next_cmd", [None, "/decompose-epic", "decompose-epic"])
def test_item_on_an_epic_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path, next_cmd: str | None
) -> None:
    import setup_task

    rec = _item_fakes(monkeypatch, tmp_path, _EPIC_FACTORY)
    argv = ["setup_task.py", "340", "--item", "1"]
    if next_cmd is not None:
        argv += ["--next-cmd", next_cmd]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as exc:
        setup_task.main()
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["scale"] == "epic"
    assert rec["git"] == [] and rec["script"] == []


@pytest.mark.parametrize("raw", ["0", "01", "x", "-1"])
def test_item_rejects_a_bad_k(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    import setup_task

    monkeypatch.setattr("sys.argv", ["setup_task.py", "42", "--item", raw])
    with pytest.raises(SystemExit) as exc:
        setup_task.main()
    assert exc.value.code == 2


def test_item_render_failure_names_only_render_agents(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    import render_agents
    import setup_task

    _item_fakes(monkeypatch, tmp_path, _TASK)

    def _boom(*_a: Any, **_k: Any) -> None:
        raise render_agents.RenderError("boom")

    monkeypatch.setattr(render_agents, "render_all", _boom)
    monkeypatch.setattr("sys.argv", ["setup_task.py", "42", "--item", "2", "--harness", "claude-code"])
    with pytest.raises(SystemExit) as exc:
        setup_task.main()
    assert exc.value.code == 1
    detail = json.loads(capsys.readouterr().out)["detail"]
    assert "render_agents.py --work-dir" in detail
    assert "next_step.py" not in detail
    assert "boom" in detail


def test_fail_render_task_text_unchanged_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    import setup_task

    with pytest.raises(SystemExit):
        setup_task._fail_render(ImportError("x"), "/wd", 1, "T", "/create-plan")
    assert "next_step.py --next-cmd create-plan" in json.loads(capsys.readouterr().out)["detail"]
