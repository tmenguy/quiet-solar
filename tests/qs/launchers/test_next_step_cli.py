"""Tests for ``scripts/qs/next_step.py`` invoked as a subprocess.

``next_step.py`` is stdlib-only and must work from any CWD (AC-9). The
phase-name → agent-name resolution is a static dict — no filesystem
scan, no ``Path.cwd()`` — so running from ``/tmp`` resolves the same as
running from a worktree.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_QS = Path(__file__).resolve().parents[3] / "scripts" / "qs"
NEXT_STEP_PY = SCRIPTS_QS / "next_step.py"


def _run(args: list[str], *, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``next_step.py`` with the given args."""
    return subprocess.run(
        [sys.executable, str(NEXT_STEP_PY), *args],
        capture_output=True,
        text=True,
        cwd=cwd or str(SCRIPTS_QS),
    )


def test_valid_phase_emits_payload_and_exits_zero(tmp_path: Path) -> None:
    """Valid phase → JSON payload on stdout, exit 0. CWD-independent."""
    result = _run(
        [
            "--next-cmd", "create-plan",
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Fix bug",
            "--harness", "claude-code",
        ],
        cwd=str(tmp_path),  # outside the repo — AC-9
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["harness"] == "claude-code"
    assert payload["agent"] == "qs-create-plan"
    assert payload["same_context"] == "create-plan"
    assert payload["new_context"].startswith("sh ")


def test_slash_form_accepted_for_back_compat(tmp_path: Path) -> None:
    """Slash form continues to work — old callers pass ``/create-plan``."""
    result = _run(
        [
            "--next-cmd", "/create-plan",
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Fix bug",
            "--harness", "claude-code",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["agent"] == "qs-create-plan"


def test_unknown_phase_emits_error_json_and_exits_nonzero(tmp_path: Path) -> None:
    """Unknown phase → JSON error on stdout, non-zero exit."""
    result = _run(
        [
            "--next-cmd", "bogus-phase",
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Fix bug",
            "--harness", "claude-code",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["error"] == "unknown phase"
    assert payload["value"] == "bogus-phase"
    assert "create-plan" in payload["known"]
    assert "release" in payload["known"]


@pytest.mark.parametrize("phase", [
    "setup-task",
    "create-plan",
    "implement-task",
    "implement-setup-task",
    "review-task",
    "finish-task",
    "release",
])
def test_every_known_phase_resolves(phase: str, tmp_path: Path) -> None:
    """Every known phase resolves end-to-end via the CLI."""
    result = _run(
        [
            "--next-cmd", phase,
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Title",
            "--harness", "claude-code",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["agent"] == f"qs-{phase}"


# --------------------------------------------------------------------------- #
# Free-form harness (codex) must NOT be regressed by the strict
# claude/opencode validation. The codex launcher carries no agent mapping
# today, so next_step.py must let it pass any --next-cmd value through
# unchanged. Regression catch for review-fix #1 + #5.
#
# OpenCode used to be in this list, but with the new static-agent
# pipeline (QS-177) opencode now resolves agents like claude —
# unknown phases raise UnknownPhaseError and emit the
# `{"error": "unknown phase", ...}` JSON contract. See
# `test_opencode_rejects_unknown_phase` and `test_opencode_happy_path`
# below for the new pins.
# --------------------------------------------------------------------------- #


def test_codex_accepts_free_form_next_cmd(tmp_path: Path) -> None:
    """Codex launcher must accept any --next-cmd string (no agent mapping)."""
    result = _run(
        [
            "--next-cmd", "anything-goes-here",
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Title",
            "--harness", "codex",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["tool"] == "codex"
    assert payload["same_context"] == "anything-goes-here"


def test_opencode_rejects_unknown_phase(tmp_path: Path) -> None:
    """OpenCode now resolves agents like claude — unknown phase → JSON error, exit 1.

    Contract change from the legacy pipeline (QS-177 Task 7.3). The
    OpenCode launcher is no longer a free-form passthrough; it enforces
    the same phase mapping as claude.

    AC #4 mandates exit code **1 specifically** (parity with
    claude) AND a ``known: [...]`` key in the JSON error
    payload — both pinned here (review fix #01 must-fix #3).
    """
    result = _run(
        [
            "--next-cmd", "bogus",
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Title",
            "--harness", "opencode",
        ],
        cwd=str(tmp_path),
    )
    # AC #4 — exit 1 (not just non-zero) to match the claude
    # contract; ``2`` is reserved for argparse user errors.
    assert result.returncode == 1, result.stderr
    payload = json.loads(result.stdout)
    assert payload["error"] == "unknown phase"
    assert payload["value"] == "bogus"
    # AC #4 — ``known`` list must surface so the user can see what
    # they meant to type (and downstream tooling can render hints).
    assert "known" in payload
    assert isinstance(payload["known"], list)
    assert len(payload["known"]) > 0
    assert "create-plan" in payload["known"]


def test_opencode_happy_path(tmp_path: Path) -> None:
    """OpenCode happy path → JSON payload with spawn_session invocation, exit 0.

    QS-177 Task 7.4 — the new opencode launcher emits a
    ``python scripts/qs/spawn_session.py …`` ``new_context`` for
    ``caller='next_step'`` (the default when dispatched via
    ``next_step.py``).
    """
    result = _run(
        [
            "--next-cmd", "create-plan",
            "--work-dir", "/tmp/wt",
            "--issue", "42",
            "--title", "Foo",
            "--harness", "opencode",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["tool"] == "opencode"
    assert payload["agent"] == "qs-create-plan"
    assert payload["same_context"] == "create-plan"
    assert payload["new_context"].startswith("python scripts/qs/spawn_session.py")


def test_codex_passes_known_phase_through_unchanged(tmp_path: Path) -> None:
    """Even a known phase under codex stays free-form — no agent key added."""
    result = _run(
        [
            "--next-cmd", "create-plan",
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Title",
            "--harness", "codex",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["tool"] == "codex"
    # codex launcher is a stub; it doesn't resolve agents.
    assert "agent" not in payload


# --------------------------------------------------------------------------- #
# Empty / whitespace --next-cmd must be rejected for ALL harnesses, including
# codex/opencode (which previously accepted it silently and produced a
# garbled payload with same_context: ""). Review-fix #02 SF2.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Review fix plan #01 — should-fix #17: existing-session prompt for the
# review-task → implement-task common loop. When both `--fix-plan-path`
# and `--pr-number` are provided, the payload carries an
# `existing_session_prompt` field; either flag absent omits the key.
# --------------------------------------------------------------------------- #


def test_existing_session_prompt_emitted_when_fix_plan_and_pr_provided(tmp_path: Path) -> None:
    """Both flags present → payload has ``existing_session_prompt`` with #PR and rel path."""
    result = _run(
        [
            "--next-cmd", "implement-task",
            "--work-dir", "/tmp/wt",
            "--issue", "177",
            "--title", "Test",
            "--harness", "claude-code",
            "--fix-plan-path", "/tmp/wt/docs/stories/QS-177.story_review_fix_#01.md",
            "--pr-number", "179",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "existing_session_prompt" in payload
    prompt = payload["existing_session_prompt"]
    # Path is worktree-relative — no absolute prefix leakage.
    assert "docs/stories/QS-177.story_review_fix_#01.md" in prompt
    assert "/tmp/wt/" not in prompt
    # PR number is surfaced for the user.
    assert "#179" in prompt


def test_existing_session_prompt_null_when_flags_missing(tmp_path: Path) -> None:
    """Neither flag provided → payload carries ``existing_session_prompt: null``.

    Review fix #01 S9: the key is always present so the consuming
    agent prose has a single shape to check (``null`` vs.
    non-empty string) instead of having to disambiguate ``key
    missing`` from ``key present but null``.
    """
    result = _run(
        [
            "--next-cmd", "implement-task",
            "--work-dir", "/tmp/wt",
            "--issue", "177",
            "--title", "Test",
            "--harness", "claude-code",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "existing_session_prompt" in payload
    assert payload["existing_session_prompt"] is None


def test_existing_session_prompt_null_when_only_fix_plan_provided(tmp_path: Path) -> None:
    """``--fix-plan-path`` alone (no PR) → key present with null value (review fix #01 S9)."""
    result = _run(
        [
            "--next-cmd", "implement-task",
            "--work-dir", "/tmp/wt",
            "--issue", "177",
            "--title", "Test",
            "--harness", "claude-code",
            "--fix-plan-path", "/tmp/wt/docs/stories/QS-177.story_review_fix_#01.md",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "existing_session_prompt" in payload
    assert payload["existing_session_prompt"] is None


def test_existing_session_prompt_null_when_only_pr_provided(tmp_path: Path) -> None:
    """``--pr-number`` alone (no fix-plan path) → key present with null value (review fix #01 S9)."""
    result = _run(
        [
            "--next-cmd", "implement-task",
            "--work-dir", "/tmp/wt",
            "--issue", "177",
            "--title", "Test",
            "--harness", "claude-code",
            "--pr-number", "179",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "existing_session_prompt" in payload
    assert payload["existing_session_prompt"] is None


# --------------------------------------------------------------------------- #
# Review fix plan #02 — should-fix #11: --pr-number rejects non-positive values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_pr", ["0", "-1", "-100"])
def test_pr_number_rejects_non_positive(tmp_path: Path, bad_pr: str) -> None:
    """``--pr-number 0`` / negative exits 2 — GitHub PR numbers are always positive."""
    result = _run(
        [
            "--next-cmd", "implement-task",
            "--work-dir", "/tmp/wt",
            "--issue", "177",
            "--title", "Test",
            "--harness", "claude-code",
            "--fix-plan-path", "/tmp/wt/x.md",
            "--pr-number", bad_pr,
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 2, result.stderr


# --------------------------------------------------------------------------- #
# Review fix plan #03 — should-fix #9: --work-dir empty rejected upstream
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_work_dir", ["", "   ", "\t"])
def test_next_step_rejects_empty_work_dir(
    tmp_path: Path, bad_work_dir: str,
) -> None:
    """Empty / whitespace ``--work-dir`` exits 2 via parser.error.

    Without this guard, the opencode launcher builds a
    ``python scripts/qs/spawn_session.py … --directory ''``
    invocation — the user pastes it, runs it, and the failure fires
    far from the original mistake (review fix #03 should-fix #9).
    """
    result = _run(
        [
            "--next-cmd", "create-plan",
            "--work-dir", bad_work_dir,
            "--issue", "42",
            "--title", "T",
            "--harness", "claude-code",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 2, result.stderr


@pytest.mark.parametrize("harness", ["claude-code", "codex", "opencode"])
def test_existing_session_prompt_emitted_for_all_harnesses(
    harness: str, tmp_path: Path,
) -> None:
    """Every launcher emits ``existing_session_prompt`` with the same prompt body."""
    result = _run(
        [
            "--next-cmd", "implement-task",
            "--work-dir", "/tmp/wt",
            "--issue", "177",
            "--title", "Test",
            "--harness", harness,
            "--fix-plan-path", "/tmp/wt/docs/stories/QS-177.story_review_fix_#01.md",
            "--pr-number", "179",
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "existing_session_prompt" in payload
    prompt = payload["existing_session_prompt"]
    assert "docs/stories/QS-177.story_review_fix_#01.md" in prompt
    assert "#179" in prompt


@pytest.mark.parametrize(
    "harness", ["claude-code", "codex", "opencode"],
)
@pytest.mark.parametrize("bad_next_cmd", ["", "   ", "\t"])
def test_empty_or_whitespace_next_cmd_rejected_for_all_harnesses(
    harness: str, bad_next_cmd: str, tmp_path: Path,
) -> None:
    """Empty / whitespace-only --next-cmd → JSON error + exit 1, every harness."""
    result = _run(
        [
            "--next-cmd", bad_next_cmd,
            "--work-dir", "/tmp/work",
            "--issue", "42",
            "--title", "Title",
            "--harness", harness,
        ],
        cwd=str(tmp_path),
    )
    assert result.returncode != 0, (
        f"--harness {harness} accepted empty --next-cmd silently (regression)"
    )
    # JSON error shape is identical to the unknown-phase case so downstream
    # parsing stays simple.
    payload = json.loads(result.stdout)
    assert payload["error"] == "empty next-cmd"
    assert payload["value"] == bad_next_cmd


# ---------------------------------------------------------------------------
# QS-357: the render hook warns and continues (never breaks a handoff)
# ---------------------------------------------------------------------------


def test_render_import_error_warns_and_still_emits_payload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A missing ``jinja2`` (ImportError) → stderr warning + normal payload."""
    import next_step

    monkeypatch.setitem(sys.modules, "render_agents", None)  # import → ImportError
    monkeypatch.setattr(
        sys, "argv",
        [
            "next_step.py", "--next-cmd", "create-plan", "--work-dir", str(tmp_path),
            "--issue", "42", "--title", "T", "--harness", "claude-code",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        next_step.main()
    assert exc.value.code == 0
    cap = capsys.readouterr()
    assert "agent render failed" in cap.err
    assert json.loads(cap.out)["agent"] == "qs-create-plan"


def test_render_render_error_warns_and_still_emits_payload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A ``RenderError`` → stderr warning + normal payload."""
    import next_step
    import render_agents

    def _boom(*_a: object, **_k: object) -> None:
        raise render_agents.RenderError("boom")

    monkeypatch.setattr(render_agents, "render_all", _boom)
    monkeypatch.setattr(
        sys, "argv",
        [
            "next_step.py", "--next-cmd", "create-plan", "--work-dir", str(tmp_path),
            "--issue", "42", "--title", "T", "--harness", "claude-code",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        next_step.main()
    assert exc.value.code == 0
    cap = capsys.readouterr()
    assert "agent render failed" in cap.err
    assert json.loads(cap.out)["agent"] == "qs-create-plan"


def _handoff_argv(work_dir: str) -> list[str]:
    return [
        "next_step.py", "--next-cmd", "create-plan", "--work-dir", work_dir,
        "--issue", "42", "--title", "T", "--harness", "claude-code",
    ]


def test_render_warns_on_unbound_facts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """QS-357 review-fix #01 S1: warn when the handoff render is task-agnostic
    (branch does not resolve to a QS_<N> issue) despite a known --issue."""
    import next_step
    import render_agents

    ctx = {"facts_state": "unbound", "lane_protocol_state": "no_lane"}
    monkeypatch.setattr(render_agents, "build_render_context", lambda *a, **k: ctx)
    monkeypatch.setattr(render_agents, "render_all", lambda *a, **k: [])
    monkeypatch.setattr(sys, "argv", _handoff_argv(str(tmp_path)))

    with pytest.raises(SystemExit) as exc:
        next_step.main()
    assert exc.value.code == 0
    cap = capsys.readouterr()
    assert "task-agnostic" in cap.err
    assert json.loads(cap.out)["agent"] == "qs-create-plan"


def test_render_degradation_warnings_are_independent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """QS-357 review-fix #01 N1: each render-degradation reason surfaces
    independently — an ``elif`` chain would suppress the second."""
    import next_step
    import render_agents

    ctx = {"facts_state": "lookup_failed", "lane_protocol_state": "file_missing"}
    monkeypatch.setattr(render_agents, "build_render_context", lambda *a, **k: ctx)
    monkeypatch.setattr(render_agents, "render_all", lambda *a, **k: [])
    monkeypatch.setattr(sys, "argv", _handoff_argv(str(tmp_path)))

    with pytest.raises(SystemExit) as exc:
        next_step.main()
    assert exc.value.code == 0
    err = capsys.readouterr().err
    assert "lookup_failed task facts" in err
    assert "lane protocol" in err


def test_handoff_survives_load_time_template_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """QS-357 review-fix #02 S3 (real path, folds N8): a load-time template
    error at handoff is funneled through RenderError → stderr warning +
    payload still emitted with exit 0 (handoff-survives-render-failure)."""
    import next_step
    import render_agents

    tdir = tmp_path / "tpl"
    tdir.mkdir()
    (tdir / "_base.md.j2").write_text("[% block body %][% endblock %]\n")
    (tdir / "qs-bad.md.j2").write_text(
        '[% extends "_base.md.j2" %][% block body %][[ 1 + [% endblock %]'
    )
    monkeypatch.setattr(render_agents, "_default_templates_dir", lambda wd: tdir)
    monkeypatch.setattr(sys, "argv", _handoff_argv(str(tmp_path)))

    with pytest.raises(SystemExit) as exc:
        next_step.main()
    assert exc.value.code == 0
    cap = capsys.readouterr()
    assert "agent render failed" in cap.err
    assert json.loads(cap.out)["agent"] == "qs-create-plan"


def test_handoff_survives_non_utf8_template(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """QS-357 review-fix #03 M1 (handoff side): a non-UTF-8 template at
    handoff funnels through RenderError → stderr warning + payload, exit 0."""
    import next_step
    import render_agents

    tdir = tmp_path / "tpl"
    tdir.mkdir()
    (tdir / "_base.md.j2").write_text("[% block body %][% endblock %]\n")
    (tdir / "qs-bad.md.j2").write_bytes(
        b'[% extends "_base.md.j2" %][% block body %]\xff\xfe[% endblock %]'
    )
    monkeypatch.setattr(render_agents, "_default_templates_dir", lambda wd: tdir)
    monkeypatch.setattr(sys, "argv", _handoff_argv(str(tmp_path)))

    with pytest.raises(SystemExit) as exc:
        next_step.main()
    assert exc.value.code == 0
    cap = capsys.readouterr()
    assert "agent render failed" in cap.err
    assert json.loads(cap.out)["agent"] == "qs-create-plan"


# --------------------------------------------------------------------------- #
# QS-358 — the handoff forwards the render context's lane to the launcher
# --------------------------------------------------------------------------- #


def _capturing_launcher(seen: dict):
    class CapturingLauncher:
        @staticmethod
        def build_payload(*_args: object, **kwargs: object) -> dict:
            seen.update(kwargs)
            return {"tool": "fake", "same_context": "x", "new_context": "y"}

    return CapturingLauncher


def test_handoff_passes_render_context_lane(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import next_step
    import render_agents

    ctx = {"facts_state": "bound", "lane_protocol_state": "inlined", "lane": "bug-factory"}
    monkeypatch.setattr(render_agents, "build_render_context", lambda *a, **k: ctx)
    monkeypatch.setattr(render_agents, "render_all", lambda *a, **k: [])
    seen: dict = {}
    monkeypatch.setitem(next_step.LAUNCHERS, "claude-code", _capturing_launcher(seen))
    monkeypatch.setattr(sys, "argv", _handoff_argv(str(tmp_path)))

    with pytest.raises(SystemExit) as exc:
        next_step.main()
    assert exc.value.code == 0
    assert seen["lane"] == "bug-factory"


def test_handoff_passes_no_lane_after_render_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import next_step
    import render_agents

    def _boom(*_a: object, **_k: object) -> None:
        raise render_agents.RenderError("boom")

    monkeypatch.setattr(render_agents, "build_render_context", _boom)
    seen: dict = {}
    monkeypatch.setitem(next_step.LAUNCHERS, "claude-code", _capturing_launcher(seen))
    monkeypatch.setattr(sys, "argv", _handoff_argv(str(tmp_path)))

    with pytest.raises(SystemExit) as exc:
        next_step.main()
    assert exc.value.code == 0
    assert "lane" in seen
    assert seen["lane"] is None
