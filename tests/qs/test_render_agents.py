"""Tests for ``scripts/qs/render_agents.py`` (QS-357).

Two layers:

- **Synthetic templates** in ``tmp_path`` exercise the renderer's
  mechanics in isolation (custom delimiters, macro import, atomic write,
  the tracked-target guard, every context state, the CLI).
- **Real templates** under ``scripts/qs/agent_templates/`` are rendered
  for both harnesses in both modes and pinned against the registry, the
  frontmatter contracts, the inlining partition and the AC texts.

Invariant (story §7): never pass the repo root as ``work_dir`` /
``out_root`` for a render, and never derive the unbound context from the
repo's current branch. The standard recipe below forces ``bound`` and
passes explicit ``templates_dir`` / ``lanes_dir``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "scripts" / "qs" / "agent_templates"
LANES_DIR = REPO_ROOT / "docs" / "workflow" / "lanes"

# ``scripts/qs`` is put on ``sys.path`` at test setup by the autouse
# ``_add_scripts_qs_to_syspath`` fixture in ``tests/qs/conftest.py`` — but
# that fires per-test, after collection. This module imports the renderer
# at module scope, so it inserts the path here too (idempotent with the
# fixture, which sees it already present and leaves it).
sys.path.insert(0, str(REPO_ROOT / "scripts" / "qs"))

import render_agents as r  # type: ignore[import-not-found]  # noqa: E402

from tests.qs.agents.const import REQUIRED_FRONTMATTER_KEYS  # noqa: E402

MODEL_COMMENT = "# model: github-copilot/claude-sonnet-4.5  # uncomment to override project default"


# ---------------------------------------------------------------------------
# Synthetic-template helpers
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _synthetic_templates(tmp_path: Path) -> Path:
    """A minimal template dir: a base, a macro, and one child template."""
    tdir = tmp_path / "templates"
    tdir.mkdir()
    _write(
        tdir / "_base.md.j2",
        "HEADER harness=[[ harness ]] stem=[[ stem ]] model=[[ model ]]\n"
        "orch=[[ orchestrator ]] lane_aware=[[ lane_aware ]]\n"
        "[% block body %][% endblock %]\n",
    )
    _write(
        tdir / "_macros.md.j2",
        '[%- macro greet(who) -%]hello [[ who ]] on [[ harness ]][%- endmacro -%]\n',
    )
    _write(
        tdir / "qs-synthetic.md.j2",
        '[% extends "_base.md.j2" %]\n'
        '[% import "_macros.md.j2" as m with context %]\n'
        "[% block body %]literal {{issue}} and {% raw %} stay inert\n"
        "[[ m.greet(\"world\") ]][% endblock %]\n",
    )
    return tdir


# ---------------------------------------------------------------------------
# build_render_context — states and branch cases
# ---------------------------------------------------------------------------


def test_context_bound_from_kwargs(tmp_path: Path) -> None:
    ctx = r.build_render_context(
        tmp_path,
        bound=True,
        issue=42,
        title="A title",
        labels=["kind:feature", "target:factory", "scale:task"],
        fetch=False,
        lanes_dir=LANES_DIR,
    )
    assert ctx["facts_state"] == "bound"
    assert ctx["issue"] == 42
    assert ctx["title"] == "A title"
    assert ctx["lane"] == "feature-factory"
    assert ctx["lane_protocol_state"] == "inlined"
    assert ctx["lane_protocol"] is not None
    assert ctx["story_file"] == "docs/stories/QS-42.story.md"
    # every key present
    for key in (
        "branch", "issue", "title", "labels", "lane", "worktree",
        "story_file", "lane_protocol", "lane_protocol_state", "model",
        "facts_state",
    ):
        assert key in ctx


def test_context_bound_false_forces_unbound(tmp_path: Path, monkeypatch) -> None:
    # Even if the branch parses as QS_99, bound=False forces unbound.
    monkeypatch.setattr(r, "_current_branch", lambda wd: "QS_99")
    ctx = r.build_render_context(tmp_path, bound=False, fetch=False, lanes_dir=LANES_DIR)
    assert ctx["facts_state"] == "unbound"
    assert ctx["issue"] is None
    assert ctx["story_file"] is None
    assert ctx["lane"] is None
    assert ctx["lane_protocol_state"] == "no_lane"


def test_context_branch_qs_number(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(r, "_current_branch", lambda wd: "QS_42")
    monkeypatch.setattr(
        r.context_mod, "fetch_issue_fields",
        lambda issue: {"title": "T", "labels": ["kind:bug", "target:product", "scale:task"], "body": ""},
    )
    ctx = r.build_render_context(tmp_path, lanes_dir=LANES_DIR)
    assert ctx["issue"] == 42
    assert ctx["facts_state"] == "bound"
    assert ctx["lane"] == "bug-product"


def test_context_branch_main_is_unbound(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(r, "_current_branch", lambda wd: "main")
    ctx = r.build_render_context(tmp_path, fetch=False, lanes_dir=LANES_DIR)
    assert ctx["issue"] is None
    assert ctx["facts_state"] == "unbound"


def test_context_lookup_failed_nonzero(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(r, "_current_branch", lambda wd: "QS_7")
    monkeypatch.setattr(
        r.context_mod, "fetch_issue_fields",
        lambda issue: {"title": "", "labels": [], "body": ""},
    )
    ctx = r.build_render_context(tmp_path, lanes_dir=LANES_DIR)
    assert ctx["facts_state"] == "lookup_failed"
    assert ctx["issue"] == 7


def test_context_lookup_failed_filenotfound(tmp_path: Path, monkeypatch) -> None:
    def _boom(issue):
        raise FileNotFoundError("gh missing")

    monkeypatch.setattr(r, "_current_branch", lambda wd: "QS_7")
    monkeypatch.setattr(r.context_mod, "fetch_issue_fields", _boom)
    ctx = r.build_render_context(tmp_path, lanes_dir=LANES_DIR)
    assert ctx["facts_state"] == "lookup_failed"


def test_context_title_none_bound_defaults(tmp_path: Path) -> None:
    ctx = r.build_render_context(
        tmp_path, bound=True, issue=42, labels=[], fetch=False, lanes_dir=LANES_DIR,
    )
    # labels=[] (not None) → lane None; title None → "Issue #42"
    assert ctx["title"] == "Issue #42"
    assert ctx["lane"] is None
    assert ctx["lane_protocol_state"] == "no_lane"


def test_context_labels_none_lane_none(tmp_path: Path) -> None:
    ctx = r.build_render_context(
        tmp_path, bound=True, issue=42, title="T", labels=None, fetch=False, lanes_dir=LANES_DIR,
    )
    assert ctx["lane"] is None


def test_context_file_missing(tmp_path: Path) -> None:
    empty_lanes = tmp_path / "emptylanes"
    empty_lanes.mkdir()
    ctx = r.build_render_context(
        tmp_path,
        bound=True,
        issue=42,
        title="T",
        labels=["kind:feature", "target:factory", "scale:task"],
        fetch=False,
        lanes_dir=empty_lanes,
    )
    assert ctx["lane_protocol_state"] == "file_missing"
    assert ctx["lane_protocol"] is None


def test_context_lane_read_oserror_raises(tmp_path: Path, monkeypatch) -> None:
    bad_lanes = tmp_path / "badlanes"
    bad_lanes.mkdir()
    (bad_lanes / "feature-factory.md").write_text("x")

    def _boom(self, *a, **k):
        raise OSError("nope")

    monkeypatch.setattr(Path, "read_text", _boom)
    with pytest.raises(r.RenderError):
        r.build_render_context(
            tmp_path,
            bound=True,
            issue=42,
            title="T",
            labels=["kind:feature", "target:factory", "scale:task"],
            fetch=False,
            lanes_dir=bad_lanes,
        )


def test_context_default_lanes_dir(tmp_path: Path) -> None:
    # No lanes_dir kwarg → falls back to the repo copy (work_dir has none).
    ctx = r.build_render_context(
        tmp_path,
        bound=True,
        issue=42,
        title="T",
        labels=["kind:feature", "target:factory", "scale:task"],
        fetch=False,
    )
    assert ctx["lane_protocol_state"] == "inlined"


# ---------------------------------------------------------------------------
# _current_branch / _fetch_fields
# ---------------------------------------------------------------------------


def test_current_branch_oserror(monkeypatch) -> None:
    def _boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(r, "run_git", _boom)
    assert r._current_branch("/x") is None


def test_current_branch_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(
        r, "run_git",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "err"),
    )
    assert r._current_branch("/x") is None


def test_current_branch_ok(monkeypatch) -> None:
    monkeypatch.setattr(
        r, "run_git",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, "QS_5\n", ""),
    )
    assert r._current_branch("/x") == "QS_5"


def test_current_branch_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        r, "run_git",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, "\n", ""),
    )
    assert r._current_branch("/x") is None


# ---------------------------------------------------------------------------
# render_all mechanics over synthetic templates
# ---------------------------------------------------------------------------


def _synthetic_context(tmp_path: Path, **over) -> dict:
    base = {
        "branch": None, "issue": None, "title": None, "labels": None,
        "lane": None, "worktree": str(tmp_path), "story_file": None,
        "lane_protocol": None, "lane_protocol_state": "no_lane",
        "model": "inherit", "facts_state": "unbound",
    }
    base.update(over)
    return base


def test_render_custom_delimiters_and_macro(tmp_path: Path) -> None:
    tdir = _synthetic_templates(tmp_path)
    out = tmp_path / "out"
    written = r.render_all(
        tmp_path, context=_synthetic_context(tmp_path),
        out_root=out, templates_dir=tdir,
    )
    assert len(written) == 2
    claude = (out / ".claude" / "agents" / "qs-synthetic.md").read_text()
    assert "literal {{issue}} and {% raw %} stay inert" in claude
    assert "hello world on claude" in claude
    opencode = (out / ".opencode" / "agents" / "qs-synthetic.md").read_text()
    assert "hello world on opencode" in opencode


def test_render_strict_undefined(tmp_path: Path) -> None:
    tdir = tmp_path / "t"
    tdir.mkdir()
    _write(tdir / "_base.md.j2", "[% block body %][% endblock %]\n")
    _write(tdir / "qs-x.md.j2", '[% extends "_base.md.j2" %][% block body %][[ nope ]][% endblock %]')
    with pytest.raises(r.RenderError):
        r.render_all(tmp_path, context=_synthetic_context(tmp_path), out_root=tmp_path / "o", templates_dir=tdir)


def test_render_deterministic(tmp_path: Path) -> None:
    tdir = _synthetic_templates(tmp_path)
    ctx = _synthetic_context(tmp_path)
    a = tmp_path / "a"
    b = tmp_path / "b"
    r.render_all(tmp_path, context=ctx, out_root=a, templates_dir=tdir)
    r.render_all(tmp_path, context=ctx, out_root=b, templates_dir=tdir)
    for name in (".claude", ".opencode"):
        assert (a / name / "agents" / "qs-synthetic.md").read_bytes() == (
            b / name / "agents" / "qs-synthetic.md"
        ).read_bytes()


def test_render_tail_normalised(tmp_path: Path) -> None:
    tdir = tmp_path / "t"
    tdir.mkdir()
    _write(tdir / "_base.md.j2", "[% block body %][% endblock %]")
    _write(tdir / "qs-x.md.j2", '[% extends "_base.md.j2" %][% block body %]body\n\n\n[% endblock %]')
    out = tmp_path / "o"
    r.render_all(tmp_path, context=_synthetic_context(tmp_path), out_root=out, templates_dir=tdir)
    text = (out / ".claude" / "agents" / "qs-x.md").read_text()
    assert text.endswith("body\n")
    assert not text.endswith("\n\n")


def test_render_writes_nothing_to_stdout(tmp_path: Path, capsys) -> None:
    tdir = _synthetic_templates(tmp_path)
    r.render_all(tmp_path, context=_synthetic_context(tmp_path), out_root=tmp_path / "o", templates_dir=tdir)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_render_underscore_files_skipped(tmp_path: Path) -> None:
    tdir = _synthetic_templates(tmp_path)
    written = r.render_all(tmp_path, context=_synthetic_context(tmp_path), out_root=tmp_path / "o", templates_dir=tdir)
    stems = {p.name for p in written}
    assert stems == {"qs-synthetic.md"}


def test_render_atomic_write_failure_leaves_no_temp(tmp_path: Path, monkeypatch) -> None:
    tdir = _synthetic_templates(tmp_path)
    out = tmp_path / "o"

    def _boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(r.os, "replace", _boom)
    with pytest.raises(r.RenderError):
        r.render_all(tmp_path, context=_synthetic_context(tmp_path), out_root=out, templates_dir=tdir)
    # no leftover temp siblings
    agents_dir = out / ".claude" / "agents"
    leftovers = list(agents_dir.glob("*.tmp")) if agents_dir.exists() else []
    assert leftovers == []


def test_render_context_none_defaults(tmp_path: Path, monkeypatch) -> None:
    tdir = _synthetic_templates(tmp_path)
    monkeypatch.setattr(r, "_current_branch", lambda wd: None)
    written = r.render_all(tmp_path, out_root=tmp_path / "o", templates_dir=tdir)
    assert len(written) == 2


def test_render_model_scalar_and_mapping(tmp_path: Path) -> None:
    tdir = _synthetic_templates(tmp_path)
    out = tmp_path / "o"
    r.render_all(
        tmp_path,
        context=_synthetic_context(tmp_path, model={"qs-synthetic": "github-copilot/x"}),
        out_root=out, templates_dir=tdir,
    )
    assert "model=github-copilot/x" in (out / ".claude" / "agents" / "qs-synthetic.md").read_text()
    out2 = tmp_path / "o2"
    r.render_all(
        tmp_path, context=_synthetic_context(tmp_path, model="inherit"),
        out_root=out2, templates_dir=tdir,
    )
    assert "model=inherit" in (out2 / ".claude" / "agents" / "qs-synthetic.md").read_text()


def test_resolve_model_mapping_default() -> None:
    assert r._resolve_model({"other": "x"}, "qs-synthetic") == "inherit"
    assert r._resolve_model("scalar", "any") == "scalar"


def test_default_dirs_work_dir_first(tmp_path: Path) -> None:
    (tmp_path / "scripts" / "qs" / "agent_templates").mkdir(parents=True)
    (tmp_path / "docs" / "workflow" / "lanes").mkdir(parents=True)
    assert r._default_templates_dir(tmp_path) == tmp_path / "scripts" / "qs" / "agent_templates"
    assert r._default_lanes_dir(tmp_path) == tmp_path / "docs" / "workflow" / "lanes"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert r._default_templates_dir(empty) == Path(r.__file__).parent / "agent_templates"
    assert r._default_lanes_dir(empty).name == "lanes"


# ---------------------------------------------------------------------------
# tracked-target guard
# ---------------------------------------------------------------------------


def _git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)


def test_guard_refuses_tracked(tmp_path: Path) -> None:
    tdir = _synthetic_templates(tmp_path)
    _git_repo(tmp_path)
    agent = tmp_path / ".claude" / "agents" / "qs-synthetic.md"
    agent.parent.mkdir(parents=True)
    agent.write_text("tracked")
    subprocess.run(["git", "add", ".claude/agents/qs-synthetic.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "x"], cwd=tmp_path, check=True)
    with pytest.raises(r.RenderError, match="tracked"):
        r.render_all(tmp_path, context=_synthetic_context(tmp_path), out_root=tmp_path, templates_dir=tdir)


def test_guard_allows_untracked(tmp_path: Path) -> None:
    tdir = _synthetic_templates(tmp_path)
    _git_repo(tmp_path)
    written = r.render_all(tmp_path, context=_synthetic_context(tmp_path), out_root=tmp_path, templates_dir=tdir)
    assert len(written) == 2


def test_guard_oserror_skips(tmp_path: Path, monkeypatch) -> None:
    def _boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(r, "run_git", _boom)
    r._guard_tracked(tmp_path)  # no raise


def test_guard_nonzero_skips(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        r, "run_git",
        lambda *a, **k: subprocess.CompletedProcess(a, 128, "", "not a repo"),
    )
    r._guard_tracked(tmp_path)  # no raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_in_process(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(r, "_current_branch", lambda wd: None)
    rc = r.main(["--work-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert str(tmp_path / ".claude" / "agents") in out
    assert len(list((tmp_path / ".claude" / "agents").glob("*.md"))) == 21


def test_main_render_error(tmp_path: Path, capsys, monkeypatch) -> None:
    def _boom(*a, **k):
        raise r.RenderError("boom")

    monkeypatch.setattr(r, "render_all", _boom)
    rc = r.main(["--work-dir", str(tmp_path)])
    assert rc == 1
    assert "boom" in capsys.readouterr().err


def test_main_subprocess(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/qs/render_agents.py", "--work-dir", str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(list((tmp_path / ".opencode" / "agents").glob("*.md"))) == 21


# ---------------------------------------------------------------------------
# Real templates — registry
# ---------------------------------------------------------------------------


def test_registry_invariants() -> None:
    assert r.LANE_AWARE <= r.ORCHESTRATORS
    assert r.ORCHESTRATORS & r.SUBAGENTS == set()
    union = r.ORCHESTRATORS | r.SUBAGENTS
    assert len(union) == 21
    glob = {p.name.removesuffix(".md.j2") for p in TEMPLATES_DIR.glob("*.md.j2") if not p.name.startswith("_")}
    assert glob == union
    assert len(r.ORCHESTRATORS) == 9
    assert len(r.SUBAGENTS) == 12


# ---------------------------------------------------------------------------
# Real templates — render fixtures
# ---------------------------------------------------------------------------


def _render_real(tmp_path: Path, *, bound: bool) -> Path:
    out = tmp_path / ("bound" if bound else "unbound")
    if bound:
        ctx = r.build_render_context(
            tmp_path, bound=True, issue=42, title="A title",
            labels=["kind:feature", "target:factory", "scale:task"],
            fetch=False, lanes_dir=LANES_DIR,
        )
    else:
        ctx = r.build_render_context(tmp_path, bound=False, fetch=False, lanes_dir=LANES_DIR)
    r.render_all(tmp_path, context=ctx, out_root=out, templates_dir=TEMPLATES_DIR)
    return out


def test_all_templates_render_both_modes(tmp_path: Path) -> None:
    for bound in (False, True):
        out = _render_real(tmp_path, bound=bound)
        for hdir in (".claude", ".opencode"):
            files = list((out / hdir / "agents").glob("*.md"))
            assert len(files) == 21
            for f in files:
                text = f.read_text()
                assert "[[" not in text
                assert "[%" not in text


def test_claude_frontmatter_contract(tmp_path: Path) -> None:
    out = _render_real(tmp_path, bound=False)
    for f in (out / ".claude" / "agents").glob("*.md"):
        stem = f.name.removesuffix(".md")
        fm = _frontmatter(f.read_text())
        data = yaml.safe_load(fm)
        assert data["name"] == stem
        assert data["description"]
        assert data["tools"]
        assert data["model"] == "inherit"


def test_opencode_frontmatter_contract(tmp_path: Path) -> None:
    out = _render_real(tmp_path, bound=False)
    for f in (out / ".opencode" / "agents").glob("*.md"):
        text = f.read_text()
        fm = _frontmatter(text)
        data = yaml.safe_load(fm)
        for key in REQUIRED_FRONTMATTER_KEYS:
            assert key in data
        assert "name" not in data
        assert MODEL_COMMENT in text


def _frontmatter(text: str) -> str:
    lines = text.split("\n")
    assert lines[0] == "---"
    close = lines.index("---", 1)
    return "\n".join(lines[1:close])


def test_lane_aware_inlining_bound(tmp_path: Path) -> None:
    out = _render_real(tmp_path, bound=True)
    lane_text = (LANES_DIR / "feature-factory.md").read_text()
    distinctive = lane_text.strip().split("\n")[0]  # first line of the lane file
    for stem in r.LANE_AWARE:
        for hdir in (".claude", ".opencode"):
            text = (out / hdir / "agents" / f"{stem}.md").read_text()
            assert "## Task facts (rendered)" in text
            assert "## Lane protocol (rendered from docs/workflow/lanes/feature-factory.md)" in text
            assert distinctive in text
            assert "This task's lane is `feature-factory`" in text
            assert "## Reference map" in text


def test_lane_aware_fallback_unbound(tmp_path: Path) -> None:
    out = _render_real(tmp_path, bound=False)
    readonly = {"qs-create-plan", "qs-diagnose-task", "qs-review-task", "qs-verify-task"}
    for stem in r.LANE_AWARE:
        text = (out / ".claude" / "agents" / f"{stem}.md").read_text()
        assert "**Lane (QS-332).** Also capture `lane`" in text
        assert "## Lane protocol" not in text
        if stem in readonly:
            assert "This phase is read-only" in text
        else:
            assert "In this implement" in text


def test_other_orchestrators_have_facts_no_lane(tmp_path: Path) -> None:
    others = r.ORCHESTRATORS - r.LANE_AWARE
    out = _render_real(tmp_path, bound=True)
    for stem in others:
        text = (out / ".claude" / "agents" / f"{stem}.md").read_text()
        assert "## Task facts (rendered)" in text
        assert "- Lane: feature-factory" in text
        assert "## Lane protocol" not in text
        assert "## Reference map" in text


def test_lane_none_suffix_only_for_lane_aware(tmp_path: Path) -> None:
    # Bound task whose labels do not resolve a lane → plain "none" for the
    # three non-lane-aware orchestrators, suffixed "none (…)" for the six.
    ctx = r.build_render_context(
        tmp_path, bound=True, issue=42, title="T", labels=[], fetch=False, lanes_dir=LANES_DIR,
    )
    out = tmp_path / "o"
    r.render_all(tmp_path, context=ctx, out_root=out, templates_dir=TEMPLATES_DIR)
    plain = (out / ".claude" / "agents" / "qs-setup-task.md").read_text()
    assert "- Lane: none · Story" in plain
    suffixed = (out / ".claude" / "agents" / "qs-create-plan.md").read_text()
    assert "- Lane: none (labels incomplete — see the Lane paragraph below) · Story" in suffixed


def test_subagents_get_nothing_extra(tmp_path: Path) -> None:
    out = _render_real(tmp_path, bound=True)
    for stem in r.SUBAGENTS:
        for hdir in (".claude", ".opencode"):
            text = (out / hdir / "agents" / f"{stem}.md").read_text()
            assert "## Task facts (rendered)" not in text
            assert "## Lane protocol" not in text
            assert "## Reference map" not in text


def test_pinned_task_facts_texts(tmp_path: Path) -> None:
    unbound = _render_real(tmp_path, bound=False)
    text = (unbound / ".claude" / "agents" / "qs-setup-task.md").read_text()
    assert (
        "## Task facts (rendered)\n"
        "No task bound (branch `unknown` is not `QS_<N>`). "
        "Run `python scripts/qs/context.py` from a task worktree."
    ) in text

    bound = _render_real(tmp_path, bound=True)
    btext = (bound / ".claude" / "agents" / "qs-setup-task.md").read_text()
    assert "## Task facts (rendered)\n- Issue: #42 — A title" in btext
    assert "Volatile facts (PR number, latest review fix, story existence) are not rendered" in btext


def test_lookup_failed_task_facts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(r, "_current_branch", lambda wd: "QS_42")
    monkeypatch.setattr(
        r.context_mod, "fetch_issue_fields",
        lambda issue: {"title": "", "labels": [], "body": ""},
    )
    ctx = r.build_render_context(tmp_path, lanes_dir=LANES_DIR)
    out = tmp_path / "o"
    r.render_all(tmp_path, context=ctx, out_root=out, templates_dir=TEMPLATES_DIR)
    text = (out / ".claude" / "agents" / "qs-setup-task.md").read_text()
    assert "Task QS_42 is bound but the issue lookup failed at render time" in text


def test_reference_map_paths_exist(tmp_path: Path) -> None:
    out = _render_real(tmp_path, bound=False)
    text = (out / ".claude" / "agents" / "qs-setup-task.md").read_text()
    section = text.split("## Reference map", 1)[1]
    expected = [
        "docs/workflow/project-rules.md",
        "docs/workflow/project-context.md",
        "docs/workflow/adversarial-review.md",
        "docs/workflow/harness.md",
        "docs/workflow/phase-protocols.md",
        "docs/workflow/lanes/",
        "docs/agents/index.md",
        "docs/stories/",
    ]
    for path in expected:
        assert f"`{path}`" in section, path
        assert (REPO_ROOT / path.rstrip("/")).exists(), path


def test_ac6_no_rules_or_context_sentences(tmp_path: Path) -> None:
    rules_sentence = "All workflow rules, phase protocols, and code-style rules live under"
    context_sentence = "Bridges HA state/entities with domain logic."
    # sanity: the pinned sentences really are distinctive lines of the docs
    assert rules_sentence in (REPO_ROOT / "docs/workflow/project-rules.md").read_text()
    assert context_sentence in (REPO_ROOT / "docs/workflow/project-context.md").read_text()
    for bound in (False, True):
        out = _render_real(tmp_path, bound=bound)
        for hdir in (".claude", ".opencode"):
            for f in (out / hdir / "agents").glob("*.md"):
                text = f.read_text()
                assert rules_sentence not in text
                assert context_sentence not in text


def test_ac9_finish_task_render_line(tmp_path: Path) -> None:
    out = _render_real(tmp_path, bound=False)
    for hdir in (".claude", ".opencode"):
        text = (out / hdir / "agents" / "qs-finish-task.md").read_text()
        assert 'render_agents.py --work-dir "$MAIN_DIR"' in text
