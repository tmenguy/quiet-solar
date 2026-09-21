#!/usr/bin/env python3
"""Render the per-worktree harness agent files from Jinja templates (QS-357).

One tracked template per agent lives under
``scripts/qs/agent_templates/`` (``qs-<name>.md.j2``, plus the
leading-underscore partials ``_base.md.j2`` / ``_macros.md.j2``). This
module renders them into the gitignored harness output directories
(``.claude/agents/`` and ``.opencode/agents/``) with the task facts and
the lane protocol inlined into the lane-aware orchestrators.

Public API:

- :func:`build_render_context` — resolve every render variable for a
  worktree (task-bound or task-agnostic); **always** returns every key.
- :func:`render_all` — render every template for both harnesses,
  writing atomically; returns the list of files written.
- :func:`main` — the ``python scripts/qs/render_agents.py`` CLI.

Custom Jinja delimiters (``[[ ]]`` for variables, ``[% %]`` for blocks)
keep the ~360 literal ``{{issue}}``-style placeholders in the agent
prose inert with no ``{% raw %}`` pass — see the story's Delimiters
decision. Comments keep Jinja's default ``{# #}``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import jinja2

import context as context_mod  # type: ignore[import-not-found]

from launchers.phases import PHASE_TO_AGENT  # type: ignore[import-not-found]

import targets  # type: ignore[import-not-found]

from utils import get_issue_from_branch, run_git  # type: ignore[import-not-found]


class RenderError(RuntimeError):
    """Raised at the two API boundaries when rendering cannot proceed.

    :func:`build_render_context` wraps lane-file read failures;
    :func:`render_all` wraps :class:`jinja2.TemplateError`, output
    ``OSError`` and the tracked-target guard. A missing ``jinja2``
    surfaces as ``ImportError`` at import time — which is why the hooks
    import this module function-locally.
    """


# ---------------------------------------------------------------------------
# Registry (story §1)
# ---------------------------------------------------------------------------

# Orchestrators are exactly the agents a phase maps to (single source of
# truth: ``launchers/phases.py`` — which imports nothing from here, so no
# cycle). 9 stems.
ORCHESTRATORS: frozenset[str] = frozenset(PHASE_TO_AGENT.values())

# The six orchestrators whose phase protocol varies by lane; they carry
# the lane file inlined and the ``lane_paragraph`` macro.
LANE_AWARE: frozenset[str] = frozenset(
    {
        "qs-create-plan",
        "qs-diagnose-task",
        "qs-implement-task",
        "qs-implement-setup-task",
        "qs-review-task",
        "qs-verify-task",
    }
)

# The 12 hidden reviewer / helper sub-agents. They get nothing extra —
# no Task facts, no lane protocol, no Reference map — so their isolation
# contracts ("plan text only", "diff only") hold.
SUBAGENTS: frozenset[str] = frozenset(
    {
        "qs-plan-critic",
        "qs-plan-concrete-planner",
        "qs-plan-dev-proxy",
        "qs-plan-scope-guardian",
        "qs-plan-delta-auditor",
        "qs-diag-root-cause-skeptic",
        "qs-diag-fix-minimalist",
        "qs-review-blind-hunter",
        "qs-review-edge-case-hunter",
        "qs-review-acceptance-auditor",
        "qs-review-coderabbit",
        "qs-review-regression-proof",
    }
)

_HARNESSES = ("claude", "opencode")
_HARNESS_DIR = {"claude": ".claude", "opencode": ".opencode"}


# ---------------------------------------------------------------------------
# Root resolution (story §2 "Roots")
# ---------------------------------------------------------------------------


def _default_templates_dir(work_dir: str) -> Path:
    """``work_dir``-first templates dir; fall back to this package's copy."""
    candidate = Path(work_dir) / "scripts" / "qs" / "agent_templates"
    if candidate.is_dir():
        return candidate
    return Path(__file__).parent / "agent_templates"


def _default_lanes_dir(work_dir: str) -> Path:
    """``work_dir``-first lanes dir; fall back to this repo's copy."""
    candidate = Path(work_dir) / "docs" / "workflow" / "lanes"
    if candidate.is_dir():
        return candidate
    return Path(__file__).parents[2] / "docs" / "workflow" / "lanes"


# ---------------------------------------------------------------------------
# Render context (story §2)
# ---------------------------------------------------------------------------


def _current_branch(work_dir: str) -> str | None:
    """The worktree's current branch, or ``None`` on any git failure.

    Uses the renderer's own ``git -C`` call (``check=False``);
    ``utils.get_current_branch()`` takes no ``cwd`` and raises.
    """
    try:
        result = run_git(["-C", work_dir, "branch", "--show-current"], check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _fetch_fields(issue: int) -> tuple[dict, bool]:
    """Return ``(fields, ok)`` — ``ok`` is ``False`` on any lookup failure.

    ``fetch_issue_fields`` already degrades a non-zero exit / unparseable
    JSON to the empty sentinel; a missing ``gh`` binary raises
    ``FileNotFoundError`` (an ``OSError``), which we catch here.
    """
    try:
        fields = context_mod.fetch_issue_fields(issue)
    except OSError:
        return {}, False
    if not fields.get("title") and not fields.get("labels"):
        return fields, False
    return fields, True


def build_render_context(
    work_dir: str | os.PathLike[str],
    *,
    bound: bool | None = None,
    issue: int | None = None,
    title: str | None = None,
    labels: list[str] | None = None,
    fetch: bool = True,
    model: str | Mapping[str, str] = "inherit",
    lanes_dir: str | os.PathLike[str] | None = None,
) -> dict:
    """Resolve every render variable for ``work_dir``.

    Always returns every key (``None`` where unresolved). ``bound=None``
    derives task-boundness from the branch; ``bound=False`` forces the
    task-agnostic render (needed by the fidelity check and by the test
    fixture while on branch ``QS_357``); ``bound=True`` forces the
    task-bound render.

    Raises:
        RenderError: if a resolvable lane file cannot be read.
    """
    work_dir = str(work_dir)
    branch = _current_branch(work_dir)

    if issue is None and bound is not False:
        issue = get_issue_from_branch(branch) if branch else None
    if bound is False:
        issue = None

    bound_task = issue is not None and issue > 0

    facts_state = "unbound"
    if bound_task:
        facts_state = "bound"
        if title is None and labels is None and fetch:
            fields, ok = _fetch_fields(issue)  # type: ignore[arg-type]
            if ok:
                title = fields["title"]
                labels = fields["labels"]
            else:
                facts_state = "lookup_failed"
        if facts_state == "bound" and title is None:
            title = f"Issue #{issue}"

    lane = targets.parse_axes(labels)["lane"] or None if labels is not None else None

    lanes_path = Path(lanes_dir) if lanes_dir is not None else _default_lanes_dir(work_dir)
    lane_protocol: str | None = None
    lane_protocol_state = "no_lane"
    if lane:
        lane_file = lanes_path / f"{lane}.md"
        if lane_file.is_file():
            try:
                lane_protocol = lane_file.read_text(encoding="utf-8")
            except OSError as exc:
                raise RenderError(f"could not read lane file {lane_file}: {exc}") from exc
            lane_protocol_state = "inlined"
        else:
            lane_protocol_state = "file_missing"
            print(
                f"warning: lane file {lane_file} is missing; the lane protocol "
                f"will not be inlined",
                file=sys.stderr,
            )

    return {
        "branch": branch,
        "issue": issue,
        "title": title,
        "labels": labels,
        "lane": lane,
        "worktree": str(Path(work_dir).resolve()),
        "story_file": f"docs/stories/QS-{issue}.story.md" if bound_task else None,
        "lane_protocol": lane_protocol,
        "lane_protocol_state": lane_protocol_state,
        "model": model,
        "facts_state": facts_state,
    }


# ---------------------------------------------------------------------------
# Rendering (story §1, §3)
# ---------------------------------------------------------------------------


def _make_env(templates_dir: Path) -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        autoescape=False,
        variable_start_string="[[",
        variable_end_string="]]",
        block_start_string="[%",
        block_end_string="%]",
    )


def _discover_stems(templates_dir: Path) -> list[str]:
    """Sorted ``qs-*`` stems from ``*.md.j2`` minus leading-underscore files."""
    stems: list[str] = []
    for path in sorted(templates_dir.glob("*.md.j2")):
        if path.name.startswith("_"):
            continue
        stems.append(path.name.removesuffix(".md.j2"))
    return stems


def _resolve_model(model: str | Mapping[str, str], stem: str) -> str:
    if isinstance(model, Mapping):
        return model.get(stem, "inherit")
    return model


def _guard_tracked(out_root: Path) -> None:
    """Refuse to render over agent files still tracked in ``out_root``'s index.

    This makes the commit-1 discipline mechanical (the tracked files must
    not be rendered over) and retires itself after commit 2 untracks the
    outputs. A non-git ``out_root`` has no guard.
    """
    try:
        result = run_git(
            ["-C", str(out_root), "ls-files", "--", ".claude/agents", ".opencode/agents"],
            check=False,
        )
    except OSError:
        return
    if result.returncode != 0:
        return
    if result.stdout.strip():
        raise RenderError(
            "refusing to overwrite tracked agent files; untrack them first "
            "(git rm --cached .claude/agents/*.md .opencode/agents/*.md): "
            + ", ".join(result.stdout.split())
        )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except OSError as exc:
        raise RenderError(f"could not write {path}: {exc}") from exc
    finally:
        if tmp.exists():
            tmp.unlink()


def render_all(
    work_dir: str | os.PathLike[str],
    *,
    context: dict | None = None,
    out_root: str | os.PathLike[str] | None = None,
    templates_dir: str | os.PathLike[str] | None = None,
) -> list[Path]:
    """Render every template for both harnesses; return the files written.

    ``context=None`` means ``build_render_context(work_dir)`` with
    defaults. Writes nothing to stdout.

    Raises:
        RenderError: on any Jinja error, output failure, or when the
            tracked-target guard trips.
    """
    work_dir = str(work_dir)
    if context is None:
        context = build_render_context(work_dir)
    templates_path = (
        Path(templates_dir) if templates_dir is not None else _default_templates_dir(work_dir)
    )
    out_path = Path(out_root) if out_root is not None else Path(work_dir)

    _guard_tracked(out_path)

    env = _make_env(templates_path)
    stems = _discover_stems(templates_path)
    model_spec = context.get("model", "inherit")

    written: list[Path] = []
    for stem in stems:
        template = env.get_template(f"{stem}.md.j2")
        for harness in _HARNESSES:
            render_ctx = {
                **context,
                "harness": harness,
                "stem": stem,
                "template": f"{stem}.md.j2",
                "orchestrator": stem in ORCHESTRATORS,
                "lane_aware": stem in LANE_AWARE,
                "model": _resolve_model(model_spec, stem),
            }
            try:
                text = template.render(**render_ctx)
            except jinja2.TemplateError as exc:
                raise RenderError(f"failed to render {stem} ({harness}): {exc}") from exc
            text = text.rstrip("\n") + "\n"
            out = out_path / _HARNESS_DIR[harness] / "agents" / f"{stem}.md"
            _atomic_write(out, text)
            written.append(out)
    return written


# ---------------------------------------------------------------------------
# CLI (story §3)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="render_agents.py",
        description="Render the per-worktree harness agent files from templates.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Worktree to render into (default: current working directory).",
    )
    args = parser.parse_args(argv)
    work_dir = args.work_dir or os.getcwd()

    try:
        written = render_all(work_dir)
    except RenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
