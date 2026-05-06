#!/usr/bin/env python3
"""Render a per-task OpenCode agent file from a template.

Reads a template from ``_qsprocess_opencode/agent_templates/<phase>.md.tmpl``,
substitutes ``{{VAR}}`` placeholders from CLI arguments, and writes the result
to ``<work_dir>/.opencode/agents/qs-<phase>-QS-<issue>.md``.

Templates use a deliberately simple ``{{VAR}}`` syntax (no Jinja dependency).
Unknown variables raise; undefined placeholders left in the output also raise
so every template variable must be accounted for.

Usage::

    python scripts/qs_opencode/render_agent.py \\
        --phase create-plan \\
        --work-dir /path/to/worktree \\
        --issue 42 \\
        --title "Story 3.2: add foo bar" \\
        --story-file _qsprocess_opencode/stories/QS-42.story.md \\
        [--pr 5] \\
        [--extra KEY=VALUE ...]

Prints a JSON payload describing the rendered agent path and its name so the
calling subagent can hand that information to the next phase.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from utils import output_json  # type: ignore[import-not-found]

# Map phase → template filename. Every phase that can be rendered must appear
# here. Reviewer sub-roles are rendered individually so implement-task can
# produce all five review-related agents in one pass.
PHASE_TEMPLATES: dict[str, str] = {
    "create-plan": "qs-create-plan.md.tmpl",
    "plan-critic": "qs-plan-critic.md.tmpl",
    "plan-concrete-planner": "qs-plan-concrete-planner.md.tmpl",
    "plan-dev-proxy": "qs-plan-dev-proxy.md.tmpl",
    "plan-scope-guardian": "qs-plan-scope-guardian.md.tmpl",
    "implement-task": "qs-implement-task.md.tmpl",
    "implement-setup-task": "qs-implement-setup-task.md.tmpl",
    "review-task": "qs-review-task.md.tmpl",
    "review-blind-hunter": "qs-review-blind-hunter.md.tmpl",
    "review-edge-case-hunter": "qs-review-edge-case-hunter.md.tmpl",
    "review-acceptance-auditor": "qs-review-acceptance-auditor.md.tmpl",
    "review-coderabbit": "qs-review-coderabbit.md.tmpl",
    "finish-task": "qs-finish-task.md.tmpl",
    "release": "qs-release.md.tmpl",
}

PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Z_][A-Z0-9_]*)\s*\}\}")


def _repo_root() -> Path:
    """Return the repository root containing ``_qsprocess_opencode/``.

    Walk up from this file's location. We must work both when invoked from
    the main checkout and from a worktree (both share the same templates via
    the main checkout's file tree — worktrees have full working copies).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "_qsprocess_opencode" / "agent_templates").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root (no _qsprocess_opencode/agent_templates/)")


def _load_template(phase: str) -> tuple[Path, str]:
    if phase not in PHASE_TEMPLATES:
        raise SystemExit(f"Unknown phase {phase!r}. Known: {sorted(PHASE_TEMPLATES)}")
    tmpl_path = _repo_root() / "_qsprocess_opencode" / "agent_templates" / PHASE_TEMPLATES[phase]
    if not tmpl_path.is_file():
        raise SystemExit(f"Template not found: {tmpl_path}")
    return tmpl_path, tmpl_path.read_text(encoding="utf-8")


def _render(template: str, context: dict[str, str]) -> str:
    """Substitute ``{{VAR}}`` placeholders. Missing keys raise."""
    missing: set[str] = set()

    def sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            missing.add(key)
            return match.group(0)
        return context[key]

    rendered = PLACEHOLDER_RE.sub(sub, template)
    if missing:
        raise SystemExit(
            f"Template references undefined variables: {sorted(missing)}. "
            f"Provide them via CLI flags or --extra KEY=VALUE."
        )
    # Second pass: ensure no stray {{...}} survived (defensive).
    leftover = PLACEHOLDER_RE.findall(rendered)
    if leftover:
        raise SystemExit(f"Unresolved placeholders after render: {sorted(set(leftover))}")
    return rendered


def _parse_extras(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--extra expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key or not key.isidentifier() or not key.isupper():
            raise SystemExit(f"--extra key must be UPPER_SNAKE_CASE identifier; got {key!r}")
        out[key] = value
    return out


def _output_path(work_dir: Path, phase: str, issue: int) -> Path:
    agent_name = f"qs-{phase}-QS-{issue}.md"
    return work_dir / ".opencode" / "agents" / agent_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a per-task OpenCode agent file from a template",
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=sorted(PHASE_TEMPLATES.keys()),
        help="Phase whose agent template to render",
    )
    parser.add_argument("--work-dir", required=True, help="Worktree where .opencode/agents/ lives")
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--branch", default=None, help="Feature branch name; defaults to QS_<issue>")
    parser.add_argument("--story-file", default="", help="Path to the story artifact (required for most phases)")
    parser.add_argument("--pr", type=int, default=None)
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra KEY=VALUE placeholders for the template (repeatable)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the agent file if it already exists",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    if not work_dir.is_dir():
        raise SystemExit(f"--work-dir does not exist: {work_dir}")

    branch = args.branch or f"QS_{args.issue}"
    context: dict[str, str] = {
        "PHASE": args.phase,
        "ISSUE": str(args.issue),
        "ISSUE_NUMBER": str(args.issue),
        "BRANCH": branch,
        "TITLE": args.title,
        "WORK_DIR": str(work_dir),
        "STORY_FILE": args.story_file or "",
        "PR_NUMBER": "" if args.pr is None else str(args.pr),
        "AGENT_NAME": f"qs-{args.phase}-QS-{args.issue}",
    }
    context.update(_parse_extras(args.extra))

    tmpl_path, template = _load_template(args.phase)
    rendered = _render(template, context)

    out_path = _output_path(work_dir, args.phase, args.issue)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Agent file already exists: {out_path}. Pass --overwrite to replace.")
    out_path.write_text(rendered, encoding="utf-8")

    output_json(
        {
            "phase": args.phase,
            "agent_name": context["AGENT_NAME"],
            "agent_file": str(out_path),
            "template": str(tmpl_path),
            "issue": args.issue,
            "work_dir": str(work_dir),
            "summary": (f"Rendered {context['AGENT_NAME']} → {out_path.relative_to(work_dir)}"),
        }
    )


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"render_agent.py: {exc}\n")
        raise SystemExit(2) from exc
