# CLAUDE.md

Quiet Solar's development pipeline. Read
[docs/workflow/project-rules.md](docs/workflow/project-rules.md) and
[docs/workflow/project-context.md](docs/workflow/project-context.md)
before doing substantive work.

**Single source of truth — no auto-memory.** Do not use or write Claude
Code auto-memory for this project. `CLAUDE.md` and `docs/` are the only
source of truth; all durable context must live in the repo, not in
per-machine memory stores.

## Commands

Each phase runs as an interactive `claude --agent qs-<phase>` session
(preferred — fresh terminal) or as a `/<phase>` slash command
(fallback — degraded one-shot UX; the GUI can instead run the phase
agent directly, see [docs/workflow/harness.md](docs/workflow/harness.md)).
See
[docs/workflow/overview.md](docs/workflow/overview.md) section
"Orchestrators are interactive sessions; sub-agents are parallel
fan-out" for the rationale.

| Phase                | Preferred (interactive)                      | Fallback (degraded)      | Where         |
| -------------------- | -------------------------------------------- | ------------------------ | ------------- |
| Setup task           | `claude --agent qs-setup-task`               | `/setup-task`            | main checkout |
| Create plan          | `claude --agent qs-create-plan`              | `/create-plan`           | worktree      |
| Diagnose task        | `claude --agent qs-diagnose-task`            | `/diagnose-task`         | worktree      |
| Implement (product)  | `claude --agent qs-implement-task`           | `/implement-task`        | worktree      |
| Implement (dev-env)  | `claude --agent qs-implement-setup-task`     | `/implement-setup-task`  | worktree      |
| Review task          | `claude --agent qs-review-task`              | `/review-task`           | worktree      |
| Verify task          | `claude --agent qs-verify-task`              | `/verify-task`           | worktree      |
| Finish task          | `claude --agent qs-finish-task`              | `/finish-task`           | worktree      |
| Release              | `claude --agent qs-release`                  | `/release`               | main checkout |

The **bug × product** lane diverges (QS-335): `setup → diagnose → fix
(implement) → verify → finish` — `diagnose-task` replaces `create-plan`
and `verify-task` replaces `review-task` for that lane only (see
[docs/workflow/lanes/bug-product.md](docs/workflow/lanes/bug-product.md)).

Agents are rendered per worktree from one Jinja template each under
[scripts/qs/agent_templates/](scripts/qs/agent_templates/) into the
gitignored `.claude/agents/` and `.opencode/agents/` (QS-357 — edit the
template, never a rendered output); slash commands stay tracked in
[.claude/commands/](.claude/commands/). Rendering runs automatically at
worktree birth, at every handoff, and post-merge on `main`; on a fresh
clone run `python scripts/qs/render_agents.py` once. Agents still resolve
volatile task context at runtime via `python scripts/qs/context.py`.
The model and thinking effort per agent come from
[scripts/qs/models.py](scripts/qs/models.py) (QS-358 — four classes, one
exact-model row per harness): the model is rendered into each agent's
frontmatter on both harnesses, the thinking effort into Claude's only;
`opencode.json` follows the `deep` row.

## Pipeline architecture

- [docs/workflow/overview.md](docs/workflow/overview.md) — the static-agent pipeline
- [docs/workflow/phase-protocols.md](docs/workflow/phase-protocols.md) — each phase's contract
- [docs/workflow/adversarial-review.md](docs/workflow/adversarial-review.md) — the 4-reviewer pattern
- [docs/workflow/harness.md](docs/workflow/harness.md) — multi-harness abstraction (Claude / OpenCode / Codex)

## Quality gate

`python scripts/qs/quality_gate.py` — pytest 100% cov + ruff + mypy +
translations. Smart scope detection skips the full suite when only
dev-infrastructure files changed. The mandatory pre-commit form is
`python scripts/qs/quality_gate.py --impacted` (testmon-selected tests
plus changed-line 100% coverage, self-healing); the whole-suite gate
runs authoritatively in CI on every PR (including the translations
value-check since QS-292). For change sets touching
`tests/qs`-pinned non-Python files (agent files, commands, workflow
docs, `.claude/settings.json`),
`python scripts/qs/quality_gate.py --quick tests/qs` is a required
pre-commit supplement — testmon cannot see non-Python files.

## Legacy OpenCode pipeline

See `legacy/` for the retired per-task-rendering OpenCode pipeline.
