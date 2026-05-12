# CLAUDE.md

Quiet Solar's development pipeline. Read
[docs/workflow/project-rules.md](docs/workflow/project-rules.md) and
[docs/workflow/project-context.md](docs/workflow/project-context.md)
before doing substantive work.

## Commands

Each phase runs as an interactive `claude --agent qs-<phase>` session
(preferred — fresh terminal) or as a `/<phase>` slash command
(fallback — degraded one-shot UX kept for Claude Desktop). See
[docs/workflow/overview.md](docs/workflow/overview.md) section
"Orchestrators are interactive sessions; sub-agents are parallel
fan-out" for the rationale.

| Phase                | Preferred (interactive)                      | Fallback (degraded)      | Where         |
| -------------------- | -------------------------------------------- | ------------------------ | ------------- |
| Setup task           | `claude --agent qs-setup-task`               | `/setup-task`            | main checkout |
| Create plan          | `claude --agent qs-create-plan`              | `/create-plan`           | worktree      |
| Implement (product)  | `claude --agent qs-implement-task`           | `/implement-task`        | worktree      |
| Implement (dev-env)  | `claude --agent qs-implement-setup-task`     | `/implement-setup-task`  | worktree      |
| Review task          | `claude --agent qs-review-task`              | `/review-task`           | worktree      |
| Finish task          | `claude --agent qs-finish-task`              | `/finish-task`           | worktree      |
| Release              | `claude --agent qs-release`                  | `/release`               | main checkout |

Static agents live in [.claude/agents/](.claude/agents/); slash commands
in [.claude/commands/](.claude/commands/). Agents discover task context
at runtime via `python scripts/qs/context.py` (no per-task rendering).

## Pipeline architecture

- [docs/workflow/overview.md](docs/workflow/overview.md) — the static-agent pipeline
- [docs/workflow/phase-protocols.md](docs/workflow/phase-protocols.md) — each phase's contract
- [docs/workflow/adversarial-review.md](docs/workflow/adversarial-review.md) — the 4-reviewer pattern
- [docs/workflow/harness.md](docs/workflow/harness.md) — multi-harness abstraction (Claude / Cursor / OpenCode / Codex)

## Quality gate

`python scripts/qs/quality_gate.py` — pytest 100% cov + ruff + mypy +
translations. Smart scope detection skips the full suite when only
dev-infrastructure files changed.

## Legacy OpenCode pipeline (unchanged)

The legacy OpenCode pipeline lives at `.opencode/`, `_qsprocess_opencode/`,
and `scripts/qs_opencode/` — kept intact and untouched until the
new static-agent pipeline is proven, then it will be retired.
[AGENTS.md](AGENTS.md) remains the OpenCode entry point.
