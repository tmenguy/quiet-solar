# Quiet Solar — OpenCode Rules

This file is the OpenCode entry point. It is self-contained — OpenCode
does NOT read `CLAUDE.md`. Project rules and code-style rules are
canonical under `docs/workflow/` (shared with the new static-agent
pipeline); OpenCode-specific machinery lives under `_qsprocess_opencode/`,
`scripts/qs_opencode/`, and `.opencode/`.

## Project overview

Quiet Solar is a Home Assistant custom component that optimizes solar energy
self-consumption through a constraint-based solver.

## Required reading (load with the Read tool on startup)

OpenCode does not auto-parse `@file` references in `AGENTS.md`, so explicitly
load these before doing any substantive work:

- `docs/workflow/project-rules.md` — project rules (commands, architecture constraints, workflow routing)
- `docs/workflow/project-context.md` — 42-rule code style set (naming, async, logging, testing patterns)
- `_qsprocess_opencode/README.md` — this workflow's directory layout and conventions

## Commands

Activate with `source venv/bin/activate` for all Python commands.

```bash
# Run all quality gates (pytest 100% coverage + ruff + mypy + translations)
python scripts/qs/quality_gate.py

# Auto-fix formatting and lint
python scripts/qs/quality_gate.py --fix

# Run tests only
source venv/bin/activate && pytest tests/ -v
```

## Architecture constraints

- **Two-layer boundary**: `home_model/` NEVER imports `homeassistant.*`. `ha_model/` bridges both.
- **Solver step size**: `SOLVER_STEP_S = 900` in `const.py` — don't touch.
- **All config keys in `const.py`** — never hardcode strings.
- **Async rules**: no blocking calls in async code, use `hass.async_add_executor_job()`.
- **Logging**: lazy `%s`, no f-strings in log calls, no periods at end.
- **Translations**: NEVER edit `translations/en.json` — edit `strings.json`, run `bash scripts/generate-translations.sh`.

## Workflow — two static agents, eight rendered per task

The OpenCode pipeline has **two static agents and two slash commands**:

| Command        | Subagent         | Phase                                                 |
| -------------- | ---------------- | ----------------------------------------------------- |
| `/setup-task`  | `qs-setup-task`  | Create issue + branch + worktree; render first per-task agent; print launcher |
| `/release`     | `qs-release`     | Tag + bump version + GitHub Release (independent of any task) |

Every downstream phase is a **per-task agent** rendered on demand from a
template into the new worktree's `.opencode/agents/` folder, named
`qs-<phase>-QS-<N>.md`:

- `qs-create-plan-QS-<N>` — write story artifact, commit, push
- `qs-implement-task-QS-<N>` — TDD implementation + quality gate + PR
- `qs-review-task-QS-<N>` — orchestrate parallel role-based review + triage
- `qs-review-blind-hunter-QS-<N>` (hidden sub-role)
- `qs-review-edge-case-hunter-QS-<N>` (hidden sub-role)
- `qs-review-acceptance-auditor-QS-<N>` (hidden sub-role)
- `qs-review-coderabbit-QS-<N>` (hidden sub-role)
- `qs-finish-task-QS-<N>` — final gate + merge PR + worktree cleanup

Each rendered agent has the issue-specific context (issue number, title,
branch, worktree, story file) baked into its system prompt and a narrow
tool/permission allowlist tuned to that phase.

## Handoff model

- **Phase 1 → 2**: new OpenCode session on the new worktree — explicit
  launcher printed by `qs-setup-task` (not a Task spawn). Before emitting
  the launcher, `qs-setup-task` renders `qs-create-plan-QS-<N>.md` into
  the worktree.
- **Phases 2 → 3 → 4 → 5**: each phase ends by spawning a **new
  interactive session** via the OpenCode HTTP API (`spawn_session.py`).
  The new session appears in the OpenCode sidebar and is fully
  interactive. Each phase ends by calling `scripts/qs_opencode/next_step.py`
  which emits a handoff JSON payload. The payload contains:
  - `render_commands` — one or more `render_agent.py` invocations the
    finishing agent must execute to materialize the next agent file(s).
  - `spawn_session_command` — the `spawn_session.py` invocation to
    create a new interactive session for the next phase.
  - `next_agent` — the `qs-<phase>-QS-<N>` name for the new session.

`implement-task → review-task` is special: it renders **five** agents in
one transition (review orchestrator + four reviewer sub-roles). The
orchestrator's **reviewer sub-roles** are still Task-spawned as
non-interactive subagents (they just return findings).

There are **no slash commands for phases 2–5** because OpenCode command
frontmatter pins a static `agent:` name and cannot dispatch to
dynamically-named agents.

See `docs/opencode-workflow-guide.md` for the full description.

## Review architecture

`qs-review-task-QS-<N>` is an orchestrator that Task-spawns four hidden
reviewer subagents in parallel:

- `qs-review-blind-hunter-QS-<N>` — diff only, no repo context
- `qs-review-edge-case-hunter-QS-<N>` — diff + repo read-only
- `qs-review-acceptance-auditor-QS-<N>` — diff + story file
- `qs-review-coderabbit-QS-<N>` — wraps the existing CodeRabbit flow

The orchestrator consolidates and triages findings before presenting them
to the user.

## Rendering and cleanup

- `scripts/qs_opencode/render_agent.py --phase <p> --work-dir <w> --issue <N> --title <t> [--story-file ...] [--pr ...] [--extra KEY=VALUE ...]` renders one template into `<w>/.opencode/agents/qs-<p>-QS-<N>.md`. Refuses to overwrite without `--overwrite`. Templates live under `_qsprocess_opencode/agent_templates/*.md.tmpl` and use `{{VAR}}` substitution.
- `scripts/qs_opencode/cleanup_agents.py --work-dir <w> --issue <N>` removes every `qs-*-QS-<N>.md` under `<w>/.opencode/agents/`. Called by `qs-finish-task-QS-<N>` just before the worktree is deleted.

## Quality gate (non-negotiable)

All code work must pass `python scripts/qs/quality_gate.py` before commit.
The gate runs pytest with 100% coverage, ruff, mypy, and translations
validation.

## Commit discipline

Never commit or push without explicit user authorization. Phase prompts
present diffs and wait for "commit" / "push" / "open PR" / "merge" before
performing those git operations.

## Hands-off areas

Do **not** modify these while working in OpenCode — they belong to the
new static-agent pipeline (Claude Code / Cursor) and remain its source
of truth:

- `.claude/**`, `CLAUDE.md`
- `.cursor/**`, `.cursorrules`
- `scripts/qs/**` (except shared utilities both pipelines read)

OpenCode owns: `_qsprocess_opencode/agent_templates/`, `scripts/qs_opencode/`,
`.opencode/`, `AGENTS.md` (this file), `opencode.json`, and
`docs/opencode-workflow-guide.md`.

Shared, read-only from both pipelines: `docs/workflow/`, `docs/stories/`,
`docs/product/`, and `scripts/qs/quality_gate.py`.
