# Quiet Solar — OpenCode pipeline

This file is the OpenCode entry point. OpenCode does NOT read
`CLAUDE.md`; project rules and code-style rules are canonical under
`docs/workflow/` (shared across harnesses), while OpenCode-specific
launcher + HTTP-API session spawn live under
`scripts/qs/launchers/opencode.py` and `scripts/qs/spawn_session.py`.

## Project overview

Quiet Solar is a Home Assistant custom component that optimizes solar
energy self-consumption through a constraint-based solver.

## Required reading (load with the Read tool on startup)

- `docs/workflow/project-rules.md` — commands, architecture
  constraints, workflow routing
- `docs/workflow/project-context.md` — 42-rule code style set
- `docs/workflow/overview.md` — static-agent pipeline overview

## Commands

Activate `source venv/bin/activate` for all Python commands.
`scripts/qs/quality_gate.py` is the **single test entry point** — it
owns the cache, `pytest-xdist` parallelization, `COVERAGE_CORE=sysmon`,
and scope detection. Raw `pytest` bypasses all four; use it only for
ad-hoc single-node debugging (positional must contain `::`).

```bash
# Full quality gate (pytest 100% cov + ruff + mypy + translations).
python scripts/qs/quality_gate.py

# Cached dev-loop default — skips gates when git state matches last pass.
python scripts/qs/quality_gate.py --cache

# Auto-fix formatting and lint.
python scripts/qs/quality_gate.py --fix

# Fast iteration on one or more test paths (files or directories).
python scripts/qs/quality_gate.py --quick tests/test_solver.py
python scripts/qs/quality_gate.py --quick tests/ha_tests

# JSON output for scripts.
python scripts/qs/quality_gate.py --json
```

## Architecture constraints

- **Two-layer boundary**: `home_model/` NEVER imports `homeassistant.*`. `ha_model/` bridges both.
- **Solver step size**: `SOLVER_STEP_S = 900` in `const.py` — don't touch.
- **All config keys in `const.py`** — never hardcode strings.
- **Async rules**: no blocking calls in async code, use `hass.async_add_executor_job()`.
- **Logging**: lazy `%s`, no f-strings in log calls, no periods at end.
- **Translations**: NEVER edit `translations/en.json` — edit `strings.json`, run `bash scripts/generate-translations.sh`.

## Pipeline — seven static agents

Each phase runs as an OpenCode session with the corresponding static
agent activated (via the agent picker in the OpenCode UI):

| Phase                | Static agent              | Where         |
| -------------------- | ------------------------- | ------------- |
| Setup task           | `qs-setup-task`           | main checkout |
| Create plan          | `qs-create-plan`          | worktree      |
| Implement (product)  | `qs-implement-task`       | worktree      |
| Implement (dev-env)  | `qs-implement-setup-task` | worktree      |
| Review task          | `qs-review-task`          | worktree      |
| Finish task          | `qs-finish-task`          | worktree      |
| Release              | `qs-release`              | main checkout |

Static agents live in `.opencode/agents/`. They discover task context
at runtime via `python scripts/qs/context.py` (no per-task rendering).

Phase handoffs additionally emit a
`python scripts/qs/spawn_session.py --agent <name> --directory <wd>`
one-liner the user may paste into a fresh terminal to spawn an in-band
session bound to the next phase. See `docs/workflow/overview.md` and
`docs/workflow/harness.md` for the launcher dispatch model.

## No slash commands

OpenCode commands accept a single static `agent:` and cannot
dynamically dispatch — the UI agent picker is the canonical activation
surface, so `.opencode/commands/` is intentionally absent.

## Legacy

The previous per-task-rendering OpenCode pipeline lives under
`legacy/`. Frozen historical code — do not modify or import. Will be
deleted after the static-agent pipeline is proven in production.

## Commit discipline

See `docs/workflow/project-rules.md` § "Workflow routing" and "Commit
authorization". Agents are authorized to commit and push as part of
their defined workflow steps (e.g., `qs-implement-task` auto-commits
+ opens a PR after the quality gate passes). Outside of agent-driven
phases, always ask the user before committing.
