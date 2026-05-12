# Quiet Solar — Project Rules

## Process authority

All workflow rules, phase protocols, and code-style rules live under
`docs/workflow/`. Harness-specific config (`.claude/`, `.cursor/`,
`.opencode/`) references these docs; it never duplicates them.

## Project overview

Quiet Solar is a Home Assistant custom component that optimizes solar
energy self-consumption through a constraint-based solver.

## Commands

Activate `source venv/bin/activate` for all Python commands.

```bash
# Run all quality gates (pytest 100% coverage + ruff + mypy + translations)
python scripts/qs/quality_gate.py

# Auto-fix formatting and lint
python scripts/qs/quality_gate.py --fix

# JSON output for scripts
python scripts/qs/quality_gate.py --json

# Run tests only
source venv/bin/activate && pytest tests/ -v

# Single test
source venv/bin/activate && pytest tests/test_solver.py::test_function_name -v
```

## Architecture constraints

- **Two-layer boundary**: `home_model/` NEVER imports `homeassistant.*`. `ha_model/` bridges both.
- **Solver step size**: `SOLVER_STEP_S = 900` in `const.py` — don't touch.
- **All config keys in `const.py`** — never hardcode strings.
- **Async rules**: no blocking calls in async code, use `hass.async_add_executor_job()`.
- **Logging**: lazy `%s`, no f-strings in log calls, no periods at end.
- **Translations**: NEVER edit `translations/en.json` — edit `strings.json`, run `bash scripts/generate-translations.sh`.

## Workflow routing

Each phase runs as an interactive `claude --agent qs-<phase>` session
(preferred — open a fresh terminal) or as a `/<phase>` slash command
(fallback — degraded one-shot UX kept for Claude Desktop). Do NOT ask
which phase to use — infer from context.

| You say                                                      | Preferred launcher                       | Fallback           |
| ------------------------------------------------------------ | ---------------------------------------- | ------------------ |
| "Setup task 3.2" / describe feature / "work on issue #42"    | `claude --agent qs-setup-task` on main   | `/setup-task`      |
| "Create plan" (inside worktree)                              | `claude --agent qs-create-plan`          | `/create-plan`     |
| "Implement task" (inside worktree)                           | `claude --agent qs-implement-task`       | `/implement-task`  |
| "Review PR #5" or "review task"                              | `claude --agent qs-review-task`          | `/review-task`     |
| "Merge PR #5" or "finish task"                               | `claude --agent qs-finish-task`          | `/finish-task`     |
| "Create a release"                                           | `claude --agent qs-release` on main      | `/release`         |
| Bug fix / small fix                                          | `claude --agent qs-setup-task` on main   | `/setup-task`      |

See [overview.md](overview.md) section "Orchestrators are interactive
sessions; sub-agents are parallel fan-out" for the rationale.

Each command delegates to a static agent under `.claude/agents/` (or
`.cursor/agents/`). Agents discover task context at runtime from
`git branch --show-current` — there is no per-task agent rendering.

**Commit authorization**: agents are authorized to commit and push as
part of their defined workflow steps (e.g., the implement-task agent
auto-commits and opens a PR after the quality gate passes). Outside of
agent-driven phases, always ask the user before committing.

## Code rules reference

Before implementing code, read [project-context.md](project-context.md)
for the full 42-rule set covering naming, async, logging, error
handling, and testing patterns.
