# Quiet Solar — Project Rules

## Process Authority

All project rules, workflows, and skills live in `_qsprocess/`. Tool-specific config (`.claude/`, `.cursor/`) references `_qsprocess/`, never duplicates it.

## Project Overview

Quiet Solar is a Home Assistant custom component that optimizes solar energy self-consumption through a constraint-based solver.

## Commands

Use `./venv` for all Python commands. Quality gates are automated:

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

## Architecture Constraints

- **Two-layer boundary**: `home_model/` NEVER imports `homeassistant.*`. `ha_model/` bridges both.
- **Solver step size**: `SOLVER_STEP_S = 900` in `const.py` — don't touch.
- **All config keys in `const.py`** — never hardcode strings.
- **Async rules**: no blocking calls in async code, use `hass.async_add_executor_job()`.
- **Logging**: lazy `%s`, no f-strings in log calls, no periods at end.
- **Translations**: NEVER edit `translations/en.json` — edit `strings.json`, run `bash scripts/generate-translations.sh`.

## Workflow Routing

Use skills (`/command`) for all development work. Do NOT ask which to use — infer from context.

| You say | Skill |
|---------|-------|
| "Setup task 3.2" or describe feature or "work on issue #42" | `/setup-task` |
| "Create plan" (inside worktree) | `/create-plan` |
| "Implement story" (inside worktree) | `/implement-story` |
| "Review PR #5" or "review story" | `/review-story` |
| "Merge PR #5" or "finish story" | `/finish-story` |
| "Create a release" | `/release` |
| Bug fix / small fix | `/setup-task` (create issue + branch + worktree directly) |

Each skill is defined in `_qsprocess/skills/` and handles all steps including quality gates.

**Commit authorization**: Skills are authorized to commit and push as part of their defined workflow steps. Outside of skill execution, always ask the user before committing.

## Code Rules Reference

Before implementing code, read `_bmad-output/project-context.md` for the full 42-rule set covering naming, async, logging, error handling, and testing patterns.
