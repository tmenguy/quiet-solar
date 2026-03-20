# Quiet Solar — Project Rules

## Process Authority

All project rules, workflows, and process decisions live in `_qsprocess/` — this is the single source of truth. This folder is tool-agnostic: it works identically whether the agent is Claude Code, Cursor, or any other tool.

- **Never** store project process rules exclusively in tool-specific locations (`.claude/`, `.cursor/`, etc.)
- Tool-specific config files (`CLAUDE.md`, `.cursorrules`) must reference `_qsprocess/`, not duplicate its content
- When a new process rule is established, add it here — not only in agent memory

## Project Overview

Quiet Solar is a Home Assistant custom component that optimizes solar energy self-consumption through a constraint-based solver.

## Commands

ALWAYS use the project's virtual environment (`./venv`) for running pytest and all Python commands.

```bash
# Activate venv first
source venv/bin/activate

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_solver.py

# Run a single test
pytest tests/test_solver.py::test_function_name -v

# Run with coverage (100% is mandatory)
pytest tests/ --cov=custom_components/quiet_solar --cov-report=term-missing

# Run only unit or integration tests
pytest tests/ -m unit
pytest tests/ -m integration
```

## Quality Gates

Before any PR or completion claim, ALL of these must pass:

1. `pytest tests/ --cov=custom_components/quiet_solar --cov-report=term-missing` — 100% coverage mandatory
2. `ruff check custom_components/quiet_solar/` — zero violations
3. `ruff format --check custom_components/quiet_solar/` — all formatted
4. `mypy custom_components/quiet_solar/` — no issues found

## Architecture Constraints

- **Two-layer boundary**: domain logic (`home_model/`) NEVER imports `homeassistant.*`. HA bridge layer (`ha_model/`) may import both.
- **Solver step size**: `SOLVER_STEP_S = 900` in `const.py` — don't touch.
- **All config keys in `const.py`** — never hardcode strings.
- **Async rules**: no blocking calls in async code, use `hass.async_add_executor_job()` for blocking operations.
- **Logging**: lazy logging with `%s`, no f-strings in log calls, no periods at end.

## Workflow Quick Reference

| You say | What happens |
|---------|-------------|
| "Fix this bug where..." | Creates issue, branch `QS_N`, implements, quality gates, PR |
| "I want to add a feature that..." | Full story planning, issue, branch `QS_N`, implements, quality gates, PR |
| "Work on issue #N" | Fetches issue — bug label → bug flow, otherwise → feature flow. Skips issue creation. |
| "Merge PR #N" | Merge commit + delete branch + worktree cleanup |
| "Create a release" | Tag `vYYYY.MM.DD.XX`, release notes from merged PRs |

## Workflow Routing

When the user describes work to do, automatically select the right workflow:

| Intent | Workflow |
|--------|----------|
| **Bug fix / small fix** — the user describes a bug, defect, or small correction | Use `/bmad-quick-dev-new-preview`. It will pick up the development lifecycle rules below. |
| **Feature / enhancement** — the user describes new functionality or a significant change | Use `/bmad-create-story` to plan, then `/bmad-dev-story` to implement. |
| **From GitHub issue** — the user says "work on issue #N" or similar | Fetch the issue with `gh issue view N`. If it has the `bug` label, route to the bug flow (`/bmad-quick-dev-new-preview`). Otherwise route to the feature flow (`/bmad-create-story` → `/bmad-dev-story`). In both cases, **skip GitHub issue creation** (Phase 1a) — the issue already exists. Still create the branch `QS_N` (Phase 1b) and follow the rest of the lifecycle. Use the issue title and body as the initial intent/context for the selected workflow. |
| **Merge PR** — the user asks to merge a PR | Follow Phase 3e (Merge PR) in `_qsprocess/workflows/development-lifecycle.md`. |
| **Release** — the user asks to create a release, cut a release, or ship a version | Follow Phase 4 (Release) in `_qsprocess/workflows/development-lifecycle.md`. |

Do NOT ask which workflow to use — infer from the user's description. When in doubt (ambiguous scope), default to `/bmad-quick-dev-new-preview`.

**Worktree mode**: All workflows create a git worktree by default (Phase 1b). If the user says "no worktree" (or similar), fall back to `git checkout -b` in the main directory. See the Appendix in `development-lifecycle.md` for details.

## Development Lifecycle

When developing stories or fixes, follow the full lifecycle in `_qsprocess/workflows/development-lifecycle.md`. This covers:
- GitHub issue and branch creation (branch naming: `QS_{{issue_number}}`)
- Worktree setup by default (symlinks venv, config, non-git custom_components)
- Quality gate commands and 100% coverage enforcement
- PR creation with risk assessment
- Worktree cleanup after merge

## Full Documentation

Before implementing any code, read these documents:

- **Development lifecycle**: `_qsprocess/workflows/development-lifecycle.md` — issue → branch → develop → PR
- **Code-level rules (42 rules)**: `_bmad-output/project-context.md` — naming, async, logging, error handling, testing anti-patterns
- **Architecture & patterns (9 patterns)**: `_bmad-output/planning-artifacts/architecture.md` — component model, decision map, implementation patterns, project structure, CI/CD strategy
- **Failure mode catalog**: `docs/failure-mode-catalog.md` — external dependency weaknesses, fallback behaviors, recovery paths (NFR20: update when new failure modes discovered)
