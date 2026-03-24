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
- **Translations**: `translations/en.json` is GENERATED — **NEVER edit it directly**. Edit `strings.json` instead, then run `bash scripts/generate-translations.sh`. This applies to both human and AI agents.

## Workflow Quick Reference

| You say | What happens |
|---------|-------------|
| "Fix this bug where..." | Issue → worktree → quick-dev → quality gates → PR |
| "I want to add a feature that..." | create-story → commit → issue → worktree → dev-story → quality gates → PR |
| "Create story 3.2" | create-story → commit story file. Stops here. |
| "Implement story 3.2" | Issue (if needed) → worktree → dev-story → quality gates → PR |
| "Work on issue #N" | Fetches issue → bug label = quick-dev, otherwise = feature flow. Skips issue creation. |
| "Merge PR #N" | Merge commit + delete branch + worktree cleanup |
| "Process PR feedback" | `/bmad-pr-review-feedback` → interactive comment processing |
| "Create a release" | Tag `vYYYY.MM.DD.XX`, release notes from merged PRs |
| Apply `auto-bmad` label on issue | Autonomous agent: branch → implement → quality gates → PR (cloud, no local setup) |

## Workflow Routing

When the user describes work to do, automatically select the right workflow. **Every development workflow MUST follow the lifecycle phases** from `development-lifecycle.md` — the routing below specifies which phases to execute.

| Intent | Workflow |
|--------|----------|
| **Bug fix / small fix** — the user describes a bug, defect, or small correction | Phase 1b (issue) → Phase 1c (worktree) → `/bmad-quick-dev-new-preview` inside the worktree → Phase 3 (quality gates, PR, review). |
| **Feature / new functionality** — the user describes new functionality or a significant change | `/bmad-create-story` → Phase 1a (commit story) → Phase 1b (issue) → Phase 1c (worktree) → `/bmad-dev-story` inside the worktree → Phase 3. |
| **Create story X.Y** — the user explicitly asks to create/plan a story | `/bmad-create-story` for the specified story → Phase 1a (commit story). Stop here — the user will ask for implementation separately. |
| **Implement story X.Y** — the user asks to implement/dev an existing story | Phase 1b (issue, if not yet created) → Phase 1c (worktree) → `/bmad-dev-story` for the specified story inside the worktree → Phase 3. |
| **From GitHub issue** — the user says "work on issue #N" or similar | Fetch the issue with `gh issue view N` and use the issue title and body as the initial intent/context. If it has the `bug` label, route to the bug flow. Otherwise route to the feature flow. In both cases, **skip Phase 1b** (issue already exists) and use issue number N for branch naming (`QS_N`). |
| **Merge PR** — the user asks to merge a PR | Follow Phase 3e (Merge & Cleanup) in `development-lifecycle.md`. |
| **Process PR feedback** — the user says "process PR feedback", "handle review comments", or "review feedback" | `/bmad-pr-review-feedback` — pulls unresolved PR review comments, presents with diff context, processes interactively (fix/discuss/reject/skip). |
| **Release** — the user asks to create a release, cut a release, or ship a version | Follow Phase 4 (Release) in `development-lifecycle.md`. |
| **Autonomous (auto-bmad)** — triggered by applying `auto-bmad` label to a GitHub issue | Runs entirely in CI via `.github/workflows/auto-bmad.yml`. See "Autonomous Flow" section in `development-lifecycle.md`. No local action needed — the agent handles everything cloud-side. |

Do NOT ask which workflow to use — infer from the user's description. When in doubt (ambiguous scope), default to the bug fix flow.

**Worktree mode**: All workflows create a git worktree by default (Phase 1c). If the user says "no worktree" (or similar), fall back to `git checkout -b` in the main directory. See the Appendix in `development-lifecycle.md` for details.

## Development Lifecycle

When developing stories or fixes, follow the full lifecycle in `_qsprocess/workflows/development-lifecycle.md`. This covers:
- Committing story artifacts to main before worktree creation (Phase 1a)
- GitHub issue creation (Phase 1b) and branch/worktree setup (Phase 1c)
- Quality gate commands and 100% coverage enforcement
- PR creation with risk assessment
- Worktree cleanup after merge

## Full Documentation

Before implementing any code, read these documents:

- **Development lifecycle**: `_qsprocess/workflows/development-lifecycle.md` — issue → branch → develop → PR
- **Code-level rules (42 rules)**: `_bmad-output/project-context.md` — naming, async, logging, error handling, testing anti-patterns
- **Architecture & patterns (9 patterns)**: `_bmad-output/planning-artifacts/architecture.md` — component model, decision map, implementation patterns, project structure, CI/CD strategy
- **Failure mode catalog**: `docs/failure-mode-catalog.md` — external dependency weaknesses, fallback behaviors, recovery paths (NFR20: update when new failure modes discovered)
