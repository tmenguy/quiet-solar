# Story 1.1: Agentic Development Workflow

Status: in-progress

## Story

As TheDev,
I want local quality tooling (Ruff, MyPy) configured and passing on the existing codebase, and the standard BMad `create-story` → `dev-story` pipeline to handle the full development lifecycle including GitHub operations (issue, branch, PR),
So that I can develop fixes and features with quality gates enforced locally, using the standard BMad workflow without custom one-off skills.

## Acceptance Criteria

1. **Given** TheDev runs Ruff on the codebase
   **When** `ruff check custom_components/quiet_solar/` executes
   **Then** all checks pass with zero violations

2. **Given** TheDev runs MyPy on the codebase
   **When** `mypy custom_components/quiet_solar/` executes
   **Then** it reports "Success: no issues found"

3. **Given** TheDev runs the full test suite with coverage
   **When** `pytest tests/ --cov=custom_components/quiet_solar --cov-report=term-missing` executes
   **Then** all tests pass with 100% coverage

4. **Given** TheDev uses the standard BMad `create-story` → `dev-story` pipeline
   **When** developing a fix or feature
   **Then** the pipeline handles issue creation, branch setup, development with quality gates, and PR creation
   **And** no custom one-off workflow skills are needed

## Tasks / Subtasks

- [x] Task 1: Install prerequisite tooling configuration
  - [x] 1.1 Add Ruff configuration to `pyproject.toml` (created new file)
  - [x] 1.2 Add MyPy configuration to `pyproject.toml`
  - [x] 1.3 Added ruff>=0.9.0 and mypy>=1.13.0 to `requirements_test.txt`
  - [x] 1.4 Verify `gh` CLI installed and authenticated

- [x] Task 2: Achieve clean Ruff and MyPy baseline
  - [x] 2.1 Ruff auto-fixed 2758 safe violations across production + test code
  - [x] 2.2 Ruff formatted 144 files (32 production + 101 test + 11 already formatted)
  - [x] 2.3 Configured 20 brownfield Ruff rule ignores
  - [x] 2.4 MyPy passes clean (configured 23 brownfield error code exclusions — tighten progressively)

- [x] Task 3: Validate with real bug fix (acceptance test)
  - [x] 3.1 Bug #8 (car user state overwritten) developed using standard BMad pipeline
  - [x] 3.2 GitHub issue #8 created, branch `fix/8-car-user-state-overwritten` created
  - [x] 3.3 Fix developed with 100% coverage maintained (3842 tests, 13709 statements)
  - [x] 3.4 Quality gates passed: pytest 100%, ruff clean, mypy clean
  - [x] 3.5 PR #9 created via `gh pr create` with risk assessment

- [x] Task 4: Create tool-agnostic project process docs in `_qsprocess/`
  - [x] 4.1 Create `_qsprocess/rules/project-rules.md` — shared project rules for both Claude and Cursor
  - [x] 4.2 Create `_qsprocess/workflows/development-lifecycle.md` — full lifecycle: issue → branch `QS_{{id}}` → develop with quality gates → PR
  - [x] 4.3 Update `CLAUDE.md` to reference `_qsprocess/rules/project-rules.md`
  - [x] 4.4 Create `.cursorrules` to reference `_qsprocess/rules/project-rules.md`
  - [x] 4.5 Dropped custom `bmad-bug-fix` and `bmad-quality-check` skills (redundant with standard BMad pipeline + `_qsprocess/` docs)

## Dev Notes

### What exists today

- **`pyproject.toml`** — Ruff + MyPy configuration (created by this story)
- **`pytest.ini`** — already configured with markers, asyncio_mode=auto, coverage options
- **`tests/conftest.py`** — FakeHass, FakeConfigEntry, integration fixtures
- **`tests/factories.py`** — factory functions for all domain test doubles
- **`requirements_test.txt`** — includes ruff>=0.9.0 and mypy>=1.13.0

### Architecture constraints

- **Two-layer boundary**: domain logic (`home_model/`) NEVER imports `homeassistant.*`
- **Solver step size**: `SOLVER_STEP_S = 900` in `const.py` — don't touch
- **All config keys in `const.py`** — never hardcode strings
- **Async rules**: no blocking calls in async code
- **Logging**: lazy logging with `%s`, no f-strings in log calls

### Decision: Drop custom skills in favor of standard BMad pipeline

The original plan created `bmad-bug-fix` and `bmad-quality-check` as custom skills. After using them on bug #8, we found:
- They duplicated what `create-story` → `dev-story` already does (but worse — no sprint tracking, no red-green-refactor, no task tracking)
- The only added value was 3 `gh` CLI commands (issue create, branch create, PR create) which are trivial and don't need a dedicated skill
- The standard BMad pipeline is more rigorous and already validated

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.1] — acceptance criteria and user story
- [Source: _bmad-output/planning-artifacts/prd.md#FR43-FR45] — developer workflow FRs
- [Source: _bmad-output/planning-artifacts/architecture.md#CI/CD Pipeline Architecture] — 3-tier pipeline design
- [Source: _bmad-output/project-context.md] — 42 code rules for implementation

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- Created `pyproject.toml` with Ruff config (10 rule categories selected, 20 brownfield ignores, per-file ignores for `__init__.py` and `tests/`)
- Created MyPy config (23 error codes disabled for brownfield baseline — tighten progressively)
- Ruff auto-fixed 2758 safe violations across production + test code
- Ruff formatted 144 files (32 production + 101 test + 11 already formatted)
- Added ruff>=0.9.0 and mypy>=1.13.0 to requirements_test.txt
- Bug #8 served as real-world acceptance test: issue created, branch created, fix developed with 100% coverage, PR #9 created
- Dropped custom `bmad-bug-fix` and `bmad-quality-check` skills — standard BMad pipeline is sufficient
- All 3842 tests pass at 100% coverage after all changes

### Change Log

- 2026-03-18: Story 1.1 implemented — tooling config, codebase formatted
- 2026-03-19: Bug #8 fix validated the workflow end-to-end (PR #9 created)
- 2026-03-19: Dropped custom bmad-bug-fix and bmad-quality-check skills — standard BMad pipeline preferred
- 2026-03-19: Created `_qsprocess/` with shared project rules and development lifecycle docs (Claude + Cursor compatible)

### File List

New files:
- `pyproject.toml`
- `_qsprocess/rules/project-rules.md`
- `_qsprocess/workflows/development-lifecycle.md`
- `.cursorrules`

Modified files:
- `CLAUDE.md` (now references `_qsprocess/rules/project-rules.md`)
- `requirements_test.txt` (added ruff, mypy)
- `custom_components/quiet_solar/**` (ruff auto-fix + format — 32 files)
- `tests/**` (ruff auto-fix + format — 101 files)

Deleted files:
- `.claude/skills/bmad-bug-fix/workflow.md`
- `.claude/skills/bmad-quality-check/workflow.md`
