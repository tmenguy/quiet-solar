# Story 1.4: PR Templates & Developer Workflow

Status: review

## Story

As TheDev,
I want PR templates with quality checklists and risk assessment categories, issue templates for bugs and features, and CODEOWNERS for auto-assignment,
So that every contribution follows consistent quality standards with minimal friction.

## Acceptance Criteria

1. **Given** a new PR is created
   **When** the PR template is loaded
   **Then** it includes a checklist (tests, coverage, Ruff, MyPy, HACS manifest)
   **And** it includes risk assessment categories (CRITICAL/HIGH/MEDIUM/LOW per architecture risk table)

2. **Given** a new PR is created
   **When** the PR template is loaded
   **Then** auto-labeling assigns area labels based on changed file paths
   **And** CODEOWNERS maps file paths to reviewers

3. **Given** a new issue is opened
   **When** the issue template is loaded
   **Then** bug reports have structured fields (steps to reproduce, expected behavior, device type)
   **And** feature requests have structured fields (use case, persona, scope tier)

## Tasks / Subtasks

- [x] Task 1: Create PR template (AC: #1)
  - [x] 1.1 Create `.github/PULL_REQUEST_TEMPLATE.md` with quality checklist and risk assessment
- [x] Task 2: Create issue templates (AC: #3)
  - [x] 2.1 Create `.github/ISSUE_TEMPLATE/bug_report.yml` with structured fields
  - [x] 2.2 Create `.github/ISSUE_TEMPLATE/feature_request.yml` with structured fields
- [x] Task 3: Create CODEOWNERS (AC: #2)
  - [x] 3.1 Create `.github/CODEOWNERS` mapping file paths to @tmenguy
- [x] Task 4: Create labeler config and workflow (AC: #2)
  - [x] 4.1 Create `.github/labeler.yml` with area-based label rules
  - [x] 4.2 Create `.github/workflows/auto-label.yml` using `actions/labeler@v5`
- [x] Task 5: Create automation workflows (AC: #2)
  - [x] 5.1 Create `.github/workflows/issue-triage.yml` for issue auto-labeling
  - [x] 5.2 Create `.github/workflows/stale.yml` for stale issue/PR management

## Dev Notes

### PR Template Checklist (from architecture.md lines 528-544)

The architecture specifies this exact checklist:

```markdown
## Checklist
- [ ] Tests pass locally (`pytest tests/`)
- [ ] 100% coverage maintained
- [ ] Ruff format + check pass
- [ ] MyPy passes
- [ ] No new `# type: ignore` or `noqa` without justification
- [ ] HACS manifest.json updated if version changed

## Risk Assessment
<!-- Which row in the risk-weighted table does this PR touch? -->
- [ ] CRITICAL (solver, constraints, charger budgeting)
- [ ] HIGH (load base, constants, orchestration)
- [ ] MEDIUM (device-specific)
- [ ] LOW (platforms, UI, docs)
```

### Risk-Weighted CI Table (from architecture.md lines 445-461)

Use this for labeler and risk assessment context:

| Change area | Risk | Files |
|---|---|---|
| CRITICAL | Every user's bill/comfort, physical safety | `solver.py`, `constraints.py`, `charger.py`, `dynamic_group.py` |
| HIGH | Base contract, widespread constants, orchestration | `load.py`, `const.py`, `home.py`, `data_handler.py` |
| MEDIUM | Device-specific, contained blast radius | `car.py`, `person.py`, `battery.py`, `config_flow.py` |
| LOW | Entity display, dashboard visual | `sensor.py`, `switch.py`, `number.py`, `select.py`, `button.py`, `ui/` |

### Labeler Path Mappings

Map file paths to area labels for auto-labeling PRs:

- `custom_components/quiet_solar/home_model/solver.py` → `area:solver`
- `custom_components/quiet_solar/home_model/constraints.py` → `area:solver`
- `custom_components/quiet_solar/ha_model/charger.py` → `area:charger`
- `custom_components/quiet_solar/ha_model/dynamic_group.py` → `area:charger`
- `custom_components/quiet_solar/ha_model/car.py` → `area:car`
- `custom_components/quiet_solar/ha_model/person.py` → `area:person`
- `custom_components/quiet_solar/ha_model/battery.py` → `area:battery`
- `custom_components/quiet_solar/ha_model/home.py` → `area:home`
- `custom_components/quiet_solar/config_flow.py` → `area:config`
- `custom_components/quiet_solar/sensor.py`, `switch.py`, etc. → `area:platform`
- `custom_components/quiet_solar/ui/` → `area:ui`
- `.github/` → `area:ci`
- `tests/` → `area:tests`

### Issue Templates

Use YAML-based issue forms (`.yml` not `.md`) for structured fields:

**Bug report fields:** description, steps to reproduce, expected behavior, actual behavior, device type (dropdown: charger, car, battery, pool, home, solver, other), HA version, quiet-solar version

**Feature request fields:** use case description, persona (dropdown: TheAdmin, Magali, TheDev), scope tier (dropdown: MVP, Post-MVP), additional context

### Automation Workflows (from architecture.md lines 500-518)

**auto-label.yml:** Use `actions/labeler@v5` triggered on `pull_request` to auto-assign area labels based on changed file paths using `.github/labeler.yml` config.

**issue-triage.yml:** Auto-label issues by keywords in title/body (bug, feature, solver, charger, documentation). Triggered on `issues: [opened]`.

**stale.yml:** Use `actions/stale@v9`. Mark stale issues/PRs after 30 days inactivity. Close after 60 days with polite message. Exempt labels: `pinned`, `security`, `critical`.

### CODEOWNERS

Single owner for now — `@tmenguy` owns everything. Structure for future expansion:

```
# Default owner
* @tmenguy

# Solver (critical)
custom_components/quiet_solar/home_model/ @tmenguy

# HA integration layer
custom_components/quiet_solar/ha_model/ @tmenguy

# CI/CD
.github/ @tmenguy
```

### What NOT to do

- Do NOT modify any existing workflows (`pr-quality.yml`, `release.yml`)
- Do NOT modify any production Python code
- Do NOT add branch protection rules (that's GitHub settings, not code)
- Do NOT create Tier 2 merge gate workflow — that would be a separate story
- Do NOT add `auto-review.yml` with risk assessment comments — keep it simple, the PR template checklist is sufficient for now

### Previous story intelligence

**Story 1.2:** Created `.github/workflows/pr-quality.yml` — 4 parallel jobs (lint, typecheck, test, hacs-validate). CI issues: pyproject.toml was untracked, wallbox import needed conditional fallback, manifest.json needed issue_tracker, repo needed topics.

**Story 1.3:** Created `.github/workflows/release.yml` — sequential quality-gate → hacs-validate → version-check → release. Updated development-lifecycle.md Phase 4.

**Key learnings:**
- GitHub Actions workflows go in `.github/workflows/`
- HACS validation requires `issue_tracker` in manifest.json and GitHub repo topics
- CI timezone must be `Europe/Paris` (`TZ: Europe/Paris` env var) to match dev environment
- Always use `actions/checkout@v4` and `actions/setup-python@v5`

### Project Structure Notes

All new files go under `.github/`:

```
.github/
├── CODEOWNERS
├── PULL_REQUEST_TEMPLATE.md
├── ISSUE_TEMPLATE/
│   ├── bug_report.yml
│   └── feature_request.yml
├── labeler.yml
└── workflows/
    ├── auto-label.yml          (new)
    ├── issue-triage.yml        (new)
    ├── pr-quality.yml          (existing - DO NOT MODIFY)
    ├── release.yml             (existing - DO NOT MODIFY)
    └── stale.yml               (new)
```

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#CI/CD Pipeline Architecture] — PR template spec (lines 528-544), supporting files (lines 520-525), automation workflows (lines 500-518), risk table (lines 445-461)
- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.4] — acceptance criteria (lines 282-301)
- [Source: _bmad-output/planning-artifacts/prd.md#FR44] — CI runs quality gates on every PR
- [Source: _bmad-output/planning-artifacts/architecture.md#AR9] — PR template with checklist and risk assessment
- [Source: _bmad-output/planning-artifacts/prd.md#NFR23] — Developer workflows require minimal manual steps
- [Source: _bmad-output/implementation-artifacts/1-2-pr-quality-gate-pipeline.md] — Story 1.2 learnings
- [Source: _bmad-output/implementation-artifacts/1-3-release-pipeline-version-migration.md] — Story 1.3 learnings

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- PR template with quality checklist (tests, coverage, Ruff, MyPy, HACS) and risk assessment (CRITICAL/HIGH/MEDIUM/LOW)
- Bug report and feature request issue templates using YAML forms with structured fields
- CODEOWNERS mapping all paths to @tmenguy
- Labeler config with 11 area labels + auto-label workflow using actions/labeler@v5
- Issue triage workflow using github-script for keyword-based auto-labeling
- Stale workflow using actions/stale@v9 (30 days stale, 60 days close)
- 3843 tests pass at 100% coverage, no regressions

### Change Log

- 2026-03-20: Story 1.4 implemented — PR templates, issue templates, CODEOWNERS, labeler, automation workflows

### File List

New files:
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/ISSUE_TEMPLATE/bug_report.yml`
- `.github/ISSUE_TEMPLATE/feature_request.yml`
- `.github/CODEOWNERS`
- `.github/labeler.yml`
- `.github/workflows/auto-label.yml`
- `.github/workflows/issue-triage.yml`
- `.github/workflows/stale.yml`

Modified files:
- `_bmad-output/implementation-artifacts/1-4-pr-templates-developer-workflow.md` (story status + tasks)
