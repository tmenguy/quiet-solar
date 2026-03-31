# Quiet Solar — Development Lifecycle

This document describes the development pipeline. Each phase is implemented by a skill backed by Python scripts in `scripts/qs/`. Follow the skills — they handle the details.

## Pipeline Overview

```
/setup-task → /create-plan → /implement-story → /review-story → /finish-story → /release
```

Each skill runs in its own context. The output of one skill gives you the command to launch the next.

## Phase 1: Task Setup (`/setup-task`)

Creates a GitHub issue (if needed), feature branch, and worktree — without touching main's checkout state. This is fast and can be run multiple times in parallel to set up several tasks.

- Creates GitHub issue via `scripts/qs/create_issue.py` (if not provided)
- Creates branch `QS_N` from `origin/main` (main is never checked out)
- Creates worktree via `scripts/worktree-setup.sh` (unless `--no-worktree`)
- Outputs tool-appropriate launch instructions for `/create-plan`

## Phase 2: Plan Creation (`/create-plan`)

Runs inside the worktree (or on `QS_N` branch). Creates the story artifact file.

- Fetches issue details for context
- Writes story file to `_bmad-output/implementation-artifacts/` via `bmad-create-story`
- Commits and pushes the story file
- Outputs tool-appropriate launch instructions for `/implement-story`

## Phase 3: Implementation (`/implement-story`)

Runs inside the worktree. TDD cycle with enforced quality gates.

- Reads story file for tasks and ACs
- For each task: write failing tests → implement → run `scripts/qs/quality_gate.py`
- 100% coverage is enforced by the script (non-negotiable)
- Creates PR via `scripts/qs/create_pr.py` with risk assessment

## Phase 4: Review (`/review-story`)

Code review + CodeRabbit review.

- CodeRabbit auto-reviews on PR creation/push (no trigger needed)
- Runs local adversarial review on the diff
- Fetches and processes all review comments interactively
- Quality gates re-run after fixes

## Phase 5: Finish (`/finish-story`)

Merge, cleanup, update.

- Final quality gate via `scripts/qs/quality_gate.py`
- Merges PR (merge commit, not squash) via `scripts/qs/finish_story.py`
- Cleans up worktree via `scripts/worktree-cleanup.sh`
- Updates epics.md with `[DONE]` status
- Pulls latest main

## Phase 6: Release (`/release`)

Tag and ship.

- Determines next tag `vYYYY.MM.DD.XX` via `scripts/qs/release.py`
- Bumps `manifest.json` version
- Commits, tags, pushes — GitHub Actions handles the rest

## Quality Gates

All gates are run by a single script: `python scripts/qs/quality_gate.py`

| Gate | Tool | Requirement |
|------|------|-------------|
| Tests | pytest | 100% coverage, zero failures |
| Lint | ruff check | Zero violations |
| Format | ruff format | All formatted |
| Types | mypy | No issues |
| Translations | generate-translations.sh | en.json up to date |

Use `--fix` to auto-fix formatting and lint. Use `--json` for structured output.

## Risk Categories

Auto-detected by `scripts/qs/create_pr.py` from changed files:

| Level | Files |
|-------|-------|
| CRITICAL | solver.py, constraints.py, charger.py, dynamic_group.py |
| HIGH | load.py, const.py, ha_model/home.py, ha_model/device.py |
| MEDIUM | ha_model/person.py, car.py, battery.py, solar.py, config_flow.py |
| LOW | sensor.py, switch.py, number.py, select.py, button.py, ui/ |

## Autonomous Flow (auto-bmad)

Apply `auto-bmad` label to a GitHub issue → `.github/workflows/auto-bmad.yml` runs a cloud agent that follows the same pipeline. No local action needed.

## Worktree Reference

```
~/Developer/homeassistant/
  quiet-solar/                    # main worktree (always on main)
    scripts/qs/                   # automation scripts
    venv/                         # shared venv
  quiet-solar-worktrees/
    QS_42/                        # worktree for issue #42
      venv -> ../../quiet-solar/venv
```

Setup: `bash scripts/worktree-setup.sh N`
Cleanup: `bash scripts/worktree-cleanup.sh N`
