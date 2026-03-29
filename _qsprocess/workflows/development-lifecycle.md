# Quiet Solar — Development Lifecycle

This document describes the development pipeline. Each phase is implemented by a skill backed by Python scripts in `scripts/qs/`. Follow the skills — they handle the details.

## Pipeline Overview

```
/create-story → /setup-story → /implement-story → /review-story → /finish-story → /release
```

Each skill runs in its own context. The output of one skill gives you the command to launch the next.

## Phase 1: Story Creation (`/create-story`)

Creates a story artifact file and commits it on a feature branch `QS_N`.

- Creates GitHub issue via `scripts/qs/create_issue.py`
- Creates branch `QS_N` (not on main — main is protected)
- Writes story file to `_bmad-output/implementation-artifacts/`
- Commits and pushes

## Phase 2: Setup (`/setup-story`)

Sets up a worktree for parallel development.

- Creates worktree via `scripts/worktree-setup.sh`
- Outputs tool-appropriate launch instructions via `build_next_step()` from `scripts/qs/utils.py` (auto-detects Cursor vs Claude Code)
- The implementation context is isolated from the main worktree

## Phase 3: Implementation (`/implement-story`)

Runs inside the worktree. TDD cycle with enforced quality gates.

- Reads story file for tasks and ACs
- For each task: write failing tests → implement → run `scripts/qs/quality_gate.py`
- 100% coverage is enforced by the script (non-negotiable)
- Creates PR via `scripts/qs/create_pr.py` with risk assessment

## Phase 4: Review (`/review-story`)

Code review + GitHub Copilot review.

- Triggers Copilot review via `scripts/qs/review_pr.py`
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
