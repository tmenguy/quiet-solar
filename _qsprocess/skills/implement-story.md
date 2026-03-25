# /implement-story

Implement a story following TDD, enforce quality gates, create PR.

## Input

- `--issue N` (required): GitHub issue number. Branch is `QS_N`.
- `--story-file PATH` (optional): path to story artifact.

If `--story-file` not given, find it via `_bmad-output/implementation-artifacts/` matching the issue.
If no story file is found, STOP and ask the user for the story file path. Do not proceed without it.

## Prerequisites

You MUST be inside the worktree directory (`../quiet-solar-worktrees/QS_N/`).
Verify: `git branch --show-current` should return `QS_N`.

## Steps

### 1. Load project context

Read `_bmad-output/project-context.md` for coding rules. Do NOT load project-rules.md or development-lifecycle.md — this skill handles the process.

### 2. Implement using BMad dev-story

Run the BMad dev-story skill which handles the full TDD cycle:
```
/bmad-dev-story
```

When it asks for the story file, provide the `--story-file` path.

BMad dev-story handles:
- Loading story tasks and acceptance criteria
- Red-green-refactor TDD cycle for each task
- Marking tasks complete in the story file
- Running tests after each task
- Tracking implementation in the Dev Agent Record

**IMPORTANT**: When bmad-dev-story runs "linting and code quality checks", use this command instead of whatever it suggests:
```bash
python scripts/qs/quality_gate.py
```
This enforces ALL project gates: pytest 100% coverage + ruff lint + ruff format + mypy + translations.

### 3. Final quality gate enforcement

After bmad-dev-story completes, run the full quality gate one more time:

```bash
python scripts/qs/quality_gate.py --json
```

ALL gates MUST pass. If any fail (especially coverage < 100%), fix before proceeding. Do NOT skip this step. bmad-dev-story may consider itself done, but the project requires 100% coverage and all gates green.

### 4. Commit all changes

Stage all relevant files (NOT .DS_Store, venv, config, __pycache__):
```bash
git add custom_components/ tests/ _bmad-output/
git commit -m "{{descriptive_message}}"
```

### 5. Create PR

```bash
python scripts/qs/create_pr.py --title "{{title}}" --summary "{{bullets}}" --issue {{issue_number}}
```

### 6. Output review command

Tell the user:
```
Implementation complete. PR #{{pr_number}} created: {{url}}

To review, run in a new context:
  cd "{{main_worktree}}" && claude --name "Review QS_{{issue}}"
Then type:
  /review-story --pr {{pr_number}} --issue {{issue_number}}
```
