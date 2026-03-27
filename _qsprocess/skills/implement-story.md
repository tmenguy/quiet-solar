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

### 2.5. Inline doc-sync (continuous)

Throughout implementation, watch for user direction that changes scope, acceptance criteria, technical approach, or tasks. When detected:

1. **Flag it**: tell the user what changed and which documents are affected
2. **Propose edits**: show specific changes to the story artifact (`_bmad-output/implementation-artifacts/*.md`) — e.g., updated ACs, modified tasks, new dev notes
3. **Wait for approval**: do NOT apply changes until the user confirms
4. **Apply if accepted**: edit the story file in place

If the change is structural (e.g., new architecture pattern, new project rule), also mention potential updates to `architecture.md` or `_qsprocess/rules/project-rules.md` — but propose those separately, not bundled with story edits. Most implementation-level adjustments only need story artifact updates.

### 3. Final quality gate enforcement

After bmad-dev-story completes, run the full quality gate one more time:

```bash
python scripts/qs/quality_gate.py --json
```

ALL gates MUST pass. If any fail (especially coverage < 100%), fix before proceeding. Do NOT skip this step. bmad-dev-story may consider itself done, but the project requires 100% coverage and all gates green.

### 3.5. Compound doc-sync

Before committing, review everything that happened during the session:

1. **Scan**: all user direction given, implementation decisions that deviated from the original spec, and new information discovered during implementation
2. **Propose**: a consolidated list of story artifact updates — AC adjustments, task modifications, dev notes additions. Present each proposed change clearly.
3. **Secondary docs**: if implementation revealed structural gaps, optionally propose updates to `architecture.md` or `_qsprocess/rules/project-rules.md` (separate from story updates)
4. **User approves/rejects** each item
5. **Apply**: edit approved changes in place. Modified documents will be included in the commit.

If inline doc-sync (Step 2.5) already captured all changes, say so and move on — don't force unnecessary updates.

### 4. Commit all changes

Stage all relevant files (NOT .DS_Store, venv, config, __pycache__):
```bash
git add custom_components/ tests/ _bmad-output/ _qsprocess/ scripts/
git commit -m "{{descriptive_message}}"
```

### 5. Create PR

```bash
python scripts/qs/create_pr.py --title "{{title}}" --summary "{{bullets}}" --issue {{issue_number}}
```

### 6. Output review command

Run `next_step.py` to generate both command options:

```bash
python scripts/qs/next_step.py --skill review-story --issue {{issue_number}} --pr {{pr_number}} --work-dir {{worktree_path}} --title "{{title}}"
```

Parse the JSON output and tell the user:

```
Implementation complete. PR #{{pr_number}} created: {{url}}

**Option A — New context (copy-paste this single command):**
  {{new_context}}

**Option B — Same context:**
  {{same_context}}
```
