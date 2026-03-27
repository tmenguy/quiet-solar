# Story 1.12: Systematic Finish-Story Workflow Enhancement

Status: review

issue: 51
branch: "QS_51"

## Story

As TheDev,
I want `/finish-story` to require zero arguments — it auto-detects the branch, auto-commits pending changes, auto-creates the PR if missing, runs all gates and validations, merges, and cleans up — all driven by a Python script, not agent rules,
So that finishing a story is a single command that handles everything end-to-end with no manual steps and no loose ends.

## Acceptance Criteria

1. **Given** TheDev runs `/finish-story` with no arguments from a feature branch (QS_N)
   **When** the script starts
   **Then** it auto-detects the current branch, derives the issue number, discovers the story file, and proceeds without requiring any `--pr`, `--issue`, or `--story-key` flags

2. **Given** there are uncommitted changes on the feature branch (staged or unstaged in `custom_components/`, `tests/`, `_bmad-output/`, `_qsprocess/`, `scripts/`)
   **When** the pre-merge phase starts
   **Then** the script auto-stages relevant files (excluding `.DS_Store`, `venv/`, `config/`, `__pycache__/`), commits them with a descriptive message, and pushes

3. **Given** no PR exists for the current branch
   **When** the script checks for an existing PR
   **Then** it auto-creates a PR against main, linking the issue, with a generated title and summary from the story file and git log

4. **Given** a PR exists (found or just created)
   **When** the pre-merge validation phase runs
   **Then** the script runs the doc-sync gate, the local quality gate, verifies CI checks passed on the remote, verifies PR body links the issue (adds link if missing), and reports all results as structured JSON
   **And** the script blocks on failures (quality gate, CI) but reports everything so the agent can present it

5. **Given** the merge completes
   **When** the post-merge phase runs
   **Then** the script closes the linked issue if still open, updates the story artifact status to "done", updates epics.md, cleans up the worktree, and pulls main
   **And** all post-merge steps are in the script, not in skill instructions

6. **Given** all steps complete (or a step fails)
   **When** the script outputs its report
   **Then** the JSON includes every step's status, and on failure includes a `recovery` field with specific instructions
   **And** the script suggests `/release` if production code changed, or "no release needed" if only process files changed

7. **Given** the skill file `_qsprocess/skills/finish-story.md` is updated
   **When** an agent executes `/finish-story`
   **Then** the skill is thin — it calls the script, presents the JSON report to the user, and handles only the interactive doc-sync resolution (which requires agent judgment)
   **And** all mechanical steps (commit, PR creation, CI check, merge, cleanup, issue close, story update, epics update) are in the Python script

## Tasks / Subtasks

- [x] Task 1: Add reusable workflow helpers to `utils.py` (AC: #1, #2, #3, #4, #5)
  - [x] 1.1 `auto_commit_and_push(message, paths)`: stage given paths (default: `custom_components/`, `tests/`, `_bmad-output/`, `_qsprocess/`, `scripts/`), skip junk (`.DS_Store`, `__pycache__/`), commit if changes exist, push. Return `{"committed": bool, "pushed": bool, "files": list}`
  - [x] 1.2 `find_pr_for_branch(branch)`: query `gh pr list --head <branch> --json number,url,state`. Return `{"pr_number": int, "url": str}` or `None`
  - [x] 1.3 `check_ci_status(pr_number)`: query `gh pr checks <N> --json name,state,conclusion`. Return `{"checks": list, "all_passed": bool, "pending": list, "failed": list}`
  - [x] 1.4 `ensure_issue_link(pr_number, issue_number)`: check PR body for `Closes #N`/`Fixes #N`, append if missing via `gh pr edit`. Return `{"linked": bool, "added": bool}`
  - [x] 1.5 `close_issue_if_open(issue_number, comment)`: check state via `gh issue view`, close if open. Return `{"closed": bool, "was_open": bool}`
  - [x] 1.6 `update_story_status(story_file, status)`: find `Status:` line in markdown, update in place. Return `{"updated": bool}`
  - [x] 1.7 `suggest_release(changed_files)`: return `"release"` if `custom_components/` in changed files, `"no-release"` otherwise

- [x] Task 2: Refactor `create_pr.py` to use shared `find_pr_for_branch()` from utils (AC: #3)
  - [x] 2.1 Extract `get_changed_files()` from `create_pr.py` into `utils.py` (it's useful in multiple scripts)
  - [x] 2.2 `create_pr.py` uses `find_pr_for_branch()` to check before creating (prevent duplicates)

- [x] Task 3: Rewrite `finish_story.py` as zero-arg orchestrator (AC: #1, #2, #3, #4, #5, #6)
  - [x] 3.1 Auto-detect: `get_current_branch()` → `get_issue_from_branch()` → `find_story_file()` → `find_pr_for_branch()`
  - [x] 3.2 Phase 1 — Prepare: `auto_commit_and_push()`, then if no PR exists call `create_pr.py` (or inline using shared utils)
  - [x] 3.3 Phase 2 — Validate: run `doc_sync.py`, run `quality_gate.py`, `check_ci_status()`, `ensure_issue_link()`
  - [x] 3.4 Phase 3 — Merge: `merge_pr()` (existing), then `close_issue_if_open()`, `update_story_status()`, `update_epics()` (existing), `cleanup_worktree()` (existing), `update_main()` (existing)
  - [x] 3.5 Phase 4 — Report: structured JSON with every step's result, `suggest_release()`, recovery instructions on failure
  - [x] 3.6 All existing flags (`--pr`, `--story-key`, `--story-file`, `--skip-quality-gate`) become optional overrides

- [x] Task 4: Slim down the skill file (AC: #7)
  - [x] 4.1 Rewrite `_qsprocess/skills/finish-story.md`: call `finish_story.py` (no args), present report, handle only the interactive doc-sync resolution (requires agent judgment)
  - [x] 4.2 Remove all mechanical steps from the skill — commit, push, PR create, merge, cleanup, issue close, story update, epics update are all in the script now

- [x] Task 5: Add tests for all new utils functions (AC: all)
  - [x] 5.1 Test `auto_commit_and_push()`: with changes, without changes, with junk files to exclude
  - [x] 5.2 Test `find_pr_for_branch()`: PR exists, no PR, gh error
  - [x] 5.3 Test `check_ci_status()`: all pass, some fail, pending, no checks
  - [x] 5.4 Test `ensure_issue_link()`: link present, link missing and added, edit failure
  - [x] 5.5 Test `close_issue_if_open()`: already closed, still open, close failure
  - [x] 5.6 Test `update_story_status()`: various status lines, missing status line
  - [x] 5.7 Test `suggest_release()`: production changes, process-only changes, mixed

- [x] Task 6: Add tests for rewritten `finish_story.py` orchestration (AC: all)
  - [x] 6.1 Test full happy path with mocked utils (all phases succeed)
  - [x] 6.2 Test auto-detect from branch (no args)
  - [x] 6.3 Test auto-create PR when none exists
  - [x] 6.4 Test failure paths: quality gate fails, CI fails, merge fails — verify recovery instructions
  - [x] 6.5 Test optional override flags still work

## Dev Notes

### Architecture: reusable utils + thin orchestrator + minimal skill

New functions go in `utils.py` so any workflow script can reuse them. `finish_story.py` becomes an orchestrator that calls utils. The skill file becomes a thin wrapper.

```
utils.py (reusable)          finish_story.py (orchestrator)     skill (thin)
├─ auto_commit_and_push()    ├─ Phase 1: Prepare                ├─ Run doc-sync (agent judgment)
├─ find_pr_for_branch()      │   auto_commit_and_push()         ├─ Call finish_story.py
├─ check_ci_status()         │   find/create PR                 └─ Present report to user
├─ ensure_issue_link()       ├─ Phase 2: Validate
├─ close_issue_if_open()     │   doc_sync, quality_gate
├─ update_story_status()     │   check_ci, ensure_issue_link
├─ suggest_release()         ├─ Phase 3: Merge + post-merge
├─ get_changed_files()       │   merge, close issue, story status
└─ (existing: run_gh, etc.)  │   epics, cleanup, pull main
                             └─ Phase 4: Report JSON
```

### Files to modify

- `scripts/qs/utils.py` — **add 7 reusable functions** (auto_commit_and_push, find_pr_for_branch, check_ci_status, ensure_issue_link, close_issue_if_open, update_story_status, suggest_release) + move `get_changed_files()` from create_pr.py
- `scripts/qs/finish_story.py` — **rewrite as orchestrator**: zero-arg auto-detect, 4-phase flow using utils
- `scripts/qs/create_pr.py` — **refactor**: use shared `find_pr_for_branch()` and `get_changed_files()` from utils
- `_qsprocess/skills/finish-story.md` — **slim down**: only doc-sync + script call + report
- `tests/test_utils.py` or `tests/test_utils_workflow.py` — **new**: tests for all new utils functions
- `tests/test_finish_story.py` — **new**: tests for orchestrator flow

### Existing patterns to follow

- **`utils.py` convention**: functions use `run_gh(args, check=False)`, inspect `returncode`/`stderr`, return dicts. Never raise on gh/git failures.
- **`create_pr.py`**: already has `get_changed_files()` which should move to utils.
- **`finish_story.py`**: existing functions (`merge_pr`, `cleanup_worktree`, `update_epics`, `update_main`) stay, new phases wrap them.
- **Test pattern**: `monkeypatch` to mock `subprocess.run`. See `tests/test_doc_sync.py`.

### `gh` CLI commands reference

- Find PR for branch: `gh pr list --head <branch> --json number,url`
- Create PR: `gh pr create --title "..." --body "..." --base main`
- CI checks: `gh pr checks <N> --json name,state,conclusion`
- PR body: `gh pr view <N> --json body`
- Edit PR body: `gh pr edit <N> --body "..."`
- Issue state: `gh issue view <N> --json state`
- Close issue: `gh issue close <N> --comment "..."`
- Merge PR: `gh pr merge <N> --merge --delete-branch`

### Auto-commit safety

`auto_commit_and_push()` only stages files in known-safe paths: `custom_components/`, `tests/`, `_bmad-output/`, `_qsprocess/`, `scripts/`. Never `.DS_Store`, `venv/`, `config/`, `__pycache__/`, `.idea/`, `.vscode/`. Uses explicit `git add <paths>` not `git add -A`.

### Backwards compatibility

Existing flags (`--pr`, `--story-key`, `--story-file`, `--skip-quality-gate`) remain as optional overrides. Zero-arg is the new default.

### References

- [Source: scripts/qs/utils.py] — current utils (147 lines, 14 functions + constant)
- [Source: scripts/qs/finish_story.py] — current script (149 lines, 5 functions + main)
- [Source: scripts/qs/create_pr.py] — PR creation (has `get_changed_files()` to extract)
- [Source: scripts/qs/doc_sync.py] — pattern for structured JSON output
- [Source: _qsprocess/skills/finish-story.md] — current skill to slim down

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
None — clean implementation with no blocking issues.

### Completion Notes List
- Added 8 reusable workflow functions to `utils.py` (auto_commit_and_push, find_pr_for_branch, check_ci_status, ensure_issue_link, close_issue_if_open, update_story_status, suggest_release, get_changed_files)
- Refactored `create_pr.py` to use shared utils; added duplicate PR prevention via find_pr_for_branch
- Rewrote `finish_story.py` as 4-phase zero-arg orchestrator: prepare → validate → merge → report
- Slimmed `finish-story.md` skill: only doc-sync (agent judgment) + script call + report presentation
- 31 tests for utils functions, 13 tests for orchestrator — 44 total, all passing
- Quality gate: 100% coverage, ruff, mypy, translations all green
- Cleaned up 5 pre-existing unused snapshots

### File List
- `scripts/qs/utils.py` — added 8 workflow helper functions + constants
- `scripts/qs/finish_story.py` — rewritten as zero-arg 4-phase orchestrator
- `scripts/qs/create_pr.py` — refactored to use shared utils
- `_qsprocess/skills/finish-story.md` — slimmed to thin skill wrapper
- `tests/test_qs_utils.py` — new: 31 tests for workflow utils
- `tests/test_qs_finish_story.py` — new: 13 tests for orchestrator
- `tests/ha_tests/__snapshots__/test_sensor.ambr` — cleaned up 5 unused snapshots
