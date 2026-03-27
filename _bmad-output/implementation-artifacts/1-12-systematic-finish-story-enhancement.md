# Story 1.12: Systematic Finish-Story Workflow Enhancement

Status: ready-for-dev

issue: 51
branch: "QS_51"

## Story

As TheDev,
I want the `/finish-story` workflow to be more systematic and complete — verifying CI status, PR approval, issue linkage, and story artifact lifecycle before and after merge, with robust error recovery and a comprehensive delivery report,
So that finishing a story is a single reliable command that leaves zero loose ends (no unclosed issues, no stale story statuses, no missed CI failures).

## Acceptance Criteria

1. **Given** TheDev runs `/finish-story --pr N`
   **When** the pre-merge validation phase starts
   **Then** the workflow verifies that all GitHub Actions CI checks on the PR have passed (not just local quality gate)
   **And** if any remote checks have failed or are still pending, the workflow reports the status and blocks merge until resolved

2. **Given** the PR is ready for merge
   **When** the workflow checks PR approval status
   **Then** it verifies the PR has at least one approval (or is self-authored with no required reviewers)
   **And** if no approval exists, it warns the user and asks whether to proceed

3. **Given** the PR exists
   **When** the workflow validates PR metadata
   **Then** it verifies the PR body references the linked issue (e.g., "Closes #N" or "Fixes #N")
   **And** if the issue link is missing, it offers to add it before merge

4. **Given** the merge completes successfully
   **When** the post-merge phase runs
   **Then** the workflow verifies the linked GitHub issue is closed (auto-closed by the PR or manually)
   **And** if the issue is still open, it closes it with a reference to the merged PR

5. **Given** the merge completes successfully
   **When** the story artifact update phase runs
   **Then** the story file's `Status:` field is updated from its current value to `done`
   **And** the change is committed on main

6. **Given** all finish steps complete
   **When** the delivery report is shown
   **Then** it includes: PR merge status, CI check summary, issue closure status, epics update status, story artifact status, worktree cleanup status, and a clear next-step suggestion (`/release` if version bump warranted, or "no release needed")

7. **Given** any step in the workflow fails (merge conflict, CI failure, network error)
   **When** the failure is detected
   **Then** the workflow reports exactly what failed and what state was left in (what was done, what wasn't)
   **And** provides specific recovery instructions rather than a generic error

## Tasks / Subtasks

- [ ] Task 1: Add CI check verification to `finish_story.py` (AC: #1)
  - [ ] 1.1 Add a `check_ci_status(pr_number)` function that uses `gh pr checks` to verify all checks passed
  - [ ] 1.2 Return structured result: list of checks with name, status, conclusion
  - [ ] 1.3 If any check failed or is pending, the script reports but does NOT block (the skill decides)
  - [ ] 1.4 Add `--skip-ci-check` flag for edge cases

- [ ] Task 2: Add PR approval and metadata validation to `finish_story.py` (AC: #2, #3)
  - [ ] 2.1 Add `check_pr_approval(pr_number)` function: query `gh pr view --json reviews,reviewRequests`
  - [ ] 2.2 Add `check_issue_link(pr_number)` function: verify PR body contains `Closes #N` or `Fixes #N` pattern
  - [ ] 2.3 Add `add_issue_link(pr_number, issue_number)` function: append `Closes #N` to PR body if missing

- [ ] Task 3: Add post-merge issue closure verification to `finish_story.py` (AC: #4)
  - [ ] 3.1 Add `verify_issue_closed(issue_number)` function: check issue state via `gh issue view`
  - [ ] 3.2 Add `close_issue(issue_number, pr_number)` function: close with comment linking merged PR
  - [ ] 3.3 Integrate into main flow after merge step

- [ ] Task 4: Add story artifact status update to `finish_story.py` (AC: #5)
  - [ ] 4.1 Add `update_story_status(story_file, status="done")` function: find `Status:` line and update
  - [ ] 4.2 Accept story file path via `--story-file` argument (reuse `find_story_file()` from utils)
  - [ ] 4.3 Stage the change for commit on main

- [ ] Task 5: Enhance delivery report in `finish_story.py` (AC: #6)
  - [ ] 5.1 Restructure output JSON to include all step results in a consistent format
  - [ ] 5.2 Add `suggest_release(changed_files)` logic: check if `custom_components/` files changed (suggests version bump) vs. only `_qsprocess/`/`scripts/` (no release needed)
  - [ ] 5.3 Include summary counts (files changed, tests added, etc.) from PR stats

- [ ] Task 6: Improve error handling and recovery guidance (AC: #7)
  - [ ] 6.1 Each step in `finish_story.py` captures its own success/failure and continues to report
  - [ ] 6.2 On failure, include `recovery` field in JSON with specific instructions
  - [ ] 6.3 Script exits with appropriate code but always outputs full JSON report

- [ ] Task 7: Update `/finish-story` skill to use new script capabilities (AC: all)
  - [ ] 7.1 Update `_qsprocess/skills/finish-story.md` to add pre-merge validation section (CI checks, approval, issue link)
  - [ ] 7.2 Update post-merge section to include issue closure verification and story artifact update
  - [ ] 7.3 Update report section to show the enhanced delivery report
  - [ ] 7.4 Add recovery guidance instructions for when steps fail

- [ ] Task 8: Add tests for new `finish_story.py` functions (AC: all)
  - [ ] 8.1 Test `check_ci_status()` with passing, failing, and pending checks
  - [ ] 8.2 Test `check_pr_approval()` with approved, no-review, and changes-requested states
  - [ ] 8.3 Test `check_issue_link()` with present and missing issue references
  - [ ] 8.4 Test `verify_issue_closed()` and `close_issue()` with open and closed states
  - [ ] 8.5 Test `update_story_status()` with various story file formats
  - [ ] 8.6 Test `suggest_release()` with production and process-only changes
  - [ ] 8.7 Test error recovery JSON output on failures

## Dev Notes

### This is a hybrid process/code story

Modifies `scripts/qs/finish_story.py` (Python script, requires tests) and `_qsprocess/skills/finish-story.md` (skill definition, no tests). The script enhancements are the primary deliverable; the skill file updates wire them into the agent workflow.

### Files to modify

- `scripts/qs/finish_story.py` — primary: add CI check, approval check, issue link, issue close, story status update, release suggestion, error recovery
- `_qsprocess/skills/finish-story.md` — update skill to use new script capabilities and add pre/post merge validation steps
- `tests/test_finish_story.py` — new: tests for all new functions (mock `gh` CLI calls)

### Existing patterns to follow

- **`finish_story.py` current structure**: functions like `merge_pr()`, `cleanup_worktree()`, `update_epics()` that return dicts. New functions must follow same pattern: return a dict with success/failure + details.
- **`utils.py` helpers**: use `run_gh()`, `run_git()`, `output_json()`, `find_story_file()`, `get_main_worktree()` — do not reinvent these.
- **Error pattern**: `run_gh(args, check=False)` then inspect `returncode` and `stderr`. Never raise on gh failures — return structured error.
- **Test pattern**: see `tests/test_quality_gate.py` or `tests/test_doc_sync.py` for how `scripts/qs/` scripts are tested. Use `monkeypatch` to mock `subprocess.run` calls.
- **Skill file style**: see story 1.11's finish-story additions — concise, numbered steps, agent instructions not code.

### Script is NOT in `custom_components/` — coverage rules differ

`scripts/qs/` scripts are utility scripts, not production HA code. They are still covered by the project's 100% coverage requirement via pytest, but they test differently (mock `gh`/`git` subprocess calls, not HA entities).

### `gh` CLI commands reference

- CI checks: `gh pr checks <N> --json name,state,conclusion`
- PR approval: `gh pr view <N> --json reviews,reviewRequests`
- PR body: `gh pr view <N> --json body`
- Edit PR body: `gh pr edit <N> --body "..."`
- Issue state: `gh issue view <N> --json state`
- Close issue: `gh issue close <N> --comment "..."`
- PR merge stats: `gh pr view <N> --json additions,deletions,changedFiles`

### Design decisions

1. **Script does validation, skill decides action**: `finish_story.py` reports status; the agent (via the skill) decides whether to block, warn, or proceed. This keeps the script reusable and the policy in the skill.
2. **Always output full JSON**: even on failure, output the complete report so the agent can show the user what happened.
3. **Backwards compatible**: new flags (`--skip-ci-check`, `--story-file`) are optional. Existing invocations continue to work.
4. **Issue closure is defensive**: check first, close only if still open. Handles both auto-close via PR and manual close gracefully.

### Previous story patterns (from 1.11)

Story 1.11 added the doc-sync gate and `doc_sync.py` script. It followed the same pattern: Python script with testable functions + skill file update. The script returns structured JSON; the skill interprets it. Use this same separation.

### References

- [Source: scripts/qs/finish_story.py] — current script (149 lines, 5 functions + main)
- [Source: _qsprocess/skills/finish-story.md] — current skill (5 steps + doc-sync gate)
- [Source: scripts/qs/utils.py] — utility functions (find_story_file, run_gh, output_json)
- [Source: scripts/qs/doc_sync.py] — pattern reference for new script functions
- [Source: _bmad-output/implementation-artifacts/1-11-living-documentation-sync.md] — previous process/code story pattern

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
