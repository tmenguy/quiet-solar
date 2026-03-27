# Story 1.13: Fix finish-story worktree self-destruct ordering bug

Status: in-progress

issue: 55
branch: "QS_55"

## Story

As TheDev,
I want `finish_story.py` post-merge phase to execute in the correct order — pulling main and updating docs before destroying the worktree,
So that post-merge housekeeping (story status, epics update) completes reliably instead of crashing when the worktree is deleted mid-execution.

## Acceptance Criteria

1. **Given** `/finish-story` runs from a worktree branch
   **When** the merge succeeds and post-merge starts
   **Then** `update_main()` runs BEFORE `cleanup_worktree()` so that the main worktree has the merged code before any file updates

2. **Given** the merge is complete and main has been pulled
   **When** story status and epics are updated
   **Then** these updates target files in the MAIN worktree (not the feature worktree), so changes survive cleanup

3. **Given** story status and epics have been updated in the main worktree
   **When** the updates are written
   **Then** the changes are auto-committed and pushed to main so they are not left as uncommitted local modifications

4. **Given** all post-merge updates are committed
   **When** `cleanup_worktree()` runs
   **Then** it is the LAST step in the phase, and its execution does not affect any prior results

5. **Given** any post-merge step fails (epics update, story status, commit)
   **When** the report is generated
   **Then** the failure is reported in the JSON output but does NOT block the overall success (merge already happened)

6. **Given** the existing tests for `finish_story.py` and `utils.py`
   **When** the fix is complete
   **Then** all existing tests pass, new tests cover the reordered flow, and 100% coverage is maintained

## Tasks / Subtasks

- [x] Task 1: Fix post-merge execution order in `phase_merge()` (AC: #1, #4)
  - [x] 1.1 Reorder: merge -> close issue -> update_main -> update story status -> update epics -> commit housekeeping -> cleanup worktree
  - [x] 1.2 Ensure `cleanup_worktree()` is unconditionally LAST

- [x] Task 2: Fix story status + epics to target main worktree paths (AC: #2)
  - [x] 2.1 After `update_main()`, resolve story file path relative to main worktree (not current worktree)
  - [x] 2.2 `update_epics()` already uses `get_main_worktree()` — verified it works correctly after reorder
  - [x] 2.3 `update_story_status()` path resolved via new `resolve_story_file_to_main()` function

- [x] Task 3: Auto-commit housekeeping changes to main (AC: #3)
  - [x] 3.1 After updating story status + epics in main worktree, stage and commit with descriptive message
  - [x] 3.2 Push the commit to origin/main
  - [x] 3.3 Return commit result in the phase output as `housekeeping_commit` key

- [x] Task 4: Update tests (AC: #5, #6)
  - [x] 4.1 Updated all `test_qs_finish_story.py` tests to reflect new ordering and new functions
  - [x] 4.2 Added test: `test_phase_merge_correct_execution_order` verifies exact 8-step sequence
  - [x] 4.3 Added test: `test_phase_merge_cleanup_runs_last_even_on_partial_failure`
  - [x] 4.4 Added tests: `test_commit_housekeeping_*` (4 tests covering changes/no-changes/failure/no-key)
  - [x] 4.5 All existing tests pass with reordered flow, 100% coverage maintained

## Dev Notes

### The bug

In `phase_merge()` (finish_story.py:238-242), the current execution order is:

```python
# CURRENT (BROKEN) ORDER:
merge_pr()                    # 1. Merge PR
close_issue_if_open()         # 2. Close issue
update_story_status()         # 3. Update story file — IN THE WORKTREE (lost on cleanup!)
update_epics()                # 4. Update epics — in main worktree (but uncommitted)
cleanup_worktree()            # 5. DESTROY the worktree — kills script's cwd
update_main()                 # 6. Pull main — CRASHES because cwd is gone
```

There are actually THREE bugs here:
1. **Ordering**: `cleanup_worktree()` runs before `update_main()`, destroying the cwd
2. **Wrong target**: `update_story_status()` writes to the worktree copy of the story file, which is destroyed on cleanup — the change is lost
3. **Uncommitted changes**: `update_epics()` and `update_story_status()` write to files in the main worktree but never commit them

### The fix

```python
# CORRECT ORDER:
merge_pr()                    # 1. Merge PR
close_issue_if_open()         # 2. Close issue
update_main()                 # 3. Pull main FIRST — get merged code
update_story_status()         # 4. Update story file — IN MAIN WORKTREE now
update_epics()                # 5. Update epics — in main worktree
auto_commit_housekeeping()    # 6. Commit + push story/epics changes to main
cleanup_worktree()            # 7. Cleanup LAST — safe, nothing depends on worktree anymore
```

### Files to modify

- `scripts/qs/finish_story.py` — reorder `phase_merge()`, fix story file path resolution, add housekeeping commit
- `tests/test_qs_finish_story.py` — update mocks and add tests for new ordering

### Key implementation details

**Resolving story file to main worktree:**
The `story_file` parameter comes from `find_story_file()` which uses `get_repo_root()`. When running from a worktree, this returns the worktree path. After `update_main()`, we need to resolve the equivalent path in the main worktree:

```python
main_dir = get_main_worktree()
repo_root = get_repo_root()
# Convert worktree-relative path to main-worktree-relative path
relative = Path(story_file).relative_to(repo_root)
main_story_file = str(main_dir / relative)
```

**Housekeeping commit:**
Use existing `run_git()` from utils. Stage only `_bmad-output/` in the main worktree. Commit with message like `chore: mark story 1.13 done, update epics`. Push to origin main. This is safe because we're on main after `update_main()`.

**Error handling:**
Post-merge steps should not block the overall success report. The merge already happened — if epics/story update or commit fails, report it but return success=True.

### Existing patterns to follow

- `utils.py` convention: functions use `run_git(args, cwd=str(main_dir))` for main worktree operations
- `cleanup_worktree()` already uses `get_main_worktree()` and `cwd=str(main_dir)` — this is correct
- `update_main()` already uses `get_main_worktree()` — just needs to run earlier
- Tests use `monkeypatch` to mock `subprocess.run`. See `tests/test_qs_finish_story.py`

### Anti-patterns to avoid

- Do NOT add try/except around each post-merge step — just reorder and let the dict results carry error info
- Do NOT change `cleanup_worktree()` internals — the function itself is fine, just called too early
- Do NOT change the phase_report structure — keep the same JSON contract
- Do NOT change the auto-detect logic (branch -> issue -> story file) — that's correct

### References

- [Source: scripts/qs/finish_story.py:201-251] — `phase_merge()` with the broken ordering
- [Source: scripts/qs/finish_story.py:59-78] — `cleanup_worktree()` destroys worktree via shell script
- [Source: scripts/qs/finish_story.py:81-86] — `update_main()` that crashes after cleanup
- [Source: scripts/qs/utils.py:345-366] — `update_story_status()` that writes to wrong location
- [Source: scripts/qs/utils.py:35-39] — `get_main_worktree()` used to resolve main worktree path
- [Source: scripts/worktree-cleanup.sh] — shell script that removes the worktree directory
- [Source: _bmad-output/implementation-artifacts/1-12-systematic-finish-story-enhancement.md] — previous story with full context

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
None — clean implementation.

### Completion Notes List
- Added `resolve_story_file_to_main()` to convert worktree paths to main worktree equivalents
- Added `commit_housekeeping()` to auto-commit and push story/epics changes to main
- Reordered `phase_merge()`: merge -> close -> update_main -> story status -> epics -> commit -> cleanup (was: merge -> close -> story -> epics -> cleanup -> main)
- Added `housekeeping_commit` key to phase_merge return dict
- 4 new tests for `resolve_story_file_to_main`, 4 for `commit_housekeeping`, 2 for execution order verification
- All 23 tests pass, 100% coverage, all quality gates green

### File List
- `scripts/qs/finish_story.py` — reordered phase_merge, added resolve_story_file_to_main and commit_housekeeping
- `tests/test_qs_finish_story.py` — updated existing tests, added 10 new tests
