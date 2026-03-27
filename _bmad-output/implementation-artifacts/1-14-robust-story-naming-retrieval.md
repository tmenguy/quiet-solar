# Story 1.14: Robust Story Naming and Retrieval

Status: ready-for-dev
issue: 60
branch: "QS_60"

## Story

As TheDev,
I want story files to be discoverable by issue number (the primary key in the workflow) with consistent naming and frontmatter,
so that scripts like `finish_story.py`, `setup_worktree.py`, and `create_issue.py` reliably find the correct story file without depending on fragile heuristics or "most recent file" fallbacks.

## Problem Analysis

### Current State

`find_story_file()` in `scripts/qs/utils.py` supports two modes:
1. **By story key** (e.g., "3.2"): normalizes to "3-2", globs for `3-2-*.md` then `*3-2*.md`
2. **No key**: returns most recently modified `.md` file (unreliable in parallel development)

### What Breaks

| Scenario | Expected | Actual |
|----------|----------|--------|
| Branch `QS_48`, no story-key arg | Find `bug-48-*.md` | Returns most recent file (wrong) |
| Branch `QS_58`, no story-key arg | Find `fix-58-*.md` | Returns most recent file (wrong) |
| finish_story with parallel worktrees | Find correct story | Finds whatever was modified last |
| Story key for non-epic stories | N/A (no key exists) | Cannot look up at all |

### Root Causes

1. **No issue-number lookup**: the primary workflow key (issue number from `QS_N` branch) cannot be used to find story files
2. **Inconsistent filenames**: 3 of 24 files use non-standard prefixes (`bug-`, `dev-`, `fix-`) instead of `N-M-` pattern
3. **Frontmatter not used for discovery**: all files have `issue:` in frontmatter but `find_story_file()` never reads it
4. **"Most recent" fallback is dangerous**: in parallel development, this returns the wrong file

## Acceptance Criteria

1. **Given** a branch `QS_N` is checked out
   **When** `find_story_file()` is called with no arguments
   **Then** it finds the story file whose frontmatter contains `issue: N`
   **And** it does NOT fall back to "most recent file"

2. **Given** a story file exists with `issue: 48` in frontmatter
   **When** `find_story_file(issue_number=48)` is called
   **Then** it returns that file regardless of its filename prefix (`bug-`, `fix-`, `dev-`, or `N-M-`)

3. **Given** a story key "3.2" is provided
   **When** `find_story_file(story_key="3.2")` is called
   **Then** it still works as before (filename pattern matching)

4. **Given** no matching file is found by issue or story key
   **When** `find_story_file()` returns None
   **Then** callers get None (no silent wrong-file fallback)

5. **Given** existing callers (`finish_story.py`, `setup_worktree.py`, `create_issue.py`)
   **When** they call `find_story_file()`
   **Then** they pass issue_number when available, falling back to story_key

6. **Given** all existing story files in `_bmad-output/implementation-artifacts/`
   **When** the migration is applied
   **Then** all 24 files have consistent frontmatter: `issue:` and `branch:` fields present

## Tasks / Subtasks

- [ ] Task 1: Extend `find_story_file()` with issue-number lookup (AC: 1, 2, 3, 4)
  - [ ] 1.1: Add `issue_number: int | None` parameter to `find_story_file()`
  - [ ] 1.2: When `issue_number` given, scan all `.md` files in artifacts dir, parse frontmatter for `issue: N` match
  - [ ] 1.3: Keep existing story_key pattern matching as secondary lookup
  - [ ] 1.4: Remove "most recent file" fallback — return None if no match
  - [ ] 1.5: Write tests for all lookup modes: by issue, by story_key, no match, edge cases

- [ ] Task 2: Update callers to use issue-number lookup (AC: 5)
  - [ ] 2.1: `finish_story.py` — pass `issue_number` from branch detection to `find_story_file()`
  - [ ] 2.2: `setup_worktree.py` — pass `issue_number` argument to `find_story_file()`
  - [ ] 2.3: `create_issue.py` — no change needed (uses story_key, not issue lookup)
  - [ ] 2.4: Write/update tests for each modified caller

- [ ] Task 3: Normalize existing story file frontmatter (AC: 6)
  - [ ] 3.1: Audit all 24 files — verify `issue:` and `branch:` fields present and consistent
  - [ ] 3.2: Fix any files missing frontmatter fields (add `issue:` and `branch:` where missing)
  - [ ] 3.3: Standardize `fix-58` YAML frontmatter block to match inline format used everywhere else

## Dev Notes

### Files to Modify

| File | Change |
|------|--------|
| `scripts/qs/utils.py` | Extend `find_story_file()` signature and logic |
| `scripts/qs/finish_story.py` | Pass issue_number to `find_story_file()` |
| `scripts/qs/setup_worktree.py` | Pass issue_number to `find_story_file()` |
| `tests/` | New/updated tests for `find_story_file()` and callers |
| `_bmad-output/implementation-artifacts/*.md` | Frontmatter normalization |

### Implementation Approach

**`find_story_file()` new signature:**
```python
def find_story_file(
    story_key: str | None = None,
    *,
    issue_number: int | None = None,
) -> Path | None:
```

**Lookup priority:**
1. If `issue_number` given → scan frontmatter of all `.md` files for `issue: N`
2. If `story_key` given → existing pattern matching (normalize "3.2" → "3-2", glob)
3. If neither → return None (no more "most recent" fallback)

**Frontmatter parsing**: simple regex on first 10 lines — look for `issue: N` or `issue: "N"`. No need for a YAML parser; all files use simple `key: value` format.

### Existing Code Patterns to Follow

- Functions in `utils.py` return structured data or simple values, never raise
- Scripts use `output_json()` for all output
- Tests must maintain 100% coverage
- Use `from utils import find_story_file` import pattern (relative, scripts run from `scripts/qs/`)

### What NOT to Change

- Do NOT rename existing story files — changing filenames would break git blame and cross-references
- Do NOT add a YAML parser dependency — regex is sufficient for frontmatter
- Do NOT change the `create-story` skill naming convention — new stories can use whatever prefix makes sense
- Do NOT change `update_story_status()` — it works fine with both inline and YAML formats

### Risk Assessment

**LOW risk**: This only touches dev workflow scripts (`scripts/qs/`), not production code. No solver, no HA integration, no device control affected.

### Project Structure Notes

- Scripts live in `scripts/qs/` — pure Python, no HA dependencies
- All scripts import from `utils.py` using relative imports
- Tests for workflow scripts should go alongside other tests in `tests/`
- Story artifacts in `_bmad-output/implementation-artifacts/` are documentation, not code

### References

- [Source: scripts/qs/utils.py#find_story_file] — current implementation (lines 97-116)
- [Source: scripts/qs/finish_story.py#run_finish_story] — primary caller (line 368)
- [Source: scripts/qs/setup_worktree.py#main] — secondary caller (line 71)
- [Source: scripts/qs/create_issue.py#main] — tertiary caller (line 36)
- [Source: _qsprocess/skills/implement-story.md] — documents expected behavior: "find it via implementation-artifacts matching the issue"

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
