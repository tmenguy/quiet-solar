# Story 1.14: Robust Story Naming and Retrieval

Status: ready-for-dev
issue: 60
branch: "QS_60"

## Story

As TheDev,
I want story files to include `Github-#N` in their filename so that `find_story_file(issue_number)` is the single, reliable way to discover story artifacts everywhere in the workflow,
so that all scripts and skills use the GitHub issue number as the only lookup key, eliminating fragile heuristics, story-key guessing, and full-path passing.

## Problem Analysis

### Current State

`find_story_file()` in `scripts/qs/utils.py` supports two modes:
1. **By story key** (e.g., "3.2"): normalizes to "3-2", globs for `3-2-*.md` then `*3-2*.md`
2. **No key**: returns most recently modified `.md` file (unreliable in parallel development)

Skills and scripts pass around `--story-file PATH`, `--story-key X.Y`, and `--issue N` inconsistently. The issue number (from `QS_N` branch) is the primary key but cannot be used to find story files.

### Root Causes

1. **No issue-number in filenames**: story files don't encode the GitHub issue number
2. **`find_story_file()` takes story_key, not issue number**: the primary workflow key is unusable
3. **Scripts pass full paths around**: `--story-file PATH` is fragile across worktrees
4. **"Most recent" fallback is dangerous**: in parallel development, returns the wrong file

## Acceptance Criteria

1. **Given** `/create-story` creates a story via bmad-create-story
   **When** the file is written
   **Then** it is renamed to include `Github-#N` in the filename (e.g., `1-14-Github-#60-slug.md`)
   **And** this happens in the `/create-story` skill post-processing, NOT by modifying bmad-create-story

2. **Given** a file named `*Github-#60*.md` exists in implementation-artifacts
   **When** `find_story_file(60)` is called
   **Then** it returns that file by filename glob (no frontmatter parsing needed)

3. **Given** a legacy file without `Github-#N` in the filename but with `issue: 48` in frontmatter
   **When** `find_story_file(48)` is called
   **Then** it falls back to scanning frontmatter and returns that file

4. **Given** no matching file exists
   **When** `find_story_file(N)` is called
   **Then** it returns None (no "most recent" fallback)

5. **Given** all scripts and skills in the workflow
   **When** they need a story file
   **Then** they call `find_story_file(issue_number)` — no `--story-file` or `--story-key` passing

6. **Given** the next-step launch commands
   **When** transitioning between skills (implement, review, finish)
   **Then** only the GitHub issue number is passed, not full paths

## Tasks / Subtasks

- [ ] Task 1: Rewrite `find_story_file()` to only take issue number (AC: 2, 3, 4)
  - [ ] 1.1: Change signature to `find_story_file(issue_number: int) -> Path | None`
  - [ ] 1.2: Primary lookup: glob for `*Github-#{issue_number}*` in artifacts dir
  - [ ] 1.3: Fallback: scan frontmatter of all `.md` files for `issue: N` (legacy files)
  - [ ] 1.4: No "most recent" fallback — return None if no match

- [ ] Task 2: Update all scripts to pass issue number only (AC: 5, 6)
  - [ ] 2.1: `finish_story.py` — remove `--story-file` and `--story-key` args, use `find_story_file(issue)`
  - [ ] 2.2: `setup_worktree.py` — remove `--story-file` and `--story-key` args, use `find_story_file(issue)`
  - [ ] 2.3: `next_step.py` — remove `--story-file` and `--story-key` args, only pass `--issue`
  - [ ] 2.4: `create_issue.py` — remove `find_story_file()` usage (story doesn't exist yet at issue creation time)
  - [ ] 2.5: `doc_sync.py` — add `--issue` arg as alternative to positional story_file path

- [ ] Task 3: Update all skills to pass issue number only (AC: 1, 5, 6)
  - [ ] 3.1: `create-story.md` — add rename step after bmad-create-story to include `Github-#N`
  - [ ] 3.2: `setup-story.md` — simplify to only pass issue number
  - [ ] 3.3: `implement-story.md` — remove `--story-file`, use `find_story_file(issue)` internally
  - [ ] 3.4: `finish-story.md` — simplify doc-sync call to use issue number
  - [ ] 3.5: `review-story.md` — simplify next_step call to only pass `--issue`

## Dev Notes

### Naming Convention

New story files: `{prefix}-Github-#{issue}-{slug}.md`
- Epic stories: `1-14-Github-#60-robust-story-naming-retrieval.md`
- Bug fixes: `bug-Github-#48-charger-constraint-oscillation.md`
- Fixes: `fix-Github-#58-remove-forecast-sensors.md`

The story_key extraction regex `^(\d+)-(\d+)` in `finish_story.py` still works since the epic-story prefix comes first.

### Files to Modify

| File | Change |
|------|--------|
| `scripts/qs/utils.py` | Rewrite `find_story_file()` to take only issue number |
| `scripts/qs/finish_story.py` | Drop `--story-file`/`--story-key`, use issue number |
| `scripts/qs/setup_worktree.py` | Drop `--story-file`/`--story-key`, use issue number |
| `scripts/qs/next_step.py` | Drop `--story-file`/`--story-key`, only `--issue` |
| `scripts/qs/create_issue.py` | Remove `find_story_file()` usage |
| `scripts/qs/doc_sync.py` | Add `--issue` alternative input |
| `_qsprocess/skills/create-story.md` | Add rename step after bmad-create-story |
| `_qsprocess/skills/setup-story.md` | Simplify to issue-only |
| `_qsprocess/skills/implement-story.md` | Remove `--story-file`, use issue lookup |
| `_qsprocess/skills/finish-story.md` | Simplify to issue-only |
| `_qsprocess/skills/review-story.md` | Simplify next_step call |

### What NOT to Change

- Do NOT modify bmad-create-story workflow/template files
- Do NOT rename existing legacy story files (they have frontmatter `issue:` as fallback)
- Do NOT add a YAML parser dependency — regex is sufficient
- Do NOT change `update_story_status()` — it works fine as-is

### Risk Assessment

**LOW risk**: Only dev workflow scripts and skill definitions. No production code.

### References

- [Source: scripts/qs/utils.py#find_story_file] — current implementation (lines 97-116)
- [Source: scripts/qs/finish_story.py] — primary consumer
- [Source: scripts/qs/next_step.py] — skill transition commands
- [Source: _qsprocess/skills/create-story.md] — where rename step goes
- [Source: _qsprocess/skills/implement-story.md] — currently passes --story-file

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
