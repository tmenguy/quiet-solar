# Cache Quality Gate Results to Avoid Redundant Runs

Status: ready-for-dev
issue: 76
branch: "QS_76"

## Story

As a developer using the QS workflow,
I want quality gate results to be cached based on git state (branch + commit),
so that redundant gate runs are skipped when no code has changed between skills.

## Acceptance Criteria

1. **Cache on pass**: When `quality_gate.py` runs with `--cache` and all gates pass, it writes a JSON cache file (`.quality_gate_cache`) containing the branch name, commit hash, and gate results.
2. **Cache hit**: On next run with `--cache`, if the branch and commit match and the working tree is clean, return cached results without running any gates. Output includes a `"cached": true` indicator.
3. **Cache invalidation**: If the commit changed, the branch changed, or the working tree is dirty, the cache is invalidated and all gates run fresh.
4. **Fix bypasses cache**: `--fix` always runs fresh regardless of cache state (fixes modify files, so caching is meaningless).
5. **No-cache override**: `--no-cache` forces a fresh run even when `--cache` is also present.
6. **Git-ignored**: `.quality_gate_cache` is added to `.gitignore`.
7. **Backward compatible**: When `--cache` is not used, behavior is identical to current implementation. No existing scripts or workflows break.
8. **Skill integration**: The workflow skills (`implement-story`, `review-story`, `finish-story`) use `--cache` on their quality gate calls where appropriate.

## Tasks / Subtasks

- [x] Task 1: Add git-based change detection to `quality_gate.py` (AC: 1, 2, 3)
  - [x] 1.1: Implement `_get_git_state()` returning `(branch, commit, is_clean)` via `git rev-parse --abbrev-ref HEAD`, `git rev-parse HEAD`, `git status --porcelain -uno`
  - [x] 1.2: Implement `_read_cache()` and `_write_cache()` for the `.quality_gate_cache` JSON file storing `{"branch", "commit", "all_passed", "results", "timestamp"}`
  - [x] 1.3: Implement `_is_cache_valid()` that checks branch match + commit match + clean working tree
- [x] Task 2: Add `--cache` and `--no-cache` CLI flags (AC: 1, 4, 5, 7)
  - [x] 2.1: Add argparse flags: `--cache` (opt-in), `--no-cache` (force fresh)
  - [x] 2.2: In `main()`, if `--cache` and not `--fix` and not `--no-cache`, check cache before running gates
  - [x] 2.3: On cache hit, output cached results (both human-readable and JSON modes) with `"cached": true`
  - [x] 2.4: On cache miss or fresh run, run all gates as before; if all pass, write cache
- [x] Task 3: Add `.quality_gate_cache` to `.gitignore` (AC: 6)
- [x] Task 4: Update `finish_story.py` to pass `--cache` (AC: 8)
  - [x] 4.1: In `run_quality_gate()`, add `--cache` to the subprocess command
- [x] Task 5: Update workflow skill docs (AC: 8)
  - [x] 5.1: Update `implement-story.md` final gate command to use `--cache`
  - [x] 5.2: Update `review-story.md` post-fix gate command to use `--cache`
  - [x] 5.3: Update `finish-story.md` to document the `--cache` behavior
- [x] Task 6: Tests (AC: 1-7)
  - [x] 6.1: Test `_get_git_state()` returns branch, commit, and clean status
  - [x] 6.2: Test cache write/read round-trip
  - [x] 6.3: Test cache hit skips gate execution
  - [x] 6.4: Test cache miss when git state changes (branch, commit, or dirty tree)
  - [x] 6.5: Test `--fix` bypasses cache
  - [x] 6.6: Test `--no-cache` bypasses cache
  - [x] 6.7: Test default (no `--cache` flag) never uses cache

## Dev Notes

### Architecture

`quality_gate.py` is a standalone script in `scripts/qs/`. It is NOT part of the HA custom component — it's dev tooling. No domain/HA boundary concerns apply. Tests for this script live in `tests/` alongside component tests.

### Key design decisions

- **Opt-in caching (`--cache`)**: Default behavior unchanged. Only callers that explicitly opt in get caching. This prevents surprises during manual development.
- **Git state, not content hash**: `branch/commit` is the cache key. Git already tracks file changes — no need to re-hash files. Dirty working tree = always cache miss; the calling skill should commit before running the gate.
- **Single cache file**: `.quality_gate_cache` is a small JSON file at repo root. Contains: `{"branch": "<name>", "commit": "<sha>", "all_passed": true, "results": [...], "timestamp": "<iso>"}`. The timestamp is informational only (not used for validation).

### Files to modify

| File | Change |
|------|--------|
| `scripts/qs/quality_gate.py` | Add hashing, caching, `--cache`/`--no-cache` flags |
| `scripts/qs/finish_story.py` | Pass `--cache` to quality_gate.py call |
| `.gitignore` | Add `.quality_gate_cache` |
| `_qsprocess/skills/implement-story.md` | Add `--cache` to final gate command |
| `_qsprocess/skills/review-story.md` | Add `--cache` to post-fix gate command |
| `_qsprocess/skills/finish-story.md` | Document `--cache` behavior |

### Existing patterns to follow

- `quality_gate.py` uses `subprocess.run()` with `capture_output=True` — keep this pattern
- `REPO_ROOT`, `SRC_DIR`, `TESTS_DIR` are already resolved via `Path` — reuse these
- `_run()` helper is internal — caching logic goes in `main()` and new helper functions
- JSON output mode (`--json`) should include `"cached": true/false` in the top-level object

### Testing approach

- Tests should mock file system or use `tmp_path` with sample files to verify hashing
- Use `monkeypatch` to intercept subprocess calls and verify gates are/aren't called
- The `_compute_content_hash()` function should be independently testable

### Project Structure Notes

- Script lives at `scripts/qs/quality_gate.py` — standard location for QS tooling
- Cache file at repo root (`.quality_gate_cache`) follows `.coverage`, `.mypy_cache` pattern
- No new dependencies needed — `hashlib` and `json` are stdlib

### References

- [Source: scripts/qs/quality_gate.py] — current implementation, all 5 gates
- [Source: scripts/qs/finish_story.py:41-47] — `run_quality_gate()` calls quality_gate.py
- [Source: _qsprocess/skills/implement-story.md:37,57] — gate calls in implement-story
- [Source: _qsprocess/skills/review-story.md:52] — gate call in review-story
- [Source: _qsprocess/skills/finish-story.md:38-39] — finish-story delegates to script

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
