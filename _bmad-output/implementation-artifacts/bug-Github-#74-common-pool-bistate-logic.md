# Refactor: Deduplicate pool and bistate update_current_metrics

issue: 74
branch: "QS_74"
Status: ready-for-dev

## Story

As a developer maintaining Quiet Solar,
I want the `update_current_metrics()` logic to exist only in the `QSBiStateDuration` parent class,
so that the pool inherits it instead of duplicating it, reducing maintenance burden and drift risk.

## Background

In QS_68 (`bug-Github-#68-carry-from-completed-constraint`), the pool's `update_current_metrics` pattern (falling back to `_last_completed_constraint` when no active constraints exist, filtered by current day window) was adopted by `QSBiStateDuration`. Both implementations are now **100% identical** — same algorithm, same variables, same attribute access. The pool override should be removed so the logic lives in one place.

## Acceptance Criteria

1. `QSPool` no longer overrides `update_current_metrics` — the method is deleted from `ha_model/pool.py`
2. `QSBiStateDuration.update_current_metrics` remains the single implementation in `ha_model/bistate_duration.py`
3. The unused import `DATETIME_MIN_UTC` is removed from `pool.py` (it was only needed by the deleted method)
4. All existing tests pass with no changes (behavior is identical)
5. Quality gate passes: `python scripts/qs/quality_gate.py`

## Tasks / Subtasks

- [ ] Task 1: Remove `update_current_metrics` override from `pool.py` (AC: #1)
  - [ ] Delete lines 55-90 in `ha_model/pool.py` (the entire `update_current_metrics` method)
  - [ ] Remove the now-unused `DATETIME_MIN_UTC` import from `pool.py` (AC: #3)
  - [ ] Remove `timedelta` import if no longer used in pool.py
- [ ] Task 2: Verify parent implementation is correct (AC: #2)
  - [ ] Confirm `QSBiStateDuration.update_current_metrics` at `bistate_duration.py:56-93` is unchanged
- [ ] Task 3: Run quality gate (AC: #4, #5)
  - [ ] Run `python scripts/qs/quality_gate.py` — all tests pass, ruff/mypy clean

## Dev Notes

### Class Hierarchy

```
QSBiStateDuration  (ha_model/bistate_duration.py)  ← update_current_metrics lives HERE
  └── QSOnOffDuration  (ha_model/on_off_duration.py)
      └── QSPool  (ha_model/pool.py)  ← REMOVE the duplicate override
```

### Duplicate Code Location

| File | Lines | Method |
|------|-------|--------|
| `ha_model/bistate_duration.py` | 56-93 | `update_current_metrics` (KEEP) |
| `ha_model/pool.py` | 55-90 | `update_current_metrics` (DELETE) |

The two implementations are byte-for-byte identical in logic. Only trivial comment differences exist. All attributes used (`_constraints`, `_last_completed_constraint`, `default_on_finish_time`, `qs_bistate_current_on_h`, `qs_bistate_current_duration_h`, `get_next_time_from_hours`) are inherited from `QSBiStateDuration` or its parents.

### Import Cleanup in pool.py

After removing the method, `DATETIME_MIN_UTC` is no longer used in `pool.py` — remove it from the import line. Also check if `timedelta` is still used (it is NOT used elsewhere in pool.py, so remove it too).

### Test Coverage

Tests in `tests/test_ha_pool.py` (lines 191-390) cover all `update_current_metrics` scenarios through the pool. They will continue to pass because the inherited parent method has identical behavior. No test changes should be needed.

### Architecture Constraints

- Two-layer boundary: both files are in `ha_model/` — no boundary issues
- No async code involved — `update_current_metrics` is synchronous
- No config keys or translations affected

### References

- [Source: ha_model/bistate_duration.py#update_current_metrics] — parent implementation (lines 56-93)
- [Source: ha_model/pool.py#update_current_metrics] — duplicate to remove (lines 55-90)
- [Source: bug-Github-#68-carry-from-completed-constraint.md] — QS_68 story that aligned the logic

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
