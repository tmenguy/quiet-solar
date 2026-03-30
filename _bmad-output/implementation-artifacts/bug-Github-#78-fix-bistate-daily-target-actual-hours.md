# Bug Fix: bistate/pool target and actual hours display sums across days instead of current day only

Status: ready-for-dev
issue: 78
branch: "QS_78"

## Story

As a user viewing the bistate/pool card,
I want the Target Hours to show only today's total scheduled duration and the Actual Hours to show only today's actual runtime,
so that I can accurately track my device's daily progress without confusion from future days' schedules.

## Bug Description

`update_current_metrics()` in `ha_model/bistate_duration.py:56-94` sums `target_value` and `current_value` from ALL entries in `self._constraints` without day filtering. When calendar events span multiple days, the metrics include future days' constraints.

**Reproduction:**
- Climate device in Calendar: End+Duration mode
- Calendar: Today 6:00-7:00, 16:00-17:00; Tomorrow 6:00-7:00, 16:00-17:00
- At 11:30, Target Hours shows 3H (today's 16:00-17:00 + tomorrow's two events)
- Expected: Target Hours = 2H (today's 6:00-7:00 + 16:00-17:00 only)

**Root cause:** The constraint list (`self._constraints`) naturally includes future events loaded by `_build_mode_constraint_items()` via `get_next_scheduled_events()`. The metrics method blindly sums all of them.

## Acceptance Criteria

1. **AC1: Calendar modes show today-only target**
   - Given a bistate device in `bistate_mode_auto` or `bistate_mode_exact_calendar` mode
   - When the calendar has events spanning multiple days
   - Then Target Hours = sum of constraint durations whose `end_of_constraint` falls within today (local midnight to local midnight)
   - And constraints ending after local midnight are excluded from the target sum

2. **AC2: Calendar modes show today-only actual**
   - Given a bistate device in calendar mode with constraints that have accumulated `current_value`
   - Then Actual Hours = sum of `current_value` for today's constraints only (same day filter as AC1)
   - And if `_last_completed_constraint` exists for today, its `current_value` is included

3. **AC3: Default/pool modes show today-only metrics**
   - Given a bistate device in `bistate_mode_default` or a pool in `bistate_mode_auto`/`pool_winter_mode`
   - Then Target Hours = constraint target for the current day cycle only
   - And Actual Hours = `_last_completed_constraint.current_value` if it belongs to today's cycle, otherwise the active constraint's `current_value`

4. **AC4: Day boundary uses local midnight**
   - The "today" window is defined as local time 00:00:00 to 23:59:59 (converted to UTC for comparison)
   - This correctly handles timezone offsets and DST transitions

5. **AC5: Sync warning for calendar modes**
   - When in calendar modes and `_last_completed_constraint` exists for today
   - If its `current_value` significantly diverges from the inferred past schedule runtime
   - Then emit a `_LOGGER.warning()` with the discrepancy details

6. **AC6: No regression on existing behavior**
   - The existing `_last_completed_constraint` day-window fallback (lines 70-92) continues to work correctly when no active constraints exist
   - All existing tests in `test_bug_74_exact_calendar_metrics.py`, `test_coverage_bistate_duration.py`, and `test_ha_pool.py` continue to pass

## Tasks / Subtasks

- [ ] Task 1: Add day-boundary helper method (AC: #4)
  - [ ] Add `_get_today_boundaries(time: datetime) -> tuple[datetime, datetime]` to `QSBiStateDuration` that returns (start_of_today_utc, end_of_today_utc) using local midnight
  - [ ] Reuse existing `get_proper_local_adapted_tomorrow()` pattern from `device.py:416-423` for the local-to-UTC conversion

- [ ] Task 2: Refactor `update_current_metrics()` to filter by day (AC: #1, #2, #3)
  - [ ] When `self._constraints` is non-empty, filter constraints: only include those whose `end_of_constraint` falls within today's boundaries
  - [ ] Sum `target_value` and `current_value` only from today's filtered constraints
  - [ ] For `_last_completed_constraint` fallback: check if it belongs to today using the same boundary logic

- [ ] Task 3: Handle `_last_completed_constraint` for today's actual (AC: #2, #3)
  - [ ] When active constraints exist for today AND `_last_completed_constraint` also belongs to today
  - [ ] Add the completed constraint's `current_value` to the actual sum (it represents already-finished work from earlier today)
  - [ ] Avoid double-counting: only add if the completed constraint is not already in `self._constraints`

- [ ] Task 4: Add sync warning for calendar modes (AC: #5)
  - [ ] After computing actual hours in calendar modes, compare with `_last_completed_constraint` if available
  - [ ] Emit `_LOGGER.warning("Metrics sync discrepancy for %s: ...")` if values diverge significantly

- [ ] Task 5: Write tests (AC: #1-#6)
  - [ ] Test calendar mode with multi-day events: only today's constraints counted
  - [ ] Test default mode with single constraint per day cycle
  - [ ] Test pool auto/winter mode day filtering
  - [ ] Test day boundary at local midnight with timezone offset
  - [ ] Test `_last_completed_constraint` inclusion for today
  - [ ] Test `_last_completed_constraint` exclusion when from yesterday
  - [ ] Verify all existing bug-74 and coverage tests still pass

## Dev Notes

### Key Code Locations

| File | Method/Area | Lines | What to change |
|------|-------------|-------|----------------|
| `ha_model/bistate_duration.py` | `update_current_metrics()` | 56-94 | Add day filtering to constraint loop |
| `ha_model/bistate_duration.py` | `_build_mode_constraint_items()` | 205-283 | No change needed — constraints are correct, just metrics filtering |
| `ha_model/pool.py` | `_build_mode_constraint_items()` | 83-113 | No change needed — same reasoning |
| `ha_model/device.py` | `get_proper_local_adapted_tomorrow()` | 416-423 | Reuse pattern for day boundaries |
| `home_model/load.py` | `_last_completed_constraint` | 770, 1121 | Understand lifecycle, no changes |

### Architecture Constraints

- **Two-layer boundary**: `update_current_metrics()` lives in `ha_model/` which is fine — it accesses `self._constraints` from the domain layer via inheritance.
- **Logging**: Use lazy `%s` formatting, no f-strings. No periods at end of messages.
- **Constraint fields**: `end_of_constraint` and `start_of_constraint` are UTC `datetime` objects. Use `DATETIME_MIN_UTC` / `DATETIME_MAX_UTC` sentinels for unset values.

### Existing Patterns to Follow

The `_last_completed_constraint` day-window filter already in `update_current_metrics()` (lines 70-92) uses `get_next_time_from_hours()` with a 26-hour window. The new approach should use a cleaner local-midnight boundary instead but maintain backward compatibility with the fallback path.

The test file `test_bug_74_exact_calendar_metrics.py` already has a `_create_bistate_device()` helper and `_make_constraint()` factory — reuse these patterns. Follow `freezegun` for time-dependent tests.

### What NOT to Change

- Do NOT modify `_build_mode_constraint_items()` — the constraint list is correct for the solver. Only the metrics display needs filtering.
- Do NOT modify `get_for_solver_constraints()` — solver needs all future constraints for planning.
- Do NOT change JS card code — the sensors will naturally reflect the corrected values.

### Testing Patterns

- Use `_create_bistate_device()` pattern from `test_bug_74_exact_calendar_metrics.py`
- Use `_make_constraint()` or `TimeBasedSimplePowerLoadConstraint` directly
- Use `freezegun` with explicit UTC times for reproducibility
- Test with timezone offsets (e.g., Europe/Paris UTC+1/+2) to verify midnight boundary correctness
- 100% coverage required — all branches in the modified method

### References

- [Source: ha_model/bistate_duration.py#update_current_metrics] — main fix location
- [Source: ha_model/device.py#get_proper_local_adapted_tomorrow] — local time conversion pattern
- [Source: tests/test_bug_74_exact_calendar_metrics.py] — test patterns and helpers
- [Source: home_model/load.py#ack_completed_constraint] — constraint lifecycle
- [Source: home_model/constraints.py] — DATETIME_MIN_UTC, DATETIME_MAX_UTC sentinels

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
