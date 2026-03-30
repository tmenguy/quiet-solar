# Bug + Refactor: Fix exact-calendar metrics and deduplicate update_current_metrics

issue: 74
branch: "QS_74"
Status: done

## Story

As a Quiet Solar user with a calendar-driven bistate device (climate/cumulus),
I want the UI card to correctly show the target duration for upcoming calendar constraints,
so that I see "3h30" instead of "0h" when my next scheduled event is tomorrow morning.

## Bug Description

A climate device in `bistate_mode_exact_calendar` has a calendar event for **tomorrow 6:30-10:00**. Current time is 10:23. The card shows **target = 0h** instead of **3h30**.

### Root Cause

`update_current_metrics` (bistate_duration.py:56-93) applies a **day-window filter** to ALL constraints, including active ones. The day window is bounded by `default_on_finish_time` (defaults to midnight):

- `end_day` = next midnight from now = **tomorrow 00:00**
- Constraint: `start_of_constraint` = tomorrow 06:30, `end_of_constraint` = tomorrow 10:00
- Filter check: `end_of_constraint (10:00) <= end_day (00:00)` → **FALSE**
- Filter check: `start_of_constraint (06:30) <= end_day (00:00)` → **FALSE**
- **Result**: constraint excluded, target = 0h

The day-window filter was copied from pool.py in QS_70, where it's correct because pool constraints always end at `default_on_finish_time` (== `end_day`). But for calendar-based modes, constraint boundaries come from calendar events and can exceed the day window.

### Fix

**Active constraints** (`_constraints`) should always be included in metrics — they are current/upcoming by definition. The **day-window filter should only apply to `_last_completed_constraint`** (to avoid showing stale metrics from a previous day cycle).

This distinction is safe because:
- Pool constraints always have `end_of_constraint` within the day window → no behavior change
- Calendar constraints can exceed the window → now correctly included
- `_last_completed_constraint` still needs filtering to avoid showing yesterday's completed hours

## Acceptance Criteria

1. **Given** a bistate in exact_calendar mode with a constraint starting after the current day-window boundary, **When** `update_current_metrics` runs, **Then** the constraint's `target_value` and `current_value` are included in the displayed metrics
2. **Given** active constraints exist, **When** `update_current_metrics` runs, **Then** all active constraints are summed without day-window filtering
3. **Given** no active constraints but `_last_completed_constraint` exists, **When** `update_current_metrics` runs, **Then** the completed constraint is included only if it falls within the current day window (existing behavior preserved)
4. `QSPool` no longer overrides `update_current_metrics` — deleted from `ha_model/pool.py`
5. All existing tests pass; new tests cover the exact-calendar bug scenario
6. Quality gate passes: `python scripts/qs/quality_gate.py`

## Tasks / Subtasks

- [x] Task 1: Fix `update_current_metrics` in `bistate_duration.py` (AC: #1, #2, #3)
  - [x] 1.1 Restructure the method: if `_constraints` exist, sum ALL of them (no day-window filter). Only apply day-window filter in the `elif _last_completed_constraint` branch.
  - [x] 1.2 The method signature and output attributes (`qs_bistate_current_on_h`, `qs_bistate_current_duration_h`) stay the same.

- [x] Task 2: Remove duplicate `update_current_metrics` from `pool.py` (AC: #4)
  - [x] 2.1 Delete the `update_current_metrics` override in `ha_model/pool.py` (lines 55-90)
  - [x] 2.2 Remove now-unused imports: `DATETIME_MIN_UTC`, `timedelta`

- [x] Task 3: Add tests for the exact-calendar bug (AC: #5)
  - [x] 3.1 Test: active constraint beyond day-window boundary → metrics show its target/current values
  - [x] 3.2 Test: active constraint within day-window → metrics still work (regression guard)
  - [x] 3.3 Test: no active constraints, last_completed within window → shows completed (existing behavior)
  - [x] 3.4 Test: no active constraints, last_completed outside window → shows 0 (existing behavior)

- [x] Task 4: Run quality gate (AC: #6)
  - [x] `python scripts/qs/quality_gate.py` — all checks pass

## Dev Notes

### Class Hierarchy

```
QSBiStateDuration  (ha_model/bistate_duration.py)  ← update_current_metrics lives HERE
  ├── QSClimateDuration  (ha_model/climate_controller.py)  ← triggers the bug in exact_calendar mode
  └── QSOnOffDuration  (ha_model/on_off_duration.py)
      └── QSPool  (ha_model/pool.py)  ← REMOVE the duplicate override
```

### Fix Shape (bistate_duration.py)

Current logic (broken for calendar constraints):
```python
# builds ct_to_probe from _constraints OR _last_completed_constraint
# then day-window-filters ALL of them  ← BUG: filters out future calendar constraints
```

Fixed logic:
```python
if self._constraints:
    # Active constraints: always include ALL (they are current by definition)
    for ct in self._constraints:
        duration_s += ct.target_value
        run_s += ct.current_value
elif self._last_completed_constraint is not None:
    # Fallback: day-window filter only the completed constraint
    ct = self._last_completed_constraint
    # ... existing day-window check ...
    duration_s += ct.target_value
    run_s += ct.current_value
```

### Why this doesn't break pool behavior

Pool constraints always have `end_of_constraint = get_next_time_from_hours(default_on_finish_time, time)`, which equals `end_day` in the filter. They always pass the filter. Removing the filter for active constraints is a no-op for pools.

### Import Cleanup in pool.py

After removing the method, `DATETIME_MIN_UTC` and `timedelta` are no longer used in `pool.py` — remove both imports.

### Existing Test Coverage

- `tests/test_ha_pool.py` (lines 191-390): pool metrics tests — call `QSPool.update_current_metrics` directly. These must be updated to NOT reference `QSPool.update_current_metrics` (method no longer exists on QSPool; use the inherited parent method).
- `tests/test_bug_70_cumulus_rapid_cycling.py` (lines 265-375): bistate metrics fallback tests — already test the parent method.

### Architecture Constraints

- Two-layer boundary: all files in `ha_model/` — no boundary issues
- `update_current_metrics` is synchronous — no async concerns
- No config keys or translations affected

### References

- [Source: ha_model/bistate_duration.py:56-93] — current (buggy) implementation
- [Source: ha_model/pool.py:55-90] — duplicate to remove
- [Source: bug-Github-#70-fix-cumulus-rapid-cycling.md] — QS_70 story that copied pool logic to bistate
- [Source: deferred-work.md] — notes `end_range` param untested, `_last_completed_constraint` wipe risk

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
None needed — straightforward logic fix.

### Completion Notes List
- Active constraints now always included in metrics (no day-window filter)
- Day-window filter preserved for `_last_completed_constraint` fallback only
- Pool `update_current_metrics` override deleted — inherits from `QSBiStateDuration`
- Pool tests updated: references changed from `QSPool` to `QSBiStateDuration`, two tests updated to test the completed-constraint path (where day-window logic still applies)
- All quality gates pass: 100% coverage, ruff, mypy, translations

### File List
- `custom_components/quiet_solar/ha_model/bistate_duration.py` — restructured `update_current_metrics`
- `custom_components/quiet_solar/ha_model/pool.py` — removed duplicate method + unused imports
- `tests/test_bug_74_exact_calendar_metrics.py` — new test file (4 tests)
- `tests/test_ha_pool.py` — updated to use parent class + adjusted 2 tests for new logic
