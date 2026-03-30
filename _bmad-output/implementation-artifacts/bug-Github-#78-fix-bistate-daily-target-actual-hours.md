# Bug Fix: bistate/pool target and actual hours display sums across days instead of current day only

Status: dev-complete
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

**Root cause is twofold:**
1. `_constraints` includes tomorrow's events (loaded via `get_next_scheduled_events()` with a 30h window) and they're all summed blindly.
2. Past completed events from today (like 6:00-7:00) are gone from `_constraints` once completed, but `_last_completed_constraint` only stores the **last** completed one — earlier completions are lost. So for calendar modes, you cannot reconstruct today's actual from constraints alone.

## Design: Two Paths by Mode Type

The fix uses distinct logic depending on whether the mode is calendar-based or not.

### Calendar-based modes (`bistate_mode_auto`, `bistate_mode_exact_calendar` with calendar attached)
- **Target**: Query the calendar for ALL of today's events (local midnight to next midnight), sum durations. This captures both past and future events regardless of constraint state.
- **Actual**: Past events (event_end <= now) = their full duration (constraint was met). Currently active constraint = its `current_value`. Use `_last_completed_constraint` only for a sanity-check warning log.

### Default/pool modes (`bistate_mode_default`, pool `bistate_mode_auto`/`pool_winter_mode`)
- **Target**: Filter `_constraints` to today only (end_of_constraint < next midnight) + `_last_completed_constraint` target if from today.
- **Actual**: Filter `_constraints` current_values to today + `_last_completed_constraint` current_value if from today.

### Mode detection
Add a virtual method `_is_calendar_based_mode(bistate_mode)` on `QSBiStateDuration` that returns `True` for `bistate_mode_auto`/`bistate_mode_exact_calendar` when a calendar is attached. `QSPool` overrides this to return `False` for `bistate_mode_auto` and `pool_winter_mode` (since pool overrides those modes to not use calendar).

## Acceptance Criteria

1. **AC1: Calendar modes show today-only target**
   - Given a bistate device in `bistate_mode_auto` or `bistate_mode_exact_calendar` mode with a calendar attached
   - When the calendar has events spanning multiple days
   - Then Target Hours = sum of **calendar event durations** for today (local midnight to next midnight), capturing both past and future events regardless of constraint state
   - And events ending after local next-midnight are excluded

2. **AC2: Calendar modes show today-only actual**
   - Given a bistate device in calendar mode
   - Then Actual Hours = sum of full durations of **past calendar events** (event_end <= now) + `current_value` of any **currently active** constraint whose `end_of_constraint` is within today
   - `_last_completed_constraint` is NOT used for the actual sum — only for the sanity-check warning (AC5)

3. **AC3: Default/pool modes show today-only metrics**
   - Given a bistate device in `bistate_mode_default` or a pool in `bistate_mode_auto`/`pool_winter_mode`
   - Then Target Hours = sum of `target_value` from `_constraints` ending within today + `_last_completed_constraint.target_value` if from today
   - And Actual Hours = sum of `current_value` from `_constraints` ending within today + `_last_completed_constraint.current_value` if from today and not already in `_constraints`

4. **AC4: Day boundary uses local midnight**
   - The "today" window is `(start_of_today_utc, start_of_tomorrow_utc)` derived from local midnight
   - This correctly handles timezone offsets and DST transitions

5. **AC5: Sync warning for calendar modes**
   - When in calendar modes and `_last_completed_constraint` exists for today
   - If its `current_value` significantly diverges from the inferred past schedule runtime
   - Then emit a `_LOGGER.warning()` with the discrepancy details

6. **AC6: Controlled test migration**
   - `test_bug_74_exact_calendar_metrics.py`: `test_active_constraint_beyond_day_window_shows_target` updated — a tomorrow-only constraint now correctly shows 0h target
   - `test_ha_pool.py`: test setups updated with `bistate_mode` where needed
   - All other existing tests in `test_coverage_bistate_duration.py` continue to pass

## Tasks / Subtasks

- [x] Task 1: Add helper methods to `QSBiStateDuration` (AC: #4)
  - [x] Add `_get_today_boundaries(time) -> tuple[datetime, datetime]` returning `(start_of_today_utc, start_of_tomorrow_utc)` using the `get_proper_local_adapted_tomorrow` pattern
  - [x] Add `_is_calendar_based_mode(bistate_mode: str) -> bool` returning `True` for `bistate_mode_auto`/`bistate_mode_exact_calendar` when `self.calendar is not None`

- [x] Task 2: Override `_is_calendar_based_mode` in `QSPool` (AC: #3)
  - [x] Return `False` for `bistate_mode_auto` and `pool_winter_mode` (pool overrides these modes to not use calendar)
  - [x] Delegate to `super()` for other modes

- [x] Task 3: Pre-compute calendar metrics in `check_load_activity_and_constraints` (AC: #1, #2)
  - [x] Before the `self.update_current_metrics(time)` call, detect if current mode is calendar-based via `_is_calendar_based_mode(bistate_mode)`
  - [x] If calendar-based: call `get_next_scheduled_events(time=today_start_utc, give_currently_running_event=True)`, filter to events ending before tomorrow
  - [x] Compute `_today_calendar_target_s` = sum of all today's event durations
  - [x] Compute `_today_calendar_past_actual_s` = sum of durations of past events (event_end <= now)
  - [x] Store `_is_current_calendar_mode = True` flag; set `False` otherwise
  - [x] Sanity-check `_last_completed_constraint` against past events, emit warning log if divergent

- [x] Task 4: Rewrite `update_current_metrics()` with two branches (AC: #1, #2, #3)
  - [x] **Calendar path** (`_is_current_calendar_mode is True`):
    - `duration_s = _today_calendar_target_s`
    - `run_s = _today_calendar_past_actual_s`
    - For each constraint in `_constraints` whose `end_of_constraint` is within today: `run_s += ct.current_value` (currently running events)
  - [x] **Default path** (non-calendar modes):
    - Compute today boundaries via `_get_today_boundaries(time)`
    - Filter `_constraints` to those ending within today, sum target/current
    - If `_last_completed_constraint` is from today and not already in `_constraints`: add its target/current
  - [x] Preserve the existing no-constraint fallback to `_last_completed_constraint` with day-window filter

- [x] Task 5: Write tests (AC: #1-#6)
  - [x] New file `tests/test_bug_78_daily_metrics.py`:
    - Calendar mode with multi-day events: only today's counted for target
    - Calendar mode actual from past events + active constraint current_value
    - Default mode: filter to today + last-completed
    - Pool auto/winter: not calendar-based, uses default path
    - Day boundary at local midnight with timezone offset
    - `_last_completed_constraint` sanity-check warning log
    - Edge: no events today but active constraints from tomorrow = 0h target
  - [x] Update `tests/test_bug_74_exact_calendar_metrics.py`: `test_active_constraint_beyond_day_window_shows_target` — a tomorrow-only constraint should now show 0h target (correct per new design)
  - [x] Update `tests/test_ha_pool.py`: add `bistate_mode` to test setups if needed, update expectations for day filtering

## Dev Notes

### Key Code Locations

| File | Method/Area | Lines | What to change |
|------|-------------|-------|----------------|
| `ha_model/bistate_duration.py` | `update_current_metrics()` | 56-94 | Rewrite with calendar/default two-path logic |
| `ha_model/bistate_duration.py` | `check_load_activity_and_constraints()` | 285-536 | Add calendar metrics pre-computation before `update_current_metrics(time)` call at line 534 |
| `ha_model/bistate_duration.py` | class body | new | Add `_get_today_boundaries()`, `_is_calendar_based_mode()` helpers |
| `ha_model/pool.py` | class body | new | Override `_is_calendar_based_mode()` to return False for auto/winter |
| `ha_model/device.py` | `get_proper_local_adapted_tomorrow()` | 416-423 | Reuse pattern for day boundaries (no changes) |
| `ha_model/device.py` | `get_next_scheduled_events()` | 527+ | Reuse as-is, passing today's midnight as `time` for calendar pre-computation |
| `home_model/load.py` | `_last_completed_constraint` | 770, 1121 | Understand lifecycle (no changes) |

### Architecture Constraints

- **Two-layer boundary**: `update_current_metrics()` lives in `ha_model/` which is fine — it accesses `self._constraints` from the domain layer via inheritance.
- **Logging**: Use lazy `%s` formatting, no f-strings. No periods at end of messages.
- **Constraint fields**: `end_of_constraint` and `start_of_constraint` are UTC `datetime` objects. Use `DATETIME_MIN_UTC` / `DATETIME_MAX_UTC` sentinels for unset values.

### Implementation Sketch

**`_get_today_boundaries`** (new helper):
```python
def _get_today_boundaries(self, time: datetime) -> tuple[datetime, datetime]:
    tomorrow_utc = self.get_proper_local_adapted_tomorrow(time)
    local_now = time.replace(tzinfo=pytz.UTC).astimezone(tz=None)
    local_today = datetime(local_now.year, local_now.month, local_now.day)
    today_utc = local_today.replace(tzinfo=None).astimezone(tz=pytz.UTC)
    return today_utc, tomorrow_utc
```

**`_is_calendar_based_mode`** (new virtual method):
```python
def _is_calendar_based_mode(self, bistate_mode: str) -> bool:
    return (
        bistate_mode in ("bistate_mode_auto", "bistate_mode_exact_calendar")
        and self.calendar is not None
    )
```

**Pool override**:
```python
def _is_calendar_based_mode(self, bistate_mode: str) -> bool:
    if bistate_mode in ("bistate_mode_auto", "pool_winter_mode"):
        return False
    return super()._is_calendar_based_mode(bistate_mode)
```

**Calendar pre-computation** (in `check_load_activity_and_constraints`, before line 534):
- Call `get_next_scheduled_events(time=today_start_utc, give_currently_running_event=True)`
- Filter events to those ending before `tomorrow_utc`
- Past events (event_end <= now): add full duration to `_today_calendar_past_actual_s`
- All today events: add duration to `_today_calendar_target_s`

**`update_current_metrics` two-branch rewrite:**
- Calendar path: use pre-computed `_today_calendar_target_s`/`_today_calendar_past_actual_s` + active constraint `current_value` for in-progress events
- Default path: day-filter `_constraints` + `_last_completed_constraint` if from today

### Existing Patterns to Follow

The `_last_completed_constraint` day-window filter already in `update_current_metrics()` (lines 70-92) uses `get_next_time_from_hours()` with a 26-hour window. The new approach should use a cleaner local-midnight boundary instead but maintain backward compatibility with the fallback path.

The test file `test_bug_74_exact_calendar_metrics.py` already has a `_create_bistate_device()` helper and `_make_constraint()` factory — reuse these patterns. Follow `freezegun` for time-dependent tests.

### What NOT to Change

- Do NOT modify `_build_mode_constraint_items()` — the constraint list is correct for the solver. Only the metrics display needs filtering.
- Do NOT modify `get_for_solver_constraints()` — solver needs all future constraints for planning.
- Do NOT change JS card code — the sensors will naturally reflect the corrected values.
- Do NOT change `get_next_scheduled_events()` — reuse as-is by passing today's midnight as `time`.

### Test Impact on Existing Tests

- **`test_bug_74_exact_calendar_metrics.py`**: `test_active_constraint_beyond_day_window_shows_target` will need updated expectations — a tomorrow-only constraint should now show 0h target (correct per new design, since it's not today's constraint).
- **`test_ha_pool.py`**: May need `bistate_mode` set on devices since `update_current_metrics` now branches on mode. Update expectations if constraints span days.

### Testing Patterns

- Use `_create_bistate_device()` pattern from `test_bug_74_exact_calendar_metrics.py`
- Use `_make_constraint()` or `TimeBasedSimplePowerLoadConstraint` directly
- Use `freezegun` with explicit UTC times for reproducibility
- Test with timezone offsets (e.g., Europe/Paris UTC+1/+2) to verify midnight boundary correctness
- 100% coverage required — all branches in the modified method

### References

- [Source: ha_model/bistate_duration.py#update_current_metrics] — main fix location, two-branch rewrite
- [Source: ha_model/bistate_duration.py#check_load_activity_and_constraints:534] — insertion point for calendar pre-computation
- [Source: ha_model/device.py#get_proper_local_adapted_tomorrow] — local time conversion pattern
- [Source: ha_model/device.py#get_next_scheduled_events] — calendar query for pre-computation
- [Source: ha_model/pool.py#_build_mode_constraint_items] — pool overrides auto/winter to not use calendar
- [Source: tests/test_bug_74_exact_calendar_metrics.py] — test patterns and helpers, expectations to update
- [Source: home_model/load.py#ack_completed_constraint] — constraint lifecycle
- [Source: home_model/constraints.py] — DATETIME_MIN_UTC, DATETIME_MAX_UTC sentinels
- [Source: Cursor plan] — fix_daily_metrics_display_23c9a2dc.plan.md (synthesized into this story)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
- Ruff format auto-fixed bistate_duration.py formatting
- Removed unused DATETIME_MIN_UTC import from bistate_duration.py (lint fix)
- Updated test_bug_70_cumulus_rapid_cycling.py to mock day boundaries (constraints with `end = time + 12h` crossed midnight)

### Completion Notes List
- All 5 tasks completed, all quality gates passing (100% coverage)
- Updated 3 existing test files for compatibility with new day-filtering behavior
- Pool metric tests updated: active + completed constraints from today are now both summed (previously only active was counted when present)
- Bug_74 test expectations updated: tomorrow-only constraint now correctly shows 0h target

### File List
- `custom_components/quiet_solar/ha_model/bistate_duration.py` — helper methods, two-branch update_current_metrics, calendar pre-computation
- `custom_components/quiet_solar/ha_model/pool.py` — `_is_calendar_based_mode` override
- `tests/test_bug_78_daily_metrics.py` — new comprehensive test file (22 tests)
- `tests/test_bug_74_exact_calendar_metrics.py` — updated expectations and day boundary mocking
- `tests/test_ha_pool.py` — updated FakeQSPool and metric tests for new behavior
- `tests/test_bug_70_cumulus_rapid_cycling.py` — added day boundary mocking for timezone safety
