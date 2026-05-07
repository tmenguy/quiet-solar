# Bug Fix: move calendar pre-compute into update_current_metrics and remove _is_current_calendar_mode

Status: done
issue: 80
branch: "QS_80"

## Story

As a developer maintaining bistate_duration.py,
I want the calendar metrics computation to live inside `update_current_metrics` (not pre-computed externally in `check_load_activity_and_constraints`),
so that the method is self-contained, the mode check is always fresh, and there is no fragile coupling between the pre-compute step and the consumer.

## Bug Description

QS_78 introduced calendar pre-computation in `check_load_activity_and_constraints` (bistate_duration.py:551-588) that stores results in three instance variables (`_is_current_calendar_mode`, `_today_calendar_target_s`, `_today_calendar_past_actual_s`). These are consumed only by `update_current_metrics`. This design has three issues:

### Issue 1: Pre-compute is in the wrong place
The calendar event fetching and metric calculation (lines 551-588) sits inside `check_load_activity_and_constraints`, but it is only consumed by `update_current_metrics` (called at line 590). If `update_current_metrics` were ever called independently, it would use stale cached values. The pre-compute should live inside `update_current_metrics` itself.

### Issue 2: `_is_current_calendar_mode` is a cached boolean
Line 83 checks `self._is_current_calendar_mode` instead of calling `self._is_calendar_based_mode(self.bistate_mode)` directly. The cached boolean adds an unnecessary layer of indirection and could become stale if the mode changes between pre-compute and consumption.

### Issue 3: Pool super() fragility (design note)
Pool's `_build_mode_constraint_items` (pool.py:89-119) returns directly for `bistate_mode_auto`/`pool_winter_mode` without calling `super()`. While this is correct behavior (pool handles these modes differently), the fact that the pre-compute lives separately from its consumer makes the overall flow harder to reason about. Moving the pre-compute into `update_current_metrics` eliminates this concern entirely.

## Fix Plan

### Task 1: Make `update_current_metrics` async and self-contained [x]

**File:** `ha_model/bistate_duration.py`

1. Change signature: `async def update_current_metrics(self, time: datetime, end_range: dt_time | None = None)`
2. Replace `if self._is_current_calendar_mode:` (line 83) with `if self._is_calendar_based_mode(self.bistate_mode):`
3. Move the calendar event fetch + metric computation (current lines 551-567) into the calendar branch of `update_current_metrics`:
   - `await self.get_next_scheduled_events(time=today_start_utc, give_currently_running_event=True)`
   - Compute `target_s` and `past_actual_s` as **local variables** (not instance vars)
   - Use them directly for `duration_s` and `run_s`
4. Move the AC5 sanity check (current lines 569-586) into `update_current_metrics` as well, inside the calendar branch
5. Optional throttle: add a cache guard so the calendar fetch only runs if `time` has advanced by more than N seconds since last fetch (user suggests "every few seconds/minutes"). Use a `_last_calendar_fetch_time` + `_cached_calendar_events` pattern, or simply accept the cost since `check_load_activity_and_constraints` already runs on a ~7s cycle.

### Task 2: Remove pre-compute section from `check_load_activity_and_constraints` [x]

**File:** `ha_model/bistate_duration.py`

1. Delete lines 551-588 (the entire "Pre-compute calendar metrics for today" block and the `else: self._is_current_calendar_mode = False` block)
2. Change line 590 from `self.update_current_metrics(time)` to `await self.update_current_metrics(time)`

### Task 3: Remove stale instance variables [x]

**File:** `ha_model/bistate_duration.py`

1. Remove from `__init__` (lines 57-59):
   - `self._is_current_calendar_mode: bool = False`
   - `self._today_calendar_target_s: float = 0.0`
   - `self._today_calendar_past_actual_s: float = 0.0`

### Task 4: Update tests [x]

Multiple test files call `device.update_current_metrics(now)` synchronously. After making it async:

- `tests/test_bug_78_daily_metrics.py` — ~10 calls to `device.update_current_metrics(now)` must become `await device.update_current_metrics(now)`, test functions must be `async`
- `tests/test_bug_74_exact_calendar_metrics.py` — ~4 calls
- `tests/test_bug_70_cumulus_rapid_cycling.py` — ~3 calls
- `tests/test_ha_pool.py` — ~8 calls using `QSBiStateDuration.update_current_metrics(pool, now)` must become `await QSBiStateDuration.update_current_metrics(pool, now)`

For calendar-path tests: ensure the fake device's `get_next_scheduled_events` is an async mock returning the expected events.

For default-path tests (pool, bistate_mode_default): `_is_calendar_based_mode` returns False, so no calendar fetch happens. These tests just need the `await` added.

## Files to Modify

| File | Change |
|------|--------|
| `ha_model/bistate_duration.py` | Make `update_current_metrics` async, inline calendar logic, remove pre-compute, remove cached vars |
| `tests/test_bug_78_daily_metrics.py` | Async test functions, `await update_current_metrics` |
| `tests/test_bug_74_exact_calendar_metrics.py` | Async test functions, `await update_current_metrics` |
| `tests/test_bug_70_cumulus_rapid_cycling.py` | Async test functions, `await update_current_metrics` |
| `tests/test_ha_pool.py` | Async test functions, `await update_current_metrics` |

## Technical Notes

- `get_next_scheduled_events` is defined in `home_model/load.py` (the AbstractLoad base class) and is async. This is why `update_current_metrics` must become async.
- The two-layer boundary is respected: `update_current_metrics` lives in `ha_model/` and calls `get_next_scheduled_events` which is on `AbstractLoad` (domain layer). The calendar entity access happens inside `get_next_scheduled_events` via HA APIs, which is fine since `QSBiStateDuration` inherits from `HADeviceMixin`.
- Pool's `_is_calendar_based_mode` override returns `False` for `bistate_mode_auto`/`pool_winter_mode`, so pool will always take the default path in `update_current_metrics` — no calendar fetch, no async overhead for pool in those modes.
