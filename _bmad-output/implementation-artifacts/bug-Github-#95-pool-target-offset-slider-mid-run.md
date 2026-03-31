# Bug Fix: Pool target offset by already-run hours when slider is adjusted mid-run

Status: done
issue: 95
branch: "QS_95"

## Story

As a pool owner using Quiet Solar,
I want the target hours slider to set an absolute daily target,
so that adjusting it mid-run shows the value I set, not an inflated number.

## Bug Description

When the pool is running and the user adjusts the target hours slider, the displayed target is offset by the hours already run instead of reflecting the slider value directly.

**Steps to reproduce:**
1. Pool is running in `bistate_mode_default`, has run for 8h, target is 12h
2. Move the target slider to 10h --> target sensor shows 18h (10 + 8)
3. Move slider to 4h --> target shows 12h (4 + 8)
4. Move slider to 0h --> target shows 8h (0 + 8)

**Expected:** `qs_bistate_current_duration_h` sensor = slider value (absolute target).
**Actual:** `qs_bistate_current_duration_h` = slider_value + already_run_hours.

**Related issues:** #78 (daily target/actual metrics), #80 (calendar precompute refactor). Both improved metrics tracking but this offset bug persists.

## Root Cause Analysis

### Confirmed state

Pool completed its full target **yesterday**, so `_last_completed_constraint` IS set from the previous day's completed constraint. Today the pool is running under a new active constraint.

### Two independent bugs in `update_current_metrics()` (default/pool path)

Both bugs are in `ha_model/bistate_duration.py` lines 116-136.

**Bug 1 -- Active constraints loop has NO day lower bound:**

```python
for ct in self._constraints:
    if ct.end_of_constraint <= tomorrow_utc:       # <-- only upper bound!
        duration_s += ct.target_value
        run_s += ct.current_value
```

This counts ANY constraint with `end <= tomorrow_utc`, including constraints from previous days that haven't been cleaned up (e.g., after storage restore, missed cleanup cycle, or constraint replacement race). The `_last_completed_constraint` check (lines 125-133) has both a lower bound (`> today_utc`) and upper bound, but the active loop only has the upper bound.

**Fix:** Add `ct.end_of_constraint > today_utc` to the active loop filter.

**Bug 2 -- #64 same-end-date guard lost in #78/#80 refactoring:**

Bug #64 (`bug-Github-#68-carry-from-completed-constraint.md`) fixed the exact same double-count scenario: when `_last_completed_constraint` shares its `end_of_constraint` with an active constraint, the active constraint already absorbed the completed one's runtime via `push_live_constraint` carry-over (`load.py:1331-1345`). The #78/#80 refactoring moved `update_current_metrics` into `bistate_duration.py` and dropped this guard.

**Fix:** Add an `already_absorbed` check -- skip lcc when an active constraint shares its end date (same-day-cycle scenario).

**Bug 3 (minor) -- Boundary condition `>` vs `>=` near midnight/DST:**

`get_next_time_from_hours(0:00)` in `device.py:451` uses `timedelta(days=1)` which adds 24 wall-clock hours rather than advancing the calendar date. On DST spring-forward days, this shifts the result by 1h, making yesterday's constraint end fall inside today's window. Using `>=` for `today_utc` would include the exact-midnight case. The proper DST fix (`timedelta(days=1)` to calendar-day arithmetic) is a broader change for a separate task.

### Data Flow When Slider Changes

1. **UI** (`ui/resources/qs-pool-card.js:571`): `_setNumber(e.default_on_duration, dragValue)` sends slider value to HA number entity
2. **Number entity** (`number.py:130-146`): `async_set_native_value()` calls `user_set_default_on_duration()`
3. **Setter** (`ha_model/bistate_duration.py:138-144`): Sets `self.default_on_duration = new_value`, calls `do_run_check_load_activity_and_constraints(time)`
4. **Constraint build** (`ha_model/bistate_duration.py:272-287`): Creates `ConstraintItemType(target_value=self.default_on_duration * 3600.0)` -- correct (10h = 36000s)
5. **Constraint creation** (`ha_model/bistate_duration.py:546-562`): Creates `TimeBasedSimplePowerLoadConstraint(target_value=36000, initial_value=0)`, calls `push_live_constraint()`
6. **Push** (`home_model/load.py:1305-1373`): Carries over `current_value` from old constraint, replaces it
7. **Metrics** (`ha_model/bistate_duration.py:576`): `update_current_metrics(time)` sums constraints -- **this is where the double-count happens**
8. **UI reads** `qs_bistate_current_duration_h` sensor as the displayed "target"

## Acceptance Criteria

1. **AC1**: When user adjusts `default_on_duration` slider to X hours while pool is running, `qs_bistate_current_duration_h` sensor immediately reflects X (not X + already_run)
2. **AC2**: `qs_bistate_current_on_h` (actual run hours) remains accurate -- reflects real accumulated runtime for the day
3. **AC3**: If user increases target above already-run hours (e.g., 8h done, set to 12h), pool schedules the remaining 4h
4. **AC4**: If user decreases target below already-run hours (e.g., 8h done, set to 4h), pool stops (target already exceeded)
5. **AC5**: Setting slider to 0h while pool has run should show target=0h, actual=N (whatever ran)
6. **AC6**: Existing carry-over logic for constraint EXTENSION (same target, extending end time) must still work -- don't break the legitimate use case
7. **AC7**: Calendar-mode pools (`bistate_mode_auto`, `bistate_mode_exact_calendar`) are unaffected
8. **AC8**: 100% test coverage maintained

## Tasks / Subtasks

- [x] Task 1: Fix active constraints loop -- add day lower bound (AC: 1, 2)
  - [x] 1.1: In `update_current_metrics()` default path (~line 120), change the active constraint filter from `ct.end_of_constraint <= tomorrow_utc` to `ct.end_of_constraint > today_utc and ct.end_of_constraint <= tomorrow_utc`

- [x] Task 2: Restore #64 same-end-date guard for lcc (AC: 1, 2, 6)
  - [x] 2.1: In `update_current_metrics()` default path (~line 125), add an `already_absorbed` guard: skip lcc when any active today-constraint shares its `end_of_constraint` or `initial_end_of_constraint`
  - [x] 2.2: Use `getattr(lcc, "initial_end_of_constraint", lcc.end_of_constraint)` for safe access

- [x] Task 3: Fix boundary condition `>` vs `>=` for lcc (AC: 1)
  - [x] 3.1: Change `lcc.end_of_constraint > today_utc` to `lcc.end_of_constraint >= today_utc` to handle exact-midnight boundary
  - [x] 3.2: Note: DST fix for `get_next_time_from_hours` (`timedelta(days=1)` to calendar-day arithmetic in `device.py:451`) is deferred to a separate task

- [x] Task 4: Tests -- same-end-date double-count (AC: 1, 2, 8)
  - [x] 4.1: In `test_ha_pool.py`: test that completed + active constraints with the same end date do NOT double-count target
  - [x] 4.2: In `test_ha_pool.py`: test that old active constraint from previous day does not leak through missing lower bound
  - [x] 4.3: In `test_bug_78_daily_metrics.py`: test same-end-date scenario on generic bistate device

- [x] Task 5: Verify existing tests pass (AC: 6, 7, 8)
  - [x] 5.1: Verify these existing tests still pass (different end dates, both counted -- not absorbed):
    - `test_pool_update_current_metrics_completed_and_active_sums_both`
    - `test_default_mode_includes_last_completed_from_today`
    - `test_pool_update_current_metrics_completed_only_shows_completed`
    - `test_pool_update_current_metrics_completed_from_today_included`
  - [x] 5.2: Run full quality gate: `python scripts/qs/quality_gate.py`

## Dev Notes

### Key Files

| File | Lines | Role |
|------|-------|------|
| `custom_components/quiet_solar/home_model/load.py` | 1305-1373 | `push_live_constraint()` -- carry-over and replacement logic |
| `custom_components/quiet_solar/ha_model/bistate_duration.py` | 66-136 | `update_current_metrics()` -- metrics summation |
| `custom_components/quiet_solar/ha_model/bistate_duration.py` | 138-144 | `user_set_default_on_duration()` -- entry point |
| `custom_components/quiet_solar/ha_model/bistate_duration.py` | 272-287 | `_build_mode_constraint_items()` -- constraint creation for default mode |
| `custom_components/quiet_solar/ha_model/bistate_duration.py` | 534-578 | `check_load_activity_and_constraints()` -- orchestrator |
| `custom_components/quiet_solar/ha_model/pool.py` | 89-119 | Pool override of `_build_mode_constraint_items()` (auto/winter modes) |
| `custom_components/quiet_solar/home_model/constraints.py` | 70-170 | `LoadConstraint` base class, `current_value`/`target_value`/`requested_target_value` |
| `custom_components/quiet_solar/home_model/constraints.py` | 2380-2401 | `TimeBasedSimplePowerLoadConstraint` |
| `custom_components/quiet_solar/number.py` | 130-146 | Number entity `async_set_native_value()` |
| `custom_components/quiet_solar/ui/resources/qs-pool-card.js` | 75, 97-101, 556-581 | UI: reads `qs_bistate_current_duration_h` as "target", sends slider value via `default_on_duration` |

### Architecture Constraints

- **Two-layer boundary**: `home_model/load.py` is domain logic (no HA imports). `ha_model/bistate_duration.py` bridges HA. Fix must respect this.
- `push_live_constraint()` is in the domain layer -- it has no concept of "user initiated" beyond `constraint.from_user`. The fix should use existing constraint attributes, not HA-specific signals.
- `update_current_metrics()` is in the HA layer -- it can use `_last_completed_constraint` and `_constraints` from the domain layer.
- Do NOT modify `SOLVER_STEP_S` or solver logic.
- Lazy logging: `_LOGGER.debug("msg %s", var)` -- no f-strings in log calls.

### Constraint Lifecycle Reference

1. **Created**: `TimeBasedSimplePowerLoadConstraint(target_value=N, initial_value=0)`
2. **Pushed**: `push_live_constraint()` -- may carry `current_value` from lcc or replaced constraint
3. **Updated**: `update_live_constraints()` -- increments `current_value` as pool runs
4. **Completed**: When `current_value >= target_value`, `ack_completed_constraint()` sets `_last_completed_constraint`
5. **Metrics**: `update_current_metrics()` sums active constraints + lcc for today

### Related Bug Story References

- `bug-Github-#78-fix-bistate-daily-target-actual-hours.md` -- fixed today-only metrics display, added date filtering
- `bug-Github-#80-bistate-calendar-precompute-refactor.md` -- refactored calendar event metrics computation
- `bug-Github-#68-carry-from-completed-constraint.md` -- added the carry-over logic that may be contributing to this bug

### Test Infrastructure

- Use `MinimalTestHome` / `MinimalTestLoad` from `tests/factories.py` for constraint tests
- Use `create_constraint()` factory for creating test constraints
- Use `freezegun` for time-dependent scenarios
- Pool-specific test fixtures may be needed -- check `tests/` for existing pool tests
- asyncio_mode=auto -- no `@pytest.mark.asyncio` decorator needed

### Fix Scope

All three changes are in `update_current_metrics()` default/pool path (`bistate_duration.py:116-136`). No changes to `push_live_constraint()`, constraint classes, or UI. This is a metrics-display-only fix -- the constraint lifecycle and solver are unaffected.

### Regression Risk

The existing tests that MUST still pass use **different** end dates for completed vs active constraints (e.g., 8:00 vs 17:00). The `already_absorbed` guard only skips lcc when it shares an end date with an active constraint, so these different-end-date scenarios are unaffected.

### Deferred Work

DST fix for `get_next_time_from_hours()` in `device.py:451` -- replacing `timedelta(days=1)` with calendar-day arithmetic like `get_proper_local_adapted_tomorrow` uses. This affects all constraint end-date computation, not just metrics, and should be a separate task.

### Project Structure Notes

- Alignment: fix stays within existing constraint/bistate architecture
- No new files needed -- fix is within existing code paths
- No UI changes needed -- the sensor values are the source of truth

### References

- [Source: home_model/load.py#push_live_constraint] -- carry-over logic lines 1331-1345, replacement logic lines 1347-1367
- [Source: ha_model/bistate_duration.py#update_current_metrics] -- metrics summation lines 116-136
- [Source: ha_model/bistate_duration.py#user_set_default_on_duration] -- slider entry point lines 138-144
- [Source: ha_model/bistate_duration.py#_build_mode_constraint_items] -- constraint creation lines 272-287
- [Source: home_model/constraints.py#LoadConstraint] -- constraint base class lines 70-170

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Completion Notes List

- Story created from GitHub issue #95 analysis
- Root cause identified via Cursor plan: two bugs in `update_current_metrics()` default path -- (1) missing day lower bound on active constraints loop, (2) #64 same-end-date lcc guard lost in #78/#80 refactoring
- All three fixes are scoped to `update_current_metrics()` in `bistate_duration.py` -- no constraint lifecycle or solver changes
- Related issues: #64/#68 (original same-end-date guard), #78 (daily metrics), #80 (calendar refactor)
- Implementation complete (2026-03-31): all 3 bugs fixed in `update_current_metrics()` default path
- Bug 1 fix: added `ct.end_of_constraint > today_utc` lower bound to active constraints loop (line 121)
- Bug 2 fix: added `already_absorbed` guard using `type()` + `initial_end_of_constraint` to skip lcc when active constraint of same type shares end date (lines 127-135)
- Bug 3 fix: changed `>` to `>=` for lcc today_utc boundary check (line 139)
- Added 5 new tests: 2 in test_ha_pool.py (yesterday leak, same-end-date), 3 in test_bug_78_daily_metrics.py (same-end-date, yesterday leak, exact-midnight boundary)
- All quality gates green, 100% coverage maintained
- Review (2026-03-31): tightened `already_absorbed` with `type(ct) == type(lcc)` to mirror push_live_constraint preconditions
- Review: added divergent `initial_end_of_constraint` tests (1 pool, 1 bistate) for extended-lcc scenario
- Review: converted pool absorption tests from MagicMock to real TimeBasedSimplePowerLoadConstraint
- Review: updated PR risk assessment from LOW to CRITICAL

### File List

- `custom_components/quiet_solar/ha_model/bistate_duration.py` — fixed `update_current_metrics()` default path
- `tests/test_ha_pool.py` — added 3 regression tests for bug #95
- `tests/test_bug_78_daily_metrics.py` — added 4 regression tests for bug #95
