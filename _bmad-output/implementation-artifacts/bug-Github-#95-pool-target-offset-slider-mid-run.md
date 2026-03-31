# Bug Fix: Pool target offset by already-run hours when slider is adjusted mid-run

Status: ready-for-dev
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

### Data Flow When Slider Changes

1. **UI** (`ui/resources/qs-pool-card.js:571`): `_setNumber(e.default_on_duration, dragValue)` sends slider value to HA number entity
2. **Number entity** (`number.py:130-146`): `async_set_native_value()` calls `user_set_default_on_duration()`
3. **Setter** (`ha_model/bistate_duration.py:138-144`): Sets `self.default_on_duration = new_value`, calls `do_run_check_load_activity_and_constraints(time)`
4. **Constraint build** (`ha_model/bistate_duration.py:272-287`): For `bistate_mode_default`, creates `ConstraintItemType(target_value=self.default_on_duration * 3600.0)` -- this is correct (10h = 36000s)
5. **Constraint creation** (`ha_model/bistate_duration.py:546-562`): Creates `TimeBasedSimplePowerLoadConstraint(target_value=36000, initial_value=0)` and calls `push_live_constraint()`
6. **Push** (`home_model/load.py:1305-1373`): Processes the new constraint against existing ones

### Bug Location: `push_live_constraint()` carry-over + `update_current_metrics()` double-counting

The bug involves an interaction between two code sections:

**Section A -- Carry-over in `push_live_constraint()` (`home_model/load.py:1331-1345`):**
```python
# Carry current_value from completed constraint for same day cycle
if (
    self._last_completed_constraint is not None
    and type(self._last_completed_constraint) == type(constraint)
    and self._last_completed_constraint.current_value > constraint.current_value
    and (same end_of_constraint check)
):
    constraint.current_value = min(
        self._last_completed_constraint.current_value,
        constraint.target_value,
    )
```

This carry-over was added to preserve accumulated runtime when extending a completed target. But when the user sets a NEW absolute target, carrying over current_value creates a mismatch: the new constraint looks partially done before it starts.

**Section B -- Replacement in `push_live_constraint()` (`home_model/load.py:1347-1367`):**
```python
if c.end_of_constraint == constraint.end_of_constraint:
    if c.score(time) == constraint.score(time):
        c.carry_info_from_other_constraint(constraint)
        return False  # keeps OLD constraint, discards new target!
    else:
        self._constraints[i] = None  # remove old
        if type(c) == type(constraint) and c.current_value > constraint.current_value:
            constraint.current_value = min(c.current_value, constraint.target_value)
```

When scores differ (different target_value), the old constraint is removed and its `current_value` is carried to the new one. When scores are the same, `carry_info_from_other_constraint()` only carries power steps (NOT `target_value`), so the old target is preserved.

**Section C -- Metrics in `update_current_metrics()` (`ha_model/bistate_duration.py:116-136`):**
```python
for ct in self._constraints:
    if ct.end_of_constraint <= tomorrow_utc:
        duration_s += ct.target_value
        run_s += ct.current_value

if self._last_completed_constraint is not None:
    lcc = self._last_completed_constraint
    if (lcc for today):
        duration_s += lcc.target_value   # <-- adds old target
        run_s += lcc.current_value        # <-- adds old run time
```

If `_last_completed_constraint` is set for today (from a previous cycle or from the old constraint being completed), its `target_value` and `current_value` are ADDED to the active constraint's values, causing double-counting that produces the slider_value + already_run_hours effect.

### Exact Bug Mechanism (most likely scenario)

When the user adjusts the slider mid-run:
1. Old active constraint (target=12h, current=8h) exists in `self._constraints`
2. New constraint (target=10h, current=0) is created
3. In `push_live_constraint()`, old is replaced by new. Current_value (8h) is carried over to new constraint
4. But `_last_completed_constraint` still holds a reference with accumulated runtime from the same day cycle
5. `update_current_metrics()` sums BOTH the active constraint AND the `_last_completed_constraint`, producing the offset

The displayed target = new_constraint.target + lcc.target = slider_value + previous_accumulated.

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

- [ ] Task 1: Reproduce and diagnose exact mechanism (AC: 1)
  - [ ] 1.1: Write a failing test that sets up a pool in `bistate_mode_default` with an active constraint (target=12h, current=8h), then calls `user_set_default_on_duration(10.0)` and asserts `qs_bistate_current_duration_h == 10.0`
  - [ ] 1.2: Add debug logging in `push_live_constraint()` to trace carry-over and replacement decisions
  - [ ] 1.3: Verify whether the bug comes from Section A (lcc carry-over), Section B (active constraint replacement), Section C (metrics double-counting), or a combination

- [ ] Task 2: Fix the carry-over / metrics interaction (AC: 1, 2, 3, 4, 5, 6)
  - [ ] 2.1: When a new constraint replaces an old one in `push_live_constraint()` due to a user-initiated target change (different `requested_target_value`), either:
    - (a) Clear `_last_completed_constraint` if the new constraint supersedes it for the same day cycle, OR
    - (b) Skip the lcc contribution in `update_current_metrics()` when an active constraint for the same day already exists, OR
    - (c) Don't carry over `current_value` when the target has changed (the carry-over is for extending a completed constraint to continue, not for resetting the target)
  - [ ] 2.2: Ensure the fix correctly handles all slider adjustment scenarios (increase, decrease, set to 0)

- [ ] Task 3: Preserve legitimate carry-over (AC: 6)
  - [ ] 3.1: Verify that extending a completed constraint (same target, extending end time after it completed early) still works
  - [ ] 3.2: Test that a constraint that naturally completes and gets a new cycle constraint still tracks correctly

- [ ] Task 4: Test coverage (AC: 7, 8)
  - [ ] 4.1: Test: slider adjustment mid-run shows correct target (the failing test from 1.1, now passing)
  - [ ] 4.2: Test: slider to value below already-run hours
  - [ ] 4.3: Test: slider to 0 while pool has run
  - [ ] 4.4: Test: slider increase above already-run
  - [ ] 4.5: Test: multiple rapid slider adjustments
  - [ ] 4.6: Test: calendar-mode pool unaffected
  - [ ] 4.7: Test: constraint extension (same target) still carries over correctly
  - [ ] 4.8: Run full quality gate: `python scripts/qs/quality_gate.py`

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

### Debugging Strategy

If the exact mechanism is unclear during implementation:
1. Add `_LOGGER.debug` in `push_live_constraint()` to log: constraint target, current_value before/after carry, lcc state, replacement decision
2. Add `_LOGGER.debug` in `update_current_metrics()` to log: each constraint's contribution to duration_s and run_s, lcc contribution
3. Write a test that simulates the exact scenario and check intermediate values

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
- Root cause analysis identifies three interacting code sections (carry-over, replacement, metrics)
- Fix strategy prioritizes approach (c) -- don't carry over current_value when target has changed
- Related issues #78, #80, #68 provide historical context on the constraint carry-over evolution
