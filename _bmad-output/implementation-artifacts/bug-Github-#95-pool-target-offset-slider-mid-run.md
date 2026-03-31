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
7. **Metrics** (`ha_model/bistate_duration.py:576`): `update_current_metrics(time)` computes sensor values from constraints
8. **UI reads** `qs_bistate_current_duration_h` sensor (mapped as `duration_limit` in dashboard YAML) as the displayed "target"

### Exact Mechanism: UNKNOWN -- investigation required

Static code analysis traced all paths through `push_live_constraint()` and `update_current_metrics()` but could NOT conclusively reproduce the `slider + already_run` offset pattern. The naive trace suggests the correct result (target=slider_value), yet the user consistently observes the offset. This means the bug involves a runtime interaction not visible through static reading.

**What we know for certain:**
- The symptom is `qs_bistate_current_duration_h = slider_value + already_run_hours`
- The constraint is created with the correct `target_value = slider_value * 3600`
- The UI reads the sensor directly -- no client-side addition

**Hypotheses to investigate (none confirmed):**

**Hypothesis A -- Constraint coexistence (old + new both survive):**
If the old active constraint and new constraint have different `end_of_constraint` values, the replacement check at `load.py:1351` would fail and BOTH would survive in `self._constraints`. Then `update_current_metrics()` would sum both targets. This could happen if:
- The old constraint's `end_of_constraint` was modified during its lifetime
- `get_next_time_from_hours()` returns a different end time at slider-adjustment time vs original constraint creation time (e.g., if `default_on_finish_time` falls between the two times, it would jump a day)
- Timezone edge cases in end-time computation

**Hypothesis B -- Score equality causing stale target:**
If the old and new constraints have equal `score()` values (despite different `target_value`), `carry_info_from_other_constraint()` is called but only carries power steps, NOT `target_value`. The old constraint keeps its original target. This wouldn't produce the exact `slider + already_run` pattern, but could cause the target to appear wrong.

**Hypothesis C -- Double constraint evaluation:**
`user_set_default_on_duration()` calls both `do_run_check_load_activity_and_constraints(time)` and then `home.update_all_states(time)`. If `update_all_states` triggers another constraint rebuild (via `device.update_states()` or periodic cycle interleaving), the constraint could be processed twice with different intermediate states.

**Hypothesis D -- `_last_completed_constraint` from earlier same-day cycle:**
If the pool had a PREVIOUS constraint cycle today that completed (e.g., pool met an earlier shorter target, then user bumped the target up), `_last_completed_constraint` would be from today and pass the date filter in `update_current_metrics()`. Its `target_value` would be added to the active constraint's target.

### Suspect Code Sections

**Section A -- `push_live_constraint()` carry-over (`home_model/load.py:1331-1345`):**
Carries `current_value` from `_last_completed_constraint` to the new constraint. The end-date guard should prevent cross-day carry, but runtime state needs verification.

**Section B -- `push_live_constraint()` replacement (`home_model/load.py:1347-1367`):**
When replacing an existing active constraint, carries `current_value` from old to new. Also the `eq_no_current` / same-score paths may keep the OLD constraint and discard the new target.

**Section C -- `update_current_metrics()` summation (`ha_model/bistate_duration.py:116-136`):**
Sums `target_value` from ALL active constraints + `_last_completed_constraint` for today. If multiple sources contribute, the total inflates.

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
  - [ ] 1.1: Write a test that sets up a pool in `bistate_mode_default` with an active constraint (target=12h, current=8h), then calls `user_set_default_on_duration(10.0)` and inspects: (a) `self._constraints` list contents, (b) `_last_completed_constraint` state, (c) `qs_bistate_current_duration_h` value
  - [ ] 1.2: Add temporary `_LOGGER.debug` in `push_live_constraint()` to log EVERY decision: lcc state, end-date comparisons, score comparisons, carry-over triggers, replacement triggers
  - [ ] 1.3: Add temporary `_LOGGER.debug` in `update_current_metrics()` to log each source's contribution to `duration_s` and `run_s`
  - [ ] 1.4: From test output, determine which hypothesis (A-D above) explains the offset. The test MUST reproduce the exact `slider + already_run` pattern before proceeding to fix

- [ ] Task 2: Fix the identified mechanism (AC: 1, 2, 3, 4, 5, 6)
  - [ ] 2.1: Based on Task 1 findings, apply the minimal targeted fix. Possible approaches depending on root cause:
    - If coexistence (Hyp A): ensure old constraint is properly removed/replaced when target changes
    - If stale target (Hyp B): ensure `carry_info_from_other_constraint()` also updates `target_value` and `requested_target_value`
    - If double eval (Hyp C): prevent redundant constraint rebuilds during slider update
    - If lcc double-count (Hyp D): clear or skip lcc when the active constraint supersedes it for the same cycle
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

### Critical Investigation Notes

The exact bug mechanism was NOT identified through static analysis. Naive code tracing suggests the correct result (target=slider_value), yet users consistently observe `slider + already_run`. **Task 1 (reproduce + diagnose) is a hard prerequisite before any fix.** Do NOT guess at a fix -- write the diagnostic test first, confirm the offset pattern, then read intermediate state to identify the actual cause.

Key questions the diagnostic test must answer:
1. After `push_live_constraint()`, how many constraints are in `self._constraints`? (One or two?)
2. What are the `end_of_constraint` values on old vs new constraint? (Same or different?)
3. Is `_last_completed_constraint` set? If so, what are its target/current/end values?
4. In `update_current_metrics()`, what does `duration_s` equal after each source is added?

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
- Root cause NOT conclusively identified through static analysis -- four hypotheses documented
- Task 1 (diagnostic test + logging) is a hard prerequisite before any fix attempt
- Data flow from UI slider to constraint to sensor is fully mapped
- Related issues #78, #80, #68 provide historical context on the constraint carry-over evolution
