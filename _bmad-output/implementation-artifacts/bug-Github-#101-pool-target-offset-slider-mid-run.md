# Bug Fix: Pool target offset by already-run hours when slider is adjusted mid-run (regression)

Status: in-progress
issue: 101
branch: "QS_101"

## Story

As a pool owner using Quiet Solar,
I want the target hours slider to set an absolute daily target,
so that adjusting it while the pool is running shows the value I set, not an inflated number.

## Bug Description

Regression of #95. When the pool is running and the user adjusts the target hours slider (ring handle), the new target is offset by the hours already run instead of being set to the slider value directly.

**Steps to reproduce:**
1. Pool is running (ON), has run for 8h, target is 12h
2. Move the target ring handle to 10h → target sensor shows 18h (10 + 8)
3. Move handle to 4h → target shows 12h (4 + 8)
4. Move handle to 0h → target shows 8h (0 + 8)

**Key user detail:** Bug occurs specifically while the pool is ON (running).

**Expected:** `qs_bistate_current_duration_h` sensor = handle position value (absolute target).
**Actual:** `qs_bistate_current_duration_h` = handle_value + already_run_hours.

**Related issues:** #95 (original fix), #78 (daily target/actual metrics), #80 (calendar precompute refactor), #68 (carry-from-completed constraint).

## Root Cause Analysis

### Status of #95 fix

All three #95 fixes in `update_current_metrics()` are still in place:
1. Day lower bound on active constraints loop (line 132): `ct.end_of_constraint > today_utc` ✓
2. `already_absorbed` guard for lcc (lines 142-147) ✓
3. `>=` boundary for lcc (line 151) ✓

Existing regression tests (22 pool + 26 bistate) all pass. The symptom must originate from a code path not covered by the #95 tests.

### Investigation areas (ranked by likelihood)

**Area 1 — `carry_info_from_other_constraint` does not update `target_value`**

When `push_live_constraint` finds the old and new constraints have the same `score()` and same `end_of_constraint`, it takes the "carry_info" path (line 1349):
```python
c.carry_info_from_other_constraint(constraint)
return False
```

`carry_info_from_other_constraint` for `MultiStepsPowerLoadConstraint` only copies `power_steps` — it does NOT update `target_value`, `current_value`, or `_degraded_type`. The old constraint stays unchanged.

This happens when:
- Pool switches from `bistate_mode_auto` to `bistate_mode_default` (UI drag triggers `_select` then `_setNumber`)
- The restored `default_on_duration` happens to produce the same `energy_score` as the auto constraint's target (same `score()`)
- The old auto constraint retains its target, ignoring the mode change

**Impact:** The old constraint's `target_value` (auto/temperature-based) is displayed instead of the slider value. This alone doesn't produce `target = slider + current`, but combined with other constraint state it could.

**Area 2 — Two-step UI flow creates intermediate constraint state**

The UI drag handler (`qs-pool-card.js:569-571`) sends two sequential service calls:
```javascript
await this._select(e.pool_mode, 'bistate_mode_default');
await this._setNumber(e.default_on_duration, dragValue);
```

Step 1 rebuilds constraints with the OLD `default_on_duration` (restored from HA state). Step 2 rebuilds with the new value. Between these steps:
- A periodic `update_loads_constraints` cycle could run (HA event loop yields during `await`)
- The intermediate constraint might be completed by `update_live_constraints` if its `target_value` is small and `current_value >= target_value` after carry-over
- This completed constraint becomes `_last_completed_constraint`
- When step 2 runs, the new constraint is pushed, but `_last_completed_constraint` from the intermediate step may contribute its `target_value` to `duration_s` if `already_absorbed` doesn't fire (different end dates due to constraint extension in `update_live_constraints`)

**Area 3 — `push_live_constraint` replacement carry-over adds `current_value` to `target_value` via constraint chain**

When a constraint is replaced (lines 1360-1367):
```python
self._constraints[i] = None
if type(c) == type(constraint) and c.current_value > constraint.current_value:
    constraint.current_value = min(c.current_value, constraint.target_value)
```

If the lcc carry-over (lines 1331-1345) ALSO fired on the same new constraint:
```python
constraint.current_value = min(lcc.current_value, constraint.target_value)
```

The `current_value` is set twice — first from lcc, then potentially from the replaced active constraint. The second assignment is `min(c.current_value, constraint.target_value)` which uses the NEW target, so it's capped correctly. BUT if the lcc check and active-replacement check produce conflicting state, the metrics computation could be wrong.

**Area 4 — `already_absorbed` guard doesn't fire when constraint `degraded_type` differs**

The `already_absorbed` check uses `type(ct) == type(lcc)` — Python `type()`, which checks the class. Both auto and default constraints are `TimeBasedSimplePowerLoadConstraint`, so the type check passes. However, the check compares `end_of_constraint` dates:

```python
already_absorbed = any(
    type(ct) == type(lcc)
    and (ct.end_of_constraint == lcc.end_of_constraint or ct.end_of_constraint == lcc_end)
    for ct in self._constraints
    if ct.end_of_constraint > today_utc and ct.end_of_constraint <= tomorrow_utc
)
```

If `update_live_constraints` extended the old constraint's `end_of_constraint` (mandatory push at line 1476: `c.end_of_constraint = new_constraint_end`), the lcc's `end_of_constraint` would differ from the new active constraint's `end_of_constraint`. The `initial_end_of_constraint` check covers some cases, but a pushed constraint's `end_of_constraint` could be a new value that doesn't match either.

### Data Flow When Slider Changes (while pool is ON)

1. **UI** (`qs-pool-card.js:569`): `_select(pool_mode, 'bistate_mode_default')` — may change mode
2. **Select entity** → `user_set_bistate_mode('bistate_mode_default')` → `check_load_activity_and_constraints(time)`:
   - Rebuilds constraint with OLD `default_on_duration` value
   - `push_live_constraint()`: may carry-info (same score) or replace (different score)
   - `update_current_metrics(time)` — intermediate state published
3. **UI** (`qs-pool-card.js:571`): `_setNumber(default_on_duration, dragValue)` — sets new target
4. **Number entity** → `user_set_default_on_duration(dragValue)` → `check_load_activity_and_constraints(time)`:
   - Rebuilds constraint with NEW `default_on_duration`
   - `push_live_constraint()`: should replace old constraint
   - `update_current_metrics(time)` — final state published
5. **Periodic cycle**: `update_loads_constraints()` → `check_load_activity_and_constraints()` → `update_live_constraints()` — may complete constraints, set lcc

## Acceptance Criteria

1. **AC1**: When user adjusts ring handle to X hours while pool is running, `qs_bistate_current_duration_h` = X (not X + already_run)
2. **AC2**: `qs_bistate_current_on_h` (actual run hours) remains accurate and reflects real accumulated runtime for the day
3. **AC3**: Pool continues running correctly after slider adjustment (remaining hours = max(0, X - already_run))
4. **AC4**: Setting target below already-run hours shows correct target (e.g., 4h target with 8h run → target=4h, actual=8h, pool stops)
5. **AC5**: Setting target to 0h → target=0h, actual=N (whatever already ran)
6. **AC6**: Carry-over logic for constraint extension (same target, extending end time) still works
7. **AC7**: Mode transition (auto→default via drag) preserves runtime correctly
8. **AC8**: Calendar-mode pools (`bistate_mode_auto`, `bistate_mode_exact_calendar`) are unaffected
9. **AC9**: 100% test coverage maintained

## Tasks / Subtasks

- [ ] Task 1: Add debug logging for constraint state during slider adjustment (AC: all)
  - [ ] 1.1: In `push_live_constraint()`, log old constraint state (target, current, end, type, degraded_type) and new constraint state before and after carry-over/replacement
  - [ ] 1.2: In `update_current_metrics()` default path, log constraint list, lcc state, and `already_absorbed` result
  - [ ] 1.3: In `user_set_default_on_duration()`, log the before/after `default_on_duration` and constraint count

- [ ] Task 2: Write reproduction test — pool running, slider adjusted mid-run (AC: 1, 2, 3)
  - [ ] 2.1: Test pool in `bistate_mode_default`, running with 8h done / 12h target, slider changed to 10h → verify `qs_bistate_current_duration_h = 10.0` and `qs_bistate_current_on_h = 8.0`
  - [ ] 2.2: Test pool in `bistate_mode_auto`, running with 8h done / 12h target (temp-based), UI triggers mode switch to default + slider set to 10h → verify same result
  - [ ] 2.3: Test two-step flow: `user_set_bistate_mode('bistate_mode_default')` followed by `user_set_default_on_duration(10.0)` with a running pool — verify no intermediate lcc leaks

- [ ] Task 3: Write reproduction test — target below already-run hours (AC: 4, 5)
  - [ ] 3.1: Pool running, 8h done, slider set to 4h → `qs_bistate_current_duration_h = 4.0`, `qs_bistate_current_on_h` reflects actual (carried current capped at 4h)
  - [ ] 3.2: Pool running, 8h done, slider set to 0h → `qs_bistate_current_duration_h = 0.0`

- [ ] Task 4: Write test for carry_info path (same-score constraint replacement) (AC: 7)
  - [ ] 4.1: Pool in auto mode with target T, switch to default with `default_on_duration` producing same `energy_score` as T → verify the active constraint's `target_value` is updated (not stale)
  - [ ] 4.2: If carry_info path is the root cause, fix `carry_info_from_other_constraint` to also propagate `target_value` and `_degraded_type` from the new constraint

- [ ] Task 5: Write test for intermediate constraint completion during two-step flow (AC: 1, 7)
  - [ ] 5.1: Pool in auto (target=12h, current=8h), `default_on_duration` restored to 1h. Mode switch creates constraint with target=1h, current=1h (capped). Verify this constraint doesn't leak into lcc after `update_live_constraints`
  - [ ] 5.2: If this is the root cause, fix by either: (a) making the UI send a single atomic "set mode + value" service, or (b) guarding `update_current_metrics` against constraints where `current_value >= target_value` (about to be completed)

- [ ] Task 6: Fix the identified root cause (AC: 1-9)
  - [ ] 6.1: Implement fix based on findings from Tasks 2-5
  - [ ] 6.2: Verify all existing tests pass (no regressions)

- [ ] Task 7: Run full quality gate (AC: 9)
  - [ ] 7.1: `python scripts/qs/quality_gate.py`

## Dev Notes

### Key Files

| File | Lines | Role |
|------|-------|------|
| `custom_components/quiet_solar/ha_model/bistate_duration.py` | 66-158 | `update_current_metrics()` — metrics summation (contains #95 fix) |
| `custom_components/quiet_solar/ha_model/bistate_duration.py` | 160-166 | `user_set_default_on_duration()` — slider entry point |
| `custom_components/quiet_solar/ha_model/bistate_duration.py` | 168-177 | `user_set_bistate_mode()` — mode change entry point |
| `custom_components/quiet_solar/ha_model/bistate_duration.py` | 294-312 | `_build_mode_constraint_items()` default mode — constraint creation |
| `custom_components/quiet_solar/ha_model/bistate_duration.py` | 349-598 | `check_load_activity_and_constraints()` — orchestrator |
| `custom_components/quiet_solar/ha_model/pool.py` | 89-119 | Pool override of `_build_mode_constraint_items()` (auto/winter modes) |
| `custom_components/quiet_solar/home_model/load.py` | 1305-1373 | `push_live_constraint()` — carry-over and replacement logic |
| `custom_components/quiet_solar/home_model/load.py` | 1375-1500 | `update_live_constraints()` — periodic constraint completion |
| `custom_components/quiet_solar/home_model/constraints.py` | 191-192 | `carry_info_from_other_constraint()` base (no-op) |
| `custom_components/quiet_solar/home_model/constraints.py` | 672-675 | `carry_info_from_other_constraint()` MultiSteps (power_steps only) |
| `custom_components/quiet_solar/home_model/constraints.py` | 249-259 | `eq_no_current()` — constraint equality check |
| `custom_components/quiet_solar/home_model/constraints.py` | 292-314 | `score()` — constraint priority scoring |
| `custom_components/quiet_solar/ui/resources/qs-pool-card.js` | 556-587 | `onUp` drag release — two sequential service calls |
| `custom_components/quiet_solar/ui/resources/qs-pool-card.js` | 42-48 | `set hass()` — re-render guard during interaction |
| `custom_components/quiet_solar/number.py` | 130-146 | Number entity `async_set_native_value()` |
| `tests/test_ha_pool.py` | all | Pool-specific regression tests |
| `tests/test_bug_78_daily_metrics.py` | all | Daily metrics regression tests |

### Architecture Constraints

- **Two-layer boundary**: `home_model/load.py` is domain logic (no HA imports). `ha_model/bistate_duration.py` bridges HA. Fix must respect this.
- `push_live_constraint()` is in the domain layer — no HA-specific signals.
- `update_current_metrics()` is in the HA layer — can use `_last_completed_constraint` and `_constraints` from domain.
- Do NOT modify `SOLVER_STEP_S` or solver logic.
- Lazy logging: `_LOGGER.debug("msg %s", var)` — no f-strings in log calls.

### Constraint Lifecycle Reference

1. **Created**: `TimeBasedSimplePowerLoadConstraint(target_value=N, initial_value=0)`
2. **Pushed**: `push_live_constraint()` — may carry `current_value` from lcc or replaced constraint, may merge via `carry_info`
3. **Updated**: `update_live_constraints()` — increments `current_value` as pool runs, may complete constraint
4. **Completed**: When `current_value >= target_value`, `ack_completed_constraint()` sets `_last_completed_constraint`
5. **Metrics**: `update_current_metrics()` sums active constraints + lcc for today

### `push_live_constraint` Decision Tree

```
For each existing constraint c:
  eq_no_current(constraint)?
    → YES: carry_info (power_steps only), return False
    → NO: same end_of_constraint?
      → NO: continue loop
      → YES: same score?
        → YES: carry_info (power_steps only), return False
        → NO: REPLACE c with constraint, carry current_value
```

**Key insight**: The `carry_info` path does NOT update `target_value`. If the mode changes but the score happens to match, the old constraint retains its stale `target_value`.

### Test Infrastructure

- Use `MinimalTestHome` / `MinimalTestLoad` from `tests/factories.py` for constraint tests
- Use `create_constraint()` factory for creating test constraints
- Use `freezegun` for time-dependent scenarios
- Pool-specific fixtures in `tests/test_ha_pool.py`
- asyncio_mode=auto — no `@pytest.mark.asyncio` decorator needed
- Need mock for `user_set_bistate_mode` + `user_set_default_on_duration` to simulate two-step UI flow

### Related Bug Story References

- `bug-Github-#95-pool-target-offset-slider-mid-run.md` — original fix (three bugs in `update_current_metrics`)
- `bug-Github-#78-fix-bistate-daily-target-actual-hours.md` — daily target/actual metrics
- `bug-Github-#80-bistate-calendar-precompute-refactor.md` — calendar event metrics refactor
- `bug-Github-#68-carry-from-completed-constraint.md` — carry-over logic for completed constraints
- `bug-Github-#64-pool-target-double-count-and-missing-handle.md` — original double-count fix

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Completion Notes List

- Story created from GitHub issue #101 (regression of #95)
- Confirmed #95 fix is still in place: day lower bound, already_absorbed guard, >= boundary
- All 48 existing regression tests pass (22 pool, 26 bistate daily metrics)
- Root cause NOT yet pinpointed — multiple hypotheses requiring test-driven investigation
- Primary hypothesis: `carry_info_from_other_constraint` path doesn't propagate `target_value` when mode changes but constraint scores match
- Secondary hypothesis: two-step UI flow (mode change + number set) creates intermediate constraint state that leaks via lcc
- User confirms bug occurs specifically while pool is ON (running)
- User describes: moving handle sets target to handle_position + current_value (already-run hours)

### File List

- `custom_components/quiet_solar/ha_model/bistate_duration.py` — primary investigation target (`update_current_metrics`, `user_set_default_on_duration`, constraint building)
- `custom_components/quiet_solar/home_model/load.py` — `push_live_constraint()` carry-over and replacement logic
- `custom_components/quiet_solar/home_model/constraints.py` — `carry_info_from_other_constraint`, `eq_no_current`, `score`
- `custom_components/quiet_solar/ha_model/pool.py` — pool-specific constraint building for auto/winter modes
- `custom_components/quiet_solar/ui/resources/qs-pool-card.js` — two-step drag release flow
- `tests/test_ha_pool.py` — pool regression tests (to extend)
- `tests/test_bug_78_daily_metrics.py` — bistate daily metrics tests (to extend)
