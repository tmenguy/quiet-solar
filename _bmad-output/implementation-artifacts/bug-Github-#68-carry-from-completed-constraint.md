# Bug Fix: Extending completed pool constraint loses accumulated runtime

issue: 68
branch: "QS_68"
Status: ready

## Story

As a pool owner using Quiet Solar,
I want to be able to extend a completed pool target (e.g., from 4h to 7h) without losing the hours already filtered,
so that the system correctly shows my accumulated runtime and only schedules the remaining hours.

## Bug Description

After a pool constraint completes (e.g., 4h target fully met with 4h of actual filtering), the user drags the card handle to extend the target to 7h. The display drops to 0h/7h instead of showing 4h/7h. The 4h of physical pump runtime vanishes from the UI and the solver schedules 7h of new filtering instead of the 3h delta.

### Reproduction steps

1. Pool completes 4h/4h constraint
2. `update_live_constraints` marks it met, calls `ack_completed_constraint`, stores in `_last_completed_constraint`. `_constraints` becomes empty
3. `update_current_metrics`: no active constraints -> uses `_last_completed_constraint` -> displays 4h/4h (correct)
4. User drags handle to 7h -> card sends `bistate_mode_default` + `default_on_duration=7`
5. `check_load_activity_and_constraints` builds a new `TimeBasedSimplePowerLoadConstraint` with `target_value=7*3600` and `initial_value=0`
6. `push_live_constraint` is called: identity check sees different target (4h != 7h) -> does not block; active constraint loop finds empty `_constraints` -> no carry; new constraint appended with `current_value=0`
7. `update_current_metrics`: active constraint exists -> ignores `_last_completed_constraint` -> shows 0h/7h. **Runtime lost.**

## Root Cause

`push_live_constraint` (`load.py:1305`) carries `current_value` from active `_constraints` being replaced (L1349-1351), but **never carries from `_last_completed_constraint`**. When the only source of accumulated runtime is the completed constraint, a new constraint starts from zero.

## Fix Plan

### 1. Insert carry-from-completed block in `push_live_constraint`

**File:** `custom_components/quiet_solar/home_model/load.py` ~L1329

Insert between the identity check `return False` (L1329) and the active constraint loop (L1331):

```python
            # Carry current_value from completed constraint for same day cycle
            # so that extending a completed target preserves accumulated runtime
            if (
                self._last_completed_constraint is not None
                and type(self._last_completed_constraint) == type(constraint)
                and self._last_completed_constraint.current_value > constraint.current_value
                and (
                    self._last_completed_constraint.end_of_constraint == constraint.end_of_constraint
                    or self._last_completed_constraint.initial_end_of_constraint
                    == constraint.end_of_constraint
                )
            ):
                constraint.current_value = min(
                    self._last_completed_constraint.current_value,
                    constraint.target_value,
                )
```

No other production lines change.

### 2. Fix stale test assertions in `tests/test_load_model.py`

Two tests added by the QS_64 branch assert behavior of reverted `load.py` changes:

1. **`test_push_live_constraint_does_not_carry_current_value_on_target_change`** -- asserts `current_value == 0.0` but the original code carries it (`80.0`). Carry between active constraints is correct (physical runtime happened). Update assertion to `80.0`.
2. **`test_disable_device_preserves_last_completed_constraint`** -- asserts completed constraint is preserved on disable, but `reset()` correctly wipes everything. Update assertion to `None`.

### 3. Add new test for carry-from-completed scenario

Add `test_push_live_constraint_carries_from_completed_constraint` that:
- Creates a load, completes a constraint (4h), stores in `_last_completed_constraint`
- Pushes a new constraint with higher target (7h) and same end date
- Asserts `current_value` is carried (4h), not zero
- Also tests the cap case: new target < completed value -> `current_value = min(completed, target)`

## Acceptance Criteria

1. After a pool constraint completes (4h/4h) and user extends target to 7h, display shows 4h/7h and solver schedules only 3 more hours
2. After a pool constraint completes (4h/4h) and user reduces target to 3h, carry gives min(4h,3h)=3h -> constraint is met -> display shows 3h/3h
3. After a pool constraint completes (4h/4h) and user sets same target 4h, identity check blocks re-push -> display stays 4h/4h
4. After reset (`_last_completed_constraint` is None), new constraint starts fresh with current_value=0
5. Existing active-constraint carry (L1349-1351) still works for mid-cycle target changes
6. All existing tests continue to pass

## Files to Modify

| File | Change |
|------|--------|
| `custom_components/quiet_solar/home_model/load.py` | Insert carry-from-completed block in `push_live_constraint` after L1329 |
| `tests/test_load_model.py` | Fix 2 stale test assertions, add carry-from-completed test |

## Dev Notes

- The carry block uses `type()` equality (not `isinstance`) to match only the exact constraint type, consistent with the existing active-constraint carry at L1350
- The `end_of_constraint` / `initial_end_of_constraint` check ensures we only carry within the same day cycle, not across unrelated constraints
- The `min(current_value, target_value)` cap prevents current_value from exceeding the new (possibly lower) target
