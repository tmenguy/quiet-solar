# Bug Fix: Pool target double-counts completed constraint and handle disappears after reset

issue: 64
branch: "QS_64"
Status: ready-for-dev

## Story

As a pool owner using Quiet Solar,
I want the pool card to correctly display my target duration without adding already-completed hours,
so that I can accurately set and see my pool filtering schedule.

## Bug Description

Two related issues observed after midnight when the pool has completed its daily filtering:

**Bug A — Target double-counting:** After a constraint completes (e.g., 4h of filtering done today), setting a new target via the card handle adds the completed hours to whatever the user sets. Setting 4h shows 8h, setting 1h shows 5h, setting 0h shows 4h. The completed constraint's `target_value` is summed with the new constraint's `target_value` in `update_current_metrics`.

**Bug B — Handle disappears after reset:** After pressing Reset, the target goes to 0h and the drag handle vanishes. The user cannot set a new target without first switching to another mode (e.g., winter) and back. The card conditionally renders the handle only when `targetHours > 0`.

## Root Cause Analysis

### Bug A — `pool.py:update_current_metrics` sums completed + active constraints

`custom_components/quiet_solar/ha_model/pool.py` lines 64-83:

```python
ct_to_probe = []
if self._last_completed_constraint is not None:
    ct_to_probe.append(self._last_completed_constraint)
if self._constraints:
    ct_to_probe.extend(self._constraints)
# ...
for ct in ct_to_probe:
    # time window filter...
    duration_s += ct.target_value
    run_s += ct.current_value
```

When user sets a new target, a new active constraint is created. But `_last_completed_constraint` from the previous completed cycle is still present. Both are summed, inflating the displayed target and run values.

**Fix:** When active constraints exist, exclude `_last_completed_constraint` from the metrics sum. The completed constraint should only contribute to metrics when there are no active constraints (i.e., showing "completed" state).

### Bug B — Card hides handle when target is 0

`custom_components/quiet_solar/ui/resources/qs-pool-card.js` line 218:

```javascript
const hasValidTarget = isEnabled && targetHours > 0;
```

After reset, `qs_bistate_current_duration_h` becomes 0, so `targetHours = 0`, so `hasValidTarget = false`, and no handle is rendered. Without the handle, there are no drag events, so the user is stuck.

**Fix:** Always show the handle when the device is enabled, regardless of target value. Change to `const hasValidTarget = isEnabled;`

## Acceptance Criteria

1. After a pool constraint completes and the user sets a new target via the card handle, the displayed target matches exactly what the user set (no addition of completed hours)
2. After a pool constraint completes, if no new target is set, the card still shows the completed constraint's values correctly
3. After pressing Reset, the drag handle remains visible and functional at position 0
4. User can drag from 0 to set a new target after reset without switching modes
5. All existing pool tests continue to pass
6. New tests cover both bug scenarios

## Tasks / Subtasks

- [ ] Task 1: Fix `update_current_metrics` in pool.py (AC: #1, #2)
  - [ ] When `self._constraints` is non-empty, skip `_last_completed_constraint` in the sum
  - [ ] When `self._constraints` is empty, continue using `_last_completed_constraint` (completed state display)
- [ ] Task 2: Fix handle visibility in qs-pool-card.js (AC: #3, #4)
  - [ ] Change `hasValidTarget` to `isEnabled` (remove `targetHours > 0` condition)
- [ ] Task 3: Update existing test `test_pool_update_current_metrics_with_last_completed_constraint` (AC: #5)
  - [ ] Existing test asserts double-counting behavior — update to assert only active constraint values when both exist
- [ ] Task 4: Add new tests (AC: #6)
  - [ ] Test: completed constraint only (no active) — should show completed values
  - [ ] Test: completed + active constraints — should show only active values
  - [ ] Test: after reset (no constraints, no completed) — metrics are zero

## Dev Notes

### Files to Modify

| File | Change |
|------|--------|
| `custom_components/quiet_solar/ha_model/pool.py` | `update_current_metrics`: skip `_last_completed_constraint` when active constraints exist |
| `custom_components/quiet_solar/ui/resources/qs-pool-card.js` | Line 218: `const hasValidTarget = isEnabled;` |
| `tests/test_ha_pool.py` | Update existing test, add 3 new tests |

### Architecture Constraints

- `pool.py` is in `ha_model/` layer — no HA imports needed for this change (method only uses datetime and constraint objects)
- The parent class `QSBiStateDuration.update_current_metrics` only reads `self._constraints[0]` (no `_last_completed_constraint`). The pool override intentionally adds multi-constraint summing for daily totals — the fix preserves this design, just excludes completed when active constraints exist
- JS card changes are UI-only, no backend impact
- Logging: use lazy `%s` format, no f-strings, no trailing periods

### Testing Standards

- 100% coverage required via `python scripts/qs/quality_gate.py`
- Tests use `FakeQSPool` pattern from existing `test_ha_pool.py` with `MagicMock` constraints
- Call `QSPool.update_current_metrics(pool, now)` directly on the fake pool instance

### References

- [Source: custom_components/quiet_solar/ha_model/pool.py#update_current_metrics]
- [Source: custom_components/quiet_solar/ui/resources/qs-pool-card.js#L218]
- [Source: custom_components/quiet_solar/home_model/load.py#push_live_constraint]
- [Source: custom_components/quiet_solar/home_model/load.py#ack_completed_constraint]
- [Source: tests/test_ha_pool.py#test_pool_update_current_metrics_with_last_completed_constraint]

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
