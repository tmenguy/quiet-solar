# Fix: Pool forgets elapsed runtime when switching from force-on to default mode

Status: ready-for-dev
issue: 118
branch: "QS_118"

## Story

As a pool owner using Quiet Solar,
I want the system to preserve my pool's accumulated runtime when I switch from force-on mode back to default mode,
so that the solver correctly recognizes hours already filtered today and does not re-schedule work that was already done.

## Problem

When a pool has been running in force-on mode for 11 hours and the user switches to default mode (e.g., 8h/day), the system creates a new constraint with `current_value=0` instead of carrying the 11h of accumulated runtime. The constraint should be immediately satisfied (11h >= 8h) but instead the solver schedules a fresh 8h of filtering.

### Reproduction steps

1. Pool is in force-on mode, running for 11 hours (`current_value = 11h * 3600`)
2. Force-on constraint: `end_of_constraint = tomorrow_midnight`, `target_value = 25h`
3. User switches mode to "Default" with 8h/day duration
4. System creates new default constraint: `end_of_constraint = configured_finish_time` (e.g., 20:00), `target_value = 8h`
5. **Bug:** New constraint starts with `current_value = 0` — the 11h of runtime is lost
6. Solver schedules 8h of new filtering instead of recognizing the constraint is already met

### Root cause

Two cooperating failures in `push_live_constraint` (`load.py:1305`):

**Failure 1 — Active constraint replacement requires matching end times (load.py:1351)**

The replacement loop only triggers when `c.end_of_constraint == constraint.end_of_constraint`. Force-on ends at tomorrow midnight; default ends at the configured finish time (e.g., 20:00). End times differ, so the old active constraint is never replaced — both constraints coexist, and the runtime carry at line 1366 never fires.

**Failure 2 — Carry-from-completed also requires matching end times (load.py:1338)**

The carry-from-completed block added by Bug #68 requires:
```python
self._last_completed_constraint.end_of_constraint == constraint.end_of_constraint
or self._last_completed_constraint.initial_end_of_constraint == constraint.end_of_constraint
```
This condition also fails on cross-mode switching because the end times differ.

**Failure 3 — Hardcoded `initial_value=0` (bistate_duration.py:586)**

All constraints created in `check_load_activity_and_constraints` use `initial_value=0`, so there is no mechanism to seed the new constraint with accumulated runtime from outside `push_live_constraint`.

### Why existing carry logic does not help

| Carry mechanism | Location | Why it fails |
|---|---|---|
| Carry from completed constraint | `load.py:1331-1345` | End time mismatch (tomorrow midnight vs 20:00) |
| Active constraint replacement | `load.py:1351-1367` | End time mismatch (same reason) |
| `reset_initial_value_to_follow_prev_if_needed` | `constraints.py:526` | Only fires when `initial_value is None`; hardcoded to `0` |

## Fix Plan

### Fix 1: Carry `current_value` across mode switches in `push_live_constraint`

**File:** `custom_components/quiet_solar/home_model/load.py`
**Location:** After the carry-from-completed block (after line 1345), before the active constraint loop (line 1347)

Add a new block that carries `current_value` from an active constraint of the same type, even when end times differ. When the new constraint replaces the old one (mode switch), mark the old one for removal:

```python
# Carry current_value from active constraint during mode switch
# (different end times, same constraint type for same load)
for i, c in enumerate(self._constraints):
    if (
        type(c) == type(constraint)
        and c.current_value > constraint.current_value
    ):
        constraint.current_value = min(c.current_value, constraint.target_value)
        self._constraints[i] = None
        _LOGGER.info(
            "Constraint %s replacing %s on mode switch, carried current_value %.1f",
            constraint.name,
            c.name,
            constraint.current_value,
        )
        break
```

This must be placed **before** the existing active constraint loop (line 1347) to ensure the old constraint is removed and its runtime is transferred before the duplicate/replacement checks.

**Why this is safe:**
- `type(c) == type(constraint)` ensures only same-type constraints transfer runtime (e.g., `TimeBasedSimplePowerLoadConstraint` to `TimeBasedSimplePowerLoadConstraint`)
- The `min(c.current_value, constraint.target_value)` cap prevents `current_value` from exceeding the new target
- Only one constraint per load is expected (pool/bistate), so the `break` is correct
- Setting `self._constraints[i] = None` prevents the stale force-on constraint from persisting alongside the new default constraint

**Important:** After this block, clean up `None` entries before the existing loop iterates, or restructure the existing loop to skip `None` entries. The existing loop at line 1347 already tolerates `None` entries since `set_live_constraints` filters them, but the `eq_no_current` call on `None` would crash. Filter first:

```python
self._constraints = [c for c in self._constraints if c is not None]
```

### Fix 2: Relax carry-from-completed for same-load constraints

**File:** `custom_components/quiet_solar/home_model/load.py`
**Location:** Lines 1331-1345

Relax the end-time matching condition in the carry-from-completed block. For same-type, same-load constraints, the end time should not be required to match — the runtime accumulated within the same day cycle is what matters:

Change:
```python
if (
    self._last_completed_constraint is not None
    and type(self._last_completed_constraint) == type(constraint)
    and self._last_completed_constraint.current_value > constraint.current_value
    and (
        self._last_completed_constraint.end_of_constraint == constraint.end_of_constraint
        or self._last_completed_constraint.initial_end_of_constraint == constraint.end_of_constraint
    )
):
```

To:
```python
if (
    self._last_completed_constraint is not None
    and type(self._last_completed_constraint) == type(constraint)
    and self._last_completed_constraint.current_value > constraint.current_value
):
```

**Risk assessment:** The end-time check was added in Bug #68 to prevent carrying across unrelated day cycles. Without it, we could carry stale runtime from yesterday's constraint. However, `_last_completed_constraint` is reset when a new day cycle starts (via `ack_completed_constraint`), so the risk is low. If needed, add a same-day check instead:

```python
and self._last_completed_constraint.end_of_constraint.date() == constraint.end_of_constraint.date()
```

**Recommendation:** Start with Fix 1 (active constraint carry) which is the primary fix. Fix 2 is a belt-and-suspenders improvement for the case where the force-on constraint has already been completed before the mode switch.

## Acceptance Criteria

1. When a pool has been running 11h in force-on mode and user switches to default (8h/day), the constraint shows 8h/8h (capped at target) and is immediately satisfied
2. When a pool has been running 5h in force-on mode and user switches to default (8h/day), the constraint shows 5h/8h and solver schedules only 3 more hours
3. When switching from default to force-on, the existing runtime is also preserved
4. When switching to off mode, constraints are still correctly cleared (existing behavior)
5. The identity check (same target + same end time as completed) still blocks re-push correctly
6. The Bug #68 carry-from-completed still works for same-mode target extension (4h→7h)
7. All existing tests continue to pass

## Tasks / Subtasks

- [ ] Task 1: Add mode-switch carry block in `push_live_constraint` (AC: 1, 2, 3)
  - [ ] Add carry-from-active block before existing loop at line 1347
  - [ ] Add `None` cleanup after the new block
  - [ ] Add debug logging for the carry
- [ ] Task 2: Relax carry-from-completed end-time condition (AC: 1, 6)
  - [ ] Remove or relax the end_of_constraint matching condition at lines 1338-1340
  - [ ] Consider adding a same-day check as a safer alternative
- [ ] Task 3: Add tests for cross-mode runtime carry (AC: 1, 2, 3, 4, 5, 6, 7)
  - [ ] Test force-on→default with runtime exceeding new target (11h→8h)
  - [ ] Test force-on→default with runtime below new target (5h→8h)
  - [ ] Test default→force-on runtime preservation
  - [ ] Test that off mode still clears constraints
  - [ ] Test that Bug #68 same-mode extension still works
  - [ ] Test that identity check still blocks re-push

## Dev Notes

### Architecture constraints

- **Two-layer boundary**: The fix is entirely in `home_model/load.py` (domain layer). No HA imports needed.
- **Logging**: Use lazy `%s` formatting, no f-strings in log calls, no trailing periods.
- **Type checking**: Use `type(c) == type(constraint)` (not `isinstance`) — consistent with existing carry logic at line 1335 and 1366.

### Key code locations

| File | Lines | What |
|---|---|---|
| `home_model/load.py` | 1305-1373 | `push_live_constraint` — main fix location |
| `home_model/load.py` | 1331-1345 | Carry-from-completed block (Bug #68) |
| `home_model/load.py` | 1347-1367 | Active constraint replacement loop |
| `home_model/load.py` | 1106-1122 | `ack_completed_constraint` — sets `_last_completed_constraint` |
| `ha_model/bistate_duration.py` | 278-356 | `_build_mode_constraint_items` — different end times per mode |
| `ha_model/bistate_duration.py` | 577-588 | Constraint creation with `initial_value=0` |
| `home_model/constraints.py` | 82-147 | `LoadConstraint.__init__` — `initial_value` handling |
| `tests/test_load_model.py` | — | Existing constraint carry tests |

### Related prior work

- **Bug #68** (`bug-Github-#68-carry-from-completed-constraint.md`): Added carry-from-completed block for same-mode target extension. This fix extends that work to cross-mode scenarios.
- The carry-from-completed block at lines 1331-1345 was the fix for Bug #68. Our fix must not break that behavior.

### Testing patterns

- Use `freezegun` for time control
- Create `TimeBasedSimplePowerLoadConstraint` instances directly
- Use `push_live_constraint` to simulate the push flow
- Assert `current_value` after push to verify carry
- Test file: `tests/test_load_model.py`

### Project Structure Notes

- Fix is entirely within `home_model/` (domain layer) — no HA bridge changes needed
- No new files needed — modify existing `load.py` and `test_load_model.py`
- No config key changes, no UI changes, no translation changes

### References

- [Source: home_model/load.py#push_live_constraint] — main fix location
- [Source: ha_model/bistate_duration.py#_build_mode_constraint_items] — shows why end times differ per mode
- [Source: bug-Github-#68-carry-from-completed-constraint.md] — prior art for carry logic

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
