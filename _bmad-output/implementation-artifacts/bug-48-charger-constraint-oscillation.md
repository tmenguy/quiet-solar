# Bug #48: Charger Person-Constraint / Best-Effort Oscillation

Status: review
issue: 48
branch: "QS_48"

## Story

As Magali (household member / non-technical),
I want the charger constraint system to be stable when a person-based constraint passes its deadline,
so that I do not receive dozens of alternating notifications every few seconds.

## Bug Description

When a person-based mandatory constraint (type 7 `CONSTRAINT_TYPE_MANDATORY_END_TIME`) passes its deadline without being fully met, it gets upgraded to ASAP (type 9 `CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE`) by `update_live_constraints`. The system then oscillates rapidly between the person ASAP constraint and the best-effort filler constraint, firing a notification on every cycle (~5 seconds).

### Observed Notifications (from user's phone)

```
Twingo (person:Magali Menguy): 75 % today 07:49 ASAP
Twingo: 100 % best effort
Twingo (person:Magali Menguy): 75 % today 07:49 ASAP
Twingo: 100 % best effort
Twingo (person:Magali Menguy): 75 % today 07:50 ASAP
Twingo: 100 % best effort
... repeating every few seconds
```

### Timeline Reconstruction

1. ~23:11: Person constraint created — 70% by tomorrow 07:30
2. ~07:30: 70% constraint completed (car reached 70%)
3. ~07:30: New person constraint — 75% by 07:30 (target recalculated by `get_best_person_next_need`)
4. ~07:30: Constraint immediately past deadline → `update_live_constraints` upgrades to type 9 (ASAP), extends `end_of_constraint`
5. ~07:30+: Rapid alternation between person ASAP and best-effort notifications

## Root Cause Analysis

### Primary mechanism: `realized_charge_target` not set for pre-existing ASAP constraints

In `charger.py` `check_load_activity_and_constraints()`:

**Lines 3157-3169**: When a pre-existing ASAP constraint is found in `self._constraints`, it is assigned to `force_constraint`, but **`realized_charge_target` is NEVER set**. It stays `None` from line 3079.

**Lines 3433-3434**: The best-effort section runs unconditionally when `realized_charge_target is None`:
```python
if realized_charge_target is None or (
    is_target_percent and realized_charge_target < self.car.car_default_charge
):
```

This means a **new best-effort constraint is pushed every cycle** alongside the existing ASAP constraint.

### Secondary mechanism: `target_charge` mismatch updates ASAP constraint

**Lines 3164-3168**: When the ASAP constraint's `target_value` (75%, from person need) differs from `target_charge` (from `get_car_target_SOC()`, potentially the car's `car_default_charge` like 80% or 100%), the target is silently updated:
```python
if force_constraint.target_value != target_charge:
    force_constraint.target_value = target_charge
    do_force_solve = True
    self.set_live_constraints(time, self._constraints)
```

This changes the constraint's `stable_name` → hash changes → notification fires.

### Tertiary mechanism: Constraint extension changes hash

In `load.py` `update_live_constraints()` **line 1460-1466**: Each time the mandatory ASAP constraint is pushed past its end time, `end_of_constraint` is updated and `type` is set to 9. The state hash (line 1166-1172) includes `end_of_constraint`, so **every extension triggers a new notification**.

### Notification spam via hash sensitivity

`get_active_state_hash()` (load.py:1143) includes `stable_name` + `load_param` + `end_of_constraint`. Any change to type, target, or end time changes the hash, triggering `do_probe_state_change` → `on_device_state_change` → push notification.

## Acceptance Criteria

1. **Given** a person-based constraint is past its deadline and upgraded to ASAP
   **When** `check_load_activity_and_constraints` finds the ASAP constraint
   **Then** `realized_charge_target` is set to the ASAP constraint's target value
   **And** an unnecessary best-effort constraint is NOT pushed on top of it

2. **Given** a person-based ASAP constraint exists
   **When** `check_load_activity_and_constraints` compares `target_value` to `target_charge`
   **Then** the person constraint's original target (from `person_min_target_charge`) is preserved
   **And** the constraint target is NOT silently overwritten by the car's general `target_charge`

3. **Given** multiple constraint state changes occur within a short window
   **When** the state hash changes
   **Then** the state hash is stable (ASAP constraints use a fixed "ASAP" literal instead of the extending `end_of_constraint` timestamp), preventing repeated notification triggers

4. **Given** a person-based constraint that was just completed (`_last_completed_constraint`)
   **When** `check_load_activity_and_constraints` evaluates whether to re-create a similar person constraint
   **Then** the completed constraint's original `end_of_constraint` (before any extension) is used for the duplicate check, preventing immediate re-creation

5. **Given** any fix applied
   **When** normal charging scenarios occur (person constraint created, met, new constraint)
   **Then** behavior is identical to current behavior (no regression in scheduling or charging)

## Tasks / Subtasks

- [x] Task 1: Set `realized_charge_target` when existing ASAP constraint found (AC: 1)
  - [x] 1.1 In `charger.py` lines 3157-3169, after finding the ASAP `force_constraint`, set `realized_charge_target = force_constraint.target_value`
  - [x] 1.2 This prevents the best-effort section (line 3433) from running unnecessarily
  - [x] 1.3 Verify: if `force_constraint.target_value < car_default_charge`, the best-effort section should STILL run to top up beyond the ASAP target — the existing `realized_charge_target < car_default_charge` check handles this correctly

- [x] Task 2: Preserve person constraint target when found as ASAP (AC: 2)
  - [x] 2.1 In `charger.py` lines 3164-3168, do NOT overwrite `force_constraint.target_value` with `target_charge` when the constraint has `load_info` containing a "person" key
  - [x] 2.2 Person-originated ASAP constraints should keep their person-specific target (75%) rather than being updated to the car's general target (80% or 100%)
  - [x] 2.3 The car's charge limit (`adapt_max_charge_limit`) should still use `target_charge` for the physical limit, just not overwrite the constraint's target

- [x] Task 3: Stabilize state hash for ASAP constraints (AC: 3)
  - [x] 3.1 In `load.py` `get_active_state_hash()`, use literal "ASAP" instead of `end_of_constraint` timestamp when constraint is `as_fast_as_possible`
  - [x] 3.2 This prevents hash changes from ASAP constraint extensions, eliminating notification spam

- [x] Task 4: Improve `push_live_constraint` duplicate detection for extended constraints (AC: 4)
  - [x] 4.1 In `load.py` `push_live_constraint()` line 1321-1329, also compare `initial_end_of_constraint` (not just `end_of_constraint`) against the last completed constraint
  - [x] 4.2 This prevents re-creating a person constraint that was just completed after being extended past its original deadline

- [x] Task 5: Write tests (AC: 1-5)
  - [x] 5.1 Test: ASAP constraint found → `realized_charge_target` is set → no unnecessary best-effort push
  - [x] 5.2 Test: Person ASAP constraint target preserved (not overwritten by car target)
  - [x] 5.3 Test: ASAP state hash uses "ASAP" literal (covered by test_load_model.py)
  - [x] 5.4 Test: Completed extended constraint blocks re-creation via `initial_end_of_constraint` check
  - [x] 5.5 Test: Normal charging flow (person constraint → met → best effort) still works correctly

## Dev Notes

### Critical Code Paths

| File | Lines | Function | Role |
|------|-------|----------|------|
| `ha_model/charger.py` | 3157-3169 | `check_load_activity_and_constraints` | Finds existing ASAP constraints — **missing `realized_charge_target` assignment** |
| `ha_model/charger.py` | 3433-3498 | `check_load_activity_and_constraints` | Best-effort section — runs when `realized_charge_target is None` |
| `ha_model/charger.py` | 3312-3417 | `check_load_activity_and_constraints` | Person constraint creation/update |
| `home_model/load.py` | 1401-1466 | `update_live_constraints` | Upgrades mandatory past-deadline to ASAP (type 9) |
| `home_model/load.py` | 1003-1016 | `do_probe_state_change` | Hash-based notification trigger |
| `home_model/load.py` | 1143-1175 | `get_active_state_hash` | Hash includes `stable_name` + `end_of_constraint` |
| `home_model/load.py` | 1309-1357 | `push_live_constraint` | Duplicate detection using `_last_completed_constraint` |
| `home_model/constraints.py` | 455-483 | `get_readable_name_for_load` | "ASAP" / "best effort" suffix based on type |
| `home_model/constraints.py` | 194-198 | `type` property | Returns `_type` (or `_degraded_type` if off-grid) |

### Architecture Constraints

- **Two-layer boundary**: `home_model/` NEVER imports `homeassistant.*`. Notification rate-limiting in `load.py` must use only stdlib types (datetime), not HA helpers.
- **Logging**: Use lazy `%s` format, no f-strings in log calls, no trailing periods.
- **Solver step size**: `SOLVER_STEP_S = 900` — do not change.
- **Constraint types** in `const.py`: FILLER_AUTO=1, FILLER=3, BEFORE_BATTERY_GREEN=5, MANDATORY_END_TIME=7, MANDATORY_AS_FAST_AS_POSSIBLE=9.

### Key Constants

- `CHARGER_SOC_TARGET_TOLERANCE_PERCENT = 1.2` (charger.py:172)
- `_update_step_s = timedelta(seconds=5)` (home.py:291)
- `best_duration_extension_to_push_constraint` returns `max(1200s, calculated)` — minimum 20-minute extension

### Testing Approach

Use the existing test infrastructure. Look at `tests/` for the test harness pattern. Tests should create charger + car fixtures, simulate constraint creation, deadline passing, and verify:
1. No duplicate best-effort constraint when ASAP exists
2. Person target preservation
3. Notification debouncing
4. Completed constraint duplicate blocking

### References

- [Source: ha_model/charger.py#check_load_activity_and_constraints lines 2907-3509]
- [Source: home_model/load.py#update_live_constraints lines 1359-1478]
- [Source: home_model/load.py#do_probe_state_change lines 1003-1016]
- [Source: home_model/constraints.py#get_readable_name_for_load lines 455-483]
- [Source: _bmad-output/project-context.md — 42-rule set for code conventions]

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6 (1M context)

### Completion Notes List
- Task 1: Changed `realized_charge_target` assignment to use `force_constraint.target_value` for existing ASAP constraints, preserving person-specific targets. User-forced constraints still use `target_charge` (no behavior change).
- Task 2: Added `has_person_info` guard in ASAP constraint target update — person-originated ASAP constraints keep their original target (e.g., 75%) instead of being silently overwritten by the car's general target (e.g., 80%).
- Task 3: Already completed in prior commit (a9cb4c6).
- Task 4: Extended duplicate detection in `push_live_constraint()` to also check `initial_end_of_constraint`, preventing re-creation of person constraints that were extended past their original deadline before completion.
- Task 5: 11 new tests across 2 files covering all acceptance criteria. 100% coverage achieved.

### File List
- `custom_components/quiet_solar/ha_model/charger.py` — Tasks 1 & 2: person ASAP target preservation and realized_charge_target fix
- `custom_components/quiet_solar/home_model/load.py` — Task 4: initial_end_of_constraint duplicate detection
- `tests/test_bug_48_charger_oscillation.py` — New: 10 tests for bug #48 fixes
- `tests/test_load_model.py` — New: 1 test for ASAP state hash coverage
- `_bmad-output/implementation-artifacts/bug-48-charger-constraint-oscillation.md` — Story status updated
