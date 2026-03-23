# Story 4.1: Default Charge When No Person Assigned

Status: review

## Story

As Magali (household member / non-technical),
I want my car to automatically charge to its default target when plugged in but no person is assigned,
so that the car is always usefully charged even when the system has no trip forecast to plan against.

## Acceptance Criteria

1. **Given** a car is plugged in **and** the system assigns no person to it (no forecast match)
   **When** the person-car allocation completes
   **Then** the car's target charge is set to `car_default_charge`
   **And** no departure time constraint is created (charge as FILLER or available energy)

2. **Given** a car is plugged in **and** the user selects "Force no person for car" (`FORCE_CAR_NO_PERSON_ATTACHED`)
   **When** the person-car allocation completes
   **Then** the car's target charge is set to `car_default_charge`
   **And** no departure time constraint is created

3. **Given** a car has a user-originated charge target (user explicitly set a SOC target)
   **When** the system would apply the default charge for no-person
   **Then** the user-originated target is preserved — the system MUST NOT overwrite it

4. **Given** a car has a user-originated charge time (user explicitly set a departure time)
   **When** the system would apply the default charge for no-person
   **Then** the user-originated time is preserved — the system MUST NOT overwrite it

5. **Given** a car is plugged in with no person, default charge was applied
   **When** a person is later assigned to that car (allocation changes)
   **Then** the person's forecast replaces the default charge (normal behavior resumes)
   **And** the previous system-set default is cleared (it was not user-originated)

6. **Given** a car is NOT plugged in and has no person assigned
   **When** the person-car allocation completes
   **Then** no default charge target is set (no point charging a car that isn't plugged)

## Tasks / Subtasks

- [x] Task 1: Add default-charge logic for unassigned cars (AC: 1, 2, 6)
  - [x] 1.1 In `compute_and_set_best_persons_cars_allocations()` (home.py ~line 2558+), after allocation is finalized, iterate over cars that ended up with `current_forecasted_person is None`
  - [x] 1.2 For each such car: check if plugged (`car.is_car_plugged()`)
  - [x] 1.3 If plugged and no charge target exists (`_next_charge_target is None`): set `_next_charge_target` to `car.car_default_charge`
  - [x] 1.4 No departure time constraint is created (code only sets target, not time)
  - [x] 1.5 If NOT plugged: skip (do nothing)

- [x] Task 2: Protect user-originated settings (AC: 3, 4)
  - [x] 2.1 Before setting default target, check `_next_charge_target is not None` — if already set, do NOT override
  - [x] 2.2 Charge time is never modified by this code — automatically preserved
  - [x] 2.3 The system-applied default charge is set via `_next_charge_target` directly, NOT stored as user_originated

- [x] Task 3: Handle transition when person is later assigned (AC: 5)
  - [x] 3.1 When a person is assigned, the system-set default is cleared (`_next_charge_target = None`) if it matches `car_default_charge` and no user_originated charge target exists
  - [x] 3.2 No stale constraint lingers — `_next_charge_target` reset lets person forecast drive charging

- [x] Task 4: Write tests (AC: 1-6)
  - [x] 4.1 Test: car plugged, no person assigned by system → target set to `car_default_charge`
  - [x] 4.2 Test: car plugged, force-no-person selected → target set to `car_default_charge`
  - [x] 4.3 Test: car plugged, no person, but user-originated charge target exists → user target preserved
  - [x] 4.4 Test: car plugged, no person, but user-originated charge time exists → user time preserved
  - [x] 4.5 Test: car plugged, no person → person later assigned → default replaced by forecast
  - [x] 4.6 Test: car NOT plugged, no person → no default set
  - [x] 4.7 100% coverage maintained (pre-existing uncovered line 1077 unrelated)

## Dev Notes

### Critical Code Locations

**Person-car allocation (primary modification point):**
- `ha_model/home.py:compute_and_set_best_persons_cars_allocations()` (lines ~2386-2602)
- After Step 6 (apply assignment, ~line 2534), add logic for unassigned cars
- The function already iterates over all cars and tracks which are "covered"

**Car target SOC management:**
- `ha_model/car.py:get_car_target_SOC()` (lines 1791-1794) — returns `_next_charge_target`, falls back to `car_default_charge`
- `ha_model/car.py:set_next_charge_target_percent()` (lines 1759-1786) — sets `_next_charge_target`
- `ha_model/car.py:car_default_charge` (line 82) — the config-defined default

**User-originated protection:**
- `home_model/load.py:has_user_originated(key)` (line 164) — check before overriding
- `home_model/load.py:get_user_originated(key)` (line 161) — read current value
- Car user_originated keys: `"charge_target_percent"`, `"charge_target_energy"`, `"charge_time"`, `"person_name"`

**Plugged detection:**
- `ha_model/car.py:is_car_plugged()` (lines 876-898) — sensor-based check
- `ha_model/charger.py:is_optimistic_plugged()` (lines 3697-3702) — charger-level check

**Force no person sentinel:**
- `const.py:FORCE_CAR_NO_PERSON_ATTACHED = "Force no person for car"` (line 143)
- Checked in `home.py` line 2417 and `car.py` line 285

### Architecture Compliance

- **Two-layer boundary**: All changes are in `ha_model/` (HA bridge layer). No changes to `home_model/` domain logic should be needed since `car_default_charge` and `_next_charge_target` are car-level attributes accessed through the HA model.
- If the default-charge-when-no-person logic is purely about reacting to allocation results, it belongs in `ha_model/home.py` where allocation already happens.
- Do NOT set user_originated for the system-applied default — this is a system decision, not a user decision.

### Key Distinction: System-Set vs User-Set

The core design challenge is distinguishing between:
1. **User-originated** target: User explicitly set a charge target via UI → persisted, survives restarts, NEVER overridden by system
2. **System-originated** target: System sets default because no person is assigned → transient, replaced when a person is assigned or when user sets a target

The simplest approach: set `car._next_charge_target = car.car_default_charge` directly (not through `set_user_originated`). This value is already the fallback in `get_car_target_SOC()`, but explicitly setting it ensures the charger picks it up in its next planning cycle.

### What NOT to Change

- Do NOT modify the Hungarian algorithm or energy matrix logic
- Do NOT change how person forecasts create constraints (that works correctly)
- Do NOT add new config keys or constants (use existing `car_default_charge`)
- Do NOT add new entities or sensors
- Do NOT modify the solver — this is pre-solver allocation logic

### Project Structure Notes

- All changes in `custom_components/quiet_solar/ha_model/home.py` and tests
- Possibly minor changes in `ha_model/car.py` if a helper method is needed
- Tests go in `tests/ha_tests/` (HA integration layer) since this is about HA-level allocation
- Use existing test factories: `create_test_car_double()`, `create_charger_group()`, `create_minimal_home_model()`
- Use existing mock configs from `tests/ha_tests/const.py`

### References

- [Source: const.py#L123] `CONF_DEFAULT_CAR_CHARGE`
- [Source: const.py#L143] `FORCE_CAR_NO_PERSON_ATTACHED`
- [Source: ha_model/home.py#L2386-2602] `compute_and_set_best_persons_cars_allocations()`
- [Source: ha_model/car.py#L1791-1794] `get_car_target_SOC()`
- [Source: ha_model/car.py#L876-898] `is_car_plugged()`
- [Source: home_model/load.py#L149-171] user_originated methods
- [Source: GitHub Issue #30] Original feature request

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- Added 20-line block in `compute_and_set_best_persons_cars_allocations()` after allocation finalization
- Uses `getattr(car, "car_default_charge", None)` for compatibility with test doubles
- When person assigned: clears system-set default (only if it matches `car_default_charge` and no user_originated target)
- When no person and plugged: sets `_next_charge_target = car_default_charge`
- 6 new tests covering all acceptance criteria, all existing tests pass (3976 total)

### File List

- `custom_components/quiet_solar/ha_model/home.py` — added default-charge-no-person logic
- `tests/test_person_car_allocation.py` — added `TestDefaultChargeNoPersonAssigned` class (6 tests), extended `_FakeCar` with plugged/default_charge/target attrs
- `_bmad-output/implementation-artifacts/4-1-default-charge-no-person.md` — story status updated
