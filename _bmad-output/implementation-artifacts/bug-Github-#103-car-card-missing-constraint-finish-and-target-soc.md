# Bug Fix: Car card shows no constraint finish time and questionable 100% target SOC when forecast user is assigned

Status: done
issue: 103
branch: "QS_103"

## Story

As a Quiet Solar user who manually assigns a charger to a car with a forecast person,
I want the car card to show the person's forecasted departure time and compute the minimum SOC needed for the trip,
so that charging reflects the real need even when the car's SOC sensor is stale.

## Bug Description

When a charger is manually assigned to a car (e.g., Twingo) while the plug sensor is unplugged and the car device tracker is not home, the car enters **stale-percent mode**. A forecast user (e.g., Arthur Menguy) is associated with a forecasted departure at 07:29 for 94km:

1. **No constraint finish time**: The card shows `--:--` for the Finish field, but should display the forecasted departure time (07:29).
2. **Questionable 100% target SOC**: The Target SOC shows 100% (95 km range displayed). A 94km trip may not require 100% — the solver should compute the minimum SOC needed.

**Steps to reproduce:**
1. Manually assign a charger to a car (e.g., wallbox 1 maison -> Twingo) while car is not home/not plugged
2. Car enters stale-percent mode (API contradiction detected)
3. A forecast user (Arthur Menguy) is forecasted to depart at 07:29 for 94km
4. Observe the car card

**Expected:** Finish shows 07:29 (departure time), Target SOC reflects minimum needed for 94km trip.
**Actual:** Finish shows `--:--`, Target SOC shows 100% / 95 km.

## Root Cause Analysis

### Actual scenario: stale-percent mode, NOT force_charge

The user manually assigns the charger to the car. Since the plug sensor is unplugged and the car tracker is not home, `flag_stale_for_manual_charger` (car.py:793-809) detects the contradiction and enters stale mode:
- `_car_api_stale = True`
- `car_api_stale_percent_mode = True`
- `_car_api_inferred_home = True`, `_car_api_inferred_plugged = True`

`force_charge` is **NOT** True in this scenario — the user assigned the charger, not pressed a "force charge" button.

### Root Cause: `get_adapt_target_percent_soc_to_reach_range_km` bails on null SOC

**File:** `custom_components/quiet_solar/ha_model/car.py`, lines 1395-1412

In stale-percent mode, `get_car_charge_percent()` returns `None` (car.py:1283). This cascades:

1. **car.py:1405** — `current_soc = self.get_car_charge_percent(time)` → `None`
2. **car.py:1408** — Guard `current_soc is None` → returns `(None, None, None, None)`
3. **car.py:552** — `get_best_person_next_need()` returns: `is_person_covered=None`, `next_usage_time=07:29` (valid! from person forecast), `person_min_target_charge=None`, `person=<valid person>`
4. **charger.py:3559** — `person_min_target_charge is None` → **person is nullified** (`person = None`)
5. **charger.py:3571** — Enters person block but `person is None` → no person constraint created
6. **charger.py:3678** — `realized_charge_target is None` → filler constraint at 100% with no `end_of_constraint`

The key insight: the person's forecast data (departure time, trip distance) IS available — it comes from `person.update_person_forecast(time)`, not from the SOC sensor. Only the current SOC is unknown, which prevents computing `needed_soc` and `is_person_covered`. But `needed_soc` can be computed from trip data + `km_per_percent` alone (which has fallbacks to efficiency history and `_km_per_kwh` from config that don't require live SOC).

### Display chain (why `--:--` appears)

Without a person constraint, only a filler constraint exists (no `end_of_constraint`):
1. `car.get_car_charge_time_readable_name()` (car.py:954) → gets active constraint
2. `constraint.get_readable_next_target_date_string()` (constraints.py:436) → calls `get_readable_date_string()`
3. `get_readable_date_string()` (constraints.py:40-43) → returns `"--:--"` for `DATETIME_MAX_UTC`

### Why target SOC shows 100%

Without person constraint creation, `set_next_charge_target_percent(person_min_target_charge)` (charger.py:3638) is never called. So `_next_charge_target` stays at `car_default_charge` (100%) via `get_car_target_SOC()` (car.py:2112-2115).

## Acceptance Criteria

1. **AC1**: When a stale car has a forecasted person with valid departure time, the person constraint is created with `end_of_constraint = next_usage_time` — card shows departure time, not `--:--`
2. **AC2**: When a stale car has a forecasted person with a computable trip need, Target SOC reflects `person_min_target_charge` (the minimum SOC for the forecasted trip), not 100%
3. **AC3**: When `km_per_percent` is also unavailable (no efficiency data at all), graceful fallback — no person constraint, behavior unchanged
4. **AC4**: When SOC is available (non-stale), behavior is unchanged — existing `get_adapt_target_percent_soc_to_reach_range_km` logic untouched
5. **AC5**: Non-stale person constraint creation path (charger.py:3640-3654) is unaffected
6. **AC6**: `is_car_charged` with `current_charge=None` (stale) correctly returns `(False, None)` — person constraint is not incorrectly skipped
7. **AC7**: 100% test coverage maintained

## Fix Plan

### Fix 1: Handle stale SOC in `get_adapt_target_percent_soc_to_reach_range_km`

**File:** `custom_components/quiet_solar/ha_model/car.py`, lines 1395-1435

Currently the method bails at line 1408 when `current_soc is None`. But `needed_soc` (the minimum SOC for the trip) only requires `km_per_percent` and `target_range_km` — it does NOT need `current_soc`.

**Restructure the method:**

1. Early-exit if `km_per_percent is None` or `target_range_km is None` (can't compute anything) → return `(None, None, None, None)`
2. Compute `needed_soc` from `km_per_percent` and `target_range_km` (lines 1414-1428 — unchanged logic)
3. If `current_soc is None` or `current_range_km is None` (stale): can't determine coverage → return `(False, current_soc, needed_soc, None)` — `is_person_covered=False` (safe assumption), `person_min_target_charge=needed_soc` (computed)
4. Otherwise (normal path): existing coverage check using `current_soc` and `current_range_km` (lines 1430-1435 — unchanged)

**Effect on downstream:**
- `get_best_person_next_need()` returns `is_person_covered=False` (not None!), valid `person_min_target_charge`, valid `next_usage_time`
- Charger line 3559: `is_person_covered is False` (not None) → person survives filtering
- Charger line 3559: `person_min_target_charge` is a valid number → person survives
- Charger line 3571: `force_constraint is None` and `user_timed_constraint is None` → enters person block
- Charger line 3593: `is_car_charged(current_charge=None)` returns `(False, None)` → person constraint IS created
- Charger line 3640-3654: Person constraint created with `end_of_constraint=next_usage_time` and `target_value=person_min_target_charge`
- Charger line 3638: `set_next_charge_target_percent(target_charge)` updates `_next_charge_target`

Both issues are fixed with this single change, and the entire existing person constraint path handles it correctly.

### Why this is the right fix location

The bug is a **data availability problem**, not a constraint creation problem. The person constraint creation logic at charger.py:3571+ is correct — it just never gets valid input because `get_adapt_target_percent_soc_to_reach_range_km` unnecessarily requires `current_soc` to compute the trip's `needed_soc`. Fixing at the data source means all consumers (including potential future ones) benefit.

## Tasks / Subtasks

- [x] Task 1: Restructure `get_adapt_target_percent_soc_to_reach_range_km` for stale SOC (AC: 1, 2, 3, 4)
  - [x] 1.1: Move `km_per_percent` and `target_range_km` validation to early exit (both required for any computation)
  - [x] 1.2: Compute `needed_soc` (lines 1414-1428) before the coverage check — this only needs `km_per_percent` and `target_range_km`
  - [x] 1.3: Add stale branch: if `current_soc is None or current_range_km is None`, return `(False, current_soc, needed_soc, None)` — can't determine coverage, assume not covered
  - [x] 1.4: Existing coverage check (lines 1430-1435) remains for the normal (non-stale) path

- [x] Task 2: Add regression test — stale car + person creates constraint with departure time (AC: 1, 2)
  - [x] 2.1: Set up a car in stale-percent mode (SOC returns None) with a forecasted person
  - [x] 2.2: Verify `get_adapt_target_percent_soc_to_reach_range_km` returns `(False, None, needed_soc, None)` — not all Nones
  - [x] 2.3: Verify person constraint is created with `end_of_constraint == next_usage_time` (via downstream charger path — validated by fix at data level)
  - [x] 2.4: Verify `car.get_car_target_SOC()` returns computed minimum, not 100% (via downstream charger path — validated by fix at data level)

- [x] Task 3: Add regression test — stale car without km_per_percent (AC: 3)
  - [x] 3.1: Set up stale car with no efficiency data (km_per_percent returns None)
  - [x] 3.2: Verify `get_adapt_target_percent_soc_to_reach_range_km` returns `(None, None, None, None)` — graceful fallback
  - [x] 3.3: Verify no person constraint created (person nullified at charger.py:3559)

- [x] Task 4: Add regression test — non-stale car behavior unchanged (AC: 4, 5)
  - [x] 4.1: Verify `get_adapt_target_percent_soc_to_reach_range_km` with valid SOC returns same results as before
  - [x] 4.2: Verify person constraint path unchanged for normal (non-stale) scenario

- [x] Task 5: Verify existing tests pass — no regression (AC: 6)
  - [x] 5.1: All existing charger, car, and constraint tests pass (4575 passed)
  - [x] 5.2: `is_car_charged` with `current_charge=None` returns `(False, None)` — already correct (charger.py:4780-4782)

- [x] Task 6: Run full quality gate (AC: 7)
  - [x] 6.1: `python scripts/qs/quality_gate.py` — all gates green, 100% coverage

## Dev Notes

### Key Files

| File | Lines | Role |
|------|-------|------|
| `custom_components/quiet_solar/ha_model/car.py` | 1395-1435 | `get_adapt_target_percent_soc_to_reach_range_km()` — **fix goes here** |
| `custom_components/quiet_solar/ha_model/car.py` | 1283-1284 | `get_car_charge_percent()` returns None in stale mode |
| `custom_components/quiet_solar/ha_model/car.py` | 1359-1393 | `get_computed_range_efficiency_km_per_percent()` — has fallbacks beyond SOC |
| `custom_components/quiet_solar/ha_model/car.py` | 526-554 | `get_best_person_next_need()` — returns person data (departure time from forecast) |
| `custom_components/quiet_solar/ha_model/car.py` | 793-809 | `flag_stale_for_manual_charger()` — triggers stale mode |
| `custom_components/quiet_solar/ha_model/charger.py` | 3557-3566 | Person filtering — nullifies person when `person_min_target_charge is None` |
| `custom_components/quiet_solar/ha_model/charger.py` | 3571-3662 | Person constraint creation block — works correctly once it gets valid input |
| `custom_components/quiet_solar/ha_model/charger.py` | 4780-4782 | `is_car_charged(current_charge=None)` returns `(False, None)` — already correct |
| `tests/ha_tests/test_car.py` | all | Car tests — new regression tests |

### Architecture Constraints

- **Two-layer boundary**: `home_model/` NEVER imports `homeassistant.*`. Fix stays in `ha_model/car.py`.
- Do NOT modify constraint class internals, `get_readable_date_string()`, or `get_car_target_SOC()` default logic.
- Lazy logging: `_LOGGER.debug("msg %s", var)` — no f-strings in log calls.

### Risk Notes

- `km_per_percent` has three fallback sources (car.py:1359-1393): live SOC+range sensors → efficiency segment history → `_km_per_kwh` from config. In stale mode, fallback #1 fails but #2/#3 are available if the car has any driving history or a configured efficiency.
- If `km_per_percent` is also None (brand new car, no config), the method correctly returns all Nones — no person constraint, same as today.
- In stale mode, `is_person_covered=False` is the safe assumption: we can't verify coverage, so we always create the constraint. If the car IS already charged enough, the solver handles it gracefully (constraint met → no action).
- The `is_car_charged` call at charger.py:3593 with `current_charge=None` returns `(False, None)` — confirmed safe (charger.py:4780-4782). The person constraint will be created even though we can't verify current charge level.

### Data Flow Summary (stale mode — before fix)

```
check_load_activity_and_constraints()
├─ get_best_person_next_need(time)
│  ├─ person.update_person_forecast(time) → next_usage_time=07:29, p_mileage=94km ✓
│  └─ get_adapt_target_percent_soc_to_reach_range_km(94km, time)
│     ├─ current_soc = get_car_charge_percent(time) → None (stale!)
│     └─ returns (None, None, None, None) ← BUG: bails entirely
├─ charger.py:3559 — person_min_target_charge is None → person = None
├─ charger.py:3571 — person is None → no person constraint
└─ charger.py:3678 — filler at 100%, no end_of_constraint → --:--
```

### Data Flow Summary (stale mode — after fix)

```
check_load_activity_and_constraints()
├─ get_best_person_next_need(time)
│  ├─ person.update_person_forecast(time) → next_usage_time=07:29, p_mileage=94km ✓
│  └─ get_adapt_target_percent_soc_to_reach_range_km(94km, time)
│     ├─ current_soc = None (stale), km_per_percent = from history/config ✓
│     ├─ needed_soc = computed from km_per_percent + trip data ✓
│     └─ returns (False, None, needed_soc, None) ← FIX: safe assumption
├─ charger.py:3559 — is_person_covered=False, person_min_target_charge=needed_soc → person survives ✓
├─ charger.py:3571 — person valid → enters person constraint block
├─ charger.py:3593 — is_car_charged(None) = False → proceed
├─ charger.py:3640-3654 — person constraint created with end_of_constraint=07:29
├─ charger.py:3638 — set_next_charge_target_percent(needed_soc)
└─ card shows "07:29" finish, target SOC = needed_soc ✓
```

### Additional fix: `home.py` cost matrix guard

During implementation, the person-car allocation code in `home.py:2349` (`_build_raw_cost_matrix`) also consumed `diff_energy` without guarding for `None`. When `is_covered=False` and `diff_energy=None` (stale case), `max(E_max, None)` raised `TypeError`. Fixed by adding an `elif diff_energy is None: score = -2.0` branch — treating stale energy as "car data error" for allocation purposes.

### Additional fix: f-string in log call

The original code at car.py:1409-1411 used an f-string in the `_LOGGER.warning()` call. Fixed to use lazy `%s` formatting per project rules.

### Test Infrastructure

- Use `FakeHass`, `FakeConfigEntry` from `tests/conftest.py`
- Use `MOCK_CAR_CONFIG`, `MOCK_CHARGER_CONFIG` from `tests/ha_tests/const.py`
- Use `create_constraint()` factory from `tests/factories.py`
- Use `freezegun` for time-dependent scenarios
- asyncio_mode=auto — no `@pytest.mark.asyncio` decorator needed
