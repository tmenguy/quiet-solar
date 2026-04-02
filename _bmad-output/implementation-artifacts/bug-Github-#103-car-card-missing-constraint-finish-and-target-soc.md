# Bug Fix: Car card shows no constraint finish time and questionable 100% target SOC when forecast user is assigned

Status: in progress
issue: 103
branch: "QS_103"

## Story

As a Quiet Solar user with a forecast person assigned to a car,
I want the car card to show the person's forecasted departure time as the charge finish deadline and compute the minimum SOC needed for the trip,
so that forced charging reflects the real need rather than defaulting to 100% with no visible deadline.

## Bug Description

When a charger is forced onto a car (e.g., Twingo) and a forecast user (e.g., Arthur Menguy) is associated:

1. **No constraint finish time**: The card shows `--:--` for the Finish field, but should display the forecasted departure time (07:29) since Arthur is forecasted to leave at 7:29 for 94km.
2. **Questionable 100% target SOC**: The Target SOC shows 100% (95 km range displayed). A 94km trip need may not require charging to 100% — the solver should compute the minimum SOC needed for the forecasted trip.

**Steps to reproduce:**
1. Force a charger onto a car (e.g., wallbox 1 maison -> Twingo)
2. Associate a forecast user (Arthur Menguy) who has a forecasted departure at 07:29 for 94km
3. Observe the car card

**Expected:** Finish shows 07:29 (departure time), Target SOC reflects minimum needed for 94km trip.
**Actual:** Finish shows `--:--`, Target SOC shows 100% / 95 km.

**Related issues:** None directly; this is a new bug in the force-charge + person interaction path.

## Root Cause Analysis

### Root Cause 1: Force constraint created without `end_of_constraint`

**File:** `custom_components/quiet_solar/ha_model/charger.py`, lines 3369-3381

When `force_charge=True`, a `force_constraint` is created WITHOUT the `end_of_constraint` parameter:

```python
force_constraint = ConstraintClass(
    total_capacity_wh=self.car.car_battery_capacity,
    type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
    degraded_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
    time=time,
    load=self,
    load_param=self.car.name,
    from_user=True,
    initial_value=car_initial_value,
    target_value=target_charge,
    power_steps=self._power_steps,
    support_auto=True,
    # NO end_of_constraint → defaults to DATETIME_MAX_UTC
)
```

Without `end_of_constraint`, it defaults to `DATETIME_MAX_UTC` (constraints.py line 121). The display chain is:

1. `car.get_car_charge_time_readable_name()` (car.py:954) → gets active constraint
2. `constraint.get_readable_next_target_date_string()` (constraints.py:436) → calls `get_readable_date_string()`
3. `get_readable_date_string()` (constraints.py:40-43) → returns `"--:--"` for `DATETIME_MAX_UTC`

### Root Cause 2: Guard prevents person logic when force_constraint exists

**File:** `custom_components/quiet_solar/ha_model/charger.py`, line 3571

The guard condition:

```python
if user_timed_constraint is None and force_constraint is None:
```

prevents the entire person-based constraint logic from executing when `force_constraint` exists. This means:

- `set_next_charge_target_percent(person_min_target_charge)` (line 3638) is **never called**
- `_next_charge_target` stays at `car_default_charge` (100%) via `get_car_target_SOC()` (car.py:2112-2115)
- The car card Target SOC shows 100%

### Why person data IS available but unused

Person data is computed **before** the force_constraint block:

- Line 3323-3329: `get_best_person_next_need(time)` returns `person`, `next_usage_time`, `person_min_target_charge`, `is_person_covered`
- Line 3357: `if force_charge is True:` creates force_constraint using `target_charge` (which is still the car's default, not the person's minimum)

The person data is fully available — it's just never used when force_charge is true.

### Constraint type semantics

`CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE` means "charge at maximum power immediately." Setting `end_of_constraint` on an ASAP constraint serves as a display deadline and solver planning horizon, but doesn't change the "charge now at max" behavior. The `degraded_type=CONSTRAINT_TYPE_MANDATORY_END_TIME` means if the solver can't charge ASAP, it falls back to time-based planning toward the end_of_constraint — making the departure time even more useful as a fallback deadline.

## Acceptance Criteria

1. **AC1**: When force-charging a car with a forecasted person, `get_car_charge_time_readable_name()` returns the person's departure time (e.g., "07:29"), not `--:--`
2. **AC2**: When force-charging a car with a forecasted person (and no explicit user target override), Target SOC reflects `person_min_target_charge` (the minimum SOC for the forecasted trip), not 100%
3. **AC3**: When force-charging a car with a forecasted person AND the user explicitly set a charge target, the explicit target takes priority over the person's minimum
4. **AC4**: When force-charging a car WITHOUT a forecasted person, behavior is unchanged: finish shows `--:--`, target stays at car_default_charge
5. **AC5**: When a non-forced person constraint is created (existing path at line 3640-3654), behavior is unchanged
6. **AC6**: Existing active ASAP constraint with person info (re-entry path at line 3389-3409) preserves person-specific target correctly
7. **AC7**: 100% test coverage maintained

## Fix Plan

### Fix 1: Enrich force_constraint with person data when available

**File:** `custom_components/quiet_solar/ha_model/charger.py`

**Location:** After the force_constraint creation block (lines 3369-3381), before the push at line 3383.

When `force_charge=True` AND `person is not None` AND `next_usage_time is not None` AND `person_min_target_charge is not None`:

1. Set `force_constraint.end_of_constraint = next_usage_time` — shows departure time in the card
2. Set `force_constraint.load_info = {"person": person.name}` — marks the constraint as person-aware (enables re-entry path at line 3398-3401 to preserve person target)
3. If the user did NOT explicitly set a target (`user_target is None`): set `target_charge = person_min_target_charge` and update `force_constraint.target_value = target_charge` — uses the computed minimum SOC instead of 100%
4. Call `await self.car.set_next_charge_target_percent(target_charge)` — updates `_next_charge_target` so the UI shows the correct Target SOC

**Key design decisions:**
- The constraint type stays `MANDATORY_AS_FAST_AS_POSSIBLE` — the user forced the charge, so charging urgency is preserved
- `end_of_constraint` serves as both display deadline and degraded-type fallback
- Explicit user target (`user_target is not None`) takes priority over person minimum
- `load_info={"person": ...}` enables the re-entry path (line 3398-3401) to recognize person-originated ASAP constraints on subsequent cycles

### Fix 2: Update realized_charge_target for force + person

**File:** `custom_components/quiet_solar/ha_model/charger.py`

**Location:** Line 3430-3431, where `realized_charge_target = target_charge` is set for `force_charge is True`.

After the enrichment in Fix 1, `target_charge` may have been updated to `person_min_target_charge`. The existing code `realized_charge_target = target_charge` will naturally pick up the updated value — no additional change needed here, but verify the interaction.

## Tasks / Subtasks

- [ ] Task 1: Enrich force_constraint with person data (AC: 1, 2, 3, 4)
  - [ ] 1.1: After force_constraint creation (line 3381) and before the push (line 3383), add a block: if `person is not None and next_usage_time is not None and person_min_target_charge is not None`, set `force_constraint.end_of_constraint = next_usage_time` and `force_constraint.load_info = {"person": person.name}`
  - [ ] 1.2: In the same block, if `user_target is None`: update `target_charge = person_min_target_charge`, `force_constraint.target_value = target_charge`, and call `await self.car.set_next_charge_target_percent(target_charge)`
  - [ ] 1.3: If `user_target is not None` (explicit target), do NOT override `target_charge` — keep the user's explicit target but still set `end_of_constraint` and `load_info`

- [ ] Task 2: Add regression test — force charge + forecasted person shows departure time (AC: 1, 2)
  - [ ] 2.1: In `tests/ha_tests/test_charger.py` or new test file: set up a charger with a car, a forecasted person with `next_usage_time` and `person_min_target_charge`, force the charge
  - [ ] 2.2: Assert the active constraint has `end_of_constraint == next_usage_time` (not `DATETIME_MAX_UTC`)
  - [ ] 2.3: Assert `car.get_car_target_SOC() == person_min_target_charge` (not 100%)
  - [ ] 2.4: Assert `car.get_car_charge_time_readable_name()` returns the formatted departure time (not `--:--`)

- [ ] Task 3: Add regression test — force charge + forecasted person + explicit user target (AC: 3)
  - [ ] 3.1: Same setup as Task 2 but with `user_target` explicitly set (e.g., 80%)
  - [ ] 3.2: Assert `force_constraint.target_value == 80` (user target, not person minimum)
  - [ ] 3.3: Assert `force_constraint.end_of_constraint == next_usage_time` (departure still shown)

- [ ] Task 4: Add regression test — force charge WITHOUT person (AC: 4)
  - [ ] 4.1: Set up a charger with a car but no forecasted person, force the charge
  - [ ] 4.2: Assert constraint has `end_of_constraint == DATETIME_MAX_UTC` (unchanged behavior)
  - [ ] 4.3: Assert `car.get_car_target_SOC() == car_default_charge` (100%)

- [ ] Task 5: Verify existing tests pass — no regression (AC: 5, 6)
  - [ ] 5.1: All existing charger and car tests pass
  - [ ] 5.2: The re-entry path for person-originated ASAP constraints (line 3398-3401) still works

- [ ] Task 6: Run full quality gate (AC: 7)
  - [ ] 6.1: `python scripts/qs/quality_gate.py` — all gates green, 100% coverage

## Dev Notes

### Key Files

| File | Lines | Role |
|------|-------|------|
| `custom_components/quiet_solar/ha_model/charger.py` | 3369-3381 | `force_constraint` creation — **fix goes here** |
| `custom_components/quiet_solar/ha_model/charger.py` | 3571 | Guard that blocks person logic when force exists (no change needed, fix is upstream) |
| `custom_components/quiet_solar/ha_model/charger.py` | 3640-3654 | Person constraint creation (existing path, verify no regression) |
| `custom_components/quiet_solar/ha_model/charger.py` | 3389-3409 | Re-entry path for active ASAP constraints with person info |
| `custom_components/quiet_solar/ha_model/car.py` | 526-554 | `get_best_person_next_need()` — person data source |
| `custom_components/quiet_solar/ha_model/car.py` | 954-967 | `get_car_charge_time_readable_name()` — finish time display |
| `custom_components/quiet_solar/ha_model/car.py` | 2080-2107 | `set_next_charge_target_percent()` — SOC target setter |
| `custom_components/quiet_solar/ha_model/car.py` | 2112-2115 | `get_car_target_SOC()` — defaults to `car_default_charge` |
| `custom_components/quiet_solar/home_model/constraints.py` | 40-43 | `get_readable_date_string()` — returns `--:--` for `DATETIME_MAX_UTC` |
| `custom_components/quiet_solar/home_model/constraints.py` | 118-121 | Default `end_of_constraint = DATETIME_MAX_UTC` |
| `tests/ha_tests/test_charger.py` | all | Charger tests — new regression tests go here |
| `tests/ha_tests/test_car.py` | all | Car tests — verify no regression |

### Architecture Constraints

- **Two-layer boundary**: `home_model/` NEVER imports `homeassistant.*`. Fix stays in `ha_model/charger.py`.
- Do NOT modify constraint class internals, `get_readable_date_string()`, or `get_car_target_SOC()` default logic.
- Lazy logging: `_LOGGER.debug("msg %s", var)` — no f-strings in log calls.

### Risk Notes

- The force_constraint type stays `MANDATORY_AS_FAST_AS_POSSIBLE` — only display/planning metadata changes, not charging urgency.
- `end_of_constraint` on ASAP constraints serves as degraded fallback deadline: if solver can't maintain max power, it plans toward this time.
- `load_info={"person": ...}` must be set for the re-entry path (line 3398-3401) to recognize the constraint on subsequent cycles and preserve the person-specific target.
- If `person_min_target_charge` is very low (e.g., 15% for a short trip), the force charge will target that low SOC. This is intentional — the user can set an explicit target to override.

### Data Flow Summary

```
check_load_activity_and_constraints()
├─ get_best_person_next_need(time) → person, next_usage_time, person_min_target_charge
├─ force_charge is True?
│  ├─ YES → create force_constraint
│  │  ├─ [FIX] if person data available: enrich with end_of_constraint + load_info + target
│  │  └─ push_live_constraint()
│  └─ NO → user_timed_constraint? → person constraint? → agenda constraint?
└─ get_car_charge_time_readable_name() → reads active constraint's end_of_constraint
```

### Test Infrastructure

- Use `FakeHass`, `FakeConfigEntry` from `tests/conftest.py`
- Use `MOCK_CAR_CONFIG`, `MOCK_CHARGER_CONFIG` from `tests/ha_tests/const.py`
- Use `create_constraint()` factory from `tests/factories.py`
- Use `freezegun` for time-dependent scenarios
- asyncio_mode=auto — no `@pytest.mark.asyncio` decorator needed
