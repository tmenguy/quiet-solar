# Bug Fix: Charger power oscillation exceeds inverter limit and person constraint forces grid import

issue: 66
branch: "QS_66"
Status: draft

## Story

As a Quiet Solar user with multiple EV chargers,
I want the charging system to respect the inverter AC output limit and avoid rapid power oscillation,
so that I don't import from the grid when solar+battery should suffice and my charger hardware isn't damaged by rapid cycling.

## Bug Description

Three related issues observed on 2026-03-29 morning:

**Bug A -- Person constraint with past deadline forces max grid import (10:55-11:20)**

After HA restart at 10:55, the person constraint for Magali (Zoe, wallbox 1) predicted departure at 09:00 UTC (11:00 CEST) which was already past. The system set `mandatory:True` + `use_available_only:False`, forcing `auto_consign` at 22,080W (32A x 3 phases). With inverter capped at 12kW and solar at ~5.5kW, the remaining ~10kW was pulled from the grid. `full_available_home_power` reached -10,499W.

**Bug B -- Solver consign oscillation between ~4.4kW and 22kW every ~7 seconds (12:58-13:10)**

The wallbox 3 portail (Twingo) received alternating `auto_green_consign` commands: ~4,417W then 22,080W, repeating every ~7 seconds. During the 22kW phases, the budgeting algorithm raised amps from 8A to 11-12A. At 12A x 3 phases (~8.3kW) plus house loads, the combined draw exceeded the 12kW inverter AC output limit. The dynamic group check at 13:08:30 confirmed `phases_amps [12.0, 12.2, 12.0]`.

**Bug C -- Dynamic group power drops to 0W without guard (12:40-13:10)**

The wallbox 3 portail received `auto_green 0.0W` (stop) commands with no minimum hold time. Drops to 0W occurred as frequently as every 7 seconds, with longer gaps of 6-14 minutes also appearing. The 20-minute on-to-off guard (`TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_ON_TO_OFF_S`) was bypassed.

## Root Cause Analysis

### Bug A -- Missing past-deadline guard for person constraints

**File:** `charger.py` lines 3456-3465 and 3539-3553

`is_person_covered` (computed in `car.py:1063-1103`) is purely a battery-level check -- it does NOT consider whether `next_usage_time` is in the past. When `is_person_covered` is False, a `CONSTRAINT_TYPE_MANDATORY_END_TIME` constraint is created with the past deadline as `end_of_constraint`.

Since `is_mandatory` is True (type 7), the solver sets `do_use_available_power_only=False` at `solver.py:715`, allowing grid draw. The constraint window collapses to "right now" (slot 0), meaning "charge at maximum power immediately from any source."

**Key asymmetry:** Calendar/agenda events have a past-deadline guard at `charger.py:3398-3403`:
```python
if start_time is not None and time > start_time:
    start_time = None  # passed it, skip
```
No equivalent guard exists for person-based constraints.

### Bug B -- Budgeting algorithm full-range oscillation (no damping)

**File:** `charger.py` lines 1338-1456 (budgeting loop) and 662-669 (power extraction)

The feedback loop:

1. **Cycle A (charger at ~4.4kW):** Available power is high (~18kW surplus). `initial_power_budget` is large and positive. For `CMD_AUTO_GREEN_CONSIGN`, `stop_on_first_change=False` (line 1348-1351), so the loop ramps **all the way to 22kW** in one cycle.

2. **Cycle B (charger at ~22kW):** The charger's own draw collapses available power to negative. `initial_power_budget` becomes ~-17kW. The loop ramps **all the way back down** to ~4.4kW.

Three compounding factors:
- **No max step-size limit** on budget changes per cycle (lines 1361-1444)
- **Feedback includes charger's own draw** -- `home_available_power = grid_consumption + battery_charge_clamped` (home.py:1658) reflects the charger's consumption, creating unity-gain negative feedback
- **Pessimistic estimator** -- `min(last_half_mean, all_mean, last_half_median, all_median)` at line 662-669 amplifies transient states

The `CHARGER_ADAPTATION_WINDOW_S = 45s` guard at line 692 only gates new decisions, but `remaining_budget_to_apply` (line 682-688) executes deferred increases on every `dyn_handle` cycle (~14s), creating visible oscillation faster than the 45s window.

### Bug C -- Shaving methods bypass on-to-off time guard

**File:** `charger.py` lines 1643-1673 and 1676-1745

The 20-minute guard is properly checked in `get_stable_dynamic_charge_status` (line 2374-2392): when `can_change_state` is False, 0 is NOT added to `possible_amps`.

However, two shaving paths bypass this:
- `_shave_current_budgets` (line 1643): runs with `allow_state_change=True` on second iteration, forcing `budgeted_amp=0` via `can_change_budget` without checking `is_ok_to_set`
- `_shave_mandatory_budgets` (line 1676): inserts 0 at the front of `possible_amps` (line 1717-1734) regardless of time guards

`apply_budgets` (line 1929-1933) then applies `budgeted_amp=0` without re-checking the guard (by design, per the comment on line 1930-1932).

## Fix Plan

### Fix A: Add past-deadline guard for person constraints

**Where:** `charger.py` around line 3456-3465

Add a check analogous to the agenda guard at line 3398:

```python
elif is_person_covered is False:
    if next_usage_time is not None and next_usage_time <= time:
        _LOGGER.info(
            "check_load_activity_and_constraints: plugged car %s person %s "
            "predicted departure %s already past (now: %s), skipping person constraint",
            self.car.name, person.name, next_usage_time, time
        )
        person = None
    else:
        _LOGGER.warning(...)  # existing log
```

This prevents a past deadline from creating a mandatory "charge now from grid" emergency.

### Fix B: Add damping / max step-size to budgeting algorithm

**Where:** `charger.py` lines 1338-1456

Two complementary changes:

**B1. Limit maximum amp change per cycle.** After computing the target budget, clamp the actual change to at most N amps per cycle (e.g., 2-3A). This prevents full-range jumps:

```python
# After determining next_possible_budgeted_amp in the while loop
max_amp_change_per_cycle = 3  # configurable
if abs(next_possible_budgeted_amp - cs.budgeted_amp) > max_amp_change_per_cycle:
    break  # stop ramping, apply partial change
```

**B2. Subtract charger's current draw from available power before budgeting.** In the budget computation (around line 1161), account for the charger's own current consumption to avoid the feedback loop:

```python
# Compute budget relative to charger's current draw, not total available
charger_current_draw = sum(cs.get_current_power() for cs in actionable_chargers)
adjusted_available = full_available_home_power + charger_current_draw
initial_power_budget = adjusted_available - diff_power_budget
```

This way, when the charger is drawing 22kW and available power is -10kW, the adjusted available becomes -10+22 = 12kW, which is the actual solar+battery capacity -- the correct target.

### Fix C: Enforce time guard in shaving methods

**Where:** `charger.py` lines 1643-1673 and 1676-1745

**C1. In `_shave_current_budgets`:** Before allowing `budgeted_amp=0`, check the time guard:

```python
# Before setting cs.budgeted_amp = 0 in the shaving loop
if next_possible_budgeted_amp == 0:
    if not cs.charger._expected_charge_state.is_ok_to_set(
        time, TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_ON_TO_OFF_S
    ):
        continue  # skip this charger, try reducing another
```

**C2. In `_shave_mandatory_budgets`:** Same guard before inserting 0 into `possible_amps`.

**Important:** The shaving methods exist to protect against exceeding the dynamic group amp limit. If the guard prevents stopping a charger AND the group limit is still exceeded, the method should reduce amps to `min_charge` instead of 0, maintaining the guard while still reducing load.

## Acceptance Criteria

1. After HA restart, if person constraint departure time is already past, the charger does NOT force maximum grid import -- it either charges from available solar only or skips the constraint
2. The charger consign does not oscillate between min and max power on consecutive cycles -- amps change by at most N steps per budget cycle
3. The combined charger + house power does not exceed the configured `max_inverter_dc_to_ac_power` (12kW) during normal green charging
4. The dynamic group charger power does not drop to 0W unless the 20-minute on-to-off guard has elapsed
5. When the shaving method needs to reduce power but the time guard blocks a full stop, the charger is reduced to `min_charge` instead
6. All existing tests continue to pass
7. New tests cover: past-deadline person constraint, budget damping, shaving with guard enforcement

## Tasks / Subtasks

- [ ] Task 1: Fix past-deadline person constraint guard (AC: #1)
  - [ ] Add `next_usage_time <= time` check at charger.py:3456-3465
  - [ ] Log the skip at info level
  - [ ] Add test: person constraint with past deadline is skipped
  - [ ] Add test: person constraint with future deadline still works
- [ ] Task 2: Add damping to budgeting algorithm (AC: #2, #3)
  - [ ] Introduce max amp change per cycle constant
  - [ ] Clamp amp changes in the budgeting while loop
  - [ ] Subtract charger's current draw when computing available power budget
  - [ ] Add test: budget doesn't jump from min to max in one cycle
  - [ ] Add test: budget correctly accounts for charger's own draw
- [ ] Task 3: Enforce time guard in shaving methods (AC: #4, #5)
  - [ ] Add `is_ok_to_set` check in `_shave_current_budgets` before setting amp to 0
  - [ ] Add `is_ok_to_set` check in `_shave_mandatory_budgets` before inserting 0
  - [ ] Fall back to `min_charge` when guard blocks full stop
  - [ ] Add test: shaving respects on-to-off guard
  - [ ] Add test: shaving falls back to min_charge when guard active

## Technical Notes

### Key Constants
- `SOLVER_STEP_S = 900` (15 min) -- do not touch
- `CHARGER_ADAPTATION_WINDOW_S = 45` seconds
- `CHARGER_STATE_REFRESH_INTERVAL_S = 14` seconds
- `TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_ON_TO_OFF_S = 60 * 20` (20 min)
- `TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_OFF_TO_ON_S = 60 * 10` (10 min)

### Key Files
- `custom_components/quiet_solar/ha_model/charger.py` -- primary fix target (all 3 bugs)
- `custom_components/quiet_solar/ha_model/car.py` -- `is_person_covered` computation
- `custom_components/quiet_solar/ha_model/person.py` -- departure prediction
- `custom_components/quiet_solar/home_model/solver.py` -- constraint mandatory/available_only logic
- `custom_components/quiet_solar/home_model/constraints.py` -- constraint types and repartition
- `custom_components/quiet_solar/ha_model/dynamic_group.py` -- group amp accounting

### Risk Assessment
- Fix A is low risk -- adds a guard that mirrors the existing agenda guard pattern
- Fix B is medium risk -- changing the budgeting feedback loop affects all charger scenarios; thorough testing of green-only, green-consign, and price modes needed
- Fix C is medium risk -- the shaving methods are safety mechanisms for amp limits; the fallback to min_charge must be verified to not cause sustained overcurrent
