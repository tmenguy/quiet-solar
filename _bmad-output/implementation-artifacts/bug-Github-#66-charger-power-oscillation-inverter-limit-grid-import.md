# Bug Fix: Charger power oscillation exceeds inverter limit and person constraint forces grid import

issue: 66
branch: "QS_66"
Status: draft

## Story

As a Quiet Solar user with EV chargers,
I want the charging system to respect the inverter AC output limit and avoid rapid power oscillation,
so that green charging never imports from the grid and my charger hardware isn't damaged by rapid cycling.

## Bug Description

Two issues observed on 2026-03-29:

**Issue 1: Zoe at ~11:00 -- 19kW from grid**

After HA restart at 10:55, the Zoe was charging at 32A/3-phase (19,148W) on wallbox 1 maison with `CMD_AUTO_FROM_CONSIGN`. Person constraint for Magali: next usage 09:00 UTC (11:00 local), car at 89% vs 100% target, `is_person_covered=False`. The solver assigned `CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE` resulting in `CMD_AUTO_FROM_CONSIGN` with max power. With `CMD_AUTO_FROM_CONSIGN`, `possible_amps = [32]` (only max), so the dynamic budgeting cannot reduce the charging at all. Grid import was -8,813W.

**Verdict**: Partially by design -- person constraints DO force fast charging from grid. However, `CMD_AUTO_FROM_CONSIGN` locking `possible_amps` to only `[max]` is too aggressive. The dynamic budget should still have room to reduce amps when the consign deadline has already passed (degraded mode).

**Issue 2: Twingo 13:04-13:10 -- oscillation exceeding 12kW inverter limit**

| Time     | Action     | home_load   | Comment                                                  |
| -------- | ---------- | ----------- | -------------------------------------------------------- |
| 13:03:21 | 8A -> 9A   | -           | budget 1982W                                             |
| 13:04:17 | 9A -> 11A  | 10,580W     | Production cap fires, allows 1419W budget, jumps 2 steps |
| 13:05:20 | 11A -> 9A  | **12,836W** | Over 12kW! Decrease triggered                            |
| 13:06:30 | 9A -> 8A   | **12,161W** | Still over 12kW                                          |
| 13:07:26 | 8A -> 11A  | 9,021W      | Low due to transition, jumps 3 steps!                    |
| 13:08:16 | 11A -> 12A | -           | phantom_surplus=7147W, still allows increase             |
| 13:09:12 | 12A -> 8A  | **14,901W** | Way over 12kW!                                           |
| 13:10:01 | 8A -> 9A   | -           | Cycle continues                                          |

**Issue 3: Power drops to 0 during transitions**

At 13:10:01: `dampening simple case wallbox 3 portail/Twingo 22.223W for 8A #phases3`. The wallbox temporarily reports near-zero power during current changes. This is **normal wallbox hardware behavior**. The dampening code correctly detects and ignores it. But the `home_load_power_value` picks up the transient, feeding into Root Cause C.

## Root Cause Analysis

### Root Cause A: Battery discharge inflates green budget

**File:** `charger.py` lines 1177-1188

When battery is discharging (`battery_asked_charge < 0`) and charger is `CMD_AUTO_GREEN_CONSIGN` + before battery:
```python
initial_power_budget = full_available_home_power - battery_asked_charge - diff_power_budget
```

At 13:04:17: `full_available_home_power=0`, `battery_asked_charge=-3787W`, so budget = `0 - (-3787) = 3787W`. The code assumes the battery discharge can be redirected to the charger, but the inverter is already at capacity. The extra budget results in **grid import**, not battery power.

### Root Cause B: Multi-step amp jumps amplify oscillation

**File:** `charger.py` lines 1348-1351

```python
if do_reset_allocation or cs.command.is_like_one_of_cmds(
    [CMD_AUTO_GREEN_CONSIGN, CMD_AUTO_PRICE, CMD_AUTO_FROM_CONSIGN]
):
    stop_on_first_change = False
```

`stop_on_first_change=False` allows the algorithm to jump 8A->11A (3 steps) in one pass. This makes the system overshoot massively, triggering the oscillation cycle.

### Root Cause C: Production cap uses stale home_load during transitions

**File:** `charger.py` lines 1248-1301

The `home_load_power_value` is measured over the past `CHARGER_ADAPTATION_WINDOW_S` (45s). When the charger just dropped power (transition), `home_load` appears low (e.g., 9,021W at 13:07:26), creating artificial headroom under the 12kW cap. The `phantom_surplus` is already computed and subtracted from the budget (line 1201) but is NOT used to correct `home_load_power_value` before the production cap check.

### Root Cause D: No cooldown between amp changes

After changing amps (e.g., at 13:07:26), the very next `dyn_handle` pass (7s later) can change amps again. The `CHARGER_ADAPTATION_WINDOW_S` check at line 690-692 only guards the first pass after a state change.

### Root Cause E (highest priority): Solver does not cap green consign by inverter power limit

**File:** `home.py` lines 1143-1152

```python
def get_home_max_static_phase_amps(self) -> int:
    static_amp = self.dyn_group_max_phase_current_conf
    if not self.is_off_grid():
        return static_amp          # <-- BUG: returns subscription limit (52A)
    if self.solar_plant:
        static_amp = min(static_amp, self.solar_plant.solar_max_phase_amps)
    return static_amp
```

`solar_max_phase_amps` (the inverter's per-phase current limit) is only applied when off-grid. When on-grid, the function returns the subscription limit (52A for a 36kVA contract). This means the solver's production amp budget allows 52A per phase, so the Twingo's full 32A power step (22,080W) passes filtering -- even though the inverter can only output 12kW (~17.4A per phase at 230V 3-phase).

The solver's `_available_power` clamping limits total energy across all slots to the inverter cap, but `adapt_repartition` runs multiple iterations that each bump the per-slot power consign one step higher. Since the energy budget is large enough, the consign climbs all the way to 22,080W.

This is called from `dyn_group_max_production_phase_current_for_budget` at line 1244. The consumption path (`dyn_group_max_phase_current_for_budget`) already guards with `if self.is_off_grid()`, so the change is safe.

## Fix Plan

### Fix E (highest priority): Cap solver production amp budget by inverter limit

**Where:** `home.py` lines 1143-1152

1. Rename `get_home_max_static_phase_amps` -> `_get_home_max_production_phase_amps_for_budget` (private, production-only scope)
2. Remove the `if not self.is_off_grid(): return` early exit so `solar_max_phase_amps` is always applied
3. Update the single call site at line 1244

```python
def _get_home_max_production_phase_amps_for_budget(self) -> int:
    static_amp = self.dyn_group_max_phase_current_conf
    if self.solar_plant:
        static_amp = min(static_amp, self.solar_plant.solar_max_phase_amps)
    return static_amp
```

This caps the production amp budget at ~17.4A per phase (for 12kW inverter), limiting the highest power step the solver can assign for green charging. The private name and explicit "production/budget" scope prevent future misuse.

### Fix A: Cap battery discharge budget by inverter headroom

**Where:** `charger.py` around line 1248 (production cap check)

After computing `new_home_power_consumption`, cap the battery-discharge portion of the budget by actual inverter headroom:
```python
if home_max_available_production_power is not None and home_load_power_value is not None:
    inverter_headroom = max(0, home_max_available_production_power - home_load_power_value)
    initial_power_budget = min(initial_power_budget, inverter_headroom)
```

Note: `home_max_available_production_power` and `home_load_power_value` are computed after the battery block (lines 1208+), so this cap must be applied at the production cap check point (line 1248), not at line 1187.

### Fix B: Limit green consign to 1-step increase per cycle

**Where:** `charger.py` lines 1348-1351

Remove `CMD_AUTO_GREEN_CONSIGN` from the `stop_on_first_change=False` block:

```python
if do_reset_allocation or cs.command.is_like_one_of_cmds(
    [CMD_AUTO_PRICE, CMD_AUTO_FROM_CONSIGN]
):
    stop_on_first_change = False
```

This keeps `stop_on_first_change=True` for `CMD_AUTO_GREEN_CONSIGN` increases, preventing multi-step jumps. Only `CMD_AUTO_FROM_CONSIGN` and `CMD_AUTO_PRICE` retain multi-step (they have mandatory deadlines).

### Fix C: Add phantom surplus to home_load before production cap

**Where:** `charger.py` line 1248

Before the production cap check, correct `home_load_power_value` for transient charger power dips:
```python
adjusted_home_load = home_load_power_value + phantom_surplus
```
Then use `adjusted_home_load` in place of `home_load_power_value` in the production cap calculation. This prevents transient wallbox dips from creating artificial headroom.

### Fix D: Per-charger amp-change cooldown

**Where:** `charger.py` -- `QSChargerGeneric` class + `dyn_handle`

1. Add `_last_amp_change_time` attribute to `QSChargerGeneric`
2. Set it when amps are changed in `_ensure_correct_state`
3. In `dyn_handle`, skip budgeting for a charger if its last amp change was less than `CHARGER_ADAPTATION_WINDOW_S` ago

## Acceptance Criteria

1. The solver never assigns a green consign above the inverter's physical production limit (~12kW)
2. Battery discharge budget is capped by actual inverter headroom -- no grid import when budget says "use battery discharge"
3. The charger increases by at most 1 amp step per budget cycle in green consign mode
4. Transient wallbox power dips do not create artificial headroom in the production cap calculation
5. After an amp change, the same charger is not re-budgeted until `CHARGER_ADAPTATION_WINDOW_S` elapses
6. All existing tests continue to pass
7. New tests cover: production cap with inverter limit, battery discharge headroom cap, single-step green increase, phantom surplus correction, amp-change cooldown

## Tasks / Subtasks

- [ ] Task 1: Fix E -- Cap solver production amp budget by inverter limit (AC: #1)
  - [ ] Rename `get_home_max_static_phase_amps` -> `_get_home_max_production_phase_amps_for_budget`
  - [ ] Remove off-grid guard so `solar_max_phase_amps` always applies
  - [ ] Update call site at line 1244
  - [ ] Add test: production amp budget reflects inverter limit when on-grid
- [ ] Task 2: Fix A -- Cap battery discharge budget by inverter headroom (AC: #2)
  - [ ] Add inverter headroom cap at production cap check point (~line 1248)
  - [ ] Add test: battery discharge budget does not exceed inverter headroom
- [ ] Task 3: Fix B -- Limit green consign to 1-step increase (AC: #3)
  - [ ] Remove `CMD_AUTO_GREEN_CONSIGN` from `stop_on_first_change=False` block
  - [ ] Add test: green consign increases by at most 1 step per cycle
- [ ] Task 4: Fix C -- Add phantom surplus to home_load before production cap (AC: #4)
  - [ ] Add `phantom_surplus` to `home_load_power_value` before production cap check
  - [ ] Add test: transient dip does not create artificial headroom
- [ ] Task 5: Fix D -- Per-charger amp-change cooldown (AC: #5)
  - [ ] Add `_last_amp_change_time` to `QSChargerGeneric`
  - [ ] Set timestamp on amp change in `_ensure_correct_state`
  - [ ] Skip budgeting in `dyn_handle` if cooldown not elapsed
  - [ ] Add test: charger is skipped during cooldown period

## Technical Notes

### Key Constants
- `SOLVER_STEP_S = 900` (15 min) -- do not touch
- `CHARGER_ADAPTATION_WINDOW_S = 45` seconds
- `CHARGER_STATE_REFRESH_INTERVAL_S = 14` seconds
- `TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_ON_TO_OFF_S = 60 * 20` (20 min)
- `TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_OFF_TO_ON_S = 60 * 10` (10 min)

### Key Files
- `custom_components/quiet_solar/ha_model/home.py` -- Fix E (solver production amp budget)
- `custom_components/quiet_solar/ha_model/charger.py` -- Fixes A, B, C, D (dynamic budgeting)

### Deferred: Person constraint `CMD_AUTO_FROM_CONSIGN` locking
Issue 1 (Zoe at 19kW) is partially by design -- person constraints force fast charging. However, `CMD_AUTO_FROM_CONSIGN` locking `possible_amps` to only `[max]` prevents any dynamic budget reduction even when the deadline has passed. This is a separate issue to address after the oscillation fixes.

### Risk Assessment
- Fix E is low risk, high impact -- single method rename + guard removal, only affects production budget path
- Fix B is low risk -- removes one command type from a list, keeps green-only behavior (already 1-step)
- Fix C is low risk -- uses already-computed `phantom_surplus` value
- Fix A is medium risk -- needs careful placement after production cap values are computed
- Fix D is medium risk -- new state tracking, needs to not block legitimate rapid changes during shaving
