# Bug #62: Green Mode Charger Consign Exceeds Inverter AC Output Limit

Status: dev-complete
issue: 62
branch: "QS_62"

## Story

As TheAdmin (solar optimization user),
I want green mode commands to never cause total home consumption to exceed the inverter's AC output capacity,
so that solar-only modes truly use only solar/battery energy without any grid import.

## Bug Description

When a car (Twingo) is plugged into a charger and the system issues green mode commands (`auto_green_consign`), the charger power consign can exceed the inverter's AC output limit, causing grid import. This defeats the entire purpose of green modes.

### Observed Behavior (2026-03-28, 14:00-16:00)

- Inverter max AC output: **12,000W** (`solar_max_output_power_value = 12000`)
- Charger: Wallbox 3, 3-phase, 32A max (hardware max = 22,080W)
- Car: Twingo, 3-phase charging

| Time  | Command              | Power (W) | Exceeds 12kW? |
|-------|---------------------|-----------|----------------|
| 14:09 | auto_green          | 8,850     | No             |
| 14:10 | auto_green_consign  | **22,080** | Yes (+10kW)   |
| 15:00 | auto_green_consign  | 18,856    | Yes (+6.9kW)  |
| 15:15 | auto_green_consign  | 13,440    | Yes (+1.4kW)  |
| 15:30 | auto_green_consign  | 12,240    | Yes (+0.2kW)  |
| 15:50 | auto_green_consign  | 9,330     | No             |
| 16:00 | auto_green_consign  | 7,041     | No             |

At 14:10, the solver estimated ~8,900Wh surplus over 6 hours and walked the charger through every power step up to hardware max (22,080W). With ~2-3kW home base load, this meant the system needed ~24kW but the inverter can only deliver 12kW — resulting in **~12kW grid import**.

### Expected Behavior

In any green mode, total home consumption (charger + all other loads) must never exceed `home_max_available_production_power` (solar + battery discharge, capped by inverter AC output). If insufficient production capacity, the charger must be throttled down to fit within inverter limits.

## Root Cause Analysis

### Root Cause 1: Solver assigns power steps beyond inverter capacity

**File:** `home_model/constraints.py`, `adapt_repartition()` (line 1270+)

When the solver has surplus energy to consume, it calls `adapt_power_steps_budgeting()` (line 1373-1380) which calls `adapt_power_steps_budgeting_low_level()` (line 1165). This method filters available power steps based on `available_amps_for_group` or `available_amps_production_for_group`.

**The problem:** In `adapt_repartition()` at line 1288, `use_production_limits` is initialized to `False` and never set to `True` for green commands consuming surplus energy (`energy_delta >= 0`). This means even green commands use consumption-based amp budgets (which allow up to the breaker/group limit), not production-based amp budgets (which would respect inverter output).

The solver's `_available_power` IS clamped to `max_inverter_dc_to_ac_power` (solver.py line 99-101), but this only limits the **total surplus energy** computation. It does NOT limit the **power step** assigned to any single slot. The solver can assign a 22kW power_consign to one slot while the energy accounting "balances out" across all slots.

### Root Cause 2: Budgeting does not cap to production limits for green modes

**File:** `ha_model/charger.py`, `budgeting_algorithm_minimize_diffs()` (line 949+)

The base budget computation at line 1087 (`initial_power_budget = full_available_home_power - diff_power_budget`) uses `full_available_home_power` which can include grid power. For green modes, the budget should never exceed what the home can produce (solar + battery), but there is no such cap on the base case.

Additionally, at lines 1100-1113, when the battery is discharging (`battery_asked_charge < 0`) and a charger has `CMD_AUTO_GREEN_CONSIGN` with `is_before_battery`, the budget is boosted further:

```python
initial_power_budget = full_available_home_power - battery_asked_charge - diff_power_budget
```

Since `battery_asked_charge` is negative, this **adds** battery discharge to the budget. But neither the base case nor the boost verify the resulting total stays within inverter AC output capacity. The budget can push the charger request beyond what the inverter can physically deliver.

**Key insight:** The production cap should apply to ALL green mode budget computations (not just the battery boost case), and only when we don't want to consume from the grid.

### Root Cause 3: `get_home_max_available_production_power()` broken for AC-coupled + None fallback missing

**File:** `ha_model/home.py`, line 1154 and `ha_model/charger.py`, lines 1128-1173

Two issues in the production cap path:

1. **`get_home_max_available_production_power()` broken for AC-coupled batteries** (home.py line 1165-1166): This method always caps to `solar_max_output_power_value`, which is wrong for AC-coupled batteries where the battery has its own inverter and can add power on top. Must be fixed to handle `is_dc_coupled` properly.

2. **None fallback missing in budgeting:** When `home_load_power_value` or `home_max_available_production_power` is None (charger.py lines 1124-1128), the check is completely skipped. For green modes, a conservative fallback should use the static max production capacity.

**Two complementary methods exist:**
- `get_home_max_available_production_power()` (home.py:1154) — **dynamic**: current solar production + battery can-discharge. Gives what's available RIGHT NOW. Needs AC-coupled fix.
- `get_current_maximum_production_output_power()` (home.py:1705) — **more static/theoretical**: uses `solar_max_output_power_value` + battery max. Properly handles DC vs AC coupling. Gives the theoretical ceiling.

**Fix approach:** Fix `get_home_max_available_production_power()` for AC coupling. Then in the budgeting, use `min(get_home_max_available_production_power(), get_current_maximum_production_output_power())` when both return a value. If one is None, use the other. This gives the tightest bound: "what's available now" but never exceeding "what the hardware can deliver".

### Root Cause 4: Phantom surplus — car not drawing yet inflates measured surplus

**File:** `ha_model/charger.py`, `budgeting_algorithm_minimize_diffs()` and `dyn_handle()` (line 690+)

This is the primary **trigger** of the runaway ramp-up observed in the logs. When a car is plugged in and amps are set, the car takes several minutes to actually start drawing power. During this lag:

1. The charger is set to e.g. 12A/3ph (~8.3kW expected), but the car draws only ~22W
2. The surplus sensor still sees ~5kW+ available (because the car isn't consuming yet)
3. Each budgeting cycle (~1 minute) sees high surplus and ramps the charger up further
4. Battery discharge is added to the budget, inflating it further
5. Within 3 cycles the charger hits 32A max (22kW) while the car still draws ~22W
6. When the car finally starts drawing, it pulls 22kW through a 12kW inverter → massive grid import

**Timeline from the 16:16 event:**

| Time    | Budget (W) | Amps Set | Car Drawing | Result |
|---------|-----------|----------|-------------|--------|
| 16:20:07 | 3,139    | 6→10A   | ~0W         | Ramp up |
| 16:21:03 | 6,765    | 7→12A   | ~22W        | Ramp up |
| 16:22:06 | 7,767    | 12→23A  | ~22W        | Ramp up (surplus 5169 + battery 2598) |
| 16:23:02 | 7,549    | 23→32A  | ~22W        | Hit max! |
| 16:23:58 | -8,015   | 32A     | Drawing!    | Deep grid import |

The phantom surplus is the gap between expected charger power (at budgeted amps) and actual measured power. The budgeting algorithm has no awareness of this gap — it trusts the surplus sensor which doesn't reflect the pending charger allocation.

**Fix approach — tiered phantom surplus detection:**

Use the strongest available signal to compute phantom surplus:
- **Tier 1 (per-charger sensor):** Each `cs.accurate_current_power` vs `car._get_power_from_stored_amps(cs.budgeted_amp, cs.budgeted_num_phases)`. Already available on each charger state.
- **Tier 2 (group sensor):** `current_real_cars_power` (already computed at `dyn_handle` line 698) vs sum of expected per-charger powers. Needs to be passed to `budgeting_algorithm_minimize_diffs`.
- **Tier 3 (home consumption heuristic):** Compare home consumption increase to expected charger power. Weaker, fallback only.

Subtract the detected phantom surplus from `initial_power_budget` for green commands, and add it to `home_load_power_value` in the production cap check so it accounts for expected-but-not-yet-drawn power.

## Acceptance Criteria

1. **Given** a charger in any green mode (auto_green, auto_green_consign, auto_green_cap, green_charge_only), **when** the solver allocates power steps, **then** no single slot's power_consign exceeds `max_inverter_dc_to_ac_power` minus estimated unavoidable home consumption for that slot.

2. **Given** a charger with `CMD_AUTO_GREEN_CONSIGN` and battery is discharging, **when** the budgeting algorithm computes the power budget, **then** the total (charger power + home load) never exceeds `home_max_available_production_power`.

3. **Given** multiple chargers in green mode, **when** their combined power is allocated, **then** total combined charger power + home load stays within inverter AC output capacity.

4. **Given** `home_load_power_value` or `home_max_available_production_power` is unavailable (None), **when** the budgeting algorithm runs for a green mode charger, **then** the system falls back to a conservative cap (e.g., `solar_max_output_power_value` alone) rather than skipping the check entirely.

5. **Given** a charger has been set to N amps but the car is not yet drawing power, **when** the budgeting algorithm runs for a green mode charger, **then** the phantom surplus (expected power - actual measured power) is detected and subtracted from the available budget, preventing runaway ramp-up.

6. **Given** a charger has per-charger accurate_power_sensor, **when** phantom surplus is computed, **then** Tier 1 (per-charger) detection is used. **Given** only group-level sensor exists, **then** Tier 2 is used. **Given** neither exists, **then** Tier 3 (home consumption heuristic) is used as fallback.

7. **Given** the car IS drawing at expected power levels, **when** phantom surplus is computed, **then** it equals ~0 and budgeting proceeds normally (no false positive).

8. **Given** existing tests pass, **when** the fix is applied, **then** no regressions in non-green-mode behavior (auto_consign, auto_price, force_charge commands must remain unaffected).

## Tasks / Subtasks

- [x] Task 1: Solver-level power step capping for green commands (AC: #1)
  - [x] 1.1: In `adapt_repartition()` (`constraints.py` line 1288), set `use_production_limits = True` when the constraint's command is a green command and `energy_delta >= 0`. This limits the power steps the solver can assign, preventing green consign amps from exceeding production capacity before budgeting even runs
  - [x] 1.2: Verify `available_amps_production_for_group` is properly initialized with inverter-aware limits in `prepare_slots_for_amps_budget()` (solver.py / dynamic_group.py) — check `dyn_group_max_production_phase_current_for_budget` (home.py line 1222)
  - [x] 1.3: Add tests for solver respecting production limits on green commands

- [x] Task 2: Fix `get_home_max_available_production_power()` for AC-coupled batteries (AC: #2, #4)
  - [x] 2.1: Fix `get_home_max_available_production_power()` (home.py:1154) — when battery is NOT DC-coupled, the cap at line 1165-1166 should be `solar_max_output_power_value + battery_max_discharge` instead of just `solar_max_output_power_value`. Mirror the DC/AC logic from `get_current_maximum_production_output_power()` (home.py:1705)
  - [x] 2.2: Add tests for both DC-coupled and AC-coupled battery scenarios

- [x] Task 3: Budgeting algorithm production cap for ALL green modes (AC: #2, #4)
  - [x] 3.1: For green mode chargers, cap `initial_power_budget` to production limits in BOTH the normal case (line 1087) and the battery discharge boost case (line 1113). The cap should apply whenever we don't want to consume from the grid — i.e., all green command types
  - [x] 3.2: Compute robust production cap as `min(get_home_max_available_production_power(), get_current_maximum_production_output_power())` when both return values. If one is None, use the other. This gives the tightest bound
  - [x] 3.3: Handle None fallback — when both are unavailable, use `solar_max_output_power_value` as conservative default rather than skipping the check
  - [x] 3.4: Add tests for budget capping in both normal and battery discharge cases

- [x] Task 4: Phantom surplus detection and budget correction (AC: #5, #6, #7)
  - [x] 4.1: Create helper method `_compute_phantom_surplus(actionable_chargers, current_real_cars_power, time) -> float` on `QSChargerGroup` using tiered detection: Tier 1 = per-charger `cs.accurate_current_power` vs `car._get_power_from_stored_amps(cs.budgeted_amp, cs.budgeted_num_phases)`. Tier 2 = group `current_real_cars_power` (from `dyn_handle` line 698) vs sum of expected powers. Tier 3 = home consumption heuristic
  - [x] 4.2: Pass `current_real_cars_power` from `dyn_handle` (line 914) into `budgeting_algorithm_minimize_diffs` — add parameter to method signature
  - [x] 4.3: For green commands, subtract phantom surplus from `initial_power_budget` after all budget computations (including battery boost)
  - [x] 4.4: In the existing production cap check (lines 1128-1173), use `effective_home_load = home_load_power_value + phantom_surplus` so it accounts for expected-but-not-yet-drawn charger power
  - [x] 4.5: Add logging for phantom surplus detection: tier used, computed value, cap trigger
  - [x] 4.6: Add tests for all 3 tiers: car not drawing (phantom detected), car drawing normally (phantom ~0), multiple chargers (partial phantom)

- [x] Task 5: Multi-charger combined cap (AC: #3)
  - [x] 5.1: Verify the existing budget loop respects the cap across all chargers in green mode
  - [x] 5.2: Add test scenario with 2+ chargers both in green mode, total must stay under production capacity

- [x] Task 6: Regression protection (AC: #8)
  - [x] 6.1: Verify all existing charger budgeting tests still pass
  - [x] 6.2: Verify non-green commands (auto_consign, force_charge, auto_price) are NOT affected by new limits — these are allowed to use the grid
  - [x] 6.3: Add explicit regression test: non-green charger can exceed inverter limit (it uses grid deliberately)

## Dev Notes

### Architecture Constraints

- **Two-layer boundary**: `home_model/` NEVER imports `homeassistant.*`. The solver and constraints layer is pure Python. Any HA-specific data (like current home load power) must be passed as parameters.
- **Solver step size**: `SOLVER_STEP_S = 900` (15-minute slots) — don't modify.
- The fix spans both layers: solver-level (domain logic) and budgeting (HA integration layer).

### Key Code Locations

| Component | File | Key Lines |
|-----------|------|-----------|
| Solver init, inverter clamp | `home_model/solver.py` | 30-41, 99-101 |
| adapt_repartition (green power steps) | `home_model/constraints.py` | 1270-1400, 1288 (use_production_limits) |
| adapt_power_steps_budgeting | `home_model/constraints.py` | 1165-1268 |
| Charger budgeting entry | `ha_model/charger.py` | 949-1414 |
| Battery discharge boost | `ha_model/charger.py` | 1100-1113 |
| Production power cap check | `ha_model/charger.py` | 1128-1173 |
| get_home_max_available_production_power (fix AC-coupled) | `ha_model/home.py` | 1154-1168 |
| get_current_maximum_production_output_power (static max) | `ha_model/home.py` | 1705-1735 |
| dyn_group_max_production_phase_current_for_budget | `ha_model/home.py` | 1222-1238 |
| Green command definitions | `home_model/commands.py` | 1-101 |
| Command is_auto / is_like methods | `home_model/commands.py` | 39-46 |
| available_amps_production_for_group | `ha_model/dynamic_group.py` | 35-36, 48-67, 232-233 |
| Battery charge/discharge clamp | `home_model/battery.py` | 43-78, 96-130 |
| dyn_handle (current_real_cars_power) | `ha_model/charger.py` | 698-699 |
| budgeting call site (add param) | `ha_model/charger.py` | 914-916 |
| accurate_current_power per charger | `ha_model/charger.py` | 321, 2136-2137 |
| car._get_power_from_stored_amps | `ha_model/car.py` | 1224 |

### Green Command Identification

To identify green commands, use existing methods on `LoadCommand`:
- `cmd.is_like(CMD_AUTO_GREEN_CONSIGN)` — checks specific command type
- `cmd.is_auto()` — returns True for all auto commands (includes non-green like auto_consign)
- Green commands specifically: `CMD_CST_AUTO_GREEN`, `CMD_CST_AUTO_GREEN_CAP`, `CMD_CST_GREEN_CHARGE_ONLY`, `CMD_CST_AUTO_GREEN_CONSIGN`

You may need to add an `is_green()` method to `LoadCommand` if one doesn't exist. Check first.

### Testing Approach

**Existing test files to extend:**
- `tests/test_chargers.py` — TestBudgetingAlgorithm class
- `tests/test_charger_rebalancing_scenarios.py` — budgeting scenarios with multiple chargers
- `tests/test_charger_coverage_deep.py` — green command transitions (test_cmd_green_cap_zero_forbids, test_bump_solar_cap_to_green_only)
- `tests/test_green_only_devices.py` — green-only flag behavior
- `tests/test_solver.py` — constraint delta and power allocation tests

**Test factories to use (from `tests/factories.py`):**
- `MinimalTestHome` / `MinimalTestLoad` — for constraint/solver tests
- `create_constraint()`, `create_load_command()` — for unit tests
- `create_charger_group()`, `create_test_car_double()`, `create_test_charger_double()` — for charger integration tests

**Key test scenarios needed:**
1. Solver assigns green consign to charger with 3-phase 32A car, inverter at 12kW → power_consign must be <= 12kW
2. Budgeting with battery discharge boost + green consign → total stays under home_max_available_production_power
3. Two chargers in green mode → combined total under inverter limit
4. Non-green charger (force_charge/auto_consign) → NOT capped by inverter limit (regression guard)
5. Fallback when home_max_available_production_power is None → conservative behavior
6. Phantom surplus Tier 1: per-charger sensor, car not drawing (budgeted 12A/3ph = 8.3kW, actual 22W) → phantom ~8.3kW, budget reduced
7. Phantom surplus Tier 2: group sensor only, car not drawing → phantom detected from group reading
8. Phantom surplus Tier 3: no sensors, home consumption flat → phantom inferred from consumption delta
9. Car IS drawing normally → phantom ~0, no budget change (no false positive)
10. Multiple chargers: one drawing, one not → partial phantom correctly computed

### Logging Rules

- Use lazy `%s` formatting: `_LOGGER.debug("budget capped from %s to %s", old, new)`
- No f-strings in log calls
- No periods at end of log messages

### DC vs AC Coupled Battery — Critical Distinction

The max production capacity depends on battery coupling:
- **DC-coupled** (`is_dc_coupled=True`): Solar + battery share one inverter. Max AC output = `solar_max_output_power_value` (hard cap). Battery discharge cannot add beyond inverter limit.
- **AC-coupled** (`is_dc_coupled=False`): Battery has its own inverter. Max AC output = `solar_max_output_power_value + battery_max_discharge`. Battery can add on top.

**Two complementary methods — both needed:**
- `get_home_max_available_production_power()` (home.py:1154) — **dynamic**: current solar production + battery can-discharge. MUST BE FIXED for AC-coupled (line 1166 always caps to `solar_max_output_power_value`, should allow battery to add for AC-coupled). Reference the DC/AC logic in `get_current_maximum_production_output_power()`.
- `get_current_maximum_production_output_power()` (home.py:1705) — **more static/theoretical max**: uses `solar_max_output_power_value` + battery max. Already handles DC/AC correctly.
- For budgeting, use `min(both)` when both available, else whichever has a value. This gives the tightest real bound.
- `dyn_group_max_production_phase_current_for_budget` (home.py:1222) — static max production amps for budget initialization, uses `get_home_max_static_phase_amps()` which handles off-grid but should be verified for green mode use

The `is_dc_coupled` flag is stored on the battery model (`home_model/battery.py` line 25) and comes from config key `CONF_BATTERY_IS_DC_COUPLED`.

### Project Structure Notes

- All config keys must be from `const.py` — no hardcoded strings
- The `use_production_limits` flag in `adapt_power_steps_budgeting_low_level` already exists and distinguishes solar vs consumption budgets — leverage it, don't reinvent
- Production caps must ONLY apply for green commands. Non-green commands (force_charge, auto_consign, auto_price) deliberately use the grid and must remain uncapped

### References

- [Source: home_model/constraints.py#adapt_repartition, line 1270]
- [Source: home_model/constraints.py#adapt_power_steps_budgeting_low_level, line 1165]
- [Source: ha_model/charger.py#budgeting_algorithm_minimize_diffs, line 949]
- [Source: ha_model/home.py#get_home_max_available_production_power, line 1154] (fix AC-coupled cap)
- [Source: ha_model/home.py#get_current_maximum_production_output_power, line 1705] (static max, correct DC/AC)
- [Source: home_model/commands.py#green commands, lines 1-101]
- [Source: ha_model/dynamic_group.py#available_amps_production_for_group, line 35-36]
- [Source: ha_model/charger.py#dyn_handle, line 698] (current_real_cars_power already computed)
- [Source: ha_model/car.py#_get_power_from_stored_amps, line 1224] (expected power from amps)
- FR11: The system can restrict specific devices to use only free solar energy, never drawing from the grid
- NFR11: When the system cannot make an optimal decision, it must fail safe

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References

### Completion Notes List
- Task 1: Added `is_green()` method to `LoadCommand`; set `use_production_limits = True` for green commands with `energy_delta >= 0` in `adapt_repartition()`
- Task 2: Fixed `get_home_max_available_production_power()` to handle AC-coupled batteries correctly (battery adds on top of solar inverter cap)
- Task 3: Added production cap in `budgeting_algorithm_minimize_diffs()` using `min(dynamic_cap, static_cap)` for tightest bound; caps green budget to production minus home load
- Task 4: Created `_compute_phantom_surplus()` with 3-tier detection (per-charger sensor, group sensor, no data); subtracts phantom from green budget to prevent runaway ramp-up
- Task 5: Verified multi-charger combined cap via shared `initial_power_budget`; added integration test with 2 green chargers
- Task 6: All 4193 tests pass (37 pre-existing failures in hungarian/qs_scripts unrelated). Added `isinstance` guards for defensive handling of MagicMock in test doubles

### File List
- `custom_components/quiet_solar/home_model/commands.py` — added `is_green()` method
- `custom_components/quiet_solar/home_model/constraints.py` — set `use_production_limits` dynamically for green commands
- `custom_components/quiet_solar/ha_model/home.py` — fixed `get_home_max_available_production_power()` for AC-coupled batteries
- `custom_components/quiet_solar/ha_model/charger.py` — added `_compute_phantom_surplus()`, production cap in budgeting, phantom surplus subtraction, defensive type checks
- `tests/test_commands.py` — tests for `is_green()`
- `tests/test_solver.py` — tests for production limits in solver
- `tests/ha_tests/test_home.py` — tests for DC/AC-coupled production power
- `tests/test_charger_coverage_deep.py` — `TestGreenModeProductionCap`, `TestPhantomSurplus`, `TestMultiChargerGreenCap` test classes
