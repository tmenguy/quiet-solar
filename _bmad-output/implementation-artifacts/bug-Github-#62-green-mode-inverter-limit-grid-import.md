# Bug #62: Green Mode Charger Consign Exceeds Inverter AC Output Limit

Status: ready-for-dev
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

### Root Cause 2: Budgeting battery discharge boost bypasses inverter cap

**File:** `ha_model/charger.py`, `budgeting_algorithm_minimize_diffs()` (line 949+)

At lines 1100-1113, when the battery is discharging (`battery_asked_charge < 0`) and a charger has `CMD_AUTO_GREEN_CONSIGN` with `is_before_battery`, the budget is boosted:

```python
initial_power_budget = full_available_home_power - battery_asked_charge - diff_power_budget
```

Since `battery_asked_charge` is negative, this **adds** battery discharge to the budget. But it does not verify the resulting total stays within inverter AC output. The boost can push the charger request beyond what the inverter can physically deliver.

### Root Cause 3: Late-stage production cap is insufficient

**File:** `ha_model/charger.py`, lines 1128-1173

There IS a check against `home_max_available_production_power` that caps the budget:
```python
initial_power_budget = min(
    initial_power_budget,
    home_max_available_production_power - home_load_power_value - diff_power_budget,
)
```

But this fires AFTER the battery boost (root cause 2) and only as a correction. The problem is that `diff_power_budget` already includes the charger's current allocation (which may be too high from the solver's consign), making the cap less effective. Also, when `home_load_power_value` or `home_max_available_production_power` is None (lines 1124-1126), the check is completely skipped.

## Acceptance Criteria

1. **Given** a charger in any green mode (auto_green, auto_green_consign, auto_green_cap, green_charge_only), **when** the solver allocates power steps, **then** no single slot's power_consign exceeds `max_inverter_dc_to_ac_power` minus estimated unavoidable home consumption for that slot.

2. **Given** a charger with `CMD_AUTO_GREEN_CONSIGN` and battery is discharging, **when** the budgeting algorithm computes the power budget, **then** the total (charger power + home load) never exceeds `home_max_available_production_power`.

3. **Given** multiple chargers in green mode, **when** their combined power is allocated, **then** total combined charger power + home load stays within inverter AC output capacity.

4. **Given** `home_load_power_value` or `home_max_available_production_power` is unavailable (None), **when** the budgeting algorithm runs for a green mode charger, **then** the system falls back to a conservative cap (e.g., `solar_max_output_power_value` alone) rather than skipping the check entirely.

5. **Given** existing tests pass, **when** the fix is applied, **then** no regressions in non-green-mode behavior (auto_consign, auto_price, force_charge commands must remain unaffected).

## Tasks / Subtasks

- [ ] Task 1: Solver-level power step capping for green commands (AC: #1)
  - [ ] 1.1: In `adapt_repartition()` (`constraints.py` line 1288), set `use_production_limits = True` when the constraint's command is a green command and `energy_delta >= 0`
  - [ ] 1.2: Verify `available_amps_production_for_group` is properly initialized with inverter-aware limits in `prepare_slots_for_amps_budget()` (solver.py / dynamic_group.py)
  - [ ] 1.3: Add tests for solver respecting production limits on green commands

- [ ] Task 2: Budgeting algorithm hard cap for green modes (AC: #2, #4)
  - [ ] 2.1: After battery discharge boost (charger.py lines 1110-1113), immediately clamp `initial_power_budget` so that `home_load_power_value + diff_power_budget + initial_power_budget <= home_max_available_production_power`
  - [ ] 2.2: Handle None fallback — when `home_max_available_production_power` is unavailable, use `solar_max_output_power_value` as conservative cap for green commands
  - [ ] 2.3: Add tests for budget capping with battery discharge

- [ ] Task 3: Multi-charger combined cap (AC: #3)
  - [ ] 3.1: Verify the existing budget loop respects the cap across all chargers in green mode
  - [ ] 3.2: Add test scenario with 2+ chargers both in green mode, total must stay under inverter limit

- [ ] Task 4: Regression protection (AC: #5)
  - [ ] 4.1: Verify all existing charger budgeting tests still pass
  - [ ] 4.2: Verify non-green commands (auto_consign, force_charge, auto_price) are NOT affected by new limits
  - [ ] 4.3: Add explicit regression test: non-green charger can exceed inverter limit (it uses grid deliberately)

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
| home_max_available_production_power | `ha_model/home.py` | 1154-1168 |
| Green command definitions | `home_model/commands.py` | 1-101 |
| Command is_auto / is_like methods | `home_model/commands.py` | 39-46 |
| available_amps_production_for_group | `ha_model/dynamic_group.py` | 35-36, 48-67, 232-233 |
| Battery charge/discharge clamp | `home_model/battery.py` | 43-78, 96-130 |

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

### Logging Rules

- Use lazy `%s` formatting: `_LOGGER.debug("budget capped from %s to %s", old, new)`
- No f-strings in log calls
- No periods at end of log messages

### Project Structure Notes

- All config keys must be from `const.py` — no hardcoded strings
- The `use_production_limits` flag in `adapt_power_steps_budgeting_low_level` already exists and distinguishes solar vs consumption budgets — leverage it, don't reinvent

### References

- [Source: home_model/constraints.py#adapt_repartition, line 1270]
- [Source: home_model/constraints.py#adapt_power_steps_budgeting_low_level, line 1165]
- [Source: ha_model/charger.py#budgeting_algorithm_minimize_diffs, line 949]
- [Source: ha_model/home.py#get_home_max_available_production_power, line 1154]
- [Source: home_model/commands.py#green commands, lines 1-101]
- [Source: ha_model/dynamic_group.py#available_amps_production_for_group, line 35-36]
- FR11: The system can restrict specific devices to use only free solar energy, never drawing from the grid
- NFR11: When the system cannot make an optimal decision, it must fail safe

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
