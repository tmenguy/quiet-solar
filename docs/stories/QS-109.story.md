# Bug Fix: home_available_power clamp ignores grid export, incorrectly zeroes available power

issue: 109
branch: "QS_109"

Status: done

## Story

As a Quiet Solar user with significant solar production,
I want the available power clamp to account for grid export,
so that controllable loads (chargers, pools) can use the exported power instead of sending it to the grid.

## Bug Description

When exporting significant power to the grid (e.g. >9000W), the `max_available_home_power` clamp incorrectly reduces `home_available_power` to 0, preventing controllable loads from consuming surplus solar.

**Log trace:**
```text
Home available_power CLAMPED to max available home power: from 9445.00 to 0.00, (inverter_output_clamped:12382.00, max_available_home_power:0.00)
```

**Impact:** Controllable loads (EV chargers, pool pumps, etc.) see 0 W available and remain idle while the system exports thousands of watts to the grid.

## Root Cause Analysis

### Sign conventions (critical context)

| Variable | Positive means | Negative means |
|----------|---------------|----------------|
| `grid_consumption` | exporting to grid | importing from grid |
| `battery_charge_clamped` | battery charging | battery discharging |
| `inverter_output_clamped` | always >= 0, AC-side output | N/A |

### The computation (line 1683)

```python
self.home_available_power = grid_consumption + battery_charge_clamped
```

This correctly represents surplus power that could be redirected to home loads: grid export + battery charging power.

### The buggy clamp (lines 1685-1728)

When `home_available_power > 0`, the code computes `max_available_home_power` as remaining inverter headroom:

**DC-coupled / no battery (lines 1690-1705):**
```python
if inverter_output_clamped >= self.solar_plant.solar_max_output_power_value:
    max_available_home_power = 0  # BUG: ignores grid export
else:
    max_available_home_power = max(0, solar_max - inverter_output_clamped)
```

**AC-coupled battery (lines 1706-1718):**
```python
max_available_home_power = max(
    0,
    max_battery_discharge + solar_max - inverter_output_clamped,
)
# BUG: also ignores grid export
```

### Why this is wrong

Grid-exported power is ALREADY being produced by the inverter. Redirecting it to home loads requires NO additional inverter capacity — the power just flows to a local load instead of the grid. The current code treats `max_available_home_power` as remaining inverter headroom only, ignoring that exported power is already-produced, redirectable capacity.

**Bug scenario (from log):**
- `inverter_output_clamped` = 12382 W (inverter at/above max)
- `solar_max_output_power_value` ~= 12000 W
- `grid_consumption` ~= 9445 W (heavy export)
- Current: `max_available_home_power = 0` (inverter at max, no headroom)
- Correct: `max_available_home_power >= 9445` (exported power is available)

## Acceptance Criteria

1. **AC1**: When exporting power to the grid, the clamp includes grid export as redirectable capacity. `max_available_home_power` accounts for `max(0, grid_consumption)` in the DC-coupled/no-battery branch.
2. **AC2**: Same fix for the AC-coupled battery branch.
3. **AC3**: When NOT exporting (grid_consumption <= 0, i.e. importing), the clamp behavior is unchanged.
4. **AC4**: The 5% tolerance and `max(0, ...)` floor on the clamp remain intact.
5. **AC5**: All existing tests pass. New tests cover grid-export clamp scenarios for both DC and AC configurations.
6. **AC6**: 100% test coverage maintained.

## Tasks / Subtasks

- [x] Task 1: Fix clamp logic in `home_available_power_sensor_state_getter` (AC: #1, #2, #3, #4)
  - [x] 1.1 Compute `grid_export_redirectable = max(0.0, grid_consumption)` before the clamp block (after line 1683, inside the `if self.home_available_power > 0` guard)
  - [x] 1.2 DC-coupled / no battery branch (lines 1690-1705): add `grid_export_redirectable` to `max_available_home_power`
  - [x] 1.3 AC-coupled battery branch (lines 1706-1718): add `grid_export_redirectable` to `max_available_home_power`
  - [x] 1.4 Keep the existing `min(max_available_home_power, solar_max_output_power_value)` secondary clamp at line 1701 — verified it still makes sense (caps at inverter max which is correct)
- [x] Task 2: Add tests for grid-export clamp scenarios (AC: #5, #6)
  - [x] 2.1 Test: DC/no-battery, inverter at max, heavy grid export — clamp should NOT reduce available power to 0
  - [x] 2.2 Test: DC/no-battery, inverter below max, moderate grid export — clamp should allow headroom + export
  - [x] 2.3 Test: AC-coupled battery, inverter at max, heavy grid export — same as 2.1 but with battery
  - [x] 2.4 Test: no grid export (grid_consumption <= 0) — clamp unchanged (regression guard)
  - [x] 2.5 Test: exact boundary case where inverter_output == solar_max and grid_consumption == home_available_power
- [x] Task 3: Run quality gates and verify (AC: #6)

## Dev Notes

### Exact fix location

**File:** `custom_components/quiet_solar/ha_model/home.py`
**Method:** `home_available_power_sensor_state_getter()` (starts ~line 1517)
**Clamp block:** lines 1685-1728

### Proposed fix (DC-coupled / no battery, lines 1689-1705)

```python
# Power currently exported to grid can be redirected to home loads
# without needing additional inverter capacity
grid_export_redirectable = max(0.0, grid_consumption)

max_available_home_power = MAX_POWER_INFINITE
if self.battery is None or is_battery_dc_coupled:
    if (
        self.solar_plant is not None
        and self.solar_plant.solar_max_output_power_value < MAX_POWER_INFINITE
    ):
        if inverter_output_clamped >= self.solar_plant.solar_max_output_power_value:
            max_available_home_power = grid_export_redirectable
        else:
            max_available_home_power = max(
                0, self.solar_plant.solar_max_output_power_value - inverter_output_clamped
            ) + grid_export_redirectable
        max_available_home_power = min(
            max_available_home_power, self.solar_plant.solar_max_output_power_value
        )
    else:
        max_available_home_power = MAX_POWER_INFINITE
```

### Proposed fix (AC-coupled battery, lines 1706-1718)

```python
else:
    max_battery_discharge = self.battery.battery_get_current_possible_max_discharge_power()
    if self.solar_plant is None:
        max_available_home_power = max_battery_discharge
    elif self.solar_plant.solar_max_output_power_value < MAX_POWER_INFINITE:
        max_available_home_power = max(
            0,
            max_battery_discharge
            + self.solar_plant.solar_max_output_power_value
            - inverter_output_clamped
            + grid_export_redirectable,
        )
```

### Important: secondary clamp at line 1701

The existing `min(max_available_home_power, solar_max_output_power_value)` caps max_available to solar max. With grid export added, this cap may be too restrictive — if inverter headroom + grid export exceeds solar max, the cap could clip legitimate available power. **Verify during implementation** whether this secondary clamp should be kept, raised, or removed. Consider that `home_available_power` itself can exceed `solar_max` when battery is also charging.

### Architecture compliance

- Fix is in `ha_model` layer (`ha_model/home.py`) — correct, no domain boundary violation
- Uses only existing local variables (`grid_consumption`) — no new imports needed
- Lazy logging unchanged (no new log calls)

### Test patterns

Existing clamp tests in `tests/ha_tests/test_home_extended_coverage.py`:
- `test_available_power_clamped` (line 2917): no-battery, solar_max=1000, inverter=300/800, grid=250 — tests clamp but NOT the grid-export scenario
- `test_available_power_clamped_with_battery` (line 2958): AC battery, solar_max=500, inverter=2000, grid=3000 — high export but focuses on battery path

New tests should follow the same pattern: use `_inject_sensor_value` to set sensor readings, call `home_non_controlled_consumption_sensor_state_getter`, assert `home.home_available_power`.

Additional clamp-related tests in `tests/ha_tests/test_home_coverage.py`:
- Line 2043: AC-coupled battery max_available_home_power
- Line 2719: max_battery_discharge with no solar
- Line 3553: max_battery_discharge cap

### References

- [Source: custom_components/quiet_solar/ha_model/home.py#home_available_power_sensor_state_getter] — lines 1517-1739
- [Source: custom_components/quiet_solar/ha_model/home.py#clamp-block] — lines 1685-1728
- [Source: custom_components/quiet_solar/ha_model/home.py#grid_consumption-computation] — lines 1629-1642
- [Source: custom_components/quiet_solar/ha_model/home.py#battery_charge_clamped-computation] — lines 1548-1623
- [Source: tests/ha_tests/test_home_extended_coverage.py#test_available_power_clamped] — line 2917
- [Source: tests/ha_tests/test_home_extended_coverage.py#test_available_power_clamped_with_battery] — line 2958

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Completion Notes List

- Story created from GitHub issue #109 analysis + codebase exploration
- Root cause: clamp computes max_available_home_power as inverter headroom only, ignoring grid export
- Fix: add max(0, grid_consumption) to max_available_home_power in both DC and AC branches
- Secondary clamp at line 1701 verified — `min(max_available_home_power, solar_max)` still correct, caps redirectable power at inverter max
- Updated existing test_available_power_clamped assertion (210→250) to reflect correct post-fix behavior
- 5 new tests added, all quality gates pass, 100% coverage maintained
- Review: DC-coupled battery charge modeled as DC bus physics — `dc_battery_redirectable = min(charge, inverter_headroom)` since charge beyond headroom is DC overflow
- Review: AC-coupled branch now has symmetric secondary cap `min(max_available, solar_max + max_battery_discharge)`
- Review: no-export regression test rewritten to use DC battery + import to actually exercise clamp block (CodeRabbit finding)

### File List

- `custom_components/quiet_solar/ha_model/home.py` (modify)
- `tests/ha_tests/test_home_extended_coverage.py` (modify)
