# Bug Fix: DC-Coupled Battery Excessive Grid Import

Status: ready-for-dev
issue: 122
branch: "QS_122"

## Story

As an operator of a DC-coupled battery system,
I want `home_available_power` to exclude the DC battery charge overflow that can never reach AC loads,
so that the dyn_handle does not see phantom surplus and increase car charging, causing unnecessary grid import.

## Acceptance Criteria

1. **DC overflow subtracted**: For DC-coupled systems, `home_available_power` at line 1683 of `ha_model/home.py` subtracts the forced DC overflow (solar production minus inverter max output, capped by battery charge) before the value is stored in history.
2. **Voluntary charge clamp**: When grid is importing (beyond -200 W noise threshold) while the battery charges voluntarily above DC overflow, `home_available_power` is clamped to 0 with a warning log.
3. **Downstream clamp untouched**: The existing `max_available_home_power` clamping block (lines 1686-1746) remains as a safety net — no regressions.
4. **AC-coupled unaffected**: Systems without DC-coupled batteries see zero change in behavior (`dc_overflow = 0`).
5. **Tests**: New parametrized tests cover DC-coupled overflow subtraction, voluntary-charge grid-import clamp, AC-coupled no-op, and edge cases (no solar plant, zero battery charge).

## Tasks / Subtasks

- [ ] **Task 1 — Fix `home_available_power` computation** (AC: 1, 2, 3, 4)
  - [ ] 1.1 After line 1683 (`self.home_available_power = grid_consumption + battery_charge_clamped`), compute `dc_overflow` for DC-coupled systems and subtract it.
  - [ ] 1.2 Add voluntary-charge grid-import clamp: if `voluntary_charge > 0` and `grid_consumption < -200.0`, clamp `home_available_power` to `min(self.home_available_power, 0.0)` with `_LOGGER.warning`.
  - [ ] 1.3 Verify the existing `max_available_home_power` clamp block is unchanged.

- [ ] **Task 2 — Tests** (AC: 5)
  - [ ] 2.1 Parametrized test for DC-coupled: solar 13200 W, inverter max 12000 W, battery charge 3600 W → dc_overflow = 1200 W, home_available_power reduced by 1200.
  - [ ] 2.2 Test voluntary-charge clamp: grid importing -500 W, voluntary charge 2400 W → home_available_power clamped to 0.
  - [ ] 2.3 Test AC-coupled no-op: same scenario, `is_dc_coupled=False` → dc_overflow = 0, no clamp.
  - [ ] 2.4 Test edge: no solar plant → dc_overflow = 0.
  - [ ] 2.5 Test edge: battery_charge_clamped = 0 → dc_overflow = 0, no voluntary clamp.

## Dev Notes

### Root Cause

In `home_non_controlled_consumption_sensor_state_getter()` (`ha_model/home.py:1683`):

```python
self.home_available_power = grid_consumption + battery_charge_clamped
```

For DC-coupled systems, `battery_charge_clamped` includes the **full** battery charge (e.g., 3.6 kW), but a portion is **forced DC overflow** — solar production exceeding inverter AC capacity that is forced into the battery and can never reach AC loads. The code already computes the correct redirectable amount at lines 1707-1709:

```python
dc_battery_redirectable = min(battery_charge_clamped, inverter_headroom)
max_available_home_power = grid_consumption + dc_battery_redirectable
```

But this is only used as an **after-the-fact clamp with 5% tolerance** (line 1738). The raw overcounted `home_available_power` gets written to history. The dyn_handle reads a median of recent samples, so it inherits overcounted values → sees phantom surplus → increases car charging → grid import.

### Fix Location

**Single file**: `custom_components/quiet_solar/ha_model/home.py`
**Single method**: `home_non_controlled_consumption_sensor_state_getter()`
**Insert after line 1683**, before the existing clamp block at line 1685.

### Exact Fix Code

```python
self.home_available_power = grid_consumption + battery_charge_clamped

# --- BEGIN FIX (issue #122) ---
# For DC-coupled: subtract the DC overflow that is forced into the battery
# and can never reach AC loads
dc_overflow = 0.0
if is_battery_dc_coupled and battery_charge_clamped > 0 and solar_production_not_clamped is not None:
    if (self.solar_plant is not None
            and self.solar_plant.solar_max_output_power_value < MAX_POWER_INFINITE):
        dc_overflow = max(0.0, solar_production_not_clamped - self.solar_plant.solar_max_output_power_value)
        dc_overflow = min(dc_overflow, battery_charge_clamped)
        self.home_available_power -= dc_overflow

# For any battery (DC or AC coupled): if grid is importing while battery
# charges voluntarily, the system is not redirecting charge to cover AC demand.
# Don't count voluntary charge as available.
if battery_charge_clamped > 0:
    voluntary_charge = battery_charge_clamped - dc_overflow
    if voluntary_charge > 0 and grid_consumption < -200.0:
        prev = self.home_available_power
        self.home_available_power = min(self.home_available_power, 0.0)
        _LOGGER.warning(
            "Home available_power CLAMPED to 0: grid importing %.2f while battery charges %.2f voluntarily (dc_overflow:%.2f), was %.2f",
            grid_consumption, voluntary_charge, dc_overflow, prev,
        )
# --- END FIX (issue #122) ---
```

### Variables Available at Fix Location

All these are already computed and in scope at line 1683:
- `is_battery_dc_coupled` (bool) — set at line 1551
- `battery_charge_clamped` (float) — set at lines 1552-1623, defaulted to 0 if None
- `solar_production_not_clamped` (float|None) — set at lines 1574-1613
- `inverter_output_clamped` (float) — set at lines 1565-1620, defaulted to 0 if None
- `grid_consumption` (float) — set at line 1629
- `self.solar_plant.solar_max_output_power_value` (float) — inverter AC max capacity
- `MAX_POWER_INFINITE` — sentinel from `const.py`

### Architecture Constraints

- **Two-layer boundary**: This fix is in `ha_model/home.py` (HA integration layer) — no domain boundary violation.
- **Logging**: Use lazy `%s` formatting, no f-strings, no trailing period.
- **Constants**: `MAX_POWER_INFINITE` already imported from `const.py`.

### Testing Patterns

Existing tests in `tests/ha_tests/test_home.py` and `tests/ha_tests/test_home_coverage.py` mock `self.battery`, `self.solar_plant`, and sensor values. Follow the same pattern:
- Mock `self.battery.is_dc_coupled`
- Mock `self.solar_plant.solar_max_output_power_value`
- Mock sensor return values for `battery_charge_clamped`, `solar_production_not_clamped`, `inverter_output_clamped`, `grid_consumption`
- Assert `self.home_available_power` after calling `home_non_controlled_consumption_sensor_state_getter()`

No existing tests cover DC-coupled + home_available_power interaction — all new test cases needed.

### Project Structure Notes

- Single file change: `custom_components/quiet_solar/ha_model/home.py`
- Tests go in existing test file or new `tests/ha_tests/test_home_dc_coupled.py`
- No config, UI, or translation changes needed

### References

- [Source: ha_model/home.py:1550-1757] — `home_non_controlled_consumption_sensor_state_getter()`
- [Source: ha_model/home.py:1683] — the overcounted `home_available_power` assignment
- [Source: ha_model/home.py:1707-1709] — existing DC redirectable computation (downstream clamp)
- [Source: external plan] — `dc-coupled_battery_grid_import_bug_8ce997f9.plan.md`

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Completion Notes List

- External plan provided as primary authority; story synthesizes plan structure with standard story format.
- The plan's Fix 1 (Part A + Part B) is the sole implementation task — no additional fixes needed.
- The -200 W threshold in Part B is a noise filter to avoid false positives from sensor jitter.
