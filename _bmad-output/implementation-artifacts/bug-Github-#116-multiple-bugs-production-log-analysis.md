# Bug Fix: Multiple bugs found in 2026-04-04 production log analysis

Status: todo
issue: 116
branch: "QS_116"

## Story

As a Quiet Solar user with multiple chargers, a battery, and several on/off loads,
I want the solver and charger budgeting to handle edge cases (empty command lists, None amp values, unavailable entities, expired constraints) gracefully,
so that the system does not crash, spam errors, or fail to allocate solar charging budget.

## Problem

Analysis of the 2026-04-04 production log revealed 6 distinct bugs, 2 critical, 2 moderate, and 2 low severity. The user reported that the car should have charged more on solar in the morning and had to manually reset one device.

### Bug 1: `adapt_repartition` IndexError on empty `power_sorted_cmds` (CRITICAL)

**File:** `home_model/constraints.py:1472`
**Frequency:** 16 occurrences + crashes `force_update_all` on user actions

`adapt_power_steps_budgeting()` can return an empty list — both via the explicit guard at line 1258 (`for_add=False` and `is_current_slot_empty=True`) and via `adapt_power_steps_budgeting_low_level` when all commands exceed the available amps budget. When `j = 0` is set at line 1413 (adding energy to an empty slot) or via `_get_lower_consign_idx_for_power`/`_get_higher_consign_idx_for_power`, `power_sorted_cmds[j]` at line 1472 raises `IndexError`.

The four guards in `compute_best_period_repartition` (lines 1791, 1975, 2078, 2235) already handle this pattern with `if len(power_sorted_cmds) == 0: continue`. The same guard is missing in `adapt_repartition`.

### Bug 2: NoneType comparison in `apply_budget_strategy` (CRITICAL)

**File:** `ha_model/charger.py:1823`
**Frequency:** 2,360 occurrences (every ~7s for all 3 wallboxes)

`QSChargerStatus.__init__` initializes `current_real_max_charging_amp` and `budgeted_amp` to `None` (lines 324, 328). `get_amps_from_values()` passes these `None` values through, producing lists like `[None, None, None]`. The `max()` call at line 1823 then fails with `TypeError: '>' not supported between instances of 'NoneType' and 'int'`.

This blocks ALL charger budgeting, directly preventing proper solar charging budget allocation — the likely cause of insufficient solar charging reported by the user.

### Bug 3: False "expected to be charging but no power detected" error (MODERATE)

**File:** `ha_model/charger.py:4667-4700`
**Frequency:** 5 occurrences (every ~21 min for ID.buzz)

The HA entity `number.id_buzz_battery_target_charge_level` was reported as unavailable. When the car target entity is unavailable, the constraint's `target_value` may be stale, causing the car to appear far from full when it's actually fully charged. After 600s with no power flow, the code triggers false error notifications and sets `DEVICE_STATUS_CHANGE_ERROR`.

### Bug 4: "no power sorted commands" in constraint repartition (MODERATE)

**File:** `home_model/constraints.py` (lines 1855, 2056, 2286)
**Frequency:** 119 occurrences

During `compute_best_period_repartition`, the `power_sorted_cmds` list is empty for some constraint types whose time window has been reduced (e.g., deadline passed for "today 08:00" leaving 0 valid slots). The code logs at ERROR level when this is expected behavior for expired constraints.

### Bug 5: Battery `probe_if_command_set` returns None at startup (LOW)

**File:** `ha_model/battery.py:112`
**Frequency:** 1 occurrence (startup only)

Startup race condition — the entity hasn't reported a value yet when the probe runs. The code already handles this defensively (returns `None`). The warning log is unnecessarily noisy.

### Bug 6: Car amp/power graph inconsistency (LOW)

**File:** `ha_model/car.py:1631`
**Frequency:** 1 occurrence for Twingo

External data quality issue — Twingo reports decreasing power when amperage increases. The code correctly rejects this data point. No code fix needed.

## Acceptance Criteria

1. **Bug 1 (IndexError):** `adapt_repartition` never crashes on empty `power_sorted_cmds`. When the list is empty, the slot is skipped with `continue`.
2. **Bug 2 (NoneType):** `apply_budget_strategy` handles charger statuses with `None` amp or phase values. Chargers with uninitialized values are skipped or defaulted to safe values.
3. **Bug 3 (false error):** The zero-power-detected check skips or reduces severity when the car target entity is unavailable/stale, preventing false error notifications.
4. **Bug 4 (noisy errors):** The "no power sorted commands" log is reduced from ERROR to WARNING (or INFO) when it occurs for constraints whose deadline has already passed, since this is expected behavior.
5. **Bug 5 (startup warning):** The battery `probe_if_command_set` warning is reduced to DEBUG during the startup grace window.
6. **Bug 6 (data quality):** No code change required (already handled correctly). Optionally reduce per-occurrence log noise.
7. **No regressions:** All existing tests pass and 100% test coverage is maintained.

## Priority Order

1. **Bug 2** — most frequent, blocks all solar charging optimization
2. **Bug 1** — crashes solver, cascades to user-facing errors
3. **Bug 3** — sends false notifications, user confusion
4. **Bug 4** — noisy errors for expired constraints
5. **Bug 5** — startup log noise
6. **Bug 6** — optional, already handled

## Tasks / Subtasks

- [ ] Task 1: Fix Bug 2 — NoneType in charger budgeting (AC: #2)
  - [ ] 1.1 In `QSChargerStatus.get_current_charging_amps()` (`charger.py:370`): guard against `current_real_max_charging_amp` or `current_active_phase_number` being `None` — return `[0.0, 0.0, 0.0]` early
  - [ ] 1.2 In `QSChargerStatus.get_budget_amps()` (`charger.py:373`): guard against `budgeted_amp` or `budgeted_num_phases` being `None` — return `[0.0, 0.0, 0.0]` early
  - [ ] 1.3 Add tests: `get_current_charging_amps()` returns zero amps when `current_real_max_charging_amp` is `None`
  - [ ] 1.4 Add tests: `get_budget_amps()` returns zero amps when `budgeted_amp` is `None`
  - [ ] 1.5 Add tests: `apply_budget_strategy` completes without error when charger status has `None` values

- [ ] Task 2: Fix Bug 1 — IndexError in `adapt_repartition` (AC: #1)
  - [ ] 2.1 In `adapt_repartition` (`constraints.py`): add `if len(power_sorted_cmds) == 0: continue` immediately after the `adapt_power_steps_budgeting` call at line 1373-1380, before any indexing into `power_sorted_cmds`
  - [ ] 2.2 Add test: `adapt_repartition` with empty `power_sorted_cmds` (mock `adapt_power_steps_budgeting` to return empty list) does not crash and returns correctly
  - [ ] 2.3 Add test: `adapt_repartition` skips slots where budget constraints eliminate all commands

- [ ] Task 3: Fix Bug 3 — false no-power-detected error (AC: #3)
  - [ ] 3.1 In `update_value_callback` (`charger.py:4667-4700`): before the 600s zero-power check, verify that the car target entity is available. If unavailable, skip the check or log at WARNING instead of ERROR and do NOT trigger `DEVICE_STATUS_CHANGE_ERROR`
  - [ ] 3.2 Add test: zero-power check is skipped when target entity is unavailable
  - [ ] 3.3 Add test: zero-power check still fires normally when target entity is available

- [ ] Task 4: Fix Bug 4 — reduce log severity for expired constraints (AC: #4)
  - [ ] 4.1 In `compute_best_period_repartition` (`constraints.py` lines 1855, 2056, 2286): change `_LOGGER.error` to `_LOGGER.warning` (or `_LOGGER.info` for the expired-deadline case)
  - [ ] 4.2 Verify existing tests still pass with log level change

- [ ] Task 5: Fix Bug 5 — reduce battery startup warning (AC: #5)
  - [ ] 5.1 In `probe_if_command_set` (`battery.py:106-118`): change `_LOGGER.warning` to `_LOGGER.debug` for the `None` return cases at startup
  - [ ] 5.2 Verify existing tests still pass

- [ ] Task 6: Quality gate (AC: #7)
  - [ ] 6.1 Run `python scripts/qs/quality_gate.py` — pytest 100% coverage + ruff + mypy + translations

## Dev Notes

### Architecture and source tree

Bug fixes span both layers:
- **Domain layer** (`home_model/`): Bug 1 (constraints.py), Bug 4 (constraints.py)
- **HA layer** (`ha_model/`): Bug 2 (charger.py), Bug 3 (charger.py), Bug 5 (battery.py)
- **Bug 6**: No change needed (car.py already correct)

This is valid — bugs 1 and 4 are pure solver logic, bugs 2, 3, 5 are HA-bridge issues.

### Files to modify

| File | What changes |
|------|-------------|
| `custom_components/quiet_solar/home_model/constraints.py` | Bug 1: empty-list guard in `adapt_repartition`; Bug 4: log level change in `compute_best_period_repartition` |
| `custom_components/quiet_solar/ha_model/charger.py` | Bug 2: None guards in `get_current_charging_amps`/`get_budget_amps`; Bug 3: entity availability check before zero-power error |
| `custom_components/quiet_solar/ha_model/battery.py` | Bug 5: log level change |
| `tests/test_constraints.py` (or similar) | Tests for Bug 1 empty-list guard |
| `tests/ha_tests/test_charger.py` (or similar) | Tests for Bug 2 None guards, Bug 3 entity check |

### Key implementation details

**Bug 1 fix pattern** — Follow the existing guards in `compute_best_period_repartition`:
```python
power_sorted_cmds, is_current_empty_command, possible_power_piloted_delta = (
    self.adapt_power_steps_budgeting(...)
)
if len(power_sorted_cmds) == 0:
    continue
```

**Bug 2 fix pattern** — Guard at the source in `get_amps_from_values` methods:
```python
def get_current_charging_amps(self) -> list[float | int]:
    if self.current_real_max_charging_amp is None or self.current_active_phase_number is None:
        return [0.0, 0.0, 0.0]
    return self.get_amps_from_values(self.current_real_max_charging_amp, self.current_active_phase_number)
```

**Bug 3 — entity availability check:** The charger object should have access to the car's target entity state. Check if the entity is `unavailable` or `unknown` before concluding there's a charging error.

**Bug 4 — log levels:** The "no power sorted commands" is expected when a constraint's deadline has passed. Reducing to WARNING avoids alarming log output for normal operation.

**Bug 5 — startup log:** The `None` values at startup are a known race condition. DEBUG is appropriate since the code already handles it defensively.

### What NOT to change

- Do NOT modify `home_model/` imports — domain boundary stays clean
- Do NOT change `adapt_power_steps_budgeting` or `adapt_power_steps_budgeting_low_level` — fix at the call sites
- Do NOT change `QSChargerStatus.__init__` default values — `None` correctly represents "not yet initialized"
- Do NOT modify solver step size or constraint evaluation logic
- Do NOT change `_add_to_amps_power_graph` (Bug 6) — already correct
- Do NOT change the `DEVICE_STATUS_CHANGE_ERROR` mechanism itself — only guard the trigger condition

### Testing patterns

For Bug 1 tests, use the factory-based approach from `tests/factories.py`:
- `create_constraint()` to create constraints with `adapt_power_steps_budgeting` that returns empty lists
- `MinimalTestLoad` / `MinimalTestHome` for lightweight test doubles

For Bug 2 tests, use `create_charger_group()` and `create_state_cmd()` from factories.
Set `current_real_max_charging_amp = None` and verify `get_current_charging_amps()` returns `[0.0, 0.0, 0.0]`.

For Bug 3 tests, mock the car target entity as unavailable and verify the error notification is not sent.

### References

- [Source: constraints.py:1472] IndexError crash site for Bug 1
- [Source: constraints.py:1257-1258] Empty list return in `adapt_power_steps_budgeting`
- [Source: constraints.py:1165-1205] `adapt_power_steps_budgeting_low_level` can also return empty
- [Source: constraints.py:1791,1975,2078,2235] Existing guard pattern for empty `power_sorted_cmds`
- [Source: charger.py:318-348] `QSChargerStatus.__init__` with `None` defaults
- [Source: charger.py:360-374] `get_amps_from_values`, `get_current_charging_amps`, `get_budget_amps`
- [Source: charger.py:1797-1823] `apply_budget_strategy` crash site for Bug 2
- [Source: charger.py:4667-4700] False error notification site for Bug 3
- [Source: constraints.py:1855,2056,2286] "no power sorted commands" log sites for Bug 4
- [Source: battery.py:100-125] `probe_if_command_set` startup warning for Bug 5
- [Source: car.py:1615-1643] `_add_to_amps_power_graph` data rejection for Bug 6

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References

### Completion Notes List

### File List
