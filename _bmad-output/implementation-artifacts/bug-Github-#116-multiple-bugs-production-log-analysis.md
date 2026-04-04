# Bug Fix: Multiple bugs found in 2026-04-04 production log analysis

Status: todo
issue: 116
branch: "QS_116"

## Story

As a Quiet Solar user with multiple chargers, a battery, and several on/off loads,
I want the solver and charger budgeting to handle edge cases (empty command lists, None amp values, unavailable entities, expired constraints, warning spam) gracefully,
so that the system does not crash, spam errors, or fail to allocate solar charging budget.

## Problem

Merged analysis from two independent reviews of the 2026-04-04 HA log revealed **9 bugs** (2 critical crashers, 2 moderate, 3 log-spam issues, 2 minor code issues). The critical TypeError crash loop blocked all charger constraint updates from 09:13 to 12:37, preventing solar-based charging for ~3.5 hours during peak morning sun.

### Timeline

- **00:39** — HA restart; minor startup warnings (lazy safe values, battery sensor not ready)
- **07:40** — "Serviette Parents" constraint error begins (every ~7s)
- **09:13** — **TypeError crash loop begins** in `apply_budget_strategy` for all 3 wallboxes (every ~7s, ~1180 total)
- **10:08-12:30** — "ID.buzz expected to be charging but no power detected" (5 times, downstream of Bug 1)
- **11:00, 12:13** — **IndexError crash** in solver `adapt_repartition` (16 total, also triggered via `force_update_all` from button/switch/select UI actions)
- **12:37** — NoneType errors end (likely user reset)
- **17:46** — "Cumulus Parents" constraint error begins

---

### Bug 1: TypeError in `apply_budget_strategy` — NoneType `budgeted_amp` (CRITICAL)

**File:** `ha_model/charger.py:1823`
**Frequency:** ~2,360 occurrences (every ~7s for all 3 wallboxes)
**Error:** `TypeError: '>' not supported between instances of 'NoneType' and 'int'`

**Root cause:** The "Fix D" cooldown filter at `charger.py:914-927` removes chargers from `budget_chargers` during their amp-change cooldown period. But `budgeting_algorithm_minimize_diffs` (called with `budget_chargers`) is the only place that sets `budgeted_amp` via `_do_prepare_budgets_for_algo`. Then `apply_budget_strategy` at line 957 is called with the full `actionable_chargers` list. Chargers excluded from budgeting still have `budgeted_amp = None` (from `__init__` at line 328), so `get_budget_amps()` returns `[None, None, None]`, and `max(curr_amps[i], budget_amps[i])` at line 1823 crashes.

Key detail: `curr_amps[i]` (from `current_real_max_charging_amp`) is always a number (defaulted to 0 at lines 2349-2351 if charging is off). Only `budget_amps[i]` is ever None.

```python
# charger.py line 1823 — the crash site
max_curr_amps = [max(curr_amps[i], budget_amps[i]) for i in range(3)]
#                     ^^^ float       ^^^ None => TypeError
```

**Impact:** Blocks ALL charger budgeting, directly preventing proper solar charging budget allocation — the root cause of insufficient solar charging reported by the user.

---

### Bug 2: IndexError in solver `adapt_repartition` (CRITICAL)

**File:** `home_model/constraints.py:1472`
**Frequency:** 16 crashes + crashes `force_update_all` on user UI actions (buttons, switches, selects)
**Error:** `IndexError: list index out of range` at `base_cmd = power_sorted_cmds[j]`

**Root cause:** `adapt_power_steps_budgeting()` can return an empty list — both via the explicit guard at line 1258 (`for_add=False` and `is_current_slot_empty=True`) and via `adapt_power_steps_budgeting_low_level` when all commands exceed the available amps budget. Two paths set `j = 0` without checking the list is non-empty:
- Line 1413: `j = 0` when `current_command_power == 0` and adding energy
- Line 1468: `j = 0` in the reduction path fallthrough ("go to the minimum power load")

Both then hit `power_sorted_cmds[j]` at line 1472 which crashes.

The four guards in `compute_best_period_repartition` (lines 1791, 1975, 2078, 2235) already handle this pattern with `if len(power_sorted_cmds) == 0: continue`. The same guard is missing in `adapt_repartition`.

---

### Bug 3: "no power sorted commands" in constraint repartition + latent bug (MODERATE)

**File:** `home_model/constraints.py` (lines 1855, 2056, 2286) and `adapt_power_steps_budgeting_low_level` (lines 1165-1215)
**Frequency:** 119 occurrences
**Devices:** Serviette Parents, Cumulus Parents, Clim Cuisine

**Root cause:** `adapt_power_steps_budgeting_low_level` returns an empty list when `available_amps_for_group` is too restrictive for any valid command. The filter loop at lines 1193-1206 walks `_power_sorted_cmds` from the top and marks every step as "too large", reducing `last_cmd_idx_ok` all the way to -1. Examples:
- **Serviette Parents** at 07:40: constraint window pushed to the past (06:00 UTC), remaining slot budget is 0 W
- **Cumulus Parents** at 17:52: solar production declining, all available power consumed by other loads

**Latent bug:** When `last_cmd_idx_ok < 0`, there is no explicit `return []` in `adapt_power_steps_budgeting_low_level` — the function falls through to `return out_sorted_commands`. While `out_sorted_commands` is initialized to `[]` at line 1173 so this works by accident, an explicit early return would make the intent clear and prevent regressions.

The log at ERROR level is wrong — this is an unsatisfiable constraint condition, not a code error.

---

### Bug 4: False "expected to be charging but no power detected" error (MODERATE — partly downstream of Bug 1)

**File:** `ha_model/charger.py:4667-4700`
**Frequency:** 5 occurrences (every ~21 min for ID.buzz)

**Root cause:** Partly downstream of Bug 1 — when the TypeError blocks all constraint updates, the charger's command pipeline is stuck. Additionally, the HA entity `number.id_buzz_battery_target_charge_level` was reported as unavailable, meaning the constraint's `target_value` may be stale. After 600s with no power flow, the code triggers false error notifications and sets `DEVICE_STATUS_CHANGE_ERROR`.

**Fix approach:** Fix Bug 1 first (primary cause). Additionally, add an entity availability check before the zero-power error — if the car target entity is unavailable, skip the check or log at a reduced level.

---

### Bug 5: `is_person_covered: False` warning spam (LOG SPAM)

**File:** `ha_model/charger.py:3561-3564`
**Frequency:** ~11,822 warnings (Zoe) + ~4,539 (Twingo)

**Root cause:** Two related issues:
1. **No rate-limiting:** the check fires every cycle (~7s) emitting the same warning for the same `(car, person, next_usage_time)` triplet, triggering HA's "logging too frequently" throttle which masks real errors from other modules.
2. **`CHARGE_TIME_CONSTRAINTS_CLEARED` context:** when the user explicitly clears a charge constraint, the system correctly won't re-add a mandatory grid-charging constraint for that trip. But it still detects that the car is NOT covered and warns every cycle.

---

### Bug 6: "REJECTING unauthorized assignment" warning spam (LOG SPAM)

**File:** `ha_model/home.py:2568-2573`
**Frequency:** ~206 warnings (every ~3 min)

**Root cause:** `scipy.linear_sum_assignment` always produces a complete matching — every person is assigned exactly one car. When a person has no authorized car currently available, the optimizer assigns the least-bad unauthorized car (at cost 1e12). The post-filter at line 2568 correctly rejects it, but the WARNING fires every cycle.

---

### Bug 7: Python 2-style `except` syntax (CODE BUG)

**Files:** `ha_model/home.py` (lines 194, 852, 4387)
**Error:** `except ValueError, TypeError:` should be `except (ValueError, TypeError):`

**Root cause:** In Python 3, `except ValueError, TypeError:` catches only `ValueError` and binds the exception instance to the name `TypeError` (shadowing the builtin). If `float()` receives `None`, the `TypeError` would propagate uncaught. This is a real bug, not just a style issue.

---

### Bug 8: Startup "Error loading lazy safe value" warning spam (LOG SPAM)

**File:** `ha_model/home.py:4389, 4409`
**Frequency:** 28 sensors at every boot

**Root cause:** Fires for every forecast sensor at boot when state is `UNAVAILABLE`/`UNKNOWN`. This is always expected at startup. Should be DEBUG.

---

### Bug 9: Battery `probe_if_command_set` returns None at startup (LOW)

**File:** `ha_model/battery.py:106-118`
**Frequency:** 1 occurrence (startup only)

Startup race condition — entity hasn't reported a value yet. Code already handles defensively. WARNING is unnecessarily noisy.

## Acceptance Criteria

1. **Bug 1 (NoneType):** `apply_budget_strategy` never crashes. Cooldown-excluded chargers have their `budgeted_amp` initialized to their current amp before budgeting runs. Defensive guard in `get_budget_amps()`/`get_current_charging_amps()` returns `[0.0, 0.0, 0.0]` for `None` values.
2. **Bug 2 (IndexError):** `adapt_repartition` never crashes on empty `power_sorted_cmds`. When the list is empty, the slot is skipped.
3. **Bug 3 (empty commands):** `adapt_power_steps_budgeting_low_level` has an explicit `return []` for the `last_cmd_idx_ok < 0` case. Log downgraded from ERROR to WARNING.
4. **Bug 4 (false error):** The zero-power-detected check skips or reduces severity when the car target entity is unavailable/stale. With Bug 1 fixed, most occurrences should resolve.
5. **Bug 5 (person warning):** The `is_person_covered: False` warning is rate-limited to once per `(car, person, next_usage_time)` triplet, then suppressed until state changes.
6. **Bug 6 (unauthorized):** Persons with no authorized car in the current set are pre-filtered before the optimizer. Post-filter rejection log downgraded to DEBUG.
7. **Bug 7 (except syntax):** All 3 occurrences use correct `except (ValueError, TypeError):` tuple syntax.
8. **Bug 8 (lazy safe):** Startup lazy safe value warnings downgraded to DEBUG.
9. **Bug 9 (battery startup):** Battery `probe_if_command_set` warnings downgraded to DEBUG.
10. **No regressions:** All existing tests pass and 100% test coverage is maintained.

## Priority Order

1. **Bug 1** — most frequent (~2,360), blocks all solar charging optimization for hours
2. **Bug 2** — crashes solver (16), cascades to user-facing errors
3. **Bug 7** — actual code bug (wrong exceptions caught), quick fix
4. **Bug 3** — 119 noisy errors for unsatisfiable constraints + latent bug
5. **Bug 4** — false notifications (mostly downstream of Bug 1)
6. **Bug 5** — ~16,361 warning spam, masks other errors
7. **Bug 6** — ~206 warning spam
8. **Bug 8** — 28 startup warnings
9. **Bug 9** — 1 startup warning

## Tasks / Subtasks

- [ ] Task 1: Fix Bug 1 — NoneType in charger budgeting (AC: #1)
  - [ ] 1.1 In `dyn_handle` cooldown filter (`charger.py:914-927`): when excluding a charger from `budget_chargers`, initialize its `budgeted_amp` to `cs.current_real_max_charging_amp or 0` and `budgeted_num_phases` to `cs.current_active_phase_number or (3 if cs.charger.physical_3p else 1)` — this ensures the excluded charger has a valid budget representing its actual state
  - [ ] 1.2 In `QSChargerStatus.get_current_charging_amps()` (`charger.py:370`): defensive guard — return `[0.0, 0.0, 0.0]` if `current_real_max_charging_amp is None` or `current_active_phase_number is None`
  - [ ] 1.3 In `QSChargerStatus.get_budget_amps()` (`charger.py:373`): defensive guard — return `[0.0, 0.0, 0.0]` if `budgeted_amp is None` or `budgeted_num_phases is None`
  - [ ] 1.4 Add tests: cooldown-excluded charger gets `budgeted_amp` set to current amp
  - [ ] 1.5 Add tests: `get_current_charging_amps()` returns zero amps when values are `None`
  - [ ] 1.6 Add tests: `get_budget_amps()` returns zero amps when values are `None`
  - [ ] 1.7 Add tests: `apply_budget_strategy` completes without error when charger status has cooldown-initialized values

- [ ] Task 2: Fix Bug 2 — IndexError in `adapt_repartition` (AC: #2)
  - [ ] 2.1 In `adapt_repartition` (`constraints.py`): add `if len(power_sorted_cmds) == 0: continue` immediately after the `adapt_power_steps_budgeting` call at line 1373-1380, before the branching logic that sets `j`
  - [ ] 2.2 Add test: `adapt_repartition` with empty `power_sorted_cmds` does not crash and returns correctly
  - [ ] 2.3 Add test: `adapt_repartition` skips slots where budget constraints eliminate all commands

- [ ] Task 3: Fix Bug 7 — Python 2-style except syntax (AC: #7)
  - [ ] 3.1 Fix `home.py:194`: change `except ValueError, TypeError, IndexError:` to `except (ValueError, TypeError, IndexError):`
  - [ ] 3.2 Fix `home.py:852`: change `except ValueError, TypeError, KeyError:` to `except (ValueError, TypeError, KeyError):`
  - [ ] 3.3 Fix `home.py:4387`: change `except ValueError, TypeError:` to `except (ValueError, TypeError):`
  - [ ] 3.4 Add tests: verify `TypeError` is caught correctly (e.g., `float(None)` → `TypeError` caught, not propagated)

- [ ] Task 4: Fix Bug 3 — empty power sorted commands + latent bug (AC: #3)
  - [ ] 4.1 In `adapt_power_steps_budgeting_low_level` (`constraints.py:1208-1215`): add explicit `if last_cmd_idx_ok < 0: return []` before the fallthrough
  - [ ] 4.2 In `compute_best_period_repartition` (`constraints.py` lines 1855, 2056, 2286): downgrade `_LOGGER.error` to `_LOGGER.warning` for the "no power sorted commands" messages
  - [ ] 4.3 Add test: `adapt_power_steps_budgeting_low_level` returns empty list when all commands exceed budget
  - [ ] 4.4 Verify existing tests still pass

- [ ] Task 5: Fix Bug 4 — false no-power-detected error (AC: #4)
  - [ ] 5.1 In `update_value_callback` (`charger.py:4667-4700`): before the 600s zero-power check, verify that the car target entity is available. If unavailable, skip the check or log at WARNING instead of ERROR and do NOT trigger `DEVICE_STATUS_CHANGE_ERROR`
  - [ ] 5.2 Add test: zero-power check is skipped when target entity is unavailable
  - [ ] 5.3 Add test: zero-power check still fires normally when target entity is available

- [ ] Task 6: Fix Bug 5 — is_person_covered warning spam (AC: #5)
  - [ ] 6.1 In charger constraint handling (`charger.py:3561-3564`): rate-limit the `is_person_covered: False` warning — emit once per `(car, person, next_usage_time)` triplet, then suppress until state changes. Use an instance-level dict to track already-warned triplets
  - [ ] 6.2 Also suppress or downgrade to DEBUG once `next_usage_time` is in the past
  - [ ] 6.3 Add test: warning emitted once for a given triplet, then suppressed
  - [ ] 6.4 Add test: warning re-emitted when triplet changes

- [ ] Task 7: Fix Bug 6 — REJECTING unauthorized assignment spam (AC: #6)
  - [ ] 7.1 In `compute_and_set_best_persons_cars_allocations` or `get_best_persons_cars_allocations` (`home.py`): pre-filter `p_s` to exclude persons who have zero authorized cars in the current `c_s` set before building the cost matrix
  - [ ] 7.2 Downgrade the post-filter rejection log from `WARNING` to `DEBUG` — after the pre-filter, this path is only reachable in race conditions
  - [ ] 7.3 Add test: persons with no authorized car are excluded from optimization
  - [ ] 7.4 Add test: post-filter rejection log fires at DEBUG

- [ ] Task 8: Fix Bug 8 and Bug 9 — startup log noise (AC: #8, #9)
  - [ ] 8.1 In `home.py:4389, 4409`: downgrade "Error loading lazy safe value" from `_LOGGER.warning` to `_LOGGER.debug`
  - [ ] 8.2 In `battery.py:106-118`: downgrade `probe_if_command_set` `_LOGGER.warning` to `_LOGGER.debug` for the `None` return cases
  - [ ] 8.3 Verify existing tests still pass

- [ ] Task 9: Quality gate (AC: #10)
  - [ ] 9.1 Run `python scripts/qs/quality_gate.py` — pytest 100% coverage + ruff + mypy + translations

## Dev Notes

### Architecture and source tree

Bug fixes span both layers:
- **Domain layer** (`home_model/`): Bug 2 (constraints.py), Bug 3 (constraints.py)
- **HA layer** (`ha_model/`): Bug 1 (charger.py), Bug 4 (charger.py), Bug 5 (charger.py), Bug 6 (home.py), Bug 7 (home.py), Bug 8 (home.py), Bug 9 (battery.py)

This is valid — bugs 2 and 3 are pure solver logic, all others are HA-bridge issues.

### Files to modify

| File | What changes |
|------|-------------|
| `custom_components/quiet_solar/ha_model/charger.py` | Bug 1: cooldown filter init + defensive guards in `get_*_amps()`; Bug 4: entity availability check before zero-power error; Bug 5: rate-limit `is_person_covered` warning |
| `custom_components/quiet_solar/home_model/constraints.py` | Bug 2: empty-list guard in `adapt_repartition`; Bug 3: explicit return in `adapt_power_steps_budgeting_low_level` + log downgrade in `compute_best_period_repartition` |
| `custom_components/quiet_solar/ha_model/home.py` | Bug 6: pre-filter persons + downgrade rejection log; Bug 7: fix except syntax (3 sites); Bug 8: downgrade lazy safe value log |
| `custom_components/quiet_solar/ha_model/battery.py` | Bug 9: downgrade startup warning |
| Test files (existing or new) | Tests for Bugs 1-7 |

### Key implementation details

**Bug 1 — Primary fix at cooldown filter site (`charger.py:914-927`):**
```python
for cs in actionable_chargers:
    if (
        cs.charger._last_amp_change_time is not None
        and (time - cs.charger._last_amp_change_time).total_seconds() < CHARGER_ADAPTATION_WINDOW_S
    ):
        # Keep current amp as budget so apply_budget_strategy won't crash
        cs.budgeted_amp = cs.current_real_max_charging_amp or 0
        cs.budgeted_num_phases = cs.current_active_phase_number or (3 if cs.charger.physical_3p else 1)
        _LOGGER.info("dyn_handle: skipping %s for budgeting, amp change cooldown ...", cs.name)
    else:
        budget_chargers.append(cs)
```
The defensive guard in `get_budget_amps()`/`get_current_charging_amps()` (returning `[0.0, 0.0, 0.0]` for `None`) is a second line of defense, NOT sufficient alone — the budget for a cooled-down charger should represent its actual state, not zero.

**Bug 2 — Guard pattern:** Follow `compute_best_period_repartition`. Insert the guard immediately after the `adapt_power_steps_budgeting` call, before the if/else branching that sets `j`:
```python
power_sorted_cmds, is_current_empty_command, possible_power_piloted_delta = (
    self.adapt_power_steps_budgeting(...)
)
if len(power_sorted_cmds) == 0:
    continue
```

**Bug 3 — Explicit return in `adapt_power_steps_budgeting_low_level`:**
```python
if last_cmd_idx_ok == len(self._power_sorted_cmds) - 1:
    return self._power_sorted_cmds
elif last_cmd_idx_ok >= 0:
    out_sorted_commands = self._power_sorted_cmds[: last_cmd_idx_ok + 1]
else:
    return []  # all commands exceed budget — explicit empty return
return out_sorted_commands
```

**Bug 5 — Rate-limiting pattern:** Add an instance-level dict (e.g., `_warned_person_triplets: dict[tuple, datetime]`) to track already-warned `(car.name, person.name, next_usage_time)` triplets. Clear entries when the triplet state changes.

**Bug 6 — Pre-filter pattern:**
```python
c_names = {c.name for c in c_s}
p_s = [
    (person, leave_time, mileage)
    for person, leave_time, mileage in p_s
    if any(c.name in c_names for c in person.get_authorized_cars())
]
```

**Bug 7 — except syntax:** `except ValueError, TypeError:` in Python 3 catches only `ValueError` and binds it to `TypeError`, shadowing the builtin. If `float(None)` is called, the `TypeError` propagates uncaught. Fix: `except (ValueError, TypeError):`.

### What NOT to change

- Do NOT modify `home_model/` imports — domain boundary stays clean
- Do NOT change `QSChargerStatus.__init__` default values — `None` correctly represents "not yet initialized"
- Do NOT modify solver step size or constraint evaluation logic
- Do NOT change `_add_to_amps_power_graph` — already correctly rejects inconsistent data
- Do NOT change the `DEVICE_STATUS_CHANGE_ERROR` mechanism itself — only guard the trigger condition
- Do NOT change `adapt_power_steps_budgeting` internals — fix at the call sites

### Testing patterns

For Bug 1 tests, use `create_charger_group()` and `create_state_cmd()` from `tests/factories.py`.
Set up a charger with `_last_amp_change_time` in cooldown, verify `budgeted_amp` is set before `apply_budget_strategy`.

For Bug 2 tests, use factory-based approach: `create_constraint()` + `MinimalTestLoad` / `MinimalTestHome`. Mock `adapt_power_steps_budgeting` to return empty list, verify no crash.

For Bug 5 tests, verify warning count: call the method multiple times with the same triplet, assert warning fires once.

For Bug 6 tests, set up persons with no authorized cars in current set, verify they're excluded from optimizer.

For Bug 7 tests, verify `float(None)` → `TypeError` is caught correctly.

### References

- [Source: charger.py:914-927] Cooldown filter (Bug 1 root cause)
- [Source: charger.py:957] `apply_budget_strategy` called with full `actionable_chargers`
- [Source: charger.py:1823] `max(curr_amps[i], budget_amps[i])` crash site (Bug 1)
- [Source: charger.py:318-348] `QSChargerStatus.__init__` with `None` defaults
- [Source: charger.py:360-374] `get_amps_from_values`, `get_current_charging_amps`, `get_budget_amps`
- [Source: constraints.py:1472] IndexError crash site (Bug 2)
- [Source: constraints.py:1413,1468] Two paths where `j = 0` without empty check
- [Source: constraints.py:1257-1258] Empty list return in `adapt_power_steps_budgeting`
- [Source: constraints.py:1165-1215] `adapt_power_steps_budgeting_low_level` latent bug
- [Source: constraints.py:1791,1975,2078,2235] Existing guard pattern
- [Source: constraints.py:1855,2056,2286] "no power sorted commands" log sites (Bug 3)
- [Source: charger.py:4667-4700] False error notification (Bug 4)
- [Source: charger.py:3561-3564] `is_person_covered: False` warning (Bug 5)
- [Source: home.py:2568-2573] REJECTING unauthorized (Bug 6)
- [Source: home.py:194,852,4387] Python 2-style except syntax (Bug 7)
- [Source: home.py:4389,4409] Lazy safe value warnings (Bug 8)
- [Source: battery.py:106-118] `probe_if_command_set` startup warning (Bug 9)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References

### Completion Notes List

### File List
