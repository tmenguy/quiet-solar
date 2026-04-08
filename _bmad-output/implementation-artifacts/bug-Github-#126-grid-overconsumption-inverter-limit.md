# Story bug-Github-#126: Replace amp-based production guard with per-slot power guard

issue: 126
branch: "QS_126"

Status: ready-for-dev

## Story
As a Quiet Solar user with multiple controllable loads,
I want the solver to enforce a per-slot maximum production power guard in watts,
so that total consumed power from all controlled loads never exceeds what the home can actually produce at any given time slot.

## Background

On 2026-04-08 at 08:49, the solver simultaneously commanded two car chargers (7,137W + 6,076W) plus three cumulus heaters (~4,568W) totaling 17,781W — far exceeding the 12kW inverter limit. The `available_amps_production_for_group` guard (amp-based, per-phase) passed because per-phase amps were within limits, but total watts exceeded the inverter capacity. The system was stuck in over-consumption for 5 minutes until the next solver run.

Root cause: the existing production guard operates in per-phase amps, which cannot enforce a total-watt inverter limit. Non-charger loads (cumulus) don't participate in the amp budget at all.

## Core Design

**Remove** the amp-based production guard:
- `available_amps_production_for_group` (field + init + update)
- `use_production_limits` (parameter throughout constraint pipeline)
- `dyn_group_max_production_phase_current_for_budget` (property)
- `_get_home_max_production_phase_amps_for_budget` (method)

**Add** a per-slot power guard in the solver:
- `max_possible_production[slot]` — dynamic per slot, recomputed after each constraint allocation
- `_total_consumed_power[slot]` — explicit array tracking `ua + sum(controlled_loads)`, updated via a utility method
- `headroom[slot] = max_possible_production[slot] - _total_consumed_power[slot]`
- Utility method `_add_consumption_delta_power(delta_power)` updates BOTH `_available_power` and `_total_consumed_power` atomically — single point of truth, no drift

**Keep** `available_amps_for_group` (subscription/breaker per-phase amp guard — independent concern).

### max_possible_production Formula

**DC-coupled battery, not empty:**
```
max_possible_production[slot] = min(
    pv[slot] + max(0, battery_possible_discharge[slot] - max(0, battery_actual_discharge[slot])),
    max_inverter_dc_to_ac_power
)
```

**DC-coupled battery, empty:**
```
max_possible_production[slot] = min(pv[slot], max_inverter_dc_to_ac_power)
```

**AC-coupled battery, not empty:**
```
max_possible_production[slot] = min(pv[slot], max_inverter_dc_to_ac_power) +
    max(0, battery_possible_discharge[slot] - max(0, battery_actual_discharge[slot]))
```

**AC-coupled battery, empty (or no battery):**
```
max_possible_production[slot] = min(pv[slot], max_inverter_dc_to_ac_power)
```

Where:
- `pv[slot]` = `self._solar_production[slot]` (fixed, from init)
- `battery_possible_discharge[slot]` = SOC-limited max discharge, from `get_best_discharge_power` utils
- `battery_actual_discharge[slot]` = what battery is currently discharging, from `_battery_get_charging_power` output
- `max_inverter_dc_to_ac_power` = `self._max_inverter_dc_to_ac_power`
- Battery assumed switchable: if charging, it can stop and start discharging (remaining = full possible discharge)

### headroom Derivation

```
headroom[slot] = max_possible_production[slot] - _total_consumed_power[slot]
```

`_total_consumed_power` is an explicit array initialized with unavoidable consumption and updated via `_add_consumption_delta_power(delta_power)` — a utility method on `PeriodSolver` that atomically updates both `_available_power` and `_total_consumed_power` with the same delta. This ensures the two arrays never drift apart, even when `_available_power` is also modified by battery operations.

## Acceptance Criteria

1. **AC1** Given a DC-coupled 12kW inverter with solar=8kW, battery_possible_discharge=5kW, battery_actual_discharge=2kW, When the solver computes max_possible_production, Then it equals min(8+max(0,5-2), 12) = min(11,12) = 11kW
2. **AC2** Given two chargers + cumulus requesting combined 17.8kW in one slot with max_possible_production=12kW and ua=1kW, When the power guard triggers, Then non-mandatory filler/green commands are reduced until headroom >= 0
3. **AC3** Given `available_amps_production_for_group` is removed, When `adapt_power_steps_budgeting_low_level` is called for a green/filler constraint, Then it checks `cmd.power_consign <= headroom[slot]` instead of amp budget
4. **AC4** Given mandatory constraints, When the power guard evaluates them, Then mandatory loads bypass the production power guard (pass headroom=None). `_available_power` is still updated to reflect their consumption, so subsequent non-mandatory constraints see reduced headroom
5. **AC5** Given the power guard triggers, When it logs the decision, Then the log includes slot index, cmd power, headroom, and max_possible_production
6. **AC6** All removed references to `available_amps_production_for_group` and `use_production_limits` replaced, no dead code remains
7. **AC7** Existing solver/charger tests adapted and passing. New tests cover: power guard enforcement, multi-load capping, mandatory exemption, DC/AC coupling variants

## Tasks / Subtasks

- [ ] Task 1 (AC: 1) Per-slot power tracking and `max_possible_production` in the solver
  - [ ] 1.1 Add `_total_consumed_power` array to `PeriodSolver.__init__`, initialized from unavoidable consumption forecast (same source as `_available_power` init)
  - [ ] 1.2 Add `_add_consumption_delta_power(self, delta_power)` utility method on `PeriodSolver` that atomically updates both `self._available_power += delta_power` and `self._total_consumed_power += delta_power`. Every place that currently does `self._available_power = self._available_power + out_power` for load consumption must use this method instead
  - [ ] 1.3 Evolve `_battery_get_charging_power` to also output per-slot `battery_actual_discharge` and `battery_possible_discharge` (using existing `get_best_discharge_power` utils). Note: user will rework this method further — design for easy replacement
  - [ ] 1.4 Add `_compute_max_possible_production()` method to `PeriodSolver` implementing the DC/AC coupling formulas above. Takes `battery_actual_discharge[slot]`, `battery_possible_discharge[slot]` from step 1.3
  - [ ] 1.5 Compute and store `self._max_possible_production` after init (solar-only, no battery) and recompute after each call to `_battery_get_charging_power` (in `solve()`)
  - [ ] 1.6 In `_allocate_constraints` (solver.py:683), after each constraint allocation (after line 727), recompute battery state via `_battery_get_charging_power` and update `_max_possible_production`
  - [ ] 1.7 Compute per-slot headroom: `headroom[slot] = max_possible_production[slot] - _total_consumed_power[slot]` and pass to constraint allocation

- [ ] Task 2 (AC: 2,3,6) Replace `use_production_limits` with power headroom in constraints
  - [ ] 2.1 Add `max_slot_power_headroom: float | None` parameter to `adapt_power_steps_budgeting_low_level` (constraints.py:1165), replacing `use_production_limits`
  - [ ] 2.2 In `adapt_power_steps_budgeting_low_level`: when `max_slot_power_headroom is not None`, filter commands by `cmd.power_consign <= max_slot_power_headroom` instead of checking amps against `available_amps_production_for_group`
  - [ ] 2.3 Thread `max_slot_power_headroom` through `adapt_power_steps_budgeting` (line 1220) and `adapt_repartition` (line 1273)
  - [ ] 2.4 In `compute_best_period_repartition` (line 1727): receive per-slot headroom array as new parameter, pass `headroom[slot]` to `adapt_power_steps_budgeting` for each slot
  - [ ] 2.5 Update call site at line 1976 (`use_production_limits=True` in available-energy-only path)
  - [ ] 2.6 Remove `use_production_limits` parameter from all methods

- [ ] Task 3 (AC: 6) Remove `available_amps_production_for_group` infrastructure
  - [ ] 3.1 Remove `available_amps_production_for_group` attribute from `QSDynamicGroup.__init__` (dynamic_group.py:36)
  - [ ] 3.2 Remove production block from `update_available_amps_for_group` (dynamic_group.py:56-64)
  - [ ] 3.3 Remove `dyn_group_max_production_phase_current_for_budget` property (dynamic_group.py:88)
  - [ ] 3.4 Remove `from_father_production_budget` parameter and all its uses from `prepare_slots_for_amps_budget` (dynamic_group.py:209, 221-233, 241)
  - [ ] 3.5 Remove `_get_home_max_production_phase_amps_for_budget` method (home.py:1167-1177)
  - [ ] 3.6 Remove `dyn_group_max_production_phase_current_for_budget` property override (home.py:1265-1281)
  - [ ] 3.7 Remove off-grid production branch in `dyn_group_max_phase_current_for_budget` (home.py:1259-1260)
  - [ ] 3.8 Remove `from_father_production_budget` from `prepare_slots_for_amps_budget` in load.py

- [ ] Task 4 (AC: 4) Power guard respects constraint priority
  - [ ] 4.1 In `_allocate_constraints` (solver.py:683): for mandatory constraints (`c.is_mandatory`), pass `headroom=None` to `compute_best_period_repartition` (no production cap)
  - [ ] 4.2 In `adapt_power_steps_budgeting_low_level`: when `max_slot_power_headroom is None`, skip power guard (return full `_power_sorted_cmds`)
  - [ ] 4.3 Log warning when mandatory load causes total_consumption > max_possible_production

- [ ] Task 5 (AC: 5) Logging
  - [ ] 5.1 In `adapt_power_steps_budgeting_low_level`: log when power guard filters commands (slot, cmd power, headroom)
  - [ ] 5.2 In `_compute_max_possible_production`: log per-slot production breakdown (solar, battery, inverter cap)
  - [ ] 5.3 Use lazy `%s` formatting per project rules

- [ ] Task 6 (AC: 6,7) Update existing tests
  - [ ] 6.1 `tests/factories.py`: remove `available_amps_production_for_group` from `MinimalTestHome` (line 331), `create_charger_group` (line 551), `FatherDeviceStub` (lines 837, 870-872). Add power-guard-compatible helpers
  - [ ] 6.2 `tests/test_constraints.py`: replace 8 `use_production_limits=` call sites (lines 1507-1709) with `max_slot_power_headroom=` parameter
  - [ ] 6.3 `tests/test_solver.py`: update `test_adapt_power_steps_budgeting_low_level_production_limits` (line 2416) to use power headroom
  - [ ] 6.4 `tests/test_ha_dynamic_group.py`: remove production budget tests (lines 191-217, 688-714)
  - [ ] 6.5 `tests/test_charger_coverage_deep.py`: remove production budget setup (lines 107, 184)
  - [ ] 6.6 `tests/test_coverage_constraints.py`: remove production budget assignments (lines 2039, 2149)
  - [ ] 6.7 `tests/test_constraint_interaction_boundaries.py`: update `use_production_limits=False` calls (lines 448, 490)

- [ ] Task 7 (AC: 7) New tests
  - [ ] 7.1 Test `_compute_max_possible_production`: DC-coupled with/without battery, AC-coupled, empty battery, no battery, no inverter limit
  - [ ] 7.2 Test power guard: 2 chargers + cumulus exceeding inverter → filler commands capped by headroom
  - [ ] 7.3 Test mandatory bypass: mandatory load allocated despite exceeding production, subsequent filler sees reduced headroom
  - [ ] 7.4 Test `_add_consumption_delta_power`: verify `_available_power` and `_total_consumed_power` stay in sync after constraint allocations, and that battery-only modifications to `_available_power` don't affect `_total_consumed_power`
  - [ ] 7.5 Test per-phase amp guard (`available_amps_for_group`) still works independently alongside power guard

## Dev Notes

### Architecture Constraints
- Power guard logic lives in `home_model/solver.py` and `home_model/constraints.py` (pure Python, no HA imports)
- `available_amps_for_group` (subscription/breaker) is untouched — independent per-phase safety
- SOLVER_STEP_S = 900 — don't touch
- Logging: lazy `%s`, no f-strings in log calls, no periods at end
- All new constants in const.py

### Key Files (production)
| File | Changes |
|------|---------|
| `home_model/solver.py` | Add `_compute_max_possible_production`, recompute in `_allocate_constraints`, pass headroom |
| `home_model/constraints.py` | Replace `use_production_limits` with `max_slot_power_headroom` in `adapt_power_steps_budgeting*` and `adapt_repartition` |
| `home_model/battery.py` | Read-only dependency (utils: `get_best_discharge_power`, `is_dc_coupled`) |
| `ha_model/dynamic_group.py` | Remove `available_amps_production_for_group`, simplify `update_available_amps_for_group` and `prepare_slots_for_amps_budget` |
| `ha_model/home.py` | Remove `_get_home_max_production_phase_amps_for_budget`, `dyn_group_max_production_phase_current_for_budget` |
| `home_model/load.py` | Remove `from_father_production_budget` from `prepare_slots_for_amps_budget` |

### Key Files (tests) — 7 files
`factories.py`, `test_constraints.py`, `test_solver.py`, `test_ha_dynamic_group.py`, `test_charger_coverage_deep.py`, `test_coverage_constraints.py`, `test_constraint_interaction_boundaries.py`

### Important Design Decisions
- **Explicit `_total_consumed_power` array** — kept in sync with `_available_power` via `_add_consumption_delta_power()` utility. Not derived, because `_available_power` is also modified by battery operations that are NOT consumption
- **No `_max_production_power` as static array** — recomputed dynamically after each constraint via `_battery_get_charging_power`
- **Battery switchable** — if battery is charging, guard assumes it can stop charging and start discharging
- **`_battery_get_charging_power` will be reworked by user** — design for easy replacement of its output interface
- **Emergency 0A dyn_handle override** — deferred to separate story

## Adversarial Review Notes

**Reviewers:** Critic, Concrete Planner, Dev Agent Proxy
**Rounds:** 1 (with extensive user refinement)

### Key findings incorporated:
- [Critic] `_available_power` already tracks consumption — Partially accepted, but user overrode: `_available_power` is also modified by battery ops, so an explicit `_total_consumed_power` with `_add_consumption_delta_power()` utility is needed to avoid drift
- [Concrete Planner] `-_available_power` after init doesn't include battery discharge → need separate `max_possible_production` computation — Accepted, led to DC/AC formula
- [Concrete Planner] Guard enforcement point must be in `adapt_power_steps_budgeting_low_level` with headroom threaded from solver — Accepted
- [Dev Proxy] `use_production_limits` has 6+ call sites across constraint pipeline — all enumerated in Task 2
- [All] Missing call site at constraints.py line 1976 — added to Task 2.5
- [Concrete Planner] Piloted device power must be included — inherited from existing `_merge_commands_slots_for_load` flow
- [Dev Proxy] Use np.float64 arrays consistent with solver — noted

### Decisions made:
- Emergency 0A dyn_handle override deferred to separate story — user requested removal of AC5/Task 5
- Battery discharge uses SOC-limited values (via `get_best_discharge_power`) — more accurate than hardware max
- Both DC and AC coupling handled in formula — user confirmed
- Battery assumed switchable (charge → discharge) — user confirmed
- Recompute battery state after each constraint allocation — user confirmed, for full accuracy

### Known risks acknowledged:
- `_battery_get_charging_power` called per-constraint in inner loop may be slow — correctness first, optimize later
- `_battery_get_charging_power` needs rework for better clamping (user will handle separately)
- Per-group production isolation lost when removing `available_amps_production_for_group` — home-level power guard replaces it
