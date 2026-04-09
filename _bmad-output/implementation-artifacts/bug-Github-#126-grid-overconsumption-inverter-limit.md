# Story bug-Github-#126: Replace amp-based production guard with per-slot power guard

issue: 126
branch: "QS_126"

Status: dev-complete

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

**DC-coupled battery:**
```
max_possible_production[slot] = min(
    pv[slot] + battery_possible_discharge[slot],
    max_inverter_dc_to_ac_power
)
```

**AC-coupled battery:**
```
max_possible_production[slot] = min(pv[slot], max_inverter_dc_to_ac_power) +
    battery_possible_discharge[slot]
```

**No battery:**
```
max_possible_production[slot] = min(pv[slot], max_inverter_dc_to_ac_power)
```

Where:
- `pv[slot]` = `self._solar_production[slot]` — **raw PV** (not `pv - ua`), stored from `create_power_slots` output
- `battery_possible_discharge[slot]` = SOC-limited max discharge, from unified `Battery.get_charger_power()` third return value
- `max_inverter_dc_to_ac_power` = `self._max_inverter_dc_to_ac_power`
- Battery assumed switchable: if charging, it can stop charging and start discharging (full possible discharge available)

### headroom Derivation

```
headroom[slot] = max_possible_production[slot] - _total_consumed_power[slot]
```

`_total_consumed_power` is an explicit array initialized with unavoidable consumption and updated via `_add_consumption_delta_power(delta_power)` — a utility method on `PeriodSolver` that atomically updates `_available_power`, `_available_power_no_battery`, and `_total_consumed_power` with the same delta. `_available_power_no_battery` provides a clean baseline (without battery ops) for `_battery_get_charging_power`, avoiding circular dependency.

## Acceptance Criteria

1. **AC1** Given a DC-coupled 12kW inverter with solar=8kW, battery_possible_discharge=5kW, When the solver computes max_possible_production, Then it equals min(8+5, 12) = 12kW
2. **AC2** Given two chargers + cumulus requesting combined 17.8kW in one slot with max_possible_production=12kW and ua=1kW, When the power guard triggers, Then non-mandatory filler/green commands are reduced until headroom >= 0
3. **AC3** Given `available_amps_production_for_group` is removed, When `adapt_power_steps_budgeting_low_level` is called for a green/filler constraint, Then it checks `cmd.power_consign <= headroom[slot]` instead of amp budget
4. **AC4** Given mandatory constraints, When the power guard evaluates them, Then mandatory loads bypass the production power guard (pass headroom=None). `_available_power` is still updated to reflect their consumption, so subsequent non-mandatory constraints see reduced headroom
5. **AC5** Given the power guard triggers, When it logs the decision, Then the log includes slot index, cmd power, headroom, and max_possible_production
6. **AC6** All removed references to `available_amps_production_for_group` and `use_production_limits` replaced, no dead code remains
7. **AC7** Existing solver/charger tests adapted and passing. New tests cover: power guard enforcement, multi-load capping, mandatory exemption, DC/AC coupling variants

## Tasks / Subtasks

- [x] Task 1 (AC: 1) Per-slot power tracking and `max_possible_production` in the solver
  - [x] 1.1 Add `_total_consumed_power` array to `PeriodSolver.__init__`, initialized from unavoidable consumption forecast
  - [x] 1.2 Add `_add_consumption_delta_power(self, delta_power)` utility method — atomically updates both `_available_power` and `_total_consumed_power`
  - [x] 1.3 Refactored: unified `Battery.get_charger_power()` replaces separate charge/discharge methods; `_battery_get_charging_power` uses `_available_power_no_battery` and returns 8-tuple
  - [x] 1.4 Add `_compute_max_possible_production()` method implementing DC/AC coupling formulas
  - [x] 1.5 Compute `_max_possible_production` after init (solar-only) and recompute with battery data before first `_allocate_constraints` and after each allocation
  - [x] 1.6 In `_allocate_constraints`, recompute battery state and `_max_possible_production` after each constraint
  - [x] 1.7 Compute per-slot headroom and pass to constraint allocation

- [x] Task 2 (AC: 2,3,6) Replace `use_production_limits` with power headroom in constraints
  - [x] 2.1 Add `max_slot_power_headroom` parameter to `adapt_power_steps_budgeting_low_level`
  - [x] 2.2 Restructured: amp guard and headroom guard are independent. Either applies when its inputs are available. **Headroom applies even without father_device** (no amp guard needed for headroom to work)
  - [x] 2.3 Thread `max_slot_power_headroom` through `adapt_power_steps_budgeting` and `adapt_repartition`
  - [x] 2.4 In `compute_best_period_repartition`: receive headroom array, pass to slot-level budgeting
  - [x] 2.5 Update call site for available-energy-only path
  - [x] 2.6 Remove `use_production_limits` parameter from all methods

- [x] Task 3 (AC: 6) Remove `available_amps_production_for_group` infrastructure
  - [x] 3.1–3.8 All removed

- [x] Task 4 (AC: 4) Power guard respects constraint priority
  - [x] 4.1 In `_allocate_constraints`: headroom applied when `always_use_available_only_power or not c.is_mandatory` (matches `do_use_available_power_only` logic). Mandatory-only constraints bypass headroom unless always_use_available_only_power is set
  - [x] 4.2 Amp guard always active. Headroom guard independent — applies whenever `max_slot_power_headroom` is not None, regardless of father_device
  - [x] 4.3 Log warning when mandatory load exceeds production

- [x] Task 5 (AC: 5) Logging
  - [x] 5.1–5.3 All implemented with lazy `%s` formatting

- [x] Task 6 (AC: 6,7) Update existing tests
  - [x] 6.1–6.7 All updated

- [x] Task 7 (AC: 7) New tests (15 tests in `tests/test_power_guard.py`)
  - [x] 7.1 `_compute_max_possible_production`: DC/AC coupling, with/without battery, inverter clamping
  - [x] 7.2 Power guard filtering and multi-load capping by headroom
  - [x] 7.3 Mandatory bypass: mandatory allocated despite exceeding production, filler sees reduced headroom
  - [x] 7.4 `_add_consumption_delta_power` sync verification, battery independence
  - [x] 7.5 Per-phase amp guard works independently alongside power guard
  - [x] 7.6 Headroom guard without father_device (new — validates headroom applies even without dynamic group parent)

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
| `home_model/battery.py` | Unified `get_charger_power()` replacing separate charge/discharge methods; `charge_from_grid` property |
| `ha_model/dynamic_group.py` | Remove `available_amps_production_for_group`, simplify `update_available_amps_for_group` and `prepare_slots_for_amps_budget` |
| `ha_model/home.py` | Remove `_get_home_max_production_phase_amps_for_budget`, `dyn_group_max_production_phase_current_for_budget` |
| `home_model/load.py` | Remove `from_father_production_budget` from `prepare_slots_for_amps_budget` |

### Key Files (tests) — 7 files
`factories.py`, `test_constraints.py`, `test_solver.py`, `test_ha_dynamic_group.py`, `test_charger_coverage_deep.py`, `test_coverage_constraints.py`, `test_constraint_interaction_boundaries.py`

### Important Design Decisions
- **Explicit `_total_consumed_power` array** — kept in sync with `_available_power` and `_available_power_no_battery` via `_add_consumption_delta_power()` utility. Not derived, because `_available_power` is also modified by battery operations that are NOT consumption
- **No `_max_production_power` as static array** — recomputed dynamically after each constraint via `_battery_get_charging_power`
- **Battery switchable** — if battery is charging, guard assumes it can stop charging and start discharging
- **Battery refactored** — `_battery_get_charging_power` reworked: uses `_available_power_no_battery` (no circular dependency), unified `Battery.get_charger_power()`, CMD_GREEN_CHARGE_ONLY enforced via `min(0.0, available_power)` clamping, returns 8-tuple (removed `battery_actual_discharge`)
- **Emergency 0A dyn_handle override** — deferred to separate story
- **Headroom condition matches `do_use_available_power_only`** — `if always_use_available_only_power or not c.is_mandatory: headroom = ...` (user correction: not just `c.is_mandatory`)
- **`adapt_repartition` headroom not gated on `support_auto`** — `use_headroom = energy_delta >= 0.0 and power_headroom is not None` (user correction: non-auto loads should also be limited by headroom)
- **Headroom guard independent of amp guard** — `adapt_power_steps_budgeting_low_level` checks headroom even when `father_device` is None (user correction: no father doesn't exempt from production limits)
- **Battery-aware production before first allocation** — `_max_possible_production` recomputed with battery data before `_allocate_constraints` (not just after each allocation)
- **`_solar_production` is raw PV** — stored directly from `create_power_slots` pv_consumption output; `_available_power` uses `pv - ua` as before, but production formula uses raw PV to avoid double-counting UA
- **`_available_power_no_battery`** — clean copy of `_available_power` tracking `ua + loads - pv` without battery ops; avoids circular dependency where battery state depends on `_available_power` which battery also modifies
- **`_add_consumption_delta_power` updates 3 arrays** — `_available_power`, `_available_power_no_battery`, `_total_consumed_power` atomically
- **UA amps subtracted from amp budget at solver init** — `solve()` subtracts UA power from per-phase amp budget before constraint allocation
- **CMD_GREEN_CHARGE_ONLY enforcement** — `_battery_get_charging_power` clamps `available_power = min(0.0, available_power)` for green-charge-only commands, preventing discharge
- **Simplified `_compute_max_possible_production`** — removed `battery_actual_discharge` parameter; uses full `possible_discharge` (battery assumed switchable)

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
- Battery discharge uses SOC-limited values (via `Battery.get_charger_power()` → `possible_discharge`) — more accurate than hardware max
- Both DC and AC coupling handled in formula — user confirmed
- Battery assumed switchable (charge → discharge) — user confirmed
- Recompute battery state after each constraint allocation — user confirmed, for full accuracy

### Known risks acknowledged:
- `_battery_get_charging_power` called per-constraint in inner loop may be slow — correctness first, optimize later
- Per-group production isolation lost when removing `available_amps_production_for_group` — home-level power guard replaces it
