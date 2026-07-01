---
title: Charger Dynamic Budgeting
slug: charger-budgeting
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/charger.py
last_verified: 2026-06-13
---

# Charger Dynamic Budgeting — the tactical layer

## TL;DR

`ha_model/charger.py` is **trust-critical**: it's where the solver's
strategic plan meets physical reality. The charger budgeting layer
operates in 45-second adaptation windows (`CHARGER_ADAPTATION_WINDOW_S`),
manages per-phase amp distribution across multiple chargers on the
same circuit, and **can override the solver** when amp budgets
conflict. Phase switching (1P↔3P), staged transitions, and dampening
(real-power measurement) all live here. A bug in the solver makes a
bad plan; a bug here trips a breaker.

## When you need this concept

- Implementing or modifying any charger behaviour
  (`QSChargerGeneric`, `QSChargerOCPP`, `QSChargerWallbox`).
- Touching the dynamic-group budgeting tree (see
  [dynamic-group-tree.md](dynamic-group-tree.md)).
- Working on phase switching, dampening, or adaptation-window logic.
- Anything that could affect physical safety — circuit limits,
  breaker margins, max amp per phase.

## Core idea

The budgeting algorithm (`budgeting_algorithm_minimize_diffs`):

1. **Priority check**: if the highest-priority charger isn't charging
   but a lower one is, trigger a reset allocation.
2. **Prepare budgets**: either keep current amps (minimise
   transitions) or reset to minimum (rebalance).
3. **Shave mandatory**: if the minimum amps still exceed the group
   limit, stop the lowest-score chargers first.
4. **Shave current**: try phase switching (1P→3P) for lower-score
   chargers before reducing amps.
5. **Smart allocation loop**: iteratively adjust each charger's
   budget toward the power target while respecting all constraints.

Then `apply_budget_strategy()`:

- For large changes, **stage** across two cycles: phase 1 reduces
  decreasing chargers (frees up amps), phase 2 increases other
  chargers next cycle (already validated safe).
- `remaining_budget_to_apply` persists between cycles to complete
  phase 2.

## Key types / structures

- `QSChargerGeneric` — base class. Power ramping, phase switching,
  budgeting state machine.
- `QSChargerOCPP` — OCPP variant. Adds transaction handling.
- `QSChargerWallbox` — Wallbox variant. Maps vendor status enums.
- `QSChargerGroup` — aggregates chargers on the same circuit.
- `QSChargerStatus` — per-charger state (amps, phases, real power,
  adaptation state).
- `charge_score` — per-charger priority. Higher = wins budget
  conflicts first.
- `CHARGER_ADAPTATION_WINDOW_S = 45` — stability requirement before
  rebalancing.
- `CHARGER_STATE_REFRESH_INTERVAL_S = 14` — state-polling cadence.

### Charge-origin tagging & `get_charge_type()` (QS-274)

Constraints carry their origin in `load_info[CONSTRAINT_ORIGINATOR_KEY]`
so the UI can distinguish *where* a charge target came from. Creation
sites stamp named constants from `const.py`:

- **manual finish time** → `CONSTRAINT_ORIGINATOR_MANUAL`
- **calendar/agenda** → `CONSTRAINT_ORIGINATOR_AGENDA` (value is exactly
  `"agenda"`, an invariant — persisted pre-QS-274 calendar constraints
  must still match)
- **person forecast** → `CONSTRAINT_ORIGINATOR_PERSON`, stamped
  *alongside* the existing `{"person": <name>}` matching key (do not
  drop that key — `_match_ct` cleanup relies on it)
- **force/override** → `CONSTRAINT_ORIGINATOR_USER_OVERRIDE` (pre-existing)

`QSChargerGeneric.get_charge_type(return_charge_errors=True)` returns
`(type, constraint)`:

- **as-fast branch:** `user_override` originator →
  `CAR_CHARGE_TYPE_MANUAL_AS_FAST_AS_POSSIBLE`, else
  `CAR_CHARGE_TYPE_AS_FAST_AS_POSSIBLE`.
- **deadline branch (precedence):** `"person"` key →
  `CAR_CHARGE_TYPE_PERSON_AUTOMATED` (person-first, keeps legacy
  `{"person": name}`-only constraints working); `elif` agenda originator
  → `CAR_CHARGE_TYPE_CALENDAR`; `else` → `CAR_CHARGE_TYPE_MANUAL`
  (untagged / `load_info=None`). The `ct.load_info and …` guard is
  mandatory to avoid `None.get(...)`.
- **`return_charge_errors=False`** skips the Faulted / No Power / Not
  Plugged short-circuits and returns the underlying type — used by the
  origin context line so a charger-error string never leaks into it.

`CAR_CHARGE_TYPE_SCHEDULE` was removed entirely; the unrelated
`WallboxChargerStatus.SCHEDULED` hardware status is a different concept
and is untouched. The `CONSTRAINT_ORIGINATOR_MANUAL/PERSON` stamps are
written for coherence/forward-compat — detection keys off the `"person"`
key and the `else`, so they are not read by `get_charge_type()` itself.

## Lifecycle / sequence (one update cycle)

```text
All chargers stable for 45s? → dampening update (measure real power)
  ↓
Detect transitions (single charger change → record power delta)
  ↓
Budget reset opportunity? (20-min timeout or HP charger waiting)
  ↓
budgeting_algorithm_minimize_diffs()
  → priority check → prepare budgets → shave mandatory → shave current
  → smart allocation loop (iterative, phase limits)
  ↓
apply_budget_strategy()
  IF large change: phase 1 (reduce first) → phase 2 (next cycle)
  ELSE: apply directly
```

## Common mistakes

- Changing budgeting logic without integration tests that simulate
  multi-charger rebalancing sequences over time. Unit tests on a
  single charger miss the trust-critical interactions.
- Bypassing the dynamic-group tree's recursive validation. A leaf
  charger can pass its local check while violating the parent
  group's circuit limit.
- Skipping the staged-transition split for "small" changes that
  turn out to overshoot per-phase limits in transient. Apply
  staging whenever the change crosses a phase boundary.
- Treating `charge_score` as a tiebreaker. It's the primary ranking
  for budget conflicts.

## See also

- [dynamic-group-tree.md](dynamic-group-tree.md) — the topology
  budgeting walks.
- [solver.md](solver.md) — the strategic layer that produces the
  plan this layer may override.
- [../principles/strategic-tactical-control.md](../principles/strategic-tactical-control.md)
  — why the split exists.
- [../use-cases/magali-plugs-in-car.md](../use-cases/magali-plugs-in-car.md)
  — the magic-moment use case this layer enables.
- [car-soc-estimation.md](car-soc-estimation.md) — `constraint_update_value_callback_soc`
  is the **sole writer** of the car's float SOC accumulator (QS-243); while
  the car is estimating it bypasses the raw sensor and seeds the constraint
  from the effective estimate instead of a forced `0.0`. **QS-281** hoisted the
  `accumulate_soc_delta` call to exactly one per **percent** callback so the
  accumulator advances during a **healthy** charge too (return discarded on the
  non-estimating branch — the constraint value is byte-identical). The hoist is
  guarded on `is_target_percent`: the energy-mode callback never advances the
  SOC accumulator, so there is no phantom delta even if the
  `can_use_charge_percent_constraints()` ⇒ capacity coupling is ever loosened.
  It drives the
  accumulator through the car's public `soc_integration_cursor` /
  `accumulate_soc_delta` accessors (no underscore reach-in), and gates the
  zero-power hardware-fault check on `car.is_soc_sensor_distrusted()` (stale /
  no-sensor) — **not** on the broad estimation flag, so a manual override on a
  healthy car still gets fault detection. The estimate is reset on the
  genuine-plug-in branch (`do_full_reset`) and the unplug edge; a boot re-attach
  preserves it (reboot survival).
