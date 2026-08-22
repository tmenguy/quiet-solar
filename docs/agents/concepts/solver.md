---
title: PeriodSolver
slug: solver
kind: concept
covers:
  - custom_components/quiet_solar/home_model/solver.py
last_verified: 2026-08-22
---

# PeriodSolver

## TL;DR

`PeriodSolver` is the strategic optimization engine. It creates
15-minute time slots aligned with constraint boundaries and tariff
changes (`SOLVER_STEP_S = 900`), then allocates power across all
loads simultaneously. The allocation hierarchy is strict: maximize
solar self-consumption → minimize grid cost → maintain comfort
commitments. The solver returns a command timeline `(load, [(time,
LoadCommand)])` — strategic by design; the tactical charger budgeting
layer (`charger-budgeting.md`) may override.

## When you need this concept

- Improving the allocation algorithm (within the existing input/output
  contract — see "Solver Optimization Strategy" in
  architecture.md).
- Debugging "the plan looked wrong yesterday" issues.
- Adding a new constraint tier or scoring axis.
- Working on a feature where the solver and charger budgeting
  interact (e.g., car charging during a tariff transition).

## Core idea

The solver runs **event-driven with a 5-minute fallback**: it
re-evaluates when constraints change or device state changes (which
reset `_last_solve_done`), and as a safety net it re-runs every 5
minutes even when nothing has changed.

Allocation algorithm:

1. Create time slots and power slots from the PV forecast +
   unavoidable consumption.
2. Allocate mandatory constraints in priority order (`MANDATORY_AS_FAST_AS_POSSIBLE`
   first, then `MANDATORY_END_TIME`, then `BEFORE_BATTERY_GREEN`).
3. Optimize battery charge/discharge to minimize grid imports.
4. Allocate filler constraints (`FILLER`, then `FILLER_AUTO`) using
   remaining surplus.
5. Return the command timeline.

Within each tier, constraints are ordered by score; ties broken by
constraint-specific criteria (e.g., deadline proximity).

### Discharge floor modelling (QS-349)

`_battery_get_charging_power` models the battery's **outage safety
floor** F (`Battery.min_discharging_power`, default 0). During a
`CMD_GREEN_CHARGE_ONLY` slot the available power is capped at F
(`min(available_power, F)`) rather than 0, so a demand slot still
discharges up to F. Each discharging slot's energy splits into an
**incompressible leak** `leak_i = min(discharge, F × duration)` — which
flows even when the slot is flipped to green-charge-only — and a
**compressible remainder** the price-bucket optimizer may move between
buckets. Pass 1 returns `prices_leak_energy_buckets` as the 9th field of
the `BatteryChargingPower` NamedTuple (review-fix #02 R8 — the production
call sites unpack by name, so a future field does not require touching
them; positional test constructions/asserts, being a tuple, still pin
the arity by design; every field is precisely typed —
`list[LoadCommand | None]` commands, `dict[float, float]` price buckets —
so mypy catches call-site misuse, review-fix #08 Y7); the
allocation pass subtracts the leak from the discharged buckets **once**
before its revisiting loop (via the single-call helper
`_leak_normalize_discharged_buckets` — subtracting inside the loop would
double-count on cheap-bucket revisits), so only compressible energy is
redistributed. The flip site keeps the leak
(`charged_energy = max(charged_energy, −leak_i)`) and debits the budget
**post-clamp**: a flipped slot's surviving discharge is exactly the leak
(`charged_energy + leak_i == 0`), so it consumes no budget and the
residual stays available for later same-price slots. At F = 0 every leak
is 0 and the whole flip site (condition, clamp, debit) reduces
algebraically to the pre-QS-349 behaviour. Off-grid solves are unaffected — off-grid
tariffs are flat, a single price bucket, so the allocation short-circuits
and no `CMD_GREEN_CHARGE_ONLY` is issued. **Documented residual**: during
`CMD_FORCE_CHARGE`, if the battery saturates or the inverter derates
mid-slot, hardware may discharge up to F for the slot's remainder —
unmodelled, bounded by `F × remainder` (≤ F/4 Wh per 15-min slot).
**Latent-trap note (review-fix #04 U10)**: the returned
`prices_remaining_grid_energy_buckets` / `remaining_grid_energy` report
*dispatch-relative* grid energy (a flipped `CMD_GREEN_CHARGE_ONLY` demand
slot uses the F-capped `available_power`, understating its physical grid
import by `net_load − F`). Harmless today (pass-1 decision inputs are
flip-free; the allocation loop re-adds removed energy explicitly), but a
future reader of the final rebinds must re-derive the physical residual
before consuming those buckets from a flipped-command call.

On forecast-proven big-sun days the solver runs an aggressive surplus
pre-discharge that deliberately over-consumes to free battery headroom
for tomorrow's surplus. That placement fills latest-first
(`adapt_repartition(..., fill_order_reverse=True)`) so the deliberate
depletion hugs the solar-surplus onset — keeping the battery full as a
buffer through the early night instead of draining it at "now".

## Key types / structures

- `PeriodSolver.solve()` — the entry point. Inputs: constraints,
  tariffs, PV forecast, battery state, loads. Output: command
  timeline.
- `SOLVER_STEP_S = 900` — discretisation step (15 minutes). Lives
  in `const.py`. **Do not touch.**
- `_last_solve_done` — timestamp guarding the 5-minute fallback;
  reset by constraint/state changes for event-driven re-evaluation.

## Lifecycle / sequence

```text
update_loads() cycle (~7s)
  ↓
update_loads_constraints()           ← constraints pushed here
  ↓
check_loads_commands()               ← ACK validation
  ↓
solver re-evaluation needed?
  YES (event or 5-min fallback) ↓
    PeriodSolver.solve()
      → command timeline
  ↓
launch commands (max 1/load/cycle, amp budget checked)
```

## Common mistakes

- Changing `SOLVER_STEP_S` — breaks every test assumption and every
  time-aligned constraint.
- Adding logic outside the solver that depends on a specific
  allocation order. The solver's contract is the timeline, not the
  intermediate decisions.
- Modifying the solver's input/output contract — see the
  "Decision 3: Solver Optimization Strategy" boundary.
- Forgetting that the **tactical charger budgeting may override** the
  solver's plan. The solver decides "charge at 7kW"; the budgeting
  layer may deliver 5kW because of a circuit constraint.

## See also

- [constraints.md](constraints.md) — the demand language the solver
  consumes.
- [commands.md](commands.md) — the action language the solver
  produces.
- [charger-budgeting.md](charger-budgeting.md) — the tactical layer
  that overrides.
- [../principles/strategic-tactical-control.md](../principles/strategic-tactical-control.md)
  — why the split exists.
- [../principles/event-driven-with-fallback.md](../principles/event-driven-with-fallback.md)
  — the re-evaluation trigger model.
