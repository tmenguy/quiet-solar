---
title: PeriodSolver
slug: solver
kind: concept
covers:
  - custom_components/quiet_solar/home_model/solver.py
last_verified: 2026-05-19
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
