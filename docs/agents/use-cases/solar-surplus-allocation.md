---
title: Solar surplus allocation
slug: solar-surplus-allocation
kind: use-case
last_verified: 2026-05-19
---

# Solar surplus allocation

## TL;DR

The sun comes up; PV production exceeds the home's baseload. Quiet-
solar's optimisation hierarchy says **maximise free solar self-
consumption first** — so the surplus goes to controllable loads
(battery charge, pool pump, car charging) in priority order, before
any kWh is exported to the grid at unfavourable rates. This is the
day-to-day "where does the surplus go?" use case.

## When you need this use case

- Reasoning about FILLER constraint behaviour.
- Modifying the solver's surplus-allocation step.
- Designing a feature that touches the optimisation hierarchy.
- Debugging "the battery isn't charging when it's sunny" issues.

## End-to-end sequence

```
1. Solar production rises → QSSolar updates merged forecast (~30s
   refresh)
   ↓
2. Load-management cycle (~7s) observes surplus = production -
   unavoidable consumption
   ↓
3. Solver re-evaluates (event triggered by forecast update OR
   constraint change OR 5-min fallback)
   ↓
4. Allocation algorithm:
   step 1: time-slot creation (15-min slots aligned with forecast)
   step 2: mandatory constraints already allocated this morning
   step 3: battery → DC-coupled charge preferred (avoids round-trip
     loss); SOC bounds respected
   step 4: filler constraints (pool, on/off duration loads) get
     the remaining surplus
   ↓
5. Charger budgeting cycle accepts the new per-load amp targets,
   applies staged transitions if changes are large
   ↓
6. Devices execute → real power matches plan within dampening
   tolerance → next cycle reads back the actuals
```

## Priority order within the surplus pool

Per the optimisation hierarchy:

1. **Mandatory** constraints that are not yet satisfied get surplus
   first (even though they would have been served at a higher tier
   too).
2. **Battery charge** (DC-coupled preferred). Saving energy for
   the evening peak-tariff window has higher economic value than
   running a filler load now.
3. **Filler** constraints (pool, on/off duration, FILLER-tier car
   charging).
4. **Filler-auto** (`FILLER_AUTO`) — only if surplus remains after
   all other tiers.

## What each layer contributes

| Layer | Contribution |
|---|---|
| Solar providers ([solar-providers.md](../concepts/solar-providers.md)) | Dampened, merged production forecast. |
| Solver ([solver.md](../concepts/solver.md)) | Allocates surplus across loads. |
| Battery ([home-model-battery.md](../concepts/home-model-battery.md), [ha-battery.md](../concepts/ha-battery.md)) | Consumes surplus within SOC + power limits. |
| Bistate-duration ([bistate-duration-devices.md](../concepts/bistate-duration-devices.md)) | Pool / on-off loads sized to surplus windows. |
| Charger budgeting ([charger-budgeting.md](../concepts/charger-budgeting.md)) | Physical-safety check on the surplus going to chargers. |

## Common mistakes when modifying this path

- Inserting a rule that diverts surplus to a "favourite" load. The
  hierarchy is global; per-load preference happens at the
  constraint level, not by bypassing the solver.
- Forgetting the dampening on the forecast. Raw production whips
  the allocation; dampened forecast yields stable plans.
- Charging the battery from grid during a surplus window. This is
  the most expensive thing the system can do — a bug here costs
  the user real money every day.

## See also

- [cheap-grid-charging.md](cheap-grid-charging.md) — the opposite
  case (no surplus, cheap grid).
- [magali-plugs-in-car.md](magali-plugs-in-car.md) — surplus
  feeding the car-charging story.
- [../principles/observe-predict-optimize.md](../principles/observe-predict-optimize.md)
  — why the solver is the right surface.
- [../concepts/solver.md](../concepts/solver.md) — the allocator.
