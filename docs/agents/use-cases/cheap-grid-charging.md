---
title: Cheap-grid charging
slug: cheap-grid-charging
kind: use-case
last_verified: 2026-05-19
---

# Cheap-grid charging

## TL;DR

It's cloudy / nighttime / the battery is empty. Quiet-solar still
needs to honour mandatory constraints (car must be ready by 7am,
hot water tank must reach setpoint). The solver picks the cheapest
grid windows from the tariff schedule and allocates loads inside
those windows. This is the "no surplus available" complement to
[solar-surplus-allocation.md](solar-surplus-allocation.md).

## When you need this use case

- Modifying tariff-aware scheduling.
- Reasoning about MANDATORY constraint behaviour with no PV.
- Designing a feature that interacts with off-peak windows.
- Debugging "the car charged at peak rate" issues.

## End-to-end sequence

```text
1. Evening: PV forecast = 0 from sundown to sunup
   ↓
2. Tariff schedule: peak 18:00–22:00; off-peak 22:00–06:00; peak
   06:00–10:00; off-peak 10:00–14:00; etc.
   ↓
3. Solver runs (event or 5-min fallback):
   step 1: time slots aligned with tariff transitions + forecast
   step 2: MANDATORY constraints allocated first
     → MANDATORY_END_TIME (car ready by 7am): solver picks cheapest
       slots within [now, deadline] that satisfy energy demand
     → off-peak slots from 22:00–06:00 dominate
   step 3: battery — discharge during peak windows to avoid grid
     imports during evening; charge during off-peak if SOC permits
   step 4: filler constraints stay paused (no surplus)
   ↓
4. Battery contribution:
   peak window (18:00–22:00): discharge to cover the home's load,
     avoiding grid import at unfavourable rate
   off-peak window (22:00–06:00): charge battery if SOC < target
     AND the math says next-day-PV won't suffice
   ↓
5. Result: car charged from cheap off-peak grid; battery played
   the arbitrage game; peak-window grid imports minimised
```

## Decision: battery as buffer vs car as direct off-peak consumer

When both the battery and the car want off-peak slots, the solver
prefers **direct car charging** over "battery first, car later" —
because the round-trip efficiency penalty (charge battery → later
discharge to charge car) wastes ~15% of the energy. Direct grid →
car charging is more efficient when the car is plugged in during
the off-peak window.

## What each layer contributes

| Layer | Contribution |
|---|---|
| Solver ([solver.md](../concepts/solver.md)) | Time-slot creation aligned with tariff schedule; allocation by priority. |
| Constraints ([constraints.md](../concepts/constraints.md)) | `MANDATORY_END_TIME` semantics. |
| Battery ([home-model-battery.md](../concepts/home-model-battery.md), [ha-battery.md](../concepts/ha-battery.md)) | Arbitrage between peak and off-peak windows. |
| Charger budgeting ([charger-budgeting.md](../concepts/charger-budgeting.md)) | Per-phase amp distribution still applies even when total power is constrained by tariff window. |

## Common mistakes when modifying this path

- Treating tariff windows as soft hints rather than the cost
  function. The solver's job is cheapest-feasible.
- Round-tripping through the battery when the car could charge
  directly. ~15% efficiency loss multiplied by every night.
- Forgetting that MANDATORY constraints must be honoured even when
  the cheapest off-peak window is too narrow. The solver may need
  to spill into adjacent peak slots — and that's correct.

## See also

- [solar-surplus-allocation.md](solar-surplus-allocation.md) — the
  opposite case.
- [magali-plugs-in-car.md](magali-plugs-in-car.md) — uses cheap-grid
  charging during cloudy nights.
- [../concepts/solver.md](../concepts/solver.md) — the optimisation
  engine.
- [../principles/observe-predict-optimize.md](../principles/observe-predict-optimize.md)
  — the cost-minimisation tier of the hierarchy.
