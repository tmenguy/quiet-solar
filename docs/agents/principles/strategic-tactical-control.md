---
title: Strategic vs tactical control
slug: strategic-tactical-control
kind: principle
last_verified: 2026-05-19
---

# Strategic (solver) vs tactical (charger budgeting) control

## TL;DR

Quiet-solar runs **two layers of control**. The **strategic layer**
(`PeriodSolver` in `home_model/solver.py`) plans in 15-minute
windows: "this car should get 7kW from 14:00 to 16:00". The
**tactical layer** (charger dynamic budgeting in
`ha_model/charger.py`) operates in real time: it manages the
physical realities of per-phase amp budgets across multiple
chargers on the same circuit, with 45-second adaptation windows,
phase switching, and staged transitions. **The tactical layer can
override the strategic layer**. If the solver says 7kW but the
circuit only allows 5kW right now, tactical wins. A solver bug
produces a bad plan; a tactical bug trips a breaker.

## When you need this principle

- Designing any feature that touches charger control. You'll need
  to understand both layers and their interaction.
- Working on the solver. Remember: your plan may be overridden.
- Touching the dynamic-group tree or per-phase budgeting.
- Reviewing a "charging improvement" PR. Ask: which layer does it
  modify? Both?

## Core idea

**Strategic** (solver):

- Time horizon: 15-minute windows, up to 24 hours ahead.
- Cadence: event-driven, with 5-minute fallback. Runs inside the
  ~7s load-management cycle.
- Output: command timeline.
- Optimises across all loads simultaneously.
- Stateless between runs — each cycle re-plans from scratch.

**Tactical** (charger budgeting):

- Time horizon: now to ~45 seconds.
- Cadence: every load-management cycle, with 45-second adaptation
  windows.
- Output: per-charger amp / phase setpoints applied right now.
- Manages physical-circuit constraints (per-phase amp limits).
- **Stateful** between runs — adaptation windows, staged
  transitions, dampening measurements persist.

**Why the split?**

- The solver's job is global optimisation; the tactical layer's job
  is local safety. Mixing them produces a layer that does neither
  well.
- The solver wants to take "long" decisions (charge until SOC=90%);
  the tactical layer wants to take "short" decisions (rebalance
  this circuit right now). The cadence mismatch makes them
  incompatible in one engine.
- The tactical layer's blast radius (tripped breakers, physical
  damage) demands defensive engineering the solver's algorithm
  doesn't need.

## Concrete implications

- **Solver bugs produce bad plans.** Detectable, recoverable in the
  next cycle.
- **Tactical bugs trip breakers.** Physical, immediate, household
  incident.
- **A feature that changes solver behaviour must verify tactical
  doesn't routinely override**. If the solver's plan is constantly
  rejected, that's a sign the plan is unrealistic.
- **Tactical can only restrict, never expand.** It will deliver
  less than the solver asked for, never more.

## Common mistakes

- Adding circuit-limit logic to the solver. The solver doesn't
  know about real-time circuit state; that's tactical.
- Adding global-optimisation logic to the tactical layer. The
  tactical layer optimises locally over 45 seconds; it can't
  reason about "what's cheapest over the next 6 hours".
- Forgetting that the solver may have already accounted for some
  tactical-layer behaviour (e.g., phase switching to maximise
  total power). Read both layers before changing either.
- Treating the solver as authoritative. Production behaviour =
  what the tactical layer actually delivered, not what the solver
  planned.

## See also

- [../concepts/solver.md](../concepts/solver.md) — the strategic
  layer.
- [../concepts/charger-budgeting.md](../concepts/charger-budgeting.md)
  — the tactical layer.
- [../concepts/dynamic-group-tree.md](../concepts/dynamic-group-tree.md)
  — the topology tactical walks.
- [event-driven-with-fallback.md](event-driven-with-fallback.md) —
  when the solver re-runs.
- [../use-cases/magali-plugs-in-car.md](../use-cases/magali-plugs-in-car.md)
  — the magic-moment use case where both layers cooperate.
