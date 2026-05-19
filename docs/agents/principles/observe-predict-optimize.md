---
title: Observe → predict → optimize
slug: observe-predict-optimize
kind: principle
last_verified: 2026-05-19
---

# Observe → predict → optimize

## TL;DR

Quiet-solar's foundational design philosophy is **observe → predict
→ optimize**. The system continuously observes household state,
predicts upcoming energy needs and production, and optimizes the
allocation of power across all controllable devices. This is the
right engineering approach because home energy is inherently
unpredictable: simple rules and manual schedules break under real
variability, while a constraint-based solver handles uncertainty
gracefully and makes the best possible decision at every moment.

## When you need this principle

- Designing a new feature that introduces a "rule" or a
  "schedule" — you're probably reaching for the wrong abstraction.
- Choosing between hardcoding behaviour vs creating a
  `LoadConstraint`.
- Justifying why we don't ship an "if solar > 3kW, turn on pool
  pump" automation.

## Core idea

**Observe**: every device reports state into history (the 3-day
ring buffer per device + the 560-day numpy ring buffer for
long-term patterns). The system has rich, time-aligned visibility
into what's happening.

**Predict**: forecasts come from outside (Solcast / OpenMeteo for
PV) and from inside (person trip prediction from mileage history,
pool temperature evolution, etc.). Predictions are probabilistic by
nature — the system must work even when predictions are wrong.

**Optimize**: the `PeriodSolver` consumes observations + predictions
+ constraints and produces a command timeline. The strict hierarchy
(solar self-consumption → grid cost → comfort commitments) governs
every trade-off.

Rules-based systems work only when reality is predictable. Home
energy is not. The constraint-based solver is the **correct
abstraction**: it expresses *what* the user wants (constraints) and
lets the optimiser figure out *how*, given current state.

## Concrete implications

- **Don't add an `automations.yaml` rule** to do what a constraint
  could do. Rules are brittle; constraints compose.
- **Don't hardcode time windows** ("charge between 22:00 and
  06:00"). Express the demand as a constraint with the relevant
  metadata; let the solver pick the windows.
- **Embrace uncertainty**: predictions will be wrong. The recovery
  story matters more than the happy path. See
  [magali.md](../personas/magali.md).
- **Make decisions traceable**: every decision must be explicable
  in terms of (observation, prediction, constraint). That's why
  `load_info.originator` exists.

## Common mistakes

- Adding a rule that bypasses the solver "for performance" or
  "because the solver is too slow". The solver runs in ~milliseconds
  on a typical home; complexity hides elsewhere.
- Treating predictions as ground truth. They're inputs, not facts.
- Inventing a parallel optimisation surface for a "special" device.
  Every controllable device joins the solver via constraints.

## See also

- [strategic-tactical-control.md](strategic-tactical-control.md) —
  the split that makes the solver tractable.
- [event-driven-with-fallback.md](event-driven-with-fallback.md) —
  the re-evaluation trigger model that keeps the solver responsive.
- [../concepts/constraints.md](../concepts/constraints.md) — the
  demand language the principle produces.
- [../concepts/solver.md](../concepts/solver.md) — the optimisation
  engine.
- [../personas/magali.md](../personas/magali.md) — the persona
  whose trust hinges on the recovery story.
