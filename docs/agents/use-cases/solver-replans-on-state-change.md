---
title: Solver replans on state change
slug: solver-replans-on-state-change
kind: use-case
last_verified: 2026-05-19
---

# Solver replans on state change — event-driven re-evaluation

## TL;DR

The solver doesn't run on a clock — it runs because **something
happened**. A constraint was pushed, a device state changed, a
command was ACKed, the user issued an override, the grid sensor
flickered. Each of these resets `_last_solve_done`; the next
load-management cycle (~7s) sees stale state and re-solves. The
periodic 5-minute fallback catches any missed event. This is the
event-driven-with-fallback principle in action.

## When you need this use case

- Adding a new trigger for solver re-evaluation.
- Touching the `_last_solve_done` reset path.
- Debugging "the solver didn't react to X" issues.
- Designing a feature whose value depends on quick replanning.

## End-to-end sequence

```text
1. Some event happens (one of):
   a. New constraint pushed (push_live_constraint or push_agenda_
      constraints)
   b. Device state changed (HADeviceMixin.add_to_history records
      a meaningful delta)
   c. Command ACKed (probe_if_command_set returned True)
   d. User override created in the UI
   e. Grid sensor flickered (OFF_GRID detection / recovery)
   ↓
2. Event handler resets _last_solve_done = None on QSHome
   ↓
3. Next load-management cycle (~7s):
   QSHome.update_loads() observes _last_solve_done is stale
   triggers PeriodSolver.solve() within this cycle
   ↓
4. Solver produces fresh command timeline; commands launched
   (max 1 per load per cycle, amp budget validated)
   ↓
5. Charger budgeting cycle accepts new amp targets; applies
   staged transitions if needed
   ↓
6. _last_solve_done updated to the solve completion timestamp
   ↓
7. ... no events happen for 5 minutes ...
   ↓
8. Periodic fallback: load-management cycle observes "5 minutes
   since last solve" → solves anyway (safety net)
```

## Why 5 minutes for the fallback?

- Long enough that the safety-net solve doesn't waste CPU when
  nothing changed.
- Short enough that any missed event surfaces within tolerable
  staleness (the user shouldn't notice).
- Hysteresis ([hysteresis-and-switching-cost.md](../principles/hysteresis-and-switching-cost.md))
  keeps frequent solves from producing frequent state changes.

## What each layer contributes

| Layer | Contribution |
|---|---|
| Solver ([solver.md](../concepts/solver.md)) | The re-evaluation entry point. |
| QSHome ([qs-home-orchestrator.md](../concepts/qs-home-orchestrator.md)) | Hosts `_last_solve_done`; runs the load-management cycle. |
| Constraints ([constraints.md](../concepts/constraints.md)) | One source of events (constraint pushes reset the timestamp). |
| HADeviceMixin ([ha-device-mixin.md](../concepts/ha-device-mixin.md)) | Another source (state changes via `add_to_history`). |
| Event-driven principle ([event-driven-with-fallback.md](../principles/event-driven-with-fallback.md)) | The architectural rule the use case embodies. |

## Common mistakes when modifying this path

- Adding a re-evaluation trigger but forgetting to reset
  `_last_solve_done`. The trigger never fires the solver.
- Bypassing the load-management cycle and calling
  `PeriodSolver.solve()` directly from an event handler. Synchronous
  invocations break the rate-limiting and concurrency guarantees.
- Lengthening the 5-minute fallback to "make it less wasteful". The
  fallback is the safety net; shortening it is fine, lengthening
  is regret.
- Treating every state delta as a re-evaluation trigger. Only
  *meaningful* deltas (the device crossed a threshold, the
  forecast shifted enough to matter) reset the timestamp.

## See also

- [../concepts/solver.md](../concepts/solver.md) — the engine.
- [../concepts/qs-home-orchestrator.md](../concepts/qs-home-orchestrator.md)
  — the host of the cycle that triggers it.
- [../principles/event-driven-with-fallback.md](../principles/event-driven-with-fallback.md)
  — the principle.
- [../principles/hysteresis-and-switching-cost.md](../principles/hysteresis-and-switching-cost.md)
  — the counterweight that keeps the system stable.
- [magali-plugs-in-car.md](magali-plugs-in-car.md) — step 7 is
  this use case in action.
