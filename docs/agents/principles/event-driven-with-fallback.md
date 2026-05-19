---
title: Event-driven with periodic fallback
slug: event-driven-with-fallback
kind: principle
last_verified: 2026-05-19
---

# Event-driven with periodic fallback

## TL;DR

The solver re-evaluates **on events** — constraint changes, device
state changes, command ACKs — and re-runs as a safety net every 5
minutes even when nothing has changed. This is the architectural
principle that lets the system be both responsive (sub-second when
something happens) and self-healing (the periodic fallback catches
any missed event). Combined with the hysteresis principle
([hysteresis-and-switching-cost.md](hysteresis-and-switching-cost.md)),
frequent re-evaluation doesn't produce frequent state changes —
because most re-evaluations decide "the plan is still right".

## When you need this principle

- Adding a new trigger for solver re-evaluation.
- Touching the `_last_solve_done` timestamp logic.
- Working on event propagation between device state changes and the
  solver.
- Debugging "the system didn't react to X" issues.

## Core idea

**Event-driven**: state changes that *could* invalidate the current
plan reset `_last_solve_done`. The next ~7s load-management cycle
sees `_last_solve_done is None` (or stale) and triggers a fresh
solve.

**Periodic fallback**: independent of events, the load-management
cycle checks: has it been >5 minutes since the last solve? If yes,
solve anyway. This catches:

- Missed events (e.g., an HA state change that didn't propagate).
- Time-based triggers (a constraint's deadline approaching).
- External condition changes the system can't observe directly.

**Why both?**

- Event-driven alone is fragile: any missed event leaves the system
  with a stale plan, possibly forever.
- Periodic alone is wasteful and slow: a 5-minute lag on user
  override is unacceptable.

Combined, the system is responsive **and** self-healing.

## Concrete implications

- **Every "the solver should consider X" path must reset
  `_last_solve_done`**. State changes, constraint pushes, command
  ACKs.
- **Don't shorten the 5-minute fallback** to "make the system more
  responsive". Responsiveness comes from events; the fallback is
  the safety net.
- **Don't lengthen the 5-minute fallback** either. 5 minutes is the
  tolerated worst-case staleness — longer becomes user-visible.
- **The hysteresis principle is the counterweight**. Frequent
  re-evaluation is safe because frequent state changes aren't
  allowed.

## Common mistakes

- Adding a re-evaluation trigger but forgetting to reset
  `_last_solve_done`. The trigger never fires the solver.
- Replacing the periodic fallback with "trust the events". Always
  a regret — events are lossy in distributed systems.
- Treating the 5-minute fallback as the primary cadence and the
  events as optimisation. It's the other way around.
- Running the solver inside an event handler synchronously. The
  ~7s load-management cycle is the right surface; events queue work
  for the next cycle.

## See also

- [../concepts/solver.md](../concepts/solver.md) — the
  re-evaluation entry point.
- [../concepts/qs-home-orchestrator.md](../concepts/qs-home-orchestrator.md)
  — the cycle that triggers the solver.
- [hysteresis-and-switching-cost.md](hysteresis-and-switching-cost.md)
  — the counterweight to frequent re-evaluation.
- [../use-cases/solver-replans-on-state-change.md](../use-cases/solver-replans-on-state-change.md)
  — the canonical event-driven scenario.
