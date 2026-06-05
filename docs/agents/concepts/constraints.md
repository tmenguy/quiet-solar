---
title: LoadConstraint
slug: constraints
kind: concept
covers:
  - custom_components/quiet_solar/home_model/constraints.py
last_verified: 2026-06-05
---

# LoadConstraint

## TL;DR

A `LoadConstraint` is a time-windowed demand on a load with progress
tracking. Five priority tiers (MANDATORY_AS_FAST_AS_POSSIBLE down to
FILLER_AUTO) determine the solver's allocation order. Every constraint
carries `load_info` metadata (originator: prediction, agenda,
user_override, system) so notifications and conflict resolution can
trace where a demand came from. The solver reads constraints via
`AbstractLoad.get_for_solver_constraints()` each cycle — never modify
constraints in place; create new ones.

## When you need this concept

- Adding a feature that creates demand (a prediction, a calendar
  trigger, a user override).
- Designing a new constraint subclass for a new demand shape (power
  vs duration vs SOC%).
- Debugging "the system didn't honour my deadline" issues.
- Working on the solver's allocation algorithm.

## Core idea

A constraint tracks `initial_value → current_value → target_value`
with a percent-complete progress signal. It has a `start_time`
(when the demand becomes eligible) and a deadline (varies by
subclass). The solver walks constraints in priority order; for each,
it picks the cheapest power slots that get from `current_value` to
`target_value` before the deadline.

The five priority tiers (highest first):

1. **MANDATORY_AS_FAST_AS_POSSIBLE** (score 9) — schedule first,
   minimize completion time regardless of cost or solar.
2. **MANDATORY_END_TIME** (score 7) — must complete by deadline;
   solver picks cheapest fitting slots.
3. **BEFORE_BATTERY_GREEN** (score 5) — runs before battery switches
   to solar-only mode (typically morning grid arbitrage).

   (Orthogonal to the tiers: `score()` adds a highest-order 1e14 term
   for any constraint whose `load_info` originator is
   `user_override`, so override constraints always win allocation
   ordering and same-end-time cluster dedup — QS-256.)
4. **FILLER** (score 3) — uses surplus only; never imports from grid.
5. **FILLER_AUTO** (score 1) — opportunistic; runs whenever any
   surplus exists.

## Key types / structures

- `LoadConstraint` — base class. Tracks progress and deadline.
- `MultiStepsPowerLoadConstraint` — power-based with step options
  (e.g., charger amperage levels).
- `MultiStepsPowerLoadConstraintChargePercent` — SOC-based; target
  is a battery percentage, not raw kWh.
- `TimeBasedSimplePowerLoadConstraint` — duration-based (e.g., pool
  pump runs for N hours at fixed power).
- `TimeBasedHoldOffConstraint` — pins a load to CMD_IDLE for its whole
  window (QS-256 user OFF-override). Zero power by default, wall-clock
  progress (`compute_value` accrues unconditionally), trivial
  repartition (CMD_IDLE commands, zero power, no energy impact), no-op
  `adapt_repartition`. Met at window end via
  `always_end_at_end_of_constraint` → acked → override reset.
- `load_info` dict — `{originator: "user_override" | "agenda" |
  "prediction" | "system"}`. Mandatory metadata.

## Lifecycle

1. **Creation** — in `update_loads_constraints()` (the 7-second load
   management cycle), constraints are pushed via
   `push_live_constraint()` or `push_agenda_constraints()`.
2. **Read** — `AbstractLoad.get_for_solver_constraints()` returns
   active constraints to the solver.
3. **Allocation** — solver assigns power slots; allocation order is
   priority tier, then constraint score within tier.
4. **Progress update** — as commands execute, `current_value`
   advances; the constraint completes when `current_value >=
   target_value`.
5. **Completion / expiry** — completed constraints stay until the
   next cycle, then are pruned.

## Common mistakes

- Creating a constraint in `__init__()` (too early — the solver
  hasn't started). Always in `update_loads_constraints()`.
- Creating a constraint without `load_info`. Notifications can't
  explain "why" without it.
- Modifying an existing constraint instead of creating a new one.
  The solver re-reads constraints each cycle; mutating mid-cycle
  causes inconsistent allocations.
- Picking the wrong tier — `MANDATORY_AS_FAST_AS_POSSIBLE` over-eager
  scheduling that ignores cheap-grid windows; or `FILLER` for a
  user-pinned demand that ends up never running on a cloudy day.

## See also

- [commands.md](commands.md) — constraints produce `LoadCommand`s.
- [solver.md](solver.md) — the allocator.
- [user-override.md](user-override.md) — `load_info: {originator:
  "user_override"}` flow.
- [../principles/observe-predict-optimize.md](../principles/observe-predict-optimize.md)
  — why constraints are the right abstraction.
