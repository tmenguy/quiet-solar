---
title: Hysteresis & switching cost
slug: hysteresis-and-switching-cost
kind: principle
last_verified: 2026-05-19
---

# Hysteresis and switching-cost protection

## TL;DR

Solver re-evaluation is event-driven and frequent — but every state
change has a **real cost** (mechanical wear on relays, user
confusion, transient over-current). Quiet-solar uses three layered
defences to keep plans stable: a **daily on/off budget**
(`num_max_on_off`), **time-based hysteresis**
(`CHANGE_ON_OFF_STATE_HYSTERESIS_S = 600`s = 10 minutes minimum
between state changes), and **multi-pass constraint adaptation**
(try free transitions first, only spend budget on the second pass).
This is an architectural principle, not a per-device tweak.

## When you need this principle

- Adding a new on/off device. You'll inherit the pattern; don't
  bypass it.
- Tuning a device's `num_max_on_off` default. The default matters —
  too tight → device never runs; too loose → relay wears out.
- Debugging "the device chattered all afternoon" issues.
- Working on the solver's adaptation loop.

## Core idea

**Three defences, layered**:

1. **Daily on/off budget** (`num_max_on_off`): per-device count of
   how many on→off or off→on transitions are allowed per 24 hours.
   When the budget is exhausted, the device stays in its current
   state for the rest of the day even if the solver wants to switch.
2. **Hysteresis** (`CHANGE_ON_OFF_STATE_HYSTERESIS_S`): minimum delay
   between consecutive state changes for the *same* device. A
   device that just turned on can't turn off for 10 minutes, no
   matter what.
3. **Multi-pass adaptation**: in the solver's allocation step, the
   first pass tries to honour constraints **without** spending
   switching budget (use the existing state where possible). Only
   if the first pass can't satisfy a constraint does the second
   pass spend a switch.

These compose: a solver running every cycle doesn't trigger a
state change every cycle. The combined effect is stability — plans
that hold over the time horizon they were planned for.

## Concrete implications

- **Every on/off device inherits this pattern** via `AbstractDevice`.
  Pool, heat pump, on/off duration, any new bistate device.
- **Modulating devices (chargers, batteries) don't use
  `num_max_on_off`** the same way. Their continuous control plane
  is a different stability problem (see
  [strategic-tactical-control.md](strategic-tactical-control.md)).
- **Solver tests must exercise the multi-pass adaptation**. A
  single-pass test passes against a single-pass solver but breaks
  in production.

## Common mistakes

- Bypassing the budget for a "special" device. Every bypass is a
  future bug report ("my X keeps switching").
- Setting `num_max_on_off` from a user's gut feel instead of from
  device wear tolerance. The default lives in `const.py` for a
  reason.
- Implementing hysteresis ad-hoc per device. The shared base
  enforces 10 minutes; reinventing it produces inconsistency.
- Triggering an extra switch in the solver because "the optimal
  plan needs it". The optimal plan that ignores switching cost
  isn't optimal — it just defers the cost.

## See also

- [../concepts/load-base.md](../concepts/load-base.md) — where the
  budget and hysteresis live.
- [../concepts/bistate-duration-devices.md](../concepts/bistate-duration-devices.md)
  — the canonical on/off consumers.
- [event-driven-with-fallback.md](event-driven-with-fallback.md) —
  the re-evaluation trigger that this principle counterbalances.
- [strategic-tactical-control.md](strategic-tactical-control.md)
  — the stability story for continuous-control devices.
