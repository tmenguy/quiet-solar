---
title: Off-grid mode
slug: off-grid-mode
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/home.py
last_verified: 2026-05-19
---

# Off-grid mode

## TL;DR

When the grid power sensor drops to zero, `QSHome` detects an outage
and transitions into `QSHomeMode.OFF_GRID`: it sheds non-mandatory
loads, restricts consumption to (solar + battery) capacity, and
broadcasts a high-priority alarm to every household member's mobile
app. `FORCED_OFF_GRID` is the admin-pinned variant (used during
inverter maintenance). The mode reverses automatically when grid
power returns; recovery is also broadcast.

## When you need this concept

- Touching grid-outage detection, the consumption-shedding policy,
  or the broadcast notification flow.
- Adding a new device whose behaviour must differ when off-grid.
- Working on `QSHomeMode` state transitions.
- Debugging "the system didn't shed load when the power went out"
  issues.

## Core idea

Off-grid mode lives in `ha_model/home.py` on `QSHome`. The detection
trigger is "grid power sensor drops to zero" (with a small debounce
to avoid false positives from sensor noise). Once tripped:

1. **Load shedding**: `FILLER` and `FILLER_AUTO` constraints are
   suspended. `MANDATORY` constraints remain — they're still
   honoured but capped at (solar + battery) available power.
2. **Battery policy switch**: battery's normal "save for evening
   arbitrage" goal is overridden by "keep the house running".
3. **Broadcast**: every household member's `mobile_app` notification
   service receives a high-priority alarm-channel message.

Recovery: when the grid sensor returns to a non-zero stable reading,
mode reverts to `NORMAL`, suspended `FILLER` constraints resume, and
a "grid restored" broadcast goes out.

`FORCED_OFF_GRID` is a manual variant: an admin toggle that pins the
mode for maintenance windows. It does not auto-recover.

## Key types / structures

- `QSHomeMode` enum: `NORMAL`, `OFF_GRID`, `FORCED_OFF_GRID`.
- Off-grid detection logic on `QSHome` (in the state polling cycle).
- Broadcast helper: routes through every configured `QSPerson`'s
  mobile-app entity.

## Lifecycle

```text
Grid sensor steady at 0W (debounced) → mode = OFF_GRID
  ↓
suspend FILLER constraints
  ↓
broadcast alarm to all household members
  ↓
... grid restored ...
  ↓
mode = NORMAL → resume FILLER → broadcast recovery
```

## Common mistakes

- Forgetting to test recovery. The "outage detected" path is easy;
  the "grid restored" path is where bugs hide.
- Treating `FORCED_OFF_GRID` like `OFF_GRID`. The admin pinned it
  for a reason — auto-recovery would defeat the purpose.
- Skipping the debounce. A flickering grid sensor causes alarm
  storms.
- Trying to keep `FILLER_AUTO` loads running on battery. The whole
  point of shedding is to extend battery life during the outage.

## See also

- [qs-home-orchestrator.md](qs-home-orchestrator.md) — `QSHome` and
  its cycles.
- [notification-routing.md](notification-routing.md) — the
  per-person broadcast path.
- [../use-cases/off-grid-kicks-in.md](../use-cases/off-grid-kicks-in.md)
  — end-to-end outage scenario.
