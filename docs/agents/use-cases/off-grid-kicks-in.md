---
title: Off-grid mode kicks in
slug: off-grid-kicks-in
kind: use-case
last_verified: 2026-05-19
---

# Off-grid mode kicks in

## TL;DR

The grid drops at 3am during a storm. Quiet-solar detects the outage
within seconds, sheds non-mandatory loads, caps consumption at
(battery + solar) capacity, and **broadcasts a high-priority alarm
to every household member's mobile app**. The home keeps running on
solar + battery for as long as the math allows. When the grid
returns, the system reverts automatically and broadcasts the
recovery. This is the resilience story that makes quiet-solar
trustworthy in homes that get grid outages.

## When you need this use case

- Modifying off-grid detection or load-shedding policy.
- Working on the broadcast notification path.
- Adding a new device whose off-grid behaviour must differ.
- Reasoning about battery-as-buffer behaviour during outages.

## End-to-end sequence

```text
1. Grid sensor: 4200W → 0W (outage)
   ↓
2. State-polling cycle (~4s) detects the drop; debounces (~10s) to
   filter sensor noise
   ↓
3. QSHomeMode transitions: NORMAL → OFF_GRID
   ↓
4. Load shedding:
   FILLER and FILLER_AUTO constraints suspended
   MANDATORY constraints kept but capped at (battery_kW + solar_kW)
   ↓
5. Battery policy override:
   normal "save for evening peak" goal disabled
   new goal: "keep the house running as long as possible"
   ↓
6. Broadcast notification (alarm channel, high priority) to every
   configured QSPerson's mobile_app entity:
   "Power outage detected. Solar + battery active. Estimated
   runtime: 8 hours at current load."
   ↓
7. ... grid returns: sensor reads stable non-zero value for the
   debounce window ...
   ↓
8. QSHomeMode: OFF_GRID → NORMAL
   ↓
9. FILLER constraints resume; battery policy reverts
   ↓
10. Broadcast recovery: "Grid restored. Normal operation resumed."
```

## The FORCED_OFF_GRID variant

`FORCED_OFF_GRID` is an admin toggle for maintenance windows. It
behaves like `OFF_GRID` but does **not** auto-revert. Step 8 is
skipped — the admin explicitly clears the flag.

## What each layer contributes

| Layer | Contribution |
|---|---|
| QSHome ([qs-home-orchestrator.md](../concepts/qs-home-orchestrator.md)) | Detects the outage; manages mode state. |
| Off-grid mode ([off-grid-mode.md](../concepts/off-grid-mode.md)) | The shedding policy and the broadcast helper. |
| Battery ([home-model-battery.md](../concepts/home-model-battery.md)) | Becomes the home's primary supply during the outage. |
| Notification ([notification-routing.md](../concepts/notification-routing.md)) | Broadcast to every household member. |
| Solver ([solver.md](../concepts/solver.md)) | Re-plans within the new cap; honours MANDATORY only. |

## Common mistakes when modifying this path

- Skipping the debounce. A flickering grid sensor causes alarm
  storms — Magali's phone buzzes every 30 seconds.
- Forgetting the recovery path. The "outage detected" code is easy;
  the "grid restored" code is where regressions hide.
- Sending the broadcast through the per-person event surface
  instead of the broadcast helper. Per-person honours quiet-hours;
  broadcast ignores them. Off-grid is broadcast.
- Letting `FILLER_AUTO` run during off-grid because "the battery
  has plenty of juice". The whole point is to extend runtime.

## See also

- [../concepts/off-grid-mode.md](../concepts/off-grid-mode.md) —
  the mechanics.
- [../concepts/qs-home-orchestrator.md](../concepts/qs-home-orchestrator.md)
  — the orchestrator that hosts the state machine.
- [../concepts/notification-routing.md](../concepts/notification-routing.md)
  — the broadcast surface.
- [../personas/magali.md](../personas/magali.md) — the persona
  whose trust the broadcast preserves.
