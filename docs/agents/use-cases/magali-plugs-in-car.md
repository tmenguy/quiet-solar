---
title: Magali plugs in her car
slug: magali-plugs-in-car
kind: use-case
last_verified: 2026-05-19
---

# Magali plugs in her car — the "magic moment"

## TL;DR

This is the canonical end-to-end use case. Magali pulls into the
driveway, plugs in her car, and walks away. Several layers cooperate
behind the scenes: presence detection, trip prediction, constraint
creation, solver allocation, charger budgeting, command execution,
notification. Nothing requires her attention. The car is ready when
she needs it. **This is the use case quiet-solar exists for.**

## When you need this use case

- Onboarding a new contributor — this is the system in one
  scenario.
- Reasoning about end-to-end paths when modifying any one layer.
- Designing a feature whose value would be visible in this path.
- Writing the README / marketing material.

## End-to-end sequence

```
1. Magali parks her car → HA detects presence (GPS + zone)
   ↓
2. QSPerson.update_states() observes presence transition
   ↓
3. Magali plugs in → charger HA entity reports "plugged in"
   ↓
4. QSCar observes the charger-car pairing → SOC reading available
   ↓
5. Person-trip-prediction runs:
   QSPerson.predict_next_trip() → distance from 31-day mileage ring
   QSCar.distance_to_kWh() → target SOC = current + prediction + margin
   ↓
6. push_live_constraint(target_SOC,
     load_info={originator: "prediction"})
   on the car-charger pair
   ↓
7. Next load-management cycle (~7s):
   PeriodSolver picks up the new constraint, allocates power slots
   over the night (prefers solar surplus tomorrow morning, cheap
   off-peak grid windows otherwise)
   ↓
8. Charger budgeting cycle (~45s):
   verifies amp budget across all chargers on the same circuit,
   sets the per-charger amps + phase, applies via OCPP/Wallbox/
   Generic protocol
   ↓
9. Notification: "Your car will be ready by 7am, charged to 78%
   (predicted trip: 42km to office)"
   ↓
10. Magali ignores the notification (95% case) — system did the
    right thing
```

## The 5% case — Magali overrides

Magali knows she has an unplanned trip tomorrow. She opens the
mobile app, taps "extend trip", picks "350km tomorrow":

```
1. Override UI captures input → user_override constraint pushed
   with load_info={originator: "user_override"}
   ↓
2. Existing prediction constraint replaced by the override
   ↓
3. Solver re-evaluates → may shift battery / grid allocation
   ↓
4. Confirmation notification: "Got it. Car will be ready by 7am,
   charged to 95%."
```

The interaction must take less than 5 seconds end-to-end. See
[../personas/magali.md](../personas/magali.md).

## What each layer contributes

| Layer | Contribution |
|---|---|
| Person & car ([person-trip-prediction.md](../concepts/person-trip-prediction.md)) | Presence detection, trip prediction, SOC tracking, person-car allocation. |
| Constraints ([constraints.md](../concepts/constraints.md)) | The demand language. `load_info` traces origin. |
| Solver ([solver.md](../concepts/solver.md)) | Strategic allocation across the night's tariff windows. |
| Charger budgeting ([charger-budgeting.md](../concepts/charger-budgeting.md)) | Tactical per-circuit amp distribution. Physical safety. |
| Notification ([notification-routing.md](../concepts/notification-routing.md)) | Per-person delivery; explains "why" via `load_info`. |
| User override ([user-override.md](../concepts/user-override.md)) | The 5% case that protects trust. |

## Common mistakes when modifying this path

- Optimising one layer without checking that the next layer can
  consume the change. Solver "improvement" that produces plans
  charger budgeting routinely overrides isn't an improvement.
- Surfacing implementation details in the notification. Magali
  cares about "ready by 7am", not "MANDATORY_END_TIME 9 with
  load_info originator prediction".
- Designing a new override flow that takes more than 5 seconds.

## See also

- [../personas/magali.md](../personas/magali.md) — the persona this
  serves.
- [solar-surplus-allocation.md](solar-surplus-allocation.md) — the
  morning-after surplus story that gets the car charged for free.
- [cheap-grid-charging.md](cheap-grid-charging.md) — the cloudy-
  night fallback path.
- [external-override.md](external-override.md) — what happens if
  Magali unplugs the cable mid-charge.
- [solver-replans-on-state-change.md](solver-replans-on-state-change.md)
  — the event-driven re-evaluation behind step 7.
