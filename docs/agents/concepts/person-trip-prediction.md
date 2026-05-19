---
title: Person & trip prediction
slug: person-trip-prediction
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/person.py
  - custom_components/quiet_solar/ha_model/car.py
last_verified: 2026-05-19
---

# Person, Car, and trip prediction

## TL;DR

`QSPerson` (`ha_model/person.py`) tracks presence and predicts trips
from GPS and historical mileage (31-day rolling window). `QSCar`
(`ha_model/car.py`) tracks SOC, charger assignment, and a custom
power→amperage lookup table per car. The two together produce the
constraints behind the "Magali plugs in her car" magic moment: the
prediction-derived constraint targets a SOC that lets the car cover
tomorrow's predicted trips with margin.

## When you need this concept

- Modifying trip prediction, mileage estimation, or the 31-day
  rolling window.
- Adding a new car model with non-linear charging curves.
- Working on person-car allocation (which car belongs to which
  person).
- Touching SOC tracking or dampening on the car side.

## Core idea

**Person side**:

- GPS-based presence detection (home / away / commuting).
- Mileage history (31 days, sensor-attribute restored) →
  per-day-of-week average → next-day prediction.
- Prediction confidence: if history is sparse or pattern match is
  weak, fall back to a conservative (over-charged) target.

**Car side**:

- SOC tracking via HA entity (manufacturer-specific).
- Charger assignment: which charger this car is plugged into.
- Person allocation: which person owns this car (drives prediction
  targeting).
- Custom power → amperage table per car: different cars accept
  different voltage / phase combinations differently.

The two compose: `QSPerson.predict_next_trip()` returns a distance →
`QSCar.distance_to_kWh()` converts → constraint `target = current +
prediction_kWh + margin` is pushed for the car's charger.

## Key types / structures

- `QSPerson(HADeviceMixin, AbstractDevice)` — person tracking class.
- `QSCar(HADeviceMixin, AbstractLoad)` — car class. Inherits constraint
  surface from `AbstractLoad`.
- 31-day mileage ring (sensor-attribute restored).
- Custom power-to-amperage lookup table on `QSCar`.

## Lifecycle

```text
GPS reading → presence state → trip detection
  ↓
Trip ended → distance recorded → mileage ring updated
  ↓
Next cycle: predict_next_trip() → distance_to_kWh()
  ↓
push_live_constraint(target_SOC, load_info={originator: "prediction"})
  ↓
Solver allocates → charger executes → SOC reaches target → constraint
  completes
```

## Common mistakes

- Treating the mileage ring as authoritative for the first few days
  after install. Cold-start predictions are deliberately
  conservative.
- Hard-coding a power → amperage curve. Each car model has its own;
  the lookup table on `QSCar` is the single source of truth.
- Forgetting `load_info={originator: "prediction"}` on
  prediction-derived constraints. Without it, the user can't tell
  why the constraint exists.
- Confusing person-car allocation with car-charger assignment. They
  are independent: a person can own multiple cars; a car can be
  plugged into any charger.

## See also

- [constraints.md](constraints.md) — the prediction creates a
  constraint.
- [user-override.md](user-override.md) — what happens when the
  prediction is wrong.
- [notification-routing.md](notification-routing.md) — per-person
  notifications.
- [../use-cases/magali-plugs-in-car.md](../use-cases/magali-plugs-in-car.md)
  — end-to-end magic moment.
- [../personas/magali.md](../personas/magali.md) — the user behind
  the prediction.
