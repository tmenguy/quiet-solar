---
title: Person & trip prediction
slug: person-trip-prediction
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/person.py
  - custom_components/quiet_solar/ha_model/car.py
last_verified: 2026-07-04
---

# Person, Car, and trip prediction

## TL;DR

`QSPerson` (`ha_model/person.py`) tracks presence and predicts trips
from GPS and historical mileage (a count-capped ring of the last 90
mileage records; boot-time recorder backfill is a separate 30-day
window). `QSCar`
(`ha_model/car.py`) tracks SOC, charger assignment, and a custom
power→amperage lookup table per car. The two together produce the
constraints behind the "Magali plugs in her car" magic moment: the
prediction-derived constraint targets a SOC that lets the car cover
tomorrow's predicted trips with margin.

## When you need this concept

- Modifying trip prediction, mileage estimation, or the retention /
  backfill windows.
- Adding a new car model with non-linear charging curves.
- Working on person-car allocation (which car belongs to which
  person).
- Touching SOC tracking or dampening on the car side.

## Core idea

**Person side**:

- GPS-based presence detection (home / away / commuting).
- Mileage history (count cap of 90 records via
  `MAX_PERSON_MILEAGE_HISTORICAL_DATA_DAYS`, sensor-attribute restored;
  boot-time recorder backfill uses the separate
  `PERSON_HISTORY_BACKFILL_DAYS = 30` so a larger retention never
  inflates the per-day recorder queries) → **outlier-resistant**
  per-weekday guess → next-day prediction. The retention is a count cap,
  not an age cutoff: with gap days the ring may span more than 90
  calendar days.
- **Per-weekday guess** (`_get_best_week_day_guess`): walk the history
  most-recent-first, same weekday only, **skip outliers**, and take the
  **max mileage** and **earliest leave time** of the last
  `PERSON_MILEAGE_NUM_GOOD_SAMPLES = 2` non-outlier records. Outlier
  records contribute neither mileage nor leave time. Whenever the bucket
  is non-empty at least one good record provably exists, so there is no
  all-outlier fallback branch.
- **Outlier detection** (`_is_mileage_outlier` / `_is_suspicious_mileage`):
  a record is *suspicious* when
  `mileage > PERSON_MILEAGE_OUTLIER_FACTOR (2.5) ×` its **leave-one-out**
  `statistics.median` same-weekday baseline (interpolated median; the
  record under test is excluded by its day key) **and**
  `mileage > PERSON_MILEAGE_OUTLIER_MIN_KM (100 km)` (absolute floor —
  rejecting normal km for a low-mileage person under-charges, the worse
  failure mode). Detection is disabled (record kept) when the
  leave-one-out bucket has fewer than
  `PERSON_MILEAGE_MIN_SAMPLES_FOR_OUTLIER_DETECTION = 4` samples —
  identical to the pre-QS-298 behavior on sparse / cold-start data (3
  samples → off, 4 → on).
- **Live-pattern recurrence rescue**: a suspicious record is kept only
  when it is one of the **2 most recent** records of its weekday bucket
  and that last-2 pair is a *live pattern* — BOTH suspicious, mutually
  similar (`abs(older − newest) ≤ PERSON_MILEAGE_RECURRENCE_TOLERANCE
  (0.2) × newest`), and within
  `PERSON_MILEAGE_RECURRENCE_WINDOW_DAYS = 21` days of each other
  (inclusive). This keeps a genuine weekly far commute while rejecting a
  one-off. A single normal day in the bucket breaks liveness and reverts
  the prediction immediately (minority regime).
- **Documented trade-offs**: the *first* occurrence of a new recurring
  long trip is rejected (bounded to one mispredicted week, until the
  second occurrence makes the pattern live); non-weekly / alternating /
  monthly cadences are rejected (calendar integration is the fix, a
  separate story); out-and-back weekend round trips cannot self-validate
  because corroboration is same-weekday only. In the **majority regime**
  (a habit that dominates the bucket) the leave-one-out median itself
  rises, the records stop being suspicious, and the prediction decays
  over a few weeks after the habit ends — the immediate-reversion
  guarantee applies to the minority regime only.
- Prediction confidence: if history is sparse or pattern match is
  weak, fall back to a conservative (over-charged) target.

**Car side**:

- SOC tracking via HA entity (manufacturer-specific). The trip-prediction
  read goes through the unified effective-SOC accessor
  `get_car_charge_percent`, which returns an **estimate** (manual override or
  charged-energy interpolation) when the SOC API is failed / inaccurate /
  absent — see [car-soc-estimation.md](car-soc-estimation.md).
- Charger assignment: which charger this car is plugged into. A manual
  charger assignment is trusted over a wrongly-"away" tracker (the car is
  kept managed via inferred home/plug flags). The override is reconciled
  tri-state against the raw reads — affirmative positives drop it, an explicit
  contradiction holds it, and a transient `unavailable`/`None` read holds the
  current state (no flicker-drop, including across a SOC-stale recovery cycle) —
  but it is still bounded by the raw-tracker
  departure auto-reset ceiling (`CAR_NOT_HOME_AUTO_RESET_S`, 15 min), and a
  user Force-Not-Stale selection drops it immediately (live presence truth
  wins) — see the manual-trust ceiling note in
  [car-soc-estimation.md](car-soc-estimation.md).
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
- 90-record mileage ring (count cap, sensor-attribute restored;
  serialized payload stays under the recorder's 16 KB attribute limit —
  guarded by a permanent test).
- Custom power-to-amperage lookup table on `QSCar`.
- `QSCar.get_car_person_readable_forecast_mileage(for_small_standalone=True)`
  — raw forecast string (`"<Person>: <km>km <date>"`); backs the
  `qs_car_person_forecast` sensor. The default `for_small_standalone=True`
  keeps the compact widget date form (`HH:MM` / `%m-%d %H:%M`) for that
  sensor and all other direct callers; QS-278 added the parameter so the
  origin line can request the normal form. Unchanged by QS-274.
- `QSCar.get_car_charge_origin_readable_string()` (QS-274) — the
  origin-responsive context line backing the new `qs_car_charge_origin`
  sensor. Pure / sync-safe: reads
  `self.charger.get_charge_type(return_charge_errors=False)` and the live
  `current_forecasted_person`. QS-278: every date-bearing branch renders
  with the **normal** `get_readable_date_string` formatting —
  `today HH:MM` / `tomorrow HH:MM` / `%Y-%m-%d %H:%M` — not the compact
  `for_small_standalone` form. Returns `"Calendar · <date>"`,
  `"Manually set to <date>"`, or the two as-fast strings for those origins;
  for the person-automated, no-charger and any-other-type cases it
  delegates to `get_car_person_readable_forecast_mileage(for_small_standalone=False)`
  — i.e. the person line `"<Person>: <forecast>"` with the leave time in
  the same normal form (including `"<Person>: No forecast"` when there is no
  prediction) or `"No forecasted person"` when no person is attached. Every
  branch stays single-line; far-out (>~24h) targets/leave times render the
  full `%Y-%m-%d %H:%M` date on one line. This is the single source of
  truth for the car card's origin context row.

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
