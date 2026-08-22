---
title: Battery (home_model)
slug: home-model-battery
kind: concept
covers:
  - custom_components/quiet_solar/home_model/battery.py
last_verified: 2026-08-20
---

# Battery (home_model) — pure-domain charge/discharge model

## TL;DR

`home_model/battery.py` is the pure-Python battery model used by the
solver. It computes safe charge / discharge power respecting SOC
bounds, inverter power limits, and DC-coupled awareness (whether the
battery shares an inverter with PV). It is **independent of HA** —
the HA-facing `QSBattery` lives in `ha_model/battery.py`
([ha-battery.md](ha-battery.md)). The split exists so the solver can
reason about batteries in unit tests without any HA dependency.

## When you need this concept

- Modifying the solver's battery optimisation step.
- Adding a new battery topology (e.g., AC-coupled vs DC-coupled).
- Changing how SOC bounds, charge/discharge power, or efficiency
  factor in.
- Writing tests that exercise battery logic without HA fixtures.

## Core idea

The model exposes battery state and safe-action queries:

- **SOC bounds**: floor and ceiling expressed as percentages; the
  model refuses to charge above the ceiling or discharge below the
  floor.
- **Power limits**: charge and discharge are clamped to the inverter's
  rated limits, separately.
- **DC-coupled awareness**: when the battery shares an inverter with
  the PV array, charging from PV avoids the AC↔DC round-trip — the
  model exposes this so the solver can prefer DC-coupled charging
  during PV surplus windows.

The solver calls into this model during step 3 of the allocation
algorithm (battery optimisation to minimise grid imports).

### Outage discharge floor (QS-349)

`min_discharging_power` is the opt-in **outage safety floor** — the
minimum discharge power the battery always keeps available even under a
discharge-limiting command. It is a `Battery.__init__` kwarg populated
from `CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE` (default `0` = today's
behaviour; a `None`/non-numeric/non-finite value from a corrupt entry is
coerced via the shared `coerce_finite_float` — used by both the domain
model and the HA bridge, review-fix #04 U9 — to a finite default for the
floor, both maxes, the capacity, and the SOC percents (U8), and a negative
max is clamped to 0 so it cannot drag the floor negative — review-fix #02
R6 / #03 T6 / #04 U8). Coerced values are also range-clamped (review-fix
#05 V3): `capacity >= 0`, SOC percents to `[0, 100]` with
`min_charge_SOC_percent <= max_charge_SOC_percent` enforced — a
finite-but-nonsense entry (negative capacity, percent 150, min > max) is
the same hand-edited threat as a non-numeric one. At init
it is clamped once to `[0, max_discharging_power]` and
integer-normalized (`float(round(...))`) so the write/read int-casts and
the raw expected value used by `probe_if_command_set` agree; a
non-integer floor would otherwise never confirm in
`probe_if_command_set` (eternal retry). The round is followed by a
re-clamp to `max_discharging_power` so a floor that rounds up past a
fractional configured max cannot break the `floor <= max` invariant
(review-fix #05 V4). The solver reads this attribute
to model the floor's discharge during `CMD_GREEN_CHARGE_ONLY` slots (see
[solver.md](solver.md)).

## Key types / structures

- `Battery` — dataclass-style domain object. Holds SOC, capacity,
  charge/discharge power limits, DC-coupled flag.
- Safe-charge / safe-discharge helpers — clamp to SOC bounds +
  inverter limits.

## Common mistakes

- Implementing battery logic in `ha_model/battery.py` that should
  live in this file. Anything that doesn't need HA APIs belongs in
  `home_model/`.
- Forgetting the DC-coupled distinction when reasoning about
  efficiency. AC-coupled batteries have a round-trip penalty that
  DC-coupled don't.
- Ignoring SOC bounds when computing "available capacity". The solver
  must use the *usable* range, not the nominal capacity.

## See also

- [ha-battery.md](ha-battery.md) — the HA-side `QSBattery`.
- [solver.md](solver.md) — calls into this model during battery
  optimisation.
- [../principles/two-layer-boundary.md](../principles/two-layer-boundary.md)
  — why this file is HA-free.
- [../use-cases/cheap-grid-charging.md](../use-cases/cheap-grid-charging.md)
  — battery + tariff interaction.
