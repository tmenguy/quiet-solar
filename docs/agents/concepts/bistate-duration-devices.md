---
title: Bistate-duration devices
slug: bistate-duration-devices
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/bistate_duration.py
  - custom_components/quiet_solar/ha_model/on_off_duration.py
  - custom_components/quiet_solar/ha_model/pool.py
last_verified: 2026-05-19
---

# Bistate-duration devices (pool, on/off duration)

## TL;DR

Bistate-duration devices are loads with two states (on / off) that
must run for a **specified duration** rather than be modulated. Pool
pumps, fixed-power boilers, and miscellaneous on/off duration loads
all use this pattern.
`ha_model/bistate_duration.py` provides the shared base;
`ha_model/on_off_duration.py` is the simplest concrete subclass;
`ha_model/pool.py` extends `on_off_duration` with
temperature-dependent filter-duration logic. All three inherit the
switching-cost protection pattern.

## When you need this concept

- Adding a new simple on/off device (irrigation, ventilation, fixed-
  power resistive heater).
- Touching pool-pump duration logic.
- Working on the bistate constraint type
  (`TimeBasedSimplePowerLoadConstraint`).
- Tweaking switching-cost / hysteresis behaviour for on/off
  devices.

## Core idea

A bistate-duration device has:

- A fixed power draw when on.
- A **target duration** (e.g., "run for 4 hours").
- The same `num_max_on_off` + 10-minute hysteresis as every other
  switching device.

The solver consumes `TimeBasedSimplePowerLoadConstraint` for these
— the constraint specifies "X hours at Y watts before deadline";
the solver picks the cheapest contiguous (or split) windows.

`QSPool` adds **temperature-dependent duration** on top of
`QSOnOffDuration`: warmer water → longer filter duration. The
extension overrides the duration calculation but inherits the rest
of the on/off behaviour unchanged.

## Key types / structures

- `bistate_duration.py` — shared base (the lifecycle, the constraint
  interface).
- `QSOnOffDuration(HADeviceMixin, AbstractLoad)` — simple switch
  loads.
- `QSPool(QSOnOffDuration)` — temperature-aware filter duration.
- `TimeBasedSimplePowerLoadConstraint` (in
  `home_model/constraints.py`) — the constraint subclass these
  devices use.

## Common mistakes

- Modulating power on a bistate device. The whole point is that
  it's on or off; if you can modulate, use
  [piloted-device-and-heat-pump.md](piloted-device-and-heat-pump.md)
  or a charger pattern instead.
- Forgetting the switching budget. On/off devices are the heaviest
  consumers of `num_max_on_off` — debugging "the pool never ran"
  often comes back to a tight daily limit.
- Hard-coding pool filter duration. The temperature-dependent
  helper on `QSPool` is the single source of truth.

## See also

- [load-base.md](load-base.md) — `AbstractLoad` base contract.
- [constraints.md](constraints.md) — the
  `TimeBasedSimplePowerLoadConstraint` subclass.
- [../principles/hysteresis-and-switching-cost.md](../principles/hysteresis-and-switching-cost.md)
  — the daily-budget pattern these devices inherit.
