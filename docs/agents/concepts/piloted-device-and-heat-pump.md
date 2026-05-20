---
title: PilotedDevice and heat pump
slug: piloted-device-and-heat-pump
kind: concept
covers:
  - custom_components/quiet_solar/home_model/load.py
  - custom_components/quiet_solar/ha_model/heat_pump.py
  - custom_components/quiet_solar/ha_model/climate_controller.py
last_verified: 2026-05-21
---

# PilotedDevice and heat pump

## TL;DR

`PilotedDevice` (in `home_model/load.py`) extends `AbstractDevice`
for devices that **pilot other devices** — they track a list of
clients and per-slot demand counts, and their command affects what
the clients see. `QSHeatPump` (`ha_model/heat_pump.py`) is the
canonical user: it pilots an auxiliary electric heater that kicks in
when the heat pump alone can't meet the climate setpoint.
`climate_controller.py` is the thin HA-facing controller that
translates climate setpoints into the heat pump's command vocabulary.

## When you need this concept

- Adding a new "primary + secondary" device (e.g., HVAC + emergency
  resistive heater).
- Modifying the heat pump's behaviour or the auxiliary trigger
  logic.
- Working on multi-client demand aggregation.
- Touching the climate controller's HA integration.

## Core idea

`PilotedDevice` adds two extensions on top of `AbstractDevice`:

- **Client list**: the devices being piloted register themselves as
  clients. The pilot's command propagates to all clients.
- **Per-slot demand counts**: when multiple clients each request
  some power, the pilot aggregates the demand and decides whether
  to engage the secondary device (e.g., turn on the aux heater).

`QSHeatPump` wraps this pattern: the primary heat-pump compressor
handles most demand; when the compressor is saturated and the
climate setpoint is still missed, the pilot engages the aux heater.
This is the structural pattern for "main device with optional
booster".

## Key types / structures

- `PilotedDevice(AbstractDevice)` — the base for pilot-capable
  devices.
- `QSHeatPump(HADeviceMixin, PilotedDevice)` — the canonical user.
- `climate_controller.py` — HA climate-entity translation.
- Client registration and demand aggregation helpers.

## Common mistakes

- Implementing a "primary + secondary" device without using
  `PilotedDevice`. You'll re-invent the client list and demand
  aggregation badly.
- Triggering the aux heater on every cycle the setpoint is missed.
  The hysteresis pattern (`hysteresis-and-switching-cost`) still
  applies — aux heaters are expensive switches.
- Conflating the pilot's command with the clients' state. The
  pilot says "engage"; whether each client engaged is observed
  through their own probes.

## See also

- [load-base.md](load-base.md) — the `AbstractDevice` base.
- [bistate-duration-devices.md](bistate-duration-devices.md) — the
  cousin pattern for simple on/off duration loads.
- [../principles/hysteresis-and-switching-cost.md](../principles/hysteresis-and-switching-cost.md)
  — the daily-budget + hysteresis pattern.
