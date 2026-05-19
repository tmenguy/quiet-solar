---
title: HADeviceMixin
slug: ha-device-mixin
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/device.py
last_verified: 2026-05-19
---

# HADeviceMixin — the bridge layer

## TL;DR

`HADeviceMixin` is the bridge between HA entities and quiet-solar's
domain model. Every HA device class inherits from `HADeviceMixin`
plus a domain class (e.g., `QSBattery(HADeviceMixin, Battery)`). It
owns: HA entity probing (`attach_ha_state_to_probe()` with numerical
conversion and custom getters), time-series history (auto-trimmed to
3 days), power-to-energy conversion via Riemann sum, and a reverse
index of HA entities by description key (`ha_entities` dict) that
the dashboard templates query.

## When you need this concept

- Adding a new device type — the HA-side class extends this mixin.
- Modifying how HA state flows into the domain model.
- Working on the dashboard (which queries `ha_entities`).
- Touching history trimming, Riemann-sum energy computation, or
  state probing.

## Core idea

The bridge pattern: every HA device class is two halves stitched
together. `HADeviceMixin` is one half — it knows about HA service
calls, entity registries, and state changes. The domain class is the
other half — it knows about the solver, constraints, and commands.
The mixin's job is to translate between them:

- **State in**: `add_to_history(time, value)` appends to a numpy
  ring buffer; trimmed to 3 days. Used by the solver for "what's
  happened recently".
- **State probing**: `attach_ha_state_to_probe(entity_id, key,
  getter=..., transform=..., numerical_conversion=True)` —
  declarative wiring of one HA entity → one tracked state.
- **Commands out**: the domain class (`AbstractDevice`) calls
  `execute_command()`; the device-specific subclass translates to
  `hass.services.async_call(...)`.
- **Dashboard index**: every probed entity registers itself in
  `ha_entities` keyed by description; the Jinja2 dashboard reads
  `device.ha_entities.get(key)` to find the entity ID at render
  time.

## Key types / structures

- `HADeviceMixin` — the mixin.
- `attach_ha_state_to_probe(...)` — declarative entity wiring.
- `add_to_history(time, value)` — numpy-backed time series.
- `ha_entities` dict — reverse index from description key → entity.
- Riemann-sum energy helpers (power → energy over a time window).

## Common mistakes

- Storing HA state directly on the domain object instead of going
  through `add_to_history()`. The history is what the solver uses.
- Forgetting to register a description key in `ha_entities` — the
  entity exists but the dashboard can't find it.
- Treating the 3-day history as authoritative for long-term analysis.
  It's deliberately short-lived; persistent history lives in the
  numpy ring buffer (separate concept).
- Doing custom transforms in the dashboard template. All conversion
  happens in `attach_ha_state_to_probe` via `transform=...`.

## See also

- [load-base.md](load-base.md) — the domain-side counterpart.
- [config-and-setup-flow.md](config-and-setup-flow.md) — how devices
  are constructed and wired.
- [../principles/two-layer-boundary.md](../principles/two-layer-boundary.md)
  — why this bridge exists.
